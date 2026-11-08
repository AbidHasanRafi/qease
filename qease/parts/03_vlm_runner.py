# %% ----------------------------------------------- 3. VLM RUNNER (hook-based)
class _Stop(Exception):
    pass

# ---- thread-safe spy on scaled_dot_product_attention: lets FastV read the EXACT post-RoPE q/k that the real attention used
#      (no re-implementation of RoPE/GQA, so it cannot drift from the model). Installed once; pass-through unless a thread asks.
_TL = threading.local()
if not getattr(torch.nn.functional.scaled_dot_product_attention, "_qease_spy", False):
    _orig_sdpa = torch.nn.functional.scaled_dot_product_attention
    def _spy_sdpa(q, k, v, *a, **kw):
        cap = getattr(_TL, "cap", None)
        if cap is not None:
            mode = cap.get("mode", "last_row")
            if mode == "last_row" and "q" not in cap:                      # FastV: first call, last query row only
                cap["q"], cap["k"] = q[:, :, -1:, :].detach(), k.detach()
            elif mode == "first_full" and "q" not in cap:                  # SparseVLM: first call, all rows
                cap["q"], cap["k"] = q.detach(), k.detach()
            elif mode == "overwrite":                                      # VisionZip: keep the LAST call (= last vision block)
                cap["q"], cap["k"] = q.detach(), k.detach()
        return _orig_sdpa(q, k, v, *a, **kw)
    _spy_sdpa._qease_spy = True
    torch.nn.functional.scaled_dot_product_attention = _spy_sdpa

# ---- SparseVLM v1.0 constants / helpers ported line-by-line from github.com/Gumpest/SparseVLMs
#      (llava/model/language_model/score.py, utils.py::cluster_and_merge/index_points, modelling_sparse_llama.py)
SPARSE_SCHEDULE = {192: [300, 200, 110], 128: [303, 110, 36], 96: [238, 48, 26], 64: [66, 30, 17]}   # tokens kept after each stage, for 576 visual tokens
SPARSE_LAYERS_OFFICIAL, SPARSE_NLAYERS_OFFICIAL, SPARSE_NTOK_OFFICIAL = [2, 6, 15], 32, 576

def sv_index_points(points, idx):
    B = points.shape[0]
    view_shape = list(idx.shape); view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape); repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long, device=points.device).view(view_shape).repeat(repeat_shape)
    return points[batch_indices, idx, :]

def sv_cluster_and_merge(x, cluster_num):
    """density-peak clustering + mean merge (official `cluster_and_merge`, einops replaced by indexing; also returns the assignment)."""
    B, N, C = x.shape
    distance = (x.unsqueeze(2) - x.unsqueeze(1)).norm(dim=-1, p=2)
    dist_matrix = distance / (C ** 0.5)
    dist_nearest, _ = torch.topk(dist_matrix, k=cluster_num, dim=-1, largest=False)
    density = (-(dist_nearest ** 2).mean(dim=-1)).exp()
    density = density + torch.rand(density.shape, device=density.device, dtype=density.dtype) * 1e-6
    mask = (density[:, None, :] > density[:, :, None]).type(x.dtype)
    dist_max = dist_matrix.flatten(1).max(dim=-1)[0][:, None, None]
    dist, _ = (dist_matrix * mask + dist_max * (1 - mask)).min(dim=-1)
    _, index_down = torch.topk(dist * density, k=cluster_num, dim=-1)
    dist_matrix = sv_index_points(dist_matrix, index_down)
    idx_cluster = dist_matrix.argmin(dim=1)
    idx_batch = torch.arange(B, device=x.device)[:, None].expand(B, cluster_num)
    idx_tmp = torch.arange(cluster_num, device=x.device)[None, :].expand(B, cluster_num)
    idx_cluster[idx_batch.reshape(-1), index_down.reshape(-1)] = idx_tmp.reshape(-1)       # centres merge to themselves
    token_weight = x.new_ones(B, N, 1)
    idx = idx_cluster + torch.arange(B, device=x.device)[:, None] * cluster_num
    all_weight = token_weight.new_zeros(B * cluster_num, 1)
    all_weight.index_add_(0, idx.reshape(B * N), token_weight.reshape(B * N, 1))
    all_weight = all_weight + 1e-6
    norm_weight = token_weight / all_weight[idx]
    x_merged = x.new_zeros(B * cluster_num, C)
    x_merged.index_add_(0, idx.reshape(B * N), (x * norm_weight).reshape(B * N, C).type(x.dtype))
    return x_merged.reshape(B, cluster_num, C), idx_cluster

def smart_resize(img, n_tokens, factor=28):
    """Resize to a multiple of 28 with <= n_tokens merged visual tokens (Qwen2-VL geometry)."""
    max_pixels = n_tokens * factor * factor
    w, h = img.size
    hb = max(factor, round(h / factor) * factor)
    wb = max(factor, round(w / factor) * factor)
    if hb * wb > max_pixels:
        beta = math.sqrt(h * w / max_pixels)
        hb = max(factor, math.floor(h / beta / factor) * factor)
        wb = max(factor, math.floor(w / beta / factor) * factor)
    return img.resize((wb, hb), Image.BICUBIC)

def find_decoder(model):
    for path in ["model.language_model", "language_model", "model.text_model", "text_model", "model.model", "model"]:
        obj, ok = model, True
        for a in path.split("."):
            if hasattr(obj, a): obj = getattr(obj, a)
            else: ok = False; break
        if ok and hasattr(obj, "layers") and hasattr(obj, "embed_tokens"):
            return obj
    raise RuntimeError("could not locate the text decoder of this VLM")

def js_div(logp_ref, logp):
    """Jensen-Shannon divergence (nats) between rows of log-probs. ref: (V,) or (B,V); logp: (B,V)."""
    p = logp_ref.exp(); q = logp.exp(); m = 0.5 * (p + q)
    lm = torch.log(m.clamp_min(1e-30))
    return 0.5 * (p * (logp_ref - lm)).sum(-1) + 0.5 * (q * (logp - lm)).sum(-1)

class VLMRunner:
    """One model replica on one GPU. Pruning = dropping visual tokens from the decoder's attention
    (positions preserved), so it is exactly equivalent to deleting them; `prefill_pruned` does it physically."""
    def __init__(self, device):
        from transformers import AutoProcessor, AutoConfig
        self.device = device
        cfgm = AutoConfig.from_pretrained(CFG["MODEL_ID"])
        cls = getattr(transformers, cfgm.architectures[0])
        qc = None
        if CFG.get("QUANT"):
            from transformers import BitsAndBytesConfig
            qc = (BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=DTYPE, bnb_4bit_quant_type="nf4")
                  if CFG["QUANT"] == "4bit" else BitsAndBytesConfig(load_in_8bit=True))
        kw = dict(attn_implementation="sdpa")
        if qc is not None:
            kw.update(quantization_config=qc, device_map={"": device})
        try:                                                       # `torch_dtype` (transformers 4.x) / `dtype` (5.x)
            self.model = cls.from_pretrained(CFG["MODEL_ID"], torch_dtype=DTYPE, **kw)
        except TypeError:
            self.model = cls.from_pretrained(CFG["MODEL_ID"], dtype=DTYPE, **kw)
        self.model = (self.model if qc is not None else self.model.to(device=device, dtype=DTYPE)).eval()   # enforce dtype whatever the version did
        for p in self.model.parameters(): p.requires_grad_(False)
        self.proc = AutoProcessor.from_pretrained(CFG["MODEL_ID"])
        self.tok = self.proc.tokenizer
        self.decoder = find_decoder(self.model)
        self.lm_head = self.model.lm_head
        self.embed = self.decoder.embed_tokens
        c = self.model.config
        self.img_tok = getattr(c, "image_token_id", None) or getattr(c, "image_token_index", None) \
            or self.tok.convert_tokens_to_ids("<|image_pad|>")
        vc = getattr(c, "vision_config", None)
        self.merge = getattr(vc, "spatial_merge_size", 2)
        self._cap, self._drop, self._capture_on = None, None, False
        self.vz_on = False                      # VisionZip needs vision-attention capture during prepare(); switched on after its self-test
        self.decoder.register_forward_pre_hook(self._hook, with_kwargs=True)
        n_llm = sum(p.numel() for n, p in self.decoder.named_parameters() if "embed" not in n)
        self.llm_params = n_llm
        eos = getattr(self.model.generation_config, "eos_token_id", None)
        eos = set(eos) if isinstance(eos, (list, tuple)) else ({eos} if eos is not None else set())
        for t_ in ("<|im_end|>", "<|endoftext|>"):
            tid = self.tok.convert_tokens_to_ids(t_)
            if isinstance(tid, int) and tid >= 0: eos.add(tid)
        if self.tok.eos_token_id is not None: eos.add(self.tok.eos_token_id)
        self.eos = eos

    # -- single hook used for capturing (prefill context) and for masked generation
    def _hook(self, mod, args, kwargs):
        if self._capture_on:
            self._cap = dict(emb=kwargs["inputs_embeds"].detach(), pos=kwargs.get("position_ids"),
                             am=kwargs.get("attention_mask"))
            raise _Stop()
        if self._drop is not None:
            am = kwargs.get("attention_mask")
            if am is not None and am.dim() == 2:
                am = am.clone(); am[:, self._drop] = 0
                kwargs["attention_mask"] = am
        return args, kwargs

    def build_inputs(self, image, question):
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}]
        text = self.proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        inp = self.proc(text=[text], images=[image], return_tensors="pt")
        return {k: v.to(self.device) for k, v in inp.items()}

    @torch.inference_mode()
    def prepare(self, rec_or_img, question=None, ds=None):
        """Run vision tower + projector once and capture the decoder inputs. Returns a context dict."""
        if isinstance(rec_or_img, dict):
            img = Image.open(POOL_DIR / rec_or_img["img"]).convert("RGB")
            question, ds = rec_or_img["question"], rec_or_img["ds"]
        else:
            img = rec_or_img
        nt = CFG["TOKENS_PER_DS"].get(ds, CFG["TOKENS_PER_DS"]["default"])
        img = smart_resize(img, nt)
        q_text = question if (isinstance(rec_or_img, dict) and rec_or_img.get("extra", {}).get("raw_prompt")) \
            else question + CFG["ANSWER_SUFFIX"]
        inputs = self.build_inputs(img, q_text)
        self._cap, self._capture_on = None, True
        if self.vz_on: _TL.cap = {"mode": "overwrite"}        # VisionZip needs q/k of the LAST vision block
        try:
            self.model(**inputs, use_cache=False)
        except _Stop:
            pass
        finally:
            self._capture_on = False
            vcap, _TL.cap = getattr(_TL, "cap", None), None
        cap = self._cap
        assert cap is not None, "decoder hook did not fire - unsupported transformers/model version"
        emb, pos, am = cap["emb"], cap["pos"], cap["am"]
        L = emb.shape[1]
        if am is None: am = torch.ones(1, L, dtype=torch.long, device=self.device)
        if pos is None: pos = torch.arange(L, device=self.device)[None]
        ids = inputs["input_ids"][0]
        img_idx = (ids == self.img_tok).nonzero().squeeze(-1)
        grid = inputs.get("image_grid_thw")
        if grid is not None:
            _, gh_, gw_ = [int(x) for x in grid[0]]
            H, W = gh_ // self.merge, gw_ // self.merge
        else:
            H, W = 1, len(img_idx)
        if H * W != len(img_idx):
            H, W = 1, len(img_idx)
        ctx = dict(emb=emb, pos=pos, am=am, img_idx=img_idx, H=H, W=W, L=L, inputs=inputs, ds=ds,
                   n_vis=len(img_idx))
        qids = self.tok(question, add_special_tokens=False, return_tensors="pt").input_ids[0][:CFG["LQ"]].to(self.device)
        ctx["qemb"] = self.embed(qids).detach() if len(qids) else torch.zeros(1, emb.shape[-1], device=self.device, dtype=emb.dtype)
        # grouping
        win = CFG["GROUP_WIN"]
        for w_ in sorted({win, 2, 3, 4, 6}):
            if w_ < win: continue
            if math.ceil(H / w_) * math.ceil(W / w_) <= CFG["GMAX"]:
                win = w_; break
        ctx["win"] = win
        gh, gw = math.ceil(H / win), math.ceil(W / win)
        r = torch.arange(len(img_idx), device=self.device) // W
        c = torch.arange(len(img_idx), device=self.device) % W
        gid = (r // win) * gw + (c // win)
        ctx.update(gh=gh, gw=gw, G=gh * gw, gid=gid)
        x = emb[0, img_idx].float()
        gf = torch.zeros(ctx["G"], x.shape[1], device=self.device).index_add_(0, gid, x)
        cnt = torch.zeros(ctx["G"], device=self.device).index_add_(0, gid, torch.ones(len(gid), device=self.device))
        ctx["gfeat"] = gf / cnt[:, None]
        ctx["gcount"] = cnt
        ctx["vz"], ctx["vz_err"] = None, None
        if self.vz_on:
            try:
                ctx["vz"] = self._vz_stats(vcap, len(img_idx))
            except Exception as e:
                ctx["vz_err"] = repr(e)[:200]
        return ctx

    # ------------------------------------------------------------------ VisionZip (port of the authors' Qwen2.5-VL implementation)
    def _vz_stats(self, cap, n_vis):
        """Official logic (Qwen2_5_VL/qwen2_5vl_visionzip.py, last vision block): per-patch score = attention RECEIVED (head-mean, summed
        over queries), averaged over each merge unit; contextual metric = last-block KEYS averaged over the merge unit and heads."""
        if not cap or "q" not in cap: raise RuntimeError("no vision attention captured (is the vision tower using sdpa?)")
        q, k = cap["q"], cap["k"]
        if q.dim() == 4: q, k = q[0], k[0]
        H, T, d = q.shape
        m2 = self.merge ** 2
        if T != n_vis * m2: raise RuntimeError(f"vision patches {T} != visual tokens {n_vis} x {m2}")
        qf, kf = q.float(), k.float()
        att = torch.softmax((qf @ kf.transpose(-1, -2)) * (d ** -0.5), dim=-1)       # (H, T, T)
        colsum = att.mean(0).sum(0)
        return dict(score=colsum.view(n_vis, m2).mean(-1), metric=kf.view(H, n_vis, m2, d).mean(2).mean(0),
                    att_total=float(colsum.sum()), T=T)

    @torch.inference_mode()
    def vz_embed(self, ctx, total):
        """`total` retained visual tokens = dominant + contextual (paper ratio 54:10 = 5.4:1). Returns (embeddings with merged
        contextual tokens written in, bool keep-mask over the full sequence). Positions of kept tokens are preserved, as in the official port."""
        vz, img = ctx["vz"], ctx["img_idx"]
        score, metric = vz["score"], vz["metric"]
        N = score.numel()
        total = int(min(N, max(2, total)))
        ctxn = min(max(1, int(total // 6.4)), total - 1)
        dom = total - ctxn
        top = score.topk(dom).indices
        is_top = torch.zeros(N, dtype=torch.bool, device=score.device); is_top[top] = True
        rest = (~is_top).nonzero().squeeze(-1)
        R = rest.numel(); ctxn = min(ctxn, R)
        mn = metric[rest]; mn = mn / mn.norm(dim=-1, keepdim=True)
        step = max(1, R // ctxn)
        target = torch.arange(0, R, step, device=score.device)[:ctxn]
        is_t = torch.zeros(R, dtype=torch.bool, device=score.device); is_t[target] = True
        h_rest = ctx["emb"][0, img].float()[rest]
        emb_new = ctx["emb"].clone()
        if int((~is_t).sum()) > 0:
            assign = (mn[~is_t] @ mn[target].T).argmax(1)
            cnt = torch.bincount(assign, minlength=ctxn).clamp(min=1).float()
            agg = torch.zeros(ctxn, h_rest.shape[-1], device=h_rest.device).index_add_(0, assign, h_rest[~is_t]) / cnt[:, None]
            ctx_tok = h_rest[target] + agg                        # official: target_hidden + aggregated_hidden
        else:
            ctx_tok = h_rest[target]
        emb_new[0, img[rest[target]]] = ctx_tok.to(emb_new.dtype)
        keep = torch.ones(ctx["L"], dtype=torch.bool, device=ctx["emb"].device); keep[img] = False
        keep[img[torch.cat([top, rest[target]])]] = True
        return emb_new, keep

    @torch.inference_mode()
    def logp_prefill(self, ctx, emb, keep_tok):
        """physically pruned prefill with arbitrary (possibly modified) embeddings, positions preserved."""
        keep = keep_tok.nonzero().squeeze(-1)
        pos = ctx["pos"][..., keep] if ctx["pos"].dim() == 3 else ctx["pos"][:, keep]
        out = self.decoder(inputs_embeds=emb[:, keep], attention_mask=torch.ones(1, len(keep), dtype=torch.long, device=self.device),
                           position_ids=pos, use_cache=False)
        return torch.log_softmax(self.lm_head(out.last_hidden_state[:, -1]).float(), -1)[0]

    @torch.inference_mode()
    def generate_custom(self, ctx, emb, keep_tok, max_new=10):
        """greedy decoding on a pruned/modified prompt (no KV cache; short answers)."""
        keep = keep_tok.nonzero().squeeze(-1)
        e = emb[:, keep]; pos = ctx["pos"][..., keep] if ctx["pos"].dim() == 3 else ctx["pos"][:, keep]
        ids = []
        for _ in range(max_new):
            out = self.decoder(inputs_embeds=e, attention_mask=torch.ones(1, e.shape[1], dtype=torch.long, device=self.device),
                               position_ids=pos, use_cache=False)
            nxt = self.lm_head(out.last_hidden_state[:, -1]).float().argmax(-1)
            if int(nxt) in self.eos: break
            ids.append(int(nxt))
            e = torch.cat([e, self.embed(nxt.view(1, 1))], 1)
            newp = (pos.max() + 1).view(*([1] * pos.dim())).expand(*pos.shape[:-1], 1)
            pos = torch.cat([pos, newp.to(pos.dtype)], -1)
        return self.tok.decode(ids, skip_special_tokens=True).strip()

    # ------------------------------------------------------------------ SparseVLM v1.0 (port of modelling_sparse_llama.py logic)
    @torch.inference_mode()
    def sparsevlm_run(self, ctx, retain, emb_ext=None, pos_ext=None, plan=None):
        """Progressive text-guided pruning. retain=None -> no pruning (identity, used as a self-test). With `plan` (from an earlier
        call) the stored decisions are replayed, which reproduces KV-cached decoding exactly because the model is causal.
        Returns (log-probs of the next token, dict(vis_per_layer, seq_per_layer), plan)."""
        layers = self.decoder.layers; nl = len(layers)
        emb = ctx["emb"] if emb_ext is None else emb_ext
        pos = ctx["pos"] if pos_ext is None else pos_ext
        nv, Lp, img = ctx["n_vis"], ctx["L"], ctx["img_idx"]
        v0, v1 = int(img[0]), int(img[-1]) + 1
        cos, sin = self._pe(pos, emb); h = emb
        pr_layers = []
        for l_ in SPARSE_LAYERS_OFFICIAL:                                  # 32-layer schedule -> this model's depth
            li = min(nl - 1, int(round(l_ * nl / SPARSE_NLAYERS_OFFICIAL)))
            if li not in pr_layers: pr_layers.append(li)
        counts = None if retain is None else [max(1, int(round(c * nv / SPARSE_NTOK_OFFICIAL))) for c in SPARSE_SCHEDULE[retain]]
        v_start, v_num, stage = v0, nv, 0
        rater = None
        if plan is None and retain is not None:                            # text raters (SparseVLM Sec. 3.2) from the INPUT embeddings
            vt, tt = emb[0, v0:v1].float(), emb[0, v1 + 1:Lp].float()      # visual tokens ; question + template tokens (after <|vision_end|>)
            m_vt = (vt @ tt.T).softmax(-1).mean(0)
            rater = (m_vt > m_vt.mean()).nonzero().squeeze(-1)
            if rater.numel() == 0: rater = torch.arange(tt.shape[0], device=emb.device)
        vis_pl, seq_pl, new_plan = [], [], []
        for li, layer in enumerate(layers):
            vis_pl.append(v_num); seq_pl.append(h.shape[1])
            if retain is None or li not in pr_layers or stage >= len(counts) or v_num <= 1:
                h = self._run_layer(layer, h, (cos, sin)); continue
            if plan is None:
                _TL.cap = {"mode": "first_full"}
                try:
                    out = self._run_layer(layer, h, (cos, sin)); cap = _TL.cap
                finally:
                    _TL.cap = None
                if "q" not in cap: raise RuntimeError("SparseVLM: attention not captured")
                q, k = cap["q"].float(), cap["k"].float()
                if k.shape[1] != q.shape[1]: k = k.repeat_interleave(q.shape[1] // k.shape[1], dim=1)
                rows = (v_start + v_num + 1) + rater                         # rater rows in the CURRENT sequence
                att = (q[:, :, rows, :] @ k.transpose(-1, -2)) * (q.shape[-1] ** -0.5)
                att = att.masked_fill((torch.arange(k.shape[2], device=h.device)[None, :] > rows[:, None])[None, None], float("-inf")).softmax(-1)
                rel = att.mean(1)[0][:, v_start:v_start + v_num].mean(0)    # mean over heads, then over raters -> relevance per visual token
                keep_n = min(counts[stage], v_num - 1)
                kept = torch.zeros(v_num, dtype=torch.bool, device=h.device); kept[rel.topk(keep_n).indices] = True
                pruned = (~kept).nonzero().squeeze(-1); merged = None
                if pruned.numel() > 0:                                       # token recycling: top-30% of pruned tokens -> clusters
                    n1 = int(pruned.numel() * 0.3) + 1
                    cand = pruned[rel[pruned].topk(min(n1, pruned.numel())).indices]
                    cn = int(cand.numel() / 10) + 1
                    _, assign = sv_cluster_and_merge(out[:, v_start + cand, :].float(), cn)
                    assign = assign[0]
                    rep = torch.stack([cand[(assign == j).nonzero().squeeze(-1)[rel[cand][(assign == j)].argmax()]] if bool((assign == j).any()) else cand[0]
                                       for j in range(cn)])               # RoPE position of a merged token = its most relevant member
                    merged = dict(cand=cand, assign=assign, rep=rep, cn=cn)
                new_plan.append(dict(kept=kept, merged=merged))
            else:
                out = self._run_layer(layer, h, (cos, sin)); kept, merged = plan[stage]["kept"], plan[stage]["merged"]
            kept_idx = kept.nonzero().squeeze(-1)
            ph, pc, ps = [out[:, :v_start], out[:, v_start + kept_idx]], [cos[..., :v_start, :], cos[..., v_start + kept_idx, :]], [sin[..., :v_start, :], sin[..., v_start + kept_idx, :]]
            if merged is not None:
                cn, asg = merged["cn"], merged["assign"]
                cnt = torch.bincount(asg, minlength=cn).clamp(min=1).float()
                mt = torch.zeros(1, cn, out.shape[-1], device=out.device).index_add_(1, asg, out[:, v_start + merged["cand"], :].float()) / cnt[None, :, None]
                ph.append(mt.to(out.dtype)); pc.append(cos[..., v_start + merged["rep"], :]); ps.append(sin[..., v_start + merged["rep"], :])
            ph.append(out[:, v_start + v_num:]); pc.append(cos[..., v_start + v_num:, :]); ps.append(sin[..., v_start + v_num:, :])
            h, cos, sin = torch.cat(ph, 1), torch.cat(pc, -2), torch.cat(ps, -2)
            v_num = int(kept_idx.numel()) + (merged["cn"] if merged is not None else 0)
            stage += 1
        logp = torch.log_softmax(self.lm_head(self.decoder.norm(h)[:, -1]).float(), -1)[0]
        return logp, dict(vis_per_layer=vis_pl, seq_per_layer=seq_pl), (plan if plan is not None else new_plan)

    @torch.inference_mode()
    def sparsevlm_generate(self, ctx, retain, first_logp, plan, max_new=10):
        emb, pos, ids, logp = ctx["emb"], ctx["pos"], [], first_logp
        for step in range(max_new):
            if step > 0:
                logp = self.sparsevlm_run(ctx, retain, emb_ext=emb, pos_ext=pos, plan=plan)[0]
            nxt = logp.argmax().view(1)
            if int(nxt) in self.eos: break
            ids.append(int(nxt))
            emb = torch.cat([emb, self.embed(nxt.view(1, 1))], 1)
            newp = (pos.max() + 1).view(*([1] * pos.dim())).expand(*pos.shape[:-1], 1)
            pos = torch.cat([pos, newp.to(pos.dtype)], -1)
        return self.tok.decode(ids, skip_special_tokens=True).strip()

    def _expand(self, t, B):
        return t.repeat(1, B, 1) if t.dim() == 3 else t.repeat(B, 1)

    @torch.inference_mode()
    def logp_for_token_masks(self, ctx, tok_keep):
        """tok_keep: (B, L) bool. Returns (B, V) float32 log-probs of the next token."""
        B_all, outs = tok_keep.shape[0], []
        chunk = CFG["MASK_CHUNK"]
        i = 0
        while i < B_all:
            try:
                tk = tok_keep[i:i + chunk]
                B = tk.shape[0]
                emb = ctx["emb"].expand(B, -1, -1)
                am = (tk & ctx["am"].bool()).long()
                out = self.decoder(inputs_embeds=emb, attention_mask=am, position_ids=self._expand(ctx["pos"], B),
                                   use_cache=False)
                h = out.last_hidden_state[:, -1]
                lg = self.lm_head(h).float()
                if not torch.isfinite(lg).all():
                    raise FloatingPointError("non-finite logits - try CFG['DTYPE']='float32' or another checkpoint")
                outs.append(torch.log_softmax(lg, -1))
                i += B
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if chunk == 1: raise
                chunk = max(1, chunk // 2)
        return torch.cat(outs, 0)

    def group_to_token_mask(self, ctx, gkeep):
        """gkeep: (B, G) bool -> (B, L) bool (non-visual tokens always kept)."""
        B = gkeep.shape[0]
        tk = torch.ones(B, ctx["L"], dtype=torch.bool, device=self.device)
        tk[:, ctx["img_idx"]] = gkeep[:, ctx["gid"]]
        return tk

    def logp_full(self, ctx):
        return self.logp_for_token_masks(ctx, torch.ones(1, ctx["L"], dtype=torch.bool, device=self.device))[0]

    @torch.inference_mode()
    def prefill_pruned(self, ctx, gkeep_1d):
        """PHYSICAL pruning: run the decoder on the shortened sequence only."""
        tk = self.group_to_token_mask(ctx, gkeep_1d[None])[0]
        keep = tk.nonzero().squeeze(-1)
        pos = ctx["pos"][..., keep] if ctx["pos"].dim() == 3 else ctx["pos"][:, keep]
        out = self.decoder(inputs_embeds=ctx["emb"][:, keep], attention_mask=torch.ones(1, len(keep), dtype=torch.long, device=self.device),
                           position_ids=pos, use_cache=False)
        return torch.log_softmax(self.lm_head(out.last_hidden_state[:, -1]).float(), -1)[0]

    @torch.inference_mode()
    def generate(self, ctx, gkeep_1d=None, max_new=10):
        """Greedy generation with a visual-token subset, by PHYSICAL pruning (dropped tokens removed, positions preserved).
        This is the same decoder call that the verified fidelity path uses (self-test 'masked-attention vs physical pruning').
        NOTE: an earlier version masked tokens through a forward-pre-hook inside model.generate(); in the installed transformers
        version that hook had no effect (every group method reproduced the full-model answer), so it is no longer used here.
        The full-token reference ('full') goes through the very same loop, so all configs are decoded identically."""
        if gkeep_1d is None:
            tk = torch.ones(ctx["L"], dtype=torch.bool, device=self.device)
        else:
            tk = self.group_to_token_mask(ctx, gkeep_1d[None])[0]
        return self.generate_custom(ctx, ctx["emb"], tk, max_new)

    # ------------------------------------------------------------------ FastV (in-LLM pruning, manual layer loop)
    def _pe(self, pos, h):
        """rotary (cos, sin) for the decoder, mirroring what the text model computes internally."""
        pid = pos[1:] if (pos.dim() == 3 and pos.shape[0] == 4) else pos
        cos, sin = self.decoder.rotary_emb(h, pid)
        return cos, sin

    @staticmethod
    def _sub_pe(pe, keep):
        return pe[0][..., keep, :], pe[1][..., keep, :]

    def _run_layer(self, layer, h, pe):
        out = layer(h, attention_mask=None, position_embeddings=pe)        # attention_mask=None + sdpa => causal
        return out[0] if isinstance(out, tuple) else out

    @torch.inference_mode()
    def fastv_prefix(self, ctx, K):
        """Layers 0..K-1 on the FULL sequence. Returns hidden state after layer K-1, rope embeddings, and the FastV score of every
        visual token = attention from the LAST token at layer K-1, averaged over heads (as in the official implementation)."""
        layers = self.decoder.layers
        h = ctx["emb"]; pe = self._pe(ctx["pos"], h)
        for i in range(K - 1):
            h = self._run_layer(layers[i], h, pe)
        _TL.cap = {}
        try:
            h = self._run_layer(layers[K - 1], h, pe)
            cap = _TL.cap
        finally:
            _TL.cap = None
        if "q" not in cap:
            raise RuntimeError("FastV: could not capture attention (is attn_implementation='sdpa' active?)")
        q, k = cap["q"].float(), cap["k"].float()
        if k.shape[1] != q.shape[1]:
            k = k.repeat_interleave(q.shape[1] // k.shape[1], dim=1)
        att = torch.softmax((q @ k.transpose(-1, -2)) * (q.shape[-1] ** -0.5), dim=-1)      # (1, heads, 1, L)
        score = att.mean(1)[0, 0]
        return h, pe, score[ctx["img_idx"]]

    @torch.inference_mode()
    def fastv_logp(self, ctx, h_pref, pe, keep_tok, K):
        """Layers K..end on the pruned sequence (visual tokens outside `keep_tok` removed, positions preserved)."""
        layers = self.decoder.layers
        keep = keep_tok.nonzero().squeeze(-1)
        hs, pes = h_pref[:, keep], self._sub_pe(pe, keep)
        for i in range(K, len(layers)):
            hs = self._run_layer(layers[i], hs, pes)
        hs = self.decoder.norm(hs)
        return torch.log_softmax(self.lm_head(hs[:, -1]).float(), -1)[0]

    @torch.inference_mode()
    def fastv_generate(self, ctx, keep_tok, K, max_new=10):
        """Greedy decoding with FastV pruning (prompt visual tokens dropped from layer K on; no KV cache, short answers only)."""
        emb, pos, kt, ids = ctx["emb"], ctx["pos"], keep_tok.clone(), []
        layers = self.decoder.layers
        for _ in range(max_new):
            pe = self._pe(pos, emb); h = emb
            for i in range(K): h = self._run_layer(layers[i], h, pe)
            keep = kt.nonzero().squeeze(-1)
            hs, pes = h[:, keep], self._sub_pe(pe, keep)
            for i in range(K, len(layers)): hs = self._run_layer(layers[i], hs, pes)
            nxt = self.lm_head(self.decoder.norm(hs)[:, -1]).float().argmax(-1)
            if int(nxt) in self.eos: break
            ids.append(int(nxt))
            emb = torch.cat([emb, self.embed(nxt.view(1, 1))], 1)
            newp = (pos.max() + 1).view(*([1] * pos.dim())).expand(*pos.shape[:-1], 1)
            pos = torch.cat([pos, newp.to(pos.dtype)], -1)
            kt = torch.cat([kt, torch.ones(1, dtype=torch.bool, device=kt.device)])
        return self.tok.decode(ids, skip_special_tokens=True).strip()


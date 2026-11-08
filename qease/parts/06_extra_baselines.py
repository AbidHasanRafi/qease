# %% ------------------------------------ 6b. RUNNER EXTENSIONS + NEW BASELINES (DivPrune, CDPruner, DART, PyramidDrop)
# Added to the class AFTER its definition (so the validated body above is untouched).
def _vlm_new_pos(self, pos, T):
    """rope positions of T new text tokens appended after the prompt (all rope axes advance together, as in generate_custom)."""
    new = pos.max() + 1 + torch.arange(T, device=pos.device)
    new = new.view(1, 1, T).expand(pos.shape[0], pos.shape[1], T) if pos.dim() == 3 else new.view(1, T).expand(pos.shape[0], T)
    return torch.cat([pos, new.to(pos.dtype)], -1)

@torch.inference_mode()
def _vlm_tf_logp(self, ctx, tok_keep, ans_ids):
    """TEACHER-FORCED log-likelihood of the answer tokens ans_ids (T,) after the prompt, under B token masks tok_keep (B, L).
    returns (B,) summed log-prob of the sequence.  Used for accuracy-aware / answer-distillation labels."""
    B_all, T = tok_keep.shape[0], int(len(ans_ids))
    ids = torch.as_tensor(list(ans_ids), device=self.device, dtype=torch.long)
    aemb = self.embed(ids)[None]
    pos_full = self._new_pos(ctx["pos"], T)
    outs, chunk, i = [], CFG["MASK_CHUNK"], 0
    while i < B_all:
        try:
            tk = tok_keep[i:i + chunk]; B = tk.shape[0]
            emb = torch.cat([ctx["emb"].expand(B, -1, -1), aemb.expand(B, -1, -1)], 1)
            am = torch.cat([(tk & ctx["am"].bool()).long(), torch.ones(B, T, dtype=torch.long, device=self.device)], 1)
            out = self.decoder(inputs_embeds=emb, attention_mask=am, position_ids=self._expand(pos_full, B), use_cache=False)
            h = out.last_hidden_state[:, ctx["L"] - 1: ctx["L"] - 1 + T]
            lp = torch.log_softmax(self.lm_head(h).float(), -1)
            outs.append(lp.gather(-1, ids[None, :, None].expand(B, T, 1)).squeeze(-1).sum(-1))
            i += B
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if chunk == 1: raise
            chunk = max(1, chunk // 2)
    return torch.cat(outs, 0)

@torch.inference_mode()
def _vlm_decode(self, ctx, emb, keep_tok, max_new=10):
    """greedy decoding on a physically pruned prompt; returns (token ids, log-probs of the FIRST answer token)."""
    keep = keep_tok.nonzero().squeeze(-1)
    e = emb[:, keep]; pos = ctx["pos"][..., keep] if ctx["pos"].dim() == 3 else ctx["pos"][:, keep]
    ids, first = [], None
    for step in range(max_new):
        out = self.decoder(inputs_embeds=e, attention_mask=torch.ones(1, e.shape[1], dtype=torch.long, device=self.device),
                           position_ids=pos, use_cache=False)
        lg = self.lm_head(out.last_hidden_state[:, -1]).float()
        if step == 0: first = torch.log_softmax(lg, -1)[0]
        nxt = lg.argmax(-1)
        if int(nxt) in self.eos: break
        ids.append(int(nxt))
        e = torch.cat([e, self.embed(nxt.view(1, 1))], 1)
        newp = (pos.max() + 1).view(*([1] * pos.dim())).expand(*pos.shape[:-1], 1)
        pos = torch.cat([pos, newp.to(pos.dtype)], -1)
    return ids, first

@torch.inference_mode()
def _vlm_dart_prefix(self, ctx, K):
    """DART-style: layers 0..K-1 on the full sequence; returns hidden state, rope, and the L2 norm of the layer-K keys (pivot criterion)."""
    layers = self.decoder.layers
    h = ctx["emb"]; pe = self._pe(ctx["pos"], h)
    for i in range(K - 1): h = self._run_layer(layers[i], h, pe)
    _TL.cap = {}
    try:
        h = self._run_layer(layers[K - 1], h, pe); cap = _TL.cap
    finally:
        _TL.cap = None
    if "k" not in cap: raise RuntimeError("DART: key capture failed (is attn_implementation='sdpa' active?)")
    return h, pe, cap["k"].float()[0].norm(dim=-1).mean(0)

@torch.inference_mode()
def _vlm_pdrop_run(self, ctx, lam, emb_ext=None, pos_ext=None, plan=None):
    """PyramidDrop (Xing et al., CVPR'25): at 3 depths keep the top-`lam` fraction of the CURRENT visual tokens ranked by the attention of the LAST
    token.  lam>=1 -> no pruning (identity, self-test).  `plan` replays earlier decisions (exact for causal decoding)."""
    layers = self.decoder.layers; nl = len(layers)
    emb = ctx["emb"] if emb_ext is None else emb_ext; pos = ctx["pos"] if pos_ext is None else pos_ext
    nv, v0 = ctx["n_vis"], int(ctx["img_idx"][0])
    cos, sin = self._pe(pos, emb); h = emb
    pl = sorted({max(1, min(nl - 1, int(round(f * nl)))) for f in CFG["PDROP_LAYER_FRACS"]})
    v_num, stage, vis_pl, seq_pl, new_plan = nv, 0, [], [], []
    for li, layer in enumerate(layers):
        vis_pl.append(v_num); seq_pl.append(h.shape[1])
        if lam >= 1.0 or li not in pl or v_num <= 1:
            h = self._run_layer(layer, h, (cos, sin)); continue
        if plan is None:
            _TL.cap = {}
            try:
                out = self._run_layer(layer, h, (cos, sin)); cap = _TL.cap
            finally:
                _TL.cap = None
            if "q" not in cap: raise RuntimeError("PyramidDrop: attention not captured")
            q, k = cap["q"].float(), cap["k"].float()
            if k.shape[1] != q.shape[1]: k = k.repeat_interleave(q.shape[1] // k.shape[1], dim=1)
            att = torch.softmax((q @ k.transpose(-1, -2)) * (q.shape[-1] ** -0.5), dim=-1)
            sc = att.mean(1)[0, 0][v0:v0 + v_num]
            kept = torch.zeros(v_num, dtype=torch.bool, device=h.device)
            kept[sc.topk(min(v_num, max(1, int(round(lam * v_num))))).indices] = True
            new_plan.append(kept)
        else:
            out = self._run_layer(layer, h, (cos, sin)); kept = plan[stage]
        kept_idx = kept.nonzero().squeeze(-1)
        keep_all = torch.cat([torch.arange(v0, device=h.device), v0 + kept_idx, torch.arange(v0 + v_num, h.shape[1], device=h.device)])
        h = out[:, keep_all]; cos, sin = cos[..., keep_all, :], sin[..., keep_all, :]
        v_num = int(kept_idx.numel()); stage += 1
    logp = torch.log_softmax(self.lm_head(self.decoder.norm(h)[:, -1]).float(), -1)[0]
    return logp, dict(vis_per_layer=vis_pl, seq_per_layer=seq_pl), (plan if plan is not None else new_plan)

@torch.inference_mode()
def _vlm_pdrop_generate(self, ctx, lam, plan, max_new=10):
    emb, pos, ids = ctx["emb"], ctx["pos"], []
    for step in range(max_new):
        logp = self.pdrop_run(ctx, lam, emb_ext=emb, pos_ext=pos, plan=plan)[0]
        nxt = logp.argmax().view(1)
        if int(nxt) in self.eos: break
        ids.append(int(nxt))
        emb = torch.cat([emb, self.embed(nxt.view(1, 1))], 1)
        newp = (pos.max() + 1).view(*([1] * pos.dim())).expand(*pos.shape[:-1], 1)
        pos = torch.cat([pos, newp.to(pos.dtype)], -1)
    return self.tok.decode(ids, skip_special_tokens=True).strip()

VLMRunner._new_pos, VLMRunner.tf_logp, VLMRunner.decode, VLMRunner.dart_prefix = _vlm_new_pos, _vlm_tf_logp, _vlm_decode, _vlm_dart_prefix
VLMRunner.pdrop_run, VLMRunner.pdrop_generate = _vlm_pdrop_run, _vlm_pdrop_generate

def decode_text(r, ids):
    return r.tok.decode(ids, skip_special_tokens=True).strip()

def pdrop_lambda(f, nl):
    """per-stage retention lambda whose average visual-token fraction over layers equals f (layers 0..p1 full, then lam, lam^2, lam^3)."""
    if f >= 1.0: return 1.0
    pl = sorted({max(1, min(nl - 1, int(round(x * nl)))) for x in CFG["PDROP_LAYER_FRACS"]})
    segs = [pl[0] + 1] + [pl[i + 1] - pl[i] for i in range(len(pl) - 1)] + [nl - pl[-1] - 1]
    avg = lambda l: sum(s * (l ** i) for i, s in enumerate(segs)) / nl
    if avg(0.0) >= f: return 0.0
    lo, hi = 0.0, 1.0
    for _ in range(30):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if avg(mid) < f else (lo, mid)
    return 0.5 * (lo + hi)

def divprune_order(ctx):
    """DivPrune-style (Alvar et al. CVPR'25): greedy max-min diversity over the visual tokens (farthest-point sampling in cosine distance)."""
    v = F.normalize(ctx["emb"][0, ctx["img_idx"]].float(), dim=-1); n = v.shape[0]
    dist = 1 - v @ v.T
    first = int(dist.mean(1).argmax())
    order, taken = [first], torch.zeros(n, dtype=torch.bool, device=v.device); taken[first] = True
    mind = dist[first].clone()
    for _ in range(n - 1):
        j = int(mind.masked_fill(taken, -1).argmax()); order.append(j); taken[j] = True
        mind = torch.minimum(mind, dist[j])
    return torch.tensor(order)

def cdpruner_order(ctx, theta=2.0):
    """CDPruner-style (conditional DPP, Zhang et al. 2025): kernel L = diag(r) S diag(r), r = exp(theta * question relevance), S = cosine similarity of the
    visual tokens; greedy MAP inference (fast Cholesky update).  Relevance uses the LLM-space question embedding (the original uses CLIP text features)."""
    v = F.normalize(ctx["emb"][0, ctx["img_idx"]].float(), dim=-1); n = v.shape[0]
    q = F.normalize(ctx["qemb"].float().mean(0, keepdim=True), dim=-1)
    rel = (v @ q.T).squeeze(-1); rel = (rel - rel.min()) / (rel.max() - rel.min() + 1e-6)
    rr = torch.exp(theta * rel)
    Lk = rr[:, None] * (v @ v.T) * rr[None, :]
    d2 = Lk.diag().clone(); Cm = torch.zeros(n, n, device=v.device)
    taken = torch.zeros(n, dtype=torch.bool, device=v.device); order = []; j = int(d2.argmax())
    for t in range(n):
        order.append(j); taken[j] = True
        if t == n - 1: break
        e = (Lk[j] - (Cm[:t].T @ Cm[:t, j] if t > 0 else 0.0)) / torch.sqrt(d2[j].clamp_min(1e-9))
        Cm[t] = e; d2 = (d2 - e * e).masked_fill(taken, -1e9); j = int(d2.argmax())
    return torch.tensor(order)

def dart_keep(ctx, h_pref, knorm, n, n_piv):
    """DART-style (Wen et al. 2025): pivots = visual tokens with the largest key norm; keep the pivots plus the tokens LEAST similar to them (drop duplicates)."""
    img = ctx["img_idx"]; nv = len(img)
    n = int(min(nv, max(1, n))); p = int(min(n_piv, n))
    piv = knorm[img].topk(p).indices
    hv = F.normalize(h_pref[0, img].float(), dim=-1)
    msim = (hv @ hv[piv].T).max(1).values
    rest = torch.ones(nv, dtype=torch.bool, device=h_pref.device); rest[piv] = False
    cand = rest.nonzero().squeeze(-1)
    pick = cand[msim[cand].topk(n - p, largest=False).indices] if n > p else cand[:0]
    sel = torch.cat([piv, pick])
    keep = torch.ones(ctx["L"], dtype=torch.bool, device=h_pref.device); keep[img] = False; keep[img[sel]] = True
    return keep, sel

def greedy_order(ctx, scores, beta):
    """redundancy-aware set construction (MMR): next group maximises  z(score) - beta * max cosine-similarity to the groups already chosen.
    beta=0 reproduces the additive top-k.  Answers the 'additive score cannot represent redundancy' criticism."""
    s = scores.float().cpu(); s = (s - s.mean()) / (s.std() + 1e-6)
    if beta <= 0: return torch.argsort(-s)
    f = F.normalize(ctx["gfeat"].float().cpu(), dim=-1); sim = f @ f.T
    G = len(s); taken = torch.zeros(G, dtype=torch.bool); mx = torch.zeros(G); order = []
    for _ in range(G):
        v = (s - beta * mx).masked_fill(taken, -1e9); j = int(v.argmax()); order.append(j); taken[j] = True
        mx = torch.maximum(mx, sim[j])
    return torch.tensor(order)

# ---- self-tests of the new baselines (a baseline is only used if it passes)
DIV_OK = CDP_OK = DART_OK = PD_OK = False
if RUNNERS and "_rec" in globals() and any(s in CFG["STAGES"] for s in ("eval", "dec", "hires", "ablate", "shift")):
    _r0 = RUNNERS[0]
    with torch.inference_mode():
        try:
            _c = _r0.prepare(_rec); _lpf = _r0.logp_full(_c); _nv = _c["n_vis"]
            if CFG["DIVPRUNE"]:
                try:
                    _o = divprune_order(_c); DIV_OK = bool(len(_o) == _nv and len(set(_o.tolist())) == _nv)
                except Exception as e:
                    FAILS.append(("divprune-selftest", "-", repr(e)[:200], traceback.format_exc()))
                log(f"DIVPRUNE_OK={DIV_OK}")
            if CFG["CDPRUNER"]:
                try:
                    _o = cdpruner_order(_c, CFG["CDP_THETA"]); CDP_OK = bool(len(_o) == _nv and len(set(_o.tolist())) == _nv)
                except Exception as e:
                    FAILS.append(("cdpruner-selftest", "-", repr(e)[:200], traceback.format_exc()))
                log(f"CDPRUNER_OK={CDP_OK}")
            if CFG["DART"]:
                try:
                    _K = CFG["DART_K"]; _h, _pe, _kn = _r0.dart_prefix(_c, _K)
                    _keep_all, _ = dart_keep(_c, _h, _kn, _nv, CFG["DART_PIVOTS"])
                    _d = js_div(_lpf, _r0.fastv_logp(_c, _h, _pe, _keep_all, _K)[None]).item()
                    _keep_q, _ = dart_keep(_c, _h, _kn, max(2, _nv // 4), CFG["DART_PIVOTS"])
                    DART_OK = bool(_d < 5e-3 and int(_keep_q[_c["img_idx"]].sum()) == max(2, _nv // 4) and bool(torch.isfinite(_kn).all()))
                    log(f"DART self-test: JS(no pruning vs model)={_d:.2e} (need <5e-3) | kept {int(_keep_q[_c['img_idx']].sum())}/{max(2, _nv // 4)} | OK={DART_OK}")
                except Exception as e:
                    FAILS.append(("dart-selftest", "-", repr(e)[:200], traceback.format_exc()))
                    log(f"DART self-test crashed -> baseline disabled: {repr(e)[:160]}")
            if CFG["PYRAMIDDROP"]:
                try:
                    _l1, _, _ = _r0.pdrop_run(_c, 1.0); _d1 = js_div(_lpf, _l1[None]).item()
                    _l2, _st, _pl = _r0.pdrop_run(_c, 0.5); _l3 = _r0.pdrop_run(_c, 0.5, plan=_pl)[0]
                    _d3 = js_div(_l2, _l3[None]).item()
                    PD_OK = bool(_d1 < 5e-3 and _d3 < 5e-3 and _st["vis_per_layer"][-1] < _nv and len(_pl) == 3 and torch.isfinite(_l2).all())
                    log(f"PYRAMIDDROP self-test: JS(no pruning)={_d1:.2e} | JS(plan replay)={_d3:.2e} | visual tokens/layer {_st['vis_per_layer'][0]}->{_st['vis_per_layer'][-1]} | OK={PD_OK}")
                except Exception as e:
                    FAILS.append(("pdrop-selftest", "-", repr(e)[:200], traceback.format_exc()))
                    log(f"PYRAMIDDROP self-test crashed -> baseline disabled: {repr(e)[:160]}")
        except Exception as e:
            FAILS.append(("newbaseline-selftest", "-", repr(e)[:200], traceback.format_exc()))
            log(f"new-baseline self-tests crashed: {repr(e)[:160]}")
    for _r in RUNNERS: _r.vz_on = bool(VZ_OK and any(s in CFG["STAGES"] for s in ("eval", "dec", "hires")))

def eval_extras(r, ctx, lp_full, orders, out):
    """first-token fidelity of the additional published baselines on the K_GRID (same protocol / budgets as every other method)."""
    nv, L, nlay = ctx["n_vis"], ctx["L"], len(r.decoder.layers)
    ns = [int(min(nv, max(1, math.ceil(f * nv)))) for f in CFG["K_GRID"]]
    am = lp_full.argmax()
    def put(name, js, fl, jsd, fld, kf):
        out["js"][name], out["flip"][name], out["js_del"][name], out["flip_del"][name], out["kfrac"][name] = js, fl, jsd, fld, kf
    for name, ok, fn in [("divprune", DIV_OK, lambda: divprune_order(ctx)), ("cdpruner", CDP_OK, lambda: cdpruner_order(ctx, CFG["CDP_THETA"]))]:
        if not ok: continue
        try:
            cv = token_level_curves(r, ctx, lp_full, fn().to(r.device))
            put(name, cv["js"], cv["flip"], cv["js_del"], cv["flip_del"], cv["kfrac"])
        except Exception as e:
            FAILS.append(("eval-" + name, ctx.get("ds", "-"), repr(e)[:200], traceback.format_exc()))
    if DART_OK:
        try:
            K = CFG["DART_K"]; h_pref, pe, kn = r.dart_prefix(ctx, K)
            lp_i, lp_d, eff = [], [], []
            for n in ns:
                keep, sel = dart_keep(ctx, h_pref, kn, n, CFG["DART_PIVOTS"])
                lp_i.append(r.fastv_logp(ctx, h_pref, pe, keep, K))
                drop = torch.ones(L, dtype=torch.bool, device=r.device); drop[ctx["img_idx"][sel]] = False
                lp_d.append(r.fastv_logp(ctx, h_pref, pe, drop, K))
                eff.append((K + (nlay - K) * (L - nv + n) / L) / nlay)
            lp_i, lp_d = torch.stack(lp_i), torch.stack(lp_d)
            put("dart", js_div(lp_full, lp_i).cpu().numpy(), (lp_i.argmax(-1) != am).cpu().numpy(), js_div(lp_full, lp_d).cpu().numpy(),
                (lp_d.argmax(-1) != am).cpu().numpy(), np.array(ns) / nv)
            out["dart_flops_frac"] = np.array(eff)
        except Exception as e:
            FAILS.append(("eval-dart", ctx.get("ds", "-"), repr(e)[:200], traceback.format_exc()))
    if PD_OK:
        try:
            js_l, fl_l, kf_l, eff = [], [], [], []
            for f in CFG["K_GRID"]:
                lam = pdrop_lambda(f, nlay); lp_s, st, _ = r.pdrop_run(ctx, lam)
                js_l.append(float(js_div(lp_full, lp_s[None])[0])); fl_l.append(bool(lp_s.argmax() != am))
                kf_l.append(sum(st["vis_per_layer"]) / nlay / nv); eff.append(sum(st["seq_per_layer"]) / nlay / L)
            nan = np.full(len(js_l), np.nan)
            put("pdrop", np.array(js_l), np.array(fl_l), nan, nan, np.array(kf_l))
            out["pdrop_flops_frac"] = np.array(eff)
        except Exception as e:
            FAILS.append(("eval-pdrop", ctx.get("ds", "-"), repr(e)[:200], traceback.format_exc()))
    return out


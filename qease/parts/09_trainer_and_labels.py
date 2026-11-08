# %% ------------------- 9b. GENERAL SELECTOR TRAINER (label / architecture / loss / mask ablations) + EXTRA LABEL GENERATORS
LN2 = math.log(2)

class MLPScorer(nn.Module):
    """per-group scorer WITHOUT cross-group attention (same inputs as the selector; the question interacts multiplicatively)."""
    def __init__(self, D, d=256):
        super().__init__()
        self.gp = nn.Sequential(nn.LayerNorm(D), nn.Linear(D, d))
        self.qp = nn.Sequential(nn.LayerNorm(D), nn.Linear(D, d))
        self.pos = nn.Sequential(nn.Linear(2, d), nn.GELU(), nn.Linear(d, d))
        self.head = nn.Sequential(nn.LayerNorm(3 * d), nn.Linear(3 * d, d), nn.GELU(), nn.Dropout(0.1), nn.Linear(d, 1))

    def encode(self, gf, gpos, gvalid, qe, qvalid):
        g = self.gp(gf.float()); q = self.qp(qe.float())
        qm = (q * qvalid[..., None].float()).sum(1) / qvalid.float().sum(1, keepdim=True).clamp_min(1)
        qm = qm[:, None].expand_as(g)
        s = self.head(torch.cat([g, qm * g, self.pos(gpos)], -1)).squeeze(-1).masked_fill(~gvalid, -1e4)
        return g, s

def pairs_from_rid(rid):
    """all pairs of probe masks that share a keep-ratio bin (so the pair differs only in WHICH groups are kept)."""
    rid = rid.tolist() if torch.is_tensor(rid) else list(rid)
    ia, ib = [], []
    for b in sorted(set(rid)):
        idx = [i for i, x in enumerate(rid) if x == b]
        for x in range(len(idx)):
            for y in range(x + 1, len(idx)):
                ia.append(idx[x]); ib.append(idx[y])
    return torch.tensor(ia), torch.tensor(ib)

def mask_sel_for(mt):
    """subset of the K probe masks: iid-random only | blobs only | one iid + one blob per ratio (K/2)."""
    idx = np.arange(len(RATIOS) * MPR); within = idx % MPR
    if mt == "iid": return idx[within < MPR // 2]
    if mt == "blob": return idx[within >= MPR // 2]
    if mt == "k12": return idx[(within == 0) | (within == MPR // 2)]
    return None

def gold_answer(rec):
    a = [str(x) for x in rec["answers"] if str(x).strip()]
    if not a: return ""
    cnt = defaultdict(int)
    for x in a: cnt[x] += 1
    return max(cnt.items(), key=lambda kv: kv[1])[0]

def train_general(name, recs, kind, device, use_q=True, arch="selector", mask_types=None, rank_w=None, risk_w=1.0, epochs=None, lab_map=None):
    """kind: js | shuf (pairwise ranking on first-token JS; shuf = labels of another image)  |  gold | seq (aux likelihood labels, needs lab_map[uid][kind])
             | attn | logo (listwise softmax over groups, needs lab_map[uid] = per-group array).   Returns the model on CPU; saved as ckpt selector_<name>.pt"""
    ck = CKPT_DIR / f"selector_{name}.pt"
    D = recs[0]["gfeat"].shape[1]
    torch.manual_seed(CFG["SEED"])
    model = (MLPScorer(D, CFG["D_MODEL"]) if arch == "mlp" else Selector(D, CFG["D_MODEL"], CFG["N_LAYERS"], CFG["N_HEADS"], use_q)).to(device)
    if ck.exists():
        model.load_state_dict(torch.load(ck, map_location=device)["state"]); return model.eval().cpu()
    rank_w = CFG["RANK_W"] if rank_w is None else rank_w
    E = epochs or CFG["ABL_EPOCHS"]
    listwise = kind in ("attn", "logo")
    if lab_map is not None:
        recs = [r for r in recs if r["uid"] in lab_map]
    N = len(recs); assert N > 50, f"{name}: only {N} labelled records"
    T = assemble(recs, device)
    ms = mask_sel_for(mask_types)
    rid = T["rid"].clone()
    if ms is not None:
        T["masks"] = T["masks"][:, torch.as_tensor(ms, device=device)]; rid = rid[torch.as_tensor(ms)]
    Lab = None
    if not listwise:
        if kind in ("js", "shuf"): Lab = T["js"].clone()
        else:
            Lab = torch.zeros(N, T["js"].shape[1], device=device)
            for i, r in enumerate(recs):
                d = torch.as_tensor(lab_map[r["uid"]][kind + "_delta"], dtype=torch.float32, device=device)
                Lab[i] = (1 - torch.exp(-d.clamp_min(0))) * LN2
        if kind == "shuf":
            Lab = Lab[torch.randperm(N, generator=torch.Generator().manual_seed(123)).to(device)]
        if ms is not None: Lab = Lab[:, torch.as_tensor(ms, device=device)]
        ia, ib = [x.to(device) for x in pairs_from_rid(rid)]
    else:
        A = torch.zeros(N, CFG["GMAX"], device=device)
        for i, r in enumerate(recs):
            a = torch.as_tensor(np.asarray(lab_map[r["uid"]], dtype=np.float32), device=device).clamp_min(0)
            A[i, :len(a)] = a[:CFG["GMAX"]]
        A = A / A.sum(1, keepdim=True).clamp_min(1e-9)
    val_idx = [i for i, r in enumerate(recs) if r["split"] == "val"]
    tr_idx = [i for i, r in enumerate(recs) if r["split"] != "val"]
    if len(val_idx) < 20:
        val_idx, tr_idx = tr_idx[-max(20, N // 12):], tr_idx[:-max(20, N // 12)]
    opt = torch.optim.AdamW(model.parameters(), lr=CFG["LR"], weight_decay=CFG["WD"])
    steps = E * math.ceil(len(tr_idx) / CFG["BATCH"])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, CFG["LR"], total_steps=steps, pct_start=0.1)
    scaler = torch.cuda.amp.GradScaler(enabled=device.startswith("cuda"))
    has_risk = hasattr(model, "risk") and risk_w > 0 and not listwise

    def batch_loss(idx):
        b = {k: T[k][idx] for k in ("gf", "gpos", "gv", "qe", "qv", "masks")}
        with torch.autocast("cuda", dtype=torch.float16, enabled=device.startswith("cuda")):
            h, s = model.encode(b["gf"], b["gpos"], b["gv"], b["qe"], b["qv"])
            rl = model.risk(h, b["gv"], b["masks"]) if has_risk else None
        s = s.float()
        if listwise:
            lp = torch.log_softmax(s.masked_fill(~b["gv"], -1e4), -1)
            loss = -(A[idx] * lp).sum(1).mean()
            return loss, loss.detach(), spearman_rows(s.masked_fill(~b["gv"], -1e4), A[idx])
        lab = Lab[idx]
        S = torch.einsum("bkg,bg->bk", b["masks"].float(), s.masked_fill(~b["gv"], 0))
        diff = S[:, ia] - S[:, ib]; dj = lab[:, ib] - lab[:, ia]; valid = dj.abs() > CFG["RANK_MARGIN"]
        l_rank = (F.binary_cross_entropy_with_logits(diff, (dj > 0).float(), reduction="none") * valid).sum() / valid.sum().clamp_min(1)
        acc = (((diff > 0) == (dj > 0)) & valid).sum() / valid.sum().clamp_min(1)
        loss = rank_w * l_rank
        if rl is not None: loss = loss + risk_w * F.binary_cross_entropy_with_logits(rl.float(), (lab / LN2).clamp(0, 1))
        return loss, loss.detach(), acc.detach().item()

    best, best_state, hist = 1e9, None, []
    for ep in range(E):
        model.train(); perm = torch.tensor(tr_idx)[torch.randperm(len(tr_idx))]; tl = 0.0
        for i in range(0, len(perm), CFG["BATCH"]):
            idx = perm[i:i + CFG["BATCH"]].to(device)
            loss = batch_loss(idx)[0]
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); scaler.step(opt); scaler.update(); sched.step(); tl += loss.item() * len(idx)
        model.eval()
        with torch.no_grad():
            vs = [batch_loss(torch.tensor(val_idx[i:i + 64], device=device)) for i in range(0, len(val_idx), 64)]
        vl = float(np.mean([v[1].item() for v in vs])); vm = float(np.mean([v[2] for v in vs]))
        hist.append(dict(name=name, epoch=ep, train_loss=tl / max(1, len(tr_idx)), val_loss=vl, val_metric=vm))
        if vl < best: best, best_state = vl, {k: x.detach().clone().cpu() for k, x in model.state_dict().items()}
        if ep % 5 == 0 or ep == E - 1: log(f"[train:{name}] ep{ep:02d} train={tl / max(1, len(tr_idx)):.4f} val={vl:.4f} metric={vm:.3f} (N={N})")
    model.load_state_dict(best_state); model.eval()
    torch.save(dict(state=best_state, name=name, arch=arch, use_q=use_q, kind=kind, val=best, n=N), ck)
    pd.DataFrame(hist).to_csv(CKPT_DIR / f"history_{name}.csv", index=False)
    del T; gc.collect(); torch.cuda.empty_cache()
    return model.cpu()

def load_extra_selectors(prefixes=None):
    """restore every trained selector in CKPT_DIR (ablation / leave-one-dataset-out variants) into SELECTORS."""
    for p in sorted(CKPT_DIR.glob("selector_*.pt")):
        n = p.stem[len("selector_"):]
        if n in ("full", "noq") or n in SELECTORS: continue
        if prefixes is not None and not any(n.startswith(x) for x in prefixes): continue
        ck = torch.load(p, map_location="cpu"); st = ck["state"]; D = st["gp.0.weight"].shape[0]
        m = MLPScorer(D, CFG["D_MODEL"]) if ck.get("arch") == "mlp" else Selector(D, CFG["D_MODEL"], CFG["N_LAYERS"], CFG["N_HEADS"], ck.get("use_q", True))
        m.load_state_dict(st); SELECTORS[n] = m.eval()

def train_many(jobs, recs_by_group, lab_maps):
    """jobs: list of (name, group, kwargs).  Trains them two at a time (one per GPU) in threads."""
    todo = [j for j in jobs if not (CKPT_DIR / f"selector_{j[0]}.pt").exists()]
    log(f"[train_many] {len(todo)} of {len(jobs)} selectors to train")
    for i in range(0, len(todo), max(1, len(DEVICES))):
        chunk, ths = todo[i:i + len(DEVICES)], []
        def _tw(job, dev):
            if dev.startswith("cuda"): torch.cuda.set_device(dev)
            name, grp, kw = job; kw = dict(kw); src = kw.pop("labsrc", None)
            try:
                train_general(name, recs_by_group[grp], device=dev, lab_map=(lab_maps.get(src) if src else None), **kw)
            except Exception as e:
                FAILS.append(("train_many", name, repr(e)[:200], traceback.format_exc())); log(f"training '{name}' failed: {repr(e)[:160]}")
        for j, job in enumerate(chunk):
            t = threading.Thread(target=_tw, args=(job, DEVICES[j % len(DEVICES)]), daemon=True); t.start(); ths.append(t)
        for t in ths:
            while t.is_alive(): t.join(0.5)

# ---- extra label generators (run on images that were already probed, with the SAME masks: make_probe_masks is seeded by the uid)
@torch.inference_mode()
def aux_one(r, rec):
    """accuracy-aware label (gold-answer likelihood drop) + final-answer distillation label (likelihood drop of the FULL model's own decoded answer)."""
    ctx = r.prepare(rec); G = ctx["G"]
    masks, rid = make_probe_masks(G, ctx["gh"], ctx["gw"], stable_hash(rec["uid"]) % 10**6)
    tk_full = torch.ones(1, ctx["L"], dtype=torch.bool, device=r.device)
    tkm = r.group_to_token_mask(ctx, masks.to(r.device))
    ids_full = r.decode(ctx, ctx["emb"], tk_full[0], CFG["GEN_MAX_NEW"])[0][:CFG["AUX_MAXTOK"]]
    gold = gold_answer(rec); best = None
    for cand in {gold, gold.capitalize(), gold.lower()}:
        ids = r.tok(cand, add_special_tokens=False).input_ids[:CFG["AUX_MAXTOK"]]
        if not ids: continue
        ll = float(r.tf_logp(ctx, tk_full, ids)[0])
        if best is None or ll > best[0]: best = (ll, ids)
    out = dict(uid=rec["uid"], K=int(masks.shape[0]))
    for tag, ids in (("gold", best[1] if best else []), ("seq", ids_full)):
        if len(ids) == 0:
            out[tag + "_delta"] = np.zeros(masks.shape[0], np.float32); out[tag + "_ok"] = False; continue
        llf = r.tf_logp(ctx, tk_full, ids)[0]; llm = r.tf_logp(ctx, tkm, ids)
        out[tag + "_delta"] = (llf - llm).float().cpu().numpy(); out[tag + "_ok"] = True
    return out

@torch.inference_mode()
def logo_one(r, rec):
    """single-group (leave-one-group-out) necessity labels: JS of the answer when exactly ONE group is removed."""
    ctx = r.prepare(rec); G = ctx["G"]; lp_full = r.logp_full(ctx)
    eye = ~torch.eye(G, dtype=torch.bool)
    lp = r.logp_for_token_masks(ctx, r.group_to_token_mask(ctx, eye.to(r.device)))
    return dict(uid=rec["uid"], delta=js_div(lp_full, lp).float().cpu().numpy())

def attn_one(r, rec):
    ctx = r.prepare(rec); sc = r.fastv_prefix(ctx, CFG["FASTV_K"])[2].float()
    return dict(uid=rec["uid"], delta=(torch.zeros(ctx["G"], device=r.device).index_add_(0, ctx["gid"], sc) / ctx["gcount"]).cpu().numpy())

def label_shard(fn, outdir, tag):
    def _w(r, shard, tick):
        sid, recs = shard
        path = outdir / f"shard_{sid:04d}.pt"
        if path.exists(): tick(len(recs)); return 0
        out = []
        for rec in recs:
            if STOP.is_set(): return 0
            try: out.append(fn(r, rec))
            except Exception as e:
                FAILS.append((tag, rec["uid"], repr(e)[:200], traceback.format_exc()))
                if "out of memory" in str(e).lower(): torch.cuda.empty_cache()
            tick()
        torch.save(out, path); return len(out)
    return _w

def probed_uids(splits=("train", "val")):
    """uids of the probed images (from the light sidecars, no features loaded), round-robin over datasets."""
    side = []
    for p in sorted(PROBE_DIR.glob("stats_*.pt")):
        side += [dict(uid=s["uid"], ds=s["ds"], split=s["split"]) for s in torch.load(p, weights_only=False) if s["split"] in splits]
    return side

def load_label_dir(d, key=None):
    out = {}
    for p in sorted(d.glob("shard_*.pt")):
        for o in torch.load(p, weights_only=False):
            out[o["uid"]] = o if key is None else o[key]
    return out


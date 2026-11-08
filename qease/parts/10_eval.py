# %% ------------------------------------------------------ 10. EVAL (cal/test)
def load_selectors():
    """Restore trained selectors from disk (lets eval/crc/gen run without re-running 'train')."""
    for v in ("full", "noq"):
        p = CKPT_DIR / f"selector_{v}.pt"
        if v not in SELECTORS and p.exists():
            st = torch.load(p, map_location="cpu")["state"]
            m = Selector(st["gp.0.weight"].shape[0], CFG["D_MODEL"], CFG["N_LAYERS"], CFG["N_HEADS"], v == "full")
            m.load_state_dict(st); SELECTORS[v] = m.eval()
            log(f"restored selector '{v}' from {p.name}")
load_selectors()

def get_sel(r, name):
    """per-device selector copy (cached)"""
    key = (name, r.device)
    if key not in _SEL_CACHE:
        _SEL_CACHE[key] = _clone_to(SELECTORS[name], r.device)
    return _SEL_CACHE[key]
_SEL_CACHE = {}
def _clone_to(m, dev):
    import copy
    return copy.deepcopy(m).to(dev).eval()

@torch.inference_mode()
def selector_order(r, ctx, name):
    """returns (order over groups desc by score, selector model, encoded h, gvalid); the raw scores are stashed in ctx['_sc'][name] (greedy set construction)."""
    sel = get_sel(r, name)
    G, GM = ctx["G"], CFG["GMAX"]
    gf = ctx["gfeat"][None]; gi = torch.arange(G, device=r.device)
    gpos = torch.stack([(gi // ctx["gw"]).float() / max(1, ctx["gh"] - 1), (gi % ctx["gw"]).float() / max(1, ctx["gw"] - 1)], 1)[None]
    gv = torch.ones(1, G, dtype=torch.bool, device=r.device)
    qe = ctx["qemb"][None]; qv = torch.ones(1, qe.shape[1], dtype=torch.bool, device=r.device)
    h, s = sel.encode(gf, gpos, gv, qe, qv)
    ctx.setdefault("_sc", {})[name] = s[0].float()
    return torch.argsort(-s[0]).cpu(), sel, h, gv

def eval_one(r, rec, tick):
    ctx = r.prepare(rec)
    G = ctx["G"]; ks = k_list(G)
    lp_full = r.logp_full(ctx)
    orders = baseline_orders(ctx, stable_hash(rec["uid"]) % 10**6)
    rhat = {}
    for name, sname in [("qease", "full"), ("qease_noq", "noq")]:
        if sname not in SELECTORS: continue
        od, sel, h, gv = selector_order(r, ctx, sname)
        orders[name] = od
        gm = order_to_masks(od, G, ks).to(r.device)
        rhat[name] = torch.sigmoid(sel.risk(h, gv, gm[None])[0].float()).cpu().numpy()   # predicted JS/ln2
    if "qease" in orders and GREEDY_BETA > 0 and "full" in ctx.get("_sc", {}): orders["qease_greedy"] = greedy_order(ctx, ctx["_sc"]["full"], GREEDY_BETA)
    names = [n for n in METHODS if n in orders]
    nk = len(ks)
    ins = torch.cat([order_to_masks(orders[n], G, ks) for n in names], 0)       # INSERTION: keep the first-k ranked groups
    allm = torch.cat([ins, ~ins], 0)                                            # DELETION : drop the first-k ranked groups
    lpm = r.logp_for_token_masks(ctx, r.group_to_token_mask(ctx, allm.to(r.device)))
    jsv = js_div(lp_full, lpm).cpu().numpy().reshape(2, len(names), nk)
    flp = (lpm.argmax(-1) != lp_full.argmax()).cpu().numpy().reshape(2, len(names), nk)
    pf = lp_full.exp()
    j25 = int(np.argmin(np.abs(np.array(CFG["K_GRID"]) - 0.25)))
    out = dict(uid=rec["uid"], ds=rec["ds"], G=G, q=rec["question"][:160],
               geom=dict(H=ctx["H"], W=ctx["W"], win=ctx["win"], gh=ctx["gh"], gw=ctx["gw"]),
               ent=float(-(pf * lp_full).sum()), ptop1=float(pf.max()),
               rhat=rhat, js={}, flip={}, js_del={}, flip_del={}, kfrac={}, sel25={})
    for i, n in enumerate(names):
        out["js"][n], out["flip"][n] = jsv[0, i], flp[0, i]
        out["js_del"][n], out["flip_del"][n] = jsv[1, i], flp[1, i]
        out["kfrac"][n] = kept_token_frac(ctx, ins[i * nk:(i + 1) * nk])
        out["sel25"][n] = ins[i * nk + j25].numpy()
    out["L"], out["n_vis"] = ctx["L"], ctx["n_vis"]
    # ---- external published scorers (plug-in), pre-LLM token-level pruning
    for name, fn in EXTERNAL_SCORERS.items():
        try:
            cv = token_level_curves(r, ctx, lp_full, torch.argsort(-fn(r, ctx).float().flatten().to(r.device)))
            for key in ("js", "flip", "js_del", "flip_del", "kfrac"): out[key][name] = cv[key]
        except Exception as e:
            FAILS.append(("external-scorer", name, repr(e)[:200], traceback.format_exc()))
    # ---- FastV (K, last-token attention at layer K-1, pruning from layer K on) at TOKEN granularity, as published
    if FASTV_OK:
        K = CFG["FASTV_K"]; nv, L = ctx["n_vis"], ctx["L"]; nlay = len(r.decoder.layers)
        h_pref, pe, sc = r.fastv_prefix(ctx, K)
        order = torch.argsort(-sc)
        ns = [int(min(nv, max(1, math.ceil(f * nv)))) for f in CFG["K_GRID"]]
        lp_i, lp_d, eff = [], [], []
        for n in ns:
            keep = torch.ones(L, dtype=torch.bool, device=r.device); keep[ctx["img_idx"]] = False; keep[ctx["img_idx"][order[:n]]] = True
            lp_i.append(r.fastv_logp(ctx, h_pref, pe, keep, K))
            drop = torch.ones(L, dtype=torch.bool, device=r.device); drop[ctx["img_idx"][order[:n]]] = False
            lp_d.append(r.fastv_logp(ctx, h_pref, pe, drop, K))
            eff.append((K + (nlay - K) * (L - nv + n) / L) / nlay)          # effective prefill compute vs full model
        lp_i, lp_d = torch.stack(lp_i), torch.stack(lp_d)
        out["js"]["fastv"], out["js_del"]["fastv"] = js_div(lp_full, lp_i).cpu().numpy(), js_div(lp_full, lp_d).cpu().numpy()
        out["flip"]["fastv"] = (lp_i.argmax(-1) != lp_full.argmax()).cpu().numpy()
        out["flip_del"]["fastv"] = (lp_d.argmax(-1) != lp_full.argmax()).cpu().numpy()
        out["kfrac"]["fastv"] = np.array(ns) / nv
        out["fastv_flops_frac"] = np.array(eff)
        out["fastv_matched"] = matched_group_eval(r, ctx, lp_full, orders, G, eff)       # group methods at FastV's effective compute
    # ---- VisionZip (dominant + merged contextual tokens; pre-LLM; budgets = the same K_GRID fractions)
    if VZ_OK:
        nv, L = ctx["n_vis"], ctx["L"]
        ns = [int(min(nv, max(2, math.ceil(f * nv)))) for f in CFG["K_GRID"]]
        lp_v = torch.stack([r.logp_prefill(ctx, *r.vz_embed(ctx, n)) for n in ns])
        dele = token_level_curves(r, ctx, lp_full, torch.argsort(-ctx["vz"]["score"]))     # deletion = drop top-n by VisionZip score (no merging)
        out["js"]["visionzip"], out["flip"]["visionzip"] = js_div(lp_full, lp_v).cpu().numpy(), (lp_v.argmax(-1) != lp_full.argmax()).cpu().numpy()
        out["js_del"]["visionzip"], out["flip_del"]["visionzip"] = dele["js_del"], dele["flip_del"]
        out["kfrac"]["visionzip"] = np.array(ns) / nv
    # ---- SparseVLM v1.0 at its 4 official budgets (in-LLM, progressive) + compute-matched group methods
    if SV_OK:
        nv, L, nlay = ctx["n_vis"], ctx["L"], len(r.decoder.layers)
        sp = dict(retain=list(CFG["SPARSE_RETAIN"]), js=[], flip=[], kfrac_avg=[], eff=[])
        for ret in sp["retain"]:
            lp_s, st, _ = r.sparsevlm_run(ctx, ret)
            sp["js"].append(float(js_div(lp_full, lp_s[None])[0])); sp["flip"].append(bool(lp_s.argmax() != lp_full.argmax()))
            sp["kfrac_avg"].append(sum(st["vis_per_layer"]) / nlay / nv); sp["eff"].append(sum(st["seq_per_layer"]) / nlay / L)
        sp["matched"] = matched_group_eval(r, ctx, lp_full, orders, G, sp["eff"])
        out["sparse"] = sp
    out = eval_extras(r, ctx, lp_full, orders, out)
    tick(); return out

def eval_shard(r, shard, tick):
    split, sid, recs = shard
    path = EVAL_DIR / f"{split}_{sid:04d}.pt"
    if path.exists(): tick(len(recs)); return 0
    out = []
    for rec in recs:
        if STOP.is_set(): return 0
        try: out.append(eval_one(r, rec, lambda *a: None))
        except Exception as e:
            FAILS.append(("eval", rec["uid"], repr(e)[:200], traceback.format_exc()))
            if "out of memory" in str(e).lower(): torch.cuda.empty_cache()
        tick()
    torch.save(out, path); return len(out)


# ---- greedy (MMR) set construction: tune the redundancy penalty beta on VALIDATION images only (never on cal/test)
GREEDY_FILE = WORK / "results" / "tables" / "greedy_beta.json"
def tune_one(r, rec, tick):
    ctx = r.prepare(rec); G = ctx["G"]; lp_full = r.logp_full(ctx)
    selector_order(r, ctx, "full"); sc = ctx["_sc"]["full"]
    ks = [int(min(G, max(1, math.ceil(f * G)))) for f in (0.125, 0.25)]
    masks = torch.cat([order_to_masks(greedy_order(ctx, sc, b), G, ks) for b in CFG["GREEDY_BETA_GRID"]], 0)
    lpm = r.logp_for_token_masks(ctx, r.group_to_token_mask(ctx, masks.to(r.device)))
    tick(); return js_div(lp_full, lpm).cpu().numpy().reshape(len(CFG["GREEDY_BETA_GRID"]), len(ks))

if GREEDY_FILE.exists():
    GREEDY_BETA = float(json.load(open(GREEDY_FILE))["beta"])
elif "eval" in CFG["STAGES"] and SELECTORS:
    _vrecs = interleave_by_ds(BY_SPLIT["val"], CFG["TUNE_N"], seed=7)
    _res = parallel_map(_vrecs, tune_one, "tune-greedy", total=len(_vrecs))
    if _res:
        _m = np.mean([x.mean(1) for x in _res], 0); GREEDY_BETA = float(CFG["GREEDY_BETA_GRID"][int(np.argmin(_m))])
        json.dump(dict(beta=GREEDY_BETA, grid=CFG["GREEDY_BETA_GRID"], mean_js=[float(x) for x in _m], n=len(_res)), open(GREEDY_FILE, "w"), indent=1)
        log(f"[tune] greedy beta grid {CFG['GREEDY_BETA_GRID']} -> mean JS {np.round(_m, 4).tolist()} -> beta={GREEDY_BETA}")
log(f"greedy beta = {GREEDY_BETA}")

if "eval" in CFG["STAGES"]:
    assert SELECTORS, "train the selector first (stage 'train')"
    for split, n in [("cal", CFG["CAL_N"]), ("test", CFG["TEST_N"])]:
        recs = interleave_by_ds(BY_SPLIT[split], n, seed=3)
        S = 25
        shards = [(split, i, recs[i * S:(i + 1) * S]) for i in range(math.ceil(len(recs) / S))]
        todo = [s for s in shards if not (EVAL_DIR / f"{s[0]}_{s[1]:04d}.pt").exists()]
        log(f"[eval:{split}] {len(recs)} samples, {len(todo)} shards to do")
        if todo:
            parallel_map(todo, eval_shard, f"eval-{split}", total=sum(len(s[2]) for s in todo))


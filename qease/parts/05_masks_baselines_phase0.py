# %% ----------------------------------------------- 6. MASKS, BASELINES, METRICS
RATIOS, MPR = CFG["RATIOS"], CFG["MASKS_PER_RATIO"]
K_MASKS = len(RATIOS) * MPR
METHODS = ["qease", "qease_noq", "qease_greedy", "random", "uniform", "qsim", "fastv", "visionzip"]   # more published baselines are added by eval_extras(); SparseVLM keeps its own record key "sparse"

def matched_group_eval(r, ctx, lp_full, orders, G, eff_targets):
    """Evaluate the group-based methods at the visual-token budget whose PRE-LLM effective prefill compute equals each target
    (L - n_vis + f*n_vis)/L  -> an exact compute-matched comparison against in-LLM pruners (FastV, SparseVLM)."""
    L, nv = ctx["L"], ctx["n_vis"]
    f_star = np.clip((np.asarray(eff_targets, float) * L - (L - nv)) / nv, 1.0 / nv, 1.0)
    ks = [int(min(G, max(1, math.ceil(f * G)))) for f in f_star]
    names = [n for n in ("qease", "qease_noq", "qsim", "uniform", "random") if n in orders]
    masks = torch.cat([order_to_masks(orders[n], G, ks) for n in names], 0)
    lpm = r.logp_for_token_masks(ctx, r.group_to_token_mask(ctx, masks.to(r.device)))
    js = js_div(lp_full, lpm).cpu().numpy().reshape(len(names), len(ks))
    fl = (lpm.argmax(-1) != lp_full.argmax()).cpu().numpy().reshape(len(names), len(ks))
    kf = kept_token_frac(ctx, masks).reshape(len(names), len(ks))
    return {n: dict(js=js[i], flip=fl[i], kfrac=kf[i]) for i, n in enumerate(names)}

# ---- plug-in slot for OTHER published methods (SparseVLM, VisionZip, ...): register a function
#        fn(runner, ctx) -> 1-D tensor of per-VISUAL-TOKEN importance (higher = keep), computed with the authors' official scoring code.
#      They are then evaluated by the same pre-LLM pruning protocol, budgets and metrics as every other method, e.g.
#        EXTERNAL_SCORERS["sparsevlm"] = lambda r, ctx: my_sparsevlm_scores(r.model, ctx["inputs"])
EXTERNAL_SCORERS = globals().get("EXTERNAL_SCORERS", {})

def token_level_curves(r, ctx, lp_full, tok_order):
    """pre-LLM token-level pruning curves for a ranking over visual tokens (insertion = keep first-n, deletion = drop first-n)."""
    nv, L = ctx["n_vis"], ctx["L"]
    ns = [int(min(nv, max(1, math.ceil(f * nv)))) for f in CFG["K_GRID"]]
    ins = torch.zeros(len(ns), nv, dtype=torch.bool)
    for i, n in enumerate(ns): ins[i, tok_order[:n].cpu()] = True
    tk = torch.ones(2 * len(ns), L, dtype=torch.bool, device=r.device)
    tk[: len(ns), ctx["img_idx"]] = ins.to(r.device)
    tk[len(ns):, ctx["img_idx"]] = ~ins.to(r.device)
    lpm = r.logp_for_token_masks(ctx, tk)
    js = js_div(lp_full, lpm).cpu().numpy().reshape(2, len(ns)); fl = (lpm.argmax(-1) != lp_full.argmax()).cpu().numpy().reshape(2, len(ns))
    return dict(js=js[0], flip=fl[0], js_del=js[1], flip_del=fl[1], kfrac=np.array(ns) / nv)

def blob_noise(gh, gw, gen):
    n = torch.rand(1, 1, gh, gw, generator=gen)
    for _ in range(2):
        n = F.avg_pool2d(F.pad(n, (1, 1, 1, 1), mode="replicate"), 3, 1)
    return n.flatten() + 1e-4 * torch.rand(gh * gw, generator=gen)

def make_probe_masks(G, gh, gw, seed):
    """K masks: for each keep-ratio, MPR/2 iid-random subsets and MPR/2 spatially coherent blobs (exact size)."""
    gen = torch.Generator().manual_seed(seed)
    masks, rid = [], []
    for ri, r in enumerate(RATIOS):
        c = int(min(G - 1, max(1, round(r * G))))
        for j in range(MPR):
            score = torch.rand(G, generator=gen) if j < MPR // 2 else blob_noise(gh, gw, gen)
            m = torch.zeros(G, dtype=torch.bool); m[score.topk(c).indices] = True
            masks.append(m); rid.append(ri)
    return torch.stack(masks), torch.tensor(rid)

def k_list(G):
    return [int(min(G, max(1, math.ceil(f * G)))) for f in CFG["K_GRID"]]

def order_to_masks(order, G, ks):
    """Nested sets: first k entries of `order` -> (len(ks), G) bool."""
    m = torch.zeros(len(ks), G, dtype=torch.bool)
    for i, k in enumerate(ks): m[i, order[:k].cpu()] = True
    return m

def baseline_orders(ctx, seed):
    G, gh, gw = ctx["G"], ctx["gh"], ctx["gw"]
    gen = torch.Generator().manual_seed(seed)
    rnd = torch.randperm(G, generator=gen)
    rr, cc = torch.arange(G) // gw, torch.arange(G) % gw
    low_disc = torch.argsort(((rr * 0.7548776662 + cc * 0.5698402910) % 1.0))        # R2 low-discrepancy -> spread-out nested sets
    q = ctx["qemb"].float().mean(0, keepdim=True)
    sim = F.cosine_similarity(ctx["gfeat"], q, dim=-1).cpu()
    return dict(random=rnd, uniform=low_disc, qsim=torch.argsort(-sim))

def kept_token_frac(ctx, gmasks):
    cnt = ctx["gcount"].cpu()
    return ((gmasks.float() * cnt).sum(1) / cnt.sum()).numpy()

def gstack(x, dev): return x.to(dev) if torch.is_tensor(x) else torch.as_tensor(x, device=dev)

# %% ------------------------------------------------------- 7. PHASE 0: HEADROOM
def phase0_one(r, rec, tick):
    ctx = r.prepare(rec)
    G = ctx["G"]
    lp_full = r.logp_full(ctx)
    eye = ~torch.eye(G, dtype=torch.bool)
    lp = r.logp_for_token_masks(ctx, r.group_to_token_mask(ctx, eye.to(r.device)))
    delta = js_div(lp_full, lp).cpu()                                  # necessity of each single group
    orders = baseline_orders(ctx, stable_hash(rec["uid"]) % 10**6)
    orders["oracle1"] = torch.argsort(-delta)
    ks = k_list(G)
    out = dict(uid=rec["uid"], ds=rec["ds"], G=G, kfrac={}, js={}, flip={})
    for name, od in orders.items():
        gm = order_to_masks(od, G, ks)
        lpm = r.logp_for_token_masks(ctx, r.group_to_token_mask(ctx, gm.to(r.device)))
        out["js"][name] = js_div(lp_full, lpm).cpu().numpy()
        out["flip"][name] = (lpm.argmax(-1) != lp_full.argmax()).cpu().numpy()
        out["kfrac"][name] = kept_token_frac(ctx, gm)
    tick(); return out

P0_FILE = WORK / "phase0" / "phase0.pt"
if "phase0" in CFG["STAGES"]:
    if P0_FILE.exists():
        P0 = torch.load(P0_FILE, weights_only=False)
    else:
        recs = interleave_by_ds(BY_SPLIT["val"], CFG["PHASE0_N"], seed=1)
        P0 = parallel_map(recs, phase0_one, "phase0", total=len(recs))
        torch.save(P0, P0_FILE)
    names = ["oracle1", "qsim", "uniform", "random"]
    log(f"PHASE 0 headroom on {len(P0)} samples  (mean JS, nats; lower is better)  - the oracle only ranks single-group ablations")
    hdr = "frac_kept   " + "  ".join(f"{n:>8s}" for n in names)
    print(hdr)
    for j, f in enumerate(CFG["K_GRID"]):
        print(f"{f:9.3f}   " + "  ".join(f"{np.mean([o['js'][n][j] for o in P0]):8.4f}" for n in names))
    lowj = [j for j, f in enumerate(CFG["K_GRID"]) if f <= 0.25]
    base_best = np.mean([min(np.mean([o["js"][n][j] for o in P0]) for n in ["qsim", "uniform", "random"]) for j in lowj])
    orc = np.mean([np.mean([o["js"]["oracle1"][j] for o in P0]) for j in lowj])
    log(f"GATE 1: mean JS at <=25% budget  oracle={orc:.4f} vs best baseline={base_best:.4f} "
        f"-> headroom {'EXISTS' if orc < 0.7*base_best else 'is SMALL (reconsider / see plan fallback)'}")



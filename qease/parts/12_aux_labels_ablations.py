# %% ------------------------------------------------ 12b. EXTRA LABELS + ABLATIONS (supervisor: label / K / mask-type / loss / architecture / window)
UID2REC = {r["uid"]: r for r in POOL}
EXPECT_BASELINES = [b for b, ok in [("fastv", FASTV_OK), ("visionzip", VZ_OK), ("divprune", DIV_OK), ("cdpruner", CDP_OK), ("dart", DART_OK), ("pdrop", PD_OK)] if ok]

if "aux" in CFG["STAGES"] and "_rec" in globals():
    _side = probed_uids(("train", "val"))
    assert _side, "run the 'probe' stage first (aux labels re-use the probed images and their masks)"
    _nmax = max(CFG["AUX_N"], CFG["LOGO_N"], CFG["ATTN_N"])
    _ordered = [UID2REC[p["uid"]] for p in interleave_by_ds(_side, _nmax, seed=21) if p["uid"] in UID2REC]
    _S = CFG["SHARD_SIZE"]
    for _tag, _dir, _n, _fn, _ok in [("aux", AUX_DIR, CFG["AUX_N"], aux_one, True), ("logo", LOGO_DIR, CFG["LOGO_N"], logo_one, True),
                                      ("attn", ATTN_DIR, CFG["ATTN_N"], attn_one, FASTV_OK)]:
        if not _ok:
            log(f"[{_tag}] skipped (FastV self-test failed)"); continue
        _recs = _ordered[:_n]
        _shards = [(i, _recs[i * _S:(i + 1) * _S]) for i in range(math.ceil(len(_recs) / _S))]
        _todo = [s for s in _shards if not (_dir / f"shard_{s[0]:04d}.pt").exists()]
        log(f"[{_tag}] {len(_recs)} images, {len(_todo)} shards to do")
        if _todo: parallel_map(_todo, label_shard(_fn, _dir, _tag), _tag, total=sum(len(s[1]) for s in _todo))

# ---- ablation grid.  Group B = all probed images (reference b_ref); group A = images that have the extra labels (reference a_js); group C = leave-one-group-out subset (reference c_js)
ABL_JOBS = [
    ("b_ref", "full", dict(kind="js")), ("b_noq", "full", dict(kind="js", use_q=False)), ("b_mlp", "full", dict(kind="js", arch="mlp")),
    ("b_norank", "full", dict(kind="js", rank_w=0.0)), ("b_norisk", "full", dict(kind="js", risk_w=0.0)),
    ("b_rank1", "full", dict(kind="js", rank_w=1.0)), ("b_rank2", "full", dict(kind="js", rank_w=2.0)),
    ("b_iid", "full", dict(kind="js", mask_types="iid")), ("b_blob", "full", dict(kind="js", mask_types="blob")),
    ("b_k12", "full", dict(kind="js", mask_types="k12")), ("b_shuf", "full", dict(kind="shuf")),
    ("a_js", "sub", dict(kind="js", labsrc="aux")), ("a_gold", "sub", dict(kind="gold", labsrc="aux")),
    ("a_seq", "sub", dict(kind="seq", labsrc="aux")), ("a_attn", "sub", dict(kind="attn", labsrc="attn")),
    ("c_js", "sub", dict(kind="js", labsrc="logo")), ("c_logo", "sub", dict(kind="logo", labsrc="logo")),
]
ABL_LABEL = dict(b_ref="Q-EASE (JS labels, reference)", b_noq="no question", b_mlp="MLP scorer (no cross-group attention)", b_norank="no rank loss",
                 b_norisk="no risk-head loss", b_rank1="rank weight beta=1", b_rank2="rank weight beta=2", b_iid="iid-random masks only", b_blob="blob masks only",
                 b_k12="K=12 masks (half)", b_shuf="shuffled labels (control)", a_js="JS labels (subset)", a_gold="gold-answer likelihood (accuracy-aware)",
                 a_seq="full-model answer likelihood (answer distillation)", a_attn="FastV-attention labels", c_js="JS labels (LOGO subset)",
                 c_logo="single-group (leave-one-group-out) labels", uniform="Uniform grid", random="Random")

def abl_eval_shard(r, shard, tick):
    sid, recs = shard
    path = ABL_DIR / f"eval_{sid:04d}.pt"
    if path.exists(): tick(len(recs)); return 0
    names = [n for n in SELECTORS if n[:2] in ("b_", "a_", "c_")]
    out = []
    for rec in recs:
        if STOP.is_set(): return 0
        try:
            ctx = r.prepare(rec); G = ctx["G"]; lp_full = r.logp_full(ctx)
            ks = [int(min(G, max(1, math.ceil(f * G)))) for f in CFG["ABL_FRACS"]]
            orders = {n: selector_order(r, ctx, n)[0] for n in names}
            bo = baseline_orders(ctx, stable_hash(rec["uid"]) % 10**6); orders["uniform"], orders["random"] = bo["uniform"], bo["random"]
            ins = torch.cat([order_to_masks(od, G, ks) for od in orders.values()], 0)
            lpm = r.logp_for_token_masks(ctx, r.group_to_token_mask(ctx, ins.to(r.device)))
            js = js_div(lp_full, lpm).cpu().numpy().reshape(len(orders), len(ks)); fl = (lpm.argmax(-1) != lp_full.argmax()).cpu().numpy().reshape(len(orders), len(ks))
            out.append(dict(uid=rec["uid"], ds=rec["ds"], js={n: js[i] for i, n in enumerate(orders)}, flip={n: fl[i] for i, n in enumerate(orders)}))
        except Exception as e:
            FAILS.append(("abl-eval", rec["uid"], repr(e)[:200], traceback.format_exc()))
        tick()
    torch.save(out, path); return len(out)

if "ablate" in CFG["STAGES"]:
    _PROBE = load_probe_records(); assert _PROBE, "probe shards missing"
    _aux, _logo, _attn = load_label_dir(AUX_DIR), load_label_dir(LOGO_DIR, "delta"), load_label_dir(ATTN_DIR, "delta")
    log(f"[ablate] labels available: aux={len(_aux)} logo={len(_logo)} attn={len(_attn)}")
    _sub = [r for r in _PROBE if r["uid"] in _aux]
    _jobs = [j for j in ABL_JOBS if not ((j[2].get("labsrc") == "aux" and not _aux) or (j[2].get("labsrc") == "logo" and not _logo) or (j[2].get("labsrc") == "attn" and not _attn))]
    train_many(_jobs, {"full": _PROBE, "sub": _sub}, {"aux": _aux, "logo": _logo, "attn": _attn})
    del _PROBE, _sub; gc.collect()
    load_extra_selectors(["b_", "a_", "c_"])
    _trecs = interleave_by_ds(BY_SPLIT["test"], CFG["ABL_TEST_N"], seed=31)
    _shards = [(i, _trecs[i * 25:(i + 1) * 25]) for i in range(math.ceil(len(_trecs) / 25))]
    _todo = [s for s in _shards if not (ABL_DIR / f"eval_{s[0]:04d}.pt").exists()]
    log(f"[ablate] evaluating {[n for n in SELECTORS if n[:2] in ('b_', 'a_', 'c_')]} on {len(_trecs)} test images ({len(_todo)} shards)")
    if _todo: parallel_map(_todo, abl_eval_shard, "abl-eval", total=sum(len(s[1]) for s in _todo))

    # ---- window size omega: needs re-grouping -> re-probe a subset with GROUP_WIN=omega (same images, own folders), train, evaluate at matched TOKEN budgets
    if CFG["ABL_OMEGA"] and CFG["ABL_OMEGA"] != CFG["GROUP_WIN"]:
        _om, _odir = CFG["ABL_OMEGA"], ABL_DIR / f"omega{CFG['ABL_OMEGA']}"
        (_odir / "probe").mkdir(parents=True, exist_ok=True); (_odir / "ckpt").mkdir(parents=True, exist_ok=True)
        _uids = [p["uid"] for p in interleave_by_ds(probed_uids(("train", "val")), CFG["ABL_OMEGA_N"], seed=41)]
        _urecs = [UID2REC[u] for u in _uids if u in UID2REC]
        _bak = (PROBE_DIR, CKPT_DIR, CFG["GROUP_WIN"])
        try:
            PROBE_DIR, CKPT_DIR, CFG["GROUP_WIN"] = _odir / "probe", _odir / "ckpt", _om
            _S = CFG["SHARD_SIZE"]; _sh = [(i, _urecs[i * _S:(i + 1) * _S]) for i in range(math.ceil(len(_urecs) / _S))]
            _todo = [s for s in _sh if not (PROBE_DIR / f"shard_{s[0]:04d}.pt").exists()]
            log(f"[ablate-omega] probing {len(_urecs)} images with window {_om}: {len(_todo)} shards")
            if _todo: parallel_map(_todo, probe_shard, "probe-omega", total=sum(len(s[1]) for s in _todo))
            _P3 = load_probe_records()
            _m3 = train_general(f"w{_om}", _P3, "js", DEVICES[0], epochs=CFG["ABL_EPOCHS"]); del _P3
        finally:
            PROBE_DIR, CKPT_DIR, CFG["GROUP_WIN"] = _bak
        _P2 = [r for r in load_probe_records() if r["uid"] in set(_uids)]
        _m2 = train_general("w2ref", _P2, "js", DEVICES[0], epochs=CFG["ABL_EPOCHS"]); del _P2
        SELECTORS[f"w{_om}"], SELECTORS["w2ref"] = _m3, _m2
        _otest = interleave_by_ds(BY_SPLIT["test"], CFG["ABL_OMEGA_TEST"], seed=51)
        for _win, _sn in [(CFG["GROUP_WIN"], "w2ref"), (_om, f"w{_om}")]:
            def _owin(r, shard, tick, _sn=_sn, _win=_win):
                sid, recs = shard; path = _odir / f"eval_win{_win}_{sid:04d}.pt"
                if path.exists(): tick(len(recs)); return 0
                out = []
                for rec in recs:
                    try:
                        ctx = r.prepare(rec); G = ctx["G"]; lp_full = r.logp_full(ctx); ks = k_list(G)
                        od = selector_order(r, ctx, _sn)[0]; ins = order_to_masks(od, G, ks)
                        lpm = r.logp_for_token_masks(ctx, r.group_to_token_mask(ctx, ins.to(r.device)))
                        out.append(dict(uid=rec["uid"], ds=rec["ds"], G=G, js=js_div(lp_full, lpm).cpu().numpy(), flip=(lpm.argmax(-1) != lp_full.argmax()).cpu().numpy(),
                                        kfrac=kept_token_frac(ctx, ins)))
                    except Exception as e:
                        FAILS.append(("abl-omega", rec["uid"], repr(e)[:200], traceback.format_exc()))
                    tick()
                torch.save(out, path); return len(out)
            CFG["GROUP_WIN"] = _win                         # one pass at a time: prepare() reads CFG["GROUP_WIN"], so no two windows may run concurrently
            try:
                _sh = [(i, _otest[i * 25:(i + 1) * 25]) for i in range(math.ceil(len(_otest) / 25))]
                _todo = [s for s in _sh if not (_odir / f"eval_win{_win}_{s[0]:04d}.pt").exists()]
                if _todo: parallel_map(_todo, _owin, f"abl-omega-{_win}", total=sum(len(s[1]) for s in _todo))
            finally:
                CFG["GROUP_WIN"] = _bak[2]
else:
    load_extra_selectors(["b_", "a_", "c_"])



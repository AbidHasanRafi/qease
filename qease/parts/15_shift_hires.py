# %% ----------------------------------------- 15. DISTRIBUTION SHIFT OF THE SELECTOR (leave-one-dataset-out) + HIGH-RESOLUTION STAGE
def shift_eval_shard(r, shard, tick):
    sid, recs = shard
    path = SHIFT_DIR / f"eval_{sid:04d}.pt"
    if path.exists(): tick(len(recs)); return 0
    out = []
    for rec in recs:
        if STOP.is_set(): return 0
        try:
            d = rec["ds"]; ctx = r.prepare(rec); G = ctx["G"]; lp_full = r.logp_full(ctx)
            ks = [int(min(G, max(1, math.ceil(f * G)))) for f in CFG["ABL_FRACS"]]
            orders = {"seen": selector_order(r, ctx, "full")[0], "heldout": selector_order(r, ctx, f"lodo_{d}")[0]}
            bo = baseline_orders(ctx, stable_hash(rec["uid"]) % 10**6); orders["uniform"], orders["random"] = bo["uniform"], bo["random"]
            ins = torch.cat([order_to_masks(od, G, ks) for od in orders.values()], 0)
            lpm = r.logp_for_token_masks(ctx, r.group_to_token_mask(ctx, ins.to(r.device)))
            js = js_div(lp_full, lpm).cpu().numpy().reshape(len(orders), len(ks)); fl = (lpm.argmax(-1) != lp_full.argmax()).cpu().numpy().reshape(len(orders), len(ks))
            out.append(dict(uid=rec["uid"], ds=d, js={n: js[i] for i, n in enumerate(orders)}, flip={n: fl[i] for i, n in enumerate(orders)}))
        except Exception as e:
            FAILS.append(("shift-eval", rec["uid"], repr(e)[:200], traceback.format_exc()))
        tick()
    torch.save(out, path); return len(out)

if "shift" in CFG["STAGES"]:
    _PROBE = load_probe_records(); assert _PROBE, "probe shards missing"
    _dsl = [d for d in DS_ORDER if any(r["ds"] == d for r in _PROBE)]
    train_many([(f"lodo_{d}", f"x_{d}", dict(kind="js", epochs=CFG["SHIFT_EPOCHS"])) for d in _dsl], {f"x_{d}": [r for r in _PROBE if r["ds"] != d] for d in _dsl}, {})
    del _PROBE; gc.collect()
    load_extra_selectors(["lodo_"])
    _trecs = []
    for _d in _dsl: _trecs += [x for x in BY_SPLIT["test"] if x["ds"] == _d][: CFG["SHIFT_TEST_PER_DS"]]
    _shards = [(i, _trecs[i * 25:(i + 1) * 25]) for i in range(math.ceil(len(_trecs) / 25))]
    _todo = [s for s in _shards if not (SHIFT_DIR / f"eval_{s[0]:04d}.pt").exists()]
    log(f"[shift] evaluating leave-one-dataset-out selectors on {len(_trecs)} held-out test images ({len(_todo)} shards)")
    if _todo: parallel_map(_todo, shift_eval_shard, "shift-eval", total=sum(len(s[1]) for s in _todo))
else:
    load_extra_selectors(["lodo_"])

if "hires" in CFG["STAGES"]:
    # same selectors / same policy, but images re-encoded with 4x the token caps (selector was trained at the low caps -> this is also a resolution-shift test)
    _saved_tok, _saved_eval = dict(CFG["TOKENS_PER_DS"]), EVAL_DIR
    (HIRES_DIR / "eval").mkdir(parents=True, exist_ok=True); (HIRES_DIR / "dec").mkdir(parents=True, exist_ok=True)
    try:
        CFG["TOKENS_PER_DS"].clear(); CFG["TOKENS_PER_DS"].update(CFG["HIRES_TOKENS"])
        _hrecs = interleave_by_ds(BY_SPLIT["test"], CFG["HIRES_N"], seed=71)
        EVAL_DIR = HIRES_DIR / "eval"
        _shards = [("test", i, _hrecs[i * 10:(i + 1) * 10]) for i in range(math.ceil(len(_hrecs) / 10))]
        _todo = [s for s in _shards if not (EVAL_DIR / f"{s[0]}_{s[1]:04d}.pt").exists()]
        log(f"[hires] first-token fidelity at {CFG['HIRES_TOKENS']} tokens: {len(_hrecs)} images, {len(_todo)} shards")
        if _todo: parallel_map(_todo, eval_shard, "eval-hires", total=sum(len(s[2]) for s in _todo))
        _items = [(x, HIRES_DIR / "dec", False) for x in _hrecs]
        _todo = [it for it in _items if not (it[1] / f"{it[0]['uid']}.json").exists()]
        log(f"[hires] decoded cascade records: {len(_todo)} to do")
        if _todo: parallel_map(_todo, dec_one, "dec-hires", total=len(_todo))
    finally:
        CFG["TOKENS_PER_DS"].clear(); CFG["TOKENS_PER_DS"].update(_saved_tok); EVAL_DIR = _saved_eval


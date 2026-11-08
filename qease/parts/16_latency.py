# %% ------------------------------------------------ 14. REAL LATENCY / FLOPs / MEMORY
def _sync(dev):
    if dev.startswith("cuda"): torch.cuda.synchronize(dev)

if "latency" in CFG["STAGES"]:
    _r = RUNNERS[0]; _dev = _r.device
    _recs = interleave_by_ds(BY_SPLIT["test"], CFG["LAT_N"], seed=5)
    _rows = []
    with torch.inference_mode():
        _r.prepare(_recs[0])                                                              # warm-up (CUDA kernels, allocator)
        for _rec in _recs:
            try:
                nt = CFG["TOKENS_PER_DS"].get(_rec["ds"], CFG["TOKENS_PER_DS"]["default"])
                img = smart_resize(Image.open(POOL_DIR / _rec["img"]).convert("RGB"), nt)
                t0 = time.perf_counter(); _r.build_inputs(img, _rec["question"]); t_pre = 1e3 * (time.perf_counter() - t0)
                _sync(_dev); t0 = time.perf_counter(); ctx = _r.prepare(_rec); _sync(_dev)
                enc_ms = max(0.0, 1e3 * (time.perf_counter() - t0) - t_pre)               # vision tower + projector (+ hook capture)
                if "full" in SELECTORS:
                    _sync(_dev); t0 = time.perf_counter(); od = selector_order(_r, ctx, "full")[0]; _sync(_dev)
                    sel_ms = 1e3 * (time.perf_counter() - t0)
                else:
                    od, sel_ms = baseline_orders(ctx, 0)["uniform"], 0.0
                G = ctx["G"]
                for f in [1.0, 0.75, 0.5, 0.25, 0.125]:
                    k = max(1, math.ceil(f * G)); gk = order_to_masks(od, G, [k])[0].to(_dev)
                    for _ in range(2): _r.prefill_pruned(ctx, gk)                         # warm-up
                    if _dev.startswith("cuda"):
                        torch.cuda.reset_peak_memory_stats(_dev); base = torch.cuda.memory_allocated(_dev)
                    _sync(_dev); t0 = time.perf_counter()
                    for _ in range(5): _r.prefill_pruned(ctx, gk)
                    _sync(_dev); pf_ms = 1e3 * (time.perf_counter() - t0) / 5
                    mem = (torch.cuda.max_memory_allocated(_dev) - base) / 2**20 if _dev.startswith("cuda") else np.nan
                    ntok = int(_r.group_to_token_mask(ctx, gk[None])[0].sum())
                    _rows.append(dict(uid=_rec["uid"], ds=_rec["ds"], frac=f, groups=k, G=G, seq_tokens=ntok, vis_tokens_total=ctx["n_vis"],
                                      enc_ms=enc_ms, selector_ms=(sel_ms if f < 1.0 else 0.0), prefill_ms=pf_ms,
                                      prefill_gflops=2 * _r.llm_params * ntok / 1e9, peak_extra_mem_mib=mem))
            except Exception as e:
                FAILS.append(("latency", _rec["uid"], repr(e)[:200], traceback.format_exc()))
    pd.DataFrame(_rows).to_csv(TAB / "raw_latency.csv", index=False)
    log(f"[latency] {len(_rows)} measurements on {_dev} -> raw_latency.csv")


# %% ---------------------------------- 16. REAL END-TO-END WALL-CLOCK OF THE CASCADE (not a model: the cascade is actually executed)
def feat_row(frac, lp, lp0, rhat, ent0, top10):
    st = first_token_stats(lp)
    return [frac, st["ent"], st["top1"], st["margin"], rhat, float(js_div(lp0, lp[None])[0]), float(lp.argmax() == lp0.argmax()), ent0, top10]

@torch.inference_mode()
def cascade_live(r, ctx, CS, key):
    """execute the escalate-or-abstain policy for one question. returns (final stage frac, abstained)."""
    pol = CS["pol"][key]; path = path_idx(CFG["CASC_PATHS"][key[0]]); grid = CFG["DEC_GRID"]; G = ctx["G"]
    od, sel, h, gv = selector_order(r, ctx, "full")
    ks = [int(min(G, max(1, math.ceil(f * G)))) for f in DEC_PRUNED]
    gm = order_to_masks(od, G, ks)
    rhat = torch.sigmoid(sel.risk(h, gv, gm.to(r.device)[None])[0].float()).cpu().numpy()
    lp0 = None; e0 = t0 = 0.0
    for t, j in enumerate(path):
        f = grid[j]
        gk = torch.ones(G, dtype=torch.bool, device=r.device) if f >= 1.0 else gm[j].to(r.device)
        lp = r.prefill_pruned(ctx, gk)
        if lp0 is None: lp0 = lp; st0 = first_token_stats(lp); e0, t0 = st0["ent"], st0["top1"]
        x = np.array([feat_row(f, lp, lp0, float(rhat[j]) if f < 1.0 else 0.0, e0, t0)], np.float32)
        u = float(CS["H"]["fail"].predict_proba(x)[0, 1])
        if u >= pol["mu"]: return f, True
        if f >= 1.0: return f, False
        if float(CS["H"]["harm"].predict_proba(x)[0, 1]) <= pol["lam"]: return f, False
    return grid[path[-1]], False

if "latency" in CFG["STAGES"] and CASC_FILE.exists() and RUNNERS:
    _CS = get_casc(); _key = next((k for k in [("4-stage", 0.05, True), ("3-stage", 0.05, True)] if k in _CS["pol"]), next(iter(_CS["pol"])))
    _r = RUNNERS[0]; _dev = _r.device; _rows = []
    _crecs = interleave_by_ds(BY_SPLIT["test"], CFG["CASC_LAT_N"], seed=81)
    with torch.inference_mode():
        _r.prepare(_crecs[0])
        for _rec in _crecs:
            try:
                ctx = _r.prepare(_rec); G = ctx["G"]; gk_full = torch.ones(G, dtype=torch.bool, device=_dev)
                for _ in range(2): _r.prefill_pruned(ctx, gk_full); cascade_live(_r, ctx, _CS, _key)           # warm-up
                _sync(_dev); t0 = time.perf_counter()
                for _ in range(3): _r.prefill_pruned(ctx, gk_full)
                _sync(_dev); t_full = 1e3 * (time.perf_counter() - t0) / 3
                t0 = time.perf_counter()
                for _ in range(3): fin = cascade_live(_r, ctx, _CS, _key)
                _sync(_dev); t_casc = 1e3 * (time.perf_counter() - t0) / 3
                _sync(_dev); t0 = time.perf_counter(); _r.prepare(_rec); _sync(_dev); t_enc = 1e3 * (time.perf_counter() - t0)
                _rows.append(dict(uid=_rec["uid"], ds=_rec["ds"], policy="|".join(map(str, _key)), enc_ms=t_enc, full_prefill_ms=t_full, cascade_ms=t_casc,
                                  final_frac=fin[0], abstained=bool(fin[1]), e2e_full_ms=t_enc + t_full, e2e_cascade_ms=t_enc + t_casc))
            except Exception as e:
                FAILS.append(("casc-latency", _rec["uid"], repr(e)[:200], traceback.format_exc()))
    pd.DataFrame(_rows).to_csv(TAB / "raw_cascade_latency.csv", index=False)
    log(f"[latency] {len(_rows)} real cascade executions ({_key}) -> raw_cascade_latency.csv")


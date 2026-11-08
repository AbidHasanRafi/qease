# %% --------------------------------------------------- 15. ANALYSIS (all tables + figures)
def run_block(name, fn):
    t0 = time.time()
    try:
        fn(); log(f"[analyze] {name}: ok ({time.time()-t0:.1f}s)")
    except Exception as e:
        FAILS.append(("analyze", name, repr(e)[:200], traceback.format_exc()))
        log(f"[analyze] {name}: FAILED -> {repr(e)[:200]}   (traceback: FAILS[-1][3]; other blocks continue)")

def sty(m):
    if m not in METHOD_STYLE:                     # methods registered through EXTERNAL_SCORERS get a style automatically
        METHOD_STYLE[m] = dict(color=_EXTRA_COLORS[len(METHOD_STYLE) % len(_EXTRA_COLORS)], marker="X", ls="-", label=m)
    return METHOD_STYLE[m]

def tidy_curve(fr, mu, lo, hi, **meta):
    return pd.DataFrame(dict(frac=fr, mean=mu, lo=lo, hi=hi, **meta))

def draw_curve(ax, d, m, band=True, **kw):
    s = sty(m)
    ax.plot(d["frac"], d["mean"], color=s["color"], marker=s["marker"], ls=s["ls"], label=s["label"], **kw)
    if band: ax.fill_between(d["frac"], d["lo"], d["hi"], color=s["color"], alpha=0.15, lw=0)

# ---- state shared by the analysis blocks
CAL = TEST = AV = FR = None
J25 = J12 = 0
DSL = []
REC_BY_UID = {r["uid"]: r for r in POOL}
BOOTM = {}

def prepare_analysis_state():
    global CAL, TEST, AV, FR, J25, J12, DSL
    CAL, TEST = load_eval("cal"), load_eval("test")
    assert len(CAL) > 20 and len(TEST) > 20, "need eval records: run stage 'eval' first"
    keys = set.intersection(*[set(r["js"].keys()) for r in CAL + TEST])
    AV = [m for m in METHODS if m in keys] + sorted(k for k in keys if k not in METHODS)       # known methods first, plug-ins after
    BASELINES[:] = [b for b in ALL_BASELINES + [k for k in AV if k not in METHODS] if b in AV]
    FR = np.array(CFG["K_GRID"]); J25 = int(np.argmin(abs(FR - 0.25))); J12 = int(np.argmin(abs(FR - 0.125)))
    DSL = [d for d in DS_ORDER if any(r["ds"] == d for r in TEST)]
    if not LAMBDAS: load_lambdas()
    log(f"[analyze] cal={len(CAL)} test={len(TEST)} methods={AV} datasets={DSL}")

# ============================== A0. run configuration & environment
def an_run_config():
    import platform
    info = dict(tag=TAG, config={k: (v if isinstance(v, (int, float, str, list, dict, type(None))) else str(v)) for k, v in CFG.items()},
                python=sys.version.split()[0], platform=platform.platform(), torch=torch.__version__, transformers=transformers.__version__,
                datasets=hfds.__version__, numpy=np.__version__, pandas=pd.__version__, matplotlib=matplotlib.__version__,
                devices=DEVICES, gpu_names=[torch.cuda.get_device_name(i) for i in range(N_GPU)], dtype=str(DTYPE),
                n_pool={d: sum(1 for r in POOL if r["ds"] == d) for d in DS_ORDER},
                n_cal=len(CAL), n_test=len(TEST), n_failures=len(FAILS), finished=time.strftime("%Y-%m-%d %H:%M:%S"),
                selector_params_M={k: round(sum(p.numel() for p in v.parameters()) / 1e6, 3) for k, v in SELECTORS.items()},
                llm_decoder_params_B=(round(RUNNERS[0].llm_params / 1e9, 3) if RUNNERS else None),
                splits="by image hash: train 62% / val 8% / cal 15% / test 15% (no image shared across splits)",
                note_guarantee="CRC guarantee is marginal over (calibration set, test point) under exchangeability.")
    (WORK / "results" / "run_config.json").write_text(json.dumps(info, indent=1, default=str))
    MANIFEST.append(("meta", "results/run_config.json", "full configuration, library versions, GPU names, sample counts"))
    if FAILS:
        save_table(pd.DataFrame([dict(stage=f[0], item=f[1], error=f[2], traceback_tail=f[3][-400:]) for f in FAILS]),
                   "failures_log", "per-sample / per-block failures logged during the run", latex=False)

# ============================== A1. dataset + probing-label statistics
def an_probe_stats():
    side = []
    for p in sorted((PROBE_DIR).glob("stats_*.pt")): side += torch.load(p, weights_only=False)
    if not side: log("   no probe sidecars found - skipping"); return
    rows = []
    for s in side:
        for j in range(len(s["js"])):
            ri, within = int(s["rid"][j]), j % MPR
            rows.append(dict(ds=s["ds"], uid=s["uid"], ratio=RATIOS[ri], mask_type="iid-random" if within < MPR // 2 else "spatial-blob",
                             js=float(s["js"][j]), flip=float(s["flip"][j]), G=s["G"], ent=s["ent"]))
    D = pd.DataFrame(rows)
    # table 1: dataset statistics
    pool = pd.DataFrame(POOL)
    t1 = pool.groupby("ds").agg(n_pool=("uid", "count"), n_train=("split", lambda x: (x == "train").sum()), n_val=("split", lambda x: (x == "val").sum()),
                                n_cal=("split", lambda x: (x == "cal").sum()), n_test=("split", lambda x: (x == "test").sum()),
                                mean_W=("W", "mean"), mean_H=("H", "mean")).reset_index()
    pr = D.drop_duplicates("uid").groupby("ds").agg(n_probed=("uid", "count"), mean_groups=("G", "mean"), mean_answer_entropy=("ent", "mean")).reset_index()
    t1 = t1.merge(pr, on="ds", how="left")
    t1["tokens_budget"] = t1["ds"].map(lambda d: CFG["TOKENS_PER_DS"].get(d, CFG["TOKENS_PER_DS"]["default"]))
    save_table(t1, "dataset_statistics", "data pool, split sizes, image sizes, probed samples, mean #groups, mean full-model answer entropy")
    # table 2: label statistics
    t2 = D.groupby(["ds", "ratio", "mask_type"]).agg(n=("js", "size"), js_mean=("js", "mean"), js_std=("js", "std"),
                                                       flip_rate=("flip", "mean")).reset_index()
    save_table(t2, "probe_label_statistics", "mean JS divergence / answer-flip rate of random & blob masks by keep-ratio and dataset (training signal)")
    # figure
    rows_f = []
    for (rt, mt), g in D.groupby(["ratio", "mask_type"]):
        mu, lo, hi, _ = boot_curve(g["js"].values, B=500)
        rows_f.append(dict(panel="a", series=mt, frac=rt, mean=mu[0], lo=lo[0], hi=hi[0]))
    for (rt, ds_), g in D.groupby(["ratio", "ds"]):
        mu, lo, hi, _ = boot_curve(g["js"].values, B=500)
        rows_f.append(dict(panel="b", series=ds_, frac=rt, mean=mu[0], lo=lo[0], hi=hi[0]))
    FD = pd.DataFrame(rows_f)
    fig, ax = plt.subplots(1, 2, figsize=(COL2, 2.4))
    for mt, c, mk in [("iid-random", "#0072B2", "o"), ("spatial-blob", "#D55E00", "s")]:
        d = FD[(FD.panel == "a") & (FD.series == mt)].sort_values("frac")
        ax[0].plot(d.frac, d["mean"], color=c, marker=mk, label=mt); ax[0].fill_between(d.frac, d.lo, d.hi, color=c, alpha=0.15, lw=0)
    for ds_ in DS_ORDER:
        d = FD[(FD.panel == "b") & (FD.series == ds_)].sort_values("frac")
        if len(d): ax[1].plot(d.frac, d["mean"], color=DS_COLOR[ds_], marker="o", ms=2.5, label=DS_LABEL[ds_])
    for a, t in zip(ax, ["(a) mask type", "(b) dataset"]):
        a.set_xlabel("fraction of visual groups kept"); a.set_ylabel("mean JS to full model (nats)"); a.set_yscale("log"); a.legend(); panel_label(a, t[:3])
    save_fig(fig, "fig_probe_labels", FD, "counterfactual probe labels: answer-distribution shift vs fraction of groups kept (training signal exists)")

# ============================== A2. phase-0 headroom gate
def an_phase0():
    f = WORK / "phase0" / "phase0.pt"
    if not f.exists(): log("   phase0 file missing - skipping"); return
    P0 = torch.load(f, weights_only=False)
    names = [n for n in ["oracle1", "qsim", "uniform", "random"] if all(n in o["js"] for o in P0)]
    rows, fdat = [], []
    fig, ax = plt.subplots(1, 2, figsize=(COL2, 2.5))
    for n in names:
        for key, a, lab in [("js", ax[0], "mean JS to full model (nats)"), ("flip", ax[1], "answer-flip rate")]:
            X = np.stack([o[key][n] for o in P0]).astype(float); mu, lo, hi, _ = boot_curve(X)
            d = tidy_curve(FR, mu, lo, hi, panel="a" if key == "js" else "b", metric=key, method=n, n=len(P0))
            fdat.append(d); draw_curve(a, d, n)
            for j in range(len(FR)):
                rows.append(dict(method=sty(n)["label"], metric=key, frac=FR[j], mean=mu[j], lo=lo[j], hi=hi[j], n=len(P0)))
    for a, lab in zip(ax, ["(a)", "(b)"]):
        frac_ticks(a, FR); a.set_xlabel("fraction of groups kept"); panel_label(a, lab)
    ax[0].set_yscale("symlog", linthresh=1e-3); ax[0].set_ylabel("mean JS to full model (nats)"); ax[1].set_ylabel("answer-flip rate"); ax[0].legend()
    save_table(pd.DataFrame(rows), "phase0_headroom", "Gate 1: counterfactual single-ablation oracle vs cheap baselines (validation images)")
    save_fig(fig, "fig_phase0_headroom", pd.concat(fdat), "Gate 1 headroom: oracle vs baselines")
    lo_b = [j for j, f_ in enumerate(FR) if f_ <= 0.25]
    orc = np.mean([[o["js"]["oracle1"][j] for o in P0] for j in lo_b])
    bb = np.mean([min(np.mean([o["js"][n][j] for o in P0]) for n in BASELINES if n in names) for j in lo_b])
    log(f"   GATE 1: mean JS at <=25% budget  oracle={orc:.4f}  best baseline={bb:.4f}  ratio={orc / max(bb, 1e-9):.2f}")

# ============================== A3. selector training curves
def an_hist():
    dfs = []
    for v in ("full", "noq"):
        p = CKPT_DIR / f"history_{v}.csv"
        if p.exists(): dfs.append(pd.read_csv(p))
    if not dfs: log("   no training history - skipping"); return
    H = pd.concat(dfs)
    fig, ax = plt.subplots(1, 3, figsize=(COL2, 2.2))
    for v, c, ls in [("full", "#D55E00", "-"), ("noq", "#CC79A7", "--")]:
        h = H[H.variant == v]
        if not len(h): continue
        lab = "Q-EASE" if v == "full" else "w/o question"
        ax[0].plot(h.epoch, h.train_loss, color=c, ls=ls, label=f"{lab} train"); ax[0].plot(h.epoch, h.val_loss, color=c, ls=":", label=f"{lab} val")
        ax[1].plot(h.epoch, h.val_pair_acc, color=c, ls=ls, label=lab); ax[2].plot(h.epoch, h.val_risk_spearman, color=c, ls=ls, label=lab)
    for a, yl, t in zip(ax, ["loss", "val. pairwise ranking acc.", "val. Spearman (risk head)"], ["(a)", "(b)", "(c)"]):
        a.set_xlabel("epoch"); a.set_ylabel(yl); a.legend(); panel_label(a, t)
    save_table(H, "selector_training_history", "per-epoch training/validation metrics of both selector variants")
    save_fig(fig, "fig_training_curves", H, "selector training curves")

# ============================== A4. fidelity vs budget, significance, faithfulness, matched-fidelity savings
def an_fidelity():
    metrics = [("js", "mean JS to full model (nats)"), ("flip", "answer-flip rate"),
               ("js_del", "JS after DELETING top-$k$ groups"), ("flip_del", "flip rate after DELETING top-$k$")]
    rows = []
    for m in AV:
        kf = arr(TEST, "kfrac", m).mean(0)
        for key, _ in metrics:
            X = arr(TEST, key, m); mu, lo, hi, M = boot_curve(X); BOOTM[(m, key)] = M
            for j in range(len(FR)):
                rows.append(dict(method=m, metric=key, frac=FR[j], kept_token_frac=kf[j], mean=mu[j], lo=lo[j], hi=hi[j], n=len(X)))
    FID = pd.DataFrame(rows)
    save_table(FID.assign(method=FID.method.map(lambda x: sty(x)["label"])), "fidelity_by_budget",
               "main result: fidelity to the full model vs visual-token budget, 95% bootstrap CIs (test split)")
    # --- Fig: fidelity curves
    fig, ax = plt.subplots(1, 2, figsize=(COL2, 2.6))
    for m in AV:
        draw_curve(ax[0], FID[(FID.method == m) & (FID.metric == "js")], m); draw_curve(ax[1], FID[(FID.method == m) & (FID.metric == "flip")], m)
    for a, yl, t in zip(ax, ["mean JS to full model (nats)", "answer-flip rate (first token)"], ["(a)", "(b)"]):
        frac_ticks(a, FR); a.set_xlabel("fraction of visual groups kept"); a.set_ylabel(yl); panel_label(a, t)
    ax[0].set_yscale("symlog", linthresh=1e-3); ax[0].legend(loc="lower left", fontsize=5.5)   # lower-left is empty (curves sit at the top)
    save_fig(fig, "fig_fidelity_curves", FID[FID.metric.isin(["js", "flip"])], "fidelity vs budget (test), shaded = 95% bootstrap CI")
    # --- Fig: insertion / deletion (evidence faithfulness)
    fig, ax = plt.subplots(1, 2, figsize=(COL2, 2.6))
    for m in AV:
        draw_curve(ax[0], FID[(FID.method == m) & (FID.metric == "js")], m); draw_curve(ax[1], FID[(FID.method == m) & (FID.metric == "js_del")], m)
    for a, yl, t in zip(ax, ["JS: keep top-$k$ (insertion, lower=better)", "JS: drop top-$k$ (deletion, higher=better)"], ["(a)", "(b)"]):
        frac_ticks(a, FR); a.set_xlabel("fraction of groups inserted / deleted"); a.set_ylabel(yl); panel_label(a, t)
    ax[0].set_yscale("symlog", linthresh=1e-3); ax[0].legend(loc="upper right")
    save_fig(fig, "fig_insertion_deletion", FID[FID.metric.isin(["js", "js_del"])], "insertion / deletion faithfulness curves")
    # --- AUC table
    rows = []
    for m in AV:
        for key, nm in [("js", "insertion AUC (JS, lower=better)"), ("flip", "insertion AUC (flip, lower=better)"),
                        ("js_del", "deletion AUC (JS, higher=better)"), ("flip_del", "deletion AUC (flip, higher=better)")]:
            a_ = trap_rows(arr(TEST, key, m), FR) / (FR[-1] - FR[0]); mu, lo, hi, _ = boot_curve(a_)
            rows.append(dict(method=sty(m)["label"], quantity=nm, auc=mu[0], lo=lo[0], hi=hi[0]))
    save_table(pd.DataFrame(rows), "auc_insertion_deletion", "normalised area under insertion/deletion curves with 95% CI")
    # --- paired significance: Q-EASE vs every other method (Holm-corrected over the whole family)
    rows = []
    for b in [m for m in AV if m != "qease"]:
        for key in ("js", "flip"):
            mu, lo, hi, p, dz = paired_test(arr(TEST, key, "qease"), arr(TEST, key, b))
            Xa, Xb = arr(TEST, key, "qease"), arr(TEST, key, b)
            for j in range(len(FR)):
                rows.append(dict(baseline=sty(b)["label"], metric=key, frac=FR[j], mean_diff_qease_minus_baseline=mu[j], lo=lo[j], hi=hi[j],
                                 p_boot=p[j], effect_size_dz=dz[j], qease_better_rate=float((Xa[:, j] < Xb[:, j]).mean()),
                                 tie_rate=float((Xa[:, j] == Xb[:, j]).mean()), n=len(Xa)))
    T = pd.DataFrame(rows); T["p_holm"] = holm(T["p_boot"].values)
    T["significant_0.05_holm"] = (T["p_holm"] < 0.05) & (T["mean_diff_qease_minus_baseline"] < 0)
    save_table(T, "paired_tests_vs_qease", "paired bootstrap tests (negative diff = Q-EASE closer to the full model); Holm-corrected over all comparisons")
    # --- matched-fidelity token savings
    rows = []
    targets = [("flip", [0.01, 0.02, 0.05, 0.10]), ("js", [0.005, 0.01, 0.02, 0.05])]
    for key, ts in targets:
        for t in ts:
            fb = {m: np.array([frac_at_target(FR, row, t) for row in BOOTM[(m, key)]]) for m in AV}
            best_base = np.nanmin(np.stack([fb[b] for b in BASELINES if b in fb]), axis=0) if any(b in fb for b in BASELINES) else None
            for m in AV:
                pt = frac_at_target(FR, FID[(FID.method == m) & (FID.metric == key)].sort_values("frac")["mean"].values, t)
                lo, hi = (np.nanpercentile(fb[m], [2.5, 97.5]) if np.isfinite(fb[m]).any() else (np.nan, np.nan))
                sav = (1 - fb[m] / best_base) if (best_base is not None and m.startswith("qease")) else None
                rows.append(dict(metric=("mean flip rate" if key == "flip" else "mean JS (nats)"), target=t, method=sty(m)["label"], frac_needed=pt,
                                 lo=lo, hi=hi, saving_vs_best_baseline_pct=(100 * np.nanmedian(sav) if sav is not None and np.isfinite(sav).any() else np.nan),
                                 saving_lo=(100 * np.nanpercentile(sav, 2.5) if sav is not None and np.isfinite(sav).any() else np.nan),
                                 saving_hi=(100 * np.nanpercentile(sav, 97.5) if sav is not None and np.isfinite(sav).any() else np.nan)))
    save_table(pd.DataFrame(rows), "tokens_at_matched_fidelity", "visual-token fraction needed to reach a target mean fidelity (lower is better); bootstrap CIs")
    print(pd.DataFrame(rows).round(3).to_string(index=False))

# ============================== A5. per-dataset / per-difficulty breakdown
def an_dataset():
    rows, fdat = [], []
    for d in DSL:
        sub = [r for r in TEST if r["ds"] == d]
        for m in AV:
            for j, tag in [(J12, "12.5%"), (J25, "25%")]:
                for key in ("js", "flip"):
                    mu, lo, hi, _ = boot_curve(arr(sub, key, m)[:, j])
                    row = dict(dataset=DS_LABEL[d], n=len(sub), method=sty(m)["label"], method_id=m, budget=tag, metric=key, mean=mu[0], lo=lo[0], hi=hi[0])
                    if m != "qease":
                        xa, xb = arr(sub, key, "qease")[:, j], arr(sub, key, m)[:, j]
                        row["qease_win_rate"], row["tie_rate"] = float((xa < xb).mean()), float((xa == xb).mean())
                        _, _, _, p, _ = paired_test(xa, xb); row["p_boot_vs_qease"] = float(p[0])
                    rows.append(row)
    T = pd.DataFrame(rows); save_table(T.drop(columns="method_id"), "per_dataset_fidelity", "fidelity per dataset at 12.5% and 25% budgets with CIs, win-rates and paired p-values of Q-EASE")
    fig, ax = plt.subplots(2, 1, figsize=(COL2, 4.2))
    for a, key, yl in zip(ax, ["js", "flip"], ["mean JS to full model (nats)", "answer-flip rate"]):
        W = 0.8 / len(AV)
        for i, m in enumerate(AV):
            d = T[(T.method_id == m) & (T.budget == "25%") & (T.metric == key)].set_index("dataset").reindex([DS_LABEL[x] for x in DSL])
            x = np.arange(len(DSL)) + (i - (len(AV) - 1) / 2) * W
            a.bar(x, d["mean"], W, yerr=[d["mean"] - d["lo"], d["hi"] - d["mean"]], color=sty(m)["color"], edgecolor="k", linewidth=0.4, capsize=1.2,
                  error_kw=dict(lw=0.6), label=sty(m)["label"])
        a.set_xticks(np.arange(len(DSL))); a.set_xticklabels([DS_LABEL[x] for x in DSL]); a.set_ylabel(yl)
    ax[0].legend(ncol=len(AV), loc="upper center", bbox_to_anchor=(0.5, 1.28), fontsize=6); panel_label(ax[0], "(a)"); panel_label(ax[1], "(b)")
    save_fig(fig, "fig_per_dataset_25pct", T[T.budget == "25%"].drop(columns="method_id"), "per-dataset fidelity at 25% budget (95% CI)")

# ============================== A6. calibrated budgets: main table + headline figures
def an_crc_main():
    rows = []
    for kind, eps in LOSS_DEFS:
        for alpha in CFG["ALPHAS"]:
            for m in AV:
                pols = ["fixed"] + ((["adaptive", "mondrian"] + (["naive"] if kind == "js" else [])) if m.startswith("qease") else [])
                for pol in pols:
                    kidx = mondrian_idx(m, kind, eps, alpha, CAL, TEST) if pol == "mondrian" else make_policy_idx(m, kind, eps, alpha, CAL, TEST, pol)
                    lv, fv = policy_eval(TEST, m, kidx, kind, eps)
                    s = summarize_policy(lv, fv)
                    rows.append(dict(loss=loss_tag(kind, eps), alpha=alpha, method=sty(m)["label"], method_id=m, policy=POLICY_LABEL[pol], policy_id=pol,
                                     n_cal=len(CAL), **s, risk_leq_alpha=bool(s["test_risk"] <= alpha),
                                     risk_ci_covers_alpha=bool(s["risk_lo"] <= alpha)))
    T = pd.DataFrame(rows)
    sav = []
    for (lt, al), g in T.groupby(["loss", "alpha"]):
        base = g[(g.policy_id == "fixed") & g.method_id.isin(BASELINES) & g.risk_leq_alpha]
        best = base.kept_frac.min() if len(base) else np.nan
        sav += [(i, 100 * (1 - T.loc[i, "kept_frac"] / best) if np.isfinite(best) else np.nan) for i in g.index]
    T["saving_vs_best_valid_fixed_baseline_pct"] = pd.Series({i: v for i, v in sav})
    save_table(T.drop(columns=["method_id", "policy_id"]), "crc_main", "calibrated budgets: realised test risk and kept tokens for fixed-k / adaptive / per-dataset / uncalibrated policies (cal->test split)")
    for lt in T["loss"].unique():
        print(f"\n=== {lt}  (kept_frac | test_risk) ===")
        print(T[T["loss"] == lt].pivot_table(index=["method", "policy"], columns="alpha", values=["kept_frac", "test_risk"]).round(3).to_string())
    # ---------------- headline figures: tokens vs alpha, realised risk vs alpha
    for kind, eps in [("flip", None), ("js", EPS0)]:
        series = [("qease", "adaptive"), ("qease", "fixed"), ("qsim", "fixed"), ("uniform", "fixed"), ("random", "fixed")]
        if "qease_noq" in AV: series.insert(2, ("qease_noq", "adaptive"))
        fdat = []
        for m, pol in series:
            for a in CFG["ALPHA_SWEEP"]:
                kidx = make_policy_idx(m, kind, eps, a, CAL, TEST, pol); lv, fv = policy_eval(TEST, m, kidx, kind, eps)
                s = summarize_policy(lv, fv); fdat.append(dict(method=m, policy=pol, alpha=a, **s))
        FD = pd.DataFrame(fdat)
        fig, ax = plt.subplots(1, 2, figsize=(COL2, 2.6))
        for m, pol in series:
            d = FD[(FD.method == m) & (FD.policy == pol)]; s = sty(m)
            lab = s["label"] + (" - adaptive" if pol == "adaptive" else " - fixed $k$")
            kw = dict(color=s["color"], marker=s["marker"], ls="-" if pol == "adaptive" else "--", lw=1.6 if pol == "adaptive" else 1.0, label=lab)
            ax[0].plot(d.alpha, d.kept_frac, **kw); ax[0].fill_between(d.alpha, d.frac_lo, d.frac_hi, color=s["color"], alpha=0.12, lw=0)
            ax[1].errorbar(d.alpha, d.test_risk, yerr=[d.test_risk - d.risk_lo, d.risk_hi - d.test_risk], capsize=1.2, elinewidth=0.6, **kw)
        mx = max(CFG["ALPHA_SWEEP"]); ax[1].plot([0, mx], [0, mx], "k:", lw=0.9, label="target $\\alpha$")
        ax[0].set_ylabel("mean fraction of visual tokens kept"); ax[1].set_ylabel(f"realised test risk ({loss_tag(kind, eps)})")
        for a_, t in zip(ax, ["(a)", "(b)"]): a_.set_xlabel("target risk $\\alpha$"); panel_label(a_, t)
        ax[0].legend(fontsize=5.8, loc="lower left")
        nm = "fig_headline_flip" if kind == "flip" else f"fig_headline_js{eps:g}"
        save_fig(fig, nm, FD, f"tokens needed for a guaranteed risk alpha ({loss_tag(kind, eps)}): adaptive CRC vs fixed-k; and realised risk vs target")

# ============================== A7. empirical validity of the guarantee (random re-splits)
def an_crc_resample():
    ALLR = CAL + TEST; n_cal = len(CAL); R = CFG["CRC_RESAMPLES"]; rng = np.random.default_rng(CFG["SEED"] + 7)
    pre = {m: dict(js=arr(ALLR, "js", m), flip=arr(ALLR, "flip", m), kf=arr(ALLR, "kfrac", m), rh=arr(ALLR, "rhat", m) if m.startswith("qease") else None) for m in AV}
    pols = [("qease", "adaptive"), ("qease", "fixed")] + ([("qease_noq", "adaptive")] if "qease_noq" in AV else []) + [(b, "fixed") for b in BASELINES if b in AV]
    rows = []
    for rep in range(R):
        perm = rng.permutation(len(ALLR)); ci, ti = perm[:n_cal], perm[n_cal:]
        for kind, eps in [("flip", None), ("js", EPS0)]:
            for alpha in CFG["ALPHAS"]:
                for m, pol in pols:
                    L = pre[m]["flip"] if kind == "flip" else (pre[m]["js"] > eps).astype(float)
                    if pol == "fixed": kidx = np.full(len(ti), calibrate_fixed(L[ci], alpha))
                    else: kidx = first_ok(pre[m]["rh"][ti], crc_adaptive_lambda(pre[m]["rh"][ci], L[ci], alpha))
                    ar = np.arange(len(ti))
                    rows.append(dict(rep=rep, loss=loss_tag(kind, eps), alpha=alpha, method=m, policy=pol,
                                     risk=float(L[ti][ar, kidx].mean()), kept_frac=float(pre[m]["kf"][ti][ar, kidx].mean())))
    RS = pd.DataFrame(rows)
    S = RS.groupby(["loss", "alpha", "method", "policy"]).agg(mean_risk=("risk", "mean"), std_risk=("risk", "std"), p05=("risk", lambda x: np.percentile(x, 5)),
                                                             p95=("risk", lambda x: np.percentile(x, 95)), max_risk=("risk", "max"),
                                                             frac_splits_with_risk_le_alpha=("risk", lambda x: np.nan),
                                                             mean_kept_frac=("kept_frac", "mean"), std_kept_frac=("kept_frac", "std")).reset_index()
    S["frac_splits_with_risk_le_alpha"] = [float((RS[(RS.loss == r_.loss) & (RS.alpha == r_.alpha) & (RS.method == r_.method) & (RS.policy == r_.policy)].risk <= r_.alpha).mean())
                                           for r_ in S.itertuples()]
    S["mean_risk_le_alpha"] = S["mean_risk"] <= S["alpha"]
    S["method"] = S["method"].map(lambda x: sty(x)["label"])
    save_table(S, "crc_validity_resampled", f"{R} random cal/test re-splits: mean realised risk must be <= alpha (CRC guarantee is in expectation)")
    # figure: distribution of realised risk at alpha = 0.10
    a0 = 0.10 if 0.10 in CFG["ALPHAS"] else CFG["ALPHAS"][len(CFG["ALPHAS"]) // 2]
    fig, ax = plt.subplots(1, 2, figsize=(COL2, 2.6)); fd = []
    for a_, (kind, eps), t in zip(ax, [("flip", None), ("js", EPS0)], ["(a)", "(b)"]):
        sub = RS[(RS.loss == loss_tag(kind, eps)) & (RS.alpha == a0)]; labs, dat = [], []
        for m, pol in pols:
            v = sub[(sub.method == m) & (sub.policy == pol)].risk.values
            dat.append(v); labs.append(sty(m)["label"].replace(" (ours)", "") + ("\nadaptive" if pol == "adaptive" else "\nfixed $k$"))
        bp = a_.boxplot(dat, widths=0.55, patch_artist=True, showfliers=True, flierprops=dict(markersize=1.5), medianprops=dict(color="k", lw=0.9))
        for patch, (m, pol) in zip(bp["boxes"], pols): patch.set_facecolor(sty(m)["color"]); patch.set_alpha(0.6)
        a_.axhline(a0, color="k", ls=":", lw=0.9); a_.set_xticks(range(1, len(pols) + 1)); a_.set_xticklabels(labs, fontsize=5.5, rotation=35, ha="right")
        a_.set_ylabel(f"realised test risk ({loss_tag(kind, eps)})"); panel_label(a_, t)
        fd.append(sub.assign(panel=t))
    save_fig(fig, "fig_crc_validity_resampled", pd.concat(fd), f"distribution of realised risk over {R} random cal/test re-splits at alpha={a0} (dotted = target)")

# ============================== A8. calibration-set size ablation
def an_calsize():
    sizes = [s for s in CFG["CAL_SIZES"] if s < len(CAL)] + [len(CAL)]
    rng = np.random.default_rng(CFG["SEED"] + 11); a0 = 0.10; rows = []
    for kind, eps in [("flip", None), ("js", EPS0)]:
        for s in sizes:
            for rep in range(CFG["CAL_SIZE_REPS"] if s < len(CAL) else 1):
                sub = [CAL[i] for i in rng.choice(len(CAL), s, replace=False)] if s < len(CAL) else CAL
                for m, pol in [("qease", "adaptive"), ("qease", "fixed"), ("qsim", "fixed"), ("random", "fixed")]:
                    if m not in AV: continue
                    kidx = make_policy_idx(m, kind, eps, a0, sub, TEST, pol); lv, fv = policy_eval(TEST, m, kidx, kind, eps)
                    rows.append(dict(loss=loss_tag(kind, eps), n_cal=s, rep=rep, method=m, policy=pol, risk=float(lv.mean()), kept_frac=float(fv.mean())))
    D = pd.DataFrame(rows)
    S = D.groupby(["loss", "n_cal", "method", "policy"]).agg(risk_mean=("risk", "mean"), risk_std=("risk", "std"), frac_mean=("kept_frac", "mean"), frac_std=("kept_frac", "std")).reset_index()
    S["alpha"] = a0; save_table(S.assign(method=S.method.map(lambda x: sty(x)["label"])), "calibration_size_ablation", f"effect of calibration-set size (alpha={a0}); std over random subsets")
    fig, ax = plt.subplots(1, 2, figsize=(COL2, 2.5)); lt = loss_tag("js", EPS0)
    for m, pol in [("qease", "adaptive"), ("qease", "fixed"), ("qsim", "fixed"), ("random", "fixed")]:
        d = S[(S.loss == lt) & (S.method == m) & (S.policy == pol)]
        if not len(d): continue
        s_ = sty(m); kw = dict(color=s_["color"], marker=s_["marker"], ls="-" if pol == "adaptive" else "--", label=s_["label"] + (" - adaptive" if pol == "adaptive" else " - fixed $k$"))
        ax[0].errorbar(d.n_cal, d.frac_mean, yerr=d.frac_std.fillna(0), capsize=1.2, elinewidth=0.6, **kw)
        ax[1].errorbar(d.n_cal, d.risk_mean, yerr=d.risk_std.fillna(0), capsize=1.2, elinewidth=0.6, **kw)
    ax[1].axhline(a0, color="k", ls=":", lw=0.9, label="target $\\alpha$")
    for a_, yl, t in zip(ax, ["mean fraction of tokens kept", f"realised test risk ({lt})"], ["(a)", "(b)"]):
        a_.set_xscale("log"); a_.set_xlabel("# calibration samples"); a_.set_ylabel(yl); panel_label(a_, t)
    ax[0].legend(fontsize=5.8)
    save_fig(fig, "fig_calibration_size", S, "calibration-set size ablation (mean +- std over random subsets)")

# ============================== A9. conditional validity, shift, per-dataset / leave-dataset-out CRC
def an_conditional():
    a0 = 0.10; rows = []
    for kind, eps in LOSS_DEFS:
        for m in [x for x in AV if x in ("qease", "qease_noq")]:
            Lt, Kf = loss_mat(TEST, m, kind, eps), arr(TEST, "kfrac", m)
            kg = make_policy_idx(m, kind, eps, a0, CAL, TEST, "adaptive"); km = mondrian_idx(m, kind, eps, a0, CAL, TEST)
            kfix = make_policy_idx(m, kind, eps, a0, CAL, TEST, "fixed")
            for d in DSL:
                ti = np.array([i for i, r in enumerate(TEST) if r["ds"] == d])
                if len(ti) < 5: continue
                cal_wo = [r for r in CAL if r["ds"] != d]
                klo = make_policy_idx(m, kind, eps, a0, cal_wo, [TEST[i] for i in ti], "adaptive") if len(cal_wo) > 20 else None
                pols = [("adaptive (global CRC)", kg[ti]), ("adaptive (per-dataset CRC)", km[ti]), ("fixed-$k$ (global CRC)", kfix[ti])]
                if klo is not None: pols.append(("adaptive (leave-dataset-out CRC)", klo))
                for pn, kk in pols:
                    lv, fv = Lt[ti, kk], Kf[ti, kk]; lo, hi = wilson(lv.sum(), len(lv))
                    rows.append(dict(loss=loss_tag(kind, eps), alpha=a0, dataset=DS_LABEL[d], n_test=len(ti), method=sty(m)["label"], method_id=m,
                                     policy=pn, test_risk=float(lv.mean()), risk_lo=lo, risk_hi=hi, kept_frac=float(fv.mean()), risk_leq_alpha=bool(lv.mean() <= a0)))
    T = pd.DataFrame(rows); save_table(T.drop(columns="method_id"), "conditional_validity", "per-dataset realised risk under global / per-dataset / leave-dataset-out calibration (marginal CRC does NOT promise per-dataset validity)")
    lt = loss_tag("js", EPS0)
    sub = T[(T["loss"] == lt) & (T.method_id == "qease")]
    pols = ["adaptive (global CRC)", "adaptive (per-dataset CRC)", "adaptive (leave-dataset-out CRC)", "fixed-$k$ (global CRC)"]
    cols = ["#D55E00", "#009E73", "#0072B2", "#7F7F7F"]
    fig, ax = plt.subplots(1, 2, figsize=(COL2, 2.5)); W = 0.8 / len(pols)
    for i, (p, c) in enumerate(zip(pols, cols)):
        d = sub[sub.policy == p].set_index("dataset").reindex([DS_LABEL[x] for x in DSL]); x = np.arange(len(DSL)) + (i - (len(pols) - 1) / 2) * W
        ax[0].bar(x, d.test_risk, W, yerr=[(d.test_risk - d.risk_lo).fillna(0), (d.risk_hi - d.test_risk).fillna(0)], color=c, edgecolor="k", linewidth=0.4, capsize=1, error_kw=dict(lw=0.5), label=p)
        ax[1].bar(x, d.kept_frac, W, color=c, edgecolor="k", linewidth=0.4)
    ax[0].axhline(a0, color="k", ls=":", lw=0.9); ax[0].set_ylabel(f"realised risk ({lt}), $\\alpha$={a0}"); ax[1].set_ylabel("mean fraction of tokens kept")
    for a_, t in zip(ax, ["(a)", "(b)"]): a_.set_xticks(range(len(DSL))); a_.set_xticklabels([DS_LABEL[x] for x in DSL], rotation=30, ha="right"); panel_label(a_, t)
    ax[0].legend(fontsize=5.5, ncol=2, loc="upper center", bbox_to_anchor=(1.1, 1.3))
    save_fig(fig, "fig_conditional_validity", sub.drop(columns="method_id"), "per-dataset validity of the calibrated budget (marginal vs per-dataset vs leave-dataset-out calibration)")

# ============================== A10. quality + calibration of the risk head
def an_riskhead():
    rows, rel = [], []
    for m in [x for x in AV if x.startswith("qease")]:
        P, J, Fl = arr(TEST, "rhat", m), arr(TEST, "js", m), arr(TEST, "flip", m)
        T_ = np.clip(J / LN2, 0, 1)
        within = row_spearman(P, T_)
        pooled = float(pd.Series(P.ravel()).corr(pd.Series(T_.ravel()), method="spearman"))
        row = dict(method=sty(m)["label"], pooled_spearman=pooled, within_sample_spearman=float(np.nanmean(within)),
                   auroc_flip=auroc(P.ravel(), Fl.ravel()), brier_vs_js=float(np.mean((P - T_) ** 2)), mae_vs_js=float(np.mean(abs(P - T_))))
        for e in CFG["EPS_LIST"]: row[f"auroc_JS>{e:g}"] = auroc(P.ravel(), (J > e).ravel())
        qs = pd.qcut(P.ravel(), 10, labels=False, duplicates="drop"); ece = 0.0
        for b in np.unique(qs):
            sel = qs == b; mp, mt_, se = P.ravel()[sel].mean(), T_.ravel()[sel].mean(), T_.ravel()[sel].std() / math.sqrt(sel.sum())
            ece += sel.mean() * abs(mp - mt_); rel.append(dict(method=m, bin=int(b), mean_predicted=mp, mean_true=mt_, sem=se, n=int(sel.sum())))
        row["ECE_10bin"] = float(ece); rows.append(row)
    save_table(pd.DataFrame(rows), "risk_head_quality", "does the risk head rank/calibrate the true answer-distribution shift? (test split)")
    R = pd.DataFrame(rel)
    fig, ax = plt.subplots(1, 2, figsize=(COL2, 2.6))
    for m in R.method.unique():
        d = R[R.method == m]; s = sty(m); ax[0].errorbar(d.mean_predicted, d.mean_true, yerr=d.sem, color=s["color"], marker=s["marker"], ls=s["ls"], capsize=1, elinewidth=0.5, label=s["label"])
    mx = float(R[["mean_predicted", "mean_true"]].max().max()); ax[0].plot([0, mx], [0, mx], "k:", lw=0.9, label="perfect")
    ax[0].set_xlabel("predicted risk (JS / ln 2)"); ax[0].set_ylabel("observed risk (JS / ln 2)"); ax[0].legend(); ax[0].set_xscale("symlog", linthresh=1e-3); ax[0].set_yscale("symlog", linthresh=1e-3); panel_label(ax[0], "(a)")
    Q = pd.DataFrame(rows); mets = [c for c in Q.columns if c.startswith("auroc")]
    W = 0.8 / len(Q)
    for i, rr in Q.iterrows():
        m_id = [k for k in AV if sty(k)["label"] == rr["method"]][0]
        ax[1].bar(np.arange(len(mets)) + (i - (len(Q) - 1) / 2) * W, [rr[c] for c in mets], W, color=sty(m_id)["color"], edgecolor="k", linewidth=0.4, label=rr["method"])
    ax[1].set_xticks(range(len(mets))); ax[1].set_xticklabels([c.replace("auroc_", "AUROC\n") for c in mets], fontsize=6); ax[1].set_ylim(0.4, 1.0); ax[1].axhline(0.5, color="k", ls=":", lw=0.8)
    ax[1].set_ylabel("AUROC of risk head"); ax[1].legend(); panel_label(ax[1], "(b)")
    save_fig(fig, "fig_risk_head", pd.concat([R.assign(panel="a"), Q.melt(id_vars="method", var_name="metric", value_name="value").assign(panel="b")]), "risk-head reliability diagram and discrimination (AUROC)")

# ============================== A11. does the adaptive budget track question difficulty?
def an_budget():
    kind, eps, alpha = "js", EPS0, 0.10
    ent = np.array([r["ent"] for r in TEST]); dsv = np.array([r["ds"] for r in TEST]); ar = np.arange(len(TEST))
    cols = {}
    for m in [x for x in AV if x.startswith("qease")]:
        kidx = make_policy_idx(m, kind, eps, alpha, CAL, TEST, "adaptive"); cols[m] = arr(TEST, "kfrac", m)[ar, kidx]
    D = pd.DataFrame(dict(uid=[r["uid"] for r in TEST], dataset=dsv, answer_entropy=ent, ptop1=[r["ptop1"] for r in TEST], **{f"frac_{m}": v for m, v in cols.items()}))
    D["entropy_decile"] = pd.qcut(D.answer_entropy, 10, labels=False, duplicates="drop")
    rows = []
    for dc, g in D.groupby("entropy_decile"):
        for m in cols:
            mu, lo, hi, _ = boot_curve(g[f"frac_{m}"].values, B=500)
            rows.append(dict(entropy_decile=int(dc), mean_entropy=g.answer_entropy.mean(), method=m, kept_frac=mu[0], lo=lo[0], hi=hi[0], n=len(g)))
    BD = pd.DataFrame(rows)
    sp = [dict(scope="all", method=sty(m)["label"], spearman_kept_vs_entropy=float(D[["answer_entropy", f"frac_{m}"]].corr(method="spearman").iloc[0, 1])) for m in cols]
    for d in DSL:
        g = D[D.dataset == d]
        if len(g) > 10: sp += [dict(scope=DS_LABEL[d], method=sty(m)["label"], spearman_kept_vs_entropy=float(g[["answer_entropy", f"frac_{m}"]].corr(method="spearman").iloc[0, 1])) for m in cols]
    save_table(pd.DataFrame(sp), "budget_vs_difficulty_spearman", f"rank-correlation between allocated budget and full-model answer entropy (CRC adaptive, {loss_tag(kind, eps)}, alpha={alpha})")
    save_table(D.drop(columns=["uid"]).groupby("dataset").agg(**{f"mean_kept_{m}": (f"frac_{m}", "mean") for m in cols}, std_kept_qease=("frac_qease", "std"),
                                                                 mean_entropy=("answer_entropy", "mean")).reset_index(),
               "budget_by_dataset", "adaptive budget allocated per dataset")
    fig, ax = plt.subplots(1, 2, figsize=(COL2, 2.5))
    for m in cols:
        d = BD[BD.method == m].sort_values("mean_entropy"); s = sty(m)
        ax[0].plot(d.mean_entropy, d.kept_frac, color=s["color"], marker=s["marker"], ls=s["ls"], label=s["label"]); ax[0].fill_between(d.mean_entropy, d.lo, d.hi, color=s["color"], alpha=0.15, lw=0)
    ax[0].set_xlabel("full-model answer entropy (nats), decile means"); ax[0].set_ylabel("mean fraction of tokens kept"); ax[0].legend(); panel_label(ax[0], "(a)")
    data = [D[D.dataset == d].frac_qease.values for d in DSL]
    bp = ax[1].boxplot(data, widths=0.55, patch_artist=True, flierprops=dict(markersize=1.5), medianprops=dict(color="k", lw=0.9))
    for patch, d in zip(bp["boxes"], DSL): patch.set_facecolor(DS_COLOR[d]); patch.set_alpha(0.6)
    ax[1].set_xticks(range(1, len(DSL) + 1)); ax[1].set_xticklabels([DS_LABEL[d] for d in DSL], rotation=30, ha="right"); ax[1].set_ylabel("fraction of tokens kept (Q-EASE adaptive)"); panel_label(ax[1], "(b)")
    save_fig(fig, "fig_budget_vs_difficulty", D, "adaptive budget vs question difficulty (a) and per-dataset budget distribution (b)")

# ============================== A12. qualitative examples + failure cases
def _draw_overlay(ax, rec, selmask, geom, title):
    nt = CFG["TOKENS_PER_DS"].get(rec["ds"], CFG["TOKENS_PER_DS"]["default"])
    img = smart_resize(Image.open(POOL_DIR / rec["img"]).convert("RGB"), nt)
    ax.imshow(img); ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    if selmask is not None:
        win, gw, gh, H, W = geom["win"], geom["gw"], geom["gh"], geom["H"], geom["W"]
        for gi in range(gh * gw):
            if not selmask[gi]:
                gr, gc = divmod(gi, gw); x0, y0 = gc * win * 28, gr * win * 28
                ax.add_patch(Rectangle((x0, y0), min(win, W - gc * win) * 28, min(win, H - gr * win) * 28, facecolor="k", alpha=0.62, lw=0))
    ax.set_title(title, fontsize=5.5, pad=2)

def an_qual():
    show = [m for m in ["qease", "qsim", "uniform", "random"] if m in AV]
    gap = lambda r: min(r["js"][b][J25] for b in BASELINES) - r["js"]["qease"][J25]
    cand = [r for r in TEST if "sel25" in r and "qease" in r["sel25"]]
    wins = []
    for d in DSL:
        g = sorted([r for r in cand if r["ds"] == d], key=gap, reverse=True)
        if g: wins.append(g[0])
    wins = wins[: CFG["N_QUAL"]]; fails = sorted(cand, key=gap)[:3]
    rows = []
    for name, group in [("fig_qualitative_wins", wins), ("fig_qualitative_failures", fails)]:
        if not group: continue
        fig, ax = plt.subplots(len(group), len(show) + 1, figsize=(COL2, 1.35 * len(group) + 0.2))
        ax = np.atleast_2d(ax); fd = []
        for i, r in enumerate(group):
            rec = REC_BY_UID[r["uid"]]; q = (r["q"] if len(r["q"]) < 48 else r["q"][:45] + "...").replace("$", "\\$").replace("_", "\\_")
            _draw_overlay(ax[i, 0], rec, None, r["geom"], f"{DS_LABEL[r['ds']]}: {q}")
            for j, m in enumerate(show):
                _draw_overlay(ax[i, j + 1], rec, r["sel25"][m], r["geom"], f"{sty(m)['label'].replace(' (ours)', '')}  JS={r['js'][m][J25]:.3f}")
                fd.append(dict(uid=r["uid"], dataset=r["ds"], question=r["q"], method=m, budget=FR[J25], js=float(r["js"][m][J25]), kept_group_indices=";".join(map(str, np.where(r["sel25"][m])[0]))))
        save_fig(fig, name, pd.DataFrame(fd), "qualitative examples at 25% budget (dark = dropped groups); " + ("largest Q-EASE advantage per dataset" if "wins" in name else "largest Q-EASE failures"))

def an_failures():
    rows = []
    for r in TEST:
        base = {b: float(r["js"][b][J25]) for b in BASELINES if b in r["js"]}
        rows.append(dict(uid=r["uid"], dataset=DS_LABEL[r["ds"]], question=r["q"], answer_entropy=r["ent"], G=r["G"], js_qease=float(r["js"]["qease"][J25]),
                         **{f"js_{b}": v for b, v in base.items()}, best_baseline_js=min(base.values()),
                         qease_advantage=min(base.values()) - float(r["js"]["qease"][J25]), qease_flip=bool(r["flip"]["qease"][J25])))
    D = pd.DataFrame(rows).sort_values("qease_advantage")
    save_table(D.head(60), "failure_cases_qease", "60 test samples where Q-EASE (25% budget) is most worse than the best baseline", latex=False)
    save_table(D.tail(60).iloc[::-1], "success_cases_qease", "60 test samples where Q-EASE (25% budget) is most better than the best baseline", latex=False)
    fam = D.assign(bin=pd.cut(D.answer_entropy, [-1, 0.1, 0.5, 1.5, 100], labels=["very confident", "confident", "uncertain", "very uncertain"]))
    save_table(fam.groupby("bin", observed=True).agg(n=("uid", "count"), mean_js_qease=("js_qease", "mean"), mean_best_baseline=("best_baseline_js", "mean"),
                                                    win_rate=("qease_advantage", lambda x: float((x > 0).mean())), flip_rate=("qease_flip", "mean")).reset_index(),
               "qease_by_answer_uncertainty", "Q-EASE vs best baseline stratified by full-model answer uncertainty (25% budget)")
# ============================== A14. real latency / memory / FLOPs
def an_latency():
    p = TAB / "raw_latency.csv"
    if not p.exists(): log("   no latency measurements - skipping"); return
    R = pd.read_csv(p)
    S = R.groupby("frac").agg(n=("uid", "nunique"), enc_ms=("enc_ms", "mean"), selector_ms=("selector_ms", "mean"), prefill_ms=("prefill_ms", "mean"),
                              prefill_ms_std=("prefill_ms", "std"), seq_tokens=("seq_tokens", "mean"), prefill_gflops=("prefill_gflops", "mean"),
                              peak_extra_mem_mib=("peak_extra_mem_mib", "mean")).reset_index().sort_values("frac", ascending=False)
    full = S[S.frac == 1.0].iloc[0]
    S["total_ms"] = S.enc_ms + S.selector_ms + S.prefill_ms
    S["prefill_speedup"] = full.prefill_ms / S.prefill_ms
    S["end_to_end_speedup_incl_selector"] = (full.enc_ms + full.prefill_ms) / S.total_ms
    S["gflops_reduction_x"] = full.prefill_gflops / S.prefill_gflops
    S["vision_tower_share_of_full_pipeline_pct"] = 100 * full.enc_ms / (full.enc_ms + full.prefill_ms)
    save_table(S, "latency_compute", f"physically pruned prefill latency / FLOPs / memory (GPU {DEVICES[0]}, batch 1); selector cost included in end-to-end speed-up")
    print(S.round(2).to_string(index=False))
    fig, ax = plt.subplots(1, 2, figsize=(COL2, 2.5)); x = np.arange(len(S))
    ax[0].bar(x, S.enc_ms, color="#999999", edgecolor="k", lw=0.4, label="vision tower + projector")
    ax[0].bar(x, S.selector_ms, bottom=S.enc_ms, color="#D55E00", edgecolor="k", lw=0.4, label="Q-EASE selector")
    ax[0].bar(x, S.prefill_ms, bottom=S.enc_ms + S.selector_ms, color="#0072B2", edgecolor="k", lw=0.4, label="LLM prefill")
    ax[0].set_xticks(x); ax[0].set_xticklabels([f"{f:g}" for f in S.frac]); ax[0].set_xlabel("fraction of groups kept"); ax[0].set_ylabel("latency (ms)"); ax[0].legend(); panel_label(ax[0], "(a)")
    ax[1].plot(S.frac, S.prefill_speedup, "o-", color="#0072B2", label="LLM prefill"); ax[1].plot(S.frac, S.end_to_end_speedup_incl_selector, "s-", color="#D55E00", label="end-to-end (incl. selector)")
    ax[1].plot(S.frac, 1 / S.frac, "k:", label="ideal $1/f$"); ax[1].set_xlabel("fraction of groups kept"); ax[1].set_ylabel("speed-up vs full"); ax[1].invert_xaxis(); ax[1].legend(); panel_label(ax[1], "(b)")
    save_fig(fig, "fig_latency", S, "measured latency breakdown and speed-up of physically pruned prefill")

# ============================== A14b. compute-equivalent comparison (FastV prunes INSIDE the LLM, Q-EASE before it)
def an_compute_equiv():
    if "fastv" not in AV or not all("fastv_flops_frac" in r and "L" in r for r in TEST):
        log("   FastV records missing - skipping compute-equivalent table"); return
    L = np.array([r["L"] for r in TEST], float); nv = np.array([r["n_vis"] for r in TEST], float)
    fv_eff = np.stack([r["fastv_flops_frac"] for r in TEST]).mean(0)
    rows = []
    for j, f in enumerate(FR):
        q_tok = arr(TEST, "kfrac", "qease")[:, j]
        q_eff = ((L - nv + q_tok * nv) / L).mean()                     # Q-EASE prunes before layer 0 -> whole prefill shrinks
        for m in AV:
            kf = arr(TEST, "kfrac", m)[:, j].mean()
            eff = (np.stack([r_[f"{m}_flops_frac"] for r_ in TEST]).mean(0)[j] if all(f"{m}_flops_frac" in r_ for r_ in TEST)
                   else ((L - nv + arr(TEST, "kfrac", m)[:, j] * nv) / L).mean())
            rows.append(dict(nominal_budget=f, method=sty(m)["label"], visual_tokens_kept_frac=kf, effective_prefill_flops_frac=eff,
                             mean_js=arr(TEST, "js", m)[:, j].mean(), flip_rate=arr(TEST, "flip", m)[:, j].mean()))
    T = pd.DataFrame(rows)
    save_table(T, "compute_equivalent_comparison", "fidelity vs EFFECTIVE prefill FLOPs fraction (FastV keeps layers < K at full length, so its compute is higher than its token fraction)")
    fig, ax = plt.subplots(1, 2, figsize=(COL2, 2.6))
    for m in AV:
        d = T[T.method == sty(m)["label"]].sort_values("effective_prefill_flops_frac"); s = sty(m)
        for a_, col in zip(ax, ["mean_js", "flip_rate"]):
            a_.plot(d.effective_prefill_flops_frac, d[col], color=s["color"], marker=s["marker"], ls=s["ls"], label=s["label"])
    for a_, yl, t in zip(ax, ["mean JS to full model (nats)", "answer-flip rate"], ["(a)", "(b)"]):
        a_.set_xlabel("effective prefill FLOPs (fraction of full model)"); a_.set_ylabel(yl); panel_label(a_, t)
    ax[0].set_yscale("symlog", linthresh=1e-3); ax[0].legend()
    save_fig(fig, "fig_compute_equivalent", T, "fidelity vs effective prefill compute (fair axis when comparing in-LLM and pre-LLM pruning)")

# ============================== A14c. EXACT compute-matched comparison against the in-LLM pruners (FastV, SparseVLM)
def an_matched_compute():
    rows = []
    def add(ref_key, blabels, eff, ref_js, ref_fl, ref_kf, matched, note):
        for b, bl in enumerate(blabels):
            mu, lo, hi, _ = boot_curve(ref_js[:, b]); fmu, flo, fhi, _ = boot_curve(ref_fl[:, b])
            rows.append(dict(reference=sty(ref_key)["label"], budget=bl, effective_prefill_flops_frac=float(eff[b]), method=sty(ref_key)["label"], is_reference=True,
                             visual_tokens_kept_frac=float(np.mean(ref_kf[:, b])), mean_js=mu[0], js_lo=lo[0], js_hi=hi[0], mean_flip=fmu[0], flip_lo=flo[0], flip_hi=fhi[0],
                             js_diff_vs_ref=0.0, diff_lo=0.0, diff_hi=0.0, p_boot=np.nan, win_rate_vs_ref=np.nan, n=len(ref_js), note=note))
            for m, d in matched.items():
                mu, lo, hi, _ = boot_curve(d["js"][:, b]); fmu, flo, fhi, _ = boot_curve(d["flip"][:, b].astype(float))
                dm, dlo, dhi, p, _ = paired_test(d["js"][:, b], ref_js[:, b])
                rows.append(dict(reference=sty(ref_key)["label"], budget=bl, effective_prefill_flops_frac=float(eff[b]), method=sty(m)["label"], is_reference=False,
                                 visual_tokens_kept_frac=float(np.mean(d["kfrac"][:, b])), mean_js=mu[0], js_lo=lo[0], js_hi=hi[0], mean_flip=fmu[0], flip_lo=flo[0], flip_hi=fhi[0],
                                 js_diff_vs_ref=dm[0], diff_lo=dlo[0], diff_hi=dhi[0], p_boot=p[0], win_rate_vs_ref=float((d["js"][:, b] < ref_js[:, b]).mean()), n=len(ref_js), note=note))
    if "fastv" in AV and all("fastv_matched" in r for r in TEST):
        names = [m for m in ("qease", "qease_noq", "qsim", "uniform", "random") if all(m in r["fastv_matched"] for r in TEST)]
        matched = {m: {k: np.stack([r["fastv_matched"][m][k] for r in TEST]).astype(float) for k in ("js", "flip", "kfrac")} for m in names}
        add("fastv", [f"{f:g}" for f in FR], np.stack([r["fastv_flops_frac"] for r in TEST]).mean(0), arr(TEST, "js", "fastv"), arr(TEST, "flip", "fastv"),
            arr(TEST, "kfrac", "fastv"), matched, "group methods get the visual-token budget that equals FastV's effective prefill FLOPs, per sample")
    if all("sparse" in r for r in TEST) and TEST:
        sp0 = TEST[0]["sparse"]; names = [m for m in ("qease", "qease_noq", "qsim", "uniform", "random") if all(m in r["sparse"]["matched"] for r in TEST)]
        matched = {m: {k: np.stack([r["sparse"]["matched"][m][k] for r in TEST]).astype(float) for k in ("js", "flip", "kfrac")} for m in names}
        add("sparsevlm", [str(x) for x in sp0["retain"]], np.stack([r["sparse"]["eff"] for r in TEST]).mean(0), np.stack([r["sparse"]["js"] for r in TEST]).astype(float),
            np.stack([r["sparse"]["flip"] for r in TEST]).astype(float), np.stack([r["sparse"]["kfrac_avg"] for r in TEST]).astype(float), matched,
            "SparseVLM budget = official RETAIN_TOKN (avg visual tokens/layer for 576 tokens); group methods matched to its effective prefill FLOPs per sample")
    if not rows: log("   no FastV/SparseVLM records - skipping compute-matched comparison"); return
    T = pd.DataFrame(rows)
    ok = ~T.is_reference; T.loc[ok, "p_holm"] = holm(T.loc[ok, "p_boot"].values)
    save_table(T, "compute_matched_vs_published", "group-based methods (incl. Q-EASE) vs FastV / SparseVLM at EXACTLY matched per-sample effective prefill compute; negative js_diff = closer to full model than the published method")
    refs = [r_ for r_ in T.reference.unique()]
    fig, ax = plt.subplots(len(refs), 2, figsize=(COL2, 2.5 * len(refs))); ax = np.atleast_2d(ax)
    inv = {sty(k)["label"]: k for k in list(METHOD_STYLE)}
    for i, ref in enumerate(refs):
        for j, (col, lo_, hi_, yl) in enumerate([("mean_js", "js_lo", "js_hi", "mean JS to full model (nats)"), ("mean_flip", "flip_lo", "flip_hi", "answer-flip rate")]):
            for mlabel, g in T[T.reference == ref].groupby("method"):
                g = g.sort_values("effective_prefill_flops_frac"); s = sty(inv.get(mlabel, "qease"))
                ax[i, j].errorbar(g.effective_prefill_flops_frac, g[col], yerr=[g[col] - g[lo_], g[hi_] - g[col]], color=s["color"], marker=s["marker"], ls=s["ls"],
                                  capsize=1.2, elinewidth=0.6, lw=1.6 if mlabel == ref else 1.0, label=mlabel)
            ax[i, j].set_xlabel("effective prefill FLOPs (fraction of full model)"); ax[i, j].set_ylabel(yl); ax[i, j].set_title(f"vs {ref}", fontsize=7)
            panel_label(ax[i, j], "(" + "abcd"[2 * i + j] + ")")
        ax[i, 0].legend(fontsize=5.5)
    fig.subplots_adjust(hspace=0.55, wspace=0.28)              # keep each row's title clear of the row above's x-label
    save_fig(fig, "fig_compute_matched_vs_published", T, "fidelity at exactly matched effective prefill compute: published in-LLM pruners vs pre-LLM group methods (error bars = 95% CI)")

# ============================== A15. manifest, README, recreate-example, zip
def an_finalize():
    MANIFEST.append(("meta", "results/README_results.md", "this index")); MANIFEST.append(("meta", "results/MANIFEST.csv", "machine-readable index"))
    M = pd.DataFrame(MANIFEST, columns=["kind", "path", "description"]); M.to_csv(WORK / "results" / "MANIFEST.csv", index=False)
    lines = [f"# Q-EASE results ({TAG})", "", f"Generated {time.strftime('%Y-%m-%d %H:%M:%S')}. Test n={len(TEST)}, calibration n={len(CAL)}.", "",
             "* Every figure has `figures/<name>.pdf|png` (vector PDF, TrueType fonts; 600-dpi PNG) and the exact tidy data in `figdata/<name>.csv`.",
             "* Every table is a CSV (+ a `.tex` booktabs-style export when pandas/jinja2 allow).",
             "* All uncertainty = 95%% percentile bootstrap over test samples (B=%d) unless stated; risks use Wilson intervals." % CFG["BOOT_B"],
             "* CRC guarantee is marginal (calibration+test exchangeable); see `conditional_validity` and `crc_validity_resampled` for how it behaves empirically.", ""]
    for kind in ["figure", "figure-data", "table", "meta"]:
        lines += [f"## {kind}", ""] + [f"* `{r.path}` - {r.description}" for r in M[M.kind == kind].itertuples()] + [""]
    (WORK / "results" / "README_results.md").write_text("\n".join(lines))
    (WORK / "results" / "recreate_figure_example.py").write_text(
        '"""Re-draw a curve figure from its exported data (no GPU / model needed)."""\nimport pandas as pd, matplotlib.pyplot as plt\n'
        'd = pd.read_csv("figdata/fig_fidelity_curves.csv")\nfig, ax = plt.subplots(figsize=(3.5, 2.6))\n'
        'for m, g in d[d.metric == "flip"].groupby("method"):\n    g = g.sort_values("frac"); ax.plot(g.frac, g["mean"], marker="o", label=m); ax.fill_between(g.frac, g.lo, g.hi, alpha=.15)\n'
        'ax.set_xscale("log", base=2); ax.set_xlabel("fraction of groups kept"); ax.set_ylabel("answer-flip rate"); ax.legend(fontsize=6)\nfig.savefig("redrawn.pdf", bbox_inches="tight")\n')
    import shutil
    z = shutil.make_archive(str(BASE / f"qease_results_{TAG}"), "zip", WORK / "results")
    log(f"[analyze] results zipped -> {z}")


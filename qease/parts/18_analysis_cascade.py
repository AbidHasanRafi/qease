# %% --------------------------------------------- 17. ANALYSIS OF THE NOVELTY (cascade) - tables, figures, certificates
METHOD_STYLE.update({
    "qease_greedy": dict(color="#A6001A", marker="o", ls="--", label="Q-EASE greedy (MMR set)"),
    "divprune":     dict(color="#117733", marker="<", ls="-",  label="DivPrune-style"),
    "cdpruner":     dict(color="#882255", marker=">", ls="-",  label="CDPruner-style (cond. DPP)"),
    "dart":         dict(color="#44AA99", marker="p", ls="-",  label="DART-style (layer 2)"),
    "pdrop":        dict(color="#AA4499", marker="d", ls="-",  label="PyramidDrop"),
})
CC = dict(full="#000000", fixed="#0072B2", static="#009E73", casc="#E69F00", casc_ab="#D55E00", uncal="#999999", oracle="#CC79A7")
CS_STATE = {}

def casc_state():
    if CS_STATE: return CS_STATE
    CS = get_casc(); assert CS is not None, "run the 'casc' stage first (needs decoded records from 'dec')"
    head, cal, test = (load_dec(DEC_DIR / x) for x in ("head", "cal", "test"))
    Ah, Ac, At = dec_arrays(head), dec_arrays(cal), dec_arrays(test)
    s_c, u_c = head_scores(CS["H"], Ac); s_t, u_t = head_scores(CS["H"], At)
    CS_STATE.update(CS=CS, Ah=Ah, Ac=Ac, At=At, s_c=s_c, u_c=u_c, s_t=s_t, u_t=u_t, sel_cost=get_sel_cost(), S=len(CFG["DEC_GRID"]), delta=CS["delta"])
    log(f"[casc] state: head={len(head)} cal={len(cal)} test={len(test)} | full-model accuracy test={At['SC'][:, -1].mean():.3f} | selector cost={CS_STATE['sel_cost']:.3f} prefill")
    return CS_STATE

def headline_key(alpha, ab):
    ks = [k for k in casc_state()["CS"]["pol"] if k[1] == alpha and k[2] == ab]
    return min(ks, key=lambda k: casc_state()["CS"]["pol"][k]["cal_cost"])           # path with the lowest CALIBRATION cost (no test peeking)

def run_policy(A, s, u, key, st=None):
    st = st or casc_state(); pol = st["CS"]["pol"][key]
    return cascade_sim(path_idx(CFG["CASC_PATHS"][key[0]]), s, u, A["COST"], pol["lam"], pol["mu"], st["sel_cost"])

def oracle_cascade(A, path, abstain=False):
    n, S = A["C"].shape; cfull = A["C"][:, -1]
    if abstain:                                                                      # perfect futility knowledge: skip every question the full model gets wrong
        return np.full(n, path[0]), ~cfull, A["COST"][:, path[0]].astype(np.float64)
    final, done, cost = np.full(n, path[-1]), np.zeros(n, bool), np.zeros(n)
    for j in path:
        cost += np.where(~done, A["COST"][:, j], 0.0)
        harmless = np.ones(n, bool) if j == S - 1 else ((~cfull) | A["C"][:, j])
        stop = ~done & harmless; final[stop] = j; done |= stop
    return final, np.zeros(n, bool), cost

def an_casc_main():
    st = casc_state(); At, Ac, S, delta = st["At"], st["Ac"], st["S"], st["delta"]; n = len(At["C"]); ar = np.arange(n); rows = []; nof = np.zeros(n, bool)
    def add(name, alpha, final, abst, cost, **extra):
        m = policy_metrics(At, final, abst, cost); mu_, lo_, hi_, _ = boot_curve(np.asarray(cost, float), B=500)
        m.update(method=name, alpha=alpha, cost_lo=lo_[0], cost_hi=hi_[0], **extra); rows.append(m)
    add("Full model", np.nan, np.full(n, S - 1), nof, np.ones(n))
    Lh_c = fixed_losses(Ac); rh_c = Ac["FE"][:, :-1, FEATS.index("rhat")]
    nus = np.concatenate([[-np.inf], np.geomspace(0.005, 0.9, CFG["CASC_LAMBDAS"])])                  # fixed grid (not derived from calibration data)
    for alpha in CFG["CASC_ALPHAS"]:
        j = fixed_choose(Lh_c, Ac["COST"], alpha, delta)
        add("Fixed budget (Q-EASE, LTT)", alpha, np.full(n, j), nof, At["COST"][:, j] + (st["sel_cost"] if j < S - 1 else 0.0), budget=CFG["DEC_GRID"][j])
        nu = static_choose(Ac, alpha, delta, nus); stg = static_stage(At["FE"][:, :-1, FEATS.index("rhat")], nu)
        add("Static adaptive (SRH only, LTT)", alpha, stg, nof, At["COST"][ar, stg] + st["sel_cost"])
        for ab, lab in [(False, "Cascade: escalate only (LTT)"), (True, "Cascade: escalate or abstain (LTT)")]:
            key = headline_key(alpha, ab); fin, abst, cost = run_policy(At, st["s_t"], st["u_t"], key, st)
            pol = st["CS"]["pol"][key]; add(lab, alpha, fin, abst, cost, path=key[0], lam=pol["lam"], mu=pol["mu"])
        key = headline_key(alpha, False); path = path_idx(CFG["CASC_PATHS"][key[0]])          # naive use of the head: lambda = alpha, no calibration
        fin, abst, cost = cascade_sim(path, st["s_t"], st["u_t"], At["COST"], alpha, np.inf, st["sel_cost"]); add("Cascade, uncalibrated (lambda=alpha)", alpha, fin, abst, cost, path=key[0])
    key = headline_key(CFG["CASC_ALPHAS"][0], False); path = path_idx(CFG["CASC_PATHS"][key[0]])
    fin, abst, cost = oracle_cascade(At, path); add("Oracle cascade (knows the labels)", np.nan, fin, abst, cost, path=key[0])
    fin, abst, cost = oracle_cascade(At, path, True); add("Oracle cascade + oracle abstention", np.nan, fin, abst, cost, path=key[0])
    T = pd.DataFrame(rows)
    T["cert_ok_point"] = np.where(T.alpha.notna(), T.lost_correct <= T.alpha, np.nan)
    T["acc_rel_full_pct"] = 100 * T.accuracy / T.accuracy_full
    T["speedup_prefill"] = 1 / T.mean_cost; T["speedup_e2e_model"] = 1 / T.e2e_cost
    save_table(T, "casc_main", f"escalate-or-abstain cascade vs fixed / static / uncalibrated / oracle on the test split (n={n}); lost_correct = P(full correct & not answered correctly); "
                               f"certificate: P_cal(risk<=alpha)>=1-{delta}; cost in units of one full prefill (selector included); e2e_cost uses the measured vision-tower share")
    print(T[["method", "alpha", "lost_correct", "accuracy", "selective_accuracy", "coverage", "mean_cost", "e2e_cost", "escalated_to_full"]].round(3).to_string(index=False))
    # ---- figure: cost vs lost-correct (a), stage usage (b)
    fig, ax = plt.subplots(1, 2, figsize=(COL2, 2.7))
    cmap = {"Full model": ("full", "*", 70), "Fixed budget (Q-EASE, LTT)": ("fixed", "s", 28), "Static adaptive (SRH only, LTT)": ("static", "^", 30),
            "Cascade: escalate only (LTT)": ("casc", "o", 30), "Cascade: escalate or abstain (LTT)": ("casc_ab", "o", 36), "Cascade, uncalibrated (lambda=alpha)": ("uncal", "x", 30),
            "Oracle cascade (knows the labels)": ("oracle", "D", 22), "Oracle cascade + oracle abstention": ("oracle", "P", 28)}
    for m, (ck, mk, sz) in cmap.items():
        d = T[T.method == m]
        if len(d): ax[0].scatter(d.mean_cost, d.lost_correct, color=CC[ck], marker=mk, s=sz, edgecolor="k", linewidths=0.4, label=m, zorder=4)
    for a in CFG["CASC_ALPHAS"]: ax[0].axhline(a, color="0.6", ls=":", lw=0.7)
    ax[0].set_xlabel("mean cost (full-prefill units, incl. selector + re-runs)"); ax[0].set_ylabel("lost-correct rate on test"); ax[0].legend(fontsize=4.8, loc="upper right"); panel_label(ax[0], "(a)")
    sdat = []
    for alpha in CFG["CASC_ALPHAS"]:
        key = headline_key(alpha, True); fin, abst, _ = run_policy(At, st["s_t"], st["u_t"], key, st)
        for j in range(S):
            sdat.append(dict(alpha=alpha, stage_frac=CFG["DEC_GRID"][j], answered=float(((fin == j) & ~abst).mean()), abstained=float(((fin == j) & abst).mean())))
    D = pd.DataFrame(sdat); xs = np.arange(len(CFG["CASC_ALPHAS"])); bottom = np.zeros(len(xs)); cols = plt.cm.viridis(np.linspace(0.15, 0.9, S))
    for j in range(S):
        v = np.array([D[(D.alpha == a) & (D.stage_frac == CFG["DEC_GRID"][j])].answered.iloc[0] for a in CFG["CASC_ALPHAS"]])
        ax[1].bar(xs, v, 0.55, bottom=bottom, color=cols[j], edgecolor="k", linewidth=0.3, label=f"answer @ {CFG['DEC_GRID'][j]:.3g}"); bottom += v
    ab_ = np.array([D[D.alpha == a].abstained.sum() for a in CFG["CASC_ALPHAS"]])
    ax[1].bar(xs, ab_, 0.55, bottom=bottom, color="#bbbbbb", edgecolor="k", hatch="//", linewidth=0.3, label="abstain")
    ax[1].set_xticks(xs); ax[1].set_xticklabels([f"$\\alpha$={a}" for a in CFG["CASC_ALPHAS"]]); ax[1].set_ylabel("fraction of test questions"); ax[1].legend(fontsize=4.8, ncol=2, loc="lower right"); panel_label(ax[1], "(b)")
    save_fig(fig, "fig_casc_main", pd.concat([T.assign(panel="a"), D.assign(panel="b")], ignore_index=True), "escalate-or-abstain: (a) cost vs lost-correct rate for every controller, (b) where the cascade stops")
    # ---- risk-coverage: sweep the futility threshold mu at fixed calibrated lambda; compare with abstaining on the full model's own confidence (needs the full run)
    key = headline_key(0.05, True); pol = st["CS"]["pol"][key]; path = path_idx(CFG["CASC_PATHS"][key[0]]); rc = []
    for mu in [np.inf] + list(np.linspace(0.98, 0.3, 24)):
        fin, abst, cost = cascade_sim(path, st["s_t"], st["u_t"], At["COST"], pol["lam"], mu, st["sel_cost"]); m = policy_metrics(At, fin, abst, cost)
        rc.append(dict(curve="cascade futility head (cheap)", mu=mu, coverage=m["coverage"], selective_accuracy=m["selective_accuracy"], cost=m["mean_cost"], lost_correct=m["lost_correct"]))
    conf = At["FE"][:, -1, FEATS.index("top1")]
    for q in np.linspace(0, 0.7, 15):
        keep = conf >= np.quantile(conf, q)
        rc.append(dict(curve="full-model confidence (needs full run)", mu=np.nan, coverage=float(keep.mean()), selective_accuracy=float(At["SC"][keep, -1].mean()), cost=1.0, lost_correct=float((At["C"][:, -1] & ~keep).mean())))
    RC = pd.DataFrame(rc); fig, ax = plt.subplots(1, 2, figsize=(COL2, 2.5))
    for cn, col in [("cascade futility head (cheap)", CC["casc_ab"]), ("full-model confidence (needs full run)", CC["fixed"])]:
        d = RC[RC.curve == cn].sort_values("coverage")
        ax[0].plot(d.coverage, d.selective_accuracy, color=col, marker="o", ms=2.5, label=cn); ax[1].plot(d.cost, d.selective_accuracy, color=col, marker="o", ms=2.5, label=cn)
    ax[0].axhline(At["SC"][:, -1].mean(), color="0.5", ls=":", lw=0.8, label="full model, no abstention")
    ax[0].set_xlabel("coverage (answered fraction)"); ax[0].set_ylabel("selective accuracy"); ax[0].legend(fontsize=5); panel_label(ax[0], "(a)")
    ax[1].set_xlabel("mean cost (full-prefill units)"); ax[1].set_ylabel("selective accuracy"); panel_label(ax[1], "(b)")
    save_fig(fig, "fig_risk_coverage", RC, "risk-coverage: abstaining with the cheap futility head vs abstaining on the full model's confidence")
    save_table(RC, "risk_coverage", "selective accuracy vs coverage / cost for the abstention rules")

def an_casc_resplit():
    """supervisor: 'single-split outcomes were read as signal' -> report the DISTRIBUTION over random re-splits, for LTT and for the expectation-only CRC rule."""
    st = casc_state(); CS = st["CS"]; delta = st["delta"]; lams, mus = CS["lams"], CS["mus"]; nl = len(lams)
    Aall = {k: (np.concatenate([st["Ac"][k], st["At"][k]]) if isinstance(st["Ac"][k], np.ndarray) else st["Ac"][k] + st["At"][k]) for k in st["Ac"]}
    s_all, u_all = np.vstack([st["s_c"], st["s_t"]]), np.vstack([st["u_c"], st["u_t"]]); n = len(s_all); B = CFG["CASC_RESPLITS"]; rng = np.random.default_rng(0)
    rows, raw = [], []
    for pname, fr in CFG["CASC_PATHS"].items():
        Lm, Cm = loss_cost_matrix(path_idx(fr), Aall, s_all, u_all, lams, mus, st["sel_cost"])
        for alpha in CFG["CASC_ALPHAS"]:
            for rule in ("ltt", "crc"):
                R, Cst = [], []
                for b in range(B):
                    perm = rng.permutation(n); ci, ti = perm[: n // 2], perm[n // 2:]
                    c = ltt_choose(Lm[ci], Cm[ci], nl, list(range(len(mus))), alpha, delta, rule)
                    R.append(float(Lm[ti, c].mean())); Cst.append(float(Cm[ti, c].mean()))
                R, Cst = np.array(R), np.array(Cst)
                # the certificate is about the POPULATION risk; with n_test = n/2 the empirical test risk fluctuates around it, so we also report P(risk <= alpha + 2 s.e.)
                se = math.sqrt(alpha * (1 - alpha) / (n // 2))
                rows.append(dict(path=pname, alpha=alpha, rule={"ltt": "Learn-then-Test (ours)", "crc": "expectation-only CRC"}[rule], n_resplits=B, n_cal=n // 2, n_test=n - n // 2,
                                 mean_risk=R.mean(), q05=np.quantile(R, 0.05), q50=np.median(R), q95=np.quantile(R, 0.95), share_risk_le_alpha=float((R <= alpha).mean()),
                                 share_risk_le_alpha_2se=float((R <= alpha + 2 * se).mean()), mean_cost=Cst.mean(), cost_q05=np.quantile(Cst, 0.05), cost_q95=np.quantile(Cst, 0.95)))
                raw += [dict(path=pname, alpha=alpha, rule=rule, split=b, test_risk=R[b], test_cost=Cst[b]) for b in range(B)]
    T = pd.DataFrame(rows); save_table(T, "casc_resplits", f"distribution over {B} random 50/50 cal/test re-splits of the {n} decoded calibration+test questions (heads fixed): test risk and cost of the chosen cascade")
    RAW = pd.DataFrame(raw); head = T[T.path == T.path.iloc[0]]
    print(T[["path", "alpha", "rule", "mean_risk", "q95", "share_risk_le_alpha", "share_risk_le_alpha_2se", "mean_cost"]].round(3).to_string(index=False))
    fig, ax = plt.subplots(1, len(CFG["CASC_ALPHAS"]), figsize=(COL2, 2.4), sharey=True); ax = np.atleast_1d(ax); pn = list(CFG["CASC_PATHS"])[1 if len(CFG["CASC_PATHS"]) > 1 else 0]
    for a_, alpha in zip(ax, CFG["CASC_ALPHAS"]):
        for rule, col in [("ltt", CC["casc_ab"]), ("crc", CC["fixed"])]:
            v = RAW[(RAW.path == pn) & (RAW.alpha == alpha) & (RAW.rule == rule)].test_risk.values
            a_.hist(v, bins=np.linspace(0, max(0.2, alpha * 2.2), 26), color=col, alpha=0.55, label={"ltt": "LTT (ours)", "crc": "expectation-only CRC"}[rule], edgecolor="k", linewidth=0.2)
        a_.axvline(alpha, color="k", ls="--", lw=0.9); a_.set_title(f"$\\alpha$={alpha} ({pn})", fontsize=7); a_.set_xlabel("test lost-correct rate")
    ax[0].set_ylabel(f"# of {B} re-splits"); ax[0].legend(fontsize=5)
    save_fig(fig, "fig_casc_resplits", RAW, "per-split test risk over random re-splits: the certificate holds for (almost) every split, not just on average")

def an_casc_conditional():
    """supervisor: conditional-coverage diagnostics beyond dataset (difficulty bins, question length) + Mondrian over DIFFICULTY."""
    st = casc_state(); At, Ac, delta = st["At"], st["Ac"], st["delta"]; CS = st["CS"]; lams, mus = CS["lams"], CS["mus"]; nl = len(lams)
    rows = []
    for alpha in (0.05, 0.10):
        if alpha not in CFG["CASC_ALPHAS"]: continue
        key = headline_key(alpha, True); path = path_idx(CFG["CASC_PATHS"][key[0]])
        fin, abst, cost = run_policy(At, st["s_t"], st["u_t"], key, st); lost = lost_correct(At["C"], fin, abst)
        e_edges = np.quantile(st["Ah"]["ent0"], [1 / 3, 2 / 3]); q_edges = np.quantile(st["Ah"]["qlen"], [1 / 3, 2 / 3])
        facs = {"dataset": At["ds"], "difficulty (stage-1 entropy tercile)": np.array(["low", "mid", "high"])[np.digitize(At["ent0"], e_edges)],
                "question length tercile": np.array(["short", "mid", "long"])[np.digitize(At["qlen"], q_edges)]}
        for fname, lev_ in facs.items():
            for lv in sorted(set(lev_)):
                m = lev_ == lv; lo, hi = wilson(int(lost[m].sum()), int(m.sum()))
                rows.append(dict(alpha=alpha, rule="global LTT", factor=fname, level=lv, n=int(m.sum()), risk=float(lost[m].mean()), lo=lo, hi=hi, cost=float(cost[m].mean()), abstain=float(abst[m].mean())))
        # Mondrian over difficulty: calibrate one (lambda, mu) per stage-1-entropy tercile, Bonferroni over bins
        nb = CFG["CASC_BINS"]; edges = np.quantile(st["Ah"]["ent0"], np.linspace(0, 1, nb + 1)[1:-1]); bc, bt = np.digitize(Ac["ent0"], edges), np.digitize(At["ent0"], edges)
        Lc, Cc = loss_cost_matrix(path, Ac, st["s_c"], st["u_c"], lams, mus, st["sel_cost"]); Lt, Ct = loss_cost_matrix(path, At, st["s_t"], st["u_t"], lams, mus, st["sel_cost"])
        allL, allC = [], []
        for b in range(nb):
            c = ltt_choose(Lc[bc == b], Cc[bc == b], nl, list(range(len(mus))), alpha, delta / nb); m = bt == b
            lo, hi = wilson(int(Lt[m, c].sum()), int(m.sum()))
            rows.append(dict(alpha=alpha, rule="Mondrian-LTT (difficulty bins)", factor="difficulty (stage-1 entropy tercile)", level=["low", "mid", "high"][b] if nb == 3 else str(b),
                             n=int(m.sum()), risk=float(Lt[m, c].mean()), lo=lo, hi=hi, cost=float(Ct[m, c].mean()), abstain=np.nan))
            allL.append(Lt[m, c]); allC.append(Ct[m, c])
        L_, C_ = np.concatenate(allL), np.concatenate(allC); lo, hi = wilson(int(L_.sum()), len(L_))
        rows.append(dict(alpha=alpha, rule="Mondrian-LTT (difficulty bins)", factor="ALL", level="ALL", n=len(L_), risk=float(L_.mean()), lo=lo, hi=hi, cost=float(C_.mean()), abstain=np.nan))
        lo, hi = wilson(int(lost.sum()), len(lost)); rows.append(dict(alpha=alpha, rule="global LTT", factor="ALL", level="ALL", n=len(lost), risk=float(lost.mean()), lo=lo, hi=hi, cost=float(cost.mean()), abstain=float(abst.mean())))
    T = pd.DataFrame(rows); save_table(T, "casc_conditional", "conditional coverage of the certificate: test risk per dataset / difficulty / question-length level for the global rule, and for Mondrian calibration over difficulty bins")
    print(T[T.factor.isin(["difficulty (stage-1 entropy tercile)", "ALL"])].round(3).to_string(index=False))
    fig, ax = plt.subplots(1, 2, figsize=(COL2, 2.5)); a0 = 0.05 if 0.05 in CFG["CASC_ALPHAS"] else CFG["CASC_ALPHAS"][0]
    d = T[(T.alpha == a0) & (T.factor == "dataset")]; xs = np.arange(len(d))
    ax[0].bar(xs, d.risk, 0.6, yerr=[d.risk - d.lo, d.hi - d.risk], color="#D55E00", edgecolor="k", linewidth=0.4, capsize=1.5); ax[0].axhline(a0, color="k", ls="--", lw=0.9)
    ax[0].set_xticks(xs); ax[0].set_xticklabels([DS_LABEL.get(x, x) for x in d.level], rotation=35, ha="right"); ax[0].set_ylabel("test lost-correct rate (95% Wilson)"); panel_label(ax[0], "(a)")
    for i, (rule, col) in enumerate([("global LTT", "#D55E00"), ("Mondrian-LTT (difficulty bins)", "#0072B2")]):
        d = T[(T.alpha == a0) & (T.rule == rule) & (T.factor == "difficulty (stage-1 entropy tercile)")]; xx = np.arange(len(d)) + (i - 0.5) * 0.35
        ax[1].bar(xx, d.risk, 0.33, yerr=[d.risk - d.lo, d.hi - d.risk], color=col, edgecolor="k", linewidth=0.4, capsize=1.5, label=rule)
    ax[1].axhline(a0, color="k", ls="--", lw=0.9); ax[1].set_xticks(np.arange(nb)); ax[1].set_xticklabels(["easy", "mid", "hard"][:nb]); ax[1].set_ylabel("test lost-correct rate"); ax[1].legend(fontsize=5); panel_label(ax[1], "(b)")
    save_fig(fig, "fig_casc_conditional", T, f"conditional coverage at alpha={a0}: (a) per dataset, (b) per difficulty bin for the global rule vs Mondrian calibration")

def an_casc_shift():
    """leave-one-dataset-out calibration (heads AND thresholds fitted without dataset d) + few-shot target-only recalibration."""
    st = casc_state(); Ah, Ac, At, delta = st["Ah"], st["Ac"], st["At"], st["delta"]; CS = st["CS"]; rows = []
    dsl = [d for d in DS_ORDER if (At["ds"] == d).sum() >= 30 and (Ac["ds"] == d).sum() >= 30]; rng = np.random.default_rng(1)
    for alpha in (0.05, 0.10):
        if alpha not in CFG["CASC_ALPHAS"]: continue
        key = headline_key(alpha, True); path = path_idx(CFG["CASC_PATHS"][key[0]]); pol = CS["pol"][key]
        for d in dsl:
            ti, ci = np.where(At["ds"] == d)[0], np.where(Ac["ds"] == d)[0]; At_d = sub_arrays(At, ti)
            # (a) in-distribution reference: global heads + global thresholds
            fin, abst, cost = cascade_sim(path, st["s_t"][ti], st["u_t"][ti], At_d["COST"], pol["lam"], pol["mu"], st["sel_cost"]); m = policy_metrics(At_d, fin, abst, cost)
            rows.append(dict(alpha=alpha, heldout=d, calibration="in-distribution (all 7 datasets)", n_test=len(ti), n_cal=int(len(Ac["C"]) - len(ci)), risk=m["lost_correct"], lo=m["lost_lo"], hi=m["lost_hi"], cost=m["mean_cost"], abstain=m["abstain_rate"]))
            # (b) leave-dataset-out: heads without d, thresholds calibrated on the other datasets only
            hi_ = np.where(Ah["ds"] != d)[0]; Hd = fit_heads(Ah, hi_); oth = np.where(Ac["ds"] != d)[0]; Ac_o = sub_arrays(Ac, oth)
            s_o, u_o = head_scores(Hd, Ac_o); lams, mus = cfg_grid(s_o); Lm, Cm = loss_cost_matrix(path, Ac_o, s_o, u_o, lams, mus, st["sel_cost"])
            c = ltt_choose(Lm, Cm, len(lams), list(range(len(mus))), alpha, delta); lam, mu = float(lams[c % len(lams)]), float(mus[c // len(lams)])
            s_d, u_d = head_scores(Hd, At_d); fin, abst, cost = cascade_sim(path, s_d, u_d, At_d["COST"], lam, mu, st["sel_cost"]); m = policy_metrics(At_d, fin, abst, cost)
            rows.append(dict(alpha=alpha, heldout=d, calibration="leave-dataset-out (heads + thresholds)", n_test=len(ti), n_cal=len(oth), risk=m["lost_correct"], lo=m["lost_lo"], hi=m["lost_hi"], cost=m["mean_cost"], abstain=m["abstain_rate"]))
            # (c) few-shot target-only recalibration (valid by itself, but with a small n)
            for nt in CFG["CASC_FEWSHOT"]:
                if nt > len(ci): continue
                sel_ = rng.choice(ci, nt, replace=False); A_t = sub_arrays(Ac, sel_); s_t_, u_t_ = head_scores(Hd, A_t); lams2, mus2 = cfg_grid(s_t_)
                Lm2, Cm2 = loss_cost_matrix(path, A_t, s_t_, u_t_, lams2, mus2, st["sel_cost"]); c2 = ltt_choose(Lm2, Cm2, len(lams2), list(range(len(mus2))), alpha, delta)
                fin, abst, cost = cascade_sim(path, s_d, u_d, At_d["COST"], float(lams2[c2 % len(lams2)]), float(mus2[c2 // len(lams2)]), st["sel_cost"]); m = policy_metrics(At_d, fin, abst, cost)
                rows.append(dict(alpha=alpha, heldout=d, calibration=f"few-shot target-only (n={nt})", n_test=len(ti), n_cal=nt, risk=m["lost_correct"], lo=m["lost_lo"], hi=m["lost_hi"], cost=m["mean_cost"], abstain=m["abstain_rate"]))
    T = pd.DataFrame(rows); save_table(T, "casc_shift", "distribution shift: calibrating on the other datasets (heads + thresholds) and testing on a held-out dataset; few-shot target-only recalibration as a remedy")
    S_ = T.groupby(["alpha", "calibration"]).agg(mean_risk=("risk", "mean"), worst_risk=("risk", "max"), share_heldout_le_alpha=("risk", lambda x: np.nan), mean_cost=("cost", "mean")).reset_index()
    for i, r_ in S_.iterrows():
        S_.loc[i, "share_heldout_le_alpha"] = float((T[(T.alpha == r_.alpha) & (T.calibration == r_.calibration)].risk <= r_.alpha).mean())
    save_table(S_, "casc_shift_summary", "shift summary: mean / worst held-out risk and share of held-out datasets whose risk is <= alpha"); print(S_.round(3).to_string(index=False))
    a0 = 0.10 if 0.10 in CFG["CASC_ALPHAS"] else CFG["CASC_ALPHAS"][-1]; fig, ax = plt.subplots(figsize=(COL2, 2.5)); cals = list(dict.fromkeys(T.calibration)); W = 0.8 / len(cals)
    for i, cn in enumerate(cals):
        d = T[(T.alpha == a0) & (T.calibration == cn)].set_index("heldout").reindex(dsl)
        ax.bar(np.arange(len(dsl)) + (i - (len(cals) - 1) / 2) * W, d.risk, W, yerr=[d.risk - d.lo, d.hi - d.risk], color=plt.cm.tab10(i), edgecolor="k", linewidth=0.3, capsize=1, label=cn)
    ax.axhline(a0, color="k", ls="--", lw=0.9); ax.set_xticks(range(len(dsl))); ax.set_xticklabels([DS_LABEL[x] for x in dsl], rotation=25, ha="right"); ax.set_ylabel(f"test lost-correct rate (alpha={a0})"); ax.legend(fontsize=4.6, ncol=2)
    save_fig(fig, "fig_casc_shift", T, "held-out-dataset risk under in-distribution vs leave-dataset-out vs few-shot calibration")

def an_decoded_baselines():
    """decoded accuracy of every baseline at 12.5 / 25 / 50 % tokens (official metrics) next to Q-EASE's nested sets; paired bootstrap on accuracy and harm."""
    recs = [o for o in load_dec(DEC_DIR / "test", base=True) if o.get("base")]
    if not recs: log("   no decoded baseline records - skipping"); return
    meths = sorted({m for o in recs for m in o["base"]}); rows = []; fr = CFG["DEC_BASE_FRACS"]; gi = {round(f, 6): i for i, f in enumerate(CFG["DEC_GRID"])}
    for f in fr:
        q_sc = np.array([o["stages"][gi[round(f, 6)]]["score"] for o in recs]); full_sc = np.array([o["stages"][-1]["score"] for o in recs]); cf = full_sc >= CFG["CORRECT_TAU"]
        pool = {"qease": (q_sc, np.array([o["stages"][gi[round(f, 6)]]["kfrac"] for o in recs]))}
        for m in meths:
            ok = [i for i, o in enumerate(recs) if o["base"].get(m, {}).get(str(f))]
            if len(ok) < 0.9 * len(recs): continue
            pool[m] = (np.array([recs[i]["base"][m][str(f)]["score"] for i in ok]), np.array([recs[i]["base"][m][str(f)]["kfrac"] for i in ok]), ok)
        for m, v in pool.items():
            sc, kf = v[0], v[1]; idx = np.arange(len(recs)) if m == "qease" else np.array(v[2]); fs, cf_ = full_sc[idx], cf[idx]
            harm = (cf_ & (sc < CFG["CORRECT_TAU"])).astype(float); q_s = q_sc[idx]; q_h = (cf_ & (q_s < CFG["CORRECT_TAU"])).astype(float)
            mu_, lo_, hi_, _ = boot_curve(sc); d_acc = paired_test(q_s, sc) if m != "qease" else (np.array([0.0]),) * 5; d_h = paired_test(harm, q_h) if m != "qease" else (np.array([0.0]),) * 5
            rows.append(dict(budget=f, method=sty(m)["label"] if m in METHOD_STYLE else m, n=len(idx), kept_token_frac=float(np.mean(kf)), accuracy=mu_[0], lo=lo_[0], hi=hi_[0],
                             acc_rel_full_pct=100 * sc.mean() / max(fs.mean(), 1e-9), harm_rate=float(harm.mean()), qease_acc_minus_method=float(d_acc[0][0]), acc_diff_lo=float(d_acc[1][0]), acc_diff_hi=float(d_acc[2][0]),
                             p_acc=float(d_acc[3][0]) if m != "qease" else np.nan, method_harm_minus_qease=float(d_h[0][0]), harm_diff_lo=float(d_h[1][0]), harm_diff_hi=float(d_h[2][0])))
    T = pd.DataFrame(rows); msk = T.p_acc.notna(); T.loc[msk, "p_holm_acc"] = holm(T.loc[msk, "p_acc"].values)
    save_table(T, "decoded_baselines", f"DECODED accuracy (official VQA-acc / ANLS / EM) of Q-EASE's nested sets and baselines at matched visual-token budgets on {len(recs)} test questions; positive qease_acc_minus_method = Q-EASE better")
    print(T[["budget", "method", "kept_token_frac", "accuracy", "acc_rel_full_pct", "harm_rate", "qease_acc_minus_method", "p_acc"]].round(3).to_string(index=False))
    fig, ax = plt.subplots(1, 2, figsize=(COL2, 2.6)); labs = list(dict.fromkeys(T.method))
    for i, lb in enumerate(labs):
        d = T[T.method == lb].sort_values("budget"); col = plt.cm.tab10(i % 10)
        ax[0].plot(d.budget, d.acc_rel_full_pct, marker="o", ms=3, color=col, label=lb); ax[1].plot(d.budget, d.harm_rate, marker="o", ms=3, color=col)
    ax[0].set_xlabel("visual-token budget"); ax[0].set_ylabel("accuracy retained (% of full model)"); ax[0].legend(fontsize=4.6); panel_label(ax[0], "(a)")
    ax[1].set_xlabel("visual-token budget"); ax[1].set_ylabel("harm rate P(full correct & wrong)"); panel_label(ax[1], "(b)")
    save_fig(fig, "fig_decoded_baselines", T, "decoded accuracy retained and harm rate vs budget for Q-EASE and all baselines (official metrics)")


def _ece(p, y, nb=10):
    p, y = np.asarray(p, float), np.asarray(y, float); q = np.quantile(p, np.linspace(0, 1, nb + 1)); b = np.clip(np.searchsorted(q[1:-1], p, side="right"), 0, nb - 1); e = 0.0
    for i in range(nb):
        m = b == i
        if m.any(): e += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(e)

def an_head_quality():
    """quality + calibration of the two heads on the test split (supervisor: SRH calibration plots on the sets that are actually used)."""
    st = casc_state(); At, s, u, S = st["At"], st["s_t"], st["u_t"], st["S"]; cfull = At["C"][:, -1]
    yh = (cfull[:, None] & ~At["C"][:, :-1]); yf = np.repeat(~cfull[:, None], S, 1); rows, rel = [], []
    rows.append(dict(head="harm P(full correct & stage wrong)", stage="all", n=int(yh.size), base_rate=float(yh.mean()), auroc=auroc(s[:, :-1].ravel(), yh.ravel()), ece=_ece(s[:, :-1].ravel(), yh.ravel())))
    for j in range(S - 1):
        rows.append(dict(head="harm", stage=CFG["DEC_GRID"][j], n=len(yh), base_rate=float(yh[:, j].mean()), auroc=auroc(s[:, j], yh[:, j]), ece=_ece(s[:, j], yh[:, j])))
    rows.append(dict(head="futility P(full wrong)", stage="all", n=int(yf.size), base_rate=float(yf.mean()), auroc=auroc(u.ravel(), yf.ravel()), ece=_ece(u.ravel(), yf.ravel())))
    for j in range(S):
        rows.append(dict(head="futility", stage=CFG["DEC_GRID"][j], n=len(yf), base_rate=float(yf[:, j].mean()), auroc=auroc(u[:, j], yf[:, j]), ece=_ece(u[:, j], yf[:, j])))
    sel_auc = auroc(At["FE"][:, :-1, FEATS.index("rhat")].ravel(), yh.ravel()); rows.append(dict(head="selector risk head (SRH) alone -> harm", stage="all", n=int(yh.size), base_rate=float(yh.mean()), auroc=sel_auc, ece=np.nan))
    T = pd.DataFrame(rows); save_table(T, "head_quality", "AUROC / ECE of the harm and futility heads on the test split (and the old SRH output as a harm predictor)"); print(T.round(3).to_string(index=False))
    fig, ax = plt.subplots(1, 2, figsize=(COL2, 2.5))
    for a_, (nm, p, y) in zip(ax, [("harm head", s[:, :-1].ravel(), yh.ravel()), ("futility head", u.ravel(), yf.ravel())]):
        q = np.unique(np.quantile(p, np.linspace(0, 1, 11))); b = np.clip(np.searchsorted(q[1:-1], p, side="right"), 0, len(q) - 2)
        for i in range(len(q) - 1):
            m = b == i
            if m.sum() < 5: continue
            lo, hi = wilson(int(y[m].sum()), int(m.sum())); rel.append(dict(head=nm, mean_pred=float(p[m].mean()), obs=float(y[m].mean()), lo=lo, hi=hi, n=int(m.sum())))
        d = pd.DataFrame([r_ for r_ in rel if r_["head"] == nm]); a_.errorbar(d.mean_pred, d.obs, yerr=[d.obs - d.lo, d.hi - d.obs], color="#D55E00", marker="o", ms=3, capsize=1.2)
        a_.plot([0, max(d.mean_pred.max(), d.obs.max())] * 1, [0, max(d.mean_pred.max(), d.obs.max())], "k:", lw=0.8); a_.set_xlabel("predicted probability"); a_.set_ylabel("observed frequency"); a_.set_title(nm, fontsize=7)
    save_fig(fig, "fig_head_calibration", pd.DataFrame(rel), "reliability diagrams of the harm and futility heads on the test split (Wilson 95% CIs)")

def an_ablation_report():
    recs = []
    for p in sorted(ABL_DIR.glob("eval_*.pt")): recs += torch.load(p, weights_only=False)
    if not recs: log("   no ablation records - skipping"); return
    names = sorted(set.intersection(*[set(r["js"]) for r in recs])); fr = CFG["ABL_FRACS"]; rows = []
    ref_of = lambda n: {"b": "b_ref", "a": "a_js", "c": "c_js"}.get(n[:1]) if n[:2] in ("b_", "a_", "c_") else None
    for n in names:
        for j, f in enumerate(fr):
            X = np.array([r["js"][n][j] for r in recs]); Fl = np.array([float(r["flip"][n][j]) for r in recs]); mu, lo, hi, _ = boot_curve(X); fm, flo, fhi, _ = boot_curve(Fl)
            row = dict(variant=n, label=ABL_LABEL.get(n, n), group=(n[:1] if n[:2] in ("b_", "a_", "c_") else "baseline"), frac=f, js=mu[0], js_lo=lo[0], js_hi=hi[0], flip=fm[0], flip_lo=flo[0], flip_hi=fhi[0], n=len(X))
            ref = ref_of(n)
            if ref in names and n != ref:
                Xr = np.array([r["js"][ref][j] for r in recs]); dm, dlo, dhi, p, _ = paired_test(X, Xr)
                row.update(reference=ref, js_minus_ref=dm[0], diff_lo=dlo[0], diff_hi=dhi[0], p_boot=p[0])
            rows.append(row)
    T = pd.DataFrame(rows); m = T.p_boot.notna() if "p_boot" in T else pd.Series(False, index=T.index)
    if m.any(): T.loc[m, "p_holm"] = holm(T.loc[m, "p_boot"].values)
    save_table(T, "ablations", f"label / architecture / loss / mask ablations: first-token JS of each trained selector on {len(recs)} test images (js_minus_ref > 0 = worse than its reference)");
    print(T[T.frac == fr[min(1, len(fr) - 1)]][[c for c in ["variant", "js", "js_lo", "js_hi", "js_minus_ref", "p_holm"] if c in T]].round(4).to_string(index=False))
    f0 = 0.25 if 0.25 in fr else fr[len(fr) // 2]; d = T[(T.frac == f0)].copy(); d = d.sort_values(["group", "js"])
    fig, ax = plt.subplots(figsize=(COL2, 3.0)); y = np.arange(len(d)); cols = {"b": "#D55E00", "a": "#0072B2", "c": "#009E73", "baseline": "#999999"}
    ax.barh(y, d.js, xerr=[d.js - d.js_lo, d.js_hi - d.js], color=[cols[g] for g in d.group], edgecolor="k", linewidth=0.3, capsize=1)
    ax.set_yticks(y); ax.set_yticklabels(d.label, fontsize=5.5); ax.set_xlabel(f"mean JS to the full model at {f0:.3g} budget (lower = better)"); ax.invert_yaxis()
    save_fig(fig, "fig_ablations", T, "ablations: supervision (JS / gold likelihood / answer distillation / attention / single-group / shuffled), architecture, loss weights, masks")
    for p in sorted(ABL_DIR.glob("omega*")):                                          # window size omega: compare at matched TOKEN budgets
        om = int(p.name.replace("omega", "")); rr = {}
        for w in (CFG["GROUP_WIN"], om):
            L_ = []
            for q in sorted(p.glob(f"eval_win{w}_*.pt")): L_ += torch.load(q, weights_only=False)
            if L_: rr[w] = L_
        if len(rr) == 2:
            rows = []
            for w, L_ in rr.items():
                JS, KF = np.stack([x["js"] for x in L_]), np.stack([x["kfrac"] for x in L_]); mu, lo, hi, _ = boot_curve(JS)
                for j in range(JS.shape[1]): rows.append(dict(window=w, budget_idx=j, kept_token_frac=KF[:, j].mean(), js=mu[j], lo=lo[j], hi=hi[j], mean_groups=float(np.mean([x["G"] for x in L_])), n=len(L_)))
            save_table(pd.DataFrame(rows), f"ablation_window_{om}", f"group window omega={CFG['GROUP_WIN']} vs {om} (selectors re-trained on the same {CFG['ABL_OMEGA_N']} images), JS vs kept-token fraction")

def an_additivity():
    """supervisor: S(M)=sum_j s_j cannot represent redundancy.  (1) pair accuracy of the additive score vs the SET-level risk head on probe masks; (2) greedy MMR vs additive top-k."""
    P = [r for r in load_probe_records() if r["split"] == "val"][:600]; rows = []
    if len(P) >= 50 and "full" in SELECTORS:
        import copy
        dev = DEVICES[0]; T = assemble(P, dev); m = copy.deepcopy(SELECTORS["full"]).to(dev).eval(); ia, ib = [x.to(dev) for x in pairs_from_rid(T["rid"])]; rid = T["rid"].to(dev)
        ca, cs, nv = defaultdict(float), defaultdict(float), defaultdict(float)
        with torch.no_grad():
            for i in range(0, len(P), 64):
                b = {k: v[i:i + 64] for k, v in T.items() if k != "rid"}
                with torch.autocast("cuda", dtype=torch.float16, enabled=dev.startswith("cuda")):
                    h, s = m.encode(b["gf"], b["gpos"], b["gv"], b["qe"], b["qv"]); rl = m.risk(h, b["gv"], b["masks"])
                S_ = torch.einsum("bkg,bg->bk", b["masks"].float(), s.float().masked_fill(~b["gv"], 0)); dj = b["js"][:, ib] - b["js"][:, ia]; valid = dj.abs() > CFG["RANK_MARGIN"]
                add_ok = ((S_[:, ia] - S_[:, ib] > 0) == (dj > 0)) & valid; srh_ok = ((rl.float()[:, ib] - rl.float()[:, ia] > 0) == (dj > 0)) & valid
                for r_ in range(len(RATIOS)):
                    sel = (rid[ia] == r_)[None].expand_as(valid) & valid
                    ca[r_] += float((add_ok & sel).sum()); cs[r_] += float((srh_ok & sel).sum()); nv[r_] += float(sel.sum())
        for r_ in range(len(RATIOS)):
            rows.append(dict(keep_ratio=RATIOS[r_], n_pairs=int(nv[r_]), additive_pair_acc=ca[r_] / max(nv[r_], 1), set_level_head_pair_acc=cs[r_] / max(nv[r_], 1)))
        tot = sum(nv.values()); rows.append(dict(keep_ratio="ALL", n_pairs=int(tot), additive_pair_acc=sum(ca.values()) / max(tot, 1), set_level_head_pair_acc=sum(cs.values()) / max(tot, 1)))
        save_table(pd.DataFrame(rows), "additivity_pair_accuracy", "probe-mask pair ranking accuracy (validation images): additive score sum_j s_j vs the set-level risk head -> how much is lost by assuming additivity")
        print(pd.DataFrame(rows).round(3).to_string(index=False)); del T; torch.cuda.empty_cache()
    if "qease_greedy" in AV:
        rows = []
        for key in ("js", "flip"):
            mu, lo, hi, p, dz = paired_test(arr(TEST, key, "qease_greedy"), arr(TEST, key, "qease"))
            for j in range(len(FR)): rows.append(dict(metric=key, frac=FR[j], greedy_minus_additive=mu[j], lo=lo[j], hi=hi[j], p_boot=p[j], beta=GREEDY_BETA, n=len(TEST)))
        T = pd.DataFrame(rows); T["p_holm"] = holm(T.p_boot.values); save_table(T, "greedy_vs_additive", f"redundancy-aware greedy (MMR, beta={GREEDY_BETA}) vs additive top-k on the test split (negative = greedy better)")

def an_selector_shift():
    recs = []
    for p in sorted(SHIFT_DIR.glob("eval_*.pt")): recs += torch.load(p, weights_only=False)
    if not recs: log("   no shift records - skipping"); return
    fr = CFG["ABL_FRACS"]; j = int(np.argmin(abs(np.array(fr) - 0.25))); rows = []
    for d in [x for x in DS_ORDER if any(r["ds"] == x for r in recs)] + ["ALL"]:
        R = recs if d == "ALL" else [r for r in recs if r["ds"] == d]
        if len(R) < 20: continue
        J = {n: np.array([r["js"][n][j] for r in R]) for n in ("seen", "heldout", "uniform", "random")}
        for n in J:
            mu, lo, hi, _ = boot_curve(J[n]); row = dict(heldout=d, selector={"seen": "trained on all datasets", "heldout": "trained WITHOUT this dataset", "uniform": "uniform grid", "random": "random"}[n],
                                                          n=len(R), js=mu[0], lo=lo[0], hi=hi[0])
            if n != "seen": dm, dlo, dhi, p, _ = paired_test(J[n], J["seen"]); row.update(js_minus_seen=dm[0], diff_lo=dlo[0], diff_hi=dhi[0], p_boot=p[0])
            rows.append(row)
    T = pd.DataFrame(rows); T["p_holm"] = np.nan; m = T.p_boot.notna(); T.loc[m, "p_holm"] = holm(T.loc[m, "p_boot"].values)
    save_table(T, "selector_shift", f"leave-one-dataset-out SELECTOR: first-token JS at {fr[j]:.3g} budget on images of the held-out dataset"); print(T[T.heldout == "ALL"].round(4).to_string(index=False))

def an_hires():
    hr = []
    for p in sorted((HIRES_DIR / "eval").glob("test_*.pt")): hr += torch.load(p, weights_only=False)
    rows = []
    if hr:
        lowres = {r["uid"]: r for r in TEST}; both = [r for r in hr if r["uid"] in lowres]; keys = sorted(set.intersection(*[set(r["js"]) for r in hr]))
        for jn, f in [(J12, 0.125), (J25, 0.25)]:
            for m in keys:
                X = np.array([r["js"][m][jn] for r in hr]); mu, lo, hi, _ = boot_curve(X); row = dict(resolution="high (" + "/".join(f"{k}:{v}" for k, v in CFG["HIRES_TOKENS"].items()) + " tokens)", method=sty(m)["label"], frac=f, n=len(hr), js=mu[0], lo=lo[0], hi=hi[0], flip=float(np.mean([r["flip"][m][jn] for r in hr])))
                rows.append(row)
                if both and all(m in lowres[r["uid"]]["js"] for r in both):
                    Y = np.array([lowres[r["uid"]]["js"][m][jn] for r in both]); mu2, lo2, hi2, _ = boot_curve(Y)
                    rows.append(dict(resolution="default caps (same images)", method=sty(m)["label"], frac=f, n=len(both), js=mu2[0], lo=lo2[0], hi=hi2[0], flip=float(np.mean([lowres[r["uid"]]["flip"][m][jn] for r in both]))))
        save_table(pd.DataFrame(rows), "hires_fidelity", "first-token fidelity at 4x the visual-token caps (selectors/baselines unchanged) vs the default caps on the same images")
        print(pd.DataFrame(rows)[lambda d: d.frac == 0.25].round(4).to_string(index=False))
    d = load_dec(HIRES_DIR / "dec")
    if len(d) >= 30 and get_casc() is not None:
        st = casc_state(); A = dec_arrays(d); s, u = head_scores(st["CS"]["H"], A); rows = []
        for alpha in CFG["CASC_ALPHAS"]:
            for ab in (False, True):
                key = headline_key(alpha, ab); fin, abst, cost = run_policy(A, s, u, key, st); m = policy_metrics(A, fin, abst, cost)
                fin2, abst2, cost2 = run_policy(st["At"], st["s_t"], st["u_t"], key, st); m2 = policy_metrics(st["At"], fin2, abst2, cost2)
                for lab, mm in (("high resolution (calibrated at default caps)", m), ("default caps (test)", m2)):
                    rows.append(dict(alpha=alpha, abstain=ab, setting=lab, n=mm["n"], lost_correct=mm["lost_correct"], lo=mm["lost_lo"], hi=mm["lost_hi"], accuracy=mm["accuracy"], accuracy_full=mm["accuracy_full"], mean_cost=mm["mean_cost"], abstain_rate=mm["abstain_rate"]))
        save_table(pd.DataFrame(rows), "hires_cascade", "the SAME calibrated cascade (heads + thresholds fitted at the default caps) applied to high-resolution inputs: resolution shift of the certificate"); print(pd.DataFrame(rows).round(3).to_string(index=False))

def an_casc_latency():
    p = TAB / "raw_cascade_latency.csv"
    if not p.exists(): log("   no cascade latency measurements - skipping"); return
    R = pd.read_csv(p); R["speedup_e2e"] = R.e2e_full_ms / R.e2e_cascade_ms; R["speedup_prefill"] = R.full_prefill_ms / R.cascade_ms
    mu, lo, hi, _ = boot_curve(R.speedup_e2e.values); mu2, lo2, hi2, _ = boot_curve(R.speedup_prefill.values)
    S = pd.DataFrame([dict(policy=R.policy.iloc[0], n=len(R), e2e_full_ms=R.e2e_full_ms.mean(), e2e_cascade_ms=R.e2e_cascade_ms.mean(), speedup_e2e_mean_of_ratios=mu[0], lo=lo[0], hi=hi[0],
                           speedup_e2e_ratio_of_means=R.e2e_full_ms.mean() / R.e2e_cascade_ms.mean(), speedup_prefill_mean=mu2[0], prefill_lo=lo2[0], prefill_hi=hi2[0],
                           abstain_rate=R.abstained.mean(), mean_final_frac=R.final_frac.mean())])
    save_table(S, "casc_latency", "REAL wall-clock of the executed cascade (selector + stage re-runs + heads) vs the full model, batch 1, one T4; prefill + vision tower, answer decoding excluded")
    print(S.round(3).to_string(index=False))
    fig, ax = plt.subplots(1, 2, figsize=(COL2, 2.4)); ax[0].hist(R.speedup_e2e, bins=20, color="#D55E00", edgecolor="k", linewidth=0.3); ax[0].axvline(1.0, color="k", ls="--", lw=0.8)
    ax[0].set_xlabel("end-to-end speed-up per question (x)"); ax[0].set_ylabel("# questions"); panel_label(ax[0], "(a)")
    g = R.groupby("final_frac").size() / len(R); ax[1].bar(range(len(g)), g.values, color="#0072B2", edgecolor="k", linewidth=0.3); ax[1].set_xticks(range(len(g))); ax[1].set_xticklabels([f"{x:.3g}" for x in g.index])
    ax[1].set_xlabel("budget at which the cascade stopped"); ax[1].set_ylabel("fraction of questions"); panel_label(ax[1], "(b)")
    save_fig(fig, "fig_casc_latency", R, "real executed-cascade latency: speed-up distribution and stopping budgets")

# ============================== ANALYSIS RUN
if "analyze" in CFG["STAGES"]:
    prepare_analysis_state()
    for _name, _fn in [("run_config", an_run_config), ("probe_stats", an_probe_stats), ("phase0", an_phase0), ("training_curves", an_hist),
                       ("fidelity+significance+faithfulness (all baselines)", an_fidelity), ("per_dataset", an_dataset), ("crc_main+headline (JS-based, legacy)", an_crc_main),
                       ("crc_validity_resampled", an_crc_resample), ("calibration_size", an_calsize), ("conditional_validity", an_conditional),
                       ("risk_head", an_riskhead), ("budget_vs_difficulty", an_budget), ("qualitative", an_qual), ("failure_cases", an_failures),
                       ("latency", an_latency), ("compute_equivalent", an_compute_equiv), ("compute_matched_vs_published", an_matched_compute),
                       ("CASCADE main (escalate or abstain)", an_casc_main), ("CASCADE re-split distributions (LTT vs CRC)", an_casc_resplit),
                       ("CASCADE conditional coverage + Mondrian over difficulty", an_casc_conditional), ("CASCADE distribution shift", an_casc_shift),
                       ("decoded baselines (official metrics)", an_decoded_baselines), ("head quality / calibration", an_head_quality),
                       ("ablations (labels/arch/loss/masks/window)", an_ablation_report), ("additivity + greedy set construction", an_additivity),
                       ("selector shift (leave-one-dataset-out)", an_selector_shift), ("high resolution", an_hires), ("cascade latency (real)", an_casc_latency),
                       ("finalize", an_finalize)]:
        run_block(_name, _fn)


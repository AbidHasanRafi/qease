"""Analysis blocks A6-A10: calibrated budgets, validity of the guarantee, calibration size, conditional validity, risk head."""
import matplotlib
matplotlib.use("Agg")                      # figures are written to disk
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from qease import state
from qease.analysis.common import sty
from qease.crc import arr, calibrate_fixed, crc_adaptive_lambda, first_ok, loss_mat, loss_tag, make_policy_idx, mondrian_idx, policy_eval, summarize_policy
from qease.reporting import BASELINES, COL2, DS_LABEL, LN2, POLICY_LABEL, panel_label, save_fig, save_table
from qease.stats import auroc, row_spearman, wilson


def an_crc_main():
    rows = []
    for kind, eps in state.LOSS_DEFS:
        for alpha in state.CFG["ALPHAS"]:
            for m in state.AV:
                pols = ["fixed"] + ((["adaptive", "mondrian"] + (["naive"] if kind == "js" else [])) if m.startswith("qease") else [])
                for pol in pols:
                    kidx = mondrian_idx(m, kind, eps, alpha, state.CAL, state.TEST) if pol == "mondrian" else make_policy_idx(m, kind, eps, alpha, state.CAL, state.TEST, pol)
                    lv, fv = policy_eval(state.TEST, m, kidx, kind, eps)
                    s = summarize_policy(lv, fv)
                    rows.append(dict(loss=loss_tag(kind, eps), alpha=alpha, method=sty(m)["label"], method_id=m, policy=POLICY_LABEL[pol], policy_id=pol,
                                     n_cal=len(state.CAL), **s, risk_leq_alpha=bool(s["test_risk"] <= alpha),
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
    for kind, eps in [("flip", None), ("js", state.EPS0)]:
        series = [("qease", "adaptive"), ("qease", "fixed"), ("qsim", "fixed"), ("uniform", "fixed"), ("random", "fixed")]
        if "qease_noq" in state.AV: series.insert(2, ("qease_noq", "adaptive"))
        fdat = []
        for m, pol in series:
            for a in state.CFG["ALPHA_SWEEP"]:
                kidx = make_policy_idx(m, kind, eps, a, state.CAL, state.TEST, pol); lv, fv = policy_eval(state.TEST, m, kidx, kind, eps)
                s = summarize_policy(lv, fv); fdat.append(dict(method=m, policy=pol, alpha=a, **s))
        FD = pd.DataFrame(fdat)
        fig, ax = plt.subplots(1, 2, figsize=(COL2, 2.6))
        for m, pol in series:
            d = FD[(FD.method == m) & (FD.policy == pol)]; s = sty(m)
            lab = s["label"] + (" - adaptive" if pol == "adaptive" else " - fixed $k$")
            kw = dict(color=s["color"], marker=s["marker"], ls="-" if pol == "adaptive" else "--", lw=1.6 if pol == "adaptive" else 1.0, label=lab)
            ax[0].plot(d.alpha, d.kept_frac, **kw); ax[0].fill_between(d.alpha, d.frac_lo, d.frac_hi, color=s["color"], alpha=0.12, lw=0)
            ax[1].errorbar(d.alpha, d.test_risk, yerr=[d.test_risk - d.risk_lo, d.risk_hi - d.test_risk], capsize=1.2, elinewidth=0.6, **kw)
        mx = max(state.CFG["ALPHA_SWEEP"]); ax[1].plot([0, mx], [0, mx], "k:", lw=0.9, label="target $\\alpha$")
        ax[0].set_ylabel("mean fraction of visual tokens kept"); ax[1].set_ylabel(f"realised test risk ({loss_tag(kind, eps)})")
        for a_, t in zip(ax, ["(a)", "(b)"]): a_.set_xlabel("target risk $\\alpha$"); panel_label(a_, t)
        ax[0].legend(fontsize=5.8, loc="lower left")
        nm = "fig_headline_flip" if kind == "flip" else f"fig_headline_js{eps:g}"
        save_fig(fig, nm, FD, f"tokens needed for a guaranteed risk alpha ({loss_tag(kind, eps)}): adaptive CRC vs fixed-k; and realised risk vs target")


def an_crc_resample():
    ALLR = state.CAL + state.TEST; n_cal = len(state.CAL); R = state.CFG["CRC_RESAMPLES"]; rng = np.random.default_rng(state.CFG["SEED"] + 7)
    pre = {m: dict(js=arr(ALLR, "js", m), flip=arr(ALLR, "flip", m), kf=arr(ALLR, "kfrac", m), rh=arr(ALLR, "rhat", m) if m.startswith("qease") else None) for m in state.AV}
    pols = [("qease", "adaptive"), ("qease", "fixed")] + ([("qease_noq", "adaptive")] if "qease_noq" in state.AV else []) + [(b, "fixed") for b in BASELINES if b in state.AV]
    rows = []
    for rep in range(R):
        perm = rng.permutation(len(ALLR)); ci, ti = perm[:n_cal], perm[n_cal:]
        for kind, eps in [("flip", None), ("js", state.EPS0)]:
            for alpha in state.CFG["ALPHAS"]:
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
    a0 = 0.10 if 0.10 in state.CFG["ALPHAS"] else state.CFG["ALPHAS"][len(state.CFG["ALPHAS"]) // 2]
    fig, ax = plt.subplots(1, 2, figsize=(COL2, 2.6)); fd = []
    for a_, (kind, eps), t in zip(ax, [("flip", None), ("js", state.EPS0)], ["(a)", "(b)"]):
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


def an_calsize():
    sizes = [s for s in state.CFG["CAL_SIZES"] if s < len(state.CAL)] + [len(state.CAL)]
    rng = np.random.default_rng(state.CFG["SEED"] + 11); a0 = 0.10; rows = []
    for kind, eps in [("flip", None), ("js", state.EPS0)]:
        for s in sizes:
            for rep in range(state.CFG["CAL_SIZE_REPS"] if s < len(state.CAL) else 1):
                sub = [state.CAL[i] for i in rng.choice(len(state.CAL), s, replace=False)] if s < len(state.CAL) else state.CAL
                for m, pol in [("qease", "adaptive"), ("qease", "fixed"), ("qsim", "fixed"), ("random", "fixed")]:
                    if m not in state.AV: continue
                    kidx = make_policy_idx(m, kind, eps, a0, sub, state.TEST, pol); lv, fv = policy_eval(state.TEST, m, kidx, kind, eps)
                    rows.append(dict(loss=loss_tag(kind, eps), n_cal=s, rep=rep, method=m, policy=pol, risk=float(lv.mean()), kept_frac=float(fv.mean())))
    D = pd.DataFrame(rows)
    S = D.groupby(["loss", "n_cal", "method", "policy"]).agg(risk_mean=("risk", "mean"), risk_std=("risk", "std"), frac_mean=("kept_frac", "mean"), frac_std=("kept_frac", "std")).reset_index()
    S["alpha"] = a0; save_table(S.assign(method=S.method.map(lambda x: sty(x)["label"])), "calibration_size_ablation", f"effect of calibration-set size (alpha={a0}); std over random subsets")
    fig, ax = plt.subplots(1, 2, figsize=(COL2, 2.5)); lt = loss_tag("js", state.EPS0)
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


def an_conditional():
    a0 = 0.10; rows = []
    for kind, eps in state.LOSS_DEFS:
        for m in [x for x in state.AV if x in ("qease", "qease_noq")]:
            Lt, Kf = loss_mat(state.TEST, m, kind, eps), arr(state.TEST, "kfrac", m)
            kg = make_policy_idx(m, kind, eps, a0, state.CAL, state.TEST, "adaptive"); km = mondrian_idx(m, kind, eps, a0, state.CAL, state.TEST)
            kfix = make_policy_idx(m, kind, eps, a0, state.CAL, state.TEST, "fixed")
            for d in state.DSL:
                ti = np.array([i for i, r in enumerate(state.TEST) if r["ds"] == d])
                if len(ti) < 5: continue
                cal_wo = [r for r in state.CAL if r["ds"] != d]
                klo = make_policy_idx(m, kind, eps, a0, cal_wo, [state.TEST[i] for i in ti], "adaptive") if len(cal_wo) > 20 else None
                pols = [("adaptive (global CRC)", kg[ti]), ("adaptive (per-dataset CRC)", km[ti]), ("fixed-$k$ (global CRC)", kfix[ti])]
                if klo is not None: pols.append(("adaptive (leave-dataset-out CRC)", klo))
                for pn, kk in pols:
                    lv, fv = Lt[ti, kk], Kf[ti, kk]; lo, hi = wilson(lv.sum(), len(lv))
                    rows.append(dict(loss=loss_tag(kind, eps), alpha=a0, dataset=DS_LABEL[d], n_test=len(ti), method=sty(m)["label"], method_id=m,
                                     policy=pn, test_risk=float(lv.mean()), risk_lo=lo, risk_hi=hi, kept_frac=float(fv.mean()), risk_leq_alpha=bool(lv.mean() <= a0)))
    T = pd.DataFrame(rows); save_table(T.drop(columns="method_id"), "conditional_validity", "per-dataset realised risk under global / per-dataset / leave-dataset-out calibration (marginal CRC does NOT promise per-dataset validity)")
    lt = loss_tag("js", state.EPS0)
    sub = T[(T["loss"] == lt) & (T.method_id == "qease")]
    pols = ["adaptive (global CRC)", "adaptive (per-dataset CRC)", "adaptive (leave-dataset-out CRC)", "fixed-$k$ (global CRC)"]
    cols = ["#D55E00", "#009E73", "#0072B2", "#7F7F7F"]
    fig, ax = plt.subplots(1, 2, figsize=(COL2, 2.5)); W = 0.8 / len(pols)
    for i, (p, c) in enumerate(zip(pols, cols)):
        d = sub[sub.policy == p].set_index("dataset").reindex([DS_LABEL[x] for x in state.DSL]); x = np.arange(len(state.DSL)) + (i - (len(pols) - 1) / 2) * W
        ax[0].bar(x, d.test_risk, W, yerr=[(d.test_risk - d.risk_lo).fillna(0), (d.risk_hi - d.test_risk).fillna(0)], color=c, edgecolor="k", linewidth=0.4, capsize=1, error_kw=dict(lw=0.5), label=p)
        ax[1].bar(x, d.kept_frac, W, color=c, edgecolor="k", linewidth=0.4)
    ax[0].axhline(a0, color="k", ls=":", lw=0.9); ax[0].set_ylabel(f"realised risk ({lt}), $\\alpha$={a0}"); ax[1].set_ylabel("mean fraction of tokens kept")
    for a_, t in zip(ax, ["(a)", "(b)"]): a_.set_xticks(range(len(state.DSL))); a_.set_xticklabels([DS_LABEL[x] for x in state.DSL], rotation=30, ha="right"); panel_label(a_, t)
    ax[0].legend(fontsize=5.5, ncol=2, loc="upper center", bbox_to_anchor=(1.1, 1.3))
    save_fig(fig, "fig_conditional_validity", sub.drop(columns="method_id"), "per-dataset validity of the calibrated budget (marginal vs per-dataset vs leave-dataset-out calibration)")


def an_riskhead():
    rows, rel = [], []
    for m in [x for x in state.AV if x.startswith("qease")]:
        P, J, Fl = arr(state.TEST, "rhat", m), arr(state.TEST, "js", m), arr(state.TEST, "flip", m)
        T_ = np.clip(J / LN2, 0, 1)
        within = row_spearman(P, T_)
        pooled = float(pd.Series(P.ravel()).corr(pd.Series(T_.ravel()), method="spearman"))
        row = dict(method=sty(m)["label"], pooled_spearman=pooled, within_sample_spearman=float(np.nanmean(within)),
                   auroc_flip=auroc(P.ravel(), Fl.ravel()), brier_vs_js=float(np.mean((P - T_) ** 2)), mae_vs_js=float(np.mean(abs(P - T_))))
        for e in state.CFG["EPS_LIST"]: row[f"auroc_JS>{e:g}"] = auroc(P.ravel(), (J > e).ravel())
        qs = pd.qcut(P.ravel(), 10, labels=False, duplicates="drop"); ece = 0.0
        for b in np.unique(qs):
            sel = qs == b; mp, mt_, se = P.ravel()[sel].mean(), T_.ravel()[sel].mean(), T_.ravel()[sel].std() / math.sqrt(sel.sum())
            ece += sel.mean() * abs(mp - mt_); rel.append(dict(method=m, bin=int(b), mean_predicted=mp, mean_true=mt_, sem=se, n=int(sel.sum())))
        row["ECE_10bin"] = float(ece); rows.append(row)
    save_table(pd.DataFrame(rows), "risk_head_quality", "does the risk head rank/calibrate the true answer-distribution shift? (test split)")
    R = pd.DataFrame(rel)
    fig, ax = plt.subplots(1, 2, figsize=(COL2, 2.6))
    for m in R.method.unique():
        d = R[R.method == m]; s = sty(m); ax[0].errorbar(d.mean_predicted, d.mean_true, yerr=d["sem"], color=s["color"], marker=s["marker"], ls=s["ls"], capsize=1, elinewidth=0.5, label=s["label"])
    mx = float(R[["mean_predicted", "mean_true"]].max().max()); ax[0].plot([0, mx], [0, mx], "k:", lw=0.9, label="perfect")
    ax[0].set_xlabel("predicted risk (JS / ln 2)"); ax[0].set_ylabel("observed risk (JS / ln 2)"); ax[0].legend(); ax[0].set_xscale("symlog", linthresh=1e-3); ax[0].set_yscale("symlog", linthresh=1e-3); panel_label(ax[0], "(a)")
    Q = pd.DataFrame(rows); mets = [c for c in Q.columns if c.startswith("auroc")]
    W = 0.8 / len(Q)
    for i, rr in Q.iterrows():
        m_id = [k for k in state.AV if sty(k)["label"] == rr["method"]][0]
        ax[1].bar(np.arange(len(mets)) + (i - (len(Q) - 1) / 2) * W, [rr[c] for c in mets], W, color=sty(m_id)["color"], edgecolor="k", linewidth=0.4, label=rr["method"])
    ax[1].set_xticks(range(len(mets))); ax[1].set_xticklabels([c.replace("auroc_", "AUROC\n") for c in mets], fontsize=6); ax[1].set_ylim(0.4, 1.0); ax[1].axhline(0.5, color="k", ls=":", lw=0.8)
    ax[1].set_ylabel("AUROC of risk head"); ax[1].legend(); panel_label(ax[1], "(b)")
    save_fig(fig, "fig_risk_head", pd.concat([R.assign(panel="a"), Q.melt(id_vars="method", var_name="metric", value_name="value").assign(panel="b")]), "risk-head reliability diagram and discrimination (AUROC)")

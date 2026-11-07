"""Analysis blocks A4-A5: fidelity versus budget, significance tests, per-dataset breakdown."""
import matplotlib
matplotlib.use("Agg")                      # figures are written to disk
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from qease import state
from qease.analysis.common import BOOTM, draw_curve, sty
from qease.crc import arr
from qease.reporting import BASELINES, COL2, DS_LABEL, frac_ticks, panel_label, save_fig, save_table
from qease.stats import boot_curve, frac_at_target, holm, paired_test, trap_rows


def an_fidelity():
    metrics = [("js", "mean JS to full model (nats)"), ("flip", "answer-flip rate"),
               ("js_del", "JS after DELETING top-$k$ groups"), ("flip_del", "flip rate after DELETING top-$k$")]
    rows = []
    for m in state.AV:
        kf = arr(state.TEST, "kfrac", m).mean(0)
        for key, _ in metrics:
            X = arr(state.TEST, key, m); mu, lo, hi, M = boot_curve(X); BOOTM[(m, key)] = M
            for j in range(len(state.FR)):
                rows.append(dict(method=m, metric=key, frac=state.FR[j], kept_token_frac=kf[j], mean=mu[j], lo=lo[j], hi=hi[j], n=len(X)))
    FID = pd.DataFrame(rows)
    save_table(FID.assign(method=FID.method.map(lambda x: sty(x)["label"])), "fidelity_by_budget",
               "main result: fidelity to the full model vs visual-token budget, 95% bootstrap CIs (test split)")
    # --- Fig: fidelity curves
    fig, ax = plt.subplots(1, 2, figsize=(COL2, 2.6))
    for m in state.AV:
        draw_curve(ax[0], FID[(FID.method == m) & (FID.metric == "js")], m); draw_curve(ax[1], FID[(FID.method == m) & (FID.metric == "flip")], m)
    for a, yl, t in zip(ax, ["mean JS to full model (nats)", "answer-flip rate (first token)"], ["(a)", "(b)"]):
        frac_ticks(a, state.FR); a.set_xlabel("fraction of visual groups kept"); a.set_ylabel(yl); panel_label(a, t)
    ax[0].set_yscale("symlog", linthresh=1e-3); ax[0].legend(loc="lower left", fontsize=5.5)   # lower-left is empty (curves sit at the top)
    save_fig(fig, "fig_fidelity_curves", FID[FID.metric.isin(["js", "flip"])], "fidelity vs budget (test), shaded = 95% bootstrap CI")
    # --- Fig: insertion / deletion (evidence faithfulness)
    fig, ax = plt.subplots(1, 2, figsize=(COL2, 2.6))
    for m in state.AV:
        draw_curve(ax[0], FID[(FID.method == m) & (FID.metric == "js")], m); draw_curve(ax[1], FID[(FID.method == m) & (FID.metric == "js_del")], m)
    for a, yl, t in zip(ax, ["JS: keep top-$k$ (insertion, lower=better)", "JS: drop top-$k$ (deletion, higher=better)"], ["(a)", "(b)"]):
        frac_ticks(a, state.FR); a.set_xlabel("fraction of groups inserted / deleted"); a.set_ylabel(yl); panel_label(a, t)
    ax[0].set_yscale("symlog", linthresh=1e-3); ax[0].legend(loc="upper right")
    save_fig(fig, "fig_insertion_deletion", FID[FID.metric.isin(["js", "js_del"])], "insertion / deletion faithfulness curves")
    # --- AUC table
    rows = []
    for m in state.AV:
        for key, nm in [("js", "insertion AUC (JS, lower=better)"), ("flip", "insertion AUC (flip, lower=better)"),
                        ("js_del", "deletion AUC (JS, higher=better)"), ("flip_del", "deletion AUC (flip, higher=better)")]:
            a_ = trap_rows(arr(state.TEST, key, m), state.FR) / (state.FR[-1] - state.FR[0]); mu, lo, hi, _ = boot_curve(a_)
            rows.append(dict(method=sty(m)["label"], quantity=nm, auc=mu[0], lo=lo[0], hi=hi[0]))
    save_table(pd.DataFrame(rows), "auc_insertion_deletion", "normalised area under insertion/deletion curves with 95% CI")
    # --- paired significance: Q-EASE vs every other method (Holm-corrected over the whole family)
    rows = []
    for b in [m for m in state.AV if m != "qease"]:
        for key in ("js", "flip"):
            mu, lo, hi, p, dz = paired_test(arr(state.TEST, key, "qease"), arr(state.TEST, key, b))
            Xa, Xb = arr(state.TEST, key, "qease"), arr(state.TEST, key, b)
            for j in range(len(state.FR)):
                rows.append(dict(baseline=sty(b)["label"], metric=key, frac=state.FR[j], mean_diff_qease_minus_baseline=mu[j], lo=lo[j], hi=hi[j],
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
            fb = {m: np.array([frac_at_target(state.FR, row, t) for row in BOOTM[(m, key)]]) for m in state.AV}
            best_base = np.nanmin(np.stack([fb[b] for b in BASELINES if b in fb]), axis=0) if any(b in fb for b in BASELINES) else None
            for m in state.AV:
                pt = frac_at_target(state.FR, FID[(FID.method == m) & (FID.metric == key)].sort_values("frac")["mean"].values, t)
                lo, hi = (np.nanpercentile(fb[m], [2.5, 97.5]) if np.isfinite(fb[m]).any() else (np.nan, np.nan))
                sav = (1 - fb[m] / best_base) if (best_base is not None and m.startswith("qease")) else None
                rows.append(dict(metric=("mean flip rate" if key == "flip" else "mean JS (nats)"), target=t, method=sty(m)["label"], frac_needed=pt,
                                 lo=lo, hi=hi, saving_vs_best_baseline_pct=(100 * np.nanmedian(sav) if sav is not None and np.isfinite(sav).any() else np.nan),
                                 saving_lo=(100 * np.nanpercentile(sav, 2.5) if sav is not None and np.isfinite(sav).any() else np.nan),
                                 saving_hi=(100 * np.nanpercentile(sav, 97.5) if sav is not None and np.isfinite(sav).any() else np.nan)))
    save_table(pd.DataFrame(rows), "tokens_at_matched_fidelity", "visual-token fraction needed to reach a target mean fidelity (lower is better); bootstrap CIs")
    print(pd.DataFrame(rows).round(3).to_string(index=False))


def an_dataset():
    rows, fdat = [], []
    for d in state.DSL:
        sub = [r for r in state.TEST if r["ds"] == d]
        for m in state.AV:
            for j, tag in [(state.J12, "12.5%"), (state.J25, "25%")]:
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
        W = 0.8 / len(state.AV)
        for i, m in enumerate(state.AV):
            d = T[(T.method_id == m) & (T.budget == "25%") & (T.metric == key)].set_index("dataset").reindex([DS_LABEL[x] for x in state.DSL])
            x = np.arange(len(state.DSL)) + (i - (len(state.AV) - 1) / 2) * W
            a.bar(x, d["mean"], W, yerr=[d["mean"] - d["lo"], d["hi"] - d["mean"]], color=sty(m)["color"], edgecolor="k", linewidth=0.4, capsize=1.2,
                  error_kw=dict(lw=0.6), label=sty(m)["label"])
        a.set_xticks(np.arange(len(state.DSL))); a.set_xticklabels([DS_LABEL[x] for x in state.DSL]); a.set_ylabel(yl)
    ax[0].legend(ncol=len(state.AV), loc="upper center", bbox_to_anchor=(0.5, 1.28), fontsize=6); panel_label(ax[0], "(a)"); panel_label(ax[1], "(b)")
    save_fig(fig, "fig_per_dataset_25pct", T[T.budget == "25%"].drop(columns="method_id"), "per-dataset fidelity at 25% budget (95% CI)")

"""Analysis blocks A13-A14: generation accuracy, latency, compute-matched comparisons with FastV / SparseVLM."""
import matplotlib
matplotlib.use("Agg")                      # figures are written to disk
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from qease import state
from qease.analysis.common import sty
from qease.crc import arr
from qease.reporting import COL2, DS_LABEL, METHOD_STYLE, panel_label, save_fig, save_table
from qease.stats import boot_curve, holm, paired_test
from qease.utils import log


def an_gen():
    files = sorted((state.WORK / "gen").glob("*.json"))
    if not files: log("   no generation results - skipping"); return
    G_ = [json.load(open(f)) for f in files]
    cfgs = [c for c in G_[0]["acc"] if all(c in g["acc"] for g in G_)]
    ds_list = [d for d in state.DS_ORDER if any(g["ds"] == d for g in G_)]
    A = pd.DataFrame([{"ds": g["ds"], **{c: g["acc"][c] for c in cfgs}} for g in G_])
    K = pd.DataFrame([{"ds": g["ds"], **{c: g["kfrac"][c] for c in cfgs}} for g in G_])
    AG = pd.DataFrame([{"ds": g["ds"], **{c: g["agree"].get(c, np.nan) for c in cfgs}} for g in G_])
    rows = []
    for scope in ["ALL"] + ds_list:
        idx = np.arange(len(A)) if scope == "ALL" else np.where(A.ds == scope)[0]
        full = A.loc[idx, "full"].values
        for c in cfgs:
            x = A.loc[idx, c].values; mu, lo, hi, M = boot_curve(x)
            rng = np.random.default_rng(state.CFG["SEED"]); bi = rng.integers(0, len(idx), size=(state.CFG["BOOT_B"], len(idx)))
            ratio = np.array([x[i].mean() / max(full[i].mean(), 1e-9) for i in bi])
            rows.append(dict(scope=DS_LABEL.get(scope, scope), n=len(idx), config=c, accuracy=mu[0], lo=lo[0], hi=hi[0], mean_kept_frac=K.loc[idx, c].mean(),
                             relative_to_full_pct=100 * x.mean() / max(full.mean(), 1e-9), rel_lo=100 * np.percentile(ratio, 2.5), rel_hi=100 * np.percentile(ratio, 97.5),
                             agreement_with_full=AG.loc[idx, c].mean() if c != "full" else 1.0))
    T = pd.DataFrame(rows); save_table(T, "generation_accuracy", "greedy-decoded accuracy (VQA-acc / ANLS / exact) and agreement with the full model, fixed + adaptive budgets, 95% CIs")
    # paired tests: Q-EASE vs every other method at the same budget tag
    rows = []
    tags = sorted({c.split("@", 1)[1] for c in cfgs if "@" in c})
    for tg in tags:
        q = f"qease@{tg}"
        if q not in cfgs: continue
        for b in [c.split("@")[0] for c in cfgs if c.endswith("@" + tg) and not c.startswith("qease@")]:
            mu, lo, hi, p, dz = paired_test(A[q].values, A[f"{b}@{tg}"].values)
            rows.append(dict(budget=tg, baseline=b, mean_acc_diff_qease_minus_baseline=mu[0], lo=lo[0], hi=hi[0], p_boot=p[0], effect_size_dz=dz[0], n=len(A)))
    if rows:
        P = pd.DataFrame(rows); P["p_holm"] = holm(P.p_boot.values); save_table(P, "generation_paired_tests", "paired bootstrap tests on accuracy (positive = Q-EASE better); equal per-sample token budgets for adaptive tags")
    # figure: accuracy retained vs tokens kept
    fig, ax = plt.subplots(1, 2, figsize=(COL2, 2.6)); All = T[T.scope == "ALL"]; fd = []
    for m in ["qease", "qease_noq", "fastv", "visionzip", "qsim", "uniform", "random"]:
        d = All[All.config.str.startswith(m + "@") & ~All.config.str.contains("adaptive") & ~All.config.str.contains("@sv")].sort_values("mean_kept_frac")
        if not len(d): continue
        d = pd.concat([d, pd.DataFrame([dict(mean_kept_frac=1.0, relative_to_full_pct=100.0, rel_lo=100.0, rel_hi=100.0)])]); s = sty(m)
        ax[0].plot(d.mean_kept_frac, d.relative_to_full_pct, color=s["color"], marker=s["marker"], ls=s["ls"], label=s["label"])
        ax[0].fill_between(d.mean_kept_frac, d.rel_lo, d.rel_hi, color=s["color"], alpha=0.12, lw=0); fd.append(d.assign(method=m, panel="a"))
        a_ = All[All.config.str.startswith(m + "@adaptive")]
        if len(a_): ax[0].scatter(a_.mean_kept_frac, a_.relative_to_full_pct, color=s["color"], marker="*", s=40, edgecolor="k", linewidths=0.4, zorder=5); fd.append(a_.assign(method=m, panel="a-adaptive"))
    ax[0].set_xlabel("mean fraction of visual tokens kept"); ax[0].set_ylabel("accuracy retained (% of full model)"); ax[0].legend(); panel_label(ax[0], "(a)")
    sub = T[(T.scope != "ALL") & T.config.str.endswith("@0.25")]
    ms_ = [m for m in ["qease", "fastv", "visionzip", "qsim", "uniform", "random"] if f"{m}@0.25" in cfgs]; W = 0.8 / max(1, len(ms_))
    for i, m in enumerate(ms_):
        d = sub[sub.config == f"{m}@0.25"].set_index("scope").reindex([DS_LABEL[x] for x in ds_list])
        ax[1].bar(np.arange(len(ds_list)) + (i - (len(ms_) - 1) / 2) * W, d.relative_to_full_pct, W, color=sty(m)["color"], edgecolor="k", linewidth=0.4, label=sty(m)["label"])
        fd.append(d.reset_index().assign(method=m, panel="b"))
    ax[1].set_xticks(range(len(ds_list))); ax[1].set_xticklabels([DS_LABEL[x] for x in ds_list], rotation=30, ha="right"); ax[1].set_ylabel("accuracy retained at 25% budget (%)"); panel_label(ax[1], "(b)")
    save_fig(fig, "fig_generation_accuracy", pd.concat(fd, ignore_index=True), "generation accuracy retained vs token budget (stars = calibrated adaptive; baselines at the same per-sample k)")


def an_latency():
    p = state.TAB / "raw_latency.csv"
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
    save_table(S, "latency_compute", f"physically pruned prefill latency / FLOPs / memory (GPU {state.DEVICES[0]}, batch 1); selector cost included in end-to-end speed-up")
    print(S.round(2).to_string(index=False))
    fig, ax = plt.subplots(1, 2, figsize=(COL2, 2.5)); x = np.arange(len(S))
    ax[0].bar(x, S.enc_ms, color="#999999", edgecolor="k", lw=0.4, label="vision tower + projector")
    ax[0].bar(x, S.selector_ms, bottom=S.enc_ms, color="#D55E00", edgecolor="k", lw=0.4, label="Q-EASE selector")
    ax[0].bar(x, S.prefill_ms, bottom=S.enc_ms + S.selector_ms, color="#0072B2", edgecolor="k", lw=0.4, label="LLM prefill")
    ax[0].set_xticks(x); ax[0].set_xticklabels([f"{f:g}" for f in S.frac]); ax[0].set_xlabel("fraction of groups kept"); ax[0].set_ylabel("latency (ms)"); ax[0].legend(); panel_label(ax[0], "(a)")
    ax[1].plot(S.frac, S.prefill_speedup, "o-", color="#0072B2", label="LLM prefill"); ax[1].plot(S.frac, S.end_to_end_speedup_incl_selector, "s-", color="#D55E00", label="end-to-end (incl. selector)")
    ax[1].plot(S.frac, 1 / S.frac, "k:", label="ideal $1/f$"); ax[1].set_xlabel("fraction of groups kept"); ax[1].set_ylabel("speed-up vs full"); ax[1].invert_xaxis(); ax[1].legend(); panel_label(ax[1], "(b)")
    save_fig(fig, "fig_latency", S, "measured latency breakdown and speed-up of physically pruned prefill")


def an_compute_equiv():
    if "fastv" not in state.AV or not all("fastv_flops_frac" in r and "L" in r for r in state.TEST):
        log("   FastV records missing - skipping compute-equivalent table"); return
    L = np.array([r["L"] for r in state.TEST], float); nv = np.array([r["n_vis"] for r in state.TEST], float)
    fv_eff = np.stack([r["fastv_flops_frac"] for r in state.TEST]).mean(0)
    rows = []
    for j, f in enumerate(state.FR):
        q_tok = arr(state.TEST, "kfrac", "qease")[:, j]
        q_eff = ((L - nv + q_tok * nv) / L).mean()                     # Q-EASE prunes before layer 0 -> whole prefill shrinks
        for m in state.AV:
            kf = arr(state.TEST, "kfrac", m)[:, j].mean()
            eff = fv_eff[j] if m == "fastv" else ((L - nv + arr(state.TEST, "kfrac", m)[:, j] * nv) / L).mean()
            rows.append(dict(nominal_budget=f, method=sty(m)["label"], visual_tokens_kept_frac=kf, effective_prefill_flops_frac=eff,
                             mean_js=arr(state.TEST, "js", m)[:, j].mean(), flip_rate=arr(state.TEST, "flip", m)[:, j].mean()))
    T = pd.DataFrame(rows)
    save_table(T, "compute_equivalent_comparison", "fidelity vs EFFECTIVE prefill FLOPs fraction (FastV keeps layers < K at full length, so its compute is higher than its token fraction)")
    fig, ax = plt.subplots(1, 2, figsize=(COL2, 2.6))
    for m in state.AV:
        d = T[T.method == sty(m)["label"]].sort_values("effective_prefill_flops_frac"); s = sty(m)
        for a_, col in zip(ax, ["mean_js", "flip_rate"]):
            a_.plot(d.effective_prefill_flops_frac, d[col], color=s["color"], marker=s["marker"], ls=s["ls"], label=s["label"])
    for a_, yl, t in zip(ax, ["mean JS to full model (nats)", "answer-flip rate"], ["(a)", "(b)"]):
        a_.set_xlabel("effective prefill FLOPs (fraction of full model)"); a_.set_ylabel(yl); panel_label(a_, t)
    ax[0].set_yscale("symlog", linthresh=1e-3); ax[0].legend()
    save_fig(fig, "fig_compute_equivalent", T, "fidelity vs effective prefill compute (fair axis when comparing in-LLM and pre-LLM pruning)")


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
    if "fastv" in state.AV and all("fastv_matched" in r for r in state.TEST):
        names = [m for m in ("qease", "qease_noq", "qsim", "uniform", "random") if all(m in r["fastv_matched"] for r in state.TEST)]
        matched = {m: {k: np.stack([r["fastv_matched"][m][k] for r in state.TEST]).astype(float) for k in ("js", "flip", "kfrac")} for m in names}
        add("fastv", [f"{f:g}" for f in state.FR], np.stack([r["fastv_flops_frac"] for r in state.TEST]).mean(0), arr(state.TEST, "js", "fastv"), arr(state.TEST, "flip", "fastv"),
            arr(state.TEST, "kfrac", "fastv"), matched, "group methods get the visual-token budget that equals FastV's effective prefill FLOPs, per sample")
    if all("sparse" in r for r in state.TEST) and state.TEST:
        sp0 = state.TEST[0]["sparse"]; names = [m for m in ("qease", "qease_noq", "qsim", "uniform", "random") if all(m in r["sparse"]["matched"] for r in state.TEST)]
        matched = {m: {k: np.stack([r["sparse"]["matched"][m][k] for r in state.TEST]).astype(float) for k in ("js", "flip", "kfrac")} for m in names}
        add("sparsevlm", [str(x) for x in sp0["retain"]], np.stack([r["sparse"]["eff"] for r in state.TEST]).mean(0), np.stack([r["sparse"]["js"] for r in state.TEST]).astype(float),
            np.stack([r["sparse"]["flip"] for r in state.TEST]).astype(float), np.stack([r["sparse"]["kfrac_avg"] for r in state.TEST]).astype(float), matched,
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

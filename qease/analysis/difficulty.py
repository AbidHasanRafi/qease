"""Analysis blocks A11-A12: budget versus question difficulty, qualitative examples and failure cases."""
import matplotlib
matplotlib.use("Agg")                      # figures are written to disk
from PIL import Image
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from qease import state
from qease.analysis.common import sty
from qease.crc import arr, loss_tag, make_policy_idx
from qease.reporting import BASELINES, COL2, DS_LABEL, panel_label, save_fig, save_table
from qease.stats import boot_curve
from qease.vlm.helpers import smart_resize


def an_budget():
    kind, eps, alpha = "js", state.EPS0, 0.10
    ent = np.array([r["ent"] for r in state.TEST]); dsv = np.array([r["ds"] for r in state.TEST]); ar = np.arange(len(state.TEST))
    cols = {}
    for m in [x for x in state.AV if x.startswith("qease")]:
        kidx = make_policy_idx(m, kind, eps, alpha, state.CAL, state.TEST, "adaptive"); cols[m] = arr(state.TEST, "kfrac", m)[ar, kidx]
    D = pd.DataFrame(dict(uid=[r["uid"] for r in state.TEST], dataset=dsv, answer_entropy=ent, ptop1=[r["ptop1"] for r in state.TEST], **{f"frac_{m}": v for m, v in cols.items()}))
    D["entropy_decile"] = pd.qcut(D.answer_entropy, 10, labels=False, duplicates="drop")
    rows = []
    for dc, g in D.groupby("entropy_decile"):
        for m in cols:
            mu, lo, hi, _ = boot_curve(g[f"frac_{m}"].values, B=500)
            rows.append(dict(entropy_decile=int(dc), mean_entropy=g.answer_entropy.mean(), method=m, kept_frac=mu[0], lo=lo[0], hi=hi[0], n=len(g)))
    BD = pd.DataFrame(rows)
    sp = [dict(scope="all", method=sty(m)["label"], spearman_kept_vs_entropy=float(D[["answer_entropy", f"frac_{m}"]].corr(method="spearman").iloc[0, 1])) for m in cols]
    for d in state.DSL:
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
    data = [D[D.dataset == d].frac_qease.values for d in state.DSL]
    bp = ax[1].boxplot(data, widths=0.55, patch_artist=True, flierprops=dict(markersize=1.5), medianprops=dict(color="k", lw=0.9))
    for patch, d in zip(bp["boxes"], state.DSL): patch.set_facecolor(state.DS_COLOR[d]); patch.set_alpha(0.6)
    ax[1].set_xticks(range(1, len(state.DSL) + 1)); ax[1].set_xticklabels([DS_LABEL[d] for d in state.DSL], rotation=30, ha="right"); ax[1].set_ylabel("fraction of tokens kept (Q-EASE adaptive)"); panel_label(ax[1], "(b)")
    save_fig(fig, "fig_budget_vs_difficulty", D, "adaptive budget vs question difficulty (a) and per-dataset budget distribution (b)")


def _draw_overlay(ax, rec, selmask, geom, title):
    nt = state.CFG["TOKENS_PER_DS"].get(rec["ds"], state.CFG["TOKENS_PER_DS"]["default"])
    img = smart_resize(Image.open(state.POOL_DIR / rec["img"]).convert("RGB"), nt)
    ax.imshow(img); ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    if selmask is not None:
        win, gw, gh, H, W = geom["win"], geom["gw"], geom["gh"], geom["H"], geom["W"]
        for gi in range(gh * gw):
            if not selmask[gi]:
                gr, gc = divmod(gi, gw); x0, y0 = gc * win * 28, gr * win * 28
                ax.add_patch(Rectangle((x0, y0), min(win, W - gc * win) * 28, min(win, H - gr * win) * 28, facecolor="k", alpha=0.62, lw=0))
    ax.set_title(title, fontsize=5.5, pad=2)


def an_qual():
    show = [m for m in ["qease", "qsim", "uniform", "random"] if m in state.AV]
    gap = lambda r: min(r["js"][b][state.J25] for b in BASELINES) - r["js"]["qease"][state.J25]
    cand = [r for r in state.TEST if "sel25" in r and "qease" in r["sel25"]]
    wins = []
    for d in state.DSL:
        g = sorted([r for r in cand if r["ds"] == d], key=gap, reverse=True)
        if g: wins.append(g[0])
    wins = wins[: state.CFG["N_QUAL"]]; fails = sorted(cand, key=gap)[:3]
    rows = []
    for name, group in [("fig_qualitative_wins", wins), ("fig_qualitative_failures", fails)]:
        if not group: continue
        fig, ax = plt.subplots(len(group), len(show) + 1, figsize=(COL2, 1.35 * len(group) + 0.2))
        ax = np.atleast_2d(ax); fd = []
        for i, r in enumerate(group):
            rec = state.REC_BY_UID[r["uid"]]; q = (r["q"] if len(r["q"]) < 48 else r["q"][:45] + "...").replace("$", "\\$").replace("_", "\\_")
            _draw_overlay(ax[i, 0], rec, None, r["geom"], f"{DS_LABEL[r['ds']]}: {q}")
            for j, m in enumerate(show):
                _draw_overlay(ax[i, j + 1], rec, r["sel25"][m], r["geom"], f"{sty(m)['label'].replace(' (ours)', '')}  JS={r['js'][m][state.J25]:.3f}")
                fd.append(dict(uid=r["uid"], dataset=r["ds"], question=r["q"], method=m, budget=state.FR[state.J25], js=float(r["js"][m][state.J25]), kept_group_indices=";".join(map(str, np.where(r["sel25"][m])[0]))))
        save_fig(fig, name, pd.DataFrame(fd), "qualitative examples at 25% budget (dark = dropped groups); " + ("largest Q-EASE advantage per dataset" if "wins" in name else "largest Q-EASE failures"))


def an_failures():
    rows = []
    for r in state.TEST:
        base = {b: float(r["js"][b][state.J25]) for b in BASELINES if b in r["js"]}
        rows.append(dict(uid=r["uid"], dataset=DS_LABEL[r["ds"]], question=r["q"], answer_entropy=r["ent"], G=r["G"], js_qease=float(r["js"]["qease"][state.J25]),
                         **{f"js_{b}": v for b, v in base.items()}, best_baseline_js=min(base.values()),
                         qease_advantage=min(base.values()) - float(r["js"]["qease"][state.J25]), qease_flip=bool(r["flip"]["qease"][state.J25])))
    D = pd.DataFrame(rows).sort_values("qease_advantage")
    save_table(D.head(60), "failure_cases_qease", "60 test samples where Q-EASE (25% budget) is most worse than the best baseline", latex=False)
    save_table(D.tail(60).iloc[::-1], "success_cases_qease", "60 test samples where Q-EASE (25% budget) is most better than the best baseline", latex=False)
    fam = D.assign(bin=pd.cut(D.answer_entropy, [-1, 0.1, 0.5, 1.5, 100], labels=["very confident", "confident", "uncertain", "very uncertain"]))
    save_table(fam.groupby("bin", observed=True).agg(n=("uid", "count"), mean_js_qease=("js_qease", "mean"), mean_best_baseline=("best_baseline_js", "mean"),
                                                    win_rate=("qease_advantage", lambda x: float((x > 0).mean())), flip_rate=("qease_flip", "mean")).reset_index(),
               "qease_by_answer_uncertainty", "Q-EASE vs best baseline stratified by full-model answer uncertainty (25% budget)")

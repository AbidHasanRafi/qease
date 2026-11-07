"""Tables of the trained-baseline and question-swap experiments."""
import numpy as np
import pandas as pd
import torch
from qease import state
from qease.extra import xstate
from qease.extra.trained_baselines import X_KIND
from qease.extra.utils import x_boot_ci, x_holm, x_paired, x_save, xd, xlog


def x_tb_report():
    recs = []
    for p in sorted(xd("tbeval").glob("test_*.pt")): recs += torch.load(p, weights_only=False)
    if not recs: xlog("no trained-baseline evaluation yet"); return
    names = [n for n in ["full", "noq"] + list(X_KIND) if all(n in r["js"] for r in recs)]
    lab = dict(full="Q-EASE", noq="Q-EASE w/o question", attn="attention-distilled selector", mlp="MLP selector (same labels)", shuf="shuffled-label selector")
    fr = state.CFG["K_GRID"]; rows = []
    for n in names:
        JS = np.stack([r["js"][n] for r in recs]); FL = np.stack([r["flip"][n] for r in recs]).astype(float)
        for j, f in enumerate(fr):
            m, lo, hi = x_boot_ci(JS[:, j]); fm, flo, fhi = x_boot_ci(FL[:, j])
            d = dict(method=lab[n], frac=f, js=m, js_lo=lo, js_hi=hi, flip=fm, flip_lo=flo, flip_hi=fhi, n=len(recs))
            if n != "full":
                Q = np.stack([r["js"]["full"] for r in recs])[:, j]; dm, dlo, dhi, p = x_paired(JS[:, j], Q)
                d.update(js_minus_qease=dm, diff_lo=dlo, diff_hi=dhi, p_boot=p)
            rows.append(d)
    T = pd.DataFrame(rows)
    if "p_boot" in T: T["p_holm"] = np.nan; msk = T.p_boot.notna(); T.loc[msk, "p_holm"] = x_holm(T.loc[msk, "p_boot"].values)
    x_save(T, "trained_baselines_fidelity", "fidelity of trained selectors with different supervision (test split)")
    piv = T[T.frac.isin([0.125, 0.25, 0.5])].pivot_table(index="method", columns="frac", values="js").round(4)
    xlog("mean JS at 12.5 / 25 / 50 % budget:\n" + piv.to_string())


def x_qswap_report():
    rows = []
    for p in sorted(xd("qswap").glob("shard_*.pt")): rows += torch.load(p, weights_only=False)
    if not rows: xlog("no question-swap results yet"); return
    out = []
    for f in xstate.XCFG["QSWAP_FRACS"]:
        J = {c: np.array([r["js"][f][c] for r in rows]) for c in ("own", "swap", "noq", "rand")}
        FL = {c: np.array([float(r["flip"][f][c]) for r in rows]) for c in ("own", "swap", "noq", "rand")}
        iou = {c: np.array([r["iou"][f][c] for r in rows]) for c in ("own_vs_swap", "rand_vs_rand", "own_vs_noq")}
        for c in ("own", "swap", "noq", "rand"):
            m, lo, hi = x_boot_ci(J[c]); fm = FL[c].mean()
            d = dict(frac=f, mask_from=c, js=m, js_lo=lo, js_hi=hi, flip=fm, n=len(rows))
            if c != "own": d["js_minus_own"], d["diff_lo"], d["diff_hi"], d["p_boot"] = x_paired(J[c], J["own"])
            out.append(d)
        out.append(dict(frac=f, mask_from="IoU own-vs-swapped-question", js=iou["own_vs_swap"].mean(), n=len(rows)))
        out.append(dict(frac=f, mask_from="IoU random-vs-random", js=iou["rand_vs_rand"].mean(), n=len(rows)))
        out.append(dict(frac=f, mask_from="IoU own-vs-no-question", js=iou["own_vs_noq"].mean(), n=len(rows)))
    T = pd.DataFrame(out); x_save(T, "question_swap", "JS of the pruned model under masks chosen for the own / a different question")
    xlog("question swap:\n" + T.round(4).to_string(index=False))

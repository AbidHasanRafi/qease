"""Analysis blocks A0-A3: run configuration, dataset / probe-label statistics, phase-0 gate, training curves."""
import matplotlib
matplotlib.use("Agg")                      # figures are written to disk
import json
import sys
import time

import datasets as hfds
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import transformers
from qease import state
from qease.analysis.common import draw_curve, sty, tidy_curve
from qease.reporting import BASELINES, COL2, DS_LABEL, frac_ticks, panel_label, save_fig, save_table
from qease.stats import boot_curve
from qease.utils import log


def an_run_config():
    import platform
    info = dict(tag=state.TAG, config={k: (v if isinstance(v, (int, float, str, list, dict, type(None))) else str(v)) for k, v in state.CFG.items()},
                python=sys.version.split()[0], platform=platform.platform(), torch=torch.__version__, transformers=transformers.__version__,
                datasets=hfds.__version__, numpy=np.__version__, pandas=pd.__version__, matplotlib=matplotlib.__version__,
                devices=state.DEVICES, gpu_names=[torch.cuda.get_device_name(i) for i in range(state.N_GPU)], dtype=str(state.DTYPE),
                n_pool={d: sum(1 for r in state.POOL if r["ds"] == d) for d in state.DS_ORDER},
                n_cal=len(state.CAL), n_test=len(state.TEST), n_failures=len(state.FAILS), finished=time.strftime("%Y-%m-%d %H:%M:%S"),
                selector_params_M={k: round(sum(p.numel() for p in v.parameters()) / 1e6, 3) for k, v in state.SELECTORS.items()},
                llm_decoder_params_B=(round(state.RUNNERS[0].llm_params / 1e9, 3) if state.RUNNERS else None),
                splits="by image hash: train 62% / val 8% / cal 15% / test 15% (no image shared across splits)",
                note_guarantee="CRC guarantee is marginal over (calibration set, test point) under exchangeability.")
    (state.WORK / "results" / "run_config.json").write_text(json.dumps(info, indent=1, default=str))
    state.MANIFEST.append(("meta", "results/run_config.json", "full configuration, library versions, GPU names, sample counts"))
    if state.FAILS:
        save_table(pd.DataFrame([dict(stage=f[0], item=f[1], error=f[2], traceback_tail=f[3][-400:]) for f in state.FAILS]),
                   "failures_log", "per-sample / per-block failures logged during the run", latex=False)


def an_probe_stats():
    side = []
    for p in sorted((state.WORK / "probe").glob("stats_*.pt")): side += torch.load(p, weights_only=False)
    if not side: log("   no probe sidecars found - skipping"); return
    rows = []
    for s in side:
        for j in range(len(s["js"])):
            ri, within = int(s["rid"][j]), j % state.MPR
            rows.append(dict(ds=s["ds"], uid=s["uid"], ratio=state.RATIOS[ri], mask_type="iid-random" if within < state.MPR // 2 else "spatial-blob",
                             js=float(s["js"][j]), flip=float(s["flip"][j]), G=s["G"], ent=s["ent"]))
    D = pd.DataFrame(rows)
    # table 1: dataset statistics
    pool = pd.DataFrame(state.POOL)
    t1 = pool.groupby("ds").agg(n_pool=("uid", "count"), n_train=("split", lambda x: (x == "train").sum()), n_val=("split", lambda x: (x == "val").sum()),
                                n_cal=("split", lambda x: (x == "cal").sum()), n_test=("split", lambda x: (x == "test").sum()),
                                mean_W=("W", "mean"), mean_H=("H", "mean")).reset_index()
    pr = D.drop_duplicates("uid").groupby("ds").agg(n_probed=("uid", "count"), mean_groups=("G", "mean"), mean_answer_entropy=("ent", "mean")).reset_index()
    t1 = t1.merge(pr, on="ds", how="left")
    t1["tokens_budget"] = t1["ds"].map(lambda d: state.CFG["TOKENS_PER_DS"].get(d, state.CFG["TOKENS_PER_DS"]["default"]))
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
    for ds_ in state.DS_ORDER:
        d = FD[(FD.panel == "b") & (FD.series == ds_)].sort_values("frac")
        if len(d): ax[1].plot(d.frac, d["mean"], color=state.DS_COLOR[ds_], marker="o", ms=2.5, label=DS_LABEL[ds_])
    for a, t in zip(ax, ["(a) mask type", "(b) dataset"]):
        a.set_xlabel("fraction of visual groups kept"); a.set_ylabel("mean JS to full model (nats)"); a.set_yscale("log"); a.legend(); panel_label(a, t[:3])
    save_fig(fig, "fig_probe_labels", FD, "counterfactual probe labels: answer-distribution shift vs fraction of groups kept (training signal exists)")


def an_phase0():
    f = state.WORK / "phase0" / "phase0.pt"
    if not f.exists(): log("   phase0 file missing - skipping"); return
    P0 = torch.load(f, weights_only=False)
    names = [n for n in ["oracle1", "qsim", "uniform", "random"] if all(n in o["js"] for o in P0)]
    rows, fdat = [], []
    fig, ax = plt.subplots(1, 2, figsize=(COL2, 2.5))
    for n in names:
        for key, a, lab in [("js", ax[0], "mean JS to full model (nats)"), ("flip", ax[1], "answer-flip rate")]:
            X = np.stack([o[key][n] for o in P0]).astype(float); mu, lo, hi, _ = boot_curve(X)
            d = tidy_curve(state.FR, mu, lo, hi, panel="a" if key == "js" else "b", metric=key, method=n, n=len(P0))
            fdat.append(d); draw_curve(a, d, n)
            for j in range(len(state.FR)):
                rows.append(dict(method=sty(n)["label"], metric=key, frac=state.FR[j], mean=mu[j], lo=lo[j], hi=hi[j], n=len(P0)))
    for a, lab in zip(ax, ["(a)", "(b)"]):
        frac_ticks(a, state.FR); a.set_xlabel("fraction of groups kept"); panel_label(a, lab)
    ax[0].set_yscale("symlog", linthresh=1e-3); ax[0].set_ylabel("mean JS to full model (nats)"); ax[1].set_ylabel("answer-flip rate"); ax[0].legend()
    save_table(pd.DataFrame(rows), "phase0_headroom", "Gate 1: counterfactual single-ablation oracle vs cheap baselines (validation images)")
    save_fig(fig, "fig_phase0_headroom", pd.concat(fdat), "Gate 1 headroom: oracle vs baselines")
    lo_b = [j for j, f_ in enumerate(state.FR) if f_ <= 0.25]
    orc = np.mean([[o["js"]["oracle1"][j] for o in P0] for j in lo_b])
    bb = np.mean([min(np.mean([o["js"][n][j] for o in P0]) for n in BASELINES if n in names) for j in lo_b])
    log(f"   GATE 1: mean JS at <=25% budget  oracle={orc:.4f}  best baseline={bb:.4f}  ratio={orc / max(bb, 1e-9):.2f}")


def an_hist():
    dfs = []
    for v in ("full", "noq"):
        p = state.WORK / "ckpt" / f"history_{v}.csv"
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

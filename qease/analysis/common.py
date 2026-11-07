"""Shared helpers of the analysis blocks."""
import matplotlib
matplotlib.use("Agg")                      # figures are written to disk
import time
import traceback

import numpy as np
import pandas as pd
from qease import state
from qease.crc import load_eval, load_lambdas
from qease.reporting import ALL_BASELINES, BASELINES, METHOD_STYLE, _EXTRA_COLORS
from qease.selection import METHODS
from qease.utils import log


def run_block(name, fn):
    t0 = time.time()
    try:
        fn(); log(f"[analyze] {name}: ok ({time.time()-t0:.1f}s)")
    except Exception as e:
        state.FAILS.append(("analyze", name, repr(e)[:200], traceback.format_exc()))
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


BOOTM = {}


def prepare_analysis_state():
    # run-time state lives in qease.state
    state.CAL, state.TEST = load_eval("cal"), load_eval("test")
    assert len(state.CAL) > 20 and len(state.TEST) > 20, "need eval records: run stage 'eval' first"
    keys = set.intersection(*[set(r["js"].keys()) for r in state.CAL + state.TEST])
    state.AV = [m for m in METHODS if m in keys] + sorted(k for k in keys if k not in METHODS)       # known methods first, plug-ins after
    BASELINES[:] = [b for b in ALL_BASELINES + [k for k in state.AV if k not in METHODS] if b in state.AV]
    state.FR = np.array(state.CFG["K_GRID"]); state.J25 = int(np.argmin(abs(state.FR - 0.25))); state.J12 = int(np.argmin(abs(state.FR - 0.125)))
    state.DSL = [d for d in state.DS_ORDER if any(r["ds"] == d for r in state.TEST)]
    if not state.LAMBDAS: load_lambdas()
    log(f"[analyze] cal={len(state.CAL)} test={len(state.TEST)} methods={state.AV} datasets={state.DSL}")

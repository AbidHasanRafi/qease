"""Logging, per-run output folders, bootstrap / paired tests / Holm correction and CSV export for the extra experiments."""
import math
import time

import numpy as np
from qease import state
from qease.extra import xstate


def xlog(*a):
    print(time.strftime("%H:%M:%S"), "[extra]", *a, flush=True)


LN2 = math.log(2)


def xd(sub=None):
    p = state.WORK / "extra" / sub if sub else state.WORK / "extra"
    p.mkdir(parents=True, exist_ok=True)
    return p


def x_boot_ci(x, B=None, seed=0):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    if len(x) == 0: return np.nan, np.nan, np.nan
    B = B or xstate.XCFG["BOOT_B"]; rng = np.random.default_rng(seed)
    m = x[rng.integers(0, len(x), (B, len(x)))].mean(1)
    return float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def x_paired(a, b, B=None, seed=1):
    """paired bootstrap of mean(a-b): (mean, lo, hi, two-sided p)."""
    d = np.asarray(a, float) - np.asarray(b, float); d = d[~np.isnan(d)]
    B = B or xstate.XCFG["BOOT_B"]; rng = np.random.default_rng(seed)
    m = d[rng.integers(0, len(d), (B, len(d)))].mean(1)
    p = 2 * min((m <= 0).mean(), (m >= 0).mean())
    return float(d.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5)), float(max(p, 1.0 / B))


def x_holm(p):
    p = np.asarray(p, float); o = np.argsort(p); out = np.empty_like(p); run = 0.0
    for r_, i in enumerate(o):
        run = max(run, (len(p) - r_) * p[i]); out[i] = min(1.0, run)
    return out


def x_save(df, name, desc=""):
    d = xd("results")
    df.to_csv(d / f"{name}.csv", index=False)
    try: df.to_latex(d / f"{name}.tex", index=False, float_format="%.3f", escape=True)
    except Exception: pass
    xlog(f"saved {name}.csv  ({len(df)} rows)  {desc}")

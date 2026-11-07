"""Statistics: bootstrap intervals, paired tests, Holm correction, Wilson intervals, rank metrics."""
import math

import numpy as np
import pandas as pd
from qease import state


# ---------------------------- statistics ----------------------------
def boot_curve(X, B=None, seed=None):
    """Percentile bootstrap over samples (rows). Same seed -> identical resamples across calls -> PAIRED bootstraps.
    returns mean, lo(2.5%), hi(97.5%), M(B x k bootstrap means)."""
    X = np.asarray(X, float)
    if X.ndim == 1: X = X[:, None]
    n, B = X.shape[0], B or state.CFG["BOOT_B"]
    idx = np.random.default_rng(state.CFG["SEED"] if seed is None else seed).integers(0, n, size=(B, n))
    M = np.stack([X[i].mean(0) for i in idx])
    return X.mean(0), np.percentile(M, 2.5, 0), np.percentile(M, 97.5, 0), M


def paired_test(Xa, Xb):
    """a - b. returns mean diff, CI, two-sided bootstrap p, paired effect size dz."""
    D = np.asarray(Xa, float) - np.asarray(Xb, float)
    if D.ndim == 1: D = D[:, None]
    mu, lo, hi, M = boot_curve(D)
    B = M.shape[0]
    p = np.maximum(np.minimum(1.0, 2 * np.minimum((M <= 0).mean(0), (M >= 0).mean(0))), 1.0 / B)
    dz = mu / (D.std(0, ddof=1) + 1e-12)
    return mu, lo, hi, p, dz


def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); m = len(p); adj = np.empty(m)
    run = 0.0
    for rank, i in enumerate(o):
        run = max(run, (m - rank) * p[i]); adj[i] = min(1.0, run)
    return adj


def wilson(k, n, z=1.96):
    if n == 0: return np.nan, np.nan
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - h) / d, (c + h) / d


def auroc(score, label):
    label = np.asarray(label).astype(bool); n1, n0 = label.sum(), (~label).sum()
    if n1 == 0 or n0 == 0: return np.nan
    rk = pd.Series(np.asarray(score, float)).rank().values
    return float((rk[label].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def row_spearman(P, T):
    rp = pd.DataFrame(P).rank(axis=1).values; rt = pd.DataFrame(T).rank(axis=1).values
    rp = rp - rp.mean(1, keepdims=True); rt = rt - rt.mean(1, keepdims=True)
    den = np.sqrt((rp ** 2).sum(1) * (rt ** 2).sum(1))
    return np.where(den > 0, (rp * rt).sum(1) / np.where(den > 0, den, 1), np.nan)


def trap_rows(Y, x):
    return ((Y[:, 1:] + Y[:, :-1]) * np.diff(x)[None] / 2).sum(1)


def frac_at_target(fr, y, t):
    """smallest budget (linear interpolation) at which curve y first drops to <= t; NaN if never."""
    for i in range(len(fr)):
        if y[i] <= t:
            if i == 0: return float(fr[0])
            x0, x1, y0, y1 = fr[i - 1], fr[i], y[i - 1], y[i]
            return float(x0 + (y0 - t) * (x1 - x0) / (y0 - y1)) if y0 > y1 else float(x1)
    return np.nan

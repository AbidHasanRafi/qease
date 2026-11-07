"""Conformal risk control: envelope loss, threshold search, fixed / adaptive / per-dataset (Mondrian) policies."""
import json

import numpy as np
import torch
from qease import state
from qease.reporting import LN2
from qease.stats import boot_curve, wilson
from qease.utils import log


# ------------------------ loading + loss helpers ---------------------
def load_eval(split):
    recs = []
    for p in sorted((state.WORK / "eval").glob(f"{split}_*.pt")):
        recs += torch.load(p, weights_only=False)
    return recs


def arr(recs, key, m):
    return np.stack([r[key][m] for r in recs]).astype(float)


def loss_tag(kind, eps):
    return "answer-flip" if kind == "flip" else f"JS>{eps:g}"


def loss_mat(recs, m, kind, eps):
    return arr(recs, "flip", m) if kind == "flip" else (arr(recs, "js", m) > eps).astype(float)


# --------------------- conformal risk control (CRC) -------------------
# Loss l_i(k) in {0,1}. A budget rule is a function of lambda (adaptive) or k (fixed).  CRC needs the loss to be
# monotone in the control parameter, so we calibrate on the upper envelope  max_{k'>=k} l(k')  (>= the raw loss),
# which keeps the guarantee valid (conservatively) even though raw loss is not exactly monotone in k.
def envelope(loss):
    return np.maximum.accumulate(loss[:, ::-1], axis=1)[:, ::-1]


def crc_bound(mean_loss, n, B=1.0):
    return n / (n + 1) * mean_loss + B / (n + 1)


def first_ok(rhat, lam):
    ok = rhat <= lam
    return np.where(ok.any(1), ok.argmax(1), rhat.shape[1] - 1)


def crc_adaptive_lambda(rhat, loss, alpha, ncand=301):
    """largest lambda whose CRC bound <= alpha. rule: k_hat(q) = first grid budget with predicted risk <= lambda."""
    n, nK = rhat.shape
    env = envelope(loss)
    cands = np.unique(np.quantile(rhat, np.linspace(0, 1, ncand)))
    ok = rhat[:, :, None] <= cands[None, None, :]                                     # (n, nK, C)
    first = np.where(ok.any(1), ok.argmax(1), nK - 1)                                 # (n, C)
    risk = env[np.arange(n)[:, None], first].mean(0)                                  # non-decreasing in lambda
    nfeas = int(np.cumprod(crc_bound(risk, n) <= alpha).sum())
    return -np.inf if nfeas == 0 else float(cands[nfeas - 1])


def calibrate_fixed(loss, alpha):
    b = crc_bound(envelope(loss).mean(0), len(loss))
    ok = np.where(b <= alpha)[0]
    return int(ok[0]) if len(ok) else loss.shape[1] - 1


def make_policy_idx(method, kind, eps, alpha, cal, test, policy):
    """grid index chosen per test sample. policy: fixed | adaptive | naive (uncalibrated raw risk threshold)."""
    Lc = loss_mat(cal, method, kind, eps)
    if policy == "fixed":
        return np.full(len(test), calibrate_fixed(Lc, alpha))
    rt = arr(test, "rhat", method)
    if policy == "adaptive":
        return first_ok(rt, crc_adaptive_lambda(arr(cal, "rhat", method), Lc, alpha))
    if policy == "naive":                                   # trust the risk head as-is: predicted JS <= eps
        return first_ok(rt, eps / LN2)
    raise ValueError(policy)


def mondrian_idx(method, kind, eps, alpha, cal, test, min_n=None):
    """group-conditional (per-dataset) CRC; datasets with < min_n calibration points fall back to the global lambda."""
    min_n = min_n or state.CFG["MONDRIAN_MIN_N"]
    rc, Lc, rt = arr(cal, "rhat", method), loss_mat(cal, method, kind, eps), arr(test, "rhat", method)
    glam = crc_adaptive_lambda(rc, Lc, alpha)
    cds, tds = np.array([r["ds"] for r in cal]), np.array([r["ds"] for r in test])
    kidx = np.zeros(len(test), int)
    for d in set(tds):
        ci, ti = np.where(cds == d)[0], np.where(tds == d)[0]
        lam = crc_adaptive_lambda(rc[ci], Lc[ci], alpha) if len(ci) >= min_n else glam
        kidx[ti] = first_ok(rt[ti], lam)
    return kidx


def policy_eval(recs, m, kidx, kind, eps):
    ar = np.arange(len(recs))
    return loss_mat(recs, m, kind, eps)[ar, kidx], arr(recs, "kfrac", m)[ar, kidx]


def summarize_policy(lv, fv):
    lo, hi = wilson(lv.sum(), len(lv))
    mu, flo, fhi, _ = boot_curve(fv)
    return dict(n_test=len(lv), test_risk=float(lv.mean()), risk_lo=lo, risk_hi=hi,
                kept_frac=float(mu[0]), frac_lo=float(flo[0]), frac_hi=float(fhi[0]))


def _lam_key(m, kind, eps, alpha): return f"{m}|{kind}|{eps}|{alpha}"


def load_lambdas():
    if state.LAM_FILE.exists():
        for k, v in json.load(open(state.LAM_FILE)).items():
            m, kind, eps, alpha = k.split("|")
            state.LAMBDAS[(m, kind, None if eps == "None" else float(eps), float(alpha))] = v


def run_crc():
    """Stage 'crc': calibrate the conformal risk-control thresholds on the calibration split."""
    _cal = load_eval("cal")
    assert len(_cal) > 20, "run the 'eval' stage first (need calibration records)"
    _alphas = sorted(set(state.CFG["ALPHAS"]) | set(state.CFG["ALPHA_SWEEP"]) | {a for (_, _, a) in state.CFG["GEN_ADAPT"]})
    for _m in [m for m in ("qease", "qease_noq") if all(m in r["rhat"] for r in _cal)]:
        for _kind, _eps in state.LOSS_DEFS:
            _L, _R = loss_mat(_cal, _m, _kind, _eps), arr(_cal, "rhat", _m)
            for _a in _alphas:
                state.LAMBDAS[(_m, _kind, _eps, _a)] = crc_adaptive_lambda(_R, _L, _a)
    json.dump({_lam_key(*k): v for k, v in state.LAMBDAS.items()}, open(state.LAM_FILE, "w"), indent=1)
    log(f"[crc] calibrated {len(state.LAMBDAS)} thresholds on {len(_cal)} calibration samples -> {state.LAM_FILE.name}")
    for _a in state.CFG["ALPHAS"]:
        _l = state.LAMBDAS.get(("qease", "js", state.EPS0, _a))
        if _l is not None: log(f"   qease  loss=JS>{state.EPS0}  alpha={_a}:  lambda={_l:.4f} (predicted-JS/ln2 threshold)")
    del _cal

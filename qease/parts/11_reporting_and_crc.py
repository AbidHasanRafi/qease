# %% ------------------------- 11. REPORTING TOOLKIT: stats, IEEE/ACM style, export
from fractions import Fraction

TAB, FIG, FDATA = WORK / "results" / "tables", WORK / "results" / "figures", WORK / "results" / "figdata"
MANIFEST = []                                    # (kind, relative path, description)
COL1, COL2 = 3.5, 7.16                           # IEEE / ACM single- and double-column widths (inches)
LN2 = math.log(2)
EPS0 = CFG["EPS_LIST"][-1]                       # main JS threshold
LOSS_DEFS = [("flip", None)] + [("js", e) for e in CFG["EPS_LIST"]]
DS_ORDER = list(CFG["N_POOL"].keys())
DS_LABEL = dict(vqav2="VQAv2", gqa="GQA", textvqa="TextVQA", docvqa="DocVQA", aokvqa="A-OKVQA", vsr="VSR", clevr="CLEVR")
DS_COLOR = dict(zip(DS_ORDER, ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#7F7F7F"]))
METHOD_STYLE = {                                  # Okabe-Ito colour-blind-safe palette, distinct markers + line styles (B/W safe)
    "qease":     dict(color="#D55E00", marker="o", ls="-",  label="Q-EASE (ours)"),
    "qease_noq": dict(color="#CC79A7", marker="s", ls="--", label="Q-EASE w/o question"),
    "qsim":      dict(color="#0072B2", marker="^", ls="-.", label="Question-similarity top-$k$"),
    "uniform":   dict(color="#009E73", marker="D", ls=":",  label="Uniform grid"),
    "random":    dict(color="#7F7F7F", marker="v", ls="-",  label="Random"),
    "oracle1":   dict(color="#000000", marker="*", ls="-",  label="Single-ablation oracle"),
    "fastv":     dict(color="#E69F00", marker="P", ls="-",  label="FastV ($K$=2, re-impl.)"),
    "visionzip": dict(color="#56B4E9", marker="X", ls="-",  label="VisionZip (port)"),
    "sparsevlm": dict(color="#8C564B", marker="h", ls="-",  label="SparseVLM v1.0 (port)"),
}
ALL_BASELINES = ["fastv", "visionzip", "qsim", "uniform", "random"]
BASELINES = list(ALL_BASELINES)                  # narrowed in prepare_analysis_state() to what is actually in the eval records
_EXTRA_COLORS = ["#56B4E9", "#F0E442", "#4D4D4D", "#999933"]
POLICY_LABEL = dict(fixed="fixed-$k$", adaptive="adaptive (CRC)", mondrian="adaptive (per-dataset CRC)", naive="adaptive (uncalibrated)")

def set_style():
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "Nimbus Roman", "Liberation Serif", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix", "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8, "legend.fontsize": 6.5,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.linewidth": 0.8, "lines.linewidth": 1.2, "lines.markersize": 3.5,
        "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True,       # Origin-style boxed axes
        "xtick.minor.visible": True, "ytick.minor.visible": True, "xtick.major.size": 3, "ytick.major.size": 3,
        "xtick.minor.size": 1.5, "ytick.minor.size": 1.5, "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.4,
        "axes.axisbelow": True, "legend.frameon": True, "legend.edgecolor": "0.3", "legend.fancybox": False,
        "legend.framealpha": 0.92, "legend.borderpad": 0.3, "legend.handlelength": 1.8, "legend.labelspacing": 0.25,
        "figure.dpi": 120, "savefig.dpi": 600, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",                                 # embeddable TrueType (IEEE/ACM requirement)
    })
set_style()

def save_table(df, name, desc, latex=True):
    p = TAB / f"{name}.csv"; df.to_csv(p, index=False)
    MANIFEST.append(("table", str(p.relative_to(WORK)), desc + f"  [columns: {', '.join(map(str, df.columns))}]"))
    if latex:
        try: (TAB / f"{name}.tex").write_text(df.to_latex(index=False, float_format=lambda x: f"{x:.4f}", escape=True))
        except Exception: pass
    return df

def save_fig(fig, name, data, desc, show=True):
    """PDF (vector) + PNG (600 dpi) + the exact tidy data that was plotted."""
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"{name}.{ext}")
    data.to_csv(FDATA / f"{name}.csv", index=False)
    MANIFEST.append(("figure", f"results/figures/{name}.pdf|png", desc))
    MANIFEST.append(("figure-data", f"results/figdata/{name}.csv", f"data behind {name}  [columns: {', '.join(map(str, data.columns))}]"))
    plt.close(fig)
    if show:
        try:
            from IPython.display import Image as IPImage, display
            display(IPImage(filename=str(FIG / f"{name}.png"), width=760))
        except Exception:
            pass

def panel_label(ax, s):
    ax.text(0.02, 0.97, s, transform=ax.transAxes, fontweight="bold", va="top", ha="left", fontsize=8)

def frac_ticks(ax, fr):
    ax.set_xscale("log", base=2)
    ax.set_xticks(fr); ax.set_xticklabels([str(Fraction(float(f)).limit_denominator(32)) for f in fr])
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())

# ---------------------------- statistics ----------------------------
def boot_curve(X, B=None, seed=None):
    """Percentile bootstrap over samples (rows). Same seed -> identical resamples across calls -> PAIRED bootstraps.
    returns mean, lo(2.5%), hi(97.5%), M(B x k bootstrap means)."""
    X = np.asarray(X, float)
    if X.ndim == 1: X = X[:, None]
    n, B = X.shape[0], B or CFG["BOOT_B"]
    idx = np.random.default_rng(CFG["SEED"] if seed is None else seed).integers(0, n, size=(B, n))
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

# ------------------------ loading + loss helpers ---------------------
def load_eval(split):
    recs = []
    for p in sorted((EVAL_DIR).glob(f"{split}_*.pt")):
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
    min_n = min_n or CFG["MONDRIAN_MIN_N"]
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

# ------------------------------------------------------- 12. CRC STAGE
LAMBDAS = {}
LAM_FILE = TAB / "lambdas.json"

def _lam_key(m, kind, eps, alpha): return f"{m}|{kind}|{eps}|{alpha}"

def load_lambdas():
    if LAM_FILE.exists():
        for k, v in json.load(open(LAM_FILE)).items():
            m, kind, eps, alpha = k.split("|")
            LAMBDAS[(m, kind, None if eps == "None" else float(eps), float(alpha))] = v

if "crc" in CFG["STAGES"]:
    _cal = load_eval("cal")
    assert len(_cal) > 20, "run the 'eval' stage first (need calibration records)"
    _alphas = sorted(set(CFG["ALPHAS"]) | set(CFG["ALPHA_SWEEP"]) | {a for (_, _, a) in CFG["GEN_ADAPT"]})
    for _m in [m for m in ("qease", "qease_noq") if all(m in r["rhat"] for r in _cal)]:
        for _kind, _eps in LOSS_DEFS:
            _L, _R = loss_mat(_cal, _m, _kind, _eps), arr(_cal, "rhat", _m)
            for _a in _alphas:
                LAMBDAS[(_m, _kind, _eps, _a)] = crc_adaptive_lambda(_R, _L, _a)
    json.dump({_lam_key(*k): v for k, v in LAMBDAS.items()}, open(LAM_FILE, "w"), indent=1)
    log(f"[crc] calibrated {len(LAMBDAS)} thresholds on {len(_cal)} calibration samples -> {LAM_FILE.name}")
    for _a in CFG["ALPHAS"]:
        _l = LAMBDAS.get(("qease", "js", EPS0, _a))
        if _l is not None: log(f"   qease  loss=JS>{EPS0}  alpha={_a}:  lambda={_l:.4f} (predicted-JS/ln2 threshold)")
    del _cal
else:
    load_lambdas()

# %% ---------------------------------------------- 13. GENERATION-BASED ACCURACY
def norm_ans(s):
    s = s.lower().strip()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())

def lev(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]

def score_answer(ds, pred, answers):
    p = norm_ans(pred); gts = [norm_ans(a) for a in answers]
    if ds == "docvqa":                                           # ANLS (threshold 0.5)
        best = 0.0
        for g in gts:
            d = lev(p, g) / max(1, max(len(p), len(g)))
            best = max(best, 1 - d if d < 0.5 else 0.0)
        return best
    if ds in ("vqav2", "textvqa", "aokvqa"):                     # simplified VQA accuracy  min(#match/3, 1)
        return min(1.0, sum(p == g for g in gts) / 3.0)
    return float(p == gts[0]) if gts else 0.0


# %% ================================================================================================================
# 14. NOVELTY  -  ESCALATE-OR-ABSTAIN CASCADE.   Pure numpy/sklearn on the decoded records (no GPU).
#   stage j (budget f_j): the pruned model answers; harm head s_j = P(full correct & stage-j wrong), futility head u_j = P(full wrong).
#   policy(lambda, mu):   u_j >= mu -> ABSTAIN ;  s_j <= lambda -> ANSWER with stage j ;  else ESCALATE to the next budget of the path (last = full model).
#   loss (task-aligned):  L = 1{ full model correct  AND  (abstained  OR  final answer wrong) }   ("lost-correct": a question the full model gets right and we do not)
#   certificate (Learn-then-Test, fixed-sequence, exact binomial p-values, Bonferroni over mu):   P_cal( E[L(lambda_hat, mu_hat)] <= alpha ) >= 1 - delta
# ================================================================================================================
CASC_FILE = CKPT_DIR / "casc.pkl"
FEATS = ["frac", "ent", "top1", "margin", "rhat", "js0", "agree0", "ent0", "top10"]

def get_sel_cost():
    """selector overhead in units of one full prefill (measured by the latency stage; 3% before that)."""
    try:
        R = pd.read_csv(TAB / "raw_latency.csv"); return float(R[R.frac < 1.0].selector_ms.mean() / R[R.frac == 1.0].prefill_ms.mean())
    except Exception:
        return 0.03

def get_vshare():
    """fraction of the full pipeline spent in the vision tower + projector (measured; 0.65 default)."""
    try:
        R = pd.read_csv(TAB / "raw_latency.csv"); f = R[R.frac == 1.0]; return float(f.enc_ms.mean() / (f.enc_ms.mean() + f.prefill_ms.mean()))
    except Exception:
        return 0.65

def dec_arrays(recs):
    n, S = len(recs), len(CFG["DEC_GRID"]); assert all(len(o["stages"]) == S for o in recs), "decoded records do not match CFG['DEC_GRID']"
    FE = np.zeros((n, S, len(FEATS)), np.float32); SC = np.zeros((n, S), np.float32); COST = np.zeros((n, S), np.float32); KF = np.zeros((n, S), np.float32)
    for i, o in enumerate(recs):
        st = o["stages"]; e0, t0 = st[0]["ent"], st[0]["top1"]; L, nv = o["L"], o["n_vis"]
        for j, s in enumerate(st):
            FE[i, j] = [s["frac"], s["ent"], s["top1"], s["margin"], s["rhat"], s["js0"], s["agree0"], e0, t0]
            SC[i, j] = s["score"]; KF[i, j] = s["kfrac"]; COST[i, j] = (L - nv + s["kfrac"] * nv) / L
    return dict(FE=FE, SC=SC, C=SC >= CFG["CORRECT_TAU"], COST=COST, KF=KF, ds=np.array([o["ds"] for o in recs]), uid=[o["uid"] for o in recs],
                qlen=np.array([o["qlen"] for o in recs]), ent0=FE[:, 0, 1].copy())

def sub_arrays(A, idx):
    return {k: (v[idx] if isinstance(v, np.ndarray) else [v[i] for i in idx]) for k, v in A.items()}

class _Const:
    def __init__(self, p): self.p = float(p)
    def predict_proba(self, X): return np.stack([np.full(len(X), 1 - self.p), np.full(len(X), self.p)], 1)

def fit_clf(X, y):
    y = y.astype(int)
    if y.sum() < 8 or (len(y) - y.sum()) < 8: return _Const(y.mean() if len(y) else 0.5)
    return HistGradientBoostingClassifier(max_iter=CFG["HEAD_ITERS"], max_depth=3, learning_rate=0.08, l2_regularization=1.0, random_state=0).fit(X, y)

def fit_heads(A, idx=None):
    """harm head: P(full correct & stage-j wrong | stage features).  futility head: P(full wrong | stage features).  Trained on HEAD records only (never cal/test)."""
    FE, C = A["FE"], A["C"]; n, S, F_ = FE.shape
    idx = np.arange(n) if idx is None else np.asarray(idx); cfull = C[idx, S - 1]
    harm = fit_clf(FE[idx, :S - 1].reshape(-1, F_), (cfull[:, None] & ~C[idx, :S - 1]).reshape(-1))
    fail = fit_clf(FE[idx].reshape(-1, F_), np.repeat(~cfull[:, None], S, 1).reshape(-1))
    return dict(harm=harm, fail=fail)

def head_scores(H, A):
    FE = A["FE"]; n, S, F_ = FE.shape
    s = np.zeros((n, S), np.float32)
    s[:, :S - 1] = H["harm"].predict_proba(FE[:, :S - 1].reshape(-1, F_))[:, 1].reshape(n, S - 1)
    u = H["fail"].predict_proba(FE.reshape(-1, F_))[:, 1].reshape(n, S).astype(np.float32)
    return s, u

def path_idx(fracs):
    g = [round(x, 6) for x in CFG["DEC_GRID"]]
    return [g.index(round(f, 6)) for f in fracs]

def cascade_sim(path, s, u, COST, lam, mu, sel_cost):
    """vectorised cascade. returns (final stage index, abstained, cumulative cost in units of one full prefill)."""
    n = s.shape[0]; active = np.ones(n, bool); final = np.full(n, path[-1]); abst = np.zeros(n, bool); cost = np.full(n, sel_cost, np.float64)
    for t, j in enumerate(path):
        cost += np.where(active, COST[:, j], 0.0)
        ab = active & (u[:, j] >= mu)
        ok = active & ~ab & (np.ones(n, bool) if t == len(path) - 1 else (s[:, j] <= lam))
        stop = ab | ok; final[stop] = j; abst |= ab; active &= ~stop
    return final, abst, cost

def lost_correct(C, final, abst):
    return C[:, -1] & (abst | ~C[np.arange(len(final)), final])

def cfg_grid(s_ref=None):
    """candidate thresholds. DATA-INDEPENDENT (fixed geometric grid), so the fixed-sequence test sequence is chosen before any calibration loss is seen."""
    return np.concatenate([[-np.inf], np.geomspace(0.005, 0.8, CFG["CASC_LAMBDAS"])]), np.array([np.inf] + sorted(CFG["CASC_MUS"], reverse=True))

def loss_cost_matrix(path, A, s, u, lams, mus, sel_cost):
    """configs are indexed c = mi*len(lams) + li  (mu-major; lambda ascending = most conservative first)."""
    n, nl = len(s), len(lams); Lm = np.zeros((n, nl * len(mus)), bool); Cm = np.zeros((n, nl * len(mus)), np.float32)
    for mi, mu in enumerate(mus):
        for li, lam in enumerate(lams):
            final, abst, cost = cascade_sim(path, s, u, A["COST"], lam, mu, sel_cost)
            Lm[:, mi * nl + li] = lost_correct(A["C"], final, abst); Cm[:, mi * nl + li] = cost
    return Lm, Cm

def ltt_choose(Lm, Cm, nl, mu_groups, alpha, delta, rule="ltt"):
    """fixed-sequence testing along increasing lambda inside each mu-group (Bonferroni over groups); among the VALID configs return the cheapest on calibration.
    rule='crc' keeps the expectation-only conformal-risk-control bound (k+1)/(n+1) <= alpha (what the first version of the paper used)."""
    n = Lm.shape[0]; k = Lm.sum(0)
    p = sps.binom.cdf(k, n, alpha); thr = delta / max(1, len(mu_groups))
    valid = []
    for mi in mu_groups:
        for li in range(nl):
            c = mi * nl + li
            if (p[c] <= thr) if rule == "ltt" else ((k[c] + 1) / (n + 1) <= alpha): valid.append(c)
            else: break
    return min(valid, key=lambda c: Cm[:, c].mean()) if valid else 0          # config 0 = (lambda=-inf, mu=inf) = always the full model: zero loss by construction

def fixed_losses(A):
    return A["C"][:, -1][:, None] & ~A["C"]                                   # (n, S): harm of answering with stage j alone

def fixed_choose(Lh, Cst, alpha, delta):
    n, S = Lh.shape; p = sps.binom.cdf(Lh.sum(0), n, alpha); valid = []
    for j in range(S - 1, -1, -1):
        if p[j] <= delta: valid.append(j)
        else: break
    return min(valid, key=lambda j: Cst[:, j].mean()) if valid else S - 1

def static_stage(rhat, nu):
    ok = rhat <= nu; return np.where(ok.any(1), ok.argmax(1), rhat.shape[1])    # last index = full model

def static_choose(A, alpha, delta, nus):
    """previous Q-EASE controller: pick the budget from the SELECTOR'S risk head only (no LLM run), threshold nu calibrated on the decoded harm loss."""
    rh = A["FE"][:, :-1, FEATS.index("rhat")]; n = len(rh); Lh = fixed_losses(A); ar = np.arange(n); valid = []
    for nu in nus:
        st = static_stage(rh, nu); k = int(Lh[ar, st].sum())
        if sps.binom.cdf(k, n, alpha) <= delta: valid.append(nu)
        else: break
    if not valid: return -np.inf
    return min(valid, key=lambda nu: A["COST"][ar, static_stage(rh, nu)].mean())

def policy_metrics(A, final, abst, cost, sel_cost=None):
    n = len(final); ar = np.arange(n); lost = lost_correct(A["C"], final, abst); ans = ~abst
    lo, hi = wilson(int(lost.sum()), n); vs = get_vshare()
    return dict(n=n, lost_correct=float(lost.mean()), lost_lo=lo, lost_hi=hi, accuracy=float(np.where(abst, 0.0, A["SC"][ar, final]).mean()),
                selective_accuracy=float(A["SC"][ar, final][ans].mean()) if ans.any() else np.nan, coverage=float(ans.mean()), abstain_rate=float(abst.mean()),
                mean_cost=float(np.mean(cost)), e2e_cost=float(vs + (1 - vs) * np.mean(cost)), accuracy_full=float(A["SC"][:, -1].mean()),
                mean_stage_frac=float(np.mean(np.array(CFG["DEC_GRID"])[final])), escalated_to_full=float((final == A["SC"].shape[1] - 1).mean()))

def get_casc():
    if CASC_FILE.exists():
        return pickle.load(open(CASC_FILE, "rb"))
    return None

def casc_fit():
    """heads on the HEAD split -> calibration of (lambda, mu) per path / alpha / abstention on the CAL split -> saved policy."""
    head, cal = load_dec(DEC_DIR / "head"), load_dec(DEC_DIR / "cal")
    assert len(head) > 100 and len(cal) > 100, "run the 'dec' stage first"
    Ah, Ac = dec_arrays(head), dec_arrays(cal); H = fit_heads(Ah); s_c, u_c = head_scores(H, Ac)
    lams, mus = cfg_grid(s_c); sel_cost = get_sel_cost(); delta = CFG["CASC_DELTA"]; pol = {}
    log(f"[casc] heads trained on {len(head)} HEAD questions ({int(Ah['C'][:, -1].sum())} full-correct); calibrating on {len(cal)} calibration questions | "
        f"full-model accuracy cal={Ac['SC'][:, -1].mean():.3f} | grid {len(lams)} lambdas x {len(mus)} mus")
    for pname, fr in CFG["CASC_PATHS"].items():
        path = path_idx(fr); Lm, Cm = loss_cost_matrix(path, Ac, s_c, u_c, lams, mus, sel_cost)
        for alpha in CFG["CASC_ALPHAS"]:
            for ab in (False, True):
                c = ltt_choose(Lm, Cm, len(lams), list(range(len(mus))) if ab else [0], alpha, delta)
                pol[(pname, alpha, ab)] = dict(lam=float(lams[c % len(lams)]), mu=float(mus[c // len(lams)]), cal_cost=float(Cm[:, c].mean()), cal_risk=float(Lm[:, c].mean()))
                log(f"   {pname:8s} alpha={alpha:.2f} {'abstain' if ab else 'no-abst'}: lambda={pol[(pname, alpha, ab)]['lam']:.3f} mu={pol[(pname, alpha, ab)]['mu']:.2f} "
                    f"cal-risk={pol[(pname, alpha, ab)]['cal_risk']:.4f} cal-cost={pol[(pname, alpha, ab)]['cal_cost']:.3f}")
    pickle.dump(dict(H=H, pol=pol, lams=lams, mus=mus, sel_cost=sel_cost, delta=delta, feats=FEATS, grid=CFG["DEC_GRID"]), open(CASC_FILE, "wb"))
    log(f"[casc] policy saved -> {CASC_FILE.name}")

if "casc" in CFG["STAGES"]:
    casc_fit()



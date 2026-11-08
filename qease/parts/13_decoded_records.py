# %% ------------------------------- 13b. OFFICIAL ANSWER METRICS + DECODED RECORDS (the independent, task-aligned evidence)
_NUMW = {"none": "0", "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10"}
_ARTS = {"a", "an", "the"}
_PUNCT = [";", r"/", "[", "]", '"', "{", "}", "(", ")", "=", "+", "\\", "_", "-", ">", "<", "@", "`", ",", "?", "!"]
_PERIOD = re.compile(r"(?!<=\d)(\.)(?!\d)")
_COMMA = re.compile(r"(\d)(,)(\d)")

def vqa_norm(s):
    """official VQA answer normalisation (punctuation, digits, articles)."""
    s = str(s).replace("\n", " ").replace("\t", " ").strip().lower(); out = s
    for p in _PUNCT:
        out = out.replace(p, "") if ((p + " " in s or " " + p in s) or (re.search(_COMMA, s) is not None)) else out.replace(p, " ")
    out = _PERIOD.sub("", out)
    return " ".join(_NUMW.get(w, w) for w in out.split() if w not in _ARTS)

def vqa_acc(pred, answers):
    """official VQA accuracy: mean over leave-one-out subsets of min(1, #matches/3)."""
    p = vqa_norm(pred); g = [vqa_norm(a) for a in answers]
    if not g: return 0.0
    if len(g) < 2: return float(p == g[0])
    return float(np.mean([min(1.0, sum(o == p for o in (g[:i] + g[i + 1:])) / 3.0) for i in range(len(g))]))

def anls(pred, answers, tau=0.5):
    p = " ".join(str(pred).lower().split()); best = 0.0
    for a in answers:
        g = " ".join(str(a).lower().split()); d = lev(p, g) / max(1, max(len(p), len(g)))
        best = max(best, 1 - d if d < tau else 0.0)
    return best

def ans_score(ds, pred, answers):
    if ds in ("vqav2", "textvqa", "aokvqa"): return vqa_acc(pred, answers)
    if ds == "docvqa": return anls(pred, answers)
    return float(vqa_norm(pred) == vqa_norm(answers[0])) if answers else 0.0          # GQA, VSR, CLEVR: exact match

def first_token_stats(lp):
    p = lp.exp(); t2 = p.topk(2).values
    return dict(ent=float(-(p * lp).sum()), top1=float(t2[0]), margin=float(t2[0] - t2[1]), arg=int(lp.argmax()))

DEC_PRUNED = [f for f in CFG["DEC_GRID"] if f < 1.0]
BASE_FLAGS = dict(qsim=True, uniform=True, random=True, fastv=FASTV_OK, visionzip=VZ_OK, divprune=DIV_OK, cdpruner=CDP_OK, dart=DART_OK, pdrop=PD_OK)

def dec_main(r, ctx, rec):
    """full answer + answer at every decoded budget of the Q-EASE nested sets, with the pruned model's own first-token statistics."""
    G, mx = ctx["G"], CFG["GEN_MAX_NEW"]
    od, sel, h, gv = selector_order(r, ctx, "full")
    ks = [int(min(G, max(1, math.ceil(f * G)))) for f in DEC_PRUNED]
    gm = order_to_masks(od, G, ks)
    rhat = torch.sigmoid(sel.risk(h, gv, gm.to(r.device)[None])[0].float()).cpu().numpy()
    ids, lp_full = r.decode(ctx, ctx["emb"], torch.ones(ctx["L"], dtype=torch.bool, device=r.device), mx)
    pred_full = decode_text(r, ids)
    st_full = first_token_stats(lp_full)
    stages = []; lp0 = None
    for j, (f, k) in enumerate(zip(DEC_PRUNED, ks)):
        tk = r.group_to_token_mask(ctx, gm[j][None].to(r.device))[0]
        ids, lp = r.decode(ctx, ctx["emb"], tk, mx); pred = decode_text(r, ids)
        if lp0 is None: lp0 = lp
        st = first_token_stats(lp)
        st.update(frac=f, k=k, kfrac=float(kept_token_frac(ctx, gm[j][None])[0]), pred=pred, score=float(ans_score(rec["ds"], pred, rec["answers"])),
                  rhat=float(rhat[j]), js0=float(js_div(lp0, lp[None])[0]), agree0=float(lp.argmax() == lp0.argmax()), js_full=float(js_div(lp_full, lp[None])[0]))
        stages.append(st)
    full = dict(st_full, frac=1.0, k=G, kfrac=1.0, pred=pred_full, score=float(ans_score(rec["ds"], pred_full, rec["answers"])), rhat=0.0,
                js0=float(js_div(lp0, lp_full[None])[0]), agree0=float(lp_full.argmax() == lp0.argmax()), js_full=0.0)
    return dict(uid=rec["uid"], ds=rec["ds"], G=G, L=ctx["L"], n_vis=ctx["n_vis"], qlen=len(rec["question"].split()), stages=stages + [full])

def dec_base(r, ctx, rec):
    """baselines at DEC_BASE_FRACS, decoded with the very same decoder (physical pruning; in-LLM methods via their own layer loops)."""
    G, nv, L, mx, img = ctx["G"], ctx["n_vis"], ctx["L"], CFG["GEN_MAX_NEW"], ctx["img_idx"]
    methods = [m for m in CFG["DEC_BASE_METHODS"] if BASE_FLAGS.get(m)]
    orders = baseline_orders(ctx, stable_hash(rec["uid"]) % 10**6)
    fv = torch.argsort(-r.fastv_prefix(ctx, CFG["FASTV_K"])[2]) if "fastv" in methods else None
    dp = r.dart_prefix(ctx, CFG["DART_K"]) if "dart" in methods else None
    div_o = divprune_order(ctx).to(r.device) if "divprune" in methods else None
    cdp_o = cdpruner_order(ctx, CFG["CDP_THETA"]).to(r.device) if "cdpruner" in methods else None
    out = {m: {} for m in methods}
    def tok_keep(order, n):
        keep = torch.ones(L, dtype=torch.bool, device=r.device); keep[img] = False; keep[img[order[:n]]] = True; return keep
    for f in CFG["DEC_BASE_FRACS"]:
        n = int(min(nv, max(2, math.ceil(f * nv)))); k = int(min(G, max(1, math.ceil(f * G))))
        for m in methods:
            try:
                if m in ("qsim", "uniform", "random"):
                    gk = order_to_masks(orders[m], G, [k])[0].to(r.device); pred = decode_text(r, r.decode(ctx, ctx["emb"], r.group_to_token_mask(ctx, gk[None])[0], mx)[0])
                    kf = float(kept_token_frac(ctx, gk[None].cpu())[0])
                elif m == "fastv":
                    pred = r.fastv_generate(ctx, tok_keep(fv, n), CFG["FASTV_K"], mx); kf = n / nv
                elif m == "visionzip":
                    e_, keep_ = r.vz_embed(ctx, n); pred = r.generate_custom(ctx, e_, keep_, mx); kf = n / nv
                elif m in ("divprune", "cdpruner"):
                    pred = r.generate_custom(ctx, ctx["emb"], tok_keep(div_o if m == "divprune" else cdp_o, n), mx); kf = n / nv
                elif m == "dart":
                    keep, _ = dart_keep(ctx, dp[0], dp[2], n, CFG["DART_PIVOTS"]); pred = r.fastv_generate(ctx, keep, CFG["DART_K"], mx); kf = n / nv
                elif m == "pdrop":
                    lam = pdrop_lambda(f, len(r.decoder.layers)); lp_s, st_, plan = r.pdrop_run(ctx, lam)
                    pred = r.pdrop_generate(ctx, lam, plan, mx); kf = sum(st_["vis_per_layer"]) / len(r.decoder.layers) / nv
                out[m][str(f)] = dict(pred=pred, score=float(ans_score(rec["ds"], pred, rec["answers"])), kfrac=float(kf))
            except Exception as e:
                FAILS.append(("dec-base-" + m, rec["uid"], repr(e)[:200], traceback.format_exc()))
    return out

def dec_one(r, item, tick):
    rec, outdir, want_base = item
    path, bpath = outdir / f"{rec['uid']}.json", outdir / f"{rec['uid']}.base.json"
    need_main, need_base = not path.exists(), (want_base and not bpath.exists())
    if not (need_main or need_base):
        tick(); return None
    ctx = r.prepare(rec)
    if need_main: json.dump(dec_main(r, ctx, rec), open(path, "w"))
    if need_base: json.dump(dec_base(r, ctx, rec), open(bpath, "w"))
    tick(); return None

def load_dec(d, base=False):
    out = []
    for p in sorted(d.glob("*.json")):
        if p.name.endswith(".base.json"): continue
        o = json.load(open(p))
        if base:
            bp = d / f"{o['uid']}.base.json"
            o["base"] = json.load(open(bp)) if bp.exists() else None
        out.append(o)
    return out

def dec_roles(n=None):
    n = n or CFG["DEC_N"]
    return {"head": interleave_by_ds(BY_SPLIT["val"], n["head"], seed=61), "cal": interleave_by_ds(BY_SPLIT["cal"], n["cal"], seed=62),
            "test": interleave_by_ds(BY_SPLIT["test"], n["test"], seed=63)}

if "dec" in CFG["STAGES"]:
    assert "full" in SELECTORS, "train the selector first"
    CFG["TOKENS_PER_DS"]["std"] = CFG["TOKENS_PER_DS"]["default"]
    for _role, _recs in dec_roles().items():
        (DEC_DIR / _role).mkdir(parents=True, exist_ok=True)
        _items = [(x, DEC_DIR / _role, _role == "test" and i < CFG["DEC_BASE_N"]) for i, x in enumerate(_recs)]
        _todo = [it for it in _items if not (it[1] / f"{it[0]['uid']}.json").exists() or (it[2] and not (it[1] / f"{it[0]['uid']}.base.json").exists())]
        log(f"[dec:{_role}] {len(_recs)} questions, {len(_todo)} to decode ({'+baselines on first ' + str(CFG['DEC_BASE_N']) if _role == 'test' else 'cascade stages only'})"
            f" | greedy decoding is the slowest stage: lower CFG['DEC_N'] if needed")
        if _todo: parallel_map(_todo, dec_one, f"dec-{_role}", total=len(_todo))



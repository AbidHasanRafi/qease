"""Experiments C and E: downstream accuracy on held-out questions and on official benchmark splits (POPE, TextVQA, VQAv2, A-OKVQA)."""
import ast
import json
import math
import random
import traceback

from PIL import Image
import datasets as hfds
import numpy as np
import pandas as pd
import torch
from qease import state
from qease.extra import xstate
from qease.crc import first_ok, loss_tag
from qease.data import _save_img, _stream, _to_pil
from qease.evaluation import selector_order
from qease.selection import baseline_orders, k_list, kept_token_frac, order_to_masks
from qease.utils import stable_hash
from qease.extra.metrics import vqa_norm, x_score
from qease.extra.trained_baselines import X_KIND
from qease.extra.utils import x_boot_ci, x_holm, x_save, xd, xlog


def x_gen(r, ctx, gk, mx):
    """greedy decoding with PHYSICAL pruning of the dropped groups (positions kept).  Deliberately does not call r.generate():
    an older version of the main script decoded through a hook that silently did not prune, so this file is independent of it."""
    tk = torch.ones(ctx["L"], dtype=torch.bool, device=r.device) if gk is None else r.group_to_token_mask(ctx, gk[None])[0]
    return r.generate_custom(ctx, ctx["emb"], tk, mx)


def x_bench_recs(n_per_ds):
    by = {}
    for rec in sorted(state.BY_SPLIT["test"], key=lambda r: stable_hash("bench" + r["uid"])):
        by.setdefault(rec["ds"], [])
        if len(by[rec["ds"]]) < n_per_ds: by[rec["ds"]].append(rec)
    return [r for d in state.CFG["N_POOL"] for r in by.get(d, [])]


def x_ans_list(x):
    if isinstance(x, str):
        try: x = ast.literal_eval(x)
        except Exception: return [x]
    if not isinstance(x, (list, tuple)): return [str(x)]
    return [str(a["answer"]) if isinstance(a, dict) else str(a) for a in x]


def x_std_build(name, n):
    """n questions from the OFFICIAL validation/test split of a standard benchmark (cached; images go to the shared pool image dir)."""
    cache = state.POOL_DIR / f"std_{name}.jsonl"
    if cache.exists():
        rows = [json.loads(l) for l in open(cache)]
        if len(rows) >= n: return rows[:n]
    rows = []

    def add(im, q, ans, extra=None):
        uid = f"std_{name}_{len(rows)}"; rel = f"images/{uid}.jpg"
        _save_img(_to_pil(im), state.POOL_DIR / rel, 1024); w, h = Image.open(state.POOL_DIR / rel).size
        rows.append(dict(ds=f"std_{name}", uid=uid, img=rel, W=w, H=h, split="test", question=q, answers=ans, extra=extra or {}, bsub="stdbench"))
    try:
        if name == "pope":                                  # 3 blocks of 3000 (adversarial / popular / random); 5 random (yes,no)-aligned chunks per block
            rng = random.Random(3); chunk = max(2, n // 15)
            for off0 in (0, 3000, 6000):
                for _ in range(5):
                    off = off0 + rng.randrange(0, 3000 - chunk); off -= off % 2
                    for ex in hfds.load_dataset("lmms-lab-encoder/POPE", split="test", streaming=True).skip(off).take(chunk):
                        add(ex["image"], ex["question"], [str(ex["answer"]).strip().lower()], dict(category=ex["category"]))
        elif name == "textvqa":
            for ex in _stream("lmms-lab-encoder/textvqa", "validation"):
                if len(rows) >= n: break
                add(ex["image"], ex["question"], x_ans_list(ex["answers"]))
        elif name == "vqav2":
            for ex in _stream("lmms-lab-encoder/VQAv2", "validation", buf=500):
                if len(rows) >= n: break
                add(ex["image"], ex["question"], x_ans_list(ex["answers"]))
        elif name == "aokvqa":
            for ex in _stream("HuggingFaceM4/A-OKVQA", "validation"):
                if len(rows) >= n: break
                add(ex["image"], ex["question"], x_ans_list(ex["direct_answers"]))
        else:
            raise ValueError(name)
    except Exception as e:
        state.FAILS.append(("std-build", name, repr(e)[:200], traceback.format_exc())); xlog(f"standard benchmark '{name}' could not be fully loaded: {repr(e)[:200]} (kept {len(rows)})")
    if rows: cache.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return rows[:n]


def x_bench_one(r, rec, tick):
    path = xd(rec.get("bsub", "bench")) / f"{rec['uid']}.json"
    if path.exists(): tick(); return None
    mx = xstate.XCFG["BENCH_MAX_NEW"]
    ctx = r.prepare(rec); G = ctx["G"]; ks = k_list(G); nv, Lp = ctx["n_vis"], ctx["L"]
    orders = baseline_orders(ctx, stable_hash(rec["uid"]) % 10**6)
    qsel = selector_order(r, ctx, "full"); orders["qease"] = qsel[0]
    if "noq" in state.SELECTORS: orders["qease_noq"] = selector_order(r, ctx, "noq")[0]
    if xstate.XCFG["BENCH_USE_TB"]:
        for n in X_KIND:
            if n in state.SELECTORS: orders[n] = selector_order(r, ctx, n)[0]
    res = dict(uid=rec["uid"], ds=rec["ds"], pred={}, score={}, kfrac={}, agree={})

    def put(name, pred, kf):
        res["pred"][name] = pred; res["score"][name] = x_score(rec["ds"], pred, rec["answers"]); res["kfrac"][name] = float(kf)
        if name != "full": res["agree"][name] = float(vqa_norm(pred) == vqa_norm(res["pred"]["full"]))

    def group(name, od, k):
        gk = order_to_masks(od, G, [k])[0].to(r.device)
        put(name, x_gen(r, ctx, gk, mx), kept_token_frac(ctx, gk[None].cpu())[0])

    put("full", x_gen(r, ctx, None, mx), 1.0)
    fv_order = torch.argsort(-r.fastv_prefix(ctx, state.CFG["FASTV_K"])[2]) if state.FASTV_OK else None

    def fastv(name, frac):
        try:
            n = int(min(nv, max(1, math.ceil(frac * nv)))); keep = torch.ones(Lp, dtype=torch.bool, device=r.device)
            keep[ctx["img_idx"]] = False; keep[ctx["img_idx"][fv_order[:n]]] = True
            put(name, r.fastv_generate(ctx, keep, state.CFG["FASTV_K"], mx), n / nv)
        except Exception as e: state.FAILS.append(("bench-fastv", rec["uid"], repr(e)[:200], traceback.format_exc()))

    def vzip(name, frac):
        try:
            n = int(min(nv, max(2, math.ceil(frac * nv)))); e_, keep_ = r.vz_embed(ctx, n)
            put(name, r.generate_custom(ctx, e_, keep_, mx), n / nv)
        except Exception as e: state.FAILS.append(("bench-visionzip", rec["uid"], repr(e)[:200], traceback.format_exc()))

    for f in xstate.XCFG["BENCH_FRACS"]:                                              # (1) fixed budgets, every method
        k = int(min(G, max(1, math.ceil(f * G))))
        for n, od in orders.items():
            try: group(f"{n}@{f}", od, k)
            except Exception as e: state.FAILS.append(("bench-group", rec["uid"], repr(e)[:200], traceback.format_exc()))
        if fv_order is not None: fastv(f"fastv@{f}", f)
        if state.VZ_OK and ctx.get("vz") is not None: vzip(f"visionzip@{f}", f)
    _, sel, h, gv = qsel                                                       # (2) calibrated adaptive budget; baselines get the SAME per-question k
    gm = order_to_masks(orders["qease"], G, ks)
    rh = torch.sigmoid(sel.risk(h, gv, gm.to(r.device)[None])[0].float()).cpu().numpy()[None]
    for kind, eps, alpha in xstate.XCFG["BENCH_ADAPT"]:
        lam = state.LAMBDAS.get(("qease", kind, eps, alpha))
        if lam is None: continue
        k = ks[int(first_ok(rh, lam)[0])]; tag = f"adaptive[{loss_tag(kind, eps)},a={alpha}]"
        for n in [m for m in ("qease", "qease_noq", "qsim", "uniform", "random", "attn", "mlp") if m in orders]:
            try: group(f"{n}@{tag}", orders[n], k)
            except Exception as e: state.FAILS.append(("bench-adapt", rec["uid"], repr(e)[:200], traceback.format_exc()))
        if fv_order is not None: fastv(f"fastv@{tag}", k / G)
        if state.VZ_OK and ctx.get("vz") is not None: vzip(f"visionzip@{tag}", k / G)
    if state.SV_OK:                                                                  # (3) SparseVLM at official budgets + compute-matched group methods
        nlay = len(r.decoder.layers)
        for ret in xstate.XCFG["BENCH_SPARSE"]:
            try:
                lp_s, st, plan = r.sparsevlm_run(ctx, ret)
                put(f"sparsevlm@sv{ret}", r.sparsevlm_generate(ctx, ret, lp_s, plan, mx), sum(st["vis_per_layer"]) / nlay / nv)
                eff = sum(st["seq_per_layer"]) / nlay / Lp
                f_star = float(np.clip((eff * Lp - (Lp - nv)) / nv, 1.0 / nv, 1.0)); k = int(min(G, max(1, math.ceil(f_star * G))))
                for n in [m for m in ("qease", "qsim", "uniform", "random") if m in orders]: group(f"{n}@sv{ret}", orders[n], k)
            except Exception as e: state.FAILS.append(("bench-sparsevlm", rec["uid"], repr(e)[:200], traceback.format_exc()))
    json.dump(res, open(path, "w")); tick(); return None


def x_bench_report(sub="bench", ds_order=None, name="bench"):
    rows = []
    for p in sorted(xd(sub).glob("*.json")):
        d = json.load(open(p))
        for c, s in d["score"].items():
            rows.append(dict(uid=d["uid"], ds=d["ds"], config=c, score=s, kfrac=d["kfrac"].get(c, np.nan), agree=d["agree"].get(c, 1.0 if c == "full" else np.nan)))
    if not rows:
        xlog("no benchmark results yet"); return None
    D = pd.DataFrame(rows); D.to_csv(xd("results") / f"{name}_raw.csv", index=False)
    dsets = [d for d in (ds_order or list(state.CFG["N_POOL"])) if d in set(D.ds)]
    cfgs = sorted(D.config.unique(), key=lambda c: (c != "full", c))
    # per-dataset table
    pr = []
    for d in dsets:
        for c in cfgs:
            x = D[(D.ds == d) & (D.config == c)]
            m, lo, hi = x_boot_ci(x.score.values)
            pr.append(dict(dataset=d, config=c, n=len(x), score=m, lo=lo, hi=hi, kfrac=x.kfrac.mean(), agree_with_full=x.agree.mean()))
    PR = pd.DataFrame(pr); x_save(PR, f"{name}_per_dataset", "official metric per dataset")
    # macro average over datasets with a stratified bootstrap (questions resampled inside each dataset)
    B = xstate.XCFG["BOOT_B"]; rng = np.random.default_rng(0); boot = np.zeros((B, len(cfgs))); S_by = {}
    for d in dsets:
        piv = D[D.ds == d].pivot_table(index="uid", columns="config", values="score").reindex(columns=cfgs)
        S = piv.values; S_by[d] = S
        idx = rng.integers(0, len(S), (B, len(S)))
        boot += np.nanmean(S[idx], axis=1) / len(dsets)
    macro = np.mean([np.nanmean(S_by[d], axis=0) for d in dsets], axis=0)
    kfr = D.groupby("config").kfrac.mean().reindex(cfgs).values; agr = D.groupby("config").agree.mean().reindex(cfgs).values
    full_i = cfgs.index("full") if "full" in cfgs else None
    MA = pd.DataFrame(dict(config=cfgs, macro_score=macro, lo=np.percentile(boot, 2.5, axis=0), hi=np.percentile(boot, 97.5, axis=0), mean_kept_frac=kfr,
                           agree_with_full=agr, relative_to_full_pct=(100 * macro / macro[full_i]) if full_i is not None else np.nan))
    x_save(MA, f"{name}_macro", "macro average over datasets (official metrics)")
    # paired comparison of Q-EASE against every other method at the same operating point
    pc = []
    for c in cfgs:
        if not c.startswith("qease@"): continue
        tag = c.split("@", 1)[1]; iq = cfgs.index(c)
        for o in cfgs:
            if o == c or not o.endswith("@" + tag) or o.startswith("qease@"): continue
            io = cfgs.index(o); dd = boot[:, iq] - boot[:, io]
            p = max(2 * min((dd <= 0).mean(), (dd >= 0).mean()), 1.0 / B)
            pc.append(dict(operating_point=tag, baseline=o.split("@")[0], qease_macro=macro[iq], baseline_macro=macro[io], diff=macro[iq] - macro[io],
                           lo=float(np.percentile(dd, 2.5)), hi=float(np.percentile(dd, 97.5)), p_boot=p))
    if pc:
        PC = pd.DataFrame(pc); PC["p_holm"] = x_holm(PC.p_boot.values); x_save(PC, f"{name}_paired_vs_qease","macro-accuracy difference Q-EASE minus baseline (paired bootstrap, Holm)")
    key = [c for c in cfgs if c == "full" or c.endswith("@0.25") or c.endswith("@0.5")]
    xlog("macro accuracy (official metrics), key configs:\n" + MA[MA.config.isin(key)].round(3).to_string(index=False))
    return MA

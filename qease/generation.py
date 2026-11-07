"""Downstream accuracy with physically pruned greedy decoding."""
import json
import math
import re
import string
import traceback

import numpy as np
import torch
from qease import state
from qease.crc import first_ok, loss_tag
from qease.evaluation import selector_order
from qease.parallel import parallel_map
from qease.selection import baseline_orders, k_list, kept_token_frac, order_to_masks
from qease.utils import log, stable_hash


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


def gen_one(r, rec, tick):
    path = state.WORK / "gen" / f"{rec['uid']}.json"
    if path.exists():
        tick(); return json.load(open(path))
    ctx = r.prepare(rec)
    G = ctx["G"]; ks = k_list(G)
    orders = baseline_orders(ctx, stable_hash(rec["uid"]) % 10**6)
    qsel = None
    if "full" in state.SELECTORS:
        qsel = selector_order(r, ctx, "full"); orders["qease"] = qsel[0]
    if "noq" in state.SELECTORS:
        orders["qease_noq"] = selector_order(r, ctx, "noq")[0]
    res = dict(uid=rec["uid"], ds=rec["ds"], acc={}, agree={}, kfrac={}, pred={})

    def run(cfg_name, gk):
        pred = r.generate(ctx, gk, state.CFG["GEN_MAX_NEW"])
        res["pred"][cfg_name] = pred
        res["acc"][cfg_name] = float(score_answer(rec["ds"], pred, rec["answers"]))
        res["kfrac"][cfg_name] = 1.0 if gk is None else float(kept_token_frac(ctx, gk[None].cpu())[0])
        if cfg_name != "full":
            res["agree"][cfg_name] = float(norm_ans(pred) == norm_ans(res["pred"]["full"]))

    fv_order = None
    if state.FASTV_OK:
        fv_order = torch.argsort(-r.fastv_prefix(ctx, state.CFG["FASTV_K"])[2])

    def run_fv(cfg_name, frac):                                  # FastV with generation (pruned from layer K, token-level)
        try:                                                     # a FastV problem must never discard the other methods' results
            n = int(min(ctx["n_vis"], max(1, math.ceil(frac * ctx["n_vis"]))))
            keep = torch.ones(ctx["L"], dtype=torch.bool, device=r.device); keep[ctx["img_idx"]] = False
            keep[ctx["img_idx"][fv_order[:n]]] = True
            pred = r.fastv_generate(ctx, keep, state.CFG["FASTV_K"], state.CFG["GEN_MAX_NEW"])
            res["pred"][cfg_name] = pred
            res["acc"][cfg_name] = float(score_answer(rec["ds"], pred, rec["answers"]))
            res["kfrac"][cfg_name] = n / ctx["n_vis"]
            res["agree"][cfg_name] = float(norm_ans(pred) == norm_ans(res["pred"]["full"]))
        except Exception as e:
            state.FAILS.append(("gen-fastv", rec["uid"], repr(e)[:200], traceback.format_exc()))

    def run_vz(cfg_name, frac):                                  # VisionZip with generation (dominant + merged contextual tokens)
        try:
            n = int(min(ctx["n_vis"], max(2, math.ceil(frac * ctx["n_vis"]))))
            e_, keep_ = r.vz_embed(ctx, n)
            pred = r.generate_custom(ctx, e_, keep_, state.CFG["GEN_MAX_NEW"])
            res["pred"][cfg_name] = pred
            res["acc"][cfg_name] = float(score_answer(rec["ds"], pred, rec["answers"]))
            res["kfrac"][cfg_name] = n / ctx["n_vis"]
            res["agree"][cfg_name] = float(norm_ans(pred) == norm_ans(res["pred"]["full"]))
        except Exception as e:
            state.FAILS.append(("gen-visionzip", rec["uid"], repr(e)[:200], traceback.format_exc()))

    run("full", None)
    for f in state.CFG["GEN_FRACS"]:                                   # fixed budgets
        k = int(min(G, max(1, math.ceil(f * G))))
        for name, od in orders.items():
            run(f"{name}@{f}", order_to_masks(od, G, [k])[0].to(r.device))
        if fv_order is not None: run_fv(f"fastv@{f}", f)
        if state.VZ_OK and ctx.get("vz") is not None: run_vz(f"visionzip@{f}", f)
    if qsel is not None and state.LAMBDAS:                             # calibrated adaptive budget; baselines get the SAME per-sample k
        _, sel, h, gv = qsel
        gm = order_to_masks(orders["qease"], G, ks)
        rh = torch.sigmoid(sel.risk(h, gv, gm.to(r.device)[None])[0].float()).cpu().numpy()[None]
        for kind, eps, alpha in state.CFG["GEN_ADAPT"]:
            lam = state.LAMBDAS.get(("qease", kind, eps, alpha))
            if lam is None: continue
            k = ks[int(first_ok(rh, lam)[0])]
            tag = f"adaptive[{loss_tag(kind, eps)},a={alpha}]"
            for name in ["qease", "qsim", "uniform", "random"]:
                if name in orders:
                    run(f"{name}@{tag}", order_to_masks(orders[name], G, [k])[0].to(r.device))
            if fv_order is not None: run_fv(f"fastv@{tag}", k / G)
            if state.VZ_OK and ctx.get("vz") is not None: run_vz(f"visionzip@{tag}", k / G)
    if state.SV_OK:                                                    # SparseVLM at official budgets; group methods at the SAME effective compute
        nlay, nv, Lp = len(r.decoder.layers), ctx["n_vis"], ctx["L"]
        for ret in state.CFG["GEN_SPARSE_RETAIN"]:
            try:
                lp_s, st, plan = r.sparsevlm_run(ctx, ret)
                pred = r.sparsevlm_generate(ctx, ret, lp_s, plan, state.CFG["GEN_MAX_NEW"])
                tag = f"sv{ret}"; eff = sum(st["seq_per_layer"]) / nlay / Lp
                res["pred"][f"sparsevlm@{tag}"] = pred
                res["acc"][f"sparsevlm@{tag}"] = float(score_answer(rec["ds"], pred, rec["answers"]))
                res["kfrac"][f"sparsevlm@{tag}"] = sum(st["vis_per_layer"]) / nlay / nv           # average visual tokens per layer / N
                res["agree"][f"sparsevlm@{tag}"] = float(norm_ans(pred) == norm_ans(res["pred"]["full"]))
                f_star = float(np.clip((eff * Lp - (Lp - nv)) / nv, 1.0 / nv, 1.0)); k = int(min(G, max(1, math.ceil(f_star * G))))
                for name in ["qease", "qsim", "uniform", "random"]:
                    if name in orders:
                        run(f"{name}@{tag}", order_to_masks(orders[name], G, [k])[0].to(r.device))
            except Exception as e:
                state.FAILS.append(("gen-sparsevlm", rec["uid"], repr(e)[:200], traceback.format_exc()))
    json.dump(res, open(path, "w")); tick(); return res


def run_gen():
    """Stage 'gen': downstream accuracy with physically pruned greedy decoding."""
    _recs = []
    for _d in state.CFG["N_POOL"]:
        _recs += [x for x in state.BY_SPLIT["test"] if x["ds"] == _d][: state.CFG["GEN_N_PER_DS"]]
    log(f"[gen] {len(_recs)} test samples  (generation is the slowest stage; lower CFG['GEN_N_PER_DS'] if needed)")
    parallel_map(_recs, gen_one, "gen", total=len(_recs))

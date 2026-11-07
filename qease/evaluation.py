"""Evaluation of every selection method on the calibration and test splits."""
import math
import traceback

import numpy as np
import torch
from qease import state
from qease.data import interleave_by_ds
from qease.parallel import parallel_map
from qease.selection import METHODS, baseline_orders, k_list, kept_token_frac, matched_group_eval, order_to_masks, token_level_curves
from qease.selector import Selector
from qease.utils import log, stable_hash
from qease.vlm.helpers import js_div


def load_selectors():
    """Restore trained selectors from disk (lets eval/crc/gen run without re-running 'train')."""
    for v in ("full", "noq"):
        p = state.WORK / "ckpt" / f"selector_{v}.pt"
        if v not in state.SELECTORS and p.exists():
            st = torch.load(p, map_location="cpu")["state"]
            m = Selector(st["gp.0.weight"].shape[0], state.CFG["D_MODEL"], state.CFG["N_LAYERS"], state.CFG["N_HEADS"], v == "full")
            m.load_state_dict(st); state.SELECTORS[v] = m.eval()
            log(f"restored selector '{v}' from {p.name}")


def get_sel(r, name):
    """per-device selector copy (cached)"""
    key = (name, r.device)
    if key not in state._SEL_CACHE:
        state._SEL_CACHE[key] = _clone_to(state.SELECTORS[name], r.device)
    return state._SEL_CACHE[key]


def _clone_to(m, dev):
    import copy
    return copy.deepcopy(m).to(dev).eval()


@torch.inference_mode()
def selector_order(r, ctx, name):
    """returns (order over groups desc by score, scores, selector model, encoded h, gvalid)"""
    sel = get_sel(r, name)
    G, GM = ctx["G"], state.CFG["GMAX"]
    gf = ctx["gfeat"][None]; gi = torch.arange(G, device=r.device)
    gpos = torch.stack([(gi // ctx["gw"]).float() / max(1, ctx["gh"] - 1), (gi % ctx["gw"]).float() / max(1, ctx["gw"] - 1)], 1)[None]
    gv = torch.ones(1, G, dtype=torch.bool, device=r.device)
    qe = ctx["qemb"][None]; qv = torch.ones(1, qe.shape[1], dtype=torch.bool, device=r.device)
    h, s = sel.encode(gf, gpos, gv, qe, qv)
    return torch.argsort(-s[0]).cpu(), sel, h, gv


def eval_one(r, rec, tick):
    ctx = r.prepare(rec)
    G = ctx["G"]; ks = k_list(G)
    lp_full = r.logp_full(ctx)
    orders = baseline_orders(ctx, stable_hash(rec["uid"]) % 10**6)
    rhat = {}
    for name, sname in [("qease", "full"), ("qease_noq", "noq")]:
        if sname not in state.SELECTORS: continue
        od, sel, h, gv = selector_order(r, ctx, sname)
        orders[name] = od
        gm = order_to_masks(od, G, ks).to(r.device)
        rhat[name] = torch.sigmoid(sel.risk(h, gv, gm[None])[0].float()).cpu().numpy()   # predicted JS/ln2
    names = [n for n in METHODS if n in orders]
    nk = len(ks)
    ins = torch.cat([order_to_masks(orders[n], G, ks) for n in names], 0)       # INSERTION: keep the first-k ranked groups
    allm = torch.cat([ins, ~ins], 0)                                            # DELETION : drop the first-k ranked groups
    lpm = r.logp_for_token_masks(ctx, r.group_to_token_mask(ctx, allm.to(r.device)))
    jsv = js_div(lp_full, lpm).cpu().numpy().reshape(2, len(names), nk)
    flp = (lpm.argmax(-1) != lp_full.argmax()).cpu().numpy().reshape(2, len(names), nk)
    pf = lp_full.exp()
    j25 = int(np.argmin(np.abs(np.array(state.CFG["K_GRID"]) - 0.25)))
    out = dict(uid=rec["uid"], ds=rec["ds"], G=G, q=rec["question"][:160],
               geom=dict(H=ctx["H"], W=ctx["W"], win=ctx["win"], gh=ctx["gh"], gw=ctx["gw"]),
               ent=float(-(pf * lp_full).sum()), ptop1=float(pf.max()),
               rhat=rhat, js={}, flip={}, js_del={}, flip_del={}, kfrac={}, sel25={})
    for i, n in enumerate(names):
        out["js"][n], out["flip"][n] = jsv[0, i], flp[0, i]
        out["js_del"][n], out["flip_del"][n] = jsv[1, i], flp[1, i]
        out["kfrac"][n] = kept_token_frac(ctx, ins[i * nk:(i + 1) * nk])
        out["sel25"][n] = ins[i * nk + j25].numpy()
    out["L"], out["n_vis"] = ctx["L"], ctx["n_vis"]
    # ---- external published scorers (plug-in), pre-LLM token-level pruning
    for name, fn in state.EXTERNAL_SCORERS.items():
        try:
            cv = token_level_curves(r, ctx, lp_full, torch.argsort(-fn(r, ctx).float().flatten().to(r.device)))
            for key in ("js", "flip", "js_del", "flip_del", "kfrac"): out[key][name] = cv[key]
        except Exception as e:
            state.FAILS.append(("external-scorer", name, repr(e)[:200], traceback.format_exc()))
    # ---- FastV (K, last-token attention at layer K-1, pruning from layer K on) at TOKEN granularity, as published
    if state.FASTV_OK:
        K = state.CFG["FASTV_K"]; nv, L = ctx["n_vis"], ctx["L"]; nlay = len(r.decoder.layers)
        h_pref, pe, sc = r.fastv_prefix(ctx, K)
        order = torch.argsort(-sc)
        ns = [int(min(nv, max(1, math.ceil(f * nv)))) for f in state.CFG["K_GRID"]]
        lp_i, lp_d, eff = [], [], []
        for n in ns:
            keep = torch.ones(L, dtype=torch.bool, device=r.device); keep[ctx["img_idx"]] = False; keep[ctx["img_idx"][order[:n]]] = True
            lp_i.append(r.fastv_logp(ctx, h_pref, pe, keep, K))
            drop = torch.ones(L, dtype=torch.bool, device=r.device); drop[ctx["img_idx"][order[:n]]] = False
            lp_d.append(r.fastv_logp(ctx, h_pref, pe, drop, K))
            eff.append((K + (nlay - K) * (L - nv + n) / L) / nlay)          # effective prefill compute vs full model
        lp_i, lp_d = torch.stack(lp_i), torch.stack(lp_d)
        out["js"]["fastv"], out["js_del"]["fastv"] = js_div(lp_full, lp_i).cpu().numpy(), js_div(lp_full, lp_d).cpu().numpy()
        out["flip"]["fastv"] = (lp_i.argmax(-1) != lp_full.argmax()).cpu().numpy()
        out["flip_del"]["fastv"] = (lp_d.argmax(-1) != lp_full.argmax()).cpu().numpy()
        out["kfrac"]["fastv"] = np.array(ns) / nv
        out["fastv_flops_frac"] = np.array(eff)
        out["fastv_matched"] = matched_group_eval(r, ctx, lp_full, orders, G, eff)       # group methods at FastV's effective compute
    # ---- VisionZip (dominant + merged contextual tokens; pre-LLM; budgets = the same K_GRID fractions)
    if state.VZ_OK:
        nv, L = ctx["n_vis"], ctx["L"]
        ns = [int(min(nv, max(2, math.ceil(f * nv)))) for f in state.CFG["K_GRID"]]
        lp_v = torch.stack([r.logp_prefill(ctx, *r.vz_embed(ctx, n)) for n in ns])
        dele = token_level_curves(r, ctx, lp_full, torch.argsort(-ctx["vz"]["score"]))     # deletion = drop top-n by VisionZip score (no merging)
        out["js"]["visionzip"], out["flip"]["visionzip"] = js_div(lp_full, lp_v).cpu().numpy(), (lp_v.argmax(-1) != lp_full.argmax()).cpu().numpy()
        out["js_del"]["visionzip"], out["flip_del"]["visionzip"] = dele["js_del"], dele["flip_del"]
        out["kfrac"]["visionzip"] = np.array(ns) / nv
    # ---- SparseVLM v1.0 at its 4 official budgets (in-LLM, progressive) + compute-matched group methods
    if state.SV_OK:
        nv, L, nlay = ctx["n_vis"], ctx["L"], len(r.decoder.layers)
        sp = dict(retain=list(state.CFG["SPARSE_RETAIN"]), js=[], flip=[], kfrac_avg=[], eff=[])
        for ret in sp["retain"]:
            lp_s, st, _ = r.sparsevlm_run(ctx, ret)
            sp["js"].append(float(js_div(lp_full, lp_s[None])[0])); sp["flip"].append(bool(lp_s.argmax() != lp_full.argmax()))
            sp["kfrac_avg"].append(sum(st["vis_per_layer"]) / nlay / nv); sp["eff"].append(sum(st["seq_per_layer"]) / nlay / L)
        sp["matched"] = matched_group_eval(r, ctx, lp_full, orders, G, sp["eff"])
        out["sparse"] = sp
    tick(); return out


def eval_shard(r, shard, tick):
    split, sid, recs = shard
    path = state.WORK / "eval" / f"{split}_{sid:04d}.pt"
    if path.exists(): tick(len(recs)); return 0
    out = []
    for rec in recs:
        if state.STOP.is_set(): return 0
        try: out.append(eval_one(r, rec, lambda *a: None))
        except Exception as e:
            state.FAILS.append(("eval", rec["uid"], repr(e)[:200], traceback.format_exc()))
            if "out of memory" in str(e).lower(): torch.cuda.empty_cache()
        tick()
    torch.save(out, path); return len(out)


def run_eval():
    """Stage 'eval': evaluate every selection method on the calibration and test splits."""
    assert state.SELECTORS, "train the selector first (stage 'train')"
    for split, n in [("cal", state.CFG["CAL_N"]), ("test", state.CFG["TEST_N"])]:
        recs = interleave_by_ds(state.BY_SPLIT[split], n, seed=3)
        S = 25
        shards = [(split, i, recs[i * S:(i + 1) * S]) for i in range(math.ceil(len(recs) / S))]
        todo = [s for s in shards if not (state.WORK / "eval" / f"{s[0]}_{s[1]:04d}.pt").exists()]
        log(f"[eval:{split}] {len(recs)} samples, {len(todo)} shards to do")
        if todo:
            parallel_map(todo, eval_shard, f"eval-{split}", total=sum(len(s[2]) for s in todo))

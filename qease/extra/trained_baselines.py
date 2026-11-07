"""Experiment A: trained baselines with equal data and architecture but different supervision (attention-distilled, MLP, shuffled-label)."""
import gc
import math
import traceback

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from qease import state
from qease.extra import xstate
from qease.evaluation import selector_order
from qease.selection import k_list, order_to_masks
from qease.selector import Selector, _pairs, assemble, spearman_rows
from qease.vlm.helpers import js_div
from qease.extra.utils import LN2, xd, xlog


class MLPScorer(nn.Module):
    """per-group scorer WITHOUT cross-group attention (same inputs as the selector; the question interacts multiplicatively)."""
    def __init__(self, D, d=256):
        super().__init__()
        self.gp = nn.Sequential(nn.LayerNorm(D), nn.Linear(D, d))
        self.qp = nn.Sequential(nn.LayerNorm(D), nn.Linear(D, d))
        self.pos = nn.Sequential(nn.Linear(2, d), nn.GELU(), nn.Linear(d, d))
        self.head = nn.Sequential(nn.LayerNorm(3 * d), nn.Linear(3 * d, d), nn.GELU(), nn.Dropout(0.1), nn.Linear(d, 1))

    def encode(self, gf, gpos, gvalid, qe, qvalid):
        g = self.gp(gf.float()); q = self.qp(qe.float())
        qm = (q * qvalid[..., None].float()).sum(1) / qvalid.float().sum(1, keepdim=True).clamp_min(1)
        qm = qm[:, None].expand_as(g)
        s = self.head(torch.cat([g, qm * g, self.pos(gpos)], -1)).squeeze(-1).masked_fill(~gvalid, -1e4)
        return g, s


X_KIND = dict(attn="attn", mlp="mlp", shuf="shuf")


def x_load_tb():
    """restore trained-baseline checkpoints into SELECTORS (so selector_order / get_sel work for them)."""
    for name, kind in X_KIND.items():
        p = state.WORK / "ckpt" / f"selector_{name}.pt"
        if name not in state.SELECTORS and p.exists():
            st = torch.load(p, map_location="cpu")["state"]
            D = st["gp.0.weight"].shape[0]
            m = MLPScorer(D) if kind == "mlp" else Selector(D, state.CFG["D_MODEL"], state.CFG["N_LAYERS"], state.CFG["N_HEADS"], True)
            m.load_state_dict(st); state.SELECTORS[name] = m.eval()
            xlog(f"restored trained baseline '{name}' from {p.name}")


def x_train(name, kind, device, recs, A_map=None):
    ck = state.WORK / "ckpt" / f"selector_{name}.pt"
    D = recs[0]["gfeat"].shape[1]
    torch.manual_seed(state.CFG["SEED"])
    model = (MLPScorer(D) if kind == "mlp" else Selector(D, state.CFG["D_MODEL"], state.CFG["N_LAYERS"], state.CFG["N_HEADS"], True)).to(device)
    if ck.exists():
        model.load_state_dict(torch.load(ck, map_location=device)["state"]); return model.eval().cpu()
    T = assemble(recs, device); N = len(recs)
    val_idx = [i for i, r in enumerate(recs) if r["split"] == "val"]; tr_idx = [i for i, r in enumerate(recs) if r["split"] != "val"]
    A = None
    if kind == "attn":
        A = torch.zeros(N, state.CFG["GMAX"], device=device)
        for i, r in enumerate(recs):
            a = A_map.get(r["uid"])
            if a is not None: A[i, :len(a)] = torch.as_tensor(a, device=device, dtype=torch.float32)
        A = A / A.sum(1, keepdim=True).clamp_min(1e-9); has = A.sum(1) > 0
        tr_idx = [i for i in tr_idx if bool(has[i])]; val_idx = [i for i in val_idx if bool(has[i])]
    if kind == "shuf":                                            # labels of a RANDOM other image: same marginals, no information
        perm = torch.randperm(N, generator=torch.Generator().manual_seed(123)).to(device); T["js"] = T["js"][perm]
    ia, ib = [x.to(device) for x in _pairs()]
    opt = torch.optim.AdamW(model.parameters(), lr=state.CFG["LR"], weight_decay=state.CFG["WD"])
    E = xstate.XCFG["TB_EPOCHS"]; steps = E * math.ceil(len(tr_idx) / state.CFG["BATCH"])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, state.CFG["LR"], total_steps=steps, pct_start=0.1)
    scaler = torch.cuda.amp.GradScaler(enabled=device.startswith("cuda"))

    def batch_loss(idx):
        b = {k: v[idx] for k, v in T.items() if k != "rid"}
        with torch.autocast("cuda", dtype=torch.float16, enabled=device.startswith("cuda")):
            h, s = model.encode(b["gf"], b["gpos"], b["gv"], b["qe"], b["qv"])
            rl = model.risk(h, b["gv"], b["masks"]) if kind == "shuf" else None
        s = s.float()
        if kind == "attn":
            lp = torch.log_softmax(s.masked_fill(~b["gv"], -1e4), -1)
            loss = -(A[idx] * lp).sum(1).mean()
            return loss, loss.detach(), spearman_rows(s.masked_fill(~b["gv"], -1e4), A[idx])
        S = torch.einsum("bkg,bg->bk", b["masks"].float(), s.masked_fill(~b["gv"], 0))
        diff = S[:, ia] - S[:, ib]; dj = b["js"][:, ib] - b["js"][:, ia]; valid = dj.abs() > state.CFG["RANK_MARGIN"]
        l_rank = (F.binary_cross_entropy_with_logits(diff, (dj > 0).float(), reduction="none") * valid).sum() / valid.sum().clamp_min(1)
        acc = (((diff > 0) == (dj > 0)) & valid).sum() / valid.sum().clamp_min(1)
        loss = l_rank
        if kind == "shuf":
            loss = F.binary_cross_entropy_with_logits(rl.float(), (b["js"] / LN2).clamp(0, 1)) + state.CFG["RANK_W"] * l_rank
        return loss, loss.detach(), acc.detach().item()

    best, best_state, hist = 1e9, None, []
    for ep in range(E):
        model.train(); perm = torch.tensor(tr_idx)[torch.randperm(len(tr_idx))]; tl = 0.0
        for i in range(0, len(perm), state.CFG["BATCH"]):
            idx = perm[i:i + state.CFG["BATCH"]].to(device)
            loss = batch_loss(idx)[0]
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); scaler.step(opt); scaler.update(); sched.step(); tl += loss.item() * len(idx)
        model.eval()
        with torch.no_grad():
            vs = [batch_loss(torch.tensor(val_idx[i:i + 64], device=device)) for i in range(0, len(val_idx), 64)]
        vl = float(np.mean([v[1].item() for v in vs])); vm = float(np.mean([v[2] for v in vs]))
        hist.append(dict(name=name, epoch=ep, train_loss=tl / max(1, len(tr_idx)), val_loss=vl, val_metric=vm))
        if vl < best: best, best_state = vl, {k: x.detach().clone().cpu() for k, x in model.state_dict().items()}
        if ep % 5 == 0 or ep == E - 1: xlog(f"[train:{name}] ep{ep:02d} train={tl / max(1, len(tr_idx)):.4f} val={vl:.4f} metric={vm:.3f}")
    model.load_state_dict(best_state); model.eval()
    torch.save(dict(state=best_state, name=name, val=best), ck)
    pd.DataFrame(hist).to_csv(xd() / f"history_{name}.csv", index=False)
    del T, A; gc.collect(); torch.cuda.empty_cache()
    return model.cpu()


def x_attn_shard(r, shard, tick):
    sid, recs = shard
    path = xd("attn") / f"shard_{sid:04d}.pt"
    if path.exists(): tick(len(recs)); return 0
    out = {}
    for rec in recs:
        if state.STOP.is_set(): return 0
        try:
            ctx = r.prepare(rec)
            sc = r.fastv_prefix(ctx, state.CFG["FASTV_K"])[2].float()                       # FastV importance of every visual token
            out[rec["uid"]] = (torch.zeros(ctx["G"], device=r.device).index_add_(0, ctx["gid"], sc) / ctx["gcount"]).cpu().numpy()
        except Exception as e:
            state.FAILS.append(("tb-attn", rec["uid"], repr(e)[:200], traceback.format_exc()))
        tick()
    torch.save(out, path); return len(out)


def x_tbeval_shard(r, shard, tick):
    sid, recs = shard
    path = xd("tbeval") / f"test_{sid:04d}.pt"
    if path.exists(): tick(len(recs)); return 0
    out = []
    names = [n for n in ["full", "noq"] + list(X_KIND) if n in state.SELECTORS]
    for rec in recs:
        if state.STOP.is_set(): return 0
        try:
            ctx = r.prepare(rec); G = ctx["G"]; ks = k_list(G); lp_full = r.logp_full(ctx)
            ins = torch.cat([order_to_masks(selector_order(r, ctx, n)[0], G, ks) for n in names], 0)
            lpm = r.logp_for_token_masks(ctx, r.group_to_token_mask(ctx, ins.to(r.device)))
            jsv = js_div(lp_full, lpm).cpu().numpy().reshape(len(names), len(ks))
            flp = (lpm.argmax(-1) != lp_full.argmax()).cpu().numpy().reshape(len(names), len(ks))
            out.append(dict(uid=rec["uid"], ds=rec["ds"], js={n: jsv[i] for i, n in enumerate(names)}, flip={n: flp[i] for i, n in enumerate(names)}))
        except Exception as e:
            state.FAILS.append(("tb-eval", rec["uid"], repr(e)[:200], traceback.format_exc()))
        tick()
    torch.save(out, path); return len(out)

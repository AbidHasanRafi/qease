"""The amortized selector: question-guided group scorer (QGS), set-level risk head (SRH) and their training."""
import gc
import math
import threading

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from qease import state
from qease.utils import log


class Selector(nn.Module):
    """Question-conditioned group scorer + set-level risk head (a few M parameters)."""
    def __init__(self, D, d=256, layers=3, heads=4, use_q=True):
        super().__init__()
        self.use_q = use_q
        self.gp = nn.Sequential(nn.LayerNorm(D), nn.Linear(D, d))
        self.qp = nn.Sequential(nn.LayerNorm(D), nn.Linear(D, d))
        self.pos = nn.Sequential(nn.Linear(2, d), nn.GELU(), nn.Linear(d, d))
        self.type_emb = nn.Embedding(2, d)
        enc = nn.TransformerEncoderLayer(d, heads, 4 * d, 0.1, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(enc, layers, enable_nested_tensor=False)
        self.score = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
        self.mask_emb = nn.Embedding(2, d)
        self.rl = nn.TransformerEncoderLayer(d, heads, 2 * d, 0.0, batch_first=True, norm_first=True)
        self.ro = nn.Sequential(nn.LayerNorm(2 * d + 1), nn.Linear(2 * d + 1, d), nn.GELU(), nn.Linear(d, 1))

    def encode(self, gf, gpos, gvalid, qe, qvalid):
        g = self.gp(gf.float()) + self.pos(gpos) + self.type_emb.weight[0]
        if self.use_q:
            q = self.qp(qe.float()) + self.type_emb.weight[1]
            x = torch.cat([q, g], 1); kpm = torch.cat([~qvalid, ~gvalid], 1)
            h = self.enc(x, src_key_padding_mask=kpm)[:, qe.shape[1]:]
        else:
            h = self.enc(g, src_key_padding_mask=~gvalid)
        return h, self.score(h).squeeze(-1).masked_fill(~gvalid, -1e4)

    def risk(self, h, gvalid, masks):
        B, K, G = masks.shape
        x = (h[:, None] + self.mask_emb(masks.long())).reshape(B * K, G, -1)
        v = gvalid[:, None].expand(B, K, G).reshape(B * K, G)
        x = self.rl(x, src_key_padding_mask=~v)
        vf = v.float().unsqueeze(-1); dropped = (~masks.reshape(B * K, G) & v).float().unsqueeze(-1)
        mean_all = (x * vf).sum(1) / vf.sum(1).clamp_min(1)
        mean_drop = (x * dropped).sum(1) / dropped.sum(1).clamp_min(1)
        keepfrac = ((masks.reshape(B * K, G) & v).float().sum(1) / v.float().sum(1).clamp_min(1)).unsqueeze(-1)
        return self.ro(torch.cat([mean_all, mean_drop, keepfrac], -1)).view(B, K)


def load_probe_records():
    recs = []
    for p in sorted((state.WORK / "probe").glob("shard_*.pt")):
        recs += torch.load(p, weights_only=False)
    return recs


def assemble(recs, device):
    N, D = len(recs), recs[0]["gfeat"].shape[1]
    GM, LQ, K = state.CFG["GMAX"], state.CFG["LQ"], recs[0]["masks"].shape[0]
    T = dict(gf=torch.zeros(N, GM, D, dtype=torch.float16), gpos=torch.zeros(N, GM, 2), gv=torch.zeros(N, GM, dtype=torch.bool),
             qe=torch.zeros(N, LQ, D, dtype=torch.float16), qv=torch.zeros(N, LQ, dtype=torch.bool),
             masks=torch.zeros(N, K, GM, dtype=torch.bool), js=torch.zeros(N, K), rid=recs[0]["rid"].clone())
    for i, r in enumerate(recs):
        G, lq = r["G"], r["qemb"].shape[0]
        T["gf"][i, :G] = r["gfeat"]; T["gpos"][i, :G] = r["gpos"]; T["gv"][i, :G] = True
        T["qe"][i, :lq] = r["qemb"]; T["qv"][i, :lq] = True; T["masks"][i, :, :G] = r["masks"]; T["js"][i] = r["js"]
    T["qv"][:, 0] = True
    return {k: v.to(device) for k, v in T.items()}


def spearman_rows(a, b):
    ra = a.argsort(-1).argsort(-1).float(); rb = b.argsort(-1).argsort(-1).float()
    ra = ra - ra.mean(-1, keepdim=True); rb = rb - rb.mean(-1, keepdim=True)
    return ((ra * rb).sum(-1) / (ra.norm(dim=-1) * rb.norm(dim=-1) + 1e-8)).mean().item()


def _pairs():
    ia, ib = [], []
    for b in range(len(state.RATIOS)):
        idx = list(range(b * state.MPR, (b + 1) * state.MPR))
        for x in range(len(idx)):
            for y in range(x + 1, len(idx)):
                ia.append(idx[x]); ib.append(idx[y])
    return torch.tensor(ia), torch.tensor(ib)


def train_selector(variant, device, recs, log_fn=print):
    use_q = (variant == "full")
    ck = state.WORK / "ckpt" / f"selector_{variant}.pt"
    D = recs[0]["gfeat"].shape[1]
    torch.manual_seed(state.CFG["SEED"])
    model = Selector(D, state.CFG["D_MODEL"], state.CFG["N_LAYERS"], state.CFG["N_HEADS"], use_q).to(device)
    if ck.exists():
        model.load_state_dict(torch.load(ck, map_location=device)["state"]); model.eval()
        log_fn(f"[train:{variant}] loaded {ck.name}"); return model
    uids = sorted({r["uid"] for r in recs})
    val_idx = [i for i, r in enumerate(recs) if r["split"] == "val"]
    tr_idx = [i for i, r in enumerate(recs) if r["split"] != "val"]
    T = assemble(recs, device)
    ia, ib = [x.to(device) for x in _pairs()]
    opt = torch.optim.AdamW(model.parameters(), lr=state.CFG["LR"], weight_decay=state.CFG["WD"])
    steps = state.CFG["EPOCHS"] * math.ceil(len(tr_idx) / state.CFG["BATCH"])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, state.CFG["LR"], total_steps=steps, pct_start=0.1)
    scaler = torch.cuda.amp.GradScaler(enabled=device.startswith("cuda"))
    LN2 = math.log(2)

    def batch_loss(idx):
        b = {k: v[idx] for k, v in T.items() if k != "rid"}
        with torch.autocast("cuda", dtype=torch.float16, enabled=device.startswith("cuda")):
            h, s = model.encode(b["gf"], b["gpos"], b["gv"], b["qe"], b["qv"])
            rl = model.risk(h, b["gv"], b["masks"])
        rl, s = rl.float(), s.float()
        tgt = (b["js"] / LN2).clamp(0, 1)
        l_risk = F.binary_cross_entropy_with_logits(rl, tgt)
        S = torch.einsum("bkg,bg->bk", b["masks"].float(), s.masked_fill(~b["gv"], 0))
        diff = S[:, ia] - S[:, ib]
        dj = b["js"][:, ib] - b["js"][:, ia]                         # >0: mask a is better (lower JS)
        valid = dj.abs() > state.CFG["RANK_MARGIN"]
        l_rank = (F.binary_cross_entropy_with_logits(diff, (dj > 0).float(), reduction="none") * valid).sum() / valid.sum().clamp_min(1)
        pair_acc = (((diff > 0) == (dj > 0)) & valid).sum() / valid.sum().clamp_min(1)
        return l_risk + state.CFG["RANK_W"] * l_rank, l_risk.detach(), l_rank.detach(), pair_acc.detach(), rl.detach(), b["js"]

    best, best_state = 1e9, None
    for ep in range(state.CFG["EPOCHS"]):
        model.train()
        perm = torch.tensor(tr_idx)[torch.randperm(len(tr_idx))]
        t_loss = 0.0
        for i in range(0, len(perm), state.CFG["BATCH"]):
            idx = perm[i:i + state.CFG["BATCH"]].to(device)
            loss = batch_loss(idx)[0]
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step()
            t_loss += loss.item() * len(idx)
        model.eval()
        with torch.no_grad():
            vs = [batch_loss(torch.tensor(val_idx[i:i + 64], device=device)) for i in range(0, len(val_idx), 64)]
        vl = float(np.mean([v[1].item() + state.CFG["RANK_W"] * v[2].item() for v in vs]))
        pa = float(np.mean([v[3].item() for v in vs]))
        sp = float(np.mean([spearman_rows(v[4], v[5]) for v in vs]))
        state.HIST.append(dict(variant=variant, epoch=ep, train_loss=t_loss / len(tr_idx), val_loss=vl, val_pair_acc=pa,
                         val_risk_spearman=sp, lr=sched.get_last_lr()[0]))
        if vl < best:
            best, best_state = vl, {k: x.detach().clone() for k, x in model.state_dict().items()}
        if ep % 2 == 0 or ep == state.CFG["EPOCHS"] - 1:
            log_fn(f"[train:{variant}] ep{ep:02d} train={t_loss/len(tr_idx):.4f} val={vl:.4f} pair_acc={pa:.3f} risk_spearman={sp:.3f}")
    model.load_state_dict(best_state); model.eval()
    torch.save(dict(state=best_state, variant=variant, val=best), ck)
    pd.DataFrame([h for h in state.HIST if h["variant"] == variant]).to_csv(state.WORK / "ckpt" / f"history_{variant}.csv", index=False)
    del T; gc.collect(); torch.cuda.empty_cache()
    return model


def run_train():
    """Stage 'train': train the question-guided selector (full and no-question variants)."""
    PROBE = load_probe_records()
    assert len(PROBE) > 0, "no probe shards found - run the 'probe' stage first"
    log(f"[train] {len(PROBE)} probed samples | split counts:",
        {s: sum(1 for r in PROBE if r['split'] == s) for s in ['train', 'val']})
    variants = ["full", "noq"]
    threads, res = [], {}
    def _tw(v, dev):
        torch.cuda.set_device(dev) if dev.startswith("cuda") else None
        res[v] = train_selector(v, dev, PROBE, log)
    for i, v in enumerate(variants):
        t = threading.Thread(target=_tw, args=(v, state.DEVICES[i % len(state.DEVICES)]), daemon=True); t.start(); threads.append(t)
    for t in threads:
        while t.is_alive(): t.join(0.5)
    state.SELECTORS.update(res)
    for v, m in state.SELECTORS.items():
        log(f"selector[{v}]: {sum(p.numel() for p in m.parameters())/1e6:.2f}M params")
    del PROBE; gc.collect()

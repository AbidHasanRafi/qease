"""Experiment B: question-swap control (does the selected set depend on the question?)."""
import math
import traceback

from PIL import Image
import torch
from qease import state
from qease.extra import xstate
from qease.evaluation import selector_order
from qease.selection import order_to_masks
from qease.utils import stable_hash
from qease.vlm.helpers import js_div
from qease.extra.utils import xd


def x_qswap_shard(r, shard, tick):
    sid, pairs = shard
    path = xd("qswap") / f"shard_{sid:04d}.pt"
    if path.exists(): tick(len(pairs)); return 0
    out = []
    fr = xstate.XCFG["QSWAP_FRACS"]
    for rec, partner in pairs:
        if state.STOP.is_set(): return 0
        try:
            img = Image.open(state.POOL_DIR / rec["img"]).convert("RGB")
            ctxA = r.prepare(rec)                                              # (image, own question)
            ctxB = r.prepare(img, partner["question"], rec["ds"])              # (same image, a DIFFERENT question)
            G = ctxA["G"]; assert ctxB["G"] == G
            lp_full = r.logp_full(ctxA)
            oA, oB = selector_order(r, ctxA, "full")[0], selector_order(r, ctxB, "full")[0]
            oN = selector_order(r, ctxA, "noq")[0]
            g1 = torch.Generator().manual_seed(stable_hash(rec["uid"]) % 10**6)
            oR1, oR2 = torch.randperm(G, generator=g1), torch.randperm(G, generator=g1)
            row = dict(uid=rec["uid"], ds=rec["ds"], js={}, flip={}, iou={})
            for f in fr:
                k = int(min(G, max(1, math.ceil(f * G))))
                M = {n: order_to_masks(o, G, [k])[0] for n, o in [("own", oA), ("swap", oB), ("noq", oN), ("rand", oR1), ("rand2", oR2)]}
                lpm = r.logp_for_token_masks(ctxA, r.group_to_token_mask(ctxA, torch.stack([M[n] for n in ("own", "swap", "noq", "rand")]).to(r.device)))
                jv = js_div(lp_full, lpm).cpu().numpy(); fl = (lpm.argmax(-1) != lp_full.argmax()).cpu().numpy()
                row["js"][f] = dict(zip(("own", "swap", "noq", "rand"), jv)); row["flip"][f] = dict(zip(("own", "swap", "noq", "rand"), fl))
                iou = lambda a, b: float((a & b).sum() / max(1, (a | b).sum()))
                row["iou"][f] = dict(own_vs_swap=iou(M["own"], M["swap"]), rand_vs_rand=iou(M["rand"], M["rand2"]), own_vs_noq=iou(M["own"], M["noq"]))
            out.append(row)
        except Exception as e:
            state.FAILS.append(("qswap", rec["uid"], repr(e)[:200], traceback.format_exc()))
        tick()
    torch.save(out, path); return len(out)

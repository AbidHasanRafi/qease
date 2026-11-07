"""Group-level counterfactual probing: the offline labels l = JS(p_full || p_masked)."""
import math
import time
import traceback

import torch
from qease import state
from qease.data import interleave_by_ds
from qease.parallel import parallel_map
from qease.selection import make_probe_masks
from qease.utils import log, stable_hash
from qease.vlm.helpers import js_div


def probe_one(r, rec):
    ctx = r.prepare(rec)
    G = ctx["G"]
    lp_full = r.logp_full(ctx)
    masks, rid = make_probe_masks(G, ctx["gh"], ctx["gw"], stable_hash(rec["uid"]) % 10**6)
    lpm = r.logp_for_token_masks(ctx, r.group_to_token_mask(ctx, masks.to(r.device)))
    js = js_div(lp_full, lpm).float().cpu()
    flip = (lpm.argmax(-1) != lp_full.argmax()).cpu()
    ent = float(-(lp_full.exp() * lp_full).sum())
    gi = torch.arange(G)
    gpos = torch.stack([(gi // ctx["gw"]).float() / max(1, ctx["gh"] - 1), (gi % ctx["gw"]).float() / max(1, ctx["gw"] - 1)], 1)
    return dict(uid=rec["uid"], ds=rec["ds"], split=rec["split"], G=G, gh=ctx["gh"], gw=ctx["gw"],
                gfeat=ctx["gfeat"].half().cpu(), gpos=gpos, qemb=ctx["qemb"][: state.CFG["LQ"]].half().cpu(),
                masks=masks, rid=rid, js=js, flip=flip, ent=ent)


def probe_shard(r, shard, tick):
    sid, recs = shard
    path = state.WORK / "probe" / f"shard_{sid:04d}.pt"
    if path.exists(): tick(len(recs)); return 0
    out = []
    for rec in recs:
        if state.STOP.is_set(): return 0
        try: out.append(probe_one(r, rec))
        except Exception as e:
            state.FAILS.append(("probe", rec["uid"], repr(e)[:200], traceback.format_exc()))
            if "out of memory" in str(e).lower(): torch.cuda.empty_cache()
        tick()
    torch.save(out, path)
    # light sidecar (no features) so label statistics can be analysed without re-loading GBs of shards
    torch.save([dict(uid=o["uid"], ds=o["ds"], split=o["split"], G=o["G"], gh=o["gh"], gw=o["gw"], js=o["js"].numpy(),
                     flip=o["flip"].numpy(), rid=o["rid"].numpy(), ent=o["ent"]) for o in out],
               state.WORK / "probe" / f"stats_{sid:04d}.pt")
    return len(out)


def run_probe():
    """Stage 'probe': group-level counterfactual probing of the frozen VLM (offline labels)."""
    train_recs = interleave_by_ds(state.BY_SPLIT["train"] + state.BY_SPLIT["val"], state.CFG["PROBE_TOTAL"], seed=2)
    S = state.CFG["SHARD_SIZE"]
    shards = [(i, train_recs[i * S:(i + 1) * S]) for i in range(math.ceil(len(train_recs) / S))]
    todo = [s for s in shards if not (state.WORK / "probe" / f"shard_{s[0]:04d}.pt").exists()]
    log(f"[probe] {len(train_recs)} images, {len(shards)} shards, {len(todo)} to do, K={state.K_MASKS} masks each")
    if todo:
        t0 = time.time()
        parallel_map(todo, probe_shard, "probe", total=sum(len(s[1]) for s in todo))
        log(f"[probe] done in {(time.time()-t0)/60:.1f} min")

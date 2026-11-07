"""Phase 0: headroom pilot (is there anything for a learned selector to gain?)."""
import numpy as np
import torch
from qease import state
from qease.data import interleave_by_ds
from qease.parallel import parallel_map
from qease.selection import baseline_orders, k_list, kept_token_frac, order_to_masks
from qease.utils import log, stable_hash
from qease.vlm.helpers import js_div


def phase0_one(r, rec, tick):
    ctx = r.prepare(rec)
    G = ctx["G"]
    lp_full = r.logp_full(ctx)
    eye = ~torch.eye(G, dtype=torch.bool)
    lp = r.logp_for_token_masks(ctx, r.group_to_token_mask(ctx, eye.to(r.device)))
    delta = js_div(lp_full, lp).cpu()                                  # necessity of each single group
    orders = baseline_orders(ctx, stable_hash(rec["uid"]) % 10**6)
    orders["oracle1"] = torch.argsort(-delta)
    ks = k_list(G)
    out = dict(uid=rec["uid"], ds=rec["ds"], G=G, kfrac={}, js={}, flip={})
    for name, od in orders.items():
        gm = order_to_masks(od, G, ks)
        lpm = r.logp_for_token_masks(ctx, r.group_to_token_mask(ctx, gm.to(r.device)))
        out["js"][name] = js_div(lp_full, lpm).cpu().numpy()
        out["flip"][name] = (lpm.argmax(-1) != lp_full.argmax()).cpu().numpy()
        out["kfrac"][name] = kept_token_frac(ctx, gm)
    tick(); return out


def run_phase0():
    """Stage 'phase0': headroom pilot (oracle group-removal ranking against cheap baselines)."""
    if state.P0_FILE.exists():
        P0 = torch.load(state.P0_FILE, weights_only=False)
    else:
        recs = interleave_by_ds(state.BY_SPLIT["val"], state.CFG["PHASE0_N"], seed=1)
        P0 = parallel_map(recs, phase0_one, "phase0", total=len(recs))
        torch.save(P0, state.P0_FILE)
    names = ["oracle1", "qsim", "uniform", "random"]
    log(f"PHASE 0 headroom on {len(P0)} samples  (mean JS, nats; lower is better)  - the oracle only ranks single-group ablations")
    hdr = "frac_kept   " + "  ".join(f"{n:>8s}" for n in names)
    print(hdr)
    for j, f in enumerate(state.CFG["K_GRID"]):
        print(f"{f:9.3f}   " + "  ".join(f"{np.mean([o['js'][n][j] for o in P0]):8.4f}" for n in names))
    lowj = [j for j, f in enumerate(state.CFG["K_GRID"]) if f <= 0.25]
    base_best = np.mean([min(np.mean([o["js"][n][j] for o in P0]) for n in ["qsim", "uniform", "random"]) for j in lowj])
    orc = np.mean([np.mean([o["js"]["oracle1"][j] for o in P0]) for j in lowj])
    log(f"GATE 1: mean JS at <=25% budget  oracle={orc:.4f} vs best baseline={base_best:.4f} "
        f"-> headroom {'EXISTS' if orc < 0.7*base_best else 'is SMALL (reconsider / see plan fallback)'}")

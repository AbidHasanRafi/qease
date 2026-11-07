"""Measured latency, FLOPs and memory of the pruned pipeline."""
import math
import time
import traceback

from PIL import Image
import numpy as np
import pandas as pd
import torch
from qease import state
from qease.data import interleave_by_ds
from qease.evaluation import selector_order
from qease.selection import baseline_orders, order_to_masks
from qease.utils import log
from qease.vlm.helpers import smart_resize


def _sync(dev):
    if dev.startswith("cuda"): torch.cuda.synchronize(dev)


def run_latency():
    """Stage 'latency': measured latency, FLOPs and memory of the pruned pipeline."""
    _r = state.RUNNERS[0]; _dev = _r.device
    _recs = interleave_by_ds(state.BY_SPLIT["test"], state.CFG["LAT_N"], seed=5)
    _rows = []
    with torch.inference_mode():
        _r.prepare(_recs[0])                                                              # warm-up (CUDA kernels, allocator)
        for _rec in _recs:
            try:
                nt = state.CFG["TOKENS_PER_DS"].get(_rec["ds"], state.CFG["TOKENS_PER_DS"]["default"])
                img = smart_resize(Image.open(state.POOL_DIR / _rec["img"]).convert("RGB"), nt)
                t0 = time.perf_counter(); _r.build_inputs(img, _rec["question"]); t_pre = 1e3 * (time.perf_counter() - t0)
                _sync(_dev); t0 = time.perf_counter(); ctx = _r.prepare(_rec); _sync(_dev)
                enc_ms = max(0.0, 1e3 * (time.perf_counter() - t0) - t_pre)               # vision tower + projector (+ hook capture)
                if "full" in state.SELECTORS:
                    _sync(_dev); t0 = time.perf_counter(); od = selector_order(_r, ctx, "full")[0]; _sync(_dev)
                    sel_ms = 1e3 * (time.perf_counter() - t0)
                else:
                    od, sel_ms = baseline_orders(ctx, 0)["uniform"], 0.0
                G = ctx["G"]
                for f in [1.0, 0.75, 0.5, 0.25, 0.125]:
                    k = max(1, math.ceil(f * G)); gk = order_to_masks(od, G, [k])[0].to(_dev)
                    for _ in range(2): _r.prefill_pruned(ctx, gk)                         # warm-up
                    if _dev.startswith("cuda"):
                        torch.cuda.reset_peak_memory_stats(_dev); base = torch.cuda.memory_allocated(_dev)
                    _sync(_dev); t0 = time.perf_counter()
                    for _ in range(5): _r.prefill_pruned(ctx, gk)
                    _sync(_dev); pf_ms = 1e3 * (time.perf_counter() - t0) / 5
                    mem = (torch.cuda.max_memory_allocated(_dev) - base) / 2**20 if _dev.startswith("cuda") else np.nan
                    ntok = int(_r.group_to_token_mask(ctx, gk[None])[0].sum())
                    _rows.append(dict(uid=_rec["uid"], ds=_rec["ds"], frac=f, groups=k, G=G, seq_tokens=ntok, vis_tokens_total=ctx["n_vis"],
                                      enc_ms=enc_ms, selector_ms=(sel_ms if f < 1.0 else 0.0), prefill_ms=pf_ms,
                                      prefill_gflops=2 * _r.llm_params * ntok / 1e9, peak_extra_mem_mib=mem))
            except Exception as e:
                state.FAILS.append(("latency", _rec["uid"], repr(e)[:200], traceback.format_exc()))
    pd.DataFrame(_rows).to_csv(state.TAB / "raw_latency.csv", index=False)
    log(f"[latency] {len(_rows)} measurements on {_dev} -> raw_latency.csv")

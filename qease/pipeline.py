"""Stage driver: runs the stages listed in CFG["STAGES"] in the order of the paper's pipeline.

pool -> load -> phase0 -> probe -> train -> eval -> crc -> gen -> latency -> analyze

Every stage is idempotent and writes to disk, so the pipeline can be re-run after an interruption and finished work is skipped.
"""
from qease import state
from qease.analysis import run_analyze
from qease.crc import load_lambdas, run_crc
from qease.data import build_pool, load_pool
from qease.evaluation import run_eval
from qease.generation import run_gen
from qease.latency import run_latency
from qease.loading import run_load
from qease.phase0 import run_phase0
from qease.probing import run_probe
from qease.selector import load_selectors, run_train
from qease.utils import check_environment, log

STAGE_ORDER = ["pool", "load", "phase0", "probe", "train", "eval", "crc", "gen", "latency", "analyze"]


def load_data_pool():
    """Read the cached pool from disk and index it by split."""
    state.POOL = load_pool()
    state.BY_SPLIT.clear()
    for r in state.POOL:
        state.BY_SPLIT[r["split"]].append(r)
    state.REC_BY_UID = {r["uid"]: r for r in state.POOL}
    log("pool:", {k: len(v) for k, v in state.BY_SPLIT.items()}, "| per dataset:",
        {d: sum(1 for r in state.POOL if r["ds"] == d) for d in state.CFG["N_POOL"]})


def run(stages=None, overrides=None, base_dir=None):
    """Run the pipeline.  `overrides` is a dict of CFG keys (e.g. {"MODEL_ID": "Qwen/Qwen2.5-VL-3B-Instruct"})."""
    ov = dict(overrides or {})
    if stages is not None:
        ov["STAGES"] = list(stages)
    check_environment()
    state.configure(ov, base_dir)
    S = state.CFG["STAGES"]
    if "pool" in S:
        build_pool()
    load_data_pool()
    if "load" in S or any(s in S for s in ["phase0", "probe", "eval", "gen", "latency"]):
        run_load()
    if "phase0" in S:
        run_phase0()
    if "probe" in S:
        run_probe()
    if "train" in S:
        run_train()
    load_selectors()                         # restore trained selectors from disk when 'train' is not part of this run
    if "eval" in S:
        run_eval()
    if "crc" in S:
        run_crc()
    else:
        load_lambdas()                       # restore calibrated thresholds from disk
    if "gen" in S:
        run_gen()
    if "latency" in S:
        run_latency()
    if "analyze" in S:
        run_analyze()
    log(f"finished. outputs in {state.WORK}/results ; failures logged: {len(state.FAILS)} (inspect state.FAILS[-1][3] for a traceback)")

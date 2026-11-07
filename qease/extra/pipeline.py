"""Driver of the extra experiments.

Pass 1 re-uses the first run's outputs (trained selectors, calibrated thresholds, evaluation records) for the base model.
Pass 2 (optional) repeats probing, training and calibration for a second model with a compact protocol and then runs the same experiments.
Every block is idempotent and writes to disk, so the driver can be re-run after an interruption.
"""
import zipfile
from pathlib import Path

from qease import state
from qease.extra import xstate
from qease.extra.preflight import x_preflight
from qease.extra.run import x_run_pass
from qease.extra.utils import xlog
from qease.pipeline import run


def export_results_zip(base_dir=None):
    """Zip every CSV / TeX the extra experiments and the analysis produced."""
    base = Path(base_dir) if base_dir else (Path("/kaggle/working") if Path("/kaggle/working").exists() else Path("."))
    zp = base / "qease_extra_results.zip"
    root = state.BASE
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        for f in (list(root.glob("*/extra/results/*")) + list(root.glob("*/extra/history_*.csv")) + list(root.glob("*/results/tables/*"))
                  + list(root.glob("*/results/figdata/*"))):
            z.write(f, f.relative_to(root.parent))
    xlog(f"DONE. results zip: {zp}  ({zp.stat().st_size / 1e6:.1f} MB)")
    return zp


def run_extra(overrides=None, base_dir=None):
    """Run the extra experiments.  `overrides` is a dict of qease.extra.config.DEFAULT_XCFG keys."""
    xstate.configure_extra(overrides)
    XC = xstate.XCFG
    x_preflight(XC["BASE_MODEL_ID"], base_dir)
    # ---- pass 1: the base model of the main run (loads the VLM, restores selectors and thresholds from disk)
    run(stages=["pool", "load"], overrides={"MODEL_ID": XC["BASE_MODEL_ID"]}, base_dir=base_dir)
    xlog(f"state: selectors={list(state.SELECTORS)}  thresholds={len(state.LAMBDAS)}  runners={len(state.RUNNERS)}  "
         f"FASTV_OK={state.FASTV_OK} VZ_OK={state.VZ_OK} SV_OK={state.SV_OK}")
    assert "full" in state.SELECTORS, "selector missing: expected WORK/ckpt/selector_full.pt from the first run"
    assert state.LAMBDAS, "no calibrated thresholds: WORK/results/tables/lambdas.json is empty or missing (needs the first run's 'crc' stage)"
    x_run_pass("base", XC["RUN_TB"], XC["RUN_QSWAP"], XC["RUN_BENCH"], XC["BENCH_N_PER_DS"], XC["RUN_STD"], XC["STD_N"])
    # ---- pass 2: a second model with a compact protocol (probe -> train -> eval -> calibrate, then the same experiments)
    if XC["RUN_MODEL2"]:
        M2 = XC["M2"]
        ov = dict(MODEL_ID=M2["MODEL_ID"], PROBE_TOTAL=M2["PROBE_TOTAL"], CAL_N=M2["CAL_N"], TEST_N=M2["TEST_N"], SPARSEVLM=M2["SPARSEVLM"])
        run(stages=["pool", "load", "probe", "train", "eval", "crc"], overrides=ov, base_dir=base_dir)
        x_run_pass("model2", False, XC["RUN_QSWAP"], XC["RUN_BENCH"], M2["BENCH_N_PER_DS"], XC["RUN_STD"], M2["STD_N"])
        if M2.get("ANALYZE"):                          # fidelity + calibration tables and figures for the second model
            run(stages=["analyze"], overrides=ov, base_dir=base_dir)
    return export_results_zip(base_dir)

"""Run-time state shared by all modules.

The original single-file pipeline kept these objects as notebook globals.  They are attributes of this module now and are
(re)initialised by `configure()`; every other module reads them as `state.NAME`.
"""
import copy
import random
import threading
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from qease.config import DEFAULT_CFG
from qease.utils import log

# ---- configuration and paths
CFG = {}
TAG = BASE = WORK = POOL_DIR = None
TAB = FIG = FDATA = None
P0_FILE = LAM_FILE = None
# ---- hardware
N_GPU = 0
DEVICES = ["cpu"]
DTYPE = None
# ---- bookkeeping
FAILS = []                  # (stage, item, repr(error), traceback)  -> inspect FAILS[-1] after a run
STOP = threading.Event()
MANIFEST = []               # (kind, relative path, description) of every exported artefact
# ---- data
POOL = []
BY_SPLIT = defaultdict(list)
REC_BY_UID = {}
# ---- models and trained objects
RUNNERS = []                # one VLMRunner per GPU
FASTV_OK = VZ_OK = SV_OK = False           # set True only if the baseline's self-test passes
EXTERNAL_SCORERS = {}                     # plug-in slot for other published methods
HIST = []
SELECTORS = {}
_SEL_CACHE = {}
LAMBDAS = {}
# ---- derived constants
RATIOS = MPR = K_MASKS = None
EPS0 = None
LOSS_DEFS = []
DS_ORDER = []
DS_COLOR = {}
# ---- analysis state
CAL = TEST = AV = FR = None
J25 = J12 = 0
DSL = []


def free_models():
    """Release the VLM replicas (needed before loading a second model in the same process)."""
    for r in RUNNERS:
        try:
            del r.model
        except Exception:
            pass
    RUNNERS.clear()
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def configure(overrides=None, base_dir=None):
    """Build the configuration, create the output folders and reset every piece of run-time state."""
    global CFG, TAG, BASE, WORK, POOL_DIR, TAB, FIG, FDATA, P0_FILE, LAM_FILE, N_GPU, DEVICES, DTYPE, FAILS, POOL, BY_SPLIT, REC_BY_UID
    global FASTV_OK, VZ_OK, SV_OK, HIST, SELECTORS, _SEL_CACHE, LAMBDAS, RATIOS, MPR, K_MASKS, EPS0, LOSS_DEFS, DS_ORDER, DS_COLOR
    global CAL, TEST, AV, FR, J25, J12, DSL, MANIFEST
    previous_model = CFG.get("MODEL_ID")
    CFG = copy.deepcopy(DEFAULT_CFG)
    CFG.update(overrides or {})
    if RUNNERS and previous_model != CFG["MODEL_ID"]:
        free_models()                                    # a different model: drop the old replicas

    TAG = CFG["RUN_TAG"] or CFG["MODEL_ID"].rstrip("/").split("/")[-1]
    BASE = Path(base_dir) if base_dir else (Path("/kaggle/working/qease") if Path("/kaggle/working").exists() else Path("./qease_out"))
    WORK = BASE / TAG                                   # per-model outputs;  the data pool is shared in BASE/pool
    POOL_DIR = BASE / "pool"
    (BASE / "pool" / "images").mkdir(parents=True, exist_ok=True)
    for sub in ["probe", "eval", "ckpt", "phase0", "gen", "results/tables", "results/figures", "results/figdata"]:
        (WORK / sub).mkdir(parents=True, exist_ok=True)


    random.seed(CFG["SEED"]); np.random.seed(CFG["SEED"]); torch.manual_seed(CFG["SEED"])
    N_GPU = torch.cuda.device_count()
    DEVICES = [f"cuda:{i}" for i in range(N_GPU)] or ["cpu"]
    DTYPE = dict(float16=torch.float16, float32=torch.float32, bfloat16=torch.bfloat16)[CFG["DTYPE"]]
    if DEVICES == ["cpu"]:
        DTYPE = torch.float32
    import transformers
    log(f"devices={DEVICES} dtype={DTYPE} transformers={transformers.__version__} torch={torch.__version__}")
    log(f"work dir: {WORK}")

    FAILS = []
    STOP.clear()
    POOL, BY_SPLIT, REC_BY_UID = [], defaultdict(list), {}
    FASTV_OK = VZ_OK = SV_OK = False
    HIST, SELECTORS, _SEL_CACHE, LAMBDAS = [], {}, {}, {}
    MANIFEST = []
    CAL = TEST = AV = FR = None
    J25 = J12 = 0
    DSL = []
    RATIOS, MPR = CFG["RATIOS"], CFG["MASKS_PER_RATIO"]
    K_MASKS = len(RATIOS) * MPR
    P0_FILE = WORK / "phase0" / "phase0.pt"
    TAB, FIG, FDATA = WORK / "results" / "tables", WORK / "results" / "figures", WORK / "results" / "figdata"
    EPS0 = CFG["EPS_LIST"][-1]                       # main JS threshold
    LOSS_DEFS = [("flip", None)] + [("js", e) for e in CFG["EPS_LIST"]]
    DS_ORDER = list(CFG["N_POOL"].keys())
    DS_COLOR = dict(zip(DS_ORDER, ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#7F7F7F"]))
    LAM_FILE = TAB / "lambdas.json"
    return CFG

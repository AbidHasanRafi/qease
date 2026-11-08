# %% ---------------------------------------------------------------- 1. CONFIG
CFG = dict(
    STAGES=["pool", "load", "phase0", "probe", "train", "eval", "crc", "aux", "ablate", "dec", "casc", "shift", "hires", "latency", "analyze"],
    PRESET="2b",
    SEED=0,
    RUN_TAG=None,              # output sub-folder; default = "final_" + model name (a fresh folder, so nothing from older runs is mixed in)
    POOL_NAME="pool_final",    # shared data-pool folder under /kaggle/working/qease  (set to "pool" to re-use the pool of the first run)
    # ---- model (any Qwen2-VL / Qwen2.5-VL checkpoint; fp16 because T4 has no bf16)
    MODEL_ID="Qwen/Qwen2-VL-2B-Instruct",
    DTYPE="float16",
    QUANT=None,                # None | "4bit" | "8bit"  (bitsandbytes; needed for 7B on a 15 GB T4)
    # visual-token budget per dataset (tokens after the 2x2 merge; 256 -> 448x448 image)
    TOKENS_PER_DS=dict(default=256, textvqa=400, docvqa=576),
    HIRES_TOKENS=dict(default=1024, textvqa=1024, docvqa=1600),   # high-resolution stage ("uncapped-ish": 4x the default caps)
    # ---- data pool: (image, question) pairs per source.  LARGER than the first run (test sets were too small: 143-431 per dataset)
    N_POOL=dict(vqav2=4000, gqa=3000, textvqa=3000, docvqa=2000, aokvqa=2500, vsr=1500, clevr=1500),
    VQAV2_SHARDS=3,            # parquet shards of VQAv2-train to read (~4k rows each)
    GQA_SPLIT="val", GQA_IMAGES=2200,
    CLEVR_TRY_ORIGINAL=False,
    # ---- grouping / probing
    GMAX=100, GROUP_WIN=2, LQ=32,
    RATIOS=[0.10, 0.20, 0.35, 0.50, 0.65, 0.80],
    MASKS_PER_RATIO=4,         # 2 iid-random + 2 spatially-coherent blobs  -> K = 24
    PROBE_TOTAL=6000, SHARD_SIZE=50, MASK_CHUNK=24,
    ANSWER_SUFFIX="\nAnswer the question using a single word or phrase.",
    # ---- budget grids.  K_GRID = first-token fidelity points (11, finer than before: the old grid jumped 0.75 -> 1.0)
    K_GRID=[1/16, 1/8, 3/16, 1/4, 5/16, 3/8, 1/2, 5/8, 3/4, 7/8, 1.0],
    DEC_GRID=[1/8, 1/4, 3/8, 1/2, 5/8, 3/4, 7/8, 1.0],      # DECODED budgets (each is one greedy generation per sample); 1.0 = full model
    PHASE0_N=120,
    # ---- selector training
    D_MODEL=256, N_LAYERS=3, N_HEADS=4, EPOCHS=25, BATCH=32, LR=3e-4, WD=0.01, RANK_MARGIN=0.002, RANK_W=0.5,
    # ---- eval / legacy (JS-based) calibration
    CAL_N=1500, TEST_N=2600,
    EPS_LIST=[0.01, 0.05], ALPHAS=[0.05, 0.10, 0.20], ALPHA_SWEEP=[0.02, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30],
    GEN_ADAPT=[("js", 0.05, 0.10), ("flip", None, 0.10)],
    GREEDY_BETA_GRID=[0.0, 0.15, 0.3, 0.6], TUNE_N=160,    # redundancy penalty of the greedy (MMR) set construction, tuned on VALIDATION images
    # ---- latency
    LAT_N=24, CASC_LAT_N=60,
    # ---- analysis
    BOOT_B=2000, CRC_RESAMPLES=100, CAL_SIZES=[25, 50, 100, 200, 400, 800], CAL_SIZE_REPS=50, N_QUAL=7, MONDRIAN_MIN_N=30,
    # ---- published baselines (self-tested at load time; a baseline whose test fails is disabled automatically)
    FASTV=True, FASTV_K=2, VISIONZIP=True, SPARSEVLM=True, SPARSE_RETAIN=[192, 128, 96, 64],
    DIVPRUNE=True, CDPRUNER=True, DART=True, DART_K=2, DART_PIVOTS=8, PYRAMIDDROP=True, CDP_THETA=2.0,
    PDROP_LAYER_FRACS=[0.07, 0.25, 0.5],
    # ---- NOVELTY: escalate-or-abstain cascade on DECODED answers
    GEN_MAX_NEW=12,
    CORRECT_TAU=0.5,           # a decoded answer counts as correct when its official score (VQA-acc / ANLS / EM) >= tau
    DEC_N=dict(head=1200, cal=1500, test=2200),     # decoded samples per role (head = trained on the VALIDATION split, then cal / test splits)
    DEC_BASE_N=700, DEC_BASE_FRACS=[0.125, 0.25, 0.5], DEC_BASE_METHODS=["qsim", "uniform", "fastv", "visionzip", "divprune", "cdpruner", "dart", "pdrop"],
    CASC_PATHS={"3-stage": [0.125, 0.375, 1.0], "4-stage": [0.125, 0.25, 0.5, 1.0], "5-stage": [0.125, 0.25, 0.5, 0.75, 1.0]},
    CASC_ALPHAS=[0.02, 0.05, 0.10], CASC_DELTA=0.10,     # certificate: P_cal( risk <= alpha ) >= 1 - delta
    CASC_LAMBDAS=40, CASC_MUS=[0.95, 0.9, 0.85, 0.8, 0.7, 0.6, 0.5],   # candidate thresholds (lambda: harm score; mu: futility score)
    CASC_RESPLITS=200, CASC_BINS=3, CASC_FEWSHOT=[50, 100, 200],
    HEAD_ITERS=120,
    # ---- aux labels, ablations, shift, high-res
    AUX_N=2000, AUX_MAXTOK=4, LOGO_N=900, ATTN_N=2000, ABL_EPOCHS=15, ABL_TEST_N=600, ABL_FRACS=[0.125, 0.25, 0.5],
    ABL_OMEGA=3, ABL_OMEGA_N=1400, ABL_OMEGA_TEST=400,
    SHIFT_EPOCHS=15, SHIFT_TEST_PER_DS=250,
    HIRES_N=350,
)
_PRESETS = {
    "2b": dict(),
    "3b": dict(MODEL_ID="Qwen/Qwen2.5-VL-3B-Instruct", PROBE_TOTAL=3500, CAL_N=1000, TEST_N=1500, SPARSEVLM=False, PYRAMIDDROP=False,
               DEC_N=dict(head=800, cal=1000, test=1400), DEC_BASE_N=400, AUX_N=1200, LOGO_N=500, ATTN_N=1200, ABL_TEST_N=400, HIRES_N=200,
               STAGES=["pool", "load", "probe", "train", "eval", "crc", "dec", "casc", "hires", "latency", "analyze"]),
    "7b": dict(MODEL_ID="Qwen/Qwen2.5-VL-7B-Instruct", QUANT="4bit", PROBE_TOTAL=2000, CAL_N=600, TEST_N=900, SPARSEVLM=False, PYRAMIDDROP=False,
               DEC_N=dict(head=500, cal=600, test=800), DEC_BASE_N=250, AUX_N=800, LOGO_N=300, ATTN_N=800, ABL_TEST_N=250, HIRES_N=120, MASK_CHUNK=8,
               STAGES=["pool", "load", "probe", "train", "eval", "dec", "casc", "latency", "analyze"]),
}
_ov = dict(globals().get("CFG_OVERRIDES", {}))
_ov.update(json.loads(os.environ.get("QEASE_OVERRIDES", "{}") or "{}"))       # optional: QEASE_OVERRIDES='{"PRESET": "3b"}' python main.py   (nothing is set by default)
CFG.update(_PRESETS.get(_ov.get("PRESET", CFG["PRESET"]), {}))
CFG.update(_ov)

TAG = CFG["RUN_TAG"] or ("final_" + CFG["MODEL_ID"].rstrip("/").split("/")[-1])
BASE = Path("/kaggle/working/qease") if Path("/kaggle/working").exists() else Path("./qease_out")
WORK = BASE / TAG                                   # per-model outputs;  the data pool is shared in BASE/<POOL_NAME>
PROBE_DIR, EVAL_DIR, CKPT_DIR = WORK / "probe", WORK / "eval", WORK / "ckpt"          # re-pointed temporarily by ablations / hires
AUX_DIR, LOGO_DIR, ATTN_DIR, DEC_DIR = WORK / "aux", WORK / "logo", WORK / "attn", WORK / "dec"
ABL_DIR, SHIFT_DIR, HIRES_DIR = WORK / "ablate", WORK / "shift", WORK / "hires"
(BASE / CFG["POOL_NAME"] / "images").mkdir(parents=True, exist_ok=True)
for _d in [PROBE_DIR, EVAL_DIR, CKPT_DIR, AUX_DIR, LOGO_DIR, ATTN_DIR, DEC_DIR, ABL_DIR, SHIFT_DIR, HIRES_DIR, WORK / "phase0",
           WORK / "results" / "tables", WORK / "results" / "figures", WORK / "results" / "figdata"]:
    _d.mkdir(parents=True, exist_ok=True)

random.seed(CFG["SEED"]); np.random.seed(CFG["SEED"]); torch.manual_seed(CFG["SEED"])
N_GPU = torch.cuda.device_count()
DEVICES = [f"cuda:{i}" for i in range(N_GPU)] or ["cpu"]
DTYPE = dict(float16=torch.float16, float32=torch.float32, bfloat16=torch.bfloat16)[CFG["DTYPE"]]
if DEVICES == ["cpu"]:
    DTYPE = torch.float32
log(f"devices={DEVICES} dtype={DTYPE} transformers={transformers.__version__} torch={torch.__version__} | preset={_ov.get('PRESET', CFG['PRESET'])}")
log(f"work dir: {WORK}")

FAILS = []            # (stage, item, repr(error), traceback)  -> inspect FAILS[-1] after a run
STOP = threading.Event()
GREEDY_BETA = 0.0     # set by the tuning step (validation images) -> used for the 'qease_greedy' method

def stable_hash(s):
    return int(hashlib.md5(str(s).encode()).hexdigest()[:8], 16)

def split_of(img_key):
    """Deterministic split by IMAGE so no image leaks across train/val/cal/test."""
    h = stable_hash(img_key) % 100
    return "train" if h < 62 else "val" if h < 70 else "cal" if h < 85 else "test"


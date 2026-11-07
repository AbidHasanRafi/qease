"""Default configuration of the pipeline (every key can be overridden: `python -m qease --set KEY=VALUE`).

STAGES lists the stages to run; finished work is skipped, so a stage can be re-run after a crash."""

DEFAULT_CFG = dict(
    STAGES=["pool", "load", "phase0", "probe", "train", "eval", "crc", "gen", "latency", "analyze"],
    SEED=0,
    RUN_TAG=None,              # output sub-folder name; default = model name (so a 2nd model never overwrites the 1st)
    # ---- model (any Qwen2-VL / Qwen2.5-VL checkpoint; fp16 because T4 has no bf16)
    MODEL_ID="Qwen/Qwen2-VL-2B-Instruct",
    DTYPE="float16",
    # visual-token budget per dataset (tokens after the 2x2 merge; 256 -> 448x448 image)
    TOKENS_PER_DS=dict(default=256, textvqa=400, docvqa=576),
    # ---- data pool: how many (image, question) pairs to take from each source
    N_POOL=dict(vqav2=2500, gqa=2000, textvqa=2000, docvqa=1500, aokvqa=1500, vsr=1000, clevr=1000),
    VQAV2_SHARDS=2,            # parquet shards of VQAv2-train to read (~4k rows each)
    GQA_SPLIT="val", GQA_IMAGES=1500,
    CLEVR_TRY_ORIGINAL=False,  # HuggingFaceM4/clevr needs a loading script (+18GB zip); fallback is used
    # ---- grouping / probing
    GMAX=100,                  # max #groups per image (window grows 2->3->4->6 to satisfy it)
    GROUP_WIN=2,
    LQ=32,                     # question tokens kept for the selector
    RATIOS=[0.10, 0.20, 0.35, 0.50, 0.65, 0.80],   # keep-ratios probed per sample
    MASKS_PER_RATIO=4,         # 2 iid-random + 2 spatially-coherent blobs  -> K = 24
    PROBE_TOTAL=6000,          # max training images probed (spread across datasets)
    SHARD_SIZE=50,
    MASK_CHUNK=24,             # masks per LLM batch (auto-halves on OOM)
    ANSWER_SUFFIX="\nAnswer the question using a single word or phrase.",
    # ---- budget grid (fractions of groups kept) used by eval / calibration
    K_GRID=[1/16, 1/8, 3/16, 1/4, 3/8, 1/2, 3/4, 1.0],
    # ---- phase 0 (oracle headroom)
    PHASE0_N=120,
    # ---- selector training
    D_MODEL=256, N_LAYERS=3, N_HEADS=4, EPOCHS=25, BATCH=32, LR=3e-4, WD=0.01,
    RANK_MARGIN=0.002, RANK_W=0.5,
    # ---- eval / calibration
    CAL_N=1500, TEST_N=2000,
    EPS_LIST=[0.01, 0.05],               # JS threshold (nats) defining "answer distribution changed"
    ALPHAS=[0.05, 0.10, 0.20],           # tolerated violation rate
    ALPHA_SWEEP=[0.02, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30],
    # ---- generation-based accuracy
    GEN_N_PER_DS=60, GEN_FRACS=[0.25, 0.50], GEN_MAX_NEW=10,
    GEN_ADAPT=[("js", 0.05, 0.10), ("flip", None, 0.10)],   # (loss, eps, alpha) operating points run with generation
    # ---- latency
    LAT_N=24,
    # ---- analysis
    BOOT_B=2000,               # bootstrap resamples for every CI / p-value
    CRC_RESAMPLES=100,         # random cal/test re-splits used to verify the CRC guarantee empirically
    CAL_SIZES=[25, 50, 100, 200, 400, 800],
    CAL_SIZE_REPS=50,
    N_QUAL=7,                  # qualitative examples (one per dataset, best wins) + 3 failure cases
    MONDRIAN_MIN_N=30,
    # ---- published baseline re-implemented inside this script (self-tested at load time; disabled automatically if a test fails)
    FASTV=True, FASTV_K=2,     # FastV (Chen et al., ECCV'24): prune at layer K using the last token's head-averaged attention at layer K-1
    VISIONZIP=True,            # VisionZip (Yang et al., CVPR'25): dominant tokens by last-vision-block attention + merged contextual tokens
    SPARSEVLM=True,            # SparseVLM v1.0 (Zhang et al., ICML'25): text-rater pruning at 3 layers + token recycling (official budgets)
    SPARSE_RETAIN=[192, 128, 96, 64],       # the four budgets of the official release (average #visual tokens/layer for 576 tokens)
    GEN_SPARSE_RETAIN=[192, 96],            # budgets also run with generation (each adds 5 generations per sample)
)

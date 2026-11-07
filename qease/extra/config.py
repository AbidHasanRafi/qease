"""Default configuration of the extra experiments (trained baselines, question swap, accuracy, standard benchmarks, second model)."""

DEFAULT_XCFG = dict(
    QEASE_SRC=None,                                   # folder with the first run's outputs to copy from (None: search /kaggle/input)
    BASE_MODEL_ID="Qwen/Qwen2-VL-2B-Instruct",
    RUN_TB=True, RUN_QSWAP=True, RUN_BENCH=True, RUN_MODEL2=True,
    # --- A  trained baselines
    TB_NAMES=["attn", "mlp", "shuf"],
    TB_EPOCHS=25,
    # --- B  question swap
    QSWAP_N=300, QSWAP_FRACS=[0.125, 0.25, 0.5],
    # --- C  downstream accuracy
    BENCH_N_PER_DS=150,                               # held-out test questions per dataset (7 datasets)
    BENCH_FRACS=[0.25, 0.50], BENCH_MAX_NEW=12,
    BENCH_ADAPT=[("flip", None, 0.10), ("js", 0.05, 0.10)],   # (loss, eps, alpha) calibrated operating points
    BENCH_SPARSE=[192],                               # SparseVLM official budgets run with generation (each adds ~5 generations)
    BENCH_USE_TB=True,                                # include the trained baselines of A in the accuracy table
    # --- E  standard benchmarks: OFFICIAL validation/test sets that are NOT part of the training pool (no image overlap)
    RUN_STD=True, STD_NAMES=["pope", "textvqa", "vqav2", "aokvqa"], STD_N=150,   # STD_N questions per benchmark
    # --- D  second model (compact protocol)
    M2=dict(MODEL_ID="Qwen/Qwen2.5-VL-3B-Instruct", PROBE_TOTAL=3000, CAL_N=600, TEST_N=800, SPARSEVLM=False,
            BENCH_N_PER_DS=100, STD_N=100, ANALYZE=True),
    BOOT_B=2000, SEED=0,
)

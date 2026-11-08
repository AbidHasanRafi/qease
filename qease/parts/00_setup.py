# =============================================================================
# Q-EASE (final)  -  Self-aware token pruning for lightweight VLMs: ESCALATE OR ABSTAIN, with a certified harm bound
# One-file Kaggle pipeline (2 x T4).  NO main(): paste into ONE code cell and run.
#
# WHAT IS NEW COMPARED WITH qease_kaggle.py / qease_extra_kaggle.py
#   NOVELTY  "Escalate or abstain"
#       The pruned VLM is run as a CASCADE over increasing visual-token budgets (e.g. 12.5% -> 37.5% -> full).  After every stage two
#       light heads, fed with the pruned model's own first-token statistics and the selector's set-risk output, estimate
#         * harm     P( full model correct  AND  pruned model wrong )          -> escalate to the next budget when it is too high
#         * futility P( full model wrong )                                     -> ABSTAIN when more tokens would not help anyway
#       The thresholds (lambda, mu) are calibrated with Learn-then-Test (fixed-sequence, exact binomial p-values), so that with
#       probability >= 1-delta over the calibration draw
#         E[ 1{full model correct AND (we abstain OR our final answer is wrong)} ]  <=  alpha            ("lost-correct rate")
#       i.e. a certified bound, on DECODED, ground-truth-aware answers (not first-token agreement), that a prune-and-escalate system
#       loses at most alpha of the questions the full model would have answered correctly.
#   SUPERVISOR FIXES (all implemented)
#       * task-aligned risk P(correct->wrong) on full decoded answers (official VQA-acc / ANLS / EM) + accuracy-aware / answer-distilled labels
#       * label ablation: JS vs gold-likelihood vs final-answer distillation vs FastV-attention vs single-group (leave-one-group-out) vs shuffled,
#         same architecture;  K (#masks), mask types, loss weights (beta), rank loss, MLP (no cross-group attention), no-question, window omega
#       * more baselines: DivPrune, CDPruner-style (conditional DPP), DART-style, PyramidDrop (+ FastV, VisionZip, SparseVLM as before)
#       * finer budget grid (11 first-token points, 7 decoded budgets) so fixed-budget baselines are not strawmen; adaptive latency end-to-end
#       * distributions over 200 random cal/test re-splits (not single splits); LTT vs expectation-only CRC; Mondrian over DIFFICULTY bins;
#         conditional-coverage diagnostics (dataset, difficulty, question length)
#       * distribution shift: leave-one-dataset-out calibration (+ few-shot recalibration) and leave-one-dataset-out SELECTOR training
#       * additivity: greedy redundancy-aware set construction (MMR) + analysis of the additive-score vs set-level-head gap + SRH calibration on top-k sets
#       * high-resolution (uncapped-ish) evaluation, larger test sets, presets for Qwen2.5-VL-3B and (4-bit) 7B
#
# STAGES   (switch with CFG["STAGES"]; every stage checkpoints to disk and is idempotent -> re-run the cell to resume)
#   pool      build a data pool from 7 HF datasets            load     load one model replica per GPU + self-tests of every baseline
#   phase0    oracle headroom gate                            probe    counterfactual probing (mask, JS) labels
#   train     main selector + no-question selector            eval     first-token fidelity of every method on cal/test (+ greedy-beta tuning)
#   crc       legacy JS-based CRC thresholds                  aux      extra labels: gold-likelihood, answer-distillation, FastV attention, leave-one-group-out
#   ablate    label / architecture / loss / mask / window ablations (trained selectors evaluated on one test subset)
#   dec       DECODED records: full answer + answer at every budget (+ baselines) + stage statistics for head-train / cal / test
#   casc      train harm + futility heads, LTT calibration of the cascade, saves the policy
#   shift     leave-one-dataset-out selectors                 hires    high-resolution evaluation with the same selectors/policy
#   latency   real prefill latency + REAL cascade wall-clock  analyze  every table / figure / CSV for the paper
#
# OUTPUT TREE (WORK/results): tables/*.csv|tex  figures/*.pdf|png  figdata/*.csv  README_results.md  MANIFEST.csv  run_config.json (+ zip)
# Needs: Kaggle "Accelerator: GPU T4 x2" and "Internet: On".   Rough budget on 2 x T4: 8-10 h for the full list (stages are resumable).
#
# PRESETS (set before running, e.g. CFG_OVERRIDES = dict(PRESET="3b")):  "2b" (default) | "3b" (Qwen2.5-VL-3B) | "7b" (Qwen2.5-VL-7B, 4-bit)
# =============================================================================

# %% ----------------------------------------------------------------- 0. SETUP
import os, sys, io, re, gc, json, math, time, random, hashlib, threading, queue, string, pickle
import traceback, subprocess, warnings, ast
from pathlib import Path
from collections import defaultdict

warnings.filterwarnings("ignore")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "0")

def _pip(*pkgs):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *pkgs])

try:
    import transformers
    from packaging import version as _v
    if _v.parse(transformers.__version__) < _v.parse("4.45.0"):
        _pip("-U", "transformers>=4.45.0")
        raise ImportError("restart kernel after upgrading transformers")
except ImportError as _e:
    raise SystemExit(f"{_e}  -> restart the kernel and run the cell again.")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from tqdm.auto import tqdm
import pandas as pd
import matplotlib
matplotlib.use("Agg")                      # figures are written to disk; PNGs are displayed inline afterwards
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy import stats as sps             # exact binomial tails for Learn-then-Test
from sklearn.ensemble import HistGradientBoostingClassifier

def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)

# Optional HF token from Kaggle secrets (name the secret HF_TOKEN) -> faster downloads
try:
    from kaggle_secrets import UserSecretsClient
    os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
    log("HF token loaded from Kaggle secrets")
except Exception:
    pass


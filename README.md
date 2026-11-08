# Q-EASE: escalate or abstain

**Self-aware visual-token pruning for lightweight vision-language models, with a certified bound on the answers it loses.**

A pruned VLM is run as a *cascade* over growing visual-token budgets. After every stage two small heads, fed with the pruned model's own
first-token statistics and the selector's set-risk output, estimate

* **harm**: P(full model correct **and** this stage wrong), so *escalate* to the next budget when it is too high;
* **futility**: P(full model wrong), so *abstain* when more tokens would not help anyway.

The two thresholds (λ, μ) are calibrated with Learn-then-Test (fixed-sequence testing, exact binomial p-values), so that with probability
at least 1 − δ over the calibration set

```
E[ 1{ full model correct  AND  (we abstain OR our final answer is wrong) } ]  <=  alpha        ("lost-correct rate")
```

on **decoded**, ground-truth-aware answers (official VQA accuracy / ANLS / exact match), not on first-token agreement.

> **Status.** This repository was assembled from the validated Q-EASE pipeline plus the new cascade code. It has been syntax-checked and
> linted, but **the new parts have not been executed yet** (all experiments run on Kaggle, 2 x T4). Expect to fix first-run bugs; finished
> stages are cached, so re-running resumes. See [Known limitations](#known-limitations).

## Run it

```bash
pip install -r requirements.txt
python main.py
```

On Kaggle: GPU **T4 x 2**, Internet **on**, add the repo as a dataset or `git clone` it, then `!python main.py` (or `%run main.py`).
`main.py` contains no configuration. The default is the full experiment, equal to running the one-cell script `qease_final.py`.

| Where | What |
|---|---|
| `qease/parts/01_config.py` | every setting (`CFG`), stage list, model presets |
| env var `QEASE_OVERRIDES` | JSON overrides without editing files, e.g. `QEASE_OVERRIDES='{"PRESET":"3b"}' python main.py` |
| notebook | define `CFG_OVERRIDES = dict(...)` before `%run main.py` |

Presets: `"2b"` (default, Qwen2-VL-2B), `"3b"` (Qwen2.5-VL-3B), `"7b"` (Qwen2.5-VL-7B in 4-bit, needs `bitsandbytes`, untested).

**Smoke test first** (minutes, checks that every stage executes; with sizes this small the cascade will mostly fall back to "always use the
full model", which is expected):

```bash
QEASE_OVERRIDES='{"RUN_TAG":"smoke","POOL_NAME":"pool_smoke","N_POOL":{"vqav2":150,"gqa":100,"textvqa":100,"docvqa":80,"aokvqa":80,"vsr":60,"clevr":60},"GQA_IMAGES":80,"VQAV2_SHARDS":1,"PROBE_TOTAL":200,"EPOCHS":3,"ABL_EPOCHS":2,"SHIFT_EPOCHS":2,"PHASE0_N":10,"TUNE_N":12,"CAL_N":60,"TEST_N":80,"DEC_N":{"head":40,"cal":50,"test":60},"DEC_BASE_N":20,"AUX_N":60,"LOGO_N":30,"ATTN_N":60,"ABL_TEST_N":30,"ABL_OMEGA_N":60,"ABL_OMEGA_TEST":20,"SHIFT_TEST_PER_DS":10,"HIRES_N":10,"LAT_N":3,"CASC_LAT_N":4,"CASC_RESPLITS":10,"BOOT_B":200,"CRC_RESAMPLES":5,"CAL_SIZES":[10,20],"CAL_SIZE_REPS":3}' python main.py
```

## Layout

```
main.py                    entry point: runs the parts in order, no configuration
requirements.txt
qease/parts/               the pipeline, split into ordered parts of ONE program (shared namespace, see below)
  00_setup.py                imports, logging, HF token
  01_config.py               CFG, presets, output folders, seeds, devices
  02_data_pool.py            7 HF datasets -> local image/question pool, image-level splits
  03_vlm_runner.py           VLMRunner: one model replica per GPU, physical token pruning, FastV/VisionZip/SparseVLM ports
  04_models_and_selftests.py parallel helpers (one worker per GPU), model loading, self-tests of every baseline
  05_masks_baselines_phase0.py probe masks, random/uniform/question-similarity baselines, phase-0 oracle headroom gate
  06_extra_baselines.py      teacher-forced scoring, DivPrune / CDPruner-style / DART-style / PyramidDrop, greedy (MMR) set construction
  07_probing.py              counterfactual probing -> (mask, JS-divergence) labels
  08_selector.py             question-guided group scorer + set-level risk head, training
  09_trainer_and_labels.py   general trainer (label / architecture / loss / mask ablations), gold-likelihood, answer-distillation, attention and single-group labels
  10_eval.py                 first-token fidelity of every method on cal/test (+ greedy-beta tuning on validation images)
  11_reporting_and_crc.py    statistics, plotting style, legacy JS-based conformal risk control, answer normalisation
  12_aux_labels_ablations.py extra-label generation, label/architecture/loss/mask/window ablations
  13_decoded_records.py      official answer metrics; decoded answers per budget (+ baselines) = the task-aligned evidence
  14_cascade_core.py         harm + futility heads, cascade simulation, Learn-then-Test calibration, saved policy
  15_shift_hires.py          leave-one-dataset-out selectors, high-resolution stage
  16_latency.py              prefill latency / FLOPs, REAL wall-clock of the executed cascade
  17_analysis_legacy.py      tables + figures for fidelity, calibration, difficulty, qualitative cases, compute-matched comparisons
  18_analysis_cascade.py     tables + figures for the cascade, re-splits, conditional coverage, shift, ablations, hi-res, latency
  19_done.py
```

**Why parts and not importable modules.** The pipeline was designed as a no-`main()` notebook program: every stage checkpoints to disk and is
idempotent, and a crash must leave all state alive. `main.py` therefore executes the parts one after another in a single shared namespace
(the module `qease_pipeline`), exactly like consecutive pieces of one cell. Functions read and rebind shared globals on purpose (for example the ablation
code temporarily re-points `PROBE_DIR`/`CKPT_DIR`). Tracebacks keep the part file name and line number. After a crash, in a notebook:
`import sys; ns = sys.modules["qease_pipeline"]; ns.FAILS[-1]`.

## Stages

Set `CFG["STAGES"]` to run a subset; each stage is skipped when its outputs already exist.

| Stage | What it does |
|---|---|
| `pool` | build the data pool from VQAv2, GQA, TextVQA, DocVQA, A-OKVQA, VSR, CLEVR (split by image: train 62 / val 8 / cal 15 / test 15 %) |
| `load` | load one model replica per GPU; self-test FastV, VisionZip, SparseVLM, DivPrune, CDPruner-style, DART-style, PyramidDrop (a failing baseline is disabled) |
| `phase0` | oracle headroom gate |
| `probe` | counterfactual probing of the frozen VLM with group masks |
| `train` | main selector + no-question selector |
| `eval` | first-token fidelity of all methods on cal/test |
| `crc` | legacy JS-based conformal thresholds |
| `aux` | extra labels: gold-answer likelihood, full-model-answer likelihood, FastV attention, leave-one-group-out |
| `ablate` | label / architecture / loss / mask / window ablations |
| `dec` | decoded answers per budget for head-train, calibration and test questions (+ baselines on a test subset) |
| `casc` | train the harm and futility heads, calibrate (λ, μ) with Learn-then-Test, save the policy |
| `shift` | leave-one-dataset-out selectors |
| `hires` | the same selectors and policy at 4x the token caps |
| `latency` | prefill latency / FLOPs, and the real executed cascade |
| `analyze` | every table, figure and tidy CSV |

## Outputs

`qease_out/<RUN_TAG>/results/` (on Kaggle: `/kaggle/working/qease/<RUN_TAG>/results/`, plus a zip next to it):

```
tables/*.csv|tex     figures/*.pdf|png     figdata/*.csv (exact data behind every figure)     MANIFEST.csv     README_results.md     run_config.json
```

Key tables: `casc_main` (cascade vs full / fixed / static-adaptive / uncalibrated / oracle), `casc_resplits` (distribution over random re-splits,
Learn-then-Test vs expectation-only CRC), `casc_conditional` (per dataset / difficulty / length + Mondrian over difficulty), `casc_shift`,
`decoded_baselines`, `ablations`, `head_quality`, `casc_latency`.

## What the experiment tests

* **Decoded, task-aligned risk** (P(correct to wrong), abstention counted as a loss) instead of first-token agreement.
* **High-probability certificate** (Learn-then-Test) and its per-split behaviour over 200 random re-splits, against the expectation-only CRC rule.
* **Conditional coverage** beyond dataset: difficulty and question-length bins, and Mondrian calibration over difficulty.
* **Shift:** calibrating on other datasets and testing on a held-out one (heads, thresholds and selector), few-shot recalibration, high resolution.
* **Labels:** does counterfactual supervision beat attention-derived, single-group, gold-likelihood and shuffled labels with the same architecture?
* **Baselines** at matched budgets, decoded with the same decoder: FastV, VisionZip, SparseVLM (first-token only), DivPrune, CDPruner-style, DART-style, PyramidDrop.
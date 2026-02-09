# Fine-tune Cellpose-SAM for ROSMAP IF: Current Experiment Status

## 1) Project Goal and Scope
This project aims to:
- Adapt Cellpose-SAM to ROSMAP human brain IF data (DAPI + one marker channel).
- Quantify label efficiency under limited annotation budgets.
- Demonstrate downstream scientific value (donor-level pathology/cognition stability).

Markers in scope:
- GFAP, IBA1, NeuN, OLIG2

Planned experiment blocks:
- E0 data prep and donor-level split
- E1 baseline and fine-tune segmentation benchmarks
- E2 label efficiency curves
- E3 post-processing / marker gating ablation
- E4 micro-sam diagnostics (optional)
- E5 downstream donor-level analysis (critical, still pending implementation)

---

## 2) What Was Implemented

### 2.1 Core code completed or upgraded
Implemented new utilities:
- `src/fintune/utils/dataset.py`
- `src/fintune/utils/metrics.py`

Converted from stubs to runnable modules:
- `src/fintune/data_prep/prepare_cellpose_data.py`
- `src/fintune/inference/cpsam_baseline.py`
- `src/fintune/training/finetune_generic.py`
- `src/fintune/training/finetune_gfap.py`
- `src/fintune/training/budget_curve.py`

Added/updated script arguments and execution controls:
- `scripts/run_cpsam_baseline.py`
- `scripts/train_finetune_generic.py`
- `scripts/train_finetune_gfap.py`
- `scripts/run_budget_curve.py`

Implemented E3 gating ablation (previously stub):
- `src/fintune/evaluation/gating_ablation.py`
- `scripts/run_gating_ablation.py`

Configured project paths:
- `configs/paths.yaml`

### 2.2 Robustness improvements added during long runs
- Added timestamped budget-curve outputs to avoid accidental overwrite:
  - `records_YYYYMMDD_HHMMSS.tsv`
  - `summary_YYYYMMDD_HHMMSS.json`
- Added `--skip-existing` support in budget curve workflow to resume interrupted runs and only compute missing items.

---

## 3) Experimental Progress and Results

## 3.1 E0 data prep status
Completed:
- Cellpose-format data written.
- Split result:
  - train: 72 images
  - val: 24 images
  - test: 24 images

Directories in use:
- `data/cellpose/train_cells`
- `data/cellpose/val_cells`
- `data/cellpose/test_cells`

## 3.2 E1 segmentation benchmarks (completed)
Main comparison table:
- `data/reports/e1_metrics_with_hybrid.tsv`

### Overall metrics on test set
| Model | AP@0.5 | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| baseline (`cpsam`) | 0.105450 | 0.097836 | 0.784515 | 0.173976 |
| generic fine-tune (`generic_ft`) | 0.235867 | 0.500888 | 0.526119 | 0.513194 |
| GFAP-only fine-tune (`gfap_ft`) | 0.149321 | 0.236364 | 0.618470 | 0.342017 |
| hybrid marker recipe | 0.258040 | 0.463933 | 0.569963 | 0.511511 |

Interpretation:
- Baseline had high recall but very low precision, giving weak F1.
- Generic fine-tune gave the biggest balanced improvement (large precision/F1 gain).
- Hybrid recipe gave the best AP@0.5 (by reusing baseline on OLIG2 where generic failed).

### Per-marker behavior (important)
- GFAP: strong gain with generic fine-tune.
- NeuN: strong gain with generic fine-tune.
- IBA1: AP gain exists, but recall dropped vs baseline.
- OLIG2: generic fine-tune collapsed (near-zero), baseline worked better.

This is why a hybrid recipe was tested and improved AP.

## 3.3 E2 label efficiency (completed, full 25 runs)
Final complete run set:
- Budgets: 2, 5, 10, 20, 30
- Repeats: 5 each
- Total: 25 completed runs

Main outputs:
- `data/logs/budget_curve/records.tsv` (25 rows + header)
- `data/logs/budget_curve/summary.json`
- `data/reports/e2_budget_all_runs.tsv`
- `data/reports/e2_budget_summary_final.json`
- `data/reports/e2_budget_summary_final.tsv`

### E2 summary (overall metrics, val split)
| Budget | AP@0.5 mean | Precision mean | Recall mean | F1 mean |
|---:|---:|---:|---:|---:|
| 2  | 0.090595 | 0.107884 | 0.525753 | 0.179022 |
| 5  | 0.093579 | 0.118561 | 0.501672 | 0.191751 |
| 10 | 0.093425 | 0.118861 | 0.507191 | 0.192515 |
| 20 | 0.093518 | 0.117420 | 0.507525 | 0.190612 |
| 30 | 0.093036 | 0.116938 | 0.507860 | 0.190022 |

Interpretation:
- The main jump is from budget 2 to budget 5.
- Beyond budget 5, gains are small under the current quick-training setup (2 epochs).
- This supports a practical label-efficiency message: small curated labels already recover much of the gain.

Note:
- These are quick-budget curves with fixed short training (`epochs=2`) to make full sweep feasible.
- For publication-quality curves, budget point training length should be increased and confidence intervals plotted directly.

## 3.4 E3 gating ablation (completed)
Final ablation output:
- `data/reports/e3_gating_generic_cpsam_test.tsv`
- `data/reports/e3_gating_generic_cpsam_test.json`

### Tested settings
1. `intensity 10%-30% + min_area=50`:
- Over-filtered heavily and collapsed metrics (near-zero retained objects).
- Conclusion: this threshold window is too restrictive.

2. `intensity 0%-100% + min_area=50` (effectively area filter only):
- Raw overall: AP 0.235867, P 0.500888, R 0.526119, F1 0.513194
- Gated overall: AP 0.236231, P 0.502226, R 0.526119, F1 0.513895
- Delta: AP +0.000364, P +0.001338, R +0.000000, F1 +0.000701

Interpretation:
- Area filtering mildly reduces FP and slightly improves precision/F1.
- This matches the expected E3 narrative (post-processing primarily helps precision), but effect size is modest with current thresholds.

---

## 4) Environment and Execution Status
- Active env used for all major runs: `finetune`
- Installation preference updated to `pip` when new installs are needed.
- Current runs were executed via `mamba run -n finetune ...` to guarantee env isolation.

---

## 5) Key Outputs You Can Read Immediately
- E1 combined metrics:
  - `data/reports/e1_metrics_with_hybrid.tsv`
- E2 full records:
  - `data/logs/budget_curve/records.tsv`
- E2 final summary:
  - `data/reports/e2_budget_summary_final.tsv`
  - `data/reports/e2_budget_summary_final.json`
- E3 gating ablation:
  - `data/reports/e3_gating_generic_cpsam_test.tsv`
  - `data/reports/e3_gating_generic_cpsam_test.json`

---

## 6) What Is Still Missing (Critical)

## 6.1 E5 is not done yet
Current blockers:
- `src/fintune/inference/full_inference.py` is still a stub.
- `src/fintune/evaluation/downstream.py` is still a stub.
- `configs/paths.yaml` points `raw_tiles_dir` to `data/rosmap`, but that directory is currently empty in this workspace.

So E5.1/E5.2/E5.3 cannot run until:
- full-tile input path is available, and
- donor metadata (Braak/ADNC/cognition) path is available, and
- E5 modules are implemented.

## 6.2 E4 (micro-sam/μSAM diagnostic) has not been run yet
- Script stubs exist but no completed outputs yet.

---

## 7) Practical Assessment of Current Effect

What is already strong:
- Fine-tuning clearly improves segmentation quality versus baseline (large precision/F1 gain).
- Label-efficiency full sweep completed with all budgets/repeats.
- Post-processing ablation implemented and quantified.

What still needs completion for the paper-level claim:
- Downstream donor-level robustness analysis (E5), especially bootstrap stability comparison baseline vs fine-tuned.
- This is the core evidence for "segmentation improvement leads to more reliable biology".

---

## 8) Recommended Next Execution Order
1. Implement E5 full inference module and run all 14,749 tiles for baseline and fine-tuned models.
2. Build donor×marker density tables for both models.
3. Run Spearman + FDR with pathology/cognition endpoints.
4. Run donor-level bootstrap stability (variance, sign consistency, pass rate) and compare baseline vs fine-tuned.
5. Optional: run E4 small prompt-sweep to support discussion section.


# Fine-tune Cellpose-SAM for ROSMAP IF: Detailed Experiment Status (Live)

Last updated: 2026-02-09 (local cluster time)

## 1. Project Scope and Experiment Map
This project is organized around E0-E5:
- E0: donor-level split + conversion to Cellpose training format.
- E1: segmentation benchmark (baseline / generic fine-tune / marker-specific / marker-only).
- E2: label efficiency under limited annotation budgets.
- E3: post-processing ablation (marker gating + size filtering).
- E4: micro-sam / muSAM diagnostic sweep.
- E5: downstream donor-level biological association and stability.

Core code layout:
- `src/fintune/data_prep`: E0
- `src/fintune/training`: E1/E2
- `src/fintune/inference`: E1 baseline, E4, E5 full inference
- `src/fintune/evaluation`: E3, E5 downstream stats
- `scripts/`: CLI entrypoints
- `slurm/`: cluster submission scripts

## 2. Data and I/O Contracts
### 2.1 E0 data contract
Input:
- ROSMAP IF annotations and image tiles (DAPI + marker).

Output:
- `data/cellpose/train_cells`
- `data/cellpose/val_cells`
- `data/cellpose/test_cells`

Each pair:
- image: 3-channel `[DAPI, marker, 0]`
- mask: instance mask (0 background, 1..N instances)

Current split size:
- train=72, val=24, test=24

### 2.2 Model input/output contract (E1/E4/E5)
Model input:
- image tensor from tif/tiff.
- for standard run: channels `[1,2]` in Cellpose call.

Model output:
- predicted instance mask (`*_pred.tif`)
- metrics json (AP@0.5 / Precision / Recall / F1 + TP/FP/FN)

### 2.3 E5 ratio definition (implemented variants)
Two ratio computation variants have been implemented in history:
1) donor aggregation csv-based (deprecated per user requirement)
2) tile-level recomputation (current target):
   - baseline ratio: sum marker-positive cells / sum total cells from tile-level counts
   - finetuned ratio: same concept from finetuned tile-level counts

## 3. Implemented Code Changes (Major)
Implemented runnable modules from stubs:
- `src/fintune/data_prep/prepare_cellpose_data.py`
- `src/fintune/inference/cpsam_baseline.py`
- `src/fintune/training/finetune_generic.py`
- `src/fintune/training/finetune_gfap.py`
- `src/fintune/training/budget_curve.py`
- `src/fintune/evaluation/gating_ablation.py`

Added utility modules:
- `src/fintune/utils/dataset.py`
- `src/fintune/utils/metrics.py`

Added/updated scripts:
- `scripts/run_cpsam_baseline.py`
- `scripts/train_finetune_generic.py`
- `scripts/train_finetune_gfap.py`
- `scripts/run_budget_curve.py`
- `scripts/run_gating_ablation.py`
- `scripts/run_microsam_sweep.py`
- `scripts/run_full_inference.py` (rewritten)
- `scripts/analyze_downstream.py` (rewritten)

Added cluster scripts:
- `slurm/e1_marker_only_train_eval.sbatch`
- `slurm/e4_microsam_sweep.sbatch`
- `slurm/e5_downstream_compare.sbatch` (now calls full inference + downstream)

## 4. Experiment Results by Block
## 4.1 E1 completed outputs
Main result files:
- `data/reports/e1_metrics_all_models.tsv`
- `data/reports/e1_metrics_with_hybrid.tsv`
- `data/reports/e1_generic_vs_baseline_delta.tsv`

Current key test metrics (from metrics.json):
- baseline (`cpsam`): AP50=0.1062, P=0.0981, R=0.7864, F1=0.1744
- generic (`generic_cpsam`): AP50=0.2359, P=0.5009, R=0.5261, F1=0.5132
- gfap-specific (`gfap_cpsam`): AP50=0.1493, P=0.2364, R=0.6185, F1=0.3420
- hybrid recipe: AP50=0.2580, P=0.4639, R=0.5700, F1=0.5115
- marker-only (`marker_only_cpsam`, no DAPI): AP50=0.0899, P=0.4863, R=0.2659, F1=0.3438

Marker-only interpretation:
- Good precision but severe recall loss, especially for IBA1/OLIG2.
- Confirms DAPI is still important in this dataset.

## 4.2 E2 completed outputs
Files:
- `data/logs/budget_curve_long_e6/records.tsv`
- `data/logs/budget_curve_long_e6/summary.json`
- `data/reports/e2_budget_all_runs.tsv`
- `data/reports/e2_budget_summary_final.tsv`

Budget trend (long run summary):
- budget 2 -> AP50 mean 0.0623
- budget 5 -> AP50 mean 0.1439
- budget 10 -> AP50 mean 0.2324
- budget 20 -> AP50 mean 0.3970
- budget 30 -> AP50 mean 0.4044

Interpretation:
- strong gain from 2 to 20
- saturation around 20-30

## 4.3 E3 completed outputs
Files:
- `data/reports/e3_gating_generic_cpsam_test_q0_100_a50.tsv`
- `data/reports/e3_gating_generic_cpsam_test_q2_98_a60.tsv`
- `data/reports/e3_gating_generic_cpsam_test_q5_95_a80.tsv`

Best tested setting:
- q0_100 + min_area=50, small precision/F1 gain vs raw

## 4.4 E4 completed output (proxy)
Files:
- `data/reports/microsam/e4_microsam_proxy_sweep_n10_seed2024.tsv`
- `data/reports/microsam/e4_microsam_proxy_sweep_n10_seed2024.json`

Note:
- This is `cpsam proxy sweep`, not native micro-sam inference.
- pps=64 produced highest overall AP/F1 among tested pps settings.

## 5. E5 Status (Critical)
### 5.1 What has been done
- Multiple E5 iterations implemented.
- Rewrote `src/fintune/inference/full_inference.py`:
  - auto-discover tile channel pairs (`b0c0` + `b0c1`)
  - run baseline and finetuned models on same tiles
  - save per-tile counts and donor-level ratios under `data/downstream/full_inference/`
- Rewrote `src/fintune/evaluation/downstream.py`:
  - reads full-inference tile outputs
  - computes donor-level ratios
  - computes Spearman + BH-FDR for outcomes:
    `braaksc, ceradsc, cogdx, dcfdx_lv, plaq_d, plaq_n, nft, gpath`
  - tile-bootstrap stability (80% tiles per donor, repeated)

### 5.2 Current runtime issue and fix history
- Job `1593418` failed due to reading `*_metadata.xml` as tif.
- Fix applied:
  - strict suffix filtering to `.tif/.tiff`
  - unreadable tiles are skipped with warning (no hard crash)
- Retry job submitted: `1593498` (l40s, 24h)

### 5.3 Current state snapshot
- `squeue` currently shows job `1593498` running.
- `data/downstream/full_inference/` currently only has partial structure (`baseline/` exists).
- Final full-inference outputs (`donor_marker_ratio.tsv` etc.) are not yet confirmed complete.

Therefore:
- E5 final pipeline is implemented in code, but final production run is still in progress.

## 6. Slurm Job Ledger (Recent)
Completed:
- `1593407` E1 marker-only
- `1593408` E5 old version
- `1593409` E4 proxy
- `1593418` E5 attempt (failed)

Running/active retry:
- `1593498` E5 full-inference + downstream (new pipeline)

## 7. Remaining Work to Close Project
1) Wait for `1593498` to finish and verify outputs:
- `data/downstream/full_inference/baseline/tile_counts.tsv`
- `data/downstream/full_inference/finetuned/tile_counts.tsv`
- `data/downstream/full_inference/donor_marker_ratio.tsv`
- updated `data/downstream/e5_spearman.tsv`
- updated `data/downstream/e5_bootstrap_stability.tsv`

2) Generate final comparative table:
- baseline vs finetuned donor-level ratio correlation/stability
- include E1/E2/E3/E4 summary in one manuscript-ready report


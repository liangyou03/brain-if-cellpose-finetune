# Fine-tune Cellpose-SAM for ROSMAP IF: Full Reproducibility Status

Last update: 2026-02-09
Repository: `brain-if-cellpose-finetune`
Environment: `mamba` env `finetune`

## 0. Experiment-to-Paper Map (What / Why / Figure)
| Block | What the experiment does | Paper contribution | Recommended figure/table |
|---|---|---|---|
| E0 | Donor-level split + Cellpose-format conversion | Reproducibility and leakage prevention | Data pipeline diagram + split count bars |
| E1.1 | Off-the-shelf `cpsam` on test | Zero-shot baseline anchor | Per-marker AP/P/R/F1 bars + prediction examples |
| E1.2 | Generic fine-tune on all markers | Main segmentation gain claim | Baseline vs generic comparison table/plot |
| E1.3 | GFAP-specific fine-tune | Supports morphology-specific adaptation | GFAP-only comparison panel |
| E1.4 | Marker-only (no DAPI) | Negative control showing DAPI value | Generic vs marker-only recall/F1 plot |
| E2 | Budget sweep with repeated sampling | Label-efficiency selling point | AP@0.5 vs budget and Recall vs budget curves |
| E3 | Gating/no-gating ablation | Explains post-processing FP/precision effect | Raw vs gated metric bars, parameter sweep table |
| E4 | Prompt sweep (proxy here) | Discussion support on prompt stability limits | pps vs AP/Recall/FP curve/table |
| E5 | Donor-level ratio association + stability | Core biological impact claim | Correlation heatmap + bootstrap stability plots |

## 1. Global Setup
## 1.1 Path config
Active config file: `configs/paths.yaml`

Key paths used:
- `data/cellpose/train_cells`, `data/cellpose/val_cells`, `data/cellpose/test_cells`
- `data/checkpoints`
- `data/predictions`
- `data/reports`
- `data/downstream`
- `data/logs/slurm`

## 1.2 Metric definition (all E1/E2/E3/E4 reports)
Implemented in `src/fintune/utils/metrics.py`:
- AP@0.5 from `cellpose.metrics.average_precision(..., threshold=[0.5])`
- TP/FP/FN summed over images
- Precision = `TP / (TP + FP + 1e-8)`
- Recall = `TP / (TP + FN + 1e-8)`
- F1 = `2PR / (P + R + 1e-8)`

## 2. E0 Data Preparation
## 2.1 Script and parameters
Command entry: `scripts/prepare_cellpose_data.py`
Main params (default):
- `--train_frac 0.6`
- `--val_frac 0.2`
- `--seed 2024`
- `--format png`

Core logic (`src/fintune/data_prep/prepare_cellpose_data.py`):
1. Read `*_cellbodies.npy` as instance masks.
2. Parse donor from filename (`marker_donor` pattern).
3. Donor-level shuffle with `numpy.random.default_rng(seed)`.
4. Split donors into train/val/test by fractions.
5. Build 3-channel image: `[DAPI, marker, zeros]`.
6. Save image + `*_mask.tif` pair.

## 2.2 Outputs
- `data/splits/donor_splits.yaml`
- `data/cellpose/train_cells` (72 pairs)
- `data/cellpose/val_cells` (24 pairs)
- `data/cellpose/test_cells` (24 pairs)

## 3. E1 Segmentation Experiments
## 3.1 E1.1 Baseline (off-the-shelf cpsam)
Script:
- `scripts/run_cpsam_baseline.py`

Important runtime params:
- `model=cpsam`
- `split=test`
- `flow_threshold=0.4`
- `cellprob_threshold=0.0`
- `channels=[1,2]`

Batch command used in slurm `e1_baseline_test.sbatch`:
```bash
mamba run -n finetune python scripts/run_cpsam_baseline.py \
  --config configs/paths.yaml --model cpsam --split test
```

Output:
- `data/predictions/cpsam/test/*.tif`
- `data/predictions/cpsam/test/metrics.json`

Observed overall (test):
- AP50=0.1062, P=0.0981, R=0.7864, F1=0.1744

## 3.2 E1.2 Generic fine-tune (all markers)
Scripts:
- train: `scripts/train_finetune_generic.py`
- eval: `scripts/run_cpsam_baseline.py --model <ckpt>`

Train params used (slurm `e1_generic_train_eval.sbatch`):
- `epochs=80`
- `lr=1e-4`
- `batch=2`
- `nimg_per_epoch=72`
- `nimg_test_per_epoch=24`
- pretrained=`cpsam`

Command:
```bash
mamba run -n finetune python scripts/train_finetune_generic.py \
  --config configs/paths.yaml --epochs 80 --lr 1e-4 --batch 2 \
  --nimg-per-epoch 72 --nimg-test-per-epoch 24
```

Output checkpoint:
- `data/checkpoints/generic_cpsam/models/generic_cpsam`

Test eval command:
```bash
mamba run -n finetune python scripts/run_cpsam_baseline.py \
  --config configs/paths.yaml \
  --model data/checkpoints/generic_cpsam/models/generic_cpsam \
  --split test
```

Output:
- `data/predictions/generic_cpsam/test/metrics.json`

Observed overall (test):
- AP50=0.2359, P=0.5009, R=0.5261, F1=0.5132

## 3.3 E1.3 GFAP-specific fine-tune
Scripts:
- `scripts/train_finetune_gfap.py`
- internally calls `finetune_generic(..., marker_filter='gfap', model_name='gfap_cpsam')`

Train params used (`e1_gfap_train_eval.sbatch`):
- `epochs=80`
- `lr=1e-4`
- `batch=2`
- `nimg_per_epoch=18`
- `nimg_test_per_epoch=6`

Output:
- checkpoint: `data/checkpoints/gfap_cpsam/models/gfap_cpsam`
- metrics: `data/predictions/gfap_cpsam/test/metrics.json`

Observed overall (test):
- AP50=0.1493, P=0.2364, R=0.6185, F1=0.3420

## 3.4 E1.4 Marker-only fine-tune (zero DAPI)
Implemented by adding `--zero-dapi` to:
- `scripts/train_finetune_generic.py`
- `scripts/run_cpsam_baseline.py`

Slurm `e1_marker_only_train_eval.sbatch` params:
- train: `epochs=8`, `lr=1e-4`, `batch=2`, `model_name=marker_only_cpsam`, `zero_dapi=True`
- eval: `split=val` and `split=test`, both with `zero_dapi=True`

Outputs:
- checkpoint: `data/checkpoints/marker_only_cpsam/models/marker_only_cpsam`
- val metrics: `data/predictions/marker_only_cpsam/val/metrics.json`
- test metrics: `data/predictions/marker_only_cpsam/test/metrics.json`

Observed overall:
- val: AP50=0.0603, P=0.3674, R=0.1773, F1=0.2391
- test: AP50=0.0899, P=0.4863, R=0.2659, F1=0.3438

## 3.5 Hybrid recipe (report-level combination)
Output file:
- `data/reports/e1_metrics_with_hybrid.tsv`

Hybrid test overall:
- AP50=0.2580, P=0.4639, R=0.5700, F1=0.5115

## 4. E2 Label Efficiency
## 4.1 Experiment design
Script: `scripts/run_budget_curve.py` -> `src/fintune/training/budget_curve.py`

Config used in long run (`slurm/e2_budget_curve_long.sbatch`):
- budgets: `2 5 10 20 30`
- repeats: `5`
- epochs: `6`
- batch: `2`
- eval split: `val`
- `nimg_per_epoch=72`
- `nimg_test_per_epoch=24`
- `--skip-existing`
- `exp_name=budget_curve_long_e6`

Total runs designed:
- 5 budgets x 5 repeats = 25 fine-tune + eval loops

Sampling logic per run:
- For each marker group, sample `min(budget, n_available)` from train split.
- Seed source: `random.Random(seed)` with default seed `2024`.

## 4.2 Outputs
- `data/logs/budget_curve_long_e6/records.tsv`
- `data/logs/budget_curve_long_e6/summary.json`
- tagged snapshots (`records_*.tsv`, `summary_*.json`)

Summary means (overall AP50):
- b2=0.0623
- b5=0.1439
- b10=0.2324
- b20=0.3970
- b30=0.4044

## 5. E3 Gating Ablation
## 5.1 Experiment design
Script: `scripts/run_gating_ablation.py`
Core function: `src/fintune/evaluation/gating_ablation.py`

Rule per predicted instance:
1. area must be `>= min_area`
2. marker mean intensity inside mask must be in `[q_low, q_high]` percentile range
3. remaining objects are relabeled sequentially

Metrics computed for raw vs gated on same split.

## 5.2 Configs run (slurm `e3_gating_sweep.sbatch`)
1) `q0_100`, `min_area=50`
2) `q5_95`, `min_area=80`
3) `q2_98`, `min_area=60`

Outputs:
- `data/reports/e3_gating_generic_cpsam_test_q0_100_a50.tsv`
- `data/reports/e3_gating_generic_cpsam_test_q5_95_a80.tsv`
- `data/reports/e3_gating_generic_cpsam_test_q2_98_a60.tsv`
- matching `.json`

## 6. E4 Diagnostic Sweep
## 6.1 What was run
Script: `scripts/run_microsam_sweep.py`
Implementation: `src/fintune/inference/microsam_sweep.py`

Because `micro_sam` is not installed, this run uses `cpsam` proxy:
- pps=16 -> flow_threshold=0.6
- pps=32 -> flow_threshold=0.4
- pps=64 -> flow_threshold=0.2

Sampling:
- `num_tiles=10` per marker from test split
- seed=`2024`

Outputs:
- `data/reports/microsam/e4_microsam_proxy_sweep_n10_seed2024.tsv`
- `data/reports/microsam/e4_microsam_proxy_sweep_n10_seed2024.json`

## 7. E5 Downstream (Most Iterated Block)
## 7.1 Current target design
User requirement enforced:
- Compute baseline vs finetuned cell ratio from scratch.
- Do not rely on external pre-aggregated donor ratio tables.
- Clinical file from repo data path: `data/ROSMAP_clinical_n69.csv`.

Current code path:
- inference: `src/fintune/inference/full_inference.py`
- downstream stats: `src/fintune/evaluation/downstream.py`
- slurm: `slurm/e5_downstream_compare.sbatch`

## 7.2 Full inference logic
1. Discover channel-paired tile files from `data/rosmap` using pattern:
   - c0: `*_b0c0x*.tif*`
   - c1: replace `b0c0` -> `b0c1`
2. Filter to real TIFF suffix (`.tif/.tiff`).
3. Run both models on same tile list:
   - baseline model: `cpsam`
   - finetuned model: `generic_cpsam` (checkpoint-resolved path if local checkpoint exists)
4. For each predicted instance mask:
   - `n_total_cells = #instances`
   - `n_marker_positive = #instances with mean(marker) > percentile(marker, 75)`
   - `ratio = n_marker_positive / n_total_cells`
5. Aggregate donor-level ratio by model x donor x marker.

Target outputs under `data/downstream/full_inference`:
- `baseline/tile_counts.tsv`
- `finetuned/tile_counts.tsv`
- `tile_counts_all.tsv`
- `donor_marker_ratio.tsv`
- `summary.json`

## 7.3 Downstream stats logic
Input defaults:
- baseline tiles: `data/downstream/full_inference/baseline/tile_counts.tsv`
- finetuned tiles: `data/downstream/full_inference/finetuned/tile_counts.tsv`
- clinical: `data/ROSMAP_clinical_n69.csv`

Outcomes used:
- `braaksc, ceradsc, cogdx, dcfdx_lv, plaq_d, plaq_n, nft, gpath`

Computation:
- donor-level ratio from tile sums
- Spearman rho + p-value
- BH-FDR across all tested combinations
- bootstrap stability (tile-level):
  - per donor resample 80% tiles
  - repeat `n_bootstrap=200`
  - report variance of rho, sign consistency, p<0.05 pass rate

Outputs:
- `data/downstream/e5_donor_marker_ratio_from_tiles.tsv`
- `data/downstream/e5_spearman.tsv`
- `data/downstream/e5_bootstrap_stability.tsv`
- `data/downstream/e5_summary.json`

## 7.4 E5 run attempts and current state
Jobs:
- `1593399` old E5 path completed
- `1593408` old E5 path completed
- `1593418` failed (xml read as tif)
- `1593498` retry with tif filtering + unreadable skip handling

Failure root cause fixed in code:
- `*_metadata.xml` was included by `*.tif*` glob and passed to tif reader.
- Fix:
  - suffix filter `.tif/.tiff`
  - try/except around tile read with skip warning

Current status note:
- cluster accounting and live queue show inconsistent state snapshots for `1593498`.
- Final E5 full outputs are not yet confirmed as complete in filesystem.

## 8. Consolidated Output Index
E1:
- `data/predictions/cpsam/test/metrics.json`
- `data/predictions/generic_cpsam/test/metrics.json`
- `data/predictions/gfap_cpsam/test/metrics.json`
- `data/predictions/marker_only_cpsam/val/metrics.json`
- `data/predictions/marker_only_cpsam/test/metrics.json`
- `data/reports/e1_metrics_with_hybrid.tsv`

E2:
- `data/logs/budget_curve_long_e6/records.tsv`
- `data/logs/budget_curve_long_e6/summary.json`
- `data/reports/e2_budget_summary_final.tsv`

E3:
- `data/reports/e3_gating_generic_cpsam_test_q0_100_a50.tsv`
- `data/reports/e3_gating_generic_cpsam_test_q2_98_a60.tsv`
- `data/reports/e3_gating_generic_cpsam_test_q5_95_a80.tsv`

E4:
- `data/reports/microsam/e4_microsam_proxy_sweep_n10_seed2024.tsv`

E5 (legacy existing files):
- `data/downstream/e5_spearman.tsv`
- `data/downstream/e5_bootstrap_stability.tsv`
- `data/downstream/e5_summary.json`

E5 (new full path expected):
- `data/downstream/full_inference/*`

## 9. What Is Still Not Fully Closed
Only one critical unfinished closure remains:
- finalize E5 full-inference + downstream rerun outputs and lock final tables.

Everything else (E0-E4 + E1.4) has runnable code, outputs, and tracked logs.

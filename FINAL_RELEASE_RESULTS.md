# Final Release Results (Current Snapshot)

## 1) Dataset and split used in current results
- Data format: 3-channel tile input `(DAPI, marker, 0)` + instance mask (`0` background, `1..N` cell ids).
- Current donor-level split in `data/cellpose/`:
  - `train_cells`: 72 images (`gfap=19, iba1=18, neun=18, olig2=17`)
  - `val_cells`: 24 images
  - `test_cells`: 24 images

## 2) E1 final test results (threshold-selected on val)
Source: `data/reports/e1_threshold_selected_test.tsv`

- `baseline` (`flow=0.2, cellprob=-1.0`):
  - Test AP50 `0.1363`, Precision `0.1313`, Recall `0.7183`, F1 `0.2220`
- `generic` (`flow=0.4, cellprob=-1.0`):
  - Test AP50 `0.7040`, Precision `0.8042`, Recall `0.8890`, F1 `0.8445`
- `gfap` (`flow=0.6, cellprob=-1.0`):
  - Test AP50 `0.4889`, Precision `0.7217`, Recall `0.6119`, F1 `0.6623`
- `marker_only` (`flow=0.4, cellprob=-1.0`):
  - Test AP50 `0.6197`, Precision `0.8230`, Recall `0.7938`, F1 `0.8082`

## 3) E2 label-efficiency figure status
- Kept figure:
  - `data/reports/fig_pub/fig_generic_budget_curve_with_effective_n.pdf`
- Important interpretation:
  - Tick label `N` is **effective train image count** (not cell count).
  - In previous run (`budget 2/5/10/20/30`), `20` and `30` are both saturated at `N=72`.

## 4) E3 post-processing (gating) final test
Source:
- `data/reports/e3_gating_test_final_generic_cpsam.tsv`
- `data/reports/e3_gating_generic_cpsam_test_q0_100_a50.tsv`

Selected setting: `q_low=0, q_high=100, min_area=50`

Overall test delta (gated - raw):
- AP50 `+0.00088`
- Precision `+0.00225`
- Recall `-0.00093`
- F1 `+0.00063`

Interpretation: gating has a small but positive net effect on overall AP50/F1 in current setup.

## 5) Cleanup applied
Removed temporary/wrong-path artifacts:
- `scripts/eval_budget_splits.py`
- `scripts/plot_generic_budget_curve_splits.py`
- `slurm/e2_budget_curve_train_val_test_plot.sbatch`
- `data/reports/fig_pub/fig_generic_budget_curve_train_val_test_budget.pdf`
- `data/reports/fig_pub/fig_generic_budget_curve_train_val_test_budget.png`
- `data/reports/fig_pub/fig_generic_budget_curve_train_val_test_train_n.pdf`
- `data/reports/fig_pub/fig_generic_budget_curve_train_val_test_train_n.png`

## 6) Budget policy update (implemented)
E2.1 budget defaults are updated to avoid saturated redundant points:
- New default budgets: `2, 5, 10, 15, 17, 19`
- Updated files:
  - `scripts/run_budget_curve.py`
  - `slurm/e2_budget_curve_long.sbatch`
  - `slurm/e2_budget_curve_force_rerun.sbatch`


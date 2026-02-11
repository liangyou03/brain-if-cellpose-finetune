# Project Report (Simple English)

Last updated: February 10, 2026

## Objective
This project aims to fine-tune Cellpose-SAM for ROSMAP brain IF images and test one main hypothesis:

Better segmentation quality should produce more reliable donor-level biological signals.

The practical goals are:
- improve segmentation on marker-specific cell bodies with limited labels,
- measure label efficiency,
- test if improved segmentation gives more stable downstream pathology/cognition associations.

## Experiments
E0 data setup:
- Donor-level split and Cellpose format conversion (`DAPI`, `marker`, `0` channel).
- Current split size: 72 train, 24 val, 24 test image-mask pairs.

E1 segmentation core:
- E1.1 baseline: off-the-shelf `cpsam` on test.
- E1.2 generic fine-tune: all markers together.
- E1.3 GFAP-specific fine-tune.
- E1.4 marker-only control (no DAPI).
- E1.5 threshold sweep on val, then fixed threshold on test.
- E1.6 seed sweep (3 seeds) for training stability.

E2 label efficiency:
- E2.2 active-sampling compare: `random` vs `hard-first` across budgets {2,5,10,20}, 5 repeats each.

E3 post-processing:
- Gating sweep on val, selected setting applied on test.

E5 downstream:
- E5.1/E5.2/E5.3 full inference + donor aggregation + correlation/stability are running now.
- E5.5 adjusted model analysis is queued and depends on E5 completion.

## Results
### 1) Main segmentation gain is strong and consistent
Test overall metrics:
- Baseline (`cpsam`): AP50 0.1065, Precision 0.0982, Recall 0.7873, F1 0.1746.
- Generic fine-tune: AP50 0.6740, Precision 0.8430, Recall 0.8312, F1 0.8370.

Analysis:
- AP50 improved by +0.5674 (about 6.3x).
- F1 improved by +0.6624.
- Baseline has very high recall but extremely low precision (strong over-segmentation / many false positives).
- Fine-tuning corrected this error pattern while keeping high recall.

### 2) Threshold selection improves operating point
`generic_thsel` test overall:
- AP50 0.7040, Precision 0.8042, Recall 0.8890, F1 0.8445.

Analysis:
- Compared with default generic threshold: recall +0.0578, precision -0.0387.
- This is a useful precision-recall tradeoff when recall is prioritized.

### 3) DAPI channel is important
Marker-only control (`marker_only_cpsam`) test overall:
- AP50 0.5152, F1 0.7465.

Analysis:
- Versus generic model: AP50 -0.1588, F1 -0.0905.
- Conclusion: removing DAPI clearly hurts segmentation quality.

### 4) Seed stability is good (not luck)
E1.6 (seeds 2024/2025/2026):
- AP50 mean 0.6738, std 0.0028.
- F1 mean 0.8361, std 0.0022.

Analysis:
- Very small variance across seeds supports reproducibility.

### 5) Active-sampling heuristic is not consistently better yet
E2.2 AP50 means (`hard - random`):
- Budget 2: -0.0601
- Budget 5: +0.0148
- Budget 10: -0.0209
- Budget 20: +0.0043

Analysis:
- Current hard-first scoring gives mixed results.
- Random sampling remains a strong baseline.
- This is still informative: the current heuristic is not robust and needs redesign.

### 6) Gating effect is small under current selected setting
Selected E3 setting (`q_low=0.0`, `q_high=1.0`, `min_area=50`) on test:
- Overall F1 changed from 0.8370 to 0.8376 (very small increase).

Analysis:
- Under current model/threshold, gating is a minor refinement, not a major driver.

### 7) Downstream (E5) is still running
Current status:
- `1593792` (E5 downstream) is RUNNING.
- `1593808` (E5 adjusted models) is PENDING with dependency on `1593792`.

Interpretation impact:
- Core segmentation claims are already strong.
- Final paper-level biological claim still depends on E5 completion.

## Methods
Data:
- Two-channel IF (DAPI + one marker), converted to 3-channel input for Cellpose (`DAPI, marker, 0`).
- Donor-level split to reduce leakage risk.

Training:
- Cellpose-SAM fine-tuning with GPU (`gpu=True`).
- Generic model uses all markers; marker-specific and marker-only controls are included.
- Standard core setting for main generic run: epochs 80, lr 1e-4, batch size 2.

Evaluation:
- Instance segmentation metrics: AP@0.5, Precision, Recall, F1.
- Threshold selection is done on val, then fixed for test.
- Reproducibility checked by multi-seed runs.

Label-efficiency:
- Multiple budgets and repeated random sampling.
- Compare baseline random sampling vs hard-first sampling strategy.

Downstream plan:
- Build donor-level marker features from full-tile inference.
- Correlate with clinical outcomes (`braaksc`, `ceradsc`, `cogdx`, `dcfdx_lv`, `plaq_d`, `plaq_n`, `nft`, `gpath`).
- Use bootstrap-based stability metrics and covariate-adjusted models.

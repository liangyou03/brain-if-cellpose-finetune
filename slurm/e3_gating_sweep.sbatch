#!/bin/bash
#SBATCH --job-name=e3_gating_sweep
#SBATCH --cluster=gpu
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH -o /ix/jbwang/liangyou/fintune/data/logs/slurm/%x-%j.out
#SBATCH -e /ix/jbwang/liangyou/fintune/data/logs/slurm/%x-%j.err

set -euo pipefail
cd /ix/jbwang/liangyou/fintune

export PYTHONPATH=src

# area-only baseline gating
mamba run -n finetune python scripts/run_gating_ablation.py \
  --config configs/paths.yaml \
  --model generic_cpsam \
  --split test \
  --intensity_thresh 0.0 1.0 \
  --min_area 50

# tighter intensity window + stronger area filter
mamba run -n finetune python scripts/run_gating_ablation.py \
  --config configs/paths.yaml \
  --model generic_cpsam \
  --split test \
  --intensity_thresh 0.05 0.95 \
  --min_area 80

# moderate filter
mamba run -n finetune python scripts/run_gating_ablation.py \
  --config configs/paths.yaml \
  --model generic_cpsam \
  --split test \
  --intensity_thresh 0.02 0.98 \
  --min_area 60

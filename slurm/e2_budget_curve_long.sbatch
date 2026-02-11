#!/bin/bash
#SBATCH --job-name=e2_budget_long
#SBATCH --cluster=gpu
#SBATCH --partition=l40s
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH -o /ix/jbwang/liangyou/fintune/data/logs/slurm/%x-%j.out
#SBATCH -e /ix/jbwang/liangyou/fintune/data/logs/slurm/%x-%j.err

set -euo pipefail
cd /ix/jbwang/liangyou/fintune

export PYTHONPATH=src
mamba run -n finetune python scripts/run_budget_curve.py \
  --config configs/paths.yaml \
  --budgets 2 5 10 15 17 19 \
  --repeats 5 \
  --epochs 6 \
  --batch 2 \
  --eval-split val \
  --nimg-per-epoch 72 \
  --nimg-test-per-epoch 24 \
  --skip-existing \
  --exp-name budget_curve_long_e6

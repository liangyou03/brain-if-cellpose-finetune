#!/bin/bash
#SBATCH --job-name=e1_seed_sweep
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
mamba run -n finetune python scripts/run_e1_seed_sweep_generic.py \
  --config configs/paths.yaml \
  --seeds 2024 2025 2026 2027 2028 \
  --epochs 80 \
  --lr 1e-4 \
  --batch 2 \
  --nimg-per-epoch 72 \
  --nimg-test-per-epoch 24 \
  --flow-threshold 0.4 \
  --cellprob-threshold 0.0 \
  --skip-existing

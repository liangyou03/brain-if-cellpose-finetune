#!/bin/bash
#SBATCH --job-name=e1_generic_train_eval
#SBATCH --cluster=gpu
#SBATCH --partition=l40s
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH -o /ix/jbwang/liangyou/fintune/data/logs/slurm/%x-%j.out
#SBATCH -e /ix/jbwang/liangyou/fintune/data/logs/slurm/%x-%j.err

set -euo pipefail
cd /ix/jbwang/liangyou/fintune

export PYTHONPATH=src
mamba run -n finetune python scripts/train_finetune_generic.py \
  --config configs/paths.yaml \
  --epochs 80 \
  --lr 1e-4 \
  --batch 2 \
  --nimg-per-epoch 72 \
  --nimg-test-per-epoch 24

mamba run -n finetune python scripts/run_cpsam_baseline.py \
  --config configs/paths.yaml \
  --model /ix/jbwang/liangyou/fintune/data/checkpoints/generic_cpsam/models/generic_cpsam \
  --split test

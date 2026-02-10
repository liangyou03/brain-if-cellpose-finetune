#!/bin/bash
#SBATCH --job-name=e2_active_sampling
#SBATCH --cluster=gpu
#SBATCH --partition=l40s
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH -o /ix/jbwang/liangyou/fintune/data/logs/slurm/%x-%j.out
#SBATCH -e /ix/jbwang/liangyou/fintune/data/logs/slurm/%x-%j.err

set -euo pipefail
cd /ix/jbwang/liangyou/fintune

export PYTHONPATH=src
mamba run -n finetune python scripts/run_e2_active_sampling_compare.py \
  --config configs/paths.yaml \
  --budgets 2 5 10 20 \
  --repeats 5 \
  --epochs 40 \
  --lr 1e-4 \
  --batch 2 \
  --eval-split val \
  --nimg-per-epoch 72 \
  --nimg-test-per-epoch 24 \
  --exp-name e2_active_sampling_compare \
  --scoring-model cpsam \
  --score-flow-a 0.4 \
  --score-cellprob-a 0.0 \
  --score-flow-b 0.2 \
  --score-cellprob-b -0.5 \
  --skip-existing

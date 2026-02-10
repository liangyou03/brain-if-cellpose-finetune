#!/bin/bash
#SBATCH --job-name=e6_lomo
#SBATCH --cluster=gpu
#SBATCH --partition=l40s
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=16:00:00
#SBATCH --array=0-3
#SBATCH -o /ix/jbwang/liangyou/fintune/data/logs/slurm/%x-%A_%a.out
#SBATCH -e /ix/jbwang/liangyou/fintune/data/logs/slurm/%x-%A_%a.err

set -euo pipefail
cd /ix/jbwang/liangyou/fintune
export PYTHONPATH=src

markers=(gfap iba1 neun olig2)
heldout="${markers[$SLURM_ARRAY_TASK_ID]}"
seed=$((2024 + SLURM_ARRAY_TASK_ID))

mamba run -n finetune python scripts/run_e6_lomo_single.py \
  --config configs/paths.yaml \
  --heldout-marker "${heldout}" \
  --epochs 40 \
  --lr 1e-4 \
  --batch 2 \
  --nimg-per-epoch 72 \
  --nimg-test-per-epoch 24 \
  --flow-threshold 0.4 \
  --cellprob-threshold 0.0 \
  --seed "${seed}"

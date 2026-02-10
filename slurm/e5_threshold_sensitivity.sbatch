#!/bin/bash
#SBATCH --job-name=e5_thresh_sens
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

mamba run -n finetune python scripts/run_e5_threshold_sensitivity.py \
  --config configs/paths.yaml \
  --baseline-model cpsam \
  --finetuned-model generic_cpsam \
  --percentiles 60 70 75 80 90 \
  --flow-threshold 0.4 \
  --cellprob-threshold 0.0 \
  --bootstrap 100 \
  --seed 2024

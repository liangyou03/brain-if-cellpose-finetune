#!/bin/bash
#SBATCH --job-name=e1_thresh_sweep
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

mamba run -n finetune python scripts/run_e1_threshold_sweep.py \
  --config configs/paths.yaml \
  --models baseline generic gfap marker_only \
  --flow-thresholds 0.2 0.4 0.6 \
  --cellprob-thresholds -1.0 -0.5 0.0 0.5 1.0 \
  --select-metric f1

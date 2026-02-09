#!/bin/bash
#SBATCH --job-name=e1_baseline_test
#SBATCH --cluster=gpu
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=05:00:00
#SBATCH -o /ix/jbwang/liangyou/fintune/data/logs/slurm/%x-%j.out
#SBATCH -e /ix/jbwang/liangyou/fintune/data/logs/slurm/%x-%j.err

set -euo pipefail
cd /ix/jbwang/liangyou/fintune

export PYTHONPATH=src
mamba run -n finetune python scripts/run_cpsam_baseline.py \
  --config configs/paths.yaml \
  --model cpsam \
  --split test

#!/bin/bash
#SBATCH --job-name=e3_gating_sweep
#SBATCH --cluster=gpu
#SBATCH --partition=l40s
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH -o /ix/jbwang/liangyou/fintune/data/logs/slurm/%x-%j.out
#SBATCH -e /ix/jbwang/liangyou/fintune/data/logs/slurm/%x-%j.err

set -euo pipefail
cd /ix/jbwang/liangyou/fintune

export PYTHONPATH=src

if [ -f data/reports/e3_gating_test_final_generic_cpsam.tsv ]; then
  echo "[skip] exists: data/reports/e3_gating_test_final_generic_cpsam.tsv"
else
  mamba run -n finetune python scripts/run_e3_val_select_test.py \
    --config configs/paths.yaml \
    --model generic_cpsam \
    --q-low 0.0 0.02 0.05 \
    --q-high 1.0 0.98 0.95 \
    --min-area 50 60 80
fi

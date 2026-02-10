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

run_if_missing () {
  local out_file="$1"
  shift
  if [ -f "$out_file" ]; then
    echo "[skip] exists: $out_file"
  else
    mamba run -n finetune python scripts/run_gating_ablation.py "$@"
  fi
}

run_if_missing \
  data/reports/e3_gating_generic_cpsam_test_q0_100_a50.tsv \
  --config configs/paths.yaml \
  --model generic_cpsam \
  --split test \
  --intensity_thresh 0.0 1.0 \
  --min_area 50

run_if_missing \
  data/reports/e3_gating_generic_cpsam_test_q5_95_a80.tsv \
  --config configs/paths.yaml \
  --model generic_cpsam \
  --split test \
  --intensity_thresh 0.05 0.95 \
  --min_area 80

run_if_missing \
  data/reports/e3_gating_generic_cpsam_test_q2_98_a60.tsv \
  --config configs/paths.yaml \
  --model generic_cpsam \
  --split test \
  --intensity_thresh 0.02 0.98 \
  --min_area 60

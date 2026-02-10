#!/bin/bash
#SBATCH --job-name=e5_downstream
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
mkdir -p data/downstream/full_inference
rm -rf data/downstream/full_inference/baseline
rm -rf data/downstream/full_inference/finetuned
rm -f data/downstream/full_inference/tile_counts_all.tsv
rm -f data/downstream/full_inference/donor_marker_ratio.tsv
rm -f data/downstream/full_inference/summary.json

mamba run -n finetune python scripts/run_full_inference.py \
  --config configs/paths.yaml \
  --baseline-model cpsam \
  --finetuned-model generic_cpsam \
  --no-resume

mamba run -n finetune python scripts/analyze_downstream.py \
  --config configs/paths.yaml \
  --pred-dir data/downstream/full_inference \
  --bootstrap 200 \
  --alpha 0.05 \
  --clinical-csv data/ROSMAP_clinical_n69.csv \
  --metrics ratio pos_density_mpx total_density_mpx \
  --seed 2024

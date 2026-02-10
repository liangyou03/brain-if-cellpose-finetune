#!/bin/bash
#SBATCH --job-name=e5_adjusted
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
mamba run -n finetune python scripts/run_e5_adjusted_models.py \
  --config configs/paths.yaml \
  --clinical-csv data/ROSMAP_clinical_n69.csv \
  --metrics ratio pos_density_mpx total_density_mpx \
  --covariates age_death msex pmi educ \
  --alpha 0.05 \
  --bootstrap 200 \
  --seed 2024

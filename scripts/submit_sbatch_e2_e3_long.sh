#!/bin/bash
set -euo pipefail
cd /ix/jbwang/liangyou/fintune
mkdir -p data/logs/slurm

jid_e2=$(sbatch slurm/e2_budget_curve_long.sbatch | awk '{print $4}')
jid_e3=$(sbatch slurm/e3_gating_sweep.sbatch | awk '{print $4}')

echo "Submitted jobs:"
echo "  E2 long: ${jid_e2}"
echo "  E3 sweep: ${jid_e3}"

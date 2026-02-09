#!/bin/bash
set -euo pipefail
cd /ix/jbwang/liangyou/fintune

mkdir -p data/logs/slurm

jid_base=$(sbatch slurm/e1_baseline_test.sbatch | awk '{print $4}')
jid_generic=$(sbatch slurm/e1_generic_train_eval.sbatch | awk '{print $4}')
jid_gfap=$(sbatch slurm/e1_gfap_train_eval.sbatch | awk '{print $4}')
jid_gating=$(sbatch --dependency=afterok:${jid_generic} slurm/e3_gating_generic_test.sbatch | awk '{print $4}')

echo "Submitted jobs:"
echo "  baseline: ${jid_base}"
echo "  generic:  ${jid_generic}"
echo "  gfap:     ${jid_gfap}"
echo "  gating:   ${jid_gating} (after generic)"

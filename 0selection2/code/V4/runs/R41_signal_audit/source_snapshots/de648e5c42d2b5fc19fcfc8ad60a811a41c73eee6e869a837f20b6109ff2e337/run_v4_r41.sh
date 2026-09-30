#!/usr/bin/env bash
set -euo pipefail
cd /public/home/shiys/0selection2
source /public/home/shiys/miniconda3/etc/profile.d/conda.sh
conda activate easynco
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
root=code/V4/runs/R41_signal_audit
python -u -m code.V4.r41_frozen_evaluation --stage prepare
if [[ ! -f "$root/test_results.json" ]]; then
    python -u -m code.V4.r41_frozen_evaluation --stage evaluate 2>&1 | tee "$root/frozen_eval.log"
fi
python -u -m code.V4.r41_frozen_evaluation --stage audit-predict 2>&1 | tee "$root/audit_predictions.log"
python -u -m code.V4.r41_label_stability --stage prepare
python -u -m code.V4.r41_label_stability --stage manifest
python -u -m code.V4.r41_label_stability --stage precheck 2>&1 | tee "$root/solver_precheck.log"
python -u -m code.V4.r41_label_stability --stage run 2>&1 | tee "$root/solver_repeats.log"
flags=(--plots)
if [[ ! -f "$root/wandb_status.json" ]]; then
    flags+=(--wandb)
fi
python -u -m code.V4.r41_analysis "${flags[@]}"

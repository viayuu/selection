#!/usr/bin/env bash
set -euo pipefail
source /public/home/shiys/miniconda3/etc/profile.d/conda.sh
conda activate easynco
cd /public/home/shiys/0selection2
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export MPLBACKEND=Agg PYTHONUNBUFFERED=1 WANDB_MODE=offline
stage=${1:-all}
root=code/V4/runs/R52_error_and_winner_mechanism
mkdir -p "$root"
if [[ "$stage" == all || "$stage" == budget ]]; then
    python -m code.V4.r52_error_budget 2>&1 | tee "$root/error_budget.log"
    python -m code.V4.r52_start_audit --stage prepare 2>&1 | tee "$root/prepare.log"
fi
if [[ "$stage" == all || "$stage" == replay ]]; then
    : "${R52_GPU_UUID:?Bind only the currently idle RTX3090 UUID}"
    export CUDA_VISIBLE_DEVICES="$R52_GPU_UUID"
    used=$(nvidia-smi --id="$R52_GPU_UUID" --query-gpu=memory.used --format=csv,noheader,nounits)
    [[ "$used" -lt 500 ]] || { printf 'Selected GPU is not idle: %s MiB\n' "$used"; exit 1; }
    python -m code.V4.r52_start_audit --stage replay 2>&1 | tee "$root/replay.log"
fi
if [[ "$stage" == all || "$stage" == analysis ]]; then
    python -m code.V4.r52_analysis 2>&1 | tee "$root/analysis.log"
fi
if [[ "$stage" == tests ]]; then
    python -m unittest code.V4.test_r52 -v 2>&1 | tee "$root/tests.log"
fi

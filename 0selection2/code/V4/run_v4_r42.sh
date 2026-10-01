#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
source /public/home/shiys/miniconda3/etc/profile.d/conda.sh
conda activate easynco
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS=4
export WANDB_MODE=offline
if [[ "${1:-}" == all ]]; then
    python -u -m code.V4.r42_experiment --stage train --group A
    python -u -m code.V4.r42_experiment --stage train --group B
    python -u -m code.V4.r42_experiment --stage test
    python -u -m code.V4.r42_analysis
else
    python -u -m code.V4.r42_experiment "$@"
fi

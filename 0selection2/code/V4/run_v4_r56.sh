#!/usr/bin/env bash
set -euo pipefail
source /public/home/shiys/miniconda3/etc/profile.d/conda.sh
conda activate easynco
cd /public/home/shiys/0selection2
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export PYTHONUNBUFFERED=1
export WANDB_MODE=offline
stage="${1:-all}"
if [[ "$stage" == tests ]]; then
    python -m unittest code.V4.test_r56 code.V4.test_r53 -q
else
    python -m code.V4.r56_experiment --stage "$stage"
fi

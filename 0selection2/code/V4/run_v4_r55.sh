#!/usr/bin/env bash
set -euo pipefail
source /public/home/shiys/miniconda3/etc/profile.d/conda.sh
conda activate easynco
cd /public/home/shiys/0selection2
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 MPLBACKEND=Agg PYTHONUNBUFFERED=1 WANDB_MODE=offline
stage="${1:-all}"
shift || true
case "$stage" in
  precheck)
    python -m unittest code.V4.test_r55 code.V4.test_r54 code.V4.test_r53
    python -m code.V4.r55_targets "$@"
    ;;
  prepare|train|evaluate)
    python -m code.V4.r55_experiment --stage "$stage" "$@"
    ;;
  all)
    python -m unittest code.V4.test_r55 code.V4.test_r54 code.V4.test_r53
    python -m code.V4.r55_experiment --stage all "$@"
    ;;
  *) printf 'Unknown stage: %s\n' "$stage" >&2; exit 2 ;;
esac

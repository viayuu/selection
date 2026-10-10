#!/usr/bin/env bash
set -euo pipefail
source /public/home/shiys/miniconda3/etc/profile.d/conda.sh
conda activate easynco
cd /public/home/shiys/0selection2
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export PYTHONUNBUFFERED=1
case "${1:-cpu}" in
  cpu)
    python -m unittest code.V4.test_r57 -v
    python -m code.V4.r57_signal
    python -m code.V4.r57_audit
    python -m code.V4.r57_routes
    ;;
  gpu)
    : "${CUDA_VISIBLE_DEVICES:?Bind only the authorized idle RTX3090 UUID}"
    if [[ "$CUDA_VISIBLE_DEVICES" != GPU-cb626bdb-4319-2f5e-b91f-17fcaa44a83d ]]; then
      printf '%s\n' 'Refusing a different GPU; verify the user-authorized idle3090.' >&2
      exit 1
    fi
    python -m code.V4.r57_probe
    ;;
  analysis)
    python -m code.V4.r57_audit
    python -m code.V4.r57_analysis
    ;;
  *)
    printf '%s\n' 'Usage: bash code/V4/run_v4_r57.sh {cpu|gpu|analysis}' >&2
    exit 2
    ;;
esac

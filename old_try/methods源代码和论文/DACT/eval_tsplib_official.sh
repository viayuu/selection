#!/usr/bin/env bash
set -euo pipefail

# Example: evaluate DACT on TSPLIB with TSPLIB official rounding cost/gap.
# This aligns the evaluation metric with EasyNCO TSPLIB logs in `final/logs/tsplib_70/*`.

TSPLIB_DIR="${TSPLIB_DIR:-EasyNCO/data/datasets/tsplib/tsplib}"
OUT_DIR="${OUT_DIR:-results}"

mkdir -p "${OUT_DIR}"

# Table-4-like settings (edit as needed):
# - T_max: 3000 or 10000
# - P (Tr): 10 for TSPLIB (paper setting)
# - val_m: 1 or 4 (augment)
# - seeds: 0..9 for "10 runs"

python 0methods/DACT/eval_tsplib_official.py \
  --tsplib_dir "${TSPLIB_DIR}" \
  --T_max 3000 \
  --P 10 \
  --val_m 4 \
  --seeds 0,1,2,3,4,5,6,7,8,9 \
  --report gap \
  --device cuda:0 \
  --out "${OUT_DIR}/dact_tsplib_official_T3000_P10_m4_seeds0-9.tsv"

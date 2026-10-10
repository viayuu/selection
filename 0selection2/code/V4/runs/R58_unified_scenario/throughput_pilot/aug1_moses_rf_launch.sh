#!/usr/bin/env bash
set -euo pipefail
source /public/home/shiys/miniconda3/etc/profile.d/conda.sh
conda activate easynco
cd /public/home/shiys/0selection2
PILOT=/public/home/shiys/0selection2/code/V4/runs/R58_unified_scenario/throughput_pilot
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPATH=/public/home/shiys/easynco_v3_bridge/exports/train_100k_20261002/python_compat:/public/home/shiys/0selection2
export WANDB_MODE=offline WANDB_DIR="$PILOT" MPLCONFIGDIR="$PILOT/mplconfig"
trap 'status=$?; printf "%s\n" "$status" > "$PILOT/aug1_moses_rf_exit_code"; exit "$status"' EXIT
exec > "$PILOT/aug1_moses_rf_controller.log" 2>&1
python -u -m code.V4.r58_throughput --stage launch-aug1-supplement --wall-seconds 1800

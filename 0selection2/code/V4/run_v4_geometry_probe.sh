#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
source /public/home/shiys/miniconda3/etc/profile.d/conda.sh
conda activate easynco
export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
STUDY=code/V4/runs/R37_tsp_local_geometry
mkdir -p "$STUDY"
if [[ ! -e "$STUDY/source.tar.gz" ]]; then
  tar --exclude=runs --exclude=__pycache__ -czf "$STUDY/source.tar.gz" code/V4 code/unified_selector/data.py code/unified_selector/registry.py
fi
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,power.draw --format=csv -l 10 >> "$STUDY/gpu.csv" &
MONITOR_PID=$!
trap 'code=$?; kill "$MONITOR_PID" 2>/dev/null || true; printf "%s\n" "$code" > "$STUDY/exit_code"' EXIT
python -m code.V4.geometry_probe --wandb "$@" 2>&1 | tee -a "$STUDY/queue.log"
python -m code.V4.report_geometry_probe 2>&1 | tee -a "$STUDY/queue.log"

#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source /public/home/shiys/miniconda3/etc/profile.d/conda.sh
conda activate easynco
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export WANDB_MODE=offline MPLBACKEND=Agg
unset VOYAGE_API_KEY

GPU_UUID=${R47_GPU_UUID:?Set R47_GPU_UUID to the authorized idle RTX3090 UUID}
export CUDA_VISIBLE_DEVICES="$GPU_UUID"
ROOT=code/V4/runs/R47_behavior_memory
mkdir -p "$ROOT"
exec > >(tee -a "$ROOT/orchestration.log") 2>&1
echo "[R47B] $(date -Iseconds) on $(hostname); only GPU $GPU_UUID; R47A disabled"
name=$(nvidia-smi --id="$GPU_UUID" --query-gpu=name --format=csv,noheader)
memory=$(nvidia-smi --id="$GPU_UUID" --query-gpu=memory.used --format=csv,noheader,nounits)
if [[ "$name" != *"RTX 3090"* ]] || (( memory > 128 )); then
    echo "Authorized GPU is not an idle RTX3090; leaving existing jobs untouched."
    exit 1
fi
nvidia-smi --id="$GPU_UUID" --query-gpu=timestamp,uuid,utilization.gpu,memory.used,power.draw \
    --format=csv -l 10 > "$ROOT/gpu_metrics.csv" &
monitor=$!
trap 'kill "$monitor" 2>/dev/null || true' EXIT
python -m unittest code.V4.test_r47 -q
python -u -m code.V4.r47_experiment "$@"
echo "[R47B] Training, mechanism check, locked-checkpoint test and analysis finished $(date -Iseconds)"

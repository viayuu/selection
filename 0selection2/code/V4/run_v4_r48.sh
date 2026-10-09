#!/usr/bin/env bash
set -euo pipefail
source /public/home/shiys/miniconda3/etc/profile.d/conda.sh
conda activate easynco
cd /public/home/shiys/0selection2
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export MPLBACKEND=Agg WANDB_MODE=offline PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${R48_GPU_UUID:?Set R48_GPU_UUID to the authorized idle RTX3090 UUID}"
unset VOYAGE_API_KEY MONGODB_API_KEY
stage="${1:-all}"
root=code/V4/runs/R48_core_diagnosis
mkdir -p "$root"
exec > >(tee -a "$root/orchestration.log") 2>&1
python -c 'import torch; assert torch.cuda.is_available() and "3090" in torch.cuda.get_device_name(0); print("R48 GPU:", torch.cuda.get_device_name(0))'
python -m unittest code.V4.test_r48 -v
if [[ "$stage" == all || "$stage" == audit ]]; then
    python -u -m code.V4.r48_solver_audit --stage all
fi
if [[ "$stage" == all || "$stage" == binary ]]; then
    python -u -m code.V4.r48_binary --stage all
fi
if [[ "$stage" == all || "$stage" == analysis ]]; then
    python -u -m code.V4.r48_analysis
fi
printf '[R48] Stage %s finished %s\n' "$stage" "$(date --iso-8601=seconds)"

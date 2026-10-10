#!/usr/bin/env bash
set -euo pipefail
source /public/home/shiys/miniconda3/etc/profile.d/conda.sh
conda activate easynco
cd /public/home/shiys/0selection2
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export MPLBACKEND=Agg WANDB_MODE=offline
unset VOYAGE_API_KEY MONGODB_API_KEY
stage=${1:-all}
root=code/V4/runs/R49_existing_data_diagnosis
mkdir -p "$root"
if [[ "$stage" == all || "$stage" == train ]]; then
    : "${R49_GPU_UUID:?Bind only the currently idle RTX3090 UUID}"
    export CUDA_VISIBLE_DEVICES="$R49_GPU_UUID"
    used=$(nvidia-smi --id="$R49_GPU_UUID" --query-gpu=memory.used --format=csv,noheader,nounits)
    [[ "$used" -lt 500 ]] || { printf 'Selected GPU is not idle: %s MiB\n' "$used"; exit 1; }
    python -c 'import torch; assert torch.cuda.device_count()==1; assert "3090" in torch.cuda.get_device_name(0); print("R49 GPU:",torch.cuda.get_device_name(0))'
fi
python -m unittest code.V4.test_r49 code.V4.test_r48 -v 2>&1 | tee -a "$root/orchestration.log"
if [[ "$stage" == all || "$stage" == prepare ]]; then
    python -m code.V4.r49_experiment --stage prepare 2>&1 | tee -a "$root/orchestration.log"
fi
if [[ "$stage" == all || "$stage" == train ]]; then
    python -m code.V4.r49_experiment --stage train 2>&1 | tee -a "$root/orchestration.log"
fi
if [[ "$stage" == all || "$stage" == analysis ]]; then
    python -m code.V4.r49_analysis 2>&1 | tee -a "$root/orchestration.log"
fi

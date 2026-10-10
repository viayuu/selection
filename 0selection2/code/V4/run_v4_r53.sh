#!/usr/bin/env bash
set -euo pipefail
source /public/home/shiys/miniconda3/etc/profile.d/conda.sh
conda activate easynco
cd /public/home/shiys/0selection2
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export MPLBACKEND=Agg WANDB_MODE=offline PYTHONUNBUFFERED=1
stage="${1:-all}"
if [[ "$stage" != prepare ]]; then
    export CUDA_VISIBLE_DEVICES="${R53_GPU_UUID:-GPU-cb626bdb-4319-2f5e-b91f-17fcaa44a83d}"
    python -c 'import torch; assert torch.cuda.device_count()==1; assert "3090" in torch.cuda.get_device_name(0); print("R53 authorized device:", torch.cuda.get_device_name(0), flush=True)'
fi
python -m unittest code.V4.test_r53
mkdir -p code/V4/runs/R53_pair_specialist
python -m code.V4.r53_experiment --stage "$stage" 2>&1 | tee -a code/V4/runs/R53_pair_specialist/run.log

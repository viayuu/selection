#!/usr/bin/env bash
set -euo pipefail
cd /public/home/shiys/0selection2
source /public/home/shiys/miniconda3/etc/profile.d/conda.sh
conda activate easynco
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=4
export WANDB_MODE=offline
python -u -m code.V4.r43_experiment "$@"

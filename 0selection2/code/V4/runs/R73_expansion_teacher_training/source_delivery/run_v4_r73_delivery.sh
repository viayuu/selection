#!/usr/bin/env bash
set -euo pipefail
source /public/home/shiys/miniconda3/etc/profile.d/conda.sh
conda activate easynco
cd /public/home/shiys/0selection2
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export WANDB_MODE=offline WANDB_PROJECT=selector WANDB_ENTITY=yjkds-southern-university-of-science-technology
export MPLBACKEND=Agg PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=/public/home/shiys/easynco_v3_bridge/exports/train_100k_20261002/python_compat:/public/home/shiys/0selection2
export CUDA_VISIBLE_DEVICES="" R73_CONDITIONAL_DELIVERY=authorized
[[ $# == 0 ]] || { printf '%s\n' 'R73 conditional delivery takes fixed arguments only.' >&2; exit 2; }
exec 9>code/V4/runs/R73_expansion_teacher_training/delivery_controller.lock
flock -n 9 || { printf '%s\n' 'The original R73 delivery controller already owns this run.' >&2; exit 2; }
exec python -B -u -m code.V4.r73_delivery

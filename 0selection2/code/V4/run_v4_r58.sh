#!/usr/bin/env bash
set -euo pipefail
source /public/home/shiys/miniconda3/etc/profile.d/conda.sh
conda activate easynco
cd /public/home/shiys/0selection2

# One previously idle3090 only; never dispatch through the user's other queues.
export CUDA_VISIBLE_DEVICES="${R58_GPU_UUID:-GPU-cb626bdb-4319-2f5e-b91f-17fcaa44a83d}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONPATH=/public/home/shiys/easynco_v3_bridge/exports/train_100k_20261002/python_compat:/public/home/shiys/0selection2
export WANDB_MODE=offline

if [[ -z "${SLURM_PROCID:-}" ]] && ! nvidia-smi -L >/dev/null 2>&1; then
    exec srun --jobid="${R58_SLURM_JOB_ID:-1465}" --overlap --exact --nodes=1 --ntasks=1 \
        --cpus-per-task=4 --mem=16G bash "$0" "$@"
fi
exec python -u -m code.V4.r58_pipeline "$@"

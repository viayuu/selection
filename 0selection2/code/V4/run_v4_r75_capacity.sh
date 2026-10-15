#!/usr/bin/env bash
set -euo pipefail
source /public/home/shiys/miniconda3/etc/profile.d/conda.sh
conda activate easynco
cd /public/home/shiys/0selection2
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export WANDB_MODE=offline WANDB_PROJECT=selector
export WANDB_ENTITY=yjkds-southern-university-of-science-technology
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg
export PYTHONPATH=/public/home/shiys/easynco_v3_bridge/exports/train_100k_20261002/python_compat:/public/home/shiys/0selection2
stage="${1:-}"
if [[ $# -gt 0 ]]; then shift; fi
case "$stage" in
    prepare)
        export CUDA_VISIBLE_DEVICES=""
        ;;
    measure)
        if [[ "$(hostname -s)" != gpu03 || "${SLURM_JOB_ID:-}" != 1465 ]]; then
            exec srun --jobid=1465 --overlap --exact --nodes=1 --nodelist=gpu03 \
                --ntasks=1 --cpus-per-task=4 --mem=16G \
                bash code/V4/run_v4_r75_capacity.sh measure "$@"
        fi
        export CUDA_VISIBLE_DEVICES=GPU-cb626bdb-4319-2f5e-b91f-17fcaa44a83d
        ;;
    *) printf '%s\n' 'Only prepare/measure are supported. Formal training awaits user batch approval.' >&2; exit 2 ;;
esac
mkdir -p code/V4/runs/R75_large_complete_labels
exec 9>code/V4/runs/R75_large_complete_labels/capacity_${stage}.lock
flock -n 9 || { printf '%s\n' 'This capacity stage is already running.' >&2; exit 2; }
exec python -B -u -m code.V4.r75_capacity "$stage" "$@"

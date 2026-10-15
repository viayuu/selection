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
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
if [[ "${R75_PIPELINE_INNER:-0}" != 1 ]]; then
    exec srun --jobid=1465 --overlap --exact --nodes=1 --nodelist=gpu03 \
        --ntasks=1 --cpus-per-task=4 --mem=19G --export=ALL,R75_PIPELINE_INNER=1 \
        bash code/V4/run_v4_r75_pipeline.sh "$@"
fi
[[ "$(hostname -s)" == gpu03 && "${SLURM_JOB_ID:-}" == 1465 && -n "${SLURM_STEP_ID:-}" ]] \
    || { printf '%s\n' 'The pipeline must run in its allocated GPU step.' >&2; exit 2; }
export CUDA_VISIBLE_DEVICES=GPU-cb626bdb-4319-2f5e-b91f-17fcaa44a83d
exec 9>code/V4/runs/R75_large_complete_labels/formal_launcher.lock
flock -n 9 || { printf '%s\n' 'Another formal R75 process still owns the run.' >&2; exit 2; }
exec python -B -u -m code.V4.r75_pipeline_runner --resume "$@"

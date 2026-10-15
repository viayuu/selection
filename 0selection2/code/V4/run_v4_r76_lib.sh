#!/usr/bin/env bash
set -euo pipefail
source /public/home/shiys/miniconda3/etc/profile.d/conda.sh
conda activate easynco
cd /public/home/shiys/0selection2
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg WANDB_MODE=offline
export PYTHONPATH=/public/home/shiys/easynco_v3_bridge/exports/train_100k_20261002/python_compat:/public/home/shiys/0selection2
ROOT=/public/home/shiys/0selection2/code/V4/runs/R76_R45A_single_reld_mt/lib_evaluation
mkdir -p "$ROOT"
if [[ "${1:-}" == analysis ]]; then
    exec python -B -u -m code.V4.r76_lib --stage analysis --root "$ROOT"
fi
if [[ "${1:-}" == __node ]]; then
    cards=(GPU-71479e7e-f8bd-e430-66ed-f8db9f0d47be GPU-597c682a-4b7f-7609-80a6-8ced791f5c5d)
    datasets=(TSPLIB CVRPLIB)
    children=()
    trap 'for pid in "${children[@]}"; do kill "$pid" 2>/dev/null || true; done' EXIT
    for index in 0 1; do
        CUDA_VISIBLE_DEVICES="${cards[$index]}" python -B -u -m code.V4.r76_lib \
            --stage evaluate --root "$ROOT" --dataset "${datasets[$index]}" --batch-size 8 \
            >"$ROOT/${datasets[$index]}.log" 2>&1 &
        children+=("$!")
    done
    failed=0
    for pid in "${children[@]}"; do wait "$pid" || failed=1; done
    exit "$failed"
fi
exec 9>"$ROOT/launch.lock"
flock -n 9 || { printf '%s\n' 'R76 LIB evaluation is already running.' >&2; exit 2; }
srun --jobid=1475 --overlap --exact --nodes=1 --nodelist=gpu05 --ntasks=1 \
    --cpus-per-task=4 --mem=8G bash code/V4/run_v4_r76_lib.sh __node
python -B -u -m code.V4.r76_lib --stage analysis --root "$ROOT"

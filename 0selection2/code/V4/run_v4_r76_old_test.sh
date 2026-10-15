#!/usr/bin/env bash
set -euo pipefail
source /public/home/shiys/miniconda3/etc/profile.d/conda.sh
conda activate easynco
cd /public/home/shiys/0selection2
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg WANDB_MODE=offline
export PYTHONPATH=/public/home/shiys/easynco_v3_bridge/exports/train_100k_20261002/python_compat:/public/home/shiys/0selection2
ROOT=/public/home/shiys/0selection2/code/V4/runs/R76_R45A_single_reld_mt/old_test_transfer
CACHE="$ROOT/input_cache"

if [[ "${1:-}" == __stage ]]; then
    mkdir -p "$CACHE"
    cp -a --reflink=auto /tmp/shiys_r75_large_complete_labels_cache/test "$CACHE/"
    exit 0
fi

if [[ "${1:-}" == __node ]]; then
    first="$2"
    if (( first == 0 )); then
        cards=(GPU-71479e7e-f8bd-e430-66ed-f8db9f0d47be GPU-597c682a-4b7f-7609-80a6-8ced791f5c5d)
    else
        cards=(GPU-cb24f168-9f18-f0d5-6f4a-6d2406958f50 GPU-cb626bdb-4319-2f5e-b91f-17fcaa44a83d)
    fi
    children=()
    trap 'for pid in "${children[@]}"; do kill "$pid" 2>/dev/null || true; done' EXIT
    for local_rank in 0 1; do
        rank="$((first+local_rank))"
        CUDA_VISIBLE_DEVICES="${cards[$local_rank]}" python -B -u -m code.V4.r76_old_test \
            --stage evaluate --root "$ROOT" --cache-root "$CACHE" --workers 4 --rank "$rank" \
            >"$ROOT/rank${rank}.log" 2>&1 &
        children+=("$!")
    done
    failed=0
    for pid in "${children[@]}"; do wait "$pid" || failed=1; done
    exit "$failed"
fi

mkdir -p "$ROOT"
if [[ "${1:-}" == analysis ]]; then
    exec python -B -u -m code.V4.r76_old_test --stage analysis --root "$ROOT"
fi
exec 9>"$ROOT/launch.lock"
flock -n 9 || { printf '%s\n' 'The old TEST diagnostic is already running.' >&2; exit 2; }
python -B -u -m code.V4.r76_old_test --stage prepare --root "$ROOT"
srun --jobid=1465 --overlap --exact --nodes=1 --nodelist=gpu03 --ntasks=1 \
    --cpus-per-task=2 --mem=4G --gres=none bash code/V4/run_v4_r76_old_test.sh __stage
children=()
trap 'for pid in "${children[@]}"; do kill "$pid" 2>/dev/null || true; done' EXIT
srun --jobid=1475 --overlap --exact --nodes=1 --nodelist=gpu05 --ntasks=1 \
    --cpus-per-task=4 --mem=8G bash code/V4/run_v4_r76_old_test.sh __node 0 \
    >"$ROOT/node4090.log" 2>&1 &
children+=("$!")
srun --jobid=1465 --overlap --exact --nodes=1 --nodelist=gpu03 --ntasks=1 \
    --cpus-per-task=4 --mem=8G bash code/V4/run_v4_r76_old_test.sh __node 2 \
    >"$ROOT/node3090.log" 2>&1 &
children+=("$!")
failed=0
for pid in "${children[@]}"; do wait "$pid" || failed=1; done
if (( failed )); then exit 1; fi
python -B -u -m code.V4.r76_old_test --stage analysis --root "$ROOT"

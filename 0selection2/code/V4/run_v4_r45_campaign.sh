#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source /public/home/shiys/miniconda3/etc/profile.d/conda.sh
conda activate easynco
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export WANDB_MODE=offline MPLBACKEND=Agg
unset VOYAGE_API_KEY

ROOT=code/V4/runs/R45_solver_code_embeddings
exec > >(tee -a "$ROOT/orchestration.log") 2>&1
echo "[R45] Started $(date -Iseconds) on $(hostname)"
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv
nvidia-smi --query-gpu=timestamp,index,uuid,utilization.gpu,memory.used,power.draw \
    --format=csv -l 10 > "$ROOT/gpu_metrics.csv" &
monitor=$!
trap 'kill "$monitor" 2>/dev/null || true' EXIT
python -m unittest code.V4.test_r45 code.V4.test_solver_code_atlas code.V4.test_r43 -q
CUDA_VISIBLE_DEVICES=0 bash code/V4/run_v4_r45.sh --stage prepare --device cuda:0

run_arm() {
    local arm=$1 gpu=$2
    echo "[R45] Launching $arm on GPU $gpu at $(date -Iseconds)"
    CUDA_VISIBLE_DEVICES=$gpu bash code/V4/run_v4_r45.sh --stage train --group "$arm" --device cuda:0 \
        > "$ROOT/${arm}.launcher.log" 2>&1
}

# B/C use the same initialization and dropout seed, in independent processes.
run_arm B 0 &
first=$!
run_arm C 1 &
second=$!
trap 'kill "$first" "$second" 2>/dev/null || true' INT TERM
while kill -0 "$first" 2>/dev/null && kill -0 "$second" 2>/dev/null; do
    sleep 10
done
if ! kill -0 "$first" 2>/dev/null; then
    wait "$first"
    run_arm A 0 &
    first=$!
else
    wait "$second"
    run_arm A 1 &
    second=$!
fi
wait "$first"
wait "$second"
trap - INT TERM

CUDA_VISIBLE_DEVICES=0 bash code/V4/run_v4_r45.sh --stage test --device cuda:0
bash code/V4/run_v4_r45.sh --stage analysis --device cpu
echo "[R45] All three arms, locked-checkpoint test, and analysis completed $(date -Iseconds)"

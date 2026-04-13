#!/usr/bin/env bash

set -euo pipefail

ROOT="/public/home/zhoucl/shiys"
cd "$ROOT"

PYTHON_BIN="/public/home/zhoucl/anaconda3/envs/easynco_zhoucl/bin/python"
SCRIPT="2cmab2/run_offline.py"
OUTPUT_ROOT="$ROOT/2cmab2/outputs/tuning_2026-04-01_stage2_5k"
MAX_PARALLEL=2

mkdir -p "$OUTPUT_ROOT"
export PYTHONUNBUFFERED=1

COMMON_ARGS=(
  --method neural_linucb
  --epochs 1
  --seed 0
  --hidden-dim 64
  --reg 1.0
  --lr 1e-3
  --max-samples-per-problem 5000
  --eval-every-steps 500
  --eval-log-every 500
  --checkpoint-every-steps 1000
  --checkpoint-every 0
  --disable-progress
  --no-quiet
)

wait_for_slot() {
  while true; do
    local running
    running=$(jobs -pr | wc -l | tr -d ' ')
    if [ "$running" -lt "$MAX_PARALLEL" ]; then
      break
    fi
    wait -n
  done
}

run_exp_bg() {
  local name="$1"
  shift

  local out_dir="$OUTPUT_ROOT/$name"

  (
    echo "======================================================================"
    echo "[$(date '+%F %T')] START $name"
    echo "out_dir=$out_dir"
    echo "extra_args=$*"
    echo "======================================================================"

    "$PYTHON_BIN" "$SCRIPT" \
      "${COMMON_ARGS[@]}" \
      --output-dir "$out_dir" \
      "$@"

    echo "[$(date '+%F %T')] END   $name"
    echo
  ) &
}

# 注意：
# - baseline 当前已经在单独的进程里运行
# - 这里只跑“剩余 7 组”，避免和 baseline 输出目录冲突

wait_for_slot
run_exp_bg "s2_01_less_rep_fit" \
  --reward-mode linear_zero_one \
  --alpha 1.0 \
  --train-every 200 \
  --representation-steps 2 \
  --representation-buffer-size 2000 \
  --representation-batch-size 256 \
  --linear-head-buffer-size 10000 \
  --initial-pulls 5 \
  --no-freeze-encoder

wait_for_slot
run_exp_bg "s2_02_more_explore" \
  --reward-mode linear_zero_one \
  --alpha 2.0 \
  --train-every 200 \
  --representation-steps 2 \
  --representation-buffer-size 2000 \
  --representation-batch-size 256 \
  --linear-head-buffer-size 10000 \
  --initial-pulls 5 \
  --no-freeze-encoder

wait_for_slot
run_exp_bg "s2_03_less_explore" \
  --reward-mode linear_zero_one \
  --alpha 0.5 \
  --train-every 200 \
  --representation-steps 2 \
  --representation-buffer-size 2000 \
  --representation-batch-size 256 \
  --linear-head-buffer-size 10000 \
  --initial-pulls 5 \
  --no-freeze-encoder

wait_for_slot
run_exp_bg "s2_04_full_linear_history" \
  --reward-mode linear_zero_one \
  --alpha 2.0 \
  --train-every 200 \
  --representation-steps 2 \
  --representation-buffer-size 2000 \
  --representation-batch-size 256 \
  --linear-head-buffer-size 0 \
  --initial-pulls 5 \
  --no-freeze-encoder

wait_for_slot
run_exp_bg "s2_05_autosaea_reward" \
  --reward-mode autosaea \
  --alpha 2.0 \
  --train-every 200 \
  --representation-steps 2 \
  --representation-buffer-size 2000 \
  --representation-batch-size 256 \
  --linear-head-buffer-size 10000 \
  --initial-pulls 5 \
  --no-freeze-encoder

wait_for_slot
run_exp_bg "s2_06_freeze_graph_encoder" \
  --reward-mode linear_zero_one \
  --alpha 2.0 \
  --train-every 200 \
  --representation-steps 2 \
  --representation-buffer-size 2000 \
  --representation-batch-size 256 \
  --linear-head-buffer-size 10000 \
  --initial-pulls 5 \
  --freeze-encoder

wait_for_slot
run_exp_bg "s2_07_low_lr" \
  --reward-mode linear_zero_one \
  --alpha 2.0 \
  --lr 3e-4 \
  --train-every 200 \
  --representation-steps 2 \
  --representation-buffer-size 2000 \
  --representation-batch-size 256 \
  --linear-head-buffer-size 10000 \
  --initial-pulls 5 \
  --no-freeze-encoder

wait
echo "[$(date '+%F %T')] ALL DONE"

#!/usr/bin/env bash

set -euo pipefail

ROOT="/public/home/zhoucl/shiys"
cd "$ROOT"

PYTHON_BIN="/public/home/zhoucl/anaconda3/envs/easynco_zhoucl/bin/python"
SCRIPT="2cmab2/run_offline.py"
OUTPUT_ROOT="$ROOT/2cmab2/outputs/tuning_2026-04-01_stage1"

mkdir -p "$OUTPUT_ROOT"

export PYTHONUNBUFFERED=1

# 这一批先做“小规模筛选”：
# - TSP 1000 + CVRP 1000
# - seed=0
# - 每 500 step 做一次 val
# - 每 500 step 保存一次 checkpoint
COMMON_ARGS=(
  --method neural_linucb
  --epochs 1
  --seed 0
  --hidden-dim 64
  --reg 1.0
  --lr 1e-3
  --max-samples-per-problem 1000
  --eval-every-steps 500
  --eval-log-every 200
  --checkpoint-every-steps 500
  --checkpoint-every 0
  --disable-progress
  --no-quiet
)

run_exp() {
  local name="$1"
  shift

  local out_dir="$OUTPUT_ROOT/$name"

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
}

# 0. 当前默认配置基线
run_exp "s1_00_baseline_default" \
  --reward-mode linear_zero_one \
  --alpha 1.0 \
  --train-every 100 \
  --representation-steps 10 \
  --representation-buffer-size 5000 \
  --representation-batch-size 512 \
  --linear-head-buffer-size 10000 \
  --initial-pulls 2 \
  --no-freeze-encoder

# 1. 降低表示层更新频率与强度，减少 recent buffer 被反复拟合
run_exp "s1_01_less_rep_fit" \
  --reward-mode linear_zero_one \
  --alpha 1.0 \
  --train-every 200 \
  --representation-steps 2 \
  --representation-buffer-size 2000 \
  --representation-batch-size 256 \
  --linear-head-buffer-size 10000 \
  --initial-pulls 5 \
  --no-freeze-encoder

# 2. 在更稳的表示层设置上，增强探索
run_exp "s1_02_more_explore" \
  --reward-mode linear_zero_one \
  --alpha 2.0 \
  --train-every 200 \
  --representation-steps 2 \
  --representation-buffer-size 2000 \
  --representation-batch-size 256 \
  --linear-head-buffer-size 10000 \
  --initial-pulls 5 \
  --no-freeze-encoder

# 3. 对照：更弱探索
run_exp "s1_03_less_explore" \
  --reward-mode linear_zero_one \
  --alpha 0.5 \
  --train-every 200 \
  --representation-steps 2 \
  --representation-buffer-size 2000 \
  --representation-batch-size 256 \
  --linear-head-buffer-size 10000 \
  --initial-pulls 5 \
  --no-freeze-encoder

# 4. 线性头保留全历史，看看是不是线性头窗口太短导致信息损失
run_exp "s1_04_full_linear_history" \
  --reward-mode linear_zero_one \
  --alpha 2.0 \
  --train-every 200 \
  --representation-steps 2 \
  --representation-buffer-size 2000 \
  --representation-batch-size 256 \
  --linear-head-buffer-size 0 \
  --initial-pulls 5 \
  --no-freeze-encoder

# 5. 改 reward 口径
run_exp "s1_05_autosaea_reward" \
  --reward-mode autosaea \
  --alpha 2.0 \
  --train-every 200 \
  --representation-steps 2 \
  --representation-buffer-size 2000 \
  --representation-batch-size 256 \
  --linear-head-buffer-size 10000 \
  --initial-pulls 5 \
  --no-freeze-encoder

# 6. 冻结图编码器，只训练上层表示与线性头
run_exp "s1_06_freeze_graph_encoder" \
  --reward-mode linear_zero_one \
  --alpha 2.0 \
  --train-every 200 \
  --representation-steps 2 \
  --representation-buffer-size 2000 \
  --representation-batch-size 256 \
  --linear-head-buffer-size 10000 \
  --initial-pulls 5 \
  --freeze-encoder

# 7. 更小学习率，检查是不是表示层更新过猛
run_exp "s1_07_low_lr" \
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

echo "[$(date '+%F %T')] ALL DONE"

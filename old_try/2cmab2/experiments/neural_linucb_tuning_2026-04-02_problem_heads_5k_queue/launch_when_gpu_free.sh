#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Neural-LinUCB 5k 调参队列
# ============================================================
# 目标：
# 1. 不打断用户当前已经在跑的实验
# 2. 只有当 GPU 明显空出来时，才自动开始下一组实验
# 3. 串行执行多组参数，避免多个新实验彼此抢 GPU
#
# 说明：
# - 这个脚本建议放到 tmux 里跑
# - 每个实验真正启动前，都会再检查一次 GPU 是否空闲
# - 当前重点验证：problem-specific heads 下，降低探索强度能否改善泛化
# ============================================================

PROJECT_ROOT="/public/home/zhoucl/shiys"
PYTHON_BIN="/public/home/zhoucl/anaconda3/envs/easynco_zhoucl/bin/python"
SCRIPT_PATH="$PROJECT_ROOT/2cmab2/run_offline.py"
OUTPUT_ROOT="$PROJECT_ROOT/2cmab2/outputs/tuning_2026-04-02_problem_heads_5k_queue"

# 等待 GPU 的判据可以按需通过环境变量覆盖。
# 默认策略比较保守：
# - GPU 利用率 <= 35%
# - 空闲显存 >= 12000MB
# - 连续满足 3 次才真正开跑
# 这样可以尽量避免刚好和别的实验抢到一起。
WAIT_UTIL_THRESHOLD="${WAIT_UTIL_THRESHOLD:-35}"
WAIT_FREE_MEM_MB="${WAIT_FREE_MEM_MB:-12000}"
WAIT_CONSECUTIVE_PASSES="${WAIT_CONSECUTIVE_PASSES:-3}"
WAIT_INTERVAL_SEC="${WAIT_INTERVAL_SEC:-120}"

mkdir -p "$OUTPUT_ROOT"

timestamp() {
  date '+%F %T'
}

log() {
  echo "[$(timestamp)] $*"
}

gpu_snapshot() {
  local util mem_used mem_total free_mem
  util="$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -n 1 | tr -d ' ')"
  mem_used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -n 1 | tr -d ' ')"
  mem_total="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n 1 | tr -d ' ')"
  free_mem=$((mem_total - mem_used))
  echo "$util $mem_used $mem_total $free_mem"
}

gpu_is_ready() {
  local util mem_used mem_total free_mem
  read -r util mem_used mem_total free_mem < <(gpu_snapshot)
  if [[ "$util" -le "$WAIT_UTIL_THRESHOLD" && "$free_mem" -ge "$WAIT_FREE_MEM_MB" ]]; then
    return 0
  fi
  return 1
}

wait_for_gpu() {
  local ok_count=0
  while true; do
    local util mem_used mem_total free_mem
    read -r util mem_used mem_total free_mem < <(gpu_snapshot)
    log "GPU 检查: util=${util}% | used=${mem_used}MB/${mem_total}MB | free=${free_mem}MB | 连续空闲次数=${ok_count}/${WAIT_CONSECUTIVE_PASSES}"
    if gpu_is_ready; then
      ok_count=$((ok_count + 1))
      if [[ "$ok_count" -ge "$WAIT_CONSECUTIVE_PASSES" ]]; then
        log "GPU 已满足启动条件，开始后续实验。"
        break
      fi
    else
      ok_count=0
    fi
    sleep "$WAIT_INTERVAL_SEC"
  done
}

COMMON_ARGS=(
  --method neural_linucb
  --epochs 1
  --seed 0
  --max-samples-per-problem 5000
  --eval-every-steps 500
  --eval-log-every 500
  --checkpoint-every 0
  --checkpoint-every-steps 500
  --disable-progress
  --no-quiet
  --representation-balance-mode problem_arm
  --representation-balance-power 0.5
  --representation-loss-reweight
  --train-every 100
  --representation-steps 10
  --representation-buffer-size 5000
  --representation-batch-size 512
  --linear-head-buffer-size 10000
  --initial-pulls 2
  --no-freeze-encoder
  --problem-specific-heads
)

run_exp() {
  local name="$1"
  shift

  local out_dir="$OUTPUT_ROOT/$name"
  mkdir -p "$out_dir"

  wait_for_gpu

  log "======================================================================"
  log "START $name"
  log "out_dir=$out_dir"
  log "extra_args=$*"
  log "======================================================================"

  "$PYTHON_BIN" "$SCRIPT_PATH" \
    "${COMMON_ARGS[@]}" \
    --output-dir "$out_dir" \
    "$@"

  log "END   $name"
  log
}

# ------------------------------------------------------------
# 实验设计
# ------------------------------------------------------------
# 这三组都是在最新“problem-specific heads + anti-collapse”实现上继续细调：
#
# 1. a05_e005
#    - alpha 从 1.0 降到 0.5
#    - train_epsilon 从 0.10 降到 0.05
#    - 目标：先做温和降探索
#
# 2. a03_e005
#    - 再进一步降低 UCB 强度
#    - 保留一点 epsilon 探索，避免太快贪心锁死
#
# 3. a03_e002
#    - 继续压低随机探索地板
#    - 用来判断 problem-heads 下是不是 epsilon 也偏大
# ------------------------------------------------------------

run_exp "ph5k_a05_e005" \
  --alpha 0.5 \
  --train-epsilon 0.05

run_exp "ph5k_a03_e005" \
  --alpha 0.3 \
  --train-epsilon 0.05

run_exp "ph5k_a03_e002" \
  --alpha 0.3 \
  --train-epsilon 0.02

log "ALL DONE"

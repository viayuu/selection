#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/public/home/zhoucl/shiys"
SCRIPT_DIR="$PROJECT_ROOT/2cmab2/experiments/neural_linucb_tuning_2026-04-02_problem_heads_5k_queue"
LAUNCH_SCRIPT="$SCRIPT_DIR/launch_when_gpu_free.sh"
SESSION_NAME="${1:-nlin_ph5k_queue_0402}"
LOG_PATH="$SCRIPT_DIR/${SESSION_NAME}.log"

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  echo "tmux session 已存在: $SESSION_NAME"
  echo "可直接查看: tmux attach -t $SESSION_NAME"
  exit 1
fi

tmux new-session -d -s "$SESSION_NAME" \
  "cd '$PROJECT_ROOT' && /bin/bash '$LAUNCH_SCRIPT' 2>&1 | tee '$LOG_PATH'"

echo "tmux session 已启动: $SESSION_NAME"
echo "日志文件: $LOG_PATH"
echo "查看实时输出: tmux attach -t $SESSION_NAME"

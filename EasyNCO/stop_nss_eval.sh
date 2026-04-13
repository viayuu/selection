#!/bin/bash
set -euo pipefail

STATE_FILE="/public/home/zhoucl/shiys/EasyNCO/results/nss_eval/停止状态.json"

controller_pids=$(pgrep -f "/public/home/zhoucl/shiys/EasyNCO/launch_nss_eval.sh|python run_nss_eval_suite.py --skip-lib-unstarted --exclude-methods difusco,t2t" || true)
worker_pids=$(pgrep -f "eval.py mode=test" || true)

if [ -z "${controller_pids}${worker_pids}" ]; then
  echo "没有找到需要停止的 NSS 评测进程。"
  exit 0
fi

for pid in $worker_pids $controller_pids; do
  kill -CONT "$pid" 2>/dev/null || true
done

sleep 1

for pid in $worker_pids $controller_pids; do
  kill -TERM "$pid" 2>/dev/null || true
done

sleep 2

for pid in $worker_pids $controller_pids; do
  if kill -0 "$pid" 2>/dev/null; then
    kill -KILL "$pid" 2>/dev/null || true
  fi
done

python /public/home/zhoucl/shiys/EasyNCO/scripts_write_stop_state.py "$STATE_FILE" "$controller_pids" "$worker_pids"
echo "已停止 NSS 评测进程。状态文件: $STATE_FILE"

#!/bin/bash
set -euo pipefail

STATE_FILE="/public/home/zhoucl/shiys/EasyNCO/results/nss_eval/暂停状态.json"

controller_pids=$(pgrep -f "/public/home/zhoucl/shiys/EasyNCO/launch_nss_eval.sh|python run_nss_eval_suite.py --skip-lib-unstarted --exclude-methods difusco,t2t" || true)
worker_pids=$(pgrep -f "eval.py mode=test" || true)

if [ -z "${controller_pids}${worker_pids}" ]; then
  echo "没有找到需要暂停的 NSS 评测进程。"
  exit 0
fi

for pid in $controller_pids $worker_pids; do
  kill -STOP "$pid"
done

python /public/home/zhoucl/shiys/EasyNCO/scripts_write_pause_state.py "$STATE_FILE" "$controller_pids" "$worker_pids"
echo "已暂停 NSS 评测进程。状态文件: $STATE_FILE"

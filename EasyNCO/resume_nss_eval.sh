#!/bin/bash
set -euo pipefail

PAUSE_STATE_FILE="/public/home/zhoucl/shiys/EasyNCO/results/nss_eval/暂停状态.json"
STOP_STATE_FILE="/public/home/zhoucl/shiys/EasyNCO/results/nss_eval/停止状态.json"

if [ -f "$STOP_STATE_FILE" ]; then
  setsid /bin/bash /public/home/zhoucl/shiys/EasyNCO/launch_nss_eval.sh < /dev/null > /public/home/zhoucl/shiys/EasyNCO/results/nss_eval/总控日志.log 2>&1 &
  echo "已按停止状态重启 NSS 评测。新总控 PID: $!"
  exit 0
fi

if [ ! -f "$PAUSE_STATE_FILE" ]; then
  echo "未找到暂停状态文件: $PAUSE_STATE_FILE"
  exit 1
fi

python /public/home/zhoucl/shiys/EasyNCO/scripts_resume_from_pause_state.py "$PAUSE_STATE_FILE"
echo "已尝试恢复 NSS 评测进程。"

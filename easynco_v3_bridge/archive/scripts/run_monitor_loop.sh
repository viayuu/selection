#!/usr/bin/env bash
set -euo pipefail

ROOT="/public/home/zhoucl/shiys/easynco_v3_bridge"
CONDA_SH="/public/home/zhoucl/anaconda3/etc/profile.d/conda.sh"
ENV_NAME="easynco_zhoucl"
PYTHONPATH_ROOT="/public/home/zhoucl/shiys"

source "$CONDA_SH"
conda activate "$ENV_NAME"
export PYTHONPATH="$PYTHONPATH_ROOT"
unset NPM_CONFIG_PREFIX || true

while true; do
  echo "[monitor] $(date '+%F %T') refreshing snapshot"
  python "$ROOT/scripts/monitor_v3.py" | tee "$ROOT/logs/v3_monitor_last.json"
  sleep 300
done

#!/usr/bin/env bash

set -euo pipefail

ROOT="/public/home/zhoucl/shiys"
RUN_SCRIPT="$ROOT/2cmab2/experiments/neural_linucb_tuning_2026-04-01_stage2_5k/launch_stage2.sh"
LOG_FILE="$ROOT/2cmab2/experiments/neural_linucb_tuning_2026-04-01_stage2_5k/setsid_stage2.log"

mkdir -p "$(dirname "$LOG_FILE")"

echo "using setsid detached launch"
echo "run_script=$RUN_SCRIPT"
echo "log_file=$LOG_FILE"

setsid /bin/bash "$RUN_SCRIPT" < /dev/null > "$LOG_FILE" 2>&1 &
echo $!

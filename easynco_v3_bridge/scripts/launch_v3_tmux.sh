#!/usr/bin/env bash
set -euo pipefail

ROOT="/public/home/zhoucl/shiys/easynco_v3_bridge"
CONDA_SH="/public/home/zhoucl/anaconda3/etc/profile.d/conda.sh"
ENV_NAME="easynco_zhoucl"
PYTHONPATH_ROOT="/public/home/zhoucl/shiys"

start_session() {
  local session_name="$1"
  local command="$2"
  local log_file="$ROOT/logs/${session_name}.tmux.log"

  if tmux has-session -t "$session_name" 2>/dev/null; then
    echo "tmux session already exists: $session_name"
    return
  fi

  tmux new-session -d -s "$session_name" \
    "bash -lc 'source \"$CONDA_SH\" && conda activate \"$ENV_NAME\" && export PYTHONPATH=\"$PYTHONPATH_ROOT\" && $command |& tee \"$log_file\"'"
  echo "started tmux session: $session_name"
}

case "${1:-}" in
  prepare-data)
    start_session \
      "v3_prepare_data" \
      "python \"$ROOT/scripts/prepare_v3_checkpoints.py\" && python \"$ROOT/scripts/generate_v3_datasets.py\" all"
    ;;
  nss-labels)
    start_session \
      "v3_nss_labels" \
      "python \"$ROOT/scripts/build_label_jobs.py\" && python \"$ROOT/scripts/run_label_queue.py\" --groups nss"
    ;;
  atsp-labels)
    start_session \
      "v3_atsp_labels" \
      "python \"$ROOT/scripts/build_label_jobs.py\" && python \"$ROOT/scripts/run_label_queue.py\" --groups atsp"
    ;;
  mvrp-labels)
    start_session \
      "v3_mvrp_labels" \
      "python \"$ROOT/scripts/prepare_v3_checkpoints.py\" && python \"$ROOT/scripts/build_label_jobs.py\" && python \"$ROOT/scripts/run_label_queue.py\" --groups mvrp"
    ;;
  deploy-now)
    start_session \
      "v3_prepare_data" \
      "python \"$ROOT/scripts/prepare_v3_checkpoints.py\" && python \"$ROOT/scripts/generate_v3_datasets.py\" all"
    start_session \
      "v3_nss_labels" \
      "python \"$ROOT/scripts/build_label_jobs.py\" && python \"$ROOT/scripts/run_label_queue.py\" --groups nss"
    ;;
  *)
    cat <<'EOF'
Usage:
  launch_v3_tmux.sh prepare-data
  launch_v3_tmux.sh nss-labels
  launch_v3_tmux.sh atsp-labels
  launch_v3_tmux.sh mvrp-labels
  launch_v3_tmux.sh deploy-now

Notes:
  - deploy-now starts two background sessions:
    1. dataset/checkpoint preparation
    2. TSP/CVRP NSS label jobs
  - ATSP and MVRP label queues should be started after their datasets are ready.
EOF
    exit 1
    ;;
esac

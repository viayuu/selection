#!/usr/bin/env bash
set -euo pipefail
source /public/home/shiys/miniconda3/etc/profile.d/conda.sh
conda activate easynco
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
root="$repo/code/V4/runs/R78_zero_selection_goal/round004"
session=R78_round004_cpu_head
export CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 WANDB_MODE=offline
case "${1:-start}" in
  start)
    if tmux has-session -t "$session" 2>/dev/null; then
      printf 'Existing R78 round004 session; refusing duplicate training.\n' >&2
      exit 1
    fi
    mkdir -p "$root"
    tmux new-session -d -s "$session" -c "$repo" \
      "source /public/home/shiys/miniconda3/etc/profile.d/conda.sh && conda activate easynco && exec env CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 WANDB_MODE=offline python -u -m code.V4.r78_frozen_score_cost_head --root '$root' > '$root/train.log' 2>&1"
    printf 'tmux=%s\nlog=%s/train.log\n' "$session" "$root"
    ;;
  run)
    cd "$repo"
    exec python -u -m code.V4.r78_frozen_score_cost_head --root "$root"
    ;;
  status)
    cat "$root/progress.json"
    ;;
  *)
    printf 'Usage: %s [start|run|status]\n' "$0" >&2
    exit 2
    ;;
esac

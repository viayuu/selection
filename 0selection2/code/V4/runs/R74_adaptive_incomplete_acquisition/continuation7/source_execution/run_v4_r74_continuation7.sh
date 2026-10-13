#!/usr/bin/env bash
set -euo pipefail
source /public/home/shiys/miniconda3/etc/profile.d/conda.sh
conda activate easynco
cd /public/home/shiys/0selection2
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export WANDB_MODE=offline WANDB_PROJECT=selector WANDB_ENTITY=yjkds-southern-university-of-science-technology
export MPLBACKEND=Agg PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=/public/home/shiys/easynco_v3_bridge/exports/train_100k_20261002/python_compat:/public/home/shiys/0selection2
[[ $# == 0 ]] || { printf '%s\n' 'Continuation7 has a fixed cache-only recipe.' >&2; exit 2; }
if [[ -z "${R74_TMUX_PANE_ID:-}" ]]; then
  [[ -n "${TMUX_PANE:-}" && "$(tmux display-message -p -t "$TMUX_PANE" '#{session_name}')" == r74_continuation7 ]] || {
    printf '%s\n' 'Use only owned r74_continuation7.' >&2; exit 2;
  }
  export R74_TMUX_PANE_ID="$TMUX_PANE"
  export R74_TMUX_PANE_PID="$(tmux display-message -p -t "$TMUX_PANE" '#{pane_pid}')"
  export R74_TMUX_SERVER_HOST="$(hostname -s)"
fi
if [[ "$(hostname -s)" != gpu03 || "${SLURM_JOB_ID:-}" != 1465 ]]; then
  exec srun --jobid=1465 --overlap --exact --nodes=1 --nodelist=gpu03 \
    --ntasks=1 --cpus-per-task=4 --mem=16G bash code/V4/run_v4_r74_continuation7.sh
fi
export CUDA_VISIBLE_DEVICES=GPU-cb626bdb-4319-2f5e-b91f-17fcaa44a83d
exec 9>code/V4/runs/R74_adaptive_incomplete_acquisition/continuation7/launcher.lock
flock -n 9 || { printf '%s\n' 'Continuation7 already owned; no automatic retry.' >&2; exit 2; }
exec python -B -u -m code.V4.r74_train_only_continuation

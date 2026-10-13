#!/usr/bin/env bash
set -euo pipefail
source /public/home/shiys/miniconda3/etc/profile.d/conda.sh
conda activate easynco
cd /public/home/shiys/0selection2
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export WANDB_MODE=offline WANDB_PROJECT=selector WANDB_ENTITY=yjkds-southern-university-of-science-technology
export MPLBACKEND=Agg PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=/public/home/shiys/easynco_v3_bridge/exports/train_100k_20261002/python_compat:/public/home/shiys/0selection2
stage="${1:-validate}"
if [[ $# -gt 0 ]]; then shift; fi
case "$stage" in
    cpu-check) export CUDA_VISIBLE_DEVICES="" ;;
    validate)
        if [[ -z "${R72_TMUX_PANE_ID:-}" ]]; then
            [[ -n "${TMUX_PANE:-}" ]] || { printf '%s\n' 'R72 must run in owned tmux r72_validate.' >&2; exit 2; }
            [[ "$(tmux display-message -p -t "$TMUX_PANE" '#{session_name}')" == r72_validate ]] || exit 2
            export R72_TMUX_PANE_ID="$TMUX_PANE"
            export R72_TMUX_PANE_PID="$(tmux display-message -p -t "$TMUX_PANE" '#{pane_pid}')"
            export R72_TMUX_SERVER_HOST="$(hostname -s)"
        fi
        if [[ "$(hostname -s)" != gpu03 || "${SLURM_JOB_ID:-}" != 1465 ]]; then
            exec srun --jobid=1465 --overlap --exact --nodes=1 --nodelist=gpu03 \
                --ntasks=1 --cpus-per-task=4 --mem=16G \
                bash code/V4/run_v4_r72.sh "$stage" "$@"
        fi
        export CUDA_VISIBLE_DEVICES=GPU-cb626bdb-4319-2f5e-b91f-17fcaa44a83d
        exec 9>code/V4/runs/R72_heterogeneous_consensus/launcher.lock
        flock -n 9 || { printf '%s\n' 'An R72 launcher already owns this run.' >&2; exit 2; }
        ;;
    *) printf 'Unknown R72 stage: %s\n' "$stage" >&2; exit 2 ;;
esac
exec python -B -u -m code.V4.r72_experiment --stage "$stage" "$@"

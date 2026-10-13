#!/usr/bin/env bash
set -euo pipefail
source /public/home/shiys/miniconda3/etc/profile.d/conda.sh
conda activate easynco
cd /public/home/shiys/0selection2
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export WANDB_MODE=offline WANDB_PROJECT=selector WANDB_ENTITY=yjkds-southern-university-of-science-technology
export MPLBACKEND=Agg PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=/public/home/shiys/easynco_v3_bridge/exports/train_100k_20261002/python_compat:/public/home/shiys/0selection2
[[ $# == 0 ]] || { printf '%s\n' 'R71 TEST takes fixed arguments only.' >&2; exit 2; }
root=code/V4/runs/R71_joint_performance_mixture
[[ -f "$root/delivery_started.json" && -f "$root/final_test_lock.json" && -f "$root/training_terminal_receipt.json" && ! -e "$root/final_test_started.json" ]] || {
    printf '%s\n' 'Require actual original TRAIN terminal0, authorized controller and exclusive VAL lock; no TEST repeat.' >&2
    exit 2
}
if [[ -z "${R71_TEST_TMUX_PANE_ID:-}" ]]; then
    [[ -n "${TMUX_PANE:-}" && "$(tmux display-message -p -t "$TMUX_PANE" '#{session_name}')" == r71_final_test ]] || {
        printf '%s\n' 'Use only owned tmux r71_final_test.' >&2
        exit 2
    }
    export R71_TEST_TMUX_PANE_ID="$TMUX_PANE"
    export R71_TEST_TMUX_PANE_PID="$(tmux display-message -p -t "$TMUX_PANE" '#{pane_pid}')"
    export R71_TEST_TMUX_SERVER_HOST="$(hostname -s)"
fi
export R71_TMUX_SESSION=r71_final_test R71_PARENT_GPU_BATON=locked-test
if [[ "$(hostname -s)" != gpu03 || "${SLURM_JOB_ID:-}" != 1465 ]]; then
    exec srun --jobid=1465 --overlap --exact --nodes=1 --nodelist=gpu03 \
        --ntasks=1 --cpus-per-task=4 --mem=16G bash code/V4/run_v4_r71_test.sh
fi
export CUDA_VISIBLE_DEVICES=GPU-cb626bdb-4319-2f5e-b91f-17fcaa44a83d
exec 9>"$root/test_launcher.lock"
flock -n 9 || { printf '%s\n' 'An R71 TEST launcher already owns this run.' >&2; exit 2; }
exec python -B -u -m code.V4.r71_frozen_evaluation --stage test

#!/usr/bin/env bash
set -euo pipefail
source /public/home/shiys/miniconda3/etc/profile.d/conda.sh
conda activate easynco
cd /public/home/shiys/0selection2
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export WANDB_MODE=offline WANDB_PROJECT=selector
export WANDB_ENTITY=yjkds-southern-university-of-science-technology
export MPLBACKEND=Agg PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=/public/home/shiys/easynco_v3_bridge/exports/train_100k_20261002/python_compat:/public/home/shiys/0selection2
[[ $# == 0 ]] || {
    printf '%s\n' 'R70 TEST requires fixed arguments.' >&2
    exit 2
}
root=code/V4/runs/R70_latent_mixup
if [[ "${R70_PARENT_GPU_BATON:-}" != locked-test ]]; then
    [[ -f "$root/delivery_started.json" ]] || {
        printf '%s\n' 'R70 TEST requires the authorized conditional CPU delivery marker.' >&2
        exit 2
    }
    export R70_PARENT_GPU_BATON=locked-test
fi
[[ -f "$root/final_test_lock.json" && -f "$root/training_terminal_receipt.json" && ! -e "$root/final_test_started.json" ]] || {
    printf '%s\n' 'Require original TRAIN1465.103 terminal0 and exclusive VAL lock; never repeat TEST.' >&2
    exit 2
}
if [[ -z "${R70_TEST_TMUX_PANE_ID:-}" ]]; then
    [[ -n "${TMUX_PANE:-}" && "$(tmux display-message -p -t "$TMUX_PANE" '#{session_name}')" == r70_final_test ]] || {
        printf '%s\n' 'Use only owned tmux r70_final_test.' >&2
        exit 2
    }
    export R70_TEST_TMUX_PANE_ID="$TMUX_PANE"
    export R70_TEST_TMUX_PANE_PID="$(tmux display-message -p -t "$TMUX_PANE" '#{pane_pid}')"
    export R70_TEST_TMUX_SERVER_HOST="$(hostname -s)"
fi
export R70_TMUX_SESSION=r70_final_test
if [[ "$(hostname -s)" != gpu03 || "${SLURM_JOB_ID:-}" != 1465 ]]; then
    exec srun --jobid=1465 --overlap --exact --nodes=1 --nodelist=gpu03 \
        --ntasks=1 --cpus-per-task=4 --mem=16G bash code/V4/run_v4_r70_test.sh
fi
export CUDA_VISIBLE_DEVICES=GPU-cb626bdb-4319-2f5e-b91f-17fcaa44a83d
exec 9>"$root/test_launcher.lock"
flock -n 9 || { printf '%s\n' 'An R70 TEST launcher already owns this run.' >&2; exit 2; }
exec python -B -u -m code.V4.r70_frozen_evaluation --stage test

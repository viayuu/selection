#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source /public/home/shiys/miniconda3/etc/profile.d/conda.sh
conda activate easynco
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1 WANDB_MODE=offline
root="${R40_ROOT:-code/V4/runs/R40_discriminative_training}"
python -m code.V4.r40_experiment --root "$root" --prepare-only
python -m unittest code.V4.test_r40 -q 2>&1 | tee "$root/unit_tests.log"
python -m code.V4.r40_watch --root "$root" > "$root/watch.log" 2>&1 &
watch_pid=$!
python -m code.V4.r40_experiment --root "$root" --group A --wandb > "$root/launch_A.log" 2>&1 &
pid_a=$!
python -m code.V4.r40_experiment --root "$root" --group B --wandb > "$root/launch_B.log" 2>&1 &
pid_b=$!
status_a=0
status_b=0
wait "$pid_a" || status_a=$?
wait "$pid_b" || status_b=$?
status=0
if (( status_a != 0 || status_b != 0 )); then
    status=1
else
    python -m code.V4.r40_analysis --root "$root" || status=$?
    if (( status == 0 )); then
        python -m code.V4.r40_verify --root "$root" || status=$?
    fi
fi
printf '%s\n' "$status" > "$root/queue.exit_code"
wait "$watch_pid"
exit "$status"

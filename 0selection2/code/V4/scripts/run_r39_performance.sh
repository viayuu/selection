#!/usr/bin/env bash
set -o pipefail
source /public/home/shiys/miniconda3/etc/profile.d/conda.sh
conda activate easynco
cd /public/home/shiys/0selection2
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export WANDB_MODE=offline PYTHONUNBUFFERED=1
root=code/V4/runs/R39_performance_model
printf '%s\n' "$BASHPID" > "$root/queue.shell_pid"
python -u -m code.V4.performance_experiment --root "$root" --epochs 60 --batch-size 640 --seeds "${R39_SEEDS:-2}" --wandb 2>&1 | tee "$root/queue.log"
status=${PIPESTATUS[0]}
printf '%s\n' "$status" > "$root/queue.exit_code"
exit "$status"

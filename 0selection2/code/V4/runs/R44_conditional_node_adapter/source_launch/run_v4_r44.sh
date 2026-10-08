#!/usr/bin/env bash
set -euo pipefail

source /public/home/shiys/miniconda3/etc/profile.d/conda.sh
conda activate easynco
cd /public/home/shiys/0selection2
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 WANDB_MODE=offline
ROOT=code/V4/runs/R44_conditional_node_adapter
mkdir -p "$ROOT"

run_arm() {
    local arm=$1 seed=$2 gpu=$3
    CUDA_VISIBLE_DEVICES=$gpu python -u -m code.V4.r44_experiment \
        --stage train --root "$ROOT" --arm "$arm" --seed "$seed" --device cuda:0 --resume \
        > "$ROOT/${arm}_seed${seed}.launcher.log" 2>&1
}

CUDA_VISIBLE_DEVICES=0 python -u -m code.V4.r44_experiment --stage prepare --root "$ROOT" --device cuda:0
run_arm C0 2 0 & first=$!
run_arm C1 2 1 & second=$!
wait "$first"
run_arm C2 2 0 & third=$!
wait "$second"
wait "$third"
python -m code.V4.r44_analysis --root "$ROOT" --screen

if python -c 'import json,sys; sys.exit(0 if json.load(open(sys.argv[1]))["passed"] else 1)' "$ROOT/stage_a_screen.json"; then
    (
        run_arm C0 17 0
        run_arm C2 17 0
        run_arm C1 42 0
    ) & first=$!
    (
        run_arm C1 17 1
        run_arm C0 42 1
        run_arm C2 42 1
    ) & second=$!
    wait "$first"
    wait "$second"
    CUDA_VISIBLE_DEVICES=0 python -u -m code.V4.r44_experiment --stage test --root "$ROOT" --device cuda:0
    python -m code.V4.r44_analysis --root "$ROOT"
fi
printf '[R44 complete] %s/comparison.md\n' "$ROOT"

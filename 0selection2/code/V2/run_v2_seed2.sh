#!/usr/bin/env bash
set -euo pipefail

source ~/.bashrc
conda activate easynco_zhoucl

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python -m code.V2.train \
  --save-dir code/V2/runs/V2_solver_query_seed2 \
  --epochs 20 \
  --batch-per-problem 128 \
  --coord-augment 8 \
  --lr 2e-4 \
  --wd 1e-4 \
  --d 128 \
  --heads 4 \
  --encoder-layers 3 \
  --cross-layers 2 \
  --set-layers 1 \
  --ce-weight 0.35 \
  --pair-weight 0.30 \
  --gap-weight 0.25 \
  --risk-weight 0.10 \
  --seed 2 \
  --device cuda:0


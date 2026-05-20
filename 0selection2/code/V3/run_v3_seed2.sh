#!/usr/bin/env bash
set -euo pipefail

cd /public/home/zhoucl/shiys/0selection2
source /public/home/zhoucl/anaconda3/etc/profile.d/conda.sh
conda activate easynco_zhoucl

python -m code.V3.train \
  --save-dir code/V3/runs/V3_naive_seed2 \
  --epochs 20 \
  --batch-per-problem 256 \
  --coord-augment 8 \
  --num-workers 2 \
  --seed 2 \
  --device cuda:0


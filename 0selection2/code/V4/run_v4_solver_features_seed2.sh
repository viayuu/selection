#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."
source /public/home/shiys/miniconda3/etc/profile.d/conda.sh
conda activate easynco

export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
RUN=code/V4/runs/R32e_solver_features_seed2_4090
mkdir -p "$RUN"
exec > >(tee -a "$RUN/train.log") 2>&1

nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,power.draw --format=csv -l 10 > "$RUN/gpu.csv" &
MONITOR_PID=$!
trap 'code=$?; kill "$MONITOR_PID" 2>/dev/null || true; printf "%s\n" "$code" > "$RUN/exit_code"' EXIT
python -m pip freeze > "$RUN/pip_after.txt"
tar --exclude=runs --exclude=__pycache__ -czf "$RUN/source.tar.gz" code/V4 code/unified_selector/data.py code/unified_selector/registry.py
date -Is > "$RUN/started_at.txt"

python -m code.V4.train \
  --save-dir "$RUN" --epochs 30 --batch-per-problem 640 \
  --num-workers 0 --lr 2e-4 --wd 1e-4 --coord-augment 0 \
  --d 128 --heads 4 --ff-hidden 512 --head-hidden 256 \
  --encoder-layers 4 --set-layers 2 --query-num 4 --dropout 0.1 \
  --solver-feature-weight 0.3 --solver-feature-hidden 128 \
  --ce-weight 0.35 --pair-weight 0.30 --gap-weight 0.20 --pre-ce-weight 0.10 \
  --top-focused-pair --topk-ce-weight 0.08 --topk-ce-rank-weights 1.0,0.4,0.2 \
  --topk-ce-mode sequential --query-div-weight 0.03 --risk-weight 0.02 \
  --gap-score-weight 0.25 --pre-score-weight 0.25 --winner-balance \
  --support-branch --support-generators 4 --support-hidden 128 \
  --support-score-weight 0.0 --support-score-mode g0 --support-g0-as-main \
  --no-support-feature-token --support-main-utility-weight 0.0 \
  --support-main-pre-weight 0.0 --support-main-gap-weight 0.0 \
  --support-loss-weight 0.20 --support-eps 0.01 --support-topk 3 \
  --support-focal-gamma-neg 4.0 --support-pos-weight 2.0 \
  --support-diversity-weight 0.05 --support-budget-weight 0.02 \
  --support-target-mode rank_partition --amp --amp-dtype fp16 \
  --sdpa --cache-gpu --wandb --wandb-mode offline \
  --log-every 50 --eval-every 1 --seed 2 --device cuda:0
date -Is > "$RUN/finished_at.txt"

#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
source /public/home/shiys/miniconda3/etc/profile.d/conda.sh
conda activate easynco
export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
STUDY=code/V4/runs/R34_training_study
mkdir -p "$STUDY"
tar --exclude=runs --exclude=__pycache__ -czf "$STUDY/source.tar.gz" code/V4 code/unified_selector/data.py code/unified_selector/registry.py
python -m pip freeze > "$STUDY/pip_freeze.txt"
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,power.draw --format=csv -l 10 > "$STUDY/gpu.csv" &
MONITOR_PID=$!
trap 'code=$?; kill "$MONITOR_PID" 2>/dev/null || true; printf "%s\n" "$code" > "$STUDY/exit_code"' EXIT

for variant in original ce winner_cost; do
  RUN="code/V4/runs/R34_${variant}_seed2_4090"
  if [[ -e "$RUN/args.json" ]]; then
    printf 'Refusing to overwrite existing run: %s\n' "$RUN"
    exit 1
  fi
  mkdir -p "$RUN"
  EXTRA=()
  if [[ "$variant" == original ]]; then
    EXTRA=(--winner-balance --top-focused-pair --pair-weight 0.30 --topk-ce-weight 0.08 --topk-ce-mode sequential)
  elif [[ "$variant" == winner_cost ]]; then
    EXTRA=(--pair-weight 0.10 --cost-scale 0.01)
  fi
  date -Is > "$RUN/started_at.txt"
  python -m code.V4.train \
    --save-dir "$RUN" --architecture dual_stream --loss-mode "$variant" \
    --epochs 60 --batch-per-problem 640 --num-workers 0 --seed 2 \
    --lr 2e-4 --wd 1e-4 --warmup-epochs 3 --lr-schedule plateau --lr-patience 4 \
    --min-lr 2e-6 --early-stop-patience 10 --min-updates 12000 --min-delta 0.001 \
    --ignore-coord-dist --native-winner --coord-augment 0 \
    --d 128 --heads 4 --ff-hidden 512 --head-hidden 256 --encoder-layers 4 --joint-layers 2 \
    --dropout 0.1 --solver-feature-weight 1.0 --solver-feature-hidden 128 \
    --ce-weight 0.35 --risk-weight 0.02 --amp --amp-dtype fp16 --sdpa --cache-gpu \
    --train-eval-every 5 --skip-test --wandb --wandb-mode offline \
    --log-every 50 --eval-every 1 --device cuda:0 "${EXTRA[@]}" 2>&1 | tee "$RUN/train.log"
  date -Is > "$RUN/finished_at.txt"
done

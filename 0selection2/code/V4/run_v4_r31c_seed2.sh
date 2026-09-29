#!/usr/bin/env bash
set -euo pipefail

cd /public/home/zhoucl/shiys/0selection2

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128}

PY=${PY:-/public/home/zhoucl/anaconda3/envs/easynco_zhoucl/bin/python}
RUN=${RUN:-code/V4/runs/R31c_p2s_set_seed2}
LOG=${LOG:-$RUN/train.log}

mkdir -p "$RUN"
exec > >(tee -a "$LOG") 2>&1

"$PY" -m code.V4.train \
  --save-dir "$RUN" \
  --epochs ${EPOCHS:-30} \
  --batch-per-problem ${BATCH_PER_PROBLEM:-192} \
  --num-workers ${NUM_WORKERS:-0} \
  --lr ${LR:-2e-4} \
  --wd ${WD:-1e-4} \
  --coord-augment ${COORD_AUGMENT:-0} \
  --d ${D_MODEL:-128} \
  --heads ${HEADS:-4} \
  --ff-hidden ${FF_HIDDEN:-512} \
  --encoder-layers ${ENCODER_LAYERS:-4} \
  --set-layers ${SET_LAYERS:-2} \
  --query-num ${QUERY_NUM:-4} \
  --dropout ${DROPOUT:-0.1} \
  --ce-weight ${CE_WEIGHT:-0.35} \
  --pair-weight ${PAIR_WEIGHT:-0.30} \
  --gap-weight ${GAP_WEIGHT:-0.20} \
  --pre-ce-weight ${PRE_CE_WEIGHT:-0.10} \
  --query-div-weight ${QUERY_DIV_WEIGHT:-0.03} \
  --risk-weight ${RISK_WEIGHT:-0.02} \
  --gap-score-weight ${GAP_SCORE_WEIGHT:-0.25} \
  --pre-score-weight ${PRE_SCORE_WEIGHT:-0.25} \
  --winner-balance \
  --amp \
  --amp-dtype ${AMP_DTYPE:-fp16} \
  --log-every ${LOG_EVERY:-50} \
  --eval-every ${EVAL_EVERY:-1} \
  --seed ${SEED:-2} \
  --device cuda:0

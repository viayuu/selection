#!/usr/bin/env bash
set -euo pipefail

cd /public/home/zhoucl/shiys/0selection2

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128}

PY=${PY:-/public/home/zhoucl/anaconda3/envs/easynco_zhoucl/bin/python}
RUN=${RUN:-code/V4/runs/R32d_g0main_seqtopk_seed2}
LOG=${LOG:-$RUN/train.log}

mkdir -p "$RUN"
exec > >(tee -a "$LOG") 2>&1

"$PY" -m code.V4.train \
  --save-dir "$RUN" \
  --epochs ${EPOCHS:-30} \
  --batch-per-problem ${BATCH_PER_PROBLEM:-448} \
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
  --top-focused-pair \
  --topk-ce-weight ${TOPK_CE_WEIGHT:-0.08} \
  --topk-ce-rank-weights ${TOPK_CE_RANK_WEIGHTS:-1.0,0.4,0.2} \
  --topk-ce-mode sequential \
  --query-div-weight ${QUERY_DIV_WEIGHT:-0.03} \
  --risk-weight ${RISK_WEIGHT:-0.02} \
  --gap-score-weight ${GAP_SCORE_WEIGHT:-0.25} \
  --pre-score-weight ${PRE_SCORE_WEIGHT:-0.25} \
  --support-branch \
  --support-generators ${SUPPORT_GENERATORS:-4} \
  --support-hidden ${SUPPORT_HIDDEN:-128} \
  --support-score-weight ${SUPPORT_SCORE_WEIGHT:-0.0} \
  --support-score-mode g0 \
  --support-g0-as-main \
  --no-support-feature-token \
  --support-main-utility-weight ${SUPPORT_MAIN_UTILITY_WEIGHT:-0.0} \
  --support-main-pre-weight ${SUPPORT_MAIN_PRE_WEIGHT:-0.0} \
  --support-main-gap-weight ${SUPPORT_MAIN_GAP_WEIGHT:-0.0} \
  --support-loss-weight ${SUPPORT_LOSS_WEIGHT:-0.20} \
  --support-eps ${SUPPORT_EPS:-0.01} \
  --support-topk ${SUPPORT_TOPK:-3} \
  --support-focal-gamma-neg ${SUPPORT_FOCAL_GAMMA_NEG:-4.0} \
  --support-pos-weight ${SUPPORT_POS_WEIGHT:-2.0} \
  --support-diversity-weight ${SUPPORT_DIVERSITY_WEIGHT:-0.05} \
  --support-budget-weight ${SUPPORT_BUDGET_WEIGHT:-0.02} \
  --support-target-mode rank_partition \
  --winner-balance \
  --amp \
  --amp-dtype ${AMP_DTYPE:-fp16} \
  --log-every ${LOG_EVERY:-50} \
  --eval-every ${EVAL_EVERY:-1} \
  --seed ${SEED:-2} \
  --device cuda:0

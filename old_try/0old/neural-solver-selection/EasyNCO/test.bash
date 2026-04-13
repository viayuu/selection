#!/bin/bash
set -euo pipefail

RUN_ID="$(date +%Y%m%d-%H%M%S)"
LOG_DIR="logs/mtreld_runs"
LOG_FILE="${LOG_DIR}/run_${RUN_ID}.txt"
mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "Log file: ${LOG_FILE}"

CKPT_NAME="epoch-5000.pt"
CKPT_DIR_MOE_LIGHT="pretrained/pretrained/reld_moe_light"
CKPT_DIR_MTL="pretrained/pretrained/reld_mtl"

PROBS="CVRP VRPTW OVRP VRPL VRPB OVRPTW OVRPB OVRPL VRPBL VRPBTW VRPLTW OVRPBL OVRPBTW OVRPLTW VRPBLTW OVRPBLTW"

mkdir -p logs/mtreld_moe_light logs/mtreld_mtl

# 先只测一个，确认跑通（需要全量时删掉下一行）
# PROBS="OVRP"

for PROB in $PROBS; do
  prob_lower=$(echo "$PROB" | tr 'A-Z' 'a-z')
  echo "===== 测试 $PROB (${prob_lower}) ====="

  python eval.py \
    settings=mtreld_settings mode=test model=mtreld problem="$prob_lower" scale=100 cuda=[0] \
    seed=2024 matmul_precision=highest \
    test_data_path="mt/$PROB/${prob_lower}100_uniform.pkl" \
    settings.test_loader.model_dirpath="$CKPT_DIR_MOE_LIGHT" settings.test_loader.model_filename="$CKPT_NAME" \
    episodes=1000 batch_size=100 \
    ++settings.env.aug_factor=8 \
    ++settings.module.decoder_strategy=greedy \
    ++settings.module.test_data_params.mode=1 \
    2>&1 | tee "logs/mtreld_moe_light/${prob_lower}_100_aug8.txt"

  python eval.py \
    settings=mtreld_mtl_settings mode=test model=mtreld_mtl problem="$prob_lower" scale=100 cuda=[0] \
    seed=2024 matmul_precision=highest \
    test_data_path="mt/$PROB/${prob_lower}100_uniform.pkl" \
    settings.test_loader.model_dirpath="$CKPT_DIR_MTL" settings.test_loader.model_filename="$CKPT_NAME" \
    episodes=1000 batch_size=100 \
    ++settings.env.aug_factor=8 \
    ++settings.module.decoder_strategy=greedy \
    ++settings.module.test_data_params.mode=1 \
    2>&1 | tee "logs/mtreld_mtl/${prob_lower}_100_aug8.txt"
done

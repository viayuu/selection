#!/usr/bin/env bash
set -euo pipefail

source /public/home/zhoucl/anaconda3/etc/profile.d/conda.sh
conda activate easynco_zhoucl

cd /public/home/zhoucl/shiys/0selection2/code/NSS_retrain
python scripts/audit_ours_data.py

python run.py --config_name config_TSP_ours.yml --loss rank --seed 2024 --gpu_id 0 &
pid_tsp=$!
python run.py --config_name config_CVRP_ours.yml --loss rank --seed 2024 --gpu_id 1 &
pid_cvrp=$!

wait "$pid_tsp"
wait "$pid_cvrp"

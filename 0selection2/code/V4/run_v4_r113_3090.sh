#!/usr/bin/env bash
set -euo pipefail
source /public/home/shiys/miniconda3/etc/profile.d/conda.sh
conda activate easynco
cd /public/home/shiys/0selection2
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=/public/home/shiys/easynco_v3_bridge/exports/train_100k_20261002/python_compat:/public/home/shiys/0selection2
exec python -B -u -m code.V4.r113_3090_queue "$@"

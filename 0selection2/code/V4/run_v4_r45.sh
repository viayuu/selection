#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source /public/home/shiys/miniconda3/etc/profile.d/conda.sh
conda activate easynco
export OMP_NUM_THREADS=4
export WANDB_MODE=offline
# With no arguments this writes configurations only. Training needs --stage train/all.
python -u -m code.V4.r45_experiment "$@"

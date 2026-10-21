#!/usr/bin/env bash
set -euo pipefail
source /public/home/shiys/miniconda3/etc/profile.d/conda.sh
conda activate easynco
cd /public/home/shiys/0selection2
: "${1:?Explicit main-thread final-ready evidence is required}"
bash code/V4/run_v4_r81.sh selftest
exec bash code/V4/run_v4_r81.sh arm --objective-tests-passed --runner-tests-passed \
    --preparation-receipt /public/home/shiys/0selection2/code/V4/runs/R81_method_conditioned_auc/prepared/prepared.json \
    --ready-evidence "$1"

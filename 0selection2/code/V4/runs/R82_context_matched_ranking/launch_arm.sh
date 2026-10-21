#!/usr/bin/env bash
set -euo pipefail
source /public/home/shiys/miniconda3/etc/profile.d/conda.sh
conda activate easynco
cd /public/home/shiys/0selection2

evidence="${1:?Pass the explicit main-thread final-ready evidence after objective/runner tests and TRAIN pair coverage}"
bash code/V4/run_v4_r82.sh selftest
bash code/V4/run_v4_r82.sh cli-check
bash code/V4/run_v4_r82.sh arm --objective-tests-passed --runner-tests-passed \
    --pair-coverage-checked --ready-evidence "$evidence"

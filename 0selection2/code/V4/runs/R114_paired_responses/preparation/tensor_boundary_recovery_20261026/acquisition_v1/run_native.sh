#!/usr/bin/env bash
source /public/home/shiys/miniconda3/etc/profile.d/conda.sh
conda activate easynco
set -euo pipefail
cd /public/home/shiys/0selection2
: "${R114_RECOVERY_HOST:?Main must set the container hostname}"
: "${R114_RECOVERY_GPU_UUID:?Main must hold the external GPU mutex}"
: "${R114_RECOVERY_CPU_IDS:?Main must assign 1-2 CPU IDs}"
: "${R114_RECOVERY_TIMEOUT_SECONDS:?Main must supply the unchanged timeout}"
export CUDA_VISIBLE_DEVICES="$R114_RECOVERY_GPU_UUID"
export PYTHONDONTWRITEBYTECODE=1
/public/home/shiys/miniconda3/envs/easynco/bin/python -B -m code.V4.r114_run_queue --plan /public/home/shiys/0selection2/code/V4/runs/R114_paired_responses/preparation/tensor_boundary_recovery_20261026/acquisition_v1/queue/plan.json --manifest /public/home/shiys/0selection2/code/V4/runs/R114_paired_responses/preparation/tensor_boundary_recovery_20261026/acquisition_v1/manifest.json --deployment-lock /public/home/shiys/0selection2/code/V4/runs/R114_paired_responses/preparation/tensor_boundary_recovery_20261026/acquisition_v1/queue/deployments.grouped.lock.json --host "$R114_RECOVERY_HOST" --gpu-uuid "$R114_RECOVERY_GPU_UUID" --cpu-ids "$R114_RECOVERY_CPU_IDS" --timeout-seconds "$R114_RECOVERY_TIMEOUT_SECONDS" --execute

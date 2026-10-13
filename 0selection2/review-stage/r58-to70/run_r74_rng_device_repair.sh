#!/usr/bin/env bash
set -euo pipefail
source /public/home/shiys/miniconda3/etc/profile.d/conda.sh
conda activate easynco
cd /public/home/shiys/0selection2
export CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONPATH=/public/home/shiys/easynco_v3_bridge/exports/train_100k_20261002/python_compat:/public/home/shiys/0selection2
export PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg
printf '[R74 CPU randperm device repair] shell=%s host=%s started=%s\n' "$$" "$(hostname -s)" "$(date -u +%FT%TZ)"
set +e
codex exec resume --model gpt-6.1-sol -c 'model_reasoning_effort="max"' \
 --dangerously-bypass-approvals-and-sandbox --json \
 --output-last-message .aris/traces/R74_rng_device_repair.response.md \
 01a134a7-bc56-7ec2-9563-076ec70a0a96 - \
 < review-stage/r58-to70/R74_rng_device_repair_executor.prompt.md \
 > .aris/traces/R74_rng_device_repair.events.jsonl \
 2> .aris/traces/R74_rng_device_repair.stderr.log
status=$?
printf '[R74 CPU randperm device repair] CLI returncode=%s finished=%s\n' "$status" "$(date -u +%FT%TZ)"
exit "$status"

#!/usr/bin/env bash
set -euo pipefail
source /public/home/shiys/miniconda3/etc/profile.d/conda.sh
conda activate easynco
cd /public/home/shiys/0selection2
export CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONPATH=/public/home/shiys/easynco_v3_bridge/exports/train_100k_20261002/python_compat:/public/home/shiys/0selection2
export PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg
printf '[R69 CPU delivery executor] shell=%s host=%s started=%s\n' "$$" "$(hostname -s)" "$(date -u +%FT%TZ)"
set +e
codex exec resume --model gpt-6.1-sol \
    -c 'model_reasoning_effort="max"' \
    --dangerously-bypass-approvals-and-sandbox --json \
    --output-last-message .aris/traces/R69_locked_delivery.response.md \
    01a131bb-a6de-7280-ae9a-ad081a39f017 - \
    < review-stage/r58-to70/R69_locked_delivery_executor.prompt.md \
    > .aris/traces/R69_locked_delivery.events.jsonl \
    2> .aris/traces/R69_locked_delivery.stderr.log
status=$?
printf '[R69 CPU delivery executor] CLI returncode=%s finished=%s\n' "$status" "$(date -u +%FT%TZ)"
exit "$status"

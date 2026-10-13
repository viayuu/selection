#!/usr/bin/env bash
set -euo pipefail
source /public/home/shiys/miniconda3/etc/profile.d/conda.sh
conda activate easynco
cd /public/home/shiys/0selection2
export CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONPATH=/public/home/shiys/easynco_v3_bridge/exports/train_100k_20261002/python_compat:/public/home/shiys/0selection2
export PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg
printf '[R73 primary TRAIN/VAL closure] shell=%s host=%s started=%s\n' "$$" "$(hostname -s)" "$(date -u +%FT%TZ)"
set +e
codex exec resume --model gpt-6-astra \
 -c 'model_reasoning_effort="max"' --dangerously-bypass-approvals-and-sandbox \
 --json --output-last-message .aris/traces/R73_closure_next_plan.response.md \
 01a12e68-4d2a-71d0-aefd-67fc1c158282 - \
 < review-stage/r58-to70/R73_closure_next_plan.prompt.md \
 > .aris/traces/R73_closure_next_plan.events.jsonl \
 2> .aris/traces/R73_closure_next_plan.stderr.log
status=$?
printf '[R73 primary TRAIN/VAL closure] CLI returncode=%s finished=%s\n' "$status" "$(date -u +%FT%TZ)"
exit "$status"

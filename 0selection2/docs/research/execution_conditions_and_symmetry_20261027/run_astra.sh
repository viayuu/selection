#!/usr/bin/env bash
# Persist this scientific sidecar across main-thread continuations.
source /public/home/shiys/miniconda3/etc/profile.d/conda.sh
conda activate easynco || exit 1
set -uo pipefail
cd /public/home/shiys/0selection2 || exit 1
export CUDA_VISIBLE_DEVICES=""
directory="$PWD/docs/research/execution_conditions_and_symmetry_20261027"
if [[ -e "$directory/cli_started.json" ]]; then
    printf '%s\n' 'Existing attempt; inspect its process and receipt instead of retrying.' >&2
    exit 2
fi
printf '{"pid":%s,"started_utc":"%s","model":"gpt-6-astra","reasoning_effort":"max","gpu_access":false}\n' \
    "$$" "$(date -u +%FT%TZ)" > "$directory/cli_started.json"
timeout --signal=TERM --kill-after=60 5400 \
    codex exec -C "$PWD" -m gpt-6-astra \
    -c 'model_reasoning_effort="max"' -c 'approval_policy="never"' \
    --sandbox danger-full-access --json \
    --output-last-message "$directory/CLI_FINAL.txt" \
    'Read docs/research/execution_conditions_and_symmetry_20261027/REQUEST.md and complete only its bounded original-paper scientific task. Own REPORT.md,sources.csv,directions.csv in that directory only; use apply_patch for manual edits. Prior subagent attempts produced no scientific files: one capacity error and one missing handle after main-thread continuation. Do not duplicate broader research or run child agents. You are explicitly configured gpt-6-astra/max; report any actual runtime configuration mismatch, not generic Codex branding. Preserve the key counterfacts already supplied by Sol: TSP physical node0 is already explicitly marked in current R94B; CVRP100 starts use endogenous decoder TopK, not the first100 customers. Distinguish fresh MATNET roles from the unrecovered old-label realization. Read a few primary papers directly; cite exact versions and sections actually accessed. Keep scope4-6pages and at mostONEmajor scientific direction or NONEready; no code review, numeric certification, experiment budgets/plans, GPU/solving/training, Git, or notifications. No query/holdout answers or R58 label repair. Main Sol owns operational implementation and mechanical PDF. Other agents may edit unrelated files; do not touch them. Finish owned deliverables, state limitations truthfully, then stop.' \
    > "$directory/cli_events.jsonl" 2> "$directory/cli_stderr.log"
rc=$?
printf '{"exit_code":%s,"finished_utc":"%s","automatic_retry":false}\n' \
    "$rc" "$(date -u +%FT%TZ)" > "$directory/cli_exit.json"
exit "$rc"

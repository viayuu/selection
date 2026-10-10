#!/usr/bin/env bash
set -euo pipefail
source /public/home/shiys/miniconda3/etc/profile.d/conda.sh
conda activate easynco
cd /public/home/shiys/0selection2
ROOT=/public/home/shiys/0selection2/code/V4/runs/R58_unified_scenario

[[ "${R58_PARENT_GPU_BATON:-}" == revised-preflight ]] || {
    printf '%s\n' 'Explicit parent revised-preflight GPU baton required.' >&2
    exit 64
}
[[ -z "$(compgen -v CUDA_MPS_ || true)" ]] || {
    printf '%s\n' 'Preflight must not inherit any CUDA_MPS_* configuration.' >&2
    exit 64
}
git diff --quiet ba108ed0e -- code/V4/r58_scenario.py code/V4/r58_environments.py \
    code/V4/r58_backends.py code/V4/r58_labels.py code/V4/r58_execution.py
exec 9> "$ROOT/revised_preflight_launch.lock"
flock -n 9
for name in controller.log started exit_code timeout_exit_code result.json state.json failure_state.json; do
    [[ ! -e "$ROOT/revised_preflight_$name" ]] || {
        printf 'Existing revised preflight artifact; no automatic retry: %s\n' "$name" >&2
        exit 64
    }
done
export CUDA_VISIBLE_DEVICES=GPU-cb626bdb-4319-2f5e-b91f-17fcaa44a83d
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 WANDB_MODE=offline
export PYTHONPATH=/public/home/shiys/easynco_v3_bridge/exports/train_100k_20261002/python_compat:/public/home/shiys/0selection2
export WANDB_DIR="$ROOT" MPLCONFIGDIR="$ROOT/mplconfig"
started=$(date +%s)
printf '%s\n' "$started" > "$ROOT/revised_preflight_started"
exec > "$ROOT/revised_preflight_controller.log" 2>&1

finish() {
    local status=$? finished complete=false copied=false launcher_status
    trap - EXIT
    set +e
    finished=$(date +%s)
    launcher_status=$status
    printf '%s\n' "$status" > "$ROOT/revised_preflight_exit_code"
    if [[ -f "$ROOT/pipeline_state.json" && "$ROOT/pipeline_state.json" -nt "$ROOT/revised_preflight_started" ]]; then
        if [[ "$status" == 0 ]] && python -c 'import json,sys; s=json.load(open(sys.argv[1])); sys.exit(not (s.get("stage")=="preflight" and s.get("status")=="preflight_complete" and s.get("finished")))' "$ROOT/pipeline_state.json"; then
            if cp -p "$ROOT/pipeline_state.json" "$ROOT/revised_preflight_state.json"; then
                complete=true
                copied=true
            fi
        else
            cp -p "$ROOT/pipeline_state.json" "$ROOT/revised_preflight_failure_state.json"
        fi
    fi
    if [[ "$status" == 0 && "$complete" != true ]]; then launcher_status=65; fi
    printf '{"started":%s,"finished":%s,"elapsed_wall_seconds":%s,"srun_returncode":%s,"launcher_returncode":%s,"preflight_complete":%s,"completed_state_copied":%s,"automatic_retry":false,"mps":false}\n' \
        "$started" "$finished" "$((finished-started))" "$status" "$launcher_status" "$complete" "$copied" \
        > "$ROOT/revised_preflight_result.json"
    exit "$launcher_status"
}
trap finish EXIT

srun --jobid=1465 --overlap --exact --nodes=1 --ntasks=1 --nodelist=gpu03 \
    --cpus-per-task=4 --mem=16G env \
    CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH="$PYTHONPATH" WANDB_MODE=offline WANDB_DIR="$WANDB_DIR" MPLCONFIGDIR="$MPLCONFIGDIR" \
    bash -c '
        set -uo pipefail
        [[ "$SLURM_JOB_ID" == 1465 && "$(hostname -s)" == gpu03 && "$CONDA_DEFAULT_ENV" == easynco ]] || exit 64
        timeout --signal=TERM --kill-after=30s 3600s \
            /public/home/shiys/miniconda3/envs/easynco/bin/python -u -m code.V4.r58_pipeline \
            --root "$1" --stage preflight --force-preflight
        status=$?
        printf "%s\n" "$status" > "$1/revised_preflight_timeout_exit_code"
        exit "$status"
    ' r58-revised-preflight "$ROOT"

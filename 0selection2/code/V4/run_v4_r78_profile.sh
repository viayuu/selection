#!/usr/bin/env bash
set -euo pipefail
source /public/home/shiys/miniconda3/etc/profile.d/conda.sh
conda activate easynco
cd /public/home/shiys/0selection2

export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg
export PYTHONPATH=/public/home/shiys/easynco_v3_bridge/exports/train_100k_20261002/python_compat:/public/home/shiys/0selection2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=offline WANDB_PROJECT=selector WANDB_ENTITY=yjkds-southern-university-of-science-technology

ROOT=/public/home/shiys/0selection2/code/V4/runs/R78_zero_selection_goal/round006
SCRIPT=/public/home/shiys/0selection2/code/V4/run_v4_r78_profile.sh
stage="${1:-help}"
ALLOCATION="${2:-$ROOT/hardware_allocation.pending.json}"

approved() {
    [[ -f "$ROOT/prepared.json" && -f "$ROOT/cpu_checks.json" ]] || exit 2
    python -B -m code.V4.r78_profile_experiment allocation --allocation "$ALLOCATION" >/dev/null
    GPU="$(jq -er '.gpu_uuid' "$ALLOCATION")"
    NODE="$(jq -er '.node' "$ALLOCATION")"
    JOB="$(jq -er '.job_id' "$ALLOCATION")"
    CACHE="$(jq -er '.cache_root' "$ALLOCATION")"
}

case "$stage" in
    prepare)
        export CUDA_VISIBLE_DEVICES=''
        exec python -B -u -m code.V4.r78_profile_experiment prepare --root "$ROOT"
        ;;
    cpu-tests)
        export CUDA_VISIBLE_DEVICES=''
        export R78_PROFILE_CHECK_RECEIPT="$ROOT/cpu_checks.json"
        python -B -u -m code.V4.test_r78_profile 2>&1 | tee -a "$ROOT/cpu_checks.log"
        ;;
    start)
        approved
        session=R78_round006_profile_seed2
        if tmux has-session -t "$session" 2>/dev/null; then
            printf 'Existing session retained: %s\n' "$session" >&2; exit 2
        fi
        printf -v command 'bash %q allocated %q >>%q 2>&1' "$SCRIPT" "$ALLOCATION" "$ROOT/launch.log"
        tmux new-session -d -s "$session" "$command"
        printf 'Requested tmux %s; log: %s/train.log\n' "$session" "$ROOT"
        ;;
    allocated)
        approved
        exec srun --jobid="$JOB" --overlap --exact --nodes=1 --nodelist="$NODE" --ntasks=1 \
            --cpus-per-task=4 --mem=9G --gres=gpu:1 bash "$SCRIPT" __worker "$ALLOCATION"
        ;;
    __worker)
        approved
        [[ "${SLURM_JOB_ID:-}" == "$JOB" && -n "${SLURM_STEP_ID:-}" && "$(hostname -s)" == "$NODE" ]] || exit 2
        export CUDA_VISIBLE_DEVICES="$GPU"
        exec timeout --signal=TERM --kill-after=10s 4h python -B -u -m code.V4.r78_profile_experiment train \
            --root "$ROOT" --allocation "$ALLOCATION" --cache-root "$CACHE" >"$ROOT/train.log" 2>&1
        ;;
    help)
        printf '%s\n' 'CPU only: bash code/V4/run_v4_r78_profile.sh prepare | cpu-tests' \
            'After R79 complete freeze+TEST/LIB delivery and explicit single4090 UUID/allocation:' \
            'bash code/V4/run_v4_r78_profile.sh start /absolute/path/to/approved_allocation.json' \
            'No separate smoke: successful GPU update1 is formal budget update1.'
        ;;
    *) exit 2 ;;
esac

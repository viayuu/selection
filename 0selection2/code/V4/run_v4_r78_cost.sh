#!/usr/bin/env bash
set -euo pipefail
source /public/home/shiys/miniconda3/etc/profile.d/conda.sh
conda activate easynco
cd /public/home/shiys/0selection2

export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg
export PYTHONPATH=/public/home/shiys/easynco_v3_bridge/exports/train_100k_20261002/python_compat:/public/home/shiys/0selection2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE="${WANDB_MODE:-offline}" WANDB_PROJECT=selector
export WANDB_ENTITY=yjkds-southern-university-of-science-technology

ROOT="${R78_COST_ROOT:-/public/home/shiys/0selection2/code/V4/runs/R78_zero_selection_goal/round003}"
R77="${R78_R77_ROOT:-/public/home/shiys/0selection2/code/V4/runs/R77_sample_size_batch1024}"
CACHE="${R78_COST_CACHE:-/tmp/shiys_R76_R45A_single_reld_mt_cache}"
ALLOCATION="${R78_COST_ALLOCATION:-$ROOT/hardware_allocation.pending.json}"
SCRIPT=/public/home/shiys/0selection2/code/V4/run_v4_r78_cost.sh
stage="${1:-help}"
arm="${2:-treatment}"

approved() {
    [[ "$arm" == control || "$arm" == treatment ]] || { printf 'Unknown arm: %s\n' "$arm" >&2; exit 2; }
    [[ "${R78_COST_APPROVED:-0}" == 1 && -f "$ROOT/prepared.json" && -f "$ALLOCATION" ]] || {
        printf '%s\n' 'Prepare only. NSS has priority; explicit user hardware approval is required before any GPU smoke or training.' >&2
        exit 2
    }
    jq -e --arg arm "$arm" '.approved == true and .arms[$arm].approved == true' "$ALLOCATION" >/dev/null || {
        printf '%s\n' 'This arm has no approved hardware allocation.' >&2; exit 2;
    }
    GPU="$(jq -er --arg arm "$arm" '.arms[$arm].gpu_uuid' "$ALLOCATION")"
    NODE="$(jq -er --arg arm "$arm" '.arms[$arm].node' "$ALLOCATION")"
    JOB="$(jq -er --arg arm "$arm" '.arms[$arm].job_id' "$ALLOCATION")"
    [[ "$JOB" =~ ^[0-9]+$ ]] || { printf '%s\n' 'Use an explicitly approved existing Slurm job.' >&2; exit 2; }
    case "$NODE:$GPU" in
        gpu05:GPU-71479e7e-f8bd-e430-66ed-f8db9f0d47be|gpu05:GPU-597c682a-4b7f-7609-80a6-8ced791f5c5d|\
        gpu03:GPU-cb24f168-9f18-f0d5-6f4a-6d2406958f50|gpu03:GPU-cb626bdb-4319-2f5e-b91f-17fcaa44a83d) ;;
        *) printf '%s\n' 'Only the four user-approved node/UUID pairs are permitted.' >&2; exit 2 ;;
    esac
}

case "$stage" in
    prepare)
        export CUDA_VISIBLE_DEVICES=''
        extra=()
        [[ -z "${R78_PREPARE_CACHE:-}" ]] || extra+=(--prepare-cache "$R78_PREPARE_CACHE")
        exec python -B -u -m code.V4.r78_fullb_cost_experiment --stage prepare --root "$ROOT" --r77-root "$R77" "${extra[@]}"
        ;;
    status|analyze|watch|first_val|final_report)
        export CUDA_VISIBLE_DEVICES=''
        exec python -B -u -m code.V4.r78_fullb_cost_experiment --stage "$stage" --root "$ROOT"
        ;;
    start|smoke)
        approved
        mode=train
        [[ "$stage" != smoke ]] || mode=smoke
        session="${R78_COST_TMUX_SESSION:-R78_round003_${arm}_${mode}}"
        if tmux has-session -t "$session" 2>/dev/null; then
            printf 'Existing tmux session retained: %s\n' "$session" >&2; exit 2
        fi
        printf -v command 'env R78_COST_APPROVED=1 R78_COST_ROOT=%q R78_COST_ALLOCATION=%q R78_COST_CACHE=%q R78_COST_RESUME=%q WANDB_MODE=%q bash %q allocated %q %q' \
            "$ROOT" "$ALLOCATION" "$CACHE" "${R78_COST_RESUME:-0}" "$WANDB_MODE" "$SCRIPT" "$arm" "$mode"
        tmux new-session -d -s "$session" "$command"
        printf 'Allocated %s/%s launch requested in tmux %s; no SSH GPU execution.\n' "$arm" "$mode" "$session"
        ;;
    allocated)
        approved
        mode="${3:-train}"
        [[ "$mode" == train || "$mode" == smoke ]] || exit 2
        exec srun --jobid="$JOB" --overlap --exact --nodes=1 --nodelist="$NODE" --ntasks=1 \
            --cpus-per-task=4 --mem=9G --gres=gpu:2 bash "$SCRIPT" __worker "$arm" "$mode"
        ;;
    start_pair)
        arm=control
        approved
        [[ "$NODE" == gpu03 && "$JOB" == 1465 ]] || exit 2
        arm=treatment
        approved
        [[ "$NODE" == gpu03 && "$JOB" == 1465 ]] || exit 2
        session="${R78_COST_TMUX_SESSION:-R78_round003_matched_pair}"
        if tmux has-session -t "$session" 2>/dev/null; then
            printf 'Existing tmux session retained: %s\n' "$session" >&2; exit 2
        fi
        printf -v command 'env R78_COST_APPROVED=1 R78_COST_ROOT=%q R78_COST_ALLOCATION=%q R78_COST_CACHE=%q R78_COST_RESUME=%q WANDB_MODE=%q bash %q __pair_allocated' \
            "$ROOT" "$ALLOCATION" "$CACHE" "${R78_COST_RESUME:-0}" "$WANDB_MODE" "$SCRIPT"
        tmux new-session -d -s "$session" "$command"
        printf 'Matched control/treatment requested in tmux %s; one8CPU/18GiB two-GPU Slurm step, independent UUID-bound processes.\n' "$session"
        ;;
    __pair_allocated)
        approved
        exec srun --jobid="$JOB" --overlap --exact --nodes=1 --nodelist="$NODE" --ntasks=1 \
            --cpus-per-task=8 --mem=18G --gres=gpu:2 bash "$SCRIPT" __pair_worker
        ;;
    __pair_worker)
        bash "$SCRIPT" __worker control train &
        control_pid=$!
        bash "$SCRIPT" __worker treatment train &
        treatment_pid=$!
        code=0
        wait "$control_pid" || code=1
        wait "$treatment_pid" || code=1
        exit "$code"
        ;;
    __worker)
        approved
        mode="${3:-train}"
        [[ "$mode" == train || "$mode" == smoke ]] || exit 2
        [[ "${SLURM_JOB_ID:-}" == "$JOB" && -n "${SLURM_STEP_ID:-}" && "$(hostname -s)" == "$NODE" ]] || {
            printf '%s\n' 'GPU work must run inside the assigned exact Slurm step.' >&2; exit 2;
        }
        mkdir -p "$ROOT/locks"
        exec 9>"$ROOT/locks/$GPU.lock"
        flock -n 9 || { printf '%s\n' 'Another R78 run owns this UUID; no process killed.' >&2; exit 2; }
        export CUDA_VISIBLE_DEVICES="$GPU"
        extra=()
        [[ "${R78_COST_RESUME:-0}" != 1 || "$mode" != train ]] || extra+=(--resume)
        exec python -B -u -m code.V4.r78_fullb_cost_experiment --stage "$mode" --arm "$arm" \
            --root "$ROOT" --allocation "$ALLOCATION" --cache-root "$CACHE" --wandb-mode "$WANDB_MODE" \
            "${extra[@]}" >"$ROOT/$arm/${mode}.log" 2>&1
        ;;
    help)
        printf '%s\n' 'CPU preparation: bash code/V4/run_v4_r78_cost.sh prepare' \
            'No GPU launch until the user assigns hardware and approves the allocation JSON.' \
            'After explicit allocation: R78_COST_APPROVED=1 R78_COST_ALLOCATION=... bash code/V4/run_v4_r78_cost.sh smoke treatment' \
            'Then launch each assigned arm: ... bash code/V4/run_v4_r78_cost.sh start control|treatment' \
            'CPU matched report after both arms finish: bash code/V4/run_v4_r78_cost.sh analyze'
        ;;
    *) printf 'Unknown R78 stage: %s\n' "$stage" >&2; exit 2 ;;
esac

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

ROOT="${R77_ROOT:-/public/home/shiys/0selection2/code/V4/runs/R77_sample_size_batch1024}"
DATA="${R77_DATA_ROOT:-/public/home/shiys/selection_data_new_only_B_filtered_single_reld_mt_seed2_20261014}"
SOURCE="${R77_SOURCE_ROOT:-/public/home/shiys/0selection2/code/V4/runs/R76_R45A_single_reld_mt}"
LOCAL_CACHE="${R77_LOCAL_CACHE:-/tmp/shiys_R76_R45A_single_reld_mt_cache}"
GPU_A="${R77_GPU_A:-GPU-71479e7e-f8bd-e430-66ed-f8db9f0d47be}"
GPU_B="${R77_GPU_B:-GPU-597c682a-4b7f-7609-80a6-8ced791f5c5d}"
JOB="${R77_JOB:-1475}"
NODE="${R77_NODE:-gpu05}"
SESSION="${R77_TMUX_SESSION:-R77_sample_size_batch1024}"
stage="${1:-help}"

approved() {
    [[ "${R77_APPROVED:-0}" == 1 ]] || {
        printf '%s\n' 'R77 awaits formal user approval. Set R77_APPROVED=1 only after allocation is confirmed.' >&2
        exit 2
    }
    [[ "$GPU_A" == GPU-71479e7e-f8bd-e430-66ed-f8db9f0d47be && \
       "$GPU_B" == GPU-597c682a-4b7f-7609-80a6-8ced791f5c5d && "$JOB" == 1475 && "$NODE" == gpu05 ]] || {
        printf '%s\n' 'Only the two user-approved job1475/gpu05 GPU UUIDs are allowed.' >&2
        exit 2
    }
    [[ -f "$ROOT/prepared.json" ]] || {
        printf '%s\n' 'Run prepare once (CPU, TRAIN only) before starting the allocated GPUs.' >&2
        exit 2
    }
}

case "$stage" in
    prepare)
        export CUDA_VISIBLE_DEVICES=''
        exec python -B -u -m code.V4.r77_data --root "$ROOT" --data-root "$DATA" --source-root "$SOURCE" \
            --cache-root "${R77_PREPARE_CACHE:-$SOURCE/cache}"
        ;;
    status)
        export CUDA_VISIBLE_DEVICES=''
        exec python -B -u -m code.V4.r77_experiment --stage status --root "$ROOT"
        ;;
    val_report)
        export CUDA_VISIBLE_DEVICES=''
        exec python -B -u -m code.V4.r77_experiment --stage val_report --root "$ROOT"
        ;;
    snapshot)
        export CUDA_VISIBLE_DEVICES=''
        exec python -B -u -m code.V4.r77_experiment --stage snapshot --root "$ROOT"
        ;;
    report_watch)
        export CUDA_VISIBLE_DEVICES=''
        exec python -B -u -m code.V4.r77_experiment --stage report_watch --root "$ROOT" --live-gpu
        ;;
    finalize_delivery)
        export CUDA_VISIBLE_DEVICES=''
        exec python -B -u -m code.V4.r77_experiment --stage finalize_delivery --root "$ROOT" --live-gpu
        ;;
    success_watch)
        export CUDA_VISIBLE_DEVICES=''
        exec python -B -u -m code.V4.r77_experiment --stage success_watch --root "$ROOT"
        ;;
    diagnostic_handoff|diagnostic_watch)
        export CUDA_VISIBLE_DEVICES=''
        exec python -B -u -m code.V4.r77_experiment --stage "$stage" --root "$ROOT"
        ;;
    monitor)
        export CUDA_VISIBLE_DEVICES=''
        exec python -B -u -m code.V4.r77_experiment --stage watch --root "$ROOT"
        ;;
    allocated)
        approved
        arm="${2:?Specify arm A or B}"
        [[ "$arm" == A || "$arm" == B ]] || { printf '%s\n' 'Arm must be A or B.' >&2; exit 2; }
        printf '%s\n' "$$" >"$ROOT/${arm}_controller.pid"
        printf '%s\n' running >"$ROOT/${arm}_execution_exit_code"
        code=0
        srun --jobid="$JOB" --overlap --exact --nodes=1 --nodelist="$NODE" --ntasks=1 --cpus-per-task=4 --mem=9G \
            --gres=gpu:1 bash /public/home/shiys/0selection2/code/V4/run_v4_r77.sh __worker "$arm" || code=$?
        printf '%s\n' "$code" >"$ROOT/${arm}_execution_exit_code"
        exit "$code"
        ;;
    __worker|train|test|finish|analysis)
        arm="${2:?Specify arm A or B}"
        [[ "$arm" == A || "$arm" == B ]] || { printf '%s\n' 'Arm must be A or B.' >&2; exit 2; }
        mode="$stage"
        [[ "$mode" != __worker ]] || mode=finish
        if [[ "$mode" != analysis ]]; then
            approved
            [[ -d "$LOCAL_CACHE/train" && -d "$LOCAL_CACHE/val" && -d "$LOCAL_CACHE/test" ]] || {
                printf 'Existing train/val/test local cache required at %s; no automatic cache rebuild.\n' "$LOCAL_CACHE" >&2
                exit 2
            }
            if [[ "$arm" == A ]]; then export CUDA_VISIBLE_DEVICES="$GPU_A"; else export CUDA_VISIBLE_DEVICES="$GPU_B"; fi
        else
            export CUDA_VISIBLE_DEVICES=''
        fi
        extra=()
        arm_directory=A_10k_seed2
        [[ "$arm" == A ]] || arm_directory=B_full_seed2
        if [[ "${R77_RESUME:-0}" == 1 && -f "$ROOT/$arm_directory/last.pt" ]]; then extra+=(--resume); fi
        exec python -B -u -m code.V4.r77_experiment --stage "$mode" --arm "$arm" --root "$ROOT" \
            --data-root "$DATA" --cache-root "$LOCAL_CACHE" --gpu-type "${R77_GPU_TYPE:-4090}" \
            --large-micro-cap "${R77_LARGE_MICRO_CAP:-352}" --eval-batch-size 256 \
            --wandb-mode "$WANDB_MODE" "${extra[@]}"
        ;;
    start)
        approved
        if tmux has-session -t "$SESSION" 2>/dev/null; then
            printf 'tmux session already exists: %s (no process replaced).\n' "$SESSION" >&2
            exit 2
        fi
        printf -v command 'env R77_APPROVED=1 R77_ROOT=%q R77_DATA_ROOT=%q R77_SOURCE_ROOT=%q R77_LOCAL_CACHE=%q R77_GPU_A=%q R77_GPU_B=%q R77_JOB=%q R77_NODE=%q R77_RESUME=%q WANDB_MODE=%q bash %q __pair' \
            "$ROOT" "$DATA" "$SOURCE" "$LOCAL_CACHE" "$GPU_A" "$GPU_B" "$JOB" "$NODE" "${R77_RESUME:-0}" \
            "$WANDB_MODE" /public/home/shiys/0selection2/code/V4/run_v4_r77.sh
        tmux new-session -d -s "$SESSION" "$command"
        printf 'R77 A/B launched in tmux %s; job=%s node=%s, one independent GPU each.\n' "$SESSION" "$JOB" "$NODE"
        ;;
    __pair)
        approved
        exec 9>"$ROOT/launcher.lock"
        flock -n 9 || { printf '%s\n' 'An R77 launcher already owns this pair.' >&2; exit 2; }
        printf '%s\n' running >"$ROOT/launcher_exit_code"
        touch "$ROOT/A_launch_marker"
        for arm in A B; do
            bash /public/home/shiys/0selection2/code/V4/run_v4_r77.sh allocated "$arm" >"$ROOT/${arm}_launcher.log" 2>&1 &
            if [[ "$arm" == A ]]; then pid_a=$!; else pid_b=$!; fi
            if [[ "$arm" == A ]]; then
                ready=0
                for attempt in {1..120}; do
                    if [[ -f "$ROOT/A_10k_seed2/launch_receipt.json" && \
                          "$ROOT/A_10k_seed2/launch_receipt.json" -nt "$ROOT/A_launch_marker" ]]; then ready=1; break; fi
                    if ! kill -0 "$pid_a" 2>/dev/null; then break; fi
                    sleep 1
                done
                if [[ "$ready" != 1 ]]; then
                    printf '%s\n' 'A failed to acquire its approved GPU; B was not started.' >&2
                    wait "$pid_a" || true
                    exit 2
                fi
            fi
        done
        printf 'controller_pid=%s\nsrun_A_pid=%s\nsrun_B_pid=%s\n' "$$" "$pid_a" "$pid_b" >"$ROOT/launcher_pids.txt"
        trap 'kill "$pid_a" "$pid_b" 2>/dev/null || true' INT TERM
        fail=0
        wait "$pid_a" || fail=1
        wait "$pid_b" || fail=1
        printf '%s\n' "$fail" >"$ROOT/launcher_exit_code"
        exit "$fail"
        ;;
    help)
        printf '%s\n' \
            'CPU preparation: bash code/V4/run_v4_r77.sh prepare' \
            'After formal approval: R77_APPROVED=1 bash code/V4/run_v4_r77.sh start' \
            'Monitor: bash code/V4/run_v4_r77.sh status' \
            'Single allocated node/GPU: R77_APPROVED=1 bash code/V4/run_v4_r77.sh finish A|B' \
            'Resume the same pair: R77_APPROVED=1 R77_RESUME=1 bash code/V4/run_v4_r77.sh start'
        ;;
    *) printf 'Unknown R77 stage: %s\n' "$stage" >&2; exit 2 ;;
esac

#!/usr/bin/env bash
set -euo pipefail
source /public/home/shiys/miniconda3/etc/profile.d/conda.sh
conda activate easynco
cd /public/home/shiys/0selection2

export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg
export WANDB_MODE="${WANDB_MODE:-offline}" WANDB_PROJECT=selector
export WANDB_ENTITY=yjkds-southern-university-of-science-technology
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ROOT=/public/home/shiys/0selection2/code/V4/runs/R79_NSS_new_data_batch1024
mkdir -p "$ROOT"

approved_dual() {
    [[ "${R79_DP_APPROVED:-0}" == 1 && "${R79_R77_RELEASED:-0}" == 1 ]] || {
        printf '%s\n' 'Dual runtime awaits explicit migration license AND R77 training+TEST resource release.' >&2
        exit 2
    }
}

if [[ "${1:-}" == __dual_worker ]]; then
    approved_dual
    task="$2"
    phase="$3"
    case "$task" in
        TSP) export CUDA_VISIBLE_DEVICES=GPU-71479e7e-f8bd-e430-66ed-f8db9f0d47be,GPU-597c682a-4b7f-7609-80a6-8ced791f5c5d ;;
        CVRP) export CUDA_VISIBLE_DEVICES=GPU-cb626bdb-4319-2f5e-b91f-17fcaa44a83d,GPU-cb24f168-9f18-f0d5-6f4a-6d2406958f50 ;;
        *) exit 2 ;;
    esac
    exec python -B -u -m code.V4.r79_nss_experiment --stage "$phase" --task "$task" \
        --root "$ROOT" --micro-batch 64 --workers 2 --threads 2 --dual-runtime "${@:4}"
fi

if [[ "${1:-}" == __single4090_worker ]]; then
    approved_dual
    task="$2"
    phase="$3"
    case "$task" in
        TSP) export CUDA_VISIBLE_DEVICES=GPU-71479e7e-f8bd-e430-66ed-f8db9f0d47be ;;
        CVRP) export CUDA_VISIBLE_DEVICES=GPU-597c682a-4b7f-7609-80a6-8ced791f5c5d ;;
        *) exit 2 ;;
    esac
    exec python -B -u -m code.V4.r79_nss_experiment --stage "$phase" --task "$task" \
        --root "$ROOT" --micro-batch 64 --workers 2 --threads 2 --single4090 "${@:4}"
fi

if [[ "${1:-}" == __worker ]]; then
    task="$2"
    stage="$3"
    case "$task" in
        TSP) export CUDA_VISIBLE_DEVICES=GPU-cb24f168-9f18-f0d5-6f4a-6d2406958f50 ;;
        CVRP) export CUDA_VISIBLE_DEVICES=GPU-cb626bdb-4319-2f5e-b91f-17fcaa44a83d ;;
        *) exit 2 ;;
    esac
    exec python -B -u -m code.V4.r79_nss_experiment --stage "$stage" --task "$task" \
        --root "$ROOT" --micro-batch 64 --workers 2 --threads 2 "${@:4}"
fi

if [[ "${1:-}" == __pair || "${1:-}" == __single4090_pair ]]; then
    worker=__worker
    if [[ "$1" == __single4090_pair ]]; then approved_dual; worker=__single4090_worker; fi
    phase="$2"
    shift 2
    pids=()
    trap 'for pid in "${pids[@]}"; do kill "$pid" 2>/dev/null || true; done' EXIT
    for task in TSP CVRP; do
        bash code/V4/run_v4_r79_nss.sh "$worker" "$task" "$phase" "$@" \
            >>"$ROOT/${task}_${phase}.log" 2>&1 &
        pids+=("$!")
    done
    failed=0
    for pid in "${pids[@]}"; do
        wait "$pid" || failed=1
    done
    pids=()
    exit "$failed"
fi

stage="${1:-pipeline}"
if [[ "$stage" == dual || "$stage" == dual_resume || "$stage" == single4090 ]]; then approved_dual; fi
if [[ ( "$stage" == dual || "$stage" == dual_resume ) && -f "$ROOT/dual_migration_outcome.json" ]]; then
    printf '%s\n' 'Original ReZero DP equivalence failed; DP disabled. Use approved single4090 fallback.' >&2
    exit 2
fi
exec 9>"$ROOT/pipeline.lock"
flock -n 9 || { printf '%s\n' 'R79 NSS pipeline already running' >&2; exit 2; }
children=()
trap 'for pid in "${children[@]}"; do kill "$pid" 2>/dev/null || true; done' EXIT

launch_pair() {
    local phase="$1"
    shift
    # Slurm 20.11 does not overlap this allocation's GPU GRES between steps.
    srun --jobid=1465 --overlap --exact --nodes=1 --nodelist=gpu03 \
        --ntasks=1 --cpus-per-task=4 --mem=9G \
        bash code/V4/run_v4_r79_nss.sh __pair "$phase" "$@" &
    children+=("$!")
    local failed=0
    wait "${children[0]}" || failed=1
    children=()
    return "$failed"
}

launch_dual() {
    local phase="$1"
    shift
    local task job node pid failed=0
    children=()
    for task in TSP CVRP; do
        if [[ "$phase" == train && -f "$ROOT/$task/training_complete.json" ]]; then continue; fi
        if [[ "$task" == TSP ]]; then job=1475; node=gpu05; else job=1465; node=gpu03; fi
        srun --jobid="$job" --overlap --exact --nodes=1 --nodelist="$node" \
            --ntasks=1 --cpus-per-task=4 --mem=9G --gres=gpu:2 \
            bash code/V4/run_v4_r79_nss.sh __dual_worker "$task" "$phase" "$@" \
            >>"$ROOT/${task}_${phase}.log" 2>&1 &
        children+=("$!")
    done
    for pid in "${children[@]}"; do wait "$pid" || failed=1; done
    children=()
    return "$failed"
}

launch_single4090() {
    local phase="$1"
    shift
    srun --jobid=1475 --overlap --exact --nodes=1 --nodelist=gpu05 \
        --ntasks=1 --cpus-per-task=4 --mem=9G --gres=gpu:2 \
        bash code/V4/run_v4_r79_nss.sh __single4090_pair "$phase" "$@" &
    children+=("$!")
    local failed=0
    wait "${children[0]}" || failed=1
    children=()
    return "$failed"
}

case "$stage" in
    prepare) python -B -u -m code.V4.r79_nss_experiment --stage prepare --root "$ROOT" ;;
    pipeline)
        python -B -u -m code.V4.r79_nss_experiment --stage prepare --root "$ROOT"
        launch_pair train
        python -B -u -m code.V4.r79_nss_experiment --stage freeze --root "$ROOT"
        launch_pair evaluate
        python -B -u -m code.V4.r79_nss_experiment --stage report --root "$ROOT"
        ;;
    resume)
        launch_pair train --resume --input-path gpu_resident
        python -B -u -m code.V4.r79_nss_experiment --stage freeze --root "$ROOT"
        launch_pair evaluate
        python -B -u -m code.V4.r79_nss_experiment --stage report --root "$ROOT"
        ;;
    optimize)
        launch_pair train --resume --input-path gpu_resident --optimize
        python -B -u -m code.V4.r79_nss_experiment --stage freeze --root "$ROOT"
        launch_pair evaluate
        python -B -u -m code.V4.r79_nss_experiment --stage report --root "$ROOT"
        ;;
    grouped)
        launch_pair train --resume --input-path gpu_resident --check-grouped
        python -B -u -m code.V4.r79_nss_experiment --stage freeze --root "$ROOT"
        launch_pair evaluate
        python -B -u -m code.V4.r79_nss_experiment --stage report --root "$ROOT"
        ;;
    dual|dual_resume)
        extra=()
        if [[ "$stage" == dual ]]; then extra+=(--check-parallel); fi
        launch_dual train --resume --input-path gpu_resident "${extra[@]}"
        python -B -u -m code.V4.r79_nss_experiment --stage freeze --root "$ROOT"
        launch_dual evaluate
        python -B -u -m code.V4.r79_nss_experiment --stage report --root "$ROOT"
        ;;
    single4090)
        launch_single4090 train --resume --input-path gpu_resident
        python -B -u -m code.V4.r79_nss_experiment --stage freeze --root "$ROOT"
        launch_single4090 evaluate
        python -B -u -m code.V4.r79_nss_experiment --stage report --root "$ROOT"
        ;;
    finish)
        python -B -u -m code.V4.r79_nss_experiment --stage freeze --root "$ROOT"
        launch_pair evaluate
        python -B -u -m code.V4.r79_nss_experiment --stage report --root "$ROOT"
        ;;
    *) printf '%s\n' 'Usage: run_v4_r79_nss.sh prepare|pipeline|optimize|grouped|resume|dual|dual_resume|single4090|finish' >&2; exit 2 ;;
esac

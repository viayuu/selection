#!/usr/bin/env bash
set -euo pipefail
source /public/home/shiys/miniconda3/etc/profile.d/conda.sh
conda activate easynco
cd /public/home/shiys/0selection2

export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg
export PYTHONPATH=/public/home/shiys/easynco_v3_bridge/exports/train_100k_20261002/python_compat:/public/home/shiys/0selection2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE="${WANDB_MODE:-offline}" WANDB_PROJECT=selector
export WANDB_ENTITY=yjkds-southern-university-of-science-technology
export NCCL_IB_DISABLE=1 NCCL_P2P_DISABLE=1
export NCCL_SOCKET_FAMILY=AF_INET NCCL_SOCKET_IFNAME='=bond0'
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

ROOT="${R76_ROOT:-/public/home/shiys/0selection2/code/V4/runs/R76_R45A_single_reld_mt}"
DATA="${R76_DATA_ROOT:-/public/home/shiys/selection_data_new_only_B_filtered_single_reld_mt_seed2_20261014}"
CACHE="$ROOT/cache"
LOCAL_CACHE="${R76_LOCAL_CACHE:-/tmp/shiys_R76_R45A_single_reld_mt_cache}"
MASTER_ADDR=12.12.12.26
MASTER_PORT="${R76_MASTER_PORT:-29776}"
GPU4090=(GPU-71479e7e-f8bd-e430-66ed-f8db9f0d47be GPU-597c682a-4b7f-7609-80a6-8ced791f5c5d)
GPU3090=(GPU-cb24f168-9f18-f0d5-6f4a-6d2406958f50 GPU-cb626bdb-4319-2f5e-b91f-17fcaa44a83d)
mkdir -p "$ROOT"

if [[ "${1:-}" == __cache_node ]]; then
    first="$2"
    split="$3"
    bash code/V4/run_v4_r76.sh __cache_worker "$first" "$split" >"$ROOT/cache_${split}_rank${first}.log" 2>&1 &
    p0="$!"
    second="$((first+1))"
    bash code/V4/run_v4_r76.sh __cache_worker "$second" "$split" >"$ROOT/cache_${split}_rank${second}.log" 2>&1 &
    p1="$!"
    trap 'kill "$p0" "$p1" 2>/dev/null || true' EXIT
    if ! wait -n; then
        kill "$p0" "$p1" 2>/dev/null || true
        wait "$p0" "$p1" 2>/dev/null || true
        exit 1
    fi
    failed=0
    wait "$p0" || failed=1
    wait "$p1" || failed=1
    exit "$failed"
fi

if [[ "${1:-}" == __cache_worker ]]; then
    rank="$2"
    split="$3"
    if (( rank < 2 )); then
        export CUDA_VISIBLE_DEVICES="${GPU4090[$rank]}"
    else
        export CUDA_VISIBLE_DEVICES="${GPU3090[$((rank-2))]}"
    fi
    splits=("$split")
    if [[ "$split" == train_val ]]; then splits=(train val); fi
    exec python -B -u -m code.V4.r76_cache --root "$ROOT" --data-root "$DATA" \
        --cache-root "$CACHE" --worker-rank "$rank" --workers 4 --device cuda:0 --splits "${splits[@]}"
fi

if [[ "${1:-}" == __stage_local ]]; then
    mkdir -p "$LOCAL_CACHE"
    for split in "${@:2}"; do
        if [[ ! -d "$LOCAL_CACHE/$split" ]]; then
            cp -a --reflink=auto "$CACHE/$split" "$LOCAL_CACHE/"
        else
            rsync -a "$CACHE/$split/" "$LOCAL_CACHE/$split/"
        fi
    done
    printf 'Local cache ready on %s: %s\n' "$(hostname -s)" "$LOCAL_CACHE"
    exit 0
fi

if [[ "${1:-}" == __capacity ]]; then
    if [[ "$2" == 4090 ]]; then
        export CUDA_VISIBLE_DEVICES="${GPU4090[0]}"
    else
        export CUDA_VISIBLE_DEVICES="${GPU3090[0]}"
    fi
    exec python -B -u -m code.V4.r76_experiment --stage capacity --root "$ROOT" \
        --data-root "$DATA" --cache-root "$LOCAL_CACHE"
fi

if [[ "${1:-}" == __ddp ]]; then
    stage="$2"
    node_rank="$3"
    if [[ "$node_rank" == 0 ]]; then
        export CUDA_VISIBLE_DEVICES="${GPU4090[0]},${GPU4090[1]}"
    else
        export CUDA_VISIBLE_DEVICES="${GPU3090[0]},${GPU3090[1]}"
    fi
    exec torchrun --nnodes=2 --nproc_per_node=2 --node_rank="$node_rank" \
        --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT" \
        -m code.V4.r76_experiment --stage "$stage" --root "$ROOT" --data-root "$DATA" \
        --cache-root "$LOCAL_CACHE" --batch-config "$ROOT/batch_config.json" --logical-batch-size 9856 --epochs 40
fi

stage="${1:-pipeline}"
exec 9>"$ROOT/pipeline.lock"
flock -n 9 || { printf '%s\n' 'An R76 pipeline already owns this run.' >&2; exit 2; }
children=()
trap 'for pid in "${children[@]}"; do kill "$pid" 2>/dev/null || true; done' EXIT

launch() {
    local job="$1" node="$2" cpus="$3" memory="$4" log="$5"
    shift 5
    srun --jobid="$job" --overlap --exact --nodes=1 --nodelist="$node" \
        --ntasks=1 --cpus-per-task="$cpus" --mem="$memory" \
        bash code/V4/run_v4_r76.sh "$@" >"$ROOT/$log" 2>&1 &
    children+=("$!")
}

wait_children() {
    local failed=0 pid
    if ! wait -n; then
        for pid in "${children[@]}"; do kill "$pid" 2>/dev/null || true; done
        wait "${children[@]}" 2>/dev/null || true
        children=()
        return 1
    fi
    for pid in "${children[@]}"; do
        if ! wait "$pid"; then failed=1; fi
    done
    children=()
    if (( failed )); then return 1; fi
}

prepare() {
    local split="$1"
    launch 1475 gpu05 8 18G "cache_${split}_4090.log" __cache_node 0 "$split"
    launch 1465 gpu03 8 18G "cache_${split}_3090.log" __cache_node 2 "$split"
    wait_children
    local splits=("$split")
    if [[ "$split" == train_val ]]; then splits=(train val); fi
    python -B -u -m code.V4.r76_cache --root "$ROOT" --data-root "$DATA" \
        --cache-root "$CACHE" --splits "${splits[@]}" --finalize
    launch 1475 gpu05 4 8G "local_stage_${split}_4090.log" __stage_local "${splits[@]}"
    launch 1465 gpu03 4 8G "local_stage_${split}_3090.log" __stage_local "${splits[@]}"
    wait_children
}

capacity() {
    launch 1475 gpu05 4 12G capacity_4090.log __capacity 4090
    launch 1465 gpu03 4 12G capacity_3090.log __capacity 3090
    wait_children
    python -B -u -m code.V4.r76_experiment --stage batch_config --root "$ROOT" \
        --data-root "$DATA" --cache-root "$CACHE"
}

distributed() {
    launch 1475 gpu05 8 18G "${1}_4090.log" __ddp "$1" 0
    launch 1465 gpu03 8 18G "${1}_3090.log" __ddp "$1" 1
    wait_children
}

case "$stage" in
    cache) prepare train_val ;;
    capacity) capacity ;;
    train) distributed train ;;
    finish)
        distributed train
        [[ -f "$ROOT/final_test_lock.json" ]] || { printf '%s\n' 'Training did not publish a final TEST lock.' >&2; exit 2; }
        prepare test
        distributed test
        python -B -u -m code.V4.r76_analysis --root "$ROOT"
        ;;
    test)
        [[ -f "$ROOT/final_test_lock.json" ]] || { printf '%s\n' 'TEST requires the selected checkpoint lock.' >&2; exit 2; }
        prepare test
        distributed test
        ;;
    analysis) python -B -u -m code.V4.r76_analysis --root "$ROOT" ;;
    pipeline)
        prepare train_val
        capacity
        distributed train
        [[ -f "$ROOT/final_test_lock.json" ]] || { printf '%s\n' 'Training did not publish a final TEST lock.' >&2; exit 2; }
        prepare test
        distributed test
        python -B -u -m code.V4.r76_analysis --root "$ROOT"
        ;;
    *) printf 'Unknown stage: %s\n' "$stage" >&2; exit 2 ;;
esac

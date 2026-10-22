#!/usr/bin/env bash
set -euo pipefail
source /public/home/shiys/miniconda3/etc/profile.d/conda.sh
conda activate easynco
cd /public/home/shiys/0selection2
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
ROOT=/public/home/shiys/0selection2/code/V4/runs/R94_pretrained_node_fusion/timing_correction
GPU=GPU-597c682a-4b7f-7609-80a6-8ced791f5c5d
CONTAINER=76045731f6b453443fe18a9bce671599bdbf17c2713b8edcdd090925eea14bf6
if [[ "${1:-controller}" == controller ]]; then
    srun --jobid=1475 --overlap --exact --gres=none -N1 -n1 \
        --nodelist=gpu05 --cpus-per-task=4 --mem=9G \
        bash /public/home/shiys/0selection2/code/V4/run_r94_timing_correction.sh entry \
        > "$ROOT/measure_launch.log" 2>&1
    python -B -u -m code.V4.diagnostics.r94_online_timing_recovery report \
        > "$ROOT/report_update.log" 2>&1
elif [[ "$1" == entry ]]; then
    exec sudo -n /usr/bin/ai_docker exec -u 2022 \
        -e CUDA_VISIBLE_DEVICES="$GPU" -e SLURM_JOB_ID=1475 \
        "$CONTAINER" bash /public/home/shiys/0selection2/code/V4/run_r94_timing_correction.sh worker
elif [[ "$1" == worker ]]; then
    grep -q "$CONTAINER" /proc/self/cgroup
    [[ "$CUDA_VISIBLE_DEVICES" == "$GPU" ]]
    exec flock -n "/public/home/shiys/0selection2/code/V4/runs/R81_method_conditioned_auc/queue_gpu_${GPU}.json" \
        taskset -c 4,5,72,73 python -B -u -m code.V4.diagnostics.r94_online_timing_recovery \
        measure --arms A B --device cuda:0 --gpu-authorized
else
    exit 2
fi

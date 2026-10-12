#!/usr/bin/env bash
set -euo pipefail
source /public/home/shiys/miniconda3/etc/profile.d/conda.sh
conda activate easynco
cd /public/home/shiys/0selection2
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export WANDB_MODE=offline MPLBACKEND=Agg PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=/public/home/shiys/easynco_v3_bridge/exports/train_100k_20261002/python_compat:/public/home/shiys/0selection2
unset CUDA_MPS_PIPE_DIRECTORY CUDA_MPS_LOG_DIRECTORY CUDA_MPS_ACTIVE_THREAD_PERCENTAGE

stage="${1:-plan}"
if [[ $# -gt 0 ]]; then shift; fi
case "$stage" in
    plan|analyze|tests|--help|-h) export CUDA_VISIBLE_DEVICES="" ;;
    prepare|train|finalize-b15|test)
        required_baton=recurrent-workspaces
        if [[ "$stage" == test ]]; then required_baton=locked-test; fi
        if [[ "$stage" == finalize-b15 ]]; then required_baton=b15-finalize; fi
        [[ "${R63_PARENT_GPU_BATON:-}" == "$required_baton" ]] || {
            printf '%s\n' 'R63 requires the explicit parent GPU baton.' >&2
            exit 2
        }
        if [[ "$(hostname -s)" != gpu03 || "${SLURM_JOB_ID:-}" != 1465 ]]; then
            exec srun --jobid=1465 --overlap --exact --nodes=1 --nodelist=gpu03 \
                --ntasks=1 --cpus-per-task=4 --mem=16G \
                bash code/V4/run_v4_r63.sh "$stage" "$@"
        fi
        export CUDA_VISIBLE_DEVICES=GPU-cb626bdb-4319-2f5e-b91f-17fcaa44a83d
        if [[ "$stage" == finalize-b15 ]]; then
            printf '[R63 B15 finalization launch] %s step=%s.%s UUID=%s\n' \
                "$(date --iso-8601=seconds)" "$SLURM_JOB_ID" "${SLURM_STEP_ID:-}" "$CUDA_VISIBLE_DEVICES"
        else
        python - <<'PY'
import csv
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

uuid = os.environ['CUDA_VISIBLE_DEVICES']
gpu = subprocess.check_output(['nvidia-smi', '--id=' + uuid,
    '--query-gpu=uuid,name,memory.used', '--format=csv,noheader,nounits'], text=True)
apps = subprocess.check_output(['nvidia-smi', '--id=' + uuid,
    '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader,nounits'], text=True)
root = Path('code/V4/runs/R63_recurrent_workspaces')
root.mkdir(parents=True, exist_ok=True)
stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
(root / ('availability_' + stamp + '.log')).write_text(gpu + apps)
rows = list(csv.reader(gpu.splitlines()))
if len(rows) != 1:
    raise SystemExit('Assigned GPU query did not return exactly one device')
actual, name, memory = [value.strip() for value in rows[0]]
if actual != uuid or '3090' not in name or int(memory) >= 500:
    raise SystemExit('Assigned RTX3090 is not idle; wait, never switch GPUs or touch queues')
if any(row and row[0].strip() == uuid for row in csv.reader(apps.splitlines())):
    raise SystemExit('Assigned GPU has a compute client; no launch')
print('[R63 launch] ' + datetime.now().astimezone().isoformat() +
      ' step=' + os.environ.get('SLURM_JOB_ID', '') + '.' + os.environ.get('SLURM_STEP_ID', '') +
      ' ' + gpu.strip(), flush=True)
PY
        fi
        ;;
    *) printf 'Unknown R63 stage: %s\n' "$stage" >&2; exit 2 ;;
esac

# After parent baton: tmux new-session -d -s r63_train 'env R63_PARENT_GPU_BATON=recurrent-workspaces bash code/V4/run_v4_r63.sh train --resume'
# Only after both arms complete and parent freezes final_test_lock.json:
# R63_PARENT_GPU_BATON=locked-test bash code/V4/run_v4_r63.sh test
# Authorized boundary recovery only: R63_PARENT_GPU_BATON=b15-finalize bash code/V4/run_v4_r63.sh finalize-b15 --resume
if [[ "$stage" == tests ]]; then
    python -B -m unittest code.V4.test_r63_experiment -v "$@"
elif [[ "$stage" == --help || "$stage" == -h ]]; then
    python -B -m code.V4.r63_experiment --help
else
    python -B -u -m code.V4.r63_experiment --stage "$stage" "$@"
fi

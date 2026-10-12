#!/usr/bin/env bash
set -euo pipefail
source /public/home/shiys/miniconda3/etc/profile.d/conda.sh
conda activate easynco
cd /public/home/shiys/0selection2
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export WANDB_MODE=offline WANDB_PROJECT=selector
export WANDB_ENTITY=yjkds-southern-university-of-science-technology
export MPLBACKEND=Agg PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=/public/home/shiys/easynco_v3_bridge/exports/train_100k_20261002/python_compat:/public/home/shiys/0selection2

stage="${1:-train}"
if [[ $# -gt 0 ]]; then shift; fi
case "$stage" in
    cpu-check) export CUDA_VISIBLE_DEVICES="" ;;
    train)
        if [[ "$(hostname -s)" != gpu03 || "${SLURM_JOB_ID:-}" != 1465 ]]; then
            exec srun --jobid=1465 --overlap --exact --nodes=1 --nodelist=gpu03 \
                --ntasks=1 --cpus-per-task=4 --mem=16G \
                bash code/V4/run_v4_r67.sh "$stage" "$@"
        fi
        export CUDA_VISIBLE_DEVICES=GPU-cb626bdb-4319-2f5e-b91f-17fcaa44a83d
        root=code/V4/runs/R67_supervised_contrastive_full18
        mkdir -p "$root"
        exec 9>"$root/launcher.lock"
        flock -n 9 || { printf '%s\n' 'An R67 launcher already owns this run.' >&2; exit 2; }
        python -B - <<'PY'
import csv
import json
import os
import socket
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

uuid = os.environ['CUDA_VISIBLE_DEVICES']
root = Path('code/V4/runs/R67_supervised_contrastive_full18')
if (root/'formal_started.json').exists():
    raise SystemExit('R67 formal training already started; preserve artifacts and do not replay')
while True:
    gpu = subprocess.check_output(['nvidia-smi', '--id='+uuid,
        '--query-gpu=index,uuid,name,memory.used,memory.total', '--format=csv,noheader,nounits'], text=True)
    apps = subprocess.check_output(['nvidia-smi', '--id='+uuid,
        '--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory', '--format=csv,noheader,nounits'], text=True)
    rows = list(csv.reader(gpu.splitlines()))
    if len(rows) != 1:
        raise SystemExit('The assigned UUID must identify exactly one GPU')
    index, actual, name, used, total = [value.strip() for value in rows[0]]
    if actual != uuid or index != '1' or '3090' not in name:
        raise SystemExit('Assigned UUID/index identity mismatch; never switch devices')
    clients = [row for row in csv.reader(apps.splitlines()) if row and row[0].strip() == uuid]
    idle = int(used) < 500 and not clients
    receipt = dict(checked_utc=datetime.now(timezone.utc).isoformat(), host=socket.gethostname(),
        slurm_step=os.environ.get('SLURM_JOB_ID', '')+'.'+os.environ.get('SLURM_STEP_ID', ''),
        gpu_uuid=actual, gpu_index=int(index), gpu_name=name, memory_used_mib=int(used),
        memory_total_mib=int(total), compute_apps=clients, idle=idle, tmux_session='r67_train',
        only_assigned_uuid_queried=True)
    with (root/'availability.jsonl').open('a') as stream:
        stream.write(json.dumps(receipt)+'\n')
    if idle:
        (root/'launch_receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
        print('[R67 assigned-GPU formal launch] '+json.dumps(receipt), flush=True)
        break
    print('[R67 wait] assigned UUID occupied; waiting15s', flush=True)
    time.sleep(15)
PY
        ;;
    *) printf 'Unknown R67 stage: %s\n' "$stage" >&2; exit 2 ;;
esac

exec python -B -u -m code.V4.r67_experiment --stage "$stage" "$@"

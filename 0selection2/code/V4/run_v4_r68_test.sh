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

[[ $# == 0 && "${R68_PARENT_GPU_BATON:-}" == locked-test ]] || {
    printf '%s\n' 'R68 TEST requires future explicit parent locked-test authorization and fixed arguments.' >&2
    exit 2
}
root=code/V4/runs/R68_sharpness_aware_finetuning
[[ -f "$root/final_test_lock.json" && -f "$root/training_terminal_receipt.json" && ! -e "$root/final_test_started.json" ]] || {
    printf '%s\n' 'Require parent lock/actual training terminal0 receipt; preserve the one-time TEST marker.' >&2
    exit 2
}
if [[ "$(hostname -s)" != gpu03 || "${SLURM_JOB_ID:-}" != 1465 ]]; then
    exec srun --jobid=1465 --overlap --exact --nodes=1 --nodelist=gpu03 \
        --ntasks=1 --cpus-per-task=4 --mem=16G bash code/V4/run_v4_r68_test.sh
fi
export CUDA_VISIBLE_DEVICES=GPU-cb626bdb-4319-2f5e-b91f-17fcaa44a83d
export R68_TMUX_SESSION=r68_final_test
exec 9>"$root/test_launcher.lock"
flock -n 9 || { printf '%s\n' 'An R68 TEST launcher already owns this run.' >&2; exit 2; }
python -B - <<'PY'
import csv
import json
import os
import socket
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from code.V4.r68_frozen_evaluation import validate_lock

root = Path('code/V4/runs/R68_sharpness_aware_finetuning').resolve()
validate_lock(root)
uuid = os.environ['CUDA_VISIBLE_DEVICES']
while True:
    gpu = subprocess.check_output(['nvidia-smi', '--id='+uuid,
        '--query-gpu=index,uuid,name,memory.used,memory.total', '--format=csv,noheader,nounits'], text=True)
    apps = subprocess.check_output(['nvidia-smi', '--id='+uuid,
        '--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory', '--format=csv,noheader,nounits'], text=True)
    rows = list(csv.reader(gpu.splitlines()))
    if len(rows) != 1:
        raise SystemExit('Assigned UUID must identify exactly one GPU')
    index, actual, name, memory, total = [v.strip() for v in rows[0]]
    if actual != uuid or index != '1' or '3090' not in name:
        raise SystemExit('Assigned UUID/index identity mismatch; never switch devices')
    clients = [row for row in csv.reader(apps.splitlines()) if row and row[0].strip() == uuid]
    idle = int(memory) < 500 and not clients
    receipt = dict(checked_utc=datetime.now(timezone.utc).isoformat(), host=socket.gethostname(),
        slurm_step=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'], gpu_uuid=actual,
        gpu_index=int(index), gpu_name=name, memory_used_mib=int(memory), memory_total_mib=int(total),
        compute_apps=clients, idle=idle, tmux_session='r68_final_test', only_assigned_uuid_queried=True)
    with (root/'test_availability.jsonl').open('a') as stream:
        stream.write(json.dumps(receipt)+'\n')
    if idle:
        (root/'test_launch_receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
        print('[R68 assigned-GPU locked TEST launch] '+json.dumps(receipt), flush=True)
        break
    print('[R68 TEST wait] assigned UUID occupied; waiting30s', flush=True)
    time.sleep(30)
PY
set +e
python -B -u -m code.V4.r68_frozen_evaluation --stage test
test_status=$?
set -e
python -B - "$test_status" <<'PY'
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

receipt = dict(finished_utc=datetime.now(timezone.utc).isoformat(), returncode=int(sys.argv[1]),
    slurm_step=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'], tmux_session='r68_final_test',
    parameter_mode='ordinary AdamW weights', perturbed_weights=False,
    scope='actual evaluator exit; parent separately records actual Slurm/tmux terminal evidence',
    duplicate_attempt_authorized=False)
Path('code/V4/runs/R68_sharpness_aware_finetuning/test_process_receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
PY
exit "$test_status"

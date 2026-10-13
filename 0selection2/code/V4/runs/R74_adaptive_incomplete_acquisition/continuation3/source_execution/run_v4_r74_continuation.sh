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
if [[ -n "${TMUX_PANE:-}" && -z "${R74_TMUX_PANE_ID:-}" ]]; then
  export R74_TMUX_PANE_ID="$TMUX_PANE"
  export R74_TMUX_PANE_PID="$(tmux display-message -p -t "$TMUX_PANE" '#{pane_pid}')"
  export R74_TMUX_SERVER_HOST="$(hostname -s)"
fi
if [[ "$(hostname -s)" != gpu03 || "${SLURM_JOB_ID:-}" != 1465 ]]; then
  exec srun --jobid=1465 --overlap --exact --nodes=1 --nodelist=gpu03 \
    --ntasks=1 --cpus-per-task=4 --mem=16G bash code/V4/run_v4_r74_continuation.sh
fi
export CUDA_VISIBLE_DEVICES=GPU-cb626bdb-4319-2f5e-b91f-17fcaa44a83d
root=code/V4/runs/R74_adaptive_incomplete_acquisition
exec 9>"$root/launcher.lock"
flock -n 9 || { printf '%s\n' 'Original R74 execution is already owned.' >&2; exit 2; }
python -B - <<'PY'
import csv
import json
import os
import socket
import subprocess
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

root = Path('code/V4/runs/R74_adaptive_incomplete_acquisition')
record = root / 'continuation3'
uuid = os.environ['CUDA_VISIBLE_DEVICES']
started = time.monotonic()
bootstrap_started_utc = datetime.now(timezone.utc).isoformat()
try:
    record.mkdir(exist_ok=True)
    if any((record/name).exists() for name in ('launch_receipt.json', 'formal_started.json',
            'execution_amendment.json', 'failures.jsonl', 'bootstrap_failures.jsonl')):
        raise RuntimeError('Single continuation already attempted; no retry')
    original = json.loads((root/'pipeline_status.json').read_text())
    if original['stage'] != 'failed_preserved_no_retry' or original['slurm_step'] != '1465.111':
        raise RuntimeError('Original failure must remain preserved')
    previous = json.loads((root/'continuation2/pipeline_status.json').read_text())
    if previous['stage'] != 'failed_preserved_no_retry' or previous['slurm_step'] != '1465.113':
        raise RuntimeError('Second continuation failure must remain preserved')
    authorization = json.loads((root/'runtime_repair3_authorization.json').read_text())
    if authorization['charged_prior_Slurm_wall_seconds'] != 1461:
        raise RuntimeError('Third execution continuation authorization differs')
    while True:
        gpu = subprocess.check_output(['nvidia-smi', '--id='+uuid,
            '--query-gpu=index,uuid,name,memory.used,memory.total', '--format=csv,noheader,nounits'], text=True)
        apps = subprocess.check_output(['nvidia-smi', '--id='+uuid,
            '--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory', '--format=csv,noheader,nounits'], text=True)
        rows = list(csv.reader(gpu.splitlines()))
        if len(rows) != 1:
            raise RuntimeError('Assigned UUID must identify exactly one GPU')
        index, actual, name, used, total = [value.strip() for value in rows[0]]
        if actual != uuid or index != '1':
            raise RuntimeError('Assigned UUID/index mismatch; no switching')
        clients = [row for row in csv.reader(apps.splitlines()) if row and row[0].strip() == uuid]
        receipt = dict(utc=datetime.now(timezone.utc).isoformat(), host=socket.gethostname(),
            slurm_step=os.environ.get('SLURM_JOB_ID', '')+'.'+os.environ.get('SLURM_STEP_ID', ''),
            gpu_uuid=actual, gpu_index=int(index), gpu_name=name, memory_used_mib=int(used),
            memory_total_mib=int(total), compute_apps=clients, idle=int(used)<500 and not clients,
            only_assigned_uuid_queried=True, tmux_session='r74_continuation3',
            tmux_pane=os.environ.get('R74_TMUX_PANE_ID', ''),
            tmux_pane_pid=os.environ.get('R74_TMUX_PANE_PID', ''),
            tmux_server_host=os.environ.get('R74_TMUX_SERVER_HOST', ''), bootstrap_pid=os.getpid(),
            bootstrap_started_utc=bootstrap_started_utc, bootstrap_seconds=time.monotonic()-started,
            original_step='1465.111', original_status='FAILED 1:0', first_continuation_step='1465.112',
            first_continuation_status='FAILED 1:0', second_continuation_step='1465.113',
            second_continuation_status='FAILED 1:0', charged_prior_Slurm_wall_seconds=1461, automatic_restart=False)
        with (record/'availability.jsonl').open('a') as stream:
            stream.write(json.dumps(receipt)+'\n')
        if receipt['idle']:
            (record/'launch_receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
            print('[R74 continuation3 assigned-UUID launch] '+json.dumps(receipt), flush=True)
            break
        print('[R74 continuation3 wait] assigned UUID occupied; waiting30s on same device', flush=True)
        time.sleep(30)
except BaseException as error:
    failure = dict(utc=datetime.now(timezone.utc).isoformat(), error_type=type(error).__name__,
        error=str(error), traceback=traceback.format_exc(), automatic_restart=False, gpu_uuid=uuid,
        slurm_step=os.environ.get('SLURM_JOB_ID', '')+'.'+os.environ.get('SLURM_STEP_ID', ''))
    with (record/'bootstrap_failures.jsonl').open('a') as stream:
        stream.write(json.dumps(failure)+'\n')
    raise
PY
exec python -B -u -m code.V4.r74_continuation

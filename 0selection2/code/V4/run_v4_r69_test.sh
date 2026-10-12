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

[[ $# == 0 && "${R69_PARENT_GPU_BATON:-}" == locked-test ]] || {
    printf '%s\n' 'Future R69 TEST requires explicit parent locked-test baton and fixed arguments.' >&2
    exit 2
}
root=code/V4/runs/R69_task_specific_finetuning
[[ -f "$root/final_test_lock.json" && -f "$root/training_terminal_receipt.json" && ! -e "$root/final_test_started.json" ]] || {
    printf '%s\n' 'Require completed original TRAIN1465.101 and parent validation lock; preserve the exclusive TEST marker.' >&2
    exit 2
}
if [[ -z "${R69_TEST_TMUX_PANE_ID:-}" ]]; then
    [[ -n "${TMUX_PANE:-}" ]] || { printf '%s\n' 'Use owned tmux r69_final_test.' >&2; exit 2; }
    [[ "$(tmux display-message -p -t "$TMUX_PANE" '#{session_name}')" == r69_final_test ]] || {
        printf '%s\n' 'Only owned tmux r69_final_test may dispatch this future TEST.' >&2
        exit 2
    }
    export R69_TEST_TMUX_PANE_ID="$TMUX_PANE"
    export R69_TEST_TMUX_PANE_PID="$(tmux display-message -p -t "$TMUX_PANE" '#{pane_pid}')"
    export R69_TEST_TMUX_SERVER_HOST="$(hostname -s)"
fi
export R69_TMUX_SESSION=r69_final_test
if [[ "$(hostname -s)" != gpu03 || "${SLURM_JOB_ID:-}" != 1465 ]]; then
    exec srun --jobid=1465 --overlap --exact --nodes=1 --nodelist=gpu03 \
        --ntasks=1 --cpus-per-task=4 --mem=16G bash code/V4/run_v4_r69_test.sh
fi
export CUDA_VISIBLE_DEVICES=GPU-cb626bdb-4319-2f5e-b91f-17fcaa44a83d
exec 9>"$root/test_launcher.lock"
flock -n 9 || { printf '%s\n' 'An R69 TEST launcher already owns this run.' >&2; exit 2; }
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

from code.V4.r69_frozen_evaluation import validate_lock

root = Path('code/V4/runs/R69_task_specific_finetuning').resolve()
uuid = os.environ['CUDA_VISIBLE_DEVICES']
try:
    if (root/'final_test_started.json').exists():
        raise RuntimeError('The one new B TEST marker already exists; no retry or replay')
    validate_lock(root, check_checkpoint_files=True)
    while True:
        gpu = subprocess.check_output(['nvidia-smi', '--id='+uuid,
            '--query-gpu=index,uuid,name,memory.used,memory.total', '--format=csv,noheader,nounits'], text=True)
        apps = subprocess.check_output(['nvidia-smi', '--id='+uuid,
            '--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory', '--format=csv,noheader,nounits'], text=True)
        rows = list(csv.reader(gpu.splitlines()))
        if len(rows) != 1:
            raise RuntimeError('Assigned UUID must identify exactly one GPU')
        index, actual, name, memory, total = [value.strip() for value in rows[0]]
        if actual != uuid or index != '1':
            raise RuntimeError('Assigned UUID/index identity mismatch; never switch devices')
        clients = [row for row in csv.reader(apps.splitlines()) if row and row[0].strip() == uuid]
        idle = int(memory) < 500 and not clients
        receipt = dict(checked_utc=datetime.now(timezone.utc).isoformat(), host=socket.gethostname(),
            slurm_step=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],
            gpu_uuid=actual, gpu_index=int(index), gpu_name=name, memory_used_mib=int(memory),
            memory_total_mib=int(total), compute_apps=clients, idle=idle, tmux_session='r69_final_test',
            tmux_pane=os.environ['R69_TEST_TMUX_PANE_ID'], tmux_pane_pid=os.environ['R69_TEST_TMUX_PANE_PID'],
            tmux_server_host=os.environ['R69_TEST_TMUX_SERVER_HOST'], only_assigned_uuid_queried=True,
            parent_baton='locked-test', new_evaluated_arms=['B'], reused_arms=['A'])
        with (root/'test_availability.jsonl').open('a') as stream:
            stream.write(json.dumps(receipt)+'\n')
        if idle:
            (root/'test_launch_receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
            print('[R69 assigned-UUID one NEW B TEST launch] '+json.dumps(receipt), flush=True)
            break
        print('[R69 TEST wait] assigned UUID busy; waiting30s; no device switching', flush=True)
        time.sleep(30)
except BaseException as error:
    failure = dict(recorded_utc=datetime.now(timezone.utc).isoformat(), error_type=type(error).__name__,
        message=str(error), traceback=traceback.format_exc(), gpu_uuid=uuid,
        slurm_step=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'], automatic_retry=False)
    with (root/'test_bootstrap_failures.jsonl').open('a') as stream:
        stream.write(json.dumps(failure)+'\n')
    raise
PY
set +e
python -B -u -m code.V4.r69_frozen_evaluation --stage test
test_status=$?
set -e
python -B - "$test_status" <<'PY'
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

root = Path('code/V4/runs/R69_task_specific_finetuning')
receipt = dict(finished_utc=datetime.now(timezone.utc).isoformat(), returncode=int(sys.argv[1]),
    slurm_step=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'], tmux_session='r69_final_test',
    tmux_pane=os.environ['R69_TEST_TMUX_PANE_ID'], new_evaluated_arms=['B'], reused_arms=['A'],
    new_test_attempts_started={'A': 0, 'B': int((root/'final_test_started.json').exists())},
    new_tests_completed={'A': 0, 'B': int(int(sys.argv[1]) == 0 and (root/'test_results_B.json').is_file())},
    historical_original_A_tests_reused=1,
    parameter_mode='ordinary AdamW weights', parameter_averaging=False, perturbed_weights=False,
    scope='actual evaluator exit ONLY; parent separately records original Slurm COMPLETED/0:0 and owned tmux gone',
    process_receipt_is_scheduler_evidence=False, automatic_retry=False, reselection=False)
path = root/'test_process_receipt.json'
with path.open('x') as stream:
    json.dump(receipt, stream, indent=2)
    stream.write('\n')
PY
exit "$test_status"

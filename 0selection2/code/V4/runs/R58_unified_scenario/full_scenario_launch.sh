#!/usr/bin/env bash
set -euo pipefail
ROOT=/public/home/shiys/0selection2/code/V4/runs/R58_unified_scenario
if [[ -e "$ROOT/DEFERRED.md" ]]; then
    printf '%s\n' 'R58 is deferred by user: DEFERRED.md blocks all launcher and worker execution.' >&2
    exit 64
fi
source /public/home/shiys/miniconda3/etc/profile.d/conda.sh
conda activate easynco
cd /public/home/shiys/0selection2

[[ "${R58_PARENT_GPU_BATON:-}" == full-scenario ]] || {
    printf '%s\n' 'Explicit parent full-scenario GPU baton required.' >&2
    exit 64
}
[[ -z "$(compgen -v CUDA_MPS_ || true)" ]] || {
    printf '%s\n' 'Full R58 must not inherit CUDA_MPS_* configuration.' >&2
    exit 64
}
export CUDA_VISIBLE_DEVICES=GPU-cb626bdb-4319-2f5e-b91f-17fcaa44a83d
export R58_GPU_UUID="$CUDA_VISIBLE_DEVICES" R58_SLURM_JOB_ID=1465
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 WANDB_MODE=offline
export PYTHONPATH=/public/home/shiys/easynco_v3_bridge/exports/train_100k_20261002/python_compat:/public/home/shiys/0selection2
export WANDB_DIR="$ROOT" MPLCONFIGDIR="$ROOT/mplconfig" R58_ROOT="$ROOT"

if [[ "${1:-}" == --worker ]]; then
    [[ "$#" == 1 && "${SLURM_JOB_ID:-}" == 1465 && "$(hostname -s)" == gpu03 &&
        "${CONDA_DEFAULT_ENV:-}" == easynco && -n "${R58_FULL_LAUNCHER_PID:-}" ]]
    exec python -u - <<'WORKER_PY'
import csv
import hashlib
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

root = Path(os.environ['R58_ROOT'])
runtime_path = root / 'full_scenario_runtime.json'
with runtime_path.open('x') as stream:
    json.dump(dict(status='starting', pid=os.getpid(), started=time.time()), stream)

def save(name, value):
    path = root / name
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)

def capture(command):
    result = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    evidence['commands'].append(dict(argv=command, returncode=result.returncode, output=result.stdout))
    if result.returncode:
        raise RuntimeError(f'Read-only command failed: {command!r}')
    return result.stdout

started = time.time()
environment = {key: os.environ.get(key) for key in (
    'CONDA_DEFAULT_ENV', 'CUDA_VISIBLE_DEVICES', 'R58_GPU_UUID', 'PYTHONPATH',
    'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS',
    'WANDB_MODE', 'WANDB_DIR', 'PYTHONDONTWRITEBYTECODE', 'SLURM_JOB_ID', 'SLURM_STEP_ID',
    'SLURM_CPUS_PER_TASK', 'SLURM_MEM_PER_NODE', 'R58_FULL_TMUX_SESSION', 'R58_FULL_LAUNCHER_PID')}
evidence = dict(started=started, status='running', passed=False, commands=[],
                hostname=socket.gethostname(), environment=environment)
runtime = dict(started=started, pid=os.getpid(), job_id=os.environ['SLURM_JOB_ID'],
               step_id=os.environ.get('SLURM_STEP_ID'), tmux_session='r58_full_scenario',
               environment=environment, automatic_retry=False, mps=False)
code, error = 1, None
try:
    launch = json.loads((root / 'full_scenario_started.json').read_text())
    if (str(launch['launcher_pid']) != os.environ['R58_FULL_LAUNCHER_PID'] or
            os.environ.get('R58_FULL_TMUX_SESSION') != 'r58_full_scenario' or
            os.environ.get('SLURM_CPUS_PER_TASK') != '4' or
            os.environ.get('SLURM_MEM_PER_NODE') != '16384'):
        raise ValueError('Worker does not match the singleton 4CPU/16G launcher')
    authorization = root / 'compute_authorization.json'
    auth = json.loads(authorization.read_text())
    target = os.environ['CUDA_VISIBLE_DEVICES']
    if auth.get('gpu_hour_limit', 100) is not None or auth.get('device_uuid') != target:
        raise ValueError('Missing explicit no-cap authorization for this GPU')
    evidence['authorization_sha256'] = hashlib.sha256(authorization.read_bytes()).hexdigest()
    processes = capture(['ps', '-eo', 'user=,pid=,comm=,args='])
    daemons = [line for line in processes.splitlines()
               if len(line.split(None, 3)) >= 3 and
               line.split(None, 3)[2].startswith('nvidia-cuda-mps')]
    evidence['all_user_mps_processes'] = daemons
    if daemons:
        raise RuntimeError('Existing all-user MPS service: do not launch or modify it')
    gpu = capture(['nvidia-smi', '-i', target,
                   '--query-gpu=uuid,name,memory.used,utilization.gpu,compute_mode',
                   '--format=csv,noheader,nounits'])
    rows = list(csv.reader(gpu.splitlines(), skipinitialspace=True))
    if (len(rows) != 1 or rows[0][0].strip() != target or
            '3090' not in rows[0][1] or float(rows[0][2]) >= 500 or
            float(rows[0][3]) != 0 or rows[0][4].strip() != 'Default'):
        raise RuntimeError('Assigned target GPU is not idle/default RTX3090')
    contexts = capture(['nvidia-smi', '-i', target, '--query-compute-apps=pid,process_name,gpu_uuid',
                        '--format=csv,noheader,nounits'])
    if contexts.strip():
        raise RuntimeError('Assigned target GPU has an existing compute context; leave user queues untouched')
    capture(['scontrol', 'show', 'job', '1465', '-o'])

    # Fail before --stage all can rerun qualification or open validation/test.
    from code.unified_selector.registry import POOLS, PROBLEMS
    from code.V4.r58_budget import WHOLE_UNLIMITED, build_whole_budget
    from code.V4.r58_labels import implementation_hashes, runtime_environment
    hashes, expected_environment = implementation_hashes(), runtime_environment()
    profiles = [root / 'preflight' / f'{problem}__{method}.json'
                for problem in PROBLEMS for method in POOLS[problem]]
    if len(profiles) != 128:
        raise ValueError('Expected unchanged 18-task/128-deployment scenario')
    for path in profiles:
        record = json.loads(path.read_text())
        if (record.get('passed') is not True or record.get('implementation_sha256') != hashes or
                record.get('runtime_environment') != expected_environment):
            raise ValueError(f'Qualification/source/environment gate not closed: {path.name}')
    report = build_whole_budget(root, gpu_hour_limit='none')
    whole = report['whole_project']
    if (whole.get('complete') is not True or whole.get('decision') != WHOLE_UNLIMITED or
            whole.get('execution_plan', {}).get('rf_mode') != 'normal4' or
            whole.get('refined_deployments') != 63 or whole.get('preflight_only_deployments') != 65):
        raise ValueError(f'No-cap/current128/measured-normal4 evidence not ready: {whole.get("blockers")}')
    evidence.update(implementation_sha256=hashes, qualified_deployments=128,
                    rf_mode='normal4', gpu_hour_policy=report['gpu_hour_policy'],
                    whole_projected_gpu_hours=whole['whole_projected_gpu_hours'])
    evidence.update(status='complete', passed=True, finished=time.time())
    evidence['wall_seconds'] = evidence['finished'] - started
    save('full_scenario_prelaunch.json', evidence)
    command = ['bash', 'code/V4/run_v4_r58.sh', '--stage', 'all', '--gpu-hour-limit', 'none']
    with (root / 'full_scenario_prelaunch.log').open('w') as stream:
        for item in evidence['commands']:
            stream.write(f"$ {item['argv']!r}\n{item['output']}\nrc={item['returncode']}\n")
    process = subprocess.Popen(command)
    runtime.update(status='running', command=command, pipeline_pid=process.pid,
                   prelaunch_finished=evidence['finished'])
    save(runtime_path.name, runtime)
    print(f"[R58 launch] step={runtime['job_id']}.{runtime['step_id']} worker_pid={os.getpid()} "
          f"pipeline_pid={process.pid}", flush=True)
    code = process.wait()
    state_path = root / 'pipeline_state.json'
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    fresh = (state.get('pid') == process.pid and state.get('started', 0) >= started and
             bool(state.get('finished')))
    complete = (code == 0 and fresh and state.get('stage') == 'all' and state.get('status') == 'complete')
    if fresh:
        name = 'full_scenario_state.json' if complete else 'full_scenario_failure_state.json'
        save(name, state)
    if code == 0 and not complete:
        error, code = 'Pipeline returned zero without a fresh full-completion state', 65
    runtime['pipeline_returncode'] = process.returncode
    runtime['scenario_complete'] = complete
except Exception as exc:
    error = repr(exc)
    print(f'[R58 launch blocked/failed] {error}', flush=True)
finally:
    finished = time.time()
    if evidence['status'] != 'complete':
        evidence.update(status='complete', passed=False, error=error, finished=finished,
                        wall_seconds=finished - started)
        save('full_scenario_prelaunch.json', evidence)
        (root / 'full_scenario_prelaunch.log').write_text(json.dumps(evidence, indent=2) + '\n')
    runtime.update(status='complete', finished=finished, wall_seconds=finished - started,
                   returncode=code, outcome='passed' if code == 0 else 'failed', error=error)
    save(runtime_path.name, runtime)
raise SystemExit(code)
WORKER_PY
fi

[[ "$#" == 0 && -z "${SLURM_STEP_ID:-}" ]]
[[ -n "${TMUX:-}" && "$(tmux display-message -p '#S')" == r58_full_scenario ]]
exec 9> "$ROOT/full_scenario_launch.lock"
flock -n 9
for name in started.json controller.log runtime.json result.json exit_code prelaunch.json prelaunch.log state.json failure_state.json sacct.log; do
    [[ ! -e "$ROOT/full_scenario_$name" ]] || {
        printf 'Existing full-scenario artifact; no automatic retry: %s\n' "$name" >&2
        exit 64
    }
done
export R58_FULL_TMUX_SESSION=r58_full_scenario R58_FULL_LAUNCHER_PID=$$
python - <<'START_PY'
import json, os, time
from pathlib import Path
path = Path(os.environ['R58_ROOT']) / 'full_scenario_started.json'
with path.open('x') as stream:
    json.dump(dict(started=time.time(), launcher_pid=int(os.environ['R58_FULL_LAUNCHER_PID']),
        tmux_session='r58_full_scenario', gpu_hour_limit=None, automatic_retry=False), stream, indent=2)
START_PY
exec > "$ROOT/full_scenario_controller.log" 2>&1
finish() {
    local code=$?
    trap - EXIT
    set +e
    python - "$code" <<'FINISH_PY'
import json, os, subprocess, sys, time
from pathlib import Path
root = Path(os.environ['R58_ROOT'])
started = json.loads((root / 'full_scenario_started.json').read_text())
path = root / 'full_scenario_runtime.json'
runtime = json.loads(path.read_text()) if path.exists() else {}
step = f"1465.{runtime['step_id']}" if runtime.get('step_id') is not None else None
sacct_code = None
if step:
    with (root / 'full_scenario_sacct.log').open('w') as stream:
        sacct_code = subprocess.run(['sacct', '-j', step, '--noheader', '--parsable2',
            '--format=JobIDRaw,JobName,NodeList,State,ExitCode,ElapsedRaw,Start,End'],
            stdout=stream, stderr=subprocess.STDOUT).returncode
finished = time.time()
result = dict(status='complete', started=started['started'], finished=finished,
    wall_seconds=finished-started['started'], srun_returncode=int(sys.argv[1]),
    outcome='passed' if int(sys.argv[1]) == 0 else 'failed',
    launcher_pid=started['launcher_pid'], tmux_session='r58_full_scenario', slurm_step=step,
    runtime_receipt=str(path), sacct_returncode=sacct_code,
    scenario_complete=runtime.get('scenario_complete', False), automatic_retry=False, mps=False)
(root / 'full_scenario_result.json').write_text(json.dumps(result, indent=2) + '\n')
(root / 'full_scenario_exit_code').write_text(sys.argv[1] + '\n')
FINISH_PY
    exit "$code"
}
trap finish EXIT
srun --jobid=1465 --overlap --exact --nodes=1 --ntasks=1 --nodelist=gpu03 \
    --cpus-per-task=4 --mem=16G \
    bash "$ROOT/full_scenario_launch.sh" --worker

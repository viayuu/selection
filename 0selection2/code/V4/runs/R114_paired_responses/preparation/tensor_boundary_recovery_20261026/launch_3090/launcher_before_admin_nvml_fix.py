"""One additive CVRP acquisition, inside the existing idle RTX3090 allocation."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import shlex
import socket
import subprocess
import sys
import time

from . import r113_3090_queue as resource


ROOT = resource.REPO / 'code/V4/runs/R114_paired_responses'
HERE = ROOT / 'preparation/tensor_boundary_recovery_20261026/launch_3090'
CONFIG = HERE / 'ready.json'
SLOT = '3090_1'
TOKEN = 'R114_RECOVERY_TOKEN'
PYTHON = sys.executable


def load():
    value = resource.read(CONFIG)
    if value.get('main_approved') is not True or value.get('slot') != SLOT:
        raise RuntimeError('Main must approve this one existing two-3090 followup')
    for name, expected in value['sources'].items():
        if resource.digest(name) != expected:
            raise RuntimeError('Frozen recovery source changed: ' + name)
    return value


def command(plan, cpus):
    item = resource.require_slot(SLOT)
    return [PYTHON, '-B', '-u', '-m', 'code.V4.r114_run_queue',
        '--plan', str(plan), '--manifest', resource.read(plan)['manifest_path'],
        '--deployment-lock', resource.read(plan)['deployment_lock_path'],
        '--host', resource.CONTAINER[:12], '--gpu-uuid', item['uuid'],
        '--cpu-ids', ','.join(map(str, cpus[:2])), '--timeout-seconds', '3600',
        '--execute', '--finalize']


def prepare(plan):
    plan = Path(plan).resolve()
    if not plan.is_relative_to(ROOT / 'preparation/tensor_boundary_recovery_20261026'):
        raise ValueError('Only the isolated corrected CVRP acquisition is eligible')
    from . import r114_label_queue as labels
    value, manifest, lock = labels.load_plan(plan)
    methods = [row['method'] for row in manifest['pools']['CVRP']]
    if (set(manifest['pools']) != {'CVRP'} or len(methods) != 10
            or {row['problem'] for row in manifest['tasks']} != {'CVRP'}
            or len(manifest['endpoints']) != 96 or len(manifest['tasks']) != 960):
        raise ValueError('Require the unchanged 32-base / 96-endpoint full legal CVRP pool')
    sources = [Path(__file__).resolve(), Path(resource.__file__).resolve(), plan,
               Path(value['manifest_path']), Path(value['deployment_lock_path']),
               Path(value['reference_lock_path'])]
    config = dict(main_approved=True, one_attempt_only=True, slot=SLOT,
        allocation=1465, gpu_uuid=resource.require_slot(SLOT)['uuid'],
        parent_queue=str(resource.HERE / 'controller.json'), wait_for='B_snapshot',
        plan=str(plan), methods=methods, task_count=len(manifest['tasks']),
        sources={str(p): resource.digest(p) for p in sources},
        mutex_inodes=resource.mutex_inodes(), token=resource.uuid.uuid4().hex,
        scientific_change='CVRP audit host views only; original v1 and other problems unchanged')
    resource.save(CONFIG, config, exclusive=True)
    return dict(ready=True, config=str(CONFIG), gpu_started=False)


def authorized():
    value = load()
    if os.environ.get(TOKEN) != value['token']:
        raise RuntimeError('Require the new main-owned recovery entry token')
    return value


def entry():
    value = authorized()
    item = resource.require_slot(SLOT)
    cgroup = Path('/proc/self/cgroup').read_text()
    if (socket.gethostname().split('.')[0] != 'gpu03'
            or os.environ.get('SLURM_JOB_ID') != '1465'
            or not os.environ.get('SLURM_STEP_ID', '').isdigit()
            or not re.search(r'/job_1465(?:/|$)', cgroup)):
        raise RuntimeError('Recovery entry must be inside original allocation1465')
    allocation = resource.original.r81_queue.allocation(item)
    container = resource.original.r81_queue.container_inspect(item)
    allowed = sorted(set(allocation['cpus']) & set(container['allowed_cpus']))
    cpus = allowed[item['index'] * 4:(item['index'] + 1) * 4]
    if len(cpus) != 4:
        raise RuntimeError('Require the same four disjoint original slot CPUs')
    receipt = dict(state='container_exec', cgroup=cgroup,
        allocation=allocation, container=container, cpus=cpus)
    resource.save(HERE / 'entry.json', receipt, exclusive=True)
    env = resource.environment(item['uuid'])
    env[resource.CPU_ENV] = ','.join(map(str, cpus))
    argv = ['sudo', '-n', '/usr/bin/ai_docker', 'exec', '-u', '2022']
    for key in ('CUDA_VISIBLE_DEVICES', 'PYTHONPATH', 'PYTHONDONTWRITEBYTECODE',
                'MPLBACKEND', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                'NUMEXPR_NUM_THREADS', 'PYTORCH_CUDA_ALLOC_CONF', 'WANDB_MODE',
                'SLURM_JOB_ID', 'SLURM_STEP_ID', resource.CPU_ENV, TOKEN):
        argv += ['-e', key + '=' + env[key]]
    shell = '\n'.join(('set -euo pipefail', 'source ' + resource.original.CONDA_SH,
        'conda activate easynco', 'cd ' + shlex.quote(str(resource.REPO)),
        'exec ' + shlex.join([PYTHON, '-B', '-u', '-m', __package__ + '.r114_recovery_launch', 'worker'])))
    argv += [resource.CONTAINER, 'bash', '-lc', shell]
    rc = subprocess.call(argv, cwd=resource.REPO, env=env)
    receipt.update(state='complete' if rc == 0 else 'returned', exit_status=rc,
                   finished_utc=resource.utc())
    resource.save(HERE / 'entry.json', receipt)
    return rc


def worker():
    value = authorized()
    receipt = dict(state='checking', initialized=False, pid=os.getpid(), started_utc=resource.utc())
    resource.save(HERE / 'worker.json', receipt, exclusive=True)
    try:
        with resource.original_mutex(SLOT, value['mutex_inodes']) as fd:
            receipt['resource_checks'] = resource.worker_checks(value, SLOT, HERE)
            cpus = receipt['resource_checks']['cpus']
            argv = command(value['plan'], cpus)
            receipt.update(state='running', initialized=True, argv=argv,
                           actual_gpu_type='3090', gpu_uuid=value['gpu_uuid'])
            resource.save(HERE / 'worker.json', receipt)
            with (HERE / 'acquisition.log').open('a') as log:
                child = subprocess.Popen(argv, cwd=resource.REPO, env=resource.environment(value['gpu_uuid']),
                                         stdout=log, stderr=subprocess.STDOUT, pass_fds=(fd,))
                receipt['child_pid'] = child.pid
                resource.save(HERE / 'worker.json', receipt)
                try:
                    rc = child.wait()
                except BaseException:
                    resource.wait_owned(child)
                    raise
            progress = resource.read(Path(value['plan']).parent / 'scheduler_progress.json')
            receipt.update(state='complete' if rc == 0 else 'blocked' if rc == 2 else 'failed',
                           exit_status=rc, acquisition_status=progress.get('status'),
                           progress_path=str(Path(value['plan']).parent / 'scheduler_progress.json'))
            return rc
    except BaseException as error:
        receipt.update(state='failed', exit_status=1, error=str(error))
        raise
    finally:
        receipt['finished_utc'] = resource.utc()
        resource.save(HERE / 'worker.json', receipt)


def wait(poll):
    value = load()
    if socket.gethostname().split('.')[0] != 'admin01':
        raise RuntimeError('Use the existing admin01 allocation controller, not direct SSH training')
    with (HERE / 'controller.lock').open('a') as guard:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (HERE / 'request.json').exists():
            raise RuntimeError('One recovery dispatch already exists; no automatic retry')
        while True:
            parent = resource.read(value['parent_queue'])
            if parent.get('jobs', {}).get(value['wait_for'], {}).get('state') == 'complete':
                try:
                    with resource.original_mutex(SLOT, value['mutex_inodes']):
                        gpu = resource.original.r81_queue.gpu_inventory(resource.require_slot(SLOT))
                        free = not gpu['compute_processes'] and gpu['memory_used_mib'] <= 512
                except resource.ResourceBusy:
                    free = False
                if free:
                    break
            resource.save(HERE / 'controller.json', dict(state='waiting_resources', pid=os.getpid(),
                updated_utc=resource.utc(), actual_gpu_type='3090', gpu_uuid=value['gpu_uuid']))
            time.sleep(poll)
        resource.save(HERE / 'request.json', dict(ready_sha256=resource.digest(CONFIG),
            started_utc=resource.utc(), one_attempt_only=True), exclusive=True)
        shell = '\n'.join(('set -euo pipefail', 'source ' + resource.original.CONDA_SH,
            'conda activate easynco', 'cd ' + shlex.quote(str(resource.REPO)),
            'exec ' + shlex.join([PYTHON, '-B', '-u', '-m', __package__ + '.r114_recovery_launch', 'entry'])))
        argv = ['srun', '--jobid=1465', '--overlap', '--exact', '--gres=none', '--immediate=5',
                '-N1', '-n1', '--nodelist=gpu03', '--cpus-per-task=4', '--mem=9G', 'bash', '-lc', shell]
        env = resource.environment()
        env[TOKEN] = value['token']
        child = subprocess.Popen(argv, cwd=resource.REPO, env=env)
        resource.save(HERE / 'controller.json', dict(state='running', pid=os.getpid(),
            transport_pid=child.pid, argv=argv, gpu_uuid=value['gpu_uuid'], updated_utc=resource.utc()))
        rc = child.wait()
        resource.save(HERE / 'controller.json', dict(state='complete' if rc == 0 else 'blocked' if rc == 2 else 'failed',
            pid=os.getpid(), transport_pid=child.pid, exit_status=rc, updated_utc=resource.utc(),
            worker_path=str(HERE / 'worker.json'), gpu_uuid=value['gpu_uuid']))
        return rc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'wait', 'entry', 'worker'))
    parser.add_argument('--plan', type=Path)
    parser.add_argument('--main-approved', action='store_true')
    parser.add_argument('--poll-seconds', type=float, default=60)
    args = parser.parse_args()
    if args.command == 'prepare':
        if not args.main_approved or args.plan is None:
            parser.error('prepare requires --plan and --main-approved')
        print(json.dumps(prepare(args.plan), indent=2))
        return 0
    if args.command == 'wait':
        if args.poll_seconds < 5:
            parser.error('Use a quiet poll interval of at least five seconds')
        return wait(args.poll_seconds)
    return entry() if args.command == 'entry' else worker()


if __name__ == '__main__':
    raise SystemExit(main())

"""R117 finite two-3090 dispatch using the existing allocation and UUID mutexes."""

import argparse
import fcntl
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import time

from . import r113_3090_queue as resource


ROOT = resource.REPO / 'code/V4/runs/R117_native_computation'
PREP = ROOT / 'preparation'
WRAPPER = resource.REPO / 'code/V4/run_v4_r117.sh'
TOKEN, MODE, ARM = 'R117_LAUNCH_TOKEN', 'R117_LAUNCH_MODE', 'R117_LAUNCH_ARM'
SLOTS = dict(A='3090_0', B='3090_1')


def folder(mode, arm):
    return PREP / 'resources' / mode / arm


def ready_path(mode):
    return PREP / (mode + '_ready.json')


def prepare(mode):
    common = ['r117_model.py', 'r117_native.py', 'r117_features.py',
        'r117_objective.py', 'r117_launch.py', 'run_v4_r117.sh']
    if mode == 'capacity':
        common.append('r117_capacity.py')
        data = dict(arms=['B'], slot_for_arm=dict(B='3090_0'), max_seconds=1200)
    else:
        from . import r117_experiment as runner
        from .r117_capacity import CAPS
        capacity_path = folder('capacity', 'B') / 'capacity.json'
        witness = resource.read(capacity_path)
        if witness.get('complete') is not True or witness.get('microbatch_caps') != CAPS:
            raise RuntimeError('One successful common capacity witness is required')
        lock = dict(runner.required_lock_fields(), data_root=str(runner.DEFAULT_DATA_ROOT),
            prepared_root=str(runner.DEFAULT_PREPARED_ROOT), primary_cache='/tmp/shiys_R84_new90k_cache',
            microbatch_caps=CAPS, eval_batch_size=witness['eval_batch_size'], lib_batch_size=128,
            native_extraction_batch_size=witness['native_extraction_batch_size'],
            capacity_witness=str(capacity_path), capacity_witness_sha256=resource.digest(capacity_path),
            feature_roots={arm: '/tmp/shiys_R117_' + arm + '_native_features' for arm in SLOTS},
            common_feature_root='/tmp/shiys_R117_common_base', authorized_slots=resource.SLOTS,
            gpu_budget_hours_per_arm=14, complete_gpu_budget_hours=28, no_4090=True)
        resource.save(ROOT / 'protocol_lock.json', lock, exclusive=True)
        common.extend(['r117_experiment.py', 'r117_reporting.py', 'r117_finish.py'])
        data = dict(arms=list(SLOTS), slot_for_arm=SLOTS, max_seconds=14 * 3600,
            protocol_lock=str(ROOT / 'protocol_lock.json'),
            protocol_lock_sha256=resource.digest(ROOT / 'protocol_lock.json'))
    sources = [resource.REPO / 'code/V4' / name for name in common]
    value = dict(main_approved=True, one_attempt_only=True, mode=mode, no_new_allocation=True,
        token=resource.uuid.uuid4().hex, mutex_inodes=resource.mutex_inodes(),
        sources={str(path): resource.digest(path) for path in sources}, **data)
    resource.save(ready_path(mode), value, exclusive=True)
    return value


def load(mode):
    value = resource.read(ready_path(mode))
    if value.get('main_approved') is not True or value.get('mode') != mode:
        raise RuntimeError('Missing current main-owned ready receipt')
    for path, sha in value['sources'].items():
        if resource.digest(path) != sha:
            raise RuntimeError('Frozen executable source changed: ' + path)
    if mode == 'training' and resource.digest(value['protocol_lock']) != value['protocol_lock_sha256']:
        raise RuntimeError('Protocol changed after ready')
    return value


def authorize(mode, arm):
    value = load(mode)
    if arm not in value['arms'] or os.environ.get(TOKEN) != value['token']:
        raise RuntimeError('Worker token/arm does not match this finite dispatch')
    return value


def command(stage, arm):
    return [sys.executable, '-B', '-u', '-m', 'code.V4.r117_experiment', stage,
        '--root', str(ROOT), '--arm', arm, '--gpu-type', '3090', '--device', 'cuda:0',
        '--wandb-mode', 'offline', '--cache-root', '/tmp/shiys_R84_new90k_cache',
        '--protocol-lock', str(ROOT / 'protocol_lock.json'), '--log-every', '18']


def entry(mode, arm):
    value = authorize(mode, arm)
    item = resource.require_slot(value['slot_for_arm'][arm])
    directory = folder(mode, arm)
    cgroup = Path('/proc/self/cgroup').read_text()
    if (socket.gethostname().split('.')[0] != 'gpu03' or os.environ.get('SLURM_JOB_ID') != '1465'
            or not re.search(r'/job_1465(?:/|$)', cgroup)):
        raise RuntimeError('Enter through authorized allocation1465, not direct SSH training')
    allocation = resource.original.r81_queue.allocation(item)
    container = resource.original.r81_queue.container_inspect(item)
    allowed = sorted(set(allocation['cpus']) & set(container['allowed_cpus']))
    cpus = allowed[item['index'] * 4:(item['index'] + 1) * 4]
    if len(cpus) != 4:
        raise RuntimeError('Require four disjoint authorized CPUs per slot')
    resource.save(directory / 'entry.json', dict(state='container_exec', cpus=cpus,
        allocation=allocation, container=container, cgroup=cgroup), exclusive=True)
    env = resource.environment(item['uuid'])
    env.update({resource.CPU_ENV: ','.join(map(str, cpus)), TOKEN: value['token'], MODE: mode, ARM: arm})
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        env[name] = '2'
    argv = ['sudo', '-n', '/usr/bin/ai_docker', 'exec', '-u', '2022']
    for key in ('CUDA_VISIBLE_DEVICES', 'PYTHONPATH', 'PYTHONDONTWRITEBYTECODE', 'MPLBACKEND',
                'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS',
                'PYTORCH_CUDA_ALLOC_CONF', 'WANDB_MODE', 'SLURM_JOB_ID', 'SLURM_STEP_ID',
                resource.CPU_ENV, TOKEN, MODE, ARM):
        argv.extend(['-e', key + '=' + env[key]])
    argv.extend([resource.CONTAINER, 'taskset', '-c', ','.join(map(str, cpus)), 'bash', str(WRAPPER), 'worker'])
    rc = subprocess.call(argv, cwd=resource.REPO, env=env)
    path = directory / 'worker.json'
    if not path.exists():
        raise RuntimeError('Container transport ended without a real worker receipt')
    return resource.read(path).get('exit_status', rc if rc else 1)


def worker(mode, arm):
    value = authorize(mode, arm)
    directory = folder(mode, arm)
    slot = value['slot_for_arm'][arm]
    item = resource.require_slot(slot)
    receipt = dict(state='checking', initialized=False, mode=mode, arm=arm,
        pid=os.getpid(), started_utc=resource.utc(), gpu_uuid=item['uuid'], stages=[])
    resource.save(directory / 'worker.json', receipt, exclusive=True)
    try:
        with resource.original_mutex(slot, value['mutex_inodes']) as fd:
            receipt['checks'] = resource.worker_checks(value, slot, directory)
            if mode == 'capacity':
                stages = [('capacity', [sys.executable, '-B', '-u', '-m', 'code.V4.r117_capacity',
                    '--output', str(directory / 'capacity.json')])]
            else:
                resource.original.check_cache({'plan': {'protocol_lock': value['protocol_lock']}})
                stages = [('train', command('train', arm)), ('finalize', command('finalize', arm))]
            deadline = time.monotonic() + value['max_seconds']
            env = resource.environment(item['uuid'])
            for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
                env[name] = '2'
            receipt.update(state='running', initialized=True)
            resource.save(directory / 'worker.json', receipt)
            for name, argv in stages:
                row = dict(stage=name, argv=argv, state='running', started_utc=resource.utc())
                receipt['stages'].append(row)
                log = ROOT / arm / (name + '.log') if mode == 'training' else directory / (name + '.log')
                log.parent.mkdir(parents=True, exist_ok=True)
                with log.open('a') as stream:
                    child = subprocess.Popen(argv, cwd=resource.REPO, env=env,
                        stdout=stream, stderr=subprocess.STDOUT, pass_fds=(fd,))
                    row.update(child_pid=child.pid, log=str(log))
                    resource.save(directory / 'worker.json', receipt)
                    try:
                        rc = child.wait(timeout=max(1., deadline - time.monotonic()))
                    except subprocess.TimeoutExpired:
                        resource.wait_owned(child)
                        row.update(state='incomplete_budget', exit_status=124, finished_utc=resource.utc())
                        receipt.update(state='incomplete_budget', exit_status=124)
                        return 124
                    except BaseException:
                        resource.wait_owned(child)
                        raise
                row.update(state='complete' if rc == 0 else 'failed', exit_status=rc, finished_utc=resource.utc())
                resource.save(directory / 'worker.json', receipt)
                if rc:
                    receipt.update(state='failed', exit_status=rc)
                    return rc
            receipt.update(state='complete', exit_status=0)
            return 0
    except BaseException as exc:
        receipt.update(state='failed', exit_status=1, error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        receipt['finished_utc'] = resource.utc()
        resource.save(directory / 'worker.json', receipt)


def dispatch(mode, arm):
    value = load(mode)
    if arm not in value['arms']:
        raise ValueError('Arm outside the finite ready receipt')
    directory = folder(mode, arm)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / 'controller.lock').open('a') as mutex:
        fcntl.flock(mutex, fcntl.LOCK_EX | fcntl.LOCK_NB)
        resource.save(directory / 'request.json', dict(started_utc=resource.utc(),
            ready_sha256=resource.digest(ready_path(mode)), mode=mode, arm=arm), exclusive=True)
        env = resource.environment()
        env.update({TOKEN: value['token'], MODE: mode, ARM: arm})
        argv = ['srun', '--jobid=1465', '--overlap', '--exact', '--gres=none', '--immediate=5',
            '-N1', '-n1', '--nodelist=gpu03', '--cpus-per-task=4', '--mem=9G',
            'bash', str(WRAPPER), 'entry']
        child = subprocess.Popen(argv, cwd=resource.REPO, env=env)
        record = dict(state='running', pid=os.getpid(), transport_pid=child.pid, argv=argv,
            mode=mode, arm=arm, updated_utc=resource.utc())
        resource.save(directory / 'controller.json', record)
        try:
            rc = child.wait()
        except BaseException:
            resource.wait_owned(child)
            raise
        record.update(state='complete' if rc == 0 else 'failed', exit_status=rc, updated_utc=resource.utc())
        resource.save(directory / 'controller.json', record)
        return rc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'dispatch', 'entry', 'worker'))
    parser.add_argument('--mode', choices=('capacity', 'training'), default=os.environ.get(MODE))
    parser.add_argument('--arm', choices=tuple(SLOTS), default=os.environ.get(ARM))
    args = parser.parse_args()
    if args.mode is None:
        parser.error('Explicit capacity/training mode required')
    if args.command == 'prepare':
        print(resource.json.dumps(prepare(args.mode)), flush=True)
        return 0
    if args.arm is None:
        parser.error('Explicit arm required')
    return globals()[args.command](args.mode, args.arm)


if __name__ == '__main__':
    raise SystemExit(main())

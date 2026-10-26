"""Deferred R113/R114 wave in the four existing allocation containers.

Waiting is not approval. Only an explicit, hash-bound main ready receipt permits
dispatch; only pre-initialization resource contention may be deferred.
"""

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import signal
import socket
import subprocess
import sys
import time
import uuid

from . import r81_queue
from . import r98_launch


REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / 'code/V4/runs/R113_executable_process'
PLAN = ROOT / 'preparation/wave_plan.json'
WRAPPER = REPO / 'code/V4/run_v4_r113_wave.sh'
CONDA_SH = '/public/home/shiys/miniconda3/etc/profile.d/conda.sh'
COMPAT = '/public/home/shiys/easynco_v3_bridge/exports/train_100k_20261002/python_compat'
SLOTS = {key: {k: v for k, v in value.items() if k != 'cap'}
         for key, value in r98_launch.SLOTS.items()}
CONTAINERS = dict(r81_queue.CONTAINERS)
JOB_IDS = ('C_transition', 'B_snapshot', 'A_scratch', 'R114_labels')
ARM_IDS = ('A_scratch', 'B_snapshot', 'C_transition')
R113_RUNTIME_FILES = ('r113_wave.py', 'r113_experiment.py', 'r113_pretrain.py', 'r113_model.py',
    'r113_processor.py', 'r113_process_data.py', 'r113_torch_inputs.py', 'r113_diagnostics.py',
    'r113_witness.py', 'r113_reporting.py')
R114_READY = REPO / 'code/V4/runs/R114_paired_responses/preparation/grouped_execution/READY_RECEIPT.json'
TOKEN_ENV = 'R113_WAVE_TOKEN'
CPU_ENV = 'R113_WAVE_CPUS'
LOCK_ENV = 'R113_WAVE_MUTEX_FD'
BUSY_EXIT = 75
LOADED_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


class ResourceBusy(RuntimeError):
    """Safe to defer only before any witness or formal child has started."""


def utc():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def save(path, value, exclusive=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, 'w') as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        if exclusive:
            os.link(temporary, path)
        else:
            temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def dispatch(plan):
    return Path(plan['root']) / 'preparation/wave_dispatch'


def load_plan(path):
    plan = read(path)
    expected_jobs = [dict(job_id=job, allowed_slots=(['4090_0', '4090_1']
        if job in ('C_transition', 'B_snapshot') else list(SLOTS))) for job in JOB_IDS]
    if (plan.get('schema_version') != 1 or plan.get('root') != str(ROOT)
            or plan.get('protocol_lock') != str(ROOT / 'protocol_lock.json')
            or plan.get('slots') != SLOTS or plan.get('jobs') != expected_jobs
            or plan.get('cpu_count') != 4 or plan.get('memory_gib') != 9
            or plan.get('idle_memory_mib') != 512
            or plan.get('automatic_training_retry') is not False):
        raise ValueError('Require the original four-slot, four-job wave plan; no extra allocation')
    return plan


def environment(gpu=''):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, PYTHONPATH=COMPAT + ':' + str(REPO),
        PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1', MPLBACKEND='Agg',
        OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4',
        NUMEXPR_NUM_THREADS='4', PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',
        WANDB_PROJECT='selector', WANDB_ENTITY='yjkds-southern-university-of-science-technology',
        WANDB_MODE='offline')
    for key in list(env):
        if key in ('RANK', 'WORLD_SIZE', 'LOCAL_RANK', 'LOCAL_WORLD_SIZE', 'GROUP_RANK',
                   'ROLE_RANK', 'ROLE_WORLD_SIZE', 'MASTER_ADDR', 'MASTER_PORT',
                   'SLURM_PROCID', 'SLURM_LOCALID', 'SLURM_NTASKS') or key.startswith(
                       ('TORCHELASTIC_', 'PMI_', 'PMIX_')):
            env.pop(key, None)
    return env


def mutex_path(slot):
    # Never redirect ROOT, create a substitute lock, or rewrite the JSON inode.
    return r81_queue.ROOT / ('queue_gpu_' + SLOTS[slot]['uuid'] + '.json')


def mutex_inodes():
    result = {}
    for slot in SLOTS:
        path = mutex_path(slot)
        if path.is_symlink() or not path.is_file():
            raise RuntimeError('Missing original r81 mutex: ' + str(path))
        result[str(path)] = path.stat().st_ino
    return result


@contextmanager
def original_mutex(slot, expected):
    path = mutex_path(slot)
    fd = os.open(path, os.O_RDWR | os.O_NOFOLLOW)
    try:
        opened, current = os.fstat(fd), path.lstat()
        if (opened.st_ino != expected.get(str(path))
                or (opened.st_dev, opened.st_ino) != (current.st_dev, current.st_ino)):
            raise RuntimeError('Original r81 mutex inode changed')
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ResourceBusy('Original r81 GPU mutex is occupied') from error
        current = path.lstat()
        if (opened.st_dev, opened.st_ino) != (current.st_dev, current.st_ino):
            raise RuntimeError('Original r81 mutex replaced while acquiring it')
        yield fd
    finally:
        os.close(fd)


def witness_output(plan, arm):
    return Path(plan['root']) / ('preparation/capacity_' + arm + '.json')


def commands(plan, job, slot, r114):
    if job == 'R114_labels':
        return [dict(name='paired_solver_labels', argv=[sys.executable, '-B', '-u', '-m',
            'code.V4.r114_run_queue', '--plan', r114['plan'], '--manifest', r114['manifest_path'],
            '--deployment-lock', r114['deployment_lock_path'], '--host',
            CONTAINERS[SLOTS[slot]['node']][1][:12], '--gpu-uuid', SLOTS[slot]['uuid'],
            '--cpu-ids', '{cpu_ids}', '--timeout-seconds', str(r114['timeout_seconds']),
            '--execute', '--finalize'])]
    lock = read(plan['protocol_lock'])
    common = ['--root', plan['root'], '--arm', job, '--protocol-lock', plan['protocol_lock'],
        '--prepared-root', lock['prepared_root'], '--data-root', lock['data_root'],
        '--cache-root', lock['primary_cache'], '--pretrain-root', lock['pretrain_root']]
    python = [sys.executable, '-B', '-u', '-m']
    result = [dict(name='capacity', argv=python + ['code.V4.r113_witness', *common,
        '--output', str(witness_output(plan, job))])]
    for name in (('train', 'finalize') if job == 'A_scratch' else ('pretrain', 'train', 'finalize')):
        result.append(dict(name=name, argv=python + ['code.V4.r113_experiment', name, *common,
            '--gpu-type', SLOTS[slot]['gpu_type'], '--device', 'cuda:0', '--wandb-mode', 'offline']))
    return result


def check_r114(path, timeout):
    from . import r114_label_queue as queue
    from . import r114_run_queue as runner
    if not path or type(timeout) is not int or timeout < 1:
        raise ValueError('Wait for R114 owner: exact final --r114-plan and positive --r114-timeout-seconds')
    path = Path(path).expanduser().resolve()
    plan, manifest, lock = queue.load_plan(path)
    runner.check_paths(plan, plan['manifest_path'], plan['deployment_lock_path'])
    owner = read(R114_READY)
    if (owner.get('status') != 'implementation_ready_not_runtime_qualified'
            or owner.get('plan_path') != str(path) or owner.get('plan_sha256') != plan['plan_sha256']
            or timeout != owner.get('execution', {}).get('timeout_seconds_per_active_stage')):
        raise ValueError('Require the exact R114 owner-ready plan and approved timeout, not an old queue')
    if (plan.get('split') != 'train' or plan.get('execution') is not False
            or plan.get('gpu_binding') is not None or not plan.get('runtime_jobs')
            or not plan.get('label_jobs') or path != Path(plan['queue_root']) / 'plan.json'
            or plan.get('policy', {}).get('automatic_retry') is not False):
        raise ValueError('Require the finalized independent TRAIN R114 waiting queue, with runtime gates')
    fresh_r114(plan)
    return dict(plan=str(path), plan_sha256=plan['plan_sha256'], timeout_seconds=timeout,
        manifest_path=plan['manifest_path'], deployment_lock_path=plan['deployment_lock_path'],
        schema_version=plan['schema_version'], runtime_jobs=len(plan['runtime_jobs']),
        label_jobs=len(plan['label_jobs']), cli_module='code.V4.r114_run_queue')


def fresh_r114(plan):
    root = Path(plan['queue_root'])
    if any((root / name).exists() for name in ('execution_binding.json', 'attempts', 'scheduler_runs')):
        raise RuntimeError('R114 queue already bound/attempted; preserve it, no implicit resume')


def fresh(plan, job=None):
    lock = read(plan['protocol_lock'])
    for arm in ARM_IDS if job is None else (() if job == 'R114_labels' else (job,)):
        paths = [Path(plan['root']) / arm]
        if arm != 'A_scratch':
            paths.append(Path(lock['pretrain_root']) / arm)
        if any(path.exists() and any(path.iterdir()) for path in paths):
            raise RuntimeError('Existing R113 outputs: ' + arm + '; no restart/resume/overwrite')
        if witness_output(plan, arm).exists():
            raise RuntimeError('Capacity check already attempted: ' + arm + '; never repeat automatically')


def source_hashes(plan, r114):
    paths = [Path(__file__).resolve(), WRAPPER, Path(plan['plan_path']), Path(plan['protocol_lock']),
        Path(r81_queue.__file__), Path(r98_launch.__file__), REPO / 'code/V4/r84_launch.py',
        Path(plan['root']) / 'process_diagnostic_inputs.json',
        Path(plan['root']) / 'preparation/main_cpu_evidence.json', R114_READY]
    paths += [REPO / 'code/V4' / name for name in R113_RUNTIME_FILES]
    paths += [REPO / 'code/V4' / name for name in
              ('r114_label_queue.py', 'r114_response_data.py', 'r114_group_worker.py',
               'r114_run_queue.py', 'r114_response_analysis.py')]
    lock = read(plan['protocol_lock'])
    paths += [Path(lock['process_manifest']), Path(lock['prepared_root']) / 'source_cache_manifest.json',
        Path(r114['plan']), Path(r114['manifest_path']), Path(r114['deployment_lock_path'])]
    queue_plan = read(r114['plan'])
    paths.append(Path(queue_plan['reference_lock_path']))
    for name, expected in read(R114_READY)['source_sha256'].items():
        source = REPO / 'code/V4' / name
        if digest(source) != expected:
            raise RuntimeError('R114 owner source freeze changed: ' + name)
        paths.append(source)
    sources = r98_launch.source_closure(paths)
    result = {name: digest(name) for name in sources}
    if result[str(Path(__file__).resolve())] != LOADED_SHA256:
        raise RuntimeError('Loaded launcher is stale; never implicitly refreeze/restart')
    return result


def check_cli(path, r114_path, timeout):
    plan = dict(load_plan(path), plan_path=str(Path(path).resolve()))
    r114 = check_r114(r114_path, timeout)
    fresh(plan)
    frozen = source_hashes(plan, r114)
    lock = read(plan['protocol_lock'])
    diagnostic_inputs = read(Path(plan['root']) / 'process_diagnostic_inputs.json')
    if (diagnostic_inputs.get('complete') is not True
            or diagnostic_inputs.get('main_protocol_lock_sha256') != digest(plan['protocol_lock'])
            or diagnostic_inputs.get('subset_sha256') != plan['diagnostic_subset_sha256']):
        raise ValueError('Require final shared R113 process diagnostic inputs bound to this protocol')
    main_evidence = read(Path(plan['root']) / 'preparation/main_cpu_evidence.json')
    if (main_evidence.get('completed_tests', {}).get('status') != 'PASS'
            or main_evidence.get('protocol_lock_sha256') != digest(plan['protocol_lock'])):
        raise ValueError('Require current main CPU-test PASS evidence for the exact locked protocol')
    manifest = read(lock['process_manifest'])
    if (digest(lock['process_manifest']) != lock.get('process_manifest_sha256')
            or manifest.get('status') != 'complete' or manifest.get('split') != 'train'):
        raise ValueError('Require the locked complete TRAIN process manifest')
    rows = [row for job in plan['jobs'] if job['job_id'] != 'R114_labels'
            for row in commands(plan, job['job_id'], job['allowed_slots'][0], r114)]
    # Intercept immediately after real argparse: no main body, backend or CUDA call.
    program = '''import argparse,importlib,json,sys
from unittest.mock import patch
from code.V4 import r113_experiment as runner
class Parsed(BaseException): pass
original=argparse.ArgumentParser.parse_args
seen=[]
def parse(self,*a,**kw):
    value=original(self,*a,**kw); seen.append(vars(value)); raise Parsed()
for row in json.loads(sys.argv[1]):
    argv=row['argv']; i=argv.index('-m'); module=importlib.import_module(argv[i+1])
    try:
        with patch.object(argparse.ArgumentParser,'parse_args',parse): module.main(argv[i+2:])
    except Parsed: pass
    else: raise RuntimeError('CLI did not stop after argparse')
    if argv[i+1]=='code.V4.r113_experiment':
        args=runner.parser().parse_args(argv[i+2:]); runner.protocol_lock(args)
    assert not seen[-1].get('resume',False)
print(json.dumps({'parsed':len(seen),'cpu_only':True}))
'''
    process = subprocess.run([sys.executable, '-B', '-c', program, json.dumps(rows)],
        cwd=REPO, env=environment(), capture_output=True, text=True, timeout=90)
    if process.returncode:
        raise RuntimeError('Exact R113 CLI/protocol check failed: ' + (process.stderr or process.stdout)[-3000:])
    samples = commands(plan, 'R114_labels', '3090_0', r114)
    samples[0]['argv'] = ['0,1' if arg == '{cpu_ids}' else arg for arg in samples[0]['argv']]
    check = subprocess.run([sys.executable, '-B', '-c', program, json.dumps(samples)],
        cwd=REPO, env=environment(), capture_output=True, text=True, timeout=90)
    if check.returncode:
        raise RuntimeError('Wait for finalized R114 run_queue --execute --finalize CLI: ' + check.stderr[-3000:])
    if frozen != source_hashes(plan, r114):
        raise RuntimeError('Sources changed during CPU-only checking; wait for source freeze')
    return dict(cpu_only=True, plan=plan, r114=r114, source_sha256=frozen,
        source_set_sha256=canonical(frozen), protocol_sha256=digest(plan['protocol_lock']),
        mutex_inodes=mutex_inodes(), exact_cli=dict(r113=json.loads(process.stdout), r114=json.loads(check.stdout)),
        commands={job['job_id']: {slot: commands(plan, job['job_id'], slot, r114)
            for slot in job['allowed_slots']} for job in plan['jobs']})


def ready(path, args):
    require_controller()
    if not args.runner_tests_passed or not args.evidence or not args.evidence.strip():
        raise ValueError('Explicit main CPU-test/source-freeze evidence is required')
    value = check_cli(path, args.r114_plan, args.r114_timeout_seconds)
    if (args.confirm_main_protocol != value['protocol_sha256']
            or args.confirm_source_set != value['source_set_sha256']
            or args.confirm_r114_plan != value['r114']['plan_sha256']):
        raise ValueError('Main must confirm the exact protocol, final source set and R114 plan hashes')
    value.update(main_approved=True, evidence=args.evidence, runner_tests_passed=True,
        confirmed_utc=utc(), token=uuid.uuid4().hex, automatic_training_retry=False)
    save(dispatch(value['plan']) / 'ready.json', value, exclusive=True)
    return dict(main_approved=True, ready=str(dispatch(value['plan']) / 'ready.json'),
        source_set_sha256=value['source_set_sha256'], gpu_started=False)


def validated_ready(path):
    plan = dict(load_plan(path), plan_path=str(Path(path).resolve()))
    value = read(dispatch(plan) / 'ready.json')
    if (value.get('main_approved') is not True or value.get('runner_tests_passed') is not True
            or not value.get('evidence') or value.get('plan') != plan
            or canonical(value['source_sha256']) != value['source_set_sha256']
            or value['source_sha256'].get(str(Path(__file__).resolve())) != LOADED_SHA256
            or any(not Path(name).is_file() or digest(name) != sha for name, sha in value['source_sha256'].items())
            or value.get('mutex_inodes') != mutex_inodes()):
        raise RuntimeError('Main approval/source/protocol/plan/original inode changed; no dispatch or retry')
    return value


def self_command(command, path, job=None, slot=None, attempt=None):
    argv = [sys.executable, '-B', '-u', '-m', 'code.V4.r113_wave', command, '--plan', str(path)]
    for key, value in (('--job', job), ('--slot', slot), ('--attempt', attempt)):
        if value is not None:
            argv += [key, str(value)]
    return argv


def srun_command(path, job, slot, attempt):
    item = SLOTS[slot]
    shell = '\n'.join(('set -euo pipefail', f'source {CONDA_SH}', 'conda activate easynco',
        'cd ' + shlex.quote(str(REPO)), 'export PYTHONDONTWRITEBYTECODE=1',
        'exec ' + shlex.join(self_command('entry', path, job, slot, attempt))))
    return ['srun', f'--jobid={item["job"]}', '--overlap', '--exact', '--gres=none', '--immediate=5',
        '-N1', '-n1', f'--nodelist={item["node"]}', '--cpus-per-task=4', '--mem=9G', 'bash', '-lc', shell]


def require_controller():
    if socket.gethostname().split('.')[0] != 'admin01' or Path(sys.prefix).name != 'easynco':
        raise RuntimeError('Controller requires admin01 + conda activate easynco; no SSH bypass')


def attempt_path(value, job, number):
    if job not in JOB_IDS or type(number) is not int or number < 1:
        raise ValueError('Explicit authorized job/positive attempt required')
    return dispatch(value['plan']) / job / f'attempt_{number:04d}'


def authorized(path, job, slot, number):
    value = validated_ready(path)
    if (not os.environ.get(TOKEN_ENV) or os.environ[TOKEN_ENV] != value['token']
            or slot not in value['commands'].get(job, {})):
        raise RuntimeError('Require the main-approved one-shot wave token and authorized slot')
    request = read(attempt_path(value, job, number) / 'request.json')
    if (request.get('job') != job or request.get('slot') != slot
            or request.get('ready_sha256') != digest(dispatch(value['plan']) / 'ready.json')):
        raise RuntimeError('Missing/mismatched controller attempt; never launch a free-standing worker')
    return value


def wait_owned(child):
    # No kill/terminate: keep the worker's inode held until its healthy child exits.
    handlers = {n: signal.signal(n, signal.SIG_IGN) for n in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT)}
    try:
        return child.wait()
    finally:
        for number, handler in handlers.items():
            signal.signal(number, handler)


def container_command(path, job, slot, number, cpus):
    item = SLOTS[slot]
    env = environment(item['uuid'])
    env[CPU_ENV] = ','.join(map(str, cpus))
    argv = ['sudo', '-n', '/usr/bin/ai_docker', 'exec', '-u', '2022']
    keys = ('CUDA_VISIBLE_DEVICES', 'PYTHONPATH', 'PYTHONDONTWRITEBYTECODE', 'PYTHONUNBUFFERED',
        'MPLBACKEND', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
        'NUMEXPR_NUM_THREADS', 'PYTORCH_CUDA_ALLOC_CONF', 'WANDB_PROJECT', 'WANDB_ENTITY',
        'WANDB_MODE', 'SLURM_JOB_ID', 'SLURM_STEP_ID', TOKEN_ENV, CPU_ENV)
    for key in keys:
        argv += ['-e', key + '=' + env[key]]
    argv += [CONTAINERS[item['node']][1], 'bash', str(WRAPPER),
             *self_command('worker', path, job, slot, number)[5:]]
    return argv, env


def entry(path, job, slot, number):
    value = authorized(path, job, slot, number)
    directory = attempt_path(value, job, number)
    item = SLOTS[slot]
    receipt = dict(state='checking', initialized=False, started_utc=utc(), job=job, slot=slot)
    save(directory / 'entry.json', receipt, exclusive=True)
    try:
        cgroup = Path('/proc/self/cgroup').read_text()
        if (socket.gethostname().split('.')[0] != item['node']
                or os.environ.get('SLURM_JOB_ID') != str(item['job'])
                or not os.environ.get('SLURM_STEP_ID', '').isdigit()
                or not re.search(r'/job_' + str(item['job']) + r'(?:/|$)', cgroup)):
            raise RuntimeError('Entry must be an srun step inside the original allocation cgroup')
        allocation = r81_queue.allocation(item)
        container = r81_queue.container_inspect(item)
        allowed = sorted(set(allocation['cpus']) & set(container['allowed_cpus']))
        cpus = allowed[item['index'] * 4:(item['index'] + 1) * 4]
        if len(cpus) != 4:
            raise RuntimeError('Cannot prove four disjoint allocation/container CPUs per physical slot')
        receipt.update(cgroup=cgroup, allocation=allocation, container=container, cpus=cpus, state='container_exec')
        save(directory / 'entry.json', receipt)
        argv, env = container_command(path, job, slot, number, cpus)
        child = subprocess.Popen(argv, cwd=REPO, env=env)
        try:
            rc = child.wait()
        except BaseException:
            wait_owned(child)
            raise
        receipt.update(state='complete' if rc == 0 else 'returned', exit_status=rc, finished_utc=utc())
        return rc
    except BaseException as error:
        receipt.update(state='failed', error=str(error), finished_utc=utc())
        raise
    finally:
        save(directory / 'entry.json', receipt)


def worker_checks(value, slot, directory):
    item = SLOTS[slot]
    identity = CONTAINERS[item['node']][1]
    cgroup = Path('/proc/self/cgroup').read_text()
    entry_record = read(directory / 'entry.json')
    cpus = entry_record['cpus']
    if (identity not in cgroup or socket.gethostname() != identity[:12] or os.getuid() != 2022
            or Path(sys.prefix).name != 'easynco' or os.environ.get('SLURM_JOB_ID') != str(item['job'])
            or not os.environ.get('SLURM_STEP_ID', '').isdigit()
            or os.environ.get('CUDA_VISIBLE_DEVICES') != item['uuid']
            or os.environ.get(CPU_ENV) != ','.join(map(str, cpus))
            or len(set(cpus)) != 4 or not set(cpus) <= os.sched_getaffinity(0)
            or entry_record.get('container', {}).get('id') != identity):
        raise RuntimeError('Require original container/cgroup, uid2022, allocation, four CPUs and exact CUDA UUID')
    os.sched_setaffinity(0, cpus)
    inventory = r81_queue.gpu_inventory(item)
    if inventory['compute_processes'] or inventory['memory_used_mib'] > value['plan']['idle_memory_mib']:
        raise ResourceBusy('GPU compute processes present or memory exceeds 512MiB; never kill')
    for process in Path('/proc').glob('[0-9]*'):
        try:
            if int(process.name) == os.getpid() or identity not in (process / 'cgroup').read_text():
                continue
            args = (process / 'cmdline').read_bytes().split(b'\0')
            if args and Path(os.fsdecode(args[0])).name.startswith('python'):
                if set(os.sched_getaffinity(int(process.name))) & set(cpus):
                    raise ResourceBusy('CPU affinity overlaps a live allocation-container Python process')
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    return dict(container_id=identity, cgroup=cgroup, uid=os.getuid(), cpus=cpus,
        gpu=inventory, cache_root='/tmp/shiys_R84_new90k_cache', cuda_initialized=False)


def check_cache(value):
    record = read(value['plan']['protocol_lock'])
    cache, prepared = Path(record['primary_cache']), Path(record['prepared_root'])
    if str(cache) != '/tmp/shiys_R84_new90k_cache':
        raise RuntimeError('Only the existing R84 warm cache is allowed; no copy/rebuild')
    manifest = read(prepared / 'source_cache_manifest.json')
    if manifest.get('cache_root') != str(cache) or manifest.get('data_root') != record['data_root']:
        raise RuntimeError('Frozen R84 cache manifest paths differ')
    administrative = {'created_utc', 'completed_utc', 'seconds', 'test_lock_root'}
    semantic = lambda data: {k: v for k, v in data.items() if k not in administrative}
    for split, rows in (('train', 81000), ('val', 4500), ('test', 4500)):
        expected = manifest['splits'][split]
        if len(expected) != 18:
            raise RuntimeError('Require full18 existing R84 cache metadata')
        for problem, metadata in expected.items():
            actual = read(cache / split / problem / 'metadata.json')
            if actual.get('complete') is not True or actual.get('rows') != rows or semantic(actual) != semantic(metadata):
                raise RuntimeError('Existing R84 cache metadata changed: ' + split + '/' + problem)


def run_stage(row, receipt, path, env, lock_fd):
    current = dict(row, state='starting', started_utc=utc(), log=str(path.parent / (row['name'] + '.log')))
    receipt['stages'].append(current)
    # Even witness initialization ends the only safe automatic-deferral window.
    receipt.update(state='running', initialized=True, phase=row['name'])
    save(path, receipt)
    with Path(current['log']).open('x') as stream:
        stream.write(shlex.join(row['argv']) + '\n')
        stream.flush()
        child = subprocess.Popen(row['argv'], cwd=REPO, env=env, pass_fds=(lock_fd,),
            stdout=stream, stderr=subprocess.STDOUT)
        current['pid'] = child.pid
        save(path, receipt)
        try:
            rc = child.wait()
        except BaseException:
            current['child_exit_status'] = wait_owned(child)
            save(path, receipt)
            raise
    current.update(state='complete' if rc == 0 else 'failed', exit_status=rc, finished_utc=utc())
    save(path, receipt)
    if rc and not (row['name'] == 'paired_solver_labels' and rc == 2):
        raise RuntimeError(f'{row["name"]} exit={rc}; preserve outputs, no automatic retry/resume')


def validate_capacity(value, arm):
    receipt = read(witness_output(value['plan'], arm))
    lock = read(value['plan']['protocol_lock'])
    selectors = [row for row in receipt.get('trials', []) if row.get('stage') == 'selector']
    pretraining = [row for row in receipt.get('trials', []) if row.get('stage') == 'process_pretraining']
    if (receipt.get('complete') is not True or receipt.get('assigned_arm') != arm
            or receipt.get('main_protocol_sha256') != value['protocol_sha256']
            or receipt.get('formal_updates') != 0 or receipt.get('saved_weights') is not False
            or [row.get('problem') for row in selectors] != ['TSP', 'CVRP', 'ATSP', 'OVRPBLTW']
            or len(pretraining) != (0 if arm == 'A_scratch' else 1)
            or any(row.get('cap') != lock['microbatch_caps'][row['problem']]
                   or row.get('rows') != lock['logical_global_batch'] for row in selectors)
            or any(row.get('cap') != lock['pretrain_microbatch_caps'][row['problem']]
                   or row.get('rows') != lock['pretraining']['logical_global_batch'] for row in pretraining)
            or any(row.get('finite') is not True for row in receipt.get('trials', []))):
        raise RuntimeError('Capacity receipt is missing/failed/stale; never shrink caps or repeat it')


def r114_completion(value, exit_status):
    config = value['r114']
    plan = read(config['plan'])
    result = read(Path(plan['queue_root']) / 'scheduler_progress.json')
    expected_status = 'complete' if exit_status == 0 else 'blocked'
    if (exit_status not in (0, 2) or result.get('status') != expected_status
            or result.get('complete') is not (exit_status == 0)
            or result.get('plan_sha256') != config['plan_sha256']
            or result.get('manifest_sha256') != plan['manifest_sha256']
            or result.get('lock_sha256') != plan['lock_sha256']
            or result.get('execution') is not True or result.get('automatic_retry') is not False
            or not result.get('labels', {}).get('path')
            or result.get('analysis', {}).get('status') != 'complete'):
        raise RuntimeError('R114 final scheduler receipt missing/stale; preserve partial outputs, never retry')
    run = Path(result['run_directory'])
    if (run.parent != Path(plan['queue_root']) / 'scheduler_runs'
            or read(run / 'result.json') != result or not Path(result['labels']['path']).is_file()):
        raise RuntimeError('R114 finalized result/partial labels missing or inconsistent')
    return result


def validate_diagnostics(value, arm):
    run = Path(value['plan']['root']) / arm
    receipt = read(run / 'process_diagnostics_complete.json')
    subset = read(Path(value['plan']['root']) / 'process_diagnostic_inputs.json')
    lock = read(run / 'final_test_lock.json')
    if (receipt.get('complete') is not True or receipt.get('arm') != arm
            or receipt.get('subset_sha256') != subset['subset_sha256']
            or receipt.get('diagnostic_source_sha256') != digest(REPO / 'code/V4/r113_diagnostics.py')
            or any(receipt.get(key) != lock[key] for key in ('checkpoint_sha256', 'state_hash', 'selected_update'))
            or set(receipt.get('files', {})) != {'json', 'csv', 'pdf'}
            or any(not Path(row['path']).is_file() or digest(row['path']) != row['sha256']
                   for row in receipt['files'].values())):
        raise RuntimeError('R113 post-lock diagnostics completion missing/partial/stale')


def worker(path, job, slot, number):
    value = authorized(path, job, slot, number)
    directory = attempt_path(value, job, number)
    receipt_path = directory / 'worker.json'
    receipt = dict(job=job, slot=slot, gpu_uuid=SLOTS[slot]['uuid'], gpu_type=SLOTS[slot]['gpu_type'],
        state='checking_resources', initialized=False, stages=[], started_utc=utc(), automatic_retry=False)
    save(receipt_path, receipt, exclusive=True)
    try:
        with original_mutex(slot, value['mutex_inodes']) as fd:
            receipt['checks'] = worker_checks(value, slot, directory)
            fresh(value['plan'], job)
            if job == 'R114_labels':
                fresh_r114(read(value['r114']['plan']))
            else:
                check_cache(value)
            env = environment(SLOTS[slot]['uuid'])
            env[LOCK_ENV] = str(fd)
            if job == 'R114_labels':
                env['R114_CPU_IDS'] = ','.join(map(str, receipt['checks']['cpus'][:2]))
                for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
                    env[key] = '1'
            for row in value['commands'][job][slot]:
                authorized(path, job, slot, number)
                row = dict(row, argv=list(row['argv']))
                if job == 'R114_labels':
                    row['argv'] = [env['R114_CPU_IDS'] if arg == '{cpu_ids}' else arg for arg in row['argv']]
                    row['argv'] = ['taskset', '-c', env['R114_CPU_IDS'], *row['argv']]
                run_stage(row, receipt, receipt_path, env, fd)
                if row['name'] == 'capacity':
                    validate_capacity(value, job)
                if row['name'] == 'finalize':
                    validate_diagnostics(value, job)
            if job == 'R114_labels':
                rc = receipt['stages'][-1]['exit_status']
                receipt['queue_result'] = r114_completion(value, rc)
                if rc == 2:
                    receipt.update(state='blocked', exit_status=2, finished_utc=utc(),
                        reason='R114 finalized partial masks/analysis; business blockage, not resource contention')
                    return 2
            receipt.update(state='complete', exit_status=0, finished_utc=utc())
            return 0
    except ResourceBusy as error:
        if receipt['initialized'] or receipt['stages']:
            receipt.update(state='failed', error=str(error), exit_status=1, finished_utc=utc())
            return 1
        receipt.update(state='resource_busy', error=str(error), exit_status=BUSY_EXIT, finished_utc=utc())
        return BUSY_EXIT
    except BaseException as error:
        receipt.update(state='failed', error=str(error), exit_status=1, finished_utc=utc())
        raise
    finally:
        save(receipt_path, receipt)


def safe_deferral(directory, rc, log):
    worker_path = directory / 'worker.json'
    if worker_path.exists():
        value = read(worker_path)
        return (rc == BUSY_EXIT and value.get('state') == 'resource_busy'
            and value.get('initialized') is False and value.get('stages') == [])
    # Missing receipts alone are NOT proof of safety. Only a rejected step creation is.
    if (directory / 'entry.json').exists():
        return False
    return bool(rc and re.search(r'srun: error: Unable to create step for job \d+: '
        r'(Requested nodes are busy|Resources temporarily unavailable|Job step creation temporarily disabled)', log))


def finish_attempt(job_state, directory, rc, log):
    if safe_deferral(directory, rc, log):
        job_state.update(state='resource_busy', reason='Pre-init resource contention; deferred without training retry')
        return
    worker_path = directory / 'worker.json'
    done = read(worker_path) if worker_path.exists() else {}
    if rc == 0 and done.get('state') == 'complete' and done.get('exit_status') == 0:
        job_state.update(state='complete', exit_status=0, finished_utc=utc())
    elif rc == 2 and done.get('state') == 'blocked' and done.get('exit_status') == 2:
        job_state.update(state='blocked', exit_status=2, finished_utc=utc(), reason=done['reason'])
    else:
        job_state.update(state='failed', exit_status=rc or 1, finished_utc=utc(),
            reason=done.get('error', 'Missing/failed worker completion; no automatic retry'))


def controller_tick(path, state, active):
    plan = dict(load_plan(path), plan_path=str(Path(path).resolve()))
    for job, (child, stream, directory) in list(active.items()):
        rc = child.poll()
        if rc is None:
            continue
        stream.close()
        finish_attempt(state['jobs'][job], directory, rc, (directory / 'transport.log').read_text())
        del active[job]
    ready_path = dispatch(plan) / 'ready.json'
    if not ready_path.exists():
        state['state'] = 'waiting_main'
        return False
    value = validated_ready(path)
    occupied = {state['jobs'][job]['slot'] for job in active}
    for spec in plan['jobs']:
        job = spec['job_id']
        current = state['jobs'][job]
        if current['state'] in ('running', 'complete', 'failed', 'blocked'):
            continue
        allowed_slots = list(spec['allowed_slots'])
        if current['state'] == 'resource_busy' and current.get('slot') in allowed_slots:
            previous = allowed_slots.index(current['slot']) + 1
            allowed_slots = allowed_slots[previous:] + allowed_slots[:previous]
        for slot in allowed_slots:
            if slot in occupied:
                continue
            try:
                with original_mutex(slot, value['mutex_inodes']):
                    pass
            except ResourceBusy:
                continue
            number = current.get('attempts', 0) + 1
            directory = attempt_path(value, job, number)
            save(directory / 'request.json', dict(job=job, slot=slot, created_utc=utc(),
                ready_sha256=digest(ready_path), initialized=False), exclusive=True)
            stream = (directory / 'transport.log').open('x')
            argv = srun_command(path, job, slot, number)
            env = environment()
            env[TOKEN_ENV] = value['token']
            current.update(state='running', slot=slot, attempts=number,
                gpu_uuid=SLOTS[slot]['uuid'], gpu_type=SLOTS[slot]['gpu_type'],
                allocation=SLOTS[slot]['job'], attempt_dir=str(directory))
            save(dispatch(plan) / 'controller.json', state)
            try:
                child = subprocess.Popen(argv, cwd=REPO, env=env, stdout=stream, stderr=subprocess.STDOUT)
            except BaseException:
                stream.close()
                current.update(state='failed', reason='Transport start failed; no implicit retry')
                raise
            current['transport_pid'] = child.pid
            active[job] = (child, stream, directory)
            occupied.add(slot)
            break
        if current['state'] not in ('running', 'complete', 'failed', 'blocked'):
            current.update(state='resource_busy', reason='Waiting for original r81 mutex / free slot; never kill')
    terminal = all(row['state'] in ('complete', 'failed', 'blocked') for row in state['jobs'].values())
    state['state'] = ('failed' if any(row['state'] == 'failed' for row in state['jobs'].values())
        else 'blocked' if any(row['state'] == 'blocked' for row in state['jobs'].values())
        else 'complete') if terminal else ('running' if active else 'waiting_resources')
    return terminal


def wait(path, poll_seconds):
    require_controller()
    if poll_seconds < 1:
        raise ValueError('Positive polling interval required')
    plan = dict(load_plan(path), plan_path=str(Path(path).resolve()))
    root = dispatch(plan)
    root.mkdir(parents=True, exist_ok=True)
    with (root / 'controller.lock').open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = dict(state='waiting_main', started_utc=utc(), pid=os.getpid(), host=socket.gethostname(),
            automatic_training_retry=False, jobs={job: dict(state='waiting_main', attempts=0) for job in JOB_IDS})
        save(root / 'controller.json', state, exclusive=True)
        active = {}
        try:
            while True:
                terminal = controller_tick(path, state, active)
                state['updated_utc'] = utc()
                save(root / 'controller.json', state)
                print(json.dumps(dict(state=state['state'], jobs={job: row['state']
                    for job, row in state['jobs'].items()}, updated_utc=state['updated_utc'])), flush=True)
                if terminal:
                    return 0 if state['state'] == 'complete' else 1
                time.sleep(poll_seconds)
        except BaseException as error:
            state.update(state='draining_after_error', error=str(error))
            save(root / 'controller.json', state)
            for job, (child, log, directory) in active.items():
                rc = wait_owned(child)
                log.close()
                finish_attempt(state['jobs'][job], directory, rc, (directory / 'transport.log').read_text())
            state.update(state='failed', finished_utc=utc())
            save(root / 'controller.json', state)
            raise


def status(path):
    plan = load_plan(path)
    root = dispatch(plan)
    return dict(plan=str(path), launches_work=False, ready_exists=(root / 'ready.json').exists(),
        controller=read(root / 'controller.json') if (root / 'controller.json').exists() else None,
        protocol_locked=read(plan['protocol_lock']).get('locked') if Path(plan['protocol_lock']).exists() else False,
        automatic_training_retry=False)


def parser():
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument('command', choices=('plan', 'status', 'check-cli', 'ready', 'wait', 'entry', 'worker'))
    value.add_argument('--plan', type=Path, default=PLAN)
    value.add_argument('--job', choices=JOB_IDS)
    value.add_argument('--slot', choices=tuple(SLOTS))
    value.add_argument('--attempt', type=int)
    value.add_argument('--poll-seconds', type=float, default=60)
    value.add_argument('--r114-plan', type=Path)
    value.add_argument('--r114-timeout-seconds', type=int)
    value.add_argument('--confirm-main-protocol')
    value.add_argument('--confirm-source-set')
    value.add_argument('--confirm-r114-plan')
    value.add_argument('--runner-tests-passed', action='store_true')
    value.add_argument('--evidence')
    return value


def main(argv=None):
    args = parser().parse_args(argv)
    path = args.plan.expanduser().resolve()
    if args.command in ('entry', 'worker') and (
            not args.job or not args.slot or args.attempt is None):
        raise ValueError('Internal transport requires job, slot, attempt and main token')
    if args.command == 'plan':
        print(json.dumps(dict(load_plan(path), launches_work=False), indent=2))
        return 0
    if args.command == 'status':
        print(json.dumps(status(path), indent=2))
        return 0
    if args.command == 'check-cli':
        result = check_cli(path, args.r114_plan, args.r114_timeout_seconds)
        print(json.dumps({k: v for k, v in result.items() if k not in ('commands', 'source_sha256', 'plan')}, indent=2))
        return 0
    if args.command == 'ready':
        print(json.dumps(ready(path, args), indent=2))
        return 0

    def interrupted(signum, frame):
        raise RuntimeError(f'Interrupted by signal {signum}; drain owned children, no restart or kill')

    for number in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT):
        signal.signal(number, interrupted)
    if args.command == 'wait':
        return wait(path, args.poll_seconds)
    if args.command == 'entry':
        return entry(path, args.job, args.slot, args.attempt)
    return worker(path, args.job, args.slot, args.attempt)


if __name__ == '__main__':
    raise SystemExit(main())

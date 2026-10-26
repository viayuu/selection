"""Resource-only R113 migration and R114 queue, exclusively gpu03/job1465.

The old wave, runners, protocol and interrupted receipts are read-only. Waiting
never approves work. A separate main-ready receipt seals this two-slot queue.
"""

import argparse
from contextlib import contextmanager
import fcntl
import json
import math
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

from . import r113_wave as original


REPO, ROOT = original.REPO, original.ROOT
HISTORICAL_MIGRATION = ROOT / 'preparation/migration_3090_20261026'
HERE = ROOT / 'preparation/migration_3090_portable_inode_20261026'
PLAN = HERE / 'plan.json'
WRAPPER = REPO / 'code/V4/run_v4_r113_3090.sh'
ADAPTER = HERE / 'finish_adapter.py'
OPS = HISTORICAL_MIGRATION / 'stop_4090_receipt.json'
R112_READY = REPO / 'code/V4/runs/R112_region_interfaces/preparation/recovery_20261026/readiness.json'
SLOTS = {key: dict(original.SLOTS[key]) for key in ('3090_0', '3090_1')}
CONTAINER = original.CONTAINERS['gpu03'][1]
JOBS = ('A_scratch', 'C_transition', 'B_snapshot', 'R114_labels')
ARMS = ('A_scratch', 'B_snapshot', 'C_transition')
TOKEN_ENV, CPU_ENV = 'R113_3090_TOKEN', 'R113_3090_CPUS'
BUSY_EXIT = original.BUSY_EXIT
ResourceBusy = original.ResourceBusy
read, save, digest, canonical, utc = original.read, original.save, original.digest, original.canonical, original.utc
environment, wait_owned = original.environment, original.wait_owned
LOADED_SHA256 = digest(__file__)
HARDWARE_NOTE = ('Resource-only 4090 to 3090 migration. B/C keep gpu_type=4090 in their '
    'historical protocol for exact resume equality; pretraining really completed on 4090. '
    'Every new worker executes on an NVML-proved RTX3090 UUID. Model, loss, order, caps, '
    'budget and selection rules are unchanged. Unsaved updates are replayed, not recovered.')


def require_slot(slot):
    if slot not in SLOTS:
        raise ValueError('Only the two original gpu03/job1465 RTX3090 UUID slots are authorized')
    return SLOTS[slot]


def default_plan(r112=R112_READY, operations=OPS):
    r114 = read(ROOT / 'preparation/wave_dispatch/ready.json')['r114']
    return dict(schema_version=1, resource_only_migration=True, root=str(ROOT),
        protocol_lock=str(ROOT / 'protocol_lock.json'), directory=str(HERE), slots=SLOTS,
        cpu_count=4, memory_gib=9, idle_memory_mib=512, automatic_training_retry=False,
        jobs=[dict(job_id=j, allowed_slots=list(SLOTS)) for j in JOBS],
        operations_receipt=str(Path(operations).resolve()),
        r112_readiness=str(Path(r112).resolve()), r112_gpu_jobs=[], r112_watcher_owner='main',
        r114=dict(plan=r114['plan'], timeout_seconds=r114['timeout_seconds']),
        capacity_policy='one_shared_fresh_selector_check_same_locked_caps_no_auto_retry',
        hardware_explanation=HARDWARE_NOTE)


def load_plan(path):
    path = Path(path).resolve()
    value = read(path)
    if path != PLAN or value != default_plan(value.get('r112_readiness', ''), value.get('operations_receipt', '')):
        raise ValueError('Require the exact independent two-slot migration plan; no old wave dispatch')
    return dict(value, plan_path=str(path))


def directory(plan):
    return Path(plan['directory'])


def mutex_path(slot):
    require_slot(slot)
    return original.mutex_path(slot)


def mutex_inodes():
    result = {}
    for slot in SLOTS:
        path = mutex_path(slot)
        if path.is_symlink() or not path.is_file():
            raise RuntimeError('Missing original r81 UUID mutex: ' + str(path))
        info = path.stat()
        result[str(path)] = dict(device=info.st_dev, inode=info.st_ino)
    return result


def same_mutex_files(expected, observed):
    # st_dev is mount-local: the same NFS file is device2065 here, device46 on gpu03.
    return (set(expected) == set(observed) and all(
        expected[path].get('inode') == observed[path].get('inode') for path in expected))


@contextmanager
def original_mutex(slot, expected):
    path = mutex_path(slot)
    fd = os.open(path, os.O_RDWR | os.O_NOFOLLOW)
    try:
        opened, current = os.fstat(fd), path.lstat()
        if (expected.get(str(path), {}).get('inode') != opened.st_ino
                or (opened.st_dev, opened.st_ino) != (current.st_dev, current.st_ino)):
            raise RuntimeError('Original r81 UUID mutex inode changed')
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ResourceBusy('Original r81 GPU mutex occupied; never stop R109') from error
        current = path.lstat()
        if (opened.st_dev, opened.st_ino) != (current.st_dev, current.st_ino):
            raise RuntimeError('Original r81 UUID mutex replaced during acquisition')
        yield fd
    finally:
        os.close(fd)


def require_old_controller_retired():
    controller = read(ROOT / 'preparation/wave_dispatch/controller.json')
    pid = controller.get('pid')
    if pid != 715420:
        raise RuntimeError('Unexpected old controller identity; main must reconcile it')
    process = Path('/proc') / str(pid)
    try:
        argv = (process / 'cmdline').read_bytes().split(b'\0')
        state = (process / 'stat').read_text().rsplit(')', 1)[1].split()[0]
    except FileNotFoundError:
        return dict(pid=pid, retired=True, observation='absent; no signals sent')
    if b'code.V4.r113_wave' in argv and state not in ('Z', 'X'):
        raise RuntimeError('Old controller still present/draining; main alone owns retirement')
    return dict(pid=pid, retired=True, observation='not a live old wave process; no signals sent')


def operations(plan, verify_checkpoints=True):
    value = read(plan['operations_receipt'])
    signals = value.get('signals', [])
    if (value.get('complete') is not True or value.get('training_restart') is not False
            or value.get('fresh_solver_calls') != 0 or len(signals) != 2
            or {r.get('arm') for r in signals} != {'B_snapshot', 'C_transition'}
            or value.get('source_dispatch') != str(ROOT / 'preparation/wave_dispatch')):
        raise ValueError('Require main-owned, complete stop receipt, not an inferred interruption')
    result = {}
    for row in signals:
        arm = row['arm']
        last = ROOT / arm / 'last.pt'
        old = Path(row['source_worker'])
        if (row.get('exited') is not True or row.get('historical_gpu_type') != '4090'
                or any(row.get(k) is not True for k in
                       ('optimizer_retained', 'scaler_retained', 'rng_retained', 'stream_retained'))
                or row.get('checkpoint') != str(last)
                or type(row.get('checkpoint_update')) is not int
                or not 0 < row['checkpoint_update'] < 57600
                or type(row.get('unpersisted_updates_at_signal')) is not int
                or row['unpersisted_updates_at_signal'] < 0
                or not old.is_relative_to(ROOT / 'preparation/wave_dispatch' / arm)):
            raise ValueError('Incomplete checkpoint/old-attempt linkage in main receipt: ' + arm)
        historical = read(old)
        stages = historical.get('stages', [])
        if (historical.get('job') != arm or historical.get('gpu_type') != '4090'
                or [s.get('name') for s in stages[:3]] != ['capacity', 'pretrain', 'train']
                or any(s.get('state') != 'complete' or s.get('exit_status') != 0 for s in stages[:2])):
            raise ValueError('Old attempt must prove completed capacity/pretraining: ' + arm)
        pre = ROOT / 'pretrain' / arm
        completed = read(pre / 'completion.json')
        if (completed.get('complete') is not True or completed.get('successful_updates') != 8640
                or completed.get('arm') != arm
                or completed.get('processor_path') != str(pre / 'processor.pt')
                or digest(pre / 'processor.pt') != completed.get('processor_file_sha256')):
            raise ValueError('Preserve the actual complete 4090 processor pretraining: ' + arm)
        if verify_checkpoints and digest(last) != row.get('checkpoint_sha256'):
            raise RuntimeError('Checkpoint advanced/changed after main stop receipt: ' + arm)
        result[arm] = dict(row, old_worker_sha256=digest(old),
            protocol_sha256=digest(ROOT / arm / 'protocol.json'),
            pretraining_completion_sha256=digest(pre / 'completion.json'),
            replay_note='Replay ' + str(row['unpersisted_updates_at_signal']) +
                        ' observed unsaved updates; last persisted update is ' + str(row['checkpoint_update']))
    return result


def owner_cpu_argv(argv, modules, command):
    if not isinstance(argv, list) or not argv or not all(isinstance(x, str) for x in argv):
        raise ValueError('R112 owner must supply a literal argv, not a shell command')
    index = 0
    if argv[0] == '/usr/bin/env':
        index = 1
        allowed = {'CUDA_VISIBLE_DEVICES', 'PYTHONPATH', 'PYTHONDONTWRITEBYTECODE',
                   'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MPLBACKEND'}
        settings = {}
        while index < len(argv) and '=' in argv[index]:
            key, value = argv[index].split('=', 1)
            if key not in allowed or key in settings:
                raise ValueError('Unexpected/duplicate R112 owner environment setting')
            settings[key] = value
            index += 1
        if settings.get('CUDA_VISIBLE_DEVICES') != '':
            raise ValueError('Owner env wrapper must explicitly hide CUDA')
    if index >= len(argv) or Path(argv[index]).resolve() != Path(sys.executable).resolve():
        raise ValueError('R112 verifier/watcher must use activated easynco Python')
    tail = argv[index + 1:]
    if (len(tail) < 3 or '-m' not in tail or tail.index('-m') + 2 >= len(tail)
            or any(x not in ('-B', '-u') for x in tail[:tail.index('-m')])
            or tail[tail.index('-m') + 1] not in modules
            or tail[tail.index('-m') + 2] != command):
        raise ValueError('Require the owner-approved saved-only R112 module/command')
    return list(argv)


def check_r112(path):
    """Consume owner-supplied CPU recovery evidence, never invent recovery argv."""
    path = Path(path)
    if not path.is_file():
        raise RuntimeError('Wait for Turing readiness file: ' + str(path))
    value = read(path)
    if not isinstance(value, dict):
        raise ValueError('R112 readiness must be an owner-authored JSON object')
    if (value.get('gpu_jobs', []) or value.get('gpu_stages_remaining', []) or value.get('gpu_stages', [])
            or value.get('gpu_compute') is not False or value.get('needs_3090_queue') is not False
            or value.get('state') != 'ready_for_single_publication'
            or value.get('finalized_source_api') is not True):
        raise ValueError('R112 owner reports missing GPU work; require an explicit revised handoff')
    verify = owner_cpu_argv(value.get('verify_argv'), {'code.V4.r112_postprocess_recovery'}, 'verify')
    if ('--device' not in verify or verify.index('--device') + 1 >= len(verify)
            or verify[verify.index('--device') + 1] != 'cpu'):
        raise ValueError('R112 owner readiness must supply exact CPU-only verify_argv')
    watcher = value.get('watcher_argv')
    if watcher:
        owner_cpu_argv(watcher, {'code.V4.r112_postprocess_recovery', 'code.V4.r112_finish'}, 'wait')
    return dict(path=str(path.resolve()), sha256=digest(path), gpu_jobs=[], verify_argv=verify,
        watcher_argv=watcher, owner_readiness=value)


def check_r114(plan):
    # This metadata-only validator does not authorize old-wave dispatch.
    result = original.check_r114(Path(plan['r114']['plan']), plan['r114']['timeout_seconds'])
    return result


def commands(plan, job, slot, r114):
    item = require_slot(slot)
    if job == 'R114_labels':
        return original.commands(plan, job, slot, r114)
    if job not in ARMS:
        raise ValueError('No additional GPU jobs are authorized')
    lock = read(plan['protocol_lock'])
    common = ['--root', plan['root'], '--arm', job, '--protocol-lock', plan['protocol_lock'],
        '--prepared-root', lock['prepared_root'], '--data-root', lock['data_root'],
        '--cache-root', lock['primary_cache'], '--pretrain-root', lock['pretrain_root']]
    historical_gpu = item['gpu_type'] if job == 'A_scratch' else '4090'
    rows = []
    for stage in ('train', 'finalize'):
        argv = [sys.executable, '-B', '-u', '-m', 'code.V4.r113_experiment', stage, *common,
                '--gpu-type', historical_gpu, '--device', 'cuda:0', '--wandb-mode', 'offline']
        if stage == 'train' and job != 'A_scratch':
            argv.append('--resume')
        rows.append(dict(name=stage, argv=argv))
    return rows


def checkpoint_metadata(links):
    program = '''import json,sys,torch
from code.V4 import r113_experiment as r
out={}
for arm,row in json.loads(sys.argv[1]).items():
    state=torch.load(row['checkpoint'],map_location='cpu',weights_only=False)
    config=state['args']; r.validate_checkpoint_config(config)
    required=('model','optimizer','scaler','rng','stream_draws','updates_by_problem','history','best','initialization')
    assert all(k in state for k in required), 'incomplete resume state'
    assert state['optimizer']['state'] and state['scaler'] and state['rng']
    assert config == r.read_json(r.DEFAULT_ROOT/arm/'protocol.json')
    assert state['state_hash']==r.json_hash(config)==row['state_hash']
    assert state['successful_updates']==row['checkpoint_update']
    assert config['gpu_type']=='4090' and config['arm']==arm
    assert config['source_sha256']=={n:r.file_hash(r.V4/n) for n in r.SOURCE_FILES}
    assert sum(state['updates_by_problem'].values())==state['successful_updates']
    out[arm]=dict(successful_updates=state['successful_updates'],state_hash=state['state_hash'],
        restore=['model','optimizer','scaler','rng','stream_draws','history','best','pending_evaluation'],
        scheduler='frozen deterministic LR schedule reconstructed from successful_updates',
        cuda_initialized=torch.cuda.is_initialized())
    assert not torch.cuda.is_initialized()
    del state
print(json.dumps(out))
'''
    process = subprocess.run([sys.executable, '-B', '-c', program, json.dumps(links)], cwd=REPO,
        env=environment(), capture_output=True, text=True, timeout=120)
    if process.returncode:
        raise RuntimeError('CPU checkpoint integrity check failed: ' + process.stderr[-3000:])
    return json.loads(process.stdout)


def cli_check(rows):
    program = '''import argparse,importlib,json,sys
from unittest.mock import patch
class Parsed(BaseException): pass
seen=[]; original=argparse.ArgumentParser.parse_args
def parse(self,*a,**kw):
    value=original(self,*a,**kw); seen.append(vars(value)); raise Parsed()
for row in json.loads(sys.argv[1]):
    argv=row['argv']; i=argv.index('-m'); module=importlib.import_module(argv[i+1])
    with patch.object(argparse.ArgumentParser,'parse_args',parse):
        try: module.main(argv[i+2:])
        except Parsed: pass
        else: raise RuntimeError('CLI did not stop at argparse')
    if argv[i+1]=='code.V4.r113_experiment':
        args=module.parser().parse_args(argv[i+2:]); module.protocol_lock(args)
print(json.dumps(dict(parsed=len(seen),cpu_only=True)))
'''
    process = subprocess.run([sys.executable, '-B', '-c', program, json.dumps(rows)], cwd=REPO,
        env=environment(), capture_output=True, text=True, timeout=120)
    if process.returncode:
        raise RuntimeError('Exact CPU argparse/protocol check failed: ' + process.stderr[-3000:])
    return json.loads(process.stdout)


def source_hashes(plan, links, r112, r114):
    old_ready = read(ROOT / 'preparation/wave_dispatch/ready.json')
    old_sources = old_ready.get('source_sha256', {})
    if (old_ready.get('main_approved') is not True
            or canonical(old_sources) != old_ready.get('source_set_sha256')):
        raise RuntimeError('Original source provenance is missing; do not refreeze scientific code')
    for name, expected in old_sources.items():
        if digest(name) != expected:
            raise RuntimeError('Original frozen scientific/runtime source changed: ' + name)
    paths = [Path(__file__), WRAPPER, ADAPTER, REPO / 'code/V4/test_r113_3090_queue.py',
        HERE / 'cpu_tests.json',
        REPO / 'code/V4/resource_3090_migration.py', REPO / 'code/V4/run_resource_3090_migration.sh',
        Path(plan['plan_path']), Path(plan['operations_receipt']), Path(r112['path']),
        ROOT / 'preparation/wave_dispatch/ready.json',
        REPO / 'code/V4/r113_finish.py', REPO / 'code/V4/r112_finish.py',
        REPO / 'code/V4/r114_finish.py', Path(r114['plan']),
        Path(r114['manifest_path']), Path(r114['deployment_lock_path'])]
    paths += [Path(name) for name in old_sources]
    for arm, row in links.items():
        paths += [Path(row['source_worker']), ROOT / 'pretrain' / arm / 'completion.json',
            ROOT / 'pretrain' / arm / 'processor.pt', ROOT / arm / 'protocol.json']
    owner = r112['owner_readiness']
    for name, expected in owner.get('source_sha256', {}).items():
        path = Path(name) if Path(name).is_absolute() else REPO / 'code/V4' / name
        if digest(path) != expected:
            raise RuntimeError('R112 recovery source changed: ' + str(path))
        paths.append(path)
    paths += [REPO / 'code/V4/r112_postprocess_recovery.py']
    sources = original.r98_launch.source_closure(paths)
    hashes = {name: digest(name) for name in sources}
    if hashes[str(Path(__file__).resolve())] != LOADED_SHA256:
        raise RuntimeError('Loaded migration queue is stale; main must restart explicitly')
    return hashes


def check_cli(path):
    plan = load_plan(path)
    retired = require_old_controller_retired()
    tests_path = directory(plan) / 'cpu_tests.json'
    tests = read(tests_path)
    test_sources = [Path(__file__).resolve(), WRAPPER, ADAPTER, REPO / 'code/V4/test_r113_3090_queue.py']
    if (tests.get('status') != 'PASS' or tests.get('cpu_only') is not True
            or tests.get('gpu_hidden') is not True or tests.get('gpu_started') is not False
            or tests.get('tests', 0) < 18 or tests.get('failures') != 0 or tests.get('errors') != 0
            or tests.get('source_sha256') != {str(p): digest(p) for p in test_sources}):
        raise RuntimeError('Require passing bounded CPU tests for these exact new sources')
    links = operations(plan)
    if (ROOT / 'A_scratch').exists() and any((ROOT / 'A_scratch').iterdir()):
        raise RuntimeError('A_scratch must be fresh; no implicit resume/overwrite')
    r112, r114 = check_r112(plan['r112_readiness']), check_r114(plan)
    verify = subprocess.run(r112['verify_argv'], cwd=REPO, env=environment(),
        capture_output=True, text=True, timeout=120)
    if verify.returncode:
        raise RuntimeError('Owner-supplied R112 recovery verification failed: ' + verify.stderr[-3000:])
    rows = [capacity_command(plan), *[row for job in JOBS for row in commands(plan, job, '3090_0', r114)]]
    rows = [dict(row, argv=['0,1' if x == '{cpu_ids}' else x for x in row['argv']]) for row in rows]
    sources = source_hashes(plan, links, r112, r114)
    value = dict(plan=plan, retired_controller=retired, links=links, r112=r112, r114=r114,
        protocol_sha256=digest(plan['protocol_lock']), mutex_inodes=mutex_inodes(),
        source_sha256=sources, source_set_sha256=canonical(sources),
        exact_cli=cli_check(rows), checkpoint_metadata=checkpoint_metadata(links),
        commands={job: {slot: commands(plan, job, slot, r114) for slot in SLOTS} for job in JOBS},
        r112_verifier=dict(exit_status=0, argv=r112['verify_argv']),
        cpu_tests=dict(path=str(tests_path), sha256=digest(tests_path), tests=tests['tests'], status='PASS'),
        cpu_only=True, gpu_started=False, automatic_training_retry=False)
    if sources != source_hashes(plan, links, r112, r114) or links != operations(plan):
        raise RuntimeError('Inputs changed during checking; no ready receipt')
    value['check_sha256'] = canonical(value)
    return value


def ready(path, args):
    original.require_controller()
    if not args.runner_tests_passed or not args.evidence or not args.evidence.strip():
        raise ValueError('Main CPU-test PASS evidence is required')
    value = check_cli(path)
    if args.confirm_check != value['check_sha256']:
        raise ValueError('Main must confirm the current exact check_sha256')
    value.update(main_approved=True, runner_tests_passed=True, evidence=args.evidence,
        token=uuid.uuid4().hex, confirmed_utc=utc())
    save(HERE / 'ready.json', value, exclusive=True)
    return dict(main_approved=True, ready=str(HERE / 'ready.json'), gpu_started=False)


def validated_ready(path):
    plan = load_plan(path)
    value = read(directory(plan) / 'ready.json')
    sources = value.get('source_sha256', {})
    approval_fields = {'check_sha256', 'main_approved', 'runner_tests_passed', 'evidence', 'token', 'confirmed_utc'}
    checked = {k: v for k, v in value.items() if k not in approval_fields}
    if (value.get('main_approved') is not True or value.get('runner_tests_passed') is not True
            or not value.get('evidence') or value.get('plan') != plan
            or canonical(checked) != value.get('check_sha256')
            or canonical(sources) != value.get('source_set_sha256')
            or sources.get(str(Path(__file__).resolve())) != LOADED_SHA256
            or any(digest(name) != sha for name, sha in sources.items())
            or not same_mutex_files(value.get('mutex_inodes', {}), mutex_inodes())):
        raise RuntimeError('Main-ready sources/plan/original UUID inode changed; no dispatch/retry')
    return value


def self_command(stage, path, job=None, slot=None, attempt=None):
    argv = [sys.executable, '-B', '-u', '-m', 'code.V4.r113_3090_queue', stage, '--plan', str(path)]
    for key, value in (('--job', job), ('--slot', slot), ('--attempt', attempt)):
        if value is not None:
            argv += [key, str(value)]
    return argv


def srun_command(path, job, slot, attempt):
    item = require_slot(slot)
    shell = '\n'.join(('set -euo pipefail', 'source ' + original.CONDA_SH, 'conda activate easynco',
        'cd ' + shlex.quote(str(REPO)), 'export PYTHONDONTWRITEBYTECODE=1',
        'exec ' + shlex.join(self_command('entry', path, job, slot, attempt))))
    return ['srun', '--jobid=1465', '--overlap', '--exact', '--gres=none', '--immediate=5',
        '-N1', '-n1', '--nodelist=' + item['node'], '--cpus-per-task=4', '--mem=9G', 'bash', '-lc', shell]


def attempt_path(value, job, number):
    if job not in JOBS or type(number) is not int or number < 1:
        raise ValueError('Explicit queue job and positive attempt required')
    return directory(value['plan']) / 'dispatch' / job / f'attempt_{number:04d}'


def authorized(path, job, slot, number):
    require_slot(slot)
    value = validated_ready(path)
    if not os.environ.get(TOKEN_ENV) or os.environ[TOKEN_ENV] != value['token']:
        raise RuntimeError('Require new main-ready token, not an old wave token')
    request = read(attempt_path(value, job, number) / 'request.json')
    if (request.get('job') != job or request.get('slot') != slot
            or request.get('ready_sha256') != digest(HERE / 'ready.json')):
        raise RuntimeError('Worker requires the controller-owned immutable attempt request')
    return value


def container_command(path, job, slot, number, cpus):
    item = require_slot(slot)
    env = environment(item['uuid'])
    env[CPU_ENV] = ','.join(map(str, cpus))
    argv = ['sudo', '-n', '/usr/bin/ai_docker', 'exec', '-u', '2022']
    keys = ('CUDA_VISIBLE_DEVICES', 'PYTHONPATH', 'PYTHONDONTWRITEBYTECODE', 'PYTHONUNBUFFERED',
        'MPLBACKEND', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
        'NUMEXPR_NUM_THREADS', 'PYTORCH_CUDA_ALLOC_CONF', 'WANDB_PROJECT', 'WANDB_ENTITY',
        'WANDB_MODE', 'SLURM_JOB_ID', 'SLURM_STEP_ID', TOKEN_ENV, CPU_ENV)
    for key in keys:
        argv += ['-e', key + '=' + env[key]]
    argv += [CONTAINER, 'bash', str(WRAPPER), *self_command('worker', path, job, slot, number)[5:]]
    return argv, env


def entry(path, job, slot, number):
    value = authorized(path, job, slot, number)
    folder = attempt_path(value, job, number)
    item = require_slot(slot)
    receipt = dict(state='checking', initialized=False, started_utc=utc(), job=job, slot=slot)
    save(folder / 'entry.json', receipt, exclusive=True)
    try:
        cgroup = Path('/proc/self/cgroup').read_text()
        if (socket.gethostname().split('.')[0] != 'gpu03' or os.environ.get('SLURM_JOB_ID') != '1465'
                or not os.environ.get('SLURM_STEP_ID', '').isdigit()
                or not re.search(r'/job_1465(?:/|$)', cgroup)):
            raise RuntimeError('Entry requires the original gpu03/job1465 srun cgroup')
        allocation = original.r81_queue.allocation(item)
        container = original.r81_queue.container_inspect(item)
        allowed = sorted(set(allocation['cpus']) & set(container['allowed_cpus']))
        cpus = allowed[item['index'] * 4:(item['index'] + 1) * 4]
        if len(cpus) != 4:
            raise RuntimeError('Cannot prove four disjoint allocation/container CPUs per slot')
        receipt.update(cgroup=cgroup, allocation=allocation, container=container, cpus=cpus, state='container_exec')
        save(folder / 'entry.json', receipt)
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
        save(folder / 'entry.json', receipt)


def worker_checks(value, slot, folder):
    item = require_slot(slot)
    cgroup = Path('/proc/self/cgroup').read_text()
    entry_record = read(folder / 'entry.json')
    cpus = entry_record['cpus']
    if (CONTAINER not in cgroup or socket.gethostname() != CONTAINER[:12] or os.getuid() != 2022
            or Path(sys.prefix).name != 'easynco' or os.environ.get('SLURM_JOB_ID') != '1465'
            or not os.environ.get('SLURM_STEP_ID', '').isdigit()
            or os.environ.get('CUDA_VISIBLE_DEVICES') != item['uuid']
            or os.environ.get(CPU_ENV) != ','.join(map(str, cpus))
            or len(set(cpus)) != 4 or not set(cpus) <= os.sched_getaffinity(0)
            or entry_record.get('container', {}).get('id') != CONTAINER):
        raise RuntimeError('Require actual original 3090 container/cgroup/uid/allocation/CPUs/UUID')
    os.sched_setaffinity(0, cpus)
    inventory = original.r81_queue.gpu_inventory(item)
    if inventory['uuid'] != item['uuid'] or 'RTX 3090' not in inventory['name']:
        raise RuntimeError('Worker must prove actual RTX3090 hardware, not historical config.gpu_type')
    if inventory['compute_processes'] or inventory['memory_used_mib'] > 512:
        raise ResourceBusy('Physical GPU still busy; R109 is never signaled or stopped')
    for process in Path('/proc').glob('[0-9]*'):
        try:
            if int(process.name) == os.getpid() or CONTAINER not in (process / 'cgroup').read_text():
                continue
            args = (process / 'cmdline').read_bytes().split(b'\0')
            if args and Path(os.fsdecode(args[0])).name.startswith('python'):
                if set(os.sched_getaffinity(int(process.name))) & set(cpus):
                    raise ResourceBusy('Assigned CPUs overlap a live container Python process')
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    return dict(container_id=CONTAINER, cgroup=cgroup, uid=os.getuid(), cpus=cpus, gpu=inventory,
        actual_gpu_type='3090', allocation=1465, cache_root='/tmp/shiys_R84_new90k_cache', cuda_initialized=False)


def check_start(value, job):
    if job == 'A_scratch':
        path = ROOT / job
        if path.exists() and any(path.iterdir()):
            raise RuntimeError('Scratch is not empty; no automatic restart')
    elif job in ('B_snapshot', 'C_transition'):
        row = value['links'][job]
        if digest(row['checkpoint']) != row['checkpoint_sha256']:
            raise RuntimeError('Explicit approved last.pt changed; no implicit resume: ' + job)
        if digest(row['source_worker']) != row['old_worker_sha256']:
            raise RuntimeError('Original interrupted receipt changed; preserve it')
    elif job == 'R114_labels':
        original.fresh_r114(read(value['r114']['plan']))


def capacity_command(plan):
    lock = read(plan['protocol_lock'])
    return dict(name='capacity', argv=[sys.executable, '-B', '-u', '-m', 'code.V4.r113_witness',
        '--root', plan['root'], '--arm', 'A_scratch', '--protocol-lock', plan['protocol_lock'],
        '--prepared-root', lock['prepared_root'], '--data-root', lock['data_root'],
        '--cache-root', lock['primary_cache'], '--pretrain-root', lock['pretrain_root'],
        '--output', str(directory(plan) / 'capacity_3090.json')])


def validate_capacity(value):
    path = directory(value['plan']) / 'capacity_3090.json'
    row = read(path)
    lock = read(value['plan']['protocol_lock'])
    trials = row.get('trials', [])
    if (row.get('complete') is not True or row.get('assigned_arm') != 'A_scratch'
            or 'RTX 3090' not in row.get('device', '') or row.get('formal_updates') != 0
            or row.get('saved_weights') is not False or row.get('scratch_only') is not True
            or row.get('main_protocol_sha256') != value['protocol_sha256']
            or [t.get('problem') for t in trials] != ['TSP', 'CVRP', 'ATSP', 'OVRPBLTW']
            or any(t.get('stage') != 'selector' or t.get('finite') is not True
                or t.get('cap') != lock['microbatch_caps'][t['problem']]
                or t.get('rows') != lock['logical_global_batch'] for t in trials)):
        raise RuntimeError('Single 3090 same-capacity check failed/stale; no cap changes/retry')
    return dict(path=str(path), sha256=digest(path), shared_identical_selector_architecture=True,
        no_pretraining_repeated=True, same_locked_caps=True)


def ensure_capacity(value, receipt, receipt_path, env, fd):
    root = directory(value['plan'])
    # Only one selector-only witness for both identical 24GB 3090s and all arms.
    with (root / 'capacity.lock').open('a') as claim:
        fcntl.flock(claim, fcntl.LOCK_EX)
        if not (root / 'capacity_3090.json').exists():
            if (root / 'capacity_attempt.json').exists():
                raise RuntimeError('Capacity already attempted; never automatically retry it')
            save(root / 'capacity_attempt.json', dict(job=receipt['job'], slot=receipt['slot'],
                actual_gpu_type='3090', gpu_uuid=receipt['gpu_uuid'], initialized=True,
                created_utc=utc(), automatic_retry=False), exclusive=True)
            original.run_stage(capacity_command(value['plan']), receipt, receipt_path, env, fd)
        receipt['capacity_evidence'] = validate_capacity(value)
        save(receipt_path, receipt)


def worker(path, job, slot, number):
    value = authorized(path, job, slot, number)
    folder = attempt_path(value, job, number)
    receipt_path = folder / 'worker.json'
    receipt = dict(job=job, slot=slot, gpu_uuid=require_slot(slot)['uuid'], gpu_type='3090',
        actual_gpu_type='3090', historical_protocol_gpu_type='3090' if job == 'A_scratch' else
            ('4090' if job in ARMS else None), hardware_explanation=HARDWARE_NOTE,
        state='checking_resources', initialized=False, stages=[], started_utc=utc(), automatic_retry=False)
    if job in value['links']:
        receipt['resume_link'] = value['links'][job]
    save(receipt_path, receipt, exclusive=True)
    try:
        with original_mutex(slot, value['mutex_inodes']) as fd:
            receipt['checks'] = worker_checks(value, slot, folder)
            check_start(value, job)
            env = environment(SLOTS[slot]['uuid'])
            env['R113_3090_MUTEX_FD'] = str(fd)
            if job in ARMS:
                original.check_cache(value)
                ensure_capacity(value, receipt, receipt_path, env, fd)
                # A peer may have been writing the witness while this job waited.
                authorized(path, job, slot, number)
                check_start(value, job)
            else:
                env['R114_CPU_IDS'] = ','.join(map(str, receipt['checks']['cpus'][:2]))
                for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
                    env[key] = '1'
            for row in value['commands'][job][slot]:
                authorized(path, job, slot, number)
                row = dict(row, argv=list(row['argv']))
                if job == 'R114_labels':
                    row['argv'] = ['taskset', '-c', env['R114_CPU_IDS'],
                        *[env['R114_CPU_IDS'] if x == '{cpu_ids}' else x for x in row['argv']]]
                original.run_stage(row, receipt, receipt_path, env, fd)
                if row['name'] == 'finalize':
                    original.validate_diagnostics(value, job)
            if job == 'R114_labels':
                rc = receipt['stages'][-1]['exit_status']
                receipt['queue_result'] = original.r114_completion(value, rc)
                if rc == 2:
                    receipt.update(state='blocked', exit_status=2, finished_utc=utc(),
                        reason='R114 terminal partial labels/analysis; no retry')
                    return 2
            receipt.update(state='complete', exit_status=0, finished_utc=utc())
            return 0
    except ResourceBusy as error:
        safe = receipt['initialized'] is False and not receipt['stages']
        rc = BUSY_EXIT if safe else 1
        receipt.update(state='resource_busy' if safe else 'failed', error=str(error), exit_status=rc, finished_utc=utc())
        return rc
    except BaseException as error:
        receipt.update(state='failed', error=str(error), exit_status=1, finished_utc=utc())
        raise
    finally:
        save(receipt_path, receipt)


def safe_deferral(folder, rc, log):
    if 'srun: error: Unable to create step for job ' in log and 'for job 1465:' not in log:
        return False
    return original.safe_deferral(folder, rc, log)


def finish_attempt(state, folder, rc, log):
    if safe_deferral(folder, rc, log):
        state.update(state='resource_busy', reason='Proved pre-init contention only; not a training retry')
        return
    done = read(folder / 'worker.json') if (folder / 'worker.json').is_file() else {}
    if rc in (0, 2) and done.get('exit_status') == rc and done.get('state') == ('complete' if rc == 0 else 'blocked'):
        state.update(state=done['state'], exit_status=rc, finished_utc=utc())
    else:
        state.update(state='failed', exit_status=rc or 1, finished_utc=utc(),
            reason=done.get('error', 'Missing/failed worker completion; no automatic retry'))


def controller_tick(path, state, active):
    plan = load_plan(path)
    for job, (child, stream, folder) in list(active.items()):
        rc = child.poll()
        if rc is None:
            continue
        stream.close()
        finish_attempt(state['jobs'][job], folder, rc, (folder / 'transport.log').read_text())
        del active[job]
    if not (directory(plan) / 'ready.json').is_file():
        state['state'] = 'waiting_main'
        return False
    value = validated_ready(path)
    occupied = {state['jobs'][j]['slot'] for j in active}
    for job in JOBS:
        current = state['jobs'][job]
        if current['state'] in ('running', 'complete', 'failed', 'blocked'):
            continue
        for slot in SLOTS:
            if slot in occupied:
                continue
            try:
                with original_mutex(slot, value['mutex_inodes']):
                    pass
            except ResourceBusy:
                continue
            number = current.get('attempts', 0) + 1
            folder = attempt_path(value, job, number)
            save(folder / 'request.json', dict(job=job, slot=slot, created_utc=utc(), initialized=False,
                ready_sha256=digest(directory(plan) / 'ready.json')), exclusive=True)
            stream = (folder / 'transport.log').open('x')
            env = environment()
            env[TOKEN_ENV] = value['token']
            current.update(state='running', slot=slot, attempts=number, gpu_uuid=SLOTS[slot]['uuid'],
                gpu_type='3090', allocation=1465, attempt_dir=str(folder))
            save(directory(plan) / 'controller.json', state)
            try:
                child = subprocess.Popen(srun_command(path, job, slot, number), cwd=REPO,
                    env=env, stdout=stream, stderr=subprocess.STDOUT)
            except BaseException:
                stream.close()
                current.update(state='failed', reason='Transport start failed; no implicit retry')
                raise
            current['transport_pid'] = child.pid
            active[job] = (child, stream, folder)
            occupied.add(slot)
            break
        if current['state'] not in ('running', 'complete', 'failed', 'blocked'):
            current.update(state='resource_busy', reason='Awaiting original UUID mutex; R109 untouched')
    terminal = all(row['state'] in ('complete', 'blocked', 'failed') for row in state['jobs'].values())
    if terminal:
        state['state'] = ('failed' if any(r['state'] == 'failed' for r in state['jobs'].values()) else
                          'blocked' if any(r['state'] == 'blocked' for r in state['jobs'].values()) else 'complete')
    else:
        state['state'] = 'running' if active else 'waiting_resources'
    return terminal


def watcher_commands(value):
    rows = [dict(name='R113', argv=[sys.executable, '-B', '-u', str(ADAPTER), 'wait', '--publish']),
            dict(name='R114', argv=[sys.executable, '-B', '-u', '-m', 'code.V4.r114_finish',
                 'wait', '--publish', '--timeout-hours', '336'])]
    # Main/Turing already own R112 publication. Record their argv, never duplicate it.
    return rows


def start_watchers(value):
    """Separate CPU-only one-shot saved-output watchers; no automatic restart."""
    root = directory(value['plan']) / 'watchers'
    root.mkdir(exist_ok=True)
    if (root / 'started.json').exists():
        raise RuntimeError('Watchers already attempted; main must inspect, never auto-restart')
    rows = watcher_commands(value)
    record = dict(started_utc=utc(), cpu_only=True, gpu_started=False, watchers=rows)
    save(root / 'started.json', record, exclusive=True)
    for row in rows:
        with (root / (row['name'] + '.log')).open('x') as log:
            child = subprocess.Popen(row['argv'], cwd=REPO, env=environment(),
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        row['pid'] = child.pid
        save(root / 'started.json', record)
    return record


def wait(path, poll_seconds):
    original.require_controller()
    if not math.isfinite(poll_seconds) or poll_seconds < 1:
        raise ValueError('Finite poll interval >=1 required')
    plan = load_plan(path)
    root = directory(plan)
    with (root / 'controller.lock').open('a') as claim:
        fcntl.flock(claim, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = dict(state='waiting_main', pid=os.getpid(), host=socket.gethostname(), started_utc=utc(),
            jobs={job: dict(state='waiting_main', attempts=0) for job in JOBS}, automatic_training_retry=False)
        save(root / 'controller.json', state, exclusive=True)
        active, watchers = {}, False
        try:
            while True:
                terminal = controller_tick(path, state, active)
                if (root / 'ready.json').is_file() and not watchers:
                    state['watchers'] = start_watchers(validated_ready(path))
                    watchers = True
                state['updated_utc'] = utc()
                save(root / 'controller.json', state)
                print(json.dumps(dict(state=state['state'], jobs={j: r['state'] for j, r in state['jobs'].items()})), flush=True)
                if terminal:
                    return 0 if state['state'] == 'complete' else 1
                time.sleep(poll_seconds)
        except BaseException as error:
            state.update(state='draining_after_error', error=str(error))
            save(root / 'controller.json', state)
            for job, (child, log, folder) in active.items():
                rc = wait_owned(child)
                log.close()
                finish_attempt(state['jobs'][job], folder, rc, (folder / 'transport.log').read_text())
            state.update(state='failed', finished_utc=utc())
            save(root / 'controller.json', state)
            raise


def status(path):
    plan = load_plan(path)
    folder = directory(plan)
    return dict(plan=str(path), launches_work=False, slots=SLOTS, ready_exists=(folder / 'ready.json').is_file(),
        controller=read(folder / 'controller.json') if (folder / 'controller.json').is_file() else None,
        hardware_explanation=HARDWARE_NOTE, automatic_training_retry=False)


def parser():
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument('command', choices=('prepare', 'plan', 'status', 'check-cli', 'ready', 'wait', 'entry', 'worker'))
    value.add_argument('--plan', type=Path, default=PLAN)
    value.add_argument('--operations-receipt', type=Path, default=OPS)
    value.add_argument('--r112-readiness', type=Path, default=R112_READY)
    value.add_argument('--job', choices=JOBS)
    value.add_argument('--slot', choices=tuple(SLOTS))
    value.add_argument('--attempt', type=int)
    value.add_argument('--poll-seconds', type=float, default=60)
    value.add_argument('--confirm-check')
    value.add_argument('--runner-tests-passed', action='store_true')
    value.add_argument('--evidence')
    return value


def main(argv=None):
    args = parser().parse_args(argv)
    path = args.plan.expanduser().resolve()
    if args.command == 'prepare':
        if path != PLAN:
            raise ValueError('Only the owned new migration plan can be created')
        save(path, default_plan(args.r112_readiness, args.operations_receipt), exclusive=True)
        print(json.dumps(dict(plan=str(path), main_approved=False, gpu_started=False)))
        return 0
    if args.command in ('entry', 'worker') and (not args.job or not args.slot or args.attempt is None):
        raise ValueError('Internal transport requires explicit job/slot/attempt and new main token')
    if args.command in ('plan', 'status', 'check-cli', 'ready'):
        if args.command == 'plan':
            result = dict(load_plan(path), launches_work=False)
        elif args.command == 'status':
            result = status(path)
        elif args.command == 'ready':
            result = ready(path, args)
        else:
            result = check_cli(path)
            save(HERE / 'cpu_check.json', result)
            result = {k: v for k, v in result.items() if k not in
                      ('plan', 'source_sha256', 'commands', 'links', 'r112', 'r114')}
        print(json.dumps(result, indent=2))
        return 0

    def interrupted(signum, frame):
        raise RuntimeError('Interrupted signal ' + str(signum) + '; drain owned children, no kill/retry')

    for number in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT):
        signal.signal(number, interrupted)
    if args.command == 'wait':
        return wait(path, args.poll_seconds)
    if args.command == 'entry':
        return entry(path, args.job, args.slot, args.attempt)
    return worker(path, args.job, args.slot, args.attempt)


if __name__ == '__main__':
    raise SystemExit(main())

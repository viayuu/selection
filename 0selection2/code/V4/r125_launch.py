"""One-shot R125 controller and allocation-owned, single-UUID worker.

``probe`` and ``validate`` never train. Only a main-confirmed ``start`` can
dispatch train -> idempotent test -> timing -> LIB -> local CPU report.
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
import shutil
import signal
import socket
import subprocess
import sys
import time
import uuid

from . import r84_launch as base


REPO = base.REPO
ROOT = REPO / 'code/V4/runs/R125_no_bq_lehd_source_embedding'
PREP = ROOT / 'preparation/launch'
WRAPPER = REPO / 'run_v4_r125.sh'
ARM, SLOT = 'code_seed2', '4090_0'
ITEM = dict(base.SLOTS[SLOT])
CONTAINER = base.transport.CONTAINERS[ITEM['node']][1]
MUTEX = base.transport.ROOT / ('queue_gpu_' + ITEM['uuid'] + '.json')
CACHE = Path('/tmp/shiys_R84_new90k_cache')
HOST_CACHE = Path('/host/tmp/shiys_R84_new90k_cache')
CACHE_SOURCE = REPO / 'code/V4/runs/R84_new90k_attention/input_cache'
R84_PREPARED = REPO / 'code/V4/runs/R84_new90k_attention/prepared'
DATA = Path('/public/home/shiys/selection_data_new90k_no_bq_lehd_seed2_20261112')
TOKEN_ENV = 'R125_LAUNCH_TOKEN'
STAGES = ('train', 'test', 'timing', 'lib', 'report')
GIB = 2**30
read, save, sha, utc = base.read, base.save, base.sha, base.utc


def environment(gpu=False, token=None):
    env = base.environment(SLOT, token='bound' if gpu else None)
    env.pop(base.TOKEN_ENV, None)
    env['WANDB_MODE'] = 'offline'
    if token:
        env[TOKEN_ENV] = token
    return env


def container_command(stage, *arguments, gpu=False, token=None):
    env = environment(gpu, token)
    names = ('CUDA_VISIBLE_DEVICES', 'PYTHONPATH', 'PYTHONDONTWRITEBYTECODE',
        'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
        'NUMEXPR_NUM_THREADS', 'PYTHONUNBUFFERED', 'MPLBACKEND',
        'PYTORCH_CUDA_ALLOC_CONF', 'WANDB_PROJECT', 'WANDB_ENTITY', 'WANDB_MODE')
    command = ['sudo', '-n', '/usr/bin/ai_docker', 'exec', '-u', str(os.getuid())]
    for name in names:
        command += ['-e', name + '=' + env[name]]
    command += ['-e', 'SLURM_JOB_ID=' + str(ITEM['job'])]
    if token:
        command += ['-e', TOKEN_ENV + '=' + token]
    # ai_docker splits shell -c payloads. All activation lives on disk.
    return command + [CONTAINER, 'bash', str(WRAPPER), stage, *map(str, arguments)]


def ssh_command(stage, *arguments, gpu=False, token=None):
    return ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10',
        '-o', 'ServerAliveInterval=30', '-o', 'ServerAliveCountMax=3', ITEM['node'],
        shlex.join(container_command(stage, *arguments, gpu=gpu, token=token))]


def mutex_identity():
    if MUTEX.is_symlink() or not MUTEX.is_file():
        raise RuntimeError('Original R84/R81 UUID mutex is missing; never recreate it')
    return dict(path=str(MUTEX), inode=MUTEX.stat().st_ino)


@contextmanager
def original_mutex(expected):
    if expected != mutex_identity():
        raise RuntimeError('Original UUID mutex inode changed')
    fd = os.open(MUTEX, os.O_RDWR | os.O_NOFOLLOW)
    try:
        opened = os.fstat(fd)
        current = MUTEX.stat()
        if (opened.st_dev, opened.st_ino) != (current.st_dev, current.st_ino):
            raise RuntimeError('UUID mutex replaced while opening')
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if os.fstat(fd).st_ino != mutex_identity()['inode']:
            raise RuntimeError('UUID mutex replaced while locking')
        yield fd
    finally:
        os.close(fd)


def cache_inventory(root):
    root = Path(root)
    result = dict(root=str(root), exists=root.is_dir(), usable=False,
        copied_bytes=0, geometry_preprocessing=False, array_contents_read=False)
    if not result['exists']:
        return result
    try:
        stamp = read(root / '.r84_cache_identity.json')
        if (Path(stamp.get('source', '')) != CACHE_SOURCE
                or Path(stamp.get('data_root', '')) != Path(read(R84_PREPARED / 'prepared.json')['data_root'])):
            raise RuntimeError('Not the original R84 NEW90k input cache')
        manifest = read(R84_PREPARED / 'source_cache_manifest.json')
        checked = []
        for split in ('train', 'val', 'test'):
            tasks = manifest['splits'].get(split, {})
            if len(tasks) != 18:
                raise RuntimeError('Original R84 manifest needs all 18 tasks in ' + split)
            for problem, entry in tasks.items():
                directory = root / split / problem
                meta = read(directory / 'metadata.json')
                if meta.get('rows') != entry['rows'] or meta.get('data_root') != manifest['data_root']:
                    raise RuntimeError('R84 cache metadata mismatch: ' + str(directory))
                files = [directory / name for name in ('n.npy', 'costs.npy', 'winner.npy')]
                files += [directory / ('matrix.npy' if problem == 'ATSP' else 'node.npy')]
                for path in files:
                    if not path.is_file() or path.stat().st_size <= 128:
                        raise RuntimeError('Missing original R84 array: ' + str(path))
                checked.append(str(directory.relative_to(root)))
        disk = shutil.disk_usage(root)
        result.update(usable=True, stamp=stamp, checked_directories=checked,
            disk=dict(total_bytes=disk.total, free_bytes=disk.free),
            validation='metadata and small stat-only array inventory; no full array scan')
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        result['error'] = str(error)
    return result


def process_inventory():
    rows = []
    for directory in Path('/proc').glob('[0-9]*'):
        try:
            if directory.stat().st_uid != os.getuid() or int(directory.name) == os.getpid():
                continue
            argv = directory.joinpath('cmdline').read_bytes().decode(errors='replace').split('\0')
            if not argv[0] or not Path(argv[0]).name.startswith(('python', 'taskset')):
                continue
            if 'code.V4.r125_launch' in argv or ('conda' in argv[0] and 'hook' in argv):
                continue
            status = directory.joinpath('status').read_text()
            if re.search(r'^State:\s+[ZX]', status, re.M):
                continue
            rows.append(dict(pid=int(directory.name), cpus=sorted(os.sched_getaffinity(int(directory.name))),
                command=[arg for arg in argv if arg]))
        except (OSError, ProcessLookupError):
            continue
    return rows


def container_probe(receipt):
    identity = base.assert_container(SLOT)
    # Slurm clients are on the controller, not in this existing container.
    allowed = sorted(os.sched_getaffinity(0))
    memory = base.memory_inventory(identity)
    caches = [cache_inventory(path) for path in (CACHE, HOST_CACHE)]
    usable = next((row for row in caches if row['usable']), None)
    processes = process_inventory()
    occupied = {cpu for row in processes for cpu in row['cpus']}
    cpus = sorted(set(allowed) - occupied)[:4]
    result = dict(captured_utc=utc(), container_id=identity, hostname=socket.gethostname(),
        cgroup=Path('/proc/self/cgroup').read_text(),
        allowed_cpus=allowed, available_cpus=sorted(set(allowed) - occupied), selected_cpus=cpus,
        processes=processes, memory=memory, gpu=base.transport.gpu_inventory(ITEM),
        caches=caches, selected_cache=usable['root'] if usable else None,
        mutex=mutex_identity(), no_new_allocation=True, training_started=False)
    save(receipt, result, exclusive=True)
    return result


def probe():
    allocation = base.transport.allocation(ITEM)
    path = PREP / 'probes' / (time.strftime('%Y%m%dT%H%M%S') + '_' + uuid.uuid4().hex[:8] + '.json')
    path.parent.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(ssh_command('container-probe', '--receipt', path),
        cwd=REPO, env=environment(), text=True, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, timeout=90)
    log = path.with_suffix('.log')
    log.write_text(completed.stdout)
    # Some platform exec wrappers mask the nested return code. A new, matching
    # shared-filesystem receipt, not rc0 alone, proves successful execution.
    if completed.returncode or not path.is_file():
        raise RuntimeError('Allocation container probe failed; see ' + str(log))
    result = read(path)
    if result.get('container_id') != CONTAINER or result.get('mutex') != mutex_identity():
        raise RuntimeError('Container probe receipt identity mismatch')
    if not set(result['allowed_cpus']) <= set(allocation['cpus']):
        raise RuntimeError('Container CPUs exceed the original allocation')
    result['allocation'] = allocation
    save(path, result)
    return dict(receipt=str(path), **result)


def command(stage, cache_root):
    prefix = [sys.executable, '-B', '-u', '-m']
    common = ['--root', str(ROOT), '--arm', ARM]
    if stage == 'publish':
        return prefix + ['code.V4.r125_publish', '--root', str(ROOT), '--push']
    if stage == 'prepare':
        return prefix + ['code.V4.r125_data', 'prepare', '--data-root', str(DATA),
            '--prepared-root', str(ROOT / 'prepared'), '--cache-root', str(cache_root)]
    if stage in ('lib', 'report'):
        return prefix + ['code.V4.r125_reporting', stage, *common,
            '--device', 'cpu' if stage == 'report' else 'cuda:0',
            *(['--lib-batch-size', '2'] if stage == 'lib' else [])]
    return prefix + ['code.V4.r125_experiment', stage, *common,
        '--prepared-root', str(ROOT / 'prepared'), '--data-root', str(DATA),
        '--cache-root', str(cache_root), '--device', 'cuda:0', '--gpu-type', '4090',
        '--large-micro-cap', '256', '--eval-batch-size', '128', '--log-every', '18',
        '--wandb-mode', 'offline']


def source_hashes():
    names = ('r125_launch.py', 'r125_experiment.py', 'r125_data.py',
        'r84_launch.py', 'r81_queue.py', 'r84_experiment.py', 'r80_experiment.py',
        'r76_experiment.py', 'r77_experiment.py', 'solver_code_encoder.py',
        'solver_code_embeddings.py', 'pairwise_selector.py', 'pairwise_objective.py')
    paths = [REPO / 'code/V4' / name for name in names] + [WRAPPER, ROOT / 'PLAN.md',
        DATA / 'scenario_manifest.json']
    return {str(path): sha(path) for path in paths}


def latest_probe(path=None):
    paths = [path] if path else sorted((PREP / 'probes').glob('*.json'), reverse=True)
    for candidate in paths:
        value = read(candidate)
        if value.get('container_id') == CONTAINER and value.get('selected_cache'):
            if value['selected_cache'] not in (str(CACHE), str(HOST_CACHE)):
                raise RuntimeError('Probe selected an unauthorized cache path')
            if value['mutex'] != mutex_identity() or len(value['selected_cpus']) != 4:
                raise RuntimeError('Probe CPU/mutex identity mismatch')
            return dict(value, receipt=str(candidate))
    raise RuntimeError('Run probe once to select the actual existing input cache')


def fresh_train():
    directory = ROOT / ARM
    if directory.exists() and any(directory.iterdir()):
        raise RuntimeError('Existing model output: no implicit fresh retraining or resume')
    if (PREP / 'train/request.json').exists():
        raise RuntimeError('Training was already requested; never automatically train twice')


def check_request(attempt):
    request = read(attempt / 'request.json')
    if request['mode'] not in ('validate', 'train', 'finish'):
        raise RuntimeError('Unknown launch mode')
    if request['mode'] != 'validate':
        for path, digest in request['sources'].items():
            if sha(path) != digest:
                raise RuntimeError('Source changed after main confirmation: ' + path)
    return request


def start(mode, evidence=None, tested=False, probe_receipt=None):
    if mode != 'validate' and (not evidence or not tested):
        raise ValueError('Require --main-ready-evidence and --runner-tests-passed')
    resources = latest_probe(probe_receipt)
    base.transport.allocation(ITEM)
    if mode == 'train':
        fresh_train()
        scenario = read(DATA / 'scenario_manifest.json')
        if scenario.get('experiment') != 'R125' or scenario.get('status') != 'complete':
            raise RuntimeError('Main must finish the mechanical R125 data build first')
    if mode == 'finish' and not (ROOT / ARM / 'training_complete.json').is_file():
        raise RuntimeError('Finish never trains; completed training is required')
    name = 'train' if mode == 'train' else mode + '_' + uuid.uuid4().hex[:8]
    attempt = PREP / name
    session = 'r125_' + name
    request = dict(mode=mode, token=uuid.uuid4().hex, evidence=evidence,
        runner_tests_passed=tested, cpus=resources['selected_cpus'],
        cache_root=resources['selected_cache'], probe_receipt=resources['receipt'],
        mutex=resources['mutex'], sources=source_hashes() if mode != 'validate' else {},
        created_utc=utc(), automatic_retry=False, no_new_allocation=True,
        training_started=False, session=session)
    with base.lock(PREP / 'start.lock'):
        if mode == 'train':
            fresh_train()
        save(attempt / 'request.json', request, exclusive=True)
        argv = ['bash', str(WRAPPER), 'controller', '--attempt', str(attempt)]
        subprocess.run(['tmux', 'new-session', '-d', '-s', session, shlex.join(argv)], check=True)
        # The controller exits with the true terminal status in controller.json.
        save(attempt / 'tmux.json', dict(session=session, argv=argv, created_utc=utc()), exclusive=True)
    return dict(session=session, attempt=str(attempt), mode=mode,
        cache_root=request['cache_root'], training_started=False)


def stop_owned(child):
    if child.poll() is None:
        os.killpg(child.pid, signal.SIGTERM)
        try:
            child.wait(timeout=15)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait()


def first_cycle(attempt):
    path = ROOT / ARM / 'status.json'
    if not path.is_file():
        return False
    status = read(path)
    if status.get('successful_updates', 0) < 18:
        return False
    keys = ('loss', 'ce', 'cmp', 'risk', 'kl', 'grad_norm')
    counts = status.get('updates_by_problem', {})
    if len(counts) != 18 or min(counts.values()) < 1:
        raise RuntimeError('First full 18-task cycle lacks a successful update per task')
    if any(not isinstance(status.get(key), (int, float)) or not math.isfinite(status[key]) for key in keys):
        raise RuntimeError('First 18-task cycle has missing/nonfinite loss or gradient')
    save(attempt / 'startup_cycle.json', dict(passed=True, checked_utc=utc(),
        successful_updates=status['successful_updates'], updates_by_problem=counts,
        metrics={key: status[key] for key in keys}, extra_capacity_probe=False), exclusive=True)
    return True


def run_stage(stage, attempt, record, request, lock_fd=None):
    argv = command(stage, request['cache_root'])
    log = attempt / (stage + '.log')
    row = dict(stage=stage, state='running', argv=argv, log=str(log), started_utc=utc())
    record['stages'].append(row)
    record.update(state='running', phase=stage)
    receipt = attempt / ('controller.json' if stage in ('report', 'publish') else 'worker.json')
    save(receipt, record)
    limits = dict(prepare=1800, train=72 * 3600, test=7200, timing=3600, lib=7200, report=3600, publish=1800)
    started = time.monotonic()
    with log.open('a', buffering=1) as stream:
        stream.write('\n[R125] ' + utc() + ' ' + shlex.join(argv) + '\n')
        child = subprocess.Popen(argv, cwd=REPO,
            env=environment(stage not in ('prepare', 'report', 'publish'), request['token']),
            stdout=stream, stderr=subprocess.STDOUT, start_new_session=True,
            pass_fds=() if lock_fd is None else (lock_fd,))
        row['pid'] = child.pid
        save(receipt, record)
        cycle_ok = stage != 'train'
        try:
            while child.poll() is None:
                if (attempt / 'abort.json').exists():
                    raise InterruptedError('Controller requested cancellation; stopping only owned stage')
                if not cycle_ok:
                    cycle_ok = first_cycle(attempt)
                    if not cycle_ok and time.monotonic() - started > 1800:
                        raise TimeoutError('No finite first 18-task cycle within 1800 seconds')
                if time.monotonic() - started > limits[stage]:
                    raise TimeoutError('Stage exceeded finite wall-time budget: ' + stage)
                time.sleep(1)
            rc = child.returncode
            if stage == 'train' and rc == 0 and not cycle_ok:
                cycle_ok = first_cycle(attempt)
                if not cycle_ok:
                    raise RuntimeError('Training ended without the first-cycle witness')
            row.update(state='complete' if rc == 0 else 'failed', exit_status=rc, finished_utc=utc())
            save(receipt, record)
            if rc:
                raise RuntimeError(stage + ' failed with exit=' + str(rc) + '; no automatic retry')
        except BaseException as error:
            stop_owned(child)
            row.update(state='failed', error=str(error), exit_status=child.returncode, finished_utc=utc())
            save(receipt, record)
            raise


def worker(attempt):
    request = check_request(attempt)
    if os.environ.get(TOKEN_ENV) != request['token']:
        raise RuntimeError('Worker token does not match this main-owned request')
    base.assert_container(SLOT)
    record = dict(state='checking', mode=request['mode'], pid=os.getpid(),
        started_utc=utc(), stages=[], automatic_retry=False, training_started=False)
    save(attempt / 'worker.json', record, exclusive=True)
    try:
        with original_mutex(request['mutex']) as fd:
            cpus = set(request['cpus'])
            if not cpus <= set(os.sched_getaffinity(0)):
                raise RuntimeError('Requested allocation CPUs no longer available')
            processes = process_inventory()
            if any(cpus & set(row['cpus']) for row in processes):
                raise RuntimeError('Selected CPUs are now occupied; do not interfere with other tasks')
            os.sched_setaffinity(0, cpus)
            gpu = base.transport.gpu_inventory(ITEM)
            if gpu['compute_processes'] or gpu['memory_used_mib'] > 512 or gpu['utilization_pct']:
                raise RuntimeError('Selected UUID became busy; no fallback GPU or retry')
            if base.memory_inventory(CONTAINER)['headroom_excluding_reclaimable_cache_bytes'] < 9 * GIB:
                raise RuntimeError('Not enough allocation RSS headroom')
            cache_root = Path(request['cache_root'])
            stamp = read(cache_root / '.r84_cache_identity.json')
            if Path(stamp['source']) != CACHE_SOURCE or any(not (cache_root / split).is_dir()
                    for split in ('train', 'val', 'test')):
                raise RuntimeError('Original cache disappeared; never regenerate geometry')
            record.update(cpus=sorted(cpus), gpu_uuid=ITEM['uuid'], container_id=CONTAINER,
                cache_root=str(cache_root), cache_copied_bytes=0, geometry_preprocessing=False)
            if request['mode'] == 'validate':
                from . import r125_experiment as runner
                parsed = runner.parser().parse_args(command('train', cache_root)[5:])
                if (parsed.resume or parsed.cache_root != cache_root or runner.BATCH_SIZE != 1024
                        or runner.MAX_UPDATES != 57600 or runner.WARMUP_UPDATES != 4320
                        or runner.micro_caps('4090', 256).get('ATSP') != 1024):
                    raise RuntimeError('Runner CLI/default recipe differs from the requested R125')
                record.update(state='complete', exit_status=0, validation='CPU parser, binding, lock, cache; no training')
                return 0
            stages = ('prepare', 'train', 'test', 'timing', 'lib') if request['mode'] == 'train' else ('test', 'timing', 'lib')
            for stage in stages:
                check_request(attempt)
                if stage == 'lib' and not (REPO / 'code/V4/r125_reporting.py').is_file():
                    record.update(state='awaiting_reporting', phase='lib', exit_status=69,
                        error='Reporting module missing; use explicit finish later, never train again')
                    return 69
                if stage == 'train':
                    prepared = read(ROOT / 'prepared/prepared.json')
                    if prepared['cache_source'] != request['cache_root']:
                        raise RuntimeError('Prepared cache path differs from the actual container cache')
                    record['prepared_sha256'] = sha(ROOT / 'prepared/prepared.json')
                    record['training_started'] = True
                run_stage(stage, attempt, record, request, fd)
            record.update(state='complete', exit_status=0)
            return 0
    except BaseException as error:
        record.update(state='failed', exit_status=1, error_type=type(error).__name__, error=str(error))
        raise
    finally:
        record['finished_utc'] = utc()
        save(attempt / 'worker.json', record)


def controller(attempt):
    request = check_request(attempt)
    record = dict(state='running', pid=os.getpid(), host=socket.gethostname(),
        mode=request['mode'], started_utc=utc(), stages=[], automatic_retry=False)
    save(attempt / 'controller.json', record, exclusive=True)
    try:
        with base.lock(attempt / 'controller.lock'):
            base.transport.allocation(ITEM)
            argv = ssh_command('worker', '--attempt', attempt,
                gpu=request['mode'] != 'validate', token=request['token'])
            log = attempt / 'transport.log'
            record.update(argv=argv, log=str(log))
            save(attempt / 'controller.json', record)
            with log.open('a') as stream:
                child = subprocess.Popen(argv, cwd=REPO, env=environment(),
                    stdout=stream, stderr=subprocess.STDOUT)
                record['transport_pid'] = child.pid
                save(attempt / 'controller.json', record)
                rc = child.wait()
            path = attempt / 'worker.json'
            if not path.is_file():
                raise RuntimeError('SSH/container exec ended without a worker receipt; never retry')
            nested = read(path)
            record['worker_state'] = nested['state']
            if nested.get('state') == 'awaiting_reporting':
                record.update(state='awaiting_reporting', exit_status=69)
                return 69
            if rc or nested.get('state') != 'complete' or nested.get('exit_status') != 0:
                raise RuntimeError('Worker failed or transport broke; inspect worker.json and logs; no automatic retry')
            if request['mode'] != 'validate':
                run_stage('report', attempt, record, request)
                run_stage('publish', attempt, record, request)
            record.update(state='complete', exit_status=0)
            return 0
    except BaseException as error:
        if request['mode'] != 'validate':
            save(attempt / 'abort.json', dict(error=str(error), created_utc=utc()))
        record.update(state='failed', exit_status=1, error_type=type(error).__name__, error=str(error))
        raise
    finally:
        record['finished_utc'] = utc()
        save(attempt / 'controller.json', record)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('command', choices=('probe', 'container-probe', 'validate', 'start',
        'finish', 'controller', 'worker'))
    result.add_argument('--receipt', type=Path)
    result.add_argument('--attempt', type=Path)
    result.add_argument('--probe-receipt', type=Path)
    result.add_argument('--main-ready-evidence')
    result.add_argument('--runner-tests-passed', action='store_true')
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    if args.command in ('worker', 'controller'):
        if args.attempt is None:
            raise ValueError('--attempt is required')
        def interrupted(signum, frame):
            raise InterruptedError('Received signal ' + str(signum))
        signal.signal(signal.SIGTERM, interrupted)
        signal.signal(signal.SIGINT, interrupted)
        return globals()[args.command](args.attempt)
    if args.command in ('validate', 'start', 'finish'):
        mode = 'train' if args.command == 'start' else args.command
        result = start(mode, args.main_ready_evidence, args.runner_tests_passed, args.probe_receipt)
    else:
        result = container_probe(args.receipt) if args.command == 'container-probe' else probe()
    print(json.dumps(result, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

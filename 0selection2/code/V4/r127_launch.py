"""One-shot R127/R83 model-only launch in gpu03/job1465's existing container.

Private transport globals never modify the frozen R125/R126 launch modules.
Probe is read-only; formal training requires an explicit main confirmation.
"""

import argparse
import importlib.util
import json
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import sys
import time
import uuid


REPO = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('code.V4._r127_transport',
    REPO / 'code/V4/r125_launch.py')
transport = importlib.util.module_from_spec(spec)
spec.loader.exec_module(transport)
base = transport.base
ROOT = REPO / 'code/V4/runs/R127_R83_no_mvmoe'
PREP = ROOT / 'preparation/launch'
DATA = Path('/public/home/shiys/selection_data_new90k_no_mvmoe_seed2_20261112')
WRAPPER = REPO / 'run_v4_r127.sh'
ARM, SLOT = 'model_only_seed2', '3090_0'
ITEM = dict(base.SLOTS[SLOT])
CPUS = [12, 13, 14, 15]
TOKEN_ENV = 'R127_LAUNCH_TOKEN'
CONTAINER = base.transport.CONTAINERS[ITEM['node']][1]
MUTEX = base.transport.ROOT / ('queue_gpu_' + ITEM['uuid'] + '.json')
CACHE = transport.CACHE
read, save, sha, utc = transport.read, transport.save, transport.sha, transport.utc

for name in ('ROOT', 'PREP', 'DATA', 'WRAPPER', 'ARM', 'SLOT', 'ITEM',
             'TOKEN_ENV', 'CONTAINER', 'MUTEX'):
    setattr(transport, name, globals()[name])


def command(stage, cache_root):
    prefix = [sys.executable, '-B', '-u', '-m']
    if stage == 'prepare':
        return prefix + ['code.V4.r127_data', 'prepare', '--data-root', str(DATA),
            '--prepared-root', str(ROOT / 'prepared'), '--cache-root', str(cache_root)]
    if stage == 'publish':
        return prefix + ['code.V4.r127_publish', '--root', str(ROOT), '--push']
    common = ['--root', str(ROOT), '--arm', ARM]
    if stage in ('lib', 'report'):
        return prefix + ['code.V4.r127_reporting', stage, *common,
            '--device', 'cpu' if stage == 'report' else 'cuda:0',
            *(['--lib-batch-size', '2'] if stage == 'lib' else [])]
    if stage not in ('train', 'test', 'timing'):
        raise ValueError('Unknown R127 stage: ' + stage)
    return prefix + ['code.V4.r127_experiment', stage, *common,
        '--prepared-root', str(ROOT / 'prepared'), '--data-root', str(DATA),
        '--cache-root', str(cache_root), '--device', 'cuda:0', '--gpu-type', '3090',
        '--large-micro-cap', '256', '--eval-batch-size', '128', '--log-every', '18',
        '--wandb-mode', 'offline']


def source_hashes():
    names = ('r127_launch.py', 'r127_experiment.py', 'r127_data.py',
        'r125_launch.py', 'r84_launch.py', 'r81_queue.py', 'r83_experiment.py',
        'r84_experiment.py', 'r80_experiment.py', 'r76_experiment.py', 'r77_experiment.py',
        'solver_code_encoder.py', 'solver_code_embeddings.py',
        'pairwise_selector.py', 'pairwise_objective.py')
    paths = [REPO / 'code/V4' / name for name in names] + [WRAPPER, ROOT / 'PLAN.md',
        DATA / 'scenario_manifest.json']
    return {str(path): sha(path) for path in paths}


def process_inventory():
    rows = []
    for directory in Path('/proc').glob('[0-9]*'):
        try:
            if directory.stat().st_uid != os.getuid() or int(directory.name) == os.getpid():
                continue
            argv = directory.joinpath('cmdline').read_bytes().decode(errors='replace').split('\0')
            if not argv[0] or not Path(argv[0]).name.startswith(('python', 'taskset')):
                continue
            if 'code.V4.r127_launch' in argv or ('conda' in argv[0] and 'hook' in argv):
                continue
            if re.search(r'^State:\s+[ZX]', directory.joinpath('status').read_text(), re.M):
                continue
            rows.append(dict(pid=int(directory.name), cpus=sorted(os.sched_getaffinity(int(directory.name))),
                command=[arg for arg in argv if arg]))
        except (OSError, ProcessLookupError):
            continue
    return rows


def cache_check(root):
    root = Path(root)
    if root != CACHE:
        raise RuntimeError('R127 must keep the actual original /tmp cache path')
    cache = transport.cache_inventory(root)
    if not cache['usable']:
        raise RuntimeError('R127 original cache unavailable; no copy or geometry rebuild: ' + str(cache.get('error', root)))
    return cache


def container_probe(receipt):
    identity = base.assert_container(SLOT)
    allowed = sorted(os.sched_getaffinity(0))
    processes = process_inventory()
    occupied = {cpu for row in processes for cpu in row['cpus']}
    cpus = CPUS if set(CPUS) <= set(allowed) - occupied else []
    cache = cache_check(CACHE)
    value = dict(captured_utc=utc(), experiment='R127', container_id=identity,
        hostname=transport.socket.gethostname(), cgroup=Path('/proc/self/cgroup').read_text(),
        allowed_cpus=allowed, available_cpus=sorted(set(allowed) - occupied), selected_cpus=cpus,
        processes=processes, memory=base.memory_inventory(identity),
        gpu=base.transport.gpu_inventory(ITEM), selected_cache=str(CACHE), caches=[cache],
        mutex=transport.mutex_identity(), no_new_allocation=True,
        training_started=False, other_tasks_stopped=False)
    save(receipt, value, exclusive=True)
    return value


def start(mode, evidence=None, tested=False, probe_receipt=None):
    if not evidence or not tested:
        raise ValueError('Require --main-ready-evidence and --runner-tests-passed')
    resources = transport.latest_probe(probe_receipt)
    if resources['selected_cpus'] != CPUS or resources['selected_cache'] != str(CACHE):
        raise RuntimeError('Require the R127 gpu03 CPU quartet and actual /tmp cache')
    base.transport.allocation(ITEM)
    if mode == 'train':
        transport.fresh_train()
        scenario = read(DATA / 'scenario_manifest.json')
        if scenario.get('experiment') != 'R127' or scenario.get('status') != 'complete':
            raise RuntimeError('Main must complete the R127 no-MVMOE data build first')
    elif mode != 'finish' or not (ROOT / ARM / 'training_complete.json').is_file():
        raise RuntimeError('Finish never trains; completed R127 training is required')
    name = 'train' if mode == 'train' else 'finish_' + uuid.uuid4().hex[:8]
    attempt, session = PREP / name, 'r127_' + name
    request = dict(mode=mode, token=uuid.uuid4().hex, evidence=evidence,
        runner_tests_passed=tested, experiment='R127', cpus=CPUS,
        cache_root=str(CACHE), probe_receipt=resources['receipt'], mutex=resources['mutex'],
        sources=source_hashes(), created_utc=utc(), automatic_retry=False,
        no_new_allocation=True, training_started=False, session=session)
    with base.lock(PREP / 'start.lock'):
        if mode == 'train':
            transport.fresh_train()
        save(attempt / 'request.json', request, exclusive=True)
        argv = ['bash', str(WRAPPER), 'controller', '--attempt', str(attempt)]
        subprocess.run(['tmux', 'new-session', '-d', '-s', session, shlex.join(argv)], check=True)
        save(attempt / 'tmux.json', dict(session=session, argv=argv, created_utc=utc()), exclusive=True)
    return dict(session=session, attempt=str(attempt), mode=mode,
        cache_root=str(CACHE), cpus=CPUS, gpu_uuid=ITEM['uuid'], training_started=False)


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
        stream.write('\n[R127] ' + utc() + ' ' + shlex.join(argv) + '\n')
        child = subprocess.Popen(argv, cwd=REPO,
            env=transport.environment(stage not in ('prepare', 'report', 'publish'), request['token']),
            stdout=stream, stderr=subprocess.STDOUT, start_new_session=True,
            pass_fds=() if lock_fd is None else (lock_fd,))
        row['pid'] = child.pid
        save(receipt, record)
        cycle_ok = stage != 'train'
        try:
            while child.poll() is None:
                if (attempt / 'abort.json').exists():
                    raise InterruptedError('R127 controller requested cancellation of its owned stage')
                if not cycle_ok:
                    cycle_ok = transport.first_cycle(attempt)
                    if not cycle_ok and time.monotonic() - started > 1800:
                        raise TimeoutError('R127 first finite 18-task cycle exceeded 1800 seconds')
                if time.monotonic() - started > limits[stage]:
                    raise TimeoutError('R127 stage exceeded its finite wall-time budget: ' + stage)
                time.sleep(1)
            if stage == 'train' and child.returncode == 0 and not cycle_ok:
                if not transport.first_cycle(attempt):
                    raise RuntimeError('R127 training ended without the first-cycle witness')
            row.update(state='complete' if child.returncode == 0 else 'failed',
                exit_status=child.returncode, finished_utc=utc())
            save(receipt, record)
            if child.returncode:
                raise RuntimeError('R127 ' + stage + ' failed; no automatic retry')
        except BaseException as error:
            transport.stop_owned(child)
            row.update(state='failed', error=str(error), exit_status=child.returncode, finished_utc=utc())
            save(receipt, record)
            raise


def worker(attempt):
    request = transport.check_request(attempt)
    if request.get('experiment') != 'R127' or os.environ.get(TOKEN_ENV) != request['token']:
        raise RuntimeError('Worker requires its main-owned R127 request and token')
    base.assert_container(SLOT)
    record = dict(experiment='R127', state='checking', mode=request['mode'], pid=os.getpid(),
        started_utc=utc(), stages=[], automatic_retry=False, training_started=False)
    save(attempt / 'worker.json', record, exclusive=True)
    try:
        with transport.original_mutex(request['mutex']) as fd:
            if request['cpus'] != CPUS or not set(CPUS) <= set(os.sched_getaffinity(0)):
                raise RuntimeError('R127 gpu03 CPU quartet changed; no CPU fallback')
            if os.environ.get('CUDA_VISIBLE_DEVICES') != ITEM['uuid']:
                raise RuntimeError('R127 requires its exact single RTX3090 UUID')
            if any(set(CPUS) & set(row['cpus']) for row in process_inventory()):
                raise RuntimeError('R127 CPUs became occupied; no interference or fallback')
            os.sched_setaffinity(0, CPUS)
            gpu = base.transport.gpu_inventory(ITEM)
            if gpu['compute_processes'] or gpu['memory_used_mib'] > 512 or gpu['utilization_pct']:
                raise RuntimeError('R127 GPU became busy; no fallback GPU or automatic retry')
            if base.memory_inventory(CONTAINER)['headroom_excluding_reclaimable_cache_bytes'] < 9 * transport.GIB:
                raise RuntimeError('Insufficient allocation RSS headroom for R127')
            cache_check(request['cache_root'])
            record.update(cpus=CPUS, gpu_uuid=ITEM['uuid'], container_id=CONTAINER,
                cache_root=request['cache_root'], cache_copied_bytes=0, geometry_preprocessing=False)
            stages = ('prepare', 'train', 'test', 'timing', 'lib') if request['mode'] == 'train' else ('test', 'timing', 'lib')
            for stage in stages:
                transport.check_request(attempt)
                if stage == 'lib' and not (REPO / 'code/V4/r127_reporting.py').is_file():
                    record.update(state='awaiting_reporting', phase='lib', exit_status=69,
                        error='R127 reporting missing; use explicit finish later, never train again')
                    return 69
                if stage == 'train':
                    prepared = read(ROOT / 'prepared/prepared.json')
                    if prepared['cache_source'] != request['cache_root']:
                        raise RuntimeError('R127 prepared cache path differs from the actual /tmp cache')
                    record.update(prepared_sha256=sha(ROOT / 'prepared/prepared.json'), training_started=True)
                run_stage(stage, attempt, record, request, fd)
            record.update(state='complete', exit_status=0)
            return 0
    except BaseException as error:
        record.update(state='failed', exit_status=1, error_type=type(error).__name__, error=str(error))
        raise
    finally:
        record['finished_utc'] = utc()
        save(attempt / 'worker.json', record)


for name in ('command', 'source_hashes', 'process_inventory', 'container_probe', 'start', 'run_stage', 'worker'):
    setattr(transport, name, globals()[name])


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('command', choices=('probe', 'container-probe', 'start', 'finish', 'controller', 'worker'))
    result.add_argument('--receipt', type=Path)
    result.add_argument('--attempt', type=Path)
    result.add_argument('--probe-receipt', type=Path)
    result.add_argument('--main-ready-evidence')
    result.add_argument('--runner-tests-passed', action='store_true')
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    if args.command in ('controller', 'worker'):
        if args.attempt is None:
            raise ValueError('--attempt is required')
        def interrupted(signum, frame):
            raise InterruptedError('R127 received signal ' + str(signum))
        signal.signal(signal.SIGTERM, interrupted)
        signal.signal(signal.SIGINT, interrupted)
        return getattr(transport, args.command)(args.attempt)
    if args.command in ('start', 'finish'):
        value = start('train' if args.command == 'start' else 'finish',
            args.main_ready_evidence, args.runner_tests_passed, args.probe_receipt)
    else:
        value = container_probe(args.receipt) if args.command == 'container-probe' else transport.probe()
    print(json.dumps(value, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

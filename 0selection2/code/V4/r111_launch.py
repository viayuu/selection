"""One-shot R111 queue. Imports and plan/check-cli never dispatch work."""

import argparse
from contextlib import ExitStack, contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import signal
import socket
import subprocess
import sys
import time
import uuid


REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / 'code/V4/runs/R111_family_conditional_features'
ARM = 'A_family_conditional'
PRIOR = REPO / 'code/V4/runs/R110_method_conditioned_ce'
UUIDS = ('GPU-71479e7e-f8bd-e430-66ed-f8db9f0d47be',
         'GPU-597c682a-4b7f-7609-80a6-8ced791f5c5d')
CPUS = (0, 1, 2, 3, 4, 5, 72, 73)
CONTAINER = '76045731f6b453443fe18a9bce671599bdbf17c2713b8edcdd090925eea14bf6'
PREPARED = REPO / 'code/V4/runs/R84_new90k_attention/prepared'
CACHE = Path('/tmp/shiys_R84_new90k_cache')
DATA = Path('/public/home/shiys/selection_data_new90k_B_filtered_single_reld_mt_seed2_20261018')
WRAPPER = REPO / 'code/V4/run_v4_r111.sh'
CONDA_SH = '/public/home/shiys/miniconda3/etc/profile.d/conda.sh'
COMPAT = '/public/home/shiys/easynco_v3_bridge/exports/train_100k_20261002/python_compat'
TOKEN_ENV = 'R111_DISPATCH_TOKEN'
STAGES = ('train', 'finalize', 'timing', 'lib', 'report')
LOADED_LAUNCH_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def utc():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, record, exclusive=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(record, indent=2, allow_nan=False) + '\n'
    if exclusive:
        with path.open('x') as stream:
            stream.write(data)
    else:
        temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
        temporary.write_text(data)
        temporary.replace(path)


def dispatch(root):
    # Explicit second entry after the first metadata-only preflight failure;
    # original dispatch receipts remain untouched. This is not an auto-retry.
    return root / 'preparation/dispatch_recovery_01'


def source_files(root):
    return [REPO / 'code/V4' / name for name in
            ('r111_launch.py', 'r111_experiment.py', 'r111_model.py', 'r111_reporting.py')
            ] + [WRAPPER, root / 'PLAN.md']


def source_hashes(root):
    return {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in source_files(root)}


def mutex_paths():
    from . import r81_queue
    paths = [r81_queue.ROOT / f'queue_gpu_{gpu}.json' for gpu in UUIDS]
    if (not all(path.is_file() and not path.is_symlink() for path in paths)
            or paths != [r81_queue.gpu_lock_path(gpu) for gpu in UUIDS]):
        raise RuntimeError('Require the two EXISTING original transport.ROOT UUID mutexes')
    return paths


def mutex_inodes():
    return {str(path): path.stat().st_ino for path in mutex_paths()}


@contextmanager
def gpu_mutexes(expected):
    # Open without O_CREAT, truncation or replacement: historical inode ownership is shared.
    with ExitStack() as stack:
        handles = []
        for path in mutex_paths():
            fd = os.open(path, os.O_RDWR | os.O_NOFOLLOW)
            handle = stack.enter_context(os.fdopen(fd, 'r+'))
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if os.fstat(fd).st_ino != expected.get(str(path)) or path.stat().st_ino != os.fstat(fd).st_ino:
                raise RuntimeError('Original GPU mutex inode changed; no replacement lock')
            handles.append(handle)
        yield [handle.fileno() for handle in handles]


def environment(cuda=''):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=cuda, PYTHONPATH=COMPAT + ':' + str(REPO),
        PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1', MPLBACKEND='Agg',
        OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', NUMEXPR_NUM_THREADS='2',
        PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True', WANDB_PROJECT='selector',
        WANDB_ENTITY='yjkds-southern-university-of-science-technology', WANDB_MODE='offline')
    for key in ('RANK', 'WORLD_SIZE', 'LOCAL_RANK', 'LOCAL_WORLD_SIZE', 'GROUP_RANK',
                'ROLE_RANK', 'ROLE_WORLD_SIZE', 'MASTER_ADDR', 'MASTER_PORT', 'TORCHELASTIC_RUN_ID'):
        env.pop(key, None)
    return env


def commands(root):
    result = []
    for stage in STAGES:
        reporting = stage in ('lib', 'report')
        options = ['--root', str(root), '--arm', ARM]
        if not reporting:
            options += ['--prepared-root', str(PREPARED), '--data-root', str(DATA),
                '--cache-root', str(CACHE), '--gpu-type', '4090', '--large-micro-cap', '256',
                '--eval-batch-size', '256', '--wandb-mode', 'offline']
        options += ['--device', 'cpu' if stage == 'report' else 'cuda:0']
        if stage == 'lib':
            options += ['--lib-batch-size', '2']
        module = 'code.V4.r111_reporting' if reporting else 'code.V4.r111_experiment'
        prefix = ([str(Path(sys.executable).with_name('torchrun')), '--standalone', '--nnodes=1',
                   '--nproc-per-node=2', '--max-restarts=0'] if stage == 'train'
                  else [sys.executable, '-B', '-u'])
        result.append(dict(name=stage, argv=[*prefix, '-m', module, stage, *options],
            cuda_visible_devices=','.join(UUIDS) if stage == 'train' else '' if stage == 'report' else UUIDS[0]))
    return result


def check_cli(root):
    hashes = source_hashes(root)  # Missing current files must fail before importing their parsers.
    program = ('import json,sys; from code.V4 import r111_experiment as e, r111_reporting as r; '
        'assert e.ARMS == ("A_family_conditional",) and e.MAX_UPDATES == 57600 and e.BATCH_SIZE == 1024; '
        'rows=json.loads(sys.argv[1]); '
        '[(r if row["name"] in ("lib","report") else e).parser().parse_args('
        'row["argv"][row["argv"].index("-m")+2:]) for row in rows]; '
        'print("five CPU-only stage parsers passed")')
    process = subprocess.run([sys.executable, '-B', '-c', program, json.dumps(commands(root))],
        cwd=REPO, env=environment(), capture_output=True, text=True, timeout=90)
    if process.returncode:
        raise RuntimeError('CPU CLI parse failed: ' + (process.stderr or process.stdout)[-3000:])
    if hashes != source_hashes(root):
        raise RuntimeError('Current source files changed during CPU parsing')
    return dict(cpu_only=True, cli=process.stdout.strip(), source_sha256=hashes, mutex_inodes=mutex_inodes())


def ready(root):
    record = read(root / 'launch_ready.json')
    if (any(record.get(key) is not True for key in
            ('main_thread_ready', 'cpu_tests_passed', 'sol_review_passed'))
            or not isinstance(record.get('evidence'), str) or not record['evidence'].strip()
            or not isinstance(record.get('confirmed_utc'), str) or not record['confirmed_utc'].strip()
            or record.get('source_sha256') != source_hashes(root)
            or record['source_sha256'].get(str(Path(__file__).resolve())) != LOADED_LAUNCH_SHA256
            or record.get('mutex_inodes') != mutex_inodes()):
        raise RuntimeError('Require explicit main-ready evidence, CPU tests, Sol review and unchanged six-file/inode freeze')
    return record


def predecessor_gate():
    complete = True
    rows = []
    for arm, gpu in zip(('A_tied_ce', 'B_private_ce'), UUIDS):
        output = PRIOR / 'preparation/dispatch' / ('R110_' + arm)
        arm_complete = True
        for filename, names in (('worker.json', ('train', 'TEST', 'timing', 'LIB')),
                                ('controller.json', ('allocation_GPU_pipeline', 'PDF')),
                                ('queue_receipt.json', None), ('status.json', None)):
            path = output / filename
            record = read(path) if path.exists() else {}
            stages = record.get('stages', [])
            if (record.get('state') in ('failed', 'error', 'aborted', 'cancelled')
                    or record.get('error') or record.get('exit_status') not in (None, 0)
                    or any(s.get('state') == 'failed' or s.get('exit_status') not in (None, 0) for s in stages)):
                raise RuntimeError(f'Original R110 terminal failure: {path}; never auto-heal/restart')
            if names is not None:
                done = (record.get('state') == 'complete' and record.get('exit_status') == 0
                    and [s.get('name') for s in stages] == list(names)
                    and all(s.get('state') == 'complete' and s.get('exit_status') == 0 for s in stages))
                if record.get('state') == 'complete' and not done:
                    raise RuntimeError(f'Original R110 incomplete rc0 stage receipt: {path}')
                if done and filename == 'worker.json' and (
                        record.get('container_id') != CONTAINER or record.get('gpu_uuid') != gpu):
                    raise RuntimeError('R110 original container/UUID receipt mismatch')
                arm_complete &= done
        if (PRIOR / arm / 'failure.json').exists():
            raise RuntimeError(f'Original R110 {arm} failure.json exists; no launch')
        rows.append(dict(arm=arm, complete=arm_complete))
        complete &= arm_complete
    return dict(complete=complete, arms=rows)


class StageFailure(RuntimeError):
    def __init__(self, stage, returncode):
        super().__init__(f'{stage} exit={returncode}; no retry or later stage')
        self.returncode = returncode


def record():
    return dict(state='starting', started_utc=utc(), pid=os.getpid(), host=socket.gethostname(),
        stages=[], automatic_retry=False, implicit_resume=False)


def failed(receipt, error):
    receipt.update(state='failed', exit_status=getattr(error, 'returncode', 1),
        error=str(error), finished_utc=utc())


def run_stage(stage, receipt, path, env, pass_fds=()):
    current = dict(stage, state='starting', started_utc=utc(), pid=None,
        log=str(path.parent / (stage['name'] + '.log')))
    receipt['stages'].append(current)
    receipt.update(state='running', phase=stage['name'])
    save(path, receipt)
    child = None
    try:
        with Path(current['log']).open('x') as stream:
            stream.write(shlex.join(stage['argv']) + '\n')
            stream.flush()
            child = subprocess.Popen(stage['argv'], cwd=REPO, env=env,
                stdout=stream, stderr=subprocess.STDOUT, pass_fds=pass_fds)
            current.update(state='running', pid=child.pid)
            save(path, receipt)
            rc = child.wait()
        current.update(state='complete' if rc == 0 else 'failed', exit_status=rc, finished_utc=utc())
        save(path, receipt)
        if rc:
            raise StageFailure(stage['name'], rc)
    except BaseException as error:
        if child is not None:
            # Keep both reservations until our child exits, even if saving its PID failed.
            handlers = {number: signal.signal(number, signal.SIG_IGN)
                        for number in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT)}
            try:
                current['child_exit_status'] = child.wait()
            finally:
                for number, handler in handlers.items():
                    signal.signal(number, handler)
        if current['state'] != 'failed':
            failed(current, error)
            save(path, receipt)
        raise


def self_command(stage, root, token):
    return [sys.executable, '-B', '-u', '-m', 'code.V4.r111_launch', stage,
            '--root', str(root), '--token', token]


def srun_command(root, token):
    shell = '\n'.join(('set -euo pipefail', f'source {CONDA_SH}', 'conda activate easynco',
        f'cd {shlex.quote(str(REPO))}', 'export PYTHONDONTWRITEBYTECODE=1',
        'exec ' + shlex.join(self_command('entry', root, token))))
    return ['srun', '--jobid=1475', '--overlap', '--exact', '--gres=none', '--immediate=5',
        '-N1', '-n1', '--nodelist=gpu05', '--cpus-per-task=8', '--mem=18G', 'bash', '-lc', shell]


def authorized(root, token):
    evidence = ready(root)
    claim = read(dispatch(root) / 'claim.json')
    if (not token or claim['token'] != token or os.environ.get(TOKEN_ENV) != token
            or claim['source_sha256'] != evidence['source_sha256']
            or claim['mutex_inodes'] != evidence['mutex_inodes']
            or not predecessor_gate()['complete']):
        raise RuntimeError('Internal dispatch requires the original one-shot claim, token and predecessor rc0')
    return claim


def entry(root, token):
    authorized(root, token)
    if socket.gethostname().split('.')[0] != 'gpu05' or os.environ.get('SLURM_JOB_ID') != '1475':
        raise RuntimeError('Entry must run in the existing gpu05 allocation1475 CPU-only srun step')
    env = environment(','.join(UUIDS))
    env.update(SLURM_JOB_ID='1475', **{TOKEN_ENV: token})
    argv = ['sudo', '-n', '/usr/bin/ai_docker', 'exec', '-u', str(os.getuid())]
    for key in (*('CUDA_VISIBLE_DEVICES', 'PYTHONPATH', 'PYTHONDONTWRITEBYTECODE', 'PYTHONUNBUFFERED',
                 'MPLBACKEND', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                 'NUMEXPR_NUM_THREADS', 'PYTORCH_CUDA_ALLOC_CONF', 'WANDB_PROJECT', 'WANDB_ENTITY',
                 'WANDB_MODE', 'SLURM_JOB_ID'), TOKEN_ENV):
        argv.extend(('-e', key + '=' + env[key]))
    argv += [CONTAINER, 'bash', str(WRAPPER), 'worker', '--root', str(root), '--token', token]
    receipt, path = record(), dispatch(root) / 'entry.json'
    save(path, receipt, exclusive=True)
    try:
        run_stage(dict(name='container_exec', argv=argv), receipt, path, env)
        receipt.update(state='complete', exit_status=0, finished_utc=utc())
    except BaseException as error:
        failed(receipt, error)
        raise
    finally:
        save(path, receipt)


def same_cache_record(actual, expected, rows):
    # Distinct node-local R84 creation receipts can describe identical inputs.
    administrative = {'created_utc', 'completed_utc', 'seconds', 'test_lock_root'}
    semantic = lambda value: {key: item for key, item in value.items() if key not in administrative}
    return (semantic(actual) == semantic(expected)
        and actual.get('complete') is True and actual.get('rows') == rows)


def worker_checks():
    cgroup = Path('/proc/self/cgroup').read_text()
    if (CONTAINER not in cgroup or socket.gethostname() != CONTAINER[:12]
            or os.environ.get('SLURM_JOB_ID') != '1475' or Path(sys.prefix).name != 'easynco'
            or os.environ.get('CUDA_VISIBLE_DEVICES') != ','.join(UUIDS)
            or not set(CPUS) <= os.sched_getaffinity(0)):
        raise RuntimeError('Require original container cgroup, exact two UUIDs, easynco and authorized eight CPUs')
    os.sched_setaffinity(0, CPUS)
    query = ['nvidia-smi', '--query-gpu=uuid,name', '--format=csv,noheader']
    probe = subprocess.run(query, capture_output=True, text=True, check=True, timeout=25)
    hardware = dict(tuple(part.strip() for part in line.split(',', 1))
                    for line in probe.stdout.strip().splitlines())
    if any(gpu not in hardware or '4090' not in hardware[gpu] for gpu in UUIDS):
        raise RuntimeError('Original two RTX4090 UUIDs not visible in the authorized container')
    manifest = read(PREPARED / 'source_cache_manifest.json')
    if manifest['data_root'] != str(DATA) or manifest['cache_root'] != str(CACHE):
        raise RuntimeError('Existing R84 manifest paths differ')
    count = 0
    for split, rows in (('train', 81000), ('val', 4500), ('test', 4500)):
        expected = manifest['splits'][split]
        if len(expected) != 18:
            raise RuntimeError('Require frozen full18 warm cache metadata')
        for problem, metadata in expected.items():
            path = CACHE / split / problem / 'metadata.json'
            actual = read(path)
            if not same_cache_record(actual, metadata, rows):
                raise RuntimeError(f'Warm cache metadata mismatch: {path}; no generation/copy/repair')
            count += 1
    return dict(container_id=CONTAINER, cgroup=cgroup, cpus=list(CPUS), gpu_uuids=list(UUIDS),
        uuid_query_argv=query, cache_root=str(CACHE), cache_metadata_paths_checked=count,
        cache_copied=False, cache_generated=False, wandb_mode='offline')


def fresh(root):
    run = root / ARM
    if run.exists() and any(run.iterdir()):
        raise RuntimeError('R111 run directory is not empty; no resume/restart or overwrite')


def worker(root, token):
    authorized(root, token)
    receipt, path = record(), dispatch(root) / 'worker.json'
    save(path, receipt, exclusive=True)
    try:
        fresh(root)
        receipt['checks'] = worker_checks()
        for stage in commands(root)[:-1]:
            ready(root)
            run_stage(stage, receipt, path, environment(stage['cuda_visible_devices']))
        receipt.update(state='complete', exit_status=0, finished_utc=utc())
    except BaseException as error:
        failed(receipt, error)
        raise
    finally:
        save(path, receipt)


def queue(root, interval=60):
    if interval < 1:
        raise ValueError('Queue interval must be positive')
    output = dispatch(root)
    if any((output / name).exists() for name in ('claim.json', 'entry.json', 'worker.json')):
        raise RuntimeError('Dispatch already attempted; no automatic restart')
    receipt, path = record(), output / 'queue.json'
    save(path, receipt, exclusive=True)
    try:
        while True:
            prior = predecessor_gate()
            receipt['state'] = 'waiting_predecessor' if not prior['complete'] else 'waiting_main_ready'
            if (root / 'launch_ready.json').exists():
                evidence = ready(root)
                if prior['complete']:
                    check_cli(root)
                    try:
                        with gpu_mutexes(evidence['mutex_inodes']) as fds:
                            evidence = ready(root)
                            if not predecessor_gate()['complete']:
                                raise RuntimeError('Original R110 completion changed before claim')
                            fresh(root)
                            token = uuid.uuid4().hex
                            claim = dict(token=token, claimed_utc=utc(), pid=os.getpid(),
                                gpu_uuids=list(UUIDS), cpus=list(CPUS), node='gpu05', job=1475,
                                container_id=CONTAINER, source_sha256=evidence['source_sha256'],
                                mutex_inodes=evidence['mutex_inodes'], main_ready=evidence,
                                stages=commands(root), automatic_retry=False)
                            save(output / 'claim.json', claim, exclusive=True)
                            env = environment()
                            env[TOKEN_ENV] = token
                            run_stage(dict(name='allocation_pipeline', argv=srun_command(root, token)),
                                receipt, path, env, pass_fds=tuple(fds))
                            done = read(output / 'worker.json')
                            if (done.get('state') != 'complete' or done.get('exit_status') != 0
                                    or [s.get('name') for s in done.get('stages', [])] != list(STAGES[:-1])
                                    or any(s.get('exit_status') != 0 for s in done['stages'])):
                                raise RuntimeError('Require worker rc0 through LIB before CPU report')
                            ready(root)
                            run_stage(commands(root)[-1], receipt, path, environment())
                            receipt.update(state='complete', exit_status=0, finished_utc=utc())
                            save(path, receipt)
                            return
                    except BlockingIOError:
                        if (output / 'claim.json').exists():
                            raise
                        receipt['state'] = 'waiting_original_mutex_release'
            save(path, receipt)
            time.sleep(interval)
    except BaseException as error:
        failed(receipt, error)
        save(path, receipt)
        raise


def parser():
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument('command', choices=('plan', 'check-cli', 'predecessor', 'queue', 'entry', 'worker'))
    value.add_argument('--root', type=Path, default=ROOT)
    value.add_argument('--interval', type=int, default=60)
    value.add_argument('--token', help=argparse.SUPPRESS)
    return value


def main(argv=None):
    args = parser().parse_args(argv)
    if args.root.resolve() != ROOT:
        raise ValueError('This one-shot launcher owns only the default R111 root')
    if args.command == 'plan':
        result = dict(root=str(ROOT), arm=ARM, stages=commands(ROOT), srun=srun_command(ROOT, '<ONE_SHOT_TOKEN>'),
            cpus=list(CPUS), gpu_uuids=list(UUIDS), source_files=list(map(str, source_files(ROOT))),
            missing_sources=[str(p) for p in source_files(ROOT) if not p.is_file()],
            mutex_inodes=mutex_inodes(), automatic_retry=False, implicit_resume=False, launches_work=False)
    elif args.command == 'check-cli':
        result = check_cli(ROOT)
    elif args.command == 'predecessor':
        result = predecessor_gate()
    else:
        def interrupted(signum, frame):
            raise RuntimeError(f'Interrupted by signal {signum}; no automatic restart')
        for number in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT):
            signal.signal(number, interrupted)
        if args.command == 'queue':
            queue(ROOT, args.interval)
        else:
            (entry if args.command == 'entry' else worker)(ROOT, args.token)
        return
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()

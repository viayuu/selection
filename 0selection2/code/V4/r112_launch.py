"""R112: two independent single-GPU pipelines in allocation1475, never DDP.

plan/check-cli do not dispatch. Witness is a main-supplied command transport,
not a formal launch; it holds original GPU mutexes but never creates receipts.
"""

import argparse
from contextlib import ExitStack, contextmanager
import csv
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
ROOT = REPO / 'code/V4/runs/R112_region_interfaces'
WRAPPER = REPO / 'code/V4/run_v4_r112.sh'
ARMS = ('A_collapsed', 'B_interfaces')
UUIDS = ('GPU-71479e7e-f8bd-e430-66ed-f8db9f0d47be',
         'GPU-597c682a-4b7f-7609-80a6-8ced791f5c5d')
GPU_BY_ARM = dict(zip(ARMS, UUIDS))
CPUS_BY_ARM = dict(zip(ARMS, ((0, 1, 2, 3), (4, 5, 72, 73))))
CPUS = tuple(cpu for arm in ARMS for cpu in CPUS_BY_ARM[arm])
CONTAINER = '76045731f6b453443fe18a9bce671599bdbf17c2713b8edcdd090925eea14bf6'
CONDA_SH = '/public/home/shiys/miniconda3/etc/profile.d/conda.sh'
COMPAT = '/public/home/shiys/easynco_v3_bridge/exports/train_100k_20261002/python_compat'
PREPARED = REPO / 'code/V4/runs/R84_new90k_attention/prepared'
CACHE = Path('/tmp/shiys_R84_new90k_cache')
DATA = Path('/public/home/shiys/selection_data_new90k_B_filtered_single_reld_mt_seed2_20261018')
TOKEN_ENV = 'R112_DISPATCH_TOKEN'
STAGES = ('train', 'test', 'timing', 'lib', 'report')
OFFLINE_REASON = 'Reuse the existing allocation offline setting; no credentials or online probe.'
LOADED_LAUNCH_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def utc():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value, exclusive=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(value, indent=2, allow_nan=False) + '\n'
    if exclusive:
        with path.open('x') as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
    else:
        temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
        temporary.write_text(data)
        temporary.replace(path)


def dispatch(root):
    return root / 'preparation/dispatch'


def source_files(root):
    return [REPO / 'code/V4' / name for name in
            ('r112_launch.py', 'r112_model.py', 'r112_experiment.py', 'r112_reporting.py')
            ] + [WRAPPER, root / 'protocol_lock.json']


def source_hashes(root):
    return {str(path): digest(path) for path in source_files(root)}


def mutex_paths():
    from . import r81_queue
    paths = [r81_queue.ROOT / f'queue_gpu_{gpu}.json' for gpu in UUIDS]
    if not all(path.is_file() and not path.is_symlink() for path in paths):
        raise RuntimeError('Require BOTH existing original r81_queue.ROOT UUID mutexes')
    return paths


def mutex_inodes():
    return {str(path): path.stat().st_ino for path in mutex_paths()}


@contextmanager
def gpu_mutexes(expected):
    # The container worker owns both locks across all children, not the Docker client.
    with ExitStack() as stack:
        for path in mutex_paths():
            fd = os.open(path, os.O_RDWR | os.O_NOFOLLOW)
            stack.enter_context(os.fdopen(fd, 'r+'))
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            opened, current = os.fstat(fd), path.stat()
            if (opened.st_ino != expected.get(str(path))
                    or (current.st_dev, current.st_ino) != (opened.st_dev, opened.st_ino)):
                raise RuntimeError('Original GPU mutex inode changed; never replace it')
        yield


def environment(cuda=''):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=cuda,
        PYTHONPATH=COMPAT + ':' + str(REPO), PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1',
        MPLBACKEND='Agg', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4',
        NUMEXPR_NUM_THREADS='4', PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',
        WANDB_PROJECT='selector', WANDB_ENTITY='yjkds-southern-university-of-science-technology',
        WANDB_MODE='offline')
    for key in list(env):
        if (key in ('RANK', 'WORLD_SIZE', 'LOCAL_RANK', 'LOCAL_WORLD_SIZE', 'GROUP_RANK',
                    'ROLE_RANK', 'ROLE_WORLD_SIZE', 'MASTER_ADDR', 'MASTER_PORT',
                    'SLURM_PROCID', 'SLURM_LOCALID', 'SLURM_NTASKS')
                or key.startswith(('TORCHELASTIC_', 'PMI_', 'PMIX_'))):
            env.pop(key, None)
    return env


def commands(root, arm):
    if arm not in ARMS:
        raise ValueError('R112 owns only A_collapsed and B_interfaces')
    result = []
    for stage in STAGES:
        reporting = stage in ('lib', 'report')
        options = ['--root', str(root), '--arm', arm]
        if not reporting:
            options += ['--prepared-root', str(PREPARED), '--data-root', str(DATA),
                '--cache-root', str(CACHE), '--protocol-lock', str(root / 'protocol_lock.json'),
                '--gpu-type', '4090', '--wandb-mode', 'offline']
        options += ['--device', 'cpu' if stage == 'report' else 'cuda:0']
        module = 'code.V4.r112_reporting' if reporting else 'code.V4.r112_experiment'
        result.append(dict(name=stage,
            argv=[sys.executable, '-B', '-u', '-m', module, stage, *options],
            cuda_visible_devices='' if stage == 'report' else GPU_BY_ARM[arm]))
    return result


def check_cli(root):
    frozen = source_hashes(root)
    if frozen[str(Path(__file__).resolve())] != LOADED_LAUNCH_SHA256:
        raise RuntimeError('Launcher changed after import; no implicit refreeze/restart')
    rows = [row for arm in ARMS for row in commands(root, arm)]
    program = '''import json,sys
from code.V4 import r112_experiment as e, r112_reporting as r
assert tuple(e.ARMS) == ("A_collapsed", "B_interfaces")
def parsed_only(args):
    assert args.root == __import__("pathlib").Path(sys.argv[2])
    assert args.arm in e.ARMS
    assert args.device == ("cpu" if args.command == "report" else "cuda:0")
    return args
r.lib_stage = r.report = parsed_only
for row in json.loads(sys.argv[1]):
    args = row["argv"][row["argv"].index("-m") + 2:]
    if row["name"] in ("lib", "report"):
        r.main(args)
    else:
        parsed = parsed_only(e.parser().parse_args(args))
        e.protocol_lock(parsed)
print("ten CPU-only stage parsers and current R112 protocol passed")
'''
    process = subprocess.run([sys.executable, '-B', '-c', program, json.dumps(rows), str(root)],
        cwd=REPO, env=environment(), capture_output=True, text=True, timeout=90)
    if process.returncode:
        raise RuntimeError('CPU CLI/protocol check failed: ' + (process.stderr or process.stdout)[-3000:])
    if frozen != source_hashes(root):
        raise RuntimeError('Current R112 source/config changed during CPU CLI checking')
    return dict(cpu_only=True, cli=process.stdout.strip(), source_sha256=frozen,
        protocol_sha256=frozen[str(root / 'protocol_lock.json')])


def self_command(command, root, action='launch', gpu=None, payload=()):
    argv = [sys.executable, '-B', '-u', '-m', 'code.V4.r112_launch', command,
            '--root', str(root), '--action', action]
    if gpu:
        argv += ['--gpu', gpu]
    if payload:
        argv += ['--', *payload]
    return argv


def srun_command(root, action='launch', gpu=None, payload=()):
    shell = '\n'.join(('set -euo pipefail', f'source {CONDA_SH}', 'conda activate easynco',
        f'cd {shlex.quote(str(REPO))}', 'export PYTHONDONTWRITEBYTECODE=1',
        'exec ' + shlex.join(self_command('entry', root, action, gpu, payload))))
    return ['srun', '--jobid=1475', '--overlap', '--exact', '--gres=none', '--immediate=5',
        '-N1', '-n1', '--nodelist=gpu05', '--cpus-per-task=8', '--mem=18G', 'bash', '-lc', shell]


def require_controller():
    if socket.gethostname().split('.')[0] != 'admin01' or Path(sys.prefix).name != 'easynco':
        raise RuntimeError('Controller must run on admin01 after conda activate easynco; no SSH bypass')


def require_entry():
    if (socket.gethostname().split('.')[0] != 'gpu05'
            or os.environ.get('SLURM_JOB_ID') != '1475'
            or not os.environ.get('SLURM_STEP_ID', '').isdigit()):
        raise RuntimeError('Entry requires the existing gpu05 allocation1475 srun step')


def authorized(root):
    claim = read(dispatch(root) / 'claim.json')
    token = os.environ.get(TOKEN_ENV)
    if (not token or token != claim.get('token') or claim.get('root') != str(root)
            or claim.get('source_sha256') != source_hashes(root)
            or claim['source_sha256'].get(str(Path(__file__).resolve())) != LOADED_LAUNCH_SHA256
            or claim.get('mutex_inodes') != mutex_inodes()):
        raise RuntimeError('Require original R112 one-shot token and unchanged source/config/inodes')
    return claim


class StageFailure(RuntimeError):
    def __init__(self, stage, returncode):
        super().__init__(f'{stage} exit={returncode}; no automatic retry/resume')
        self.returncode = returncode


def record(**fields):
    return dict(state='starting', started_utc=utc(), pid=os.getpid(), host=socket.gethostname(),
        stages=[], automatic_retry=False, implicit_resume=False, **fields)


def failed(receipt, error):
    receipt.update(state='failed', exit_status=getattr(error, 'returncode', 1),
        error=str(error), finished_utc=utc())


def safe_argv(argv):
    return [TOKEN_ENV + '=<REDACTED>' if item.startswith(TOKEN_ENV + '=') else item for item in argv]


def wait_owned(child):
    # Do not release reservations while our already-running child still owns CUDA work.
    handlers = {number: signal.signal(number, signal.SIG_IGN)
                for number in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT)}
    try:
        return child.wait()
    finally:
        for number, handler in handlers.items():
            signal.signal(number, handler)


def run_stage(stage, receipt, path, env):
    current = dict(stage, argv=safe_argv(stage['argv']), state='starting', started_utc=utc(),
        pid=None, log=str(path.parent / (stage['name'] + '.log')))
    receipt['stages'].append(current)
    receipt.update(state='running', phase=stage['name'])
    save(path, receipt)
    child = None
    try:
        with Path(current['log']).open('x') as stream:
            stream.write(shlex.join(current['argv']) + '\n')
            stream.flush()
            child = subprocess.Popen(stage['argv'], cwd=REPO, env=env,
                stdout=stream, stderr=subprocess.STDOUT)
            current.update(state='running', pid=child.pid)
            save(path, receipt)
            rc = child.wait()
        current.update(state='complete' if rc == 0 else 'failed', exit_status=rc, finished_utc=utc())
        save(path, receipt)
        if rc:
            raise StageFailure(stage['name'], rc)
    except BaseException as error:
        if child is not None:
            current['child_exit_status'] = wait_owned(child)
        if current['state'] != 'failed':
            failed(current, error)
        save(path, receipt)
        raise


def container_command(root, action, gpu=None, payload=()):
    env = environment(gpu if action == 'witness' else ','.join(UUIDS))
    for key in ('SLURM_JOB_ID', 'SLURM_STEP_ID'):
        env[key] = os.environ[key]
    argv = ['sudo', '-n', '/usr/bin/ai_docker', 'exec', '-u', '2022']
    keys = ['CUDA_VISIBLE_DEVICES', 'PYTHONPATH', 'PYTHONDONTWRITEBYTECODE', 'PYTHONUNBUFFERED',
        'MPLBACKEND', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
        'NUMEXPR_NUM_THREADS', 'PYTORCH_CUDA_ALLOC_CONF', 'WANDB_PROJECT', 'WANDB_ENTITY',
        'WANDB_MODE', 'SLURM_JOB_ID', 'SLURM_STEP_ID']
    if action == 'launch':
        keys.append(TOKEN_ENV)
    for key in keys:
        argv += ['-e', key + '=' + env[key]]
    argv += [CONTAINER, 'bash', str(WRAPPER), 'worker', '--root', str(root), '--action', action]
    if gpu:
        argv += ['--gpu', gpu]
    if payload:
        argv += ['--', *payload]
    return argv, env


def entry(root, action='launch', gpu=None, payload=()):
    if action == 'witness':
        require_entry()
        argv, env = container_command(root, action, gpu, payload)
        return subprocess.run(argv, cwd=REPO, env=env, check=True).returncode
    receipt, path = record(), dispatch(root) / 'entry.json'
    save(path, receipt, exclusive=True)
    try:
        authorized(root)
        require_entry()
        argv, env = container_command(root, action)
        run_stage(dict(name='container_exec', argv=argv), receipt, path, env)
        receipt.update(state='complete', exit_status=0, finished_utc=utc())
    except BaseException as error:
        failed(receipt, error)
        raise
    finally:
        save(path, receipt)


def nvml_idle(gpus):
    def query(fields, width, compute=False):
        argv = ['nvidia-smi', '--id=' + ','.join(gpus),
            ('--query-compute-apps=' if compute else '--query-gpu=') + fields,
            '--format=csv,noheader,nounits']
        process = subprocess.run(argv, capture_output=True, text=True, check=True, timeout=25)
        rows = list(csv.reader(process.stdout.splitlines(), skipinitialspace=True))
        rows = [[item.strip() for item in row] for row in rows if row]
        if any(len(row) != width for row in rows):
            raise RuntimeError('Malformed NVML output; no CUDA probe or automatic retry')
        return rows

    devices = query('uuid,name,memory.used', 3)
    if (len(devices) != len(gpus) or {row[0] for row in devices} != set(gpus)
            or any('4090' not in row[1] for row in devices)):
        raise RuntimeError('Require exact authorized RTX4090 UUIDs')
    processes = query('gpu_uuid,pid,process_name,used_gpu_memory', 4, compute=True)
    if (any(row[0] not in gpus or not row[1].isdigit() for row in processes)
            or processes or any(int(row[2]) > 512 for row in devices)):
        raise RuntimeError('Authorized GPU busy: compute process or memory >512MiB; no retry')
    return dict(devices=devices, compute_processes=processes, api='NVML via nvidia-smi; no CUDA context')


def same_cache_record(actual, expected, rows):
    administrative = {'created_utc', 'completed_utc', 'seconds', 'test_lock_root'}
    semantic = lambda value: {key: item for key, item in value.items() if key not in administrative}
    return (semantic(actual) == semantic(expected)
        and actual.get('complete') is True and actual.get('rows') == rows)


def worker_checks(gpus=UUIDS, check_cache=True):
    cpus = tuple(cpu for arm in ARMS if GPU_BY_ARM[arm] in gpus for cpu in CPUS_BY_ARM[arm])
    cgroup = Path('/proc/self/cgroup').read_text()
    if (CONTAINER not in cgroup or socket.gethostname() != CONTAINER[:12]
            or os.getuid() != 2022 or os.environ.get('SLURM_JOB_ID') != '1475'
            or not os.environ.get('SLURM_STEP_ID', '').isdigit()
            or Path(sys.prefix).name != 'easynco'
            or os.environ.get('CUDA_VISIBLE_DEVICES') != ','.join(gpus)
            or not set(cpus) <= os.sched_getaffinity(0)):
        raise RuntimeError('Require original container/cgroup, uid2022, allocation1475, easynco and slot CPUs/UUIDs')
    os.sched_setaffinity(0, cpus)
    inventory = nvml_idle(gpus)
    checked = 0
    if check_cache:
        manifest = read(PREPARED / 'source_cache_manifest.json')
        if manifest.get('data_root') != str(DATA) or manifest.get('cache_root') != str(CACHE):
            raise RuntimeError('Existing R84 warm cache manifest paths differ')
        for split, rows in (('train', 81000), ('val', 4500), ('test', 4500)):
            expected = manifest['splits'][split]
            if len(expected) != 18:
                raise RuntimeError('Require existing full18 R84 warm cache metadata')
            for problem, metadata in expected.items():
                path = CACHE / split / problem / 'metadata.json'
                if not same_cache_record(read(path), metadata, rows):
                    raise RuntimeError(f'Warm cache metadata mismatch: {path}; no copy/rebuild/repair')
                checked += 1
    return dict(container_id=CONTAINER, cgroup=cgroup, cpus=list(cpus), gpu_uuids=list(gpus),
        nvml=inventory, cache_root=str(CACHE), cache_metadata_paths_checked=checked,
        cache_copied=False, cache_generated=False, wandb_mode='offline', wandb_reason=OFFLINE_REASON)


def fresh(root):
    for arm in ARMS:
        run = root / arm
        if run.exists() and any(run.iterdir()):
            raise RuntimeError(f'{arm} run directory is not empty; no resume/restart/overwrite')


def run_arms(root):
    receipts, active, positions = {}, {}, dict.fromkeys(ARMS, 0)

    def start(arm):
        authorized(root)
        row = commands(root, arm)[positions[arm]]
        path = dispatch(root) / (arm + '.json')
        argv = ['taskset', '--cpu-list', ','.join(map(str, CPUS_BY_ARM[arm])), *row['argv']]
        current = dict(row, argv=argv, state='starting', started_utc=utc(), pid=None,
            log=str(dispatch(root) / (arm + '_' + row['name'] + '.log')))
        receipt = receipts[arm]
        receipt['stages'].append(current)
        receipt.update(state='running', phase=row['name'])
        save(path, receipt)
        stream = Path(current['log']).open('x')
        try:
            stream.write(shlex.join(argv) + '\n')
            stream.flush()
            child = subprocess.Popen(argv, cwd=REPO, env=environment(row['cuda_visible_devices']),
                stdout=stream, stderr=subprocess.STDOUT)
        except BaseException as error:
            stream.close()
            failed(current, error)
            failed(receipt, error)
            save(path, receipt)
            raise
        active[arm] = (child, stream, current)
        current.update(state='running', pid=child.pid)
        save(path, receipt)

    try:
        for arm in ARMS:
            receipt = record(arm=arm, gpu_uuid=GPU_BY_ARM[arm], cpus=list(CPUS_BY_ARM[arm]))
            save(dispatch(root) / (arm + '.json'), receipt, exclusive=True)
            receipts[arm] = receipt
        for arm in ARMS:
            start(arm)
        while active:
            for arm in list(active):
                child, stream, current = active[arm]
                rc = child.poll()
                if rc is None:
                    continue
                stream.close()
                current.update(state='complete' if rc == 0 else 'failed', exit_status=rc, finished_utc=utc())
                receipt, path = receipts[arm], dispatch(root) / (arm + '.json')
                if rc:
                    failed(receipt, StageFailure(arm + '/' + current['name'], rc))
                save(path, receipt)
                del active[arm]
                if rc:
                    continue
                positions[arm] += 1
                if positions[arm] < len(STAGES):
                    start(arm)
                else:
                    receipt.update(state='complete', exit_status=0, finished_utc=utc())
                    save(path, receipt)
            if active:
                time.sleep(.2)
    except BaseException as error:
        for arm, (child, stream, current) in active.items():
            current['child_exit_status'] = wait_owned(child)
            stream.close()
            failed(current, error)
        for arm, receipt in receipts.items():
            if receipt['state'] not in ('complete', 'failed'):
                failed(receipt, error)
            save(dispatch(root) / (arm + '.json'), receipt)
        raise
    errors = [receipt['exit_status'] for receipt in receipts.values() if receipt['state'] != 'complete']
    if errors:
        raise StageFailure('independent arm pipeline(s)', errors[0])
    return receipts


def worker(root, action='launch', gpu=None, payload=()):
    if action == 'witness':
        with gpu_mutexes(mutex_inodes()):
            checks = worker_checks((gpu,), check_cache=False)
            print(json.dumps(dict(action='witness', formal_metadata_written=False,
                gpu_mutexes_acquired=True, checks=checks)), file=sys.stderr, flush=True)
            child = subprocess.Popen(list(payload), cwd=REPO, env=environment(gpu))
            try:
                rc = child.wait()
            except BaseException:
                wait_owned(child)
                raise
            if rc:
                raise StageFailure('witness', rc)
        return
    receipt, path = record(arms={arm: str(dispatch(root) / (arm + '.json')) for arm in ARMS}), dispatch(root) / 'worker.json'
    save(path, receipt, exclusive=True)
    try:
        claim = authorized(root)
        with gpu_mutexes(claim['mutex_inodes']):
            fresh(root)
            receipt['checks'] = worker_checks()
            receipt.update(state='running')
            save(path, receipt)
            receipt['arm_results'] = run_arms(root)
            receipt.update(state='complete', exit_status=0, finished_utc=utc())
    except BaseException as error:
        failed(receipt, error)
        raise
    finally:
        save(path, receipt)


def launch(root, confirmation):
    receipt, path = record(), dispatch(root) / 'launch.json'
    save(path, receipt, exclusive=True)
    try:
        require_controller()
        evidence = check_cli(root)
        if not confirmation or confirmation != evidence['protocol_sha256']:
            raise RuntimeError('Wait for main: --confirm-main-config must equal final protocol SHA256')
        fresh(root)
        token = uuid.uuid4().hex
        claim = dict(root=str(root), token=token, claimed_utc=utc(), pid=os.getpid(),
            job=1475, node='gpu05', container_id=CONTAINER, gpu_by_arm=GPU_BY_ARM,
            cpus_by_arm=CPUS_BY_ARM, source_sha256=evidence['source_sha256'],
            mutex_inodes=mutex_inodes(), main_confirmed_protocol_sha256=confirmation,
            stages={arm: commands(root, arm) for arm in ARMS}, automatic_retry=False)
        save(dispatch(root) / 'claim.json', claim, exclusive=True)
        env = environment()
        env[TOKEN_ENV] = token
        run_stage(dict(name='allocation_pipeline', argv=srun_command(root)), receipt, path, env)
        done = read(dispatch(root) / 'worker.json')
        if (done.get('state') != 'complete' or done.get('exit_status') != 0
                or set(done.get('arm_results', {})) != set(ARMS)
                or any(row.get('state') != 'complete' or row.get('exit_status') != 0
                    or [s.get('name') for s in row.get('stages', [])] != list(STAGES)
                    or any(s.get('state') != 'complete' or s.get('exit_status') != 0 for s in row['stages'])
                    for row in done['arm_results'].values())):
            raise RuntimeError('Require BOTH independent pipelines complete through report at rc0')
        receipt.update(state='complete', exit_status=0, finished_utc=utc())
    except BaseException as error:
        failed(receipt, error)
        raise
    finally:
        save(path, receipt)


def witness(root, gpu, payload):
    require_controller()
    return subprocess.run(srun_command(root, 'witness', gpu, payload), cwd=REPO,
        env=environment(), check=True).returncode


def plan(root):
    protocol = root / 'protocol_lock.json'
    return dict(root=str(root), launches_work=False, independent_single_gpu=True,
        gpu_by_arm=GPU_BY_ARM, cpus_by_arm=CPUS_BY_ARM, total_step_cpus=8, total_step_memory='18G',
        stages={arm: commands(root, arm) for arm in ARMS}, srun=srun_command(root),
        protocol_path=str(protocol), protocol_sha256=digest(protocol) if protocol.is_file() else None,
        missing_sources=[str(path) for path in source_files(root) if not path.is_file()],
        witness_requires_main_payload=True, witness_writes_formal_metadata=False,
        automatic_retry=False, implicit_resume=False, cache_copied=False, cache_generated=False,
        wandb_mode='offline', wandb_reason=OFFLINE_REASON)


def parser():
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument('command', choices=('plan', 'check-cli', 'launch', 'entry', 'worker', 'witness'))
    value.add_argument('--root', type=Path, default=ROOT)
    value.add_argument('--action', choices=('launch', 'witness'), default='launch')
    value.add_argument('--gpu', choices=UUIDS)
    value.add_argument('--confirm-main-config', help='Explicit main-confirmed protocol_lock.json SHA256')
    return value


def main(argv=None):
    raw = list(sys.argv[1:] if argv is None else argv)
    split = raw.index('--') if '--' in raw else len(raw)
    args, payload = parser().parse_args(raw[:split]), raw[split + 1:]
    root = args.root.expanduser().resolve()
    if root != ROOT:
        raise ValueError('R112 launcher owns only the assigned R112_region_interfaces root')
    is_witness = args.command == 'witness' or args.action == 'witness'
    if is_witness and (not args.gpu or not payload
            or args.command not in ('witness', 'entry', 'worker')):
        raise ValueError('Witness requires --gpu <authorized UUID> -- <main-supplied command>')
    if not is_witness and (payload or args.gpu):
        raise ValueError('Payload/single-GPU overrides are only allowed for witness transport')
    if args.command in ('plan', 'check-cli'):
        result = plan(root) if args.command == 'plan' else check_cli(root)
        print(json.dumps(result, indent=2, allow_nan=False))
        return

    def interrupted(signum, frame):
        raise RuntimeError(f'Interrupted by signal {signum}; no automatic restart')

    for number in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT):
        signal.signal(number, interrupted)
    if args.command == 'launch':
        launch(root, args.confirm_main_config)
    elif args.command == 'witness':
        witness(root, args.gpu, payload)
    else:
        (entry if args.command == 'entry' else worker)(root, args.action, args.gpu, payload)


if __name__ == '__main__':
    main()

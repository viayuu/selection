"""Fixed-allocation R80 launcher; no training or reporting implementation."""

import argparse
import csv
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import signal
import socket
import subprocess
import sys
from datetime import datetime, timezone


REPO = Path('/public/home/shiys/0selection2')
ROOT = REPO / 'code/V4/runs/R80_training_and_code_ablation'
WRAPPER = REPO / 'code/V4/run_v4_r80.sh'
SOURCE = REPO / 'code/V4/runs/R76_R45A_single_reld_mt/cache'
CACHE = Path('/tmp/shiys_R76_R45A_single_reld_mt_cache')
DATA = Path('/public/home/shiys/selection_data_new_only_B_filtered_single_reld_mt_seed2_20261014')
GROUPS = {
    'A_ce_only': dict(job=1465, node='gpu03', index=0, gpu_type='3090', cap=384,
        uuid='GPU-cb24f168-9f18-f0d5-6f4a-6d2406958f50', batch=1024, updates=49680),
    'B_batch512': dict(job=1475, node='gpu05', index=0, gpu_type='4090', cap=352,
        uuid='GPU-71479e7e-f8bd-e430-66ed-f8db9f0d47be', batch=512, updates=99360),
    'C_lr3e4': dict(job=1465, node='gpu03', index=1, gpu_type='3090', cap=384,
        uuid='GPU-cb626bdb-4319-2f5e-b91f-17fcaa44a83d', batch=1024, updates=49680),
    'D_code_embedding': dict(job=1475, node='gpu05', index=1, gpu_type='4090', cap=352,
        uuid='GPU-597c682a-4b7f-7609-80a6-8ced791f5c5d', batch=1024, updates=49680),
}
CONTAINERS = {
    'gpu03': ('1465_gpu03', 'b8ced5d8a8cdafdc5006f827a0d36c319c61056a65316af4ff9937453fa0635c'),
    'gpu05': ('1475_gpu05', '76045731f6b453443fe18a9bce671599bdbf17c2713b8edcdd090925eea14bf6'),
}


def utc():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    if path.parent != ROOT or not (path.name.startswith('launch') and path.suffix == '.json'):
        raise ValueError(f'Launcher JSON outside its owned scope: {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + f'_tmp_{os.getpid()}.json')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def capture(command):
    return subprocess.check_output(command, text=True, stderr=subprocess.STDOUT, timeout=30)


def source_hashes():
    paths = [WRAPPER, REPO / 'code/V4/r80_launch.py',
        REPO / 'code/V4/r80_experiment.py', REPO / 'code/V4/r80_reporting.py']
    return {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in paths if path.is_file()}


def manifest():
    return dict(experiment='R80', created_utc=utc(), root=str(ROOT),
        data_root=str(DATA), cache_root=str(CACHE), source_cache=str(SOURCE),
        seed=2, initialization='scratch', problems=18, groups=GROUPS,
        cpu_per_process=4, memory_per_process='9G', torch_threads=2,
        node_total_cpu=8, node_total_memory='18G', node_physical_gpu_count=2,
        resource_policy='existing jobs only; overlapping steps inherit allocated GPUs; one UUID per process',
        training_rule='fixed successful-update budgets; unchanged effective batch; no resume/retry',
        pipeline=['train', 'test', 'LIB_and_PDF_report'],
        pipeline_rule='next stage only after exit 0; errors persist; no automatic restart',
        selection='one VAL-selected checkpoint locked before TEST/LIB; no TEST/LIB tuning',
        wandb=dict(enabled=True, project='selector',
            entity='yjkds-southern-university-of-science-technology',
            policy='prefer online with existing credentials, including netrc; explicit offline fallback'),
        startup_check='one bounded check only: A/C/D >=18 successful updates; B >=36; all18 tasks',
        continuous_monitor=False, external_notifications=False, git_commit=False,
        source_sha256=source_hashes())


def srun_command(group, mode, arm):
    return ['srun', f'--jobid={group["job"]}', '--overlap', '--exact', '-N1', '-n1',
        f'--nodelist={group["node"]}', '--cpus-per-task=4', '--mem=9G',
        'bash', str(WRAPPER), mode, arm]


def assert_ready():
    gate = read(ROOT / 'launch_ready.json')
    if not (gate.get('main_thread_ready') is True and gate.get('runner_cpu_tests_passed') is True):
        raise RuntimeError('Main-thread ready and runner base CPU tests must both be recorded')
    if not gate.get('evidence') or not gate.get('confirmed_utc'):
        raise RuntimeError('Ready gate lacks concrete confirmation evidence')
    for name in ('r80_experiment.py', 'r80_reporting.py'):
        if not (REPO / 'code/V4' / name).is_file():
            raise RuntimeError(f'Missing agreed CLI module: {name}')


def start():
    assert_ready()
    if (ROOT / 'launch_manifest.json').exists():
        previous = read(ROOT / 'launch_manifest.json')
        if previous.get('submitted_utc'):
            raise RuntimeError('R80 already submitted; no implicit restart/resume is supported')
    for arm in GROUPS:
        if (ROOT / f'launch_{arm}_controller.json').exists() or (ROOT / f'launch_{arm}.json').exists():
            raise RuntimeError(f'{arm}: existing launch receipt; refusing a second launch')
        if subprocess.run(['tmux', 'has-session', '-t', f'r80_{arm}'],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0:
            raise RuntimeError(f'{arm}: tmux session already exists')
        for name in ('last.pt', 'best.pt', 'status.json', 'training_complete.json'):
            if (ROOT / arm / name).exists():
                raise RuntimeError(f'{arm}: {name} exists; from-scratch launch refused')
    plan = manifest()
    plan['submitted_utc'] = utc()
    plan['sessions'] = {}
    write(ROOT / 'launch_manifest.json', plan)
    for arm in GROUPS:
        session = f'r80_{arm}'
        command = shlex.join(['bash', str(WRAPPER), 'allocated', arm])
        result = subprocess.run(['tmux', 'new-session', '-d', '-s', session, command])
        plan['sessions'][arm] = dict(session=session, submission_exit_status=result.returncode,
            command=command, submitted_utc=utc())
        write(ROOT / 'launch_manifest.json', plan)
        if result.returncode:
            raise RuntimeError(f'{session}: submission failed; remaining groups not started')
        subprocess.run(['tmux', 'set-option', '-t', session, 'remain-on-exit', 'on'], check=True)
        print(f'{arm}: tmux={session}; job={GROUPS[arm]["job"]}; GPU={GROUPS[arm]["uuid"]}', flush=True)


def allocated(arm, container=False):
    assert_ready()
    group = GROUPS[arm]
    path = ROOT / f'launch_{arm}{"_container" if container else ""}_controller.json'
    receipt = dict(arm=arm, controller_pid=os.getpid(), controller_host=socket.gethostname(),
        session=f'r80_{arm}', started_utc=utc(), state='starting', group=group,
        command=srun_command(group, 'container-entry' if container else 'worker', arm))
    if container:
        receipt['command'].insert(9, '--gres=none')
        receipt['resource_correction'] = 'Original GPU-GRES step never started a worker; use existing allocation-owned container, not a new allocation'
    if path.exists():
        raise RuntimeError('Controller receipt exists; refusing duplicate startup')
    write(path, receipt)
    child = None
    try:
        child = subprocess.Popen(receipt['command'])
        receipt.update(srun_pid=child.pid, state='running')
        write(path, receipt)

        def forward(signum, _frame):
            if child.poll() is None:
                child.send_signal(signum)

        signal.signal(signal.SIGTERM, forward)
        signal.signal(signal.SIGINT, forward)
        code = child.wait()
        receipt.update(state='complete' if code == 0 else 'failed', exit_status=code,
            finished_utc=utc())
    except BaseException as error:
        receipt.update(state='failed', exit_status=1, error=f'{type(error).__name__}: {error}',
            finished_utc=utc())
        write(path, receipt)
        raise
    write(path, receipt)
    return code


def container_entry(arm):
    group = GROUPS[arm]
    if socket.gethostname().split('.')[0] != group['node'] or os.environ.get('SLURM_JOB_ID') != str(group['job']):
        raise RuntimeError('Container entry must run in its existing allocation CPU step')
    name, identity = CONTAINERS[group['node']]
    actual = capture(['sudo', '-n', '/usr/bin/ai_docker', 'inspect', '--format={{.Id}}', name]).strip()
    if actual != identity:
        raise RuntimeError('Existing allocation container ID changed')
    command = ['sudo', '-n', '/usr/bin/ai_docker', 'exec', '-u', str(os.getuid()), name,
        'bash', str(WRAPPER), 'container-worker', arm]
    probe = ['sudo', '-n', '/usr/bin/ai_docker', 'exec', '-u', str(os.getuid()), name,
        'test', '-d', str(CACHE / 'test')]
    copied = False
    if subprocess.run(probe).returncode:
        stage = ['sudo', '-n', '/usr/bin/ai_docker', 'exec', '-u', str(os.getuid()), name,
            'bash', str(WRAPPER), 'container-stage', arm]
        subprocess.run(stage, check=True)
        copied = True
    write(ROOT / f'launch_{arm}_container_entry.json', dict(arm=arm, node=group['node'],
        job=group['job'], container_name=name, container_id=identity,
        existing_host_cache=str(CACHE), cache_copied_to_container=copied,
        copied_content='existing frozen R76 cache only; no solver/embedding/preprocessing',
        command=command, started_utc=utc(), pid=os.getpid()))
    return subprocess.run(command).returncode


def container_stage(arm):
    group = GROUPS[arm]
    proof = container_proof(group)
    if shutil.disk_usage('/tmp').free < 20 * 2**30:
        raise RuntimeError('Insufficient container-local storage for the frozen cache')
    local_source = Path('/host/tmp') / CACHE.name
    source = local_source if (local_source / 'train/ATSP/metadata.json').is_file() else SOURCE
    record = dict(node=group['node'], container=proof, started_utc=utc(), source=str(source),
        target=str(CACHE), state='copying', copied_files=0, copied_bytes=0,
        operation='necessary existing metadata + arrays only; no preprocessing')
    path = ROOT / f'launch_container_cache_{group["node"]}.json'
    write(path, record)
    for split in ('train', 'val', 'test'):
        metadata_paths = sorted((source / split).glob('*/metadata.json'))
        if len(metadata_paths) != 18:
            raise RuntimeError('Frozen source cache does not contain all18 task metadata records')
        for metadata_path in metadata_paths:
            metadata = read(metadata_path)
            files = [metadata_path.parent / spec['file'] for spec in metadata['arrays'].values()]
            files.append(metadata_path)
            for source_path in files:
                target = CACHE / source_path.relative_to(source)
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.is_file() and target.stat().st_size == source_path.stat().st_size:
                    continue
                temporary = target.with_name(target.name + f'.stage_{os.getpid()}')
                shutil.copyfile(source_path, temporary)
                temporary.replace(target)
                record['copied_files'] += 1
                record['copied_bytes'] += source_path.stat().st_size
            record.update(last_completed_task=metadata['problem'], last_completed_split=split,
                updated_utc=utc())
            write(path, record)
            print(f'[R80 frozen cache copy] {group["node"]} {split}/{metadata["problem"]}: {record["copied_bytes"] / 2**30:.2f} GiB', flush=True)
    record.update(state='complete', completed_utc=utc())
    write(path, record)
    return 0


def container_proof(group):
    name, identity = CONTAINERS[group['node']]
    cgroup = Path('/proc/self/cgroup').read_text()
    if identity not in cgroup or socket.gethostname() != identity[:12]:
        raise RuntimeError('Worker is not inside the allocation-owned, identity-locked container')
    os.environ['SLURM_JOB_ID'] = str(group['job'])
    os.environ['SLURM_STEP_ID'] = 'existing_allocation_container'
    cpus = sorted(os.sched_getaffinity(0))
    os.sched_setaffinity(0, cpus[:4])
    return dict(name=name, id=identity, cgroup=cgroup, assigned_cpus=cpus[:4],
        allocated_cpu_limit=10, allocated_memory_bytes=20 * 2**30,
        transport='existing allocation CPU srun -> allocation-owned container exec')


def cache_inventory():
    count, size = 0, 0
    for split in ('train', 'val', 'test'):
        sources = sorted((SOURCE / split).glob('*/metadata.json'))
        if len(sources) != 18:
            raise RuntimeError(f'Shared cache {split}: expected 18 metadata records')
        for source in sources:
            target = CACHE / source.relative_to(SOURCE)
            metadata = read(target)
            if not metadata.get('complete') or metadata.get('cache_format') != 'r76_scenario_v1':
                raise RuntimeError(f'Incomplete/incompatible existing cache: {target}')
            if target.read_bytes() != source.read_bytes():
                raise RuntimeError(f'Cache metadata differs from frozen R76 source: {target}')
            for spec in metadata['arrays'].values():
                path = target.parent / spec['file']
                source_path = source.parent / spec['file']
                if path.stat().st_size != source_path.stat().st_size:
                    raise RuntimeError(f'Cache file size differs from R76 source: {path}')
                size += path.stat().st_size
                count += 1
    return dict(path=str(CACHE), array_count=count, array_bytes=size,
        available_bytes=shutil.disk_usage(CACHE).free,
        validation='metadata identity + all array file sizes; no TEST values read',
        copied_files=0, reused=True)


def gpu_inventory(group):
    text = capture(['nvidia-smi', '--query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu',
        '--format=csv,noheader,nounits'])
    rows = [row for row in csv.reader(text.splitlines(), skipinitialspace=True) if row[1] == group['uuid']]
    if len(rows) != 1:
        raise RuntimeError('Assigned GPU UUID not present on expected node')
    row = rows[0]
    if int(row[0]) != group['index'] or group['gpu_type'] not in row[2] or int(row[3]) > 512:
        raise RuntimeError(f'Assigned GPU no longer matches idle approved resource: {row}')
    apps = capture(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory',
        '--format=csv,noheader,nounits'])
    if any(row and row[0] == group['uuid'] for row in csv.reader(apps.splitlines(), skipinitialspace=True)):
        raise RuntimeError('Assigned GPU already has a compute process; no process was stopped')
    return dict(index=int(row[0]), uuid=row[1], name=row[2], memory_used_mib=int(row[3]),
        memory_total_mib=int(row[4]), utilization_pct=int(row[5]))


def wandb_mode():
    record = dict(enabled=True, requested='online_if_existing_credentials', checked_utc=utc())
    try:
        from wandb.sdk.lib import apikey
        credential = apikey.api_key()
        mode = 'online' if credential else 'offline'
        source = 'WANDB_API_KEY' if os.environ.get('WANDB_API_KEY') else 'existing_wandb_or_netrc'
        record.update(selected_mode=mode, credential_present=bool(credential),
            credential_source=source if credential else 'none')
        if not credential:
            record['offline_reason'] = 'No existing W&B credential found; no login prompt was started'
    except Exception as error:
        mode = 'offline'
        record.update(selected_mode=mode, offline_reason=f'Credential lookup failed: {type(error).__name__}: {error}')
    return mode, record


def runner_command(stage, arm, mode):
    group = GROUPS[arm]
    return [sys.executable, '-B', '-u', '-m', 'code.V4.r80_experiment',
        stage, '--root', str(ROOT), '--arm', arm,
        '--data-root', str(DATA), '--cache-root', str(CACHE),
        '--gpu-type', group['gpu_type'], '--large-micro-cap', str(group['cap']),
        '--device', 'cuda:0', '--eval-batch-size', '256', '--log-every', '18', '--wandb-mode', mode]


def worker(arm, container=False):
    group = GROUPS[arm]
    proof = container_proof(group) if container else None
    path = ROOT / f'launch_{arm}.json'
    state = dict(arm=arm, worker_pid=os.getpid(), host=socket.gethostname(),
        job_id=os.environ.get('SLURM_JOB_ID'), step_id=os.environ.get('SLURM_STEP_ID'),
        started_utc=utc(), state='preflight', group=group, stages=[], container=proof)
    if path.exists():
        raise RuntimeError('Worker receipt exists; refusing duplicate startup')
    lock_path = Path('/tmp') / f'shiys_R80_{group["uuid"]}.lock'
    with lock_path.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        write(path, state)
        try:
            if (not container and socket.gethostname().split('.')[0] != group['node']) or str(group['job']) != state['job_id']:
                raise RuntimeError('Worker is not inside its user-approved allocation/node')
            state['gpu_preflight'] = gpu_inventory(group)
            state['cache'] = cache_inventory()
            mode, state['wandb'] = wandb_mode()
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=group['uuid'], R80_APPROVED='1', WANDB_MODE=mode)
            state.update(state='running', cuda_visible_devices=group['uuid'], source_sha256=source_hashes())
            write(path, state)
            commands = [('train', runner_command('train', arm, mode), 'train.log'),
                ('test', runner_command('test', arm, mode), 'test.log'),
                ('LIB_and_PDF_report', [sys.executable, '-B', '-u', '-m', 'code.V4.r80_reporting',
                    '--root', str(ROOT), '--arm', arm, '--device', 'cuda:0'], 'reporting.log')]
            directory = ROOT / arm
            directory.mkdir(parents=True, exist_ok=True)
            for stage, command, log_name in commands:
                record = dict(stage=stage, command=command, log=str(directory / log_name),
                    started_utc=utc(), state='starting')
                state['stages'].append(record)
                write(path, state)
                with (directory / log_name).open('a', buffering=1) as output:
                    output.write(f'\n[R80 launcher] {utc()} {shlex.join(command)}\n')
                    process = subprocess.Popen(command, cwd=REPO, env=env, stdout=output,
                        stderr=subprocess.STDOUT)
                    record.update(pid=process.pid, state='running')
                    write(path, state)
                    code = process.wait()
                    output.write(f'\n[R80 launcher] {utc()} stage={stage} exit_status={code}\n')
                record.update(exit_status=code, state='complete' if code == 0 else 'failed', finished_utc=utc())
                write(path, state)
                if code:
                    state.update(state='failed', failed_stage=stage, exit_status=code, finished_utc=utc())
                    write(path, state)
                    return code
            state.update(state='complete', exit_status=0, finished_utc=utc())
            write(path, state)
            return 0
        except BaseException as error:
            state.update(state='failed', exit_status=1, error=f'{type(error).__name__}: {error}', finished_utc=utc())
            write(path, state)
            raise


def gpu_snapshot():
    node = socket.gethostname().split('.')[0]
    if node not in {group['node'] for group in GROUPS.values()}:
        raise RuntimeError('GPU snapshot must execute inside an existing approved GPU job')
    groups = {arm: group for arm, group in GROUPS.items() if group['node'] == node}
    if any(str(group['job']) != os.environ.get('SLURM_JOB_ID') for group in groups.values()):
        raise RuntimeError('GPU snapshot allocation mismatch')
    devices = capture(['nvidia-smi', '--query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu',
        '--format=csv,noheader,nounits'])
    processes = capture(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory',
        '--format=csv,noheader,nounits'])
    approved = {group['uuid'] for group in groups.values()}
    processes = [row for row in csv.reader(processes.splitlines(), skipinitialspace=True)
                 if len(row) >= 4 and row[0] in approved]
    namespaces = {}
    for row in processes:
        try:
            lines = Path('/proc') / row[1] / 'status'
            namespaces[row[1]] = next(line.split()[1:] for line in lines.read_text().splitlines()
                if line.startswith('NSpid:'))
        except (OSError, StopIteration):
            namespaces[row[1]] = [row[1]]
    result = dict(captured_utc=utc(), node=node, job_id=os.environ['SLURM_JOB_ID'],
        devices=[row for row in csv.reader(devices.splitlines(), skipinitialspace=True) if row[1] in approved],
        processes=processes, pid_namespaces=namespaces)
    write(ROOT / f'launch_gpu_{node}.json', result)
    print(json.dumps(result, indent=2))


def startup_check():
    result = dict(checked_utc=utc(), continuous_monitor=False, passed=True, groups={})
    expected = {path.parent.name for path in (SOURCE / 'train').glob('*/metadata.json')}
    for arm, group in GROUPS.items():
        minimum = 36 if arm == 'B_batch512' else 18
        item = dict(minimum_successful_updates=minimum, passed=False)
        try:
            launch = read(ROOT / f'launch_{arm}.json')
            if launch['state'] == 'failed':
                raise RuntimeError(f'Launch failed: {launch}')
            updates = [json.loads(line) for line in (ROOT / arm / 'updates.jsonl').read_text().splitlines() if line]
            record = next(row for row in updates if row['successful_updates'] >= minimum)
            counts = record['updates_by_problem']
            fields = ('loss', 'ce', 'cmp', 'risk', 'kl', 'true3', 'pred3', 'all_pairs', 'grad_norm')
            finite = all(math.isfinite(record[key]) for key in fields)
            coverage = (set(counts) == expected and len(expected) == 18
                and min(counts.values()) >= (2 if arm == 'B_batch512' else 1)
                and sum(counts.values()) == record['successful_updates'])
            train = next(stage for stage in launch['stages'] if stage['stage'] == 'train')
            gpu = read(ROOT / f'launch_gpu_{group["node"]}.json')
            process_matches = [row for row in gpu['processes'] if len(row) >= 4
                and row[0] == group['uuid'] and (row[1] == str(train['pid'])
                    or gpu.get('pid_namespaces', {}).get(row[1], [None])[-1] == str(train['pid']))]
            device = next(row for row in gpu['devices'] if row[1] == group['uuid'])
            correct_binding = len(process_matches) == 1 and launch['cuda_visible_devices'] == group['uuid']
            item.update(successful_updates=record['successful_updates'], loss=record['loss'],
                grad_norm=record['grad_norm'], all_loss_parts_finite=finite,
                amp_retries=record['amp_retries'], updates_by_problem=counts, full18_coverage=coverage,
                train_pid=train['pid'], worker_pid=launch['worker_pid'], gpu_uuid=group['uuid'],
                gpu_process=process_matches, gpu_snapshot_utc=gpu['captured_utc'],
                gpu_memory_used_mib=int(device[3]), gpu_utilization_pct=int(device[5]),
                binding_verified=correct_binding, wandb=launch['wandb'],
                passed=finite and coverage and correct_binding)
        except (OSError, ValueError, KeyError, StopIteration, RuntimeError) as error:
            item.update(error=f'{type(error).__name__}: {error}')
        result['groups'][arm] = item
        result['passed'] &= item['passed']
    write(ROOT / 'launch_startup_check.json', result)
    print(json.dumps(result, indent=2))
    return 0 if result['passed'] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('plan', 'start', 'allocated', 'worker', 'status', 'gpu-snapshot',
        'check-startup', 'allocated-container', 'container-entry', 'container-worker', 'container-stage',
        'preserve-failed-preflight'))
    parser.add_argument('arm', nargs='?', choices=tuple(GROUPS))
    args = parser.parse_args()
    if args.stage == 'plan':
        path = ROOT / 'launch_manifest.json'
        if path.exists() and read(path).get('submitted_utc'):
            raise RuntimeError('Refusing to replace an already submitted launch manifest')
        value = manifest()
        value['launch_state'] = 'prepared_waiting_for_cpu_tests_and_main_ready'
        value['srun_commands'] = {arm: srun_command(group, 'worker', arm) for arm, group in GROUPS.items()}
        write(path, value)
        print(json.dumps(value, indent=2))
    elif args.stage == 'start':
        start()
    elif args.stage == 'gpu-snapshot':
        gpu_snapshot()
    elif args.stage == 'check-startup':
        return startup_check()
    elif args.stage == 'status':
        for arm in GROUPS:
            path = ROOT / f'launch_{arm}.json'
            print(json.dumps(read(path) if path.is_file() else dict(arm=arm, state='not_started')))
    else:
        if not args.arm:
            parser.error('allocated and worker require an arm')
        if args.stage == 'allocated-container':
            return allocated(args.arm, container=True)
        if args.stage == 'container-entry':
            return container_entry(args.arm)
        if args.stage == 'container-worker':
            return worker(args.arm, container=True)
        if args.stage == 'container-stage':
            return container_stage(args.arm)
        if args.stage == 'preserve-failed-preflight':
            failed = read(ROOT / f'launch_{args.arm}.json')
            if failed.get('state') != 'failed' or failed.get('stages'):
                raise RuntimeError('Only failed preflight before any training stage can be preserved for this resource correction')
            for name in ('last.pt', 'best.pt', 'status.json', 'protocol.json'):
                if (ROOT / args.arm / name).exists():
                    raise RuntimeError('Runner state exists; do not repeat a started training job')
            for suffix in ('', '_container_controller', '_container_entry'):
                path = ROOT / f'launch_{args.arm}{suffix}.json'
                preserved = path.with_name(path.stem + '_failed_cache_preflight.json')
                if preserved.exists():
                    raise RuntimeError('Preserved failed attempt already exists; no implicit retry')
                if path.exists():
                    path.rename(preserved)
            return 0
        return allocated(args.arm) if args.stage == 'allocated' else worker(args.arm)
    return 0


if __name__ == '__main__':
    sys.exit(main())

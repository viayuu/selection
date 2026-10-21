"""R84 resource preparation and explicit-ready launches in existing allocations.

The runner supplies exact argv and a completed NEW cache. This module only
stages that cache, binds resources, and runs those commands in order.
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
import shutil
import socket
import subprocess
import sys
import time
import uuid

from . import r81_queue as transport


REPO = transport.REPO
ROOT = REPO / 'code/V4/runs/R84_new90k_launch'
CONDA_SH = Path('/public/home/shiys/miniconda3/etc/profile.d/conda.sh')
FONT = Path('/usr/share/fonts/google-noto-cjk/NotoSansCJK-Regular.ttc')
COMPAT = '/public/home/shiys/easynco_v3_bridge/exports/train_100k_20261002/python_compat'
SLOTS = {
    '3090_0': transport.CANDIDATES['A_ce_only'],
    '3090_1': transport.CANDIDATES['C_lr3e4'],
    '4090_0': transport.CANDIDATES['B_batch512'],
    '4090_1': transport.CANDIDATES['D_code_embedding'],
}
ARM_SLOTS = dict(A_baseline='3090_0', B_attention_direct='4090_0',
    C_focal='3090_1', D_attention_focal='4090_1')
OLD_CACHES = (transport.CACHE, transport.SOURCE)
TOKEN_ENV = 'R84_LAUNCH_TOKEN'


def utc():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def save(path, value, exclusive=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.tmp.{uuid.uuid4().hex}')
    try:
        with temporary.open('x') as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write('\n')
        if exclusive:
            os.link(temporary, path)
        else:
            temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def lock(path, blocking=False, existing=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('r' if existing else 'a+') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        yield


def activated(argv):
    return '\n'.join(('set -euo pipefail', f'source {shlex.quote(str(CONDA_SH))}',
        'conda activate easynco', f'cd {shlex.quote(str(REPO))}',
        'export PYTHONDONTWRITEBYTECODE=1', 'exec ' + shlex.join(argv)))


def self_command(stage, spec=None, slot=None, token=None):
    command = [sys.executable, '-B', '-u', '-m', 'code.V4.r84_launch', stage]
    for key, value in (('--spec', spec), ('--slot', slot), ('--token', token)):
        if value is not None:
            command += [key, str(value)]
    return command


def srun_command(stage, slot, spec=None, token=None, cpus=1, memory='256M'):
    item = SLOTS[slot]
    return ['srun', f'--jobid={item["job"]}', '--overlap', '--exact', '--gres=none',
        '--immediate=5', '-N1', '-n1', f'--nodelist={item["node"]}',
        f'--cpus-per-task={cpus}', f'--mem={memory}', 'bash', '-lc',
        activated(self_command(stage, spec, slot, token))]


def environment(slot=None, token=None):
    result = dict(os.environ, CUDA_VISIBLE_DEVICES=SLOTS[slot]['uuid'] if token else '',
        PYTHONPATH=COMPAT + ':' + str(REPO), PYTHONDONTWRITEBYTECODE='1',
        OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2',
        NUMEXPR_NUM_THREADS='2', PYTHONUNBUFFERED='1', MPLBACKEND='Agg',
        PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True', WANDB_PROJECT='selector',
        WANDB_ENTITY='yjkds-southern-university-of-science-technology')
    if slot:
        result['SLURM_JOB_ID'] = str(SLOTS[slot]['job'])
    if token:
        result[TOKEN_ENV] = token
    return result


def container_command(stage, slot, spec=None, token=None):
    identity = transport.CONTAINERS[SLOTS[slot]['node']][1]
    command = ['sudo', '-n', '/usr/bin/ai_docker', 'exec', '-u', str(os.getuid())]
    variables = environment(slot, token)
    names = ('CUDA_VISIBLE_DEVICES', 'PYTHONPATH', 'PYTHONDONTWRITEBYTECODE',
        'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
        'NUMEXPR_NUM_THREADS', 'PYTHONUNBUFFERED', 'MPLBACKEND',
        'PYTORCH_CUDA_ALLOC_CONF', 'WANDB_PROJECT', 'WANDB_ENTITY', 'SLURM_JOB_ID')
    for key in (*names, *((TOKEN_ENV,) if token else ())):
        command += ['-e', key + '=' + variables[key]]
    # ai_docker tokenizes shell -c strings; use an on-disk shell entry so conda
    # activation and argument boundaries survive the platform wrapper.
    wrapper = REPO / 'code/V4/run_v4_r84_launch.sh'
    return command + [identity, 'bash', str(wrapper), *self_command(stage, spec, slot, token)[5:]]


def assert_container(slot):
    item = SLOTS[slot]
    identity = transport.CONTAINERS[item['node']][1]
    cgroup = Path('/proc/self/cgroup').read_text()
    if (identity not in cgroup or socket.gethostname() != identity[:12]
            or os.environ.get('SLURM_JOB_ID') != str(item['job'])
            or Path(sys.prefix).name != 'easynco'):
        raise RuntimeError('Worker must use easynco in the original allocation-owned container')
    return identity


def memory_inventory(identity):
    base = Path('/sys/fs/cgroup/memory')
    options = [base / 'docker' / identity, base]
    for directory in options:
        limit_file = directory / 'memory.limit_in_bytes'
        if not limit_file.is_file() or int(limit_file.read_text()) != 20 * 2**30:
            continue
        fields = dict(line.split() for line in (directory / 'memory.stat').read_text().splitlines())
        usage = int((directory / 'memory.usage_in_bytes').read_text())
        rss = int(fields.get('total_rss', fields.get('rss', 0)))
        cache = int(fields.get('total_cache', fields.get('cache', 0)))
        return dict(limit_bytes=20 * 2**30, usage_bytes=usage, rss_bytes=rss,
            file_cache_bytes=cache, headroom_excluding_reclaimable_cache_bytes=20 * 2**30 - rss)
    raise RuntimeError('Cannot read the allocation container 20 GiB memory cgroup')


def container_resources(slot):
    identity = assert_container(slot)
    item = SLOTS[slot]
    gpus = {key: transport.gpu_inventory(value) for key, value in SLOTS.items()
        if value['node'] == item['node']}
    return dict(container_id=identity, cpus=sorted(os.sched_getaffinity(0)),
        memory=memory_inventory(identity), gpus=gpus)


def node_resources(slot):
    item = SLOTS[slot]
    if (socket.gethostname().split('.')[0] != item['node']
            or os.environ.get('SLURM_JOB_ID') != str(item['job'])
            or f'/job_{item["job"]}/' not in Path('/proc/self/cgroup').read_text()):
        raise RuntimeError('Node entry must use the authorized allocation CPU step')
    allocation = transport.allocation(item)
    container = transport.container_inspect(item)
    raw = transport.capture(container_command('container-resources', slot), timeout=60)
    raw = re.sub(r'\ADistribute Task Container\.\r?\ntask_id:\d+\r?\n', '', raw)
    nested = json.loads(raw)
    if (nested['container_id'] != container['id']
            or not set(nested['cpus']) <= set(allocation['cpus'])):
        raise RuntimeError('Container allocation identity or CPU set changed')
    gpu_pids = {int(row[1]) for gpu in nested['gpus'].values() for row in gpu['compute_processes']}
    processes = []
    found = set()
    for path in Path('/proc').glob('[0-9]*'):
        try:
            details = dict(line.split(':', 1) for line in (path / 'status').read_text().splitlines() if ':' in line)
            pids = {int(value) for value in details.get('NSpid', path.name).split()}
            matched = gpu_pids & pids
            if not matched and container['id'] not in (path / 'cgroup').read_text():
                continue
            args = [v for v in (path / 'cmdline').read_bytes().decode(errors='replace').split('\0') if v]
            if not args or 'container-resources' in args or 'node-resources' in args:
                continue
            python = Path(args[0]).name.startswith('python')
            if any(Path(value).name == 'conda' for value in args) and 'hook' in args:
                continue
            if not matched and not python:
                continue
            found.update(matched)
            processes.append(dict(pid=int(path.name), namespace_pids=sorted(pids),
                cpus=sorted(os.sched_getaffinity(int(path.name))), command=args))
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    if gpu_pids - found:
        raise RuntimeError('Cannot attribute a live GPU process to its CPU affinity')
    allowed = sorted(set(nested['cpus']) & set(container['allowed_cpus']))
    return dict(captured_utc=utc(), node=item['node'], allocation=allocation,
        container=container, memory=nested['memory'], allowed_cpus=allowed,
        processes=processes, gpus=nested['gpus'])


def resources():
    result = dict(captured_utc=utc(), nodes={}, no_GPU_context_created=True,
        training_started=False, existing_processes_stopped=False)
    for slot in ('3090_0', '4090_0'):
        result['nodes'][SLOTS[slot]['node']] = json.loads(transport.capture(
            srun_command('node-resources', slot), timeout=90))
    old = REPO / 'code/V4/runs/R83_model_only_source_embedding'
    result['R83'] = {name: read(old / name) for name in ('launch_status.json',)
        if (old / name).is_file()}
    state = old / 'model_only_seed2/status.json'
    if state.is_file():
        value = read(state)
        result['R83']['model'] = {key: value.get(key) for key in
            ('phase', 'successful_updates', 'selected_update', 'updated_utc', 'host', 'gpu')}
    save(ROOT / 'resources.json', result)
    return result


def absolute(value):
    path = Path(value)
    if not path.is_absolute():
        raise ValueError('Data, cache and output paths must be absolute')
    return path.resolve()


def load_spec(path):
    spec = read(path)
    if spec.get('schema_version') != 1 or not re.fullmatch(r'[A-Za-z0-9_-]+', spec.get('run_id', '')):
        raise ValueError('schema_version=1 and a safe unique run_id are required')
    for key in ('run_dir', 'data_root', 'prepared_root', 'cache_source', 'cache_root'):
        spec[key] = str(absolute(spec[key]))
    for key in ('cache_source', 'cache_root'):
        if any(absolute(spec[key]) == old.resolve() or old.resolve() in absolute(spec[key]).parents
                for old in OLD_CACHES):
            raise ValueError('R84 requires a NEW cache; the R76 cache cannot be relabeled')
    if absolute(spec['cache_source']) == absolute(spec['cache_root']):
        raise ValueError('Shared cache source and node-local cache destination must differ')
    if not absolute(spec['cache_root']).is_relative_to(Path('/tmp')):
        raise ValueError('Node-local cache_root must be below /tmp')
    slots = spec.get('slots', list(SLOTS))
    if not slots or len(set(slots)) != len(slots) or any(key not in SLOTS for key in slots):
        raise ValueError('Choose unique authorized slot names')
    spec['slots'] = slots
    if spec.get('cpu_count', 4) != 4 or spec.get('memory_gib', 9) != 9:
        raise ValueError('Existing resources support four CPUs and nine GiB per experiment')
    stages = spec.get('stages', [])
    if not stages or not any(stage.get('device') == 'gpu' for stage in stages):
        raise ValueError('Main must supply exact GPU-stage argv')
    names = set()
    cpu_seen = False
    for stage in stages:
        name = stage.get('name', '')
        if not re.fullmatch(r'[A-Za-z0-9_-]+', name) or name in names:
            raise ValueError('Stage names must be safe and unique')
        names.add(name)
        if stage.get('device') not in ('gpu', 'cpu'):
            raise ValueError('Stage device must be gpu or cpu')
        cpu_seen |= stage['device'] == 'cpu'
        if cpu_seen and stage['device'] == 'gpu':
            raise ValueError('GPU stages must precede admin01 CPU/PDF stages')
        args = stage.get('argv')
        if not isinstance(args, list) or not args or any(not isinstance(arg, str) or not arg for arg in args):
            raise ValueError('Supply exact argument lists, not a shell command')
        if Path(args[0]).name in ('sh', 'bash', 'zsh'):
            raise ValueError('Runner commands must be direct argv, without shell eval')
    return spec


def receipts(spec):
    return ROOT / spec['run_id']


def cache_manifest(source):
    source = Path(source)
    if not source.is_dir():
        raise ValueError('Main-supplied completed NEW cache_source does not exist')
    files = []
    for path in sorted(source.rglob('*')):
        if path.is_symlink():
            raise ValueError('Cache staging does not follow symlinks')
        if not path.is_file():
            continue
        relative = str(path.relative_to(source))
        row = dict(path=relative, bytes=path.stat().st_size)
        if path.suffix == '.json':
            row['sha256'] = sha(path)
        files.append(row)
    if not files or not any(row['path'].endswith('.npy') for row in files):
        raise ValueError('The prepared cache needs actual array files')
    raw = json.dumps(files, sort_keys=True, separators=(',', ':')).encode()
    return dict(source=str(source), files=files, bytes=sum(row['bytes'] for row in files),
        identity_sha256=hashlib.sha256(raw).hexdigest())


def commands(spec, slot):
    item = SLOTS[slot]
    values = dict(spec, device='cuda:0', gpu_uuid=item['uuid'], gpu_type=item['gpu_type'],
        micro_cap=str(item['cap']), python=sys.executable)
    result = []
    for stage in spec['stages']:
        context = dict(values, device='cpu' if stage['device'] == 'cpu' else 'cuda:0')
        result.append(dict(stage, argv=[arg.format_map(context) for arg in stage['argv']]))
    return result


def prepare(path):
    spec = load_spec(path)
    if 'MAIN_SUPPLIES' in json.dumps(spec):
        raise ValueError('Main must replace every template placeholder with final paths and exact CLI')
    output = receipts(spec)
    if any((output / name).exists() for name in ('ready.json', 'claim.json', 'controller.json')):
        raise RuntimeError('This run is already armed/claimed; no implicit replacement')
    for key in ('data_root', 'prepared_root'):
        if not Path(spec[key]).is_dir():
            raise ValueError(f'Main-supplied {key} does not exist')
    manifest = cache_manifest(spec['cache_source'])
    sources = [Path(__file__).resolve(), REPO / 'code/V4/run_v4_r84_launch.sh',
        Path(transport.__file__).resolve()]
    sources += [absolute(value) for value in spec.get('source_files', [])]
    record = dict(prepared_utc=utc(), spec=spec, cache=manifest,
        source_sha256={str(value): sha(value) for value in sources},
        commands={slot: commands(spec, slot) for slot in spec['slots']},
        training_started=False, main_thread_ready=False)
    save(output / 'plan.json', record)
    return record


def check_ready(path):
    spec = load_spec(path)
    output = receipts(spec)
    plan, ready = read(output / 'plan.json'), read(output / 'ready.json')
    if (spec != plan['spec'] or ready.get('main_thread_ready') is not True
            or ready.get('runner_tests_passed') is not True or not ready.get('evidence')):
        raise RuntimeError('Explicit main-thread ready and matching exact CLI required')
    if ready['plan_sha256'] != sha(output / 'plan.json'):
        raise RuntimeError('Launch plan changed after main ready')
    if any(sha(name) != value for name, value in plan['source_sha256'].items()):
        raise RuntimeError('Prepared launcher/runner source changed after ready')
    return spec, plan


def session_name(spec):
    return 'r84_' + spec['run_id']


def start(path, evidence, tested):
    if socket.gethostname().split('.')[0] != 'admin01' or not FONT.is_file():
        raise RuntimeError('Controller and CJK PDF stages require admin01')
    if not evidence or not tested:
        raise RuntimeError('Main must provide explicit ready evidence and runner test confirmation')
    spec = load_spec(path)
    output = receipts(spec)
    with lock(output / 'start.lock'):
        plan = read(output / 'plan.json')
        if plan['spec'] != spec:
            raise RuntimeError('Run prepare after the main has supplied the final paths and CLI')
        if Path(spec['run_dir']).exists() and any(Path(spec['run_dir']).iterdir()):
            raise RuntimeError('Model run_dir must be fresh; no implicit resume')
        save(output / 'ready.json', dict(main_thread_ready=True, evidence=evidence,
            runner_tests_passed=True, confirmed_utc=utc(), plan_sha256=sha(output / 'plan.json')), exclusive=True)
        check_ready(path)
        command = shlex.join(['bash', '-lc', activated(self_command('controller', path))])
        subprocess.run(['tmux', 'new-session', '-d', '-s', session_name(spec), command], check=True)
        subprocess.run(['tmux', 'set-option', '-t', session_name(spec), 'remain-on-exit', 'on'], check=True)
        save(output / 'tmux.json', dict(created_utc=utc(), session=session_name(spec), command=command), exclusive=True)
    return dict(session=session_name(spec), receipts=str(output), training_started=False)


def stage_cache(spec, manifest):
    destination = Path(spec['cache_root'])
    stamp = destination / '.r84_cache_identity.json'
    with lock(destination.with_name(destination.name + '.r84_stage.lock'), blocking=True):
        if stamp.is_file():
            if read(stamp).get('identity_sha256') != manifest['identity_sha256']:
                raise RuntimeError('Existing local cache belongs to another dataset')
        else:
            if destination.exists() and any(destination.iterdir()):
                raise RuntimeError('Nonempty unstamped local cache requires explicit review')
            if cache_manifest(spec['cache_source']) != manifest:
                raise RuntimeError('Prepared NEW cache source changed')
            if shutil.disk_usage(destination.parent).free < manifest['bytes'] + 2**30:
                raise RuntimeError('Insufficient node-local space for the NEW cache')
            for row in manifest['files']:
                target = destination / row['path']
                target.parent.mkdir(parents=True, exist_ok=True)
                temporary = target.with_name(target.name + '.r84_tmp')
                shutil.copyfile(Path(spec['cache_source']) / row['path'], temporary)
                temporary.replace(target)
            save(stamp, dict(source=spec['cache_source'], identity_sha256=manifest['identity_sha256'],
                staged_utc=utc(), data_root=spec['data_root'], bytes=manifest['bytes']))
        for row in manifest['files']:
            target = destination / row['path']
            if not target.is_file() or target.stat().st_size != row['bytes']:
                raise RuntimeError('Node-local cache array is missing or incomplete')
            if 'sha256' in row and sha(target) != row['sha256']:
                raise RuntimeError('Node-local cache metadata identity mismatch')
    return dict(root=str(destination), bytes=manifest['bytes'], files=len(manifest['files']),
        identity_sha256=manifest['identity_sha256'])


def run_stage(stage, spec, record, receipt, env):
    log = receipts(spec) / (stage['name'] + '.log')
    current = dict(stage, started_utc=utc(), state='running', log=str(log))
    record.setdefault('stages', []).append(current)
    record.update(state='running', phase=stage['name'])
    save(receipt, record)
    with log.open('a', buffering=1) as stream:
        stream.write(f'\n[R84] {utc()} {shlex.join(stage["argv"])}\n')
        child = subprocess.Popen(stage['argv'], cwd=REPO, env=env, stdout=stream, stderr=subprocess.STDOUT)
        current['pid'] = child.pid
        save(receipt, record)
        try:
            rc = child.wait()
        except BaseException:
            child.wait()
            raise
    current.update(state='complete' if rc == 0 else 'failed', exit_status=rc, finished_utc=utc())
    save(receipt, record)
    if rc:
        raise RuntimeError(f'{stage["name"]} exit={rc}; no later stages or automatic retry')


def entry(path, slot, token):
    spec, _ = check_ready(path)
    claim = read(receipts(spec) / 'claim.json')
    if claim['slot'] != slot or claim['token'] != token:
        raise RuntimeError('Worker claim mismatch')
    probe = node_resources(slot)
    gpu = probe['gpus'][slot]
    occupied = {cpu for row in probe['processes'] for cpu in row['cpus']}
    if gpu['compute_processes'] or gpu['memory_used_mib'] > 512 or set(claim['cpus']) & occupied:
        raise RuntimeError('Claimed GPU/CPU became busy before entry')
    save(receipts(spec) / 'entry.json', probe, exclusive=True)
    return subprocess.run(container_command('worker', slot, path, token), check=False).returncode


def worker(path, slot, token):
    spec, plan = check_ready(path)
    identity = assert_container(slot)
    output = receipts(spec)
    claim = read(output / 'claim.json')
    if (claim['token'] != token or claim['slot'] != slot or os.environ.get(TOKEN_ENV) != token
            or os.environ.get('CUDA_VISIBLE_DEVICES') != SLOTS[slot]['uuid']):
        raise RuntimeError('Worker requires its unique token and exact one-UUID binding')
    if not set(claim['cpus']) <= set(os.sched_getaffinity(0)):
        raise RuntimeError('Claimed CPUs no longer available')
    os.sched_setaffinity(0, claim['cpus'])
    record = dict(started_utc=utc(), container_id=identity, slot=slot,
        gpu_uuid=SLOTS[slot]['uuid'], cpus=claim['cpus'], state='preparing',
        automatic_retry=False, stages=[])
    save(output / 'worker.json', record, exclusive=True)
    try:
        record['cache'] = stage_cache(spec, plan['cache'])
        save(output / 'worker.json', record)
        for stage in commands(spec, slot):
            if stage['device'] == 'gpu':
                run_stage(stage, spec, record, output / 'worker.json', environment(slot, token))
        record.update(state='complete', exit_status=0, finished_utc=utc())
        save(output / 'worker.json', record)
        return 0
    except BaseException as error:
        record.update(state='failed', exit_status=1, error=str(error), finished_utc=utc())
        save(output / 'worker.json', record)
        raise


def reserved_cpus(node, exclude):
    occupied = set()
    for path in ROOT.glob('*/claim.json'):
        if path.parent == exclude:
            continue
        claim = read(path)
        controller = path.with_name('controller.json')
        state = read(controller).get('state') if controller.is_file() else None
        if claim.get('node') == node and state not in ('complete', 'failed'):
            occupied.update(claim['cpus'])
    return occupied


def pipeline(path, spec, slot, claim):
    output = receipts(spec)
    record = dict(started_utc=utc(), slot=slot, node=SLOTS[slot]['node'],
        state='claimed', stages=[], automatic_retry=False)
    save(output / 'controller.json', record, exclusive=True)
    try:
        stage = dict(name='allocation_GPU_pipeline', device='gpu',
            argv=srun_command('entry', slot, path, claim['token'], cpus=4, memory='9G'))
        run_stage(stage, spec, record, output / 'controller.json', environment())
        worker_record = read(output / 'worker.json')
        if worker_record.get('state') != 'complete' or worker_record.get('exit_status') != 0:
            raise RuntimeError('CPU report requires rc0 GPU pipeline')
        for stage in commands(spec, slot):
            if stage['device'] == 'cpu':
                run_stage(stage, spec, record, output / 'controller.json', environment())
        record.update(state='complete', exit_status=0, finished_utc=utc())
        save(output / 'controller.json', record)
        return 0
    except BaseException as error:
        record.update(state='failed', exit_status=1, error=str(error), finished_utc=utc())
        save(output / 'controller.json', record)
        raise


def controller(path):
    if socket.gethostname().split('.')[0] != 'admin01' or not FONT.is_file():
        raise RuntimeError('Controller/PDF must run on admin01 with CJK fonts')
    spec, _ = check_ready(path)
    output = receipts(spec)
    with lock(output / 'controller.lock'):
        if (output / 'claim.json').exists() or (output / 'controller.json').exists():
            raise RuntimeError('No duplicate launch or implicit resume/retry')
        while True:
            for slot in spec['slots']:
                item = SLOTS[slot]
                claimed = False
                try:
                    gpu_lock = transport.ROOT / f'queue_gpu_{item["uuid"]}.json'
                    # Reuse the old mutex inode; an unused slot may not yet have
                    # a mutex file. Creating that empty lock changes no receipt.
                    with lock(gpu_lock, existing=gpu_lock.exists()):
                        with lock(ROOT / (item['node'] + '.lock')):
                            probe = json.loads(transport.capture(srun_command('node-resources', slot), timeout=90))
                            gpu = probe['gpus'][slot]
                            occupied = {cpu for row in probe['processes'] for cpu in row['cpus']}
                            occupied |= reserved_cpus(item['node'], output)
                            cpus = sorted(set(probe['allowed_cpus']) - occupied)[:4]
                            if (gpu['compute_processes'] or gpu['memory_used_mib'] > 512
                                    or gpu['utilization_pct'] or len(cpus) != 4
                                    or probe['memory']['headroom_excluding_reclaimable_cache_bytes'] < 9 * 2**30):
                                continue
                            check_ready(path)
                            claim = dict(slot=slot, node=item['node'], gpu_uuid=item['uuid'], cpus=cpus,
                                token=uuid.uuid4().hex, claimed_utc=utc(), resources=probe)
                            save(output / 'claim.json', claim, exclusive=True)
                            claimed = True
                        return pipeline(path, spec, slot, claim)
                except BlockingIOError:
                    if claimed:
                        raise
                    continue
            save(output / 'status.json', dict(state='waiting_resources', updated_utc=utc(),
                training_started=False, tmux=session_name(spec)))
            time.sleep(30)


def template(arm=None):
    run_id = arm or 'MAIN_SUPPLIES_UNIQUE_RUN_ID'
    return dict(schema_version=1, run_id=run_id,
        run_dir='/public/home/shiys/0selection2/code/V4/runs/R84_new90k/' + run_id,
        data_root='/public/home/shiys/selection_data_new90k_B_filtered_single_reld_mt_seed2_20261018',
        prepared_root='/public/home/shiys/MAIN_SUPPLIES_PREPARED_ROOT',
        cache_source='/public/home/shiys/MAIN_SUPPLIES_COMPLETED_NEW_CACHE',
        cache_root='/tmp/shiys_R84_new90k_MAIN_SUPPLIES_CACHE_ID',
        slots=[ARM_SLOTS[arm]] if arm else list(SLOTS), cpu_count=4, memory_gib=9, source_files=[],
        protocol=dict(seed=2, logical_batch=1024, max_successful_updates=57600,
            evaluation_every_updates=1440, warmup_updates=4320,
            solver_features='MAIN_SUPPLIES_FINAL_R77_OR_R83_CHOICE'),
        stages=[dict(name='train_VALlock_TEST', device='gpu',
            argv=['{python}', '-B', '-u', '-m', 'MAIN_SUPPLIES_RUNNER', 'train',
                '--data-root', '{data_root}', '--cache-root', '{cache_root}',
                '--root', '{run_dir}', '--device', '{device}']),
            dict(name='LIB', device='gpu', argv=['{python}', '-B', '-u', '-m', 'MAIN_SUPPLIES_LIB_CLI']),
            dict(name='PDF', device='cpu', argv=['{python}', '-B', '-u', '-m', 'MAIN_SUPPLIES_REPORT_CLI'])])


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('stage', choices=('resources', 'template', 'prepare', 'start',
        'controller', 'node-resources', 'container-resources', 'entry', 'worker'))
    result.add_argument('--spec', type=Path)
    result.add_argument('--slot', choices=tuple(SLOTS))
    result.add_argument('--arm', choices=tuple(ARM_SLOTS), help='Optional template identity, never starts a job')
    result.add_argument('--token')
    result.add_argument('--main-ready-evidence')
    result.add_argument('--runner-tests-passed', action='store_true')
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    if args.stage not in ('resources', 'template', 'node-resources', 'container-resources') and not args.spec:
        raise ValueError('--spec from main is required')
    if args.stage in ('node-resources', 'container-resources', 'entry', 'worker') and not args.slot:
        raise ValueError('--slot is required')
    if args.stage == 'resources':
        value = resources()
    elif args.stage == 'template':
        value = template(args.arm)
    elif args.stage == 'prepare':
        value = prepare(args.spec)
    elif args.stage == 'start':
        value = start(args.spec, args.main_ready_evidence, args.runner_tests_passed)
    elif args.stage == 'controller':
        return controller(args.spec)
    elif args.stage == 'node-resources':
        value = node_resources(args.slot)
    elif args.stage == 'container-resources':
        value = container_resources(args.slot)
    elif args.stage == 'entry':
        return entry(args.spec, args.slot, args.token)
    else:
        return worker(args.spec, args.slot, args.token)
    print(json.dumps(value, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

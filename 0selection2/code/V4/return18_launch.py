"""Literal, one-attempt Return18 resource transport; no experiment defaults."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import time
import uuid
import xml.etree.ElementTree as ET

from .r120_launch import NODES as AUTHORIZED_NODES


REPO = Path('/public/home/shiys/0selection2')
RUNS = REPO / 'code/V4/runs'
DISPATCH = RUNS / 'Return18_post_R120/dispatch'
R120 = RUNS / 'R120_multidomain_selector'
WRAPPER = REPO / 'code/V4/run_v4_return18.sh'
MUTEX_ROOT = RUNS / 'R81_method_conditioned_auc'
CACHE = Path('/tmp/shiys_R84_new90k_cache')
SOURCE = RUNS / 'R84_new90k_attention/input_cache'
SCHEMA = 'return18-resource-v1'
KINDS = {'collect', 'cache', 'train', 'test', 'lib', 'report'}
SLOTS = {f'{node}_{lane}': dict(node=node, lane=lane, job=item['job'],
         container=item['container'], identity=item['identity'], gpu_uuid=gpu)
         for node, item in AUTHORIZED_NODES.items() for lane, gpu in enumerate(item['uuids'])}
TRANSPORT_SOURCES = ('code/V4/return18_launch.py', 'code/V4/run_v4_return18.sh',
                     'code/V4/r120_launch.py', 'code/V4/r81_queue.py')
LOADED_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
WAIT_EXIT = 75
PORTABLE_CACHE_FIELDS = frozenset(('directory', 'cache_root', 'created_utc',
                                   'completed_utc', 'seconds', 'test_lock_root'))


class Waiting(RuntimeError):
    """A proven prerequisite/resource wait before any stage attempt."""


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def read(path):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f'Duplicate JSON key: {key}')
            result[key] = value
        return result
    return json.loads(Path(path).read_text(), object_pairs_hook=pairs,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def write(path, value, *, exclusive=False):
    path = Path(path)
    if not path.resolve().is_relative_to(DISPATCH.resolve()):
        raise ValueError('Transport writes must remain in the owned dispatch directory')
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(value, indent=2, allow_nan=False) + '\n'
    if exclusive:
        with path.open('x') as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
    else:
        temporary = path.with_name('.' + path.name + f'.{os.getpid()}.tmp')
        temporary.write_text(data)
        temporary.replace(path)


def repo_path(value):
    path = Path(value)
    return (path if path.is_absolute() else REPO / path).resolve()


def owned_root(value, arm=None):
    path = repo_path(value)
    relative = path.relative_to(RUNS.resolve())
    head = relative.parts[0]
    allowed = head == 'Return18_post_R120' or bool(re.fullmatch(r'R12[1-4](?:_[A-Za-z0-9_.-]+)?', head))
    if not allowed or (arm and head != 'Return18_post_R120' and not (head == arm or head.startswith(arm + '_'))):
        raise ValueError(f'Output outside its Return18/version ownership: {value}')
    return path


def keys(value, required, optional=()):
    if not isinstance(value, dict) or set(value) - set(required) - set(optional) or set(required) - set(value):
        raise ValueError(f'Expected keys {required}, optional {optional}')


def identifier(value):
    if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,79}', value):
        raise ValueError('Identifiers must be safe nonempty path components')


def pinned(reference):
    keys(reference, ('path', 'sha256'))
    if not isinstance(reference['sha256'], str) or not re.fullmatch('[0-9a-f]{64}', reference['sha256']):
        raise ValueError('A literal lowercase SHA256 is required')
    path = repo_path(reference['path'])
    if not path.is_file() or sha256(path) != reference['sha256']:
        raise ValueError(f'Missing or changed pinned input: {path}')
    return path


def runner_source(argv):
    if not isinstance(argv, list) or not argv or any(not isinstance(x, str) or not x or '\x00' in x for x in argv):
        raise ValueError('Stage argv must be a literal nonempty string array')
    if argv[0] not in ('python', 'python3', sys.executable):
        raise ValueError('Only the active Python interpreter is supported; no shell/SSH/torchrun')
    index = 1
    while index < len(argv) and argv[index] in ('-B', '-u'):
        index += 1
    if index < len(argv) and argv[index] == '-m':
        if index + 1 >= len(argv) or not re.fullmatch(r'code\.V4\.[A-Za-z_][A-Za-z0-9_.]*', argv[index + 1]):
            raise ValueError('Expected a pinned code.V4 Python module')
        path = REPO / (argv[index + 1].replace('.', '/') + '.py')
    else:
        if index >= len(argv) or not argv[index].endswith('.py'):
            raise ValueError('Inline Python and interpreter configuration flags are forbidden')
        path = repo_path(argv[index])
    if not path.resolve().is_relative_to((REPO / 'code/V4').resolve()):
        raise ValueError('Runner source must be in code/V4')
    if path.resolve() in {REPO / name for name in TRANSPORT_SOURCES}:
        raise ValueError('Nested transports are forbidden')
    return str(path.resolve().relative_to(REPO))


def validate_plan(path, expected_sha256=None):
    path = Path(path).resolve()
    raw_hash = sha256(path)
    if expected_sha256 and raw_hash != expected_sha256:
        raise ValueError('Plan bytes changed after dispatch')
    plan = read(path)
    keys(plan, ('schema', 'name', 'main_approved', 'approved_by', 'approved_utc',
                'protocol', 'source_sha256', 'jobs'), ('cache_test_recovery',))
    if plan['schema'] != SCHEMA or plan['main_approved'] is not True:
        raise ValueError('An explicit main-approved Return18 plan is required')
    identifier(plan['name'])
    if not isinstance(plan['approved_by'], str) or not plan['approved_by'].strip():
        raise ValueError('Main approval provenance is required')
    stamp = datetime.fromisoformat(plan['approved_utc'].replace('Z', '+00:00'))
    if stamp.tzinfo is None:
        raise ValueError('Approval timestamp requires a timezone')
    protocol = pinned(plan['protocol'])
    owned_root(str(protocol))
    if not isinstance(read(protocol), dict):
        raise ValueError('The single existing protocol must be a JSON object')
    hashes = plan['source_sha256']
    if not isinstance(hashes, dict) or not set(TRANSPORT_SOURCES) <= set(hashes):
        raise ValueError('Pin all four transport/helper sources and every runner source')
    for source, digest in hashes.items():
        if Path(source).is_absolute() or str(repo_path(source).relative_to(REPO)) != source:
            raise ValueError('Source hash keys must be canonical repository-relative paths')
        pinned(dict(path=source, sha256=digest))
    if hashes[TRANSPORT_SOURCES[0]] != LOADED_SHA256:
        raise ValueError('Loaded transport differs from its source; restart explicitly, never launch stale code')
    recovery = plan.get('cache_test_recovery')
    if recovery is not None:
        artifact = read(pinned(recovery))
        keys(artifact, ('schema', 'test_sources'))
        if artifact['schema'] != 'return18-frozen-test-v1' or set(artifact['test_sources']) != set(AUTHORIZED_NODES):
            raise ValueError('Recovery artifact requires explicit frozen TEST sources for both nodes')
        if any(not isinstance(x, str) or not Path(x).is_absolute() for x in artifact['test_sources'].values()):
            raise ValueError('Recovery TEST source paths must be absolute container-visible directories')
    jobs = plan['jobs']
    if not isinstance(jobs, list) or not jobs:
        raise ValueError('A finite nonempty literal job list is required')
    seen, roots = set(), set()
    for job in jobs:
        keys(job, ('id', 'arm', 'slot', 'root', 'stages'), ('depends_on',))
        identifier(job['id'])
        if job['id'] in seen or job['arm'] not in ('R121', 'R122', 'R123', 'R124') or job['slot'] not in SLOTS:
            raise ValueError('Duplicate job, unsupported arm or unauthorized slot')
        dependencies = job.get('depends_on', [])
        if not isinstance(dependencies, list) or any(x not in seen for x in dependencies):
            raise ValueError('Dependencies must reference earlier literal jobs (acyclic order)')
        seen.add(job['id'])
        root = owned_root(job['root'], job['arm'])
        if root in roots:
            raise ValueError('Each one-attempt job needs a distinct root')
        roots.add(root)
        if not isinstance(job['stages'], list) or not job['stages']:
            raise ValueError('Every job needs a finite nonempty stage list')
        stage_ids = set()
        for stage in job['stages']:
            keys(stage, ('id', 'kind', 'argv'))
            identifier(stage['id'])
            if stage['id'] in stage_ids or stage['kind'] not in KINDS:
                raise ValueError('Invalid/duplicate typed stage')
            stage_ids.add(stage['id'])
            argv = stage['argv']
            if runner_source(argv) not in hashes:
                raise ValueError('The stage runner source must be explicitly pinned')
            root_values = []
            for index, arg in enumerate(argv):
                flag, equals, value = arg.partition('=')
                if flag == '--device':
                    value = value if equals else argv[index + 1]
                    if value not in ('cuda', 'cuda:0', 'cpu'):
                        raise ValueError('Only local CUDA device zero (or CPU) is valid after UUID binding')
                if flag == '--gpu-type':
                    value = value if equals else argv[index + 1]
                    expected_type = '3090' if SLOTS[job['slot']]['node'] == 'gpu03' else '4090'
                    if value != expected_type:
                        raise ValueError('Literal --gpu-type must match the assigned slot; no automatic cap changes')
                if flag in ('--root', '--run-root', '--output-root', '--output-dir'):
                    value = value if equals else argv[index + 1]
                    target = owned_root(value, job['arm'])
                    if not target.is_relative_to(root):
                        raise ValueError('Stage output must stay within its declared job root')
                    if flag == '--root':
                        root_values.append(target)
                if flag in ('--protocol', '--protocol-path'):
                    value = value if equals else argv[index + 1]
                    if repo_path(value) != protocol:
                        raise ValueError('All stages must use the same approved protocol')
                if flag == '--cache-root':
                    value = value if equals else argv[index + 1]
                    if repo_path(value) not in (CACHE, SOURCE.resolve()):
                        raise ValueError('Cache argv must name frozen R84 shared or container-local inputs')
                if flag in ('--resume', '--restart', '--auto-retry') or 'CUDA_VISIBLE_DEVICES=' in arg:
                    raise ValueError('No implicit resume/retry or nested CUDA environment assignment')
            if root_values != [root]:
                raise ValueError('Each stage must literally declare exactly one --root equal to job.root')
    return plan, raw_hash


def r120_ready(root=R120):
    required = ('final_test_lock.json', 'test_complete.json', 'reports/report_complete.json',
                'protocol.json', 'best.pt')
    missing = [name for name in required if not (root / name).is_file()]
    if missing:
        raise Waiting('R120 still owns all slots; missing ' + ', '.join(missing))
    lock, test, report = (read(root / name) for name in required[:3])
    if test.get('complete') is not True or report.get('complete') is not True or report.get('lib_complete') is not True:
        raise Waiting('R120 TEST/LIB/report not complete; publication is not a prerequisite')
    if report.get('complete24') is not True:
        raise Waiting('R120 full24 report is not complete')
    digest = lock.get('checkpoint_sha256')
    if (lock.get('experiment') != 'R120' or lock.get('selected_without_test') is not True
            or lock.get('per_problem_assembly') is not False
            or repo_path(lock['checkpoint']) != (root / 'best.pt').resolve()
            or digest != sha256(root / 'best.pt')):
        raise ValueError('R120 unique VAL checkpoint lock is invalid')
    config = read(root / 'protocol.json')
    state_hash = hashlib.sha256(json.dumps(config, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    if lock.get('state_hash') != state_hash or lock.get('inference_rule') != config.get('inference_rule'):
        raise ValueError('R120 protocol/inference differs from its unique lock')
    for receipt in (test, report):
        if receipt.get('checkpoint_sha256') != digest or receipt.get('selected_update') != lock.get('selected_update'):
            raise ValueError('R120 TEST/report checkpoint identities differ')
    for split in ('val', 'test', 'TSPLIB', 'CVRPLIB'):
        path = root / 'evaluation' / split / 'selected_best/metrics.json'
        if not path.is_file():
            raise Waiting(f'R120 saved {split} evaluation missing')
        identity = read(path).get('identity', {})
        if identity.get('checkpoint_sha256') != digest or identity.get('successful_updates') != lock['selected_update']:
            raise ValueError(f'R120 saved {split} uses another checkpoint')
    return dict(checkpoint_sha256=digest, selected_update=lock['selected_update'],
                report_complete=True, lib_complete=True, publication_required=False)


def job_dir(plan, job):
    return DISPATCH / plan['name'] / 'jobs' / job['id']


def find_job(plan, name):
    return next(job for job in plan['jobs'] if job['id'] == name)


def srun_command(plan_path, job, token, digest=None):
    slot = SLOTS[job['slot']]
    return ['srun', f'--jobid={slot["job"]}', '--overlap', '--exact', '--gres=none',
            '-N1', '-n1', f'--nodelist={slot["node"]}', '--cpus-per-task=4', '--mem=9G',
            'bash', str(WRAPPER), 'entry', '--plan', str(plan_path), '--job', job['id'],
            '--token', token, '--plan-sha256', digest or sha256(plan_path)]


def entry_identity(slot, hostname, environment, cgroup):
    if (hostname.split('.')[0] != slot['node'] or environment.get('SLURM_JOB_ID') != str(slot['job'])
            or not re.search(r'(?:/|^)job_' + str(slot['job']) + r'(?:/|$)', cgroup)
            or not environment.get('SLURM_STEP_ID')):
        raise ValueError('Entry must prove its original allocation job cgroup and Slurm CPU step')


def lane_cpus(slot, allocation_cpus, container_cpus):
    allowed = sorted(set(allocation_cpus) & set(container_cpus))
    if len(allowed) < 8:
        raise ValueError('Cannot prove two disjoint four-CPU lanes inside the allocation/container')
    return allowed[slot['lane'] * 4:(slot['lane'] + 1) * 4]


def mutex_path(slot):
    return MUTEX_ROOT / f'queue_gpu_{slot["gpu_uuid"]}.json'


def entry(args, plan, digest, job):
    from .r81_queue import allocation, container_inspect
    slot = SLOTS[job['slot']]
    cgroup = Path('/proc/self/cgroup').read_text()
    entry_identity(slot, socket.gethostname(), os.environ, cgroup)
    candidate = dict(node=slot['node'], job=slot['job'])
    allocated, container = allocation(candidate), container_inspect(candidate)
    if (container['name'], container['id']) != (slot['container'], slot['identity']):
        raise ValueError('Allocation-owned container identity changed')
    cpus = lane_cpus(slot, allocated['cpus'], container['allowed_cpus'])
    affinity_path = DISPATCH / ('affinity_' + slot['node'] + '.json')
    affinity_path.parent.mkdir(parents=True, exist_ok=True)
    with affinity_path.with_suffix('.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        mapping = dict(job=slot['job'], container=slot['identity'],
                       cpus=sorted(set(allocated['cpus']) & set(container['allowed_cpus']))[:8])
        if affinity_path.exists():
            if read(affinity_path) != mapping:
                raise ValueError('Previously pinned disjoint lane CPU mapping changed; do not reassign')
        else:
            write(affinity_path, mapping, exclusive=True)
    inode = mutex_path(slot).stat().st_ino
    proof = dict(utc=utc(), job=slot['job'], node=slot['node'], cgroup=cgroup,
                 step=os.environ['SLURM_STEP_ID'], allocation=allocated, container=container,
                 assigned_cpus=cpus, mutex_inode=inode, plan_sha256=digest, token=args.token)
    write(job_dir(plan, job) / 'transports' / (args.token + '.entry.json'), proof, exclusive=True)
    env = dict(RETURN18_CPUS=','.join(map(str, cpus)), RETURN18_TOKEN=args.token,
               RETURN18_PLAN_SHA256=digest, RETURN18_MUTEX_INODE=str(inode),
               SLURM_JOB_ID=str(slot['job']), SLURM_STEP_ID=os.environ['SLURM_STEP_ID'],
               CUDA_VISIBLE_DEVICES=slot['gpu_uuid'])
    argv = ['sudo', '-n', '/usr/bin/ai_docker', 'exec', '-u', str(os.getuid())]
    for key, value in env.items():
        argv += ['-e', f'{key}={value}']
    argv += [slot['container'], 'bash', str(WRAPPER), 'worker', '--plan', str(args.plan),
             '--job', job['id'], '--token', args.token]
    return subprocess.call(argv)


def worker_identity(slot, environment, hostname, cgroup, affinity):
    if slot['identity'] not in cgroup or hostname != slot['identity'][:12]:
        raise ValueError('Worker is not in the immutable allocation-owned container')
    if environment.get('SLURM_JOB_ID') != str(slot['job']) or environment.get('CUDA_VISIBLE_DEVICES') != slot['gpu_uuid']:
        raise ValueError('Worker allocation or single CUDA UUID binding differs')
    cpus = [int(x) for x in environment.get('RETURN18_CPUS', '').split(',') if x]
    if len(set(cpus)) != 4 or len(cpus) != 4 or not set(cpus) <= set(affinity):
        raise ValueError('Worker needs exactly four proven available CPUs')
    return cpus


@contextmanager
def gpu_mutex(path, inode):
    # Never create, unlink, truncate or compare st_dev across NFS mount namespaces.
    descriptor = os.open(path, os.O_RDWR | os.O_NOFOLLOW)
    try:
        if os.fstat(descriptor).st_ino != inode or Path(path).stat().st_ino != inode:
            raise ValueError('Original R81 mutex inode changed')
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise Waiting('Original assigned UUID mutex is genuinely busy') from error
        if Path(path).stat().st_ino != inode:
            raise ValueError('R81 mutex path replaced during lock acquisition')
        yield descriptor
    finally:
        os.close(descriptor)


def gpu_probe(slot):
    from .r81_queue import capture
    # One NVML snapshot, only after locking; no CUDA context and no process kills.
    raw = capture(['nvidia-smi', '--id=' + slot['gpu_uuid'], '-q', '-x'])
    devices = ET.fromstring(raw).findall('gpu')
    if len(devices) != 1:
        raise ValueError('Expected exactly one assigned GPU in the NVML snapshot')
    device = devices[0]
    gpu_uuid, name = device.findtext('uuid'), device.findtext('product_name', '')
    if gpu_uuid != slot['gpu_uuid'] or ('3090' if slot['node'] == 'gpu03' else '4090') not in name:
        raise ValueError('Assigned GPU UUID/hardware identity mismatch')
    memory = device.findtext('fb_memory_usage/used', '')
    utilization = device.findtext('utilization/gpu_util', '')
    if not re.fullmatch(r'\d+ MiB', memory) or not re.fullmatch(r'\d+ %', utilization):
        raise ValueError('Unprovable GPU memory/utilization snapshot')
    processes = device.find('processes')
    if processes is None or (processes.text or '').strip() not in ('', 'None'):
        raise ValueError('Cannot prove the GPU process inventory')
    row = dict(uuid=gpu_uuid, name=name, memory_mib=int(memory.split()[0]),
               utilization=int(utilization.split()[0]),
               processes=[dict(pid=item.findtext('pid'), name=item.findtext('process_name'))
                          for item in processes.findall('process_info')])
    if row['processes'] or row['memory_mib'] > 512 or row['utilization'] > 10:
        raise Waiting('Assigned locked GPU is genuinely busy: ' + json.dumps(row))
    return row


def portable_cache_identity(local_path, reference_path):
    """Permit provenance differences only with exact labels and sampled inputs."""
    import numpy as np
    local_path, reference_path = Path(local_path), Path(reference_path)
    local, reference = read(local_path), read(reference_path)
    semantic = lambda value: {key: item for key, item in value.items() if key not in PORTABLE_CACHE_FIELDS}
    if semantic(local) != semantic(reference):
        changed = sorted(key for key in set(local) | set(reference)
                         if key not in PORTABLE_CACHE_FIELDS and local.get(key) != reference.get(key))
        raise ValueError(f'Frozen cache semantic metadata differs: {local_path}: {changed}')
    if local.get('split') == 'test':
        for metadata in (local, reference):
            root = metadata.get('test_lock_root')
            if not root or not (Path(root) / 'final_test_lock.json').is_file():
                raise ValueError('Portable TEST cache must retain its existing historical TEST access lock')
    arrays = local.get('arrays', {})
    if not {'costs', 'winner'} <= set(arrays):
        raise ValueError('Portable cache identity requires both native costs and winner arrays')
    rows = sorted({0, 1, int(local['rows']) // 2, int(local['rows']) - 2, int(local['rows']) - 1})
    if rows[0] < 0 or rows[-1] >= int(local['rows']):
        raise ValueError('Portable cache needs at least two rows')
    proof = dict(changed_fields=sorted(key for key in set(local) | set(reference)
                    if local.get(key) != reference.get(key)), reference=str(reference_path),
                 label_sha256={}, input_sample_rows=rows, sampled_input_arrays=[],
                 input_proof='fixed-row exact comparison, not a full input-array hash')
    for name, spec in arrays.items():
        if Path(spec['file']).name != spec['file']:
            raise ValueError('Portable array filename escapes its task directory')
        left, right = local_path.parent / spec['file'], reference_path.parent / spec['file']
        if name in ('costs', 'winner'):
            digest = sha256(left)
            if digest != sha256(right):
                raise ValueError(f'Frozen native {name} full SHA256 differs: {left}')
            proof['label_sha256'][name] = digest
        else:
            a, b = np.load(left, mmap_mode='r', allow_pickle=False), np.load(right, mmap_mode='r', allow_pickle=False)
            try:
                if (a.shape != tuple(spec['shape']) or b.shape != a.shape
                        or a.dtype != np.dtype(spec['dtype']) or b.dtype != a.dtype
                        or not np.array_equal(a[rows], b[rows])):
                    raise ValueError(f'Frozen input array fixed-row comparison differs: {left}')
                proof['sampled_input_arrays'].append(name)
            finally:
                del a, b
    return proof


def cache_inventory(root):
    import numpy as np
    tasks = sorted(path.parent.name for path in (SOURCE / 'train').glob('*/metadata.json'))
    if len(tasks) != 18:
        raise ValueError('Frozen R84 source must enumerate exactly 18 routing tasks')
    missing, files, portable = [], 0, []
    for split, rows in (('train', 81000), ('val', 4500), ('test', 4500)):
        for task in tasks:
            directory = root / split / task
            path = directory / 'metadata.json'
            if not path.is_file():
                missing.append(str(path.relative_to(root)))
                continue
            metadata = read(path)
            if (metadata.get('complete') is not True or metadata.get('split') != split
                    or metadata.get('problem') != task or metadata.get('rows') != rows
                    or metadata.get('scenario_id') != 'new90k_seed2_B_filtered_MOEL_as_reld_mt_v1'):
                raise ValueError(f'Unexpected frozen cache metadata: {path}')
            original = SOURCE / split / task / 'metadata.json'
            if split == 'test' and not original.is_file():
                published = R120 / 'existing_test_cache'
                marker = published / 'complete.json'
                if marker.is_file() and read(marker).get('complete') is True:
                    original = published / 'test' / task / 'metadata.json'
            if original.is_file() and sha256(original) != sha256(path):
                portable.append(dict(split=split, problem=task, **portable_cache_identity(path, original)))
            for spec in metadata['arrays'].values():
                target = directory / spec['file']
                if Path(spec['file']).name != spec['file']:
                    raise ValueError('Cache array paths must stay in their task directory')
                if not target.is_file():
                    missing.append(str(target.relative_to(root)))
                    continue
                with target.open('rb') as stream:
                    version = np.lib.format.read_magic(stream)
                    if version == (1, 0):
                        shape, _, dtype = np.lib.format.read_array_header_1_0(stream)
                    elif version == (2, 0):
                        shape, _, dtype = np.lib.format.read_array_header_2_0(stream)
                    else:
                        raise ValueError('Unsupported frozen cache NPY header')
                    expected = stream.tell() + int(dtype.itemsize) * math.prod(shape)
                if (not shape or shape[0] != rows or list(shape) != spec['shape'] or str(dtype) != spec['dtype'] or dtype.hasobject
                        or target.stat().st_size != expected):
                    raise ValueError(f'Frozen array header/size mismatch: {target}')
                files += 1
    return dict(complete=not missing, missing=missing, arrays=files,
                splits=['train', 'val', 'test'], portable_metadata_checks=portable)


def frozen_test_source(plan, slot):
    recovery = plan.get('cache_test_recovery')
    if recovery is not None:
        artifact = read(pinned(recovery))
        return Path(artifact['test_sources'][slot['node']])
    if (SOURCE / 'test/TSP/metadata.json').is_file():
        return SOURCE / 'test'
    published = R120 / 'existing_test_cache'
    marker = published / 'complete.json'
    if not marker.is_file():
        raise Waiting('Wait for parent R120 frozen TEST publication complete.json')
    proof = read(marker)
    if (proof.get('complete') is not True or proof.get('preprocessing') is not False
            or proof.get('solver_calls') != 0
            or proof.get('source_container') not in {item['identity'][:12] for item in AUTHORIZED_NODES.values()}):
        raise ValueError('Parent TEST publication lacks frozen allocation-container provenance')
    return published / 'test'


def ensure_cache(plan, slot):
    lock_path = DISPATCH / ('cache_' + slot['node'] + '.lock')
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        inventory = cache_inventory(CACHE)
        if inventory['complete']:
            return dict(inventory, root=str(CACHE), copied=False, preprocessing=False)
        sources = {split: SOURCE / split for split in ('train', 'val')}
        if any(name.startswith('test/') for name in inventory['missing']):
            sources['test'] = frozen_test_source(plan, slot)
        for split in {Path(name).parts[0] for name in inventory['missing']}:
            source = sources[split]
            if not source.is_dir() or not (source / 'TSP/metadata.json').is_file():
                raise Waiting(f'Frozen cache recovery source not present inside container: {source}')
            destination = CACHE / split
            destination.mkdir(parents=True, exist_ok=True)
            subprocess.run(['cp', '-an', '--reflink=auto', str(source) + '/.', str(destination)], check=True)
        inventory = cache_inventory(CACHE)
        if not inventory['complete']:
            raise ValueError('Frozen copy is incomplete; no feature preprocessing or automatic repair')
        receipt = dict(inventory, root=str(CACHE), copied=True, preprocessing=False, utc=utc(),
                       sources={key: str(value) for key, value in sources.items()})
        write(DISPATCH / ('cache_' + slot['node'] + '.json'), receipt)
        return receipt


def run_stages(plan_path, digest, plan, job, descriptor, receipt):
    directory = job_dir(plan, job)
    attempt = directory / 'attempt.json'
    write(attempt, receipt, exclusive=True)
    process = None
    interrupted = []

    def forward(signum, _frame):
        interrupted.append(signum)
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signum)

    handlers = {sig: signal.signal(sig, forward) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        if any(stage['kind'] == 'train' for stage in job['stages']):
            root = str(repo_path(job['root']))
            claim = hashlib.sha256(root.encode()).hexdigest()
            write(DISPATCH / 'train_claims' / (claim + '.json'),
                  dict(root=root, job=job['id'], plan_sha256=digest, utc=utc(),
                       automatic_retry=False), exclusive=True)
        for stage in job['stages']:
            if interrupted:
                raise RuntimeError('Worker interrupted; do not start another stage')
            validate_plan(plan_path, digest)
            row = dict(stage, started_utc=utc(), state='running', log=str(directory / (stage['id'] + '.log')))
            receipt['stages'].append(row)
            write(attempt, receipt)
            with Path(row['log']).open('x') as log:
                process = subprocess.Popen(stage['argv'], cwd=REPO, env=dict(os.environ,
                    RETURN18_GPU_MUTEX_FD=str(descriptor)), stdout=log, stderr=subprocess.STDOUT,
                    start_new_session=True, pass_fds=(descriptor,))
                row['pid'] = process.pid
                write(attempt, receipt)
                code = process.wait()
                process = None
            row.update(exit_status=code, state='complete' if code == 0 else 'failed', ended_utc=utc())
            write(directory / (stage['id'] + '.json'), row, exclusive=True)
            if code or interrupted:
                code = code or 128 + interrupted[0]
                receipt.update(state='failed', exit_status=code, ended_utc=utc())
                write(attempt, receipt)
                return code
        receipt.update(state='complete', exit_status=0, ended_utc=utc())
        write(attempt, receipt)
        return 0
    except BaseException as error:
        if process is not None:
            # Keep the reservation until our child exits; never terminate unrelated work.
            process.wait()
        receipt.update(state='failed', exit_status=1, ended_utc=utc(), error=f'{type(error).__name__}: {error}')
        if receipt['stages'] and receipt['stages'][-1]['state'] == 'running':
            receipt['stages'][-1].update(state='failed', exit_status=1, ended_utc=utc(), error=str(error))
        write(attempt, receipt)
        raise
    finally:
        for sig, handler in handlers.items():
            signal.signal(sig, handler)


def worker(args, plan, digest, job):
    slot = SLOTS[job['slot']]
    directory = job_dir(plan, job)
    if (directory / 'attempt.json').exists():
        raise ValueError('Permanent one-attempt receipt already exists; no restart')
    if os.environ.get('RETURN18_TOKEN') != args.token or os.environ.get('RETURN18_PLAN_SHA256') != digest:
        raise ValueError('Worker token/plan differs from the verified allocation entry')
    proof = read(directory / 'transports' / (args.token + '.entry.json'))
    if proof['plan_sha256'] != digest or proof['token'] != args.token:
        raise ValueError('Entry proof token/plan mismatch')
    cpus = worker_identity(slot, os.environ, socket.gethostname(), Path('/proc/self/cgroup').read_text(), os.sched_getaffinity(0))
    entry_identity(slot, proof['node'], dict(SLURM_JOB_ID=str(proof['job']), SLURM_STEP_ID=proof['step']), proof['cgroup'])
    if (proof['assigned_cpus'] != cpus or proof['container']['id'] != slot['identity']
            or proof['mutex_inode'] != int(os.environ['RETURN18_MUTEX_INODE'])
            or cpus != lane_cpus(slot, proof['allocation']['cpus'], proof['container']['allowed_cpus'])
            or os.environ.get('SLURM_STEP_ID') != proof['step']):
        raise ValueError('Worker CPU/container identity differs from its entry proof')
    os.sched_setaffinity(0, cpus)
    prerequisite = r120_ready()
    if any(stage['kind'] == 'train' for stage in job['stages']):
        for name in ('last.pt', 'best.pt', 'training_complete.json', 'test_complete.json', 'status.json'):
            if (repo_path(job['root']) / name).exists():
                raise ValueError('Fresh-only job refuses existing training artifact: ' + name)
    with gpu_mutex(mutex_path(slot), int(os.environ['RETURN18_MUTEX_INODE'])) as descriptor:
        gpu = gpu_probe(slot)
        cache = ensure_cache(plan, slot)
        receipt = dict(state='running', started_utc=utc(), job=job['id'], arm=job['arm'], slot=job['slot'],
            plan_sha256=digest, protocol=plan['protocol'], token=args.token, pid=os.getpid(),
            gpu=gpu, cpus=cpus, entry=proof, r120=prerequisite, cache=cache, stages=[], automatic_retry=False)
        return run_stages(args.plan, digest, plan, job, descriptor, receipt)


def dispatch(args, plan, digest):
    directory = DISPATCH / plan['name']
    controller = directory / 'controller.json'
    record = dict(state='waiting', started_utc=utc(), plan_sha256=digest, jobs={}, automatic_retry=False)
    write(controller, record, exclusive=True)
    frozen = directory / 'plan.json'
    # Preserve the approved bytes, rather than reserializing the SHA-bound plan.
    with frozen.open('xb') as stream:
        stream.write(args.plan.read_bytes())
        stream.flush()
        os.fsync(stream.fileno())
    if sha256(frozen) != digest:
        raise ValueError('Plan changed while freezing it')
    processes, completed, failed, counts = {}, set(), set(), {}
    try:
        while len(completed | failed) < len(plan['jobs']):
            validate_plan(frozen, digest)
            try:
                record['r120'] = r120_ready()
            except Waiting as error:
                if not args.wait:
                    raise
                record.update(state='waiting_r120', reason=str(error))
                write(controller, record)
                time.sleep(args.poll_seconds)
                continue
            busy_slots = {job['slot'] for job, _, _, _ in processes.values()}
            for job in plan['jobs']:
                name = job['id']
                if name in completed | failed or name in processes or job['slot'] in busy_slots:
                    continue
                dependencies = set(job.get('depends_on', []))
                if dependencies & failed:
                    failed.add(name)
                    record['jobs'][name] = dict(state='blocked_dependency', exit_status=1)
                    continue
                if not dependencies <= completed:
                    continue
                if (job_dir(plan, job) / 'attempt.json').exists():
                    raise ValueError('An existing worker attempt forbids dispatch/retry: ' + name)
                token = uuid.uuid4().hex
                command = srun_command(frozen, job, token, digest)
                path = job_dir(plan, job) / 'transports' / (token + '.log')
                path.parent.mkdir(parents=True, exist_ok=True)
                log = path.open('x')
                try:
                    process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
                except BaseException:
                    log.close()
                    raise
                processes[name] = (job, process, log, token)
                busy_slots.add(job['slot'])
                counts[name] = counts.get(name, 0) + 1
                record['jobs'][name] = dict(state='transport', command=command, pid=process.pid,
                    token=token, log=str(path), resource_checks=counts[name])
            for name, (job, process, log, token) in list(processes.items()):
                code = process.poll()
                if code is None:
                    continue
                log.close()
                del processes[name]
                result_path = job_dir(plan, job) / 'transports' / (token + '.result.json')
                result = read(result_path) if result_path.is_file() else {}
                attempt_path = job_dir(plan, job) / 'attempt.json'
                waiting = (code == WAIT_EXIT and result.get('state') == 'waiting'
                    and result.get('token') == token and result.get('plan_sha256') == digest
                    and not attempt_path.exists())
                transport = dict(record['jobs'][name], exit_status=code, result=result, ended_utc=utc())
                write(job_dir(plan, job) / 'transports' / (token + '.json'), transport, exclusive=True)
                if waiting and args.wait:
                    record['jobs'][name] = dict(state='waiting_resource', reason=result.get('reason'))
                    continue
                attempt = read(attempt_path) if attempt_path.exists() else {}
                success = code == 0 and attempt.get('state') == 'complete' and attempt.get('plan_sha256') == digest
                (completed if success else failed).add(name)
                record['jobs'][name] = dict(state='complete' if success else 'failed', exit_status=code,
                    attempt=str(attempt_path), result=result)
            record.update(state='running', completed=sorted(completed), failed=sorted(failed))
            write(controller, record)
            if len(completed | failed) < len(plan['jobs']):
                time.sleep(args.poll_seconds)
        record.update(state='failed' if failed else 'complete', exit_status=1 if failed else 0, ended_utc=utc())
        write(controller, record)
        return record['exit_status']
    except BaseException as error:
        record.update(state='failed', exit_status=1, error=f'{type(error).__name__}: {error}', ended_utc=utc())
        write(controller, record)
        raise
    finally:
        # Do not cancel submitted work or release its inherited GPU mutex early.
        for _, process, log, _ in processes.values():
            process.wait()
            log.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('describe', 'dispatch', 'entry', 'worker'))
    parser.add_argument('--plan', required=True, type=Path)
    parser.add_argument('--job')
    parser.add_argument('--token')
    parser.add_argument('--plan-sha256')
    parser.add_argument('--wait', action='store_true', help='Wait only before attempts for proven busy resources/prerequisites')
    parser.add_argument('--poll-seconds', type=float, default=30)
    args = parser.parse_args(argv)
    args.plan = args.plan.resolve()
    if args.poll_seconds < 1 or not math.isfinite(args.poll_seconds):
        parser.error('--poll-seconds must be finite and >=1')
    result_path = None
    try:
        expected = os.environ.get('RETURN18_PLAN_SHA256') if args.action == 'worker' else args.plan_sha256
        plan, digest = validate_plan(args.plan, expected)
        if args.action == 'describe':
            print(json.dumps(dict(plan_sha256=digest, plan=plan, slots=SLOTS,
                resource_contract=dict(cpus=4, memory='9G', gres='none', cache=str(CACHE)),
                commands={job['id']: srun_command(args.plan, job, '<transport-token>') for job in plan['jobs']}), indent=2))
            return 0
        if args.action == 'dispatch':
            if args.job or args.token:
                raise ValueError('dispatch selects only the literal plan jobs')
            return dispatch(args, plan, digest)
        if not args.job or not args.token:
            raise ValueError('entry/worker require --plan --job --token from the controller')
        if args.action == 'entry' and not args.plan_sha256:
            raise ValueError('entry requires its controller-pinned --plan-sha256')
        identifier(args.token)
        job = find_job(plan, args.job)
        result_path = job_dir(plan, job) / 'transports' / (args.token + '.result.json')
        code = entry(args, plan, digest, job) if args.action == 'entry' else worker(args, plan, digest, job)
        if args.action == 'worker':
            write(result_path, dict(state='complete' if code == 0 else 'failed', exit_status=code,
                token=args.token, plan_sha256=digest, utc=utc()), exclusive=True)
        return code
    except Waiting as error:
        if result_path is not None:
            write(result_path, dict(state='waiting', token=args.token, plan_sha256=digest,
                reason=str(error), exit_status=WAIT_EXIT, utc=utc()), exclusive=True)
        print(str(error), file=sys.stderr)
        return WAIT_EXIT
    except Exception as error:
        if result_path is not None and not result_path.exists():
            write(result_path, dict(state='failed', token=args.token, plan_sha256=digest,
                reason=f'{type(error).__name__}: {error}', exit_status=1, utc=utc()), exclusive=True)
        print(f'{type(error).__name__}: {error}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())

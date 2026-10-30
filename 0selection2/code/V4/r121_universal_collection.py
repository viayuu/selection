"""Versioned single-root resume with explicit CPU native inputs for all TSP/ATSP."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from functools import lru_cache
import errno
import fcntl
import hashlib
import inspect
import json
import os
from pathlib import Path
import traceback
import uuid

from . import r121_collection as base
from . import r121_input_boundary as boundary

VERSION = 'r121_universal_explicit_cpu_collection_v2_20261030'
ROOT = base.data.ROOT / 'collection_explicit_cpu_v2'
PARENT = boundary.PARENT
OVERLAY = boundary.ROOT
BOUNDARY_SHA = '0e869a9f134ba7d3eeb43ede720e7169a50b40e57a02f91ccc60a370707debbe'
ORIGINAL_SOLVE = base.solve_native
ORIGINAL_WORKER = base.worker


def create(root=ROOT, parent=PARENT, overlay=OVERLAY):
    root, parent, overlay = (base.data.owned(p) for p in (root, parent, overlay))
    if root in (parent, overlay) or not root.parent == parent.parent == overlay.parent:
        raise ValueError('Use a NEW sibling root, never either live source tree')
    if base.sha(base.__file__) != boundary.FROZEN_SHA or base.sha(boundary.__file__) != BOUNDARY_SHA:
        raise ValueError('The known original and CPU-boundary helper identities must remain frozen')
    protocol, manifest, inventory = base.load(parent)
    old_recovery, old_protocol, _, _ = boundary.load(overlay)
    groups = base.task_groups(manifest)
    typed = sorted(k for k in groups if k.split('/')[0] in ('TSP', 'ATSP'))
    if (protocol != old_protocol or old_recovery['parent_root'] != str(parent)
            or len(groups) != 89 or len(manifest['tasks']) != 113920 or len(typed) != 13
            or protocol['mode'] != 'cached_native_v1' or protocol['lanes'] != 4
            or protocol['batch_size'] != 1 or any(len(v) != 1280 for v in groups.values())
            or len(old_recovery['targets']) != 5 or not set(old_recovery['targets']) <= set(typed)):
        raise ValueError('Require the identical native full18/89/113920 cached protocol and known five-key overlay')
    from . import r114_group_worker
    sources = {str(Path(p).resolve()): base.sha(p) for p in (__file__, base.__file__, boundary.__file__,
        base.data.__file__, base.data.r114.__file__, r114_group_worker.__file__)}
    pins = {str(p): base.sha(p) for p in
        (parent / 'protocol.json', overlay / 'protocol.json', overlay / 'recovery.json')}
    universal = base.data.seal(dict(version=VERSION, parent_root=str(parent), input_boundary_root=str(overlay),
        input_boundary_recovery_sha256=old_recovery['recovery_sha256'], input_boundary_keys=old_recovery['targets'],
        native_collection_sha256=protocol['collection_sha256'], explicit_cpu_deployments=typed,
        source_sha256=sources, immutable_source_metadata_sha256=pins,
        source_policy='five immutable input-boundary target keys from inputV1; other84 from original parent; no root/cost selection',
        mutable_prefix_snapshot_at_create=False, bootstrap_requires_main_physical_mutex=True,
        native_budget_changes=False, deployment_seed_changes=False, candidate_pool_unchanged=True,
        only_native_change="TSP/ATSP torch.tensor(payload['item'], dtype=torch.float32, device='cpu')",
        other_tasks='unchanged original session.solve', original_failure_flags_adopted=False), 'universal_sha256')
    base.write(root / 'protocol.json', protocol)
    base.write(root / 'universal.json', universal)
    return dict(root=str(root), universal_sha256=universal['universal_sha256'], deployments=89,
        planned_labels=113920, explicit_cpu_deployments=typed, prefixes_snapshotted=False, solver_execution=False)


def load(root):
    root = base.data.owned(root)
    universal = base.read(root / 'universal.json')
    base.data.verify_seal(universal, 'universal_sha256')
    if universal['version'] != VERSION:
        raise ValueError('Wrong universal layer version')
    for path, digest in {**universal['source_sha256'], **universal['immutable_source_metadata_sha256']}.items():
        if base.sha(path) != digest:
            raise ValueError('Pinned universal source/protocol changed: ' + path)
    protocol, manifest, inventory = base.load(root)
    if protocol['collection_sha256'] != universal['native_collection_sha256']:
        raise ValueError('Native collection identity changed')
    return universal, protocol, manifest, inventory


def lane_keys(protocol, lane):
    if type(lane) is not int or not 0 <= lane < protocol['lanes']:
        raise ValueError('Use a fixed original lane0..3')
    return [key for index, key in enumerate(protocol['group_order']) if index % protocol['lanes'] == lane]


def require_mutex():
    value = os.environ.get('RETURN18_GPU_MUTEX_FD')
    if value is None:
        raise PermissionError('Bootstrap requires the inherited MAIN physical GPU mutex FD')
    descriptor = int(value)
    info = os.fstat(descriptor)
    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    return dict(fd=descriptor, device=info.st_dev, inode=info.st_ino)


def assert_sources_quiet(universal, assigned, lane):
    roots = {Path(universal[k]).resolve() for k in ('parent_root', 'input_boundary_root')}
    modules = ('code.V4.r121_collection', 'code.V4.r121_input_boundary', 'code.V4.r121_parallel_collection')
    for path in Path('/proc').glob('[0-9]*/cmdline'):
        try:
            if path.stat().st_uid != os.getuid():
                continue
            argv = path.read_bytes().decode(errors='replace').strip('\0').split('\0')
            if not any(m in argv for m in modules):
                continue
            def option(name):
                for index, arg in enumerate(argv):
                    if arg == name and index + 1 < len(argv):
                        return argv[index + 1]
                    if arg.startswith(name + '='):
                        return arg[len(name) + 1:]
                return None
            source_root = option('--root')
            if source_root is None or Path(source_root).resolve() not in roots:
                continue
            if option('--deployment') in assigned or option('--lane') == str(lane):
                raise base.NotReady('Old assigned producer is still active; preserve/drain PID ' + path.parent.name)
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue


def source_root(universal, key):
    return Path(universal['input_boundary_root'] if key in universal['input_boundary_keys'] else universal['parent_root'])


def validator(manifest, inventory, protocol):
    endpoints = {e['endpoint_id']: e for e in manifest['endpoints']}

    @lru_cache(maxsize=16)
    def item(endpoint_id):
        endpoint = endpoints[endpoint_id]
        if base.sha(endpoint['input_path']) != endpoint['input_sha256']:
            raise ValueError('Frozen endpoint input changed: ' + endpoint_id)
        return base.read(endpoint['input_path'])['item']

    def validate(row, task, stage, binding_sha):
        base.validate_receipt(row, task, protocol, stage)
        key = task['problem'] + '/' + task['method']
        if (row.get('profile_sha256') != inventory['deployments'][key]['profile_sha256']
                or row.get('binding_sha256') != binding_sha or row.get('device') != 'cuda:0'
                or row.get('execution_mode') != 'cached_native_v1'
                or row.get('model_seed') != base.data.r114.seed_for(2, task['problem'], task['method'], 'model')):
            raise ValueError('Native source/profile/physical binding differs: ' + key)
        base.data.check_tour(task['problem'], item(task['endpoint_id']), row['tour'], row['cost'])
    return validate


def copy_receipt(root, source, task, stage, lane, universal, binding_sha, validate):
    relative = Path('preflight') / (base.slug(task['problem'] + '/' + task['method']) + '.json') if stage == 'preflight' else Path('labels') / (task['task_id'] + '.json')
    old_path, new_path = source / relative, root / relative
    content = old_path.read_bytes()
    row = json.loads(content)
    validate(row, task, stage, binding_sha)
    key = task['problem'] + '/' + task['method']
    if key in universal['input_boundary_keys'] and (row.get('input_boundary_version') != boundary.VERSION
            or row.get('input_boundary_recovery_sha256') != universal['input_boundary_recovery_sha256']
            or row.get('observed_native_input', {}).get('device') != 'cpu'):
        raise ValueError('InputV1 source receipt lacks its real CPU-boundary provenance: ' + key)
    digest = hashlib.sha256(content).hexdigest()
    new_path.parent.mkdir(parents=True, exist_ok=True)
    if new_path.exists():
        if new_path.read_bytes() != content:
            raise ValueError('Preserve a different existing destination receipt: ' + str(new_path))
    else:
        try:
            os.link(old_path, new_path)
        except OSError as error:
            if error.errno != errno.EXDEV:
                raise
            with new_path.open('xb') as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
    if base.sha(new_path) != digest:
        raise ValueError('Source changed during byte-identical adoption')
    provenance = dict(kind='adopted_inputV1' if key in universal['input_boundary_keys'] else 'adopted_original',
        task_id=task['task_id'], stage=stage, lane=lane, source_root=str(source), source_path=str(old_path),
        source_receipt_sha256=digest, destination_path=str(new_path), destination_sha256=digest,
        universal_sha256=universal['universal_sha256'], collection_sha256=universal['native_collection_sha256'],
        byte_identical=True, new_solver_execution=False)
    base.write(root / 'adoption' / relative, provenance)
    return digest


def bootstrap_lane(root, lane, release, binding):
    root = base.data.owned(root)
    universal, protocol, manifest, inventory = load(root)
    assigned, groups = lane_keys(protocol, lane), base.task_groups(manifest)
    release_data = base.data.require_release(release, protocol['plan_sha256'])
    binding_sha = base.data.require_binding(binding, release_data, 'cuda:0')
    mutex = require_mutex()
    assert_sources_quiet(universal, assigned, lane)
    marker = root / 'bootstrap' / ('lane' + str(lane) + '.json')
    if marker.exists():
        saved = base.read(marker)
        base.data.verify_seal(saved, 'bootstrap_sha256')
        if saved['binding_sha256'] != binding_sha or saved['universal_sha256'] != universal['universal_sha256']:
            raise ValueError('Existing lane bootstrap has a different binding/layer')
        return dict(saved, status='already_bootstrapped')
    validate = validator(manifest, inventory, protocol)
    original_ids = {e['endpoint_id'] for e in manifest['endpoints'] if e['cell'] == 'original'}
    rows = {}
    for key in assigned:
        source, tasks = source_root(universal, key), groups[key]
        failure_path = source / 'unavailable' / (base.slug(key) + '.json')
        source_error = None
        if failure_path.exists():
            failure = base.read(failure_path)
            message = failure.get('first_error', '')
            if (key.split('/')[0] not in ('TSP', 'ATSP') or failure.get('error_type') != 'TypeError'
                    or 'cuda:0' not in message or 'numpy' not in message):
                raise base.NotReady('Unrelated original first error is not a generic retry target: '
                    + str(failure_path) + ': ' + message)
            source_error = dict(path=str(failure_path), sha256=base.sha(failure_path),
                covered_by='explicit CPU native input constructor; old failure preserved')
        original = next(t for t in tasks if t['endpoint_id'] in original_ids)
        binding_ref = base.read(source / 'deployment_bindings' / (base.slug(key) + '.json'))
        expected_binding = dict(deployment=key, binding_sha256=binding_sha, collection_sha256=protocol['collection_sha256'])
        if binding_ref != expected_binding:
            raise ValueError('Original per-key binding does not match this MAIN lane: ' + key)
        qualification_sha = copy_receipt(root, source, original, 'preflight', lane, universal, binding_sha, validate)
        state = base.progress(source, key)
        committed = state['next_index']
        if (type(committed) is not int or type(state['completed']) is not int
                or not 0 <= committed <= len(tasks) or state['completed'] != committed
                or (committed and state.get('collection_sha256') != protocol['collection_sha256'])):
            raise ValueError('Invalid original atomic prefix state: ' + key)
        adopted, last_sha = 0, None
        for index, task in enumerate(tasks):
            path = source / 'labels' / (task['task_id'] + '.json')
            if not path.exists():
                if index < committed or any((source / 'labels' / (t['task_id'] + '.json')).exists() for t in tasks[index + 1:]):
                    raise ValueError('Source receipts are not a complete contiguous prefix: ' + key)
                break
            last_sha = copy_receipt(root, source, task, 'label', lane, universal, binding_sha, validate)
            adopted += 1
            if adopted == committed and (state.get('last_receipt_sha256') != last_sha
                    or Path(state.get('last_receipt', '')).resolve() != path.resolve()):
                raise ValueError('Atomic source progress does not bind its last successful receipt: ' + key)
        if adopted < committed:
            raise ValueError('Never discard a committed source receipt: ' + key)
        base.write(root / 'deployment_bindings' / (base.slug(key) + '.json'), expected_binding)
        next_state = dict(next_index=adopted, completed=adopted,
            status='complete' if adopted == len(tasks) else 'running',
            collection_sha256=protocol['collection_sha256'], adopted_labels=adopted,
            universal_sha256=universal['universal_sha256'])
        if adopted:
            next_state.update(last_receipt=str(root / 'labels' / (tasks[adopted - 1]['task_id'] + '.json')),
                last_receipt_sha256=last_sha)
        base.atomic(root / 'progress' / (base.slug(key) + '.json'), next_state)
        rows[key] = dict(source_root=str(source), adopted_labels=adopted, next_index=adopted,
            source_atomic_progress=committed, adopted_saved_uncommitted_tail=adopted - committed,
            preflight_sha256=qualification_sha, source_previous_error=source_error, source_failure_flags_copied=False)
    result = base.data.seal(dict(status='bootstrap_complete', version=VERSION, lane=lane,
        assigned_deployments=assigned, deployments=rows, adopted_labels=sum(r['adopted_labels'] for r in rows.values()),
        binding_sha256=binding_sha, universal_sha256=universal['universal_sha256'],
        collection_sha256=protocol['collection_sha256'], main_mutex=mutex,
        old_records_overwritten=False, new_solver_execution=False), 'bootstrap_sha256')
    base.write(marker, result)
    return result


def explicit_solver(universal):
    source = inspect.getsource(ORIGINAL_SOLVE)
    old = "native = torch.tensor(payload['item'], dtype=torch.float32)"
    if source.count(old) != 1:
        raise ValueError('Known single native constructor no longer matches')
    namespace = dict(ORIGINAL_SOLVE.__globals__)
    exec(compile(source.replace(old, old[:-1] + ", device='cpu')"), str(Path(__file__).resolve()) + ':explicit_cpu', 'exec'), namespace)
    typed_solve = namespace['solve_native']

    def solve(session, manifest, task):
        typed = task['problem'] in ('TSP', 'ATSP')
        row = (typed_solve if typed else ORIGINAL_SOLVE)(session, manifest, task)
        row.update(universal_version=VERSION, universal_sha256=universal['universal_sha256'],
            universal_execution='new_explicit_cpu' if typed else 'new_unchanged_session_solve',
            native_input_placement='explicit_cpu' if typed else 'unchanged_session.solve')
        return row
    return solve


def traceback_worker():
    # Local frozen control flow; only failure metadata gains the complete traceback.
    source = inspect.getsource(ORIGINAL_WORKER)
    old = "failure = dict(task, stage=phase, status='unavailable', first_error=str(error),"
    if source.count(old) != 1:
        raise ValueError('Frozen worker exception site changed')
    namespace = dict(ORIGINAL_WORKER.__globals__, traceback=traceback)
    exec(compile(source.replace(old, old + '\n                    traceback=traceback.format_exc(),'),
        str(Path(__file__).resolve()) + ':traceback_only_worker', 'exec'), namespace)
    return namespace['worker']


def worker(root, key, phase, max_jobs, device, release, binding):
    root = base.data.owned(root)
    universal, protocol, _, _ = load(root)
    if key not in protocol['group_order']:
        raise ValueError('Unknown legal deployment')
    lane = protocol['group_order'].index(key) % protocol['lanes']
    marker = base.read(root / 'bootstrap' / ('lane' + str(lane) + '.json'))
    base.data.verify_seal(marker, 'bootstrap_sha256')
    release_data = base.data.require_release(release, protocol['plan_sha256'])
    binding_sha = base.data.require_binding(binding, release_data, device)
    if marker['universal_sha256'] != universal['universal_sha256'] or marker['binding_sha256'] != binding_sha:
        raise ValueError('Bootstrap/main physical binding or universal layer changed')
    previous = base.solve_native
    base.solve_native = explicit_solver(universal)
    try:
        result = traceback_worker()(root, key, phase, max_jobs, device, release, binding)
    finally:
        base.solve_native = previous
    return dict(result, universal_version=VERSION, universal_sha256=universal['universal_sha256'])


def status(root):
    root = base.data.owned(root)
    universal, protocol, manifest, _ = load(root)
    rows = {}
    for key, tasks in base.task_groups(manifest).items():
        state = base.progress(root, key)
        rows[key] = dict(planned=len(tasks), completed=state['completed'],
            preflight=(root / 'preflight' / (base.slug(key) + '.json')).is_file(),
            unavailable=(root / 'unavailable' / (base.slug(key) + '.json')).exists())
    lanes = [lane for lane in range(protocol['lanes']) if (root / 'bootstrap' / ('lane' + str(lane) + '.json')).is_file()]
    return dict(root=str(root), universal_sha256=universal['universal_sha256'], deployments=rows,
        bootstrapped_lanes=lanes, planned_labels=len(manifest['tasks']),
        completed_labels=sum(r['completed'] for r in rows.values()),
        full89_complete=len(lanes) == protocol['lanes'] and all(r['preflight'] and not r['unavailable']
            and r['completed'] == r['planned'] for r in rows.values()), solver_execution=False)


def export(root, output=None):
    root = base.data.owned(root)
    universal, protocol, manifest, inventory = load(root)
    if not status(root)['full89_complete']:
        raise base.NotReady('Require complete NEW single-root full18/89/113920 receipts; no partial-pool export')
    output = base.data.owned(output or root / 'family_labels.jsonl')
    if output.exists():
        raise ValueError('Preserve existing complete export')
    output.parent.mkdir(parents=True, exist_ok=True)
    validate = validator(manifest, inventory, protocol)
    groups = defaultdict(list)
    for task in manifest['tasks']:
        groups[task['endpoint_id']].append(task)
    bindings = {key: base.read(root / 'deployment_bindings' / (base.slug(key) + '.json'))['binding_sha256']
        for key in base.task_groups(manifest)}
    temp = output.with_name('.' + output.name + '.' + uuid.uuid4().hex)
    digest, counts, endpoints = hashlib.sha256(), Counter(), 0
    try:
        with temp.open('xb') as stream:
            for endpoint in manifest['endpoints']:
                records, provenance = [], []
                for task in groups[endpoint['endpoint_id']]:
                    path = root / 'labels' / (task['task_id'] + '.json')
                    row = base.read(path)
                    key = task['problem'] + '/' + task['method']
                    validate(row, task, 'label', bindings[key])
                    adoption = root / 'adoption' / 'labels' / path.name
                    if adoption.exists():
                        ref = base.read(adoption)
                        if (ref['universal_sha256'] != universal['universal_sha256']
                                or ref['source_receipt_sha256'] != base.sha(path)
                                or ref['source_receipt_sha256'] != base.sha(ref['source_path'])
                                or Path(ref['source_root']) != source_root(universal, key)):
                            raise ValueError('Adopted native receipt/source provenance changed')
                        kind = ref['kind']
                        provenance.append(ref)
                    else:
                        kind = 'new_explicit_cpu' if task['problem'] in ('TSP', 'ATSP') else 'new_unchanged_session_solve'
                        if (row.get('universal_version') != VERSION or row.get('universal_sha256') != universal['universal_sha256']
                                or row.get('universal_execution') != kind):
                            raise ValueError('New label lacks universal explicit-input provenance')
                        provenance.append(dict(kind=kind, task_id=task['task_id'], receipt_path=str(path),
                            receipt_sha256=base.sha(path), universal_sha256=universal['universal_sha256'],
                            native_input_placement=row['native_input_placement']))
                    counts[kind] += 1
                    records.append(row)
                targets = base.data.full_pool_targets(manifest['pools'][endpoint['problem']], records)
                encoded = (json.dumps(dict(endpoint=endpoint, targets=targets,
                    collection_sha256=protocol['collection_sha256'], universal_version=VERSION,
                    universal_sha256=universal['universal_sha256'], label_provenance=provenance),
                    sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()
                stream.write(encoded)
                digest.update(encoded)
                endpoints += 1
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temp, output)
    finally:
        temp.unlink(missing_ok=True)
    result = dict(complete=True, full18=True, version=base.data.VERSION, candidate_pool_unchanged=True,
        historical_labels_replaced=False, path=str(output), sha256=digest.hexdigest(), endpoints=endpoints,
        labels=len(manifest['tasks']), collection_sha256=protocol['collection_sha256'],
        family_input_manifest=protocol['input_manifest'], manifest_sha256=manifest['manifest_sha256'],
        universal_version=VERSION, universal_sha256=universal['universal_sha256'],
        native_receipt_root=str(root), label_provenance_counts=dict(counts),
        source_roots=dict(original=universal['parent_root'], inputV1=universal['input_boundary_root']),
        qualification='byte-identical adopted native successful originals, not new universal preflight solves')
    base.write(output.with_suffix('.receipt.json'), result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    init = commands.add_parser('create')
    init.add_argument('--root', type=Path, default=ROOT)
    init.add_argument('--parent', type=Path, default=PARENT)
    init.add_argument('--input-boundary-root', type=Path, default=OVERLAY)
    bootstrap = commands.add_parser('bootstrap-lane')
    bootstrap.add_argument('--root', type=Path, default=ROOT)
    bootstrap.add_argument('--lane', type=int, choices=range(4), required=True)
    bootstrap.add_argument('--release', type=Path, required=True)
    bootstrap.add_argument('--binding', type=Path, required=True)
    run = commands.add_parser('worker')
    run.add_argument('--root', type=Path, default=ROOT)
    run.add_argument('--deployment', required=True)
    run.add_argument('--phase', choices=('preflight', 'label'), required=True)
    run.add_argument('--max-jobs', type=int, required=True)
    run.add_argument('--device', default='cuda:0')
    run.add_argument('--release', type=Path, required=True)
    run.add_argument('--binding', type=Path, required=True)
    run.add_argument('--result', type=Path, required=True)
    for name in ('status', 'export'):
        sub = commands.add_parser(name)
        sub.add_argument('--root', type=Path, default=ROOT)
        if name == 'export':
            sub.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == 'create':
            result = create(args.root, args.parent, args.input_boundary_root)
        elif args.command == 'bootstrap-lane':
            result = bootstrap_lane(args.root, args.lane, args.release, args.binding)
        elif args.command == 'worker':
            if args.root.resolve() not in args.result.resolve().parents:
                raise ValueError('Result must remain under the NEW universal root')
            result = worker(args.root, args.deployment, args.phase, args.max_jobs, args.device, args.release, args.binding)
            base.write(args.result, result)
        elif args.command == 'status':
            result = status(args.root)
        else:
            result = export(args.root, args.output)
    except Exception as error:
        result = dict(status='NotReady' if isinstance(error, base.NotReady) else 'failed',
            first_error=str(error), error_type=type(error).__name__, traceback=traceback.format_exc(), automatic_retry=False)
        if args.command == 'worker' and args.root.resolve() in args.result.resolve().parents:
            base.write(args.result, result)
        print(json.dumps(result, sort_keys=True), flush=True)
        return 3 if isinstance(error, base.NotReady) else 1
    print(json.dumps(result, sort_keys=True), flush=True)
    return 2 if result.get('status') == 'unavailable' else 0


if __name__ == '__main__':
    raise SystemExit(main())

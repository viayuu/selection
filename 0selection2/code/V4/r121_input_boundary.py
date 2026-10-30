"""Trace-backed CPU input-boundary overlay; no edits to frozen native code."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import inspect
import json
import os
from pathlib import Path
import traceback
import uuid

from . import r121_collection as base

VERSION = 'r121_explicit_cpu_native_input_v1_20261030'
FROZEN_SHA = '876cff5800f0c0a1067b7a36eee3e154b4b73dedb8030873a69dd7cefec49674'
ROOT = base.data.ROOT / 'collection_input_boundary_v1'
PARENT = base.data.ROOT / 'collection_persistent_v1'
TRACE = base.data.ROOT / 'collection_output_diagnostic_v1/ATSP__UNICO_MatPOENet_attempt1.json'
ORIGINAL_SOLVE = base.solve_native


def validate_trace(trace, protocol, inventory):
    observed = trace.get('native_input_metadata', {}).get('first', [])
    frames = trace.get('exception', {}).get('frames', [])
    if (trace.get('status') != 'diagnostic_failed' or trace.get('stage') != 'native_backend.run'
            or trace.get('native_calls') != 1 or trace.get('input_or_output_normalization') is not False
            or trace.get('frozen_collector_sha256') != FROZEN_SHA
            or trace.get('collection_sha256') != protocol['collection_sha256']
            or not observed or observed[0].get('device') != 'cuda:0'
            or not any(f['name'] == 'audit_tours' and 'np.asarray(item[0]' in (f.get('line') or '') for f in frames)
            or trace.get('profile_sha256') != inventory['deployments'][trace['deployment']]['profile_sha256']):
        raise ValueError('Actual pinned native audit/input-device traceback is required; no inferred repair')


def cpu_solver(root, recovery):
    source = inspect.getsource(ORIGINAL_SOLVE)
    before = "native = torch.tensor(payload['item'], dtype=torch.float32)"
    after = "native = torch.tensor(payload['item'], dtype=torch.float32, device='cpu')"
    if source.count(before) != 1:
        raise ValueError('Frozen tensor constructor changed; no speculative patch')
    namespace = dict(ORIGINAL_SOLVE.__globals__)
    exec(compile(source.replace(before, after), str(Path(__file__).resolve()) + ':one_cpu_input_boundary', 'exec'), namespace)
    compiled = namespace['solve_native']

    def solve(session, manifest, task):
        native_run = session.backend.run
        observed = {}

        def checked(items):
            observed.update(device=str(items[0].device), dtype=str(items[0].dtype), shape=list(items[0].shape))
            if len(items) != 1 or items[0].device.type != 'cpu':
                raise ValueError('The native audit requires the explicit CPU singleton input')
            return native_run(items)

        session.backend.run = checked
        try:
            row = compiled(session, manifest, task)
        except Exception as error:
            evidence = dict(task_id=task['task_id'], input_boundary_version=VERSION,
                recovery_sha256=recovery['recovery_sha256'], observed_native_input=observed,
                error_type=type(error).__name__, first_error=str(error),
                traceback=traceback.format_exc(), automatic_retry=False)
            base.write(Path(root) / 'tracebacks' / (task['task_id'] + '_' + uuid.uuid4().hex + '.json'), evidence)
            raise
        finally:
            session.backend.run = native_run
        row.update(input_boundary_version=VERSION, input_boundary_recovery_sha256=recovery['recovery_sha256'],
            observed_native_input=observed, solver_model_seed_profile_changes=False)
        return row

    return solve


def create(parent=PARENT, root=ROOT, trace_path=TRACE):
    parent, root = base.data.owned(parent), base.data.owned(root)
    if root == parent or parent in root.parents:
        raise ValueError('Use a NEW sibling overlay, never the live parent tree')
    if base.sha(base.__file__) != FROZEN_SHA:
        raise ValueError('Only the frozen876 collector is supported')
    protocol, manifest, inventory = base.load(parent)
    trace = base.read(trace_path)
    validate_trace(trace, protocol, inventory)
    groups = base.task_groups(manifest)
    originals = {e['endpoint_id'] for e in manifest['endpoints'] if e['cell'] == 'original'}
    original_tasks = {key: next(t for t in group if t['endpoint_id'] in originals)
        for key, group in groups.items()}
    failures, reused, attempted, other_errors = {}, {}, set(), {}
    for path in sorted((parent / 'unavailable').glob('*.json')):
        row = base.read(path)
        key = row['problem'] + '/' + row['method']
        if row.get('stage') != 'preflight' or row.get('status') != 'unavailable' or key not in original_tasks:
            raise base.NotReady('Original qualification first error requires separate handling: ' + str(path))
        task = original_tasks[key]
        if (row['collection_sha256'] != protocol['collection_sha256']
                or any(row[k] != task[k] for k in
                    ('task_id', 'input_sha256', 'solver_seed', 'deployment_id', 'global_id'))):
            raise ValueError('Original failed task/recipe/input changed: ' + key)
        binding = base.read(parent / 'deployment_bindings' / (base.slug(key) + '.json'))
        if (binding['binding_sha256'] != row['binding_sha256'] or binding['deployment'] != key
                or binding['collection_sha256'] != protocol['collection_sha256']):
            raise ValueError('Original failed physical binding changed: ' + key)
        attempted.add(key)
        if (row.get('problem') not in ('TSP', 'ATSP') or row.get('stage') != 'preflight'
                or row.get('error_type') != 'TypeError' or 'cuda:0' not in row.get('first_error', '')
                or 'numpy' not in row['first_error']):
            other_errors[key] = dict(path=str(path), first_error=row.get('first_error'))
            continue
        failures[key] = dict(path=str(path), sha256=base.sha(path), task_id=row['task_id'],
            binding_sha256=row['binding_sha256'], profile_sha256=inventory['deployments'][key]['profile_sha256'])
    if trace['deployment'] not in failures or trace['binding_sha256'] != failures[trace['deployment']]['binding_sha256']:
        raise ValueError('Actual trace must bind one of the preserved original failures')
    for path in sorted((parent / 'preflight').glob('*.json')):
        row = base.read(path)
        key = row['problem'] + '/' + row['method']
        base.validate_receipt(row, original_tasks[key], protocol, 'preflight')
        if row['profile_sha256'] != inventory['deployments'][key]['profile_sha256']:
            raise ValueError('Existing qualification profile changed')
        binding = base.read(parent / 'deployment_bindings' / (base.slug(key) + '.json'))
        if (binding['binding_sha256'] != row['binding_sha256'] or binding['deployment'] != key
                or binding['collection_sha256'] != protocol['collection_sha256'] or key in attempted):
            raise ValueError('Original qualification physical binding/status conflict: ' + key)
        attempted.add(key)
        reused[key] = dict(path=str(path), sha256=base.sha(path), binding_sha256=row['binding_sha256'])
    pending = sorted(set(groups) - attempted)
    if pending or other_errors:
        raise base.NotReady(json.dumps(dict(original_qualifications_attempted=len(attempted), expected=len(groups),
            pending=pending, other_first_errors=other_errors, automatic_retry=False), sort_keys=True))
    from . import r114_group_worker
    sources = {str(Path(p).resolve()): base.sha(p) for p in
        (__file__, base.__file__, base.data.__file__, base.data.r114.__file__, r114_group_worker.__file__)}
    recovery = base.data.seal(dict(version=VERSION, parent_root=str(parent),
        parent_protocol_file_sha256=base.sha(parent / 'protocol.json'), parent_collection_sha256=protocol['collection_sha256'],
        actual_trace=dict(path=str(Path(trace_path).resolve()), sha256=base.sha(trace_path)),
        targets=sorted(failures), original_failures=failures, reused_qualification_refs=reused,
        original_qualification_attempts=len(attempted), expected_deployments=len(groups),
        source_sha256=sources, only_code_change="torch.tensor(..., dtype=float32, device='cpu') at native item builder",
        backend_device_transfer='unchanged original backend .to(self.device)',
        global_default_device_change=False, solver_model_seed_profile_changes=False,
        old_records_overwritten=False, full_native89_unchanged=True), 'recovery_sha256')
    base.write(root / 'protocol.json', protocol)
    base.write(root / 'recovery.json', recovery)
    return dict(root=str(root), recovery_sha256=recovery['recovery_sha256'], targets=recovery['targets'],
        reused_qualifications=len(reused), solver_execution=False, actual_gpu_recovery='PENDING main per-key preflights')


def load(root):
    root = base.data.owned(root)
    recovery = base.read(root / 'recovery.json')
    base.data.verify_seal(recovery, 'recovery_sha256')
    if recovery['version'] != VERSION:
        raise ValueError('Wrong input-boundary layer')
    for path, digest in recovery['source_sha256'].items():
        if base.sha(path) != digest:
            raise ValueError('Pinned boundary/helper changed: ' + path)
    parent = Path(recovery['parent_root'])
    if base.sha(parent / 'protocol.json') != recovery['parent_protocol_file_sha256']:
        raise ValueError('Parent protocol changed')
    protocol, manifest, inventory = base.load(parent)
    if base.read(root / 'protocol.json') != protocol:
        raise ValueError('Native parent protocol must remain identical')
    proof = recovery['actual_trace']
    if base.sha(proof['path']) != proof['sha256']:
        raise ValueError('Actual traceback evidence changed')
    validate_trace(base.read(proof['path']), protocol, inventory)
    for ref in recovery['original_failures'].values():
        if base.sha(ref['path']) != ref['sha256']:
            raise ValueError('Original first-error record changed')
    for ref in recovery['reused_qualification_refs'].values():
        if base.sha(ref['path']) != ref['sha256']:
            raise ValueError('Reused original qualification changed')
    return recovery, protocol, manifest, inventory


def worker(root, key, phase, max_jobs, device, release, binding):
    recovery, protocol, manifest, _ = load(root)
    if key not in recovery['targets']:
        raise ValueError('Only snapshotted actually failed CUDA/NumPy keys may execute in this overlay')
    actual_release = base.data.require_release(release, protocol['plan_sha256'])
    actual_binding = base.data.require_binding(binding, actual_release, device)
    failure = recovery['original_failures'][key]
    if actual_binding != failure['binding_sha256']:
        raise ValueError('Use THIS key original physical binding, not a shared replacement GPU')
    if phase == 'preflight':
        originals = {e['endpoint_id'] for e in manifest['endpoints'] if e['cell'] == 'original'}
        task = next(t for t in base.task_groups(manifest)[key] if t['endpoint_id'] in originals)
        if task['task_id'] != failure['task_id']:
            raise ValueError('Rerun exactly the failed original')
    previous = base.solve_native
    base.solve_native = cpu_solver(root, recovery)
    try:
        result = base.worker(root, key, phase, max_jobs, device, release, binding)
    finally:
        base.solve_native = previous
    return dict(result, input_boundary_version=VERSION, recovery_sha256=recovery['recovery_sha256'])


def status(root):
    recovery, _, manifest, _ = load(root)
    parent = Path(recovery['parent_root'])
    rows = []
    for key, tasks in base.task_groups(manifest).items():
        actual = Path(root) if key in recovery['targets'] else parent
        progress = base.progress(actual, key)
        rows.append(dict(deployment=key, root=str(actual), planned=len(tasks), completed=progress['completed'],
            preflight=(actual / 'preflight' / (base.slug(key) + '.json')).exists(),
            unavailable=(actual / 'unavailable' / (base.slug(key) + '.json')).exists()))
    return dict(recovery_sha256=recovery['recovery_sha256'], deployments=rows,
        planned_labels=len(manifest['tasks']), completed_labels=sum(r['completed'] for r in rows),
        full89_complete=all(r['completed'] == r['planned'] and not r['unavailable'] for r in rows))


def export(root, output=None):
    recovery, protocol, manifest, inventory = load(root)
    if not status(root)['full89_complete']:
        raise base.NotReady('Full18/89 parent+input-overlay labels required; no partial/reduced-pool export')
    root, parent = Path(root), Path(recovery['parent_root'])
    output = base.data.owned(output or root / 'family_labels.jsonl')
    if output.exists():
        raise ValueError('Preserve existing complete export')
    grouped = defaultdict(list)
    for task in manifest['tasks']:
        grouped[task['endpoint_id']].append(task)
    temp = output.with_name('.' + output.name + '.' + uuid.uuid4().hex)
    digest, count = hashlib.sha256(), 0
    try:
        with temp.open('xb') as stream:
            for endpoint in manifest['endpoints']:
                item = base.read(endpoint['input_path'])['item']
                records = []
                for task in grouped[endpoint['endpoint_id']]:
                    key = task['problem'] + '/' + task['method']
                    actual = root if key in recovery['targets'] else parent
                    row = base.read(actual / 'labels' / (task['task_id'] + '.json'))
                    base.validate_receipt(row, task, protocol)
                    if row['profile_sha256'] != inventory['deployments'][key]['profile_sha256']:
                        raise ValueError('Native profile identity changed')
                    if key in recovery['targets'] and row.get('input_boundary_recovery_sha256') != recovery['recovery_sha256']:
                        raise ValueError('Missing input-layer provenance')
                    base.data.check_tour(task['problem'], item, row['tour'], row['cost'])
                    records.append(row)
                targets = base.data.full_pool_targets(manifest['pools'][endpoint['problem']], records)
                encoded = (json.dumps(dict(endpoint=endpoint, targets=targets, collection_sha256=protocol['collection_sha256']),
                    sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()
                stream.write(encoded)
                digest.update(encoded)
                count += 1
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temp, output)
    finally:
        temp.unlink(missing_ok=True)
    result = dict(complete=True, full18=True, version=base.data.VERSION, candidate_pool_unchanged=True,
        historical_labels_replaced=False, path=str(output), sha256=digest.hexdigest(), endpoints=count,
        labels=len(manifest['tasks']), collection_sha256=protocol['collection_sha256'],
        family_input_manifest=protocol['input_manifest'], manifest_sha256=manifest['manifest_sha256'],
        input_boundary_version=VERSION, input_boundary_recovery_sha256=recovery['recovery_sha256'],
        native_receipt_roots=dict(parent=str(parent), recovered=str(root)))
    base.write(output.with_suffix('.receipt.json'), result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    init = commands.add_parser('create')
    init.add_argument('--parent', type=Path, default=PARENT)
    init.add_argument('--trace', type=Path, default=TRACE)
    init.add_argument('--root', type=Path, default=ROOT)
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
            result = create(args.parent, args.root, args.trace)
        elif args.command == 'worker':
            if args.root.resolve() not in args.result.resolve().parents:
                raise ValueError('New worker result must remain inside the NEW overlay')
            result = worker(args.root, args.deployment, args.phase, args.max_jobs, args.device, args.release, args.binding)
            base.write(args.result, result)
        elif args.command == 'status':
            result = status(args.root)
        else:
            result = export(args.root, args.output)
    except base.NotReady as error:
        print(json.dumps(dict(status='NotReady', first_error=str(error), automatic_retry=False)), flush=True)
        return 3
    print(json.dumps(result, sort_keys=True), flush=True)
    return 2 if result.get('status') == 'unavailable' else 0


if __name__ == '__main__':
    raise SystemExit(main())

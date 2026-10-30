"""Output-only recovery overlay; never modify the live frozen collection tree."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import inspect
import json
from pathlib import Path
import shlex
import subprocess
import sys
import uuid

import numpy as np

from . import r121_collection as base

VERSION = 'r121_native_output_serialization_recovery_v1_20261030'
FROZEN_SHA = '876cff5800f0c0a1067b7a36eee3e154b4b73dedb8030873a69dd7cefec49674'
TARGETS = ('ATSP/UNICO_MatPOENet', 'TSP/OMNI')
ROOT = base.data.ROOT / 'collection_serialization_recovery_v1'
ORIGINAL_SOLVE = base.solve_native


def normalize_output(cost, tour):
    """Move native output only; no arithmetic, search, seed or route changes."""
    plain_tour = base.data.r114.plain(tour)
    scalar = np.asarray(base.data.r114.plain(cost))
    if scalar.size != 1:
        raise ValueError('Native singleton output must have one cost')
    return float(scalar.reshape(-1)[0]), plain_tour


def recovered_solver(recovery_sha256):
    # Execute the pinned helper with exactly two output-boundary substitutions.
    # Its native input, adapter, seeds, lazy lifecycle and budget stay verbatim.
    source = inspect.getsource(ORIGINAL_SOLVE)
    before = "    audit = data.check_tour(task['problem'], payload['item'], solved.tours[0], solved.costs[0])"
    after = "    cost, tour = normalize_output(solved.costs[0], solved.tours[0])\n    audit = data.check_tour(task['problem'], payload['item'], tour, cost)"
    result_before = "cost=float(solved.costs[0]), tour=data.r114.plain(solved.tours[0]), route_audit=audit,"
    result_after = "cost=cost, tour=tour, route_audit=audit,"
    if source.count(before) != 1 or source.count(result_before) != 1:
        raise ValueError('Frozen helper boundary changed; no speculative patch')
    namespace = dict(ORIGINAL_SOLVE.__globals__, normalize_output=normalize_output)
    exec(compile(source.replace(before, after).replace(result_before, result_after),
        str(Path(__file__).resolve()) + ':frozen_helper_output_boundary', 'exec'), namespace)
    compiled = namespace['solve_native']

    def solve(session, manifest, task):
        row = compiled(session, manifest, task)
        row.update(output_serialization_version=VERSION, output_serialization_recovery_sha256=recovery_sha256,
            solver_model_seed_profile_changes=False)
        return row

    return solve


def load(root):
    root = base.data.owned(root)
    recovery = base.read(root / 'recovery.json')
    base.data.verify_seal(recovery, 'recovery_sha256')
    for path, digest in recovery['source_sha256'].items():
        if base.sha(path) != digest:
            raise ValueError('Pinned output recovery/helper changed: ' + path)
    parent = Path(recovery['parent_root'])
    if base.sha(parent / 'protocol.json') != recovery['parent_protocol_file_sha256']:
        raise ValueError('Parent protocol changed')
    protocol, manifest, inventory = base.load(parent)
    if base.read(root / 'protocol.json') != protocol:
        raise ValueError('Overlay must retain the exact native parent protocol')
    for failure in recovery['previous_failures'].values():
        if base.sha(failure['path']) != failure['sha256']:
            raise ValueError('Preserved parent first-error evidence changed')
    return recovery, protocol, manifest, inventory


def create(parent, root=ROOT):
    parent, root = base.data.owned(parent), base.data.owned(root)
    if root == parent or parent in root.parents:
        raise ValueError('Recovery must be a NEW sibling tree, never write under the live parent')
    if base.sha(base.__file__) != FROZEN_SHA:
        raise ValueError('Only the user-frozen876 collector is eligible')
    protocol, manifest, inventory = base.load(parent)
    tasks = {t['task_id']: t for t in manifest['tasks']}
    failures, qualifications = {}, {}
    for key in TARGETS:
        path = parent / 'unavailable' / (base.slug(key) + '.json')
        row = base.read(path)
        if (row.get('stage') != 'preflight' or row.get('status') != 'unavailable'
                or row.get('error_type') != 'TypeError' or 'cuda:0' not in row.get('first_error', '')
                or 'numpy' not in row['first_error'] or row['collection_sha256'] != protocol['collection_sha256']
                or row['task_id'] not in tasks or any(row[k] != tasks[row['task_id']][k]
                    for k in ('input_sha256', 'solver_seed', 'deployment_id', 'global_id'))):
            raise ValueError('Not the authorized output-conversion first failure: ' + key)
        failures[key] = dict(path=str(path), sha256=base.sha(path), task_id=row['task_id'],
            binding_sha256=row['binding_sha256'], first_error=row['first_error'])
    for path in sorted((parent / 'preflight').glob('*.json')):
        row = base.read(path)
        key = row['problem'] + '/' + row['method']
        base.validate_receipt(row, tasks[row['task_id']], protocol, 'preflight')
        if row['profile_sha256'] != inventory['deployments'][key]['profile_sha256']:
            raise ValueError('Existing qualification profile changed')
        binding = base.read(parent / 'deployment_bindings' / (base.slug(key) + '.json'))
        if binding['binding_sha256'] != row['binding_sha256']:
            raise ValueError('Existing qualification physical binding changed')
        qualifications[key] = dict(path=str(path), sha256=base.sha(path), binding_sha256=row['binding_sha256'])
    from . import r114_group_worker
    sources = {str(Path(path).resolve()): base.sha(path) for path in
        (__file__, base.__file__, base.data.__file__, base.data.r114.__file__, r114_group_worker.__file__)}
    recovery = base.data.seal(dict(version=VERSION, parent_root=str(parent),
        parent_collection_sha256=protocol['collection_sha256'],
        parent_protocol_file_sha256=base.sha(parent / 'protocol.json'),
        allowed_recovery_deployments=list(TARGETS), previous_failures=failures,
        reused_qualification_refs=qualifications, source_sha256=sources,
        change='cost scalar and tour moved CPU/plain BEFORE independent audit; output conversion only',
        solver_model_seed_profile_changes=False, parent_tree_writes=False,
        native_candidate_pool_unchanged=True, production_export='complete parent+overlay union only',
        solver_execution=False), 'recovery_sha256')
    base.write(root / 'protocol.json', protocol)
    base.write(root / 'recovery.json', recovery)
    return dict(root=str(root), recovery_sha256=recovery['recovery_sha256'],
        reused_successful_qualifications=len(qualifications), pending_recovery_deployments=list(TARGETS),
        parent_labels='remain live in parent; read at complete federated export', solver_execution=False)


def worker(root, key, phase, max_jobs, device, release, binding):
    recovery, protocol, manifest, inventory = load(root)
    if key not in TARGETS:
        raise ValueError('Overlay executes ONLY the two authorized failed deployments; use frozen parent for others')
    release_data = base.data.require_release(release, protocol['plan_sha256'])
    actual_binding = base.data.require_binding(binding, release_data, device)
    if actual_binding != recovery['previous_failures'][key]['binding_sha256']:
        raise ValueError('Recovery must use the original failure physical binding')
    if phase == 'preflight':
        originals = {e['endpoint_id'] for e in manifest['endpoints'] if e['cell'] == 'original'}
        task = next(t for t in base.task_groups(manifest)[key] if t['endpoint_id'] in originals)
        if task['task_id'] != recovery['previous_failures'][key]['task_id']:
            raise ValueError('Recovery must rerun exactly the failed original once')
    base.solve_native = recovered_solver(recovery['recovery_sha256'])
    result = base.worker(root, key, phase, max_jobs, device, release, binding)
    return dict(result, output_serialization_version=VERSION, recovery_sha256=recovery['recovery_sha256'])


def wave(root, phase, lane, max_jobs, execute=False, device=None, release=None, binding=None):
    if max_jobs < 1:
        raise ValueError('Finite positive wave required')
    recovery, protocol, manifest, _ = load(root)
    keys = [k for k in base.lane_keys(base.task_groups(manifest), lane, protocol['lanes']) if k in TARGETS]
    if not execute:
        return dict(solver_execution=False, assigned_recovery_deployments=keys, max_jobs=max_jobs,
            old_successful_qualifications_reused=len(recovery['reused_qualification_refs']))
    results, count = [], 0
    for key in keys:
        if count >= max_jobs:
            break
        if phase == 'preflight' and (Path(root) / 'preflight' / (base.slug(key) + '.json')).exists():
            continue
        if (Path(root) / 'unavailable' / (base.slug(key) + '.json')).exists():
            raise base.NotReady('Recovery has its own terminal first failure; no automatic retry')
        budget = 1 if phase == 'preflight' else max_jobs - count
        token = uuid.uuid4().hex
        result_path = Path(root) / 'worker_receipts' / (token + '.json')
        log = Path(root) / 'logs' / (token + '.log')
        log.parent.mkdir(parents=True, exist_ok=True)
        command = [sys.executable, '-B', '-m', 'code.V4.r121_collection_recovery', 'worker',
            '--root', str(root), '--deployment', key, '--phase', phase, '--max-jobs', str(budget),
            '--device', str(device), '--release', str(release), '--binding', str(binding), '--result', str(result_path)]
        shell = 'source /public/home/shiys/miniconda3/etc/profile.d/conda.sh && conda activate easynco && exec ' + shlex.join(command)
        with log.open('x') as output:
            completed = subprocess.run(['bash', '-c', shell], cwd=str(Path(__file__).resolve().parents[2]),
                stdout=output, stderr=subprocess.STDOUT)
        if not result_path.exists():
            raise base.NotReady('Recovery child first fatal; inspect ' + str(log))
        row = base.read(result_path)
        results.append(dict(row, log=str(log), rc=completed.returncode))
        count += row.get('attempted', 0)
        if completed.returncode:
            break
    result = dict(recovery_sha256=recovery['recovery_sha256'], phase=phase, lane=lane,
        actual_tasks_attempted=count, max_jobs=max_jobs, deployments=results,
        status='blocked' if any(r['rc'] for r in results) else 'wave_complete')
    base.write(Path(root) / 'waves' / (uuid.uuid4().hex + '.json'), result)
    return result


def status(root):
    recovery, protocol, manifest, _ = load(root)
    parent = Path(recovery['parent_root'])
    rows = []
    for key, tasks in base.task_groups(manifest).items():
        actual_root = Path(root) if key in TARGETS else parent
        state = base.progress(actual_root, key)
        rows.append(dict(deployment=key, root=str(actual_root), planned=len(tasks),
            receipt_progress=state['completed'],
            preflight=(actual_root / 'preflight' / (base.slug(key) + '.json')).exists(),
            unavailable=(actual_root / 'unavailable' / (base.slug(key) + '.json')).exists()))
    return dict(recovery_sha256=recovery['recovery_sha256'], deployments=rows,
        planned_labels=len(manifest['tasks']), receipt_progress=sum(r['receipt_progress'] for r in rows),
        all_labels_complete=all(r['receipt_progress'] == r['planned'] and not r['unavailable'] for r in rows))


def export(root, output=None):
    recovery, protocol, manifest, _ = load(root)
    if not status(root)['all_labels_complete']:
        raise base.NotReady('All18/89 parent+overlay native labels required; partial export prohibited')
    root, parent = Path(root), Path(recovery['parent_root'])
    output = base.data.owned(output or root / 'family_labels.jsonl')
    if output.exists():
        raise ValueError('Preserve the existing complete export')
    by_endpoint = defaultdict(list)
    for task in manifest['tasks']:
        by_endpoint[task['endpoint_id']].append(task)
    temp = output.with_name('.' + output.name + '.' + uuid.uuid4().hex)
    digest, count = hashlib.sha256(), 0
    try:
        with temp.open('xb') as stream:
            for endpoint in manifest['endpoints']:
                records = []
                item = base.read(endpoint['input_path'])['item']
                for task in by_endpoint[endpoint['endpoint_id']]:
                    key = task['problem'] + '/' + task['method']
                    actual_root = root if key in TARGETS else parent
                    row = base.read(actual_root / 'labels' / (task['task_id'] + '.json'))
                    base.validate_receipt(row, task, protocol)
                    if key in TARGETS and row.get('output_serialization_recovery_sha256') != recovery['recovery_sha256']:
                        raise ValueError('Recovered native label lacks pinned output-layer provenance')
                    base.data.check_tour(task['problem'], item, row['tour'], row['cost'])
                    records.append(row)
                targets = base.data.full_pool_targets(manifest['pools'][endpoint['problem']], records)
                encoded = (json.dumps(dict(endpoint=endpoint, targets=targets,
                    collection_sha256=protocol['collection_sha256']), sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()
                stream.write(encoded)
                digest.update(encoded)
                count += 1
        # Immutable output, without replacing an existing export.
        import os
        os.link(temp, output)
    finally:
        temp.unlink(missing_ok=True)
    result = dict(complete=True, version=base.data.VERSION, full18=True, candidate_pool_unchanged=True,
        historical_labels_replaced=False, path=str(output), sha256=digest.hexdigest(), endpoints=count,
        labels=len(manifest['tasks']), collection_sha256=protocol['collection_sha256'],
        family_input_manifest=protocol['input_manifest'], manifest_sha256=manifest['manifest_sha256'],
        output_serialization_version=VERSION, output_serialization_recovery_sha256=recovery['recovery_sha256'],
        native_receipt_roots=dict(parent=str(parent), recovered=str(root)))
    base.write(output.with_suffix('.receipt.json'), result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    init = commands.add_parser('create')
    init.add_argument('--parent', type=Path, required=True)
    init.add_argument('--root', type=Path, default=ROOT)
    for name in ('worker', 'wave'):
        sub = commands.add_parser(name)
        sub.add_argument('--root', type=Path, default=ROOT)
        sub.add_argument('--phase', choices=('preflight', 'label'), required=True)
        sub.add_argument('--max-jobs', type=int, required=True)
        sub.add_argument('--device')
        sub.add_argument('--release', type=Path)
        sub.add_argument('--binding', type=Path)
        if name == 'worker':
            sub.add_argument('--deployment', required=True)
            sub.add_argument('--result', type=Path, required=True)
        else:
            sub.add_argument('--lane', type=int, required=True)
            sub.add_argument('--execute', action='store_true')
    for name in ('status', 'export'):
        sub = commands.add_parser(name)
        sub.add_argument('--root', type=Path, default=ROOT)
        if name == 'export':
            sub.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == 'create':
            result = create(args.parent, args.root)
        elif args.command == 'worker':
            if args.root.resolve() not in args.result.resolve().parents:
                raise ValueError('Worker result must stay in the NEW overlay tree')
            result = worker(args.root, args.deployment, args.phase, args.max_jobs, args.device, args.release, args.binding)
            base.write(args.result, result)
        elif args.command == 'wave':
            result = wave(args.root, args.phase, args.lane, args.max_jobs, args.execute, args.device, args.release, args.binding)
        elif args.command == 'status':
            result = status(args.root)
        else:
            result = export(args.root, args.output)
    except base.NotReady as error:
        print(json.dumps(dict(status='NotReady', first_error=str(error), automatic_retry=False)), flush=True)
        return 3
    print(json.dumps(result, sort_keys=True), flush=True)
    return 2 if result.get('status') in ('blocked', 'unavailable') else 0


if __name__ == '__main__':
    raise SystemExit(main())

"""Finite, main-bound native full18 collection. All solver imports are lazy."""

from __future__ import annotations

import argparse
from collections import defaultdict
import gc
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
import uuid

from . import r121_structural_data as data

ROOT = data.ROOT / 'collection_v1'
COLLECTION_VERSION = 'r121_full18_collection_v1_20261030'
MODES = ('cold_native', 'cached_native_v1')


class NotReady(RuntimeError):
    pass


def read(path):
    return data.r114.read_json(path)


def sha(path):
    return data.r114.file_hash(path)


def write(path, value):
    data.r114.write_new_json(data.owned(path), value)


def atomic(path, value):
    path = data.owned(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name('.' + path.name + '.' + uuid.uuid4().hex)
    with temp.open('x') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    temp.replace(path)


def slug(deployment):
    return deployment.replace('/', '__')


def task_groups(manifest):
    groups = defaultdict(list)
    for task in manifest['tasks']:
        groups[task['problem'] + '/' + task['method']].append(task)
    return {key: sorted(rows, key=lambda r: r['task_id']) for key, rows in sorted(groups.items())}


def lane_keys(groups, lane, lanes):
    if lanes < 1 or not 0 <= lane < lanes:
        raise ValueError('Invalid fixed collection lane')
    return [key for index, key in enumerate(sorted(groups)) if index % lanes == lane]


def create(inputs, root=ROOT, lanes=4, mode='cold_native'):
    root = data.owned(root)
    if mode not in MODES or lanes < 1:
        raise ValueError('Unknown execution-layer version or lane count')
    if not Path(inputs).is_file():
        raise NotReady('Full18 CPU materialization has not produced its manifest yet')
    manifest = read(inputs)
    data.verify_seal(manifest, 'manifest_sha256')
    plan, selection, inventory, generation = data.load_plan(manifest['plan_path'])
    if (manifest['plan_sha256'] != plan['plan_sha256'] or not manifest['full18_prepared']
            or not manifest['full_planned_population_prepared']
            or len(manifest['tasks']) != plan['planned_label_tasks']
            or len(manifest['endpoints']) != plan['planned_endpoints']):
        raise ValueError('Collection requires the ENTIRE frozen full18 population, not a smoke prefix')
    groups = task_groups(manifest)
    expected = {p + '/' + c['method'] for p, pool in selection['pools'].items() for c in pool}
    if set(groups) != expected or len(groups) != 89:
        raise ValueError('Frozen89 legal deployments must all be retained')
    sources = {str(Path(__file__).resolve()): sha(__file__), **generation['source_sha256']}
    payload = dict(schema_version=1, collection_version=COLLECTION_VERSION,
        generation_version=data.VERSION, input_manifest=str(Path(inputs).resolve()),
        input_manifest_sha256=sha(inputs), manifest_sha256=manifest['manifest_sha256'],
        plan_path=manifest['plan_path'], plan_sha256=plan['plan_sha256'],
        inventory_sha256=inventory['inventory_sha256'], mode=mode, lanes=lanes,
        group_order=sorted(groups), planned_labels=len(manifest['tasks']),
        source_sha256=sources, batch_size=1, native_budget_changes=False,
        deployment_seed_changes=False, old_label_replacement=False,
        execution_layer_change=('none: fresh backend per task' if mode == 'cold_native' else
            'explicit new persistent backend protocol; not asserted fresh/warm or historical equivalence'),
        solver_execution=False, R120_release_required=True)
    payload = data.seal(payload, 'collection_sha256')
    write(root / 'protocol.json', payload)
    return payload


def load(root):
    root = data.owned(root)
    protocol = read(root / 'protocol.json')
    data.verify_seal(protocol, 'collection_sha256')
    if sha(protocol['input_manifest']) != protocol['input_manifest_sha256']:
        raise ValueError('Full18 input manifest changed')
    for path, digest in protocol['source_sha256'].items():
        if sha(path) != digest:
            raise ValueError('Pinned collector/generator changed: ' + path)
    manifest = read(protocol['input_manifest'])
    data.verify_seal(manifest, 'manifest_sha256')
    plan, selection, inventory, _ = data.load_plan(protocol['plan_path'])
    if (plan['plan_sha256'] != protocol['plan_sha256']
            or manifest['manifest_sha256'] != protocol['manifest_sha256']
            or inventory['inventory_sha256'] != protocol['inventory_sha256']):
        raise ValueError('Collection plan/manifest/deployment identity mismatch')
    return protocol, manifest, inventory


def progress(root, key):
    path = root / 'progress' / (slug(key) + '.json')
    return read(path) if path.exists() else dict(next_index=0, completed=0, status='pending')


def validate_receipt(row, task, protocol, stage='label'):
    keys = ('task_id', 'endpoint_id', 'base_id', 'problem', 'method', 'global_id',
            'input_sha256', 'solver_seed', 'deployment_id')
    if (any(row.get(k) != task.get(k) for k in keys) or row.get('stage') != stage
            or row.get('version') != data.VERSION or row.get('status') != 'ok'
            or row.get('collection_sha256') != protocol['collection_sha256']
            or row.get('route_audit', {}).get('verified') is not True
            or row.get('source') != 'fresh_solver'):
        raise ValueError('Receipt does not bind this successful native task/protocol')
    cost = row.get('cost')
    if not isinstance(cost, (int, float)) or not 0 < cost < float('inf'):
        raise ValueError('Missing/nonpositive/nonfinite routing cost')


def _session(manifest, inventory, key, device):
    from .r114_group_worker import NativeSession
    deployment = inventory['deployments'][key]
    for path, digest in deployment['expected_artifacts'].items():
        if sha(path) != digest:
            raise ValueError('Recorded source/weight unavailable or changed: ' + path)
    lock = dict(scripts_root=deployment['scripts_root'], overlay_root=deployment['overlay_root'],
        legacy_root=deployment['legacy_root'], lock_sha256=inventory['inventory_sha256'],
        deployments={key: deployment})
    problem, method = key.split('/', 1)
    return NativeSession(manifest, lock, problem, method, device)


def solve_native(session, manifest, task):
    from .r114_group_worker import reset_lazy_runner, reset_rng
    endpoint = next(e for e in manifest['endpoints'] if e['endpoint_id'] == task['endpoint_id'])
    if sha(endpoint['input_path']) != task['input_sha256']:
        raise ValueError('Native endpoint file changed')
    payload = read(endpoint['input_path'])
    data.validate_input(task['problem'], payload['item'])
    if task['problem'] not in ('TSP', 'ATSP'):
        return session.solve(task['task_id'])
    torch = session.torch
    native = torch.tensor(payload['item'], dtype=torch.float32)
    reset_lazy_runner(session.backend, task['method'])
    reset_rng(torch, task['solver_seed'], session.device)
    if session.device.startswith('cuda'):
        torch.cuda.synchronize()
    start = time.monotonic()
    solved = session.backend.run([native])
    if session.device.startswith('cuda'):
        torch.cuda.synchronize()
    if len(solved.costs) != 1 or len(solved.tours) != 1:
        raise ValueError('Actual native adapter returned no singleton cost/tour')
    audit = data.check_tour(task['problem'], payload['item'], solved.tours[0], solved.costs[0])
    return dict(task, version=data.VERSION, status='ok', source='fresh_solver',
        cost=float(solved.costs[0]), tour=data.r114.plain(solved.tours[0]), route_audit=audit,
        profile_sha256=session.profile_sha256, backend_audit=data.r114.plain(solved.audit),
        solve_seconds=time.monotonic() - start, batch_size=1)


def worker(root, key, phase, max_jobs, device, release, binding):
    if max_jobs < 1 or phase not in ('preflight', 'label'):
        raise ValueError('A positive finite worker budget and explicit phase are required')
    root = data.owned(root)
    protocol, manifest, inventory = load(root)
    release_data = data.require_release(release, protocol['plan_sha256'])
    binding_sha = data.require_binding(binding, release_data, device)
    groups = task_groups(manifest)
    if key not in groups:
        raise ValueError('Unknown legal deployment')
    deployment = inventory['deployments'][key]
    binding_path = root / 'deployment_bindings' / (slug(key) + '.json')
    write(binding_path, dict(deployment=key, binding_sha256=binding_sha,
        collection_sha256=protocol['collection_sha256']))
    unavailable = root / 'unavailable' / (slug(key) + '.json')
    if unavailable.exists():
        raise NotReady('Terminal first error already recorded; no automatic retry: ' + key)
    if deployment['status'].startswith('blocked'):
        raise NotReady('Recorded deployment recipe is blocked: ' + key)
    preflight_path = root / 'preflight' / (slug(key) + '.json')
    group = groups[key]
    if phase == 'preflight':
        originals = {e['endpoint_id'] for e in manifest['endpoints'] if e['cell'] == 'original'}
        selected = [next(t for t in group if t['endpoint_id'] in originals)]
        if preflight_path.exists():
            validate_receipt(read(preflight_path), selected[0], protocol, 'preflight')
            if read(preflight_path)['binding_sha256'] != binding_sha:
                raise ValueError('Preflight physical binding changed')
            return dict(completed=0, reused=1, deployment=key, phase=phase, status='already_complete')
    else:
        if not preflight_path.exists():
            raise NotReady('Actual saved preflight is required before labels: ' + key)
        q = read(preflight_path)
        if (q['status'] != 'ok' or q['collection_sha256'] != protocol['collection_sha256']
                or q['binding_sha256'] != binding_sha or q['profile_sha256'] != deployment['profile_sha256']):
            raise ValueError('Preflight source/profile/physical binding mismatch')
        state = progress(root, key)
        if state.get('status') == 'unavailable':
            raise NotReady('Deployment has a recorded first error')
        selected = group[state['next_index']:state['next_index'] + max_jobs]
    session, completed, reused, attempted = None, 0, 0, 0
    started = time.monotonic()
    for task in selected:
        attempted += 1
        path = preflight_path if phase == 'preflight' else root / 'labels' / (task['task_id'] + '.json')
        if path.exists():
            row = read(path)
            validate_receipt(row, task, protocol, phase)
            if row['binding_sha256'] != binding_sha:
                raise ValueError('Completed label belongs to another physical binding')
            reused += 1
        else:
            begin = time.monotonic()
            try:
                if session is None:
                    session = _session(manifest, inventory, key, device)
                row = solve_native(session, manifest, task)
                row.update(stage=phase, device=device, binding_sha256=binding_sha,
                    collection_version=COLLECTION_VERSION, collection_sha256=protocol['collection_sha256'],
                    execution_mode=protocol['mode'], version=data.VERSION,
                    deployment_id=task['deployment_id'], global_id=task['global_id'],
                    model_seed=data.r114.seed_for(2, task['problem'], task['method'], 'model'),
                    historical_scalar_replay=False, wall_seconds_including_load=time.monotonic() - begin)
                validate_receipt(row, task, protocol, phase)
                write(path, row)
                completed += 1
            except Exception as error:
                failure = dict(task, stage=phase, status='unavailable', first_error=str(error),
                    error_type=type(error).__name__, collection_sha256=protocol['collection_sha256'],
                    binding_sha256=binding_sha, automatic_retry=False, candidate_pool_unchanged=True)
                write(unavailable, failure)
                atomic(root / 'progress' / (slug(key) + '.json'),
                    dict(progress(root, key), status='unavailable', first_error_path=str(unavailable)))
                return dict(status='unavailable', deployment=key, completed=completed,
                    reused=reused, attempted=attempted, first_error=failure,
                    elapsed_s=time.monotonic() - started)
            if protocol['mode'] == 'cold_native':
                del session
                session = None
                gc.collect()
        if phase == 'label':
            state = progress(root, key)
            state.update(next_index=state['next_index'] + 1, completed=state['completed'] + 1,
                status='complete' if state['next_index'] + 1 == len(group) else 'running',
                collection_sha256=protocol['collection_sha256'], last_receipt=str(path), last_receipt_sha256=sha(path))
            atomic(root / 'progress' / (slug(key) + '.json'), state)
        print(json.dumps(dict(event='native_task_complete', phase=phase, deployment=key,
            task_id=task['task_id'], completed=completed, reused=reused)), flush=True)
    return dict(status='finite_budget_complete', deployment=key, phase=phase,
        completed=completed, reused=reused, attempted=attempted, elapsed_s=time.monotonic() - started)


def wave(root, phase, lane, max_jobs, execute=False, device=None, release=None, binding=None):
    if max_jobs < 1:
        raise ValueError('A wave must have a finite positive --max-jobs')
    root = data.owned(root)
    protocol, manifest, _ = load(root)
    groups = task_groups(manifest)
    keys = lane_keys(groups, lane, protocol['lanes'])
    if not execute:
        return dict(solver_execution=False, phase=phase, lane=lane, max_jobs=max_jobs,
            assigned_deployments=keys, required='explicit execute/device/main R120 release/physical binding')
    release_data = data.require_release(release, protocol['plan_sha256'])
    binding_sha = data.require_binding(binding, release_data, device)
    write(root / 'lanes' / f'{lane}.json', dict(lane=lane, binding_sha256=binding_sha,
        collection_sha256=protocol['collection_sha256']))
    summaries, count = [], 0
    for key in keys:
        if count >= max_jobs:
            break
        if (root / 'unavailable' / (slug(key) + '.json')).exists():
            summaries.append(dict(deployment=key, status='unavailable', automatic_retry=False))
            continue
        if phase == 'preflight' and (root / 'preflight' / (slug(key) + '.json')).exists():
            continue
        if phase == 'label':
            if not (root / 'preflight' / (slug(key) + '.json')).exists():
                summaries.append(dict(deployment=key, status='preflight_pending'))
                continue
            if progress(root, key)['next_index'] >= len(groups[key]):
                continue
        budget = 1 if phase == 'preflight' else min(max_jobs - count,
            len(groups[key]) - progress(root, key)['next_index'])
        token = uuid.uuid4().hex
        result_path = root / 'worker_receipts' / (token + '.json')
        log = root / 'logs' / (token + '.log')
        log.parent.mkdir(parents=True, exist_ok=True)
        command = [sys.executable, '-B', '-m', 'code.V4.r121_collection', 'worker', '--root', str(root),
            '--deployment', key, '--phase', phase, '--max-jobs', str(budget), '--device', device,
            '--release', str(release), '--binding', str(binding), '--result', str(result_path)]
        shell = 'source /public/home/shiys/miniconda3/etc/profile.d/conda.sh && conda activate easynco && exec ' + shlex.join(command)
        with log.open('x') as output:
            result = subprocess.run(['bash', '-c', shell], cwd=str(Path(__file__).resolve().parents[2]),
                env=dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
                    NUMEXPR_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1'), stdout=output, stderr=subprocess.STDOUT)
        if not result_path.exists():
            summaries.append(dict(deployment=key, status='worker_failed', first_error_log=str(log), rc=result.returncode))
            break
        record = read(result_path)
        summaries.append(dict(record, log=str(log), rc=result.returncode))
        consumed = record.get('attempted', record.get('completed', 0) + record.get('reused', 0))
        if not isinstance(consumed, int) or not 0 <= consumed <= budget:
            raise ValueError('Child exceeded its finite assigned job budget')
        count += consumed
        if result.returncode and record.get('status') != 'unavailable':
            break
    summary = dict(collection_sha256=protocol['collection_sha256'], phase=phase, lane=lane,
        binding_sha256=binding_sha, max_jobs=max_jobs, actual_tasks_attempted=count,
        finite_wave=True, all18_labels_complete=False, deployments=summaries,
        status='blocked' if any(x['status'] in ('unavailable', 'worker_failed') for x in summaries) else 'wave_complete')
    write(root / 'waves' / (phase + '_' + str(lane) + '_' + uuid.uuid4().hex + '.json'), summary)
    return summary


def status(root):
    root = data.owned(root)
    protocol, manifest, _ = load(root)
    groups = task_groups(manifest)
    rows = []
    for key, tasks in groups.items():
        state = progress(root, key)
        rows.append(dict(deployment=key, planned=len(tasks), receipt_progress=state['completed'],
            preflight=(root / 'preflight' / (slug(key) + '.json')).exists(),
            unavailable=(root / 'unavailable' / (slug(key) + '.json')).exists(), status=state['status']))
    return dict(collection_sha256=protocol['collection_sha256'], deployments=rows,
        planned_labels=len(manifest['tasks']), receipt_progress=sum(x['receipt_progress'] for x in rows),
        all_labels_complete=all(x['receipt_progress'] == x['planned'] and not x['unavailable'] for x in rows),
        interpretation='saved receipt progress only; export independently validates all native certificates')


def export(root, output=None):
    root = data.owned(root)
    protocol, manifest, _ = load(root)
    if not status(root)['all_labels_complete']:
        raise NotReady('All18/89 complete native labels are required; no partial ready family pool')
    output = data.owned(output or root / 'family_labels.jsonl')
    if output.exists():
        raise ValueError('Preserve existing export; choose a new path')
    by_endpoint = defaultdict(list)
    for task in manifest['tasks']:
        by_endpoint[task['endpoint_id']].append(task)
    temporary = output.with_name('.' + output.name + '.' + uuid.uuid4().hex)
    digest, count = hashlib.sha256(), 0
    try:
        with temporary.open('xb') as stream:
            for endpoint in manifest['endpoints']:
                records = []
                for task in by_endpoint[endpoint['endpoint_id']]:
                    path = root / 'labels' / (task['task_id'] + '.json')
                    row = read(path)
                    validate_receipt(row, task, protocol)
                    item = read(endpoint['input_path'])['item']
                    data.check_tour(task['problem'], item, row['tour'], row['cost'])
                    records.append(row)
                targets = data.full_pool_targets(manifest['pools'][endpoint['problem']], records)
                row = dict(endpoint=endpoint, targets=targets, collection_sha256=protocol['collection_sha256'])
                encoded = (json.dumps(row, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()
                stream.write(encoded)
                digest.update(encoded)
                count += 1
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    receipt = dict(complete=True, version=data.VERSION, collection_sha256=protocol['collection_sha256'],
        path=str(output), sha256=digest.hexdigest(), endpoints=count, labels=len(manifest['tasks']),
        full18=True, candidate_pool_unchanged=True, historical_labels_replaced=False,
        family_input_manifest=protocol['input_manifest'], manifest_sha256=manifest['manifest_sha256'])
    write(output.with_suffix('.receipt.json'), receipt)
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    init = commands.add_parser('create')
    init.add_argument('--inputs', type=Path, required=True)
    init.add_argument('--root', type=Path, default=ROOT)
    init.add_argument('--lanes', type=int, default=4)
    init.add_argument('--mode', choices=MODES, default='cold_native')
    for name in ('wave', 'worker'):
        sub = commands.add_parser(name)
        sub.add_argument('--root', type=Path, default=ROOT)
        sub.add_argument('--phase', choices=('preflight', 'label'), required=True)
        sub.add_argument('--max-jobs', type=int, required=True)
        sub.add_argument('--device')
        sub.add_argument('--release', type=Path)
        sub.add_argument('--binding', type=Path)
        if name == 'wave':
            sub.add_argument('--lane', type=int, required=True)
            sub.add_argument('--execute', action='store_true')
        else:
            sub.add_argument('--deployment', required=True)
            sub.add_argument('--result', type=Path, required=True)
    for name in ('status', 'export'):
        sub = commands.add_parser(name)
        sub.add_argument('--root', type=Path, default=ROOT)
        if name == 'export':
            sub.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == 'create':
            result = create(args.inputs, args.root, args.lanes, args.mode)
        elif args.command == 'wave':
            result = wave(args.root, args.phase, args.lane, args.max_jobs, args.execute,
                args.device, args.release, args.binding)
        elif args.command == 'worker':
            result = worker(args.root, args.deployment, args.phase, args.max_jobs,
                args.device, args.release, args.binding)
            write(args.result, result)
        elif args.command == 'status':
            result = status(args.root)
        else:
            result = export(args.root, args.output)
    except NotReady as error:
        print(json.dumps(dict(status='NotReady', first_error=str(error), automatic_retry=False)), flush=True)
        return 3
    print(json.dumps(result, sort_keys=True), flush=True)
    return 2 if result.get('status') in ('unavailable', 'blocked') else 0


if __name__ == '__main__':
    raise SystemExit(main())

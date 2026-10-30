"""Finite, source-fixed R121 resume with an independently bounded L audit."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import contextmanager
import errno
from functools import lru_cache
import hashlib
import inspect
import json
import os
from pathlib import Path
import traceback
import uuid

import numpy as np

from . import r121_collection as base
from . import r121_input_boundary as boundary
from . import r121_universal_collection as universal

VERSION = 'r121_numeric_boundary_collection_v3_20261030'
CHECK_VERSION = 'r121_independent_L_fp32_plus_one_ulp_v3_20261030'
ROOT = base.data.ROOT / 'collection_numeric_boundary_v3'
PARENT, OVERLAY, UNIVERSAL = boundary.PARENT, boundary.ROOT, universal.ROOT
DIAGNOSTIC = base.data.ROOT / 'route_limit_diagnostic_v1/diagnostic.json'
ENV = base.data.r114.WORKSPACE / 'reld-nco-main/Multi-Task/envs/VRPBLEnv.py'
EPS = 1e-5
STRICT_ROUTE = base.data.r114.check_route
ORIGINAL_SOLVE = base.solve_native
IDENTITY = ('task_id', 'endpoint_id', 'base_id', 'problem', 'method', 'global_id',
    'input_sha256', 'solver_seed', 'deployment_id')


def _boundary_evidence(problem, item, route, distance, left, right):
    fields = base.data.r114.fields(problem, item)
    xy = np.concatenate((fields['depot'][None], fields['xy'])).astype(np.float32)
    charged = route[:-1] if problem.startswith('OVRP') else route
    edges = np.linalg.norm(np.diff(xy[charged], axis=0), axis=1).astype(np.float32)
    prefix = np.float32(0.)
    for edge in edges:
        prefix = np.float32(prefix + edge)
    threshold = np.float32(np.float32(item['route_limit']) + np.float32(EPS))
    upper = np.nextafter(threshold, np.float32(np.inf), dtype=np.float32)
    if not (np.isfinite(distance) and distance <= float(upper) and prefix <= threshold):
        raise ValueError('Route-length limit violation')
    return dict(left_depot_position=left, right_depot_position=right,
        route=list(route), charged_edges=len(edges), raw_fp64_distance=float(distance),
        independent_fp32_sequential_distance=float(prefix), limit=float(item['route_limit']),
        epsilon=EPS, threshold_fp32=float(threshold), one_ulp32=float(upper - threshold),
        upper_fp64_one_ulp=float(upper), excess_over_original_strict=float(distance - item['route_limit'] - EPS),
        fp32_is_independent_cpu_emulation_not_a_native_cuda_state=True)


def numeric_check_route(problem, item, tour, cost):
    """Try the original complete audit first; replace only its L exception."""
    cases = []
    try:
        result = STRICT_ROUTE(problem, item, tour, cost)
    except ValueError as error:
        if str(error) != 'Route-length limit violation' or 'L' not in problem:
            raise

        def accept(p, original, route, distance, left, right):
            cases.append(_boundary_evidence(p, original, route, distance, left, right))

        source = inspect.getsource(STRICT_ROUTE)
        old = 'raise ValueError("Route-length limit violation")'
        if source.count(old) != 1:
            raise ValueError('The frozen independent L exception site changed')
        namespace = dict(STRICT_ROUTE.__globals__, _numeric_boundary=accept)
        source = source.replace(old, '_numeric_boundary(problem, item, route, distance, left, right)')
        exec(compile(source, str(Path(__file__).resolve()) + ':bounded_L_only', 'exec'), namespace)
        # The remainder of the original audit still checks later routes and total cost.
        result = namespace['check_route'](problem, item, tour, cost)
    result['independent_numeric_validation'] = dict(version=CHECK_VERSION,
        mode='bounded_L_boundary' if cases else 'original_strict', boundaries=cases,
        original_strict_first=True, other_constraints_and_cost_unchanged=True)
    return result


def _environment_pin():
    import ast
    tree = ast.parse(ENV.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'VRPBLEnv')
    step = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'step')
    values = [n.value.value for n in ast.walk(step) if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == 'round_error_epsilon' for t in n.targets)
        and isinstance(n.value, ast.Constant)]
    if values != [EPS] or 'self.length[self.at_the_depot] = 0' not in ENV.read_text():
        raise ValueError('Native environment EPS/reset differs; stop numerical recovery')
    return dict(path=str(ENV), sha256=base.sha(ENV), epsilon=EPS)


def source_root(recovery, key):
    return Path(recovery['source_root_by_deployment'][key])


def _diagnostic_proof(path, parent, protocol, manifest, inventory):
    path = Path(path).resolve()
    diagnostic = base.read(path)
    task = next((t for t in manifest['tasks'] if t['task_id'] == diagnostic.get('task', {}).get('task_id')), None)
    if task is None or any(task.get(k) != diagnostic['task'].get(k) for k in IDENTITY):
        raise ValueError('Diagnostic does not bind an exact original native task')
    key = task['problem'] + '/' + task['method']
    profile = inventory['deployments'][key]['profile_sha256']
    failure_path = parent / 'unavailable' / (base.slug(key) + '.json')
    failure = base.read(failure_path)
    binding = base.read(parent / 'deployment_bindings' / (base.slug(key) + '.json'))
    endpoint = next(e for e in manifest['endpoints'] if e['endpoint_id'] == task['endpoint_id'])
    if (key != 'VRPBL/reld_mt' or endpoint['cell'] != 'original'
            or diagnostic.get('profile_sha256') != profile
            or diagnostic.get('binding_sha256') != binding['binding_sha256']
            or diagnostic.get('original_collection_sha256') != protocol['collection_sha256']
            or diagnostic.get('solver_seed') != task['solver_seed']
            or diagnostic.get('diagnostic_only') is not True or diagnostic.get('not_a_training_label') is not True
            or diagnostic.get('backend_audit', {}).get('feasible') is not True
            or diagnostic.get('audit_error') != 'ValueError: Route-length limit violation'
            or failure.get('first_error') != 'Route-length limit violation'
            or failure.get('error_type') != 'ValueError' or failure.get('stage') != 'label'
            or failure.get('status') != 'unavailable'
            or binding.get('deployment') != key or binding.get('collection_sha256') != protocol['collection_sha256']
            or any(failure.get(k) != task.get(k) for k in IDENTITY)
            or failure.get('binding_sha256') != diagnostic['binding_sha256']
            or failure.get('collection_sha256') != protocol['collection_sha256']
            or base.sha(endpoint['input_path']) != task['input_sha256']
            or Path(diagnostic['input_path']).resolve() != Path(endpoint['input_path']).resolve()):
        raise ValueError('Diagnostic profile/input/binding/failure differs; stop numerical recovery')
    for name, digest in diagnostic['source_sha256'].items():
        if base.sha(Path(__file__).parent / name) != digest:
            raise ValueError('Diagnostic source differs: ' + name)
    audit = numeric_check_route(task['problem'], base.read(endpoint['input_path'])['item'],
        diagnostic['raw_tour'], diagnostic['raw_cost'])
    if not audit['independent_numeric_validation']['boundaries']:
        raise ValueError('Diagnostic is not the claimed numeric L boundary')
    return dict(deployment=key, task_id=task['task_id'], task_identity={k: task.get(k) for k in IDENTITY},
        diagnostic_path=str(path), diagnostic_sha256=base.sha(path), failure_path=str(failure_path),
        failure_sha256=base.sha(failure_path), binding_sha256=diagnostic['binding_sha256'],
        profile_sha256=profile, validation=audit['independent_numeric_validation'], diagnostic_promoted_to_label=False)


def create(root=ROOT, parent=PARENT, overlay=OVERLAY, universal_root=UNIVERSAL, diagnostic=DIAGNOSTIC):
    root, parent, overlay, universal_root = (base.data.owned(p) for p in (root, parent, overlay, universal_root))
    if root in (parent, overlay, universal_root) or len({p.parent for p in (root, parent, overlay, universal_root)}) != 1:
        raise ValueError('Use a NEW sibling root, never overwrite a source root')
    if (base.sha(base.__file__) != boundary.FROZEN_SHA or base.sha(boundary.__file__) != universal.BOUNDARY_SHA
            or base.sha(universal.__file__) != '92c3bc0ec8afeab1220e653106f786b746a0a1ba96fdb8a3996228f3a5e7e296'):
        raise ValueError('Known frozen collector/input/universal source changed')
    protocol, manifest, inventory = base.load(parent)
    input_v1 = base.read(overlay / 'recovery.json')
    old_v2 = base.read(universal_root / 'universal.json')
    base.data.verify_seal(input_v1, 'recovery_sha256')
    base.data.verify_seal(old_v2, 'universal_sha256')
    for path, digest in {**old_v2['source_sha256'], **old_v2['immutable_source_metadata_sha256']}.items():
        if base.sha(path) != digest:
            raise ValueError('Existing V2 immutable pin changed: ' + path)
    groups = base.task_groups(manifest)
    typed = sorted(k for k in groups if k.split('/')[0] in ('TSP', 'ATSP'))
    if (len(groups) != 89 or len(manifest['tasks']) != 113920 or len(typed) != 13
            or protocol['mode'] != 'cached_native_v1' or protocol['batch_size'] != 1 or protocol['lanes'] != 4
            or any(len(tasks) != 1280 for tasks in groups.values())
            or len(input_v1['targets']) != 5 or not set(input_v1['targets']) <= set(typed)
            or old_v2['input_boundary_keys'] != input_v1['targets']
            or input_v1['parent_root'] != str(parent) or old_v2['parent_root'] != str(parent)
            or old_v2['input_boundary_root'] != str(overlay)
            or any(base.read(p / 'protocol.json') != protocol for p in (overlay, universal_root))):
        raise ValueError('Require the SAME native full18/89/113920 protocol and original five-key InputV1')
    policy = {key: str(overlay if key in input_v1['targets'] else
        universal_root if index % 4 == 1 else parent) for index, key in enumerate(protocol['group_order'])}
    proof = _diagnostic_proof(diagnostic, parent, protocol, manifest, inventory)
    if policy[proof['deployment']] != str(parent):
        raise ValueError('The one diagnosed numerical failure must remain Original-sourced')
    from . import r114_group_worker
    sources = {str(Path(p).resolve()): base.sha(p) for p in (__file__, base.__file__, boundary.__file__,
        universal.__file__, base.data.__file__, base.data.r114.__file__, r114_group_worker.__file__)}
    pins = {str(p): base.sha(p) for p in (parent / 'protocol.json', overlay / 'protocol.json',
        overlay / 'recovery.json', universal_root / 'protocol.json', universal_root / 'universal.json')}
    pins.update({proof['diagnostic_path']: proof['diagnostic_sha256'], proof['failure_path']: proof['failure_sha256']})
    environment = _environment_pin()
    sources[environment['path']] = environment['sha256']
    recovery = base.data.seal(dict(version=VERSION, checker_version=CHECK_VERSION, parent_root=str(parent),
        input_boundary_root=str(overlay), universal_root=str(universal_root),
        input_boundary_keys=input_v1['targets'], input_boundary_recovery_sha256=input_v1['recovery_sha256'],
        source_universal_sha256=old_v2['universal_sha256'], native_collection_sha256=protocol['collection_sha256'],
        source_root_by_deployment=policy, source_policy='fixed five InputV1; remaining lane1 V2; all other Original; no longest-prefix/mincost selection',
        explicit_cpu_deployments=typed, source_sha256=sources, immutable_metadata_sha256=pins,
        environment=environment, diagnosed_failure=proof, source_prefix_snapshotted_at_create=False,
        independent_rule='strict first; L only: FP64 <= nextafter32(float32(R+1e-5),+inf) AND sequential FP32 <= float32(R+1e-5)',
        solver_model_weights_seeds_budget_unchanged=True, candidate_pool_unchanged=True,
        historical_labels_changed=False, diagnostic_promoted_to_label=False), 'numeric_sha256')
    base.write(root / 'protocol.json', protocol)
    base.write(root / 'numeric.json', recovery)
    return dict(root=str(root), numeric_sha256=recovery['numeric_sha256'], deployments=89,
        planned_labels=113920, source_policy=policy, solver_execution=False, prefixes_snapshotted=False)


def load(root):
    root = base.data.owned(root)
    recovery = base.read(root / 'numeric.json')
    base.data.verify_seal(recovery, 'numeric_sha256')
    if recovery['version'] != VERSION or recovery['checker_version'] != CHECK_VERSION:
        raise ValueError('Wrong numerical collection/checker version')
    for path, digest in {**recovery['source_sha256'], **recovery['immutable_metadata_sha256']}.items():
        if base.sha(path) != digest:
            raise ValueError('Pinned numerical recovery evidence/source changed: ' + path)
    protocol, manifest, inventory = base.load(root)
    if recovery['native_collection_sha256'] != protocol['collection_sha256']:
        raise ValueError('Native collection protocol changed')
    return recovery, protocol, manifest, inventory


def assert_sources_quiet(recovery, assigned, lane):
    roots = {source_root(recovery, key).resolve() for key in assigned}
    for path in Path('/proc').glob('[0-9]*/cmdline'):
        try:
            if path.stat().st_uid != os.getuid() or int(path.parent.name) == os.getpid():
                continue
            argv = path.read_bytes().decode(errors='replace').strip('\0').split('\0')
            def option(name):
                for index, value in enumerate(argv):
                    if value == name and index + 1 < len(argv):
                        return argv[index + 1]
                    if value.startswith(name + '='):
                        return value[len(name) + 1:]
                return None
            used = [option('--root'), option('--collection-root'), option('--source-root')]
            if any(p and Path(p).resolve() in roots for p in used) and (option('--deployment') in assigned or option('--lane') == str(lane)):
                raise base.NotReady('Fixed source producer still active; wait, do not stop PID ' + path.parent.name)
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue


def _source_kind(recovery, key):
    source = source_root(recovery, key)
    return 'inputV1' if source == Path(recovery['input_boundary_root']) else 'v2' if source == Path(recovery['universal_root']) else 'original'


def validator(manifest, inventory, protocol):
    endpoints = {e['endpoint_id']: e for e in manifest['endpoints']}

    @lru_cache(maxsize=16)
    def item(endpoint_id):
        e = endpoints[endpoint_id]
        if base.sha(e['input_path']) != e['input_sha256']:
            raise ValueError('Pinned endpoint input changed')
        return base.read(e['input_path'])['item']

    def validate(row, task, stage, binding_sha, adopted=False):
        base.validate_receipt(row, task, protocol, stage)
        key = task['problem'] + '/' + task['method']
        if (row.get('profile_sha256') != inventory['deployments'][key]['profile_sha256']
                or row.get('binding_sha256') != binding_sha or row.get('device') != 'cuda:0'
                or row.get('execution_mode') != protocol['mode']
                or row.get('model_seed') != base.data.r114.seed_for(2, task['problem'], task['method'], 'model')):
            raise ValueError('Native profile/input/seed/physical binding mismatch: ' + key)
        native = item(task['endpoint_id'])
        if adopted or task['problem'] in ('TSP', 'ATSP'):
            base.data.check_tour(task['problem'], native, row['tour'], row['cost'])
            evidence = dict(version=CHECK_VERSION, mode='original_strict', boundaries=[],
                original_strict_first=True, other_constraints_and_cost_unchanged=True)
        else:
            audit = numeric_check_route(task['problem'], native, row['tour'], row['cost'])
            evidence = audit['independent_numeric_validation']
        if not adopted and row.get('independent_numeric_validation') != evidence:
            raise ValueError('New receipt lacks its exact independent numeric evidence')
        return evidence
    return validate


def _source_provenance(recovery, source, relative, row, key, digest):
    kind = _source_kind(recovery, key)
    chain = None
    if kind == 'inputV1':
        if (row.get('input_boundary_version') != boundary.VERSION
                or row.get('input_boundary_recovery_sha256') != recovery['input_boundary_recovery_sha256']
                or row.get('observed_native_input', {}).get('device') != 'cpu'):
            raise ValueError('InputV1 CPU input provenance missing')
    elif kind == 'v2':
        adoption = source / 'adoption' / relative
        if adoption.exists():
            ref = base.read(adoption)
            if (ref.get('universal_sha256') != recovery['source_universal_sha256']
                    or ref.get('source_receipt_sha256') != digest or base.sha(ref['source_path']) != digest):
                raise ValueError('V2 source adoption chain changed')
            chain = dict(path=str(adoption), sha256=base.sha(adoption))
        elif (row.get('universal_version') != universal.VERSION
                or row.get('universal_sha256') != recovery['source_universal_sha256']
                or row.get('universal_execution') != ('new_explicit_cpu' if key.split('/')[0] in ('TSP', 'ATSP') else 'new_unchanged_session_solve')):
            raise ValueError('V2 new receipt provenance missing')
    return kind, chain


def copy_receipt(root, source, task, stage, lane, recovery, binding_sha, validate):
    key = task['problem'] + '/' + task['method']
    relative = Path('preflight') / (base.slug(key) + '.json') if stage == 'preflight' else Path('labels') / (task['task_id'] + '.json')
    old, new = source / relative, root / relative
    content = old.read_bytes()
    row, digest = json.loads(content), hashlib.sha256(content).hexdigest()
    validate(row, task, stage, binding_sha, adopted=True)
    kind, chain = _source_provenance(recovery, source, relative, row, key, digest)
    new.parent.mkdir(parents=True, exist_ok=True)
    if new.exists():
        if new.read_bytes() != content:
            raise ValueError('Never overwrite different destination receipt bytes')
    else:
        try:
            os.link(old, new)
        except OSError as error:
            if error.errno != errno.EXDEV:
                raise
            with new.open('xb') as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
    if base.sha(new) != digest or base.sha(old) != digest:
        raise ValueError('Source receipt changed during adoption')
    base.write(root / 'adoption' / relative, dict(kind='adopted_' + kind, task_id=task['task_id'], stage=stage,
        lane=lane, source_root=str(source), source_path=str(old), source_receipt_sha256=digest,
        destination_path=str(new), destination_sha256=digest, source_layer_reference=chain,
        numeric_sha256=recovery['numeric_sha256'], collection_sha256=recovery['native_collection_sha256'],
        byte_identical=True, new_solver_execution=False, independent_numeric_version=CHECK_VERSION))
    return digest


def _covered_failure(recovery, source, key, task, stage, binding_sha, protocol):
    path = source / 'unavailable' / (base.slug(key) + '.json')
    if not path.exists():
        return None
    failure = base.read(path)
    if (task is None or any(failure.get(k) != task.get(k) for k in IDENTITY)
            or failure.get('stage') != stage or failure.get('binding_sha256') != binding_sha
            or failure.get('collection_sha256') != protocol['collection_sha256']):
        raise base.NotReady('Source failure does not bind the exact next task/input/seed/binding: ' + str(path))
    message = failure.get('first_error', '')
    if (key in recovery['explicit_cpu_deployments'] and failure.get('error_type') == 'TypeError'
            and 'cuda:0' in message and 'numpy' in message):
        return dict(path=str(path), sha256=base.sha(path), covered_by='all13 explicit CPU input construction')
    proof = recovery['diagnosed_failure']
    if (key == proof['deployment'] and task['task_id'] == proof['task_id']
            and str(path) == proof['failure_path'] and base.sha(path) == proof['failure_sha256']
            and binding_sha == proof['binding_sha256'] and failure.get('error_type') == 'ValueError'
            and message == 'Route-length limit violation'):
        return dict(path=str(path), sha256=base.sha(path), covered_by=CHECK_VERSION,
            diagnostic_sha256=proof['diagnostic_sha256'])
    raise base.NotReady('Unknown/hard source failure is NOT an automatic retry target: ' + str(path))


def bootstrap_lane(root, lane, release, binding):
    root = base.data.owned(root)
    recovery, protocol, manifest, inventory = load(root)
    assigned, groups = universal.lane_keys(protocol, lane), base.task_groups(manifest)
    release_data = base.data.require_release(release, protocol['plan_sha256'])
    binding_sha = base.data.require_binding(binding, release_data, 'cuda:0')
    mutex = universal.require_mutex()
    assert_sources_quiet(recovery, assigned, lane)
    marker = root / 'bootstrap' / ('lane' + str(lane) + '.json')
    if marker.exists():
        saved = base.read(marker)
        base.data.verify_seal(saved, 'bootstrap_sha256')
        if saved['numeric_sha256'] != recovery['numeric_sha256'] or saved['binding_sha256'] != binding_sha:
            raise ValueError('Existing bootstrap has a different recovery/binding')
        return dict(saved, status='already_bootstrapped')
    validate, deployments = validator(manifest, inventory, protocol), {}
    originals = {e['endpoint_id'] for e in manifest['endpoints'] if e['cell'] == 'original'}
    for key in assigned:
        source, tasks = source_root(recovery, key), groups[key]
        expected = dict(deployment=key, binding_sha256=binding_sha, collection_sha256=protocol['collection_sha256'])
        if base.read(source / 'deployment_bindings' / (base.slug(key) + '.json')) != expected:
            raise ValueError('Fixed source per-key physical binding differs: ' + key)
        original = next(t for t in tasks if t['endpoint_id'] in originals)
        qsha = copy_receipt(root, source, original, 'preflight', lane, recovery, binding_sha, validate)
        state = base.progress(source, key)
        committed = state['next_index']
        if (type(committed) is not int or type(state['completed']) is not int
                or not 0 <= committed <= len(tasks) or state['completed'] != committed
                or committed and state.get('collection_sha256') != protocol['collection_sha256']):
            raise ValueError('Invalid source atomic prefix: ' + key)
        adopted, last_sha = 0, None
        for index, task in enumerate(tasks):
            path = source / 'labels' / (task['task_id'] + '.json')
            if not path.exists():
                if index < committed or any((source / 'labels' / (t['task_id'] + '.json')).exists() for t in tasks[index + 1:]):
                    raise ValueError('Fixed source is not a contiguous valid prefix: ' + key)
                break
            last_sha = copy_receipt(root, source, task, 'label', lane, recovery, binding_sha, validate)
            adopted += 1
            if adopted == committed and (state.get('last_receipt_sha256') != last_sha
                    or Path(state.get('last_receipt', '')).resolve() != path.resolve()):
                raise ValueError('Source atomic state does not bind its last committed receipt')
        error = _covered_failure(recovery, source, key, tasks[adopted] if adopted < len(tasks) else None,
            'label', binding_sha, protocol)
        if state.get('status') == 'unavailable' and error is None:
            raise base.NotReady('Unexplained unavailable source progress; preserve first error')
        base.write(root / 'deployment_bindings' / (base.slug(key) + '.json'), expected)
        next_state = dict(next_index=adopted, completed=adopted, adopted_labels=adopted,
            status='complete' if adopted == len(tasks) else 'running',
            collection_sha256=protocol['collection_sha256'], numeric_sha256=recovery['numeric_sha256'])
        if adopted:
            next_state.update(last_receipt=str(root / 'labels' / (tasks[adopted - 1]['task_id'] + '.json')), last_receipt_sha256=last_sha)
        base.atomic(root / 'progress' / (base.slug(key) + '.json'), next_state)
        deployments[key] = dict(source_root=str(source), source_atomic_progress=committed, adopted_labels=adopted,
            next_index=adopted, adopted_saved_uncommitted_tail=adopted - committed,
            preflight_sha256=qsha, source_previous_error=error, source_failure_flags_copied=False)
    result = base.data.seal(dict(status='bootstrap_complete', version=VERSION, lane=lane,
        assigned_deployments=assigned, deployments=deployments, binding_sha256=binding_sha,
        numeric_sha256=recovery['numeric_sha256'], collection_sha256=protocol['collection_sha256'],
        adopted_labels=sum(r['adopted_labels'] for r in deployments.values()), main_mutex=mutex,
        old_records_overwritten=False, new_solver_execution=False), 'bootstrap_sha256')
    base.write(marker, result)
    return result


@contextmanager
def _numeric_audit_scope():
    previous = base.data.r114.check_route
    base.data.r114.check_route = numeric_check_route
    try:
        yield
    finally:
        base.data.r114.check_route = previous


def numeric_solver(recovery):
    source = inspect.getsource(ORIGINAL_SOLVE)
    old = "native = torch.tensor(payload['item'], dtype=torch.float32)"
    if source.count(old) != 1:
        raise ValueError('Frozen CPU-constructor boundary changed')
    namespace = dict(ORIGINAL_SOLVE.__globals__)
    exec(compile(source.replace(old, old[:-1] + ", device='cpu')"), str(Path(__file__).resolve()) + ':explicit_cpu', 'exec'), namespace)
    typed_solve = namespace['solve_native']

    def solve(session, manifest, task):
        typed = task['problem'] in ('TSP', 'ATSP')
        with _numeric_audit_scope():
            row = (typed_solve if typed else ORIGINAL_SOLVE)(session, manifest, task)
        evidence = row['route_audit'].get('independent_numeric_validation', dict(version=CHECK_VERSION,
            mode='original_strict', boundaries=[], original_strict_first=True, other_constraints_and_cost_unchanged=True))
        row.update(numeric_version=VERSION, numeric_sha256=recovery['numeric_sha256'],
            numeric_execution='new_explicit_cpu' if typed else 'new_unchanged_session_bounded_independent_audit',
            native_input_placement='explicit_cpu' if typed else 'unchanged_session.solve',
            independent_numeric_validation=evidence)
        return row
    return solve


def worker(root, key, phase, max_jobs, device, release, binding):
    root = base.data.owned(root)
    recovery, protocol, _, _ = load(root)
    if key not in protocol['group_order'] or phase not in ('label', 'preflight') or not 1 <= max_jobs <= 1280:
        raise ValueError('Use a legal deployment and finite budget1..1280')
    if phase == 'preflight' and not (root / 'preflight' / (base.slug(key) + '.json')).exists():
        raise base.NotReady('Qualifications must be adopted, never freshly rerun')
    lane = protocol['group_order'].index(key) % protocol['lanes']
    marker = base.read(root / 'bootstrap' / ('lane' + str(lane) + '.json'))
    base.data.verify_seal(marker, 'bootstrap_sha256')
    released = base.data.require_release(release, protocol['plan_sha256'])
    binding_sha = base.data.require_binding(binding, released, device)
    universal.require_mutex()
    if marker['numeric_sha256'] != recovery['numeric_sha256'] or marker['binding_sha256'] != binding_sha:
        raise ValueError('Numeric bootstrap/physical binding differs')
    previous = base.solve_native
    base.solve_native = numeric_solver(recovery)
    try:
        result = universal.traceback_worker()(root, key, phase, max_jobs, device, release, binding)
    finally:
        base.solve_native = previous
    return dict(result, numeric_version=VERSION, numeric_sha256=recovery['numeric_sha256'])


def status(root):
    root = base.data.owned(root)
    recovery, protocol, manifest, _ = load(root)
    rows = {}
    for key, tasks in base.task_groups(manifest).items():
        state = base.progress(root, key)
        rows[key] = dict(planned=len(tasks), completed=state['completed'],
            preflight=(root / 'preflight' / (base.slug(key) + '.json')).is_file(),
            unavailable=(root / 'unavailable' / (base.slug(key) + '.json')).exists())
    lanes = [i for i in range(protocol['lanes']) if (root / 'bootstrap' / ('lane' + str(i) + '.json')).exists()]
    return dict(root=str(root), numeric_sha256=recovery['numeric_sha256'], deployments=rows,
        bootstrapped_lanes=lanes, planned_labels=len(manifest['tasks']),
        completed_labels=sum(r['completed'] for r in rows.values()),
        full89_complete=len(lanes) == protocol['lanes'] and all(r['preflight'] and not r['unavailable']
            and r['completed'] == r['planned'] for r in rows.values()), solver_execution=False)


def _adoption_reference(root, relative, recovery, key):
    path = root / 'adoption' / relative
    ref = base.read(path)
    source = source_root(recovery, key)
    if (ref.get('numeric_sha256') != recovery['numeric_sha256'] or Path(ref['source_root']) != source
            or Path(ref['source_path']) != source / relative or ref.get('byte_identical') is not True
            or ref['source_receipt_sha256'] != base.sha(root / relative)
            or ref['source_receipt_sha256'] != base.sha(source / relative)):
        raise ValueError('Fixed-source adoption provenance changed')
    if ref.get('source_layer_reference'):
        chain = ref['source_layer_reference']
        if base.sha(chain['path']) != chain['sha256']:
            raise ValueError('Nested V2 adoption proof changed')
    return ref


def export(root, output=None):
    root = base.data.owned(root)
    recovery, protocol, manifest, inventory = load(root)
    if not status(root)['full89_complete']:
        raise base.NotReady('Require complete all89 SINGLE-root113920 pool; never export partial candidates')
    output = base.data.owned(output or root / 'family_labels.jsonl')
    if root not in output.parents or output.exists():
        raise ValueError('Use a new export under the numeric root; never overwrite')
    output.parent.mkdir(parents=True, exist_ok=True)
    validate, groups = validator(manifest, inventory, protocol), defaultdict(list)
    bindings = {key: base.read(root / 'deployment_bindings' / (base.slug(key) + '.json'))['binding_sha256']
        for key in base.task_groups(manifest)}
    originals = {e['endpoint_id'] for e in manifest['endpoints'] if e['cell'] == 'original'}
    for key, tasks in base.task_groups(manifest).items():
        relative = Path('preflight') / (base.slug(key) + '.json')
        _adoption_reference(root, relative, recovery, key)
        validate(base.read(root / relative), next(t for t in tasks if t['endpoint_id'] in originals),
            'preflight', bindings[key], adopted=True)
    for task in manifest['tasks']:
        groups[task['endpoint_id']].append(task)
    temp = output.with_name('.' + output.name + '.' + uuid.uuid4().hex)
    digest, counts, cases, endpoints = hashlib.sha256(), Counter(), [], 0
    try:
        with temp.open('xb') as stream:
            for endpoint in manifest['endpoints']:
                records, provenance = [], []
                for task in groups[endpoint['endpoint_id']]:
                    key, relative = task['problem'] + '/' + task['method'], Path('labels') / (task['task_id'] + '.json')
                    row = base.read(root / relative)
                    adopted = (root / 'adoption' / relative).exists()
                    evidence = validate(row, task, 'label', bindings[key], adopted=adopted)
                    if adopted:
                        ref = _adoption_reference(root, relative, recovery, key)
                    else:
                        expected = 'new_explicit_cpu' if task['problem'] in ('TSP', 'ATSP') else 'new_unchanged_session_bounded_independent_audit'
                        if (row.get('numeric_version') != VERSION or row.get('numeric_sha256') != recovery['numeric_sha256']
                                or row.get('numeric_execution') != expected):
                            raise ValueError('New native receipt lacks numeric-layer identity')
                        ref = dict(kind=expected, task_id=task['task_id'], receipt_path=str(root / relative),
                            receipt_sha256=base.sha(root / relative), numeric_sha256=recovery['numeric_sha256'])
                    if evidence['boundaries']:
                        cases.append(dict(task_id=task['task_id'], receipt_sha256=base.sha(root / relative),
                            source=ref['kind'], evidence=evidence))
                    provenance.append(dict(ref, independent_numeric_validation=evidence))
                    records.append(row)
                    counts[ref['kind']] += 1
                targets = base.data.full_pool_targets(manifest['pools'][endpoint['problem']], records)
                encoded = (json.dumps(dict(endpoint=endpoint, targets=targets,
                    collection_sha256=protocol['collection_sha256'], numeric_version=VERSION,
                    numeric_sha256=recovery['numeric_sha256'], independent_numeric_version=CHECK_VERSION,
                    label_provenance=provenance), sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()
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
        numeric_version=VERSION, numeric_sha256=recovery['numeric_sha256'], independent_numeric_version=CHECK_VERSION,
        source_policy=recovery['source_root_by_deployment'], label_provenance_counts=dict(counts),
        boundary_case_count=len(cases), boundary_cases=cases, diagnostic_evidence=recovery['diagnosed_failure'],
        native_receipt_root=str(root), qualification='byte-identical adopted successful originals; no repeated native preflights')
    base.write(output.with_suffix('.receipt.json'), result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    init = commands.add_parser('create')
    init.add_argument('--root', type=Path, default=ROOT)
    init.add_argument('--parent', type=Path, default=PARENT)
    init.add_argument('--input-boundary-root', type=Path, default=OVERLAY)
    init.add_argument('--universal-root', type=Path, default=UNIVERSAL)
    init.add_argument('--diagnostic', type=Path, default=DIAGNOSTIC)
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
            result = create(args.root, args.parent, args.input_boundary_root, args.universal_root, args.diagnostic)
        elif args.command == 'bootstrap-lane':
            result = bootstrap_lane(args.root, args.lane, args.release, args.binding)
        elif args.command == 'worker':
            if args.root.resolve() not in args.result.resolve().parents:
                raise ValueError('Result must remain under the NEW numeric root')
            result = worker(args.root, args.deployment, args.phase, args.max_jobs, args.device, args.release, args.binding)
            base.write(args.result, result)
        elif args.command == 'status':
            result = status(args.root)
        else:
            result = export(args.root, args.output)
    except Exception as error:
        result = dict(status='NotReady' if isinstance(error, base.NotReady) else 'failed',
            first_error=str(error), error_type=type(error).__name__, traceback=traceback.format_exc(), automatic_retry=False)
        if (args.command == 'worker' and args.root.resolve() in args.result.resolve().parents
                and args.root.resolve() not in {PARENT.resolve(), OVERLAY.resolve(), UNIVERSAL.resolve()}
                and (args.root / 'numeric.json').is_file()):
            base.write(args.result, result)
        print(json.dumps(result, sort_keys=True), flush=True)
        return 3 if isinstance(error, base.NotReady) else 1
    print(json.dumps(result, sort_keys=True), flush=True)
    return 2 if result.get('status') == 'unavailable' else 0


if __name__ == '__main__':
    raise SystemExit(main())

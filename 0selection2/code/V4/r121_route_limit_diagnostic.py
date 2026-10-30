"""Capture one failed native response without promoting it to a family label."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
import traceback

import numpy as np

from . import r121_collection as base


REPO = Path('/public/home/shiys/0selection2')
PREPARATION = REPO / 'code/V4/runs/R121_structural_families/preparation'
COLLECTION = PREPARATION / 'collection_persistent_v1'
ROOT = PREPARATION / 'route_limit_diagnostic_v1'
TASK = 'VRPBL_src98457_original__reld_mt'
RELEASE = COLLECTION / 'main_r120_release.json'
BINDING = COLLECTION / 'main_binding_lane0.json'
PLAN = REPO / 'code/V4/runs/Return18_post_R120/dispatch/main_approved_r121_route_limit_diag_v1.json'


def route_lengths(problem, item, tour):
    fields = base.data.r114.fields(problem, item)
    xy64 = np.concatenate((fields['depot'][None], fields['xy'])).astype(np.float64)
    xy32 = xy64.astype(np.float32)
    sequence = [int(x) for x in tour]
    if not sequence or sequence[0] != 0:
        sequence.insert(0, 0)
    if sequence[-1] != 0:
        sequence.append(0)
    depot_positions = [i for i, node in enumerate(sequence) if node == 0]
    rows = []
    for index, (start, end) in enumerate(zip(depot_positions[:-1], depot_positions[1:])):
        route = sequence[start:end + 1]
        if len(route) < 3:
            continue
        if problem.startswith('OVRP'):
            route = route[:-1]
        edge64 = np.linalg.norm(np.diff(xy64[route], axis=0), axis=1)
        edge32 = np.linalg.norm(np.diff(xy32[route], axis=0), axis=1).astype(np.float32)
        sequential32 = np.float32(0.)
        for edge in edge32:
            sequential32 = np.float32(sequential32 + edge)
        rows.append(dict(route_index=index, customer_count=sum(node != 0 for node in route),
            charged_edges=len(edge64), route=route, input_fp64_length=float(edge64.sum()),
            rounded_input_fp64_length=float(np.linalg.norm(np.diff(xy32[route].astype(np.float64), axis=0), axis=1).sum()),
            numpy_fp32_sequential_length=float(sequential32),
            input_fp64_excess=float(edge64.sum())-float(item['route_limit']),
            limit=float(item['route_limit']), independent_existing_tolerance=1e-5,
            float32_calculation_is_approximate_witness_not_native_state=True))
    return rows


def emit_plan(root):
    from . import return18_launch as transport
    if Path(root).resolve() != ROOT:
        raise ValueError('Use the one literal diagnostic root')
    names = [*transport.TRANSPORT_SOURCES, 'code/V4/r121_route_limit_diagnostic.py',
        'code/V4/r121_collection.py', 'code/V4/r121_structural_data.py',
        'code/V4/r114_group_worker.py', 'code/V4/r114_response_data.py']
    plan = dict(schema=transport.SCHEMA, name='r121_one_route_limit_diagnostic_v1',
        main_approved=True, approved_by='main: actual terminal VRPBL/reld_mt route-limit error; one raw native response, no label promotion or automatic retry',
        approved_utc=transport.utc(), protocol=dict(path=str((COLLECTION/'protocol.json').relative_to(REPO)),
            sha256=transport.sha256(COLLECTION/'protocol.json')),
        source_sha256={name: transport.sha256(REPO/name) for name in names},
        jobs=[dict(id='R121_route_limit_diagnostic', arm='R121', slot='gpu05_1',
            root=str(ROOT.relative_to(REPO)), stages=[dict(id='one_failed_original_raw_route', kind='collect',
                argv=['python', '-B', '-u', '-m', 'code.V4.r121_route_limit_diagnostic', 'run',
                    '--root', str(ROOT), '--device', 'cuda:0', '--release', str(RELEASE),
                    '--binding', str(BINDING)])])])
    transport.write(PLAN, plan, exclusive=True)
    _, digest = transport.validate_plan(PLAN)
    return dict(plan=str(PLAN), sha256=digest, solver_execution=False)


def run(args):
    import torch
    root = base.data.owned(args.root)
    if root != ROOT or args.device != 'cuda:0':
        raise PermissionError('Use the declared single diagnostic and assigned local CUDA device')
    protocol, manifest, inventory = base.load(COLLECTION)
    release = base.data.require_release(args.release, protocol['plan_sha256'])
    binding_sha = base.data.require_binding(args.binding, release, args.device)
    task = next(t for t in manifest['tasks'] if t['task_id'] == TASK)
    endpoint = next(e for e in manifest['endpoints'] if e['endpoint_id'] == task['endpoint_id'])
    if base.sha(endpoint['input_path']) != task['input_sha256']:
        raise ValueError('Actual failed input changed')
    payload = base.read(endpoint['input_path'])
    result_path = root / 'diagnostic.json'
    if result_path.exists():
        raise FileExistsError('Preserve the one raw-response attempt')
    root.mkdir(parents=True, exist_ok=True)
    session = base._session(manifest, inventory, task['problem'] + '/' + task['method'], args.device)
    # Match the original runner through backend.run, but persist BEFORE its audit.
    from .r114_group_worker import reset_rng, reset_lazy_runner
    reset_lazy_runner(session.backend, task['method'])
    reset_rng(torch, task['solver_seed'], args.device)
    torch.cuda.synchronize()
    started = time.monotonic()
    native = session.backend.run([payload['item']])
    torch.cuda.synchronize()
    if len(native.costs) != 1 or len(native.tours) != 1:
        raise ValueError('A real singleton route is required for diagnosis')
    tour = base.data.r114.plain(native.tours[0])
    raw = dict(diagnostic_only=True, not_a_training_label=True, historical_costs_changed=False,
        task=task, input_path=endpoint['input_path'], profile_sha256=session.profile_sha256,
        binding_sha256=binding_sha, original_collection_sha256=protocol['collection_sha256'],
        raw_cost=float(native.costs[0]), raw_tour=tour, backend_audit=base.data.r114.plain(native.audit),
        native_seconds=time.monotonic()-started, route_metrics=route_lengths(task['problem'], payload['item'], tour),
        source_sha256={Path(p).name: base.sha(p) for p in (__file__, base.__file__, base.data.__file__)},
        model_or_solver_protocol_changed=False, solver_seed=task['solver_seed'], automatic_retry=False)
    try:
        raw['independent_audit'] = base.data.check_tour(task['problem'], payload['item'], tour, native.costs[0])
        raw['audit_passed'] = True
    except Exception as error:
        raw.update(audit_passed=False, audit_error=f'{type(error).__name__}: {error}', audit_traceback=traceback.format_exc())
    base.write(result_path, raw)
    print(json.dumps(dict(diagnostic_complete=True, audit_passed=raw['audit_passed'],
        audit_error=raw.get('audit_error'), route_count=len(raw['route_metrics']),
        max_route_length=max(row['input_fp64_length'] for row in raw['route_metrics']),
        max_excess=max(row['input_fp64_excess'] for row in raw['route_metrics']),
        raw_cost=raw['raw_cost'], training_label_written=False, path=str(result_path))), flush=True)
    return raw


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('emit-plan', 'run'))
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--release', type=Path, default=RELEASE)
    parser.add_argument('--binding', type=Path, default=BINDING)
    args = parser.parse_args(argv)
    if args.command == 'emit-plan':
        print(json.dumps(emit_plan(args.root)))
    else:
        run(args)


if __name__ == '__main__':
    main()

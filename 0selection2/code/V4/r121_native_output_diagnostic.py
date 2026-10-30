"""One explicit native diagnostic attempt, full traceback, no recovery/retry.

The frozen collector and failed recovery stay untouched. This tool instruments
only the native backend call boundary; it never normalizes or repairs its inputs.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import socket
import time
import traceback

from . import r121_collection as base

VERSION = 'r121_native_cuda_numpy_trace_v1_20261030'
FROZEN_SHA = '876cff5800f0c0a1067b7a36eee3e154b4b73dedb8030873a69dd7cefec49674'
ROOT = base.data.ROOT / 'collection_output_diagnostic_v1'
PARENT = base.data.ROOT / 'collection_persistent_v1'
FAILED = base.data.ROOT / 'collection_serialization_recovery_v1/unavailable/ATSP__UNICO_MatPOENet.json'
KEYS = ('ATSP/UNICO_MatPOENet', 'TSP/OMNI')


def metadata(value, depth=0):
    """No NumPy conversion, tensor values, .item(), or device transfers here."""
    result = dict(type=type(value).__module__ + '.' + type(value).__qualname__)
    for name in ('device', 'dtype', 'shape', 'is_cuda'):
        item = getattr(value, name, None)
        if item is not None:
            result[name] = list(item) if name == 'shape' else str(item)
    if depth < 2 and isinstance(value, (list, tuple)):
        result.update(length=len(value), first=[metadata(v, depth + 1) for v in value[:2]])
    return result


def exception_record(error):
    return dict(error_type=type(error).__name__, first_error=str(error),
        traceback=''.join(traceback.format_exception(type(error), error, error.__traceback__)),
        frames=[dict(filename=frame.filename, lineno=frame.lineno, name=frame.name, line=frame.line)
                for frame in traceback.extract_tb(error.__traceback__)])


def native_once(session, manifest, task, state):
    original = session.backend.run

    def traced(items):
        state['stage'] = 'native_backend.run'
        state['native_calls'] += 1
        state['native_input_metadata'] = metadata(items)
        try:
            result = original(items)
        except Exception as error:
            state['native_backend_exception'] = exception_record(error)
            raise
        state['native_output_metadata'] = dict(costs=metadata(result.costs), tours=metadata(result.tours))
        state['stage'] = 'after_native_backend.run_before_collector_audit'
        return result

    session.backend.run = traced
    try:
        # Deliberately use the original helper, NOT either recovery's substitutions.
        return base.solve_native(session, manifest, task)
    finally:
        session.backend.run = original


def run(args):
    if base.sha(base.__file__) != FROZEN_SHA:
        raise ValueError('Diagnostic requires the exact frozen876 collector')
    protocol, manifest, inventory = base.load(args.parent)
    failure = base.read(args.failed_attempt)
    key = failure['problem'] + '/' + failure['method']
    if key not in KEYS or failure['stage'] != 'preflight' or failure['status'] != 'unavailable':
        raise ValueError('Only an explicitly preserved failed original from these two deployments is allowed')
    task = next(t for t in manifest['tasks'] if t['task_id'] == failure['task_id'])
    if (failure['collection_sha256'] != protocol['collection_sha256']
            or any(failure[k] != task[k] for k in ('solver_seed', 'input_sha256', 'deployment_id', 'global_id'))):
        raise ValueError('Failed-attempt task/recipe/input identity changed')
    description = dict(version=VERSION, deployment=key, task_id=task['task_id'],
        source_failed_attempt=dict(path=str(args.failed_attempt), sha256=base.sha(args.failed_attempt),
            first_error=failure['first_error']), collection_sha256=protocol['collection_sha256'],
        manifest_sha256=manifest['manifest_sha256'], frozen_collector_sha256=FROZEN_SHA,
        diagnostic_source_sha256=base.sha(__file__), solver_model_seed_profile_changes=False,
        input_or_output_normalization=False, automatic_retry=False, new_labels=0,
        new_qualifications=0, max_native_calls=1)
    if not args.execute:
        return dict(description, solver_execution=False,
            required='one explicit main-bound execute, fresh output, original physical binding; no retry')
    release = base.data.require_release(args.release, protocol['plan_sha256'])
    binding_sha = base.data.require_binding(args.binding, release, args.device)
    if binding_sha != failure['binding_sha256']:
        raise ValueError('Diagnostic must retain the original physical binding')
    output = base.data.owned(args.output)
    parent = base.data.owned(args.parent)
    if parent == output.parent or parent in output.parents or FAILED.parent.parent in output.parents:
        raise ValueError('Diagnostic output must stay OUTSIDE both frozen attempt trees')
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise ValueError('Preserve existing diagnostic; never rerun this output')
    claim = output.with_suffix('.claim.json')
    # A crashed or previously claimed diagnostic is NOT automatically rerun.
    with claim.open('x') as stream:
        json.dump(dict(description, binding_sha256=binding_sha, host=socket.gethostname(), pid=os.getpid()), stream, sort_keys=True)
    state = dict(description, binding_sha256=binding_sha, host=socket.gethostname(), pid=os.getpid(),
        stage='native_session_initialization', native_calls=0, solver_execution=True)
    started = time.monotonic()
    try:
        session = base._session(manifest, inventory, key, args.device)
        state['profile_sha256'] = session.profile_sha256
        state['native_backend_type'] = type(session.backend).__module__ + '.' + type(session.backend).__qualname__
        state['torch_default_device_after_native_init'] = str(session.torch.get_default_device())
        state['stage'] = 'frozen_solve_native_before_backend'
        solved = native_once(session, manifest, task, state)
        state.update(status='diagnostic_passed_NOT_a_qualification', stage='independent_audit_passed',
            diagnostic_cost=solved['cost'], route_audit=solved['route_audit'])
    except Exception as error:
        state.update(status='diagnostic_failed', exception=exception_record(error))
        trace_path = output.with_suffix('.traceback.txt')
        with trace_path.open('x') as stream:
            stream.write(state['exception']['traceback'])
        state['traceback_file'] = str(trace_path)
        print(state['exception']['traceback'], file=__import__('sys').stderr, flush=True)
    state['elapsed_wall_s'] = time.monotonic() - started
    base.write(output, state)
    return state


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--parent', type=Path, default=PARENT)
    parser.add_argument('--failed-attempt', type=Path, default=FAILED)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--release', type=Path)
    parser.add_argument('--binding', type=Path)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args(argv)
    args.root = args.root.expanduser().resolve()
    args.output = (args.output or args.root / 'ATSP__UNICO_MatPOENet_attempt1.json').expanduser().resolve()
    if args.root not in args.output.parents:
        parser.error('--output must be inside --root')
    result = run(args)
    print(json.dumps(result, sort_keys=True), flush=True)
    return 2 if result.get('status') == 'diagnostic_failed' else 0


if __name__ == '__main__':
    raise SystemExit(main())

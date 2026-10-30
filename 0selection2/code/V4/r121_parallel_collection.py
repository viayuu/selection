"""Two-process, finite scheduling of the unchanged persistent native collector."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid

from . import r121_collection as base


VERSION = 'r121_parallel_collection_v1'
REPO = Path(__file__).resolve().parents[2]


def stamp():
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


def write(path, value, *, exclusive=False):
    path = base.data.owned(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n'
    if exclusive:
        with path.open('x') as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
    else:
        temporary = path.with_name('.' + path.name + '.' + uuid.uuid4().hex)
        temporary.write_text(content)
        temporary.replace(path)


def option(argv, flag):
    for index, item in enumerate(argv):
        if item == flag and index + 1 < len(argv):
            return argv[index + 1]
        if item.startswith(flag + '='):
            return item[len(flag) + 1:]
    return None


def assert_no_old_producer(root, keys, lane):
    # Frozen workers cannot cooperate with a new mutex: require a clean handoff.
    for path in Path('/proc').glob('[0-9]*/cmdline'):
        try:
            if path.stat().st_uid != os.getuid():
                continue
            argv = path.read_bytes().decode(errors='replace').strip('\0').split('\0')
            if 'code.V4.r121_collection' not in argv:
                continue
            module = argv.index('code.V4.r121_collection')
            action = argv[module + 1] if len(argv) > module + 1 else None
            declared = option(argv, '--root')
            worker_root = Path(declared).resolve() if declared else base.ROOT.resolve()
            if worker_root != root:
                continue
            if ((action == 'worker' and option(argv, '--deployment') in keys)
                    or (action == 'wave' and option(argv, '--lane') == str(lane))):
                raise base.NotReady(f'Existing frozen collector PID {path.parent.name}; parent must finish its lane handoff')
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue


def layout(root, lane, concurrency, maximum):
    if concurrency != 2 or type(maximum) is not int or not 1 <= maximum <= 1280 or lane not in range(4):
        raise ValueError('Require lane 0..3, concurrency=2 and a finite per-key budget 1..1280')
    root = base.data.owned(root)
    protocol, manifest, inventory = base.load(root)
    groups = base.task_groups(manifest)
    order = protocol['group_order']
    if (protocol['mode'] != 'cached_native_v1' or protocol['lanes'] != 4
            or protocol['batch_size'] != 1 or protocol['native_budget_changes'] is not False
            or protocol['deployment_seed_changes'] is not False
            or len(order) != 89 or len(set(order)) != 89 or set(order) != set(groups)
            or protocol['planned_labels'] != 113920 or sum(map(len, groups.values())) != 113920
            or any(len(tasks) != 1280 for tasks in groups.values())):
        raise ValueError('Only the frozen full113920/89 persistent native batch1 protocol is supported')
    assigned = [key for index, key in enumerate(order) if index % 4 == lane]
    return root, protocol, manifest, inventory, groups, assigned


def snapshot(root, protocol, manifest, inventory, groups, assigned, maximum, binding_sha=None):
    originals = {row['endpoint_id'] for row in manifest['endpoints'] if row['cell'] == 'original'}
    rows = {}
    for key in assigned:
        state = base.progress(root, key)
        unavailable = root / 'unavailable' / (base.slug(key) + '.json')
        preflight = root / 'preflight' / (base.slug(key) + '.json')
        row = dict(deployment=key, planned=len(groups[key]), start_index=state['next_index'],
                   initial_completed=state['completed'], automatic_retry=False)
        rows[key] = row
        if (unavailable.exists() or state.get('status') == 'unavailable'
                or inventory['deployments'][key]['status'].startswith('blocked')):
            row.update(status='needs_other_recovery_root', first_error_path=str(unavailable),
                       first_error_sha256=base.sha(unavailable) if unavailable.is_file() else None)
            continue
        if (type(state['next_index']) is not int or type(state['completed']) is not int
                or not 0 <= state['next_index'] <= len(groups[key]) or state['completed'] != state['next_index']
                or (state['next_index'] and state.get('collection_sha256') != protocol['collection_sha256'])):
            raise ValueError('Invalid frozen per-key progress: ' + key)
        if not preflight.is_file():
            row.update(status='qualification_pending', preflight=str(preflight))
            continue
        qualification = base.read(preflight)
        original = next(task for task in groups[key] if task['endpoint_id'] in originals)
        base.validate_receipt(qualification, original, protocol, 'preflight')
        if (qualification.get('profile_sha256') != inventory['deployments'][key]['profile_sha256']
                or (binding_sha is not None and qualification.get('binding_sha256') != binding_sha)):
            raise ValueError('Existing native qualification/profile/binding mismatch: ' + key)
        row.update(preflight=str(preflight), preflight_sha256=base.sha(preflight),
                   max_jobs=min(maximum, len(groups[key]) - state['next_index']),
                   status='already_complete' if state['next_index'] == len(groups[key]) else 'qualified')
    return rows


def child_command(root, key, maximum, device, release, binding, result):
    return [sys.executable, '-B', '-u', '-m', 'code.V4.r121_collection', 'worker',
            '--root', str(root), '--deployment', key, '--phase', 'label', '--max-jobs', str(maximum),
            '--device', device, '--release', str(release), '--binding', str(binding), '--result', str(result)]


def inherited_fds():
    value = os.environ.get('RETURN18_GPU_MUTEX_FD')
    if value is None:
        return ()
    descriptor = int(value)
    os.fstat(descriptor)
    return (descriptor,)


def run(root, lane, concurrency, maximum, device, release, binding, *, describe=False):
    root, protocol, manifest, inventory, groups, assigned = layout(root, lane, concurrency, maximum)
    if describe:
        rows = snapshot(root, protocol, manifest, inventory, groups, assigned, maximum)
        return dict(version=VERSION, lane=lane, concurrency=2, max_jobs_per_deployment=maximum,
            deployments=rows, group_order=protocol['group_order'], full89_complete=False,
            planned_labels=113920, solver_execution=False, candidate_pool_unchanged=True), 0
    if device != 'cuda:0':
        raise ValueError('Use the same assigned local cuda:0 as the frozen native preflights')
    release, binding = Path(release).resolve(), Path(binding).resolve()
    release_data = base.data.require_release(release, protocol['plan_sha256'])
    binding_sha = base.data.require_binding(binding, release_data, device)
    if len(base.read(binding)['cpu_ids']) != 4:
        raise ValueError('The existing lane binding must retain exactly four CPUs')
    directory = root / VERSION / ('lane' + str(lane))
    receipt_path = directory / 'controller.json'
    if receipt_path.exists():
        raise FileExistsError('Preserve the existing one-shot lane receipt; no automatic controller restart')
    assert_no_old_producer(root, assigned, lane)
    rows = snapshot(root, protocol, manifest, inventory, groups, assigned, maximum, binding_sha)
    pass_fds = inherited_fds()
    started = time.monotonic()
    receipt = dict(version=VERSION, state='running', started_utc=stamp(), root=str(root), lane=lane,
        concurrency=2, max_jobs_per_deployment=maximum, binding_sha256=binding_sha,
        collection_sha256=protocol['collection_sha256'], planned_labels=113920,
        group_order=protocol['group_order'], assigned_deployments=assigned, deployments=rows,
        qualified_segment_complete=False, full89_complete=False, export_performed=False,
        candidate_pool_unchanged=True, native_budget_changes=False, deployment_seed_changes=False,
        batch_size=1, automatic_retry=False, controller_source_sha256=base.sha(__file__))
    write(receipt_path, receipt, exclusive=True)
    pending = [key for key in assigned if rows[key]['status'] == 'qualified']
    qualified = [key for key in assigned if rows[key]['status'] in ('qualified', 'already_complete')]
    active, failed, interrupted = {}, False, []
    first_error = None

    def forward(signum, _frame):
        interrupted.append(signum)
        for process, _, _ in active.values():
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signum)
                except ProcessLookupError:
                    pass

    handlers = {sig: signal.signal(sig, forward) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        while pending or active:
            while pending and len(active) < concurrency and not failed and not interrupted:
                key = pending.pop(0)
                row = rows[key]
                assert_no_old_producer(root, [key], lane)
                current = base.progress(root, key)
                if ((root / 'unavailable' / (base.slug(key) + '.json')).exists()
                        or current.get('status') == 'unavailable'
                        or current['next_index'] != row['start_index']
                        or current['completed'] != row['initial_completed']):
                    raise base.NotReady('Per-key state changed after planning; do not duplicate or retry: ' + key)
                child_dir = directory / 'jobs' / base.slug(key)
                log, result = child_dir / 'worker.log', child_dir / 'result.json'
                child_dir.mkdir(parents=True, exist_ok=True)
                if result.exists():
                    raise FileExistsError('Existing per-key child result retained: ' + key)
                command = child_command(root, key, row['max_jobs'], device, release, binding, result)
                row.update(status='starting', command=command, log=str(log), result=str(result), started_utc=stamp())
                write(receipt_path, receipt)
                output = log.open('x')
                try:
                    process = subprocess.Popen(command, cwd=REPO, env=dict(os.environ,
                        OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                        NUMEXPR_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1'),
                        stdout=output, stderr=subprocess.STDOUT, start_new_session=True, pass_fds=pass_fds)
                except BaseException:
                    output.close()
                    raise
                active[key] = process, output, time.monotonic()
                row.update(status='running', pid=process.pid)
                write(receipt_path, receipt)
            for key, (process, output, begin) in list(active.items()):
                code = process.poll()
                if code is None:
                    continue
                output.close()
                del active[key]
                row = rows[key]
                result_path = Path(row['result'])
                result = base.read(result_path) if result_path.is_file() else {}
                state = base.progress(root, key)
                attempted = result.get('attempted')
                valid = (code == 0 and result.get('status') == 'finite_budget_complete'
                    and result.get('deployment') == key and result.get('phase') == 'label'
                    and type(attempted) is int and attempted == row['max_jobs']
                    and result.get('completed', 0) + result.get('reused', 0) == attempted
                    and state['next_index'] == row['start_index'] + attempted
                    and state['completed'] == state['next_index']
                    and state.get('collection_sha256') == protocol['collection_sha256'])
                row.update(status='finite_budget_complete' if valid else 'failed', exit_status=code,
                    ended_utc=stamp(), wall_seconds=time.monotonic() - begin,
                    result_record=result, end_index=state['next_index'])
                if not valid:
                    failed = True
                    row['first_error'] = result.get('first_error', 'Child nonzero, missing result, or finite budget/progress identity mismatch; inspect worker.log')
                    if first_error is None:
                        first_error = dict(deployment=key, exit_status=code, first_error=row['first_error'], log=row['log'])
                write(receipt_path, receipt)
            if failed or interrupted:
                for key in pending:
                    rows[key].update(status='not_started_after_failure')
                pending.clear()
            if active:
                time.sleep(.1)
        receipt.update(state='failed' if failed or interrupted else 'finite_budget_complete',
            exit_status=1 if failed or interrupted else 0, ended_utc=stamp(),
            elapsed_s=time.monotonic() - started, first_error=first_error,
            qualified_segment_complete=bool(qualified) and not failed and not interrupted and all(
                rows[key]['status'] == 'already_complete' or
                (rows[key]['status'] == 'finite_budget_complete' and rows[key]['end_index'] == rows[key]['planned'])
                for key in qualified),
            full89_complete=False)
        write(receipt_path, receipt)
        return receipt, receipt['exit_status']
    except BaseException as error:
        receipt.update(state='failed', exit_status=1, first_error=first_error or str(error),
                       ended_utc=stamp(), elapsed_s=time.monotonic() - started)
        write(receipt_path, receipt)
        raise
    finally:
        # Only our children are waited on; never kill other users' work.
        for key, (process, output, begin) in active.items():
            code = process.wait()
            output.close()
            row = rows[key]
            result_path = Path(row['result'])
            row.update(status='drained_after_controller_failure', exit_status=code,
                ended_utc=stamp(), wall_seconds=time.monotonic() - begin,
                result_record=base.read(result_path) if result_path.is_file() else {})
        if active:
            write(receipt_path, receipt)
        for sig, handler in handlers.items():
            signal.signal(sig, handler)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True, help='Existing frozen collection parent')
    parser.add_argument('--lane', type=int, choices=range(4), required=True)
    parser.add_argument('--concurrency', type=int, choices=(2,), default=2)
    parser.add_argument('--max-jobs-per-deployment', type=int, default=1280)
    parser.add_argument('--device')
    parser.add_argument('--release', type=Path)
    parser.add_argument('--binding', type=Path)
    parser.add_argument('--describe', action='store_true', help='CPU-only plan; never qualify or execute solvers')
    args = parser.parse_args(argv)
    if not args.describe and (args.device is None or args.release is None or args.binding is None):
        parser.error('Production requires explicit --device --release --binding')
    try:
        result, code = run(args.root, args.lane, args.concurrency, args.max_jobs_per_deployment,
                           args.device, args.release, args.binding, describe=args.describe)
    except Exception as error:
        print(json.dumps(dict(state='failed', first_error=f'{type(error).__name__}: {error}',
                              automatic_retry=False, full89_complete=False)), flush=True)
        return 1
    print(json.dumps(result, sort_keys=True), flush=True)
    return code


if __name__ == '__main__':
    raise SystemExit(main())

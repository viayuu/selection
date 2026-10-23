"""Quiet CPU whole-version R104 evidence gate and opt-in one-shot publication.

No model, array, weight, queue, notification or Git action runs on import/status.
Only saved-result CPU comparison is allowed after both arm pipelines finish.
"""

import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time


REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / 'code/V4/runs/R104_capacity_scaling'
LAUNCH = REPO / 'code/V4/runs/R98_parallel_dispatch'
ARMS = ('A_width256', 'B_depth8')
REFERENCE_FLAGS = ('nss_root', 'historical_root', 'r89_root', 'r89a_root')
RECORD_NAMES = ('protocol.json', 'status.json', 'initialization.json', 'training_complete.json',
    'final_test_lock.json', 'final_metrics.json', 'test_results.json', 'inference_timing.json',
    'lib_stage_complete.json', 'lib_stage_status.json', 'postprocess_status.json',
    'full_model_statistics/protocol.json')


def private():
    spec = importlib.util.spec_from_file_location('code.V4._r104_goal_admin', REPO / 'code/V4/r98_goal.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


adapter = private()
PROBLEMS, optional, save, utc = adapter.PROBLEMS, adapter.optional, adapter.save, adapter.utc
file_hash, json_hash = adapter.file_hash, adapter.json_hash


def receipt_errors(path, record):
    """Normalize scalar/list errors without throwing away the original message."""
    if not isinstance(record, dict):
        return [dict(path=str(path), error='Expected a JSON receipt object')]
    errors = []
    phases = [str(record.get(k, '')).lower() for k in ('state', 'phase')]
    rc = record.get('exit_status', record.get('returncode', record.get('exit_code')))
    failed = any('failed' in p or p in ('error', 'stopped_on_error') for p in phases) or rc not in (None, 0)
    for key in ('error', 'errors'):
        values = record.get(key)
        if values:
            for value in values if isinstance(values, list) else [values]:
                errors.append(dict(path=str(path), error=value))
    if failed and not errors:
        errors.append(dict(path=str(path), error='phase=' + str(phases) + '; rc=' + str(rc)))
    stages = record.get('stages', [])
    if not isinstance(stages, list):
        errors.append(dict(path=str(path), error='Expected a stages list'))
    else:
        for stage in stages:
            errors.extend(receipt_errors(path, stage))
    return errors


def read_record(path, errors):
    try:
        record = optional(path)
        errors.extend(receipt_errors(path, record))
        return record if isinstance(record, dict) else {}
    except (OSError, TypeError, ValueError) as error:
        errors.append(dict(path=str(path), error=str(error)))
        return {}


def strict_split(result, rows, lock):
    identity = result.get('identity', result)
    return (bool(lock.get('checkpoint_sha256'))
        and identity.get('checkpoint_sha256') == lock.get('checkpoint_sha256')
        and adapter.split_ready(result, rows, lock))


def selected_train_ready(final, config, lock):
    train = final.get('train', {})
    if strict_split(train, 81000, lock):
        return True
    identity = train.get('identity', {})
    val = final.get('val', {})
    bound, val_identity = val.get('checkpoint_binding', {}), val.get('identity', {})
    rows = train.get('per_problem', {})
    model_sha = identity.get('model_state_sha256')
    # A capture-time TRAIN file SHA can be null; exact model-state proof must
    # match the runner's explicit selected-VAL-to-final-checkpoint binding.
    return (identity.get('checkpoint_sha256') is None and bool(model_sha)
        and identity.get('state_hash') == lock.get('state_hash')
        and identity.get('successful_updates') == lock.get('selected_update')
        and model_sha == val_identity.get('model_state_sha256') == bound.get('model_state_sha256')
        and bool(config.get('selected_val_binding_rule'))
        and bound.get('rule') == config.get('selected_val_binding_rule')
        and bound.get('checkpoint_sha256') == val_identity.get('checkpoint_sha256') == lock.get('checkpoint_sha256')
        and bound.get('state_hash') == lock.get('state_hash')
        and bound.get('selected_update') == lock.get('selected_update')
        and bound.get('archive_identity_unchanged') is True
        and isinstance(rows, dict) and set(rows) == set(PROBLEMS)
        and all(isinstance(row, dict) and row.get('n') == 81000 for row in rows.values())
        and isinstance(train.get('macro'), dict) and bool(train['macro']))


def timing_ready(record, lock):
    counts = dict(A_width256=7288969, B_depth8=3426057)
    if (not adapter.same_state(record, lock) or record.get('complete') is not True
            or record.get('experiment') != 'R104' or record.get('arm') != lock.get('arm')
            or record.get('parameter_count') != counts.get(lock.get('arm'))
            or record.get('checkpoint_sha256') != lock.get('checkpoint_sha256')
            or record.get('inference_rule') != lock.get('inference_rule')
            or record.get('repetitions') != 3 or record.get('warmup_batches') != 1
            or record.get('all_current_model_included') is not True
            or record.get('donor_present') is not False or record.get('cached_feature_latency_used') is not False
            or record.get('end_to_end_measured') is not False
            or not str(record.get('device', '')).startswith('cuda')
            or record.get('scope') != 'complete_standalone_model_original_GPU_resident_inputs'
            or not record.get('excludes')):
        return False
    rows = record.get('per_problem', [])
    if (not isinstance(rows, list) or len(rows) != len(PROBLEMS)
            or any(not isinstance(row, dict) for row in rows)
            or {row.get('problem') for row in rows} != set(PROBLEMS)):
        return False
    try:
        for row in rows:
            if row.get('instances') != 128:
                return False
            # R104 has no donor/transform feature path. Time its actual selector.
            measurement = row.get('standalone_model', {})
            samples, mean = measurement.get('batch_seconds'), measurement.get('mean_seconds_per_instance')
            if (not isinstance(samples, list) or len(samples) != 3
                    or any(isinstance(v, bool) or not isinstance(v, (int, float))
                           or not math.isfinite(v) or v <= 0 for v in [*samples, mean])
                    or not math.isclose(mean, sum(samples) / (3 * 128), rel_tol=1e-6, abs_tol=1e-12)):
                return False
    except (TypeError, ValueError):
        return False
    return True


def arm_checks(arm, run, records, main_lock_sha):
    checks = adapter.r99_checks(arm, run, records)
    checks.update(adapter.common_checks('R104', arm, run, records))
    config, lock = records['protocol.json'], records['final_test_lock.json']
    final, report = records['final_metrics.json'], records['full_model_statistics/protocol.json']
    post = records['postprocess_status.json']
    selected = lock.get('selected_update')
    checks['unique_VAL_lock'] &= (lock.get('selection_split', 'val') == 'val'
        and isinstance(selected, int) and not isinstance(selected, bool) and 0 < selected <= 57600
        and bool(config.get('inference_rule')) and lock.get('inference_rule') == config.get('inference_rule')
        and lock.get('rule') == config.get('checkpoint_rule') and 'macro_vs_sbs' in str(lock.get('rule', '')))
    checks['full_TRAIN'] = selected_train_ready(final, config, lock)
    checks['full_VAL'] &= strict_split(final.get('val', {}), 4500, lock)
    checks['full_TEST'] &= strict_split(records['test_results.json'].get('result', {}), 4500, lock)
    checks['CPU_report'] = (post.get('complete') is True and post.get('phase') == 'complete'
        and post.get('exit_status') == 0 and report.get('complete') is True
        and report.get('experiment') == 'R104' and report.get('checkpoint_lock') == lock)
    checks['main_protocol_lock'] = (bool(main_lock_sha)
        and config.get('main_protocol_lock', {}).get('sha256') == main_lock_sha)
    lib = records['lib_stage_complete.json']
    # The frozen reporter supplies checkpoint/datasets; worker stages supply rc0.
    checks['LIB'] = (lib.get('experiment') == 'R104' and lib.get('complete') is True
        and lib.get('checkpoint_sha256') == lock.get('checkpoint_sha256')
        and set(lib.get('datasets', [])) == {'TSPLIB', 'CVRPLIB'}
        and lib.get('exit_status', 0) == 0 and lib.get('state_hash', lock.get('state_hash')) == lock.get('state_hash'))
    timing_path = run / 'inference_timing.json'
    checks['timing_bound_to_report'] = (timing_path.is_file()
        and (report.get('inference_timing') == records['inference_timing.json']
            or report.get('inference_timing', {}).get('sha256') == file_hash(timing_path)))
    checks['all18_timing'] = timing_ready(records['inference_timing.json'], lock)
    checks['fixed36'] = (report.get('target_identity', {}).get('target_count') == 36
        and bool(report.get('target_identity', {}).get('sha256')))
    return checks


def status(version='R104', root=None, launch_root=LAUNCH, *, persist=False):
    if version != 'R104':
        raise ValueError('This private watcher owns only R104')
    root, launch_root = Path(root or ROOT).resolve(), Path(launch_root).resolve()
    errors, rows, configs, locks = [], [], {}, {}
    main_path = root / 'protocol_lock.json'
    main_sha = file_hash(main_path) if main_path.is_file() else None
    for arm in ARMS:
        run, identifier = root / arm, 'R104_' + arm
        records = {name: read_record(run / name, errors) for name in RECORD_NAMES}
        configs[arm], locks[arm] = records['protocol.json'], records['final_test_lock.json']
        output = launch_root / identifier
        spec = read_record(launch_root / 'specs' / (identifier + '.json'), errors)
        controller = read_record(output / 'controller.json', errors)
        worker = read_record(output / 'worker.json', errors)
        for name in ('queue_receipt.json', 'status.json'):
            read_record(output / name, errors)
        try:
            checks = arm_checks(arm, run, records, main_sha)
            expected = [('train', 'gpu'), ('TEST', 'gpu'), ('timing', 'gpu'), ('LIB', 'gpu'), ('PDF', 'cpu')]
            checks.update(launch_spec=spec.get('run_id') == identifier and spec.get('run_dir') == str(run)
                and spec.get('slots') == [{'A_width256': '3090_0', 'B_depth8': '3090_1'}[arm]]
                and [(s.get('name'), s.get('device')) for s in spec.get('stages', [])] == expected,
                controller_rc0=adapter.stages_ready(controller, ['allocation_GPU_pipeline', 'PDF']),
                worker_all_GPU_stages_rc0=adapter.stages_ready(worker, ['train', 'TEST', 'timing', 'LIB']))
        except (AttributeError, KeyError, TypeError, ValueError) as error:
            checks = dict(valid_receipts=False)
            errors.append(dict(path=str(run), error='Malformed completion evidence: ' + str(error)))
        checks.update({name: adapter.pdf_ready(run / 'full_model_statistics' / name) for name in adapter.R99_PDFS})
        rows.append(dict(arm=arm, run_id=identifier, root=str(run), checks=checks,
            ready=all(checks.values()), missing=[k for k, v in checks.items() if not v]))
    # Capacity axes intentionally differ in params, model dimensions and full init.
    match_keys = ('data_root', 'prepared_root', 'rows_per_problem', 'candidate_pools', 'baselines',
        'max_successful_updates', 'inference_rule', 'main_protocol_lock')
    matched = all(configs[ARMS[0]].get(k) is not None
        and configs[ARMS[0]].get(k) == configs[ARMS[1]].get(k) for k in match_keys)
    comparison = read_record(root / 'comparison_artifacts/protocol.json', errors)
    for name in ('goal_watch_receipt.json', 'goal_compare_receipt.json', 'github_publication.json'):
        read_record(root / name, errors)
    compared = (comparison.get('experiment') == 'R104' and comparison.get('complete') is True
        and comparison.get('exit_status', 0) == 0 and comparison.get('checkpoints') == locks
        and comparison.get('training') is False and comparison.get('inference') is False
        and comparison.get('parameter_matched') is False and comparison.get('equal_flops') is False
        and comparison.get('identical_full_initialization') is False
        and all((root / 'comparison_artifacts' / name).is_file()
                and (root / 'comparison_artifacts' / name).stat().st_size > 0 for name in
                ('paired_changes_B_vs_A.csv', 'target_recovery_B_vs_A.csv', 'target_recovery_summary_B_vs_A.csv')))
    ready = matched and all(row['ready'] for row in rows)
    complete = ready and compared and not errors
    result = dict(experiment='R104', arms=rows, errors=errors, matched_controls=matched,
        all_arms_ready=ready, comparison_complete=compared, complete=complete,
        phase='failed' if errors else 'complete' if complete else 'waiting_for_comparison' if ready else 'waiting_for_arms',
        cpu_only=True, automatic_retry=False, goal_complete=False, goal_status='active_unresolved',
        comparison_boundary='Two capacity axes, not equal full initialization, parameters or FLOPs')
    if persist:
        path, prior = root / 'goal_progress.json', optional(root / 'goal_progress.json')
        if {k: v for k, v in prior.items() if k != 'updated_utc'} == result:
            return prior
        result['updated_utc'] = utc()
        save(path, result)
    return result


def require_ready(root, launch_root, evidence):
    if not isinstance(evidence, str) or not evidence.strip():
        raise ValueError('Explicit main-ready evidence required before watch/compare/publication')
    from . import r104_launch as launch
    if Path(root).resolve() != launch.MODEL_ROOT.resolve() or Path(launch_root).resolve() != launch.ROOT.resolve():
        raise RuntimeError('Watcher requires canonical main-ready R104 roots')
    for arm in ARMS:
        path = Path(launch_root) / 'specs' / ('R104_' + arm + '.json')
        spec, plan = launch.check_ready(path)
        if plan.get('evidence') != evidence or spec.get('run_dir') != str(Path(root).resolve() / arm):
            raise RuntimeError('Both source-frozen arm plans must match main-ready evidence')


def compare_once(root=ROOT, launch_root=LAUNCH, evidence=None, **references):
    root = Path(root).resolve()
    require_ready(root, launch_root, evidence)
    current = status(root=root, launch_root=launch_root)
    if current['errors'] or not current['all_arms_ready'] or current['comparison_complete']:
        raise RuntimeError('CPU comparison requires BOTH complete pipelines, once only')
    with (root / 'goal_compare.lock').open('a') as claim:
        adapter.fcntl.flock(claim, adapter.fcntl.LOCK_EX | adapter.fcntl.LOCK_NB)
        path = root / 'goal_compare_receipt.json'
        if path.exists():
            raise RuntimeError('Comparison already attempted; preserve receipt, no retry')
        argv = [sys.executable, '-B', '-u', '-m', 'code.V4.r104_reporting', 'compare', '--root', str(root), '--device', 'cpu']
        for key in REFERENCE_FLAGS:
            if references.get(key) is not None:
                argv.extend(('--' + key.replace('_', '-'), str(Path(references[key]).resolve())))
        receipt = dict(state='running', started_utc=utc(), argv=argv, cpu_only=True,
            automatic_retry=False, main_ready_evidence=evidence)
        save(path, receipt)
        try:
            require_ready(root, launch_root, evidence)
            with (root / 'goal_compare.log').open('a') as log:
                process = subprocess.run(argv, cwd=REPO, env=adapter.cpu_environment(),
                    stdout=log, stderr=subprocess.STDOUT, check=False)
            if process.returncode or not status(root=root, launch_root=launch_root)['complete']:
                raise RuntimeError('CPU comparison lacks rc0/same-lock saved CSV/provenance; no retry')
            receipt.update(state='complete', exit_status=0, completed_utc=utc())
        except BaseException as error:
            receipt.update(state='failed', exit_status=1, error=str(error), completed_utc=utc())
            raise
        finally:
            save(path, receipt)


def publication_paths(version, root, launch_root, git_root):
    if version != 'R104' or Path(root).resolve() != ROOT.resolve() or Path(launch_root).resolve() != LAUNCH.resolve():
        raise RuntimeError('Publication accepts only canonical R104 roots')
    root, launch_root, git_root = (Path(p).resolve() for p in (root, launch_root, git_root))
    ready = optional(launch_root / ('R104_' + ARMS[0]) / 'ready.json')
    require_ready(root, launch_root, ready.get('evidence'))
    # Frozen plans name exact source files, not a future-changing version glob.
    from .r104_launch import PUBLICATION_EXTRAS
    candidates = [REPO / name for name in PUBLICATION_EXTRAS]
    for arm in ARMS:
        output = launch_root / ('R104_' + arm)
        plan = optional(output / 'plan.json')
        candidates += [Path(p) for p in plan.get('source_sha256', {})
            if Path(p).parent == REPO / 'code/V4'
            and Path(p).name.startswith(('r104_', 'test_r104_', 'run_v4_r104_'))]
        candidates += [launch_root / 'specs' / ('R104_' + arm + '.json')]
        candidates += [output / name for name in ('plan.json', 'ready.json', 'controller.json', 'worker.json')]
    for run in (root, *(root / arm for arm in ARMS)):
        candidates += [p for p in run.iterdir() if p.suffix.lower() in {'.json', '.csv', '.pdf', '.md'}
            and not p.name.startswith(('goal_', 'github_', 'wandb', 'prediction', 'outcome'))]
        for directory in ('preparation', 'full_model_statistics', 'comparison_artifacts'):
            for parent, dirs, files in os.walk(run / directory):
                dirs[:] = [n for n in dirs if n not in adapter.PRUNED
                    and n not in {'sources', 'runtime', 'cache', 'teacher', 'teacher_cache', 'donor_cache'}
                    and not (Path(parent) / n).is_symlink()]
                candidates += [Path(parent) / n for n in files if Path(n).suffix.lower() in {'.json', '.csv', '.pdf', '.md'}]
    result = set()
    for path in candidates:
        if (not path.is_file() or path.is_symlink() or any(p.is_symlink() for p in path.parents)
                or path.name.lower().startswith(('prediction', 'outcome', 'cost_array', 'winner_array', 'raw_cost', 'labels'))
                or 'wandb' in path.name.lower()):
            continue
        resolved = path.resolve()
        if not resolved.is_relative_to(REPO.resolve()) or not resolved.is_relative_to(git_root):
            raise RuntimeError('Curated R104 asset escapes repository')
        if path.stat().st_size > adapter.MAX_ASSET_BYTES:
            raise RuntimeError('Curated R104 asset exceeds32MiB')
        if path.suffix.lower() == '.json' and not adapter._safe_json(optional(path)):
            continue
        result.add(str(resolved.relative_to(git_root)))
    if not result:
        raise RuntimeError('No curated R104 assets')
    return sorted(result)


# Only the private administrative instance is adapted, never a live watcher.
adapter.VERSIONS = dict(R104=ROOT.name)
adapter.model_root = lambda version: ROOT
adapter.status = status
adapter.publication_paths = publication_paths


def publish(root=ROOT, launch_root=LAUNCH, evidence=None):
    require_ready(root, launch_root, evidence)
    return adapter.publish('R104', root, launch_root)


def watch(root=ROOT, launch_root=LAUNCH, interval=120, publish_github=False, evidence=None, **references):
    if interval <= 0:
        raise ValueError('Watcher interval must be positive')
    root = Path(root).resolve()
    require_ready(root, launch_root, evidence)
    with (root / 'goal_watch.lock').open('a') as claim:
        adapter.fcntl.flock(claim, adapter.fcntl.LOCK_EX | adapter.fcntl.LOCK_NB)
        path = root / 'goal_watch_receipt.json'
        if path.exists():
            prior = optional(path)
            if prior.get('state') == 'complete' and prior.get('exit_status') == 0:
                return 0
            raise RuntimeError('Watcher already attempted; no replacement or retry')
        receipt = dict(experiment='R104', state='watching', pid=os.getpid(), started_utc=utc(),
            cpu_only=True, automatic_retry=False, github_on_completion=publish_github,
            main_ready_evidence=evidence, goal_complete=False, goal_status='active_unresolved')
        save(path, receipt)
        try:
            while True:
                require_ready(root, launch_root, evidence)
                current = status(root=root, launch_root=launch_root, persist=True)
                if current['errors']:
                    raise RuntimeError('R104 saved stage failure: ' + json.dumps(current['errors']))
                if current['all_arms_ready'] and not current['comparison_complete']:
                    compare_once(root, launch_root, evidence, **references)
                    continue
                if current['complete']:
                    if publish_github:
                        publish(root, launch_root, evidence)
                    receipt.update(state='complete', exit_status=0, pipeline_complete=True, completed_utc=utc())
                    save(path, receipt)
                    print(json.dumps(dict(experiment='R104', phase='complete', goal_complete=False)), flush=True)
                    return 0
                time.sleep(interval)
        except BaseException as error:
            receipt.update(state='stopped_on_error', exit_status=1, error=str(error), completed_utc=utc())
            save(path, receipt)
            print(json.dumps(dict(experiment='R104', phase='failed', error=str(error))), flush=True)
            return 1


def parser():
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument('--watch', action='store_true')
    value.add_argument('--publish-github', action='store_true')
    value.add_argument('--root', type=Path, default=ROOT)
    value.add_argument('--launch-root', type=Path, default=LAUNCH)
    value.add_argument('--interval', type=int, default=120)
    value.add_argument('--evidence')
    for key in REFERENCE_FLAGS:
        value.add_argument('--' + key.replace('_', '-'), type=Path)
    return value


def main(argv=None):
    args = parser().parse_args(argv)
    if args.watch:
        return watch(args.root, args.launch_root, args.interval, args.publish_github, args.evidence,
            **{key: getattr(args, key) for key in REFERENCE_FLAGS})
    if args.publish_github:
        publish(args.root, args.launch_root, args.evidence)
    else:
        print(json.dumps(status(root=args.root, launch_root=args.launch_root), indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

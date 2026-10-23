"""Quiet future R102 whole-version watcher, private R98 administrative adapter.

No launch, inference, arrays, weights, transfer or Git runs on import/status.
Main must explicitly authorize watching/comparing/publishing after source ready.
Only saved-evidence CPU comparison may run, once BOTH arm pipelines finish.
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
ROOT = REPO / 'code/V4/runs/R102_full_type_pretrained_fusion'
LAUNCH = REPO / 'code/V4/runs/R98_parallel_dispatch'
ARMS = ('A', 'B')
REFERENCE_FLAGS = ('nss_root', 'historical_root', 'r94_root')
PROJECTION_SHAPES = {'instance_encoder.nco_projections.' + name + '.weight': [128, width]
                     for name, width in (('TSP', 128), ('ATSP', 512), ('VRP', 128))}


def private():
    spec = importlib.util.spec_from_file_location('code.V4._r102_goal_admin', REPO / 'code/V4/r98_goal.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


adapter = private()
PROBLEMS, optional, save, utc = adapter.PROBLEMS, adapter.optional, adapter.save, adapter.utc
file_hash, json_hash = adapter.file_hash, adapter.json_hash
_old_common, _old_status = adapter.common_checks, adapter.status


def timing_ready(record, lock):
    if (not adapter.same_state(record, lock) or record.get('complete') is not True
            or record.get('checkpoint_sha256') != lock.get('checkpoint_sha256')
            or record.get('inference_rule') != lock.get('inference_rule')
            or record.get('includes_native_encoder') is not True or record.get('online_donor_verified') is not True
            or record.get('all18_donor_coverage') is not True or record.get('cache_hit_is_total') is not False
            or record.get('repetitions') != 3 or not record.get('scope') or not record.get('excludes')):
        return False
    rows = record.get('per_problem', [])
    if (not isinstance(rows, list) or len(rows) != len(PROBLEMS)
            or any(not isinstance(row, dict) for row in rows)
            or {row.get('problem') for row in rows} != set(PROBLEMS)):
        return False
    try:
        for row in rows:
            if (row.get('instances') != 128 or row.get('online_donor_verified') is not True
                    or row.get('online_fields_removed') != ['nco_nodes', 'nco_node_mask']):
                return False
            for name, calls in (('donor_plus_selector', 4), ('cache_hit_selector', 0)):
                measurement = row.get(name, {})
                samples, value = measurement.get('batch_seconds', []), measurement.get('mean_seconds_per_instance')
                if (not isinstance(samples, list) or len(samples) != 3
                        or measurement.get('donor_forward_calls') != calls or measurement.get('warmup_batches') != 1
                        or any(isinstance(v, bool) or not math.isfinite(float(v)) or float(v) <= 0
                               for v in [*samples, value])
                        or not math.isclose(float(value), sum(samples) / (3 * 128), rel_tol=1e-6, abs_tol=1e-12)):
                    return False
    except (AttributeError, TypeError, ValueError):
        return False
    return True


def arm_checks(version, arm, run, records):
    checks = adapter.r99_checks(arm, run, records)
    config, lock = records['protocol.json'], records['final_test_lock.json']
    report, post = records['full_model_statistics/protocol.json'], records['postprocess_status.json']
    init = optional(run / 'initialization.json')
    checks['unique_VAL_lock'] &= lock.get('selection_split') == 'val'
    checks['external_pretraining_scope'] = all(
        record.get('external_solver_pretraining') is True
        and record.get('extra_type_pretraining') is (arm == 'B')
        for record in (config, init, report))
    donors = config.get('donor_provenance', {}).get('components', {})
    extra_variant = 'random' if arm == 'A' else 'pretrained'
    checks['R102_fresh_full_type'] = (config.get('r102_schema_version') == 1
        and config.get('initialization') == 'fresh_r84_plus_full_type_fusion'
        and config.get('donor_variant') == {'A': 'extra_random', 'B': 'full_pretrained'}[arm]
        and set(donors) == {'TSP', 'ATSP', 'VRP'}
        and all(isinstance(donors.get(k), dict) and donors[k].get('variant') == v
                for k, v in (('TSP', extra_variant), ('ATSP', extra_variant), ('VRP', 'pretrained'))))
    checks['fresh_three_projections'] = (init.get('historical_selector_checkpoint_loaded') is False
        and init.get('trainable_parameter_count') == config.get('trainable_parameter_count') == 1941257
        and init.get('new_parameter_count') == config.get('new_parameter_count') == 98304
        and init.get('removed_parameter_names') == [] and bool(init.get('common_parameter_sha256'))
        and init.get('common_parameter_sha256') == config.get('retained_parameter_sha256')
        and init.get('new_parameter_sha256') == config.get('new_parameter_sha256')
        and config.get('projection_shapes') == PROJECTION_SHAPES
        and init.get('new_parameter_shapes') == PROJECTION_SHAPES)
    checks['full_TRAIN'] = adapter.split_ready(records['final_metrics.json'].get('train', {}), 81000, lock)
    checks['native_LIB49_100'] = records['lib_stage_complete.json'].get('native_counts') == dict(TSPLIB=49, CVRPLIB=100)
    checks['CPU_report'] = (post.get('complete') is True and post.get('phase') == 'complete'
        and post.get('exit_status') == 0 and report.get('experiment') == 'R102'
        and report.get('complete') is True and report.get('checkpoint_lock') == lock)
    checks['online_full18_timing'] = timing_ready(records['inference_timing.json'], lock)
    checks['fixed36'] = (report.get('target_identity', {}).get('target_count') == 36
        and bool(report.get('target_identity', {}).get('sha256')))
    return checks


def status(version='R102', root=None, launch_root=LAUNCH, *, persist=False):
    if version != 'R102':
        raise ValueError('This private watcher owns only R102')
    root, launch_root = Path(root or ROOT).resolve(), Path(launch_root).resolve()
    result = _old_status('R102', root, launch_root, persist=False)
    configs, locks, timings = {}, {}, {}
    for row in result['arms']:
        run, arm = root / row['arm'], row['arm']
        configs[arm], locks[arm] = optional(run / 'protocol.json'), optional(run / 'final_test_lock.json')
        row['checks'].pop('RESULTS.pdf', None)
        spec = optional(launch_root / 'specs' / ('R102_' + arm + '.json'))
        expected = ('input_features', 'train', 'TEST', 'timing', 'LIB', 'PDF')
        row['checks']['exact_pipeline'] = ([(s.get('name'), s.get('device')) for s in spec.get('stages', [])]
            == [(name, 'cpu' if name == 'PDF' else 'gpu') for name in expected]
            and spec.get('slots') == [{'A': '4090_0', 'B': '4090_1'}[arm]]
            and spec.get('routing_feature_root') == '/tmp/shiys_R94_B_node_features'
            and spec.get('routing_mode') == 'reuse_original_R94B')
        row['missing'] = [key for key, value in row['checks'].items() if not value]
        row['ready'] = all(row['checks'].values())
        timing = run / 'inference_timing.json'
        timings[arm] = file_hash(timing) if timing.is_file() else None
        result['errors'].extend(adapter.receipt_errors(run / 'initialization.json', optional(run / 'initialization.json')))
    match_keys = ('data_root', 'prepared_root', 'rows_per_problem', 'candidate_pools', 'baselines',
        'model_spec', 'max_successful_updates', 'inference_rule', 'retained_parameter_sha256',
        'new_parameter_sha256', 'trainable_parameter_count', 'projection_shapes', 'main_protocol_lock')
    matched = (bool(configs['A']) and bool(configs['B'])
        and all(configs['A'].get(key) is not None and configs['A'].get(key) == configs['B'].get(key) for key in match_keys)
        and bool(configs['A'].get('donor_provenance', {}).get('components', {}).get('VRP'))
        and configs['A'].get('donor_provenance', {}).get('components', {}).get('VRP')
            == configs['B'].get('donor_provenance', {}).get('components', {}).get('VRP'))
    comparison = optional(root / 'comparison_artifacts/protocol.json')
    reports = {arm: optional(root / arm / 'full_model_statistics/protocol.json') for arm in ARMS}
    compared = (comparison.get('experiment') == 'R102' and comparison.get('complete') is True
        and comparison.get('exit_status') == 0 and comparison.get('checkpoints') == locks
        and comparison.get('target_identity', {}).get('target_count') == 36
        and all(bool(timings[arm]) and comparison.get('inference_timing', {}).get(arm, {}).get('sha256') == timings[arm]
                and comparison.get('target_identity') == reports[arm].get('target_identity') for arm in ARMS)
        and adapter.pdf_ready(root / 'comparison_artifacts/COMPARISON.pdf') and adapter.pdf_ready(root / 'RESULTS.pdf'))
    ready = all(row['ready'] for row in result['arms']) and matched
    complete = ready and compared and not result['errors']
    result.update(all_arms_ready=ready, matched_controls=matched, comparison_complete=compared, complete=complete,
        phase='failed' if result['errors'] else 'complete' if complete else 'waiting_for_comparison' if ready else 'waiting_for_arms',
        goal_complete=False, goal_status='active_unresolved')
    if persist:
        path, prior = root / 'goal_progress.json', optional(root / 'goal_progress.json')
        if {k: v for k, v in prior.items() if k != 'updated_utc'} == result:
            return prior
        result['updated_utc'] = utc()
        save(path, result)
    return result


def require_ready(root, launch_root, evidence):
    """Check explicit source readiness without opening any checkpoint/array."""
    if not isinstance(evidence, str) or not evidence.strip():
        raise ValueError('Main must explicitly supply team-ready evidence before watch/compare/publication')
    for arm in ARMS:
        output = Path(launch_root) / ('R102_' + arm)
        plan, ready = optional(output / 'plan.json'), optional(output / 'ready.json')
        spec = optional(Path(launch_root) / 'specs' / ('R102_' + arm + '.json'))
        if (not plan or not ready or ready.get('main_thread_ready') is not True
                or ready.get('runner_tests_passed') is not True or ready.get('evidence') != evidence
                or plan.get('evidence') != evidence or plan.get('spec') != spec
                or spec.get('run_dir') != str(Path(root).resolve() / arm)
                or ready.get('plan_sha256') != file_hash(output / 'plan.json')
                or not plan.get('source_sha256')):
            raise RuntimeError('Both arm plans must be explicitly source-frozen by main before watching')
        for name, digest in plan['source_sha256'].items():
            # The launcher validates R84 init identity; this CPU watcher never opens .pt.
            path = Path(name)
            if path.suffix in ('.py', '.sh', '.json', '.md') and (not path.is_file() or file_hash(path) != digest):
                raise RuntimeError('Source changed since readiness; no watcher restart or automatic refreeze')


def compare_once(root=ROOT, launch_root=LAUNCH, evidence=None, **references):
    root = Path(root).resolve()
    require_ready(root, launch_root, evidence)
    current = status(root=root, launch_root=launch_root)
    if current['errors'] or not current['all_arms_ready'] or current['comparison_complete']:
        raise RuntimeError('CPU comparison requires BOTH complete matched full18 pipelines, once only')
    path = root / 'goal_compare_receipt.json'
    with (root / 'goal_compare.lock').open('a') as claim:
        adapter.fcntl.flock(claim, adapter.fcntl.LOCK_EX | adapter.fcntl.LOCK_NB)
        if path.exists():
            raise RuntimeError('Comparison already attempted; preserve receipt, no retry')
        argv = [sys.executable, '-B', '-u', '-m', 'code.V4.r102_reporting', 'compare',
                '--root', str(root), '--device', 'cpu']
        for key in REFERENCE_FLAGS:
            if references.get(key) is not None:
                argv.extend(('--' + key.replace('_', '-'), str(Path(references[key]).resolve())))
        receipt = dict(state='running', started_utc=utc(), argv=argv, cpu_only=True,
                       automatic_retry=False, main_ready_evidence=evidence)
        save(path, receipt)
        try:
            with (root / 'goal_compare.log').open('a') as log:
                process = subprocess.run(argv, cwd=REPO, env=adapter.cpu_environment(),
                    stdout=log, stderr=subprocess.STDOUT, check=False)
            if process.returncode or not status(root=root, launch_root=launch_root)['complete']:
                raise RuntimeError('Comparison rc0 and same-lock timing/fixed36 PDFs are required; no retry')
            receipt.update(state='complete', exit_status=0, completed_utc=utc())
        except BaseException as error:
            receipt.update(state='failed', exit_status=1, error=str(error), completed_utc=utc())
            raise
        finally:
            save(path, receipt)


def publication_paths(version, root, launch_root, git_root):
    root, launch_root, git_root = (Path(p).resolve() for p in (root, launch_root, git_root))
    if version != 'R102' or root != ROOT.resolve() or launch_root != LAUNCH.resolve():
        raise RuntimeError('Publication accepts only canonical R102 roots')
    from .r98_launch import source_closure
    code = REPO / 'code/V4'
    sources = [p for pattern in ('r102_*.py', 'test_r102*.py', 'run_v4_r102*.sh', 'r102*.md') for p in code.glob(pattern)]
    sources += [code / name for name in ('r98_launch.py', 'r98_goal.py', 'r84_launch.py', 'r81_queue.py',
        'r101_goal.py', 'r80_reporting.py', 'r84_reporting.py', 'r85_reporting.py', 'r87_reporting.py',
        'run_v4_r98_launch.sh', 'run_v4_r84_launch.sh', 'run_v4_r81.sh') if (code / name).is_file()]
    # Local Python dependencies are needed to reproduce the private adapters.
    candidates = [Path(p) for p in source_closure(sources)]
    candidates += [REPO / 'docs/research' / name for name in
        ('pretrained_representation_coverage_tsp_atsp.md', 'post_r100_broad_direction.md')]
    pruned = adapter.PRUNED | {'teacher_cache', 'teacher', 'sources', 'runtime', 'routing_cache',
        'routing_export', 'export', 'exports', 'cache', 'pretrained', 'donor_cache'}
    for run in (root, *(root / arm for arm in ARMS)):
        candidates += [p for p in run.glob('*') if p.suffix.lower() in {'.json', '.csv', '.pdf', '.md'}
            and not p.name.startswith(('goal_', 'github_', 'wandb', 'prediction', 'outcome'))]
        for directory in ('preparation', 'prep', 'full_model_statistics', 'comparison_artifacts'):
            for parent, dirs, files in os.walk(run / directory):
                dirs[:] = [name for name in dirs if name not in pruned and not (Path(parent) / name).is_symlink()]
                candidates += [Path(parent) / name for name in files
                    if Path(name).suffix.lower() in {'.json', '.csv', '.pdf', '.md'}]
    for arm in ARMS:
        candidates.append(launch_root / 'specs' / ('R102_' + arm + '.json'))
        candidates += [launch_root / ('R102_' + arm) / name for name in
                       ('plan.json', 'ready.json', 'worker.json', 'controller.json')]
    result = set()
    for path in candidates:
        if (not path.is_file() or path.is_symlink() or 'wandb' in path.name.lower()
                or path.name.lower().startswith(('prediction', 'outcome', 'cost_array', 'winner_array', 'raw_cost', 'labels'))
                or any(parent.is_symlink() for parent in path.parents if parent != git_root)):
            continue
        resolved = path.resolve()
        if not resolved.is_relative_to(REPO.resolve()) or not resolved.is_relative_to(git_root):
            raise RuntimeError('Curated source/report escapes repository')
        if path.stat().st_size > adapter.MAX_ASSET_BYTES:
            raise RuntimeError('Curated source/report unexpectedly exceeds32MiB')
        if path.suffix.lower() == '.json' and not adapter._safe_json(optional(path)):
            continue
        result.add(str(resolved.relative_to(git_root)))
    if not result:
        raise RuntimeError('No curated R102 assets')
    return sorted(result)


# Only this private administrative module is adapted, not any live watcher.
adapter.VERSIONS = dict(R102=ROOT.name)
adapter.arms = lambda version: ARMS
adapter.run_root = lambda version, root, arm: root / arm
adapter.common_checks = lambda version, arm, run, records: _old_common('R102', arm, run, records)
adapter.frozen_checks = arm_checks
adapter.FROZEN_PDFS = adapter.R99_PDFS
adapter.status = status
adapter.publication_paths = publication_paths


def publish(root=ROOT, launch_root=LAUNCH, evidence=None):
    require_ready(root, launch_root, evidence)
    return adapter.publish('R102', root, launch_root)


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
            raise RuntimeError('Watcher already attempted; no replacement or timeout takeover')
        receipt = dict(experiment='R102', state='watching', pid=os.getpid(), started_utc=utc(), cpu_only=True,
            github_on_completion=publish_github, main_ready_evidence=evidence, automatic_retry=False,
            goal_complete=False, goal_status='active_unresolved')
        save(path, receipt)
        try:
            while True:
                current = status(root=root, launch_root=launch_root, persist=True)
                if current['errors']:
                    raise RuntimeError('R102 stage failed; no repair, model action, watcher replacement or retry')
                if current['all_arms_ready'] and not current['comparison_complete']:
                    compare_once(root, launch_root, evidence, **references)
                    continue
                if current['complete']:
                    if publish_github:
                        publish(root, launch_root, evidence)
                    receipt.update(state='complete', exit_status=0, pipeline_complete=True, completed_utc=utc())
                    save(path, receipt)
                    print(json.dumps(dict(experiment='R102', phase='complete', pipeline_complete=True,
                                          goal_complete=False)), flush=True)
                    return 0
                # Do not print periodic progress or missing-artifact changes.
                time.sleep(interval)
        except BaseException as error:
            receipt.update(state='stopped_on_error', exit_status=1, error=str(error), completed_utc=utc())
            save(path, receipt)
            print(json.dumps(dict(experiment='R102', phase='failed', error=str(error))), flush=True)
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

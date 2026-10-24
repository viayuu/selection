"""Quiet CPU R108 whole-version gate and explicit one-shot GitHub watcher.

Only saved JSON/PDF evidence is read by status. No model, tensor, hardware,
launch or Git action is performed on import/status. No routing diagnostic.
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
ROOT = REPO / 'code/V4/runs/R108_conditional_density_ratio'
LAUNCH = ROOT / 'preparation/dispatch'
ARMS = ('A_prior_ce', 'B_density_ratio')
REFERENCE_FLAGS = ('nss_root', 'historical_root', 'r89_root', 'r89a_root', 'r94b_root')
RECORD_NAMES = ('protocol.json', 'status.json', 'initialization.json', 'training_complete.json',
    'final_test_lock.json', 'final_metrics.json', 'test_results.json', 'inference_timing.json',
    'lib_stage_complete.json', 'lib_stage_status.json', 'postprocess_status.json',
    'full_model_statistics/protocol.json')
TRAINABLE_PARAMETERS = 1842953
BOUNDARY = ('Same scratch R84/R88B model, fixed TRAIN-only (problem, exactN) prior buffers '
    'and 1,842,953 trainables; A pure CE(logpi+t), B product-noise binary density ratio. '
    'Single forward/dropout0.1, no pair/risk/RDrop. Canonical ratio normalization differs; '
    'CE common-shift gauge cancels shared last-bias data gradient; ratio anchors origin, '
    'so identical forward/parameters do not imply identical active gradient freedom; '
    'no equal gradient-scale, equal-FLOPs or selection-improvement claim. '
    'Pipeline completion is not evidence of selection improvement.')


def private():
    spec = importlib.util.spec_from_file_location('code.V4._r108_goal_admin', REPO / 'code/V4/r105_goal.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


admin = private()
adapter = admin.adapter
PROBLEMS, optional, save, utc = admin.PROBLEMS, admin.optional, admin.save, admin.utc
file_hash, json_hash = admin.file_hash, admin.json_hash
receipt_errors, read_record = admin.receipt_errors, admin.read_record
strict_split, selected_train_ready = admin.strict_split, admin.selected_train_ready
stages_ready = admin.stages_ready


def val_binding_ready(final, config, lock, run):
    val = final.get('val', {})
    bound, identity = val.get('checkpoint_binding', {}), val.get('identity', {})
    capture = val.get('archive_capture_identity', {})
    return (strict_split(val, 4500, lock) and bool(config.get('selected_val_binding_rule'))
        and bound.get('rule') == config['selected_val_binding_rule']
        and bound.get('checkpoint_sha256') == identity.get('checkpoint_sha256') == lock.get('checkpoint_sha256')
        and bound.get('state_hash') == identity.get('state_hash') == lock.get('state_hash')
        and bound.get('selected_update') == identity.get('successful_updates') == lock.get('selected_update')
        and bool(bound.get('model_state_sha256'))
        and bound['model_state_sha256'] == identity.get('model_state_sha256') == capture.get('model_state_sha256')
        and bound.get('archive_identity_unchanged') is True
        and capture.get('experiment') == 'R108' and capture.get('arm') == lock.get('arm')
        and capture.get('split') == 'val' and 'checkpoint_sha256' in capture
        and capture['checkpoint_sha256'] in (None, lock.get('checkpoint_sha256'))
        and 'archive_capture_checkpoint_sha256' in bound
        and bound['archive_capture_checkpoint_sha256'] == capture['checkpoint_sha256']
        and identity == dict(capture, checkpoint_sha256=lock.get('checkpoint_sha256'))
        and capture == optional(run / 'evaluation/val/selected_best/identity.json'))


def timing_ready(record, lock, config):
    count = config.get('trainable_parameter_count')
    if (type(count) is not int or count <= 0 or record.get('parameter_count') != count
            or not adapter.same_state(record, lock) or record.get('complete') is not True
            or record.get('experiment') != 'R108' or record.get('arm') != lock.get('arm')
            or record.get('checkpoint_sha256') != lock.get('checkpoint_sha256')
            or record.get('selected_update') != lock.get('selected_update')
            or record.get('inference_rule') != lock.get('inference_rule')
            or record.get('repetitions') != 3 or record.get('warmup_batches') != 1
            or record.get('all_current_model_included') is not True
            or record.get('prior_lookup_and_residual_computation_included') is not True
            or record.get('donor_present') is not False or record.get('cached_feature_latency_used') is not False
            or record.get('input_labels_read') is not False or record.get('end_to_end_measured') is not False
            or not str(record.get('device', '')).startswith('cuda')
            or record.get('scope') != 'complete_standalone_model_original_GPU_resident_inputs'
            or not record.get('excludes')):
        return False
    rows = record.get('per_problem', [])
    if (not isinstance(rows, list) or len(rows) != len(PROBLEMS)
            or any(not isinstance(row, dict) for row in rows)
            or {row.get('problem') for row in rows} != set(PROBLEMS)):
        return False
    for row in rows:
        measurement = row.get('standalone_model', {})
        samples, mean = measurement.get('batch_seconds'), measurement.get('mean_seconds_per_instance')
        if (row.get('instances') != 128 or not isinstance(samples, list) or len(samples) != 3
                or any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in [*samples, mean])
                or not math.isclose(mean, sum(samples) / (3 * 128), rel_tol=1e-6, abs_tol=1e-12)):
            return False
    return True


def arm_checks(arm, run, records, main_lock_sha):
    checks = adapter.r99_checks(arm, run, records)
    checks.update(adapter.common_checks('R108', arm, run, records))
    config, lock = records['protocol.json'], records['final_test_lock.json']
    final, report = records['final_metrics.json'], records['full_model_statistics/protocol.json']
    post, selected = records['postprocess_status.json'], lock.get('selected_update')
    rule = str(lock.get('rule', '')).lower()
    checks['unique_VAL_lock'] &= (lock.get('selection_split', 'val') == 'val'
        and type(selected) is int and 0 < selected <= 57600 and selected % 1440 == 0
        and bool(config.get('inference_rule')) and lock.get('inference_rule') == config.get('inference_rule')
        and lock.get('rule') == config.get('checkpoint_rule')
        and all(word in rule for word in ('strict minimum', 'val', 'macro_vs_sbs', 'first')))
    checks['full_TRAIN'] = selected_train_ready(final, config, lock)
    checks['full_VAL'] &= val_binding_ready(final, config, lock, run)
    checks['full_TEST'] &= strict_split(records['test_results.json'].get('result', {}), 4500, lock)
    checks['CPU_report'] = (post.get('complete') is True and post.get('phase') == 'complete'
        and type(post.get('exit_status')) is int and post['exit_status'] == 0
        and report.get('complete') is True and report.get('experiment') == 'R108'
        and report.get('checkpoint_lock') == lock)
    checks['main_protocol_lock'] = (bool(main_lock_sha)
        and config.get('main_protocol_lock', {}).get('sha256') == main_lock_sha)
    lib = records['lib_stage_complete.json']
    checks['LIB'] = (lib.get('experiment') == 'R108' and lib.get('complete') is True
        and lib.get('checkpoint_sha256') == lock.get('checkpoint_sha256')
        and set(lib.get('datasets', [])) == {'TSPLIB', 'CVRPLIB'}
        and lib.get('exit_status', 0) == 0 and lib.get('state_hash', lock.get('state_hash')) == lock.get('state_hash'))
    timing_path = run / 'inference_timing.json'
    checks['timing_bound_to_report'] = (timing_path.is_file()
        and (report.get('inference_timing') == records['inference_timing.json']
            or report.get('inference_timing', {}).get('sha256') == file_hash(timing_path)))
    checks['all18_timing'] = timing_ready(records['inference_timing.json'], lock, config)
    checks['timing_selected_model'] = (bool(final.get('val', {}).get('identity', {}).get('model_state_sha256'))
        and records['inference_timing.json'].get('model_state_sha256')
            == final.get('val', {}).get('identity', {}).get('model_state_sha256'))
    checks['fixed36'] = (report.get('target_identity', {}).get('target_count') == 36
        and bool(report.get('target_identity', {}).get('sha256')))
    prior = config.get('priors_sha256')
    checks['TRAIN_prior_provenance'] = (isinstance(prior, str) and len(prior) == 64
        and config.get('priors_path') == str(run.parent / 'prepared/priors.json')
        and config.get('prior_spec') == report.get('prior_spec') and bool(config.get('prior_spec'))
        and report.get('prior_provenance', {}).get('sha256') == prior
        and report.get('prior_provenance', {}).get('fit_split') == 'train'
        and records['initialization.json'].get('priors_sha256') == prior
        and records['initialization.json'].get('prior_fit_split') == 'train')
    checks['same_model_NSS4'] = nss4_ready(run, lock)
    return checks


def nss4_ready(run, lock):
    import csv
    path = run / 'full_model_statistics/nss_comparison.csv'
    if not path.is_file():
        return False
    with path.open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    fields = ('current_top1_native_pct', 'current_mean_cost', 'nss_top1_native_pct',
        'nss_mean_cost', 'status', 'data_version', 'candidate_pool', 'winner_rule')
    return (len(rows) == 4 and {r.get('dataset') for r in rows}
        == {'TSP TEST', 'CVRP TEST', 'TSPLIB', 'CVRPLIB'}
        and all(r.get('current_checkpoint_sha256') == lock.get('checkpoint_sha256')
            and all(k in r for k in fields) and bool(r['status']) for r in rows))


def status(version='R108', root=None, launch_root=LAUNCH, *, persist=False):
    if version != 'R108':
        raise ValueError('This private watcher owns only R108')
    root, launch_root = Path(root or ROOT).resolve(), Path(launch_root).resolve()
    errors, rows, configs, locks, inits = [], [], {}, {}, {}
    main_path = root / 'protocol_lock.json'
    main_sha = file_hash(main_path) if main_path.is_file() else None
    science = read_record(main_path, errors)
    for arm, slot in zip(ARMS, ('4090_0', '4090_1')):
        run, identifier = root / arm, 'R108_' + arm
        records = {name: read_record(run / name, errors) for name in RECORD_NAMES}
        configs[arm], locks[arm], inits[arm] = (records[k] for k in
            ('protocol.json', 'final_test_lock.json', 'initialization.json'))
        output = launch_root / identifier
        spec = read_record(launch_root / 'specs' / (identifier + '.json'), errors)
        controller = read_record(output / 'controller.json', errors)
        worker = read_record(output / 'worker.json', errors)
        for name in ('queue_receipt.json', 'status.json'):
            read_record(output / name, errors)
        try:
            checks = arm_checks(arm, run, records, main_sha)
            init, config = inits[arm], configs[arm]
            count = science.get('trainable_parameters', {}).get(arm)
            checks['main_science'] = (science.get('experiment') == 'R108' and science.get('locked') is True
                and science.get('arms') == list(ARMS) and science.get('seed') == 2)
            checks['parameter_count'] = (type(count) is int and count == TRAINABLE_PARAMETERS
                and config.get('trainable_parameter_count') == count
                and init.get('trainable_parameter_count') == count)
            checks['fresh_initialization'] = (init.get('fresh_initialization') is True
                and init.get('historical_checkpoint_loaded') is False and init.get('state_dict_restored') is False
                and init.get('seed') == 2
                and init.get('trainable_sha256') == science.get('original_fresh_trainable_sha256')
                and bool(init.get('trainable_sha256'))
                and init.get('trainable_sha256') == init.get('fresh_trainable_sha256')
                and bool(init.get('trainable_sha256'))
                and init.get('state_sha256') == init.get('fresh_state_sha256')
                    == init.get('common_fresh_model_state_sha256')
                and bool(init.get('state_sha256')) and init.get('cross_arm_complete_state_verified') is True
                and init.get('identical_complete_fresh_parameters') is True
                and init.get('active_functional_freedom_matched') is False
                and init.get('new_trainable_parameter_count') == 0)
            expected = [(n, 'cpu' if n == 'PDF' else 'gpu') for n in ('train', 'TEST', 'timing', 'LIB', 'PDF')]
            checks.update(launch_spec=spec.get('run_id') == identifier and spec.get('run_dir') == str(run)
                and spec.get('slots') == [slot] and spec.get('spec_draft') is False
                and [(s.get('name'), s.get('device')) for s in spec.get('stages', [])] == expected,
                controller_rc0=stages_ready(controller, ['allocation_GPU_pipeline', 'PDF']),
                worker_all_GPU_stages_rc0=stages_ready(worker, ['train', 'TEST', 'timing', 'LIB']))
        except (OSError, AttributeError, KeyError, TypeError, ValueError) as error:
            checks = dict(valid_receipts=False)
            errors.append(dict(path=str(run), error='Malformed completion evidence: ' + str(error)))
        checks.update({name: adapter.pdf_ready(run / 'full_model_statistics' / name) for name in adapter.R99_PDFS})
        rows.append(dict(arm=arm, run_id=identifier, root=str(run), checks=checks,
            ready=all(checks.values()), missing=[k for k, v in checks.items() if not v]))
    match_keys = ('data_root', 'prepared_root', 'rows_per_problem', 'candidate_pools', 'baselines',
        'max_successful_updates', 'inference_rule', 'main_protocol_lock', 'trainable_parameter_count',
        'microbatch_caps', 'eval_batch_size', 'prior_spec', 'priors_sha256', 'priors_path', 'loss')
    matched = all(configs[ARMS[0]].get(k) is not None
        and configs[ARMS[0]].get(k) == configs[ARMS[1]].get(k) for k in match_keys)
    identical_init = all(bool(inits[ARMS[0]].get(k))
        and inits[ARMS[0]].get(k) == inits[ARMS[1]].get(k)
        for k in ('trainable_sha256', 'state_sha256', 'common_fresh_model_state_sha256'))
    comparison = read_record(root / 'comparison_artifacts/protocol.json', errors)
    for name in ('goal_watch_receipt.json', 'goal_compare_receipt.json', 'github_publication.json'):
        read_record(root / name, errors)
    compared = (comparison.get('experiment') == 'R108' and comparison.get('complete') is True
        and comparison.get('exit_status', 0) == 0 and comparison.get('checkpoints') == locks
        and comparison.get('training') is False and comparison.get('inference') is False
        and comparison.get('identical_complete_fresh_parameters') is True
        and comparison.get('total_parameter_matched') is True
        and comparison.get('active_parameter_matched') is False and comparison.get('equal_flops_claim') is False
        and comparison.get('equal_gradient_scale_claim') is False
        and comparison.get('priors_sha256') == configs[ARMS[0]].get('priors_sha256')
        and all((root / 'comparison_artifacts' / name).is_file()
                and (root / 'comparison_artifacts' / name).stat().st_size > 0 for name in
                ('paired_changes_B_vs_A.csv', 'target_recovery_B_vs_A.csv', 'target_recovery_summary_B_vs_A.csv')))
    ready = matched and identical_init and all(row['ready'] for row in rows)
    complete = ready and compared and not errors
    result = dict(experiment='R108', arms=rows, errors=errors, matched_controls=matched,
        identical_full_initialization=identical_init, all_arms_ready=ready,
        comparison_complete=compared, complete=complete,
        phase='failed' if errors else 'complete' if complete else 'waiting_for_comparison' if ready else 'waiting_for_arms',
        cpu_only=True, automatic_retry=False, goal_complete=False, goal_status='active_unresolved',
        comparison_boundary=BOUNDARY)
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
    from . import r108_launch as launch
    if Path(root).resolve() != launch.MODEL_ROOT.resolve() or Path(launch_root).resolve() != launch.ROOT.resolve():
        raise RuntimeError('Watcher requires canonical main-ready R108 roots')
    for arm in ARMS:
        spec, plan = launch.check_ready(Path(launch_root) / 'specs' / ('R108_' + arm + '.json'))
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
        argv = [sys.executable, '-B', '-u', '-m', 'code.V4.r108_reporting', 'compare', '--root', str(root), '--device', 'cpu']
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
    if version != 'R108' or Path(root).resolve() != ROOT.resolve() or Path(launch_root).resolve() != LAUNCH.resolve():
        raise RuntimeError('Publication accepts only canonical R108 roots')
    root, launch_root, git_root = (Path(p).resolve() for p in (root, launch_root, git_root))
    from .r108_launch import PUBLICATION_EXTRAS, SOURCE_NAMES
    ready = optional(launch_root / ('R108_' + ARMS[0]) / 'ready.json')
    require_ready(root, launch_root, ready.get('evidence'))
    own = {name for name in SOURCE_NAMES if name.startswith(('r108_', 'test_r108_', 'run_v4_r108_'))}
    own.add('run_v4_r108_launch.sh')
    candidates = [REPO / name for name in PUBLICATION_EXTRAS]
    for arm in ARMS:
        output = launch_root / ('R108_' + arm)
        plan = optional(output / 'plan.json')
        candidates += [Path(p) for p in plan.get('source_sha256', {})
            if Path(p).parent == REPO / 'code/V4' and Path(p).name in own]
        candidates += [launch_root / 'specs' / ('R108_' + arm + '.json')]
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
                or path.name.lower().startswith(('prediction', 'outcome', 'cost_array', 'winner_array',
                                                'raw_cost', 'raw_label', 'labels', 'token'))
                or any(word in path.name.lower() for word in ('wandb', 'credential', 'secret'))):
            continue
        resolved = path.resolve()
        if not resolved.is_relative_to(REPO.resolve()) or not resolved.is_relative_to(git_root):
            raise RuntimeError('Curated R108 asset escapes repository')
        if path.stat().st_size > adapter.MAX_ASSET_BYTES:
            raise RuntimeError('Curated R108 asset exceeds32MiB')
        if path.suffix.lower() == '.json' and not adapter._safe_json(optional(path)):
            continue
        result.add(str(resolved.relative_to(git_root)))
    if not result:
        raise RuntimeError('No curated R108 assets')
    return sorted(result)


# The inherited Git implementation holds <git-dir>/codex-publication.lock.
adapter.VERSIONS = dict(R108=ROOT.name)
adapter.LAUNCH = LAUNCH
adapter.model_root = lambda version: ROOT
adapter.status = status
adapter.publication_paths = publication_paths


def publish(root=ROOT, launch_root=LAUNCH, evidence=None):
    require_ready(root, launch_root, evidence)
    return adapter.publish('R108', root, launch_root)


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
        receipt = dict(experiment='R108', state='watching', pid=os.getpid(), started_utc=utc(),
            cpu_only=True, automatic_retry=False, github_on_completion=publish_github,
            main_ready_evidence=evidence, goal_complete=False, goal_status='active_unresolved')
        save(path, receipt)
        try:
            while True:
                require_ready(root, launch_root, evidence)
                current = status(root=root, launch_root=launch_root, persist=True)
                if current['errors']:
                    raise RuntimeError('R108 saved stage failure: ' + json.dumps(current['errors']))
                if current['all_arms_ready'] and not current['comparison_complete']:
                    compare_once(root, launch_root, evidence, **references)
                    continue
                if current['complete']:
                    if publish_github:
                        publish(root, launch_root, evidence)
                    receipt.update(state='complete', exit_status=0, pipeline_complete=True, completed_utc=utc())
                    save(path, receipt)
                    print(json.dumps(dict(experiment='R108', phase='complete', goal_complete=False)), flush=True)
                    return 0
                time.sleep(interval)
        except BaseException as error:
            receipt.update(state='stopped_on_error', exit_status=1, error=str(error), completed_utc=utc())
            save(path, receipt)
            print(json.dumps(dict(experiment='R108', phase='failed', error=str(error))), flush=True)
            return 1


def parser():
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument('--watch', action='store_true')
    value.add_argument('--publish-github', '--publish', dest='publish_github', action='store_true')
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

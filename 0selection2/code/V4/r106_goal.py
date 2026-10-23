"""Quiet saved-evidence R106 whole-version gate and explicit publication.

No torch, array, GPU query, launch, Git or notification runs on import/status.
Pending reports and declared historical failures are not current failures.
"""

import argparse
import csv
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

from . import r106_launch as launch


REPO, ROOT, LAUNCH = launch.REPO, launch.MODEL_ROOT, launch.ROOT
ARMS = launch.ARMS
REFERENCE_FLAGS = ('nss_root', 'historical_root', 'r94_root', 'r102_root')
RECORD_NAMES = ('protocol.json', 'status.json', 'initialization.json', 'training_complete.json',
    'final_test_lock.json', 'final_metrics.json', 'test_results.json', 'inference_timing.json',
    'lib_stage_complete.json', 'lib_stage_status.json', 'postprocess_status.json',
    'full_model_statistics/protocol.json')
BOUNDARY = ('Identical complete fresh tensors and pinned pretrained encoders; native frozen versus '
    'fine-tuned is the intervention. Trainable freedom and training compute differ. '
    'External pretraining data/overlap are unknown; native always online, unrelated to R102 recovery.')
adapter = launch.private('_r106_whole_version', 'r98_goal.py')
PROBLEMS, optional, save, utc = adapter.PROBLEMS, adapter.optional, adapter.save, adapter.utc
file_hash, json_hash = adapter.file_hash, adapter.json_hash


def receipt_errors(path, record):
    """Inspect live receipt state, not stage strings or historical error notes."""
    if not isinstance(record, dict):
        return []
    phases = [str(record.get(key, '')).lower() for key in ('state', 'phase')]
    rc = record.get('exit_status', record.get('returncode', record.get('exit_code')))
    failed = any(value in ('failed', 'error', 'stopped_on_error', 'stopped_on_failure')
        for value in phases) or (type(rc) is int and rc != 0)
    errors = []
    if failed:
        values = record.get('errors') or record.get('error') or ('phase=' + str(phases) + '; rc=' + str(rc))
        errors.extend(dict(path=str(path), error=value) for value in (values if isinstance(values, list) else [values]))
    stages = record.get('stages', [])
    if isinstance(stages, list):
        for child in stages:
            if isinstance(child, dict):
                errors.extend(receipt_errors(path, child))
    return errors


def read_record(path, errors, invalid):
    try:
        record = optional(path)
        if not isinstance(record, dict):
            invalid.append(dict(path=str(path), error='Expected a JSON receipt object'))
            return {}
        rc = record.get('exit_status', record.get('returncode', record.get('exit_code')))
        if rc is not None and type(rc) is not int:
            invalid.append(dict(path=str(path), error='Exit code is not an integer'))
        errors.extend(receipt_errors(path, record))
        return record
    except (OSError, TypeError, ValueError) as error:
        invalid.append(dict(path=str(path), error=str(error)))
        return {}


def stages_ready(record, expected):
    stages = record.get('stages', [])
    return (adapter.rc0(record) and isinstance(stages, list)
        and all(isinstance(stage, dict) for stage in stages)
        and [stage.get('name') for stage in stages] == list(expected)
        and all(adapter.rc0(stage) for stage in stages) and not receipt_errors('<stages>', record))


def selected_parent_ready(final, lock):
    return (bool(lock.get('checkpoint_sha256')) and bool(lock.get('state_hash'))
        and final.get('checkpoint_sha256') == lock['checkpoint_sha256']
        and final.get('state_hash') == lock['state_hash']
        and final.get('selected_update') == lock.get('selected_update'))


def split_bound(result, rows, lock, parent=None):
    if not isinstance(result, dict) or not adapter.split_ready(dict(result,
            identity=dict(result.get('identity', {}), checkpoint_sha256=lock.get('checkpoint_sha256'))), rows, lock):
        return False
    identity = result.get('identity', {})
    if identity.get('state_hash') != lock.get('state_hash'):
        return False
    capture_step = identity.get('successful_updates', identity.get('selected_update'))
    if capture_step is not None and capture_step != lock.get('selected_update'):
        return False
    if identity.get('checkpoint_sha256') == lock.get('checkpoint_sha256') and lock.get('checkpoint_sha256'):
        return True
    if identity.get('checkpoint_sha256', 'missing') is not None or not parent or not selected_parent_ready(parent, lock):
        return False
    val_identity = parent.get('val', {}).get('identity', {})
    signature = identity.get('model_state_sha256')
    # Capture-time file SHA may be null. Bind the enclosing final state, exact
    # selected update and the common TRAIN/VAL tensor signature, never None==None.
    return (bool(signature) and signature == val_identity.get('model_state_sha256')
        and capture_step == val_identity.get('successful_updates', val_identity.get('selected_update'))
            == lock.get('selected_update')
        and val_identity.get('state_hash') == lock.get('state_hash')
        and (not lock.get('model_state_sha256') or signature == lock['model_state_sha256']))


def timing_ready(record, lock, final):
    if (record.get('experiment') != 'R106' or record.get('arm') != lock.get('arm')
            or record.get('complete') is not True or not adapter.same_state(record, lock)
            or record.get('checkpoint_sha256') != lock.get('checkpoint_sha256')
            or record.get('inference_rule') != lock.get('inference_rule')
            or record.get('selected_update') != lock.get('selected_update')
            or record.get('all_current_model_included') is not True or record.get('donor_present') is not True
            or record.get('cached_feature_latency_used') is not False or record.get('end_to_end_measured') is not False
            or record.get('input_labels_read') is not False or record.get('model_reselection') is not False
            or record.get('scope') != 'online_native_plus_selector_GPU_resident_inputs'
            or not str(record.get('device', '')).startswith('cuda:')
            or not record.get('scope') or not record.get('excludes')
            or not record.get('model_state_sha256')
            or record['model_state_sha256'] != final.get('val', {}).get('identity', {}).get('model_state_sha256')):
        return False
    repetitions = record.get('repetitions')
    if repetitions != 3 or record.get('warmup_batches') != 1:
        return False
    rows = record.get('per_problem', [])
    if (not isinstance(rows, list) or len(rows) != 18 or any(not isinstance(row, dict) for row in rows)
            or {row.get('problem') for row in rows} != set(PROBLEMS)):
        return False
    for row in rows:
        measurement = row.get('online_native_plus_selector', {})
        samples, mean, count = measurement.get('batch_seconds'), measurement.get('mean_seconds_per_instance'), row.get('instances')
        if (count != 128 or measurement.get('donor_calls') != [1] * repetitions
                or measurement.get('warmup_donor_calls') != 1
                or measurement.get('native_output_dtype') != ['torch.float32']
                or not isinstance(samples, list) or len(samples) != repetitions
                or any(type(value) not in (int, float) or not math.isfinite(value) or value <= 0 for value in [*samples, mean])
                or not math.isclose(mean, sum(samples) / (repetitions * count), rel_tol=1e-6, abs_tol=1e-12)):
            return False
    return True


def capture_ready(run, result, split, model_sha, lock):
    capture = result.get('archive_capture_identity', result.get('identity', {}))
    path = result.get('prediction_archive')
    if not isinstance(path, str):
        return False
    path = Path(path).resolve()
    return (path.is_relative_to((run / 'evaluation' / split).resolve())
        and capture == optional(path.parent / 'identity.json')
        and capture.get('experiment') == 'R106' and capture.get('arm') == lock.get('arm')
        and capture.get('split') == split and capture.get('checkpoint_sha256', 'missing') in (None, lock.get('checkpoint_sha256'))
        and bool(model_sha) and capture.get('model_state_sha256') == model_sha
        and capture.get('state_hash') == lock.get('state_hash')
        and capture.get('successful_updates') == lock.get('selected_update'))


def binding_ready(proof, config, lock, signature):
    return (bool(config.get('selected_val_binding_rule')) and bool(signature)
        and proof.get('rule') == config['selected_val_binding_rule']
        and proof.get('checkpoint_sha256') == lock.get('checkpoint_sha256')
        and proof.get('model_state_sha256') == signature
        and proof.get('state_hash') == lock.get('state_hash')
        and proof.get('selected_update') == lock.get('selected_update')
        and proof.get('archive_identity_unchanged') is True)


def selected_splits_ready(run, final, config, lock, report):
    val, train = final.get('val', {}), final.get('train', {})
    signature = val.get('identity', {}).get('model_state_sha256')
    val_ready = (selected_parent_ready(final, lock) and split_bound(val, 4500, lock, final)
        and val.get('identity', {}).get('checkpoint_sha256') == lock.get('checkpoint_sha256')
        and binding_ready(val.get('checkpoint_binding', {}), config, lock, signature)
        and capture_ready(run, val, 'val', signature, lock))
    train_ready = (val_ready and split_bound(train, 81000, lock, final)
        and train.get('identity', {}).get('model_state_sha256') == signature
        and capture_ready(run, train, 'train', signature, lock)
        and binding_ready(report.get('selected_train_capture_proof',
            report.get('selected_val_binding', {})), config, lock, signature))
    return train_ready, val_ready


def initialization_ready(arm, init, config):
    counts = init.get('parameter_counts', {})
    total, selector, native = (counts.get(key) for key in ('total', 'selector', 'native'))
    if any(type(value) is not int or value <= 0 for value in (total, selector, native)):
        return False
    fine = arm == 'B_finetune'
    return (total == selector + native and init.get('seed') == 2
        and init.get('historical_selector_checkpoint_loaded') is False
        and bool(init.get('initial_state_sha256')) and bool(init.get('native_state_sha256'))
        and bool(init.get('selector_parameter_sha256'))
        and bool(init.get('common_parameter_sha256'))
        and counts.get('selector_trainable') == selector
        and counts.get('native_trainable') == (native if fine else 0)
        and counts.get('trainable') == (total if fine else selector)
        and init.get('total_parameter_count') == total
        and init.get('trainable_parameter_count') == counts['trainable']
        and config.get('core_parameter_count') == selector
        and config.get('native_trainable') is fine and config.get('native_lr_scale') == .1
        and init.get('common_fresh_model_state_sha256') == init.get('initial_state_sha256')
        and init.get('identical_complete_fresh_parameters') is True)


def nss4_ready(run, lock):
    path = run / 'full_model_statistics/nss_comparison.csv'
    if not path.is_file():
        return False
    with path.open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    return (len(rows) == 4 and {row.get('dataset') for row in rows}
        == {'TSP TEST', 'CVRP TEST', 'TSPLIB', 'CVRPLIB'}
        and all(row.get('current_checkpoint_sha256') == lock.get('checkpoint_sha256')
            and all(key in row for key in ('current_top1_native_pct', 'current_mean_cost',
                'nss_top1_native_pct', 'nss_mean_cost', 'status', 'data_version', 'candidate_pool', 'winner_rule'))
            and bool(row.get('status')) for row in rows))


def arm_checks(arm, run, records, main_sha):
    config, lock = records['protocol.json'], records['final_test_lock.json']
    final, report = records['final_metrics.json'], records['full_model_statistics/protocol.json']
    train, post = records['training_complete.json'], records['postprocess_status.json']
    checks = adapter.common_checks('R106', arm, run, records)
    selected = lock.get('selected_update')
    rule = str(lock.get('rule', '')).lower()
    checks['unique_VAL_lock'] &= (lock.get('selection_split', 'val') == 'val'
        and type(selected) is int and 0 < selected <= 57600 and selected % 1440 == 0
        and bool(config.get('inference_rule')) and lock.get('inference_rule') == config['inference_rule']
        and lock.get('rule') == config.get('checkpoint_rule')
        and all(word in rule for word in ('strict minimum', 'val', 'macro_vs_sbs', 'first')))
    checks['main_protocol_lock'] = bool(main_sha) and config.get('main_protocol_lock', {}).get('sha256') == main_sha
    checks['training_complete'] = (config.get('max_successful_updates') == 57600
        and train.get('successful_updates') == 57600 and train.get('state_hash') == lock.get('state_hash')
        and all(train.get('updates_by_problem', {}).get(problem) == 3200 for problem in PROBLEMS))
    checks['selected_parent'] = selected_parent_ready(final, lock)
    checks['full_TRAIN'], checks['full_VAL'] = selected_splits_ready(run, final, config, lock, report)
    test = records['test_results.json']
    checks['full_TEST'] = (test.get('complete') is True and test.get('checkpoint_sha256') == lock.get('checkpoint_sha256')
        and adapter.same_state(test, lock) and split_bound(test.get('result', {}), 4500, lock))
    lib = records['lib_stage_complete.json']
    checks['LIB'] = (lib.get('experiment') == 'R106' and lib.get('complete') is True
        and lib.get('checkpoint_sha256') == lock.get('checkpoint_sha256')
        and set(lib.get('datasets', [])) == {'TSPLIB', 'CVRPLIB'}
        and lib.get('native_counts') == {'TSPLIB': 49, 'CVRPLIB': 100})
    checks['CPU_report'] = (post.get('complete') is True and post.get('phase') == 'complete'
        and post.get('exit_status') == 0 and report.get('complete') is True
        and report.get('experiment') == 'R106' and report.get('checkpoint_lock') == lock)
    checks['all18_online_timing'] = timing_ready(records['inference_timing.json'], lock, final)
    timing_path = run / 'inference_timing.json'
    checks['timing_bound_to_report'] = (timing_path.is_file()
        and (report.get('inference_timing') == records['inference_timing.json']
            or report.get('inference_timing', {}).get('sha256') == file_hash(timing_path)))
    checks['fixed36'] = (report.get('target_identity', {}).get('target_count') == 36
        and bool(report.get('target_identity', {}).get('sha256')))
    checks['same_model_NSS4'] = nss4_ready(run, lock)
    checks['fresh_initialization'] = initialization_ready(arm, records['initialization.json'], config)
    checks['native_online'] = (config.get('native_execution') == 'online_all_splits'
        and config.get('native_feature_cache') is False and config.get('native_eval') is True
        and config.get('current_query_labels_in_forward') is False)
    return checks


def status(version='R106', root=None, launch_root=LAUNCH, *, persist=False):
    if version != 'R106':
        raise ValueError('This private watcher owns only R106')
    root, launch_root = Path(root or ROOT).resolve(), Path(launch_root).resolve()
    errors, invalid, rows, configs, locks, inits = [], [], [], {}, {}, {}
    main_path = root / 'protocol_lock.json'
    main_science = read_record(main_path, errors, invalid)
    main_sha = file_hash(main_path) if main_path.is_file() else None
    for arm in ARMS:
        run, identifier = root / arm, launch.run_id(arm)
        records = {name: read_record(run / name, errors, invalid) for name in RECORD_NAMES}
        configs[arm], locks[arm], inits[arm] = (records[key] for key in
            ('protocol.json', 'final_test_lock.json', 'initialization.json'))
        output = launch_root / identifier
        spec = read_record(launch_root / 'specs' / (identifier + '.json'), errors, invalid)
        controller = read_record(output / 'controller.json', errors, invalid)
        worker = read_record(output / 'worker.json', errors, invalid)
        read_record(output / 'status.json', errors, invalid)
        try:
            checks = arm_checks(arm, run, records, main_sha)
            checks['R84_fresh_identity'] = (bool(main_science.get('original_fresh_trainable_sha256'))
                and inits[arm].get('common_parameter_sha256') == main_science['original_fresh_trainable_sha256'])
            checks.update(launch_spec=spec.get('run_id') == identifier and spec.get('run_dir') == str(run)
                and spec.get('slots') == [launch.ARM_SLOTS[arm]] and spec.get('spec_draft') is False
                and spec.get('cpus') == launch.SLOT_CPUS[launch.ARM_SLOTS[arm]]
                and [(stage.get('name'), stage.get('device')) for stage in spec.get('stages', [])]
                    == [(name, 'cpu' if name == 'PDF' else 'gpu') for name in launch.STAGES],
                controller_rc0=stages_ready(controller, ['allocation_GPU_pipeline', 'PDF']),
                worker_all_GPU_stages_rc0=stages_ready(worker, ['train', 'TEST', 'timing', 'LIB']))
        except (OSError, AttributeError, KeyError, TypeError, ValueError) as error:
            checks = dict(valid_receipts=False)
            invalid.append(dict(path=str(run), error='Incomplete/malformed evidence: ' + str(error)))
        checks.update({name: adapter.pdf_ready(run / 'full_model_statistics' / name) for name in adapter.R99_PDFS})
        rows.append(dict(arm=arm, run_id=identifier, root=str(run), checks=checks,
            ready=all(checks.values()), missing=[key for key, value in checks.items() if not value]))
    match_keys = ('data_root', 'prepared_root', 'rows_per_problem', 'candidate_pools', 'baselines',
        'max_successful_updates', 'inference_rule', 'main_protocol_lock', 'microbatch_caps',
        'task_order_sha256', 'native_pair_context', 'native_lr_scale')
    matched = all(configs[ARMS[0]].get(key) is not None
        and configs[ARMS[0]].get(key) == configs[ARMS[1]].get(key) for key in match_keys)
    identical = all(bool(inits[ARMS[0]].get(key)) and inits[ARMS[0]].get(key) == inits[ARMS[1]].get(key)
        for key in ('initial_state_sha256', 'native_state_sha256', 'selector_parameter_sha256', 'common_parameter_sha256'))
    matched &= all(bool(inits[ARMS[0]].get('parameter_counts', {}).get(key))
        and inits[ARMS[0]].get('parameter_counts', {}).get(key) == inits[ARMS[1]].get('parameter_counts', {}).get(key)
        for key in ('total', 'selector', 'native'))
    main_locked = main_science.get('experiment') == 'R106' and main_science.get('locked') is True
    comparison = read_record(root / 'comparison_artifacts/protocol.json', errors, invalid)
    for name in ('goal_watch_receipt.json', 'goal_compare_receipt.json', 'github_publication.json'):
        read_record(root / name, errors, invalid)
    compared = (comparison.get('experiment') == 'R106' and comparison.get('complete') is True
        and comparison.get('checkpoints') == locks and comparison.get('training') is False
        and comparison.get('inference') is False and all((root / 'comparison_artifacts' / name).is_file()
            and (root / 'comparison_artifacts' / name).stat().st_size > 0 for name in
            ('paired_changes_B_vs_A.csv', 'target_recovery_B_vs_A.csv', 'target_recovery_summary_B_vs_A.csv')))
    ready = main_locked and matched and identical and all(row['ready'] for row in rows)
    complete = ready and compared and not errors and not invalid
    result = dict(experiment='R106', arms=rows, errors=errors, invalid_receipts=invalid,
        matched_controls=matched, identical_full_initialization=identical, all_arms_ready=ready,
        comparison_complete=compared, complete=complete,
        phase='failed' if errors else 'complete' if complete else 'waiting_for_comparison' if ready else 'waiting_for_arms',
        cpu_only=True, automatic_retry=False, goal_complete=False, goal_status='active_unresolved',
        comparison_boundary=BOUNDARY)
    if persist:
        path, prior = root / 'goal_progress.json', optional(root / 'goal_progress.json')
        if {key: value for key, value in prior.items() if key != 'updated_utc'} == result:
            return prior
        result['updated_utc'] = utc()
        save(path, result)
    return result


def require_ready(root, launch_root, evidence):
    if not isinstance(evidence, str) or not evidence.strip():
        raise ValueError('Explicit main-ready evidence required before watch/compare/publication')
    if Path(root).resolve() != launch.MODEL_ROOT.resolve() or Path(launch_root).resolve() != launch.ROOT.resolve():
        raise RuntimeError('Only canonical main-ready R106 roots are accepted')
    for arm in ARMS:
        spec, plan = launch.check_ready(Path(launch_root) / 'specs' / (launch.run_id(arm) + '.json'))
        if plan.get('evidence') != evidence or spec.get('run_dir') != str(Path(root).resolve() / arm):
            raise RuntimeError('Both source-frozen arm plans must match main evidence')


def compare_once(root=ROOT, launch_root=LAUNCH, evidence=None, **references):
    root = Path(root).resolve()
    require_ready(root, launch_root, evidence)
    current = status(root=root, launch_root=launch_root)
    if current['errors'] or current['invalid_receipts'] or not current['all_arms_ready'] or current['comparison_complete']:
        raise RuntimeError('CPU comparison requires both complete pipelines, once only')
    with (root / 'goal_compare.lock').open('a') as claim:
        adapter.fcntl.flock(claim, adapter.fcntl.LOCK_EX | adapter.fcntl.LOCK_NB)
        path = root / 'goal_compare_receipt.json'
        if path.exists():
            raise RuntimeError('Comparison already attempted; preserve receipt, no retry')
        argv = [sys.executable, '-B', '-u', '-m', 'code.V4.r106_reporting', 'compare', '--root', str(root), '--device', 'cpu']
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
                raise RuntimeError('CPU comparison lacks rc0/same-lock saved outputs; no retry')
            receipt.update(state='complete', exit_status=0, completed_utc=utc())
        except BaseException as error:
            receipt.update(state='failed', exit_status=1, error=str(error), completed_utc=utc())
            raise
        finally:
            save(path, receipt)


def publication_paths(version, root, launch_root, git_root):
    if version != 'R106' or Path(root).resolve() != ROOT.resolve() or Path(launch_root).resolve() != LAUNCH.resolve():
        raise RuntimeError('Publication accepts only the complete canonical R106 version')
    root, launch_root, git_root = (Path(value).resolve() for value in (root, launch_root, git_root))
    ready = optional(launch_root / launch.run_id(ARMS[0]) / 'ready.json')
    require_ready(root, launch_root, ready.get('evidence'))
    own = {name for name in (*launch.SOURCE_NAMES, *launch.OPTIONAL_SOURCE_NAMES)
        if name.startswith(('r106_', 'test_r106_'))}
    own.add('run_v4_r106_launch.sh')
    candidates = [REPO / 'code/V4' / name for name in own]
    for arm in ARMS:
        output = launch_root / launch.run_id(arm)
        candidates += [launch_root / 'specs' / (launch.run_id(arm) + '.json')]
        candidates += [output / name for name in ('plan.json', 'ready.json', 'controller.json', 'worker.json')]
    for run in (root, *(root / arm for arm in ARMS)):
        if not run.is_dir():
            continue
        candidates += [path for path in run.iterdir() if path.suffix.lower() in {'.json', '.csv', '.pdf', '.md'}
            and not path.name.startswith(('goal_', 'github_', 'wandb', 'prediction', 'outcome'))]
        for directory in ('preparation', 'full_model_statistics', 'comparison_artifacts'):
            for parent, dirs, files in os.walk(run / directory):
                dirs[:] = [name for name in dirs if name not in adapter.PRUNED
                    and name not in {'dispatch', 'sources', 'runtime', 'cache', 'donor_cache'}
                    and not (Path(parent) / name).is_symlink()]
                candidates += [Path(parent) / name for name in files if Path(name).suffix.lower() in {'.json', '.csv', '.pdf', '.md'}]
    result = set()
    for path in candidates:
        if (not path.is_file() or path.is_symlink() or any(parent.is_symlink() for parent in path.parents)
                or path.name.lower().startswith(('prediction', 'outcome', 'cost_array', 'winner_array', 'raw_cost', 'raw_label', 'labels', 'token'))
                or any(word in path.name.lower() for word in ('wandb', 'credential', 'secret', 'password'))):
            continue
        resolved = path.resolve()
        if not resolved.is_relative_to(REPO.resolve()) or not resolved.is_relative_to(git_root):
            raise RuntimeError('Curated R106 asset escapes repository')
        if path.stat().st_size > adapter.MAX_ASSET_BYTES:
            raise RuntimeError('Curated R106 asset exceeds32MiB')
        if path.suffix.lower() == '.json' and not adapter._safe_json(optional(path)):
            continue
        result.add(str(resolved.relative_to(git_root)))
    if not result:
        raise RuntimeError('No curated R106 assets')
    return sorted(result)


adapter.VERSIONS = dict(R106=ROOT.name)
adapter.LAUNCH = LAUNCH
adapter.model_root = lambda version: ROOT
adapter.status, adapter.publication_paths = status, publication_paths


def publish(root=ROOT, launch_root=LAUNCH, evidence=None):
    require_ready(root, launch_root, evidence)
    return adapter.publish('R106', root, launch_root)


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
            raise RuntimeError('Watcher already attempted; preserve receipt, no retry')
        receipt = dict(experiment='R106', state='watching', pid=os.getpid(), started_utc=utc(),
            cpu_only=True, automatic_retry=False, github_on_completion=publish_github,
            main_ready_evidence=evidence, goal_complete=False, goal_status='active_unresolved')
        save(path, receipt)
        try:
            while True:
                require_ready(root, launch_root, evidence)
                current = status(root=root, launch_root=launch_root, persist=True)
                if current['errors']:
                    raise RuntimeError('R106 current stage failure: ' + json.dumps(current['errors']))
                if current['all_arms_ready'] and not current['invalid_receipts'] and not current['comparison_complete']:
                    compare_once(root, launch_root, evidence, **references)
                    continue
                if current['complete']:
                    if publish_github:
                        publish(root, launch_root, evidence)
                    receipt.update(state='complete', exit_status=0, pipeline_complete=True, completed_utc=utc())
                    save(path, receipt)
                    print(json.dumps(dict(experiment='R106', phase='complete', goal_complete=False)), flush=True)
                    return 0
                time.sleep(interval)
        except BaseException as error:
            receipt.update(state='stopped_on_error', exit_status=1, error=str(error), completed_utc=utc())
            save(path, receipt)
            print(json.dumps(dict(experiment='R106', phase='failed', error=str(error))), flush=True)
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

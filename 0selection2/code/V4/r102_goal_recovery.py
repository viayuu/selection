"""Read-only R102 recovery status and opt-in quiet, one-shot CPU delivery gate.

Only new recovery receipts are written. Original failed routes stay failed.
No model, array, checkpoint, GPU, queue or Git operation occurs on import/status.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from . import r84_launch as resource
from . import r102_goal as previous
from . import r102_gpu_recovery_entry as gpu_entry


REPO, ROOT, LAUNCH = previous.REPO, previous.ROOT, previous.LAUNCH
BASE = ROOT / 'preparation/postprocess_recovery'
VIEW = BASE / 'report_view'
ARMS, PROBLEMS = previous.ARMS, previous.PROBLEMS
admin = previous.private()
optional, file_hash, utc = admin.optional, admin.file_hash, admin.utc
LOCKS = dict(A=(33120, '10694b4d0fed80f1baec0d5e7de15cceec210eca5dfa846623783c4b9d6e7a2c'),
             B=(20160, '7df434c762a0e0b7726ab67211f63f51b55d1f86d108ec307a0aede0060f921f'))
SLOTS = dict(A='4090_0', B='4090_1')
CPUS = dict(A=[0, 1, 2, 3], B=[4, 5, 72, 73])
NEW_SOURCES = ('r102_gpu_recovery_entry.py', 'run_v4_r102_gpu_recovery.sh',
    'r102_goal_recovery.py', 'test_r102_goal_recovery.py',
    'r102_postprocess_recovery.py', 'test_r102_postprocess_recovery.py',
    'r102_cache_identity_probe.py', 'run_v4_r102_cache_probe.sh')
BUILD_FIELDS = {'created_utc', 'completed_utc', 'seconds', 'test_lock_root'}
DELIVERY_CONTINUATION = False


def delivery_history():
    watcher = BASE / 'watch_receipt.json'
    record = optional(watcher)
    if (record.get('state') != 'stopped_on_error' or record.get('exit_status') != 1
            or record.get('error') != 'Recovery failed; preserve every original/new receipt, no automatic retry'):
        raise RuntimeError('Delivery continuation requires the preserved administrative watcher failure')
    paths = (watcher, BASE / 'source_provenance.json')
    return {path.name: dict(path=str(path), sha256=file_hash(path)) for path in paths}


def receipt_errors(path, record):
    # Science receipts list stage names; launcher receipts contain nested stage objects.
    stages = record.get('stages', []) if isinstance(record, dict) else []
    if stages and all(isinstance(stage, str) for stage in stages):
        expected = {'timing_observation': 0, 'lib_completion': 0}
        if (record.get('schema') != 'r102_postprocess_recovery_v1'
                or record.get('complete') is not True or record.get('exit_status') != 0
                or stages not in (['timing_observation', 'lib_completion'], ['CPU_report'])
                or record.get('stage_exit_statuses') != expected):
            return [dict(path=str(path), error='Invalid named-stage completion receipt')]
        record = dict(record, stages=[])
    return admin.receipt_errors(path, record)


def original(arm):
    if arm not in ARMS:
        raise ValueError('Only the original R102 A/B locks are permitted')
    run, dispatch = ROOT / arm, LAUNCH / ('R102_' + arm)
    worker, controller = (optional(dispatch / name) for name in ('worker.json', 'controller.json'))
    config, lock, train, state, error, marker = (optional(run / name) for name in
        ('protocol.json', 'final_test_lock.json', 'training_complete.json', 'status.json', 'error.json', 'test_started.json'))
    if (worker.get('state') != 'failed' or worker.get('phase') != 'train' or worker.get('exit_status') != 1
            or [(s.get('name'), s.get('state'), s.get('exit_status')) for s in worker.get('stages', [])]
                != [('input_features', 'complete', 0), ('train', 'failed', 1)]
            or controller.get('state') != 'failed' or controller.get('exit_status') != 1
            or state.get('phase') != 'failed' or state.get('successful_updates') != 57600
            or error.get('type') != 'ValueError' or error.get('message') != 'Full frozen R84 header identity differs'
            or state.get('error') != error):
        raise RuntimeError('Only the original TRAIN-stage first-TEST header failure may be recovered')
    item = resource.SLOTS[SLOTS[arm]]
    identity = resource.transport.CONTAINERS[item['node']][1]
    if (worker.get('slot') != SLOTS[arm] or worker.get('gpu_uuid') != item['uuid']
            or worker.get('container_id') != identity or worker.get('cpus') != CPUS[arm]):
        raise RuntimeError('Original allocation/container/UUID/CPU assignment differs')
    checks = admin.common_checks('R102', arm, run, {'protocol.json': config, 'final_test_lock.json': lock})
    if (not all(checks.values()) or (lock.get('selected_update'), lock.get('checkpoint_sha256')) != LOCKS[arm]
            or lock.get('selection_split') != 'val' or lock.get('inference_rule') != config.get('inference_rule')
            or lock.get('rule') != config.get('checkpoint_rule') or config.get('max_successful_updates') != 57600
            or train.get('successful_updates') != 57600 or train.get('selected_best') != lock
            or train.get('state_hash') != lock.get('state_hash') or train.get('test_read') is not False
            or any(train.get('updates_by_problem', {}).get(p) != 3200 for p in PROBLEMS)
            or marker.get('checkpoint_sha256') != lock.get('checkpoint_sha256')):
        raise RuntimeError('Need original full57600 TRAIN and the unique original VAL-only checkpoint lock')
    for name, digest in config.get('source_sha256', {}).items():
        if Path(name).name != name or file_hash(REPO / 'code/V4' / name) != digest:
            raise RuntimeError('Original scientific source changed: ' + name)
    watcher = optional(ROOT / 'goal_watch_receipt.json')
    if watcher.get('state') != 'stopped_on_error' or watcher.get('exit_status') != 1:
        raise RuntimeError('Original watcher must remain stopped_on_error')
    paths = {name: run / name for name in ('protocol.json', 'initialization.json', 'final_test_lock.json',
        'training_complete.json', 'final_metrics.json', 'status.json', 'error.json', 'test_started.json')}
    paths.update({'worker.json': dispatch / 'worker.json', 'controller.json': dispatch / 'controller.json',
                  'goal_watch_receipt.json': ROOT / 'goal_watch_receipt.json'})
    return dict(checkpoint_lock=lock, sha256={name: file_hash(path) for name, path in paths.items()})


def require_no_original_test(arm):
    run = ROOT / arm
    if any((run / name).exists() or (run / name).is_symlink() for name in
           ('test_results.json', 'test_complete.json', 'test_input_features_ready.json', 'evaluation/test')):
        raise RuntimeError('Original full TEST predictions must not already exist')


def proof_ready():
    path = BASE / 'array_equality_proof.json'
    proof = optional(path)
    expected = dict(schema='r102_cross_node_test_array_identity_v1', complete=True, all_arrays_equal=True,
        problem_count=18, array_count=107, gpu03_header_matches_frozen_reference=True,
        other_gpu05_header_fields_equal=True, arrays_or_headers_modified=False, no_solver_or_label_generation=True)
    if (any(proof.get(k) != v for k, v in expected.items())
            or set(proof.get('allowed_build_receipt_fields', [])) != BUILD_FIELDS
            or set(proof.get('per_problem', {})) != set(PROBLEMS)
            or any(row.get('arrays_equal') is not True
                   or set(row.get('gpu05_different_header_fields', [])) != BUILD_FIELDS
                   for row in proof.get('per_problem', {}).values())):
        raise RuntimeError('Require the complete107-array cross-node identity proof, only four build fields differ')
    for node in ('gpu03', 'gpu05'):
        record = proof.get('nodes', {}).get(node, {})
        manifest = BASE / (node + '_hashes.json')
        if record.get('path') != str(manifest) or record.get('sha256') != file_hash(manifest):
            raise RuntimeError('Cross-node array proof manifest changed')
    return proof


def required_sources():
    return [*(REPO / 'code/V4' / name for name in NEW_SOURCES),
            BASE / 'RECOVERY_CONTRACT.md', BASE / 'LAUNCH_RECOVERY.md']


def require_ready(evidence):
    if not isinstance(evidence, str) or not evidence.strip():
        raise ValueError('Main must explicitly supply recovery source-ready evidence')
    return dict(evidence=evidence)


def source_provenance(evidence):
    value = dict(schema='r102_recovery_source_provenance_v1', experiment='R102',
        source_sha256={str(path): file_hash(path) for path in required_sources() if path.is_file()},
        scientific_source_sha256={str(REPO / 'code/V4' / name): digest for arm in ARMS
            for name, digest in optional(ROOT / arm / 'protocol.json').get('source_sha256', {}).items()},
        main_ready_evidence=evidence, originals={arm: original(arm) for arm in ARMS},
        array_proof=dict(path=str(BASE / 'array_equality_proof.json'), sha256=file_hash(BASE / 'array_equality_proof.json')),
        original_failed_route_complete=False, goal_complete=False)
    if DELIVERY_CONTINUATION:
        value['preserved_administrative_failure'] = delivery_history()
    path = BASE / ('delivery_source_provenance.json' if DELIVERY_CONTINUATION else 'source_provenance.json')
    try:
        resource.save(path, value, exclusive=True)
    except FileExistsError:
        if optional(path) != value:
            raise RuntimeError('Recovery provenance changed; no automatic replacement')
    return dict(path=str(path), sha256=file_hash(path))


def checked_metadata(record, *, within=None):
    within = BASE if within is None else within
    path = Path(record['path'])
    if (not path.resolve().is_relative_to(within.resolve()) or path.is_symlink()
            or path.suffix.lower() not in ('.json', '.pdf', '.csv', '.md')
            or path.stat().st_size > admin.MAX_ASSET_BYTES or file_hash(path) != record['sha256']):
        raise RuntimeError('Changed/foreign recovery metadata artifact')
    return path


def gpu_entry_path(arm):
    post = BASE / arm / gpu_entry.POST_TEST_ENTRY
    if post.exists():
        receipt = optional(post)
        continuation = gpu_entry.post_test_evidence(arm)
        if (receipt.get('post_test_continuation') is not True or receipt.get('stage') != 'post-test'
                or any(receipt.get(key) != value for key, value in continuation.items())):
            raise RuntimeError('Post-test entry must preserve the exact failure and completed first TEST')
        return post
    path = BASE / arm / gpu_entry.UUID_CORRECTION_ENTRY
    if not path.exists():
        return BASE / arm / 'gpu_entry.json'
    receipt = optional(path)
    observation = gpu_entry.uuid_preflight_observation(arm)
    if (receipt.get('uuid_format_correction') is not True
            or any(receipt.get(key) != value for key, value in observation.items())):
        raise RuntimeError('UUID correction receipt must preserve the main-observed first preflight failure')
    return path


def post_test_boundary(arm, record):
    boundary = record.get('numerical_boundary')
    if (record.get('completion_kind') != 'post_test_original_timing_observation'
            or record.get('test_rerun') is not False or record.get('online_efficacy_bitwise_certified') is not False
            or record.get('stage_exit_statuses') != dict(timing_observation=0, lib_completion=0)
            or not isinstance(boundary, str) or not boundary.strip()
            or set(record.get('timing_observations', {})) != set(PROBLEMS)):
        raise RuntimeError('Post-test completion must disclose all18 numerical observations, not bitwise online certification')
    prior = record['prior_timing_failure']
    current = gpu_entry.post_test_evidence(arm)
    if prior['failure'] != current['preserved_first_error']:
        raise RuntimeError('Post-test completion lost the exact preserved timing failure')
    checked_metadata(prior['failure'])
    if checked_metadata(prior['observation']) != BASE / arm / 'timing_checks/TSP.json':
        raise RuntimeError('Original timing check must remain unchanged')
    for name, expected in (('gpu_entry.json', current['preserved_gpu_entry']),
                            (gpu_entry.UUID_CORRECTION_ENTRY, current['preserved_uuid_correction_entry'])):
        if prior['preserved_entries'].get(name) != expected:
            raise RuntimeError('Post-test completion lost a preserved failed entry')
        checked_metadata(expected)
    for problem, artifact in record['timing_observations'].items():
        path = checked_metadata(artifact)
        observation = optional(path)
        if (path != BASE / arm / 'timing_observations' / (problem + '.json')
                or observation.get('problem') != problem or observation.get('instances') != 128
                or observation.get('scope') != 'first128 TEST rows/task only'
                or observation.get('numerical_boundary') != boundary or observation.get('finite') is not True
                or observation.get('inference_changed') is not False or observation.get('grouping_changed') is not False):
            raise RuntimeError('Numerical observations must retain the original inference and first128-only boundary')


def stage_complete(arm, stage, *, require_entry=True):
    path = BASE / arm / (stage + '_complete.json')
    record = optional(path)
    if not record:
        return False
    origin = original(arm)
    lock = origin['checkpoint_lock']
    post_test = gpu_entry_path(arm).name == gpu_entry.POST_TEST_ENTRY
    if (record.get('schema') != 'r102_postprocess_recovery_v1' or record.get('experiment') != 'R102'
            or record.get('arm') != arm or record.get('complete') is not True or record.get('exit_status') != 0
            or any(record.get(key) != lock[key] for key in ('checkpoint_sha256', 'state_hash', 'selected_update'))
            or record.get('report_view') != str(VIEW) or record.get('training') is not False
            or record.get('model_reselection') is not False
            or (BASE / arm / 'post_test_error.json').exists()
            or ((BASE / arm / 'first_error.json').exists() and not post_test)):
        raise RuntimeError('Recovery completion must bind the original unique VAL lock')
    if post_test:
        post_test_boundary(arm, record)
    provenance = optional(checked_metadata(record['origin']))
    selected_signature = optional(ROOT / arm / 'final_metrics.json').get('val', {}).get('identity', {}).get('model_state_sha256')
    if (provenance.get('original_root') != str(ROOT) or provenance.get('arm') != arm
            or provenance.get('checkpoint_sha256') != lock['checkpoint_sha256']
            or provenance.get('selected_update') != lock['selected_update']
            or not provenance.get('model_state_sha256')
            or provenance.get('model_state_sha256') != selected_signature
            or record.get('model_state_sha256') != provenance.get('model_state_sha256')):
        raise RuntimeError('Recovery origin/model+buffer signature does not match the selected original')
    paths = {'worker.json': LAUNCH / ('R102_' + arm) / 'worker.json',
             'controller.json': LAUNCH / ('R102_' + arm) / 'controller.json',
             'goal_watch_receipt.json': ROOT / 'goal_watch_receipt.json'}
    for name, digest in origin['sha256'].items():
        if provenance.get('original_files', {}).get(str(paths.get(name, ROOT / arm / name))) != digest:
            raise RuntimeError('Recovery origin lost original failure evidence: ' + name)
    outputs = record.get('outputs', {})
    required = (('test_results.json', 'test_complete.json', 'inference_timing.json', 'lib_stage_complete.json',
                 'evaluation/test/selected_best/identity.json', 'evaluation/test/selected_best/rank000.npz') if stage == 'gpu'
                else ('postprocess_status.json', 'full_model_statistics/protocol.json', *admin.R99_PDFS))
    if not set(required) <= set(outputs):
        raise RuntimeError('Recovery receipt lacks required actual TEST/report artifacts')
    for name in required:
        record_path = Path(outputs[name]['path'])
        expected = VIEW / arm / (('full_model_statistics/' if name in admin.R99_PDFS else '') + name)
        if record_path != expected:
            raise RuntimeError('Recovery output is not in the canonical report view')
        # The science worker verifies payload hashes. Administration never opens arrays.
        if record_path.suffix == '.npz':
            if not record_path.is_file() or record_path.is_symlink():
                raise RuntimeError('Actual TEST prediction archive is missing')
        else:
            checked_metadata(outputs[name], within=VIEW / arm)
    if stage == 'report':
        if checked_metadata(record['gpu_completion']) != BASE / arm / 'gpu_complete.json':
            raise RuntimeError('CPU report is not bound to the original recovery GPU stage')
        if post_test:
            report = optional(VIEW / arm / 'full_model_statistics/protocol.json')
            if (report.get('recovery_numerical_boundary') != record['numerical_boundary']
                    or report.get('online_efficacy_bitwise_certified') is not False
                    or report.get('timing_observations') != record['timing_observations']):
                raise RuntimeError('CPU report must disclose the preserved cached TEST and numerical boundary')
            for key, name in (('note', 'numerical_boundary.md'), ('pdf', 'numerical_boundary.pdf')):
                artifact = report['numerical_boundary_artifacts'][key]
                if (checked_metadata(artifact) != VIEW / arm / 'full_model_statistics' / name
                        or record['outputs'].get('numerical_boundary_' + key) != artifact):
                    raise RuntimeError('CPU report numerical boundary delivery is incomplete')
    if require_entry:
        entry = optional(gpu_entry_path(arm) if stage == 'gpu' else BASE / arm / (stage + '_entry.json'))
        if not entry and stage == 'report':
            return True
        if (not admin.rc0(entry) or entry.get('original') != origin or not entry.get('main_ready_evidence')):
            return False
        if not entry.get('source_sha256'):
            raise RuntimeError('Missing recovery entry source provenance')
        for path, digest in entry['source_sha256'].items():
            if file_hash(path) != digest:
                raise RuntimeError('Recovery entry/helper source changed since the explicit attempt')
    return True


def arm_status(arm):
    origin = original(arm)
    run, view = ROOT / arm, VIEW / arm
    checks = {stage: stage_complete(arm, stage) for stage in ('gpu', 'report')}
    names = ('protocol.json', 'final_test_lock.json', 'training_complete.json', 'final_metrics.json')
    records = {name: optional(run / name) for name in names}
    for name in ('test_results.json', 'inference_timing.json', 'lib_stage_complete.json',
                 'postprocess_status.json', 'full_model_statistics/protocol.json'):
        records[name] = optional(view / name)
    # Reuse original science/signature checks; only new results/timing/report paths differ.
    checks.update(previous.arm_checks('R102', arm, run, records))
    final, lock = records['final_metrics.json'], origin['checkpoint_lock']
    val = final.get('val', {})
    identity = val.get('identity', {})
    if identity.get('checkpoint_sha256') is None:
        # During training VAL precedes serialization; bind its identity to the parent lock.
        bound_val = dict(val, identity=dict(identity, checkpoint_sha256=lock['checkpoint_sha256']))
        checks['full_VAL'] = (final.get('checkpoint_sha256') == lock['checkpoint_sha256']
            and final.get('state_hash') == lock['state_hash']
            and final.get('selected_update') == lock['selected_update']
            and identity.get('successful_updates') == lock['selected_update']
            and identity.get('state_hash') == lock['state_hash']
            and admin.split_ready(bound_val, 4500, lock))
    timing_path = view / 'inference_timing.json'
    report = records['full_model_statistics/protocol.json']
    checks['timing_bound_to_report'] = (timing_path.is_file()
        and report.get('inference_timing', {}).get('sha256') == file_hash(timing_path))
    for name in admin.R99_PDFS:
        checks[name] = admin.pdf_ready(view / 'full_model_statistics' / name)
    identity = optional(view / 'evaluation/test/selected_best/identity.json')
    lock = origin['checkpoint_lock']
    checks['actual_locked_TEST'] = (records['test_results.json'].get('selected_update') == lock['selected_update']
        and identity.get('successful_updates') == lock['selected_update']
        and identity.get('checkpoint_sha256') == lock['checkpoint_sha256']
        and identity.get('state_hash') == lock['state_hash']
        and bool(identity.get('model_state_sha256'))
        and identity.get('model_state_sha256') == records['final_metrics.json'].get('val', {}).get('identity', {}).get('model_state_sha256'))
    return dict(arm=arm, ready=all(checks.values()), checks=checks,
                missing=[key for key, value in checks.items() if not value], original=origin)


def comparison_ready():
    protocol = optional(VIEW / 'comparison_artifacts/protocol.json')
    receipt = optional(BASE / 'compare_complete.json')
    if not protocol or not receipt:
        return False
    locks = {arm: original(arm)['checkpoint_lock'] for arm in ARMS}
    target = protocol.get('target_identity', {})
    return (receipt.get('schema') == 'r102_postprocess_recovery_v1' and receipt.get('complete') is True
        and receipt.get('exit_status') == 0 and receipt.get('report_view') == str(VIEW)
        and receipt.get('training') is False and receipt.get('inference') is False
        and receipt.get('result') == protocol
        and all(checked_metadata(receipt['reports'][arm]) == BASE / arm / 'report_complete.json' for arm in ARMS)
        and protocol.get('experiment') == 'R102' and protocol.get('complete') is True
        and protocol.get('exit_status') == 0 and protocol.get('checkpoints') == locks
        and target.get('target_count') == 36 and bool(target.get('sha256'))
        and all(protocol.get('target_identity') == optional(VIEW / arm / 'full_model_statistics/protocol.json').get('target_identity')
            and (VIEW / arm / 'inference_timing.json').is_file()
            and protocol.get('inference_timing', {}).get(arm, {}).get('sha256')
                == file_hash(VIEW / arm / 'inference_timing.json') for arm in ARMS)
        and admin.pdf_ready(VIEW / 'comparison_artifacts/COMPARISON.pdf') and admin.pdf_ready(VIEW / 'RESULTS.pdf'))


def status(*, persist=False):
    errors, arms, preserved = [], [], []
    try:
        proof_ready()
        arms = [arm_status(arm) for arm in ARMS]
        for arm in ARMS:
            selected = gpu_entry_path(arm)
            if selected.name in (gpu_entry.UUID_CORRECTION_ENTRY, gpu_entry.POST_TEST_ENTRY):
                old = BASE / arm / 'gpu_entry.json'
                preserved.extend(admin.receipt_errors(old, optional(old)))
            if selected.name == gpu_entry.POST_TEST_ENTRY:
                for name in (gpu_entry.UUID_CORRECTION_ENTRY, 'first_error.json'):
                    path = BASE / arm / name
                    preserved.extend(admin.receipt_errors(path, optional(path)))
            errors.extend(receipt_errors(selected, optional(selected)))
            for name in ('gpu_complete.json', 'report_entry.json', 'report_complete.json', 'first_error.json', 'post_test_error.json'):
                path = BASE / arm / name
                if name == 'first_error.json' and selected.name == gpu_entry.POST_TEST_ENTRY:
                    continue
                errors.extend(receipt_errors(path, optional(path)))
        for name in ('compare_entry.json', 'compare_complete.json', 'watch_receipt.json', 'github_publication.json',
                     'comparison/first_error.json'):
            if name == 'watch_receipt.json' and DELIVERY_CONTINUATION:
                delivery_history()
                preserved.extend(admin.receipt_errors(BASE / name, optional(BASE / name)))
                continue
            errors.extend(receipt_errors(BASE / name, optional(BASE / name)))
        if DELIVERY_CONTINUATION:
            errors.extend(receipt_errors(BASE / 'delivery_watch_receipt.json', optional(BASE / 'delivery_watch_receipt.json')))
        compared = comparison_ready()
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        errors.append(dict(error=str(error)))
        compared = False
    ready = len(arms) == 2 and all(row['ready'] for row in arms)
    compared = compared and admin.rc0(optional(BASE / 'compare_entry.json'))
    complete = ready and compared and not errors
    result = dict(experiment='R102', recovery='POSTPROCESS', complete=complete, all_arms_ready=ready,
        comparison_complete=compared, arms=arms, errors=errors,
        preserved_uuid_preflight_failures=[row for row in preserved if Path(row['path']).name == 'gpu_entry.json'],
        preserved_prior_recovery_failures=preserved,
        phase='failed' if errors else 'complete_with_disclosed_postprocess_recovery' if complete
              else 'waiting_for_comparison' if ready else 'waiting_for_recovery',
        original_failed_route_complete=False, original_watcher_preserved=True,
        goal_complete=False, goal_status='active_unresolved')
    if persist:
        resource.save(BASE / 'goal_progress.json', result)
    return result


def compare_once(evidence):
    require_ready(evidence)
    current = status()
    if current['errors'] or not current['all_arms_ready'] or current['comparison_complete']:
        raise RuntimeError('CPU compare requires BOTH locked GPU/report pipelines, exactly once')
    path = BASE / 'compare_entry.json'
    argv = [sys.executable, '-B', '-u', '-m', 'code.V4.r102_postprocess_recovery',
            'compare', '--root', str(ROOT), '--device', 'cpu']
    value = dict(state='running', started_utc=utc(), argv=argv, cpu_only=True, automatic_retry=False,
                 main_ready_evidence=evidence, originals={arm: original(arm) for arm in ARMS},
                 source_provenance=source_provenance(evidence))
    with resource.lock(BASE / 'compare.lock'):
        resource.save(path, value, exclusive=True)
        try:
            with (BASE / 'compare.log').open('a') as log:
                result = subprocess.run(argv, cwd=REPO, env=admin.cpu_environment(),
                    stdout=log, stderr=subprocess.STDOUT, check=False)
            if result.returncode or not comparison_ready():
                raise RuntimeError('CPU compare requires rc0 and complete same-lock comparison; no retry')
            require_ready(evidence)
            if value['originals'] != {arm: original(arm) for arm in ARMS}:
                raise RuntimeError('Original failure evidence changed during comparison')
            value.update(state='complete', exit_status=0, completed_utc=utc())
        except BaseException as error:
            value.update(state='failed', exit_status=1, error=str(error), completed_utc=utc())
            raise
        finally:
            resource.save(path, value)
    return value


def publication_paths(version, root, launch_root, git_root):
    if version != 'R102' or Path(root).resolve() != BASE.resolve() or Path(launch_root).resolve() != LAUNCH.resolve():
        raise RuntimeError('Only the canonical whole R102 recovery may be published')
    from .r98_launch import source_closure
    git_root = Path(git_root).resolve()
    code = REPO / 'code/V4'
    sources = required_sources()
    for pattern in ('r102_*.py', 'test_r102*.py', 'run_v4_r102*.sh', 'r102*.md'):
        sources += list(code.glob(pattern))
    sources += [code / name for name in ('r98_goal.py', 'r98_launch.py', 'r84_launch.py', 'r81_queue.py',
        'r80_reporting.py', 'r84_reporting.py', 'r85_reporting.py', 'r87_reporting.py')]
    for arm in ARMS:
        sources += [code / name for name in optional(ROOT / arm / 'protocol.json').get('source_sha256', {})]
    candidates = [Path(path) for path in source_closure(sources)]
    candidates += [ROOT / name for name in ('protocol_lock.json', 'goal_watch_receipt.json')]
    for arm in ARMS:
        candidates += [ROOT / arm / name for name in ('protocol.json', 'initialization.json', 'final_test_lock.json',
            'training_complete.json', 'final_metrics.json', 'status.json', 'error.json', 'test_started.json', 'metrics.csv')]
    pruned = admin.PRUNED | {'report_view', 'cache', 'routing_cache', 'donor_cache', 'exports', 'pretrained', 'sources',
                            'test_features', 'test_cache', 'feature_cache', 'routing_features'}
    for directory in (BASE, *(VIEW / arm / 'full_model_statistics' for arm in ARMS), VIEW / 'comparison_artifacts'):
        for parent, dirs, files in os.walk(directory):
            dirs[:] = [name for name in dirs if name not in pruned and not (Path(parent) / name).is_symlink()]
            candidates += [Path(parent) / name for name in files if Path(name).suffix.lower() in {'.json', '.md', '.csv', '.pdf'}]
    candidates += [VIEW / 'RESULTS.pdf']
    candidates += [VIEW / 'RESULTS.md']
    for arm in ARMS:
        candidates += [VIEW / arm / name for name in ('test_results.json', 'inference_timing.json', 'lib_stage_complete.json')]
    excluded = ('prediction', 'label', 'outcome', 'raw_cost', 'cost_array', 'winner_array', 'per_instance', 'wandb', 'token')
    result = set()
    for path in candidates:
        if (not path.is_file() or path.is_symlink() or path.name.lower().startswith(excluded)
                or any(p.is_symlink() for p in path.parents)
                or path.suffix.lower() not in {'.py', '.sh', '.md', '.json', '.csv', '.pdf'}):
            continue
        resolved = path.resolve()
        if not resolved.is_relative_to(REPO.resolve()) or not resolved.is_relative_to(git_root):
            raise RuntimeError('Curated R102 source/report escapes the repository')
        if path.stat().st_size > admin.MAX_ASSET_BYTES:
            raise RuntimeError('Curated source/report exceeds32MiB')
        if path.suffix == '.json' and not admin._safe_json(optional(path)):
            continue
        result.add(str(resolved.relative_to(git_root)))
    return sorted(result)


def publish(evidence):
    require_ready(evidence)
    if not status()['complete']:
        raise RuntimeError('No WIP publication: both arms and one comparison must be complete')
    source_provenance(evidence)
    # Only a fresh private administrative adapter is changed, never the old watcher.
    base = previous.private()
    base.model_root = lambda version: BASE
    base.status = lambda *args, **kwargs: status()
    base.publication_paths = publication_paths
    return base.publish('R102', BASE, LAUNCH)


def watch(evidence, *, interval=120, publish_on_completion=False):
    if interval < 120:
        raise ValueError('Recovery watcher must stay quiet for at least120 seconds between polls')
    require_ready(evidence)
    path = BASE / ('delivery_watch_receipt.json' if DELIVERY_CONTINUATION else 'watch_receipt.json')
    value = dict(experiment='R102', recovery='POSTPROCESS', state='watching', started_utc=utc(), pid=os.getpid(),
        interval=interval, cpu_only=True, automatic_retry=False, main_ready_evidence=evidence,
        publication_authorized=publish_on_completion, original_watcher_preserved=True,
        originals={arm: original(arm) for arm in ARMS}, goal_complete=False)
    with resource.lock(BASE / 'watch.lock'):
        resource.save(path, value, exclusive=True)
        try:
            source_provenance(evidence)
            while True:
                require_ready(evidence)
                current = status(persist=True)
                if current['errors']:
                    raise RuntimeError('Recovery failed; preserve every original/new receipt, no automatic retry')
                if current['all_arms_ready'] and not current['comparison_complete']:
                    compare_once(evidence)
                    continue
                if current['complete']:
                    if publish_on_completion:
                        publish(evidence)
                    value.update(state='complete', exit_status=0, completed_utc=utc(), delivery_complete=True)
                    resource.save(path, value)
                    return 0
                time.sleep(interval)
        except BaseException as error:
            value.update(state='stopped_on_error', exit_status=1, error=str(error), completed_utc=utc())
            resource.save(path, value)
            return 1


def parser():
    value = argparse.ArgumentParser(description=__doc__)
    actions = value.add_mutually_exclusive_group()
    actions.add_argument('--watch', action='store_true')
    actions.add_argument('--compare', action='store_true')
    value.add_argument('--publish', action='store_true')
    value.add_argument('--interval', type=int, default=120)
    value.add_argument('--evidence')
    value.add_argument('--delivery-continuation', action='store_true',
        help='Explicit CPU-only continuation; preserve the failed first administrative watcher')
    return value


def main(argv=None):
    global DELIVERY_CONTINUATION
    args = parser().parse_args(argv)
    DELIVERY_CONTINUATION = args.delivery_continuation
    if args.watch:
        return watch(args.evidence, interval=args.interval, publish_on_completion=args.publish)
    if args.compare:
        compare_once(args.evidence)
    if args.publish:
        publish(args.evidence)
    print(json.dumps(status(), indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

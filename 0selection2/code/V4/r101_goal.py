"""Quiet CPU R101 whole-version delivery gate and one-shot curated publication.

Reuse a PRIVATE r98_goal instance, never alter a running historical watcher.
Stage I and Stage II budgets remain separate. No teacher arrays, final model
weights, pretraining weights, training, inference or GPU launch are opened.
Only the final saved-data R101 CPU comparison may be run, once both arms pass.
"""

import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time


REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / 'code/V4/runs/R101_hint_pretraining'
LAUNCH = REPO / 'code/V4/runs/R98_parallel_dispatch'
ARMS = ('A', 'B')


def private():
    spec = importlib.util.spec_from_file_location('code.V4._r101_goal_admin', REPO / 'code/V4/r98_goal.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


adapter = private()
PROBLEMS, optional, save, utc = adapter.PROBLEMS, adapter.optional, adapter.save, adapter.utc
file_hash, json_hash = adapter.file_hash, adapter.json_hash
TEACHER_TASKS = tuple(p for p in PROBLEMS if p not in ('TSP', 'ATSP'))
_old_common, _old_checks, _old_status = adapter.common_checks, adapter.r99_checks, adapter.status


def arm_checks(version, arm, run, records):
    checks = _old_checks(arm, run, records)
    config, lock = records['protocol.json'], records['final_test_lock.json']
    pretrain = optional(run / 'pretrain_complete.json')
    init = optional(run / 'initialization.json')
    lineage = config.get('pretraining_lineage', {})
    spec = config.get('hint_pretraining', {})
    teacher = pretrain.get('teacher_identity', {})
    checks['Stage_I_12800'] = (pretrain.get('experiment') == 'R101' and pretrain.get('arm') == arm
        and pretrain.get('complete') is True and pretrain.get('successful_updates') == 12800
        and pretrain.get('selected_without_validation') is True
        and pretrain.get('pretrain_spec') == spec and spec.get('updates') == 12800
        and spec.get('logical_batch') == 1024 and spec.get('label_inputs') is False
        and spec.get('teacher_split') == 'train' and set(spec.get('teacher_scope', [])) == set(TEACHER_TASKS)
        and set(pretrain.get('updates_by_problem', {})) == set(TEACHER_TASKS)
        and all(pretrain['updates_by_problem'][p] == 800 for p in TEACHER_TASKS)
        and teacher.get('variant') == {'A': 'random', 'B': 'pretrained'}[arm]
        and teacher.get('split') == 'train' and teacher.get('rows_per_task') == 81000
        and teacher.get('no_teacher_in_final_model') is True
        and set(teacher.get('metadata_sha256', {})) == set(TEACHER_TASKS))
    fields = ('experiment', 'arm', 'complete', 'successful_updates', 'checkpoint_sha256', 'encoder_sha256',
        'initial_trainable_sha256', 'unchanged_non_encoder_sha256', 'teacher_identity', 'pretrain_spec', 'protocol_sha256')
    checks['parent_pretrain_SHA'] = (bool(pretrain.get('checkpoint_sha256'))
        and all(key in pretrain and lineage.get(key) == pretrain[key] for key in fields)
        and config.get('main_protocol_lock', {}).get('sha256') == pretrain.get('protocol_sha256')
        and config.get('pretrained_encoder_sha256') == pretrain.get('encoder_sha256')
        and init.get('pretrained_encoder_sha256') == pretrain.get('encoder_sha256'))
    checks['non_encoder_fresh_and_reset'] = (bool(pretrain.get('unchanged_non_encoder_sha256'))
        and init.get('unchanged_non_encoder_sha256') == pretrain.get('unchanged_non_encoder_sha256')
        and config.get('unchanged_non_encoder_sha256') == pretrain.get('unchanged_non_encoder_sha256')
        and init.get('initial_trainable_sha256') == pretrain.get('initial_trainable_sha256')
        and init.get('r77_fresh_trainable_sha256') == pretrain.get('initial_trainable_sha256')
        and init.get('historical_selector_checkpoint_loaded') is False
        and all(init.get(key) is True for key in ('optimizer_reset', 'scaler_reset', 'rng_reset', 'task_sample_streams_reset'))
        and init.get('transferred_parameter_prefix') == 'instance_encoder.'
        and init.get('trainable_parameter_count') == config.get('trainable_parameter_count') == 1842953
        and config.get('new_trainable_parameters') == 0)
    timing = records['inference_timing.json']
    checks['final_student_no_teacher'] = (config.get('deployment_teacher') is False
        and config.get('downstream_teacher_inputs') is False and config.get('downstream_hint_loss') is False
        and timing.get('student_only') is True and timing.get('teacher_arrays_opened') is False
        and timing.get('online_teacher') is False)
    post, report = records['postprocess_status.json'], records['full_model_statistics/protocol.json']
    checks['CPU_report'] = (post.get('complete') is True and post.get('phase') == 'complete'
        and post.get('exit_status') == 0 and report.get('complete') is True
        and report.get('experiment') == 'R101' and report.get('checkpoint_lock') == lock)
    for name in adapter.R99_PDFS:
        checks[name] = adapter.pdf_ready(run / 'full_model_statistics' / name)
    evidence = report.get('pretraining_evidence', {})
    receipt_path, trace_path = run / 'pretrain_complete.json', run / 'pretrain_updates.jsonl'
    checks['pretraining_report_bound'] = (receipt_path.is_file() and trace_path.is_file()
        and evidence.get('receipt_sha256') == file_hash(receipt_path)
        and evidence.get('trace_sha256') == file_hash(trace_path))
    return checks


def status(version='R101', root=None, launch_root=LAUNCH, *, persist=True):
    if version != 'R101':
        raise ValueError('This private watcher owns R101 only')
    root, launch_root = Path(root or ROOT).resolve(), Path(launch_root).resolve()
    result = _old_status('R101', root, launch_root, persist=False)
    for row in result['arms']:
        # Per-arm reports live under full_model_statistics; RESULTS.pdf is version-wide.
        row['checks'].pop('RESULTS.pdf', None)
        row['missing'] = [key for key, value in row['checks'].items() if not value]
        row['ready'] = all(row['checks'].values())
    result['all_arms_ready'] = all(row['ready'] for row in result['arms'])
    configs, locks = {}, {}
    for arm in ARMS:
        run = root / arm
        configs[arm], locks[arm] = optional(run / 'protocol.json'), optional(run / 'final_test_lock.json')
        for name in ('pretrain_status.json', 'pretrain_complete.json', 'initialization.json'):
            result['errors'].extend(adapter.receipt_errors(run / name, optional(run / name)))
    matched = (bool(configs['A']) and bool(configs['B'])
        and all(configs['A'].get(key) == configs['B'].get(key) for key in
            ('data_root', 'prepared_root', 'rows_per_problem', 'candidate_pools', 'candidate_pool_ids',
             'baselines', 'hint_pretraining', 'max_successful_updates', 'inference_rule',
             'unchanged_non_encoder_sha256', 'trainable_parameter_count')))
    comparison = optional(root / 'comparison_artifacts/protocol.json')
    compared = (comparison.get('experiment') == 'R101' and comparison.get('complete') is True
        and comparison.get('exit_status') == 0 and comparison.get('checkpoints') == locks
        and adapter.pdf_ready(root / 'comparison_artifacts/COMPARISON.pdf') and adapter.pdf_ready(root / 'RESULTS.pdf'))
    ready = result['all_arms_ready'] and matched
    complete = ready and compared and not result['errors']
    result.update(all_arms_ready=ready, matched_stage_I_II=matched, comparison_complete=compared,
        complete=complete, phase='failed' if result['errors'] else 'complete' if complete
        else 'waiting_for_comparison' if ready else 'waiting_for_arms',
        stage_I_updates_per_arm=12800, stage_II_updates_per_arm=57600,
        goal_complete=False, goal_status='active_unresolved')
    if persist:
        path = root / 'goal_progress.json'
        prior = optional(path)
        if {k: v for k, v in prior.items() if k != 'updated_utc'} == result:
            return prior
        result['updated_utc'] = utc()
        save(path, result)
    return result


def compare_once(root=ROOT, launch_root=LAUNCH, **references):
    root = Path(root).resolve()
    current = status(root=root, launch_root=launch_root)
    if current['errors'] or not current['all_arms_ready'] or current['comparison_complete']:
        raise RuntimeError('R101 compare requires both complete matched Stage-I/II and native TEST/LIB/PDF pipelines')
    path = root / 'goal_compare_receipt.json'
    with (root / 'goal_compare.lock').open('a') as claim:
        adapter.fcntl.flock(claim, adapter.fcntl.LOCK_EX | adapter.fcntl.LOCK_NB)
        if path.exists():
            raise RuntimeError('Comparison already attempted; preserve receipt, no automatic retry')
        argv = [sys.executable, '-B', '-u', '-m', 'code.V4.r101_reporting', 'compare',
                '--root', str(root), '--device', 'cpu']
        for key in adapter.REFERENCE_FLAGS:
            if references.get(key) is not None:
                argv.extend(['--' + key.replace('_', '-'), str(Path(references[key]).resolve())])
        receipt = dict(state='running', started_utc=utc(), argv=argv, cpu_only=True, automatic_retry=False)
        save(path, receipt)
        try:
            with (root / 'goal_compare.log').open('a', encoding='utf-8') as log:
                process = subprocess.run(argv, cwd=REPO, env=adapter.cpu_environment(),
                    stdout=log, stderr=subprocess.STDOUT, check=False)
            if process.returncode or not status(root=root, launch_root=launch_root)['complete']:
                raise RuntimeError('R101 compare failed or lacks same-lock final PDFs; no retry')
            receipt.update(state='complete', exit_status=0, completed_utc=utc())
        except BaseException as error:
            receipt.update(state='failed', exit_status=1, error=str(error), completed_utc=utc())
            raise
        finally:
            save(path, receipt)


def publication_paths(version, root, launch_root, git_root):
    root, launch_root, git_root = (Path(p).resolve() for p in (root, launch_root, git_root))
    if version != 'R101' or root != ROOT.resolve() or launch_root != LAUNCH.resolve():
        raise RuntimeError('Publication accepts only canonical R101 roots')
    code = REPO / 'code/V4'
    candidates = [p for pattern in ('r101_*.py', 'test_r101*.py', 'run_v4_r101*.sh', 'r101*README*.md')
                  for p in code.glob(pattern)]
    for run in (root, root / 'A', root / 'B'):
        candidates += [p for p in run.glob('*') if p.suffix.lower() in {'.json', '.csv', '.pdf', '.md'}
            and not p.name.startswith(('goal_', 'github_', 'wandb', 'pretrain_wandb', 'prediction', 'outcome'))]
        for directory in ('preparation', 'prep', 'full_model_statistics', 'comparison_artifacts'):
            for parent, dirs, files in os.walk(run / directory):
                dirs[:] = [name for name in dirs if name not in adapter.PRUNED and name not in
                           ('teacher_cache', 'teacher', 'sources', 'runtime') and not (Path(parent) / name).is_symlink()]
                candidates += [Path(parent) / name for name in files
                               if Path(name).suffix.lower() in {'.json', '.csv', '.pdf', '.md'}]
    for arm in ARMS:
        candidates.append(launch_root / 'specs' / ('R101_'+arm+'.json'))
        candidates += [launch_root / ('R101_'+arm) / name for name in
                       ('plan.json', 'ready.json', 'worker.json', 'controller.json', 'teacher_stage.json')]
    result = set()
    for path in candidates:
        if (not path.is_file() or path.is_symlink() or 'wandb' in path.name.lower()
                or any(parent.is_symlink() for parent in path.parents if parent != git_root)):
            continue
        if not path.resolve().is_relative_to(REPO.resolve()) or not path.resolve().is_relative_to(git_root):
            raise RuntimeError('Curated R101 asset escapes repository')
        if path.stat().st_size > adapter.MAX_ASSET_BYTES:
            raise RuntimeError('Curated R101 metadata/report unexpectedly exceeds32MiB')
        if path.suffix.lower() == '.json' and not adapter._safe_json(optional(path)):
            continue
        result.add(str(path.resolve().relative_to(git_root)))
    if not result:
        raise RuntimeError('No curated R101 assets to publish')
    return sorted(result)


# Only the private administrative instance is adapted; existing watcher globals stay intact.
adapter.VERSIONS = dict(R101=ROOT.name)
adapter.arms = lambda version: ARMS
adapter.run_root = lambda version, root, arm: root / arm
adapter.common_checks = lambda version, arm, run, records: _old_common('R101', arm, run, records)
adapter.frozen_checks = arm_checks
adapter.FROZEN_PDFS = (*adapter.R99_PDFS, 'pretraining_curves.pdf')
adapter.status = status
adapter.publication_paths = publication_paths


def publish(root=ROOT, launch_root=LAUNCH):
    return adapter.publish('R101', root, launch_root)


def watch(root=ROOT, launch_root=LAUNCH, interval=120, publish_github=False, **references):
    if interval <= 0:
        raise ValueError('Watcher interval must be positive')
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    with (root / 'goal_watch.lock').open('a') as claim:
        adapter.fcntl.flock(claim, adapter.fcntl.LOCK_EX | adapter.fcntl.LOCK_NB)
        path = root / 'goal_watch_receipt.json'
        if path.exists():
            prior = optional(path)
            if prior.get('state') == 'complete' and prior.get('exit_status') == 0:
                return 0
            raise RuntimeError('Watcher already attempted; preserve original receipt, no retry')
        receipt = dict(experiment='R101', state='watching', started_utc=utc(), cpu_only=True,
            github_on_completion=publish_github, automatic_retry=False, goal_complete=False, goal_status='active_unresolved')
        save(path, receipt)
        previous = None
        try:
            while True:
                current = status(root=root, launch_root=launch_root)
                signature = json_hash({k: v for k, v in current.items() if k != 'updated_utc'})
                if signature != previous:
                    print(json.dumps(dict(version='R101', phase=current['phase'],
                        missing={r['arm']: r['missing'] for r in current['arms']}, errors=current['errors'])), flush=True)
                    previous = signature
                if current['errors']:
                    raise RuntimeError('R101 stage failed; no training, model repair or retry')
                if current['all_arms_ready'] and not current['comparison_complete']:
                    compare_once(root, launch_root, **references)
                    continue
                if current['complete']:
                    if publish_github:
                        publish(root, launch_root)
                    receipt.update(state='complete', exit_status=0, pipeline_complete=True, completed_utc=utc())
                    save(path, receipt)
                    return 0
                time.sleep(interval)
        except BaseException as error:
            receipt.update(state='stopped_on_error', exit_status=1, error=str(error), completed_utc=utc())
            save(path, receipt)
            return 1


def parser():
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument('--watch', action='store_true')
    value.add_argument('--publish-github', action='store_true')
    value.add_argument('--root', type=Path, default=ROOT)
    value.add_argument('--launch-root', type=Path, default=LAUNCH)
    value.add_argument('--interval', type=int, default=120)
    for flag in adapter.REFERENCE_FLAGS:
        value.add_argument('--' + flag.replace('_', '-'), type=Path)
    return value


def main(argv=None):
    args = parser().parse_args(argv)
    if args.watch:
        return watch(args.root, args.launch_root, args.interval, args.publish_github,
            **{key: getattr(args, key) for key in adapter.REFERENCE_FLAGS})
    if args.publish_github:
        publish(args.root, args.launch_root)
    else:
        print(json.dumps(status(root=args.root, launch_root=args.launch_root), indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

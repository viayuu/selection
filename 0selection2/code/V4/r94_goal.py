"""Quiet R94 CPU completion watcher and opt-in serialized version publication.

Never trains, launches, infers, retries failed stages or selects a checkpoint.
Only saved JSON/PDF/CSV receipts are inspected, not checkpoints or raw arrays.
No watcher or publication starts on import. Reuses pure R93 administration
helpers without rebinding any imported shared module global.
"""

import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from . import r93_goal as r93


REPO = Path(__file__).resolve().parents[2]
MODEL_ROOT = REPO / 'code/V4/runs/R94_pretrained_node_fusion'
ROOT = DEFAULT_ROOT = MODEL_ROOT
LAUNCH = ROOT.parent / 'R94_launch'
ARMS = ('A', 'B')
BUDGETS = dict.fromkeys(ARMS, 57600)
PROBLEMS = r93.PROBLEMS
REFERENCE_FLAGS = ('nss_root', 'historical_root', 'r89_root')
REQUIRED_PDFS = ('MODEL_STATISTICS.pdf', 'selection_distributions.pdf',
                 'target_recovery_and_train.pdf', 'learning_curves.pdf',
                 'training_and_method_diagnostics.pdf')
utc, optional, save = r93.utc, r93.optional, r93.save
cpu_environment, nonempty = r93.cpu_environment, r93.nonempty


def json_hash(record):
    return hashlib.sha256(json.dumps(record, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def file_hash(path):
    with Path(path).open('rb') as stream:
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def receipt_errors(arm, path, record):
    errors = r93.receipt_errors(arm, path, record)
    for stage in record.get('stages', []):
        for nested in stage.get('stages', []):
            errors.extend(receipt_errors(arm, path, nested))
    return errors


def _split_ready(result, arm, split, rows, lock, model_hash):
    if not isinstance(result, dict):
        return False
    identity = result.get('identity', {})
    per_problem = result.get('per_problem', {})
    return (bool(model_hash) and set(per_problem) == set(PROBLEMS)
        and all(r.get('n') == rows for r in per_problem.values())
        and identity.get('experiment') == 'R94' and identity.get('arm') == arm
        and identity.get('split') == split and identity.get('successful_updates') == lock.get('selected_update')
        and identity.get('state_hash') == lock.get('state_hash')
        and identity.get('model_state_sha256') == model_hash
        and identity.get('checkpoint_sha256') in (None, lock.get('checkpoint_sha256')))


def timing_ready(record, lock):
    """Small timing receipt check; no model, cache, or native-asset imports."""
    if (record.get('complete') is not True or not lock.get('checkpoint_sha256')
            or record.get('checkpoint_sha256') != lock.get('checkpoint_sha256')
            or record.get('state_hash') != lock.get('state_hash')
            or record.get('inference_rule') != lock.get('inference_rule')
            or record.get('includes_native_encoder') is not True
            or record.get('repetitions') != 3 or not record.get('scope') or not record.get('excludes')):
        return False
    rows = record.get('per_problem', [])
    if isinstance(rows, dict):
        rows = [dict(v, problem=k) for k, v in rows.items()]
    if (not isinstance(rows, list) or len(rows) != len(PROBLEMS)
            or any(not isinstance(r, dict) for r in rows)
            or {r.get('problem') for r in rows} != set(PROBLEMS)):
        return False
    try:
        for row in rows:
            if row.get('instances') != 128:
                return False
            for scope in ('donor_plus_selector', 'cache_hit_selector'):
                measurement = row.get(scope, {})
                samples, value = measurement.get('batch_seconds', []), measurement.get('mean_seconds_per_instance')
                if (not isinstance(samples, list) or len(samples) != 3
                        or any(isinstance(v, bool) or not math.isfinite(float(v)) or float(v) <= 0 for v in [*samples, value])
                        or not math.isclose(float(value), sum(samples) / (3 * 128), rel_tol=1e-6, abs_tol=1e-12)):
                    return False
    except (ValueError, TypeError, AttributeError):
        return False
    return True


def status(root=ROOT, launch_root=LAUNCH):
    root, launch_root = Path(root).resolve(), Path(launch_root).resolve()
    arms, errors, locks, timing_hashes = [], [], {}, {}
    for arm, budget in BUDGETS.items():
        run, launch = root / arm, launch_root / arm
        names = ('protocol.json', 'status.json', 'training_complete.json', 'final_test_lock.json',
                 'best_eval.json', 'final_metrics.json', 'test_results.json', 'inference_timing.json',
                 'lib_stage_complete.json', 'postprocess_status.json')
        records = {name: optional(run / name) for name in names}
        config, live, train, lock, best, final, test, timing, lib, post = (records[k] for k in names)
        locks[arm] = lock
        protocol = optional(run / 'full_model_statistics/protocol.json')
        controller, worker = (optional(launch / k) for k in ('controller.json', 'worker.json'))
        checkpoint, state_hash = lock.get('checkpoint_sha256'), lock.get('state_hash')
        test_result = test.get('result', {})
        test_identity = test_result.get('identity', {})
        model_hash = test_identity.get('model_state_sha256')
        selected_update = lock.get('selected_update', 0)
        timing_path = run / 'inference_timing.json'
        timing_hashes[arm] = file_hash(timing_path) if timing_path.is_file() else None
        same_lock = protocol.get('checkpoint_lock', {})
        required = dict(
            protocol=config.get('experiment') == 'R94' and config.get('arm') == arm
                and config.get('r94_schema_version') == 1
                and config.get('initialization') == 'scratch_shared_r77_plus_node_fusion'
                and config.get('frozen_donor', {}).get('variant') == {'A': 'random', 'B': 'pretrained'}[arm]
                and config.get('max_successful_updates') == budget and bool(state_hash)
                and json_hash(config) == state_hash,
            training_complete=train.get('successful_updates') == budget
                and train.get('state_hash') == state_hash
                and all(train.get('updates_by_problem', {}).get(p) == budget // len(PROBLEMS) for p in PROBLEMS),
            unique_VAL_lock=lock.get('experiment') == 'R94' and lock.get('arm') == arm
                and lock.get('selected_without_test') is True and lock.get('per_problem_assembly') is False
                and isinstance(selected_update, int) and not isinstance(selected_update, bool)
                and 0 < selected_update <= budget and bool(checkpoint)
                and lock.get('rule') == config.get('checkpoint_rule')
                and 'macro_vs_sbs' in str(lock.get('rule', ''))
                and bool(lock.get('inference_rule')) and lock.get('inference_rule') == config.get('inference_rule'),
            full_TRAIN=best.get('successful_updates') == selected_update
                and _split_ready(best.get('train', {}), arm, 'train', 81000, lock, model_hash)
                and _split_ready(final.get('train', {}), arm, 'train', 81000, lock, model_hash),
            full_VAL=final.get('selected_update') == selected_update
                and final.get('checkpoint_sha256') == checkpoint and final.get('state_hash') == state_hash
                and _split_ready(final.get('val', {}), arm, 'val', 4500, lock, model_hash),
            full_TEST=test.get('complete') is True and test.get('checkpoint_sha256') == checkpoint
                and test.get('state_hash') == state_hash and test_identity.get('checkpoint_sha256') == checkpoint
                and _split_ready(test_result, arm, 'test', 4500, lock, model_hash),
            inference_timing=timing_ready(timing, lock),
            LIB=lib.get('complete') is True and lib.get('exit_status') == 0
                and lib.get('checkpoint_sha256') == checkpoint
                and lib.get('state_hash') == state_hash and lib.get('model_state_sha256') == model_hash
                and set(lib.get('datasets', [])) == {'TSPLIB', 'CVRPLIB'},
            PDF=post.get('complete') is True and post.get('phase') == 'complete' and post.get('exit_status') == 0
                and protocol.get('complete') is True and same_lock.get('checkpoint_sha256') == checkpoint
                and same_lock.get('state_hash') == state_hash
                and protocol.get('inference_timing', {}).get('status') == 'complete'
                and protocol.get('inference_timing', {}).get('pdf_panel') is True
                and bool(timing_hashes[arm]) and protocol.get('inference_timing', {}).get('sha256') == timing_hashes[arm],
            controller_rc0=controller.get('state') == 'complete' and controller.get('exit_status') == 0,
            worker_rc0=worker.get('state') == 'complete' and worker.get('exit_status') == 0)
        for name in REQUIRED_PDFS:
            required[name] = nonempty(run / 'full_model_statistics' / name)
        for name, record in records.items():
            errors.extend(receipt_errors(arm, run / name, record))
        for name in ('controller.json', 'worker.json', 'status.json'):
            errors.extend(receipt_errors(arm, launch / name, optional(launch / name)))
        errors.extend(receipt_errors(arm, run / 'lib_stage_status.json', optional(run / 'lib_stage_status.json')))
        phase = live.get('phase', 'not_started')
        if phase == 'not_started':
            queue = optional(launch / 'status.json')
            phase = queue.get('state', queue.get('phase', phase))
        arms.append(dict(arm=arm, phase=phase, successful_updates=live.get('successful_updates', train.get('successful_updates', 0)),
            budget=budget, training_unit='successful_neural_updates', checks=required,
            missing=[key for key, ok in required.items() if not ok], ready=all(required.values()),
            ready_for_comparison=all(required.values())))
    for name in ('goal_compare_receipt.json', 'github_publication.json'):
        errors.extend(receipt_errors('watcher', root / name, optional(root / name)))
    failure = optional(root / 'goal_watch_error.json')
    if failure:
        errors.append(dict(arm='watcher', path=str(root / 'goal_watch_error.json'), error=failure.get('error')))
    comparison = optional(root / 'comparison_artifacts/protocol.json')
    receipt = optional(root / 'goal_compare_receipt.json')
    ready = all(row['ready'] for row in arms)
    delivered = (comparison.get('experiment') == 'R94' and comparison.get('complete') is True
        and comparison.get('exit_status') == 0
        and (not receipt or receipt.get('state') == 'complete' and receipt.get('exit_status') == 0)
        and nonempty(root / 'comparison_artifacts/COMPARISON.pdf') and nonempty(root / 'RESULTS.pdf')
        and all(comparison.get('checkpoints', {}).get(arm, {}) == locks[arm]
                and comparison.get('inference_timing', {}).get(arm, {}).get('sha256') == timing_hashes[arm] for arm in ARMS))
    complete = ready and delivered and not errors
    result = dict(experiment='R94', checked_utc=utc(), arms=arms, errors=errors,
        all_arms_ready=ready, comparison_complete=complete,
        phase='failed' if errors else 'comparison_complete' if complete else 'ready_for_comparison' if ready else 'waiting_for_arms',
        cpu_only=True, automatic_retry=False, goal_complete=False, goal_status='active_unresolved',
        note='Saved receipts only, not live GPU evidence. No training/launch/inference/retry starts here.')
    save(root / 'goal_progress.json', result)
    return result


def run_once(root, name, argv):
    root = Path(root)
    path = root / (name + '_receipt.json')
    with (root / (name + '.lock')).open('a') as claim:
        fcntl.flock(claim, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if path.exists():
            raise RuntimeError('Stage already attempted; no automatic retry: ' + name)
        receipt = dict(state='running', started_utc=utc(), argv=argv, cpu_only=True, automatic_retry=False)
        save(path, receipt)
        try:
            with (root / (name + '.log')).open('a', encoding='utf-8') as log:
                result = subprocess.run(argv, cwd=str(REPO), env=cpu_environment(),
                    stdout=log, stderr=subprocess.STDOUT, check=False)
            receipt.update(state='complete' if result.returncode == 0 else 'failed',
                exit_status=result.returncode, completed_utc=utc())
            if result.returncode:
                raise RuntimeError('CPU stage failed; inspect ' + name + '.log; no automatic retry')
        except BaseException as error:
            receipt.update(state='failed', error=str(error), completed_utc=utc())
            receipt.setdefault('exit_status', 1)
            raise
        finally:
            save(path, receipt)


def compare(root, launch_root=LAUNCH, **references):
    current = status(root, launch_root)
    if current['errors'] or not current['all_arms_ready']:
        raise RuntimeError('R94 compare requires both full-budget same-lock TRAIN/VAL/TEST/timing/LIB/PDF pipelines with rc0')
    argv = [sys.executable, '-B', '-u', '-m', 'code.V4.r94_reporting', 'compare',
            '--root', str(Path(root).resolve()), '--device', 'cpu']
    for key in REFERENCE_FLAGS:
        if references.get(key) is not None:
            argv.extend(['--' + key.replace('_', '-'), str(Path(references[key]).resolve())])
    run_once(root, 'goal_compare', argv)
    if not status(root, launch_root)['comparison_complete']:
        path = Path(root) / 'goal_compare_receipt.json'
        record = optional(path)
        record.update(state='failed', exit_status=1,
            error='Comparison rc0 without complete same-checkpoint timing/PDF delivery')
        save(path, record)
        raise RuntimeError(record['error'])


def publication_paths(root, launch_root, git_root):
    """Allowlist this complete version; never glob unrelated dirty work into Git."""
    root, launch_root, git_root = (Path(p).resolve() for p in (root, launch_root, git_root))
    if root != ROOT.resolve() or launch_root != LAUNCH.resolve():
        raise RuntimeError('Publication accepts only canonical R94 roots')
    files = list((REPO / 'code/V4').glob('r94_*.py'))
    files += list((REPO / 'code/V4').glob('test_r94*.py'))
    files += list((REPO / 'code/V4').glob('run_v4_r94*.sh'))
    files += [REPO / 'code/V4/diagnostics' / name for name in
              ('io_readaround_probe.py', 'run_io_readaround_probe.sh')]
    files += [REPO / 'docs/research' / name for name in (
        'pretrained_nco_representation_beyond_r50.md', 'nco_node_fusion_source_feasibility.md')]
    files += [root / name for name in ('protocol_lock.json', 'PLAN.md', 'LAUNCH.md',
        'RESULTS.md', 'RESULTS.pdf', 'IO_PIPELINE_RECOVERY.md', 'IO_PIPELINE_RECOVERY.pdf',
        'io_readaround_evidence.json', 'startup_slow_io_attempt1/ARCHIVE.json')]
    receipts = ('protocol.json', 'status.json', 'initialization.json', 'training_complete.json',
        'final_test_lock.json', 'best_eval.json', 'final_metrics.json', 'test_results.json',
        'history.json', 'inference_timing.json', 'lib_stage_complete.json', 'lib_stage_status.json', 'postprocess_status.json')
    statistics = {'protocol.json', 'RESULTS.md', *REQUIRED_PDFS, 'learning_curves.png',
        'metrics_by_problem.csv', 'single_method_comparison.csv', 'nss_comparison.csv',
        'selection_distribution.csv', 'method_selection_train_val_test.csv', 'zero_selection_summary.csv',
        'train_metrics_by_problem.csv', 'val_empirical_screen.csv', 'inference_timing.csv',
        'target_recovery.csv', 'target_recovery_summary.csv'}
    for stem in ('paired_changes', 'target_recovery', 'target_recovery_summary'):
        statistics.update(stem + '_vs_' + ref + '.csv' for ref in ('R89B', 'R84A'))
    for arm in ARMS:
        run, launch = root / arm, launch_root / arm
        files += [run / name for name in receipts]
        files += [run / 'full_model_statistics' / name for name in statistics]
        files += [launch / name for name in ('controller.json', 'worker.json', 'status.json')]
        files += [run / 'lib_evaluation' / (dataset + '_results.json') for dataset in ('TSPLIB', 'CVRPLIB')]
        files += [run / 'full_model_statistics/figures' / (problem + '_selection_distribution.' + suffix)
                  for problem in PROBLEMS for suffix in ('pdf', 'png')]
    files += [launch_root / name for name in ('protocol.json', 'launch_startup_check.json', 'launch_receipt.json')]
    files += [root / 'comparison_artifacts' / name for name in (
        'protocol.json', 'COMPARISON.pdf', 'comparison.csv', 'paired_changes_B_vs_A.csv',
        'target_recovery_B_vs_A.csv', 'target_recovery_summary_B_vs_A.csv', 'val_empirical_screen.csv')]
    result = set()
    for path in files:
        if not path.is_file():
            continue
        # Reject symlinked ancestors as well as symlinked files.
        if path.is_symlink() or any(parent.is_symlink() for parent in path.parents if parent != git_root):
            continue
        if path.stat().st_size > 32 * 1024 * 1024:
            raise RuntimeError('Curated R94 asset unexpectedly exceeds32MiB: ' + str(path))
        relative = path.resolve().relative_to(git_root)
        if path.suffix == '.json' and r93._sensitive_json(optional(path)):
            raise RuntimeError('Sensitive JSON excluded from publication: ' + str(path))
        result.add(str(relative))
    if not result:
        raise RuntimeError('No curated R94 assets to publish')
    return sorted(result)


def git_output(argv, cwd):
    return subprocess.check_output(['git', *argv], cwd=str(cwd), text=True).strip()


def publish(root=ROOT, launch_root=LAUNCH):
    """One opt-in curated commit/push under the shared cross-version Git lock."""
    root, launch_root = Path(root).resolve(), Path(launch_root).resolve()
    git_root = Path(git_output(['rev-parse', '--show-toplevel'], REPO)).resolve()
    git_dir = Path(git_output(['rev-parse', '--absolute-git-dir'], REPO)).resolve()
    with (git_dir / 'codex-publication.lock').open('a') as mutex:
        fcntl.flock(mutex, fcntl.LOCK_EX)
        current = status(root, launch_root)
        if current['errors'] or not current['comparison_complete']:
            raise RuntimeError('Publish requires the complete R94 version, not individual completed arms')
        receipt_path = root / 'github_publication.json'
        prior = optional(receipt_path)
        if prior.get('complete') is True:
            return prior
        if receipt_path.exists():
            raise RuntimeError('Publication already attempted; no automatic staging/commit/push retry')
        if git_output(['branch', '--show-current'], git_root) != 'main':
            raise RuntimeError('Expected main; no branch was changed')
        if git_output(['diff', '--cached', '--name-only', '-z'], git_root):
            raise RuntimeError('Pre-existing staged changes; nothing staged or reverted by R94')
        paths = publication_paths(root, launch_root, git_root)
        receipt = dict(complete=False, state='staging', started_utc=utc(), files=paths,
            remote='origin/main', curated_only=True, automatic_retry=False,
            global_mutex=str(git_dir / 'codex-publication.lock'))
        save(receipt_path, receipt)
        try:
            for start in range(0, len(paths), 80):
                subprocess.run(['git', 'add', '-f', '--', *paths[start:start + 80]], cwd=str(git_root), check=True)
            staged = set(filter(None, git_output(['diff', '--cached', '--name-only', '-z'], git_root).split('\0')))
            if not staged or not staged.issubset(set(paths)) or git_output(['branch', '--show-current'], git_root) != 'main':
                raise RuntimeError('Index/branch changed; retain it untouched for inspection')
            receipt.update(state='committing', files=sorted(staged))
            save(receipt_path, receipt)
            subprocess.run(['git', 'commit', '-m', 'r94: publish completed frozen-donor node-fusion controls'], cwd=str(git_root), check=True)
            receipt.update(state='committed_pending_push', commit=git_output(['rev-parse', 'HEAD'], git_root))
            save(receipt_path, receipt)
            subprocess.run(['git', 'push', 'origin', 'HEAD:refs/heads/main'], cwd=str(git_root), check=True)
            receipt.update(complete=True, state='complete', exit_status=0, completed_utc=utc())
            save(receipt_path, receipt)
            return receipt
        except BaseException as error:
            receipt.update(state='failed', error=str(error), exit_status=1, completed_utc=utc())
            save(receipt_path, receipt)
            raise


def watch(root=ROOT, launch_root=LAUNCH, interval=120, publish_github=False, **references):
    if interval <= 0:
        raise ValueError('Watcher interval must be positive')
    root, launch_root = Path(root).resolve(), Path(launch_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    with (root / 'goal_watch.lock').open('a') as claim:
        fcntl.flock(claim, fcntl.LOCK_EX | fcntl.LOCK_NB)
        receipt_path = root / 'goal_watch_receipt.json'
        receipt = dict(state='watching', started_utc=utc(), pid=os.getpid(), cpu_only=True,
            interval_seconds=interval, github_on_completion=publish_github, automatic_retry=False,
            training_started=False, inference_started=False, goal_complete=False)
        save(receipt_path, receipt)
        rc = 1
        try:
            while True:
                current = status(root, launch_root)
                if current['errors']:
                    raise RuntimeError('R94 stage failure; inspect goal_progress.json; no automatic retry')
                if current['all_arms_ready'] and not current['comparison_complete']:
                    compare(root, launch_root, **references)
                    current = status(root, launch_root)
                if current['comparison_complete']:
                    if publish_github:
                        publish(root, launch_root)
                    rc = 0
                    return rc
                time.sleep(interval)
        except BaseException as error:
            save(root / 'goal_watch_error.json', dict(error=f'{type(error).__name__}: {error}',
                traceback=traceback.format_exc(), updated_utc=utc(), automatic_retry=False))
            return rc
        finally:
            receipt.update(state='completed' if rc == 0 else 'stopped_on_error', exit_status=rc,
                completed_utc=utc(), pipeline_complete=rc == 0, github=optional(root / 'github_publication.json'))
            save(receipt_path, receipt)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('command', choices=('status', 'watch', 'publish'))
    result.add_argument('--root', type=Path, default=ROOT)
    result.add_argument('--launch-root', type=Path, default=LAUNCH)
    result.add_argument('--interval', type=int, default=120)
    result.add_argument('--publish-github', action='store_true')
    for flag in REFERENCE_FLAGS:
        result.add_argument('--' + flag.replace('_', '-'), type=Path)
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    if args.interval <= 0:
        raise ValueError('Watcher interval must be positive')
    if args.command == 'status':
        print(json.dumps(status(args.root, args.launch_root), indent=2))
        return 0
    if args.command == 'publish':
        publish(args.root, args.launch_root)
        return 0
    return watch(args.root, args.launch_root, args.interval, args.publish_github,
        **{key: getattr(args, key) for key in REFERENCE_FLAGS})


if __name__ == '__main__':
    sys.exit(main())

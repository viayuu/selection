"""Quiet CPU R95 completion watcher and one-shot, globally serialized publication.

No training, inference, launch, CUDA retry or model selection is performed here.
Git mutation is opt-in, after both full-budget arms and the CPU comparison pass.
"""

import argparse
from datetime import datetime, timezone
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


REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / 'code/V4/runs/R95_adaptive_example_sampling'
DEFAULT_ROOT = ROOT
LAUNCH = ROOT.parent / 'R95_launch'
ARMS = ('A', 'B')
BUDGETS = dict.fromkeys(ARMS, 57600)
PROBLEMS = ('TSP', 'CVRP', 'ATSP', 'OVRP', 'VRPB', 'VRPL', 'VRPTW', 'OVRPTW',
            'OVRPB', 'OVRPL', 'VRPBL', 'VRPBTW', 'VRPLTW', 'OVRPBL', 'OVRPBTW',
            'OVRPLTW', 'VRPBLTW', 'OVRPBLTW')
REFERENCE_FLAGS = ('nss_root', 'historical_root', 'r89_root')
REQUIRED_PDFS = ('MODEL_STATISTICS.pdf', 'selection_distributions.pdf',
                 'target_recovery_and_train.pdf', 'learning_curves.pdf', 'sampling_diagnostics.pdf')
SAMPLING = dict(warm_natural_updates=7200, refresh_every_updates=7200,
    proposal_uniform_mixture=.5, proxy='deterministic_full_loss_final_affine_gradient_norm',
    correction='1/(N*q)', normalization='arithmetic_batch_mean_no_self_normalization')


def utc():
    return datetime.now(timezone.utc).isoformat()


def optional(path):
    path = Path(path)
    return json.loads(path.read_text(encoding='utf-8')) if path.is_file() else {}


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    try:
        temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def cpu_environment():
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='')
    env.update({key: '2' for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')})
    return env


def receipt_errors(arm, path, record):
    result = []
    for current in [record]:
        phases = [str(current.get(key, '')).lower() for key in ('phase', 'state')]
        rc = current.get('exit_status', current.get('returncode', current.get('exit_code')))
        if any('failed' in phase or phase in ('error', 'stopped_on_error') for phase in phases) or rc not in (None, 0):
            result.append(dict(arm=arm, path=str(path), error=current.get('error',
                f"stage={current.get('name', '')} phases={phases} rc={rc}"), automatic_retry=False))
    for stage in record.get('stages', []):
        result.extend(receipt_errors(arm, path, stage))
    return result


def nonempty(path):
    return path.is_file() and path.stat().st_size > 0


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def file_hash(path):
    with Path(path).open('rb') as stream:
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def split_ready(result, arm, split, rows, lock, model_hash):
    if not isinstance(result, dict):
        return False
    identity, per_problem = result.get('identity', {}), result.get('per_problem', {})
    return (bool(model_hash) and set(per_problem) == set(PROBLEMS)
        and all(isinstance(row, dict) and row.get('n') == rows for row in per_problem.values())
        and identity.get('experiment') == 'R95' and identity.get('arm') == arm
        and identity.get('split') == split and identity.get('successful_updates') == lock.get('selected_update')
        and identity.get('state_hash') == lock.get('state_hash')
        and identity.get('model_state_sha256') == model_hash
        and identity.get('checkpoint_sha256') in (None, lock.get('checkpoint_sha256')))


def stages_ready(record, names):
    stages = record.get('stages', [])
    by_name = {row.get('name'): row for row in stages}
    return (len(by_name) == len(stages) and record.get('state') == 'complete'
        and record.get('exit_status') == 0 and all(
            by_name.get(name, {}).get('state') == 'complete'
            and by_name[name].get('exit_status') == 0 for name in names))


def status(root=ROOT, launch_root=LAUNCH):
    """Bounded saved JSON/PDF check; never open weights or prediction arrays."""
    root, launch_root = Path(root).resolve(), Path(launch_root).resolve()
    arms, errors, locks, sampling_sources = [], [], {}, {}
    scientific = optional(root / 'protocol_lock.json')
    scientific_sha = file_hash(root / 'protocol_lock.json') if scientific else None
    for arm, budget in BUDGETS.items():
        run, launch = root / arm, launch_root / arm
        names = ('protocol.json', 'status.json', 'training_complete.json', 'final_test_lock.json',
                 'best_eval.json', 'final_metrics.json', 'test_results.json', 'test_complete.json',
                 'lib_stage_complete.json', 'postprocess_status.json')
        records = {name: optional(run / name) for name in names}
        config, live, train, lock, best, final, test, test_done, lib, post = (records[k] for k in names)
        locks[arm] = lock
        protocol = optional(run / 'full_model_statistics/protocol.json')
        controller, worker = (optional(launch / name) for name in ('controller.json', 'worker.json'))
        ready, plan = (optional(launch / name) for name in ('ready.json', 'plan.json'))
        checkpoint, state_hash = lock.get('checkpoint_sha256'), lock.get('state_hash')
        result = test.get('result', {})
        model_hash = result.get('identity', {}).get('model_state_sha256')
        selected = lock.get('selected_update', 0)
        same_lock = protocol.get('checkpoint_lock', {})
        summaries = protocol.get('sampling_diagnostics', {})
        sources = summaries.get('source_sha256', {})
        sampling_sources[arm] = sources
        expected_sources = {str(run / 'proposals' / f'update{step:07d}.json')
            for step in range(7200, budget + 1, 7200)}
        proposal_ready = set(sources) == expected_sources and all(
            Path(name).is_file() and file_hash(name) == digest for name, digest in sources.items())
        required = dict(
            main_ready=ready.get('main_thread_ready') is True and ready.get('runner_tests_passed') is True
                and bool(ready.get('evidence')) and bool(plan) and bool(plan.get('main_ready_evidence'))
                and ready.get('plan_sha256') == (file_hash(launch / 'plan.json') if plan else None),
            scientific_lock=scientific.get('experiment') == 'R95' and scientific.get('locked') is True
                and scientific.get('arms') == list(ARMS) and scientific.get('sampling') == SAMPLING
                and scientific.get('dropout') == dict(A=.1, B=.1)
                and config.get('main_protocol_lock', {}).get('record') == scientific
                and config.get('main_protocol_lock', {}).get('sha256') == scientific_sha,
            protocol=config.get('experiment') == 'R95' and config.get('arm') == arm
                and config.get('r95_schema_version') == 1
                and config.get('initialization') == 'scratch_shared_r77'
                and config.get('model_factory') == 'code.V4.r95_experiment.load_checkpoint'
                and config.get('max_successful_updates') == budget
                and config.get('sampling') == SAMPLING and config.get('dropout') == .1
                and bool(state_hash) and json_hash(config) == state_hash,
            training_complete=train.get('successful_updates') == budget and train.get('state_hash') == state_hash
                and train.get('selected_best') == lock
                and set(train.get('updates_by_problem', {})) == set(PROBLEMS)
                and all(n == budget // len(PROBLEMS) for n in train.get('updates_by_problem', {}).values()),
            unique_VAL_lock=lock.get('experiment') == 'R95' and lock.get('arm') == arm
                and lock.get('selected_without_test') is True and lock.get('per_problem_assembly') is False
                and type(selected) is int and 0 < selected <= budget and bool(checkpoint)
                and lock.get('rule') == config.get('checkpoint_rule') and 'macro_vs_sbs' in str(lock.get('rule', ''))
                and bool(lock.get('inference_rule')) and lock.get('inference_rule') == config.get('inference_rule'),
            full_TRAIN=best.get('successful_updates') == selected
                and split_ready(best.get('train', {}), arm, 'train', 81000, lock, model_hash)
                and split_ready(final.get('train', {}), arm, 'train', 81000, lock, model_hash),
            full_VAL=final.get('selected_update') == selected
                and final.get('checkpoint_sha256') == checkpoint and final.get('state_hash') == state_hash
                and split_ready(final.get('val', {}), arm, 'val', 4500, lock, model_hash),
            full_TEST=test.get('complete') is True and test.get('checkpoint_sha256') == checkpoint
                and test.get('state_hash') == state_hash
                and result.get('identity', {}).get('checkpoint_sha256') == checkpoint
                and split_ready(result, arm, 'test', 4500, lock, model_hash)
                and final.get('complete') is True and final.get('test_read') is True and final.get('test') == result
                and test_done.get('full_tests') == 1 and test_done.get('checkpoint_sha256') == checkpoint
                and test_done.get('saved_predictions_reusable') is True,
            LIB=lib.get('complete') is True and lib.get('exit_status') == 0
                and lib.get('checkpoint_sha256') == checkpoint and lib.get('state_hash') == state_hash
                and lib.get('model_state_sha256') == model_hash
                and set(lib.get('datasets', [])) == {'TSPLIB', 'CVRPLIB'},
            PDF=post.get('complete') is True and post.get('phase') == 'complete' and post.get('exit_status') == 0
                and protocol.get('complete') is True and same_lock == lock
                and protocol.get('rdrop') is True and protocol.get('one_forward') is False,
            sampling_diagnostics=summaries.get('complete') is True and summaries.get('sampling') == SAMPLING
                and summaries.get('raw_arrays_read') is False and proposal_ready,
            controller_rc0=stages_ready(controller, ('allocation_GPU_pipeline', 'PDF')),
            worker_rc0=stages_ready(worker, ('train', 'TEST', 'LIB')))
        for name in REQUIRED_PDFS:
            required[name] = nonempty(run / 'full_model_statistics' / name)
        for name, record in records.items():
            errors.extend(receipt_errors(arm, run / name, record))
        for name in ('controller.json', 'worker.json', 'status.json', 'queue_status.json'):
            errors.extend(receipt_errors(arm, launch / name, optional(launch / name)))
        errors.extend(receipt_errors(arm, run / 'lib_stage_status.json', optional(run / 'lib_stage_status.json')))
        phase = live.get('phase', 'not_started')
        if phase == 'not_started':
            queue = optional(launch / 'queue_status.json')
            phase = queue.get('state', phase)
        arms.append(dict(arm=arm, phase=phase, successful_updates=live.get('successful_updates', train.get('successful_updates', 0)),
            budget=budget, training_unit='successful_neural_updates', checks=required,
            missing=[name for name, ok in required.items() if not ok], ready=all(required.values()),
            ready_for_comparison=all(required.values())))
    for name in ('goal_compare_receipt.json', 'github_publication.json'):
        errors.extend(receipt_errors('watcher', root / name, optional(root / name)))
    failure = optional(root / 'goal_watch_error.json')
    if failure:
        errors.append(dict(arm='watcher', path=str(root / 'goal_watch_error.json'), error=failure.get('error')))
    comparison = optional(root / 'comparison_artifacts/protocol.json')
    receipt = optional(root / 'goal_compare_receipt.json')
    ready = all(row['ready'] for row in arms)
    delivered = (comparison.get('experiment') == 'R95' and comparison.get('complete') is True
        and comparison.get('exit_status') == 0
        and (not receipt or receipt.get('state') == 'complete' and receipt.get('exit_status') == 0)
        and nonempty(root / 'comparison_artifacts/COMPARISON.pdf') and nonempty(root / 'RESULTS.pdf')
        and comparison.get('causal_claim') is False and comparison.get('sampling') == SAMPLING
        and all(comparison.get('checkpoints', {}).get(arm, {}) == locks[arm]
            and comparison.get('sampling_diagnostics', {}).get(arm, {}).get('source_sha256') == sampling_sources[arm]
            for arm in ARMS))
    complete = ready and delivered and not errors
    result = dict(experiment='R95', checked_utc=utc(), arms=arms, errors=errors,
        all_arms_ready=ready, comparison_complete=complete,
        phase='failed' if errors else 'comparison_complete' if complete else
              'ready_for_comparison' if ready else 'waiting_for_arms',
        cpu_only=True, automatic_retry=False, goal_complete=False, goal_status='active_unresolved',
        note='Saved receipts only, not live GPU evidence. No training/launch/inference/retry starts here.')
    save(root / 'goal_progress.json', result)
    return result

def _run_once(root, name, argv):
    root = Path(root)
    receipt_path = root / (name + '_receipt.json')
    if receipt_path.exists():
        raise RuntimeError('Stage was already attempted; no automatic retry: ' + name)
    receipt = dict(state='running', started_utc=utc(), argv=argv, cpu_only=True, automatic_retry=False)
    save(receipt_path, receipt)
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
        save(receipt_path, receipt)


def run_once(root, name, argv):
    with (Path(root) / (name + '.lock')).open('a') as claim:
        fcntl.flock(claim, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _run_once(root, name, argv)


def compare(root, launch_root=LAUNCH, **references):
    current = status(root, launch_root)
    if current['errors'] or not current['all_arms_ready']:
        raise RuntimeError('R95 compare requires both full-budget TEST/LIB/PDF pipelines with rc0')
    argv = [sys.executable, '-B', '-u', '-m', 'code.V4.r95_reporting', 'compare',
            '--root', str(Path(root).resolve()), '--device', 'cpu']
    for key in REFERENCE_FLAGS:
        if references.get(key) is not None:
            argv.extend(['--' + key.replace('_', '-'), str(Path(references[key]).resolve())])
    run_once(root, 'goal_compare', argv)
    if not status(root, launch_root)['comparison_complete']:
        path = Path(root) / 'goal_compare_receipt.json'
        record = optional(path)
        record.update(state='failed', error='Comparison rc0 without complete same-checkpoint PDF delivery')
        save(path, record)
        raise RuntimeError(record['error'])


def _sensitive_json(value):
    forbidden = {'api_key', 'apikey', 'password', 'access_token', 'refresh_token',
                 'client_secret', 'private_key', 'credentials', 'wandb_api_key',
                 'openai_api_key', 'raw_labels', 'labels',
                 'native_labels', 'cost_matrix', 'raw_arrays', 'sampling_weight',
                 'sampling_row', 'gradient_norms', 'draw_probabilities', 'q'}
    if isinstance(value, dict):
        return any(str(key).lower() in forbidden or _sensitive_json(item) for key, item in value.items())
    if isinstance(value, list):
        return any(_sensitive_json(item) for item in value)
    return False


def publication_paths(root, launch_root, git_root):
    root, launch_root, git_root = map(lambda p: Path(p).resolve(), (root, launch_root, git_root))
    # Publication is intentionally narrower than monitoring a caller's custom root.
    if root != ROOT.resolve() or launch_root != LAUNCH.resolve():
        raise RuntimeError('Publication accepts only the canonical R95 roots')
    files = []
    files += [root / 'PLAN.md', root / 'LAUNCH.md', root / 'protocol_lock.json',
              REPO / 'docs/research/unbiased_importance_sampling_screen.md']
    for arm in ARMS:
        plan = optional(launch_root / arm / 'plan.json')
        for name, digest in plan.get('source_sha256', {}).items():
            path = Path(name)
            if (path.parent == REPO / 'code/V4'
                    and (path.name.startswith(('r95_', 'test_r95_')) and path.suffix == '.py'
                         or path.name == 'run_v4_r95_launch.sh')):
                if path.is_symlink() or not path.is_file() or file_hash(path) != digest:
                    raise RuntimeError('Frozen R95 publication source changed: ' + str(path))
                files.append(path)
        scalar_keys = {'n', 'q_min', 'q_max', 'max_inverse_weight', 'proposal_effective_population',
                       'gradient_norm_mean', 'gradient_norm_max', 'deterministic_last_layer_second_moment_ratio'}
        for step in range(7200, 57601, 7200):
            path = root / arm / 'proposals' / f'update{step:07d}.json'
            if not path.is_file():
                continue
            record = optional(path)
            if (record.get('experiment') != 'R95' or record.get('arm') != arm
                    or record.get('complete') is not True or record.get('split') != 'train'
                    or record.get('successful_updates') != step or record.get('rows_per_problem') != 81000
                    or record.get('sampling') != SAMPLING
                    or set(record.get('per_problem', {})) != set(PROBLEMS)
                    or any(not isinstance(row, dict) or set(row) != scalar_keys
                        or any(type(value) not in (int, float) or not math.isfinite(value)
                               for value in row.values()) for row in record.get('per_problem', {}).values())):
                raise RuntimeError('Only own-TRAIN scalar proposal receipts may be published: ' + str(path))
            files.append(path)
    blocked = {'wandb', 'evaluation', 'lib_evaluation', 'prepared', 'by_problem',
               'figures', 'models', 'tree_models', 'feature_cache', 'input_cache', 'cache',
               '__pycache__', 'labels', 'credentials', 'proposals', 'sources',
               'startup_attempt1', 'raw_arrays'}
    for base in (root, launch_root):
        for path in base.rglob('*'):
            if path.is_symlink() or not path.is_file() or path.suffix.lower() not in ('.json', '.csv', '.md', '.pdf', '.png'):
                continue
            parts = path.relative_to(base).parts
            if any(part.lower() in blocked or part.startswith('.') for part in parts):
                continue
            if path.name.startswith(('goal_', 'github_')):
                continue
            if any(word in path.name.lower() for word in ('checkpoint', 'credential', 'secret', 'password', 'token', 'label', 'cache')):
                continue
            files.append(path)
    result = set()
    for path in files:
        if not path.is_file() or path.is_symlink() or path.stat().st_size > 32 * 1024 * 1024:
            continue
        resolved = path.resolve()
        relative = resolved.relative_to(git_root)
        if path.suffix == '.json' and _sensitive_json(optional(path)):
            raise RuntimeError('Sensitive JSON excluded from publication: ' + str(path))
        result.add(str(relative))
    if not result:
        raise RuntimeError('No curated R95 assets to publish')
    return sorted(result)


def git_output(argv, cwd):
    return subprocess.check_output(['git', *argv], cwd=str(cwd), text=True).strip()


def publish(root=ROOT, launch_root=LAUNCH):
    """One opt-in commit/push, protected by the shared R92/R94 Git-directory lock."""
    root, launch_root = Path(root).resolve(), Path(launch_root).resolve()
    current = status(root, launch_root)
    if current['errors'] or not current['comparison_complete']:
        raise RuntimeError('No Git access before main-ready whole-R95 completion')
    git_root = Path(git_output(['rev-parse', '--show-toplevel'], REPO)).resolve()
    git_dir = Path(git_output(['rev-parse', '--absolute-git-dir'], REPO)).resolve()
    # All index/HEAD checks and mutations must be inside this cross-version lock.
    with (git_dir / 'codex-publication.lock').open('a') as mutex:
        fcntl.flock(mutex, fcntl.LOCK_EX)
        current = status(root, launch_root)
        if current['errors'] or not current['comparison_complete']:
            raise RuntimeError('Publish requires both successful full arms and the completed comparison')
        receipt_path = root / 'github_publication.json'
        prior = optional(receipt_path)
        if prior.get('complete') is True:
            return prior
        if receipt_path.exists():
            raise RuntimeError('Publication was already attempted; no automatic staging/commit/push retry')
        if git_output(['branch', '--show-current'], git_root) != 'main':
            raise RuntimeError('Expected main; no branch was changed')
        if git_output(['diff', '--cached', '--name-only', '-z'], git_root):
            raise RuntimeError('Pre-existing staged changes; nothing staged or reverted by R95')
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
                raise RuntimeError('Index or branch changed during publication; retain it untouched for inspection')
            receipt.update(state='committing', files=sorted(staged))
            save(receipt_path, receipt)
            subprocess.run(['git', 'commit', '-m', 'r95: publish completed adaptive sampling controls'], cwd=str(git_root), check=True)
            commit = git_output(['rev-parse', 'HEAD'], git_root)
            receipt.update(state='committed_pending_push', commit=commit)
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
            training_started=False, goal_complete=False)
        save(receipt_path, receipt)
        rc = 1
        try:
            while True:
                current = status(root, launch_root)
                if current['errors']:
                    raise RuntimeError('R95 stage failure; inspect goal_progress.json; no automatic retry')
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
            # Every terminal return updates this receipt, including receipt-detected failures.
            receipt.update(state='completed' if rc == 0 else 'stopped_on_error',
                exit_status=rc, completed_utc=utc(), pipeline_complete=rc == 0,
                github=optional(root / 'github_publication.json'))
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
    if args.interval < 1:
        raise ValueError('Watcher interval must be positive')
    if args.command == 'status':
        print(json.dumps(status(args.root, args.launch_root), indent=2))
        return 0
    if args.command == 'publish':
        publish(args.root, args.launch_root)
        return 0
    return watch(args.root, args.launch_root, args.interval, args.publish_github,
        **{name: getattr(args, name) for name in REFERENCE_FLAGS})


if __name__ == '__main__':
    sys.exit(main())

"""Quiet CPU completion, comparison and one-shot curated Git publication for R92."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

os.environ['CUDA_VISIBLE_DEVICES'] = ''
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '2'

from .r84_goal import optional, save, utc


REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / 'code/V4/runs/R92_broad_learning'
LAUNCH = ROOT.parent / 'R92_launch'
BUDGETS = dict(A=57600, B=57600, C=3000, D=3000)


def status(root=ROOT, launch_root=LAUNCH):
    root, launch_root = Path(root), Path(launch_root)
    arms, errors = [], []
    for arm, budget in BUDGETS.items():
        run, launch = root / arm, launch_root / arm
        live, train, lock, test, lib, report = [optional(run / name) for name in
            ('status.json', 'training_complete.json', 'final_test_lock.json',
             'test_results.json', 'lib_stage_complete.json', 'postprocess_status.json')]
        required = dict(training_complete=train.get('successful_updates') == budget,
            unique_VAL_lock=bool(lock.get('selected_without_test')) and lock.get('per_problem_assembly') is False,
            full_TEST=test.get('complete') is True, LIB=lib.get('complete') is True,
            PDF=report.get('complete') is True and report.get('phase') == 'complete')
        for name in ('MODEL_STATISTICS.pdf', 'selection_distributions.pdf'):
            path = run / 'full_model_statistics' / name
            required[name] = path.is_file() and path.stat().st_size > 0
        for path in (run / 'status.json', run / 'postprocess_status.json',
                     launch / 'controller.json', launch / 'worker.json'):
            receipt = optional(path)
            phase = str(receipt.get('phase', receipt.get('state', ''))).lower()
            if 'failed' in phase or phase == 'error':
                errors.append(dict(arm=arm, path=str(path), error=receipt.get('error', phase)))
        arms.append(dict(arm=arm, phase=live.get('phase', 'not_started'),
            successful_updates=live.get('successful_updates', 0), budget=budget,
            training_unit='successful_neural_updates' if arm in 'AB' else 'trees_per_task',
            missing=[key for key, value in required.items() if not value], ready=all(required.values())))
    comparison = optional(root / 'comparison_artifacts/protocol.json')
    complete = comparison.get('complete') is True and (root / 'comparison_artifacts/COMPARISON.pdf').is_file()
    result = dict(experiment='R92', checked_utc=utc(), arms=arms, errors=errors,
        all_arms_ready=all(row['ready'] for row in arms), comparison_complete=complete,
        phase='failed' if errors else 'comparison_complete' if complete else 'waiting_for_arms',
        cpu_only=True, automatic_retry=False, goal_complete=False, goal_status='active_unresolved',
        note='Saved receipts, not live GPU evidence. No new training or retry is started here.')
    save(root / 'goal_progress.json', result)
    return result


def run_once(root, name, argv):
    receipt_path = root / (name + '_receipt.json')
    if receipt_path.exists():
        value = optional(receipt_path)
        if value.get('exit_status') == 0:
            return
        raise RuntimeError('Previous stage attempt exists; no automatic retry: ' + name)
    receipt = dict(state='running', started_utc=utc(), argv=argv, automatic_retry=False)
    save(receipt_path, receipt)
    with (root / (name + '.log')).open('a') as log:
        process = subprocess.run(argv, cwd=str(REPO), env=dict(os.environ),
            stdout=log, stderr=subprocess.STDOUT, check=False)
    receipt.update(state='complete' if process.returncode == 0 else 'failed',
                   exit_status=process.returncode, completed_utc=utc())
    save(receipt_path, receipt)
    if process.returncode:
        raise RuntimeError('Stage failed; inspect ' + name + '.log')


def publish(root):
    """Only allowlisted R92 assets; never absorb a pre-existing staged change."""
    git_root = Path(subprocess.check_output(['git', 'rev-parse', '--show-toplevel'], cwd=REPO, text=True).strip())
    branch = subprocess.check_output(['git', 'branch', '--show-current'], cwd=git_root, text=True).strip()
    if branch != 'main':
        raise RuntimeError('Expected main branch for the explicitly authorized publication; no branch changed')
    staged = subprocess.check_output(['git', 'diff', '--cached', '--name-only'], cwd=git_root, text=True)
    if staged.strip():
        raise RuntimeError('Pre-existing staged changes require human/main inspection; nothing was staged by this publisher')
    files = list((REPO / 'code/V4').glob('r92*.py')) + list((REPO / 'code/V4').glob('test_r92*.py'))
    files += list((REPO / 'code/V4').glob('run_v4_r92*.sh'))
    files += [REPO / 'docs/research/r92_broad_direction_research.md']
    for base in (root, LAUNCH):
        for path in base.rglob('*'):
            if not path.is_file() or path.suffix not in ('.json', '.csv', '.md', '.pdf', '.png'):
                continue
            relative = path.relative_to(base)
            if any(part in ('wandb', 'feature_cache', 'input_cache', 'models', 'by_problem', 'figures')
                   for part in relative.parts):
                continue
            if 'evaluation' in relative.parts or path.name.endswith('_publish_receipt.json'):
                continue
            if path.stat().st_size > 32 * 1024 * 1024:
                continue
            files.append(path)
    paths = sorted({str(path.resolve().relative_to(git_root)) for path in files if path.is_file()})
    if not paths:
        raise RuntimeError('No curated R92 assets to publish')
    publication = dict(complete=False, phase='staging', started_utc=utc(),
        remote='origin/main', curated_only=True, files=len(paths), automatic_retry=False)
    save(root / 'github_publication.json', publication)
    for start in range(0, len(paths), 80):
        subprocess.run(['git', 'add', '-f', '--', *paths[start:start + 80]], cwd=git_root, check=True)
    # The index is intentionally narrow, and unrelated dirty files are untouched.
    subprocess.run(['git', 'commit', '-m', 'r92: publish completed full18 hierarchy and tree experiments'], cwd=git_root, check=True)
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=git_root, text=True).strip()
    publication.update(phase='committed_pending_push', commit=commit)
    save(root / 'github_publication.json', publication)
    subprocess.run(['git', 'push', 'origin', 'main'], cwd=git_root, check=True)
    save(root / 'github_publication.json', dict(complete=True, commit=commit,
        completed_utc=utc(), files=len(paths), remote='origin/main', curated_only=True))


def watch(root=ROOT, launch_root=LAUNCH, interval=120, publish_github=False):
    root, launch_root = Path(root).resolve(), Path(launch_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    with (root / 'goal_watch.lock').open('a') as claim:
        fcntl.flock(claim, fcntl.LOCK_EX | fcntl.LOCK_NB)
        save(root / 'goal_watch_receipt.json', dict(state='watching', pid=os.getpid(),
            started_utc=utc(), cpu_only=True, interval_seconds=interval,
            github_on_completion=publish_github, automatic_retry=False))
        while True:
            try:
                current = status(root, launch_root)
                if current['errors']:
                    return 1
                if current['all_arms_ready'] and not current['comparison_complete']:
                    run_once(root, 'goal_compare', [sys.executable, '-B', '-u', '-m',
                        'code.V4.r92_reporting', 'compare', '--root', str(root), '--device', 'cpu'])
                    current = status(root, launch_root)
                if current['comparison_complete']:
                    if publish_github and not optional(root / 'github_publication.json').get('complete'):
                        publish(root)
                    save(root / 'goal_watch_receipt.json', dict(state='completed_goal_active',
                        completed_utc=utc(), goal_complete=False, cpu_only=True,
                        github=optional(root / 'github_publication.json')))
                    return 0
            except Exception as error:
                save(root / 'goal_watch_error.json', dict(error=str(error),
                    traceback=traceback.format_exc(), updated_utc=utc(), automatic_retry=False))
                return 1
            time.sleep(interval)


def main(argv=None):
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('command', choices=('status', 'watch', 'publish'))
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--launch-root', type=Path, default=LAUNCH)
    parser.add_argument('--interval', type=int, default=120)
    parser.add_argument('--publish-github', action='store_true')
    args = parser.parse_args(argv)
    if args.command == 'status':
        print(json.dumps(status(args.root, args.launch_root), indent=2))
        return 0
    if args.command == 'publish':
        current = status(args.root, args.launch_root)
        if current['errors'] or not current['comparison_complete']:
            raise RuntimeError('Publish requires successful completion and all reports')
        publish(args.root)
        return 0
    return watch(args.root, args.launch_root, args.interval, args.publish_github)


if __name__ == '__main__':
    sys.exit(main())

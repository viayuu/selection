"""Quiet CPU receipt watcher for R88; never starts GPU work or retries stages."""

import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = ''
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '2'

from . import r84_goal as receipts


ROOT = Path(__file__).resolve().parent / 'runs/R88_execution_role_inputs'
LAUNCH = ROOT.parent / 'R88_launch'
REPO = Path(__file__).resolve().parents[2]
ARMS = ('A_canonical_anchor', 'B_execution_anchor')


def status(root=ROOT, launch_root=LAUNCH):
    root, launch_root = Path(root).resolve(), Path(launch_root).resolve()
    arms = [receipts.arm_snapshot(root, launch_root, arm) for arm in ARMS]
    errors = [dict(arm=row['arm'], **error) for row in arms for error in row['errors']]
    for row in arms:
        path = launch_root / row['arm'] / 'status.json'
        queue = receipts.optional(path)
        state = str(queue.get('state', queue.get('phase', '')))
        row['queue_state'] = state or None
        if row['phase'] == 'not_started' and state:
            row['phase'] = state
        if 'failed' in state.lower() or state.lower() == 'error':
            errors.append(dict(arm=row['arm'], path=str(path), error=queue.get('error', state)))
    failure = receipts.optional(root / 'goal_watch_error.json')
    if failure:
        errors.append(dict(arm='watcher', path=str(root / 'goal_watch_error.json'), error=failure['error']))
    protocol = root / 'comparison_artifacts/protocol.json'
    pdf = protocol.with_name('COMPARISON.pdf')
    comparison = receipts.optional(protocol)
    delivered = comparison.get('complete') is True and pdf.is_file() and pdf.stat().st_size > 0
    if comparison.get('complete') is True and not delivered:
        errors.append(dict(arm='comparison', path=str(pdf), error='Complete comparison receipt has no nonempty PDF'))
    ready = all(row['ready_for_comparison'] for row in arms)
    complete = ready and delivered
    result = dict(experiment='R88', checked_utc=receipts.utc(), poll_seconds=60,
        phase='failed' if errors else 'comparison_complete' if complete else
              'ready_for_comparison' if ready else 'waiting_for_arms',
        arms=arms, errors=errors, error_flags=bool(errors), all_arms_ready=ready,
        comparison_complete=complete, comparison_pdf=str(pdf) if complete else None,
        goal_status='active_unresolved', goal_complete=False, cpu_only=True,
        training_started_by_monitor=False, automatic_retry=False,
        next_action='manual_failure_review' if errors else 'research_handoff_needed' if complete else
                    'compare_saved_predictions' if ready else 'wait_for_committed_arms',
        liveness_note='Saved stage receipts only; queueing is not failure. No GPU/process liveness claims.')
    receipts.save(root / 'goal_progress.json', result)
    return result


def compare(root, launch_root):
    root = Path(root).resolve()
    attempt_path = root / 'goal_compare_receipt.json'
    if receipts.optional(attempt_path):
        raise RuntimeError('Comparison already attempted; inspect retained receipts and resolve manually, never auto retry')
    command = [sys.executable, '-u', '-m', 'code.V4.r88_reporting', 'compare',
               '--root', str(root), '--device', 'cpu']
    attempt = dict(state='running', started_utc=receipts.utc(), watcher_pid=os.getpid(),
                   argv=command, cpu_only=True, automatic_retry=False)
    receipts.save(attempt_path, attempt)
    with (root / 'goal_compare.log').open('a', encoding='utf-8') as log:
        process = subprocess.run(command, cwd=str(REPO), env=dict(os.environ),
                                 stdout=log, stderr=subprocess.STDOUT, check=False)
    attempt.update(exit_status=process.returncode, completed_utc=receipts.utc(),
                   state='command_succeeded' if process.returncode == 0 else 'failed')
    receipts.save(attempt_path, attempt)
    if process.returncode != 0:
        raise RuntimeError(f'CPU comparison exited {process.returncode}; retain goal_compare.log, no automatic retry')
    if not status(root, launch_root)['comparison_complete']:
        raise RuntimeError('CPU comparison returned zero without complete protocol.json and nonempty COMPARISON.pdf')
    attempt['state'] = 'complete'
    receipts.save(attempt_path, attempt)


def watch(root=ROOT, launch_root=LAUNCH, interval=60, max_polls=None):
    root, launch_root = Path(root).resolve(), Path(launch_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    with (root / 'goal_watch.lock').open('a') as claim:
        fcntl.flock(claim, fcntl.LOCK_EX | fcntl.LOCK_NB)
        receipt = dict(state='watching', started_utc=receipts.utc(), pid=os.getpid(),
                       interval_seconds=interval, cpu_only=True, automatic_retry=False,
                       training_started=False, goal_status='active_unresolved', goal_complete=False)
        receipts.save(root / 'goal_watch_receipt.json', receipt)
        snapshot, polls = None, 0
        while True:
            try:
                snapshot = status(root, launch_root)
                if snapshot['errors']:
                    receipt.update(state='stopped_on_error', completed_utc=receipts.utc())
                    receipts.save(root / 'goal_watch_receipt.json', receipt)
                    return 1
                if snapshot['comparison_complete']:
                    receipt.update(state='comparison_complete_goal_active', completed_utc=receipts.utc())
                    receipts.save(root / 'goal_watch_receipt.json', receipt)
                    return 0
                if snapshot['all_arms_ready']:
                    compare(root, launch_root)
                    continue
            except Exception as error:
                failure = dict(error=f'{type(error).__name__}: {error}', updated_utc=receipts.utc(),
                    traceback=traceback.format_exc(), automatic_retry=False,
                    action='Inspect retained stage receipts/logs; manual repair only. Do not restart training.')
                receipts.save(root / 'goal_watch_error.json', failure)
                failed = dict(snapshot or {}, experiment='R88', phase='failed', error_flags=True,
                    errors=[dict(arm='watcher', **failure)], goal_status='active_unresolved', goal_complete=False,
                    cpu_only=True, automatic_retry=False, next_action='manual_failure_review')
                receipts.save(root / 'goal_progress.json', failed)
                receipt.update(state='stopped_on_error', completed_utc=receipts.utc())
                receipts.save(root / 'goal_watch_receipt.json', receipt)
                return 1
            polls += 1
            if max_polls is not None and polls >= max_polls:
                receipt.update(state='bounded_poll_complete_goal_active', completed_utc=receipts.utc())
                receipts.save(root / 'goal_watch_receipt.json', receipt)
                return 0
            time.sleep(interval)


def main(argv=None):
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('command', choices=('status', 'watch'))
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--launch-root', type=Path, default=LAUNCH)
    parser.add_argument('--interval', type=int, default=60)
    parser.add_argument('--max-polls', type=int)
    args = parser.parse_args(argv)
    if args.interval < 1 or (args.max_polls is not None and args.max_polls < 1):
        parser.error('Poll interval and bounded poll counts must be positive')
    if args.command == 'status':
        print(json.dumps(status(args.root, args.launch_root), indent=2))
        return 0
    return watch(args.root, args.launch_root, args.interval, args.max_polls)


if __name__ == '__main__':
    sys.exit(main())

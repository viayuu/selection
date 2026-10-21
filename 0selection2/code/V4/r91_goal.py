"""Quiet CPU-only R91 completion handoff; never launches or retries GPU jobs."""

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

from . import r84_goal as receipts


ROOT = Path(__file__).resolve().parent / 'runs/R91_probability_link_control'
LAUNCH = ROOT.parent / 'R91_launch'
REPO = Path(__file__).resolve().parents[2]
ARMS = ('A_utility_ce', 'B_utility_probabilities')


def status(root=ROOT, launch_root=LAUNCH):
    root, launch_root = Path(root).resolve(), Path(launch_root).resolve()
    arms = [receipts.arm_snapshot(root, launch_root, arm) for arm in ARMS]
    errors = [dict(arm=row['arm'], **error) for row in arms for error in row['errors']]
    for row in arms:
        queue_path = launch_root / row['arm'] / 'status.json'
        queue = receipts.optional(queue_path)
        queue_state = str(queue.get('state', queue.get('phase', '')))
        if row['phase'] == 'not_started' and queue_state:
            row['phase'] = queue_state
        if 'failed' in queue_state.lower() or queue_state.lower() == 'error':
            errors.append(dict(arm=row['arm'], path=str(queue_path), error=queue.get('error', queue_state)))
    failure = receipts.optional(root / 'goal_watch_error.json')
    if failure:
        errors.append(dict(arm='watcher', error=failure['error']))
    comparison_path = root / 'comparison_artifacts/protocol.json'
    pdf = comparison_path.with_name('COMPARISON.pdf')
    comparison = receipts.optional(comparison_path)
    delivered = comparison.get('complete') is True and pdf.is_file() and pdf.stat().st_size > 0
    ready = all(row['ready_for_comparison'] for row in arms)
    complete = ready and delivered
    if comparison.get('complete') is True and not delivered:
        errors.append(dict(arm='comparison', path=str(pdf), error='Missing completed comparison PDF'))
    result = dict(experiment='R91', checked_utc=receipts.utc(), arms=arms, errors=errors,
        phase='failed' if errors else 'comparison_complete' if complete else
              'ready_for_comparison' if ready else 'waiting_for_arms',
        error_flags=bool(errors), all_arms_ready=ready, comparison_complete=complete,
        comparison_pdf=str(pdf) if complete else None, goal_status='active_unresolved',
        goal_complete=False, cpu_only=True, training_started_by_monitor=False,
        automatic_retry=False,
        liveness_note='Saved receipts only, not live GPU/process evidence. No automatic restart.')
    receipts.save(root / 'goal_progress.json', result)
    return result


def compare(root, launch_root):
    root = Path(root).resolve()
    receipt_path = root / 'goal_compare_receipt.json'
    if receipts.optional(receipt_path):
        raise RuntimeError('Comparison was already attempted; retain evidence, never auto retry')
    argv = [sys.executable, '-B', '-u', '-m', 'code.V4.r91_reporting', 'compare',
            '--root', str(root), '--device', 'cpu']
    receipt = dict(state='running', started_utc=receipts.utc(), watcher_pid=os.getpid(),
                   argv=argv, cpu_only=True, automatic_retry=False)
    receipts.save(receipt_path, receipt)
    with (root / 'goal_compare.log').open('a', encoding='utf-8') as log:
        process = subprocess.run(argv, cwd=str(REPO), env=dict(os.environ),
                                 stdout=log, stderr=subprocess.STDOUT, check=False)
    receipt.update(exit_status=process.returncode, completed_utc=receipts.utc(),
                   state='command_succeeded' if process.returncode == 0 else 'failed')
    receipts.save(receipt_path, receipt)
    if process.returncode != 0:
        raise RuntimeError('CPU comparison failed; inspect goal_compare.log, no automatic retry')
    if not status(root, launch_root)['comparison_complete']:
        raise RuntimeError('Comparison returned rc0 without complete delivery evidence')
    receipt['state'] = 'complete'
    receipts.save(receipt_path, receipt)


def watch(root=ROOT, launch_root=LAUNCH, interval=60):
    root, launch_root = Path(root).resolve(), Path(launch_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    with (root / 'goal_watch.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        receipt_path = root / 'goal_watch_receipt.json'
        receipt = dict(state='watching', started_utc=receipts.utc(), pid=os.getpid(),
            interval_seconds=interval, cpu_only=True, automatic_retry=False,
            training_started=False, goal_complete=False)
        receipts.save(receipt_path, receipt)
        while True:
            try:
                snapshot = status(root, launch_root)
                if snapshot['errors']:
                    receipt.update(state='stopped_on_error', completed_utc=receipts.utc())
                    receipts.save(receipt_path, receipt)
                    return 1
                if snapshot['comparison_complete']:
                    receipt.update(state='comparison_complete_goal_active', completed_utc=receipts.utc())
                    receipts.save(receipt_path, receipt)
                    return 0
                if snapshot['all_arms_ready']:
                    compare(root, launch_root)
                    continue
            except Exception as error:
                receipts.save(root / 'goal_watch_error.json', dict(
                    error=f'{type(error).__name__}: {error}', traceback=traceback.format_exc(),
                    updated_utc=receipts.utc(), automatic_retry=False))
                status(root, launch_root)
                receipt.update(state='stopped_on_error', completed_utc=receipts.utc())
                receipts.save(receipt_path, receipt)
                return 1
            time.sleep(interval)


def main(argv=None):
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('command', choices=('status', 'watch'))
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--launch-root', type=Path, default=LAUNCH)
    parser.add_argument('--interval', type=int, default=60)
    args = parser.parse_args(argv)
    if args.interval < 1:
        parser.error('Interval must be positive')
    if args.command == 'status':
        print(json.dumps(status(args.root, args.launch_root), indent=2))
        return 0
    return watch(args.root, args.launch_root, args.interval)


if __name__ == '__main__':
    sys.exit(main())

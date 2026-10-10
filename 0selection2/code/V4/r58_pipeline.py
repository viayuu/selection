"""Single-GPU R58 controller; no interaction with existing experiment queues."""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from ..unified_selector.registry import POOLS, PROBLEMS
from .r58_analysis import progress, summarize
from .r58_labels import implementation_hashes, lock, publish, runtime_environment
from .r58_scenario import ROOT, save_json, write_contract


def run(args):
    args.root.mkdir(parents=True, exist_ok=True)
    write_contract(args.root)
    state = dict(stage=args.stage, pid=os.getpid(), cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                 slurm_job_id=os.environ.get('SLURM_JOB_ID'), started=time.time(), status='running')
    def update(**values):
        state.update(values)
        save_json(args.root / 'pipeline_state.json', state)
    update()
    def command(module, arguments, log):
        directory = args.root / 'logs'
        directory.mkdir(exist_ok=True)
        cmd = [sys.executable, '-u', '-m', module, '--root', str(args.root), *arguments]
        update(current_command=cmd, current_log=str(directory / log))
        print('[R58 command] ' + ' '.join(cmd), flush=True)
        with (directory / log).open('a') as stream:
            process = subprocess.run(cmd, stdout=stream, stderr=subprocess.STDOUT, cwd=str(ROOT.parents[3]))
        return process.returncode
    try:
        if args.stage in ('all', 'preflight'):
            # Cover every candidate before expanding repeats or generating labels.
            preflight_hashes = implementation_hashes()
            preflight_environment = runtime_environment()
            failed = []
            for problem in PROBLEMS:
                for method in POOLS[problem]:
                    existing = args.root / 'preflight' / f'{problem}__{method}.json'
                    record = json.loads(existing.read_text()) if existing.exists() else {}
                    if (not args.force_preflight and record.get('passed') and
                            record.get('implementation_sha256') == preflight_hashes and
                            record.get('runtime_environment') == preflight_environment):
                        continue
                    # Invalidate before launch: a killed worker or controller must
                    # not leave an older passing record eligible for release.
                    record.update(passed=False, error='Preflight rerun pending; cached qualification invalidated')
                    save_json(existing, record)
                    code = command('code.V4.r58_labels', ['--stage', 'probe', '--problem', problem, '--solver', method],
                                   f'preflight__{problem}__{method}.log')
                    print(f'[R58 preflight] {problem}/{method} returncode={code}', flush=True)
                    if code:
                        failed.append(f'{problem}/{method}: exit {code}')
                        record = json.loads(existing.read_text())
                        record.update(passed=False, subprocess_returncode=code,
                                      error=record.get('error') or f'Preflight subprocess exited {code}')
                        save_json(existing, record)
                    progress(args.root)
                    if implementation_hashes() != preflight_hashes:
                        raise RuntimeError('Deployment source changed during preflight; restart with --force-preflight')
            status = progress(args.root)
            if failed or not status['complete_preflight']:
                update(status='blocked_preflight', failed_subprocesses=failed, finished=time.time())
                return 2
        if args.stage == 'preflight':
            update(status='preflight_complete', finished=time.time())
            return 0
        if args.stage in ('all', 'labels'):
            if not (args.root / 'deployments.lock.json').exists():
                lock(args.root)
            backhaul = [p for p in PROBLEMS if 'B' in p]
            priority = backhaul + [p for p in PROBLEMS if p not in backhaul and p != 'TSP'] + ['TSP']
            completed_columns = []
            for problem in priority:
                for split in ('train', 'val', 'test'):
                    for method in POOLS[problem]:
                        code = command('code.V4.r58_labels', ['--stage', 'generate', '--problem', problem,
                            '--solver', method, '--split', split], f'labels__{problem}__{split}__{method}.log')
                        if code:
                            raise RuntimeError(f'Column failed: {problem}/{split}/{method}; see recorded log/failure')
                        completed_columns.append(f'{problem}/{split}/{method}')
                        update(completed_columns=completed_columns, columns_completed=len(completed_columns))
            publish(args.root)
        if args.stage in ('all', 'train'):
            code = command('code.V4.r58_experiment', ['--stage', 'all'], 'training.log')
            if code:
                raise RuntimeError('Full baseline training/test failed; see logs/training.log')
        summarize(args.root)
        update(status='complete' if (args.root / 'test_results.json').exists() else 'labels_complete', finished=time.time())
        return 0
    except Exception as error:
        update(status='failed', error=repr(error), finished=time.time())
        progress(args.root)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['all', 'preflight', 'labels', 'train', 'analysis'], default='all')
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--force-preflight', action='store_true', help='Recheck all candidates with frozen current sources')
    args = parser.parse_args()
    if args.stage == 'analysis':
        summarize(args.root)
    else:
        raise SystemExit(run(args))


if __name__ == '__main__':
    main()

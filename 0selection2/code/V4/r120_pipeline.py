"""Finish the versioned labels, then train/evaluate/report one R120 model.

This CPU controller only schedules the already authorized work. Annotation
workers and model checkpoints survive a controller failure. It never restarts
an unsuccessful training stage or edits the data protocol after TEST.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .r120_launch import ROOT, REPO, NODES, GPU_MUTEX, write


SUPPLEMENT = Path('/public/home/shiys/nonrouting_supplement_20261029')
NSS_REFERENCE = REPO / 'code/V4/runs/R84_new90k_attention/nss_reference'
SHARED_CACHE = REPO / 'code/V4/runs/R84_new90k_attention/input_cache'
CODE = (
    'nonrouting_data.py', 'test_nonrouting_data.py',
    'nonrouting_supplement.py', 'test_nonrouting_supplement.py',
    'r120_model.py', 'r120_objective.py', 'r120_experiment.py', 'r120_reporting.py',
    'r120_launch.py', 'r120_capacity.py', 'r120_pipeline.py', 'run_v4_r120.sh',
    'test_r120_model.py', 'test_r120_objective.py',
    'test_r120_experiment.py', 'test_r120_reporting.py',
)


def read(path):
    return json.loads(Path(path).read_text())


def now():
    return datetime.now(timezone.utc).isoformat()


def state(root, phase, **fields):
    value = dict(updated_utc=now(), phase=phase, pid=os.getpid(), **fields)
    write(root / 'pipeline_status.json', value)
    print(json.dumps(value), flush=True)


def pause_between_stages(root):
    while (root / 'PAUSE_NEW_WORK').exists():
        state(root, 'paused_between_stages', running_experiments_interrupted=False)
        time.sleep(30)


def run_stage(root, name, argv):
    pause_between_stages(root)
    record = root / 'pipeline_stages' / (name + '.json')
    if record.exists():
        previous = read(record)
        if previous.get('exit_status') == 0 and previous['command'] == argv:
            return previous
        raise RuntimeError(f'{name}: previous stage did not complete; preserve it for inspection')
    log = root / (name + '.log')
    value = dict(stage=name, command=argv, started_utc=now(), log=str(log), state='running')
    write(record, value)
    state(root, name, log=str(log))
    with log.open('a') as stream:
        process = subprocess.Popen(argv, cwd=REPO, stdout=stream, stderr=subprocess.STDOUT)
        value['pid'] = process.pid
        write(record, value)
        code = process.wait()
    value.update(exit_status=code, ended_utc=now(), state='complete' if code == 0 else 'failed')
    write(record, value)
    if code:
        raise RuntimeError(f'{name}: exit={code}; see {log}')
    return value


def annotations(root):
    plan = read(root / 'resources/as_full_pool_plan.json')
    while True:
        completed, failed = [], []
        for slot, jobs in plan['slots'].items():
            for worker in range(plan['parallel_per_slot']):
                path = root / 'resources' / f'as_full_pool_{slot}_w{worker}.json'
                receipt = read(path) if path.exists() else {}
                if receipt.get('state') == 'complete':
                    completed.append(slot + '_w' + str(worker))
                elif receipt.get('state') == 'failed':
                    failed.append(str(path))
        if failed:
            raise RuntimeError('An annotation shard failed: ' + ', '.join(failed))
        # Count saved immutable rows, not the number of started worker processes.
        count = sum(1 for split in ('train', 'val', 'test', 'benchmark')
                    for _ in (SUPPLEMENT / 'native_as_labels' / split).glob('*.json'))
        state(root, 'completing_mkp_labels', completed_shards=len(completed), total_shards=16,
              saved_native_labels=count, target_native_labels=12970,
              expected_next_stage='export complete declared pool, then full24 training')
        if len(completed) == 16:
            return
        time.sleep(60)


def wait_for_annotation_exit(root):
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        acquired = []
        try:
            for item in NODES.values():
                for uuid in item['uuids']:
                    handle = (GPU_MUTEX / f'queue_gpu_{uuid}.json').open('a+')
                    acquired.append(handle)
                    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return
        except BlockingIOError:
            time.sleep(2)
        finally:
            for handle in acquired:
                handle.close()
    raise RuntimeError('Annotation parent still holds an authorized GPU; do not start duplicate work')


def curated_paths(root, git_root):
    files = [REPO / 'code/V4' / name for name in CODE]
    research = REPO / 'docs/research/multidomain_selection_design_20261029'
    files.extend(research / name for name in (
        'REPORT.md', 'HANDOFF.md', 'sources.csv', 'directions.csv', 'ROUTING.json',
        'REPORT.pdf', 'HANDOFF.pdf', 'render_pdf.py', 'pdf_build.json'))
    for directory in (root / 'implement-stage', root / 'reports'):
        files.extend(p for p in directory.rglob('*') if p.is_file() and p.suffix in
                     ('.md', '.json', '.csv', '.pdf', '.png'))
    for directory in (root, root / 'resources', root / 'pipeline_stages'):
        files.extend(p for p in directory.iterdir() if p.is_file() and p.suffix in ('.md', '.json', '.csv')
                     and p.name != 'github_publication.json')
    files.extend((root / 'resources/train_startup_attempt1_bridge').glob('*.json'))
    for name in ('as_dispatch.py', 'GPU_COMMANDS.md', 'protocol.json', 'manifest.json',
                 'native_as_dispatch_protocol.json', 'audit.json'):
        files.append(SUPPLEMENT / name)
    files.extend(p for p in (SUPPLEMENT / 'sources').rglob('*') if p.is_file())
    for name in ('prepared_cpu', 'prepared_with_as'):
        files.append(SUPPLEMENT / name / 'manifest.json')
    paths = sorted({str(p.resolve().relative_to(git_root)) for p in files
                    if p.is_file() and p.stat().st_size < 8 * 1024**2})
    if any(Path(p).suffix in ('.pt', '.npy', '.npz', '.pkl', '.jsonl') for p in paths):
        raise ValueError('Weights and raw data must stay outside the small publication')
    return paths


def publish(root):
    def git(*argv):
        return subprocess.check_output(['git', *argv], cwd=REPO, text=True).strip()
    git_root, git_dir = Path(git('rev-parse', '--show-toplevel')), Path(git('rev-parse', '--absolute-git-dir'))
    with (git_dir / 'codex-publication.lock').open('a+') as claim:
        fcntl.flock(claim, fcntl.LOCK_EX)
        if git('branch', '--show-current') != 'main':
            raise RuntimeError('The publication branch changed; preserve it for inspection')
        if git('diff', '--cached', '--name-only'):
            raise RuntimeError('Existing staged changes retained; no R120 files staged')
        paths = curated_paths(root, git_root)
        before = git('rev-parse', 'HEAD')
        for start in range(0, len(paths), 60):
            subprocess.run(['git', 'add', '-f', '--', *paths[start:start + 60]], cwd=git_root, check=True)
        staged = set(git('diff', '--cached', '--name-only').splitlines())
        if not staged or not staged <= set(paths) or git('rev-parse', 'HEAD') != before:
            raise RuntimeError('Concurrent Git state changed; preserve staged files for inspection')
        subprocess.run(['git', 'commit', '-m', 'r120: add six typed optimization domains and a complete-pool selector'],
                       cwd=git_root, check=True)
        commit = git('rev-parse', 'HEAD')
        subprocess.run(['git', 'push', 'origin', 'HEAD'], cwd=git_root, check=True)
        write(root / 'github_publication.json', dict(complete=True, commit=commit,
            files=sorted(staged), published_utc=now(), commits_for_this_version=1))
        return commit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--publish-github', action='store_true')
    args = parser.parse_args()
    root = args.root.resolve()
    with (root / 'pipeline_controller.lock').open('a+') as claim:
        fcntl.flock(claim, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            annotations(root)
            wait_for_annotation_exit(root)
            run_stage(root, 'export_full_pool', [sys.executable, '-B', '-u', '-m',
                'code.V4.nonrouting_supplement', 'export', '--include-as', '--root', str(SUPPLEMENT)])
            common = ['--root', str(root), '--supplement-root', str(SUPPLEMENT / 'prepared_with_as'),
                '--logical-batch-size', '1024', '--microbatch-size', '320',
                '--microbatch-cap', 'MIS=96', '--microbatch-cap', 'MKP=120',
                '--eval-batch-size', '256', '--rank-weights', '5,5,3,3']
            run_stage(root, 'prepare_full24', [sys.executable, '-B', '-u', '-m',
                'code.V4.r120_experiment', 'prepare', '--cache-root', str(SHARED_CACHE), *common])
            launch = [sys.executable, '-B', '-u', '-m', 'code.V4.r120_launch', 'launch']
            run_stage(root, 'train_full24', [*launch, '--stage', 'train', *common])
            run_stage(root, 'locked_lib', [*launch, '--stage', 'lib', *common])
            run_stage(root, 'full_report', [sys.executable, '-B', '-u', '-m', 'code.V4.r120_experiment',
                'report', '--root', str(root), '--nss-root', str(NSS_REFERENCE)])
            report = read(root / 'reports/report_complete.json')
            if not report.get('complete'):
                raise RuntimeError('The final report has not completed')
            state(root, 'complete', report=str(root / 'reports/RESULTS.pdf'),
                  test_lock=str(root / 'final_test_lock.json'))
            if args.publish_github:
                state(root, 'publishing_completed_version')
                commit = publish(root)
                state(root, 'complete', commit=commit, report=str(root / 'reports/RESULTS.pdf'))
        except BaseException as error:
            state(root, 'failed', error=str(error), running_workers_interrupted=False)
            raise


if __name__ == '__main__':
    main()

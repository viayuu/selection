"""Quiet R97 completion/publication watcher. Never trains, infers or retries."""

import argparse
import fcntl
import os
from pathlib import Path
import subprocess
import time
import traceback

from .r80_reporting import file_hash, read_json, utc, write_json


REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / 'code/V4/runs/R97_tabular_in_context'
LAUNCH = ROOT.parent / 'R97_launch'


def optional(path):
    return read_json(path) if path.is_file() else {}


def status(root=ROOT, launch=LAUNCH):
    run, post, controller = (optional(path) for path in
        (root/'run_complete.json', root/'postprocess_status.json', launch/'A/controller.json'))
    errors = []
    for path in (root/'status.json', root/'postprocess_status.json',
                 launch/'A/controller.json', launch/'A/worker.json', launch/'A/queue_status.json'):
        record = optional(path)
        phase = str(record.get('phase', record.get('state', '')))
        if 'failed' in phase or record.get('exit_status') not in (None, 0):
            errors.append(dict(path=str(path), error=record.get('error', phase)))
    required = dict(full18_VAL_TEST_LIB=run.get('complete') is True,
        CPU_report=post.get('complete') is True and post.get('phase') == 'complete',
        controller_rc0=controller.get('state') == 'complete' and controller.get('exit_status') == 0)
    for name in ('MODEL_STATISTICS.pdf', 'selection_distributions.pdf', 'target_recovery.pdf'):
        path = root/'full_model_statistics'/name
        required[name] = path.is_file() and path.stat().st_size > 0
    result = dict(experiment='R97', checked_utc=utc(), required=required, errors=errors,
        complete=not errors and all(required.values()), cpu_only=True, automatic_retry=False,
        goal_complete=False, note='Receipts are completion evidence, not live GPU evidence')
    write_json(root/'goal_progress.json', result)
    return result


def publication_paths(root, launch, git_root):
    paths = [REPO/'AGENTS.md', REPO/'docs/research/post_r94_tabular_icl_screen.md',
             REPO/'docs/research/tabicl_runtime_feasibility.md']
    paths += [p for pattern in ('r97*.py', 'test_r97*.py', 'run_v4_r97*.sh')
              for p in (REPO/'code/V4').glob(pattern)]
    for base in (root, launch):
        for path in base.rglob('*'):
            if not path.is_file() or path.suffix not in ('.md', '.json', '.csv', '.pdf', '.png'):
                continue
            if any(part in ('runtime', 'wandb', 'evaluation', 'by_problem')
                   for part in path.relative_to(base).parts):
                continue
            if path.name == 'github_publication.json' or path.stat().st_size > 32*1024**2:
                continue
            paths.append(path)
    return sorted({str(p.resolve().relative_to(git_root)) for p in paths if p.is_file()})


def git(args, directory):
    return subprocess.check_output(['git', *args], cwd=directory, text=True).strip()


def publish(root=ROOT, launch=LAUNCH):
    git_root = Path(git(['rev-parse', '--show-toplevel'], REPO))
    git_dir = Path(git(['rev-parse', '--absolute-git-dir'], REPO))
    with (git_dir/'codex-publication.lock').open('a') as mutex:
        fcntl.flock(mutex, fcntl.LOCK_EX)
        if not status(root, launch)['complete']:
            raise RuntimeError('R97 whole-version delivery is not complete')
        receipt_path = root/'github_publication.json'
        prior = optional(receipt_path)
        if prior.get('complete') is True:
            return prior
        if receipt_path.exists():
            raise RuntimeError('Publication already attempted; no automatic retry')
        if git(['branch', '--show-current'], git_root) != 'main' or git(['diff', '--cached', '--name-only'], git_root):
            raise RuntimeError('Unexpected branch or existing staged changes; index left untouched')
        paths = publication_paths(root, launch, git_root)
        receipt = dict(complete=False, phase='staging', started_utc=utc(), files=paths,
            remote='origin/main', curated_only=True, automatic_retry=False)
        write_json(receipt_path, receipt)
        try:
            for start in range(0, len(paths), 80):
                subprocess.run(['git', 'add', '-f', '--', *paths[start:start+80]], cwd=git_root, check=True)
            staged = set(filter(None, git(['diff', '--cached', '--name-only', '-z'], git_root).split('\0')))
            if not staged or not staged.issubset(set(paths)):
                raise RuntimeError('Unexpected index mutation; no reset/revert was performed')
            subprocess.run(['git', 'commit', '-m', 'r97: publish frozen full18 tabular context selection'], cwd=git_root, check=True)
            receipt.update(phase='committed_pending_push', commit=git(['rev-parse', 'HEAD'], git_root))
            write_json(receipt_path, receipt)
            subprocess.run(['git', 'push', 'origin', 'HEAD:refs/heads/main'], cwd=git_root, check=True)
            receipt.update(complete=True, phase='complete', completed_utc=utc())
            write_json(receipt_path, receipt)
            return receipt
        except BaseException as error:
            receipt.update(phase='failed', error=str(error), completed_utc=utc())
            write_json(receipt_path, receipt)
            raise


def watch(root, launch, publish_github, interval=120):
    root.mkdir(parents=True, exist_ok=True)
    with (root/'goal_watch.lock').open('a') as mutex:
        fcntl.flock(mutex, fcntl.LOCK_EX | fcntl.LOCK_NB)
        receipt = dict(state='watching', pid=os.getpid(), started_utc=utc(),
            cpu_only=True, github_on_completion=publish_github, automatic_retry=False)
        write_json(root/'goal_watch_receipt.json', receipt)
        try:
            while True:
                current = status(root, launch)
                if current['errors']:
                    raise RuntimeError('R97 failed; no automatic rerun/retry')
                if current['complete']:
                    if publish_github:
                        publish(root, launch)
                    receipt.update(state='complete', completed_utc=utc(), exit_status=0)
                    write_json(root/'goal_watch_receipt.json', receipt)
                    return 0
                time.sleep(interval)
        except BaseException as error:
            receipt.update(state='stopped_on_error', error=str(error), exit_status=1,
                           completed_utc=utc(), traceback=traceback.format_exc())
            write_json(root/'goal_watch_receipt.json', receipt)
            return 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('status', 'watch', 'publish'))
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--launch-root', type=Path, default=LAUNCH)
    parser.add_argument('--publish-github', action='store_true')
    args = parser.parse_args(argv)
    if args.command == 'watch':
        return watch(args.root, args.launch_root, args.publish_github)
    if args.command == 'publish':
        publish(args.root, args.launch_root)
    else:
        print(status(args.root, args.launch_root), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

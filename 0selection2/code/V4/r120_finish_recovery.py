"""Finish an already trained R120 without repeating a single optimizer step.

TEST arrays were retained in the allocation-owned containers, not in the
shared TRAIN/VAL cache copied to the host-network DDP workers. This recovery
streams that existing frozen TEST cache to each host and evaluates the unique
already locked checkpoint. The original failed stage and logs are retained.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import fcntl
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time

from .r120_launch import CACHE, NODES, REPO, ROOT, capture, now, write


WRAPPER = REPO / 'code/V4/run_v4_r120_finish_recovery.sh'


def read(path):
    import json
    return json.loads(Path(path).read_text())


def copy_existing_test(node):
    item = NODES[node]
    cgroup = Path('/proc/self/cgroup').read_text()
    if (socket.gethostname().split('.')[0] != node
            or os.environ.get('SLURM_JOB_ID') != str(item['job'])
            or f'job_{item["job"]}' not in cgroup):
        raise RuntimeError('Cache transfer must stay inside its existing allocation CPU step')
    identity = capture(['sudo', '-n', '/usr/bin/ai_docker', 'inspect', '--format={{.Id}}',
                        item['container']])
    if identity != item['identity']:
        raise RuntimeError('The authorized allocation container changed')
    started = time.monotonic()
    CACHE.mkdir(parents=True, exist_ok=True)
    with (CACHE / '.r120_test_copy.lock').open('a+') as claim:
        fcntl.flock(claim, fcntl.LOCK_EX)
        marker = CACHE / '.r120_existing_test_copy_complete.json'
        if not marker.exists():
            argv = ['sudo', '-n', '/usr/bin/ai_docker', 'exec', '-u', str(os.getuid()),
                    item['container'], 'bash', str(WRAPPER), 'publish-test']
            # ai_docker emits platform text on stdout, so it is not a binary
            # tar transport. Copy through a shared, frozen array-only directory.
            subprocess.run(argv, check=True)
            source = ROOT / 'existing_test_cache'
            if not (source / 'complete.json').is_file():
                raise RuntimeError('The container did not publish a complete existing TEST cache')
            shutil.copytree(source / 'test', CACHE / 'test', dirs_exist_ok=True)
        # Opening the copied arrays checks the existing schema, candidate
        # identities, original FP64 costs and native winners. No preprocessing.
        from .r76_experiment import open_cache
        from code.unified_selector.registry import PROBLEMS
        config = read(ROOT / 'protocol.json')
        manifest = config['prepared']['routing_manifest']
        data_root = Path(config['prepared']['routing_data_root'])
        checked = {}
        for problem in PROBLEMS:
            cache = open_cache(CACHE, 'test', problem, manifest, data_root)
            try:
                checked[problem] = dict(rows=cache.size, pool=list(cache.pool))
            finally:
                cache.close()
        receipt = dict(complete=True, node=node, job=item['job'], container=identity,
            source='allocation-container:' + str(CACHE / 'test'), destination=str(CACHE / 'test'),
            updated_utc=now(), seconds=time.monotonic() - started, problems=checked,
            existing_frozen_arrays_only=True, preprocessing=False, solver_calls=0,
            checkpoint_changed=False, cgroup=cgroup)
        write(marker, receipt)
        write(ROOT / 'resources' / (node + '_test_cache_copy.json'), receipt)
        print('Copied existing TEST cache: ' + node, flush=True)


def publish_existing_test():
    cgroup = Path('/proc/self/cgroup').read_text()
    if not any(item['identity'] in cgroup for item in NODES.values()):
        raise RuntimeError('TEST arrays must come from an authorized allocation-owned container')
    source = ROOT / 'existing_test_cache'
    source.mkdir(parents=True, exist_ok=True)
    with (source / 'publish.lock').open('a+') as claim:
        fcntl.flock(claim, fcntl.LOCK_EX)
        if (source / 'complete.json').is_file():
            return
        if not (CACHE / 'test/TSP/metadata.json').is_file():
            raise FileNotFoundError('The container has no existing TEST cache')
        shutil.copytree(CACHE / 'test', source / 'test', dirs_exist_ok=True)
        write(source / 'complete.json', dict(complete=True, completed_utc=now(),
            source_container=socket.gethostname(), source=str(CACHE / 'test'),
            preprocessing=False, solver_calls=0))


def finish(publish):
    from . import r120_pipeline as pipeline
    complete, lock = read(ROOT / 'training_complete.json'), read(ROOT / 'final_test_lock.json')
    if not complete.get('complete') or complete['best']['update'] != lock['selected_update']:
        raise RuntimeError('Finish only the already completed train and unique VAL lock')
    if (ROOT / 'test_complete.json').exists():
        if read(ROOT / 'test_complete.json').get('checkpoint_sha256') != lock['checkpoint_sha256']:
            raise RuntimeError('Existing TEST belongs to another checkpoint')
    archive = ROOT / 'resources/test_startup_attempt1_missing_cache'
    archive.mkdir(parents=True, exist_ok=True)
    for path in (ROOT / 'pipeline_status.json', ROOT / 'pipeline_stages/train_full24.json',
                 ROOT / 'resources/train_exit.json'):
        destination = archive / path.name
        if path.is_file() and not destination.exists():
            shutil.copyfile(path, destination)
    with (ROOT / 'pipeline_controller.lock').open('a+') as claim:
        fcntl.flock(claim, fcntl.LOCK_EX | fcntl.LOCK_NB)
        pipeline.state(ROOT, 'copying_existing_test_cache', training_restarted=False)
        with ExitStack() as stack:
            workers = []
            for node, item in NODES.items():
                argv = ['srun', f'--jobid={item["job"]}', '--overlap', '--exact', '--gres=none',
                    '--immediate=10', '-N1', '-n1', '--nodelist=' + node,
                    '--cpus-per-task=1', '--mem=1G', 'bash', str(WRAPPER), 'copy-test', '--node', node]
                stream = stack.enter_context((ROOT / ('copy_test_' + node + '.log')).open('a'))
                process = subprocess.Popen(argv, cwd=REPO, stdout=stream, stderr=subprocess.STDOUT)
                workers.append((node, process))
            for node, process in workers:
                if process.wait():
                    raise RuntimeError('Existing cache transfer failed on ' + node)
        command = read(ROOT / 'pipeline_stages/train_full24.json')['command']
        index = command.index('--stage') + 1
        if command[index] != 'train':
            raise RuntimeError('Unexpected original model command; do not infer a training protocol')
        test = list(command)
        test[index] = 'test'
        pipeline.run_stage(ROOT, 'locked_test_existing_cache_recovery', test)
        lib = list(command)
        lib[index] = 'lib'
        pipeline.run_stage(ROOT, 'locked_lib', lib)
        pipeline.run_stage(ROOT, 'full_report', [sys.executable, '-B', '-u', '-m',
            'code.V4.r120_experiment', 'report', '--root', str(ROOT),
            '--nss-root', str(pipeline.NSS_REFERENCE)])
        if not read(ROOT / 'reports/report_complete.json').get('complete'):
            raise RuntimeError('The final report has not completed')
        write(ROOT / 'resources/test_cache_recovery.json', dict(complete=True, completed_utc=now(),
            training_restarted=False, repeated_optimizer_updates=0,
            original_training_successful_updates=complete['successful_updates'],
            selected_update=lock['selected_update'], checkpoint_sha256=lock['checkpoint_sha256'],
            original_failed_stage_retained=True))
        pipeline.state(ROOT, 'complete', report=str(ROOT / 'reports/RESULTS.pdf'))
        if publish:
            pipeline.CODE += ('r120_finish_recovery.py', 'run_v4_r120_finish_recovery.sh',
                              'r120_progress_report.py', 'r120_label_distribution.py')
            commit = pipeline.publish(ROOT)
            pipeline.state(ROOT, 'complete', report=str(ROOT / 'reports/RESULTS.pdf'), commit=commit)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('publish-test', 'copy-test', 'finish'))
    parser.add_argument('--node', choices=tuple(NODES))
    parser.add_argument('--publish-github', action='store_true')
    args = parser.parse_args()
    if args.command == 'publish-test':
        publish_existing_test()
        return 0
    if args.command == 'copy-test':
        if args.node is None:
            parser.error('copy-test requires --node')
        copy_existing_test(args.node)
        return 0
    finish(args.publish_github)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

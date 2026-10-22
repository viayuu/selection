"""One explicit, allocation-bound R99 post-training stage; never run TRAIN/TEST."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from . import r84_launch as resource


ROOT = resource.REPO / 'code/V4/runs/R99_global_competition_input'
CPUS = dict(A=[12, 13, 14, 15], B=[16, 17, 64, 65])
SLOTS = dict(A='3090_0', B='3090_1')


def execute(arm, stage):
    slot = SLOTS[arm]
    resource.assert_container(slot)
    gpu = resource.SLOTS[slot]['uuid']
    if os.environ.get('CUDA_VISIBLE_DEVICES') != gpu:
        raise RuntimeError('Bind exactly the original R99 GPU UUID')
    if not set(CPUS[arm]) <= os.sched_getaffinity(0):
        raise RuntimeError('Original allocated CPU quartet is unavailable')
    os.sched_setaffinity(0, set(CPUS[arm]))
    dispatch = resource.REPO / 'code/V4/runs/R98_parallel_dispatch' / ('R99_' + arm)
    worker, controller = (resource.read(dispatch / n) for n in ('worker.json', 'controller.json'))
    if (worker.get('state') != 'failed' or worker.get('phase') != 'timing'
            or controller.get('state') != 'failed'
            or [(s.get('name'), s.get('state'), s.get('exit_status'))
                for s in worker.get('stages', [])[:3]] !=
                [('input_features', 'complete', 0), ('train', 'complete', 0), ('TEST', 'complete', 0)]):
        raise RuntimeError('Only the documented post-TEST timing failure may be recovered')
    run = ROOT / arm
    lock = resource.read(run / 'final_test_lock.json')
    test = resource.read(run / 'test_results.json')
    if (test.get('complete') is not True or any(test.get(k) != lock.get(k)
            for k in ('checkpoint_sha256', 'state_hash', 'selected_update'))):
        raise RuntimeError('The unique original checkpoint TEST is not complete')
    receipt_path = ROOT / 'preparation/manual_stages' / (arm + '_' + stage + '.json')
    if stage == 'lib':
        argv = [sys.executable, '-B', '-u', '-m', 'code.V4.r99_reporting', 'lib',
                '--root', str(ROOT), '--arm', arm, '--device', 'cuda:0', '--lib-batch-size', '2']
    else:
        argv = [sys.executable, '-B', '-u', '-m', 'code.V4.r99_postprocess_recovery',
                stage, '--root', str(ROOT), '--arm', arm, '--device', 'cuda:0',
                '--cache-root', '/tmp/shiys_R84_new90k_cache',
                '--feature-root', '/tmp/shiys_R99_' + arm + '_relation_features']
        if stage == 'diagnose':
            argv += ['--problems', 'TSP', '--tf32-policies', 'default', 'off', '--max-seconds', '90']
    gpu_lock = resource.transport.ROOT / ('queue_gpu_' + gpu + '.json')
    with resource.lock(gpu_lock, existing=True):
        if receipt_path.exists():
            raise RuntimeError('This explicit stage already attempted; retain it, no automatic retry')
        value = dict(experiment='R99', arm=arm, stage=stage, state='running',
            started_utc=resource.utc(), slot=slot, gpu_uuid=gpu,
            container_id=resource.assert_container(slot), cpus=CPUS[arm],
            argv=argv, automatic_retry=False, original_failure_preserved=True,
            original_worker_sha256=resource.sha(dispatch / 'worker.json'),
            original_controller_sha256=resource.sha(dispatch / 'controller.json'),
            checkpoint_lock=lock, training=False, test_rerun=False, model_reselection=False)
        resource.save(receipt_path, value, exclusive=True)
        try:
            result = subprocess.run(argv, cwd=resource.REPO, check=False)
            value.update(state='complete' if result.returncode == 0 else 'failed',
                         exit_status=result.returncode, completed_utc=resource.utc())
            if result.returncode:
                raise RuntimeError(stage + ' failed; preserve original attempt, no retry')
        except BaseException as error:
            value.update(state='failed', error=str(error), completed_utc=resource.utc())
            value.setdefault('exit_status', 1)
            raise
        finally:
            resource.save(receipt_path, value)
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', choices=('A', 'B'), required=True)
    parser.add_argument('--stage', choices=('lib', 'diagnose', 'timing'), required=True)
    args = parser.parse_args()
    value = execute(args.arm, args.stage)
    print(json.dumps({k: value[k] for k in ('arm', 'stage', 'state', 'exit_status')}))


if __name__ == '__main__':
    main()

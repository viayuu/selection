"""One explicit R102 first-TEST recovery in the original allocation; no TRAIN."""

import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import subprocess
import sys

from . import r84_launch as resource


ROOT = resource.REPO / 'code/V4/runs/R102_full_type_pretrained_fusion'
LAUNCH = resource.REPO / 'code/V4/runs/R98_parallel_dispatch'
CPUS = dict(A=[0, 1, 2, 3], B=[4, 5, 72, 73])
SLOTS = dict(A='4090_0', B='4090_1')
LOCKS = dict(A=(33120, '10694b4d0fed80f1baec0d5e7de15cceec210eca5dfa846623783c4b9d6e7a2c'),
             B=(20160, '7df434c762a0e0b7726ab67211f63f51b55d1f86d108ec307a0aede0060f921f'))
UUID_ERROR = 'Actual CUDA UUID differs from the original assignment'
UUID_CORRECTION_ENTRY = 'gpu_uuid_format_correction_entry.json'
TIMING_DRIFT_ERROR = 'R102 cached-vs-online drift; preserve first error, do not change inference/grouping'
POST_TEST_ENTRY = 'post_test_entry.json'


def uuid_preflight_observation(arm, *, require_absent=False):
    base = ROOT / 'preparation/postprocess_recovery'
    path = base / 'uuid_preflight_observation.json'
    observation = resource.read(path)
    if (observation.get('schema') != 'r102_uuid_preflight_observation_v1'
            or observation.get('main_authorizes_one_explicit_format_correction') is not True
            or set(observation.get('arms', {})) != set(SLOTS)):
        raise RuntimeError('Require main observation of BOTH original UUID preflight failures')
    for side, row in observation['arms'].items():
        if (row.get('error') != UUID_ERROR or any(row.get(key) is not True
                for key in ('origin_absent', 'shadow_absent', 'test_outputs_absent'))):
            raise RuntimeError('Only the observed pre-origin UUID-format failure may continue')
        for key, expected in (('entry', base / side / 'gpu_entry.json'),
                              ('log', base / (side + '_gpu.log'))):
            record = row[key]
            if (record.get('path') != str(expected) or expected.is_symlink()
                    or resource.sha(expected) != record.get('sha256')):
                raise RuntimeError('Original UUID preflight receipt/log changed')
        failed = resource.read(base / side / 'gpu_entry.json')
        if (failed.get('state') != 'failed' or type(failed.get('exit_status')) is not int
                or failed['exit_status'] != 1 or failed.get('arm') != side or failed.get('stage') != 'gpu'
                or UUID_ERROR not in (base / (side + '_gpu.log')).read_text()):
            raise RuntimeError('Need the exact failed GPU preflight receipt and CUDA UUID error log')
        if require_absent and side == arm:
            absent = [base / side / name for name in ('origin.json', 'test_started.json',
                'test_results.json', 'test_complete.json', 'gpu_complete.json', 'first_error.json')]
            absent.append(base / 'report_view' / side)
            if any(item.exists() or item.is_symlink() for item in absent):
                raise RuntimeError('UUID correction is only before origin/shadow/TEST creation')
    row = observation['arms'][arm]
    return dict(uuid_preflight_observation=dict(path=str(path), sha256=resource.sha(path)),
                preserved_gpu_entry=row['entry'], preserved_gpu_log=row['log'])


def original(arm):
    run, dispatch = ROOT / arm, LAUNCH / ('R102_' + arm)
    worker, controller = (resource.read(dispatch / name) for name in ('worker.json', 'controller.json'))
    lock, train, status, error = (resource.read(run / name) for name in
        ('final_test_lock.json', 'training_complete.json', 'status.json', 'error.json'))
    if (worker.get('state') != 'failed' or worker.get('phase') != 'train' or worker.get('exit_status') != 1
            or [(s.get('name'), s.get('state'), s.get('exit_status')) for s in worker.get('stages', [])]
                != [('input_features', 'complete', 0), ('train', 'failed', 1)]
            or controller.get('state') != 'failed' or controller.get('exit_status') != 1
            or status.get('phase') != 'failed' or status.get('successful_updates') != 57600
            or error.get('message') != 'Full frozen R84 header identity differs'
            or train.get('successful_updates') != 57600 or train.get('selected_best') != lock
            or (lock.get('selected_update'), lock.get('checkpoint_sha256')) != LOCKS[arm]
            or lock.get('selection_split') != 'val' or lock.get('selected_without_test') is not True
            or lock.get('per_problem_assembly') is not False):
        raise RuntimeError('Only the original failed TRAIN wrapper, full57600 and unique VAL lock may continue')
    item = resource.SLOTS[SLOTS[arm]]
    if (worker.get('slot') != SLOTS[arm] or worker.get('gpu_uuid') != item['uuid']
            or worker.get('cpus') != CPUS[arm]
            or worker.get('container_id') != resource.transport.CONTAINERS[item['node']][1]):
        raise RuntimeError('Original resource assignment changed')
    names = ('protocol.json', 'initialization.json', 'final_test_lock.json', 'training_complete.json',
             'final_metrics.json', 'status.json', 'error.json', 'test_started.json')
    paths = {name: run / name for name in names}
    paths.update({'worker.json': dispatch / 'worker.json', 'controller.json': dispatch / 'controller.json',
                  'goal_watch_receipt.json': ROOT / 'goal_watch_receipt.json'})
    return dict(checkpoint_lock=lock, sha256={name: resource.sha(path) for name, path in paths.items()})


def post_test_evidence(arm):
    base = ROOT / 'preparation/postprocess_recovery'
    previous = base / arm / UUID_CORRECTION_ENTRY
    receipt = resource.read(previous)
    observation = uuid_preflight_observation(arm)
    failure = base / arm / 'first_error.json'
    error = resource.read(failure)
    test_path = base / 'report_view' / arm / 'test_results.json'
    test = resource.read(test_path)
    lock = resource.read(ROOT / arm / 'final_test_lock.json')
    if (receipt.get('state') != 'failed' or receipt.get('exit_status') != 1
            or receipt.get('uuid_format_correction') is not True or receipt.get('original') != original(arm)
            or any(receipt.get(key) != value for key, value in observation.items())
            or error.get('stage') != 'timing' or error.get('type') != 'ValueError' or error.get('exit_status') != 1
            or error.get('message') != TIMING_DRIFT_ERROR
            or test.get('complete') is not True
            or any(test.get(key) != lock[key] for key in ('checkpoint_sha256', 'state_hash', 'selected_update'))):
        raise RuntimeError('Post-test continuation requires the exact timing drift and completed same-lock first TEST')
    return dict(**observation, preserved_uuid_correction_entry=dict(path=str(previous), sha256=resource.sha(previous)),
        preserved_first_error=dict(path=str(failure), sha256=resource.sha(failure)),
        preserved_completed_test=dict(path=str(test_path), sha256=resource.sha(test_path)))


def completed(arm, stage, lock):
    stage = 'gpu' if stage == 'post-test' else stage
    path = ROOT / 'preparation/postprocess_recovery' / arm / (stage + '_complete.json')
    if not path.is_file():
        return False
    record = resource.read(path)
    return (record.get('schema') == 'r102_postprocess_recovery_v1' and record.get('complete') is True
        and record.get('exit_status') == 0 and record.get('arm') == arm
        and all(record.get(key) == lock[key] for key in ('checkpoint_sha256', 'state_hash', 'selected_update')))


@contextmanager
def reservation(arm):
    slot = SLOTS[arm]
    item = resource.SLOTS[slot]
    identity = resource.assert_container(slot)
    if (item['job'] != 1475 or item['node'] != 'gpu05'
            or os.environ.get('CUDA_VISIBLE_DEVICES') != item['uuid']):
        raise RuntimeError('Use the original1475/gpu05 container and exact original GPU UUID')
    if not set(CPUS[arm]) <= os.sched_getaffinity(0):
        raise RuntimeError('Original allocated CPU quartet is unavailable')
    path = resource.transport.ROOT / ('queue_gpu_' + item['uuid'] + '.json')
    if path.is_symlink():
        raise RuntimeError('Only the existing original R81 GPU mutex is permitted')
    expected = path.stat()
    # Never create/replace this file: other queues synchronize on its inode.
    with path.open('r') as handle:
        stat = os.fstat(handle.fileno())
        actual = dict(path=str(path), device=stat.st_dev, inode=stat.st_ino)
        if (stat.st_dev, stat.st_ino) != (expected.st_dev, expected.st_ino):
            raise RuntimeError('Original GPU mutex inode changed; no lock recreation')
        resource.fcntl.flock(handle, resource.fcntl.LOCK_EX | resource.fcntl.LOCK_NB)
        if (path.stat().st_dev, path.stat().st_ino) != (stat.st_dev, stat.st_ino):
            raise RuntimeError('GPU mutex replaced while acquiring reservation')
        prior = os.sched_getaffinity(0)
        try:
            os.sched_setaffinity(0, set(CPUS[arm]))
            yield dict(slot=slot, gpu_uuid=item['uuid'], container_id=identity,
                       cpus=CPUS[arm], cpu_count=4, allocation_memory='9G', gpu_mutex=actual)
        finally:
            os.sched_setaffinity(0, prior)


def command(arm, stage):
    gpu = stage in ('gpu', 'post-test')
    argv = [sys.executable, '-B', '-u', '-m', 'code.V4.r102_postprocess_recovery',
            'finish-gpu' if stage == 'post-test' else stage,
            '--root', str(ROOT), '--arm', arm, '--device', 'cuda:0' if gpu else 'cpu']
    if gpu:
        argv += ['--cache-root', '/tmp/shiys_R84_new90k_cache',
                 '--feature-root', '/tmp/shiys_R102_' + arm + '_extra_nodes',
                 '--routing-feature-root', '/tmp/shiys_R94_B_node_features']
    return argv


def _execute(arm, stage, evidence, resources, *, uuid_format_correction=False):
    if uuid_format_correction and stage != 'gpu':
        raise ValueError('UUID format correction is only an explicit GPU preflight continuation')
    origin = original(arm)
    base = ROOT / 'preparation/postprocess_recovery'
    name = UUID_CORRECTION_ENTRY if uuid_format_correction else POST_TEST_ENTRY if stage == 'post-test' else stage + '_entry.json'
    path = base / arm / name
    if path.exists() or (base / arm / (('gpu' if stage == 'post-test' else stage) + '_complete.json')).exists():
        raise RuntimeError('Stage already attempted; preserve all receipts, no terminal retry')
    correction = uuid_preflight_observation(arm, require_absent=True) if uuid_format_correction else {}
    continuation = post_test_evidence(arm) if stage == 'post-test' else {}
    if stage == 'gpu':
        if any((ROOT / arm / name).exists() or (ROOT / arm / name).is_symlink()
               for name in ('test_results.json', 'test_complete.json', 'test_input_features_ready.json', 'evaluation/test')):
            raise RuntimeError('Original TEST outputs must not exist; first completed TEST only')
    elif stage == 'report' and not completed(arm, 'gpu', origin['checkpoint_lock']):
        raise RuntimeError('CPU report requires the new complete same-lock GPU stage')
    argv = command(arm, stage)
    sources = [Path(__file__), resource.REPO / 'code/V4/run_v4_r102_gpu_recovery.sh',
               resource.REPO / 'code/V4/r102_postprocess_recovery.py',
               resource.REPO / 'code/V4/r102_cache_identity_probe.py',
               resource.REPO / 'code/V4/run_v4_r102_cache_probe.sh']
    value = dict(schema='r102_recovery_entry_v1', experiment='R102', arm=arm, stage=stage,
        state='running', started_utc=resource.utc(), argv=argv, main_ready_evidence=evidence,
        original=origin, original_worker_sha256=origin['sha256']['worker.json'],
        original_controller_sha256=origin['sha256']['controller.json'],
        source_sha256={str(source): resource.sha(source) for source in sources},
        checkpoint_lock=origin['checkpoint_lock'], resources=resources,
        automatic_retry=False, training=False, optimizer=False, model_reselection=False,
        first_full_TEST=stage == 'gpu', original_failure_preserved=True)
    if uuid_format_correction:
        value.update(uuid_format_correction=True, **correction)
    if stage == 'post-test':
        value.update(post_test_continuation=True, test_rerun=False, inference_rule_changed=False, **continuation)
    resource.save(path, value, exclusive=True)
    try:
        environment = dict(os.environ)
        if stage == 'report':
            environment.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
        environment['PYTHONDONTWRITEBYTECODE'] = '1'
        result = subprocess.run(argv, cwd=resource.REPO, env=environment, check=False)
        value['exit_status'] = result.returncode
        if result.returncode or not completed(arm, stage, origin['checkpoint_lock']):
            raise RuntimeError('Recovery needs rc0 AND validated completion evidence; no retry')
        if original(arm) != origin:
            raise RuntimeError('Original failure/lock evidence changed during recovery')
        if uuid_format_correction and uuid_preflight_observation(arm) != correction:
            raise RuntimeError('UUID preflight observation changed during explicit correction')
        if stage == 'post-test' and post_test_evidence(arm) != continuation:
            raise RuntimeError('Prior failure or completed TEST changed during post-test continuation')
        value.update(state='complete', completed_utc=resource.utc())
    except BaseException as error:
        value.update(state='failed', error=str(error), completed_utc=resource.utc())
        if value.get('exit_status', 0) == 0:
            value['exit_status'] = 1
        raise
    finally:
        resource.save(path, value)
    return value


def execute(arm, stage, evidence, *, uuid_format_correction=False):
    if arm not in SLOTS or stage not in ('gpu', 'post-test', 'report'):
        raise ValueError('Only A/B gpu/post-test/report postprocessing is permitted')
    if not isinstance(evidence, str) or not evidence.strip():
        raise ValueError('Main must explicitly authorize this one-shot recovery')
    if uuid_format_correction and stage != 'gpu':
        raise ValueError('--uuid-format-correction requires --stage gpu')
    if stage in ('gpu', 'post-test'):
        with reservation(arm) as resources:
            return _execute(arm, stage, evidence, resources, uuid_format_correction=uuid_format_correction)
    return _execute(arm, stage, evidence, dict(cpu_only=True, CUDA_VISIBLE_DEVICES=''))


def parser():
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument('--arm', choices=('A', 'B'), required=True)
    value.add_argument('--stage', choices=('gpu', 'post-test', 'report'), required=True)
    value.add_argument('--evidence', required=True)
    value.add_argument('--uuid-format-correction', action='store_true')
    return value


def main(argv=None):
    args = parser().parse_args(argv)
    value = execute(args.arm, args.stage, args.evidence, uuid_format_correction=args.uuid_format_correction)
    print(json.dumps({key: value[key] for key in ('arm', 'stage', 'state', 'exit_status')}))


if __name__ == '__main__':
    main()

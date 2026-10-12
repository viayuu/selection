"""One authorized original18 TEST per complete validation-frozen R66 bundle."""

import argparse
import gc
import json
import os
import socket
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from code.unified_selector.registry import PROBLEMS
from .multitask_probe import file_hash, state_hash
from .performance_targets import read_raw_costs
from .r59_experiment import GPU_UUID, WORKSPACE, atomic_json
from .r64_experiment import evaluate, synchronize
from .r66_experiment import ADOPTED_PLAN, ROOT, RUN_NAMES, source_hashes
from .r66_model import BASE_PATH, BASE_SHA256, INFERENCE_RULE, load_deployment_bundle
from .train import configure_torch, make_loader


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def require_terminal(receipt):
    if (receipt['state'] != 'COMPLETED' or receipt['exit_code'] != '0:0'
            or receipt['tmux_terminal'] is not True or receipt['slurm_step'] != '1465.94'
            or receipt['tmux_session'] != 'r66_train'):
        raise ValueError('R66 TEST requires actual successful terminal evidence for the owned training')


def validate_lock(root):
    protocol = read(root/'protocol.json')
    candidates = read(root/'validation_candidates.json')
    lock = read(root/'final_test_lock.json')
    terminal = read(root/'training_terminal_receipt.json')
    require_terminal(terminal)
    if (lock.get('authorized') is not True or set(lock.get('models', {})) != {'A', 'B'}
            or lock.get('inference_rule') != INFERENCE_RULE or lock.get('coefficient') != 1.
            or candidates['inference_rule'] != INFERENCE_RULE or candidates['coefficient'] != 1.
            or lock.get('official_arm') != candidates['proposed_official_arm']
            or lock.get('proposed_official_arm') != candidates['proposed_official_arm']
            or lock.get('base_checkpoint') != str(BASE_PATH.relative_to(WORKSPACE))
            or lock.get('base_checkpoint_sha256') != BASE_SHA256
            or lock.get('base_state_sha256') != protocol['base_state_sha256']
            or lock.get('base_parameter_mode') != 'averaged x'
            or lock.get('training_terminal_receipt') != 'training_terminal_receipt.json'):
        raise ValueError('TEST requires the exact parent-authorized complete pair, base and deployment rule')
    if (protocol['source_hashes'] != source_hashes()
            or read(root/'source_launch/hashes.json') != protocol['source_hashes']
            or protocol['adopted_plan_sha256'] != file_hash(ADOPTED_PLAN)):
        raise ValueError('Frozen R66 launch sources or adopted plan changed')
    proposed = 'A' if candidates['models']['A']['native_correct'] >= candidates['models']['B']['native_correct'] else 'B'
    if proposed != lock['official_arm']:
        raise ValueError('Official arm must be the published maximum validation count, tie A')
    planned = read(root/'planned_configs.json')
    for arm in ('A', 'B'):
        record, expected = lock['models'][arm], candidates['models'][arm]
        if any(record.get(key) != value for key, value in expected.items()):
            raise ValueError('Lock must retain every published validation candidate field exactly')
        if (type(record['native_correct']) is not int or type(record['epoch']) is not int
                or record['native_total'] != 18000 or record['parameter_mode'] != 'averaged x'):
            raise ValueError('Freeze requires original integer validation counts and saved averaged x')
        directory = root/RUN_NAMES[arm]
        published, unique = Path(record['path']), Path(record['unique_bundle_path'])
        expected_unique = directory/'best_checkpoints'/f'epoch{record["epoch"]:03d}.pt'
        if published != directory/'best.pt' or unique != expected_unique:
            raise ValueError('Only the published best and its original unique selected-epoch bundle are allowed')
        if (file_hash(published) != record['sha256']
                or file_hash(unique) != record['unique_bundle_sha256']):
            raise ValueError('A frozen complete R66 bundle changed')
        result = read(directory/'result.json')
        if ((result['epochs_completed'], result['successful_updates'], result['presentations']) != (40, 9600, 7200000)
                or result['best']['epoch'] != record['epoch']
                or result['best']['val']['native_correct'] != record['native_correct']
                or result['best_checkpoint_sha256'] != record['sha256']):
            raise ValueError('Both fixed-horizon training results must agree with their published candidates')
        # Atomic torch saves use different archive names; compare the actual saved states.
        bundles = [torch.load(path, map_location='cpu', weights_only=False) for path in (published, unique)]
        for bundle in bundles:
            if (bundle['epoch']+1 != record['epoch'] or bundle['native_correct'] != record['native_correct']
                    or bundle['native_total'] != 18000 or bundle['parameter_mode'] != 'averaged x'
                    or bundle['coefficient'] != 1. or bundle['inference_rule'] != INFERENCE_RULE
                    or bundle['args'] != planned[arm] or bundle['args'] != read(directory/'args.json')
                    or bundle['args']['group'] != arm or bundle['base_checkpoint_sha256'] != BASE_SHA256
                    or bundle['base_state_sha256'] != protocol['base_state_sha256']
                    or state_hash(bundle['base']) != protocol['base_state_sha256']
                    or bundle['selector_forwards'] != 2 or bundle['solver_executions'] != 1
                    or bundle['cache_allowed_at_deployment'] is not False):
                raise ValueError('Complete saved-x bundle metadata or fixed base identity mismatch')
        if state_hash(bundles[0]['member']) != state_hash(bundles[1]['member']):
            raise ValueError('Unique epoch bundle must contain exactly the published selected member x')
        del bundles
    return protocol, lock


def final_test(root, device):
    protocol, lock = validate_lock(root)
    marker = dict(started_utc=now(), started_unix=time.time(), lock=lock,
        evaluation_module=str(Path(__file__).resolve()), evaluation_module_sha256=file_hash(Path(__file__)),
        slurm_step=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],
        tmux_session='r66_final_test', host=socket.gethostname(), gpu_uuid=GPU_UUID,
        one_complete_original_test_per_frozen_model=True, allowed_attempts=1,
        batch_size=128, cached_base_inference=False, optimizer_mode_transition=False,
        selector_forwards_per_query=2, completed_solvers_per_query=1)
    with (root/'final_test_started.json').open('x') as stream:
        json.dump(marker, stream, indent=2, allow_nan=False)
        stream.write('\n')
    started, results, current = time.monotonic(), {}, None
    try:
        raw, loaders, manifest = {}, {}, {}
        for task in PROBLEMS:
            record = read_raw_costs(task, 'test')
            original = protocol['data_manifest']['train'][task]
            if (len(record['winner']) != 1000 or record['costs'].dtype != np.float64
                    or any(record[key] != original[key] for key in ('pool', 'pool_ids'))):
                raise ValueError('TEST must retain original1000 rows/task, FP64 costs and complete legal pools')
            loader = make_loader(task, 'test', 128, 0, shuffle=False, cache_device=device)
            loader.drop_last = False
            if loader.size != 1000:
                raise ValueError('TEST input coverage must be exactly1000 original rows/task')
            np.testing.assert_array_equal(loader.batch['ind'].cpu(), record['winner'])
            np.testing.assert_array_equal(loader.batch['pool_ids'].cpu(), record['pool_ids'])
            np.testing.assert_array_equal(loader.batch['costs'].cpu(), record['costs'].astype(np.float32))
            raw[task], loaders[task] = record, loader
            manifest[task] = dict(rows=1000, pool=record['pool'], pool_ids=record['pool_ids'],
                label_hash=record['label_hash'], data_hash=record['data_hash'], costs_dtype='raw FP64',
                row_names='str(original zero-based dataset/label row index)',
                index_range_inclusive=[0, 999], ordered_original_rows=True)
        atomic_json(root/'test_data_manifest.json', manifest)
        for arm in ('A', 'B'):
            current = arm
            record = lock['models'][arm]
            atomic_json(root/'test_progress.json', dict(status='evaluating', arm=arm,
                completed_arms=list(results), recorded_utc=now(), duplicate_attempt_allowed=False))
            model = load_deployment_bundle(record['unique_bundle_path'], device=device)
            if model.training or model.base.training or model.member.training or model.use_cached_base:
                raise ValueError('TEST requires both live selectors in eval with saved x and caching disabled')
            if any(parameter.dtype != torch.float32 for parameter in model.parameters()):
                raise ValueError('Normal TEST inference requires FP32 parameters')
            synchronize(device)
            before = time.monotonic()
            value = evaluate(model, loaders, raw, batch_size=128,
                prediction_dir=root/RUN_NAMES[arm]/'test_predictions', timing_path=None)
            synchronize(device)
            elapsed = time.monotonic()-before
            if (set(value['per_problem']) != set(PROBLEMS) or value['native_total'] != 18000
                    or any(entry['n'] != 1000 for entry in value['per_problem'].values())):
                raise ValueError('Each frozen bundle must complete all18 original TEST tasks exactly once')
            value.update(complete=True, evaluation_passes=1, complete_evaluation_seconds=elapsed,
                evaluated_bundle_path=record['unique_bundle_path'],
                evaluated_bundle_sha256=record['unique_bundle_sha256'],
                parameter_mode='averaged x', coefficient=1., inference_rule=INFERENCE_RULE,
                selector_forwards=2, solver_executions=0, deployment_completed_solvers_per_query=1,
                cached_base_inference=False, batch_size=128, tails_retained=True,
                timing_scope='ordinary full evaluation: live F and member forwards, gather/transfer, metrics and prediction saving; no optimizer or solver')
            results[arm] = value
            atomic_json(root/('test_results_'+arm+'.json'), value)
            atomic_json(root/'test_progress.json', dict(status='arm_complete', arm=arm,
                completed_arms=list(results), recorded_utc=now(), duplicate_attempt_allowed=False))
            print('[R66 locked TEST] '+json.dumps(dict(arm=arm, native_correct=value['native_correct'],
                native_total=value['native_total'], complete_evaluation_seconds=elapsed)), flush=True)
            del model
            gc.collect()
            torch.cuda.empty_cache()
        atomic_json(root/'test_results.json', dict(models=results, lock=lock, complete=True,
            wall_seconds=time.monotonic()-started, tests_per_frozen_model=1,
            original_full18=True, references_reuse_existing_official_tests=True,
            selector_forwards_per_query=2, deployment_completed_solvers_per_query=1,
            cached_base_inference=False, separate_latency_benchmark=False, solver_executions=0))
        atomic_json(root/'test_progress.json', dict(status='complete', completed_arms=list(results),
            recorded_utc=now(), duplicate_attempt_allowed=False))
        print('[R66 paired locked TEST complete] no rerun or reselection', flush=True)
    except BaseException as error:
        atomic_json(root/'final_test_failure.json', dict(recorded_utc=now(), failed_arm=current,
            completed_arms=list(results), exception_type=type(error).__name__, message=str(error),
            traceback=traceback.format_exc(), marker_preserved=True, rerun_authorized=False))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    args = parser.parse_args()
    root = args.root.resolve()
    if (root != ROOT.resolve() or os.environ.get('R66_PARENT_GPU_BATON') != 'locked-test'
            or os.environ.get('CUDA_VISIBLE_DEVICES') != GPU_UUID
            or os.environ.get('SLURM_JOB_ID') != '1465' or socket.gethostname().split('.')[0] != 'gpu03'
            or os.environ.get('R66_TMUX_SESSION') != 'r66_final_test'):
        parser.error('Use the locked-test launcher on existing1465/gpu03/assigned UUID and canonical R66 root')
    torch.set_num_threads(1)
    configure_torch()
    final_test(root, 'cuda:0')


if __name__ == '__main__':
    main()

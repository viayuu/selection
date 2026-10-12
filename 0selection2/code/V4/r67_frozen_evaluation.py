"""Future parent freeze and one authorized original18 TEST of saved R67 selectors."""

import argparse
import gc
import json
import math
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
from .r64_model import INFERENCE_RULE, NativeUtilitySelector
from .r67_experiment import ADOPTED_PLAN, ROOT, RUN_NAMES, source_hashes
from .r67_model import INPUT_KEYS
from .train import configure_torch, make_loader


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def delivery_hashes():
    paths = [WORKSPACE/'code/V4'/name for name in
             ('r67_frozen_evaluation.py', 'run_v4_r67_test.sh', 'r67_analysis.py')]
    paths.append(ROOT/'DELIVERY_PROTOCOL.md')
    return {str(path.relative_to(WORKSPACE)): file_hash(path) for path in paths}


def require_terminal(receipt, session, expected_step=None):
    if (receipt.get('state') != 'COMPLETED' or receipt.get('exit_code') != '0:0'
            or receipt.get('tmux_terminal') is not True or receipt.get('tmux_session') != session
            or not receipt.get('slurm_step', '').startswith('1465.')
            or (expected_step is not None and receipt['slurm_step'] != expected_step)):
        raise ValueError('Require actual terminal0 evidence for '+session+' and its owned Slurm step')


def completed_training(root):
    protocol, candidates = read(root/'protocol.json'), read(root/'validation_candidates.json')
    complete = read(root/'paired_training_complete.json')
    terminal = read(root/'training_terminal_receipt.json')
    require_terminal(terminal, 'r67_train', read(root/'launch_receipt.json')['slurm_step'])
    if (protocol['source_hashes'] != source_hashes()
            or read(root/'source_launch/hashes.json') != protocol['source_hashes']
            or protocol['adopted_plan_sha256'] != file_hash(ADOPTED_PLAN)
            or candidates['inference_rule'] != INFERENCE_RULE
            or set(candidates['models']) != {'A', 'B'}
            or (complete['epochs_per_arm'], complete['updates_per_arm'], complete['presentations_per_arm'],
                complete['updates_pair'], complete['presentations_pair']) != (80, 19200, 14400000, 38400, 28800000)):
        raise ValueError('Require unchanged launch sources and actual complete published paired training')
    records = {}
    for arm, name in RUN_NAMES.items():
        directory = root/name
        result, history = read(directory/'result.json'), read(directory/'history.json')
        launches = read(directory/'wall_time.json')
        first = [row for row in history if row['stage'] == 1]
        second = [row for row in history if row['stage'] == 2]
        if (len(first) != 40 or len(second) != 40
                or [r['epoch'] for r in first] != list(range(1, 41))
                or [r['epoch'] for r in second] != list(range(1, 41))
                or any(r['val'] is not None for r in first)
                or any(r['val']['native_total'] != 18000 or type(r['val']['native_correct']) is not int for r in second)
                or (result['epochs_completed'], result['successful_updates'], result['presentations']) != (80, 19200, 14400000)
                or (result['actual_execution']['recorded_successful_optimizer_steps'],
                    result['actual_execution']['recorded_query_presentations']) != (19200, 14400000)
                or [(r['stage'], r['successful_updates'], r['presentations']) for r in result['stages']]
                   != [(1, 9600, 7200000), (2, 9600, 7200000)]
                or not launches or any(r['status'] != 'complete' or r['slurm_step'] != terminal['slurm_step'] for r in launches)):
            raise ValueError('Both actual fixed40+40epoch arms must finish before freeze')
        selected = max(second, key=lambda row: row['val']['native_correct'])
        candidate = candidates['models'][arm]
        if (candidate['epoch'] != selected['epoch'] or candidate['global_epoch'] != selected['global_epoch']
                or candidate['native_correct'] != selected['val']['native_correct']
                or result['best']['epoch'] != selected['epoch']
                or result['best']['val']['native_correct'] != candidate['native_correct']
                or candidate['native_total'] != 18000 or candidate['parameter_mode'] != 'averaged x'
                or candidate['projector_included'] is not False
                or Path(candidate['path']).resolve() != directory/'best.pt'
                or result['best_checkpoint_sha256'] != candidate['sha256']):
            raise ValueError('Only earliest maximum INTEGER validation count may choose stage2 best')
        if any(set(row['val']['per_problem']) != set(PROBLEMS)
               or any(value['n'] != 1000 for value in row['val']['per_problem'].values()) for row in second):
            raise ValueError('Every stage2 selection evaluation must cover all original18 tasks and rows')
        records[arm] = candidate
    official = 'A' if records['A']['native_correct'] >= records['B']['native_correct'] else 'B'
    if candidates['proposed_official_arm'] != official:
        raise ValueError('Official arm must use validation integer count, exacttie A')
    return protocol, candidates, records, official


def read_selector_checkpoint(path, record, config):
    path = Path(path).resolve()
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    if (checkpoint.get('stage') != 2 or checkpoint.get('projector_included') is not False
            or checkpoint.get('parameter_mode') != 'averaged x' or checkpoint.get('checkpoint_mode') != 'eval'
            or checkpoint.get('epoch') != record['epoch']-1
            or checkpoint.get('global_epoch') != record['global_epoch']
            or checkpoint.get('native_correct') != record['native_correct']
            or checkpoint.get('native_total') != 18000 or checkpoint.get('inference_rule') != INFERENCE_RULE
            or checkpoint.get('args') != config or not isinstance(checkpoint.get('model'), dict)
            or any(key.startswith(('projector.', 'selector.')) for key in checkpoint['model'])
            or checkpoint.get('stage_successful_updates') != record['epoch']*240
            or checkpoint.get('global_successful_updates') != 9600+record['epoch']*240):
        raise ValueError('Require the complete selected stage2 NativeUtilitySelector saved directly at x')
    return checkpoint


def freeze(root, authorized):
    if not authorized:
        raise ValueError('Parent must explicitly use --authorize-test after actual paired terminal0')
    if (root/'final_test_started.json').exists() or (root/'final_test_lock.json').exists():
        raise ValueError('Preserve existing freeze and exclusive TEST marker; no replay/reselection')
    protocol, candidates, records, official = completed_training(root)
    frozen = {}
    for arm, candidate in records.items():
        directory = root/RUN_NAMES[arm]
        config = read(directory/'args.json')
        if config != dict(protocol, arm=arm):
            raise ValueError('Frozen arm args must match the unchanged training protocol')
        published = Path(candidate['path']).resolve()
        if file_hash(published) != candidate['sha256']:
            raise ValueError('Published best checkpoint identity changed')
        checkpoint = read_selector_checkpoint(published, candidate, config)
        digest = state_hash(checkpoint['model'])
        unique = []
        paths = [directory/'best_checkpoints'/f'epoch{epoch:03d}.pt'
                 for epoch in (candidate['epoch'], candidate['global_epoch'])]
        paths += [Path(candidate[key]) for key in ('unique_checkpoint_path', 'unique_bundle_path') if key in candidate]
        for path in dict.fromkeys(path.resolve() for path in paths):
            if not path.exists() or path == published:
                continue
            saved = read_selector_checkpoint(path, candidate, config)
            if state_hash(saved['model']) != digest:
                raise ValueError('Existing unique selected-epoch copy differs from published best')
            unique.append(dict(path=str(path), sha256=file_hash(path)))
        inference = unique[0] if unique else dict(path=str(published), sha256=candidate['sha256'])
        frozen[arm] = dict(candidate, path=str(published), model_state_sha256=digest,
            inference_path=inference['path'], inference_sha256=inference['sha256'],
            unique_checkpoints=unique, model_class='NativeUtilitySelector', stage=2,
            checkpoint_mode='eval', optimizer_mode_transition=False)
    lock = dict(authorized=True, frozen_utc=now(), official_arm=official, models=frozen,
        inference_rule=INFERENCE_RULE, selection='max integer VAL correct; earliest epoch exacttie; armtie A',
        original_full18=True, problems=list(PROBLEMS), rows_per_task=1000, native_total=18000,
        full_legal_pools={task: protocol['data_manifest']['train'][task]['pool_ids'] for task in PROBLEMS},
        selector_forwards_per_query=1, final_completed_solvers_per_query=1, projector_included=False,
        parameter_mode='averaged x', tests_per_frozen_model=1, test_batch_size=128, tails_retained=True,
        training_terminal_receipt_sha256=file_hash(root/'training_terminal_receipt.json'),
        validation_candidates_sha256=file_hash(root/'validation_candidates.json'),
        protocol_sha256=file_hash(root/'protocol.json'), delivery_source_hashes=delivery_hashes(),
        test_reselection=False, costs_select_checkpoint=False, task_wise_assembly=False)
    with (root/'final_test_lock.json').open('x') as stream:
        json.dump(lock, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(frozen=True, official_arm=official, models=frozen, test_executed=False)), flush=True)


def validate_lock(root):
    protocol, candidates, records, official = completed_training(root)
    lock = read(root/'final_test_lock.json')
    if (lock.get('authorized') is not True or lock.get('official_arm') != official
            or set(lock.get('models', {})) != {'A', 'B'} or lock.get('inference_rule') != INFERENCE_RULE
            or lock.get('problems') != list(PROBLEMS) or lock.get('original_full18') is not True
            or lock.get('native_total') != 18000 or lock.get('rows_per_task') != 1000
            or lock.get('full_legal_pools') != {p: protocol['data_manifest']['train'][p]['pool_ids'] for p in PROBLEMS}
            or lock.get('selector_forwards_per_query') != 1 or lock.get('final_completed_solvers_per_query') != 1
            or lock.get('projector_included') is not False or lock.get('parameter_mode') != 'averaged x'
            or lock.get('tests_per_frozen_model') != 1 or lock.get('test_batch_size') != 128
            or lock.get('tails_retained') is not True or lock.get('test_reselection') is not False
            or lock.get('costs_select_checkpoint') is not False or lock.get('task_wise_assembly') is not False
            or lock.get('delivery_source_hashes') != delivery_hashes()):
        raise ValueError('Require the exact parent-frozen one-selector full18 deployment lock')
    for key, name in (('training_terminal_receipt_sha256', 'training_terminal_receipt.json'),
                      ('validation_candidates_sha256', 'validation_candidates.json'), ('protocol_sha256', 'protocol.json')):
        if lock[key] != file_hash(root/name):
            raise ValueError('Frozen completion/selection/protocol identity changed')
    for arm, candidate in records.items():
        record = lock['models'][arm]
        if any(record.get(key) != value for key, value in candidate.items()):
            raise ValueError('Lock must retain every original published validation candidate field')
        if record['stage'] != 2 or record['model_class'] != 'NativeUtilitySelector' or record['optimizer_mode_transition'] is not False:
            raise ValueError('No stage1/projector/composite/optimizer-mode deployment is permitted')
        copies = [dict(path=record['path'], sha256=record['sha256']), *record['unique_checkpoints']]
        if dict(path=record['inference_path'], sha256=record['inference_sha256']) not in copies:
            raise ValueError('Inference path must be an identical frozen published/unique selector')
        for copy in copies:
            path = Path(copy['path']).resolve()
            if not path.is_relative_to(root/RUN_NAMES[arm]) or file_hash(path) != copy['sha256']:
                raise ValueError('Frozen checkpoint path or file identity changed')
            checkpoint = read_selector_checkpoint(path, record, read(root/RUN_NAMES[arm]/'args.json'))
            if state_hash(checkpoint['model']) != record['model_state_sha256']:
                raise ValueError('Frozen complete selector state identity changed')
    return protocol, lock


def load_selector(record, config, device):
    checkpoint = read_selector_checkpoint(record['inference_path'], record, config)
    if file_hash(record['inference_path']) != record['inference_sha256'] or state_hash(checkpoint['model']) != record['model_state_sha256']:
        raise ValueError('Locked inference checkpoint changed')
    model = NativeUtilitySelector(**checkpoint['args']['model_params'])
    model.load_state_dict(checkpoint['model'], strict=True)
    model.to(device).eval()
    if any(parameter.dtype != torch.float32 for parameter in model.parameters()):
        raise ValueError('TEST uses one complete FP32 native selector directly, with no optimizer')
    return model


def final_test(root, device):
    protocol, lock = validate_lock(root)
    marker = dict(started_utc=now(), started_unix=time.time(), lock=lock, allowed_attempts=1,
        evaluation_module_sha256=file_hash(Path(__file__)), host=socket.gethostname(), pid=os.getpid(),
        slurm_step=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'], tmux_session='r67_final_test',
        gpu_uuid=GPU_UUID, batch_size=128, selector_forwards_per_query=1, optimizer_mode_transition=False,
        cached_test_scores=False, separate_latency_benchmark=False)
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
                raise ValueError('TEST requires original1000 rows/task, FP64 costs and complete legal pools')
            loader = make_loader(task, 'test', 128, 0, shuffle=False, cache_device=device)
            loader.drop_last = False
            if loader.size != 1000:
                raise ValueError('Original TEST input coverage changed')
            np.testing.assert_array_equal(loader.batch['ind'].cpu(), record['winner'])
            np.testing.assert_array_equal(loader.batch['pool_ids'].cpu(), record['pool_ids'])
            np.testing.assert_array_equal(loader.batch['costs'].cpu(), record['costs'].astype(np.float32))
            raw[task], loaders[task] = record, loader
            manifest[task] = dict(rows=1000, pool=record['pool'], pool_ids=record['pool_ids'],
                data_hash=record['data_hash'], label_hash=record['label_hash'], costs_dtype='raw FP64',
                ordered_original_rows=True, index_range_inclusive=[0, 999])
        atomic_json(root/'test_data_manifest.json', manifest)
        for arm in ('A', 'B'):
            current = arm
            record = lock['models'][arm]
            atomic_json(root/'test_progress.json', dict(status='evaluating', arm=arm, completed_arms=list(results), recorded_utc=now()))
            model = load_selector(record, read(root/RUN_NAMES[arm]/'args.json'), device)
            calls = []

            def input_boundary(module, inputs):
                batch = inputs[0]
                if not set(batch) <= INPUT_KEYS:
                    raise ValueError('Labels, costs and predictions must never enter model forward')
                calls.append(len(batch['n']))

            hook = model.register_forward_pre_hook(input_boundary)
            synchronize(device)
            before = time.monotonic()
            value = evaluate(model, loaders, raw, batch_size=128,
                             prediction_dir=root/RUN_NAMES[arm]/'test_predictions', timing_path=None)
            synchronize(device)
            elapsed = time.monotonic()-before
            hook.remove()
            if (set(value['per_problem']) != set(PROBLEMS) or value['native_total'] != 18000
                    or any(entry['n'] != 1000 for entry in value['per_problem'].values())
                    or calls != ([128]*7+[104])*18 or len(calls) != 18*math.ceil(1000/128)):
                raise ValueError('Each locked selector must complete exactly one original18 TEST pass including tails')
            value.update(complete=True, evaluation_passes=1, complete_evaluation_seconds=elapsed,
                evaluated_checkpoint_path=record['inference_path'], evaluated_checkpoint_sha256=record['inference_sha256'],
                model_state_sha256=record['model_state_sha256'], parameter_mode='averaged x',
                selector_forwards_per_query=1, forward_batch_calls=len(calls), solver_executions=0,
                deployment_completed_solvers_per_query=1, inference_rule=INFERENCE_RULE,
                batch_size=128, tails_retained=True, optimizer_mode_transition=False, cached_test_scores=False,
                timing_scope='ordinary complete evaluation: raw-input selector, gather/transfer, metrics and prediction saving; no singleton benchmark or solver')
            results[arm] = value
            atomic_json(root/('test_results_'+arm+'.json'), value)
            print('[R67 locked TEST] '+json.dumps(dict(arm=arm, native_correct=value['native_correct'], native_total=18000,
                                                      complete_evaluation_seconds=elapsed)), flush=True)
            del model
            gc.collect()
            torch.cuda.empty_cache()
        atomic_json(root/'test_results.json', dict(models=results, lock=lock, complete=True,
            wall_seconds=time.monotonic()-started, tests_per_frozen_model=1, original_full18=True,
            selector_forwards_per_query=1, deployment_completed_solvers_per_query=1,
            separate_latency_benchmark=False, solver_executions=0, references_reuse_existing_official_tests=True))
        atomic_json(root/'test_progress.json', dict(status='complete', completed_arms=list(results), recorded_utc=now()))
    except BaseException as error:
        atomic_json(root/'final_test_failure.json', dict(recorded_utc=now(), failed_arm=current, completed_arms=list(results),
            exception_type=type(error).__name__, message=str(error), traceback=traceback.format_exc(),
            marker_preserved=True, rerun_authorized=False))
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('freeze', 'test'), required=True)
    parser.add_argument('--authorize-test', action='store_true')
    args = parser.parse_args(argv)
    torch.set_num_threads(1)
    if args.stage == 'freeze':
        freeze(ROOT, args.authorize_test)
        return
    if (args.authorize_test or os.environ.get('R67_PARENT_GPU_BATON') != 'locked-test'
            or os.environ.get('CUDA_VISIBLE_DEVICES') != GPU_UUID or os.environ.get('SLURM_JOB_ID') != '1465'
            or socket.gethostname().split('.')[0] != 'gpu03' or os.environ.get('R67_TMUX_SESSION') != 'r67_final_test'):
        parser.error('Use the explicit parent-authorized existing1465/gpu03/assigned UUID TEST launcher')
    configure_torch()
    final_test(ROOT, 'cuda:0')


if __name__ == '__main__':
    main()

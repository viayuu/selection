"""Future authorized R69 freeze and ONE new B TEST; fixed R68A is JSON reuse only."""

import argparse
import json
import math
import os
import socket
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from code.unified_selector.registry import PROBLEMS
from .multitask_probe import file_hash, state_hash
from .performance_targets import read_raw_costs
from .r59_experiment import GPU_UUID, WORKSPACE, atomic_json
from .r64_experiment import evaluate, synchronize
from .r64_model import INFERENCE_RULE
from .r67_frozen_evaluation import now, read, require_terminal
from .r69_experiment import CONTROL_ROOT, ROOT, RUN_NAME
from .r69_model import INPUT_KEYS, PARAMETER_MODE, make_bank
from .train import configure_torch, make_loader


TRAIN_STEP = '1465.101'
NEW_TESTS_REQUIRED = {'A': False, 'B': True}
CONTROL_RUN = CONTROL_ROOT/'A_plain_adamw_seed2'


def delivery_hashes():
    paths = [WORKSPACE/'code/V4'/name for name in
             ('r69_frozen_evaluation.py', 'run_v4_r69_test.sh', 'r69_analysis.py')]
    paths.append(ROOT/'DELIVERY_PROTOCOL.md')
    return {str(path.relative_to(WORKSPACE)): file_hash(path) for path in paths}


def require_scheduler_terminal(receipt, session, expected_step):
    require_terminal(receipt, session, expected_step)
    elapsed = receipt.get('elapsed_seconds')
    scheduler, tmux = receipt.get('scheduler_evidence', {}), receipt.get('tmux_evidence', {})
    if (type(elapsed) not in (int, float) or not math.isfinite(elapsed) or elapsed < 0
            or not isinstance(scheduler, dict) or scheduler.get('returncode') != 0
            or not isinstance(tmux, dict) or tmux.get('returncode') != 1
            or 'tmux has-session' not in tmux.get('command', '')
            or session not in tmux.get('command', '')
            or not any(text in tmux.get('stderr', '') for text in ('can\'t find session', 'no server running', 'no sessions'))):
        raise ValueError('Require original scheduler elapsed/evidence and owned tmux-gone evidence, not process exit alone')
    lines = [line.strip().split('|') for line in scheduler.get('stdout', '').splitlines() if line.strip()]
    matching = [row for row in lines if len(row) >= 7 and row[0] == expected_step]
    if (len(matching) != 1 or matching[0][1:3] != ['COMPLETED', '0:0']
            or matching[0][6].split('.')[0] != 'gpu03' or float(matching[0][3]) != elapsed
            or expected_step not in scheduler.get('command', '')):
        raise ValueError('Original owned Slurm step must actually be COMPLETED/0:0 on gpu03; RUNNING/0:0 is insufficient')


def require_evaluation(value, rows_per_task):
    if (set(value['per_problem']) != set(PROBLEMS)
            or type(value['native_correct']) is not int
            or value['native_total'] != 18*rows_per_task
            or any(entry['n'] != rows_per_task or type(entry.get('native_correct')) is not int
                   or not 0 <= entry['native_correct'] <= rows_per_task
                   for entry in value['per_problem'].values())
            or value['native_correct'] != sum(entry['native_correct'] for entry in value['per_problem'].values())
            or not math.isclose(value['macro']['macro_top1'], value['native_correct']/value['native_total'], abs_tol=1e-12)):
        raise ValueError('Require complete original equal18-task metrics and saved integer native counts')


def reused_control():
    lock_path, test_path = CONTROL_ROOT/'final_test_lock.json', CONTROL_ROOT/'test_results.json'
    lock, test = read(lock_path), read(test_path)
    record, value = lock['models']['A'], test['models']['A']
    if (record['epoch'] != 28 or record['global_epoch'] != 68
            or record['native_correct'] != 8644 or record['native_total'] != 18000
            or record['parameter_mode'] != PARAMETER_MODE or lock['official_arm'] != 'A'
            or record['model_class'] != 'NativeUtilitySelector'
            or any(record.get(key) is not False for key in ('projector_included', 'perturbed_weights',
                                                           'parameter_averaging', 'optimizer_mode_transition'))
            or test.get('complete') is not True or test['lock'] != lock
            or value.get('complete') is not True or value['evaluation_passes'] != 1
            or value['evaluated_checkpoint_sha256'] != record['inference_sha256']
            or value['model_state_sha256'] != record['model_state_sha256']):
        raise ValueError('Reuse only the existing locked R68A28/global68/8644 identity and its unchanged saved TEST')
    require_evaluation(value, 1000)
    provenance = dict(label='reused R68A shared control', lock_path=str(lock_path),
        lock_sha256=file_hash(lock_path), lock_json_pointer='/models/A',
        test_path=str(test_path), test_sha256=file_hash(test_path), test_json_pointer='/models/A',
        result_path=str(CONTROL_RUN/'result.json'), result_sha256=file_hash(CONTROL_RUN/'result.json'),
        history_path=str(CONTROL_RUN/'history.json'), history_sha256=file_hash(CONTROL_RUN/'history.json'),
        original_tests_already_executed=1, new_R69_tests=0, new_R69_training=False,
        weights_loaded=False, inference=False, recount=False, reselection=False,
        historical_test_exposure=True, reuse_is_protocol_equivalence_not_independent_replication=True)
    return record, value, provenance


def completed_training(root):
    if root.resolve() != ROOT.resolve():
        raise ValueError('Use only the canonical R69 run root')
    terminal, launch = read(root/'training_terminal_receipt.json'), read(root/'launch_receipt.json')
    require_scheduler_terminal(terminal, 'r69_train', TRAIN_STEP)
    if launch['slurm_step'] != TRAIN_STEP or launch['gpu_uuid'] != GPU_UUID or launch['gpu_index'] != 1:
        raise ValueError('Preserve the original R69 trainer/allocation/UUID identity')
    protocol, candidates = read(root/'protocol.json'), read(root/'validation_candidates.json')
    complete = read(root/'training_complete.json')
    directory = root/RUN_NAME
    result, history, launches = read(directory/'result.json'), read(directory/'history.json'), read(directory/'wall_time.json')
    if (read(root/'source_launch/hashes.json') != protocol['source_hashes']
            or protocol['parameter_mode'] != PARAMETER_MODE
            or protocol['new_arms'] != {'B': RUN_NAME} or candidates['new_arms'] != ['B']
            or candidates['reused_arms'] != ['A'] or set(candidates['models']) != {'A', 'B'}
            or any(protocol[key] is not False for key in ('sam', 'projector_included', 'parameter_averaging'))
            or protocol['inference_rule'] != INFERENCE_RULE or candidates['inference_rule'] != INFERENCE_RULE
            or complete['new_arms'] != ['B'] or complete['epochs'] != 40
            or (complete['new_updates'], complete['new_presentations'], complete['successful_view_rows']) != (9600, 7200000, 14400000)
            or (complete['inherited_stage1_updates_once'], complete['inherited_stage1_presentations_once']) != (9600, 7200000)
            or len(history) != 40 or [row['epoch'] for row in history] != list(range(1, 41))
            or [row['global_epoch'] for row in history] != list(range(41, 81))
            or (result['epochs_completed'], result['successful_updates'], result['presentations'],
                result['successful_view_rows']) != (40, 9600, 7200000, 14400000)
            or (result['actual_execution']['recorded_successful_optimizer_steps'],
                result['actual_execution']['recorded_query_presentations']) != (9600, 7200000)
            or result['actual_execution']['recorded_amp_retries'] != result['amp_retries']
            or result['attempted_view_rows'] < 14400000 or result['deployment_parameter_mode'] != PARAMETER_MODE
            or result['all18_common_epoch'] is not True or result['new_arms'] != ['B']
            or result['parameter_counts']['B'] != 33173154 or result['parameter_counts']['shared'] != 1842953
            or set(result['updates_by_problem']) != set(PROBLEMS) or set(result['updates_by_problem'].values()) != {1600}
            or set(result['rows_by_problem']) != set(PROBLEMS) or set(result['rows_by_problem'].values()) != {400000}
            or not launches or any(row['status'] != 'complete' or row['slurm_step'] != TRAIN_STEP
                                   or row['host'].split('.')[0] != 'gpu03' for row in launches)):
        raise ValueError('Require actual completed one-new-B fixed40E/9600updates/7.2Mrows and all40 published VAL epochs')
    for row in history:
        require_evaluation(row['val'], 1000)
        if (row['parameter_mode'] != PARAMETER_MODE or row['all18_common_epoch'] is not True
                or row['successful_updates'] != row['epoch']*240 or row['presentations'] != row['epoch']*180000
                or row['successful_view_rows'] != 2*row['presentations']
                or row['attempted_view_rows'] < row['successful_view_rows']):
            raise ValueError('Full18 VAL and cumulative schedule/bundle counters must agree at every common epoch')
        if row['epoch'] % 5 == 0:
            require_evaluation(row['train'], 10000)
    selected = max(history, key=lambda row: row['val']['native_correct'])
    candidate = candidates['models']['B']
    unique = directory/f'best_checkpoints/epoch{selected["epoch"]:03d}.pt'
    expected = dict(path=str(directory/'best.pt'), epoch=selected['epoch'], global_epoch=selected['global_epoch'],
        selected_epoch_checkpoint=str(unique), native_correct=selected['val']['native_correct'], native_total=18000,
        val=selected['val']['macro'], parameter_mode=PARAMETER_MODE, model_type='complete_task_specific_bank',
        all18_common_epoch=True)
    if (any(candidate.get(key) != value for key, value in expected.items())
            or result['best']['epoch'] != selected['epoch'] or result['best']['global_epoch'] != selected['global_epoch']
            or result['best']['val'] != selected['val'] or result['final'] != history[-1]
            or result['selected_epoch_checkpoint'] != str(unique)
            or result['best_checkpoint_sha256'] != candidate['sha256']):
        raise ValueError('B best/result/unique checkpoint must agree with earliest maximum INTEGER full18 VAL count')
    require_evaluation(result['best']['train'], 10000)
    control, _, _ = reused_control()
    for key, value in candidates['models']['A'].items():
        if control.get(key) != value:
            raise ValueError('The reused candidate must retain fixed R68A identity without reselection')
    official = 'B' if candidate['native_correct'] > 8644 else 'A'
    if candidates['proposed_official_arm'] != official:
        raise ValueError('Official B requires strictly>8644 VAL correct; tie/lower fixed A')
    args = read(directory/'args.json')
    if args != dict(protocol, arm='B', model_type='complete_task_specific_bank'):
        raise ValueError('Published B args must retain its exact frozen training protocol')
    return protocol, candidates, candidate, official


def read_bundle(path, record, args):
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    state = checkpoint.get('model')
    if (not isinstance(state, dict) or checkpoint.get('args') != args
            or checkpoint.get('epoch') != record['epoch']-1 or checkpoint.get('common_epoch') != record['epoch']
            or checkpoint.get('global_epoch') != record['global_epoch'] or record['global_epoch'] != 40+record['epoch']
            or checkpoint.get('branch_epochs') != [record['epoch']]*18
            or checkpoint.get('problem_names') != list(PROBLEMS)
            or checkpoint.get('all18_complete_branches') is not True
            or checkpoint.get('parameter_mode') != PARAMETER_MODE or checkpoint.get('checkpoint_mode') != 'eval'
            or checkpoint.get('projector_included') is not False
            or checkpoint.get('inference_rule') != INFERENCE_RULE
            or checkpoint.get('native_correct') != record['native_correct'] or checkpoint.get('native_total') != 18000
            or checkpoint.get('macro') != record['val']
            or checkpoint.get('successful_updates') != record['epoch']*240
            or checkpoint.get('presentations') != record['epoch']*180000
            or any(args[key] is not False for key in ('sam', 'projector_included', 'parameter_averaging'))):
        raise ValueError('Require the complete ordinary-weight R69 bank and its one-common-epoch saved metadata')
    suffixes = []
    for index in range(18):
        prefix = f'branches.{index}.'
        suffixes.append({key[len(prefix):] for key in state if key.startswith(prefix)})
    allowed = {f'branches.{index}.{suffix}' for index, names in enumerate(suffixes) for suffix in names}
    if (not suffixes[0] or any(names != suffixes[0] for names in suffixes)
            or set(state) != allowed or any('projector' in key for key in state)
            or checkpoint.get('model_state_sha256') != state_hash(state)):
        raise ValueError('All18 complete branch states must be present in original order, without assembly/projector')
    return checkpoint


def freeze(root, authorized):
    if not authorized:
        raise ValueError('Future explicit parent freeze authorization requires --authorize-test')
    if any((root/name).exists() for name in ('final_test_lock.json', 'final_test_started.json')):
        raise ValueError('Preserve existing lock/TEST marker; no reselection or retry')
    protocol, _, candidate, official = completed_training(root)
    control, _, provenance = reused_control()
    args = read(root/RUN_NAME/'args.json')
    paths = [Path(candidate[key]) for key in ('path', 'selected_epoch_checkpoint')]
    states = []
    for path in paths:
        if not path.resolve().is_relative_to(root/RUN_NAME) or file_hash(path) != candidate['sha256']:
            raise ValueError('Published B best and its selected unique bundle must have identical file identity')
        checkpoint = read_bundle(path, candidate, args)
        states.append(checkpoint['model_state_sha256'])
        del checkpoint
    if len(set(states)) != 1:
        raise ValueError('Published and unique complete B states must agree')
    bank = dict(candidate, inference_path=candidate['selected_epoch_checkpoint'],
        inference_sha256=candidate['sha256'], model_state_sha256=states[0],
        model_class='TaskSpecificSelector', factory='code.V4.r69_model.make_bank',
        problem_names=list(PROBLEMS), common_epoch=candidate['epoch'], branch_epochs=[candidate['epoch']]*18,
        checkpoint_mode='eval', parameter_averaging=False, perturbed_weights=False,
        optimizer_mode_transition=False, projector_included=False)
    lock = dict(authorized=True, frozen_utc=now(), official_arm=official, models={'A': control, 'B': bank},
        control_reuse=provenance, new_tests_required=NEW_TESTS_REQUIRED,
        new_test_attempts_allowed={'A': 0, 'B': 1}, historical_original_tests_reused={'A': 1},
        selection='B earliest maximum INTEGER full18 VAL count; official B strictly>8644, tie/lower fixed A',
        original_full18=True, problems=list(PROBLEMS), rows_per_task=1000, native_total=18000,
        full_legal_pools={task: protocol['data_manifest']['train'][task]['pool_ids'] for task in PROBLEMS},
        inference_rule=INFERENCE_RULE, test_batch_size=128, tail_rows=104,
        selector_forwards_per_query=1, deployment_completed_solvers_per_query=1, evaluation_solver_executions=0,
        parameter_mode=PARAMETER_MODE, parameter_averaging=False, perturbed_weights=False,
        optimizer_mode_transition=False, projector_included=False, task_wise_assembly=False,
        test_reselection=False, costs_select_checkpoint=False, inherited_initialization=protocol['initialization'],
        schedule_identity=protocol['schedule_identity'], new_arms=['B'], reused_arms=['A'],
        new_training_updates=9600, new_selected_rows=7200000, successful_view_rows=14400000,
        capacity_matched=False, per_parameter_optimizer_histories_matched=False,
        training_terminal_receipt_sha256=file_hash(root/'training_terminal_receipt.json'),
        protocol_sha256=file_hash(root/'protocol.json'), args_sha256=file_hash(root/RUN_NAME/'args.json'),
        validation_candidates_sha256=file_hash(root/'validation_candidates.json'),
        delivery_source_hashes=delivery_hashes())
    with (root/'final_test_lock.json').open('x') as stream:
        json.dump(lock, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(frozen=True, official_arm=official, selected_B_epoch=bank['epoch'],
        new_tests_required=NEW_TESTS_REQUIRED, test_executed=False)), flush=True)


def validate_lock(root, check_checkpoint_files=False):
    protocol, _, candidate, official = completed_training(root)
    lock = read(root/'final_test_lock.json')
    control, _, provenance = reused_control()
    if (lock.get('authorized') is not True or lock['official_arm'] != official
            or lock['models']['A'] != control or lock['control_reuse'] != provenance
            or lock['new_tests_required'] != NEW_TESTS_REQUIRED or lock['new_test_attempts_allowed'] != {'A': 0, 'B': 1}
            or lock['historical_original_tests_reused'] != {'A': 1}
            or lock['new_arms'] != ['B'] or lock['reused_arms'] != ['A']
            or lock['problems'] != list(PROBLEMS) or lock['original_full18'] is not True
            or lock['native_total'] != 18000 or lock['rows_per_task'] != 1000
            or lock['test_batch_size'] != 128 or lock['tail_rows'] != 104
            or lock['full_legal_pools'] != {task: protocol['data_manifest']['train'][task]['pool_ids'] for task in PROBLEMS}
            or lock['inference_rule'] != INFERENCE_RULE or lock['parameter_mode'] != PARAMETER_MODE
            or lock['selector_forwards_per_query'] != 1 or lock['deployment_completed_solvers_per_query'] != 1
            or lock['evaluation_solver_executions'] != 0
            or any(lock[key] is not False for key in ('parameter_averaging', 'perturbed_weights',
                'optimizer_mode_transition', 'projector_included', 'task_wise_assembly', 'test_reselection', 'costs_select_checkpoint'))
            or lock['inherited_initialization'] != protocol['initialization']
            or lock['schedule_identity'] != protocol['schedule_identity'] or lock['delivery_source_hashes'] != delivery_hashes()):
        raise ValueError('Require the unchanged parent-frozen one-new-B/reused-A original18 lock')
    for key, path in (('training_terminal_receipt_sha256', root/'training_terminal_receipt.json'),
                      ('protocol_sha256', root/'protocol.json'), ('args_sha256', root/RUN_NAME/'args.json'),
                      ('validation_candidates_sha256', root/'validation_candidates.json')):
        if lock[key] != file_hash(path):
            raise ValueError('Frozen parent evidence/protocol/candidate identity changed')
    bank = lock['models']['B']
    if (any(bank.get(key) != value for key, value in candidate.items())
            or bank['model_class'] != 'TaskSpecificSelector' or bank['factory'] != 'code.V4.r69_model.make_bank'
            or bank['problem_names'] != list(PROBLEMS) or bank['common_epoch'] != candidate['epoch']
            or bank['branch_epochs'] != [candidate['epoch']]*18
            or bank['inference_path'] != candidate['selected_epoch_checkpoint']
            or bank['inference_sha256'] != candidate['sha256']
            or any(bank[key] is not False for key in ('parameter_averaging', 'perturbed_weights',
                                                     'optimizer_mode_transition', 'projector_included'))):
        raise ValueError('Freeze one complete B bundle with unchanged ordinary-weight deployment metadata')
    if check_checkpoint_files:
        for key in ('path', 'inference_path'):
            path = Path(bank[key]).resolve()
            if not path.is_relative_to(root/RUN_NAME) or file_hash(path) != bank['inference_sha256']:
                raise ValueError('Locked B checkpoint bytes/path changed')
    return protocol, lock


def load_bank(root, record, config, device):
    checkpoint = read_bundle(Path(record['inference_path']), record, config)
    if checkpoint['model_state_sha256'] != record['model_state_sha256']:
        raise ValueError('Locked complete B bank state changed')
    # The factory requires a construction state; the strict load replaces the ENTIRE bank before any forward.
    initial = torch.load(root/'shared_initialization.pt', map_location='cpu', weights_only=False)['selector']
    model = make_bank(config['model_params'], initial)
    model.load_state_dict(checkpoint['model'], strict=True)
    model.to(device).eval()
    if sum(p.numel() for p in model.parameters()) != 33173154 or any(p.dtype != torch.float32 for p in model.parameters()):
        raise ValueError('TEST requires all18 complete FP32 ordinary selectors, loaded directly without transition')
    return model


def final_test(root, device):
    protocol, lock = validate_lock(root, check_checkpoint_files=True)
    _, control_metrics, _ = reused_control()
    marker = dict(started_utc=now(), started_unix=time.time(), lock=lock, allowed_attempts=1,
        models_to_evaluate=['B'], new_tests_required=NEW_TESTS_REQUIRED,
        historical_original_A_tests_reused=1, host=socket.gethostname(), pid=os.getpid(),
        slurm_step=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'], tmux_session='r69_final_test',
        tmux_pane=os.environ['R69_TEST_TMUX_PANE_ID'], tmux_pane_pid=os.environ['R69_TEST_TMUX_PANE_PID'],
        gpu_uuid=GPU_UUID, batch_size=128, parent_baton='locked-test',
        evaluation_module_sha256=file_hash(Path(__file__)), automatic_retry=False)
    with (root/'final_test_started.json').open('x') as stream:
        json.dump(marker, stream, indent=2, allow_nan=False)
        stream.write('\n')
    started = time.monotonic()
    try:
        raw, loaders, manifest = {}, {}, {}
        for task in PROBLEMS:
            value = read_raw_costs(task, 'test')
            original = protocol['data_manifest']['train'][task]
            if (len(value['winner']) != 1000 or value['costs'].dtype != np.float64
                    or any(value[key] != original[key] for key in ('pool', 'pool_ids'))):
                raise ValueError('Require original full1000 TEST rows/task, raw FP64 costs and unchanged legal pool order')
            loader = make_loader(task, 'test', 128, 0, shuffle=False, cache_device=device)
            loader.drop_last = False
            if loader.size != 1000:
                raise ValueError('Original TEST input coverage changed')
            np.testing.assert_array_equal(loader.batch['ind'].cpu(), value['winner'])
            np.testing.assert_array_equal(loader.batch['pool_ids'].cpu(), value['pool_ids'])
            np.testing.assert_array_equal(loader.batch['costs'].cpu(), value['costs'].astype(np.float32))
            raw[task], loaders[task] = value, loader
            manifest[task] = dict(rows=1000, pool=value['pool'], pool_ids=value['pool_ids'],
                data_hash=value['data_hash'], label_hash=value['label_hash'], costs_dtype='raw FP64',
                index_range_inclusive=[0, 999], ordered_original_rows=True)
        atomic_json(root/'test_data_manifest.json', manifest)
        atomic_json(root/'test_progress.json', dict(status='evaluating', new_arm='B', reused_arm='A', recorded_utc=now()))
        model = load_bank(root, lock['models']['B'], read(root/RUN_NAME/'args.json'), device)
        calls, branch_calls = [], []

        def input_boundary(module, inputs):
            batch = inputs[0]
            if not set(batch) <= INPUT_KEYS:
                raise ValueError('Labels/costs/predictions must stay outside selector forward')
            branch_id = model.branch_index(batch['problem_id'])
            if batch['pool_ids'].detach().cpu().tolist() != lock['full_legal_pools'][PROBLEMS[branch_id]]:
                raise ValueError('Retain original full-pool candidate order and global solver IDs')
            calls.append((branch_id, len(batch['n'])))

        def branch_boundary(index):
            def observe(module, inputs):
                batch = inputs[0]
                if model.branch_index(batch['problem_id']) != index or not set(batch) <= INPUT_KEYS:
                    raise ValueError('Only known problem identity may activate its one complete branch')
                branch_calls.append((index, len(batch['n'])))
            return observe

        hooks = [model.register_forward_pre_hook(input_boundary)]
        hooks += [branch.register_forward_pre_hook(branch_boundary(index)) for index, branch in enumerate(model.branches)]
        synchronize(device)
        before = time.monotonic()
        value = evaluate(model, loaders, raw, batch_size=128, prediction_dir=root/RUN_NAME/'test_predictions', timing_path=None)
        synchronize(device)
        elapsed = time.monotonic()-before
        for hook in hooks:
            hook.remove()
        require_evaluation(value, 1000)
        expected = [(index, rows) for index in range(18) for rows in [128]*7+[104]]
        if calls != expected or branch_calls != expected:
            raise ValueError('ONE routed full-pool selector forward/query; exactly144 batches including104-row tails')
        record = lock['models']['B']
        value.update(complete=True, evaluation_passes=1, evaluation_origin='new R69 B TEST',
            complete_evaluation_seconds=elapsed, evaluated_checkpoint_path=record['inference_path'],
            evaluated_checkpoint_sha256=record['inference_sha256'], model_state_sha256=record['model_state_sha256'],
            parameter_mode=PARAMETER_MODE, parameter_averaging=False, perturbed_weights=False,
            optimizer_mode_transition=False, projector_included=False, all18_common_epoch=True,
            common_epoch=record['common_epoch'], problem_names=list(PROBLEMS),
            selector_forwards_per_query=1, active_branch_forward_calls=len(branch_calls), forward_batch_calls=len(calls),
            solver_executions=0, deployment_completed_solvers_per_query=1, inference_rule=INFERENCE_RULE,
            batch_size=128, tail_rows=104, tails_retained=True, cached_test_scores=False,
            separate_latency_benchmark=False,
            timing_scope='ordinary full18 evaluation walltime including gather/transfer, metrics and NPZ saving; not singleton latency or solver execution')
        atomic_json(root/'test_results_B.json', value)
        atomic_json(root/'test_results.json', dict(models={'A': control_metrics, 'B': value}, lock=lock,
            complete=True, wall_seconds=time.monotonic()-started, original_full18=True,
            new_tests_executed={'A': 0, 'B': 1}, historical_original_tests_reused={'A': 1},
            tests_per_new_model=1, control_reuse=lock['control_reuse'],
            model_execution={'A': 'reused R68A JSON; zero new inference/recount/tests', 'B': 'one new R69 original18 TEST'},
            selector_forwards_per_query=1, deployment_completed_solvers_per_query=1, solver_executions=0,
            parameter_mode=PARAMETER_MODE, parameter_averaging=False, perturbed_weights=False,
            optimizer_mode_transition=False, separate_latency_benchmark=False, prediction_replay=False))
        atomic_json(root/'test_progress.json', dict(status='complete', completed_new_arms=['B'], reused_arms=['A'], recorded_utc=now()))
        print('[R69 one NEW B TEST] '+json.dumps(dict(native_correct=value['native_correct'], native_total=18000,
            complete_evaluation_seconds=elapsed, new_A_tests=0, historical_A_tests_reused=1)), flush=True)
    except BaseException as error:
        atomic_json(root/'final_test_failure.json', dict(recorded_utc=now(), failed_new_arm='B',
            exception_type=type(error).__name__, message=str(error), traceback=traceback.format_exc(),
            marker_preserved=True, automatic_retry=False, reselection=False))
        raise


def completed_test(root):
    protocol, lock = validate_lock(root)
    test, marker = read(root/'test_results.json'), read(root/'final_test_started.json')
    terminal = read(root/'test_terminal_receipt.json')
    require_scheduler_terminal(terminal, 'r69_final_test', marker['slurm_step'])
    process, launch = read(root/'test_process_receipt.json'), read(root/'test_launch_receipt.json')
    _, control_metrics, _ = reused_control()
    if (marker['slurm_step'] == TRAIN_STEP or marker['tmux_session'] != 'r69_final_test'
            or marker['models_to_evaluate'] != ['B'] or marker['allowed_attempts'] != 1
            or marker['lock'] != lock or test['lock'] != lock or test['complete'] is not True
            or test['models']['A'] != control_metrics or test['control_reuse'] != lock['control_reuse']
            or test['new_tests_executed'] != {'A': 0, 'B': 1} or test['historical_original_tests_reused'] != {'A': 1}
            or test['tests_per_new_model'] != 1 or test['original_full18'] is not True
            or process['returncode'] != 0 or process['slurm_step'] != marker['slurm_step']
            or process['tmux_session'] != 'r69_final_test' or process['new_evaluated_arms'] != ['B']
            or process['new_test_attempts_started'] != {'A': 0, 'B': 1}
            or process['new_tests_completed'] != {'A': 0, 'B': 1}
            or launch['slurm_step'] != marker['slurm_step'] or launch['tmux_session'] != 'r69_final_test'
            or launch['gpu_uuid'] != GPU_UUID or launch['gpu_index'] != 1 or launch['idle'] is not True
            or test['solver_executions'] != 0 or test['prediction_replay'] is not False
            or test['separate_latency_benchmark'] is not False):
        raise ValueError('Require completed original new B TEST, unchanged saved A reuse, and separate actual Slurm/tmux terminal0')
    value, record = test['models']['B'], lock['models']['B']
    require_evaluation(value, 1000)
    if (value['complete'] is not True or value['evaluation_passes'] != 1
            or value['forward_batch_calls'] != 144 or value['active_branch_forward_calls'] != 144
            or value['selector_forwards_per_query'] != 1 or value['batch_size'] != 128 or value['tail_rows'] != 104
            or value['evaluated_checkpoint_sha256'] != record['inference_sha256']
            or value['model_state_sha256'] != record['model_state_sha256']
            or value['common_epoch'] != record['common_epoch'] or value['parameter_mode'] != PARAMETER_MODE
            or any(value[key] is not False for key in ('parameter_averaging', 'perturbed_weights',
                'optimizer_mode_transition', 'projector_included', 'cached_test_scores', 'separate_latency_benchmark'))):
        raise ValueError('Require the exact frozen complete ordinary B bank, one live TEST evaluation, no replay or transition')
    return protocol, lock, test, terminal


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('freeze', 'test'), required=True)
    parser.add_argument('--authorize-test', action='store_true')
    args = parser.parse_args(argv)
    torch.set_num_threads(1)
    if args.stage == 'freeze':
        if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
            parser.error('Future parent-authorized freeze is CPU-only: CUDA_VISIBLE_DEVICES=')
        freeze(ROOT, args.authorize_test)
        return
    if (args.authorize_test or os.environ.get('R69_PARENT_GPU_BATON') != 'locked-test'
            or os.environ.get('CUDA_VISIBLE_DEVICES') != GPU_UUID or os.environ.get('SLURM_JOB_ID') != '1465'
            or socket.gethostname().split('.')[0] != 'gpu03'
            or os.environ.get('R69_TMUX_SESSION') != 'r69_final_test'
            or not os.environ.get('R69_TEST_TMUX_PANE_ID')):
        parser.error('Use future explicit parent locked-test baton and owned r69_final_test assigned1465/gpu03/UUID launcher')
    configure_torch()
    final_test(ROOT, 'cuda:0')


if __name__ == '__main__':
    main()

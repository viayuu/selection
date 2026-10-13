"""Independent CPU controller and conditionally authorized ONE R71 B TEST."""

import argparse
import csv
import json
import os
import shlex
import socket
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

from code.unified_selector.registry import PROBLEMS
from .multitask_probe import file_hash
from .r59_experiment import GPU_UUID, WORKSPACE, atomic_json
from .r69_frozen_evaluation import require_evaluation, require_scheduler_terminal
from .r69_model import INPUT_KEYS, PARAMETER_MODE
from .r71_experiment import CONTROL_ROOT, ROOT, RUN_NAME, append_jsonl
from .r71_model import INFERENCE_RULE


TRAIN_STEP = '1465.105'
CONTROL_RUN = CONTROL_ROOT / 'B_latent_mixup_seed2'
DELIVERY_FILES = ('r71_frozen_evaluation.py', 'run_v4_r71_test.sh', 'r71_analysis.py')
NEW_TESTS_REQUIRED = {'A': False, 'B': True}


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def exclusive_json(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def delivery_hashes():
    paths = [WORKSPACE / 'code/V4' / name for name in DELIVERY_FILES] + [ROOT / 'DELIVERY_PROTOCOL.md']
    return {str(path.relative_to(WORKSPACE)): file_hash(path) for path in paths}


def reused_control():
    lock_path, test_path = CONTROL_ROOT / 'final_test_lock.json', CONTROL_ROOT / 'test_results.json'
    old_lock, old_test = read(lock_path), read(test_path)
    record, value = old_lock['models']['B'], old_test['models']['B']
    candidate = read(ROOT / 'reused_control.json')['model']
    if (any(record.get(key) != item for key, item in candidate.items())
            or (record['epoch'], record['global_epoch'], record['native_correct'], record['native_total']) != (23, 63, 8709, 18000)
            or record['all18_common_epoch'] is not True or record['model_class'] != 'TaskSpecificSelector'
            or old_test['complete'] is not True or old_test['lock'] != old_lock
            or value['complete'] is not True or value['evaluation_passes'] != 1
            or value['native_total'] != 18000
            or value['evaluated_checkpoint_sha256'] != record['inference_sha256']
            or value['model_state_sha256'] != record['model_state_sha256']):
        raise ValueError('Reuse only the exact existing locked R70B23 complete-bank TEST object')
    provenance = dict(label='fixed R71 A = historical R70B23/global63', lock_path=str(lock_path),
        lock_json_pointer='/models/B', test_path=str(test_path), test_json_pointer='/models/B',
        history_path=str(CONTROL_RUN / 'history.json'), result_path=str(CONTROL_RUN / 'result.json'),
        original_tests_already_executed=1, new_R71_tests=0, new_R71_training=False,
        weights_loaded=False, inference=False, recount=False, reselection=False,
        historical_test_exposure=True, reuse_is_protocol_equivalence_not_independent_replication=True)
    return record, value, provenance


def completed_training():
    require_scheduler_terminal(read(ROOT / 'training_terminal_receipt.json'), 'r71_train', TRAIN_STEP)
    launch = read(ROOT / 'launch_receipt.json')
    protocol, complete = read(ROOT / 'protocol.json'), read(ROOT / 'training_complete.json')
    candidates = read(ROOT / 'validation_candidates.json')
    result, history = read(ROOT / RUN_NAME / 'result.json'), read(ROOT / RUN_NAME / 'history.json')
    if (launch['slurm_step'] != TRAIN_STEP or launch['gpu_uuid'] != GPU_UUID or launch['gpu_index'] != 1
            or protocol['new_arms'] != {'B': RUN_NAME} or protocol['inference_rule'] != INFERENCE_RULE
            or read(ROOT / 'source_launch/hashes.json') != protocol['source_hashes']
            or complete['new_arms'] != ['B'] or complete['epochs'] != 40
            or (complete['new_updates'], complete['new_presentations'], complete['successful_view_rows']) != (9600, 7200000, 14400000)
            or len(history) != 40 or [row['epoch'] for row in history] != list(range(1, 41))
            or [row['global_epoch'] for row in history] != list(range(41, 81))
            or (result['epochs_completed'], result['successful_updates'], result['presentations'], result['successful_view_rows']) != (40, 9600, 7200000, 14400000)
            or result['actual_execution']['recorded_successful_optimizer_steps'] != 9600
            or result['actual_execution']['recorded_query_presentations'] != 7200000
            or result['actual_execution']['recorded_amp_retries'] != result['amp_retries']
            or result['attempted_view_rows'] < 14400000 or result['all18_common_epoch'] is not True
            or result['parameter_counts']['B'] != 38231040 or result['parameter_counts']['A'] != 33173154
            or set(result['updates_by_problem']) != set(PROBLEMS) or set(result['updates_by_problem'].values()) != {1600}
            or set(result['rows_by_problem']) != set(PROBLEMS) or set(result['rows_by_problem'].values()) != {400000}):
        raise ValueError('Require original completed40E/9600updates/7.2Mrows/14.4Mencoder views')
    for row in history:
        require_evaluation(row['val'], 1000)
        if (row['all18_common_epoch'] is not True or row['successful_updates'] != 240 * row['epoch']
                or row['presentations'] != 180000 * row['epoch']
                or row['successful_view_rows'] != 2 * row['presentations']
                or row['attempted_view_rows'] < row['successful_view_rows']):
            raise ValueError('Every common epoch must retain full18 VAL and actual cumulative counters')
        if row['epoch'] % 5 == 0:
            require_evaluation(row['train'], 10000)
        elif row.get('train') is not None:
            raise ValueError('TRAIN measurements must remain sparse')
    selected = max(history, key=lambda row: row['val']['native_correct'])
    candidate = candidates['models']['B']
    unique = ROOT / RUN_NAME / f'best_checkpoints/epoch{selected["epoch"]:03d}.pt'
    expected = dict(epoch=selected['epoch'], global_epoch=selected['global_epoch'],
        native_correct=selected['val']['native_correct'], native_total=18000, val=selected['val']['macro'],
        selected_epoch_checkpoint=str(unique), parameter_mode=PARAMETER_MODE,
        all18_common_epoch=True, model_type='complete_joint_mixture_bank', persistent_inference_buffers=True)
    official = 'B' if candidate['native_correct'] > 8709 else 'A'
    if (any(candidate.get(key) != item for key, item in expected.items())
            or result['best']['epoch'] != selected['epoch'] or result['best']['val'] != selected['val']
            or result['final'] != history[-1] or result['selected_epoch_checkpoint'] != str(unique)
            or result['best_checkpoint_sha256'] != candidate['sha256']
            or candidates['proposed_official_arm'] != official
            or candidates['models']['A'] != read(ROOT / 'reused_control.json')['model']):
        raise ValueError('B must be earliest maximum INTEGER full18 VAL; official B strictly>8709')
    require_evaluation(result['best']['train'], 10000)
    return protocol, candidate, official


def read_bundle(path, record, args):
    import torch
    from .multitask_probe import state_hash

    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    state = checkpoint.get('model')
    if (not isinstance(state, dict) or checkpoint.get('args') != args
            or checkpoint.get('epoch') != record['epoch'] - 1 or checkpoint.get('common_epoch') != record['epoch']
            or checkpoint.get('global_epoch') != record['global_epoch'] or record['global_epoch'] != 40 + record['epoch']
            or checkpoint.get('branch_epochs') != [record['epoch']] * 18
            or checkpoint.get('problem_names') != list(PROBLEMS)
            or checkpoint.get('all18_complete_branches') is not True or checkpoint.get('persistent_inference_buffers') is not True
            or checkpoint.get('parameter_mode') != PARAMETER_MODE or checkpoint.get('checkpoint_mode') != 'eval'
            or checkpoint.get('projector_included') is not False or checkpoint.get('inference_rule') != INFERENCE_RULE
            or checkpoint.get('native_correct') != record['native_correct'] or checkpoint.get('native_total') != 18000
            or checkpoint.get('macro') != record['val'] or checkpoint.get('successful_updates') != record['epoch'] * 240
            or checkpoint.get('presentations') != record['epoch'] * 180000
            or checkpoint.get('model_state_sha256') != state_hash(state)):
        raise ValueError('Require the actual complete common-epoch mixture bank and all persistent inference buffers')
    allowed = set()
    for index, task in enumerate(PROBLEMS):
        prefix = f'branches.{index}.'
        names = {key[len(prefix):] for key in state if key.startswith(prefix)}
        allowed.update(prefix + name for name in names)
        normalizer = args['target_normalizers'][task]
        k, d = len(normalizer['pool_ids']), len(normalizer['mean'])
        head = prefix + 'density_head.'
        if (not names or d != k - 1 or any(name.startswith('score_head.') or 'projector' in name for name in names)
                or state[head + 'normal_vectors'].shape != (2048, d)
                or state[head + 'layers.0.weight'].shape != (128, 384 * k)
                or state[head + 'layers.2.weight'].shape != (8 * (1 + d + d * (d + 1) // 2), 128)
                or not torch.equal(state[head + 'canonical_pool_ids'], torch.tensor(normalizer['pool_ids'], dtype=torch.long))
                or not torch.equal(state[head + 'target_mean'], torch.tensor(normalizer['mean'], dtype=torch.float64))
                or not torch.equal(state[head + 'target_scale'], torch.tensor(normalizer['scale'], dtype=torch.float64))):
            raise ValueError('Preserve each task complete legal pool, density shape and TRAIN-only buffers')
    if set(state) != allowed:
        raise ValueError('Exactly18 complete branches in original problem-ID order are required')
    return checkpoint


def load_bank(record, args, device):
    from .r71_model import make_bank

    checkpoint = read_bundle(Path(record['inference_path']), record, args)
    state = checkpoint['model']
    # The factory needs a trunk-shaped template; strict whole-bank restoration follows.
    prefix = 'branches.0.'
    template = {key[len(prefix):]: value for key, value in state.items()
                if key.startswith(prefix) and not key.startswith(prefix + 'density_head.')}
    bank = make_bank(args['model_params'], template, args['target_normalizers'], device='cpu')
    bank.load_state_dict(state, strict=True)
    if sum(p.numel() for p in bank.parameters()) != 38231040 or any(hasattr(b, 'score_head') for b in bank.branches):
        raise ValueError('Restore the full R71 bank, without scalar heads or alternate readouts')
    return bank.to(device).eval()


def freeze():
    if any((ROOT / name).exists() for name in ('final_test_lock.json', 'final_test_started.json')):
        raise ValueError('Preserve the exclusive existing VAL lock/TEST marker; no reselection or retry')
    protocol, candidate, official = completed_training()
    control, _, provenance = reused_control()
    path = Path(candidate['selected_epoch_checkpoint']).resolve()
    if not path.is_relative_to(ROOT / RUN_NAME) or file_hash(path) != candidate['sha256']:
        raise ValueError('Freeze only the published unique selected complete B bank')
    checkpoint = read_bundle(path, candidate, read(ROOT / RUN_NAME / 'args.json'))
    record = dict(candidate, inference_path=str(path), inference_sha256=candidate['sha256'],
        model_state_sha256=checkpoint['model_state_sha256'], model_class='TaskSpecificSelector[JointMixtureSelector]',
        factory='code.V4.r71_model.make_bank then strict load_state_dict of ALL18 branches and buffers',
        common_epoch=candidate['epoch'], branch_epochs=[candidate['epoch']] * 18, problem_names=list(PROBLEMS),
        checkpoint_mode='eval', parameter_averaging=False, projector_included=False,
        persistent_normalization_order_and_Sobol=True)
    lock = dict(authorized=True, conditional_parent_authorization=True, frozen_utc=now(), official_arm=official,
        models={'A': control, 'B': record}, control_reuse=provenance, new_tests_required=NEW_TESTS_REQUIRED,
        new_test_attempts_allowed={'A': 0, 'B': 1}, historical_original_tests_reused={'A': 1},
        new_arms=['B'], reused_arms=['A'], selection='earliest max INTEGER full18 VAL; official B strictly>8709 else fixed A23',
        original_full18=True, problems=list(PROBLEMS), rows_per_task=1000, native_total=18000,
        full_legal_pools={task: protocol['data_manifest']['train'][task]['pool_ids'] for task in PROBLEMS},
        inference_rule=INFERENCE_RULE, test_batch_size=128, tail_rows=104, selector_forwards_per_query=1,
        predicted_vectors_per_query=16384, streamed_components=1, streamed_vectors=256,
        probability_floor_for_reporting_ce_only=1e-12, deployment_completed_solvers_per_query=1,
        evaluation_solver_executions=0, parameter_mode=PARAMETER_MODE, task_wise_assembly=False,
        test_reselection=False, costs_select_checkpoint=False, NLL_selects_checkpoint=False,
        parameter_capacity_matched=False, FLOPs_matched=False, A_parameters=33173154, B_parameters=38231040,
        training_terminal=read(ROOT / 'training_terminal_receipt.json'), delivery_source_hashes=delivery_hashes())
    exclusive_json(ROOT / 'final_test_lock.json', lock)
    print('[R71 VAL freeze] ' + json.dumps(dict(official_arm=official, B_epoch=record['epoch'],
        B_VAL_correct=record['native_correct'], A_VAL_correct=8709, TEST_executed=False)), flush=True)


def validate_lock():
    protocol, candidate, official = completed_training()
    lock = read(ROOT / 'final_test_lock.json')
    if (lock['authorized'] is not True or lock['official_arm'] != official
            or any(lock['models']['B'].get(key) != value for key, value in candidate.items())
            or lock['models']['B']['inference_path'] != candidate['selected_epoch_checkpoint']
            or lock['new_test_attempts_allowed'] != {'A': 0, 'B': 1} or lock['new_tests_required'] != NEW_TESTS_REQUIRED
            or lock['test_batch_size'] != 128 or lock['tail_rows'] != 104 or lock['inference_rule'] != INFERENCE_RULE
            or lock['delivery_source_hashes'] != delivery_hashes()):
        raise ValueError('Preserve the VAL-selected bank/buffers, probability rule and one-TEST budget')
    return protocol, lock


def gpu_bootstrap():
    if (ROOT / 'final_test_started.json').exists():
        raise ValueError('The exclusive one-new-B TEST marker exists; no retry')
    while True:
        gpu = subprocess.check_output(['nvidia-smi', '--id=' + GPU_UUID,
            '--query-gpu=index,uuid,name,memory.used,memory.total', '--format=csv,noheader,nounits'], text=True)
        apps = subprocess.check_output(['nvidia-smi', '--id=' + GPU_UUID,
            '--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory', '--format=csv,noheader,nounits'], text=True)
        rows = list(csv.reader(gpu.splitlines()))
        if len(rows) != 1:
            raise ValueError('Only the assigned UUID may be queried')
        index, actual, name, used, total = [value.strip() for value in rows[0]]
        if index != '1' or actual != GPU_UUID:
            raise ValueError('Assigned UUID/index mismatch; no device switch')
        clients = [row for row in csv.reader(apps.splitlines()) if row and row[0].strip() == GPU_UUID]
        receipt = dict(checked_utc=now(), host=socket.gethostname(),
            slurm_step=os.environ['SLURM_JOB_ID'] + '.' + os.environ['SLURM_STEP_ID'], gpu_uuid=actual, gpu_index=1,
            gpu_name=name, memory_used_mib=int(used), memory_total_mib=int(total), compute_apps=clients,
            idle=int(used) < 500 and not clients, tmux_session='r71_final_test',
            tmux_pane=os.environ['R71_TEST_TMUX_PANE_ID'], tmux_pane_pid=os.environ['R71_TEST_TMUX_PANE_PID'],
            only_assigned_uuid_queried=True)
        append_jsonl(ROOT / 'test_availability.jsonl', receipt)
        if receipt['idle']:
            atomic_json(ROOT / 'test_launch_receipt.json', receipt)
            return
        print('[R71 TEST wait] assigned UUID occupied; wait30s on the same device', flush=True)
        time.sleep(30)


def final_test():
    import numpy as np
    import torch
    from .performance_targets import read_raw_costs
    from .r64_experiment import synchronize
    from .r71_experiment import evaluate
    from .train import configure_torch, make_loader

    protocol, lock = validate_lock()
    _, control_metrics, provenance = reused_control()
    gpu_bootstrap()
    marker = dict(started_utc=now(), lock=lock, allowed_attempts=1, models_to_evaluate=['B'],
        tmux_session='r71_final_test', tmux_pane=os.environ['R71_TEST_TMUX_PANE_ID'],
        tmux_pane_pid=os.environ['R71_TEST_TMUX_PANE_PID'], host=socket.gethostname(), pid=os.getpid(),
        gpu_uuid=GPU_UUID, slurm_step=os.environ['SLURM_JOB_ID'] + '.' + os.environ['SLURM_STEP_ID'], automatic_retry=False)
    exclusive_json(ROOT / 'final_test_started.json', marker)
    configure_torch()
    torch.set_num_threads(1)
    started = time.monotonic()
    raw, loaders, manifest = {}, {}, {}
    for task in PROBLEMS:
        value = read_raw_costs(task, 'test')
        original = protocol['data_manifest']['train'][task]
        if (len(value['winner']) != 1000 or value['costs'].dtype != np.float64
                or any(value[key] != original[key] for key in ('pool', 'pool_ids'))):
            raise ValueError('Original full18x1000 TEST/raw FP64 costs/full legal pools required')
        loader = make_loader(task, 'test', 128, 0, shuffle=False, cache_device='cuda:0')
        loader.drop_last = False
        if loader.size != 1000:
            raise ValueError('Original TEST input coverage changed')
        np.testing.assert_array_equal(loader.batch['ind'].cpu(), value['winner'])
        np.testing.assert_array_equal(loader.batch['pool_ids'].cpu(), value['pool_ids'])
        raw[task], loaders[task] = value, loader
        manifest[task] = dict(rows=1000, pool=value['pool'], pool_ids=value['pool_ids'],
            data_hash=value['data_hash'], label_hash=value['label_hash'], costs_dtype='raw FP64', ordered_original_rows=True)
    atomic_json(ROOT / 'test_data_manifest.json', manifest)
    model = load_bank(lock['models']['B'], read(ROOT / RUN_NAME / 'args.json'), 'cuda:0')
    calls, branch_calls, encoder_calls = [], [], []

    def input_boundary(module, inputs):
        batch = inputs[0]
        index = model.branch_index(batch['problem_id'])
        if (not set(batch) <= INPUT_KEYS
                or batch['pool_ids'].cpu().tolist() != lock['full_legal_pools'][PROBLEMS[index]]):
            raise ValueError('One full-pool label-free probability inference per query')
        calls.append((index, len(batch['n'])))

    def branch_boundary(index, target):
        def observe(module, inputs):
            batch = inputs[0]
            if model.branch_index(batch['problem_id']) != index or not set(batch) <= INPUT_KEYS:
                raise ValueError('Exactly one original known-problem branch/encoder must activate')
            target.append((index, len(batch['n'])))
        return observe

    hooks = [model.register_forward_pre_hook(input_boundary)]
    for index, branch in enumerate(model.branches):
        hooks += [branch.register_forward_pre_hook(branch_boundary(index, branch_calls)),
                  branch.instance_encoder.register_forward_pre_hook(branch_boundary(index, encoder_calls))]
    synchronize('cuda:0')
    before = time.monotonic()
    value = evaluate(model, loaders, raw, 128, ROOT / RUN_NAME / 'test_predictions')
    synchronize('cuda:0')
    elapsed = time.monotonic() - before
    for hook in hooks:
        hook.remove()
    require_evaluation(value, 1000)
    expected = [(index, rows) for index in range(18) for rows in [128] * 7 + [104]]
    if calls != expected or branch_calls != expected or encoder_calls != expected:
        raise ValueError('Require144 wrapper/active-branch/encoder batches and all104-row tails')
    record = lock['models']['B']
    value.update(complete=True, evaluation_passes=1, evaluation_origin='ONE new R71 B original full18 TEST',
        complete_evaluation_seconds=elapsed, evaluated_checkpoint_path=record['inference_path'],
        evaluated_checkpoint_sha256=record['inference_sha256'], model_state_sha256=record['model_state_sha256'],
        common_epoch=record['common_epoch'], all18_common_epoch=True, parameter_mode=PARAMETER_MODE,
        persistent_inference_buffers=True, selector_forwards_per_query=1, active_branch_forward_calls=len(branch_calls),
        forward_batch_calls=len(calls), encoder_batch_calls=len(encoder_calls), batch_size=128,
        tail_rows=104, tails_retained=True, cached_test_scores=False, predicted_vectors_per_query=16384,
        solver_executions=0, deployment_completed_solvers_per_query=1, inference_rule=INFERENCE_RULE,
        density_NLL_reuses_same_encoded_parameters=True, separate_latency_benchmark=False,
        timing_scope='whole full18 evaluation including probability integration/transfer/metrics/NPZ; not singleton or solver latency')
    atomic_json(ROOT / 'test_results_B.json', value)
    atomic_json(ROOT / 'test_results.json', dict(models={'A': control_metrics, 'B': value}, lock=lock,
        complete=True, wall_seconds=time.monotonic() - started, original_full18=True,
        new_tests_executed={'A': 0, 'B': 1}, historical_original_tests_reused={'A': 1}, tests_per_new_model=1,
        control_reuse=provenance, solver_executions=0, prediction_replay=False,
        model_execution={'A': 'unchanged R70 test_results.json/models/B object; no new inference', 'B': 'ONE new original full18 frozen-probability TEST'}))
    print('[R71 ONE new B TEST complete] ' + json.dumps(dict(native_correct=value['native_correct'],
        native_total=18000, complete_evaluation_seconds=elapsed, new_A_tests=0, new_B_tests=1)), flush=True)


def completed_test():
    protocol, lock = validate_lock()
    test, marker = read(ROOT / 'test_results.json'), read(ROOT / 'final_test_started.json')
    terminal, process = read(ROOT / 'test_terminal_receipt.json'), read(ROOT / 'test_process_receipt.json')
    require_scheduler_terminal(terminal, 'r71_final_test', marker['slurm_step'])
    _, original_A, _ = reused_control()
    value = test['models']['B']
    if (test['complete'] is not True or test['lock'] != lock or marker['lock'] != lock
            or test['models']['A'] != original_A or test['new_tests_executed'] != {'A': 0, 'B': 1}
            or process['returncode'] != 0 or process['slurm_step'] != marker['slurm_step']
            or marker['slurm_step'] == TRAIN_STEP or marker['allowed_attempts'] != 1
            or value['evaluation_passes'] != 1 or value['forward_batch_calls'] != 144
            or value['active_branch_forward_calls'] != 144 or value['encoder_batch_calls'] != 144
            or value['predicted_vectors_per_query'] != 16384 or value['solver_executions'] != 0
            or value['common_epoch'] != lock['models']['B']['epoch']):
        raise ValueError('Require independently completed ONE B TEST and unchanged historical A object')
    require_evaluation(value, 1000)
    return protocol, lock, test, terminal


def capture_command(command):
    recorded, started = now(), time.monotonic()
    result = subprocess.run(command, text=True, capture_output=True)
    return dict(command=shlex.join(command), recorded_utc=recorded, finished_utc=now(),
        wall_seconds=time.monotonic() - started, returncode=result.returncode, stdout=result.stdout, stderr=result.stderr)


def terminal_observation(step, session):
    scheduler = capture_command(['sacct', '-j', step, '--noheader', '--parsable2',
        '--format=JobIDRaw,State,ExitCode,ElapsedRaw,Start,End,NodeList'])
    tmux = capture_command(['tmux', 'has-session', '-t', '=' + session])
    records = [line.strip().split('|') for line in scheduler['stdout'].splitlines() if line.strip()]
    matches = [row for row in records if len(row) >= 7 and row[0] == step]
    row = matches[0] if len(matches) == 1 else None
    absent = tmux['returncode'] == 1 and any(text in tmux['stderr'] for text in ("can't find session", 'no server running', 'no sessions'))
    return dict(recorded_utc=now(), slurm_step=step, tmux_session=session,
        state=row[1] if row else 'UNAVAILABLE', exit_code=row[2] if row else None,
        elapsed_seconds=float(row[3]) if row and row[3].isdigit() else None,
        node=row[6] if row else None, tmux_terminal=absent, scheduler_evidence=scheduler, tmux_evidence=tmux)


def wait_terminal(step, session, filename, interval):
    failed = ('FAILED', 'CANCELLED', 'TIMEOUT', 'OUT_OF_MEMORY', 'NODE_FAIL', 'PREEMPTED', 'BOOT_FAIL', 'DEADLINE', 'REVOKED')
    while True:
        observation = terminal_observation(step, session)
        if session == 'r71_train':
            status = read(ROOT / 'current_status.json')
            observation['training_status'] = {key: status.get(key) for key in
                ('status', 'epoch', 'completed_epochs', 'successful_updates', 'presentations', 'selected_B_epoch', 'selected_B_validation_correct')}
            observation['pipeline_stage'] = read(ROOT / 'pipeline_status.json')['stage']
        append_jsonl(ROOT / 'delivery_monitor.jsonl', observation)
        atomic_json(ROOT / 'delivery_current_status.json', dict(stage='waiting_' + session, **observation))
        print('[R71 delivery monitor] ' + json.dumps(dict(session=session, step=step, state=observation['state'],
            exit_code=observation['exit_code'], tmux_gone=observation['tmux_terminal'], training=observation.get('training_status'))), flush=True)
        if observation['state'].startswith(failed) or (observation['state'] == 'COMPLETED' and observation['exit_code'] != '0:0'):
            atomic_json(ROOT / filename, observation)
            raise RuntimeError('Original owned step failed; preserve evidence; no automatic retry: ' + step)
        if observation['state'] == 'COMPLETED' and observation['tmux_terminal']:
            require_scheduler_terminal(observation, session, step)
            atomic_json(ROOT / filename, observation)
            return observation
        if session == 'r71_final_test' and observation['tmux_terminal'] and (ROOT / 'test_bootstrap_failure.json').exists():
            raise RuntimeError('ONE TEST bootstrap failed; preserve failure; no retry')
        time.sleep(interval)


def deliver():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or os.environ.get('R71_CONDITIONAL_DELIVERY') != 'authorized':
        raise ValueError('Independent delivery controller must remain CPU-only and parent-authorized')
    preparation = read(ROOT / 'delivery_preparation.json')
    if preparation['syntax_check']['returncode'] != 0 or preparation['delivery_source_hashes'] != delivery_hashes():
        raise ValueError('The ONE new combined syntax invocation must pass before controller launch')
    pane = os.environ.get('TMUX_PANE', '')
    identity = capture_command(['tmux', 'display-message', '-p', '-t', pane, '#{session_name}|#{pane_id}|#{pane_pid}'])
    if identity['returncode'] != 0 or not identity['stdout'].startswith('r71_delivery|'):
        raise ValueError('Independent controller belongs only in owned r71_delivery, not delivery_prep')
    exclusive_json(ROOT / 'delivery_started.json', dict(started_utc=now(), pid=os.getpid(), host=socket.gethostname(),
        tmux='r71_delivery', pane=pane, pane_pid=identity['stdout'].strip().split('|')[2], tmux_identity=identity,
        command='python -B -u -m code.V4.r71_frozen_evaluation --stage deliver',
        parent_conditional_authorization=True, original_training_step=TRAIN_STEP, monitor_interval_seconds=240,
        future_execution='UNVERIFIED until actually completed', no_training_stop_or_restart=True, no_test_retry=True, no_Git=True))
    try:
        wait_terminal(TRAIN_STEP, 'r71_train', 'training_terminal_receipt.json', 240)
        completed_training()
        freeze()
        command = 'bash code/V4/run_v4_r71_test.sh >> code/V4/runs/R71_joint_performance_mixture/test_bootstrap.log 2>&1'
        dispatch = capture_command(['tmux', 'new-session', '-d', '-s', 'r71_final_test', '-c', str(WORKSPACE), command])
        atomic_json(ROOT / 'test_tmux_dispatch.json', dict(dispatch, session='r71_final_test', attempts=1))
        if dispatch['returncode'] != 0:
            raise RuntimeError('ONE TEST tmux dispatch failed; never replace an existing session or retry')
        while not (ROOT / 'test_launch_receipt.json').exists():
            if (ROOT / 'test_bootstrap_failure.json').exists():
                raise RuntimeError('ONE TEST bootstrap failed; no automatic retry')
            owned = capture_command(['tmux', 'has-session', '-t', '=r71_final_test'])
            if owned['returncode'] == 1:
                path = ROOT / 'test_bootstrap.log'
                atomic_json(ROOT / 'test_bootstrap_failure.json', dict(recorded_utc=now(), tmux_evidence=owned,
                    message='Owned TEST tmux exited before assigned-UUID launch receipt',
                    original_log_tail=path.read_text()[-8000:] if path.exists() else None, automatic_retry=False))
                raise RuntimeError('ONE TEST bootstrap exited before launch; no automatic retry')
            time.sleep(30)
        step = read(ROOT / 'test_launch_receipt.json')['slurm_step']
        if not step.startswith('1465.') or step == TRAIN_STEP:
            raise ValueError('TEST requires its one new step within existing allocation1465')
        wait_terminal(step, 'r71_final_test', 'test_terminal_receipt.json', 30)
        completed_test()
        command = [sys.executable, '-B', '-u', '-m', 'code.V4.r71_analysis', '--authorize-analysis']
        with (ROOT / 'analysis.log').open('x') as stream:
            analysis = subprocess.run(command, cwd=WORKSPACE, stdout=stream, stderr=subprocess.STDOUT)
        atomic_json(ROOT / 'analysis_process_receipt.json', dict(recorded_utc=now(), command=shlex.join(command),
            returncode=analysis.returncode, invocations=1, cpu_only=True, automatic_retry=False))
        if analysis.returncode != 0:
            raise RuntimeError('ONCE CPU analysis failed; preserve failure; no automatic retry')
        receipt = read(ROOT / 'delivery_receipt.json')
        receipt['owned_training_and_test_tmux_gone'] = dict(train=capture_command(['tmux', 'has-session', '-t', '=r71_train']),
            test=capture_command(['tmux', 'has-session', '-t', '=r71_final_test']))
        atomic_json(ROOT / 'delivery_receipt.json', receipt)
        atomic_json(ROOT / 'delivery_current_status.json', dict(stage='complete', recorded_utc=now(),
            analysis_invocations=1, parent_reviews_audit_memory_Git_pending=True))
        print('[R71 conditional delivery complete] ' + json.dumps(receipt['decision']), flush=True)
    except BaseException as error:
        atomic_json(ROOT / 'delivery_failure.json', dict(recorded_utc=now(), error_type=type(error).__name__,
            message=str(error), traceback=traceback.format_exc(), automatic_retry=False, original_failures_preserved=True))
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('deliver', 'test'), required=True)
    args = parser.parse_args(argv)
    if args.stage == 'deliver':
        deliver()
        return
    if (os.environ.get('R71_PARENT_GPU_BATON') != 'locked-test' or os.environ.get('CUDA_VISIBLE_DEVICES') != GPU_UUID
            or os.environ.get('SLURM_JOB_ID') != '1465' or socket.gethostname().split('.')[0] != 'gpu03'
            or os.environ.get('R71_TMUX_SESSION') != 'r71_final_test'):
        parser.error('Use only owned r71_final_test/existing1465/gpu03/assigned UUID after VAL lock')
    status = 1
    try:
        final_test()
        status = 0
    except BaseException as error:
        filename = 'final_test_failure.json' if (ROOT / 'final_test_started.json').exists() else 'test_bootstrap_failure.json'
        atomic_json(ROOT / filename, dict(recorded_utc=now(), error_type=type(error).__name__, message=str(error),
            traceback=traceback.format_exc(), automatic_retry=False, marker_preserved=True))
        raise
    finally:
        exclusive_json(ROOT / 'test_process_receipt.json', dict(recorded_utc=now(), returncode=status,
            slurm_step=os.environ['SLURM_JOB_ID'] + '.' + os.environ['SLURM_STEP_ID'], tmux_session='r71_final_test',
            new_evaluated_arms=['B'], new_test_attempts_started={'A': 0, 'B': int((ROOT / 'final_test_started.json').exists())},
            new_tests_completed={'A': 0, 'B': int(status == 0)}, historical_original_A_tests_reused=1,
            process_receipt_is_scheduler_evidence=False, automatic_retry=False))


if __name__ == '__main__':
    main()

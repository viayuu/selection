"""Parent-authorized R72 lock, ONE live composite TEST and ONCE CPU delivery."""

import argparse
import csv
import json
import math
import os
import shlex
import shutil
import socket
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

from .r72_experiment import GPU_UUID, ROOT, WORKSPACE, append, exclusive, read, write


VAL_STEP = '1465.107'
TEST_SESSION = 'r72_final_test'
DELIVERY_SOURCES = ('r72_frozen_evaluation.py', 'run_v4_r72_test.sh', 'r72_analysis.py')


def now():
    return datetime.now(timezone.utc).isoformat()


def require_terminal(receipt, step, session):
    scheduler, tmux = receipt['scheduler_evidence'], receipt['tmux_evidence']
    rows = [line.strip().split('|') for line in scheduler['stdout'].splitlines() if line.strip()]
    rows = [row for row in rows if len(row) >= 7 and row[0] == step]
    if (receipt['slurm_step'] != step or receipt['tmux_session'] != session
            or receipt['state'] != 'COMPLETED' or receipt['exit_code'] != '0:0'
            or receipt['tmux_terminal'] is not True or scheduler['returncode'] != 0
            or step not in scheduler['command'] or len(rows) != 1 or rows[0][1:3] != ['COMPLETED', '0:0']
            or rows[0][6].split('.')[0] != 'gpu03' or float(rows[0][3]) != receipt['elapsed_seconds']
            or tmux['returncode'] != 1 or session not in tmux['command']
            or not any(text in tmux['stderr'] for text in ("can't find session", 'no server running', 'no sessions'))):
        raise ValueError('Require independent actual Slurm COMPLETED/0:0 and owned tmux absence')


def require_evaluation(value, integer_counts=True):
    from code.unified_selector.registry import PROBLEMS
    if set(value['per_problem']) != set(PROBLEMS) or any(entry['n'] != 1000 for entry in value['per_problem'].values()):
        raise ValueError('Require original full18 native 1000-row evaluation per task')
    if integer_counts and (value['native_total'] != 18000 or type(value['native_correct']) is not int
            or value['native_correct'] != sum(entry['native_correct'] for entry in value['per_problem'].values())
            or not math.isclose(value['macro']['macro_top1'], value['native_correct']/18000, abs_tol=1e-12)):
        raise ValueError('Require saved complete native integer counts')


def completed_validation():
    from .r72_consensus import INFERENCE_RULE, MEMBER_NAMES
    protocol, candidate = read(ROOT / 'protocol.json'), read(ROOT / 'validation_candidates.json')
    identity, value = read(ROOT / 'composite_metadata.json'), read(ROOT / 'validation_results.json')
    terminal, launch = read(ROOT / 'validation_terminal_receipt.json'), read(ROOT / 'launch_receipt.json')
    cpu = read(ROOT / 'cpu_check_receipt.json')
    require_terminal(terminal, VAL_STEP, 'r72_validate')
    require_evaluation(value)
    if (launch['slurm_step'] != VAL_STEP or launch['gpu_uuid'] != GPU_UUID or launch['gpu_index'] != 1
            or protocol['member_order'] != list(MEMBER_NAMES) or protocol['inference_rule'] != INFERENCE_RULE
            or candidate['composite'] != identity or candidate['native_correct'] != 8923
            or value['native_correct'] != 8923 or candidate['val'] != value['macro']
            or candidate['proposed_official_arm'] != 'B' or candidate['fixed_A_original_VAL_correct'] != 8885
            or (identity['stored_parameters'], identity['trainable_parameters'], identity['active_parameters_per_query'])
               != (36859060, 0, 5528859)
            or not cpu['passed'] or cpu['actual_invocations'] != 1 or cpu['failed_invocations'] != 0
            or [value['counters'][key] for key in ('query_batches', 'query_rows', 'member_forwards',
                 'member_row_forwards', 'R70_active_branch_calls')] != [144, 18000, 432, 54000, 144]
            or not all(protocol[key] == 0 for key in ('new_TRAIN_epochs', 'new_optimizer_updates', 'new_TRAIN_forward_rows'))):
        raise ValueError('Preserve the one original R72 VAL, complete fixed bundle, rule and zero-training budget')
    return protocol, candidate, identity, value, terminal


def reused_control():
    base = WORKSPACE / 'code/V4/runs'
    trace_path = base / 'R64_pcgrad_full18/reference/R60B_official.json'
    trace = read(trace_path)
    path = Path(trace['provenance']['test']['path'])
    historical = base / 'R63_recurrent_workspaces'
    if path != historical / 'reference/R60B/test_eval.json' or trace['selected_epoch'] != 39:
        raise ValueError('Trace unchanged R60B39 to its original official TEST JSON')
    value = read(path)
    original = read(historical / 'test_results.json')['R60B']
    record = read(historical / 'final_test_lock.json')['R60B']
    receipt = read(historical / 'test_protocol.json')
    expected = read(ROOT / 'protocol.json')['members']['R60B39']['original_record']
    if (value != original or receipt['status'] != 'complete' or receipt['locked_models']['R60B'] != record
            or record['epoch'] != 39 or record['sha256'] != expected['sha256']
            or record['path'] != expected['path'] or trace['provenance']['test_reused_without_rerun'] is not True):
        raise ValueError('Reuse the exact original R60B39 object; no reference inference/recount/reselection')
    require_evaluation(value, integer_counts=False)
    provenance = dict(label='fixed A = historical original R60B39', trace_path=str(trace_path),
        trace_json_pointer='/provenance/test', test_path=str(path), test_json_pointer='',
        original_aggregate_path=str(historical / 'test_results.json'), original_aggregate_json_pointer='/R60B',
        original_lock_path=str(historical / 'final_test_lock.json'), original_lock_json_pointer='/R60B',
        original_process_receipt=str(historical / 'test_protocol.json'), original_slurm_step=receipt['slurm_step'],
        original_tests_already_executed=1, new_R72_tests=0, original_object_unchanged=True,
        standalone_weight_load=False, standalone_inference=False, prediction_recount=False, reselection=False,
        historical_test_exposure=True, reuse_is_protocol_equivalence_not_replication=True)
    return record, value, provenance


def delivery_hashes():
    from .multitask_probe import file_hash
    files = [WORKSPACE / 'code/V4' / name for name in DELIVERY_SOURCES] + [ROOT / 'DELIVERY_PROTOCOL.md']
    return {str(path.relative_to(WORKSPACE)): file_hash(path) for path in files}


def freeze():
    from .multitask_probe import file_hash
    if any((ROOT / name).exists() for name in ('final_test_lock.json', 'final_test_started.json')):
        raise ValueError('Preserve existing exclusive lock/TEST marker; no reselection or retry')
    protocol, candidate, identity, _, terminal = completed_validation()
    if not read(ROOT / 'delivery_preparation.json')['syntax_passed']:
        raise ValueError('The one delivery syntax invocation must have passed')
    path = Path(identity['path'])
    if path != ROOT / 'consensus_complete.pt' or file_hash(path) != identity['sha256']:
        raise ValueError('Lock only the original complete offline composite created before VAL')
    control, _, provenance = reused_control()
    record = dict(identity, native_correct=8923, native_total=18000, val=candidate['val'],
        inference_path=str(path), inference_sha256=identity['sha256'],
        model_class='FrozenConsensus', factory='code.V4.r72_consensus.restore_composite',
        all_three_complete_states_and_buffers=True, no_mutable_historical_checkpoint_dependencies=True)
    lock = dict(authorized=True, parent_authorized=True, frozen_utc=now(), official_arm='B',
        models={'A': control, 'B': record}, control_reuse=provenance,
        selection_rule=protocol['official_rule'], A_original_VAL_correct=8885, B_original_VAL_correct=8923,
        member_order=protocol['member_order'], members=protocol['members'], inference_rule=protocol['inference_rule'],
        full_legal_pools={task: value['pool_ids'] for task, value in protocol['validation_manifest'].items()},
        problems=list(protocol['validation_manifest']), full18=True, native_total=18000, rows_per_task=1000,
        test_batch_size=128, tail_rows=104, query_batches=144, member_forwards=432, member_row_forwards=54000,
        new_tests_required={'A': False, 'B': True}, new_test_attempts_allowed={'A': 0, 'B': 1},
        historical_original_tests_reused={'A': 1}, no_component_evaluation=True, no_checkpoint_reselection=True,
        new_TRAIN_epochs=0, new_optimizer_updates=0, new_TRAIN_forward_rows=0, trainable_parameters=0,
        deployed_selectors_per_query=3, deployment_completed_solvers_per_query=1, evaluation_solver_executions=0,
        CE_definition=protocol['CE_definition'], density_NLL_applicable=False,
        validation_terminal=terminal, delivery_source_hashes=delivery_hashes(), automatic_retry=False)
    exclusive('final_test_lock.json', lock)
    snapshot = ROOT / 'source_delivery'
    snapshot.mkdir()
    for name in DELIVERY_SOURCES:
        shutil.copy2(WORKSPACE / 'code/V4' / name, snapshot / name)
    shutil.copy2(ROOT / 'DELIVERY_PROTOCOL.md', snapshot / 'DELIVERY_PROTOCOL.md')
    print('[R72 mechanical VAL lock] ' + json.dumps(dict(official_arm='B', B_VAL_correct=8923,
        A_VAL_correct=8885, complete_bundle=str(path), frozen_utc=lock['frozen_utc'], TEST_executed=False)), flush=True)
    return lock


def validate_lock():
    protocol, candidate, identity, _, _ = completed_validation()
    lock = read(ROOT / 'final_test_lock.json')
    if (lock['authorized'] is not True or lock['official_arm'] != 'B'
            or lock['models']['B']['inference_sha256'] != identity['sha256']
            or lock['models']['B']['inference_path'] != identity['path']
            or lock['B_original_VAL_correct'] != candidate['native_correct']
            or lock['selection_rule'] != protocol['official_rule'] or lock['member_order'] != protocol['member_order']
            or lock['inference_rule'] != protocol['inference_rule']
            or lock['new_test_attempts_allowed'] != {'A': 0, 'B': 1}
            or lock['delivery_source_hashes'] != delivery_hashes()):
        raise ValueError('Retain the original immutable complete-bundle lock and single TEST authorization')
    return protocol, lock


def assigned_idle():
    if (socket.gethostname().split('.')[0] != 'gpu03' or os.environ.get('SLURM_JOB_ID') != '1465'
            or os.environ.get('CUDA_VISIBLE_DEVICES') != GPU_UUID):
        raise ValueError('Use only allocation1465/gpu03/assigned UUID')
    while True:
        commands = [
            ['nvidia-smi', '--id=' + GPU_UUID, '--query-gpu=index,uuid,name,memory.used,memory.total',
             '--format=csv,noheader,nounits'],
            ['nvidia-smi', '--id=' + GPU_UUID, '--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory',
             '--format=csv,noheader,nounits']]
        outputs = [subprocess.check_output(command, text=True) for command in commands]
        rows = list(csv.reader(outputs[0].splitlines()))
        if len(rows) != 1:
            raise ValueError('Query only the one assigned UUID')
        index, uuid, name, used, total = [item.strip() for item in rows[0]]
        if uuid != GPU_UUID or index != '1':
            raise ValueError('Assigned GPU identity mismatch; no device switch')
        clients = [row for row in csv.reader(outputs[1].splitlines()) if row and row[0].strip() == GPU_UUID]
        value = dict(checked_utc=now(), host=socket.gethostname(),
            slurm_step=os.environ['SLURM_JOB_ID'] + '.' + os.environ['SLURM_STEP_ID'], pid=os.getpid(),
            command=shlex.join([sys.executable] + sys.argv), gpu_uuid=uuid, gpu_index=1, gpu_name=name,
            physical_memory_used_mib=int(used), physical_memory_total_mib=int(total), compute_apps=clients,
            idle=int(used) < 500 and not clients, only_assigned_uuid_queried=True,
            tmux_session=TEST_SESSION, tmux_pane=os.environ['R72_TEST_TMUX_PANE_ID'],
            tmux_pane_pid=int(os.environ['R72_TEST_TMUX_PANE_PID']),
            tmux_server_host=os.environ['R72_TEST_TMUX_SERVER_HOST'], query_commands=commands, raw_stdout=outputs)
        append('test_availability.jsonl', value)
        if value['idle']:
            exclusive('test_launch_receipt.json', value)
            print('[R72 assigned-UUID TEST launch] ' + json.dumps(value), flush=True)
            return value
        print('[R72 TEST wait] Assigned UUID occupied; wait30s on the same device', flush=True)
        time.sleep(30)


def final_test():
    protocol, lock = validate_lock()
    if (ROOT / 'final_test_started.json').exists():
        raise ValueError('The one new B TEST was already started; never repeat it')
    launch = assigned_idle()
    exclusive('final_test_started.json', dict(started_utc=now(), lock=lock, launch=launch,
        allowed_attempts=1, models_to_evaluate=['B'], automatic_retry=False))
    import numpy as np
    import torch
    from code.unified_selector.registry import PROBLEMS
    from .multitask_probe import file_hash, gather_batch
    from .performance_targets import read_raw_costs
    from .r40_experiment import aggregate, families
    from .r43_experiment import policy_metrics
    from .r72_consensus import INPUT_KEYS, MEMBER_NAMES, model_inputs, restore_composite
    from .train import make_loader

    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    whole_started = time.monotonic()
    record = lock['models']['B']
    path = Path(record['inference_path'])
    if file_hash(path) != record['inference_sha256']:
        raise ValueError('The original locked composite changed')
    bundle = torch.load(path, map_location='cpu', weights_only=False)
    if (bundle['format'] != 'R72_complete_frozen_consensus_v1' or bundle['protocol'] != protocol
            or bundle['member_order'] != list(MEMBER_NAMES) or bundle['inference_rule'] != lock['inference_rule']
            or bundle['parameter_counts'] != record['parameter_counts']
            or bundle['all_required_states_and_buffers_included'] is not True or bundle['optimizer_included'] is not False):
        raise ValueError('Restore only all three full states/configs/buffers from the offline locked bundle')
    model = restore_composite(bundle, 'cuda:0')
    del bundle
    if sum(p.numel() for p in model.parameters()) != 36859060 or any(p.requires_grad for p in model.parameters()):
        raise ValueError('All original three complete member states must remain frozen')
    raw, loaders, manifest = {}, {}, {}
    for task in PROBLEMS:
        value = read_raw_costs(task, 'test')
        original = protocol['validation_manifest'][task]
        if (len(value['winner']) != 1000 or value['costs'].dtype != np.float64
                or any(value[key] != original[key] for key in ('pool', 'pool_ids'))):
            raise ValueError('Preserve all original full18x1000 TEST rows and original full legal pools')
        loader = make_loader(task, 'test', 128, 0, shuffle=False, cache_device='cuda:0')
        loader.drop_last = False
        np.testing.assert_array_equal(loader.batch['pool_ids'].cpu(), value['pool_ids'])
        np.testing.assert_array_equal(loader.batch['ind'].cpu(), value['winner'])
        if loader.size != 1000:
            raise ValueError('Original TEST input coverage changed')
        # The unchanged geometry branches calculate label-free features on demand.
        raw[task], loaders[task] = value, loader
        manifest[task] = dict(rows=1000, pool=value['pool'], pool_ids=value['pool_ids'],
            data_hash=value['data_hash'], label_hash=value['label_hash'], costs_dtype='raw FP64',
            original_row_node_and_physical_role_order=True, geometry='original input-only on-demand features; saved TRAIN buffers')
    write('test_data_manifest.json', manifest)
    counters = dict(query_batches=0, query_rows=0, member_forwards=0, member_row_forwards=0,
                    member_calls={name: 0 for name in MEMBER_NAMES}, R70_active_branch_calls=0)
    calls, branch_calls = [], []

    def member_observer(name):
        def observe(module, inputs):
            batch = inputs[0]
            index = int(batch['problem_id'])
            if (not set(batch) <= INPUT_KEYS
                    or batch['pool_ids'].cpu().tolist() != lock['full_legal_pools'][PROBLEMS[index]]):
                raise ValueError('Member inputs must contain only original full-pool instance information')
            counters['member_forwards'] += 1
            counters['member_row_forwards'] += len(batch['n'])
            counters['member_calls'][name] += 1
            calls.append((name, index, len(batch['n'])))
        return observe

    def branch_observer(index):
        def observe(module, inputs):
            batch = inputs[0]
            if int(batch['problem_id']) != index or not set(batch) <= INPUT_KEYS:
                raise ValueError('Activate exactly the original known-problem R70 branch')
            counters['R70_active_branch_calls'] += 1
            branch_calls.append((index, len(batch['n'])))
        return observe

    hooks = [member.register_forward_pre_hook(member_observer(name)) for name, member in zip(MEMBER_NAMES, model.members)]
    hooks += [branch.register_forward_pre_hook(branch_observer(index)) for index, branch in enumerate(model.members[2].branches)]
    import wandb
    run = wandb.init(**protocol['wandb'], name='R72_ONE_locked_composite_TEST', dir=str(ROOT),
        id='r72-test-' + launch['slurm_step'].replace('.', '-'),
        config=dict(lock=lock, new_tests={'A': 0, 'B': 1}, trainable_parameters=0))
    predictions = ROOT / 'test_predictions'
    predictions.mkdir()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    before = time.monotonic()
    results, timings = {}, []
    write('delivery_current_status.json', dict(stage='ONE_full18_live_B_TEST', updated_utc=now(), launch=launch))
    with torch.no_grad():
        for task_index, task in enumerate(PROBLEMS):
            loader = loaders[task]
            columns = {key: [] for key in ('logits', 'Q', 'canonical_logits', 'member_returned_scores',
                                          'doubled_midranks', 'pred', 'pred_global_id')}
            canonical_ids = None
            for start in range(0, 1000, 128):
                indices = torch.arange(start, min(start+128, 1000))
                batch = model_inputs(gather_batch(loader, indices))
                torch.cuda.synchronize()
                batch_started = time.monotonic()
                output = model(batch)
                torch.cuda.synchronize()
                forward_seconds = time.monotonic()-batch_started
                counters['query_batches'] += 1
                counters['query_rows'] += len(indices)
                for key in columns:
                    columns[key].append(output[key].cpu())
                canonical_ids = output['canonical_pool_ids'][0].cpu().numpy()
                free, total = torch.cuda.mem_get_info()
                timing = dict(recorded_utc=now(), problem=task, start_index=start, rows=len(indices),
                    true_nodes=loader.lengths[indices].tolist(),
                    three_member_forward_and_aggregation_seconds=forward_seconds,
                    elapsed_seconds=time.monotonic()-before, counters=json.loads(json.dumps(counters)),
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                    physical_used_bytes=total-free, physical_total_bytes=total, aggregation_dtype=str(output['Q'].dtype),
                    all_legal_scores_finite=True, target_fields_forwarded=[], solver_executions=0)
                timings.append(timing)
                append('test_batches.jsonl', timing)
                write('delivery_current_status.json', dict(stage='ONE_full18_live_B_TEST', **timing))
                if counters['query_batches'] == 1:
                    exclusive('test_first_formal_batch.json', dict(**timing, launch=launch,
                        evaluated_bundle=record['inference_path'], evaluated_bundle_sha256=record['inference_sha256']))
                    print('[R72 first actual TEST batch] ' + json.dumps(timing), flush=True)
                del output
            values = {key: torch.cat(parts, dim=1 if key in ('member_returned_scores', 'doubled_midranks') else 0).numpy()
                      for key, parts in columns.items()}
            metrics, pred = policy_metrics(values['logits'], raw[task])
            np.testing.assert_array_equal(pred, values['pred'])
            metrics['native_correct'] = int(np.count_nonzero(pred == raw[task]['winner']))
            metrics['CE_definition'] = protocol['CE_definition']
            results[task] = metrics
            np.savez_compressed(predictions / f'{task}.npz', indices=np.arange(1000),
                aggregate_scores=values['logits'], logits=values['logits'], Q=values['Q'],
                canonical_aggregate_scores=values['canonical_logits'], member_returned_scores=values['member_returned_scores'],
                doubled_midranks=values['doubled_midranks'], member_names=np.array(MEMBER_NAMES),
                canonical_pool_ids=canonical_ids, canonical_legal_mask=np.ones((1000, len(canonical_ids)), dtype=bool),
                pred=pred, pred_global_id=values['pred_global_id'], winner=raw[task]['winner'], costs=raw[task]['costs'],
                nodes=loader.lengths.numpy(), pool_ids=np.array(raw[task]['pool_ids']), pool=np.array(raw[task]['pool']))
            run.log({'test/' + task + '/top1': metrics['top1'], 'test/' + task + '/Borda_score_CE': metrics['ce'],
                     'test/native_rows_completed': counters['query_rows']})
            print('[R72 TEST task] ' + json.dumps(dict(problem=task, native_correct=metrics['native_correct'],
                rows=1000, elapsed_seconds=time.monotonic()-before)), flush=True)
    torch.cuda.synchronize()
    elapsed = time.monotonic()-before
    expected_queries = [(index, rows) for index in range(18) for rows in [128]*7+[104]]
    expected_calls = [(name, index, rows) for index, rows in expected_queries for name in MEMBER_NAMES]
    if (calls != expected_calls or branch_calls != expected_queries
            or [counters[key] for key in ('query_batches', 'query_rows', 'member_forwards', 'member_row_forwards',
                                         'R70_active_branch_calls')] != [144, 18000, 432, 54000, 144]
            or [entry['rows'] for entry in timings].count(104) != 18):
        raise ValueError('Require exactly all18000 TEST rows/144 batches/432 member calls/54000 member rows/18 tails')
    value = dict(per_problem=results, macro=aggregate(results), families=families(results),
        native_correct=sum(entry['native_correct'] for entry in results.values()), native_total=18000,
        complete=True, evaluation_passes=1, split='test', seconds=elapsed, complete_evaluation_seconds=elapsed,
        forward_and_aggregation_seconds=sum(entry['three_member_forward_and_aggregation_seconds'] for entry in timings),
        counters=counters, query_batch_calls=144, active_R70_branch_calls=144, batch_size=128, tail_rows=104,
        tails_retained=True, CE_definition=protocol['CE_definition'], density_NLL_applicable=False,
        inference_rule=lock['inference_rule'], evaluated_checkpoint_path=record['inference_path'],
        evaluated_checkpoint_sha256=record['inference_sha256'], complete_offline_bundle_loads=1,
        standalone_A_or_component_checkpoint_loads=0, standalone_A_or_component_evaluations=0,
        cached_test_scores=False, prediction_replay=False, solver_executions=0, deployed_selectors_per_query=3,
        deployment_completed_solvers_per_query=1, trainable_parameters=0, new_optimizer_updates=0,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        peak_observed_physical_used_bytes=max(entry['physical_used_bytes'] for entry in timings),
        timing_scope='Whole original full18 TEST with three live sequential forwards, on-demand input geometry, '
                     'integer aggregation, transfers, metrics and NPZ; excludes initial checkpoint/data loading. '
                     'Not singleton or completed-solver latency.', separate_latency_benchmark=False)
    value['macro']['macro_top1'] = value['native_correct']/18000
    require_evaluation(value)
    _, original_A, provenance = reused_control()
    write('test_results_B.json', value)
    write('test_results.json', dict(models={'A': original_A, 'B': value}, lock=lock, complete=True,
        wall_seconds=time.monotonic()-whole_started, original_full18=True,
        new_tests_executed={'A': 0, 'B': 1}, historical_original_tests_reused={'A': 1},
        original_A_object_unchanged=True, control_reuse=provenance, prediction_replay=False, solver_executions=0))
    run.log({'test/native_correct': value['native_correct'], 'test/macro_top1': value['macro']['macro_top1'],
             'test/Borda_score_CE': value['macro']['macro_ce'], 'runtime/complete_evaluation_seconds': elapsed})
    run.finish()
    for hook in hooks:
        hook.remove()
    exclusive('test_process_receipt.json', dict(recorded_utc=now(), returncode=0, launch=launch,
        slurm_step=launch['slurm_step'], evaluation_passes=1, complete_evaluation_seconds=elapsed,
        whole_process_work_seconds=time.monotonic()-whole_started, counters=counters, scheduler_terminal_pending=True))
    print('[R72 ONE B TEST process complete] ' + json.dumps(dict(native_correct=value['native_correct'],
        native_total=18000, complete_evaluation_seconds=elapsed, new_tests={'A': 0, 'B': 1})), flush=True)


def capture(command):
    started, recorded = time.monotonic(), now()
    process = subprocess.run(command, text=True, capture_output=True)
    return dict(command=shlex.join(command), recorded_utc=recorded, finished_utc=now(),
        wall_seconds=time.monotonic()-started, returncode=process.returncode, stdout=process.stdout, stderr=process.stderr)


def terminal_observation(step, session):
    scheduler = capture(['sacct', '-j', step, '--noheader', '--parsable2',
                         '--format=JobIDRaw,State,ExitCode,ElapsedRaw,Start,End,NodeList'])
    tmux = capture(['tmux', 'has-session', '-t', '=' + session])
    rows = [line.strip().split('|') for line in scheduler['stdout'].splitlines() if line.strip()]
    matching = [row for row in rows if len(row) >= 7 and row[0] == step]
    row = matching[0] if len(matching) == 1 else None
    absent = tmux['returncode'] == 1 and any(text in tmux['stderr'] for text in ("can't find session", 'no server running', 'no sessions'))
    return dict(recorded_utc=now(), slurm_step=step, tmux_session=session, state=row[1] if row else 'UNAVAILABLE',
        exit_code=row[2] if row else None, elapsed_seconds=float(row[3]) if row and row[3].isdigit() else None,
        node=row[6] if row else None, tmux_terminal=absent, scheduler_evidence=scheduler, tmux_evidence=tmux)


def completed_test():
    protocol, lock = validate_lock()
    test, marker = read(ROOT / 'test_results.json'), read(ROOT / 'final_test_started.json')
    terminal, process = read(ROOT / 'test_terminal_receipt.json'), read(ROOT / 'test_process_receipt.json')
    launch = marker['launch']
    require_terminal(terminal, launch['slurm_step'], TEST_SESSION)
    _, original_A, _ = reused_control()
    value = test['models']['B']
    require_evaluation(value)
    if (test['complete'] is not True or test['lock'] != lock or marker['lock'] != lock
            or test['models']['A'] != original_A or test['new_tests_executed'] != {'A': 0, 'B': 1}
            or value['evaluation_passes'] != 1 or process['returncode'] != 0
            or process['slurm_step'] != launch['slurm_step'] or marker['allowed_attempts'] != 1
            or launch['slurm_step'] == VAL_STEP or launch['gpu_uuid'] != GPU_UUID or launch['gpu_index'] != 1
            or value['evaluated_checkpoint_sha256'] != lock['models']['B']['inference_sha256']
            or [value['counters'][key] for key in ('query_batches', 'query_rows', 'member_forwards',
                'member_row_forwards', 'R70_active_branch_calls')] != [144, 18000, 432, 54000, 144]):
        raise ValueError('Require actual terminal0 ONE new B TEST and unchanged original A object')
    return protocol, lock, test, terminal


def deliver():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('The delivery controller is CPU-only')
    exclusive('delivery_started.json', dict(started_utc=now(), pid=os.getpid(), allowed_new_B_tests=1,
        allowed_analysis_invocations=1, automatic_retry=False, command=shlex.join([sys.executable]+sys.argv)))
    freeze()
    existing = capture(['tmux', 'has-session', '-t', '=' + TEST_SESSION])
    if existing['returncode'] != 1:
        raise ValueError('Never replace an existing owned TEST session')
    command = ('source /public/home/shiys/miniconda3/etc/profile.d/conda.sh && conda activate easynco && '
               'bash code/V4/run_v4_r72_test.sh > code/V4/runs/R72_heterogeneous_consensus/test.log 2>&1')
    launched = capture(['tmux', 'new-session', '-d', '-s', TEST_SESSION, '-c', str(WORKSPACE), command])
    exclusive('test_tmux_launch_receipt.json', launched)
    if launched['returncode'] != 0:
        raise RuntimeError('Owned TEST tmux launch failed; preserve failure without retry')
    print('[R72 owned TEST tmux launched] ' + json.dumps(launched), flush=True)
    failed = ('FAILED', 'CANCELLED', 'TIMEOUT', 'OUT_OF_MEMORY', 'NODE_FAIL', 'PREEMPTED', 'BOOT_FAIL', 'DEADLINE', 'REVOKED')
    while not (ROOT / 'test_launch_receipt.json').exists():
        live = capture(['tmux', 'has-session', '-t', '=' + TEST_SESSION])
        if live['returncode'] != 0:
            raise RuntimeError('Owned TEST bootstrap ended before launch receipt; preserve test.log, no retry')
        time.sleep(5)
    step = read(ROOT / 'test_launch_receipt.json')['slurm_step']
    while True:
        observation = terminal_observation(step, TEST_SESSION)
        append('delivery_monitor.jsonl', observation)
        print('[R72 TEST terminal monitor] ' + json.dumps(dict(step=step, state=observation['state'],
            exit_code=observation['exit_code'], tmux_gone=observation['tmux_terminal'])), flush=True)
        if observation['state'].startswith(failed) or (observation['state'] == 'COMPLETED' and observation['exit_code'] != '0:0'):
            exclusive('test_terminal_receipt.json', observation)
            raise RuntimeError('Original ONE B TEST step failed; no automatic retry')
        if observation['state'] == 'COMPLETED' and observation['tmux_terminal']:
            require_terminal(observation, step, TEST_SESSION)
            exclusive('test_terminal_receipt.json', observation)
            break
        time.sleep(5)
    completed_test()
    analysis_command = [sys.executable, '-B', '-u', '-m', 'code.V4.r72_analysis']
    started = time.monotonic()
    with (ROOT / 'analysis.log').open('x') as log:
        process = subprocess.run(analysis_command, cwd=WORKSPACE, env=dict(os.environ, CUDA_VISIBLE_DEVICES=''),
                                 stdout=log, stderr=subprocess.STDOUT)
    exclusive('analysis_process_receipt.json', dict(recorded_utc=now(), command=shlex.join(analysis_command),
        returncode=process.returncode, wall_seconds=time.monotonic()-started, invocations=1, cpu_only=True))
    if process.returncode != 0:
        raise RuntimeError('ONCE CPU analysis failed; preserve marker/log, no retry')
    analysis = read(ROOT / 'analysis_receipt.json')
    decision = read(ROOT / 'analysis.json')['decision']
    exclusive('delivery_receipt.json', dict(recorded_utc=now(), validation=read(ROOT / 'validation_terminal_receipt.json'),
        new_B_test=read(ROOT / 'test_terminal_receipt.json'), analysis=analysis,
        analysis_process=read(ROOT / 'analysis_process_receipt.json'), decision=decision,
        control_reuse=read(ROOT / 'final_test_lock.json')['control_reuse'], new_tests_executed={'A': 0, 'B': 1},
        original_A_tests_already_executed=1, new_TRAIN_epochs=0, new_optimizer_updates=0,
        new_TRAIN_forward_rows=0, solver_executions=0, owned_TEST_tmux_gone=True,
        parent_reviews_goal_audit_memory_and_Git_pending=True, worker_Git_actions=False,
        Git_alone_contains_all_inference_assets=False, R58='deferred; eight legacy B semantics disclosed/unfixed'))
    write('delivery_current_status.json', dict(stage='delivery_complete_terminal0_ONCE_analysis0',
        updated_utc=now(), decision=decision, parent_review_pending=True))
    print('[R72 delivery complete] ' + json.dumps(dict(analysis=analysis, decision=decision)), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('deliver', 'test'), required=True)
    args = parser.parse_args()
    try:
        (deliver if args.stage == 'deliver' else final_test)()
    except BaseException as error:
        append('delivery_failures.jsonl', dict(recorded_utc=now(), stage=args.stage, pid=os.getpid(),
            error_type=type(error).__name__, message=str(error), traceback=traceback.format_exc(), automatic_retry=False))
        raise


if __name__ == '__main__':
    main()

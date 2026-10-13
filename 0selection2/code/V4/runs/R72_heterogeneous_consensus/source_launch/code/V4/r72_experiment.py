"""R72: one synthetic check and one fixed full18 consensus VAL; no TRAIN/TEST."""

import argparse
import csv
import json
import os
import shutil
import socket
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path


WORKSPACE = Path(__file__).resolve().parents[2]
ROOT = WORKSPACE / 'code/V4/runs/R72_heterogeneous_consensus'
GPU_UUID = 'GPU-cb626bdb-4319-2f5e-b91f-17fcaa44a83d'
SESSION = 'r72_validate'
SOURCES = ('r72_consensus.py', 'r72_experiment.py', 'r72_cpu_check.py', 'run_v4_r72.sh', 'R72_README.md',
           'pairwise_selector.py', 'solver_code_encoder.py', 'dual_stream.py', 'r64_model.py',
           'r69_model.py', 'r68_model.py', 'local_geometry.py', 'tensor_loader.py',
           'performance_targets.py', 'r43_experiment.py', 'r40_experiment.py')


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def write(name, value):
    path = ROOT / name
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    os.replace(temporary, path)


def exclusive(name, value):
    with (ROOT / name).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def append(name, value):
    with (ROOT / name).open('a') as stream:
        stream.write(json.dumps(value, allow_nan=False) + '\n')


def cpu_check():
    if (ROOT / 'cpu_check_receipt.json').exists():
        raise ValueError('The single R72 CPU invocation was already attempted; preserve it')
    exclusive('cpu_check_started.json', dict(recorded_utc=now(), pid=os.getpid(), invocation=1,
        command='bash code/V4/run_v4_r72.sh cpu-check', automatic_retry=False))
    try:
        from .r72_cpu_check import run_check
        receipt = dict(run_check(), actual_invocations=1, failed_invocations=0, completed_utc=now())
    except BaseException as error:
        write('cpu_check_receipt.json', dict(passed=False, actual_invocations=1, failed_invocations=1,
            recorded_utc=now(), error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc()))
        raise
    write('cpu_check_receipt.json', receipt)
    print('[R72 CPU check] ' + json.dumps(receipt), flush=True)


def assigned_idle():
    if (socket.gethostname().split('.')[0] != 'gpu03' or os.environ.get('SLURM_JOB_ID') != '1465'
            or os.environ.get('CUDA_VISIBLE_DEVICES') != GPU_UUID):
        raise ValueError('Use only original allocation1465/gpu03/assigned UUID')
    while True:
        commands = [
            ['nvidia-smi', '--id=' + GPU_UUID, '--query-gpu=index,uuid,name,memory.used,memory.total',
             '--format=csv,noheader,nounits'],
            ['nvidia-smi', '--id=' + GPU_UUID,
             '--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory', '--format=csv,noheader,nounits']]
        outputs = [subprocess.check_output(command, text=True) for command in commands]
        rows = list(csv.reader(outputs[0].splitlines()))
        if len(rows) != 1:
            raise ValueError('Only the assigned UUID may be queried')
        index, actual, name, used, total = [v.strip() for v in rows[0]]
        if index != '1' or actual != GPU_UUID:
            raise ValueError('Assigned GPU UUID/index mismatch; no device switch')
        clients = [row for row in csv.reader(outputs[1].splitlines()) if row and row[0].strip() == GPU_UUID]
        receipt = dict(checked_utc=now(), host=socket.gethostname(),
            slurm_step=os.environ['SLURM_JOB_ID'] + '.' + os.environ['SLURM_STEP_ID'],
            pid=os.getpid(), command=' '.join(sys.argv), gpu_uuid=actual, gpu_index=1, gpu_name=name,
            physical_memory_used_mib=int(used), physical_memory_total_mib=int(total), compute_apps=clients,
            idle=int(used) < 500 and not clients, only_assigned_uuid_queried=True, tmux_session=SESSION,
            tmux_pane=os.environ['R72_TMUX_PANE_ID'], tmux_pane_pid=int(os.environ['R72_TMUX_PANE_PID']),
            tmux_server_host=os.environ['R72_TMUX_SERVER_HOST'], query_commands=commands, raw_stdout=outputs)
        append('availability.jsonl', receipt)
        if receipt['idle']:
            exclusive('launch_receipt.json', receipt)
            print('[R72 assigned-UUID launch] ' + json.dumps(receipt), flush=True)
            return receipt
        print('[R72 wait] Assigned UUID occupied; waiting30s on the same device', flush=True)
        time.sleep(30)


def fixed_metadata():
    from .r72_consensus import INPUT_KEYS, INFERENCE_RULE, MEMBER_NAMES
    runroot = WORKSPACE / 'code/V4/runs'
    locations = (
        ('R60_expanded_partial_supervision', 'B_expanded_seed2', 'best.pt', 39, None),
        ('R67_supervised_contrastive_full18', 'B_contrastive_pretrain_seed2', 'best.pt', 36, 76),
        ('R70_latent_mixup', 'B_latent_mixup_seed2', 'best_checkpoints/epoch023.pt', 23, 63))
    records, configs = {}, {}
    for name, (experiment, run, filename, epoch, global_epoch) in zip(MEMBER_NAMES, locations):
        root = runroot / experiment
        directory = root / run
        args, result = read(directory / 'args.json'), read(directory / 'result.json')
        if result['best']['epoch'] != epoch or (global_epoch is not None and result['best']['global_epoch'] != global_epoch):
            raise ValueError('Use only the preapproved historical selected checkpoints; no reselection')
        if name == MEMBER_NAMES[0]:
            record = read(root / 'locked_checkpoints.json')['B']
            if record['epoch'] != 39 or args['model_params']['pair_mode'] != 'score_difference':
                raise ValueError('Retain original R60 cost-selected score-difference/maximin policy')
        else:
            record = read(root / 'validation_candidates.json')['models']['B']
            if (record['epoch'], record['global_epoch']) != (epoch, global_epoch):
                raise ValueError('Retain the exact R67/R70 common historical epochs')
        if name == MEMBER_NAMES[2] and (record['selected_epoch_checkpoint'] != str(directory / filename)
                or record['all18_common_epoch'] is not True):
            raise ValueError('R70 must be its unique complete18 common-epoch bank')
        records[name] = dict(path=str(directory / filename), epoch=epoch, global_epoch=global_epoch,
            original_record=record, args_path=str(directory / 'args.json'), result_path=str(directory / 'result.json'),
            constructor=('code.V4.solver_code_encoder.make_r45_model' if name == MEMBER_NAMES[0] else
                         'code.V4.r64_model.NativeUtilitySelector' if name == MEMBER_NAMES[1] else
                         'code.V4.r69_model.make_bank + strict complete-bank state'),
            returned_scores='score-difference/maximin logits' if name == MEMBER_NAMES[0] else 'direct-utility logits',
            parameter_mode='saved averaged x; eval; no optimizer transition/projector' if name == MEMBER_NAMES[1]
                           else 'saved ordinary weights; eval',
            all_original_buffers=True, input_keys=sorted(INPUT_KEYS))
        configs[name] = args
    manifest = configs[MEMBER_NAMES[0]]['data_manifest']['val']
    manifest_keys = ('pool', 'pool_ids', 'data_hash', 'label_hash')
    if any(any(configs[name]['data_manifest']['val'][task][key] != value[key]
               for task, value in manifest.items() for key in manifest_keys) for name in MEMBER_NAMES[1:]):
        raise ValueError('The original native validation manifests must match across all three policies')
    if any(configs[name]['geometry_stats'] != configs[MEMBER_NAMES[0]]['geometry_stats'] for name in MEMBER_NAMES[1:]):
        raise ValueError('Keep the existing common original TRAIN-only geometry definition')
    protocol = dict(recorded_utc=now(), new_arms=['B'], reused_arms=['A'], members=records,
        member_order=list(MEMBER_NAMES), inference_rule=INFERENCE_RULE, input_keys=sorted(INPUT_KEYS),
        aggregation_trainable_parameters=0, aggregation_arithmetic='int64',
        CE_definition='Borda-score CE: legal stable log-softmax of z=-Q/6; not calibrated probability quality',
        density_NLL_applicable=False, all_parameters_frozen=True, eval_mode=True, FP32=True, no_grad=True,
        validation_manifest=manifest, full18=True, validation_rows=18000, rows_per_task=1000,
        batch_size=128, tail_rows=104, query_batches=144, member_forwards=432, member_row_forwards=54000,
        new_TRAIN_epochs=0, new_optimizer_updates=0, new_TRAIN_forward_rows=0, evaluation_solver_executions=0,
        deployed_selectors_per_query=3, deployed_completed_solvers_per_query=1,
        fixed_control=dict(label='R60B39', original_VAL_correct=8885, native_total=18000,
            saved_record=records[MEMBER_NAMES[0]]['original_record'], separate_weight_load=False,
            standalone_inference=False, recount=False, reselection=False, new_training=False, new_TEST=False,
            original_TEST_reused_later=True, reuse_is_protocol_equivalence_not_replication=True),
        official_rule='B iff native INTEGER VAL correct>8885; otherwise unchanged A=R60B39',
        membership_search=False, checkpoint_reselection=False, component_metrics_computed=False,
        capacity_or_compute_matched=False, historical_TEST_exposure_disclosed=True, NEW_R71_TEST_read=False,
        TEST_authorized=False, final_TEST_lock_created=False, automatic_retry=False,
        goal_status='ACTIVE/unachieved', goal_original_TEST_threshold=12600,
        deferred=['R58', 'eight legacy B semantic inconsistencies; disclosed/unfixed, not native70 veto'],
        limitations=['R66 weak ensemble counterevidence', 'instance error complementarity unestablished',
                     'R67/R70 share stage1 pretraining', 'consensus may dilute strongest member'],
        wandb=dict(project='selector', entity='yjkds-southern-university-of-science-technology', mode='offline'))
    return protocol, configs


def historical_accounting(protocol):
    from .r72_consensus import MEMBER_NAMES
    historical = {}
    for name in MEMBER_NAMES:
        result = read(protocol['members'][name]['result_path'])
        historical[name] = dict(source=protocol['members'][name]['result_path'],
            **{key: result[key] for key in ('epochs_completed', 'successful_updates', 'presentations',
                'successful_view_rows', 'amp_retries', 'wall_seconds', 'extra_rows_by_problem') if key in result})
    stage1_path = Path(protocol['members'][MEMBER_NAMES[1]]['result_path']).with_name('stage1_result.json')
    stage1 = read(stage1_path)
    shared = dict(source=str(stage1_path), counted_once=True, shared_by=['R67B', 'R70B'],
        **{key: stage1[key] for key in ('epochs_completed', 'successful_updates', 'presentations',
                                       'amp_retries', 'wall_seconds')})
    return dict(new_R72_training=dict(epochs=0, updates=0, rows=0, encoder_view_rows=0),
        historical_members=historical, inherited_R67B_stage1_once=shared,
        accounting_note='R67 total already includes stage1 once; R70 historical result is stage2 only. '
                        'Do not add shared stage1 a second time. R60 expanded supervision remains historical.')


def create_bundle(protocol, configs):
    import torch
    from code.unified_selector.registry import GLOBAL_SOLVERS, PROBLEMS
    from .multitask_probe import file_hash
    from .r59_experiment import atomic_save
    from .r72_consensus import INFERENCE_RULE, MEMBER_NAMES, restore_member
    payload, members, counts = {}, [], {}
    for name in MEMBER_NAMES:
        metadata = protocol['members'][name]
        path = Path(metadata['path'])
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        if checkpoint['args'] != configs[name] or checkpoint['epoch'] != metadata['epoch'] - 1:
            raise ValueError('Original checkpoint constructor arguments/selected epoch mismatch')
        if configs[name]['model_params']['solver_feature_spec']['solver_names'] != GLOBAL_SOLVERS:
            raise ValueError('Retain original global solver identities')
        if name == MEMBER_NAMES[1] and (checkpoint.get('parameter_mode') != 'averaged x'
                or checkpoint.get('checkpoint_mode') != 'eval' or checkpoint.get('global_epoch') != 76
                or checkpoint.get('projector_included') is not False or checkpoint.get('stage') != 2):
            raise ValueError('R67 requires saved complete stage2 averaged-x eval weights without projector')
        if name == MEMBER_NAMES[2] and (checkpoint.get('global_epoch') != 63
                or checkpoint.get('common_epoch') != 23 or checkpoint.get('branch_epochs') != [23] * 18
                or checkpoint.get('all18_complete_branches') is not True
                or checkpoint.get('problem_names') != list(PROBLEMS) or checkpoint.get('checkpoint_mode') != 'eval'):
            raise ValueError('R70 requires all18 complete branches from common epoch23/global63')
        digest = file_hash(path)
        if digest != metadata['original_record']['sha256']:
            raise ValueError('Required historical member file identity changed')
        state = checkpoint['model']
        member = restore_member(name, configs[name]['model_params'], state)
        if any('projector' in key for key in state):
            raise ValueError('No projector belongs in the composite')
        counts[name] = sum(parameter.numel() for parameter in member.parameters())
        payload[name] = dict(model=state, args=configs[name], metadata=metadata, source_sha256=digest,
            checkpoint_metadata={key: value for key, value in checkpoint.items() if key not in ('model', 'args')})
        members.append(member)
    bundle = dict(format='R72_complete_frozen_consensus_v1', created_utc=now(), member_order=list(MEMBER_NAMES),
        members=payload, inference_rule=INFERENCE_RULE, protocol=protocol,
        problem_names=list(PROBLEMS), global_solver_names=list(GLOBAL_SOLVERS),
        all_required_states_and_buffers_included=True, no_mutable_pointer_dependency=True,
        optimizer_included=False, trainable_parameters=0, parameter_counts=counts)
    path = ROOT / 'consensus_complete.pt'
    if path.exists():
        raise ValueError('The one complete composite bundle already exists; never replace it')
    atomic_save(path, bundle)
    identity = dict(path=str(path), sha256=file_hash(path), created_utc=bundle['created_utc'],
        member_order=list(MEMBER_NAMES), member_source_sha256={name: payload[name]['source_sha256'] for name in MEMBER_NAMES},
        parameter_counts=counts, stored_parameters=sum(counts.values()), trainable_parameters=0,
        active_parameters_per_query=counts[MEMBER_NAMES[0]]+counts[MEMBER_NAMES[1]]+counts[MEMBER_NAMES[2]]//18,
        complete_states=True, all18_R70_common_epoch=True, checkpoint_loads=3,
        standalone_A_or_component_evaluations=0, TEST_lock=False)
    write('composite_metadata.json', identity)
    return members, identity


def validate():
    if not read(ROOT / 'cpu_check_receipt.json')['passed']:
        raise ValueError('The one synthetic CPU check must pass before formal validation')
    for name in ('protocol.json', 'formal_started.json', 'validation_results.json'):
        if (ROOT / name).exists():
            raise ValueError('Preserve existing R72 artifacts; no automatic retry/restart')
    launch = assigned_idle()
    import numpy as np
    import torch
    from code.unified_selector.registry import PROBLEMS
    from .local_geometry import FIELDS
    from .multitask_probe import file_hash, gather_batch
    from .performance_targets import read_raw_costs
    from .r40_experiment import aggregate, families
    from .r43_experiment import policy_metrics
    from .r72_consensus import FrozenConsensus, MEMBER_NAMES, model_inputs
    from .train import make_loader

    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    protocol, configs = fixed_metadata()
    snapshot = ROOT / 'source_launch'
    snapshot.mkdir()
    source_hashes = {}
    for filename in SOURCES:
        path = WORKSPACE / 'code/V4' / filename
        destination = snapshot / 'code/V4' / filename
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
        source_hashes[str(path.relative_to(WORKSPACE))] = file_hash(path)
    plan = ROOT / 'approved_plan.md'
    shutil.copy2(plan, snapshot / 'approved_plan.md')
    protocol.update(source_hashes=source_hashes, approved_plan_sha256=file_hash(plan),
                    launch=launch, historical_accounting=historical_accounting(protocol))
    exclusive('protocol.json', protocol)
    exclusive('formal_started.json', dict(started_utc=now(), pid=os.getpid(), slurm_step=launch['slurm_step'],
        tmux_session=SESSION, gpu_uuid=GPU_UUID, stage='one original full18 VAL', allowed_attempts=1,
        new_TRAIN_updates=0, TEST_authorized=False, automatic_retry=False))
    write('pipeline_status.json', dict(stage='constructing_complete_fixed_bundle', updated_utc=now()))
    preparation_started = time.monotonic()
    members, identity = create_bundle(protocol, configs)
    model = FrozenConsensus(members).to('cuda:0').eval()
    del members
    raw, loaders, manifest = {}, {}, {}
    geometry_dir = WORKSPACE / 'code/V4/runs/R39_performance_model/geometry_cache'
    if read(geometry_dir / 'stats.json') != configs[MEMBER_NAMES[0]]['geometry_stats']:
        raise ValueError('Only the existing original TRAIN-fitted geometry cache may be used')
    for task in PROBLEMS:
        value = read_raw_costs(task, 'val')
        original = protocol['validation_manifest'][task]
        if (len(value['winner']) != 1000 or value['costs'].dtype != np.float64
                or any(value[key] != original[key] for key in ('pool', 'pool_ids', 'data_hash', 'label_hash'))):
            raise ValueError('Preserve all original18x1000 VAL rows, pools, native labels and raw FP64 costs')
        loader = make_loader(task, 'val', 128, 0, shuffle=False, cache_device='cuda:0')
        loader.drop_last = False
        np.testing.assert_array_equal(loader.batch['pool_ids'].cpu(), value['pool_ids'])
        np.testing.assert_array_equal(loader.batch['ind'].cpu(), value['winner'])
        if loader.size != 1000:
            raise ValueError('Original validation input coverage changed')
        if loader.batch['kind'] == 'coord':
            features = torch.load(geometry_dir / f'{task}_val.pt', map_location='cpu', weights_only=True)
            if features.shape != (1000, loader.batch['node'].shape[1], len(FIELDS)):
                raise ValueError('Existing label-free geometry cache shape mismatch')
            loader.batch['node_geom'] = features.to('cuda:0')
        raw[task], loaders[task] = value, loader
        manifest[task] = dict(original, rows=1000, costs_dtype='raw FP64', original_row_and_node_order=True)
    write('validation_data_manifest.json', manifest)
    counters = dict(query_batches=0, query_rows=0, member_forwards=0, member_row_forwards=0,
                    member_calls={name: 0 for name in MEMBER_NAMES}, R70_active_branch_calls=0)

    def member_observer(name):
        def observe(module, inputs):
            batch = inputs[0]
            if any(key not in protocol['input_keys'] for key in batch):
                raise ValueError('Labels/auxiliary fields crossed the member forward boundary')
            counters['member_forwards'] += 1
            counters['member_row_forwards'] += len(batch['n'])
            counters['member_calls'][name] += 1
        return observe

    handles = [member.register_forward_pre_hook(member_observer(name))
               for name, member in zip(MEMBER_NAMES, model.members)]
    def branch_observer(module, inputs):
        counters['R70_active_branch_calls'] += 1
    handles += [branch.register_forward_pre_hook(branch_observer) for branch in model.members[2].branches]
    prediction_dir = ROOT / 'validation_predictions'
    prediction_dir.mkdir()
    import wandb
    wandb_run = wandb.init(**protocol['wandb'], name='R72_fixed_heterogeneous_consensus',
                          config=protocol, dir=str(ROOT), id='r72-' + launch['slurm_step'].replace('.', '-'))
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.monotonic()
    results, timings = {}, []
    write('pipeline_status.json', dict(stage='formal_full18_validation', updated_utc=now(),
        composite=identity, preparation_seconds=time.monotonic()-preparation_started))
    with torch.no_grad():
        for task in PROBLEMS:
            loader = loaders[task]
            columns = {key: [] for key in ('logits', 'Q', 'canonical_logits', 'member_returned_scores',
                                          'doubled_midranks', 'pred', 'pred_global_id')}
            canonical_ids = None
            for start in range(0, 1000, 128):
                indices = torch.arange(start, min(start + 128, 1000))
                batch = model_inputs(gather_batch(loader, indices))
                if batch['pool_ids'].cpu().tolist() != raw[task]['pool_ids']:
                    raise ValueError('Preserve original cost-column/global-ID semantics')
                torch.cuda.synchronize()
                before = time.monotonic()
                output = model(batch)
                torch.cuda.synchronize()
                forward_seconds = time.monotonic()-before
                counters['query_batches'] += 1
                counters['query_rows'] += len(indices)
                for key in columns:
                    value = output[key].cpu()
                    columns[key].append(value)
                canonical_ids = output['canonical_pool_ids'][0].cpu().numpy()
                memory_free, memory_total = torch.cuda.mem_get_info()
                record = dict(recorded_utc=now(), problem=task, rows=len(indices), start_index=start,
                    padded_nodes=int(batch['node'].shape[1]) if 'node' in batch else int(batch['matrix'].shape[1]),
                    three_member_forward_and_aggregation_seconds=forward_seconds,
                    elapsed_seconds=time.monotonic()-started, counters=json.loads(json.dumps(counters)),
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                    physical_used_bytes=memory_total-memory_free, physical_total_bytes=memory_total,
                    legal_scores_finite=True, aggregation_dtype=str(output['Q'].dtype),
                    target_fields_forwarded=[], solver_executions=0)
                timings.append(record)
                append('validation_batches.jsonl', record)
                write('current_status.json', dict(status='RUNNING', **record))
                if counters['query_batches'] == 1:
                    write('first_formal_batch.json', dict(**record, launch=launch, composite=identity,
                        completed_original_VAL_rows=len(indices), new_TRAIN_updates=0))
                    print('[R72 first formal VAL batch] ' + json.dumps(record), flush=True)
                del output
            values = {key: torch.cat(parts, dim=1 if key in ('member_returned_scores', 'doubled_midranks') else 0).numpy()
                      for key, parts in columns.items()}
            result, pred = policy_metrics(values['logits'], raw[task])
            np.testing.assert_array_equal(pred, values['pred'])
            result['native_correct'] = int(np.count_nonzero(pred == raw[task]['winner']))
            result['CE_definition'] = protocol['CE_definition']
            results[task] = result
            np.savez_compressed(prediction_dir / f'{task}.npz', indices=np.arange(1000),
                aggregate_scores=values['logits'], logits=values['logits'], Q=values['Q'],
                canonical_aggregate_scores=values['canonical_logits'],
                member_returned_scores=values['member_returned_scores'], doubled_midranks=values['doubled_midranks'],
                member_names=np.array(MEMBER_NAMES), canonical_pool_ids=canonical_ids,
                pred=pred, pred_global_id=values['pred_global_id'], winner=raw[task]['winner'],
                costs=raw[task]['costs'], nodes=loader.lengths.numpy(), pool_ids=np.array(raw[task]['pool_ids']),
                pool=np.array(raw[task]['pool']))
            wandb_run.log({'val/' + task + '/top1': result['top1'],
                           'val/' + task + '/Borda_score_CE': result['ce'],
                           'val/native_rows_completed': counters['query_rows']})
            print('[R72 VAL task] ' + json.dumps(dict(problem=task, rows=result['n'],
                native_correct=result['native_correct'], Borda_score_CE=result['ce'],
                elapsed_seconds=time.monotonic()-started)), flush=True)
    torch.cuda.synchronize()
    seconds = time.monotonic()-started
    if (counters['query_batches'], counters['query_rows'], counters['member_forwards'],
            counters['member_row_forwards'], counters['R70_active_branch_calls']) != (144, 18000, 432, 54000, 144):
        raise ValueError('Require exactly144 query batches/432 member forwards/54000 member rows/full18000 VAL')
    if set(counters['member_calls'].values()) != {144} or [record['rows'] for record in timings].count(104) != 18:
        raise ValueError('Every query uses all3 members; preserve18 tails of104')
    evaluation = dict(per_problem=results, macro=aggregate(results), families=families(results),
        native_correct=sum(value['native_correct'] for value in results.values()), native_total=18000,
        seconds=seconds, complete=True, evaluation_passes=1, split='val', counters=counters,
        CE_definition=protocol['CE_definition'], density_NLL_applicable=False,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        peak_observed_physical_used_bytes=max(record['physical_used_bytes'] for record in timings),
        forward_and_aggregation_seconds=sum(record['three_member_forward_and_aggregation_seconds'] for record in timings),
        wall_scope='Complete full18 VAL including three sequential member forwards, aggregation, CPU transfers, '
                   'saved scores and metrics; excludes checkpoint/data preparation. Not singleton/solver latency.',
        standalone_component_evaluations=0, standalone_A_evaluations=0, solver_executions=0,
        new_TRAIN_epochs=0, new_optimizer_updates=0, new_TRAIN_forward_rows=0)
    evaluation['macro']['macro_top1'] = evaluation['native_correct']/18000
    if any(value['n'] != 1000 for value in results.values()) or set(results) != set(PROBLEMS):
        raise ValueError('Require complete equal18 native validation coverage')
    official = 'B' if evaluation['native_correct'] > 8885 else 'A'
    candidate = dict(composite=identity, native_correct=evaluation['native_correct'], native_total=18000,
        val=evaluation['macro'], inference_rule=protocol['inference_rule'], one_fixed_candidate=True,
        proposed_official_arm=official, official_rule=protocol['official_rule'], parent_freeze_pending=True,
        fixed_A_original_VAL_correct=8885, TEST_authorized=False, TEST_executed=False,
        final_TEST_lock_created=False, no_epoch_or_member_selection=True)
    write('validation_results.json', evaluation)
    write('validation_candidates.json', candidate)
    wandb_run.log({'val/macro_top1': evaluation['macro']['macro_top1'],
                   'val/Borda_score_CE': evaluation['macro']['macro_ce'],
                   'val/native_correct': evaluation['native_correct'], 'runtime/validation_seconds': seconds})
    wandb_run.finish()
    for handle in handles:
        handle.remove()
    write('validation_complete.json', dict(completed_utc=now(), evaluation=evaluation, candidate=candidate,
        launch=launch, historical_accounting=protocol['historical_accounting'], goal_status='ACTIVE/unachieved',
        scheduler_terminal_verified=False, terminal_receipt_pending=True))
    write('current_status.json', dict(status='VALIDATION_PROCESS_COMPLETE', native_correct=evaluation['native_correct'],
        native_total=18000, proposed_official_arm=official, counters=counters, seconds=seconds,
        scheduler_terminal_verified=False, updated_utc=now()))
    write('pipeline_status.json', dict(stage='validation_process_complete_terminal_pending', updated_utc=now()))
    print('[R72 complete VAL process; scheduler terminal pending] ' + json.dumps(candidate), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('cpu-check', 'validate'), required=True)
    args = parser.parse_args()
    try:
        (cpu_check if args.stage == 'cpu-check' else validate)()
    except BaseException as error:
        if args.stage == 'validate':
            append('failures.jsonl', dict(recorded_utc=now(), pid=os.getpid(), error_type=type(error).__name__,
                error=str(error), traceback=traceback.format_exc(), automatic_retry=False,
                slurm_step=os.environ.get('SLURM_JOB_ID', '') + '.' + os.environ.get('SLURM_STEP_ID', '')))
            write('pipeline_status.json', dict(stage='failed_preserved_no_retry', updated_utc=now(), error=str(error)))
        raise


if __name__ == '__main__':
    main()

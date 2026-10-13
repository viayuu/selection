"""One formal R74 bounded TRAIN/VAL acquisition pass, then the fixed40E policy."""

import argparse
import gc
import json
import os
import shutil
import socket
import time
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from code.unified_selector.registry import GLOBAL_SOLVERS, P2I, PROBLEMS
from .multitask_probe import file_hash, state_hash
from .performance_targets import read_raw_costs
from .r40_experiment import aggregate, families
from .r43_experiment import policy_metrics
from .r59_experiment import GPU_UUID, REFERENCE, WORKSPACE, atomic_json, atomic_save
from .r64_experiment import dropout_seed, paired_triples, prefer_checkpoint, synchronize, triple_indices
from .r65_optimizer import ARM_CONFIGS, configure_update_lr, evaluation_mode, make_optimizer, observed_lr, training_mode
from .r67_model import cpu_state
from .r74_acquisition import BQ_ROOT, BRIDGE, EASY_ROOT, RELD_ROOT, SCALARS, ProbeRecipes, recipe_records
from .r74_data import CachedTask, GroupWriter, NativeInputs, probe_ids
from .r74_gradients import GradientLayout, update_triple
from .r74_model import INFERENCE_RULE, PROBE_NAMES, R74Model
from .train import set_seed
from .training_monitor import capture_rng


ROOT = WORKSPACE / 'code/V4/runs/R74_adaptive_incomplete_acquisition'
DIRECTORY = ROOT / 'B_adaptive_seed2'
R73 = WORKSPACE / 'code/V4/runs/R73_expansion_teacher_training'
BASE = R73 / 'B_teacher_training_seed2/best_checkpoints/epoch023.pt'
R67 = WORKSPACE / 'code/V4/runs/R67_supervised_contrastive_full18'
OWNED = ('r74_acquisition.py', 'r74_model.py', 'r74_data.py', 'r74_gradients.py',
         'r74_experiment.py', 'r74_cpu_check.py', 'run_v4_r74.sh', 'R74_README.md')
REUSED = ('r64_model.py', 'dual_stream.py', 'V4Model.py', 'local_geometry.py', 'solver_features.py',
    'r64_experiment.py', 'r65_gradients.py', 'r65_optimizer.py', 'r67_model.py', 'r51_probe.py',
    'r50_solver_encoder.py', 'r48_common.py', 'r41_label_stability.py', 'r57_probe.py',
    'r59_experiment.py', 'r45_experiment.py', 'r43_experiment.py', 'r42_experiment.py', 'r40_experiment.py',
    'performance_targets.py', 'performance_experiment.py', 'performance_evaluation.py',
    'pairwise_objective.py', 'pairwise_selector.py', 'multitask_probe.py', 'train.py', 'training_monitor.py',
    'r65_official_schedulefree/__init__.py', 'r65_official_schedulefree/adamw_schedulefree.py',
    'r65_official_schedulefree/provenance.json', 'r65_official_schedulefree/LICENSE')


def now():
    return datetime.now(timezone.utc).isoformat()


def append(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as stream:
        stream.write(json.dumps(value, allow_nan=False)+'\n')
        stream.flush()


def memory(device):
    free, total = torch.cuda.mem_get_info(device)
    return dict(allocated_bytes=torch.cuda.memory_allocated(device), reserved_bytes=torch.cuda.memory_reserved(device),
        peak_allocated_bytes=torch.cuda.max_memory_allocated(device), peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
        physical_used_bytes=total-free, physical_total_bytes=total)


def status(stage, launch, **fields):
    record_root = ROOT / launch.get('continuation_record', 'continuation') if launch.get('execution_continuation') else ROOT
    atomic_json(record_root / 'current_status.json', dict(stage=stage, updated_utc=now(), launch=launch, **fields))
    atomic_json(record_root / 'pipeline_status.json', dict(stage=stage, updated_utc=now(), pid=os.getpid(),
                                                 slurm_step=launch['slurm_step']))


def source_identity():
    return {name: file_hash(Path(__file__).parent / name) for name in OWNED}


def prepare(launch):
    for name in ('protocol.json', 'formal_started.json', 'feature_cache_complete.json', 'training_complete.json'):
        if (ROOT / name).exists():
            raise ValueError('Preserve the original R74 formal process; no automatic restart: '+name)
    check = json.loads((ROOT / 'cpu_check.json').read_text())
    if not check['passed'] or check['source_hashes'] != source_identity():
        raise ValueError('One synthetic invocation must pass on the actual R74 sources')
    base_args = json.loads((R73 / 'B_teacher_training_seed2/args.json').read_text())
    schedule_identity = json.loads((R67 / 'schedule_seed2.json').read_text())
    schedule = paired_triples({task: 10000 for task in PROBLEMS}, epochs=80, seed=2, batch_size=256)
    if state_hash(schedule) != schedule_identity['sha256']:
        raise ValueError('The original seed2 global41-80 native task/row schedule changed')
    r48 = json.loads((WORKSPACE / 'code/V4/runs/R48_core_diagnosis/audit_plan.json').read_text())['solvers']
    if not (REFERENCE / 'geometry_cache/stats.json').exists():
        raise ValueError('Original TRAIN geometry cache required; no normalization refit')
    geometry = json.loads((REFERENCE / 'geometry_cache/stats.json').read_text())
    if geometry != base_args['geometry_stats']:
        raise ValueError('Frozen R73 normalizer identity and native geometry cache differ')
    protocol = dict(experiment='R74_adaptive_incomplete_acquisition', seed=2, new_arms=['B'],
        approved_plan=str(ROOT / 'approved_plan.md'), package_comparison=True,
        adaptivity_causally_isolated=False, exposure_capacity_or_compute_matched=False,
        fixed_control=dict(name='R73B23/global63', checkpoint=str(BASE), native_correct=8959, native_total=18000,
            history=str(R73 / 'B_teacher_training_seed2/history.json'), result=str(R73 / 'B_teacher_training_seed2/result.json'),
            reuse='unchanged historical protocol control, not fresh replication', standalone_forwards=0),
        frozen_base=dict(checkpoint=str(BASE), complete_shared_selector=True, trained_head_retained=True,
            parameter_mode='averaged x', fp32=True, trainable=False, forwards_per_generation_input=1),
        model_params=base_args['model_params'], geometry_stats=geometry,
        geometry_normalization='unchanged original TRAIN-fitted buffers; no refit',
        problems=list(PROBLEMS), solver_names=list(GLOBAL_SOLVERS), legal_task_solver_pairs=128,
        recipes=recipe_records(), reld_runtime_contracts={key: value['runtime_params_contract'] for key, value in r48.items()},
        extractor_scope='new separately locked FP32 greedy augmentation1 recipes, not historical deployment recertification',
        quarter_rule='h=(n//4-1)//2; initial required node then h and h further selections; depot consumes selections',
        noncompletion='before/after every step: zero finished/done; required visited<=1+2h<=n//4',
        no_completion_reward_reconstruction_or_final_cost_calls=True,
        token_dim=403, token_fields=dict(node_embeddings=['start', 'current', 'distinct_visited_mean'],
            embedding_dim=128, scalars=list(SCALARS), probe_identity=list(PROBE_NAMES), checkpoints=['shallow', 'deep']),
        scalar_conventions=dict(selections='all selections including initialization depot where applicable',
            visited='distinct required nodes; depot excluded for VRP quarter/count',
            visited_mean='all distinct actually visited physical nodes including a visited depot',
            partial_distance='sum of observed directed/euclidean traversed edges only, not a completed objective',
            scale='n*max(input mean off-diagonal edge distance,1e-8)',
            unavailable_fields=0, legal_fraction='finite legal entries / last decoder decision vector length'),
        storage='lossless factorization: original FP32 F embeddings, integer start/current, packed visited identities, '
                'exact FP32 visited means/scalars; CPU memmap; no token quantization',
        gate_information='root h only; second h+mean currently observed shallow adapted tokens+first-probe onehot only',
        leaves=['MTL shallow+deep', 'MOEL shallow+deep', 'both shallow'],
        leaf_weights='a*b,(1-a)*c,a*(1-b)+(1-a)*(1-c)', views_per_leaf=2,
        objective='expected two-view native CE + .35*min(u/720,1)*weighted symmetric legal KL; detach weights for KL only',
        correction_final_zero=True, gate_finals_zero=True, gate_dropout=0., adapter_dropout=.1,
        generation_batch=16, generation_grouping='task/trueN/native start count; backhaul positive-count identity retained',
        generation=dict(train_rows=180000, val_rows=18000, train_initializations=330000,
            val_initializations=33000, frozen_base_query_rows=198000, checkpoints_per_probe=2,
            counterfactual=True, deployment_selected_path_timing=False),
        optimizer=ARM_CONFIGS['B'], optimizer_parameters='new policy only; frozen F/probes excluded',
        schedule=schedule_identity, global_epochs=[41, 80], epochs=40, updates_per_epoch=240,
        successful_updates=9600, native_presentations=7200000, successful_leaf_view_rows=38400000,
        batch_size=256, tasks_per_update=3, effective_batch=768, tail_per_task=16, effective_tail=48,
        gradients='same-state FP32 task mean/3, one clip1, one step; whole triple RNG-restored retry<=20',
        dropout_seed='r64.dropout_seed(9600+successful_before); new seed2 policy, no historical mask-equivalence claim',
        evaluation='hard policy only; complete native VAL every epoch FP32 batch128/tails104',
        train_evaluation='full native TRAIN every5/final/selected; sparse actual measurements only',
        checkpoint_rule='max integer original full18 VAL correct/18000, earliest exact tie, one complete averaged-x policy',
        official_arm_rule='B iff selected VAL native_correct>8959 else fixed A', inference_rule=INFERENCE_RULE,
        new_measured_solver_labels=0, completed_solver_executions=0, training_expansion=False,
        final_deployment='loading+F+every actually acquired prefix and gate+final utility+one fully restarted chosen solver',
        prefix_deduction=False, end_to_end_latency_measured=False,
        historical_accounting='R73/R67/R70/R60 historical training separate; shared inherited stage1 counted once',
        inherited_stage1=dict(updates=9600, rows=7200000, seconds=2764.217798, counted_once=True),
        test_read=False, test_authorized=False, parent_freeze_pending=True,
        historical_TEST_exposure_disclosed=True, TEST_outcomes_used_for_design=False,
        goal_status='ACTIVE; unachieved', goal_native_correct=12600, goal_native_total=18000,
        R58='deferred', eight_legacy_B_semantics='deferred/unfixed; not native70 numeric veto',
        wandb=dict(project='selector', entity='yjkds-southern-university-of-science-technology', mode='offline'))
    paths = [Path(__file__).parent / name for name in OWNED+REUSED]
    paths += [WORKSPACE / 'code/unified_selector' / name for name in ('data.py', 'registry.py')]
    external_roots = [BQ_ROOT / 'model', BQ_ROOT / 'utils', RELD_ROOT / 'CVRP',
        RELD_ROOT / 'Multi-Task/models', RELD_ROOT / 'Multi-Task/envs',
        EASY_ROOT / 'neural_solvers/methods/matnet']
    external = [path for root in external_roots for path in root.rglob('*.py')]
    external += [BQ_ROOT / 'learning/tsp/decoding.py', RELD_ROOT / 'CVRP/config.yml',
        RELD_ROOT / 'Multi-Task/utils.py', BRIDGE / 'reld_bridge_common.py', BRIDGE / 'run_reld_cvrp_single_bridge.py',
        EASY_ROOT / 'neural_solvers/envs/ATSPEnv.py', EASY_ROOT / 'utils/utils.py']
    sources = {}
    for path in paths:
        destination = ROOT / 'source_launch' / path.relative_to(WORKSPACE)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
        sources[str(path)] = file_hash(path)
    for path in sorted(set(external)):
        destination = ROOT / 'source_launch/external' / path.relative_to(Path('/public/home/shiys'))
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
        sources[str(path)] = file_hash(path)
    shutil.copy2(ROOT / 'approved_plan.md', ROOT / 'source_launch/approved_plan.md')
    protocol['source_identities'] = sources
    atomic_json(ROOT / 'source_launch/identities.json', sources)
    atomic_json(ROOT / 'protocol.json', protocol)
    atomic_json(ROOT / 'control_reuse.json', protocol['fixed_control'])
    atomic_json(ROOT / 'schedule_seed2.json', dict(original=schedule_identity, used_global_epochs=[41, 80]))
    DIRECTORY.mkdir(exist_ok=True)
    atomic_json(DIRECTORY / 'args.json', dict(protocol, arm='B'))
    atomic_json(ROOT / 'formal_started.json', dict(launch, utc=now(), phases=['feature_generation', 'fixed40E_policy'],
                                                 automatic_restart=False, test_authorized=False))
    return protocol, schedule, r48


def restore_base(protocol, device, record_root=ROOT):
    checkpoint = torch.load(BASE, map_location='cpu', weights_only=False)
    if (checkpoint['epoch'] != 23 or checkpoint['global_epoch'] != 63 or checkpoint['native_correct'] != 8959
            or checkpoint['native_total'] != 18000 or checkpoint['parameter_mode'] != 'averaged x'
            or checkpoint['checkpoint_mode'] != 'eval' or not checkpoint['complete_shared_selector']):
        raise ValueError('Exact complete R73B23/global63 averaged-x base required')
    params = checkpoint['args']['model_params']
    if json.loads(json.dumps(params)) != protocol['model_params']:
        raise ValueError('Frozen base constructor identity changed')
    model = R74Model(params)
    model.base.load_state_dict(checkpoint['model'], strict=True)
    for field in ('mean', 'std'):
        if not torch.equal(getattr(model.base.instance_encoder.geometry_residual, field),
                           torch.tensor(protocol['geometry_stats'][field], dtype=torch.float32)):
            raise ValueError('All original normalization buffers must be retained')
    model.to(device).eval()
    counts = dict(full_model_parameters=sum(p.numel() for p in model.parameters()),
        frozen_base_parameters=sum(p.numel() for p in model.base.parameters()),
        trainable_policy_parameters=sum(p.numel() for p in model.policy.parameters()),
        frozen_probe_parameters_not_in_policy_model=True)
    atomic_json(record_root / 'initialization.json', dict(utc=now(), checkpoint=str(BASE), complete_strict_restore=True,
        epoch=23, global_epoch=63, base_parameter_mode='averaged x', trained_head_retained=True,
        original_buffers_retained=True, normalization_refitted=False, new_parameter_seed=2, **counts))
    initial_path = DIRECTORY / 'initial.pt' if record_root == ROOT else record_root / 'initial.pt'
    atomic_save(initial_path, dict(model=cpu_state(model), args=protocol, parameter_counts=counts))
    return model, counts


def generate(protocol, r48, model, counts, launch, wb, device, continuation=None):
    started, totals, per_task, loaded = time.monotonic(), Counter(), {}, {}
    record_root, first_batch_number = ROOT, 1
    if continuation is not None:
        record_root = ROOT / launch.get('continuation_record', 'continuation')
        totals.update(continuation['initial_counters'])
        per_task.update(continuation['completed_tasks'])
        loaded.update(continuation['loaded_probes'])
        first_batch_number += totals['query_batches']
    base_execution = continuation.get('base_execution') if continuation is not None else None
    partial_base = continuation.get('partial_base') if continuation is not None else None
    def account_base(rows, reused=False, replayed=False):
        if base_execution is None:
            return
        base_execution['new_forward_rows'] += 0 if reused else rows
        base_execution['new_forward_batches'] += int(not reused)
        base_execution['reused_failed_output_rows'] += rows if reused else 0
        base_execution['replayed_uncommitted_rows'] += rows if replayed else 0
        base_execution['actual_cumulative_forward_rows'] = base_execution['prior_total_forward_rows']+base_execution['new_forward_rows']
        base_execution['actual_cumulative_forward_batches'] = base_execution['prior_total_forward_batches']+base_execution['new_forward_batches']
        atomic_json(record_root / 'base_execution_progress.json', dict(utc=now(), **base_execution))
    def on_loaded(name, probe_model, recipe):
        path = ROOT / 'probe_states'
        path.mkdir(exist_ok=True)
        state_path = path / (name+'.pt')
        restored = continuation is not None and name in continuation['loaded_probes']
        if restored:
            saved = torch.load(state_path, map_location='cpu', weights_only=False)
            if saved['recipe'] != recipe or not saved['fp32'] or not saved['frozen']:
                raise ValueError('Original frozen probe snapshot contract changed: '+name)
            probe_model.load_state_dict(saved['model'], strict=True)
        else:
            if state_path.exists():
                raise ValueError('Preserve existing frozen probe snapshot: '+name)
            atomic_save(state_path, dict(model=cpu_state(probe_model), recipe=recipe, fp32=True, frozen=True))
        loaded[name] = dict(parameters=sum(p.numel() for p in probe_model.parameters()),
            trainable_parameters=0, recipe=recipe, state=str(state_path), loaded_utc=now(),
            original_frozen_snapshot_strictly_restored=restored)
        atomic_json(record_root / 'probe_initialization.json', loaded)
    probes = ProbeRecipes(device, r48, on_loaded)
    raw = {split: {problem: read_raw_costs(problem, split) for problem in PROBLEMS} for split in ('train', 'val')}
    if len(GLOBAL_SOLVERS) != 22 or sum(len(value['pool_ids']) for value in raw['train'].values()) != 128:
        raise ValueError('All original full legal pools must remain intact')
    manifest = {split: {problem: {key: value[key] for key in ('pool', 'pool_ids', 'data_hash', 'label_hash')}
        for problem, value in entries.items()} for split, entries in raw.items()}
    expected_manifest = json.loads((REFERENCE / 'data_manifest.json').read_text())
    if manifest != expected_manifest:
        raise ValueError('Original native data/label/candidate identities changed')
    if continuation is not None and manifest != json.loads((ROOT / 'data_manifest.json').read_text()):
        raise ValueError('Continuation data/label/candidate identity differs from original formal pass')
    atomic_json(record_root / 'data_manifest.json', manifest)
    torch.cuda.reset_peak_memory_stats(device)
    for split in ('train', 'val'):
        for problem in PROBLEMS:
            if split+'/'+problem in per_task:
                continue
            status('formal_feature_generation', launch, split=split, problem=problem, counters=dict(totals), successful_updates=0)
            native = NativeInputs(problem, split, REFERENCE / 'geometry_cache')
            probes.ensure(problem)
            task_root = ROOT / 'feature_cache' / split / problem
            if partial_base is not None and split == 'train' and problem == 'ATSP':
                task_root = record_root / 'feature_cache' / split / problem
            task_root.mkdir(parents=True, exist_ok=False)
            task_records, task_counts, task_started = [], Counter(), time.monotonic()
            for (n, starts, positives), rows in sorted(native.groups.items()):
                writer = GroupWriter(task_root / f'n{n}_s{starts}_p{positives}', rows, n, starts,
                                     raw[split][problem]['pool_ids'], problem)
                for start in range(0, len(rows), 16):
                    stop = min(start+16, len(rows))
                    source_rows = rows[start:stop]
                    synchronize(device)
                    batch_started = time.monotonic()
                    batch = native.gather(source_rows, device)
                    if (batch['pool_ids'].tolist() != raw[split][problem]['pool_ids']
                            or not np.array_equal(batch['ind'].cpu().numpy(), raw[split][problem]['winner'][source_rows])):
                        raise ValueError('Native source/target/full-pool alignment changed')
                    partial_match = (partial_base is not None and split == 'train' and problem == 'ATSP'
                                     and (n, starts, positives, start, stop) == (20, 20, 0, 0, 16))
                    reused = partial_match and partial_base['reuse_available']
                    if reused:
                        directory = Path(partial_base['directory'])
                        previous = json.loads((directory / 'metadata.json').read_text())
                        source = np.load(directory / 'source_indices.npy', allow_pickle=False)
                        if (not np.array_equal(source, rows) or previous['complete'] or previous['rows'] != len(rows)
                                or previous['pool_ids'] != raw[split][problem]['pool_ids']):
                            raise ValueError('Preserved partial ATSP base row alignment changed')
                        base = {}
                        for key in ('nodes', 'features', 'h', 'base_logits'):
                            array = np.load(directory / (key+'.npy'), mmap_mode='r', allow_pickle=False)
                            spec = writer.metadata['specs'][key]
                            if (list(array.shape) != spec['shape'] or array.dtype.name != spec['dtype']
                                    or not np.isfinite(array[start:stop]).all()):
                                raise ValueError('Persisted partial base features are invalid: '+key)
                            base[key] = torch.from_numpy(np.array(array[start:stop], copy=True)).to(device)
                        account_base(stop-start, reused=True)
                        atomic_json(record_root / 'partial_base_reuse.json', dict(utc=now(), directory=str(directory),
                            source_indices=source_rows, rows=stop-start, finite=True, row_alignment_verified=True,
                            base_forward_repeated=False, original_partial_files_unchanged=True))
                    else:
                        base = model.extract_base(batch)
                        account_base(stop-start, replayed=partial_match)
                    writer.write_base(start, stop, base)
                    observations, action_calls, decoder_calls = [], 0, 0
                    for probe in probe_ids(problem):
                        trajectory = probes.start(problem, native.instances, source_rows, probe)
                        if trajectory.n != n or trajectory.starts != starts or trajectory.node_count != base['nodes'].shape[1]:
                            raise ValueError('Actual native start/node geometry differs from the input-only group')
                        for checkpoint in (0, 1):
                            value = trajectory.acquire_increment(base['nodes'])
                            writer.write_observation(start, stop, probe, checkpoint, value)
                            observations.append(dict(value['receipt'], checkpoint=checkpoint))
                        action_calls += trajectory.model_calls
                        decoder_calls += trajectory.decoder_calls
                        del trajectory, value
                    synchronize(device)
                    rows_count, probe_count = stop-start, len(probe_ids(problem))
                    totals.update(base_query_rows=rows_count, query_batches=1, trajectory_initializations=rows_count*probe_count,
                        start_trajectories=rows_count*probe_count*starts, model_action_calls=action_calls,
                        decoder_calls=decoder_calls, completed_trajectories=0)
                    totals[split+'_rows'] += rows_count
                    totals[split+'_initializations'] += rows_count*probe_count
                    task_counts.update(rows=rows_count, query_batches=1, trajectory_initializations=rows_count*probe_count,
                                       model_action_calls=action_calls, decoder_calls=decoder_calls)
                    metrics = dict(utc=now(), split=split, problem=problem, problem_id=P2I[problem],
                        source_indices=source_rows, rows=rows_count, true_n=n, starts=starts,
                        observations=observations, counters=dict(totals), seconds=time.monotonic()-batch_started,
                        wall_seconds=time.monotonic()-started, counterfactual_generation=True, **memory(device))
                    if continuation is not None:
                        elapsed = time.monotonic()-continuation['process_started']
                        metrics.update(continuation_generation_seconds=time.monotonic()-started,
                            continuation_formal_seconds=elapsed,
                            original_failed_step_seconds=continuation.get('original_step_elapsed_seconds', continuation['original_elapsed_seconds']),
                            charged_prior_Slurm_wall_seconds=continuation['original_elapsed_seconds'],
                            base_execution=dict(base_execution) if base_execution is not None else None,
                            wall_seconds=continuation['original_elapsed_seconds']+elapsed,
                            continuation_counters={key: value-continuation['initial_counters'].get(key, 0)
                                                   for key, value in totals.items()})
                    append(record_root / 'feature_batches.jsonl', metrics)
                    if totals['query_batches'] == 1:
                        writer.flush()
                        first = dict(metrics, launch=launch, parameter_counts=counts, loaded_probes=loaded,
                            successful_updates=0, first_student_update_observed=False, executed_solvers=0)
                        atomic_json(ROOT / 'first_formal_extraction_batch.json', first)
                        atomic_json(ROOT / 'handoff.json', dict(status='formal acquisition cache generation running',
                            command='bash code/V4/run_v4_r74.sh train', tmux='r74_train', launch=launch,
                            first_formal_extraction_batch=first, cpu_check=json.loads((ROOT / 'cpu_check.json').read_text()),
                            storage=protocol['storage'], test_authorized=False, goal_status='ACTIVE; unachieved'))
                        print('[R74 first FORMAL extraction batch] '+json.dumps(first), flush=True)
                    if continuation is not None and totals['query_batches'] == first_batch_number:
                        writer.flush()
                        first = dict(metrics, launch=launch, parameter_counts=counts, loaded_probes=loaded,
                            successful_updates=0, first_student_update_observed=False, executed_solvers=0,
                            completed_prefix_tasks=list(continuation['completed_tasks']))
                        atomic_json(record_root / 'first_actual_atsp_batch.json', first)
                        atomic_json(record_root / 'handoff.json', dict(status='continuation formal acquisition running',
                            command=launch['command'], tmux=launch['tmux_session'], launch=launch,
                            first_actual_atsp_batch=first, original_job_status='FAILED 1:0; never reclassified',
                            amendment=str(record_root / 'execution_amendment.json'),
                            test_authorized=False, goal_status='ACTIVE; unachieved'))
                        print('[R74 first ACTUAL continuation ATSP batch] '+json.dumps(first), flush=True)
                    if totals['query_batches'] == first_batch_number or totals['query_batches'] % 32 == 0:
                        writer.flush()
                        status('formal_feature_generation', launch, last_batch=metrics, successful_updates=0)
                        print(f'[R74 extraction] {split}/{problem} rows={totals["base_query_rows"]}/198000 '
                              f'initializations={totals["trajectory_initializations"]}/363000', flush=True)
                        if wb is not None:
                            wb.log({'phase': 'feature_generation', **{'generation/'+key: value for key, value in totals.items()},
                                'generation/wall_seconds': metrics['wall_seconds'],
                                'generation/peak_allocated_bytes': metrics['peak_allocated_bytes']})
                    del batch, base
                task_records.append(writer.finish())
                del writer
            metadata = dict(problem=problem, split=split, rows=native.size, groups=task_records,
                counters=dict(task_counts), complete=True, seconds=time.monotonic()-task_started,
                source_identity=manifest[split][problem], labels_used_in_features=False)
            atomic_json(task_root / 'metadata.json', metadata)
            per_task[split+'/'+problem] = metadata
            del native
            gc.collect()
    if (totals['base_query_rows'] != 198000 or totals['train_rows'] != 180000 or totals['val_rows'] != 18000
            or totals['train_initializations'] != 330000 or totals['val_initializations'] != 33000
            or totals['completed_trajectories'] != 0):
        raise ValueError('Formal generation must retain every original row and all locked counterfactual trajectories')
    receipt = dict(completed_utc=now(), counters=dict(totals), per_task=per_task,
        seconds=time.monotonic()-started, memory=memory(device), probes=loaded, complete=True,
        executed_solvers=0, completed_solver_labels_added=0, selected_path_latency_measured=False)
    if continuation is not None:
        elapsed = time.monotonic()-continuation['process_started']
        receipt.update(seconds=continuation['original_elapsed_seconds']+elapsed,
            continuation_generation_seconds=time.monotonic()-started, continuation_formal_seconds=elapsed,
            original_failed_step_seconds=continuation.get('original_step_elapsed_seconds', continuation['original_elapsed_seconds']),
            charged_prior_Slurm_wall_seconds=continuation['original_elapsed_seconds'],
            original_job_status='FAILED 1:0; never reclassified',
            initial_counters=continuation['initial_counters'], completed_prefix_tasks=list(continuation['completed_tasks']),
            continuation_counters={key: value-continuation['initial_counters'].get(key, 0)
                                   for key, value in totals.items()},
            execution_amendment=str(record_root / 'execution_amendment.json'))
        if base_execution is not None:
            expected_executed = totals['base_query_rows']+base_execution['prior_failed_forward_rows']-base_execution['reused_failed_output_rows']
            if base_execution['actual_cumulative_forward_rows'] != expected_executed:
                raise ValueError('Actual base forwards and failed/reused work accounting differ')
            receipt['base_execution'] = dict(base_execution)
    atomic_json(ROOT / 'feature_cache_complete.json', receipt)
    del probes
    gc.collect()
    torch.cuda.empty_cache()
    atomic_json(ROOT / 'probes_released.json', dict(utc=now(), before_optimizer_construction=True,
        no_live_probe_states=True, frozen_base_retained_as_B_component=True))
    caches = {split: {problem: CachedTask(per_task[split+'/'+problem], raw[split][problem])
                     for problem in PROBLEMS} for split in ('train', 'val')}
    return caches, raw, receipt


@torch.no_grad()
def evaluate(model, caches, raw, device, prediction_dir=None):
    model.eval()
    if prediction_dir is not None:
        prediction_dir.mkdir(parents=True, exist_ok=True)
    started, results, paths, gate_values, batches = time.monotonic(), {}, Counter(), {}, 0
    for problem in PROBLEMS:
        cache, scores, hard = caches[problem], [], []
        for start in range(0, cache.size, 128):
            stop = min(start+128, cache.size)
            batch = cache.gather(np.arange(start, stop), device)
            output = model(batch)
            scores.append(output['logits'].cpu().numpy())
            hard.extend(output['hard_paths'].cpu().tolist())
            for key, value in output['gate_probabilities'].items():
                gate_values.setdefault(problem+'/'+key, []).extend(value.cpu().tolist())
            batches += 1
            del batch, output
        scores = np.concatenate(scores)
        result, pred = policy_metrics(scores, raw[problem])
        result['native_correct'] = int(np.count_nonzero(pred == raw[problem]['winner']))
        results[problem] = result
        paths.update({problem+'/'+str(key): value for key, value in Counter(hard).items()})
        if prediction_dir is not None:
            np.savez_compressed(prediction_dir / (problem+'.npz'), logits=scores, pred=pred,
                winner=raw[problem]['winner'], costs=raw[problem]['costs'], pool_ids=raw[problem]['pool_ids'],
                pool=raw[problem]['pool'], indices=np.arange(cache.size), hard_paths=np.asarray(hard),
                true_nodes=np.asarray([cache.groups[int(group)]['record']['true_n'] for group in cache.locations[:, 0]]))
    total = sum(value['n'] for value in results.values())
    correct = sum(value['native_correct'] for value in results.values())
    macro = aggregate(results)
    macro['macro_top1'] = correct/total
    return dict(per_problem=results, macro=macro, families=families(results), native_correct=correct,
        native_total=total, hard_path_counts=dict(paths), gate_probabilities={key: float(np.mean(value)) for key, value in gate_values.items()},
        query_batches=batches, hard_policy=True, fp32=True, seconds=time.monotonic()-started,
        timing_scope='cached hard-policy evaluation; not live acquisition or completed-solver latency')


def train(protocol, schedule, model, caches, raw, generation, launch, wb, device):
    set_seed(2)
    policy = model.policy
    optimizer = make_optimizer(policy, 'B')
    scaler = torch.amp.GradScaler('cuda', enabled=True)
    layout = GradientLayout(policy)
    atomic_json(DIRECTORY / 'optimizer_initialization.json', dict(utc=now(), fresh=True,
        parameter_elements=layout.receipt['total_elements'], frozen_base_in_optimizer=False,
        frozen_probes_in_optimizer=False, config=ARM_CONFIGS['B'], state_entries=len(optimizer.state)))
    successful = presentations = views = retries = growth = 0
    history, best, best_model, best_path, lr_trace = [], None, None, None, []
    attempted, active_counts, per_task_updates = Counter(), Counter(), Counter()
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats(device)
    def save_last(epoch, step):
        was_training = model.training
        evaluation_mode(model, optimizer, 'B')
        atomic_save(DIRECTORY / 'last.pt', dict(model=cpu_state(model), optimizer=optimizer.state_dict(),
            scaler=scaler.state_dict(), args=protocol, epoch=epoch, global_epoch=40+epoch, next_step=step,
            parameter_mode='averaged x', checkpoint_mode='eval', successful_updates=successful,
            presentations=presentations, successful_leaf_view_rows=views, attempted=dict(attempted),
            amp_retries=retries, rng=capture_rng(), history=history, best=best, best_model=best_model,
            best_checkpoint=str(best_path) if best_path else None, lr_trace=lr_trace, scale_growth_successes=growth,
            updates_by_problem=dict(per_task_updates), active_gradient_counts=dict(active_counts)))
        atomic_json(DIRECTORY / 'history.json', history)
        if was_training:
            training_mode(model, optimizer, 'B')
    for epoch in range(1, 41):
        training_mode(model, optimizer, 'B')
        totals, task_values, epoch_started = Counter(), {}, time.monotonic()
        for step in range(240):
            entries = triple_indices(schedule, 40+epoch-1, step)
            batches = [caches['train'][problem].gather(indices, device) for problem, indices in entries]
            expected_rows = 16 if step//6 == 39 else 256
            if any(len(batch['winner']) != expected_rows for batch in batches):
                raise ValueError('Real native256 task rows and16-row tails must be retained')
            native_seed = dropout_seed(9600+successful)
            set_seed(native_seed)
            expected_lr = configure_update_lr(optimizer, 'B', successful)
            consistency = .35 * min((successful+1)/720, 1.)
            synchronize(device)
            update_started = time.monotonic()
            values = update_triple(policy, optimizer, scaler, batches, consistency, layout, growth)
            synchronize(device)
            successful += 1
            rows = sum(len(batch['winner']) for batch in batches)
            presentations += rows
            views += values['successful_leaf_view_rows']
            growth, retries = values['scale_growth_successes'], retries+values['retries']
            attempted.update(values['attempted'])
            active_counts.update(values.pop('active_gradient_names'))
            lr = observed_lr(optimizer, 'B', expected_lr)
            lr_trace.append(lr)
            for task, (problem, indices) in enumerate(entries):
                per_task_updates[problem] += 1
                item = task_values.setdefault(problem, Counter())
                item['rows'] += len(indices)
                for key in ('loss', 'native_ce', 'kl'):
                    item[key] += values['task_losses'][task][key]*len(indices)
            metrics = dict(utc=now(), epoch=epoch, global_epoch=40+epoch, update_in_epoch=step+1,
                successful_update=successful, native_presentations=presentations, successful_leaf_view_rows=views,
                task_indices={problem: indices.tolist() for problem, indices in entries}, rows=rows,
                task_rows=[len(batch['winner']) for batch in batches], dropout_seed=native_seed,
                lr=lr, consistency_weight=consistency, cumulative_retries=retries, cumulative_attempted=dict(attempted),
                seconds=time.monotonic()-update_started, wall_seconds=time.monotonic()-started,
                parameter_mode='y', **values, **memory(device))
            append(DIRECTORY / 'updates.jsonl', metrics)
            for key in ('loss', 'native_ce', 'kl'):
                totals[key] += values[key]*rows
            totals['rows'] += rows
            totals['grad_norm'] += values['grad_norm']
            totals['clip_coefficient'] += values['clip_coefficient']
            totals['retries'] += values['retries']
            if successful == 1:
                atomic_json(ROOT / 'first_formal_optimization_update.json', dict(metrics, launch=launch))
            if successful == 1 or (step+1) % 60 == 0:
                save_last(epoch, step+1)
                status('policy_training', launch, last_metrics=metrics, feature_generation_complete=True)
                print(f'[R74 B] E{epoch}/40 u{successful} CE={values["native_ce"]:.6f} '
                      f'KL={values["kl"]:.6f} L={values["loss"]:.6f} retries={retries}', flush=True)
                if wb is not None:
                    wb.log({'phase': 'policy_training', **{'update/'+key: metrics[key] for key in
                        ('successful_update', 'loss', 'native_ce', 'kl', 'grad_norm', 'clip_coefficient',
                         'lr', 'scale', 'cumulative_retries', 'wall_seconds', 'peak_allocated_bytes')}})
            del batches
        if totals['rows'] != 180000 or successful != epoch*240 or any(value['rows'] != 10000 for value in task_values.values()):
            raise ValueError('Every epoch must contain all180000 original native rows')
        evaluation_mode(model, optimizer, 'B')
        validation = evaluate(model, caches['val'], raw['val'], device, DIRECTORY / f'val_predictions/epoch{epoch:03d}')
        training = evaluate(model, caches['train'], raw['train'], device, DIRECTORY / f'train_predictions/epoch{epoch:03d}') if epoch % 5 == 0 else None
        if validation['native_total'] != 18000 or validation['query_batches'] != 144:
            raise ValueError('Only full18 hard-policy validation may select the complete policy')
        optimization = {key: totals[key]/180000 for key in ('loss', 'native_ce', 'kl')}
        optimization.update(grad_norm=totals['grad_norm']/240, clip_coefficient=totals['clip_coefficient']/240,
            retries=int(totals['retries']), lr=lr, seconds=time.monotonic()-epoch_started)
        row = dict(epoch=epoch, global_epoch=40+epoch, successful_updates=successful, native_presentations=presentations,
            successful_leaf_view_rows=views, attempted=dict(attempted), optimization=optimization,
            task_optimization={problem: {key: value[key]/value['rows'] for key in ('loss', 'native_ce', 'kl')}
                               for problem, value in task_values.items()}, train=training, val=validation, parameter_mode='averaged x')
        history.append(row)
        if prefer_checkpoint(validation, None if best is None else best['val']):
            best, best_model = row, cpu_state(model)
            folder = DIRECTORY / 'best_checkpoints'
            folder.mkdir(exist_ok=True)
            best_path = folder / f'epoch{epoch:03d}.pt'
            atomic_save(best_path, dict(model=best_model, args=protocol, epoch=epoch, global_epoch=40+epoch,
                native_correct=validation['native_correct'], native_total=18000, parameter_mode='averaged x',
                checkpoint_mode='eval', complete_policy=True, frozen_base_included=True,
                probe_states=str(ROOT / 'probe_states'), inference_rule=INFERENCE_RULE))
            atomic_json(DIRECTORY / 'best_checkpoint.json', dict(path=str(best_path), epoch=epoch, global_epoch=40+epoch,
                native_correct=validation['native_correct'], native_total=18000, parent_freeze_pending=True))
            atomic_json(DIRECTORY / 'best_eval.json', best)
        save_last(epoch, 240)
        append(ROOT / 'curves/B_metrics.jsonl', row)
        print(f'[R74 VAL] E{epoch} hard native_correct={validation["native_correct"]}/18000', flush=True)
        if wb is not None:
            wb.log({'phase': 'cached_hard_policy_evaluation', 'epoch': epoch,
                    **{'val/'+key: value for key, value in validation['macro'].items()},
                    'val/native_correct': validation['native_correct'],
                    **{'gate/'+key: value for key, value in validation['gate_probabilities'].items()},
                    **{'path/'+key: value for key, value in validation['hard_path_counts'].items()}})
    if (successful != 9600 or presentations != 7200000 or views != 38400000 or len(history) != 40
            or set(per_task_updates) != set(PROBLEMS) or any(value != 1600 for value in per_task_updates.values())):
        raise ValueError('Exact40E native/leaf-view budget must complete')
    evaluation_mode(model, optimizer, 'B')
    atomic_save(DIRECTORY / 'final.pt', dict(model=cpu_state(model), args=protocol, epoch=40, global_epoch=80,
        parameter_mode='averaged x', checkpoint_mode='eval', complete_policy=True, inference_rule=INFERENCE_RULE))
    selected_training = best['train']
    if selected_training is None:
        model.load_state_dict(best_model, strict=True)
        model.eval()
        selected_training = evaluate(model, caches['train'], raw['train'], device, DIRECTORY / 'train_predictions/best')
    best_eval = dict(best, train=selected_training, validation_recount=False)
    atomic_json(DIRECTORY / 'best_eval.json', best_eval)
    identity = dict(path=str(best_path), epoch=best['epoch'], global_epoch=best['global_epoch'],
        native_correct=best['val']['native_correct'], native_total=18000, parameter_mode='averaged x', complete_policy=True)
    result = dict(best=best_eval, final=history[-1], best_checkpoint=identity, epochs_completed=40,
        successful_updates=successful, native_presentations=presentations, successful_leaf_view_rows=views,
        attempted=dict(attempted), amp_retries=retries, wall_seconds=time.monotonic()-started,
        memory=memory(device), generation_wall_seconds=generation['seconds'], generation_counters=generation['counters'],
        proposed_official_arm='B' if identity['native_correct'] > 8959 else 'A',
        official_arm_rule=protocol['official_arm_rule'], checkpoint_rule=protocol['checkpoint_rule'],
        inference_rule=INFERENCE_RULE, parent_freeze_pending=True, test_authorized=False, goal_status='ACTIVE; unachieved')
    atomic_json(DIRECTORY / 'parameter_update_counts.json', dict(gradient_present=dict(active_counts),
        task_updates=dict(per_task_updates), new_policy_only=True, frozen_base_updates=0, frozen_probe_updates=0))
    atomic_json(DIRECTORY / 'result.json', result)
    atomic_json(ROOT / 'validation_candidates.json', dict(B=identity, A=protocol['fixed_control'],
        proposed_official_arm=result['proposed_official_arm'], parent_freeze_pending=True, test_authorized=False))
    atomic_json(ROOT / 'training_complete.json', dict(completed_utc=now(), **{key: result[key] for key in
        ('epochs_completed', 'successful_updates', 'native_presentations', 'successful_leaf_view_rows', 'amp_retries',
         'best_checkpoint', 'proposed_official_arm')}, test_authorized=False, parent_freeze_pending=True))
    status('training_complete', launch, successful_updates=successful, proposed_official_arm=result['proposed_official_arm'])


def cpu_check():
    if any((ROOT / name).exists() for name in ('cpu_check_started.json', 'cpu_check.json', 'formal_started.json')):
        raise ValueError('Preserve original R74 CPU invocation; no repeat')
    atomic_json(ROOT / 'cpu_check_started.json', dict(utc=now(), pid=os.getpid(), invocation=1,
        command='bash code/V4/run_v4_r74.sh cpu-check'))
    for name in OWNED:
        if name.endswith('.py'):
            path = Path(__file__).parent / name
            compile(path.read_text(), str(path), 'exec')
    import subprocess
    subprocess.run(['bash', '-n', str(Path(__file__).parent / 'run_v4_r74.sh')], check=True)
    from .r74_cpu_check import focused_check
    receipt = focused_check()
    receipt.update(completed_utc=now(), pid=os.getpid(), invocation=1, source_hashes=source_identity(),
                   command='bash code/V4/run_v4_r74.sh cpu-check', combined_owned_source_syntax=True)
    atomic_json(ROOT / 'cpu_check.json', receipt)
    print('[R74 one synthetic CPU] '+json.dumps(receipt), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('cpu-check', 'train'), required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    wb, launch, started = None, None, time.monotonic()
    try:
        if args.stage == 'cpu-check':
            cpu_check()
            return
        if (os.environ.get('CUDA_VISIBLE_DEVICES') != GPU_UUID or os.environ.get('SLURM_JOB_ID') != '1465'
                or socket.gethostname().split('.')[0] != 'gpu03'):
            raise ValueError('Only original allocation1465/gpu03/assigned UUID is authorized')
        launch = json.loads((ROOT / 'launch_receipt.json').read_text())
        launch.update(trainer_pid=os.getpid(), started_utc=now(), command='bash code/V4/run_v4_r74.sh train',
            slurm_step=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'], status='running')
        atomic_json(ROOT / 'trainer_launch.json', launch)
        protocol, schedule, r48 = prepare(launch)
        import wandb
        wb = wandb.init(**protocol['wandb'], name='R74_B_bounded_acquisition_seed2', config=protocol, dir=str(ROOT),
                        id='r74-b-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S'))
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision('highest')
        set_seed(2)
        device = 'cuda:0'
        model, counts = restore_base(protocol, device)
        caches, raw, generation = generate(protocol, r48, model, counts, launch, wb, device)
        train(protocol, schedule, model, caches, raw, generation, launch, wb, device)
        launch.update(status='complete', completed_utc=now(), total_formal_wall_seconds=time.monotonic()-started)
        atomic_json(ROOT / 'trainer_launch.json', launch)
    except BaseException as error:
        failure = dict(utc=now(), requested_stage=args.stage, pid=os.getpid(), error_type=type(error).__name__,
            error=str(error), traceback=traceback.format_exc(), automatic_restart=False,
            slurm_step=os.environ.get('SLURM_JOB_ID', '')+'.'+os.environ.get('SLURM_STEP_ID', ''),
            update_failure_receipt=getattr(error, 'r74_update_receipt', None))
        append(ROOT / ('cpu_check_failures.jsonl' if args.stage == 'cpu-check' else 'failures.jsonl'), failure)
        if args.stage == 'train':
            atomic_json(ROOT / 'pipeline_status.json', dict(failure, stage='failed_preserved_no_retry'))
        raise
    finally:
        if wb is not None:
            wb.finish()


if __name__ == '__main__':
    main()

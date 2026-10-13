"""One R71 joint performance-mixture arm; original TRAIN/VAL only."""

import argparse
import gc
import json
import os
import shutil
import socket
import sys
import time
from collections import Counter
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from code.unified_selector.registry import GLOBAL_SOLVERS, PROBLEMS
from .multitask_probe import Tee, file_hash, gather_batch, state_hash
from .performance_experiment import SOURCE_FILES
from .r40_experiment import aggregate, families
from .r42_experiment import prepare_data
from .r43_experiment import policy_metrics
from .r59_experiment import GPU_UUID, REFERENCE, WORKSPACE, atomic_json, atomic_save, check_original_scope
from .r64_experiment import dropout_seed, executed_updates, paired_triples, prefer_checkpoint, synchronize, triple_indices
from .r69_experiment import append_jsonl, utc_now
from .r69_model import OPTIMIZER_CONFIG, PARAMETER_MODE, cpu_state, make_optimizer, model_inputs, update_lr
from .r71_gradients import GradientLayout, update_triple
from .r71_model import CE_FLOOR, INFERENCE_RULE, fit_normalizer, make_bank, standardized_targets
from .train import configure_torch, print_eval, set_seed
from .training_monitor import capture_rng, restore_rng


ROOT = WORKSPACE / 'code/V4/runs/R71_joint_performance_mixture'
CONTROL_ROOT = WORKSPACE / 'code/V4/runs/R70_latent_mixup'
INITIAL_ROOT = WORKSPACE / 'code/V4/runs/R69_task_specific_finetuning'
RUN_NAME = 'B_joint_mixture_seed2'
OWNED_SOURCES = ('r71_model.py', 'r71_gradients.py', 'r71_experiment.py', 'r71_cpu_check.py',
                 'run_v4_r71.sh', 'R71_README.md')
REUSED_SOURCES = ('r64_model.py', 'r64_experiment.py', 'r64_gradients.py', 'r65_gradients.py',
    'r45_experiment.py', 'r59_experiment.py', 'r43_experiment.py', 'r42_experiment.py',
    'r40_experiment.py', 'pairwise_objective.py', 'pairwise_selector.py', 'solver_code_encoder.py',
    'r67_model.py', 'r67_experiment.py', 'r68_model.py', 'r68_gradient.py', 'r69_model.py',
    'r69_gradients.py', 'r69_experiment.py')
UPDATES, PRESENTATIONS, VIEW_ROWS, CONTROL_CORRECT = 9600, 7200000, 14400000, 8709
LOSS_FIELDS = ('loss', 'nll_per_dimension', 'joint_nll')


def source_hashes():
    paths = [Path(__file__).parent / name for name in sorted(set(SOURCE_FILES + OWNED_SOURCES + REUSED_SOURCES))]
    paths += [WORKSPACE / 'code/unified_selector' / name for name in ('registry.py', 'data.py')]
    return {str(path.relative_to(WORKSPACE)): file_hash(path) for path in paths}


def planned_config():
    return dict(seed=2, experiment='joint performance-mixture likelihood and winner-probability inference package',
        new_arms={'B': RUN_NAME}, reused_control='fixed complete R70B23/global63/8709 of18000',
        control_reuse='protocol equivalence, not independent replication',
        epochs=40, updates_per_epoch=240, successful_updates=UPDATES, presentations=PRESENTATIONS,
        successful_view_rows=VIEW_ROWS, active_updates_per_branch=1600, selected_rows_per_branch=400000,
        batch_size=256, tasks_per_update=3, effective_batch_size=768, tail_per_task=16,
        final_effective_batch_size=48, eval_batch_size=128, problems=list(PROBLEMS), solver_names=list(GLOBAL_SOLVERS),
        model='18 complete full-information encode_pairs backbones; d128/instance4/joint2/handcrafted34+ID; scalar head deleted',
        routing='one known original problem_id activates one complete branch; no costs/winners in forward',
        head='Linear(384K,128), GELU, Linear(128,8*(1+D+D*(D+1)/2)); D=K-1; no head dropout',
        components=8, covariance='full scale_tril; diagonal softplus(raw)+1e-3',
        likelihood_backend='installed torch.distributions.MixtureSameFamily(Categorical,MultivariateNormal(scale_tril))',
        head_seed='710002+problem_id; ordinary first Linear; zero final weights; uniform logits; unit covariance; mean biases N(0,.1^2); outer RNG preserved',
        initial_zero_backbone_gradients='intended consequence of zero output weights; production initialization unchanged',
        initialization='ORIGINAL R67B stage1 terminal averaged-x retained trunk via original R69 pre-fine-tuning container; never trained R69/R70 weights',
        targets='original FP64 c_j-c_reference; canonical ascending global IDs; first legal solver is reference',
        target_normalization='each nonreference coordinate: original task TRAIN10000 mean/population std; max(std,1e-6); FP64 then FP32 target interface',
        normalizer_fitting_split='train only', persistent_buffers='canonical IDs, FP64 means/scales, fixed Sobol normal table, triangular indices',
        objective='mean of two ordinary backbone-dropout joint NLLs divided by D; density head/likelihood/reductions FP32 outside AMP',
        density_nll_is_winner_ce=False, auxiliary_losses=False, mixed_head_rows=0,
        optimizer=OPTIMIZER_CONFIG, parameter_mode=PARAMETER_MODE,
        learning_rate='3e-4*min(successful_update/720,1)', warmup_successful_updates=720,
        gradient_reduction='each active disjoint task FP32 gradient divided by3; one combined clip1 and one AdamW step',
        inactive='grad=None; no decay/moment/counter/parameter advancement', clip_norm=1.,
        retry='20 attempts; whole same triple/rows/RNG; scaler backoff; no optimizer/LR/cursor/successful-clock advancement',
        amp_growth='2000 successful updates; reset growth-success counter on failure',
        schedule='EXACT seed2 R67/R69/R70 global41-80 task/row/dropoutseed schedule; changed head means dropout tensors need not match A bitwise',
        dropout_successful_offset=9600, full_coverage=True, drop_last=False, microbatching=False, accumulation=False,
        full_val_eval='original18*1000 every epoch, FP32batch128/tails, fixed probability rule',
        full_train_eval='original18*10000 every5/final/selected best, FP32batch128/tails, fixed probability rule',
        checkpoint_rule='max INTEGER full18 validation correct; earliest tie; all18 branches one common epoch',
        official_arm_rule='B only when best INTEGER full18 VAL correct>8709; otherwise fixed R70B23 A',
        inference_rule=INFERENCE_RULE, sobol_vectors_per_task=2048, sobol_seed='710003+problem_id',
        sobol_uniform_clip=[2. ** -24, 1. - 2. ** -24], streamed_components=1, streamed_vectors=256,
        predicted_vectors_per_query=16384, selector_forwards_per_query=1, executed_solvers_in_training_or_eval=0,
        final_completed_solvers_per_query=1, probability_floor_for_reporting_ce_only=CE_FLOOR,
        parameter_capacity_matched=False, FLOPs_matched=False, expected_A_parameters=33173154,
        expected_B_parameters='approximately38.2M; actual factory count recorded', examples_updates_matched=True,
        retained_parameter_active_update_recipe_matched=True, extra_density_head_and_probability_compute=True,
        inherited_stage1_once=dict(updates=9600, selected_rows=7200000, wall_seconds=2764.217798),
        reused_A_training=dict(updates=9600, selected_rows=7200000, wall_seconds=3668.194599, amp_retries=5),
        capacity_update='ONE discarded TRAIN-only known worst-padded scheduled real256x3 update; entire bank Adam moments materialized; recreate formal state',
        no_per_task_best_stitching=True, cost_based_selection=False, new_labels=False, augmentation=False,
        projector_included=False, parameter_averaging=False, candidate_pruning=False,
        scheduler=None, early_stopping=False, all_parameters_trainable=True,
        original_pools_global_ids_native_winners=True, directed_ATSP_preserved=True,
        wandb=dict(project='selector', entity='yjkds-southern-university-of-science-technology', mode='offline'),
        adopted_plan=str(ROOT / 'approved_plan.md'), review_independence='same-family', acceptance_status='provisional',
        historical_test_exposure=True, new_R70_test_used_for_design_or_selection=False,
        test_read=False, test_authorized=False, automatic_test=False, automatic_analysis=False, automatic_freeze=False,
        control_retraining=False, control_checkpoint_load=False, control_recount=False, control_reselection=False,
        control_retest=False, reference_inference=False, no_automatic_restart=True, git_actions=False,
        later_test='parent later freezes selected whole bank/buffers/official arm, then authorizes ONE original full18 new B TEST regardless of improvement; reuse A TEST unchanged',
        deferred=['R58', 'eight legacy B feasible-domain semantic mismatches; disclosed/unfixed, not a native70 veto'],
        target_native_macro_top1=.70, goal_status='active; unachieved',
        assets='checkpoints/NPZ/rawupdates/W&B/cache/regenerable schedule local; parent owns one completed-version commit')


def pipeline(args, stage, **details):
    receipt = dict(recorded_utc=utc_now(), stage=stage, pid=os.getpid(),
        slurm_step=os.environ.get('SLURM_JOB_ID', '') + '.' + os.environ.get('SLURM_STEP_ID', ''), **details)
    atomic_json(args.root / 'pipeline_status.json', receipt)
    append_jsonl(args.root / 'pipeline.jsonl', receipt)


def prepare(args):
    if (args.root / 'protocol.json').exists() or (args.root / 'formal_started.json').exists():
        raise ValueError('R71 already prepared/started; preserve original artifacts; never replay')
    protocol = planned_config()
    protocol.update(source_hashes=source_hashes(), approved_plan_sha256=file_hash(args.root / 'approved_plan.md'))
    for relative in protocol['source_hashes']:
        destination = args.root / 'source_launch' / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(WORKSPACE / relative, destination)
    atomic_json(args.root / 'source_launch/hashes.json', protocol['source_hashes'])
    original_protocol = json.loads((INITIAL_ROOT / 'protocol.json').read_text())
    prior_initialization = json.loads((INITIAL_ROOT / 'initialization.json').read_text())
    source = INITIAL_ROOT / 'shared_initialization.pt'
    initial = torch.load(source, map_location='cpu', weights_only=False)['selector']
    if state_hash(initial) != prior_initialization['starting_selector_hash']:
        raise ValueError('The original PRE-fine-tuning R67B-derived initialization is required')
    trunk = {key: value for key, value in initial.items() if not key.startswith('score_head.')}
    if state_hash(trunk) != prior_initialization['inherited_trunk_hash']:
        raise ValueError('Only the original terminal stage1 averaged-x trunk may be inherited')
    atomic_save(args.root / 'backbone_initialization.pt', dict(trunk=trunk, source=str(source)))
    initialization = dict(source_container=str(source), inherited_trunk_hash=prior_initialization['inherited_trunk_hash'],
        original_prefinetuning_selector_hash=prior_initialization['starting_selector_hash'],
        old_scalar_head_deleted=True, old_scalar_head_loaded=False, trained_R69_R70_weights_loaded=False,
        inherited_stage1_updates_once=9600, inherited_stage1_presentations_once=7200000,
        inherited_stage1_wall_seconds_once=2764.217798, head_seed_rule='710002+problem_id',
        sobol_seed_rule='710003+problem_id', outer_head_initialization_rng_preserved=True,
        fresh_encoders=False, all_retained_backbone_and_new_head_trainable=True)
    atomic_json(args.root / 'initialization.json', initialization)
    inherited = json.loads((INITIAL_ROOT / 'inherited_source_identity.json').read_text())
    atomic_json(args.root / 'inherited_source_identity.json', inherited)
    selected = json.loads((CONTROL_ROOT / 'validation_candidates.json').read_text())['models']['B']
    if (selected['epoch'], selected['global_epoch'], selected['native_correct'], selected['native_total']) != (23, 63, CONTROL_CORRECT, 18000):
        raise ValueError('R71 A must be the exact already-selected complete R70B23 bank')
    history_path = CONTROL_ROOT / 'B_latent_mixup_seed2/history.json'
    result_path = CONTROL_ROOT / 'B_latent_mixup_seed2/result.json'
    history = json.loads(history_path.read_text())
    if len(history) != 40 or history[22]['val']['native_correct'] != CONTROL_CORRECT:
        raise ValueError('Fixed control requires its existing complete40E TRAIN/VAL history')
    shutil.copy2(history_path, args.root / 'reused_A_history.json')
    shutil.copy2(result_path, args.root / 'reused_A_result.json')
    control = dict(model=selected, history=str(history_path), result=str(result_path),
        reused_history=str(args.root / 'reused_A_history.json'), reused_result=str(args.root / 'reused_A_result.json'),
        identity='fixed R70B23/global63; original records reused unchanged; no weights/load/recount/reselection/inference',
        protocol_equivalence_not_replication=True, historical_updates=9600, historical_presentations=7200000,
        historical_successful_view_rows=14400000, historical_successful_mixed_head_rows=14400000,
        historical_wall_seconds=3668.194599, historical_amp_retries=5, parameters=33173154, new_compute=0,
        test='later reuse already-executed R70B TEST object unchanged; no TEST read in this phase')
    atomic_json(args.root / 'reused_control.json', control)
    if not (args.reference / 'geometry_cache/stats.json').is_file():
        raise ValueError('Require existing TRAIN-fitted geometry cache; no rebuilding')
    pipeline(args, 'loading_original_TRAIN_VAL')
    loaders, raw, geometry = prepare_data(args)
    check_original_scope(loaders, raw)
    if geometry != original_protocol['geometry_stats']:
        raise ValueError('Original geometry normalization must be preserved')
    if len(GLOBAL_SOLVERS) != 22 or sum(len(record['pool_ids']) for record in raw['train'].values()) != 128:
        raise ValueError('All original22 global IDs/128 legal deployments must remain')
    normalizers = {}
    for task in PROBLEMS:
        normalizers[task] = fit_normalizer(raw['train'][task]['costs'], raw['train'][task]['pool_ids'])
        if normalizers[task]['fitted_rows'] != 10000:
            raise ValueError('Fit each coordinate on exactly the original task TRAIN10000 rows')
        loaders['train'][task].batch['density_target'] = standardized_targets(
            raw['train'][task]['costs'], raw['train'][task]['pool_ids'], normalizers[task]).to(args.device)
    atomic_json(args.root / 'target_normalizers.json', normalizers)
    sizes = {task: loader.size for task, loader in loaders['train'].items()}
    reference = json.loads((CONTROL_ROOT / 'schedule_seed2.json').read_text())
    full = paired_triples(sizes, epochs=80, seed=2, batch_size=256)
    if state_hash(full) != reference['original80epoch_sha256']:
        raise ValueError('Original80E schedule identity mismatch')
    schedule = {key: value[40:80] if key in ('permutations', 'cycles', 'triples') else value for key, value in full.items()}
    if state_hash(schedule) != reference['selected_stage2_sha256']:
        raise ValueError('Exact global41-80 task/row schedule identity mismatch')
    del full
    atomic_json(args.root / 'schedule_seed2.json', dict(reference, reference=str(CONTROL_ROOT / 'schedule_seed2.json'),
        control_new_execution=False, dropout_seed_schedule_same=True, dropout_tensors_bitwise_matching_claim=False))
    protocol.update(model_params=original_protocol['model_params'], initialization=initialization,
        inherited_identity=inherited, reused_control=control, schedule_identity=reference,
        geometry_stats=geometry, target_normalizers=normalizers, sizes=sizes,
        data_manifest=json.loads((args.root / 'data_manifest.json').read_text()))
    protocol = json.loads(json.dumps(protocol))
    atomic_json(args.root / 'protocol.json', protocol)
    directory = args.root / RUN_NAME
    directory.mkdir(exist_ok=True)
    atomic_json(directory / 'args.json', dict(protocol, arm='B', model_type='complete_joint_mixture_bank'))
    atomic_json(args.root / 'inheritance_budget.json', dict(new_B=dict(updates=UPDATES, selected_rows=PRESENTATIONS,
        encoder_view_rows=VIEW_ROWS, runtime='measured formal B only; discarded capacity extra'),
        reused_A=protocol['reused_A_training'], inherited_stage1_once=protocol['inherited_stage1_once'],
        A_parameters=33173154, B_parameters='actual parameter_counts.json', capacity_matched=False,
        extra_density_head_and_probability_compute=True, independent_control_replication=False))
    return protocol, loaders, raw, schedule, initial


def capacity_update(args, data):
    protocol, loaders, _, schedule, initial = data
    selection = json.loads((CONTROL_ROOT / 'capacity_receipt.json').read_text())['selection']
    entries = triple_indices(schedule, selection['epoch'] - 1, selection['update_in_epoch'] - 1)
    nodes = [int(loaders['train'][task].lengths[indices].max()) for task, indices in entries]
    if [task for task, _ in entries] != ['VRPLTW', 'CVRP', 'TSP'] or nodes != [101, 500, 499]:
        raise ValueError('Require the same known worst-padded scheduled real256 triple; no size sweep')
    receipt = dict(status='started', started_utc=utc_now(), selection=selection,
        selection_source='existing R70 original schedule/input geometry; exact same schedule; no metrics used',
        discarded=True, TRAIN_only=True, real_batch_per_task=256, effective_batch=768,
        formal_model_or_optimizer_transfer=False, automatic_restart=False)
    atomic_json(args.root / 'capacity_started.json', receipt)
    pipeline(args, 'one_discarded_capacity_update', selection=selection)
    rng, started = capture_rng(), time.monotonic()
    model = optimizer = layout = scaler = batches = parameter = None
    try:
        model = make_bank(protocol['model_params'], initial, protocol['target_normalizers'], args.device)
        receipt['initial_bank_state_sha256'] = state_hash(model.state_dict())
        optimizer, layout = make_optimizer(model), GradientLayout(model)
        # Capacity includes both moments for all18 branches, not just the active triple.
        for parameter in model.parameters():
            optimizer.state[parameter].update(step=torch.zeros((), dtype=torch.float32),
                exp_avg=torch.zeros_like(parameter), exp_avg_sq=torch.zeros_like(parameter))
        storage = sum(value.numel() * value.element_size() for state in optimizer.state.values()
                      for value in state.values() if torch.is_tensor(value) and value.is_cuda)
        expected = 2 * sum(p.numel() * p.element_size() for p in model.parameters())
        if storage != expected:
            raise ValueError('Materialize the entire bank Adam first/second moments')
        receipt.update(full_bank_parameters=sum(p.numel() for p in model.parameters()),
            optimizer_storage='materialized entire18 bank, not merely reserved or three active branches',
            optimizer_moment_gpu_bytes=storage, optimizer_moment_gpu_gib=storage / 2**30,
            all_bank_moments_including_inactive_and_unused=True, optimizer_step_scalars='CPU FP32 zero per parameter')
        atomic_json(args.root / 'capacity_started.json', receipt)
        scaler = torch.amp.GradScaler('cuda', enabled=True)
        batches = [gather_batch(loaders['train'][task], indices) for task, indices in entries]
        if [len(batch['ind']) for batch in batches] != [256] * 3:
            raise ValueError('Capacity uses actual256/task without microbatching')
        u = selection['successful_update']
        model.train()
        set_seed(dropout_seed(9600 + u - 1))
        torch.cuda.reset_peak_memory_stats(args.device)
        synchronize(args.device)
        update_started = time.monotonic()
        values = update_triple(model, optimizer, scaler, batches, layout, update_lr(u - 1), u)
        synchronize(args.device)
        free, total = torch.cuda.mem_get_info(args.device)
        receipt.update(status='passed', finished_utc=utc_now(), wall_seconds=time.monotonic() - started,
            update_seconds=time.monotonic() - update_started, values=values,
            peak_memory_gib=torch.cuda.max_memory_allocated(args.device) / 2**30,
            peak_reserved_gib=torch.cuda.max_memory_reserved(args.device) / 2**30,
            physical_used_gib=(total - free) / 2**30, physical_total_gib=total / 2**30,
            indices={task: indices.tolist() for task, indices in entries})
        atomic_json(args.root / 'capacity_receipt.json', receipt)
        print('[R71 capacity passed] ' + json.dumps({key: receipt[key] for key in
            ('status', 'full_bank_parameters', 'optimizer_moment_gpu_bytes', 'update_seconds', 'peak_memory_gib', 'peak_reserved_gib')}), flush=True)
    except BaseException as error:
        receipt.update(status='failed', finished_utc=utc_now(), wall_seconds=time.monotonic() - started,
            error_type=type(error).__name__, error=str(error))
        atomic_json(args.root / 'capacity_failure.json', receipt)
        raise
    finally:
        model = optimizer = layout = scaler = batches = parameter = None
        gc.collect()
        torch.cuda.empty_cache()
        restore_rng(rng)
    atomic_json(args.root / 'capacity_discarded.json', dict(recorded_utc=utc_now(),
        mutated_model_optimizer_scaler_and_graphs_discarded=True, pre_capacity_rng_restored=True,
        formal_model_recreation_required=True, no_weight_optimizer_or_stream_transfer=True))


@torch.no_grad()
def evaluate(model, loaders, raw, batch_size=128, prediction_dir=None):
    was_training, results = model.training, {}
    model.eval()
    started = time.monotonic()
    if prediction_dir is not None:
        prediction_dir.mkdir(parents=True, exist_ok=True)
    try:
        for problem, loader in loaders.items():
            branch = model.branches[model.branch_index(loader.batch['problem_id'])]
            head = branch.density_head
            normalizer = dict(pool_ids=head.canonical_pool_ids.cpu().tolist(),
                mean=head.target_mean.cpu().tolist(), scale=head.target_scale.cpu().tolist())
            targets = standardized_targets(raw[problem]['costs'], raw[problem]['pool_ids'], normalizer)
            probabilities, density_nll = [], 0.
            for start in range(0, loader.size, batch_size):
                indices = torch.arange(start, min(start + batch_size, loader.size))
                batch = model_inputs(gather_batch(loader, indices))
                output = model(batch)
                values = output['probabilities'].float()
                if not bool(torch.isfinite(values).all()) or not bool((values >= 0).all()):
                    raise RuntimeError('Nonfinite native winner probabilities')
                probabilities.append(values.cpu())
                density_nll += float(head.nll(output['density'], targets[indices].to(values.device))) * len(indices)
            probabilities = torch.cat(probabilities).numpy()
            if len(probabilities) != loader.size:
                raise ValueError('Evaluation must include every original row and tail')
            # Existing cost/rank observables use unfloored probabilities and global-ID ties.
            result, pred = policy_metrics(probabilities, raw[problem])
            native = torch.from_numpy(probabilities[np.arange(loader.size), raw[problem]['winner']])
            result.update(ce=float(-native.clamp_min(CE_FLOOR).log().mean()),
                native_correct=int(np.count_nonzero(pred == raw[problem]['winner'])),
                density_nll_per_dimension=density_nll / loader.size,
                ce_definition='native winner probability, reporting-only floor1e-12',
                predicted_vectors_per_query=16384, encoder_forwards_per_query=1, executed_solvers=0)
            results[problem] = result
            if prediction_dir is not None:
                np.savez_compressed(prediction_dir / f'{problem}.npz', indices=np.arange(loader.size),
                    probabilities=probabilities, selection_scores=probabilities, pred=pred,
                    winner=raw[problem]['winner'], costs=raw[problem]['costs'],
                    pool_ids=np.array(raw[problem]['pool_ids']), pool=np.array(raw[problem]['pool']), nodes=loader.lengths.numpy())
        value = dict(per_problem=results, macro=aggregate(results), families=families(results),
            native_correct=sum(r['native_correct'] for r in results.values()),
            native_total=sum(r['n'] for r in results.values()), seconds=time.monotonic() - started,
            inference_rule=INFERENCE_RULE, whole_evaluation_time_is_singleton_or_solver_latency=False)
        value['macro']['macro_top1'] = value['native_correct'] / value['native_total']
        value['macro']['macro_density_nll_per_dimension'] = float(np.mean([r['density_nll_per_dimension'] for r in results.values()]))
        for family, members in [('TSP', ['TSP']), ('CVRP', ['CVRP']), ('ATSP', ['ATSP']),
                ('MVRP', [p for p in PROBLEMS if p not in ('TSP', 'CVRP', 'ATSP')])]:
            value['families'][family]['macro_density_nll_per_dimension'] = float(np.mean([results[p]['density_nll_per_dimension'] for p in members]))
        return value
    finally:
        model.train(was_training)


def first_update_handoff(args, launch, metrics):
    atomic_json(args.root / 'handoff.json', dict(status='formal B fixed40E training running',
        command='bash code/V4/run_v4_r71.sh train', training_command='python -B -u -m code.V4.r71_experiment --stage train',
        tmux='r71_train', tmux_pane=launch['tmux_pane'], tmux_pane_pid=launch['tmux_pane_pid'],
        tmux_server_host=launch['tmux_server_host'], slurm_job='1465', slurm_step=launch['slurm_step'],
        host=launch['host'], trainer_pid=launch['pid'], parent_pid=os.getppid(), gpu_uuid=GPU_UUID, gpu_index=1,
        arm='B', new_arms=['B'], reused_control='fixed R70B23/global63/8709; no new A compute',
        real_batch_per_task=256, task_rows=metrics['task_rows'], effective_batch=metrics['rows'],
        first_finite_formal_update=metrics, parameter_counts=json.loads((args.root / 'parameter_counts.json').read_text()),
        cpu_check=json.loads((args.root / 'cpu_check.json').read_text()), capacity=json.loads((args.root / 'capacity_receipt.json').read_text()),
        memory={key: metrics[key] for key in ('peak_memory_gib', 'peak_reserved_gib', 'physical_used_gib', 'physical_total_gib')},
        changed_files=['code/V4/' + name for name in OWNED_SOURCES] + [str(args.root.relative_to(WORKSPACE)) + '/'],
        next_normal_monitor_instruction='Leave original fixed40E trainer live; no automatic restart. Parent owns later conditional delivery preparation, VAL freeze, one new B original full18 TEST, reviews, memory and one completed-version Git publication.',
        inherited_stage1_once=dict(updates=9600, selected_rows=7200000, wall_seconds=2764.217798),
        reused_A_historical=dict(updates=9600, selected_rows=7200000, wall_seconds=3668.194599, retries=5),
        test_authorized=False, test_read=False, no_git_actions=True, goal_status='active; unachieved'))


def train_arm(args, data, launch):
    protocol, loaders, raw, schedule, initial = data
    directory = args.root / RUN_NAME
    config = dict(protocol, arm='B', model_type='complete_joint_mixture_bank')
    model = make_bank(protocol['model_params'], initial, protocol['target_normalizers'], args.device)
    initial_hash = state_hash(model.state_dict())
    if initial_hash != json.loads((args.root / 'capacity_receipt.json').read_text())['initial_bank_state_sha256']:
        raise ValueError('Recreated formal bank must exactly match the approved pre-capacity initial state')
    counts_by_branch = [sum(p.numel() for p in branch.parameters()) for branch in model.branches]
    parameter_counts = dict(A=33173154, B=sum(counts_by_branch), branch_counts=dict(zip(PROBLEMS, counts_by_branch)),
        branches=18, trainable_B=sum(p.numel() for p in model.parameters() if p.requires_grad),
        density_head_counts={task: sum(p.numel() for p in branch.density_head.parameters()) for task, branch in zip(PROBLEMS, model.branches)},
        capacity_matched=False, FLOPs_matched=False, retained_parameter_active_update_recipe_matched=True,
        branch_active_updates=1600, rows_per_branch=400000)
    atomic_json(args.root / 'parameter_counts.json', parameter_counts)
    optimizer, layout = make_optimizer(model), GradientLayout(model)
    scaler = torch.amp.GradScaler('cuda', enabled=True)
    atomic_save(args.root / 'formal_initial_bank.pt', dict(model=cpu_state(model), args=config,
        model_state_sha256=initial_hash, persistent_buffers_saved_before_validation=True))
    atomic_json(directory / 'initialization.json', dict(initial_bank_state_sha256=initial_hash,
        inherited_identical_retained_trunks=True, private_task_density_heads=True, scalar_head_deleted=True,
        all_parameters_trainable=all(p.requires_grad for p in model.parameters()), optimizer_fresh=not optimizer.state,
        scaler_fresh=scaler.state_dict(), recreated_after_capacity_discard=True, no_capacity_model_optimizer_transfer=True,
        normalizers_ordering_Sobol_persistent=True, zero_output_weights_expected_zero_initial_backbone_gradient=True,
        optimizer_storage='fresh lazy AdamW; capacity separately materialized entire bank moments',
        parameter_counts=parameter_counts, gradient_layout=layout.receipt, new_successful_updates=0, new_presentations=0))
    atomic_json(args.root / 'formal_started.json', dict(started_utc=utc_now(), pid=os.getpid(), new_arms=['B'],
        control_reused=True, capacity_discarded=True, fresh_optimizer_state_entries=len(optimizer.state),
        test_authorized=False, no_automatic_restart=True))
    pipeline(args, 'formal_fixed40E', launch=launch)
    history, lr_trace, best, best_model = [], [], None, None
    successful = presentations = view_presentations = attempted_views = growth_successes = total_retries = 0
    counts, task_rows = Counter(), Counter()
    wandb_run = None
    if args.wandb:
        import wandb
        wandb_run = wandb.init(**protocol['wandb'], name='R71_' + RUN_NAME, config=config,
            dir=str(directory), id='r71-b-' + launch['id'])

    def save_state(epoch, step):
        atomic_save(directory / 'last.pt', dict(model=cpu_state(model), optimizer=optimizer.state_dict(),
            scaler=scaler.state_dict(), args=config, epoch=epoch, global_epoch=40 + epoch, next_step=step,
            successful_updates=successful, presentations=presentations, successful_view_rows=view_presentations,
            attempted_view_rows=attempted_views, scale_growth_successes=growth_successes, amp_retries=total_retries,
            updates_by_problem=dict(counts), rows_by_problem=dict(task_rows), rng=capture_rng(), history=history,
            best=best, lr_trace=lr_trace, parameter_mode=PARAMETER_MODE, all18_common_epoch=True,
            checkpoint_mode='eval', projector_included=False, persistent_inference_buffers=True))
        atomic_json(directory / 'history.json', history)

    def save_bundle(path, state, epoch, validation):
        atomic_save(path, dict(model=state, args=config, epoch=epoch - 1, global_epoch=40 + epoch,
            common_epoch=epoch, branch_epochs=[epoch] * 18, problem_names=list(PROBLEMS),
            native_correct=validation['native_correct'], native_total=18000, macro=validation['macro'],
            parameter_mode=PARAMETER_MODE, checkpoint_mode='eval', projector_included=False,
            inference_rule=INFERENCE_RULE, successful_updates=epoch * 240, presentations=epoch * 180000,
            model_state_sha256=state_hash(state), all18_complete_branches=True, persistent_inference_buffers=True))

    try:
        for epoch in range(1, 41):
            model.train()
            torch.cuda.reset_peak_memory_stats(args.device)
            started = time.monotonic()
            totals, task_stats = Counter(), {}
            for step in range(240):
                entries = triple_indices(schedule, epoch - 1, step)
                batches = [gather_batch(loaders['train'][task], indices) for task, indices in entries]
                expected_rows = 256 if step < 234 else 16
                if [len(batch['ind']) for batch in batches] != [expected_rows] * 3:
                    raise ValueError('Preserve real256/task and16/task tails')
                lr, view_seed = update_lr(successful), dropout_seed(9600 + successful)
                set_seed(view_seed)
                synchronize(args.device)
                before = time.monotonic()
                values = update_triple(model, optimizer, scaler, batches, layout, lr, successful + 1, growth_successes)
                synchronize(args.device)
                seconds, growth_successes = time.monotonic() - before, values['scale_growth_successes']
                rows = sum(len(indices) for _, indices in entries)
                successful += 1
                presentations += rows
                view_presentations += values['successful_view_rows']
                attempted_views += values['attempted_view_rows']
                total_retries += values['retries']
                lr_trace.append(lr)
                free, total = torch.cuda.mem_get_info(args.device)
                metrics = dict(launch_id=launch['id'], arm='B', epoch=epoch, global_epoch=40 + epoch, stage=2,
                    update_in_epoch=step + 1, successful_update=successful, schedule_successful_update=9600 + successful,
                    presentations=presentations, view_presentations=view_presentations, attempted_view_presentations=attempted_views,
                    tasks=[task for task, _ in entries], task_rows=[len(indices) for _, indices in entries], rows=rows,
                    actual_forward_rows=[len(batch['ind']) for batch in batches],
                    batch_indices_sha256=[state_hash(indices) for _, indices in entries], dropout_seed=view_seed,
                    lr=lr, gradient_parameter_mode=PARAMETER_MODE, seconds=seconds, queries_per_second=rows / seconds,
                    peak_memory_gib=torch.cuda.max_memory_allocated(args.device) / 2**30,
                    peak_reserved_gib=torch.cuda.max_memory_reserved(args.device) / 2**30,
                    physical_used_gib=(total - free) / 2**30, physical_total_gib=total / 2**30, **values)
                append_jsonl(directory / 'updates.jsonl', metrics)
                for key in LOSS_FIELDS:
                    totals[key] += values[key] * rows
                for key in ('grad_norm', 'clip_coefficient', 'retries'):
                    totals[key] += values[key]
                totals['samples'] += rows
                totals['updates'] += 1
                for key in ('successful_view_rows', 'attempted_view_rows'):
                    totals[key] += values[key]
                for index, (task, indices) in enumerate(entries):
                    counts[task] += 1
                    task_rows[task] += len(indices)
                    item = task_stats.setdefault(task, dict(rows=0, batches=0, full_grad_norm=0., **{key: 0. for key in LOSS_FIELDS}))
                    item['rows'] += len(indices)
                    item['batches'] += 1
                    for key in LOSS_FIELDS:
                        item[key] += values['task_losses'][index][key] * len(indices)
                    item['full_grad_norm'] += values['full_task_grad_norms'][index]
                if successful == 1:
                    atomic_json(directory / 'first_formal_update.json', dict(metrics,
                        selected_indices={task: indices.tolist() for task, indices in entries}, launch=launch))
                    print('[R71 first FORMAL update] ' + json.dumps({key: metrics[key] for key in
                        ('successful_update', 'tasks', 'task_rows', 'rows', 'lr', 'nll_per_dimension', 'joint_nll',
                         'loss', 'grad_norm', 'seconds', 'peak_memory_gib', 'retries', 'finite_gradient')}), flush=True)
                if successful == 1 or (step + 1) % 60 == 0:
                    save_state(epoch, step + 1)
                    launch.update(status='running', epoch=epoch, global_epoch=40 + epoch,
                        successful_updates=successful, presentations=presentations, heartbeat_unix=time.time())
                    atomic_json(directory / 'launches' / (launch['id'] + '.json'), launch)
                    atomic_json(args.root / 'current_status.json', dict(launch, last_metrics=metrics))
                    if successful == 1:
                        first_update_handoff(args, launch, metrics)
                    print(f'[R71B] epoch={epoch}/40 update={step + 1}/240 total={successful} '
                          f'NLL/D={values["nll_per_dimension"]:.6f} lr={lr:.9g} retries={total_retries}', flush=True)
                if wandb_run is not None:
                    wandb_run.log({key: metrics[key] for key in ('epoch', 'global_epoch', 'successful_update',
                        'loss', 'nll_per_dimension', 'joint_nll', 'lr', 'grad_norm', 'clip_coefficient', 'scale', 'retries',
                        'peak_memory_gib', 'peak_reserved_gib', 'queries_per_second', 'presentations',
                        'view_presentations', 'attempted_view_presentations')})
            if (totals['samples'] != 180000 or totals['updates'] != 240 or len(task_stats) != 18 or successful != epoch * 240
                    or any(item['rows'] != 10000 or item['batches'] != 40 for item in task_stats.values())):
                raise ValueError('Each epoch must cover all180000 original TRAIN rows once')
            optimization = {key: totals[key] / 180000 for key in LOSS_FIELDS}
            optimization.update({key: totals[key] / 240 for key in ('grad_norm', 'clip_coefficient')})
            optimization.update(lr=lr, lr_first=lr_trace[(epoch - 1) * 240], samples=180000, updates=240,
                amp_retries=int(totals['retries']), successful_view_rows=int(totals['successful_view_rows']),
                attempted_view_rows=int(totals['attempted_view_rows']), seconds=time.monotonic() - started,
                peak_memory_gib=torch.cuda.max_memory_allocated(args.device) / 2**30)
            for item in task_stats.values():
                for key in LOSS_FIELDS:
                    item[key] /= item['rows']
                item['full_grad_norm'] /= item['batches']
            validation = evaluate(model, loaders['val'], raw['val'], 128, directory / f'val_predictions/epoch{epoch:03d}')
            if validation['native_total'] != 18000 or any(value['n'] != 1000 for value in validation['per_problem'].values()):
                raise ValueError('Full18 VAL requires all18000 original rows')
            training = None
            if epoch % 5 == 0:
                training = evaluate(model, loaders['train'], raw['train'], 128, directory / f'train_predictions/epoch{epoch:03d}')
                if training['native_total'] != 180000:
                    raise ValueError('Full eval-mode TRAIN coverage required')
            row = dict(epoch=epoch, global_epoch=40 + epoch, stage=2, successful_updates=successful,
                presentations=presentations, successful_view_rows=view_presentations, attempted_view_rows=attempted_views,
                updates_by_problem=dict(counts), rows_by_problem=dict(task_rows), optimization=optimization,
                task_optimization=task_stats, train=training, val=validation, parameter_mode=PARAMETER_MODE,
                all18_common_epoch=True, inference_rule=INFERENCE_RULE)
            history.append(row)
            if prefer_checkpoint(validation, None if best is None else best['val']):
                best, best_model = row, cpu_state(model)
                unique = directory / f'best_checkpoints/epoch{epoch:03d}.pt'
                unique.parent.mkdir(exist_ok=True)
                save_bundle(unique, best_model, epoch, validation)
                shutil.copy2(unique, directory / 'best.pt')
                atomic_json(directory / 'best_eval.json', best)
            if epoch == 40:
                save_bundle(directory / 'final.pt', cpu_state(model), epoch, validation)
            save_state(epoch, 240)
            append_jsonl(args.root / 'curves/B_metrics.jsonl', row)
            atomic_json(args.root / 'current_status.json', dict(launch, status='running', completed_epochs=epoch,
                successful_updates=successful, presentations=presentations, selected_B_epoch=best['epoch'],
                selected_B_validation_correct=best['val']['native_correct'],
                last_validation_correct=validation['native_correct'], heartbeat_unix=time.time()))
            print_eval(f'R71B val epoch{epoch}', validation['per_problem'], validation['macro'])
            if wandb_run is not None:
                wb = dict(epoch=epoch, global_epoch=40 + epoch, **{'optimization/' + key: value for key, value in optimization.items()})
                for split, value in (('train_eval', training), ('val', validation)):
                    if value is not None:
                        wb.update({split + '/' + key: entry for key, entry in value['macro'].items()})
                        wb[split + '/native_correct'] = value['native_correct']
                        for task, result in value['per_problem'].items():
                            wb.update({f'{split}/{task}/{key}': result[key] for key in
                                ('top1', 'top2', 'top3', 'ce', 'density_nll_per_dimension', 'mean_cost', 'actual_regret_pct')})
                wandb_run.log(wb)
        if (successful != UPDATES or presentations != PRESENTATIONS or view_presentations != VIEW_ROWS
                or len(history) != 40 or set(counts.values()) != {1600} or set(task_rows.values()) != {400000}):
            raise ValueError('R71 fixed40E budget/coverage mismatch')
        counters = {name: int(optimizer.state[p]['step']) if p in optimizer.state else 0 for name, p in model.named_parameters()}
        atomic_json(directory / 'parameter_step_counts.json', dict(steps=counters, branch_activations=dict(counts),
            explanation='Active-gradient parameters only; unused input paths may remain at zero'))
        model.load_state_dict(best_model, strict=True)
        selected_train = evaluate(model, loaders['train'], raw['train'], 128, directory / 'train_predictions/best')
        if selected_train['native_total'] != 180000:
            raise ValueError('Selected bank TRAIN evaluation requires all180000 rows')
        best_eval = dict(best, train=selected_train)
        atomic_json(directory / 'best_eval.json', best_eval)
        atomic_json(directory / 'result.json', dict(best=best_eval, final=history[-1], epochs_completed=40,
            successful_updates=successful, presentations=presentations, successful_view_rows=view_presentations,
            attempted_view_rows=attempted_views, amp_retries=total_retries, actual_execution=executed_updates(directory),
            updates_by_problem=dict(counts), rows_by_problem=dict(task_rows),
            best_checkpoint_sha256=file_hash(directory / 'best.pt'), final_checkpoint_sha256=file_hash(directory / 'final.pt'),
            selected_epoch_checkpoint=str(directory / f'best_checkpoints/epoch{best["epoch"]:03d}.pt'),
            lr_trace_sha256=state_hash(lr_trace), parameter_counts=parameter_counts,
            deployment_parameter_mode=PARAMETER_MODE, checkpoint_rule=protocol['checkpoint_rule'], inference_rule=INFERENCE_RULE,
            initialization=protocol['initialization'], fixed_horizon=True, stop_reason='fixed40epoch joint-mixture likelihood training',
            all18_common_epoch=True, new_arms=['B'], reused_control=str(args.root / 'reused_control.json'),
            capacity_compute_accounting='one discarded TRAIN update extra; excluded from formal9600 budget/runtime',
            test_read=False, test_authorized=False))
    finally:
        if wandb_run is not None:
            wandb_run.finish()


def run(args, data):
    directory = args.root / RUN_NAME
    (directory / 'launches').mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    launch = dict(id=stamp, arm='B', status='started', started_unix=time.time(), host=socket.gethostname(),
        pid=os.getpid(), gpu_uuid=GPU_UUID, formal=True, tmux_session='r71_train',
        tmux_pane=os.environ.get('R71_TMUX_PANE_ID', ''), tmux_pane_pid=os.environ.get('R71_TMUX_PANE_PID', ''),
        tmux_server_host=os.environ.get('R71_TMUX_SERVER_HOST', ''),
        slurm_step=os.environ.get('SLURM_JOB_ID', '') + '.' + os.environ.get('SLURM_STEP_ID', ''))
    started, status = time.monotonic(), 'failed'
    atomic_json(directory / 'launches' / (stamp + '.json'), launch)
    try:
        train_arm(args, data, launch)
        status = 'complete'
    finally:
        launch.update(status=status, wall_seconds=time.monotonic() - started, finished_unix=time.time())
        atomic_json(directory / 'launches' / (stamp + '.json'), launch)
        atomic_json(directory / 'wall_time.json', [launch])
        atomic_json(args.root / 'current_status.json', launch)
        if status == 'complete':
            result = json.loads((directory / 'result.json').read_text())
            result['wall_seconds'] = launch['wall_seconds']
            atomic_json(directory / 'result.json', result)


def publish_candidate(root):
    result = json.loads((root / RUN_NAME / 'result.json').read_text())
    control = json.loads((root / 'reused_control.json').read_text())['model']
    candidate = dict(path=str(root / RUN_NAME / 'best.pt'), sha256=result['best_checkpoint_sha256'],
        selected_epoch_checkpoint=result['selected_epoch_checkpoint'], epoch=result['best']['epoch'],
        global_epoch=result['best']['global_epoch'], native_correct=result['best']['val']['native_correct'],
        native_total=18000, val=result['best']['val']['macro'], parameter_mode=PARAMETER_MODE,
        model_type='complete_joint_mixture_bank', all18_common_epoch=True, persistent_inference_buffers=True)
    official = 'B' if candidate['native_correct'] > CONTROL_CORRECT else 'A'
    atomic_json(root / 'validation_candidates.json', dict(models={'A': control, 'B': candidate},
        proposed_official_arm=official, official_rule='B strictly>8709; tie/lower fixed R70B23 A',
        new_arms=['B'], reused_arms=['A'], parent_freeze_pending=True, inference_rule=INFERENCE_RULE,
        test_authorized=False, later_test='parent later freezes common-epoch bank/buffers/official arm before ONE full18 new B TEST regardless of improvement; A TEST reused unchanged'))


def cpu_check(args):
    from .r71_cpu_check import focused_check
    path = args.root / 'cpu_check_invocations.jsonl'
    previous = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    if any(item['status'] == 'passed' for item in previous):
        raise ValueError('The one focused R71 CPU check already passed; no repeat campaign')
    invocation, started = 1 + sum(item['status'] == 'started' for item in previous), time.monotonic()
    receipt = dict(invocation=invocation, type='focused synthetic density/normalization/order/isolation/probability/whole-retry CPU check',
        started_utc=utc_now(), pid=os.getpid(), status='started', gpu_used=False, source_hashes=source_hashes())
    append_jsonl(path, receipt)
    with (args.root / 'cpu_check.log').open('a') as stream, redirect_stdout(Tee(sys.stdout, stream)), redirect_stderr(Tee(sys.stderr, stream)):
        try:
            result = dict(focused_check(), invocation=invocation, started_utc=receipt['started_utc'],
                wall_seconds=time.monotonic() - started, source_hashes=receipt['source_hashes'])
            atomic_json(args.root / 'cpu_check.json', result)
            append_jsonl(path, dict(receipt, status='passed', wall_seconds=result['wall_seconds']))
            print('[R71 focused CPU check] ' + json.dumps(result), flush=True)
        except BaseException as error:
            append_jsonl(path, dict(receipt, status='failed', wall_seconds=time.monotonic() - started,
                error_type=type(error).__name__, error=str(error)))
            raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('cpu-check', 'train'), default='train')
    args = parser.parse_args(argv)
    args.root, args.reference, args.device, args.batch_size, args.wandb = ROOT, REFERENCE, 'cuda:0', 256, True
    torch.set_num_threads(1)
    args.root.mkdir(parents=True, exist_ok=True)
    if args.stage == 'cpu-check':
        cpu_check(args)
        return
    if (os.environ.get('CUDA_VISIBLE_DEVICES') != GPU_UUID or os.environ.get('SLURM_JOB_ID') != '1465'
            or socket.gethostname().split('.')[0] != 'gpu03'):
        parser.error('R71 requires existing Slurm1465/gpu03/index1 assigned-UUID launcher')
    check = json.loads((args.root / 'cpu_check.json').read_text())
    if not check.get('passed') or check['source_hashes'] != source_hashes():
        parser.error('The focused CPU check must pass on these exact launch sources')
    configure_torch()
    set_seed(2)
    with (args.root / 'pipeline.log').open('a') as stream, redirect_stdout(Tee(sys.stdout, stream)), redirect_stderr(Tee(sys.stderr, stream)):
        try:
            data = prepare(args)
            capacity_update(args, data)
            run(args, data)
            publish_candidate(args.root)
            atomic_json(args.root / 'training_complete.json', dict(completed_utc=utc_now(), new_arms=['B'],
                reused_control='fixed R70B23', epochs=40, new_updates=UPDATES, new_presentations=PRESENTATIONS,
                successful_view_rows=VIEW_ROWS, inherited_stage1_updates_once=9600,
                inherited_stage1_presentations_once=7200000, parent_freeze_pending=True, test_authorized=False))
            pipeline(args, 'training_complete_parent_freeze_pending')
            print('[R71 training complete] one new B; parent freeze pending; no TEST/analysis executed', flush=True)
        except BaseException as error:
            failure = dict(recorded_utc=utc_now(), error_type=type(error).__name__, error=str(error), pid=os.getpid(),
                slurm_step=os.environ.get('SLURM_JOB_ID', '') + '.' + os.environ.get('SLURM_STEP_ID', ''),
                automatic_restart=False, samples_skipped=False, duplicate_attempt_authorized=False)
            atomic_json(args.root / 'terminal_failure.json', failure)
            pipeline(args, 'failed', failure=failure)
            raise


if __name__ == '__main__':
    main()

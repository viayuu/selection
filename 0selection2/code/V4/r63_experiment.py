"""R63 validation-selected workspace training and separately authorized locked TEST."""

import argparse
import json
import math
import os
import shutil
import sys
import time
from collections import Counter
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import numpy as np
import torch

from code.unified_selector.registry import GLOBAL_SOLVERS, PROBLEMS
from .multitask_probe import Tee, file_hash, gather_batch, state_hash
from .pairwise_objective import inputs_only
from .performance_experiment import SOURCE_FILES
from .r40_experiment import aggregate, families
from .r42_experiment import CostController, paired_schedule, prepare_data
from .r43_experiment import add_diagnostics, evaluate, policy_metrics, strong_comparisons
from .r59_experiment import (GPU_UUID, REFERENCE, R43_ARGS, WORKSPACE, atomic_json, atomic_save,
                             cached_nominations, check_original_scope, publish_best)
from .r63_graph import build_workspace_graph
from .r63_model import initialize, update_batch
from .train import configure_torch, print_eval, set_seed
from .training_monitor import capture_rng, restore_rng


ROOT = WORKSPACE / 'code/V4/runs/R63_recurrent_workspaces'
RUN_NAMES = {'A': 'A_static_memory_seed2', 'B': 'B_communicating_memory_seed2'}
LOSS_FIELDS = ('loss', 'ce', 'kl', 'grad_norm')
NEW_SOURCES = ('r63_model.py', 'r63_graph.py', 'r63_experiment.py', 'run_v4_r63.sh',
               'r59_experiment.py', 'r45_experiment.py', 'r43_experiment.py', 'r42_experiment.py',
               'r40_experiment.py', 'pairwise_selector.py', 'pairwise_objective.py', 'solver_code_encoder.py')


def planned_config(args):
    return dict(seed=args.seed, max_epochs=args.epochs, batch_size=args.batch_size,
        microbatch_size=16, eval_batch_size=args.eval_batch_size, learning_rate=1e-4, weight_decay=1e-4,
        dropout=.1, warmup_epochs=3, clip_norm=1., rdrop_weight=.35, rdrop_ramp_epochs=3,
        min_epochs=15, stop_patience=8, lr_patience=3, lr_factor=.5, top1_min_delta_pp=.001,
        arms=RUN_NAMES, problems=list(PROBLEMS), solver_names=list(GLOBAL_SOLVERS),
        loss='mean of two native-ind full-class CE forwards + FP32 valid symmetric KL; no cost loss',
        checkpoint_rule='strict maximum full18 validation native macro_top1; ties retain earlier epoch',
        shared_controller='A observes negative 100*macro_top1 via existing minimizing controller; B follows E/LR',
        initialization='identical complete fresh seed2 state; original R45A encoders and readout; no historical weights',
        architecture=dict(rounds=8, width=128, message_width=32, graph='input-only directed low16 out/in + depot/3special union',
                          A='initial H0 neighbor memory without detach', B='previous-round neighbor memory',
                          gate='sigmoid of shared additive source/target/solver/distance scalar projections',
                          readout='candidate-specific masked mean/max, original solver-bank decoder, direct logits'),
        full_coverage=True, drop_last=False, natural_sampling=True, no_augmentation=True,
        dropout_passes=2, optimizer_updates_per_batch=1, geometry_fitted_on='original TRAIN only',
        inference_timing='first16 validation queries/task single, remaining queries batched, each exactly once in final selected eval',
        incumbent=dict(name='R60 B best39', macro_top1=.4936111111111112,
                       macro_actual_regret_pct=.9227890652257744), target_native_macro_top1=.70,
        wandb=dict(project='selector', entity='yjkds-southern-university-of-science-technology', mode='offline'),
        test_read=False, gpu_preflight=False)


def source_hashes():
    files = [Path(__file__).parent / name for name in sorted(set(SOURCE_FILES + NEW_SOURCES))]
    files += [WORKSPACE / 'code/unified_selector' / name for name in ('registry.py', 'data.py')]
    return {str(path.relative_to(WORKSPACE)): file_hash(path) for path in files}


def load_inputs(args):
    loaders, raw, geometry = prepare_data(args)
    check_original_scope(loaders, raw)
    reference, references = cached_nominations(raw)
    return loaders, raw, geometry, reference, references


def prepare(args):
    protocol = planned_config(args)
    protocol.update(source_hashes=source_hashes(), r43_args_sha256=file_hash(R43_ARGS))
    for relative, digest in protocol['source_hashes'].items():
        destination = args.root / 'source_launch' / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() and file_hash(destination) != digest:
            raise ValueError('Preserve the existing R63 launch snapshot')
        if not destination.exists():
            shutil.copy2(WORKSPACE / relative, destination)
    atomic_json(args.root / 'source_launch/hashes.json', protocol['source_hashes'])
    loaders, raw, geometry, reference, references = load_inputs(args)
    sizes = {p: loader.size for p, loader in loaders['train'].items()}
    schedule = paired_schedule(sizes, args.epochs, args.batch_size, args.seed)
    atomic_save(args.root / 'schedule_seed2.pt', schedule)
    model, params = initialize('A', geometry, args.device, args.seed)
    initial = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    atomic_save(args.root / 'selector_initial.pt', initial)
    protocol.update(initial_model_hash=state_hash(initial),
        model_params=dict(A=params, B=dict(params, neighbor_memory='previous_round')),
        trainable_parameters=sum(p.numel() for p in model.parameters()), geometry_stats=geometry,
        reference_hashes=references, data_manifest=json.loads((args.root / 'data_manifest.json').read_text()),
        sizes=sizes, updates_per_epoch={p: math.ceil(n / args.batch_size) for p, n in sizes.items()},
        schedule_hash=state_hash(schedule))
    atomic_json(args.root / 'protocol.json', protocol)
    print('[R63 prepared] Original full18 data and shared scratch state; next update is formal training', flush=True)


def load_training_data(args):
    protocol = json.loads((args.root / 'protocol.json').read_text())
    current_sources, launch_sources = source_hashes(), protocol['source_hashes']
    # Authorized entry edits: locked TEST and separately recorded, zero-update B15 finalization.
    entry_scripts = ('code/V4/r63_experiment.py', 'code/V4/run_v4_r63.sh')
    if ({p: h for p, h in current_sources.items() if p not in entry_scripts}
            != {p: h for p, h in launch_sources.items() if p not in entry_scripts}
            or file_hash(R43_ARGS) != protocol['r43_args_sha256']
            or any(protocol[k] != getattr(args, a) for k, a in
                   (('seed', 'seed'), ('max_epochs', 'epochs'), ('batch_size', 'batch_size'),
                    ('eval_batch_size', 'eval_batch_size')))):
        raise ValueError('R63 source/configuration changed after preparation')
    amendments = {p: dict(launch_sha256=launch_sources[p], current_sha256=current_sources[p])
                  for p in entry_scripts if launch_sources[p] != current_sources[p]}
    if amendments:
        print('[R63 authorized entry amendments; original update recipe retained] '
              + json.dumps(amendments, sort_keys=True), flush=True)
    loaders, raw, geometry, reference, references = load_inputs(args)
    if (geometry != protocol['geometry_stats'] or references != protocol['reference_hashes']
            or json.loads((args.root / 'data_manifest.json').read_text()) != protocol['data_manifest']):
        raise ValueError('R63 original data, geometry or diagnostic reference changed')
    schedule = torch.load(args.root / 'schedule_seed2.pt', map_location='cpu', weights_only=False)
    if state_hash(schedule) != protocol['schedule_hash']:
        raise ValueError('R63 shared original batch schedule changed')
    return protocol, loaders, raw, geometry, schedule, reference


def shared_replay_schedule(root):
    directory = root / RUN_NAMES['A']
    result = json.loads((directory / 'result.json').read_text())
    history = json.loads((directory / 'history.json').read_text())
    value = dict(controller_arm='A', controller_metric='native macro_top1', epochs=result['epochs_completed'],
                 successful_updates=result['successful_updates'], lr=[row['optimization']['lr'] for row in history])
    if len(value['lr']) != value['epochs']:
        raise ValueError('Incomplete A-controlled LR trace')
    path = root / 'replay_schedule.json'
    if path.exists() and json.loads(path.read_text()) != value:
        raise ValueError('R63 common LR/horizon changed')
    atomic_json(path, value)
    return value


def b15_amendment(root):
    path = root / 'budget_amendment.json'
    if not path.exists():
        return None
    value = json.loads(path.read_text())
    fixed = dict(A_epochs=40, B_epochs=15, A_successful_updates=56880, B_successful_updates=21330,
                 stop_after_completed_validation=True, paired_horizon=False)
    if any(value.get(k) != v for k, v in fixed.items()) or not value.get('requested_at'):
        raise ValueError('Only the explicit user-authorized A40/B15 amendment is accepted')
    return value


def synchronize(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


@torch.no_grad()
def selected_validation(model, loaders, raw, reference, batch_size, directory, epoch):
    """One normal selected-checkpoint evaluation, with no repeated timing queries."""
    was_training = model.training
    model.eval()
    results, per_timing = {}, {}
    prediction_dir = directory / 'val_predictions/best'
    prediction_dir.mkdir(parents=True, exist_ok=True)
    validation_start = time.monotonic()
    try:
        for problem, loader in loaders.items():
            device = loader.batch['ind'].device
            scores, margins, single_queries, batched = [], [], [], []
            groups = [torch.tensor([i]) for i in range(min(16, loader.size))]
            groups += [torch.arange(i, min(i+batch_size, loader.size)) for i in range(16, loader.size, batch_size)]
            for indices in groups:
                synchronize(device)
                started = time.monotonic()
                clean = inputs_only(gather_batch(loader, indices))
                synchronize(device)
                gathered = time.monotonic()
                clean['workspace_graph'] = build_workspace_graph(clean)
                synchronize(device)
                graphed = time.monotonic()
                output = model(clean)
                synchronize(device)
                forwarded = time.monotonic()
                scores.append(output['logits'].float().cpu())
                margins.append(output['pair_margin'].float().cpu())
                ended = time.monotonic()
                row = dict(indices=indices.tolist(), true_sizes=loader.lengths[indices].tolist(),
                           queries=len(indices), input_gather_seconds=gathered-started,
                           graph_construction_seconds=graphed-gathered,
                           selector_forward_wall_seconds=forwarded-graphed,
                           output_transfer_seconds=ended-forwarded, total_seconds=ended-started)
                (single_queries if int(indices[0]) < 16 else batched).append(row)
            scores, margins = torch.cat(scores).numpy(), torch.cat(margins).numpy()
            if len(scores) != loader.size:
                raise ValueError('Final validation timing must count every original query exactly once')
            result, pred = policy_metrics(scores, raw[problem])
            result.update(strong_comparisons(margins, raw[problem], reference[problem]))
            results[problem] = result
            np.savez_compressed(prediction_dir / f'{problem}.npz', indices=np.arange(len(pred)), logits=scores,
                pair_margin=margins, pred=pred, winner=raw[problem]['winner'], costs=raw[problem]['costs'],
                pool_ids=np.array(raw[problem]['pool_ids']), pool=np.array(raw[problem]['pool']),
                reference_top2=reference[problem], nodes=loader.lengths.numpy())
            component_keys = ('input_gather_seconds', 'graph_construction_seconds', 'selector_forward_wall_seconds',
                              'output_transfer_seconds', 'total_seconds')
            per_timing[problem] = dict(queries=loader.size, single_queries=single_queries,
                first_query_seconds=single_queries[0]['total_seconds'],
                subsequent_single_median_seconds=float(np.median([r['total_seconds'] for r in single_queries[1:]])),
                remaining_batched=dict(queries=sum(r['queries'] for r in batched), batch_size=batch_size,
                    batches=len(batched), **{k: sum(r[k] for r in batched) for k in component_keys}),
                **{k: sum(r[k] for r in single_queries+batched) for k in component_keys})
        timing = dict(checkpoint_epoch=epoch, includes_graph_construction=True,
            measurement_scope='Selected checkpoint final normal val evaluation: fixed first16/task batch1, remainder batched; every query once',
            includes=['CPU gather/dispatch and resident-device input slicing', 'input-only graph construction',
                      'dense encoder, recurrent workspaces and direct-logit readout', 'output transfer to CPU'],
            excludes=['disk input decoding/collation and cache population', 'cached node_geom feature preparation',
                      'model initialization/checkpoint loading', 'any solver execution'],
            input_cache='Original inputs and original-TRAIN geometry features already cached by the standard loader',
            summary=dict(queries=sum(r['queries'] for r in per_timing.values()),
                single_queries=sum(len(r['single_queries']) for r in per_timing.values()),
                remaining_batched_queries=sum(r['remaining_batched']['queries'] for r in per_timing.values()),
                total_selection_seconds=sum(r['total_seconds'] for r in per_timing.values()),
                graph_construction_seconds=sum(r['graph_construction_seconds'] for r in per_timing.values()),
                selector_forward_wall_seconds=sum(r['selector_forward_wall_seconds'] for r in per_timing.values()),
                validation_wall_seconds_including_metrics_and_prediction_io=time.monotonic()-validation_start),
            per_problem=per_timing)
        atomic_json(directory / 'inference_timing.json', timing)
    finally:
        model.train(was_training)
    return add_diagnostics(dict(per_problem=results, macro=aggregate(results), families=families(results)))


def train(arm, args, directory, data, started, prior_wall):
    protocol, loaders, raw, geometry, schedule, reference = data
    shared = (json.loads((args.root / 'replay_schedule.json').read_text()) if args.stage == 'finalize-b15'
              else shared_replay_schedule(args.root)) if arm == 'B' else None
    horizon = shared['epochs'] if shared is not None else args.epochs
    model, params = initialize(arm, geometry, args.device, args.seed)
    initial = torch.load(args.root / 'selector_initial.pt', map_location='cpu', weights_only=False)
    if state_hash(initial) != protocol['initial_model_hash'] or params != protocol['model_params'][arm]:
        raise ValueError('R63 shared initial state or arm definition changed')
    model.load_state_dict(initial, strict=True)
    config = dict(protocol, group=arm, model_params=params, shared_replay_schedule=shared, wandb_mode='offline')
    path = directory / 'args.json'
    if path.exists() and json.loads(path.read_text()) != config:
        raise ValueError('R63 arm configuration changed')
    atomic_json(path, config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scaler = torch.amp.GradScaler('cuda', enabled=args.device.startswith('cuda'))
    controller = CostController()
    history, best, best_model, checkpoint = [], None, None, None
    successful, retries, peak_lr, start_epoch, stop = 0, 0, 1e-4, 0, False
    counts = Counter({p: 0 for p in protocol['sizes']})
    if (directory / 'last.pt').exists():
        if not args.resume:
            raise ValueError('Resume the unchanged epoch-atomic R63 checkpoint')
        checkpoint = torch.load(directory / 'last.pt', map_location='cpu', weights_only=False)
        if checkpoint['args'] != config or checkpoint['initial_model_hash'] != protocol['initial_model_hash']:
            raise ValueError('R63 resume source/data/configuration changed')
        model.load_state_dict(checkpoint['model'], strict=True)
        optimizer.load_state_dict(checkpoint['optimizer'])
        scaler.load_state_dict(checkpoint['scaler'])
        controller.__dict__.update(checkpoint['controller'])
        history, best, best_model = checkpoint['history'], checkpoint['best'], checkpoint['best_model']
        successful, retries = checkpoint['successful_updates'], checkpoint['amp_retries']
        counts = Counter(checkpoint['updates_by_problem'])
        peak_lr, start_epoch, stop = checkpoint['peak_lr'], checkpoint['epoch']+1, checkpoint['stop']
        prior_wall = max(prior_wall, checkpoint['wall_seconds'])
        if args.stage != 'finalize-b15':
            publish_best(directory, checkpoint)
    amendment = None
    if args.stage == 'finalize-b15':
        amendment = b15_amendment(args.root)
        if (arm != 'B' or amendment is None or checkpoint is None or start_epoch != 15
                or len(history) != 15 or successful != 21330 or best['epoch'] > 15
                or history[-1]['train'] is None or len(history[-1]['val']['per_problem']) != 18
                or len(history[-1]['train']['per_problem']) != 18):
            raise ValueError('B15 finalization requires the committed complete epoch15; no further updates allowed')
        horizon = 15
        print('[R63 B15 finalization] committed epoch15; ZERO optimizer updates; original checkpoint config retained', flush=True)
    wandb_run = None
    try:
        if args.wandb:
            import wandb
            wandb_run = wandb.init(**protocol['wandb'], name='R63_' + directory.name, config=config, dir=str(directory))
        restore_rng(checkpoint['rng']) if checkpoint is not None else set_seed(args.seed)
        steps = len(schedule['orders'][0])
        for epoch in range(start_epoch, horizon):
            if stop:
                break
            model.train()
            if args.device.startswith('cuda'):
                torch.cuda.reset_peak_memory_stats(args.device)
                torch.cuda.synchronize(args.device)
            epoch_start, cursor, totals = time.monotonic(), Counter(), Counter()
            for step, problem in enumerate(schedule['orders'][epoch]):
                offset = cursor[problem] * args.batch_size
                indices = schedule['permutations'][epoch][problem][offset:offset+args.batch_size]
                progress = min(1., (successful+1) / (3*steps))
                lr = 1e-4*progress if epoch < 3 else shared['lr'][epoch] if shared is not None else peak_lr
                for group in optimizer.param_groups:
                    group['lr'] = lr
                batch = gather_batch(loaders['train'][problem], indices)
                if successful == 0:
                    synchronize(batch['ind'].device)
                    first_started = time.monotonic()
                values = update_batch(model, optimizer, scaler, batch, .35*progress)
                if successful == 0:
                    synchronize(batch['ind'].device)
                    first = dict(problem=problem, rows=len(indices), microbatch_size=16,
                                 seconds=time.monotonic()-first_started, finite_gradient=True, **values)
                    if args.device.startswith('cuda'):
                        first['peak_memory_gib'] = torch.cuda.max_memory_allocated(args.device)/2**30
                    atomic_json(directory / 'first_formal_update.json', first)
                    print('[R63 first FORMAL update] ' + json.dumps(first), flush=True)
                cursor[problem] += 1
                successful += 1
                counts[problem] += 1
                retries += values['skipped']
                for key in LOSS_FIELDS:
                    totals[key] += values[key] * len(indices)
                totals['samples'] += len(indices)
                if (step+1) % 180 == 0 or step+1 == steps:
                    print(f'[R63{arm}] epoch={epoch+1} update={step+1}/{steps} total={successful} '
                          f'loss={totals["loss"]/totals["samples"]:.5f} lr={lr:.3g}', flush=True)
            if totals['samples'] != sum(protocol['sizes'].values()) or dict(cursor) != protocol['updates_per_epoch']:
                raise ValueError('Every original example and tail must be covered')
            if args.device.startswith('cuda'):
                torch.cuda.synchronize(args.device)
            optimization = {k: totals[k] / totals['samples'] for k in LOSS_FIELDS}
            optimization.update(lr=lr, consistency_weight=.35*progress, amp_retries=retries,
                                samples=totals['samples'], seconds=time.monotonic()-epoch_start)
            if args.device.startswith('cuda'):
                optimization['peak_memory_gib'] = torch.cuda.max_memory_allocated(args.device)/2**30
            validation = evaluate(model, loaders['val'], raw['val'], reference['val'], args.eval_batch_size)
            top1 = validation['macro']['macro_top1']
            if not math.isfinite(top1):
                raise ValueError('Nonfinite R63 validation native Top1')
            halve, stop = controller.observe(-100*top1, epoch+1) if arm == 'A' else (False, False)
            training = (evaluate(model, loaders['train'], raw['train'], reference['train'], args.eval_batch_size)
                        if (epoch+1) % 5 == 0 or stop or epoch+1 == horizon else None)
            row = dict(epoch=epoch+1, successful_updates=successful, updates_by_problem=dict(counts),
                       optimization=optimization, train=training, val=validation)
            history.append(row)
            if halve:
                peak_lr *= .5
            if best is None or top1 > best['val']['macro']['macro_top1']:
                best = row
                best_model = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            checkpoint = dict(model=model.state_dict(), optimizer=optimizer.state_dict(), scaler=scaler.state_dict(),
                args=config, epoch=epoch, successful_updates=successful, updates_by_problem=dict(counts),
                amp_retries=retries, controller=dict(vars(controller)), peak_lr=peak_lr, stop=stop,
                rng=capture_rng(), history=history, best=best, best_model=best_model,
                initial_model_hash=protocol['initial_model_hash'], wall_seconds=prior_wall+time.monotonic()-started)
            atomic_save(directory / 'last.pt', checkpoint)
            publish_best(directory, checkpoint)
            print_eval(f'R63{arm} val epoch {epoch+1}', validation['per_problem'], validation['macro'])
            if wandb_run is not None:
                metrics = dict(epoch=epoch+1, successful_updates=successful,
                               **{'optimization/' + k: v for k, v in optimization.items()})
                for split, value in (('train_eval', training), ('val', validation)):
                    if value is not None:
                        metrics.update({split + '/' + k: v for k, v in value['macro'].items()})
                wandb_run.log(metrics)
        if shared is not None and successful != shared['successful_updates'] and amendment is None:
            raise ValueError('R63 A/B successful update totals differ')
        model.load_state_dict(best_model, strict=True)
        best_eval = dict(best, train=evaluate(model, loaders['train'], raw['train'], reference['train'], args.eval_batch_size),
                        val=selected_validation(model, loaders['val'], raw['val'], reference['val'],
                                                args.eval_batch_size, directory, best['epoch']))
        atomic_json(directory / 'best_eval.json', best_eval)
        last5 = {k: float(np.mean([r['val']['macro'][k] for r in history[-5:]])) for k in history[-1]['val']['macro']}
        atomic_json(directory / 'result.json', dict(best=best_eval, final=history[-1], last5_val=last5,
            epochs_completed=len(history), successful_updates=successful, updates_by_problem=dict(counts),
            wall_seconds=prior_wall+time.monotonic()-started, best_checkpoint_sha256=file_hash(directory / 'best.pt'),
            checkpoint_rule=protocol['checkpoint_rule'],
            paired_horizon=amendment is None, budget_amendment=amendment,
            stop_reason=('user-authorized B15 stop after full validation; unequal A40/B15 compute' if amendment is not None else
                         'matched A-controlled horizon' if arm == 'B' else
                         'validation native-Top1 early stop' if stop else 'maximum epochs'), test_read=False))
    finally:
        if wandb_run is not None:
            wandb_run.finish()


def run(arm, args, data):
    directory = args.root / RUN_NAMES[arm]
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / 'result.json').exists():
        if not args.resume:
            raise ValueError('Preserve completed R63 arms; use --resume')
        return
    with (directory / 'train.log').open('a') as stream, redirect_stdout(Tee(sys.stdout, stream)), redirect_stderr(Tee(sys.stderr, stream)):
        path = directory / 'wall_time.json'
        timing = json.loads(path.read_text()) if path.exists() else []
        prior, started, status = sum(row['seconds'] for row in timing), time.monotonic(), 'failed'
        try:
            train(arm, args, directory, data, started, prior)
            status = 'complete'
        finally:
            timing.append(dict(seconds=time.monotonic()-started, status=status))
            atomic_json(path, timing)
            if status == 'complete':
                result = json.loads((directory / 'result.json').read_text())
                result['wall_seconds'] = sum(row['seconds'] for row in timing)
                atomic_json(directory / 'result.json', result)


def lock_validation_checkpoints(args, protocol):
    records, validation, successes = {}, {}, []
    for arm, name in RUN_NAMES.items():
        directory = args.root / name
        result = json.loads((directory / 'result.json').read_text())
        path = directory / 'best.pt'
        if file_hash(path) != result['best_checkpoint_sha256']:
            raise ValueError('R63 validation-selected checkpoint changed')
        records[arm] = dict(path=str(path), sha256=result['best_checkpoint_sha256'], epoch=result['best']['epoch'],
                           val=result['best']['val']['macro'], model_params=protocol['model_params'][arm],
                           checkpoint_rule=protocol['checkpoint_rule'])
        validation[arm] = result['best']['val']
        successes.append((result['epochs_completed'], result['successful_updates']))
    if successes[0] != successes[1]:
        amendment = b15_amendment(args.root)
        if amendment is None or successes != [(40, 56880), (15, 21330)]:
            raise ValueError('R63 unequal budgets require the explicit A40/B15 user amendment')
        for record in records.values():
            record.update(paired_horizon=False, budget_amendment=amendment)
    path = args.root / 'locked_checkpoints.json'
    if path.exists() and json.loads(path.read_text()) != records:
        raise ValueError('R63 validation checkpoint lock is immutable')
    atomic_json(path, records)
    atomic_json(args.root / 'validation_results.json', validation)


def final_test(args):
    from .performance_targets import read_raw_costs
    from .solver_code_encoder import make_r45_model
    from .train import make_loader

    protocol = json.loads((args.root / 'protocol.json').read_text())
    training_locks = json.loads((args.root / 'locked_checkpoints.json').read_text())
    locks = json.loads((args.root / 'final_test_lock.json').read_text())
    if set(training_locks) != set(RUN_NAMES) or set(locks) != {'A', 'B', 'R60B'}:
        raise ValueError('Both completed R63 arms and the fixed R60B reference must be locked before TEST')
    source_files = {path: digest for path, digest in protocol['source_hashes'].items()
                    if Path(path).name not in ('r63_experiment.py', 'run_v4_r63.sh')}
    if any(file_hash(WORKSPACE / path) != digest for path, digest in source_files.items()):
        raise ValueError('R63 inference sources changed after training; no TEST')
    directories = {arm: args.root / name for arm, name in RUN_NAMES.items()}
    directories['R60B'] = WORKSPACE / 'code/V4/runs/R60_expanded_partial_supervision/B_expanded_seed2'
    results = {arm: json.loads((directory / 'result.json').read_text()) for arm, directory in directories.items()}
    budgets = [(results[a]['epochs_completed'], results[a]['successful_updates']) for a in RUN_NAMES]
    amendment = b15_amendment(args.root)
    if budgets[0] != budgets[1] and (amendment is None or budgets != [(40, 56880), (15, 21330)]):
        raise ValueError('Both R63 arms must complete their common horizon or the explicit A40/B15 amendment before TEST')
    for arm, record in locks.items():
        expected = directories[arm] / 'best.pt'
        result = results[arm]
        if (Path(record['path']).resolve() != expected.resolve() or record['sha256'] != file_hash(expected)
                or record['sha256'] != result['best_checkpoint_sha256']
                or record['epoch'] != result['best']['epoch']):
            raise ValueError(f'{arm}: TEST lock must identify the existing validation-selected checkpoint')
        if arm in RUN_NAMES and any(record[k] != training_locks[arm][k] for k in ('sha256', 'epoch')):
            raise ValueError('R63 final TEST lock differs from the preexisting validation lock')
    if locks['R60B']['epoch'] != 39:
        raise ValueError('R60B reference is the already validation-selected best39, not a new selection')
    if (args.root / 'test_results.json').exists():
        raise ValueError('Final TEST already completed; reuse its outputs without another evaluation')
    receipt = dict(status='started', locked_models=locks, selection_frozen_before_test=True,
        paired_horizon=budgets[0] == budgets[1], budget_amendment=amendment,
        policy='ONE original full18/fullpool TEST per fixed model; no TEST reselection or training',
        diagnostic_reference_top2='Two fixed best TRAIN mean-cost methods per task; not TEST-query nominations',
        inference_source_hashes=source_files,
        test_entry_source_hashes={name: file_hash(Path(__file__).parent / name)
                                 for name in ('r63_experiment.py', 'run_v4_r63.sh')},
        slurm_step=os.environ.get('SLURM_JOB_ID', '') + '.' + os.environ.get('SLURM_STEP_ID', ''),
        gpu_uuid=os.environ.get('CUDA_VISIBLE_DEVICES'), started_unix=time.time())
    with (args.root / 'final_test_started.json').open('x') as stream:
        json.dump(receipt, stream, indent=2)
    started = time.monotonic()
    try:
        loaders, raw, reference = {}, {}, {}
        for problem in PROBLEMS:
            raw[problem] = read_raw_costs(problem, 'test')
            train_raw = read_raw_costs(problem, 'train')
            train_manifest = protocol['data_manifest']['train'][problem]
            if (raw[problem]['costs'].dtype != np.float64 or len(raw[problem]['winner']) != 1000
                    or raw[problem]['pool_ids'] != train_manifest['pool_ids']
                    or raw[problem]['pool'] != train_manifest['pool']
                    or any(train_raw[k] != train_manifest[k] for k in ('data_hash', 'label_hash', 'pool_ids', 'pool'))):
                raise ValueError('Final TEST must retain original input/label conventions and complete legal pools')
            loader = make_loader(problem, 'test', args.eval_batch_size, 0, coord_augment=0,
                                 shuffle=False, cache_device=args.device)
            loader.drop_last = False
            np.testing.assert_array_equal(loader.batch['ind'].cpu(), raw[problem]['winner'])
            np.testing.assert_array_equal(loader.batch['pool_ids'].cpu(), raw[problem]['pool_ids'])
            np.testing.assert_array_equal(loader.batch['costs'].cpu(), raw[problem]['costs'].astype(np.float32))
            if loader.size != 1000:
                raise ValueError('Final TEST must retain every query, including partial batches')
            loaders[problem] = loader
            top2 = np.argsort(train_raw['costs'].mean(0), kind='stable')[:2]
            reference[problem] = np.broadcast_to(top2, (loader.size, 2)).copy()
        receipt['test_data_manifest'] = {p: dict(instances=len(r['winner']),
            **{k: r[k] for k in ('data_hash', 'label_hash', 'pool_ids', 'pool')}) for p, r in raw.items()}
        atomic_json(args.root / 'test_protocol.json', receipt)
        evaluations = {}
        for arm in ('A', 'B', 'R60B'):
            checkpoint = torch.load(locks[arm]['path'], map_location='cpu', weights_only=False)
            if checkpoint['epoch']+1 != locks[arm]['epoch']:
                raise ValueError('Locked checkpoint epoch metadata differs')
            if arm in RUN_NAMES:
                model, params = initialize(arm, protocol['geometry_stats'], args.device, args.seed)
                if checkpoint['args']['model_params'] != params:
                    raise ValueError('Locked R63 model definition differs from training')
                prediction_dir = directories[arm] / 'test_predictions'
            else:
                model = make_r45_model(checkpoint['args']['model_params']).to(args.device)
                prediction_dir = args.root / 'reference/R60B/test_predictions'
            model.load_state_dict(checkpoint['model'], strict=True)
            geometry = model.instance_encoder.geometry_residual
            for key in ('mean', 'std'):
                expected = torch.tensor(protocol['geometry_stats'][key], device=args.device)
                if not torch.equal(getattr(geometry, key), expected):
                    raise ValueError('TEST must use the saved original-TRAIN geometry normalization')
            # No cache refit: the unchanged geometry branch computes input-only features on demand.
            evaluations[arm] = evaluate(model, loaders, raw, reference, args.eval_batch_size, prediction_dir)
            atomic_json(prediction_dir.parent / 'test_eval.json', evaluations[arm])
            print_eval(f'R63 final TEST {arm}', evaluations[arm]['per_problem'], evaluations[arm]['macro'])
            del model, checkpoint
        atomic_json(args.root / 'test_results.json', evaluations)
        receipt.update(status='complete', test_read=True, validation={a: r['best']['val'] for a, r in results.items()})
    except Exception as error:
        receipt.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        receipt.update(wall_seconds=time.monotonic()-started, finished_unix=time.time())
        atomic_json(args.root / 'test_protocol.json', receipt)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('plan', 'prepare', 'train', 'finalize-b15', 'test', 'analyze'), default='plan')
    parser.add_argument('--arm', choices=tuple(RUN_NAMES))
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--seed', type=int, choices=(2,), default=2)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--batch-size', type=int, choices=(128,), default=128)
    parser.add_argument('--eval-batch-size', type=int, default=128)
    parser.add_argument('--device', choices=('cuda:0', 'cpu'), default='cuda:0')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--wandb', action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args(argv)
    if not 15 <= args.epochs <= 40 or args.eval_batch_size < 1:
        parser.error('R63 uses min15/max40 epochs and a positive evaluation batch size')
    args.root, args.reference = args.root.resolve(), REFERENCE
    if args.stage in ('prepare', 'train', 'finalize-b15', 'test'):
        required_baton = {'test': 'locked-test', 'finalize-b15': 'b15-finalize'}.get(args.stage, 'recurrent-workspaces')
        if (args.device != 'cuda:0' or os.environ.get('R63_PARENT_GPU_BATON') != required_baton
                or os.environ.get('CUDA_VISIBLE_DEVICES') != GPU_UUID or os.environ.get('SLURM_JOB_ID') != '1465'):
            parser.error('R63 requires the explicit parent baton and assigned Slurm/UUID launcher')
        if args.stage == 'test' and args.arm is not None:
            parser.error('Final TEST evaluates all three locked models exactly once, not selected arms')
        if args.stage == 'finalize-b15':
            boundary = args.root / RUN_NAMES['B'] / 'b15_boundary_stop.json'
            if (args.arm not in (None, 'B') or not args.resume or b15_amendment(args.root) is None
                    or not boundary.exists() or not json.loads(boundary.read_text()).get('process_exited')):
                parser.error('B15 finalization requires its user amendment, completed own-process boundary stop and --resume')
            args.arm = 'B'
        configure_torch()
    torch.set_num_threads(1)
    args.root.mkdir(parents=True, exist_ok=True)
    if args.stage == 'plan':
        atomic_json(args.root / 'planned_configs.json', planned_config(args))
        print('[R63 plan only] No CUDA, model, solver, training or test access')
    if args.stage in ('prepare', 'train') and not (args.root / 'protocol.json').exists():
        prepare(args)
    if args.stage in ('train', 'finalize-b15'):
        data = load_training_data(args)
        for arm in (args.arm,) if args.arm else RUN_NAMES:
            run(arm, args, data)
        if all((args.root / name / 'result.json').exists() for name in RUN_NAMES.values()):
            lock_validation_checkpoints(args, data[0])
    if args.stage == 'analyze':
        from .r63_analysis import summarize
        summarize(args.root)
    if args.stage == 'test':
        final_test(args)


if __name__ == '__main__':
    main()

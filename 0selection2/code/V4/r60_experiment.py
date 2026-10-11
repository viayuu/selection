"""R60: scratch full-pool selection with size-matched replay or expanded pair supervision."""

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
from .r42_experiment import CostController, paired_schedule, prepare_data
from .r43_experiment import attach_targets, evaluate
from .r45_experiment import initialize as initialize_r45
from .r59_experiment import (GPU_UUID, REFERENCE, R43_ARGS, WORKSPACE, atomic_json, atomic_save,
                             cached_nominations, check_original_scope, publish_best)
from .r60_objective import AUXILIARY_WEIGHT, update_batch
from .train import configure_torch, print_eval, set_seed
from .training_monitor import capture_rng, restore_rng


ROOT = WORKSPACE / 'code/V4/runs/R60_expanded_partial_supervision'
RUN_NAMES = {'A': 'A_replay_seed2', 'B': 'B_expanded_seed2'}
MVRP = tuple(p for p in PROBLEMS if p not in ('TSP', 'CVRP', 'ATSP'))
LOSS_FIELDS = ('loss', 'base_loss', 'ce', 'cmp', 'risk', 'kl', 'true3', 'pred3', 'all_pairs', 'grad_norm')
NEW_SOURCES = ('r60_experiment.py', 'r60_objective.py', 'r60_data.py', 'run_v4_r60.sh',
               'r59_experiment.py', 'r45_experiment.py', 'r43_experiment.py', 'r42_experiment.py',
               'r40_experiment.py', 'pairwise_objective.py', 'pairwise_selector.py',
               'solver_code_encoder.py', 'solver_code_embeddings.py', 'solver_code_defaults.py',
               'solver_code_corpus.py', 'solver_code_views.py')


def initialize(geometry, device, seed=2):
    return initialize_r45('A', geometry, device, seed=seed)


def planned_config(args):
    old = json.loads(R43_ARGS.read_text())
    fixed = dict(seed=2, batch_size=128, learning_rate=1e-4, weight_decay=1e-4, dropout=.1,
                 warmup_epochs=3, rdrop_weight=.35, rdrop_ramp_epochs=3, lr_patience=3,
                 min_epochs=15, stop_patience=8, cost_min_delta_pct=.001)
    if any(old.get(k) != v for k, v in fixed.items()):
        raise ValueError('R60 requires the unchanged R43/R45A training rules')
    return dict(fixed, max_epochs=args.epochs, eval_batch_size=args.eval_batch_size, clip_norm=1.,
                arms=RUN_NAMES, solver_names=list(GLOBAL_SOLVERS), problems=list(PROBLEMS),
                loss=old['loss'], auxiliary_weight=AUXILIARY_WEIGHT,
                auxiliary_pair=['RELD_MOEL', 'RELD_MTL'], auxiliary_forward_pool='full original seven',
                auxiliary_rows_per_task_epoch=10000, expanded_rows_per_task=90000,
                auxiliary_costs='original FP64; exact ties excluded from loss, never resampled',
                auxiliary_sampling='B shuffled nine-epoch traversal; A same-task/true-size replacement',
                initialization='identical scratch R45A handcrafted ScoreDifferenceSelector; no transfer',
                full_coverage=True, drop_last=False, no_augmentation=True,
                dropout_passes=2, auxiliary_dropout_passes=1, optimizer_updates_per_batch=1,
                checkpoint_rule='strict minimum validation macro_vs_sbs_pct',
                shared_controller='A determines LR trace and horizon; B follows exactly, selects its own best',
                geometry_fitted_on='original TRAIN only', test_read=False,
                wandb=dict(project='selector', entity='yjkds-southern-university-of-science-technology', mode='offline'))


def source_hashes():
    files = [Path(__file__).parent / n for n in sorted(set(SOURCE_FILES + NEW_SOURCES))]
    files += [WORKSPACE / 'code/unified_selector' / n for n in ('registry.py', 'data.py')]
    return {str(p.relative_to(WORKSPACE)): file_hash(p) for p in files}


def verify_sources(protocol):
    if source_hashes() != protocol['source_hashes'] or file_hash(R43_ARGS) != protocol['r43_args_sha256']:
        raise ValueError('R60 training sources/configuration changed after preparation')


def expanded_data(args):
    from . import r60_data
    source = args.source_root if args.source_root is not None else Path(r60_data.SOURCE_ROOT)
    return r60_data.prepare_expanded(args.root / 'expanded_cache', source)


def extra_schedule(datasets, original_lengths, epochs, seed=2, rows_per_epoch=10000):
    from .r60_data import paired_auxiliary_schedule
    plan = paired_auxiliary_schedule(datasets, original_lengths, epochs=epochs,
                                     epoch_size=rows_per_epoch, seed=seed)
    return {a: {p: [torch.from_numpy(indices) for indices in rows] for p, rows in entries.items()}
            for a, entries in plan.items()}


def auxiliary_batch(arm, problem, indices, loaders, raw, datasets, device):
    if arm == 'B':
        return datasets[problem].gather(indices, device)
    original = raw[problem]
    columns = [original['pool'].index(name) for name in ('RELD_MOEL', 'RELD_MTL')]
    costs = original['costs'][indices.numpy()][:, columns]
    return inputs_only(gather_batch(loaders[problem], indices)), torch.from_numpy(costs)


def load_inputs(args):
    loaders, raw, geometry = prepare_data(args)
    check_original_scope(loaders, raw)
    for split in loaders:
        attach_targets(loaders[split], raw[split])
    datasets = expanded_data(args)
    if set(datasets) != set(MVRP):
        raise ValueError('R60 requires the 30 completed ReLD columns across all 15 MVRP tasks')
    counts = {}
    for problem, dataset in datasets.items():
        costs = dataset.pair_costs
        indices = np.asarray(dataset.source_indices)
        if (costs.dtype != np.float64 or costs.shape != (90000, 2) or len(dataset.nodes) != 90000
                or not np.array_equal(np.sort(indices), np.arange(10000, 100000))):
            raise ValueError('Expansion must contain exactly the 90k new rows and original FP64 cost pairs')
        counts[problem] = dict(original_rows=10000, expanded_rows=len(indices), source_index_min=int(indices.min()),
                               source_index_max=int(indices.max()), pair_columns=2)
    reference, references = cached_nominations(raw)
    manifest = {p: dataset.manifest for p, dataset in datasets.items()}
    return loaders, raw, geometry, datasets, manifest, counts, reference, references


def prepare(args):
    if (args.root / 'protocol.json').exists():
        return
    protocol = planned_config(args)
    protocol.update(source_hashes=source_hashes(), r43_args_sha256=file_hash(R43_ARGS))
    snapshot = args.root / 'source_launch'
    for relative, digest in protocol['source_hashes'].items():
        destination = snapshot / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() and file_hash(destination) != digest:
            raise ValueError('Preserve the existing R60 launch snapshot')
        if not destination.exists():
            shutil.copy2(WORKSPACE / relative, destination)
    atomic_json(snapshot / 'hashes.json', protocol['source_hashes'])
    loaders, raw, geometry, datasets, manifest, counts, reference, references = load_inputs(args)
    sizes = {p: loader.size for p, loader in loaders['train'].items()}
    schedule = paired_schedule(sizes, args.epochs, args.batch_size, args.seed)
    extras = extra_schedule(datasets, {p: loaders['train'][p].lengths for p in MVRP}, args.epochs, args.seed)
    atomic_save(args.root / 'schedule_seed2.pt', dict(original=schedule, auxiliary=extras))
    model, params = initialize(geometry, args.device, args.seed)
    initial = dict(initial_model_hash=state_hash(model.state_dict()),
                   trainable_parameters=sum(p.numel() for p in model.parameters()))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scaler = torch.amp.GradScaler('cuda', enabled=args.device.startswith('cuda'))
    problem = next(p for p in schedule['orders'][0] if p in MVRP)
    original = gather_batch(loaders['train'][problem], schedule['permutations'][0][problem][:args.batch_size])
    extra = auxiliary_batch('B', problem, extras['B'][problem][0][:args.batch_size],
                            loaders['train'], raw['train'], datasets, args.device)
    model.train()
    torch.cuda.reset_peak_memory_stats(args.device)
    torch.cuda.synchronize(args.device)
    started = time.monotonic()
    witness = update_batch(model, optimizer, scaler, original, .35, extra)
    torch.cuda.synchronize(args.device)
    witness.update(problem=problem, batch=len(original['ind']), seconds=time.monotonic() - started,
                   peak_memory_gib=torch.cuda.max_memory_allocated(args.device) / 2**30)
    if not all(math.isfinite(v) for v in witness.values() if isinstance(v, (int, float))):
        raise ValueError('Nonfinite R60 combined-update witness')
    print('[R60 one-update] ' + json.dumps(witness), flush=True)
    protocol.update(data_manifest=json.loads((args.root / 'data_manifest.json').read_text()),
                    expanded_manifest=manifest, data_counts=counts, sizes=sizes,
                    updates_per_epoch={p: (n + args.batch_size - 1) // args.batch_size for p, n in sizes.items()},
                    geometry_stats=geometry, schedule_hash=state_hash(dict(original=schedule, auxiliary=extras)),
                    model_params={a: params for a in RUN_NAMES}, initialization_checks={a: initial for a in RUN_NAMES},
                    reference_hashes=references, runtime_witness=witness,
                    runtime=dict(torch_version=str(torch.__version__), gpu=torch.cuda.get_device_name(0)))
    atomic_json(args.root / 'protocol.json', protocol)
    del model, optimizer, scaler
    torch.cuda.empty_cache()
    print('[R60 prepared] One disposable combined update; both arms will start from identical scratch weights', flush=True)


def load_training_data(args):
    protocol = json.loads((args.root / 'protocol.json').read_text())
    verify_sources(protocol)
    for key, expected in (('seed', args.seed), ('batch_size', args.batch_size), ('max_epochs', args.epochs),
                          ('eval_batch_size', args.eval_batch_size)):
        if protocol[key] != expected:
            raise ValueError('R60 prepared configuration changed: ' + key)
    loaders, raw, geometry, datasets, manifest, counts, reference, references = load_inputs(args)
    if (manifest != protocol['expanded_manifest'] or counts != protocol['data_counts']
            or geometry != protocol['geometry_stats'] or references != protocol['reference_hashes']
            or json.loads((args.root / 'data_manifest.json').read_text()) != protocol['data_manifest']):
        raise ValueError('R60 source data, geometry or diagnostic references changed')
    schedule = torch.load(args.root / 'schedule_seed2.pt', map_location='cpu', weights_only=False)
    if state_hash(schedule) != protocol['schedule_hash']:
        raise ValueError('R60 original/auxiliary schedule changed')
    return protocol, loaders, raw, geometry, schedule['original'], schedule['auxiliary'], datasets, reference


def shared_replay_schedule(root):
    result = json.loads((root / RUN_NAMES['A'] / 'result.json').read_text())
    history = json.loads((root / RUN_NAMES['A'] / 'history.json').read_text())
    value = dict(controller_arm='A', epochs=result['epochs_completed'], successful_updates=result['successful_updates'],
                 lr=[row['optimization']['lr'] for row in history])
    if len(value['lr']) != value['epochs']:
        raise ValueError('Incomplete replay-controlled LR trace')
    path = root / 'replay_schedule.json'
    if path.exists() and json.loads(path.read_text()) != value:
        raise ValueError('R60 shared LR/horizon changed')
    atomic_json(path, value)
    return value


def train(arm, args, directory, data, started, prior_wall):
    protocol, loaders, raw, geometry, schedule, extras, datasets, reference = data
    shared = shared_replay_schedule(args.root) if arm == 'B' else None
    horizon = shared['epochs'] if shared is not None else args.epochs
    model, params = initialize(geometry, args.device, args.seed)
    initial_hash = state_hash(model.state_dict())
    if (initial_hash != protocol['initialization_checks'][arm]['initial_model_hash']
            or params != protocol['model_params'][arm]):
        raise ValueError('R60 scratch initialization changed')
    config = dict(protocol, group=arm, model_params=params, shared_replay_schedule=shared, wandb_mode='offline')
    path = directory / 'args.json'
    if path.exists() and json.loads(path.read_text()) != config:
        raise ValueError('R60 arm configuration changed')
    atomic_json(path, config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scaler = torch.amp.GradScaler('cuda', enabled=args.device.startswith('cuda'))
    controller = CostController()
    history, best, best_model = [], None, None
    successful, retries, peak_lr, start_epoch, stop = 0, 0, 1e-4, 0, False
    counts = Counter({p: 0 for p in protocol['sizes']})
    extra_counts = Counter({p: 0 for p in extras[arm]})
    checkpoint = None
    if (directory / 'last.pt').exists():
        if not args.resume:
            raise ValueError('Partial R60 arm exists; resume its unchanged protocol')
        checkpoint = torch.load(directory / 'last.pt', map_location='cpu', weights_only=False)
        if checkpoint['args'] != config or checkpoint['initial_model_hash'] != initial_hash:
            raise ValueError('R60 resume source/data/configuration changed')
        model.load_state_dict(checkpoint['model'], strict=True)
        optimizer.load_state_dict(checkpoint['optimizer'])
        scaler.load_state_dict(checkpoint['scaler'])
        controller.__dict__.update(checkpoint['controller'])
        history, best, best_model = checkpoint['history'], checkpoint['best'], checkpoint['best_model']
        successful, retries = checkpoint['successful_updates'], checkpoint['amp_retries']
        counts, extra_counts = Counter(checkpoint['updates_by_problem']), Counter(checkpoint['extra_rows_by_problem'])
        peak_lr, start_epoch, stop = checkpoint['peak_lr'], checkpoint['epoch'] + 1, checkpoint['stop']
        prior_wall = max(prior_wall, checkpoint['wall_seconds'])
        publish_best(directory, checkpoint)
    wandb_run = None
    try:
        if args.wandb:
            import wandb
            wandb_run = wandb.init(**protocol['wandb'], name='R60_' + directory.name, config=config, dir=str(directory))
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
                indices = schedule['permutations'][epoch][problem][offset:offset + args.batch_size]
                progress = min(1., (successful + 1) / (3 * steps))
                lr = (1e-4 * progress if epoch < 3 else
                      shared['lr'][epoch] if shared is not None else peak_lr)
                for group in optimizer.param_groups:
                    group['lr'] = lr
                auxiliary = None
                if problem in extras[arm]:
                    extra_indices = extras[arm][problem][epoch][offset:offset + len(indices)]
                    auxiliary = auxiliary_batch(arm, problem, extra_indices, loaders['train'], raw['train'],
                                                datasets, args.device)
                values = update_batch(model, optimizer, scaler, gather_batch(loaders['train'][problem], indices),
                                      .35 * progress, auxiliary)
                if not all(math.isfinite(v) for v in values.values()):
                    raise ValueError('Nonfinite R60 update; no alternative loss or skipped examples')
                cursor[problem] += 1
                successful += 1
                counts[problem] += 1
                retries += values['skipped']
                for key in LOSS_FIELDS:
                    totals[key] += values[key] * len(indices)
                totals['samples'] += len(indices)
                totals['extra_aux_bce_sum'] += values['extra_aux_bce'] * values['extra_aux_samples']
                for key in ('extra_aux_samples', 'extra_aux_rows', 'extra_aux_correct'):
                    totals[key] += values[key]
                if auxiliary is not None:
                    extra_counts[problem] += len(indices)
                if (step + 1) % 180 == 0 or step + 1 == steps:
                    print(f'[R60{arm}] epoch={epoch+1} update={step+1}/{steps} total={successful} '
                          f'loss={totals["loss"]/totals["samples"]:.5f} lr={lr:.3g}', flush=True)
            if totals['samples'] != sum(protocol['sizes'].values()) or dict(cursor) != protocol['updates_per_epoch']:
                raise ValueError('R60 must cover every original example, including tails')
            if args.device.startswith('cuda'):
                torch.cuda.synchronize(args.device)
            valid = totals['extra_aux_samples']
            optimization = {k: totals[k] / totals['samples'] for k in LOSS_FIELDS}
            optimization.update(lr=lr, consistency_weight=.35 * progress, amp_skipped=retries,
                samples=totals['samples'], seconds=time.monotonic() - epoch_start,
                extra_aux_bce=totals['extra_aux_bce_sum'] / max(1, valid),
                extra_aux_accuracy=totals['extra_aux_correct'] / max(1, valid), extra_aux_samples=valid,
                extra_aux_rows=totals['extra_aux_rows'], extra_aux_ties=totals['extra_aux_rows'] - valid)
            if args.device.startswith('cuda'):
                optimization['peak_memory_gib'] = torch.cuda.max_memory_allocated(args.device) / 2**30
            validation = evaluate(model, loaders['val'], raw['val'], reference['val'], args.eval_batch_size,
                                  directory / 'val_predictions' / f'epoch{epoch+1:03d}')
            cost = validation['macro']['macro_vs_sbs_pct']
            if not math.isfinite(cost):
                raise ValueError('Nonfinite R60 validation selection cost')
            halve, stop = controller.observe(cost, epoch + 1) if arm == 'A' else (False, False)
            training = (evaluate(model, loaders['train'], raw['train'], reference['train'], args.eval_batch_size)
                        if (epoch + 1) % 5 == 0 or stop or epoch + 1 == horizon else None)
            row = dict(epoch=epoch+1, successful_updates=successful, updates_by_problem=dict(counts),
                       extra_rows_by_problem=dict(extra_counts), optimization=optimization, train=training, val=validation)
            history.append(row)
            if halve:
                peak_lr *= .5
            if best is None or cost < best['val']['macro']['macro_vs_sbs_pct']:
                best = row
                best_model = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            checkpoint = dict(model=model.state_dict(), optimizer=optimizer.state_dict(), scaler=scaler.state_dict(),
                args=config, epoch=epoch, successful_updates=successful, updates_by_problem=dict(counts),
                extra_rows_by_problem=dict(extra_counts), amp_retries=retries, controller=dict(vars(controller)),
                peak_lr=peak_lr, stop=stop, rng=capture_rng(), history=history, best=best, best_model=best_model,
                initial_model_hash=initial_hash, wall_seconds=prior_wall + time.monotonic() - started)
            atomic_save(directory / 'last.pt', checkpoint)
            publish_best(directory, checkpoint)
            print_eval(f'R60{arm} val epoch {epoch+1}', validation['per_problem'], validation['macro'])
            if wandb_run is not None:
                metrics = dict(epoch=epoch+1, successful_updates=successful,
                               **{'optimization/' + k: v for k, v in optimization.items()})
                for split, value in (('train_eval', training), ('val', validation)):
                    if value is not None:
                        metrics.update({split + '/' + k: v for k, v in value['macro'].items()})
                wandb_run.log(metrics)
        if shared is not None and successful != shared['successful_updates']:
            raise ValueError('R60 A/B successful update totals differ')
        model.load_state_dict(best_model, strict=True)
        best_eval = dict(best, train=evaluate(model, loaders['train'], raw['train'], reference['train'], args.eval_batch_size),
                        val=evaluate(model, loaders['val'], raw['val'], reference['val'], args.eval_batch_size,
                                     directory / 'val_predictions/best'))
        atomic_json(directory / 'best_eval.json', best_eval)
        last5 = {k: float(np.mean([r['val']['macro'][k] for r in history[-5:]])) for k in history[-1]['val']['macro']}
        result = dict(best=best_eval, final=history[-1], last5_val=last5, epochs_completed=len(history),
                      successful_updates=successful, updates_by_problem=dict(counts),
                      extra_rows_by_problem=dict(extra_counts), wall_seconds=prior_wall + time.monotonic() - started,
                      stop_reason=('matched replay-controlled horizon' if arm == 'B' else
                                   'validation early stop' if stop else 'maximum epochs'),
                      best_checkpoint_sha256=file_hash(directory / 'best.pt'), test_read=False)
        atomic_json(directory / 'result.json', result)
    finally:
        if wandb_run is not None:
            wandb_run.finish()


def run(arm, args, data):
    directory = args.root / RUN_NAMES[arm]
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / 'result.json').exists():
        if not args.resume:
            raise ValueError('Completed R60 arm exists; preserve it or resume to skip it')
        return
    path = directory / 'wall_time.json'
    timing = json.loads(path.read_text()) if path.exists() else []
    prior, started, status = sum(r['seconds'] for r in timing), time.monotonic(), 'failed'
    try:
        with (directory / 'train.log').open('a') as stream, redirect_stdout(Tee(sys.stdout, stream)), redirect_stderr(Tee(sys.stderr, stream)):
            train(arm, args, directory, data, started, prior)
        status = 'complete'
    finally:
        timing.append(dict(seconds=time.monotonic() - started, status=status))
        atomic_json(path, timing)
        if status == 'complete':
            result = json.loads((directory / 'result.json').read_text())
            result['wall_seconds'] = sum(r['seconds'] for r in timing)
            atomic_json(directory / 'result.json', result)


def lock_validation_checkpoints(args, protocol):
    records, validation, successes = {}, {}, []
    for arm, name in RUN_NAMES.items():
        directory = args.root / name
        result = json.loads((directory / 'result.json').read_text())
        path = directory / 'best.pt'
        if file_hash(path) != result['best_checkpoint_sha256']:
            raise ValueError('R60 validation-selected checkpoint changed')
        records[arm] = dict(path=str(path), sha256=result['best_checkpoint_sha256'], epoch=result['best']['epoch'],
                            val=result['best']['val']['macro'], model_params=protocol['model_params'][arm])
        validation[arm] = result['best']['val']
        successes.append((result['epochs_completed'], result['successful_updates']))
    if successes[0] != successes[1]:
        raise ValueError('R60 arms must complete identical horizons and successful update counts')
    path = args.root / 'locked_checkpoints.json'
    if path.exists() and json.loads(path.read_text()) != records:
        raise ValueError('R60 validation checkpoint lock is immutable')
    atomic_json(path, records)
    atomic_json(args.root / 'validation_results.json', validation)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('plan', 'prepare', 'train', 'analyze', 'all'), default='plan')
    parser.add_argument('--arm', '--group', choices=tuple(RUN_NAMES))
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--source-root', type=Path)
    parser.add_argument('--seed', type=int, choices=(2,), default=2)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--batch-size', type=int, choices=(128,), default=128)
    parser.add_argument('--eval-batch-size', type=int, default=128)
    parser.add_argument('--device', choices=('cuda:0', 'cpu'), default='cuda:0')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--wandb', action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args(argv)
    if not 15 <= args.epochs <= 40 or args.eval_batch_size < 1:
        parser.error('R60 uses min15/max40 epochs and a positive evaluation batch size')
    args.root, args.reference = args.root.resolve(), REFERENCE
    if args.stage in ('prepare', 'train', 'all'):
        if (args.device != 'cuda:0' or os.environ.get('R60_PARENT_GPU_BATON') != 'expanded-partial-supervision'
                or os.environ.get('CUDA_VISIBLE_DEVICES') != GPU_UUID or os.environ.get('SLURM_JOB_ID') != '1465'):
            parser.error('R60 requires the explicit parent baton and assigned Slurm/UUID launcher')
        configure_torch()
    torch.set_num_threads(1)
    args.root.mkdir(parents=True, exist_ok=True)
    if args.stage == 'plan':
        atomic_json(args.root / 'planned_configs.json', planned_config(args))
        print('[R60 plan only] No CUDA, model, solve, training or test access')
    if args.stage in ('prepare', 'all') or args.stage == 'train' and not (args.root / 'protocol.json').exists():
        prepare(args)
    if args.stage in ('train', 'all'):
        data = load_training_data(args)
        for arm in (args.arm,) if args.arm else RUN_NAMES:
            run(arm, args, data)
        if all((args.root / name / 'result.json').exists() for name in RUN_NAMES.values()):
            lock_validation_checkpoints(args, data[0])
    if args.stage in ('analyze', 'all'):
        from .r60_analysis import summarize
        summarize(args.root)


if __name__ == '__main__':
    main()

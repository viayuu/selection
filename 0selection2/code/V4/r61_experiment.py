"""R61 paired saved-route pretraining followed by original full-pool selection."""

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
from .performance_experiment import SOURCE_FILES
from .r42_experiment import CostController, paired_schedule, prepare_data
from .r43_experiment import attach_targets, evaluate, update_batch
from .r45_experiment import initialize as initialize_r45
from .r59_experiment import (GPU_UUID, REFERENCE, R43_ARGS, WORKSPACE, atomic_json, atomic_save,
                             cached_nominations, check_original_scope, publish_best)
from . import r61_pretraining as behavior
from .train import configure_torch, print_eval, set_seed
from .training_monitor import capture_rng, restore_rng


ROOT = WORKSPACE / 'code/V4/runs/R61_behavior_pretraining'
RUN_NAMES = {'A': 'A_replay_seed2', 'B': 'B_expanded_seed2'}
MVRP = tuple(PROBLEMS[3:])
LOSS_FIELDS = ('loss', 'ce', 'cmp', 'risk', 'kl', 'true3', 'pred3', 'all_pairs', 'grad_norm')
NEW_SOURCES = ('r61_experiment.py', 'r61_pretraining.py', 'r61_data.py', 'run_v4_r61.sh',
               'r60_data.py', 'r59_experiment.py', 'r45_experiment.py', 'r43_experiment.py',
               'r42_experiment.py', 'r40_experiment.py', 'r54_model.py', 'r54_routes.py', 'pair_specialist.py',
               'pairwise_selector.py', 'pairwise_objective.py', 'solver_code_encoder.py',
               'solver_code_embeddings.py', 'solver_code_defaults.py', 'solver_code_corpus.py',
               'solver_code_views.py')


def initialize(geometry, device, seed=2):
    return initialize_r45('A', geometry, device, seed=seed)


def initialize_pretrainer(geometry, device, seed=2):
    selector, params = initialize(geometry, device, seed)
    return behavior.BehaviorPretrainer(selector.instance_encoder, params['embedding_dim']).to(device)


def selector_rest(state):
    return {k: v for k, v in state.items() if not k.startswith('instance_encoder.')}


def planned_config(args):
    historical = json.loads(R43_ARGS.read_text())
    fixed = dict(seed=2, batch_size=128, learning_rate=1e-4, weight_decay=1e-4, dropout=.1,
                 warmup_epochs=3, rdrop_weight=.35, rdrop_ramp_epochs=3, lr_patience=3,
                 min_epochs=15, stop_patience=8, cost_min_delta_pct=.001)
    loss = dict(ce=.35, comparison=.35, risk=.02, cost_scale=.01,
                focus_weights=dict(true3=.5, predicted3=.25, all_pairs=.25))
    if any(historical.get(k) != v for k, v in fixed.items()) or historical['loss'] != loss:
        raise ValueError('R61 requires the unchanged R43/R45A training objective and rules')
    return dict(fixed, max_epochs=args.epochs, eval_batch_size=args.eval_batch_size, clip_norm=1.,
        arms=RUN_NAMES, solver_names=list(GLOBAL_SOLVERS), problems=list(PROBLEMS), loss=loss,
        pretrain=dict(passes=2, rows_per_task_pass=90000, tasks=list(MVRP), successful_updates=21120,
            loss='exact R54 normalized successor CE, mean of two methods; no other objective',
            learning_rate='linear 10560-update warmup to 1e-4; cosine 10560 updates to 1e-5',
            sampling='B complete shuffled passes; A same-task/actual-size replacement',
            transfer='final encoder only, fresh full-pool optimizer; no validation routes'),
        initialization='shared scratch R45A encoder and R54 heads; identical fresh selector rest',
        execution_order=['A_pretrain', 'A_finetune', 'B_pretrain', 'B_finetune'],
        full_coverage=True, drop_last=False, no_augmentation=True, dropout_passes=2,
        optimizer_updates_per_batch=1, checkpoint_rule='strict minimum validation macro_vs_sbs_pct',
        shared_controller='A determines LR trace and horizon; B follows exactly',
        geometry_fitted_on='original TRAIN only', test_read=False,
        incumbent=dict(name='R60 B best39', macro_top1=.4936111111111112,
                       macro_actual_regret_pct=.9227890652257744),
        wandb=dict(project='selector', entity='yjkds-southern-university-of-science-technology', mode='offline'))


def source_hashes():
    files = [Path(__file__).parent / name for name in sorted(set(SOURCE_FILES + NEW_SOURCES))]
    files += [WORKSPACE / 'code/unified_selector' / name for name in ('registry.py', 'data.py')]
    return {str(path.relative_to(WORKSPACE)): file_hash(path) for path in files}


def load_inputs(args):
    from .r61_data import prepare_behavior
    saved = prepare_behavior(args.root / 'data_cache')
    if set(saved) != {'old', 'new'}:
        raise ValueError('R61 requires old replay and independent expanded behavior data')
    datasets = {'A': saved['old'], 'B': saved['new']}
    for arm, count in (('A', 10000), ('B', 90000)):
        if set(datasets[arm]) != set(MVRP) or any(len(d.nodes) != count for d in datasets[arm].values()):
            raise ValueError('All 15 complete TRAIN route datasets are required for both arms')
    manifest = {arm: {p: d.manifest for p, d in entries.items()} for arm, entries in datasets.items()}
    loaders, raw, geometry = prepare_data(args)
    check_original_scope(loaders, raw)
    for split in loaders:
        attach_targets(loaders[split], raw[split])
    reference, references = cached_nominations(raw)
    return loaders, raw, geometry, datasets, manifest, reference, references


def prepare(args):
    if (args.root / 'protocol.json').exists():
        return
    protocol = planned_config(args)
    protocol.update(source_hashes=source_hashes(), r43_args_sha256=file_hash(R43_ARGS))
    for relative, digest in protocol['source_hashes'].items():
        destination = args.root / 'source_launch' / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() and file_hash(destination) != digest:
            raise ValueError('Preserve the existing R61 launch snapshot')
        if not destination.exists():
            shutil.copy2(WORKSPACE / relative, destination)
    atomic_json(args.root / 'source_launch/hashes.json', protocol['source_hashes'])
    loaders, raw, geometry, datasets, manifest, reference, references = load_inputs(args)
    sizes = {p: loader.size for p, loader in loaders['train'].items()}
    schedules = dict(original=paired_schedule(sizes, args.epochs, args.batch_size, args.seed),
                     behavior=behavior.matched_schedule(datasets, args.batch_size, args.seed))
    if sum(map(len, schedules['behavior']['orders'])) != 21120:
        raise ValueError('R61 stage1 must contain exactly 21120 updates including tails')
    atomic_save(args.root / 'schedule_seed2.pt', schedules)
    selector, params = initialize(geometry, args.device, args.seed)
    selector_state = {k: v.detach().cpu().clone() for k, v in selector.state_dict().items()}
    atomic_save(args.root / 'selector_initial.pt', selector_state)
    del selector
    model = initialize_pretrainer(geometry, args.device, args.seed)
    pretrain_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    atomic_save(args.root / 'pretrain_initial.pt', pretrain_state)
    protocol.update(pretrain_initial_hash=state_hash(pretrain_state), selector_initial_hash=state_hash(selector_state),
        selector_rest_hash=state_hash(selector_rest(selector_state)), model_params={a: params for a in RUN_NAMES},
        pretrain_parameters=sum(p.numel() for p in model.parameters()))
    problem = schedules['behavior']['orders'][0][0]
    indices = schedules['behavior']['permutations']['B'][0][problem][:128]
    batch, targets = datasets['B'][problem].gather(indices, args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scaler = torch.amp.GradScaler('cuda', enabled=args.device.startswith('cuda'))
    if args.device.startswith('cuda'):
        torch.cuda.reset_peak_memory_stats(args.device)
        torch.cuda.synchronize(args.device)
    started = time.monotonic()
    witness = behavior.update_batch(model, optimizer, scaler, batch, targets)
    if args.device.startswith('cuda'):
        torch.cuda.synchronize(args.device)
        witness['peak_memory_gib'] = torch.cuda.max_memory_allocated(args.device) / 2**30
    witness.update(problem=problem, batch=len(indices), seconds=time.monotonic() - started,
                   disposable=True, formal_training_starts_from_shared_initial_state=True)
    print('[R61 single pretrain update] ' + json.dumps(witness), flush=True)
    protocol.update(runtime_witness=witness, geometry_stats=geometry, behavior_manifest=manifest,
        reference_hashes=references, data_manifest=json.loads((args.root / 'data_manifest.json').read_text()),
        sizes=sizes, updates_per_epoch={p: math.ceil(n / args.batch_size) for p, n in sizes.items()},
        schedule_hash=state_hash(schedules))
    atomic_json(args.root / 'protocol.json', protocol)
    print('[R61 prepared] One disposable update; next is formal A pretraining with reset weights/RNG', flush=True)


def load_training_data(args):
    protocol = json.loads((args.root / 'protocol.json').read_text())
    if (source_hashes() != protocol['source_hashes'] or file_hash(R43_ARGS) != protocol['r43_args_sha256']
            or any(protocol[k] != getattr(args, a) for k, a in
                   (('seed', 'seed'), ('max_epochs', 'epochs'), ('batch_size', 'batch_size'),
                    ('eval_batch_size', 'eval_batch_size')))):
        raise ValueError('R61 source/configuration changed after preparation')
    loaders, raw, geometry, datasets, manifest, reference, references = load_inputs(args)
    if (manifest != protocol['behavior_manifest'] or geometry != protocol['geometry_stats']
            or references != protocol['reference_hashes']
            or json.loads((args.root / 'data_manifest.json').read_text()) != protocol['data_manifest']):
        raise ValueError('R61 source data, original geometry or references changed')
    schedule = torch.load(args.root / 'schedule_seed2.pt', map_location='cpu', weights_only=False)
    if state_hash(schedule) != protocol['schedule_hash']:
        raise ValueError('R61 shared schedules changed')
    return protocol, loaders, raw, geometry, schedule, datasets, reference


def shared_replay_schedule(root):
    result = json.loads((root / RUN_NAMES['A'] / 'result.json').read_text())
    history = json.loads((root / RUN_NAMES['A'] / 'history.json').read_text())
    value = dict(controller_arm='A', epochs=result['epochs_completed'], successful_updates=result['successful_updates'],
                 lr=[row['optimization']['lr'] for row in history])
    if len(value['lr']) != value['epochs']:
        raise ValueError('Incomplete A-controlled LR trace')
    path = root / 'replay_schedule.json'
    if path.exists() and json.loads(path.read_text()) != value:
        raise ValueError('R61 shared LR/horizon changed')
    atomic_json(path, value)
    return value


def finetune(arm, args, directory, data, pretrain, started, prior_wall):
    protocol, loaders, raw, geometry, schedules, datasets, reference = data
    schedule = schedules['original']
    shared = shared_replay_schedule(args.root) if arm == 'B' else None
    horizon = shared['epochs'] if shared is not None else args.epochs
    model, params = initialize(geometry, args.device, args.seed)
    initial = torch.load(args.root / 'selector_initial.pt', map_location='cpu', weights_only=False)
    if (state_hash(initial) != protocol['selector_initial_hash'] or params != protocol['model_params'][arm]
            or state_hash(selector_rest(model.state_dict())) != protocol['selector_rest_hash']):
        raise ValueError('Identical fresh selector initialization changed')
    model.load_state_dict(initial, strict=True)
    encoder = torch.load(directory / 'pretrain/encoder_final.pt', map_location='cpu', weights_only=False)
    if state_hash(encoder) != pretrain['encoder_hash'] or pretrain['successful_updates'] != 21120:
        raise ValueError('Only the final, complete pretrained encoder may transfer')
    model.instance_encoder.load_state_dict(encoder, strict=True)
    initial_hash = state_hash(model.state_dict())
    config = dict(protocol, group=arm, model_params=params, shared_replay_schedule=shared,
                  pretrained_encoder_hash=pretrain['encoder_hash'], wandb_mode='offline')
    path = directory / 'args.json'
    if path.exists() and json.loads(path.read_text()) != config:
        raise ValueError('R61 arm configuration changed')
    atomic_json(path, config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scaler = torch.amp.GradScaler('cuda', enabled=args.device.startswith('cuda'))
    controller = CostController()
    history, best, best_model = [], None, None
    successful, retries, peak_lr, start_epoch, stop = 0, 0, 1e-4, 0, False
    counts = Counter({p: 0 for p in protocol['sizes']})
    checkpoint = None
    if (directory / 'last.pt').exists():
        if not args.resume:
            raise ValueError('Resume the unchanged R61 fine-tuning checkpoint')
        checkpoint = torch.load(directory / 'last.pt', map_location='cpu', weights_only=False)
        if checkpoint['args'] != config or checkpoint['initial_model_hash'] != initial_hash:
            raise ValueError('R61 resume source/data/configuration changed')
        model.load_state_dict(checkpoint['model'], strict=True)
        optimizer.load_state_dict(checkpoint['optimizer'])
        scaler.load_state_dict(checkpoint['scaler'])
        controller.__dict__.update(checkpoint['controller'])
        history, best, best_model = checkpoint['history'], checkpoint['best'], checkpoint['best_model']
        successful, retries = checkpoint['successful_updates'], checkpoint['amp_retries']
        counts = Counter(checkpoint['updates_by_problem'])
        peak_lr, start_epoch, stop = checkpoint['peak_lr'], checkpoint['epoch'] + 1, checkpoint['stop']
        prior_wall = max(prior_wall, checkpoint['wall_seconds'])
        publish_best(directory, checkpoint)
    wandb_run = None
    try:
        if args.wandb:
            import wandb
            wandb_run = wandb.init(**protocol['wandb'], name='R61_' + directory.name, config=config, dir=str(directory))
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
                values = update_batch(model, optimizer, scaler, gather_batch(loaders['train'][problem], indices),
                                      .35 * progress)
                if not all(math.isfinite(v) for v in values.values()):
                    raise ValueError('Nonfinite R61 full-pool update; no skipped examples or alternate objective')
                cursor[problem] += 1
                successful += 1
                counts[problem] += 1
                retries += values['skipped']
                for key in LOSS_FIELDS:
                    totals[key] += values[key] * len(indices)
                totals['samples'] += len(indices)
                if (step + 1) % 180 == 0 or step + 1 == steps:
                    print(f'[R61{arm} finetune] epoch={epoch+1} update={step+1}/{steps} total={successful} '
                          f'loss={totals["loss"]/totals["samples"]:.5f} lr={lr:.3g}', flush=True)
            if totals['samples'] != sum(protocol['sizes'].values()) or dict(cursor) != protocol['updates_per_epoch']:
                raise ValueError('Every original example and tail must be covered')
            if args.device.startswith('cuda'):
                torch.cuda.synchronize(args.device)
            optimization = {k: totals[k] / totals['samples'] for k in LOSS_FIELDS}
            optimization.update(lr=lr, consistency_weight=.35 * progress, amp_skipped=retries,
                                samples=totals['samples'], seconds=time.monotonic() - epoch_start)
            if args.device.startswith('cuda'):
                optimization['peak_memory_gib'] = torch.cuda.max_memory_allocated(args.device) / 2**30
            validation = evaluate(model, loaders['val'], raw['val'], reference['val'], args.eval_batch_size,
                                  directory / 'val_predictions' / f'epoch{epoch+1:03d}')
            cost = validation['macro']['macro_vs_sbs_pct']
            if not math.isfinite(cost):
                raise ValueError('Nonfinite R61 validation selection cost')
            halve, stop = controller.observe(cost, epoch + 1) if arm == 'A' else (False, False)
            training = (evaluate(model, loaders['train'], raw['train'], reference['train'], args.eval_batch_size)
                        if (epoch + 1) % 5 == 0 or stop or epoch + 1 == horizon else None)
            row = dict(epoch=epoch+1, successful_updates=successful, updates_by_problem=dict(counts),
                       optimization=optimization, train=training, val=validation)
            history.append(row)
            if halve:
                peak_lr *= .5
            if best is None or cost < best['val']['macro']['macro_vs_sbs_pct']:
                best = row
                best_model = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            checkpoint = dict(model=model.state_dict(), optimizer=optimizer.state_dict(), scaler=scaler.state_dict(),
                args=config, epoch=epoch, successful_updates=successful, updates_by_problem=dict(counts),
                amp_retries=retries, controller=dict(vars(controller)), peak_lr=peak_lr, stop=stop,
                rng=capture_rng(), history=history, best=best, best_model=best_model,
                initial_model_hash=initial_hash, wall_seconds=prior_wall + time.monotonic() - started)
            atomic_save(directory / 'last.pt', checkpoint)
            publish_best(directory, checkpoint)
            print_eval(f'R61{arm} val epoch {epoch+1}', validation['per_problem'], validation['macro'])
            if wandb_run is not None:
                metrics = dict(epoch=epoch+1, successful_updates=successful,
                               **{'optimization/' + k: v for k, v in optimization.items()})
                for split, value in (('train_eval', training), ('val', validation)):
                    if value is not None:
                        metrics.update({split + '/' + k: v for k, v in value['macro'].items()})
                wandb_run.log(metrics)
        if shared is not None and successful != shared['successful_updates']:
            raise ValueError('R61 A/B fine-tuning successful update totals differ')
        model.load_state_dict(best_model, strict=True)
        best_eval = dict(best, train=evaluate(model, loaders['train'], raw['train'], reference['train'], args.eval_batch_size),
                        val=evaluate(model, loaders['val'], raw['val'], reference['val'], args.eval_batch_size,
                                     directory / 'val_predictions/best'))
        atomic_json(directory / 'best_eval.json', best_eval)
        last5 = {k: float(np.mean([r['val']['macro'][k] for r in history[-5:]])) for k in history[-1]['val']['macro']}
        result = dict(best=best_eval, final=history[-1], last5_val=last5, epochs_completed=len(history),
                      successful_updates=successful, updates_by_problem=dict(counts),
                      wall_seconds=prior_wall + time.monotonic() - started,
                      pretrain_updates=pretrain['successful_updates'], pretrain_wall_seconds=pretrain['wall_seconds'],
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
            raise ValueError('Preserve completed R61 arms; use --resume')
        return
    with (directory / 'train.log').open('a') as stream, redirect_stdout(Tee(sys.stdout, stream)), redirect_stderr(Tee(sys.stderr, stream)):
        pretrain = behavior.train(arm, args, directory / 'pretrain', data[5], data[4]['behavior'],
            lambda: initialize_pretrainer(data[3], args.device, args.seed), data[0])
        path = directory / 'wall_time.json'
        timing = json.loads(path.read_text()) if path.exists() else []
        prior, started, status = sum(row['seconds'] for row in timing), time.monotonic(), 'failed'
        try:
            finetune(arm, args, directory, data, pretrain, started, prior)
            status = 'complete'
        finally:
            timing.append(dict(seconds=time.monotonic() - started, status=status))
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
            raise ValueError('R61 validation-selected checkpoint changed')
        records[arm] = dict(path=str(path), sha256=result['best_checkpoint_sha256'], epoch=result['best']['epoch'],
                            val=result['best']['val']['macro'], model_params=protocol['model_params'][arm])
        validation[arm] = result['best']['val']
        successes.append((result['epochs_completed'], result['successful_updates'], result['pretrain_updates']))
    if successes[0] != successes[1]:
        raise ValueError('R61 arms must have matching stage budgets')
    path = args.root / 'locked_checkpoints.json'
    if path.exists() and json.loads(path.read_text()) != records:
        raise ValueError('R61 validation checkpoint lock is immutable')
    atomic_json(path, records)
    atomic_json(args.root / 'validation_results.json', validation)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('plan', 'prepare', 'train', 'analyze', 'all'), default='plan')
    parser.add_argument('--arm', '--group', choices=tuple(RUN_NAMES))
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
        parser.error('R61 uses min15/max40 fine-tune epochs and a positive evaluation batch size')
    args.root, args.reference = args.root.resolve(), REFERENCE
    if args.stage in ('prepare', 'train', 'all'):
        if (args.device != 'cuda:0' or os.environ.get('R61_PARENT_GPU_BATON') != 'behavior-pretraining'
                or os.environ.get('CUDA_VISIBLE_DEVICES') != GPU_UUID or os.environ.get('SLURM_JOB_ID') != '1465'):
            parser.error('R61 requires the explicit parent baton and assigned Slurm/UUID launcher')
        configure_torch()
    torch.set_num_threads(1)
    args.root.mkdir(parents=True, exist_ok=True)
    if args.stage == 'plan':
        atomic_json(args.root / 'planned_configs.json', planned_config(args))
        print('[R61 plan only] No CUDA, model, solve, training or test access')
    if args.stage in ('prepare', 'all') or args.stage == 'train' and not (args.root / 'protocol.json').exists():
        prepare(args)
    if args.stage in ('train', 'all'):
        data = load_training_data(args)
        for arm in (args.arm,) if args.arm else RUN_NAMES:
            run(arm, args, data)
        if all((args.root / name / 'result.json').exists() for name in RUN_NAMES.values()):
            lock_validation_checkpoints(args, data[0])
    if args.stage in ('analyze', 'all'):
        from .r61_analysis import summarize
        summarize(args.root)


if __name__ == '__main__':
    main()

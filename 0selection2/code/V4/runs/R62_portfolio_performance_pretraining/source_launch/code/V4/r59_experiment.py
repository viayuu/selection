"""R59 paired trainable routing encoders; original train/validation only."""

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
import torch.nn.functional as F

from code.unified_selector.registry import GLOBAL_SOLVERS, PROBLEMS
from .multitask_probe import Tee, file_hash, gather_batch, state_hash
from .pairwise_objective import LABEL_KEYS, inputs_only
from .performance_experiment import SOURCE_FILES
from .r42_experiment import CostController, paired_schedule, prepare_data
from .r43_experiment import attach_targets, evaluate, update_batch
from .train import configure_torch, print_eval, set_seed
from .training_monitor import capture_rng, restore_rng


WORKSPACE = Path(__file__).resolve().parents[2]
ROOT = WORKSPACE / 'code/V4/runs/R59_trainable_solver_encoder'
REFERENCE = WORKSPACE / 'code/V4/runs/R39_performance_model'
R45 = WORKSPACE / 'code/V4/runs/R45_solver_code_embeddings'
R43_ARGS = WORKSPACE / 'code/V4/runs/R43_pairwise_selection/score_difference_seed2/args.json'
ORIGINAL_DATA = WORKSPACE / 'data'
RUN_NAMES = {'A': 'A_random_seed2', 'B': 'B_pretrained_seed2'}
GPU_UUID = 'GPU-cb626bdb-4319-2f5e-b91f-17fcaa44a83d'
LOSS_FIELDS = ('loss', 'ce', 'cmp', 'risk', 'kl', 'true3', 'pred3', 'all_pairs', 'grad_norm')
NEW_SOURCES = ('r59_experiment.py', 'r59_transfer.py', 'run_v4_r59.sh',
               'r43_experiment.py', 'r42_experiment.py', 'r40_experiment.py',
               'pairwise_selector.py', 'pairwise_objective.py', 'solver_code_encoder.py',
               'solver_code_embeddings.py', 'solver_code_defaults.py', 'solver_code_corpus.py',
               'solver_code_views.py', 'r45_experiment.py')


def initialize(arm, geometry, device, seed):
    from .r59_transfer import initialize as build
    return build(arm, geometry, device, seed)


def native_parameters(model):
    from .r59_transfer import native_parameters as parameters
    return parameters(model)


def atomic_json(path, value):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    os.replace(temporary, path)


def atomic_save(path, value):
    temporary = path.with_name(path.name + '.tmp')
    torch.save(value, temporary)
    os.replace(temporary, path)


def planned_config(args):
    historical = json.loads(R43_ARGS.read_text())
    fixed = dict(seed=2, batch_size=128, learning_rate=1e-4, weight_decay=1e-4,
                 dropout=.1, warmup_epochs=3, rdrop_weight=.35, rdrop_ramp_epochs=3,
                 lr_patience=3, min_epochs=15, stop_patience=8, cost_min_delta_pct=.001)
    if any(historical.get(k) != v for k, v in fixed.items()):
        raise ValueError('Historical R43/R45 training rules changed')
    loss = dict(ce=.35, comparison=.35, risk=.02, cost_scale=.01,
                focus_weights=dict(true3=.5, predicted3=.25, all_pairs=.25))
    if historical['loss'] != loss or historical['model_params']['pair_mode'] != 'score_difference':
        raise ValueError('R59 requires the exact R45A score-difference objective')
    return dict(fixed, max_epochs=args.epochs, eval_batch_size=args.eval_batch_size,
                native_learning_rate=3e-5, native_lr_ratio=.3, clip_norm=1., loss=loss,
                arms=RUN_NAMES, solver_names=list(GLOBAL_SOLVERS), problems=list(PROBLEMS),
                initialization='same trainable architecture; A random, B ReLD-MTL encoder weights only',
                base_model_params=historical['model_params'],
                full_coverage=True, drop_last=False, natural_sampling=True, no_augmentation=True,
                optimizer_updates_per_batch=1, dropout_passes=2,
                cost_dtype='original FP64 comparisons and regret; FP32 supervision and KL',
                checkpoint_rule='strict minimum validation macro_vs_sbs_pct',
                scheduling='same rules and paired seed2 batches; validation-driven LR/stopping per arm',
                screen=dict(B_top1_min_gain_over_R45A_pp=3., B_top1_min_gain_over_A_pp=2.,
                            B_actual_regret_no_worse_than_both=True, historical_top1_threshold_pct=52.1167),
                original_data_only=True, expansion_inputs=False, expansion_labels=False, test_read=False,
                pretraining_limitations=[
                    'External routing pretraining, not equal total training data.',
                    'Historical pretraining inputs/source revision and absence of overlap are not certified.',
                    'Legacy B-domain deployment warning remains unresolved.'],
                wandb=dict(project='selector', entity='yjkds-southern-university-of-science-technology', mode='offline'))


def make_optimizer(model):
    native = list(native_parameters(model))
    all_parameters = list(model.parameters())
    identities = {id(p) for p in native}
    if (not native or len(identities) != len(native)
            or not identities <= {id(p) for p in all_parameters}
            or not all(p.requires_grad for p in all_parameters)):
        raise ValueError('The native encoder must be a distinct, fully trainable parameter group')
    rest = [p for p in all_parameters if id(p) not in identities]
    if not rest:
        raise ValueError('Missing selector parameters outside the native encoder')
    return torch.optim.AdamW([dict(params=native, lr=3e-5, name='native', lr_ratio=.3),
                              dict(params=rest, lr=1e-4, name='selector', lr_ratio=1.)], weight_decay=1e-4)


def learning_rates(optimizer, successful, steps, epoch, peak):
    progress = min(1., (successful + 1) / (3 * steps))
    lr = 1e-4 * progress if epoch < 3 else peak
    for group in optimizer.param_groups:
        group['lr'] = lr * group['lr_ratio']
    return lr, progress


def common_parameters(model):
    native = {id(p) for p in native_parameters(model)}
    return {name: p for name, p in model.named_parameters() if id(p) not in native}


def source_hashes():
    files = [Path(__file__).parent / name for name in sorted(set(SOURCE_FILES + NEW_SOURCES))]
    files += [WORKSPACE / 'code/unified_selector' / name for name in ('registry.py', 'data.py')]
    return {str(p.relative_to(WORKSPACE)): file_hash(p) for p in files}


def verify_sources(protocol):
    if source_hashes() != protocol['source_hashes'] or file_hash(R43_ARGS) != protocol['r43_args_sha256']:
        raise ValueError('R59 training sources/configuration changed after preparation')
    for path, digest in protocol['reference_hashes'].items():
        if file_hash(path) != digest:
            raise ValueError('A fixed train/validation diagnostic reference changed: ' + path)


def verify_data_files(protocol):
    if set(protocol['data_manifest']) != {'train', 'val'}:
        raise ValueError('R59 provenance must contain training and validation only')
    for split, entries in protocol['data_manifest'].items():
        for problem, record in entries.items():
            directory = ORIGINAL_DATA / f'{problem}{split}'
            for name, key in (('dataset.pkl', 'data_hash'), ('raw_label.pkl', 'label_hash')):
                if file_hash(directory / name) != record[key]:
                    raise ValueError('R59 original input/labels changed: ' + str(directory / name))


def check_original_scope(loaders, raw):
    if set(raw) != {'train', 'val'} or set(loaders) != {'train', 'val'}:
        raise ValueError('Only original training and validation are authorized')
    for split, expected in (('train', 10000), ('val', 1000)):
        if set(raw[split]) != set(PROBLEMS) or set(loaders[split]) != set(PROBLEMS):
            raise ValueError('R59 requires all 18 original tasks')
        for problem in PROBLEMS:
            record = raw[split][problem]
            if (loaders[split][problem].size != expected or len(record['winner']) != expected
                    or record['costs'].dtype != np.float64):
                raise ValueError('R59 requires original split sizes and raw FP64 costs')


def cached_nominations(raw):
    # These references are diagnostics only; neither reference supplies training targets.
    best_path = R45 / 'A_handcrafted_seed2/best_eval.json'
    epoch = json.loads(best_path.read_text())['epoch']
    bindings, reference = {str(best_path): file_hash(best_path)}, {'train': {}, 'val': {}}
    for split in reference:
        directory = (R45 / 'reference_predictions/train' if split == 'train' else
                     R45 / 'A_handcrafted_seed2/val_predictions' / f'epoch{epoch:03d}')
        for problem, record in raw[split].items():
            path = directory / f'{problem}.npz'
            with np.load(path, allow_pickle=False) as saved:
                for key, expected in (('pool_ids', record['pool_ids']), ('winner', record['winner']),
                                      ('costs', record['costs']), ('indices', np.arange(len(record['winner'])))):
                    np.testing.assert_array_equal(saved[key], expected)
                scores = saved['logits'].copy()
            if scores.shape != record['costs'].shape or not np.isfinite(scores).all():
                raise ValueError('Invalid cached diagnostic logits: ' + str(path))
            order = np.argsort(record['pool_ids'], kind='stable')
            reference[split][problem] = order[np.argsort(-scores[:, order], axis=-1, kind='stable')][:, :2]
            bindings[str(path)] = file_hash(path)
    return reference, bindings


def real_preflight(model, loaders, args):
    optimizer = make_optimizer(model)
    scaler = torch.amp.GradScaler('cuda', enabled=args.device.startswith('cuda'))
    rows = []
    for problem in ('TSP', 'CVRP', 'ATSP', 'OVRPTW'):
        loader = loaders[problem]
        batch = gather_batch(loader, loader.lengths.argsort(descending=True)[:128])
        sample = gather_batch(loader, loader.lengths.argsort(descending=True)[:2])
        model.eval()
        with torch.no_grad():
            output = model(inputs_only(sample))
            poisoned = dict(sample)
            for key in LABEL_KEYS:
                if key in poisoned:
                    poisoned[key] = torch.full_like(poisoned[key], 0)
            torch.testing.assert_close(model(poisoned)['logits'], output['logits'], rtol=0, atol=0)
            padded = dict(sample)
            for key in ('node', 'node_geom'):
                if torch.is_tensor(sample.get(key)):
                    padded[key] = F.pad(sample[key], (0, 0, 0, 3), value=777.)
            if torch.is_tensor(sample.get('matrix')):
                padded['matrix'] = F.pad(sample['matrix'], (0, 3, 0, 3), value=777.)
            padded['node_mask'] = F.pad(sample['node_mask'], (0, 3), value=False)
            other = model(inputs_only(padded))['logits']
            torch.testing.assert_close(other, output['logits'], rtol=5e-5, atol=5e-5)
            if output['logits'].shape != sample['costs'].shape or not torch.isfinite(output['logits']).all():
                raise ValueError('Preflight must retain every legal solver')
            padding_error = float((other - output['logits']).abs().max())
        model.train()
        if args.device.startswith('cuda'):
            torch.cuda.reset_peak_memory_stats(args.device)
            torch.cuda.synchronize(args.device)
        started = time.monotonic()
        values = update_batch(model, optimizer, scaler, batch, .35)
        if not all(math.isfinite(v) for v in values.values()):
            raise ValueError('Nonfinite preflight objective/gradients')
        native_norm = None
        if problem in ('CVRP', 'OVRPTW'):
            gradients = [p.grad for p in native_parameters(model)]
            if any(g is None or not torch.isfinite(g).all() for g in gradients):
                raise ValueError('Missing or nonfinite native encoder gradients')
            native_norm = sum(float(g.float().square().sum()) for g in gradients) ** .5
            if native_norm <= 0:
                raise ValueError('The trainable native encoder did not receive gradients')
        if args.device.startswith('cuda'):
            torch.cuda.synchronize(args.device)
        row = dict(problem=problem, batch=len(batch['ind']), max_nodes=int(batch['n'].max()),
                   seconds=time.monotonic() - started, label_isolation=True,
                   padding_max_abs_error=padding_error, native_gradient_norm=native_norm, **values)
        if args.device.startswith('cuda'):
            row['peak_memory_gib'] = torch.cuda.max_memory_allocated(args.device) / 2**30
        rows.append(row)
        print('[R59 preflight] ' + json.dumps(row), flush=True)
    return rows


def prepare(args):
    if (args.root / 'protocol.json').exists():
        load_training_data(args)
        return
    config = planned_config(args)
    config.update(source_hashes=source_hashes(), r43_args_sha256=file_hash(R43_ARGS))
    snapshot = args.root / 'source_launch'
    snapshot.mkdir(parents=True, exist_ok=True)
    for relative in config['source_hashes']:
        destination = snapshot / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() and file_hash(destination) != config['source_hashes'][relative]:
            raise ValueError('Preserve the existing R59 launch snapshot')
        if not destination.exists():
            shutil.copy2(WORKSPACE / relative, destination)
    atomic_json(snapshot / 'hashes.json', config['source_hashes'])
    loaders, raw, geometry = prepare_data(args)
    check_original_scope(loaders, raw)
    for split in loaders:
        attach_targets(loaders[split], raw[split])
    reference, config['reference_hashes'] = cached_nominations(raw)
    config['data_manifest'] = json.loads((args.root / 'data_manifest.json').read_text())
    config['sizes'] = {p: loader.size for p, loader in loaders['train'].items()}
    config['updates_per_epoch'] = {p: (n + 127) // 128 for p, n in config['sizes'].items()}
    config['geometry_stats'] = geometry
    schedule = paired_schedule(config['sizes'], args.epochs, 128, 2)
    config['schedule_hash'] = state_hash(schedule)
    schedule_path = args.root / 'schedule_seed2.pt'
    if schedule_path.exists() and state_hash(torch.load(schedule_path, weights_only=False)) != config['schedule_hash']:
        raise ValueError('The paired R59 schedule changed')
    atomic_save(schedule_path, schedule)
    witnesses, model_params = {}, {}
    for arm in RUN_NAMES:
        model, model_params[arm] = initialize(arm, geometry, args.device, 2)
        native = list(native_parameters(model))
        witnesses[arm] = dict(common_parameters_hash=state_hash(common_parameters(model)),
                              initial_model_hash=state_hash(model.state_dict()),
                              trainable_parameters=sum(p.numel() for p in model.parameters()),
                              native_parameters=sum(p.numel() for p in native))
        witnesses[arm]['runtime'] = real_preflight(model, loaders['train'], args)
        del model
        if args.device.startswith('cuda'):
            torch.cuda.empty_cache()
    if witnesses['A']['common_parameters_hash'] != witnesses['B']['common_parameters_hash']:
        raise ValueError('A/B non-native initial parameters differ')
    if witnesses['A']['initial_model_hash'] == witnesses['B']['initial_model_hash']:
        raise ValueError('A/B must differ in native encoder initialization')
    config.update(initialization_checks=witnesses, model_params=model_params,
                  runtime=dict(torch_version=str(torch.__version__), device=args.device,
                               gpu=torch.cuda.get_device_name(0) if args.device.startswith('cuda') else None))
    verify_sources(config)
    atomic_json(args.root / 'protocol.json', config)
    print('[R59 prepared] Disposable updates passed; both training arms start fresh; test_read=False', flush=True)


def load_training_data(args):
    protocol = json.loads((args.root / 'protocol.json').read_text())
    verify_sources(protocol)
    for key, expected in (('seed', args.seed), ('batch_size', args.batch_size), ('max_epochs', args.epochs),
                          ('eval_batch_size', args.eval_batch_size)):
        if protocol[key] != expected:
            raise ValueError('R59 prepared configuration changed: ' + key)
    if json.loads((args.root / 'data_manifest.json').read_text()) != protocol['data_manifest']:
        raise ValueError('R59 prepared data manifest changed')
    loaders, raw, geometry = prepare_data(args)
    check_original_scope(loaders, raw)
    if geometry != protocol['geometry_stats']:
        raise ValueError('Train-fitted geometry statistics changed')
    for split in loaders:
        attach_targets(loaders[split], raw[split])
    schedule = torch.load(args.root / 'schedule_seed2.pt', map_location='cpu', weights_only=False)
    if state_hash(schedule) != protocol['schedule_hash']:
        raise ValueError('R59 paired batch schedule changed')
    reference, hashes = cached_nominations(raw)
    if hashes != protocol['reference_hashes']:
        raise ValueError('R59 fixed diagnostic references changed')
    return protocol, loaders, raw, geometry, schedule, reference


def publish_best(directory, checkpoint):
    # last.pt is the epoch commit; reconstruct derived files after an interrupted publication.
    best = checkpoint['best']
    atomic_save(directory / 'best.pt', dict(model=checkpoint['best_model'], args=checkpoint['args'],
        epoch=best['epoch'] - 1, macro=best['val']['macro'], successful_updates=best['successful_updates']))
    atomic_json(directory / 'history.json', checkpoint['history'])
    atomic_json(directory / 'best_eval.json', best)


def train(arm, args, directory, data, started, prior_wall):
    protocol, loaders, raw, geometry, schedule, reference = data
    model, params = initialize(arm, geometry, args.device, 2)
    initial_hash = state_hash(model.state_dict())
    if (initial_hash != protocol['initialization_checks'][arm]['initial_model_hash']
            or params != protocol['model_params'][arm]):
        raise ValueError('R59 model/initial weights changed after preflight')
    config = dict(protocol, group=arm, model_params=params, wandb_mode='offline')
    config_path = directory / 'args.json'
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError('R59 arm configuration changed')
    atomic_json(config_path, config)
    optimizer = make_optimizer(model)
    scaler = torch.amp.GradScaler('cuda', enabled=args.device.startswith('cuda'))
    controller = CostController()
    history, best, best_model = [], None, None
    successful, retries, peak_lr, start_epoch, stop = 0, 0, 1e-4, 0, False
    counts = Counter({p: 0 for p in protocol['sizes']})
    checkpoint = None
    if (directory / 'last.pt').exists():
        if not args.resume:
            raise ValueError('Partial R59 arm exists; use --resume without changing its protocol')
        checkpoint = torch.load(directory / 'last.pt', map_location='cpu', weights_only=False)
        if checkpoint['args'] != config or checkpoint['initial_model_hash'] != initial_hash:
            raise ValueError('R59 resume source/data/model configuration changed')
        model.load_state_dict(checkpoint['model'], strict=True)
        optimizer.load_state_dict(checkpoint['optimizer'])
        scaler.load_state_dict(checkpoint['scaler'])
        controller.__dict__.update(checkpoint['controller'])
        history, best, best_model = checkpoint['history'], checkpoint['best'], checkpoint['best_model']
        successful, retries = checkpoint['successful_updates'], checkpoint['amp_retries']
        counts, peak_lr = Counter(checkpoint['updates_by_problem']), checkpoint['peak_lr']
        start_epoch, stop = checkpoint['epoch'] + 1, checkpoint['stop']
        prior_wall = max(prior_wall, checkpoint['wall_seconds'])
        publish_best(directory, checkpoint)
    wandb_run = None
    try:
        if args.wandb:
            import wandb
            wandb_run = wandb.init(**protocol['wandb'], name='R59_' + directory.name,
                                   config=config, dir=str(directory))
        if checkpoint is None:
            initial = {s: evaluate(model, loaders[s], raw[s], reference[s], args.eval_batch_size)
                       for s in ('train', 'val')}
            atomic_json(directory / 'initial_eval.json', initial)
            set_seed(2)
        else:
            restore_rng(checkpoint['rng'])
        steps = len(schedule['orders'][0])
        for epoch in range(start_epoch, args.epochs):
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
                lr, progress = learning_rates(optimizer, successful, steps, epoch, peak_lr)
                values = update_batch(model, optimizer, scaler,
                                      gather_batch(loaders['train'][problem], indices), .35 * progress)
                if not all(math.isfinite(v) for v in values.values()):
                    raise ValueError('Nonfinite R59 update; no alternative loss or batch retry')
                cursor[problem] += 1
                successful += 1
                counts[problem] += 1
                retries += values['skipped']
                for key in LOSS_FIELDS:
                    totals[key] += values[key] * len(indices)
                totals['samples'] += len(indices)
                if (step + 1) % 180 == 0 or step + 1 == steps:
                    print(f'[R59{arm}] epoch={epoch+1} update={step+1}/{steps} total={successful} '
                          f'loss={totals["loss"]/totals["samples"]:.5f} lr={lr:.3g}', flush=True)
            if totals['samples'] != sum(protocol['sizes'].values()) or dict(cursor) != protocol['updates_per_epoch']:
                raise ValueError('R59 must cover every task/example once, including tails')
            if args.device.startswith('cuda'):
                torch.cuda.synchronize(args.device)
            optimization = {k: totals[k] / totals['samples'] for k in LOSS_FIELDS}
            optimization.update(lr=lr, native_lr=lr * .3, consistency_weight=.35 * progress,
                                amp_skipped=retries, samples=totals['samples'],
                                seconds=time.monotonic() - epoch_start)
            if args.device.startswith('cuda'):
                optimization['peak_memory_gib'] = torch.cuda.max_memory_allocated(args.device) / 2**30
            validation = evaluate(model, loaders['val'], raw['val'], reference['val'], args.eval_batch_size,
                                  directory / 'val_predictions' / f'epoch{epoch+1:03d}')
            cost = validation['macro']['macro_vs_sbs_pct']
            if not math.isfinite(cost):
                raise ValueError('Nonfinite checkpoint-selection cost')
            halve, stop = controller.observe(cost, epoch + 1)
            training = (evaluate(model, loaders['train'], raw['train'], reference['train'], args.eval_batch_size)
                        if (epoch + 1) % 5 == 0 or stop or epoch + 1 == args.epochs else None)
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
            print_eval(f'R59{arm} val epoch {epoch+1}', validation['per_problem'], validation['macro'])
            if wandb_run is not None:
                metrics = dict(epoch=epoch+1, successful_updates=successful,
                               **{'optimization/' + k: v for k, v in optimization.items()})
                for split, value in (('train_eval', training), ('val', validation)):
                    if value is not None:
                        metrics.update({split + '/' + k: v for k, v in value['macro'].items()})
                wandb_run.log(metrics)
            from .r43_analysis import plot_run
            plot_run(directory)
        model.load_state_dict(best_model, strict=True)
        best_eval = dict(best, train=evaluate(model, loaders['train'], raw['train'], reference['train'], args.eval_batch_size),
                        val=evaluate(model, loaders['val'], raw['val'], reference['val'], args.eval_batch_size,
                                     directory / 'val_predictions/best'))
        if best_eval['val']['macro'] != best['val']['macro']:
            raise ValueError('Validation-selected checkpoint did not replay exactly')
        verify_sources(protocol)
        verify_data_files(protocol)
        atomic_json(directory / 'best_eval.json', best_eval)
        last5 = {k: float(np.mean([r['val']['macro'][k] for r in history[-5:]])) for k in history[-1]['val']['macro']}
        result = dict(best=best_eval, final=history[-1], last5_val=last5, epochs_completed=len(history),
                      successful_updates=successful, updates_by_problem=dict(counts),
                      wall_seconds=prior_wall + time.monotonic() - started,
                      stop_reason='validation early stop' if stop else 'maximum epochs',
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
            raise ValueError('Completed R59 arm exists; preserve it or use --resume to skip it')
        return
    timing_path = directory / 'wall_time.json'
    timing = json.loads(timing_path.read_text()) if timing_path.exists() else []
    prior, started, status = sum(r['seconds'] for r in timing), time.monotonic(), 'failed'
    try:
        with (directory / 'train.log').open('a') as stream, redirect_stdout(Tee(sys.stdout, stream)), redirect_stderr(Tee(sys.stderr, stream)):
            train(arm, args, directory, data, started, prior)
        status = 'complete'
    finally:
        timing.append(dict(seconds=time.monotonic() - started, status=status))
        atomic_json(timing_path, timing)
        if status == 'complete':
            result = json.loads((directory / 'result.json').read_text())
            result['wall_seconds'] = sum(r['seconds'] for r in timing)
            atomic_json(directory / 'result.json', result)


def lock_validation_checkpoints(args):
    protocol = json.loads((args.root / 'protocol.json').read_text())
    verify_sources(protocol)
    verify_data_files(protocol)
    records, validation = {}, {}
    for arm, name in RUN_NAMES.items():
        directory = args.root / name
        result = json.loads((directory / 'result.json').read_text())
        path = directory / 'best.pt'
        if file_hash(path) != result['best_checkpoint_sha256']:
            raise ValueError('R59 validation-selected checkpoint changed')
        records[arm] = dict(path=str(path), sha256=file_hash(path), epoch=result['best']['epoch'],
                            val=result['best']['val']['macro'], model_params=protocol['model_params'][arm])
        validation[arm] = result['best']['val']
    path = args.root / 'locked_checkpoints.json'
    if path.exists() and json.loads(path.read_text()) != records:
        raise ValueError('R59 validation checkpoint lock is immutable')
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
        parser.error('R59 uses min15/max40 epochs and a positive evaluation batch size')
    args.root = args.root.resolve()
    args.reference = REFERENCE
    if args.stage in ('prepare', 'train', 'all'):
        if (args.device != 'cuda:0' or os.environ.get('R59_PARENT_GPU_BATON') != 'trainable-solver-encoder'
                or os.environ.get('CUDA_VISIBLE_DEVICES') != GPU_UUID
                or os.environ.get('SLURM_JOB_ID') != '1465'):
            parser.error('Training/preflight require the explicit parent baton and assigned Slurm/UUID launcher')
        configure_torch()
    torch.set_num_threads(1)
    args.root.mkdir(parents=True, exist_ok=True)
    if args.stage == 'plan':
        atomic_json(args.root / 'planned_configs.json', planned_config(args))
        print('[R59 plan only] No model, CUDA, training, validation evaluation, or test access')
    if args.stage in ('prepare', 'all'):
        prepare(args)
    if args.stage in ('train', 'all'):
        data = load_training_data(args)
        for arm in (args.arm,) if args.arm else RUN_NAMES:
            run(arm, args, data)
        del data
        if all((args.root / name / 'result.json').exists() for name in RUN_NAMES.values()):
            lock_validation_checkpoints(args)
    if args.stage in ('analyze', 'all'):
        from .r59_analysis import summarize
        summarize(args.root)


if __name__ == '__main__':
    main()

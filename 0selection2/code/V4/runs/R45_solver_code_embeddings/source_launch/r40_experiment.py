"""R40: paired small-batch dual-stream/direct classification, no test access."""

import argparse
import json
import math
import shutil
import sys
import time
from collections import Counter
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from code.unified_selector.registry import PROBLEMS
from .direct_selector import make_r40_model
from .multitask_probe import TaskRandomStreams, Tee, dump, file_hash, gather_batch, state_hash, update_batch
from .performance_evaluation import POLICY_METRICS, decision_metrics
from .performance_experiment import SOURCE_FILES, fit_args, model_params, prepare_geometry
from .performance_targets import read_raw_costs
from .tensor_loader import TensorBatchLoader
from .train import configure_torch, make_loader, print_eval, set_seed
from .training_monitor import capture_rng


ROOT = Path('code/V4/runs/R40_discriminative_training')
REFERENCE = Path('code/V4/runs/R39_performance_model')
RUN_NAMES = {'A': 'dual_stream_seed2', 'B': 'direct_seed2'}
METRICS = ('ce', *POLICY_METRICS)
R40_SOURCES = ('direct_selector.py', 'r40_experiment.py', 'r40_analysis.py', 'r40_verify.py', 'r40_watch.py', 'test_r40.py', 'run_v4_r40.sh')


def subdivide_schedule(original, batch_size=64):
    batches = {}
    ratios = set()
    for p in PROBLEMS:
        old = original['batches'][p]
        if old.size(-1) % batch_size:
            raise ValueError('R40 batch must exactly divide the reference batch')
        ratios.add(old.size(-1) // batch_size)
        batches[p] = old.reshape(old.size(0), -1, batch_size)
    if len(ratios) != 1:
        raise ValueError('Reference tasks must use the same training batch size')
    ratio = ratios.pop()
    orders = []
    for old in original['orders']:
        order = []
        for start in range(0, len(old), len(PROBLEMS)):
            cycle = old[start:start + len(PROBLEMS)]
            if set(cycle) != set(PROBLEMS) or len(cycle) != len(PROBLEMS):
                raise ValueError('Each reference cycle must update every problem exactly once')
            order.extend(cycle * ratio)
        orders.append(order)
    return dict(batches=batches, orders=orders)


def learning_rate(successful_updates, total_updates, peak=2e-4, minimum=2e-6, warmup_updates=8100):
    step = successful_updates + 1
    if step <= warmup_updates:
        return peak * step / warmup_updates
    progress = (step - warmup_updates) / (total_updates - warmup_updates)
    return minimum + (peak - minimum) * (1 + math.cos(math.pi * progress)) / 2


def prepare_data(args, write=False):
    raw = {split: {p: read_raw_costs(p, split) for p in PROBLEMS} for split in ('train', 'val')}
    manifest = {s: {p: {key: r[key] for key in ('pool', 'pool_ids', 'label_hash', 'data_hash')}
                    for p, r in entries.items()} for s, entries in raw.items()}
    if manifest != json.loads((args.reference / 'data_manifest.json').read_text()):
        raise ValueError('R40 must use the same data and candidate mapping as R39A')
    if write:
        dump(args.root / 'data_manifest.json', manifest)
    elif manifest != json.loads((args.root / 'data_manifest.json').read_text()):
        raise ValueError('Prepared R40 data changed before training')
    loaders = {s: {} for s in raw}
    for split in raw:
        for p in PROBLEMS:
            loader = make_loader(p, split, args.batch_size, 0, shuffle=False, cache_device=args.device)
            labels = raw[split][p]
            np.testing.assert_array_equal(loader.batch['ind'].cpu(), labels['winner'])
            np.testing.assert_array_equal(loader.batch['costs'].cpu(), labels['costs'].astype(np.float32))
            np.testing.assert_array_equal(loader.batch['pool_ids'].cpu(), labels['pool_ids'])
            if loader.size != len(labels['winner']):
                raise ValueError('Instance and raw-label coverage disagree')
            loaders[split][p] = loader
    geometry = prepare_geometry(loaders, raw, args.reference / 'geometry_cache')
    return loaders, raw, geometry


def initialize(group, seed, geometry, device='cuda:0', parameters=None):
    set_seed(seed)
    params = parameters or dict(model_params('A', sdpa=str(device).startswith('cuda')),
                                architecture='dual_stream' if group == 'A' else 'direct')
    model = make_r40_model(params).to(device)
    branch = model.instance_encoder.geometry_residual
    with torch.no_grad():
        branch.mean.copy_(torch.tensor(geometry['mean'], device=device))
        branch.std.copy_(torch.tensor(geometry['std'], device=device))
    return model, params


def aggregate(results):
    return {'macro_' + key: float(np.mean([r[key] for r in results.values()])) for key in METRICS}


def families(results):
    groups = {p: [p] for p in ('TSP', 'CVRP', 'ATSP')}
    groups['MVRP'] = [p for p in PROBLEMS if p not in groups]
    return {name: aggregate({p: results[p] for p in tasks}) for name, tasks in groups.items()}


def classification_metrics(logits, labels, mask=None):
    mask = np.ones_like(logits, dtype=bool) if mask is None else mask
    if logits.shape != labels['costs'].shape or not np.isfinite(logits[mask]).all():
        raise ValueError('Classification logits must match the complete valid candidate pool')
    result, pred = decision_metrics(logits, labels, mask)
    result.update(ce=float(F.cross_entropy(torch.from_numpy(np.where(mask, logits, -np.inf)),
                                          torch.from_numpy(labels['winner']))), pool=labels['pool'])
    result['method_top1'] = dict(zip(labels['pool'], (np.bincount(labels['winner'], minlength=len(labels['pool'])) / len(pred)).tolist()))
    result['method_mean_cost'] = dict(zip(labels['pool'], labels['costs'].mean(0).tolist()))
    fixed_rank = labels['costs'].mean(0).argsort(kind='stable')
    for k in (1, 2, 3):
        result[f'best_top{k}'] = float(np.isin(labels['winner'], fixed_rank[:k]).mean())
    for name, values, value, higher in (('top1', result['method_top1'], result['top1'], True),
                                        ('cost', result['method_mean_cost'], result['mean_cost'], False)):
        result[f'{name}_exceeds_methods'] = [s for s, v in values.items() if (value > v if higher else value < v)]
        result[f'{name}_not_exceeds_methods'] = [s for s, v in values.items() if not (value > v if higher else value < v)]
    return result, pred


@torch.no_grad()
def evaluate(model, loaders, raw, batch_size=256, prediction_dir=None):
    was_training = model.training
    model.eval()
    if prediction_dir is not None:
        prediction_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for p in PROBLEMS:
        outputs = [model(batch)['logits'].float().cpu() for batch in TensorBatchLoader(loaders[p].batch, batch_size, False, False)]
        logits = torch.cat(outputs).numpy()
        result, pred = classification_metrics(logits, raw[p])
        results[p] = result
        if prediction_dir is not None:
            np.savez_compressed(prediction_dir / f'{p}.npz', indices=np.arange(len(pred)), logits=logits,
                                pred=pred, winner=raw[p]['winner'], costs=raw[p]['costs'], pool_ids=np.array(raw[p]['pool_ids']))
    model.train(was_training)
    return dict(per_problem=results, macro=aggregate(results), families=families(results))


def source_snapshot(root):
    directory = root / 'source_launch'
    directory.mkdir(exist_ok=True)
    sources = {}
    for name in set(SOURCE_FILES) | set(R40_SOURCES):
        path = Path(__file__).parent / name
        sources[name] = file_hash(path)
        shutil.copy2(path, directory / name)
    for name in ('data.py', 'registry.py'):
        path = Path(__file__).parents[1] / 'unified_selector' / name
        sources[name] = file_hash(path)
        shutil.copy2(path, directory / name)
    dump(directory / 'hashes.json', sources)
    return sources


def prepare(args):
    if (args.root / 'protocol.json').exists():
        raise ValueError('R40 is already prepared; preserve this experiment root')
    args.root.mkdir(parents=True, exist_ok=True)
    loaders, raw, geometry = prepare_data(args, write=True)
    original_path = args.reference / 'protocol' / 'seed2.pt'
    original = torch.load(original_path, map_location='cpu', weights_only=False)
    schedule = subdivide_schedule(original, args.batch_size)
    if len(schedule['orders']) != args.epochs:
        raise ValueError('R40 epochs must match the reference sample budget')
    for p in PROBLEMS:
        if schedule['batches'][p].shape != (args.epochs, 150, 64):
            raise ValueError('R40 requires 150 genuine batch64 updates per task per epoch')
        for values in schedule['batches'][p].reshape(args.epochs, -1):
            if len(values.unique()) != 9600 or int(values.min()) < 0 or int(values.max()) >= loaders['train'][p].size:
                raise ValueError('Reference sample budget has duplicate or invalid instance indices')
    torch.save(schedule, args.root / 'schedule_seed2.pt')
    source_hashes = source_snapshot(args.root)
    witnesses, encoder_hashes = [], {}
    for group in ('A', 'B'):
        model, params = initialize(group, args.seed, geometry, args.device)
        encoder_hashes[group] = state_hash(model.instance_encoder.state_dict())
        assert all(p.requires_grad for p in model.parameters())
        model.eval()
        for p in PROBLEMS:
            batch = gather_batch(loaders['train'][p], torch.arange(2))
            with torch.no_grad():
                expected = model(batch)['logits']
                clean = {k: v for k, v in batch.items() if k not in ('costs', 'ind', 'performance_target')}
                torch.testing.assert_close(model(clean)['logits'], expected, atol=0, rtol=0)
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
        scaler = torch.amp.GradScaler('cuda')
        model.train()
        for p in ('TSP', 'CVRP', 'ATSP'):
            indices = loaders['train'][p].lengths.argsort(descending=True)[:64]
            batch = gather_batch(loaders['train'][p], indices)
            torch.cuda.reset_peak_memory_stats()
            started = time.monotonic()
            values = update_batch(model, optimizer, scaler, batch, fit_args('A'))
            witnesses.append(dict(group=group, problem=p, batch=64, seconds=time.monotonic() - started,
                                  peak_memory_gib=torch.cuda.max_memory_allocated() / 2**30, **values))
            print('[witness] ' + json.dumps(witnesses[-1]), flush=True)
        del model, optimizer, scaler
        torch.cuda.empty_cache()
    if encoder_hashes['A'] != encoder_hashes['B']:
        raise ValueError('Paired shared encoder initializations differ')
    reference = json.loads((args.reference / 'winner_cost_seed2' / 'result.json').read_text())
    train_reference = json.loads((args.reference / 'winner_cost_seed2' / 'train_eval_epoch060.json').read_text())
    dump(args.root / 'reference_R39A.json', dict(best=reference['best'], final=reference['final'], last5=reference['last5'],
                                               train_final=train_reference, path=str(args.reference), test_read=False))
    dump(args.root / 'runtime_witness.json', dict(witnesses=witnesses, same_encoder_initialization=True,
                                                encoder_hashes=encoder_hashes, label_free_tasks=list(PROBLEMS), test_read=False))
    protocol = dict(seed=2, groups=['A', 'B'], expected_runs=2, epochs=60, batch_size=64, eval_batch_size=args.eval_batch_size,
                    successful_updates_per_problem=9000, global_successful_updates=162000, gradient_accumulation=1,
                    samples_per_problem_per_epoch=9600, samples_per_run=10368000, sample_budget_matches_R39=True,
                    peak_lr=2e-4, minimum_lr=2e-6, warmup_updates=8100, dropout=.1, weight_decay=1e-4,
                    loss=vars(fit_args('A')), initialization='scratch; matched shared encoder',
                    checkpoint_rule='minimum validation macro_vs_sbs_pct', source_hashes=source_hashes,
                    schedule_hash=state_hash(schedule), original_schedule_hash=file_hash(original_path),
                    geometry_stats=geometry, geometry_cache=str(args.reference / 'geometry_cache'),
                    no_test=True, no_augmentation=True, no_performance_head=True,
                    note='The final dual-stream/direct from-scratch plan supersedes the earlier continuation/CE-only plan.')
    dump(args.root / 'protocol.json', protocol)
    print('[prepared] two seed2 runs; 162000 successful updates each; no test access', flush=True)


def run(group, args):
    directory = args.root / RUN_NAMES[group]
    directory.mkdir(exist_ok=True)
    if (directory / 'history.json').exists():
        raise ValueError('A training run already exists; do not silently overwrite or restart it')
    with (directory / 'train.log').open('w') as stream, redirect_stdout(Tee(sys.stdout, stream)), redirect_stderr(Tee(sys.stderr, stream)):
        _run(group, args, directory)


def _run(group, args, directory):
    protocol = json.loads((args.root / 'protocol.json').read_text())
    if (args.seed, args.epochs, args.batch_size, args.eval_batch_size) != (2, 60, 64, protocol['eval_batch_size']):
        raise ValueError('Training arguments differ from the prepared R40 protocol')
    loaders, raw, geometry = prepare_data(args)
    schedule = torch.load(args.root / 'schedule_seed2.pt', map_location='cpu', weights_only=False)
    if state_hash(schedule) != protocol['schedule_hash']:
        raise ValueError('Prepared sample/task sequence changed')
    model, params = initialize(group, args.seed, geometry, args.device)
    encoder_hash = state_hash(model.instance_encoder.state_dict())
    witness = json.loads((args.root / 'runtime_witness.json').read_text())
    if encoder_hash != witness['encoder_hashes'][group]:
        raise ValueError('Launch initialization differs from the runtime witness')
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    scaler = torch.amp.GradScaler('cuda')
    streams = TaskRandomStreams(args.seed, PROBLEMS)
    config = dict(protocol, group=group, model_params=params, decision_head='classification',
                  optimizer_restored=False, weights_restored=False, encoder_initial_hash=encoder_hash,
                  model_initial_hash=state_hash(model.state_dict()), test_read=False, wandb_mode='offline')
    dump(directory / 'args.json', config)
    wandb_run = None
    if args.wandb:
        import wandb
        wandb_run = wandb.init(project='selector', entity='yjkds-southern-university-of-science-technology',
                               name='R40_' + directory.name, config=config, dir=str(directory), mode='offline')
    initial = {s: evaluate(model, loaders[s], raw[s], args.eval_batch_size) for s in ('train', 'val')}
    dump(directory / 'initial_eval.json', initial)
    history, best_record, successful, skipped = [], None, 0, 0
    started = time.monotonic()
    counts = Counter({p: 0 for p in PROBLEMS})
    print(f'[start] {directory.name} scratch=True batch=64 accumulation=1 total_updates=162000', flush=True)

    def checkpoint(epoch, record, name):
        torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), scaler=scaler.state_dict(),
                        args=config, epoch=epoch - 1, successful_updates=successful, updates_by_problem=dict(counts),
                        macro=record['val'], rng=capture_rng(), task_rng=streams.states), directory / name)

    with (directory / 'update_trace.jsonl').open('w') as trace:
        for epoch_index, order in enumerate(schedule['orders']):
            model.train()
            positions, meters = Counter(), []
            epoch_started = time.monotonic()
            for p in order:
                lr = learning_rate(successful, protocol['global_successful_updates'])
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
                indices = schedule['batches'][p][epoch_index, positions[p]]
                with streams.activate(p):
                    values = update_batch(model, optimizer, scaler, gather_batch(loaders['train'][p], indices), fit_args('A'))
                positions[p] += 1
                counts[p] += 1
                successful += 1
                skipped += values['skipped']
                meters.append(values)
                trace.write(json.dumps(dict(step=successful, epoch=epoch_index + 1, problem=p, batch_size=64,
                                            batch_index=positions[p] - 1, batch_hash=state_hash(indices), lr=lr, **values)) + '\n')
                if successful % 300 == 0:
                    stats = {key: float(np.mean([v[key] for v in meters[-300:]])) for key in ('loss', 'ce', 'pair', 'risk', 'grad_norm')}
                    speed = len(meters) * 64 / (time.monotonic() - epoch_started)
                    print(f"ep{epoch_index + 1:03d} step{successful:06d} loss={stats['loss']:.5f} ce={stats['ce']:.5f} "
                          f"grad={stats['grad_norm']:.4f} lr={lr:.3g} samples/s={speed:.0f} "
                          f"elapsed={(time.monotonic() - started)/60:.1f}m", flush=True)
                    if wandb_run:
                        wandb_run.log({**{'train/' + k: v for k, v in stats.items()}, 'train/lr': lr, 'speed/samples_per_second': speed}, step=successful)
            if set(positions) != set(PROBLEMS) or set(positions.values()) != {150}:
                raise ValueError('Every epoch must execute 150 successful updates for each task')
            trace.flush()
            epoch = epoch_index + 1
            payload = evaluate(model, loaders['val'], raw['val'], args.eval_batch_size, directory / 'predictions' / f'val_e{epoch:03d}')
            print_eval(f'eval epoch {epoch} {group}', payload['per_problem'], payload['macro'])
            record = dict(epoch=epoch, updates=successful, updates_by_problem=dict(counts), lr=lr,
                          train={k: float(np.mean([v[k] for v in meters])) for k in meters[0]},
                          val=payload['macro'], per_problem=payload['per_problem'], families=payload['families'],
                          skipped_attempts_total=skipped, elapsed_seconds=time.monotonic() - started)
            if epoch % 5 == 0:
                train_eval = evaluate(model, loaders['train'], raw['train'], args.eval_batch_size,
                                      directory / 'predictions' / 'train_final' if epoch == 60 else None)
                record['train_eval'] = train_eval['macro']
                dump(directory / f'train_eval_epoch{epoch:03d}.json', train_eval)
                print_eval(f'train_eval epoch {epoch} {group}', train_eval['per_problem'], train_eval['macro'])
            if best_record is None or record['val']['macro_vs_sbs_pct'] < best_record['val']['macro_vs_sbs_pct']:
                best_record = record
                checkpoint(epoch, record, 'best.pt')
                print(f"[save] best.pt epoch={epoch} vs_sbs={record['val']['macro_vs_sbs_pct']:+.5f}%", flush=True)
            checkpoint(epoch, record, 'last.pt')
            history.append(record)
            dump(directory / 'history.json', history)
            print(f'[epoch done {epoch}] successful_updates={successful} per_problem={counts["TSP"]} skipped={skipped}', flush=True)
            if wandb_run:
                tracked = {'val/' + k: v for k, v in payload['macro'].items()}
                for p, r in payload['per_problem'].items():
                    tracked.update({f'val/{p}/{k}': r[k] for k in METRICS})
                if 'train_eval' in record:
                    tracked.update({'train_eval/' + k: v for k, v in record['train_eval'].items()})
                wandb_run.log(tracked, step=successful)
            if epoch % 5 == 0:
                from .r40_analysis import plot_run
                plot_run(directory)
    result = dict(group=group, seed=2, epochs=60, updates=successful, updates_by_problem=dict(counts), skipped_attempts=skipped,
                  best=best_record, final=history[-1], elapsed_seconds=time.monotonic() - started,
                  last5={k: float(np.mean([r['val'][k] for r in history[-5:]])) for k in history[-1]['val']},
                  train_final=history[-1]['train_eval'], encoder_initial_hash=encoder_hash, schedule_hash=protocol['schedule_hash'],
                  source_hashes=protocol['source_hashes'], test_read=False)
    dump(directory / 'result.json', result)
    if wandb_run:
        wandb_run.summary.update(dict(best_epoch=best_record['epoch'], final_val_top1=history[-1]['val']['macro_top1']))
        wandb_run.finish()
    print(f'[done] {directory.name} epochs=60 successful_updates={successful} test_read=False', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--reference', type=Path, default=REFERENCE)
    parser.add_argument('--seed', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--eval-batch-size', type=int, default=256)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--group', choices=('A', 'B'))
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--wandb', action='store_true')
    args = parser.parse_args()
    configure_torch()
    torch.set_num_threads(1)
    if args.prepare_only:
        prepare(args)
    else:
        if args.group is None:
            parser.error('--group A/B is required for training')
        run(args.group, args)


if __name__ == '__main__':
    main()

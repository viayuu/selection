"""R42: paired full-coverage training, R-Drop, validation-only selection."""

import argparse
import json
import random
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
from .multitask_probe import Tee, dump, file_hash, gather_batch, state_hash, task_seed
from .performance_experiment import SOURCE_FILES, fit_args, model_params, prepare_geometry
from .performance_targets import read_raw_costs
from .r40_experiment import aggregate, classification_metrics, families
from .relation_graph import EDGE_FIELDS, build_relation_graph
from .solver_query import make_r42_model
from .tensor_loader import TensorBatchLoader
from .train import configure_torch, make_loader, print_eval, selector_loss, set_seed
from .training_monitor import capture_rng, restore_rng


ROOT = Path('code/V4/runs/R42_relation_query')
REFERENCE = Path('code/V4/runs/R39_performance_model')
RUN_NAMES = {'A': 'dual_stream_rdrop_seed2', 'B': 'relation_query_rdrop_seed2'}
NEW_SOURCES = ('relation_graph.py', 'relation_encoder.py', 'solver_query.py', 'r42_experiment.py',
               'r42_analysis.py', 'test_r42.py', 'run_v4_r42.sh')
LABEL_KEYS = ('costs', 'ind', 'performance_target')


def inputs_only(batch):
    return {key: value for key, value in batch.items() if key not in LABEL_KEYS}


def paired_schedule(sizes, epochs=40, batch_size=128, seed=2):
    generators = {p: torch.Generator().manual_seed(task_seed(seed, p)) for p in sizes}
    task_rng = random.Random(seed + 701000003)
    permutations, orders = [], []
    for _ in range(epochs):
        permutations.append({p: torch.randperm(size, generator=generators[p]) for p, size in sizes.items()})
        remaining = {p: (size + batch_size - 1) // batch_size for p, size in sizes.items()}
        order = []
        while any(remaining.values()):
            cycle = [p for p in sizes if remaining[p]]
            task_rng.shuffle(cycle)
            for p in cycle:
                order.append(p)
                remaining[p] -= 1
        orders.append(order)
    return dict(permutations=permutations, orders=orders)


def symmetric_kl(first, second, mask=None):
    if mask is None:
        mask = torch.ones_like(first, dtype=torch.bool)
    log_p = F.log_softmax(first.float().masked_fill(~mask, -1e9), -1)
    log_q = F.log_softmax(second.float().masked_fill(~mask, -1e9), -1)
    return (.5 * (log_p.exp() - log_q.exp()) * (log_p - log_q)).masked_fill(~mask, 0).sum(-1).mean()


def update_batch(model, optimizer, scaler, batch, consistency_weight):
    clean = inputs_only(batch)
    if model.params['architecture'] == 'relation_query':
        clean['relation_graph'] = build_relation_graph(clean)
    rng = capture_rng()
    for retry in range(20):
        restore_rng(rng)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=batch['costs'].device.type, dtype=torch.float16, enabled=scaler.is_enabled()):
            first, second = model(clean), model(clean)
        first = {'logits': first['logits'].float()}
        second = {'logits': second['logits'].float()}
        loss1, parts1 = selector_loss(first, batch['costs'], None, fit_args('A'), winner=batch['ind'])
        loss2, parts2 = selector_loss(second, batch['costs'], None, fit_args('A'), winner=batch['ind'])
        kl = symmetric_kl(first['logits'], second['logits'], batch.get('solver_mask'))
        loss = .5 * (loss1 + loss2) + consistency_weight * kl
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        old_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        if scaler.get_scale() >= old_scale:
            return dict(loss=float(loss.detach()), kl=float(kl.detach()),
                        ce=float((parts1['ce'] + parts2['ce']).detach() / 2),
                        pair=float((parts1['pair'] + parts2['pair']).detach() / 2),
                        risk=float((parts1['risk'] + parts2['risk']).detach() / 2),
                        grad_norm=float(grad_norm), skipped=retry)
        print(f'[AMP retry] same batch, lower scale={scaler.get_scale()}', flush=True)
    raise RuntimeError('Repeated nonfinite gradients on the same R42 batch')


class CostController:
    def __init__(self, min_delta=.001, min_epochs=15, patience=8, lr_patience=3):
        self.min_delta, self.min_epochs = min_delta, min_epochs
        self.patience, self.lr_patience = patience, lr_patience
        self.effective_best = float('inf')
        self.bad_epochs = self.lr_bad_epochs = 0

    def observe(self, cost, epoch):
        improved = cost < self.effective_best - self.min_delta
        if improved:
            self.effective_best, self.bad_epochs, self.lr_bad_epochs = cost, 0, 0
        else:
            self.bad_epochs += 1
            self.lr_bad_epochs += 1
        if epoch <= 3:
            self.lr_bad_epochs = 0
        halve = epoch > 3 and self.lr_bad_epochs >= self.lr_patience
        if halve:
            self.lr_bad_epochs = 0
        return halve, epoch >= self.min_epochs and self.bad_epochs >= self.patience


def prepare_data(args):
    raw = {s: {p: read_raw_costs(p, s) for p in PROBLEMS} for s in ('train', 'val')}
    manifest = {s: {p: {k: r[k] for k in ('pool', 'pool_ids', 'data_hash', 'label_hash')}
                    for p, r in entries.items()} for s, entries in raw.items()}
    if manifest != json.loads((args.reference / 'data_manifest.json').read_text()):
        raise ValueError('R42 must retain the R39 data split and candidate identities')
    path = args.root / 'data_manifest.json'
    if path.exists() and manifest != json.loads(path.read_text()):
        raise ValueError('Data changed after preparation')
    dump(path, manifest)
    loaders = {s: {} for s in raw}
    for split in raw:
        for p in PROBLEMS:
            loader = make_loader(p, split, args.batch_size, 0, shuffle=False, cache_device=args.device)
            np.testing.assert_array_equal(loader.batch['ind'].cpu(), raw[split][p]['winner'])
            np.testing.assert_array_equal(loader.batch['pool_ids'].cpu(), raw[split][p]['pool_ids'])
            np.testing.assert_array_equal(loader.batch['costs'].cpu(), raw[split][p]['costs'].astype(np.float32))
            if loader.size != len(raw[split][p]['winner']):
                raise ValueError('Raw-label and instance coverage disagree')
            loader.drop_last = False
            loaders[split][p] = loader
    geometry = prepare_geometry(loaders, raw, args.reference / 'geometry_cache')
    return loaders, raw, geometry


def initialize(group, geometry, device, seed=2):
    set_seed(seed)
    params = dict(model_params('A', sdpa=str(device).startswith('cuda')),
                  architecture='dual_stream' if group == 'A' else 'relation_query', edge_dim=32)
    model = make_r42_model(params).to(device)
    with torch.no_grad():
        branch = model.instance_encoder.geometry_residual
        branch.mean.copy_(torch.tensor(geometry['mean'], device=device))
        branch.std.copy_(torch.tensor(geometry['std'], device=device))
    return model, params


@torch.no_grad()
def evaluate(model, loaders, raw, batch_size=128, prediction_dir=None):
    was_training = model.training
    model.eval()
    results = {}
    if prediction_dir is not None:
        prediction_dir.mkdir(parents=True, exist_ok=True)
    for p, loader in loaders.items():
        outputs = [model(inputs_only(batch))['logits'].float().cpu()
                   for batch in TensorBatchLoader(loader.batch, batch_size, False, False)]
        logits = torch.cat(outputs).numpy()
        if len(logits) != loader.size:
            raise ValueError('Evaluation dropped instances')
        results[p], pred = classification_metrics(logits, raw[p])
        if prediction_dir is not None:
            np.savez_compressed(prediction_dir / f'{p}.npz', indices=np.arange(len(pred)), logits=logits,
                                pred=pred, winner=raw[p]['winner'], costs=raw[p]['costs'],
                                pool_ids=np.array(raw[p]['pool_ids']), pool=np.array(raw[p]['pool']),
                                nodes=loader.lengths.numpy())
    model.train(was_training)
    return dict(per_problem=results, macro=aggregate(results), families=families(results))


def snapshot(root):
    directory = root / 'source_launch'
    directory.mkdir(exist_ok=True)
    files = [Path(__file__).parent / name for name in set(SOURCE_FILES + NEW_SOURCES)]
    files += [Path(__file__).parents[1] / 'unified_selector' / name for name in ('data.py', 'registry.py')]
    hashes = {str(p): file_hash(p) for p in files}
    if (directory / 'hashes.json').exists():
        if hashes != json.loads((directory / 'hashes.json').read_text()):
            raise ValueError('Launch source changed; preserve this experiment and use a new root')
    else:
        for path in files:
            shutil.copy2(path, directory / path.name)
        dump(directory / 'hashes.json', hashes)
    return hashes


def prepare(args):
    args.root.mkdir(parents=True, exist_ok=True)
    sources = snapshot(args.root)
    loaders, raw, geometry = prepare_data(args)
    sizes = {p: loader.size for p, loader in loaders['train'].items()}
    schedule = paired_schedule(sizes, args.epochs, args.batch_size, args.seed)
    torch.save(schedule, args.root / 'schedule_seed2.pt')
    witness = []
    for group in ('A', 'B'):
        model, params = initialize(group, geometry, args.device)
        initial_hash = state_hash(model.state_dict())
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
        scaler = torch.amp.GradScaler('cuda', enabled=str(args.device).startswith('cuda'))
        model.eval()
        with torch.no_grad():
            for p in PROBLEMS:
                batch = gather_batch(loaders['train'][p], torch.arange(2))
                torch.testing.assert_close(model(batch)['logits'], model(inputs_only(batch))['logits'], atol=0, rtol=0)
        model.train()
        for p in ('TSP', 'CVRP', 'ATSP', 'OVRPTW'):
            batch = gather_batch(loaders['train'][p], loaders['train'][p].lengths.argsort(descending=True)[:args.batch_size])
            torch.cuda.reset_peak_memory_stats()
            started = time.monotonic()
            values = update_batch(model, optimizer, scaler, batch, .35)
            row = dict(group=group, problem=p, batch=len(batch['ind']), max_nodes=int(batch['n'].max()),
                       seconds=time.monotonic() - started, peak_memory_gib=torch.cuda.max_memory_allocated() / 2**30, **values)
            witness.append(row)
            print('[witness] ' + json.dumps(row), flush=True)
        dump(args.root / f'initial_{group}.json', dict(state_hash=initial_hash, params=params,
              trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad)))
        del model, optimizer, scaler
        torch.cuda.empty_cache()
    dump(args.root / 'runtime_witness.json', dict(witnesses=witness, labels_absent_from_forward=True, test_read=False))
    protocol = dict(seed=args.seed, groups=['A', 'B'], max_epochs=args.epochs, batch_size=args.batch_size,
                    eval_batch_size=args.eval_batch_size, full_train_coverage=True, drop_last=False,
                    sizes=sizes, updates_per_epoch={p: (n + args.batch_size - 1) // args.batch_size for p, n in sizes.items()},
                    learning_rate=1e-4, weight_decay=1e-4, dropout=.1, clip_norm=1., warmup_epochs=3,
                    lr_patience=3, lr_factor=.5, min_epochs=15, stop_patience=8, cost_min_delta_pct=.001,
                    rdrop_weight=.35, rdrop_ramp_epochs=3, loss=vars(fit_args('A')),
                    optimizer_updates_per_batch=1, dropout_passes_per_batch=2,
                    checkpoint_rule='strict minimum validation macro_vs_sbs_pct; early-stop min_delta does not restrict checkpoint saving',
                    initialization='scratch; seed2; all learnable parameters trainable', schedule_hash=state_hash(schedule),
                    geometry_stats=geometry, source_hashes=sources, no_augmentation=True,
                    cost_dtype='original raw_label FP64 for evaluation; historical FP32 winner-cost training',
                    edge_fields=list(EDGE_FIELDS), travel_time='Euclidean distance / speed=1, verified original TW environments',
                    route_limit='travel distance; exclude depot return for open routes; excludes service duration',
                    neighbor_ties='include all kth-neighbor ties; merge repeated edges', test_read=False)
    dump(args.root / 'protocol.json', protocol)
    print('[prepared] full coverage, seed2, A/B, 2 forwards -> 1 update; no test access', flush=True)


def run(group, args):
    directory = args.root / RUN_NAMES[group]
    directory.mkdir(exist_ok=True)
    if (directory / 'result.json').exists() or (directory / 'last.pt').exists():
        raise ValueError('Do not overwrite an existing R42 run')
    with (directory / 'train.log').open('w') as stream, redirect_stdout(Tee(sys.stdout, stream)), redirect_stderr(Tee(sys.stderr, stream)):
        _run(group, args, directory)


def _run(group, args, directory):
    protocol = json.loads((args.root / 'protocol.json').read_text())
    if (args.seed, args.epochs, args.batch_size, args.eval_batch_size) != (protocol['seed'], protocol['max_epochs'], protocol['batch_size'], protocol['eval_batch_size']):
        raise ValueError('Runtime arguments differ from the locked protocol')
    loaders, raw, geometry = prepare_data(args)
    schedule = torch.load(args.root / 'schedule_seed2.pt', map_location='cpu', weights_only=False)
    if state_hash(schedule) != protocol['schedule_hash']:
        raise ValueError('Paired schedule changed')
    model, params = initialize(group, geometry, args.device, args.seed)
    if state_hash(model.state_dict()) != json.loads((args.root / f'initial_{group}.json').read_text())['state_hash']:
        raise ValueError('Training initialization differs from the preflight')
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scaler = torch.amp.GradScaler('cuda', enabled=str(args.device).startswith('cuda'))
    config = dict(protocol, group=group, model_params=params, wandb_mode='offline')
    dump(directory / 'args.json', config)
    wandb_run = None
    if args.wandb:
        import wandb
        wandb_run = wandb.init(project='selector', entity='yjkds-southern-university-of-science-technology',
                               name='R42_' + directory.name, config=config, dir=str(directory), mode='offline')
    initial = {s: evaluate(model, loaders[s], raw[s], args.eval_batch_size) for s in ('train', 'val')}
    dump(directory / 'initial_eval.json', initial)
    print_eval('initial val', initial['val']['per_problem'], initial['val']['macro'])
    controller = CostController()
    history, best, successful, amp_skipped = [], None, 0, 0
    counts = Counter({p: 0 for p in PROBLEMS})
    peak_lr = 1e-4
    updates_per_epoch = len(schedule['orders'][0])
    for epoch in range(args.epochs):
        model.train()
        started, weighted, cursor = time.monotonic(), Counter(), Counter()
        for step, p in enumerate(schedule['orders'][epoch]):
            offset = cursor[p] * args.batch_size
            idx = schedule['permutations'][epoch][p][offset:offset + args.batch_size]
            cursor[p] += 1
            batch = gather_batch(loaders['train'][p], idx)
            progress = min(1., (successful + 1) / (3 * updates_per_epoch))
            lr = 1e-4 * progress if epoch < 3 else peak_lr
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            values = update_batch(model, optimizer, scaler, batch, .35 * progress)
            successful += 1
            counts[p] += 1
            amp_skipped += values['skipped']
            for key in ('loss', 'kl', 'ce', 'pair', 'risk'):
                weighted[key] += values[key] * len(idx)
            weighted['samples'] += len(idx)
            if (step + 1) % 180 == 0 or step + 1 == updates_per_epoch:
                print(f'[train epoch {epoch+1}] update={step+1}/{updates_per_epoch} global={successful} '
                      f'loss={weighted["loss"]/weighted["samples"]:.5f} KL={weighted["kl"]/weighted["samples"]:.5f} '
                      f'lr={lr:.2g} samples/s={weighted["samples"]/(time.monotonic()-started):.0f}', flush=True)
        if any(cursor[p] != protocol['updates_per_epoch'][p] for p in PROBLEMS):
            raise ValueError('Task update coverage differs from the protocol')
        optimization = {k: weighted[k] / weighted['samples'] for k in ('loss', 'kl', 'ce', 'pair', 'risk')}
        optimization.update(seconds=time.monotonic() - started, samples=int(weighted['samples']), lr=lr,
                            consistency_weight=.35 * progress, amp_skipped=amp_skipped)
        validation = evaluate(model, loaders['val'], raw['val'], args.eval_batch_size, directory / 'val_predictions' / f'epoch{epoch+1:03d}')
        training = evaluate(model, loaders['train'], raw['train'], args.eval_batch_size)
        print_eval(f'eval epoch {epoch+1}', validation['per_problem'], validation['macro'])
        print(f'[train_eval epoch {epoch+1}] top1={training["macro"]["macro_top1"]:.5f} '
              f'CE={training["macro"]["macro_ce"]:.5f} actual_regret={training["macro"]["macro_actual_regret_pct"]:.5f}%', flush=True)
        row = dict(epoch=epoch+1, successful_updates=successful, updates_by_problem=dict(counts),
                   optimization=optimization, train=training, val=validation)
        history.append(row)
        cost = validation['macro']['macro_vs_sbs_pct']
        halve, stop = controller.observe(cost, epoch+1)
        if halve:
            peak_lr *= .5
            print(f'[schedule] validation plateau: next lr={peak_lr:.2g}', flush=True)
        checkpoint = dict(model=model.state_dict(), optimizer=optimizer.state_dict(), scaler=scaler.state_dict(),
                          args=config, epoch=epoch, macro=validation['macro'], updates_by_problem=dict(counts),
                          successful_updates=successful, rng=capture_rng(), controller=vars(controller), peak_lr=peak_lr)
        torch.save(checkpoint, directory / 'last.pt')
        if best is None or cost < best['val']['macro']['macro_vs_sbs_pct']:
            best = row
            torch.save(checkpoint, directory / 'best.pt')
            dump(directory / 'best_eval.json', row)
            print(f'[best] epoch={epoch+1} vs_sbs={cost:+.6f}% top1={validation["macro"]["macro_top1"]:.5f}', flush=True)
        dump(directory / 'history.json', history)
        if wandb_run is not None:
            metrics = {'epoch': epoch+1, 'successful_updates': successful, **{'optimization/' + k: v for k, v in optimization.items()}}
            for s, value in (('train_eval', training), ('val', validation)):
                metrics.update({s + '/' + k: v for k, v in value['macro'].items()})
                for family, scores in value['families'].items():
                    metrics.update({f'{s}/{family}/{k}': v for k, v in scores.items()})
            metrics['gpu/peak_memory_gib'] = torch.cuda.max_memory_allocated() / 2**30
            wandb_run.log(metrics)
        # Analysis is deterministic and does not change any training setting.
        from .r42_analysis import plot_run
        plot_run(directory)
        if stop:
            print(f'[early stop] epoch={epoch+1}: {controller.bad_epochs} epochs without effective cost improvement', flush=True)
            break
    last5 = {s: {k: float(np.mean([r[s]['macro'][k] for r in history[-5:]])) for k in history[-1][s]['macro']}
             for s in ('train', 'val')}
    dump(directory / 'result.json', dict(best=best, final=history[-1], last5=last5, epochs_completed=len(history),
                                       successful_updates=successful, updates_by_problem=dict(counts),
                                       stop_reason='validation early stop' if stop else 'maximum epochs', test_read=False))
    if wandb_run is not None:
        wandb_run.finish()
    print(f'[complete {group}] epochs={len(history)} best_epoch={best["epoch"]} test_read=False', flush=True)


def lock_and_test(args):
    root = args.root
    path = root / 'locked_checkpoints.json'
    records = {}
    for group, name in RUN_NAMES.items():
        directory = root / name
        if not (directory / 'result.json').exists():
            raise ValueError('Both validation-selected runs must finish before test')
        ckpt_path = directory / 'best.pt'
        checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        records[group] = dict(path=str(ckpt_path), sha256=file_hash(ckpt_path), epoch=checkpoint['epoch']+1,
                              val=checkpoint['macro'], model_params=checkpoint['args']['model_params'])
    if path.exists() and records != json.loads(path.read_text()):
        raise ValueError('Locked best checkpoints changed')
    dump(path, records)
    if (root / 'test_results.json').exists():
        raise ValueError('Test already completed: do not reselect or silently repeat it')
    from .r41_frozen_evaluation import attach_geometry, checked_loader
    loaders, raw, manifest = {}, {}, {}
    for p in PROBLEMS:
        loaders[p], raw[p] = checked_loader(p, 'test', args.eval_batch_size, args.device)
        attach_geometry(loaders[p], args.device)
        manifest[p] = {k: raw[p][k] for k in ('pool', 'pool_ids', 'data_hash', 'label_hash')}
    results = {}
    for group, record in records.items():
        model = make_r42_model(record['model_params']).to(args.device)
        model.load_state_dict(torch.load(record['path'], map_location='cpu', weights_only=False)['model'], strict=True)
        results[group] = evaluate(model, loaders, raw, args.eval_batch_size, root / 'test_predictions' / group)
        print_eval(f'test R42{group} epoch {record["epoch"]}', results[group]['per_problem'], results[group]['macro'])
        del model
        torch.cuda.empty_cache()
    dump(root / 'test_data_manifest.json', manifest)
    dump(root / 'test_results.json', results)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=('prepare', 'train', 'test'), required=True)
    parser.add_argument('--group', choices=('A', 'B'))
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--reference', type=Path, default=REFERENCE)
    parser.add_argument('--seed', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--eval-batch-size', type=int, default=128)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--wandb', action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    configure_torch()
    torch.set_num_threads(4)
    if args.stage == 'prepare':
        prepare(args)
    elif args.stage == 'train':
        if args.group is None:
            parser.error('--group is required for training')
        run(args.group, args)
    else:
        lock_and_test(args)


if __name__ == '__main__':
    main()

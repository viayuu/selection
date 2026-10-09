"""R43 paired full-pool comparison training; validation selects the final policies."""

import argparse
import json
import shutil
import sys
import time
from collections import Counter
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import numpy as np
import torch

from code.unified_selector.registry import PROBLEMS
from .dual_stream import DualStreamSelector
from .multitask_probe import Tee, dump, file_hash, gather_batch, state_hash
from .pairwise_objective import comparison_targets, inputs_only, predicted_top3, single_objective
from .pairwise_selector import make_r43_model
from .performance_experiment import SOURCE_FILES, model_params
from .r40_experiment import aggregate, classification_metrics, families
from .r42_experiment import CostController, paired_schedule, prepare_data, symmetric_kl
from .tensor_loader import TensorBatchLoader
from .train import configure_torch, print_eval, set_seed
from .training_monitor import capture_rng, restore_rng


ROOT = Path('code/V4/runs/R43_pairwise_selection')
R42 = Path('code/V4/runs/R42_relation_query')
RUN_NAMES = {'A': 'score_difference_seed2', 'B': 'explicit_pair_seed2'}
DIAGNOSTICS = ('true3_accuracy', 'reference_top2_accuracy')
NEW_SOURCES = ('pairwise_selector.py', 'pairwise_objective.py', 'r43_experiment.py',
               'r43_analysis.py', 'test_r43.py', 'run_v4_r43.sh')


def initialize(group, geometry, device, seed=2):
    set_seed(seed)
    params = dict(model_params('A', sdpa=str(device).startswith('cuda')), architecture='pairwise',
                  pair_mode='score_difference' if group == 'A' else 'explicit')
    model = make_r43_model(params).to(device)
    with torch.no_grad():
        branch = model.instance_encoder.geometry_residual
        branch.mean.copy_(torch.tensor(geometry['mean'], device=device))
        branch.std.copy_(torch.tensor(geometry['std'], device=device))
    return model, params


def attach_targets(loaders, raw):
    for p, loader in loaders.items():
        device = loader.batch['costs'].device
        loader.batch.update({k: v.to(device) for k, v in comparison_targets(raw[p]['costs']).items()})


def update_batch(model, optimizer, scaler, batch, consistency_weight):
    clean, rng = inputs_only(batch), capture_rng()
    for retry in range(20):
        restore_rng(rng)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=batch['ind'].device.type, dtype=torch.float16, enabled=scaler.is_enabled()):
            first, second = model(clean), model(clean)
        focus = predicted_top3(first['logits'], second['logits'], batch['pool_ids'], first['solver_mask'])
        loss1, parts1 = single_objective(first, batch, focus)
        loss2, parts2 = single_objective(second, batch, focus)
        kl = symmetric_kl(first['logits'], second['logits'], first['solver_mask'])
        loss = .5 * (loss1 + loss2) + consistency_weight * kl
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        old_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        if scaler.get_scale() >= old_scale:
            return dict(loss=float(loss.detach()), kl=float(kl.detach()),
                        **{k: float((parts1[k] + parts2[k]).detach() / 2) for k in parts1},
                        grad_norm=float(grad_norm), skipped=retry)
        print(f'[AMP retry] same batch; lower scale={scaler.get_scale()}', flush=True)
    raise RuntimeError('Repeated nonfinite gradients on the same R43 batch')


def policy_metrics(scores, raw):
    # Stable ranking in ascending global identity order also defines exact score ties.
    order = np.argsort(raw['pool_ids'], kind='stable')
    labels = dict(raw, costs=raw['costs'][:, order], winner=np.argsort(order)[raw['winner']],
                  pool=[raw['pool'][j] for j in order], pool_ids=[raw['pool_ids'][j] for j in order])
    result, pred = classification_metrics(scores[:, order], labels)
    result['pool'] = raw['pool']
    result['pick_dist'] = np.asarray(result['pick_dist'])[np.argsort(order)].tolist()
    return result, order[pred]


def strong_comparisons(margins, raw, reference_top2):
    i, j = np.triu_indices(margins.shape[-1], 1)
    targets = comparison_targets(raw['costs'])
    valid = targets['cmp_valid'].numpy() & targets['cmp_true3'].numpy()
    correct = margins[:, i, j] * (raw['costs'][:, j] - raw['costs'][:, i]) > 0
    counts = valid.sum(-1)
    per_instance = (correct & valid).sum(-1) / np.maximum(counts, 1)
    rows = np.arange(len(margins))
    a, b = reference_top2.T
    difference = raw['costs'][rows, b] - raw['costs'][rows, a]
    reference_valid = difference != 0
    reference_correct = margins[rows, a, b] * difference > 0
    return dict(true3_accuracy=float(per_instance[counts > 0].mean()) if (counts > 0).any() else 0.,
                true3_pairs=int(counts.sum()), true3_instances=int((counts > 0).sum()),
                true3_pair_accuracy=float(correct[valid].mean()) if valid.any() else 0.,
                reference_top2_accuracy=float(reference_correct[reference_valid].mean()) if reference_valid.any() else 0.,
                reference_top2_pairs=int(reference_valid.sum()),
                reference_top2_zero_margins=int(((margins[rows, a, b] == 0) & reference_valid).sum()))


def add_diagnostics(value):
    per = value['per_problem']
    value['macro'].update({'macro_' + key: float(np.mean([r[key] for r in per.values()])) for key in DIAGNOSTICS})
    for family, scores in value['families'].items():
        tasks = [family] if family != 'MVRP' else [p for p in per if p not in ('TSP', 'CVRP', 'ATSP')]
        scores.update({'macro_' + key: float(np.mean([per[p][key] for p in tasks])) for key in DIAGNOSTICS})
    return value


@torch.no_grad()
def evaluate(model, loaders, raw, reference, batch_size=128, prediction_dir=None):
    was_training = model.training
    model.eval()
    results = {}
    if prediction_dir is not None:
        prediction_dir.mkdir(parents=True, exist_ok=True)
    try:
        for p, loader in loaders.items():
            scores, margins = [], []
            for batch in TensorBatchLoader(loader.batch, batch_size, False, False):
                output = model(inputs_only(batch))
                scores.append(output['logits'].float().cpu())
                margins.append(output['pair_margin'].float().cpu())
            scores, margins = torch.cat(scores).numpy(), torch.cat(margins).numpy()
            if len(scores) != loader.size:
                raise ValueError('Evaluation must retain the final partial batch')
            result, pred = policy_metrics(scores, raw[p])
            result.update(strong_comparisons(margins, raw[p], reference[p]))
            results[p] = result
            if prediction_dir is not None:
                np.savez_compressed(prediction_dir / f'{p}.npz', indices=np.arange(len(pred)), logits=scores,
                                    pair_margin=margins, pred=pred, winner=raw[p]['winner'], costs=raw[p]['costs'],
                                    pool_ids=np.array(raw[p]['pool_ids']), pool=np.array(raw[p]['pool']),
                                    reference_top2=reference[p], nodes=loader.lengths.numpy())
    finally:
        model.train(was_training)
    return add_diagnostics(dict(per_problem=results, macro=aggregate(results), families=families(results)))


def reference_nominations(args, split, loaders, raw):
    checkpoint_path = R42 / 'dual_stream_rdrop_seed2' / 'best.pt'
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    directory = args.root / 'reference_predictions' / split
    directory.mkdir(parents=True, exist_ok=True)
    source = R42 / ('test_predictions/A' if split == 'test' else
                    f'dual_stream_rdrop_seed2/val_predictions/epoch{checkpoint["epoch"] + 1:03d}')
    model, nominations = None, {}
    for p, loader in loaders.items():
        cache = directory / f'{p}.npz'
        existing = cache if cache.exists() else source / f'{p}.npz'
        if split == 'train' and not cache.exists():
            existing = cache
        if existing.exists():
            with np.load(existing) as saved:
                np.testing.assert_array_equal(saved['pool_ids'], raw[p]['pool_ids'])
                np.testing.assert_array_equal(saved['winner'], raw[p]['winner'])
                np.testing.assert_array_equal(saved['costs'], raw[p]['costs'])
                scores = saved['logits'].copy()
        else:
            if model is None:
                model = DualStreamSelector(**checkpoint['args']['model_params']).to(args.device).eval()
                model.load_state_dict(checkpoint['model'], strict=True)
            with torch.no_grad():
                scores = torch.cat([model(inputs_only(batch))['logits'].float().cpu()
                                    for batch in TensorBatchLoader(loader.batch, args.eval_batch_size, False, False)]).numpy()
        order = np.argsort(raw[p]['pool_ids'], kind='stable')
        rank = order[np.argsort(-scores[:, order], axis=-1, kind='stable')]
        nominations[p] = rank[:, :2]
        if not cache.exists():
            np.savez_compressed(cache, logits=scores, pred=rank[:, 0], top2=rank[:, :2],
                                winner=raw[p]['winner'], costs=raw[p]['costs'],
                                pool_ids=np.array(raw[p]['pool_ids']), indices=np.arange(len(scores)))
    return nominations


def prepare(args):
    args.root.mkdir(parents=True, exist_ok=True)
    snapshot = args.root / 'source_launch'
    snapshot.mkdir(exist_ok=True)
    hashes = {}
    for name in set(SOURCE_FILES + NEW_SOURCES + ('r42_experiment.py', 'r40_experiment.py', 'training_monitor.py')):
        source = Path(__file__).parent / name
        hashes[name] = file_hash(source)
        if not (snapshot / name).exists():
            shutil.copy2(source, snapshot / name)
    dump(snapshot / 'hashes.json', hashes)
    loaders, raw, geometry = prepare_data(args)
    for split in loaders:
        attach_targets(loaders[split], raw[split])
    sizes = {p: loader.size for p, loader in loaders['train'].items()}
    schedule = paired_schedule(sizes, args.epochs, args.batch_size, args.seed)
    torch.save(schedule, args.root / 'schedule_seed2.pt')
    common, witness = [], []
    for group in ('A', 'B'):
        model, params = initialize(group, geometry, args.device, args.seed)
        shared = {k: v for k, v in model.state_dict().items()
                  if k.startswith(('instance_encoder.', 'solver_encoder.', 'joint_layers.', 'pool.'))}
        common.append(state_hash(shared))
        dump(args.root / f'initial_{group}.json', dict(shared_hash=common[-1], model_hash=state_hash(model.state_dict()),
             parameters=sum(p.numel() for p in model.parameters()), model_params=params))
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
        scaler = torch.amp.GradScaler('cuda', enabled=str(args.device).startswith('cuda'))
        batch = gather_batch(loaders['train']['CVRP'], loaders['train']['CVRP'].lengths.argsort(descending=True)[:args.batch_size])
        started = time.monotonic()
        values = update_batch(model, optimizer, scaler, batch, .35)
        row = dict(group=group, batch=len(batch['ind']), max_nodes=int(batch['n'].max()),
                   seconds=time.monotonic()-started, **values)
        witness.append(row)
        print('[preflight] ' + json.dumps(row), flush=True)
        del model, optimizer, scaler
    if common[0] != common[1]:
        raise ValueError('A/B shared backbone initialization must be identical')
    dump(args.root / 'runtime_witness.json', witness)
    reference = {s: reference_nominations(args, s, loaders[s], raw[s]) for s in loaders}
    protocol = dict(seed=args.seed, batch_size=args.batch_size, max_epochs=args.epochs, eval_batch_size=args.eval_batch_size,
                    initialization='scratch, identical common parameters, all trainable', learning_rate=1e-4,
                    weight_decay=1e-4, dropout=.1, clip_norm=1., warmup_epochs=3,
                    rdrop_weight=.35, rdrop_ramp_epochs=3, lr_patience=3, min_epochs=15, stop_patience=8,
                    cost_min_delta_pct=.001, sizes=sizes, full_coverage=True, drop_last=False,
                    updates_per_epoch={p: (n + args.batch_size - 1) // args.batch_size for p, n in sizes.items()},
                    optimizer_updates_per_batch=1, dropout_passes=2, no_augmentation=True, natural_sampling=True,
                    loss=dict(ce=.35, comparison=.35, risk=.02, cost_scale=.01,
                              focus_weights=dict(true3=.5, predicted3=.25, all_pairs=.25)),
                    cost_dtype='original FP64 comparisons and regret; FP32 supervision and KL',
                    pair_zero_margin='incorrect in strict sign accuracy; exact-cost ties excluded',
                    true3_accuracy='per-instance mean over unequal true-top3 pairs, then mean over nonempty instances',
                    decision='maximin excluding diagonal/invalid opponents; ties by ascending global solver ID',
                    checkpoint_rule='strict minimum validation macro_vs_sbs_pct',
                    schedule_hash=state_hash(schedule), shared_initial_hash=common[0], geometry_stats=geometry,
                    reference_checkpoint=dict(path=str(R42 / 'dual_stream_rdrop_seed2/best.pt'),
                                              sha256=file_hash(R42 / 'dual_stream_rdrop_seed2/best.pt')),
                    source_hashes=hashes, test_read=False)
    dump(args.root / 'protocol.json', protocol)
    return loaders, raw, geometry, schedule, reference


def run(group, args, data):
    directory = args.root / RUN_NAMES[group]
    directory.mkdir(exist_ok=True)
    if (directory / 'last.pt').exists() or (directory / 'result.json').exists():
        raise ValueError('Do not overwrite an existing completed or partial R43 run')
    with (directory / 'train.log').open('w') as stream, redirect_stdout(Tee(sys.stdout, stream)), redirect_stderr(Tee(sys.stderr, stream)):
        _run(group, args, directory, data)


def _run(group, args, directory, data):
    loaders, raw, geometry, schedule, reference = data
    protocol = json.loads((args.root / 'protocol.json').read_text())
    model, params = initialize(group, geometry, args.device, args.seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scaler = torch.amp.GradScaler('cuda', enabled=str(args.device).startswith('cuda'))
    config = dict(protocol, group=group, model_params=params, wandb_mode='offline')
    dump(directory / 'args.json', config)
    wandb_run = None
    if args.wandb:
        import wandb
        wandb_run = wandb.init(project='selector', entity='yjkds-southern-university-of-science-technology',
                               name='R43_' + directory.name, config=config, dir=str(directory), mode='offline')
    initial = {s: evaluate(model, loaders[s], raw[s], reference[s], args.eval_batch_size) for s in ('train', 'val')}
    dump(directory / 'initial_eval.json', initial)
    print_eval('initial val', initial['val']['per_problem'], initial['val']['macro'])
    controller = CostController()
    history, best, successful, amp_skipped = [], None, 0, 0
    counts, peak_lr = Counter({p: 0 for p in PROBLEMS}), 1e-4
    updates_per_epoch = len(schedule['orders'][0])
    stop = False
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
            for key in ('loss', 'ce', 'cmp', 'risk', 'kl', 'true3', 'pred3', 'all_pairs'):
                weighted[key] += values[key] * len(idx)
            weighted['samples'] += len(idx)
            if (step + 1) % 180 == 0 or step + 1 == updates_per_epoch:
                print(f'[train epoch {epoch+1}] update={step+1}/{updates_per_epoch} global={successful} '
                      f'loss={weighted["loss"]/weighted["samples"]:.5f} cmp={weighted["cmp"]/weighted["samples"]:.5f} '
                      f'KL={weighted["kl"]/weighted["samples"]:.5f} lr={lr:.2g} '
                      f'samples/s={weighted["samples"]/(time.monotonic()-started):.0f}', flush=True)
        if int(weighted['samples']) != sum(protocol['sizes'].values()) or any(cursor[p] != protocol['updates_per_epoch'][p] for p in PROBLEMS):
            raise ValueError('Each epoch must cover every task and every training instance exactly once')
        optimization = {k: weighted[k] / weighted['samples'] for k in ('loss', 'ce', 'cmp', 'risk', 'kl', 'true3', 'pred3', 'all_pairs')}
        optimization.update(seconds=time.monotonic() - started, samples=int(weighted['samples']), lr=lr,
                            consistency_weight=.35 * progress, amp_skipped=amp_skipped)
        validation = evaluate(model, loaders['val'], raw['val'], reference['val'], args.eval_batch_size,
                              directory / 'val_predictions' / f'epoch{epoch+1:03d}')
        cost = validation['macro']['macro_vs_sbs_pct']
        halve, stop = controller.observe(cost, epoch+1)
        training = None
        if (epoch + 1) % 5 == 0 or stop or epoch + 1 == args.epochs:
            training = evaluate(model, loaders['train'], raw['train'], reference['train'], args.eval_batch_size)
            print(f'[train_eval epoch {epoch+1}] top1={training["macro"]["macro_top1"]:.5f} '
                  f'CE={training["macro"]["macro_ce"]:.5f} actual_regret={training["macro"]["macro_actual_regret_pct"]:.5f}%', flush=True)
        print_eval(f'eval epoch {epoch+1}', validation['per_problem'], validation['macro'])
        print(f'[comparisons epoch {epoch+1}] true3={validation["macro"]["macro_true3_accuracy"]:.5f} '
              f'R42A_top2={validation["macro"]["macro_reference_top2_accuracy"]:.5f}', flush=True)
        row = dict(epoch=epoch+1, successful_updates=successful, updates_by_problem=dict(counts),
                   optimization=optimization, train=training, val=validation)
        history.append(row)
        if halve:
            peak_lr *= .5
            print(f'[schedule] validation cost plateau: next lr={peak_lr:.2g}', flush=True)
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
                if value is not None:
                    metrics.update({s + '/' + k: v for k, v in value['macro'].items()})
            metrics['gpu/peak_memory_gib'] = torch.cuda.max_memory_allocated() / 2**30
            wandb_run.log(metrics)
        from .r43_analysis import plot_run
        plot_run(directory)
        if stop:
            print(f'[early stop] epoch={epoch+1}: {controller.bad_epochs} rounds without effective cost improvement', flush=True)
            break
    last5 = {}
    for split in ('train', 'val'):
        points = [r[split]['macro'] for r in history if r[split] is not None][-5:]
        last5[split] = {k: float(np.mean([r[k] for r in points])) for k in points[-1]}
    dump(directory / 'result.json', dict(best=best, final=history[-1], last5=last5, epochs_completed=len(history),
         successful_updates=successful, updates_by_problem=dict(counts),
         stop_reason='validation early stop' if stop else 'maximum epochs', test_read=False))
    if wandb_run is not None:
        wandb_run.finish()
    print(f'[complete R43{group}] epochs={len(history)} best_epoch={best["epoch"]}', flush=True)


def lock_and_test(args):
    from .r41_frozen_evaluation import attach_geometry, checked_loader
    path = args.root / 'locked_checkpoints.json'
    if (args.root / 'test_results.json').exists():
        raise ValueError('Test already completed; do not reselect checkpoints from test')
    records = {}
    for group, name in RUN_NAMES.items():
        directory = args.root / name
        if not (directory / 'result.json').exists():
            raise ValueError('Both runs must finish validation selection before test')
        checkpoint_path = directory / 'best.pt'
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        records[group] = dict(path=str(checkpoint_path), sha256=file_hash(checkpoint_path),
                              epoch=checkpoint['epoch']+1, val=checkpoint['macro'], model_params=checkpoint['args']['model_params'])
    if path.exists() and json.loads(path.read_text()) != records:
        raise ValueError('Locked checkpoints changed')
    dump(path, records)
    loaders, raw = {}, {}
    for p in PROBLEMS:
        loaders[p], raw[p] = checked_loader(p, 'test', args.eval_batch_size, args.device)
        attach_geometry(loaders[p], args.device)
    reference = reference_nominations(args, 'test', loaders, raw)
    results = {}
    for group, record in records.items():
        model = make_r43_model(record['model_params']).to(args.device)
        model.load_state_dict(torch.load(record['path'], map_location='cpu', weights_only=False)['model'], strict=True)
        results[group] = evaluate(model, loaders, raw, reference, args.eval_batch_size, args.root / 'test_predictions' / group)
        print_eval(f'test R43{group} epoch {record["epoch"]}', results[group]['per_problem'], results[group]['macro'])
        del model
    dump(args.root / 'test_data_manifest.json', {p: {k: r[k] for k in ('pool', 'pool_ids', 'data_hash', 'label_hash')} for p, r in raw.items()})
    dump(args.root / 'test_results.json', results)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=('all', 'train', 'test', 'analysis'), default='all')
    parser.add_argument('--group', choices=('A', 'B'))
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--reference', type=Path, default=Path('code/V4/runs/R39_performance_model'))
    parser.add_argument('--seed', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--eval-batch-size', type=int, default=128)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--wandb', action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    configure_torch()
    torch.set_num_threads(4)
    if args.stage in ('all', 'train'):
        data = prepare(args)
        for group in ('A', 'B') if args.stage == 'all' else (args.group,):
            if group is None:
                parser.error('--group is required for --stage train')
            run(group, args, data)
        del data
    if args.stage in ('all', 'test'):
        lock_and_test(args)
    if args.stage in ('all', 'analysis'):
        from .r43_analysis import summarize
        summarize(args.root)


if __name__ == '__main__':
    main()

"""R47B only: from-scratch training, refreshed train memory, locked validation/test."""

import argparse
import json
import shutil
import subprocess
import sys
import time
from collections import Counter
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import numpy as np
import torch

from code.unified_selector.registry import GLOBAL_SOLVERS, PROBLEMS
from .behavior_memory import (BehaviorBank, attach_query_ids, refresh_memory,
                              restore_behavior, shuffled_behavior)
from .multitask_probe import Tee, dump, file_hash, gather_batch, state_hash
from .pairwise_objective import inputs_only
from .r40_experiment import aggregate, families
from .r42_experiment import CostController, paired_schedule, prepare_data
from .r43_experiment import (add_diagnostics, attach_targets, policy_metrics,
                            reference_nominations, strong_comparisons, update_batch)
from .r45_experiment import R43_ARGS, parameters
from .r47_model import BehaviorMemorySelector, load_r47_checkpoint
from .tensor_loader import TensorBatchLoader
from .train import configure_torch, print_eval, set_seed
from .training_monitor import capture_rng


ROOT = Path('code/V4/runs/R47_behavior_memory')
R45_ROOT = Path('code/V4/runs/R45_solver_code_embeddings')
RUN_NAME = 'B_behavior_memory_seed2'
SOURCES = ('behavior_memory.py', 'r47_model.py', 'r47_experiment.py', 'r47_analysis.py',
           'test_r47.py', 'run_v4_r47.sh', 'dual_stream.py', 'V4Model.py', 'solver_features.py',
           'pairwise_selector.py', 'pairwise_objective.py', 'r43_experiment.py',
           'r42_experiment.py', 'r45_experiment.py', 'tensor_loader.py', 'multitask_probe.py',
           'performance_targets.py', 'local_geometry.py')


def initialize(geometry, loaders, raw, device, seed=2):
    set_seed(seed)
    params = dict(parameters('A', device.startswith('cuda')), architecture='behavior_memory',
                  neighbors=32, retrieval_init_seed=470002)
    model = BehaviorMemorySelector(**params).to(device)
    with torch.no_grad():
        branch = model.instance_encoder.geometry_residual
        branch.mean.copy_(torch.tensor(geometry['mean'], device=device))
        branch.std.copy_(torch.tensor(geometry['std'], device=device))
    for p, loader in loaders.items():
        model.memory[p] = BehaviorBank.from_training(loader, raw[p], 2*params['embedding_dim']+1)
    return model, params


def references(args, split, loaders, raw):
    source = R45_ROOT / 'reference_predictions' / split
    if source.exists():
        shutil.copytree(source, args.root / 'reference_predictions' / split, dirs_exist_ok=True)
    return reference_nominations(args, split, loaders, raw)


def prepare(args):
    loaders, raw, geometry = prepare_data(args)
    identities = {}
    for split in loaders:
        attach_targets(loaders[split], raw[split])
        identities[split] = attach_query_ids(loaders[split])
    for p in PROBLEMS:
        identities['val'][p]['base_ids_also_in_train'] = int(torch.isin(
            loaders['val'][p].batch['base_id'], loaders['train'][p].batch['base_id']).sum())
    dump(args.root / 'input_identity_audit.json', identities)
    sizes = {p: loader.size for p, loader in loaders['train'].items()}
    schedule = paired_schedule(sizes, args.epochs, args.batch_size, args.seed)
    torch.save(schedule, args.root / 'schedule_seed2.pt')
    model, params = initialize(geometry, loaders['train'], raw['train'], args.device, args.seed)
    initial_hash = state_hash(dict(model.named_parameters()))
    refreshed = refresh_memory(model, loaders['train'], args.eval_batch_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scaler = torch.amp.GradScaler('cuda', enabled=args.device.startswith('cuda'))
    witnesses = []
    for p in ('TSP', 'CVRP', 'ATSP', 'OVRPTW'):
        batch = gather_batch(loaders['train'][p], loaders['train'][p].lengths.argsort(descending=True)[:args.batch_size])
        torch.cuda.reset_peak_memory_stats() if args.device.startswith('cuda') else None
        started = time.monotonic()
        values = update_batch(model, optimizer, scaler, batch, .35)
        gradients = {}
        for name in ('retrieval.key', 'retrieval.value', 'retrieval.correction', 'instance_encoder', 'joint_layers'):
            grads = [v.grad for v in dict(model.named_modules())[name].parameters() if v.grad is not None]
            if not grads or not all(torch.isfinite(g).all() for g in grads) or not any(g.abs().sum() > 0 for g in grads):
                raise ValueError('Missing or nonfinite gradients in ' + name)
            gradients[name] = float(sum(g.abs().sum() for g in grads))
        model.eval()
        with torch.no_grad():
            output = model(inputs_only(batch))
            if (model.memory[p].base_ids[output['neighbor_indices']] == batch['base_id'][:, None]).any():
                raise ValueError('A query retrieved itself or a related reference')
        model.train()
        row = dict(problem=p, batch=len(batch['ind']), max_nodes=int(batch['n'].max()),
                   seconds=time.monotonic()-started, gradients=gradients, self_matches=0, **values)
        if args.device.startswith('cuda'):
            row['peak_memory_gib'] = torch.cuda.max_memory_allocated()/2**30
        witnesses.append(row)
        print('[preflight] ' + json.dumps(row), flush=True)
    config = dict(group='B', groups=['B'], R47A='not trained', seed=args.seed, sizes=sizes,
        max_epochs=args.epochs, batch_size=args.batch_size, eval_batch_size=args.eval_batch_size,
        model_params=params, solver_names=GLOBAL_SOLVERS, initial_parameters_hash=initial_hash,
        trainable_parameters=sum(p.numel() for p in model.parameters()), initialization='scratch; no selector weights loaded',
        learning_rate=1e-4, weight_decay=1e-4, dropout=.1, clip_norm=1., warmup_epochs=3,
        rdrop_weight=.35, rdrop_ramp_epochs=3, lr_patience=3, min_epochs=15, stop_patience=8,
        cost_min_delta_pct=.001, loss=json.loads(R43_ARGS.read_text())['loss'],
        schedule_hash=state_hash(schedule), geometry_stats=geometry, full_coverage=True, drop_last=False,
        updates_per_epoch={p: (n+args.batch_size-1)//args.batch_size for p, n in sizes.items()},
        optimizer_updates_per_batch=1, dropout_passes=2, no_augmentation=True, natural_sampling=True,
        checkpoint_rule='strict minimum validation macro_vs_sbs_pct',
        screening=dict(top1_gain_percentage_points=3, regret_relative_reduction_pct=10,
                       maximum_top1_drop_percentage_points_for_regret_target=1,
                       reference='historical R45A; not a freshly trained R47A control'),
        memory=dict(references='train only; every training instance can be a query/reference', neighbors=32,
            summary='pre-joint instance Encoder masked mean/max + log1p(n)/6',
            refresh='epoch start and before formal evaluation; eval/no_grad; keys recomputed with current W_K',
            distance='unscaled negative squared L2 with shared LayerNorm/Linear key mapping',
            gradients='query Encoder and key map; selected reference keys use same trainable map; historical graph summaries detached',
            gap='original FP64 relative regret; log1p(gap/.01) for memory input only',
            self_exclusion='input-derived base ID, all splits including train_eval; no query supervision in retrieval',
            identity='coordinate node-order/D4 canonicalization at 1e-6 precision; exact ordered ATSP matrix',
            identity_limit='no external lineage IDs available; unmarked nonidentical ATSP permutations cannot be inferred',
            behavior_shuffle='whole gap/winner rows, seed4702, within exact problem and exact node count; validation only',
            size_prior='train-only size-quartile mean original relative gaps; no neural model',
            preflight_snapshot=refreshed), preflight=witnesses, test_read=False,
        git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        git_status=subprocess.check_output(['git', 'status', '--short'], text=True).splitlines(),
        runtime=dict(torch=torch.__version__, device=args.device,
                     gpu=torch.cuda.get_device_name() if args.device.startswith('cuda') else None))
    snapshot = args.root / 'source_launch'
    snapshot.mkdir(exist_ok=True)
    config['source_hashes'] = {name: file_hash(Path(__file__).parent / name) for name in SOURCES}
    for name in SOURCES:
        shutil.copy2(Path(__file__).parent / name, snapshot / name)
    dump(args.root / 'protocol.json', config)
    del model, optimizer, scaler
    if args.device.startswith('cuda'):
        torch.cuda.empty_cache()
    reference = {s: references(args, s, loaders[s], raw[s]) for s in loaders}
    return config, loaders, raw, geometry, schedule, reference


@torch.no_grad()
def evaluate(model, loaders, raw, reference, batch_size=128, prediction_dir=None):
    was_training = model.training
    model.eval()
    results = {}
    if prediction_dir is not None:
        prediction_dir.mkdir(parents=True, exist_ok=True)
    try:
        for p, loader in loaders.items():
            scores, margins, indices, weights = [], [], [], []
            for batch in TensorBatchLoader(loader.batch, batch_size, False, False):
                output = model(inputs_only(batch))
                if (model.memory[p].base_ids[output['neighbor_indices']] == batch['base_id'][:, None]).any():
                    raise ValueError('Self/related-instance exclusion failed during evaluation')
                for values, key in ((scores, 'logits'), (margins, 'pair_margin'),
                                    (indices, 'neighbor_indices'), (weights, 'neighbor_weights')):
                    values.append(output[key].detach().cpu())
            scores, margins, indices, weights = [torch.cat(v).numpy() for v in (scores, margins, indices, weights)]
            if len(scores) != loader.size:
                raise ValueError('Evaluation must not drop tail batches')
            result, pred = policy_metrics(scores, raw[p])
            result.update(strong_comparisons(margins, raw[p], reference[p]))
            result['memory_self_matches'] = 0
            result['neighbor_effective_count'] = float(np.exp(-(weights*np.log(weights.clip(1e-30))).sum(-1)).mean())
            results[p] = result
            if prediction_dir is not None:
                np.savez_compressed(prediction_dir / f'{p}.npz', indices=np.arange(len(pred)), logits=scores,
                    pair_margin=margins, pred=pred, winner=raw[p]['winner'], costs=raw[p]['costs'],
                    pool_ids=np.array(raw[p]['pool_ids']), pool=np.array(raw[p]['pool']), reference_top2=reference[p],
                    nodes=loader.lengths.numpy(), neighbor_indices=indices, neighbor_weights=weights,
                    query_base_ids=loader.batch['base_id'].cpu().numpy())
    finally:
        model.train(was_training)
    return add_diagnostics(dict(per_problem=results, macro=aggregate(results), families=families(results)))


def train(args, data):
    protocol, loaders, raw, geometry, schedule, reference = data
    directory = args.root / RUN_NAME
    directory.mkdir(exist_ok=True)
    if (directory / 'last.pt').exists():
        raise ValueError('Do not overwrite an existing R47B run')
    with (directory / 'train.log').open('w') as stream, redirect_stdout(Tee(sys.stdout, stream)), redirect_stderr(Tee(sys.stderr, stream)):
        model, params = initialize(geometry, loaders['train'], raw['train'], args.device, args.seed)
        if state_hash(dict(model.named_parameters())) != protocol['initial_parameters_hash']:
            raise ValueError('Training must not reuse preflight optimization weights')
        config = dict(protocol, model_params=params, wandb_mode='offline')
        dump(directory / 'args.json', config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
        scaler = torch.amp.GradScaler('cuda', enabled=args.device.startswith('cuda'))
        wandb_run = None
        if args.wandb:
            import wandb
            wandb_run = wandb.init(project='selector', entity='yjkds-southern-university-of-science-technology',
                name='R47B_behavior_memory_seed2', config=config, dir=str(directory), mode='offline')
        memory = refresh_memory(model, loaders['train'], args.eval_batch_size)
        initial = {s: evaluate(model, loaders[s], raw[s], reference[s], args.eval_batch_size) for s in ('train', 'val')}
        dump(directory / 'initial_eval.json', initial)
        print_eval('R47B initial val', initial['val']['per_problem'], initial['val']['macro'])
        controller, history, best, successful, peak_lr = CostController(), [], None, 0, 1e-4
        counts, retries, stop = Counter({p: 0 for p in PROBLEMS}), 0, False
        steps = len(schedule['orders'][0])
        for epoch in range(args.epochs):
            refresh_started = time.monotonic()
            memory = refresh_memory(model, loaders['train'], args.eval_batch_size)
            refresh_seconds = time.monotonic()-refresh_started
            model.train()
            started, totals, cursor = time.monotonic(), Counter(), Counter()
            if args.device.startswith('cuda'):
                torch.cuda.reset_peak_memory_stats()
            for step, p in enumerate(schedule['orders'][epoch]):
                offset = cursor[p]*args.batch_size
                idx = schedule['permutations'][epoch][p][offset:offset+args.batch_size]
                cursor[p] += 1
                progress = min(1., (successful+1)/(3*steps))
                lr = 1e-4*progress if epoch < 3 else peak_lr
                for item in optimizer.param_groups:
                    item['lr'] = lr
                values = update_batch(model, optimizer, scaler, gather_batch(loaders['train'][p], idx), .35*progress)
                successful += 1
                counts[p] += 1
                retries += values['skipped']
                for key in ('loss', 'ce', 'cmp', 'risk', 'kl', 'true3', 'pred3', 'all_pairs', 'grad_norm'):
                    totals[key] += values[key]*len(idx)
                totals['samples'] += len(idx)
                if (step+1) % 180 == 0 or step+1 == steps:
                    print(f'[epoch {epoch+1}] update={step+1}/{steps} total={successful} '
                          f'loss={totals["loss"]/totals["samples"]:.5f} KL={totals["kl"]/totals["samples"]:.5f} '
                          f'lr={lr:.2g} samples/s={totals["samples"]/(time.monotonic()-started):.0f}', flush=True)
            if totals['samples'] != sum(protocol['sizes'].values()) or dict(cursor) != protocol['updates_per_epoch']:
                raise ValueError('Every task must cover every training row, including the tail')
            optimization = {k: v/totals['samples'] for k, v in totals.items() if k != 'samples'}
            optimization.update(lr=lr, consistency_weight=.35*progress, amp_skipped=retries,
                samples=totals['samples'], seconds=time.monotonic()-started, memory_refresh_seconds=refresh_seconds)
            if args.device.startswith('cuda'):
                optimization['peak_memory_gib'] = torch.cuda.max_memory_allocated()/2**30
            memory = refresh_memory(model, loaders['train'], args.eval_batch_size)
            validation = evaluate(model, loaders['val'], raw['val'], reference['val'], args.eval_batch_size,
                                  directory / 'val_predictions' / f'epoch{epoch+1:03d}')
            cost = validation['macro']['macro_vs_sbs_pct']
            halve, stop = controller.observe(cost, epoch+1)
            training = evaluate(model, loaders['train'], raw['train'], reference['train'], args.eval_batch_size) if (
                (epoch+1) % 5 == 0 or stop or epoch+1 == args.epochs) else None
            row = dict(epoch=epoch+1, successful_updates=successful, updates_by_problem=dict(counts),
                       optimization=optimization, memory=memory, train=training, val=validation)
            history.append(row)
            if halve:
                peak_lr *= .5
            checkpoint = dict(model=model.state_dict(), optimizer=optimizer.state_dict(), scaler=scaler.state_dict(),
                args=config, epoch=epoch, macro=validation['macro'], successful_updates=successful,
                updates_by_problem=dict(counts), rng=capture_rng(), controller=vars(controller), peak_lr=peak_lr,
                memory_snapshot=memory)
            torch.save(checkpoint, directory / 'last.pt')
            if best is None or cost < best['val']['macro']['macro_vs_sbs_pct']:
                best = row
                torch.save(checkpoint, directory / 'best.pt')
                dump(directory / 'best_eval.json', row)
            dump(directory / 'history.json', history)
            print_eval(f'R47B val epoch {epoch+1}', validation['per_problem'], validation['macro'])
            if training:
                print('[train_eval] ' + json.dumps(training['macro']), flush=True)
            if wandb_run:
                metrics = dict(epoch=epoch+1, successful_updates=successful,
                               **{'optimization/'+k: v for k, v in optimization.items()})
                for split, value in (('train_eval', training), ('val', validation)):
                    if value:
                        metrics.update({split+'/'+k: v for k, v in value['macro'].items()})
                wandb_run.log(metrics)
            from .r43_analysis import plot_run
            plot_run(directory)
            if stop:
                break
        points = [r['val']['macro'] for r in history[-5:]]
        dump(directory / 'result.json', dict(best=best, final=history[-1],
            last5_val={k: float(np.mean([r[k] for r in points])) for k in points[-1]},
            epochs_completed=len(history), successful_updates=successful, updates_by_problem=dict(counts),
            stop_reason='validation early stop' if stop else 'maximum epochs', test_read=False))
        if wandb_run:
            wandb_run.finish()


def size_prior(memory, loaders, raw, path):
    results, choices = {}, {}
    for p, loader in loaders.items():
        bank = memory[p]
        nodes, gaps = bank.nodes.cpu().numpy(), bank.gap.cpu().numpy()
        bounds = np.unique(np.quantile(nodes, [.25, .5, .75]))
        train_bins = np.searchsorted(bounds, nodes, side='right')
        means = np.stack([gaps[train_bins == b].mean(0) if (train_bins == b).any() else gaps.mean(0)
                          for b in range(len(bounds)+1)])
        query_bins = np.searchsorted(bounds, loader.lengths.numpy(), side='right')
        results[p], _ = policy_metrics(-means[query_bins], raw[p])
        choices[p] = dict(boundaries=bounds.tolist(), mean_relative_gaps=means.tolist(),
                          chosen_solver_ids=bank.pool_ids[torch.from_numpy(means.argmin(-1)).to(bank.pool_ids.device)].tolist())
    value = dict(per_problem=results, macro=aggregate(results), families=families(results),
                 definition='train-only problem/size-quartile mean original relative gaps', choices=choices)
    dump(path, value)
    return value


def lock_and_test(args, data=None):
    from .r41_frozen_evaluation import attach_geometry, checked_loader
    if (args.root / 'test_results.json').exists():
        raise ValueError('R47B test is already complete; do not reselect using test')
    if not (args.root / RUN_NAME / 'result.json').exists():
        raise ValueError('Training and validation selection must finish before test')
    if data is None:
        loaders, raw, _ = prepare_data(args)
        for split in loaders:
            attach_query_ids(loaders[split])
        reference = {s: references(args, s, loaders[s], raw[s]) for s in loaders}
    else:
        _, loaders, raw, _, _, reference = data
    path = args.root / RUN_NAME / 'best.pt'
    model, checkpoint = load_r47_checkpoint(path, args.device)
    dump(args.root / 'locked_checkpoints.json', dict(B=dict(path=str(path.resolve()), sha256=file_hash(path),
        epoch=checkpoint['epoch']+1, val=checkpoint['macro'])))
    saved_summaries = {p: bank.summary.clone() for p, bank in model.memory.items()}
    memory = refresh_memory(model, loaders['train'], args.eval_batch_size)
    maximum_error = max(float((bank.summary-saved_summaries[p]).abs().max())
                        for p, bank in model.memory.items())
    if not np.isfinite(maximum_error) or maximum_error > 1e-4:
        raise ValueError('Selected-checkpoint memory rebuild differs from its validation snapshot')
    memory['rebuild_maximum_absolute_error'] = maximum_error
    memory['validation_summary_sha256'] = checkpoint['memory_snapshot']['summary_sha256']
    del saved_summaries
    real = evaluate(model, loaders['val'], raw['val'], reference['val'], args.eval_batch_size,
                    args.root / 'mechanism_predictions' / 'real')
    original, permutations = shuffled_behavior(model.memory)
    try:
        shuffled = evaluate(model, loaders['val'], raw['val'], reference['val'], args.eval_batch_size,
                            args.root / 'mechanism_predictions' / 'shuffled')
    finally:
        restore_behavior(model.memory, original)
    dump(args.root / 'mechanism_check.json', dict(real=real, shuffled=shuffled, seed=4702,
        rule='whole gap/winner rows permuted within exact problem and node count; keys and model unchanged',
        permutations=permutations, checkpoint_sha256=file_hash(path), memory=memory))
    size_prior(model.memory, loaders['val'], raw['val'], args.root / 'size_prior_val.json')
    dump(args.root / 'locked_memory.json', memory)
    for p in PROBLEMS:
        loaders.setdefault('test', {})[p], raw.setdefault('test', {})[p] = checked_loader(p, 'test', args.eval_batch_size, args.device)
        attach_geometry(loaders['test'][p], args.device)
    identities = attach_query_ids(loaders['test'])
    dump(args.root / 'test_input_identity_audit.json', identities)
    reference['test'] = references(args, 'test', loaders['test'], raw['test'])
    started = time.monotonic()
    result = evaluate(model, loaders['test'], raw['test'], reference['test'], args.eval_batch_size,
                      args.root / 'test_predictions' / 'B')
    dump(args.root / 'test_results.json', dict(B=result))
    dump(args.root / 'test_data_manifest.json', {p: {k: r[k] for k in ('pool', 'pool_ids', 'data_hash', 'label_hash')}
                                               for p, r in raw['test'].items()})
    dump(args.root / 'evaluation_runtime.json', dict(instances=sum(v.size for v in loaders['test'].values()),
         seconds=time.monotonic()-started, includes='inference, FP64 metrics and compressed prediction writing'))
    size_prior(model.memory, loaders['test'], raw['test'], args.root / 'size_prior_test.json')
    print_eval('R47B locked best test', result['per_problem'], result['macro'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('all', 'test', 'analysis'), default='all')
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--reference', type=Path, default=Path('code/V4/runs/R39_performance_model'))
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--seed', type=int, choices=(2,), default=2)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--eval-batch-size', type=int, default=128)
    parser.add_argument('--wandb', action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    configure_torch()
    torch.set_num_threads(4)
    if args.stage == 'all':
        data = prepare(args)
        train(args, data)
        lock_and_test(args, data)
    elif args.stage == 'test':
        lock_and_test(args)
    from .r47_analysis import summarize
    summarize(args.root)


if __name__ == '__main__':
    main()

"""R45 three-arm, from-scratch R43A experiments. Default action only writes a plan."""

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
from .multitask_probe import Tee, dump, file_hash, gather_batch, state_hash
from .pairwise_objective import inputs_only, predicted_top3, single_objective
from .performance_experiment import SOURCE_FILES
from .r42_experiment import CostController, paired_schedule, prepare_data
from .r43_experiment import attach_targets, evaluate, reference_nominations, update_batch
from .solver_code_corpus import ROOT, write_json
from .solver_code_embeddings import bundle_provenance, load_bundle
from .solver_code_encoder import derangement, load_r45_checkpoint, make_r45_model
from .train import configure_torch, print_eval, set_seed
from .training_monitor import capture_rng


ARMS = dict(A='handcrafted', B='code', C='shuffled_code')
RUN_NAMES = {arm: f'{arm}_{mode}_seed2' for arm, mode in ARMS.items()}
NEW_SOURCES = ('solver_source_manifest.json', 'solver_code_corpus.py', 'solver_code_embeddings.py',
               'solver_code_encoder.py', 'solver_code_defaults.py', 'r45_experiment.py', 'r45_analysis.py',
               'solver_code_views.py', 'test_r45.py', 'test_solver_code_defaults.py', 'test_solver_code_requests.py',
               'test_solver_code_atlas.py', 'run_v4_r45_embedding.sh', 'run_v4_r45.sh', 'run_v4_r45_campaign.sh')
R43_ARGS = Path('code/V4/runs/R43_pairwise_selection/score_difference_seed2/args.json')


def parameters(group, sdpa=True):
    params = json.loads(R43_ARGS.read_text())['model_params']
    if params['pair_mode'] != 'score_difference' or params.get('node_adapter_mode'):
        raise ValueError('R45 must start from the R43A model definition, without an R44 adapter')
    params.update(sdpa=sdpa, solver_representation=ARMS[group], permutation_seed=4502, code_init_seed=450002)
    return params


def initialize(group, geometry, device='cpu', bundle=None, seed=2):
    set_seed(seed)
    params = parameters(group, sdpa=str(device).startswith('cuda'))
    model = make_r45_model(params, bundle=bundle).to(device)
    with torch.no_grad():
        branch = model.instance_encoder.geometry_residual
        branch.mean.copy_(torch.tensor(geometry['mean'], device=device))
        branch.std.copy_(torch.tensor(geometry['std'], device=device))
    return model, params


def common_parameters(model):
    return {k: p for k, p in model.named_parameters()
            if not k.startswith('solver_encoder.') or k.startswith(('solver_encoder.solver_emb.', 'solver_encoder.norm.'))}


def planned_config(args):
    historical = json.loads(R43_ARGS.read_text())
    expected = dict(batch_size=128, seed=2, learning_rate=1e-4, weight_decay=1e-4, dropout=.1,
                    warmup_epochs=3, rdrop_weight=.35, rdrop_ramp_epochs=3, lr_patience=3,
                    min_epochs=15, stop_patience=8, cost_min_delta_pct=.001)
    for key, value in expected.items():
        if historical.get(key) != value:
            raise ValueError(f'Historical R43A {key} differs from the planned R45 protocol')
    if historical['loss'] != dict(ce=.35, comparison=.35, risk=.02, cost_scale=.01,
                                   focus_weights=dict(true3=.5, predicted3=.25, all_pairs=.25)):
        raise ValueError('R45 must reuse the exact R43A comparison objective')
    return dict(expected, max_epochs=args.epochs, batch_size=args.batch_size, eval_batch_size=args.eval_batch_size,
                solver_names=GLOBAL_SOLVERS, initialization='scratch; old common modules retained explicitly',
                arms=ARMS, model_params={group: parameters(group, args.device.startswith('cuda')) for group in ARMS},
                permutation_seed=4502, semantic_rows_C=derangement().tolist(),
                loss=historical['loss'], clip_norm=1., full_coverage=True, drop_last=False,
                natural_sampling=True, no_augmentation=True, dropout_passes=2, optimizer_updates_per_batch=1,
                checkpoint_rule='strict minimum validation macro_vs_sbs_pct',
                scheduling='same validation-driven rules, not necessarily identical realized LR/stop times',
                embeddings_path=str(args.embeddings.resolve()), test_read=False)


def prepare(args):
    bundle = load_bundle(args.embeddings)
    config = planned_config(args)
    config['embedding_lock'] = json.loads(args.embeddings.with_suffix('.lock.json').read_text())
    config['embedding_provenance'] = bundle_provenance(bundle)
    snapshot = args.root / 'source_launch'
    snapshot.mkdir(parents=True, exist_ok=True)
    sources = set(SOURCE_FILES + NEW_SOURCES + ('pairwise_selector.py', 'pairwise_objective.py',
                                               'r43_experiment.py', 'r43_analysis.py', 'r42_experiment.py', 'r40_experiment.py'))
    hashes = {name: file_hash(Path(__file__).parent / name) for name in sources}
    for name in ('registry.py', 'data.py'):
        hashes['unified_selector/' + name] = file_hash(Path(__file__).parents[1] / 'unified_selector' / name)
    hash_path = snapshot / 'hashes.json'
    if hash_path.exists() and json.loads(hash_path.read_text()) != hashes:
        raise ValueError('Source changed after preparation; preserve this run and use a new root')
    if not hash_path.exists():
        for name in sources:
            shutil.copy2(Path(__file__).parent / name, snapshot / name)
        for name in ('registry.py', 'data.py'):
            shutil.copy2(Path(__file__).parents[1] / 'unified_selector' / name, snapshot / name)
        dump(hash_path, hashes)
    config['source_hashes'] = hashes
    config['r43_args_sha256'] = file_hash(R43_ARGS)
    config['git_commit'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    config['git_status'] = subprocess.check_output(['git', 'status', '--short'], text=True).splitlines()
    loaders, raw, geometry = prepare_data(args)
    for split in loaders:
        attach_targets(loaders[split], raw[split])
    config['sizes'] = {p: loader.size for p, loader in loaders['train'].items()}
    config['size_boundaries'] = {p: np.unique(np.quantile(loader.lengths.numpy(), [.25, .5, .75])).tolist()
                                 for p, loader in loaders['train'].items()}
    config['updates_per_epoch'] = {p: (n + args.batch_size - 1) // args.batch_size for p, n in config['sizes'].items()}
    config['geometry_stats'] = geometry
    schedule = paired_schedule(config['sizes'], args.epochs, args.batch_size, args.seed)
    config['schedule_hash'] = state_hash(schedule)
    schedule_path = args.root / 'schedule_seed2.pt'
    if schedule_path.exists() and state_hash(torch.load(schedule_path, weights_only=False)) != config['schedule_hash']:
        raise ValueError('Paired data schedule changed')
    torch.save(schedule, schedule_path)
    witnesses = {}
    for group in ARMS:
        model, _ = initialize(group, geometry, args.device, bundle, args.seed)
        witnesses[group] = dict(common_parameters_hash=state_hash(common_parameters(model)),
                                parameters_hash=state_hash(dict(model.named_parameters())),
                                trainable_parameters=sum(p.numel() for p in model.parameters()), runtime=[])
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
        scaler = torch.amp.GradScaler('cuda', enabled=args.device.startswith('cuda'))
        for p in ('TSP', 'CVRP', 'ATSP', 'OVRPTW'):
            loader = loaders['train'][p]
            batch = gather_batch(loader, loader.lengths.argsort(descending=True)[:args.batch_size])
            if args.device.startswith('cuda'):
                torch.cuda.reset_peak_memory_stats(args.device)
            started = time.monotonic()
            values = update_batch(model, optimizer, scaler, batch, .35)
            branch = model.solver_encoder.feature_mlp if group == 'A' else model.solver_encoder.fusion
            if not any(v.grad is not None and v.grad.abs().sum() > 0 for v in branch.parameters()):
                raise ValueError('Solver feature branch did not receive gradients on real inputs')
            row = dict(problem=p, batch=len(batch['ind']), max_nodes=int(batch['n'].max()),
                       seconds=time.monotonic()-started, **values)
            if args.device.startswith('cuda'):
                row['peak_memory_gib'] = torch.cuda.max_memory_allocated(args.device) / 2**30
            witnesses[group]['runtime'].append(row)
            print('[preflight ' + group + '] ' + json.dumps(row), flush=True)
        del model, optimizer, scaler
        if args.device.startswith('cuda'):
            torch.cuda.empty_cache()
    if len({v['common_parameters_hash'] for v in witnesses.values()}) != 1:
        raise ValueError('A/B/C common initial parameters differ')
    if witnesses['B']['parameters_hash'] != witnesses['C']['parameters_hash']:
        raise ValueError('B/C must differ only in fixed semantic ownership, not trainable initial values')
    config['initialization_checks'] = witnesses
    config['runtime'] = dict(torch_version=torch.__version__, device=args.device,
                             gpu=torch.cuda.get_device_name(args.device) if args.device.startswith('cuda') else None)
    protocol = args.root / 'protocol.json'
    if protocol.exists() and json.loads(protocol.read_text()) != config:
        raise ValueError('R45 protocol changed; do not reuse this experiment root')
    write_json(protocol, config)
    print('[prepared] Disposable full-batch update witnesses passed; training starts fresh; no test or API read', flush=True)


def load_training_data(args):
    config = json.loads((args.root / 'protocol.json').read_text())
    if file_hash(R43_ARGS) != config['r43_args_sha256']:
        raise ValueError('The reference R43A model configuration changed after preparation')
    for key, value in (('batch_size', args.batch_size), ('max_epochs', args.epochs), ('seed', args.seed)):
        if config[key] != value:
            raise ValueError(f'{key} differs from the locked training protocol')
    for name, sha in config['source_hashes'].items():
        path = Path(__file__).parents[1] / name if name.startswith('unified_selector/') else Path(__file__).parent / name
        if file_hash(path) != sha:
            raise ValueError('Training source changed after preparation: ' + name)
    if file_hash(args.embeddings) != config['embedding_lock']['sha256']:
        raise ValueError('Embedding bundle changed after preparation')
    bundle = load_bundle(args.embeddings)
    loaders, raw, geometry = prepare_data(args)
    for split in loaders:
        attach_targets(loaders[split], raw[split])
    schedule = torch.load(args.root / 'schedule_seed2.pt', weights_only=False)
    if state_hash(schedule) != config['schedule_hash']:
        raise ValueError('Paired batch schedule changed')
    reference = {s: reference_nominations(args, s, loaders[s], raw[s]) for s in loaders}
    return config, bundle, loaders, raw, geometry, schedule, reference


def run(group, args, data):
    directory = args.root / RUN_NAMES[group]
    directory.mkdir(exist_ok=True)
    if (directory / 'last.pt').exists():
        raise ValueError('Refusing to overwrite an existing R45 run')
    with (directory / 'train.log').open('w') as stream, redirect_stdout(Tee(sys.stdout, stream)), redirect_stderr(Tee(sys.stderr, stream)):
        train(group, args, directory, data)


def train(group, args, directory, data):
    protocol, bundle, loaders, raw, geometry, schedule, reference = data
    model, params = initialize(group, geometry, args.device, bundle, args.seed)
    config = dict(protocol, group=group, model_params=params, wandb_mode='offline')
    dump(directory / 'args.json', config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scaler = torch.amp.GradScaler('cuda', enabled=args.device.startswith('cuda'))
    wandb_run = None
    if args.wandb:
        import wandb
        wandb_run = wandb.init(project='selector', entity='yjkds-southern-university-of-science-technology',
                               name='R45_' + directory.name, config=config, dir=str(directory), mode='offline')
    initial = {s: evaluate(model, loaders[s], raw[s], reference[s], args.eval_batch_size) for s in ('train', 'val')}
    dump(directory / 'initial_eval.json', initial)
    controller, history, best, successful, peak_lr = CostController(), [], None, 0, 1e-4
    counts, retries = Counter({p: 0 for p in PROBLEMS}), 0
    steps = len(schedule['orders'][0])
    for epoch in range(args.epochs):
        model.train()
        if args.device.startswith('cuda'):
            torch.cuda.reset_peak_memory_stats(args.device)
        started, totals, cursor = time.monotonic(), Counter(), Counter()
        for step, p in enumerate(schedule['orders'][epoch]):
            offset = cursor[p] * args.batch_size
            indices = schedule['permutations'][epoch][p][offset:offset + args.batch_size]
            cursor[p] += 1
            progress = min(1., (successful + 1) / (3 * steps))
            lr = 1e-4 * progress if epoch < 3 else peak_lr
            for item in optimizer.param_groups:
                item['lr'] = lr
            values = update_batch(model, optimizer, scaler, gather_batch(loaders['train'][p], indices), .35 * progress)
            successful += 1
            counts[p] += 1
            retries += values['skipped']
            for key in ('loss', 'ce', 'cmp', 'risk', 'kl', 'true3', 'pred3', 'all_pairs', 'grad_norm'):
                totals[key] += values[key] * len(indices)
            totals['samples'] += len(indices)
            if (step + 1) % 180 == 0 or step + 1 == steps:
                print(f'[epoch {epoch+1}] update={step+1}/{steps} total={successful} '
                      f'loss={totals["loss"]/totals["samples"]:.5f} KL={totals["kl"]/totals["samples"]:.5f} '
                      f'lr={lr:.2g} samples/s={totals["samples"]/(time.monotonic()-started):.0f}', flush=True)
        if totals['samples'] != sum(protocol['sizes'].values()) or dict(cursor) != protocol['updates_per_epoch']:
            raise ValueError('Every task must cover all samples once, including tail batches')
        optimization = {k: v / totals['samples'] for k, v in totals.items() if k != 'samples'}
        optimization.update(lr=lr, consistency_weight=.35 * progress, amp_skipped=retries,
                            samples=totals['samples'], seconds=time.monotonic()-started)
        if args.device.startswith('cuda'):
            optimization['peak_memory_gib'] = torch.cuda.max_memory_allocated(args.device) / 2**30
        validation = evaluate(model, loaders['val'], raw['val'], reference['val'], args.eval_batch_size,
                              directory / 'val_predictions' / f'epoch{epoch+1:03d}')
        cost = validation['macro']['macro_vs_sbs_pct']
        halve, stop = controller.observe(cost, epoch+1)
        training = evaluate(model, loaders['train'], raw['train'], reference['train'], args.eval_batch_size) if (
            (epoch + 1) % 5 == 0 or stop or epoch + 1 == args.epochs) else None
        row = dict(epoch=epoch+1, successful_updates=successful, updates_by_problem=dict(counts),
                   optimization=optimization, train=training, val=validation)
        history.append(row)
        if halve:
            peak_lr *= .5
        checkpoint = dict(model=model.state_dict(), optimizer=optimizer.state_dict(), scaler=scaler.state_dict(),
                          args=config, epoch=epoch, macro=validation['macro'], successful_updates=successful,
                          updates_by_problem=dict(counts), rng=capture_rng(), controller=vars(controller), peak_lr=peak_lr)
        torch.save(checkpoint, directory / 'last.pt')
        if best is None or cost < best['val']['macro']['macro_vs_sbs_pct']:
            best = row
            torch.save(checkpoint, directory / 'best.pt')
            dump(directory / 'best_eval.json', row)
        dump(directory / 'history.json', history)
        print_eval(f'R45{group} val epoch {epoch+1}', validation['per_problem'], validation['macro'])
        if wandb_run:
            metrics = dict(epoch=epoch+1, successful_updates=successful,
                           **{'optimization/' + k: v for k, v in optimization.items()})
            for split, value in (('train_eval', training), ('val', validation)):
                if value is not None:
                    metrics.update({split + '/' + k: v for k, v in value['macro'].items()})
            wandb_run.log(metrics)
        from .r43_analysis import plot_run
        plot_run(directory)
        if stop:
            break
    points = [r['val']['macro'] for r in history[-5:]]
    last5 = {k: float(np.mean([r[k] for r in points])) for k in points[-1]}
    dump(directory / 'result.json', dict(best=best, final=history[-1], last5_val=last5,
         epochs_completed=len(history), successful_updates=successful, updates_by_problem=dict(counts),
         stop_reason='validation early stop' if stop else 'maximum epochs', test_read=False))
    if wandb_run:
        wandb_run.finish()


def lock_and_test(args):
    from .r41_frozen_evaluation import attach_geometry, checked_loader
    if (args.root / 'test_results.json').exists():
        raise ValueError('Test already completed; no test-driven checkpoint reselection')
    records = {}
    for group, name in RUN_NAMES.items():
        directory = args.root / name
        if not (directory / 'result.json').exists():
            raise ValueError('All three arms must complete validation selection before test')
        path = directory / 'best.pt'
        model, checkpoint = load_r45_checkpoint(path)
        records[group] = dict(path=str(path.resolve()), sha256=file_hash(path), epoch=checkpoint['epoch']+1,
                              val=checkpoint['macro'], model_params=checkpoint['args']['model_params'])
        del model
    lock = args.root / 'locked_checkpoints.json'
    if lock.exists() and json.loads(lock.read_text()) != records:
        raise ValueError('Locked checkpoints changed')
    dump(lock, records)
    loaders, raw = {}, {}
    for p in PROBLEMS:
        loaders[p], raw[p] = checked_loader(p, 'test', args.eval_batch_size, args.device)
        attach_geometry(loaders[p], args.device)
    reference = reference_nominations(args, 'test', loaders, raw)
    results = {}
    for group, record in records.items():
        model, _ = load_r45_checkpoint(record['path'], args.device)
        results[group] = evaluate(model, loaders, raw, reference, args.eval_batch_size, args.root / 'test_predictions' / group)
        del model
    dump(args.root / 'test_data_manifest.json', {p: {k: r[k] for k in ('pool', 'pool_ids', 'data_hash', 'label_hash')} for p, r in raw.items()})
    dump(args.root / 'test_results.json', results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('plan', 'prepare', 'train', 'test', 'analysis', 'all'), default='plan')
    parser.add_argument('--group', choices=tuple(ARMS))
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--reference', type=Path, default=Path('code/V4/runs/R39_performance_model'))
    parser.add_argument('--embeddings', type=Path, default=ROOT / 'solver_code_vectors_atlas.pt')
    parser.add_argument('--seed', type=int, choices=(2,), default=2)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--eval-batch-size', type=int, default=128)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--wandb', action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if not 1 <= args.epochs <= 40 or args.batch_size != 128:
        parser.error('R45 uses batch128 without accumulation and at most40 epochs')
    args.root.mkdir(parents=True, exist_ok=True)
    configure_torch()
    torch.set_num_threads(4)
    if args.stage == 'plan':
        write_json(args.root / 'planned_configs.json', planned_config(args))
        print('[plan only] No API call, model training, validation evaluation, or test read')
    if args.stage in ('prepare', 'all'):
        prepare(args)
    if args.stage in ('train', 'all'):
        if args.stage == 'train' and args.group is None:
            parser.error('--group is required for --stage train')
        data = load_training_data(args)
        for group in ARMS if args.stage == 'all' else (args.group,):
            run(group, args, data)
        del data
    if args.stage in ('test', 'all'):
        lock_and_test(args)
    if args.stage in ('analysis', 'all'):
        from .r45_analysis import summarize
        summarize(args.root)


if __name__ == '__main__':
    main()

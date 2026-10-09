"""R46A only: from-scratch source-token interaction, then locked-checkpoint test."""

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
from .pairwise_objective import inputs_only
from .r42_experiment import CostController, paired_schedule, prepare_data
from .r43_experiment import attach_targets, evaluate, reference_nominations, update_batch
from .r45_experiment import R43_ARGS
from .r46_model import load_r46_checkpoint, make_r46_model
from .solver_code_embeddings import bundle_provenance, load_bundle
from .solver_code_tokens import source_package_audit
from .train import configure_torch, print_eval, set_seed
from .training_monitor import capture_rng


ROOT = Path('code/V4/runs/R46_code_token_interaction')
R45_ROOT = Path('code/V4/runs/R45_solver_code_embeddings')
RUN_NAME = 'A_code_tokens_seed2'
SOURCES = ('solver_code_tokens.py', 'r46_model.py', 'r46_experiment.py', 'r46_analysis.py',
           'test_r46.py', 'run_v4_r46.sh', 'V4Model.py', 'dual_stream.py', 'local_geometry.py',
           'pairwise_selector.py', 'pairwise_objective.py', 'r43_experiment.py', 'r42_experiment.py',
           'r45_experiment.py', 'solver_code_encoder.py', 'solver_code_embeddings.py',
           'tensor_loader.py', 'multitask_probe.py', 'performance_targets.py')


def parameters(device='cpu'):
    original = json.loads(R43_ARGS.read_text())['model_params']
    keys = ('embedding_dim', 'head_num', 'qkv_dim', 'ff_hidden_dim', 'head_hidden_dim',
            'encoder_layer_num', 'joint_layer_num', 'dropout', 'stats_dim', 'rezero',
            'local_geometry', 'geometry_mode', 'ignore_coord_dist', 'native_winner', 'pair_mode')
    return dict({key: original[key] for key in keys}, architecture='source_token_interaction',
                solver_representation='source_tokens', sdpa=str(device).startswith('cuda'))


def initialize(geometry, bundle, device, seed=2):
    set_seed(seed)
    params = parameters(device)
    model = make_r46_model(params, bundle).to(device)
    with torch.no_grad():
        residual = model.instance_encoder.geometry_residual
        residual.mean.copy_(torch.tensor(geometry['mean'], device=device))
        residual.std.copy_(torch.tensor(geometry['std'], device=device))
    return model, params


def prepare(args):
    bundle = load_bundle(args.embeddings)
    loaders, raw, geometry = prepare_data(args)
    for split in loaders:
        attach_targets(loaders[split], raw[split])
    sizes = {p: loader.size for p, loader in loaders['train'].items()}
    schedule = paired_schedule(sizes, args.epochs, args.batch_size, args.seed)
    audit = source_package_audit(bundle, {p: raw['train'][p]['pool'] for p in PROBLEMS})
    dump(args.root / 'source_package_audit.json', audit)
    print('[source packages] ' + json.dumps(audit), flush=True)
    historical = json.loads(R43_ARGS.read_text())
    config = dict(seed=args.seed, group='A', groups=['A'], R46B='not implemented or run',
                  max_epochs=args.epochs, batch_size=args.batch_size, eval_batch_size=args.eval_batch_size,
                  solver_names=GLOBAL_SOLVERS, model_params=parameters(args.device),
                  initialization='scratch; no selector checkpoint loaded; all learnable parameters trainable',
                  learning_rate=1e-4, weight_decay=1e-4, dropout=.1, clip_norm=1.,
                  warmup_epochs=3, rdrop_weight=.35, rdrop_ramp_epochs=3,
                  lr_patience=3, min_epochs=15, stop_patience=8, cost_min_delta_pct=.001,
                  loss=historical['loss'], sizes=sizes, full_coverage=True, drop_last=False,
                  updates_per_epoch={p: (n+args.batch_size-1)//args.batch_size for p, n in sizes.items()},
                  optimizer_updates_per_batch=1, dropout_passes=2, natural_sampling=True, no_augmentation=True,
                  checkpoint_rule='strict minimum validation macro_vs_sbs_pct',
                  schedule_hash=state_hash(schedule), geometry_stats=geometry, source_package_audit=audit,
                  embeddings_path=str(args.embeddings.resolve()), embedding_sha256=file_hash(args.embeddings),
                  embedding_provenance=bundle_provenance(bundle), test_read=False,
                  git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                  git_status=subprocess.check_output(['git', 'status', '--short'], text=True).splitlines())
    snapshot = args.root / 'source_launch'
    snapshot.mkdir(exist_ok=True)
    config['source_hashes'] = {name: file_hash(Path(__file__).parent / name) for name in SOURCES}
    for name in SOURCES:
        shutil.copy2(Path(__file__).parent / name, snapshot / name)
    torch.save(schedule, args.root / 'schedule_seed2.pt')
    model, _ = initialize(geometry, bundle, args.device, args.seed)
    config['trainable_parameters'] = sum(p.numel() for p in model.parameters())
    config['initial_parameters_hash'] = state_hash(dict(model.named_parameters()))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scaler = torch.amp.GradScaler('cuda', enabled=args.device.startswith('cuda'))
    witnesses = []
    for p in ('TSP', 'CVRP', 'ATSP', 'OVRPTW'):
        loader = loaders['train'][p]
        batch = gather_batch(loader, loader.lengths.argsort(descending=True)[:args.batch_size])
        if args.device.startswith('cuda'):
            torch.cuda.reset_peak_memory_stats(args.device)
        started = time.monotonic()
        values = update_batch(model, optimizer, scaler, batch, .35)
        gradients = {}
        for name in ('solver_encoder.projection', 'source_pool', 'joint_layers', 'instance_encoder'):
            parameters_ = dict(model.named_modules())[name].parameters()
            grads = [v.grad for v in parameters_ if v.grad is not None]
            if not grads or not all(torch.isfinite(g).all() for g in grads) or not any(g.abs().sum() > 0 for g in grads):
                raise ValueError('Missing/nonfinite gradient in ' + name)
            gradients[name] = float(sum(g.abs().sum() for g in grads))
        row = dict(problem=p, batch=len(batch['ind']), max_nodes=int(batch['n'].max()),
                   seconds=time.monotonic()-started, gradients=gradients, **values)
        if args.device.startswith('cuda'):
            row['peak_memory_gib'] = torch.cuda.max_memory_allocated(args.device)/2**30
        witnesses.append(row)
        print('[preflight] ' + json.dumps(row), flush=True)
    config['preflight'] = witnesses
    config['runtime'] = dict(torch=torch.__version__, device=args.device,
                            gpu=torch.cuda.get_device_name(args.device) if args.device.startswith('cuda') else None)
    dump(args.root / 'protocol.json', config)
    del model, optimizer, scaler
    if args.device.startswith('cuda'):
        torch.cuda.empty_cache()
    reference = {}
    for split in loaders:
        source = R45_ROOT / 'reference_predictions' / split
        if source.exists():
            shutil.copytree(source, args.root / 'reference_predictions' / split, dirs_exist_ok=True)
        reference[split] = reference_nominations(args, split, loaders[split], raw[split])
    return config, bundle, loaders, raw, geometry, schedule, reference


def train(args, data):
    protocol, bundle, loaders, raw, geometry, schedule, reference = data
    directory = args.root / RUN_NAME
    directory.mkdir(exist_ok=True)
    if (directory / 'last.pt').exists():
        raise ValueError('Do not overwrite a previous R46 run')
    with (directory / 'train.log').open('w') as stream, redirect_stdout(Tee(sys.stdout, stream)), redirect_stderr(Tee(sys.stderr, stream)):
        model, params = initialize(geometry, bundle, args.device, args.seed)
        if state_hash(dict(model.named_parameters())) != protocol['initial_parameters_hash']:
            raise ValueError('Preflight weights must not be reused as training initialization')
        config = dict(protocol, model_params=params, wandb_mode='offline')
        dump(directory / 'args.json', config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
        scaler = torch.amp.GradScaler('cuda', enabled=args.device.startswith('cuda'))
        wandb_run = None
        if args.wandb:
            import wandb
            wandb_run = wandb.init(project='selector', entity='yjkds-southern-university-of-science-technology',
                                   name='R46A_source_tokens_seed2', config=config, dir=str(directory), mode='offline')
        initial = {s: evaluate(model, loaders[s], raw[s], reference[s], args.eval_batch_size) for s in ('train', 'val')}
        dump(directory / 'initial_eval.json', initial)
        print_eval('R46A initial val', initial['val']['per_problem'], initial['val']['macro'])
        controller, history, best, successful, peak_lr = CostController(), [], None, 0, 1e-4
        counts, retries, stop = Counter({p: 0 for p in PROBLEMS}), 0, False
        steps = len(schedule['orders'][0])
        for epoch in range(args.epochs):
            model.train()
            started, totals, cursor = time.monotonic(), Counter(), Counter()
            if args.device.startswith('cuda'):
                torch.cuda.reset_peak_memory_stats(args.device)
            for step, p in enumerate(schedule['orders'][epoch]):
                offset = cursor[p]*args.batch_size
                indices = schedule['permutations'][epoch][p][offset:offset+args.batch_size]
                cursor[p] += 1
                progress = min(1., (successful+1)/(3*steps))
                lr = 1e-4*progress if epoch < 3 else peak_lr
                for item in optimizer.param_groups:
                    item['lr'] = lr
                values = update_batch(model, optimizer, scaler, gather_batch(loaders['train'][p], indices), .35*progress)
                successful += 1
                counts[p] += 1
                retries += values['skipped']
                for key in ('loss', 'ce', 'cmp', 'risk', 'kl', 'true3', 'pred3', 'all_pairs', 'grad_norm'):
                    totals[key] += values[key]*len(indices)
                totals['samples'] += len(indices)
                if (step+1) % 180 == 0 or step+1 == steps:
                    print(f'[epoch {epoch+1}] update={step+1}/{steps} total={successful} '
                          f'loss={totals["loss"]/totals["samples"]:.5f} KL={totals["kl"]/totals["samples"]:.5f} '
                          f'lr={lr:.2g} samples/s={totals["samples"]/(time.monotonic()-started):.0f}', flush=True)
            if totals['samples'] != sum(protocol['sizes'].values()) or dict(cursor) != protocol['updates_per_epoch']:
                raise ValueError('Each task must cover every instance once, including its tail batch')
            optimization = {k: v/totals['samples'] for k, v in totals.items() if k != 'samples'}
            optimization.update(lr=lr, consistency_weight=.35*progress, amp_skipped=retries,
                                samples=totals['samples'], seconds=time.monotonic()-started)
            if args.device.startswith('cuda'):
                optimization['peak_memory_gib'] = torch.cuda.max_memory_allocated(args.device)/2**30
            validation = evaluate(model, loaders['val'], raw['val'], reference['val'], args.eval_batch_size,
                                  directory / 'val_predictions' / f'epoch{epoch+1:03d}')
            cost = validation['macro']['macro_vs_sbs_pct']
            halve, stop = controller.observe(cost, epoch+1)
            training = evaluate(model, loaders['train'], raw['train'], reference['train'], args.eval_batch_size) if (
                (epoch+1) % 5 == 0 or stop or epoch+1 == args.epochs) else None
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
            print_eval(f'R46A val epoch {epoch+1}', validation['per_problem'], validation['macro'])
            if wandb_run:
                metrics = dict(epoch=epoch+1, successful_updates=successful, **{'optimization/'+k: v for k, v in optimization.items()})
                for split, value in (('train_eval', training), ('val', validation)):
                    if value is not None:
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


def lock_and_test(args):
    from .r41_frozen_evaluation import attach_geometry, checked_loader
    if (args.root / 'test_results.json').exists():
        raise ValueError('R46A test already completed; do not reselect using test')
    if not (args.root / RUN_NAME / 'result.json').exists():
        raise ValueError('Training and validation selection must finish before test')
    path = args.root / RUN_NAME / 'best.pt'
    model, checkpoint = load_r46_checkpoint(path, args.device)
    record = dict(path=str(path.resolve()), sha256=file_hash(path), epoch=checkpoint['epoch']+1,
                  val=checkpoint['macro'])
    dump(args.root / 'locked_checkpoints.json', dict(A=record))
    loaders, raw = {}, {}
    for p in PROBLEMS:
        loaders[p], raw[p] = checked_loader(p, 'test', args.eval_batch_size, args.device)
        attach_geometry(loaders[p], args.device)
    source = R45_ROOT / 'reference_predictions' / 'test'
    if source.exists():
        shutil.copytree(source, args.root / 'reference_predictions' / 'test', dirs_exist_ok=True)
    reference = reference_nominations(args, 'test', loaders, raw)
    started = time.monotonic()
    result = evaluate(model, loaders, raw, reference, args.eval_batch_size, args.root / 'test_predictions' / 'A')
    dump(args.root / 'test_results.json', dict(A=result))
    dump(args.root / 'test_data_manifest.json', {p: {k: r[k] for k in ('pool', 'pool_ids', 'data_hash', 'label_hash')} for p, r in raw.items()})
    dump(args.root / 'evaluation_runtime.json', dict(instances=sum(v.size for v in loaders.values()),
         seconds=time.monotonic()-started, includes='inference, FP64 metrics and compressed prediction writing'))
    print_eval('R46A locked best test', result['per_problem'], result['macro'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('all', 'test', 'analysis'), default='all')
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--reference', type=Path, default=Path('code/V4/runs/R39_performance_model'))
    parser.add_argument('--embeddings', type=Path, default=R45_ROOT / 'solver_code_vectors_atlas.pt')
    parser.add_argument('--seed', type=int, choices=(2,), default=2)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--batch-size', type=int, choices=(128,), default=128)
    parser.add_argument('--eval-batch-size', type=int, default=128)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--wandb', action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if not 15 <= args.epochs <= 40:
        parser.error('Use the R45 minimum15/maximum40 epoch rules')
    args.root.mkdir(parents=True, exist_ok=True)
    configure_torch()
    torch.set_num_threads(4)
    if args.stage == 'all':
        if (args.root / RUN_NAME / 'last.pt').exists():
            parser.error('Existing R46 run is preserved; do not overwrite')
        data = prepare(args)
        train(args, data)
        del data
        if args.device.startswith('cuda'):
            torch.cuda.empty_cache()
    if args.stage in ('all', 'test'):
        lock_and_test(args)
    if args.stage in ('all', 'analysis'):
        from .r46_analysis import summarize
        summarize(args.root)


if __name__ == '__main__':
    main()

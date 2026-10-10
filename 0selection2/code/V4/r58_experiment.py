"""One from-scratch R45A baseline on the independently released scenario_v2."""

import argparse
import json
import shutil
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from ..unified_selector.registry import GLOBAL_SOLVERS, PROBLEMS
from .multitask_probe import gather_batch, state_hash
from .pairwise_objective import inputs_only
from .r40_experiment import aggregate, families
from .r42_experiment import CostController, paired_schedule
from .r43_experiment import policy_metrics, update_batch
from .r45_experiment import parameters
from .solver_code_encoder import make_r45_model
from .r58_data import attach_test_geometry, make_loader, prepare, release_check
from .r58_scenario import ROOT, file_hash, save_json
from .tensor_loader import TensorBatchLoader
from .train import configure_torch, set_seed
from .training_monitor import capture_rng, restore_rng


RUN = 'R45A_scenario_v2_seed2'


def source_snapshot(directory):
    files = ('V4Model.py', 'dual_stream.py', 'local_geometry.py', 'solver_features.py',
             'pairwise_selector.py', 'pairwise_objective.py', 'r43_experiment.py',
             'r42_experiment.py', 'r40_experiment.py', 'solver_code_encoder.py',
             'train.py', 'multitask_probe.py', 'tensor_loader.py', 'training_monitor.py',
             'performance_experiment.py', 'performance_evaluation.py', 'performance_targets.py',
             'r45_experiment.py', 'r58_data.py', 'r58_scenario.py', 'r58_experiment.py')
    base = Path(__file__).parent
    sources = {name: base / name for name in files}
    sources.update({f'unified_selector/{name}': base.parent / 'unified_selector' / name
                    for name in ('registry.py', 'data.py')})
    sources['r43_reference_args.json'] = base / 'runs/R43_pairwise_selection/score_difference_seed2/args.json'
    hashes = {name: file_hash(path) for name, path in sources.items()}
    snapshot = directory / 'source_launch'
    path = snapshot / 'hashes.json'
    if path.exists() and json.loads(path.read_text()) != hashes:
        raise ValueError('Training source changed; preserve the run and use a new experiment revision')
    if not path.exists():
        for name, source in sources.items():
            target = snapshot / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        save_json(path, hashes)
    return hashes


def save_checkpoint(path, checkpoint):
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(checkpoint, temporary)
    temporary.replace(path)


def resume_checkpoint(directory, device):
    choices = [torch.load(directory / name, map_location=device, weights_only=False)
               for name in ('last.pt', 'best.pt') if (directory / name).exists()]
    if not choices:
        return None
    # If saving best succeeded just before interruption, resume that completed
    # epoch rather than an older last.pt. Both contain optimizer/RNG/controller.
    checkpoint = max(choices, key=lambda c: c['epoch'])
    if (directory / 'best.pt').exists():
        best = torch.load(directory / 'best.pt', map_location='cpu', weights_only=False)
        save_json(directory / 'best_eval.json', best['history'][-1])
    return checkpoint


def check_test_checkpoint(checkpoint, result, locked, release_sha256):
    config = checkpoint['args']
    if (config['deployment_lock_sha256'] != locked['lock_sha256'] or
            config['release_sha256'] != release_sha256):
        raise ValueError('Checkpoint was trained on another deployment/release')
    best = result['best']
    metric = checkpoint['macro']['macro_actual_regret_pct']
    if (best['epoch'] != checkpoint['epoch'] + 1 or best['val']['macro'] != checkpoint['macro'] or
            result['best_actual_regret_pct'] != metric):
        raise ValueError('Checkpoint and completed validation selection disagree')
    if metric != min(row['val']['macro']['macro_actual_regret_pct'] for row in checkpoint['history']):
        raise ValueError('Checkpoint is not the validation minimum in its completed history')


def remaining_epochs(controller, start_epoch, maximum=40):
    stopped = start_epoch >= controller.min_epochs and controller.bad_epochs >= controller.patience
    return range(start_epoch, start_epoch if stopped else maximum)


def initialize(geometry, device):
    set_seed(2)
    params = parameters('A', sdpa=str(device).startswith('cuda'))
    model = make_r45_model(params).to(device)
    with torch.no_grad():
        branch = model.instance_encoder.geometry_residual
        branch.mean.copy_(torch.tensor(geometry['mean'], device=device))
        branch.std.copy_(torch.tensor(geometry['std'], device=device))
    return model, params


@torch.no_grad()
def evaluate(model, loaders, raw, batch_size=128, output_dir=None):
    was_training = model.training
    model.eval()
    per_problem = {}
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
    try:
        for problem, loader in loaders.items():
            scores = torch.cat([model(inputs_only(batch))['logits'].float().cpu()
                for batch in TensorBatchLoader(loader.batch, batch_size, False, False)]).numpy()
            metrics, pred = policy_metrics(scores, raw[problem])
            per_problem[problem] = metrics
            if output_dir is not None:
                np.savez_compressed(output_dir / f'{problem}.npz', indices=np.arange(len(pred)),
                    logits=scores, pred=pred, costs=raw[problem]['costs'], winner=raw[problem]['winner'],
                    pool=np.asarray(raw[problem]['pool']), pool_ids=np.asarray(raw[problem]['pool_ids']),
                    nodes=loader.lengths.numpy(), label_sha256=raw[problem]['label_hash'])
    finally:
        model.train(was_training)
    return dict(per_problem=per_problem, macro=aggregate(per_problem), families=families(per_problem))


def train(args):
    directory = args.root / RUN
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / 'result.json').exists():
        print('[R58] completed training already present; preserve it', flush=True)
        return
    locked, release = release_check(args.root)
    training_sources = source_snapshot(directory)
    loaders, raw, geometry = prepare(args.root, args.device)
    sizes = {p: loader.size for p, loader in loaders['train'].items()}
    schedule = paired_schedule(sizes, 40, 128, 2)
    schedule_path = directory / 'schedule_seed2.pt'
    if not schedule_path.exists():
        torch.save(schedule, schedule_path)
    model, params = initialize(geometry, args.device)
    config = dict(scenario='scenario_v2', group='single_full_baseline', seed=2,
        solver_names=list(GLOBAL_SOLVERS), pools=locked['pools'], source_hashes=training_sources,
        model_params=params, batch_size=128, max_epochs=40, learning_rate=1e-4, weight_decay=1e-4,
        dropout=.1, warmup_epochs=3, rdrop_weight=.35, rdrop_ramp_epochs=3, clip_norm=1.,
        lr_patience=3, min_epochs=15, stop_patience=8, min_delta_pct=.001,
        checkpoint_rule='strict minimum FULL 18-task validation macro_actual_regret_pct',
        loss=dict(ce=.35, comparison=.35, risk=.02, cost_scale=.01,
                  true3=.5, predicted3=.25, all_pairs=.25),
        initialization='all trainable parameters from scratch; no selector or solver weights loaded',
        full_coverage=True, drop_last=False, augmentation=False,
        sizes=sizes, schedule_sha256=state_hash(schedule), geometry_stats=geometry,
        deployment_lock_sha256=locked['lock_sha256'], release_sha256=file_hash(args.root / 'scenario_v2/release.json'),
        historical_comparison='old R45A and old results are a different scenario, not matched performance baselines',
        wandb_mode='offline')
    path = directory / 'args.json'
    if path.exists() and json.loads(path.read_text()) != config:
        raise ValueError('Training configuration changed during a run')
    save_json(path, config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scaler = torch.amp.GradScaler('cuda', enabled=str(args.device).startswith('cuda'))
    controller, successful, peak_lr, start_epoch = CostController(), 0, 1e-4, 0
    history, counts, best_cost = [], Counter({p: 0 for p in PROBLEMS}), float('inf')
    checkpoint = resume_checkpoint(directory, args.device)
    if checkpoint is not None:
        if checkpoint['args'] != config:
            raise ValueError('Resume checkpoint belongs to another scenario/configuration')
        model.load_state_dict(checkpoint['model'], strict=True)
        optimizer.load_state_dict(checkpoint['optimizer'])
        scaler.load_state_dict(checkpoint['scaler'])
        controller.__dict__.update(checkpoint['controller'])
        successful, peak_lr, start_epoch = checkpoint['successful_updates'], checkpoint['peak_lr'], checkpoint['epoch'] + 1
        history, counts, best_cost = checkpoint['history'], Counter(checkpoint['updates_by_problem']), checkpoint['best_cost']
        restore_rng(checkpoint['rng'])
    else:
        save_json(directory / 'initialization.json', dict(model_sha256=state_hash(model.state_dict()),
            trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad)))
    wandb_run = None
    if args.wandb:
        import wandb
        wandb_run = wandb.init(project='selector', entity='yjkds-southern-university-of-science-technology',
            name='R58_scenario_v2_seed2', config=config, dir=str(directory), mode='offline')
    updates_per_epoch = len(schedule['orders'][0])
    stopped = start_epoch >= controller.min_epochs and controller.bad_epochs >= controller.patience
    for epoch in remaining_epochs(controller, start_epoch):
        model.train()
        started, weighted, cursor = time.monotonic(), Counter(), Counter()
        for step, problem in enumerate(schedule['orders'][epoch]):
            offset = cursor[problem] * 128
            indices = schedule['permutations'][epoch][problem][offset:offset + 128]
            cursor[problem] += 1
            batch = gather_batch(loaders['train'][problem], indices)
            progress = min(1., (successful + 1) / (3 * updates_per_epoch))
            lr = 1e-4 * progress if epoch < 3 else peak_lr
            for group in optimizer.param_groups:
                group['lr'] = lr
            values = update_batch(model, optimizer, scaler, batch, .35 * progress)
            successful += 1
            counts[problem] += 1
            for key, value in values.items():
                weighted[key] += value * len(indices)
            weighted['samples'] += len(indices)
            if (step + 1) % 180 == 0:
                print(f'[R58 epoch {epoch + 1}] {step + 1}/{updates_per_epoch} loss='
                      f'{weighted["loss"] / weighted["samples"]:.6f}', flush=True)
        if weighted['samples'] != sum(sizes.values()):
            raise RuntimeError('Incomplete epoch data coverage')
        optimization = {k: v / weighted['samples'] for k, v in weighted.items() if k != 'samples'}
        optimization.update(lr=lr, samples=int(weighted['samples']), seconds=time.monotonic() - started)
        validation = evaluate(model, loaders['val'], raw['val'], output_dir=directory / 'val_predictions/latest')
        cost = validation['macro']['macro_actual_regret_pct']
        halve, stopped = controller.observe(cost, epoch + 1)
        training = None
        if (epoch + 1) % 5 == 0 or stopped or epoch == 39:
            training = evaluate(model, loaders['train'], raw['train'])
        row = dict(epoch=epoch + 1, successful_updates=successful, train=training,
                   val=validation, optimization=optimization, updates_by_problem=dict(counts))
        history.append(row)
        improved = cost < best_cost
        if improved:
            best_cost = cost
        if halve:
            peak_lr *= .5
        checkpoint = dict(model=model.state_dict(), optimizer=optimizer.state_dict(), scaler=scaler.state_dict(),
            args=config, epoch=epoch, macro=validation['macro'], controller=vars(controller), peak_lr=peak_lr,
            successful_updates=successful, updates_by_problem=dict(counts), rng=capture_rng(),
            history=history, best_cost=best_cost)
        if improved:
            save_checkpoint(directory / 'best.pt', checkpoint)
            save_json(directory / 'best_eval.json', row)
        save_checkpoint(directory / 'last.pt', checkpoint)
        save_json(directory / 'history.json', history)
        print(f'[R58 validation] epoch={epoch + 1} Top1={validation["macro"]["macro_top1"]:.6f} '
              f'actual_regret={cost:.6f}% best={best_cost:.6f}%', flush=True)
        if wandb_run is not None:
            metrics = dict(epoch=epoch + 1, successful_updates=successful,
                **{'optimization/' + k: v for k, v in optimization.items()})
            metrics.update({'val/' + k: v for k, v in validation['macro'].items()})
            if training is not None:
                metrics.update({'train_eval/' + k: v for k, v in training['macro'].items()})
            wandb_run.log(metrics)
        if stopped:
            break
    save_json(directory / 'result.json', dict(epochs_completed=len(history), best_actual_regret_pct=best_cost,
        best=json.loads((directory / 'best_eval.json').read_text()), final=history[-1],
        successful_updates=successful, stop_reason='validation early stop' if stopped else 'maximum epochs',
        test_evaluated=False))
    if wandb_run is not None:
        wandb_run.finish()


def test(args):
    directory = args.root / RUN
    if not (directory / 'result.json').exists():
        raise ValueError('Finish validation selection before test')
    if (args.root / 'test_results.json').exists():
        print('[R58] locked test already evaluated; preserve it', flush=True)
        return
    locked, _ = release_check(args.root)
    source_snapshot(directory)
    checkpoint_path = directory / 'best.pt'
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    result = json.loads((directory / 'result.json').read_text())
    check_test_checkpoint(checkpoint, result, locked, file_hash(args.root / 'scenario_v2/release.json'))
    record = dict(path=str(checkpoint_path), sha256=file_hash(checkpoint_path),
                  epoch=checkpoint['epoch'] + 1, validation=checkpoint['macro'],
                  deployment_lock_sha256=locked['lock_sha256'])
    path = args.root / 'locked_checkpoint.json'
    if path.exists() and json.loads(path.read_text()) != record:
        raise ValueError('Test checkpoint changed after lock')
    save_json(path, record)
    model = make_r45_model(checkpoint['args']['model_params']).to(args.device)
    model.load_state_dict(checkpoint['model'], strict=True)
    loaders, raw = {}, {}
    for problem in PROBLEMS:
        loaders[problem], raw[problem] = make_loader(args.root, problem, 'test', locked['pools'][problem], args.device)
        attach_test_geometry(loaders[problem])
    result = evaluate(model, loaders, raw, output_dir=args.root / 'test_predictions')
    save_json(args.root / 'test_results.json', result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['train', 'test', 'all'], default='all')
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--wandb', action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    configure_torch()
    torch.set_num_threads(4)
    if args.stage in ('all', 'train'):
        train(args)
    if args.stage in ('all', 'test'):
        test(args)


if __name__ == '__main__':
    main()

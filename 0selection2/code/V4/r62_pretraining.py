"""Observed auxiliary-deployment comparisons; the complete selector stays trainable."""

import json
import math
import time
from collections import Counter

import torch
import torch.nn.functional as F

from .multitask_probe import state_hash
from .pairwise_objective import inputs_only
from .r59_experiment import atomic_json, atomic_save
from .r61_pretraining import learning_rate, matched_schedule
from .train import set_seed
from .training_monitor import capture_rng, restore_rng


def pair_objective(output, costs, observed):
    margins = output['pair_margin'].float()
    if (costs.dtype != torch.float64 or observed.dtype != torch.bool or costs.ndim != 2
            or costs.shape != observed.shape or margins.shape != (*costs.shape, costs.shape[1])):
        raise ValueError('Auxiliary FP64 costs, observed cells and complete-pool margins must align')
    if not output['solver_mask'].all():
        raise ValueError('Observed-cell masks must not remove legal candidates from the forward')
    costs, observed = costs.to(margins.device), observed.to(margins.device)
    observed = observed & torch.isfinite(costs)
    i, j = torch.triu_indices(costs.shape[1], costs.shape[1], 1, device=margins.device)
    valid = observed[:, i] & observed[:, j] & (costs[:, i] != costs[:, j])
    target = costs[:, i] < costs[:, j]
    values = margins[:, i, j]
    bce = F.binary_cross_entropy_with_logits(values, target.float(), reduction='none')
    counts = valid.sum(-1)
    loss = ((bce * valid).sum(-1) / counts.clamp_min(1)).mean()
    correct = (((values > 0) == target) & (values != 0) & valid).sum()
    return loss, dict(pairs=int(valid.sum()), correct_pairs=int(correct),
                      comparison_rows=int((counts > 0).sum()), rows=len(costs),
                      observed_cells=int(observed.sum()))


def transfer_state(pretrained, initial):
    if set(pretrained) != set(initial) or not any(k.startswith('score_head.') for k in initial):
        raise ValueError('Transfer requires the unchanged full selector and its original score head')
    return {k: (initial[k] if k.startswith('score_head.') else value).detach().cpu().clone()
            for k, value in pretrained.items()}


def update_batch(model, optimizer, scaler, batch, costs, observed):
    clean = {k: v for k, v in inputs_only(batch).items()
             if k not in ('observed_costs', 'observed_mask', 'source_indices', 'indices', 'winner', 'route')}
    rng = capture_rng()
    for retry in range(20):
        restore_rng(rng)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=next(model.parameters()).device.type, dtype=torch.float16,
                            enabled=scaler.is_enabled()):
            output = model(clean)
        loss, stats = pair_objective(output, costs, observed)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        old_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        if scaler.get_scale() >= old_scale:
            values = dict(loss=float(loss.detach()), grad_norm=float(norm), skipped=retry, **stats)
            if not all(math.isfinite(v) for v in values.values()):
                raise ValueError('Nonfinite R62 auxiliary update')
            return values
        print(f'[AMP retry] same auxiliary batch; lower scale={scaler.get_scale()}', flush=True)
    raise RuntimeError('Repeated nonfinite gradients on the same R62 auxiliary batch')


def train(arm, args, directory, datasets, schedule, initialize, protocol):
    directory.mkdir(parents=True, exist_ok=True)
    result_path = directory / 'result.json'
    if result_path.exists():
        if not args.resume:
            raise ValueError('Preserve completed R62 pretraining; use --resume')
        return json.loads(result_path.read_text())
    model, _ = initialize()
    initial = torch.load(args.root / 'selector_initial.pt', map_location='cpu', weights_only=False)
    if state_hash(initial) != protocol['selector_initial_hash']:
        raise ValueError('Shared scratch selector initialization changed')
    model.load_state_dict(initial, strict=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scaler = torch.amp.GradScaler('cuda', enabled=args.device.startswith('cuda'))
    config = dict(group=arm, source_hashes=protocol['source_hashes'], auxiliary_manifest=protocol['auxiliary_manifest'],
                  schedule_hash=protocol['schedule_hash'], initial_hash=protocol['selector_initial_hash'])
    path = directory / 'args.json'
    if path.exists() and json.loads(path.read_text()) != config:
        raise ValueError('R62 pretraining source/configuration changed')
    atomic_json(path, config)
    successful, retries, pass_index, next_step, previous_wall = 0, 0, 0, 0, 0.
    cursor, totals, all_counts, history, checkpoint, first_update = Counter(), Counter(), Counter(), [], None, None
    path = directory / 'last.pt'
    if path.exists():
        if not args.resume:
            raise ValueError('Resume the unchanged R62 pretraining checkpoint')
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        if checkpoint['args'] != config:
            raise ValueError('R62 pretraining source/data/schedule changed')
        model.load_state_dict(checkpoint['model'], strict=True)
        optimizer.load_state_dict(checkpoint['optimizer'])
        scaler.load_state_dict(checkpoint['scaler'])
        successful, retries = checkpoint['successful_updates'], checkpoint['amp_retries']
        pass_index, next_step = checkpoint['pass_index'], checkpoint['next_step']
        cursor, totals, all_counts = (Counter(checkpoint[k]) for k in ('cursor', 'totals', 'all_counts'))
        history, previous_wall, first_update = checkpoint['history'], checkpoint['wall_seconds'], checkpoint['first_update']
    wandb_run, started = None, time.monotonic()
    try:
        if args.wandb:
            import wandb
            wandb_run = wandb.init(**protocol['wandb'], name=f'R62_{arm}_pretrain', config=config, dir=str(directory))
        restore_rng(checkpoint['rng']) if checkpoint is not None else set_seed(args.seed)
        steps = len(schedule['orders'][0])
        model.train()
        if args.device.startswith('cuda'):
            torch.cuda.reset_peak_memory_stats(args.device)
        for current_pass in range(pass_index, 2):
            order = schedule['orders'][current_pass]
            for step in range(next_step, len(order)):
                problem = order[step]
                offset = cursor[problem] * args.batch_size
                indices = schedule['permutations'][arm][current_pass][problem][offset:offset + args.batch_size]
                batch, costs, observed = datasets[arm][problem].gather(indices, args.device)
                lr = learning_rate(successful, steps)
                for group in optimizer.param_groups:
                    group['lr'] = lr
                if successful == 0 and args.device.startswith('cuda'):
                    torch.cuda.synchronize(args.device)
                update_start = time.monotonic()
                values = update_batch(model, optimizer, scaler, batch, costs, observed)
                if successful == 0:
                    if args.device.startswith('cuda'):
                        torch.cuda.synchronize(args.device)
                    first_update = dict(values, problem=problem, batch=len(indices),
                                        seconds=time.monotonic() - update_start, formal=True)
                    print(f'[R62{arm} first formal update] ' + json.dumps(first_update), flush=True)
                successful += 1
                retries += values['skipped']
                cursor[problem] += 1
                totals['loss_sum'] += values['loss'] * len(indices)
                totals['grad_norm_sum'] += values['grad_norm'] * len(indices)
                for key in ('rows', 'pairs', 'correct_pairs', 'comparison_rows', 'observed_cells'):
                    totals[key] += values[key]
                    all_counts[key] += values[key]
                if successful == 1 or successful % 180 == 0 or step + 1 == len(order):
                    optimization = dict(loss=totals['loss_sum'] / totals['rows'],
                        grad_norm=totals['grad_norm_sum'] / totals['rows'],
                        pair_accuracy=totals['correct_pairs'] / max(1, totals['pairs']),
                        lr=lr, amp_skipped=retries, seconds=previous_wall + time.monotonic() - started,
                        **{k: totals[k] for k in ('rows', 'pairs', 'comparison_rows', 'observed_cells')})
                    if args.device.startswith('cuda'):
                        optimization['peak_memory_gib'] = torch.cuda.max_memory_allocated(args.device) / 2**30
                    history.append(dict(epoch=current_pass+1, pass_id=current_pass+1, step_in_pass=step+1,
                                        successful_updates=successful, optimization=optimization))
                    atomic_save(path, dict(args=config, model=model.state_dict(), optimizer=optimizer.state_dict(),
                        scaler=scaler.state_dict(), successful_updates=successful, amp_retries=retries,
                        pass_index=current_pass, next_step=step+1, cursor=dict(cursor), totals=dict(totals),
                        all_counts=dict(all_counts), history=history, rng=capture_rng(), first_update=first_update,
                        wall_seconds=optimization['seconds']))
                    atomic_json(directory / 'history.json', history)
                    print(f'[R62{arm} pretrain] pass={current_pass+1}/2 update={step+1}/{len(order)} '
                          f'total={successful} loss={optimization["loss"]:.5f} '
                          f'pair_accuracy={optimization["pair_accuracy"]:.5f} lr={lr:.3g}', flush=True)
                    if wandb_run is not None:
                        wandb_run.log(dict(successful_updates=successful, pass_id=current_pass+1,
                                          **{'optimization/' + k: v for k, v in optimization.items()}))
            next_step, cursor, totals = 0, Counter(), Counter()
        if successful != sum(map(len, schedule['orders'])):
            raise ValueError('Both complete auxiliary passes and tails are required')
        final = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        atomic_save(directory / 'model_final.pt', final)
        result = dict(successful_updates=successful, epochs_completed=2, passes_completed=2,
                      wall_seconds=previous_wall + time.monotonic() - started, model_hash=state_hash(final),
                      final=history[-1], first_update=first_update, coverage=dict(all_counts), test_read=False,
                      checkpoint_rule='final update only; no auxiliary validation selection')
        atomic_json(result_path, result)
        return result
    finally:
        if wandb_run is not None:
            wandb_run.finish()

"""Fixed saved-TRAIN successor pretraining, with no selector or cost objective."""

import json
import math
import time
from collections import Counter

import numpy as np
import torch
from torch import nn

from .multitask_probe import state_hash
from .pair_specialist import INPUT_FIELDS
from .r42_experiment import paired_schedule
from .r54_model import SuccessorHead, successor_objective
from .r59_experiment import atomic_json, atomic_save
from .train import set_seed
from .training_monitor import capture_rng, restore_rng


class BehaviorPretrainer(nn.Module):
    def __init__(self, encoder, dim):
        super().__init__()
        self.instance_encoder = encoder
        self.successor_heads = nn.ModuleList([SuccessorHead(dim), SuccessorHead(dim)])

    def forward(self, batch):
        clean = {key: batch[key] for key in INPUT_FIELDS if key in batch}
        nodes, mask = self.instance_encoder(clean)
        logits = torch.stack([head(nodes, mask, clean['node'][..., :2])
                              for head in self.successor_heads], 1)
        return logits, mask[:, 1:]


def matched_schedule(datasets, batch_size=128, seed=2):
    """B traverses twice; A replacement rows match every actual B size in order."""
    if set(datasets) != {'A', 'B'} or set(datasets['A']) != set(datasets['B']):
        raise ValueError('Both arms must contain the same behavior tasks')
    sizes = {p: len(d.nodes) for p, d in datasets['B'].items()}
    base = paired_schedule(sizes, 2, batch_size, seed)
    rng, replay = np.random.default_rng(seed), []
    for permutations in base['permutations']:
        entries = {}
        for problem, indices in permutations.items():
            old = np.asarray(datasets['A'][problem].nodes)
            wanted = np.asarray(datasets['B'][problem].nodes)[indices.numpy()]
            selected = np.empty(len(indices), dtype=np.int64)
            for size in np.unique(wanted):
                bucket = np.flatnonzero(old == size)
                if not len(bucket):
                    raise ValueError(f'No original TRAIN replay rows for {problem} size={size}')
                positions = np.flatnonzero(wanted == size)
                selected[positions] = rng.choice(bucket, len(positions), replace=True)
            entries[problem] = torch.from_numpy(selected)
        replay.append(entries)
    return dict(orders=base['orders'], permutations={'A': replay, 'B': base['permutations']})


def learning_rate(successful, steps_per_pass):
    step = successful + 1
    if step <= steps_per_pass:
        return 1e-4 * step / steps_per_pass
    progress = (step - steps_per_pass) / steps_per_pass
    return 1e-5 + .5 * (1e-4 - 1e-5) * (1. + math.cos(math.pi * progress))


def update_batch(model, optimizer, scaler, batch, successor):
    rng = capture_rng()
    for retry in range(20):
        restore_rng(rng)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=batch['node'].device.type, dtype=torch.float16,
                            enabled=scaler.is_enabled()):
            logits, mask = model(batch)
        loss, parts = successor_objective(logits, successor, mask)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        old_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        if scaler.get_scale() >= old_scale:
            values = dict(loss=float(loss.detach()), grad_norm=float(norm), skipped=retry,
                          **{k: float(v.detach()) for k, v in parts.items()})
            values['successor_accuracy'] = .5 * (values['moel_accuracy'] + values['mtl_accuracy'])
            if not all(math.isfinite(v) for v in values.values()):
                raise ValueError('Nonfinite R61 successor update')
            return values
        print(f'[AMP retry] same successor batch; lower scale={scaler.get_scale()}', flush=True)
    raise RuntimeError('Repeated nonfinite gradients on the same R61 successor batch')


def train(arm, args, directory, datasets, schedule, initialize, protocol):
    directory.mkdir(parents=True, exist_ok=True)
    result_path = directory / 'result.json'
    if result_path.exists():
        if not args.resume:
            raise ValueError('Preserve completed R61 pretraining; use --resume')
        return json.loads(result_path.read_text())
    model = initialize()
    initial = torch.load(args.root / 'pretrain_initial.pt', map_location='cpu', weights_only=False)
    if state_hash(initial) != protocol['pretrain_initial_hash']:
        raise ValueError('Shared pretraining initialization changed')
    model.load_state_dict(initial, strict=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scaler = torch.amp.GradScaler('cuda', enabled=args.device.startswith('cuda'))
    config = dict(group=arm, source_hashes=protocol['source_hashes'],
                  behavior_manifest=protocol['behavior_manifest'], schedule_hash=protocol['schedule_hash'],
                  initial_hash=protocol['pretrain_initial_hash'])
    atomic_json(directory / 'args.json', config)
    successful, retries, pass_index, next_step, previous_wall = 0, 0, 0, 0, 0.
    cursor, totals, history, checkpoint = Counter(), Counter(), [], None
    path = directory / 'last.pt'
    if path.exists():
        if not args.resume:
            raise ValueError('Resume the unchanged R61 pretraining checkpoint')
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        if checkpoint['args'] != config:
            raise ValueError('R61 pretraining source/data/schedule changed')
        model.load_state_dict(checkpoint['model'], strict=True)
        optimizer.load_state_dict(checkpoint['optimizer'])
        scaler.load_state_dict(checkpoint['scaler'])
        successful, retries = checkpoint['successful_updates'], checkpoint['amp_retries']
        pass_index, next_step = checkpoint['pass_index'], checkpoint['next_step']
        cursor, totals = Counter(checkpoint['cursor']), Counter(checkpoint['totals'])
        history, previous_wall = checkpoint['history'], checkpoint['wall_seconds']
    wandb_run = None
    started = time.monotonic()
    try:
        if args.wandb:
            import wandb
            wandb_run = wandb.init(**protocol['wandb'], name=f'R61_{arm}_pretrain',
                                   config=config, dir=str(directory))
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
                batch, targets = datasets[arm][problem].gather(indices, args.device)
                lr = learning_rate(successful, steps)
                for group in optimizer.param_groups:
                    group['lr'] = lr
                values = update_batch(model, optimizer, scaler, batch, targets)
                successful += 1
                retries += values['skipped']
                cursor[problem] += 1
                totals['samples'] += len(indices)
                totals['customers'] += values['customers']
                for key in ('loss', 'moel_loss', 'mtl_loss', 'grad_norm'):
                    totals[key] += values[key] * len(indices)
                for key in ('moel_accuracy', 'mtl_accuracy', 'successor_accuracy'):
                    totals[key] += values[key] * values['customers']
                if successful % 180 == 0 or step + 1 == len(order):
                    optimization = {k: totals[k] / max(1, totals['samples'])
                                    for k in ('loss', 'moel_loss', 'mtl_loss', 'grad_norm')}
                    optimization.update({k: totals[k] / max(1, totals['customers'])
                                         for k in ('moel_accuracy', 'mtl_accuracy', 'successor_accuracy')})
                    optimization.update(lr=lr, amp_skipped=retries, samples=totals['samples'],
                                        seconds=previous_wall + time.monotonic() - started)
                    if args.device.startswith('cuda'):
                        optimization['peak_memory_gib'] = torch.cuda.max_memory_allocated(args.device) / 2**30
                    history.append(dict(epoch=current_pass + 1, pass_id=current_pass + 1,
                                        step_in_pass=step + 1, successful_updates=successful,
                                        optimization=optimization))
                    checkpoint = dict(args=config, model=model.state_dict(), optimizer=optimizer.state_dict(),
                        scaler=scaler.state_dict(), successful_updates=successful, amp_retries=retries,
                        pass_index=current_pass, next_step=step + 1, cursor=dict(cursor), totals=dict(totals),
                        history=history, rng=capture_rng(), wall_seconds=optimization['seconds'])
                    atomic_save(path, checkpoint)
                    atomic_json(directory / 'history.json', history)
                    print(f'[R61{arm} pretrain] pass={current_pass+1}/2 update={step+1}/{len(order)} '
                          f'total={successful} loss={optimization["loss"]:.5f} '
                          f'successor_accuracy={optimization["successor_accuracy"]:.5f} lr={lr:.3g}', flush=True)
                    if wandb_run is not None:
                        wandb_run.log(dict(successful_updates=successful, pass_id=current_pass+1,
                                          **{'optimization/' + k: v for k, v in optimization.items()}))
            next_step, cursor, totals = 0, Counter(), Counter()
        expected = sum(len(order) for order in schedule['orders'])
        if successful != expected:
            raise ValueError('Both complete pretraining passes and all tails are required')
        encoder = {k: v.detach().cpu().clone() for k, v in model.instance_encoder.state_dict().items()}
        atomic_save(directory / 'encoder_final.pt', encoder)
        result = dict(successful_updates=successful, epochs_completed=2, passes_completed=2,
                      query_presentations=2 * sum(len(d.nodes) for d in datasets['B'].values()),
                      wall_seconds=previous_wall + time.monotonic() - started,
                      encoder_hash=state_hash(encoder), final=history[-1], test_read=False,
                      checkpoint_rule='final update only; no validation-route access or selection')
        atomic_json(result_path, result)
        return result
    finally:
        if wandb_run is not None:
            wandb_run.finish()

"""Original R77 selection supervision plus input-only learned coarsening losses."""

import torch
from torch import nn

from .pairwise_objective import inputs_only, predicted_top3, single_objective
from .r42_experiment import symmetric_kl
from .r76_experiment import (PART_KEYS, capture_rank_rng, restore_rank_rng,
                            slice_batch)


class HierarchyObjective(nn.Module):
    def __init__(self, selector, arm):
        super().__init__()
        if arm not in ('A', 'B'):
            raise ValueError('Only the two neural R92 arms use this objective')
        self.selector, self.arm = selector, arm
        self.metrics = {}

    def forward(self, batch, consistency_weight):
        device = batch['ind'].device
        with torch.autocast(device_type=device.type, dtype=torch.float16,
                            enabled=device.type == 'cuda'):
            first = self.selector(inputs_only(batch))
            second = self.selector(inputs_only(batch))
        focus = predicted_top3(first['logits'], second['logits'],
                               batch['pool_ids'], first['solver_mask'])
        loss1, parts1 = single_objective(first, batch, focus)
        loss2, parts2 = single_objective(second, batch, focus)
        kl = symmetric_kl(first['logits'], second['logits'], first['solver_mask'])
        aux = .5 * (first['auxiliary_loss'] + second['auxiliary_loss'])
        loss = .5 * (loss1 + loss2) + consistency_weight * kl + aux
        self.metrics = dict(hierarchy_auxiliary=aux.detach(), **{
            'hierarchy_' + name: .5 * (first['auxiliary_parts'][name]
                                       + second['auxiliary_parts'][name]).detach()
            for name in ('link', 'entropy')})
        parts = {name: .5 * (parts1[name] + parts2[name]) for name in parts1}
        parts.update(loss=loss, kl=kl)
        return loss, torch.stack([parts[name].detach() for name in PART_KEYS[:-1]])


def update_batch(objective, optimizer, scaler, batch, weight, runtime, cap,
                 max_retries=20, *, problem):
    """Per-graph auxiliary means preserve exact logical-batch row weighting."""
    count = len(batch['ind'])
    if runtime.world_size != 1 or count <= 0 or cap <= 0:
        raise ValueError('R92 requires real rows and one assigned GPU per arm')
    rng = capture_rank_rng(runtime.device)
    for retry in range(max_retries):
        restore_rank_rng(rng, runtime.device)
        optimizer.zero_grad(set_to_none=True)
        parts_sum = torch.zeros(len(PART_KEYS) - 1, device=runtime.device, dtype=torch.float64)
        metrics = {name: parts_sum.new_zeros(()) for name in
                   ('hierarchy_auxiliary', 'hierarchy_link', 'hierarchy_entropy')}
        bad = torch.zeros((), device=runtime.device, dtype=torch.int32)
        for start in range(0, count, cap):
            stop = min(count, start + cap)
            loss, parts = objective(slice_batch(batch, start, stop), weight)
            scaler.scale(loss * ((stop - start) / count)).backward()
            bad |= (~torch.isfinite(loss.detach())).to(torch.int32)
            parts_sum.add_(parts.double(), alpha=stop - start)
            for name, value in objective.metrics.items():
                metrics[name].add_(value.double(), alpha=(stop - start) / count)
        scaler.unscale_(optimizer)
        norm = torch.nn.utils.clip_grad_norm_(objective.parameters(), 1.)
        bad |= (~torch.isfinite(norm)).to(torch.int32)
        if bad.item():
            if scaler.is_enabled():
                scaler.update(new_scale=scaler.get_scale() * .5)
            continue
        scaler.step(optimizer)
        scaler.update()
        return torch.cat((parts_sum / count, norm.detach().reshape(1))), retry, metrics
    raise RuntimeError('Nonfinite gradients persisted; no rows skipped or successful update accepted')

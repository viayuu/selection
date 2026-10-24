"""Stateless R108 one-forward CE / canonical conditional density-ratio pair."""

import torch
from torch import nn
import torch.nn.functional as F

from .pairwise_objective import inputs_only
from .r76_experiment import PART_KEYS
from .r108_model import ARMS


class ConditionalPriorObjective(nn.Module):
    """No costs, pair/risk/R-Drop objectives, queues or extra normalizers."""

    def __init__(self, selector, arm):
        super().__init__()
        if arm not in ARMS or getattr(selector, 'r108_arm', arm) != arm:
            raise ValueError('Unknown or mismatched R108 objective arm')
        self.selector, self.arm = selector, arm

    def forward(self, batch, consistency_weight=0.):
        if batch is None:
            zero = next(self.selector.parameters()).sum() * 0.
            return zero, zero.detach().new_zeros(len(PART_KEYS) - 1)
        device = batch['ind'].device
        with torch.autocast(device_type=device.type, dtype=torch.float16,
                            enabled=device.type == 'cuda'):
            output = self.selector(inputs_only(batch))
        mask, y = output['solver_mask'], batch['ind']
        scores = output['utility'].float().masked_fill(~mask, -torch.inf)
        if self.arm == ARMS[0]:
            loss = F.cross_entropy(scores, y)
            ce = loss.detach()
        else:
            # Mask before softplus: illegal NaN/Inf slots must have zero gradient.
            t = output['residual_utility'].float().masked_fill(~mask, 0.)
            pi = output['conditional_prior'].detach().float().masked_fill(~mask, 0.)
            positive = F.softplus(-t.gather(1, y[:, None]).squeeze(1))
            negative = (pi * F.softplus(t)).sum(-1)
            loss = (.5 * positive + .5 * negative).mean()
            with torch.no_grad():
                ce = F.cross_entropy(scores, y)
        parts = loss.detach().new_zeros(len(PART_KEYS) - 1)
        parts[PART_KEYS.index('loss')] = loss.detach()
        parts[PART_KEYS.index('ce')] = ce
        return loss, parts


def make_objective(selector, arm):
    return ConditionalPriorObjective(selector, arm)

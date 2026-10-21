"""R93 single-forward, unweighted native-winner CE on raw legal utilities."""

import torch
from torch import nn
import torch.nn.functional as F

from .pairwise_objective import inputs_only
from .r76_experiment import PART_KEYS


class PlainClassificationObjective(nn.Module):
    """No comparison, risk, consistency, auxiliary loss, or mutable state."""

    def __init__(self, selector):
        super().__init__()
        self.selector = selector

    def forward(self, batch, consistency_weight=0.):
        # The shared engine's legacy weight argument has no effect in R93.
        if batch is None:
            zero = next(self.selector.parameters()).sum() * 0.
            return zero, zero.detach().new_zeros(len(PART_KEYS) - 1)
        device = batch['ind'].device
        with torch.autocast(device_type=device.type, dtype=torch.float16,
                            enabled=device.type == 'cuda'):
            output = self.selector(inputs_only(batch))
        logits = output['utility'].float().masked_fill(~output['solver_mask'], float('-inf'))
        loss = F.cross_entropy(logits, batch['ind'])
        parts = loss.detach().new_zeros(len(PART_KEYS) - 1)
        parts[PART_KEYS.index('loss')] = loss.detach()
        parts[PART_KEYS.index('ce')] = loss.detach()
        return loss, parts


PlainCEObjective = PlainClassificationObjective

"""Stateless, single-forward whole-legal-pool raw-utility CE for R110."""

import torch
from torch import nn
import torch.nn.functional as F

from .pairwise_objective import inputs_only
from .r76_experiment import PART_KEYS


ARMS = ('A_tied_ce', 'B_private_ce')
MODEL_ARM_MAP = dict(zip(ARMS, ('A_tied_late', 'B_private_late')))


class MethodConditionedCEObjective(nn.Module):
    """No priors, class weights, pair/risk losses, KL or objective buffers."""

    def __init__(self, selector, arm):
        super().__init__()
        if arm not in ARMS or getattr(selector, 'r109_arm', MODEL_ARM_MAP[arm]) != MODEL_ARM_MAP[arm]:
            raise ValueError('Unknown or mismatched R110 model/objective arm')
        self.selector, self.arm = selector, arm

    def forward(self, batch, consistency_weight=0.):
        if batch is None:
            zero = next(self.selector.parameters()).sum() * 0.
            return zero, zero.detach().new_zeros(len(PART_KEYS) - 1)
        device = batch['ind'].device
        with torch.autocast(device_type=device.type, dtype=torch.float16,
                            enabled=device.type == 'cuda'):
            output = self.selector(inputs_only(batch))
        utility, mask, y = output['utility'].float(), output['solver_mask'], batch['ind']
        if (utility.ndim != 2 or mask.shape != utility.shape or mask.dtype != torch.bool
                or y.shape != (utility.size(0),) or y.dtype != torch.long
                or not mask.any(-1).all() or (y < 0).any() or (y >= utility.size(1)).any()):
            raise ValueError('R110 CE requires a nonempty legal pool and native local winner indices')
        if not mask.gather(1, y[:, None]).all():
            raise ValueError('R110 native winner must belong to the legal pool')
        # Mask before CE so illegal NaN/Inf slots have exactly zero gradient.
        loss = F.cross_entropy(utility.masked_fill(~mask, -torch.inf), y)
        parts = loss.detach().new_zeros(len(PART_KEYS) - 1)
        parts[PART_KEYS.index('loss')] = loss.detach()
        parts[PART_KEYS.index('ce')] = loss.detach()
        return loss, parts


def make_objective(selector, arm):
    return MethodConditionedCEObjective(selector, arm)

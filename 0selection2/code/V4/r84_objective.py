"""R84 additive focal penalty on the two existing R77 dropout forwards.

Runner API: ``FocalRDropObjective(selector)(batch, consistency_weight)`` returns
``(loss, parts)`` with the original ``r76_experiment.PART_KEYS[:-1]`` order.
``last_focal`` and ``last_focal_extra`` are detached scalar tensors from that
same forward, respectively before and after multiplication by ``focal_weight``.
They describe the current microbatch, not the logical-batch mean. In the local
runner's update function, call ``reset_focal_metrics()`` at the start of EVERY
AMP attempt, ``accumulate_focal_metrics(count)`` after each forward with its
real, non-dummy row count, and ``focal_metrics()`` only after a successful step.
These diagnostics keep the original PART_KEYS vector unchanged, discard failed
attempts and require no third selector forward. They retain no graph. If using
DDP, access these methods on ``objective.module`` and reduce row-weighted sums
and counts across ranks before claiming a global-batch mean.

The production C/D recipe is fixed at focal_weight=0.35, gamma=2: the original
0.35 CE + 0.35 comparison + 0.02 risk + R-Drop remains intact. The configurable
zero weight and zero gamma exist for equivalence checks, not parameter search.
"""

import math

import torch
import torch.nn.functional as F

from .pairwise_objective import inputs_only, predicted_top3, single_objective
from .r42_experiment import symmetric_kl
from .r76_experiment import PART_KEYS, RDropObjective


FOCAL_WEIGHT = .35
FOCAL_GAMMA = 2.
FOCAL_RECIPE = dict(weight=FOCAL_WEIGHT, gamma=FOCAL_GAMMA,
                    lambda_relative_ce=1., alpha=None, detach_modulation=False,
                    dropout_reduction='mean_of_two_independent_forward_losses')


def focal_loss(logits, labels, solver_mask=None, gamma=FOCAL_GAMMA):
    """Mean (1-p_native)**gamma * -log(p_native), computed stably in FP32."""
    if not math.isfinite(gamma) or gamma < 0:
        raise ValueError('Focal gamma must be finite and nonnegative')
    scores = logits.float()
    if solver_mask is not None:
        scores = scores.masked_fill(~solver_mask, float('-inf'))
    nll = F.cross_entropy(scores, labels, reduction='none')
    if gamma == 0:
        return nll.mean()
    # expm1 avoids cancellation at p_true near one; the modulation stays in autograd.
    modulation = (-torch.expm1(-nll)).pow(gamma)
    return (modulation * nll).mean()


class FocalRDropObjective(RDropObjective):
    """Original R77 objective plus a fixed, unbalanced CE+focal extra penalty."""

    def __init__(self, selector, focal_weight=FOCAL_WEIGHT, gamma=FOCAL_GAMMA):
        super().__init__(selector)
        if not math.isfinite(focal_weight) or focal_weight < 0:
            raise ValueError('Focal weight must be finite and nonnegative')
        if not math.isfinite(gamma) or gamma < 0:
            raise ValueError('Focal gamma must be finite and nonnegative')
        self.focal_weight = float(focal_weight)
        self.gamma = float(gamma)
        self.last_focal = None
        self.last_focal_extra = None
        self.reset_focal_metrics()

    def reset_focal_metrics(self):
        """Start one optimizer/AMP attempt; the runner calls again on retries."""
        self._focal_sum = None
        self._focal_rows = 0

    def accumulate_focal_metrics(self, count):
        """Record the last microbatch using real rows; idle/dummy rows count zero."""
        if not isinstance(count, int) or count < 0:
            raise ValueError('Focal metric row count must be a nonnegative integer')
        if not count:
            return
        if self.last_focal is None:
            raise RuntimeError('Record focal metrics only after an objective forward')
        if self._focal_sum is None:
            self._focal_sum = self.last_focal.new_zeros((), dtype=torch.float64)
        self._focal_sum.add_(self.last_focal.double(), alpha=count)
        self._focal_rows += count

    def focal_metrics(self):
        """Detached, real-row-weighted means for the current successful attempt."""
        if self._focal_rows:
            focal = self._focal_sum / self._focal_rows
        else:
            focal = next(self.selector.parameters()).detach().new_zeros((), dtype=torch.float64)
        return dict(focal=focal, focal_extra=focal * self.focal_weight)

    def forward(self, batch, consistency_weight):
        if batch is None:
            # The attention-only arm has no score_head.
            zero = next(self.selector.parameters()).sum() * 0.
            self.last_focal = zero.detach()
            self.last_focal_extra = zero.detach()
            return zero, zero.detach().new_zeros(len(PART_KEYS) - 1)
        device = batch['ind'].device
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == 'cuda'):
            first = self.selector(inputs_only(batch))
            second = self.selector(inputs_only(batch))
        focus = predicted_top3(first['logits'], second['logits'], batch['pool_ids'], first['solver_mask'])
        loss1, parts1 = single_objective(first, batch, focus)
        loss2, parts2 = single_objective(second, batch, focus)
        kl = symmetric_kl(first['logits'], second['logits'], first['solver_mask'])
        loss = .5 * (loss1 + loss2) + consistency_weight * kl
        focal = .5 * (focal_loss(first['logits'], batch['ind'], first['solver_mask'], self.gamma)
                      + focal_loss(second['logits'], batch['ind'], second['solver_mask'], self.gamma))
        self.last_focal = focal.detach()
        self.last_focal_extra = self.last_focal * self.focal_weight
        if self.focal_weight:
            loss = loss + self.focal_weight * focal
        parts = {key: .5 * (parts1[key] + parts2[key]) for key in parts1}
        parts.update(loss=loss, kl=kl)
        return loss, torch.stack([parts[key].detach() for key in PART_KEYS[:-1]])

"""R91 controlled utility probability links; original maximin inference is intact."""

import torch
from torch import nn
import torch.nn.functional as F

from .pairwise_objective import comparison_loss, inputs_only, predicted_top3
from .r42_experiment import symmetric_kl
from .r76_experiment import PART_KEYS


ARMS = ('A_utility_ce', 'B_utility_probabilities')


def probability_links(arm):
    if arm not in ARMS:
        raise ValueError('Unknown R91 probability-link arm')
    return dict(ce='utility', risk='utility' if arm == ARMS[1] else 'maximin',
                rdrop='utility' if arm == ARMS[1] else 'maximin',
                comparison='original utility differences',
                comparison_focus='original two-forward averaged maximin Top3',
                inference='original maximin')


def utility_logits(output):
    return output['utility'].float().masked_fill(~output['solver_mask'], float('-inf'))


def single_objective(output, batch, focus, arm):
    links = probability_links(arm)
    utility = utility_logits(output)
    ce = F.cross_entropy(utility, batch['ind'])
    cmp, parts = comparison_loss(output['pair_margin'], batch, focus)
    risk_scores = utility if links['risk'] == 'utility' else output['logits'].float()
    risk = (risk_scores.softmax(-1) * batch['cost_regret']).sum(-1).clamp_max(1.).mean() / .01
    loss = .35 * ce + .35 * cmp + .02 * risk
    return loss, dict(ce=ce, cmp=cmp, risk=risk, **parts)


class ProbabilityLinkObjective(nn.Module):
    """Two original dropout forwards, no new output head or objective state."""

    def __init__(self, selector, arm):
        super().__init__()
        probability_links(arm)
        self.selector, self.arm = selector, arm

    def forward(self, batch, consistency_weight):
        if batch is None:
            zero = next(self.selector.parameters()).sum() * 0.
            return zero, zero.detach().new_zeros(len(PART_KEYS) - 1)
        device = batch['ind'].device
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == 'cuda'):
            first = self.selector(inputs_only(batch))
            second = self.selector(inputs_only(batch))
        # The average of raw utilities can change Top3 even though each individual
        # utility and maximin ranking agrees; preserve the original focus exactly.
        focus = predicted_top3(first['logits'], second['logits'],
                               batch['pool_ids'], first['solver_mask'])
        loss1, parts1 = single_objective(first, batch, focus, self.arm)
        loss2, parts2 = single_objective(second, batch, focus, self.arm)
        if self.arm == ARMS[1]:
            kl = symmetric_kl(utility_logits(first), utility_logits(second), first['solver_mask'])
        else:
            kl = symmetric_kl(first['logits'], second['logits'], first['solver_mask'])
        loss = .5 * (loss1 + loss2) + consistency_weight * kl
        parts = {key: .5 * (parts1[key] + parts2[key]) for key in parts1}
        parts.update(loss=loss, kl=kl)
        return loss, torch.stack([parts[key].detach() for key in PART_KEYS[:-1]])

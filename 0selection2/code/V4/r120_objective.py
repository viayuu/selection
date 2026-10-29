"""Direction/tie-aware complete-information supervision with TRAIN-only units."""

import math

import torch
from torch import nn
import torch.nn.functional as F


ADVANTAGE_WEIGHT = .25
RISK_WEIGHT = .10


def safe_model_inputs(batch):
    return {key: batch[key] for key in ('problem', 'inputs', 'pool_ids', 'solver_mask') if key in batch}


def native_targets(labels, scale):
    objective = labels['native_objective']
    observed = labels['observed_mask'].bool()
    if objective.dtype != torch.float64:
        raise ValueError('Native objectives must retain FP64 precision through target construction')
    direction = labels.get('objective_direction', labels.get('direction'))
    if direction not in ('min', 'max'):
        raise ValueError('Native objective direction must be min or max')
    if not math.isfinite(float(scale)) or float(scale) <= 0:
        raise ValueError('A positive task scale fitted on TRAIN is required')
    signed = objective if direction == 'min' else -objective
    oracle = signed.masked_fill(~observed, float('inf')).amin(-1, keepdim=True)
    # Missing values can be NaN, so mask before all arithmetic and reductions.
    clean = signed.masked_fill(~observed, 0.)
    mean = clean.sum(-1, keepdim=True) / observed.sum(-1, keepdim=True).clamp_min(1)
    advantage = ((clean - mean) / scale).masked_fill(~observed, 0.).float()
    excess = ((clean - oracle) / scale).masked_fill(~observed, 0.).float()
    best = observed & (signed == oracle)
    return best, advantage, excess


def single_loss(output, labels, scale):
    mask = output['solver_mask'].bool()
    observed = labels['observed_mask'].bool()
    if not torch.equal(mask, observed):
        raise ValueError('Formal complete-pool training requires labels for every declared legal deployment')
    best, target, excess = native_targets(labels, scale)
    supplied_best = labels.get('all_best_mask')
    if supplied_best is not None and not torch.equal(best, supplied_best.bool()):
        raise ValueError('Saved exact best mask differs from original native objectives')
    scores = output['logits'].float().masked_fill(~mask, float('-inf'))
    native = labels.get('native_winner_index')
    if native is not None:
        ce = F.cross_entropy(scores, native.long())
    else:
        # All tied methods are valid decisions; no artificial equal-share target.
        ce = (torch.logsumexp(scores, -1) - torch.logsumexp(scores.masked_fill(~best, float('-inf')), -1)).mean()
    error = F.smooth_l1_loss(output['advantage'].float(), target, reduction='none')
    advantage = (error.masked_fill(~mask, 0.).sum(-1) / mask.sum(-1).clamp_min(1)).mean()
    risk = (scores.softmax(-1) * excess).sum(-1).mean()
    loss = ce + ADVANTAGE_WEIGHT * advantage + RISK_WEIGHT * risk
    prediction = scores.detach().argmax(-1, keepdim=True)
    any_best = best.gather(-1, prediction).float().mean()
    native_accuracy = ((prediction[:, 0] == native).float().mean()
                       if native is not None else any_best)
    return loss, dict(loss=loss.detach(), ce=ce.detach(), advantage=advantage.detach(), risk=risk.detach(),
                      train_any_best_accuracy=any_best.detach(),
                      train_mixed_accuracy=native_accuracy.detach(),
                      train_hard_scaled_excess=excess.gather(-1, prediction).mean().detach())


def loss_for_outputs(first, second, labels, consistency_weight=0., zero_oracle_scale=1.):
    """Compatibility helper; R120 has one forward and no dropout consistency term."""
    if consistency_weight:
        raise ValueError('R120 does not add R-Drop to its fixed multidomain objective')
    if second is not None:
        raise ValueError('R120 uses one forward per microbatch')
    return single_loss(first, labels, zero_oracle_scale)


class MultidomainObjective(nn.Module):
    def __init__(self, selector, zero_oracle_scales):
        super().__init__()
        self.selector = selector
        self.scales = dict(zero_oracle_scales)

    def forward(self, batch, consistency_weight=0.):
        if consistency_weight:
            raise ValueError('R120 has no R-Drop term')
        output = self.selector(safe_model_inputs(batch))
        return single_loss(output, batch['labels'], self.scales[batch['problem']])

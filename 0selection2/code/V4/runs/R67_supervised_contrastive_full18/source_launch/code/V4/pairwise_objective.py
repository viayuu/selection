"""Full-pool comparisons: original-precision labels and per-instance focus weights."""

import numpy as np
import torch
import torch.nn.functional as F

from .pairwise_selector import rank_by_solver_ids


LABEL_KEYS = ('costs', 'ind', 'performance_target', 'cmp_target', 'cmp_valid', 'cmp_true3', 'cost_regret')


def inputs_only(batch):
    return {key: value for key, value in batch.items() if key not in LABEL_KEYS}


def comparison_targets(costs, mask=None):
    if costs.dtype != np.float64:
        raise ValueError('Comparison targets require original FP64 costs, not upcast cached FP32')
    mask = np.ones_like(costs, dtype=bool) if mask is None else mask
    if not mask.any(-1).all() or not np.isfinite(costs[mask]).all():
        raise ValueError('Every instance requires finite costs and a nonempty pool')
    k = costs.shape[-1]
    i, j = np.triu_indices(k, 1)
    ranked = np.sort(np.where(mask, costs, np.inf), axis=-1)
    boundary = ranked[np.arange(len(costs)), np.minimum(mask.sum(-1), 3) - 1]
    true3 = mask & (costs <= boundary[:, None])
    valid = mask[:, i] & mask[:, j] & (costs[:, i] != costs[:, j])
    oracle = np.where(mask, costs, np.inf).min(-1)
    if (oracle <= 0).any():
        raise ValueError('R43 relative risk requires positive solver costs')
    regret = np.where(mask, (costs - oracle[:, None]) / oracle[:, None], 0.)
    return dict(cmp_target=torch.from_numpy((costs[:, i] < costs[:, j]).astype(np.float32)),
                cmp_valid=torch.from_numpy(valid),
                cmp_true3=torch.from_numpy(true3[:, i] & true3[:, j]),
                cost_regret=torch.from_numpy(regret.astype(np.float32)))


def predicted_top3(first, second, ids, mask):
    ranks = rank_by_solver_ids((first.detach() + second.detach()) * .5, ids)
    focus = torch.zeros_like(mask).scatter_(-1, ranks[:, :min(3, mask.size(-1))], True)
    return focus & mask


def comparison_loss(margins, batch, pred3):
    k = margins.size(-1)
    i, j = torch.triu_indices(k, k, 1, device=margins.device)
    values = F.binary_cross_entropy_with_logits(margins[:, i, j].float(), batch['cmp_target'], reduction='none')
    sets = (batch['cmp_true3'], pred3[:, i] & pred3[:, j], torch.ones_like(batch['cmp_valid']))
    means, nonempty = [], []
    for group in sets:
        valid = group & batch['cmp_valid']
        count = valid.sum(-1)
        means.append((values * valid).sum(-1) / count.clamp_min(1))
        nonempty.append(count > 0)
    means, nonempty = torch.stack(means, -1), torch.stack(nonempty, -1)
    weights = means.new_tensor([.50, .25, .25]) * nonempty
    loss = ((means * weights).sum(-1) / weights.sum(-1).clamp_min(1e-12)).mean()
    parts = {name: means[:, col].mean() for col, name in enumerate(('true3', 'pred3', 'all_pairs'))}
    return loss, parts


def single_objective(output, batch, pred3):
    scores = output['logits'].float()
    ce = F.cross_entropy(scores, batch['ind'])
    cmp, parts = comparison_loss(output['pair_margin'], batch, pred3)
    risk = (scores.softmax(-1) * batch['cost_regret']).sum(-1).clamp_max(1.).mean() / .01
    loss = .35 * ce + .35 * cmp + .02 * risk
    return loss, dict(ce=ce, cmp=cmp, risk=risk, **parts)

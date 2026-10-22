"""R95 importance weighting of the unchanged R77/R88B compound objective.

The proxy is the exact per-row deterministic eval gradient norm for the shared
final affine weight AND bias, conditional on its input hidden states. It is not
a full-parameter gradient or a stochastic/dropout gradient. No network forward,
feature archive, sampling correction, or new loss is used to compute it.
"""

import torch
import torch.nn.functional as F
from torch import nn

from .pairwise_objective import inputs_only as _original_inputs_only, predicted_top3
from .pairwise_selector import maximin_logits
from .r76_experiment import PART_KEYS


SAMPLING_KEYS = ('sampling_weight', 'sampling_row')
_COMPARISON_KEYS = ('true3', 'pred3', 'all_pairs')


def inputs_only(batch):
    """Keep sampling metadata and all original supervision out of forward."""
    return {key: value for key, value in _original_inputs_only(batch).items()
            if key not in SAMPLING_KEYS}


def _single_rows(output, batch, focus):
    scores = output['logits'].float()
    margins = output['pair_margin']
    i, j = torch.triu_indices(margins.size(-1), margins.size(-1), 1, device=margins.device)
    values = F.binary_cross_entropy_with_logits(
        margins[:, i, j].float(), batch['cmp_target'], reduction='none')
    groups = (batch['cmp_true3'], focus[:, i] & focus[:, j], torch.ones_like(batch['cmp_valid']))
    means, nonempty = [], []
    for group in groups:
        valid = group & batch['cmp_valid']
        count = valid.sum(-1)
        means.append((values * valid).sum(-1) / count.clamp_min(1))
        nonempty.append(count > 0)
    means, nonempty = torch.stack(means, -1), torch.stack(nonempty, -1)
    focus_weights = means.new_tensor([.50, .25, .25]) * nonempty
    comparison = (means * focus_weights).sum(-1) / focus_weights.sum(-1).clamp_min(1e-12)
    # Keep risk unscaled here so scalar reductions retain the original mean/.01.
    risk = (scores.softmax(-1) * batch['cost_regret']).sum(-1).clamp_max(1.)
    return dict(ce=F.cross_entropy(scores, batch['ind'], reduction='none'),
                cmp=comparison, risk=risk,
                **{name: means[:, col] for col, name in enumerate(_COMPARISON_KEYS)})


def _paired_rows(first, second, batch):
    mask = first['solver_mask']
    focus = predicted_top3(first['logits'], second['logits'], batch['pool_ids'], mask)
    first_rows = _single_rows(first, batch, focus)
    if first is second:
        # Deterministic evaluation reuses one output; R-Drop is exactly zero.
        return first_rows, first_rows, first_rows['ce'].new_zeros(first_rows['ce'].shape)
    second_rows = _single_rows(second, batch, focus)
    log_p = F.log_softmax(first['logits'].float().masked_fill(~mask, -1e9), -1)
    log_q = F.log_softmax(second['logits'].float().masked_fill(~mask, -1e9), -1)
    kl = (.5 * (log_p.exp() - log_q.exp()) * (log_p - log_q)).masked_fill(~mask, 0).sum(-1)
    return first_rows, second_rows, kl


def _compound(parts):
    return .35 * parts['ce'] + .35 * parts['cmp'] + .02 * parts['risk']


def per_row_objective(first, second, batch, consistency_weight):
    """Return unweighted row losses and all ``PART_KEYS[:-1]`` row diagnostics.

    The two dropout passes share the original detached predicted-top3 focus. Labels
    come from the frozen raw-supervision fields, including native ``ind``; costs are
    never re-argmined. Sampling metadata is deliberately ignored by this pure loss.
    Pass the same output twice for deterministic eval with explicitly zero R-Drop.
    """
    first_rows, second_rows, kl = _paired_rows(first, second, batch)
    first_parts = dict(first_rows, risk=first_rows['risk'] / .01)
    second_parts = dict(second_rows, risk=second_rows['risk'] / .01)
    total = .5 * (_compound(first_parts) + _compound(second_parts)) + consistency_weight * kl
    parts = {key: .5 * (first_parts[key] + second_parts[key]) for key in first_parts}
    parts.update(loss=total, kl=kl)
    return total, {key: parts[key] for key in PART_KEYS[:-1]}


class WeightedRDropObjective(nn.Module):
    """Original two-forward loss, using mean(weight * row), never sum weights.

    ``sampling_weight[B]`` is the already computed inverse-importance multiplier,
    not the sampling probability. Missing weights mean ones. The module adds no
    parameters/buffers; the original selector and its stochastic forward are intact.
    """

    def __init__(self, selector):
        super().__init__()
        self.selector = selector

    def forward(self, batch, consistency_weight):
        if batch is None:
            zero = next(self.selector.score_head.parameters()).sum() * 0.
            return zero, zero.detach().new_zeros(len(PART_KEYS) - 1)
        device = batch['ind'].device
        clean = inputs_only(batch)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == 'cuda'):
            first = self.selector(clean)
            second = self.selector(clean)
        first_rows, second_rows, row_kl = _paired_rows(first, second, batch)
        weights = torch.as_tensor(batch.get('sampling_weight', torch.ones_like(row_kl)),
                                  device=row_kl.device, dtype=row_kl.dtype).detach()
        if weights.shape != row_kl.shape or not torch.isfinite(weights).all() or (weights < 0).any():
            raise ValueError('sampling_weight must be finite nonnegative inverse weights with shape [B]')

        def reduce(rows):
            parts = {key: (value * weights).mean() for key, value in rows.items()}
            parts['risk'] = parts['risk'] / .01
            return parts

        parts1, parts2 = reduce(first_rows), reduce(second_rows)
        kl = (row_kl * weights).mean()
        loss = .5 * (_compound(parts1) + _compound(parts2)) + consistency_weight * kl
        parts = {key: .5 * (parts1[key] + parts2[key]) for key in parts1}
        parts.update(loss=loss, kl=kl)
        return loss, torch.stack([parts[key].detach() for key in PART_KEYS[:-1]])


def final_affine_proxy(output, batch, last_hidden):
    """Detached nonnegative ``[B]`` norms of the complete deterministic loss.

    ``last_hidden[B,K,D]`` is the input to ``selector.score_head[-1]``, captured
    during an existing eval forward. Rebuild the SAME masked score differences and
    maximin logits from a detached utility leaf, then differentiate
    ``.35*CE + .35*comparison + .02*risk``. Predicted focus is detached as in training;
    eval R-Drop is zero. Inverse sampling weights never enter this base-loss proxy.

    For each row, dW = sum_m(dL/du_m * hidden_m) and db = sum_m(dL/du_m).
    The norm is taken AFTER those sums, preserving all component/solver gradient
    cancellations. Only this scalar norm is returned; hidden states are not stored.
    Works under an outer ``torch.no_grad()`` and never touches model ``.grad``.
    """
    utility = output['utility']
    if utility.ndim != 2 or last_hidden.ndim != 3 or last_hidden.shape[:2] != utility.shape:
        raise ValueError('Expected utility [B,K] and final affine input last_hidden [B,K,D]')
    mask = output['solver_mask'].detach()
    if mask.shape != utility.shape or not mask.any(-1).all():
        raise ValueError('Every utility row requires a matching nonempty solver mask')
    with torch.enable_grad():
        utility = utility.detach().float().requires_grad_(True)
        margins = utility[:, :, None] - utility[:, None, :]
        margins = margins.masked_fill(~(mask[:, :, None] & mask[:, None, :]), 0.)
        rebuilt = dict(utility=utility, pair_margin=margins, solver_mask=mask,
                       logits=maximin_logits(margins, mask))
        rows, _ = per_row_objective(rebuilt, rebuilt, batch, 0.)
        gradient, = torch.autograd.grad(rows.sum(), utility)
    hidden = last_hidden.detach().to(device=gradient.device, dtype=gradient.dtype)
    grad_weight = torch.bmm(gradient[:, None, :], hidden).squeeze(1)
    grad_bias = gradient.sum(-1, keepdim=True)
    return torch.linalg.vector_norm(torch.cat((grad_weight, grad_bias), -1), dim=-1).detach()

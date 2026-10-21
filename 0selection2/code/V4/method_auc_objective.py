"""R81B: method-conditional AUC with full-logical-batch utility gradient replay."""

import math

import torch
from torch import nn
import torch.nn.functional as F

from .pairwise_objective import inputs_only, predicted_top3, single_objective
from .pairwise_selector import maximin_logits
from .r42_experiment import symmetric_kl
from .r76_experiment import capture_rank_rng, restore_rank_rng, slice_batch


AUC_POS_KEY = 'r81_auc_pos'
AUC_NEG_KEY = 'r81_auc_neg'
AUC_LABEL_KEYS = (AUC_POS_KEY, AUC_NEG_KEY)


def auc_inputs_only(batch):
    return inputs_only({key: value for key, value in batch.items() if key not in AUC_LABEL_KEYS})


def _bool_tensor(value, shape, device, name):
    value = torch.as_tensor(value, device=device)
    if value.dtype != torch.bool or tuple(value.shape) != tuple(shape):
        raise ValueError(f'{name} must be bool with shape {tuple(shape)}')
    return value


@torch.no_grad()
def method_auc_targets(costs, mask=None):
    costs = torch.as_tensor(costs)
    if costs.dtype != torch.float64 or costs.ndim != 2 or not all(costs.shape):
        raise ValueError('AUC targets require original native FP64 costs [N,K], not upcast FP32')
    mask = (torch.ones_like(costs, dtype=torch.bool) if mask is None else
            _bool_tensor(mask, costs.shape, costs.device, 'mask'))
    if not mask.any(-1).all() or not torch.isfinite(costs[mask]).all():
        raise ValueError('Every row needs a nonempty legal pool with finite native costs')
    minimum = costs.masked_fill(~mask, float('inf')).min(-1, keepdim=True).values
    minima = mask & (costs == minimum)
    # Tied minima are excluded for themselves; other strict losers remain negatives.
    return {AUC_POS_KEY: minima & (minima.sum(-1, keepdim=True) == 1),
            AUC_NEG_KEY: mask & (costs > minimum)}


def train_eligible_methods(pos, neg):
    pos = torch.as_tensor(pos)
    if pos.ndim != 2:
        raise ValueError('TRAIN targets must be [N,K]')
    pos = _bool_tensor(pos, pos.shape, pos.device, 'pos')
    neg = _bool_tensor(neg, pos.shape, pos.device, 'neg')
    if (pos & neg).any():
        raise ValueError('Positive and negative sets must be disjoint')
    return pos.any(0) & neg.any(0)


def method_competitive_scores(utility, mask):
    """Competitive log-odds-like scores, not calibrated probabilities."""
    if utility.ndim != 2 or not all(utility.shape) or not utility.is_floating_point():
        raise ValueError('Utility must be floating point [B,K]')
    mask = _bool_tensor(mask, utility.shape, utility.device, 'mask')
    if not mask.any(-1).all():
        raise ValueError('Every row must retain a legal method')
    u = utility if utility.dtype == torch.float64 else utility.float()
    u = u.masked_fill(~mask, 0.)
    k = u.shape[1]
    opponents = mask[:, None, :] & ~torch.eye(k, dtype=torch.bool, device=u.device)[None]
    has_opponent = mask & opponents.any(-1)
    other = u[:, None, :].expand(-1, k, -1).masked_fill(~opponents, float('-inf'))
    # Mask BEFORE the reduction: masking an all-inf logsumexp afterwards still
    # backpropagates NaN. Singleton/padded scores are finite, neutral zeros.
    other = other.masked_fill(~has_opponent[:, :, None], 0.)
    scores = u - torch.logsumexp(other, -1)
    return scores.masked_fill(~has_opponent, 0.)


def _auc_masks(utility, mask, pos, neg, train_eligible):
    pos = _bool_tensor(pos, utility.shape, utility.device, 'pos')
    neg = _bool_tensor(neg, utility.shape, utility.device, 'neg')
    eligible = _bool_tensor(train_eligible, (utility.shape[1],), utility.device, 'train_eligible')
    if ((pos & neg) | ((pos | neg) & ~mask)).any():
        raise ValueError('AUC sets must be disjoint and contain only legal candidates')
    return pos, neg, eligible


def method_auc_loss(utility, mask, pos, neg, train_eligible):
    scores = method_competitive_scores(utility, mask)
    mask = _bool_tensor(mask, utility.shape, utility.device, 'mask')
    pos, neg, eligible = _auc_masks(utility, mask, pos, neg, train_eligible)
    loss = scores.sum() * 0.
    count = int(eligible.sum().item())
    if not count:
        return loss
    for method in eligible.nonzero(as_tuple=True)[0].tolist():
        positive, negative = scores[pos[:, method], method], scores[neg[:, method], method]
        if positive.numel() and negative.numel():
            loss = loss + F.softplus(negative[:, None] - positive[None, :]).mean()
    return loss / count


def _shared_method_axis(batch, shape):
    ids = batch['pool_ids']
    if ids.ndim == 1 and ids.shape == (shape[1],):
        return
    if (ids.shape == shape and torch.equal(ids, ids[:1].expand_as(ids))):
        return
    raise ValueError('A task batch requires one shared method column axis, aligned to TRAIN eligibility')


def _utility_output(utility, mask):
    margins = utility[:, :, None] - utility[:, None, :]
    margins = margins.masked_fill(~(mask[:, :, None] & mask[:, None, :]), 0.)
    return dict(logits=maximin_logits(margins, mask), pair_margin=margins, solver_mask=mask,
                utility=utility)


class CachedAUCObjective(nn.Module):
    def __init__(self, selector, auc_weight=.35, ramp_updates=3726):
        super().__init__()
        if not math.isfinite(auc_weight) or auc_weight < 0 or ramp_updates <= 0:
            raise ValueError('AUC weight must be finite/nonnegative and ramp_updates positive')
        self.selector = selector
        self.auc_weight = auc_weight
        self.ramp_updates = ramp_updates

    def coefficient(self, successful_update):
        if successful_update < 0 or int(successful_update) != successful_update:
            raise ValueError('successful_update must be a nonnegative integer target update index')
        return self.auc_weight * min(1., successful_update / self.ramp_updates)

    def views(self, batch):
        device = batch['ind'].device
        clean = auc_inputs_only(batch)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == 'cuda'):
            first, second = self.selector(clean), self.selector(clean)
        if any('utility' not in output for output in (first, second)):
            raise ValueError('R81B needs ScoreDifferenceSelector utility; ExplicitPairSelector is unchanged')
        return first, second

    def base(self, first, second, batch, consistency_weight):
        focus = predicted_top3(first['logits'], second['logits'], batch['pool_ids'], first['solver_mask'])
        loss1, parts1 = single_objective(first, batch, focus)
        loss2, parts2 = single_objective(second, batch, focus)
        kl = symmetric_kl(first['logits'], second['logits'], first['solver_mask'])
        loss = .5 * (loss1 + loss2) + consistency_weight * kl
        parts = {key: .5 * (parts1[key] + parts2[key]) for key in parts1}
        return loss, dict(parts, kl=kl, base_loss=loss)

    def full_loss(self, u1, u2, batch, *, consistency_weight, successful_update,
                  train_eligible, pos=None, neg=None):
        if u1.shape != u2.shape or u1.ndim != 2 or u1.shape[0] != len(batch['ind']):
            raise ValueError('Both utility views must cover the same complete logical batch [B,K]')
        _shared_method_axis(batch, u1.shape)
        mask = (torch.ones_like(u1, dtype=torch.bool) if batch.get('solver_mask') is None else
                _bool_tensor(batch['solver_mask'], u1.shape, u1.device, 'solver_mask'))
        pos = batch[AUC_POS_KEY] if pos is None else pos
        neg = batch[AUC_NEG_KEY] if neg is None else neg
        base, parts = self.base(_utility_output(u1, mask), _utility_output(u2, mask),
                                batch, consistency_weight)
        auc = .5 * (method_auc_loss(u1, mask, pos, neg, train_eligible)
                    + method_auc_loss(u2, mask, pos, neg, train_eligible))
        weight = self.coefficient(successful_update)
        auxiliary = weight * auc
        loss = base + auxiliary
        return loss, dict(parts, auc_loss=auc, auxiliary_loss=auxiliary,
                          auc_weight=loss.new_tensor(weight), loss=loss)


def utility_update(objective, optimizer, scaler, batch, micro_cap, *, consistency_weight,
                   successful_update, train_eligible, pos=None, neg=None, check_replay=False,
                   max_retries=20, deadline_check=lambda: None):
    """One single-device logical update; no cached trainable features survive it."""
    n, device = len(batch['ind']), batch['ind'].device
    if n <= 0 or micro_cap <= 0 or int(micro_cap) != micro_cap or max_retries <= 0:
        raise ValueError('Require a nonempty batch, positive integer micro cap and positive retries')
    micro_cap = int(micro_cap)
    # Resolve labels and axis errors before any model forward, including no_grad.
    pos = batch[AUC_POS_KEY] if pos is None else pos
    neg = batch[AUC_NEG_KEY] if neg is None else neg
    pos = torch.as_tensor(pos, device=device)
    if pos.ndim != 2 or pos.shape[0] != n:
        raise ValueError('AUC targets must cover the entire logical batch')
    _shared_method_axis(batch, pos.shape)
    if batch['pool_ids'].ndim == 2:
        # slice_batch intentionally leaves pool_ids static; identical rows share one axis.
        batch = dict(batch, pool_ids=batch['pool_ids'][0])
    mask = (torch.ones_like(pos, dtype=torch.bool) if batch.get('solver_mask') is None else
            _bool_tensor(batch['solver_mask'], pos.shape, device, 'solver_mask'))
    if not mask.any(-1).all():
        raise ValueError('Every row must retain a legal method')
    pos, neg, eligible = _auc_masks(pos, mask, pos, neg, train_eligible)
    objective.coefficient(successful_update)
    initial_rng = capture_rank_rng(device)
    chunks = [(start, min(start + micro_cap, n)) for start in range(0, n, micro_cap)]
    for retry in range(max_retries):
        deadline_check()
        restore_rank_rng(initial_rng, device)
        optimizer.zero_grad(set_to_none=True)
        states, first_u, second_u = [], [], []
        with torch.no_grad():
            for start, stop in chunks:
                deadline_check()
                states.append(capture_rank_rng(device))
                first, second = objective.views(slice_batch(batch, start, stop))
                for output in (first, second):
                    if (output['utility'].shape != mask[start:stop].shape
                            or not torch.equal(output['solver_mask'], mask[start:stop])):
                        raise ValueError('Selector utility/mask must match the supervised candidate axis')
                first_u.append(first['utility'].detach().float())
                second_u.append(second['utility'].detach().float())
        final_rng = capture_rank_rng(device)
        try:
            u1 = torch.cat(first_u).detach().requires_grad_()
            u2 = torch.cat(second_u).detach().requires_grad_()
            loss, parts = objective.full_loss(u1, u2, batch,
                consistency_weight=consistency_weight, successful_update=successful_update,
                train_eligible=eligible, pos=pos, neg=neg)
            # These derivatives ALREADY include the global means, view mean and lambda.
            du1, du2 = torch.autograd.grad(loss, (u1, u2))
            bad = ~torch.isfinite(loss.detach())
            bad |= ~torch.isfinite(du1).all() | ~torch.isfinite(du2).all()
            for index, (start, stop) in enumerate(chunks):
                deadline_check()
                restore_rank_rng(states[index], device)
                first, second = objective.views(slice_batch(batch, start, stop))
                if check_replay:
                    for actual, expected in ((first['utility'], first_u[index]),
                                             (second['utility'], second_u[index])):
                        if not torch.allclose(actual.float(), expected, rtol=0., atol=0., equal_nan=True):
                            raise RuntimeError('Exact utility/RNG replay failed')
                scaled_du = scaler.scale(torch.stack((du1[start:stop], du2[start:stop])))
                torch.autograd.backward((first['utility'], second['utility']), tuple(scaled_du.unbind(0)))
        finally:
            restore_rank_rng(final_rng, device)
        scaler.unscale_(optimizer)
        norm = torch.nn.utils.clip_grad_norm_(objective.parameters(), 1.)
        bad |= ~torch.isfinite(norm)
        if bad.item():
            if scaler.is_enabled():
                scaler.update(new_scale=scaler.get_scale() * .5)
            continue
        scaler.step(optimizer)
        scaler.update()
        metrics = {key: float(value.detach()) for key, value in parts.items()}
        metrics['grad_norm'] = float(norm.detach())
        return metrics, retry
    raise RuntimeError('Nonfinite retries exhausted; no batch was skipped or optimizer stepped on overflow')

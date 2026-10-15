"""R78 round006: full-logical-batch profile supervision with RNG replay."""

import torch
from torch import nn
import torch.nn.functional as F

from .pairwise_objective import inputs_only, predicted_top3, single_objective
from .pairwise_selector import maximin_logits
from .r42_experiment import symmetric_kl
from .r76_experiment import capture_rank_rng, restore_rank_rng, slice_batch


PROFILE_KEY = 'r78_train_profile'
PROFILE_WEIGHT = .10


def cost_profiles(costs, mask=None):
    if costs.dtype != torch.float64 or costs.ndim != 2:
        raise ValueError('Construct profiles from original FP64 native TRAIN costs')
    mask = torch.ones_like(costs, dtype=torch.bool) if mask is None else mask.bool()
    if (mask.shape != costs.shape or not mask.any(1).all()
            or not torch.equal(mask, mask[:1].expand_as(mask))):
        raise ValueError('A task batch must share exactly one fixed legal candidate pool')
    valid = costs[:, mask[0]]
    if not torch.isfinite(valid).all() or not (valid > 0).all():
        raise ValueError('Native valid costs must be finite and positive')
    u = (valid / valid.min(1, keepdim=True).values).log()
    return u - u.mean(1, keepdim=True)


def profile_order(q):
    if q.dtype != torch.float64 or q.ndim != 2 or len(q) < 2:
        raise ValueError('FP64 profiles require at least two instances')
    n = len(q)
    # cdist materializes N*N distances, not N*N*K or N*N*embedding_dim.
    distances = torch.cdist(q, q, p=1) / q.shape[1]
    off_diag = ~torch.eye(n, dtype=torch.bool, device=q.device)
    distances = distances[off_diag].reshape(n, n - 1)
    columns = torch.arange(n, device=q.device).expand(n, n)[off_diag].reshape(n, n - 1)
    sorted_d, order = distances.sort(dim=1, descending=True, stable=True)
    columns = columns.gather(1, order)
    ends = torch.ones_like(sorted_d, dtype=torch.bool)
    ends[:, :-1] = sorted_d[:, :-1] != sorted_d[:, 1:]
    positions = torch.arange(n - 1, device=q.device).expand_as(columns)
    next_end = torch.where(ends, positions, n).flip(1).cummin(1).values.flip(1)
    return columns, next_end


def profile_loss(features, ordering):
    e = F.normalize(features.float(), p=2, dim=1, eps=1e-12)
    columns, group_end = ordering
    similarities = (e @ e.T).gather(1, columns)
    denominators = similarities.logcumsumexp(1).gather(1, group_end)
    return (denominators - similarities).mean()


def full_profile_forward(selector, batch):
    inputs = inputs_only({k: v for k, v in batch.items() if k != PROFILE_KEY})
    features, mask = selector.encode_pairs(inputs)
    # The head MUST execute even in the no-grad cache pass: its dropout uses RNG.
    utility = selector.score_head(features).squeeze(-1).float()
    margins = utility[:, :, None] - utility[:, None, :]
    margins = margins.masked_fill(~(mask[:, :, None] & mask[:, None, :]), 0.)
    return dict(logits=maximin_logits(margins, mask), pair_margin=margins,
                solver_mask=mask, profile_h=features[:, 0, :selector.params['embedding_dim']])


class ProfileObjective(nn.Module):
    def __init__(self, selector):
        super().__init__()
        self.selector = selector

    def views(self, batch):
        device = batch['ind'].device
        with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == 'cuda'):
            return full_profile_forward(self.selector, batch), full_profile_forward(self.selector, batch)

    def base(self, first, second, batch):
        focus = predicted_top3(first['logits'], second['logits'], batch['pool_ids'], first['solver_mask'])
        l1, p1 = single_objective(first, batch, focus)
        l2, p2 = single_objective(second, batch, focus)
        kl = symmetric_kl(first['logits'], second['logits'], first['solver_mask'])
        loss = .5 * (l1 + l2) + .35 * kl
        parts = {key: .5 * (p1[key].detach() + p2[key].detach()) for key in ('ce', 'cmp', 'risk')}
        return loss, dict(parts, base_loss=loss.detach(), kl=kl.detach())


def cached_update(objective, optimizer, scaler, batch, micro_cap, *, check_replay=False,
                  max_retries=20, deadline_check=lambda: None):
    """One globally clipped optimizer update; cached gradients include both pair roles."""
    n, device = len(batch['ind']), batch['ind'].device
    if not 0 < micro_cap <= n:
        raise ValueError('Physical cap must be in [1, logical batch size]')
    ordering = profile_order(batch[PROFILE_KEY])
    initial_rng = capture_rank_rng(device)
    chunks = [(start, min(start + micro_cap, n)) for start in range(0, n, micro_cap)]
    for retry in range(max_retries):
        deadline_check()
        restore_rank_rng(initial_rng, device)
        optimizer.zero_grad(set_to_none=True)
        states, first_h, second_h = [], [], []
        with torch.no_grad():
            for start, stop in chunks:
                deadline_check()
                states.append(capture_rank_rng(device))
                first, second = objective.views(slice_batch(batch, start, stop))
                first_h.append(first['profile_h'].float())
                second_h.append(second['profile_h'].float())
        final_rng = capture_rank_rng(device)
        h1 = torch.cat(first_h).detach().requires_grad_()
        h2 = torch.cat(second_h).detach().requires_grad_()
        aux = PROFILE_WEIGHT * .5 * (profile_loss(h1, ordering) + profile_loss(h2, ordering))
        # autograd includes keys as well as anchors; these derivatives are unscaled.
        g1, g2 = torch.autograd.grad(aux, (h1, h2))
        totals = {}
        bad = ~torch.isfinite(aux.detach())
        for index, (start, stop) in enumerate(chunks):
            deadline_check()
            restore_rank_rng(states[index], device)
            micro = slice_batch(batch, start, stop)
            first, second = objective.views(micro)
            if check_replay:
                for actual, expected in ((first['profile_h'], first_h[index]),
                                         (second['profile_h'], second_h[index])):
                    if not torch.allclose(actual.float(), expected, rtol=0., atol=0., equal_nan=True):
                        raise RuntimeError('Exact feature/RNG replay failed; no approximate substitute allowed')
            base, parts = objective.base(first, second, micro)
            # g already includes .10, the two-view mean and the GLOBAL pair mean.
            injection = ((first['profile_h'].float() * g1[start:stop]).sum()
                         + (second['profile_h'].float() * g2[start:stop]).sum())
            combined = base * ((stop - start) / n) + injection
            scaler.scale(combined).backward()
            bad |= ~torch.isfinite(combined.detach())
            for key, value in parts.items():
                totals[key] = totals.get(key, 0.) + value.double() * ((stop - start) / n)
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
        totals.update(profile_loss=aux.detach().double() / PROFILE_WEIGHT,
                      auxiliary_loss=aux.detach().double(), grad_norm=norm.detach().double())
        totals['loss'] = totals['base_loss'] + totals['auxiliary_loss']
        return {key: float(value) for key, value in totals.items()}, retry
    raise RuntimeError('Nonfinite retries exhausted without skipping a batch or stepping on overflow')

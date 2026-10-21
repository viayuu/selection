"""Full-logical-batch conditional relative-regret ranking for R82B."""

import torch
import torch.nn.functional as F

from .method_auc_objective import (
    CachedAUCObjective, _bool_tensor, _shared_method_axis, _utility_output,
    auc_inputs_only, method_competitive_scores,
)
from .r76_experiment import capture_rank_rng, restore_rank_rng, slice_batch


CMR_GROUP_KEY = 'r82_context_group'
CMR_REGRET_KEY = 'r82_relative_regret'
CMR_LABEL_KEYS = (CMR_GROUP_KEY, CMR_REGRET_KEY)


def cmr_inputs_only(batch):
    return auc_inputs_only({key: value for key, value in batch.items() if key not in CMR_LABEL_KEYS})


def _targets(utility, mask, regrets, groups):
    regrets = torch.as_tensor(regrets, device=utility.device)
    groups = torch.as_tensor(groups, device=utility.device)
    if regrets.dtype != torch.float64 or regrets.shape != utility.shape or regrets.requires_grad:
        raise ValueError('CMR requires original-FP64-derived, detached regrets [B,K]')
    if groups.dtype != torch.int64 or groups.shape != (utility.shape[0],) or (groups < 0).any():
        raise ValueError('CMR TRAIN group IDs must be nonnegative int64 [B]')
    if not torch.isfinite(regrets[mask]).all() or (regrets[mask] < 0).any():
        raise ValueError('Legal relative-regret labels must be finite and nonnegative')
    return regrets.masked_fill(~mask, 0.), groups


def context_matched_loss(utility, mask, regrets, groups):
    """Mean pairs within context, contexts within method, then active methods."""
    scores = method_competitive_scores(utility, mask)
    mask = _bool_tensor(mask, utility.shape, utility.device, 'mask')
    regrets, groups = _targets(utility, mask, regrets, groups)
    unique, inverse = torch.unique(groups, sorted=True, return_inverse=True)
    left, right = torch.triu_indices(len(groups), len(groups), 1, device=utility.device)
    matching = inverse[left] == inverse[right]
    left, right = left[matching], right[matching]
    difference = regrets[right] - regrets[left]
    valid = mask[left] & mask[right] & (difference != 0.)
    weight = (difference.abs() / .01).clamp_max(1.).to(scores.dtype)
    signed_score = difference.sign().to(scores.dtype) * (scores[left] - scores[right])
    pair_loss = (weight * F.softplus(-signed_score)).masked_fill(~valid, 0.)
    index = inverse[left, None].expand_as(pair_loss)
    shape = (len(unique), utility.shape[1])
    sums = scores.new_zeros(shape).scatter_add(0, index, pair_loss)
    counts = torch.zeros(shape, device=scores.device, dtype=torch.int64).scatter_add(0, index, valid.long())
    active_groups = counts > 0
    groups_per_method = active_groups.sum(0)
    active_methods = groups_per_method > 0
    group_means = sums / counts.clamp_min(1)
    method_means = group_means.sum(0) / groups_per_method.clamp_min(1)
    loss = method_means.sum() / active_methods.sum().clamp_min(1) + scores.sum() * 0.
    active_pairs = valid.any(-1).long()
    covered = torch.zeros(len(groups), dtype=torch.int64, device=scores.device)
    covered.scatter_add_(0, left, active_pairs)
    covered.scatter_add_(0, right, active_pairs)
    stats = dict(cmr_effective_methods=active_methods.sum(), cmr_effective_groups=active_groups.sum(),
        cmr_pair_count=counts.sum(), cmr_matched_instance_pairs=loss.new_tensor(len(left)),
        cmr_row_coverage=(covered > 0).float().mean())
    return loss, stats


class CachedCMRObjective(CachedAUCObjective):
    def __init__(self, selector, cmr_weight=.10, ramp_updates=3726):
        super().__init__(selector, auc_weight=cmr_weight, ramp_updates=ramp_updates)

    def views(self, batch):
        return super().views({key: value for key, value in batch.items() if key not in CMR_LABEL_KEYS})

    def full_loss(self, u1, u2, batch, *, consistency_weight, successful_update, groups=None, regrets=None):
        if u1.shape != u2.shape or u1.ndim != 2 or u1.shape[0] != len(batch['ind']):
            raise ValueError('Both utility views must cover the entire logical batch')
        _shared_method_axis(batch, u1.shape)
        mask = (torch.ones_like(u1, dtype=torch.bool) if batch.get('solver_mask') is None else
                _bool_tensor(batch['solver_mask'], u1.shape, u1.device, 'solver_mask'))
        groups = batch[CMR_GROUP_KEY] if groups is None else groups
        regrets = batch[CMR_REGRET_KEY] if regrets is None else regrets
        base, parts = self.base(_utility_output(u1, mask), _utility_output(u2, mask), batch, consistency_weight)
        first, stats = context_matched_loss(u1, mask, regrets, groups)
        second, _ = context_matched_loss(u2, mask, regrets, groups)
        cmr = .5 * (first + second)
        weight = self.coefficient(successful_update)
        auxiliary = weight * cmr
        return base + auxiliary, dict(parts, **stats, cmr_loss=cmr, auxiliary_loss=auxiliary,
            cmr_weight=cmr.new_tensor(weight), loss=base + auxiliary)


def utility_update(objective, optimizer, scaler, batch, micro_cap, *, consistency_weight,
                   successful_update, groups=None, regrets=None, check_replay=False,
                   max_retries=20, deadline_check=lambda: None):
    """Full-batch utility derivatives with replay; cached references never persist."""
    n, device = len(batch['ind']), batch['ind'].device
    if n <= 0 or micro_cap <= 0 or int(micro_cap) != micro_cap or max_retries <= 0:
        raise ValueError('Require nonempty batch, positive integer micro cap and positive retries')
    micro_cap = int(micro_cap)
    groups = batch[CMR_GROUP_KEY] if groups is None else groups
    regrets = batch[CMR_REGRET_KEY] if regrets is None else regrets
    regrets = torch.as_tensor(regrets, device=device)
    if regrets.ndim != 2 or regrets.shape[0] != n:
        raise ValueError('Regrets must cover the whole logical batch')
    _shared_method_axis(batch, regrets.shape)
    if batch['pool_ids'].ndim == 2:
        batch = dict(batch, pool_ids=batch['pool_ids'][0])
    mask = (torch.ones_like(regrets, dtype=torch.bool) if batch.get('solver_mask') is None else
            _bool_tensor(batch['solver_mask'], regrets.shape, device, 'solver_mask'))
    if not mask.any(-1).all():
        raise ValueError('Every row must retain at least one legal method')
    regrets, groups = _targets(regrets, mask, regrets, groups)
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
                        raise ValueError('Utility/mask do not match the original candidate axis')
                first_u.append(first['utility'].detach().float())
                second_u.append(second['utility'].detach().float())
        final_rng = capture_rank_rng(device)
        try:
            u1 = torch.cat(first_u).detach().requires_grad_()
            u2 = torch.cat(second_u).detach().requires_grad_()
            loss, parts = objective.full_loss(u1, u2, batch, consistency_weight=consistency_weight,
                successful_update=successful_update, groups=groups, regrets=regrets)
            du1, du2 = torch.autograd.grad(loss, (u1, u2))
            bad = ~torch.isfinite(loss.detach()) | ~torch.isfinite(du1).all() | ~torch.isfinite(du2).all()
            for index, (start, stop) in enumerate(chunks):
                deadline_check()
                restore_rank_rng(states[index], device)
                first, second = objective.views(slice_batch(batch, start, stop))
                if check_replay:
                    for actual, expected in ((first['utility'], first_u[index]), (second['utility'], second_u[index])):
                        if not torch.allclose(actual.float(), expected, rtol=0., atol=0., equal_nan=True):
                            raise RuntimeError('Exact utility/RNG replay failed')
                scaled = scaler.scale(torch.stack((du1[start:stop], du2[start:stop])))
                torch.autograd.backward((first['utility'], second['utility']), tuple(scaled.unbind(0)))
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
        return dict({key: float(value.detach()) for key, value in parts.items()}, grad_norm=float(norm.detach())), retry
    raise RuntimeError('Nonfinite retries exhausted; no batch was skipped or stepped on overflow')

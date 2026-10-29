"""Full-pool hard-decision proxies and TRAIN-only constraint feedback.

The selector still performs one ordinary forward. These functions consume
labels only in supervision, never as query inputs. The explicit controller
commit belongs after a successful complete logical-batch optimizer update.
"""

from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F

from .pairwise_selector import rank_by_solver_ids


LOG_TWO = math.log(2.)
QUERY_INPUTS = ('node', 'matrix', 'node_mask', 'n', 'node_geom', 'kind',
                'problem_id', 'coord_dist', 'cbits', 'problem_desc', 'd_in',
                'pool_ids', 'solver_mask', 'mask')


def hard_action_terms(utility, costs, winner, legal_mask, pool_ids, fixed_target):
    """Per-query hard values and differentiable upper bounds, no averaging yet."""
    if (utility.ndim != 2 or costs.dtype != torch.float64 or costs.shape != utility.shape
            or legal_mask.shape != utility.shape or legal_mask.dtype != torch.bool
            or winner.shape != utility.shape[:1] or winner.dtype != torch.long
            or fixed_target.shape != winner.shape or fixed_target.dtype != torch.bool):
        raise ValueError('Require aligned utility, original FP64 costs, native winners and masks')
    if (not legal_mask.any(-1).all() or (winner < 0).any()
            or (winner >= utility.size(-1)).any()
            or not legal_mask.gather(1, winner[:, None]).all()):
        raise ValueError('Every query requires a legal native winner')
    if (not torch.isfinite(costs[legal_mask]).all()
            or not torch.isfinite(utility[legal_mask]).all()):
        raise ValueError('Legal costs and scores must be finite')
    if costs.requires_grad or fixed_target.requires_grad:
        raise ValueError('TRAIN supervision must be detached')
    oracle = costs.masked_fill(~legal_mask, torch.inf).min(-1).values
    if (oracle <= 0).any() or not torch.equal(costs.gather(1, winner[:, None])[:, 0], oracle):
        raise ValueError('Native winner must be an exact minimum with a positive routing cost')
    regret = ((costs - oracle[:, None]) / oracle[:, None]).masked_fill(~legal_mask, 0.)
    scores = utility.float().masked_fill(~legal_mask, -torch.inf)
    ce = F.cross_entropy(scores, winner, reduction='none') / LOG_TWO
    ids = pool_ids.to(device=scores.device)
    if ids.ndim == 1:
        ids = ids[None].expand_as(scores)
    if ids.shape != scores.shape:
        raise ValueError('Every score needs its fixed deployment ID for stable hard ties')
    prediction = rank_by_solver_ids(scores.detach(), ids)[:, 0]
    wrong = prediction != winner
    hard_regret = regret.gather(1, prediction[:, None])[:, 0]
    hard_miss = (wrong & fixed_target).double()
    k = scores.size(1)
    if k == 1:
        risk_proxy = scores[:, 0] * 0.
    else:
        # Top-two avoids an O(batch * candidates^2) opponent tensor. Illegal
        # NaNs were masked before any score arithmetic or gradient operation.
        top, order = scores.topk(2, dim=-1)
        maximum_other = top[:, :1].expand_as(scores).clone()
        maximum_other.scatter_(1, order[:, :1], top[:, 1:2])
        singleton = legal_mask.sum(-1) == 1
        margin = torch.where(legal_mask, scores - maximum_other, torch.zeros_like(scores))
        margin = margin.masked_fill(singleton[:, None], 0.)
        coefficients = F.softplus(margin) / LOG_TWO
        risk_proxy = (coefficients * regret.to(scores.dtype)).masked_fill(~legal_mask, 0.).sum(-1)
    return dict(classification_proxy=ce, regret_proxy=risk_proxy,
        missed_proxy=ce * fixed_target.to(ce.dtype), prediction=prediction,
        hard_error=wrong.double(), hard_regret=hard_regret,
        hard_fixed_miss=hard_miss, fixed_population=fixed_target.double())


class HardDecisionController(nn.Module):
    """A deterministic heuristic dual update, not a theoretical game solver."""

    def __init__(self, problems, *, train_regret_budget, train_fixed_miss_budget,
                 train_fixed_population, budget_source_split='train',
                 risk_unit=.01, dual_lr=.05, dual_cap=20., ema_decay=.9):
        super().__init__()
        if budget_source_split != 'train':
            raise ValueError('Constraint budgets must be fitted on TRAIN, never validation or TEST')
        if (not problems or len(set(problems)) != len(problems)
                or min(train_regret_budget, train_fixed_miss_budget) < 0
                or not 0 < train_fixed_population <= 1 or risk_unit <= 0
                or not 0 <= ema_decay < 1 or dual_lr <= 0 or dual_cap <= 0):
            raise ValueError('Invalid task scope, TRAIN budgets or controller settings')
        values = (train_regret_budget, train_fixed_miss_budget, train_fixed_population,
                  risk_unit, dual_lr, dual_cap, ema_decay)
        if not all(math.isfinite(value) for value in values):
            raise ValueError('Budgets and controller settings must be finite')
        self.problems = tuple(problems)
        self.dual_lr, self.dual_cap, self.ema_decay = dual_lr, dual_cap, ema_decay
        self.register_buffer('budget', torch.tensor([train_regret_budget, train_fixed_miss_budget], dtype=torch.float64))
        self.register_buffer('unit', torch.tensor([risk_unit, train_fixed_population], dtype=torch.float64))
        self.register_buffer('multipliers', torch.ones(2, dtype=torch.float64))
        self.register_buffer('ema_violation', torch.zeros(2, dtype=torch.float64))
        self.register_buffer('task_sum', torch.zeros(len(problems), 2, dtype=torch.float64))
        self.register_buffer('task_count', torch.zeros(len(problems), dtype=torch.long))
        self.register_buffer('committed_updates', torch.zeros((), dtype=torch.long))
        self.register_buffer('completed_cycles', torch.zeros((), dtype=torch.long))

    def primal_loss(self, terms):
        """Side-effect-free, correctly aggregatable over the complete logical batch."""
        error = terms['classification_proxy'].mean()
        proxy = torch.stack((terms['regret_proxy'].mean(), terms['missed_proxy'].mean()))
        normalized = (proxy - self.budget.to(proxy)) / self.unit.to(proxy)
        loss = error + (self.multipliers.detach().to(proxy) * normalized).sum()
        return loss, dict(classification=error.detach(), regret_proxy=proxy[0].detach(),
                          fixed_miss_proxy=proxy[1].detach())

    @torch.no_grad()
    def commit(self, problem, *, hard_regret, hard_fixed_miss, successful_update):
        """Caller supplies means after all microbatches/ranks succeeded exactly once."""
        if not self.training:
            raise RuntimeError('Evaluation must not update TRAIN constraint feedback')
        if successful_update != int(self.committed_updates) + 1:
            raise ValueError('Commit only the next successful logical update; never replay an AMP retry')
        if problem not in self.problems:
            raise ValueError('Unknown problem in the fixed full-pool controller')
        point = self.budget.new_tensor([float(hard_regret), float(hard_fixed_miss)])
        if not torch.isfinite(point).all() or (point < 0).any() or point[1] > 1:
            raise ValueError('Hard TRAIN feedback must be finite nonnegative means')
        task = self.problems.index(problem)
        self.task_sum[task].add_(point)
        self.task_count[task].add_(1)
        self.committed_updates.add_(1)
        if not (self.task_count > 0).all():
            return False
        hard = (self.task_sum / self.task_count[:, None]).mean(0)
        violation = (hard - self.budget) / self.unit
        if int(self.completed_cycles) == 0:
            self.ema_violation.copy_(violation)
        else:
            self.ema_violation.mul_(self.ema_decay).add_(violation, alpha=1. - self.ema_decay)
        self.multipliers.add_(self.ema_violation, alpha=self.dual_lr).clamp_(0., self.dual_cap)
        self.task_sum.zero_()
        self.task_count.zero_()
        self.completed_cycles.add_(1)
        return True


class HardDecisionObjective(nn.Module):
    """One final-selector forward; controller commits are external to autograd."""

    def __init__(self, selector, controller):
        super().__init__()
        self.selector = selector
        self.controller = controller.to(next(selector.parameters()).device)
        self.last_hard = None

    def forward(self, batch, consistency_weight=0.):
        if consistency_weight != 0.:
            raise ValueError('R122 does not add R-Drop or a second classification objective')
        inputs = {key: batch[key] for key in QUERY_INPUTS if key in batch}
        device = batch['ind'].device
        with torch.autocast(device_type=device.type, dtype=torch.float16,
                            enabled=device.type == 'cuda'):
            output = self.selector(inputs)
        terms = hard_action_terms(output['utility'], batch['costs'], batch['ind'],
            output['solver_mask'], batch['pool_ids'], batch['fixed_target'])
        loss, values = self.controller.primal_loss(terms)
        self.last_hard = {key: terms[key].detach().mean() for key in
                          ('hard_error', 'hard_regret', 'hard_fixed_miss', 'fixed_population')}
        zeros = loss.detach().new_zeros(())
        # The shared runner's legacy columns are kept explicit: CE is measured
        # in bits, risk is the unscaled proxy; unused comparison/KL are zero.
        parts = torch.stack((loss.detach(), values['classification'], zeros,
                             values['regret_proxy'], zeros, zeros, zeros, zeros))
        return loss, parts

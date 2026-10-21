"""R85 TRAIN objectives and transactional, single-GPU logical-batch state.

Runner contract (all arms):
    batch = objective.begin_attempt(logical_batch, consistency_weight)
    for micro in slices_of(batch):
        loss, parts = objective(micro, consistency_weight)
        (loss * len(micro['ind']) / len(batch['ind'])).backward()
        objective.accumulate(len(micro['ind']))
    extras = objective.finish_attempt(success=optimizer_really_stepped)

Call begin_attempt again after restoring RNG for an AMP retry. Statistics and
subset RNG advance ONLY in finish_attempt(True). Store training_state_dict()
under a separate checkpoint key; it contains no selector weights. Loading the
selector for inference does not require this state. No DDP reduction is supplied.

A/B replace only 0.35 CE with 0.35 OVR on raw utilities. ``parts['ce']`` remains
native maximin CE. OVR uses log(K)/log(2) times the valid-method mean BCE, then
the row mean. EQL statistics are absolute derivatives of this 0.35 OVR term,
before AMP scaling or clipping, averaged over two dropout views and all real
logical rows. Weights are detached and frozen for the complete attempt.

The main thread locked all four recipes after its primary-source handoff.
EQL's data-dependent class weights need not estimate natural posteriors;
the matched, unweighted OVR arm is essential. C uses conditional nonwinner KL,
not full winner-probability regression. This module does not authorize a run.
The EQL adaptation keeps natural weights until a problem/method has observed
positive gradient; it is not an exact replication of the detection objective.
"""

import math

import torch
import torch.nn.functional as F

from code.unified_selector.registry import GLOBAL_SOLVERS, PROBLEMS
from .pairwise_objective import inputs_only, predicted_top3, single_objective
from .r42_experiment import symmetric_kl
from .r76_experiment import PART_KEYS, RDropObjective


CLASSIFICATION_WEIGHT = .35
EQL_GAMMA, EQL_MU, EQL_ALPHA = 12., .8, 4.
EQL_RECIPE = dict(gamma=EQL_GAMMA, mu=EQL_MU, alpha=EQL_ALPHA, initial_ratio=100.,
                  epsilon=1e-10, scope='problem_id_global_solver_id',
                  support_guard='natural weights until first positive gradient',
                  gradient='weighted_035_ovr_unscaled_unclipped_two_views_logical_mean',
                  protocol_status='locked_after_main_primary_source_handoff',
                  primary_source='https://github.com/tztztztztz/eqlv2/blob/'
                  '907fad905b0cac6372f7ec4035f48ef3d0801098/mmdet/models/losses/eqlv2.py')


def _ovr_terms(utility, labels, mask, positive_weight=None, negative_weight=None):
    if utility.ndim != 2 or mask.shape != utility.shape or labels.shape != utility.shape[:1]:
        raise ValueError('OVR requires utilities/mask [B,K] and native labels [B]')
    mask = mask.bool()
    scores = utility.float().masked_fill(~mask, 0.)
    target = F.one_hot(labels.long(), scores.shape[-1]).to(scores.dtype)
    positive_weight = 1. if positive_weight is None else positive_weight
    negative_weight = 1. if negative_weight is None else negative_weight
    weights = target * positive_weight + (1. - target) * negative_weight
    k = mask.sum(-1, keepdim=True).clamp_min(1).to(scores.dtype)
    scale = k.log() / math.log(2.) / k
    values = F.binary_cross_entropy_with_logits(scores, target, reduction='none')
    per_row = (values * weights).masked_fill(~mask, 0.).sum(-1) * scale.squeeze(-1)
    with torch.no_grad():
        absolute_grad = ((scores.sigmoid() - target).abs() * weights * scale).masked_fill(~mask, 0.)
        positive = (absolute_grad * target).mean(0) * CLASSIFICATION_WEIGHT
        negative = (absolute_grad * (1. - target)).mean(0) * CLASSIFICATION_WEIGHT
    return per_row.mean(), positive, negative


def ovr_classification_loss(utility, labels, solver_mask, positive_weight=None, negative_weight=None):
    """Natural OVR or fixed-weight OVR; labels are the original native winner."""
    return _ovr_terms(utility, labels, solver_mask, positive_weight, negative_weight)[0]


def cost_profile_kl(logits, relative_regret, solver_mask, labels, temperature=.01):
    """Conditional nonwinner KL, excluding the unchanged native winner.

    q is softmax(-FP64-derived regret/0.01) over legal nonwinner methods. p is
    final maximin softmax on that same set. Rows with fewer than two legal
    nonwinners contribute zero to the full logical-row mean. The winner logit
    has exactly zero direct derivative through this term.
    """
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError('Cost-profile temperature must be finite and positive')
    if logits.shape != relative_regret.shape or logits.shape != solver_mask.shape:
        raise ValueError('Cost-profile labels and mask must match logits [B,K]')
    if labels.shape != logits.shape[:1]:
        raise ValueError('Conditional KL requires original native winner labels [B]')
    nonwinner = solver_mask.bool() & ~F.one_hot(labels.long(), logits.shape[-1]).bool()
    active = nonwinner.sum(-1) >= 2
    dummy = torch.arange(logits.shape[-1], device=logits.device)[None] == 0
    safe_mask = torch.where(active[:, None], nonwinner, dummy)
    # A finite dummy distribution keeps unsupported rows out of log_softmax NaNs.
    target_logits = torch.where(active[:, None], -relative_regret.detach().float() / temperature,
                                torch.zeros_like(logits, dtype=torch.float32))
    target_logits = target_logits.masked_fill(~safe_mask, float('-inf'))
    logq = F.log_softmax(target_logits, -1).masked_fill(~safe_mask, 0.)
    q = logq.exp().masked_fill(~safe_mask, 0.)
    scores = torch.where(active[:, None], logits.float(), torch.zeros_like(logits, dtype=torch.float32))
    logp = F.log_softmax(scores.masked_fill(~safe_mask, float('-inf')), -1).masked_fill(~safe_mask, 0.)
    return (q * (logq - logp)).sum(-1).masked_fill(~active, 0.).mean()


def _shared_problem_id(value):
    if isinstance(value, str):
        if value not in PROBLEMS:
            raise ValueError('Unknown TRAIN problem')
        return PROBLEMS.index(value)
    values = torch.as_tensor(value).detach().cpu().reshape(-1)
    if values.numel() == 0 or not torch.equal(values, values[:1].expand_as(values)):
        raise ValueError('A logical batch must have one shared problem_id')
    problem_id = int(values[0])
    if float(values[0]) != problem_id or not 0 <= problem_id < len(PROBLEMS):
        raise ValueError('problem_id is outside the stable registry')
    return problem_id


class _AttemptObjective(RDropObjective):
    """No optimizer/AMP side effects: only the runner can accept an attempt."""

    state_kind = 'r85_base'

    def __init__(self, selector):
        super().__init__(selector)
        self.successful_steps = 0
        self._active = False
        self._last_metrics = None
        self._accepted_metrics = {}

    def begin_attempt(self, batch, weight):
        if batch is None or len(batch['ind']) == 0:
            raise ValueError('R85 accepts real nonempty single-GPU logical batches')
        if batch.get('split', 'train') != 'train':
            raise ValueError('Objective state may only consume TRAIN labels')
        if not math.isfinite(weight) or weight < 0:
            raise ValueError('R-Drop weight must be finite and nonnegative')
        self._problem_id = _shared_problem_id(batch['problem_id'])
        self._logical_rows = len(batch['ind'])
        ids = torch.as_tensor(batch['pool_ids']).detach().cpu().long()
        if ids.ndim == 2:
            if not torch.equal(ids, ids[:1].expand_as(ids)):
                raise ValueError('A logical batch must retain one shared method axis')
            ids = ids[0]
        if (ids.ndim != 1 or ids.numel() == 0 or ids.unique().numel() != ids.numel()
                or (ids < 0).any() or (ids >= len(GLOBAL_SOLVERS)).any()):
            raise ValueError('Use unique, stable global pool_ids, including reld_mt=16')
        self._pool_ids = ids
        mask = batch.get('solver_mask')
        if mask is None:
            mask = torch.ones(self._logical_rows, len(ids), device=batch['ind'].device, dtype=torch.bool)
            batch = dict(batch, solver_mask=mask)
        if mask.shape != (self._logical_rows, len(ids)) or not mask.any(-1).all():
            raise ValueError('Each real row must retain a nonempty legal pool')
        labels = batch['ind']
        if (labels < 0).any() or (labels >= len(ids)).any() or not mask.gather(1, labels[:, None]).all():
            raise ValueError('Every native winner must be a legal method')
        self._active = True
        self._weight = float(weight)
        self._seen_rows = 0
        self._sums = {}
        self._last_metrics = None
        self._last_consumed = True
        self._begin_state(batch)
        return batch

    def _begin_state(self, batch):
        pass

    def _record_state(self, count):
        pass

    def _commit_state(self):
        pass

    def accumulate(self, real_rows):
        if not self._active or self._last_metrics is None:
            raise RuntimeError('accumulate must follow a forward in an active attempt')
        if isinstance(real_rows, bool) or not isinstance(real_rows, int) or real_rows < 0:
            raise ValueError('Real row count must be a nonnegative integer')
        if not real_rows:
            return
        if self._last_consumed or real_rows != self._last_rows:
            raise RuntimeError('Each complete real microbatch must be accumulated exactly once')
        if self._seen_rows + real_rows > self._logical_rows:
            raise RuntimeError('Microbatches exceeded the logical batch row count')
        for name, value in self._last_metrics.items():
            if name not in self._sums:
                self._sums[name] = value.new_zeros((), dtype=torch.float64)
            self._sums[name].add_(value.double(), alpha=real_rows)
        self._record_state(real_rows)
        self._seen_rows += real_rows
        self._last_consumed = True

    def finish_attempt(self, success=True):
        if not self._active:
            raise RuntimeError('No active logical-batch attempt')
        if success:
            if self._seen_rows != self._logical_rows or not self._last_consumed:
                raise RuntimeError('Commit requires exactly all real logical rows')
            self._commit_state()
            self.successful_steps += 1
            self._accepted_metrics = {key: value / self._logical_rows for key, value in self._sums.items()}
        self._active = False
        self._sums = {}
        self._last_metrics = None
        return dict(self._accepted_metrics) if success else {}

    def metrics(self):
        return dict(self._accepted_metrics)

    def training_state_dict(self):
        """Committed training-only state; never include selector/model tensors."""
        return dict(schema_version=1, kind=self.state_kind, successful_steps=self.successful_steps)

    def load_training_state_dict(self, state):
        if self._active:
            raise RuntimeError('Restore state between logical attempts')
        if state.get('schema_version') != 1 or state.get('kind') != self.state_kind:
            raise ValueError('Checkpoint objective kind/schema does not match')
        steps = state.get('successful_steps')
        if isinstance(steps, bool) or not isinstance(steps, int) or steps < 0:
            raise ValueError('Invalid successful step count')
        self.successful_steps = steps
        self._accepted_metrics = {}

    def _classification(self, output, batch, base_loss, parts):
        return base_loss, {}

    def _extra(self, first, second, batch):
        return first['logits'].new_zeros(()), {}

    def _model_batch(self, batch):
        return inputs_only(batch)

    def _selector_forward(self, batch):
        return self.selector(self._model_batch(batch))

    def forward(self, batch, consistency_weight):
        if not self._active:
            raise RuntimeError('Call begin_attempt(logical_batch, weight) before each attempt')
        if float(consistency_weight) != self._weight:
            raise ValueError('R-Drop weight cannot change inside a logical batch')
        if batch is None:
            zero = next(self.selector.parameters()).sum() * 0.
            return zero, zero.detach().new_zeros(len(PART_KEYS) - 1)
        if self._last_metrics is not None and not self._last_consumed:
            raise RuntimeError('Accumulate the prior microbatch before another forward')
        device = batch['ind'].device
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == 'cuda'):
            first = self._selector_forward(batch)
            second = self._selector_forward(batch)
        focus = predicted_top3(first['logits'], second['logits'], batch['pool_ids'], first['solver_mask'])
        original1, parts1 = single_objective(first, batch, focus)
        original2, parts2 = single_objective(second, batch, focus)
        loss1, metrics1 = self._classification(first, batch, original1, parts1)
        loss2, metrics2 = self._classification(second, batch, original2, parts2)
        self._after_views(first, second, batch)
        kl = symmetric_kl(first['logits'], second['logits'], first['solver_mask'])
        extra, metrics_extra = self._extra(first, second, batch)
        loss = .5 * (loss1 + loss2) + consistency_weight * kl + extra
        parts = {key: .5 * (parts1[key] + parts2[key]) for key in parts1}
        parts.update(loss=loss, kl=kl)
        metrics = {key: .5 * (metrics1[key] + metrics2[key]) for key in metrics1}
        metrics.update(metrics_extra)
        self._last_metrics = {key: value.detach() for key, value in metrics.items()}
        self._last_rows, self._last_consumed = len(batch['ind']), False
        return loss, torch.stack([parts[key].detach() for key in PART_KEYS[:-1]])

    def _after_views(self, first, second, batch):
        pass


class OVRRDropObjective(_AttemptObjective):
    """A natural Bernoulli control or B EQLv2-inspired utility classification."""

    def __init__(self, selector, mode='natural', gamma=EQL_GAMMA, mu=EQL_MU, alpha=EQL_ALPHA):
        super().__init__(selector)
        if mode not in ('natural', 'eql_v2'):
            raise ValueError('OVR mode must be natural or eql_v2')
        if not all(math.isfinite(value) for value in (gamma, mu, alpha)) or gamma <= 0 or alpha < 0:
            raise ValueError('EQL requires finite gamma>0, mu and alpha>=0')
        self.mode, self.gamma, self.mu, self.alpha = mode, float(gamma), float(mu), float(alpha)
        self.state_kind = 'r85_ovr_' + mode
        shape = (len(PROBLEMS), len(GLOBAL_SOLVERS))
        self._positive_gradient = torch.zeros(shape, dtype=torch.float64)
        self._negative_gradient = torch.zeros_like(self._positive_gradient)

    def _begin_state(self, batch):
        positive = self._positive_gradient[self._problem_id, self._pool_ids]
        negative = self._negative_gradient[self._problem_id, self._pool_ids]
        self.ratio = torch.where((positive + negative) == 0.,
                                 torch.full_like(positive, EQL_RECIPE['initial_ratio']),
                                 positive / (negative + EQL_RECIPE['epsilon']))
        if self.mode == 'eql_v2':
            negative_weight = torch.where(positive > 0.,
                torch.sigmoid(self.gamma * (self.ratio - self.mu)), torch.ones_like(positive))
            positive_weight = 1. + self.alpha * (1. - negative_weight)
        else:
            positive_weight, negative_weight = torch.ones_like(positive), torch.ones_like(negative)
        device = batch['ind'].device
        self.positive_weight = positive_weight.to(device=device, dtype=torch.float32)
        self.negative_weight = negative_weight.to(device=device, dtype=torch.float32)
        self._pending_positive = torch.zeros(len(self._pool_ids), device=device, dtype=torch.float64)
        self._pending_negative = torch.zeros_like(self._pending_positive)

    def _classification(self, output, batch, base_loss, parts):
        ovr = ovr_classification_loss(output['utility'], batch['ind'], output['solver_mask'],
                                      self.positive_weight, self.negative_weight)
        loss = CLASSIFICATION_WEIGHT * ovr + .35 * parts['cmp'] + .02 * parts['risk']
        return loss, dict(ovr=ovr, ovr_extra=CLASSIFICATION_WEIGHT * ovr,
                          native_ce=parts['ce'],
                          eql_positive_weight=self.positive_weight.mean(),
                          eql_negative_weight=self.negative_weight.mean())

    def _after_views(self, first, second, batch):
        gradients = [_ovr_terms(output['utility'], batch['ind'], output['solver_mask'],
                     self.positive_weight, self.negative_weight)[1:] for output in (first, second)]
        self._last_positive = .5 * (gradients[0][0] + gradients[1][0])
        self._last_negative = .5 * (gradients[0][1] + gradients[1][1])

    def _record_state(self, count):
        self._pending_positive.add_(self._last_positive.double(), alpha=count)
        self._pending_negative.add_(self._last_negative.double(), alpha=count)

    def _commit_state(self):
        positive = self._pending_positive.detach().cpu() / self._logical_rows
        negative = self._pending_negative.detach().cpu() / self._logical_rows
        if not torch.isfinite(positive).all() or not torch.isfinite(negative).all():
            raise RuntimeError('Cannot commit nonfinite EQL gradient statistics')
        self._positive_gradient[self._problem_id, self._pool_ids] += positive
        self._negative_gradient[self._problem_id, self._pool_ids] += negative

    def training_state_dict(self):
        return dict(super().training_state_dict(), gamma=self.gamma, mu=self.mu, alpha=self.alpha,
                    support_guard=EQL_RECIPE['support_guard'],
                    problem_names=list(PROBLEMS), solver_names=list(GLOBAL_SOLVERS),
                    positive_gradient=self._positive_gradient.clone(),
                    negative_gradient=self._negative_gradient.clone())

    def load_training_state_dict(self, state):
        if ((state.get('gamma'), state.get('mu'), state.get('alpha')) != (self.gamma, self.mu, self.alpha)
                or state.get('support_guard') != EQL_RECIPE['support_guard']
                or state.get('problem_names') != list(PROBLEMS)
                or state.get('solver_names') != list(GLOBAL_SOLVERS)):
            raise ValueError('EQL recipe or stable task/method registry changed')
        arrays = []
        for name in ('positive_gradient', 'negative_gradient'):
            value = torch.as_tensor(state[name]).detach().cpu().double().clone()
            if (value.shape != self._positive_gradient.shape or not torch.isfinite(value).all()
                    or (value < 0).any()):
                raise ValueError('EQL statistics must be finite nonnegative registry arrays')
            arrays.append(value)
        super().load_training_state_dict(state)
        self._positive_gradient, self._negative_gradient = arrays


class CostProfileKLRDropObjective(_AttemptObjective):
    """C: retain R77 and add 0.35 conditional nonwinner KL on final logits."""

    state_kind = 'r85_cost_profile_kl'

    def __init__(self, selector, auxiliary_weight=.35):
        super().__init__(selector)
        if not math.isfinite(auxiliary_weight) or auxiliary_weight < 0:
            raise ValueError('Auxiliary weight must be finite and nonnegative')
        self.auxiliary_weight = float(auxiliary_weight)

    def _extra(self, first, second, batch):
        value = .5 * sum(cost_profile_kl(out['logits'], batch['cost_regret'], out['solver_mask'], batch['ind'])
                         for out in (first, second))
        return self.auxiliary_weight * value, dict(cost_KL=value, cost_KL_extra=self.auxiliary_weight * value)


class CandidateSubsetRDropObjective(_AttemptObjective):
    """D: one label-independent legal subset per whole logical step.

    The model returns aux_subset_logits [B,K] for keyword aux_solver_mask [B,K].
    target_fn(batch, mask) returns a dictionary with local [B] ``ind`` from
    FP64-derived cmp_target/cmp_valid; the default helper belongs to that factory.
    """

    state_kind = 'r85_candidate_subset'

    def __init__(self, selector, auxiliary_weight=.35, seed=2, target_fn=None):
        super().__init__(selector)
        if not math.isfinite(auxiliary_weight) or auxiliary_weight < 0:
            raise ValueError('Auxiliary weight must be finite and nonnegative')
        self.auxiliary_weight = float(auxiliary_weight)
        self._subset_generator = torch.Generator(device='cpu').manual_seed(seed)
        self._target_fn = target_fn

    def begin_attempt(self, batch, weight):
        batch = super().begin_attempt(batch, weight)
        mask = batch['solver_mask'].detach().cpu().bool()
        if not torch.equal(mask, mask[:1].expand_as(mask)):
            raise ValueError('The shared subset needs a shared legal pool across logical rows')
        candidates = torch.where(mask[0])[0]
        generator = torch.Generator(device='cpu')
        generator.set_state(self._subset_generator.get_state())
        subset = torch.zeros_like(mask[0])
        if len(candidates) > 2:
            size = int(torch.randint(2, len(candidates), (), generator=generator))
            chosen = candidates[torch.randperm(len(candidates), generator=generator)[:size]]
            subset[chosen] = True
            result = dict(batch, aux_solver_mask=subset.to(batch['ind'].device)[None].expand_as(batch['solver_mask']))
        else:
            result = {key: value for key, value in batch.items() if key != 'aux_solver_mask'}
        self._pending_subset_rng = generator.get_state()
        return result

    def _commit_state(self):
        self._subset_generator.set_state(self._pending_subset_rng)

    def _selector_forward(self, batch):
        model_batch = self._model_batch(batch)
        mask = model_batch.pop('aux_solver_mask', None)
        if mask is None:
            return self.selector(model_batch)
        return self.selector(model_batch, aux_solver_mask=mask)

    def _extra(self, first, second, batch):
        if 'aux_solver_mask' not in batch:
            zero = first['logits'].new_zeros(())
            return zero, dict(subsetCE=zero, subsetCE_extra=zero)
        target_fn = self._target_fn
        if target_fn is None:
            from .r85_candidate_selector import subset_targets
            target_fn = subset_targets
        mask = batch['aux_solver_mask'].bool()
        labels = target_fn(batch, mask)['ind'].detach()
        value = .5 * sum(F.cross_entropy(output['aux_subset_logits'].float().masked_fill(~mask, float('-inf')),
                                        labels) for output in (first, second))
        return self.auxiliary_weight * value, dict(subsetCE=value, subsetCE_extra=self.auxiliary_weight * value)

    def training_state_dict(self):
        return dict(super().training_state_dict(), auxiliary_weight=self.auxiliary_weight,
                    subset_rng=self._subset_generator.get_state().clone())

    def load_training_state_dict(self, state):
        if state.get('auxiliary_weight') != self.auxiliary_weight:
            raise ValueError('Subset auxiliary weight changed')
        generator = torch.Generator(device='cpu')
        generator.set_state(state['subset_rng'].detach().cpu())
        super().load_training_state_dict(state)
        self._subset_generator.set_state(generator.get_state())


def make_objective(selector, arm):
    """Locked R85 arm names; factory import is code.V4.r85_objectives (plural)."""
    if arm == 'A_ovr_control':
        return OVRRDropObjective(selector, mode='natural')
    if arm == 'B_gradient_balance':
        return OVRRDropObjective(selector, mode='eql_v2')
    if arm == 'C_cost_profile':
        return CostProfileKLRDropObjective(selector)
    if arm == 'D_candidate_subset':
        return CandidateSubsetRDropObjective(selector)
    raise ValueError('Unknown R85 objective arm: ' + str(arm))

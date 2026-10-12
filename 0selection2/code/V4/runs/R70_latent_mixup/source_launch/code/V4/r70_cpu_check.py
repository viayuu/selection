"""One focused synthetic check of mixed gradients, isolated RNG and whole retry."""

import copy
import math

import torch
from torch import nn

from .r69_model import INPUT_KEYS, TaskSpecificSelector, make_optimizer, update_lr
from .r70_gradients import (GradientLayout, auxiliary_seed, clean_views,
                           interpolate_features, mixed_cross_entropy, update_triple)
from .training_monitor import capture_rng, restore_rng


class ObservedHead(nn.Sequential):
    def __init__(self, fail_on=None):
        super().__init__(nn.Linear(4, 8), nn.GELU(), nn.Dropout(.25), nn.Linear(8, 1))
        self.inputs, self.dropouts = [], []
        self.fail_on, self.observer = fail_on, None
        self[2].register_forward_hook(self.record_dropout)

    def record_dropout(self, module, args, output):
        self.dropouts.append(output.detach().clone())

    def forward(self, features):
        if self.observer is not None:
            self.observer()
        self.inputs.append(features.detach().clone())
        result = super().forward(features)
        return result * float('nan') if len(self.inputs) == self.fail_on else result


class ToySelector(nn.Module):
    def __init__(self, fail_on=None):
        super().__init__()
        self.stem = nn.Linear(3, 4)
        self.candidates = nn.Parameter(torch.randn(4, 4) / 10)
        self.dropout = nn.Dropout(.25)
        self.score_head = ObservedHead(fail_on)
        self.rows = []

    def encode_pairs(self, batch):
        assert set(batch) <= INPUT_KEYS and 'ind' not in batch and 'costs' not in batch
        self.rows.append(batch['node'].clone())
        return self.dropout(self.stem(batch['node']))[:, None, :] + self.candidates[None], batch['solver_mask']


def focused_check():
    torch.set_num_threads(1)
    torch.manual_seed(700001)
    mask = torch.tensor([[True, False, True, True]]).expand(4, -1)
    winner, permutation = torch.tensor([0, 2, 3, 2]), torch.tensor([1, 0, 3, 2])
    left, partner = torch.randn(4, 4, 4, requires_grad=True), torch.randn(4, 4, 4, requires_grad=True)
    head = nn.Linear(4, 1)
    coefficient = .37
    latent = interpolate_features(left, partner, coefficient)
    output = dict(logits=head(latent).squeeze(-1).masked_fill(~mask, -float('inf')), solver_mask=mask)
    loss = mixed_cross_entropy([output, output], winner, permutation, coefficient)
    loss.backward()
    assert latent.dtype == torch.float32 and torch.isfinite(loss)
    assert torch.isfinite(left.grad).all() and torch.isfinite(partner.grad).all()
    assert left.grad.abs().sum() > 0 and partner.grad.abs().sum() > 0
    torch.testing.assert_close(left.grad / coefficient, partner.grad / (1. - coefficient))
    for boundary in (0., 1.):
        assert torch.isfinite(mixed_cross_entropy([output, output], winner, permutation, boundary))

    fixture = dict(node=torch.tensor([[.2, .5, .8], [.7, .4, .1], [.9, .3, .6], [.1, .8, .4]]),
        node_mask=torch.ones(4, 1, dtype=torch.bool), solver_mask=mask,
        pool_ids=torch.tensor([7, 2, 9, 1]), ind=winner, costs=torch.full((4, 4), float('nan')))
    batches = [dict(fixture, problem_id=index) for index in range(3)]
    model = TaskSpecificSelector([ToySelector(fail_on=4 if index == 2 else None) for index in range(18)])
    reference = copy.deepcopy(model)
    optimizer = make_optimizer(model)
    parameters_before = [p.detach().clone() for p in model.parameters()]
    step_calls = []

    def before_head():
        assert not optimizer.state and optimizer.param_groups[0]['lr'] == 3e-4
        assert all(torch.equal(p, before) for p, before in zip(model.parameters(), parameters_before))

    for branch in model.branches[:3]:
        branch.score_head.observer = before_head
    optimizer.register_step_post_hook(lambda opt, args, kwargs: step_calls.append(1))
    torch.manual_seed(700003)
    rng = capture_rng()
    with torch.no_grad():
        for batch in batches:
            clean_views(reference.branches[batch['problem_id']],
                        {key: value for key, value in batch.items() if key in INPUT_KEYS},
                        False, dict(encoder=0, mixed_head=0))
    ordinary_after = capture_rng()
    restore_rng(rng)
    scaler = torch.amp.GradScaler('cpu', enabled=True, init_scale=8., growth_interval=2000)
    result = update_triple(model, optimizer, scaler, batches, .35 / 720, GradientLayout(model),
                           update_lr(0), successful_update=1, growth_successes=1999)
    assert result['retries'] == 1 and result['retry_attempts'][0]['task'] == 2
    assert result['scale'] == 4. and result['scale_growth_successes'] == 1
    assert len(step_calls) == 1 and result['global_clip_calls'] == 1
    assert result['successful_view_rows'] == result['successful_mixed_head_rows'] == 24
    assert result['attempted_view_rows'] == result['attempted_mixed_head_rows'] == 48
    assert torch.equal(torch.get_rng_state(), ordinary_after['torch'])
    for index, branch in enumerate(model.branches[:3]):
        assert len(branch.score_head.inputs) == 8 and len(branch.rows) == 4
        for call in range(4):
            torch.testing.assert_close(branch.score_head.inputs[call], branch.score_head.inputs[call + 4], rtol=0, atol=0)
            torch.testing.assert_close(branch.score_head.dropouts[call], branch.score_head.dropouts[call + 4], rtol=0, atol=0)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(auxiliary_seed(1, index))
            pi, lam = torch.randperm(4), float(torch.rand(()))
        assert result['auxiliary_draws'][index]['lambda_value'] == lam
        for view in range(2):
            torch.testing.assert_close(branch.score_head.inputs[view + 2],
                interpolate_features(branch.score_head.inputs[view], branch.score_head.inputs[view][pi], lam), rtol=0, atol=0)
        assert all(torch.equal(branch.rows[view], branch.rows[view + 2]) for view in range(2))
    assert all(int(state['step']) == 1 for state in optimizer.state.values())
    assert all(p.grad is None and p not in optimizer.state for branch in model.branches[3:] for p in branch.parameters())
    assert math.isfinite(result['loss']) and math.isfinite(result['grad_norm'])
    return dict(passed=True, focused_cpu_checks=1, synthetic_fixtures_only=True, gpu_used=False,
        masked_mix_ce_finite_at_interior_and_boundaries=True, gradients_through_both_partners=True,
        fp32_interpolation_and_loss=True, two_legal_logprob_gathers=True,
        clean_dropout_rng_identical_to_clean_only_reference=True,
        shared_pi_lambda_across_views=True, auxiliary_head_dropout_isolated=True,
        injected_third_task_mixed_output_nonfinite=True, injected_whole_update_retries=1,
        retry_rows_clean_dropout_pi_lambda_auxiliary_dropout_identical=True,
        failed_attempt_no_optimizer_or_lr_advancement=True, one_clip_and_optimizer_step=True,
        scaler_backoff_and_success_growth_reset=True, inactive_handling_reused_from_R69=True,
        clean_ce=result['clean_ce'], mix_ce=result['mix_ce'], kl=result['kl'], loss=result['loss'],
        test_read=False, training_pilot=False)

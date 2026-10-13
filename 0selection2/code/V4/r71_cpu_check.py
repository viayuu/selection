"""One synthetic invocation for the R71 density, inference and update paths."""

import copy
import math

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical, MixtureSameFamily, MultivariateNormal

from .r69_model import INPUT_KEYS, TaskSpecificSelector, make_optimizer, update_lr
from .r71_gradients import GradientLayout, update_triple
from .r71_model import JointMixtureHead, fit_normalizer, standardized_targets
from .training_monitor import capture_rng, restore_rng


class FixtureSelector(nn.Module):
    def __init__(self, problem_id, normalizer, fail_on=None):
        super().__init__()
        self.stem = nn.Linear(3, 384)
        self.ids = nn.Embedding(10, 384)
        self.dropout = nn.Dropout(.25)
        self.density_head = JointMixtureHead(problem_id, normalizer)
        self.features, self.inputs, self.calls = [], [], 0
        self.fail_on, self.observer = fail_on, None

    def density_parameters(self, batch, amp=False):
        assert set(batch) <= INPUT_KEYS and 'costs' not in batch and 'density_target' not in batch
        clean, positions, mask = self.density_head.canonical_inputs(batch)
        self.calls += 1
        if self.observer:
            self.observer()
        features = self.dropout(self.stem(clean['node']))[:, None] + self.ids(clean['pool_ids'])[None]
        if features.requires_grad:
            features.retain_grad()
        self.features.append(features)
        self.inputs.append(clean['node'].detach().clone())
        result = self.density_head(features)
        if self.calls == self.fail_on:
            result['mixture_logits'] = result['mixture_logits'] * float('nan')
        return result, positions, mask

    def forward(self, batch):
        parameters, positions, mask = self.density_parameters(batch)
        probability = self.density_head.winner_probabilities(parameters)
        return self.density_head.restore_probabilities(probability, positions, mask)


def focused_check():
    torch.set_num_threads(1)
    torch.manual_seed(710001)
    costs = np.array([[4., 1., 3.], [8., 6., 2.], [2., 3., 4.], [5., 2., 6.]], dtype=np.float64)
    pool = [7, 2, 5]
    normalizer = fit_normalizer(costs, pool)
    differences = costs[:, [2, 0]] - costs[:, [1]]
    np.testing.assert_array_equal(normalizer['pool_ids'], [2, 5, 7])
    np.testing.assert_allclose(normalizer['mean'], differences.mean(0), rtol=0, atol=0)
    np.testing.assert_allclose(normalizer['scale'], differences.std(0, ddof=0), rtol=0, atol=0)
    targets = standardized_targets(costs, pool, normalizer)
    torch.testing.assert_close(targets, torch.tensor((differences - differences.mean(0)) / differences.std(0), dtype=torch.float32))
    constant = fit_normalizer(np.ones((4, 3), dtype=np.float64), pool)
    assert constant['scale'] == [1e-6, 1e-6]
    torch.testing.assert_close(standardized_targets(costs[:, [1, 2, 0]], [2, 5, 7], normalizer), targets, rtol=0, atol=0)

    rng = torch.get_rng_state().clone()
    head = JointMixtureHead(0, normalizer)
    assert torch.equal(rng, torch.get_rng_state())
    assert not bool(head.layers[2].weight.any()) and not any(isinstance(m, nn.Dropout) for m in head.modules())
    features = torch.randn(4, 3, 384, requires_grad=True)
    parameters = head(features)
    torch.testing.assert_close(parameters['scale_tril'], torch.eye(2).expand(4, 8, 2, 2), rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(parameters['mixture_logits'], torch.zeros(4, 8), rtol=0, atol=0)
    expected = -MixtureSameFamily(Categorical(logits=parameters['mixture_logits']),
        MultivariateNormal(parameters['location'], scale_tril=parameters['scale_tril'])).log_prob(targets).mean() / 2
    loss = head.nll(parameters, targets)
    torch.testing.assert_close(loss, expected, rtol=0, atol=0)
    loss.backward()
    assert torch.isfinite(loss) and features.grad.abs().sum() == 0 and head.layers[0].weight.grad.abs().sum() == 0
    assert head.layers[2].weight.grad.abs().sum() > 0
    inference_rng = torch.get_rng_state().clone()
    probabilities = head.winner_probabilities(parameters)
    torch.testing.assert_close(probabilities.sum(-1), torch.ones(4))
    torch.testing.assert_close(head.winner_probabilities(parameters), probabilities, rtol=0, atol=0)
    torch.testing.assert_close(head.winner_probabilities({k: v[:1] for k, v in parameters.items()}), probabilities[:1], rtol=0, atol=0)
    assert torch.equal(inference_rng, torch.get_rng_state())
    restored = copy.deepcopy(head)
    restored.load_state_dict(head.state_dict(), strict=True)
    torch.testing.assert_close(restored.winner_probabilities(parameters), probabilities, rtol=0, atol=0)

    fixture = dict(node=torch.tensor([[.2, .5, .8], [.7, .4, .1], [.9, .3, .6], [.1, .8, .4]]),
        node_mask=torch.ones(4, 1, dtype=torch.bool), pool_ids=torch.tensor([7, -1, 2, 5]),
        solver_mask=torch.tensor([[True, False, True, True]]).expand(4, -1),
        costs=torch.full((4, 4), float('nan')), ind=torch.tensor([2, 0, 3, 2]), density_target=targets)
    clean, positions, mask = head.canonical_inputs(fixture)
    assert clean['pool_ids'].tolist() == [2, 5, 7] and clean['solver_mask'].all()
    assert positions.tolist() == [[2, 3, 0]] * 4
    tie = head.restore_probabilities(torch.tensor([[.5, .5, 0.]]).expand(4, -1), positions, mask)
    assert tie['pred'].tolist() == [2] * 4 and not bool(tie['probabilities'][:, 1].any())

    model = TaskSpecificSelector([FixtureSelector(i, normalizer, fail_on=2 if i == 2 else None) for i in range(18)])
    # Nonzero output weights are a connectivity TEST FIXTURE only; production stays zero.
    with torch.no_grad():
        for branch in model.branches:
            branch.density_head.layers[2].weight.normal_(0., .002)
    model.eval()
    ordinary = model(dict(fixture, problem_id=0))
    poisoned = dict(fixture, problem_id=0, costs=torch.full((4, 4), 1e30), ind=torch.full((4,), -999),
                    density_target=torch.full_like(targets, float('nan')))
    torch.testing.assert_close(model(poisoned)['probabilities'], ordinary['probabilities'], rtol=0, atol=0)
    reordered = dict(fixture, problem_id=0, pool_ids=torch.tensor([5, 7, 2]), solver_mask=torch.ones(4, 3, dtype=torch.bool))
    torch.testing.assert_close(model(reordered)['probabilities'], ordinary['probabilities'][:, [3, 0, 2]], rtol=0, atol=0)
    for branch in model.branches:
        branch.features, branch.inputs, branch.calls = [], [], 0
    model.train()
    optimizer = make_optimizer(model)
    before = [p.detach().clone() for p in model.parameters()]
    steps = []

    def no_advancement():
        assert not optimizer.state and optimizer.param_groups[0]['lr'] == 3e-4
        assert all(torch.equal(p, saved) for p, saved in zip(model.parameters(), before))

    for branch in model.branches[:3]:
        branch.observer = no_advancement
    optimizer.register_step_post_hook(lambda opt, args, kwargs: steps.append(1))
    batches = [dict(fixture, problem_id=i) for i in range(3)]
    torch.manual_seed(710004)
    rng = capture_rng()
    reference = copy.deepcopy(model)
    for branch in reference.branches:
        branch.fail_on, branch.observer = None, None
    with torch.no_grad():
        for batch in batches:
            for _ in range(2):
                reference.branches[batch['problem_id']].density_parameters({k: v for k, v in batch.items() if k in INPUT_KEYS})
    ordinary_rng = torch.get_rng_state().clone()
    restore_rng(rng)
    scaler = torch.amp.GradScaler('cpu', enabled=True, init_scale=8., growth_interval=2000)
    result = update_triple(model, optimizer, scaler, batches, GradientLayout(model), update_lr(0), 1, growth_successes=1999)
    assert result['retries'] == 1 and result['retry_attempts'][0]['task'] == 2
    assert result['scale'] == 4. and result['scale_growth_successes'] == 1
    assert len(steps) == 1 and result['global_clip_calls'] == 1
    assert result['successful_view_rows'] == 24 and result['attempted_view_rows'] == 48
    assert torch.equal(ordinary_rng, torch.get_rng_state())
    for branch in model.branches[:3]:
        assert len(branch.features) == 4
        for view in range(2):
            torch.testing.assert_close(branch.features[view], branch.features[view + 2], rtol=0, atol=0)
            assert torch.equal(branch.inputs[view], branch.inputs[view + 2])
            gradient = branch.features[view + 2].grad
            assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0
        assert branch.stem.weight.grad.abs().sum() > 0 and branch.density_head.layers[0].weight.grad.abs().sum() > 0
    assert all(int(state['step']) == 1 for state in optimizer.state.values())
    assert all(p.grad is None and p not in optimizer.state for branch in model.branches[3:] for p in branch.parameters())
    assert math.isfinite(result['loss']) and math.isfinite(result['grad_norm'])
    return dict(passed=True, focused_cpu_checks=1, synthetic_fixtures_only=True, gpu_used=False,
        installed_MixtureSameFamily_MultivariateNormal_likelihood=True,
        FP64_TRAIN_population_normalization_and_FP32_loss=True,
        canonical_order_mask_and_original_column_restoration=True,
        no_target_access_in_forward=True, persistent_fixed_Sobol_table_and_query_independence=True,
        probability_global_ID_tie=True, production_zero_output_init_verified=True,
        initial_zero_backbone_gradient_expected=True, both_view_connectivity_nonzero_fixture_only=True,
        injected_whole_update_retries=1, same_rows_dropout_RNG_and_parameters_on_retry=True,
        failed_attempt_no_optimizer_or_lr_advancement=True, one_clip_and_one_optimizer_step=True,
        scaler_backoff_growth_reset=True, inactive_handling_reused_from_R69=True,
        synthetic_nll_per_dimension=result['nll_per_dimension'], test_read=False, training_pilot=False)

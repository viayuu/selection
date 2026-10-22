"""Focused CPU-only R95 tests: no GPU, real datasets, training jobs, or Git."""

import copy
import unittest
from functools import partial, update_wrapper
from itertools import product

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .pairwise_objective import LABEL_KEYS, comparison_targets, predicted_top3, single_objective
from .pairwise_selector import maximin_logits
from .r42_experiment import symmetric_kl
from .r76_experiment import PART_KEYS, RDropObjective
from .r88_model import ARMS, make_model
from .r95_objective import (SAMPLING_KEYS, WeightedRDropObjective, final_affine_proxy,
                            inputs_only, per_row_objective)
from .test_r76 import fixture_batch, fixture_model


def output_from_utility(utility, mask=None):
    mask = torch.ones_like(utility, dtype=torch.bool) if mask is None else mask
    margins = utility[:, :, None] - utility[:, None, :]
    margins = margins.masked_fill(~(mask[:, :, None] & mask[:, None, :]), 0.)
    return dict(utility=utility, solver_mask=mask, pair_margin=margins,
                logits=maximin_logits(margins, mask))


def supervision(costs, mask=None, ids=None):
    costs = np.asarray(costs, dtype=np.float64)
    mask = np.ones_like(costs, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    batch = comparison_targets(costs, mask)
    batch.update(costs=torch.from_numpy(costs.astype(np.float32)),
                 ind=torch.from_numpy(np.where(mask, costs, np.inf).argmin(-1)),
                 solver_mask=torch.from_numpy(mask),
                 pool_ids=torch.arange(costs.shape[1]) if ids is None else torch.tensor(ids))
    return batch


def take_rows(mapping, rows, batch=False):
    return {key: value if (not torch.is_tensor(value) or value.ndim == 0
                          or batch and key == 'pool_ids' and value.ndim == 1) else value[rows]
            for key, value in mapping.items()}


def original_loss(first, second, batch, weight):
    focus = predicted_top3(first['logits'], second['logits'], batch['pool_ids'], first['solver_mask'])
    loss1, parts1 = single_objective(first, batch, focus)
    loss2, parts2 = single_objective(second, batch, focus)
    kl = symmetric_kl(first['logits'], second['logits'], first['solver_mask'])
    loss = .5 * (loss1 + loss2) + weight * kl
    parts = {key: .5 * (parts1[key] + parts2[key]) for key in parts1}
    parts.update(loss=loss, kl=kl)
    return loss, torch.stack([parts[key].detach() for key in PART_KEYS[:-1]])


def assert_parameter_gradients(first, second, multiplier=1.):
    assert dict(first.named_parameters()).keys() == dict(second.named_parameters()).keys()
    for (name, parameter), (_, reference) in zip(first.named_parameters(), second.named_parameters()):
        assert (parameter.grad is None) == (reference.grad is None), name
        if parameter.grad is not None:
            torch.testing.assert_close(parameter.grad, multiplier * reference.grad,
                                       rtol=2e-5, atol=2e-7, msg=name)


class TinySelector(nn.Module):
    def __init__(self):
        super().__init__()
        self.score_head = nn.Sequential(nn.Linear(3, 6), nn.GELU(), nn.Dropout(.25), nn.Linear(6, 1))
        self.seen = []

    def forward(self, batch):
        self.seen.append(set(batch))
        return output_from_utility(self.score_head(batch['features']).squeeze(-1), batch['solver_mask'])


def tiny_batch():
    batch = supervision([[1., 1.2, 1.8, 3.], [4., 1., 2., 3.], [2., 2., 2., 2.]])
    batch['features'] = torch.randn(3, 4, 3)
    return batch


def test_unit_weights_recover_original_loss_diagnostics_rng_and_all_parameter_gradients(role, problem, consistency):
    base, config = fixture_model()
    reference = base if role == 'R77' else make_model(config, ARMS[1], state_dict=base.state_dict())
    weighted = copy.deepcopy(reference)
    batch, _ = fixture_batch(problem, 3)
    if problem == 'TSP':
        batch['node'] = batch['node'][..., :2].clone()
    torch.manual_seed(951)
    expected, expected_parts = RDropObjective(reference.train())(batch, consistency)
    expected_rng = torch.get_rng_state().clone()
    expected.backward()
    torch.manual_seed(951)
    actual, parts = WeightedRDropObjective(weighted.train())(
        dict(batch, sampling_weight=torch.ones(3), sampling_row=torch.tensor([17, 9, 17])), consistency)
    actual.backward()
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=2e-7)
    torch.testing.assert_close(parts, expected_parts, rtol=1e-6, atol=2e-7)
    torch.testing.assert_close(torch.get_rng_state(), expected_rng, rtol=0, atol=0)
    assert_parameter_gradients(weighted, reference)
    assert not parts.requires_grad


def test_nonuniform_weights_cover_both_dropout_passes_all_terms_and_actual_row_mean():
    torch.manual_seed(95)
    reference = TinySelector().train()
    weighted = copy.deepcopy(reference)
    batch = tiny_batch()
    weights = torch.tensor([.2, 3., 1.3], requires_grad=True)
    torch.manual_seed(782)
    first, second = reference(inputs_only(batch)), reference(inputs_only(batch))
    losses, diagnostics = [], []
    for row in range(3):
        loss, parts = original_loss(take_rows(first, slice(row, row + 1)),
                                    take_rows(second, slice(row, row + 1)),
                                    take_rows(batch, slice(row, row + 1), batch=True), .35)
        losses.append(loss)
        diagnostics.append(parts)
    expected = (torch.stack(losses) * weights.detach()).mean()
    expected_parts = (torch.stack(diagnostics) * weights.detach()[:, None]).mean(0)
    expected.backward()
    torch.manual_seed(782)
    actual, parts = WeightedRDropObjective(weighted)(
        dict(batch, sampling_weight=weights, sampling_row=torch.tensor([5, 5, 1])), .35)
    actual.backward()
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(parts, expected_parts)
    assert_parameter_gradients(weighted, reference)
    assert weights.grad is None
    assert not torch.isclose(actual, expected * len(weights) / weights.detach().sum())
    assert parts[PART_KEYS.index('kl')] > 0
    assert set(LABEL_KEYS + SAMPLING_KEYS).isdisjoint(set.union(*weighted.seen))


def test_missing_weights_and_uniform_rescaling_do_not_self_normalize():
    torch.manual_seed(94)
    base = TinySelector().eval()
    scaled = copy.deepcopy(base)
    batch = tiny_batch()
    expected, parts = WeightedRDropObjective(base)(batch, .35)
    actual, scaled_parts = WeightedRDropObjective(scaled)(dict(batch, sampling_weight=torch.full((3,), 7.)), .35)
    expected.backward()
    actual.backward()
    torch.testing.assert_close(actual, 7 * expected)
    torch.testing.assert_close(scaled_parts, 7 * parts)
    assert_parameter_gradients(scaled, base, multiplier=7.)


def test_unit_weight_training_preserves_mask_singleton_and_pool_id_semantics(case):
    torch.manual_seed(91)
    batch = tiny_batch()
    if case == 'masked':
        mask = [[True, False, True, True], [False, True, False, False], [True, True, False, True]]
        batch.update(supervision(batch['costs'].numpy(), mask=mask))
    if case == 'one_row':
        batch = take_rows(batch, slice(0, 1), batch=True)
    if case == 'one_candidate':
        batch = dict(supervision([[1.], [2.], [3.]]), features=batch['features'][:, :1])
    if case == 'row_pool_ids':
        batch['pool_ids'] = torch.tensor([[9, 3, 7, 1], [3, 1, 9, 7], [7, 9, 1, 3]])
    reference = TinySelector().train()
    weighted = copy.deepcopy(reference)
    torch.manual_seed(12)
    expected, expected_parts = RDropObjective(reference)(batch, .35)
    expected.backward()
    torch.manual_seed(12)
    actual, parts = WeightedRDropObjective(weighted)(
        dict(batch, sampling_weight=torch.ones(len(batch['ind']))), .35)
    actual.backward()
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(parts, expected_parts)
    assert_parameter_gradients(weighted, reference)


def test_zero_weights_remove_every_supervised_and_consistency_term():
    model = TinySelector().train()
    loss, parts = WeightedRDropObjective(model)(dict(tiny_batch(), sampling_weight=torch.zeros(3)), .35)
    torch.testing.assert_close(loss, torch.tensor(0.), rtol=0, atol=0)
    torch.testing.assert_close(parts, torch.zeros(len(PART_KEYS) - 1), rtol=0, atol=0)
    loss.backward()
    for parameter in model.parameters():
        torch.testing.assert_close(parameter.grad, torch.zeros_like(parameter), rtol=0, atol=0)


def test_none_batch_has_original_graph_connected_zero_and_no_forward():
    reference, _ = fixture_model()
    weighted = copy.deepcopy(reference)
    expected, expected_parts = RDropObjective(reference)(None, .35)
    actual, parts = WeightedRDropObjective(weighted)(None, .35)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(parts, expected_parts, rtol=0, atol=0)
    actual.backward()
    expected.backward()
    assert_parameter_gradients(weighted, reference)


def test_invalid_importance_weights_are_rejected(weights):
    try:
        WeightedRDropObjective(TinySelector())(dict(tiny_batch(), sampling_weight=weights), .35)
    except ValueError as error:
        assert 'sampling_weight' in str(error)
    else:
        raise AssertionError('Expected invalid sampling_weight to raise ValueError')


def test_per_row_terms_match_original_shared_focus_and_keep_native_winners():
    costs = [[1. + 5e-10, 1., 2., 2., 4.], [1., 1., 1., 1., 1.]]
    batch = supervision(costs, ids=[9, 3, 5, 1, 7])
    batch['ind'][1] = 3  # Preserve the native label even when every raw cost ties.
    assert batch['ind'][0] == 1 and batch['costs'][0].argmin() == 0
    first = output_from_utility(torch.tensor([[4., 3., 2., 1., 0.], [0., 0., 0., 0., 0.]], requires_grad=True))
    second = output_from_utility(torch.tensor([[-4., -3., -2., -1., 0.], [0., 0., 0., 0., 0.]], requires_grad=True))
    rows, parts = per_row_objective(first, second, batch, .35)
    assert tuple(parts) == PART_KEYS[:-1]
    for row in range(2):
        expected, expected_parts = original_loss(take_rows(first, slice(row, row + 1)),
                                                take_rows(second, slice(row, row + 1)),
                                                take_rows(batch, slice(row, row + 1), batch=True), .35)
        torch.testing.assert_close(rows[row], expected)
        torch.testing.assert_close(torch.stack([parts[key][row] for key in PART_KEYS[:-1]]), expected_parts)
    assert parts['cmp'][1] == 0
    unchanged, _ = per_row_objective(first, second, dict(batch, sampling_weight=torch.tensor([9., 0.])), .35)
    torch.testing.assert_close(unchanged, rows, rtol=0, atol=0)
    rows.sum().backward()
    assert torch.isfinite(first['utility'].grad).all()


def test_sparse_comparison_focus_renormalizes_each_row_and_empty_groups_are_zero():
    batch = supervision([[1., 2., 3., 4.]] * 4)
    batch['cmp_valid'][:] = False
    batch['cmp_true3'][:] = False
    batch['cmp_valid'][0, 2] = True  # Only (0,3): neither true3 nor predicted3.
    batch['cmp_valid'][1, 0] = True  # Only (0,1): predicted3 and all pairs.
    batch['cmp_valid'][2, [0, 2]] = True
    batch['cmp_true3'][2, 2] = True
    output = output_from_utility(torch.tensor([[3., 2., 1., 0.]] * 4, requires_grad=True))
    _, parts = per_row_objective(output, output, batch, .35)
    pair01 = F.binary_cross_entropy_with_logits(torch.tensor(1.), torch.tensor(1.))
    pair03 = F.binary_cross_entropy_with_logits(torch.tensor(3.), torch.tensor(1.))
    expected = torch.stack((pair03, pair01, .5 * pair03 + .25 * pair01 + .25 * (pair01 + pair03) / 2,
                            torch.tensor(0.)))
    torch.testing.assert_close(parts['cmp'], expected)
    torch.testing.assert_close(parts['kl'], torch.zeros(4), rtol=0, atol=0)


def affine_reference(output, batch, hidden):
    """Independent parameter autograd through the original reduced objective."""
    norms, component_norm_sums = [], []
    for row in range(len(hidden)):
        weight = torch.zeros(1, hidden.size(-1), requires_grad=True)
        bias = torch.zeros(1, requires_grad=True)
        utility = output['utility'][row:row + 1].detach() + F.linear(hidden[row:row + 1].detach(), weight, bias).squeeze(-1)
        rebuilt = output_from_utility(utility, output['solver_mask'][row:row + 1])
        row_batch = take_rows(batch, slice(row, row + 1), batch=True)
        focus = predicted_top3(rebuilt['logits'], rebuilt['logits'], row_batch['pool_ids'], rebuilt['solver_mask'])
        loss, parts = single_objective(rebuilt, row_batch, focus)
        gradient = torch.autograd.grad(loss, (weight, bias), retain_graph=True)
        norms.append(torch.cat([value.flatten() for value in gradient]).norm())
        component_norm_sums.append(sum(torch.cat([value.flatten() for value in torch.autograd.grad(
            coefficient * parts[key], (weight, bias), retain_graph=True)]).norm()
            for coefficient, key in ((.35, 'ce'), (.35, 'cmp'), (.02, 'risk'))))
    return torch.stack(norms), torch.stack(component_norm_sums)


def test_proxy_is_exact_complete_deterministic_final_weight_and_bias_norm(case):
    torch.manual_seed(350)
    costs = [[1., 1.2, 2., 4.], [3., 2., 1., 2.]]
    mask = np.ones((2, 4), dtype=bool)
    if case == 'masked':
        mask[0, 1] = False
        mask[1, 0] = False
    if case == 'ties_native':
        costs = [[1. + 5e-10, 1., 2., 2.], [1., 1., 1., 1.]]
    if case == 'one_candidate':
        costs, mask = [[1.], [2.]], np.ones((2, 1), dtype=bool)
    if case == 'one_legal':
        mask[:] = False
        mask[0, 2] = mask[1, 1] = True
    batch = supervision(costs, mask=mask)
    if case == 'ties_native':
        batch['ind'][1] = 3
    hidden = torch.randn(2, len(costs[0]), 5, requires_grad=True)
    utility = torch.randn(2, len(costs[0]), requires_grad=True)
    if case == 'ties_native':
        utility = torch.zeros_like(utility, requires_grad=True)
    if case == 'sparse':
        batch['cmp_valid'][:] = False
        batch['cmp_true3'][:] = False
        batch['cmp_valid'][0, 2] = True
    if case == 'clamped':
        batch['cost_regret'][:] = 100.
    output = output_from_utility(utility, batch['solver_mask'])
    expected, component_sum = affine_reference(output, batch, hidden)
    before_rng = torch.get_rng_state().clone()
    with torch.no_grad():
        actual = final_affine_proxy(output, batch, hidden)
    torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-7)
    assert actual.shape == (2,) and not actual.requires_grad
    assert torch.isfinite(actual).all() and (actual >= 0).all()
    assert utility.grad is None and hidden.grad is None
    torch.testing.assert_close(torch.get_rng_state(), before_rng, rtol=0, atol=0)
    if case in ('one_candidate', 'one_legal'):
        torch.testing.assert_close(actual, torch.zeros(2), rtol=0, atol=0)
    if case == 'dense':
        assert (component_sum > actual + 1e-4).any()
    if case == 'clamped':
        _, parts = per_row_objective(output, output, batch, .35)
        torch.testing.assert_close(parts['risk'], torch.full((2,), 100.))
        risk_gradient, = torch.autograd.grad(parts['risk'].sum(), utility)
        torch.testing.assert_close(risk_gradient, torch.zeros_like(utility), rtol=0, atol=0)


def test_existing_eval_prehook_proxy_needs_no_forward_or_model_gradient_mutation(role):
    base, config = fixture_model()
    model = base if role == 'R77' else make_model(config, ARMS[1], state_dict=base.state_dict())
    batch, _ = fixture_batch('TSP', 3)
    batch['node'] = batch['node'][..., :2].clone()
    captured = []
    handle = model.score_head[-1].register_forward_pre_hook(lambda module, args: captured.append(args[0].detach()))
    model.eval()
    with torch.no_grad():
        output = model(inputs_only(batch))
    handle.remove()
    assert len(captured) == 1
    for parameter in model.parameters():
        parameter.grad = torch.full_like(parameter, .125)
    expected, _ = affine_reference(output, batch, captured[0])
    with torch.no_grad():
        actual = final_affine_proxy(output, batch, captured[0])
    torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-7)
    for parameter in model.parameters():
        torch.testing.assert_close(parameter.grad, torch.full_like(parameter, .125), rtol=0, atol=0)


def test_proxy_rebuilds_logits_ignores_sampling_raw_costs_and_is_chunk_invariant():
    torch.manual_seed(17)
    batch = tiny_batch()
    output = output_from_utility(torch.randn(3, 4), batch['solver_mask'])
    hidden = torch.randn(3, 4, 5)
    expected = final_affine_proxy(output, batch, hidden)
    poisoned_output = dict(output, logits=torch.full_like(output['logits'], float('nan')),
                           pair_margin=torch.full_like(output['pair_margin'], float('nan')))
    changed = dict(batch, costs=torch.full_like(batch['costs'], float('nan')),
                   sampling_weight=torch.tensor([0., 9., float('nan')]), sampling_row=torch.tensor([9, 9, 9]))
    torch.testing.assert_close(final_affine_proxy(poisoned_output, changed, hidden), expected, rtol=0, atol=0)
    pieces = [final_affine_proxy(take_rows(output, slice(row, row + 1)),
                                take_rows(batch, slice(row, row + 1), batch=True), hidden[row:row + 1])
              for row in range(3)]
    torch.testing.assert_close(torch.cat(pieces), expected, rtol=0, atol=0)


def test_proxy_all_zero_gradient_is_zero_without_an_artificial_floor():
    batch = supervision([[1.], [2.]])
    output = output_from_utility(torch.zeros(2, 1), batch['solver_mask'])
    torch.testing.assert_close(final_affine_proxy(output, batch, torch.zeros(2, 1, 3)), torch.zeros(2), rtol=0, atol=0)


def test_input_filter_is_nonmutating_and_removes_every_supervision_key():
    batch = dict(tiny_batch(), sampling_weight=torch.ones(3), sampling_row=torch.arange(3))
    original_keys = set(batch)
    cleaned = inputs_only(batch)
    assert SAMPLING_KEYS == ('sampling_weight', 'sampling_row')
    assert set(LABEL_KEYS + SAMPLING_KEYS).isdisjoint(cleaned)
    assert set(batch) == original_keys
    assert all(value is batch[key] for key, value in cleaned.items())


def load_tests(loader, tests, pattern):
    """Use standard-library FunctionTestCase for explicit dependency-free cases."""
    previous_threads = torch.get_num_threads()
    suite = unittest.TestSuite()

    def add(function, **kwargs):
        check = update_wrapper(partial(function, **kwargs), function)
        description = function.__name__ + (f' {kwargs}' if kwargs else '')
        suite.addTest(unittest.FunctionTestCase(
            check, setUp=lambda: torch.set_num_threads(2),
            tearDown=lambda: torch.set_num_threads(previous_threads), description=description))

    for role, problem, consistency in product(('R77', 'R88B'), ('TSP', 'ATSP', 'OVRPTW'), (0., .35)):
        add(test_unit_weights_recover_original_loss_diagnostics_rng_and_all_parameter_gradients,
            role=role, problem=problem, consistency=consistency)
    for case in ('masked', 'one_row', 'one_candidate', 'row_pool_ids'):
        add(test_unit_weight_training_preserves_mask_singleton_and_pool_id_semantics, case=case)
    for weights in (torch.ones(3, 1), torch.ones(2), torch.tensor([1., -1., 1.]),
                    torch.tensor([1., float('nan'), 1.]), torch.tensor([1., float('inf'), 1.])):
        add(test_invalid_importance_weights_are_rejected, weights=weights)
    for case in ('dense', 'masked', 'sparse', 'ties_native', 'clamped', 'one_candidate', 'one_legal'):
        add(test_proxy_is_exact_complete_deterministic_final_weight_and_bias_norm, case=case)
    for role in ('R77', 'R88B'):
        add(test_existing_eval_prehook_proxy_needs_no_forward_or_model_gradient_mutation, role=role)
    for function in (
            test_nonuniform_weights_cover_both_dropout_passes_all_terms_and_actual_row_mean,
            test_missing_weights_and_uniform_rescaling_do_not_self_normalize,
            test_zero_weights_remove_every_supervised_and_consistency_term,
            test_none_batch_has_original_graph_connected_zero_and_no_forward,
            test_per_row_terms_match_original_shared_focus_and_keep_native_winners,
            test_sparse_comparison_focus_renormalizes_each_row_and_empty_groups_are_zero,
            test_proxy_rebuilds_logits_ignores_sampling_raw_costs_and_is_chunk_invariant,
            test_proxy_all_zero_gradient_is_zero_without_an_artificial_floor,
            test_input_filter_is_nonmutating_and_removes_every_supervision_key):
        add(function)
    return suite


if __name__ == '__main__':
    unittest.main()

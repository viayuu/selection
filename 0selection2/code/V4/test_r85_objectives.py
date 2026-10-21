"""Focused CPU checks for R85 loss semantics and transactional training state."""

import copy
import io
import unittest

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from .pairwise_objective import LABEL_KEYS, comparison_targets
from .pairwise_selector import maximin_logits
from .r76_experiment import PART_KEYS, RDropObjective, slice_batch
from .r85_objectives import (CLASSIFICATION_WEIGHT, CandidateSubsetRDropObjective,
                            CostProfileKLRDropObjective, EQL_RECIPE, OVRRDropObjective,
                            cost_profile_kl, make_objective, ovr_classification_loss)


class UtilitySelector(nn.Module):
    def __init__(self, dropout=0.):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(4, 7), nn.Tanh(), nn.Dropout(dropout))
        self.score_head = nn.Linear(7, 5)
        self.outputs = []

    def forward(self, batch, aux_solver_mask=None):
        if set(batch).intersection(LABEL_KEYS):
            raise AssertionError('Supervision reached the selector forward')
        utility = self.score_head(self.encoder(batch['x'])).float()
        mask = batch['solver_mask']
        margins = utility[:, :, None] - utility[:, None, :]
        output = dict(utility=utility, solver_mask=mask, logits=maximin_logits(margins, mask),
                      pair_margin=margins.masked_fill(~(mask[:, :, None] & mask[:, None, :]), 0.))
        if aux_solver_mask is not None:
            # The production subset model has its own shared-parameter joint pass.
            output['aux_subset_logits'] = maximin_logits(margins, aux_solver_mask)
        self.outputs.append(output)
        return output


def fixture(rows=11):
    costs = np.array([[1. + .1 * ((m - i) % 5) for m in range(5)] for i in range(rows)], dtype=np.float64)
    costs[0] = [1., 1. + 1e-10, 1.2, 1.3, 1.4]
    labels = torch.from_numpy(costs.argmin(-1)).long()
    labels[0] = 1  # Preserve native labels, including a native/argmin discrepancy.
    return dict(x=torch.randn(rows, 4), problem_id=0, pool_ids=torch.tensor([4, 16, 9, 1, 20]),
                solver_mask=torch.ones(rows, 5, dtype=torch.bool), ind=labels,
                costs=torch.from_numpy(costs), **comparison_targets(costs))


def complete_attempt(objective, batch, cap=None, weight=.175, success=True):
    batch = objective.begin_attempt(batch, weight)
    count, loss_sum = len(batch['ind']), torch.zeros(())
    for start in range(0, count, cap or count):
        stop = min(start + (cap or count), count)
        micro = slice_batch(batch, start, stop)
        loss, _ = objective(micro, weight)
        scaled = loss * ((stop - start) / count)
        scaled.backward()
        loss_sum += scaled.detach()
        objective.accumulate(stop - start)
    metrics = objective.finish_attempt(success)
    return loss_sum, metrics


def first_allowed_target(batch, mask):
    # Target selection belongs to the separate candidate-selector owner's tests.
    assert batch['cmp_target'].dtype == torch.float32
    assert batch['cmp_valid'].dtype == torch.bool
    return dict(ind=mask.long().argmax(-1))


class R85ObjectivesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(2)

    def test_natural_ovr_literal_scaling_padding_and_finite_gradient(self):
        scores = torch.tensor([[0., 0., 0., float('nan')], [0., 0., float('inf'), float('nan')],
                               [0., float('nan'), float('inf'), float('-inf')]], requires_grad=True)
        mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0], [1, 0, 0, 0]], dtype=torch.bool)
        loss = ovr_classification_loss(scores, torch.tensor([2, 0, 0]), mask)
        torch.testing.assert_close(loss, (torch.log(torch.tensor(3.)) + torch.log(torch.tensor(2.))) / 3)
        loss.backward()
        self.assertTrue(torch.isfinite(scores.grad).all())
        self.assertTrue((scores.grad[~mask] == 0.).all())
        self.assertTrue((scores.grad[2] == 0.).all())
        logits = torch.tensor([[1000., -1000.], [-1000., 1000.]], dtype=torch.float16, requires_grad=True)
        extreme = ovr_classification_loss(logits, torch.tensor([0, 0]), torch.ones_like(logits, dtype=torch.bool))
        extreme.backward()
        self.assertTrue(torch.isfinite(extreme) and torch.isfinite(logits.grad).all())

    def test_replaces_only_ce_preserves_native_diagnostic_and_two_dropout_views(self):
        batch = fixture()
        old = RDropObjective(UtilitySelector(.3))
        objective = OVRRDropObjective(copy.deepcopy(old.selector))
        rng = torch.get_rng_state()
        old_loss, old_parts = old(batch, .175)
        torch.set_rng_state(rng)
        objective.begin_attempt(batch, .175)
        loss, parts = objective(batch, .175)
        first, second = objective.selector.outputs
        self.assertFalse(torch.equal(first['utility'], second['utility']))
        ovr = .5 * sum(ovr_classification_loss(out['utility'], batch['ind'], out['solver_mask'])
                        for out in (first, second))
        torch.testing.assert_close(loss, old_loss + .35 * (ovr - old_parts[1]), atol=2e-7, rtol=2e-6)
        torch.testing.assert_close(parts[1:], old_parts[1:], rtol=0, atol=0)
        self.assertEqual(tuple(parts.shape), (len(PART_KEYS) - 1,))
        self.assertEqual(len(objective.selector.outputs), 2)
        loss.backward()
        for name, parameter in objective.selector.named_parameters():
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
            self.assertGreater(parameter.grad.abs().sum().item(), 0., name)
        self.assertEqual(batch['ind'][0].item(), 1)
        self.assertEqual(batch['costs'][0].argmin().item(), 0)
        objective.accumulate(len(batch['ind']))
        metrics = objective.finish_attempt(True)
        torch.testing.assert_close(metrics['ovr'], ovr.detach().double())

    def test_eql_first_neutral_and_actual_weighted_two_view_logit_gradients(self):
        batch = fixture()
        objective = OVRRDropObjective(UtilitySelector(.3), mode='eql_v2')
        objective.begin_attempt(batch, .1)
        self.assertTrue((objective.ratio == 100.).all())
        self.assertTrue((objective.positive_weight == 1.).all())
        self.assertTrue((objective.negative_weight == 1.).all())
        loss, _ = objective(batch, .1)
        outputs = objective.selector.outputs
        classification = CLASSIFICATION_WEIGHT * .5 * sum(
            ovr_classification_loss(out['utility'], batch['ind'], out['solver_mask'],
                                     objective.positive_weight, objective.negative_weight) for out in outputs)
        gradients = torch.autograd.grad(classification, [out['utility'] for out in outputs], retain_graph=True)
        positive = F.one_hot(batch['ind'], 5).bool()
        expected_positive = sum(gradient.abs().masked_fill(~positive, 0.).sum(0).double() for gradient in gradients)
        expected_negative = sum(gradient.abs().masked_fill(positive, 0.).sum(0).double() for gradient in gradients)
        loss.backward()
        objective.accumulate(len(batch['ind']))
        before = objective.training_state_dict()
        self.assertEqual(before['successful_steps'], 0)
        self.assertEqual(before['positive_gradient'].sum().item(), 0.)
        objective.finish_attempt(True)
        state = objective.training_state_dict()
        ids = batch['pool_ids']
        torch.testing.assert_close(state['positive_gradient'][0, ids], expected_positive)
        torch.testing.assert_close(state['negative_gradient'][0, ids], expected_negative)
        objective.begin_attempt(batch, .1)
        ratio = expected_positive / (expected_negative + EQL_RECIPE['epsilon'])
        expected_negative_weight = torch.sigmoid(12. * (ratio - .8))
        torch.testing.assert_close(objective.negative_weight.double(), expected_negative_weight)
        torch.testing.assert_close(objective.positive_weight.double(), 1. + 4. * (1. - expected_negative_weight))
        self.assertFalse(objective.positive_weight.requires_grad or objective.negative_weight.requires_grad)
        objective.finish_attempt(False)

    def test_logical_1024_uneven_microbatch_loss_gradient_and_committed_state_equivalence(self):
        batch = fixture(1024)
        full = OVRRDropObjective(UtilitySelector(), mode='eql_v2')
        state = full.training_state_dict()
        state['positive_gradient'][0, batch['pool_ids']] = torch.tensor([.2, .01, .12, 1., .5], dtype=torch.float64)
        state['negative_gradient'][0, batch['pool_ids']] = torch.tensor([1., 3., .8, .7, 2.], dtype=torch.float64)
        full.load_training_state_dict(state)
        micro = OVRRDropObjective(copy.deepcopy(full.selector), mode='eql_v2')
        micro.load_training_state_dict(state)
        full_loss, full_metrics = complete_attempt(full, batch)
        micro_loss, micro_metrics = complete_attempt(micro, batch, cap=384)
        torch.testing.assert_close(full_loss, micro_loss, rtol=2e-6, atol=2e-7)
        for left, right in zip(full.selector.parameters(), micro.selector.parameters()):
            torch.testing.assert_close(left.grad, right.grad, rtol=2e-5, atol=2e-7)
        for key in ('positive_gradient', 'negative_gradient'):
            torch.testing.assert_close(full.training_state_dict()[key], micro.training_state_dict()[key])
        for key in full_metrics:
            torch.testing.assert_close(full_metrics[key], micro_metrics[key])

    def test_eql_zero_positive_support_keeps_natural_weights_and_guard_is_checkpointed(self):
        batch = fixture()
        objective = OVRRDropObjective(UtilitySelector(), mode='eql_v2')
        state = objective.training_state_dict()
        unsupported, supported = batch['pool_ids'][:2].tolist()
        state['negative_gradient'][0, unsupported] = 2.
        state['positive_gradient'][0, supported] = .05
        state['negative_gradient'][0, supported] = 3.
        self.assertEqual(state['support_guard'], 'natural weights until first positive gradient')
        objective.load_training_state_dict(state)
        objective.begin_attempt(batch, .175)
        self.assertEqual(objective.negative_weight[0].item(), 1.)
        self.assertEqual(objective.positive_weight[0].item(), 1.)
        ratio = torch.tensor(.05 / (3. + EQL_RECIPE['epsilon']), dtype=torch.float64)
        expected_negative = torch.sigmoid(12. * (ratio - .8)).float()
        torch.testing.assert_close(objective.negative_weight[1], expected_negative)
        torch.testing.assert_close(objective.positive_weight[1], 1. + 4. * (1. - expected_negative))
        self.assertLess(objective.negative_weight[1].item(), .01)
        objective.finish_attempt(False)
        with self.assertRaisesRegex(ValueError, 'recipe'):
            objective.load_training_state_dict(dict(state, support_guard='none'))

    def test_retry_discards_statistics_and_checkpoint_resumes_without_model_duplication(self):
        batch = fixture()
        objective = OVRRDropObjective(UtilitySelector(.3), mode='eql_v2')
        before = objective.training_state_dict()
        rng = torch.get_rng_state()
        complete_attempt(objective, batch, cap=4, success=False)
        torch.testing.assert_close(before['positive_gradient'], objective.training_state_dict()['positive_gradient'])
        self.assertEqual(objective.successful_steps, 0)
        objective.zero_grad(set_to_none=True)
        torch.set_rng_state(rng)
        complete_attempt(objective, batch, cap=4)
        accepted = objective.training_state_dict()
        self.assertEqual(accepted['successful_steps'], 1)
        self.assertTrue((accepted['positive_gradient'] > 0).any())
        self.assertEqual(set(objective.state_dict()), {'selector.' + key for key in objective.selector.state_dict()})
        buffer = io.BytesIO()
        torch.save(accepted, buffer)
        buffer.seek(0)
        restored = OVRRDropObjective(UtilitySelector(.3), mode='eql_v2')
        restored.selector.load_state_dict(objective.selector.state_dict())
        restored.load_training_state_dict(torch.load(buffer, weights_only=False))
        objective.begin_attempt(batch, .175)
        restored.begin_attempt(batch, .175)
        torch.testing.assert_close(restored.negative_weight, objective.negative_weight, rtol=0, atol=0)
        # A copy returned for checkpointing must not expose live buffers.
        accepted['positive_gradient'].zero_()
        self.assertGreater(restored.training_state_dict()['positive_gradient'].sum().item(), 0.)
        objective.finish_attempt(False)
        restored.finish_attempt(False)

    def test_problem_global_method_axis_and_no_state_change_without_all_rows(self):
        batch = fixture()
        objective = OVRRDropObjective(UtilitySelector(), mode='eql_v2')
        complete_attempt(objective, batch)
        state = objective.training_state_dict()
        self.assertGreater(state['positive_gradient'][0, 16].item(), 0.)
        self.assertEqual(state['positive_gradient'][7].sum().item(), 0.)
        other = dict(batch, problem_id=7)
        objective.begin_attempt(other, .175)
        self.assertTrue((objective.positive_weight == 1.).all())
        loss, _ = objective(slice_batch(other, 0, 3), .175)
        loss.backward()
        objective.accumulate(3)
        with self.assertRaisesRegex(RuntimeError, 'exactly all'):
            objective.finish_attempt(True)
        objective.finish_attempt(False)
        with self.assertRaisesRegex(ValueError, 'TRAIN'):
            objective.begin_attempt(dict(batch, split='val'), .175)

    def test_conditional_kl_literal_winner_zero_derivative_and_unsupported_rows(self):
        scores = torch.tensor([[123., .3, -1., float('nan')], [.2, -8., 1., 2.],
                               [float('nan'), float('nan'), float('nan'), float('nan')]], requires_grad=True)
        mask = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1], [1, 1, 0, 0]], dtype=torch.bool)
        labels = torch.tensor([0, 1, 0])
        regrets = torch.tensor([[0., .001, .02, float('nan')], [.02, 0., .001, .003],
                                [0., .001, float('nan'), float('nan')]], requires_grad=True)
        loss = cost_profile_kl(scores, regrets, mask, labels)
        literal = torch.zeros(())
        for row, indices in ((0, [1, 2]), (1, [0, 2, 3])):
            logq = F.log_softmax(-regrets[row, indices].detach() / .01, -1)
            logp = F.log_softmax(scores[row, indices], -1)
            literal += (logq.exp() * (logq - logp)).sum() / 3
        torch.testing.assert_close(loss, literal)
        loss.backward()
        self.assertTrue(torch.isfinite(scores.grad).all())
        self.assertEqual(scores.grad[0, 0].item(), 0.)
        self.assertEqual(scores.grad[1, 1].item(), 0.)
        self.assertTrue((scores.grad[2] == 0.).all())
        self.assertIsNone(regrets.grad)
        two = torch.randn(3, 2, requires_grad=True)
        zero = cost_profile_kl(two, torch.zeros_like(two), torch.ones_like(two, dtype=torch.bool), torch.zeros(3, dtype=torch.long))
        zero.backward()
        self.assertEqual(zero.item(), 0.)
        self.assertTrue((two.grad == 0.).all())

    def test_c_retains_old_losses_and_adds_two_view_conditional_kl(self):
        batch = fixture()
        old = RDropObjective(UtilitySelector(.3))
        objective = CostProfileKLRDropObjective(copy.deepcopy(old.selector))
        rng = torch.get_rng_state()
        base, old_parts = old(batch, .175)
        torch.set_rng_state(rng)
        objective.begin_attempt(batch, .175)
        loss, parts = objective(batch, .175)
        extra = .5 * sum(cost_profile_kl(out['logits'], batch['cost_regret'], out['solver_mask'], batch['ind'])
                          for out in objective.selector.outputs)
        torch.testing.assert_close(loss, base + .35 * extra, rtol=0, atol=0)
        torch.testing.assert_close(parts[1:], old_parts[1:], rtol=0, atol=0)
        self.assertEqual(len(objective.selector.outputs), 2)
        objective.accumulate(len(batch['ind']))
        torch.testing.assert_close(objective.finish_attempt(True)['cost_KL'], extra.detach().double())

    def test_d_label_independent_shared_subset_retry_and_checkpoint_rng(self):
        batch = fixture()
        objective = CandidateSubsetRDropObjective(UtilitySelector(), target_fn=first_allowed_target)
        rng = torch.get_rng_state().clone()
        prepared = objective.begin_attempt(batch, .175)
        subset = prepared['aux_solver_mask'].clone()
        self.assertTrue(torch.equal(subset, subset[:1].expand_as(subset)))
        self.assertTrue(2 <= int(subset[0].sum()) < 5)
        self.assertTrue(torch.equal(torch.get_rng_state(), rng))
        state = objective.training_state_dict()
        objective.finish_attempt(False)
        changed = dict(batch, ind=(batch['ind'] + 1) % 5, costs=torch.full_like(batch['costs'], float('nan')))
        retry = objective.begin_attempt(changed, .175)
        self.assertTrue(torch.equal(retry['aux_solver_mask'], subset))
        torch.testing.assert_close(state['subset_rng'], objective.training_state_dict()['subset_rng'])
        objective.finish_attempt(False)
        complete_attempt(objective, batch, cap=4)
        accepted = objective.training_state_dict()
        self.assertFalse(torch.equal(accepted['subset_rng'], state['subset_rng']))
        restored = CandidateSubsetRDropObjective(UtilitySelector(), target_fn=first_allowed_target)
        restored.load_training_state_dict(accepted)
        next_mask = objective.begin_attempt(batch, .175)['aux_solver_mask']
        restored_mask = restored.begin_attempt(batch, .175)['aux_solver_mask']
        self.assertTrue(torch.equal(next_mask, restored_mask))
        objective.finish_attempt(False)
        restored.finish_attempt(False)

    def test_d_matches_manual_auxiliary_and_microbatch_equivalence(self):
        batch = fixture()
        full = CandidateSubsetRDropObjective(UtilitySelector(), target_fn=first_allowed_target)
        micro = CandidateSubsetRDropObjective(copy.deepcopy(full.selector), target_fn=first_allowed_target)
        full_loss, full_metrics = complete_attempt(full, batch)
        micro_loss, micro_metrics = complete_attempt(micro, batch, cap=4)
        torch.testing.assert_close(full_loss, micro_loss)
        torch.testing.assert_close(full_metrics['subsetCE'], micro_metrics['subsetCE'])
        self.assertGreater(full_metrics['subsetCE'].item(), 0.)
        torch.testing.assert_close(full_metrics['subsetCE_extra'], full_metrics['subsetCE'] * .35)
        for left, right in zip(full.selector.parameters(), micro.selector.parameters()):
            torch.testing.assert_close(left.grad, right.grad, rtol=2e-5, atol=2e-7)

    def test_real_d_model_default_target_dictionary_and_keyword_mask(self):
        from .r85_candidate_selector import make_candidate_model
        from .test_r76 import fixture_batch, fixture_model
        _, params = fixture_model(dropout=0.)
        model = make_candidate_model(params).train()
        objective = make_objective(model, 'D_candidate_subset')
        batch, _ = fixture_batch('TSP', 3)
        loss, metrics = complete_attempt(objective, batch, cap=2)
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(metrics['subsetCE'].item(), 0.)
        self.assertEqual(objective.successful_steps, 1)
        for module in (model.instance_encoder.coord_proj, *model.joint_layers, model.score_head):
            gradients = [p.grad for p in module.parameters() if p.grad is not None]
            self.assertTrue(gradients)
            self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))
            self.assertGreater(sum(gradient.abs().sum().item() for gradient in gradients), 0.)


if __name__ == '__main__':
    unittest.main()

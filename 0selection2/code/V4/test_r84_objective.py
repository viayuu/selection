"""Focused CPU checks for the additive R84 focal objective; no training launch."""

import copy
import unittest

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from .pairwise_objective import LABEL_KEYS, comparison_targets
from .r76_experiment import PART_KEYS, RDropObjective, slice_batch
from .r84_objective import FOCAL_GAMMA, FOCAL_WEIGHT, FocalRDropObjective, focal_loss


class RecordingSelector(nn.Module):
    def __init__(self):
        super().__init__()
        self.score_head = nn.Sequential(nn.Linear(4, 7), nn.Tanh(), nn.Dropout(.4), nn.Linear(7, 3))
        self.outputs = []

    def forward(self, batch):
        if any(key in batch for key in LABEL_KEYS):
            raise AssertionError('Supervision reached the model forward')
        scores = self.score_head(batch['x']).float()
        mask = batch['solver_mask']
        margins = scores[:, :, None] - scores[:, None, :]
        output = dict(logits=scores.masked_fill(~mask, float('-inf')), solver_mask=mask,
                      pair_margin=margins.masked_fill(~(mask[:, :, None] & mask[:, None, :]), 0.))
        self.outputs.append(output)
        return output


def fixture():
    costs = np.asarray([[1., 1. + 1e-10, 1.6], [1.3, 1., 1.2],
                        [1.4, 1.3, 1.], [1., 1.2, 1.5], [1.2, 1., 1.3]], dtype=np.float64)
    # Preserve the native label even when it differs from a re-derived argmin.
    batch = dict(x=torch.randn(5, 4), solver_mask=torch.ones(5, 3, dtype=torch.bool),
                 pool_ids=torch.tensor([4, 1, 9]), ind=torch.tensor([1, 1, 2, 0, 1]),
                 costs=torch.from_numpy(costs), **comparison_targets(costs))
    return batch


class R84ObjectiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(2)

    def test_gamma_zero_is_exact_ce(self):
        logits = torch.randn(7, 4, requires_grad=True)
        labels = torch.tensor([0, 1, 2, 3, 0, 1, 3])
        loss = focal_loss(logits, labels, gamma=0)
        ce = F.cross_entropy(logits, labels)
        torch.testing.assert_close(loss, ce, rtol=0, atol=0)
        torch.testing.assert_close(torch.autograd.grad(loss, logits)[0],
                                   torch.autograd.grad(ce, logits)[0], rtol=0, atol=0)

    def test_gamma_two_hard_relative_penalty_and_full_modulation_gradient(self):
        logits = torch.tensor([[4., -1.], [-1., 4.]], requires_grad=True)
        labels = torch.zeros(2, dtype=torch.long)
        losses = [focal_loss(logits[i:i+1], labels[i:i+1]) for i in range(2)]
        nll = F.cross_entropy(logits, labels, reduction='none')
        self.assertGreater((losses[1] / nll[1]).item(), (losses[0] / nll[0]).item())
        literal = (((1. - logits.softmax(-1)[:, 0]) ** 2) * nll).mean()
        actual = focal_loss(logits, labels)
        grad = torch.autograd.grad(actual, logits, retain_graph=True)[0]
        torch.testing.assert_close(actual, literal)
        torch.testing.assert_close(grad, torch.autograd.grad(literal, logits, retain_graph=True)[0])
        detached = (((1. - logits.softmax(-1)[:, 0]).detach() ** 2) * nll).mean()
        self.assertFalse(torch.allclose(grad, torch.autograd.grad(detached, logits)[0]))

    def test_extreme_logits_padding_single_candidate_and_fp32_gradients(self):
        logits = torch.tensor([[1000., -1000., float('nan')], [-1000., 1000., float('inf')],
                               [0., float('nan'), float('-inf')]], dtype=torch.float16,
                              requires_grad=True)
        mask = torch.tensor([[True, True, False], [True, True, False], [True, False, False]])
        labels = torch.tensor([0, 0, 0])
        loss = focal_loss(logits, labels, mask)
        self.assertEqual(loss.dtype, torch.float32)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertTrue((logits.grad[~mask] == 0).all())
        self.assertTrue((logits.grad[2] == 0).all())

    def test_two_forward_literal_addition_native_labels_and_metrics(self):
        batch = fixture()
        reference = RDropObjective(RecordingSelector()).train()
        for gamma in (0., FOCAL_GAMMA):
            reference.selector.outputs.clear()
            focal = FocalRDropObjective(copy.deepcopy(reference.selector), gamma=gamma).train()
            rng = torch.get_rng_state()
            base, base_parts = reference(batch, .175)
            torch.set_rng_state(rng)
            loss, parts = focal(batch, .175)
            outputs = focal.selector.outputs
            self.assertEqual(len(outputs), 2)
            self.assertFalse(torch.equal(outputs[0]['logits'], outputs[1]['logits']))
            expected = .5 * sum(focal_loss(out['logits'], batch['ind'], out['solver_mask'], gamma)
                                 for out in outputs)
            torch.testing.assert_close(loss, base + FOCAL_WEIGHT * expected, rtol=0, atol=0)
            self.assertEqual(parts.shape, (len(PART_KEYS) - 1,))
            torch.testing.assert_close(parts[0], loss.detach(), rtol=0, atol=0)
            torch.testing.assert_close(parts[1:], base_parts[1:], rtol=0, atol=0)
            torch.testing.assert_close(focal.last_focal, expected.detach(), rtol=0, atol=0)
            torch.testing.assert_close(focal.last_focal_extra, FOCAL_WEIGHT * expected.detach(), rtol=0, atol=0)
            self.assertFalse(parts.requires_grad or focal.last_focal.requires_grad)
            self.assertEqual(batch['ind'][0].item(), 1)
            self.assertEqual(batch['costs'][0].argmin().item(), 0)

    def test_zero_weight_exact_old_loss_parts_gradient_and_rng(self):
        batch = fixture()
        old = RDropObjective(RecordingSelector()).train()
        new = FocalRDropObjective(copy.deepcopy(old.selector), focal_weight=0).train()
        rng = torch.get_rng_state()
        old_loss, old_parts = old(batch, .35)
        old_loss.backward()
        after_old = torch.get_rng_state()
        torch.set_rng_state(rng)
        new_loss, new_parts = new(batch, .35)
        new_loss.backward()
        torch.testing.assert_close(new_loss, old_loss, rtol=0, atol=0)
        torch.testing.assert_close(new_parts, old_parts, rtol=0, atol=0)
        for left, right in zip(old.parameters(), new.parameters()):
            torch.testing.assert_close(left.grad, right.grad, rtol=0, atol=0)
        self.assertTrue(torch.equal(torch.get_rng_state(), after_old))
        self.assertEqual(len(new.selector.outputs), 2)
        self.assertEqual(new.last_focal_extra.item(), 0.)

    def test_weighted_logging_resets_failed_attempt_and_ignores_dummy_rows(self):
        batch = fixture()
        objective = FocalRDropObjective(RecordingSelector()).train()
        objective(batch, .35)
        objective.accumulate_focal_metrics(5)
        objective.reset_focal_metrics()  # Discard the simulated failed AMP attempt.
        expected = torch.zeros((), dtype=torch.float64)
        for start, stop in ((0, 2), (2, 5)):
            loss, parts = objective(slice_batch(batch, start, stop), .35)
            self.assertEqual(len(parts), len(PART_KEYS) - 1)
            expected += objective.last_focal.double() * (stop - start)
            objective.accumulate_focal_metrics(stop - start)
        calls = len(objective.selector.outputs)
        objective.accumulate_focal_metrics(0)
        metrics = objective.focal_metrics()
        self.assertEqual(len(objective.selector.outputs), calls)
        torch.testing.assert_close(metrics['focal'], expected / 5, rtol=0, atol=0)
        torch.testing.assert_close(metrics['focal_extra'], FOCAL_WEIGHT * expected / 5)
        self.assertFalse(metrics['focal'].requires_grad)
        self.assertEqual(objective._focal_rows, 5)

    def test_empty_rank_does_not_require_score_head(self):
        selector = nn.Linear(3, 2)
        objective = FocalRDropObjective(selector)
        loss, parts = objective(None, .35)
        self.assertEqual(loss.item(), 0.)
        self.assertEqual(parts.shape, (len(PART_KEYS) - 1,))
        self.assertTrue((parts == 0).all())
        self.assertEqual(objective.last_focal.item(), 0.)
        loss.backward()
        self.assertTrue((selector.weight.grad == 0).all())


if __name__ == '__main__':
    unittest.main()

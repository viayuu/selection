"""Tiny CPU algebra/gradient checks for the exact locked R108 objectives."""

import copy
import unittest
from unittest.mock import patch

import torch
from torch import nn
import torch.nn.functional as F

from .pairwise_objective import LABEL_KEYS
from .pairwise_selector import maximin_logits
from .r76_experiment import PART_KEYS
from .r108_model import ARMS
from .r108_objective import ConditionalPriorObjective, make_objective


class ToySelector(nn.Module):
    def __init__(self, dropout=.1):
        super().__init__()
        self.scores = nn.Parameter(torch.tensor([[.8, -.7, 1.4], [-.2, 1.1, .4]]))
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pi', torch.tensor([[.6, .3, .1], [.1, .2, .7]]))
        self.outputs = []

    def forward(self, batch):
        if set(batch) & set(LABEL_KEYS):
            raise AssertionError('Supervision in selector input')
        t = self.dropout(self.scores)
        s = t + self.pi.log()
        mask = batch['solver_mask']
        margin = s[:, :, None] - s[:, None, :]
        output = dict(utility=s, residual_utility=t, conditional_prior=self.pi,
                      solver_mask=mask, pair_margin=margin, logits=maximin_logits(margin, mask))
        self.outputs.append(output)
        return output


def fixture():
    return dict(ind=torch.tensor([1, 2]), solver_mask=torch.ones(2, 3, dtype=torch.bool),
                **{key: object() for key in LABEL_KEYS if key != 'ind'})


class R108ObjectiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        guard = patch('torch.cuda._lazy_init', side_effect=AssertionError('CPU test cannot initialize CUDA'))
        guard.start()
        self.addCleanup(guard.stop)

    def test_one_dropout_forward_exact_math_gradient_and_parts(self):
        for arm in ARMS:
            objective = make_objective(ToySelector(), arm).train()
            batch = fixture()
            loss, parts = objective(batch, consistency_weight=float('nan'))
            self.assertEqual(len(objective.selector.outputs), 1)
            output = objective.selector.outputs[0]
            s, t, pi = [output[k] for k in ('utility', 'residual_utility', 'conditional_prior')]
            ce = F.cross_entropy(s, batch['ind'])
            expected = ce if arm == ARMS[0] else (
                .5 * F.softplus(-t.gather(1, batch['ind'][:, None]).squeeze(1))
                + .5 * (pi * F.softplus(t)).sum(-1)).mean()
            torch.testing.assert_close(loss, expected, rtol=0, atol=0)
            self.assertEqual(parts.shape, (len(PART_KEYS) - 1,))
            torch.testing.assert_close(parts[0], expected.detach(), rtol=0, atol=0)
            torch.testing.assert_close(parts[1], ce.detach(), rtol=0, atol=0)
            self.assertTrue((parts[2:] == 0).all())
            self.assertFalse(parts.requires_grad)
            self.assertNotEqual(ce.item(), F.cross_entropy(output['logits'], batch['ind']).item())
            actual = torch.autograd.grad(loss, t, retain_graph=True)[0]
            onehot = F.one_hot(batch['ind'], 3)
            gradient = (s.softmax(-1) - onehot) / 2 if arm == ARMS[0] else (
                .5 * ((pi + onehot) * t.sigmoid() - onehot) / 2)
            torch.testing.assert_close(actual, gradient, rtol=2e-6, atol=2e-8)
            loss.backward()
            self.assertTrue(torch.isfinite(objective.selector.scores.grad).all())
            self.assertIsNone(objective.selector.pi.grad)

    def test_true_winner_is_in_product_noise_and_eta_over_pi_is_stationary(self):
        eta = torch.tensor([.2, .5, .3], dtype=torch.float64)
        pi = torch.tensor([.65, .25, .10], dtype=torch.float64)
        t = (eta / pi).log().requires_grad_()
        population = .5 * (eta * F.softplus(-t)).sum() + .5 * (pi * F.softplus(t)).sum()
        gradient = torch.autograd.grad(population, t)[0]
        torch.testing.assert_close(gradient, torch.zeros_like(t), rtol=0, atol=1e-16)
        torch.testing.assert_close((pi.log() + t).softmax(-1), eta, rtol=1e-14, atol=1e-14)
        self.assertNotEqual(t.argmax().item(), eta.argmax().item())
        selector = ToySelector(dropout=0.)
        loss, _ = make_objective(selector, ARMS[1])(fixture())
        output = selector.outputs[0]
        y = fixture()['ind']
        removed = output['conditional_prior'].clone().scatter_(1, y[:, None], 0.)
        wrong = (.5 * F.softplus(-output['residual_utility'].gather(1, y[:, None]).squeeze(1))
                 + .5 * (removed * F.softplus(output['residual_utility'])).sum(-1)).mean()
        self.assertGreater(loss.item(), wrong.item())

    def test_cost_poison_consistency_weight_rng_and_query_label_isolation(self):
        for arm in ARMS:
            left = make_objective(ToySelector(), arm).train()
            right = copy.deepcopy(left)
            batch = fixture()
            rng = torch.get_rng_state()
            loss, parts = left(batch, 0.)
            after = torch.get_rng_state()
            loss.backward()
            torch.set_rng_state(rng)
            other, other_parts = right(dict(batch, costs=torch.full((2, 3), -999.)), 999.)
            other.backward()
            self.assertTrue(torch.equal(after, torch.get_rng_state()))
            torch.testing.assert_close(parts, other_parts, rtol=0, atol=0)
            torch.testing.assert_close(left.selector.scores.grad, right.selector.scores.grad, rtol=0, atol=0)
            selector = ToySelector().eval()
            objective = make_objective(selector, arm)
            first, _ = objective(batch)
            second, _ = objective(dict(batch, ind=torch.tensor([0, 0])))
            self.assertNotEqual(first.item(), second.item())
            for key in selector.outputs[0]:
                torch.testing.assert_close(selector.outputs[0][key], selector.outputs[1][key], rtol=0, atol=0)

    def test_illegal_nonfinite_entries_masked_before_softplus(self):
        selector = ToySelector(dropout=0.)
        with torch.no_grad():
            selector.scores[0, 2] = float('nan')
            selector.scores[1, 1] = float('inf')
            selector.pi[0, 2] = 0.
            selector.pi[1] = torch.tensor([.1, 0., .9])
        mask = torch.tensor([[True, True, False], [True, False, True]])
        for arm in ARMS:
            objective = make_objective(copy.deepcopy(selector), arm)
            loss, parts = objective(dict(ind=torch.tensor([1, 2]), solver_mask=mask))
            self.assertTrue(torch.isfinite(loss))
            self.assertTrue(torch.isfinite(parts).all())
            loss.backward()
            self.assertTrue(torch.isfinite(objective.selector.scores.grad).all())
            self.assertTrue((objective.selector.scores.grad[~mask] == 0).all())

    def test_unequal_microbatches_are_exact_row_weighted_no_queue(self):
        class TableSelector(nn.Module):
            def __init__(self):
                super().__init__()
                self.scores = nn.Parameter(torch.arange(15, dtype=torch.float32).reshape(5, 3) / 7.)

            def forward(self, batch):
                t = self.scores[batch['row']]
                pi = t.new_tensor([.2, .5, .3]).expand_as(t)
                return dict(utility=pi.log() + t, residual_utility=t,
                            conditional_prior=pi, solver_mask=torch.ones_like(t, dtype=torch.bool))
        for arm in ARMS:
            full = make_objective(TableSelector(), arm)
            micro = copy.deepcopy(full)
            batch = dict(row=torch.arange(5), ind=torch.tensor([0, 2, 1, 2, 0]))
            loss, parts = full(batch)
            loss.backward()
            accumulated = torch.zeros_like(parts)
            for start, stop in ((0, 2), (2, 5)):
                loss, piece = micro({k: v[start:stop] for k, v in batch.items()})
                (loss * ((stop-start) / 5)).backward()
                accumulated += piece * ((stop-start) / 5)
            torch.testing.assert_close(full.selector.scores.grad, micro.selector.scores.grad, rtol=1e-6, atol=2e-8)
            torch.testing.assert_close(parts, accumulated, rtol=1e-6, atol=2e-8)
            self.assertEqual(set(dict(full.named_buffers())), set())

    def test_idle_batch_no_forward_and_unknown_arm_guard(self):
        for arm in ARMS:
            objective = ConditionalPriorObjective(ToySelector(), arm)
            loss, parts = objective(None)
            self.assertEqual(loss.item(), 0.)
            self.assertTrue((parts == 0).all())
            self.assertEqual(objective.selector.outputs, [])
            loss.backward()
        with self.assertRaises(ValueError):
            make_objective(ToySelector(), 'other')


if __name__ == '__main__':
    unittest.main()

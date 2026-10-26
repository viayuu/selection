"""Four bounded CPU checks for the finite R116 subset sidecar."""

import itertools
import math
import unittest
from unittest.mock import patch

import torch

from .r116_subset import (champion_log_probs, full_ranking_nll,
                          stable_cost_order, subset_structure)


class R116SubsetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        guard = patch('torch.cuda._lazy_init', side_effect=AssertionError('CPU tests only'))
        guard.start()
        self.addCleanup(guard.stop)
        self.generator = torch.Generator(device='cpu').manual_seed(116)

    def test_k4_exhaustive_permutations_and_finite_gradients(self):
        k, batch = 4, 3
        structure = subset_structure(k)
        logits = torch.randn(batch, len(structure['state_ids']), k,
                             dtype=torch.float64, generator=self.generator)
        logits = logits.masked_fill(~structure['masks'], torch.nan).requires_grad_()
        rows = {state: row for row, state in enumerate(structure['state_ids'].tolist())}
        paths = [[] for _ in range(k)]
        for elimination_order in itertools.permutations(range(k)):
            remaining, path = (1 << k) - 1, logits.new_zeros(batch)
            for removed in elimination_order[:-1]:
                row = rows[remaining]
                legal = [j for j in range(k) if remaining & (1 << j)]
                path = path + logits[:, row, removed] - logits[:, row, legal].logsumexp(-1)
                remaining ^= 1 << removed
            paths[elimination_order[-1]].append(path)
        expected = torch.stack([torch.stack(p, -1).logsumexp(-1) for p in paths], -1)
        actual = champion_log_probs(logits)
        torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
        target = torch.tensor([0, 2, 3])
        loss = -actual.gather(1, target[:, None]).mean()
        reference = -expected.gather(1, target[:, None]).mean()
        gradient = torch.autograd.grad(loss, logits, retain_graph=True)[0]
        reference_gradient = torch.autograd.grad(reference, logits)[0]
        torch.testing.assert_close(gradient, reference_gradient, rtol=1e-11, atol=1e-13)
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertTrue((logits.grad.masked_select(~structure['masks']) == 0).all())
        self.assertGreater(float(logits.grad.abs().sum()), 0.)

    def test_static_logits_exact_dp_unit_mass_k1_through_k10(self):
        for k in range(1, 11):
            with self.subTest(k=k):
                structure = subset_structure(k, device='cpu')
                expected_ids = [i for i in range(1 << k) if bin(i).count('1') >= 2]
                self.assertEqual(structure['state_ids'].tolist(), expected_ids)
                self.assertEqual(structure['state_ids'].dtype, torch.long)
                self.assertEqual(structure['masks'].dtype, torch.bool)
                self.assertEqual(structure['masks'].shape, ((1 << k) - k - 1, k))
                bits = 1 << torch.arange(k)
                self.assertTrue(torch.equal(structure['masks'],
                    (structure['state_ids'][:, None] & bits) != 0))
                self.assertIs(structure, subset_structure(k, torch.device('cpu')))
                static = torch.stack((torch.zeros(k, dtype=torch.float64),
                    torch.linspace(-2., 2., k, dtype=torch.float64),
                    torch.linspace(-1000., 1000., k, dtype=torch.float64)))
                logits = static[:, None].expand(-1, len(expected_ids), -1).clone()
                logits = logits.masked_fill(~structure['masks'], torch.inf).requires_grad_()
                champion = champion_log_probs(logits)
                self.assertEqual(champion.shape, (3, k))
                self.assertTrue(torch.isfinite(champion).all())
                torch.testing.assert_close(champion.logsumexp(-1), torch.zeros(3, dtype=torch.float64),
                                           rtol=0, atol=1e-12)
                torch.testing.assert_close(champion[0].exp(), torch.full((k,), 1. / k,
                                           dtype=torch.float64), rtol=1e-12, atol=1e-12)
                if k == 2:
                    torch.testing.assert_close(champion, static.log_softmax(-1).flip(-1),
                                               rtol=1e-12, atol=1e-12)
                (-champion.mean()).backward()
                self.assertTrue(torch.isfinite(logits.grad).all())
                if k > 2:
                    self.assertLess(champion[2, -1].item(), -1000.)
        for bad_k in (0, 11, True, 2.5):
            with self.assertRaises(ValueError):
                subset_structure(bad_k)
        with self.assertRaises(ValueError):
            champion_log_probs(torch.zeros(2, 10, 4))

    def test_candidate_permutation_equivariance(self):
        k, batch = 5, 2
        structure = subset_structure(k)
        logits = torch.randn(batch, len(structure['state_ids']), k,
                             dtype=torch.float64, generator=self.generator, requires_grad=True)
        permutation = torch.tensor([3, 0, 4, 1, 2])
        old_ids = (structure['masks'].long() * (1 << permutation)).sum(-1)
        old_rows = torch.searchsorted(structure['state_ids'], old_ids)
        changed = logits[:, old_rows][:, :, permutation]
        original = champion_log_probs(logits)
        permuted = champion_log_probs(changed)
        torch.testing.assert_close(permuted, original[:, permutation], rtol=1e-12, atol=1e-12)
        weights = torch.randn(batch, k, dtype=torch.float64, generator=self.generator)
        gradient = torch.autograd.grad((original * weights).sum(), logits)[0]
        changed_gradient = torch.autograd.grad((permuted * weights[:, permutation]).sum(), changed)[0]
        torch.testing.assert_close(changed_gradient, gradient[:, old_rows][:, :, permutation],
                                   rtol=1e-11, atol=1e-12)
        order = weights.argsort(-1)
        changed_order = permutation.argsort()[order]
        torch.testing.assert_close(full_ranking_nll(changed, changed_order),
                                   full_ranking_nll(logits, order), rtol=1e-12, atol=1e-12)

    def test_fp64_native_ties_and_full_ranking_nll(self):
        costs = torch.tensor([[1. + 2e-9, 1., 1. + 1e-9, 1. + 3e-9],
                              [2., 2., 2. + 1e-9, 2.],
                              [3. + 1e-9, 3., 3., 3. + 1e-9],
                              [5., 5., 5., 5.]], dtype=torch.float64)
        winner, ids = torch.tensor([1, 0, 2, 2]), torch.tensor([40, 10, 30, 20])
        expected_order = torch.tensor([[1, 2, 0, 3], [0, 1, 3, 2], [2, 1, 3, 0], [2, 1, 3, 0]])
        order = stable_cost_order(costs, winner, ids)
        self.assertTrue(torch.equal(order, expected_order))
        self.assertTrue((costs[0].float() == 1.).all())
        with self.assertRaisesRegex(ValueError, 'strictly above'):
            stable_cost_order(costs[:1], torch.tensor([2]), ids)
        with self.assertRaisesRegex(ValueError, 'FP64'):
            stable_cost_order(costs.float(), winner, ids)
        permutation = torch.tensor([2, 0, 3, 1])
        inverse = permutation.argsort()
        self.assertTrue(torch.equal(stable_cost_order(costs[:, permutation], inverse[winner],
                                                     ids[permutation]), inverse[order]))
        structure = subset_structure(4)
        logits = torch.randn(4, len(structure['state_ids']), 4,
                             dtype=torch.float64, generator=self.generator)
        logits = logits.masked_fill(~structure['masks'], torch.nan).requires_grad_()
        rows = {state: row for row, state in enumerate(structure['state_ids'].tolist())}
        expected_loss = []
        for b in range(4):
            remaining, terms = 15, []
            for removed in order[b].flip(0)[:-1].tolist():
                row = rows[remaining]
                legal = [j for j in range(4) if remaining & (1 << j)]
                terms.append(logits[b, row, legal].logsumexp(-1) - logits[b, row, removed])
                remaining ^= 1 << removed
            self.assertEqual(remaining, 1 << winner[b].item())
            expected_loss.append(torch.stack(terms).mean())
        expected_loss = torch.stack(expected_loss)
        loss = full_ranking_nll(logits, order)
        self.assertEqual(loss.shape, (4,))
        torch.testing.assert_close(loss, expected_loss, rtol=1e-12, atol=1e-12)
        gradient = torch.autograd.grad(loss.mean(), logits, retain_graph=True)[0]
        reference_gradient = torch.autograd.grad(expected_loss.mean(), logits)[0]
        torch.testing.assert_close(gradient, reference_gradient, rtol=1e-12, atol=1e-12)
        loss.mean().backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertTrue((logits.grad.masked_select(~structure['masks']) == 0).all())
        uniform = full_ranking_nll(torch.zeros_like(logits), order)
        torch.testing.assert_close(uniform, torch.full_like(uniform, math.log(24.) / 3.))
        singleton = torch.empty(3, 0, 1, requires_grad=True)
        only_order = stable_cost_order(torch.ones(3, 1, dtype=torch.float64),
                                       torch.zeros(3, dtype=torch.long), torch.tensor([40]))
        empty_loss = full_ranking_nll(singleton, only_order)
        torch.testing.assert_close(empty_loss, torch.zeros(3), rtol=0, atol=0)
        empty_loss.sum().backward()
        self.assertEqual(singleton.grad.shape, singleton.shape)
        with self.assertRaisesRegex(ValueError, 'permutation'):
            full_ranking_nll(torch.zeros_like(logits), torch.zeros_like(order))


if __name__ == '__main__':
    unittest.main()

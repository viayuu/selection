"""Two synthetic checks; the third check is the run's real first TRAIN update."""

import unittest
import numpy as np
import torch

from .r78_frozen_score_cost_head import (
    BATCH, UPDATES, batches, head, head_order, learning_rate, masked_softmax_fp32, objective, train_targets,
)


class FrozenCostHeadChecks(unittest.TestCase):
    def test_masked_FP32_probability_and_exact_identity_ties(self):
        logits = np.array([[1e-10, 0.], [0., 0.], [3., -np.inf]], dtype=np.float32)
        mask = ~np.isneginf(logits)
        p, _ = masked_softmax_fp32(logits, mask)
        self.assertEqual(p.dtype, np.float32)
        self.assertEqual(p[2, 1], 0.)
        g = head(torch.from_numpy(p), torch.zeros(2, 2, dtype=torch.float64)).numpy()
        np.testing.assert_array_equal(head_order(g, logits, mask, np.array([17, 3]))[:, 0], [0, 1, 0])

    def test_FP64_target_unit_prior_schedule_and_natural_tail_fill(self):
        costs = np.array([[1., 1. + 1e-10], [1., 2.]], dtype=np.float64)
        mask = np.ones_like(costs, dtype=bool)
        target, a = train_targets(costs, mask)
        self.assertEqual(target.dtype, np.float64)
        self.assertGreater(target[0, 1], 0.)
        self.assertAlmostEqual(a, ((costs[:, 1] - 1) / 2).mean(), places=14)
        p = torch.tensor([[.4, .6], [.2, .8]], dtype=torch.float32)
        d = torch.full((2, 2), .2, dtype=torch.float64)
        expected = (head(p, d) - torch.from_numpy(target)).square().mean() + d.square().sum() / 4
        self.assertEqual(float(objective(p, torch.from_numpy(target), torch.from_numpy(mask), d, 2)), float(expected))
        self.assertEqual(learning_rate(0), .01)
        self.assertEqual(learning_rate(UPDATES), 0.)
        stream, rng = batches(1025, 2, BATCH), np.random.default_rng(2)
        first, second = rng.permutation(1025), rng.permutation(1025)
        np.testing.assert_array_equal(next(stream), first[:BATCH])
        np.testing.assert_array_equal(next(stream), np.concatenate([first[BATCH:], second[:BATCH - 1]]))


if __name__ == '__main__':
    unittest.main()

import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from .multitask_probe import batch_schedule, state_hash
from .tsp_learnability import (bucket_rows, fit_args, fit_passed, fresh_model,
                               gap_and_regret, raw_tsp, size_boundaries, stratified_indices)
from .train import selector_loss


class LearnabilityTests(unittest.TestCase):
    def test_gap_is_not_actual_regret_and_keeps_raw_precision(self):
        costs = np.array([[10., 10.00000001, 12.], [1., 1., 2.]], dtype=np.float64)
        gap, regret = gap_and_regret(costs, np.array([2, 1]))
        self.assertGreater(gap[0], 0.)
        self.assertLess(gap[0], .01)
        np.testing.assert_allclose(regret, [20., 0.])
        self.assertEqual(gap[1], 0.)
        self.assertEqual(costs.astype(np.float32)[0, 0], costs.astype(np.float32)[0, 1])

    def test_buckets_include_right_boundary_and_account_for_all_regret(self):
        gap = np.array([0., .001, .01, .1, .5, 1.])
        groups = np.searchsorted([0, .01, .1, .5], gap, side="left")
        np.testing.assert_array_equal(groups, [0, 1, 1, 2, 3, 4])
        rows = bucket_rows("val", groups, list("abcde"), np.array([1, 0, 1, 0, 0, 1]), np.arange(6.), True)
        self.assertEqual(sum(r["n"] for r in rows), 6)
        self.assertAlmostEqual(sum(r["share_of_total_regret"] for r in rows), 1.)

    def test_repeated_size_boundaries_merge_without_empty_last_bucket(self):
        sizes = np.array([50] * 8 + [100] * 8)
        np.testing.assert_array_equal(size_boundaries(sizes), [50.])
        np.testing.assert_array_equal(size_boundaries(np.full(10, 50)), [])

    def test_stratified_subsets_are_distinct_paired_and_nested(self):
        winner = np.repeat(np.arange(8), 30)
        groups = np.tile(np.arange(4), 60)
        a = stratified_indices(winner, groups, 128, 2)
        b = stratified_indices(winner, groups, 32, 2, available=a)
        self.assertEqual(len(np.unique(a)), 128)
        self.assertTrue(set(b) <= set(a))
        np.testing.assert_array_equal(np.bincount(winner[a]), [16] * 8)
        np.testing.assert_array_equal(np.bincount(winner[b]), [4] * 8)
        np.testing.assert_array_equal(a, stratified_indices(winner, groups, 128, 2))
        self.assertEqual(set(groups[b]), {0, 1, 2, 3})
        self.assertTrue(torch.equal(batch_schedule(128, 32, 3000, 2), batch_schedule(128, 32, 3000, 2)))

    def test_fresh_optimizer_preserves_weights_but_not_old_moments(self):
        source = nn.Linear(4, 2)
        optimizer = torch.optim.AdamW(source.parameters())
        source(torch.ones(2, 4)).sum().backward()
        optimizer.step()
        checkpoint = dict(model=copy.deepcopy(source.state_dict()), optimizer=optimizer.state_dict(),
                          args={"model_params": {"dropout": .1}})
        with patch("code.V4.tsp_learnability.make_selector", side_effect=lambda p: nn.Linear(4, 2)):
            a, opt_a, params = fresh_model(checkpoint, "cpu", 2e-4)
            b, opt_b, _ = fresh_model(checkpoint, "cpu", 2e-4)
        self.assertEqual(state_hash(a.state_dict()), state_hash(checkpoint["model"]))
        self.assertEqual(state_hash(a.state_dict()), state_hash(b.state_dict()))
        self.assertFalse(opt_a.state)
        self.assertFalse(opt_b.state)
        self.assertEqual(params["dropout"], 0.)
        self.assertEqual(checkpoint["args"]["model_params"]["dropout"], .1)
        self.assertEqual(opt_a.param_groups[0]["weight_decay"], 0.)
        self.assertEqual(opt_a.param_groups[0]["lr"], 2e-4)

    def test_loss_and_pass_rule_use_unweighted_ce(self):
        logits = torch.tensor([[3., 0.], [0., 3.]], requires_grad=True)
        costs, winner = torch.tensor([[1., 2.], [2., 1.]]), torch.tensor([0, 1])
        ce, parts = selector_loss(dict(logits=logits), costs, None, fit_args("ce"), winner=winner)
        self.assertAlmostEqual(float(ce), .35 * float(parts["ce"]), places=6)
        wc, parts = selector_loss(dict(logits=logits), costs, None, fit_args("winner_cost"), winner=winner)
        self.assertAlmostEqual(float(wc), .35 * float(parts["ce"]) + .1 * float(parts["pair"]) + .02 * float(parts["risk"]), places=6)
        self.assertFalse(fit_passed(dict(top1=1., ce=.06)))
        self.assertFalse(fit_passed(dict(top1=.98, ce=.01)))
        self.assertTrue(fit_passed(dict(top1=.99, ce=.05)))

    def test_test_split_is_rejected_before_reading(self):
        with self.assertRaises(ValueError):
            raw_tsp("test")


if __name__ == "__main__":
    unittest.main()

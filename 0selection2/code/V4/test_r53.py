"""Focused R53 label, cost sensitivity, fixed gate, and leakage checks."""

import unittest

import numpy as np
import torch

from .pair_specialist import (PAIR, PairSpecialist, apply_specialist, binary_objective,
                             gate_mask, ordinal_scores, pair_targets, stable_order)
from .performance_evaluation import decision_metrics
from .r42_experiment import CostController, paired_schedule
from .multitask_probe import state_hash


class R53Tests(unittest.TestCase):
    def test_fp64_label_direction_and_exact_ties(self):
        cost = np.array([[1., 1.+1e-10, .9], [2., 1., .8], [1., 1., .7]], dtype=np.float64)
        y, w = pair_targets(cost, [*PAIR, 'other'])
        np.testing.assert_array_equal(y, [1, 0, -1])
        np.testing.assert_allclose(w, [1e-10/.9, 1/.8, 0], rtol=1e-6)
        with self.assertRaises(ValueError):
            pair_targets(cost.astype(np.float32), [*PAIR, 'other'])

    def test_fixed_weight_mean_not_batch_mean(self):
        z = torch.zeros(2, requires_grad=True)
        y, w = torch.tensor([1, 0]), torch.tensor([.01, .03])
        loss = binary_objective(z, y, w, True, .01)
        torch.testing.assert_close(loss, torch.tensor(2*np.log(2), dtype=torch.float32))
        loss.backward()
        self.assertGreater(abs(z.grad[1]), abs(z.grad[0]))
        with self.assertRaises(ValueError):
            binary_objective(z, torch.tensor([-1, 0]), w, False, .01)

    def test_gate_uses_only_original_two_names(self):
        pool = ['other', *PAIR]
        order = np.array([[1, 2, 0], [2, 1, 0], [0, 1, 2], [1, 0, 2]])
        result, gate = apply_specialist(order, pool, np.array([-1., 1., 100., -100.]))
        np.testing.assert_array_equal(gate, [True, True, False, False])
        np.testing.assert_array_equal(result, [[2, 1, 0], [1, 2, 0], [0, 1, 2], [1, 0, 2]])
        np.testing.assert_array_equal(result[:, 2:], order[:, 2:])

    def test_ties_in_baseline_scores_can_be_reordered(self):
        order = stable_order(np.zeros((1, 3)), [16, 17, 18])
        new, _ = apply_specialist(order, [*PAIR, 'other'], np.array([-1.]))
        np.testing.assert_array_equal(new, [[1, 0, 2]])
        zero, _ = apply_specialist(new, [*PAIR, 'other'], np.array([0.]))
        self.assertEqual(int(zero[0, 0]), 0)

    def test_candidate_permutation(self):
        pool, ids = [*PAIR, 'other'], np.array([16, 17, 19])
        scores = np.array([[3., 2., 1.], [1., 2., 3.]])
        expected, _ = apply_specialist(stable_order(scores, ids), pool, [-1., 1.])
        perm = np.array([2, 0, 1])
        result, _ = apply_specialist(stable_order(scores[:, perm], ids[perm]), [pool[i] for i in perm], [-1., 1.])
        np.testing.assert_array_equal(perm[result], expected)

    def test_unavailable_pair_does_not_trigger(self):
        order = np.array([[1, 0]])
        result, gate = apply_specialist(order, ['a', 'b'], None)
        np.testing.assert_array_equal(result, order)
        self.assertFalse(gate.any())

    def test_final_metrics_replay_from_ranking(self):
        raw = dict(costs=np.array([[1., 2., 3.], [2., 1., 3.]], dtype=np.float64),
                   winner=np.array([0, 1]), pool=[*PAIR, 'other'])
        order, _ = apply_specialist(np.array([[1, 0, 2], [0, 1, 2]]), raw['pool'], [1., -1.])
        m, pred = decision_metrics(ordinal_scores(order), raw)
        self.assertEqual(m['top1'], 1.)
        self.assertEqual(m['actual_regret_pct'], 0.)
        np.testing.assert_array_equal(pred, raw['winner'])

    def test_sampling_plan_full_coverage_and_initialization(self):
        sizes = dict(OVRP=259, VRPTW=131)
        first, second = paired_schedule(sizes, 2, 128, 2), paired_schedule(sizes, 2, 128, 2)
        self.assertEqual(state_hash(first), state_hash(second))
        for permutation in first['permutations']:
            for p, size in sizes.items():
                torch.testing.assert_close(permutation[p].sort().values, torch.arange(size))

    def test_controller_uses_integrated_cost_not_binary_accuracy(self):
        controller = CostController(min_epochs=15)
        controller.observe(.9, 1)
        controller.observe(.8, 2)
        self.assertEqual(controller.effective_best, .8)
        for epoch in range(3, 16):
            halve, stop = controller.observe(.81, epoch)
        self.assertTrue(stop)

    def test_input_whitelist_excludes_every_supervision_field(self):
        from .pair_specialist import INPUT_FIELDS
        for forbidden in ('costs', 'ind', 'binary_label', 'cost_weight', 'pool_ids', 'solver_mask', 'base_id'):
            self.assertNotIn(forbidden, INPUT_FIELDS)


if __name__ == '__main__':
    unittest.main()

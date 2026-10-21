"""Small numerical witnesses for the no-training VAL catalog diagnostic."""

import unittest

import numpy as np

from .validation_learning_map import overlap, ranked


class ValidationLearningMapTests(unittest.TestCase):
    def test_global_solver_id_ties_and_descending_score(self):
        np.testing.assert_array_equal(ranked(np.array([[1., 1., 0.], [0., 3., 3.]]), np.array([9, 2, 5])),
                                      np.array([[1, 0, 2], [1, 2, 0]]))

    def test_common_misses_cost_budget_and_native_zero_cost_tie(self):
        costs = np.array([[1., 2.], [2., 1.], [1., 1.], [1., 3.]], dtype=np.float64)
        predictions = np.array([[1, 0, 1, 1], [0, 0, 1, 1]])
        summary, correct, regret, selected = overlap(predictions, costs, costs.argmin(1), 0)
        self.assertEqual(summary['reference_errors'], 4)
        self.assertEqual(summary['common_misses'], 3)
        self.assertEqual(summary['common_misses_with_zero_reference_regret'], 1)
        self.assertEqual(summary['any_model_correct_count'], 1)
        self.assertAlmostEqual(summary['reference_regret_share_common_misses_pct'], 75.)
        self.assertEqual(summary['reference_actual_regret_pct'], 100.)

    def test_non_fp64_mismatched_and_non_native_labels_rejected(self):
        costs = np.array([[1., 2.], [2., 1.]], dtype=np.float64)
        for changed_costs, winner in [(costs.astype(np.float32), costs.argmin(1)),
                                      (costs, np.array([1, 1]))]:
            with self.assertRaises(ValueError):
                overlap(np.array([[1, 1]]), changed_costs, winner, 0)


if __name__ == '__main__':
    unittest.main()

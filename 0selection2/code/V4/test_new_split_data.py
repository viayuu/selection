"""Small CPU checks for candidate filtering and derived target identities."""

import unittest

import numpy as np

from .export_new_split_data import B_PROBLEMS, derive_targets, filtered_pool, train_baselines


class NewSplitTargetsTest(unittest.TestCase):
    def setUp(self):
        self.pool = ["MTPOMO", "MVMOE", "MoSES_CaDA", "MoSES_RF",
                     "RELD_MOEL", "RELD_MTL", "RouteFinder"]

    def test_filter_all_eight_b_tasks_only(self):
        for problem in B_PROBLEMS:
            methods, kept = filtered_pool(problem, self.pool)
            self.assertEqual(methods, ["MTPOMO", "MVMOE", "RELD_MOEL", "RELD_MTL"])
            self.assertEqual(kept, [0, 1, 4, 5])
        for problem in ["OVRP", "VRPL", "VRPTW", "OVRPTW", "OVRPL", "VRPLTW", "OVRPLTW"]:
            self.assertEqual(filtered_pool(problem, self.pool), (self.pool, list(range(7))))

    def test_removed_winner_is_recomputed(self):
        _, kept = filtered_pool("VRPB", self.pool)
        original = np.asarray([[7., 6., 1., 2., 5., 4., 3.]])
        target = derive_targets(original[:, kept])
        np.testing.assert_array_equal(target["winner"], [3])
        np.testing.assert_array_equal(target["ranking"], [[3, 2, 1, 0]])
        np.testing.assert_array_equal(target["rank"], [[3, 2, 1, 0]])
        np.testing.assert_array_equal(target["oracle_cost"], [4.])

    def test_fp64_decisions_survive_fp32_rounding(self):
        costs = np.asarray([[1.00000002, 1.00000001, 1.00000001]])
        target = derive_targets(costs)
        self.assertEqual(int(costs.astype(np.float32).argmin()), 0)
        np.testing.assert_array_equal(target["winner"], [1])
        np.testing.assert_array_equal(target["ranking"], [[1, 2, 0]])
        np.testing.assert_array_equal(target["rank"], [[2, 0, 1]])

    def test_relative_regret_and_oracle_identity(self):
        target = derive_targets([[3., 2., 4.], [2., 4., 2.]])
        np.testing.assert_allclose(target["relative_regret"], [[.5, 0., 1.], [0., 1., 0.]])
        np.testing.assert_allclose(target["gap_pct"], target["relative_regret"] * 100)
        np.testing.assert_array_equal(target["winner"], [1, 0])

    def test_sbs_is_selected_from_train_only(self):
        baseline = train_baselines(np.asarray([[1., 3.], [2., 4.]]), ["MTPOMO", "MVMOE"])
        self.assertEqual(baseline["sbs_solver"], "MTPOMO")
        self.assertEqual(baseline["sbs_slot"], 0)
        self.assertEqual(baseline["selection_split"], "train")

    def test_reject_invalid_costs(self):
        for values in [[[0., 1.]], [[np.nan, 1.]], [[np.inf, 1.]]]:
            with self.assertRaises(ValueError):
                derive_targets(values)


if __name__ == "__main__":
    unittest.main()

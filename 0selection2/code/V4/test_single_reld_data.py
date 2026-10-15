"""Focused CPU checks for a single, aliased ReLD deployment."""

import unittest

import numpy as np

from code.unified_selector.registry import GLOBAL_SOLVERS, S2I
from .export_new_split_data import derive_targets, train_baselines
from .export_single_reld_data import ALIAS, REMOVED, RETAINED, single_reld_pool
from .new_split_data import scenario_solver_ids


class SingleReLDTest(unittest.TestCase):
    def test_b_pool_has_three_candidates(self):
        methods, kept, ids = single_reld_pool(["MTPOMO", "MVMOE", RETAINED, REMOVED])
        self.assertEqual(methods, ["MTPOMO", "MVMOE", ALIAS])
        self.assertEqual(kept, [0, 1, 2])
        self.assertEqual(ids, [S2I["MTPOMO"], S2I["MVMOE"], S2I[RETAINED]])

    def test_other_mvrp_pool_retains_original_column_order(self):
        pool = ["MTPOMO", "MVMOE", "MoSES_CaDA", "MoSES_RF", RETAINED, REMOVED, "RouteFinder"]
        methods, kept, ids = single_reld_pool(pool)
        self.assertEqual(methods, ["MTPOMO", "MVMOE", "MoSES_CaDA", "MoSES_RF", ALIAS, "RouteFinder"])
        self.assertEqual(kept, [0, 1, 2, 3, 4, 6])
        self.assertEqual(ids, sorted(ids))
        self.assertNotEqual(methods, sorted(methods))

    def test_no_reld_pool_is_unchanged(self):
        pool = ["BQ", "ELG", "LEHD"]
        methods, kept, _ = single_reld_pool(pool)
        self.assertEqual(methods, pool)
        self.assertEqual(kept, [0, 1, 2])

    def test_costs_are_not_best_of_two(self):
        _, kept, _ = single_reld_pool(["MTPOMO", "MVMOE", RETAINED, REMOVED])
        old = np.asarray([[9., 8., 7., 1.], [6., 5., 2., 3.]], dtype=np.float64)
        targets = derive_targets(old[:, kept])
        np.testing.assert_array_equal(targets["oracle_cost"], [7., 2.])
        np.testing.assert_array_equal(targets["costs"][:, 2], old[:, 2])
        np.testing.assert_array_equal(targets["winner"], [2, 2])
        self.assertEqual(train_baselines(targets["costs"], ["MTPOMO", "MVMOE", ALIAS])["sbs_solver"], ALIAS)

    def test_alias_uses_moel_id_and_legacy_manifest_still_loads(self):
        legacy = dict(global_solvers=GLOBAL_SOLVERS, global_solver_ids=S2I)
        self.assertEqual(scenario_solver_ids(legacy), S2I)
        names = [ALIAS if name == RETAINED else name for name in GLOBAL_SOLVERS if name != REMOVED]
        manifest = dict(global_solvers=names, solver_aliases={ALIAS: RETAINED})
        ids = scenario_solver_ids(manifest)
        self.assertEqual(ids[ALIAS], S2I[RETAINED])
        self.assertNotIn(REMOVED, ids)
        manifest["global_solver_ids"] = dict(ids, **{ALIAS: S2I[REMOVED]})
        with self.assertRaises(ValueError):
            scenario_solver_ids(manifest)

    def test_refuse_missing_moel(self):
        with self.assertRaises(ValueError):
            single_reld_pool(["MTPOMO", REMOVED])


if __name__ == "__main__":
    unittest.main()

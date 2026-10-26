"""Two small CPU-only, synthetic native-input contract checks."""

import copy
import unittest
from unittest.mock import patch

import numpy as np
import torch

from . import r115_features as r
from .r114_response_data import apply_permutation, validate_intervention
from .r92_tree_features import COLUMNS


def native_item(problem):
    xy = [[0.10, 0.05], [0.15, 0.07], [0.20, 0.09], [0.25, 0.11]]
    demand = [0.1, 0.2, 0.3, 0.4]
    if problem == "CVRP":
        return dict(depot=[[[0., 0.]]], loc=[xy], demand=[demand])
    return dict(depot_xy=[[0., 0.]], node_xy=xy, node_demand=demand, capacity=1.,
                route_limit=3., service_time=[0.1] * 4,
                tw_start=[0.6, 0.8, 1., 1.2], tw_end=[1.2, 1.4, 1.6, 1.8])


class NativeFeatureTests(unittest.TestCase):
    def test_legal_swaps_preserve_input_geometry_but_change_correlations(self):
        self.assertEqual(r.COLUMNS, COLUMNS)
        self.assertEqual(len(COLUMNS), 357)
        for problem, family, column in (
                ("CVRP", "demand", "corr_x_demand"),
                ("VRPLTW", "demand", "corr_x_demand"),
                ("VRPLTW", "tw_center", "corr_tw_start_radius_depot")):
            with self.subTest(problem=problem, family=family):
                item = native_item(problem)
                saved = copy.deepcopy(item)
                changed = apply_permutation(problem, item, family, [3, 2, 1, 0])
                self.assertTrue(validate_intervention(problem, item, changed, family)["legal"])
                with patch.object(r, "features", wraps=r.features) as call:
                    before = r.native_phi(problem, item)
                batch = call.call_args.args[0]
                self.assertTrue(all(v.device.type == "cpu" for v in batch.values()
                                    if isinstance(v, torch.Tensor)))
                np.testing.assert_array_equal(batch["node"][0, 0, 2:].numpy(),
                                              [0, 1] + ([] if problem == "CVRP" else [0] * 4))
                self.assertTrue(batch["node_mask"].all())
                after = r.native_phi(problem, changed)
                self.assertEqual(before.shape, (357,))
                self.assertEqual(before.dtype, np.float32)
                self.assertTrue(np.isfinite(before).all() and np.isfinite(after).all())
                self.assertEqual(before[COLUMNS.index("n")], 5)
                self.assertEqual(before[COLUMNS.index("d_in")], 4 if problem == "CVRP" else 8)
                self.assertEqual(before[COLUMNS.index("coord_dist")], 0)
                self.assertGreater(abs(float(before[COLUMNS.index(column)] -
                                             after[COLUMNS.index(column)])), 1e-4)
                geometric = [i for i, name in enumerate(COLUMNS) if name.startswith("geom_")]
                np.testing.assert_array_equal(before[geometric], after[geometric])
                self.assertEqual(item, saved)

    def test_hidden_labels_costs_winners_and_ids_rejected_before_features(self):
        with patch.object(r, "features", side_effect=AssertionError("must not extract features")):
            for problem in ("CVRP", "VRPLTW"):
                item = native_item(problem)
                for key in ("label", "labels", "cost", "costs", "winner", "ind", "id",
                            "base_id", "endpoint_id", "source_index", "coord_dist"):
                    with self.subTest(problem=problem, key=key):
                        with self.assertRaisesRegex(ValueError, "Unexpected native fields"):
                            r.native_phi(problem, dict(item, **{key: object()}))


if __name__ == "__main__":
    unittest.main()

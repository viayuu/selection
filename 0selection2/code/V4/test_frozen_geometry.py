import copy
import unittest

import numpy as np
import torch

from .geometry_probe import initialize, metrics_from_logits, original_state
from .multitask_probe import state_hash, update_batch
from .report_frozen_geometry import decision_changes
from .test_dual_stream import params
from .test_solver_features import sample_batch
from .tsp_learnability import fit_args
from .V4Model import make_selector


class FrozenGeometryTests(unittest.TestCase):
    def test_frozen_parameters_stay_equal_and_branch_receives_gradient(self):
        checkpoint = dict(model=copy.deepcopy(make_selector(params()).state_dict()), args=dict(model_params=params()))
        stats = dict(mean=[0.] * 12, std=[1.] * 12)
        model, optimizer, scaler, _ = initialize(checkpoint, "real", stats, 2, "cpu", freeze_base=True)
        before = state_hash(original_state(model))
        module = model.instance_encoder.geometry_residual
        trainable = {id(p) for p in model.parameters() if p.requires_grad}
        self.assertEqual(trainable, {id(p) for p in module.parameters()})
        self.assertEqual(len(optimizer.param_groups), 1)
        self.assertEqual(optimizer.param_groups[0]["name"], "geometry")
        self.assertFalse(optimizer.state)
        batch = sample_batch("TSP")
        batch["ind"] = batch["costs"].argmin(1)
        batch["node_geom"] = torch.randn(*batch["node"].shape[:2], 12)
        model.train()
        for _ in range(2):
            update_batch(model, optimizer, scaler, batch, fit_args("winner_cost"))
        self.assertEqual(state_hash(original_state(model)), before)
        self.assertTrue(model.training)
        for p in module.parameters():
            self.assertIsNotNone(p.grad)
            self.assertGreater(float(p.grad.norm()), 0)
        self.assertTrue(all(p.grad is None for p in model.parameters() if not p.requires_grad))

    def test_expected_and_argmax_regret_are_distinct(self):
        raw = dict(winner=np.array([0, 0]), costs=np.array([[1., 2.], [2., 4.]]))
        # Both hard decisions are correct, while half of the probability has cost.
        metrics, _ = metrics_from_logits(torch.tensor([[.001, 0.], [.001, 0.]]), raw)
        self.assertEqual(metrics["actual_regret_pct"], 0.)
        self.assertAlmostEqual(metrics["expected_regret_pct"], 49.975, places=3)

    def test_change_categories_and_cost_accounting(self):
        raw = dict(winner=np.array([0, 0, 0, 0, 0]), nodes=np.array([10] * 5),
                   costs=np.array([[1., 2., 4.]] * 5))
        base = dict(pred=np.array([1, 0, 1, 1, 0]), logits=np.array([[0., 1., 0.]] * 5))
        new = dict(pred=np.array([0, 1, 2, 1, 0]), logits=np.array([[0., 1., 0.]] * 5))
        rows, _, instances = decision_changes(base, new, raw, np.array([10]), 2)
        total = [r for r in rows if r["dimension"] == "ALL"]
        self.assertEqual([r["n"] for r in total], [1] * 5)
        self.assertAlmostEqual(sum(r["contribution_mean_cost"] for r in total), .4)
        self.assertEqual(sum(r["net_correct"] for r in total), 0)
        np.testing.assert_array_equal(instances["category"],
                                      ["corrected", "harmed", "wrong_switched", "wrong_unchanged", "correct_unchanged"])


if __name__ == "__main__":
    unittest.main()

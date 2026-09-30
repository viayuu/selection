import copy
import unittest

import torch
import torch.nn.functional as F

from .V4Model import make_selector
from .local_geometry import FIELDS, local_geometry
from .multitask_probe import gather_batch
from .tensor_loader import TensorBatchLoader
from .test_dual_stream import params
from .test_solver_features import sample_batch


class GeometryTests(unittest.TestCase):
    def test_field_order_and_degenerate_clouds(self):
        self.assertEqual(len(FIELDS), 12)
        self.assertEqual(FIELDS[:4], ("k4_radius", "k4_mean_distance", "k4_distance_cv", "k4_anisotropy"))
        for n in (1, 2, 3, 8, 20):
            xy = torch.zeros(2, n, 2)
            mask = torch.ones(2, n, dtype=torch.bool)
            mask[1] = False
            values = local_geometry(xy, mask)
            self.assertTrue(torch.isfinite(values).all())
            torch.testing.assert_close(values, torch.zeros_like(values))

    def test_k_boundary_includes_all_equal_distance_neighbors(self):
        xy = torch.tensor([[[0, 0], [3, 4], [3, -4], [-3, 4], [-3, -4],
                            [4, 3], [4, -3], [-4, 3], [-4, -3]]], dtype=torch.float64)
        values = local_geometry(xy, torch.ones(1, 9, dtype=torch.bool))
        torch.testing.assert_close(values[0, 0, :4], values[0, 0, 4:8], atol=1e-12, rtol=1e-12)
        self.assertAlmostEqual(float(values[0, 0, 2]), 0.)
        self.assertAlmostEqual(float(values[0, 0, 3]), 0.)

    def test_permutation_padding_and_rigid_transform_consistency(self):
        torch.manual_seed(3)
        xy = torch.rand(2, 23, 2, dtype=torch.float64)
        mask = torch.ones(2, 23, dtype=torch.bool)
        mask[1, 17:] = False
        expected = local_geometry(xy, mask)
        order = torch.randperm(23)
        torch.testing.assert_close(local_geometry(xy[:, order], mask[:, order]), expected[:, order], atol=1e-10, rtol=1e-10)
        padded = local_geometry(F.pad(xy, (0, 0, 0, 7), value=999), F.pad(mask, (0, 7)))
        torch.testing.assert_close(padded[:, :23], expected, atol=1e-10, rtol=1e-10)
        torch.testing.assert_close(padded[:, 23:], torch.zeros_like(padded[:, 23:]))
        rotation = torch.tensor([[.6, -.8], [.8, .6]], dtype=torch.float64)
        torch.testing.assert_close(local_geometry(3 * (xy @ rotation) + 8, mask), expected, atol=1e-9, rtol=1e-9)

    def test_zero_initialization_preserves_base_and_labels_are_not_input(self):
        torch.manual_seed(2)
        base = make_selector(params()).eval()
        for mode in ("zero", "real"):
            modified = make_selector(dict(params(), local_geometry=True, geometry_mode=mode)).eval()
            missing, unexpected = modified.load_state_dict(base.state_dict(), strict=False)
            self.assertFalse(unexpected)
            self.assertTrue(all("geometry_residual" in k for k in missing))
            batch = sample_batch("TSP")
            expected = base(batch)["logits"]
            del batch["costs"]
            torch.testing.assert_close(modified(batch)["logits"], expected, atol=0, rtol=0)
            batch["node_geom"] = torch.randn(2, 5, 12)
            torch.testing.assert_close(modified(batch)["logits"], expected, atol=0, rtol=0)

    def test_nonzero_branch_respects_padding_and_node_order(self):
        modified = make_selector(dict(params(), local_geometry=True, geometry_mode="real")).eval()
        with torch.no_grad():
            modified.instance_encoder.geometry_residual.projection[-1].weight.normal_(0, .02)
        batch = sample_batch("TSP")
        batch["node_geom"] = local_geometry(batch["node"][..., :2], batch["node_mask"]).float()
        expected = modified(batch)["logits"]
        padded = dict(batch, node=F.pad(batch["node"], (0, 0, 0, 4), value=100),
                      node_geom=F.pad(batch["node_geom"], (0, 0, 0, 4), value=100),
                      node_mask=F.pad(batch["node_mask"], (0, 4)))
        torch.testing.assert_close(modified(padded)["logits"], expected, atol=2e-6, rtol=2e-5)
        order = torch.tensor([4, 1, 3, 0, 2])
        permuted = dict(batch, node=batch["node"][:, order], node_geom=batch["node_geom"][:, order],
                        node_mask=batch["node_mask"][:, order])
        torch.testing.assert_close(modified(permuted)["logits"], expected, atol=2e-6, rtol=2e-5)

    def test_cache_gather_crops_geometry_with_nodes(self):
        batch = dict(node=torch.rand(4, 9, 8), node_geom=torch.rand(4, 9, 12),
                     node_mask=torch.ones(4, 9, dtype=torch.bool), n=torch.tensor([2, 9, 3, 5]),
                     pool_ids=torch.arange(8), costs=torch.ones(4, 8))
        loader = TensorBatchLoader(batch, 2, False, False)
        selected = gather_batch(loader, torch.tensor([0, 2]))
        self.assertEqual(selected["node_geom"].shape, (2, 3, 12))
        self.assertEqual(next(iter(loader))["node_geom"].shape, (2, 9, 12))

    def test_warmup_cosine_is_based_on_successful_updates(self):
        from .geometry_probe import lr_factor
        self.assertAlmostEqual(lr_factor(1, 1000), .02)
        self.assertAlmostEqual(lr_factor(50, 1000), 1.)
        self.assertAlmostEqual(lr_factor(1000, 1000), .1)
        self.assertGreater(lr_factor(51, 1000), lr_factor(500, 1000))


if __name__ == "__main__":
    unittest.main()

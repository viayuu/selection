"""Focused R51 feature semantics, isolation, initialization and gradient tests."""

import inspect
import unittest

import numpy as np
import torch

from .multitask_probe import state_hash
from .r48_common import binary_labels, binary_metrics
from .r50_readout import SolverEncoderReadout
from .r51_probe import (BEHAVIOR_DIM, FEATURE_NAMES, action_statistics, input_length_scale,
                        open_segment_distance, probe, summarize_pomo)
from .r51_readout import BehaviorReadout, common_state, copy_common_initialization
from .r51_experiment import inputs


class R51Tests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(2)
        self.static = BehaviorReadout().eval()
        self.enhanced = BehaviorReadout(True).eval()
        copy_common_initialization(self.static, self.enhanced)
        self.encoded = torch.randn(3, 6, 2, 128)
        self.raw = torch.randn(3, 6, 7)
        self.mask = torch.tensor([[1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1]], dtype=torch.bool)
        self.behavior = torch.randn(3, BEHAVIOR_DIM)

    def test_full_feature_definition(self):
        self.assertEqual(BEHAVIOR_DIM, 270)
        self.assertEqual(len(set(FEATURE_NAMES)), BEHAVIOR_DIM)
        self.assertIn('RELD_MOEL/step1/served_customer_fraction/mean', FEATURE_NAMES)
        self.assertIn('RELD_MTL/step10/open_distance_over_mean_distance/q90', FEATURE_NAMES)

    def test_static_is_exact_r50_readout(self):
        reference = SolverEncoderReadout().eval()
        reference.load_state_dict(self.static.state_dict(), strict=True)
        torch.testing.assert_close(self.static(self.encoded, self.raw, self.mask),
                                   reference(self.encoded, self.raw, self.mask), rtol=0, atol=0)

    def test_common_parameters_identical(self):
        self.assertEqual(state_hash(common_state(self.static)), state_hash(common_state(self.enhanced)))

    def test_open_route_excludes_return_to_depot(self):
        previous = torch.tensor([[[0., 0.], [3., 4.], [3., 4.]]])
        selected_xy = torch.tensor([[[3., 4.], [0., 0.], [6., 8.]]])
        selected = torch.tensor([[1, 0, 2]])
        torch.testing.assert_close(open_segment_distance(previous, selected_xy, selected), torch.tensor([[5., 0., 5.]]))

    def test_probability_entropy_and_margin(self):
        probabilities = torch.tensor([[[.5, .5, 0.], [1., 0., 0.], [.8, .2, 0.]]])
        mask = torch.tensor([[[0., 0., -torch.inf], [0., -torch.inf, -torch.inf], [0., 0., -torch.inf]]])
        entropy, margin = action_statistics(probabilities, mask)
        torch.testing.assert_close(entropy[0, :2], torch.tensor([1., 0.]))
        torch.testing.assert_close(margin, torch.tensor([[0., 1., .6]]))
        self.assertTrue(torch.isfinite(entropy).all())

    def test_input_scale_uses_only_input_distances(self):
        depot = torch.tensor([[[0., 0.]]])
        xy = torch.tensor([[[3., 0.], [0., 4.]]])
        fields = [torch.ones(1, 2) for _ in range(4)]
        torch.testing.assert_close(input_length_scale((depot, xy, *fields)), torch.tensor([4.]))
        torch.testing.assert_close(input_length_scale((depot * 2, xy * 2, *fields)), torch.tensor([8.]))

    def test_pomo_statistics_have_no_best_start_filter(self):
        values = torch.arange(10.).view(1, 5, 2)
        summary = summarize_pomo(values)
        torch.testing.assert_close(summary[:, :, 0], values.mean(1))
        torch.testing.assert_close(summary[:, :, 1], values.std(1, correction=0))
        self.assertEqual(summary.shape, (1, 2, 5))
        self.assertTrue(torch.isfinite(summarize_pomo(values[:, :1])).all())

    def test_padding_and_customer_order(self):
        expected = self.enhanced(self.encoded, self.raw, self.mask, self.behavior)
        encoded = torch.cat((self.encoded, torch.randn(3, 3, 2, 128) * 100), 1)
        raw = torch.cat((self.raw, torch.randn(3, 3, 7) * 100), 1)
        mask = torch.cat((self.mask, torch.zeros(3, 3, dtype=torch.bool)), 1)
        torch.testing.assert_close(expected, self.enhanced(encoded, raw, mask, self.behavior), rtol=1e-6, atol=1e-7)
        order = torch.tensor([0, 4, 2, 1, 5, 3])
        torch.testing.assert_close(expected, self.enhanced(self.encoded[:, order], self.raw[:, order], self.mask[:, order], self.behavior),
                                   rtol=1e-6, atol=1e-7)

    def test_no_costs_or_winner_in_probe_or_readout(self):
        signature = inspect.signature(probe)
        self.assertEqual(list(signature.parameters), ['model', 'module', 'batch', 'keep_witness'])
        cache = dict(encoded=self.encoded, raw_node=self.raw, node_mask=self.mask, behavior=self.behavior,
                     costs=torch.rand(3, 2), winner=torch.tensor([0, 1, 0]))
        expected = self.enhanced(*inputs(cache, [0, 1, 2], True))
        cache.update(costs=torch.full((3, 2), torch.nan), winner=torch.tensor([1, 0, 1]))
        torch.testing.assert_close(expected, self.enhanced(*inputs(cache, [0, 1, 2], True)), rtol=0, atol=0)

    def test_behavior_and_static_blocks_receive_gradients(self):
        self.enhanced.train()
        logits = self.enhanced(self.encoded, self.raw, self.mask, self.behavior)
        torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1, 0])).backward()
        for block in (self.enhanced.moel_norm, self.enhanced.mtl_norm, self.enhanced.node_mlp,
                      self.enhanced.behavior_mlp, self.enhanced.head):
            grads = [p.grad for p in block.parameters()]
            self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in grads))
            self.assertGreater(sum(float(g.abs().sum()) for g in grads), 0)
        self.assertIsNone(self.behavior.grad)
        self.assertNotIn('behavior_mean', dict(self.enhanced.named_parameters()))

    def test_behavior_required_only_by_b(self):
        with self.assertRaises(ValueError):
            self.enhanced(self.encoded, self.raw, self.mask)
        torch.testing.assert_close(self.static(self.encoded, self.raw, self.mask),
            self.static(self.encoded, self.raw, self.mask, self.behavior * 100), rtol=0, atol=0)

    def test_fp64_labels_and_saved_prediction_metrics(self):
        costs = np.array([[1., 1. + 1e-10], [2., 2.], [3.1, 3.]], dtype=np.float64)
        np.testing.assert_array_equal(binary_labels(costs), [0, -1, 1])
        metrics = binary_metrics(np.array([[1., 0.], [0., 1.], [0., 1.]]), costs)
        self.assertEqual(metrics['n'], 2)
        self.assertEqual(metrics['accuracy'], 1.)


if __name__ == '__main__':
    unittest.main()

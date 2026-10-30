"""Tiny CPU fixtures, never production labels or empirical benefit evidence."""

import unittest

import torch

from code.unified_selector.registry import P2I, constraint_bits, problem_descriptor
from .solver_features import get_solver_feature_spec
from . import r121_model as api


def params():
    return dict(embedding_dim=16, head_num=2, qkv_dim=8, ff_hidden_dim=32,
        encoder_layer_num=1, joint_layer_num=1, head_hidden_dim=24, dropout=0.,
        stats_dim=12, sdpa=True, solver_feature_hidden=16, solver_feature_weight=1.,
        solver_feature_spec=get_solver_feature_spec(), solver_representation='handcrafted',
        pair_mode='score_difference', local_geometry=False, ignore_coord_dist=True)


def batch(size=4):
    generator = torch.Generator().manual_seed(121)
    costs = torch.tensor([[1.02, 1.01, 1.03]], dtype=torch.float64).repeat(size, 1)
    return dict(problem_id=P2I['TSP'], kind='coord', node=torch.rand(size, 5, 2, generator=generator),
        node_mask=torch.ones(size, 5, dtype=torch.bool), n=torch.full((size,), 5), d_in=2,
        cbits=torch.as_tensor(constraint_bits('TSP')).float().repeat(size, 1),
        problem_desc=torch.as_tensor(problem_descriptor('TSP')).float().repeat(size, 1),
        coord_dist=torch.zeros(size, dtype=torch.long), pool_ids=torch.tensor([0, 3, 5]),
        solver_mask=torch.ones(size, 3, dtype=torch.bool), costs=costs, ind=costs.argmin(1))


class StructuralModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(2)
        self.model = api.make_model(params()).eval()

    def test_raw_forward_and_metadata_cost_invariance(self):
        data = batch()
        first = self.model(data)
        data.update(costs=torch.zeros_like(data['costs']), ind=torch.zeros_like(data['ind']),
            family_id='must not enter forward', axis_state={'answer': 100},
            current_cost=torch.full((4, 3), float('nan')))
        second = self.model(data)
        for key in ('logits', 'advantage_units', 'pair_margin'):
            torch.testing.assert_close(first[key], second[key], rtol=0, atol=0)
        torch.testing.assert_close(first['advantage_units'].sum(1), torch.zeros(4), atol=1e-6, rtol=0)

    def test_three_terms_backpropagate_into_same_query_encoder(self):
        data = batch()
        output = self.model(data)
        terms = api.family_losses(output, data)
        loss = api.natural_loss(output, data).mean() + sum(value.mean() for value in terms.values())
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(float(self.model.instance_encoder.coord_proj.weight.grad.abs().sum()), 0.)
        self.assertGreater(float(self.model.advantage_head[-1].weight.grad.abs().sum()), 0.)
        self.assertEqual(api.LOSS_WEIGHTS, dict(natural_ce=1., endpoint_huber=1., response_huber=1.))

    def test_potential_responses_telescope(self):
        potential = self.model(batch())['advantage_units']
        response = (potential[1] - potential[0]) + (potential[2] - potential[1]) + (potential[0] - potential[2])
        torch.testing.assert_close(response, torch.zeros(3), atol=1e-6, rtol=0)

    def test_native_target_units_and_no_silent_epsilon_or_fp32(self):
        costs = torch.tensor([[1., 2., 4.]], dtype=torch.float64)
        mask = torch.ones_like(costs, dtype=torch.bool)
        target = api.advantage_targets(costs, mask)
        torch.testing.assert_close(target, (costs.log() - costs.log().mean()) / .01)
        for invalid in (costs.float(), torch.zeros_like(costs)):
            with self.assertRaises(ValueError):
                api.advantage_targets(invalid, mask)

    def test_pairs_and_native_winners_required(self):
        data = batch(3)
        with self.assertRaises(ValueError):
            api.family_losses(self.model(data), data)
        data = batch()
        data['ind'][:] = 0
        with self.assertRaises(ValueError):
            api.natural_loss(self.model(data), data)


if __name__ == '__main__':
    unittest.main()

"""Bounded CPU checks for the boundary-preserving mechanism and its safety."""

import unittest

import torch

from code.unified_selector.registry import P2I, POOLS, S2I, constraint_bits, problem_descriptor
from . import r112_model as m
from .r77_data import parameter_hash
from .V4Model import get_default_model_params


# Same small fixtures previously re-exported by R109; no unpublished test dependency.
def full18_params():
    params = get_default_model_params()
    params.update(architecture='pairwise', joint_layer_num=2, local_geometry=True,
        geometry_mode='real', ignore_coord_dist=True, native_winner=True,
        decision_head='classification', pair_mode='score_difference',
        solver_representation='handcrafted', solver_feature_weight=1., sdpa=True)
    return params


def input_batch(problem):
    pool = [name for name in POOLS[problem] if name != 'RELD_MTL']
    if problem not in ('TSP', 'CVRP', 'ATSP') and 'B' in problem:
        pool = [name for name in pool if name not in ('RouteFinder', 'MoSES_RF', 'MoSES_CaDA')]
    generator = torch.Generator().manual_seed(107)
    node = torch.rand(2, 6, 8, generator=generator)
    mask = torch.tensor([[True, True, True, True, False, False], [True] * 6])
    node[..., 2] = torch.tensor([0., .3, -.4, .2, -.1, .5])
    node[..., 3:] = 0.
    if problem == 'TSP':
        node = node[..., :2].clone()
    elif problem != 'ATSP':
        node[:, 0, 3] = 1.
        bits = constraint_bits(problem)
        if bits[3]:
            node[:, 1:, 4] = 7.25
        if bits[4]:
            node[:, 1:, 5] = .37
            node[:, 1:, 6] = .5
            node[:, 1:, 7] = 4.75
    node = node.masked_fill(~mask[..., None], 0.)
    matrix = torch.rand(2, 6, 6, generator=generator) + .1
    matrix.diagonal(dim1=1, dim2=2).zero_()
    return dict(problem_id=P2I[problem], pool_ids=torch.tensor([S2I[name] for name in pool]),
        kind='matrix' if problem == 'ATSP' else 'coord', node=node, matrix=matrix,
        node_mask=mask, n=mask.sum(-1), coord_dist=torch.full((2,), 4),
        cbits=torch.tensor([constraint_bits(problem)] * 2, dtype=torch.float32),
        problem_desc=torch.tensor([problem_descriptor(problem)] * 2), node_geom=torch.zeros(2, 6, 12))


class RegionInterfaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.a = m.make_model(full18_params(), m.ARMS[0]).eval()
        cls.b = m.make_model(full18_params(), m.ARMS[1]).eval()

    @torch.no_grad()
    def test_identical_initialization_and_all_input_kinds(self):
        self.assertEqual(parameter_hash(self.a.state_dict()), parameter_hash(self.b.state_dict()))
        for problem in ('TSP', 'CVRP', 'ATSP', 'OVRPBLTW'):
            batch = input_batch(problem)
            output = self.b(batch)
            self.assertTrue(torch.isfinite(output['logits']).all())
            self.assertEqual(output['logits'].shape[-1], batch['pool_ids'].numel())

    @torch.no_grad()
    def test_conditions_really_collapse_only_in_control(self):
        batch = self.b.prepare_role_inputs(input_batch('CVRP')) if hasattr(self.b, 'prepare_role_inputs') else input_batch('CVRP')
        for model, collapse in ((self.a, True), (self.b, False)):
            tokens, mask = model.instance_encoder(batch)
            states = tokens[:, :4 * 36].reshape(2, 4, 36, -1)
            legal = mask[:, :4 * 36].reshape(2, 4, 36)
            difference = states - (states * legal[..., None]).sum(2, keepdim=True) / legal.sum(2, keepdim=True)[..., None].clamp_min(1)
            magnitude = difference.masked_fill(~legal[..., None], 0).abs().max().item()
            self.assertLess(magnitude, 2e-6) if collapse else self.assertGreater(magnitude, 1e-4)

    @torch.no_grad()
    def test_label_isolation_and_candidate_order(self):
        batch = input_batch('TSP')
        expected = self.b(batch)['logits']
        changed = self.b(dict(batch, costs=object(), ind=object(), winner=object(), routes=object()))['logits']
        torch.testing.assert_close(changed, expected, atol=0, rtol=0)
        order = torch.arange(batch['pool_ids'].numel() - 1, -1, -1)
        permuted = self.b(dict(batch, pool_ids=batch['pool_ids'][order]))['logits']
        torch.testing.assert_close(permuted, expected[:, order], atol=3e-6, rtol=1e-5)

    def test_joint_boundary_and_node_gradients(self):
        self.b.zero_grad(set_to_none=True)
        output = self.b(input_batch('CVRP'))
        torch.nn.functional.cross_entropy(output['logits'], torch.tensor([0, 1])).backward()
        for name in ('instance_encoder.stem.coord_proj.weight', 'instance_encoder.seed.0.weight',
                     'instance_encoder.up.0.0.weight', 'instance_encoder.down.0.0.weight'):
            gradient = dict(self.b.named_parameters())[name].grad
            self.assertIsNotNone(gradient, name)
            self.assertTrue(torch.isfinite(gradient).all(), name)
            self.assertGreater(gradient.abs().max().item(), 0, name)


if __name__ == '__main__':
    unittest.main()

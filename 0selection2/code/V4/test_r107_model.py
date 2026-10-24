"""Small CPU-only algebra/contract checks; no cache, checkpoint or GPU jobs."""

import copy
import unittest
from unittest.mock import patch

import torch
from torch import nn

from code.unified_selector.registry import P2I, POOLS, PROBLEMS, S2I, constraint_bits, problem_descriptor
from .dual_stream import DualStreamSelector
from .pairwise_objective import LABEL_KEYS
from .pairwise_selector import maximin_logits
from .r77_data import parameter_hash
from .r88_model import RoleInputSelector
from .r107_model import (
    ARMS, COMPOSED_FIELDS, EDGE_DIM, FRESH_REFERENCE_SHA256, ORIGINAL, PARAMETER_COUNTS,
    PATH_INIT_SEED, PATH_SPEC, ROLE_ARM, SPEC, ExplicitPathSelector, RelationRound,
    build_path_inputs, initialize_shared, make_model,
)
from .V4Model import EncoderLayer, get_default_model_params


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


def relabel(batch, order):
    result = dict(batch)
    for name in ('node', 'node_mask', 'node_geom'):
        result[name] = result[name][:, order]
    result['matrix'] = result['matrix'][:, order][:, :, order]
    return result


def path_slot(graph, i, j, k, batch=0):
    first = ((graph.neighbors[batch, i] == j) & graph.edge_mask[batch, i]).nonzero()[0, 0].item()
    second = ((graph.neighbors[batch, j] == k) & graph.edge_mask[batch, j]).nonzero()[0, 0].item()
    return first, second


class R107ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)
        cls.params = full18_params()
        with torch.device('cpu'), torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(2)
            original = RoleInputSelector(ROLE_ARM, **cls.params)
        trainable = {name: value.detach().clone() for name, value in original.named_parameters()}
        cls.initial = dict(seed=2, initialization='scratch', model_params=cls.params,
            trainable=trainable, trainable_sha256=parameter_hash(trainable))
        cls.geometry = dict(mean=[.21] * 12, std=[1.31] * 12, split='train', fitted_on='frozen TRAIN fixture')
        cls.models = {arm: initialize_shared(cls.initial, cls.geometry, arm)[0] for arm in ARMS}

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        for model in self.models.values():
            model.eval().zero_grad(set_to_none=True)

    def test_counts_shared_fresh_tensors_rng_no_file_load_and_original_topology(self):
        self.assertEqual(self.initial['trainable_sha256'], FRESH_REFERENCE_SHA256)
        self.assertEqual(SPEC['path_spec'], PATH_SPEC)
        self.assertEqual(parameter_hash(self.models[ARMS[0]].state_dict()),
                         parameter_hash(self.models[ARMS[1]].state_dict()))
        for arm, model in self.models.items():
            self.assertEqual(sum(p.numel() for p in model.parameters()), PARAMETER_COUNTS[arm])
            self.assertEqual(model.initialization_receipt['independently_initialized_parameter_count'], 49280)
            self.assertEqual(model.initialization_receipt['path_init_seed'], PATH_INIT_SEED)
            self.assertEqual(model.initialization_receipt['intentionally_inactive_composed_weight_count'],
                             4096 if arm == ARMS[0] else 0)
            self.assertEqual(model.params['path_arm'], arm)
            for name, value in ORIGINAL.items():
                self.assertEqual(model.params[name], value)
            for name, tensor in self.initial['trainable'].items():
                actual = dict(model.named_parameters())[name]
                torch.testing.assert_close(actual, tensor, rtol=0, atol=0)
                self.assertNotEqual(actual.data_ptr(), tensor.data_ptr())
            self.assertIs(ExplicitPathSelector.forward, RoleInputSelector.forward)
            self.assertIs(ExplicitPathSelector.encode_state, DualStreamSelector.encode_state)
            self.assertEqual(len(model.instance_encoder.layers), 4)
            self.assertTrue(all(type(layer) is EncoderLayer for layer in model.instance_encoder.layers))
            self.assertEqual(len(model.joint_layers), 2)
            self.assertEqual(model.solver_encoder.feature_mlp[0].in_features, 34)
            self.assertEqual(model.score_head[0].in_features, 384)
        rng = torch.get_rng_state().clone()
        with patch('torch.load', side_effect=AssertionError('no file initialization')):
            make_model(self.params, ARMS[0])
        torch.testing.assert_close(rng, torch.get_rng_state(), rtol=0, atol=0)

    @torch.no_grad()
    def test_all18_input_only_original_maximin_and_two_during_encoder_rounds(self):
        for arm, model in self.models.items():
            for problem in PROBLEMS:
                with self.subTest(arm=arm, problem=problem):
                    batch = input_batch(problem)
                    calls = []
                    handles = [layer.register_forward_hook(lambda m, a, o, i=i: calls.append(('layer', i)))
                               for i, layer in enumerate(model.instance_encoder.layers)]
                    handles += [layer.to_nodes.register_forward_hook(lambda m, a, o, i=i: calls.append(('relation', i)))
                                for i, layer in enumerate(model.instance_encoder.path_rounds)]
                    try:
                        output = model(batch)
                    finally:
                        for handle in handles:
                            handle.remove()
                    self.assertEqual(calls, [('layer', 0), ('relation', 0), ('layer', 1),
                        ('layer', 2), ('relation', 1), ('layer', 3)])
                    self.assertEqual(set(output), {'utility', 'logits', 'solver_mask', 'pair_margin'})
                    self.assertTrue(torch.isfinite(output['logits']).all())
                    torch.testing.assert_close(output['pair_margin'],
                        output['utility'][:, :, None] - output['utility'][:, None, :], rtol=0, atol=0)
                    torch.testing.assert_close(output['logits'],
                        maximin_logits(output['pair_margin'], output['solver_mask']), rtol=0, atol=0)
                    contaminated = dict(batch, **{key: object() for key in LABEL_KEYS},
                                        routes=object(), winner=object(), solver_outputs=object())
                    actual = model(contaminated)
                    for name in output:
                        torch.testing.assert_close(output[name], actual[name], rtol=0, atol=0)

    @torch.no_grad()
    def test_customer_and_candidate_permutation_equivariance_including_tsp_role(self):
        order = torch.tensor([4, 2, 0, 5, 1, 3])
        for arm, model in self.models.items():
            for problem in ('TSP', 'ATSP', 'CVRP', 'OVRPBLTW'):
                with self.subTest(arm=arm, problem=problem):
                    batch = model.prepare_role_inputs(input_batch(problem))
                    original = model(batch)
                    changed = model(relabel(batch, order))
                    torch.testing.assert_close(original['utility'], changed['utility'], atol=2e-6, rtol=2e-5)
                    nodes = model.instance_encoder(batch)[0]
                    other = model.instance_encoder(relabel(batch, order))[0]
                    torch.testing.assert_close(nodes[:, order], other, atol=4e-6, rtol=2e-5)
                    ids = batch['pool_ids']
                    candidate_order = torch.arange(ids.numel() - 1, -1, -1)
                    changed = model(dict(batch, pool_ids=ids[candidate_order]))
                    torch.testing.assert_close(original['utility'][:, candidate_order], changed['utility'], atol=2e-6, rtol=2e-5)

    def test_explicit_current_resource_arithmetic_open_return_no_classical_b_filter(self):
        batch = input_batch('VRPBLTW')
        batch = {key: value[:1] if isinstance(value, torch.Tensor) and key != 'pool_ids' else value
                 for key, value in batch.items()}
        raw = torch.zeros(1, 4, 8)
        raw[0, :, 0] = torch.tensor([0., 1., 2., 4.])
        raw[0, 0, 3] = 1.
        raw[0, 1:, 2] = torch.tensor([.4, -.6, .3])
        raw[0, 1:, 4] = 7.
        raw[0, 1:, 5] = torch.tensor([.25, .5, .75])
        raw[0, 2, 6] = 5.
        raw[0, 1:, 7] = torch.tensor([20., 8., 9.])
        batch.update(node=raw, node_mask=torch.ones(1, 4, dtype=torch.bool), n=torch.tensor([4]))
        graph = build_path_inputs(batch)
        a, b = path_slot(graph, 1, 2, 3)
        self.assertTrue(graph.path_mask[0, 1, a, b])  # pickup followed by delivery is not removed.
        values = dict(zip(COMPOSED_FIELDS, graph.composed[0, 1, a, b].tolist()))
        expected = dict(travel_sum=3., depot_walk_billed_travel=8., route_limit_slack=-1.,
            arrival_j=2.25, service_start_j=5., departure_j=5.5, arrival_k=7.5,
            service_start_k=7.5, departure_k=8.25, window_slack_j=3., window_slack_k=1.5,
            waiting_sum_jk=2.75, signed_prefix_i=.4, signed_prefix_j=-.2, signed_prefix_k=.1,
            signed_prefix_min=-.2, signed_prefix_max=.4, signed_prefix_span=.6,
            positive_demand_sum=.7, negative_demand_magnitude_sum=.6, service_sum=1.5)
        for name, value in expected.items():
            self.assertAlmostEqual(values[name], value, places=5, msg=name)
        opened = dict(batch, cbits=batch['cbits'].clone())
        opened['cbits'][:, 1] = 1.
        other = build_path_inputs(opened).composed[0, 1, a, b]
        self.assertEqual(other[COMPOSED_FIELDS.index('depot_walk_billed_travel')].item(), 4.)
        self.assertEqual(other[COMPOSED_FIELDS.index('route_limit_slack')].item(), 3.)
        # A real depot is not given the stored all-zero window as a deadline.
        c, d = path_slot(graph, 1, 0, 2)
        self.assertTrue(graph.path_mask[0, 1, c, d])
        self.assertEqual(graph.composed[0, 1, c, d, COMPOSED_FIELDS.index('window_slack_j')].item(), 0.)

    @torch.no_grad()
    def test_directional_matrix_and_valid_masks_depot_and_exact_boundary_ties(self):
        batch = input_batch('ATSP')
        graph = build_path_inputs(batch)
        reverse = build_path_inputs(dict(batch, matrix=batch['matrix'].transpose(-1, -2)))
        self.assertFalse(torch.equal(graph.edge_primitives, reverse.edge_primitives))
        for problem in ('TSP', 'CVRP', 'OVRPBLTW', 'ATSP'):
            batch = self.models[ARMS[0]].prepare_role_inputs(input_batch(problem))
            graph = build_path_inputs(batch)
            self.assertEqual(graph.neighbors.shape[-1], 4)
            self.assertEqual(graph.path_mask.shape[-2:], (4, 4))
            self.assertFalse(graph.edge_mask[~batch['node_mask']].any())
            self.assertTrue(torch.equal(graph.edge_primitives[~graph.edge_mask],
                                        torch.zeros_like(graph.edge_primitives[~graph.edge_mask])))
            if problem not in ('TSP', 'ATSP'):
                customers = batch['node_mask'] & (batch['node'][..., 3] == 0)
                self.assertTrue((graph.neighbors[..., 3][customers] == 0).all())
        # Five equal-cost nonself neighbors overflow four: drop the entire tied
        # class rather than prune by index. A TSP role is never a routing depot.
        batch = input_batch('ATSP')
        batch['node_mask'][:] = True
        batch['n'][:] = 6
        batch['matrix'][:] = 1.
        batch['matrix'].diagonal(dim1=1, dim2=2).zero_()
        self.assertFalse(build_path_inputs(batch).edge_mask.any())
        for model in self.models.values():
            torch.testing.assert_close(model(batch)['utility'],
                model(relabel(batch, torch.tensor([5, 3, 1, 4, 2, 0])))['utility'], atol=2e-6, rtol=2e-5)

    def test_joint_mixed_difference_vs_exact_edge_factorization(self):
        graph = build_path_inputs(input_batch('ATSP'))
        graph.composed.zero_()
        i, first = 0, 0
        j = graph.neighbors[0, i, first].item()
        second = graph.path_mask[0, i, first].nonzero()[0, 0].item()
        block = RelationRound().double()
        block.phi[3] = nn.Identity()
        with torch.no_grad():
            for parameter in block.parameters():
                parameter.zero_()
            block.phi[0].weight[0, 0] = 1.
            block.phi[0].weight[0, EDGE_DIM] = 1.
            block.phi[2].weight[0, 0] = 1.
        deltas = {}
        for arm in ARMS:
            results = []
            for x, y in ((0., 0.), (1., 0.), (0., 1.), (1., 1.)):
                edge = torch.zeros(2, 6, 4, EDGE_DIM, dtype=torch.float64)
                edge[0, i, first, 0] = x
                edge[0, j, second, 0] = y
                shape = (2, 6, 4, 32) if arm == ARMS[0] else (2, 6, 4, 4, 32)
                state, _ = block.update(edge, graph, torch.zeros(shape, dtype=torch.float64), arm)
                value = (.5 * (state[0, i, first, 0] + state[0, j, second, 0]) if arm == ARMS[0]
                         else state[0, i, first, second, 0])
                results.append(value.item())
            deltas[arm] = results[3] - results[1] - results[2] + results[0]
        self.assertAlmostEqual(deltas[ARMS[0]], 0., places=12)
        self.assertGreater(abs(deltas[ARMS[1]]), .1)

    def test_gradients_to_relations_raw_nodes_and_original_trunk(self):
        for arm, model in self.models.items():
            for problem in ('TSP', 'ATSP', 'OVRPBLTW'):
                with self.subTest(arm=arm, problem=problem):
                    model.zero_grad(set_to_none=True)
                    batch = input_batch(problem)
                    name = 'matrix' if problem == 'ATSP' else 'node'
                    batch[name] = batch[name].clone().requires_grad_()
                    output = model(batch)
                    (output['utility'].square().mean() + output['logits'].square().mean()).backward()
                    self.assertTrue(torch.isfinite(batch[name].grad).all())
                    self.assertGreater(batch[name].grad.abs().sum().item(), 0.)
                    for block in model.instance_encoder.path_rounds:
                        self.assertGreater(block.to_nodes.weight.grad.abs().sum().item(), 0.)
                        self.assertTrue(torch.isfinite(block.phi[0].weight.grad).all())
                        composed = block.phi[0].weight.grad[:, 2 * EDGE_DIM:2 * EDGE_DIM + 32]
                        if arm == ARMS[0]:
                            self.assertEqual(composed.count_nonzero().item(), 0)
                        else:
                            self.assertGreater(composed.abs().sum().item(), 0.)
                    for stage in (model.instance_encoder.path_node_projection,
                                  model.instance_encoder.layers[0], model.instance_encoder.layers[3],
                                  model.solver_encoder, model.joint_layers, model.pool, model.decoder, model.score_head):
                        gradients = [p.grad for p in stage.parameters() if p.grad is not None]
                        self.assertTrue(gradients)
                        self.assertTrue(all(torch.isfinite(g).all() for g in gradients))
                        self.assertGreater(sum(g.abs().sum().item() for g in gradients), 0.)

    @torch.no_grad()
    def test_padding_tiny_graphs_frozen_geometry_strict_restore_and_reject_trained_donor(self):
        for arm, model in self.models.items():
            before = parameter_hash(model.instance_encoder.geometry_residual.state_dict())
            for problem in ('TSP', 'ATSP', 'CVRP'):
                batch = input_batch(problem)
                original = model(batch)['utility']
                changed = dict(batch, node=batch['node'].clone(), matrix=batch['matrix'].clone())
                changed['node'][..., :2].masked_fill_(~batch['node_mask'][..., None], float('nan'))
                valid = batch['node_mask'][:, :, None] & batch['node_mask'][:, None, :]
                changed['matrix'].masked_fill_(~valid, float('nan'))
                torch.testing.assert_close(original, model(changed)['utility'], rtol=0, atol=0)
                tiny = dict(batch, node=batch['node'][:, :1], matrix=batch['matrix'][:, :1, :1],
                    node_mask=torch.ones(2, 1, dtype=torch.bool), n=torch.ones(2, dtype=torch.long),
                    node_geom=batch['node_geom'][:, :1])
                self.assertFalse(build_path_inputs(model.prepare_role_inputs(tiny)).path_mask.any())
                self.assertTrue(torch.isfinite(model(tiny)['utility']).all())
            model.train()
            model(input_batch('CVRP'))
            self.assertEqual(before, parameter_hash(model.instance_encoder.geometry_residual.state_dict()))
            torch.testing.assert_close(model.instance_encoder.geometry_residual.mean, torch.tensor(self.geometry['mean']))
            saved = copy.deepcopy(model.state_dict())
            saved['score_head.3.bias'].add_(.1)
            restored = make_model(model.params, arm, state=saved)
            self.assertEqual(parameter_hash(saved), parameter_hash(restored.state_dict()))
            self.assertFalse(restored.initialization_receipt['fresh_initialization'])
            with self.assertRaises(RuntimeError):
                make_model(model.params, arm, state={})
        bad = copy.deepcopy(self.initial)
        next(iter(bad['trainable'].values())).add_(.01)
        bad['trainable_sha256'] = parameter_hash(bad['trainable'])
        with self.assertRaises(ValueError):
            initialize_shared(bad, self.geometry, ARMS[0])
        for geometry in (dict(self.geometry, split='val'), dict(self.geometry, std=[0.] * 12)):
            with self.assertRaises(ValueError):
                initialize_shared(self.initial, geometry, ARMS[0])


if __name__ == '__main__':
    unittest.main()

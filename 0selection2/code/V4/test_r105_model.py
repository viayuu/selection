"""Bounded full-width CPU tensor checks; no data/cost/checkpoint/job generation."""

import copy
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F

from code.unified_selector.registry import (
    P2I, POOLS, PROBLEMS, S2I, constraint_bits, problem_descriptor,
)
from .dual_stream import DualStreamSelector
from .pairwise_objective import LABEL_KEYS, inputs_only
from .pairwise_selector import ScoreDifferenceSelector, maximin_logits, rank_by_solver_ids
from .r77_data import parameter_hash
from .r84_experiment import make_model as make_r84_model
from .r88_model import ROLE_SLOT, RoleInputSelector
from .r105_model import (
    ARMS, EXPERT_INIT_SEED, EXPERT_SPEC, FRESH_REFERENCE_SHA256, FRESH_SEED,
    ORIGINAL, PARAMETER_COUNT, ROLE_ARM, SPEC, DenseExpertFeedForward,
    initialize_shared, make_model,
)
from .V4Model import EncoderLayer, FeedForward, get_default_model_params


def full18_params():
    params = get_default_model_params()
    params.update(architecture='pairwise', joint_layer_num=2, local_geometry=True,
                  geometry_mode='real', ignore_coord_dist=True, native_winner=True,
                  decision_head='classification', pair_mode='score_difference',
                  solver_representation='handcrafted', solver_feature_weight=1.0, sdpa=True)
    return params


def input_batch(problem):
    """Tiny in-memory input tensors, deliberately without any supervision."""
    pool = [name for name in POOLS[problem] if name != 'RELD_MTL']
    if 'B' in problem and problem not in ('TSP', 'CVRP', 'ATSP'):
        pool = [name for name in pool if name not in ('RouteFinder', 'MoSES_RF', 'MoSES_CaDA')]
    node = torch.linspace(.01, .97, 80).reshape(2, 5, 8)
    mask = torch.tensor([[True, True, True, False, False], [True, True, True, True, True]])
    if problem == 'TSP':
        node = node[..., :2].clone()
    matrix = torch.arange(50).reshape(2, 5, 5).float().remainder(13) / 13 + .2
    matrix.diagonal(dim1=1, dim2=2).zero_()
    return dict(problem_id=P2I[problem], pool_ids=torch.tensor([S2I[name] for name in pool]),
                kind='matrix' if problem == 'ATSP' else 'coord', node=node, matrix=matrix,
                node_mask=mask, n=mask.sum(-1), coord_dist=torch.full((2,), 4),
                cbits=torch.tensor([constraint_bits(problem)] * 2, dtype=torch.float32),
                problem_desc=torch.tensor([problem_descriptor(problem)] * 2),
                node_geom=torch.zeros(2, 5, 12))


def relabel(batch, order):
    result = dict(batch)
    for name in ('node', 'node_mask', 'node_geom'):
        result[name] = result[name][:, order]
    result['matrix'] = result['matrix'][:, order][:, :, order]
    return result


class R105ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)
        cls.params = full18_params()
        with torch.device('cpu'), torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(FRESH_SEED)
            original = make_r84_model(cls.params, 'A_baseline')
        reference = {name: tensor.detach().clone() for name, tensor in original.named_parameters()}
        cls.initial = dict(seed=2, initialization='scratch', model_params=cls.params,
                           trainable=reference, trainable_sha256=parameter_hash(reference))
        cls.geometry = dict(mean=[.2] * 12, std=[1.3] * 12, fitted_on='synthetic TRAIN-only moments')
        cls.models = {arm: initialize_shared(cls.initial, cls.geometry, arm)[0] for arm in ARMS}

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        for model in self.models.values():
            model.eval().zero_grad(set_to_none=True)

    def assert_outputs_equal(self, first, second):
        self.assertEqual(set(first), {'logits', 'pair_margin', 'solver_mask', 'utility'})
        self.assertEqual(set(first), set(second))
        for name in first:
            torch.testing.assert_close(first[name], second[name], rtol=0, atol=0)

    def test_exact_arithmetic_dimensions_and_inactive_control_disclosure(self):
        self.assertEqual(SPEC['expert_spec'], EXPERT_SPEC)
        self.assertEqual(SPEC['parameter_count'], 3425561)
        self.assertEqual(1842953 + 12 * 131712 + 4 * 516, PARAMETER_COUNT)
        for arm, model in self.models.items():
            with self.subTest(arm=arm):
                self.assertEqual(sum(p.numel() for p in model.parameters()), PARAMETER_COUNT)
                self.assertTrue(all(p.requires_grad for p in model.parameters()))
                self.assertEqual(model.role_arm, ROLE_ARM)
                self.assertEqual(model.params['expert_arm'], arm)
                for name, value in ORIGINAL.items():
                    self.assertEqual(model.params[name], value)
                self.assertEqual(len(model.joint_layers), 2)
                self.assertEqual((model.score_head[0].in_features, model.score_head[0].out_features), (384, 256))
                self.assertEqual(model.solver_encoder.feature_mlp[0].in_features, 34)
                for layer in model.instance_encoder.layers:
                    self.assertIs(type(layer), EncoderLayer)
                    self.assertIsInstance(layer.ffn, DenseExpertFeedForward)
                    self.assertEqual(len(layer.ffn.experts), 4)
                    self.assertEqual((layer.ffn.router.in_features, layer.ffn.router.out_features), (128, 4))
                    self.assertIsNone(layer.ffn.last_routing)
                    self.assertIsNone(layer.ffn._capture_node_mask)
                    self.assertEqual(layer.ffn.router.weight.count_nonzero().item(), 0)
                    self.assertEqual(layer.ffn.router.bias.count_nonzero().item(), 0)
                    for expert in layer.ffn.experts:
                        self.assertIs(type(expert), FeedForward)
                        self.assertEqual(sum(p.numel() for p in expert.parameters()), 131712)
                        self.assertEqual(expert.net[2].p, .1)
                receipt = model.initialization_receipt
                self.assertEqual(receipt['common_parameter_count'], 1316105)
                self.assertEqual(receipt['expert0_copied_parameter_count'], 526848)
                self.assertEqual(receipt['independently_initialized_parameter_count'], 1580544)
                self.assertEqual(receipt['zero_initialized_parameter_count'], 2064)
                self.assertEqual(receipt['original_parameter_count'], 1842953)
                self.assertEqual(receipt['parameter_count'], PARAMETER_COUNT)
                self.assertEqual(receipt['intentionally_inactive_parameter_count'], 2048 if arm == ARMS[0] else 0)
                self.assertTrue(receipt['equal_total_parameters_not_equal_active_freedom'])
                self.assertFalse(hasattr(model, 'performance_head'))

    def test_exact_r84_copy_isolated_expert_seed_no_alias_and_complete_ab_identity(self):
        reference = self.initial['trainable']
        self.assertEqual(self.initial['trainable_sha256'], FRESH_REFERENCE_SHA256)
        first, second = (self.models[arm] for arm in ARMS)
        self.assertEqual(parameter_hash(first.state_dict()), parameter_hash(second.state_dict()))
        for arm, model in self.models.items():
            receipt, actual = model.initialization_receipt, dict(model.named_parameters())
            self.assertEqual(receipt['r84_fresh_trainable_sha256'], FRESH_REFERENCE_SHA256)
            self.assertEqual(receipt['expert_init_seed'], EXPERT_INIT_SEED)
            for name in receipt['common_parameter_names']:
                torch.testing.assert_close(actual[name], reference[name], rtol=0, atol=0)
                self.assertNotEqual(actual[name].data_ptr(), reference[name].data_ptr())
            for target, source in receipt['expert0_parameter_mapping'].items():
                torch.testing.assert_close(actual[target], reference[source], rtol=0, atol=0)
                self.assertNotEqual(actual[target].data_ptr(), reference[source].data_ptr())
            self.assertEqual(len({p.data_ptr() for p in actual.values()}), len(actual))
            for layer in model.instance_encoder.layers:
                for i in range(4):
                    for j in range(i):
                        self.assertFalse(torch.equal(layer.ffn.experts[i].net[0].weight,
                                                     layer.ffn.experts[j].net[0].weight))
            torch.testing.assert_close(model.instance_encoder.geometry_residual.mean, torch.tensor(self.geometry['mean']))
            torch.testing.assert_close(model.instance_encoder.geometry_residual.std, torch.tensor(self.geometry['std']))
        with torch.device('cpu'), torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(EXPERT_INIT_SEED)
            for layer in first.instance_encoder.layers:
                for index in range(1, 4):
                    expected = FeedForward(128, 512, .1)
                    self.assertEqual(parameter_hash(expected.state_dict()),
                                     parameter_hash(layer.ffn.experts[index].state_dict()))
                nn.Linear(128, 4)  # Router construction, subsequently zeroed.
        self.assertEqual(parameter_hash(reference), self.initial['trainable_sha256'])

    @torch.no_grad()
    def test_all18_same_initial_function_dropout_rng_and_original_return_path(self):
        first, second = (self.models[arm].train() for arm in ARMS)
        for model in (first, second):
            self.assertIs(type(model).forward, RoleInputSelector.forward)
            self.assertIs(type(model).encode_state, DualStreamSelector.encode_state)
            self.assertIs(type(model).encode_pairs, DualStreamSelector.encode_pairs)
        for problem in PROBLEMS:
            with self.subTest(problem=problem):
                batch = input_batch(problem)
                torch.random.default_generator.manual_seed(226)
                expected = first(batch)
                rng = torch.get_rng_state().clone()
                torch.random.default_generator.manual_seed(226)
                actual = second(batch)
                self.assert_outputs_equal(actual, expected)
                torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
                self.assertTrue(torch.isfinite(actual['utility']).all())
                self.assertTrue(torch.isfinite(actual['logits']).all())
                difference = actual['utility'][:, :, None] - actual['utility'][:, None, :]
                torch.testing.assert_close(actual['pair_margin'], difference, rtol=0, atol=0)
                torch.testing.assert_close(actual['logits'], maximin_logits(difference, actual['solver_mask']), rtol=0, atol=0)
                torch.random.default_generator.manual_seed(226)
                direct = ScoreDifferenceSelector.forward(second, second.prepare_role_inputs(batch))
                self.assert_outputs_equal(actual, direct)

    def test_router_feature_gradient_static_zero_input_finite_nonzero_and_all_experts(self):
        for arm, model in self.models.items():
            for problem in ('TSP', 'ATSP', 'OVRPBLTW'):
                with self.subTest(arm=arm, problem=problem):
                    model.train().zero_grad(set_to_none=True)
                    torch.random.default_generator.manual_seed(226)
                    output = model(input_batch(problem))
                    loss = output['utility'].square().mean() + output['logits'].square().mean()
                    self.assertTrue(torch.isfinite(loss))
                    loss.backward()
                    for layer in model.instance_encoder.layers:
                        router = layer.ffn.router
                        self.assertIsNotNone(router.weight.grad)
                        self.assertTrue(torch.isfinite(router.weight.grad).all())
                        if arm == ARMS[0]:
                            self.assertEqual(router.weight.grad.count_nonzero().item(), 0)
                        else:
                            self.assertGreater(router.weight.grad.abs().sum().item(), 0.)
                        self.assertTrue(torch.isfinite(router.bias.grad).all())
                        self.assertGreater(router.bias.grad.abs().sum().item(), 0.)
                        for expert in layer.ffn.experts:
                            for parameter in expert.parameters():
                                self.assertIsNotNone(parameter.grad)
                                self.assertTrue(torch.isfinite(parameter.grad).all())
                                self.assertGreater(parameter.grad.abs().sum().item(), 0.)
                    for stage in (model.solver_encoder, *model.joint_layers, model.pool, model.decoder, model.score_head):
                        gradients = [p.grad for p in stage.parameters() if p.grad is not None]
                        self.assertTrue(gradients)
                        self.assertTrue(all(torch.isfinite(g).all() for g in gradients))
                        self.assertGreater(sum(g.abs().sum().item() for g in gradients), 0.)

    @torch.no_grad()
    def test_learned_bias_static_not_uniform_and_conditional_state_dependence_dense_mixture(self):
        x = torch.zeros(2, 6, 128)
        x[1, :, 0] = 2.
        for arm, source in self.models.items():
            branch = copy.deepcopy(source.instance_encoder.layers[0].ffn).eval()
            branch.router.bias.copy_(torch.tensor([.2, -.3, .5, -.7]))
            branch.router.weight[:, 0].copy_(torch.tensor([-1., 0., 1., 2.]))
            branch._capture_node_mask = torch.ones(2, 3, dtype=torch.bool)
            calls = []
            handles = [expert.register_forward_hook(lambda module, args, output: calls.append(module))
                       for expert in branch.experts]
            try:
                actual = branch(x)
            finally:
                for handle in handles:
                    handle.remove()
                branch._capture_node_mask = None
            gate_input = torch.zeros_like(x) if arm == ARMS[0] else x
            weights = branch.router(gate_input).float().softmax(-1)
            expected = sum(expert(x).float() * weights[..., i:i + 1]
                           for i, expert in enumerate(branch.experts))
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            self.assertEqual(calls, list(branch.experts))
            torch.testing.assert_close(branch.last_routing, weights[:, 3:].mean(1), rtol=0, atol=0)
            self.assertFalse(torch.allclose(branch.last_routing[0], torch.full((4,), .25)))
            if arm == ARMS[0]:
                torch.testing.assert_close(branch.last_routing[0], branch.last_routing[1], rtol=0, atol=0)
            else:
                self.assertFalse(torch.allclose(branch.last_routing[0], branch.last_routing[1]))

    @torch.no_grad()
    def test_capture_valid_node_graph_mean_outputs_rng_flags_and_no_default_hooks(self):
        for arm, source in self.models.items():
            model = copy.deepcopy(source).train()
            for layer in model.instance_encoder.layers:
                layer.ffn.router.bias.copy_(torch.tensor([.2, -.4, .6, -.1]))
                layer.ffn.router.weight.copy_(torch.linspace(-.3, .4, 512).reshape(4, 128))
            batch = input_batch('TSP')
            torch.random.default_generator.manual_seed(226)
            expected = model(batch)
            rng = torch.get_rng_state().clone()
            torch.random.default_generator.manual_seed(226)
            recorded, handles = [], []
            for layer in model.instance_encoder.layers:
                handles.append(layer.ffn.register_forward_pre_hook(
                    lambda module, args: recorded.append(args[0].detach().clone())))
            try:
                routing, actual = model.capture_routing(batch)
            finally:
                for handle in handles:
                    handle.remove()
            self.assert_outputs_equal(actual, expected)
            torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
            self.assertTrue(model.training)
            self.assertEqual(tuple(routing.shape), (2, 4, 4))
            self.assertEqual(routing.dtype, torch.float32)
            self.assertFalse(routing.requires_grad)
            torch.testing.assert_close(routing.sum(-1), torch.ones(2, 4))
            snapshots = []
            for i, layer in enumerate(model.instance_encoder.layers):
                branch = layer.ffn
                state = torch.zeros_like(recorded[i]) if arm == ARMS[0] else recorded[i]
                weights = F.linear(state.float(), branch.router.weight, branch.router.bias).softmax(-1)[:, 3:]
                expected_mean = (weights.masked_fill(~batch['node_mask'][..., None], 0.).sum(1)
                                 / batch['node_mask'].sum(1, keepdim=True))
                torch.testing.assert_close(routing[:, i], expected_mean, rtol=0, atol=0)
                self.assertIsNone(branch._capture_node_mask)
                self.assertIsNone(branch.last_routing.grad_fn)
                snapshots.append(branch.last_routing)
            model(batch)
            for snapshot, layer in zip(snapshots, model.instance_encoder.layers):
                self.assertIs(layer.ffn.last_routing, snapshot)
            self.assertFalse(any(module._forward_hooks or module._forward_pre_hooks for module in model.modules()))

    def test_capture_failure_cleans_flags_and_rejects_empty_mask(self):
        model = copy.deepcopy(self.models[ARMS[1]])
        batch = input_batch('TSP')
        bad = model.prepare_role_inputs(batch)
        bad['node'][0, 0, ROLE_SLOT] = .5
        with self.assertRaises(ValueError):
            model.capture_routing(bad)
        for layer in model.instance_encoder.layers:
            self.assertIsNone(layer.ffn._capture_node_mask)
            self.assertIsNone(layer.ffn.last_routing)
        with self.assertRaises(ValueError):
            model.capture_routing(dict(batch, node_mask=torch.zeros_like(batch['node_mask'])))
        model.capture_routing(batch)

    @torch.no_grad()
    def test_candidate_permutation_masks_padding_singletons_and_ties(self):
        for arm, model in self.models.items():
            for problem in ('TSP', 'ATSP', 'OVRPBLTW'):
                with self.subTest(arm=arm, problem=problem):
                    batch = input_batch(problem)
                    expected = model(batch)
                    k = batch['pool_ids'].numel()
                    order = torch.arange(k - 1, -1, -1)
                    permuted = model(dict(batch, pool_ids=batch['pool_ids'][order]))
                    for key in ('logits', 'utility'):
                        torch.testing.assert_close(permuted[key], expected[key][:, order], rtol=1e-4, atol=2e-6)
                    torch.testing.assert_close(permuted['pair_margin'],
                        expected['pair_margin'][:, order][:, :, order], rtol=1e-4, atol=2e-6)
                    ids = torch.cat([batch['pool_ids'], torch.tensor([-1, 100000])])
                    mask = torch.ones(2, k + 2, dtype=torch.bool)
                    mask[:, -2:] = False
                    padded = model(dict(batch, pool_ids=ids, solver_mask=mask))
                    torch.testing.assert_close(padded['logits'][:, :k], expected['logits'], rtol=1e-4, atol=2e-6)
                    self.assertTrue(torch.isneginf(padded['logits'][:, -2:]).all())
                    self.assertEqual(padded['pair_margin'][:, -2:].count_nonzero().item(), 0)
                    self.assertEqual(padded['pair_margin'][:, :, -2:].count_nonzero().item(), 0)
                    row_ids = batch['pool_ids'].repeat(2, 1)
                    row_ids[1] = row_ids[1].flip(0)
                    row_mask = torch.ones(2, k, dtype=torch.bool)
                    row_mask[0, -1] = False
                    row_mask[1, 0] = False
                    masked = model(dict(batch, pool_ids=row_ids, solver_mask=row_mask))
                    for row in range(2):
                        compact_batch = {key: value[row:row + 1] if torch.is_tensor(value)
                                         and key != 'pool_ids' else value for key, value in batch.items()}
                        compact = model(dict(compact_batch, pool_ids=row_ids[row, row_mask[row]]))
                        torch.testing.assert_close(masked['logits'][row, row_mask[row]],
                            compact['logits'][0], rtol=1e-4, atol=2e-6)
                    single = model(dict(batch, pool_ids=batch['pool_ids'][:1]))
                    self.assertEqual(single['logits'].count_nonzero().item(), 0)
                    with self.assertRaises(ValueError):
                        model(dict(batch, solver_mask=torch.zeros(2, k, dtype=torch.bool)))
                    with self.assertRaises(ValueError):
                        model(dict(batch, solver_mask=torch.ones(2, k + 1, dtype=torch.bool)))
        ids = torch.tensor([18, 3, 16])
        torch.testing.assert_close(ids[rank_by_solver_ids(torch.zeros(2, 3), ids)], torch.tensor([[3, 16, 18]] * 2))

    @torch.no_grad()
    def test_node_padding_and_role_preserving_relabel(self):
        for arm, model in self.models.items():
            for problem in ('TSP', 'ATSP', 'OVRPBLTW'):
                batch = model.prepare_role_inputs(input_batch(problem))
                expected = model(batch)
                changed = copy.deepcopy(batch)
                padding = ~changed['node_mask']
                changed['node'][padding] = -12.
                if problem == 'TSP':
                    changed['node'][..., ROLE_SLOT][padding] = 0.
                changed['node_geom'][padding] = 11.
                pair_padding = padding[:, :, None] | padding[:, None, :]
                changed['matrix'][pair_padding] = 19.
                for key in ('logits', 'utility'):
                    torch.testing.assert_close(model(changed)[key], expected[key], rtol=1e-4, atol=2e-6)
                permuted = model(relabel(batch, torch.tensor([4, 2, 0, 3, 1])))
                torch.testing.assert_close(permuted['logits'], expected['logits'], rtol=1e-4, atol=2e-6)

    @torch.no_grad()
    def test_label_winner_cost_isolation_input_immutability_and_capture_no_labels(self):
        for arm, model in self.models.items():
            for problem in ('TSP', 'ATSP', 'OVRPBLTW'):
                batch = input_batch(problem)
                snapshot = copy.deepcopy(batch)
                expected = model(batch)
                poisoned = dict(batch)
                for name in (*LABEL_KEYS, 'winner', 'query_labels', 'query_winner', 'query_costs',
                             'donor_features', 'donor_labels', 'donor_costs', 'context'):
                    poisoned[name] = object()
                self.assertFalse(set(LABEL_KEYS) & set(model.prepare_role_inputs(poisoned)))
                self.assert_outputs_equal(model(poisoned), expected)
                routing, captured = model.capture_routing(poisoned)
                self.assert_outputs_equal(captured, expected)
                plain_routing, plain = model.capture_routing(inputs_only(batch))
                torch.testing.assert_close(routing, plain_routing, rtol=0, atol=0)
                self.assert_outputs_equal(plain, expected)
                for name, value in snapshot.items():
                    if torch.is_tensor(value):
                        torch.testing.assert_close(batch[name], value, rtol=0, atol=0)

    @torch.no_grad()
    def test_strict_fresh_and_trained_restore_without_fresh_provenance_claim(self):
        for arm, model in self.models.items():
            state = copy.deepcopy(model.state_dict())
            fresh = make_model(model.params, arm, state)
            self.assertTrue(fresh.initialization_receipt['restored_trainables_match_fresh'])
            self.assertFalse(fresh.initialization_receipt['fresh_initialization'])
            state['instance_encoder.layers.0.ffn.router.weight'].add_(.03)
            state['instance_encoder.layers.1.ffn.router.bias'].add_(torch.tensor([.1, -.2, .3, -.4]))
            state['instance_encoder.layers.2.ffn.experts.3.net.0.weight'].add_(.02)
            state['score_head.3.bias'].add_(.1)
            state['instance_encoder.geometry_residual.mean'].add_(.01)
            restored = make_model(model.params, arm, state).eval()
            receipt = restored.initialization_receipt
            self.assertEqual(receipt['initialization'], 'strict_caller_supplied_in_memory_restore')
            self.assertTrue(receipt['state_dict_restored'])
            self.assertFalse(receipt['fresh_initialization'])
            self.assertFalse(receipt['restored_trainables_match_fresh'])
            self.assertFalse(receipt['restored_state_matches_fresh'])
            self.assertIsNone(receipt['historical_checkpoint_loaded'])
            self.assertEqual(receipt['trainable_sha256'], parameter_hash(dict(restored.named_parameters())))
            self.assertEqual(receipt['state_sha256'], parameter_hash(state))
            for name, value in state.items():
                torch.testing.assert_close(restored.state_dict()[name], value, rtol=0, atol=0)
            direct = copy.deepcopy(model).eval()
            direct.load_state_dict(state, strict=True)
            for problem in ('TSP', 'ATSP'):
                self.assert_outputs_equal(restored(input_batch(problem)), direct(input_batch(problem)))
            missing = dict(state)
            missing.pop('score_head.3.weight')
            with self.assertRaises(RuntimeError):
                make_model(model.params, arm, missing)
            with self.assertRaises(RuntimeError):
                make_model(model.params, arm, dict(state, unexpected=torch.zeros(1)))
        with self.assertRaises(ValueError):
            make_model(self.models[ARMS[0]].params, ARMS[1])

    def test_cpu_autocast_fp32_routing_safe_mixing_and_finite_backward(self):
        for arm, source in self.models.items():
            branch = copy.deepcopy(source.instance_encoder.layers[0].ffn).eval()
            x = torch.linspace(-2., 2., 2 * 6 * 128).reshape(2, 6, 128).requires_grad_()
            branch._capture_node_mask = torch.tensor([[True, False, False], [True, True, True]])
            with torch.autocast(device_type='cpu', dtype=torch.bfloat16):
                output = branch(x)
            self.assertEqual(output.dtype, torch.bfloat16)
            self.assertEqual(branch.last_routing.dtype, torch.float32)
            self.assertTrue(torch.isfinite(output).all())
            output.float().square().mean().backward()
            self.assertTrue(torch.isfinite(x.grad).all())
            for parameter in branch.parameters():
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all())
            branch._capture_node_mask = None

    def test_reject_altered_contract_donor_records_and_invalid_geometry(self):
        changes = [dict(seed=3), dict(embedding_dim=256), dict(encoder_layer_num=8),
                   dict(dropout=0.), dict(local_geometry=False), dict(solver_feature_weight=.3),
                   dict(role_arm='A_canonical_anchor'), dict(pair_mode='explicit_pair'),
                   dict(selection_head='direct_attention'), dict(solver_representation='code'),
                   dict(residual_head=True), dict(expert_spec=dict(EXPERT_SPEC, experts_per_layer=8))]
        for changed in changes:
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                make_model(dict(self.params, **changed), ARMS[0])
        with self.assertRaises(ValueError):
            make_model(self.params, 'C_unplanned')
        for changed in (dict(self.initial, seed=3), dict(self.initial, initialization='trained'),
                        dict(self.initial, trainable_sha256='invalid')):
            with self.assertRaises(ValueError):
                initialize_shared(changed, self.geometry, ARMS[0])
        donor = copy.deepcopy(self.initial)
        donor['trainable']['score_head.3.bias'].add_(1.)
        donor['trainable_sha256'] = parameter_hash(donor['trainable'])
        with self.assertRaisesRegex(ValueError, 'no donor'):
            initialize_shared(donor, self.geometry, ARMS[0])
        for geometry in (dict(mean=[0.] * 11, std=[1.] * 12),
                         dict(mean=[float('nan')] * 12, std=[1.] * 12),
                         dict(mean=[0.] * 12, std=[0.] * 12)):
            with self.assertRaises(ValueError):
                initialize_shared(self.initial, geometry, ARMS[0])

    def test_factory_rng_isolation_determinism_no_file_loads_or_gpu_initialization(self):
        with (patch('torch.load', side_effect=AssertionError('No checkpoint file reads')),
              patch('builtins.open', side_effect=AssertionError('No file reads')),
              patch('io.open', side_effect=AssertionError('No pathlib reads')),
              patch('torch.cuda._lazy_init', side_effect=AssertionError('No GPU initialization'))):
            for arm in ARMS:
                torch.random.default_generator.manual_seed(999)
                rng = torch.get_rng_state().clone()
                model = make_model(self.params, arm)
                torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
                torch.rand(23)
                second = make_model(self.params, arm)
                self.assertEqual(parameter_hash(model.state_dict()), parameter_hash(second.state_dict()))
                self.assertTrue(all(p.device.type == 'cpu' for p in model.parameters()))


if __name__ == '__main__':
    unittest.main()

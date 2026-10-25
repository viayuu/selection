"""Six bounded CPU checks; reuse R109's toy inputs, never launch experiments."""

import copy
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F

from code.unified_selector.registry import GLOBAL_SOLVERS, PROBLEMS, S2I
from . import r111_model as m
from .pairwise_objective import LABEL_KEYS
from .r77_data import parameter_hash
from .r88_model import RoleInputSelector
from .test_r109_model import full18_params, input_batch
from .V4Model import EncoderLayer


class FamilyConditionalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)
        cls.params = full18_params()
        with torch.device('cpu'), torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(2)
            base = RoleInputSelector(m.ROLE_ARM, **cls.params)
        trainable = {name: value.detach().clone() for name, value in base.named_parameters()}
        cls.initial = dict(seed=2, initialization='scratch', model_params=cls.params,
            trainable=trainable, trainable_sha256=parameter_hash(trainable))
        cls.geometry = dict(mean=[.21] * 12, std=[1.31] * 12, split='train')
        cls.model, cls.resolved, cls.receipt = m.initialize_shared(cls.initial, cls.geometry, m.ARMS[0])

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        self.model.eval().zero_grad(set_to_none=True)

    @torch.no_grad()
    def test_full18_normalization_and_fixed_global_family_schema(self):
        self.assertEqual(m.ARMS, ('A_family_conditional',))
        self.assertEqual(len(m.FAMILIES), 4)
        self.assertEqual(self.resolved['r111_spec'], m.SPEC)
        self.assertEqual(self.resolved['decision_head'], 'logprob')
        self.assertEqual(self.model.solver_family_ids[S2I['RELD_MTL']].item(), -1)
        self.assertEqual(len(self.model.solver_family_ids), len(GLOBAL_SOLVERS))
        for problem in PROBLEMS:
            output = self.model(input_batch(problem))
            torch.testing.assert_close(output['logits'].logsumexp(-1), torch.zeros(2), atol=2e-6, rtol=0)
            self.assertIs(output['logits'], output['utility'])
            self.assertTrue(torch.isfinite(output['logits']).all())
            root = output['family_logits'].log_softmax(-1).exp()
            for family in range(4):
                mass = (output['logits'].exp() * (output['family_ids'] == family)).sum(-1)
                torch.testing.assert_close(mass, root[:, family], atol=2e-6, rtol=1e-5)

    def test_rowwise_absent_families_singletons_and_finite_backward(self):
        ids = torch.tensor([[S2I['DIFUSCO'], S2I['BQ'], S2I['LEHD'], -1],
                            [S2I['RELD_CVRP'], -1, -1, -1]])
        batch = dict(input_batch('TSP'), pool_ids=ids, solver_mask=ids >= 0)
        calls = []
        handles = [layer.register_forward_hook(lambda module, args, result, name=name: calls.append(name))
                   for name, layers in self.model.family_node_layers.items() for layer in layers]
        try:
            output = self.model(batch)
        finally:
            for handle in handles:
                handle.remove()
        self.assertEqual(calls, ['other_constructive', 'other_constructive'])
        logits, root = output['logits'], output['family_logits'].log_softmax(-1)
        torch.testing.assert_close(logits.logsumexp(-1), torch.zeros(2), atol=1e-6, rtol=0)
        torch.testing.assert_close(logits[0, 0], root[0, 0], atol=0, rtol=0)
        self.assertEqual(logits[1, 0].item(), 0.)
        self.assertTrue(torch.isneginf(logits[~batch['solver_mask']]).all())
        F.nll_loss(logits, torch.tensor([0, 0])).backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in self.model.parameters() if p.grad is not None))
        with self.assertRaises(ValueError):
            self.model(dict(batch, solver_mask=torch.zeros_like(ids, dtype=torch.bool)))
        with self.assertRaises(ValueError):
            self.model(dict(input_batch('TSP'), pool_ids=torch.tensor([S2I['RELD_MTL']])))

    @torch.no_grad()
    def test_candidate_order_equivariance_and_masked_padding(self):
        for problem in ('TSP', 'ATSP'):
            batch = input_batch(problem)
            expected = self.model(batch)
            ids, order = batch['pool_ids'], torch.arange(batch['pool_ids'].numel() - 1, -1, -1)
            changed = self.model(dict(batch, pool_ids=ids[order]))
            torch.testing.assert_close(changed['logits'], expected['logits'][:, order], atol=2e-6, rtol=1e-5)
            torch.testing.assert_close(changed['family_logits'], expected['family_logits'], atol=2e-6, rtol=1e-5)
            padded = self.model(dict(batch, pool_ids=torch.cat((ids, torch.tensor([-1]))),
                solver_mask=torch.tensor([[True] * len(ids) + [False]] * 2)))
            torch.testing.assert_close(padded['logits'][:, :-1], expected['logits'], atol=2e-6, rtol=1e-5)
            self.assertTrue(torch.isneginf(padded['logits'][:, -1]).all())

    @torch.no_grad()
    def test_forward_never_reads_labels_or_outcomes(self):
        for problem in ('TSP', 'CVRP', 'ATSP', 'OVRPBLTW'):
            batch = input_batch(problem)
            original_nodes = batch['node'].clone()
            expected = self.model(batch)
            contaminated = dict(batch, **{key: object() for key in LABEL_KEYS},
                                winner=object(), routes=object(), solver_outputs=object())
            actual = self.model(contaminated)
            for name in expected:
                torch.testing.assert_close(actual[name], expected[name], atol=0, rtol=0)
            torch.testing.assert_close(batch['node'], original_nodes, atol=0, rtol=0)

    def test_single_winner_family_private_gradient_isolation(self):
        self.model.train()
        ids = torch.tensor(m.ACTIVE_IDS)
        batch = dict(input_batch('ATSP'), pool_ids=ids)
        expected_bias = self.model.instance_encoder._matrix_features_and_bias(
            batch['matrix'], batch['node_mask'], 0)[1].detach()
        self.assertFalse(torch.equal(expected_bias, expected_bias.transpose(-1, -2)))
        calls = []
        handles = [layer.register_forward_pre_hook(lambda module, args, kwargs, name=name:
                   calls.append((name, args[0].shape[1], kwargs['attn_bias'].detach())), with_kwargs=True)
                   for name, layers in self.model.family_node_layers.items() for layer in layers]
        try:
            with patch('code.V4.r111_model.checkpoint', wraps=m.checkpoint) as recompute:
                output = self.model(batch)
            self.assertEqual(recompute.call_count, 6)
            for call in recompute.call_args_list:
                self.assertIs(call.kwargs['use_reentrant'], False)
                self.assertIs(call.kwargs['preserve_rng_state'], True)
        finally:
            for handle in handles:
                handle.remove()
        self.assertEqual([name for name, _, _ in calls], ['diffusion'] * 2 + ['reld'] * 2 + ['other_constructive'] * 2)
        for _, node_count, bias in calls:
            self.assertEqual(node_count, batch['node_mask'].size(1))
            torch.testing.assert_close(bias, expected_bias, atol=0, rtol=0)
        output['family_logits'].retain_grad()
        winner = (ids == S2I['DIFUSCO']).nonzero(as_tuple=True)[0].item()
        F.nll_loss(output['logits'], torch.full((2,), winner, dtype=torch.long)).backward()
        for family, layers in self.model.family_node_layers.items():
            gradients = [p.grad for p in layers.parameters()]
            magnitude = sum(float(g.square().sum()) for g in gradients if g is not None)
            self.assertTrue(all(torch.isfinite(g).all() for g in gradients if g is not None))
            if family == 'diffusion':
                self.assertGreater(magnitude, 0.)
            else:
                self.assertEqual(magnitude, 0.)
        self.assertTrue((output['family_logits'].grad[:, 0] < 0).all())
        self.assertTrue((output['family_logits'].grad[:, 1:] > 0).all())
        self.assertGreater(float(self.model.instance_encoder.matrix_proj.weight.grad.square().sum()), 0.)

    def test_fresh_copy_determinism_geometry_and_strict_in_memory_restore(self):
        self.assertEqual(self.initial['trainable_sha256'], m.FRESH_REFERENCE_SHA256)
        for name, tensor in self.initial['trainable'].items():
            actual = dict(self.model.named_parameters())[name]
            torch.testing.assert_close(actual, tensor, atol=0, rtol=0)
            self.assertNotEqual(actual.data_ptr(), tensor.data_ptr())
        branch = self.model.instance_encoder.geometry_residual
        torch.testing.assert_close(branch.mean, torch.tensor(self.geometry['mean']))
        torch.testing.assert_close(branch.std, torch.tensor(self.geometry['std']))
        self.assertEqual(len(self.model.instance_encoder.layers), 4)
        self.assertEqual(len(self.model.joint_layers), 2)
        self.assertEqual(self.model.solver_encoder.feature_mlp[0].in_features, 34)
        for layers in self.model.family_node_layers.values():
            self.assertEqual(len(layers), 2)
            for layer, source in zip(layers, self.model.instance_encoder.layers[-2:]):
                self.assertIs(type(layer), EncoderLayer)
                for name, tensor in layer.named_parameters():
                    original = dict(source.named_parameters())[name]
                    torch.testing.assert_close(tensor, original, atol=0, rtol=0)
                    self.assertNotEqual(tensor.data_ptr(), original.data_ptr())
        rng = torch.get_rng_state().clone()
        with patch('torch.load', side_effect=AssertionError('no file/checkpoint initialization')):
            other, params, receipt = m.initialize_shared(self.initial, self.geometry, m.ARMS[0])
            restored = m.make_model(params, m.ARMS[0], state_dict=copy.deepcopy(other.state_dict()))
        torch.testing.assert_close(torch.get_rng_state(), rng, atol=0, rtol=0)
        self.assertEqual(parameter_hash(other.state_dict()), parameter_hash(self.model.state_dict()))
        self.assertEqual(parameter_hash(restored.state_dict()), parameter_hash(other.state_dict()))
        self.assertFalse(receipt['historical_checkpoint_loaded'])
        self.assertEqual(receipt['branch_init_seed'], 111002)
        with self.assertRaises(ValueError):
            m.initialize_shared(dict(self.initial, initialization='trained'), self.geometry, m.ARMS[0])


if __name__ == '__main__':
    unittest.main()

"""Bounded CPU checks for the candidate-conditioned graph branches."""

import copy
import unittest
from unittest.mock import patch

import torch

from code.unified_selector.registry import PROBLEMS, S2I
from . import r109_model as m
from .pairwise_objective import LABEL_KEYS
from .r77_data import parameter_hash
from .r88_model import RoleInputSelector
from .test_r107_model import full18_params, input_batch, relabel


class MethodNodeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)
        cls.params = full18_params()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(2)
            base = RoleInputSelector(m.ROLE_ARM, **cls.params)
        trainable = {k: v.detach().clone() for k, v in base.named_parameters()}
        cls.initial = dict(seed=2, initialization='scratch', model_params=cls.params,
            trainable=trainable, trainable_sha256=parameter_hash(trainable))
        cls.geometry = dict(mean=[.21] * 12, std=[1.31] * 12, split='train')
        cls.models = {a: m.initialize_shared(cls.initial, cls.geometry, a)[0] for a in m.ARMS}

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        for model in self.models.values():
            model.eval().zero_grad(set_to_none=True)

    def test_original_common_fresh_parameters_and_declared_private_counts(self):
        self.assertEqual(self.initial['trainable_sha256'], m.FRESH_REFERENCE_SHA256)
        for arm, model in self.models.items():
            self.assertEqual(sum(p.numel() for p in model.parameters()), m.PARAMETER_COUNTS[arm])
            for name, expected in self.initial['trainable'].items():
                torch.testing.assert_close(dict(model.named_parameters())[name], expected, rtol=0, atol=0)
            self.assertEqual(len(model.instance_encoder.layers), 4)
            self.assertEqual(len(model.joint_layers), 2)
        self.assertEqual(len(self.models[m.ARMS[0]].late_banks), 1)
        self.assertEqual(len(self.models[m.ARMS[1]].late_banks), len(m.ACTIVE_IDS))

    @torch.no_grad()
    def test_same_initial_function_full18_and_input_label_isolation(self):
        for problem in PROBLEMS:
            batch = input_batch(problem)
            outputs = [model(batch) for model in self.models.values()]
            torch.testing.assert_close(outputs[0]['logits'], outputs[1]['logits'], rtol=0, atol=0)
            contaminated = dict(batch, **{key: object() for key in LABEL_KEYS}, routes=object())
            for model, expected in zip(self.models.values(), outputs):
                actual = model(contaminated)
                self.assertTrue(torch.isfinite(actual['logits']).all())
                for name in actual:
                    torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0)

    @torch.no_grad()
    def test_candidate_permutation_padding_and_global_identity(self):
        batch = input_batch('TSP')
        k = len(batch['pool_ids'])
        order = torch.arange(k - 1, -1, -1)
        for model in self.models.values():
            expected = model(batch)['logits']
            moved = dict(batch, pool_ids=batch['pool_ids'][order])
            torch.testing.assert_close(model(moved)['logits'], expected[:, order], rtol=1e-5, atol=2e-6)
            padded = dict(batch, pool_ids=torch.cat((batch['pool_ids'], torch.tensor([-1]))),
                          solver_mask=torch.tensor([[True] * k + [False]] * 2))
            actual = model(padded)
            torch.testing.assert_close(actual['logits'][:, :k], expected, rtol=1e-5, atol=2e-6)
            self.assertTrue(torch.isneginf(actual['logits'][:, -1]).all())

    @torch.no_grad()
    def test_relabel_with_preserved_physical_start_role(self):
        order = torch.tensor([2, 0, 4, 1, 5, 3])
        for model in self.models.values():
            for problem in ('TSP', 'CVRP', 'ATSP'):
                batch = model.prepare_role_inputs(input_batch(problem))
                torch.testing.assert_close(model(batch)['logits'], model(relabel(batch, order))['logits'],
                                           rtol=1e-4, atol=3e-6)

    def test_private_graph_receives_gradient_and_unused_methods_do_not(self):
        model = self.models[m.ARMS[1]].train()
        batch = input_batch('TSP')
        output = model(batch)
        torch.nn.functional.cross_entropy(output['logits'], torch.tensor([1, 4])).backward()
        for global_id in batch['pool_ids'].tolist():
            gradients = [p.grad for p in model.late_banks[str(global_id)].parameters()]
            self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in gradients))
            self.assertGreater(sum(float(g.square().sum()) for g in gradients), 0.)
        self.assertTrue(all(p.grad is None for p in model.late_banks[str(S2I['MTPOMO'])].parameters()))
        self.assertGreater(float(model.instance_encoder.coord_proj.weight.grad.square().sum()), 0.)

    def test_checkpoint_dropout_recomputation_matches_eager_gradients(self):
        model = self.models[m.ARMS[0]].train()
        batch = input_batch('TSP')
        rng = torch.get_rng_state().clone()
        first = model(batch)['logits']
        first.square().mean().backward()
        grads = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
        model.zero_grad(set_to_none=True)
        torch.set_rng_state(rng)
        def eager(layer, x, **kwargs):
            kwargs.pop('use_reentrant')
            kwargs.pop('preserve_rng_state')
            return layer(x, **kwargs)
        with patch('code.V4.r109_model.checkpoint', eager):
            second = model(batch)['logits']
            second.square().mean().backward()
        torch.testing.assert_close(first, second, rtol=0, atol=0)
        for name, value in grads.items():
            torch.testing.assert_close(dict(model.named_parameters())[name].grad, value, rtol=2e-5, atol=2e-7)

    def test_strict_checkpoint_restore_and_no_implicit_file_reads(self):
        model = self.models[m.ARMS[0]]
        with patch('torch.load', side_effect=AssertionError('factory must be in-memory')):
            restored = m.make_model(model.params, m.ARMS[0], state_dict=copy.deepcopy(model.state_dict()))
        self.assertEqual(parameter_hash(restored.state_dict()), parameter_hash(model.state_dict()))
        with self.assertRaises(ValueError):
            m.initialize_shared(dict(self.initial, initialization='trained'), self.geometry, m.ARMS[0])


if __name__ == '__main__':
    unittest.main()

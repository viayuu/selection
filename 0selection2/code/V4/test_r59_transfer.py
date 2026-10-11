"""CPU-only R59 encoder transfer and selector integration checks."""

import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import PropertyMock, patch

import torch
from torch import nn
from torch.nn import functional as F

from code.unified_selector.registry import GLOBAL_SOLVERS, POOLS, PROBLEMS
from .r59_transfer import (NATIVE_CONFIG, NATIVE_PREFIX, PaddingAwareMTLEncoder, initialize,
                           make_r59_model, native_inputs, native_parameters)
from .test_solver_features import sample_batch


GEOMETRY = dict(mean=[0.] * 12, std=[1.] * 12)


def routing_batch(problem='OVRPBLTW'):
    batch = sample_batch(problem)
    if problem not in ('TSP', 'ATSP'):
        node = batch['node']
        node[:, 0, 2:] = 0.
        node[:, 0, 3] = 1.
        node[:, 1:, 3] = 0.
        node[:, 1:, 2] = torch.tensor([.2, -.1, .3, -.2]) if 'B' in problem else .2
        node[:, 1:, 4] = 3. if 'L' in problem else 0.
        node[:, 1:, 5:] = .1 if 'TW' in problem else 0.
        if 'TW' in problem:
            node[:, 1:, 7] = 2.
        if problem == 'CVRP':
            batch['node'] = node[..., :4].clone()
    return batch


class R59TransferTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        with patch('torch.cuda._lazy_init', side_effect=AssertionError('CPU tests must not initialize CUDA')):
            cls.random, cls.random_params = initialize('A', GEOMETRY)
            cls.pretrained, cls.pretrained_params = initialize('B', GEOMETRY)
        source = cls.pretrained_params['native_provenance']['source']['path']
        spec = importlib.util.spec_from_file_location('r59_native_mtl_parity', source)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        cls.reference = module.MTL_Encoder(**NATIVE_CONFIG).cpu().eval()
        cls.reference.load_state_dict(cls.pretrained.instance_encoder.native_encoder.state_dict(), strict=True)
        cls.pretrained.eval()
        cls.random.eval()

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def setUp(self):
        torch.random.default_generator.manual_seed(59)

    @torch.no_grad()
    def test_native_unpadded_fp32_parity(self):
        for problem in ('CVRP', 'OVRP', 'OVRPBLTW'):
            with self.subTest(problem=problem):
                depot, customers, mask = native_inputs(routing_batch(problem))
                native = self.reference(depot, customers)
                local = self.pretrained.instance_encoder.native_encoder(depot, customers, mask)
                self.assertEqual(local.dtype, torch.float32)
                torch.testing.assert_close(local, native, rtol=1e-6, atol=1e-6)

    @torch.no_grad()
    def test_sdpa_mask_and_zero_dropout_semantics_on_cpu(self):
        batch = routing_batch()
        batch['node_mask'][0, -2:] = False
        inputs = native_inputs(batch)
        encoder = self.pretrained.instance_encoder.native_encoder
        expected = encoder(*inputs)
        # Exercise dispatch with CPU tensors; the actual SDPA kernel remains CPU-only.
        with patch.object(torch.Tensor, 'is_cuda', new_callable=PropertyMock, return_value=True), \
             patch('torch.nn.functional.scaled_dot_product_attention', wraps=F.scaled_dot_product_attention) as sdpa:
            actual = encoder(*inputs)
        self.assertEqual(sdpa.call_count, 6)
        for call in sdpa.call_args_list:
            self.assertEqual(call.kwargs['attn_mask'].dtype, torch.bool)
            torch.testing.assert_close(call.kwargs['attn_mask'], inputs[2][:, None, None, :])
            self.assertEqual(call.kwargs['dropout_p'], 0.)
            self.assertFalse(call.kwargs['is_causal'])
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)

    def test_pretrained_weights_exact_and_other_initial_parameters_matched(self):
        state_a, state_b = self.random.state_dict(), self.pretrained.state_dict()
        self.assertEqual(state_a.keys(), state_b.keys())
        differences = []
        for key in state_a:
            if key.startswith(NATIVE_PREFIX):
                if not torch.equal(state_a[key], state_b[key]):
                    differences.append(key)
            else:
                torch.testing.assert_close(state_a[key], state_b[key], rtol=0, atol=0)
        self.assertTrue(differences)
        checkpoint = torch.load(self.pretrained_params['native_checkpoint']['path'], map_location='cpu', weights_only=False)
        expected = {key[len('encoder.'):]: value for key, value in checkpoint['model_state_dict'].items()
                    if key.startswith('encoder.')}
        actual = self.pretrained.instance_encoder.native_encoder.state_dict()
        self.assertEqual(expected.keys(), actual.keys())
        for key in actual:
            torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
        self.assertEqual(len(self.pretrained.instance_encoder.native_encoder.layers), 6)
        self.assertFalse(any(isinstance(m, (nn.LayerNorm, nn.Dropout, nn.BatchNorm1d, nn.InstanceNorm1d))
                             for m in self.pretrained.instance_encoder.native_encoder.modules()))

    def test_native_mapping_and_absent_time_windows(self):
        batch = routing_batch('OVRPBLTW')
        depot, customers, mask = native_inputs(batch)
        torch.testing.assert_close(depot, batch['node'][:, :1, :2], rtol=0, atol=0)
        torch.testing.assert_close(customers, batch['node'][:, 1:, [0, 1, 2, 6, 7]], rtol=0, atol=0)
        self.assertTrue((customers[..., 2] < 0).any())
        for problem in ('CVRP', 'OVRP', 'VRPB', 'VRPL'):
            batch = routing_batch(problem)
            if problem != 'CVRP':
                batch['node'][..., 6:8] = 99.
            _, customers, _ = native_inputs(batch)
            self.assertTrue((customers[..., 3:] == 0).all())
        torch.testing.assert_close(mask, batch['node_mask'])

    @torch.no_grad()
    def test_mixed_size_padding_and_no_cross_batch_information(self):
        batch = routing_batch()
        batch['node_mask'][0, -2:] = False
        batch['n'][0] -= 2
        original, mask = self.pretrained.instance_encoder(batch)
        original_logits = self.pretrained(batch)['logits']
        padded = copy.deepcopy(batch)
        padded['node'] = F.pad(batch['node'], (0, 0, 0, 3), value=float('nan'))
        padded['node_mask'] = F.pad(mask, (0, 3), value=False)
        padded_nodes, _ = self.pretrained.instance_encoder(padded)
        torch.testing.assert_close(padded_nodes[:, :5], original, rtol=2e-5, atol=2e-5)
        torch.testing.assert_close(self.pretrained(padded)['logits'], original_logits, rtol=2e-5, atol=2e-5)
        self.assertTrue((padded_nodes[:, 5:] == 0).all())
        single = {key: value[:1] if torch.is_tensor(value) and key != 'pool_ids' else value
                  for key, value in batch.items()}
        single['node'] = single['node'][:, :3]
        single['node_mask'] = single['node_mask'][:, :3]
        alone, _ = self.pretrained.instance_encoder(single)
        torch.testing.assert_close(alone[0], original[0, :3], rtol=2e-5, atol=2e-5)
        changed = copy.deepcopy(batch)
        changed['node'][1, :, :2] += 100.
        torch.testing.assert_close(self.pretrained.instance_encoder(changed)[0][0], original[0], rtol=0, atol=0)

    @torch.no_grad()
    def test_all_legal_pools_and_legacy_paths(self):
        self.assertEqual((len(PROBLEMS), len(GLOBAL_SOLVERS), sum(map(len, POOLS.values()))), (18, 22, 128))
        for problem in PROBLEMS:
            with self.subTest(problem=problem):
                batch = routing_batch(problem)
                out = self.pretrained(batch)
                self.assertEqual(out['logits'].shape, (2, len(POOLS[problem])))
                self.assertTrue(torch.isfinite(out['logits']).all())
                if problem in ('TSP', 'ATSP'):
                    encoded, mask = self.pretrained.instance_encoder(batch)
                    legacy, legacy_mask = self.pretrained.instance_encoder.legacy_encoder(batch)
                    torch.testing.assert_close(encoded, legacy, rtol=0, atol=0)
                    torch.testing.assert_close(mask, legacy_mask)

    @torch.no_grad()
    def test_candidate_permutation_padding_singleton_and_customer_permutation(self):
        batch = routing_batch('CVRP')
        original = self.pretrained(batch)['logits']
        order = torch.randperm(len(batch['pool_ids']))
        torch.testing.assert_close(self.pretrained(dict(batch, pool_ids=batch['pool_ids'][order]))['logits'],
                                   original[:, order], rtol=2e-5, atol=2e-5)
        ids = F.pad(batch['pool_ids'], (0, 2), value=-1)
        mask = torch.ones(2, len(ids), dtype=torch.bool)
        mask[:, -2:] = False
        padded = self.pretrained(dict(batch, pool_ids=ids, solver_mask=mask))['logits']
        torch.testing.assert_close(padded[:, :-2], original, rtol=2e-5, atol=2e-5)
        self.assertTrue(torch.isneginf(padded[:, -2:]).all())
        single = self.pretrained(dict(batch, pool_ids=batch['pool_ids'][:1]))['logits']
        torch.testing.assert_close(single, torch.zeros_like(single), rtol=0, atol=0)
        customers = torch.tensor([0, 3, 1, 4, 2])
        reordered = dict(batch, node=batch['node'][:, customers], node_mask=batch['node_mask'][:, customers])
        torch.testing.assert_close(self.pretrained(reordered)['logits'], original, rtol=2e-5, atol=2e-5)

    @torch.no_grad()
    def test_labels_never_enter_forward_and_raw_service_limit_are_preserved(self):
        batch = routing_batch()
        clean = {key: value for key, value in batch.items() if key not in ('costs', 'ind', 'winner')}
        before = self.pretrained(clean)['logits']
        labelled = dict(clean, costs=torch.full_like(batch['costs'], float('nan')), ind=torch.tensor([-999, 999]),
                        winner=None, raw_costs=torch.zeros(2, 100), relative_regret=float('nan'), true_top3=None)
        torch.testing.assert_close(self.pretrained(labelled)['logits'], before, rtol=0, atol=0)
        native_before = self.pretrained.instance_encoder.native_encoder(*native_inputs(clean))
        for column in (4, 5):
            changed = copy.deepcopy(clean)
            changed['node'][:, 1:, column] += .5
            native_after = self.pretrained.instance_encoder.native_encoder(*native_inputs(changed))
            torch.testing.assert_close(native_after, native_before, rtol=0, atol=0)
            self.assertFalse(torch.allclose(self.pretrained(changed)['logits'], before))

    def test_native_and_common_paths_trainable_with_disjoint_lr_groups(self):
        for original in (self.random, self.pretrained):
            model = copy.deepcopy(original).train()
            native = list(native_parameters(model))
            self.assertTrue(all(parameter.requires_grad for parameter in native))
            native_ids = {id(parameter) for parameter in native}
            named_ids = {id(parameter) for name, parameter in model.named_parameters() if name.startswith(NATIVE_PREFIX)}
            self.assertEqual(native_ids, named_ids)
            others = [parameter for parameter in model.parameters() if id(parameter) not in native_ids]
            optimizer = torch.optim.AdamW([dict(params=native, lr=3e-5), dict(params=others, lr=1e-4)])
            before = model.instance_encoder.native_encoder.embedding_node.weight.detach().clone()
            loss = F.cross_entropy(model(routing_batch())['logits'], torch.tensor([0, 1]))
            loss.backward()
            modules = [model.instance_encoder.native_encoder, model.instance_encoder.fusion,
                       model.instance_encoder.legacy_encoder.coord_proj,
                       model.instance_encoder.legacy_encoder.condition_encoder,
                       model.joint_layers, model.solver_encoder, model.score_head]
            for module in modules:
                gradients = [p.grad for p in module.parameters() if p.grad is not None]
                self.assertTrue(gradients)
                self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))
                self.assertGreater(sum(gradient.abs().sum().item() for gradient in gradients), 0.)
            optimizer.step()
            self.assertFalse(torch.equal(before, model.instance_encoder.native_encoder.embedding_node.weight))

    def test_offline_checkpoint_factory_and_geometry_path(self):
        model = self.pretrained
        self.assertIs(model.instance_encoder.geometry_residual, model.instance_encoder.legacy_encoder.geometry_residual)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'selector.pt'
            torch.save(dict(model=model.state_dict(), args=dict(model_params=self.pretrained_params)), path)
            checkpoint = torch.load(path, map_location='cpu', weights_only=False)
            params = copy.deepcopy(checkpoint['args']['model_params'])
            params['native_checkpoint']['path'] = '/unavailable/native.pt'
            params['native_provenance']['source']['path'] = '/unavailable/MTLModel.py'
            with patch('pathlib.Path.open', side_effect=AssertionError('Offline factory must not read external files')), \
                 patch('torch.load', side_effect=AssertionError('Factory must not reload native weights')):
                restored = make_r59_model(params, state_dict=checkpoint['model']).eval()
            batch = routing_batch()
            with torch.no_grad():
                torch.testing.assert_close(restored(batch)['logits'], model(batch)['logits'], rtol=0, atol=0)
        json.dumps(self.pretrained_params)

    def test_rejects_wrong_native_config_and_bad_depot_or_mask(self):
        for change in (dict(norm='layer'), dict(head_num=4), dict(encoder_layer_num=4)):
            with self.assertRaises(ValueError):
                PaddingAwareMTLEncoder(**dict(NATIVE_CONFIG, **change))
        with self.assertRaises(ValueError):
            initialize('C', GEOMETRY)
        batch = routing_batch()
        batch['node'][:, 0, 3] = 0.
        with self.assertRaises(ValueError):
            native_inputs(batch)
        batch = routing_batch()
        batch['node_mask'][:, 0] = False
        with self.assertRaises(ValueError):
            self.pretrained(batch)


if __name__ == '__main__':
    unittest.main()

"""Small synthetic CPU checks; never import/run a native donor or load weights."""

import copy
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn
import torch.nn.functional as F

from code.unified_selector.registry import PROBLEMS
from .pairwise_objective import LABEL_KEYS
from .pairwise_selector import maximin_logits
from .r88_model import make_model as make_original
from .r94_model import (
    ARMS, ARM_VARIANTS, COVERED_PROBLEMS, DONOR_FIELDS, FUSION_SEED,
    INITIALIZATION, MODEL_ROLE_ARM, PROJECTION_NAME, ROLE_SLOT, SELECTION_HEAD,
    SPEC, inputs_with_donor, make_model,
)
from .test_dual_stream import params as small_params
from .test_r76 import fixture_batch


class FakeDonor(nn.Module):
    """Deliberately tiny stand-in, not native encoder/compatibility evidence."""

    SUPPORTED_PROBLEMS = COVERED_PROBLEMS

    def __init__(self, variant='random'):
        super().__init__()
        self.variant = variant
        self.provenance = dict(source='fake_cpu_only', variant=variant)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(17 if variant == 'random' else 18)
            self.projection = nn.Linear(8, 128, bias=False)
        self.dropout = nn.Dropout(.8)
        self.register_buffer('marker', torch.tensor([1. if variant == 'random' else 2.]))
        self.calls = []

    def forward(self, batch):
        if PROBLEMS[int(batch['problem_id'])] not in self.SUPPORTED_PROBLEMS:
            raise AssertionError('Unsupported tasks must never reach the donor')
        if set(batch) & set(LABEL_KEYS):
            raise AssertionError('Supervision reached the frozen donor')
        self.calls.append(dict(training=self.training, grad=torch.is_grad_enabled(), keys=set(batch)))
        return self.dropout(self.projection(batch['node']))


def fixture_params():
    return dict(small_params(), encoder_layer_num=4, joint_layer_num=2, dropout=.1,
                local_geometry=True, geometry_mode='real', pair_mode='score_difference',
                solver_representation='handcrafted', architecture='pairwise', ignore_coord_dist=True)


def cached_batch(problem='CVRP', rows=2):
    batch, _ = fixture_batch(problem, rows)
    if problem == 'TSP':
        batch['node'] = batch['node'][..., :2].clone()
    elif problem != 'ATSP':
        batch['node'][..., ROLE_SLOT] = 0.
        batch['node'][:, 0, ROLE_SLOT] = 1.
    values = torch.arange(rows * 5 * 128, dtype=torch.float32).reshape(rows, 5, 128)
    batch.update(nco_nodes=(values * .13).sin(), nco_node_mask=batch['node_mask'].clone())
    return batch


def model_pair(arm='A'):
    params = fixture_params()
    donor = FakeDonor(ARM_VARIANTS[arm])
    torch.manual_seed(2)
    original = make_original(params, MODEL_ROLE_ARM)
    torch.manual_seed(2)
    model = make_model(params, arm, donor=donor)
    return model, original


def permute_nodes(batch, order):
    result = dict(batch)
    for key in ('node', 'node_mask', 'node_geom', *DONOR_FIELDS):
        if key in result:
            result[key] = result[key][:, order]
    if 'matrix' in result:
        result['matrix'] = result['matrix'][:, order][:, :, order]
    return result


class R94ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_original_parameter_identity_only_new_projection_and_matched_arms_rng(self):
        models = []
        self.assertEqual(ARMS, ('A', 'B'))
        for arm in ARMS:
            donor = FakeDonor(ARM_VARIANTS[arm])
            torch.manual_seed(2)
            original = make_original(fixture_params(), MODEL_ROLE_ARM)
            expected_rng = torch.get_rng_state().clone()
            torch.manual_seed(2)
            model = make_model(fixture_params(), arm, donor=donor)
            torch.testing.assert_close(torch.get_rng_state(), expected_rng, rtol=0, atol=0)
            retained = dict(original.named_parameters())
            trainable = {name: value for name, value in model.named_parameters() if value.requires_grad}
            self.assertEqual(set(trainable) - set(retained), {PROJECTION_NAME})
            self.assertFalse(set(retained) - set(trainable))
            for name, value in retained.items():
                torch.testing.assert_close(trainable[name], value, rtol=0, atol=0)
            self.assertEqual(len(model.instance_encoder.layers), 4)
            self.assertEqual(len(model.joint_layers), 2)
            self.assertEqual(model.solver_encoder.solver_features.shape[-1], 34)
            self.assertIsNone(model.instance_encoder.nco_projection.bias)
            self.assertFalse(model.instance_encoder.nco_norm.elementwise_affine)
            self.assertEqual(model.params['r94_schema_version'], 1)
            self.assertEqual(model.params['initialization'], INITIALIZATION)
            self.assertEqual(model.params['selection_head'], SELECTION_HEAD)
            self.assertEqual(model.model_spec, SPEC)
            self.assertEqual(model.frozen_donor['variant'], ARM_VARIANTS[arm])
            self.assertEqual(model.params['frozen_donor']['provenance'], donor.provenance)
            self.assertTrue(all(module.p == .1 for name, module in model.named_modules()
                                if isinstance(module, nn.Dropout) and not name.startswith('donor.')))
            models.append(trainable)
        for name in models[0]:
            torch.testing.assert_close(models[0][name], models[1][name], rtol=0, atol=0)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(FUSION_SEED)
            expected = nn.Linear(128, fixture_params()['embedding_dim'], bias=False)
        torch.testing.assert_close(models[0][PROJECTION_NAME], expected.weight, rtol=0, atol=0)

    def test_full18_coverage_original_pools_outputs_and_donor_dependence(self):
        model, original = model_pair()
        model.eval()
        original.eval()
        for problem in PROBLEMS:
            with self.subTest(problem=problem):
                batch = cached_batch(problem)
                output = model(batch)
                changed = dict(batch, nco_nodes=-batch['nco_nodes'].flip(-1))
                altered = model(changed)
                self.assertEqual(set(output), {'utility', 'logits', 'pair_margin', 'solver_mask'})
                self.assertEqual(output['logits'].shape, (2, len(batch['pool_ids'])))
                self.assertTrue(output['solver_mask'].all())
                torch.testing.assert_close(output['logits'], maximin_logits(output['pair_margin'], output['solver_mask']),
                                           rtol=0, atol=0)
                if problem in COVERED_PROBLEMS:
                    self.assertFalse(torch.equal(output['logits'], altered['logits']))
                else:
                    for key in output:
                        torch.testing.assert_close(output[key], altered[key], rtol=0, atol=0)
                        torch.testing.assert_close(output[key], original(batch)[key], rtol=0, atol=0)
        self.assertEqual(model.donor.calls, [])

    def test_early_fusion_enters_all_four_layers_and_gradients_stop_at_H(self):
        model, _ = model_pair()
        model.train()
        batch = cached_batch('OVRPBLTW', 3)
        batch['nco_nodes'].requires_grad_()
        entering = []

        def capture(module, args):
            args[0].retain_grad()
            entering.append(args[0])

        handles = [layer.register_forward_pre_hook(capture) for layer in model.instance_encoder.layers]
        output = model(batch)
        for handle in handles:
            handle.remove()
        branch = model.instance_encoder
        expected = (branch.coord_proj(batch['node'])
                    + branch.geometry_residual(batch['node'], batch['node_mask'], batch['n'], batch['node_geom'])
                    + branch.nco_projection(branch.nco_norm(batch['nco_nodes'].detach())))
        torch.testing.assert_close(entering[0][:, 3:], expected, rtol=0, atol=0)
        F.cross_entropy(output['logits'], batch['ind']).backward()
        self.assertEqual(len(entering), 4)
        for token in entering:
            self.assertTrue(torch.isfinite(token.grad).all())
            self.assertGreater(token.grad[:, 3:].abs().sum().item(), 0.)
        modules = [branch.nco_projection, branch.coord_proj, branch.geometry_residual,
                   branch.condition_encoder, *branch.layers, model.solver_encoder,
                   *model.joint_layers, model.pool, model.decoder, model.score_head]
        for module in modules:
            grads = [p.grad for p in module.parameters() if p.grad is not None]
            self.assertTrue(grads)
            self.assertTrue(all(torch.isfinite(grad).all() for grad in grads))
            self.assertGreater(sum(grad.abs().sum().item() for grad in grads), 0.)
        self.assertIsNone(batch['nco_nodes'].grad)
        self.assertTrue(all(p.grad is None and not p.requires_grad for p in model.donor.parameters()))

    def test_padding_and_synchronous_node_permutation_preserve_alignment(self):
        model, _ = model_pair()
        model.eval()
        batch = cached_batch()
        expected = model(batch)['logits']
        padded = dict(batch,
                      node=F.pad(batch['node'], (0, 0, 0, 3), value=123.),
                      matrix=F.pad(batch['matrix'], (0, 3, 0, 3), value=123.),
                      node_geom=F.pad(batch['node_geom'], (0, 0, 0, 3), value=99.),
                      node_mask=F.pad(batch['node_mask'], (0, 3), value=False),
                      nco_nodes=F.pad(batch['nco_nodes'], (0, 0, 0, 3), value=float('nan')),
                      nco_node_mask=F.pad(batch['nco_node_mask'], (0, 3), value=False))
        torch.testing.assert_close(model(padded)['logits'], expected, atol=2e-6, rtol=3e-5)
        order = torch.tensor([6, 2, 0, 7, 4, 1, 5, 3])
        permuted = permute_nodes(padded, order)
        torch.testing.assert_close(model(permuted)['logits'], expected, atol=2e-6, rtol=3e-5)
        # Keeping H in the old order is a different graph-feature alignment.
        wrong = dict(permute_nodes(batch, torch.tensor([2, 4, 0, 3, 1])),
                     nco_nodes=batch['nco_nodes'])
        self.assertFalse(torch.allclose(model(wrong)['logits'], expected, atol=1e-9, rtol=1e-8))
        invalid = dict(batch, nco_node_mask=~batch['nco_node_mask'])
        with self.assertRaises(ValueError):
            model(invalid)

    def test_input_isolation_and_cache_retention_across_role_preparation(self):
        model, _ = model_pair()
        model.eval()
        batch = cached_batch()
        expected = model(batch)
        changed = copy.deepcopy(batch)
        for key in LABEL_KEYS:
            if key in changed:
                value = changed[key]
                changed[key] = torch.full_like(value, float('nan') if value.is_floating_point() else 0)
        changed.update(winner=torch.full_like(batch['ind'], -999), performance_target=torch.tensor(float('nan')),
                       nco_costs=torch.tensor(float('nan')), nco_pooled_teacher=torch.tensor(float('nan')))
        prepared = inputs_with_donor(changed)
        self.assertFalse(set(prepared) & set(LABEL_KEYS))
        self.assertEqual({key for key in prepared if key.startswith('nco_')}, set(DONOR_FIELDS))
        self.assertIs(prepared['nco_nodes'], changed['nco_nodes'])
        for key, value in expected.items():
            torch.testing.assert_close(model(changed)[key], value, rtol=0, atol=0)
        from .r88_model import prepare_role_inputs

        def stripping_role(inputs, arm):
            result = prepare_role_inputs(inputs, arm)
            return {key: value for key, value in result.items() if key not in DONOR_FIELDS}

        with patch('code.V4.r94_model.prepare_role_inputs', side_effect=stripping_role):
            torch.testing.assert_close(model(batch)['logits'], expected['logits'], rtol=0, atol=0)

    def test_tsp_physical_node0_role_and_unsupported_donor_is_never_called(self):
        model, original = model_pair()
        model.eval()
        original.eval()
        batch = cached_batch('TSP')
        prepared = model.prepare_inputs(batch)
        self.assertEqual(prepared['node'][..., ROLE_SLOT].argmax(-1).tolist(), [0, 0])
        self.assertEqual(prepared['node'][..., ROLE_SLOT].sum(-1).tolist(), [1., 1.])
        order = torch.tensor([2, 4, 0, 3, 1])
        permuted = permute_nodes(prepared, order)
        self.assertEqual(model.prepare_inputs(permuted)['node'][..., ROLE_SLOT].argmax(-1).tolist(), [2, 2])
        torch.testing.assert_close(model(permuted)['logits'], model(batch)['logits'], atol=2e-6, rtol=3e-5)
        self.assertEqual(model.prepare_inputs(permute_nodes(batch, order))['node'][..., ROLE_SLOT].argmax(-1).tolist(), [0, 0])
        for problem in ('TSP', 'ATSP'):
            bad_cache = dict(cached_batch(problem), nco_nodes=torch.tensor(float('nan')), nco_node_mask=None)
            self.assertFalse(set(model.prepare_inputs(bad_cache)) & set(DONOR_FIELDS))
            torch.testing.assert_close(model(bad_cache)['logits'], original(bad_cache)['logits'], rtol=0, atol=0)
        self.assertEqual(model.donor.calls, [])

    def test_same_candidate_reorder_and_padding_keep_original_scorer(self):
        model, _ = model_pair()
        model.eval()
        batch = cached_batch('OVRPTW')
        output = model(batch)
        order = torch.tensor([3, 1, 5, 0, 4, 2])
        permuted = model(dict(batch, pool_ids=batch['pool_ids'][order]))
        for key in ('logits', 'utility'):
            torch.testing.assert_close(permuted[key], output[key][:, order], atol=2e-6, rtol=3e-5)
        torch.testing.assert_close(permuted['pair_margin'], output['pair_margin'][:, order][:, :, order],
                                   atol=2e-6, rtol=3e-5)
        ids = torch.cat([batch['pool_ids'], torch.tensor([-1, -1])])
        mask = torch.ones(2, len(ids), dtype=torch.bool)
        mask[:, -2:] = False
        padded = model(dict(batch, pool_ids=ids, solver_mask=mask))
        torch.testing.assert_close(padded['logits'][:, :-2], output['logits'], atol=2e-6, rtol=3e-5)
        self.assertTrue(torch.isneginf(padded['logits'][:, -2:]).all())
        self.assertTrue((padded['pair_margin'][:, -2:] == 0).all())

    def test_frozen_eval_donor_cache_policy_raw_fallback_and_cache_validation(self):
        model, _ = model_pair()
        for mode in (True, False, True):
            model.train(mode)
            self.assertFalse(any(module.training for module in model.donor.modules()))
            self.assertTrue(all(not p.requires_grad for p in model.donor.parameters()))
        batch = cached_batch()
        raw = {key: value for key, value in batch.items() if key not in DONOR_FIELDS}
        with self.assertRaises(ValueError):
            model(dict(raw, split='lib'))
        model.eval()
        for split in ('train', 'val', 'validation'):
            with self.subTest(split=split), self.assertRaises(ValueError):
                model(dict(raw, split=split))
        raw['node'].requires_grad_()
        prepared = model.prepare_inputs(raw)
        self.assertEqual(len(model.donor.calls), 1)
        self.assertFalse(model.donor.calls[0]['training'])
        self.assertFalse(model.donor.calls[0]['grad'])
        self.assertFalse(prepared['nco_nodes'].requires_grad)
        self.assertEqual(prepared['nco_nodes'].dtype, torch.float16)
        self.assertNotIn('nco_nodes', raw)
        model(prepared)['logits'].sum().backward()
        self.assertEqual(len(model.donor.calls), 1)
        self.assertTrue(all(p.grad is None for p in model.donor.parameters()))
        for invalid in (dict(batch, nco_nodes=batch['nco_nodes'][..., :127]),
                        dict(batch, nco_nodes=torch.full_like(batch['nco_nodes'], float('nan'))),
                        dict(batch, nco_node_mask=batch['nco_node_mask'].float()),
                        {key: value for key, value in batch.items() if key != 'nco_node_mask'}):
            with self.assertRaises(ValueError):
                model(invalid)

    def test_untagged_raw_eval_matches_fp16_cache_exactly_and_is_label_isolated(self):
        for arm in ARMS:
            with self.subTest(arm=arm):
                model, _ = model_pair(arm)
                model.eval()
                raw = {key: value for key, value in cached_batch().items() if key not in DONOR_FIELDS}
                with torch.no_grad():
                    native_fp32 = model.donor(inputs_with_donor(raw))
                self.assertEqual(native_fp32.dtype, torch.float32)
                quantized = native_fp32.half()
                self.assertTrue((native_fp32 != quantized.float()).any())
                cached = dict(raw, nco_nodes=quantized, nco_node_mask=raw['node_mask'])
                expected = model(cached)
                normalization_inputs = []
                handle = model.instance_encoder.nco_norm.register_forward_pre_hook(
                    lambda module, args: normalization_inputs.append(args[0].detach().clone()))
                actual = model(raw)
                handle.remove()
                torch.testing.assert_close(normalization_inputs[0], quantized.float(), rtol=0, atol=0)
                for key, value in expected.items():
                    torch.testing.assert_close(actual[key], value, rtol=0, atol=0)
                self.assertEqual(len(model.donor.calls), 2)
                changed = copy.deepcopy(raw)
                changed['costs'].fill_(float('nan'))
                changed['cost_regret'].fill_(float('nan'))
                changed['ind'].fill_(-123)
                changed['cmp_target'].fill_(float('nan'))
                changed['winner'] = torch.full_like(raw['ind'], -999)
                for key, value in expected.items():
                    torch.testing.assert_close(model(changed)[key], value, rtol=0, atol=0)

    def test_default_factory_defers_donor_construction_and_uses_supported_API(self):
        calls = []

        def build_fake(variant, checkpoint_path=None):
            calls.append((variant, checkpoint_path))
            return FakeDonor(variant)

        module = SimpleNamespace(StaticReLDDonor=build_fake)
        with patch.dict(sys.modules, {'code.V4.r94_donor': module}):
            for arm in ARMS:
                model = make_model(fixture_params(), arm)
                self.assertIsInstance(model.donor, FakeDonor)
        self.assertEqual(calls, [('random', None), ('pretrained', None)])

    def test_strict_config_guards_and_strict_checkpoint_includes_frozen_donor(self):
        for key, value in (('encoder_layer_num', 3), ('joint_layer_num', 1), ('dropout', .2),
                           ('local_geometry', False), ('geometry_mode', 'zero'),
                           ('selection_head', 'direct_attention'), ('pair_mode', 'explicit_pair'),
                           ('solver_representation', 'code'), ('role_arm', 'A_canonical_anchor'),
                           ('r94_schema_version', 2), ('initialization', 'trained_selector'),
                           ('model_spec', dict(SPEC, projection_bias=True)), ('residual_head', True)):
            with self.subTest(key=key), self.assertRaises(ValueError):
                make_model(dict(fixture_params(), **{key: value}), 'A', donor=FakeDonor())
        with self.assertRaises(ValueError):
            make_model(fixture_params(), 'C', donor=FakeDonor())
        with self.assertRaises(ValueError):
            make_model(fixture_params(), 'B', donor=FakeDonor())
        for arm in ARMS:
            model, _ = model_pair(arm)
            model.eval()
            state = copy.deepcopy(model.state_dict())
            donor_keys = [key for key in state if key.startswith('donor.')]
            self.assertTrue(donor_keys)
            restored = make_model(model.params, arm, state_dict=state, donor=FakeDonor(ARM_VARIANTS[arm])).eval()
            torch.testing.assert_close(restored(cached_batch())['logits'], model(cached_batch())['logits'], rtol=0, atol=0)
            self.assertTrue(all(not p.requires_grad for p in restored.donor.parameters()))
            for missing in (PROJECTION_NAME, donor_keys[0], 'score_head.3.weight'):
                incomplete = dict(state)
                del incomplete[missing]
                with self.subTest(missing=missing), self.assertRaises(RuntimeError):
                    make_model(model.params, arm, state_dict=incomplete, donor=FakeDonor(ARM_VARIANTS[arm]))
            mismatch = copy.deepcopy(model.params)
            mismatch['frozen_donor']['provenance']['source'] = 'different_source'
            with self.assertRaises(ValueError):
                make_model(mismatch, arm, donor=FakeDonor(ARM_VARIANTS[arm]))


if __name__ == '__main__':
    unittest.main()

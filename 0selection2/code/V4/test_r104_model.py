"""Small offline CPU checks of the real R104 capacities; no dataset/job IO."""

import copy
import unittest
from unittest.mock import patch

import torch

from code.unified_selector.registry import PROBLEMS
from .dual_stream import DualStreamSelector
from .pairwise_objective import LABEL_KEYS, inputs_only
from .pairwise_selector import ScoreDifferenceSelector, maximin_logits, rank_by_solver_ids
from .r76_experiment import RDropObjective
from .r77_data import parameter_hash
from .r84_experiment import make_model as make_r84_model
from .r88_model import RoleInputSelector, make_model as make_r88_model
from .r104_model import (
    ARMS, CAPACITIES, CAPACITY_FIELDS, CAPACITY_INIT_SEED, ORIGINAL, ROLE_ARM,
    SPEC, initialize_shared, make_model, model_dimensions,
)
from .test_r76 import fixture_batch
from .V4Model import get_default_model_params


def full18_params():
    params = get_default_model_params()
    params.update(architecture='pairwise', joint_layer_num=2, local_geometry=True,
                  geometry_mode='real', ignore_coord_dist=True, native_winner=True,
                  decision_head='classification', pair_mode='score_difference',
                  solver_representation='handcrafted', solver_feature_weight=1.0, sdpa=True)
    return params


def batch_for(problem, rows=2):
    batch, _ = fixture_batch(problem, rows)
    if problem == 'TSP':
        batch['node'] = batch['node'][..., :2].clone()
    return batch


class R104ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.params = full18_params()
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(2)
            cls.original = make_r84_model(cls.params, 'A_baseline')
        reference = {name: value.detach().clone() for name, value in cls.original.named_parameters()}
        cls.initial = dict(seed=2, initialization='scratch', model_params=cls.params,
                           trainable=reference, trainable_sha256=parameter_hash(reference))
        cls.geometry = dict(mean=[.2] * 12, std=[1.3] * 12, fitted_on='synthetic TRAIN-only fixture')
        cls.models = {arm: initialize_shared(cls.initial, cls.geometry, arm)[0] for arm in ARMS}

    def setUp(self):
        for model in self.models.values():
            model.eval()
            model.zero_grad(set_to_none=True)

    def test_actual_dimensions_counts_and_only_explicit_capacity_overrides(self):
        expected_counts = {ARMS[0]: 7288969, ARMS[1]: 3426057}
        expected_common = {ARMS[0]: 1417, ARMS[1]: 1842953}
        self.assertEqual(sum(p.numel() for p in self.original.parameters()), 1842953)
        self.assertEqual(SPEC['original'], ORIGINAL)
        self.assertEqual(SPEC['arms'], CAPACITIES)
        for arm, model in self.models.items():
            with self.subTest(arm=arm):
                dims = model_dimensions(model)
                self.assertEqual({key: dims[key] for key in CAPACITY_FIELDS}, CAPACITIES[arm])
                self.assertEqual(dims['solver_feature_dim'], 34)
                self.assertEqual(dims['geometry_input_dim'], 21)
                self.assertEqual(dims['geometry_hidden_dim'], 64)
                self.assertEqual(dims['geometry_output_dim'], dims['embedding_dim'])
                self.assertEqual(dims['score_head_input_dim'], 3 * dims['embedding_dim'])
                self.assertEqual(dims['attention_projection_dim'], dims['embedding_dim'])
                self.assertEqual(dims['attention_modules'], 9 if arm == ARMS[0] else 17)
                self.assertEqual(sum(p.numel() for p in model.parameters()), expected_counts[arm])
                receipt = model.initialization_receipt
                self.assertEqual(receipt['parameter_count'], expected_counts[arm])
                self.assertEqual(receipt['common_parameter_count'], expected_common[arm])
                self.assertEqual(receipt['common_parameter_count']
                                 + receipt['independently_initialized_parameter_count'], expected_counts[arm])
                self.assertEqual(model.capacity_report['actual_dimensions'], dims)
                for key, value in self.params.items():
                    if key not in CAPACITY_FIELDS:
                        self.assertEqual(model.params[key], value)
                self.assertFalse(hasattr(model, 'performance_head'))
                self.assertFalse(hasattr(model.instance_encoder, 'summary_proj'))
                self.assertTrue(all(p.device.type == 'cpu' for p in model.parameters()))
        self.assertEqual(self.params['embedding_dim'], 128)
        self.assertEqual(self.params['encoder_layer_num'], 4)

    @torch.no_grad()
    def test_full18_whole_original_score_path_exact_numeric_and_dropout_rng_parity(self):
        for arm, model in self.models.items():
            self.assertIs(type(model).forward, RoleInputSelector.forward)
            self.assertIs(type(model).encode_state, DualStreamSelector.encode_state)
            self.assertIs(type(model).encode_pairs, DualStreamSelector.encode_pairs)
            original_path = make_r88_model(model.params, ROLE_ARM, model.state_dict()).train()
            model.train()
            for problem in PROBLEMS:
                with self.subTest(arm=arm, problem=problem):
                    batch = batch_for(problem)
                    torch.random.default_generator.manual_seed(226)
                    expected = original_path(batch)
                    rng = torch.get_rng_state().clone()
                    torch.random.default_generator.manual_seed(226)
                    actual = model(batch)
                    self.assertEqual(set(actual), {'logits', 'pair_margin', 'solver_mask', 'utility'})
                    for key in expected:
                        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
                    torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
                    self.assertTrue(torch.isfinite(actual['utility']).all())
                    self.assertTrue(torch.isfinite(actual['logits']).all())
                    difference = actual['utility'][:, :, None] - actual['utility'][:, None, :]
                    torch.testing.assert_close(actual['pair_margin'], difference, rtol=0, atol=0)
                    torch.testing.assert_close(actual['logits'],
                        maximin_logits(difference, actual['solver_mask']), rtol=0, atol=0)
                    prepared = model.prepare_role_inputs(batch)
                    # R104 does not replace or bypass the complete original scorer.
                    torch.random.default_generator.manual_seed(226)
                    direct = ScoreDifferenceSelector.forward(model, prepared)
                    torch.testing.assert_close(actual['logits'], direct['logits'], rtol=0, atol=0)

    def test_finite_original_loss_backward_reaches_all_added_layers_and_original_stages(self):
        for arm, model in self.models.items():
            for problem in ('TSP', 'ATSP', 'OVRPBLTW'):
                with self.subTest(arm=arm, problem=problem):
                    model.train().zero_grad(set_to_none=True)
                    batch = batch_for(problem)
                    loss, parts = RDropObjective(model)(batch, .35)
                    self.assertTrue(torch.isfinite(loss))
                    self.assertTrue(torch.isfinite(parts).all())
                    loss.backward()
                    projection = (model.instance_encoder.matrix_proj if problem == 'ATSP'
                                  else model.instance_encoder.coord_proj)
                    stages = [projection, model.solver_encoder.solver_emb,
                              model.solver_encoder.feature_mlp, *model.instance_encoder.layers,
                              model.pool, model.decoder, model.score_head]
                    stages.extend(update for layer in model.joint_layers
                                  for update in (layer.node_update, layer.solver_update))
                    for stage in stages:
                        gradients = [p.grad for p in stage.parameters() if p.grad is not None]
                        self.assertTrue(gradients)
                        self.assertTrue(all(torch.isfinite(g).all() for g in gradients))
                        self.assertGreater(sum(g.abs().sum().item() for g in gradients), 0.)
                    self.assertIsNone(model.solver_encoder.solver_features.grad)

    @torch.no_grad()
    def test_native_candidate_order_masks_padding_singleton_and_ties(self):
        for arm, model in self.models.items():
            for problem in ('TSP', 'ATSP', 'OVRPBLTW'):
                with self.subTest(arm=arm, problem=problem):
                    batch = batch_for(problem)
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
                    # Different native masks/order per row agree with their compact pools.
                    row_ids = batch['pool_ids'].repeat(2, 1)
                    row_ids[1] = row_ids[1].flip(0)
                    row_mask = torch.ones(2, k, dtype=torch.bool)
                    row_mask[0, -1] = False
                    row_mask[1, 0] = False
                    masked = model(dict(batch, pool_ids=row_ids, solver_mask=row_mask))
                    shuffled = model(dict(batch, pool_ids=row_ids[:, order], solver_mask=row_mask[:, order]))
                    torch.testing.assert_close(shuffled['logits'], masked['logits'][:, order], rtol=1e-4, atol=2e-6)
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
        ties = rank_by_solver_ids(torch.zeros(2, 3), ids)
        torch.testing.assert_close(ids[ties], torch.tensor([[3, 16, 18], [3, 16, 18]]))

    @torch.no_grad()
    def test_label_cost_and_donor_isolation_without_input_mutation(self):
        for arm, model in self.models.items():
            batch = batch_for('TSP')
            snapshot = copy.deepcopy(batch)
            expected = model(inputs_only(batch))
            poisoned = dict(batch)
            for name in LABEL_KEYS:
                poisoned[name] = object()
            for name in ('winner', 'query_labels', 'query_costs', 'query_winner',
                         'donor_features', 'donor_labels', 'donor_costs', 'context'):
                poisoned[name] = object()
            observed = []
            hook = model.instance_encoder.register_forward_pre_hook(
                lambda module, arguments: observed.append(arguments[0]))
            try:
                actual = model(poisoned)
            finally:
                hook.remove()
            self.assertFalse(set(LABEL_KEYS) & set(observed[0]))
            for key in expected:
                torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
            for key, value in snapshot.items():
                if torch.is_tensor(value):
                    torch.testing.assert_close(batch[key], value, rtol=0, atol=0)
            self.assertFalse(any('donor' in name or 'context' in name for name, _ in model.named_parameters()))
            for problem in ('TSP', 'ATSP', 'OVRPBLTW'):
                raw = batch_for(problem)
                prepared = model.prepare_role_inputs(raw)
                for key in ('matrix', 'node_geom', 'node_mask', 'cbits', 'problem_desc', 'coord_dist'):
                    self.assertIs(prepared[key], raw[key])
                if problem != 'TSP':
                    self.assertIs(prepared['node'], raw['node'])
                else:
                    torch.testing.assert_close(prepared['node'][..., :2], raw['node'], rtol=0, atol=0)
                    self.assertEqual(prepared['node'][:, 0, 3].tolist(), [1., 1.])
                    self.assertEqual(prepared['node'][..., 3].sum(-1).tolist(), [1., 1.])

    def test_fresh_source_reuse_shapes_determinism_rng_and_no_alias(self):
        reference = self.initial['trainable']
        for arm, model in self.models.items():
            receipt = model.initialization_receipt
            self.assertEqual(receipt['r84_fresh_trainable_sha256'], self.initial['trainable_sha256'])
            self.assertFalse(receipt['historical_checkpoint_loaded'])
            self.assertEqual(receipt['removed_parameter_names'], [])
            self.assertEqual(receipt['capacity_init_seed'], CAPACITY_INIT_SEED)
            actual = dict(model.named_parameters())
            for name in receipt['common_parameter_names']:
                torch.testing.assert_close(actual[name], reference[name], rtol=0, atol=0)
                self.assertNotEqual(actual[name].data_ptr(), reference[name].data_ptr())
            for name, shapes in receipt['resized_parameter_shapes'].items():
                self.assertEqual(shapes['original'], list(reference[name].shape))
                self.assertEqual(shapes['actual'], list(actual[name].shape))
            torch.random.default_generator.manual_seed(999)
            rng = torch.get_rng_state().clone()
            first = make_model(self.params, arm)
            torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
            torch.rand(23)
            second = make_model(self.params, arm)
            self.assertEqual(parameter_hash(dict(first.named_parameters())),
                             parameter_hash(dict(second.named_parameters())))
            self.assertEqual(parameter_hash(dict(first.named_parameters())), receipt['trainable_sha256'])
            torch.testing.assert_close(model.instance_encoder.geometry_residual.mean, torch.tensor(self.geometry['mean']))
            torch.testing.assert_close(model.instance_encoder.geometry_residual.std, torch.tensor(self.geometry['std']))
        self.assertEqual(parameter_hash(reference), self.initial['trainable_sha256'])
        deep = self.models[ARMS[1]]
        self.assertFalse(torch.equal(deep.instance_encoder.layers[0].attn.Wq.weight,
                                     deep.instance_encoder.layers[4].attn.Wq.weight))

    @torch.no_grad()
    def test_strict_fresh_state_restore_and_reject_cross_arm_or_incomplete_state(self):
        for arm, model in self.models.items():
            restored = make_model(model.params, arm, model.state_dict()).eval()
            self.assertTrue(restored.initialization_receipt['restored_trainables_match_fresh'])
            self.assertTrue(restored.initialization_receipt['state_dict_restored'])
            for name, value in model.state_dict().items():
                torch.testing.assert_close(restored.state_dict()[name], value, rtol=0, atol=0)
            for problem in ('TSP', 'ATSP'):
                batch = batch_for(problem)
                for key, value in model(batch).items():
                    torch.testing.assert_close(restored(batch)[key], value, rtol=0, atol=0)
            missing = dict(model.state_dict())
            missing.pop('score_head.3.weight')
            with self.assertRaises(RuntimeError):
                make_model(model.params, arm, missing)
        with self.assertRaises(ValueError):
            make_model(self.models[ARMS[0]].params, ARMS[1])

    def test_reject_unbounded_heads_roles_nonfresh_sources_and_bad_geometry(self):
        for params, arm in ((self.params, 'C_unbounded'),
                            (dict(self.params, seed=3), ARMS[0]),
                            (dict(self.params, role_arm='A_canonical_anchor'), ARMS[0]),
                            (dict(self.params, solver_representation='code'), ARMS[0]),
                            (dict(self.params, pair_mode='explicit_pair'), ARMS[0]),
                            (dict(self.params, residual_head=True), ARMS[0]),
                            (dict(self.params, selection_head='direct_attention'), ARMS[0]),
                            (dict(self.params, capacity_spec=dict(CAPACITIES[ARMS[0]], head_num=8)), ARMS[0])):
            with self.subTest(arm=arm, params=params), self.assertRaises(ValueError):
                make_model(params, arm)
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

    def test_factory_has_no_checkpoint_file_or_gpu_initialization_calls(self):
        with (patch('torch.load', side_effect=AssertionError('No checkpoint file reads')),
              patch('torch.cuda._lazy_init', side_effect=AssertionError('No GPU initialization'))):
            for arm in ARMS:
                model = make_model(self.params, arm)
                self.assertTrue(all(p.device.type == 'cpu' for p in model.parameters()))


if __name__ == '__main__':
    unittest.main()

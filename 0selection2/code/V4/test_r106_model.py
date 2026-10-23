"""Small CPU implementation witnesses using real pinned native encoder weights.

No dataset rows, GPU, training job, scientific selection or protocol mutation.
"""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from code.unified_selector.registry import (POOLS, PROBLEMS, S2I, constraint_bits,
                                            problem_descriptor)
from . import r106_model as models
from . import r102_donor as native
from . import r94_donor as reld
from .pairwise_objective import comparison_targets
from .pairwise_selector import maximin_logits
from .r76_experiment import RDropObjective
from .r77_data import parameter_hash
from .r88_model import make_model as original_model, prepare_role_inputs
from .r106_donor import component_for_problem, matnet_roles


PREPARED = Path(__file__).resolve().parent / 'runs/R84_new90k_attention/prepared'


def raw_batch(problem, lengths=(5, 8), slots=11):
    """Audited layouts, non-prefix masks, asymmetric ATSP and physical depot0."""
    generator = torch.Generator().manual_seed(10600 + PROBLEMS.index(problem))
    rows = len(lengths)
    mask = torch.zeros(rows, slots, dtype=torch.bool)
    for row, count in enumerate(lengths):
        mask[row, torch.linspace(0, slots - 1, count).long()] = True
    excluded = {'RELD_MTL'}
    if 'B' in problem:
        excluded.update(('RouteFinder', 'MoSES_RF', 'MoSES_CaDA'))
    pool = [S2I[name] for name in POOLS[problem] if name not in excluded]
    result = dict(
        problem_id=PROBLEMS.index(problem), problem_name=problem,
        kind='matrix' if problem == 'ATSP' else 'coord',
        n=torch.tensor(lengths), node_mask=mask, pool_ids=torch.tensor(pool),
        coord_dist=torch.full((rows,), 4),
        cbits=torch.tensor([constraint_bits(problem)] * rows, dtype=torch.float32),
        problem_desc=torch.tensor([problem_descriptor(problem)] * rows, dtype=torch.float32),
    )
    if problem == 'ATSP':
        matrix = torch.rand(rows, slots, slots, generator=generator)
        matrix.diagonal(dim1=1, dim2=2).zero_()
        result['matrix'] = matrix.masked_fill(~(mask[:, :, None] & mask[:, None, :]), float('nan'))
        result['d_in'] = 0
    else:
        width = 2 if problem == 'TSP' else 4 if problem == 'CVRP' else 8
        node = torch.zeros(rows, slots, width)
        node[..., :2] = torch.rand(rows, slots, 2, generator=generator)
        if width > 2:
            node[..., 2] = torch.rand(rows, slots, generator=generator) * .2
            if 'B' in problem:
                node[:, 1::2, 2] *= -1
            node[:, 0, 2] = 0.
            node[:, 0, 3] = 1.
            if width == 8 and 'L' in problem:
                node[:, 1:, 4] = 3.
            if width == 8 and 'TW' in problem:
                node[:, 1:, 5] = .2
                node[:, 1:, 6] = .1
                node[:, 1:, 7] = .9
        result.update(node=node.masked_fill(~mask[..., None], float('nan')), d_in=width,
                      node_geom=torch.zeros(rows, slots, 12))
    k = len(pool)
    costs = np.stack([1. + np.roll(np.arange(k) * .01, row) for row in range(rows)]).astype(np.float64)
    result.update(comparison_targets(costs))
    result.update(costs=torch.from_numpy(costs.astype(np.float32)),
                  ind=torch.from_numpy(costs.argmin(1)))
    return result


def exact_reference(donor, batch):
    """Original unmodified native forwards, one tiny unpadded row at a time."""
    clean = models.input_only(batch)
    problem = clean['problem_name']
    result = torch.zeros(*clean['node_mask'].shape, models.WIDTH_BY_PROBLEM[problem])
    for row, count in enumerate(clean['n'].tolist()):
        positions = clean['node_mask'][row].nonzero().flatten()
        if problem == 'TSP':
            hidden = donor.tsp(clean['node'][row:row + 1, positions, :2])
        elif problem == 'ATSP':
            raw = clean['matrix'][row:row + 1, positions][:, :, positions]
            roles = (raw.new_zeros(1, count, 256), torch.eye(256)[:count].unsqueeze(0))
            hidden = torch.cat(donor.atsp(*roles, raw), -1)
        else:
            single = {key: value[row:row + 1] if torch.is_tensor(value) and key != 'pool_ids'
                      else value for key, value in clean.items()}
            hidden, _ = donor.routing.encoder(**reld.build_inputs(single))
        result[row, positions] = hidden.squeeze(0)
    return result


class R106ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        initial = torch.load(PREPARED / 'identicalinit_seed2.pt', map_location='cpu', weights_only=False)
        with (PREPARED / 'geometry_stats.json').open() as stream:
            geometry = json.load(stream)
        cls.initial, cls.geometry = initial, geometry
        cls.a, cls.params_a, cls.receipt_a = models.initialize_shared(initial, geometry, 'A_frozen')
        cls.b, cls.params_b, cls.receipt_b = models.initialize_shared(initial, geometry, 'B_finetune')
        cls.initial_state = copy.deepcopy(cls.b.state_dict())

    def setUp(self):
        torch.random.default_generator.manual_seed(1062)
        for model in (self.a, self.b):
            model.load_state_dict(self.initial_state, strict=True)
            model.zero_grad(set_to_none=True)
            model.eval()
            self.assertIsNone(model._native_pair)

    def test_01_actual_counts_full_initial_state_and_original_structure(self):
        self.assertEqual(models.ARMS, ('A_frozen', 'B_finetune'))
        for arm, model, receipt in ((models.ARMS[0], self.a, self.receipt_a),
                                    (models.ARMS[1], self.b, self.receipt_b)):
            counts = models.parameter_counts(model)
            self.assertEqual(counts, receipt['parameter_counts'])
            self.assertEqual(counts['total'], 11965097)
            self.assertEqual(counts['selector_trainable'], 1941257)
            self.assertEqual(counts['projections'], 98304)
            self.assertEqual(counts['native'], 10023840)
            self.assertEqual(counts['native_trainable'], 0 if arm == 'A_frozen' else 10023840)
            self.assertEqual(len(model.instance_encoder.layers), 4)
            self.assertEqual(len(model.joint_layers), 2)
            self.assertEqual(model.solver_encoder.solver_features.shape[-1], 34)
            self.assertFalse(any('decoder' in name for name, _ in model.donor.named_modules()))
            self.assertEqual(parameter_hash(model.state_dict()), receipt['initial_state_sha256'])
            self.assertFalse(any(key.startswith(('r102_', 'r94_')) for key in model.params))
            self.assertNotIn('frozen_donor', model.params)
            model.train()
            self.assertTrue(all(not module.training for module in model.donor.modules()))
            self.assertTrue(all(getattr(p, '_r106_native', False) for p in model.donor.parameters()))
            self.assertTrue(all(p.requires_grad == (arm == 'B_finetune') for p in model.donor.parameters()))
        self.assertEqual(self.receipt_a['initial_state_sha256'], self.receipt_b['initial_state_sha256'])
        for name, value in self.a.state_dict().items():
            self.assertTrue(torch.equal(value, self.b.state_dict()[name]), name)
        for name, parameter in self.b.named_parameters():
            if name in self.initial['trainable']:
                self.assertTrue(torch.equal(parameter, self.initial['trainable'][name]), name)
        print('R106 initial_state_sha256:', self.receipt_b['initial_state_sha256'])
        print('R106 actual parameter counts:', self.receipt_b['parameter_counts'])

    def test_02_masked_native_vs_original_exact_n_and_padding(self):
        with torch.no_grad(), reld._native_precision('cpu'):
            for problem in ('TSP', 'CVRP', 'ATSP', 'OVRPBLTW'):
                batch = raw_batch(problem)
                expected = exact_reference(self.b.donor, batch)
                actual = self.b.donor(models.input_only(batch))
                valid = batch['node_mask']
                difference = (actual[valid] - expected[valid]).abs()
                print(f'R106 native/exact {problem}: maxabs={difference.max().item():.9g} '
                      f'rms={difference.square().mean().sqrt().item():.9g}')
                # Tiny MatNet graphs amplify FP32 spatial-reduction differences.
                # Do not weaken the much tighter ELG/ReLD comparison to match it.
                atol, rtol = (6e-3, 3e-4) if problem == 'ATSP' else (1e-5, 1e-5)
                torch.testing.assert_close(actual[valid], expected[valid], atol=atol, rtol=rtol)
                self.assertLess(difference.square().mean().sqrt().item(), 5e-4)
                self.assertEqual(actual[~valid].count_nonzero().item(), 0)
                # Repack the SAME raw valid data into a shorter prefix batch.
                for row, count in enumerate(batch['n'].tolist()):
                    positions = valid[row].nonzero().flatten()
                    single = {key: value[row:row + 1] if torch.is_tensor(value) and key != 'pool_ids'
                              else value for key, value in models.input_only(batch).items()}
                    single['node_mask'] = torch.ones(1, count, dtype=torch.bool)
                    if problem == 'ATSP':
                        single['matrix'] = batch['matrix'][row:row + 1, positions][:, :, positions]
                    else:
                        single['node'] = batch['node'][row:row + 1, positions]
                        single['node_geom'] = batch['node_geom'][row:row + 1, positions]
                    compact = self.b.donor(single)
                    torch.testing.assert_close(actual[row, positions], compact[0], atol=atol, rtol=rtol)

    def test_03_native_eval_is_deterministic_and_rng_free(self):
        self.b.train()
        for problem in ('TSP', 'CVRP', 'ATSP'):
            batch = models.input_only(raw_batch(problem))
            before = torch.get_rng_state().clone()
            first, second = self.b.donor(batch), self.b.donor(batch)
            self.assertTrue(torch.equal(before, torch.get_rng_state()), problem)
            torch.testing.assert_close(first, second, atol=0, rtol=0)
            self.assertTrue(first.requires_grad)

    def test_04_full18_forward_whitelist_and_external_cache_rejection(self):
        observed = []
        hook = self.a.donor.register_forward_pre_hook(lambda module, args: observed.append(set(args[0])))
        try:
            with torch.no_grad():
                for problem in PROBLEMS:
                    batch = raw_batch(problem)
                    expected = self.a(batch)
                    changed = dict(batch, costs=object(), ind=object(), winner=object(),
                                   arbitrary_supervision=object(), cmp_target=object())
                    actual = self.a(changed)
                    torch.testing.assert_close(actual['logits'], expected['logits'], atol=0, rtol=0)
                    self.assertTrue(torch.isfinite(actual['logits']).all(), problem)
                    torch.testing.assert_close(actual['logits'],
                        maximin_logits(actual['pair_margin'], actual['solver_mask']), atol=0, rtol=0)
                    self.assertTrue(all(keys <= set(models.INPUT_ALLOWLIST) for keys in observed))
                    with self.assertRaises(ValueError):
                        self.a(dict(batch, nco_nodes=torch.zeros(1)))
                    with self.assertRaises(ValueError):
                        self.a(dict(batch, **{models.NATIVE_FIELD: torch.zeros(1)}))
        finally:
            hook.remove()

    def test_05_first_loss_gradients_reach_b_native_but_not_a(self):
        for model in (self.a, self.b):
            model.train()
            for problem in ('TSP', 'CVRP', 'ATSP'):
                model.zero_grad(set_to_none=True)
                batch = raw_batch(problem)
                with model.native_pair_context(models.input_only(batch)):
                    loss, _ = RDropObjective(model)(batch, .17)
                loss.backward()
                group = component_for_problem(problem)
                prefix = {'TSP': 'tsp.', 'ATSP': 'atsp.', 'VRP': 'routing.'}[group]
                native_gradients = [p.grad for name, p in model.donor.named_parameters()
                                    if name.startswith(prefix) and p.grad is not None]
                if model.arm == 'A_frozen':
                    self.assertFalse(native_gradients)
                    self.assertTrue(all(p.grad is None for p in model.donor.parameters()))
                else:
                    self.assertTrue(native_gradients, problem)
                    self.assertTrue(all(torch.isfinite(g).all() for g in native_gradients))
                    self.assertGreater(sum(g.abs().sum().item() for g in native_gradients), 0., problem)
                    projection = model.instance_encoder.nco_projections[group].weight
                    self.assertGreater(projection.grad.abs().sum().item(), 0.)
                    if group == 'VRP':
                        gate = model.donor.routing.encoder.layers[0].feedForward
                        self.assertGreater(gate.w_gate.grad.abs().sum().item(), 0.)
                        self.assertIsNone(gate.w_noise.grad)
                self.assertTrue(all(not module.training for module in model.donor.modules()))

    def test_06_pair_context_matches_two_online_graphs_loss_and_gradients(self):
        for model, problems in ((self.b, ('TSP', 'CVRP', 'ATSP')), (self.a, ('CVRP',))):
            model.train()
            objective = RDropObjective(model)
            for problem in problems:
                batch = raw_batch(problem)
                rng = torch.get_rng_state().clone()
                model.zero_grad(set_to_none=True)
                with patch.object(model.donor, 'forward', wraps=model.donor.forward) as forward:
                    separate, separate_parts = objective(batch, .17)
                    self.assertEqual(forward.call_count, 2)
                separate.backward()
                gradients = {name: p.grad.clone() for name, p in model.named_parameters() if p.grad is not None}
                model.zero_grad(set_to_none=True)
                torch.set_rng_state(rng)
                with patch.object(model.donor, 'forward', wraps=model.donor.forward) as forward:
                    with model.native_pair_context(models.input_only(batch)):
                        shared, shared_parts = objective(batch, .17)
                    self.assertEqual(forward.call_count, 1)
                self.assertIsNone(model._native_pair)
                shared.backward()
                torch.testing.assert_close(shared, separate, atol=2e-6, rtol=2e-6)
                torch.testing.assert_close(shared_parts, separate_parts, atol=2e-6, rtol=2e-6)
                current = {name: p.grad for name, p in model.named_parameters() if p.grad is not None}
                self.assertEqual(set(current), set(gradients))
                for name, expected in gradients.items():
                    # A common additive bias before spatial norm has zero exact
                    # derivative; tiny-N MatNet can leave FP32 cancellation dust.
                    cancellation_bias = (name.startswith('donor.atsp.') and name.endswith(
                        ('multi_head_combine.bias', 'feed_forward.W2.bias', 'mixed_score_MHA.mix2_bias')))
                    atol = 3e-4 if cancellation_bias else 3e-6
                    torch.testing.assert_close(current[name], expected, atol=atol, rtol=3e-3,
                                               msg=lambda msg: f'{model.arm}/{problem}/{name}: {msg}')
                numerator = sum((current[name] - value).double().square().sum().item()
                                for name, value in gradients.items())
                denominator = sum(value.double().square().sum().item() for value in gradients.values())
                self.assertLess((numerator / max(denominator, 1e-30)) ** .5, 1e-3)
                print(f'R106 pair {model.arm}/{problem}: loss_abs={abs(shared.item()-separate.item()):.9g} '
                      f'gradient_relative_l2={(numerator / max(denominator, 1e-30)) ** .5:.9g}')

    def test_07_context_scope_wrong_batch_mutation_and_update_guards(self):
        model = self.b
        batch = raw_batch('TSP')
        with self.assertRaisesRegex(RuntimeError, 'input/state changed'):
            with model.native_pair_context(batch):
                model(dict(batch, node=batch['node'].clone()))
        self.assertIsNone(model._native_pair)
        with self.assertRaisesRegex(RuntimeError, 'input/state changed'):
            with model.native_pair_context(batch):
                batch['node'].add_(.1)
                model(batch)
        self.assertIsNone(model._native_pair)
        for field in ('node_mask', 'n'):
            batch = raw_batch('TSP')
            with self.assertRaisesRegex(RuntimeError, 'input/state changed'):
                with model.native_pair_context(batch):
                    # Even an unversioned .data edit of these witnesses fails.
                    batch[field].data.reshape(-1)[0] = 0
                    model(batch)
            self.assertIsNone(model._native_pair)
        batch = raw_batch('TSP')
        with self.assertRaisesRegex(RuntimeError, 'input/state changed'):
            with model.native_pair_context(batch):
                with torch.no_grad():
                    next(model.score_head.parameters()).add_(.1)
                model(batch)
        self.assertIsNone(model._native_pair)
        with model.native_pair_context(batch):
            with self.assertRaisesRegex(RuntimeError, 'Nested'):
                with model.native_pair_context(batch):
                    pass
            model(batch)
            model(dict(batch, costs=object(), ind=object()))
            with self.assertRaisesRegex(RuntimeError, 'two preparations'):
                model(batch)
        self.assertIsNone(model._native_pair)
        with patch.object(model.donor, 'forward', wraps=model.donor.forward) as forward:
            model(batch)
            model(batch)
            self.assertEqual(forward.call_count, 2)

    def test_08_physical_tsp_anchor_and_matnet_dual_roles_distance(self):
        batch = raw_batch('TSP')
        prepared = self.a.prepare_inputs(batch)
        role = prepared['node'][..., 3]
        self.assertTrue((role[:, 0] == 1).all())
        self.assertTrue((role.sum(1) == 1).all())
        perm = torch.arange(batch['node_mask'].shape[1] - 1, -1, -1)
        marked = prepare_role_inputs(models._sanitize(models.input_only(batch)), models.MODEL_ROLE_ARM)
        changed = dict(marked)
        for key in ('node', 'node_mask', 'node_geom'):
            changed[key] = marked[key][:, perm]
        with torch.no_grad():
            torch.testing.assert_close(self.a(changed)['logits'], self.a(marked)['logits'], atol=2e-5, rtol=2e-5)
            batch = raw_batch('ATSP')
            roles = matnet_roles(batch['node_mask'])
            self.assertEqual(roles[0].count_nonzero().item(), 0)
            for row, count in enumerate(batch['n'].tolist()):
                torch.testing.assert_close(roles[1][row, batch['node_mask'][row]], torch.eye(256)[:count])
            first = self.b.donor(models.input_only(batch))
            self.assertFalse(torch.allclose(first[..., :256], first[..., 256:]))
            transposed = self.b.donor(models.input_only(dict(batch, matrix=batch['matrix'].transpose(1, 2))))
            self.assertGreater((first - transposed).abs().max().item(), 1e-3)

    def test_09_zero_fusion_recovers_original_selector(self):
        with torch.random.fork_rng(devices=[]):
            original = original_model(self.initial['model_params'], models.MODEL_ROLE_ARM).eval()
        state = original.state_dict()
        for name in state:
            state[name] = self.initial_state[name]
        original.load_state_dict(state, strict=True)
        with torch.no_grad():
            for projection in self.a.instance_encoder.nco_projections.values():
                projection.weight.zero_()
            for problem in ('TSP', 'CVRP', 'ATSP'):
                batch = models._sanitize(models.input_only(raw_batch(problem)))
                torch.testing.assert_close(self.a(batch)['logits'], original(batch)['logits'], atol=2e-6, rtol=2e-6)

    def test_10_strict_state_only_restore_and_independent_schema(self):
        state = copy.deepcopy(self.b.state_dict())
        # A trained native tensor, not its original external value, must survive.
        name = 'donor.tsp.embedding.weight'
        state[name].add_(.125)
        with patch.object(native, '_load_encoder_weights', side_effect=AssertionError('no external weights')), \
                patch.object(torch, 'load', side_effect=AssertionError('no weight file I/O')):
            restored = models.make_model(self.params_b, 'B_finetune', state_dict=state)
        self.assertTrue(all(torch.equal(value, restored.state_dict()[key]) for key, value in state.items()))
        self.assertTrue(all(p.requires_grad and p._r106_native for p in restored.donor.parameters()))
        missing = dict(state)
        missing.pop(name)
        with self.assertRaisesRegex(ValueError, 'COMPLETE'):
            models.make_model(self.params_b, 'B_finetune', state_dict=missing)
        wrong_dtype = dict(state, **{name: state[name].double()})
        with self.assertRaisesRegex(ValueError, 'shape/dtype'):
            models.make_model(self.params_b, 'B_finetune', state_dict=wrong_dtype)
        with self.assertRaises(ValueError):
            models.make_model(dict(self.params_b, frozen_donor={}), 'B_finetune')
        with self.assertRaises(ValueError):
            models.initialize_shared(dict(self.initial, initialization='trained'), self.geometry, 'A_frozen')
        with tempfile.TemporaryDirectory(prefix='r106_cpu_') as directory:
            path = Path(directory) / 'model.pt'
            torch.save(dict(args=dict(arm='B_finetune', model_params=self.params_b), model=state), path)
            load = torch.load

            def only_checkpoint(filename, *args, **kwargs):
                self.assertEqual(Path(filename), path)
                return load(filename, *args, **kwargs)

            with patch.object(torch, 'load', side_effect=only_checkpoint) as reads:
                loaded, checkpoint = models.load_checkpoint(path)
            self.assertEqual(reads.call_count, 1)
            self.assertEqual(parameter_hash(loaded.state_dict()), parameter_hash(checkpoint['model']))


if __name__ == '__main__':
    unittest.main()

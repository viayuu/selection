"""Five focused CPU tests of R88 input roles; no GPU or real dataset reads."""

import copy
import unittest

import torch
import torch.nn.functional as F

from code.unified_selector.registry import PROBLEMS
from .local_geometry import local_geometry
from .pairwise_objective import LABEL_KEYS
from .r76_experiment import RDropObjective
from .r77_data import parameter_hash
from .r84_experiment import initialize_shared
from .r88_model import ARMS, ROLE_SLOT, ROLE_SPEC, make_model, prepare_role_inputs
from .test_r76 import fixture_batch, fixture_model


def tsp_batch(rows=2):
    batch, _ = fixture_batch('TSP', rows)
    # Real TSP inputs contain only xy, unlike the generic eight-channel fixture.
    batch['node'] = batch['node'][..., :2].clone()
    return batch


def relabel(batch, order):
    result = dict(batch)
    for key in ('node', 'node_mask', 'node_geom'):
        if key in result:
            result[key] = result[key][:, order]
    if 'matrix' in result:
        result['matrix'] = result['matrix'][:, order][:, :, order]
    return result


class R88ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        self.base, self.params = fixture_model(dropout=.1)
        self.models = [make_model(self.params, arm, state_dict=self.base.state_dict()) for arm in ARMS]

    def test_all18_original_backbone_and_exact_non_tsp_numeric_rng_parity(self):
        self.base.train()
        for problem in PROBLEMS:
            batch, _ = fixture_batch(problem, 2)
            if problem == 'TSP':
                batch['node'] = batch['node'][..., :2].clone()
            for model in self.models:
                with self.subTest(problem=problem, arm=model.role_arm):
                    prepared = model.prepare_role_inputs(batch)
                    if problem != 'TSP':
                        self.assertIs(prepared['node'], batch['node'])
                        self.assertIs(prepared['matrix'], batch['matrix'])
                    torch.manual_seed(226)
                    expected = self.base(prepared if problem == 'TSP' else batch)
                    rng = torch.get_rng_state().clone()
                    torch.manual_seed(226)
                    actual = model.train()(batch)
                    self.assertEqual(set(actual), set(expected))
                    for key, value in expected.items():
                        torch.testing.assert_close(actual[key], value, rtol=0, atol=0)
                    torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)

    def test_role_padding_tie_set_equivariance_and_stats_geometry_unchanged(self):
        batch = tsp_batch()
        xy = torch.tensor([[[.9, .1], [.1, .8], [.1, .2], [.4, .1], [.1, .2], [-9., -9.]],
                           [[-9., -9.], [-9., -9.], [.7, .4], [.3, .9], [.3, .2], [-9., -9.]]])
        mask = torch.tensor([[True, True, True, True, True, False],
                             [False, False, True, True, True, False]])
        batch.update(node=xy, matrix=torch.cdist(xy, xy), node_mask=mask,
                     n=mask.sum(-1), node_geom=local_geometry(xy, mask).float())
        raw = copy.deepcopy(batch)
        padded = self.base.instance_encoder._pad_node(xy)
        self.assertEqual(padded[..., ROLE_SLOT].count_nonzero().item(), 0)
        expected_stats = self.base.instance_encoder._coord_stats(padded, mask)
        expected_bias = self.base.instance_encoder._coord_bias(padded, mask, 3)
        for arm, indices, counts in zip(ARMS, ([2, 4], [0, 2]), ([2., 1.], [1., 1.])):
            with self.subTest(arm=arm):
                prepared = prepare_role_inputs(batch, arm)
                node = prepared['node']
                self.assertEqual(tuple(node.shape), (2, 6, 8))
                self.assertEqual(node[..., ROLE_SLOT].sum(-1).tolist(), counts)
                self.assertEqual(node[..., ROLE_SLOT].argmax(-1).tolist(), indices)
                self.assertEqual(node[..., ROLE_SLOT][~mask].count_nonzero().item(), 0)
                torch.testing.assert_close(node[..., :2], xy, rtol=0, atol=0)
                torch.testing.assert_close(self.base.instance_encoder._coord_stats(node, mask),
                                           expected_stats, rtol=0, atol=0)
                torch.testing.assert_close(self.base.instance_encoder._coord_bias(node, mask, 3),
                                           expected_bias, rtol=0, atol=0)
                torch.testing.assert_close(local_geometry(node[..., :2], mask).float(), batch['node_geom'], rtol=0, atol=0)
                for key in ('node_geom', 'cbits', 'problem_desc', 'coord_dist', 'n', 'node_mask'):
                    self.assertIs(prepared[key], batch[key])
                projection_inputs = []
                geometry = self.base.instance_encoder.geometry_residual
                hook = geometry.projection.register_forward_pre_hook(
                    lambda module, arguments: projection_inputs.append(arguments[0].detach().clone()))
                geometry(padded, mask, batch['n'], batch['node_geom'])
                geometry(node, mask, batch['n'], batch['node_geom'])
                hook.remove()
                expected_delta = torch.zeros_like(projection_inputs[0])
                expected_delta[..., ROLE_SLOT] = node[..., ROLE_SLOT]
                torch.testing.assert_close(projection_inputs[1] - projection_inputs[0], expected_delta, rtol=0, atol=0)
                if arm == ARMS[0]:
                    self.assertEqual(node[0, :, ROLE_SLOT].nonzero().flatten().tolist(), [2, 4])
                    order = torch.tensor([4, 5, 1, 3, 0, 2])
                    from_raw = prepare_role_inputs(relabel(batch, order), arm)
                    from_marked = prepare_role_inputs(relabel(prepared, order), arm)
                    torch.testing.assert_close(from_raw['node'], node[:, order], rtol=0, atol=0)
                    torch.testing.assert_close(from_marked['node'], node[:, order], rtol=0, atol=0)
                    # A partial tie set is not a valid explicit canonical marker.
                    partial = node.clone()
                    partial[0, 4, ROLE_SLOT] = 0.
                    with self.assertRaises(ValueError):
                        prepare_role_inputs(dict(batch, node=partial), arm)
        torch.testing.assert_close(batch['node'], raw['node'], rtol=0, atol=0)
        for arm in ARMS:
            explicit = prepare_role_inputs(batch, arm)['node']
            torch.testing.assert_close(prepare_role_inputs(dict(batch, node=explicit), arm)['node'], explicit, rtol=0, atol=0)
        explicit = F.pad(xy, (0, 6))
        explicit[:, 3, ROLE_SLOT] = 1.
        torch.testing.assert_close(prepare_role_inputs(dict(batch, node=explicit), ARMS[1])['node'], explicit, rtol=0, atol=0)
        with self.assertRaises(ValueError):
            prepare_role_inputs(dict(batch, node=explicit), ARMS[0])
        for value in (.5, -1., float('nan')):
            invalid = padded.clone()
            invalid[:, 0, ROLE_SLOT] = value
            with self.assertRaises(ValueError):
                prepare_role_inputs(dict(batch, node=invalid), ARMS[0])
        invalid = explicit.clone()
        invalid[:, 4, ROLE_SLOT] = 1.
        with self.assertRaises(ValueError):
            prepare_role_inputs(dict(batch, node=invalid), ARMS[1])
        invalid = padded.clone()
        invalid[:, 5, ROLE_SLOT] = 1.
        with self.assertRaises(ValueError):
            prepare_role_inputs(dict(batch, node=invalid), ARMS[1])

    def test_pure_relabel_preserves_physical_role_but_new_execution_start_differs(self):
        batch = tsp_batch(3)
        batch['node'] = torch.tensor([[[.9, .9], [.1, .6], [.4, .2], [.2, .3], [.7, .8]]]).expand(3, -1, -1).clone()
        order = torch.tensor([2, 4, 0, 3, 1])
        for model in self.models:
            with self.subTest(arm=model.role_arm):
                model.eval()
                prepared = model.prepare_role_inputs(batch)
                permuted = relabel(prepared, order)
                preserved = model.prepare_role_inputs(permuted)
                torch.testing.assert_close(preserved['node'], permuted['node'], rtol=0, atol=0)
                torch.testing.assert_close(model(permuted)['logits'], model(prepared)['logits'], rtol=3e-5, atol=3e-6)
                recomputed = model.prepare_role_inputs(relabel(batch, order))
                if model.role_arm == ARMS[0]:
                    torch.testing.assert_close(recomputed['node'], permuted['node'], rtol=0, atol=0)
                else:
                    self.assertEqual(permuted['node'][0, :, ROLE_SLOT].argmax().item(), 2)
                    self.assertEqual(recomputed['node'][0, :, ROLE_SLOT].argmax().item(), 0)
                    pure_xy = permuted['node'][0, permuted['node'][0, :, ROLE_SLOT].bool(), :2]
                    changed_xy = recomputed['node'][0, recomputed['node'][0, :, ROLE_SLOT].bool(), :2]
                    self.assertFalse(torch.equal(pure_xy, changed_xy))
                    self.assertFalse(torch.allclose(model(recomputed)['logits'], model(permuted)['logits'], rtol=1e-8, atol=1e-9))

    def test_label_isolation_and_role_attribute_gradient_reaches_original_encoder(self):
        batch = tsp_batch(4)
        for model in self.models:
            with self.subTest(arm=model.role_arm):
                model.eval()
                expected = model(batch)
                changed = copy.deepcopy(batch)
                for key in ('costs', 'cost_regret', 'cmp_target'):
                    changed[key].fill_(float('nan'))
                changed['ind'].fill_(-123)
                changed['cmp_valid'].fill_(False)
                self.assertFalse(set(LABEL_KEYS) & set(model.prepare_role_inputs(changed)))
                for key, value in expected.items():
                    torch.testing.assert_close(model(changed)[key], value, rtol=0, atol=0)
                model.train()
                model.zero_grad(set_to_none=True)
                loss, parts = RDropObjective(model)(batch, .35)
                self.assertTrue(torch.isfinite(loss))
                self.assertTrue(torch.isfinite(parts).all())
                loss.backward()
                gradient = model.instance_encoder.coord_proj.weight.grad
                self.assertTrue(torch.isfinite(gradient).all())
                self.assertGreater(gradient[:, ROLE_SLOT].abs().sum().item(), 0.)
                for module in (*model.joint_layers, model.pool, model.decoder, model.score_head):
                    self.assertGreater(sum(p.grad.abs().sum().item() for p in module.parameters()
                                           if p.grad is not None), 0.)

    def test_r84_fresh_factory_full_parameter_identity_and_strict_state_roundtrip(self):
        reference = {name: value.detach().clone() for name, value in self.base.named_parameters()}
        initial = dict(seed=2, initialization='scratch', trainable=reference,
                       trainable_sha256=parameter_hash(reference), model_params=self.params)
        geometry = dict(mean=[.2] * 12, std=[1.3] * 12)
        batch = tsp_batch()
        for arm in ARMS:
            model, params, receipt = initialize_shared(initial, geometry, arm, model_factory=make_model)
            self.assertEqual(params['role_arm'], arm)
            self.assertEqual(params['role_spec'], ROLE_SPEC)
            self.assertEqual(set(model.state_dict()), set(self.base.state_dict()))
            self.assertEqual(set(dict(model.named_parameters())), set(reference))
            self.assertEqual(parameter_hash(dict(model.named_parameters())), initial['trainable_sha256'])
            self.assertEqual(receipt['new_parameter_names'], [])
            self.assertEqual(receipt['removed_parameter_names'], [])
            torch.testing.assert_close(model.instance_encoder.geometry_residual.mean, torch.tensor(geometry['mean']))
            torch.testing.assert_close(model.instance_encoder.geometry_residual.std, torch.tensor(geometry['std']))
            model.eval()
            restored = make_model(params, arm, state_dict=model.state_dict()).eval()
            for key, value in model(batch).items():
                torch.testing.assert_close(restored(batch)[key], value, rtol=0, atol=0)
            incomplete = dict(model.state_dict())
            incomplete.pop('score_head.3.weight')
            with self.assertRaises(RuntimeError):
                make_model(params, arm, state_dict=incomplete)
        with self.assertRaises(ValueError):
            make_model(self.models[0].params, ARMS[1])
        with self.assertRaises(ValueError):
            make_model(dict(self.params, role_spec=dict(ROLE_SPEC, role_slot=7)), ARMS[0])


if __name__ == '__main__':
    unittest.main()

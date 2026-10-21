"""Eight focused R89 CPU tests, including real GradScaler and mixed steps."""

import copy
import unittest
from unittest.mock import patch

import torch
from torch import nn

from . import r89_optim as optim
from .r88_model import make_model
from .test_r76 import fixture_model


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = nn.Linear(4, 3)
        self.input = nn.Linear(4, 4)
        self.score_head = nn.Sequential(nn.Linear(3, 3), nn.GELU(), nn.Identity(), nn.Linear(3, 1))

    def forward(self, x):
        return self.score_head(self.decoder(self.input(x)))


def assign_gradients(model, seed):
    generator = torch.Generator().manual_seed(seed)
    for parameter in model.parameters():
        parameter.grad = torch.randn(parameter.shape, generator=generator)


class R89OptimizerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(2)
        self.model = TinyModel()

    def assert_tree_equal(self, first, second):
        if torch.is_tensor(first):
            torch.testing.assert_close(first, second, rtol=0, atol=0)
        elif isinstance(first, dict):
            self.assertEqual(set(first), set(second))
            for key in first:
                self.assert_tree_equal(first[key], second[key])
        elif isinstance(first, (tuple, list)):
            self.assertEqual(len(first), len(second))
            for left, right in zip(first, second):
                self.assert_tree_equal(left, right)
        else:
            self.assertEqual(first, second)

    def test_hidden_linear_whitelist_exhaustive_exclusion_and_manifest(self):
        base, params = fixture_model(dropout=0.)
        model = make_model(params, 'B_execution_anchor', state_dict=base.state_dict())
        model.pool.add_module('thin_probe', nn.Linear(1, 3, bias=False))
        model.pool.add_module('embedding_probe', nn.Embedding(4, 3))
        optimizer = optim.build_optimizer(model, 'B_muon_hybrid')
        manifest = optim.optimizer_manifest(optimizer)
        all_names = [entry['name'] for group in manifest['groups'] for entry in group['parameters']]
        self.assertEqual(set(all_names), set(dict(model.named_parameters())))
        self.assertEqual(len(all_names), len(set(all_names)))
        hidden = set(optimizer.param_groups[0]['param_names'])
        self.assertIn('decoder.Wq.weight', hidden)
        self.assertIn('score_head.0.weight', hidden)
        self.assertIn('pool.0.weight', hidden)
        for name in dict(model.named_parameters()):
            if (name.startswith(('solver_encoder.', 'instance_encoder.coord_proj.', 'instance_encoder.matrix_proj.',
                                 'instance_encoder.condition_encoder.', 'instance_encoder.geometry_residual.'))
                    or name.endswith('.bias') or 'norm' in name or name.startswith('score_head.3.')
                    or name.startswith(('pool.thin_probe.', 'pool.embedding_probe.'))):
                self.assertNotIn(name, hidden)
        self.assertEqual(manifest['total_parameters'], sum(p.numel() for p in model.parameters()))
        self.assertIs(type(optimizer._adamw), torch.optim.AdamW)
        self.assertIs(optimizer._adamw.state, optimizer.state)
        self.assertIs(optimizer._adamw.param_groups[0], optimizer.param_groups[1])

    def test_control_is_true_adamw_original_order_and_bitwise_steps(self):
        other = copy.deepcopy(self.model)
        control = optim.build_optimizer(self.model, 'A_adamw')
        reference = torch.optim.AdamW(other.parameters(), lr=1e-4, weight_decay=1e-4)
        self.assertIs(type(control), torch.optim.AdamW)
        self.assertEqual([id(p) for p in control.param_groups[0]['params']], [id(p) for p in self.model.parameters()])
        for index in range(3):
            assign_gradients(self.model, index)
            assign_gradients(other, index)
            control.step()
            reference.step()
        self.assert_tree_equal(self.model.state_dict(), other.state_dict())
        self.assert_tree_equal(control.state_dict(), reference.state_dict())

    def test_ns5_bf16_zero_tall_wide_finite_and_only_moonshot_scale(self):
        for rows, cols in ((2, 2), (3, 29), (29, 3)):
            for zero in (False, True):
                with self.subTest(shape=(rows, cols), zero=zero):
                    grad = torch.zeros(rows, cols) if zero else torch.randn(rows, cols)
                    result = optim.zeropower_via_newtonschulz5(grad)
                    self.assertEqual(result.dtype, torch.bfloat16)
                    self.assertEqual(result.shape, grad.shape)
                    self.assertTrue(torch.isfinite(result).all())
                    if zero:
                        self.assertEqual(result.count_nonzero().item(), 0)
                    momentum = torch.zeros_like(grad)
                    before = grad.clone()
                    update = optim.muon_update(grad, momentum)
                    expected_momentum = grad * (1 - .95)
                    torch.testing.assert_close(momentum, expected_momentum)
                    mixed = grad.lerp(momentum, .95)
                    expected = optim.zeropower_via_newtonschulz5(mixed).float() * (.2 * max(rows, cols) ** .5)
                    torch.testing.assert_close(update, expected, rtol=0, atol=0)
                    torch.testing.assert_close(grad, before, rtol=0, atol=0)
                    self.assertTrue(torch.isfinite(update).all())
        with torch.autocast(device_type='cpu', dtype=torch.bfloat16):
            self.assertEqual(optim.zeropower_via_newtonschulz5(torch.randn(3, 7)).dtype, torch.bfloat16)

    def test_real_mixed_step_aux_adamw_and_strict_state_roundtrip_continue(self):
        optimizer = optim.build_optimizer(self.model, 'B_muon_hybrid')
        reference = copy.deepcopy(self.model)
        aux_names = set(optimizer.param_groups[1]['param_names'])
        aux = torch.optim.AdamW([p for name, p in reference.named_parameters() if name in aux_names],
                                lr=1e-4, weight_decay=1e-4)
        for step in range(2):
            assign_gradients(self.model, 41 + step)
            assign_gradients(reference, 41 + step)
            optimizer.step()
            aux.step()
        for name, parameter in self.model.named_parameters():
            if name in aux_names:
                torch.testing.assert_close(parameter, dict(reference.named_parameters())[name], rtol=0, atol=0)
        saved_model, saved = copy.deepcopy(self.model.state_dict()), copy.deepcopy(optimizer.state_dict())
        resumed_model = TinyModel()
        resumed_model.load_state_dict(saved_model)
        resumed = optim.build_optimizer(resumed_model, 'B_muon_hybrid')
        resumed.load_state_dict(saved)
        self.assertIs(resumed._adamw.state, resumed.state)
        self.assertIs(resumed._adamw.param_groups[0], resumed.param_groups[1])
        self.assert_tree_equal(saved, resumed.state_dict())
        for instance, owner in ((self.model, optimizer), (resumed_model, resumed)):
            assign_gradients(instance, 100)
            for group in owner.param_groups:
                group['lr'] = 2e-5
            owner.step()
        self.assert_tree_equal(self.model.state_dict(), resumed_model.state_dict())
        self.assert_tree_equal(optimizer.state_dict(), resumed.state_dict())
        self.assertEqual(optimizer.completed_steps, 3)
        metrics = optim.optimizer_metrics(optimizer)
        self.assertEqual(metrics['muon_step_applied'].item(), 1.)
        self.assertGreater(metrics['muon_direction_rms'].item(), 0.)
        for parameter in optimizer.param_groups[0]['params']:
            self.assertEqual(optimizer.state[parameter]['step'], 3)
            self.assertEqual(optimizer.state[parameter]['ns_calls'], 3)
            self.assertEqual(optimizer.state[parameter]['momentum_buffer'].dtype, torch.float32)
        bad = copy.deepcopy(saved)
        bad['param_groups'][0]['ns_steps'] = 4
        with self.assertRaises(ValueError):
            resumed.load_state_dict(bad)

    def test_logical_batch_micro_accumulation_orthogonalizes_only_once_after_clip(self):
        other = copy.deepcopy(self.model)
        optimizer = optim.build_optimizer(self.model, 'B_muon_hybrid')
        full = optim.build_optimizer(other, 'B_muon_hybrid')
        x, y = torch.randn(32, 4), torch.randn(32, 1)
        with patch.object(optim, 'zeropower_via_newtonschulz5', wraps=optim.zeropower_via_newtonschulz5) as ns:
            for start in range(0, 32, 7):
                stop = min(start + 7, 32)
                loss = (self.model(x[start:stop]) - y[start:stop]).square().mean()
                (loss * ((stop - start) / 32)).backward()
            self.assertEqual(ns.call_count, 0)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.)
            optimizer.step()
            self.assertEqual(ns.call_count, len(optimizer.param_groups[0]['params']))
        (other(x) - y).square().mean().backward()
        torch.nn.utils.clip_grad_norm_(other.parameters(), 1.)
        full.step()
        for parameter, reference in zip(self.model.parameters(), other.parameters()):
            torch.testing.assert_close(parameter, reference, rtol=1e-5, atol=2e-6)
        self.assertEqual(optimizer.completed_steps, 1)
        for parameter in optimizer.param_groups[0]['params']:
            self.assertEqual(optimizer.state[parameter]['ns_calls'], 1)

    def test_real_cpu_gradscaler_overflow_skips_entire_state_and_parameters(self):
        optimizer = optim.build_optimizer(self.model, 'B_muon_hybrid')
        scaler = torch.amp.GradScaler('cpu', init_scale=8.)
        x = torch.randn(5, 4)
        scaler.scale(self.model(x).square().mean()).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.)
        scaler.step(optimizer)
        scaler.update()
        state, parameters = copy.deepcopy(optimizer.state_dict()), copy.deepcopy(self.model.state_dict())
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(self.model(x).square().mean()).backward()
        optimizer.param_groups[1]['params'][0].grad.flatten()[0] = float('inf')
        scaler.unscale_(optimizer)
        with patch.object(optimizer, 'step', wraps=optimizer.step) as step:
            scaler.step(optimizer)
            self.assertEqual(step.call_count, 0)
        scaler.update()
        self.assert_tree_equal(optimizer.state_dict(), state)
        self.assert_tree_equal(self.model.state_dict(), parameters)

    def test_nonfinite_direction_rejects_without_momentum_adam_or_parameter_changes(self):
        optimizer = optim.build_optimizer(self.model, 'B_muon_hybrid')
        assign_gradients(self.model, 23)
        parameters, state = copy.deepcopy(self.model.state_dict()), copy.deepcopy(optimizer.state_dict())
        with patch.object(optim, 'zeropower_via_newtonschulz5', side_effect=lambda value, steps: torch.full_like(value, float('nan'))):
            with self.assertRaises(FloatingPointError):
                optimizer.step()
        self.assertIsNotNone(optimizer.last_failure)
        self.assert_tree_equal(optimizer.state_dict(), state)
        self.assert_tree_equal(self.model.state_dict(), parameters)
        self.assertEqual(optimizer.completed_steps, 0)
        self.assertEqual(len(optimizer.state), 0)

    def test_no_global_torch_mutation_missing_grads_skip_and_fixed_provenance(self):
        adam_class, optimizer_class = torch.optim.AdamW, torch.optim.Optimizer
        optimizer = optim.build_optimizer(self.model, 'B_muon_hybrid')
        optimizer.zero_grad(set_to_none=True)
        before = copy.deepcopy(self.model.state_dict())
        optimizer.step()
        self.assert_tree_equal(self.model.state_dict(), before)
        self.assertEqual(len(optimizer.state), 0)
        self.assertEqual(optim.optimizer_metrics(optimizer)['muon_step_applied'].item(), 0.)
        self.assertIs(torch.optim.AdamW, adam_class)
        self.assertIs(torch.optim.Optimizer, optimizer_class)
        manifest = optim.optimizer_manifest(optimizer)
        self.assertEqual(manifest['provenance']['commit'], optim.SOURCE_COMMIT)
        self.assertEqual(manifest['provenance']['update_scaling'], '0.2*sqrt(max(rows,cols))')
        self.assertFalse(manifest['provenance']['aspect_ratio_scale_applied'])
        self.assertTrue(all(group['lr'] == 1e-4 and group['weight_decay'] == 1e-4 for group in optimizer.param_groups))


if __name__ == '__main__':
    unittest.main()

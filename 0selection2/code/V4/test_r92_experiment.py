"""Synthetic two-thread CPU witnesses for R92 initialization/checkpoint wiring."""

import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from code.unified_selector.registry import GLOBAL_SOLVERS
from . import r92_experiment as runner
from .r76_experiment import save_checkpoint
from .r77_data import parameter_hash
from .r92_model import ARCHITECTURE, make_model
from .test_r92_model import graph_batch, model_params


def fresh_fixture():
    """The R84 fresh baseline factory, never a trained or on-disk checkpoint."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(2)
        params = dict(model_params(), pair_mode='score_difference',
                      solver_representation='handcrafted')
        model = runner.engine.make_model(params, 'A_baseline')
    values = {name: value.detach().clone() for name, value in model.named_parameters()}
    initial = dict(seed=2, initialization='scratch', model_params=copy.deepcopy(model.params),
                   trainable=values, trainable_sha256=parameter_hash(values))
    width = model.instance_encoder.geometry_residual.mean.numel()
    return initial, dict(mean=[.2] * width, std=[1.3] * width)


def checkpoint_state(model, params, receipt, arm):
    config = dict(experiment='R92', r92_schema_version=1, arm=arm, seed=2,
                  max_successful_updates=runner.BUDGETS[arm], solver_names=list(GLOBAL_SOLVERS),
                  model_factory=runner.MODEL_FACTORY, model_params=copy.deepcopy(params),
                  objective=runner.objective_spec(arm), hierarchy=copy.deepcopy(ARCHITECTURE))
    return dict(args=config, state_hash=runner.json_hash(config), model=model.state_dict(),
                initialization=copy.deepcopy(receipt))


class R92ExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        guard = patch('torch.cuda._lazy_init', side_effect=AssertionError('CPU tests must not initialize CUDA'))
        guard.start()
        self.addCleanup(guard.stop)

    def test_retained_fresh_r84_subset_and_common_new_parameters_match_exactly(self):
        initial, geometry = fresh_fixture()
        source_hash = parameter_hash(initial['trainable'])
        models, receipts = {}, {}
        for arm in ('A', 'B'):
            torch.manual_seed(103 if arm == 'A' else 907)
            before_rng = torch.get_rng_state().clone()
            model, params, receipt = runner.initialize_shared(initial, geometry, arm)
            self.assertTrue(torch.equal(torch.get_rng_state(), before_rng))
            models[arm], receipts[arm] = model, receipt
            actual = dict(model.named_parameters())
            retained = {name: value for name, value in actual.items()
                        if name in initial['trainable'] and value.shape == initial['trainable'][name].shape}
            self.assertEqual(receipt['common_parameter_names'], sorted(retained))
            self.assertEqual(receipt['common_parameter_sha256'], parameter_hash(retained))
            self.assertEqual(receipt['trainable_sha256'], parameter_hash(actual))
            self.assertEqual(receipt['new_parameter_names'], sorted(set(actual) - set(retained)))
            self.assertEqual(receipt['removed_parameter_names'], sorted(set(initial['trainable']) - set(retained)))
            self.assertFalse(receipt['historical_checkpoint_loaded'])
            self.assertEqual(receipt['new_common_seed'], 920002)
            self.assertEqual(params, model.params)
            self.assertEqual(params['r92_architecture'], ARCHITECTURE)
            self.assertEqual(params['r92_arm'], arm)
            for prefix in ('instance_encoder.coord_proj.', 'solver_encoder.solver_emb.',
                           'solver_encoder.feature_mlp.', 'joint_layers.', 'score_head.'):
                self.assertTrue(any(name.startswith(prefix) for name in retained))
            for name, value in retained.items():
                torch.testing.assert_close(value, initial['trainable'][name], rtol=0, atol=0)
            branch = model.instance_encoder.geometry_residual
            torch.testing.assert_close(branch.mean, torch.tensor(geometry['mean']), rtol=0, atol=0)
            torch.testing.assert_close(branch.std, torch.tensor(geometry['std']), rtol=0, atol=0)
        a, b = (dict(models[arm].named_parameters()) for arm in ('A', 'B'))
        self.assertTrue(set(a) < set(b))
        self.assertTrue(all(name.startswith('instance_encoder.assignment_nets.') for name in set(b) - set(a)))
        for name in a:
            torch.testing.assert_close(a[name], b[name], rtol=0, atol=0)
        self.assertEqual(receipts['A']['new_shared_parameter_sha256'], receipts['B']['new_shared_parameter_sha256'])
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(920002)
            template = make_model(initial['model_params'], 'B')
        for name, value in b.items():
            if name not in receipts['B']['common_parameter_names']:
                torch.testing.assert_close(value, dict(template.named_parameters())[name], rtol=0, atol=0)
        self.assertEqual(parameter_hash(initial['trainable']), source_hash)

    def test_nonfresh_or_corrupted_initialization_rejected_before_construction(self):
        initial, geometry = fresh_fixture()
        invalid = [dict(initial, seed=1), dict(initial, initialization='trained_checkpoint'),
                   dict(initial, trainable_sha256='0' * 64)]
        changed = copy.deepcopy(initial)
        next(iter(changed['trainable'].values())).add_(1.)
        invalid.append(changed)
        with patch.object(runner, 'make_model', side_effect=AssertionError('invalid source reached construction')):
            for source in invalid:
                with self.assertRaises(ValueError):
                    runner.initialize_shared(source, geometry, 'A')
            for arm in ('C', 'D', 'unknown'):
                with self.assertRaises(ValueError):
                    runner.initialize_shared(initial, geometry, arm)

    def test_checkpoint_roundtrip_preserves_params_buffers_and_coordinate_matrix_predictions(self):
        initial, geometry = fresh_fixture()
        for arm in ('A', 'B'):
            with self.subTest(arm=arm):
                model, params, receipt = runner.initialize_shared(initial, geometry, arm)
                # Non-initial values ensure loading really restores the checkpoint.
                with torch.no_grad():
                    model.score_head[-1].bias.add_(.125)
                state = checkpoint_state(model, params, receipt, arm)
                with tempfile.TemporaryDirectory() as temporary:
                    path = Path(temporary) / 'synthetic.pt'
                    save_checkpoint(path, state)
                    restored, loaded = runner.load_checkpoint(path, device='cpu')
                self.assertEqual(loaded['args']['model_params'], params)
                self.assertEqual(restored.params, params)
                self.assertEqual(loaded['state_hash'], runner.json_hash(loaded['args']))
                self.assertEqual(loaded['initialization'], receipt)
                for name, value in model.state_dict().items():
                    torch.testing.assert_close(value, restored.state_dict()[name], rtol=0, atol=0)
                model.eval()
                restored.eval()
                for problem in ('TSP', 'ATSP'):
                    batch = graph_batch(problem, rows=2, nodes=9)
                    with torch.no_grad():
                        original_output, restored_output = model(batch), restored(batch)
                    for name in ('logits', 'utility', 'pair_margin', 'solver_mask', 'auxiliary_loss'):
                        torch.testing.assert_close(original_output[name], restored_output[name], rtol=0, atol=0)
                self.assertTrue(all(value.device.type == 'cpu' for value in restored.parameters()))

    def test_checkpoint_rejects_bad_hash_changed_objective_and_incomplete_state(self):
        initial, geometry = fresh_fixture()
        model, params, receipt = runner.initialize_shared(initial, geometry, 'B')
        state = checkpoint_state(model, params, receipt, 'B')
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'synthetic.pt'
            invalid_hash = dict(state, state_hash='0' * 64)
            save_checkpoint(path, invalid_hash)
            with self.assertRaisesRegex(ValueError, 'hash'):
                runner.load_checkpoint(path)
            changed = copy.deepcopy(state)
            changed['args']['objective']['hierarchy_auxiliary']['link'] = .2
            changed['state_hash'] = runner.json_hash(changed['args'])
            save_checkpoint(path, changed)
            with self.assertRaises(ValueError):
                runner.load_checkpoint(path)
            incomplete = dict(state, model=dict(state['model']))
            del incomplete['model']['score_head.0.weight']
            save_checkpoint(path, incomplete)
            with self.assertRaises(RuntimeError):
                runner.load_checkpoint(path)


if __name__ == '__main__':
    unittest.main()

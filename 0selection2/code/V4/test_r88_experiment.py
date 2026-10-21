"""Focused CPU runner/model checks for R88; no dataset scans or GPU launch."""

import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn.functional as F

from code.unified_selector.registry import GLOBAL_SOLVERS, PROBLEMS
from . import r88_experiment as runner
from . import r88_reporting as reporter
from .r76_experiment import RDropObjective, Runtime, save_checkpoint, write_json
from .r77_data import parameter_hash
from .test_r76 import CountingSGD, fixture_batch
from .test_r80 import fresh_fixture


def native_tsp_batch(size):
    batch, costs = fixture_batch('TSP', size)
    node = batch['node'].clone()
    node[..., 2:] = 0.
    return dict(batch, node=node), costs


class FailOnce(RDropObjective):
    def __init__(self, model):
        super().__init__(model)
        self.failed = False

    def forward(self, batch, weight):
        loss, parts = super().forward(batch, weight)
        if not self.failed:
            self.failed = True
            loss = loss * loss.new_tensor(float('nan'))
        return loss, parts


class R88RunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_cli_schedule_and_reporting_keep_full18_without_loading_data(self):
        with patch.object(torch, 'load', side_effect=AssertionError('CLI must not open data')):
            for arm in runner.ARMS:
                with redirect_stdout(io.StringIO()):
                    result = runner.main(['describe', '--arm', arm])
                self.assertEqual((result['experiment'], result['role_slot'], result['new_parameter_count']), ('R88', 3, 0))
                self.assertEqual((result['batch_size'], result['max_updates'], result['warmup_updates'],
                    result['eval_every']), (1024, 57600, 4320, 1440))
                for mode in ('train', 'test'):
                    args = runner.parser().parse_args([mode, '--arm', arm, '--wandb-mode', 'offline'])
                    self.assertEqual((args.experiment, args.gpu_type, args.large_micro_cap, args.eval_batch_size),
                        ('R88', '3090', 352, 256))
                for mode in ('lib', 'report'):
                    self.assertEqual(reporter.parser().parse_args([mode, '--arm', arm]).command, mode)
        self.assertEqual(set(runner.task_order(57600)), set(PROBLEMS))
        self.assertEqual(runner.learning_rate(4320), 1e-4)
        self.assertEqual(runner.learning_rate(57600), 1e-6)
        self.assertEqual(reporter.FACTORY, 'code.V4.r88_experiment')

    def test_fresh_initialization_retains_every_parameter_and_no_added_state(self):
        initial, geometry = fresh_fixture()
        hashes = []
        for arm in runner.ARMS:
            model, params, receipt = runner.initialize_shared(initial, geometry, arm)
            hashes.append(parameter_hash(model.state_dict()))
            self.assertEqual((receipt['new_parameter_count'], receipt['new_parameter_names'],
                receipt['removed_parameter_names']), (0, [], []))
            self.assertFalse(receipt['historical_checkpoint_loaded'])
            self.assertEqual(params, model.params)
            self.assertEqual(params['role_arm'], arm)
            self.assertIn('role_spec', params)
            self.assertFalse(any(name.startswith('residual_') for name in params))
            self.assertEqual(receipt['common_parameter_sha256'], initial['trainable_sha256'])
            for name, expected in initial['trainable'].items():
                torch.testing.assert_close(dict(model.named_parameters())[name], expected, rtol=0, atol=0)
            torch.testing.assert_close(model.instance_encoder.geometry_residual.mean,
                torch.as_tensor(geometry['mean']), rtol=0, atol=0)
        self.assertEqual(hashes[0], hashes[1])

    def test_remaining17_problem_outputs_and_dropout_rng_equal_the_original(self):
        initial, geometry = fresh_fixture()
        original, _, _ = runner.engine.initialize_shared(initial, geometry, 'A_baseline')
        models = [runner.initialize_shared(initial, geometry, arm)[0] for arm in runner.ARMS]
        for problem in PROBLEMS:
            if problem == 'TSP':
                continue
            batch, _ = fixture_batch(problem, 2)
            torch.manual_seed(330)
            expected = original(runner.engine.inputs_only(batch))
            rng = torch.get_rng_state()
            for model in models:
                torch.manual_seed(330)
                actual = model(runner.engine.inputs_only(batch))
                for key in expected:
                    torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
                torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)

    def test_tsp_slot3_encodes_the_two_anchors_without_changing_coord_statistics(self):
        initial, geometry = fresh_fixture()
        batch, _ = native_tsp_batch(3)
        node = F.pad(batch['node'], (0, 8 - batch['node'].shape[-1])).clone()
        node[:, :, :2] = .9
        node[:, 0, :2] = node.new_tensor([.8, .8])
        node[:, 1, :2] = node.new_tensor([.1, .4])
        node[:, 2, :2] = node.new_tensor([.1, .2])
        node[:, :, 3] = 0.
        batch = dict(batch, node=node)
        mask = torch.arange(node.shape[1])[None] < batch['n'][:, None]
        for arm, anchor in zip(runner.ARMS, (2, 0)):
            model, _, _ = runner.initialize_shared(initial, geometry, arm)
            model.eval()
            observed = []
            handle = model.instance_encoder.register_forward_pre_hook(
                lambda module, inputs: observed.append(inputs[0]['node'].detach().clone()))
            try:
                with torch.no_grad():
                    model(runner.engine.inputs_only(batch))
            finally:
                handle.remove()
            self.assertEqual(len(observed), 1)
            marked = observed[0]
            self.assertTrue(torch.equal(marked[:, :, 3].sum(-1), torch.ones(len(node))))
            self.assertTrue((marked[:, :, 3].argmax(-1) == anchor).all())
            keep = [i for i in range(8) if i != 3]
            torch.testing.assert_close(marked[:, :, keep], node[:, :, keep], rtol=0, atol=0)
            torch.testing.assert_close(model.instance_encoder._coord_stats(marked, mask),
                model.instance_encoder._coord_stats(node, mask), rtol=0, atol=0)
        torch.testing.assert_close(batch['node'], node, rtol=0, atol=0)

    def test_real_updates_and_amp_retry_keep_original_stateless_loss(self):
        initial, geometry = fresh_fixture()
        batch, _ = native_tsp_batch(5)
        for arm in runner.ARMS:
            outcomes = []
            for fail in (False, True):
                model, _, _ = runner.initialize_shared(initial, geometry, arm)
                objective = FailOnce(model) if fail else runner.make_objective(model, arm)
                if not fail:
                    self.assertIs(type(objective), RDropObjective)
                    self.assertEqual(runner.checkpoint_extras(objective), {})
                    runner.restore_objective(objective, {})
                optimizer = CountingSGD(model.parameters())
                before = parameter_hash(model.state_dict())
                torch.manual_seed(88)
                parts, retries, extras = runner.update_batch(objective, optimizer,
                    torch.amp.GradScaler('cpu', enabled=False), batch, .1,
                    Runtime(0, 1, torch.device('cpu')), cap=2, problem='TSP')
                self.assertEqual((optimizer.steps, retries, extras), (1, int(fail), {}))
                self.assertTrue(torch.isfinite(parts).all())
                self.assertNotEqual(before, parameter_hash(model.state_dict()))
                outcomes.append((parameter_hash(model.state_dict()), parts, torch.get_rng_state()))
            self.assertEqual(outcomes[0][0], outcomes[1][0])
            torch.testing.assert_close(outcomes[0][1], outcomes[1][1], rtol=0, atol=0)
            torch.testing.assert_close(outcomes[0][2], outcomes[1][2], rtol=0, atol=0)

    def test_checkpoint_strict_reconstruction_has_no_objective_or_training_label_dependency(self):
        initial, geometry = fresh_fixture()
        batch, _ = native_tsp_batch(2)
        for arm in runner.ARMS:
            model, params, _ = runner.initialize_shared(initial, geometry, arm)
            config = dict(experiment='R88', r88_schema_version=1, arm=arm, seed=2,
                initialization='scratch_shared_r77', model_params=params,
                solver_names=GLOBAL_SOLVERS, max_successful_updates=57600)
            state = dict(args=config, state_hash=runner.json_hash(config), model=model.state_dict())
            with tempfile.TemporaryDirectory() as temporary:
                path = Path(temporary) / 'model.pt'
                save_checkpoint(path, state)
                with patch.object(runner, 'make_objective', side_effect=AssertionError('no training labels')):
                    restored, _ = runner.load_checkpoint(path)
                model.eval()
                restored.eval()
                with torch.no_grad():
                    torch.testing.assert_close(restored(runner.engine.inputs_only(batch))['logits'],
                        model(runner.engine.inputs_only(batch))['logits'], rtol=0, atol=0)
                save_checkpoint(path, dict(state, objective_state={}))
                with self.assertRaises(ValueError):
                    runner.load_checkpoint(path)

    def test_main_lock_is_required_and_wrong_input_slot_is_rejected(self):
        args = runner.parser().parse_args(['train', '--arm', runner.ARMS[0]])
        with tempfile.TemporaryDirectory() as temporary:
            args.root = Path(temporary)
            with patch.object(runner.engine, 'train') as train:
                with self.assertRaises(FileNotFoundError):
                    runner.train(args, torch.device('cpu'))
                train.assert_not_called()
            path = args.root / 'protocol_lock.json'
            record = dict(experiment='R88', locked=True, arms=list(runner.ARMS), seed=2,
                batch_size=1024, max_updates=57600, warmup_updates=4320, eval_every=1440, role_slot=3)
            write_json(path, record)
            self.assertTrue(runner.protocol_lock(args)['record']['locked'])
            write_json(path, dict(record, role_slot=7))
            with self.assertRaises(ValueError):
                runner.protocol_lock(args)


if __name__ == '__main__':
    unittest.main()

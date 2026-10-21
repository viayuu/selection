"""Focused R87 runner/model integration checks; CPU fixtures, no real jobs."""

import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import torch

from code.unified_selector.registry import GLOBAL_SOLVERS, PROBLEMS
from . import r87_experiment as runner
from .r76_experiment import RDropObjective, Runtime, save_checkpoint, write_json
from .r77_data import parameter_hash
from .test_r76 import CountingSGD, fixture_batch
from .test_r80 import fresh_fixture


class FailOnce(RDropObjective):
    def __init__(self, selector):
        super().__init__(selector)
        self.failed = False

    def forward(self, batch, weight):
        loss, parts = super().forward(batch, weight)
        if not self.failed:
            self.failed = True
            loss = loss * loss.new_tensor(float('nan'))
        return loss, parts


class R87RunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_exact_cli_and_schedule_without_reading_weights(self):
        with patch.object(torch, 'load', side_effect=AssertionError('describe must not load data')):
            for arm in runner.ARMS:
                with redirect_stdout(io.StringIO()):
                    result = runner.main(['describe', '--arm', arm])
                self.assertEqual(result['experiment'], 'R87')
                self.assertEqual((result['batch_size'], result['max_updates'], result['warmup_updates'],
                    result['eval_every']), (1024, 57600, 4320, 1440))
                for mode in ('train', 'test'):
                    args = runner.parser().parse_args([mode, '--arm', arm, '--wandb-mode', 'offline'])
                    self.assertEqual((args.experiment, args.gpu_type, args.large_micro_cap,
                        args.eval_batch_size), ('R87', '4090', 352, 256))
        self.assertEqual(set(runner.task_order(57600)), set(PROBLEMS))
        self.assertEqual(runner.learning_rate(4320), 1e-4)
        self.assertEqual(runner.learning_rate(57600), 1e-6)

    def test_initializer_loads_all_common_and_only_adds_matched_zero_output_residual(self):
        initial, geometry = fresh_fixture()
        models, receipts = [], []
        for arm in runner.ARMS:
            model, params, receipt = runner.initialize_shared(initial, geometry, arm)
            models.append(model)
            receipts.append(receipt)
            self.assertEqual(params, model.params)
            self.assertEqual(receipt['common_parameter_sha256'], initial['trainable_sha256'])
            self.assertEqual(receipt['new_parameter_names'], ['residual.W', 'residual.v'])
            self.assertFalse(receipt['historical_checkpoint_loaded'])
            for name, expected in initial['trainable'].items():
                torch.testing.assert_close(dict(model.named_parameters())[name], expected, rtol=0, atol=0)
            self.assertEqual(model.residual.v.count_nonzero().item(), 0)
            torch.testing.assert_close(model.instance_encoder.geometry_residual.mean,
                torch.as_tensor(geometry['mean']), rtol=0, atol=0)
        self.assertEqual(receipts[0]['new_parameter_count'], receipts[1]['new_parameter_count'])
        self.assertEqual(receipts[0]['trainable_parameter_count'], receipts[1]['trainable_parameter_count'])
        canonical = models[0].residual.W.reshape(-1, len(GLOBAL_SOLVERS), 32).permute(1, 0, 2)
        torch.testing.assert_close(canonical, models[1].residual.W, rtol=0, atol=0)
        batch, _ = fixture_batch('TSP', 3)
        for model in models:
            model.eval()
        torch.testing.assert_close(models[0](runner.engine.inputs_only(batch))['logits'],
            models[1](runner.engine.inputs_only(batch))['logits'], rtol=0, atol=0)

    def test_both_real_models_update_with_exact_original_stateless_objective(self):
        initial, geometry = fresh_fixture()
        batch, _ = fixture_batch('TSP', 5)
        for arm in runner.ARMS:
            model, _, _ = runner.initialize_shared(initial, geometry, arm)
            objective = runner.make_objective(model, arm)
            self.assertIs(type(objective), RDropObjective)
            optimizer = CountingSGD(model.parameters())
            before = parameter_hash(model.state_dict())
            parts, retries, extras = runner.update_batch(objective, optimizer,
                torch.amp.GradScaler('cpu', enabled=False), batch, .1,
                Runtime(0, 1, torch.device('cpu')), cap=2, problem='TSP')
            self.assertEqual((optimizer.steps, retries, extras), (1, 0, {}))
            self.assertTrue(torch.isfinite(parts).all())
            self.assertNotEqual(before, parameter_hash(model.state_dict()))
            self.assertGreater(model.residual.v.abs().sum().item(), 0.)
            self.assertEqual(runner.checkpoint_extras(objective), {})
            runner.restore_objective(objective, {})
            with self.assertRaises(ValueError):
                runner.restore_objective(objective, dict(objective_state={}))

    def test_microbatch_real_row_weighting_matches_whole_batch_without_dropout(self):
        initial, geometry = fresh_fixture()
        batch, _ = fixture_batch('CVRP', 5)
        for arm in runner.ARMS:
            records = []
            for cap in (2, 5):
                model, _, _ = runner.initialize_shared(initial, geometry, arm)
                model.eval()
                objective = runner.make_objective(model, arm)
                optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
                parts, retries, _ = runner.update_batch(objective, optimizer,
                    torch.amp.GradScaler('cpu', enabled=False), batch, .1,
                    Runtime(0, 1, torch.device('cpu')), cap=cap, problem='CVRP')
                self.assertEqual(retries, 0)
                records.append((model.state_dict(), parts))
            torch.testing.assert_close(records[0][1], records[1][1], rtol=2e-5, atol=2e-6)
            for name in records[0][0]:
                torch.testing.assert_close(records[0][0][name], records[1][0][name], rtol=2e-5, atol=2e-6)

    def test_amp_retry_replays_dropout_without_duplicate_optimizer_update(self):
        initial, geometry = fresh_fixture()
        batch, _ = fixture_batch('TSP', 5)
        states = []
        for fail in (False, True):
            model, _, _ = runner.initialize_shared(initial, geometry, runner.ARMS[1])
            objective = FailOnce(model) if fail else runner.make_objective(model, runner.ARMS[1])
            optimizer = CountingSGD(model.parameters())
            torch.manual_seed(982)
            parts, retries, _ = runner.update_batch(objective, optimizer,
                torch.amp.GradScaler('cpu', enabled=False), batch, .1,
                Runtime(0, 1, torch.device('cpu')), cap=2, problem='TSP')
            self.assertEqual((optimizer.steps, retries), (1, int(fail)))
            states.append((parameter_hash(model.state_dict()), torch.get_rng_state(), parts))
        self.assertEqual(states[0][0], states[1][0])
        torch.testing.assert_close(states[0][1], states[1][1], rtol=0, atol=0)
        torch.testing.assert_close(states[0][2], states[1][2], rtol=0, atol=0)

    def test_checkpoint_factory_roundtrip_requires_no_objective_or_labels(self):
        initial, geometry = fresh_fixture()
        for arm in runner.ARMS:
            model, params, _ = runner.initialize_shared(initial, geometry, arm)
            with torch.no_grad():
                model.residual.v.fill_(.123)
            config = dict(experiment='R87', r87_schema_version=1, arm=arm, seed=2,
                initialization='scratch_shared_r77', model_params=params,
                solver_names=GLOBAL_SOLVERS, max_successful_updates=57600)
            state = dict(args=config, state_hash=runner.json_hash(config), model=model.state_dict())
            with tempfile.TemporaryDirectory() as temporary:
                path = Path(temporary) / 'model.pt'
                save_checkpoint(path, state)
                with patch.object(runner, 'make_objective', side_effect=AssertionError('labels not available')):
                    restored, _ = runner.load_checkpoint(path)
                self.assertEqual(parameter_hash(restored.state_dict()), parameter_hash(model.state_dict()))
                save_checkpoint(path, dict(state, objective_state={}))
                with self.assertRaises(ValueError):
                    runner.load_checkpoint(path)

    def test_main_lock_gates_before_training_and_rejects_changed_settings(self):
        args = runner.parser().parse_args(['train', '--arm', runner.ARMS[0]])
        with tempfile.TemporaryDirectory() as temporary:
            args.root = Path(temporary)
            with patch.object(runner.engine, 'train') as train:
                with self.assertRaises(FileNotFoundError):
                    runner.train(args, torch.device('cpu'))
                train.assert_not_called()
            path = args.root / 'protocol_lock.json'
            record = dict(experiment='R87', locked=True, arms=list(runner.ARMS), seed=2,
                batch_size=1024, max_updates=57600, warmup_updates=4320, eval_every=1440)
            write_json(path, record)
            self.assertTrue(runner.protocol_lock(args)['record']['locked'])
            write_json(path, dict(record, batch_size=512))
            with self.assertRaises(ValueError):
                runner.protocol_lock(args)


if __name__ == '__main__':
    unittest.main()

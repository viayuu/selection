"""Small actual CPU integrations for isolated R89 optimizer construction/state."""

import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import torch

from code.unified_selector.registry import GLOBAL_SOLVERS, PROBLEMS
from . import r84_experiment as shared
from . import r88_experiment
from . import r89_experiment as runner
from . import r89_reporting as reporter
from .r76_experiment import RDropObjective, Runtime, save_checkpoint, write_json
from .r77_data import parameter_hash
from .r89_engine_adapter import PrivateEngineAdapter, TrainingHooks
from .test_r80 import fresh_fixture
from .test_r88_experiment import native_tsp_batch


def equal_state(test, left, right):
    if torch.is_tensor(left):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, dict):
        test.assertEqual(set(left), set(right))
        for key in left:
            equal_state(test, left[key], right[key])
    elif isinstance(left, (list, tuple)):
        test.assertEqual(len(left), len(right))
        for a, b in zip(left, right):
            equal_state(test, a, b)
    else:
        test.assertEqual(left, right)


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


class R89RunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_exact_cli_fixed_schedule_and_original_objective(self):
        with patch.object(torch, 'load', side_effect=AssertionError('describe must not load data')):
            for arm in runner.ARMS:
                with redirect_stdout(io.StringIO()):
                    value = runner.main(['describe', '--arm', arm])
                self.assertEqual((value['experiment'], value['model_role_arm'], value['new_parameter_count']),
                    ('R89', 'B_execution_anchor', 0))
                self.assertEqual((value['batch_size'], value['max_updates'], value['warmup_updates'],
                    value['eval_every']), (1024, 57600, 4320, 1440))
                for mode in ('train', 'test'):
                    args = runner.parser().parse_args([mode, '--arm', arm, '--wandb-mode', 'offline'])
                    self.assertEqual((args.gpu_type, args.large_micro_cap, args.eval_batch_size), ('4090', 352, 256))
                for mode in ('lib', 'report'):
                    self.assertEqual(reporter.parser().parse_args([mode, '--arm', arm]).command, mode)
        self.assertEqual(set(runner.task_order(57600)), set(PROBLEMS))
        self.assertEqual(runner.learning_rate(4320), 1e-4)
        self.assertEqual(runner.learning_rate(57600), 1e-6)

    def test_private_engine_facade_preserves_shared_torch_and_actual_optimizer_types(self):
        initial, geometry = fresh_fixture()
        original_adamw, original_optim = torch.optim.AdamW, torch.optim
        for arm in runner.ARMS:
            selector, _, _ = runner.initialize_shared(initial, geometry, arm)
            adapter = PrivateEngineAdapter(arm)
            adapter.bind(selector)
            optimizer = adapter.engine.torch.optim.AdamW(selector.parameters(), lr=1e-4, weight_decay=1e-4)
            self.assertIs(torch.optim.AdamW, original_adamw)
            self.assertIs(torch.optim, original_optim)
            self.assertIs(shared.torch, torch)
            self.assertIs(r88_experiment.engine.torch.optim.AdamW, original_adamw)
            self.assertIsNot(adapter.engine, shared)
            self.assertIsNot(adapter.engine.torch, torch)
            self.assertIs(adapter.engine.torch.nn, torch.nn)
            self.assertIs(adapter.engine.torch.amp, torch.amp)
            if arm == runner.ARMS[0]:
                self.assertIs(type(optimizer), original_adamw)
            else:
                self.assertEqual(type(optimizer).__name__, 'MuonHybrid')
                self.assertEqual([group['algorithm'] for group in optimizer.param_groups], ['muon', 'adamw'])
                self.assertIs(type(optimizer._adamw), original_adamw)
            receipt = adapter.static_receipt()
            self.assertTrue(receipt['shared_torch_adamw_unchanged'])
            self.assertEqual(sum(group['parameter_count'] for group in receipt['groups']),
                sum(parameter.numel() for parameter in selector.parameters()))

    def test_both_fresh_models_are_identical_with_no_objective_state(self):
        initial, geometry = fresh_fixture()
        states, configurations = [], []
        for arm in runner.ARMS:
            selector, params, receipt = runner.initialize_shared(initial, geometry, arm)
            self.assertEqual(receipt['common_parameter_sha256'], initial['trainable_sha256'])
            self.assertEqual((receipt['new_parameter_count'], params['role_arm']), (0, 'B_execution_anchor'))
            objective = runner.make_objective(selector, arm)
            self.assertIs(type(objective), RDropObjective)
            self.assertEqual(runner.checkpoint_extras(objective), {})
            runner.restore_objective(objective, {})
            states.append(parameter_hash(selector.state_dict()))
            configurations.append(params)
        self.assertEqual(states[0], states[1])
        self.assertEqual(configurations[0], configurations[1])

    def test_injected_control_has_exact_original_adamw_update_state_and_rng(self):
        initial, geometry = fresh_fixture()
        batch, _ = native_tsp_batch(5)
        results = []
        for injected in (False, True):
            selector, _, _ = runner.initialize_shared(initial, geometry, runner.ARMS[0])
            objective = runner.make_objective(selector, runner.ARMS[0])
            if injected:
                adapter = PrivateEngineAdapter(runner.ARMS[0])
                adapter.bind(selector)
                optimizer = adapter.engine.torch.optim.AdamW(selector.parameters(), lr=1e-4, weight_decay=1e-4)
                engine = adapter.engine
            else:
                optimizer = torch.optim.AdamW(selector.parameters(), lr=1e-4, weight_decay=1e-4)
                engine = shared
            torch.manual_seed(891)
            parts, retries, _ = engine.update_batch(objective, optimizer,
                torch.amp.GradScaler('cpu', enabled=False), batch, .1,
                Runtime(0, 1, torch.device('cpu')), cap=2)
            self.assertEqual(retries, 0)
            results.append((parameter_hash(selector.state_dict()), optimizer.state_dict(), parts, torch.get_rng_state()))
        self.assertEqual(results[0][0], results[1][0])
        for index in (1, 2, 3):
            equal_state(self, results[0][index], results[1][index])

    def test_actual_hybrid_amp_retry_and_state_resume_match_uninterrupted_steps(self):
        initial, geometry = fresh_fixture()
        batch, _ = native_tsp_batch(5)
        complete = []
        for fail in (False, True):
            model, _, _ = runner.initialize_shared(initial, geometry, runner.ARMS[1])
            adapter = PrivateEngineAdapter(runner.ARMS[1])
            adapter.bind(model)
            optimizer = adapter.engine.torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
            objective = FailOnce(model) if fail else runner.make_objective(model, runner.ARMS[1])
            torch.manual_seed(892)
            _, retries, _ = adapter.engine.update_batch(objective, optimizer,
                torch.amp.GradScaler('cpu', enabled=False), batch, .1,
                Runtime(0, 1, torch.device('cpu')), cap=2)
            self.assertEqual((retries, adapter.successful_steps, optimizer.completed_steps), (int(fail), 1, 1))
            metrics = adapter.step_metrics()
            self.assertEqual(float(metrics['muon_step_applied']), 1.)
            self.assertGreater(float(metrics['muon_direction_rms']), 0.)
            self.assertTrue(all(torch.isfinite(torch.as_tensor(value)).all() for value in metrics.values()))
            complete.append((parameter_hash(model.state_dict()), optimizer.state_dict()))
        self.assertEqual(complete[0][0], complete[1][0])
        equal_state(self, complete[0][1], complete[1][1])
        checkpoint = dict(successful_updates=1, optimizer_provenance=adapter.checkpoint_receipt())
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'cpu-optimizer.pt'
            save_checkpoint(path, dict(checkpoint, model=model.state_dict(), optimizer=optimizer.state_dict()))
            state = torch.load(path, map_location='cpu', weights_only=False)
        restored, _, _ = runner.initialize_shared(initial, geometry, runner.ARMS[1])
        restored.load_state_dict(state['model'])
        resumed = PrivateEngineAdapter(runner.ARMS[1])
        resumed.bind(restored)
        restored_optimizer = resumed.prepare_optimizer(1e-4, 1e-4)
        resumed.restore_step_receipt(state)
        restored_optimizer.load_state_dict(state['optimizer'])
        for value in restored_optimizer.state.values():
            for key, tensor in value.items():
                if torch.is_tensor(tensor):
                    value[key] = tensor.cpu()
        for selector, opt, context in ((model, optimizer, adapter), (restored, restored_optimizer, resumed)):
            torch.manual_seed(893)
            context.engine.update_batch(runner.make_objective(selector, runner.ARMS[1]), opt,
                torch.amp.GradScaler('cpu', enabled=False), batch, .1,
                Runtime(0, 1, torch.device('cpu')), cap=2)
        self.assertEqual((adapter.successful_steps, resumed.successful_steps), (2, 2))
        self.assertEqual(parameter_hash(model.state_dict()), parameter_hash(restored.state_dict()))
        equal_state(self, optimizer.state_dict(), restored_optimizer.state_dict())

    def test_checkpoint_model_factory_does_not_construct_an_optimizer(self):
        initial, geometry = fresh_fixture()
        for arm in runner.ARMS:
            model, params, _ = runner.initialize_shared(initial, geometry, arm)
            config = dict(experiment='R89', r89_schema_version=1, arm=arm, seed=2,
                initialization='scratch_shared_r77', model_params=params,
                solver_names=GLOBAL_SOLVERS, max_successful_updates=57600)
            state = dict(args=config, state_hash=runner.json_hash(config), model=model.state_dict())
            with tempfile.TemporaryDirectory() as temporary:
                path = Path(temporary) / 'model.pt'
                save_checkpoint(path, state)
                with patch('code.V4.r89_optim.build_optimizer', side_effect=AssertionError('inference must be optimizer free')):
                    reconstructed, _ = runner.load_checkpoint(path)
                self.assertEqual(parameter_hash(reconstructed.state_dict()), parameter_hash(model.state_dict()))

    def test_main_lock_blocks_before_private_engine_and_rejects_muon_lr_multiplier(self):
        args = runner.parser().parse_args(['train', '--arm', runner.ARMS[1]])
        with tempfile.TemporaryDirectory() as temporary:
            args.root = Path(temporary)
            with patch.object(runner, 'PrivateEngineAdapter') as adapter:
                with self.assertRaises(FileNotFoundError):
                    runner.train(args, torch.device('cpu'))
                adapter.assert_not_called()
            record = dict(experiment='R89', locked=True, arms=list(runner.ARMS), seed=2,
                batch_size=1024, max_updates=57600, warmup_updates=4320, eval_every=1440)
            path = args.root / 'protocol_lock.json'
            write_json(path, record)
            self.assertTrue(runner.protocol_lock(args)['record']['locked'])
            write_json(path, dict(record, B_muon_hybrid=dict(lr_multiplier=5.)))
            with self.assertRaises(ValueError):
                runner.protocol_lock(args)


if __name__ == '__main__':
    unittest.main()

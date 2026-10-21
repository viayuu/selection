"""CPU integration checks with the actual R85 model and objective factories."""

import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import torch

from code.unified_selector.registry import GLOBAL_SOLVERS, PROBLEMS
from . import r85_experiment as runner
from .r76_experiment import Runtime, save_checkpoint
from .r77_data import parameter_hash
from .r85_objectives import CandidateSubsetRDropObjective
from .test_r76 import fixture_batch
from .test_r80 import fresh_fixture


def assert_state_equal(test, left, right):
    if torch.is_tensor(left):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, dict):
        test.assertEqual(set(left), set(right))
        for key in left:
            assert_state_equal(test, left[key], right[key])
    else:
        test.assertEqual(left, right)


class RetryCandidate(CandidateSubsetRDropObjective):
    def __init__(self, selector, fail=False):
        super().__init__(selector)
        self.fail_once, self.attempt_masks = fail, []

    def begin_attempt(self, batch, weight):
        prepared = super().begin_attempt(batch, weight)
        self.attempt_masks.append(prepared['aux_solver_mask'].clone())
        return prepared

    def forward(self, batch, weight):
        loss, parts = super().forward(batch, weight)
        if self.fail_once:
            self.fail_once = False
            loss = loss * loss.new_tensor(float('nan'))
        return loss, parts


class R85RunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_exact_cli_and_common_budget_do_not_open_data(self):
        with patch.object(torch, 'load', side_effect=AssertionError('CLI must not open data')):
            for arm in runner.ARMS:
                with redirect_stdout(io.StringIO()):
                    description = runner.main(['describe', '--arm', arm])
                self.assertEqual(description['experiment'], 'R85')
                self.assertEqual((description['batch_size'], description['max_updates'],
                    description['warmup_updates'], description['eval_every']), (1024, 57600, 4320, 1440))
                for command in ('train', 'test'):
                    args = runner.parser().parse_args([command, '--arm', arm, '--wandb-mode', 'offline'])
                    self.assertEqual(args.experiment, 'R85')
        self.assertEqual(runner.learning_rate(4320), 1e-4)
        self.assertEqual(runner.learning_rate(57600), 1e-6)
        self.assertEqual(set(runner.task_order(57600)), set(PROBLEMS))

    def test_all_four_actual_models_objectives_update_and_resume_auxiliary_state(self):
        initial, geometry = fresh_fixture()
        batch, _ = fixture_batch('TSP', 5)
        hashes = []
        for arm in runner.ARMS:
            with self.subTest(arm=arm):
                model, params, receipt = runner.initialize_shared(initial, geometry, arm)
                hashes.append(parameter_hash(dict(model.named_parameters())))
                self.assertFalse(receipt['historical_checkpoint_loaded'])
                self.assertEqual(receipt['new_parameter_names'], [])
                objective = runner.make_objective(model, arm)
                optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
                scaler = torch.amp.GradScaler('cpu', enabled=False)
                before = parameter_hash(dict(model.named_parameters()))
                parts, retries, extras = runner.update_batch(objective, optimizer, scaler, batch,
                    .1, Runtime(0, 1, torch.device('cpu')), cap=2, problem='TSP')
                self.assertEqual(retries, 0)
                self.assertTrue(torch.isfinite(parts).all())
                self.assertTrue(extras)
                self.assertNotEqual(before, parameter_hash(dict(model.named_parameters())))
                self.assertEqual(objective.successful_steps, 1)
                saved = runner.checkpoint_extras(objective)
                clone = runner.make_objective(model, arm)
                runner.restore_objective(clone, saved)
                assert_state_equal(self, clone.training_state_dict(), objective.training_state_dict())
                with self.assertRaises(ValueError):
                    runner.restore_objective(clone, {})
        self.assertEqual(len(set(hashes)), 1)

    def test_candidate_amp_retry_replays_mask_dropout_and_commits_once(self):
        initial, geometry = fresh_fixture()
        batch, _ = fixture_batch('TSP', 5)
        outcomes = []
        for fail in (False, True):
            model, _, _ = runner.initialize_shared(initial, geometry, 'D_candidate_subset')
            objective = RetryCandidate(model, fail=fail)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
            scaler = torch.amp.GradScaler('cpu', enabled=False)
            torch.manual_seed(17)
            parts, retries, _ = runner.update_batch(objective, optimizer, scaler, batch,
                .1, Runtime(0, 1, torch.device('cpu')), cap=2, problem='TSP')
            self.assertEqual(retries, int(fail))
            self.assertEqual(objective.successful_steps, 1)
            for mask in objective.attempt_masks:
                self.assertTrue(torch.equal(mask, objective.attempt_masks[0]))
                self.assertTrue(torch.equal(mask, mask[:1].expand_as(mask)))
                self.assertTrue((mask.sum(-1) >= 2).all())
                self.assertTrue((mask.sum(-1) < len(batch['pool_ids'])).all())
            outcomes.append((parameter_hash(model.state_dict()), objective.training_state_dict(),
                torch.get_rng_state(), parts))
        self.assertEqual(outcomes[0][0], outcomes[1][0])
        assert_state_equal(self, outcomes[0][1], outcomes[1][1])
        torch.testing.assert_close(outcomes[0][2], outcomes[1][2], rtol=0, atol=0)
        torch.testing.assert_close(outcomes[0][3], outcomes[1][3], rtol=0, atol=0)

    def test_checkpoint_factory_needs_no_objective_or_training_labels(self):
        initial, geometry = fresh_fixture()
        for arm in runner.ARMS:
            model, params, _ = runner.initialize_shared(initial, geometry, arm)
            config = dict(experiment='R85', arm=arm, seed=2, initialization='scratch_shared_r77',
                solver_names=GLOBAL_SOLVERS, model_params=params, max_successful_updates=57600)
            state = dict(args=config, state_hash=runner.json_hash(config), model=model.state_dict())
            with tempfile.TemporaryDirectory() as temporary:
                path = Path(temporary) / 'synthetic.pt'
                save_checkpoint(path, state)
                with patch.object(runner, 'make_objective', side_effect=AssertionError('inference must be label free')):
                    restored, _ = runner.load_checkpoint(path)
            self.assertEqual(parameter_hash(restored.state_dict()), parameter_hash(model.state_dict()))

    def test_formal_training_requires_main_protocol_lock(self):
        args = runner.parser().parse_args(['train', '--arm', 'A_ovr_control'])
        with tempfile.TemporaryDirectory() as temporary:
            args.root = Path(temporary)
            with self.assertRaises(FileNotFoundError), patch.object(runner.engine, 'train') as train:
                runner.train(args, torch.device('cpu'))
            train.assert_not_called()


if __name__ == '__main__':
    unittest.main()

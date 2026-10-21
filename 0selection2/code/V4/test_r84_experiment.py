"""CPU witnesses for common initialization, actual updates, and immutable locks."""

import io
import tempfile
import unittest
from collections import Counter
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import torch

from code.unified_selector.registry import GLOBAL_SOLVERS, PROBLEMS
from . import r84_experiment as runner
from .r76_experiment import Runtime
from .r77_data import parameter_hash
from .test_r76 import fixture_batch
from .test_r80 import fresh_fixture


class R84RunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_all_arm_cli_descriptions_do_not_open_data(self):
        with patch.object(torch, 'load', side_effect=AssertionError('describe must not load weights')):
            for arm in runner.ARMS:
                with redirect_stdout(io.StringIO()):
                    result = runner.main(['describe', '--arm', arm])
                self.assertEqual((result['batch_size'], result['max_updates'], result['eval_every'], result['warmup_updates']),
                                 (1024, 57600, 1440, 4320))
                self.assertEqual(result['focal'], arm in runner.FOCAL_ARMS)
                self.assertEqual(result['attention'], arm in runner.ATTENTION_ARMS)
                for command in ('train', 'test'):
                    args = runner.parser().parse_args([command, '--arm', arm, '--wandb-mode', 'offline'])
                    self.assertEqual(args.experiment, 'R84')

    def test_full18_common_schedule_and_fixed_lr(self):
        order = runner.task_order(runner.MAX_UPDATES)
        self.assertEqual(len(order), 57600)
        self.assertEqual(Counter(order), Counter(dict.fromkeys(PROBLEMS, 3200)))
        self.assertTrue(all(a != b for a, b in zip(order, order[1:])))
        self.assertEqual(runner.learning_rate(4320), 1e-4)
        self.assertEqual(runner.learning_rate(57600), 1e-6)
        self.assertEqual(runner.consistency_weight(4320), .35)
        self.assertEqual(runner.micro_caps('3090')['TSP'], 384)
        self.assertEqual(runner.micro_caps('4090')['CVRP'], 352)

    def test_same_fresh_common_parameters_and_strict_roundtrip_all_arms(self):
        initial, geometry = fresh_fixture()
        references = {}
        batch, _ = fixture_batch('TSP', 3)
        for arm in runner.ARMS:
            model, params, receipt = runner.initialize_shared(initial, geometry, arm)
            self.assertFalse(receipt['historical_checkpoint_loaded'])
            for name, tensor in model.named_parameters():
                torch.testing.assert_close(tensor, initial['trainable'][name], rtol=0, atol=0)
            references[arm] = parameter_hash(dict(model.named_parameters()))
            config = dict(experiment='R84', arm=arm, seed=2, initialization='scratch_shared_r77',
                model_params=params, solver_names=GLOBAL_SOLVERS, max_successful_updates=57600)
            state = dict(args=config, state_hash=runner.json_hash(config), model=model.state_dict(), initialization=receipt)
            with tempfile.TemporaryDirectory() as temporary:
                path = Path(temporary) / 'synthetic.pt'
                torch.save(state, path)
                restored, _ = runner.load_checkpoint(path)
            model.eval()
            restored.eval()
            with torch.no_grad():
                torch.testing.assert_close(model(runner.inputs_only(batch))['logits'],
                                           restored(runner.inputs_only(batch))['logits'], rtol=0, atol=0)
        self.assertEqual(references['A_baseline'], references['C_focal'])
        self.assertEqual(references['B_attention_direct'], references['D_attention_focal'])

    def test_each_arm_one_finite_microbatched_cpu_update(self):
        initial, geometry = fresh_fixture()
        batch, _ = fixture_batch('TSP', 3)
        for arm in runner.ARMS:
            model, _, _ = runner.initialize_shared(initial, geometry, arm)
            objective = runner.make_objective(model, arm)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
            scaler = torch.amp.GradScaler('cpu', enabled=False)
            before = parameter_hash(dict(model.named_parameters()))
            parts, retries, focal = runner.update_batch(objective, optimizer, scaler, batch, .1,
                Runtime(0, 1, torch.device('cpu')), cap=2)
            self.assertEqual(retries, 0)
            self.assertTrue(torch.isfinite(parts).all())
            self.assertNotEqual(before, parameter_hash(dict(model.named_parameters())))
            if arm in runner.FOCAL_ARMS:
                self.assertGreater(float(focal['focal']), 0.)
                self.assertEqual(objective._focal_rows, 3)
            else:
                self.assertEqual(float(focal['focal']), 0.)

    def test_resume_requires_matching_schedule_and_update_boundary(self):
        order = runner.task_order(runner.MAX_UPDATES)
        step = runner.EVAL_EVERY
        counts = Counter(order[:step])
        config = dict(synthetic=True)
        state = dict(args=config, state_hash=runner.json_hash(config), successful_updates=step,
            history=[{}], pending_evaluation=None, updates_by_problem=dict(counts),
            stream_draws={p: counts[p] * 1024 for p in PROBLEMS})
        runner.validate_resume(state, config, order)
        state['stream_draws']['TSP'] += 1
        with self.assertRaises(ValueError):
            runner.validate_resume(state, config, order)


if __name__ == '__main__':
    unittest.main()

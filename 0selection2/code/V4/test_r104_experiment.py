"""Focused CPU integration checks, no training launch or real cache reads."""

import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from code.unified_selector.registry import PROBLEMS
from . import r104_experiment as runner
from .r76_experiment import Runtime, save_checkpoint, write_json
from .r77_data import parameter_hash


class TinyObjective(torch.nn.Module):
    def __init__(self, fail=False):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(()))
        self.fail = fail

    def forward(self, batch, weight):
        loss = (self.weight * batch['x']).square().mean()
        if self.fail:
            self.fail = False
            loss = loss * float('nan')
        return loss, loss.detach().expand(8)


class R104RunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_cli_main_lock_schedule_and_private_engine_caps(self):
        for arm in runner.ARMS:
            for command in ('train', 'test', 'timing', 'describe'):
                args = runner.parser().parse_args([command, '--arm', arm])
                self.assertEqual((args.large_micro_cap, args.eval_batch_size), (128, 128))
                self.assertEqual(args.cache_root, Path('/tmp/shiys_R84_new90k_cache'))
        args = runner.parser().parse_args(['train', '--arm', runner.ARMS[0]])
        # Read only: prove compatibility with main's already-created scientific lock.
        self.assertTrue(runner.protocol_lock(args)['record']['locked'])
        original_caps = runner.engine.micro_caps('3090', 128)
        local = runner.private_engine()
        self.assertIs(local.train.__globals__['micro_caps'], runner.micro_caps)
        self.assertIs(local.evaluate.__wrapped__.__globals__['micro_caps'], runner.micro_caps)
        self.assertIs(local.test.__globals__['write_json'], runner._engine_write_json)
        self.assertEqual(runner.engine.micro_caps('3090', 128), original_caps)
        self.assertEqual(original_caps['ATSP'], 1024)
        self.assertEqual(runner.micro_caps()['ATSP'], 512)
        self.assertEqual(runner.micro_caps()['TSP'], 128)
        self.assertEqual(set(runner.task_order(57600)), set(PROBLEMS))
        self.assertEqual(runner.learning_rate(4320), 1e-4)
        self.assertEqual(runner.learning_rate(57600), 1e-6)
        with self.assertRaises(ValueError):
            runner.micro_caps('3090', 64)

    def test_protocol_uses_actual_capacity_and_full_fixed_caps(self):
        for arm in runner.ARMS:
            with tempfile.TemporaryDirectory() as temporary:
                args = runner.parser().parse_args(['train', '--arm', arm, '--root', temporary])
                args.prepared_root = Path(temporary) / 'prepared'
                write_json(args.prepared_root / 'prepared.json', {})
                lock = dict(runner.required_lock_fields(), data_root=str(args.data_root),
                    prepared_root=str(args.prepared_root), primary_cache=str(args.cache_root))
                write_json(args.root / 'protocol_lock.json', lock)
                params = dict(runner.models.CAPACITIES[arm], role_arm=runner.MODEL_ROLE_ARM)
                manifest = dict(splits=dict(train={p: dict(pool=['x'], pool_ids=[0]) for p in PROBLEMS}))
                config = runner.protocol(args, dict(trainable_sha256='fresh'), manifest, {}, {},
                    params, runner.task_order(57600), runner.micro_caps())
                runner.validate_checkpoint_config(config)
                self.assertEqual(config['trainable_parameter_count'], runner.PARAMETER_COUNTS[arm])
                self.assertEqual(config['microbatch_caps']['ATSP'], 512)
                self.assertEqual(config['loss'], runner.engine.LOSS_RECIPE)
                self.assertFalse(config['parameter_matched'])
                broken = copy.deepcopy(config)
                broken['microbatch_caps']['ATSP'] = 1024
                with self.assertRaises(ValueError):
                    runner.validate_checkpoint_config(broken)

    def test_fixed_micro_row_weighted_retry_accepts_one_update(self):
        batch = dict(ind=torch.ones(1024, dtype=torch.long), x=torch.linspace(.1, 1., 1024))
        outcomes = []
        for fail in (False, True):
            objective = TinyObjective(fail)
            optimizer = torch.optim.SGD(objective.parameters(), lr=.01)
            with patch.object(optimizer, 'step', wraps=optimizer.step) as step:
                parts, retries, extras = runner.update_batch(objective, optimizer,
                    torch.amp.GradScaler('cpu', enabled=False), batch, .35,
                    Runtime(0, 1, torch.device('cpu')), 128, problem='TSP')
            self.assertEqual(step.call_count, 1)
            self.assertEqual(retries, int(fail))
            self.assertTrue(torch.isfinite(parts).all())
            self.assertEqual(set(extras), {'focal', 'focal_extra'})
            outcomes.append(objective.weight.detach().clone())
        torch.testing.assert_close(*outcomes, rtol=0, atol=0)
        with self.assertRaises(ValueError):
            runner.update_batch(objective, optimizer, None, batch, .35, None, 512, problem='TSP')

    def test_official_val_binding_preserves_null_capture_and_rejects_wrong_state(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            config = dict(experiment='R104', arm=runner.ARMS[0])
            state_hash = runner.json_hash(config)
            weights = dict(weight=torch.tensor([.2]))
            save_checkpoint(run / 'best.pt', dict(args=config, model=weights,
                state_hash=state_hash, successful_updates=1440))
            lock = dict(arm=runner.ARMS[0], checkpoint_sha256=runner.file_hash(run / 'best.pt'),
                state_hash=state_hash, selected_update=1440)
            write_json(run / 'final_test_lock.json', lock)
            identity = dict(experiment='R104', arm=runner.ARMS[0], split='val', checkpoint_sha256=None,
                state_hash=state_hash, model_state_sha256=parameter_hash(weights), successful_updates=1440)
            archive = run / 'evaluation/val/update0001440/rank000.npz'
            write_json(archive.parent / 'identity.json', identity)
            original_bytes = (archive.parent / 'identity.json').read_bytes()
            final = dict(state_hash=state_hash, selected_update=1440,
                checkpoint_sha256=lock['checkpoint_sha256'], val=dict(identity=identity, prediction_archive=str(archive)))
            result = runner.bind_selected_val(run, final)
            self.assertEqual(result['val']['identity']['checkpoint_sha256'], lock['checkpoint_sha256'])
            self.assertIsNone(result['val']['archive_capture_identity']['checkpoint_sha256'])
            self.assertEqual((archive.parent / 'identity.json').read_bytes(), original_bytes)
            self.assertEqual(runner.bind_selected_val(run, result), result)
            poisoned = copy.deepcopy(final)
            poisoned['val']['identity']['model_state_sha256'] = 'different'
            with self.assertRaises(ValueError):
                runner.bind_selected_val(run, poisoned)


if __name__ == '__main__':
    unittest.main()

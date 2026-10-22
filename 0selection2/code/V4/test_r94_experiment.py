"""Small R94 protocol/engine tests; no allocation or donor-weight access."""

from argparse import Namespace
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import torch

from . import r94_experiment as experiment
from .r76_experiment import read_json, write_json


class R94ProtocolTests(unittest.TestCase):
    def test_original_objective_not_r93_pure_ce(self):
        value = experiment.objective_spec('A')
        self.assertEqual(value, experiment.objective_spec('B'))
        self.assertEqual(value['coefficients'], dict(ce=.35, comparison=.35, risk=.02))
        self.assertEqual(value['forward_passes'], 2)
        self.assertFalse(value['route_targets'])

    def test_feature_roots_do_not_overlap(self):
        a = Namespace(arm='A', feature_root=None)
        b = Namespace(arm='B', feature_root=None)
        self.assertNotEqual(experiment.feature_root(a), experiment.feature_root(b))

    def test_lock_rejects_changed_data_or_budget(self):
        original = read_json(experiment.DEFAULT_ROOT / 'protocol_lock.json')
        with TemporaryDirectory() as directory:
            root = Path(directory)
            args = Namespace(root=root, protocol_lock=None,
                data_root=Path(original['data_root']), prepared_root=Path(original['prepared_root']))
            write_json(root / 'protocol_lock.json', original)
            self.assertEqual(experiment.protocol_lock(args)['record'], original)
            for key, value in (('max_updates', 200), ('donor_variants', dict(A='pretrained', B='pretrained'))):
                write_json(root / 'protocol_lock.json', dict(original, **{key: value}))
                with self.assertRaises(ValueError):
                    experiment.protocol_lock(args)

    def test_per_task_microcap_not_universal352(self):
        batch = dict(ind=torch.zeros(1024, dtype=torch.int64))
        with patch.object(experiment.engine, 'update_batch', return_value='accepted') as update:
            self.assertEqual(experiment.update_batch(None, None, None, batch, .2, None,
                1024, problem='VRPL'), 'accepted')
            self.assertEqual(experiment.update_batch(None, None, None, batch, .2, None,
                352, problem='TSP'), 'accepted')
            with self.assertRaises(ValueError):
                experiment.update_batch(None, None, None, batch, .2, None, 352, problem='VRPL')
            self.assertEqual(update.call_count, 2)

    def test_all_cli_stages_have_fixed_defaults(self):
        for stage in ('prepare', 'train', 'test', 'timing'):
            for arm in experiment.ARMS:
                args = experiment.parser().parse_args([stage, '--arm', arm])
                self.assertEqual(args.eval_batch_size, 256)
                self.assertEqual(args.large_micro_cap, 352)
                self.assertEqual(args.wandb_mode, 'offline')
                self.assertFalse(args.resume)

    def test_no_mutable_objective_state(self):
        self.assertEqual(experiment.checkpoint_extras(None), {})
        experiment.restore_objective(None, {})
        with self.assertRaises(ValueError):
            experiment.restore_objective(None, dict(objective_state={}))

    def test_private_engine_has_no_shared_global_patch(self):
        original_open = experiment.engine.open_cache
        original_inputs = experiment.engine.inputs_only
        with TemporaryDirectory() as directory:
            module = experiment.private_engine('A', directory)
            self.assertIs(experiment.engine.open_cache, original_open)
            self.assertIs(experiment.engine.inputs_only, original_inputs)
            self.assertIsNot(module, experiment.engine)
            self.assertIsNot(module.open_cache, original_open)


if __name__ == '__main__':
    unittest.main()

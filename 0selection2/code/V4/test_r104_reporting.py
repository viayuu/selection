"""Minimal saved-evidence reporting checks; no checkpoint/data/GPU inference."""

import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from . import r104_reporting as report
from .r76_experiment import write_json


class R104ReportingTests(unittest.TestCase):
    def test_cli_and_official_binding_capture_view(self):
        for arm in report.ARMS:
            for command in ('lib', 'report', 'status'):
                args = report.parser().parse_args([command, '--arm', arm])
                self.assertEqual(args.arm, arm)
                self.assertEqual(args.r89a_root.name, 'A_adamw')
        lock = dict(checkpoint_sha256='checkpoint', state_hash='protocol', selected_update=1440)
        capture = dict(checkpoint_sha256=None, state_hash='protocol', successful_updates=1440,
                       model_state_sha256='weights')
        val = dict(archive_capture_identity=capture,
            identity=dict(capture, checkpoint_sha256='checkpoint'),
            checkpoint_binding=dict(rule=report.runner.VAL_BINDING_RULE,
                checkpoint_sha256='checkpoint', state_hash='protocol', selected_update=1440,
                model_state_sha256='weights', archive_identity_unchanged=True))
        self.assertEqual(report.capture_val(val, lock)['identity'], capture)
        self.assertEqual(val['identity']['checkpoint_sha256'], 'checkpoint')
        altered = copy.deepcopy(val)
        altered['checkpoint_binding']['selected_update'] += 1
        with self.assertRaises(ValueError):
            report.capture_val(altered, lock)

    def test_true_rescues_false_picks_and_full_population_net_cost(self):
        costs = np.array([[1., 3., 4.], [4., 1., 3.], [3., 4., 1.], [1., 2., 3.]])
        def prediction(pred):
            return SimpleNamespace(n=4, pool=['x', 'y', 'z'], pool_ids=np.arange(3),
                indices=np.arange(4), source_indices=np.arange(4), costs=costs,
                winner=costs.argmin(1), pred=np.array(pred))
        old = {split: {p: prediction([1, 1, 0, 0]) for p in report.PROBLEMS}
               for split in ('TRAIN', 'VAL', 'TEST')}
        new = {split: {p: prediction([0, 0, 2, 1]) for p in report.PROBLEMS}
               for split in old}
        problem = report.PROBLEMS[0]
        identity = dict(identity=dict(rule='fixed historical target'), categories={problem + '/x': 'zero'})
        pairs, rows, totals = report.comparison_tables(old, new, [(problem, 'x')], identity, 'R104', 'old')
        self.assertEqual((rows[0]['rescued'], rows[0]['lost'], rows[0]['new_false_pick_count']), (1, 1, 1))
        self.assertAlmostEqual(rows[0]['new_false_pick_cost_delta_contribution'], .75)
        self.assertAlmostEqual(totals[0]['full18_net_mean_cost_delta'], 0.)
        self.assertEqual(len([r for r in pairs if r['problem'] == 'ALL']), 3)

    def test_timing_requires_actual_all_model_gpu_scope_not_cached_or_donor(self):
        lock = dict(checkpoint_sha256='checkpoint', state_hash='protocol')
        row = dict(instances=128, standalone_model=dict(batch_seconds=[.128] * 3,
                                                       mean_seconds_per_instance=.001))
        value = dict(experiment='R104', arm=report.ARMS[0], complete=True,
            checkpoint_sha256='checkpoint', state_hash='protocol', repetitions=3, warmup_batches=1,
            scope='complete_standalone_model_original_GPU_resident_inputs', all_current_model_included=True,
            donor_present=False, cached_feature_latency_used=False, end_to_end_measured=False,
            parameter_count=7288969, inference_rule=report.runner.INFERENCE_RULE, device='cuda:0',
            per_problem=[dict(row, problem=p) for p in report.PROBLEMS])
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            write_json(run / 'inference_timing.json', value)
            self.assertEqual(len(report.inference_timing(run, lock)['per_problem']), 18)
            write_json(run / 'inference_timing.json', dict(value, donor_present=True))
            with self.assertRaises(ValueError):
                report.inference_timing(run, lock)
            write_json(run / 'inference_timing.json', dict(value, per_problem=value['per_problem'][:-1]))
            with self.assertRaises(ValueError):
                report.inference_timing(run, lock)


if __name__ == '__main__':
    unittest.main()

"""Focused CPU evidence/adapter checks, no real checkpoint/data/label reads."""

import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from . import r105_reporting as report
from .r76_experiment import write_json
from .test_r105_experiment import capture_fixture


def timing_fixture(lock):
    return dict(experiment='R105', arm=lock['arm'], complete=True,
        checkpoint_sha256=lock['checkpoint_sha256'], state_hash=lock['state_hash'],
        selected_update=lock['selected_update'], model_state_sha256='weights',
        model_factory=report.runner.MODEL_FACTORY, inference_rule=report.runner.INFERENCE_RULE,
        model_reselection=False, routing_calibration=False, repetitions=3, warmup_batches=1,
        scope='complete_standalone_model_original_GPU_resident_inputs', all_current_model_included=True,
        all_four_experts_computed=True, donor_present=False, cached_feature_latency_used=False,
        end_to_end_measured=False, input_labels_read=False,
        sample_policy='first128 frozen TEST inputs per problem', parameter_count=3425561, device='cuda:0',
        excludes='dataset I/O, CPU-to-GPU transfer, training, routing diagnostic and solver execution',
        per_problem=[dict(problem=p, instances=128, standalone_model=dict(
            batch_seconds=[.128] * 3, mean_seconds_per_instance=.001)) for p in report.PROBLEMS])


def routing_fixture(lock):
    n = torch.tensor([3] * 127 + [5])
    batch = dict(n=n, node_mask=torch.arange(5)[None] < n[:, None])
    return dict(experiment='R105', arm=lock['arm'], complete=True,
        checkpoint_sha256=lock['checkpoint_sha256'], state_hash=lock['state_hash'],
        selected_update=lock['selected_update'], model_state_sha256='weights',
        model_factory=report.runner.MODEL_FACTORY, inference_rule=report.runner.INFERENCE_RULE,
        model_reselection=False, routing_calibration=False, **report.runner.ROUTING_DIAGNOSTIC,
        labels_read=False, targets_read=False, auxiliary_loss=False, solver_quotas=False,
        aggregation='entropy of graph-mean mixture, not mean node entropy',
        per_problem=[report.runner.routing_statistics(torch.full((128, 4, 4), .25), batch, p)
                     for p in report.PROBLEMS])


class R105ReportingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_cli_and_private_static_factory_root_isolation(self):
        for arm in report.ARMS:
            for command in ('lib', 'report', 'status'):
                args = report.parser().parse_args([command, '--arm', arm])
                self.assertEqual(args.arm, arm)
                self.assertEqual(args.root, report.runner.DEFAULT_ROOT)
                self.assertEqual(args.lib_batch_size, 2)
                self.assertEqual(args.r89a_root.name, 'A_adamw')
        self.assertIsNone(report.parser().parse_args(['compare']).arm)
        for private, original in ((report.common, report.r80_reporting),
                (report.shared, report.r84_reporting), (report.previous, report.r85_reporting),
                (report.helpers, report.r87_reporting)):
            self.assertIsNot(private, original)
            self.assertEqual(private.FACTORY, 'code.V4.r105_experiment')
            self.assertEqual(private.DEFAULT_ROOT, report.DEFAULT_ROOT)
            self.assertNotEqual(getattr(original, 'FACTORY', None), private.FACTORY)
            self.assertNotEqual(original.DEFAULT_ROOT, private.DEFAULT_ROOT)
        self.assertIs(report.shared.build_reports.__globals__, report.shared.__dict__)
        self.assertIs(report.shared.common, report.common)
        self.assertIs(report.helpers.previous, report.previous)
        self.assertIs(report.previous.selected_train, report.selected_train)

    def test_lib_consumes_explicit_current_factory_and_custom_root(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = report.parser().parse_args(['lib', '--root', temporary, '--arm', report.ARMS[1]])
            run = args.root / args.arm
            run.mkdir()
            lock = dict(arm=args.arm, checkpoint_sha256='checkpoint', state_hash='protocol')
            test = dict(identity=dict(model_state_sha256='weights'))
            with patch.object(report, 'locked_inputs', return_value=(run, {}, lock, test, {})), \
                    patch.object(report, 'inference_timing', return_value=dict(model_state_sha256='weights')), \
                    patch.object(report, 'routing_receipt', return_value=dict(model_state_sha256='weights')), \
                    patch.object(report.previous, 'validate_nss_reference', return_value={}), \
                    patch.object(report.common, 'r80_lib', return_value=dict(TSPLIB={}, CVRPLIB={})) as lib:
                report.lib_stage(args)
            self.assertEqual(lib.call_args.args[:2], (args.root, run))
            self.assertEqual(lib.call_args.kwargs['factory_module'], 'code.V4.r105_experiment')

    def test_captured_val_binding_and_null_train_require_exact_proof(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            state, lock, val, final = capture_fixture(run)
            official = report.runner.bind_selected_val(run, final)['val']
            self.assertEqual(report.capture_val(official, lock)['identity'], val['identity'])
            self.assertIsNone(val['identity']['checkpoint_sha256'])
            broken = copy.deepcopy(official)
            broken['checkpoint_binding']['selected_update'] += 1
            with self.assertRaises(ValueError):
                report.capture_val(broken, lock)
            _, lock, train, _ = capture_fixture(run, 'train')
            write_json(run / 'best_eval.json', dict(successful_updates=1440, train=train))
            before = (Path(train['prediction_archive']).parent / 'identity.json').read_bytes()
            result, _ = report.selected_train(run, state['args'], lock)
            self.assertIsNone(result['identity']['checkpoint_sha256'])
            self.assertEqual((Path(train['prediction_archive']).parent / 'identity.json').read_bytes(), before)
            train['identity']['model_state_sha256'] = 'wrong'
            write_json(run / 'best_eval.json', dict(successful_updates=1440, train=train))
            write_json(Path(train['prediction_archive']).parent / 'identity.json', train['identity'])
            with self.assertRaises(ValueError):
                report.selected_train(run, state['args'], lock)

    def test_report_locked_inputs_calls_actual_per_arm_factory(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary) / report.ARMS[0]
            state, lock, _, final = capture_fixture(run)
            official = report.runner.bind_selected_val(run, final)['val']
            config = dict(state['args'], model_factory=report.runner.MODEL_FACTORY)
            test = dict(identity=dict(model_state_sha256=official['identity']['model_state_sha256']))
            model = SimpleNamespace(state_dict=lambda: state['model'])
            with patch.object(report.shared, 'locked_inputs', return_value=(run, config, lock, test, official)), \
                    patch.object(report.runner, 'load_checkpoint', return_value=(model, dict(state, args=config))) as factory, \
                    patch.object(report.runner, 'selected_capture_proof', return_value=(official['archive_capture_identity'], official['checkpoint_binding'])):
                result = report.locked_inputs(run.parent, report.ARMS[0])
            factory.assert_called_once_with(run / 'best.pt', 'cpu')
            self.assertEqual(result[4]['identity'], official['archive_capture_identity'])

    def test_timing_metadata_all18_exact_denominator_and_arm(self):
        lock = dict(arm=report.ARMS[0], checkpoint_sha256='checkpoint', state_hash='protocol', selected_update=1440)
        value = timing_fixture(lock)
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            write_json(run / 'inference_timing.json', value)
            self.assertEqual(len(report.inference_timing(run, lock)['per_problem']), 18)
            for field, altered in (('donor_present', True), ('arm', report.ARMS[1]),
                    ('cached_feature_latency_used', True), ('end_to_end_measured', True),
                    ('device', 'cpu'), ('per_problem', value['per_problem'][:-1])):
                write_json(run / 'inference_timing.json', dict(value, **{field: altered}))
                with self.assertRaises(ValueError, msg=field):
                    report.inference_timing(run, lock)
            wrong = copy.deepcopy(value)
            wrong['per_problem'][0]['standalone_model']['mean_seconds_per_instance'] *= 2
            write_json(run / 'inference_timing.json', wrong)
            with self.assertRaises(ValueError):
                report.inference_timing(run, lock)

    def test_routing_metadata_first128_exact_n_entropy_and_singletons(self):
        lock = dict(arm=report.ARMS[0], checkpoint_sha256='checkpoint', state_hash='protocol', selected_update=1440)
        value = routing_fixture(lock)
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            write_json(run / 'routing_diagnostic.json', value)
            result = report.routing_receipt(run, lock)
            singletons = [r for r in result['per_problem'][0]['within_exact_n'] if r['exact_n'] == 5]
            self.assertTrue(all(r['instance_weight_std'] is None for r in singletons))
            self.assertEqual(len(report.routing_csv(result)), 18 * (4 + 8))
            for field, altered in (('selection_use', True), ('labels_read', True),
                    ('routing_calibration', True), ('split', 'test'), ('arm', report.ARMS[1])):
                write_json(run / 'routing_diagnostic.json', dict(value, **{field: altered}))
                with self.assertRaises(ValueError, msg=field):
                    report.routing_receipt(run, lock)
            broken = copy.deepcopy(value)
            broken['per_problem'][0]['within_exact_n'].pop()
            write_json(run / 'routing_diagnostic.json', broken)
            with self.assertRaises(ValueError):
                report.routing_receipt(run, lock)

    def test_true_recovery_false_picks_and_full_population_net_cost(self):
        costs = np.array([[1., 3., 4.], [4., 1., 3.], [3., 4., 1.], [1., 2., 3.]])
        def prediction(pred):
            return SimpleNamespace(n=4, pool=['x', 'y', 'z'], pool_ids=np.arange(3),
                indices=np.arange(4), source_indices=np.arange(4), costs=costs,
                winner=costs.argmin(1), pred=np.array(pred))
        old = {s: {p: prediction([1, 1, 0, 0]) for p in report.PROBLEMS} for s in ('TRAIN', 'VAL', 'TEST')}
        new = {s: {p: prediction([0, 0, 2, 1]) for p in report.PROBLEMS} for s in old}
        problem = report.PROBLEMS[0]
        identity = dict(identity=dict(rule='fixed historical target'), categories={problem + '/x': 'zero'})
        pairs, rows, totals = report.comparison_tables(old, new, [(problem, 'x')], identity, 'R105', 'old')
        self.assertEqual((rows[0]['rescued'], rows[0]['lost'], rows[0]['new_false_pick_count']), (1, 1, 1))
        self.assertAlmostEqual(rows[0]['new_false_pick_cost_delta_contribution'], .75)
        self.assertAlmostEqual(totals[0]['full18_net_mean_cost_delta'], 0.)
        self.assertEqual(len([r for r in pairs if r['problem'] == 'ALL']), 3)


if __name__ == '__main__':
    unittest.main()

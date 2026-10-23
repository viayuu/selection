"""Small CPU report/evidence checks; no model inference, GPU or real result search."""

import copy
import csv
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from . import r106_reporting as report
from .r76_experiment import write_json
from .test_r106_experiment import capture_fixture


def timing_fixture(lock):
    core, native = report.runner.CORE_PARAMETER_COUNT, report.runner.NATIVE_PARAMETER_COUNT
    frozen = lock['arm'] == report.ARMS[0]
    return dict(experiment='R106', arm=lock['arm'], complete=True,
        checkpoint_sha256=lock['checkpoint_sha256'], state_hash=lock['state_hash'],
        selected_update=lock['selected_update'], model_state_sha256='weights',
        model_factory=report.runner.MODEL_FACTORY, inference_rule=report.runner.INFERENCE_RULE,
        model_reselection=False, repetitions=3, warmup_batches=1,
        scope='online_native_plus_selector_GPU_resident_inputs', all_current_model_included=True,
        donor_present=True, cached_feature_latency_used=False, batch128_amortized=True,
        singleton_measured=False, end_to_end_measured=False, input_labels_read=False,
        sample_policy='first128 frozen TEST inputs per problem', device='cuda:0',
        core_parameter_count=core, native_parameter_count=native, total_parameter_count=core + native,
        trainable_parameter_count=core if frozen else core + native, native_trainable_parameter_count=0 if frozen else native,
        excludes='dataset I/O, CPU-to-GPU transfer, training and solver decoding/execution',
        per_problem=[dict(problem=p, instances=128, online_native_plus_selector=dict(
            batch_seconds=[.128] * 3, mean_seconds_per_instance=.001,
            donor_calls=[1, 1, 1], warmup_donor_calls=1, native_output_dtype=['torch.float32'])) for p in report.PROBLEMS])


def history_fixture():
    def metrics(cost):
        return dict(macro=dict(macro_ce=.5, macro_top1=.7, macro_mean_cost=2.,
            macro_actual_regret_pct=.4, macro_vs_sbs_pct=cost))
    history = []
    for update in range(1440, 57601, 1440):
        # Exact minimum ties at1440/2880: only the first is admissible.
        history.append(dict(successful_updates=update,
            optimization=dict(loss=.5, ce=.3, cmp=.2, risk=.1, kl=.01),
            val=metrics(-1. if update <= 2880 else -.5),
            train=metrics(-2.) if update == 1440 or update % 7200 == 0 else None))
    return history


class R106ReportingTests(unittest.TestCase):
    def test_cli_factory_and_private_historical_helper_isolation(self):
        for arm in report.ARMS:
            for command in ('lib', 'report', 'status'):
                args = report.parser().parse_args([command, '--arm', arm])
                self.assertEqual(args.root, report.DEFAULT_ROOT)
                self.assertEqual(args.r102_root, report.DEFAULT_R102B)
        self.assertIsNone(report.parser().parse_args(['compare']).arm)
        self.assertIn('preparation/postprocess_recovery/report_view/B', str(report.DEFAULT_R102B))
        for private, original in ((report.common, report.r80_reporting),
                (report.shared, report.r84_reporting), (report.previous, report.r85_reporting),
                (report.helpers, report.r87_reporting)):
            self.assertIsNot(private, original)
            self.assertEqual(private.FACTORY, 'code.V4.r106_experiment')
            self.assertNotEqual(getattr(original, 'FACTORY', None), private.FACTORY)
        self.assertIs(report.shared.learning_curves, report.learning_curves)
        self.assertIs(report.previous.selected_train, report.selected_train)

    def test_exact_native_inclusive_capture_and_old_nullable_VAL_are_not_missing(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary) / 'current'
            state, lock, val, final = capture_fixture(run)
            archive = Path(val['prediction_archive']).parent / 'identity.json'
            before = archive.read_bytes()
            bound = report.runner.bind_selected_val(run, final)['val']
            self.assertEqual(report.capture_val(bound, lock)['identity'], val['identity'])
            poisoned = copy.deepcopy(bound)
            poisoned['checkpoint_binding']['model_state_sha256'] = 'different native state'
            with self.assertRaises(ValueError):
                report.capture_val(poisoned, lock)
            self.assertEqual(archive.read_bytes(), before)
            run = Path(temporary) / 'historical'
            state, lock, val, _ = capture_fixture(run, experiment='R102')
            test = dict(identity=dict(model_state_sha256=val['identity']['model_state_sha256']))
            with patch.object(report.runner, 'make_model', side_effect=AssertionError('No historical model construction')):
                captured = report.historical_val(run, state['args'], lock, test, val)
            self.assertIsNone(captured['identity']['checkpoint_sha256'])
            corrupted = copy.deepcopy(val)
            corrupted['identity']['successful_updates'] = 2880
            write_json(Path(val['prediction_archive']).parent / 'identity.json', corrupted['identity'])
            with self.assertRaises(ValueError):
                report.historical_val(run, state['args'], lock, test, corrupted)

    def test_online_timing_all18_donor_hooks_denominator_and_trainability(self):
        for arm in report.ARMS:
            lock = dict(arm=arm, checkpoint_sha256='checkpoint', state_hash='protocol', selected_update=1440)
            value = timing_fixture(lock)
            with tempfile.TemporaryDirectory() as temporary:
                run = Path(temporary)
                write_json(run / 'inference_timing.json', value)
                self.assertEqual(len(report.inference_timing(run, lock)['per_problem']), 18)
                for field, wrong in (('donor_present', False), ('cached_feature_latency_used', True),
                        ('end_to_end_measured', True), ('singleton_measured', True), ('device', 'cpu'),
                        ('per_problem', value['per_problem'][:-1]), ('native_trainable_parameter_count', -1)):
                    write_json(run / 'inference_timing.json', dict(value, **{field: wrong}))
                    with self.assertRaises(ValueError, msg=field):
                        report.inference_timing(run, lock)
                for field, wrong in (('mean_seconds_per_instance', .002), ('donor_calls', [0, 0, 0]),
                        ('native_output_dtype', ['torch.float16'])):
                    corrupted = copy.deepcopy(value)
                    corrupted['per_problem'][0]['online_native_plus_selector'][field] = wrong
                    write_json(run / 'inference_timing.json', corrupted)
                    with self.assertRaises(ValueError, msg=field):
                        report.inference_timing(run, lock)

    def test_lib_native49_100_actual_factory_and_custom_root(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = report.parser().parse_args(['lib', '--root', temporary,
                '--arm', report.ARMS[1], '--device', 'cuda:0'])
            run = args.root / args.arm
            run.mkdir()
            lock = dict(arm=args.arm, checkpoint_sha256='checkpoint', state_hash='protocol', selected_update=1440)
            test = dict(identity=dict(model_state_sha256='weights'))
            results = {d: dict(train_sbs_reference=dict(n=n)) for d, n in report.LIB_COUNTS.items()}
            with patch.object(report, 'locked_inputs', return_value=(run, {}, lock, test, {})), \
                    patch.object(report.previous, 'validate_nss_reference', return_value={}), \
                    patch.object(report.common, 'r80_lib', return_value=results) as lib:
                report.lib_stage(args)
            self.assertEqual(lib.call_args.args[:2], (args.root, run))
            self.assertEqual(lib.call_args.kwargs['factory_module'], report.FACTORY)
            receipt = report._completed_lib(run, lock)
            self.assertEqual(receipt['native_counts'], report.LIB_COUNTS)
            write_json(run / 'lib_stage_complete.json', dict(receipt, native_feature_cache=True))
            with self.assertRaises(ValueError):
                report._completed_lib(run, lock)

    def test_nss4_explicit_release_and_candidate_bank_columns(self):
        rows = [dict(dataset=dataset, problem=problem) for dataset, problem in
            (('TSP TEST', 'TSP'), ('CVRP TEST', 'CVRP'), ('TSPLIB', 'TSP'), ('CVRPLIB', 'CVRP'))]
        config = dict(data_root='/synthetic/NEW90k', candidate_pools=dict(TSP=['x'], CVRP=['y']),
            candidate_pool_ids=dict(TSP=[0], CVRP=[1]))
        annotated = report.nss_comparison_rows(config, rows)
        self.assertEqual(len(annotated), 4)
        self.assertEqual(annotated[0]['data_version'], 'NEW90k frozen TEST4500')
        self.assertEqual(annotated[0]['candidate_pool'], '["x"]')
        self.assertIn('native LIB', annotated[2]['data_version'])
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'nss_comparison.csv'
            report.common.write_csv(path, annotated)
            with path.open(newline='') as stream:
                saved = list(csv.DictReader(stream))
            self.assertEqual(saved[0]['dataset'], 'TSP TEST')
        with self.assertRaises(ValueError):
            report.nss_comparison_rows(config, rows[:-1])

    def test_fixed_recovery_false_picks_and_aligned_equal_task_net_cost(self):
        costs = np.array([[1., 3., 4.], [4., 1., 3.], [3., 4., 1.], [1., 2., 3.]], dtype=np.float64)
        def prediction(pred):
            return SimpleNamespace(n=4, pool=['x', 'y', 'z'], pool_ids=np.arange(3),
                indices=np.arange(4), source_indices=np.arange(4), costs=costs,
                winner=costs.argmin(1), pred=np.array(pred))
        old = {s: {p: prediction([1, 1, 0, 0]) for p in report.PROBLEMS} for s in ('TRAIN', 'VAL', 'TEST')}
        new = {s: {p: prediction([0, 0, 2, 1]) for p in report.PROBLEMS} for s in old}
        problem = report.PROBLEMS[0]
        identity = dict(identity=dict(rule='fixed historical target'), categories={problem + '/x': 'zero'})
        pairs, rows, totals = report.comparison_tables(old, new, [(problem, 'x')], identity, 'R106', 'old')
        self.assertEqual((rows[0]['rescued'], rows[0]['lost'], rows[0]['new_false_pick_count']), (1, 1, 1))
        self.assertAlmostEqual(rows[0]['new_false_pick_cost_delta_contribution'], .75)
        self.assertAlmostEqual(rows[0]['new_false_pick_cost_contribution'], 1.)
        self.assertEqual(totals[0]['full18_net_mean_cost_delta'], 0.)
        self.assertEqual(totals[0]['new_false_pick_cost_delta_contribution'], .75 / 18)
        self.assertEqual(len([r for r in pairs if r['problem'] == 'ALL']), 3)

    def test_first_exact_VAL_tie_and_complete_learning_curve(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            output = run / 'full_model_statistics'
            output.mkdir()
            write_json(run / 'history.json', history_fixture())
            self.assertEqual(len(report.validated_history(run, dict(selected_update=1440))), 40)
            with self.assertRaises(ValueError):
                report.validated_history(run, dict(selected_update=2880))
            report.learning_curves(run, output, 'R106/test_fixture', dict(selected_update=1440))
            self.assertGreater((output / 'learning_curves.pdf').stat().st_size, 100)

    def test_same_axis_native_distribution_real_model_Top1_and_test_lib_order(self):
        import matplotlib.pyplot as plt
        report.render_environment()
        helper = report.common.reporting_helpers()
        helper.setup_style()
        costs = np.array([[1., 2.], [2., 1.], [1., 2.], [2., 1.]], dtype=np.float64)
        data = helper.Predictions(Path('fixture.npz'), ['x', 'y'], np.zeros(4, dtype=int), costs.argmin(1),
            costs, np.arange(4), np.arange(4), np.arange(2), np.zeros(4, dtype=bool))
        library = helper.Predictions(Path('library_fixture.npz'), data.pool, np.zeros(49, dtype=int),
            np.arange(49) % 2, np.ones((49, 2)), np.arange(49), np.arange(49), np.arange(2), None)
        panels = helper.make_panels('TSP', data, library, data.pool)
        lock = dict(checkpoint_sha256='f' * 64, epoch=1, selected_update=1440)
        fig = report.common._distribution(helper, 'TSP', data.pool, panels, 'synthetic NEW90k', 'R106/A_frozen', lock)
        try:
            titles = [axis.get_title() for axis in fig.axes]
            self.assertIn('TEST', titles[0])
            self.assertIn('TSPLIB', titles[1])
            self.assertIn('Top1=50.00%', titles[0])
            self.assertIn(f'Top1={library.top1_pct:.2f}%', titles[1])
            self.assertEqual([len(axis.patches) for axis in fig.axes], [4, 4])
        finally:
            plt.close(fig)


if __name__ == '__main__':
    unittest.main()

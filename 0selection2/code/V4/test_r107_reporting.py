"""Minimal synthetic CPU contracts; no data, real checkpoints, inference or PDFs."""

import copy
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import FunctionType, SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import torch

from . import r107_reporting as report
from .r76_experiment import write_json


REAL_RUNNER = report._runner


def model_lock(arm=None):
    return dict(experiment='R107', arm=arm or report.ARMS[0],
        checkpoint_sha256='synthetic-checkpoint', state_hash='synthetic-protocol',
        selected_update=1440, epoch=1)


def official_val(lock, rule, model_hash='synthetic-weights'):
    capture = dict(experiment='R107', arm=lock['arm'], split='val',
        checkpoint_sha256=None, state_hash=lock['state_hash'],
        successful_updates=lock['selected_update'], model_state_sha256=model_hash)
    binding = dict(rule=rule, checkpoint_sha256=lock['checkpoint_sha256'],
        state_hash=lock['state_hash'], selected_update=lock['selected_update'],
        archive_identity_unchanged=True, archive_capture_checkpoint_sha256=None,
        model_state_sha256=model_hash)
    return dict(identity=dict(capture, checkpoint_sha256=lock['checkpoint_sha256']),
        archive_capture_identity=capture, checkpoint_binding=binding)


def timing_fixture(lock, runner):
    return dict(synthetic_mock_only=True, experiment='R107', arm=lock['arm'], complete=True,
        checkpoint_sha256=lock['checkpoint_sha256'], state_hash=lock['state_hash'],
        selected_update=lock['selected_update'], model_state_sha256='synthetic-weights',
        model_factory=runner.MODEL_FACTORY, inference_rule=runner.INFERENCE_RULE,
        model_reselection=False, routing_calibration=False, repetitions=3, warmup_batches=1,
        scope='complete_standalone_model_original_GPU_resident_inputs',
        all_current_model_included=True, explicit_path_computation_included=True,
        donor_present=False, cached_feature_latency_used=False, end_to_end_measured=False,
        input_labels_read=False, sample_policy='first128 frozen TEST inputs per problem',
        parameter_count=runner.PARAMETER_COUNTS[lock['arm']], device='cuda:0',
        excludes='dataset I/O, CPU-to-GPU transfer, training and solver execution',
        per_problem=[dict(problem=p, instances=128, standalone_model=dict(
            batch_seconds=[.128] * 3, mean_seconds_per_instance=.001)) for p in report.PROBLEMS])


def initialization_fixture(count=1):
    return dict(common_fresh_model_state_sha256='synthetic-fresh-state',
        cross_arm_complete_state_verified=True, trainable_parameter_count=count)


class R107ReportingTests(unittest.TestCase):
    def setUp(self):
        self.runner = SimpleNamespace(ARMS=report.ARMS, BUDGETS=report.BUDGETS,
            MODEL_FACTORY=report.FACTORY + '.load_checkpoint',
            PARAMETER_COUNTS=dict.fromkeys(report.ARMS, 1),
            VAL_BINDING_RULE='synthetic-exact-state-binding', INFERENCE_RULE='synthetic-maximin',
            load_checkpoint=Mock(), selected_capture_proof=Mock(return_value=({}, {'synthetic_proof': True})))
        self.runner_patch = patch.object(report, '_runner', return_value=self.runner)
        self.runner_patch.start()
        self.addCleanup(self.runner_patch.stop)

    def test_cli_and_private_module_globals_do_not_mutate_history(self):
        for arm in report.ARMS:
            for command in ('lib', 'report', 'status'):
                args = report.parser().parse_args([command, '--arm', arm])
                self.assertEqual(args.root, report.DEFAULT_ROOT)
                self.assertEqual(args.lib_batch_size, 2)
                self.assertIsNone(args.r94b_root)
        self.assertIsNone(report.parser().parse_args(['compare']).arm)
        for private, original in ((report.common, report.r80_reporting),
                (report.shared, report.r84_reporting), (report.previous, report.r85_reporting),
                (report.helpers, report.r87_reporting)):
            self.assertIsNot(private, original)
            self.assertEqual(private.FACTORY, report.FACTORY)
            self.assertNotEqual(original.DEFAULT_ROOT, private.DEFAULT_ROOT)
            self.assertNotEqual(getattr(original, 'FACTORY', None), private.FACTORY)
        self.assertIs(report.shared.build_reports.__globals__, report.shared.__dict__)
        self.assertIs(report.shared.common, report.common)
        self.assertIs(report.helpers.previous, report.previous)
        self.assertIs(report.previous.selected_train, report.selected_train)
        self.assertIsNot(report.shared.write_json, report.r84_reporting.write_json)
        self.assertFalse(hasattr(report, 'routing_receipt'))
        self.assertFalse(hasattr(report, 'routing_csv'))

    def test_lazy_factory_is_real_r107_only_and_status_does_not_import_it(self):
        with patch.object(report.importlib, 'import_module', return_value=self.runner) as factory:
            self.assertIs(REAL_RUNNER(), self.runner)
            factory.assert_called_once_with('code.V4.r107_experiment')
            self.runner.ARMS = ('old-A', 'old-B')
            with self.assertRaises(ValueError):
                REAL_RUNNER()
        with tempfile.TemporaryDirectory() as temporary, \
                patch.object(report.importlib, 'import_module', side_effect=AssertionError('no factory')), \
                patch('builtins.print'), patch.object(report, 'write_json') as writer:
            value = report.main(['status', '--root', temporary, '--arm', report.ARMS[0]])
        self.assertEqual(value['phase'], 'not_started')
        writer.assert_not_called()

    def test_val_binding_preserves_capture_and_rejects_wrong_state_update_or_arm(self):
        lock = model_lock()
        val = official_val(lock, self.runner.VAL_BINDING_RULE)
        before = copy.deepcopy(val)
        self.assertEqual(report.capture_val(val, lock)['identity'], val['archive_capture_identity'])
        self.assertEqual(val, before)
        for field, value in (('selected_update', 2880), ('model_state_sha256', 'wrong'),
                ('archive_identity_unchanged', False), ('checkpoint_sha256', 'wrong')):
            bad = copy.deepcopy(val)
            bad['checkpoint_binding'][field] = value
            with self.assertRaises(ValueError, msg=field):
                report.capture_val(bad, lock)
        bad = copy.deepcopy(val)
        bad['archive_capture_identity']['arm'] = report.ARMS[1]
        with self.assertRaises(ValueError):
            report.capture_val(bad, lock)

    def test_selected_train_null_sha_requires_main_proof_without_archive_rewrite(self):
        lock = model_lock()
        config = dict(experiment='R107', arm=lock['arm'])
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            path = run / 'evaluation/train/synthetic/rank000.npz'
            identity = dict(official_val(lock, self.runner.VAL_BINDING_RULE)['archive_capture_identity'], split='train')
            write_json(path.parent / 'identity.json', identity)
            summary = dict(identity=identity, prediction_archive=str(path))
            write_json(run / 'best_eval.json', dict(successful_updates=1440, train=summary))
            before = (path.parent / 'identity.json').read_bytes()
            self.assertIsNone(report.selected_train(run, config, lock)[0]['identity']['checkpoint_sha256'])
            self.runner.selected_capture_proof.assert_called_once_with(run, summary, lock, 'train')
            self.assertEqual((path.parent / 'identity.json').read_bytes(), before)
            self.runner.selected_capture_proof.side_effect = ValueError('different selected model')
            with self.assertRaises(ValueError):
                report.selected_train(run, config, lock)

    def test_locked_factory_model_hash_and_actual_parameter_count(self):
        model = torch.nn.Linear(1, 1, bias=False)
        lock = model_lock()
        config = dict(experiment='R107', arm=lock['arm'], model_factory=self.runner.MODEL_FACTORY,
            trainable_parameter_count=1)
        state = dict(args=config, model=model.state_dict(), successful_updates=1440, epoch=1)
        weights = report.parameter_hash(model.state_dict())
        val = official_val(lock, self.runner.VAL_BINDING_RULE, weights)
        test = dict(identity=dict(model_state_sha256=weights))
        self.runner.load_checkpoint.return_value = (model, state)
        self.runner.selected_capture_proof.return_value = (val['archive_capture_identity'], val['checkpoint_binding'])
        run = Path('/synthetic-only') / lock['arm']
        with patch.object(report.shared, 'locked_inputs', return_value=(run, config, lock, test, val)):
            self.assertIsNone(report.locked_inputs(run.parent, lock['arm'])[4]['identity']['checkpoint_sha256'])
            self.runner.load_checkpoint.assert_called_once_with(run / 'best.pt', 'cpu')
            config['trainable_parameter_count'] = 2
            with self.assertRaises(ValueError):
                report.locked_inputs(run.parent, lock['arm'])
            config['trainable_parameter_count'] = 1
            test['identity']['model_state_sha256'] = 'other-checkpoint'
            with self.assertRaises(ValueError):
                report.locked_inputs(run.parent, lock['arm'])

    def test_timing_complete_model_paths_all18_denominator_and_no_donor(self):
        lock = model_lock()
        value = timing_fixture(lock, self.runner)
        with patch.object(report, 'read_json', return_value=value):
            self.assertEqual(len(report.inference_timing('/synthetic-only', lock)['per_problem']), 18)
        for field, altered in (('donor_present', True), ('arm', report.ARMS[1]),
                ('explicit_path_computation_included', False), ('all_current_model_included', False),
                ('cached_feature_latency_used', True), ('end_to_end_measured', True),
                ('input_labels_read', True), ('model_reselection', True), ('parameter_count', 2),
                ('device', 'cpu'), ('per_problem', value['per_problem'][:-1]),
                ('per_problem', [value['per_problem'][0]] * 18)):
            with patch.object(report, 'read_json', return_value=dict(value, **{field: altered})):
                with self.assertRaises(ValueError, msg=field):
                    report.inference_timing('/synthetic-only', lock)
        for changed in (dict(mean_seconds_per_instance=.002), dict(batch_seconds=[.128, 0., .128]),
                dict(batch_seconds=[.128, float('nan'), .128]), dict(batch_seconds=['.128'] * 3)):
            broken = copy.deepcopy(value)
            broken['per_problem'][0]['standalone_model'].update(changed)
            with patch.object(report, 'read_json', return_value=broken), self.assertRaises(ValueError):
                report.inference_timing('/synthetic-only', lock)

    def test_lib_explicit_factory_root_same_state_and_no_routing_receipt(self):
        lock = model_lock(report.ARMS[1])
        test = dict(identity=dict(model_state_sha256='synthetic-weights'))
        with tempfile.TemporaryDirectory() as temporary:
            args = report.parser().parse_args(['lib', '--root', temporary, '--arm', lock['arm'], '--device', 'cuda:0'])
            run = args.root / args.arm
            run.mkdir()
            with patch.object(report, 'locked_inputs', return_value=(run, {}, lock, test, {})), \
                    patch.object(report, 'inference_timing', return_value=dict(model_state_sha256='synthetic-weights')), \
                    patch.object(report.previous, 'validate_nss_reference', return_value={}), \
                    patch.object(report.common, 'r80_lib', return_value=dict(TSPLIB={}, CVRPLIB={})) as lib, \
                    patch.object(report, 'write_json') as writer:
                report.lib_stage(args)
            self.assertEqual(lib.call_args.args[:2], (args.root, run))
            self.assertEqual(lib.call_args.kwargs['factory_module'], report.FACTORY)
            receipt = next(c.args[1] for c in writer.call_args_list if c.args[0].name == 'lib_stage_complete.json')
            self.assertEqual(receipt['selected_update'], lock['selected_update'])
            self.assertEqual(receipt['model_state_sha256'], 'synthetic-weights')
            self.assertNotIn('routing_diagnostic', receipt)
            with patch.object(report, 'locked_inputs', return_value=(run, {}, lock, test, {})), \
                    patch.object(report, 'inference_timing', return_value=dict(model_state_sha256='wrong')), \
                    patch.object(report.common, 'r80_lib') as wrong_lib:
                with self.assertRaisesRegex(ValueError, 'same locked TEST model'):
                    report.lib_stage(args)
            wrong_lib.assert_not_called()

    def test_optional_r94_absence_and_invalid_predictions_do_not_block_base(self):
        refs = {name: dict(run=Path('/synthetic-only'), config={}, lock={}, val={}, test={},
                provenance=dict(model=name)) for name in ('R89A', 'R89B', 'R84A')}
        for mode in ('not_requested', 'missing', 'invalid_predictions', 'included'):
            args = report.parser().parse_args(['report', '--arm', report.ARMS[0]])
            args.r94b_root = None if mode == 'not_requested' else Path('/synthetic-only/R94/B')
            def paired(*values):
                if mode == 'invalid_predictions' and values[-1] == 'R94B':
                    raise AssertionError('source indices differ')
                return [], [], []
            with patch.object(report, 'selected_predictions', return_value=({}, [], [], {}, {})), \
                    patch.object(report, 'references', return_value=copy.deepcopy(refs)), \
                    patch.object(report, 'load_targets', return_value=([], {})), \
                    patch.object(report, 'comparison_tables', side_effect=paired), \
                    patch.object(report, 'historical_reference', return_value=refs['R84A'],
                        side_effect=FileNotFoundError('missing optional R94B') if mode == 'missing' else None) as optional:
                value = report.diagnostic_tables(Path('/synthetic-only'), dict(arm=args.arm), {}, {}, {}, args)
            self.assertTrue(set(refs).issubset(value['historical']))
            self.assertEqual('R94B' in value['historical'], mode == 'included')
            status = value['optional_historical']['R94B']
            self.assertEqual(status['status'], mode if mode in ('included', 'not_requested') else 'unavailable')
            if mode == 'not_requested':
                optional.assert_not_called()
            elif mode != 'included':
                self.assertIn('error', status)

    def test_fixed36_identity_rejects_target_reselection(self):
        targets = [('synthetic-problem', str(i)) for i in range(36)]
        with patch.object(report.helpers, 'load_targets', return_value=(targets, dict(path='/synthetic-only'))), \
                patch.object(report, 'file_hash', return_value='synthetic-target-hash'):
            self.assertEqual(report.load_targets({}, {}, {})[1]['target_count'], 36)
        with patch.object(report.helpers, 'load_targets', return_value=(targets[:-1], {})), self.assertRaises(ValueError):
            report.load_targets({}, {}, {})

    def test_rescues_false_picks_cost_regret_and_equal_problem_denominator(self):
        costs = np.array([[1., 3., 4.], [4., 1., 3.], [3., 4., 1.], [1., 2., 3.]])
        def prediction(pred):
            return SimpleNamespace(n=4, pool=['x', 'y', 'z'], pool_ids=np.arange(3),
                indices=np.arange(4), source_indices=np.arange(4), costs=costs,
                winner=costs.argmin(1), pred=np.array(pred))
        old = {s: {p: prediction([1, 1, 0, 0]) for p in report.PROBLEMS} for s in ('TRAIN', 'VAL', 'TEST')}
        new = {s: {p: prediction([0, 0, 2, 1]) for p in report.PROBLEMS} for s in old}
        problem = report.PROBLEMS[0]
        identity = dict(identity=dict(rule='fixed synthetic historical target'), categories={problem + '/x': 'zero'})
        pairs, rows, totals = report.comparison_tables(old, new, [(problem, 'x')], identity, 'R107', 'old')
        self.assertEqual((rows[0]['rescued'], rows[0]['lost'], rows[0]['new_false_pick_count']), (1, 1, 1))
        self.assertAlmostEqual(rows[0]['new_false_pick_cost_delta_contribution'], .75)
        self.assertAlmostEqual(rows[0]['removed_false_pick_cost_delta_contribution'], -.5)
        self.assertAlmostEqual(rows[0]['target_winner_cost_delta_contribution'], -.25)
        self.assertAlmostEqual(totals[0]['new_false_pick_cost_delta_contribution'], .75 / 18)
        self.assertAlmostEqual(totals[0]['full18_net_mean_cost_delta'], 0.)
        self.assertEqual(len([r for r in pairs if r['problem'] == 'ALL']), 3)
        old['TEST'][problem].source_indices = np.array([9, 1, 2, 3])
        with self.assertRaises(AssertionError):
            report.comparison_tables(old, new, [(problem, 'x')], identity, 'R107', 'old')

    def test_retained_nss_four_rows_bind_the_same_current_checkpoint(self):
        config = dict(experiment='R107', arm=report.ARMS[0], data_root='/synthetic-only/data',
            rows_per_problem=dict(test=4))
        plan = dict(data_root=config['data_root'], retrained_on_new90k=False,
            checkpoints={p: dict(sha256='synthetic-nss') for p in ('TSP', 'CVRP')})
        data = SimpleNamespace(n=4, top1_pct=50., mean_cost=2., path=Path('/synthetic-only/current.npz'))
        predictions = {(s, p): data for s, p in (('TEST', 'TSP'), ('TEST', 'CVRP'), ('TSPLIB', 'TSP'), ('CVRPLIB', 'CVRP'))}
        helper = SimpleNamespace(load_predictions=Mock(return_value=data), check_comparison_alignment=Mock(return_value={}))
        metadata = dict(checkpoint=dict(sha256='synthetic-nss'), top1_native_ind=.5, mean_cost=2.)
        with patch.object(report.common, 'reporting_helpers', return_value=helper), \
                patch.object(report.shared, 'read_json', side_effect=[plan, *[metadata] * 4]), \
                patch.object(report.shared.np, 'load', return_value=MagicMock()):
            rows, alignment = report.shared.nss_comparisons(config, model_lock(), predictions, Path('/synthetic-only/nss'))
        self.assertEqual([r['dataset'] for r in rows], ['TSP TEST', 'CVRP TEST', 'TSPLIB', 'CVRPLIB'])
        self.assertEqual(len(alignment), 4)
        self.assertEqual({r['current_checkpoint_sha256'] for r in rows}, {'synthetic-checkpoint'})

    def test_combined_distribution_axes_colors_order_and_real_model_top1(self):
        helper = report.common.reporting_helpers()
        test = SimpleNamespace(n=4, pool=['y', 'x'], pred=np.array([0, 0, 0, 0]),
            winner=np.array([1, 1, 1, 0]), top1_pct=25.)
        n = helper.LIBS['TSP'][1]
        lib = SimpleNamespace(n=n, pool=['x', 'y'], pred=np.zeros(n, dtype=int),
            winner=np.zeros(n, dtype=int), top1_pct=100.)
        panels = helper.make_panels('TSP', test, lib, ['x', 'y'])
        choices = next(p for p in panels if p['split'] == 'TEST' and p['kind'] == 'model_selection')
        winners = next(p for p in panels if p['split'] == 'TEST' and p['kind'] == 'true_winner')
        np.testing.assert_array_equal(choices['counts'], [0, 4])
        np.testing.assert_array_equal(winners['counts'], [3, 1])
        self.assertEqual(choices['top1'], 25.)
        self.assertIsNone(winners['top1'])
        fig, axes = MagicMock(), np.empty(2, dtype=object)
        axes[0], axes[1] = MagicMock(), MagicMock()
        fig.texts = []
        for axis in axes:
            axis.get_legend_handles_labels.return_value = ([], [])
        plotting = SimpleNamespace(subplots=Mock(return_value=(fig, axes)))
        plot = FunctionType(helper.plot_distribution.__code__,
            dict(helper.plot_distribution.__globals__, plt=plotting))
        adapter = SimpleNamespace(plot_distribution=plot, SHA256=helper.SHA256)
        report.common._distribution(adapter, 'TSP', ['x', 'y'], panels, 'synthetic-data', 'R107/A', model_lock())
        self.assertIn('TSP TEST', axes[0].set_title.call_args.args[0])
        self.assertIn('25.00%', axes[0].set_title.call_args.args[0])
        self.assertIn('TSPLIB', axes[1].set_title.call_args.args[0])
        for axis in axes:
            self.assertEqual(axis.barh.call_count, 2)
            self.assertEqual(len({c.kwargs['color'] for c in axis.barh.call_args_list}), 2)
        plotting.subplots.assert_called_once()

    def test_real_small_plot_format_in_memory_no_pdf_or_result_artifacts(self):
        helper = report.common.reporting_helpers()
        helper.setup_style()
        test = SimpleNamespace(n=4, pool=['y', 'x'], pred=np.array([0, 0, 0, 0]),
            winner=np.array([1, 1, 1, 0]), top1_pct=25.)
        for problem, dataset in (('TSP', 'TSPLIB'), ('CVRP', 'CVRPLIB')):
            n = helper.LIBS[problem][1]
            lib = SimpleNamespace(n=n, pool=['x', 'y'], pred=np.zeros(n, dtype=int),
                winner=np.zeros(n, dtype=int), top1_pct=100.)
            panels = helper.make_panels(problem, test, lib, ['x', 'y'])
            fig = report.common._distribution(helper, problem, ['x', 'y'], panels,
                'SYNTHETIC FORMAT CHECK ONLY', 'R107/SYNTHETIC', model_lock())
            try:
                fig.canvas.draw()
                self.assertEqual(len(fig.axes), 2)
                self.assertIn(problem + ' TEST', fig.axes[0].get_title())
                self.assertIn(helper.MODEL_TOP1 + '=25.00%', fig.axes[0].get_title())
                self.assertIn(dataset, fig.axes[1].get_title())
                for axis in fig.axes:
                    self.assertEqual([t.get_text() for t in axis.get_yticklabels()], ['x', 'y'])
                    self.assertEqual(len(axis.patches), 4)
                    self.assertEqual(len({p.get_facecolor() for p in axis.patches}), 2)
                np.testing.assert_allclose([p.get_width() for p in fig.axes[0].patches], [0., 100., 75., 25.])
                self.assertTrue(fig.legends)
                self.assertTrue(any('native/FP64 ind' in t.get_text() for t in fig.texts))
                self.assertFalse(any('R77B' in t.get_text() for t in fig.texts))
            finally:
                helper.plt.close(fig)

    def test_complete_initialization_counts_not_equal_active_freedom_or_flops(self):
        config = dict(trainable_parameter_count=1)
        for change in ({}, dict(trainable_parameter_count=2), dict(cross_arm_complete_state_verified=False),
                dict(common_fresh_model_state_sha256=None)):
            with patch.object(report, 'read_json', return_value=dict(initialization_fixture(), **change)):
                if change:
                    with self.assertRaises(ValueError):
                        report.initialization_evidence('/synthetic-only', config)
                else:
                    self.assertEqual(report.initialization_evidence('/synthetic-only', config)['trainable_parameter_count'], 1)
        self.assertIn('not global selector additivity', report.BOUNDARY)
        self.assertIn('not imply equal active functional freedom or equal FLOPs', report.BOUNDARY)

    def test_paired_compare_requires_same_complete_initialization_and_parameter_count(self):
        for difference in ('none', 'state', 'count'):
            with tempfile.TemporaryDirectory() as temporary, ExitStack() as stack:
                args = report.parser().parse_args(['compare', '--root', temporary])
                configs = {a: {k: 'synthetic-same-data' for k in ('data_root', 'rows_per_problem',
                    'candidate_pools', 'candidate_pool_ids', 'baselines', 'winner_rule')} for a in report.ARMS}
                init = {a: initialization_fixture() for a in report.ARMS}
                if difference == 'state':
                    init[report.ARMS[1]]['common_fresh_model_state_sha256'] = 'another-fresh-state'
                if difference == 'count':
                    init[report.ARMS[1]]['trainable_parameter_count'] = 2
                stack.enter_context(patch.object(report, 'locked_inputs', side_effect=lambda root, arm:
                    (root / arm, configs[arm], model_lock(arm), {}, {})))
                stack.enter_context(patch.object(report, 'selected_predictions', return_value=({}, [], [], {}, {})))
                stack.enter_context(patch.object(report, 'initialization_evidence', side_effect=lambda run, config: init[run.name]))
                stack.enter_context(patch.object(report, 'references', return_value=dict(R84A=dict(config={}, lock={}))))
                stack.enter_context(patch.object(report, 'load_targets', return_value=([], {'synthetic': True})))
                stack.enter_context(patch.object(report, 'comparison_tables', return_value=([], [], [])))
                stack.enter_context(patch.object(report.common, 'write_csv'))
                writer = stack.enter_context(patch.object(report, 'write_json'))
                if difference != 'none':
                    with self.assertRaisesRegex(ValueError, 'complete fresh tensor equality'):
                        report.compare(args)
                    writer.assert_not_called()
                else:
                    result = report.compare(args)
                    self.assertTrue(result['total_parameter_matched'])
                    self.assertFalse(result['active_parameter_matched'])
                    self.assertFalse(result['equal_flops_claim'])
                    self.assertFalse(result['inference'])
                    self.assertEqual(set(result['checkpoints']), set(report.ARMS))

    def test_report_saved_cpu_pipeline_contains_train_and_publishes_only_after_pdfs(self):
        self._report_pipeline(missing_pdf=False)
        self._report_pipeline(missing_pdf=True)

    def _report_pipeline(self, missing_pdf):
        lock = model_lock()
        config = dict(arm=lock['arm'], model_spec={'synthetic': True}, path_spec={'synthetic': True},
            trainable_parameter_count=1)
        test = dict(identity=dict(model_state_sha256='synthetic-weights'))
        val = official_val(lock, self.runner.VAL_BINDING_RULE)
        tables = dict(methods=[], summaries=[], historical={}, optional_historical=dict(R94B=dict(status='not_requested')),
            target_identity={}, results=dict(TRAIN={}), pairs={}, recovery={}, recovery_summary={})
        lib = dict(experiment='R107', arm=lock['arm'], complete=True, checkpoint_sha256=lock['checkpoint_sha256'],
            state_hash=lock['state_hash'], selected_update=lock['selected_update'],
            model_state_sha256='synthetic-weights', model_factory=self.runner.MODEL_FACTORY, datasets=list(report.LIBS))
        def read(path):
            return {'lib_stage_complete.json': lib, 'protocol.json': dict(complete=False),
                'final_metrics.json': dict(val=val)}[path.name]
        with tempfile.TemporaryDirectory() as temporary, ExitStack() as stack:
            args = report.parser().parse_args(['report', '--root', temporary, '--arm', lock['arm']])
            run = args.root / args.arm
            run.mkdir()
            stack.enter_context(patch.object(report, 'locked_inputs', return_value=(run, config, lock, test, val)))
            stack.enter_context(patch.object(report, 'read_json', side_effect=read))
            writer = stack.enter_context(patch.object(report, 'write_json'))
            stack.enter_context(patch.object(report, 'inference_timing', return_value=timing_fixture(lock, self.runner)))
            stack.enter_context(patch.object(report.previous, 'validate_nss_reference', return_value={}))
            stack.enter_context(patch.object(report.shared, 'load_saved_splits', return_value=({}, [dict(split='VAL')], [])))
            stack.enter_context(patch.object(report.shared, 'saved_libs'))
            stack.enter_context(patch.object(report.shared, 'nss_comparisons', return_value=([], {})))
            stack.enter_context(patch.object(report, 'diagnostic_tables', return_value=tables))
            stack.enter_context(patch.object(report, 'initialization_evidence', return_value=initialization_fixture()))
            stack.enter_context(patch.object(report.helpers, 'train_metric_rows', return_value=[dict(split='TRAIN')]))
            build = stack.enter_context(patch.object(report.shared, 'build_reports', return_value=dict(complete=True)))
            stack.enter_context(patch.object(report.common, 'write_csv'))
            stack.enter_context(patch.object(report, 'diagnostics_pdf', return_value='synthetic-only-not-rendered'))
            stack.enter_context(patch.object(report.Path, 'is_file', autospec=True,
                side_effect=lambda path: not missing_pdf or path.name != report.REQUIRED_PDFS[0]))
            stack.enter_context(patch.object(report.Path, 'stat', return_value=SimpleNamespace(st_size=1)))
            if missing_pdf:
                with self.assertRaisesRegex(ValueError, 'required PDF missing'):
                    report.report(args)
            else:
                self.assertTrue(report.report(args)['complete'])
            metrics = build.call_args.args[6]
            self.assertEqual([r['split'] for r in metrics], ['TRAIN', 'VAL'])
            written = [c.args[1] for c in writer.call_args_list]
            self.assertEqual(written[-1]['phase'], 'postprocess_failed' if missing_pdf else 'complete')
            if missing_pdf:
                self.assertFalse(any(v.get('complete') for v in written))

    def test_render_metadata_never_claims_complete_before_full_verification(self):
        with patch.object(report, 'write_json') as writer:
            report.shared.write_json(Path('/synthetic-only/protocol.json'), dict(complete=True))
        self.assertFalse(writer.call_args.args[1]['complete'])

    def test_report_rejects_gpu_before_any_artifact_or_factory_access(self):
        args = report.parser().parse_args(['report', '--arm', report.ARMS[0], '--device', 'cuda:0'])
        with patch.object(report, 'locked_inputs') as load, self.assertRaises(ValueError):
            report.report(args)
        load.assert_not_called()


if __name__ == '__main__':
    unittest.main()

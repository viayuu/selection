"""Synthetic CPU contracts/rendering; no production data, cache or GPU access."""

import copy
import re
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import FunctionType, SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import torch

from . import r109_reporting as report
from .r76_experiment import write_json


REAL_RUNNER = report._runner


def model_lock(arm=None):
    return dict(experiment='R109', arm=arm or report.ARMS[0],
        checkpoint_sha256='synthetic-checkpoint', state_hash='synthetic-protocol',
        selected_update=1440, epoch=1)


def official_val(lock, rule, model_hash='synthetic-weights'):
    capture = dict(experiment='R109', arm=lock['arm'], split='val',
        checkpoint_sha256=None, state_hash=lock['state_hash'],
        successful_updates=lock['selected_update'], model_state_sha256=model_hash)
    binding = dict(rule=rule, checkpoint_sha256=lock['checkpoint_sha256'],
        state_hash=lock['state_hash'], selected_update=lock['selected_update'],
        archive_identity_unchanged=True, archive_capture_checkpoint_sha256=None,
        model_state_sha256=model_hash)
    return dict(identity=dict(capture, checkpoint_sha256=lock['checkpoint_sha256']),
        archive_capture_identity=capture, checkpoint_binding=binding)


def timing_fixture(lock, runner):
    count = runner.CAPS['eval_batch_size']
    return dict(synthetic_mock_only=True, experiment='R109', arm=lock['arm'], complete=True,
        checkpoint_sha256=lock['checkpoint_sha256'], state_hash=lock['state_hash'],
        selected_update=lock['selected_update'], model_state_sha256='synthetic-weights',
        model_factory=runner.MODEL_FACTORY, inference_rule=runner.INFERENCE_RULE,
        model_reselection=False, routing_calibration=False, repetitions=3, warmup_batches=1,
        scope='complete_standalone_model_original_GPU_resident_inputs',
        all_current_model_included=True, method_conditioned_node_refinement_included=True,
        donor_present=False, cached_feature_latency_used=False, end_to_end_measured=False,
        input_labels_read=False, sample_policy=f'first{count} frozen TEST inputs per problem',
        parameter_count=runner.PARAMETER_COUNTS[lock['arm']], device='cuda:0',
        excludes='dataset I/O, CPU-to-GPU transfer, training and solver execution',
        per_problem=[dict(problem=p, instances=count, standalone_model=dict(
            batch_seconds=[count * .001] * 3, mean_seconds_per_instance=.001)) for p in report.PROBLEMS])


def initialization_fixture(arm=None, runner=None):
    arm = arm or report.ARMS[0]
    count = runner.PARAMETER_COUNTS[arm] if runner else 1
    original = runner.PARAMETER_COUNTS['original'] if runner else 1
    fresh_hash = runner.ORIGINAL_FRESH_SHA256 if runner else 'synthetic-common-base'
    return dict(common_parameter_sha256=fresh_hash, r84_fresh_trainable_sha256=fresh_hash,
        common_parameter_count=original, original_parameter_count=original,
        parameter_count=count, trainable_parameter_count=count, new_parameter_count=count - original,
        seed=2, branch_init_seed=runner.BRANCH_INIT_SEED if runner else 109002,
        fresh_initialization=True, historical_checkpoint_loaded=False, state_dict_restored=False,
        identical_complete_fresh_parameters=False, total_parameter_matched=False,
        equal_total_parameters=False, equal_flops_claim=False,
        fresh_state_sha256='synthetic-fresh-state-' + arm,
        fresh_trainable_sha256='synthetic-fresh-trainables-' + arm)


class R109ReportingTests(unittest.TestCase):
    def setUp(self):
        self.runner = SimpleNamespace(ARMS=report.ARMS, BUDGETS=report.BUDGETS,
            MODEL_FACTORY=report.FACTORY + '.load_checkpoint',
            PARAMETER_COUNTS={'original': 1, report.ARMS[0]: 1, report.ARMS[1]: 2},
            CAPS=dict(eval_batch_size=128), BRANCH_INIT_SEED=109002,
            ORIGINAL_FRESH_SHA256='synthetic-common-base',
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

    def test_lazy_factory_is_real_r109_only_and_status_does_not_import_it(self):
        with patch.object(report.importlib, 'import_module', return_value=self.runner) as factory:
            self.assertIs(REAL_RUNNER(), self.runner)
            factory.assert_called_once_with('code.V4.r109_experiment')
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
        config = dict(experiment='R109', arm=lock['arm'])
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
        config = dict(experiment='R109', arm=lock['arm'], model_factory=self.runner.MODEL_FACTORY,
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

    def test_timing_complete_model_refinement_all18_denominator_and_no_donor(self):
        lock = model_lock()
        value = timing_fixture(lock, self.runner)
        with patch.object(report, 'read_json', return_value=value):
            self.assertEqual(len(report.inference_timing('/synthetic-only', lock)['per_problem']), 18)
        for field, altered in (('donor_present', True), ('arm', report.ARMS[1]),
                ('method_conditioned_node_refinement_included', False), ('all_current_model_included', False),
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
        pairs, rows, totals = report.comparison_tables(old, new, [(problem, 'x')], identity, 'R109', 'old')
        self.assertEqual((rows[0]['rescued'], rows[0]['lost'], rows[0]['new_false_pick_count']), (1, 1, 1))
        self.assertAlmostEqual(rows[0]['new_false_pick_cost_delta_contribution'], .75)
        self.assertAlmostEqual(rows[0]['removed_false_pick_cost_delta_contribution'], -.5)
        self.assertAlmostEqual(rows[0]['target_winner_cost_delta_contribution'], -.25)
        self.assertAlmostEqual(totals[0]['new_false_pick_cost_delta_contribution'], .75 / 18)
        self.assertAlmostEqual(totals[0]['full18_net_mean_cost_delta'], 0.)
        self.assertEqual(len([r for r in pairs if r['problem'] == 'ALL']), 3)
        old['TEST'][problem].source_indices = np.array([9, 1, 2, 3])
        with self.assertRaises(AssertionError):
            report.comparison_tables(old, new, [(problem, 'x')], identity, 'R109', 'old')

    def test_retained_nss_four_rows_bind_the_same_current_checkpoint(self):
        config = dict(experiment='R109', arm=report.ARMS[0], data_root='/synthetic-only/data',
            rows_per_problem=dict(test=4))
        plan = dict(data_root=config['data_root'], retrained_on_new90k=False,
            checkpoints={p: dict(sha256='synthetic-nss') for p in ('TSP', 'CVRP')})
        data = SimpleNamespace(n=4, pool=['x', 'y'], top1_pct=50., mean_cost=2.,
            path=Path('/synthetic-only/current.npz'))
        predictions = {(s, p): data for s, p in (('TEST', 'TSP'), ('TEST', 'CVRP'), ('TSPLIB', 'TSP'), ('CVRPLIB', 'CVRP'))}
        helper = SimpleNamespace(load_predictions=Mock(return_value=data), check_comparison_alignment=Mock(return_value={}))
        metadata = dict(checkpoint=dict(sha256='synthetic-nss'), top1_native_ind=.5, mean_cost=2.)
        with patch.object(report.common, 'reporting_helpers', return_value=helper), \
                patch.object(report.shared, 'read_json', side_effect=[plan, *[metadata] * 4]), \
                patch.object(report.shared, 'np', SimpleNamespace(
                    load=Mock(return_value=MagicMock()), testing=np.testing)):
            rows, alignment = report.shared.nss_comparisons(config, model_lock(), predictions, Path('/synthetic-only/nss'))
        self.assertEqual([r['dataset'] for r in rows], ['TSP TEST', 'CVRP TEST', 'TSPLIB', 'CVRPLIB'])
        self.assertEqual(len(alignment), 4)
        self.assertEqual({r['current_checkpoint_sha256'] for r in rows}, {'synthetic-checkpoint'})
        self.assertEqual([r['data_version'] for r in rows], ['data', 'data',
            'TSPLIB original native LIB / historical R79 NSS comparison',
            'CVRPLIB original native LIB / historical R79 NSS comparison'])
        self.assertEqual({r['candidate_pool'] for r in rows}, {'["x", "y"]'})

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
        report.common._distribution(adapter, 'TSP', ['x', 'y'], panels, 'synthetic-data', 'R109/A', model_lock())
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
                'SYNTHETIC FORMAT CHECK ONLY', 'R109/SYNTHETIC', model_lock())
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

    def test_initialization_matches_original_base_but_not_complete_state_or_counts(self):
        config = dict(arm=report.ARMS[0], trainable_parameter_count=1)
        for change in ({}, dict(trainable_parameter_count=2),
                dict(identical_complete_fresh_parameters=True), dict(total_parameter_matched=True),
                dict(equal_total_parameters=True), dict(equal_flops_claim=True),
                dict(common_parameter_sha256='wrong'), dict(branch_init_seed=1),
                dict(historical_checkpoint_loaded=True), dict(fresh_initialization=False),
                dict(original_parameter_count=2), dict(fresh_state_sha256=None)):
            with patch.object(report, 'read_json',
                    return_value=dict(initialization_fixture(runner=self.runner), **change)):
                if change:
                    with self.assertRaises(ValueError, msg=str(change)):
                        report.initialization_evidence('/synthetic-only', config)
                else:
                    value = report.initialization_evidence('/synthetic-only', config)
                    self.assertEqual(value['common_parameter_sha256'], self.runner.ORIGINAL_FRESH_SHA256)
        self.assertIn('not identical', report.BOUNDARY)
        self.assertIn('not parameter/FLOP-matched', report.BOUNDARY)

    def test_paired_compare_matches_base_and_prototype_without_whole_state_equality(self):
        for difference in ('none', 'base', 'prototype', 'equal_count', 'equal_state'):
            with tempfile.TemporaryDirectory() as temporary, ExitStack() as stack:
                args = report.parser().parse_args(['compare', '--root', temporary])
                configs = {a: {k: 'synthetic-same-data' for k in ('data_root', 'rows_per_problem',
                    'candidate_pools', 'candidate_pool_ids', 'baselines', 'winner_rule')} for a in report.ARMS}
                init = {a: initialization_fixture(a, self.runner) for a in report.ARMS}
                b = init[report.ARMS[1]]
                if difference == 'base':
                    b['common_parameter_sha256'] = 'another-fresh-base'
                elif difference == 'prototype':
                    b['branch_init_seed'] += 1
                elif difference == 'equal_count':
                    b['trainable_parameter_count'] = 1
                elif difference == 'equal_state':
                    b['fresh_state_sha256'] = init[report.ARMS[0]]['fresh_state_sha256']
                stack.enter_context(patch.object(report, 'locked_inputs', side_effect=lambda root, arm:
                    (root / arm, configs[arm], model_lock(arm), {}, {})))
                stack.enter_context(patch.object(report, 'selected_predictions', return_value=({}, [], [], {}, {})))
                stack.enter_context(patch.object(report, 'initialization_evidence', side_effect=lambda run, config: init[run.name]))
                stack.enter_context(patch.object(report, 'completed_report_metrics', return_value={}))
                stack.enter_context(patch.object(report, 'references', return_value=dict(R84A=dict(config={}, lock={}))))
                stack.enter_context(patch.object(report, 'load_targets', return_value=([], {'synthetic': True})))
                stack.enter_context(patch.object(report, 'comparison_tables', return_value=([], [], [])))
                stack.enter_context(patch.object(report.common, 'write_csv'))
                delivery = dict(comparison_pdf=str(args.root / 'MODEL_COMPARISON.pdf'),
                    results_markdown=str(args.root / 'RESULTS.md'))
                stack.enter_context(patch.object(report, 'comparison_reports', return_value=delivery))
                stack.enter_context(patch.object(report.Path, 'is_file', return_value=True))
                stack.enter_context(patch.object(report.Path, 'stat', return_value=SimpleNamespace(st_size=1)))
                writer = stack.enter_context(patch.object(report, 'write_json'))
                if difference != 'none':
                    with self.assertRaisesRegex(ValueError, 'matched original base/prototypes'):
                        report.compare(args)
                    writer.assert_not_called()
                else:
                    result = report.compare(args)
                    self.assertFalse(result['total_parameter_matched'])
                    self.assertFalse(result['identical_complete_fresh_parameters'])
                    self.assertTrue(result['identical_late_prototype_initialization'])
                    self.assertFalse(result['active_parameter_matched'])
                    self.assertFalse(result['equal_flops_claim'])
                    self.assertFalse(result['inference'])
                    self.assertEqual(result['trainable_parameters'], {report.ARMS[0]: 1, report.ARMS[1]: 2})
                    self.assertEqual(set(result['checkpoints']), set(report.ARMS))

    def test_report_saved_cpu_pipeline_contains_train_and_publishes_only_after_pdfs(self):
        self._report_pipeline(missing_pdf=False)
        self._report_pipeline(missing_pdf=True)

    def _report_pipeline(self, missing_pdf):
        lock = model_lock()
        config = dict(arm=lock['arm'], model_spec={'synthetic': True}, branch_init_seed=109002,
            trainable_parameter_count=1)
        test = dict(identity=dict(model_state_sha256='synthetic-weights'))
        val = official_val(lock, self.runner.VAL_BINDING_RULE)
        tables = dict(methods=[], summaries=[], historical={}, optional_historical=dict(R94B=dict(status='not_requested')),
            target_identity={}, results=dict(TRAIN={}), pairs={}, recovery={}, recovery_summary={})
        lib = dict(experiment='R109', arm=lock['arm'], complete=True, checkpoint_sha256=lock['checkpoint_sha256'],
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
            stack.enter_context(patch.object(report, 'validate_nss_csv', return_value=dict(rows=4)))
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

    def test_csv_delivery_has_real_nss_metadata_and_tolerates_bom_on_read(self):
        lock = model_lock()
        config = dict(data_root='/synthetic-only/new90k_release')
        predictions = {(s, p): SimpleNamespace(pool=['x', 'y']) for _, p, s in report.NSS_DATASETS}
        rows = [dict(dataset=d, current_checkpoint_sha256=lock['checkpoint_sha256'],
            nss_checkpoint_sha256='historical-R79', status='complete',
            current_top1_native_pct=25., current_mean_cost=2., nss_top1_native_pct=30., nss_mean_cost=3.,
            **report._nss_metadata(config, predictions, d, p, s)) for d, p, s in report.NSS_DATASETS]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'nss_comparison.csv'
            report.write_csv(path, rows)
            self.assertFalse(path.read_bytes().startswith(b'\xef\xbb\xbf'))
            self.assertEqual(report.read_csv(path)[0]['dataset'], 'TSP TEST')
            self.assertEqual(report.validate_nss_csv(path, config, lock, predictions)['rows'], 4)
            with path.open('r', encoding='utf-8') as stream:
                self.assertEqual(len(list(report.csv.DictReader(stream))), 4)
            content = path.read_text(encoding='utf-8')
            path.write_text(content, encoding='utf-8-sig')
            self.assertTrue(report.validate_nss_csv(path, config, lock, predictions)['same_current_checkpoint'])
            for changed in (dict(data_version=''), dict(candidate_pool='["removed"]'),
                    dict(current_checkpoint_sha256='other'), dict(current_top1_native_pct=float('nan')),
                    dict(status='missing'), dict(winner_rule='FP32 argmin')):
                broken = copy.deepcopy(rows)
                broken[0].update(changed)
                report.write_csv(path, broken)
                with self.assertRaises(ValueError, msg=str(changed)):
                    report.validate_nss_csv(path, config, lock, predictions)
            for field in ('data_version', 'candidate_pool'):
                broken = [{k: v for k, v in row.items() if k != field} for row in rows]
                report.write_csv(path, broken)
                with self.assertRaises(ValueError):
                    report.validate_nss_csv(path, config, lock, predictions)
            self.assertIsNot(report.common.write_csv, report.r80_reporting.write_csv)

    def test_timing_follows_main_locked_eval_count_not_a_reporter_constant(self):
        lock = model_lock()
        for count in (64, 128):
            self.runner.CAPS['eval_batch_size'] = count
            value = timing_fixture(lock, self.runner)
            with patch.object(report, 'read_json', return_value=value):
                self.assertEqual(report.inference_timing('/synthetic-only', lock)['per_problem'][0]['instances'], count)
            bad = dict(value, sample_policy=f'first{count * 2} frozen TEST inputs per problem')
            with patch.object(report, 'read_json', return_value=bad), self.assertRaises(ValueError):
                report.inference_timing('/synthetic-only', lock)
        old_paths_only = timing_fixture(lock, self.runner)
        old_paths_only.pop('method_conditioned_node_refinement_included')
        old_paths_only['explicit_path_computation_included'] = True
        with patch.object(report, 'read_json', return_value=old_paths_only), self.assertRaises(ValueError):
            report.inference_timing('/synthetic-only', lock)

    def test_actual_runner_selected_capture_binds_exact_state_without_archive_rewrite(self):
        runner = REAL_RUNNER()
        config = dict(experiment='R109', arm=report.ARMS[0])
        state = dict(args=config, state_hash=runner.json_hash(config), successful_updates=1440,
            model={'synthetic_weight': torch.tensor([2.])})
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            torch.save(state, run / 'best.pt')
            lock = dict(model_lock(), state_hash=state['state_hash'],
                checkpoint_sha256=report.file_hash(run / 'best.pt'))
            val = official_val(lock, runner.VAL_BINDING_RULE, report.parameter_hash(state['model']))
            archive = run / 'evaluation/val/synthetic/rank000.npz'
            val['prediction_archive'] = str(archive)
            write_json(archive.parent / 'identity.json', val['archive_capture_identity'])
            before = (archive.parent / 'identity.json').read_bytes()
            capture, proof = runner.selected_capture_proof(run, val, lock, 'val', state)
            self.assertEqual(proof, val['checkpoint_binding'])
            self.assertIsNone(capture['checkpoint_sha256'])
            self.assertEqual((archive.parent / 'identity.json').read_bytes(), before)
            bad = copy.deepcopy(val)
            bad['archive_capture_identity']['successful_updates'] += 1
            with self.assertRaises(ValueError):
                runner.selected_capture_proof(run, bad, lock, 'val', state)

    def test_actual_runner_input_only_timing_never_forwards_labels_routes_or_context(self):
        runner = REAL_RUNNER()
        mask = torch.ones(2, 3, dtype=torch.bool)
        batch = dict(node_mask=mask, node=torch.zeros(2, 3, 2), pool_ids=torch.tensor([0, 1]),
            costs=object(), ind=object(), performance_target=object(), winner=object(),
            routes=object(), path_fields=object(), donor_features=object(), context_labels=object())
        seen = []
        def model(raw):
            seen.append(set(raw))
            return dict(logits=torch.zeros(2, 2), solver_mask=torch.ones(2, 2, dtype=torch.bool))
        value = runner.measure_inputs(model, batch, torch.device('cpu'))
        self.assertEqual(seen, [{'node_mask', 'node', 'pool_ids'}] * 4)
        self.assertEqual(len(value['batch_seconds']), 3)
        self.assertTrue(all(s > 0 for s in value['batch_seconds']))

    def test_actual_runner_input_cache_reader_opens_only_existing_input_arrays(self):
        runner = REAL_RUNNER()
        config = dict(cache_root='/synthetic-only/cache', data_root='/synthetic-only/data',
            eval_batch_size=2, candidate_pools=dict(TSP=['x', 'y']), candidate_pool_ids=dict(TSP=[0, 1]))
        record = dict(problem='TSP', split='test', rows=4500, data_root=config['data_root'],
            pool=['x', 'y'], pool_ids=[0, 1], kind='coord', problem_id=0, cbits=[0.] * 5,
            problem_desc=[0.] * 4, mask=[1.] * 2, d_in=2)
        arrays = dict(n=np.array([3, 4]), coord_dist=np.array([0, 0]),
            node=np.zeros((2, 4, 2), dtype=np.float32), node_geom=np.zeros((2, 4, 12), dtype=np.float32))
        specs = dict.fromkeys((*arrays, 'costs', 'ind', 'winner', 'performance_target'), object())
        with patch.object(report.Path, 'is_dir', return_value=True), \
                patch.object(runner, 'read_json', return_value=record), \
                patch('code.V4.r76_cache._validate_record', return_value=specs), \
                patch('code.V4.r75_cache._open_arrays', return_value=arrays) as opened, \
                patch('code.V4.r75_cache._close') as closed:
            value = runner.cached_input_batch(config, dict(splits=dict(test=dict(TSP=record))),
                'test', 'TSP', torch.device('cpu'))
        self.assertEqual(set(opened.call_args.args[1]), {'n', 'coord_dist', 'node', 'node_geom'})
        self.assertFalse({'costs', 'ind', 'winner', 'performance_target'} & value.keys())
        closed.assert_called_once_with(arrays)

    def _completed_metrics_fixture(self, run, arm):
        config = dict(arm=arm, trainable_parameter_count=self.runner.PARAMETER_COUNTS[arm])
        lock = model_lock(arm)
        rows = []
        for split, problem in [*((s, p) for s in ('TRAIN', 'VAL', 'TEST')
                for p in ('ALL', *report.PROBLEMS)), *report.LIBS.items()]:
            rows.append(dict(split=split, problem=problem, checkpoint_sha256=lock['checkpoint_sha256'],
                top1_pct=25. if arm == report.ARMS[0] else 50., mean_cost=2., sbs_cost=3., oracle_cost=1.,
                vs_sbs_pct=-100. / 3., vs_oracle_pct=100., actual_regret_pct=100.))
        output = run / 'full_model_statistics'
        output.mkdir(parents=True)
        write_json(output / 'protocol.json', dict(experiment='R109', arm=arm, complete=True,
            checkpoint_lock=lock, model_factory=self.runner.MODEL_FACTORY))
        for name in report.REQUIRED_PDFS:
            (output / name).write_bytes(b'%PDF-SYNTHETIC-COMPLETION-FIXTURE')
        report.write_csv(output / 'metrics_by_problem.csv', rows)
        return config, lock, rows

    def test_root_comparison_requires_completed_arm_and_full_same_lock_metrics(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary) / report.ARMS[0]
            config, lock, rows = self._completed_metrics_fixture(run, report.ARMS[0])
            self.assertEqual(len(report.completed_report_metrics(run, config, lock)), 59)
            for change in ('missing_train', 'other_checkpoint', 'duplicate'):
                broken = copy.deepcopy(rows)
                if change == 'missing_train':
                    broken = [r for r in broken if r['split'] != 'TRAIN']
                elif change == 'other_checkpoint':
                    broken[0]['checkpoint_sha256'] = 'other'
                else:
                    broken.append(broken[0])
                report.write_csv(run / 'full_model_statistics/metrics_by_problem.csv', broken)
                with self.assertRaises(ValueError, msg=change):
                    report.completed_report_metrics(run, config, lock)
            report.write_csv(run / 'full_model_statistics/metrics_by_problem.csv', rows)
            (run / 'full_model_statistics' / report.REQUIRED_PDFS[0]).unlink()
            with self.assertRaisesRegex(ValueError, 'completed per-arm PDF'):
                report.completed_report_metrics(run, config, lock)

    def test_synthetic_root_comparison_renders_five_pages_and_markdown_on_cpu(self):
        records, metrics = {}, {}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for arm in report.ARMS:
                config, lock, _ = self._completed_metrics_fixture(root / arm, arm)
                records[arm] = (config, lock)
                metrics[arm] = report.completed_report_metrics(root / arm, config, lock)
            pairs = [dict(split=s, problem='ALL', mean_cost_delta=0., actual_regret_delta_pp=0.)
                     for s in ('TRAIN', 'VAL', 'TEST')]
            totals = [dict(split=s, winner_count=4, correct_count=2, rescued=2, lost=1,
                new_false_pick_count=1, new_false_pick_cost_delta_contribution=.3,
                removed_false_pick_cost_delta_contribution=-.1,
                target_winner_cost_delta_contribution=-.2, full18_net_mean_cost_delta=0.)
                for s in ('TRAIN', 'VAL', 'TEST')]
            delivery = report.comparison_reports(root, records, metrics, pairs, totals, dict(sha256='synthetic-target36'))
            pdf = Path(delivery['comparison_pdf']).read_bytes()
            self.assertTrue(pdf.startswith(b'%PDF'))
            self.assertEqual(len(re.findall(rb'/Type /Page\b', pdf)), 5)
            text = Path(delivery['results_markdown']).read_text(encoding='utf-8')
            self.assertIn('NOT parameter/FLOP-matched', text)
            for split in ('TRAIN', 'VAL', 'TEST'):
                self.assertIn('| ' + split + ' | A |', text)
                self.assertIn('| ' + split + ' | B |', text)
                self.assertIn('| ' + split + ' | 4 | 25.000000 | 50.000000 |', text)
            self.assertFalse(list(root.glob('.*.tmp')))


if __name__ == '__main__':
    unittest.main()


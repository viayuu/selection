"""Small synthetic CPU report contracts; no actual checkpoints, labels or GPU."""

import copy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from . import r99_reporting as reporting
from .r76_experiment import read_json, write_json
from .test_r99_experiment import synthetic_config


def lock(arm='A'):
    return dict(arm=arm, checkpoint_sha256=arm * 64, state_hash='state-' + arm,
        inference_rule='locked maximin', epoch=1, selected_update=1440)


def timing_record(arm='A'):
    measurement = dict(batch_seconds=[.1, .2, .3], mean_seconds_per_instance=.6 / (3 * 128))
    return dict(experiment='R99', arm=arm, variant=reporting.VARIANTS[arm], complete=True,
        checkpoint_sha256=arm * 64, state_hash='state-' + arm,
        repetitions=3, warmup_batches=1, device='cuda:0', inference_rule='locked maximin',
        includes_input_transform=True, raw_input_source='original_base_cache_without_relation_fields',
        online_transform_verified=True, online_fp16_matches_cache=True, model_reselection=False,
        full_system_measured=False, end_to_end_measured=False,
        scope='first128 GPU-resident TEST input rows, synchronized wall time', excludes='I/O and host transfer',
        preprocessing=dict(experiment='R99', arm=arm, variant=reporting.VARIANTS[arm],
            complete=True, splits=['train', 'val'], seconds=12.),
        test_preprocessing=dict(experiment='R99', arm=arm, variant=reporting.VARIANTS[arm],
            complete=True, splits=['test'], seconds=3., checkpoint_sha256=arm * 64, state_hash='state-' + arm),
        per_problem=[dict(problem=p, instances=128, max_marginal_residual=.02,
            **{scope: copy.deepcopy(measurement) for scope in reporting.TIMING_SCOPES}) for p in reporting.PROBLEMS])


class R99ReportingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.temporary = tempfile.TemporaryDirectory()
        cls.configs = {arm: synthetic_config(cls.temporary.name, arm)[1] for arm in reporting.ARMS}

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def setUp(self):
        self.original_modules = {}
        for name in ('shared', 'previous', 'common', 'helpers', 'r93'):
            module = getattr(reporting, name)
            self.original_modules[name] = (module, dict(vars(module)))
            replacement = patch.object(reporting, name, SimpleNamespace(**vars(module)))
            replacement.start()
            self.addCleanup(replacement.stop)
        guard = patch('torch.cuda._lazy_init', side_effect=AssertionError('CPU report tests must not touch CUDA'))
        guard.start()
        self.addCleanup(guard.stop)

    def config(self, arm='A'):
        return copy.deepcopy(self.configs[arm])

    def test_cli_factory_titles_and_evidence_boundary(self):
        self.assertEqual(reporting.MODEL_FACTORY, 'code.V4.r99_experiment.load_checkpoint')
        self.assertEqual(reporting.parser().parse_args(['report']).root.name, 'R99_global_competition_input')
        for stage in ('lib', 'report', 'compare', 'status'):
            args = reporting.parser().parse_args([stage, '--arm', 'B', '--device', 'cpu', '--lib-batch-size', '2'])
            self.assertEqual(args.arm, 'B')
        for phrase in ('row-softmax', 'fixed64 Sinkhorn', 'no pretrained donor', 'CE=.35',
                'comparison=.35', 'risk=.02', 'R-Drop=.35', 'target36', 'Development-exposed TEST',
                'cache-hit selector is not full inference', 'directed matrix ATSP'):
            self.assertIn(phrase, reporting.REFERENCE_BOUNDARY)

    def test_real_shaped_configs_validate_without_constructing_factory(self):
        a, b = self.config(), self.config('B')
        original = copy.deepcopy((a, b))
        with patch.object(reporting.importlib, 'import_module') as factory:
            receipt = reporting.validate_matched_configs(a, b)
        factory.assert_not_called()
        self.assertTrue(receipt['matched'])
        self.assertEqual(receipt['relation_initialization_sha256'], a['relation_initialization_sha256'])
        self.assertEqual((a, b), original)
        b['gpu_type'] = '4090'
        b['feature_cache_root'] = '/different-runtime-location'
        reporting.same_data(a, b, matched=True)
        hardware = reporting.hardware_disclosure(dict(A=a, B=b))
        self.assertFalse(hardware['same_reported_hardware'])
        self.assertFalse(hardware['bitwise_hardware_matched'])

    def test_matched_config_rejects_all_control_drifts_and_unknown_fields(self):
        changes = (('relation_initialization_sha256', 'trained-state'), ('new_parameter_sha256', 'different'),
            ('common_trainable_sha256', 'different'), ('preparation_sha256', 'different'),
            ('task_order_sha256', 'different'), ('global_batch_size', 512), ('logical_global_batch', 512),
            ('warmup_updates', 1), ('max_successful_updates', 57599), ('eval_batch_size', 128),
            ('weight_decay', 0.), ('dropout', 0.), ('rdrop_weight', 0.), ('projection_seed', 2),
            ('loss', dict(ce=1., comparison=0., risk=0.)), ('objective', dict(new_loss=True)),
            ('relation_variant', 'row'), ('train_eval_every_updates', 1440),
            ('microbatch_caps', dict(TSP=352)), ('checkpoint_rule', 'TEST-best'),
            ('inference_rule', 'new-temperature'), ('external_pretraining', True), ('new_core_control', True))
        for key, value in changes:
            b = self.config('B')
            b[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                reporting.validate_matched_configs(self.config(), b)
        b = self.config('B')
        b['model_params']['new_head'] = True
        with self.assertRaises(ValueError):
            reporting.validate_matched_configs(self.config(), b)

    def test_schema_rejects_pool_first_transform_grid_and_new_loss(self):
        for key, value in (('width', 64), ('iterations', 128), ('temperature', .1), ('labels_used', True)):
            config = self.config()
            config['relation_transform'][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                reporting.validate_report_config(config)
        config = self.config()
        config['model_spec']['fusion_location'] = 'pool-first'
        with self.assertRaises(ValueError):
            reporting.validate_report_config(config)

    def test_late_bound_factory_validation_never_loads_a_model(self):
        record = (Path('/model/A'), self.config(), {}, {}, {})
        with patch.object(reporting.shared, 'locked_inputs', return_value=record) as load, \
                patch.object(reporting.importlib, 'import_module') as factory:
            self.assertEqual(reporting.locked_inputs(Path('/model'), 'A'), record)
            load.assert_called_once_with(Path('/model'), 'A', experiment='R99', arms=reporting.BUDGETS)
            factory.return_value.validate_checkpoint_config.assert_called_once_with(record[1])
            factory.return_value.load_checkpoint.assert_not_called()
        with self.assertRaises(ValueError):
            reporting.locked_inputs(Path('/model'), 'C')

    def test_historical_data_alignment_does_not_require_matched_training(self):
        a, old = self.config(), self.config('B')
        old.update(experiment='R84', initialization='historical', dropout=0.)
        reporting.same_data(a, old)
        for key in ('data_root', 'rows_per_problem', 'candidate_pools', 'candidate_pool_ids', 'winner_rule', 'baselines'):
            bad = copy.deepcopy(old)
            bad.pop(key)
            with self.subTest(key=key), self.assertRaises(ValueError):
                reporting.same_data(a, bad)

    def test_fixed_target36_hash_no_reselection(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'targets.json'
            write_json(path, dict(synthetic=True))
            identity = dict(path=str(path))
            targets = [('TSP', str(i)) for i in range(36)]
            with patch.object(reporting.helpers, 'load_targets', return_value=(targets, identity)):
                _, result = reporting.load_targets({}, {}, {})
            self.assertEqual(result['target_count'], 36)
            self.assertEqual(result['sha256'], reporting.file_hash(path))
            with patch.object(reporting.helpers, 'load_targets', return_value=(targets[:-1], identity)), self.assertRaises(ValueError):
                reporting.load_targets({}, {}, {})

    def test_saved_state_guard_and_model_labels(self):
        val = dict(identity=dict(model_state_sha256='same'))
        test = copy.deepcopy(val)
        with patch.object(reporting.helpers, 'selected_predictions', return_value=(
                {}, [dict(model='R87/A')], [dict(model='R87/A')], {}, {})):
            result = reporting.selected_predictions(None, self.config('B'), {}, val, test)
        self.assertEqual(result[1][0]['model'], 'R99/B')
        test['identity']['model_state_sha256'] = 'different'
        with self.assertRaises(ValueError):
            reporting.selected_predictions(None, self.config(), {}, val, test)

    def test_val_screen_descriptive_never_selects_or_uses_nan(self):
        metric = dict(macro_top1=.7, macro_mean_cost=10., macro_actual_regret_pct=.4)
        refs = {name: dict(val=dict(macro=dict(metric))) for name in ('R89B', 'R84A')}
        val = dict(macro=dict(metric, macro_top1=.71, macro_mean_cost=9.9, macro_actual_regret_pct=.39))
        rows = reporting.val_screen(self.config(), lock(), val, refs)
        self.assertTrue(all(row['model'] == 'R99/A' and row['joint_val_pass'] and not row['checkpoint_reselection'] for row in rows))
        val['macro']['macro_top1'] = float('nan')
        with self.assertRaises(ValueError):
            reporting.val_screen(self.config(), lock(), val, refs)

    def test_gpu_report_and_nonallocated_lib_rejected_before_data_reads(self):
        with patch.object(reporting, 'locked_inputs') as load:
            for fn in (reporting.report, reporting.compare):
                with self.assertRaises(ValueError):
                    fn(SimpleNamespace(device='cuda:0'))
            for device in ('cpu', 'cuda', 'mps'):
                with self.assertRaises(ValueError):
                    reporting.lib_stage(SimpleNamespace(device=device))
            load.assert_not_called()

    def test_lib_uses_r99_factory_and_preserves_failure_without_retry(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            args = reporting.parser().parse_args(['lib', '--arm', 'A', '--device', 'cuda:0'])
            with patch.object(reporting, 'locked_inputs', return_value=(run, self.config(), lock(),
                    dict(identity=dict(model_state_sha256='model-state')), {})), \
                    patch.object(reporting.previous, 'validate_nss_reference', return_value={}), \
                    patch.object(reporting.common, 'r80_lib', return_value={'TSPLIB': {}, 'CVRPLIB': {}}) as lib, \
                    patch.object(reporting.shared, 'build_reports') as pdf:
                reporting.lib_stage(args)
            self.assertEqual(lib.call_args.kwargs['factory_module'], 'code.V4.r99_experiment')
            pdf.assert_not_called()
            self.assertEqual(read_json(run / 'lib_stage_complete.json')['state_hash'], lock()['state_hash'])
            with patch.object(reporting, 'locked_inputs', return_value=(run, self.config(), lock(),
                    dict(identity=dict(model_state_sha256='model-state')), {})), \
                    patch.object(reporting.previous, 'validate_nss_reference', return_value={}), \
                    patch.object(reporting.common, 'r80_lib', side_effect=RuntimeError('synthetic')) as lib:
                with self.assertRaises(RuntimeError):
                    reporting.lib_stage(args)
            self.assertEqual(lib.call_count, 1)
            self.assertFalse(read_json(run / 'lib_stage_status.json')['automatic_retry'])

    def test_timing_requires_online_transform_full18_residuals_and_separate_preprocessing(self):
        with tempfile.TemporaryDirectory() as tmp:
            run, record = Path(tmp), timing_record()
            write_json(run / 'inference_timing.json', record)
            result = reporting.inference_timing(run, lock())
            self.assertEqual(len(result['measurements']), 18)
            self.assertTrue(result['transform_plus_selector_measured'])
            self.assertFalse(result['end_to_end_measured'])
            self.assertEqual(result['preprocessing_seconds'], 12.)
            self.assertEqual(result['test_preprocessing_seconds'], 3.)
            changes = (('online_transform_verified', False), ('online_fp16_matches_cache', False),
                ('raw_input_source', 'feature_wrapped_cache'), ('checkpoint_sha256', 'other'),
                ('state_hash', 'other'), ('repetitions', 1), ('full_system_measured', True),
                ('preprocessing', {}), ('test_preprocessing', {}), ('per_problem', record['per_problem'][:-1]))
            for key, value in changes:
                write_json(run / 'inference_timing.json', dict(record, **{key: value}))
                with self.subTest(key=key), self.assertRaises(ValueError):
                    reporting.inference_timing(run, lock())
            for residual in (float('nan'), -1., True):
                bad = copy.deepcopy(record)
                bad['per_problem'][0]['max_marginal_residual'] = residual
                # JSON writer rejects NaN too; test the consumer independently.
                with patch.object(reporting, 'read_json', return_value=bad), self.assertRaises(ValueError):
                    reporting.inference_timing(run, lock())
            bad = copy.deepcopy(record)
            bad['per_problem'][0]['transform_plus_selector']['mean_seconds_per_instance'] *= 2
            write_json(run / 'inference_timing.json', bad)
            with self.assertRaises(ValueError):
                reporting.inference_timing(run, lock())

    def test_report_missing_timing_fails_before_render_and_records_noncompletion(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            args = reporting.parser().parse_args(['report', '--arm', 'A'])
            with patch.object(reporting, 'locked_inputs', return_value=(run, self.config(), lock(), {}, {})), \
                    patch.object(reporting, '_completed_lib'), \
                    patch.object(reporting.shared, 'build_reports') as render:
                with self.assertRaises(FileNotFoundError):
                    reporting.report(args)
            render.assert_not_called()
            self.assertEqual(read_json(run / 'postprocess_status.json')['exit_status'], 1)

    def test_compare_requires_both_completed_arm_reports_before_saved_comparison(self):
        def inputs(root, arm):
            return Path('/synthetic') / arm, self.config(arm), lock(arm), {}, {}
        with patch.object(reporting, 'locked_inputs', side_effect=inputs), \
                patch.object(reporting, 'completed_arm_report', side_effect=ValueError('not complete')), \
                patch.object(reporting, 'selected_predictions') as selected:
            with self.assertRaises(ValueError):
                reporting.compare(SimpleNamespace(device='cpu', root=Path('/synthetic')))
        selected.assert_not_called()

    def test_native_grouped_bars_top1_and_test_before_lib(self):
        import matplotlib.pyplot as plt
        helper = reporting.common.reporting_helpers()
        helper.setup_style()
        pred = helper.Predictions(Path('/synthetic'), ['m0', 'm1'], np.array([0, 0, 1, 1]),
            np.array([0, 1, 1, 0]), np.ones((4, 2), dtype=np.float64), np.arange(4), None, np.arange(2), None)
        for problem, dataset in (('TSP', 'TSPLIB'), ('CVRP', 'CVRPLIB')):
            count = 49 if problem == 'TSP' else 100
            lib = helper.Predictions(Path('/synthetic/lib'), pred.pool, np.zeros(count, dtype=int),
                np.arange(count) % 2, np.ones((count, 2)), np.arange(count), None, np.arange(2), None)
            panels = helper.make_panels(problem, pred, lib, pred.pool)
            fig = reporting.common._distribution(helper, problem, pred.pool, panels, 'NEW90k', 'R99/A', lock())
            try:
                self.assertEqual(len(fig.axes), 2)
                self.assertIn(problem + ' TEST', fig.axes[0].get_title())
                self.assertIn(dataset, fig.axes[1].get_title())
                for axis, data in zip(fig.axes, (pred, lib)):
                    self.assertIn(f'\u6a21\u578bTop1={data.top1_pct:.2f}%', axis.get_title())
                    self.assertEqual(len(axis.patches), 2 * len(pred.pool))
                self.assertTrue(all(p['top1'] is None for p in panels if p['kind'] == 'true_winner'))
            finally:
                plt.close(fig)

    def test_learning_curves_reject_missing_saved_train_val(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            write_json(run / 'history.json', [dict(successful_updates=1440, train=None, val={})])
            with self.assertRaises(ValueError):
                reporting.learning_curves(run, self.config(), lock())

    def test_no_historical_module_globals_are_rebound(self):
        reporting.validate_matched_configs(self.config(), self.config('B'))
        for module, snapshot in self.original_modules.values():
            for name, value in snapshot.items():
                self.assertIs(getattr(module, name), value, name)


if __name__ == '__main__':
    unittest.main()

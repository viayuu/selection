"""CPU-only synthetic R94 contracts; no real assets, factory or full reports."""

import copy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from . import r94_reporting as reporting


def config(arm='A'):
    variant = reporting.VARIANTS[arm]
    donor = dict(variant=variant, provenance=dict(variant=variant,
        checkpoint_path=None if arm == 'A' else '/synthetic/pinned.pt',
        checkpoint_sha256=None if arm == 'A' else 'p' * 64,
        random_seed=940002 if arm == 'A' else None,
        checkpoint_pin=dict(sha256='p' * 64), native_source_sha256='same-donor-source',
        output_dim=128, supported_problems=[p for p in reporting.PROBLEMS if p not in ('TSP', 'ATSP')],
        spec=dict(layers=6, width=128)))
    spec = dict(r94_schema_version=1, donor_width=128, encoder_layer_num=4,
        covered_problems=[p for p in reporting.PROBLEMS if p not in ('TSP', 'ATSP')],
        unsupported_problems=['TSP', 'ATSP'], fusion_location='before_first_original_instance_encoder_layer')
    return dict(experiment='R94', r94_schema_version=1, arm=arm,
        initialization=reporting.INITIALIZATION, model_factory=reporting.MODEL_FACTORY,
        model_spec=spec, frozen_donor=dict(variant=variant, frozen=True, eval_only=True),
        data_root='/synthetic/frozen-data', rows_per_problem=dict(train=81000, val=4500, test=4500),
        baselines={p: dict(selection_split='train', sbs_solver='m0') for p in reporting.PROBLEMS},
        candidate_pools={p: ['m0', 'm1'] for p in reporting.PROBLEMS},
        candidate_pool_ids={p: [0, 1] for p in reporting.PROBLEMS}, winner_rule='native FP64 ind',
        common_trainable_sha256='r77-scratch-hash', node_fusion_initialization_sha256='all-fresh-hash',
        preparation_sha256='prepared-source-hash', seed=2, global_batch_size=1024,
        logical_global_batch=1024, max_successful_updates=57600, warmup_updates=4320,
        eval_every_updates=1440, eval_batch_size=256, task_order_sha256='same-stream-hash',
        lr=dict(peak=1e-4, end=1e-6), optimizer='AdamW', weight_decay=1e-4, grad_clip=1.,
        dropout=.1, rdrop_weight=.35, rdrop_ramp_updates=4320,
        loss=dict(ce=.35, comparison=.35, risk=.02, cost_scale=.01),
        checkpoint_rule='strict full18 VAL macro_vs_sbs_pct; first ties', inference_rule='locked maximin',
        main_protocol_lock=dict(sha256='same-lock'), objective=dict(forward_passes=2, rdrop=True),
        microbatch_caps={p: 352 if p in ('TSP', 'CVRP') else 1024 for p in reporting.PROBLEMS},
        feature_cache_root='/synthetic/features/' + arm, gpu_type='3090' if arm == 'A' else '4090',
        model_params=dict(dropout=.1, model_spec=copy.deepcopy(spec), frozen_donor=donor,
            r94_fresh_trainable_sha256='all-fresh-hash', role_arm='B_execution_anchor'))


def lock(arm='A'):
    return dict(checkpoint_sha256=arm * 64, state_hash='state-' + arm,
        inference_rule='locked maximin', epoch=1, selected_update=1440)


def timing_record(arm='A'):
    measurement = dict(batch_seconds=[.1, .2, .3], mean_seconds_per_instance=.6 / (3 * 128))
    return dict(experiment='R94', arm=arm, complete=True, checkpoint_sha256=arm * 64,
        state_hash='state-' + arm, repetitions=3, device='cuda:0',
        inference_rule='locked maximin', includes_native_encoder=True,
        scope='first128 input rows, synchronized GPU wall time', excludes='I/O and one-time preparation',
        per_problem=[dict(problem=p, instances=128, donor_plus_selector=copy.deepcopy(measurement),
            cache_hit_selector=copy.deepcopy(measurement)) for p in reporting.PROBLEMS])


def actual_shaped_config(arm='A'):
    """Mirror runner/model/donor metadata shape, with invented hashes/paths only."""
    result = config(arm)
    result['frozen_donor'].update(
        pretraining_source='existing R48 pinned ReLD-MOEL solver encoder only'
            if arm == 'B' else 'none: fixed random donor seed940002',
        known_unknowns='original pretraining instance manifest, training CLI and overlap are unavailable'
            if arm == 'B' else 'not applicable: no pretrained weights loaded')
    result['model_spec'].update(backbone='original_full18_r88b_r77_dualstream',
        model_role_arm='B_execution_anchor', role_slot=3, donor='frozen_static_reld_moel_encoder',
        fusion='coord_proj(raw8)+geometry_residual+linear(layernorm(stopgrad(H)))',
        layer_norm_affine=False, projection_bias=False, projection_seed=940003,
        raw_fallback_quantization='float16_like_frozen_feature_cache',
        new_trainable_parameter_names=['instance_encoder.nco_projection.weight'],
        joint_layer_num=2, solver_features=34, dropout=.1,
        selection_head='original_score_mlp_maximin', objective='original_r77_ce_comparison_risk_rdrop')
    params = result['model_params']
    params.update(model_spec=copy.deepcopy(result['model_spec']), r94_schema_version=1,
        initialization=reporting.INITIALIZATION, architecture='pairwise', ignore_coord_dist=True,
        encoder_layer_num=4, joint_layer_num=2, local_geometry=True, geometry_mode='real',
        rezero=False, solver_representation='handcrafted', pair_mode='score_difference',
        selection_head='original_score_mlp_maximin')
    provenance = params['frozen_donor']['provenance']
    provenance.update(native_source_files={'MOEModel_Light.py': 'same-source-hash'},
        checkpoint_pin=dict(path='/synthetic/reld/epoch-5000.pt', sha256='p' * 64),
        known_unknown_pretraining_source=dict(applies_to_loaded_weights=arm == 'B',
            known='six source training tasks', training_tasks=['CVRP', 'OVRP', 'VRPB', 'VRPL', 'VRPTW', 'OVRPTW'],
            unknown=['historical CLI', 'training instance manifest', 'exact trajectory', 'overlap with selector inputs']))
    result.update(feature_precision='native static encoder FP32, per-node output stored FP16; same online quantization',
        feature_cache_identity='input cache identity + donor specification; no performance targets',
        objective=dict(main='unchanged R77 CE/comparison/risk plus symmetric R-Drop',
            coefficients=dict(ce=.35, comparison=.35, risk=.02),
            rdrop='linear ramp to .35 over4320 updates', forward_passes=2, teacher_gradients=False,
            teacher_auxiliary_loss=False, route_targets=False, cross_instance_objective=False,
            mutable_objective_state=False), source_sha256={'r94_experiment.py': 'same-source'})
    return result


class ReportingTests(unittest.TestCase):
    def setUp(self):
        # Mock private adapters, never attributes of the imported historical modules.
        self.original_modules = {}
        for name in ('shared', 'previous', 'common', 'helpers', 'r93'):
            module = getattr(reporting, name)
            self.original_modules[name] = (module, dict(vars(module)))
            adapter = SimpleNamespace(**vars(module))
            replacement = patch.object(reporting, name, adapter)
            replacement.start()
            self.addCleanup(replacement.stop)

    def test_cli_factory_and_original_objective_boundary(self):
        self.assertEqual(reporting.ARMS, ('A', 'B'))
        self.assertEqual(reporting.BUDGETS, dict(A=57600, B=57600))
        self.assertEqual(reporting.MODEL_FACTORY, 'code.V4.r94_experiment.load_checkpoint')
        for command in ('lib', 'report', 'compare', 'status'):
            args = reporting.parser().parse_args([command, '--root', '/model', '--arm', 'B',
                '--device', 'cpu', '--lib-batch-size', '1', '--nss-root', '/nss',
                '--historical-root', '/r84', '--r89-root', '/r89'])
            self.assertEqual(args.root, Path('/model'))
            self.assertEqual(args.arm, 'B')
        self.assertEqual(reporting.parser().parse_args(['report']).root.name, 'R94_pretrained_node_fusion')
        for phrase in ('CE=.35', 'comparison=.35', 'risk=.02', 'R-Drop=.35', 'not R93 pure CE',
                       'not budget-matched', 'R50', 'not16-task pretraining', 'not full-system'):
            self.assertIn(phrase, reporting.REFERENCE_BOUNDARY)

    def test_factory_is_late_bound_and_not_constructed_by_schema_check(self):
        record = (Path('/model/A'), config(), {}, {}, {})
        with patch.object(reporting.shared, 'locked_inputs', return_value=record) as load, \
                patch.object(reporting.importlib, 'import_module') as factory:
            self.assertEqual(reporting.locked_inputs(Path('/model'), 'A'), record)
            load.assert_called_once_with(Path('/model'), 'A', experiment='R94', arms=reporting.BUDGETS)
            factory.return_value.validate_checkpoint_config.assert_called_once_with(record[1])
            factory.return_value.load_checkpoint.assert_not_called()
            record[1]['model_factory'] = 'code.V4.r93_experiment.load_checkpoint'
            with self.assertRaises(ValueError):
                reporting.locked_inputs(Path('/model'), 'A')
        with self.assertRaises(ValueError):
            reporting.locked_inputs(Path('/model'), 'C')

    def test_matched_donor_control_accepts_only_weight_provenance_and_placement_differences(self):
        a, b = config('A'), config('B')
        original = copy.deepcopy((a, b))
        reporting.same_data(a, b, matched=True)
        self.assertEqual((a, b), original)
        hardware = reporting.hardware_disclosure({'A': a, 'B': b})
        self.assertFalse(hardware['same_reported_hardware'])
        self.assertFalse(hardware['bitwise_hardware_matched'])

    def test_actual_shaped_configs_allow_only_donor_provenance_differences(self):
        a, b = actual_shaped_config(), actual_shaped_config('B')
        original = copy.deepcopy((a, b))
        with patch.object(reporting.importlib, 'import_module') as imports:
            reporting.same_data(a, b, matched=True)
            result = reporting.validate_matched_configs(a, b)
        imports.assert_not_called()
        self.assertTrue(result['matched'])
        self.assertEqual(result['node_fusion_initialization_sha256'], a['node_fusion_initialization_sha256'])
        self.assertEqual((a, b), original)
        self.assertEqual(a['model_params']['architecture'], 'pairwise')
        self.assertTrue(a['model_params']['ignore_coord_dist'])
        for key, value in [('native_source_sha256', 'different-source'),
                           ('checkpoint_pin', dict(sha256='different-pin')), ('supported_problems', ['CVRP']),
                           ('output_dim', 64), ('spec', dict(layers=1))]:
            bad = copy.deepcopy(b)
            bad['model_params']['frozen_donor']['provenance'][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                reporting.validate_matched_configs(a, bad)

    def test_matched_control_rejects_init_batches_micros_loss_rule_and_unknown_core_changes(self):
        for key, value in (
                ('node_fusion_initialization_sha256', 'trained-selector'),
                ('common_trainable_sha256', 'different'), ('preparation_sha256', 'different'),
                ('task_order_sha256', 'different'), ('global_batch_size', 512), ('logical_global_batch', 512),
                ('warmup_updates', 1), ('max_successful_updates', 57599), ('eval_batch_size', 64),
                ('weight_decay', 0.), ('dropout', 0.), ('rdrop_weight', 0.), ('rdrop_ramp_updates', 0),
                ('loss', dict(ce=1., comparison=0., risk=0.)), ('objective', dict(rdrop=False)),
                ('microbatch_caps', dict(TSP=128)), ('checkpoint_rule', 'TEST-best'),
                ('inference_rule', 'changed ties'), ('new_core_control', True)):
            other = config('B')
            other[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                reporting.same_data(config(), other, matched=True)
        other = config('B')
        other['model_params']['new_head'] = True
        with self.assertRaises(ValueError):
            reporting.same_data(config(), other, matched=True)
        other = config('B')
        other['model_params']['frozen_donor']['provenance']['spec']['width'] = 256
        with self.assertRaises(ValueError):
            reporting.same_data(config(), other, matched=True)

    def test_schema_rejects_changed_donor_coverage_or_r50_pool_first(self):
        for key, value in [('donor_width', 64), ('encoder_layer_num', 0),
                           ('covered_problems', ['CVRP']), ('unsupported_problems', []),
                           ('fusion_location', 'pool-first')]:
            other = config()
            other['model_spec'][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                reporting.validate_report_config(other)

    def test_historical_data_and_candidate_pool_alignment_is_strict(self):
        a, b = config(), config('B')
        b.update(experiment='R84', initialization='old', dropout=0.)
        reporting.same_data(a, b)
        for key in ('data_root', 'rows_per_problem', 'baselines', 'candidate_pools', 'candidate_pool_ids', 'winner_rule'):
            bad = copy.deepcopy(b)
            bad.pop(key)
            with self.subTest(key=key), self.assertRaises(ValueError):
                reporting.same_data(a, bad)

    def test_target36_is_fixed_and_hash_bound(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'target_list.json'
            path.write_text('{}', encoding='utf-8')
            identity = dict(path=str(path))
            with patch.object(reporting.helpers, 'load_targets', return_value=([('TSP', str(i)) for i in range(36)], identity)):
                targets, result = reporting.load_targets({}, {}, {})
            self.assertEqual(len(targets), result['target_count'])
            self.assertEqual(result['sha256'], reporting.file_hash(path))
            with patch.object(reporting.helpers, 'load_targets', return_value=([('TSP', 'm0')], identity)):
                with self.assertRaises(ValueError):
                    reporting.load_targets({}, {}, {})

    def test_saved_state_guard_and_r94_relabel(self):
        val = dict(identity=dict(model_state_sha256='same'))
        test = copy.deepcopy(val)
        with patch.object(reporting.helpers, 'selected_predictions', return_value=(
                {}, [dict(model='R87/A')], [dict(model='R87/A')], {}, {})):
            result = reporting.selected_predictions(None, config('B'), {}, val, test)
        self.assertEqual(result[1][0]['model'], 'R94/B')
        test['identity']['model_state_sha256'] = 'other'
        with self.assertRaises(ValueError):
            reporting.selected_predictions(None, config(), {}, val, test)

    def test_val_screen_is_descriptive_and_finite(self):
        metric = dict(macro_top1=.7, macro_mean_cost=10., macro_actual_regret_pct=.4)
        refs = {name: dict(val=dict(macro=dict(metric))) for name in ('R89B', 'R84A')}
        val = dict(macro=dict(metric, macro_top1=.71, macro_mean_cost=9.9, macro_actual_regret_pct=.39))
        rows = reporting.val_screen(config(), lock(), val, refs)
        self.assertEqual([r['reference_role'] for r in rows], ['main', 'secondary'])
        self.assertTrue(all(r['model'] == 'R94/A' and r['joint_val_pass'] and not r['checkpoint_reselection'] for r in rows))
        val['macro']['macro_top1'] = float('nan')
        with self.assertRaises(ValueError):
            reporting.val_screen(config(), lock(), val, refs)

    def test_gpu_admin_and_nonexplicit_lib_device_rejected_before_inputs(self):
        with patch.object(reporting, 'locked_inputs') as load:
            for fn in (reporting.report, reporting.compare):
                with self.assertRaises(ValueError):
                    fn(SimpleNamespace(device='cuda:0'))
            for device in ('cpu', 'cuda', 'mps'):
                with self.assertRaises(ValueError):
                    reporting.lib_stage(SimpleNamespace(device=device))
            load.assert_not_called()

    def test_mocked_lib_uses_r94_factory_and_never_builds_cpu_reports(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            args = reporting.parser().parse_args(['lib', '--arm', 'A', '--device', 'cuda:0'])
            with patch.object(reporting, 'locked_inputs', return_value=(run, config(), lock(),
                    dict(identity=dict(model_state_sha256='model-state')), {})), \
                    patch.object(reporting.previous, 'validate_nss_reference', return_value={}), \
                    patch.object(reporting.common, 'r80_lib', return_value={'TSPLIB': {}, 'CVRPLIB': {}}) as lib, \
                    patch.object(reporting.shared, 'build_reports') as pdf:
                reporting.lib_stage(args)
            self.assertEqual(lib.call_args.kwargs['factory_module'], 'code.V4.r94_experiment')
            pdf.assert_not_called()
            self.assertEqual(reporting.read_json(run / 'lib_stage_complete.json')['exit_status'], 0)

    def test_mocked_lib_failure_is_preserved_without_retry(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            args = reporting.parser().parse_args(['lib', '--arm', 'A', '--device', 'cuda:0'])
            with patch.object(reporting, 'locked_inputs', return_value=(run, config(), lock(),
                    dict(identity=dict(model_state_sha256='model-state')), {})), \
                    patch.object(reporting.previous, 'validate_nss_reference', return_value={}), \
                    patch.object(reporting.common, 'r80_lib', side_effect=RuntimeError('synthetic failure')) as lib:
                with self.assertRaises(RuntimeError):
                    reporting.lib_stage(args)
            self.assertEqual(lib.call_count, 1)
            failure = reporting.read_json(run / 'lib_stage_status.json')
            self.assertEqual(failure['phase'], 'lib_failed')
            self.assertFalse(failure['automatic_retry'])

    def test_timing_is_required_same_checkpoint_full18_and_scope_separated(self):
        with tempfile.TemporaryDirectory() as tmp:
            run, record = Path(tmp), timing_record()
            with self.assertRaises(ValueError):
                reporting.inference_timing(run, lock())
            reporting.write_json(run / 'inference_timing.json', record)
            result = reporting.inference_timing(run, lock())
            self.assertEqual(len(result['measurements']), 18)
            self.assertTrue(result['donor_plus_selector_measured'])
            self.assertFalse(result['full_system_measured'])
            self.assertFalse(result['end_to_end_measured'])
            self.assertAlmostEqual(result['measurements'][0]['cache_hit_selector_ms_per_instance'], 1000 * .6 / 384)
            self.assertEqual(result['sha256'], reporting.file_hash(run / 'inference_timing.json'))
            for field, value in [('complete', False), ('checkpoint_sha256', 'other'),
                                 ('state_hash', 'other'), ('repetitions', 1), ('per_problem', record['per_problem'][:-1])]:
                bad = dict(record, **{field: value})
                reporting.write_json(run / 'inference_timing.json', bad)
                with self.subTest(field=field), self.assertRaises(ValueError):
                    reporting.inference_timing(run, lock())
            bad = copy.deepcopy(record)
            bad['per_problem'][0]['cache_hit_selector']['mean_seconds_per_instance'] *= 2
            reporting.write_json(run / 'inference_timing.json', bad)
            with self.assertRaises(ValueError):
                reporting.inference_timing(run, lock())

    def test_report_blocks_missing_timing_and_records_failure_without_render(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            args = reporting.parser().parse_args(['report', '--arm', 'A'])
            # Stop before saved arrays/rendering: use only synthetic dictionaries.
            with patch.object(reporting, 'locked_inputs', return_value=(run, config(), lock(), {}, {})), \
                    patch.object(reporting, '_completed_lib'), \
                    patch.object(reporting.previous, 'validate_nss_reference', return_value={}), \
                    patch.object(reporting.shared, 'load_saved_splits', return_value=({}, [], [])), \
                    patch.object(reporting.shared, 'saved_libs'), \
                    patch.object(reporting.shared, 'nss_comparisons', return_value=([], {})), \
                    patch.object(reporting, 'diagnostic_tables', return_value={}), \
                    patch.object(reporting.shared, 'build_reports') as render:
                with self.assertRaisesRegex(ValueError, 'inference_timing'):
                    reporting.report(args)
            render.assert_not_called()
            self.assertEqual(reporting.read_json(run / 'postprocess_status.json')['exit_status'], 1)

    def test_learning_curves_need_real_saved_train_val_not_synthetic_filled_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            reporting.write_json(run / 'history.json', [dict(successful_updates=1440, train=None, val={})])
            with self.assertRaises(ValueError):
                reporting.learning_curves(run, config(), lock())

    def test_distribution_native_top1_paired_same_axis_test_before_lib(self):
        import matplotlib.pyplot as plt
        helper = reporting.common.reporting_helpers()
        helper.setup_style()
        pred = helper.Predictions(Path('/synthetic/saved'), ['BQ', 'T2T'], np.array([0, 0, 1, 1]),
            np.array([0, 1, 1, 0]), np.ones((4, 2), dtype=np.float64), np.arange(4), None, np.arange(2), None)
        lib = helper.Predictions(Path('/synthetic/lib'), pred.pool, np.zeros(49, dtype=int),
            np.arange(49) % 2, np.ones((49, 2)), np.arange(49), None, np.arange(2), None)
        panels = helper.make_panels('TSP', pred, lib, pred.pool)
        fig = reporting.common._distribution(helper, 'TSP', pred.pool, panels, 'NEW90k', 'R94/A', lock())
        self.addCleanup(plt.close, fig)
        self.assertEqual(len(fig.axes), 2)
        self.assertIn('TSP TEST', fig.axes[0].get_title())
        self.assertIn('TSPLIB', fig.axes[1].get_title())
        for axis, data in zip(fig.axes, (pred, lib)):
            self.assertIn(f'\u6a21\u578bTop1={data.top1_pct:.2f}%', axis.get_title())
            self.assertEqual(len(axis.patches), 2 * len(pred.pool))
        self.assertTrue(all(p['top1'] is None for p in panels if p['kind'] == 'true_winner'))

    def test_helpers_globals_are_not_rebound(self):
        reporting.same_data(config(), config('B'), matched=True)
        for module, snapshot in self.original_modules.values():
            for name, value in snapshot.items():
                self.assertIs(getattr(module, name), value, name)


if __name__ == '__main__':
    unittest.main()

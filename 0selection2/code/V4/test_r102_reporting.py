"""Saved-array reporting fixtures only; never load a selector or donor weight."""

import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from . import r102_experiment as runner
from . import r102_reporting as reporting
from . import r84_reporting, r87_reporting
from .r76_experiment import read_json, write_json
from .test_r102_experiment import synthetic_config


def full_provenance_fixture(arm):
    """Actual metadata construction, with no native encoder or weight I/O."""
    from . import r102_donor as donor
    reld = donor.reld

    class RoutingMetadata(torch.nn.Identity):
        def provenance(self):
            return dict(variant='pretrained', checkpoint_path=str(reld.DEFAULT_CHECKPOINT),
                checkpoint_sha256=reld.CHECKPOINT_SHA256, native_source_sha256='d' * 64,
                native_source_files={name: 'c' * 64 for name in ('MOEModel_Light.py', 'MOELayer.py')},
                output_dim=128, supported_problems=list(reld.SUPPORTED_PROBLEMS), random_seed=None,
                checkpoint_pin=dict(path=str(reld.DEFAULT_CHECKPOINT), sha256=reld.CHECKPOINT_SHA256),
                spec=copy.deepcopy(reld.SPEC), known_unknown_pretraining_source=dict(
                    applies_to_loaded_weights=True,
                    known='R48 pinned ReLD-MOEL epoch5000/Train_ALL; six source training tasks',
                    training_tasks=['CVRP', 'OVRP', 'VRPB', 'VRPL', 'VRPTW', 'OVRPTW'],
                    unknown=['historical CLI', 'training instance manifest', 'exact trajectory',
                             'overlap with selector inputs'],
                    note='16-task input coverage is not 16-task pretraining or a feasibility certificate'))

    with patch.object(donor, '_build_tsp', side_effect=torch.nn.Identity), \
            patch.object(donor, '_build_atsp', side_effect=torch.nn.Identity), \
            patch.object(donor, '_load_encoder_weights'), \
            patch.object(donor, '_sha256', return_value='c' * 64), \
            patch.object(reld, 'StaticReLDDonor', side_effect=RoutingMetadata), \
            patch.object(torch, 'load', side_effect=AssertionError('Metadata fixture cannot read weights')):
        return donor.FullTypeDonor(arm).provenance()


def lock(arm='A'):
    return dict(arm=arm, checkpoint_sha256=arm.lower() * 64, state_hash='synthetic-' + arm,
        inference_rule=runner.INFERENCE_RULE, epoch=1, selected_update=1440, selection_split='val')


def timing_record(arm='A'):
    row = dict(online_donor_verified=True, online_fields_removed=['nco_nodes', 'nco_node_mask'])
    for scope, calls in (('donor_plus_selector', 4), ('cache_hit_selector', 0)):
        row[scope] = dict(batch_seconds=[.1, .2, .3], mean_seconds_per_instance=.6 / (3 * 128),
            donor_forward_calls=calls, warmup_batches=1)
    return dict(experiment='R102', arm=arm, variant=runner.ARM_VARIANTS[arm], complete=True,
        checkpoint_sha256=lock(arm)['checkpoint_sha256'], state_hash=lock(arm)['state_hash'],
        inference_rule=runner.INFERENCE_RULE, device='cuda:0', repetitions=3, warmup_batches=1,
        scope='GPU-resident batch128 amortized; not singleton', excludes='I/O, host transfer and preparation',
        raw_input_source='original_base_cache_without_both_nco_fields', includes_native_encoder=True,
        online_donor_verified=True, all18_donor_coverage=True, includes_solver_decoder=False,
        full_system_measured=False, end_to_end_measured=False, singleton_measured=False,
        cache_hit_is_total=False, model_reselection=False,
        per_problem=[dict(problem=p, instances=128, **copy.deepcopy(row)) for p in reporting.PROBLEMS])


def prediction_rows(pred):
    return {p: SimpleNamespace(pool=['m0', 'm1'], pred=np.array(pred), winner=np.array([0, 1, 0]),
        costs=np.array([[1., 3.], [3., 1.], [1., 4.]], dtype=np.float64), n=3,
        indices=np.arange(3), source_indices=np.arange(3), pool_ids=np.arange(2)) for p in reporting.PROBLEMS}


class R102ReportingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.temporary = tempfile.TemporaryDirectory()
        cls.configs = {arm: synthetic_config(cls.temporary.name, arm)[1] for arm in reporting.ARMS}
        for arm, config in cls.configs.items():
            config['donor_provenance'] = full_provenance_fixture(arm)
            config['model_params']['frozen_donor']['provenance'] = copy.deepcopy(config['donor_provenance'])

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def setUp(self):
        guard = patch('torch.cuda._lazy_init', side_effect=AssertionError('CPU reports cannot initialize CUDA'))
        guard.start()
        self.addCleanup(guard.stop)

    def config(self, arm='A'):
        return copy.deepcopy(self.configs[arm])

    def test_private_helpers_and_cpu_cli_no_global_rebinding(self):
        self.assertIsNot(reporting.shared, r84_reporting)
        self.assertIsNot(reporting.helpers, r87_reporting)
        self.assertIsNot(reporting.common, r84_reporting.common)
        self.assertIs(reporting.shared.build_reports.__globals__, vars(reporting.shared))
        self.assertEqual(reporting.MODEL_FACTORY, 'code.V4.r102_experiment.load_checkpoint')
        args = reporting.parser().parse_args(['report', '--arm', 'A'])
        reporting.require_cpu(args)
        args.device = 'cuda:0'
        with self.assertRaises(ValueError):
            reporting.require_cpu(args)
        for phrase in ('BOTH', 'PRETRAINED R94B', 'not matched', 'target36', 'GPU-resident batch128',
                       'NOT singleton', 'Development-exposed TEST', '1941257', '98304'):
            self.assertIn(phrase, reporting.REFERENCE_BOUNDARY)

    def test_dictionary_matched_gate_allows_only_extra_donor_weights(self):
        a, b = self.config(), self.config('B')
        before = copy.deepcopy((a, b))
        with patch.object(torch, 'load', side_effect=AssertionError('dictionary validation cannot load weights')):
            result = reporting.validate_matched_configs(a, b)
        self.assertTrue(result['routing_donor_identical'])
        self.assertFalse(result['historical_R94B_matched'])
        self.assertEqual(before, (a, b))
        b['gpu_type'] = '3090'
        reporting.same_data(a, b, matched=True)
        for key, value in (('routing_feature_cache_root', '/different'), ('new_control', True),
                ('common_trainable_sha256', 'c' * 64), ('task_order_sha256', 'd' * 64),
                ('auxiliary_loss', True), ('early_stop', True)):
            bad = self.config('B')
            bad[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                reporting.validate_matched_configs(a, bad)
        for component, field in (('VRP', 'native_source_sha256'), ('TSP', 'output_dim'), ('ATSP', 'checkpoint_pin')):
            bad = self.config('B')
            bad['donor_provenance']['components'][component][field] = 'different'
            bad['model_params']['frozen_donor']['provenance'] = copy.deepcopy(bad['donor_provenance'])
            with self.subTest(component=component, field=field), self.assertRaises(ValueError):
                reporting.validate_matched_configs(a, bad)

    def test_full_provenance_allows_expected_extra_flags_and_retains_routing_metadata(self):
        a, b = self.config(), self.config('B')
        before = copy.deepcopy((a, b))
        for arm, config in (('A', a), ('B', b)):
            value = config['donor_provenance']
            self.assertEqual(value['external_pretraining_by_component'],
                             dict(TSP=arm == 'B', ATSP=arm == 'B', VRP=True))
            normalized = reporting._donor_controls(value)
            self.assertEqual(normalized['external_pretraining_by_component'], dict(VRP=True))
            self.assertEqual(normalized['components']['VRP'], value['components']['VRP'])
            self.assertEqual(normalized['spec'], value['spec'])
        with patch.object(torch, 'load', side_effect=AssertionError('No weights in matched report validation')):
            self.assertTrue(reporting.validate_matched_configs(a, b)['routing_donor_identical'])
        self.assertEqual((a, b), before)

    def test_full_provenance_rejects_wrong_vrp_extra_and_non_boolean_flags(self):
        for arm in reporting.ARMS:
            expected = dict(TSP=arm == 'B', ATSP=arm == 'B', VRP=True)
            invalid = [None, dict(expected, VRP=False), dict(expected, VRP=1),
                dict(expected, TSP=arm != 'B'), dict(expected, ATSP=arm != 'B'),
                dict(expected, TSP=int(arm == 'B')), dict(expected, unexpected=True),
                {key: value for key, value in expected.items() if key != 'ATSP'}]
            for flags in invalid:
                configs = {name: self.config(name) for name in reporting.ARMS}
                bad = configs[arm]
                bad['donor_provenance']['external_pretraining_by_component'] = flags
                bad['model_params']['frozen_donor']['provenance'] = copy.deepcopy(bad['donor_provenance'])
                with self.subTest(arm=arm, flags=flags), self.assertRaisesRegex(ValueError, 'external_pretraining_by_component'):
                    reporting.validate_matched_configs(configs['A'], configs['B'])

    def test_locked_inputs_uses_r102_validator_not_historical_model_factory(self):
        inputs = Path('/synthetic/A'), self.config(), lock(), {}, {}
        with patch.object(reporting.shared, 'locked_inputs', return_value=inputs) as saved, \
                patch.object(torch, 'load', side_effect=AssertionError('no weights')):
            self.assertEqual(reporting.locked_inputs('/synthetic', 'A'), inputs)
        saved.assert_called_once_with(Path('/synthetic'), 'A', experiment='R102', arms=reporting.BUDGETS)

    def test_timing_full18_exact_hook_counts_and_separate_cache_scope(self):
        with tempfile.TemporaryDirectory() as directory:
            run, record = Path(directory), timing_record()
            write_json(run / 'inference_timing.json', record)
            result = reporting.inference_timing(run, lock())
            self.assertEqual(len(result['measurements']), 18)
            self.assertFalse(result['cache_hit_is_total'])
            self.assertFalse(result['end_to_end_measured'])
            for key, value in (('raw_input_source', 'cached_nodes'), ('all18_donor_coverage', False),
                    ('cache_hit_is_total', True), ('singleton_measured', True), ('per_problem', record['per_problem'][:-1]),
                    ('checkpoint_sha256', 'different'), ('arm', 'B'), ('device', 'cpu')):
                write_json(run / 'inference_timing.json', dict(record, **{key: value}))
                with self.subTest(key=key), self.assertRaises(ValueError):
                    reporting.inference_timing(run, lock())
            for scope, field, value in (('donor_plus_selector', 'donor_forward_calls', 0),
                    ('cache_hit_selector', 'donor_forward_calls', 4),
                    ('donor_plus_selector', 'mean_seconds_per_instance', .5),
                    ('donor_plus_selector', 'batch_seconds', [True, .2, .3])):
                bad = copy.deepcopy(record)
                bad['per_problem'][0][scope][field] = value
                write_json(run / 'inference_timing.json', bad)
                with self.subTest(scope=scope, field=field), self.assertRaises(ValueError):
                    reporting.inference_timing(run, lock())
            bad = copy.deepcopy(record)
            bad['per_problem'][0]['online_fields_removed'] = ['nco_nodes']
            write_json(run / 'inference_timing.json', bad)
            with self.assertRaises(ValueError):
                reporting.inference_timing(run, lock())

    def test_real_rescues_losses_and_new_false_pick_cost(self):
        old = {s: prediction_rows([1, 0, 0]) for s in ('TRAIN', 'VAL', 'TEST')}
        new = {s: prediction_rows([0, 1, 1]) for s in ('TRAIN', 'VAL', 'TEST')}
        targets = [(p, method) for p in reporting.PROBLEMS for method in ('m0', 'm1')]
        identity = dict(identity=dict(rule='fixed R84 target36'), categories={p + '/' + m: 'zero' for p, m in targets})
        pairs, recovery, totals = reporting.comparison_tables(old, new, targets, identity, 'R102/B', 'R102/A')
        row = next(r for r in recovery if r['split'] == 'TEST' and r['problem'] == 'TSP' and r['solver'] == 'm1')
        self.assertEqual((row['rescued'], row['lost'], row['new_false_pick_count']), (1, 0, 1))
        self.assertEqual(row['new_false_pick_mean_cost'], 4.)
        self.assertAlmostEqual(row['new_false_pick_cost_contribution'], 4 / 3)
        self.assertAlmostEqual(row['new_false_pick_cost_delta_contribution'], 1.)
        self.assertEqual((totals[-1]['rescued'], totals[-1]['lost']), (36, 18))
        self.assertAlmostEqual(totals[-1]['new_false_pick_cost_contribution'], 4 / 3)
        self.assertEqual(next(r for r in pairs if r['split'] == 'TEST' and r['problem'] == 'ALL')['net_correct'], 18)

    def test_fixed36_not_reselected_from_current_predictions(self):
        with tempfile.TemporaryDirectory() as directory:
            config = self.config()
            path = Path(directory) / 'targets.json'
            config['main_protocol_lock']['record']['target_list'] = str(path)
            identity = dict(rule='fixed historical R84 zero/rare36', data_root=config['data_root'],
                baseline_checkpoint_sha256='a' * 64)
            rows = [dict(problem=p, method=m, category='zero') for p in reporting.PROBLEMS for m in ('m0', 'm1')]
            write_json(path, dict(identity=identity, targets=rows))
            with patch.object(torch, 'load', side_effect=AssertionError('no weights')):
                targets, receipt = reporting.load_targets(config, config, dict(checkpoint_sha256='a' * 64))
            self.assertEqual(len(targets), receipt['target_count'])
            self.assertEqual(receipt['target_count'], 36)
            self.assertTrue(receipt['sha256'])
            write_json(path, dict(identity=identity, targets=rows[:-1]))
            with self.assertRaises(ValueError):
                reporting.load_targets(config, config, dict(checkpoint_sha256='a' * 64))

    def test_native_lib49_100_same_bank(self):
        config = self.config()
        predictions = {(dataset, problem): SimpleNamespace(n=n, pool=config['candidate_pools'][problem],
            pool_ids=np.array(config['candidate_pool_ids'][problem]))
            for dataset, problem, n in (('TSPLIB', 'TSP', 49), ('CVRPLIB', 'CVRP', 100))}
        reporting.validate_lib_population(predictions, config)
        predictions['TSPLIB', 'TSP'].n = 48
        with self.assertRaises(ValueError):
            reporting.validate_lib_population(predictions, config)

    def test_same_axis_native_distribution_top1_test_then_lib(self):
        import matplotlib.pyplot as plt
        reporting.render_environment()
        helper = reporting.common.reporting_helpers()
        helper.setup_style()
        test = helper.Predictions(Path('/synthetic'), ['m0', 'm1'], np.array([0, 0, 1, 1]),
            np.array([0, 1, 1, 0]), np.ones((4, 2)), np.arange(4), None, np.arange(2), None)
        for problem, dataset, n in (('TSP', 'TSPLIB', 49), ('CVRP', 'CVRPLIB', 100)):
            lib = helper.Predictions(Path('/synthetic/lib'), test.pool, np.zeros(n, dtype=int),
                np.arange(n) % 2, np.ones((n, 2)), np.arange(n), None, np.arange(2), None)
            panels = helper.make_panels(problem, test, lib, test.pool)
            fig = reporting.common._distribution(helper, problem, test.pool, panels, 'NEW90k', 'R102/A', lock())
            try:
                self.assertIn(problem + ' TEST', fig.axes[0].get_title())
                self.assertIn(dataset, fig.axes[1].get_title())
                for axis, data in zip(fig.axes, (test, lib)):
                    self.assertIn('\u6a21\u578bTop1=' + f'{data.top1_pct:.2f}%', axis.get_title())
                    self.assertEqual(len(axis.patches), 4)
            finally:
                plt.close(fig)

    def test_learning_curves_saved_complete_schedule_and_no_fabrication(self):
        macro = dict(macro_ce=.4, macro_top1=.5, macro_mean_cost=2., macro_actual_regret_pct=1., macro_vs_sbs_pct=-1.)
        history = [dict(successful_updates=step, optimization=dict(loss=.4, ce=.5, cmp=.2, risk=.1, kl=.01),
            val=dict(macro=macro), train=dict(macro=macro) if step % 7200 == 0 or step == 1440 else None)
            for step in range(1440, 57601, 1440)]
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            (run / 'full_model_statistics').mkdir()
            write_json(run / 'history.json', history)
            with patch.object(torch, 'load', side_effect=AssertionError('saved curves cannot load model')):
                reporting.learning_curves(run, self.config(), lock())
            self.assertGreater((run / 'full_model_statistics/learning_curves.pdf').stat().st_size, 100)
            self.assertEqual(len((run / 'full_model_statistics/learning_curves.csv').read_text().splitlines()), 41)
            write_json(run / 'history.json', history[:-1])
            with self.assertRaises(ValueError):
                reporting.learning_curves(run, self.config(), lock())

    def test_report_failure_never_invokes_inference_or_claims_complete(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            args = reporting.parser().parse_args(['report', '--arm', 'A'])
            inputs = run, self.config(), lock(), {}, {}
            with patch.object(reporting, 'locked_inputs', return_value=inputs), \
                    patch.object(reporting, 'render_environment', return_value=dict(device='cpu')), \
                    patch.object(reporting, '_completed_lib', side_effect=ValueError('missing LIB')), \
                    patch.object(runner, 'load_checkpoint', side_effect=AssertionError('no inference')) as factory:
                with self.assertRaisesRegex(ValueError, 'missing LIB'):
                    reporting.report(args)
            factory.assert_not_called()
            status = read_json(run / 'postprocess_status.json')
            self.assertFalse(status['complete'])
            self.assertFalse(status['automatic_retry'])


if __name__ == '__main__':
    unittest.main()

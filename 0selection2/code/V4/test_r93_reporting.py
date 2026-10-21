"""Focused CPU contracts: no checkpoints, GPU work, process launch or Git mutation."""

import copy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from . import r93_reporting as reporting


def config(arm='A'):
    return dict(experiment='R93', arm=arm, model_factory=reporting.MODEL_FACTORY,
        data_root='/frozen/new90k', rows_per_problem=dict(train=81000, val=4500, test=4500),
        baselines={'TSP': {'selection_split': 'train'}}, candidate_pools={'TSP': ['BQ', 'T2T']},
        candidate_pool_ids={'TSP': [0, 1]}, winner_rule='native FP64 ind',
        common_trainable_sha256='fresh-R84', preparation_sha256='TRAIN-only',
        initialization='scratch_shared_r77', seed=2, global_batch_size=1024,
        logical_global_batch=1024, max_successful_updates=57600, warmup_updates=4320,
        eval_every_updates=1440, eval_batch_size=256, weight_decay=1e-4,
        dropout=reporting.DROPOUT[arm], gpu_type='3090' if arm == 'A' else '4090',
        objective={'forward_passes': 1, 'rdrop': False},
        model_params=dict(dropout=reporting.DROPOUT[arm], r93_arm=arm,
            role_arm='B_execution_anchor', r93_spec={'base': 'original full18'}))


class ReportingTests(unittest.TestCase):
    def test_parser_contract_has_two_arms_and_factory(self):
        self.assertEqual(reporting.ARMS, ('A', 'B'))
        self.assertEqual(reporting.BUDGETS, dict(A=57600, B=57600))
        self.assertEqual(reporting.MODEL_FACTORY, 'code.V4.r93_experiment.load_checkpoint')
        self.assertIn('released3090_1', reporting.REFERENCE_BOUNDARY)
        self.assertIn('4090_1', reporting.REFERENCE_BOUNDARY)
        self.assertIn('not bitwise hardware-matched', reporting.REFERENCE_BOUNDARY)
        for command in ('lib', 'report', 'status', 'compare'):
            args = reporting.parser().parse_args([command, '--root', '/model', '--arm', 'B',
                '--device', 'cpu', '--nss-root', '/nss', '--historical-root', '/r84', '--r89-root', '/r89'])
            self.assertEqual(args.root, Path('/model'))
            self.assertEqual(args.arm, 'B')
        self.assertEqual(reporting.parser().parse_args(['report']).root.name, 'R93_plain_classification')

    def test_locked_factory_validation_is_late_bound_and_strict(self):
        result = (Path('/model/A'), config(), {}, {}, {})
        with patch.object(reporting.shared, 'locked_inputs', return_value=result) as load, \
                patch.object(reporting.importlib, 'import_module') as module:
            self.assertEqual(reporting.locked_inputs(Path('/model'), 'A'), result)
            load.assert_called_once_with(Path('/model'), 'A', experiment='R93', arms=reporting.BUDGETS)
            module.return_value.validate_checkpoint_config.assert_called_once_with(result[1])
            result[1]['model_factory'] = 'code.V4.r92_experiment.load_checkpoint'
            with self.assertRaises(ValueError):
                reporting.locked_inputs(Path('/model'), 'A')

    def test_matched_pair_accepts_only_dropout_and_hardware_difference(self):
        reporting.same_data(config('A'), config('B'), matched=True)
        for key, value in [('global_batch_size', 512), ('weight_decay', 0.),
                           ('common_trainable_sha256', 'trained-checkpoint'), ('objective', {'rdrop': True})]:
            other = config('B')
            other[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                reporting.same_data(config('A'), other, matched=True)
        other = config('B')
        other['model_params']['new_head'] = True
        with self.assertRaises(ValueError):
            reporting.same_data(config('A'), other, matched=True)

    def test_fixed_target36_is_not_reselected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'target_list.json'
            path.write_text('{}')
            identity = dict(path=str(path))
            with patch.object(reporting.helpers, 'load_targets', return_value=([('TSP', str(i)) for i in range(36)], identity)):
                targets, result = reporting.load_targets({}, {}, {})
            self.assertEqual(len(targets), result['target_count'])
            self.assertEqual(len(result['sha256']), 64)
            with patch.object(reporting.helpers, 'load_targets', return_value=([('TSP', 'BQ')], identity)):
                with self.assertRaises(ValueError):
                    reporting.load_targets({}, {}, {})

    def test_saved_state_guard_and_r93_relabel(self):
        val = dict(identity=dict(model_state_sha256='same'))
        test = copy.deepcopy(val)
        with patch.object(reporting.helpers, 'selected_predictions', return_value=(
                {}, [{'model': 'R87/A'}], [{'model': 'R87/A'}], {}, {})):
            result = reporting.selected_predictions(None, config('B'), {}, val, test)
            self.assertEqual(result[1][0]['model'], 'R93/B')
        test['identity']['model_state_sha256'] = 'changed'
        with self.assertRaises(ValueError):
            reporting.selected_predictions(None, config(), {}, val, test)

    def test_finite_val_screen_discloses_main_and_secondary(self):
        metric = dict(macro_top1=.7, macro_mean_cost=10., macro_actual_regret_pct=.4)
        refs = {name: {'val': {'macro': dict(metric)}} for name in ('R89B', 'R84A')}
        current = dict(macro=dict(metric, macro_top1=.71, macro_mean_cost=9.9, macro_actual_regret_pct=.39))
        rows = reporting.val_screen(config(), {'checkpoint_sha256': 'lock'}, current, refs)
        self.assertEqual([r['reference_role'] for r in rows], ['main', 'secondary'])
        self.assertTrue(all(r['joint_val_pass'] and not r['checkpoint_reselection'] for r in rows))
        current['macro']['macro_top1'] = float('nan')
        with self.assertRaises(ValueError):
            reporting.val_screen(config(), {}, current, refs)

    def test_cpu_admin_rejects_gpu_before_reading_inputs(self):
        with patch.object(reporting, 'locked_inputs') as load:
            for function in (reporting.report, reporting.compare):
                with self.assertRaises(ValueError):
                    function(SimpleNamespace(device='cuda:0'))
            load.assert_not_called()

    def test_lib_uses_only_r93_factory_and_never_builds_pdf(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            args = reporting.parser().parse_args(['lib', '--arm', 'A', '--device', 'cuda:0'])
            lock = {'checkpoint_sha256': 'locked'}
            test = {'identity': {'model_state_sha256': 'model-state'}}
            with patch.object(reporting, 'locked_inputs', return_value=(run, config(), lock, test, {})), \
                    patch.object(reporting.previous, 'validate_nss_reference', return_value={}), \
                    patch.object(reporting.common, 'r80_lib', return_value={'TSPLIB': {}, 'CVRPLIB': {}}) as lib, \
                    patch.object(reporting.shared, 'build_reports') as pdf:
                reporting.lib_stage(args)
            self.assertEqual(lib.call_args.kwargs['factory_module'], 'code.V4.r93_experiment')
            pdf.assert_not_called()
            self.assertEqual(reporting.read_json(run / 'lib_stage_complete.json')['exit_status'], 0)

    def test_distribution_uses_same_axes_native_top1_and_test_before_lib(self):
        import matplotlib.pyplot as plt
        helper = reporting.common.reporting_helpers()
        helper.setup_style()
        pred = helper.Predictions(Path('/saved'), ['BQ', 'T2T'], np.array([0, 0, 1, 1]),
            np.array([0, 1, 1, 0]), np.ones((4, 2)), np.arange(4), None, np.arange(2), None)
        lib = helper.Predictions(Path('/saved-lib'), pred.pool, np.zeros(49, dtype=int),
            np.arange(49) % 2, np.ones((49, 2)), np.arange(49), None, np.arange(2), None)
        panels = helper.make_panels('TSP', pred, lib, pred.pool)
        fig = reporting.common._distribution(helper, 'TSP', pred.pool, panels, 'NEW90k',
            'R93/A', dict(epoch=1, selected_update=1440, checkpoint_sha256='a' * 64))
        self.addCleanup(plt.close, fig)
        self.assertEqual(len(fig.axes), 2)
        self.assertIn('TSP TEST', fig.axes[0].get_title())
        self.assertIn('TSPLIB', fig.axes[1].get_title())
        for axis, data in zip(fig.axes, (pred, lib)):
            self.assertIn(f'\u6a21\u578bTop1={data.top1_pct:.2f}%', axis.get_title())
            self.assertEqual(len(axis.patches), 2 * len(pred.pool))
        self.assertTrue(all(p['top1'] is None for p in panels if p['kind'] == 'true_winner'))
        self.assertTrue(any('R93/A' in text.get_text() for text in fig.texts))


if __name__ == '__main__':
    unittest.main()

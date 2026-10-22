"""Small dummy CPU witnesses only: no real data preparation, GPU or TabICL."""

from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from . import r98_experiment as experiment
from .test_r98_reporting import arrays, baseline


class ExperimentTests(unittest.TestCase):
    def test_exact_cli_defaults_and_root_without_arm(self):
        for command in ('prepare', 'run', 'report'):
            args = experiment.normalize_args(experiment.parser().parse_args([command, '--root', '/tmp/r98-test']))
            self.assertEqual(args.root, Path('/tmp/r98-test'))
            self.assertEqual(args.target_root, args.root / 'targets')
            self.assertEqual(args.protocol_lock, args.root / 'protocol_lock.json')
            self.assertNotEqual(args.context_root, args.root / 'context')
        args = experiment.parser().parse_args(['run', '--device', 'cuda:0'])
        self.assertEqual(args.device, 'cuda:0')
        with self.assertRaises(SystemExit):
            experiment.parser().parse_args(['run', '--resume'])

    def test_raw_gap_prediction_preserves_native_metrics_and_negative_values(self):
        values = arrays()
        gaps = values['expected_gaps'].copy()
        gaps[0] = [-1e-300, -2e-300]
        raw = {key: values[key] for key in ('costs', 'winner', 'pool', 'pool_ids')}
        config = dict(candidate_pool_ids={'TSP': [2, 7]}, baselines={'TSP': baseline()})
        metrics, data = experiment.saved_prediction('TSP', gaps, raw,
            values['release_source_indices'], values['nodes'], config)
        self.assertEqual(data['pred'][0], 1)
        self.assertEqual(metrics['top1'], .75)
        self.assertEqual(metrics['sbs_selection_split'], 'train')
        np.testing.assert_array_equal(data['expected_gaps'], gaps)
        np.testing.assert_array_equal(data['probabilities'][0], [.5, .5])
        self.assertLess(data['expected_gaps'][0, 1], 0)

    def test_test_lock_precedes_any_query_feature_access(self):
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(root=Path(directory))
            with self.assertRaises(FileNotFoundError) as error:
                experiment.query_inputs('TSP', 'test', args, {}, None)
            self.assertIn('final_test_lock.json', str(error.exception))

    def test_guarded_scalar_fit_and_mean_receive_only_historical_targets_and_features(self):
        events = []

        class Dummy:
            def fit(self, x, y, *, kv_cache):
                events.append(('fit', len(x), y.copy(), kv_cache))

            def predict(self, features, *, output_type):
                events.append(('predict', features.copy(), output_type))
                return np.array([-.1, .2])

        model = SimpleNamespace(call_records=[])
        config = dict(rows_per_problem=dict(train=4))

        def guarded(function, config, torch, **kwargs):
            self.assertTrue(kwargs['flash'])
            return function(), dict(seconds=.01)

        wrapped = experiment.GuardedRegressor(Dummy(), 1, model, config, None)
        with patch.object(experiment, 'guarded_call', side_effect=guarded):
            wrapped.fit(np.zeros((4, 2)), np.arange(4, dtype=np.float64), kv_cache=False)
            pred = wrapped.predict(np.ones((2, 2)), output_type='mean')
        np.testing.assert_array_equal(pred, [-.1, .2])
        self.assertEqual(events[0][0:2], ('fit', 4))
        self.assertFalse(events[0][-1])
        self.assertEqual(events[1][-1], 'mean')
        self.assertEqual([r['stage'] for r in model.call_records], ['fit_method', 'predict_method'])
        with self.assertRaisesRegex(ValueError, 'all historical'):
            wrapped.fit(np.zeros((3, 2)), np.zeros(3))

    def test_predict_checks_all_nonconstant_method_calls_and_overlap(self):
        context = SimpleNamespace(source_ids=np.array([1, 2]), y=np.array([0, 1]),
                                  pool=('alpha', 'beta'), pool_ids=(2, 7))
        model = SimpleNamespace(context=context, constant_method_slots=[0], call_records=[])

        def predict(x, ids):
            model.call_records.extend([dict(stage=stage, method_slot=1) for stage in ('fit_method', 'predict_method')])
            return np.array([[0., -.02], [0., .03]])

        model.predict_features = predict
        gaps, timing = experiment.predict(model, np.ones((2, 2)), np.array([100, 101]), {}, None)
        self.assertLess(gaps[0, 1], 0.)
        self.assertEqual(timing['sequential_fullcontext_fits'], 1)
        self.assertTrue(timing['includes_all_sequential_method_fits'])
        with self.assertRaisesRegex(ValueError, 'overlap'):
            experiment.predict(model, np.ones((2, 2)), np.array([1, 101]), {}, None)
        model.call_records = []
        model.predict_features = lambda x, ids: np.zeros((2, 2))
        with self.assertRaisesRegex(ValueError, 'one sequential'):
            experiment.predict(model, np.ones((2, 2)), np.array([100, 101]), {}, None)

    def test_evaluation_predicts_before_opening_labels(self):
        events = []
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            feature_root = root / 'features'
            (feature_root / 'TRAIN_VAL').mkdir(parents=True)
            np.save(feature_root / 'TRAIN_VAL/TSP.npy', np.full((81004, 2), 20., np.float32))
            args = SimpleNamespace(root=root, feature_root=feature_root)
            config = dict(candidate_pool_ids={'TSP': [2, 7]}, baselines={'TSP': baseline()},
                          public_checkpoint_sha256='a'*64)
            model = SimpleNamespace(call_records=[], context=SimpleNamespace(y=np.zeros(2)))
            values = arrays()
            raw = {key: values[key] for key in ('costs', 'winner', 'pool', 'pool_ids')}

            def inputs(*args):
                events.append('inputs')
                return np.zeros((4, 2)), values['release_source_indices']

            def predict(*args):
                events.append('predict')
                self.assertNotIn('labels', events)
                return values['expected_gaps'], dict(stage='predict', seconds=.01)

            def labels(*args):
                events.append('labels')
                witness = experiment.read_json(root / 'resource_witness.json')
                self.assertFalse(witness['labels_loaded'])
                return raw

            with patch.object(experiment, 'PROBLEMS', ['TSP']), \
                    patch.object(experiment, 'fitted', return_value=(model, dict(seconds=.01))), \
                    patch.object(experiment, 'query_inputs', side_effect=inputs), \
                    patch.object(experiment, 'predict', side_effect=predict), \
                    patch.object(experiment.previous, 'native_labels', side_effect=labels), \
                    patch.object(experiment, 'close_model'):
                result = experiment.evaluate_split('val', args, config, None, [])
            self.assertEqual(events, ['inputs', 'predict', 'labels'])
            self.assertTrue(result['complete'])
            self.assertEqual(result['optimizer_updates'], 0)
            with np.load(root / 'evaluation/val/selected_best/rank000.npz') as saved:
                self.assertIn('TSP__expected_gaps', saved.files)

    def test_dummy_run_full_pipeline_order_and_offline_receipt(self):
        events = []
        with tempfile.TemporaryDirectory() as directory:
            args = experiment.normalize_args(experiment.parser().parse_args([
                'run', '--root', directory, '--device', 'cuda:0']))
            config = dict(experiment='R98', contexts={}, targets={}, public_checkpoint_sha256='a'*64,
                inference_rule=experiment.INFERENCE_RULE,
                **{key: str(getattr(args, key)) for key in ('data_root', 'prepared_root', 'feature_root', 'target_root')})
            tracker = SimpleNamespace(id='synthetic', dir=str(args.root), log=lambda data: None,
                                      finish=lambda: events.append('wandb_finish'))

            def split(name, *args):
                events.append(name)
                if name == 'test':
                    experiment.require_test_lock(args[0].root, config)
                return dict(macro=dict(macro_top1=.5))

            def bind(args, config):
                experiment.require_test_lock(args.root, config)
                self.assertEqual(events, ['val'])
                events.append('bind_test_inputs')

            def libs(*args):
                events.append('LIB')
                return dict(TSPLIB={}, CVRPLIB={})

            fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True))
            fake_wandb = SimpleNamespace(init=lambda **kwargs: tracker)
            with patch.dict(sys.modules, torch=fake_torch, wandb=fake_wandb), \
                    patch.object(experiment, 'validate_protocol', return_value=config), \
                    patch.object(experiment, 'evaluate_split', side_effect=split), \
                    patch.object(experiment.previous, 'lock_test_inputs', side_effect=bind), \
                    patch.object(experiment, 'evaluate_libs', side_effect=libs):
                experiment.run(args)
            self.assertEqual(events, ['val', 'bind_test_inputs', 'test', 'LIB', 'wandb_finish'])
            self.assertTrue(experiment.read_json(args.root / 'run_complete.json')['complete'])
            self.assertEqual(experiment.read_json(args.root / 'wandb_receipt.json')['mode'], 'offline')
            with self.assertRaisesRegex(RuntimeError, 'Preserve existing R98'):
                experiment.ensure_fresh_run(args.root)


if __name__ == '__main__':
    unittest.main()

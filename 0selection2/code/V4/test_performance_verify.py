import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from .performance_experiment import snapshot
from .performance_verify import assert_metrics, check_checkpoint_record, check_history, check_provenance, check_run_identity, declared_runs, verify_predictions


class ReplayGuards(unittest.TestCase):
    def budget(self, root):
        value = dict(seeds=[2], groups=['A', 'B'], expected_runs=2, epochs_per_run=60)
        (root / 'run_budget.json').write_text(json.dumps(value))

    def completed(self, root, name):
        directory = root / name
        directory.mkdir()
        for filename in ('args.json', 'history.json', 'result.json', 'best.pt', 'last.pt'):
            (directory / filename).touch()

    def test_empty_inventory_cannot_report_success(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.budget(root)
            with self.assertRaises(ValueError):
                verify_predictions(root)
            self.assertFalse((root / 'prediction_replay.json').exists())

    def test_missing_run_checkpoint_and_unexpected_run_fail(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.budget(root)
            self.completed(root, 'winner_cost_seed2')
            with self.assertRaises(ValueError):
                declared_runs(root)
            self.completed(root, 'performance_seed2')
            _, expected = declared_runs(root)
            self.assertEqual(len(expected), 2)
            (root / 'performance_seed2' / 'best.pt').unlink()
            with self.assertRaises(ValueError):
                declared_runs(root)
            (root / 'performance_seed2' / 'best.pt').touch()
            self.completed(root, 'winner_cost_seed3')
            with self.assertRaises(ValueError):
                declared_runs(root)

    def test_duplicate_and_missing_problem_coverage_fail(self):
        history = [dict(epoch=i, updates=i*2, updates_by_problem={'TSP': 2}, per_problem={'TSP': {}}) for i in (1, 2)]
        train = {'TSP': dict(winner=np.array([0, 1, 0, 1]))}
        with patch('code.V4.performance_verify.PROBLEMS', ['TSP']):
            check_history(history, 2, 2, train)
            history[1]['epoch'] = 1
            with self.assertRaises(ValueError):
                check_history(history, 2, 2, train)
            history[1]['epoch'] = 2
            history[1]['per_problem'] = {}
            with self.assertRaises(ValueError):
                check_history(history, 2, 2, train)

    def test_successful_update_counts_are_checked(self):
        history = [dict(epoch=1, updates=3, updates_by_problem={'TSP': 2}, per_problem={'TSP': {}})]
        with patch('code.V4.performance_verify.PROBLEMS', ['TSP']):
            with self.assertRaises(ValueError):
                check_history(history, 1, 2, {'TSP': dict(winner=np.zeros(4))})

    def test_diagnostic_and_macro_key_omissions_fail(self):
        with self.assertRaises(ValueError):
            assert_metrics({'ce': 1., 'top1_performance': .5}, {'ce': 1.})
        with self.assertRaises(AssertionError):
            assert_metrics({'macro_top1': .5}, {'macro_top1': .6})

    def test_result_group_seed_and_epoch_mislabeling_fail(self):
        config = dict(group='A', seed=2, epochs=60)
        check_run_identity(config, config.copy(), 'A', 2, 60)
        for key, value in (('group', 'B'), ('seed', 3), ('epochs', 59)):
            with self.subTest(key=key), self.assertRaises(ValueError):
                check_run_identity(config, {**config, key: value}, 'A', 2, 60)

    def test_checkpoint_is_bound_to_selected_record_and_run(self):
        config = dict(group='A', seed=2, epochs=60)
        best = dict(epoch=53, updates=14310, val={'macro_top1': .48})
        final = dict(epoch=60, updates=16200, val={'macro_top1': .47})
        result = dict(best=best, final=final)
        checkpoint = dict(epoch=52, successful_updates=14310, args=config, macro=best['val'])
        self.assertEqual(check_checkpoint_record(checkpoint, 'best.pt', config, result), best)
        last = dict(epoch=59, successful_updates=16200, args=config, macro=final['val'])
        self.assertEqual(check_checkpoint_record(last, 'last.pt', config, result), final)
        with self.assertRaises(ValueError):
            check_checkpoint_record(last, 'best.pt', config, result)
        for key, value in (('successful_updates', 1), ('args', {**config, 'seed': 3})):
            with self.subTest(key=key), self.assertRaises(ValueError):
                check_checkpoint_record({**checkpoint, key: value}, 'best.pt', config, result)
        with self.assertRaises(AssertionError):
            check_checkpoint_record({**checkpoint, 'macro': final['val']}, 'best.pt', config, result)

    def test_scales_data_and_launch_hashes_are_independent_checks(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            snapshot(root)
            entry = dict(costs=np.array([[1., 3.], [2., 4.]], dtype=np.float64), winner=np.array([0, 0]),
                         pool=['a', 'b'], pool_ids=[0, 1], label_hash='labels', data_hash='instances')
            identities = {key: entry[key] for key in ('pool', 'pool_ids', 'label_hash', 'data_hash')}
            manifest = {s: {'TSP': identities} for s in ('train', 'val')}
            (root / 'data_manifest.json').write_text(json.dumps(manifest))
            scales = {'TSP': dict(scale=1., fitted_on='train', instances=2, **identities)}
            data = {'TSP': entry}
            with patch('code.V4.performance_verify.PROBLEMS', ['TSP']):
                check_provenance(root, data, data, scales)
                scales['TSP']['scale'] = 2.
                with self.assertRaises(AssertionError):
                    check_provenance(root, data, data, scales)
                scales['TSP']['scale'] = 1.
                scales['TSP']['label_hash'] = 'different'
                with self.assertRaises(ValueError):
                    check_provenance(root, data, data, scales)
                scales['TSP']['label_hash'] = 'labels'
                (root / 'source_launch' / 'dual_stream.py').write_text('changed')
                with self.assertRaises(ValueError):
                    check_provenance(root, data, data, scales)


if __name__ == '__main__':
    unittest.main()

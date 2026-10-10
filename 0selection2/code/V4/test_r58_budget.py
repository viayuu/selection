"""Synthetic CPU-only fixtures for R58 fresh-solve budget accounting."""

import json
import tempfile
import unittest
from argparse import Namespace
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import numpy as np

from . import r58_budget
from .r58_scenario import CONTRACT, canonical_hash, file_hash, save_json


HASHES = {'fixture.py': 'current'}


def distribution(sizes):
    return dict(sizes=sizes, count=len(sizes), min=min(sizes), max=max(sizes),
                histogram=dict(sorted(Counter(sizes).items())), input_sha256='train')


def profile(problem='TSP', method='BQ', observations=((0, 10, 1.), (1, 20, 3.))):
    return dict(problem=problem, solver=method, split='train', passed=True,
                deployment={'fixture': True}, implementation_sha256=HASHES.copy(),
                contract_sha256=canonical_hash(CONTRACT), input_sha256='train',
                results=[dict(index=i, true_size=n, seconds=t) for i, n, t in observations])


class BudgetTests(unittest.TestCase):
    def test_true_size_weighting_and_conservative_reference(self):
        train = distribution([10, 20] + [15] * 8 + [10] * 10)
        estimate = r58_budget.estimate_profile(profile(), train, 1)
        expected_mean = (11 * 1 + 8 * 2 + 1 * 3) / 20
        reference_mean = (11 * 1 + 8 * 3 + 1 * 3) / 20
        self.assertAlmostEqual(estimate['seconds_per_instance'], expected_mean)
        self.assertNotEqual(estimate['seconds_per_instance'], 2.)
        self.assertAlmostEqual(estimate['splits']['train']['seconds'], expected_mean * 10000)
        self.assertAlmostEqual(estimate['fresh_solve_gpu_hours'], expected_mean * 12000 / 3600)
        self.assertAlmostEqual(estimate['conservative_reference_gpu_hours'], reference_mean * 12000 / 3600)

    def test_rf_warm_not_cold_and_not_batch_probe(self):
        train = distribution([10, 20])
        record = profile('CVRP', 'RouteFinder')
        for row in record['results']:
            row['first_call_seconds'] = 10000.
        record['batch_probe'] = {'seconds_per_instance': .0001}
        estimate = r58_budget.estimate_profile(record, train, 1)
        self.assertEqual(estimate['seconds_per_instance'], 2.)
        self.assertAlmostEqual(estimate['fresh_solve_gpu_hours'], 12000 * 2 / 3600)
        del record['results'][0]['seconds']
        with self.assertRaisesRegex(ValueError, 'cold timing is not a fallback'):
            r58_budget.estimate_profile(record, train, 1)

    def test_stale_contract_input_split_and_failure_excluded(self):
        train = distribution([10, 20])
        changes = [({'implementation_sha256': {'fixture.py': 'old'}}, 'stale_implementation'),
                   ({'contract_sha256': 'old'}, 'stale_contract'),
                   ({'input_sha256': 'old'}, 'stale_train_input'),
                   ({'split': 'val'}, 'wrong_problem_solver_or_split'),
                   ({'solver': 'OTHER'}, 'wrong_problem_solver_or_split'),
                   ({'passed': False}, 'not_passed'),
                   ({'deployment': {}}, 'missing_deployment_profile')]
        for change, expected in changes:
            with self.subTest(change=change):
                record = profile()
                record.update(change)
                self.assertEqual(r58_budget.qualification_reason(record, 'TSP', 'BQ', HASHES, train), expected)

    def test_no_size_extrapolation(self):
        for sizes in ([10, 20, 9], [10, 20, 21]):
            with self.subTest(sizes=sizes), self.assertRaisesRegex(ValueError, 'extrapolation refused'):
                r58_budget.estimate_profile(profile(), distribution(sizes), 1)
        single = profile(observations=((0, 10, 1.),))
        self.assertEqual(r58_budget.estimate_profile(single, distribution([10] * 4), 1)['seconds_per_instance'], 1.)
        with self.assertRaisesRegex(ValueError, 'extrapolation refused'):
            r58_budget.estimate_profile(single, distribution([10, 11]), 1)

    def test_bad_timing_and_wrong_size_refused(self):
        train = distribution([10, 20])
        for seconds in (None, 0, -1, float('nan'), float('inf'), True):
            record = profile()
            record['results'][0]['seconds'] = seconds
            with self.subTest(seconds=seconds), self.assertRaises(ValueError):
                r58_budget.estimate_profile(record, train, 1)
        record = profile()
        record['results'][0]['true_size'] = 99
        with self.assertRaisesRegex(ValueError, 'does not match original TRAIN'):
            r58_budget.estimate_profile(record, train, 1)

    def test_batch_uses_actual_max_size_measurement_and_size_group_tails(self):
        sizes = [50] * 5001 + [100] * 4999
        record = profile('VRPB', 'MTPOMO', ((0, 50, 800.), (5001, 100, 900.)))
        record['batch_probe'] = dict(indices=list(range(5001, 5017)), seconds=16., seconds_per_instance=1.)
        estimate = r58_budget.estimate_profile(record, distribution(sizes), 16)
        self.assertEqual(estimate['model'], 'worst_scale_batch_upper_ish_point')
        self.assertEqual(estimate['seconds_per_instance'], 1.)
        self.assertEqual(estimate['splits']['train']['projected_batches'], 626)
        self.assertEqual(estimate['splits']['val']['projected_batches'], 64)
        self.assertEqual(estimate['splits']['test']['projected_batches'], 64)
        self.assertEqual(estimate['splits']['train']['seconds'], 626 * 16)
        self.assertAlmostEqual(estimate['fresh_solve_gpu_hours'], (626 + 64 + 64) * 16 / 3600)

    def test_batch_at_wrong_scale_partial_missing_or_inconsistent_refused(self):
        train = distribution([50] * 16 + [100] * 16)
        record = profile('VRPB', 'MTPOMO', ((0, 50, 1.), (16, 100, 3.)))
        batches = [{}, dict(indices=list(range(16)), seconds=16., seconds_per_instance=1.),
                   dict(indices=list(range(16, 31)), seconds=16., seconds_per_instance=1.),
                   dict(indices=list(range(16, 32)), seconds=16., seconds_per_instance=2.)]
        for batch in batches:
            with self.subTest(batch=batch), self.assertRaises(ValueError):
                r58_budget.estimate_profile(dict(record, batch_probe=batch), train, 16)

    def build_fixture(self, root, methods, records, train):
        for method, record in records.items():
            save_json(root / 'preflight' / f'TSP__{method}.json', record)
        with patch.object(r58_budget, 'PROBLEMS', ['TSP']), \
             patch.object(r58_budget, 'POOLS', {'TSP': methods}), \
             patch.object(r58_budget, 'implementation_hashes', return_value=HASHES), \
             patch.object(r58_budget, 'train_distribution', return_value=train):
            return r58_budget.build_budget(root)

    def test_stale_never_in_current_subtotal_and_incomplete_not_full_estimate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stale = profile(method='OLD', observations=((0, 10, 99999.), (1, 20, 99999.)))
            stale['implementation_sha256'] = {'fixture.py': 'stale'}
            report = self.build_fixture(root, ['BQ', 'OLD', 'MISSING'], {'BQ': profile(), 'OLD': stale},
                                        distribution([10, 20]))
            self.assertEqual(report['coverage']['current_passing_deployments'], 1)
            self.assertEqual(report['coverage']['estimated_deployments'], 1)
            self.assertFalse(report['coverage']['complete_budget'])
            self.assertAlmostEqual(report['budget']['current_profile_subtotal_gpu_hours'], 2 * 12000 / 3600)
            self.assertIsNone(report['budget']['all_deployments_gpu_hours'])
            self.assertEqual(report['budget']['all_deployments_status'], 'unknown_incomplete_coverage')
            self.assertEqual(report['budget']['decision'], 'not_cleared_incomplete_coverage')
            self.assertIsNone(report['deployments'][1]['estimate'])
            self.assertIn('stale_implementation', r58_budget.markdown(report))

    def test_over_100h_current_subset_triggers_skip_without_claiming_complete(self):
        with tempfile.TemporaryDirectory() as directory:
            expensive = profile(observations=((0, 10, 31.), (1, 20, 31.)))
            report = self.build_fixture(Path(directory), ['BQ', 'MISSING'], {'BQ': expensive}, distribution([10, 20]))
            self.assertEqual(report['budget']['current_profile_point_status'], '>100h')
            self.assertIsNone(report['budget']['all_deployments_gpu_hours'])
            self.assertEqual(report['budget']['decision'], 'skip_full_fresh_solve_projected_over_100h')

    def test_all_128_coverage_and_exact_100h_status(self):
        with tempfile.TemporaryDirectory() as directory:
            methods = [f'method{i}' for i in range(128)]
            seconds = 100 * 3600 / (12000 * 128)
            records = {m: profile(method=m, observations=((0, 10, seconds), (1, 20, seconds))) for m in methods}
            report = self.build_fixture(Path(directory), methods, records, distribution([10, 20]))
            self.assertTrue(report['coverage']['all_current_passing'])
            self.assertTrue(report['coverage']['complete_budget'])
            self.assertEqual(report['coverage']['estimated_deployments'], 128)
            self.assertEqual(report['budget']['all_deployments_status'], '<=100h')
            self.assertAlmostEqual(report['budget']['all_deployments_gpu_hours'], 100.)
            self.assertEqual(report['projected_solver_instance_evaluations'], 1536000)

    def test_passing_profile_with_missing_timing_reported_separately(self):
        with tempfile.TemporaryDirectory() as directory:
            report = self.build_fixture(Path(directory), ['BQ'], {'BQ': profile(observations=())}, distribution([10, 20]))
            self.assertTrue(report['coverage']['all_current_passing'])
            self.assertEqual(report['coverage']['estimated_deployments'], 0)
            self.assertEqual(len(report['coverage']['missing_timing_coverage']), 1)
            self.assertFalse(report['coverage']['complete_budget'])
            self.assertIsNone(report['budget']['all_deployments_gpu_hours'])

    def test_original_train_loader_only_and_no_label_field_access(self):
        class PhysicalOnly(dict):
            def get(self, key, *args):
                if key in ('cost', 'costs', 'ind', 'winner'):
                    raise AssertionError('label access')
                return super().get(key, *args)

        item = PhysicalOnly(loc=np.zeros((1, 7, 2)), depot=np.zeros((1, 1, 2)), demand=np.zeros((1, 7)))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'CVRPtrain' / 'dataset.pkl'
            path.parent.mkdir()
            path.write_bytes(b'train fixture')
            with patch.object(r58_budget, 'load_instances', return_value=([item], path)) as load:
                train = r58_budget.train_distribution('CVRP')
            load.assert_called_once_with('CVRP', 'train')
            self.assertEqual(train['histogram'], {7: 1})
            self.assertEqual(train['input_sha256'], file_hash(path))

    def test_cli_writes_only_budget_artifacts_without_touching_preflight(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = self.build_fixture(root, ['BQ'], {'BQ': profile()}, distribution([10, 20]))
            original = (root / 'preflight/TSP__BQ.json').read_bytes()
            output = root / 'reports'
            with patch.object(r58_budget, 'build_budget', return_value=report), \
                 patch('torch.cuda.init', side_effect=AssertionError('GPU access')), \
                 patch('torch.cuda.is_available', side_effect=AssertionError('GPU query')), \
                 patch('builtins.print'):
                self.assertEqual(r58_budget.main(['--root', str(root), '--output-dir', str(output)]), 0)
            self.assertEqual(sorted(p.name for p in output.iterdir()), ['BUDGET.md', 'budget.json'])
            self.assertEqual((root / 'preflight/TSP__BQ.json').read_bytes(), original)
            saved = json.loads((output / 'budget.json').read_text())
            self.assertEqual(saved['coverage'], report['coverage'])
            text = (output / 'BUDGET.md').read_text()
            self.assertIn('ESTIMATED FROM TRAIN', text)
            self.assertIn('does not certify total R58 work <=100h', text)
            self.assertNotIn('\n+ ', text)

    def test_cpu_only_cli_end_to_end_reads_train_not_val_or_test(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'original/TSPtrain/dataset.pkl'
            source.parent.mkdir(parents=True)
            source.write_bytes(b'original TRAIN fixture')
            record = profile()
            record['input_sha256'] = file_hash(source)
            save_json(root / 'preflight/TSP__BQ.json', record)
            output = root / 'budget'
            with patch.object(r58_budget, 'PROBLEMS', ['TSP']), \
                 patch.object(r58_budget, 'POOLS', {'TSP': ['BQ']}), \
                 patch.object(r58_budget, 'implementation_hashes', return_value=HASHES), \
                 patch.object(r58_budget, 'load_instances', return_value=(
                     [np.zeros((1, 10, 2)), np.zeros((1, 20, 2))], source)) as load, \
                 patch('torch.cuda.init', side_effect=AssertionError('GPU access')), \
                 patch('torch.cuda.is_available', side_effect=AssertionError('GPU query')), \
                 patch('code.V4.r58_labels.make_backend', side_effect=AssertionError('solver launch')), \
                 patch('builtins.print'):
                self.assertEqual(r58_budget.main(['--root', str(root), '--output-dir', str(output)]), 0)
            load.assert_called_once_with('TSP', 'train')
            saved = json.loads((output / 'budget.json').read_text())
            self.assertTrue(saved['coverage']['complete_budget'])
            self.assertEqual(saved['projection_counts'], {'train': 10000, 'val': 1000, 'test': 1000})
            self.assertEqual(saved['train_distributions']['TSP']['histogram'], {'10': 1, '20': 1})
            self.assertEqual(saved['train_distributions']['TSP']['path'], str(source))
            self.assertAlmostEqual(saved['budget']['all_deployments_gpu_hours'], 2 * 12000 / 3600)

    def test_source_change_during_snapshot_refused(self):
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(r58_budget, 'PROBLEMS', ['TSP']), \
             patch.object(r58_budget, 'POOLS', {'TSP': []}), \
             patch.object(r58_budget, 'train_distribution', return_value=distribution([10])), \
             patch.object(r58_budget, 'implementation_hashes', side_effect=[HASHES, {'fixture.py': 'changed'}]):
            with self.assertRaisesRegex(RuntimeError, 'changed while estimating'):
                r58_budget.build_budget(Path(directory))

    def assert_pipeline_budget_blocked(self, complete, expected_status):
        from . import r58_pipeline

        for stage in ('all', 'labels'):
            for locked in (False, True):
                with self.subTest(stage=stage, existing_lock=locked), tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    records = {'BQ': profile(observations=((0, 10, 60.), (1, 20, 60.)))}
                    if not complete:
                        records['MISSING_TIMING'] = profile(method='MISSING_TIMING', observations=())
                    for record in records.values():
                        record['runtime_environment'] = {'fixture': True}
                    report = self.build_fixture(root, list(records), records, distribution([10, 20]))
                    # An incomplete over-budget subset must still be marked budget_not_cleared.
                    self.assertEqual(report['coverage']['complete_budget'], complete)
                    self.assertEqual(report['budget']['decision'], 'skip_full_fresh_solve_projected_over_100h')
                    lock_path = root / 'deployments.lock.json'
                    if locked:
                        save_json(lock_path, {'fixture': 'existing lock'})
                    original_lock = lock_path.read_bytes() if locked else None
                    args = Namespace(root=root, stage=stage, force_preflight=False)
                    with patch.object(r58_pipeline, 'PROBLEMS', ['TSP']), \
                         patch.object(r58_pipeline, 'POOLS', {'TSP': list(records)}), \
                         patch.object(r58_pipeline, 'implementation_hashes', return_value=HASHES), \
                         patch.object(r58_pipeline, 'runtime_environment', return_value={'fixture': True}), \
                         patch.object(r58_pipeline, 'progress', return_value={'complete_preflight': True}) as progress, \
                         patch.object(r58_pipeline, 'build_budget', return_value=report) as build, \
                         patch.object(r58_pipeline, 'lock') as lock, \
                         patch.object(r58_pipeline.subprocess, 'run', side_effect=AssertionError('command launched')) as command, \
                         patch.object(r58_pipeline, 'publish') as publish, \
                         patch.object(r58_pipeline, 'summarize') as summarize:
                        self.assertEqual(r58_pipeline.run(args), 3)
                        build.assert_called_once_with(root)
                        self.assertEqual(progress.call_count, int(stage == 'all'))
                        lock.assert_not_called()
                        command.assert_not_called()
                        publish.assert_not_called()
                        summarize.assert_not_called()
                    state = json.loads((root / 'pipeline_state.json').read_text())
                    self.assertEqual(state['status'], expected_status)
                    self.assertEqual(state['budget'], report['budget'])
                    self.assertEqual(state['budget_coverage'], report['coverage'])
                    self.assertIn('finished', state)
                    self.assertNotIn('current_command', state)
                    self.assertEqual(json.loads((root / 'budget.json').read_text())['budget'], report['budget'])
                    self.assertEqual((root / 'BUDGET.md').read_text(), r58_budget.markdown(report))
                    self.assertFalse((root / 'logs').exists())
                    if locked:
                        self.assertEqual(lock_path.read_bytes(), original_lock)
                    else:
                        self.assertFalse(lock_path.exists())

    def test_pipeline_over_budget_never_locks_or_generates(self):
        self.assert_pipeline_budget_blocked(True, 'skipped_over_budget')

    def test_pipeline_incomplete_budget_never_locks_or_generates(self):
        self.assert_pipeline_budget_blocked(False, 'budget_not_cleared')


if __name__ == '__main__':
    unittest.main()

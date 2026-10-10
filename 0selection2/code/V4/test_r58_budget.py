"""Synthetic CPU-only fixtures for R58 fresh-solve budget accounting."""

import copy
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
                    report['whole_project'] = dict(complete=complete, accounting_rule=r58_budget.ACCOUNTING_RULE,
                        blockers=[], decision=r58_budget.WHOLE_OVER if complete else 'not_cleared_incomplete_whole_plan',
                        whole_projected_gpu_hours=101., rf_family={'mode': 'normal4'}, components_seconds={})
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
                         patch.object(r58_pipeline, 'build_whole_budget', return_value=report) as build, \
                         patch.object(r58_pipeline, 'write_execution_approval') as approve, \
                         patch.object(r58_pipeline, 'lock') as lock, \
                         patch.object(r58_pipeline.subprocess, 'run', side_effect=AssertionError('command launched')) as command, \
                         patch.object(r58_pipeline, 'publish') as publish, \
                         patch.object(r58_pipeline, 'summarize') as summarize:
                        self.assertEqual(r58_pipeline.run(args), 3)
                        build.assert_called_once_with(root)
                        self.assertEqual(progress.call_count, int(stage == 'all'))
                        lock.assert_not_called()
                        approve.assert_not_called()
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
                    self.assertFalse((root / 'execution_plan.json').exists())
                    self.assertFalse((root / 'execution_approval.json').exists())
                    if locked:
                        self.assertEqual(lock_path.read_bytes(), original_lock)
                    else:
                        self.assertFalse(lock_path.exists())

    def test_pipeline_over_budget_never_locks_or_generates(self):
        self.assert_pipeline_budget_blocked(True, 'skipped_over_budget')

    def test_pipeline_incomplete_budget_never_locks_or_generates(self):
        self.assert_pipeline_budget_blocked(False, 'budget_not_cleared')


class WholeBudgetTests(unittest.TestCase):
    def fixture(self, root):
        from . import r58_timing_refinement as refinement

        scales = (50, 56, 63, 69, 75, 81, 88, 94, 100)
        train = distribution([n for n in scales for _ in range(1111)] + [100])
        for problem in r58_budget.PROBLEMS:
            for method in r58_budget.POOLS[problem]:
                record = profile(problem, method, ((0, 50, .01), (9999, 100, .03)))
                if r58_budget.inference_batch_size(problem, method) == 16:
                    record['batch_probe'] = dict(indices=list(range(9984, 10000)), seconds=.16, seconds_per_instance=.01)
                save_json(root / 'preflight' / f'{problem}__{method}.json', record)
        with patch.object(r58_budget, 'implementation_hashes', return_value=HASHES), \
             patch.object(r58_budget, 'train_distribution', return_value=train):
            base = r58_budget.build_budget(root)
        artifacts = {}
        for name in ('selection', 'run_state', 'summary', 'prerequisite_spent'):
            path = root / 'evidence' / f'{name}.json'
            save_json(path, {'fixture': name})
            artifacts[str(path)] = file_hash(path)
        evidence = dict(estimates=[dict(problem=t['problem'], solver=t['method'],
            mode='normal4' if t['method'] in refinement.RF_METHODS else 'serial1',
            reference_seconds=3900., fresh_solve_seconds=2100., projected_session_overhead_seconds=300.)
            for t in refinement.targets()],
            rf_family=dict(mode='normal4', complete=True, mps_credit=False), spent_seconds=6087.,
            evidence=artifacts,
            preflights={str(p): file_hash(p) for p in (root / 'preflight').glob('*.json')})
        return base, evidence

    def whole(self, root, base, evidence):
        with patch.object(r58_budget, 'build_budget', return_value=copy.deepcopy(base)), \
             patch.object(r58_budget, 'load_refinement', return_value=evidence), \
             patch.object(r58_budget, 'implementation_hashes', return_value=HASHES):
            return r58_budget.build_whole_budget(root)

    def test_fixed_rule_strict_limit_and_nonfinite_spend_refused(self):
        components = r58_budget.whole_components(2 * 3600, 80 * 3600, 3600)
        self.assertEqual(components['whole_projected_gpu_hours'], 96.)
        self.assertEqual(components['components_seconds']['label_contingency'], 8 * 3600)
        self.assertEqual(components['decision'], r58_budget.WHOLE_CLEARED)
        boundary = r58_budget.whole_components(3600, (100 * 3600 - 3600 - 5 * 3600) / 1.10, 0.)
        self.assertAlmostEqual(boundary['whole_projected_gpu_hours'], 100.)
        self.assertEqual(boundary['decision'], r58_budget.WHOLE_OVER)
        for value in (float('nan'), float('inf'), -1, True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                r58_budget.whole_components(value, 3600, 0)

    def test_refined63_overhead_once_and_unchanged65_conservative_no_speedup(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base, evidence = self.fixture(root)
            report = self.whole(root, base, evidence)
            whole = report['whole_project']
            self.assertTrue(whole['complete'])
            refined = {(r['problem'], r['solver']) for r in evidence['estimates']}
            remaining = sum(r['estimate']['conservative_reference_gpu_hours'] * 3600
                            for r in base['deployments'] if (r['problem'], r['solver']) not in refined)
            self.assertAlmostEqual(whole['components_seconds']['remaining_label_reference'], 63 * 3600 + remaining)
            self.assertEqual(whole['components_seconds']['measured63_session_overhead'], 63 * 300)
            self.assertAlmostEqual(whole['whole_reference_seconds'],
                6087 + 1.10 * (63 * 3600 + remaining) + 63 * 300 + 5 * 3600)
            self.assertEqual(whole['execution_plan']['rf_mode'], 'normal4')
            self.assertIn('H63 is charged ONCE', r58_budget.markdown(report))
            self.assertFalse((root / 'execution_plan.json').exists())
            self.assertFalse((root / 'execution_approval.json').exists())

    def test_missing_stale_incomplete_or_changed_roster_never_clears(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base, evidence = self.fixture(root)
            for reason in ('missing refinement', 'stale implementation', 'incomplete63', 'unqualified48'):
                with self.subTest(reason=reason), \
                     patch.object(r58_budget, 'build_budget', return_value=copy.deepcopy(base)), \
                     patch.object(r58_budget, 'load_refinement', side_effect=ValueError(reason)):
                    report = r58_budget.build_whole_budget(root)
                    self.assertFalse(report['whole_project']['complete'])
                    with self.assertRaises(ValueError):
                        r58_budget.write_execution_approval(root, report)
            stale = copy.deepcopy(base)
            stale['coverage']['all_current_passing'] = False
            with patch.object(r58_budget, 'build_budget', return_value=stale), \
                 patch.object(r58_budget, 'load_refinement') as load:
                self.assertFalse(r58_budget.build_whole_budget(root)['whole_project']['complete'])
                load.assert_not_called()
            evidence['estimates'].pop()
            self.assertFalse(self.whole(root, base, evidence)['whole_project']['complete'])
            self.assertFalse((root / 'execution_approval.json').exists())

    def test_completed_raw_passes_profiles_and_max_reference_recomputed(self):
        from . import r58_timing_refinement as refinement
        from .r58_execution import execution_profile, make_execution_plan
        from .test_r58_timing_refinement import raw_group

        anchors = refinement.select_anchors([n for n in (50, 75, 100) for _ in range(8)], 3)
        target = dict(problem='CVRP', method='RouteFinder', anchor_count=3)
        deployment = {'fixture': 'current production'}
        execution = execution_profile('CVRP', 'RouteFinder', make_execution_plan('normal4'))
        selection = dict(datasets={'CVRP': {'anchors': anchors}},
                         profiles={'CVRP__RouteFinder': {'deployment': deployment}})
        passes = []
        for pass_id in (1, 2):
            observations = []
            for anchor in anchors:
                group = dict(id=anchor['id'], indices=anchor['indices'])
                raw = raw_group(group, anchor['true_size'])
                raw.update(deployment_profile=deployment, execution_profile=execution)
                observations.append(dict(raw=raw, outer_seconds=4. + pass_id))
            passes.append(observations)
        measured = dict(complete=True, mode='normal4', worker_exitcodes=[0] * 4, passes=passes,
            prepare=dict(worker_pids=[100, 101, 102, 103], deployment_profile=deployment, execution_profile=execution),
            controls_not_attached=True, control_isolation=dict(existing_mps_processes=[], existing_target_context_pids=[]),
            session_wall_seconds=43.)
        histogram = {50: 4, 75: 12, 100: 8}
        measured['estimate'] = refinement.timing_estimate(histogram, anchors,
            [[g['outer_seconds'] for g in p] for p in passes], 43.)
        estimate = r58_budget.validated_measurement(selection, target, 'normal4', measured, histogram)
        self.assertEqual(estimate['max_pass_reference_seconds_per_instance'], 1.5)
        self.assertEqual(estimate['projected_session_overhead_seconds'], 30.)
        bad = copy.deepcopy(measured)
        bad['estimate']['reference_seconds'] -= 1
        with self.assertRaisesRegex(ValueError, 'actual timed passes'):
            r58_budget.validated_measurement(selection, target, 'normal4', bad, histogram)
        bad = copy.deepcopy(measured)
        bad['passes'][1][0]['raw']['deployment_profile'] = {'fixture': 'stale'}
        with self.assertRaisesRegex(ValueError, 'profile changed'):
            r58_budget.validated_measurement(selection, target, 'normal4', bad, histogram)

    def test_saved63_records_actual_spend_and_summary_checked_end_to_end_cpu_only(self):
        from . import r58_timing_refinement as refinement
        from .r58_execution import execution_profile, make_execution_plan
        from .test_r58_timing_refinement import raw_group

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base, _ = self.fixture(root)
            datasets = copy.deepcopy(base['train_distributions'])
            for problem, dataset in datasets.items():
                if problem != 'ATSP':
                    sizes = [int(n) for n, frequency in dataset['histogram'].items() for _ in range(frequency)]
                    dataset['anchors'] = refinement.select_anchors(sizes, 9 if problem in ('TSP', 'CVRP') else 3)
            selection = dict(implementation_sha256=HASHES, selection_sha256='fixed CPU fixture',
                targets=refinement.targets(), datasets=datasets,
                profiles={f'{r["problem"]}__{r["solver"]}': dict(path=r['profile_path'],
                    sha256=file_hash(r['profile_path']), deployment={'fixture': True}) for r in base['deployments']})
            records = {}
            for target in selection['targets']:
                key = f'{target["problem"]}__{target["method"]}'
                anchors, groups, _ = refinement.groups_for(selection, target)
                records[key] = {}
                for mode in target['modes']:
                    if mode == 'mps4':
                        records[key][mode] = dict(complete=False, error='MPS unavailable; no retry/credit')
                        continue
                    execution = execution_profile(target['problem'], target['method'], make_execution_plan(mode))
                    passes, outer = [], 4. if mode == 'serial1' else 2.
                    for _ in (1, 2):
                        observations = []
                        for anchor, group in zip(anchors, groups):
                            raw = raw_group(group, anchor['true_size'], mode)
                            raw.update(deployment_profile={'fixture': True}, execution_profile=execution)
                            raw['groups'][0]['seconds'] = outer * .75
                            observations.append(dict(raw=raw, outer_seconds=outer))
                        passes.append(observations)
                    session_wall = outer * 2 * len(anchors) + 10
                    measured = dict(complete=True, mode=mode, worker_exitcodes=[0] * execution['workers'],
                        prepare=dict(worker_pids=passes[0][0]['raw']['worker_pids'],
                                     deployment_profile={'fixture': True}, execution_profile=execution),
                        passes=passes, session_wall_seconds=session_wall, controls_not_attached=True,
                        control_isolation=dict(existing_mps_processes=[], existing_target_context_pids=[]))
                    measured['estimate'] = refinement.timing_estimate(datasets[target['problem']]['histogram'],
                        anchors, [[g['outer_seconds'] for g in p] for p in passes], session_wall)
                    records[key][mode] = measured
            original = (404.7381637785584, 181.29578457027674, 629.78885849379, 270.6426247423515)
            entries = [dict(path=f'pilot_{i}', wall_seconds=wall) for i, wall in enumerate(original)]
            entries += [dict(path='revised_preflight_state.json', wall_seconds=200.),
                        dict(path='readonly_guard_receipt.json', wall_seconds=1.)]
            ledger = dict(entries=entries, spent_seconds=sum(e['wall_seconds'] for e in entries))
            prerequisite_path = root / 'prerequisite_spent.json'
            save_json(prerequisite_path, dict(before_additional_pilot_seconds=3894.,
                slurm_outer_overhead_seconds=6.5345684150234,
                completed_additional_pilot_controller_seconds=sum(original)))
            wall = sum(m.get('session_wall_seconds', 0) for modes in records.values() for m in modes.values()) + 10
            state = dict(status='complete', finished=10000., wall_seconds=wall, ledger=ledger, records=records,
                cumulative_wall_seconds=ledger['spent_seconds'] + wall,
                prerequisite_spent=refinement.prerequisite_accounting(prerequisite_path, ledger))
            summary = refinement.summarize(selection, state)
            for name, value in [('selection', selection), ('run_state', state), ('summary', summary)]:
                save_json(root / 'timing_refinement' / f'{name}.json', value)
            with patch.object(refinement, 'verify_selection'), \
                 patch.object(r58_budget, 'build_budget', return_value=copy.deepcopy(base)), \
                 patch.object(r58_budget, 'implementation_hashes', return_value=HASHES), \
                 patch('torch.cuda.init', side_effect=AssertionError('GPU access')), \
                 patch('torch.cuda.is_available', side_effect=AssertionError('GPU query')):
                evidence = r58_budget.load_refinement(root, base)
                self.assertAlmostEqual(evidence['spent_seconds'], 5387 + 200 + 1 + wall)
                self.assertEqual(evidence['rf_family']['mode'], 'normal4')
                report = r58_budget.build_whole_budget(root)
                self.assertTrue(report['whole_project']['complete'])
                self.assertEqual(report['whole_project']['decision'], r58_budget.WHOLE_OVER)
                with self.assertRaises(ValueError):
                    r58_budget.write_execution_approval(root, report)
                summary['whole_r58_spent_seconds'] -= 1
                save_json(root / 'timing_refinement/summary.json', summary)
                self.assertFalse(r58_budget.build_whole_budget(root)['whole_project']['complete'])
            self.assertFalse((root / 'execution_approval.json').exists())

    def test_plan_and_approval_match_core_schema_but_changed_allowances_or_lock_block_writes(self):
        from .r58_execution import read_execution_plan, require_budget_approval

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base, evidence = self.fixture(root)
            report = self.whole(root, base, evidence)
            with patch.object(r58_budget, 'implementation_hashes', return_value=HASHES):
                approval = r58_budget.write_execution_approval(root, report)
            self.assertEqual(require_budget_approval(root, read_execution_plan(root), HASHES), approval)
            original = (root / 'execution_approval.json').read_bytes()
            changed = copy.deepcopy(report)
            changed['whole_project']['components_seconds']['label_contingency'] /= 2
            with patch.object(r58_budget, 'implementation_hashes', return_value=HASHES), self.assertRaises(ValueError):
                r58_budget.write_execution_approval(root, changed)
            save_json(root / 'deployments.lock.json', {'execution_plan': {'old': True}})
            with patch.object(r58_budget, 'implementation_hashes', return_value=HASHES), self.assertRaisesRegex(ValueError, 'immutable lock'):
                r58_budget.write_execution_approval(root, report)
            self.assertEqual((root / 'execution_approval.json').read_bytes(), original)

    def test_pipeline_label_only_report_cannot_approve_or_generate(self):
        from . import r58_pipeline

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base, _ = self.fixture(root)
            args = Namespace(root=root, stage='labels', force_preflight=False)
            with patch.object(r58_pipeline, 'build_whole_budget', return_value=base), \
                 patch.object(r58_pipeline, 'write_execution_approval') as approve, \
                 patch.object(r58_pipeline, 'lock') as lock, \
                 patch.object(r58_pipeline.subprocess, 'run') as generate:
                self.assertEqual(r58_pipeline.run(args), 3)
                approve.assert_not_called()
                lock.assert_not_called()
                generate.assert_not_called()
            self.assertEqual(json.loads((root / 'pipeline_state.json').read_text())['status'], 'budget_not_cleared')

    def test_pipeline_writes_cleared_core_plan_and_approval_before_any_lock_or_generation(self):
        from . import r58_pipeline
        from .r58_execution import read_execution_plan, require_budget_approval

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base, evidence = self.fixture(root)
            report = self.whole(root, base, evidence)
            args = Namespace(root=root, stage='labels', force_preflight=False)
            def assert_cleared(*unused, **kwargs):
                approval = require_budget_approval(root, read_execution_plan(root), HASHES)
                self.assertLess(approval['whole_projected_gpu_hours'], 100)
                return Namespace(returncode=0)
            with patch.object(r58_pipeline, 'PROBLEMS', ['TSP']), \
                 patch.object(r58_pipeline, 'POOLS', {'TSP': ['BQ']}), \
                 patch.object(r58_pipeline, 'build_whole_budget', return_value=report), \
                 patch.object(r58_budget, 'implementation_hashes', return_value=HASHES), \
                 patch.object(r58_pipeline, 'lock', side_effect=assert_cleared) as lock, \
                 patch.object(r58_pipeline.subprocess, 'run', side_effect=assert_cleared) as generate, \
                 patch.object(r58_pipeline, 'publish'), patch.object(r58_pipeline, 'summarize'), \
                 patch('builtins.print'):
                self.assertEqual(r58_pipeline.run(args), 0)
                lock.assert_called_once_with(root)
                self.assertEqual(generate.call_count, 3)


if __name__ == '__main__':
    unittest.main()

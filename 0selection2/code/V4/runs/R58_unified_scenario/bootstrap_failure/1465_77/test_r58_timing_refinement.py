"""Focused CPU fixtures; no model, CUDA context or MPS service is started."""

import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from . import r58_timing_refinement as refinement
from .r58_scenario import save_json


def raw_group(group, n=50, mode='normal4'):
    workers = 1 if mode == 'serial1' else 4
    return dict(worker_pids=list(range(100, 100 + workers)),
        groups=[dict(id=group['id'], indices=group['indices'], seconds=4., resumed=0, solved=4,
                     worker_paths=[f'worker_{i}.json' for i in range(workers)])],
        rows=[dict(index=i, true_size=n, cost=float(i + 1), reported_cost=float(i + 1),
                   route=[0, 1, 2, 0], feasible=True, seed=2, inference_batch=[i], worker=k % workers)
              for k, i in enumerate(group['indices'])])


def qualified(seconds, mode='normal4'):
    group = dict(id='n50', indices=[0, 1, 2, 3])
    return dict(complete=True, controls_not_attached=True, attachment_verified=True,
        estimate=dict(fresh_solve_seconds=seconds),
        passes=[[dict(raw=raw_group(group, mode=mode))] for _ in (1, 2)])


class RefinementTests(unittest.TestCase):
    def test_precommitted_scope_quantiles_determinism_and_disjoint_warmups(self):
        sizes = [n for n in range(50, 59) for _ in range(20)]
        anchors = refinement.select_anchors(sizes, 9)
        self.assertEqual(anchors, refinement.select_anchors(sizes, 9))
        self.assertEqual([a['true_size'] for a in anchors], list(range(50, 59)))
        for anchor in anchors:
            self.assertEqual(len(set(anchor['indices'])), 4)
            self.assertFalse(set(anchor['indices']) & set(anchor['warmup_indices']))
            self.assertTrue(all(sizes[i] == anchor['true_size'] for i in anchor['indices'] + anchor['warmup_indices']))
        targets = refinement.targets()
        self.assertEqual(len(targets), 63)
        self.assertEqual(sum(t['modes'] == ['serial1', 'normal4', 'mps4'] for t in targets), 48)
        self.assertEqual(sum(t['anchor_count'] == 9 for t in targets), 15)
        with self.assertRaisesRegex(ValueError, 'collapse'):
            refinement.select_anchors([50] * 100, 9)
        with self.assertRaisesRegex(ValueError, 'disjoint warmup'):
            refinement.select_anchors([50] * 4 + [60] * 4 + [70] * 4, 3)

    def test_train_weighting_both_passes_max_reference_nested5_and_overhead(self):
        anchors = [dict(true_size=n) for n in range(1, 10)]
        histogram = {n: (92 if n == 2 else 1) for n in range(1, 10)}
        a = [4.] * 9
        b = [12. if n == 2 else 4. for n in range(1, 10)]
        estimate = refinement.timing_estimate(histogram, anchors, [a, b], sum(a + b) + 10.)
        self.assertAlmostEqual(estimate['seconds_per_instance'], 1.92)
        self.assertAlmostEqual(estimate['max_pass_reference_seconds_per_instance'], 2.84)
        self.assertEqual(estimate['nested5_seconds_per_instance'], 1.)
        self.assertAlmostEqual(estimate['fresh_solve_seconds'], 12000 * 1.92 + 30)
        self.assertEqual(estimate['measured_session_overhead_seconds'], 10.)
        self.assertEqual(estimate['pass_seconds_per_instance'], [1., 2.84])
        with self.assertRaisesRegex(ValueError, 'extrapolation'):
            refinement.integrate({10: 1}, [(1, 1.), (9, 2.)])
        with self.assertRaisesRegex(ValueError, 'both complete'):
            refinement.timing_estimate(histogram, anchors, [a], 100.)

    def test_all_four_workers_not_any_exit_and_no_resumed_or_infeasible_cases(self):
        group = dict(id='n50', indices=[0, 1, 2, 3])
        raw = raw_group(group)
        refinement.validate_group(raw, group, 50, 'normal4')
        bad = copy.deepcopy(raw)
        bad['groups'][0]['worker_paths'].pop()
        with self.assertRaisesRegex(ValueError, 'completion'):
            refinement.validate_group(bad, group, 50, 'normal4')
        bad = copy.deepcopy(raw)
        bad['rows'][3]['worker'] = 0
        with self.assertRaisesRegex(ValueError, 'all intended workers'):
            refinement.validate_group(bad, group, 50, 'normal4')
        bad = copy.deepcopy(raw)
        bad['rows'][3]['feasible'] = False
        with self.assertRaisesRegex(ValueError, 'Independent check'):
            refinement.validate_group(bad, group, 50, 'normal4')
        with self.assertRaisesRegex(ValueError, 'attachment proof'):
            refinement.validate_group(raw, group, 50, 'mps4')

    def test_mps_global_not_per_task_fastest_and_requires_both_controls(self):
        expected = [('CVRP', 'RouteFinder'), ('VRPB', 'RouteFinder')]
        records = {f'{p}__{m}': dict(serial1=qualified(200), normal4=qualified(100), mps4=qualified(s, 'mps4'))
                   for (p, m), s in zip(expected, (80, 110))}
        self.assertEqual(refinement.choose_rf_family(records, expected)['mode'], 'mps4')
        records['VRPB__RouteFinder']['mps4']['estimate']['fresh_solve_seconds'] = 130
        self.assertEqual(refinement.choose_rf_family(records, expected)['mode'], 'normal4')
        records['VRPB__RouteFinder']['mps4']['estimate']['fresh_solve_seconds'] = 50
        records['VRPB__RouteFinder']['mps4']['passes'][1][0]['raw']['rows'][0]['cost'] += 1e-12
        self.assertFalse(refinement.choose_rf_family(records, expected)['mps_credit'])
        records.pop('VRPB__RouteFinder')
        self.assertIsNone(refinement.choose_rf_family(records, expected)['mode'])

    def test_cap_charges_all_prior_attempts_new_preflight_and_reserves_shutdown(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            walls = [404.7381637785584, 181.29578457027674, 629.78885849379, 270.6426247423515]
            for name, wall in zip(refinement.PILOT_STATES, walls):
                save_json(root / name, dict(status='complete', finished=12345., wall_seconds=wall,
                                           combined_single_gpu_wall_seconds=99999.))
            preflight = root / 'new_preflight.json'
            save_json(preflight, dict(status='preflight_complete', started=100., finished=300.))
            ledger = refinement.spent_ledger(root, preflight)
            self.assertAlmostEqual(ledger['spent_seconds'], 1686.4654315849766)
            self.assertAlmostEqual(ledger['measurement_seconds'], 8993.534568415023)
            with self.assertRaisesRegex(ValueError, 'Duplicate'):
                refinement.spent_ledger(root, preflight, [preflight])
            save_json(preflight, dict(status='preflight_complete', started=100., finished=11000.))
            with self.assertRaisesRegex(ValueError, 'exhausted'):
                refinement.spent_ledger(root, preflight)
        self.assertEqual(refinement.timeout_for(100., receives=2, now=80.), 10.)
        with self.assertRaises(TimeoutError):
            refinement.timeout_for(100., now=100.)

    def test_runner_uses_real_production_api_one_persistent_session_two_passes(self):
        from . import r58_execution

        anchors = refinement.select_anchors([n for n in (50, 60, 70) for _ in range(8)], 3)
        dataset = dict(path='/original/TRAIN/dataset.pkl', input_sha256='train', anchors=anchors,
                       histogram={50: 8, 60: 8, 70: 8})
        selection = dict(datasets={'CVRP': dataset}, profiles={'CVRP__RouteFinder': {'deployment': {'current': True}}},
            selection_sha256='selection', implementation_sha256={'current.py': 'hash'},
            execution_sha256='execution', refiner_sha256='refiner', contract_sha256='contract')
        target = dict(problem='CVRP', method='RouteFinder', anchor_count=3)
        with tempfile.TemporaryDirectory() as temporary, \
             patch.object(r58_execution, 'ExecutionSession') as constructor, \
             patch.object(r58_execution.PrivateMPS, 'check_machine', return_value={'existing_mps_processes': []}):
            session = constructor.return_value
            session.__enter__.return_value = session
            session.prepare.return_value = {'worker_pids': [100, 101, 102, 103]}
            session.processes = [SimpleNamespace(exitcode=0) for _ in range(4)]
            def run_groups(groups, pass_id):
                group = groups[0]
                n = next(a['true_size'] for a in anchors if a['id'] == group['id'])
                return raw_group(group, n)
            session.run_groups.side_effect = run_groups
            result = refinement.measure_target(selection, target, 'normal4', temporary, float('inf'))
            self.assertTrue(result['complete'], result.get('error'))
            self.assertEqual(constructor.call_count, 1)
            self.assertEqual(session.prepare.call_count, 1)
            self.assertEqual(session.run_groups.call_count, 6)
            self.assertEqual([c.kwargs['pass_id'] for c in session.run_groups.call_args_list], [1, 1, 1, 2, 2, 2])
            self.assertEqual(constructor.call_args.kwargs['input_path'], dataset['path'])
            self.assertEqual(constructor.call_args.kwargs['expected_profile'], {'current': True})
            self.assertNotIn('instances', constructor.call_args.kwargs)

    def test_prerequisite_spend_deduplicates_pilot_and_does_not_change_three_hour_cap(self):
        ledger = dict(entries=[{'wall_seconds': v} for v in (404.7381637785584, 181.29578457027674,
            629.78885849379, 270.6426247423515)])
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'prerequisite_spent.json'
            receipt = dict(before_additional_pilot_seconds=3894,
                slurm_outer_overhead_seconds=6.5345684150234,
                completed_additional_pilot_controller_seconds=1486.4654315849766,
                total_before_final_preflight_seconds=5387)
            save_json(path, receipt)
            result = refinement.prerequisite_accounting(path, ledger)
            self.assertAlmostEqual(result['add_to_refinement_ledger_seconds'], 3900.5345684150234)
            self.assertNotEqual(result['add_to_refinement_ledger_seconds'], 5387)
            receipt['completed_additional_pilot_controller_seconds'] = 100.
            save_json(path, receipt)
            with self.assertRaisesRegex(ValueError, 'deduplication'):
                refinement.prerequisite_accounting(path, ledger)
        self.assertEqual(refinement.CAP_SECONDS, 10800.)

    def test_expired_allowance_never_constructs_session(self):
        from . import r58_execution

        anchors = refinement.select_anchors([n for n in (50, 60, 70) for _ in range(8)], 3)
        selection = dict(datasets={'CVRP': dict(path='TRAIN', input_sha256='train', anchors=anchors)},
            profiles={'CVRP__RouteFinder': {'deployment': {'current': True}}},
            selection_sha256='selection', implementation_sha256={}, execution_sha256='execution',
            refiner_sha256='refiner', contract_sha256='contract')
        with tempfile.TemporaryDirectory() as temporary, \
             patch.object(r58_execution, 'ExecutionSession') as constructor, \
             patch.object(r58_execution.PrivateMPS, 'check_machine', return_value={}):
            result = refinement.measure_target(selection,
                dict(problem='CVRP', method='RouteFinder', anchor_count=3), 'normal4', temporary, 0.)
            self.assertFalse(result['complete'])
            constructor.assert_not_called()


if __name__ == '__main__':
    unittest.main()

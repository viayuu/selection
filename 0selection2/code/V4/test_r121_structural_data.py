"""CPU-only family/schema tests. No solver, model, CUDA probe or historical labels."""

import copy
import unittest
from unittest.mock import patch

import numpy as np

from . import r121_structural_data as d


def fixture(problem, seed):
    rng = np.random.default_rng(seed)
    n = 20
    xy = rng.uniform(.1, .9, (n, 2)).astype(np.float32)
    if problem == 'TSP':
        return xy[None].tolist()
    if problem == 'ATSP':
        return d.closure(rng.uniform(.1, 1., (n, n))).astype(np.float32)[None].tolist()
    depot = np.asarray([.5, .5], dtype=np.float32)
    q = (np.arange(n) % 5 + 1).astype(np.float32) / 25
    if 'B' in problem:
        q[np.arange(n) % 3 == 0] *= -1
    if problem == 'CVRP':
        return dict(depot=depot[None, None].tolist(), loc=xy[None].tolist(), demand=q[None].tolist())
    item = dict(depot_xy=depot[None].tolist(), node_xy=xy.tolist(), node_demand=q.tolist(), capacity=1.)
    if 'L' in problem:
        item['route_limit'] = 3.
    if 'TW' in problem:
        item.update(service_time=np.full(n, np.float32(.2)).tolist(),
            tw_start=np.full(n, .7).tolist(), tw_end=np.full(n, 1.9).tolist())
    return item


class FamilyTests(unittest.TestCase):
    def test_all18_distinct_factorial_and_original(self):
        for problem in d.PROBLEMS:
            with self.subTest(problem=problem):
                original, proposal = fixture(problem, 1), fixture(problem, 2)
                family = d.make_family(problem, original, proposal)
                self.assertEqual(len(family), 5)
                self.assertEqual(family[0]['item'], original)
                self.assertEqual(len({r['item_sha256'] for r in family}), 5)
                self.assertEqual([r['cell'] for r in family], ['original', 'z00', 'z01', 'z10', 'z11'])
                self.assertTrue(all(r['input_check']['legal'] for r in family))

    def test_B_semantics_constants_and_whole_demand_multiset(self):
        for problem in d.PROBLEMS[1:]:
            if problem == 'ATSP':
                continue
            original = fixture(problem, 1)
            before = d.r114.fields(problem, original)
            for row in d.make_family(problem, original, fixture(problem, 2)):
                after = d.r114.fields(problem, row['item'])
                np.testing.assert_array_equal(np.sign(after['q']), np.sign(before['q']))
                np.testing.assert_array_equal(np.sort(after['q']), np.sort(before['q']))
                for key in ('capacity', 'route_limit', 'service_time'):
                    if key in original:
                        self.assertEqual(row['item'][key], original[key])

    def test_ATSP_directed_closure_not_a_relabel(self):
        family = d.make_family('ATSP', fixture('ATSP', 1), fixture('ATSP', 2))
        for row in family[1:]:
            matrix = np.asarray(row['item'])[0]
            self.assertFalse(np.allclose(matrix, matrix.T))
            for k in range(len(matrix)):
                self.assertTrue(np.all(matrix <= matrix[:, k, None] + matrix[None, k, :] + 2e-7))
        tour = np.arange(20)
        costs = [np.asarray(row['item'])[0][tour, np.roll(tour, -1)].sum() for row in family[1:]]
        self.assertGreater(np.ptp(costs), .01)

    def test_shared_geometry_group_across_tasks(self):
        cv = fixture('CVRP', 7)
        vr = fixture('VRPB', 7)
        self.assertEqual(d.base_group('CVRP', cv), d.base_group('VRPB', vr))
        for row in d.make_family('VRPB', vr, fixture('VRPB', 8)):
            self.assertNotIn('family_id', row['item'])
            self.assertNotIn('cost', row['item'])

    def test_degenerate_load_axis_is_not_fake_coverage(self):
        original = fixture('CVRP', 7)
        original['demand'] = [[.1] * 20]
        with self.assertRaisesRegex(ValueError, 'Collapsed structural axis'):
            d.make_family('CVRP', original, fixture('CVRP', 8))

    def test_TSP_ATSP_native_tour_audit(self):
        order = list(range(20))
        for problem in ('TSP', 'ATSP'):
            item = fixture(problem, 3)
            values = np.asarray(item)[0]
            cost = values[order, np.roll(order, -1)].sum() if problem == 'ATSP' else np.linalg.norm(values - np.roll(values, -1, axis=0), axis=1).sum()
            self.assertTrue(d.check_tour(problem, item, order, float(cost))['verified'])
            with self.assertRaises(ValueError):
                d.check_tour(problem, item, [0] * 20, float(cost))


def receipts(costs):
    return [dict(method=method, global_id=gid, deployment_id='new-protocol/' + method,
        endpoint_id='unit_endpoint', input_sha256='unit_input', base_id='unit_base',
        version=d.VERSION, status='ok', stage='label', route_audit=dict(verified=True), cost=cost)
        for method, gid, cost in zip(('a', 'b'), (1, 7), costs)]


class SchemaAndGateTests(unittest.TestCase):
    pool = [dict(method='a', global_id=1), dict(method='b', global_id=7)]

    def test_FP64_native_winner_and_exact_ties(self):
        target = d.full_pool_targets(self.pool, receipts([1.00000002, 1.00000001]))
        self.assertEqual(target['native_winner_global_id'], 7)
        self.assertEqual(target['global_ids'], [1, 7])
        self.assertEqual(target['global_ids'].index(target['native_winner_global_id']), 1)
        self.assertAlmostEqual(sum(target['centered_log_advantage']), 0.)
        tied = d.full_pool_targets(self.pool, receipts([1., 1.]))
        self.assertEqual(tied['all_best_mask'], [True, True])
        self.assertEqual(tied['native_winner_global_id'], 1)

    def test_missing_zero_preflight_or_mixed_input_rejected(self):
        with self.assertRaises(ValueError):
            d.full_pool_targets(self.pool, receipts([1., 2.])[:1])
        with self.assertRaises(ValueError):
            d.full_pool_targets(self.pool, receipts([0., 2.]))
        for field, value in (('stage', 'preflight'), ('input_sha256', 'another_input')):
            rows = receipts([1., 2.])
            rows[0][field] = value
            with self.assertRaises(ValueError):
                d.full_pool_targets(self.pool, rows)

    def test_execute_is_blocked_before_any_backend_import(self):
        task = dict(task_id='unit', problem='TSP', method='a')
        manifest = d.seal(dict(plan_path='unit_plan', plan_sha256='plan', tasks=[task]), 'manifest_sha256')
        inventory = dict(deployments={'TSP/a': dict(status='recipe_present_runtime_pending')})
        with patch.object(d.r114, 'read_json', return_value=manifest), patch.object(
                d, 'load_plan', return_value=(dict(plan_sha256='plan'), {}, inventory, {})):
            with self.assertRaisesRegex(PermissionError, 'no solver loaded'):
                d.run_task('unit_manifest', 'unit', 'unused', execute=True, device='cuda:0')

    def test_immutable_plan_identity(self):
        payload = d.seal(dict(version=d.VERSION, budget=256), 'plan_sha256')
        d.verify_seal(payload, 'plan_sha256')
        modified = copy.deepcopy(payload)
        modified['budget'] = 257
        with self.assertRaises(ValueError):
            d.verify_seal(modified, 'plan_sha256')


if __name__ == '__main__':
    unittest.main()

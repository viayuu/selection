"""CPU contract checks; no actual solver/model/GPU is initialized."""

import unittest
from unittest.mock import patch

from . import r121_collection as c


def task(index=0, method='BQ'):
    return dict(task_id=f'TSP_{index}__{method}', endpoint_id=f'TSP_{index}', base_id='unit_base',
        problem='TSP', method=method, global_id=0, input_sha256='input', solver_seed=2,
        deployment_id='unit_deployment')


class CollectorTests(unittest.TestCase):
    def test_lanes_cover_every_deployment_once(self):
        groups = {f'problem/{i:02d}': [] for i in range(89)}
        assigned = [c.lane_keys(groups, i, 4) for i in range(4)]
        self.assertEqual(sorted(x for rows in assigned for x in rows), sorted(groups))
        self.assertEqual([len(rows) for rows in assigned], [23, 22, 22, 22])

    def test_native_tie_and_deployment_identity_are_not_dropped(self):
        t = task()
        row = dict(t, version=c.data.VERSION, collection_sha256='collection', stage='label',
            status='ok', route_audit=dict(verified=True), source='fresh_solver', cost=1.)
        c.validate_receipt(row, t, dict(collection_sha256='collection'))
        for field, value in (('stage', 'preflight'), ('cost', 0.), ('global_id', 1), ('status', 'unavailable')):
            with self.subTest(field=field), self.assertRaises(ValueError):
                c.validate_receipt(dict(row, **{field: value}), t, dict(collection_sha256='collection'))

    def test_finite_budget_is_mandatory(self):
        with self.assertRaises(ValueError):
            c.wave(c.ROOT, 'label', 0, 0)
        with self.assertRaises(ValueError):
            c.worker(c.ROOT, 'TSP/BQ', 'label', 0, None, None, None)

    def test_worker_release_gate_precedes_backend(self):
        with patch.object(c, 'load', return_value=(dict(plan_sha256='plan'), {}, {})), patch.object(c, '_session') as factory:
            with self.assertRaises(PermissionError):
                c.worker(c.ROOT, 'TSP/BQ', 'preflight', 1, 'cuda:0', None, None)
            factory.assert_not_called()

    def test_dry_wave_does_not_load_backend_or_allocate(self):
        manifest = dict(tasks=[task(0), task(1)])
        with patch.object(c, 'load', return_value=(dict(lanes=4), manifest, {})), patch.object(c, '_session') as factory:
            result = c.wave(c.ROOT, 'preflight', 0, 1)
            self.assertFalse(result['solver_execution'])
            self.assertEqual(result['assigned_deployments'], ['TSP/BQ'])
            factory.assert_not_called()

    def test_incomplete_export_cannot_publish(self):
        with patch.object(c, 'load', return_value=({}, {}, {})), patch.object(c, 'status',
                return_value=dict(all_labels_complete=False)):
            with self.assertRaises(c.NotReady):
                c.export(c.ROOT)


if __name__ == '__main__':
    unittest.main()

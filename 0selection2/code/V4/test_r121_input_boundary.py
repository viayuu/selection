"""CPU constructor and REAL native audit tests, not claimed GPU qualification."""

import io
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from . import r121_input_boundary as layer
from .test_r121_structural_data import fixture


class InputBoundaryTests(unittest.TestCase):
    def test_explicit_cpu_constructor_reaches_real_native_audit_for_TSP_and_ATSP(self):
        from . import r114_group_worker
        self.addCleanup(torch.set_default_device, torch.get_default_device())
        torch.set_default_device('meta')
        native_path = '/public/home/shiys/easynco_v3_bridge/scripts/train_100k_backends.py'
        import ast
        # Execute the frozen audit function itself, without importing/loading solvers.
        tree = ast.parse(Path(native_path).read_text())
        audit = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'audit_tours')
        namespace = dict(np=np, sample_scale=lambda p, item: item.shape[1])
        exec(compile(ast.Module(body=[audit], type_ignores=[]), native_path, 'exec'), namespace)
        real_audit = namespace['audit_tours']
        for problem in ('TSP', 'ATSP'):
            item = fixture(problem, 1)
            values = np.asarray(item)[0]
            tour = list(range(len(values)))
            cost = float(values[tour, np.roll(tour, -1)].sum()) if problem == 'ATSP' else float(np.linalg.norm(values - np.roll(values, -1, axis=0), axis=1).sum())
            seen = {}

            def tensor(data, **kwargs):
                seen.update(kwargs)
                return torch.tensor(data, **kwargs)

            def native_run(items):
                certificate = real_audit(problem, items, [tour], [cost])
                return SimpleNamespace(costs=[cost], tours=[tour], audit=certificate)

            fake_torch = SimpleNamespace(tensor=tensor, float32=torch.float32)
            session = SimpleNamespace(torch=fake_torch, device='cpu',
                backend=SimpleNamespace(run=native_run), profile_sha256='unchanged')
            task = dict(problem=problem, method='unit_fixture', endpoint_id='fixture', input_sha256='fixture', solver_seed=123)
            with patch.object(layer.base, 'sha', return_value='fixture'), \
                    patch.object(layer.base, 'read', return_value=dict(item=item)), \
                    patch.object(r114_group_worker, 'reset_rng') as seed:
                implicit = torch.tensor(item, dtype=torch.float32)
                self.assertEqual(implicit.device.type, 'meta')
                with self.assertRaises(TypeError):
                    real_audit(problem, [implicit], [tour], [cost])
                row = layer.cpu_solver(layer.ROOT, {'recovery_sha256': 'fixture'})(session,
                    {'endpoints': [dict(endpoint_id='fixture', input_path='fixture')]}, task)
            self.assertEqual(seen['device'], 'cpu')
            self.assertEqual(row['observed_native_input']['device'], 'cpu')
            self.assertEqual(row['cost'], cost)
            self.assertEqual(row['tour'], tour)
            self.assertTrue(row['route_audit']['verified'])
            self.assertEqual(torch.get_default_device().type, 'meta')
            seed.assert_called_once_with(fake_torch, 123, 'cpu')

    def test_real_trace_required_before_protocol_creation(self):
        with self.assertRaises(ValueError):
            layer.validate_trace({}, {'collection_sha256': 'fixture'}, {'deployments': {}})

    def test_different_physical_binding_rejected_before_native_worker(self):
        recovery = dict(targets=['TSP/OMNI'], original_failures={'TSP/OMNI': dict(binding_sha256='original')})
        with patch.object(layer, 'load', return_value=(recovery, {'plan_sha256': 'plan'}, {}, {})), \
                patch.object(layer.base.data, 'require_release', return_value={}), \
                patch.object(layer.base.data, 'require_binding', return_value='other'), \
                patch.object(layer.base, 'worker') as native_worker:
            with self.assertRaisesRegex(ValueError, 'THIS key original physical binding'):
                layer.worker(layer.ROOT, 'TSP/OMNI', 'preflight', 1, 'cuda:0', 'release', 'binding')
            native_worker.assert_not_called()

    def test_literal_root_create_cli(self):
        with patch.object(layer, 'create', return_value={'solver_execution': False}) as create, \
                patch('sys.stdout', new_callable=io.StringIO):
            self.assertEqual(layer.main(['create', '--root', str(layer.ROOT), '--parent', str(layer.PARENT),
                '--trace', str(layer.TRACE)]), 0)
        create.assert_called_once_with(Path(layer.PARENT), Path(layer.ROOT), Path(layer.TRACE))


if __name__ == '__main__':
    unittest.main()

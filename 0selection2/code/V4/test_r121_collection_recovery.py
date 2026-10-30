"""No CUDA/solver access: emulate native CUDA cost/tour outputs on CPU."""

from types import SimpleNamespace
import unittest
from unittest.mock import patch, Mock

import numpy as np
import torch

from . import r121_collection_recovery as recovery
from .test_r121_structural_data import fixture


class DeviceOutput:
    def __init__(self, value):
        self.value = np.asarray(value)

    def __array__(self, *args, **kwargs):
        raise TypeError("can't convert cuda:0 device type tensor to numpy")

    def detach(self):
        return self

    def cpu(self):
        return torch.as_tensor(self.value)


class OutputRecoveryTests(unittest.TestCase):
    def check_case(self, problem, method):
        item = fixture(problem, 1)
        values = np.asarray(item)[0]
        tour = np.arange(len(values))
        cost = float(values[tour, np.roll(tour, -1)].sum()) if problem == 'ATSP' else float(np.linalg.norm(values - np.roll(values, -1, axis=0), axis=1).sum())
        task = dict(problem=problem, method=method, endpoint_id='fixture', input_sha256='fixture', solver_seed=123)
        endpoint = dict(endpoint_id='fixture', input_path='fixture.json')
        cuda = SimpleNamespace(synchronize=Mock(), manual_seed_all=Mock())
        cpu_torch = SimpleNamespace(tensor=torch.tensor, float32=torch.float32,
            manual_seed=torch.manual_seed, cuda=cuda)
        backend = SimpleNamespace(run=Mock(return_value=SimpleNamespace(costs=[DeviceOutput(cost)],
            tours=[DeviceOutput(tour)], audit={'fixture': True})))
        session = SimpleNamespace(torch=cpu_torch, device='cuda:0', backend=backend, profile_sha256='unchanged')
        data = recovery.base.data
        with patch.object(recovery.base, 'sha', return_value='fixture'), \
                patch.object(recovery.base, 'read', return_value={'item': item}), \
                patch('code.V4.r114_group_worker.reset_rng') as reset:
            # Reproduce the old failure with the SAME already-plainable CUDA tour:
            # cost reaches np.isfinite before a scalar conversion in the old path.
            with self.assertRaisesRegex(TypeError, 'cuda:0'):
                recovery.ORIGINAL_SOLVE(session, {'endpoints': [endpoint]}, task)
            backend.run.reset_mock()
            reset.reset_mock()
            result = recovery.recovered_solver('fixture_recovery')(session, {'endpoints': [endpoint]}, task)
        self.assertEqual(result['cost'], cost)
        self.assertEqual(result['tour'], tour.tolist())
        self.assertTrue(result['route_audit']['verified'])
        self.assertEqual(result['profile_sha256'], 'unchanged')
        self.assertEqual(result['output_serialization_recovery_sha256'], 'fixture_recovery')
        self.assertFalse(result['solver_model_seed_profile_changes'])
        backend.run.assert_called_once()
        reset.assert_called_once_with(cpu_torch, 123, 'cuda:0')

    def test_ATSP_UNICO_native_cuda_cost_and_tour(self):
        self.check_case('ATSP', 'UNICO_MatPOENet')

    def test_TSP_OMNI_native_cuda_cost_and_tour(self):
        self.check_case('TSP', 'OMNI')


if __name__ == '__main__':
    unittest.main()

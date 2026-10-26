"""CPU-only equivalence checks; --benchmark times repeated raw TRAIN inputs.

No GPU allocation, targets, native solver, training, or generation-directory
writes. Standard invocation: python -B -m unittest code.V4.test_r113_torch_inputs.
"""

import sys
import time
import unittest
from unittest import mock

import torch

from .r113_process_data import (
    DEFAULT_CACHE, INPUT_KEYS, PROBLEMS, ProgramSpec, TASK_TYPES,
    TrainInputCache, compile_inputs,
)
from .r113_torch_inputs import compile_inputs_torch


def coordinate_fixture(problem):
    sizes = torch.tensor([2, 5, 10, 17])
    batch, size = len(sizes), 17
    dimension = 2 if problem == 'TSP' else 4 if problem == 'CVRP' else 8
    node = torch.zeros(batch, size, dimension)
    index = torch.arange(size)
    node[..., 0] = (index % 3).float() / 7.
    node[..., 1] = (index % 2).float() / 5.
    c, _, backhaul, length, tw = TASK_TYPES[problem]
    if c:
        depot = torch.tensor([0, 1, 3, 5])
        node[..., 2] = (index % 5 + 1).float() / 30.
        if backhaul:
            node[..., 2] *= torch.where(index % 3 == 0, -1., 1.)
        node[torch.arange(batch), depot, 2] = 0.
        node[torch.arange(batch), depot, 3] = 1.
        if length:
            node[..., 4] = 3.
            node[torch.arange(batch), depot, 4] = 0.
        if tw:
            node[..., 5] = .2
            node[..., 6] = index.float() / 20.
            node[..., 7] = 3. + index.float() / 40.
            node[torch.arange(batch), depot, 5:8] = 0.
    mask = index[None] < sizes[:, None]
    node[~mask] = float('nan')
    return dict(kind='coord', node=node, n=sizes, node_mask=mask,
                cbits=torch.tensor(TASK_TYPES[problem]).expand(batch, -1),
                problem_id=PROBLEMS.index(problem), coord_dist=torch.tensor([0, 1, 2, 4]))


def matrix_fixture():
    sizes = torch.tensor([2, 5, 10, 17])
    size = 17
    matrix = (torch.arange(4 * size * size).reshape(4, size, size) % 19).float() / 64.
    # Nonzero diagonals test off-diagonal means versus diagonal-preserving arcs.
    mask = torch.arange(size)[None] < sizes[:, None]
    matrix = matrix.masked_fill(~(mask[:, :, None] & mask[:, None]), float('nan'))
    return dict(kind='matrix', matrix=matrix, n=sizes, node_mask=mask, problem_id=2)


def assert_equivalent(test, actual, expected):
    test.assertEqual(tuple(actual), INPUT_KEYS)
    test.assertEqual(set(actual), set(expected))
    for key in INPUT_KEYS:
        with test.subTest(key=key):
            test.assertEqual(actual[key].shape, expected[key].shape)
            test.assertEqual(actual[key].dtype, expected[key].dtype)
            test.assertEqual(actual[key].device, expected[key].device)
            if actual[key].is_floating_point():
                torch.testing.assert_close(actual[key], expected[key], rtol=2e-6, atol=2e-7)
            else:
                test.assertTrue(torch.equal(actual[key], expected[key]), key)


class TorchInputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_all18_synthetic_every_key(self):
        for problem in PROBLEMS:
            with self.subTest(problem=problem):
                raw = matrix_fixture() if problem == 'ATSP' else coordinate_fixture(problem)
                assert_equivalent(self, compile_inputs_torch(raw, problem), compile_inputs(raw, problem))

    @unittest.skipUnless((DEFAULT_CACHE / 'train/TSP/metadata.json').is_file(), 'Frozen TRAIN cache not present')
    def test_two_actual_train_inputs_per_task(self):
        for problem in PROBLEMS:
            with self.subTest(problem=problem):
                with TrainInputCache(DEFAULT_CACHE, problem) as source:
                    raw, _ = source.batch(torch.arange(2).numpy())
                assert_equivalent(self, compile_inputs_torch(raw, problem), compile_inputs(raw, problem))

    def test_TSP_raw_xy_and_R88_customer_role_pad_are_equivalent(self):
        raw = coordinate_fixture('TSP')
        expected = compile_inputs(raw, 'TSP')
        padded = torch.nn.functional.pad(raw['node'], (0, 6))
        padded[:, 0, 3] = 1.
        # Role points to raw vertex zero, which need not be the canonical anchor.
        assert_equivalent(self, compile_inputs_torch(dict(raw, node=padded), 'TSP'), expected)
        raw['node'].requires_grad_()
        inputs = compile_inputs_torch(raw, 'TSP')
        self.assertFalse(any(value.requires_grad for value in inputs.values()))

    def test_near_ties_use_FP64_before_neighbor_selection(self):
        raw = coordinate_fixture('TSP')
        raw['node'] = torch.stack((torch.arange(17).float(), torch.zeros(17)), -1)[None]
        raw['node'][0, :5] = torch.tensor([[0., 0.], [1., 0.], [1., 2. ** -13],
                                         [1., 2. ** -12], [1., 2. ** -11]])
        raw['n'] = torch.tensor([17])
        raw['node_mask'] = torch.ones(1, 17, dtype=torch.bool)
        raw['cbits'] = torch.zeros(1, 5)
        raw['coord_dist'] = torch.tensor([4])
        with torch.autocast('cpu', dtype=torch.bfloat16):
            compiled = compile_inputs_torch(raw, 'TSP')
        assert_equivalent(self, compiled, compile_inputs(raw, 'TSP'))

    def test_signed_and_classical_graph_types_stay_independent(self):
        raw = coordinate_fixture('VRPB')
        program = compile_inputs_torch(raw, 'VRPB')
        self.assertTrue(torch.equal(program['graph_mask'], torch.tensor([[False, True, True]]).expand(4, -1)))
        self.assertTrue((program['nodes'][..., 2] < 0).any())
        self.assertFalse(program['graph_mask'][:, 0].any())
        # The compiler produces graph types, not exact compatibility/teacher bits.
        self.assertNotIn('compatible', program)

    def test_allowlist_ignores_geometry_and_rejects_labels(self):
        raw = coordinate_fixture('CVRP')
        expected = compile_inputs(raw, 'CVRP')
        assert_equivalent(self, compile_inputs_torch(dict(raw, node_geom=object()), 'CVRP'), expected)
        for key in ('costs', 'ind', 'winner', 'targets', 'hint', 'routes', 'cmp_target', 'compatible', 'surprise'):
            with self.subTest(key=key), self.assertRaises(ValueError):
                compile_inputs_torch(dict(raw, **{key: None}), 'CVRP')
        with self.assertRaises(ValueError):
            compile_inputs_torch(raw, 'CVRP', ProgramSpec(8, 2))

    def test_no_host_conversion_or_teacher_fallback(self):
        raw = coordinate_fixture('VRPBLTW')
        original = {key: value.clone() for key, value in raw.items() if torch.is_tensor(value)}
        with mock.patch.object(torch.Tensor, 'cpu', side_effect=AssertionError('host transfer')), \
             mock.patch.object(torch.Tensor, 'numpy', side_effect=AssertionError('NumPy conversion')), \
             mock.patch.object(torch.Tensor, 'item', side_effect=AssertionError('scalar sync')), \
             mock.patch.object(torch.Tensor, 'tolist', side_effect=AssertionError('host list')), \
             mock.patch('code.V4.r113_process_data.compile_inputs', side_effect=AssertionError('CPU fallback')), \
             mock.patch('code.V4.r113_process_data.execute_program', side_effect=AssertionError('teacher access')):
            compiled = compile_inputs_torch(dict(raw, node_geom=object()), 'VRPBLTW')
            matrix = compile_inputs_torch(matrix_fixture(), 'ATSP')
        self.assertEqual(compiled['nodes'].shape, (4, 12, 12))
        self.assertEqual(matrix['edge_features'].shape, (4, 12, 12, 2))
        for key, value in original.items():
            torch.testing.assert_close(raw[key], value, equal_nan=True)

    def test_tensor_validation_and_noncontiguous_input(self):
        raw = coordinate_fixture('CVRP')
        buffer = raw['node'].repeat_interleave(2, dim=2)
        assert_equivalent(self, compile_inputs_torch(dict(raw, node=buffer[..., ::2]), 'CVRP'), compile_inputs(raw, 'CVRP'))
        with self.assertRaises(RuntimeError):
            compile_inputs_torch(dict(raw, cbits=torch.ones(4, 5)), 'CVRP')
        with self.assertRaises(RuntimeError):
            compile_inputs_torch(dict(raw, n=torch.ones(4, dtype=torch.int64)), 'CVRP')


def benchmark():
    """Repeat only two real TRAIN inputs to B=512; no new assets or targets."""
    import json
    import statistics
    torch.set_num_threads(2)
    measurements = []
    for problem in ('TSP', 'CVRP', 'ATSP', 'VRPBLTW'):
        with TrainInputCache(DEFAULT_CACHE, problem) as source:
            raw, _ = source.batch(torch.arange(2).numpy())
        raw = {key: value.repeat((256,) + (1,) * (value.ndim - 1)) if torch.is_tensor(value) and value.ndim
               else value for key, value in raw.items()}
        compile_inputs_torch(raw, problem)
        timings = []
        for _ in range(3):
            started = time.perf_counter()
            result = compile_inputs_torch(raw, problem)
            timings.append(time.perf_counter() - started)
        started = time.perf_counter()
        reference = compile_inputs(raw, problem)
        baseline = time.perf_counter() - started
        assert_equivalent(unittest.TestCase(), result, reference)
        median = statistics.median(timings)
        measurements.append(dict(problem=problem, batch=512, padded_N=raw['node' if problem != 'ATSP' else 'matrix'].shape[1],
                                 torch_seconds=median, reference_seconds=baseline, speedup=baseline / median))
    print(json.dumps(dict(cpu_threads=2, real_train_rows_read=8, repeated_input_benchmark=True,
                          gpu_used=False, targets_used=False, measurements=measurements), indent=2))


if __name__ == '__main__':
    if sys.argv[1:] == ['--benchmark']:
        benchmark()
    else:
        unittest.main()

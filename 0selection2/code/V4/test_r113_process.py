"""Small CPU-only R113 semantic/interface tests; no solver or training jobs."""

import argparse
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

from .r113_process_data import (
    BINARY_FIELDS, PROBLEMS, ProgramSpec, SCALAR_FIELDS, SCENARIO,
    TASK_TYPES, TrainInputCache, build_assets, compile_inputs,
    compose_resource_envelopes, execute_program, load_shard, propagate_compatibility,
)
from .r113_processor import TypedProcessOperator, process_loss, soft_directed_propagation


def raw_routing(problem, demands, *, x=None, service=None, early=None, late=None, limit=1.):
    n = len(demands) + 1
    d = 4 if problem == 'CVRP' else 8
    node = torch.zeros(1, n, d)
    node[0, 0, 3] = 1.
    node[0, 1:, 0] = torch.tensor(x if x is not None else np.arange(1, n) / 10., dtype=torch.float32)
    node[0, 1:, 2] = torch.tensor(demands, dtype=torch.float32)
    if TASK_TYPES[problem][3]:
        node[0, 1:, 4] = limit
    if TASK_TYPES[problem][4]:
        node[0, 1:, 5] = torch.tensor(service if service is not None else [0.] * (n - 1))
        node[0, 1:, 6] = torch.tensor(early if early is not None else [0.] * (n - 1))
        node[0, 1:, 7] = torch.tensor(late if late is not None else [5.] * (n - 1))
    return dict(kind='coord', node=node, n=torch.tensor([n]))


def last(target, key, program):
    return target[key][0, int(program['steps'][0])]


class ProcessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_envelope_associative_and_not_commutative(self):
        a, b, c = (np.array(v) for v in ((.8, 0., .8), (-.9, -.9, 0.), (.3, 0., .3)))
        join = compose_resource_envelopes
        np.testing.assert_allclose(join(join(a, b), c), join(a, join(b, c)))
        self.assertFalse(np.allclose(join(a, b), join(b, a)))
        np.testing.assert_allclose(join(torch.tensor(a), torch.tensor(b)), join(a, b))

    def test_signed_interval_and_classical_B_are_different_graphs(self):
        p = compile_inputs(raw_routing('VRPB', [-.4, .4]), 'VRPB')
        y = execute_program(p)
        flags = last(y, 'binary', p)
        self.assertTrue(flags[BINARY_FIELDS.index('signed_interval_ok')])
        self.assertFalse(flags[BINARY_FIELDS.index('signed_full_start_ok')])
        self.assertFalse(flags[BINARY_FIELDS.index('classical_precedence_ok')])
        self.assertTrue(y['compatible'][0, 1, 0, 1])
        self.assertFalse(y['compatible'][0, 2, 0, 1])
        s = last(y, 'scalar', p)
        self.assertAlmostEqual(float(s[3]), 0., places=6)
        self.assertAlmostEqual(float(s[4]), .6, places=6)
        p2 = compile_inputs(raw_routing('VRPB', [.8, -.9]), 'VRPB')
        y2 = execute_program(p2)
        self.assertTrue(last(y2, 'binary', p2)[6])
        self.assertFalse(last(y2, 'binary', p2)[1])

    def test_open_closed_distance_limit_is_not_service_time(self):
        programs = [compile_inputs(raw_routing(task, [.1, .1], x=[.4, .8]), task)
                    for task in ('VRPL', 'OVRPL')]
        closed, opened = [execute_program(p) for p in programs]
        self.assertAlmostEqual(float(last(closed, 'scalar', programs[0])[8]), 1.6, places=6)
        self.assertAlmostEqual(float(last(opened, 'scalar', programs[1])[8]), .8, places=6)
        self.assertFalse(last(closed, 'binary', programs[0])[5])
        self.assertTrue(last(opened, 'binary', programs[1])[5])
        self.assertFalse(closed['compatible'][0, 1, 0, 1])
        self.assertTrue(opened['compatible'][0, 1, 0, 1])

    def test_time_wait_service_deadline_no_depot_horizon(self):
        raw = raw_routing('VRPTW', [.1, .1], service=[.3, .1], early=[.5, 0.], late=[.55, .7])
        p = compile_inputs(raw, 'VRPTW')
        y = execute_program(p)
        s = last(y, 'scalar', p)
        self.assertAlmostEqual(float(s[9]), .9, places=6)
        self.assertAlmostEqual(float(s[11]), 1., places=6)
        self.assertAlmostEqual(float(s[12]), .4, places=6)
        self.assertAlmostEqual(float(s[13]), -.2, places=6)
        self.assertFalse(last(y, 'binary', p)[4])
        self.assertFalse(y['compatible'][0, 1, 0, 1])
        raw['node'][0, 2, 7] = 2.
        p = compile_inputs(raw, 'VRPTW')
        self.assertTrue(last(execute_program(p), 'binary', p)[4])

    def test_atsp_directed_raw_arcs_and_input_only_schedule(self):
        matrix = torch.tensor([[[0., 2., 9., 1.], [5., 0., 3., 2.],
                                [1., 8., 0., 4.], [7., 6., 2., 0.]]])
        p = compile_inputs(dict(kind='matrix', matrix=matrix, n=torch.tensor([4])), 'ATSP')
        y = execute_program(p)
        anchor, seq = int(p['anchor_index'][0]), p['sequence'][0, :2]
        expected = matrix[0, anchor, seq[0]] + matrix[0, seq[0], seq[1]] + matrix[0, seq[1], anchor]
        self.assertAlmostEqual(float(last(y, 'scalar', p)[8]), float(expected))
        self.assertEqual(int(p['steps'][0]), 2)
        self.assertEqual(float(p['edge_features'][0, 0, 1, 1]), float(matrix[0, seq[1], seq[0]]))
        self.assertTrue(p['graph_mask'][0, 0])
        self.assertFalse(p['graph_mask'][0, 1:].any())
        self.assertFalse(p['scalar_mask'][0, :7].any())

    def test_all18_explicit_types_and_padding(self):
        for problem in PROBLEMS:
            with self.subTest(problem=problem):
                if problem == 'ATSP':
                    raw = dict(kind='matrix', matrix=torch.ones(1, 6, 6) - torch.eye(6)[None], n=torch.tensor([6]))
                elif problem == 'TSP':
                    raw = dict(kind='coord', node=torch.arange(12).float().reshape(1, 6, 2) / 12., n=torch.tensor([6]))
                else:
                    raw = raw_routing(problem, [.2, -.1, .3, -.2, .1] if TASK_TYPES[problem][2] else [.1] * 5)
                p = compile_inputs(raw, problem)
                y = execute_program(p)
                self.assertEqual(int(p['steps'][0]), 3)
                self.assertEqual(y['scalar'].shape, (1, 13, len(SCALAR_FIELDS)))
                self.assertTrue(torch.isfinite(y['scalar']).all())
                self.assertFalse((y['scalar'] * ~p['scalar_mask'][:, None]).any())
                self.assertEqual(int(y['transition_mask'].sum()), 3)
                self.assertTrue(torch.equal(y['scalar'][:, 3], y['scalar'][:, -1]))

    def test_directional_boolean_and_soft_propagation(self):
        graph = torch.tensor([[[[False, True, False], [False, False, True], [False, False, False]]]])
        topology = graph[:, 0]
        reach = torch.tensor([[[True, False, False]]])
        once, frontier = propagate_compatibility(reach, graph, topology[:, None])
        self.assertEqual(once.tolist(), [[[True, True, False]]])
        soft, soft_frontier = soft_directed_propagation(reach.float(), graph.float(), topology)
        self.assertTrue(torch.equal(soft.bool(), once))
        self.assertTrue(torch.equal(soft_frontier.bool(), frontier))
        twice, _ = propagate_compatibility(once, graph, topology[:, None])
        self.assertTrue(twice.all())

    def test_reject_labels_wrong_task_and_negative_non_B(self):
        raw = raw_routing('CVRP', [.1, .2])
        with self.assertRaises(ValueError):
            compile_inputs(dict(raw, costs=torch.zeros(1, 3)), 'CVRP')
        with self.assertRaises(ValueError):
            compile_inputs(raw_routing('CVRP', [-.1, .2]), 'CVRP')
        with self.assertRaises(ValueError):
            compile_inputs(dict(raw, cbits=torch.ones(1, 5)), 'CVRP')

    def test_self_feedback_teacher_mode_and_finite_backward(self):
        p = compile_inputs(raw_routing('VRPB', [.1] * 5), 'VRPB', ProgramSpec(4, 2))
        y = execute_program(p)
        model = TypedProcessOperator(16).eval()
        with torch.no_grad():
            for value in model.resource_operator.parameters():
                value.zero_()
            model.resource_operator[-1].bias.fill_(1.)
            with mock.patch('code.V4.r113_process_data.execute_program', side_effect=AssertionError('teacher used')):
                out = model(p)
        self.assertAlmostEqual(float(out['scalar'][0, 2, 0]), 2.)
        self.assertTrue(torch.equal(out['scalar'][:, 3], out['scalar'][:, 4]))
        with self.assertRaises(PermissionError):
            model.rollout(p, targets=y, teacher_forcing=True)
        with self.assertRaises(PermissionError):
            model.rollout(p, targets=y)
        model.train()
        teacher_out = model.rollout(p, targets=y, teacher_forcing=True)
        self.assertAlmostEqual(float(teacher_out['scalar'][0, 2, 0]), 1.1, places=6)
        self.assertAlmostEqual(float(teacher_out['scalar_delta'][0, 1, 0]), 1., places=6)
        loss = process_loss(teacher_out, y, p)['total']
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(all(torch.isfinite(v.grad).all() for v in model.parameters() if v.grad is not None))

    def test_loss_ignores_padding_inactive_types_and_diagonal(self):
        p = compile_inputs(raw_routing('CVRP', [.1, .2]), 'CVRP', ProgramSpec(4, 2))
        y = execute_program(p)
        model = TypedProcessOperator(16).eval()
        with torch.no_grad():
            out = model(p)
            before = process_loss(out, y, p)['total']
            bad = {key: value.clone() for key, value in y.items()}
            bad['scalar'][:, 3:] = 999.
            bad['scalar'][:, :, 9:14] = 999.
            bad['scalar_delta'][:, 2:] = 999.
            bad['binary'][:, 3:] = True
            bad['reached'][:, 3:] = True
            bad['frontier'][:, 3:] = True
            bad['compatible'][:, 0] = True
            diagonal = torch.arange(4)
            bad['compatible'][:, 1, diagonal, diagonal] = True
            after = process_loss(out, bad, p)['total']
        self.assertAlmostEqual(float(before), float(after), places=6)

    def test_microbatch_sample_weighting_matches_whole_batch(self):
        short = compile_inputs(raw_routing('CVRP', [.1, .2]), 'CVRP', ProgramSpec(4, 2))
        long = compile_inputs(raw_routing('VRPB', [.1, -.2, .3, -.1, .2]), 'VRPB', ProgramSpec(4, 2))
        p = {key: torch.cat((short[key], long[key], long[key])) for key in short}
        y = execute_program(p)
        model = TypedProcessOperator(16).eval()
        with torch.no_grad():
            whole = process_loss(model(p), y, p)['total']
            pieces = []
            for start, end in ((0, 1), (1, 3)):
                local = {key: value[start:end] for key, value in p.items()}
                targets = {key: value[start:end] for key, value in y.items()}
                pieces.append(process_loss(model(local), targets, local)['total'] * (end - start) / 3.)
        self.assertAlmostEqual(float(whole), float(sum(pieces)), places=5)

    def test_train_reader_whitelist_build_load_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cache, data = root / 'cache', root / 'data'
            directory = cache / 'train/CVRP'
            directory.mkdir(parents=True)
            source = data / 'CVRPtrain'
            source.mkdir(parents=True)
            raw = raw_routing('CVRP', [.1, .2])
            arrays = dict(node=raw['node'].numpy(), n=np.array([3], np.int64), coord_dist=np.array([4], np.int64))
            for key, value in arrays.items():
                np.save(directory / (key + '.npy'), value)
            np.savez(source / 'input_metadata.npz', source_index=np.array([1234], np.int64))
            metadata = dict(split='train', problem='CVRP', problem_id=PROBLEMS.index('CVRP'),
                scenario_id=SCENARIO, cache_format='r76_scenario_v1', cache_version=1,
                complete=True, rows_written=1, rows=1, data_root=str(data), kind='coord', d_in=4,
                cbits=list(TASK_TYPES['CVRP']),
                arrays={key: dict(shape=list(value.shape), dtype=str(value.dtype)) for key, value in arrays.items()})
            (directory / 'metadata.json').write_text(json.dumps(metadata))
            load = np.load
            opened = []
            def tracked(path, *args, **kwargs):
                opened.append(str(path))
                return load(path, *args, **kwargs)
            with mock.patch('code.V4.r113_process_data.np.load', side_effect=tracked):
                with TrainInputCache(cache, 'CVRP', data) as reader:
                    _, ids = reader.batch(np.array([0], np.int64))
                    self.assertEqual(int(ids['source_index'][0]), 1234)
            self.assertEqual({Path(path).name for path in opened},
                             {'node.npy', 'n.npy', 'coord_dist.npy', 'input_metadata.npz'})
            args = argparse.Namespace(cache_root=cache, data_root=data, problems=['CVRP'],
                output=root / 'assets', max_steps=4, neighbors=2, rows_per_problem=1, start_row=0, shard_size=8)
            manifest = build_assets(args)
            p, y, ids = load_shard(args.output / manifest['shards'][0]['path'])
            self.assertEqual(int(ids['source_index'][0]), 1234)
            self.assertTrue(torch.equal(y['scalar'], execute_program(p)['scalar']))
            with self.assertRaises(FileExistsError):
                build_assets(args)
            metadata['split'] = 'val'
            (directory / 'metadata.json').write_text(json.dumps(metadata))
            with mock.patch('code.V4.r113_process_data.np.load', side_effect=AssertionError('array opened')):
                with self.assertRaises(PermissionError):
                    TrainInputCache(cache, 'CVRP', data)


if __name__ == '__main__':
    unittest.main()

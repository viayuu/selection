"""CPU tests of logical-batch gradients, label isolation and real saved evaluation."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP

from . import r120_experiment as experiment
from .r120_objective import MultidomainObjective


class TinySelector(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 3)
        self.unused_other_domain = nn.Parameter(torch.zeros(2))

    def forward(self, batch):
        if set(batch) != {'problem', 'inputs', 'pool_ids', 'solver_mask'}:
            raise ValueError('Labels leaked into the selector')
        utility = self.linear(batch['inputs']['x'])
        return dict(logits=utility, utility=utility, solver_mask=batch['solver_mask'],
                    pair_margin=utility[:, :, None] - utility[:, None, :],
                    advantage=utility - utility.mean(1, keepdim=True))


def fixture_batch(count=7):
    generator = torch.Generator().manual_seed(20)
    values = torch.rand(count, 3, generator=generator, dtype=torch.float64) + 1
    return dict(problem='MIS', inputs={'x': torch.randn(count, 2, generator=generator)},
        pool_ids=torch.arange(3), solver_mask=torch.ones(count, 3, dtype=torch.bool),
        labels=dict(native_objective=values, observed_mask=torch.ones_like(values, dtype=torch.bool),
            all_best_mask=values == values.max(1, keepdim=True).values, direction='max'),
        metadata=[{'instance_id': str(i)} for i in range(count)], row_indices=torch.arange(count))


def ddp_worker(rank, path, output, unequal=False):
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method='file://' + path, rank=rank, world_size=2)
    try:
        torch.manual_seed(2)
        model = TinySelector()
        objective = DDP(MultidomainObjective(model, {'MIS': 2.}),
                        find_unused_parameters=True, broadcast_buffers=False)
        batch = fixture_batch(8)
        indices = experiment.rank_indices(np.arange(8), rank, 2, (5, 3) if unequal else None)
        local = experiment.slice_batch(batch, int(indices[0]), int(indices[-1]) + 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=.03)
        experiment.update_batch(objective, optimizer, torch.amp.GradScaler('cuda', enabled=False),
            local, experiment.Runtime(rank, 2, torch.device('cpu')), microbatch_size=3, global_count=8)
        if rank == 0:
            torch.save(model.state_dict(), output)
    finally:
        dist.destroy_process_group()


def native_fixture(problem, count=8):
    if problem == 'MIS':
        inputs = dict(num_nodes=torch.tensor([4]), edge_index=torch.tensor([[[0, 1], [1, 2]]]),
            node_mask=torch.ones(1, 4, dtype=torch.bool), edge_mask=torch.ones(1, 2, dtype=torch.bool))
    elif problem == 'BPP':
        inputs = dict(sizes=torch.tensor([[6, 5, 4, 3]]), capacity=torch.tensor([10]),
                      item_mask=torch.ones(1, 4, dtype=torch.bool))
    else:
        inputs = dict(profits=torch.tensor([[6, 5, 4]]),
            resources=torch.tensor([[[2, 1], [1, 3], [2, 2]]]), capacities=torch.tensor([[4, 4]]),
            item_mask=torch.ones(1, 3, dtype=torch.bool), resource_mask=torch.ones(1, 2, dtype=torch.bool))
    inputs = {key: value.expand(count, *value.shape[1:]).clone() for key, value in inputs.items()}
    values = torch.tensor([[1., 2.], [3., 2.]], dtype=torch.float64).repeat(count // 2, 1)
    direction = 'min' if problem == 'BPP' else 'max'
    best = values.amin(1, keepdim=True) if direction == 'min' else values.amax(1, keepdim=True)
    return dict(problem=problem, inputs=inputs, pool_ids=torch.tensor([1, 2]),
        solver_mask=torch.ones(count, 2, dtype=torch.bool), labels=dict(native_objective=values,
            observed_mask=torch.ones(count, 2, dtype=torch.bool), all_best_mask=values == best,
            direction=direction))


def native_selector():
    from .r120_model import MultidomainSelector, default_config
    torch.manual_seed(2)
    return MultidomainSelector(default_config(['BQ', 'NR::fixture_a', 'NR::fixture_b'],
        embedding_dim=32, qkv_dim=8, ff_hidden_dim=64, head_hidden_dim=32,
        encoder_layer_num=2, joint_layer_num=1, dropout=0.))


def native_ddp_worker(rank, path, output):
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method='file://' + path, rank=rank, world_size=2)
    try:
        model = native_selector()
        objective = DDP(MultidomainObjective(model, dict(MIS=2., BPP=2., MKP=2.)),
                        find_unused_parameters=True, broadcast_buffers=False)
        optimizer = torch.optim.SGD(model.parameters(), lr=.03)
        for problem in ('MIS', 'BPP', 'MKP'):
            indices = experiment.rank_indices(np.arange(8), rank, 2, (5, 3))
            local = experiment.slice_batch(native_fixture(problem), int(indices[0]), int(indices[-1]) + 1)
            experiment.update_batch(objective, optimizer, torch.amp.GradScaler('cuda', enabled=False),
                local, experiment.Runtime(rank, 2, torch.device('cpu')), microbatch_size=3, global_count=8)
        if rank == 0:
            torch.save(model.state_dict(), output)
    finally:
        dist.destroy_process_group()


class RunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_cli_defaults_and_formal_budget(self):
        for action in ('prepare', 'train', 'test', 'lib', 'report'):
            args = experiment.parser().parse_args([action])
            self.assertEqual(args.max_updates, 57600)
            self.assertEqual(args.logical_batch_size, 1024)
            self.assertEqual(args.microbatch_size, 64)
            self.assertEqual(args.seed, 2)
            self.assertEqual(args.early_stop_patience, 6)

    def test_uniform_tasks_and_disjoint_real_rank_splits(self):
        order = experiment.task_order(72)
        for first in range(0, 72, 24):
            self.assertEqual(set(order[first:first + 24]), set(experiment.PROBLEMS))
        parts = [experiment.rank_indices(np.arange(11), rank, 4) for rank in range(4)]
        np.testing.assert_array_equal(np.concatenate(parts), np.arange(11))
        self.assertEqual([len(part) for part in parts], [2, 3, 3, 3])
        weighted = [experiment.rank_indices(np.arange(1024), rank, 4, (5, 5, 3, 3)) for rank in range(4)]
        self.assertEqual([len(part) for part in weighted], [320, 320, 192, 192])
        np.testing.assert_array_equal(np.concatenate(weighted), np.arange(1024))

    def test_successful_update_learning_rate_axis(self):
        self.assertAlmostEqual(experiment.learning_rate(4320), 1e-4)
        self.assertAlmostEqual(experiment.learning_rate(57600), 1e-6)
        with self.assertRaises(ValueError):
            experiment.learning_rate(0)

    def test_slice_keeps_static_pool_and_labels_outside_forward(self):
        batch = fixture_batch(3)
        micro = experiment.slice_batch(batch, 1, 3)
        self.assertEqual(len(micro['pool_ids']), 3)
        self.assertEqual(len(micro['inputs']['x']), 2)
        self.assertEqual(len(micro['labels']['native_objective']), 2)
        self.assertEqual(micro['metadata'][0]['instance_id'], '1')
        TinySelector()(experiment.model_inputs(micro))

    def test_microbatch_matches_full_batch_gradient(self):
        states = []
        for cap in (7, 3):
            torch.manual_seed(2)
            model = TinySelector()
            objective = MultidomainObjective(model, {'MIS': 2.})
            optimizer = torch.optim.SGD(model.parameters(), lr=.03)
            values = experiment.update_batch(objective, optimizer,
                torch.amp.GradScaler('cuda', enabled=False), fixture_batch(),
                experiment.Runtime(0, 1, torch.device('cpu')), microbatch_size=cap, global_count=7)
            self.assertEqual(values['amp_retries'], 0)
            self.assertGreater(values['grad_norm'], 0)
            states.append(model.state_dict())
        for key in states[0]:
            torch.testing.assert_close(states[0][key], states[1][key], rtol=1e-6, atol=1e-7)

    def test_two_rank_ddp_matches_single_logical_batch(self):
        import torch.multiprocessing as mp
        with TemporaryDirectory() as directory:
            directory = Path(directory)
            mp.spawn(ddp_worker, args=(str(directory / 'gloo_init'), str(directory / 'ddp.pt')),
                     nprocs=2, join=True)
            actual = torch.load(directory / 'ddp.pt', map_location='cpu', weights_only=True)
            torch.manual_seed(2)
            model = TinySelector()
            objective = MultidomainObjective(model, {'MIS': 2.})
            optimizer = torch.optim.SGD(model.parameters(), lr=.03)
            experiment.update_batch(objective, optimizer, torch.amp.GradScaler('cuda', enabled=False),
                fixture_batch(8), experiment.Runtime(0, 1, torch.device('cpu')),
                microbatch_size=8, global_count=8)
            for key in actual:
                torch.testing.assert_close(actual[key], model.state_dict()[key], rtol=2e-6, atol=2e-7)

    def test_unequal_rank_rows_and_micro_counts_keep_logical_gradient(self):
        import torch.multiprocessing as mp
        with TemporaryDirectory() as directory:
            directory = Path(directory)
            mp.spawn(ddp_worker, args=(str(directory / 'gloo_init'), str(directory / 'ddp.pt'), True),
                     nprocs=2, join=True)
            actual = torch.load(directory / 'ddp.pt', map_location='cpu', weights_only=True)
            torch.manual_seed(2)
            model = TinySelector()
            optimizer = torch.optim.SGD(model.parameters(), lr=.03)
            experiment.update_batch(MultidomainObjective(model, {'MIS': 2.}), optimizer,
                torch.amp.GradScaler('cuda', enabled=False), fixture_batch(8),
                experiment.Runtime(0, 1, torch.device('cpu')), microbatch_size=8, global_count=8)
            for key in actual:
                torch.testing.assert_close(actual[key], model.state_dict()[key], rtol=2e-6, atol=2e-7)

    def test_saved_evaluation_preserves_original_objective_and_tail(self):
        class Task:
            problem, objective_direction = 'MIS', 'max'
            candidate_solver_ids = ('a', 'b', 'c')
            label_arrays = {'native_objective': fixture_batch()['labels']['native_objective'].numpy(),
                            'observed_mask': np.ones((7, 3), dtype=bool)}

            def __len__(self):
                return 7

        class Store:
            def open(self, problem, split):
                return Task()

            def batch(self, problem, split, indices, pools):
                full = fixture_batch()
                return experiment.slice_batch(full, int(indices[0]), int(indices[-1]) + 1)

            def close_split(self, split):
                pass

        with TemporaryDirectory() as directory:
            root = Path(directory)
            config = dict(model=dict(problem_names=['MIS'], pools={'MIS': [0, 1, 2]}),
                references={'MIS': dict(objective_scale=2., sbs_slot=0, sbs_solver='a')},
                eval_batch_size=3, microbatch_size=3, microbatch_caps={})
            result = experiment.evaluate(TinySelector(), Store(), config,
                experiment.Runtime(0, 1, torch.device('cpu')), split='val', tag='smoke',
                step=1, root=root)
            with np.load(root / 'evaluation/val/smoke/by_problem/MIS.npz') as saved:
                self.assertEqual(saved['native_objective'].dtype, np.float64)
                np.testing.assert_array_equal(saved['indices'], np.arange(7))
            self.assertEqual(result['per_problem']['MIS']['n'], 7)
            self.assertFalse(result['complete24'])

    def test_checkpoint_key_uses_any_best_then_fixed_scale(self):
        first = {'aggregates': {'ALL24': {'task_macro_any_best_top1_pct': 70.,
                                         'task_macro_train_scaled_excess': .02}}}
        second = {'aggregates': {'ALL24': {'task_macro_any_best_top1_pct': 70.,
                                          'task_macro_train_scaled_excess': .01}}}
        self.assertGreater(experiment.validation_key(second), experiment.validation_key(first))

    def test_bounded_prefetch_preserves_finite_request_order(self):
        class Store:
            def batch(self, problem, split, indices, pools):
                value = fixture_batch(len(indices))
                value['row_indices'] = torch.as_tensor(indices)
                return value

        requests = [(1, 'MIS', np.asarray([2, 1])), (2, 'MIS', np.asarray([0, 3]))]
        with experiment.BatchPrefetch(Store(), requests, {'MIS': [0, 1, 2]}, torch.device('cpu')) as batches:
            actual = list(batches)
        self.assertEqual([row[0] for row in actual], [1, 2])
        np.testing.assert_array_equal(actual[1][2]['row_indices'], [0, 3])

    def test_real_shared_model_ddp_switches_unused_domains(self):
        import torch.multiprocessing as mp
        with TemporaryDirectory() as directory:
            directory = Path(directory)
            mp.spawn(native_ddp_worker, args=(str(directory / 'gloo_init'), str(directory / 'ddp.pt')),
                     nprocs=2, join=True)
            actual = torch.load(directory / 'ddp.pt', map_location='cpu', weights_only=True)
            model = native_selector()
            objective = MultidomainObjective(model, dict(MIS=2., BPP=2., MKP=2.))
            optimizer = torch.optim.SGD(model.parameters(), lr=.03)
            for problem in ('MIS', 'BPP', 'MKP'):
                experiment.update_batch(objective, optimizer, torch.amp.GradScaler('cuda', enabled=False),
                    native_fixture(problem), experiment.Runtime(0, 1, torch.device('cpu')),
                    microbatch_size=8, global_count=8)
            for key in actual:
                torch.testing.assert_close(actual[key], model.state_dict()[key], rtol=1e-5, atol=2e-6)

    def test_prefetch_checkpoint_resume_preserves_stream_and_dropout(self):
        class DropoutSelector(TinySelector):
            def __init__(self, config):
                super().__init__()
                self.dropout = nn.Dropout(.2)

            def forward(self, batch):
                return super().forward(dict(batch, inputs={'x': self.dropout(batch['inputs']['x'])}))

        class Task:
            problem, objective_direction = 'MIS', 'max'
            candidate_solver_ids = ('a', 'b', 'c')
            label_arrays = {'native_objective': fixture_batch(8)['labels']['native_objective'].numpy(),
                            'observed_mask': np.ones((8, 3), dtype=bool)}

            def __len__(self):
                return 8

        class Store:
            def __init__(self, *args, **kwargs):
                pass

            def open(self, *args):
                return Task()

            def batch(self, problem, split, indices, pools):
                indices = torch.as_tensor(indices)
                full = fixture_batch(8)
                full['inputs']['x'] = full['inputs']['x'][indices]
                full['labels'] = {key: value[indices] if torch.is_tensor(value) else value
                                  for key, value in full['labels'].items()}
                full['solver_mask'] = full['solver_mask'][indices]
                full['row_indices'] = indices
                full['metadata'] = [full['metadata'][int(index)] for index in indices]
                return full

            def close_split(self, split):
                pass

            def close(self):
                pass

        with TemporaryDirectory() as directory:
            parent = Path(directory)
            geometry = parent / 'geometry.json'
            experiment.write_json(geometry, {})
            prepared = dict(data_version='CPU unit-test fixture', complete24=False,
                model=dict(problem_names=['MIS'], solver_names=['a', 'b', 'c'], pools={'MIS': [0, 1, 2]}),
                references={'MIS': dict(objective_scale=2., sbs_slot=0, sbs_solver='a')},
                fresh_initialization=str(parent / 'no_initialization.pt'), geometry_stats=str(geometry),
                rows={'MIS': dict(train=8, val=8, test=8)}, cache_root=str(parent))

            def arguments(root):
                return experiment.parser().parse_args(['train', '--root', str(root), '--device', 'cpu',
                    '--smoke-updates', '4', '--smoke-rows', '8', '--logical-batch-size', '3',
                    '--microbatch-size', '2', '--eval-batch-size', '3', '--eval-every', '2',
                    '--warmup-updates', '1'])

            complete, resumed = parent / 'complete', parent / 'resumed'
            for root in (complete, resumed):
                experiment.write_json(root / 'prepared.json', prepared)

            class StopAfterSavedCheckpoint(experiment.Runtime):
                def barrier(self):
                    if (resumed / 'last.pt').exists():
                        raise RuntimeError('Intentional CPU test interruption after checkpoint')

            with patch.object(experiment, 'DataStore', Store), patch.object(experiment, '_tracker', return_value=None):
                experiment.train(arguments(complete), experiment.Runtime(0, 1, torch.device('cpu')),
                                 model_factory=DropoutSelector)
                with self.assertRaisesRegex(RuntimeError, 'Intentional CPU test interruption'):
                    experiment.train(arguments(resumed), StopAfterSavedCheckpoint(0, 1, torch.device('cpu')),
                                     model_factory=DropoutSelector)
                args = arguments(resumed)
                args.resume = True
                experiment.train(args, experiment.Runtime(0, 1, torch.device('cpu')),
                                 model_factory=DropoutSelector)
            left = torch.load(complete / 'last.pt', map_location='cpu', weights_only=False)
            right = torch.load(resumed / 'last.pt', map_location='cpu', weights_only=False)
            self.assertEqual(left['stream_draws'], right['stream_draws'])
            self.assertEqual(right['stream_draws']['MIS'], 12)
            for key in left['model']:
                torch.testing.assert_close(left['model'][key], right['model'][key], rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()

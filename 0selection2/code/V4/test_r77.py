"""Small CPU protocol witnesses, without real data, GPU probes or solver execution."""

import tempfile
import unittest
from argparse import Namespace
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from . import r77_experiment as r77
from .r76_experiment import PART_KEYS, Runtime, fixed_sbs_metrics, update_batch, write_json
from .r77_data import GROUPS, IndexedCache, hierarchical_sample, parameter_hash, proportional_counts
from .test_r76 import CountingSGD, FakeCache, fixture_model


class ScalarObjective(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(.1))
        self.draws = []

    def forward(self, batch, consistency_weight):
        draw = torch.rand(())
        self.draws.append(float(draw))
        loss = ((self.weight * batch['x'] - batch['y']) ** 2).mean() + draw * 0.
        return loss, loss.detach().repeat(len(PART_KEYS) - 1)


class R77Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_input_only_hierarchical_subset(self):
        metadata = dict(scale=np.repeat([50, 100], [24, 36]),
            distribution=np.tile(['uniform', 'clustered', 'ring'], 20),
            capacity_type=np.tile(['scale', 'random'], 30), winner=np.zeros(60))
        first, receipt = hierarchical_sample(metadata, size=23, seed=2)
        metadata['winner'][:] = 99
        second, _ = hierarchical_sample(metadata, size=23, seed=2)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(len(np.unique(first)), 23)
        self.assertEqual(receipt['fields'], ['scale', 'distribution', 'capacity_type'])
        self.assertFalse(receipt['winner_used'])
        np.testing.assert_array_equal(np.bincount((metadata['scale'][first] == 100).astype(int)), [9, 14])
        self.assertEqual(sum(row['selected'] for row in receipt['strata']), 23)
        self.assertTrue(all(row['selected'] <= row['eligible'] for row in receipt['strata']))

    def test_optional_input_levels(self):
        rows, receipt = hierarchical_sample(dict(scale=np.repeat([20, 30], 5)), size=7)
        self.assertEqual(len(rows), 7)
        self.assertEqual(receipt['fields'], ['scale'])
        np.testing.assert_array_equal(proportional_counts([5, 5], 7), [4, 3])

    def test_tail_stream_and_resume(self):
        stream = r77.ShuffleStream(11, 77)
        rows = torch.cat([stream.take(8) for _ in range(6)])
        self.assertEqual(len(rows), 48)
        for start in range(0, 44, 11):
            self.assertEqual(sorted(rows[start:start + 11].tolist()), list(range(11)))
        resumed = r77.ShuffleStream(11, 77, draws=stream.draws)
        torch.testing.assert_close(stream.take(1024), resumed.take(1024))

    def test_task_axis_and_exact_budget(self):
        order = r77.task_order()
        self.assertEqual((r77.MAX_UPDATES, r77.WARMUP_UPDATES, r77.EVAL_EVERY), (49680, 3726, 1242))
        self.assertEqual(order, r77.task_order())
        self.assertTrue(all(a != b for a, b in zip(order, order[1:])))
        self.assertEqual(set(Counter(order).values()), {2760})
        self.assertEqual(set(Counter(order[:1242]).values()), {69})

    def test_identical_fixed_lr(self):
        a = [r77.learning_rate(update) for update in range(1, r77.MAX_UPDATES + 1)]
        b = [r77.learning_rate(update) for update in range(1, r77.MAX_UPDATES + 1)]
        self.assertEqual(a, b)
        self.assertEqual(a[3725], 1e-4)
        self.assertEqual(a[-1], 1e-6)
        self.assertTrue(all(x >= y for x, y in zip(a[3725:], a[3726:])))
        self.assertEqual(r77.micro_caps()['TSP'], 352)
        self.assertEqual(r77.micro_caps()['ATSP'], 1024)
        with self.assertRaises(ValueError):
            r77.micro_caps('4090', 353)

    def test_identical_parameters_separate_geometry_buffers(self):
        model, params = fixture_model()
        parameters = {name: value.detach().clone() for name, value in model.named_parameters()}
        initial = dict(model_params=params, trainable=parameters, trainable_sha256=parameter_hash(parameters))
        n = len(model.instance_encoder.geometry_residual.mean)
        a = r77.initialize_identical(initial, dict(mean=[0.] * n, std=[1.] * n), torch.device('cpu'))
        b = r77.initialize_identical(initial, dict(mean=[3.] * n, std=[2.] * n), torch.device('cpu'))
        self.assertEqual(parameter_hash(dict(a.named_parameters())), parameter_hash(dict(b.named_parameters())))
        self.assertFalse(torch.equal(a.instance_encoder.geometry_residual.mean, b.instance_encoder.geometry_residual.mean))

    def test_subset_gather_keeps_source_identity(self):
        source = FakeCache('TSP', size=5)
        view = IndexedCache(source, [1, 3, 4])
        view.batch(torch.tensor([2, 0]))
        self.assertEqual(source.seen, [4, 1])
        np.testing.assert_array_equal(view.raw['costs'], source.raw['costs'][[1, 3, 4]])
        self.assertEqual(source.size, 5)
        with self.assertRaises(IndexError):
            view.source_rows([3])

    def test_accumulation_global_mean_and_single_clip(self):
        batch = dict(ind=torch.zeros(5, dtype=torch.long), x=torch.arange(5) * .1, y=torch.ones(5) * .2)
        models = [ScalarObjective(), ScalarObjective()]
        for model, cap in zip(models, (5, 2)):
            optimizer = CountingSGD(model.parameters())
            scaler = torch.amp.GradScaler('cpu', enabled=False)
            with patch('torch.nn.utils.clip_grad_norm_', wraps=torch.nn.utils.clip_grad_norm_) as clip:
                _, retries = update_batch(model, optimizer, scaler, batch, .35, 5, 5,
                    Runtime(0, 1, torch.device('cpu')), microbatch_cap=cap)
                self.assertEqual(clip.call_count, 1)
            self.assertEqual((optimizer.steps, retries), (1, 0))
        torch.testing.assert_close(models[0].weight, models[1].weight)

    def test_overflow_replays_entire_logical_batch(self):
        objective = ScalarObjective()
        optimizer = CountingSGD(objective.parameters())
        scaler = torch.amp.GradScaler('cpu', init_scale=8., growth_interval=100)
        injected = [False]

        def overflow(gradient):
            if not injected[0]:
                injected[0] = True
                return torch.full_like(gradient, float('nan'))
            return gradient

        hook = objective.weight.register_hook(overflow)
        batch = dict(ind=torch.zeros(5, dtype=torch.long), x=torch.arange(5) * .1, y=torch.ones(5) * .2)
        _, retries = update_batch(objective, optimizer, scaler, batch, .35, 5, 5,
            Runtime(0, 1, torch.device('cpu')), microbatch_cap=2, max_retries=3)
        hook.remove()
        self.assertEqual((optimizer.steps, retries, scaler.get_scale()), (1, 1, 4.))
        self.assertEqual(objective.draws[:3], objective.draws[3:])

    def test_common_sbs_and_no_prelock_test(self):
        raw = dict(costs=np.asarray([[1., 8.], [4., 2.]]), winner=np.asarray([0, 1]),
            pool=['a', 'b'], pool_ids=[0, 1])
        scores = np.asarray([[1., 0.], [1., 0.]])
        common = dict(selection_split='train', sbs_solver='b', sbs_slot=1, mean_cost_by_solver={'b': 5.})
        own = dict(selection_split='train', sbs_solver='a', sbs_slot=0, mean_cost_by_solver={'a': 2.5})
        primary, _ = fixed_sbs_metrics(scores, raw, common)
        secondary, _ = fixed_sbs_metrics(scores, raw, own)
        self.assertEqual((primary['vs_sbs_pct'], secondary['vs_sbs_pct']), (-50., 0.))
        with tempfile.TemporaryDirectory() as directory:
            args = Namespace(root=Path(directory), arm='A', data_root=Path('/fixture'))
            with self.assertRaises(PermissionError):
                r77.authorize_test(args)


if __name__ == '__main__':
    unittest.main()

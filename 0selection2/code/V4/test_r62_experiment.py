"""Seconds-scale CPU fixtures for partial comparisons, transfer and exact exposure."""

import copy
import io
import tempfile
import unittest
from argparse import Namespace
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from . import r62_pretraining as pretraining
from .multitask_probe import state_hash
from .r59_experiment import atomic_save
from .train import set_seed


class TinySelector(nn.Module):
    def __init__(self):
        super().__init__()
        self.instance_encoder = nn.Linear(2, 4)
        self.solver_encoder = nn.Embedding(3, 4)
        self.joint = nn.Linear(4, 4)
        self.decoder = nn.Linear(4, 4)
        self.score_head = nn.Linear(4, 1)
        self.dropout = nn.Dropout(.1)
        self.calls = []

    def forward(self, batch):
        self.calls.append({k: v.detach().clone() if torch.is_tensor(v) else v for k, v in batch.items()})
        encoded = self.instance_encoder(batch['node'].mean(1))
        solvers = self.solver_encoder(batch['pool_ids'])
        joint = torch.tanh(self.joint(encoded[:, None] + solvers[None]))
        utility = self.score_head(self.dropout(torch.tanh(self.decoder(joint)))).squeeze(-1)
        return dict(pair_margin=utility[:, :, None] - utility[:, None, :],
                    solver_mask=torch.ones_like(utility, dtype=torch.bool))


def initialize():
    set_seed(2)
    return TinySelector(), dict(fixture=True)


class Dataset:
    def __init__(self, nodes):
        self.nodes = np.asarray(nodes)

    def gather(self, indices, device='cpu'):
        indices = np.asarray(indices)
        sizes = torch.as_tensor(self.nodes[indices], device=device)
        maximum = int(sizes.max())
        node = torch.arange(len(indices) * maximum * 2, device=device).reshape(len(indices), maximum, 2).float() / 50
        costs = torch.tensor([[1., 2., 3.]], dtype=torch.float64, device=device).repeat(len(indices), 1)
        observed = torch.ones_like(costs, dtype=torch.bool)
        batch = dict(node=node, pool_ids=torch.arange(3, device=device), n=sizes,
                     costs=costs.float(), observed_mask=observed, source_indices=torch.as_tensor(indices))
        return batch, costs, observed


class RetryScaler:
    def __init__(self):
        self.value, self.attempt, self.successes = 128., 0, 0

    def is_enabled(self):
        return False

    def scale(self, loss):
        return loss

    def unscale_(self, optimizer):
        pass

    def get_scale(self):
        return self.value

    def step(self, optimizer):
        if self.attempt:
            optimizer.step()
            self.successes += 1

    def update(self):
        if not self.attempt:
            self.value /= 2
        self.attempt += 1


class R62Tests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_fp64_direction_missing_ties_and_per_query_mean(self):
        costs = torch.tensor([[1., 1. + 1e-10, float('nan')], [3., 1., 2.],
                              [2., 2., float('inf')], [5., float('nan'), float('nan')]], dtype=torch.float64)
        observed = torch.isfinite(costs)
        margins = torch.arange(36).reshape(4, 3, 3).float().div(10).requires_grad_()
        output = dict(pair_margin=margins, solver_mask=torch.ones(4, 3, dtype=torch.bool))
        loss, stats = pretraining.pair_objective(output, costs, observed)
        expected = (F.binary_cross_entropy_with_logits(margins[0, 0, 1], torch.tensor(1.)) +
                    F.binary_cross_entropy_with_logits(margins[1, [0, 0, 1], [1, 2, 2]],
                                                      torch.tensor([0., 0., 1.]))) / 4
        torch.testing.assert_close(loss, expected, rtol=0, atol=0)
        self.assertEqual(stats['pairs'], 4)
        self.assertEqual(stats['comparison_rows'], 2)
        self.assertEqual(stats['rows'], 4)
        loss.backward()
        self.assertLess(float(margins.grad[0, 0, 1]), 0.)
        self.assertGreater(float(margins.grad[1, 0, 1]), 0.)
        self.assertEqual(float(margins.grad[2:].abs().sum()), 0.)
        with self.assertRaises(ValueError):
            pretraining.pair_objective(output, costs.float(), observed)
        output['solver_mask'][0, 2] = False
        with self.assertRaisesRegex(ValueError, 'remove legal candidates'):
            pretraining.pair_objective(output, costs, observed)

    def test_single_forward_retry_all_modules_and_fresh_head_transfer(self):
        batch, costs, observed = Dataset([3, 4]).gather([0, 1])
        model, _ = initialize()
        fresh = copy.deepcopy(model.state_dict())
        expected = copy.deepcopy(model)
        rng = pretraining.capture_rng()
        scaler = RetryScaler()
        with redirect_stdout(io.StringIO()):
            values = pretraining.update_batch(model, torch.optim.SGD(model.parameters(), lr=.02),
                                              scaler, batch, costs, observed)
        self.assertEqual(values['skipped'], 1)
        self.assertEqual(scaler.successes, 1)
        self.assertEqual(len(model.calls), 2)
        self.assertEqual(state_hash(model.calls[0]), state_hash(model.calls[1]))
        self.assertFalse(set(model.calls[0]) & {'costs', 'observed_mask', 'source_indices'})
        self.assertEqual(len(model.calls[0]['pool_ids']), 3)
        pretraining.restore_rng(rng)
        plain = torch.amp.GradScaler('cuda', enabled=False)
        pretraining.update_batch(expected, torch.optim.SGD(expected.parameters(), lr=.02), plain, batch, costs, observed)
        self.assertEqual(len(expected.calls), 1)
        self.assertEqual(state_hash(model.state_dict()), state_hash(expected.state_dict()))
        for name in ('instance_encoder', 'solver_encoder', 'joint', 'decoder', 'score_head'):
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in getattr(model, name).parameters()))
        trained = model.state_dict()
        transferred = pretraining.transfer_state(trained, fresh)
        for key in fresh:
            torch.testing.assert_close(transferred[key], fresh[key] if key.startswith('score_head.') else trained[key],
                                       rtol=0, atol=0)
        self.assertNotEqual(state_hash(transferred), state_hash(fresh))

    def test_two_pass_size_matching_tail_and_atomic_resume(self):
        datasets = {'A': {'OVRP': Dataset([3, 4, 3])}, 'B': {'OVRP': Dataset([3, 4] * 6 + [3])}}
        schedule = pretraining.matched_schedule(datasets, batch_size=4)
        for pass_index in range(2):
            a = schedule['permutations']['A'][pass_index]['OVRP']
            b = schedule['permutations']['B'][pass_index]['OVRP']
            self.assertEqual(sorted(b.tolist()), list(range(13)))
            self.assertEqual(len(b[12:16]), 1)
            np.testing.assert_array_equal(datasets['A']['OVRP'].nodes[a], datasets['B']['OVRP'].nodes[b])
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            initial = initialize()[0].state_dict()
            atomic_save(root / 'selector_initial.pt', initial)
            protocol = dict(selector_initial_hash=state_hash(initial), source_hashes={}, auxiliary_manifest={},
                            schedule_hash=state_hash(schedule))
            args = Namespace(root=root, resume=False, device='cpu', seed=2, batch_size=4, wandb=False)
            with redirect_stdout(io.StringIO()):
                complete = pretraining.train('B', args, root / 'complete', datasets, schedule, initialize, protocol)
            original_update, counter = pretraining.update_batch, 0

            def interrupted(*arguments):
                nonlocal counter
                counter += 1
                if counter == 5:
                    raise RuntimeError('fixture interrupted after pass one')
                return original_update(*arguments)

            with patch.object(pretraining, 'update_batch', side_effect=interrupted), redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(RuntimeError, 'fixture interrupted'):
                    pretraining.train('B', args, root / 'resumed', datasets, schedule, initialize, protocol)
            args.resume = True
            with redirect_stdout(io.StringIO()):
                resumed = pretraining.train('B', args, root / 'resumed', datasets, schedule, initialize, protocol)
            self.assertEqual(complete['successful_updates'], 8)
            self.assertEqual(resumed['coverage']['rows'], 26)
            self.assertEqual(complete['model_hash'], resumed['model_hash'])
            checkpoints = [torch.load(root / name / 'last.pt', weights_only=False) for name in ('complete', 'resumed')]
            for key in ('model', 'optimizer', 'rng', 'cursor', 'all_counts'):
                self.assertEqual(state_hash(checkpoints[0][key]), state_hash(checkpoints[1][key]))


if __name__ == '__main__':
    unittest.main()

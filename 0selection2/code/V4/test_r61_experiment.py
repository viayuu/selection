"""Small CPU fixtures for fixed route learning, paired exposure and resumption."""

import copy
import io
import json
import tempfile
import unittest
from argparse import Namespace
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from . import r61_experiment as experiment
from . import r61_pretraining as behavior
from .multitask_probe import state_hash
from .r54_model import successor_objective
from .r59_experiment import atomic_save
from .train import set_seed


class TinyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 8)
        self.dropout = nn.Dropout(.1)
        self.calls = []

    def forward(self, batch):
        self.calls.append({k: v.detach().clone() if torch.is_tensor(v) else v for k, v in batch.items()})
        return self.dropout(self.linear(batch['node'])), batch['node_mask']


def initialize():
    set_seed(2)
    return behavior.BehaviorPretrainer(TinyEncoder(), 8)


class Dataset:
    def __init__(self, nodes):
        self.nodes = np.asarray(nodes)
        self.manifest = dict(nodes=self.nodes.tolist(), split='train')

    def gather(self, indices, device='cpu'):
        sizes = torch.as_tensor(self.nodes[np.asarray(indices)], device=device)
        maximum = int(sizes.max())
        values = torch.arange(len(indices) * maximum * 2, device=device).reshape(len(indices), maximum, 2).float() / 50
        batch = dict(node=values, node_mask=torch.arange(maximum, device=device)[None] < sizes[:, None], n=sizes,
                     costs=torch.zeros(len(indices), 2, device=device), ind=torch.zeros(len(indices), device=device),
                     successor=torch.full((len(indices), 2, maximum - 1), 777, device=device))
        targets = torch.full((len(indices), 2, maximum - 1), -100, dtype=torch.long, device=device)
        for row, size in enumerate(sizes.tolist()):
            route = torch.arange(2, size + 1, device=device)
            route[-1] = 0
            targets[row, :, :size - 1] = route
        return batch, targets


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


class R61Tests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_exact_r54_loss_one_forward_and_retry_same_rng(self):
        batch, targets = Dataset([4, 5]).gather([0, 1])
        model = initialize()
        expected = copy.deepcopy(model)
        rng = behavior.capture_rng()
        scaler = RetryScaler()
        with redirect_stdout(io.StringIO()):
            values = behavior.update_batch(model, torch.optim.SGD(model.parameters(), lr=.02), scaler, batch, targets)
        self.assertEqual(values['skipped'], 1)
        self.assertEqual(scaler.successes, 1)
        self.assertEqual(len(model.instance_encoder.calls), 2)
        self.assertEqual(state_hash(model.instance_encoder.calls[0]), state_hash(model.instance_encoder.calls[1]))
        self.assertFalse(set(model.instance_encoder.calls[0]) & {'costs', 'ind', 'successor'})
        behavior.restore_rng(rng)
        optimizer = torch.optim.SGD(expected.parameters(), lr=.02)
        logits, mask = expected(batch)
        loss, parts = successor_objective(logits, targets, mask)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(expected.parameters(), 1.)
        optimizer.step()
        self.assertEqual(state_hash(model.state_dict()), state_hash(expected.state_dict()))
        self.assertAlmostEqual(values['loss'], float(loss), places=6)
        self.assertAlmostEqual(values['successor_accuracy'], float((parts['moel_accuracy'] + parts['mtl_accuracy']) / 2))
        for head in model.successor_heads:
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in head.parameters()))
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.instance_encoder.parameters()))

    def test_two_complete_passes_exact_sizes_tails_and_rng_isolation(self):
        datasets = {'A': {'OVRP': Dataset([4, 5, 4, 5])},
                    'B': {'OVRP': Dataset([4, 5] * 5)}}
        set_seed(93)
        rng = state_hash(behavior.capture_rng())
        plan = behavior.matched_schedule(datasets, batch_size=3)
        self.assertEqual(rng, state_hash(behavior.capture_rng()))
        self.assertEqual([len(order) for order in plan['orders']], [4, 4])
        for pass_index in range(2):
            a = plan['permutations']['A'][pass_index]['OVRP']
            b = plan['permutations']['B'][pass_index]['OVRP']
            self.assertEqual(sorted(b.tolist()), list(range(10)))
            np.testing.assert_array_equal(datasets['A']['OVRP'].nodes[a], datasets['B']['OVRP'].nodes[b])
            self.assertEqual(len(b[9:12]), 1)
        with self.assertRaisesRegex(ValueError, 'No original TRAIN'):
            behavior.matched_schedule({'A': {'OVRP': Dataset([4])}, 'B': {'OVRP': Dataset([5])}})
        self.assertEqual(behavior.learning_rate(10559, 10560), 1e-4)
        self.assertAlmostEqual(behavior.learning_rate(21119, 10560), 1e-5)

    def test_pretrain_atomic_resume_final_encoder_only(self):
        datasets = {'A': {'OVRP': Dataset([4, 5])}, 'B': {'OVRP': Dataset([4, 5] * 3)}}
        schedule = behavior.matched_schedule(datasets, batch_size=2)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            initial = initialize().state_dict()
            atomic_save(root / 'pretrain_initial.pt', initial)
            protocol = dict(pretrain_initial_hash=state_hash(initial), source_hashes={},
                            behavior_manifest={}, schedule_hash=state_hash(schedule))
            args = Namespace(root=root, resume=False, device='cpu', seed=2, batch_size=2, wandb=False)
            with redirect_stdout(io.StringIO()):
                complete = behavior.train('B', args, root / 'complete', datasets, schedule, initialize, protocol)
            original_update = behavior.update_batch
            counter = 0

            def interrupted(*arguments):
                nonlocal counter
                counter += 1
                if counter == 4:
                    raise RuntimeError('fixture interruption after pass one')
                return original_update(*arguments)

            with patch.object(behavior, 'update_batch', side_effect=interrupted), redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(RuntimeError, 'fixture interruption'):
                    behavior.train('B', args, root / 'resumed', datasets, schedule, initialize, protocol)
            args.resume = True
            with redirect_stdout(io.StringIO()):
                resumed = behavior.train('B', args, root / 'resumed', datasets, schedule, initialize, protocol)
            self.assertEqual(complete['successful_updates'], 6)
            self.assertEqual(resumed['successful_updates'], 6)
            self.assertEqual(complete['encoder_hash'], resumed['encoder_hash'])
            checkpoints = [torch.load(root / name / 'last.pt', weights_only=False) for name in ('complete', 'resumed')]
            for key in ('model', 'optimizer', 'rng', 'successful_updates', 'cursor'):
                self.assertEqual(state_hash(checkpoints[0][key]), state_hash(checkpoints[1][key]))
            encoder = torch.load(root / 'resumed/encoder_final.pt', weights_only=False)
            self.assertFalse(any('successor' in key for key in encoder))
            self.assertEqual(set(encoder), set(initialize().instance_encoder.state_dict()))

    def test_fixed_plan_and_common_finetune_lr_trace(self):
        config = experiment.planned_config(Namespace(epochs=40, eval_batch_size=128))
        self.assertEqual(config['pretrain']['successful_updates'], 21120)
        self.assertEqual(config['execution_order'], ['A_pretrain', 'A_finetune', 'B_pretrain', 'B_finetune'])
        self.assertFalse(config['test_read'])
        self.assertEqual(config['loss']['ce'], .35)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            arm = root / experiment.RUN_NAMES['A']
            arm.mkdir()
            experiment.atomic_json(arm / 'result.json', dict(epochs_completed=3, successful_updates=4266))
            experiment.atomic_json(arm / 'history.json', [dict(optimization=dict(lr=lr)) for lr in (1e-5, 1e-4, 5e-5)])
            shared = experiment.shared_replay_schedule(root)
            self.assertEqual(shared['lr'], [1e-5, 1e-4, 5e-5])
            self.assertEqual(shared['successful_updates'], 4266)
        with patch.dict('os.environ', {}, clear=True), redirect_stdout(io.StringIO()):
            with self.assertRaises(SystemExit):
                experiment.main(['--stage', 'train'])


if __name__ == '__main__':
    unittest.main()

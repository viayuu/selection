"""Small CPU fixtures for observed pair loss, one update, matched exposure and resume."""

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

from code.unified_selector.registry import GLOBAL_SOLVERS, POOLS, PROBLEMS
from . import r60_experiment as experiment
from . import r60_objective as objective
from .multitask_probe import state_hash
from .pairwise_objective import comparison_targets, inputs_only, predicted_top3, single_objective
from .r42_experiment import paired_schedule, symmetric_kl
from .tensor_loader import TensorBatchLoader


class TinySelector(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = nn.Linear(2, len(GLOBAL_SOLVERS))
        self.dropout = nn.Dropout(.1)
        self.calls = []

    def forward(self, batch):
        self.calls.append({k: v.detach().clone() if torch.is_tensor(v) else v for k, v in batch.items()})
        mask = batch['node_mask'][..., None]
        value = (batch['node'] * mask).sum(1) / mask.sum(1)
        score = self.head(self.dropout(value))[:, batch['pool_ids']]
        margin = score[:, :, None] - score[:, None, :]
        diagonal = torch.eye(score.size(1), dtype=torch.bool)[None]
        logits = margin.masked_fill(diagonal, float('inf')).min(-1).values
        return dict(logits=logits, pair_margin=margin, solver_mask=torch.ones_like(score, dtype=torch.bool))


def initialize(geometry, device, seed=2):
    experiment.set_seed(seed)
    return TinySelector().to(device), dict(fixture=True, pair_mode='score_difference')


def batch_and_raw(problem='OVRP', size=3):
    pool = POOLS[problem]
    ids = [GLOBAL_SOLVERS.index(name) for name in pool]
    costs = np.full((size, len(pool)), 3., dtype=np.float64)
    if problem in experiment.MVRP:
        costs[:, pool.index('RELD_MOEL')] = [1., 2., 1.][:size]
        costs[:, pool.index('RELD_MTL')] = [1. + 1e-10, 1., 1.][:size]
    else:
        costs[:, 0], costs[:, 1] = 1., 2.
    lengths = torch.tensor([3, 4, 3][:size])
    nodes = torch.arange(size * 8, dtype=torch.float32).reshape(size, 4, 2) / 20.
    batch = dict(node=nodes, node_mask=torch.arange(4)[None] < lengths[:, None], n=lengths,
                 pool_ids=torch.tensor(ids), costs=torch.tensor(costs, dtype=torch.float32),
                 ind=torch.tensor(costs.argmin(-1)), **comparison_targets(costs))
    raw = dict(costs=costs, winner=costs.argmin(-1), pool=pool, pool_ids=ids)
    return batch, raw


class ExpandedFixture:
    def __init__(self, problem):
        self.batch, raw = batch_and_raw(problem)
        self.size = 27
        self.nodes = np.tile([3, 4, 3], 9)
        self.pair_costs = np.tile([[1., 1. + 1e-10], [2., 1.], [1., 1.]], (9, 1))

    def gather(self, indices, device):
        indices = torch.as_tensor(indices)
        local = indices % 3
        loader = TensorBatchLoader(self.batch, 2, False, False)
        clean = inputs_only(experiment.gather_batch(loader, local))
        return clean, torch.tensor(self.pair_costs[indices.numpy()], dtype=torch.float64)


def fixture_data():
    loaders, raw, reference = ({'train': {}, 'val': {}} for _ in range(3))
    for split in loaders:
        for problem in PROBLEMS:
            batch, record = batch_and_raw(problem)
            loaders[split][problem] = TensorBatchLoader(batch, 2, False, False)
            raw[split][problem] = record
            reference[split][problem] = np.tile([0, 1], (3, 1))
    datasets = {p: ExpandedFixture(p) for p in experiment.MVRP}
    sizes = {p: 3 for p in PROBLEMS}
    extras = experiment.extra_schedule(datasets, {p: loaders['train'][p].lengths for p in datasets},
                                       epochs=5, rows_per_epoch=3)
    model, params = initialize({}, 'cpu')
    protocol = dict(sizes=sizes, updates_per_epoch={p: 2 for p in PROBLEMS},
                    initialization_checks={a: dict(initial_model_hash=state_hash(model.state_dict()))
                                           for a in experiment.RUN_NAMES},
                    model_params={a: params for a in experiment.RUN_NAMES}, source_hashes={}, test_read=False)
    return protocol, loaders, raw, {}, paired_schedule(sizes, 5, 2, 2), extras, datasets, reference


class ForcedRetryScaler:
    def __init__(self):
        self.scale_value, self.attempt, self.steps = 128., 0, 0

    def is_enabled(self):
        return False

    def scale(self, value):
        return value

    def unscale_(self, optimizer):
        pass

    def get_scale(self):
        return self.scale_value

    def step(self, optimizer):
        if self.attempt:
            optimizer.step()
            self.steps += 1

    def update(self):
        if not self.attempt:
            self.scale_value /= 2
        self.attempt += 1


class ShortController:
    def __init__(self):
        self.observed = 0

    def observe(self, cost, epoch):
        self.observed += 1
        return epoch == 3, epoch == 4


class R60ExperimentTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_fp64_pair_sign_ties_and_full_pool(self):
        batch, raw = batch_and_raw()
        ids = batch['pool_ids'].flip(0)
        margins = torch.zeros(3, 7, 7, requires_grad=True)
        output = dict(pair_margin=margins, solver_mask=torch.ones(3, 7, dtype=torch.bool))
        costs = torch.tensor([[1., 1. + 1e-10], [2., 1.], [1., 1.]], dtype=torch.float64)
        loss, stats = objective.pair_objective(output, ids, costs)
        self.assertEqual(stats['extra_aux_samples'], 2)
        loss.backward()
        moel = int((ids == GLOBAL_SOLVERS.index('RELD_MOEL')).nonzero()[0])
        mtl = int((ids == GLOBAL_SOLVERS.index('RELD_MTL')).nonzero()[0])
        self.assertLess(float(margins.grad[0, moel, mtl]), 0)
        self.assertGreater(float(margins.grad[1, moel, mtl]), 0)
        self.assertEqual(float(margins.grad[2].abs().sum()), 0)
        zero, stats = objective.pair_objective(output, ids, torch.ones(3, 2, dtype=torch.float64))
        self.assertEqual(float(zero), 0)
        self.assertEqual(stats['extra_aux_samples'], 0)
        with self.assertRaises(ValueError):
            objective.pair_objective(output, ids, costs.float())
        with self.assertRaises(ValueError):
            objective.pair_objective(output, ids[:2], costs)

    def test_two_plus_one_forwards_and_single_combined_update(self):
        batch, raw = batch_and_raw()
        model, _ = initialize({}, 'cpu')
        manual = copy.deepcopy(model)
        extra = (inputs_only(batch), torch.from_numpy(raw['costs'][:, [4, 5]]))
        optimizer = torch.optim.SGD(model.parameters(), lr=.02)
        expected_optimizer = torch.optim.SGD(manual.parameters(), lr=.02)
        rng = experiment.capture_rng()
        with patch.object(optimizer, 'step', wraps=optimizer.step) as step:
            values = objective.update_batch(model, optimizer, torch.amp.GradScaler('cuda', enabled=False),
                                            batch, .2, extra)
        self.assertEqual(step.call_count, 1)
        self.assertEqual(len(model.calls), 3)
        self.assertTrue(all(not set(call) & set(comparison_targets(raw['costs'])) for call in model.calls))
        self.assertTrue(all('costs' not in call and 'ind' not in call for call in model.calls))
        experiment.restore_rng(rng)
        first, second, third = [manual(inputs_only(batch)) for _ in range(3)]
        focus = predicted_top3(first['logits'], second['logits'], batch['pool_ids'], first['solver_mask'])
        first_loss, _ = single_objective(first, batch, focus)
        second_loss, _ = single_objective(second, batch, focus)
        pair, _ = objective.pair_objective(third, batch['pool_ids'], extra[1])
        loss = .5 * (first_loss + second_loss) + .2 * symmetric_kl(first['logits'], second['logits']) + .35 * pair
        loss.backward()
        torch.nn.utils.clip_grad_norm_(manual.parameters(), 1.)
        expected_optimizer.step()
        self.assertEqual(state_hash(model.state_dict()), state_hash(manual.state_dict()))
        self.assertAlmostEqual(values['loss'], float(loss), places=6)
        plain, _ = initialize({}, 'cpu')
        objective.update_batch(plain, torch.optim.SGD(plain.parameters(), lr=.02),
                               torch.amp.GradScaler('cuda', enabled=False), batch, .2)
        self.assertEqual(len(plain.calls), 2)

    def test_amp_retry_replays_both_batches_and_rng(self):
        batch, raw = batch_and_raw()
        extra = (inputs_only(batch), torch.from_numpy(raw['costs'][:, [4, 5]]))
        model, _ = initialize({}, 'cpu')
        expected = copy.deepcopy(model)
        rng = experiment.capture_rng()
        scaler = ForcedRetryScaler()
        with redirect_stdout(io.StringIO()):
            values = objective.update_batch(model, torch.optim.SGD(model.parameters(), lr=.02), scaler, batch, .2, extra)
        self.assertEqual(values['skipped'], 1)
        self.assertEqual(scaler.steps, 1)
        self.assertEqual(len(model.calls), 6)
        for first, repeated in zip(model.calls[:3], model.calls[3:]):
            self.assertEqual(state_hash(first), state_hash(repeated))
        experiment.restore_rng(rng)
        objective.update_batch(expected, torch.optim.SGD(expected.parameters(), lr=.02),
                               torch.amp.GradScaler('cuda', enabled=False), batch, .2, extra)
        self.assertEqual(state_hash(model.state_dict()), state_hash(expected.state_dict()))

    def test_auxiliary_traversal_size_matching_tails_and_rng_isolation(self):
        datasets = {'OVRP': ExpandedFixture('OVRP')}
        old = {'OVRP': torch.tensor([3, 4, 3])}
        before = torch.get_rng_state().clone()
        plan = experiment.extra_schedule(datasets, old, 10, rows_per_epoch=3)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        np.testing.assert_array_equal(torch.cat(plan['B']['OVRP'][:9]).sort().values, np.arange(27))
        for a, b in zip(plan['A']['OVRP'], plan['B']['OVRP']):
            np.testing.assert_array_equal(old['OVRP'][a], datasets['OVRP'].nodes[b.numpy()])
            self.assertEqual([len(a[:2]), len(a[2:])], [2, 1])
        self.assertEqual(state_hash(plan), state_hash(experiment.extra_schedule(datasets, old, 10, rows_per_epoch=3)))
        datasets['OVRP'].pair_costs[:] = 999.
        self.assertEqual(state_hash(plan), state_hash(experiment.extra_schedule(datasets, old, 10, rows_per_epoch=3)))

    def test_epoch_atomic_resume_and_b_follows_a_lr_and_horizon(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = fixture_data()
            def args(path):
                return Namespace(root=path, epochs=5, batch_size=2, eval_batch_size=2,
                                 device='cpu', seed=2, resume=True, wandb=False)
            with patch.object(experiment, 'initialize', initialize), patch.object(experiment, 'CostController', ShortController), \
                 patch.object(experiment, 'print_eval'), redirect_stdout(io.StringIO()):
                experiment.run('A', args(root / 'full'), data)
                experiment.run('B', args(root / 'full'), data)
                experiment.run('A', args(root / 'resume'), data)
                publish = experiment.publish_best
                def interrupted(path, checkpoint):
                    if checkpoint['epoch'] == 1:
                        raise RuntimeError('fixture interruption after epoch commit')
                    return publish(path, checkpoint)
                with patch.object(experiment, 'publish_best', interrupted), self.assertRaisesRegex(RuntimeError, 'fixture interruption'):
                    experiment.run('B', args(root / 'resume'), data)
                experiment.run('B', args(root / 'resume'), data)
            a = json.loads((root / 'full/A_replay_seed2/result.json').read_text())
            b = json.loads((root / 'resume/B_expanded_seed2/result.json').read_text())
            self.assertEqual((a['epochs_completed'], b['epochs_completed']), (4, 4))
            self.assertEqual(a['successful_updates'], b['successful_updates'])
            self.assertEqual(b['successful_updates'], 4 * 36)
            self.assertEqual(b['extra_rows_by_problem'], {p: 12 for p in experiment.MVRP})
            histories = [json.loads((root / 'full' / name / 'history.json').read_text()) for name in experiment.RUN_NAMES.values()]
            self.assertEqual([r['optimization']['lr'] for r in histories[0]],
                             [r['optimization']['lr'] for r in histories[1]])
            checkpoints = [torch.load(root / run / 'B_expanded_seed2/last.pt', weights_only=False) for run in ('full', 'resume')]
            for key in ('model', 'optimizer', 'scaler', 'rng', 'controller', 'successful_updates',
                        'updates_by_problem', 'extra_rows_by_problem', 'best_model'):
                self.assertEqual(state_hash(checkpoints[0][key]), state_hash(checkpoints[1][key]), key)
            self.assertEqual(checkpoints[0]['controller']['observed'], 0)
            self.assertIsNotNone(b['best']['train'])
            self.assertIsNotNone(b['final']['train'])
            self.assertFalse(b['test_read'])
            for problem in PROBLEMS:
                with np.load(root / 'resume/B_expanded_seed2/val_predictions/best' / f'{problem}.npz') as saved:
                    self.assertEqual(saved['costs'].dtype, np.float64)
                    self.assertTrue({'indices', 'logits', 'pair_margin', 'pred', 'winner', 'costs', 'pool_ids', 'pool', 'nodes'} <= set(saved.files))

    def test_plan_and_cli_keep_fixed_rules_and_no_test_stage(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            experiment.main(['--stage', 'plan', '--root', directory])
            value = json.loads((Path(directory) / 'planned_configs.json').read_text())
            self.assertEqual(value['arms'], dict(A='A_replay_seed2', B='B_expanded_seed2'))
            self.assertEqual((value['learning_rate'], value['auxiliary_weight']), (1e-4, .35))
            self.assertEqual((value['max_epochs'], value['min_epochs'], value['stop_patience']), (40, 15, 8))
            self.assertFalse(value['test_read'])
            with patch.dict('os.environ', {}, clear=True), patch.object(experiment, 'prepare') as prepare, \
                 patch.object(experiment, 'configure_torch') as configure, self.assertRaises(SystemExit):
                experiment.main(['--stage', 'train', '--root', directory])
            prepare.assert_not_called()
            configure.assert_not_called()


if __name__ == '__main__':
    unittest.main()

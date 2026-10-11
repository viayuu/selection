"""CPU fixtures for R59 policy, optimizer groups, atomic resume and validation I/O."""

import copy
import io
import json
import tempfile
import time
import unittest
from argparse import Namespace
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from code.unified_selector.registry import PROBLEMS
from . import r59_experiment as experiment
from .multitask_probe import file_hash, state_hash
from .pairwise_objective import comparison_targets
from .r42_experiment import paired_schedule
from .r43_experiment import update_batch
from .tensor_loader import TensorBatchLoader


class TinySelector(nn.Module):
    def __init__(self):
        super().__init__()
        self.native = nn.Linear(2, 3)
        self.dropout = nn.Dropout(.1)
        self.head = nn.Linear(3, 2)

    def forward(self, batch):
        mask = batch['node_mask'][..., None]
        x = (batch['node'] * mask).sum(1) / mask.sum(1)
        score = self.head(self.dropout(self.native(x)))
        margin = score[:, :, None] - score[:, None, :]
        logits = margin.masked_fill(torch.eye(2, dtype=torch.bool)[None], float('inf')).min(-1).values
        return dict(logits=logits, pair_margin=margin, solver_mask=torch.ones_like(score, dtype=torch.bool))


def initialize(arm, geometry, device, seed):
    experiment.set_seed(seed)
    model = TinySelector().to(device)
    if arm == 'B':
        with torch.no_grad():
            model.native.weight.add_(.2)
    return model, dict(fixture=True, pair_mode='score_difference')


def fixture_data(epochs=5):
    loaders, raw, reference = ({'train': {}, 'val': {}} for _ in range(3))
    for split in loaders:
        size = 3 if split == 'train' else 2
        for problem in PROBLEMS:
            costs = np.array([[1., 1. + 1e-10] if i % 2 == 0 else [3., 1.] for i in range(size)], dtype=np.float64)
            winner = costs.argmin(-1)
            batch = dict(node=torch.arange(size * 6, dtype=torch.float32).reshape(size, 3, 2) / 10.,
                         node_mask=torch.ones(size, 3, dtype=torch.bool), n=torch.full((size,), 3),
                         pool_ids=torch.tensor([0, 1]), costs=torch.tensor(costs, dtype=torch.float32),
                         ind=torch.tensor(winner), **comparison_targets(costs))
            loaders[split][problem] = TensorBatchLoader(batch, 2, False, False)
            raw[split][problem] = dict(costs=costs, winner=winner, pool_ids=[0, 1], pool=['a', 'b'])
            reference[split][problem] = np.tile([0, 1], (size, 1))
    sizes = {p: 3 for p in PROBLEMS}
    params, initial = {}, {}
    for arm in experiment.RUN_NAMES:
        model, params[arm] = initialize(arm, {}, 'cpu', 2)
        initial[arm] = dict(initial_model_hash=state_hash(model.state_dict()))
    protocol = dict(sizes=sizes, updates_per_epoch={p: 2 for p in PROBLEMS},
                    initialization_checks=initial, model_params=params, source_hashes={},
                    data_manifest={'train': {}, 'val': {}}, test_read=False)
    return protocol, loaders, raw, {}, paired_schedule(sizes, epochs, 2, 2), reference


def args(root, epochs=5):
    return Namespace(root=Path(root), epochs=epochs, batch_size=2, eval_batch_size=2,
                     seed=2, device='cpu', resume=True, wandb=False)


class R59ExperimentTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_plan_reuses_exact_rules_and_has_no_test_stage(self):
        with tempfile.TemporaryDirectory() as directory:
            with redirect_stdout(io.StringIO()):
                experiment.main(['--stage', 'plan', '--root', directory])
            config = json.loads((Path(directory) / 'planned_configs.json').read_text())
            self.assertEqual(config['arms'], {'A': 'A_random_seed2', 'B': 'B_pretrained_seed2'})
            self.assertEqual((config['seed'], config['batch_size'], config['max_epochs']), (2, 128, 40))
            self.assertEqual((config['min_epochs'], config['stop_patience'], config['lr_patience']), (15, 8, 3))
            self.assertEqual(config['loss'], dict(ce=.35, comparison=.35, risk=.02, cost_scale=.01,
                                                focus_weights=dict(true3=.5, predicted3=.25, all_pairs=.25)))
            self.assertFalse(config['test_read'])
            self.assertFalse(config['expansion_inputs'])
            self.assertEqual(config['checkpoint_rule'], 'strict minimum validation macro_vs_sbs_pct')
            self.assertIs(experiment.update_batch, update_batch)
            for stage in ('test', 'test-completion'):
                with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    experiment.main(['--stage', stage, '--root', directory])

    def test_training_without_baton_fails_before_work(self):
        with patch.dict('os.environ', {}, clear=True), patch.object(experiment, 'prepare') as prepare:
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                experiment.main(['--stage', 'all'])
        prepare.assert_not_called()

    def test_both_arms_partition_all_parameters_and_keep_lr_ratio(self):
        with patch.object(experiment, 'native_parameters', lambda m: m.native.parameters()):
            common = []
            for arm in experiment.RUN_NAMES:
                model, _ = initialize(arm, {}, 'cpu', 2)
                common.append(state_hash(experiment.common_parameters(model)))
                optimizer = experiment.make_optimizer(model)
                native, rest = optimizer.param_groups
                self.assertEqual({id(p) for p in native['params']}, {id(p) for p in model.native.parameters()})
                self.assertEqual({id(p) for p in rest['params']}, {id(p) for p in model.head.parameters()})
                self.assertEqual((native['lr'], rest['lr']), (3e-5, 1e-4))
                for successful, epoch, peak in ((0, 0, 1e-4), (20, 2, 1e-4), (30, 3, 5e-5), (40, 4, 2.5e-5)):
                    lr, progress = experiment.learning_rates(optimizer, successful, 10, epoch, peak)
                    self.assertAlmostEqual(native['lr'] / rest['lr'], .3)
                    self.assertEqual(rest['lr'], lr)
                    self.assertGreater(progress, 0)
                    self.assertLessEqual(progress, 1)
            self.assertEqual(common[0], common[1])
        model = TinySelector()
        with patch.object(experiment, 'native_parameters', return_value=[nn.Parameter(torch.ones(1))]):
            with self.assertRaisesRegex(ValueError, 'parameter group'):
                experiment.make_optimizer(model)

    def test_paired_schedule_retains_every_tail(self):
        sizes = {p: 10000 for p in PROBLEMS}
        first = paired_schedule(sizes, 2, 128, 2)
        self.assertEqual(state_hash(first), state_hash(paired_schedule(sizes, 2, 128, 2)))
        for epoch in range(2):
            self.assertEqual(len(first['orders'][epoch]), 79 * 18)
            for problem in PROBLEMS:
                self.assertEqual(first['orders'][epoch].count(problem), 79)
                indices = first['permutations'][epoch][problem]
                torch.testing.assert_close(indices.sort().values, torch.arange(10000))
                self.assertEqual(len(indices[78 * 128:]), 16)

    def test_cached_references_use_only_matching_train_validation_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = {split: {'TSP': dict(costs=np.array([[1., 2.]], dtype=np.float64), winner=np.array([0]),
                                      pool_ids=[3, 1])} for split in ('train', 'val')}
            best = root / 'A_handcrafted_seed2/best_eval.json'
            best.parent.mkdir(parents=True)
            experiment.atomic_json(best, {'epoch': 25})
            for split, path in (('train', root / 'reference_predictions/train/TSP.npz'),
                                ('val', root / 'A_handcrafted_seed2/val_predictions/epoch025/TSP.npz')):
                path.parent.mkdir(parents=True)
                np.savez(path, logits=np.zeros((1, 2)), indices=[0], **{k: v for k, v in raw[split]['TSP'].items()})
            with patch.object(experiment, 'R45', root):
                nominations, hashes = experiment.cached_nominations(raw)
                np.testing.assert_array_equal(nominations['train']['TSP'], [[1, 0]])
                self.assertEqual(len(hashes), 3)
                raw['val']['TSP']['costs'][0, 1] = 2. + 1e-10
                with self.assertRaises(AssertionError):
                    experiment.cached_nominations(raw)

    def test_original_scope_rejects_expansion_and_fp32_targets(self):
        raw = {s: {p: dict(winner=np.empty(n), costs=np.empty((n, 2), dtype=np.float64)) for p in PROBLEMS}
               for s, n in (('train', 10000), ('val', 1000))}
        loaders = {s: {p: Namespace(size=len(r['winner'])) for p, r in entries.items()} for s, entries in raw.items()}
        experiment.check_original_scope(loaders, raw)
        loaders['train']['TSP'].size = 100000
        with self.assertRaises(ValueError):
            experiment.check_original_scope(loaders, raw)
        loaders['train']['TSP'].size = 10000
        raw['train']['TSP']['costs'] = raw['train']['TSP']['costs'].astype(np.float32)
        with self.assertRaises(ValueError):
            experiment.check_original_scope(loaders, raw)

    def test_epoch_atomic_resume_matches_uninterrupted_training(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = fixture_data()
            with patch.object(experiment, 'initialize', initialize), \
                 patch.object(experiment, 'native_parameters', lambda m: m.native.parameters()), \
                 patch.object(experiment, 'verify_sources'), patch.object(experiment, 'verify_data_files'), \
                 patch.object(experiment, 'print_eval'), patch('code.V4.r43_analysis.plot_run'), \
                 redirect_stdout(io.StringIO()):
                experiment.run('A', args(root / 'full'), data)
                real_publish = experiment.publish_best
                def interrupted(path, checkpoint):
                    if checkpoint['epoch'] == 1:
                        raise RuntimeError('fixture crash after epoch commit, before best publication')
                    return real_publish(path, checkpoint)
                with patch.object(experiment, 'publish_best', interrupted):
                    with self.assertRaisesRegex(RuntimeError, 'fixture crash'):
                        experiment.run('A', args(root / 'resume'), data)
                experiment.run('A', args(root / 'resume'), data)
            full = root / 'full/A_random_seed2'
            resumed = root / 'resume/A_random_seed2'
            checkpoints = [torch.load(p / 'last.pt', map_location='cpu', weights_only=False) for p in (full, resumed)]
            for key in ('model', 'optimizer', 'scaler', 'rng', 'controller', 'successful_updates',
                        'updates_by_problem', 'peak_lr', 'best_model'):
                self.assertEqual(state_hash(checkpoints[0][key]), state_hash(checkpoints[1][key]), key)
            result = json.loads((resumed / 'result.json').read_text())
            self.assertEqual(result['epochs_completed'], 5)
            self.assertEqual(result['successful_updates'], 5 * 36)
            self.assertEqual(result['updates_by_problem'], {p: 10 for p in PROBLEMS})
            self.assertGreater(result['wall_seconds'], 0)
            self.assertIsNotNone(result['best']['train'])
            self.assertIsNotNone(result['final']['train'])
            history = json.loads((resumed / 'history.json').read_text())
            self.assertEqual([r['epoch'] for r in history], [1, 2, 3, 4, 5])
            self.assertTrue(all(r['train'] is None for r in history[:4]))
            self.assertIn('macro_actual_regret_pct', history[0]['val']['macro'])
            self.assertEqual(result['best']['epoch'], min(history, key=lambda r: r['val']['macro']['macro_vs_sbs_pct'])['epoch'])
            for problem in PROBLEMS:
                with np.load(resumed / 'val_predictions/best' / f'{problem}.npz') as saved:
                    self.assertEqual(saved['costs'].dtype, np.float64)
                    self.assertTrue({'indices', 'logits', 'pair_margin', 'pred', 'winner', 'costs',
                                     'pool_ids', 'pool', 'nodes'} <= set(saved.files))
            self.assertFalse(list(resumed.glob('*test*')))
            altered = list(data)
            altered[0] = dict(data[0], source_hashes={'changed': 'fixture'})
            (resumed / 'result.json').unlink()
            with patch.object(experiment, 'initialize', initialize), \
                 patch.object(experiment, 'native_parameters', lambda m: m.native.parameters()), \
                 redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, 'configuration changed'):
                experiment.run('A', args(root / 'resume'), tuple(altered))

    def test_validation_lock_is_stable_and_checkpoint_change_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            experiment.atomic_json(root / 'protocol.json', {'model_params': {'A': {}, 'B': {}}})
            for name in experiment.RUN_NAMES.values():
                arm = root / name
                arm.mkdir()
                experiment.atomic_save(arm / 'best.pt', {'model': torch.ones(1)})
                experiment.atomic_json(arm / 'result.json', dict(best=dict(epoch=3, val=dict(macro={'macro_top1': .5})),
                                                                 best_checkpoint_sha256=file_hash(arm / 'best.pt')))
            with patch.object(experiment, 'verify_sources'), patch.object(experiment, 'verify_data_files'):
                experiment.lock_validation_checkpoints(args(root))
                original = (root / 'locked_checkpoints.json').read_bytes()
                experiment.lock_validation_checkpoints(args(root))
                self.assertEqual(original, (root / 'locked_checkpoints.json').read_bytes())
                experiment.atomic_save(root / 'A_random_seed2/best.pt', {'model': torch.zeros(1)})
                with self.assertRaisesRegex(ValueError, 'checkpoint changed'):
                    experiment.lock_validation_checkpoints(args(root))

    def test_atomic_save_failure_preserves_committed_epoch(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'last.pt'
            experiment.atomic_save(path, {'epoch': 0})
            with patch.object(experiment.os, 'replace', side_effect=OSError('fixture failure')):
                with self.assertRaises(OSError):
                    experiment.atomic_save(path, {'epoch': 1})
            self.assertEqual(torch.load(path, weights_only=False)['epoch'], 0)


if __name__ == '__main__':
    unittest.main()

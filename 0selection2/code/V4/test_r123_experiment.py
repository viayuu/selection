"""Bounded CPU cache identity, protocol and locked-TEST checks."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from code.unified_selector.registry import PROBLEMS
from .r123_experiment import (BankPredictionCache, BATCH_SIZE, MAX_UPDATES,
    PREPARATION, json_hash, learning_rate, parser, selection_key, sha256,
    test_permission as require_test_permission, write_json, sha256_array, micro_caps,
    open_bank_cache, SharedBankPredictionCache, backward_logical_batch)
from .r123_model import brier_loss


class ExperimentTests(unittest.TestCase):
    def test_portable_row_keyed_cache_and_identity_collision(self):
        with tempfile.TemporaryDirectory(prefix='learner_test_', dir=PREPARATION) as temporary:
            root = Path(temporary)
            rows = np.asarray([5, 101], dtype=np.int64)
            spec = dict(shape=[2, 3, 2], row_source='explicit_shard_identity')
            arrays = dict(logits=np.arange(12, dtype=np.float32).reshape(2, 3, 2), row_indices=rows,
                source_indices=np.asarray([123, 456], dtype=np.int32),
                source_identity=np.asarray(['TSP:shaA:0', 'TSP:shaB:4'], dtype='S100'))
            spec.update(source_indices_sha256=sha256_array(arrays['source_indices']),
                        source_identity_sha256=sha256_array(arrays['source_identity']))
            for name, array in arrays.items():
                np.save(root / (name + '.npy'), array, allow_pickle=False)
            metadata = dict(spec=spec, complete=True, written=2,
                artifacts={name + '.npy': sha256(root / (name + '.npy')) for name in arrays})
            write_json(root / 'metadata.json', metadata)
            cache = BankPredictionCache(root, spec, rows)
            torch.testing.assert_close(cache.take([1]), torch.from_numpy(arrays['logits'][[1]]))
            cache.close()
            with self.assertRaises(ValueError):
                BankPredictionCache(root, spec, rows[::-1])
            metadata['complete'] = False
            write_json(root / 'metadata.json', metadata)
            with self.assertRaises(ValueError):
                BankPredictionCache(root, spec, rows)
            cache = BankPredictionCache(root, spec, rows, allow_incomplete=True)
            cache.close()

    def test_full18_checkpoint_order_is_regret_then_top1_then_earliest(self):
        def value(risk, top1):
            return dict(per_problem={p: {} for p in PROBLEMS},
                macro=dict(macro_actual_regret_pct=risk, macro_top1=top1))
        self.assertLess(selection_key(value(.2, .5), 720), selection_key(value(.3, .9), 720))
        self.assertLess(selection_key(value(.2, .8), 720), selection_key(value(.2, .7), 720))
        self.assertLess(selection_key(value(.2, .8), 720), selection_key(value(.2, .8), 1440))
        with self.assertRaises(ValueError):
            selection_key(dict(per_problem={'TSP': {}}, macro=value(.2, .8)['macro']), 720)

    def test_test_access_fails_before_any_checkpoint_or_test_data_read(self):
        with tempfile.TemporaryDirectory(prefix='learner_test_', dir=PREPARATION) as temporary:
            with patch('torch.load') as load:
                with self.assertRaises(PermissionError):
                    require_test_permission(temporary)
                with self.assertRaises(PermissionError):
                    require_test_permission(None)
                load.assert_not_called()

    def test_lr_and_frozen_cli_defaults(self):
        self.assertAlmostEqual(learning_rate(540), 1e-4)
        self.assertAlmostEqual(learning_rate(MAX_UPDATES), 1e-6)
        self.assertLess(learning_rate(1), learning_rate(540))
        self.assertEqual(BATCH_SIZE, 1024)
        self.assertEqual(parser().parse_args(['describe']).device, 'cpu')
        self.assertIsNone(parser().parse_args(['train']).microbatch)
        self.assertEqual(parser().parse_args(['train']).predict_batch, 128)
        self.assertEqual((micro_caps('3090')['TSP'], micro_caps('3090')['VRPTW']), (128, 512))
        self.assertEqual((micro_caps('4090')['CVRP'], micro_caps('4090')['ATSP']), (256, 1024))

    def test_shared_prediction_consumer_reuses_existing_archives(self):
        data = dict(logits=np.ones((2, 3, 4), dtype=np.float32),
            row_indices=np.asarray([17, 88], dtype=np.int64), source_indices=np.asarray([123, 456]),
            source_identity=np.asarray(['TSP:sha:0', 'TSP:sha:1']), metadata=[])
        context = dict(_shared_prediction_context={})
        with patch('code.V4.r123_prediction_cache.load_scores', return_value=data) as load:
            with tempfile.TemporaryDirectory(prefix='learner_test_', dir=PREPARATION) as temporary:
                cache = open_bank_cache(temporary, 'train', 'TSP', context)
                self.assertIsInstance(cache, SharedBankPredictionCache)
                np.testing.assert_array_equal(cache.row_indices, [17, 88])
                self.assertEqual(tuple(cache.take([1]).shape), (1, 3, 4))
                load.assert_called_once()
                cache.close()

    def test_true_sample_weighting_matches_full_logical_batch_gradient(self):
        model = torch.nn.Linear(4, 3)
        features = torch.randn(7, 4)
        labels = torch.randint(2, (7, 3)).float()
        brier_loss(model(features), labels).backward()
        full = {name: p.grad.clone() for name, p in model.named_parameters()}
        model.zero_grad()
        for start in range(0, 7, 3):
            stop = min(start + 3, 7)
            (brier_loss(model(features[start:stop]), labels[start:stop]) * ((stop - start) / 7)).backward()
        for name, parameter in model.named_parameters():
            torch.testing.assert_close(parameter.grad, full[name])

    def test_runner_backward_uses_source_rows_without_labels_in_gate(self):
        rows = np.asarray([8, 3, 22, 5, 30, 17, 2], dtype=np.int64)
        features = torch.randn(31, 4)

        class Source:
            def batch(self, selected, device='cpu'):
                self_rows = torch.from_numpy(np.array(selected))
                return dict(node=features[self_rows, None], ind=self_rows % 2,
                    pool_ids=torch.tensor([5, 1]), costs=torch.full((len(selected), 2), -999.))

        class Gate(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(4, 3)

            def forward(self, clean):
                if set(clean) & {'costs', 'ind', 'expert_logits'}:
                    raise AssertionError('Supervision reached gate forward')
                return dict(reliability_logits=self.linear(clean['node'][:, 0]))

        bank = SharedBankPredictionCache(dict(logits=np.random.default_rng(2).normal(size=(7, 3, 2)).astype(np.float32),
            row_indices=rows, source_indices=rows.copy(), source_identity=rows.astype('S100'), metadata=[]))
        gate = Gate()
        optimizer = torch.optim.SGD(gate.parameters(), lr=.1)
        loss = backward_logical_batch(gate, optimizer, Source(), bank, np.arange(7), 'cpu', 7)
        gradients = {name: parameter.grad.clone() for name, parameter in gate.named_parameters()}
        split_loss = backward_logical_batch(gate, optimizer, Source(), bank, np.arange(7), 'cpu', 3)
        self.assertAlmostEqual(loss, split_loss, places=6)
        for name, parameter in gate.named_parameters():
            torch.testing.assert_close(parameter.grad, gradients[name])


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()

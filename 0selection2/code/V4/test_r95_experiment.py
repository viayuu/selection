"""Focused CPU integration witnesses for immutable TRAIN-only R95 proposals."""

import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from . import r95_experiment as runner
from .pairwise_objective import comparison_targets
from .pairwise_selector import maximin_logits
from .r76_experiment import save_checkpoint
from .r77_data import parameter_hash
from .test_r93_experiment import fresh_fixture


def configured(directory, arm='A'):
    initial, geometry = fresh_fixture()
    model, params, receipt = runner.initialize_shared(initial, geometry, arm)
    args = runner.parser().parse_args(['describe', '--arm', arm, '--root', str(directory),
        '--prepared-root', str(directory / 'prepared'), '--data-root', str(directory / 'data'),
        '--cache-root', str(directory / 'cache'), '--device', 'cpu'])
    lock = copy.deepcopy(runner.read_json(runner.DEFAULT_ROOT / 'protocol_lock.json'))
    lock.update(data_root=str(args.data_root), prepared_root=str(args.prepared_root))
    runner.write_json(args.root / 'protocol_lock.json', lock)
    runner.write_json(args.prepared_root / 'prepared.json', dict(fixture=True))
    from code.unified_selector.registry import POOLS, PROBLEMS, S2I
    pools = {p: dict(pool=POOLS[p], pool_ids=[S2I[s] for s in POOLS[p]]) for p in PROBLEMS}
    config = runner.protocol(args, dict(trainable_sha256=initial['trainable_sha256']),
        dict(splits=dict(train=pools)), geometry, {}, params,
        runner.task_order(), runner.micro_caps('3090', 352))
    return args, config, model, receipt, initial


class R95ExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_fresh_original_parameters_and_checkpoint_roundtrip(self):
        with tempfile.TemporaryDirectory() as temporary:
            hashes = []
            for arm in runner.ARMS:
                args, config, model, receipt, initial = configured(Path(temporary), arm)
                runner.validate_checkpoint_config(config)
                self.assertEqual(parameter_hash(dict(model.named_parameters())), initial['trainable_sha256'])
                self.assertEqual(receipt['new_parameter_count'], 0)
                self.assertEqual(config['rdrop_weight'], .35)
                self.assertNotIn('focal_extra', config['loss'])
                hashes.append(parameter_hash(dict(model.named_parameters())))
                path = args.root / (arm + '.pt')
                state = dict(model=model.state_dict(), args=config,
                    state_hash=runner.json_hash(config), successful_updates=1440)
                save_checkpoint(path, state)
                loaded, _ = runner.load_checkpoint(path)
                self.assertEqual(parameter_hash(loaded.state_dict()), parameter_hash(model.state_dict()))
            self.assertEqual(hashes[0], hashes[1])

    def test_bad_sampling_objective_and_protocol_hash_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            args, config, model, _, _ = configured(Path(temporary))
            for change in ('sampling', 'loss', 'hash'):
                changed = copy.deepcopy(config)
                if change == 'sampling':
                    changed['sampling']['proposal_uniform_mixture'] = .1
                elif change == 'loss':
                    changed['objective']['coefficients']['comparison'] = 0.
                path = args.root / (change + '.pt')
                save_checkpoint(path, dict(args=changed, model=model.state_dict(),
                    state_hash='wrong' if change == 'hash' else runner.json_hash(changed)))
                with self.subTest(change=change), self.assertRaises(ValueError):
                    runner.load_checkpoint(path)

    def test_private_sampler_engine_does_not_mutate_shared_runner(self):
        original_stream, original_cache = runner.engine.ShuffleStream, runner.engine.open_cache
        with tempfile.TemporaryDirectory() as temporary:
            args, _, _, _, _ = configured(Path(temporary), 'B')
            private = runner.training_engine(args)
            self.assertIsNot(private, runner.engine)
            self.assertIs(runner.engine.ShuffleStream, original_stream)
            self.assertIs(runner.engine.open_cache, original_cache)
            stream = private.ShuffleStream(81000, runner.task_seed(2, 'TSP'))
            rows = stream.take(1024)
            reference = original_stream(81000, runner.task_seed(2, 'TSP')).take(1024)
            torch.testing.assert_close(rows.indices, reference, rtol=0, atol=0)
            torch.testing.assert_close(rows.weights, torch.ones(1024), rtol=0, atol=0)

    def test_proposals_reject_other_arm_split_and_changed_bytes(self):
        with tempfile.TemporaryDirectory() as temporary, patch.object(runner, 'PROBLEMS', ('TSP',)), \
                patch.object(runner, 'SPLIT_ROWS', dict(train=3, val=2, test=2)):
            args = SimpleNamespace(root=Path(temporary), arm='B')
            directory = args.root / 'B/evaluation/train/update0007200'
            identity = dict(arm='B', state_hash='state', model_state_sha256='model', successful_updates=7200)
            runner.write_json(directory / 'identity.json', identity)
            runner.save_npz(directory / 'rank000.npz', {'TSP__sampling_gradient_norm': np.array([0., 1., 2.])})
            arrays = dict(TSP__sampling_gradient_norm=np.array([0., 1., 2.]))
            runner.write_proposal(args, directory, identity, arrays)
            runner.write_proposal(args, directory, identity, arrays)
            q = runner.proposal_provider(args, 'state')('TSP', 7200)
            np.testing.assert_allclose(q, np.array([1/6, 1/3, 1/2]), rtol=1e-12)
            path, record_path = runner.proposal_paths(args, 7200)
            record = runner.read_json(record_path)
            for changes in (dict(arm='A'), dict(split='val'), dict(state_hash='other')):
                runner.write_json(record_path, dict(record, **changes))
                with self.subTest(changes=changes), self.assertRaises(ValueError):
                    runner.proposal_provider(args, 'state')('TSP', 7200)
            runner.write_json(record_path, record)
            with path.open('ab') as stream:
                stream.write(b'changed')
            with self.assertRaises(ValueError):
                runner.proposal_provider(args, 'state')('TSP', 7200)

    def test_scheduled_train_proxy_uses_same_forward_and_restores_training(self):
        costs = np.array([[1., 1.2], [1.3, 1.], [1., 1.]], dtype=np.float64)
        features = torch.arange(18, dtype=torch.float32).reshape(3, 2, 3) / 10

        class Tiny(nn.Module):
            def __init__(self):
                super().__init__()
                self.score_head = nn.Sequential(nn.Linear(3, 1))
                self.calls = 0

            def forward(self, batch):
                if any(k in batch for k in ('ind', 'costs', 'cost_regret', 'sampling_weight', 'sampling_row')):
                    raise AssertionError('Supervision entered model forward')
                self.calls += 1
                utility = self.score_head(batch['features']).squeeze(-1)
                margins = utility[:, :, None] - utility[:, None, :]
                mask = torch.ones_like(utility, dtype=torch.bool)
                return dict(utility=utility, pair_margin=margins, solver_mask=mask,
                            logits=maximin_logits(margins, mask))

        class Cache:
            size = 3
            pool, pool_ids = ['a', 'b'], [0, 1]
            lengths = np.array([4, 5, 6])
            raw = dict(costs=costs, winner=costs.argmin(1))
            closed = False

            def batch(self, indices, device='cpu'):
                indices = indices.cpu().numpy()
                return dict(features=features[indices], pool_ids=torch.tensor([0, 1]),
                    ind=torch.from_numpy(costs.argmin(1)[indices]),
                    costs=torch.from_numpy(costs[indices].astype(np.float32)),
                    **comparison_targets(costs[indices]))

            def close(self):
                self.closed = True

        cache, model = Cache(), Tiny().train()
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            source = base / 'data/TSPtrain'
            source.mkdir(parents=True)
            np.savez(source / 'input_metadata.npz', source_index=np.array([5, 6, 7]))
            args = SimpleNamespace(root=base, arm='B', data_root=base / 'data', cache_root=base / 'cache',
                eval_batch_size=2, large_micro_cap=352, gpu_type='3090', device='cpu')
            with patch.object(runner, 'PROBLEMS', ('TSP',)), \
                    patch.object(runner, 'SPLIT_ROWS', dict(train=3, val=2, test=2)), \
                    patch.object(runner, 'status_update'), \
                    patch.object(runner, 'open_cache', return_value=cache), \
                    patch.object(runner.engine, 'fixed_sbs_metrics', return_value=(dict(n=3), np.zeros(3, dtype=int))), \
                    patch.object(runner.engine, 'summarize', side_effect=lambda _: dict(macro={}, native_total=3)):
                result = runner.evaluate_sampling_train(model, args, {}, dict(TSP={}),
                    'update0007200', 7200, 'fixed')
                self.assertEqual(model.calls, 2)
                self.assertTrue(model.training)
                self.assertTrue(cache.closed)
                self.assertEqual(len(model.score_head[-1]._forward_pre_hooks), 0)
                path, record_path = runner.proposal_paths(args, 7200)
                self.assertTrue(path.is_file())
                self.assertEqual(runner.read_json(record_path)['split'], 'train')
                self.assertEqual(result['native_total'], 3)
                runner.evaluate_sampling_train(model, args, {}, dict(TSP={}), 'update0007200', 7200, 'fixed')
                self.assertEqual(model.calls, 2)


if __name__ == '__main__':
    unittest.main()

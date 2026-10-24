"""Small CPU contracts for the R107 runner; no data generation or GPU work."""

import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from . import r107_experiment as runner
from .r76_experiment import Runtime, save_checkpoint, write_json
from .r77_data import parameter_hash
from .V4Model import get_default_model_params


def full18_params():
    params = get_default_model_params()
    params.update(architecture='pairwise', joint_layer_num=2, local_geometry=True,
        geometry_mode='real', ignore_coord_dist=True, native_winner=True,
        decision_head='classification', pair_mode='score_difference',
        solver_representation='handcrafted', solver_feature_weight=1.0, sdpa=True)
    return params


class TinyObjective(torch.nn.Module):
    def __init__(self, fail=False):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(()))
        self.fail = fail

    def forward(self, batch, weight):
        loss = (self.weight * batch['x']).square().mean()
        if self.fail:
            self.fail = False
            loss = loss * float('nan')
        return loss, loss.detach().expand(8)


class R107RunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_cli_schedule_and_private_engine(self):
        for arm in runner.ARMS:
            for command in ('train', 'test', 'timing', 'describe', 'status'):
                args = runner.parser().parse_args([command, '--arm', arm])
                self.assertEqual((args.large_micro_cap, args.eval_batch_size), (256, 128))
                self.assertEqual(args.cache_root, Path('/tmp/shiys_R84_new90k_cache'))
                self.assertEqual(args.wandb_mode, 'offline')
        local = runner.private_engine()
        self.assertIs(local.train.__globals__['micro_caps'], runner.micro_caps)
        self.assertIs(local.evaluate.__wrapped__.__globals__['micro_caps'], runner.micro_caps)
        self.assertEqual(runner.micro_caps()['ATSP'], 1024)
        self.assertEqual(set(runner.task_order(57600)), set(runner.PROBLEMS))
        self.assertEqual(runner.learning_rate(4320), 1e-4)
        self.assertEqual(runner.learning_rate(57600), 1e-6)

    def test_common_fresh_initialization(self):
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(2)
            source = runner.engine.make_model(full18_params(), 'A_baseline')
        weights = {k: v.detach().clone() for k, v in source.named_parameters()}
        initial = dict(seed=2, initialization='scratch', model_params=source.params,
            trainable=weights, trainable_sha256=parameter_hash(weights))
        geometry = dict(mean=[.2] * 12, std=[1.3] * 12, fitted_on='synthetic TRAIN only')
        hashes = []
        for arm in runner.ARMS:
            model, _, receipt = runner.initialize_shared(initial, geometry, arm)
            self.assertEqual(sum(p.numel() for p in model.parameters()), runner.PARAMETER_COUNTS[arm])
            self.assertTrue(receipt['cross_arm_complete_state_verified'])
            hashes.append(receipt['common_fresh_model_state_sha256'])
        self.assertEqual(*hashes)

    def test_protocol_factory_round_trip(self):
        for arm in runner.ARMS:
            with self.subTest(arm=arm), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                args = runner.parser().parse_args(['train', '--arm', arm, '--root', str(root)])
                args.prepared_root = root / 'prepared'
                write_json(args.prepared_root / 'prepared.json', {})
                lock = dict(runner.required_lock_fields(), data_root=str(args.data_root),
                    prepared_root=str(args.prepared_root), primary_cache=str(args.cache_root))
                write_json(root / 'protocol_lock.json', lock)
                model = runner.make_model(full18_params(), arm)
                manifest = dict(splits=dict(train={p: dict(pool=['synthetic'], pool_ids=[0])
                                                   for p in runner.PROBLEMS}))
                config = runner.protocol(args, dict(trainable_sha256=runner.ORIGINAL_FRESH_SHA256),
                    manifest, {}, {}, model.params, runner.task_order(57600), runner.micro_caps())
                path = root / 'synthetic_checkpoint.pt'
                save_checkpoint(path, dict(model=model.state_dict(), args=config,
                    state_hash=runner.json_hash(config), successful_updates=1440, epoch=1))
                restored, state = runner.load_checkpoint(path, 'cpu')
                self.assertEqual(parameter_hash(model.state_dict()), parameter_hash(restored.state_dict()))
                self.assertEqual(state['args'], config)
                for field, value in (('auxiliary_loss', True), ('source_vectors', True),
                        ('path_spec', {}), ('model_factory', 'code.V4.r105_experiment.load_checkpoint')):
                    broken = copy.deepcopy(config)
                    broken[field] = value
                    with self.assertRaises(ValueError, msg=field):
                        runner.validate_checkpoint_config(broken)

    def test_row_weighted_microbatches_and_single_retry(self):
        batch = dict(ind=torch.ones(1024, dtype=torch.long), x=torch.linspace(.1, 1., 1024))
        outcomes = []
        for fail in (False, True):
            objective = TinyObjective(fail)
            optimizer = torch.optim.SGD(objective.parameters(), lr=.01)
            with patch.object(optimizer, 'step', wraps=optimizer.step) as step:
                parts, retries, _ = runner.update_batch(objective, optimizer,
                    torch.amp.GradScaler('cpu', enabled=False), batch, .35,
                    Runtime(0, 1, torch.device('cpu')), 256, problem='TSP')
            self.assertEqual(step.call_count, 1)
            self.assertEqual(retries, int(fail))
            self.assertTrue(torch.isfinite(parts).all())
            outcomes.append(objective.weight.detach().clone())
        torch.testing.assert_close(*outcomes, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()

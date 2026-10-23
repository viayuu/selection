"""Bounded CPU runner contracts; synthetic tensors only, no jobs or real data."""

import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from code.unified_selector.registry import P2I, constraint_bits, problem_descriptor
from . import r105_experiment as runner
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


def protocol_fixture(root, arm):
    args = runner.parser().parse_args(['train', '--arm', arm, '--root', str(root)])
    args.prepared_root = Path(root) / 'prepared'
    write_json(args.prepared_root / 'prepared.json', {})
    lock = dict(runner.required_lock_fields(), data_root=str(args.data_root),
        prepared_root=str(args.prepared_root), primary_cache=str(args.cache_root),
        descriptive_main_field='retained without changing scientific fields')
    write_json(args.root / 'protocol_lock.json', lock)
    model = runner.make_model(full18_params(), arm)
    manifest = dict(splits=dict(train={p: dict(pool=['synthetic'], pool_ids=[0]) for p in runner.PROBLEMS}))
    config = runner.protocol(args, dict(trainable_sha256=runner.ORIGINAL_FRESH_SHA256),
        manifest, {}, {}, model.params, runner.task_order(57600), runner.micro_caps())
    return args, model, config


def capture_fixture(run, split='val'):
    run.mkdir(parents=True, exist_ok=True)
    config = dict(experiment='R105', arm=runner.ARMS[0])
    state_hash = runner.json_hash(config)
    weights = dict(weight=torch.tensor([.2]), buffer=torch.tensor([1.3]))
    state = dict(args=config, model=weights, state_hash=state_hash, successful_updates=1440, epoch=1)
    save_checkpoint(run / 'best.pt', state)
    lock = dict(experiment='R105', arm=runner.ARMS[0], checkpoint_sha256=runner.file_hash(run / 'best.pt'),
        state_hash=state_hash, selected_update=1440, epoch=1)
    write_json(run / 'final_test_lock.json', lock)
    identity = dict(experiment='R105', arm=runner.ARMS[0], split=split, checkpoint_sha256=None,
        state_hash=state_hash, model_state_sha256=parameter_hash(weights), successful_updates=1440)
    archive = run / 'evaluation' / split / 'update0001440/rank000.npz'
    write_json(archive.parent / 'identity.json', identity)
    summary = dict(identity=identity, prediction_archive=str(archive))
    final = dict(state_hash=state_hash, selected_update=1440, checkpoint_sha256=lock['checkpoint_sha256'],
        val=summary, epoch=1)
    return state, lock, summary, final


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


class R105RunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_cli_actual_main_lock_schedule_and_private_caps(self):
        for arm in runner.ARMS:
            for command in ('train', 'test', 'timing', 'describe', 'status'):
                args = runner.parser().parse_args([command, '--arm', arm])
                self.assertEqual((args.large_micro_cap, args.eval_batch_size), (128, 128))
                self.assertEqual(args.root, runner.DEFAULT_ROOT)
                self.assertEqual(args.wandb_mode, 'offline')
                self.assertEqual(args.cache_root, Path('/tmp/shiys_R84_new90k_cache'))
        args = runner.parser().parse_args(['train', '--arm', runner.ARMS[0]])
        record = runner.protocol_lock(args)['record']
        self.assertTrue(record['locked'])
        for key, value in runner.required_lock_fields().items():
            self.assertEqual(record[key], value, key)
        original = runner.engine.micro_caps('3090', 128)
        local = runner.private_engine()
        self.assertIs(local.train.__globals__['micro_caps'], runner.micro_caps)
        self.assertIs(local.evaluate.__wrapped__.__globals__['micro_caps'], runner.micro_caps)
        self.assertIs(local.test.__globals__['write_json'], runner._engine_write_json)
        self.assertEqual(runner.engine.micro_caps('3090', 128), original)
        self.assertEqual((original['ATSP'], runner.micro_caps()['ATSP']), (1024, 512))
        self.assertEqual(set(runner.task_order(57600)), set(runner.PROBLEMS))
        self.assertEqual((runner.learning_rate(4320), runner.learning_rate(57600)), (1e-4, 1e-6))
        self.assertEqual(runner.consistency_weight(4320), .35)
        with self.assertRaises(ValueError):
            runner.micro_caps('3090', 64)

    def test_strict_protocol_and_per_arm_trained_factory_round_trip(self):
        for arm in runner.ARMS:
            with self.subTest(arm=arm), tempfile.TemporaryDirectory() as temporary:
                args, model, config = protocol_fixture(Path(temporary), arm)
                self.assertEqual(config['trainable_parameter_count'], 3425561)
                self.assertEqual(config['loss'], runner.engine.LOSS_RECIPE)
                self.assertTrue(config['identical_complete_fresh_parameters'])
                self.assertFalse(config['active_parameter_matched'])
                with torch.no_grad():
                    model.instance_encoder.layers[0].ffn.router.bias[0].add_(.2)
                    model.instance_encoder.geometry_residual.mean.add_(.1)
                path = Path(temporary) / 'trained.pt'
                state = dict(model=model.state_dict(), args=config, state_hash=runner.json_hash(config),
                    successful_updates=1440, epoch=1)
                save_checkpoint(path, state)
                restored, checkpoint = runner.load_checkpoint(path, 'cpu')
                self.assertEqual(restored.expert_arm, arm)
                self.assertEqual(parameter_hash(restored.state_dict()), parameter_hash(model.state_dict()))
                self.assertEqual(checkpoint['args'], config)
                self.assertFalse(restored.initialization_receipt['fresh_initialization'])
                for field, changed in (('model_factory', 'code.V4.r104_experiment.load_checkpoint'),
                        ('arm', 'A_width256'), ('solver_quotas', True), ('auxiliary_loss', True),
                        ('gate', dict(runner.GATE, A_inactive_feature_weight_parameters=0))):
                    broken = dict(config, **{field: changed})
                    with self.assertRaises(ValueError, msg=field):
                        runner.validate_checkpoint_config(broken)
                broken = copy.deepcopy(config)
                broken['model_params']['expert_arm'] = runner.ARMS[1 - runner.ARMS.index(arm)]
                with self.assertRaises(ValueError):
                    runner.validate_checkpoint_config(broken)
                state['state_hash'] = 'wrong'
                save_checkpoint(path, state)
                with self.assertRaises(ValueError):
                    runner.load_checkpoint(path)
                state['state_hash'] = runner.json_hash(config)
                state['objective_state'] = {}
                save_checkpoint(path, state)
                with self.assertRaises(ValueError):
                    runner.load_checkpoint(path)
                lock = runner.protocol_lock(args)['record']
                write_json(args.root / 'protocol_lock.json', dict(lock, experts_per_layer=3))
                with self.assertRaises(ValueError):
                    runner.protocol_lock(args)

    def test_fresh_complete_state_equality_verified_without_data(self):
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(2)
            source = runner.engine.make_model(full18_params(), 'A_baseline')
        weights = {k: v.detach().clone() for k, v in source.named_parameters()}
        initial = dict(seed=2, initialization='scratch', model_params=source.params,
            trainable=weights, trainable_sha256=parameter_hash(weights))
        geometry = dict(mean=[.2] * 12, std=[1.3] * 12, fitted_on='synthetic TRAIN only')
        receipts = []
        for arm in runner.ARMS:
            model, _, receipt = runner.initialize_shared(initial, geometry, arm)
            receipts.append(receipt)
            self.assertTrue(receipt['cross_arm_complete_state_verified'])
            self.assertEqual(sum(p.numel() for p in model.parameters()), 3425561)
        self.assertEqual(receipts[0]['common_fresh_model_state_sha256'], receipts[1]['common_fresh_model_state_sha256'])
        self.assertEqual([r['inactive_router_feature_parameters'] for r in receipts], [2048, 0])

    def test_original_row_weighted_retry_accepts_one_successful_update(self):
        batch = dict(ind=torch.ones(1024, dtype=torch.long), x=torch.linspace(.1, 1., 1024))
        outcomes = []
        for fail in (False, True):
            objective = TinyObjective(fail)
            optimizer = torch.optim.SGD(objective.parameters(), lr=.01)
            with patch.object(optimizer, 'step', wraps=optimizer.step) as step:
                parts, retries, _ = runner.update_batch(objective, optimizer,
                    torch.amp.GradScaler('cpu', enabled=False), batch, .35,
                    Runtime(0, 1, torch.device('cpu')), 128, problem='TSP')
            self.assertEqual(step.call_count, 1)
            self.assertEqual(retries, int(fail))
            self.assertTrue(torch.isfinite(parts).all())
            outcomes.append(objective.weight.detach().clone())
        torch.testing.assert_close(*outcomes, rtol=0, atol=0)
        with self.assertRaises(ValueError):
            runner.update_batch(objective, optimizer, None, batch, .35, None, 512, problem='TSP')
        self.assertEqual(runner.checkpoint_extras(runner.make_objective(torch.nn.Linear(1, 1), runner.ARMS[0])), {})

    def test_selected_val_binding_preserves_archive_and_rejects_wrong_state(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            state, lock, val, final = capture_fixture(run)
            path = Path(val['prediction_archive']).parent / 'identity.json'
            captured_bytes = path.read_bytes()
            result = runner.bind_selected_val(run, final)
            self.assertEqual(result['val']['identity']['checkpoint_sha256'], lock['checkpoint_sha256'])
            self.assertIsNone(result['val']['archive_capture_identity']['checkpoint_sha256'])
            self.assertEqual(path.read_bytes(), captured_bytes)
            self.assertEqual(runner.bind_selected_val(run, result), result)
            for field, value in (('model_state_sha256', 'different'), ('state_hash', 'different'),
                                 ('successful_updates', 2880), ('arm', runner.ARMS[1])):
                poisoned = copy.deepcopy(final)
                poisoned['val']['identity'][field] = value
                with self.assertRaises(ValueError, msg=field):
                    runner.bind_selected_val(run, poisoned)
            state['model']['buffer'].add_(1)
            save_checkpoint(run / 'best.pt', state)
            with self.assertRaises(ValueError):
                runner.bind_selected_val(run, final)

    def test_input_only_cache_never_opens_labels_and_masks_forward(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            record = dict(problem='TSP', split='val', rows=4500, data_root=str(root / 'data'),
                kind='coord', problem_id=0, pool=['x'], pool_ids=[0], cbits=[0] * 6,
                problem_desc=[0] * 8, mask=[1], d_in=2)
            directory = root / 'val/TSP'
            write_json(directory / 'metadata.json', record)
            config = dict(cache_root=str(root), data_root=record['data_root'],
                candidate_pools=dict(TSP=['x']), candidate_pool_ids=dict(TSP=[0]))
            manifest = dict(splits=dict(val=dict(TSP=record)))
            arrays = dict(n=np.full(128, 3, dtype=np.int64), coord_dist=np.zeros(128, dtype=np.int64),
                node=np.zeros((128, 3, 2), dtype=np.float32), node_geom=np.zeros((128, 3, 12), dtype=np.float32))
            specs = {k: {} for k in (*arrays, 'costs', 'winner')}
            with patch('code.V4.r76_cache._validate_record', return_value=specs), \
                    patch('code.V4.r75_cache._open_arrays', return_value=arrays) as opened, \
                    patch('code.V4.r75_cache._close'):
                batch = runner.cached_input_batch(config, manifest, 'val', 'TSP', torch.device('cpu'))
            self.assertEqual(set(opened.call_args.args[1]), set(arrays))
            self.assertEqual(batch['node'].shape, (128, 3, 2))
            self.assertFalse({'costs', 'ind', 'winner', 'cmp_target'} & batch.keys())
            self.assertEqual(runner.raw_inputs(dict(batch, costs='poison', ind='poison')), batch)

    def test_routing_statistics_distinguish_between_size_and_within_exact_size(self):
        n = torch.tensor([3] * 64 + [5] * 64)
        batch = dict(n=n, node_mask=torch.arange(5)[None] < n[:, None])
        weights = torch.full((128, 4, 4), .25)
        weights[64:] = torch.tensor([.4, .3, .2, .1])
        row = runner.routing_statistics(weights, batch, 'TSP')
        self.assertEqual(row['exact_n_counts'], {'3': 64, '5': 64})
        self.assertGreater(max(row['layers'][0]['instance_weight_std']), 0)
        self.assertTrue(all(max(r['instance_weight_range']) == 0 for r in row['within_exact_n']))
        self.assertEqual(len(row['within_exact_n']), 8)
        self.assertAlmostEqual(row['within_exact_n'][0]['mean_graph_mixture_entropy_nats'], np.log(4))

    def test_real_capture_api_input_only_first128_and_cpu_timing_adapter(self):
        n = torch.tensor([3] * 64 + [5] * 64)
        batch = dict(problem_id=P2I['TSP'], kind='coord', n=n,
            node_mask=torch.arange(5)[None] < n[:, None],
            node=torch.linspace(.01, .97, 10).reshape(1, 5, 2).repeat(128, 1, 1),
            node_geom=torch.zeros(128, 5, 12), coord_dist=torch.full((128,), 4),
            cbits=torch.tensor([constraint_bits('TSP')] * 128, dtype=torch.float32),
            problem_desc=torch.tensor([problem_descriptor('TSP')] * 128),
            pool_ids=torch.tensor([0, 1]))
        for arm in runner.ARMS:
            model = runner.make_model(full18_params(), arm).eval()
            state_hash = parameter_hash(model.state_dict())
            routes = runner._routing_weights(model, runner.raw_inputs(batch))
            self.assertEqual(routes.shape, (128, 4, 4))
            self.assertEqual(routes.dtype, torch.float32)
            self.assertFalse(routes.requires_grad)
            row = runner.routing_statistics(routes, batch, 'TSP')
            self.assertEqual(row['exact_n_counts'], {'3': 64, '5': 64})
            np.testing.assert_allclose(row['layers'][0]['mean_expert_weights'], [.25] * 4, atol=1e-7)
            # Exercise the adapter on CPU, never label this a formal GPU receipt.
            samples = runner.measure_inputs(model, batch, torch.device('cpu'))
            self.assertEqual(len(samples['batch_seconds']), 3)
            self.assertGreater(samples['mean_seconds_per_instance'], 0.)
            self.assertEqual(parameter_hash(model.state_dict()), state_hash)


if __name__ == '__main__':
    unittest.main()

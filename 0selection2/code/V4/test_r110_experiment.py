"""CPU runner contracts, with no real cache arrays, GPU jobs or launches."""

import copy
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from . import r76_experiment as original
from . import r108_objective as historical_objective
from . import r110_experiment as runner
from .pairwise_objective import LABEL_KEYS
from .pairwise_selector import maximin_logits
from .r76_experiment import Runtime, save_checkpoint, write_json
from .r77_data import parameter_hash
from .V4Model import get_default_model_params


def full18_params(arm):
    params = get_default_model_params()
    params.update(architecture='pairwise', joint_layer_num=2, local_geometry=True,
        geometry_mode='real', ignore_coord_dist=True, native_winner=True, dropout=.1,
        decision_head='classification', pair_mode='score_difference',
        solver_representation='handcrafted', solver_feature_weight=1., sdpa=True,
        r109_arm=runner.MODEL_ARM_MAP[arm], r109_spec=copy.deepcopy(runner.models.SPEC),
        role_arm=runner.MODEL_ROLE_ARM)
    return params


def config_fixture(root, arm):
    args = runner.parser().parse_args(['train', '--arm', arm, '--root', str(root)])
    args.prepared_root, args.data_root, args.cache_root = root / 'prepared', root / 'data', root / 'cache'
    write_json(args.prepared_root / 'prepared.json', {})
    record = dict(runner.required_lock_fields(), data_root=str(args.data_root),
        prepared_root=str(args.prepared_root), primary_cache=str(args.cache_root))
    write_json(root / 'protocol_lock.json', record)
    manifest = dict(splits=dict(train={p: dict(pool=['synthetic'], pool_ids=[0]) for p in runner.PROBLEMS}))
    config = runner.protocol(args, dict(trainable_sha256=runner.ORIGINAL_FRESH_SHA256),
        manifest, {}, {}, full18_params(arm), runner.task_order(runner.MAX_UPDATES), runner.micro_caps())
    return args, config


class TinySelector(torch.nn.Module):
    def __init__(self, arm):
        super().__init__()
        self.r109_arm = runner.MODEL_ARM_MAP[arm]
        self.weight = torch.nn.Parameter(torch.tensor([.1, -.2, .3]))
        self.calls = 0

    def forward(self, batch):
        if set(batch).intersection(LABEL_KEYS):
            raise AssertionError('Supervision reached selector inputs')
        self.calls += 1
        utility = batch['node'][:, 0, :1] * self.weight[None]
        mask = torch.ones_like(utility, dtype=torch.bool)
        margins = utility[:, :, None] - utility[:, None, :]
        return dict(utility=utility, solver_mask=mask, logits=maximin_logits(margins, mask))


def capture_fixture(root):
    config = dict(experiment='R110', arm=runner.ARMS[0], synthetic=True)
    state = dict(args=config, model={'weight': torch.tensor([.4])},
                 state_hash=runner.json_hash(config), successful_updates=1440)
    save_checkpoint(root / 'best.pt', state)
    lock = dict(arm=config['arm'], checkpoint_sha256=runner.file_hash(root / 'best.pt'),
                state_hash=state['state_hash'], selected_update=1440)
    write_json(root / 'final_test_lock.json', lock)
    summaries = {}
    for split in ('train', 'val'):
        directory = root / 'evaluation' / split / 'update0001440'
        identity = dict(experiment='R110', arm=config['arm'], split=split, successful_updates=1440,
            model_state_sha256=parameter_hash(state['model']), state_hash=state['state_hash'], checkpoint_sha256=None)
        write_json(directory / 'identity.json', identity)
        np.savez(directory / 'rank000.npz', synthetic=np.zeros(1))
        summaries[split] = dict(identity=copy.deepcopy(identity), prediction_archive=str(directory / 'rank000.npz'))
    return state, lock, summaries


class R110RunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        guard = patch('torch.cuda._lazy_init', side_effect=AssertionError('CPU tests must not initialize CUDA'))
        guard.start()
        self.addCleanup(guard.stop)

    def test_cli_constants_schedule_and_private_engine_do_not_mutate_history(self):
        before = (runner.engine.micro_caps, runner.engine.write_json, runner.engine.consistency_weight,
                  original.RDropObjective.forward, historical_objective.ConditionalPriorObjective.forward)
        local = runner.private_engine()
        self.assertIsNot(local, runner.engine)
        self.assertIs(local.train.__globals__['micro_caps'], runner.micro_caps)
        self.assertIs(local.evaluate.__wrapped__.__globals__['micro_caps'], runner.micro_caps)
        self.assertIs(local.train.__globals__['write_json'], runner._engine_write_json)
        self.assertIs(local.train.__globals__['consistency_weight'], runner.consistency_weight)
        self.assertEqual([local.consistency_weight(u) for u in (0, 1, 4320, 57600)], [0.] * 4)
        self.assertEqual(runner.engine.consistency_weight(4320), .35)
        self.assertEqual(runner.CAPS, dict(large_micro_cap=256, other_micro_cap=1024, eval_batch_size=256))
        self.assertEqual(runner.PARAMETER_COUNTS,
            dict(original=1842953, A_tied_ce=2305033, B_private_ce=10220553))
        for arm in runner.ARMS:
            for command in ('train', 'test', 'timing', 'describe', 'status'):
                args = runner.parser().parse_args([command, '--arm', arm])
                self.assertEqual((args.large_micro_cap, args.eval_batch_size, args.gpu_type), (256, 256, '4090'))
                self.assertEqual(args.wandb_mode, 'offline')
                self.assertEqual(args.cache_root, runner.DEFAULT_CACHE_ROOT)
            with redirect_stdout(io.StringIO()):
                description = runner.main(['describe', '--arm', arm])
            self.assertEqual(description['model_arm'], runner.MODEL_ARM_MAP[arm])
            self.assertEqual(description['eval_batch_size'], 256)
        order = runner.task_order(runner.MAX_UPDATES)
        self.assertEqual({p: order.count(p) for p in runner.PROBLEMS}, dict.fromkeys(runner.PROBLEMS, 3200))
        self.assertEqual((runner.BATCH_SIZE, runner.MAX_UPDATES, runner.WARMUP_UPDATES,
                          runner.EVAL_EVERY, runner.TRAIN_EVERY), (1024, 57600, 4320, 1440, 7200))
        self.assertEqual((runner.learning_rate(4320), runner.learning_rate(57600)), (1e-4, 1e-6))
        self.assertEqual(before, (runner.engine.micro_caps, runner.engine.write_json, runner.engine.consistency_weight,
            original.RDropObjective.forward, historical_objective.ConditionalPriorObjective.forward))
        with self.assertRaises(ValueError):
            runner.micro_caps(large_cap=128)

    def test_factories_delegate_mapped_arm_and_keep_r109_params(self):
        for arm in runner.ARMS:
            params, state = full18_params(arm), {'synthetic': torch.ones(1)}
            untouched = copy.deepcopy(params)
            with patch.object(runner.models, 'make_model', return_value='mapped') as factory:
                self.assertEqual(runner.make_model(params, arm, state), 'mapped')
                factory.assert_called_once_with(params, runner.MODEL_ARM_MAP[arm], state)
            self.assertEqual(params, untouched)
            initial = dict(trainable_sha256=runner.ORIGINAL_FRESH_SHA256)
            geometry = dict(split='train')
            model = torch.nn.Linear(1, 1, bias=False)
            receipt = dict(common_parameter_sha256=runner.ORIGINAL_FRESH_SHA256)
            with patch.dict(runner.PARAMETER_COUNTS, {arm: 1}), patch.object(runner.models,
                    'initialize_shared', return_value=(model, params, receipt)) as initialize:
                _, actual_params, actual_receipt = runner.initialize_shared(initial, geometry, arm)
                initialize.assert_called_once_with(initial, geometry, runner.MODEL_ARM_MAP[arm], 'cpu')
            self.assertEqual(actual_params['r109_arm'], runner.MODEL_ARM_MAP[arm])
            self.assertFalse(actual_receipt['identical_complete_fresh_parameters'])
            self.assertFalse(actual_receipt['total_parameter_matched'])
        with self.assertRaises(ValueError):
            runner.make_model({}, 'unknown')
        with self.assertRaises(ValueError):
            runner.initialize_shared(dict(trainable_sha256='trained'), {}, runner.ARMS[0])

    def test_actual_model_checkpoint_roundtrip_needs_no_prior_or_live_lock(self):
        for arm in runner.ARMS:
            with self.subTest(arm=arm), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                _, config = config_fixture(root, arm)
                model = runner.make_model(config['model_params'], arm)
                self.assertEqual(sum(p.numel() for p in model.parameters()), runner.PARAMETER_COUNTS[arm])
                self.assertEqual(model.params['r109_arm'], runner.MODEL_ARM_MAP[arm])
                self.assertEqual(config['source_sha256']['r109_model.py'], runner.file_hash(runner.V4 / 'r109_model.py'))
                self.assertFalse(any('r108' in name for name in config['source_sha256']))
                self.assertNotIn('prior_spec', config)
                with torch.no_grad():
                    next(model.parameters()).add_(.01)
                state = dict(model=model.state_dict(), args=config, state_hash=runner.json_hash(config),
                             successful_updates=1440, epoch=1)
                path = root / 'synthetic_trained.pt'
                save_checkpoint(path, state)
                with patch.object(runner, 'read_json', side_effect=AssertionError('Restore must use embedded protocol')):
                    restored, loaded = runner.load_checkpoint(path)
                self.assertEqual(parameter_hash(restored.state_dict()), parameter_hash(state['model']))
                self.assertEqual(loaded['successful_updates'], 1440)
                self.assertEqual(restored.params['r109_arm'], runner.MODEL_ARM_MAP[arm])
                broken = dict(state, model=dict(state['model']))
                del broken['model'][next(iter(broken['model']))]
                save_checkpoint(path, broken)
                with self.assertRaises(RuntimeError):
                    runner.load_checkpoint(path)
                save_checkpoint(path, dict(state, objective_state={}))
                with self.assertRaises(ValueError):
                    runner.load_checkpoint(path)
                for field, value in (('rdrop_weight', .35), ('loss', runner.engine.LOSS_RECIPE),
                                     ('priors_path', '/forbidden/prior.json'), ('total_parameter_matched', True)):
                    altered = copy.deepcopy(config)
                    altered[field] = value
                    with self.assertRaises(ValueError, msg=field):
                        runner.validate_checkpoint_config(altered)

    def test_single_forward_row_weighted_logical_batch_and_stateless_objective(self):
        batch = dict(ind=torch.arange(1024) % 3, node=torch.linspace(.01, .1, 1024)[:, None, None],
                     costs=torch.zeros(1024, 3), cost_regret=torch.ones(1024, 3))
        for arm in runner.ARMS:
            outcomes = []
            for problem, cap in (('TSP', 256), ('ATSP', 1024)):
                selector = TinySelector(arm)
                objective = runner.make_objective(selector, arm)
                self.assertEqual(runner.checkpoint_extras(objective), {})
                runner.restore_objective(objective, {})
                with self.assertRaises(ValueError):
                    runner.restore_objective(objective, {'objective_state': {}})
                optimizer = torch.optim.SGD(selector.parameters(), lr=.01)
                with patch.object(optimizer, 'step', wraps=optimizer.step) as step:
                    parts, retries, _ = runner.update_batch(objective, optimizer,
                        torch.amp.GradScaler('cpu', enabled=False), batch, 0.,
                        Runtime(0, 1, torch.device('cpu')), cap, problem=problem)
                self.assertEqual((step.call_count, retries, selector.calls), (1, 0, 1024 // cap))
                self.assertTrue(torch.isfinite(parts).all())
                self.assertTrue((parts[2:-1] == 0).all())
                outcomes.append((selector.weight.detach().clone(), parts))
            for first, second in zip(*outcomes):
                torch.testing.assert_close(first, second, rtol=1e-6, atol=1e-7)
        objective = runner.make_objective(TinySelector(runner.ARMS[0]), runner.ARMS[0])
        optimizer = torch.optim.SGD(objective.parameters(), lr=.01)
        for changed, cap in ((dict(batch, ind=batch['ind'][:1000]), 256), (batch, 128),
                             (dict(batch, context_winner=torch.ones(1024)), 256)):
            with self.assertRaises(ValueError):
                runner.update_batch(objective, optimizer, torch.amp.GradScaler('cpu', enabled=False),
                    changed, 0., Runtime(0, 1, torch.device('cpu')), cap, problem='TSP')

    def test_selected_train_val_binding_is_exact_and_preserves_archives(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, lock, summaries = capture_fixture(root)
            paths = [Path(s['prediction_archive']) for s in summaries.values()]
            paths += [p.parent / 'identity.json' for p in paths]
            hashes = {str(p): runner.file_hash(p) for p in paths}
            final = dict(state_hash=lock['state_hash'], selected_update=1440,
                         checkpoint_sha256=lock['checkpoint_sha256'], **summaries)
            bound = runner.bind_selected_val(root, final)
            for split in ('train', 'val'):
                self.assertIsNone(bound[split]['archive_capture_identity']['checkpoint_sha256'])
                self.assertEqual(bound[split]['identity']['checkpoint_sha256'], lock['checkpoint_sha256'])
                self.assertEqual(bound[split]['checkpoint_binding']['rule'], runner.VAL_BINDING_RULE)
            self.assertEqual(bound, runner.bind_selected_val(root, bound))
            self.assertEqual(hashes, {str(p): runner.file_hash(p) for p in paths})
            for field, wrong in (('model_state_sha256', 'wrong'), ('state_hash', 'wrong'),
                                 ('successful_updates', 2880), ('arm', runner.ARMS[1]),
                                 ('checkpoint_sha256', 'wrong'), ('experiment', 'R109')):
                invalid = copy.deepcopy(summaries['val'])
                invalid['identity'][field] = wrong
                write_json(Path(invalid['prediction_archive']).parent / 'identity.json', invalid['identity'])
                with self.assertRaises(ValueError, msg=field):
                    runner.selected_capture_proof(root, invalid, lock, 'val')
            runner._engine_write_json(root / 'unrelated.json', {'unchanged': True})
            self.assertEqual(runner.read_json(root / 'unrelated.json'), {'unchanged': True})

    def test_test_authorization_reuses_only_complete_preexisting_full18_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args, config = config_fixture(root, runner.ARMS[0])
            run = root / args.arm
            run.mkdir()
            save_checkpoint(run / 'best.pt', dict(model={'synthetic': torch.ones(1)}))
            lock = dict(experiment='R110', arm=args.arm, selected_without_test=True,
                per_problem_assembly=False, state_hash=runner.json_hash(config),
                checkpoint=str(run / 'best.pt'), checkpoint_sha256=runner.file_hash(run / 'best.pt'),
                selected_update=1440, inference_rule=runner.INFERENCE_RULE)
            write_json(run / 'protocol.json', config)
            write_json(run / 'final_test_lock.json', lock)
            write_json(run / 'training_complete.json', dict(successful_updates=runner.MAX_UPDATES))
            manifest = dict(splits=dict(test={p: dict(rows=4500, complete=True) for p in runner.PROBLEMS}))
            path = args.prepared_root / 'source_cache_manifest.json'
            write_json(path, manifest)
            with patch.object(runner.engine, 'ensure_test_cache', side_effect=AssertionError('Never build historical TEST')):
                self.assertEqual(runner.ensure_test_cache(args, lock), manifest)
            manifest['splits']['test'].pop(runner.PROBLEMS[0])
            write_json(path, manifest)
            with self.assertRaises(PermissionError):
                runner.ensure_test_cache(args, lock)

    def test_input_only_timing_schema_includes_method_node_refinement(self):
        selector = TinySelector(runner.ARMS[0])
        batch = dict(node=torch.ones(4, 1, 1), node_mask=torch.ones(4, 1, dtype=torch.bool),
                     ind=torch.zeros(4, dtype=torch.long), costs=torch.ones(4, 3), winner=object())
        measured = runner.measure_inputs(selector, batch, torch.device('cpu'))
        self.assertEqual(selector.calls, 4)
        self.assertEqual(len(measured['batch_seconds']), 3)
        self.assertFalse(set(runner.raw_inputs(batch)).intersection(('ind', 'costs', 'winner')))
        lock = dict(checkpoint_sha256='checkpoint', state_hash='protocol', selected_update=1440)
        value = dict(experiment='R110', arm=runner.ARMS[0], complete=True, **lock,
            model_state_sha256='model', model_factory=runner.MODEL_FACTORY, inference_rule=runner.INFERENCE_RULE,
            parameter_count=runner.PARAMETER_COUNTS[runner.ARMS[0]], all_current_model_included=True,
            method_conditioned_node_refinement_included=True, input_labels_read=False,
            donor_present=False, cached_feature_latency_used=False, model_reselection=False,
            scope='complete_standalone_model_original_GPU_resident_inputs',
            per_problem=[dict(problem=p, instances=128,
                standalone_model=dict(batch_seconds=[.1, .2, .3], mean_seconds_per_instance=.6 / 384))
                for p in runner.PROBLEMS])
        self.assertEqual(runner.validate_timing_record(value, lock, runner.ARMS[0], 'model'), value)
        for field, wrong in (('method_conditioned_node_refinement_included', False), ('state_hash', 'other')):
            with self.assertRaises(ValueError):
                runner.validate_timing_record(dict(value, **{field: wrong}), lock, runner.ARMS[0], 'model')


if __name__ == '__main__':
    unittest.main()

"""Bounded CPU runner contracts; synthetic artifacts never become run results."""

import copy
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from . import r109_experiment as runner
from . import r76_experiment as original
from .pairwise_selector import maximin_logits
from .r76_experiment import Runtime, save_checkpoint, write_json
from .r77_data import parameter_hash
from .V4Model import get_default_model_params


def full18_params(arm=None):
    params = get_default_model_params()
    params.update(runner.DIMENSIONS, architecture='pairwise', local_geometry=True,
        geometry_mode='real', ignore_coord_dist=True, native_winner=True, dropout=.1,
        decision_head='classification', pair_mode='score_difference',
        solver_representation='handcrafted', solver_feature_weight=1.0, sdpa=True)
    if arm is not None:
        params.update(r109_arm=arm, r109_spec=copy.deepcopy(runner.models.SPEC),
                      role_arm=runner.MODEL_ROLE_ARM)
    return params


class TinyStoredModel(torch.nn.Module):
    def __init__(self, params, count):
        super().__init__()
        self.params = copy.deepcopy(params)
        self.weight = torch.nn.Parameter(torch.linspace(.1, .3, count))
        self.register_buffer('geometry', torch.tensor([.2, 1.3]))


@contextmanager
def tiny_factory():
    """Only the round-trip test substitutes explicitly synthetic parameter counts."""
    counts = dict(original=1, A_tied_late=2, B_private_late=4)

    def factory(params, arm, state_dict=None):
        model = TinyStoredModel(params, counts[arm])
        if state_dict is not None:
            model.load_state_dict(state_dict, strict=True)
        return model

    with patch.object(runner, 'PARAMETER_COUNTS', counts), patch.object(runner, 'make_model', factory):
        yield


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


class TinyObjective(torch.nn.Module):
    def __init__(self, fail=False, always_bad=False):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(()))
        self.fail, self.always_bad, self.draws = fail, always_bad, []

    def forward(self, batch, weight):
        self.draws.append(torch.rand_like(batch['x']))
        loss = (self.weight * batch['x']).square().mean()
        if self.fail or self.always_bad:
            self.fail = False
            loss = loss * float('nan')
        return loss, loss.detach().expand(8)


class TinyScoreSelector(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([.3, -.2, .1]))
        self.seen, self.outputs = [], []

    def forward(self, batch):
        self.seen.append(set(batch))
        utility = batch['node'][:, 0, :1] * self.weight[None]
        utility = utility * (1 + .1 * len(self.seen))
        margins = utility[:, :, None] - utility[:, None, :]
        mask = torch.ones_like(utility, dtype=torch.bool)
        output = dict(logits=maximin_logits(margins, mask), pair_margin=margins, solver_mask=mask)
        self.outputs.append(output)
        return output


def capture_fixture(root, split='val'):
    root.mkdir(parents=True, exist_ok=True)
    config = dict(experiment='R109', arm=runner.ARMS[0], synthetic=True)
    state = dict(args=config, model={'weight': torch.tensor([.4])},
                 state_hash=runner.json_hash(config), successful_updates=1440)
    save_checkpoint(root / 'best.pt', state)
    lock = dict(arm=config['arm'], checkpoint_sha256=runner.file_hash(root / 'best.pt'),
                state_hash=state['state_hash'], selected_update=1440)
    write_json(root / 'final_test_lock.json', lock)
    directory = root / 'evaluation' / split / 'update0001440'
    identity = dict(experiment='R109', arm=config['arm'], split=split, successful_updates=1440,
        model_state_sha256=parameter_hash(state['model']), state_hash=state['state_hash'], checkpoint_sha256=None)
    write_json(directory / 'identity.json', identity)
    np.savez(directory / 'rank000.npz', synthetic=np.zeros(1))
    summary = dict(identity=copy.deepcopy(identity), prediction_archive=str(directory / 'rank000.npz'))
    return state, lock, summary


class R109RunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_cli_schedule_single_caps_and_isolated_private_engine(self):
        for arm in runner.ARMS:
            for command in ('train', 'test', 'timing', 'describe', 'status'):
                args = runner.parser().parse_args([command, '--arm', arm])
                self.assertEqual((args.large_micro_cap, args.eval_batch_size),
                                 (runner.CAPS['large_micro_cap'], runner.CAPS['eval_batch_size']))
                self.assertEqual(args.cache_root, runner.DEFAULT_CACHE_ROOT)
                self.assertEqual(args.wandb_mode, 'offline')
        local = runner.private_engine()
        self.assertIsNot(local, runner.engine)
        self.assertIs(local.train.__globals__['micro_caps'], runner.micro_caps)
        self.assertIs(local.evaluate.__wrapped__.__globals__['micro_caps'], runner.micro_caps)
        self.assertIs(local.train.__globals__['write_json'], runner._engine_write_json)
        self.assertIsNot(runner.engine.micro_caps, runner.micro_caps)
        self.assertIs(runner.engine.write_json, runner.write_json)
        self.assertEqual(runner.micro_caps()['ATSP'], runner.CAPS['other_micro_cap'])
        with patch.dict(runner.CAPS, large_micro_cap=32, other_micro_cap=128, eval_batch_size=16):
            args = runner.parser().parse_args(['describe', '--arm', runner.ARMS[0]])
            self.assertEqual((args.large_micro_cap, args.eval_batch_size), (32, 16))
            self.assertEqual(runner.micro_caps()['TSP'], 32)
            self.assertEqual(runner.micro_caps()['ATSP'], 128)
            self.assertEqual(runner.required_lock_fields()['eval_batch_size'], 16)
        with self.assertRaises(ValueError):
            runner.micro_caps(large_cap=runner.CAPS['large_micro_cap'] + 1)
        order = runner.task_order(runner.MAX_UPDATES)
        self.assertEqual(set(order), set(runner.PROBLEMS))
        self.assertTrue(all(order.count(p) == 3200 for p in runner.PROBLEMS))
        self.assertEqual(runner.learning_rate(4320), 1e-4)
        self.assertEqual(runner.learning_rate(57600), 1e-6)
        self.assertEqual(runner.consistency_weight(4320), .35)

    def test_common_original_initialization_without_cross_arm_fullhash_claim(self):
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(2)
            source = runner.engine.make_model(full18_params(), 'A_baseline')
        weights = {k: v.detach().clone() for k, v in source.named_parameters()}
        initial = dict(seed=2, initialization='scratch', model_params=source.params,
                       trainable=weights, trainable_sha256=parameter_hash(weights))
        geometry = dict(mean=[.2] * 12, std=[1.3] * 12, split='train')
        hashes, counts = [], []
        with patch.object(runner.models, 'initialize_shared', wraps=runner.models.initialize_shared) as initialize:
            for arm in runner.ARMS:
                model, params, receipt = runner.initialize_shared(initial, geometry, arm)
                count = sum(p.numel() for p in model.parameters())
                self.assertEqual(count, runner.PARAMETER_COUNTS[arm])
                self.assertEqual(params['r109_arm'], arm)
                self.assertEqual(receipt['common_parameter_sha256'], runner.ORIGINAL_FRESH_SHA256)
                self.assertFalse(receipt['identical_complete_fresh_parameters'])
                self.assertFalse(receipt['total_parameter_matched'])
                self.assertEqual(receipt['branch_init_seed'], runner.models.BRANCH_INIT_SEED)
                self.assertNotIn('cross_arm_complete_state_verified', receipt)
                hashes.append(receipt['fresh_state_sha256'])
                counts.append(count)
            self.assertEqual(initialize.call_count, len(runner.ARMS))
        self.assertNotEqual(*hashes)
        self.assertNotEqual(*counts)
        exported = runner.models.P_COUNTS if hasattr(runner.models, 'P_COUNTS') else runner.models.PARAMETER_COUNTS
        self.assertIs(runner.PARAMETER_COUNTS, exported)
        with self.assertRaises(ValueError):
            runner.initialize_shared(dict(initial, trainable_sha256='trained'), geometry, runner.ARMS[0])

    def test_original_stateless_two_forward_rdrop_and_loss(self):
        selector = TinyScoreSelector()
        objective = runner.make_objective(selector, runner.ARMS[0])
        self.assertIs(type(objective), original.RDropObjective)
        n = 4
        batch = dict(node=torch.linspace(.5, 1., n)[:, None, None], pool_ids=torch.tensor([0, 1, 2]),
            ind=torch.tensor([0, 1, 2, 0]), costs=torch.ones(n, 3),
            cmp_target=torch.ones(n, 3), cmp_valid=torch.ones(n, 3, dtype=torch.bool),
            cmp_true3=torch.ones(n, 3, dtype=torch.bool), cost_regret=torch.full((n, 3), .01))
        loss, parts = objective(batch, .23)
        self.assertEqual(len(selector.outputs), 2)
        self.assertTrue(all(keys == {'node', 'pool_ids'} for keys in selector.seen))
        first, second = selector.outputs
        focus = original.predicted_top3(first['logits'], second['logits'], batch['pool_ids'], first['solver_mask'])
        loss1, _ = original.single_objective(first, batch, focus)
        loss2, _ = original.single_objective(second, batch, focus)
        expected = .5 * (loss1 + loss2) + .23 * original.symmetric_kl(
            first['logits'], second['logits'], first['solver_mask'])
        torch.testing.assert_close(loss, expected, rtol=0, atol=0)
        self.assertEqual(parts.shape, (8,))
        loss.backward()
        self.assertTrue(torch.isfinite(selector.weight.grad).all())
        self.assertEqual(runner.checkpoint_extras(objective), {})
        runner.restore_objective(objective, {})
        with self.assertRaises(ValueError):
            runner.restore_objective(objective, {'objective_state': {}})
        with self.assertRaises(TypeError):
            runner.checkpoint_extras(TinyObjective())

    def test_row_weighted_microbatches_single_step_and_rng_replay(self):
        batch = dict(ind=torch.arange(1024), x=torch.linspace(.01, .1, 1024))
        for cap, problem in ((runner.CAPS['large_micro_cap'], 'TSP'), (300, 'ATSP')):
            outcomes, rngs = [], []
            with patch.dict(runner.CAPS, other_micro_cap=300):
                for fail in (False, True):
                    with torch.random.fork_rng(devices=[]):
                        torch.random.default_generator.manual_seed(109)
                        objective = TinyObjective(fail=fail)
                        optimizer = torch.optim.SGD(objective.parameters(), lr=.01)
                        with patch.object(optimizer, 'step', wraps=optimizer.step) as step:
                            parts, retries, _ = runner.update_batch(objective, optimizer,
                                torch.amp.GradScaler('cpu', enabled=False), batch, .35,
                                Runtime(0, 1, torch.device('cpu')), cap, problem=problem)
                        self.assertEqual(step.call_count, 1)
                        self.assertEqual(retries, int(fail))
                        self.assertTrue(torch.isfinite(parts).all())
                        outcomes.append(objective.weight.detach().clone())
                        rngs.append(torch.get_rng_state().clone())
                        if fail:
                            micros = (1024 + cap - 1) // cap
                            self.assertEqual(len(objective.draws), 2 * micros)
                            for before, replay in zip(objective.draws[:micros], objective.draws[micros:]):
                                torch.testing.assert_close(before, replay, rtol=0, atol=0)
            torch.testing.assert_close(*outcomes, rtol=0, atol=0)
            torch.testing.assert_close(*rngs, rtol=0, atol=0)
            torch.testing.assert_close(outcomes[0], 1 - .02 * batch['x'].square().mean(), rtol=0, atol=1e-7)

    def test_update_rejects_changed_rows_caps_context_and_never_accepts_nan(self):
        batch = dict(ind=torch.ones(1024, dtype=torch.long), x=torch.ones(1024))
        objective = TinyObjective(always_bad=True)
        optimizer = torch.optim.SGD(objective.parameters(), lr=.01)
        runtime, scaler = Runtime(0, 1, torch.device('cpu')), torch.amp.GradScaler('cpu', enabled=False)
        cap = runner.CAPS['large_micro_cap']
        for bad, bad_cap in ((dict(batch, ind=batch['ind'][:-1]), cap),
                             (dict(batch, donor_logits=torch.ones(1)), cap), (batch, cap + 1)):
            with self.assertRaises(ValueError):
                runner.update_batch(objective, optimizer, scaler, bad, .35, runtime, bad_cap, problem='TSP')
        with patch.object(optimizer, 'step', wraps=optimizer.step) as step:
            with self.assertRaises(RuntimeError):
                runner.update_batch(objective, optimizer, scaler, batch, .35, runtime, cap,
                                    problem='TSP', max_retries=2)
            step.assert_not_called()

    def test_protocol_and_complete_checkpoint_factory_round_trip(self):
        with tiny_factory():
            for arm in runner.ARMS:
                with self.subTest(arm=arm), tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    _, config = config_fixture(root, arm)
                    model = runner.make_model(config['model_params'], arm)
                    path = root / 'synthetic_checkpoint.pt'
                    state = dict(model=model.state_dict(), args=config,
                        state_hash=runner.json_hash(config), successful_updates=1440, epoch=1)
                    save_checkpoint(path, state)
                    restored, loaded = runner.load_checkpoint(path, 'cpu')
                    self.assertEqual(parameter_hash(model.state_dict()), parameter_hash(restored.state_dict()))
                    self.assertEqual(loaded['args'], config)
                    self.assertIs(runner.from_checkpoint, runner.load_checkpoint)
                    self.assertIs(runner.load_model, runner.load_checkpoint)
                    for field, value in (('auxiliary_loss', True), ('source_vectors', True),
                            ('new_features', True), ('new_targets', True), ('solver_quotas', True),
                            ('total_parameter_matched', True), ('identical_complete_fresh_parameters', True),
                            ('branch_init_seed', 1), ('model_spec', {}), ('eval_batch_size', 1),
                            ('model_factory', 'code.V4.r107_experiment.load_checkpoint')):
                        broken = copy.deepcopy(config)
                        broken[field] = value
                        with self.assertRaises(ValueError, msg=field):
                            runner.validate_checkpoint_config(broken)
                    for field, value in (('r109_arm', 'another_arm'), ('r109_spec', {}),
                                         ('encoder_layer_num', 6), ('role_arm', 'A_canonical_anchor')):
                        broken = copy.deepcopy(config)
                        broken['model_params'][field] = value
                        with self.assertRaises(ValueError, msg=field):
                            runner.validate_checkpoint_config(broken)
                    save_checkpoint(path, dict(state, objective_state={}))
                    with self.assertRaises(ValueError):
                        runner.load_checkpoint(path)

    def test_main_lock_and_preexisting_full18_test_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args, config = config_fixture(root, runner.ARMS[0])
            lock = dict(checkpoint_sha256='synthetic', state_hash='synthetic')
            tasks = {p: dict(rows=runner.SPLIT_ROWS['test'], complete=True) for p in runner.PROBLEMS}
            manifest = dict(splits=dict(test=tasks))
            path = args.prepared_root / 'source_cache_manifest.json'
            write_json(path, manifest)
            with patch.object(runner, 'authorize_test', return_value=(lock, config)):
                self.assertEqual(runner.ensure_test_cache(args, lock), manifest)
                manifest['splits']['test']['TSP']['complete'] = False
                write_json(path, manifest)
                with self.assertRaises(PermissionError):
                    runner.ensure_test_cache(args, lock)
                with self.assertRaises(PermissionError):
                    runner.ensure_test_cache(args, dict(lock, state_hash='other'))
            record = runner.read_json(root / 'protocol_lock.json')
            record['logical_global_batch'] = 256
            write_json(root / 'protocol_lock.json', record)
            with self.assertRaises(ValueError):
                runner.protocol_lock(args)

    def test_test_authorization_requires_completed_budget_and_unique_frozen_lock(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args, config = config_fixture(root, runner.ARMS[0])
            run = root / args.arm
            run.mkdir()
            state_hash = runner.json_hash(config)
            save_checkpoint(run / 'best.pt', dict(args=config, model={'synthetic': torch.ones(1)},
                state_hash=state_hash, successful_updates=1440))
            lock = dict(experiment='R109', arm=args.arm, checkpoint=str(run / 'best.pt'),
                checkpoint_sha256=runner.file_hash(run / 'best.pt'), state_hash=state_hash,
                selected_update=1440, selected_without_test=True, per_problem_assembly=False,
                inference_rule=runner.INFERENCE_RULE)
            write_json(run / 'protocol.json', config)
            write_json(run / 'final_test_lock.json', lock)
            write_json(run / 'training_complete.json', dict(successful_updates=runner.MAX_UPDATES))
            self.assertEqual(runner.authorize_test(args), (lock, config))
            write_json(run / 'training_complete.json', dict(successful_updates=runner.MAX_UPDATES - 1))
            with self.assertRaises(PermissionError):
                runner.authorize_test(args)
            write_json(run / 'training_complete.json', dict(successful_updates=runner.MAX_UPDATES))
            write_json(run / 'final_test_lock.json', dict(lock, per_problem_assembly=True))
            with self.assertRaises(PermissionError):
                runner.authorize_test(args)
            write_json(run / 'final_test_lock.json', lock)
            record = runner.read_json(root / 'protocol_lock.json')
            write_json(root / 'protocol_lock.json', dict(record, main_change_after_training=True))
            with self.assertRaises(PermissionError):
                runner.authorize_test(args)

    def test_selected_val_binding_preserves_capture_and_full_train_proof(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, lock, summary = capture_fixture(root)
            original_summary = copy.deepcopy(summary)
            archive = Path(summary['prediction_archive'])
            hashes = (runner.file_hash(archive), runner.file_hash(archive.parent / 'identity.json'))
            final = dict(state_hash=lock['state_hash'], selected_update=lock['selected_update'],
                         checkpoint_sha256=lock['checkpoint_sha256'], val=summary)
            bound = runner.bind_selected_val(root, final)
            self.assertEqual(summary, original_summary)
            self.assertEqual(bound['val']['archive_capture_identity'], original_summary['identity'])
            self.assertEqual(bound['val']['identity']['checkpoint_sha256'], lock['checkpoint_sha256'])
            self.assertEqual(bound['val']['checkpoint_binding']['rule'], runner.VAL_BINDING_RULE)
            self.assertEqual(runner.bind_selected_val(root, bound), bound)
            self.assertEqual(hashes, (runner.file_hash(archive), runner.file_hash(archive.parent / 'identity.json')))
            runner._engine_write_json(root / 'final_metrics.json', final)
            self.assertEqual(runner.read_json(root / 'final_metrics.json'), bound)
            _, train_lock, training = capture_fixture(root, 'train')
            capture, proof = runner.selected_capture_proof(root, training, train_lock, 'train')
            self.assertEqual(capture['split'], 'train')
            self.assertIsNone(proof['archive_capture_checkpoint_sha256'])

    def test_selected_capture_rejects_wrong_arm_state_update_and_external_archive(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            state, lock, summary = capture_fixture(root)
            for field, value in (('arm', runner.ARMS[1]), ('successful_updates', 2880),
                                 ('state_hash', 'different'), ('model_state_sha256', 'different')):
                bad = copy.deepcopy(summary)
                bad['identity'][field] = value
                write_json(Path(bad['prediction_archive']).parent / 'identity.json', bad['identity'])
                with self.assertRaises(ValueError, msg=field):
                    runner.selected_capture_proof(root, bad, lock, 'val', state)
            write_json(Path(summary['prediction_archive']).parent / 'identity.json', summary['identity'])
            with self.assertRaises(ValueError):
                runner.selected_capture_proof(root, dict(summary, prediction_archive=str(root / 'foreign.npz')),
                                              lock, 'val', state)
            with self.assertRaises(ValueError):
                runner.selected_capture_proof(root, summary, dict(lock, checkpoint_sha256='other'), 'val', state)

    def test_input_only_cache_uses_eval_cap_and_opens_no_supervision_arrays(self):
        with tempfile.TemporaryDirectory() as directory:
            root, count = Path(directory), runner.CAPS['eval_batch_size']
            for problem, kind in (('TSP', 'coord'), ('ATSP', 'matrix')):
                cache = root / 'cache' / 'test' / problem
                config = dict(cache_root=str(root / 'cache'), data_root=str(root / 'data'), eval_batch_size=count,
                    candidate_pools={problem: ['synthetic0', 'synthetic1']}, candidate_pool_ids={problem: [0, 1]})
                record = dict(problem=problem, split='test', rows=runner.SPLIT_ROWS['test'],
                    pool=config['candidate_pools'][problem], pool_ids=[0, 1], data_root=config['data_root'],
                    kind=kind, problem_id=runner.PROBLEMS.index(problem), cbits=[0.] * 5,
                    problem_desc=[0.] * 12, mask=[1., 1.], d_in=8)
                write_json(cache / 'metadata.json', record)
                manifest = dict(splits=dict(test={problem: record}))
                arrays = dict(n=np.full(count, 4, dtype=np.int64), coord_dist=np.ones(count, dtype=np.float32),
                    node=np.zeros((count, 4, 8), dtype=np.float32), node_geom=np.zeros((count, 4, 12), dtype=np.float32),
                    matrix=np.zeros((count, 4, 4), dtype=np.float32))
                specs = dict.fromkeys((*arrays, 'costs', 'winner', 'cmp_target'), 'synthetic')
                with patch('code.V4.r76_cache._validate_record', return_value=specs), \
                        patch('code.V4.r75_cache._open_arrays', return_value=arrays) as opened, \
                        patch('code.V4.r75_cache._close') as closed:
                    batch = runner.cached_input_batch(config, manifest, 'test', problem, torch.device('cpu'))
                    self.assertEqual(len(batch['node_mask']), count)
                    keys = set(opened.call_args.args[1])
                    self.assertEqual(keys, {'n', 'coord_dist', 'node', 'node_geom'} if kind == 'coord'
                                     else {'n', 'coord_dist', 'matrix'})
                    closed.assert_called_once_with(arrays)
                self.assertFalse(set(batch) & {'costs', 'winner', 'ind', 'cmp_target'})
                contaminated = dict(batch, costs=object(), ind=object(), donor_hidden=object())
                self.assertEqual(set(runner.raw_inputs(contaminated)), set(batch))

    def test_timing_metadata_contract_with_entire_gpu_work_mocked_on_cpu(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = runner.parser().parse_args(['timing', '--arm', runner.ARMS[0], '--root', str(root)])
            run = root / args.arm
            state, lock, _ = capture_fixture(run)
            model = torch.nn.Linear(1, 1, bias=False)
            state['model'] = copy.deepcopy(model.state_dict())
            save_checkpoint(run / 'best.pt', state)
            lock['checkpoint_sha256'] = runner.file_hash(run / 'best.pt')
            record = dict(complete=True, checkpoint_sha256=lock['checkpoint_sha256'],
                state_hash=lock['state_hash'], result=dict(identity=dict(model_state_sha256=parameter_hash(state['model']))))
            write_json(run / 'test_results.json', record)
            config = dict(eval_batch_size=runner.CAPS['eval_batch_size'])
            measurement = dict(batch_seconds=[.1, .1, .1], mean_seconds_per_instance=.1 / config['eval_batch_size'])
            with patch.object(runner, 'authorize_test', return_value=(lock, config)), \
                    patch.object(runner, 'load_checkpoint', return_value=(model, state)), \
                    patch.object(runner, 'ensure_test_cache', return_value={}), \
                    patch.object(runner, 'cached_input_batch', return_value={}) as inputs, \
                    patch.object(runner, 'measure_inputs', return_value=measurement), \
                    patch('torch.cuda.synchronize', side_effect=AssertionError('no CUDA work in CPU contract')):
                result = runner.timing(args, torch.device('cuda:0'))
            self.assertEqual(inputs.call_count, len(runner.PROBLEMS))
            self.assertTrue(result['method_conditioned_node_refinement_included'])
            self.assertEqual(result['parameter_count'], runner.PARAMETER_COUNTS[args.arm])
            self.assertFalse(result['input_labels_read'])
            self.assertEqual([r['problem'] for r in result['per_problem']], list(runner.PROBLEMS))
            self.assertTrue(all(r['instances'] == config['eval_batch_size'] for r in result['per_problem']))
            self.assertTrue(all(call.kwargs['count'] == config['eval_batch_size'] for call in inputs.call_args_list))
            self.assertNotIn('explicit_path_computation_included', result)
            with self.assertRaises(ValueError):
                runner.timing(args, torch.device('cpu'))

    def test_measure_inputs_executes_entire_model_and_rejects_nonfinite_predictions(self):
        model = TinyScoreSelector().eval()
        batch = dict(node=torch.ones(2, 1, 1), node_mask=torch.ones(2, 1, dtype=torch.bool),
            pool_ids=torch.tensor([0, 1, 2]), costs=object(), ind=object(), teacher_hidden=object())
        with patch('torch.cuda.synchronize', side_effect=AssertionError('no CUDA work in CPU contract')):
            result = runner.measure_inputs(model, batch, torch.device('cpu'))
        self.assertEqual(len(model.outputs), 4)
        self.assertTrue(all(keys == {'node', 'node_mask', 'pool_ids'} for keys in model.seen))
        self.assertEqual(len(result['batch_seconds']), 3)
        self.assertAlmostEqual(result['mean_seconds_per_instance'], sum(result['batch_seconds']) / 6)
        with torch.no_grad():
            model.weight.fill_(float('nan'))
        with self.assertRaises(ValueError):
            runner.measure_inputs(model, batch, torch.device('cpu'))


if __name__ == '__main__':
    unittest.main()

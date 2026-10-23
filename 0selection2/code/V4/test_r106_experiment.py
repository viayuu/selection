"""Bounded CPU interface checks. No data rows, GPU, launch or scientific results."""

import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from . import r106_experiment as runner
from .r76_experiment import PART_KEYS, RDropObjective, Runtime, save_checkpoint, write_json
from .r77_data import parameter_hash
from .V4Model import get_default_model_params


def full18_params():
    params = get_default_model_params()
    params.update(architecture='pairwise', joint_layer_num=2, local_geometry=True,
        geometry_mode='real', ignore_coord_dist=True, native_winner=True,
        decision_head='classification', pair_mode='score_difference',
        solver_representation='handcrafted', solver_feature_weight=1.0, sdpa=True)
    return params


def protocol_fixture(root, model, arm):
    args = runner.parser().parse_args(['train', '--arm', arm, '--root', str(root)])
    args.prepared_root = Path(root) / 'prepared'
    write_json(args.prepared_root / 'prepared.json', {})
    lock = dict(runner.required_lock_fields(), data_root=str(args.data_root),
        prepared_root=str(args.prepared_root), primary_cache=str(args.cache_root), main_extra='preserved')
    write_json(args.root / 'protocol_lock.json', lock)
    manifest = dict(splits=dict(train={p: dict(pool=['synthetic'], pool_ids=[0]) for p in runner.PROBLEMS}))
    config = runner.protocol(args, dict(trainable_sha256=runner.ORIGINAL_FRESH_SHA256),
        manifest, {}, {}, model.params, runner.task_order(57600), runner.micro_caps())
    return args, config


def capture_fixture(run, split='val', experiment='R106'):
    run.mkdir(parents=True, exist_ok=True)
    arm = runner.ARMS[0] if experiment == 'R106' else 'B'
    config = dict(experiment=experiment, arm=arm)
    state_hash = runner.json_hash(config)
    weights = dict(weight=torch.tensor([.2]), buffer=torch.tensor([1.3]))
    state = dict(args=config, model=weights, state_hash=state_hash, successful_updates=1440, epoch=1)
    save_checkpoint(run / 'best.pt', state)
    lock = dict(experiment=experiment, arm=arm, checkpoint_sha256=runner.file_hash(run / 'best.pt'),
        state_hash=state_hash, selected_update=1440, epoch=1)
    write_json(run / 'final_test_lock.json', lock)
    identity = dict(experiment=experiment, arm=arm, split=split, checkpoint_sha256=None,
        state_hash=state_hash, model_state_sha256=parameter_hash(weights), successful_updates=1440)
    archive = run / 'evaluation' / split / 'update0001440/rank000.npz'
    write_json(archive.parent / 'identity.json', identity)
    np.savez(archive, fixture=np.array([0]))
    summary = dict(identity=identity, prediction_archive=str(archive), macro={})
    final = dict(state_hash=state_hash, selected_update=1440, checkpoint_sha256=lock['checkpoint_sha256'],
        val=summary, epoch=1)
    return state, lock, summary, final


class TinyObjective(torch.nn.Module):
    def __init__(self, fail=False):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(.2))
        self.fail = fail

    def forward(self, batch, weight):
        loss = (self.weight * batch['x']).square().mean()
        if self.fail:
            self.fail = False
            loss = loss * float('nan')
        return loss, loss.detach().expand(len(PART_KEYS) - 1)


class R106RunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(2)
            original = runner.engine.make_model(full18_params(), 'A_baseline')
        weights = {k: v.detach().clone() for k, v in original.named_parameters()}
        initial = dict(seed=2, initialization='scratch', model_params=original.params,
            trainable=weights, trainable_sha256=parameter_hash(weights))
        geometry = dict(mean=[.2] * 12, std=[1.3] * 12)
        cls.models, cls.receipts = {}, {}
        for arm in runner.ARMS:
            model, _, receipt = runner.initialize_shared(initial, geometry, arm)
            cls.models[arm], cls.receipts[arm] = model, receipt
        cls.initial_state = copy.deepcopy(cls.models[runner.ARMS[0]].state_dict())

    def setUp(self):
        for model in self.models.values():
            model.load_state_dict(self.initial_state)
            model.zero_grad(set_to_none=True)
            model.eval()

    def test_cli_lock_contract_private_optimizer_and_eval_cap_isolation(self):
        for arm in runner.ARMS:
            for command in ('train', 'test', 'timing', 'describe', 'status'):
                args = runner.parser().parse_args([command, '--arm', arm])
                self.assertEqual((args.large_micro_cap, args.atsp_micro_cap, args.other_micro_cap, args.eval_batch_size),
                                 (64, 128, 256, 128))
                self.assertEqual(args.root, runner.DEFAULT_ROOT)
        original = runner.engine.micro_caps('4090', 128)
        private, evaluation = runner.private_engine(), runner.private_evaluator()
        self.assertIs(private.torch.optim.AdamW, runner.NativeScaledAdamW)
        self.assertIs(runner.engine.torch.optim.AdamW, torch.optim.AdamW)
        self.assertIsNot(private.torch, torch)
        self.assertEqual(runner.engine.micro_caps('4090', 128), original)
        self.assertEqual(min(128, evaluation.micro_caps()['TSP']), 128)
        self.assertEqual(runner.micro_caps()['TSP'], 64)
        args = runner.parser().parse_args(['describe', '--arm', runner.ARMS[0]])
        with patch.object(runner, 'model_module', side_effect=AssertionError('No model for CLI/lock contract')):
            value = runner.describe(args)
        self.assertEqual(value['required_protocol_lock'], runner.required_lock_fields())
        custom = runner.required_lock_fields(32, 64, 128, 64)
        self.assertEqual((custom['large_micro_cap'], custom['eval_batch_size']), (32, 64))
        final = runner.parser().parse_args(['train', '--arm', runner.ARMS[1],
            '--large-micro-cap', '256', '--atsp-micro-cap', '128', '--other-micro-cap', '1024',
            '--eval-batch-size', '128'])
        science = runner.required_lock_fields(**runner._execution(final))
        caps = runner.micro_caps(final.gpu_type, final.large_micro_cap,
            atsp_cap=final.atsp_micro_cap, other_cap=final.other_micro_cap)
        self.assertEqual((caps['TSP'], caps['ATSP'], caps['OVRPBLTW']), (256, 128, 1024))
        self.assertEqual(science['other_micro_cap'], 1024)
        self.assertEqual((runner.learning_rate(4320), runner.learning_rate(57600)), (1e-4, 1e-6))
        self.assertEqual(runner.consistency_weight(4320), .35)

    def test_same_complete_fresh_state_and_portable_factory_without_native_weight_reads(self):
        a, b = (self.receipts[arm] for arm in runner.ARMS)
        self.assertEqual(a['common_fresh_model_state_sha256'], b['common_fresh_model_state_sha256'])
        self.assertEqual([a['trainable_parameter_count'], b['trainable_parameter_count']], [1941257, 11965097])
        for arm in runner.ARMS:
            with self.subTest(arm=arm), tempfile.TemporaryDirectory() as temporary:
                model = self.models[arm]
                args, config = protocol_fixture(Path(temporary), model, arm)
                self.assertEqual(config['loss'], runner.engine.LOSS_RECIPE)
                with torch.no_grad():
                    next(model.donor.parameters()).add_(.001)
                    model.instance_encoder.geometry_residual.mean.add_(.1)
                path = Path(temporary) / 'checkpoint.pt'
                state = dict(args=config, model=model.state_dict(), state_hash=runner.json_hash(config),
                    successful_updates=1440, epoch=1)
                save_checkpoint(path, state)
                original_load = torch.load
                def guarded_load(filename, *positional, **keywords):
                    self.assertEqual(Path(filename), path, 'Restore attempted external native weight read')
                    return original_load(filename, *positional, **keywords)
                with patch('torch.load', side_effect=guarded_load):
                    restored, loaded = runner.load_checkpoint(path, 'cpu')
                self.assertEqual(parameter_hash(restored.state_dict()), parameter_hash(model.state_dict()))
                self.assertEqual(loaded['args'], config)
                self.assertTrue(all(not m.training for m in restored.donor.modules()))
                if arm == runner.ARMS[1]:
                    locked = runner.protocol_lock(args)['record']
                    args.large_micro_cap, args.atsp_micro_cap, args.other_micro_cap = 256, 128, 1024
                    execution = runner._execution(args)
                    write_json(args.root / 'protocol_lock.json', dict(locked, **runner.required_lock_fields(**execution)))
                    config = runner.protocol(args, dict(trainable_sha256=runner.ORIGINAL_FRESH_SHA256),
                        dict(splits=dict(train={p: dict(pool=['synthetic'], pool_ids=[0]) for p in runner.PROBLEMS})),
                        {}, {}, model.params, runner.task_order(57600),
                        runner.micro_caps(args.gpu_type, 256, atsp_cap=128, other_cap=1024))
                    self.assertEqual(config['microbatch_caps']['TSP'], 256)
                    self.assertEqual(config['microbatch_caps']['OVRPBLTW'], 1024)
                for field, wrong in (('native_feature_cache', True), ('native_lr_scale', 1.),
                        ('native_trainable', arm == runner.ARMS[0]), ('model_factory', 'code.V4.r102_experiment.load_checkpoint')):
                    with self.assertRaises(ValueError, msg=field):
                        runner.validate_checkpoint_config(dict(config, **{field: wrong}))
                record = runner.protocol_lock(args)['record']
                write_json(args.root / 'protocol_lock.json', dict(record, native_eval=False))
                with self.assertRaises(ValueError):
                    runner.protocol_lock(args)

    def test_real_thin_objective_one_native_graph_none_and_exception_clear(self):
        from .test_r106_model import raw_batch
        model = self.models[runner.ARMS[1]]
        model.train()
        objective = runner.make_objective(model, runner.ARMS[1])
        batch = raw_batch('TSP', lengths=(4, 5), slots=6)
        with patch.object(model.donor, 'forward', wraps=model.donor.forward) as donor, \
                patch.object(model, 'native_pair_context', wraps=model.native_pair_context) as context:
            loss, parts = objective(batch, .17)
        self.assertEqual(donor.call_count, 1)
        self.assertFalse(set(context.call_args.args[0]) & {'ind', 'costs', 'cmp_target', 'cost_regret'})
        self.assertIsNone(model._native_pair)
        self.assertEqual(len(parts), len(PART_KEYS) - 1)
        loss.backward()
        self.assertTrue(any(p.grad is not None for p in model.donor.parameters()))
        self.assertTrue(all(not m.training for m in model.donor.modules()))
        with patch.object(model, 'native_pair_context', side_effect=AssertionError('batchNone must bypass context')):
            zero, _ = objective(None, .17)
        self.assertEqual(float(zero), 0.)
        with patch.object(RDropObjective, 'forward', side_effect=RuntimeError('fixture')):
            with self.assertRaisesRegex(RuntimeError, 'fixture'):
                objective(batch, .17)
        self.assertIsNone(model._native_pair)
        self.assertEqual(runner.checkpoint_extras(objective), {})

    def test_optimizer_native_scale_survives_r84_lr_resets_and_resume(self):
        for frozen in (False, True):
            core, native = torch.nn.Parameter(torch.tensor([.4])), torch.nn.Parameter(torch.tensor([.5]), requires_grad=not frozen)
            native._r106_native = True
            ref_core, ref_native = torch.nn.Parameter(core.detach().clone()), torch.nn.Parameter(native.detach().clone())
            optimizer = runner.NativeScaledAdamW([core, native], lr=1e-4, weight_decay=1e-4)
            reference = torch.optim.AdamW([dict(params=[ref_core], lr=1e-4),
                dict(params=[ref_native], lr=1e-5)], weight_decay=1e-4)
            for update in (1, 4320, 57600):
                lr = runner.learning_rate(update)
                for g in optimizer.param_groups:
                    g['lr'] = lr
                reference.param_groups[0]['lr'], reference.param_groups[1]['lr'] = lr, lr * .1
                core.grad = ref_core.grad = torch.ones_like(core)
                native.grad = ref_native.grad = None if frozen else torch.ones_like(native)
                optimizer.step()
                reference.step()
                torch.testing.assert_close(core, ref_core, rtol=0, atol=0)
                torch.testing.assert_close(native, ref_native, rtol=0, atol=0)
                self.assertEqual([g['lr'] for g in optimizer.param_groups], [lr, lr])
            if frozen:
                self.assertEqual(float(native), float(torch.tensor(.5)))
            state = copy.deepcopy(optimizer.state_dict())
            restored = runner.NativeScaledAdamW([core, native], lr=1e-4, weight_decay=1e-4)
            restored.load_state_dict(state)
            self.assertEqual(restored.r106_successful_steps, 3)
            self.assertEqual(restored.group_structure(), state['r106_group_structure'])
            state['param_groups'][1]['lr_scale'] = 1.
            with self.assertRaises(ValueError):
                restored.load_state_dict(state)
            with patch.object(torch.optim.AdamW, 'step', side_effect=RuntimeError('step fixture')):
                with self.assertRaises(RuntimeError):
                    optimizer.step()
            self.assertEqual([g['lr'] for g in optimizer.param_groups], [lr, lr])
            self.assertEqual(optimizer.r106_successful_steps, 3)

    def test_actual_row_normalization_and_same_batch_nonfinite_retry(self):
        batch = dict(ind=torch.ones(1024, dtype=torch.long), x=torch.linspace(.1, .6, 1024))
        reference = TinyObjective()
        expected = torch.optim.SGD(reference.parameters(), lr=.01)
        reference(batch, .35)[0].backward()
        torch.nn.utils.clip_grad_norm_(reference.parameters(), 1.)
        expected.step()
        caps = dict(runner.micro_caps(), TSP=170)
        for fail in (False, True):
            objective = TinyObjective(fail)
            optimizer = torch.optim.SGD(objective.parameters(), lr=.01)
            with patch.object(runner.private_engine(), '_r106_caps', caps), \
                    patch.object(optimizer, 'step', wraps=optimizer.step) as step:
                parts, retries, _ = runner.update_batch(objective, optimizer,
                    torch.amp.GradScaler('cpu', enabled=False), batch, .35,
                    Runtime(0, 1, torch.device('cpu')), 170, problem='TSP')
            self.assertEqual((step.call_count, retries), (1, int(fail)))
            torch.testing.assert_close(objective.weight, reference.weight, rtol=0, atol=3e-8)
            self.assertTrue(torch.isfinite(parts).all())

    def test_selected_capture_nullable_binding_and_test_idempotence(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = root / runner.ARMS[0]
            _, lock, val, final = capture_fixture(run)
            archive = Path(val['prediction_archive']).parent / 'identity.json'
            before = archive.read_bytes()
            bound = runner.bind_selected_val(run, final)
            self.assertEqual(bound['val']['identity']['checkpoint_sha256'], lock['checkpoint_sha256'])
            self.assertIsNone(bound['val']['archive_capture_identity']['checkpoint_sha256'])
            self.assertEqual(runner.bind_selected_val(run, bound), bound)
            broken = copy.deepcopy(final)
            broken['val']['identity']['successful_updates'] += 1440
            with self.assertRaises(ValueError):
                runner.bind_selected_val(run, broken)
            args = runner.parser().parse_args(['test', '--root', str(root), '--arm', runner.ARMS[0]])
            result = dict(identity=dict(val['identity'], split='test', checkpoint_sha256=lock['checkpoint_sha256']),
                per_problem={p: dict(n=4500) for p in runner.PROBLEMS}, macro=dict(macro_vs_sbs_pct=0.))
            write_json(run / 'final_metrics.json', bound)
            write_json(run / 'test_results.json', dict(complete=True, checkpoint_sha256=lock['checkpoint_sha256'],
                state_hash=lock['state_hash'], result=result))
            config = dict(cache_root=str(args.cache_root), prepared_root=str(args.prepared_root),
                gpu_type='4090', eval_batch_size=128, microbatch_caps=runner.micro_caps())
            with patch.object(runner, 'authorize_test', return_value=(lock, config)), \
                    patch.object(runner, 'load_checkpoint', side_effect=AssertionError('No new inference')), \
                    patch.object(runner, 'ensure_test_cache', side_effect=AssertionError('No cache generation')), \
                    patch.object(runner, 'evaluate', side_effect=AssertionError('No reselection/evaluation')):
                self.assertEqual(runner.test(args, torch.device('cpu')), result)
                self.assertEqual(runner.test(args, torch.device('cpu')), result)
            self.assertEqual(archive.read_bytes(), before)

    def test_input_only_cache_and_real_online_timing_adapter(self):
        from .test_r106_model import raw_batch
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            record = dict(problem='TSP', split='test', rows=4500, data_root=str(root / 'data'),
                kind='coord', problem_id=0, pool=['x'], pool_ids=[0], cbits=[0] * 6,
                problem_desc=[0] * 8, mask=[1], d_in=2)
            write_json(root / 'test/TSP/metadata.json', record)
            config = dict(cache_root=str(root), data_root=record['data_root'],
                candidate_pools=dict(TSP=['x']), candidate_pool_ids=dict(TSP=[0]))
            manifest = dict(splits=dict(test=dict(TSP=record)))
            arrays = dict(n=np.full(128, 3, dtype=np.int64), coord_dist=np.zeros(128, dtype=np.int64),
                node=np.zeros((128, 3, 2), dtype=np.float32), node_geom=np.zeros((128, 3, 12), dtype=np.float32))
            specs = {k: {} for k in (*arrays, 'costs', 'winner')}
            with patch('code.V4.r76_cache._validate_record', return_value=specs), \
                    patch('code.V4.r75_cache._open_arrays', return_value=arrays) as opened, \
                    patch('code.V4.r75_cache._close'):
                batch = runner.cached_input_batch(config, manifest, 'test', 'TSP', torch.device('cpu'))
            self.assertEqual(set(opened.call_args.args[1]), set(arrays))
            self.assertFalse({'costs', 'ind', 'winner', 'cmp_target', 'nco_nodes'} & batch.keys())
        model = self.models[runner.ARMS[1]]
        before = parameter_hash(model.state_dict())
        samples = runner.measure_inputs(model, raw_batch('TSP', lengths=(4, 5), slots=6), torch.device('cpu'))
        self.assertEqual(samples['donor_calls'], [1, 1, 1])
        self.assertAlmostEqual(samples['mean_seconds_per_instance'], sum(samples['batch_seconds']) / 6)
        self.assertEqual(parameter_hash(model.state_dict()), before)


if __name__ == '__main__':
    unittest.main()

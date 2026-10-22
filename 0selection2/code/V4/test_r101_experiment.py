"""Focused synthetic CPU witnesses, no real caches, teacher files or GPU work."""

import copy
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import torch

from code.unified_selector.registry import PROBLEMS
from . import r101_experiment as runner
from . import r101_pretrain as pretrainer
from .r76_experiment import read_json, write_json
from .r77_data import parameter_hash
from .test_r76 import fixture_batch
from .test_r80 import fresh_fixture


SYNTHETIC_SPEC = runner.model_spec()


def arguments(directory, arm='A'):
    root = Path(directory)
    return runner.parser().parse_args(['train', '--arm', arm, '--device', 'cpu',
        '--root', str(root / 'run'), '--prepared-root', str(root / 'prepared'),
        '--data-root', str(root / 'data'), '--cache-root', str(root / 'cache')])


def install_lock(args):
    value = dict(runner.required_lock_fields(), data_root=str(args.data_root),
        prepared_root=str(args.prepared_root), source_provenance='synthetic CPU test')
    write_json(args.root / 'protocol_lock.json', value)
    return value


def teacher_identity(arm):
    covered = pretrainer.COVERED
    donor = dict(variant=runner.VARIANTS[arm], output_dim=128,
        checkpoint_sha256=None if arm == 'A' else runner.TEACHER_CHECKPOINT_SHA256,
        random_seed=940002 if arm == 'A' else None)
    return dict(variant=runner.VARIANTS[arm], split='train', rows_per_task=81000,
        metadata_sha256=dict.fromkeys(covered, 'd' * 64),
        donor={p: copy.deepcopy(donor) for p in covered}, no_teacher_in_final_model=True,
        interface_coverage_not_pretraining_task_coverage=True)


def endpoint(initial, geometry, arm='A', protocol_sha256='b' * 64):
    model = pretrainer.initialize_fresh(initial, geometry)
    encoder = {k: v.clone() for k, v in model.instance_encoder.state_dict().items()}
    name = next(iter(dict(model.instance_encoder.named_parameters())))
    encoder[name].add_(.001 if arm == 'A' else .002)
    return dict(experiment='R101', arm=arm, successful_updates=12800, complete=True,
        instance_encoder=encoder, encoder_sha256=parameter_hash(encoder),
        initial_trainable_sha256=initial['trainable_sha256'],
        unchanged_non_encoder_sha256=pretrainer.non_encoder_hash(model),
        teacher_identity=teacher_identity(arm), pretrain_spec=copy.deepcopy(pretrainer.SPEC),
        protocol_sha256=protocol_sha256)


def synthetic_config(directory, arm='A', initial=None, geometry=None):
    args = arguments(directory, arm)
    install_lock(args)
    write_json(args.prepared_root / 'prepared.json', dict(synthetic=True))
    if initial is None:
        initial, geometry = fresh_fixture()
    state = endpoint(initial, geometry, arm, runner.file_hash(args.root / 'protocol_lock.json'))
    lineage = {k: copy.deepcopy(v) for k, v in state.items() if k != 'instance_encoder'}
    lineage['checkpoint_sha256'] = ('a' if arm == 'A' else 'c') * 64
    ready = dict(trainable_sha256=initial['trainable_sha256'])
    manifest = dict(splits=dict(train={p: dict(pool=['m0', 'm1'], pool_ids=[0, 1]) for p in PROBLEMS}))
    baselines = {p: dict(selection_split='train', sbs_solver='m0') for p in PROBLEMS}
    count = sum(t.numel() for t in initial['trainable'].values())
    with patch.object(runner, 'PARAMETER_COUNT', count), \
            patch.object(runner, 'model_spec', return_value=copy.deepcopy(SYNTHETIC_SPEC)):
        model, params, receipt = runner.initialize_shared(initial, geometry, arm, pretrain_state=state)
        config = runner.protocol(args, ready, manifest, geometry, baselines, params,
            runner.task_order(runner.MAX_UPDATES), runner.micro_caps('4090', 352), lineage=lineage)
    return args, config, model, receipt, state


class R101RunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        guard = patch('torch.cuda._lazy_init', side_effect=AssertionError('CPU tests must not initialize CUDA'))
        guard.start()
        self.addCleanup(guard.stop)
        schema = patch.object(runner, 'model_spec', return_value=copy.deepcopy(SYNTHETIC_SPEC))
        schema.start()
        self.addCleanup(schema.stop)
        self.initial, self.geometry = fresh_fixture()
        self.count = sum(t.numel() for t in self.initial['trainable'].values())
        count = patch.object(runner, 'PARAMETER_COUNT', self.count)
        count.start()
        self.addCleanup(count.stop)

    def test_cli_and_main_lock_no_checkpoint_reads(self):
        with patch.object(torch, 'load', side_effect=AssertionError('no weights')):
            with redirect_stdout(io.StringIO()):
                description = runner.main(['describe', '--arm', 'A', '--device', 'cpu'])
                runner.main(['status', '--arm', 'B', '--device', 'cpu', '--root', '/nonexistent/r101-cpu'])
        self.assertFalse(description['deployment_teacher'])
        for command in ('pretrain', 'train', 'test', 'timing', 'describe', 'status'):
            args = runner.parser().parse_args([command, '--arm', 'B'])
            self.assertEqual((args.large_micro_cap, args.eval_batch_size, args.wandb_mode, args.gpu_type),
                             (352, 256, 'offline', '4090'))
        with tempfile.TemporaryDirectory() as tmp:
            args = arguments(tmp)
            record = install_lock(args)
            self.assertEqual(runner.protocol_lock(args)['record'], record)
            self.assertEqual(record['hint_pretraining'], pretrainer.SPEC)
            for key, value in (('pretrain_updates', 1), ('pretrain_warmup', 1), ('max_updates', 1),
                    ('train_eval_every', 1440), ('large_micro_cap', 384), ('data_root', '/different')):
                write_json(args.root / 'protocol_lock.json', dict(record, **{key: value}))
                with self.subTest(key=key), self.assertRaises(ValueError):
                    runner.protocol_lock(args)

    def test_transfer_only_instance_encoder_and_original_role_geometry(self):
        for arm in runner.ARMS:
            state = endpoint(self.initial, self.geometry, arm)
            model, params, receipt = runner.initialize_shared(self.initial, self.geometry, arm, pretrain_state=state)
            self.assertEqual(set(dict(model.named_parameters())), set(self.initial['trainable']))
            self.assertEqual(sum(p.numel() for p in model.parameters()), self.count)
            self.assertEqual(pretrainer.non_encoder_hash(model), state['unchanged_non_encoder_sha256'])
            self.assertEqual(parameter_hash(model.instance_encoder.state_dict()), state['encoder_sha256'])
            for name, tensor in model.named_parameters():
                if not name.startswith('instance_encoder.'):
                    torch.testing.assert_close(tensor, self.initial['trainable'][name], rtol=0, atol=0)
            self.assertEqual(params['role_arm'], 'B_execution_anchor')
            self.assertEqual(params['r101_arm'], arm)
            self.assertTrue(all(receipt[k] for k in ('optimizer_reset', 'scaler_reset', 'rng_reset', 'task_sample_streams_reset')))

    def test_transfer_rejects_stale_lineage_changed_nonencoder_or_geometry(self):
        good = endpoint(self.initial, self.geometry)
        for key, value in (('experiment', 'R61'), ('successful_updates', 12799), ('arm', 'B'),
                ('initial_trainable_sha256', 'c' * 64), ('unchanged_non_encoder_sha256', 'c' * 64)):
            with self.subTest(key=key), self.assertRaises(ValueError):
                runner.initialize_shared(self.initial, self.geometry, 'A', pretrain_state=dict(good, **{key: value}))
        bad = copy.deepcopy(good)
        key = 'geometry_residual.mean'
        bad['instance_encoder'][key].add_(1.)
        bad['encoder_sha256'] = parameter_hash(bad['instance_encoder'])
        with self.assertRaises(ValueError):
            runner.initialize_shared(self.initial, self.geometry, 'A', pretrain_state=bad)

    def test_teacher_train_scope_pin_and_no_outcome_lineage(self):
        for arm in runner.ARMS:
            identity = teacher_identity(arm)
            runner.validate_teacher_identity(identity, arm)
            identity['split'] = 'val'
            with self.assertRaises(ValueError):
                runner.validate_teacher_identity(identity, arm)
        identity = teacher_identity('B')
        identity['donor']['CVRP']['checkpoint_sha256'] = 'e' * 64
        with self.assertRaises(ValueError):
            runner.validate_teacher_identity(identity, 'B')

    def test_pretrain_delegation_and_stageII_closure_never_use_global_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = arguments(tmp)
            args.teacher_root = Path(tmp) / 'teacher'
            install_lock(args)
            with patch.object(runner, 'prepared_config'), patch.object(pretrainer, 'pretrain', return_value='done') as call:
                self.assertEqual(runner.pretrain(args, torch.device('cpu')), 'done')
                call.assert_called_once_with(args, torch.device('cpu'))
            state = endpoint(self.initial, self.geometry)
            lineage = dict(synthetic=True)
            def train_call(received, device, *, integration):
                self.assertIs(received, args)
                self.assertIsNot(integration, runner)
                initialized = integration.initialize_shared(self.initial, self.geometry, 'A', device)
                self.assertEqual(parameter_hash(initialized[0].instance_encoder.state_dict()), state['encoder_sha256'])
                return 'trained'
            with patch.object(runner, 'read_pretraining', return_value=(state, lineage)), \
                    patch.object(runner.engine, 'train', side_effect=train_call):
                self.assertEqual(runner.train(args, torch.device('cpu')), 'trained')
            self.assertFalse(hasattr(runner, 'root'))

    def test_pretrain_endpoint_receipt_hash_and_no_teacher_read_at_transfer(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = arguments(tmp)
            install_lock(args)
            state = endpoint(self.initial, self.geometry, protocol_sha256=runner.protocol_lock(args)['sha256'])
            path = args.root / 'A/pretrain_end.pt'
            path.parent.mkdir(parents=True)
            torch.save(state, path)
            receipt = {k: v for k, v in state.items() if k != 'instance_encoder'}
            receipt['checkpoint_sha256'] = runner.file_hash(path)
            write_json(path.parent / 'pretrain_complete.json', receipt)
            loaded, lineage = runner.read_pretraining(args)
            self.assertIsNone(args.teacher_root)
            self.assertEqual(loaded['encoder_sha256'], lineage['encoder_sha256'])
            receipt['checkpoint_sha256'] = 'e' * 64
            write_json(path.parent / 'pretrain_complete.json', receipt)
            with self.assertRaises(ValueError):
                runner.read_pretraining(args)

    def test_final_factory_self_contained_without_teacher_or_pretraining_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            args, config, model, _, _ = synthetic_config(tmp)
            path = args.root / 'A/final.pt'
            path.parent.mkdir(parents=True)
            state = dict(args=config, state_hash=runner.json_hash(config), model=model.state_dict())
            torch.save(state, path)
            original_load = torch.load
            def only_final(source, **kwargs):
                self.assertEqual(Path(source), path)
                return original_load(source, **kwargs)
            with patch.object(torch, 'load', side_effect=only_final), \
                    patch.object(runner, 'read_pretraining', side_effect=AssertionError('no endpoint')), \
                    patch.object(runner, 'read_json', side_effect=AssertionError('no teacher/protocol files')):
                restored, _ = runner.load_checkpoint(path)
            self.assertEqual(parameter_hash(restored.state_dict()), parameter_hash(model.state_dict()))
            for field, value in (('downstream_hint_loss', True), ('downstream_teacher_inputs', True),
                    ('deployment_teacher', True), ('train_eval_every_updates', 1), ('rdrop_weight', 0.),
                    ('checkpoint_rule', 'TEST best'), ('pretrained_encoder_sha256', 'e' * 64)):
                with self.subTest(field=field), self.assertRaises(ValueError):
                    runner.validate_checkpoint_config(dict(config, **{field: value}))
            bad = copy.deepcopy(config)
            bad['model_params']['dropout'] = .2
            with self.assertRaises(ValueError):
                runner.validate_checkpoint_config(bad)

    def test_original_objective_microcaps_and_input_only_student_timing(self):
        state = endpoint(self.initial, self.geometry)
        model, _, _ = runner.initialize_shared(self.initial, self.geometry, 'A', pretrain_state=state)
        self.assertIs(type(runner.make_objective(model, 'A')), runner.RDropObjective)
        self.assertEqual(runner.checkpoint_extras(None), {})
        with self.assertRaises(ValueError):
            runner.restore_objective(None, dict(objective_state={}))
        batch = dict(ind=torch.zeros(1024, dtype=torch.long))
        with patch.object(runner.engine, 'update_batch', return_value=(1, 0, {})) as step:
            self.assertEqual(runner.update_batch(None, None, None, batch, .35, None, 352, problem='CVRP'), (1, 0, {}))
            for bad, cap in ((dict(batch, teacher_nodes=torch.zeros(1)), 352), (batch, 1024)):
                with self.assertRaises(ValueError):
                    runner.update_batch(None, None, None, bad, .35, None, cap, problem='CVRP')
            self.assertEqual(step.call_count, 1)
        for problem in ('TSP', 'CVRP', 'ATSP'):
            batch, _ = fixture_batch(problem, 2)
            if problem == 'TSP':
                batch['node'] = batch['node'][..., :2]
            batch.update(teacher_nodes=torch.full((2,), float('nan')), winner='forbidden')
            keys = []
            handle = model.register_forward_pre_hook(lambda module, inputs: keys.append(set(inputs[0])))
            try:
                values = runner.measure_inputs(model.eval(), batch, torch.device('cpu'))
            finally:
                handle.remove()
            self.assertEqual(len(values['batch_seconds']), 3)
            self.assertTrue(all(not (k & {'teacher_nodes', 'winner', 'costs', 'ind'}) for k in keys))
            if problem == 'ATSP':
                self.assertTrue(all('matrix' in k for k in keys))


if __name__ == '__main__':
    unittest.main()

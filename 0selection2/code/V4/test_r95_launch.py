"""Focused CPU R95 launch contracts; no Slurm, container, GPU or Git execution."""

import argparse
from contextlib import ExitStack, nullcontext
import copy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from . import r93_launch, r94_launch
from . import r95_launch as launch
from . import r95_reporting as reporting


def protocol_fixture():
    return dict(experiment='R95', locked=True, arms=['A', 'B'], seed=2,
        task_scope='full18', batch_size=1024, logical_global_batch=1024,
        max_updates=57600, warmup_updates=4320, eval_every=1440,
        arm_budgets=dict(A=57600, B=57600), large_micro_cap=352, eval_batch_size=256,
        dropout=dict(A=.1, B=.1), sampling=copy.deepcopy(launch.SAMPLING),
        rows_per_problem=dict(train=81000, val=4500, test=4500),
        data_root=str(launch.DATA), prepared_root=str(launch.PREPARED), wandb_mode='offline')


def runner_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('train', 'test', 'describe', 'status'))
    for name in ('root', 'prepared-root', 'data-root', 'cache-root', 'protocol-lock'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--arm', choices=('A', 'B'), required=True)
    parser.add_argument('--device', required=True)
    parser.add_argument('--gpu-type', choices=('3090', '4090'), required=True)
    parser.add_argument('--large-micro-cap', type=int, required=True)
    parser.add_argument('--eval-batch-size', type=int, required=True)
    parser.add_argument('--wandb-mode', choices=('offline',), required=True)
    parser.add_argument('--resume', action='store_true')
    return parser


class LaunchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        for name in ('run', 'Popen', 'check_output'):
            self.stack.enter_context(patch.object(launch.base.subprocess, name,
                side_effect=AssertionError('No external command in focused CPU tests')))
        self.real_sources = launch._sources
        self.stack.enter_context(patch.object(launch, '_sources', return_value=[str(Path(launch.__file__))]))

    def patch(self, target, name, **kwargs):
        return self.stack.enter_context(patch.object(target, name, **kwargs))

    def test_fixed_allocations_and_private_adapters_leave_r94_unchanged(self):
        self.assertEqual(launch.ARM_SLOTS, dict(A='3090_0', B='4090_0'))
        self.assertEqual(set(launch.SLOTS), {'3090_0', '4090_0'})
        self.assertEqual(launch.SLOTS['3090_0']['job'], 1465)
        self.assertEqual(launch.SLOTS['4090_0']['job'], 1475)
        self.assertEqual(launch.PREDECESSORS, dict(A='A', B='B'))
        self.assertEqual(r93_launch.ARM_SLOTS, dict(A='3090_1', B='4090_1'))
        self.assertEqual(r94_launch.ARM_SLOTS, dict(A='3090_1', B='4090_1'))
        self.assertIsNot(launch.base, r94_launch.base)
        self.assertIsNot(launch.base, r93_launch.base)
        self.assertEqual(launch.MODEL_ROOT.name, 'R95_adaptive_example_sampling')

    def test_templates_are_fresh_original_loss_gpu_pipeline_and_cpu_pdf(self):
        for arm in launch.ARMS:
            spec = launch.template(arm)
            stages = launch.commands(spec, launch.ARM_SLOTS[arm])
            self.assertEqual([r['name'] for r in stages], ['train', 'TEST', 'LIB', 'PDF'])
            self.assertEqual([r['device'] for r in stages], ['gpu', 'gpu', 'gpu', 'cpu'])
            self.assertEqual((spec['cpu_count'], spec['memory_gib']), (4, 9))
            self.assertEqual(spec['protocol']['dropout'], .1)
            self.assertEqual(spec['protocol']['sampling'], launch.SAMPLING)
            self.assertIn('original_R77', spec['protocol']['objective'])
            self.assertEqual(spec['cache_root'], str(launch.CACHE_ROOT))
            self.assertFalse(any('--resume' in r['argv'] for r in stages))
            self.assertNotIn('prepare', [r['argv'][5] for r in stages])
            self.assertEqual(stages[0]['argv'][4:6], ['code.V4.r95_experiment', 'train'])
            self.assertEqual(stages[-1]['argv'][-1], 'cpu')
        with self.assertRaises(ValueError):
            launch.commands(launch.template('A'), '3090_1')

    def test_exact_main_lock_and_sampling_fields(self):
        path = self.root / 'protocol_lock.json'
        self.patch(launch, 'PROTOCOL_LOCK', new=path)
        value = protocol_fixture()
        launch.base.save(path, value)
        self.assertEqual(launch.locked_protocol(), value)
        for key, invalid in (('locked', False), ('max_updates', 50000), ('dropout', dict(A=.1, B=0)),
                ('wandb_mode', 'online'), ('global_batch_size', 512), ('large_micro_cap', 176),
                ('eval_every', 7200), ('arm_budgets', dict(A=57600, B=50000))):
            launch.base.save(path, dict(value, **{key: invalid}))
            with self.subTest(key=key), self.assertRaises(ValueError):
                launch.locked_protocol()
        for key in launch.SAMPLING:
            changed = copy.deepcopy(value)
            changed['sampling'][key] = 'different'
            launch.base.save(path, changed)
            with self.subTest(key=key), self.assertRaises(ValueError):
                launch.locked_protocol()

    def test_parser_contract_is_read_only_and_no_resume(self):
        runner = SimpleNamespace(ARMS=launch.ARMS, parser=runner_parser)
        self.patch(launch, 'runtime_modules', return_value=(runner, reporting))
        self.patch(launch.base, 'save', side_effect=AssertionError('No parser writes'))
        result = launch.parse_commands(protocol_fixture())
        self.assertEqual(result['parsed_count'], 9)
        self.assertEqual([r['command'] for r in result['commands']],
            ['train', 'test', 'lib', 'report'] * 2 + ['compare'])
        self.assertFalse(result['training_started'])
        self.assertFalse(result['resource_probe_performed'])
        parser = runner_parser()
        parser.set_defaults(resume=True)
        runner.parser = lambda: parser
        with self.assertRaisesRegex(RuntimeError, 'resume'):
            launch.parse_commands(protocol_fixture())

    def test_readiness_confirmation_precedes_probes_or_writes(self):
        parse = self.patch(launch, 'parse_commands')
        save = self.patch(launch.base, 'save')
        for evidence, tested in (('', True), (' ', True), ('final-tests', False)):
            for action in (launch.prepare, launch.start, launch.write_specs, launch.queue_prepared):
                with self.subTest(action=action.__name__), self.assertRaises(RuntimeError):
                    action(self.root / 'spec.json', evidence, tested) if action != launch.write_specs else action(evidence, tested)
        parse.assert_not_called()
        save.assert_not_called()
        with self.assertRaises(RuntimeError):
            launch.adapter.authorize_resources(None, '3090_0')

    def predecessor_fixture(self):
        self.patch(launch, 'R92_MODELS', new=self.root / 'models')
        self.patch(launch, 'R92_LAUNCH', new=self.root / 'launch')
        for arm in launch.ARMS:
            run, directory = launch.R92_MODELS / arm, launch.R92_LAUNCH / arm
            run.mkdir(parents=True)
            (run / 'best.pt').write_bytes(b'unchanged synthetic checkpoint')
            lock = dict(experiment='R92', arm=arm, checkpoint=str(run / 'best.pt'),
                checkpoint_sha256=launch.base.sha(run / 'best.pt'), state_hash='same',
                selected_without_test=True, per_problem_assembly=False)
            launch.base.save(run / 'final_test_lock.json', lock)
            launch.base.save(run / 'training_complete.json', dict(successful_updates=57600,
                selected_best=lock, state_hash='same'))
            launch.base.save(run / 'test_results.json', dict(complete=True, checkpoint_sha256=lock['checkpoint_sha256']))
            launch.base.save(run / 'lib_stage_complete.json', dict(complete=True,
                checkpoint_sha256=lock['checkpoint_sha256'], datasets=['TSPLIB', 'CVRPLIB']))
            post = dict(phase='complete', complete=True)
            for key, name in (('model_statistics_pdf', 'MODEL_STATISTICS.pdf'),
                    ('selection_distributions_pdf', 'selection_distributions.pdf'),
                    ('target_recovery_pdf', 'target_recovery_and_train.pdf')):
                path = run / 'full_model_statistics' / name
                path.parent.mkdir(exist_ok=True)
                path.write_bytes(b'%PDF-fixture')
                post[key] = str(path)
            launch.base.save(run / 'postprocess_status.json', post)
            for name, stages in (('worker', ('train', 'TEST', 'LIB')),
                    ('controller', ('allocation_GPU_pipeline', 'PDF'))):
                launch.base.save(directory / (name + '.json'), dict(state='complete', exit_status=0,
                    stages=[dict(name=stage, state='complete', exit_status=0) for stage in stages]))

    def test_predecessors_require_same_lock_test_lib_pdf_and_controller(self):
        self.predecessor_fixture()
        for arm in launch.ARMS:
            self.assertTrue(launch.predecessor_gate(arm)['complete'])
        launch.base.save(launch.R92_LAUNCH / 'A/controller.json', dict(state='reporting', exit_status=0))
        self.assertFalse(launch.predecessor_gate('A')['complete'])
        self.assertFalse(launch.predecessor_gate('A')['failed'])
        launch.base.save(launch.R92_LAUNCH / 'A/controller.json', dict(state='failed', exit_status=7))
        self.assertTrue(launch.predecessor_gate('A')['failed'])
        (launch.R92_MODELS / 'B/best.pt').write_bytes(b'changed')
        self.assertTrue(launch.predecessor_gate('B')['failed'])

    def test_queue_waits_then_starts_only_once_and_never_retries(self):
        self.patch(launch, 'queue_prepared', return_value=(dict(run_id='A'), {}))
        self.patch(launch.base, 'receipts', return_value=self.root)
        self.patch(launch.base, 'lock', return_value=nullcontext())
        self.patch(launch, 'predecessor_gate', side_effect=[
            dict(complete=False, failed=False), dict(complete=True, failed=False)])
        start = self.patch(launch, 'start', return_value=0)
        sleep = self.patch(launch.time, 'sleep')
        self.assertEqual(launch.queue(self.root / 'spec.json', 'final-tests', True), 0)
        start.assert_called_once()
        sleep.assert_called_once_with(60)
        launch.base.save(self.root / 'attempt.json', dict(exit_status=1))
        with self.assertRaises(RuntimeError):
            launch.queue(self.root / 'spec.json', 'final-tests', True)
        self.assertEqual(start.call_count, 1)

    def test_existing_cpu_step_container_uuid_and_shared_mutex_routing(self):
        for slot in launch.SLOTS:
            argv = launch.base.srun_command('entry', slot, self.root / 'spec.json', 'token', cpus=4, memory='9G')
            self.assertIn('--gres=none', argv)
            self.assertIn('--cpus-per-task=4', argv)
            self.assertIn('--mem=9G', argv)
            self.assertIn('--jobid=' + str(launch.SLOTS[slot]['job']), argv)
            container = launch.adapter.container_command('worker', slot, self.root / 'spec.json', 'token')
            self.assertIn('CUDA_VISIBLE_DEVICES=' + launch.SLOTS[slot]['uuid'], container)
            self.assertIn('WANDB_MODE=offline', container)
            self.assertIn(str(launch.WRAPPER), container)
            self.assertIn('R95_LAUNCH_TOKEN=token', container)
        self.assertIs(launch.base.transport, launch.transport)

    def test_frozen_sources_include_dependencies_lock_wrappers_not_future_test_glob(self):
        paths = {Path(name) for name in self.real_sources(protocol_fixture())}
        for name in ('r95_model.py', 'r95_experiment.py', 'r95_objective.py', 'r95_sampling.py',
                'r95_launch.py', 'r95_reporting.py', 'r95_goal.py', 'r93_launch.py',
                'r84_experiment.py', 'r76_experiment.py', 'pairwise_selector.py'):
            self.assertIn(launch.REPO / 'code/V4' / name, paths)
        self.assertIn(launch.PROTOCOL_LOCK, paths)
        self.assertIn(launch.WRAPPER, paths)
        self.assertIn(launch.PREPARED / 'identicalinit_seed2.pt', paths)
        self.assertTrue(set(launch.TEST_SOURCES).issubset({p.name for p in paths}))
        self.assertFalse(any(p.suffix in ('.npy', '.npz', '.pkl') for p in paths))
        source = Path(launch.__file__).read_text()
        self.assertNotIn("glob('test_r95", source)


if __name__ == '__main__':
    unittest.main()

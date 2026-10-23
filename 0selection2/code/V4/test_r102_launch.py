"""Focused CPU fixtures. No probes, copies, queues or GPU workers are run."""

import copy
import json
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from . import r102_launch as launch
from . import r102_wave as wave


def spec_fixture(repo, arm='A'):
    model = repo / 'code/V4/runs/R102_full_type_pretrained_fusion'
    science = dict(experiment='R102', locked=True, seed=2, task_scope='full18', arms=['A', 'B'],
        data_root=str(launch.DATA_ROOT), prepared_root=str(launch.PREPARED_ROOT),
        rows_per_problem=dict(train=81000, val=4500, test=4500), logical_global_batch=1024,
        max_updates=57600, warmup_updates=4320, eval_every=1440)
    launch.base.save(model / 'protocol_lock.json', science)
    with patch.object(launch, 'MODEL_ROOT', model):
        spec = dict(schema_version=1, run_id='R102_' + arm, run_dir=str(model / arm), common_root=str(model),
            science_protocol=str(model / 'protocol_lock.json'), data_root=str(launch.DATA_ROOT),
            prepared_root=str(launch.PREPARED_ROOT), cache_source=str(launch.CACHE_SOURCE),
            cache_root=str(launch.CACHE_ROOT), cache_kind='r84_inputs', wandb_mode='offline',
            feature_root=str(launch.FEATURE_ROOTS[arm]), routing_feature_root=str(launch.ROUTING_ROOT),
            routing_mode='reuse_original_R94B', cpus=launch.SLOT_CPUS[launch.ARM_SLOTS[arm]],
            slots=[launch.ARM_SLOTS[arm]], cpu_count=4, memory_gib=9,
            stages=launch.expected_stages(arm), source_files=[])
    path = repo / ('R102_' + arm + '.json')
    launch.base.save(path, spec)
    return model, path, spec, science


class LaunchTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='r102-launch-cpu-')
        self.addCleanup(temporary.cleanup)
        self.repo = Path(temporary.name)

    def test_private_adapters_keep_original_mutexes_and_new_4090_bindings(self):
        from . import r98_launch, r84_launch, r81_queue, r101_launch
        self.assertIsNot(launch.adapter, r98_launch)
        self.assertIsNot(launch.base, r84_launch)
        self.assertIsNot(launch.base.transport, r81_queue)
        self.assertEqual(launch.base.transport.ROOT, r81_queue.ROOT)
        self.assertEqual(launch.ROOT, r98_launch.ROOT)
        self.assertEqual(launch.ARM_SLOTS, dict(A='4090_0', B='4090_1'))
        self.assertEqual(r101_launch.ARM_SLOTS, dict(A='4090_1', B='4090_0'))
        self.assertEqual(r98_launch.TOKEN_ENV, 'R98_DISPATCH_TOKEN')
        self.assertEqual(r84_launch.TOKEN_ENV, 'R84_LAUNCH_TOKEN')
        self.assertIs(launch.base.worker.__globals__['stage_cache'], launch.stage_cache)
        self.assertIs(launch.base.entry.__globals__['node_resources'], launch.node_resources)

    def test_container_and_allocation_commands_cannot_escape_original_gpu05(self):
        for slot in launch.SLOTS:
            item = launch.SLOTS[slot]
            self.assertEqual((item['job'], item['node'], item['gpu_type'], item['cap']), (1475, 'gpu05', '4090', 352))
            argv = launch.container_command('worker', slot, '/tmp/spec.json', 'token')
            self.assertIn('76045731f6b453443fe18a9bce671599bdbf17c2713b8edcdd090925eea14bf6', argv)
            self.assertIn(str(launch.WRAPPER), argv)
            self.assertNotIn('ssh', argv)
            step = launch.base.srun_command('entry', slot, '/tmp/spec.json', 'token', cpus=4, memory='9G')
            for value in ('--jobid=1475', '--nodelist=gpu05', '--cpus-per-task=4', '--mem=9G', '--gres=none'):
                self.assertIn(value, step)
            self.assertIn('code.V4.r102_launch', ' '.join(step))
            self.assertEqual(launch.environment(slot, 'token')['CUDA_VISIBLE_DEVICES'], item['uuid'])
            self.assertEqual(launch.environment()['WANDB_MODE'], 'offline')
        self.assertFalse(set(launch.SLOT_CPUS['4090_0']) & set(launch.SLOT_CPUS['4090_1']))
        with patch.object(launch, 'original_node_resources', return_value=dict(allowed_cpus=[0, 1, 2, 3, 4, 5, 72, 73])):
            self.assertEqual(launch.node_resources('4090_1')['allowed_cpus'], [4, 5, 72, 73])
        with patch.object(launch, 'original_node_resources', return_value=dict(allowed_cpus=[0, 1, 2, 3])):
            with self.assertRaisesRegex(RuntimeError, 'affinity'):
                launch.node_resources('4090_1')

    def test_exact_six_stages_direct_B_routing_and_reject_obsolete_staging(self):
        model, path, original, science = spec_fixture(self.repo)
        with patch.object(launch, 'MODEL_ROOT', model):
            spec = launch.load_spec(path)
            commands = launch.base.commands(spec, '4090_0')
            self.assertEqual(tuple(s['name'] for s in commands), launch.STAGES)
            for stage in commands[:4]:
                args = stage['argv']
                self.assertEqual(args[args.index('--gpu-type') + 1], '4090')
                self.assertEqual(args[args.index('--routing-feature-root') + 1], '/tmp/shiys_R94_B_node_features')
            self.assertEqual(commands[-2]['argv'][-2:], ['--lib-batch-size', '2'])
            self.assertEqual(commands[-1]['argv'][-2:], ['--device', 'cpu'])
            for change in (dict(slots=['3090_0']), dict(routing_stage_json='/tmp/obsolete.json'),
                           dict(routing_feature_root='/tmp/shiys_R102_shared_vrp_B'), dict(cpus=[4, 5, 72, 73])):
                launch.base.save(path, dict(original, **change))
                with self.assertRaises(ValueError):
                    launch.load_spec(path)
            launch.base.save(path, original)
            launch.base.save(model / 'protocol_lock.json', dict(science, logical_global_batch=512))
            with self.assertRaises(ValueError):
                launch.load_spec(path)

    def test_main_evidence_and_source_hash_gate_no_fixed_test_count(self):
        for evidence, tested in ((None, True), ('', True), ('CPU evidence', False)):
            with self.assertRaises(ValueError):
                launch.prepare('/absent', evidence, tested)
            with self.assertRaises(ValueError):
                launch.start('/absent', evidence, tested)
        model, path, spec, _ = spec_fixture(self.repo)
        dispatch = self.repo / 'dispatch'
        output = dispatch / spec['run_id']
        source = self.repo / 'tiny.py'
        source.write_text('value = 1\n')
        launch.base.save(output / 'plan.json', dict(spec=spec, source_sha256={str(source): launch.base.sha(source)}))
        launch.base.save(output / 'ready.json', dict(main_thread_ready=True, runner_tests_passed=True,
            evidence='main chooses evidence', plan_sha256=launch.base.sha(output / 'plan.json')))
        with patch.object(launch, 'MODEL_ROOT', model), patch.object(launch.base, 'ROOT', dispatch):
            launch.base.check_ready(path)
            source.write_text('value = 2\n')
            with self.assertRaisesRegex(RuntimeError, 'source changed'):
                launch.base.check_ready(path)

    def test_all_native_sources_are_required_and_native_edit_invalidates_ready(self):
        model, path, spec, _ = spec_fixture(self.repo)
        (model / 'PLAN.md').write_text('# Tiny plan\n')
        preparation = model / 'preparation'
        preparation.mkdir()
        (preparation / 'LAUNCH_PLAN.md').write_text('# Tiny launch plan\n')
        native_root = self.repo / 'external_native'
        native = [native_root / relative for relative in (
            'ELG/models.py', 'MatNet/matnet_encoder.py', 'Transformer/attr_component.py',
            'ReLD/models/MOEModel_Light.py', 'ReLD/models/MOELayer.py')]
        for source in native:
            source.parent.mkdir(parents=True, exist_ok=True)
            source.write_text('VERSION = 1\n')
        donor = SimpleNamespace(ELG_SOURCE=native[0], MATNET_SOURCE=native[1], MATNET_BACKBONE=native[2],
            reld=SimpleNamespace(RELD_ROOT=native_root / 'ReLD'))
        dispatch = self.repo / 'dispatch'
        output = dispatch / spec['run_id']
        with patch.dict(sys.modules, {'code.V4.r102_donor': donor}), \
                patch.object(launch, 'MODEL_ROOT', model), patch.object(launch.base, 'ROOT', dispatch):
            sources = launch.source_files(spec)
            self.assertTrue({str(p.resolve()) for p in native} <= set(sources))
            launch.base.save(output / 'plan.json', dict(spec=spec,
                source_sha256={name: launch.base.sha(name) for name in sources}))
            launch.base.save(output / 'ready.json', dict(main_thread_ready=True, runner_tests_passed=True,
                evidence='tiny native source fixture', plan_sha256=launch.base.sha(output / 'plan.json')))
            launch.base.check_ready(path)
            native[0].write_text('VERSION = 2\n')
            with self.assertRaisesRegex(RuntimeError, 'source changed'):
                launch.base.check_ready(path)
            native[0].write_text('VERSION = 1\n')
            for source in native:
                with self.subTest(required=source.name):
                    source.unlink()
                    with self.assertRaisesRegex(RuntimeError, 'All five small native donor Python sources are required'):
                        launch.source_files(spec)
                    source.write_text('VERSION = 1\n')
            launch.base.check_ready(path)

    def test_predecessors_are_independent_full_R101_pipelines_not_R99(self):
        from .test_r101_goal import fixture
        model, dispatch, _ = fixture(self.repo)
        for prior_arm, slot in (('A', '4090_1'), ('B', '4090_0')):
            path = dispatch / 'specs' / ('R101_' + prior_arm + '.json')
            launch.base.save(path, dict(launch.base.read(path), slots=[slot]))
        with patch.object(launch, 'REPO', self.repo), patch.object(launch, 'ROOT', dispatch):
            self.assertTrue(launch.predecessor_gate('A')['complete'])
            self.assertTrue(launch.predecessor_gate('B')['complete'])
            failed = dispatch / 'R101_A/worker.json'
            record = launch.base.read(failed)
            launch.base.save(failed, dict(record, state='failed', exit_status=1))
            self.assertTrue(launch.predecessor_gate('A')['complete'])
            self.assertFalse(launch.predecessor_gate('B')['complete'])
            launch.base.save(failed, record)
            (model / 'B/full_model_statistics/learning_curves.pdf').unlink()
            self.assertFalse(launch.predecessor_gate('A')['complete'])
            self.assertTrue(launch.predecessor_gate('B')['complete'])
            self.assertFalse((self.repo / 'code/V4/runs/R99_global_competition_input').exists())

    def test_queue_waits_once_and_never_retries_terminal_failure(self):
        _, _, spec, _ = spec_fixture(self.repo)
        output = self.repo / 'queue'
        output.mkdir()
        with patch.object(launch.base, 'check_ready', return_value=(spec, dict(evidence='ready'))), \
                patch.object(launch.base, 'receipts', return_value=output), \
                patch.object(launch, 'predecessor_gate', side_effect=[dict(complete=False, errors=[]), dict(complete=True, errors=[])]), \
                patch.object(launch.time, 'sleep') as sleep, \
                patch.object(launch, 'start', side_effect=RuntimeError('terminal failure')) as start:
            with self.assertRaisesRegex(RuntimeError, 'terminal failure'):
                launch.queue('/spec', 'ready', True, interval=1)
            saved = (output / 'queue_receipt.json').read_bytes()
            with self.assertRaisesRegex(RuntimeError, 'already attempted'):
                launch.queue('/spec', 'ready', True)
            self.assertEqual((output / 'queue_receipt.json').read_bytes(), saved)
        self.assertEqual(start.call_count, 1)
        self.assertEqual(sleep.call_count, 1)

    def test_primary_cache_has_no_copy_or_rebuild_fallback(self):
        with patch.object(launch.adapter, 'original_stage_cache') as copy_cache, \
                patch.object(launch, 'routing_manifest') as routing:
            with self.assertRaisesRegex(RuntimeError, 'no rebuild or fresh copy'):
                launch.stage_cache(dict(cache_root=str(self.repo / 'missing')), dict(identity_sha256='f'*64))
        copy_cache.assert_not_called()
        routing.assert_not_called()

    def test_direct_routing_manifest_reads_no_payload_or_TEST(self):
        root, cache = self.repo / 'features', self.repo / 'inputs'
        donor = dict(variant='pretrained', checkpoint_sha256=launch.RELD_SHA256, output_dim=128,
                     supported_problems=list(launch.ROUTING_PROBLEMS))
        for split, rows in (('train', 81000), ('val', 4500)):
            for problem in launch.ROUTING_PROBLEMS:
                original = dict(complete=True, rows=rows, true_length_max=2, problem=problem,
                               split=split, data_root=str(launch.DATA_ROOT))
                launch.base.save(cache / split / problem / 'metadata.json', original)
                launch.base.save(root / split / problem / 'metadata.json', dict(format='r94_frozen_nodes_v1',
                    complete=True, input_only=True, rows=rows, rows_written=rows, shape=[rows, 2, 128],
                    split=split, problem=problem, dtype='float16', file='nco_nodes.npy', donor=donor,
                    source_cache=dict(metadata_sha256=launch.predecessor_goal.json_hash(original))))
                (root / split / problem / 'nco_nodes.npy').write_bytes(b'tiny-stat-only-fixture')
        original_open, original_stat = Path.open, Path.stat

        def no_arrays(path, *args, **kwargs):
            if path.suffix == '.npy' or 'test' in path.parts:
                raise AssertionError('No array or TEST opens')
            return original_open(path, *args, **kwargs)

        def payload_stat(path, *args, **kwargs):
            if path.suffix == '.npy':
                return SimpleNamespace(st_mode=stat.S_IFREG | 0o644, st_size=2**40)
            return original_stat(path, *args, **kwargs)

        with patch.object(Path, 'open', no_arrays), patch.object(Path, 'stat', payload_stat):
            result = launch.routing_manifest(root, cache)
            self.assertFalse(result['payload_opened'])
            self.assertFalse(result['TEST_opened'])
            self.assertFalse(result['copy'])
        wrong = root / 'val/CVRP/metadata.json'
        record = launch.base.read(wrong)
        launch.base.save(wrong, dict(record, donor=dict(donor, variant='random')))
        with patch.object(Path, 'stat', payload_stat), self.assertRaisesRegex(ValueError, 'pretrained R94B'):
            launch.routing_manifest(root, cache)

    def test_wave_preserves_existing_spec_and_never_launches(self):
        _, _, spec, _ = spec_fixture(self.repo)
        with patch.object(wave, 'make_spec', return_value=spec), patch.object(launch, 'ROOT', self.repo / 'dispatch'), \
                patch.object(launch, 'start') as start, patch.object(launch, 'queue') as queue:
            path = wave.write_spec('A')
            self.assertEqual(wave.write_spec('A'), path)
            modified = copy.deepcopy(spec)
            modified['slots'] = ['3090_0']
            with patch.object(wave, 'make_spec', return_value=modified), self.assertRaisesRegex(RuntimeError, 'Preserve'):
                wave.write_spec('A')
        start.assert_not_called()
        queue.assert_not_called()

    def test_current_runner_and_report_parsers_accept_exact_cli_without_execution(self):
        _, _, spec, _ = spec_fixture(self.repo)
        commands = launch.base.commands(spec, '4090_0')
        script = '''
import importlib,json,sys
stages = json.loads(sys.argv[1])
for stage in stages:
    argv = stage['argv']
    module = importlib.import_module(argv[4])
    args = module.parser().parse_args(argv[5:])
    assert not getattr(args, 'resume', False)
print('six exact stage parsers accepted; no stage invoked')
'''
        result = subprocess.run([sys.executable, '-B', '-c', script, json.dumps(commands)],
            cwd=launch.REPO, env=launch.environment(), text=True, capture_output=True, timeout=45)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('no stage invoked', result.stdout)


if __name__ == '__main__':
    unittest.main()

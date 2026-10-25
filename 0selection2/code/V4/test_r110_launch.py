"""Bounded CPU administrative contracts; no real specs, hardware or launches."""

from contextlib import redirect_stdout
import copy
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from . import r110_launch as launch
from . import r110_wave as wave


class LaunchTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix='r110-admin-fake-')
        self.addCleanup(temp.cleanup)
        self.temp = Path(temp.name)
        self.root = self.temp / 'R110_method_conditioned_ce'
        self.dispatch = self.root / 'preparation/dispatch'
        for name, value in (('MODEL_ROOT', self.root), ('ROOT', self.dispatch),
                            ('DATA_ROOT', self.temp / 'data'), ('PREPARED_ROOT', self.temp / 'prepared'),
                            ('CACHE_SOURCE', self.temp / 'cache_source'), ('CACHE_ROOT', self.temp / 'cache')):
            context = patch.object(launch, name, value)
            context.start()
            self.addCleanup(context.stop)
        context = patch.object(launch.base, 'ROOT', self.dispatch)
        context.start()
        self.addCleanup(context.stop)

    def locked_spec(self, arm=None):
        science = dict(launch.runner_contract().required_lock_fields(),
            data_root=str(launch.DATA_ROOT), prepared_root=str(launch.PREPARED_ROOT),
            primary_cache=str(launch.CACHE_ROOT))
        launch.base.save(self.root / 'protocol_lock.json', science)
        spec = wave.make_spec(arm or launch.ARMS[0], final=True)
        path = self.dispatch / 'specs' / (spec['run_id'] + '.json')
        launch.base.save(path, spec)
        return path, spec, science

    def test_exact_original_resources_and_private_transport(self):
        from . import r108_launch, r98_launch, r84_launch, r81_queue
        self.assertIsNot(launch.adapter, r98_launch)
        self.assertIsNot(launch.base, r84_launch)
        self.assertIsNot(launch.base.transport, r81_queue)
        self.assertEqual(launch.base.transport.ROOT, r81_queue.ROOT)
        self.assertEqual(r108_launch.TOKEN_ENV, 'R108_DISPATCH_TOKEN')
        self.assertEqual(set(launch.SLOTS), {'4090_0', '4090_1'})
        for slot in launch.SLOTS:
            for key in ('job', 'node', 'uuid'):
                self.assertEqual(launch.SLOTS[slot][key], r108_launch.SLOTS[slot][key])
            self.assertEqual((launch.SLOTS[slot]['job'], launch.SLOTS[slot]['node']), (1475, 'gpu05'))
            command = launch.container_command('worker', slot, '/tmp/fake-spec', 'fake-token')
            self.assertIn(r81_queue.CONTAINERS['gpu05'][1], command)
            self.assertIn(str(launch.WRAPPER), command)
            self.assertIn('R110_DISPATCH_TOKEN=fake-token', command)
            self.assertNotIn('ssh', command)
            step = launch.base.srun_command('entry', slot, '/tmp/fake-spec', cpus=4, memory='9G')
            for flag in ('--jobid=1475', '--nodelist=gpu05', '--gres=none', '--cpus-per-task=4', '--mem=9G'):
                self.assertIn(flag, step)
        self.assertEqual(launch.SLOT_CPUS, r108_launch.SLOT_CPUS)
        with patch.object(launch, 'original_node_resources', return_value=dict(allowed_cpus=list(range(80)))):
            self.assertEqual(launch.node_resources('4090_1')['allowed_cpus'], [4, 5, 72, 73])
        with patch.object(launch, 'original_node_resources', return_value=dict(allowed_cpus=[12])):
            with self.assertRaisesRegex(RuntimeError, 'affinity'):
                launch.node_resources('4090_0')

    def test_draft_cli_is_readiness_free_and_write_free(self):
        with patch.object(launch.base, 'save', side_effect=AssertionError('No writes')), \
                patch.object(launch.base, 'read', side_effect=AssertionError('No lock reads')), \
                patch.object(launch.base.subprocess, 'run', side_effect=AssertionError('No subprocess')):
            output = io.StringIO()
            with redirect_stdout(output):
                self.assertEqual(wave.main([launch.ARMS[0]]), 0)
            spec = json.loads(output.getvalue())
        self.assertTrue(spec['spec_draft'])
        self.assertFalse(spec['main_thread_ready'])
        self.assertFalse(self.root.exists())
        self.assertIn('r110_memory_witness.py', launch.SOURCE_NAMES)
        self.assertIn('run_v4_r110_witness.sh', launch.SOURCE_NAMES)

    def test_final_spec_consumes_runner_contract_and_refuses_overwrite(self):
        with self.assertRaisesRegex(ValueError, 'draft is not ready'):
            wave.make_spec(launch.ARMS[0], final=True)
        path, spec, science = self.locked_spec()
        self.assertEqual(launch.load_spec(path), spec)
        saved = path.read_bytes()
        self.assertEqual(wave.write_spec(launch.ARMS[0]), path)
        self.assertEqual(path.read_bytes(), saved)
        launch.base.save(path, dict(spec, unrelated=True))
        with self.assertRaisesRegex(RuntimeError, 'Preserve existing'):
            wave.write_spec(launch.ARMS[0])
        science['max_updates'] = 1
        launch.base.save(self.root / 'protocol_lock.json', science)
        with self.assertRaisesRegex(ValueError, 'runner/model'):
            launch.load_spec(path)

    def test_exact_stage_parsers_and_dynamic_shared_caps(self):
        from . import r110_reporting as reporter
        runner = launch.runner_contract()
        path, spec, _ = self.locked_spec()
        for arm in launch.ARMS:
            stages = launch.base.commands(wave.make_spec(arm, final=True), launch.ARM_SLOTS[arm])
            self.assertEqual(tuple(s['name'] for s in stages), launch.STAGES)
            for index, stage in enumerate(stages):
                argv = stage['argv']
                args = (reporter if index >= 3 else runner).parser().parse_args(argv[argv.index('-m') + 2:])
                self.assertEqual(args.device, 'cpu' if index == 4 else 'cuda:0')
                if index < 3:
                    runner.protocol_lock(args)
                    self.assertEqual(args.large_micro_cap, runner.CAPS['large_micro_cap'])
                if index == 3:
                    self.assertEqual(args.lib_batch_size, 2)
        caps = dict(runner.CAPS, large_micro_cap=32, other_micro_cap=128, eval_batch_size=32)
        with patch.object(runner, 'CAPS', caps):
            self.assertEqual(launch.microcaps(), caps)
            self.assertIn('32', launch.expected_stages(launch.ARMS[0])[0]['argv'])
        for mutate in (lambda s: s.update(spec_draft=True), lambda s: s.update(cpus=[12, 13, 14, 15]),
                       lambda s: s.update(feature_root='/tmp/new-features'),
                       lambda s: s['stages'][-2]['argv'].__setitem__(-1, '1')):
            changed = copy.deepcopy(spec)
            mutate(changed)
            launch.base.save(path, changed)
            with self.assertRaises(ValueError):
                launch.load_spec(path)

    def test_source_closure_optional_note_stays_out_if_not_frozen(self):
        source = self.temp / 'runtime.py'
        source.write_text('from code.V4 import r109_model\n')
        note = self.temp / 'late_note.md'
        with patch.object(launch, 'declared_sources', return_value=[source]), \
                patch.object(launch, 'PUBLICATION_EXTRAS', (str(note),)):
            frozen = launch.source_files(dict(source_files=[str(source)]))
            self.assertTrue(any(p.endswith('/r109_model.py') for p in frozen))
            note.write_text('arrived after final spec\n')
            self.assertEqual(launch.source_files(dict(source_files=[str(source)])), frozen)
            self.assertIn(str(note), launch.source_files(dict(source_files=[str(source), str(note)])))
            with self.assertRaisesRegex(ValueError, 'named dependencies'):
                launch.source_files(dict(source_files=['/tmp/unrelated.py']))

    def test_one_shot_freeze_and_post_ready_edits_fail_closed(self):
        source = self.temp / 'owned.txt'
        source.write_text('before freeze\n')
        for directory in (launch.DATA_ROOT, launch.PREPARED_ROOT, launch.CACHE_SOURCE):
            directory.mkdir()
        (launch.CACHE_SOURCE / 'fake.npy').write_bytes(b'fake cache fixture, no arrays loaded')
        with patch.object(launch, 'declared_sources', return_value=[source]), \
                patch.object(launch.adapter, 'parse_commands', return_value=[dict(argv=['fake'])]):
            path, spec, _ = self.locked_spec()
            result = launch.prepare(path, 'main CPU contracts', True)
            self.assertFalse(result['gpu_started'])
            self.assertTrue(result['source_freeze_ready'])
            self.assertEqual(launch.check_ready(path)[1]['evidence'], 'main CPU contracts')
            ready = (launch.base.receipts(spec) / 'ready.json').read_bytes()
            with self.assertRaises(RuntimeError):
                launch.prepare(path, 'main CPU contracts', True)
            self.assertEqual((launch.base.receipts(spec) / 'ready.json').read_bytes(), ready)
            source.write_text('changed after freeze\n')
            with self.assertRaisesRegex(RuntimeError, 'changed after ready'):
                launch.check_ready(path)

    def test_cache_is_in_place_only_never_copied_or_repaired(self):
        cache = launch.CACHE_ROOT
        cache.mkdir()
        path = cache / 'array.npy'
        path.write_bytes(b'fake-only')
        manifest = dict(identity_sha256='fake-identity', files=[dict(path=path.name, bytes=path.stat().st_size)])
        launch.base.save(cache / '.r84_cache_identity.json', dict(identity_sha256='fake-identity'))
        spec = dict(cache_root=str(cache))
        self.assertFalse(launch.stage_cache(spec, manifest)['copied'])
        path.unlink()
        with self.assertRaisesRegex(RuntimeError, 'no repair or copy'):
            launch.stage_cache(spec, manifest)
        self.assertFalse(path.exists())

    def test_predecessor_is_assigned_original_R108_not_GitHub_or_other_arm(self):
        arm, prior_arm = launch.ARMS[0], launch.PREDECESSORS[launch.ARMS[0]]
        dispatch = self.temp / 'old-dispatch'
        model = self.temp / 'old-R108'
        identifier = 'R108_' + prior_arm
        launch.base.save(dispatch / 'specs' / (identifier + '.json'), dict(run_id=identifier,
            run_dir=str(model / prior_arm), slots=['4090_0'],
            stages=[dict(name=n, device='cpu' if n == 'PDF' else 'gpu') for n in launch.STAGES]))
        for name, stages in (('worker', ['train', 'TEST', 'timing', 'LIB']),
                             ('controller', ['allocation_GPU_pipeline', 'PDF'])):
            launch.base.save(dispatch / identifier / (name + '.json'), dict(state='complete', exit_status=0,
                stages=[dict(name=n, state='complete', exit_status=0) for n in stages]))
        current = dict(arms=[dict(arm=prior_arm, checks=dict(full_TRAIN=True, full_VAL=True, full_TEST=True, same_model_NSS4=False))],
            errors=[dict(path=str(model / 'github_publication.json'), error='unpublished')], complete=False)
        with patch.object(launch, 'PREDECESSOR_ROOT', model), \
                patch.object(launch, 'PREDECESSOR_DISPATCH', dispatch), \
                patch.object(launch.predecessor_goal, 'status', return_value=current):
            self.assertTrue(launch.predecessor_gate(arm)['complete'])
            launch.base.save(dispatch / identifier / 'worker.json', dict(state='complete', exit_status=0, stages=[]))
            self.assertFalse(launch.predecessor_gate(arm)['complete'])

    def test_start_cannot_bypass_main_or_predecessor_gate(self):
        for action in (launch.prepare, launch.start, launch.queue):
            with self.assertRaises(ValueError):
                action('/tmp/never-written')
        spec = dict(run_id=launch.run_id(launch.ARMS[0]))
        with patch.object(launch, 'check_ready', return_value=(spec, dict(evidence='main'))), \
                patch.object(launch, 'predecessor_gate', return_value=dict(errors=[], complete=False)), \
                patch.object(launch, 'original_start') as start:
            with self.assertRaisesRegex(RuntimeError, 'original R108'):
                launch.start('/tmp/fake', 'main', True)
            with self.assertRaises(ValueError):
                launch.start('/tmp/fake', 'different', True)
        start.assert_not_called()


if __name__ == '__main__':
    unittest.main()

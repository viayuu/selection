"""Focused CPU tests for resource isolation, ready gating, and cache reuse."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from . import r84_launch as historical
from . import r85_launch as launch


class R85LaunchTests(unittest.TestCase):
    def test_ovr_eql_pair_uses_same_hardware_and_micro_cap(self):
        self.assertEqual(launch.ARM_SLOTS['A_ovr_control'], '3090_0')
        self.assertEqual(launch.ARM_SLOTS['B_gradient_balance'], '3090_1')
        for arm in ('A_ovr_control', 'B_gradient_balance'):
            slot = launch.ARM_SLOTS[arm]
            command = launch.base.commands(launch.final_spec(arm), slot)[0]['argv']
            self.assertEqual(command[command.index('--gpu-type') + 1], '3090')
            self.assertEqual(command[command.index('--large-micro-cap') + 1], '384')
        self.assertEqual({launch.ARM_SLOTS[arm] for arm in ('C_cost_profile', 'D_candidate_subset')},
                         {'4090_0', '4090_1'})
        spec = launch.final_spec('D_candidate_subset')
        raw = spec['stages'][0]['argv']
        self.assertEqual(raw[raw.index('--large-micro-cap') + 1], '256')
        self.assertEqual(spec['protocol']['large_instance_micro_cap'], 256)

    def test_private_adapter_does_not_change_historical_launcher(self):
        self.assertEqual(historical.ROOT, launch.R84_ROOT)
        self.assertEqual(historical.TOKEN_ENV, 'R84_LAUNCH_TOKEN')
        self.assertIsNot(historical, launch.base)
        self.assertEqual(launch.base.ROOT, launch.ROOT)
        self.assertEqual(launch.base.TOKEN_ENV, 'R85_LAUNCH_TOKEN')

    def test_four_unique_bindings_use_owned_container_and_cpu_step(self):
        self.assertEqual(len({item['uuid'] for item in launch.SLOTS.values()}), 4)
        for slot, item in launch.SLOTS.items():
            command = launch.container_command('worker', slot, '/final/spec.json', 'unit-token')
            self.assertIn(str(launch.WRAPPER), command)
            self.assertNotIn(str(launch.REPO / 'code/V4/run_v4_r84_launch.sh'), command)
            self.assertIn('CUDA_VISIBLE_DEVICES=' + item['uuid'], command)
            self.assertIn('R85_LAUNCH_TOKEN=unit-token', command)
            self.assertIn(launch.base.transport.CONTAINERS[item['node']][1], command)
            step = launch.base.srun_command('entry', slot, cpus=4, memory='9G')
            self.assertIn('--jobid=' + str(item['job']), step)
            self.assertIn('--gres=none', step)
            self.assertIn('--cpus-per-task=4', step)
            self.assertIn('--mem=9G', step)
            self.assertIn('code.V4.r85_launch', step[-1])
            self.assertNotIn('ssh', step)

    def test_start_requires_main_ready_without_side_effects(self):
        with mock.patch.object(launch.base.socket, 'gethostname', return_value='admin01'), \
                mock.patch.object(Path, 'is_file', return_value=True), \
                mock.patch.object(launch.base.subprocess, 'run') as run:
            with self.assertRaisesRegex(RuntimeError, 'explicit ready'):
                launch.base.start(Path('/missing.json'), '', False)
            run.assert_not_called()

    def test_specs_keep_protocol_owned_by_main_and_require_fresh_training(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'spec.json'
            value = launch.template('unit')
            value['stages'] = [dict(name='train', device='gpu', argv=['{python}', '-m', 'final.runner'])]
            value['protocol']['main_protocol'] = 'final-unit-protocol'
            value['protocol']['max_successful_updates'] = 12345
            path.write_text(json.dumps(value))
            parsed = launch.load_spec(path)
            self.assertEqual(parsed['protocol']['max_successful_updates'], 12345)
            self.assertEqual(parsed['cache_root'], str(launch.CACHE_ROOT))
            value['protocol']['initialization'] = 'resume'
            path.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError, 'fresh weights'):
                launch.load_spec(path)

    def test_cache_reuse_reads_metadata_and_sizes_without_copying_or_test_access(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache = root / 'local'
            (cache / 'train' / 'TSP').mkdir(parents=True)
            (cache / 'val').mkdir()
            array = cache / 'train/TSP/node.npy'
            array.write_bytes(b'payload')
            stamp = cache / '.r84_cache_identity.json'
            stamp.write_text(json.dumps(dict(data_root='/data', source='/source', identity_sha256='old-without-test')))
            before = stamp.read_bytes()
            manifest = dict(bytes=7, identity_sha256='new-train-val-only',
                            files=[dict(path='train/TSP/node.npy', bytes=7)])
            spec = dict(cache_root=str(cache), data_root='/data', cache_source='/source')
            with mock.patch.object(launch.base.shutil, 'copyfile', side_effect=AssertionError('must reuse')):
                result = launch.stage_cache(spec, manifest)
            self.assertEqual(result['copied_bytes'], 0)
            self.assertTrue(result['test_cache_ignored'])
            self.assertFalse(result['cache_audit_repeated'])
            self.assertEqual(stamp.read_bytes(), before)
            bad = dict(manifest, files=[dict(path='test/TSP/costs.npy', bytes=7)])
            with self.assertRaisesRegex(RuntimeError, 'excludes TEST'):
                launch.stage_cache(spec, bad)
            array.unlink()
            (cache / 'train/TSP').rmdir()
            (cache / 'train').rmdir()
            with self.assertRaisesRegex(RuntimeError, 'no automatic rebuild'):
                launch.stage_cache(spec, manifest)

    def test_gpu_stages_remain_before_cpu_pdf_and_commands_keep_exact_argv(self):
        spec = launch.template('unit')
        spec['stages'] = [dict(name='gpu', device='gpu', argv=['{python}', 'runner.py', '{device}']),
                          dict(name='pdf', device='cpu', argv=['{python}', 'report.py', '{device}'])]
        stages = launch.base.commands(spec, '4090_1')
        self.assertEqual(stages[0]['argv'][-1], 'cuda:0')
        self.assertEqual(stages[1]['argv'][-1], 'cpu')
        self.assertEqual(launch.base.environment()['CUDA_VISIBLE_DEVICES'], '')
        self.assertEqual(launch.base.session_name(spec), 'r85_unit')


if __name__ == '__main__':
    unittest.main()

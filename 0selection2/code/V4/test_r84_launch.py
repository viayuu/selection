"""CPU-only checks for the isolated resource launcher, without running jobs."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from . import r84_launch as launch


class LauncherTests(unittest.TestCase):
    def write_spec(self, directory):
        root = Path(directory)
        value = launch.template('A_baseline')
        for name in ('run_dir', 'data_root', 'prepared_root', 'cache_source'):
            value[name] = str(root / name)
        value['cache_root'] = '/tmp/r84_CPU_test_new_cache'
        value['protocol']['solver_features'] = 'R77'
        value['stages'] = [dict(name='train', device='gpu',
            argv=['{python}', '-m', 'example.runner', '--cache-root', '{cache_root}', '--device', '{device}']),
            dict(name='report', device='cpu', argv=['{python}', '-m', 'example.report', '--device', '{device}'])]
        path = root / 'spec.json'
        path.write_text(json.dumps(value))
        return path

    def test_all_four_uuid_bindings_and_shell_free_container_entry(self):
        self.assertEqual(len({row['uuid'] for row in launch.SLOTS.values()}), 4)
        for slot in launch.SLOTS:
            command = launch.container_command('worker', slot, Path('/shared/spec.json'), 'token')
            self.assertIn('CUDA_VISIBLE_DEVICES=' + launch.SLOTS[slot]['uuid'], command)
            self.assertNotIn('-lc', command)
            self.assertIn(str(launch.REPO / 'code/V4/run_v4_r84_launch.sh'), command)
            srun = launch.srun_command('entry', slot)
            self.assertIn('--gres=none', srun)
            self.assertIn('--jobid=' + str(launch.SLOTS[slot]['job']), srun)

    def test_exact_stage_device_substitution(self):
        with tempfile.TemporaryDirectory() as directory:
            spec = launch.load_spec(self.write_spec(directory))
            stages = launch.commands(spec, '3090_0')
            self.assertEqual(stages[0]['argv'][-1], 'cuda:0')
            self.assertEqual(stages[1]['argv'][-1], 'cpu')
            self.assertEqual(launch.environment()['CUDA_VISIBLE_DEVICES'], '')

    def test_legacy_cache_cannot_be_relabeled_new(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_spec(directory)
            value = launch.read(path)
            for name in ('cache_source', 'cache_root'):
                bad = dict(value, **{name: str(launch.transport.CACHE)})
                path.write_text(json.dumps(bad))
                with self.assertRaisesRegex(ValueError, 'NEW cache'):
                    launch.load_spec(path)

    def test_start_cannot_run_without_main_ready(self):
        with mock.patch.object(launch.socket, 'gethostname', return_value='admin01'), \
                mock.patch.object(Path, 'is_file', return_value=True), \
                mock.patch.object(launch.subprocess, 'run') as run:
            with self.assertRaisesRegex(RuntimeError, 'explicit ready'):
                launch.start(Path('/does/not/exist.json'), '', False)
            run.assert_not_called()

    def test_new_cache_stages_once_and_rejects_wrong_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'new_source'
            source.mkdir()
            (source / 'nodes.npy').write_bytes(b'array-payload')
            (source / 'meta.json').write_text('{"dataset":"new90k"}')
            spec = dict(cache_source=str(source), cache_root=str(root / 'local'), data_root='/new90k')
            manifest = launch.cache_manifest(source)
            first = launch.stage_cache(spec, manifest)
            with mock.patch.object(launch.shutil, 'copyfile', side_effect=AssertionError('must reuse new cache')):
                self.assertEqual(launch.stage_cache(spec, manifest), first)
            bad = dict(manifest, identity_sha256='wrong')
            with self.assertRaisesRegex(RuntimeError, 'another dataset'):
                launch.stage_cache(spec, bad)

    def test_parallel_arms_reserve_disjoint_cpus(self):
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(launch, 'ROOT', Path(directory)):
            first = Path(directory) / 'A_baseline'
            launch.save(first / 'claim.json', dict(node='gpu03', cpus=[12, 13, 14, 15]))
            reserved = launch.reserved_cpus('gpu03', Path(directory) / 'C_focal')
            self.assertEqual(reserved, {12, 13, 14, 15})
            launch.save(first / 'controller.json', dict(state='complete'))
            self.assertEqual(launch.reserved_cpus('gpu03', Path(directory) / 'C_focal'), set())

    def test_failed_stage_stops_pipeline_before_report(self):
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(launch, 'ROOT', Path(directory)):
            spec = dict(run_id='unit', stages=[])
            output = launch.receipts(spec)
            output.mkdir()
            with mock.patch.object(launch, 'run_stage', side_effect=RuntimeError('train exit=1')), \
                    mock.patch.object(launch, 'commands') as commands:
                with self.assertRaisesRegex(RuntimeError, 'train exit=1'):
                    launch.pipeline(Path('/spec.json'), spec, '3090_0', dict(token='t'))
                commands.assert_not_called()
                self.assertEqual(launch.read(output / 'controller.json')['state'], 'failed')


if __name__ == '__main__':
    unittest.main()

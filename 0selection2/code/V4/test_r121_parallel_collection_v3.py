"""Three bounded CPU-only resource-adapter fixtures; no collector/GPU launch."""

import hashlib
import json
import os
from pathlib import Path
import tempfile
import types
import unittest
from unittest import mock

from . import r121_parallel_collection_v3 as adapter


def setUpModule():
    allowed = sorted(os.sched_getaffinity(0))
    os.sched_setaffinity(0, set(allowed[:2]))


class AdapterTests(unittest.TestCase):
    def test_bootstrap_precedes_snapshot_new_receipt_and_no_repeat(self):
        metadata, events = adapter.provenance(), []
        pinned = adapter.scheduler(metadata)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / adapter.ROOT_NAME
            release, binding = root / 'release.json', root / 'binding.json'
            receipt = root / adapter.VERSION / 'lane2/controller.json'

            def bootstrap(*args):
                self.assertEqual(args, (root, 2, release, binding))
                events.append('bootstrap')
                return dict(status='bootstrap_complete', numeric_sha256='numeric-seal', bootstrap_sha256='adoption-seal')

            def execute(*args):
                events.append('layout_snapshot')
                self.assertEqual(args, (root, 2, 2, 1280, 'cuda:0', release, binding))
                value = dict(state='finite_budget_complete', controller_source_sha256=adapter.LEGACY_SHA256)
                pinned.write(receipt, value, exclusive=True)
                return value, 0

            numeric = types.SimpleNamespace(bootstrap_lane=mock.Mock(side_effect=bootstrap))
            with mock.patch.object(pinned.base.data, 'ROOT', Path(temporary)), \
                    mock.patch.object(adapter, 'provenance', return_value=metadata), \
                    mock.patch.object(adapter, 'scheduler', return_value=pinned), \
                    mock.patch.object(adapter, 'require_mutex', side_effect=lambda *args: events.append('mutex')), \
                    mock.patch.object(adapter.importlib, 'import_module', return_value=numeric) as imported, \
                    mock.patch.object(pinned, 'run', side_effect=execute):
                result, code = adapter.run(root, 2, 2, 1280, 'cuda:0', release, binding)
                original = receipt.read_bytes()
                with self.assertRaises(FileExistsError):
                    adapter.run(root, 2, 2, 1280, 'cuda:0', release, binding)
            self.assertEqual(events, ['mutex', 'bootstrap', 'layout_snapshot'])
            imported.assert_called_once_with(adapter.WORKER_MODULE)
            numeric.bootstrap_lane.assert_called_once()
            self.assertEqual(code, 0)
            self.assertEqual(receipt.read_bytes(), original)
            for record in (result, json.loads(original)):
                self.assertEqual(record['version'], 'r121_parallel_collection_v3')
                self.assertEqual(record['legacy_scheduler_sha256'], adapter.LEGACY_SHA256)
                self.assertEqual(record['resource_v2_sha256'], adapter.V2_SHA256)
                self.assertEqual(record['adapter_sha256'], adapter.ADAPTER_SHA256)
                self.assertEqual(record['numeric_collector_sha256'], adapter.NUMERIC_SHA256)
                self.assertEqual(record['bootstrap_sha256'], 'adoption-seal')
                self.assertFalse(record['automatic_retry'])
                self.assertFalse(record['full89_complete'])

    def test_only_child_module_changes_original_lane_order_and_budgets(self):
        from . import r121_parallel_collection as original
        pinned = adapter.scheduler(adapter.provenance())
        old_version, old_command = original.VERSION, original.child_command
        args = (Path('/new'), 'ATSP/ICAM_ATSP', 1279, 'cuda:0', Path('/release'), Path('/binding'), Path('/result'))
        before, after = original.child_command(*args), pinned.child_command(*args)
        self.assertEqual(len(before), len(after))
        self.assertEqual([(a, b) for a, b in zip(before, after) if a != b],
                         [('code.V4.r121_collection', 'code.V4.r121_numeric_collection')])
        order = [f'P{i}/method' for i in range(89)]
        groups = {key: range(1280) for key in order}
        protocol = dict(mode='cached_native_v1', lanes=4, batch_size=1,
            native_budget_changes=False, deployment_seed_changes=False,
            group_order=order, planned_labels=113920)
        with mock.patch.object(pinned.base.data, 'owned', side_effect=Path), \
                mock.patch.object(pinned.base, 'load', return_value=(protocol, {}, {})), \
                mock.patch.object(pinned.base, 'task_groups', return_value=groups):
            for lane in range(4):
                self.assertEqual(pinned.layout(Path('/new'), lane, 2, 1280)[-1], order[lane::4])
            for concurrency, maximum in ((1, 1280), (3, 1280), (2, 1281), (2, 0)):
                with self.assertRaises(ValueError):
                    pinned.layout(Path('/new'), 0, concurrency, maximum)
        self.assertEqual(original.VERSION, old_version)
        self.assertIs(original.child_command, old_command)
        self.assertEqual(hashlib.sha256(adapter.LEGACY_PATH.read_bytes()).hexdigest(), adapter.LEGACY_SHA256)
        self.assertEqual(hashlib.sha256(adapter.V2_PATH.read_bytes()).hexdigest(), adapter.V2_SHA256)

    def test_original_mutex_inode_UUID_required_not_NFS_device_number(self):
        gpu = 'GPU-unit-fixture'
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / 'code/V4/runs/R81_method_conditioned_auc' / ('queue_gpu_' + gpu + '.json')
            path.parent.mkdir(parents=True)
            path.write_text('{}')
            inode = path.stat().st_ino
            legacy = types.SimpleNamespace(REPO=root, base=types.SimpleNamespace(read=lambda p: {'gpu_uuid': gpu}))
            with mock.patch.dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, RETURN18_GPU_MUTEX_FD='57'), \
                    mock.patch.object(adapter.os, 'fstat', return_value=types.SimpleNamespace(st_ino=inode, st_dev=-999)) as stat, \
                    mock.patch.object(adapter.fcntl, 'flock') as flock:
                adapter.require_mutex(legacy, Path('/binding'))
                flock.assert_called_once_with(57, adapter.fcntl.LOCK_EX | adapter.fcntl.LOCK_NB)
                stat.return_value.st_ino = inode + 1
                with self.assertRaises(PermissionError):
                    adapter.require_mutex(legacy, Path('/binding'))
                stat.return_value.st_ino = inode
                with mock.patch.dict(os.environ, CUDA_VISIBLE_DEVICES='GPU-wrong'):
                    with self.assertRaises(PermissionError):
                        adapter.require_mutex(legacy, Path('/binding'))
                with mock.patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': gpu}, clear=True):
                    with self.assertRaises(PermissionError):
                        adapter.require_mutex(legacy, Path('/binding'))
                self.assertEqual(flock.call_count, 1)


if __name__ == '__main__':
    unittest.main()

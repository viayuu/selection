"""Resource adapter for the explicit-CPU collector, not a new native algorithm.

Data-owner API: r121_universal_collection.bootstrap_lane(root, lane, release,
binding) validates/adopts the latest actual prefix and qualifications into
collection_explicit_cpu_v2. It must leave protocol/preflight/progress compatible
with the frozen r121_collection readers used by the a9e scheduler. Its worker CLI
must accept the unchanged legacy worker flags. Bootstrap runs under the ORIGINAL
assigned UUID mutex BEFORE any scheduler layout, snapshot, or child launch.

Production: python -m code.V4.r121_parallel_collection_v2 --root NEW_ROOT
--lane 0 --concurrency 2 --max-jobs-per-deployment 1280 --device cuda:0
--release RELEASE_JSON --binding BINDING_JSON. Parent owns all real launches.
--describe is static CPU-only adapter metadata; --self-test runs two mocked cases.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import types


VERSION = 'r121_parallel_collection_v2'
LEGACY_SHA256 = 'a9e686d13f0acb4e956178a4d21bf91c3851307a05cc1839ed411333c4bad218'
LEGACY_PATH = Path(__file__).with_name('r121_parallel_collection.py')
WORKER_MODULE = 'code.V4.r121_universal_collection'
ADAPTER_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def provenance():
    current = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if current != ADAPTER_SHA256:
        raise ValueError('Loaded v2 adapter source changed; refuse stale execution')
    return dict(version=VERSION, legacy_scheduler_sha256=LEGACY_SHA256,
                adapter_sha256=ADAPTER_SHA256, native_worker_module=WORKER_MODULE,
                resource_only_adapter=True, native_algorithm_change=False,
                full89_complete=False, automatic_retry=False)


def scheduler(metadata):
    source = LEGACY_PATH.read_bytes()
    if hashlib.sha256(source).hexdigest() != LEGACY_SHA256:
        raise ValueError('The frozen a9e scheduler source changed')
    # Private globals only: never mutate the imported live scheduler or its file.
    module = types.ModuleType('code.V4._r121_parallel_collection_v2_pinned')
    module.__package__ = 'code.V4'
    module.__file__ = str(LEGACY_PATH)
    exec(compile(source, str(LEGACY_PATH), 'exec'), module.__dict__)
    module.VERSION = VERSION
    original_command, original_write = module.child_command, module.write

    def child_command(*args, **kwargs):
        command = original_command(*args, **kwargs)
        index = command.index('-m') + 1
        if command[index] != 'code.V4.r121_collection':
            raise ValueError('Unexpected frozen child module; do not infer a replacement')
        command[index] = WORKER_MODULE
        return command

    def write(path, value, *, exclusive=False):
        value.update(metadata)
        return original_write(path, value, exclusive=exclusive)

    module.child_command, module.write = child_command, write
    return module


def require_mutex(legacy, binding):
    """Retain the inherited original inode; no GPU probe, new lock or unlink."""
    record = legacy.base.read(binding)
    gpu = record.get('gpu_uuid')
    if not gpu or os.environ.get('CUDA_VISIBLE_DEVICES') != gpu:
        raise PermissionError('Explicit bootstrap must retain its assigned CUDA UUID')
    value = os.environ.get('RETURN18_GPU_MUTEX_FD')
    if value is None:
        raise PermissionError('Parent must acquire/pass the original UUID mutex before bootstrap')
    descriptor = int(value)
    path = legacy.REPO / 'code/V4/runs/R81_method_conditioned_auc' / ('queue_gpu_' + gpu + '.json')
    if os.fstat(descriptor).st_ino != path.stat().st_ino:
        raise PermissionError('Inherited descriptor is not the original assigned UUID mutex inode')
    # Idempotent for the inherited locked open-file description; never unlock it.
    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)


def run(root, lane, concurrency, maximum, device, release, binding):
    if lane not in range(4) or concurrency != 2 or not 1 <= maximum <= 1280 or device != 'cuda:0':
        raise ValueError('Require lane0..3, concurrency2, budget1..1280 and cuda:0')
    metadata = provenance()
    legacy = scheduler(metadata)
    root = legacy.base.data.owned(root)
    if root.name != 'collection_explicit_cpu_v2':
        raise ValueError('Bootstrap is allowed only in the NEW collection_explicit_cpu_v2 root')
    release, binding = Path(release).resolve(), Path(binding).resolve()
    receipt = root / VERSION / ('lane' + str(lane)) / 'controller.json'
    if receipt.exists():
        raise FileExistsError('Preserve the v2 one-shot controller; no bootstrap/retry of an existing attempt')
    require_mutex(legacy, binding)
    universal = importlib.import_module(WORKER_MODULE)
    bootstrap = getattr(universal, 'bootstrap_lane', None)
    if not callable(bootstrap):
        raise ValueError('Data owner must provide bootstrap_lane(root, lane, release, binding)')
    bootstrap(root, lane, release, binding)
    metadata.update(bootstrap_completed=True, bootstrap_before_snapshot=True,
                    bootstrap_api=WORKER_MODULE + '.bootstrap_lane')
    universal_path = getattr(universal, '__file__', None)
    if universal_path:
        metadata['universal_collector_sha256'] = legacy.base.sha(universal_path)
    result, code = legacy.run(root, lane, concurrency, maximum, device, release, binding)
    result.update(metadata)
    return result, code


def self_test():
    import tempfile
    import unittest
    from unittest import mock

    class AdapterTests(unittest.TestCase):
        def test_bootstrap_precedes_layout_and_receipt_records_both_hashes(self):
            events, metadata = [], provenance()
            pinned = scheduler(metadata)
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary) / 'collection_explicit_cpu_v2'
                def bootstrap(*args):
                    events.append('bootstrap')
                    self.assertEqual(args, (root, 2, root / 'release.json', root / 'binding.json'))
                    root.mkdir()
                def execute(*args):
                    events.append('layout_snapshot')
                    value = dict(state='finite_budget_complete', controller_source_sha256=LEGACY_SHA256)
                    pinned.write(root / 'receipt.json', value, exclusive=True)
                    return value, 0
                with mock.patch.object(pinned.base.data, 'ROOT', Path(temporary)), \
                        mock.patch(__name__ + '.scheduler', return_value=pinned), \
                        mock.patch(__name__ + '.require_mutex') as mutex, \
                        mock.patch.object(importlib, 'import_module', return_value=types.SimpleNamespace(bootstrap_lane=bootstrap)), \
                        mock.patch.object(pinned, 'run', side_effect=execute):
                    result, code = run(root, 2, 2, 1280, 'cuda:0', root / 'release.json', root / 'binding.json')
                self.assertEqual(events, ['bootstrap', 'layout_snapshot'])
                mutex.assert_called_once()
                self.assertEqual(code, 0)
                saved = json.loads((root / 'receipt.json').read_text())
                for row in (saved, result):
                    self.assertEqual(row['legacy_scheduler_sha256'], LEGACY_SHA256)
                    self.assertEqual(row['adapter_sha256'], ADAPTER_SHA256)
                    self.assertEqual(row['version'], VERSION)
                    self.assertFalse(row['full89_complete'])
                    self.assertFalse(row['native_algorithm_change'])

        def test_child_module_only_change_and_no_mutation_of_frozen_scheduler(self):
            import code.V4.r121_parallel_collection as original
            old_version, old_command = original.VERSION, original.child_command
            pinned = scheduler(provenance())
            args = (Path('/new'), 'ATSP/ICAM_ATSP', 1279, 'cuda:0', Path('/release'), Path('/binding'), Path('/result'))
            before = original.child_command(*args)
            after = pinned.child_command(*args)
            self.assertEqual([(a, b) for a, b in zip(before, after) if a != b],
                             [('code.V4.r121_collection', WORKER_MODULE)])
            self.assertEqual(original.VERSION, old_version)
            self.assertIs(original.child_command, old_command)
            self.assertEqual(hashlib.sha256(LEGACY_PATH.read_bytes()).hexdigest(), LEGACY_SHA256)

    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(AdapterTests))
    return 0 if result.wasSuccessful() else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path)
    parser.add_argument('--lane', type=int, choices=range(4))
    parser.add_argument('--concurrency', type=int, choices=(2,), default=2)
    parser.add_argument('--max-jobs-per-deployment', type=int, default=1280)
    parser.add_argument('--device')
    parser.add_argument('--release', type=Path)
    parser.add_argument('--binding', type=Path)
    parser.add_argument('--describe', action='store_true', help='Static CPU-only adapter contract, no root adoption')
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args(argv)
    if args.self_test:
        return self_test()
    try:
        if args.describe:
            value = provenance()
            scheduler(value)
            print(json.dumps(dict(value, solver_execution=False, root_adoption=False,
                required_bootstrap='bootstrap_lane(root, lane, release, binding)',
                root_protocol_contract='frozen r121_collection.load/qualification/progress compatible'), sort_keys=True))
            return 0
        if any(value is None for value in (args.root, args.lane, args.device, args.release, args.binding)):
            parser.error('Production requires --root --lane --device --release --binding')
        result, code = run(args.root, args.lane, args.concurrency, args.max_jobs_per_deployment,
                           args.device, args.release, args.binding)
        print(json.dumps(result, sort_keys=True), flush=True)
        return code
    except Exception as error:
        print(json.dumps(dict(provenance(), state='failed', first_error=f'{type(error).__name__}: {error}')), flush=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())

"""Resource-only numeric-root adapter of the frozen a9e two-process scheduler.

Uses the frozen v2 resource-wrapper contract, with a distinct v3 controller
namespace and numeric collector. Main creates/seals the new root and owns all
real launches. Numeric bootstrap alone adopts the fixed InputV1 / lane1 V2 /
other Original sources after source drain, under the inherited original mutex.
--describe is CPU-only and does not import the numeric collector or adopt data.
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


VERSION = 'r121_parallel_collection_v3'
ROOT_NAME = 'collection_numeric_boundary_v3'
LEGACY_SHA256 = 'a9e686d13f0acb4e956178a4d21bf91c3851307a05cc1839ed411333c4bad218'
V2_SHA256 = '42d6e560ddcc13dc04868dc547ae796b8d8783986d318cc8d19a77981aa8475e'
NUMERIC_SHA256 = 'a9c638f48cbb8164171ef95bef77ae3b159a254968eddc78a71676b7dc3df50e'
LEGACY_PATH = Path(__file__).with_name('r121_parallel_collection.py')
V2_PATH = Path(__file__).with_name('r121_parallel_collection_v2.py')
NUMERIC_PATH = Path(__file__).with_name('r121_numeric_collection.py')
WORKER_MODULE = 'code.V4.r121_numeric_collection'
ADAPTER_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def provenance():
    for path, expected in ((Path(__file__), ADAPTER_SHA256), (V2_PATH, V2_SHA256),
                           (NUMERIC_PATH, NUMERIC_SHA256)):
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError('Pinned resource/collector source changed: ' + str(path))
    return dict(version=VERSION, legacy_scheduler_sha256=LEGACY_SHA256,
        resource_v2_sha256=V2_SHA256, adapter_sha256=ADAPTER_SHA256,
        numeric_collector_sha256=NUMERIC_SHA256, native_worker_module=WORKER_MODULE,
        resource_only_adapter=True, native_algorithm_change=False,
        full89_complete=False, automatic_retry=False)


def scheduler(metadata):
    source = LEGACY_PATH.read_bytes()
    if hashlib.sha256(source).hexdigest() != LEGACY_SHA256:
        raise ValueError('The frozen a9e scheduler source changed')
    module = types.ModuleType('code.V4._r121_parallel_collection_v3_pinned')
    module.__package__ = 'code.V4'
    module.__file__ = str(LEGACY_PATH)
    exec(compile(source, str(LEGACY_PATH), 'exec'), module.__dict__)
    module.VERSION = VERSION
    original_command, original_write = module.child_command, module.write

    def child_command(*args, **kwargs):
        command = original_command(*args, **kwargs)
        index = command.index('-m') + 1
        if command[index] != 'code.V4.r121_collection':
            raise ValueError('Unexpected frozen child module')
        command[index] = WORKER_MODULE
        return command

    def write(path, value, *, exclusive=False):
        value.update(metadata)
        return original_write(path, value, exclusive=exclusive)

    module.child_command, module.write = child_command, write
    return module


def require_mutex(legacy, binding):
    """Same v2 UUID/inode check; retain the parent's locked open description."""
    gpu = legacy.base.read(binding).get('gpu_uuid')
    if not gpu or os.environ.get('CUDA_VISIBLE_DEVICES') != gpu:
        raise PermissionError('Numeric bootstrap must retain its assigned CUDA UUID')
    value = os.environ.get('RETURN18_GPU_MUTEX_FD')
    if value is None:
        raise PermissionError('Parent must pass the original UUID mutex FD before bootstrap')
    descriptor = int(value)
    original = legacy.REPO / 'code/V4/runs/R81_method_conditioned_auc' / ('queue_gpu_' + gpu + '.json')
    if os.fstat(descriptor).st_ino != original.stat().st_ino:
        raise PermissionError('Inherited FD is not the original assigned UUID mutex inode')
    # st_dev may differ across NFS mounts. Never unlink, unlock, close or probe GPU.
    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)


def run(root, lane, concurrency, maximum, device, release, binding):
    if (lane not in range(4) or concurrency != 2 or type(maximum) is not int
            or not 1 <= maximum <= 1280 or device != 'cuda:0'):
        raise ValueError('Require lane0..3, concurrency2, budget1..1280 and cuda:0')
    metadata = provenance()
    legacy = scheduler(metadata)
    root = legacy.base.data.owned(root)
    if root.name != ROOT_NAME:
        raise ValueError('Only the NEW collection_numeric_boundary_v3 root is allowed')
    release, binding = Path(release).resolve(), Path(binding).resolve()
    receipt = root / VERSION / ('lane' + str(lane)) / 'controller.json'
    if receipt.exists():
        raise FileExistsError('Preserve the v3 one-shot controller; no bootstrap/retry of this attempt')
    require_mutex(legacy, binding)
    numeric = importlib.import_module(WORKER_MODULE)
    bootstrap = getattr(numeric, 'bootstrap_lane', None)
    if not callable(bootstrap):
        raise ValueError('Numeric API must provide bootstrap_lane(root, lane, release, binding)')
    # Numeric API verifies release/binding and source silence BEFORE adopting any
    # latest valid prefix. Do not snapshot, create roots or choose sources here.
    adopted = bootstrap(root, lane, release, binding)
    if not isinstance(adopted, dict) or adopted.get('status') not in ('bootstrap_complete', 'already_bootstrapped'):
        raise ValueError('Numeric bootstrap did not report a completed adoption')
    metadata.update(bootstrap_completed=True, bootstrap_before_snapshot=True,
        bootstrap_api=WORKER_MODULE + '.bootstrap_lane',
        numeric_sha256=adopted['numeric_sha256'], bootstrap_sha256=adopted['bootstrap_sha256'])
    result, code = legacy.run(root, lane, concurrency, maximum, device, release, binding)
    result.update(metadata)
    return result, code


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path)
    parser.add_argument('--lane', type=int, choices=range(4))
    parser.add_argument('--concurrency', type=int, choices=(2,), default=2)
    parser.add_argument('--max-jobs-per-deployment', type=int, default=1280)
    parser.add_argument('--device')
    parser.add_argument('--release', type=Path)
    parser.add_argument('--binding', type=Path)
    parser.add_argument('--describe', action='store_true')
    args = parser.parse_args(argv)
    try:
        if args.describe:
            value = provenance()
            scheduler(value)
            print(json.dumps(dict(value, solver_execution=False, root_adoption=False,
                required_root=ROOT_NAME, required_bootstrap='bootstrap_lane(root, lane, release, binding)',
                lanes=[0, 1, 2, 3], concurrency=2, max_jobs_per_deployment=1280,
                planned_labels=113920, deployments=89, batch_size=1,
                controller_path='<root>/' + VERSION + '/lane<0..3>/controller.json'), sort_keys=True))
            return 0
        if any(value is None for value in (args.root, args.lane, args.device, args.release, args.binding)):
            parser.error('Production requires --root --lane --device --release --binding')
        result, code = run(args.root, args.lane, args.concurrency, args.max_jobs_per_deployment,
                           args.device, args.release, args.binding)
        print(json.dumps(result, sort_keys=True), flush=True)
        return code
    except Exception as error:
        print(json.dumps(dict(version=VERSION, adapter_sha256=ADAPTER_SHA256,
            state='failed', automatic_retry=False, first_error=f'{type(error).__name__}: {error}')), flush=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())

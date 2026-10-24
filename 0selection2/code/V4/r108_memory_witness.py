"""Bounded TRAIN-input memory witness before R108's first formal update."""

import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import time
from uuid import UUID

import numpy as np
import torch

from . import r84_experiment as inputs
from . import r84_launch as resources
from . import r108_model as models
from .pairwise_objective import inputs_only


V4 = Path(__file__).resolve().parent
ROOT = V4 / 'runs/R108_conditional_density_ratio'
PREPARED = V4 / 'runs/R84_new90k_attention/prepared'
DATA = Path('/public/home/shiys/selection_data_new90k_B_filtered_single_reld_mt_seed2_20261018')
CACHE = Path('/tmp/shiys_R84_new90k_cache')
SLOTS = {'A_prior_ce': '4090_0', 'B_density_ratio': '4090_1'}
CPUS = {'4090_0': {0, 1, 2, 3}, '4090_1': {4, 5, 72, 73}}


@contextmanager
def reservation(arm):
    slot = SLOTS[arm]
    item = resources.SLOTS[slot]
    container = resources.assert_container(slot)
    if (item['job'] != 1475 or item['node'] != 'gpu05'
            or os.environ.get('CUDA_VISIBLE_DEVICES') != item['uuid']
            or not CPUS[slot] <= os.sched_getaffinity(0)):
        raise RuntimeError('Use existing1475/gpu05 container, assigned UUID and CPU quartet')
    path = resources.transport.ROOT / ('queue_gpu_' + item['uuid'] + '.json')
    if path.is_symlink() or not path.is_file():
        raise RuntimeError('Reuse the original GPU mutex; never create or replace it')
    before = path.stat()
    with path.open('r') as handle:
        resources.fcntl.flock(handle, resources.fcntl.LOCK_EX | resources.fcntl.LOCK_NB)
        stat = os.fstat(handle.fileno())
        if (before.st_dev, before.st_ino) != (stat.st_dev, stat.st_ino):
            raise RuntimeError('Existing GPU mutex identity changed')
        if UUID(str(torch.cuda.get_device_properties(0).uuid).removeprefix('GPU-')) != UUID(item['uuid'].removeprefix('GPU-')):
            raise RuntimeError('Actual CUDA device differs from the assigned UUID')
        previous = os.sched_getaffinity(0)
        try:
            os.sched_setaffinity(0, CPUS[slot])
            yield dict(slot=slot, gpu_uuid=item['uuid'], container_id=container,
                cpus=sorted(CPUS[slot]), gpu_mutex=dict(path=str(path), inode=stat.st_ino, device=stat.st_dev))
        finally:
            os.sched_setaffinity(0, previous)


def measure(model, raw, problem, cap):
    model.train()
    model.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    with torch.autocast(device_type='cuda', dtype=torch.float16):
        first = model(raw)
    loss = first['residual_utility'][first['solver_mask']].float().square().mean()
    (loss * 1024.).backward()
    torch.cuda.synchronize()
    groups = dict(original=[])
    nonzero = dict(original=0)
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        group = 'original'
        groups[group].append(bool(torch.isfinite(parameter.grad).all()))
        nonzero[group] += int(bool(parameter.grad.abs().max() > 0))
    if (not torch.isfinite(loss) or any(not flags or not all(flags) or not nonzero[key]
                                      for key, flags in groups.items())):
        raise RuntimeError('Need finite nonzero gradients through the original residual scorer')
    result = dict(problem=problem, microbatch=cap, maximum_n=int(raw['n'].max()),
        minimum_n=int(raw['n'].min()), dropout_forwards=1,
        gradient_scalar='mean squared legal residual utilities, no labels or costs', synthetic_gradient_scale=1024.,
        gradient_tensors={key: len(value) for key, value in groups.items()}, nonzero_gradient_tensors=nonzero,
        seconds=time.monotonic() - started, peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
        peak_reserved_mib=torch.cuda.max_memory_reserved() / 2**20)
    model.zero_grad(set_to_none=True)
    return result


def execute(args):
    if not args.name.replace('_', '').replace('-', '').isalnum():
        raise ValueError('Use a simple witness name')
    shapes = [('TSP', args.large_cap), ('CVRP', args.large_cap), ('ATSP', args.other_cap), ('VRPBLTW', args.other_cap)]
    directory = ROOT / 'preparation/kernel_witness' / args.name
    if any((directory / name).exists() for name in ('started.json', 'complete.json', 'error.json')):
        raise RuntimeError('Already attempted; preserve original evidence')
    with reservation(args.arm) as allocation:
        metadata = dict(experiment='R108', arm=args.arm, started_utc=resources.utc(),
            scope='input-driven single-view memory and gradient witness', resources=allocation,
            optimizer_steps=0, formal_train_started=False, TEST_or_LIB_opened=False,
            labels_used_for_scalar_or_cap_selection=False,
            TRAIN_target_arrays_mapped_by_original_cache_loader=True,
            row_selection='largest true-N TRAIN rows; never costs or winners', automatic_retry=False,
            source_sha256={p.name: resources.sha(p) for p in (Path(__file__), V4 / 'r108_model.py')})
        resources.save(directory / 'started.json', metadata, exclusive=True)
        try:
            torch.set_num_threads(2)
            inputs.configure_torch()
            initial = torch.load(PREPARED / 'identicalinit_seed2.pt', map_location='cpu', weights_only=False)
            model, _, initialization = models.initialize_shared(initial,
                resources.read(PREPARED / 'geometry_stats.json'), args.arm, 'cuda:0',
                priors=resources.read(ROOT / 'prepared/priors.json'))
            del initial
            manifest = resources.read(PREPARED / 'source_cache_manifest.json')
            rows = []
            for problem, cap in shapes:
                cache = inputs.open_cache(CACHE, 'train', problem, manifest, DATA)
                try:
                    indices = np.argsort(cache.lengths, kind='stable')[-cap:].copy()
                    batch = cache.batch(indices, device='cuda:0')
                    raw = inputs_only(batch)
                    del batch
                    rows.append(measure(model, raw, problem, cap))
                    resources.save(directory / (problem + '.json'), rows[-1], exclusive=True)
                    print(json.dumps(rows[-1]), flush=True)
                    del raw
                    torch.cuda.empty_cache()
                finally:
                    cache.close()
            result = dict(metadata, complete=True, exit_status=0, completed_utc=resources.utc(),
                initialization=initialization, measurements=rows,
                total_device_memory_mib=torch.cuda.get_device_properties(0).total_memory / 2**20,
                no_training_or_evaluation_metrics=True)
            resources.save(directory / 'complete.json', result, exclusive=True)
            return result
        except BaseException as error:
            resources.save(directory / 'error.json', dict(metadata, complete=False, exit_status=1,
                type=type(error).__name__, error=str(error), completed_utc=resources.utc()), exclusive=True)
            raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', choices=models.ARMS, required=True)
    parser.add_argument('--name', default='primary')
    parser.add_argument('--large-cap', type=int, default=512)
    parser.add_argument('--other-cap', type=int, default=1024)
    return execute(parser.parse_args(argv))


if __name__ == '__main__':
    main()


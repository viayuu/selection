"""One input-driven R106 memory/gradient witness; never a training run."""

import argparse
import json
import re
from pathlib import Path
import time
from uuid import UUID

import numpy as np
import torch

from . import r84_experiment as inputs
from . import r84_launch as resources
from . import r106_model as models
from .pairwise_objective import inputs_only
from .r102_gpu_recovery_entry import reservation


V4 = Path(__file__).resolve().parent
ROOT = V4 / 'runs/R106_native_encoder_adaptation'
PREPARED = V4 / 'runs/R84_new90k_attention/prepared'
DATA = Path('/public/home/shiys/selection_data_new90k_B_filtered_single_reld_mt_seed2_20261018')
CACHE = Path('/tmp/shiys_R84_new90k_cache')
SHAPES = (('TSP', 64), ('CVRP', 64), ('ATSP', 128), ('VRPBLTW', 256))


def normalized_uuid(value):
    return UUID(str(value).removeprefix('GPU-'))


def measure(selector, raw, problem, cap):
    selector.train()
    selector.zero_grad(set_to_none=True)
    device = next(selector.parameters()).device
    calls = []
    hook = selector.donor.register_forward_hook(lambda *args: calls.append(1))
    try:
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        started = time.monotonic()
        with selector.native_pair_context(raw):
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                first, second = selector(raw), selector(raw)
            # Synthetic input-only gradient scalar, not an accuracy/loss result.
            loss = .5 * (first['logits'][first['solver_mask']].float().square().mean()
                         + second['logits'][second['solver_mask']].float().square().mean())
        (loss * 1024.).backward()
        torch.cuda.synchronize(device)
        seconds = time.monotonic() - started
        groups = dict(native=[], selector=[])
        nonzero = dict(native=0, selector=0)
        for parameter in selector.parameters():
            if parameter.grad is None:
                continue
            group = 'native' if getattr(parameter, '_r106_native', False) else 'selector'
            groups[group].append(bool(torch.isfinite(parameter.grad).all()))
            nonzero[group] += int(bool(parameter.grad.abs().max() > 0))
        if (len(calls) != 1 or not torch.isfinite(loss)
                or any(not flags or not all(flags) or not nonzero[group] for group, flags in groups.items())):
            raise RuntimeError('Need one real native call and finite nonzero gradients on both paths')
        return dict(problem=problem, microbatch=cap, maximum_n=int(raw['n'].max()),
            minimum_n=int(raw['n'].min()), native_calls=len(calls),
            native_gradient_tensors=len(groups['native']), selector_gradient_tensors=len(groups['selector']),
            nonzero_gradient_tensors=nonzero, two_dropout_forwards=True,
            gradient_scalar='mean squared legal logits; no target labels',
            synthetic_gradient_scale=1024.,
            seconds=seconds, peak_allocated_mib=torch.cuda.max_memory_allocated(device) / 2**20,
            peak_reserved_mib=torch.cuda.max_memory_reserved(device) / 2**20)
    finally:
        hook.remove()
        selector.zero_grad(set_to_none=True)


def execute(args):
    if not re.fullmatch(r'[A-Za-z0-9_-]+', args.name):
        raise ValueError('Use a simple local witness name')
    caps = dict(TSP=args.large_cap, CVRP=args.large_cap, ATSP=args.atsp_cap, VRPBLTW=args.other_cap)
    shapes = [(problem, caps[problem]) for problem, _ in SHAPES
              if args.problem is None or problem == args.problem]
    if any(cap <= 0 or cap > 1024 for _, cap in shapes):
        raise ValueError('Witness microcaps must be within logical batch1024')
    directory = ROOT / 'preparation/kernel_witness' / args.name
    if any((directory / name).exists() for name in ('started.json', 'complete.json', 'error.json')):
        raise RuntimeError('Witness already attempted; preserve the original evidence')
    with reservation(args.slot_arm) as allocation:
        expected = resources.SLOTS[allocation['slot']]['uuid']
        if normalized_uuid(torch.cuda.get_device_properties(0).uuid) != normalized_uuid(expected):
            raise RuntimeError('Actual CUDA hardware differs from the reserved UUID')
        metadata = dict(experiment='R106', arm='B_finetune', scope='input-driven memory and gradient witness',
            started_utc=resources.utc(), resources=allocation, optimizer_steps=0, formal_train_started=False,
            labels_used_for_scalar_or_cap_selection=False, model_selection=False, TEST_or_LIB_opened=False,
            TRAIN_target_arrays_mapped_by_original_cache_loader=True,
            row_selection='largest true N TRAIN rows, stable ordering; never costs or winners',
            automatic_retry=False, shapes=[dict(problem=p, cap=c) for p, c in shapes],
            source_sha256={p.name: resources.sha(p) for p in
                (Path(__file__), V4 / 'r106_model.py', V4 / 'r106_donor.py')})
        resources.save(directory / 'started.json', metadata, exclusive=True)
        try:
            initial = torch.load(PREPARED / 'identicalinit_seed2.pt', map_location='cpu', weights_only=False)
            geometry = resources.read(PREPARED / 'geometry_stats.json')
            model, _, initialization = models.initialize_shared(initial, geometry, 'B_finetune', 'cuda:0')
            del initial
            manifest = resources.read(PREPARED / 'source_cache_manifest.json')
            results = []
            for problem, cap in shapes:
                cache = inputs.open_cache(CACHE, 'train', problem, manifest, DATA)
                try:
                    rows = np.argsort(cache.lengths, kind='stable')[-cap:].copy()
                    batch = cache.batch(rows, device='cuda:0')
                    raw = inputs_only(batch)
                    del batch
                    results.append(measure(model, raw, problem, cap))
                    resources.save(directory / (problem + '.json'), results[-1], exclusive=True)
                    print(json.dumps(results[-1]), flush=True)
                    del raw
                    torch.cuda.empty_cache()
                finally:
                    cache.close()
            result = dict(metadata, complete=True, exit_status=0, completed_utc=resources.utc(),
                initialization=initialization, measurements=results,
                total_device_memory_mib=torch.cuda.get_device_properties(0).total_memory / 2**20,
                no_training_or_evaluation_metrics=True, native_parameter_gradient_path_confirmed=True)
            resources.save(directory / 'complete.json', result, exclusive=True)
            return result
        except BaseException as error:
            resources.save(directory / 'error.json', dict(metadata, complete=False, exit_status=1,
                type=type(error).__name__, error=str(error), completed_utc=resources.utc()), exclusive=True)
            raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--slot-arm', choices=('A', 'B'), default='B')
    parser.add_argument('--name', default='primary')
    parser.add_argument('--large-cap', type=int, default=64)
    parser.add_argument('--atsp-cap', type=int, default=128)
    parser.add_argument('--other-cap', type=int, default=256)
    parser.add_argument('--problem', choices=[p for p, _ in SHAPES])
    args = parser.parse_args(argv)
    execute(args)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

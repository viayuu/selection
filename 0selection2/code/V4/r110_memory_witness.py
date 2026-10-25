"""Input-only one-forward witness for the R110 graph models on allocated4090."""

import argparse
import importlib.util
import json
from pathlib import Path
import time

import numpy as np
import torch

from . import r109_model as models
from .pairwise_objective import inputs_only


V4 = Path(__file__).resolve().parent
ROOT = V4 / 'runs/R110_method_conditioned_ce'
spec = importlib.util.spec_from_file_location('code.V4._r110_memory_resources', V4 / 'r108_memory_witness.py')
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
ARMS = ('A_tied_ce', 'B_private_ce')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', choices=ARMS, required=True)
    parser.add_argument('--name', default='primary')
    parser.add_argument('--large-cap', type=int, default=256)
    parser.add_argument('--other-cap', type=int, default=1024)
    args = parser.parse_args(argv)
    if not args.name.replace('_', '').replace('-', '').isalnum():
        raise ValueError('Invalid witness name')
    directory = ROOT / 'preparation/kernel_witness' / args.name
    if any((directory / p).exists() for p in ('started.json', 'complete.json', 'error.json')):
        raise RuntimeError('Preserve original witness evidence')
    base.SLOTS[args.arm] = dict(zip(ARMS, ('4090_0', '4090_1')))[args.arm]
    with base.reservation(args.arm) as allocation:
        metadata = dict(experiment='R110', arm=args.arm, resources=allocation,
            optimizer_steps=0, formal_train_started=False, TEST_or_LIB_opened=False,
            labels_used_for_scalar_or_cap_selection=False, started_utc=base.resources.utc())
        base.resources.save(directory / 'started.json', metadata, exclusive=True)
        try:
            torch.set_num_threads(2)
            base.inputs.configure_torch()
            initial = torch.load(base.PREPARED / 'identicalinit_seed2.pt', map_location='cpu', weights_only=False)
            model, _, receipt = models.initialize_shared(initial,
                base.resources.read(base.PREPARED / 'geometry_stats.json'),
                models.ARMS[ARMS.index(args.arm)], 'cuda:0')
            manifest = base.resources.read(base.PREPARED / 'source_cache_manifest.json')
            measurements = []
            for problem, cap in (('TSP', args.large_cap), ('CVRP', args.large_cap),
                                 ('ATSP', args.other_cap), ('VRPBLTW', args.other_cap)):
                cache = base.inputs.open_cache(base.CACHE, 'train', problem, manifest, base.DATA)
                try:
                    rows = np.argsort(cache.lengths, kind='stable')[-cap:].copy()
                    raw = inputs_only(cache.batch(rows, device='cuda:0'))
                    model.train().zero_grad(set_to_none=True)
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                    started = time.monotonic()
                    with torch.autocast(device_type='cuda', dtype=torch.float16):
                        output = model(raw)
                    loss = output['utility'][output['solver_mask']].float().square().mean()
                    (loss * 1024).backward()
                    torch.cuda.synchronize()
                    groups = {'original': [], 'late': []}
                    for name, p in model.named_parameters():
                        if p.grad is not None:
                            groups['late' if name.startswith('late_banks.') else 'original'].append(p.grad)
                    if not torch.isfinite(loss) or any(not g or not all(torch.isfinite(t).all() for t in g)
                        or not any(bool(t.abs().max() > 0) for t in g) for g in groups.values()):
                        raise RuntimeError('Finite gradients required through shared and candidate graph paths')
                    record = dict(problem=problem, microbatch=cap, maximum_n=int(raw['n'].max()),
                        peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
                        peak_reserved_mib=torch.cuda.max_memory_reserved() / 2**20,
                        seconds=time.monotonic() - started, dropout_forwards=1,
                        finite_nonzero_shared_and_late_gradients=True)
                    measurements.append(record)
                    base.resources.save(directory / (problem + '.json'), record, exclusive=True)
                    print(json.dumps(record), flush=True)
                    model.zero_grad(set_to_none=True)
                    del output, loss, raw, groups
                    torch.cuda.empty_cache()
                finally:
                    cache.close()
            base.resources.save(directory / 'complete.json', dict(metadata, complete=True,
                exit_status=0, measurements=measurements, initialization=receipt), exclusive=True)
        except BaseException as error:
            base.resources.save(directory / 'error.json', dict(metadata, complete=False,
                type=type(error).__name__, error=str(error)), exclusive=True)
            raise


if __name__ == '__main__':
    main()

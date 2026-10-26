"""Small input-only two-view witness, inside the authorized warm GPU allocation."""

import argparse
import json
import time

import numpy as np
import torch

from . import r109_model as models
from . import r107_memory_witness as base
from .pairwise_objective import inputs_only


ROOT = base.V4 / 'runs/R109_method_conditioned_nodes'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', choices=models.ARMS, required=True)
    parser.add_argument('--name', default='primary')
    parser.add_argument('--large-cap', type=int, default=64)
    parser.add_argument('--other-cap', type=int, default=256)
    args = parser.parse_args(argv)
    if not args.name.replace('_', '').replace('-', '').isalnum():
        raise ValueError('Invalid witness name')
    directory = ROOT / 'preparation/kernel_witness' / args.name
    if any((directory / p).exists() for p in ('started.json', 'complete.json', 'error.json')):
        raise RuntimeError('Preserve original witness evidence')
    assigned = dict(zip(models.ARMS, base.SLOTS.values()))[args.arm]
    base.SLOTS[args.arm] = assigned
    with base.reservation(args.arm) as allocation:
        metadata = dict(experiment='R109', arm=args.arm, resources=allocation,
            optimizer_steps=0, formal_train_started=False, TEST_or_LIB_opened=False,
            labels_used_for_scalar_or_cap_selection=False, started_utc=base.resources.utc())
        base.resources.save(directory / 'started.json', metadata, exclusive=True)
        try:
            torch.set_num_threads(2)
            base.inputs.configure_torch()
            initial = torch.load(base.PREPARED / 'identicalinit_seed2.pt', map_location='cpu', weights_only=False)
            model, _, receipt = models.initialize_shared(initial,
                base.resources.read(base.PREPARED / 'geometry_stats.json'), args.arm, 'cuda:0')
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
                        first, second = model(raw), model(raw)
                    loss = .5 * (first['logits'][first['solver_mask']].float().square().mean()
                               + second['logits'][second['solver_mask']].float().square().mean())
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
                        seconds=time.monotonic() - started, two_dropout_forwards=True,
                        finite_nonzero_shared_and_late_gradients=True)
                    measurements.append(record)
                    base.resources.save(directory / (problem + '.json'), record, exclusive=True)
                    print(json.dumps(record), flush=True)
                    model.zero_grad(set_to_none=True)
                    del first, second, loss, raw, groups
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

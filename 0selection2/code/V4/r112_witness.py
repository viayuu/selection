"""Disposable real-input memory/timing check; never writes model weights."""

import argparse
import gc
import json
from pathlib import Path
import socket
import time

import numpy as np
import torch

from .r112_model import ARMS, initialize_shared
from .r76_cache import ScenarioCache
from .r76_experiment import RDropObjective, slice_batch, utc, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache-root', type=Path, default=Path('/tmp/shiys_R84_new90k_cache'))
    parser.add_argument('--prepared-root', type=Path, default=Path(__file__).with_name('runs') / 'R84_new90k_attention/prepared')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--caps', type=int, nargs='+', default=[64, 128])
    parser.add_argument('--problems', nargs='+', default=['TSP', 'CVRP', 'ATSP', 'OVRPBLTW'])
    parser.add_argument('--batch-size', type=int, default=512)
    args = parser.parse_args()
    torch.set_num_threads(2)
    if not torch.cuda.is_available():
        raise RuntimeError('Run the bounded witness inside the allocated GPU container')
    initial = torch.load(args.prepared_root / 'identicalinit_seed2.pt', map_location='cpu', weights_only=False)
    geometry = json.loads((args.prepared_root / 'geometry_stats.json').read_text())
    result = dict(started_utc=utc(), hostname=socket.gethostname(),
        cgroup=Path('/proc/self/cgroup').read_text(), device=torch.cuda.get_device_name(0),
        logical_batch=args.batch_size, scratch_only=True, writes_weights=False, trials=[])
    for problem in args.problems:
        cache = ScenarioCache(args.cache_root / 'train' / problem)
        # Input size only: deliberately stress the longest real TRAIN graphs.
        rows = np.argsort(np.asarray(cache.lengths), kind='stable')[-args.batch_size:]
        batch = cache.batch(rows, device='cuda:0')
        for cap in args.caps:
            if cap > args.batch_size:
                continue
            for arm in ARMS:
                model = objective = optimizer = scaler = None
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                trial = dict(problem=problem, arm=arm, cap=cap, rows=len(rows), max_nodes=int(max(cache.lengths[rows])))
                try:
                    model, _, _ = initialize_shared(initial, geometry, arm, 'cuda:0')
                    model.train()
                    objective = RDropObjective(model)
                    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
                    scaler = torch.amp.GradScaler('cuda', init_scale=1024.)
                    torch.cuda.synchronize()
                    started, loss_sum = time.perf_counter(), 0.
                    for start in range(0, len(rows), cap):
                        stop = min(start + cap, len(rows))
                        loss, _ = objective(slice_batch(batch, start, stop), .35)
                        if not torch.isfinite(loss):
                            raise RuntimeError('Nonfinite real-input loss')
                        scaler.scale(loss * ((stop - start) / len(rows))).backward()
                        loss_sum += loss.detach().item() * ((stop - start) / len(rows))
                    scaler.unscale_(optimizer)
                    norm = torch.nn.utils.clip_grad_norm_(objective.parameters(), 1.)
                    if not torch.isfinite(norm):
                        raise RuntimeError('Nonfinite real-input gradient')
                    scaler.step(optimizer)
                    scaler.update()
                    torch.cuda.synchronize()
                    trial.update(ok=True, seconds=time.perf_counter() - started, loss=loss_sum,
                        gradient_norm=norm.item(), allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                        reserved_gib=torch.cuda.max_memory_reserved() / 2**30)
                except torch.cuda.OutOfMemoryError as error:
                    trial.update(ok=False, error_type='CUDA_OOM', error=str(error))
                finally:
                    result['trials'].append(trial)
                    write_json(args.output, result)
                    print(json.dumps(trial, allow_nan=False), flush=True)
                    del model, objective, optimizer, scaler
                    gc.collect()
                    torch.cuda.empty_cache()
        del batch
        cache.close()
    result['finished_utc'] = utc()
    write_json(args.output, result)


if __name__ == '__main__':
    main()

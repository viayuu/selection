"""One disposable capacity check on an already assigned GPU; no saved weights."""

import argparse
import gc
import json
from pathlib import Path
from types import SimpleNamespace
import socket
import time

import numpy as np
import torch

from . import r113_experiment as runner
from . import r113_pretrain as process
from .r76_cache import ScenarioCache
from .r76_experiment import PART_KEYS, RDropObjective, file_hash, read_json, utc, write_json


def run(args):
    lock = runner.protocol_lock(args)
    caps, pretrain_caps, _ = runner.locked_caps(lock['record'])
    if args.output.exists():
        raise FileExistsError('Preserve the previous capacity receipt: ' + str(args.output))
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('Capacity check requires one scheduler-bound allocated GPU')
    torch.set_num_threads(2)
    runner.engine.configure_torch()
    initial = torch.load(args.prepared_root / 'identicalinit_seed2.pt',
                         map_location='cpu', weights_only=False)
    geometry = read_json(args.prepared_root / 'geometry_stats.json')
    device = torch.device('cuda:0')
    result = dict(experiment='R113_capacity', complete=False, started_utc=utc(),
        main_protocol_sha256=lock['sha256'], assigned_arm=args.arm,
        device=torch.cuda.get_device_name(0), hostname=socket.gethostname(),
        cgroup=Path('/proc/self/cgroup').read_text(), logical_batch=runner.BATCH_SIZE,
        scratch_only=True, saved_weights=False, formal_updates=0, trials=[],
        source_sha256={name: file_hash(runner.V4 / name) for name in runner.SOURCE_FILES})
    write_json(args.output, result)
    model = objective = optimizer = scaler = batch = dataset = None
    try:
        # All three arms deploy the identical architecture; test it once per job.
        for problem in ('TSP', 'CVRP', 'ATSP', 'OVRPBLTW'):
            gc.collect()
            torch.cuda.empty_cache()
            with_cache = ScenarioCache(args.cache_root / 'train' / problem)
            try:
                rows = np.argsort(np.asarray(with_cache.lengths), kind='stable')[-runner.BATCH_SIZE:]
                maximum = int(np.max(with_cache.lengths[rows]))
                batch = with_cache.batch(rows, device=device)
            finally:
                with_cache.close()
            model, _, _ = runner.initialize_shared(initial, geometry, 'A_scratch', device)
            model.train()
            objective = RDropObjective(model)
            optimizer = torch.optim.AdamW(model.parameters(), lr=runner.PEAK_LR, weight_decay=1e-4)
            scaler = torch.amp.GradScaler('cuda', init_scale=1024.)
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            start = time.perf_counter()
            parts, retries, _ = runner.update_batch(objective, optimizer, scaler, batch, .35,
                SimpleNamespace(device=device, world_size=1), caps[problem], problem=problem)
            torch.cuda.synchronize()
            if not torch.isfinite(parts).all():
                raise RuntimeError('Nonfinite real-input capacity update')
            result['trials'].append(dict(stage='selector', problem=problem, cap=caps[problem],
                rows=len(rows), max_nodes=maximum, seconds=time.perf_counter() - start,
                amp_retries=retries, finite=True,
                parts={key: float(value) for key, value in zip(PART_KEYS, parts)},
                allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                reserved_gib=torch.cuda.max_memory_reserved() / 2**30))
            write_json(args.output, result)
            del model, objective, optimizer, scaler, batch
            model = objective = optimizer = scaler = batch = None

        if args.arm in process.ARMS:
            gc.collect()
            torch.cuda.empty_cache()
            dataset = process.ProcessDataset(lock['record']['process_manifest'], args.data_root, cache_shards=2)
            program, targets, _ = dataset.batch('OVRPBLTW', np.arange(process.BATCH_SIZE), device=device)
            model, _ = process.extract_processor(initial, geometry, args.arm, device)
            model.train()
            optimizer = torch.optim.AdamW(model.parameters(), lr=process.PEAK_LR, weight_decay=1e-4)
            scaler = torch.amp.GradScaler('cuda', init_scale=1024.)
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            start = time.perf_counter()
            losses, retries, norm = process.update_batch(model, optimizer, scaler, program, targets,
                process.MODE[args.arm], pretrain_caps['OVRPBLTW'])
            torch.cuda.synchronize()
            result['trials'].append(dict(stage='process_pretraining', problem='OVRPBLTW',
                mode=process.MODE[args.arm], cap=pretrain_caps['OVRPBLTW'], rows=process.BATCH_SIZE,
                seconds=time.perf_counter() - start, amp_retries=retries, finite=True,
                losses=losses, gradient_norm=norm,
                allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                reserved_gib=torch.cuda.max_memory_reserved() / 2**30))
        result.update(complete=True, finished_utc=utc())
        write_json(args.output, result)
        return result
    except BaseException as error:
        result.update(complete=False, failed_utc=utc(), error_type=type(error).__name__,
                      error=str(error), automatic_retry=False)
        write_json(args.output, result)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=runner.DEFAULT_ROOT)
    parser.add_argument('--arm', choices=runner.ARMS, required=True)
    parser.add_argument('--protocol-lock', type=Path)
    parser.add_argument('--prepared-root', type=Path, default=runner.DEFAULT_PREPARED_ROOT)
    parser.add_argument('--data-root', type=Path, default=runner.DEFAULT_DATA_ROOT)
    parser.add_argument('--cache-root', type=Path, default=runner.DEFAULT_CACHE_ROOT)
    parser.add_argument('--pretrain-root', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    args.pretrain_root = args.pretrain_root or args.root / 'pretrain'
    args.large_micro_cap = args.eval_batch_size = None
    for key, value in vars(args).copy().items():
        if isinstance(value, Path):
            setattr(args, key, value.expanduser().resolve())
    result = run(args)
    print(json.dumps(dict(complete=result['complete'], trials=len(result['trials'])), allow_nan=False))


if __name__ == '__main__':
    main()

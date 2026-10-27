"""One four-corner native/selector capacity witness; no saved training weights."""

import argparse
from pathlib import Path
import time

import numpy as np
import torch

from code.unified_selector.registry import PROBLEMS
from .r76_experiment import open_cache, read_json, write_json
from .r84_prepare import DEFAULT_DATA_ROOT, DEFAULT_PREPARED_ROOT
from .r117_model import make_model, inputs_with_donor


CAPS = {p: 256 if p in ('TSP', 'CVRP', 'ATSP') else 512 for p in PROBLEMS}


def capacity(output):
    from .r117_objective import RDropObjective
    torch.set_num_threads(2)
    if not torch.cuda.is_available() or 'RTX 3090' not in torch.cuda.get_device_name(0):
        raise RuntimeError('Use only a launcher-authorized RTX3090')
    device = torch.device('cuda:0')
    initial = torch.load(DEFAULT_PREPARED_ROOT / 'identicalinit_seed2.pt', map_location='cpu', weights_only=False)
    geometry = read_json(DEFAULT_PREPARED_ROOT / 'geometry_stats.json')
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(2)
        model = make_model(initial['model_params'], 'B')
    with torch.no_grad():
        for name, value in model.named_parameters():
            if name in initial['trainable']:
                value.copy_(initial['trainable'][name])
        branch = model.instance_encoder.geometry_residual
        branch.mean.copy_(torch.as_tensor(geometry['mean'], dtype=branch.mean.dtype))
        branch.std.copy_(torch.as_tensor(geometry['std'], dtype=branch.std.dtype))
    model.to(device)
    objective = RDropObjective(model)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=1e-4, weight_decay=1e-4)
    scaler = torch.amp.GradScaler('cuda')
    manifest = read_json(DEFAULT_PREPARED_ROOT / 'source_cache_manifest.json')
    started, records = time.monotonic(), []
    for problem in ('TSP', 'CVRP', 'ATSP', 'OVRPBLTW'):
        cache = open_cache(Path('/tmp/shiys_R84_new90k_cache'), 'train', problem, manifest, DEFAULT_DATA_ROOT)
        try:
            indices = np.argsort(cache.lengths, kind='stable')[-CAPS[problem]:].copy()
            batch = cache.batch(indices, device=device)
            small = {key: (value[:8] if torch.is_tensor(value) and value.ndim and value.shape[0] == len(indices)
                           and key != 'pool_ids' else value) for key, value in inputs_with_donor(batch).items()}
            # Native extraction has its own bounded memory scope, without any labels.
            model.eval()
            tick = time.monotonic()
            with torch.no_grad():
                native = model.donor(small)
            native_seconds = time.monotonic() - tick
            for key, source in (('nco_nodes', 'base_nodes'), ('native_nodes', 'native_nodes')):
                value = native[source].to(torch.float16)
                repeats = (len(indices) + len(value) - 1) // len(value)
                batch[key] = value.repeat(repeats, 1, 1)[:len(indices)]
            batch['nco_node_mask'] = batch['native_node_mask'] = batch['node_mask']
            model.train()
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
            tick = time.monotonic()
            loss, _ = objective(batch, .35)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            if not torch.isfinite(loss) or not torch.isfinite(gradient):
                raise FloatingPointError('Nonfinite original two-dropout loss/gradient')
            scaler.step(optimizer)
            scaler.update()
            torch.cuda.synchronize(device)
            row = dict(problem=problem, microbatch=CAPS[problem], native_rows=8,
                n_max=int(batch['n'].max()), peak_reserved_mib=torch.cuda.max_memory_reserved(device) / 2**20,
                peak_allocated_mib=torch.cuda.max_memory_allocated(device) / 2**20,
                loss=float(loss.detach()), gradient_norm=float(gradient.detach()),
                native_seconds=native_seconds, selector_step_seconds=time.monotonic() - tick)
            print(row, flush=True)
            records.append(row)
            del batch, small, native, loss
        finally:
            cache.close()
    result = dict(complete=True, formal_training_started=False, microbatch_caps=CAPS,
        eval_batch_size=256, native_extraction_batch_size=128, corners=records,
        native_input_only=True, test_read=False, seconds=time.monotonic() - started)
    write_json(output, result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    capacity(args.output)

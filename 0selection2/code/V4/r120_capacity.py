"""Bounded raw-input memory measurements; no optimizer or formal model training."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from code.unified_selector.registry import GLOBAL_SOLVERS
from .nonrouting_data import NonRoutingDataset, collate_nonrouting
from .r120_model import MultidomainSelector, default_config
from .r120_launch import write


REPO = Path('/public/home/shiys/0selection2')
READY = REPO / 'code/V4/runs/R84_new90k_attention/prepared'
NR = Path('/public/home/shiys/selection_data_multidomain_seed2_20261029')
SUPPLEMENT = Path('/public/home/shiys/nonrouting_supplement_20261029/prepared_cpu')


def read(path):
    return json.loads(Path(path).read_text())


def to_device(value, device):
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {key: to_device(item, device) for key, item in value.items()}
    return value


def representative(problem, size, cache_root, config):
    if problem in ('TSP', 'ATSP', 'VRPBLTW'):
        from .r76_experiment import open_cache
        from .r120_experiment import ROUTING_INPUT_KEYS
        manifest = read(READY / 'source_cache_manifest.json')
        ready = read(READY / 'prepared.json')
        cache = open_cache(cache_root, 'train', problem, manifest, ready['data_root'])
        try:
            index = int(np.argmax(cache.lengths))
            old = cache.batch([index] * size, device='cpu')
            inputs = {key: old[key] for key in ROUTING_INPUT_KEYS if key in old}
        finally:
            cache.close()
    else:
        root = NR if problem in ('MIS', 'JSSP', 'FJSP') else SUPPLEMENT
        dataset = NonRoutingDataset(root, problem, 'train', allow_incomplete=True)
        try:
            key = {'MIS': 'edge_index', 'JSSP': 'duration', 'FJSP': 'processing_time',
                   'MKP': 'resources'}[problem]
            index = int(np.diff(dataset._offsets[key]).argmax())
            sample = dataset[index]
            inputs = collate_nonrouting([sample] * size)['inputs']
        finally:
            dataset.close()
    return dict(problem=problem, inputs=inputs, pool_ids=torch.tensor(config['pools'][problem]))


def one_trial(selector, batch):
    selector.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.autocast('cuda', dtype=torch.float16):
        output = selector(batch)
    # Only measures the graph needed by both real readout heads. This number
    # is never a quality metric, a training update, or a replacement label.
    loss = output['utility'].float().square().mean() + .25 * output['advantage'].float().square().mean()
    loss.backward()
    torch.cuda.synchronize()
    result = dict(seconds=time.perf_counter() - started,
                  peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
                  peak_reserved_mib=torch.cuda.max_memory_reserved() / 2**20,
                  finite=bool(torch.isfinite(loss).item()))
    selector.zero_grad(set_to_none=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--slot', required=True)
    parser.add_argument('--cache-root', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(2)
    ready = read(NR / 'manifest.json')
    supplement = read(SUPPLEMENT / 'manifest.json')
    merged = dict(ready['nonrouting'], **supplement['nonrouting'])
    manifest = read(READY / 'source_cache_manifest.json')
    names = list(GLOBAL_SOLVERS) + sorted({f'NR::{p}::{name}' for p, entry in merged.items()
        for name in entry['candidate_solver_ids']})
    config = default_config(names)
    config['pools'] = {p: [names.index(f'NR::{p}::{name}') for name in entry['candidate_solver_ids']]
                       for p, entry in merged.items()}
    config['pools'].update({p: record['pool_ids'] for p, record in manifest['splits']['train'].items()})
    selector = MultidomainSelector(config).cuda().train()
    selector.frontend.set_routing_geometry(read(READY / 'geometry_stats.json'))
    device = torch.cuda.get_device_properties(0)
    maximum = 320 if '4090' in device.name else 192
    result = dict(purpose='capacity only; no optimizer updates or quality results',
                  slot=args.slot, device=device.name, uuid=str(device.uuid), tasks={})
    initial = dict(TSP=32, ATSP=32, VRPBLTW=32, MIS=8, JSSP=16, FJSP=32, MKP=32)
    for problem, count in initial.items():
        batch = to_device(representative(problem, count, args.cache_root, config), torch.device('cuda:0'))
        measured = one_trial(selector, batch)
        del batch
        target = .78 * device.total_memory / 2**20
        proposed = min(maximum, max(count, int(count * target / max(measured['peak_allocated_mib'], 1) / 8) * 8))
        row = dict(initial_batch=count, initial=measured, proposed_batch=proposed, chosen_batch=count)
        if proposed > count:
            try:
                batch = to_device(representative(problem, proposed, args.cache_root, config), torch.device('cuda:0'))
                confirmation = one_trial(selector, batch)
                row['confirmation'] = confirmation
                if confirmation['finite'] and confirmation['peak_reserved_mib'] < .9 * device.total_memory / 2**20:
                    row['chosen_batch'] = proposed
            except torch.cuda.OutOfMemoryError as error:
                row['confirmation_oom'] = str(error).splitlines()[0]
                selector.zero_grad(set_to_none=True)
            finally:
                if 'batch' in locals():
                    del batch
                torch.cuda.empty_cache()
        result['tasks'][problem] = row
        write(args.root / 'resources' / f'capacity_{args.slot}.json', result)
        print(json.dumps(dict(problem=problem, **row)), flush=True)
    result['complete'] = True
    write(args.root / 'resources' / f'capacity_{args.slot}.json', result)


if __name__ == '__main__':
    main()

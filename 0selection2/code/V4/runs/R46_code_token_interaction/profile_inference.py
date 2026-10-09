"""Post-selection latency comparison on fixed validation inputs, one authorized GPU."""

import json
from pathlib import Path

import numpy as np
import torch

from code.unified_selector.registry import GLOBAL_SOLVERS
from code.V4.multitask_probe import dump, file_hash, gather_batch
from code.V4.pairwise_objective import inputs_only
from code.V4.performance_analysis import write_csv
from code.V4.r41_frozen_evaluation import attach_geometry
from code.V4.r45_experiment import ROOT as R45_ROOT
from code.V4.r46_experiment import ROOT, RUN_NAME
from code.V4.r46_model import load_r46_checkpoint
from code.V4.solver_code_encoder import load_r45_checkpoint
from code.V4.train import configure_torch, make_loader


@torch.no_grad()
def main():
    if not (ROOT / 'test_results.json').exists():
        raise ValueError('Run only after validation selection and test are complete')
    configure_torch()
    torch.set_num_threads(4)
    device = 'cuda:0'
    batches = {}
    for problem in ('TSP', 'CVRP', 'ATSP', 'OVRPTW'):
        loader = make_loader(problem, 'val', 128, 0, shuffle=False, cache_device=device)
        attach_geometry(loader, device)
        batches[problem] = inputs_only(gather_batch(loader, torch.arange(128)))
    entries = [('R45A', R45_ROOT / 'A_handcrafted_seed2/best.pt', load_r45_checkpoint),
               ('R45B', R45_ROOT / 'B_code_seed2/best.pt', load_r45_checkpoint),
               ('R46A', ROOT / RUN_NAME / 'best.pt', load_r46_checkpoint)]
    rows, dynamic = [], []
    for name, path, load in entries:
        model, _ = load(path, device)
        for problem, batch in batches.items():
            for _ in range(3):
                model(batch)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            times = []
            for _ in range(20):
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                model(batch)
                end.record()
                end.synchronize()
                times.append(start.elapsed_time(end))
            row = dict(model=name, checkpoint_sha256=file_hash(path), problem=problem, batch=128,
                       max_nodes=int(batch['n'].max()), median_ms=float(np.median(times)),
                       p90_ms=float(np.quantile(times, .9)), peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
            rows.append(row)
            print(json.dumps(row), flush=True)
            if name == 'R46A':
                small = {key: value[:2] if isinstance(value, torch.Tensor) and value.ndim and value.shape[0] == 128 else value
                         for key, value in batch.items()}
                _, codes, h, _, _, mask = model.encode_source_state(small)
                solvers, weights = model.source_pool(h, codes, mask)
                for j, identity in enumerate(small['pool_ids'].tolist()):
                    valid = mask[0, j]
                    dynamic.append(dict(problem=problem, solver=GLOBAL_SOLVERS[identity],
                         source_state_l2_difference=float((codes[0, j, valid]-codes[1, j, valid]).norm()),
                         solver_state_l2_difference=float((solvers[0, j]-solvers[1, j]).norm()),
                         pooling_weight_l1_difference=float((weights[0, j]-weights[1, j]).abs().sum())))
        del model
        torch.cuda.empty_cache()
    write_csv(ROOT / 'inference_latency.csv', rows)
    dump(ROOT / 'dynamic_evidence.json', dict(inputs='validation instances0 and1; same solver identities', values=dynamic))
    dump(ROOT / 'inference_latency_protocol.json', dict(gpu=torch.cuda.get_device_name(0),
         inputs='first128 validation instances per problem; identical across models',
         evaluation_dtype='FP32, historical TF32/SDPA; no AMP/dropout', warmup=3, repeats=20,
         timing='CUDA events, forward only; excludes input loading, geometry caching, disk writes',
         source_sha256=file_hash(Path(__file__))))
    lines = ['# Inference latency on the authorized RTX3090', '',
             'FP32 eval; same128 validation instances; median of20 CUDA-event measurements after3 warmups.',
             'Forward only; no data preparation, geometry caching or I/O. Peak allocation includes the resident input batches.', '',
             '| Model | Problem | Batch | Max nodes | Median ms | p90 ms | Peak GiB |',
             '|---|---|---:|---:|---:|---:|---:|']
    for row in rows:
        lines.append(f'| {row["model"]} | {row["problem"]} | 128 | {row["max_nodes"]} | '
                     f'{row["median_ms"]:.3f} | {row["p90_ms"]:.3f} | {row["peak_allocated_gib"]:.3f} |')
    (ROOT / 'inference_latency.md').write_text('\n'.join(lines)+'\n')
    audit = json.loads((ROOT / 'source_package_audit.json').read_text())
    collisions = []
    for problem, groups in audit['identical_in_pool'].items():
        with np.load(ROOT / 'test_predictions/A' / f'{problem}.npz') as saved:
            names = saved['pool'].tolist()
            for first, second in groups:
                a, b = names.index(first), names.index(second)
                difference = np.abs(saved['logits'][:, a]-saved['logits'][:, b])
                collisions.append(dict(problem=problem, first=first, second=second,
                    first_winner_count=int((saved['winner'] == a).sum()), second_winner_count=int((saved['winner'] == b).sum()),
                    first_pick_count=int((saved['pred'] == a).sum()), second_pick_count=int((saved['pred'] == b).sum()),
                    max_absolute_logit_difference=float(difference.max()), exactly_equal_count=int((difference == 0).sum())))
    write_csv(ROOT / 'identical_source_predictions.csv', collisions)


if __name__ == '__main__':
    main()

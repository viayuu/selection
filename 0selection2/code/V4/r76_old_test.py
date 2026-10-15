"""Evaluate the locked R76 checkpoint on the original, pool-aligned TEST.

This is an additional cross-scenario diagnostic, not another checkpoint search.
Historical solver costs are retained; only candidate identities and derived
targets change. The formal R76 TEST artifacts are never modified.
"""

import argparse
import csv
import json
import socket
import time
from pathlib import Path

import numpy as np
import torch

from code.unified_selector.registry import GLOBAL_SOLVERS, PROBLEMS
from .new_split_data import scenario_solver_ids
from .pairwise_objective import inputs_only
from .r40_experiment import aggregate, families
from .r75_cache import ProblemCache
from .r76_experiment import (
    PinnedPrefetch, file_hash, fixed_sbs_metrics, load_r76_checkpoint,
    read_json, utc, write_json,
)
from .train import configure_torch


DEFAULT_RUN = Path('code/V4/runs/R76_R45A_single_reld_mt')
DEFAULT_REFERENCE = Path('code/V4/runs/R45_solver_code_embeddings')
DEFAULT_SCENARIO = Path(
    '/public/home/shiys/selection_data_new_only_B_filtered_single_reld_mt_seed2_20261014')


def align_labels(old, methods, pool_ids, aliases):
    """Select cost columns by deployment name, then derive all targets in FP64."""
    source_pool = list(old['pool'])
    source_names = [aliases.get(name, name) for name in methods]
    if (len(set(source_names)) != len(methods) or not methods
            or pool_ids != sorted(set(pool_ids))):
        raise ValueError('Candidates must have unique deployments and stable global IDs')
    slots = [source_pool.index(name) for name in source_names]
    source_ids = np.asarray(old['pool_ids'], dtype=np.int64)
    if source_ids[slots].tolist() != pool_ids:
        raise ValueError('Cost columns and solver IDs name different deployments')
    costs = np.asarray(old['costs'])[:, slots].copy()
    if costs.dtype != np.float64 or not np.isfinite(costs).all() or np.any(costs <= 0):
        raise ValueError('Historical costs must be positive finite FP64 values')
    order = np.argsort(costs, axis=1, kind='stable')
    winner = order[:, 0]
    oracle = costs[np.arange(len(costs)), winner]
    old_winner = np.asarray(old['winner'], dtype=np.int64)
    source_winner_ids = source_ids[old_winner]
    result = dict(
        costs=costs, winner=winner, cost_order=order,
        ordinal_rank=np.argsort(order, axis=1, kind='stable'), oracle=oracle,
        relative_gap=(costs - oracle[:, None]) / oracle[:, None],
        pool=np.asarray(methods), pool_ids=np.asarray(pool_ids, dtype=np.int64),
        source_slots=np.asarray(slots, dtype=np.int64),
        source_winner=old_winner, source_winner_global_id=source_winner_ids,
        source_pool=np.asarray(source_pool), source_pool_ids=source_ids,
        indices=np.asarray(old['indices'], dtype=np.int64),
        nodes=np.asarray(old['nodes'], dtype=np.int64),
    )
    changed = int(np.count_nonzero(source_winner_ids != result['pool_ids'][winner]))
    return result, changed


def aligned_inputs(batch, pool_ids):
    input_keys = ('problem_id', 'kind', 'd_in', 'n', 'node', 'node_geom', 'matrix',
                  'node_mask', 'cbits', 'problem_desc', 'coord_dist', 'pool_ids',
                  'mask', 'solver_mask')
    clean = {key: value for key, value in inputs_only(batch).items() if key in input_keys}
    device = clean['node_mask'].device
    count = clean['node_mask'].shape[0]
    clean['pool_ids'] = torch.tensor(pool_ids, dtype=torch.long, device=device)
    mask = torch.zeros(len(GLOBAL_SOLVERS), dtype=torch.float32, device=device)
    mask[clean['pool_ids']] = 1
    clean['mask'] = mask.repeat(count, 1)
    clean['solver_mask'] = torch.ones(count, len(pool_ids), dtype=torch.bool, device=device)
    return clean


class AlignedCache:
    def __init__(self, cache, pool_ids):
        self.cache, self.pool_ids = cache, pool_ids

    def batch(self, indices, device='cpu'):
        return aligned_inputs(self.cache.batch(indices, device), self.pool_ids)


def verify_checkpoint(run):
    lock = read_json(run / 'final_test_lock.json')
    checkpoint = run / 'best.pt'
    if not lock['selected_without_test'] or file_hash(checkpoint) != lock['checkpoint_sha256']:
        raise ValueError('The R76 validation-selected checkpoint no longer matches its lock')
    return checkpoint, lock


def prepare(args):
    checkpoint, lock = verify_checkpoint(args.run_root)
    if (args.root / 'protocol.json').exists():
        existing = read_json(args.root / 'protocol.json')
        if existing['checkpoint_sha256'] != lock['checkpoint_sha256']:
            raise ValueError('This diagnostic belongs to another checkpoint')
        print('[old TEST] Existing locked diagnostic preparation retained', flush=True)
        return
    scenario = read_json(args.scenario_root / 'scenario_manifest.json')
    if scenario['status'] != 'complete':
        raise ValueError('The R76 scenario is incomplete')
    identities = scenario_solver_ids(scenario)
    evidence = read_json(args.reference_root / 'test_data_manifest.json')
    (args.root / 'labels').mkdir(parents=True, exist_ok=True)
    records = {}
    for problem in PROBLEMS:
        path = args.reference_root / 'test_predictions' / 'A' / f'{problem}.npz'
        with np.load(path, allow_pickle=False) as old:
            methods = scenario['problems'][problem]['methods']
            ids = [identities[name] for name in methods]
            labels, changed = align_labels(old, methods, ids, scenario.get('solver_aliases', {}))
        if len(labels['winner']) != 1000 or not np.array_equal(labels['indices'], np.arange(1000)):
            raise ValueError(f'{problem}: expected all original 1,000 TEST rows in native order')
        np.savez_compressed(args.root / 'labels' / f'{problem}.npz', **labels)
        costs = labels['costs']
        sbs_slot = int(costs.mean(0).argmin())
        records[problem] = dict(
            rows=len(costs), methods=methods, pool_ids=ids, source_methods=labels['source_pool'].tolist(),
            retained_source_slots=labels['source_slots'].tolist(), winner_changes=changed,
            original_dataset_sha256=evidence[problem]['data_hash'],
            original_labels_sha256=evidence[problem]['label_hash'],
            original_predictions=str(path.resolve()),
            posthoc_old_test_sbs=methods[sbs_slot],
            posthoc_old_test_sbs_cost=float(costs[:, sbs_slot].mean()),
            train_sbs=scenario['problems'][problem]['train_sbs'],
        )
    write_json(args.root / 'protocol.json', dict(
        experiment='R76_old_test_transfer', created_utc=utc(),
        checkpoint=str(checkpoint.resolve()), checkpoint_sha256=lock['checkpoint_sha256'],
        checkpoint_epoch=lock['epoch'], selection_rule=lock['rule'],
        original_test_rows=18000, seed=2, workers=args.workers,
        scenario_id=scenario['scenario_id'], scenario_manifest_sha256=file_hash(
            args.scenario_root / 'scenario_manifest.json'),
        frozen_normalization='R76 TRAIN geometry statistics stored in checkpoint; no refitting',
        labels='FP64 historical costs, filtered by deployment name; winner/rank/Oracle/gap recomputed',
        cost_regeneration=False, retraining=False, checkpoint_reselection=False,
        official_r76_test_unchanged=True,
        semantic_limit='Candidate pools aligned, but historical deployment costs are not relabeled to new deployment protocols',
        coord_dist='R76 ignore_coord_dist=True; no index-based distribution inference is used by the model',
        problems=records, total_winner_changes=sum(r['winner_changes'] for r in records.values()),
    ))
    print('[old TEST] Prepared all18 / 18,000 rows; winner changes=' + str(
        sum(r['winner_changes'] for r in records.values())), flush=True)


@torch.inference_mode()
def evaluate(args):
    _, lock = verify_checkpoint(args.run_root)
    protocol = read_json(args.root / 'protocol.json')
    if protocol['checkpoint_sha256'] != lock['checkpoint_sha256'] or args.workers != protocol['workers']:
        raise ValueError('Worker protocol differs from diagnostic preparation')
    receipt = args.root / f'worker{args.rank}_complete.json'
    if receipt.exists():
        raise RuntimeError('This worker already completed; use --stage analysis without repeating inference')
    configure_torch()
    torch.set_num_threads(2)
    device = torch.device(args.device)
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    model, checkpoint = load_r76_checkpoint(args.run_root / 'best.pt', device)
    if not checkpoint['args']['model_params'].get('ignore_coord_dist'):
        raise ValueError('Historical distribution metadata needs an explicit policy for this model')
    model.eval()
    started, completed = time.monotonic(), []
    (args.root / 'predictions').mkdir(parents=True, exist_ok=True)
    for problem in PROBLEMS[args.rank::args.workers]:
        record = protocol['problems'][problem]
        cache = ProblemCache(args.cache_root / 'test' / problem)
        try:
            if cache.split != 'test' or cache.size != record['rows']:
                raise ValueError('Cache is not the original complete TEST')
            sources = cache.metadata['source_files']
            for source, key in zip(sources[:2], ('original_dataset_sha256', 'original_labels_sha256')):
                if file_hash(Path(source['path'])) != record[key]:
                    raise ValueError(f'{problem}: input/label source differs from original R45 TEST')
            with np.load(record['original_predictions'], allow_pickle=False) as original:
                if cache.pool != original['pool'].tolist() or not np.array_equal(cache.raw['costs'], original['costs']):
                    raise ValueError(f'{problem}: cached columns differ from original FP64 costs')
            with np.load(args.root / 'labels' / f'{problem}.npz', allow_pickle=False) as stored:
                labels = {key: stored[key] for key in stored.files}
            requests = ((problem, torch.arange(start, min(start + args.batch_size, cache.size)),
                         min(args.batch_size, cache.size - start))
                        for start in range(0, cache.size, args.batch_size))
            scores = []
            with PinnedPrefetch({problem: AlignedCache(cache, record['pool_ids'])}, requests, device) as batches:
                for _, _, _, batch in batches:
                    scores.append(model(batch)['logits'].float().cpu().numpy())
            scores = np.concatenate(scores)
            if scores.shape != labels['costs'].shape or not np.isfinite(scores).all():
                raise ValueError('Inference omitted rows or produced invalid legal-candidate scores')
            pred = np.argsort(-scores, axis=1, kind='stable')[:, 0]
            np.savez_compressed(args.root / 'predictions' / f'{problem}.npz',
                                logits=scores, pred=pred, **labels)
            accuracy = float(np.mean(pred == labels['winner'])) * 100
            print(f'[old TEST rank{args.rank}] {problem}: n={cache.size} Top1={accuracy:.2f}%', flush=True)
            completed.append(problem)
        finally:
            cache.close()
    write_json(receipt, dict(rank=args.rank, workers=args.workers, problems=completed,
        checkpoint_sha256=lock['checkpoint_sha256'], host=socket.gethostname(),
        device=torch.cuda.get_device_name(device) if device.type == 'cuda' else str(device),
        seconds=time.monotonic() - started, completed_utc=utc()))


def analyze(args):
    protocol = read_json(args.root / 'protocol.json')
    for rank in range(protocol['workers']):
        receipt = read_json(args.root / f'worker{rank}_complete.json')
        if (receipt['problems'] != PROBLEMS[rank::protocol['workers']]
                or receipt['checkpoint_sha256'] != protocol['checkpoint_sha256']):
            raise ValueError('Missing or mismatched worker completion')
    per_problem, rows = {}, []
    for problem in PROBLEMS:
        with np.load(args.root / 'predictions' / f'{problem}.npz', allow_pickle=False) as saved:
            raw = {k: saved[k] for k in ('costs', 'winner', 'pool', 'pool_ids')}
            raw['pool'], raw['pool_ids'] = raw['pool'].tolist(), raw['pool_ids'].tolist()
            metrics, pred = fixed_sbs_metrics(saved['logits'], raw, protocol['problems'][problem]['train_sbs'])
            np.testing.assert_array_equal(pred, saved['pred'])
            metrics['native_correct'] = int(np.count_nonzero(pred == raw['winner']))
        per_problem[problem] = metrics
        rows.append(dict(problem=problem, n=metrics['n'], correct=metrics['native_correct'],
            top1_pct=metrics['top1'] * 100, top2_pct=metrics['top2'] * 100,
            top3_pct=metrics['top3'] * 100, mean_cost=metrics['mean_cost'],
            actual_regret_pct=metrics['actual_regret_pct'], vs_train_sbs_pct=metrics['vs_sbs_pct'],
            train_sbs=metrics['sbs_name'], winner_changes=protocol['problems'][problem]['winner_changes']))
    result = dict(per_problem=per_problem, macro=aggregate(per_problem), families=families(per_problem),
        n=sum(x['n'] for x in rows), correct=sum(x['correct'] for x in rows),
        protocol=str((args.root / 'protocol.json').resolve()),
        checkpoint_sha256=protocol['checkpoint_sha256'], completed_utc=utc())
    write_json(args.root / 'results.json', result)
    with (args.root / 'comparison.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    macro = result['macro']
    new = read_json(args.run_root / 'test_results.json')['result']['macro']
    lines = [
        '# R76: original TEST with current candidate pools', '',
        f"Locked checkpoint: epoch {protocol['checkpoint_epoch']}; `{protocol['checkpoint_sha256']}`.",
        'No retraining, checkpoint selection, solver execution, or changes to the formal R76 TEST.', '',
        '| Dataset | Instances | Top1 | Top2 | Top3 | Actual regret |',
        '|---|---:|---:|---:|---:|---:|',
        f"| Original TEST, aligned candidates | {result['n']:,} | {macro['macro_top1']*100:.4f}% | "
        f"{macro['macro_top2']*100:.4f}% | {macro['macro_top3']*100:.4f}% | {macro['macro_actual_regret_pct']:.4f}% |",
        f"| R76 new-scenario TEST | 70,110 | {new['macro_top1']*100:.4f}% | {new['macro_top2']*100:.4f}% | "
        f"{new['macro_top3']*100:.4f}% | {new['macro_actual_regret_pct']:.4f}% |", '',
        '## Label alignment',
        '- B tasks: remove RouteFinder, MoSES_RF, and MoSES_CaDA.',
        '- All MVRP tasks: remove RELD_MTL; retain RELD_MOEL at global ID16 and rename it reld_mt.',
        '- Preserve every original instance, historical FP64 cost, and native row order.',
        '- Recompute winner, cost ranking, Oracle, relative gaps, and post-hoc old TEST SBS.',
        f"- {protocol['total_winner_changes']:,} original winner identities changed after filtering.",
        '- Report vs_SBS against the R76 TRAIN-selected fixed deployment, evaluated on these historical costs.', '',
        '## Interpretation Boundary',
        'This aligns candidates, not all historical solver deployment protocols. No unknown cost is fabricated.',
        'Historical/new cost-generation differences, including previously identified time-window semantics, remain.',
        'The old/new TEST comparison is a cross-scenario transfer diagnostic, not a controlled data-size effect.',
        'R45A original 47.9111% used the unfiltered candidate pools and is not a same-pool comparison.', '',
        '## By Problem',
        '| Problem | Pool size | Top1 | Actual regret |', '|---|---:|---:|---:|',
    ]
    lines += [f"| {p} | {len(per_problem[p]['pool'])} | {per_problem[p]['top1']*100:.2f}% | "
              f"{per_problem[p]['actual_regret_pct']:.4f}% |" for p in PROBLEMS]
    (args.root / 'RESULTS.md').write_text('\n'.join(lines) + '\n')
    import wandb
    run = wandb.init(project='selector', entity='yjkds-southern-university-of-science-technology',
                     name='R76_old_test_transfer', mode='offline', dir=str(args.root), config=protocol)
    run.log({'old_test/' + k: v for k, v in macro.items()})
    run.finish()
    print(json.dumps(result['macro'], indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('prepare', 'evaluate', 'analysis'), required=True)
    parser.add_argument('--run-root', type=Path, default=DEFAULT_RUN)
    parser.add_argument('--root', type=Path)
    parser.add_argument('--reference-root', type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument('--scenario-root', type=Path, default=DEFAULT_SCENARIO)
    parser.add_argument('--cache-root', type=Path)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    if args.workers < 1 or not 0 <= args.rank < args.workers or args.batch_size < 1:
        parser.error('Use positive workers/batch size and a valid rank')
    for key in ('run_root', 'reference_root', 'scenario_root'):
        setattr(args, key, getattr(args, key).expanduser().resolve())
    args.root = (args.root or args.run_root / 'old_test_transfer').expanduser().resolve()
    if args.stage == 'evaluate' and args.cache_root is None:
        parser.error('Evaluation requires --cache-root')
    args.root.mkdir(parents=True, exist_ok=True)
    {'prepare': prepare, 'evaluate': evaluate, 'analysis': analyze}[args.stage](args)


if __name__ == '__main__':
    main()

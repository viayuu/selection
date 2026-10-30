"""Principal D1 learner: original full18 queries plus native structural families.

Preparation/describe are CPU-only. Train/test/lib require main's explicit release
and binding. This module never allocates a GPU, dispatches a solver, or edits data.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from functools import lru_cache
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import torch

from code.unified_selector.registry import GLOBAL_SOLVERS, P2I, constraint_bits, problem_descriptor
from . import r121_structural_data as structural
from . import r121_model as model_api
from .r75_data import RealTrainDataset, build_batch
from .r76_experiment import (capture_rank_rng, file_hash, fixed_sbs_metrics, open_cache,
    read_json, restore_rank_rng, save_checkpoint, slice_batch, utc, write_json)
from .r77_data import parameter_hash
from .r77_experiment import ShuffleStream, micro_caps
from .r80_experiment import json_hash, lock_best, save_npz, wandb_run
from . import r84_experiment as natural
from .r84_prepare import DEFAULT_DATA_ROOT, DEFAULT_PREPARED_ROOT, SPLIT_ROWS

ROOT = structural.ROOT / 'learner_v1'
DEFAULT_FAMILY = structural.ROOT / 'collection_explicit_cpu_v2/family_labels.jsonl'
DEFAULT_ISOLATION = structural.ROOT / 'learner_natural_isolation_v1.json'
ISOLATION_VERSION = 'r121_known_family_val_source_record_isolation_v1'
PROBLEMS = structural.PROBLEMS
ARM = 'principal'
VERSION = 'r121_raw_query_structural_potential_v1_20261030'
BATCH_SIZE, NATURAL_ROWS, FAMILY_PAIRS = 1024, 820, 102
MAX_UPDATES, WARMUP, EVAL_EVERY = 14400, 1080, 720
TRAIN_DIAGNOSTIC_ROWS, TRAIN_DIAGNOSTIC_SEED = 1024, 2
SELECT_RULE = 'minimum full18 natural VAL macro_actual_regret_pct, then maximum macro_top1, then earliest update'
INFERENCE_RULE = 'eval mode, original score-MLP maximin logits; exact ties use ascending global solver ID'


def task_order():
    generator = np.random.default_rng(2)
    return [str(p) for _ in range(MAX_UPDATES // 18) for p in generator.permutation(PROBLEMS)]


def exposures(step):
    return dict(logical_input_slots=step * BATCH_SIZE, natural_rows=step * NATURAL_ROWS,
        family_endpoint_slots=step * FAMILY_PAIRS * 2, family_pairs=step * FAMILY_PAIRS)


def lr_at(update):
    if not 1 <= update <= MAX_UPDATES:
        raise ValueError('Learning rate requires a successful update index')
    if update <= WARMUP:
        return 1e-4 * update / WARMUP
    fraction = (update - WARMUP) / (MAX_UPDATES - WARMUP)
    return 1e-6 + .5 * (1e-4 - 1e-6) * (1. + math.cos(math.pi * fraction))


def family_rows(path):
    path = Path(path).resolve()
    receipt = read_json(path.with_suffix('.receipt.json'))
    if (receipt.get('complete') is not True or receipt.get('full18') is not True
            or receipt.get('candidate_pool_unchanged') is not True
            or receipt.get('historical_labels_replaced') is not False
            or receipt.get('sha256') != file_hash(path) or receipt.get('path') != str(path)):
        raise ValueError('A complete original89-recipe native export is required')
    manifest = read_json(receipt['family_input_manifest'])
    structural.verify_seal(manifest, 'manifest_sha256')
    if (manifest['manifest_sha256'] != receipt['manifest_sha256']
            or manifest.get('full18_prepared') is not True
            or manifest.get('full_planned_population_prepared') is not True):
        raise ValueError('Family export/input manifest identity changed')
    expected = {e['endpoint_id']: e for e in manifest['endpoints']}
    grouped, seen, partitions = defaultdict(list), set(), {}
    with path.open() as stream:
        for line in stream:
            row = json.loads(line)
            e, t = row['endpoint'], row['targets']
            eid, p = e['endpoint_id'], e['problem']
            ids = [int(c['global_id']) for c in manifest['pools'][p]]
            costs = np.asarray(t['native_cost_fp64'], dtype=np.float64)
            if (eid in seen or e != expected.get(eid) or t['global_ids'] != ids
                    or len(costs) != len(ids) or not np.isfinite(costs).all() or (costs <= 0).any()
                    or not all(t['observed_mask']) or len(t['observed_mask']) != len(ids)
                    or t['native_winner_global_id'] != ids[int(costs.argmin())]
                    or row['collection_sha256'] != receipt['collection_sha256']):
                raise ValueError('Incomplete, relabeled or cross-protocol family row: ' + eid)
            np.testing.assert_allclose(t['centered_log_advantage'], np.log(costs) - np.log(costs).mean(), rtol=1e-12, atol=1e-12)
            group = e['group_id']
            partition = e['family_partition']
            if partitions.setdefault(group, partition) != partition:
                raise ValueError('Shared base/geometry group crosses family partitions, including across tasks')
            seen.add(eid)
            grouped[p].append(row)
    if set(grouped) != set(PROBLEMS) or seen != set(expected) or len(seen) != receipt['endpoints']:
        raise ValueError('All18 complete endpoints, not a partial family training pool, are required')
    for p, rows in grouped.items():
        rows.sort(key=lambda r: r['endpoint']['endpoint_id'])
        family_edges(rows, 'family_train')
        family_edges(rows, 'family_val')
    return dict(grouped), receipt, manifest


def family_edges(rows, partition):
    groups, group_partitions = defaultdict(dict), defaultdict(set)
    for index, row in enumerate(rows):
        endpoint = row['endpoint']
        groups[endpoint['family_id']][endpoint['cell']] = index
        group_partitions[endpoint['group_id']].add(endpoint['family_partition'])
    if any(len(parts) != 1 for parts in group_partitions.values()):
        raise ValueError('Base/geometry groups cannot cross diagnostic family partitions')
    edges = []
    for family in sorted(groups):
        cells = groups[family]
        if set(cells) != {'original', 'z00', 'z01', 'z10', 'z11'}:
            raise ValueError('Retain all five structural family endpoints')
        parent = cells['original']
        if rows[parent]['endpoint']['family_partition'] == partition:
            edges.extend((parent, cells[name]) for name in ('z00', 'z01', 'z10', 'z11'))
    if not edges:
        raise ValueError('Both TRAIN-derived family partitions must be nonempty')
    return np.asarray(edges, dtype=np.int64)


def index_hash(values):
    return hashlib.sha256(np.asarray(values, dtype='<i8').tobytes()).hexdigest()


def fixed_train_diagnostic_subset(eligible):
    selected, records = {}, {}
    for p in PROBLEMS:
        rows = np.asarray(eligible[p], dtype=np.int64)
        if rows.ndim != 1 or not len(rows) or len(np.unique(rows)) != len(rows):
            raise ValueError('Diagnostic sampling requires nonempty distinct eligible TRAIN rows: ' + p)
        seed = structural.r114.seed_for(TRAIN_DIAGNOSTIC_SEED, p, 'r121_fixed_train_diagnostic_v1')
        generator = np.random.default_rng(seed)
        selected[p] = np.sort(generator.choice(rows, size=min(TRAIN_DIAGNOSTIC_ROWS, len(rows)), replace=False))
        records[p] = dict(rows=len(selected[p]), seed=seed, indices_sha256=index_hash(selected[p]),
            eligible_indices_sha256=index_hash(rows))
    return selected, dict(version='r121_fixed_eligible_train_diagnostic_v1', seed=TRAIN_DIAGNOSTIC_SEED,
        max_rows_per_problem=TRAIN_DIAGNOSTIC_ROWS, per_problem=records, every_updates=EVAL_EVERY,
        selection='independent NumPy default_rng; uniform without replacement; sorted local indices; no winner strata',
        population='fixed eligible TRAIN subset, NOT complete TRAIN', eligible_only=True,
        used_for_checkpoint=False, eval_mode=True, no_grad=True, training_rng_restored=True,
        total_rows_per_interval=sum(r['rows'] for r in records.values()),
        gradient_exposures_unchanged=True, full81000_train_sbs_retained=True)


def diagnostic_subset_from_isolation(isolation):
    path = Path(isolation['path'])
    if file_hash(path) != isolation['sha256']:
        raise ValueError('Diagnostic source-record isolation changed')
    saved = read_json(path)
    structural.verify_seal(saved, 'isolation_sha256')
    eligible = {}
    for p in PROBLEMS:
        record = saved['per_problem'][p]
        mask = np.ones(record['original_train_rows'], dtype=bool)
        mask[np.asarray(record['excluded_local_rows'], dtype=np.int64)] = False
        eligible[p] = np.flatnonzero(mask)
        if index_hash(eligible[p]) != record['eligible_local_rows_sha256']:
            raise ValueError('Diagnostic eligible row identity changed: ' + p)
    return fixed_train_diagnostic_subset(eligible)


def natural_eligibility(manifest, data_root):
    """Resolve known family-heldout source records, never infer local row IDs."""
    originals = [e for e in manifest['endpoints'] if e['cell'] == 'original']

    def aliases(e):
        return {('geometry', e['group_id']), ('base', e['base_id']),
            ('source_base', e['source_base_id']), ('record', e['problem'], int(e['source_index']))}

    heldout = set().union(*(aliases(e) for e in originals if e['family_partition'] == 'family_val'))
    if not heldout:
        raise ValueError('Known family-val original source records are required')
    # Shared selected geometries/source identities can connect different tasks.
    while True:
        expanded = set(heldout)
        for e in originals:
            keys = aliases(e)
            if keys & heldout:
                expanded.update(keys)
        if expanded == heldout:
            break
        heldout = expanded
    associated = [e for e in manifest['endpoints'] if aliases(e) & heldout]
    if any(e['family_partition'] != 'family_val' for e in associated):
        raise ValueError('Known heldout geometry/source association would receive family-train gradients')
    records = [e for e in originals if aliases(e) & heldout]
    record_refs = sorted((dict(problem=e['problem'], source_index=int(e['source_index']),
        source_base_id=e['source_base_id'], base_id=e['base_id'], group_id=e['group_id'],
        family_id=e['family_id'], endpoint_id=e['endpoint_id']) for e in records),
        key=lambda e: (e['problem'], e['source_index'], e['endpoint_id']))
    endpoint_ids = {e['endpoint_id'] for e in associated}
    task_ids = sorted(t['task_id'] for t in manifest.get('tasks', []) if t['endpoint_id'] in endpoint_ids)
    eligible, per_problem = {}, {}
    for p in PROBLEMS:
        path = Path(data_root).resolve() / (p + 'train') / 'input_metadata.npz'
        with np.load(path, allow_pickle=False) as metadata:
            source = metadata['source_index'].copy()
        if (source.shape != (SPLIT_ROWS['train'],) or not np.issubdtype(source.dtype, np.integer)
                or (source < 0).any()):
            raise ValueError('Actual natural TRAIN source-index mapping is invalid: ' + p)
        excluded_sources = sorted({e['source_index'] for e in record_refs if e['problem'] == p})
        if set(excluded_sources) - set(source.tolist()):
            raise ValueError('A known family-val source is absent from actual natural TRAIN: ' + p)
        mask = np.isin(source, excluded_sources)
        excluded_rows = np.flatnonzero(mask).astype(np.int64)
        eligible[p] = np.flatnonzero(~mask).astype(np.int64)
        if not len(eligible[p]):
            raise ValueError('Natural CE has no eligible source records: ' + p)
        per_problem[p] = dict(original_train_rows=len(source), eligible_rows=len(eligible[p]),
            excluded_rows=len(excluded_rows), excluded_source_indices=excluded_sources,
            excluded_local_rows=excluded_rows.tolist(), eligible_local_rows_sha256=index_hash(eligible[p]),
            excluded_local_rows_sha256=index_hash(excluded_rows), source_index_array_sha256=index_hash(source),
            input_metadata=dict(path=str(path), sha256=file_hash(path)))
    identity = structural.seal(dict(version=ISOLATION_VERSION, data_root=str(Path(data_root).resolve()),
        family_manifest_sha256=manifest['manifest_sha256'], per_problem=per_problem,
        known_heldout_records=record_refs, known_geometry_groups=sorted({e['group_id'] for e in records}),
        associated_endpoint_count=len(endpoint_ids), associated_endpoint_ids_sha256=json_hash(sorted(endpoint_ids)),
        associated_native_task_count=len(task_ids), associated_native_task_ids_sha256=json_hash(task_ids),
        mapping='actual TRAIN input_metadata.source_index -> all matching local rows; never endpoint.local_index',
        gradient_scope='known source records and known selected geometry/base associations across all tasks',
        unknown_content_copies_exhaustively_excluded=False, independent_new_benchmark=False,
        original_splits_and_files_unchanged=True, full81000_normalization_retained=True,
        full81000_train_sbs_retained=True), 'isolation_sha256')
    return eligible, identity


def save_isolation(args, manifest):
    eligible, identity = natural_eligibility(manifest, args.data_root)
    path = structural.owned(args.natural_isolation)
    structural.r114.write_new_json(path, identity)
    summary = dict(path=str(path), sha256=file_hash(path), isolation_sha256=identity['isolation_sha256'],
        version=ISOLATION_VERSION, gradient_scope=identity['gradient_scope'],
        full81000_normalization_retained=True, full81000_train_sbs_retained=True,
        unknown_content_copies_exhaustively_excluded=False, independent_new_benchmark=False,
        per_problem={p: {k: v for k, v in record.items() if k not in
            ('excluded_local_rows', 'excluded_source_indices')} for p, record in identity['per_problem'].items()})
    return eligible, summary


def prepare_isolation(args):
    if args.device != 'cpu':
        raise PermissionError('Source-record isolation preparation is CPU-only')
    manifest = read_json(args.family_input_manifest)
    structural.verify_seal(manifest, 'manifest_sha256')
    _, summary = save_isolation(args, manifest)
    return dict(summary, solver_execution=False, model_training=False, GPU_execution=False)


def family_population(rows, isolation):
    per_problem = {}
    for p in PROBLEMS:
        per_problem[p] = {}
        for partition in ('family_train', 'family_val'):
            endpoints = [r['endpoint'] for r in rows[p] if r['endpoint']['family_partition'] == partition]
            per_problem[p][partition] = dict(endpoints=len(endpoints),
                base_families=len({e['family_id'] for e in endpoints}),
                geometry_groups=len({e['group_id'] for e in endpoints}),
                changed_input_task_pairs=sum(e['cell'] != 'original' for e in endpoints))
    return dict(per_problem=per_problem,
        family_endpoints=sum(v[s]['endpoints'] for v in per_problem.values() for s in v),
        additional_train_changed_endpoint_records=sum(v['family_train']['changed_input_task_pairs'] for v in per_problem.values()),
        natural_ce_eligible_rows_per_problem={p: v['eligible_rows'] for p, v in isolation['per_problem'].items()},
        known_holdout_source_records_receive_gradients=False, heldout_base_still_in_natural_CE=False,
        unseen_gradient_scope=isolation['gradient_scope'], strict_unseen_base=False,
        full81000_normalization_and_sbs_retained=True, exhaustive_content_copy_novelty=False,
        changed_endpoint_counts_are_not_exhaustive_content_unique_counts=True,
        independent_new_benchmark=False, natural_ce_isolation=isolation,
        family_slot_count_is_not_unique_input_count=True)


class NativeInputs:
    def __init__(self, rows, problem):
        self.rows, self.problem = rows, problem

    @lru_cache(maxsize=16)
    def __getitem__(self, index):
        endpoint = self.rows[int(index)]['endpoint']
        if file_hash(endpoint['input_path']) != endpoint['input_sha256']:
            raise ValueError('Native family input changed')
        item = read_json(endpoint['input_path'])['item']
        # UnifiedProblemDataset expects matrix.ndim, unlike the JSON certificate.
        return np.asarray(item, dtype=np.float32) if self.problem == 'ATSP' else item


class EndpointDataset(RealTrainDataset):
    """Reuse original feature extraction; never point it at an old label folder."""
    def __init__(self, problem, rows, pool):
        self.problem, self.split, self.coord_augment = problem, 'train', 0
        self.instances, self.rows = NativeInputs(rows, problem), rows
        self.size = self.base_N = self.N = len(rows)
        self.instance_ids = tuple(str(i) for i in range(self.size))
        self.pool = tuple(c['method'] for c in pool)
        self.pool_order = [int(c['global_id']) for c in pool]
        self.K_p, self.pid = len(pool), P2I[problem]
        self.mask = torch.zeros(len(GLOBAL_SOLVERS), dtype=torch.float32)
        self.mask[self.pool_order] = 1.
        self.cbits, self.problem_desc = constraint_bits(problem), problem_descriptor(problem)
        self.raw_costs = np.asarray([r['targets']['native_cost_fp64'] for r in rows], dtype=np.float64)
        self.raw_winner = self.raw_costs.argmin(-1).astype(np.int64)
        self.labels = {str(i): dict(cost=self.raw_costs[i], ind=int(self.raw_winner[i])) for i in range(self.size)}


def prepare_family(args):
    if args.device != 'cpu':
        raise PermissionError('Feature materialization is CPU-only')
    rows, receipt, manifest = family_rows(args.family_labels)
    root = structural.owned(args.family_cache)
    root.mkdir(parents=True, exist_ok=True)
    records = {}
    for p in PROBLEMS:
        directory = root / p
        complete = directory / 'complete.json'
        if complete.exists():
            saved = read_json(complete)
            if saved['family_export_sha256'] != receipt['sha256']:
                raise ValueError('Do not overwrite another family feature population')
            records[p] = saved
            continue
        if directory.exists() and list(directory.iterdir()):
            raise ValueError('Incomplete feature cache is preserved; choose a fresh --family-cache')
        directory.mkdir(parents=True, exist_ok=True)
        dataset = EndpointDataset(p, rows[p], manifest['pools'][p])
        maximum = max(r['endpoint']['scale'] for r in rows[p]) + 1
        arrays, constant = {}, None
        for start in range(0, dataset.size, 16):
            stop = min(start + 16, dataset.size)
            batch = build_batch(dataset, list(range(start, stop)), device='cpu', include_geometry=True, geometry_batch_size=8)
            if constant is None:
                constant = dict(problem_id=batch['problem_id'], kind=batch['kind'],
                    d_in=batch.get('d_in'), cbits=batch['cbits'][0].tolist(),
                    problem_desc=batch['problem_desc'][0].tolist(), mask=batch['mask'][0].tolist(),
                    pool=list(dataset.pool), pool_ids=dataset.pool_order, size=dataset.size,
                    maximum=maximum, family_export_sha256=receipt['sha256'],
                    endpoint_ids=[r['endpoint']['endpoint_id'] for r in rows[p]])
                shapes = dict(n=((dataset.size,), np.int64), costs=(dataset.raw_costs.shape, np.float64),
                    winner=((dataset.size,), np.int64))
                if batch['kind'] == 'coord':
                    shapes.update(node=((dataset.size, maximum, batch['node'].shape[-1]), np.float32),
                        node_geom=((dataset.size, maximum, batch['node_geom'].shape[-1]), np.float32))
                else:
                    shapes['matrix'] = ((dataset.size, maximum, maximum), np.float32)
                for key, (shape, dtype) in shapes.items():
                    arrays[key] = np.lib.format.open_memmap(directory / (key + '.npy'), mode='w+', dtype=dtype, shape=shape)
                    arrays[key][:] = 0
            arrays['n'][start:stop] = batch['n'].numpy()
            arrays['costs'][start:stop] = dataset.raw_costs[start:stop]
            arrays['winner'][start:stop] = dataset.raw_winner[start:stop]
            n = int(batch['n'].max())
            if n > maximum:
                raise ValueError('Native feature dimensions exceed the frozen input scale')
            for key in ('node', 'node_geom', 'matrix'):
                if key in arrays:
                    if key == 'matrix':
                        arrays[key][start:stop, :n, :n] = batch[key].numpy()
                    else:
                        arrays[key][start:stop, :n] = batch[key].numpy()
        for value in arrays.values():
            value.flush()
        del arrays
        constant['files'] = {path.name: dict(sha256=file_hash(path), bytes=path.stat().st_size,
            mtime_ns=path.stat().st_mtime_ns) for path in sorted(directory.glob('*.npy'))}
        write_json(complete, constant)
        records[p] = constant
        print(json.dumps(dict(event='family_cpu_features_complete', problem=p, endpoints=dataset.size)), flush=True)
    result = dict(complete=True, full18=True, family_export=receipt, cache_root=str(root),
        records=records, label_changes=False, input_changes=False, solver_execution=False, device='cpu')
    write_json(root / 'complete.json', result)
    return result


class FamilyCache:
    def __init__(self, root, problem, rows, export_sha):
        directory = Path(root) / problem
        self.metadata = read_json(directory / 'complete.json')
        if (self.metadata['family_export_sha256'] != export_sha
                or self.metadata['endpoint_ids'] != [r['endpoint']['endpoint_id'] for r in rows]):
            raise ValueError('Family features/labels/order differ')
        self.arrays = {}
        for filename, identity in self.metadata['files'].items():
            path = directory / filename
            if path.stat().st_size != identity['bytes'] or path.stat().st_mtime_ns != identity['mtime_ns']:
                if file_hash(path) != identity['sha256']:
                    raise ValueError('Prepared family feature cache changed')
            self.arrays[path.stem] = np.load(path, mmap_mode='r', allow_pickle=False)
        self.size = self.metadata['size']
        self.pool_ids, self.pool = self.metadata['pool_ids'], self.metadata['pool']
        self.edges = {part: family_edges(rows, part) for part in ('family_train', 'family_val')}
        self.indices = {part: np.asarray([i for i, r in enumerate(rows) if r['endpoint']['family_partition'] == part], dtype=np.int64)
                        for part in self.edges}

    def batch(self, indices, device='cpu'):
        indices = np.asarray(indices, dtype=np.int64)
        if indices.ndim != 1 or not len(indices) or (indices < 0).any() or (indices >= self.size).any():
            raise ValueError('Invalid family endpoint rows')
        count, record = len(indices), self.metadata
        n = torch.from_numpy(self.arrays['n'][indices].copy())
        maximum = int(n.max())
        batch = dict(problem_id=record['problem_id'], kind=record['kind'], n=n,
            node_mask=torch.arange(maximum)[None, :] < n[:, None],
            cbits=torch.tensor(record['cbits'], dtype=torch.float32).repeat(count, 1),
            problem_desc=torch.tensor(record['problem_desc'], dtype=torch.float32).repeat(count, 1),
            mask=torch.tensor(record['mask'], dtype=torch.float32).repeat(count, 1),
            coord_dist=torch.zeros(count, dtype=torch.long), pool_ids=torch.tensor(self.pool_ids, dtype=torch.long),
            solver_mask=torch.ones(count, len(self.pool_ids), dtype=torch.bool),
            costs=torch.from_numpy(self.arrays['costs'][indices].copy()),
            ind=torch.from_numpy(self.arrays['winner'][indices].copy()))
        for key in ('node', 'node_geom', 'matrix'):
            if key in self.arrays:
                values = self.arrays[key][indices, :maximum]
                if key == 'matrix':
                    values = values[:, :, :maximum]
                batch[key] = torch.from_numpy(np.ascontiguousarray(values))
        if record['d_in'] is not None:
            batch['d_in'] = record['d_in']
        return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}

    def close(self):
        self.arrays.clear()


def authorized(args):
    manifest = read_json(args.family_input_manifest)
    structural.verify_seal(manifest, 'manifest_sha256')
    release = structural.require_release(args.release, manifest['plan_sha256'])
    binding = structural.require_binding(args.binding, release, args.device)
    return dict(release_sha256=file_hash(args.release), binding_sha256=binding)


def status(args, phase, step, **values):
    write_json(args.root / ARM / 'status.json', dict(experiment='R121', arm=ARM, phase=phase,
        pid=os.getpid(), updated_utc=utc(), successful_updates=step, max_successful_updates=MAX_UPDATES,
        sample_exposures=step * BATCH_SIZE, exposure=exposures(step), **values))


def protocol(args, ready, manifest, geometry, baselines, params, family_receipt, population):
    _, train_diagnostic = diagnostic_subset_from_isolation(population['natural_ce_isolation'])
    return dict(experiment='R121', arm=ARM, version=VERSION, seed=2,
        initialization='scratch_shared_r84_plus_fresh_advantage_head',
        solver_names=list(GLOBAL_SOLVERS), model_params=params, model_factory='code.V4.r121_experiment.load_checkpoint',
        scenario_solver_aliases=manifest.get('scenario_solver_aliases', {}),
        data_root=str(args.data_root.resolve()), cache_root=str(args.cache_root.resolve()),
        test_cache_root=str(args.test_cache_root.resolve()) if args.test_cache_root else None,
        prepared_root=str(args.prepared_root.resolve()), preparation_sha256=file_hash(args.prepared_root / 'prepared.json'),
        common_trainable_sha256=ready['trainable_sha256'], geometry_stats=geometry,
        rows_per_problem=SPLIT_ROWS, train_rows_per_problem=81000, heldout_rows_per_problem=4500,
        candidate_pools={p: manifest['splits']['train'][p]['pool'] for p in PROBLEMS},
        candidate_pool_ids={p: manifest['splits']['train'][p]['pool_ids'] for p in PROBLEMS},
        winner_rule=natural.WINNER_RULE, baselines=baselines, own_train_sbs=baselines,
        global_batch_size=BATCH_SIZE, logical_global_batch=BATCH_SIZE, natural_rows_per_update=NATURAL_ROWS,
        family_pairs_per_update=FAMILY_PAIRS, max_successful_updates=MAX_UPDATES,
        updates_per_problem=MAX_UPDATES // 18, warmup_updates=WARMUP, eval_every_updates=EVAL_EVERY,
        total_sample_exposures=MAX_UPDATES * BATCH_SIZE, exposures=exposures(MAX_UPDATES),
        family_labels=family_receipt, family_cache=str(args.family_cache.resolve()), family_population=population,
        natural_ce_isolation=population['natural_ce_isolation'],
        natural_ce_eligible_rows_per_problem=population['natural_ce_eligible_rows_per_problem'],
        train_diagnostic=train_diagnostic,
        normalization_and_sbs_population='original full81000 TRAIN/task retained for historical comparability',
        family_partition='family_train loss; family_val known associated source records excluded from natural CE',
        historical_control='not matched: new structural population and different natural/family exposure',
        loss=dict(weights=model_api.LOSS_WEIGHTS, advantage_unit=model_api.ADVANTAGE_UNIT, huber_delta=1.,
            response='same centered potential(changed) - potential(original)',
            normalization='CE/820, endpoint/204, response/102; legal method means per row',
            other_terms=False, metadata_in_forward=False),
        lr=dict(peak=1e-4, end=1e-6, warmup_updates=WARMUP, max_updates=MAX_UPDATES),
        dropout=.1, optimizer='AdamW', weight_decay=1e-4, grad_clip=1., fixed_budget=True, early_stop=False,
        selection_head='original_score_mlp_maximin',
        objective=dict(classification='natural native maximin CE + shared endpoint/response Huber'),
        microbatch_caps=micro_caps(args.gpu_type, args.micro_cap), gpu_type=args.gpu_type,
        eval_batch_size=args.eval_batch_size, checkpoint_rule=SELECT_RULE, inference_rule=INFERENCE_RULE,
        task_order_sha256=hashlib.sha256('\n'.join(task_order()).encode()).hexdigest(),
        source_sha256={name: file_hash(Path(__file__).parent / name) for name in
            ('r121_model.py', 'r121_experiment.py', 'r121_structural_data.py', 'pairwise_selector.py',
             'dual_stream.py', 'V4Model.py', 'r75_data.py', 'r76_experiment.py', 'r84_experiment.py')},
        test_rnd_disclosure='Original holdouts participated in historical R&D; not claimed blind',
        wandb=True, wandb_mode=args.wandb_mode)


def load_checkpoint(path, device='cpu'):
    state = torch.load(path, map_location='cpu', weights_only=False)
    config = state['args']
    if (config.get('experiment') != 'R121' or config.get('arm') != ARM or config.get('version') != VERSION
            or config.get('seed') != 2 or config.get('max_successful_updates') != MAX_UPDATES
            or config.get('solver_names') != list(GLOBAL_SOLVERS) or config.get('loss', {}).get('weights') != model_api.LOSS_WEIGHTS
            or state.get('state_hash') != json_hash(config) or config.get('checkpoint_rule') != SELECT_RULE):
        raise ValueError('Not an unchanged principal R121 checkpoint')
    for name, digest in config['source_sha256'].items():
        if file_hash(Path(__file__).parent / name) != digest:
            raise ValueError('Locked learner source changed: ' + name)
    return model_api.make_model(config['model_params'], state['model']).to(device), state


load_model = load_checkpoint
from_checkpoint = load_checkpoint


def native_batch(cache, rows, device):
    batch = cache.batch(rows, device='cpu')
    # ScenarioCache intentionally returns FP32 training costs; D1 targets must not.
    batch['costs'] = torch.from_numpy(cache.raw['costs'][np.asarray(rows)].copy())
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


@torch.no_grad()
def evaluate(model, args, manifest, baselines, split, tag, step, state_hash, checkpoint_sha=None):
    directory = args.root / ARM / 'evaluation' / split / tag
    directory.mkdir(parents=True, exist_ok=True)
    identity = dict(experiment='R121', arm=ARM, split=split, tag=tag, successful_updates=step,
        state_hash=state_hash, model_state_sha256=parameter_hash(model.state_dict()),
        checkpoint_sha256=checkpoint_sha, winner_rule=natural.WINNER_RULE)
    saved_identity = directory / 'identity.json'
    if saved_identity.exists() and read_json(saved_identity) != identity:
        raise ValueError('Saved evaluation is for another checkpoint or rule')
    write_json(saved_identity, identity)
    if (directory / 'metrics.json').exists() and (directory / 'rank000.npz').exists():
        return read_json(directory / 'metrics.json')
    started, training, arrays, per_problem = time.monotonic(), model.training, {}, {}
    model.eval()
    device = next(model.parameters()).device
    try:
        for p in PROBLEMS:
            status(args, 'evaluation', step, split=split, evaluation_problem=p)
            shard = directory / 'by_problem' / (p + '.npz')
            if shard.exists():
                with np.load(shard, allow_pickle=False) as archive:
                    data = {key: archive[key] for key in archive.files}
            else:
                cache_root = args.test_cache_root if split == 'test' and args.test_cache_root else args.cache_root
                cache = open_cache(cache_root, split, p, manifest, args.data_root)
                try:
                    cap = min(args.eval_batch_size, micro_caps(args.gpu_type, args.micro_cap)[p])
                    scores = [model(cache.batch(torch.arange(i, min(i + cap, cache.size)), device))['logits'].float().cpu().numpy()
                              for i in range(0, cache.size, cap)]
                    with np.load(args.data_root / (p + split) / 'input_metadata.npz', allow_pickle=False) as metadata:
                        source = metadata['source_index'].copy()
                    data = dict(logits=np.concatenate(scores), indices=np.arange(cache.size, dtype=np.int64),
                        source_indices=np.arange(cache.size, dtype=np.int64), release_source_indices=source,
                        costs=cache.raw['costs'].copy(), winner=cache.raw['winner'].copy(),
                        pool=np.asarray(cache.pool), pool_ids=np.asarray(cache.pool_ids, dtype=np.int64), nodes=cache.lengths.copy())
                    save_npz(shard, data)
                finally:
                    cache.close()
            if (len(data['logits']) != SPLIT_ROWS[split] or data['costs'].dtype != np.float64
                    or data['winner'].dtype != np.int64 or not np.array_equal(data['winner'], data['costs'].argmin(1))):
                raise ValueError('Incomplete native full18 evaluation')
            raw = {key: data[key] for key in ('costs', 'winner', 'pool', 'pool_ids')}
            raw['pool'], raw['pool_ids'] = raw['pool'].tolist(), raw['pool_ids'].tolist()
            per_problem[p], data['pred'] = fixed_sbs_metrics(data['logits'], raw, baselines[p])
            arrays.update({p + '__' + key: value for key, value in data.items()})
    finally:
        model.train(training)
    save_npz(directory / 'rank000.npz', arrays)
    result = natural.summarize(per_problem)
    result.update(identity=identity, seconds=time.monotonic() - started, successful_updates=step,
        sample_exposures=step * BATCH_SIZE, exposure=exposures(step), tag=tag,
        prediction_archive=str((directory / 'rank000.npz').resolve()))
    write_json(directory / 'metrics.json', result)
    print(json.dumps(dict(event='full18_evaluation', split=split, step=step, **result['macro'])), flush=True)
    return result


@torch.no_grad()
def train_subset_diagnostic(model, caches, baselines, indices, definition, cap=128):
    training, device = model.training, next(model.parameters()).device
    # Even an unexpected stochastic eval/cache operation must not change training RNG.
    rng, started, per_problem = capture_rank_rng(device), time.monotonic(), {}
    try:
        model.eval()
        for p in PROBLEMS:
            rows, cache = np.asarray(indices[p], dtype=np.int64), caches[p]
            record = definition['per_problem'][p]
            if len(rows) != record['rows'] or index_hash(rows) != record['indices_sha256']:
                raise ValueError('Fixed TRAIN diagnostic indices changed: ' + p)
            scores = [model(cache.batch(rows[i:i + cap], device))['logits'].float().cpu().numpy()
                for i in range(0, len(rows), cap)]
            raw = dict(costs=cache.raw['costs'][rows].copy(), winner=cache.raw['winner'][rows].copy(),
                pool=list(cache.pool), pool_ids=list(cache.pool_ids))
            per_problem[p], _ = fixed_sbs_metrics(np.concatenate(scores), raw, baselines[p])
    finally:
        try:
            model.train(training)
        finally:
            restore_rank_rng(rng, device)
    result = natural.summarize(per_problem)
    result.update(population='fixed eligible TRAIN subset, NOT complete TRAIN', used_for_checkpoint=False,
        subset=definition, seconds=time.monotonic() - started, eval_mode=True, no_grad=True, training_rng_restored=True)
    return result


@torch.no_grad()
def family_diagnostic(model, caches, cap=128):
    training, device, results = model.training, next(model.parameters()).device, {}
    model.eval()
    try:
        for p, cache in caches.items():
            edges = cache.edges['family_val']
            losses = dict(endpoint_huber=0., response_huber=0.)
            width = max(1, cap // 2)
            for start in range(0, len(edges), width):
                batch = cache.batch(edges[start:start + width].reshape(-1), device)
                parts = model_api.family_losses(model(batch), batch)
                for key, value in parts.items():
                    losses[key] += float(value.sum())
            results[p] = dict(endpoint_huber=losses['endpoint_huber'] / (len(edges) * 2),
                response_huber=losses['response_huber'] / len(edges), pairs=len(edges))
    finally:
        model.train(training)
    return dict(per_problem=results, used_for_checkpoint=False, training_derived=True,
        natural_base_may_be_ce_trained=False, known_holdout_source_records_receive_gradients=False,
        unseen_gradient_scope='known source records and selected geometry/base associations only',
        full81000_normalization_and_sbs_retained=True, strict_unseen_base=False,
        exhaustive_content_copy_novelty=False, independent_new_benchmark=False,
        macro={key: float(np.mean([r[key] for r in results.values()])) for key in ('endpoint_huber', 'response_huber')})


def choose_key(val, step):
    macro = val['macro']
    key = (float(macro['macro_actual_regret_pct']), -float(macro['macro_top1']), int(step))
    if not all(math.isfinite(x) for x in key):
        raise ValueError('Nonfinite official VAL cannot select a checkpoint')
    return key


def backward_parts(model, natural_data, family_data, cap, scaler):
    totals = dict(natural_ce=0., endpoint_huber=0., response_huber=0.)
    for start in range(0, NATURAL_ROWS, cap):
        batch = slice_batch(natural_data, start, min(start + cap, NATURAL_ROWS))
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=next(model.parameters()).is_cuda):
            values = model_api.natural_loss(model(batch), batch)
            loss = values.sum() / NATURAL_ROWS
        scaler.scale(loss).backward()
        totals['natural_ce'] += float(loss.detach())
    family_cap = max(2, cap - cap % 2)
    for start in range(0, 2 * FAMILY_PAIRS, family_cap):
        batch = slice_batch(family_data, start, min(start + family_cap, 2 * FAMILY_PAIRS))
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=next(model.parameters()).is_cuda):
            values = model_api.family_losses(model(batch), batch)
            pieces = dict(endpoint_huber=values['endpoint_huber'].sum() / (2 * FAMILY_PAIRS),
                response_huber=values['response_huber'].sum() / FAMILY_PAIRS)
            loss = sum(pieces.values())
        scaler.scale(loss).backward()
        for key, value in pieces.items():
            totals[key] += float(value.detach())
    totals['loss'] = sum(totals.values())
    totals['ce'] = totals['natural_ce']
    return totals


def train(args):
    authorization = authorized(args)
    device = torch.device(args.device)
    ready, manifest, geometry, baselines = natural.prepared_config(args)
    rows, receipt, family_manifest = family_rows(args.family_labels)
    eligible, isolation = save_isolation(args, family_manifest)
    diagnostic_indices, diagnostic_definition = fixed_train_diagnostic_subset(eligible)
    initial = torch.load(args.prepared_root / 'identicalinit_seed2.pt', map_location='cpu', weights_only=False)
    selector, params, initialization = model_api.initialize_shared(initial, geometry)
    del initial
    config = protocol(args, ready, manifest, geometry, baselines, params, receipt, family_population(rows, isolation))
    if config['train_diagnostic'] != diagnostic_definition:
        raise ValueError('Protocol and actual fixed TRAIN diagnostic population differ')
    if family_manifest['plan_sha256'] != read_json(args.family_input_manifest)['plan_sha256']:
        raise ValueError('Authorized family plan differs from training export')
    for p in PROBLEMS:
        if [c['global_id'] for c in family_manifest['pools'][p]] != config['candidate_pool_ids'][p]:
            raise ValueError('Natural/family legal pools differ: ' + p)
    root, state_hash = structural.owned(args.root) / ARM, json_hash(config)
    root.mkdir(parents=True, exist_ok=True)
    lock_path, last = root / 'protocol.json', root / 'last.pt'
    if lock_path.exists() and read_json(lock_path) != config:
        raise ValueError('Preserve the existing principal learner protocol')
    if (root / 'training_complete.json').exists():
        return dict(status='already_complete', next_command='test', path=str(root))
    if last.exists() and not args.resume:
        raise ValueError('Existing training state requires explicit --resume')
    write_json(lock_path, config)
    selector.to(device)
    optimizer = torch.optim.AdamW(selector.parameters(), lr=1e-4, weight_decay=1e-4)
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda')
    step, history, best, counts, pending, retries = 0, [], None, Counter(), None, 0
    cursors, pair_cursors = dict.fromkeys(PROBLEMS, 0), dict.fromkeys(PROBLEMS, 0)
    if last.exists():
        selector, checkpoint = load_checkpoint(last, device)
        if checkpoint['args'] != config or checkpoint['successful_updates'] % EVAL_EVERY:
            raise ValueError('Resume only an unchanged completed interval/pending VAL state')
        optimizer = torch.optim.AdamW(selector.parameters(), lr=1e-4, weight_decay=1e-4)
        optimizer.load_state_dict(checkpoint['optimizer'])
        scaler.load_state_dict(checkpoint['scaler'])
        step, history, best = checkpoint['successful_updates'], checkpoint['history'], checkpoint['best']
        counts, pending = Counter(checkpoint['updates_by_problem']), checkpoint['pending_evaluation']
        cursors, pair_cursors = checkpoint['stream_draws'], checkpoint['pair_draws']
        initialization, retries = checkpoint['initialization'], checkpoint['amp_retries']
        restore_rank_rng(checkpoint['rng'], device)
    else:
        from .train import set_seed
        set_seed(2)
        write_json(root / 'initialization.json', initialization)
    natural_caches, family_caches, streams, pairs, tracker = {}, {}, {}, {}, None
    try:
        for index, p in enumerate(PROBLEMS):
            natural_caches[p] = open_cache(args.cache_root, 'train', p, manifest, args.data_root)
            family_caches[p] = FamilyCache(args.family_cache, p, rows[p], receipt['sha256'])
            if natural_caches[p].size != 81000 or family_caches[p].pool_ids != config['candidate_pool_ids'][p]:
                raise ValueError('Incomplete natural population or mismatched family pool')
            streams[p] = ShuffleStream(len(eligible[p]), 20000 + index, cursors[p])
            pairs[p] = ShuffleStream(len(family_caches[p].edges['family_train']), 120000 + index, pair_cursors[p])
        order = task_order()
        if Counter(order[:step]) != counts or any(streams[p].draws != counts[p] * NATURAL_ROWS
                or pairs[p].draws != counts[p] * FAMILY_PAIRS for p in PROBLEMS):
            raise ValueError('Successful updates and both population cursors disagree')

        def snapshot():
            return dict(model=selector.state_dict(), args=config, state_hash=state_hash,
                optimizer=optimizer.state_dict(), scaler=scaler.state_dict(), epoch=len(history),
                successful_updates=step, updates_by_problem=dict(counts), initialization=initialization,
                sample_exposures=step * BATCH_SIZE, exposure=exposures(step),
                stream_draws={p: s.draws for p, s in streams.items()}, pair_draws={p: s.draws for p, s in pairs.items()},
                history=history, best=best, pending_evaluation=pending, amp_retries=retries,
                rng=capture_rank_rng(device), resource_authorization=authorization)

        if not last.exists():
            save_checkpoint(last, snapshot())
        tracker_rng = capture_rank_rng(device)
        tracker = wandb_run(args, config)
        restore_rank_rng(tracker_rng, device)
        interval_losses, interval_steps = Counter(), 0
        while step < MAX_UPDATES or pending is not None:
            if pending is None:
                selector.train()
                p = order[step]
                local_rows = eligible[p][streams[p].take(NATURAL_ROWS).numpy()]
                batch = native_batch(natural_caches[p], local_rows, device)
                edge_rows = family_caches[p].edges['family_train'][pairs[p].take(FAMILY_PAIRS).numpy()].reshape(-1)
                family = family_caches[p].batch(edge_rows, device)
                for group in optimizer.param_groups:
                    group['lr'] = lr_at(step + 1)
                rng = capture_rank_rng(device)
                for attempt in range(3):
                    optimizer.zero_grad(set_to_none=True)
                    restore_rank_rng(rng, device)
                    losses = backward_parts(selector, batch, family, config['microbatch_caps'][p], scaler)
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(selector.parameters(), 1.)
                    before = scaler.get_scale()
                    scaler.step(optimizer)
                    scaler.update()
                    if scaler.get_scale() >= before:
                        break
                    retries += 1
                else:
                    raise FloatingPointError('Three AMP overflows on the SAME logical batch; no silent lost exposure')
                step += 1
                counts[p] += 1
                interval_steps += 1
                interval_losses.update(losses)
                if step % 18 == 0:
                    status(args, 'training', step, task=p, lr=lr_at(step), amp_retries=retries)
                if step % EVAL_EVERY:
                    continue
                pending = dict(optimization={key: value / interval_steps for key, value in interval_losses.items()},
                    amp_retries=retries)
                save_checkpoint(last, snapshot())
            val = evaluate(selector, args, manifest, baselines, 'val', f'update{step:07d}', step, state_hash)
            train_metrics = train_subset_diagnostic(selector, natural_caches, baselines,
                diagnostic_indices, diagnostic_definition,
                min(args.eval_batch_size, min(config['microbatch_caps'].values())))
            diagnostic = family_diagnostic(selector, family_caches, min(args.eval_batch_size, 128))
            entry = dict(epoch=step // EVAL_EVERY, successful_updates=step, val=val, train=train_metrics,
                family_val=diagnostic, exposure=exposures(step), **pending)
            history.append(entry)
            key = choose_key(val, step)
            if best is None or key < tuple(best['selection_key']):
                best = dict(epoch=len(history), successful_updates=step, selection_key=list(key),
                    val=val, state_hash=state_hash)
                save_checkpoint(root / 'best.pt', snapshot())
            pending, interval_steps, interval_losses = None, 0, Counter()
            save_checkpoint(last, snapshot())
            write_json(root / 'history.json', history)
            if tracker is not None:
                tracker.log(dict(successful_updates=step, **{f'val/{k}': v for k, v in val['macro'].items()},
                    **{f'train/{k}': v for k, v in entry['optimization'].items()},
                    **{f'family_val/{k}': v for k, v in diagnostic['macro'].items()}), step=step)
        selected_lock = lock_best(root, best, config, state_hash)
        write_json(root / 'training_complete.json', dict(experiment='R121', arm=ARM, complete=True,
            successful_updates=step, selected_best=selected_lock, state_hash=state_hash,
            exposure=exposures(step), amp_retries=retries, updated_utc=utc()))
        status(args, 'training_complete_unique_val_lock', step, selected_update=best['successful_updates'])
        return dict(complete=True, training=True, automatic_test=False, next_command='test', selected_best=best)
    finally:
        for cache in [*natural_caches.values(), *family_caches.values()]:
            cache.close()
        if tracker is not None:
            tracker.finish()


def locked_training(args):
    root = args.root / ARM
    config, complete, lock = (read_json(root / name) for name in ('protocol.json', 'training_complete.json', 'final_test_lock.json'))
    if (config['experiment'] != 'R121' or config['version'] != VERSION or complete['successful_updates'] != MAX_UPDATES
            or config['max_successful_updates'] != MAX_UPDATES or lock['state_hash'] != json_hash(config)
            or file_hash(root / 'best.pt') != lock['checkpoint_sha256'] or lock['rule'] != SELECT_RULE
            or lock['selected_without_test'] is not True or lock['per_problem_assembly'] is not False):
        raise ValueError('Completed principal learner and one unchanged VAL checkpoint required')
    return root, config, lock


def test(args):
    authorized(args)
    root, config, lock = locked_training(args)
    args.gpu_type, args.micro_cap = config['gpu_type'], config['microbatch_caps']['TSP']
    args.eval_batch_size = config['eval_batch_size']
    if args.test_cache_root is None and config.get('test_cache_root'):
        args.test_cache_root = Path(config['test_cache_root'])
    _, manifest, _, baselines = natural.prepared_config(args)
    if str(args.data_root.resolve()) != config['data_root'] or baselines != config['baselines']:
        raise ValueError('Never change heldout population or TRAIN-fitted SBS')
    model, state = load_checkpoint(root / 'best.pt', args.device)
    if state['successful_updates'] != lock['selected_update']:
        raise ValueError('Checkpoint update differs from unique VAL lock')
    val = evaluate(model, args, manifest, baselines, 'val', 'selected_best', lock['selected_update'], lock['state_hash'])
    val_identity = val['identity']
    if val['macro'] != state['history'][-1]['val']['macro']:
        raise ValueError('Selected VAL results changed at the same checkpoint')
    result = evaluate(model, args, manifest, baselines, 'test', 'selected_best', lock['selected_update'],
        lock['state_hash'], lock['checkpoint_sha256'])
    identity = dict(experiment='R121', arm=ARM, epoch=lock['epoch'], successful_updates=lock['selected_update'],
        selected_update=lock['selected_update'], state_hash=lock['state_hash'], checkpoint_sha256=lock['checkpoint_sha256'])
    write_json(root / 'test_results.json', dict(identity, complete=True, result=result, updated_utc=utc()))
    write_json(root / 'final_metrics.json', dict(identity, val=val, test=result, val_identity=val_identity,
        population_exposure=config['exposures'], original_training_rows_per_problem=81000,
        natural_ce_eligible_rows_per_problem=config['natural_ce_eligible_rows_per_problem'],
        natural_ce_isolation=config['natural_ce_isolation']))
    status(args, 'full18_test_complete', MAX_UPDATES, selected_update=lock['selected_update'])
    return result


def lib(args):
    authorized(args)
    from . import r80_reporting as common
    run, config, lock, result, _ = common.locked_inputs(args.root, ARM, experiment='R121', arms={ARM: MAX_UPDATES})
    outputs = common.r80_lib(args.root, run, config, lock, args.device, args.lib_batch_size,
        lambda phase, **fields: status(args, phase, MAX_UPDATES, **fields),
        result['identity']['model_state_sha256'], factory_module='code.V4.r121_experiment')
    write_json(run / 'lib_stage_complete.json', dict(complete=True, experiment='R121',
        checkpoint_sha256=lock['checkpoint_sha256'], datasets=list(outputs), updated_utc=utc()))
    return outputs


def structural_pdf(run, output, config, lock):
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    history = read_json(run / 'history.json')
    path = output / 'structural_objectives.pdf'
    population, exposure = config['family_population'], config['exposures']
    train_inputs = sum(v['family_train']['endpoints'] for v in population['per_problem'].values())
    diagnostic_inputs = sum(v['family_val']['endpoints'] for v in population['per_problem'].values())
    with PdfPages(path) as pdf:
        fig = plt.figure(figsize=(12, 8))
        fig.text(.07, .92, 'R121 | Structural population and actual sample exposure', fontsize=17)
        lines = [
            'Same unique natural-VAL checkpoint: update ' + str(lock['selected_update']),
            'Checkpoint SHA256: ' + lock['checkpoint_sha256'],
            'Natural release:81000 rows/task; CE excludes known family-val source records; VAL/TEST unchanged4500.',
            'CE eligible records: ' + str(sum(population['natural_ce_eligible_rows_per_problem'].values()))
                + '; per-task counts and exact mapping hashes are in the locked protocol.',
            'TRAIN metric curves use a fixed eligible-only subset <=1024/task, NOT all81000; never select checkpoints.',
            f"Family endpoints: TRAIN={train_inputs}; TRAIN-derived diagnostic={diagnostic_inputs}.",
            'Auxiliary TRAIN changed-endpoint records (not exhaustive unique-content count): ' + str(population['additional_train_changed_endpoint_records']),
            'Each logical1024:820 natural rows +102(original,changed) pairs=204 endpoint slots.',
            f"Full-budget exposures: natural={exposure['natural_rows']}; family slots={exposure['family_endpoint_slots']}; pairs={exposure['family_pairs']}.",
            'Repeated original endpoints and repeated epochs are exposures, NOT new unique instances.',
            'Diagnostic: known source records/geometry associations unseen by gradients, not a new independent benchmark.',
            'Loss=1*natural native CE +1*endpoint Huber +1*response Huber; log-advantage unit=0.01.',
            'Original raw-query score head and auxiliary potential share the query/candidate representation.',
            'Responses use the SAME centered potential difference; no family metadata/current cost in forward.',
            'Population/exposure differs from history: NOT an update/data-matched control or a benefit claim.',
            'Seed2 fresh initialization; full81000 TRAIN normalization and TRAIN-SBS retained; no exhaustive copy-novelty claim.',
            'TEST informed historical R&D. No TEST/LIB checkpoint, threshold, or inference-rule selection.']
        for i, line in enumerate(lines):
            fig.text(.07, .86 - i * .048, line, fontsize=9.3)
        pdf.savefig(fig)
        plt.close(fig)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for key in model_api.LOSS_WEIGHTS:
            axes[0].plot([r['successful_updates'] for r in history], [r['optimization'][key] for r in history], label=key)
        axes[0].plot([r['successful_updates'] for r in history],
            [r['train']['macro']['macro_ce'] for r in history], label='eval CE: fixed eligible TRAIN subset')
        for key in ('endpoint_huber', 'response_huber'):
            axes[1].plot([r['successful_updates'] for r in history], [r['family_val']['macro'][key] for r in history], label=key)
        axes[0].set_title('Three fixed loss terms + fixed TRAIN-subset eval CE')
        axes[1].set_title('Known source-record gradient holdout; TRAIN-derived')
        for ax in axes:
            ax.legend(fontsize=8)
            ax.set_xlabel('Successful updates')
            ax.axvline(lock['selected_update'], color='#a64b39', linestyle='--')
            ax.grid(alpha=.2)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)
    return dict(path=str(path), pages=2, checkpoint_sha256=lock['checkpoint_sha256'],
        population=population, exposure=exposure, model_inference=False)


def scoped_learning_curves(run, output, config, lock):
    """Redraw the saved curves locally with the TRAIN metric population visible."""
    import matplotlib.pyplot as plt
    history = read_json(run / 'history.json')
    diagnostic = config['train_diagnostic']
    counts = [record['rows'] for record in diagnostic['per_problem'].values()]
    scope = str(counts[0]) + '/task' if len(set(counts)) == 1 else f'{min(counts)}-{max(counts)}/task'
    model = config['experiment'] + '/' + config['arm']
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    try:
        fig.suptitle(model + ' | successful updates; one locked VAL checkpoint', fontsize=15)
        for name in ('loss', 'ce', 'cmp', 'risk', 'kl', 'focal_extra', 'ovr', 'ovr_extra',
                'cost_KL', 'cost_KL_extra', 'subsetCE', 'subsetCE_extra'):
            points = [(r['successful_updates'], r['optimization'][name]) for r in history
                if name in r.get('optimization', {})]
            if points:
                axes[0, 0].plot(*zip(*points), label=name)
        axes[0, 0].legend(fontsize=8)
        axes[0, 0].set_title('TRAIN interval mean objectives (training batches)')
        for ax, key, factor, title in ((axes[0, 1], 'macro_top1', 100., 'Native Top1 (%)'),
                (axes[1, 0], 'macro_actual_regret_pct', 1., 'Actual regret (%)'),
                (axes[1, 1], 'macro_vs_sbs_pct', 1., 'vs fixed TRAIN SBS (%)')):
            for split, label in (('train', f'TRAIN: fixed eligible subset ({scope})'),
                    ('val', 'VAL: full frozen validation')):
                points = [(r['successful_updates'], r[split]['macro'][key] * factor)
                    for r in history if r.get(split) is not None]
                if points:
                    ax.plot(*zip(*points), label=label)
            ax.set_title(title)
            ax.legend(fontsize=8)
        for ax in axes.flat:
            ax.axvline(lock['selected_update'], color='#c45358', linestyle='--', linewidth=1)
            ax.set_xlabel('Successful optimizer updates')
            ax.grid(alpha=.2)
        fig.text(.025, .045,
            f'TRAIN metric curves: fixed eligible TRAIN subset ({scope}; seed {diagnostic["seed"]}); NOT full 81,000 TRAIN rows/task.',
            fontsize=9)
        fig.text(.025, .02,
            'SBS was fitted on full 81k/task. Subset metrics do not select checkpoints; locked-checkpoint full TRAIN is reported separately.',
            fontsize=9)
        fig.tight_layout(rect=(0, .08, 1, .96))
        fig.savefig(output / 'learning_curves.pdf')
        fig.savefig(output / 'learning_curves.png', dpi=160)
    finally:
        plt.close(fig)
    return dict(train_scope='fixed eligible TRAIN subset: ' + scope, full_train=False,
        history_sha256=file_hash(run / 'history.json'), model_inference=False)


def report(args):
    from . import r80_reporting as common
    from . import r84_reporting as reports
    run, config, lock, result, val = common.locked_inputs(args.root, ARM, experiment='R121', arms={ARM: MAX_UPDATES})
    predictions, metrics, methods = reports.load_saved_splits(run, config, lock, result, val)
    reports.saved_libs(run, config, lock, result, predictions, metrics, methods)
    comparisons, alignment = reports.nss_comparisons(config, lock, predictions, args.nss_root)
    reports.build_reports(run, config, lock, result, val, predictions, metrics, methods, comparisons, alignment)
    output = run / 'full_model_statistics'
    curves = scoped_learning_curves(run, output, config, lock)
    supplement = structural_pdf(run, output, config, lock)
    saved_protocol = read_json(output / 'protocol.json')
    saved_protocol.update(family_population=config['family_population'], exposure=config['exposures'],
        train_diagnostic=config['train_diagnostic'], learning_curves_scope=curves, structural_objectives=supplement,
        loss=config['loss'], historical_control=config['historical_control'])
    write_json(output / 'protocol.json', saved_protocol)
    return dict(complete=True, path=str(output), checkpoint_sha256=lock['checkpoint_sha256'])


def finish(args):
    """Publish readiness from saved results only, never infer or select again."""
    from . import r80_reporting as common
    run, config, lock, result, val = common.locked_inputs(args.root, ARM, experiment='R121', arms={ARM: MAX_UPDATES})
    completion = read_json(run / 'lib_stage_complete.json')
    if (completion.get('complete') is not True or completion.get('checkpoint_sha256') != lock['checkpoint_sha256']
            or set(completion.get('datasets', [])) != {'TSPLIB', 'CVRPLIB'}):
        raise ValueError('Finish requires both native LIBs at the same locked checkpoint')
    rendering = report(args)
    output = run / 'full_model_statistics'
    names = ('MODEL_STATISTICS.pdf', 'selection_distributions.pdf', 'learning_curves.pdf',
        'training_and_method_diagnostics.pdf', 'structural_objectives.pdf', 'metrics_by_problem.csv',
        'selection_distribution.csv', 'single_method_comparison.csv', 'nss_comparison.csv', 'protocol.json')
    artifacts = {}
    for name in names:
        path = output / name
        if not path.is_file() or not path.stat().st_size:
            raise ValueError('Missing required saved report artifact: ' + str(path))
        if path.suffix == '.pdf':
            with path.open('rb') as stream:
                if stream.read(5) != b'%PDF-':
                    raise ValueError('Invalid required PDF: ' + str(path))
        artifacts[name] = dict(path=str(path.resolve()), sha256=file_hash(path), bytes=path.stat().st_size)
    summary = dict(experiment='R121', arm=ARM, complete=True, updated_utc=utc(),
        checkpoint_sha256=lock['checkpoint_sha256'], selected_update=lock['selected_update'],
        state_hash=lock['state_hash'], VAL=val['macro'], TEST=result['macro'],
        all18=True, native_libs=['TSPLIB', 'CVRPLIB'], model_factory=config['model_factory'],
        family_export=config['family_labels'], family_population=config['family_population'],
        train_diagnostic=config['train_diagnostic'],
        loss=config['loss'], exposure=config['exposures'],
        historical_control=config['historical_control'], artifacts=artifacts, rendering=rendering,
        solver_execution=False, model_inference=False, checkpoint_reselection=False,
        resource_release='main-owned; this saved-only finish does not assert GPU release',
        Git_operations=False, research_goal_complete=False)
    write_json(run / 'delivery_manifest.json', summary)
    return summary


def describe(args):
    return dict(experiment='R121', version=VERSION, arm=ARM, raw_query_model='ScoreDifferenceSelector',
        all18=list(PROBLEMS), original_natural_train_per_problem=81000, logical_batch=BATCH_SIZE,
        natural_ce_sampling='81000 minus known family-val source-record/selected-geometry associations via actual metadata',
        natural_isolation=str(args.natural_isolation), isolation_exists=args.natural_isolation.is_file(),
        full81000_normalization_and_sbs_retained=True, independent_new_benchmark=False,
        natural_rows=NATURAL_ROWS, family_pairs=FAMILY_PAIRS, loss_weights=model_api.LOSS_WEIGHTS,
        advantage_unit=model_api.ADVANTAGE_UNIT, max_updates=MAX_UPDATES, warmup=WARMUP,
        eval_every=EVAL_EVERY, seed=2, exposure=exposures(MAX_UPDATES), checkpoint_rule=SELECT_RULE,
        train_diagnostic=dict(max_rows_per_problem=TRAIN_DIAGNOSTIC_ROWS, seed=TRAIN_DIAGNOSTIC_SEED,
            every_updates=EVAL_EVERY, population='fixed eligible TRAIN subset, NOT complete TRAIN',
            used_for_checkpoint=False, training_rng_restored=True),
        family_export_exists=args.family_labels.is_file(), family_feature_cache_exists=(args.family_cache / 'complete.json').is_file(),
        family_labels=str(args.family_labels), requires_complete_native89=True, R120_release_required=True,
        solver_execution=False, GPU_execution=False, legacy_B_unchanged=True, R58='deferred')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('describe', 'prepare-isolation', 'prepare-family', 'train', 'test', 'lib', 'report', 'finish'))
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--data-root', type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument('--prepared-root', type=Path, default=DEFAULT_PREPARED_ROOT)
    parser.add_argument('--cache-root', type=Path)
    parser.add_argument('--test-cache-root', type=Path)
    parser.add_argument('--family-labels', type=Path, default=DEFAULT_FAMILY)
    parser.add_argument('--family-cache', type=Path, default=ROOT / 'family_cache')
    parser.add_argument('--family-input-manifest', type=Path, default=structural.ROOT / 'full18_inputs_v1/manifest.json')
    parser.add_argument('--natural-isolation', type=Path, default=DEFAULT_ISOLATION)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--release', type=Path)
    parser.add_argument('--binding', type=Path)
    parser.add_argument('--gpu-type', choices=('4090', '3090'), default='4090')
    parser.add_argument('--micro-cap', type=int, default=128)
    parser.add_argument('--eval-batch-size', type=int, default=128)
    parser.add_argument('--lib-batch-size', type=int, default=1)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--wandb-mode', choices=('online', 'offline', 'disabled'), default='online')
    parser.add_argument('--nss-root', type=Path, default=Path(__file__).parent / 'runs/R84_new90k_attention/nss_reference')
    args = parser.parse_args(argv)
    args.root, args.arm, args.experiment = structural.owned(args.root), ARM, 'R121'
    if args.test_cache_root and args.test_cache_root.name == 'test' and (args.test_cache_root / 'TSP').is_dir():
        args.test_cache_root = args.test_cache_root.parent
    if args.eval_batch_size < 1 or args.lib_batch_size < 1 or args.micro_cap < 2:
        raise ValueError('Positive capacities and pair-preserving micro-cap>=2 required')
    result = {'describe': describe, 'prepare-isolation': prepare_isolation, 'prepare-family': prepare_family, 'train': train,
        'test': test, 'lib': lib, 'report': report, 'finish': finish}[args.command](args)
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

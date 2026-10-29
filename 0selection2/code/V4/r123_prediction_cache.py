"""Frozen shared historical-bank prediction cache; no dependency on the learner.

One expert is loaded at a time. A complete problem/expert NPZ is atomically
published with explicit current row/source identities. No query labels are
stored or passed to a model. R124 can consume the same R77B archive directly.
"""

from __future__ import annotations

import argparse
import fcntl
import gc
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

from code.unified_selector.registry import PROBLEMS
from .r76_cache import ScenarioCache
from .r84_prepare import DEFAULT_PREPARED_ROOT, SPLIT_ROWS
from .r123_meta_data import (CURRENT_ROOT, DEFAULT_EXPERTS, PREPARATION, json_hash,
    load_expert, load_meta_indices, model_inputs, read_json, sha256, source_record_ids)


ROOT = PREPARATION.parent
SCHEMA = 'r123_shared_expert_npz_v1'


def array_hash(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def precision_policy():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')


def context(prepared_root=DEFAULT_PREPARED_ROOT, input_cache_root=None):
    meta = read_json(PREPARATION / 'manifest.json')
    ready = read_json(Path(prepared_root) / 'prepared.json')
    cm = read_json(Path(prepared_root) / 'source_cache_manifest.json')
    if (not meta['meta_training_ready_at_source_record_scope']
            or meta['selected_experts'] != list(DEFAULT_EXPERTS)
            or Path(ready['data_root']).resolve() != CURRENT_ROOT.resolve()):
        raise ValueError('Require fixed R77B/R80A/R83 certificate and current R84 data')
    for path, record in meta['evidence_files'].items():
        if Path(path).suffix == '.py' and sha256(path) != record['sha256']:
            raise ValueError('Certified immutable model/input source changed: ' + path)
    for expert in DEFAULT_EXPERTS:
        r = meta['experts'][expert]
        if sha256(r['checkpoint_path']) != r['checkpoint_sha256']:
            raise ValueError('Frozen checkpoint changed: ' + expert)
    return dict(meta=meta, ready=ready, cm=cm, prepared_root=Path(prepared_root),
        input_cache_root=input_cache_root, identity=meta['preparation_identity'])


def authorize_test(root, test_lock_root=None):
    run = Path(test_lock_root) if test_lock_root is not None else Path(root) / 'preparation/learner_run'
    path = run / 'final_test_lock.json'
    if not path.is_file():
        raise PermissionError('TEST cache requires the completed R123 unique VAL checkpoint lock')
    lock = read_json(path)
    if (lock.get('experiment') != 'R123' or lock.get('completed_updates') != 7200
            or lock.get('selected_without_test') is not True
            or lock.get('checkpoint_sha256') != sha256(run / 'best.pt')
            or lock.get('protocol_sha256') != json_hash(read_json(run / 'protocol.json'))):
        raise PermissionError('R123 TEST lock changed/incomplete')
    return lock


def rows_for(split, problem):
    if split == 'train':
        return load_meta_indices(problem)
    if split not in ('val', 'test'):
        raise ValueError('Expected train/val/test')
    return np.arange(SPLIT_ROWS[split], dtype=np.int64)


def input_directory(ctx, split, problem):
    cm = (read_json(ctx['prepared_root'] / 'source_cache_manifest.json')
          if split == 'test' else ctx['cm'])
    try:
        r = cm['splits'][split][problem]
    except KeyError as error:
        raise FileNotFoundError('Existing R84 input cache missing; no input regeneration here') from error
    directory = Path(r['directory'])
    if ctx['input_cache_root'] is not None:
        directory = Path(ctx['input_cache_root']) / directory.relative_to(ctx['ready']['cache_source'])
    return directory, r


def identity(ctx, split, problem, expert, rows, *, complete=True):
    metadata_path = CURRENT_ROOT / (problem + split) / 'input_metadata.npz'
    digest = sha256(metadata_path)
    if split == 'train' and digest != ctx['meta']['evidence_files'][str(metadata_path)]['sha256']:
        raise ValueError('TRAIN source identity metadata changed since certification')
    with np.load(metadata_path, allow_pickle=False) as stream:
        source = stream['source_index'][rows].copy()
    lineage = read_json(PREPARATION / 'source_catalogues.json')[problem]
    generator_path = Path(lineage['generation_metadata'])
    if sha256(generator_path) != ctx['meta']['evidence_files'][str(generator_path)]['sha256']:
        raise ValueError('Frozen input generator changed')
    qualified = source_record_ids(problem, source, read_json(generator_path))
    record, bank = ctx['meta']['problems'][problem], ctx['meta']['experts'][expert]
    _, input_record = input_directory(ctx, split, problem)
    if (input_record['rows'] != SPLIT_ROWS[split] or input_record['pool'] != record['candidates']
            or input_record['pool_ids'] != record['global_solver_ids']):
        raise ValueError('Input cache row count/current legal pool differs')
    info = dict(schema=SCHEMA, split=split, problem=problem, expert_id=expert,
        bank_order=list(DEFAULT_EXPERTS), checkpoint_path=bank['checkpoint_path'],
        checkpoint_sha256=bank['checkpoint_sha256'], factory=bank['factory'],
        meta_preparation_identity=ctx['identity'], complete=complete, rows=len(rows),
        pool=record['candidates'], pool_ids=record['global_solver_ids'],
        input_metadata_sha256=digest, input_lineage_path=str(generator_path),
        input_lineage_sha256=sha256(generator_path), row_indices_sha256=array_hash(rows),
        source_indices_sha256=array_hash(source), source_identity_sha256=array_hash(qualified),
        precision='FP32 eval, no autocast, TF32 disabled; original complete-pool logits',
        score_ties='ascending global solver ID', query_labels_stored=False,
        certificate_boundary='source-record only; associated base/content copies unverified',
        helper_sha256=sha256(__file__))
    return info, source, qualified


def archive_path(root, split, problem, expert):
    return Path(root) / 'prediction_cache' / split / problem / (expert + '.npz')


def read_expert(root, split, problem, expert, ctx=None, *, test_lock_root=None, allow_incomplete=False):
    if split == 'test':
        authorize_test(root, test_lock_root)
    ctx = context() if ctx is None else ctx
    path = archive_path(root, split, problem, expert)
    with np.load(path, allow_pickle=False) as z:
        data = {key: z[key].copy() for key in z.files}
    saved = json.loads(str(data.pop('metadata_json')))
    rows = rows_for(split, problem)
    if allow_incomplete:
        rows = rows[:saved['rows']]
    expected, _, _ = identity(ctx, split, problem, expert, rows,
                              complete=len(rows) == len(rows_for(split, problem)))
    digest = saved.pop('logits_sha256')
    if (saved != expected or (not saved['complete'] and not allow_incomplete)
            or data['logits'].dtype != np.float32
            or data['logits'].shape != (len(rows), len(saved['pool']))
            or not np.isfinite(data['logits']).all()
            or not np.array_equal(data['row_indices'], rows)
            or array_hash(data['logits']) != digest
            or array_hash(data['source_indices']) != saved['source_indices_sha256']
            or array_hash(data['source_identity']) != saved['source_identity_sha256']
            or data['pool'].tolist() != saved['pool'] or data['pool_ids'].tolist() != saved['pool_ids']):
        raise ValueError('Expert prediction cache identity/rows/pool/content differs: ' + str(path))
    data['metadata'] = saved
    return data


def load_scores(root, split, problem, ctx=None, *, experts=DEFAULT_EXPERTS, test_lock_root=None):
    """Shared R124 API: scores [N,E,K] and exact current local row IDs."""
    ctx = context() if ctx is None else ctx
    records = [read_expert(root, split, problem, e, ctx, test_lock_root=test_lock_root) for e in experts]
    if any(not np.array_equal(r['row_indices'], records[0]['row_indices']) for r in records[1:]):
        raise ValueError('Experts must have identical current-row ordering')
    return dict(logits=np.stack([r['logits'] for r in records], axis=1),
        row_indices=records[0]['row_indices'], source_indices=records[0]['source_indices'],
        source_identity=records[0]['source_identity'], pool=records[0]['pool'],
        pool_ids=records[0]['pool_ids'], expert_ids=list(experts),
        metadata=[r['metadata'] for r in records])


def publish(path, data):
    temporary = path.with_name('.' + path.name + '.' + str(os.getpid()) + '.tmp')
    with temporary.open('wb') as stream:
        np.savez(stream, **data)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def preflight(args, ctx, problems):
    if 'test' in args.split:
        authorize_test(args.root, args.test_lock_root)
    for split in args.split:
        for problem in problems:
            rows = rows_for(split, problem)
            if args.limit is not None:
                rows = rows[:args.limit]
            identity(ctx, split, problem, args.experts[0], rows,
                     complete=len(rows) == len(rows_for(split, problem)))
            directory, record = input_directory(ctx, split, problem)
            cached = read_json(directory / 'metadata.json')
            if (cached['source_files'] != record['source_files'] or not cached['complete']
                    or cached['rows'] != SPLIT_ROWS[split]):
                raise ValueError('Existing R84 input cache metadata mismatch before model/CUDA load')
            for name in cached['arrays']:
                if not (directory / (name + '.npy')).is_file():
                    raise FileNotFoundError('Transport must copy existing input array: ' + str(directory / (name + '.npy')))


@torch.no_grad()
def build(args):
    torch.set_num_threads(2)
    precision_policy()
    ctx = context(args.prepared_root, args.input_cache_root)
    if args.shards < 1 or not 0 <= args.shard_index < args.shards or args.batch_size < 1:
        raise ValueError('Valid disjoint problem shard and positive prediction batch required')
    problems = [p for p in PROBLEMS if p in args.problem][args.shard_index::args.shards]
    preflight(args, ctx, problems)
    if args.describe:
        result = dict(root=str(args.root), problems=problems, split=args.split,
            experts=args.experts, batch_size=args.batch_size, device=args.device,
            train_rows_per_problem=10886, val_rows_per_problem=4500,
            helper_sha256=sha256(__file__), CUDA_initialized=torch.cuda.is_initialized())
        print(json.dumps(result), flush=True)
        return result
    for expert in args.experts:
        model = None
        try:
            for split in args.split:
                for problem in problems:
                    path = archive_path(args.root, split, problem, expert)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    with path.with_suffix('.lock').open('a') as lock:
                        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        if path.is_file():
                            read_expert(args.root, split, problem, expert, ctx,
                                test_lock_root=args.test_lock_root, allow_incomplete=args.limit is not None)
                            print('reuse', split, problem, expert, flush=True)
                            continue
                        if model is None:
                            model, _ = load_expert(expert, device=args.device)
                        rows = rows_for(split, problem)
                        full_count = len(rows)
                        if args.limit is not None:
                            rows = rows[:args.limit]
                        info, source, qualified = identity(ctx, split, problem, expert, rows,
                                                           complete=len(rows) == full_count)
                        directory, _ = input_directory(ctx, split, problem)
                        cache = ScenarioCache(directory)
                        scores = np.empty((len(rows), len(info['pool'])), dtype=np.float32)
                        try:
                            for start in range(0, len(rows), args.batch_size):
                                stop = min(start + args.batch_size, len(rows))
                                inputs = model_inputs(cache.batch(rows[start:stop], device=args.device))
                                with torch.autocast(device_type=torch.device(args.device).type, enabled=False):
                                    scores[start:stop] = model(inputs)['logits'].float().cpu().numpy()
                            if not np.isfinite(scores).all():
                                raise ValueError('Missing/nonfinite complete-expert scores')
                            info['logits_sha256'] = array_hash(scores)
                            publish(path, dict(logits=scores, row_indices=rows, source_indices=source,
                                source_identity=qualified, pool=np.asarray(info['pool']),
                                pool_ids=np.asarray(info['pool_ids'], dtype=np.int64),
                                metadata_json=np.asarray(json.dumps(info, sort_keys=True))))
                            print('complete', split, problem, expert, len(rows), flush=True)
                        finally:
                            cache.close()
        finally:
            del model
            gc.collect()


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--prepared-root', type=Path, default=DEFAULT_PREPARED_ROOT)
    p.add_argument('--input-cache-root', type=Path)
    p.add_argument('--test-lock-root', type=Path)
    p.add_argument('--split', nargs='+', choices=('train', 'val', 'test'), default=['train', 'val'])
    p.add_argument('--problem', '--problems', nargs='+', choices=PROBLEMS, default=list(PROBLEMS))
    p.add_argument('--experts', nargs='+', choices=DEFAULT_EXPERTS, default=list(DEFAULT_EXPERTS))
    p.add_argument('--shard-index', type=int, default=0)
    p.add_argument('--shards', type=int, default=1)
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--device', default='cpu')
    p.add_argument('--limit', type=int)
    p.add_argument('--describe', action='store_true')
    return p


if __name__ == '__main__':
    args = parser().parse_args()
    if args.limit is not None and args.limit < 1:
        raise ValueError('Positive CPU smoke limit required')
    build(args)

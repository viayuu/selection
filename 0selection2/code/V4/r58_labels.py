"""Resumable, fail-closed scenario_v2 generation and release validation."""

import argparse
import json
import os
import pickle
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from ..unified_selector.registry import GLOBAL_SOLVERS, POOLS, PROBLEMS
from .r58_backends import make_backend
from .r58_execution import (ExecutionSession, backend_batch_size, execution_profile,
    group_batches, read_execution_plan, require_budget_approval)
from .r58_scenario import (CONTRACT, INPUT_ROOT, ROOT, canonical_hash, check_route,
                           fields, file_hash, save_json, write_contract)


def load_instances(problem, split):
    path = INPUT_ROOT / f'{problem}{split}' / 'dataset.pkl'
    with path.open('rb') as stream:
        return pickle.load(stream), path


def size(problem, item):
    return fields(problem, item)['n']


def preflight_indices(problem, items):
    sizes = np.array([size(problem, item) for item in items])
    targets = np.unique(np.quantile(sizes, [0, .5, 1]).astype(int))
    rng = np.random.default_rng(2)
    return sorted(int(rng.choice(np.flatnonzero(sizes == target))) for target in targets)


def implementation_hashes():
    return {name: file_hash(Path(__file__).with_name(name)) for name in
            ('r58_scenario.py', 'r58_environments.py', 'r58_backends.py', 'r58_labels.py', 'r58_execution.py')}


def inference_batch_size(problem, method):
    return backend_batch_size(problem, method)


def inference_plan(problem, instances, method):
    scales = np.array([size(problem, item) for item in instances])
    batches = []
    width = inference_batch_size(problem, method)
    for scale in sorted(set(scales.tolist())):
        indices = np.flatnonzero(scales == scale).tolist()
        batches.extend(indices[start:start + width] for start in range(0, len(indices), width))
    return batches


def runtime_environment():
    import importlib.metadata
    return dict(python=sys.version, torch=torch.__version__,
        packages={name: importlib.metadata.version(name) for name in
                  ('torchrl', 'tensordict', 'numpy', 'rl4co', 'lightning')},
        pythonpath=os.environ.get('PYTHONPATH', ''))


def seed_solver(seed=2):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_float32_matmul_precision('highest')


def timed_run(backend, items):
    torch.cuda.synchronize()
    before = time.perf_counter()
    output = backend.run(items)
    torch.cuda.synchronize()
    return output, time.perf_counter() - before


def probe(problem, method, root):
    write_contract(root)
    seed_solver()
    items, path = load_instances(problem, 'train')
    indices = preflight_indices(problem, items)
    record = dict(problem=problem, solver=method, split='train', indices=indices,
        input_sha256=file_hash(path), contract_sha256=canonical_hash(CONTRACT),
        implementation_sha256=implementation_hashes(), passed=False,
        runtime_environment=runtime_environment(),
        candidate_qualification='functionality/feasibility only; never selected from cost ranking')
    try:
        backend = make_backend(problem, method)
        record['deployment'] = backend.profile()
        results = []
        for index in indices:
            warmup = {}
            if method in ('RouteFinder', 'MoSES_RF', 'MoSES_CaDA'):
                seed_solver()
                first, seconds = timed_run(backend, [items[index]])
                warmup = dict(first_call_seconds=seconds, first_call_cost=float(first.costs[0]),
                              timing_note='Warm call below excludes deferred checkpoint/import loading')
            seed_solver()
            output, elapsed = timed_run(backend, [items[index]])
            checked = check_route(problem, items[index], output.tours[0], output.costs[0])
            if not checked['feasible'] or not checked['cost_matches']:
                raise ValueError(checked)
            results.append(dict(index=index, true_size=size(problem, items[index]),
                                seconds=elapsed, independent=checked, route=output.tours[0], **warmup))
        if inference_batch_size(problem, method) > 1:
            target = max(indices, key=lambda i: size(problem, items[i]))
            scale = size(problem, items[target])
            companions = [i for i, item in enumerate(items) if i != target and size(problem, item) == scale]
            batch_indices = [target, *companions[:inference_batch_size(problem, method) - 1]]
            seed_solver()
            batch_output, elapsed = timed_run(backend, [items[i] for i in batch_indices])
            record['batch_probe'] = dict(indices=batch_indices, seconds=elapsed,
                seconds_per_instance=elapsed / len(batch_indices),
                first_instance_single_cost=next(r['independent']['cost'] for r in results if r['index'] == target),
                first_instance_batched_cost=float(batch_output.costs[0]),
                first_instance_same_tour=batch_output.tours[0] == next(r['route'] for r in results if r['index'] == target))
        record.update(passed=True, results=results)
    except Exception as error:
        record.update(error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
    save_json(root / 'preflight' / f'{problem}__{method}.json', record)
    print(json.dumps({key: record.get(key) for key in ('problem', 'solver', 'passed', 'error')}, ensure_ascii=False), flush=True)
    return record['passed']


def lock(root):
    contract = write_contract(root)
    plan = read_execution_plan(root)
    approval = require_budget_approval(root, plan, implementation_hashes())
    environment = runtime_environment()
    profiles, missing = {}, []
    for problem in PROBLEMS:
        profiles[problem] = {}
        for method in POOLS[problem]:
            path = root / 'preflight' / f'{problem}__{method}.json'
            if not path.exists():
                missing.append(f'{problem}/{method}: no preflight')
                continue
            record = json.loads(path.read_text())
            if (record['contract_sha256'] != contract['contract_sha256'] or
                    record['implementation_sha256'] != implementation_hashes()):
                missing.append(f'{problem}/{method}: stale implementation or contract')
            elif record.get('runtime_environment') != environment:
                missing.append(f'{problem}/{method}: preflight used a different runtime environment')
            elif not record['passed']:
                missing.append(f'{problem}/{method}: {record.get("error")}')
            else:
                profiles[problem][method] = record['deployment']
    if missing:
        save_json(root / 'publication_blockers.json', {'missing': missing, 'training_allowed': False})
        raise RuntimeError(f'{len(missing)} deployments not qualified. No scenario release or training is allowed.')
    payload = dict(scenario='scenario_v2', seed=2, contract_sha256=contract['contract_sha256'],
        implementation_sha256=implementation_hashes(), global_solver_order=list(GLOBAL_SOLVERS),
        pools={p: list(POOLS[p]) for p in PROBLEMS}, deployments=profiles,
        all_cost_columns='fresh solve under this new locked deployment; no uncertified old-label reuse',
        execution_plan=plan, execution_approval=approval,
        execution_profiles={p: {m: execution_profile(p, m, plan) for m in POOLS[p]} for p in PROBLEMS},
        inference_batch_size={p: {m: inference_batch_size(p, m) for m in POOLS[p]} for p in PROBLEMS},
        batch_plan='ascending true size, then original instance index; unchanged fixed batches, retain tail; '
                   'size-homogeneous groups <=100 cases except an unsplit original batch; persistent workers',
        random_state='seed=2 reset before each fixed inference batch; no seed search',
        runtime_environment=environment,
        failure_rule='stop affected column and publication; preserve complete input split',
        test_policy='deployments immutable before any scenario_v2 validation/test generation')
    path = root / 'deployments.lock.json'
    payload['lock_sha256'] = canonical_hash(payload)
    if path.exists() and json.loads(path.read_text()) != payload:
        raise ValueError('Locked deployment changed; use a new scenario revision')
    save_json(path, payload)
    return payload


def read_lock(root):
    path = root / 'deployments.lock.json'
    payload = json.loads(path.read_text())
    claimed = payload['lock_sha256']
    if canonical_hash({k: v for k, v in payload.items() if k != 'lock_sha256'}) != claimed:
        raise ValueError('Deployment lock corrupt')
    if (payload['global_solver_order'] != list(GLOBAL_SOLVERS) or
            payload['pools'] != {p: list(POOLS[p]) for p in PROBLEMS}):
        raise ValueError('Solver registry names/order changed after deployment lock')
    if payload['implementation_sha256'] != implementation_hashes():
        raise ValueError('Implementation changed after deployment lock')
    if payload['contract_sha256'] != canonical_hash(CONTRACT):
        raise ValueError('Contract changed after deployment lock')
    if payload['runtime_environment'] != runtime_environment():
        raise ValueError('Runtime environment differs from the locked deployment')
    plan = read_execution_plan(root)
    if payload.get('execution_plan') != plan:
        raise ValueError('Execution plan differs from locked deployment')
    if payload.get('execution_approval') != require_budget_approval(root, plan, implementation_hashes()):
        raise ValueError('Budget approval differs from locked deployment')
    expected = {p: {m: execution_profile(p, m, plan) for m in POOLS[p]} for p in PROBLEMS}
    if payload.get('execution_profiles') != expected:
        raise ValueError('Execution scheduler differs from locked deployment')
    return payload


def column_metadata(locked, problem, method, split, count, input_hash):
    return dict(problem=problem, split=split, solver=method, n=count,
        input_sha256=input_hash, lock_sha256=locked['lock_sha256'],
        deployment_sha256=canonical_hash(locked['deployments'][problem][method]))


def check_column_metadata(path, expected):
    metadata_path = path.with_suffix('.metadata.json')
    if not metadata_path.exists() or json.loads(metadata_path.read_text()) != expected:
        raise ValueError(f'Column provenance differs from the locked deployment/input: {path}')


def generate(problem, method, split, root):
    locked = read_lock(root)
    if method not in locked['pools'][problem]:
        raise ValueError('Ineligible solver')
    instances, input_path = load_instances(problem, split)
    directory = root / 'scenario_v2' / f'{problem}{split}'
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / 'columns' / f'{method}.jsonl'
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = column_metadata(locked, problem, method, split, len(instances), file_hash(input_path))
    metadata_path = path.with_suffix('.metadata.json')
    if path.exists() or metadata_path.exists():
        check_column_metadata(path, metadata)
    save_json(metadata_path, metadata)
    completed = {}
    if path.exists():
        with path.open() as stream:
            for line in stream:
                row = json.loads(line)
                index = row['index']
                if index in completed or not 0 <= index < len(instances):
                    raise ValueError('Duplicate or out-of-range row in resumed column')
                checked = check_route(problem, instances[index], row['route'], row['cost'])
                if not checked['feasible'] or not checked['cost_matches']:
                    raise ValueError('Cached route fails current release check')
                completed[index] = row
    if len(completed) == len(instances):
        return
    groups = group_batches(problem, instances, inference_plan(problem, instances, method))
    pending = [group for group in groups if not all(index in completed for index in group['indices'])]
    directory = directory / 'execution' / method
    # Warm the extreme input sizes, loading both RF checkpoint buckets before work.
    warmups = [pending[0]] if len(pending) == 1 else [pending[0], pending[-1]]
    try:
        with ExecutionSession(problem, method, locked['execution_plan'], directory, metadata,
                input_path=input_path, expected_profile=locked['deployments'][problem][method]) as session:
            session.prepare(warmups)
            result = session.run_groups(pending, pass_id='production')
        for row in result['rows']:
            completed[row['index']] = row
        if set(completed) != set(range(len(instances))):
            raise ValueError('Incomplete executed column; no instance may be dropped')
        temporary = path.with_suffix('.jsonl.tmp')
        with temporary.open('w') as stream:
            for index in range(len(instances)):
                stream.write(json.dumps(completed[index], allow_nan=False) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
        print(f'[R58] {problem}/{split}/{method} {len(completed)}/{len(instances)}', flush=True)
    except Exception as error:
        save_json(root / 'failures' / f'{problem}__{split}__{method}.json',
            dict(**metadata, error=repr(error), traceback=traceback.format_exc(),
                 no_instance_dropped=True, training_allowed=False))
        raise


def publish(root):
    locked = read_lock(root)
    release, blocked = [], []
    for problem in PROBLEMS:
        pool = locked['pools'][problem]
        for split in ('train', 'val', 'test'):
            instances, source = load_instances(problem, split)
            input_hash = file_hash(source)
            directory = root / 'scenario_v2' / f'{problem}{split}'
            columns = []
            for method in pool:
                path = directory / 'columns' / f'{method}.jsonl'
                if not path.exists():
                    blocked.append(str(path))
                    continue
                check_column_metadata(path, column_metadata(locked, problem, method, split, len(instances), input_hash))
                values = np.full(len(instances), np.nan, dtype=np.float64)
                with path.open() as stream:
                    for line in stream:
                        row = json.loads(line)
                        i = row['index']
                        if not 0 <= i < len(instances) or np.isfinite(values[i]):
                            raise ValueError('Duplicate or invalid instance index')
                        checked = check_route(problem, instances[i], row['route'], row['cost'])
                        if not checked['feasible'] or not checked['cost_matches']:
                            raise ValueError('Independent publication check failed')
                        values[i] = checked['cost']
                if not np.isfinite(values).all() or (values <= 0).any():
                    blocked.append(str(path) + ': incomplete/invalid costs')
                    continue
                columns.append(values)
            if len(columns) != len(pool) or any(not np.isfinite(c).all() for c in columns):
                continue
            costs = np.stack(columns, axis=1)
            winner = costs.argmin(1)
            order = np.argsort(costs, axis=1, kind='stable')
            oracle = costs.min(1)
            labels = {str(i): dict(cost=costs[i].tolist(), ind=int(winner[i]),
                gap=((costs[i] - oracle[i]) / oracle[i]).tolist(), rank=order[i].tolist())
                for i in range(len(costs))}
            temporary = directory / 'raw_label.pkl.tmp'
            with temporary.open('wb') as stream:
                pickle.dump(labels, stream, protocol=pickle.HIGHEST_PROTOCOL)
            temporary.replace(directory / 'raw_label.pkl')
            link = directory / 'dataset.pkl'
            if not link.exists():
                link.symlink_to(source.resolve())
            if link.resolve() != source.resolve():
                raise ValueError('Scenario inputs changed')
            np.savez_compressed(directory / 'performance.npz', costs=costs, winner=winner,
                order=order, pool=np.asarray(pool), lock_sha256=locked['lock_sha256'])
            record = dict(problem=problem, split=split, n=len(costs), pool=pool,
                input_sha256=file_hash(source), label_sha256=file_hash(directory / 'raw_label.pkl'),
                column_sha256={m: file_hash(directory / 'columns' / f'{m}.jsonl') for m in pool})
            save_json(directory / 'manifest.json', record)
            release.append(record)
    result = dict(scenario='scenario_v2', lock_sha256=locked['lock_sha256'],
                  ready=not blocked and len(release) == 54, datasets=release, blocked=blocked,
                  test_used_only_for_label_generation=True)
    save_json(root / 'scenario_v2' / 'release.json', result)
    if not result['ready']:
        raise RuntimeError(f'Scenario incomplete: {len(blocked)} blockers; training prohibited')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['contract', 'probe', 'lock', 'generate', 'publish'], required=True)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--problem', choices=PROBLEMS)
    parser.add_argument('--solver', choices=GLOBAL_SOLVERS)
    parser.add_argument('--split', choices=['train', 'val', 'test'])
    args = parser.parse_args()
    if args.stage == 'contract':
        write_contract(args.root)
    elif args.stage == 'probe':
        if not probe(args.problem, args.solver, args.root):
            raise SystemExit(2)
    elif args.stage == 'lock':
        lock(args.root)
    elif args.stage == 'generate':
        generate(args.problem, args.solver, args.split, args.root)
    else:
        publish(args.root)


if __name__ == '__main__':
    main()

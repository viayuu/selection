"""Main-dispatched native MKP AS, with an equivalent vectorized20-ant capacity mask."""

import argparse
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

PROJECT = Path('/public/home/shiys/0selection2')
sys.path.insert(0, str(PROJECT))
from code.V4 import nonrouting_supplement as data


def native_class():
    path = data.ROOT / 'sources/deepaco_mkp_aco.py'
    if data.file_sha(path) != data.NATIVE_AS_SHA:
        raise ValueError('Pinned native source changed')
    spec = importlib.util.spec_from_file_location('_supplement_as_native', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    import torch

    class VectorizedCapacityACO(module.ACO):
        def update_knapsack(self, mask, knapsack, new_item):
            if new_item is not None:
                mask[torch.arange(self.n_ants, device=self.device), new_item] = 0
                knapsack += self.weight[new_item]
            # The native20 independent candidate tests are identical elementwise.
            infeasible = (knapsack[:, None, :] + self.weight[None, :, :] > self.n // 2).any(dim=2)
            mask.masked_fill_(infeasible, 0)
            mask[:, -1] = 1
            return mask, knapsack

    return module.ACO, VectorizedCapacityACO


def implementation_protocol():
    return dict(schema_version=1, solver_id=data.AS_ID, native_candidate_protocol=data.candidate_protocols()[data.AS_ID],
        execution_change='only vectorize independent capacity tests across20 ants; native run/gen_sol/sample/pheromone unchanged',
        implementation=dict(path=str(Path(__file__).resolve()), sha256=data.file_sha(__file__)),
        data_adapter=dict(path=str(Path(data.__file__).resolve()), sha256=data.file_sha(data.__file__)),
        unchanged=['20 ants', '100 iterations', 'max profit', 'default heuristic', 'alpha1', 'beta1',
                   'decay0.9', 'nonelitist', 'nonminmax', 'per-input seed', 'native alltime-best selection'],
        integer_certificate='independent original-integer resource/profit evaluation; never repair',
        benchmark_is_external_only=True)


def solve(instance, device='cpu', vectorized=True):
    import torch
    data.validate_instance(instance)
    original, fast = native_class()
    cls = fast if vectorized else original
    values = instance['data']
    n = len(values['profits'])
    seed = int(data.digest([data.SEED, instance['input_identity'], data.AS_ID])[:8], 16)
    target = torch.device(device)
    devices = [] if target.type == 'cpu' else [target.index if target.index is not None else torch.cuda.current_device()]
    previous_dtype = torch.get_default_dtype()
    started = time.perf_counter()
    try:
        torch.set_default_dtype(torch.float64)
        with torch.random.fork_rng(devices=devices), torch.no_grad():
            torch.manual_seed(seed)
            profits = torch.tensor(values['profits'], dtype=torch.float64, device=target)
            weights = torch.tensor(values['weights'], dtype=torch.float64, device=target).T
            capacities = torch.tensor(values['capacities'], dtype=torch.float64, device=target)
            weights = weights * (n // 2) / capacities
            native = cls(profits, weights, n_ants=20, decay=.9, alpha=1, beta=1,
                         elitist=False, min_max=False, device=str(target))
            objective, path = native.run(100)
            if target.type == 'cuda':
                torch.cuda.synchronize(target)
            path = [int(j) for j in path.cpu().tolist()]
            native_objective = float(objective.item())
            if any(j < 0 or j > n for j in path):
                raise ValueError('Native path contains non-item/dummy ID')
            solution = dict(selected=[j for j in path if j < n])
            try:
                integer_objective = data.validate_solution(instance, solution)
            except ValueError as error:
                raise data.NativeCertificateError(str(error), dict(native_path=path,
                    solution=solution, native_objective=native_objective, seed=seed)) from error
            if integer_objective != native_objective:
                raise ValueError('Original integer and native objectives differ')
            loads = [sum(row[j] for j in solution['selected']) for row in values['weights']]
            normalized_loads = weights[solution['selected']].sum(0).cpu().tolist()
    finally:
        torch.set_default_dtype(previous_dtype)
    protocol = implementation_protocol()
    return dict(solver_id=data.AS_ID, method='AntSystem', family_id='AntSystem', status='feasible',
        feasible=True, valid=True, objective=integer_objective, native_objective=integer_objective,
        objective_direction='max', solution=solution, runtime_s=time.perf_counter() - started,
        input_identity=instance['input_identity'], instance_id=instance['instance_id'],
        input_dataset_version=instance['dataset_version'], split=instance['split'],
        source_id=instance.get('source_id', instance['instance_id']), seed=seed, deterministic=False,
        label_protocol_identity=data.digest(protocol), candidate_protocol=protocol,
        execution_details=dict(device=str(target), dtype='float64', n_ants=20, iterations=100,
            capacity_vectorized=vectorized, native_path=path, native_objective=native_objective,
            original_integer_loads=loads, original_integer_capacities=values['capacities'],
            normalized_loads=normalized_loads, normalized_capacity=n // 2,
            native_source_sha256=data.NATIVE_AS_SHA, integer_certificate=True, repair=False),
        bound=None, optimality_proven=False, inference_mode='native_100_iteration_search')


def records(root, split, start, stop):
    if split == 'benchmark':
        first = data.LEGACY / 'runs/20261025_\u6784\u9020\u6cd5\u9996\u6279/\u5b9e\u4f8b\u6807\u7b7e/MKP/benchmark'
        for path in sorted(first.glob('*.jsonl')):
            sha = data.file_sha(path)
            with path.open() as stream:
                for line_number, line in enumerate(stream, 1):
                    record = json.loads(line)
                    instance = record['instance']
                    index = instance['global_index']
                    if start <= index < stop:
                        yield instance, dict(path=str(path), sha256=sha, line=line_number, external_benchmark_only=True)
        return
    receipt = data.read(root / 'domain_receipts/MKP.json')
    for shard in receipt['shards']:
        if shard['split'] != split or shard['stop_index'] <= start or shard['first_index'] >= stop:
            continue
        path = Path(shard['path'])
        if data.file_sha(path) != shard['sha256']:
            raise ValueError('Completed CPU input shard changed')
        with path.open() as stream:
            for line_number, line in enumerate(stream, 1):
                record = json.loads(line)
                instance = record['instance']
                index = instance['global_index']
                if start <= index < stop:
                    yield instance, dict(path=str(path), sha256=shard['sha256'], line=line_number)


def profile(root, native_reference=True):
    import torch
    torch.set_num_threads(1)
    outputs = []
    for cell in (0, 13, 26):
        index = next(i for i in range(27) if data.make_instance('MKP', 'witness', i)['cell_index'] == cell)
        instance = data.make_instance('MKP', 'witness', index)
        fast = solve(instance, 'cpu', True)
        reference = solve(instance, 'cpu', False) if native_reference else None
        if reference is not None:
            if fast['objective'] != reference['objective'] or fast['solution'] != reference['solution']:
                raise ValueError('Capacity vectorization changed full100-iteration native output')
            if fast['execution_details']['native_path'] != reference['execution_details']['native_path']:
                raise ValueError('Capacity vectorization changed the full native path/RNG trajectory')
        item = dict(parameters=instance['parameters'], instance_id=instance['instance_id'],
            input_identity=instance['input_identity'], vectorized_runtime_s=fast['runtime_s'],
            native_runtime_s=reference['runtime_s'] if reference else None,
            exact_native_path_equal=True if reference else None, objective=fast['objective'],
            certificate=fast['solution'], native_path=fast['execution_details']['native_path'])
        outputs.append(item)
        print(json.dumps(dict(event='as_full100_cpu_witness', **{k:v for k,v in item.items()
            if k not in ('certificate', 'native_path')})), flush=True)
    # A second fresh complete instance at each represented cell is a real small batch.
    batch_times = []
    for cell in (0, 13, 26):
        index = next(i for i in range(27) if data.make_instance('MKP', 'witness', i)['cell_index'] == cell) + 27
        instance = data.make_instance('MKP', 'witness', index)
        label = solve(instance, 'cpu', True)
        batch_times.append(dict(parameters=instance['parameters'], runtime_s=label['runtime_s'],
            input_identity=instance['input_identity'], objective=label['objective'], certificate=label['solution']))
    mean = sum(item['runtime_s'] for item in batch_times) / len(batch_times)
    report = dict(protocol=implementation_protocol(), full_iterations=100, ants=20, device='cpu',
        paired_native_witnesses=outputs, small_batch=batch_times, small_batch_mean_s=mean,
        rough_12970_instance_cpu_hours=12970 * mean / 3600,
        estimate_scope='three represented cells only; includes fresh second instances, NOT a GPU or all27-cell estimate',
        GPU_time_unmeasured=True, created_utc=datetime.now(timezone.utc).isoformat())
    data.write(root / 'as_vectorized_cpu_witness.json', report, immutable=True)
    return report


def run(args):
    import torch
    torch.set_num_threads(1)
    root = data.owned(args.root)
    if args.device != 'cpu' and not args.main_authorized_gpu:
        raise PermissionError('Main-owned GPU dispatch requires --main-authorized-gpu')
    total = 270 if args.split == 'benchmark' else data.COUNTS['MKP'][args.split]
    stop = args.stop if args.stop is not None else total
    if not 0 <= args.start < stop <= total:
        raise ValueError('Invalid exact input interval')
    protocol = implementation_protocol()
    data.write(root / 'native_as_dispatch_protocol.json', protocol, immutable=True)
    completed, started, timings, labels_manifest = 0, time.perf_counter(), [], []
    for instance, source in records(root, args.split, args.start, stop):
        index = instance['global_index']
        path = root / 'native_as_labels' / args.split / f'{index:06d}.json'
        if path.exists():
            label = data.read(path)
            if (label['input_identity'] != instance['input_identity'] or label['solver_id'] != data.AS_ID
                    or data.validate_solution(instance, label['solution']) != label['objective']):
                raise ValueError('Existing AS label does not bind this certified input')
        else:
            try:
                label = solve(instance, args.device)
            except Exception as error:
                data.write(root / 'native_as_failures' / args.split / f'{index:06d}.json',
                    dict(status='failed', first_error=str(error), error_type=type(error).__name__,
                         evidence=getattr(error, 'evidence', None), input_identity=instance['input_identity'],
                         source=source, automatic_retry=False))
                raise
            label['source_reference'] = source
            data.write(path, label, immutable=True)
        timings.append(label['runtime_s'])
        completed += 1
        labels_manifest.append(dict(path=str(path), sha256=data.file_sha(path), input_identity=instance['input_identity']))
        if completed % 5 == 0 or completed == stop - args.start:
            print(json.dumps(dict(event='as_label_complete', split=args.split, index=index, completed=completed,
                total=stop - args.start, mean_s=sum(timings)/len(timings), elapsed_s=time.perf_counter()-started)), flush=True)
    if completed != stop - args.start:
        raise ValueError('Missing native input tail; refuse complete receipt')
    receipt = dict(complete=True, solver_id=data.AS_ID, split=args.split, start=args.start, stop=stop,
        completed=completed, device=args.device, protocol=protocol, labels=labels_manifest,
        mean_s=sum(timings)/len(timings), elapsed_s=time.perf_counter() - started,
        benchmark_is_external_only=args.split == 'benchmark', completed_utc=datetime.now(timezone.utc).isoformat())
    data.write(root / 'native_as_receipts' / f'{args.split}_{args.start:06d}_{stop:06d}.json', receipt, immutable=True)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('profile', 'label'))
    parser.add_argument('--root', type=Path, default=data.ROOT)
    parser.add_argument('--split', choices=('train', 'val', 'test', 'benchmark'), default='train')
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--stop', type=int)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--main-authorized-gpu', action='store_true')
    args = parser.parse_args()
    result = profile(data.owned(args.root)) if args.command == 'profile' else run(args)
    print(json.dumps({k:v for k,v in result.items() if k not in ('labels', 'paired_native_witnesses', 'small_batch')}, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()

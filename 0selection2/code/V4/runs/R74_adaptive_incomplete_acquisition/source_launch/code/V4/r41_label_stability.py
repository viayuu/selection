"""R41 sampling is size-only and precedes test; repeats never replace historical labels."""

import argparse
import ast
import json
import pickle
import shutil
import subprocess
import sys
import time
import types
from pathlib import Path

import numpy as np
import torch

from code.unified_selector.registry import DATA_ROOT
from .multitask_probe import dump, file_hash, state_hash
from .performance_targets import read_raw_costs
from .tsp_learnability import size_boundaries


ROOT = Path('code/V4/runs/R41_signal_audit')
HOME = Path('/public/home/shiys')
BRIDGE = HOME / 'easynco_v3_bridge/scripts'
OVRPTW_FIELDS = ('depot_xy', 'node_xy', 'node_demand', 'service_time', 'tw_start', 'tw_end')


def snapshot_stage(root, stage):
    base = Path(__file__).parent
    names = ('r41_frozen_evaluation.py', 'r41_label_stability.py', 'r41_analysis.py', 'test_r41.py',
             'run_v4_r41.sh', 'V4Model.py', 'dual_stream.py', 'direct_selector.py', 'solver_features.py',
             'local_geometry.py', 'tensor_loader.py', 'train.py', 'multitask_probe.py',
             'performance_targets.py', 'performance_evaluation.py', 'r40_experiment.py')
    sources = [base / name for name in names] + [BRIDGE / 'reld_bridge_common.py']
    hashes = {str(path.resolve()): file_hash(path) for path in sources}
    directory = root / 'source_snapshots' / state_hash(hashes)
    directory.mkdir(parents=True, exist_ok=True)
    for source in sources:
        target = directory / source.name
        if target.exists():
            if file_hash(target) != hashes[str(source.resolve())]:
                raise ValueError('An immutable source snapshot changed')
        else:
            shutil.copy2(source, target)
    dump(directory / 'hashes.json', hashes)
    path = root / 'source_revisions.json'
    revisions = json.loads(path.read_text()) if path.exists() else []
    revisions.append(dict(stage=stage, snapshot=str(directory), source_hashes=hashes,
                          captured_at=time.strftime('%Y-%m-%d %H:%M:%S %z'),
                          scope='Selected-source snapshot, not the full dependency closure. Captured before this stage, not retroactively.'))
    dump(path, revisions)
    return directory


def reld_params_contract(entry):
    source = Path(entry['original_entry'])
    if file_hash(source) != entry['source_sha256'] or file_hash(entry['checkpoint']['path']) != entry['checkpoint']['sha256']:
        raise ValueError('Pinned original RELD source or weight changed')
    module = relocated_reld_module(source)
    key = 'reld_mtl' if entry['solver'] == 'RELD_MTL' else 'reld_moel'
    checkpoint = torch.load(entry['checkpoint']['path'], map_location='cpu', weights_only=False)
    return module.model_init_params(module.MODEL_SPECS[key], checkpoint['problem'])


def validate_saved_run(entry, sample, split, witness, costs, indices, instances, seen_pids, repeat, precheck=False):
    expected_indices = sample['precheck_indices'] if precheck else sample['indices']
    np.testing.assert_array_equal(indices, expected_indices)
    if costs.shape != (len(expected_indices),) or not np.isfinite(costs).all() or (costs <= 0).any():
        raise ValueError('Invalid solver costs or coverage')
    expected_seed = 2 if entry['randomness'] == 'deterministic' else repeat + 2
    if any(witness.get(key) != value for key, value in dict(problem=entry['problem'], solver=entry['solver'],
          split=split, seed=expected_seed, precheck=precheck, model_eval=True, independent_process=True,
          augmentation=entry['decode_budget']['aug_factor'], sample_size=entry['decode_budget']['sample_size']).items()):
        raise ValueError('Stored solver output violates its run contract')
    if witness['checkpoint'] != entry['checkpoint'] or witness['source_sha256'] != entry['source_sha256']:
        raise ValueError('Stored solver output belongs to a different source/checkpoint')
    if witness['params'] != entry['runtime_params_contract']:
        raise ValueError('Stored solver decoding/model parameters changed')
    pid = witness.get('pid')
    if not isinstance(pid, int) or pid <= 0 or pid in seen_pids:
        raise ValueError('Solver repeats lack distinct independent process records')
    chunks = witness.get('input_witness', [])
    covered = [i for chunk in chunks for i in chunk['original_indices']]
    if not chunks or sorted(covered) != sorted(expected_indices) or len(set(covered)) != len(covered):
        raise ValueError('Input witness does not cover exactly the locked sample')
    originals = dict(zip(sample['indices'], instances))
    for chunk in chunks:
        if not chunk['input_matches_original'] or tuple(chunk['fields']) != OVRPTW_FIELDS:
            raise ValueError('Solver input witness uses different fields')
        selected = [originals[i] for i in chunk['original_indices']]
        if not 0 < len(selected) <= entry['decode_budget']['original_batch_size'] or any(
                len(item['node_xy']) != chunk['scale'] for item in selected):
            raise ValueError('Solver input scale or batch budget changed')
        tensors = tuple(torch.stack([torch.as_tensor(item[field], dtype=torch.float32) for item in selected])
                        for field in OVRPTW_FIELDS)
        if state_hash(tensors) != chunk['input_hash']:
            raise ValueError('Solver input hash does not match original fields')
    seen_pids.add(pid)


def make_manifest(root):
    if (root / 'solver_manifest.json').exists():
        manifest = json.loads((root / 'solver_manifest.json').read_text())
        for entry in manifest['solvers'].values():
            if entry.get('backend') == 'reld':
                params = reld_params_contract(entry)
                if entry.get('runtime_params_contract', params) != params:
                    raise ValueError('Original RELD parameter contract changed')
                entry['runtime_params_contract'] = params
        dump(root / 'solver_manifest.json', manifest)
        return manifest
    samples = json.loads((root / 'audit_indices.json').read_text())
    records = {}
    for problem, record in samples['problems'].items():
        for solver in record['splits']['train']['pool']:
            entry = dict(problem=problem, solver=solver, status='blocked_provenance', randomness='unknown',
                 original_entry=None, checkpoint=None, decode_budget=None, normalization=None,
                 blocked=['Original solver checkpoint binding, decoding budget and invocation are not fully recoverable.'],
                 cost_files={s: dict(path=str(DATA_ROOT / f'{problem}{s}/results' / f'result_{solver}.txt'),
                                    sha256=file_hash(DATA_ROOT / f'{problem}{s}/results' / f'result_{solver}.txt'))
                             for s in ('train', 'val')})
            if problem == 'TSP':
                legacy_name = {'BQ': 'bq', 'OMNI': 'Omni'}.get(solver, solver)
                entry['label_ancestor'] = f'/public/home/shiys/0selection/neural-solver-selection/datasets/TSP*/results/result_{legacy_name}.txt'
                entry['blocked'].append('Result ancestry is verified but does not establish a solver-side generation recipe.')
                if solver == 'OMNI':
                    entry['train_only_evidence'] = str(HOME / 'EasyNCO/results/nss_eval/omni_tsp/TSPtrain/eval.log')
                    entry['blocked'].append('OMNI train has an EasyNCO recipe; validation came from NSS and is not the same proven recipe.')
            elif solver in ('RELD_MTL', 'RELD_MOEL'):
                name = 'reld_mtl' if solver == 'RELD_MTL' else 'reld_moe_light'
                path = HOME / f'reld-nco-main/Multi-Task/pretrained/{name}/epoch-5000.pt'
                entry.update(status='verified_static', randomness='deterministic', backend='reld',
                     checkpoint=dict(path=str(path), sha256=file_hash(path)),
                     original_entry=str(BRIDGE / 'reld_bridge_common.py'), source_sha256=file_hash(BRIDGE / 'reld_bridge_common.py'),
                     functions=['collate_problem_batch', 'instantiate_reld_model', '_run_no_aug_batch'],
                     decode_budget=dict(eval_type='argmax', starts='N customers', aug_factor=1, sample_size=1,
                                        aggregation='minimum cost over N starts', original_batch_size=128),
                     normalization='Existing float32 conversion; already-normalized demand/capacity=1; original depot/customer/TW/service fields.',
                     historical_evidence=str(HOME / 'easynco_v3_bridge/exports/reld_nss_like_no_aug_v1/_bridge_state/reld_bridge_events.jsonl'),
                     portability='Only relocate the old filesystem root in the original module AST; no algorithm/budget edits.',
                     blocked=[], limitations=['Historical checkpoint digest and full CLI argv unavailable; current named weight is pinned.',
                                              'Historical RNG seed unavailable; inspected algorithm uses deterministic argmax.'])
                entry['runtime_params_contract'] = reld_params_contract(entry)
            elif solver in ('MTPOMO', 'MVMOE'):
                entry['original_entry'] = str(HOME / 'EasyNCO/eval.py')
                entry['blocked'] = ['Historical shard config has top-level greedy but omits module decoder_strategy; effective historical sampling/greedy remains unverified.']
            else:
                entry['original_entry'] = str(BRIDGE / 'paper_source_routefinder_runner.py')
                entry['blocked'] = ['Historical rf_easnco_env/rl4co dependency binding and effective decode/RNG behavior are not yet verified.']
            records[f'{problem}/{solver}'] = entry
    manifest = dict(solvers=records, review_independence='same-family', acceptance_status='provisional',
                    prohibition='No current-default or differently budgeted solver substitutes.',
                    note='Static source evidence is not runtime verification. Historical checkpoint hashes were not retained.')
    dump(root / 'solver_manifest.json', manifest)
    return manifest


def proportional_indices(nodes, cuts, count, seed):
    groups = np.searchsorted(cuts, nodes, side='left')
    counts = np.bincount(groups, minlength=len(cuts) + 1)
    if count > len(nodes):
        raise ValueError('Sample size exceeds the dataset')
    exact = counts * count / len(nodes)
    quotas = np.floor(exact).astype(int)
    for group in np.argsort(-(exact - quotas), kind='stable')[:count - quotas.sum()]:
        quotas[group] += 1
    rng = np.random.default_rng(seed)
    values = np.concatenate([rng.choice(np.flatnonzero(groups == group), size=quota, replace=False)
                             for group, quota in enumerate(quotas)])
    return rng.permutation(values), groups, quotas


def load_instances(problem, split):
    with (DATA_ROOT / f'{problem}{split}' / 'dataset.pkl').open('rb') as stream:
        instances = pickle.load(stream)
    nodes = np.array([len(item[0]) if problem == 'TSP' else len(item['node_xy']) + 1 for item in instances])
    return instances, nodes


def prepare_indices(root):
    if (root / 'audit_indices.json').exists():
        return json.loads((root / 'audit_indices.json').read_text())
    if (root / 'test_results.json').exists():
        raise ValueError('The label audit sampling must be locked before viewing R41 test results')
    result = dict(seed=2, selection='proportional size quartiles, no winner/model/gap conditioning',
                  size_boundaries_fit='training nodes only, reused on validation', problems={})
    for problem in ('TSP', 'OVRPTW'):
        _, train_nodes = load_instances(problem, 'train')
        cuts = size_boundaries(train_nodes)
        record = dict(size_boundaries=cuts.tolist(), splits={})
        for split in ('train', 'val'):
            instances, nodes = load_instances(problem, split)
            seed = 2 + 1009 * ('OVRPTW' == problem) + 100003 * (split == 'val')
            indices, groups, quotas = proportional_indices(nodes, cuts, 64, seed)
            precheck, _, _ = proportional_indices(nodes[indices], cuts, 8, seed + 17)
            raw = read_raw_costs(problem, split)
            record['splits'][split] = dict(indices=indices.tolist(), precheck_indices=indices[precheck].tolist(),
                 nodes=nodes[indices].tolist(), bucket_counts=quotas.tolist(),
                 dataset_hash=raw['data_hash'], label_hash=raw['label_hash'], pool=raw['pool'], pool_ids=raw['pool_ids'],
                 instance_hashes=[state_hash(instances[i]) for i in indices])
        result['problems'][problem] = record
    dump(root / 'audit_indices.json', result)
    print('[audit locked] TSP and OVRPTW: 64 train + 64 val each, size-proportional; no test-informed selection', flush=True)
    return result


def export_inputs(root):
    samples = json.loads((root / 'audit_indices.json').read_text())
    directory = root / 'audit_inputs'
    directory.mkdir(exist_ok=True)
    entries = {}
    for problem, record in samples['problems'].items():
        for split, sample in record['splits'].items():
            source = DATA_ROOT / f'{problem}{split}' / 'dataset.pkl'
            if file_hash(source) != sample['dataset_hash']:
                raise ValueError('Dataset changed after sampling was locked')
            instances, _ = load_instances(problem, split)
            selected = [instances[index] for index in sample['indices']]
            if [state_hash(item) for item in selected] != sample['instance_hashes']:
                raise ValueError('Selected instance identity changed')
            path = directory / f'{problem}_{split}.pkl'
            with path.open('wb') as stream:
                pickle.dump(selected, stream, protocol=pickle.HIGHEST_PROTOCOL)
            entries[f'{problem}/{split}'] = dict(path=str(path), sha256=file_hash(path),
                  original_indices=sample['indices'], instance_hashes=sample['instance_hashes'], source=str(source))
    dump(root / 'audit_input_manifest.json', entries)
    return entries


def relocated_reld_module(source):
    tree = ast.parse(source.read_text(), filename=str(source))
    replacements = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and node.value == '/public/home/zhoucl/shiys':
            node.value = str(HOME)
            replacements += 1
    if replacements != 1:
        raise ValueError('Expected exactly one old-root constant in the original RELD module')
    module = types.ModuleType('r41_original_reld_bridge')
    module.__file__ = str(source)
    sys.modules[module.__name__] = module
    exec(compile(ast.fix_missing_locations(tree), str(source), 'exec'), module.__dict__)
    return module


def run_worker(args):
    manifest = json.loads((args.root / 'solver_manifest.json').read_text())
    entry = manifest['solvers'][f'{args.problem}/{args.solver}']
    if entry['status'] not in ('verified_static', 'runtime_verified') or entry['backend'] != 'reld':
        raise ValueError('Only original-config-verified backends may run')
    source, checkpoint = Path(entry['original_entry']), entry['checkpoint']
    if file_hash(source) != entry['source_sha256'] or file_hash(checkpoint['path']) != checkpoint['sha256']:
        raise ValueError('Original backend source or pinned weight changed')
    sample = json.loads((args.root / 'audit_indices.json').read_text())['problems'][args.problem]['splits'][args.split]
    with (args.root / 'audit_inputs' / f'{args.problem}_{args.split}.pkl').open('rb') as stream:
        instances = pickle.load(stream)
    if [state_hash(item) for item in instances] != sample['instance_hashes']:
        raise ValueError('Worker input is not the original locked sample')
    positions = [sample['indices'].index(i) for i in sample['precheck_indices']] if args.precheck else list(range(64))
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(1)
    module = relocated_reld_module(source)
    key = 'reld_mtl' if args.solver == 'RELD_MTL' else 'reld_moel'
    if str(module.MODEL_SPECS[key]['checkpoint']) != checkpoint['path']:
        raise ValueError('Original RELD factory resolved a different checkpoint')
    device = torch.device('cuda:0')
    model, params = module.instantiate_reld_model(key, device)
    env_cls = module.get_env_class(args.problem)
    costs = np.full(len(positions), np.nan, dtype=np.float64)
    started = time.monotonic()
    grouped = {}
    for slot, position in enumerate(positions):
        grouped.setdefault(len(instances[position]['node_xy']), []).append((slot, position))
    input_witness = []
    for scale, members in grouped.items():
        for start in range(0, len(members), 128):
            chunk = members[start:start + 128]
            slots, selected = zip(*chunk)
            batch, resolved_scale = module.collate_problem_batch(args.problem + args.split, instances, selected)
            if resolved_scale != scale:
                raise ValueError('Solver input scale changed')
            # Check every field after the exact historical float32 input conversion.
            fields = OVRPTW_FIELDS
            for tensor, field in zip(batch, fields):
                expected = torch.stack([torch.as_tensor(instances[i][field], dtype=torch.float32) for i in selected])
                torch.testing.assert_close(tensor, expected, atol=0, rtol=0)
            values = module._run_no_aug_batch(model, env_cls, batch, scale, device).numpy().astype(np.float64)
            if not np.isfinite(values).all() or (values <= 0).any():
                raise ValueError('Original solver returned invalid cost')
            costs[list(slots)] = values
            input_witness.append(dict(scale=scale, original_indices=[sample['indices'][i] for i in selected],
                 input_hash=state_hash(batch), fields=fields, input_matches_original=True))
            print(f'[solver] {args.solver}/{args.split} n={scale} completed={np.isfinite(costs).sum()}/{len(costs)}', flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, costs=costs, indices=np.array([sample['indices'][i] for i in positions]))
    dump(args.output.with_suffix('.json'), dict(solver=args.solver, problem=args.problem, split=args.split,
         seed=args.seed, pid=__import__('os').getpid(), independent_process=True, seconds=time.monotonic()-started,
         checkpoint=checkpoint, source_sha256=entry['source_sha256'], params=params, input_witness=input_witness,
         augmentation=1, sample_size=1, model_eval=True, precheck=args.precheck,
         gpu=torch.cuda.get_device_name(0), torch_version=torch.__version__))


def execute_repeats(root, precheck=False):
    manifest = json.loads((root / 'solver_manifest.json').read_text())
    entries = [entry for entry in manifest['solvers'].values() if entry['status'] in ('verified_static', 'runtime_verified')]
    for entry in entries:
        problem, solver = entry['problem'], entry['solver']
        if not precheck:
            directory = root / 'solver_runs' / problem / 'train' / solver
            if not all((directory / f'precheck_{r}.npz').exists() for r in range(2)):
                raise ValueError(f'{solver}: the eight-instance independent-process precheck must finish first')
        for split in (('train',) if precheck else ('train', 'val')):
            for repeat in range(2 if entry['randomness'] == 'deterministic' else 3):
                directory = root / 'solver_runs' / problem / split / solver
                directory.mkdir(parents=True, exist_ok=True)
                path = directory / f'{"precheck" if precheck else "repeat"}_{repeat}.npz'
                if path.exists():
                    continue
                command = [sys.executable, '-u', '-m', 'code.V4.r41_label_stability', '--stage', 'worker',
                           '--root', str(root), '--problem', problem, '--solver', solver, '--split', split,
                           '--seed', str(2 if entry['randomness'] == 'deterministic' else repeat + 2), '--output', str(path)]
                if precheck:
                    command.append('--precheck')
                print('[launch] ' + ' '.join(command), flush=True)
                with path.with_suffix('.log').open('w') as stream:
                    result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
                if result.returncode:
                    entry['status'] = 'blocked_runtime'
                    entry['blocked'].append(f'Original-config process failed; inspect {path.with_suffix(".log")}')
                    dump(root / 'solver_manifest.json', manifest)
                    print(f'[blocked runtime] {solver}/{split} exit={result.returncode}', flush=True)
                    break
            if entry['status'] == 'blocked_runtime':
                break
        if entry['status'] != 'blocked_runtime':
            entry['status'] = 'runtime_verified'
        dump(root / 'solver_manifest.json', manifest)
    collect_costs(root)


def collect_costs(root):
    samples = json.loads((root / 'audit_indices.json').read_text())
    manifest = make_manifest(root)
    values, prechecks, seen_pids = {}, [], set()
    for problem, record in samples['problems'].items():
        for split, sample in record['splits'].items():
            indices = np.array(sample['indices'])
            raw = read_raw_costs(problem, split)
            if raw['data_hash'] != sample['dataset_hash'] or raw['label_hash'] != sample['label_hash']:
                raise ValueError('Original dataset/labels changed during the audit')
            with (root / 'audit_inputs' / f'{problem}_{split}.pkl').open('rb') as stream:
                instances = pickle.load(stream)
            if [state_hash(item) for item in instances] != sample['instance_hashes']:
                raise ValueError('Original locked solver inputs changed')
            costs = np.full((3, 64, len(sample['pool'])), np.nan)
            for j, solver in enumerate(sample['pool']):
                directory = root / 'solver_runs' / problem / split / solver
                for repeat in range(3):
                    path = directory / f'repeat_{repeat}.npz'
                    if path.exists():
                        witness = json.loads(path.with_suffix('.json').read_text())
                        entry = manifest['solvers'][f'{problem}/{solver}']
                        with np.load(path) as saved:
                            validate_saved_run(entry, sample, split, witness, saved['costs'], saved['indices'],
                                               instances, seen_pids, repeat)
                            costs[repeat, :, j] = saved['costs']
                    precheck = directory / f'precheck_{repeat}.npz'
                    if precheck.exists():
                        with np.load(precheck) as saved:
                            entry = manifest['solvers'][f'{problem}/{solver}']
                            witness = json.loads(precheck.with_suffix('.json').read_text())
                            validate_saved_run(entry, sample, split, witness, saved['costs'], saved['indices'],
                                               instances, seen_pids, repeat, precheck=True)
                            historical = raw['costs'][saved['indices'], j]
                            prechecks.append(dict(problem=problem, split=split, solver=solver, repeat=repeat,
                                 n=len(saved['indices']), seconds=json.loads(precheck.with_suffix('.json').read_text())['seconds'],
                                 mean_abs_historical_error=float(np.abs(saved['costs'] - historical).mean()),
                                 max_abs_historical_error=float(np.abs(saved['costs'] - historical).max()),
                                 projected_full_64_seconds=json.loads(precheck.with_suffix('.json').read_text())['seconds'] * 8))
            prefix = f'{problem}_{split}'
            values[prefix + '_costs'] = costs
            values[prefix + '_observed'] = np.isfinite(costs)
            values[prefix + '_indices'] = indices
            values[prefix + '_historical'] = raw['costs'][indices]
            values[prefix + '_winner'] = raw['winner'][indices]
            with np.load(root / 'audit_predictions' / f'{problem}_{split}.npz') as saved:
                np.testing.assert_array_equal(saved['indices'], indices)
                values[prefix + '_pred'] = saved['pred']
    np.savez_compressed(root / 'repeated_costs.npz', **values)
    dump(root / 'solver_repeat_integrity.json', dict(passed=True, independent_processes=len(seen_pids),
         checked_observed_costs=sum(int(np.isfinite(value).sum()) for key, value in values.items() if key.endswith('_costs')),
         source_and_weights_pinned=True, full_decode_contract=True, exact_original_input_coverage=True))
    dump(root / 'precheck_summary.json', prechecks)
    blocked = [entry for entry in manifest['solvers'].values() if entry['blocked']]
    lines = ['# R41 blocked original solver dependencies/provenance', '',
             'Available weights alone do not prove that they generated the existing labels. No substitute configuration was run.', '',
             '| Problem | Solver | Status | Missing evidence / dependency |', '| --- | --- | --- | --- |']
    for entry in blocked:
        lines.append(f'| {entry["problem"]} | {entry["solver"]} | {entry["status"]} | {"; ".join(entry["blocked"])} |')
    (root / 'blocked_dependencies.md').write_text('\n'.join(lines) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--stage', choices=('prepare', 'manifest', 'precheck', 'run', 'collect', 'worker'), default='prepare')
    parser.add_argument('--problem', default='OVRPTW')
    parser.add_argument('--solver', default='RELD_MTL')
    parser.add_argument('--split', choices=('train', 'val'), default='train')
    parser.add_argument('--seed', type=int, default=2)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--precheck', action='store_true')
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    snapshot_stage(args.root, 'label-' + args.stage)
    if args.stage == 'prepare':
        prepare_indices(args.root)
        export_inputs(args.root)
    elif args.stage == 'manifest':
        make_manifest(args.root)
    elif args.stage == 'worker':
        run_worker(args)
    elif args.stage == 'collect':
        collect_costs(args.root)
    else:
        execute_repeats(args.root, precheck=args.stage == 'precheck')


if __name__ == '__main__':
    main()

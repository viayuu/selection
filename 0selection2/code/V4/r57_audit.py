"""Inventory all legal train/validation deployments without inventing recipes."""

import argparse
import csv
import json
import subprocess
from pathlib import Path

import numpy as np

from code.unified_selector.registry import DATA_ROOT, GLOBAL_SOLVERS, POOLS, PROBLEMS, S2I
from .multitask_probe import dump, file_hash
from .performance_targets import read_raw_costs
from .r48_common import write_csv
from .r53_data import BASELINE
from .r57_signal import ROOT


MANIFEST = Path('code/V4/solver_source_manifest.json')
EVIDENCE = Path('code/V4/runs/R45_solver_code_embeddings') / '\u6765\u6e90\u8865\u5168\u6838\u5bf9\u8bc1\u636e.json'
NSS_UNKNOWN = {'BQ', 'DIFUSCO', 'DIFUSCO500', 'ELG', 'LEHD', 'T2T', 'T2T500'}


def deployment_rows(manifest, solver, problem, split):
    entry = next(row for row in manifest['solvers'] if row['solver_name'] == solver)
    if entry['solver_id'] != S2I[solver]:
        raise ValueError('Global solver ID mismatch')
    values = [row for row in entry['deployments'] if problem in row['problems'] and split in row['splits']]
    if not values:
        raise ValueError(f'Missing manifest entry: {problem}/{solver}/{split}')
    return values


def executable_status(solver, problem):
    if solver in ('RELD_MOEL', 'RELD_MTL'):
        return 'pinned_R48_R54_recipe_available'
    if solver in ('GLOP', 'MATNET', 'MATPOENET', 'ICAM'):
        return 'archived_recipe_current_source_only'
    if solver in ('ICAM_ATSP', 'UNICO_MatPOENet', 'RELD_CVRP'):
        return 'explicit_bridge_current_source_only'
    if solver in ('MTPOMO', 'MVMOE') and problem not in ('CVRP',):
        return 'blocked_effective_decoder_and_environment_revision'
    if solver in ('MoSES_RF', 'MoSES_CaDA', 'RouteFinder'):
        return 'blocked_historical_decoder_environment_and_resolved_policy'
    return 'blocked_original_NSS_recipe_or_split_binding'


def runtime_evidence(root, solver):
    path = root / 'runtime_preflight.csv'
    if not path.exists():
        return dict(preflight_status='not_attempted', runtime_log='')
    family = 'RELD_PAIR' if solver in ('RELD_MOEL', 'RELD_MTL') else solver
    with path.open(newline='') as stream:
        row = next((r for r in csv.DictReader(stream) if r['solver'] == family), None)
    if row is None:
        return dict(preflight_status='not_attempted', runtime_log='')
    return dict(preflight_status=row['status'], runtime_log=row['log'])


def resolve_file(manifest, value):
    if ':' not in value or value.startswith('/'):
        return Path(value)
    root, relative = value.split(':', 1)
    return Path(manifest['roots'][root]) / relative


def input_contract(problem):
    if problem == 'TSP':
        return dict(raw=['xy'], selector=['xy', 'true_size', 'problem_description'],
            absent=['designated_physical_start', 'designated_physical_target', 'runtime_recipe'],
            transformations='coordinates converted to FP32; no start/end marker or node index')
    if problem == 'CVRP':
        return dict(raw=['depot', 'loc', 'demand'], selector=['xy', 'demand', 'depot_flag', 'true_size', 'problem_description'],
            absent=['physical_start_subset', 'runtime_recipe'],
            transformations='explicit depot first; normalized demand is passed unchanged; capacity convention fixed, no raw capacity field')
    if problem == 'ATSP':
        return dict(raw=['directed_cost_matrix'], selector=['directed_cost_matrix', 'true_size', 'problem_description'],
            absent=['runtime_seed', 'NN_initialization_tie_rule', 'runtime_recipe'],
            transformations='FP32 directed matrix; matrix feature reductions are not assumed permutation-invariant without replay')
    return dict(raw=['depot_xy', 'node_xy', 'node_demand', 'capacity', 'route_limit?', 'service_time?', 'tw_start?', 'tw_end?'],
        selector=['xy', 'signed_demand', 'depot_flag', 'route_limit_if_L', 'customer_service_and_TW_if_TW', 'true_size', 'problem_description'],
        absent=['explicit_capacity_token', 'depot_deadline', 'backhaul_class', 'speed', 'physical_start_subset', 'runtime_recipe'],
        transformations='capacity=1 and route_limit=3 checked from actual data; depot feature slots for TW/service are zero, not a separately specified deadline')


def run(root=ROOT):
    root.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(MANIFEST.read_text())
    evidence = json.loads(EVIDENCE.read_text())
    if [row['solver_name'] for row in manifest['solvers']] != GLOBAL_SOLVERS:
        raise ValueError('Manifest solver ordering differs from registry')
    bindings = {(row['problem'], row['split'], row['method']): row for row in evidence['label_bindings']
                if row['split'] in ('train', 'val')}
    full, flat, source_inventory, blocked = [], [], {}, []
    args = json.loads((BASELINE / 'args.json').read_text())
    for problem in PROBLEMS:
        raws = {split: read_raw_costs(problem, split) for split in ('train', 'val')}
        for slot, solver in enumerate(POOLS[problem]):
            split_records = {}
            for split, raw in raws.items():
                deployments = deployment_rows(manifest, solver, problem, split)
                binding = bindings[(problem, split, solver)]
                target = DATA_ROOT / (problem + split) / 'results' / ('result_' + solver + '.txt')
                with target.open(newline='') as stream:
                    parsed = list(csv.reader(stream))
                ids = np.asarray([int(row[0]) for row in parsed])
                values = np.asarray([float(row[1]) for row in parsed], np.float64)
                np.testing.assert_array_equal(np.sort(ids), np.arange(len(raw['winner'])))
                np.testing.assert_array_equal(values[np.argsort(ids)], raw['costs'][:, slot])
                if file_hash(target) != binding['target_sha256']:
                    raise ValueError('Prior ancestry evidence no longer binds to current result column')
                ancestors = [row for row in binding['ancestors'] if row['status'] == 'match']
                if not ancestors:
                    raise ValueError('No exact ancestor match in existing evidence')
                for deployment in deployments:
                    implementation = manifest['implementations'][deployment['implementation']]
                    for role in implementation.get('views', {}).values():
                        for source in role.get('sources', []):
                            path = resolve_file(manifest, source['file'])
                            source_inventory[str(path)] = dict(exists=path.exists(),
                                sha256=file_hash(path) if path.is_file() else None,
                                hash_scope='current file, not historical run version')
                split_records[split] = dict(n=len(raw['winner']), data_hash=raw['data_hash'],
                    label_hash=raw['label_hash'], column_path=str(target.resolve()), column_hash=file_hash(target),
                    raw_column_matches_result_exactly=True, ancestry=ancestors,
                    native_winner_not_exact_minimum=int((raw['costs'][np.arange(len(raw['winner'])), raw['winner']] != raw['costs'].min(1)).sum()),
                    ancestry_scope='previous exact source match re-bound to unchanged current result file; no test access',
                    deployments=deployments)
            train_d, val_d = [split_records[split]['deployments'] for split in ('train', 'val')]
            signature = lambda ds: [dict(implementation=d['implementation'], resolved_config=d.get('resolved_config', {}),
                checkpoint=d.get('checkpoint'), checkpoints=d.get('checkpoints'),
                selection=d.get('checkpoint_selection'), scales=d.get('scales'), scale_range=d.get('scale_range')) for d in ds]
            same_description = signature(train_d) == signature(val_d)
            recipe_status = executable_status(solver, problem)
            runtime = runtime_evidence(root, solver)
            status = recipe_status
            if runtime['preflight_status'] not in ('passed', 'not_attempted'):
                status = runtime['preflight_status']
                if not status.startswith('blocked'):
                    status = 'blocked_' + status
            gaps = sorted(set(gap for d in (*train_d, *val_d) for gap in d.get('gaps', [])))
            record = dict(problem=problem, solver=solver, solver_id=S2I[solver], cost_column=slot,
                selector_input=input_contract(problem), splits=split_records,
                same_recovered_train_val_description=same_description,
                fully_certified_historical_deployment=False, recipe_status=recipe_status,
                runtime_status=status, **runtime, remaining_gaps=gaps)
            full.append(record)
            flat.append(dict(problem=problem, solver=solver, solver_id=S2I[solver], cost_column=slot,
                train_n=split_records['train']['n'], val_n=split_records['val']['n'],
                raw_result_column_alignment=True, historical_binding=';'.join(sorted({d['binding_status'] for d in (*train_d, *val_d)})),
                same_recovered_train_val_description=same_description,
                historical_train_val_execution_consistency='not_certified',
                recipe_status=recipe_status, runtime_status=status, **runtime,
                train_origins=';'.join(a['origin'] for a in split_records['train']['ancestry']),
                val_origins=';'.join(a['origin'] for a in split_records['val']['ancestry']),
                selector_missing_execution_conditions=';'.join(input_contract(problem)['absent']),
                remaining_gaps=' | '.join(gaps)))
            if status.startswith('blocked'):
                blocked.append(flat[-1])
    dump(root / 'deployment_audit.json', dict(schema='r57.deployment_audit.v1',
        baseline_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        original_manifest=str(MANIFEST.resolve()), manifest_sha256=file_hash(MANIFEST),
        evidence_sha256=file_hash(EVIDENCE), splits=['train', 'val'], legal_deployments=len(full),
        label_columns_checked=2 * len(full), labels_replaced=False, historical_defaults_assumed=False,
        selector=dict(checkpoint=str((BASELINE / 'best.pt').resolve()), checkpoint_sha256=file_hash(BASELINE / 'best.pt'),
            epoch=json.loads((BASELINE / 'best_eval.json').read_text())['epoch'],
            ignore_coord_dist=args['model_params']['ignore_coord_dist'],
            index_guessed_distribution_is_used=not args['model_params']['ignore_coord_dist']),
        deployments=full))
    write_csv(root / 'deployment_audit.csv', flat)
    write_csv(root / 'blocked_runtime_deployments.csv', blocked)
    dump(root / 'current_source_inventory.json', source_inventory)
    text = ['# R57 Runtime Recipe Gaps', '',
        'All 128 legal problem/solver pairs and 256 train/val cost columns are inventoried.',
        'Exact cost ancestry does not certify a historical invocation. No current defaults are used as historical settings.', '',
        'Recipe availability and actual preflight status are recorded separately; failed native-library preflights are also blockers.', '',
        '| Problem | Solver | Preflight | Blocker / log |', '|---|---|---|---|']
    text.extend(f"| {row['problem']} | {row['solver']} | {row['preflight_status']} | "
                f"{row['remaining_gaps'].replace(' | ', '; ')}; {row['runtime_log']} |" for row in blocked)
    (root / 'blocked_dependencies.md').write_text('\n'.join(text) + '\n')
    return full


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    run(parser.parse_args().root)

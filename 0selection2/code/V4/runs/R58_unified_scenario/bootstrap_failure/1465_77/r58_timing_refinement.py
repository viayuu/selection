"""One precommitted TRAIN-only measurement using production ExecutionSession.

prepare and summarize are CPU-only. run is an explicitly invoked, bounded
measurement, never label generation or a whole-project budget approval.
"""

import argparse
import bisect
import json
import math
import pickle
import random
import time
from collections import Counter
from pathlib import Path

from ..unified_selector.registry import POOLS, PROBLEMS
from .r58_scenario import CONTRACT, INPUT_ROOT, ROOT, canonical_hash, fields, file_hash, save_json


RF_METHODS = ('RouteFinder', 'MoSES_RF', 'MoSES_CaDA')
RF_PROBLEMS = tuple(p for p in PROBLEMS if p not in ('TSP', 'ATSP'))
OTHER_METHODS = {
    'TSP': ('DIFUSCO', 'DIFUSCO500', 'T2T', 'T2T500', 'BQ', 'ELG', 'LEHD', 'OMNI'),
    'CVRP': ('BQ', 'ELG', 'ICAM', 'LEHD', 'MVMOE', 'OMNI', 'RELD_CVRP'),
}
CAP_SECONDS = 10800.
SHUTDOWN_SECONDS = 120.
PILOT_STATES = ('attempt1/run_state.json', 'diffusion_retry_state.json',
                'aug1_state.json', 'aug1_moses_rf_state.json')
OUTPUT = ROOT / 'timing_refinement'


def read_json(path):
    return json.loads(Path(path).read_text())


def positive(value, name, allow_zero=False):
    if (isinstance(value, bool) or not isinstance(value, (int, float)) or
            not math.isfinite(value) or value < 0 or (not allow_zero and value == 0)):
        raise ValueError(f'Invalid {name}')
    return value


def select_anchors(sizes, count):
    """Empirical nearest-rank quantiles, with distinct held-out warmups."""
    if count not in (3, 9) or not sizes or any(type(n) is not int or n <= 0 for n in sizes):
        raise ValueError('Require positive TRAIN sizes and three or nine anchors')
    ordered = sorted(sizes)
    denominator = count - 1
    scales = [ordered[(2 * (len(sizes) - 1) * k + denominator) // (2 * denominator)]
              for k in range(count)]
    if len(set(scales)) != count:
        raise ValueError('Quantiles collapse; do not replace or search for alternative anchors')
    rng, anchors = random.Random(2), []
    for n in scales:
        eligible = [i for i, scale in enumerate(sizes) if scale == n]
        if len(eligible) < 5:
            raise ValueError(f'TRAIN size {n} needs four timed cases and a disjoint warmup')
        chosen = rng.sample(eligible, 5)
        anchors.append(dict(id=f'n{n}', true_size=n, indices=chosen[:4], warmup_indices=chosen[4:]))
    return anchors


def targets():
    rows = [dict(problem=p, method=m, modes=['serial1', 'normal4', 'mps4'], anchor_count=3)
            for p in RF_PROBLEMS for m in RF_METHODS]
    rows.extend(dict(problem=p, method=m, modes=['serial1'], anchor_count=9)
                for p, methods in OTHER_METHODS.items() for m in methods)
    return rows


def groups_for(selection, target):
    anchors = selection['datasets'][target['problem']]['anchors']
    if target['anchor_count'] == 3 and len(anchors) == 9:
        anchors = [anchors[i] for i in (0, 4, 8)]
    if len(anchors) != target['anchor_count']:
        raise ValueError('Frozen anchor count differs')
    groups = [dict(id=a['id'], indices=a['indices']) for a in anchors]
    warmups = [dict(id='warm_' + a['id'], indices=a['warmup_indices']) for a in anchors]
    return anchors, groups, warmups


def prepare(root=ROOT, output=OUTPUT):
    from .r58_budget import qualification_reason
    from .r58_labels import implementation_hashes

    hashes = implementation_hashes()
    datasets = {}
    for problem in PROBLEMS:
        path = INPUT_ROOT / f'{problem}train' / 'dataset.pkl'
        with path.open('rb') as stream:
            items = pickle.load(stream)
        sizes = [int(fields(problem, item)['n']) for item in items]
        if len(sizes) != 10000:
            raise ValueError(f'{problem}: expected original 10000-case TRAIN')
        datasets[problem] = dict(path=str(path.resolve()), input_sha256=file_hash(path), count=len(sizes),
            histogram=dict(sorted(Counter(sizes).items())))
        if problem != 'ATSP':
            datasets[problem]['anchors'] = select_anchors(sizes, 9 if problem in OTHER_METHODS else 3)
        del items
    profiles = {}
    for problem in PROBLEMS:
        for method in POOLS[problem]:
            path = Path(root) / 'preflight' / f'{problem}__{method}.json'
            record = read_json(path)
            reason = qualification_reason(record, problem, method, hashes, datasets[problem])
            if reason:
                raise ValueError(f'{problem}/{method}: {reason}; parent must finish fresh128 preflight')
            profiles[f'{problem}__{method}'] = dict(path=str(path.resolve()), sha256=file_hash(path),
                deployment=record['deployment'])
    if len(profiles) != 128 or hashes != implementation_hashes():
        raise ValueError('Require a stable current128 implementation freeze')
    selection = dict(seed=2, passes=[1, 2], timed_cases_per_anchor=4, split='train',
        cpu_threads_per_worker=1, inference_batch_size=1, targets=targets(), datasets=datasets,
        implementation_sha256=hashes, contract_sha256=canonical_hash(CONTRACT), profiles=profiles,
        refiner_sha256=file_hash(__file__), execution_sha256=file_hash(Path(__file__).with_name('r58_execution.py')),
        aggregation='TRAIN histogram piecewise linear, mean of both passes; max-pass and nested5/9 references; no extrapolation',
        overhead='Three measured cold session overheads per projected deployment (one per split); '
                 'outer group wall includes checks and all group/measurement writes',
        mps_adoption='Entire RF family only: all48 complete and equivalent to both controls, '
                     'default MPS attachment verified, total projection including overhead better than normal4',
        cap_seconds=CAP_SECONDS, shutdown_reserve_seconds=SHUTDOWN_SECONDS,
        excluded='No validation/test data, labels, training, recipe overrides or full-generation approval')
    selection['selection_sha256'] = canonical_hash(selection)
    path = Path(output) / 'selection.json'
    if path.exists():
        if read_json(path) != selection:
            raise ValueError('Precommit differs; refusing to replace selected cases or source provenance')
    else:
        save_json(path, selection)
    return selection


def verify_selection(selection):
    from .r58_labels import implementation_hashes

    body = {k: v for k, v in selection.items() if k != 'selection_sha256'}
    if (canonical_hash(body) != selection['selection_sha256'] or
            selection['implementation_sha256'] != implementation_hashes() or
            selection['contract_sha256'] != canonical_hash(CONTRACT) or
            selection['refiner_sha256'] != file_hash(__file__) or
            selection['execution_sha256'] != file_hash(Path(__file__).with_name('r58_execution.py')) or
            selection['targets'] != targets()):
        raise ValueError('Selection or current source freeze differs')
    for entry in [*selection['datasets'].values(), *selection['profiles'].values()]:
        if file_hash(entry['path']) != entry.get('sha256', entry.get('input_sha256')):
            raise ValueError('Frozen original TRAIN or current preflight changed')


def spent_ledger(pilot_dir, new_preflight_state, additional_states=()):
    """Charge each completed actual wall once, never sum cumulative counters."""
    paths = [Path(pilot_dir) / name for name in PILOT_STATES]
    paths.extend([Path(new_preflight_state), *map(Path, additional_states)])
    if len({p.resolve() for p in paths}) != len(paths):
        raise ValueError('Duplicate spent-state receipt')
    entries = []
    for path in paths:
        state = read_json(path)
        if state.get('status') not in ('complete', 'preflight_complete') or not state.get('finished'):
            raise ValueError(f'{path}: require completed measured wall receipt')
        wall = state.get('wall_seconds')
        if wall is None:
            wall = state['finished'] - state['started']
        wall = positive(wall, 'spent wall_seconds')
        if state.get('started') is not None:
            wall = max(wall, positive(state['finished'] - state['started'], 'spent outer interval'))
        entries.append(dict(path=str(path.resolve()), sha256=file_hash(path), wall_seconds=wall))
    spent = sum(e['wall_seconds'] for e in entries)
    if spent + SHUTDOWN_SECONDS >= CAP_SECONDS:
        raise ValueError('Cumulative three-hour allowance exhausted, including new preflight and shutdown reserve')
    return dict(entries=entries, spent_seconds=spent, remaining_seconds=CAP_SECONDS - spent,
                measurement_seconds=CAP_SECONDS - spent - SHUTDOWN_SECONDS)


def timeout_for(deadline, receives=1, now=None):
    remaining = deadline - (time.monotonic() if now is None else now)
    if remaining <= 0:
        raise TimeoutError('Measurement allowance exhausted; retain budget block')
    # One single-group production call has a ready and a completion receive.
    return remaining / receives


def integrate(histogram, points):
    points = sorted(points)
    scales = [n for n, _ in points]
    if not points or len(set(scales)) != len(scales):
        raise ValueError('Missing or duplicate timing anchors')
    total = count = 0.
    for key, frequency in histogram.items():
        n = int(key)
        if type(frequency) is not int or frequency <= 0:
            raise ValueError('Invalid TRAIN histogram frequency')
        if not scales[0] <= n <= scales[-1]:
            raise ValueError('TRAIN size extrapolation refused')
        right = bisect.bisect_left(scales, n)
        if scales[right] == n:
            seconds = points[right][1]
        else:
            a, b = points[right - 1], points[right]
            seconds = a[1] + (b[1] - a[1]) * (n - a[0]) / (b[0] - a[0])
        total += frequency * positive(seconds, 'anchor seconds')
        count += frequency
    return total / count


def timing_estimate(histogram, anchors, passes, session_wall_seconds):
    if len(passes) != 2 or any(len(p) != len(anchors) for p in passes):
        raise ValueError('Require both complete fixed passes')
    for seconds in [*passes[0], *passes[1]]:
        positive(seconds, 'outer group seconds')
    group_wall = sum(map(sum, passes))
    if positive(session_wall_seconds, 'outer session wall') < group_wall:
        raise ValueError('Session wall excludes measured groups')
    overhead = session_wall_seconds - group_wall
    scales = [a['true_size'] for a in anchors]
    per_pass = [[s / 4 for s in p] for p in passes]
    means = [(a + b) / 2 for a, b in zip(*per_pass)]
    maxima = [max(a, b) for a, b in zip(*per_pass)]
    point = integrate(histogram, list(zip(scales, means)))
    reference = integrate(histogram, list(zip(scales, maxima)))
    result = dict(seconds_per_instance=point, max_pass_reference_seconds_per_instance=reference,
        pass_seconds_per_instance=[integrate(histogram, list(zip(scales, p))) for p in per_pass],
        measured_session_overhead_seconds=overhead, projected_session_overhead_seconds=3 * overhead,
        fresh_solve_seconds=12000 * point + 3 * overhead,
        reference_seconds=12000 * reference + 3 * overhead, extrapolation='none')
    if len(anchors) == 9:
        five = (0, 2, 4, 6, 8)
        nested = integrate(histogram, [(scales[i], means[i]) for i in five])
        nested_ref = integrate(histogram, [(scales[i], maxima[i]) for i in five])
        result.update(nested5_seconds_per_instance=nested,
            nested5_minus9_seconds_per_instance=nested - point,
            nested5_reference_seconds=12000 * nested_ref + 3 * overhead)
        result['reference_seconds'] = max(result['reference_seconds'], result['nested5_reference_seconds'])
    return result


def validate_group(raw, group, n, mode):
    workers = 1 if mode == 'serial1' else 4
    pids = raw['worker_pids']
    if len(pids) != workers or len(set(pids)) != workers or len(raw['groups']) != 1:
        raise ValueError('All intended workers must complete the actual group barrier')
    report = raw['groups'][0]
    if (report['id'] != group['id'] or report['indices'] != group['indices'] or
            report['solved'] != 4 or report['resumed'] != 0 or
            len(set(report['worker_paths'])) != workers):
        raise ValueError('Missing worker completion, resumed measurement or changed workload')
    rows = raw['rows']
    if len(rows) != 4 or {r['index'] for r in rows} != set(group['indices']):
        raise ValueError('Missing or duplicate measured case')
    if {r['worker'] for r in rows} != set(range(workers)):
        raise ValueError('Not all intended workers solved their assigned cases')
    for row in rows:
        if (row['true_size'] != n or row.get('feasible') is not True or row['seed'] != 2 or
                row['inference_batch'] != [row['index']]):
            raise ValueError('Independent check, size, seed or singleton inference differs')
        positive(row['cost'], 'paid FP64 cost', allow_zero=True)
    if mode == 'mps4':
        proof = raw.get('mps_proof') or {}
        if (proof.get('verified') is not True or proof.get('default_settings') is not True or
                set(proof.get('worker_pids', [])) != set(pids)):
            raise ValueError('Actual default-MPS4 attachment proof missing')


def equivalent(left, right):
    def physical(route):
        route = list(route)
        while len(route) > 1 and route[-1] == route[-2] == 0:
            route.pop()
        return route
    if not left.get('complete') or not right.get('complete'):
        return False
    for a, b in zip(left['passes'], right['passes']):
        aa = {r['index']: r for g in a for r in g['raw']['rows']}
        bb = {r['index']: r for g in b for r in g['raw']['rows']}
        if aa.keys() != bb.keys() or not aa:
            return False
        if any(aa[i].get('feasible') is not True or bb[i].get('feasible') is not True or
               aa[i]['cost'] != bb[i]['cost'] or physical(aa[i]['route']) != physical(bb[i]['route']) for i in aa):
            return False
    return len(left['passes']) == len(right['passes']) == 2


def choose_rf_family(records, expected=None):
    expected = expected or [(p, m) for p in RF_PROBLEMS for m in RF_METHODS]
    normal_total = mps_total = 0.
    mps_ok = True
    for problem, method in expected:
        modes = records.get(f'{problem}__{method}', {})
        serial, normal, mps = (modes.get(m, {}) for m in ('serial1', 'normal4', 'mps4'))
        if (not equivalent(serial, normal) or not serial.get('controls_not_attached') or
                not normal.get('controls_not_attached')):
            return dict(mode=None, complete=False, mps_credit=False, reason='normal4 qualification incomplete')
        normal_total += normal['estimate']['fresh_solve_seconds']
        if (not equivalent(serial, mps) or not equivalent(normal, mps) or not mps.get('attachment_verified')):
            mps_ok = False
        else:
            mps_total += mps['estimate']['fresh_solve_seconds']
    adopt = mps_ok and mps_total < normal_total
    return dict(mode='mps4' if adopt else 'normal4', complete=True, mps_credit=adopt,
        normal4_family_seconds=normal_total, mps4_family_seconds=mps_total if mps_ok else None,
        reason='all-family measured benefit' if adopt else 'zero MPS credit; measured normal4 only')


def measure_target(selection, target, mode, directory, deadline):
    from .r58_execution import ExecutionSession, PrivateMPS, make_execution_plan

    problem, method = target['problem'], target['method']
    anchors, groups, warmups = groups_for(selection, target)
    dataset = selection['datasets'][problem]
    profile = selection['profiles'][f'{problem}__{method}']['deployment']
    metadata = dict(selection_sha256=selection['selection_sha256'],
        implementation_sha256=selection['implementation_sha256'], execution_sha256=selection['execution_sha256'],
        refiner_sha256=selection['refiner_sha256'], contract_sha256=selection['contract_sha256'],
        train_path=dataset['path'], train_input_sha256=dataset['input_sha256'],
        preflight=selection['profiles'][f'{problem}__{method}'], cpu_threads_per_worker=1, seed=2)
    directory = Path(directory)
    record = dict(complete=False, mode=mode, passes=[[], []], controls_not_attached=False,
                  attachment_verified=False)
    start = time.perf_counter()
    try:
        if mode != 'mps4':
            # Production's read-only availability guard also proves no daemon
            # exists for serial/normal controls to attach to. It never starts MPS.
            import os
            if any(k.startswith('CUDA_MPS_') for k in os.environ):
                raise ValueError('Controls require no inherited MPS configuration')
            record['control_isolation'] = PrivateMPS().check_machine()
            record['controls_not_attached'] = True
        session = ExecutionSession(problem, method, make_execution_plan(mode), directory, metadata,
            input_path=dataset['path'], expected_profile=profile,
            timeout_seconds=timeout_for(deadline))
        with session:
            session.timeout = timeout_for(deadline)
            record['prepare'] = session.prepare(warmups)
            record['startup_and_warmup_wall_seconds'] = time.perf_counter() - start
            for pass_id in (1, 2):
                for anchor, group in zip(anchors, groups):
                    session.timeout = timeout_for(deadline, receives=2)
                    before = time.perf_counter()
                    raw = session.run_groups([group], pass_id=pass_id)
                    validate_group(raw, group, anchor['true_size'], mode)
                    if raw['worker_pids'] != record['prepare']['worker_pids']:
                        raise ValueError('Workers were reloaded or replaced during measurement')
                    save_json(directory / f'pass_{pass_id}_{group["id"]}_raw.json', raw)
                    record['passes'][pass_id - 1].append(dict(raw=raw, outer_seconds=time.perf_counter() - before))
            cleanup_start = time.perf_counter()
        record['cleanup_seconds'] = time.perf_counter() - cleanup_start
        record['worker_exitcodes'] = [p.exitcode for p in session.processes]
        if len(record['worker_exitcodes']) != (1 if mode == 'serial1' else 4) or any(record['worker_exitcodes']):
            raise ValueError('All intended workers must exit successfully')
        record['attachment_verified'] = mode == 'mps4'
        record['complete'] = True
    except Exception as error:
        record['error'] = repr(error)
    finally:
        record['session_wall_seconds'] = time.perf_counter() - start
        if record['complete']:
            record['estimate'] = timing_estimate(dataset['histogram'], anchors,
                [[g['outer_seconds'] for g in p] for p in record['passes']], record['session_wall_seconds'])
        save_json(directory / 'measurement.json', record)
    return record


def prerequisite_accounting(path, ledger):
    receipt = read_json(path)
    before = positive(receipt['before_additional_pilot_seconds'], 'earlier R58 spend', allow_zero=True)
    outer = positive(receipt['slurm_outer_overhead_seconds'], 'Slurm outer overhead', allow_zero=True)
    pilot = sum(entry['wall_seconds'] for entry in ledger['entries'][:4])
    if not math.isclose(pilot, receipt['completed_additional_pilot_controller_seconds'], abs_tol=1e-6):
        raise ValueError('Prerequisite receipt and four pilot receipts differ; deduplication unresolved')
    return dict(path=str(Path(path).resolve()), sha256=file_hash(path),
                add_to_refinement_ledger_seconds=before + outer,
                policy='Add earlier R58 development/preflights plus Slurm outer overhead only; '
                       'the four pilot controller walls are already in the refinement ledger')


def summarize(selection, state):
    family = choose_rf_family(state['records'])
    estimates, missing = [], []
    for target in selection['targets']:
        key = f'{target["problem"]}__{target["method"]}'
        mode = family['mode'] if target['method'] in RF_METHODS else 'serial1'
        measured = state['records'].get(key, {}).get(mode, {})
        if not measured.get('complete'):
            missing.append(key)
        else:
            estimates.append(dict(deployment=key, mode=mode, **measured['estimate']))
    extra = state.get('prerequisite_spent', {}).get('add_to_refinement_ledger_seconds')
    return dict(status=state['status'], selection_sha256=selection['selection_sha256'], rf_family=family,
        expected_refined_deployments=63, estimated_refined_deployments=len(estimates),
        missing_refined_deployments=missing, unrefined_deployments_requiring_current_budget=65,
        refined_subtotal_gpu_hours=sum(e['fresh_solve_seconds'] for e in estimates) / 3600,
        refined_reference_gpu_hours=sum(e['reference_seconds'] for e in estimates) / 3600,
        cumulative_refinement_allowance_seconds=state['cumulative_wall_seconds'],
        whole_r58_spent_seconds=state['cumulative_wall_seconds'] + extra if extra is not None else None,
        projection_counts=dict(train=10000, val=1000, test=1000),
        validation_test_sizes='Estimated from original TRAIN; validation/test data never opened',
        baseline_context='R45 A: optimization 4552.904s; post-W&B-init wall 4976.866s includes '
                         'validation/train-eval, not preparation/cache/test. Not measured R58 time '
                         'and not automatically added as an allowance.',
        limitations='Subset measurement, no hard guarantee. Whole128 remaining labels, additional IO, '
                    'baseline/evaluation and explicit contingency still require a complete <100h plan.',
        full_generation_authorized=False, estimates=estimates)


def run(root, output, new_preflight_state, additional_states=()):
    output = Path(output)
    selection = read_json(output / 'selection.json')
    verify_selection(selection)
    if (output / 'run_state.json').exists():
        raise ValueError('One refinement attempt only; previous failures and timings must be preserved')
    ledger = spent_ledger(Path(root) / 'throughput_pilot', new_preflight_state, additional_states)
    prerequisite = prerequisite_accounting(Path(root) / 'prerequisite_spent.json', ledger)
    started, clock = time.time(), time.monotonic()
    deadline = clock + ledger['measurement_seconds']
    state = dict(status='running', started=started, ledger=ledger, prerequisite_spent=prerequisite,
                 records={}, full_generation_authorized=False)
    save_json(output / 'run_state.json', state)
    try:
        mps_available = True
        for target in selection['targets']:
            key = f'{target["problem"]}__{target["method"]}'
            state['records'][key] = {}
            for mode in target['modes']:
                timeout_for(deadline)
                if mode == 'mps4' and not mps_available:
                    state['records'][key][mode] = dict(complete=False, error='MPS unavailable; no retries or credit')
                    continue
                record = measure_target(selection, target, mode, output / f'{key}__{mode}', deadline)
                state['records'][key][mode] = record
                save_json(output / 'run_state.json', state)
                if not record['complete']:
                    if mode == 'mps4':
                        mps_available = False
                    else:
                        raise RuntimeError(f'{key}/{mode} incomplete; retain budget block')
        verify_selection(selection)
        state['status'] = 'complete'
    except Exception as error:
        state.update(status='incomplete', error=repr(error))
    finally:
        state.update(finished=time.time(), wall_seconds=time.monotonic() - clock)
        state['cumulative_wall_seconds'] = ledger['spent_seconds'] + state['wall_seconds']
        if state['cumulative_wall_seconds'] > CAP_SECONDS:
            state.update(status='incomplete', error='Cumulative wall cap exceeded; no budget clearance')
        state['rf_family'] = choose_rf_family(state['records'])
        save_json(output / 'run_state.json', state)
        save_json(output / 'summary.json', summarize(selection, state))
    return 0 if state['status'] == 'complete' and state['rf_family']['complete'] else 3


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'run', 'summarize'])
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--new-preflight-state', type=Path)
    parser.add_argument('--additional-spent-state', type=Path, action='append', default=[])
    args = parser.parse_args(argv)
    if args.stage == 'prepare':
        selection = prepare(args.root, args.output)
        print(json.dumps(dict(selection_sha256=selection['selection_sha256'], targets=len(selection['targets']))))
    elif args.stage == 'run':
        if args.new_preflight_state is None:
            parser.error('run requires --new-preflight-state: new preflight consumes the same three-hour cap')
        return run(args.root, args.output, args.new_preflight_state, args.additional_spent_state)
    else:
        selection = read_json(args.output / 'selection.json')
        verify_selection(selection)
        state = read_json(args.output / 'run_state.json')
        summary = summarize(selection, state)
        save_json(args.output / 'summary.json', summary)
        print(json.dumps({k: v for k, v in summary.items() if k != 'estimates'}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

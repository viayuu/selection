"""CPU-only fresh-solve budget from current preflights and original TRAIN sizes."""

import argparse
import bisect
import copy
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from ..unified_selector.registry import POOLS, PROBLEMS
from .r58_labels import implementation_hashes, inference_batch_size, load_instances, size
from .r58_scenario import CONTRACT, ROOT, canonical_hash, file_hash, save_json


SPLIT_COUNTS = {'train': 10000, 'val': 1000, 'test': 1000}
LIMIT_HOURS = 100
BASELINE_EVALUATION_SECONDS = 3 * 3600
UNACCOUNTED_OVERHEAD_SECONDS = 2 * 3600
LABEL_CONTINGENCY_FRACTION = .10
WHOLE_CLEARED = 'whole_plan_reference_under_100h'
WHOLE_OVER = 'skip_whole_plan_at_or_over_100h'
ACCOUNTING_RULE = 'G = S_actual + L_reference + H63 + 3h + 2h + 0.10 * L_reference < 100h'
TRAINING_CONTEXT = (
    'Observed existing R45A history: 40 epochs at about 114 seconds of optimization per epoch '
    '(about 1.3 GPUh, excluding evaluation). Training under 3h on a 3090 is an expectation, '
    'not a measured R58 training budget. This context is NOT added to the solve projection. '
    'Even a label-solve estimate <=100h does not certify total R58 work <=100h.'
)
LIMITATIONS = [
    'Budget measurement, not an experiment or performance claim; no solves or training are launched.',
    'Only original TRAIN dataset.pkl files are opened. No validation/test inputs or cost files are read.',
    'Validation/test counts are fixed at 1000 each; their size distributions are ESTIMATED FROM TRAIN.',
    'Only passing TRAIN profiles matching current implementation, contract and TRAIN input hashes contribute.',
    'Runtime environments are recorded as measurement provenance, not compared to this CPU reporting process. '
    'Deployment lock/release qualification is a separate gate.',
    'Singletons: linear interpolation of results.seconds by true size (maximum of repeats at each size). '
    'The conservative reference uses the larger adjacent endpoint. No extrapolation outside observed sizes.',
    'RF-family results.seconds are warm timings; first_call_seconds NEVER contributes. '
    'Other methods may retain first-call overhead in results.seconds.',
    'Width16: only a measured full batch at maximum TRAIN size is used. Charge that whole batch for every '
    'size-homogeneous batch, including tails; val/test use TRAIN proportions with batch counts rounded up. '
    'This is a worst-scale, upper-ish POINT estimate, not a measured mean or a proven upper bound. '
    'No speedup is inferred from singleton timings.',
    'Excludes training, deferred initialization/checkpoint loading, dataset/label IO, and outer CPU route validation. '
    'Recorded backend timings can already include internal CPU checks; these are not subtracted.',
    'Serial solver elapsed time is expressed as single-GPU occupancy hours, not measured kernel-active time. '
    'No scaling to other GPUs or parallel speedup is assumed; no new device measurement is made.',
    'Sparse timings have no confidence interval or hard budget guarantee. Conservative references are not bounds. '
    'JSON retains arithmetic precision for audit, not timing accuracy.',
    'Preflight may still be running: this is a timestamped snapshot. Rerun after final preflight.',
]


def train_distribution(problem):
    items, path = load_instances(problem, 'train')
    sizes = [size(problem, item) for item in items]
    if not sizes or min(sizes) <= 0:
        raise ValueError(f'{problem}: empty or invalid TRAIN sizes')
    return dict(path=str(path.resolve()), input_sha256=file_hash(path), count=len(sizes),
                min=min(sizes), max=max(sizes), histogram=dict(sorted(Counter(sizes).items())),
                sizes=sizes)


def qualification_reason(record, problem, method, hashes, train):
    if record.get('implementation_sha256') != hashes:
        return 'stale_implementation'
    if record.get('contract_sha256') != canonical_hash(CONTRACT):
        return 'stale_contract'
    if (record.get('problem'), record.get('solver'), record.get('split')) != (problem, method, 'train'):
        return 'wrong_problem_solver_or_split'
    if record.get('input_sha256') != train['input_sha256']:
        return 'stale_train_input'
    if record.get('passed') is not True:
        return 'not_passed'
    if not isinstance(record.get('deployment'), dict) or not record['deployment']:
        return 'missing_deployment_profile'
    return None


def timing_points(record, train):
    points, observations = {}, []
    for result in record.get('results', []):
        index, n = result.get('index'), result.get('true_size')
        if (type(index) is not int or not 0 <= index < train['count'] or
                type(n) is not int or n != train['sizes'][index]):
            raise ValueError('timing index/true_size does not match original TRAIN')
        seconds = result.get('seconds')
        if not isinstance(seconds, (int, float)) or isinstance(seconds, bool) or not math.isfinite(seconds) or seconds <= 0:
            raise ValueError('missing or invalid results.seconds; cold timing is not a fallback')
        points[n] = max(points.get(n, 0.), seconds)
        observations.append(dict(index=index, true_size=n, seconds=seconds))
    if not points:
        raise ValueError('no per-size timing observations')
    return sorted(points.items()), observations


def estimate_profile(record, train, width):
    points, observations = timing_points(record, train)
    scales = [n for n, _ in points]
    histogram = train['histogram']
    if train['min'] < scales[0] or train['max'] > scales[-1]:
        raise ValueError('TRAIN sizes outside observed timing support; extrapolation refused')
    estimate = dict(observations=observations, observed_size_range=[scales[0], scales[-1]],
                    extrapolation='none', inference_batch_size=width, splits={})
    if width == 1:
        total = reference = 0.
        for n, count in histogram.items():
            right = bisect.bisect_left(scales, n)
            if scales[right] == n:
                seconds = conservative = points[right][1]
            else:
                a, b = points[right - 1], points[right]
                seconds = a[1] + (b[1] - a[1]) * (n - a[0]) / (b[0] - a[0])
                conservative = max(a[1], b[1])
            total += count * seconds
            reference += count * conservative
        estimate.update(model='true_size_linear_interpolation',
                        seconds_per_instance=total / train['count'])
        for split, count in SPLIT_COUNTS.items():
            estimate['splits'][split] = dict(
                seconds=total * count / train['count'],
                conservative_reference_seconds=reference * count / train['count'])
    else:
        batch = record.get('batch_probe', {})
        indices = batch.get('indices', [])
        if (len(indices) != width or len(set(indices)) != width or
                any(type(i) is not int or not 0 <= i < train['count'] or
                    train['sizes'][i] != train['max'] for i in indices)):
            raise ValueError('requires a full deployment-width batch at maximum TRAIN true size')
        seconds = batch.get('seconds')
        per_instance = batch.get('seconds_per_instance')
        if (not isinstance(seconds, (int, float)) or isinstance(seconds, bool) or
                not math.isfinite(seconds) or seconds <= 0 or
                not isinstance(per_instance, (int, float)) or isinstance(per_instance, bool) or
                not math.isfinite(per_instance) or
                not math.isclose(per_instance, seconds / width, rel_tol=1e-6, abs_tol=1e-12)):
            raise ValueError('invalid or inconsistent batch_probe timing')
        estimate.update(model='worst_scale_batch_upper_ish_point',
                        seconds_per_instance=seconds / width,
                        batch_measurement=dict(indices=indices, true_size=train['max'], seconds=seconds,
                                               seconds_per_instance=seconds / width))
        for split, count in SPLIT_COUNTS.items():
            # Ceil each projected size group separately, never pool tails across sizes.
            denominator = train['count'] * width
            batches = sum((frequency * count + denominator - 1) // denominator
                          for frequency in histogram.values())
            estimate['splits'][split] = dict(seconds=batches * seconds,
                conservative_reference_seconds=batches * seconds, projected_batches=batches,
                tail_policy='charge each partial batch as a full measured maximum-size batch')
    for value in estimate['splits'].values():
        value['gpu_hours'] = value['seconds'] / 3600
        value['conservative_reference_gpu_hours'] = value['conservative_reference_seconds'] / 3600
    estimate['fresh_solve_gpu_hours'] = sum(value['gpu_hours'] for value in estimate['splits'].values())
    estimate['conservative_reference_gpu_hours'] = sum(
        value['conservative_reference_gpu_hours'] for value in estimate['splits'].values())
    return estimate


def build_budget(root=ROOT):
    started = datetime.now(timezone.utc).isoformat()
    hashes = implementation_hashes()
    trains = {problem: train_distribution(problem) for problem in PROBLEMS}
    deployments = []
    for problem in PROBLEMS:
        for method in POOLS[problem]:
            path = Path(root) / 'preflight' / f'{problem}__{method}.json'
            row = dict(problem=problem, solver=method, profile_path=str(path.resolve()),
                       current_passing=False, estimate=None)
            try:
                record = json.loads(path.read_text())
            except FileNotFoundError:
                row['status'] = 'missing_preflight'
            except json.JSONDecodeError:
                row['status'] = 'invalid_preflight_json'
            else:
                reason = qualification_reason(record, problem, method, hashes, trains[problem])
                row['status'] = reason or 'current_passing'
                if reason is None:
                    row['current_passing'] = True
                    row['runtime_environment'] = record.get('runtime_environment')
                    try:
                        row['estimate'] = estimate_profile(record, trains[problem], inference_batch_size(problem, method))
                    except ValueError as error:
                        row.update(status='missing_timing_coverage', timing_error=str(error))
            deployments.append(row)
    if implementation_hashes() != hashes:
        raise RuntimeError('Deployment implementation changed while estimating; rerun on the current freeze')
    measured = [row for row in deployments if row['estimate'] is not None]
    missing = [{key: row[key] for key in ('problem', 'solver', 'status')}
               for row in deployments if not row['current_passing']]
    timing_gaps = [{key: row[key] for key in ('problem', 'solver', 'timing_error')}
                   for row in deployments if row['current_passing'] and row['estimate'] is None]
    total = sum(row['estimate']['fresh_solve_gpu_hours'] for row in measured)
    reference = sum(row['estimate']['conservative_reference_gpu_hours'] for row in measured)
    complete = len(measured) == len(deployments)
    over = max(total, reference) > LIMIT_HOURS
    budget = dict(threshold_gpu_hours=LIMIT_HOURS,
        current_profile_subtotal_gpu_hours=total,
        current_profile_conservative_reference_gpu_hours=reference,
        current_profile_point_status='>100h' if total > LIMIT_HOURS else '<=100h',
        conservative_reference_status='>100h' if reference > LIMIT_HOURS else '<=100h',
        all_deployments_gpu_hours=total if complete else None,
        all_deployments_status=('>100h' if over else '<=100h') if complete else
                               ('>100h_projected_current_subset' if over else 'unknown_incomplete_coverage'),
        decision='skip_full_fresh_solve_projected_over_100h' if over else
                 ('point_estimate_within_100h_not_guaranteed' if complete else 'not_cleared_incomplete_coverage'))
    return dict(scenario='R58 scenario_v2 fresh solve',
        snapshot_started_utc=started, snapshot_finished_utc=datetime.now(timezone.utc).isoformat(),
        root=str(Path(root).resolve()), implementation_sha256=hashes, contract_sha256=canonical_hash(CONTRACT),
        projection_counts=SPLIT_COUNTS, projected_solver_instance_evaluations=len(deployments) * sum(SPLIT_COUNTS.values()),
        size_distribution_policy='original TRAIN only; validation/test distributions estimated from TRAIN',
        train_distributions={p: {k: v for k, v in train.items() if k != 'sizes'} for p, train in trains.items()},
        warnings=[f'{p}: observed TRAIN count {train["count"]} differs from fixed projection count 10000'
                  for p, train in trains.items() if train['count'] != SPLIT_COUNTS['train']],
        coverage=dict(expected_deployments=len(deployments),
            current_passing_deployments=sum(row['current_passing'] for row in deployments),
            estimated_deployments=len(measured), all_current_passing=not missing,
            complete_budget=complete, missing_deployments=missing, missing_timing_coverage=timing_gaps),
        budget=budget, limitations=LIMITATIONS, training_context=TRAINING_CONTEXT, deployments=deployments)


def whole_components(spent_seconds, label_reference_seconds, session_overhead_seconds):
    from .r58_timing_refinement import positive

    for name, value in [('spent', spent_seconds), ('label reference', label_reference_seconds),
                        ('measured63 overhead', session_overhead_seconds)]:
        positive(value, name, allow_zero=True)
    components = dict(actual_spent=spent_seconds, remaining_label_reference=label_reference_seconds,
        measured63_session_overhead=session_overhead_seconds,
        baseline_preparation_evaluation=BASELINE_EVALUATION_SECONDS,
        otherwise_unaccounted_overhead=UNACCOUNTED_OVERHEAD_SECONDS,
        label_contingency=LABEL_CONTINGENCY_FRACTION * label_reference_seconds)
    total = sum(components.values())
    return dict(components_seconds=components, whole_reference_seconds=total,
        whole_projected_gpu_hours=total / 3600,
        label_reference_limit_gpu_hours=(95 - (spent_seconds + session_overhead_seconds) / 3600) / 1.10,
        decision=WHOLE_CLEARED if total < LIMIT_HOURS * 3600 and
                 not math.isclose(total, LIMIT_HOURS * 3600, rel_tol=0., abs_tol=1e-6) else WHOLE_OVER)


def validated_measurement(selection, target, mode, measured, histogram):
    from .r58_execution import execution_profile, make_execution_plan
    from .r58_timing_refinement import groups_for, timing_estimate, validate_group

    anchors, groups, _ = groups_for(selection, target)
    workers = 1 if mode == 'serial1' else 4
    if (measured.get('complete') is not True or measured.get('mode') != mode or
            measured.get('worker_exitcodes') != [0] * workers or
            len(measured.get('passes', [])) != 2):
        raise ValueError('Incomplete refinement workers or passes')
    expected_profile = selection['profiles'][f'{target["problem"]}__{target["method"]}']['deployment']
    expected_execution = execution_profile(target['problem'], target['method'], make_execution_plan(mode))
    prepared = measured['prepare']
    if (prepared['deployment_profile'] != expected_profile or
            prepared['execution_profile'] != expected_execution):
        raise ValueError('Refinement startup differs from the current production deployment')
    seconds = []
    for timed_pass in measured['passes']:
        if len(timed_pass) != len(groups):
            raise ValueError('Incomplete fixed refinement anchors')
        values = []
        for anchor, group, observation in zip(anchors, groups, timed_pass):
            raw = observation['raw']
            validate_group(raw, group, anchor['true_size'], mode)
            if (raw['worker_pids'] != prepared['worker_pids'] or
                    raw['deployment_profile'] != expected_profile or raw['execution_profile'] != expected_execution or
                    raw['groups'][0]['seconds'] > observation['outer_seconds']):
                raise ValueError('Workers/profile changed or timing omits completed group work')
            values.append(observation['outer_seconds'])
        seconds.append(values)
    if mode != 'mps4':
        isolation = measured.get('control_isolation', {})
        if (measured.get('controls_not_attached') is not True or
                isolation.get('existing_mps_processes') != [] or
                isolation.get('existing_target_context_pids') != []):
            raise ValueError('Non-MPS control isolation proof missing')
    elif measured.get('attachment_verified') is not True:
        raise ValueError('MPS attachment qualification missing')
    estimate = timing_estimate(histogram, anchors, seconds, measured['session_wall_seconds'])
    if estimate != measured['estimate']:
        raise ValueError('Saved refinement estimate differs from its two actual timed passes')
    return estimate


def load_refinement(root, report):
    from . import r58_timing_refinement as refinement

    directory = Path(root) / 'timing_refinement'
    paths = {name: directory / f'{name}.json' for name in ('selection', 'run_state', 'summary')}
    selection, state, summary = (json.loads(paths[name].read_text()) for name in paths)
    refinement.verify_selection(selection)
    if (selection['implementation_sha256'] != report['implementation_sha256'] or
            selection['targets'] != refinement.targets() or len(selection['targets']) != 63 or
            state.get('status') != 'complete' or not state.get('finished')):
        raise ValueError('Require complete63 refinement under current fresh128 source freeze')
    expected_profiles = {f'{r["problem"]}__{r["solver"]}' for r in report['deployments']}
    if set(selection['profiles']) != expected_profiles:
        raise ValueError('Refinement preflight roster differs from current128')
    for problem, train in report['train_distributions'].items():
        sampled = selection['datasets'][problem]
        if (train['count'] != 10000 or train['input_sha256'] != sampled['input_sha256'] or
                {int(n): c for n, c in train['histogram'].items()} !=
                {int(n): c for n, c in sampled['histogram'].items()}):
            raise ValueError('Refinement histogram differs from original TRAIN')
    records = copy.deepcopy(state['records'])
    if set(records) != {f'{t["problem"]}__{t["method"]}' for t in selection['targets']}:
        raise ValueError('Incomplete or changed63 refinement roster')
    for target in selection['targets']:
        modes = records[f'{target["problem"]}__{target["method"]}']
        if set(modes) != set(target['modes']):
            raise ValueError('Missing precommitted execution condition')
        for mode in target['modes']:
            measured = modes[mode]
            if mode == 'mps4' and measured.get('complete') is not True:
                continue
            measured['estimate'] = validated_measurement(selection, target, mode, measured,
                report['train_distributions'][target['problem']]['histogram'])
    family = refinement.choose_rf_family(records)
    if family.get('complete') is not True or family.get('mode') not in ('normal4', 'mps4'):
        raise ValueError('All48 RF-family qualification incomplete')
    if summary != refinement.summarize(selection, state) or summary['rf_family'] != family:
        raise ValueError('Final summary differs from completed raw refinement evidence')
    ledger = state['ledger']
    if len(ledger['entries']) < 5 or len({e['path'] for e in ledger['entries']}) != len(ledger['entries']):
        raise ValueError('Require four pilot receipts plus new preflight without duplicate charges')
    charged = sum(refinement.positive(e['wall_seconds'], 'spent receipt') for e in ledger['entries'])
    wall = refinement.positive(state['wall_seconds'], 'refinement outer wall')
    session_walls = sum(refinement.positive(m.get('session_wall_seconds', 0.), 'session wall', allow_zero=True)
                        for modes in records.values() for m in modes.values())
    cumulative = refinement.positive(state['cumulative_wall_seconds'], 'cumulative refinement allowance')
    if (not math.isclose(charged, ledger['spent_seconds'], abs_tol=1e-6) or
            not math.isclose(cumulative, charged + wall, abs_tol=1e-6) or wall < session_walls or
            cumulative > refinement.CAP_SECONDS):
        raise ValueError('Actual cumulative three-hour accounting incomplete or exceeded')
    prerequisite_path = Path(root) / 'prerequisite_spent.json'
    prerequisite = refinement.prerequisite_accounting(prerequisite_path, ledger)
    if prerequisite != state.get('prerequisite_spent'):
        raise ValueError('Earlier R58 spend receipt changed or is missing')
    spent = cumulative + prerequisite['add_to_refinement_ledger_seconds']
    if not math.isclose(spent, summary['whole_r58_spent_seconds'], abs_tol=1e-6):
        raise ValueError('Whole-R58 spend is not deduplicated consistently')
    estimates = []
    for target in selection['targets']:
        mode = family['mode'] if target['method'] in refinement.RF_METHODS else 'serial1'
        measured = records[f'{target["problem"]}__{target["method"]}'][mode]
        estimates.append(dict(problem=target['problem'], solver=target['method'], mode=mode,
                              **measured['estimate']))
    paths['prerequisite_spent'] = prerequisite_path
    return dict(estimates=estimates, rf_family=family, spent_seconds=spent,
        evidence={str(path.resolve()): file_hash(path) for path in paths.values()},
        preflights={profile['path']: profile['sha256'] for profile in selection['profiles'].values()})


def build_whole_budget(root=ROOT):
    """Read-only mechanical gate; never publish an execution approval here."""
    report = build_budget(root)
    whole = dict(complete=False, decision='not_cleared_incomplete_whole_plan',
                 accounting_rule=ACCOUNTING_RULE, blockers=[], execution_plan=None)
    report['whole_project'] = whole
    coverage = report['coverage']
    if (not coverage['complete_budget'] or not coverage['all_current_passing'] or
            coverage['current_passing_deployments'] != 128 or coverage['estimated_deployments'] != 128 or
            coverage['expected_deployments'] != 128):
        whole['blockers'].append('Require fresh current128 passing profiles with supported conservative timings')
        return report
    try:
        evidence = load_refinement(root, report)
        refined = {(r['problem'], r['solver']): r for r in evidence['estimates']}
        if len(refined) != 63 or len(evidence['estimates']) != 63:
            raise ValueError('Require complete63 refined estimates without duplicates')
        remaining = [r for r in report['deployments'] if (r['problem'], r['solver']) not in refined]
        if len(remaining) != 65 or len(refined) + len(remaining) != len(report['deployments']):
            raise ValueError('Refined63 and unchanged65 do not partition current128')
        startup = sum(r['projected_session_overhead_seconds'] for r in refined.values())
        refined_reference = sum(r['reference_seconds'] - r['projected_session_overhead_seconds']
                                for r in refined.values())
        other_reference = sum(r['estimate']['conservative_reference_gpu_hours'] * 3600 for r in remaining)
        components = whole_components(evidence['spent_seconds'], refined_reference + other_reference, startup)
        from .r58_execution import make_execution_plan
        plan = make_execution_plan(evidence['rf_family']['mode'])
        if implementation_hashes() != report['implementation_sha256']:
            raise ValueError('Current implementation changed while integrating final budget')
        whole.update(complete=True, **components, rf_family=evidence['rf_family'], execution_plan=plan,
            refined_deployments=63, preflight_only_deployments=65,
            refined_label_reference_seconds=refined_reference, preflight65_reference_seconds=other_reference,
            refined_estimates=evidence['estimates'], evidence=evidence['evidence'], preflights=evidence['preflights'])
    except (OSError, ValueError, KeyError, TypeError, IndexError, ZeroDivisionError) as error:
        whole['blockers'].append(str(error))
    return report


def write_execution_approval(root, report):
    """Called only by the whole-plan pipeline gate, immediately before locking."""
    from .r58_execution import validate_execution_plan

    whole = report.get('whole_project', {})
    if (whole.get('complete') is not True or whole.get('decision') != WHOLE_CLEARED or
            not 0 < whole.get('whole_projected_gpu_hours', float('inf')) < LIMIT_HOURS or
            report['implementation_sha256'] != implementation_hashes()):
        raise ValueError('Complete current whole-plan reference <100h required before approval')
    components = whole['components_seconds']
    recalculated = whole_components(components['actual_spent'], components['remaining_label_reference'],
                                    components['measured63_session_overhead'])
    if any(whole[key] != value for key, value in recalculated.items()):
        raise ValueError('Fixed008 accounting or allowance was altered')
    plan = validate_execution_plan(whole['execution_plan'])
    if plan['rf_mode'] != whole['rf_family']['mode'] or whole['rf_family'].get('complete') is not True:
        raise ValueError('Execution plan differs from globally qualified RF family')
    for path, digest in {**whole['evidence'], **whole['preflights']}.items():
        if file_hash(path) != digest:
            raise ValueError('Budget evidence changed before deployment lock')
    approval = dict(approved=True, execution_plan_sha256=canonical_hash(plan),
        implementation_sha256=report['implementation_sha256'], deployments=128,
        whole_projected_gpu_hours=whole['whole_projected_gpu_hours'], accounting_rule=ACCOUNTING_RULE,
        components_seconds=components, evidence=whole['evidence'], preflight_sha256=whole['preflights'])
    root = Path(root)
    lock_path = root / 'deployments.lock.json'
    if lock_path.exists():
        locked = json.loads(lock_path.read_text())
        if locked.get('execution_plan') != plan or locked.get('execution_approval') != approval:
            raise ValueError('Existing immutable lock differs from the cleared plan/approval')
    save_json(root / 'execution_plan.json', plan)
    save_json(root / 'execution_approval.json', approval)
    return approval


def markdown(report):
    coverage, budget = report['coverage'], report['budget']
    text = ['# R58 Fresh-Solve Budget', '',
        '**Budget measurement only. No performance claim or launch authorization.**', '',
        f'Snapshot: {report["snapshot_finished_utc"]}.', '',
        f'- Current passing deployments: {coverage["current_passing_deployments"]}/{coverage["expected_deployments"]}.',
        f'- Timing coverage: {coverage["estimated_deployments"]}/{coverage["expected_deployments"]}; '
        f'all-deployment estimate complete: {coverage["complete_budget"]}.',
        f'- Current-profile fresh-solve subtotal: ~{budget["current_profile_subtotal_gpu_hours"]:.3g} GPUh '
        f'({budget["current_profile_point_status"]}).',
        f'- Conservative reference for the same subset: ~{budget["current_profile_conservative_reference_gpu_hours"]:.3g} GPUh '
        f'({budget["conservative_reference_status"]}); not a proven bound.',
        f'- Full projection status: `{budget["all_deployments_status"]}`. Decision: `{budget["decision"]}`.',
        f'- Fixed counts per deployment: 10000 TRAIN + 1000 validation + 1000 test; '
        f'{report["projected_solver_instance_evaluations"]:,} evaluations over the full roster.', '',
        '## Assumptions And Limits', '']
    text.extend(f'- {note}' for note in report['limitations'])
    text.extend(f'- WARNING: {note}' for note in report['warnings'])
    text += ['', '## Training Context (Excluded)', '', report['training_context']]
    text += ['', '## Original TRAIN Size Distributions', '',
             '| Problem | Count | True size range | Distinct sizes |', '|---|---:|---:|---:|']
    for problem, train in report['train_distributions'].items():
        text.append(f'| {problem} | {train["count"]} | {train["min"]}..{train["max"]} | {len(train["histogram"])} |')
    text += ['', '## Current Profile Estimates', '',
        'Hours cover all three fixed-count splits; validation/test sizes are estimated from TRAIN. '
        'Stale/unpassed/missing profiles have NO current estimate. Batch rows use only one worst-scale measurement.', '',
        '| Problem | Method | Status / model | Seconds/instance | Fresh GPUh | Conservative reference GPUh |',
        '|---|---|---|---:|---:|---:|']
    for row in report['deployments']:
        estimate = row['estimate']
        if estimate is None:
            text.append(f'| {row["problem"]} | {row["solver"]} | {row["status"]} | - | - | - |')
        else:
            precision = '.2g' if estimate['inference_batch_size'] > 1 else '.3g'
            text.append(f'| {row["problem"]} | {row["solver"]} | {estimate["model"]} | '
                f'~{estimate["seconds_per_instance"]:.2g} | ~{estimate["fresh_solve_gpu_hours"]:{precision}} | '
                f'~{estimate["conservative_reference_gpu_hours"]:{precision}} |')
    text += ['', 'For batch rows seconds/instance is the maximum-size full-batch observation, before tail charges.', '',
             '## Missing Coverage', '']
    for row in coverage['missing_deployments']:
        text.append(f'- {row["problem"]}/{row["solver"]}: {row["status"]}.')
    for row in coverage['missing_timing_coverage']:
        text.append(f'- {row["problem"]}/{row["solver"]}: {row["timing_error"]}.')
    if coverage['complete_budget']:
        text.append('None: every deployment has a current passing profile and a supported timing estimate.')
    text += ['', 'Audit details: sibling `budget.json` contains TRAIN histograms and paths/hashes, accepted timing '
             'observations, batch indices, per-split arithmetic, and excluded-profile reasons.', '',
             'Regenerate (CPU only):', '', '```bash',
             'source /public/home/shiys/miniconda3/etc/profile.d/conda.sh', 'conda activate easynco',
             'CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \\',
             '  python -m code.V4.r58_budget --root ' + report['root'], '```', '']
    whole = report.get('whole_project')
    if whole is not None:
        section = ['# R58 Whole-Project Budget', '', f'`{whole["accounting_rule"]}`', '',
            f'Whole-plan evidence complete: {whole["complete"]}. Decision: `{whole["decision"]}`.', '']
        if whole['complete']:
            section += [f'Whole reference: ~{whole["whole_projected_gpu_hours"]:.4g} GPUh. '
                        f'RF family: `{whole["rf_family"]["mode"]}`.', '',
                        '| Component | GPUh |', '|---|---:|']
            section.extend(f'| {name} | {seconds / 3600:.4g} |'
                           for name, seconds in whole['components_seconds'].items())
            section += ['', 'H63 is charged ONCE: removed from the refined reference before adding the '
                        'three-times measured session overhead. The65 unchanged estimates retain current '
                        'preflight conservative references. No unmeasured parallel speedup is assumed.', '']
        section.extend(f'- BLOCKED: {reason}' for reason in whole['blockers'])
        section += ['', 'The3h baseline and2h otherwise-unaccounted overhead are fixed allowances, not '
                    'measured R58 completion times. The10% label contingency and timing references are '
                    'not confidence bounds or guarantees. Validation/test sizes are estimated from TRAIN.', '',
                    'The preflight-only diagnostics below are NOT a whole-project approval.', '']
        text = section + text
    return '\n'.join(text)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT, help='Read only preflight/<problem>__<method>.json here')
    parser.add_argument('--output-dir', type=Path, help='Write budget.json and BUDGET.md here (default: --root)')
    parser.add_argument('--whole', action='store_true', help='Read completed refinement and apply fixed008 whole-project gate')
    args = parser.parse_args(argv)
    report = build_whole_budget(args.root) if args.whole else build_budget(args.root)
    output = args.output_dir or args.root
    save_json(output / 'budget.json', report)
    (output / 'BUDGET.md').write_text(markdown(report))
    print(json.dumps(dict(coverage={k: v for k, v in report['coverage'].items() if not isinstance(v, list)},
                          budget=report['budget'], whole_project=report.get('whole_project'),
                          output_dir=str(output.resolve()))))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

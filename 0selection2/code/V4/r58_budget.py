"""CPU-only fresh-solve budget from current preflights and original TRAIN sizes."""

import argparse
import bisect
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
    return '\n'.join(text)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT, help='Read only preflight/<problem>__<method>.json here')
    parser.add_argument('--output-dir', type=Path, help='Write budget.json and BUDGET.md here (default: --root)')
    args = parser.parse_args(argv)
    report = build_budget(args.root)
    output = args.output_dir or args.root
    save_json(output / 'budget.json', report)
    (output / 'BUDGET.md').write_text(markdown(report))
    print(json.dumps(dict(coverage={k: v for k, v in report['coverage'].items() if not isinstance(v, list)},
                          budget=report['budget'], output_dir=str(output.resolve()))))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

"""One fixed cache-only all18 VAL A/B probability-average screen."""

import argparse
import json
import os
from pathlib import Path
import time
import traceback

os.environ['CUDA_VISIBLE_DEVICES'] = ''
for _name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_name] = '2'

import numpy as np

from code.unified_selector.registry import PROBLEMS
from .performance_evaluation import decision_metrics
from .r40_experiment import aggregate
from .r76_probability_diagnostics import file_hash, masked_softmax_fp32, read_json
from .r78_zero_selection import progress, utc, write_csv, write_json


GOAL = Path(__file__).resolve().parent / 'runs/R78_zero_selection_goal'
ROOT = GOAL / 'round005'
PLAN = GOAL / 'review-stage/AB_EQUAL_WEIGHT_SCREEN_PLAN.md'
MANIFEST = GOAL.parent / 'R77_sample_size_batch1024/selected_train_val_predictions/manifest.json'
CHUNK = 1024


def probability_average(a, b, mask):
    pa, _ = masked_softmax_fp32(a, mask)
    pb, _ = masked_softmax_fp32(b, mask)
    return np.float32(.5) * pa + np.float32(.5) * pb


def mixture_nll(probabilities, winner):
    normalized = probabilities.astype(np.float64)
    normalized /= normalized.sum(1, keepdims=True, dtype=np.float64)
    return -np.log(normalized[np.arange(len(winner)), winner])


def cache_problem(saved, problem, row_order=None):
    prefix = problem + '__'
    indices = saved[prefix + 'indices']
    rows = np.argsort(indices, kind='stable') if row_order is None else row_order
    ids = saved[prefix + 'pool_ids']
    columns = np.argsort(ids, kind='stable')
    inverse = np.argsort(columns)
    value = {name: saved[prefix + name][rows] for name in ('logits', 'costs', 'source_indices', 'nodes')}
    value['logits'], value['costs'] = value['logits'][:, columns], value['costs'][:, columns]
    value.update(winner=inverse[saved[prefix + 'winner'][rows]], pool=saved[prefix + 'pool'][columns],
                 pool_ids=ids[columns], indices=indices[rows], mask=~np.isneginf(value['logits']))
    if (len(rows) != 3895 or len(np.unique(value['source_indices'])) != len(rows)
            or value['logits'].dtype != np.float32 or value['costs'].dtype != np.float64):
        raise ValueError('Expected complete3895-row native FP32-logit/FP64-cost VAL cache')
    return value


def align_problem(saved_a, saved_b, problem):
    b = cache_problem(saved_b, problem)
    source_a = saved_a[problem + '__source_indices']
    ordered = np.argsort(source_a, kind='stable')
    locations = np.searchsorted(source_a[ordered], b['source_indices'])
    if (locations >= len(ordered)).any():
        raise ValueError('A lacks a B source row')
    a = cache_problem(saved_a, problem, ordered[locations])
    for field in ('source_indices', 'nodes', 'pool', 'pool_ids', 'costs', 'winner', 'mask'):
        if not np.array_equal(a[field], b[field]):
            raise ValueError(f'{problem}: native A/B {field} mismatch')
    return a, b


def metrics(probabilities, nll, value, reference):
    raw = dict(costs=value['costs'], winner=value['winner'], pool=value['pool'].tolist())
    result, choice = decision_metrics(probabilities, raw, value['mask'])
    fixed, oracle = reference['sbs_cost'], result['oracle_cost']
    result.update(ce=float(nll.mean(dtype=np.float64)), pool=raw['pool'],
        sbs=fixed, sbs_cost=fixed, sbs_name=reference['sbs_name'], sbs_selection_split='train',
        vs_sbs_pct=(result['mean_cost'] / fixed - 1) * 100,
        vbs_gap_closed_pct=(fixed - result['mean_cost']) / (fixed - oracle) * 100 if fixed != oracle else 0.,
        native_correct=int((choice == value['winner']).sum()),
        CE_definition='FP64 log of normalized fixed FP32 probability mixture; descriptive')
    return result, choice


def run(root):
    started, started_utc = time.monotonic(), utc()
    root.mkdir(parents=True, exist_ok=True)
    if (root / 'protocol.json').exists():
        raise FileExistsError('Round005 already screened; no repeated candidate or rescue')
    manifest = read_json(MANIFEST)
    members = {}
    for name, point, update in (('A', 13, 16146), ('B', 28, 34776)):
        arm = manifest['arms'][name]
        if (arm['evaluation_point'], arm['selected_update']) != (point, update) or not arm['ready']:
            raise ValueError('Unexpected locked member')
        geometry = Path(arm['geometry_normalization'])
        members[name] = dict(checkpoint=arm['checkpoint'], checkpoint_sha256=arm['checkpoint_sha256'],
            epoch=point, update=update, probability_weight=.5, original_data_root=arm['data_root'],
            original_preprocessing=dict(geometry_normalization=str(geometry), sha256=file_hash(geometry),
                                        state=read_json(geometry)),
            VAL_cache=arm['files']['val'])
    baseline = read_json(Path(manifest['arms']['B']['files']['val']['metrics_path']))
    if not manifest['complete'] or set(baseline['per_problem']) != set(PROBLEMS):
        raise ValueError('Both members and all18 B reference tasks must be locked')
    protocol = dict(started_utc=started_utc, pid=os.getpid(), plan=str(PLAN), plan_sha256=file_hash(PLAN),
        source_manifest=str(MANIFEST), source_manifest_sha256=file_hash(MANIFEST), members=members,
        rule='argmax(0.5*masked_softmax_FP32(z_A,T1)+0.5*masked_softmax_FP32(z_B,T1))',
        exact_tie_policy='ascending original global solver ID', all_tasks_same_rule=True,
        chunk_size=CHUNK, processing_chunks_are_not_training_batches=True, optimizer_updates=0,
        candidate_VAL_endpoints=1, rows=70110, tasks=18, GPU_used=False, backbone_passes=0,
        solver_calls=0, external_API_calls=0, TEST_read=False, LIB_read=False,
        deployment=dict(networks=2, approximate_network_arithmetic_vs_B=2,
                        real_forward_wall_time_measured=False, screening_is_not_deployment_inference=True))
    write_json(root / 'protocol.json', protocol)
    progress(root, 'round005_running_fixed_CPU_VAL_screen', round_number=5, pid=os.getpid(),
        optimizer_updates_executed=0, GPU_used=False, TEST_read=False, LIB_read=False,
        next_action='Execute only one fixed all18 equal-probability VAL candidate; no training, scan or TEST/LIB.')
    per, pools, masks, decisions, task_rows, methods, changes, categories = {}, {}, {}, {}, [], [], [], []
    with np.load(members['A']['VAL_cache']['path'], allow_pickle=False) as saved_a, \
            np.load(members['B']['VAL_cache']['path'], allow_pickle=False) as saved_b:
        for problem in PROBLEMS:
            a, b = align_problem(saved_a, saved_b, problem)
            n, k = b['logits'].shape
            mixed, nll = np.empty((n, k), dtype=np.float32), np.empty(n, dtype=np.float64)
            for start in range(0, n, CHUNK):
                end = min(n, start + CHUNK)
                mixed[start:end] = probability_average(a['logits'][start:end], b['logits'][start:end], b['mask'][start:end])
                nll[start:end] = mixture_nll(mixed[start:end], b['winner'][start:end])
            per[problem], chosen = metrics(mixed, nll, b, baseline['per_problem'][problem])
            old = np.argmax(np.where(b['mask'], b['logits'], -np.inf), axis=1)
            pools[problem] = dict(names=b['pool'].tolist(), global_ids=b['pool_ids'].tolist())
            masks[problem + '__valid_mask'] = b['mask']
            decisions.update({problem + '__source_indices': b['source_indices'],
                              problem + '__probabilities': mixed, problem + '__choice_global_id': b['pool_ids'][chosen]})
            for rule, metric, choice in (('B', baseline['per_problem'][problem], old), ('ensemble', per[problem], chosen)):
                task_rows.append(dict(problem=problem, rule=rule, rows=n, top1_pct=100 * metric['top1'],
                    mean_cost=metric['mean_cost'], actual_regret_pct=metric['actual_regret_pct'],
                    ce=metric['ce'], vs_sbs_pct=metric['vs_sbs_pct']))
                support, picks = np.bincount(b['winner'], minlength=k), np.bincount(choice, minlength=k)
                for slot, method in enumerate(b['pool']):
                    methods.append(dict(problem=problem, rule=rule, method=str(method), solver_id=int(b['pool_ids'][slot]),
                        winner_count=int(support[slot]), selected_count=int(picks[slot]), rows=n,
                        zero_with_winner=bool(support[slot] > 0 and picks[slot] == 0)))
            changed = old != chosen
            corrected = (old != b['winner']) & (chosen == b['winner'])
            harmed = (old == b['winner']) & (chosen != b['winner'])
            wrong = changed & (old != b['winner']) & (chosen != b['winner'])
            row_ids = np.arange(n)
            delta = b['costs'][row_ids, chosen] - b['costs'][row_ids, old]
            oracle = np.where(b['mask'], b['costs'], np.inf).min(1)
            changes.append(dict(problem=problem, changed_predictions=int(changed.sum()), corrected=int(corrected.sum()),
                harmed=int(harmed.sum()), wrong_to_wrong=int(wrong.sum()), exact_cost_tie_changes=int((changed & (delta == 0)).sum()),
                negative_cost_change_sum=float(delta[delta < 0].sum(dtype=np.float64)),
                positive_cost_change_sum=float(delta[delta > 0].sum(dtype=np.float64)),
                mean_cost_delta=float(delta.mean(dtype=np.float64)),
                actual_regret_delta_pp=float((delta / oracle * 100).mean(dtype=np.float64))))
            for category, keep in (('corrected', corrected), ('harmed', harmed), ('wrong_to_wrong', wrong)):
                d = delta[keep]
                categories.append(dict(problem=problem, category=category, changed_rows=int(keep.sum()),
                    cost_decrease_count=int((d < 0).sum()), cost_increase_count=int((d > 0).sum()),
                    negative_cost_change_sum=float(d[d < 0].sum(dtype=np.float64)),
                    positive_cost_change_sum=float(d[d > 0].sum(dtype=np.float64))))
    ensemble = dict(per_problem=per, macro=aggregate(per), native_total=sum(m['n'] for m in per.values()),
                    native_correct=sum(m['native_correct'] for m in per.values()))
    if ensemble['native_total'] != 70110:
        raise ValueError('The sole VAL candidate must include all70110 rows')
    reference, candidate = baseline['macro'], ensemble['macro']
    gate = dict(native_top1_improves=candidate['macro_top1'] > reference['macro_top1'],
        raw_mean_cost_not_worse=candidate['macro_mean_cost'] <= reference['macro_mean_cost'],
        actual_regret_not_worse=candidate['macro_actual_regret_pct'] <= reference['macro_actual_regret_pct'])
    passed = all(gate.values())
    np.savez_compressed(root / 'locked_candidate_masks.npz', **masks)
    np.savez_compressed(root / 'VAL_candidate_predictions.npz', **decisions)
    locked = dict(locked_utc=utc(), members=members, candidate_pools=pools,
        valid_masks=dict(path=str(root / 'locked_candidate_masks.npz'), sha256=file_hash(root / 'locked_candidate_masks.npz'),
                        runtime_rule='original task candidate-valid masks; inactive candidates have probability0'),
        probability='each member raw native final maximin logits -> masked FP32 softmax at T1',
        aggregation='fixed probability weights A0.5/B0.5 for all18 tasks; not logit or weight averaging',
        exact_ties='ascending original global solver ID', VAL_passed=passed,
        formal_TEST_LIB_authorized=False, deployment=protocol['deployment'])
    write_json(root / ('ensemble_release_manifest.json' if passed else 'screen_rule_manifest.json'), locked)
    cold = {rule: dict(zero_with_winner_pairs=sum(r['zero_with_winner'] for r in methods if r['rule'] == rule),
             zero_winner_count=sum(r['winner_count'] for r in methods if r['rule'] == rule and r['zero_with_winner']))
            for rule in ('B', 'ensemble')}
    for item in cold.values():
        item['zero_winner_coverage_pct'] = 100 * item['zero_winner_count'] / 70110
    totals = {key: sum(row[key] for row in changes) for key in
              ('changed_predictions', 'corrected', 'harmed', 'wrong_to_wrong', 'exact_cost_tie_changes')}
    seconds = time.monotonic() - started
    outcome = 'exploratory_VAL_pass_pending_owner_TEST_release' if passed else 'fixed_equal_probability_route_closed_negative'
    report = dict(completed_utc=utc(), outcome=outcome, passed=passed, gate=gate, runtime_seconds=seconds,
        runtime_scope='CPU cached-score screen, not real two-network deployment inference',
        B=baseline, ensemble=ensemble, fallback_B=baseline, selected_rule='ensemble' if passed else 'fallback_B',
        cold_methods=cold, changed_decisions=totals, deployment=protocol['deployment'],
        optimizer_updates=0, GPU_used=False, TEST_read=False, LIB_read=False, formal_TEST_LIB_authorized=False,
        interpretation='Repeated-use VAL exploratory screen only; no per-task safety or cost-of-deployment certification')
    write_json(root / 'results.json', report)
    write_csv(root / 'all18_VAL_metrics.csv', task_rows)
    write_csv(root / 'native_method_counts.csv', methods)
    write_csv(root / 'decision_changes.csv', changes)
    write_csv(root / 'decision_change_costs_by_category.csv', categories)
    lines = ['# R78 Round005 Fixed Equal-Probability Screen', '', 'Outcome: ' + outcome + '.',
        'One complete70110-row VAL candidate /18 tasks /chunks1024. No TRAIN readout or optimizer update.',
        f'CPU cached-screen runtime: {seconds:.6f}s; not real deployment latency.', '',
        '| Rule | Native Top1 % | Raw mean cost | Actual regret % | CE |', '|---|---:|---:|---:|---:|']
    for rule, m in (('B', reference), ('ensemble', candidate), ('fallback_B', reference)):
        lines.append(f"| {rule} | {100*m['macro_top1']:.9f} | {m['macro_mean_cost']:.12f} | {m['macro_actual_regret_pct']:.9f} | {m['macro_ce']:.9f} |")
    lines += ['', 'Gate: ' + json.dumps(gate) + '.', 'Selected whole18 rule: ' + report['selected_rule'] + '.',
        'CE_E is descriptive -log(normalized probability mixture); B CE is the unchanged locked VAL record.',
        'Changed decisions: ' + json.dumps(totals) + '.', 'Cold methods: ' + json.dumps(cold) + '.',
        'Per-task/category adverse FP64 cost changes remain visible; no cross-task raw-cost pooling.',
        'Real deployment needs both frozen model forwards/weight sets, about2x network arithmetic; actual latency/memory is unmeasured.',
        'No TEST/LIB, new weight/temperature/logit rule, taskwise rescue, GPU benchmark, model API, training or follow-up screen.',
        'File goal remains active; native and legacy app goals are not queried or changed.']
    (root / 'RESULTS.md').write_text('\n'.join(lines) + '\n')
    progress(root, 'round005_VAL_pass_waiting_explicit_TEST_release' if passed else 'awaiting_nss_locked_train_val',
        round_number=5, execution_complete=True, pid=os.getpid(), optimizer_updates_executed=0,
        outcome=outcome, gate=gate, selected_rule=report['selected_rule'], runtime_seconds=seconds,
        reports=str(root / 'results.json'), GPU_used=False, TEST_read=False, LIB_read=False, goal_complete=False,
        native_goal_queried_or_mutated=False,
        next_action='Deliver the one locked ensemble manifest and all VAL results; wait for explicit owner TEST approval and tiny real-forward deployment check.' if passed else
                    'Close fixed equal-probability route, retain B and await locked NSS TRAIN/VAL; no rescue, new screen, TEST/LIB or extra check.')
    print(json.dumps(dict(outcome=outcome, gate=gate, runtime_seconds=seconds, B=reference,
                         ensemble=candidate, changed_decisions=totals, cold_methods=cold), indent=2), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    args = parser.parse_args()
    try:
        run(args.root)
    except Exception as error:
        if (args.root / 'protocol.json').exists() and not (args.root / 'results.json').exists():
            write_json(args.root / 'failure.json', dict(error=str(error), traceback=traceback.format_exc()))
            progress(args.root, 'round005_incomplete_screen_retaining_B', round_number=5, error=str(error),
                     optimizer_updates_executed=0, GPU_used=False, TEST_read=False, LIB_read=False, goal_complete=False)
        raise


if __name__ == '__main__':
    main()

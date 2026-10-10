"""Summarize R57 evidence and replay validation metrics, without model selection."""

import argparse
import csv
import importlib.metadata
import json
from pathlib import Path
import sys

import numpy as np
import torch

from code.unified_selector.registry import POOLS, PROBLEMS
from .multitask_probe import dump, file_hash
from .r48_common import PAIR, write_csv
from .r57_routes import check_route
from .r57_signal import ROOT, load_instances


def read_csv(path):
    with Path(path).open(newline='') as stream:
        return list(csv.DictReader(stream))


def runtime_analysis(root):
    records, metadata = [], []
    for path in sorted((root / 'runtime').glob('*/*.jsonl')):
        if 'preflight' in path.name or path.parent.name == 'SELECTOR':
            continue
        records.extend(json.loads(line) for line in path.read_text().splitlines())
    for path in sorted((root / 'runtime').glob('*/*.metadata.json')):
        metadata.append(dict(path=str(path.resolve()), **json.loads(path.read_text())))
    original = {(r['problem'], r['split'], r['index'], r['solver']): r
                for r in records if r['variant'] == 'original_0'}
    selector_path = root / 'runtime/SELECTOR/original_0.jsonl'
    selector = {}
    if selector_path.exists():
        selector = {(r['problem'], r['split'], r['index'], r['variant']): r
                    for r in map(json.loads, selector_path.read_text().splitlines())}
    rows = []
    instance_cache = {}
    val_cache = {}
    for r in records:
        key = (r['problem'], r['split'], r['index'], r['solver'])
        if key not in original:
            continue
        first = original[key]
        row = {k: r[k] for k in ('problem', 'split', 'index', 'solver', 'variant', 'true_size',
                                 'seed', 'cost', 'historical_cost', 'physical_start_set_changed')}
        row.update(abs_historical_cost_error=abs(r['cost'] - r['historical_cost']),
            historical_cost_matches=bool(np.isclose(r['cost'], r['historical_cost'], atol=5e-5, rtol=2e-6)),
            cost_difference_from_original=r['cost'] - first['cost'],
            cost_changed_beyond_tolerance=bool(not np.isclose(r['cost'], first['cost'], atol=5e-5, rtol=2e-6)),
            exact_cost_changed=r['cost'] != first['cost'], seconds=r['seconds'],
            original_physical_starts=json.dumps(r['original_physical_starts']),
            current_physical_starts=json.dumps(r['current_physical_starts']),
            independent_route_status='not_verified_no_route')
        if r['split'] == 'val':
            if r['problem'] not in val_cache:
                with np.load(root / 'signal_predictions' / (r['problem'] + '.npz'), allow_pickle=False) as a:
                    val_cache[r['problem']] = {k: a[k] for k in ('costs', 'winner', 'R45A_pred', 'pool')}
            saved_val = val_cache[r['problem']]
            i, pool = r['index'], list(saved_val['pool'])
            pred, winner = int(saved_val['R45A_pred'][i]), int(saved_val['winner'][i])
            cost = saved_val['costs'][i]
            row.update(old_selected_solver=pool[pred], old_native_winner=pool[winner],
                probed_method_was_selected=r['solver'] == pool[pred],
                probed_method_was_winner=r['solver'] == pool[winner],
                old_selector_regret_pct=float((cost[pred] / cost.min() - 1) * 100))
        if r['route_original_ids'] is not None and r['problem'] not in ('TSP', 'ATSP', 'CVRP'):
            group = (r['problem'], r['split'])
            if group not in instance_cache:
                instance_cache[group] = load_instances(*group)
            checked = check_route(instance_cache[group][r['index']], r['route_original_ids'], r['problem'])
            row.update(independent_route_status='classical_checked', independent_cost=checked['cost'],
                independent_cost_matches=bool(np.isclose(checked['cost'], r['cost'], atol=5e-5, rtol=2e-6)),
                explicit_classical_feasible=checked['explicit_classical_feasible'],
                signed_load_ok=checked['signed_load_ok'])
        skey = (r['problem'], r['split'], r['index'], r['variant'])
        original_score = selector.get((*skey[:3], 'original_0'))
        score = selector.get(skey)
        if score is not None and original_score is not None:
            difference = np.asarray(score['scores']) - np.asarray(original_score['scores'])
            row.update(selector_max_abs_score_difference=float(np.max(np.abs(difference))),
                selector_choice_changed=int(np.argmax(score['scores'])) != int(np.argmax(original_score['scores'])))
        rows.append(row)
    write_csv(root / 'solver_audit.csv', rows)
    summary = []
    for key in sorted({(r['problem'], r['split'], r['solver'], r['variant']) for r in rows}):
        group = [r for r in rows if (r['problem'], r['split'], r['solver'], r['variant']) == key]
        summary.append(dict(problem=key[0], split=key[1], solver=key[2], variant=key[3], n=len(group),
            historical_cost_matches=sum(r['historical_cost_matches'] for r in group),
            original_cost_changed=sum(r['cost_changed_beyond_tolerance'] for r in group),
            exact_cost_changed=sum(r['exact_cost_changed'] for r in group),
            physical_start_set_changed=sum(r['physical_start_set_changed'] is True for r in group),
            route_available=sum(r['independent_route_status'] != 'not_verified_no_route' for r in group),
            max_abs_original_cost_difference=max(abs(r['cost_difference_from_original']) for r in group)))
    write_csv(root / 'runtime_summary.csv', summary)
    affected = []
    for problem, variant in sorted({(r['problem'], r['variant']) for r in rows if r['split'] == 'val'}):
        group = [r for r in rows if r['problem'] == problem and r['variant'] == variant and r['split'] == 'val']
        by_index = {}
        for r in group:
            by_index.setdefault(r['index'], []).append(r)
        any_changed = [v for v in by_index.values() if any(r['cost_changed_beyond_tolerance'] for r in v)]
        chosen_changed = [v for v in by_index.values() if any(r['cost_changed_beyond_tolerance'] and r['probed_method_was_selected'] for r in v)]
        winner_changed = [v for v in by_index.values() if any(r['cost_changed_beyond_tolerance'] and r['probed_method_was_winner'] for r in v)]
        affected.append(dict(problem=problem, variant=variant, inspected_val_instances=len(by_index),
            any_probed_method_cost_changed=len(any_changed), selected_probed_method_cost_changed=len(chosen_changed),
            old_winner_probed_method_cost_changed=len(winner_changed),
            observed_old_regret_all_pp=sum(v[0]['old_selector_regret_pct'] for v in any_changed) / 1000 / 18,
            observed_selected_method_old_regret_all_pp=sum(v[0]['old_selector_regret_pct'] for v in chosen_changed) / 1000 / 18,
            not_extrapolated=True, full_pool_winner_change_verified=False))
    write_csv(root / 'runtime_affected_budget.csv', affected)
    selector_rows = []
    for key, score in selector.items():
        if key[-1] == 'original_0':
            continue
        first = selector[(*key[:3], 'original_0')]
        difference = np.asarray(score['scores']) - np.asarray(first['scores'])
        selector_rows.append(dict(problem=key[0], split=key[1], index=key[2], variant=key[3],
            max_abs_score_difference=float(np.max(np.abs(difference))),
            argmax_changed=int(np.argmax(score['scores'])) != int(np.argmax(first['scores']))))
    write_csv(root / 'selector_permutation.csv', selector_rows)
    pair_rows = []
    index = {(r['problem'], r['split'], r['index'], r['solver'], r['variant']): r for r in records}
    for problem, split, i in sorted({key[:3] for key in index if key[3] == PAIR[0]}):
        first_a = index.get((problem, split, i, PAIR[0], 'original_0'))
        first_b = index.get((problem, split, i, PAIR[1], 'original_0'))
        if not first_a or not first_b:
            continue
        for variant in ('original_0', 'original_1', 'preserve_roles', 'reassign_roles'):
            a = index.get((problem, split, i, PAIR[0], variant))
            b = index.get((problem, split, i, PAIR[1], variant))
            if not a or not b:
                continue
            gap, before = b['cost'] - a['cost'], first_b['cost'] - first_a['cost']
            tolerance = 5e-5 + 2e-6 * max(a['cost'], b['cost'])
            sign = lambda value: int(np.sign(value))
            numeric = lambda value: 0 if abs(value) <= tolerance else sign(value)
            pair_rows.append(dict(problem=problem, split=split, index=i, variant=variant,
                moel_cost=a['cost'], mtl_cost=b['cost'], original_moel_cost=first_a['cost'],
                original_mtl_cost=first_b['cost'], pair_winner_changed_strict=sign(gap) != sign(before),
                pair_winner_changed_tolerant=numeric(gap) != numeric(before),
                strict_pair_winner=sign(gap), numerical_pair_winner=numeric(gap),
                physical_start_set_changed=a['physical_start_set_changed'],
                full_pool_winner_verified=False))
    write_csv(root / 'permutation_audit.csv', pair_rows)
    status = json.loads((root / 'runtime_status.json').read_text()) if (root / 'runtime_status.json').exists() else {}
    observed = {(r['problem'], r['solver']) for r in rows if r['variant'] == 'original_0'}
    coverage = []
    for r in read_csv(root / 'deployment_audit.csv'):
        key = r['problem'], r['solver']
        runtime = [x for x in rows if (x['problem'], x['solver']) == key and x['variant'] == 'original_0']
        coverage.append(dict(problem=r['problem'], solver=r['solver'], runtime_rows=len(runtime),
            original_runtime_completed=key in observed, full_historical_recipe_certified=False,
            route_independently_checked=any(x['independent_route_status'] != 'not_verified_no_route' for x in runtime),
            static_status=r['runtime_status'], blockers=r['remaining_gaps']))
    write_csv(root / 'runtime_coverage.csv', coverage)
    dump(root / 'runtime_provenance.json', dict(metadata=metadata, status=status))
    return rows, selector_rows, pair_rows, coverage


def scope_budgets(root):
    rows = []
    for problem in PROBLEMS:
        with np.load(root / 'signal_predictions' / (problem + '.npz'), allow_pickle=False) as a:
            costs, winner, pred = a['costs'], a['winner'], a['R45A_pred']
            pool = list(a['pool'])
        regret = (costs[np.arange(len(pred)), pred] - costs.min(1)) / costs.min(1)
        pair = [pool.index(s) for s in PAIR if s in pool]
        mutual = np.isin(pred, pair) & np.isin(winner, pair) & (pred != winner)
        current_signed = [pool.index(s) for s in ('MTPOMO', 'MVMOE', *PAIR) if s in pool]
        current_classical = [pool.index(s) for s in ('MoSES_CaDA', 'MoSES_RF', 'RouteFinder') if s in pool]
        cross_contract = ((np.isin(pred, current_signed) & np.isin(winner, current_classical)) |
                          (np.isin(pred, current_classical) & np.isin(winner, current_signed)))
        active = problem not in ('TSP', 'CVRP', 'ATSP') and 'B' in problem
        rows.append(dict(problem=problem, n=len(pred), R45A_top1=float((pred == winner).mean()),
            actual_regret_pct=float(regret.mean() * 100), all_regret_contribution_pp=float(regret.mean() * 100 / 18),
            current_contract_backhaul_conflict=active,
            moel_mtl_mutual_errors=int(mutual.sum()),
            moel_mtl_mutual_regret_all_pp=float((regret * mutual).mean() * 100 / 18),
            cross_current_contract_errors=int(cross_contract.sum()) if active else 0,
            cross_current_contract_regret_all_pp=float((regret * cross_contract).mean() * 100 / 18) if active else 0.,
            attribution='observed old-label budget, NOT recoverable improvement and NOT all historical native contracts certified'))
    total = sum(r['all_regret_contribution_pp'] for r in rows)
    for row in rows:
        row['share_of_all_regret_pct'] = row['all_regret_contribution_pp'] / total * 100
    write_csv(root / 'contract_scope_budget.csv', rows)
    return rows


def replay(root):
    saved = read_csv(root / 'signal_metrics.csv')
    checks = []
    with np.load(root / 'permutation_maps.npz', allow_pickle=False) as maps:
        for problem in PROBLEMS:
            with np.load(root / 'signal_predictions' / (problem + '.npz'), allow_pickle=False) as a:
                c, y, scores = a['costs'], a['winner'], a['R45A_scores']
                for name in ('R45A', 'conditional_majority', 'conditional_mean_gap'):
                    pred = a[name + '_pred']
                    row = next(r for r in saved if r['problem'] == problem and r['strategy'] == name and r['scope'] == 'full')
                    np.testing.assert_allclose(float(row['top1']), (pred == y).mean(), rtol=0, atol=1e-12)
                    np.testing.assert_allclose(float(row['mean_cost']), c[np.arange(len(y)), pred].mean(), rtol=0, atol=1e-12)
                    expected = ((c[np.arange(len(y)), pred] - c.min(1)) / c.min(1) * 100).mean()
                    np.testing.assert_allclose(float(row['actual_regret_pct']), expected, rtol=0, atol=1e-12)
                for repeat, mapping in enumerate(maps[problem]):
                    np.testing.assert_array_equal(np.sort(mapping), np.arange(len(y)))
                    np.testing.assert_array_equal(a['true_size'][mapping], a['true_size'])
                    self_rows = mapping == np.arange(len(y))
                    np.testing.assert_array_equal(self_rows, ~a['permutation_eligible'])
                    pred = scores[mapping].argmax(1)
                    row = next(r for r in saved if r['problem'] == problem and r['strategy'] == 'permuted_R45A'
                               and r['scope'] == 'full' and int(r['repeat']) == repeat)
                    np.testing.assert_allclose(float(row['actual_regret_pct']),
                        ((c[np.arange(len(y)), pred] - c.min(1)) / c.min(1) * 100).mean(), rtol=0, atol=1e-12)
                checks.append(dict(problem=problem, rows=len(y), policies=3, permutations=len(maps[problem]), status='passed'))
    dump(root / 'prediction_replay.json', dict(status='passed', checks=checks,
        metrics=['native Top1', 'FP64 mean cost', 'actual regret'], macro_weighting='equal 1/18'))


def plots(root, signal):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    strategies = ('R45A', 'conditional_majority', 'conditional_mean_gap', 'permuted_R45A')
    colors = ('#16755c', '#555b6e', '#c07824', '#b54848')
    fig, axes = plt.subplots(2, 1, figsize=(16, 8), sharex=True, constrained_layout=True)
    x = np.arange(len(PROBLEMS))
    for j, (strategy, color) in enumerate(zip(strategies, colors)):
        values = [next(r for r in signal if r['problem'] == p and r['strategy'] == strategy and r['scope'] == 'full') for p in PROBLEMS]
        axes[0].bar(x + (j - 1.5) * .2, [float(r['top1']) * 100 for r in values], width=.19, label=strategy, color=color)
        axes[1].bar(x + (j - 1.5) * .2, [float(r['actual_regret_pct']) for r in values], width=.19, color=color)
    axes[0].set_ylabel('Native Top1 (%)')
    axes[1].set_ylabel('Actual regret (%)')
    axes[0].legend(ncol=4, fontsize=9)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(PROBLEMS, rotation=45, ha='right', fontsize=9)
    for axis in axes:
        axis.spines[['top', 'right']].set_visible(False)
        axis.grid(axis='y', alpha=.15)
        axis.set_axisbelow(True)
    fig.savefig(root / 'instance_signal.png', dpi=170)
    plt.close(fig)


def run(root=ROOT):
    signal = read_csv(root / 'signal_comparison.csv')
    route = read_csv(root / 'route_check_summary.csv')
    runtime, selector, pair_rows, coverage = runtime_analysis(root)
    budgets = scope_budgets(root)
    replay(root)
    plots(root, signal)
    experiment = json.loads((root / 'deployment_audit.json').read_text())
    config = dict(experiment='R57', seed=2, training=False, test_evaluation=False, labels_replaced=False,
        access_scope=dict(numerical_analysis=['train', 'val'], solver_runs=['train', 'val'],
            reviewer_test_cost_files_hashed=128, reviewer_test_cost_values_parsed=False,
            note='Independent reviewer hashed full test result-file bytes for identity only; no test instances or performance metrics were analyzed.'),
        historical_defaults_assumed=False, python=sys.version, torch=torch.__version__, numpy=np.__version__,
        packages={name: importlib.metadata.version(name) for name in ('torchrl', 'tensordict')},
        baseline=experiment['selector'], legal_problem_solver_pairs=128,
        idle_gpu_uuid='GPU-cb626bdb-4319-2f5e-b91f-17fcaa44a83d',
        runtime_coverage=sum(r['original_runtime_completed'] for r in coverage),
        static_coverage=128, full_historical_certifications=0,
        source_files={str(path): file_hash(path) for path in sorted(Path('code/V4').glob('*r57*')) if path.is_file()},
        commands=['bash code/V4/run_v4_r57.sh cpu', 'bash code/V4/run_v4_r57.sh gpu', 'bash code/V4/run_v4_r57.sh analysis'])
    dump(root / 'config.json', config)
    train_routes = [r for r in route if r['split'] == 'train']
    val_routes = [r for r in route if r['split'] == 'val']
    train_fail = sum(int(r['explicit_classical_feasible_failures']) for r in train_routes)
    val_fail = sum(int(r['explicit_classical_feasible_failures']) for r in val_routes)
    b_routes = [r for r in train_routes if 'B' in r['problem']]
    b_total = sum(int(r['n']) for r in b_routes)
    all_rows = [r for r in signal if r['problem'] == 'ALL' and r['scope'] == 'full']
    full_regret = next(float(r['actual_regret_pct']) for r in all_rows if r['strategy'] == 'R45A')
    b_budget = sum(r['all_regret_contribution_pp'] for r in budgets if r['current_contract_backhaul_conflict'])
    cross_budget = sum(r['cross_current_contract_regret_all_pp'] for r in budgets)
    originals = [r for r in runtime if r['variant'] == 'original_0']
    flipped = [r for r in pair_rows if r['variant'] == 'reassign_roles'
               and r['pair_winner_changed_tolerant']]
    text = ['# R57: Task Contracts and Instance-Level Selection Signal', '',
        '## Scope', '',
        'No selector training, test evaluation, new instances, labels or checkpoint changes. Seed=2.',
        'All18 tasks / 22 solver names / 128 legal task-solver pairs / 256 train-val cost columns audited.',
        'Numerical results use original FP64 costs and native winner; ALL is an equal-weight problem macro.',
        'All 256 checked train/val cost columns match their saved result files exactly; native winner is a '
        'minimum-cost method for every checked instance. These checks establish alignment, not common task semantics.',
        'Only the previously idle RTX3090 was used. Existing unrelated work was not stopped.', '',
        'Access disclosure: numerical analysis and runtime probes use train/val only. The independent reviewer '
        'additionally read full bytes of 128 historical test cost files for SHA256 identity checks; no test '
        'cost values were parsed and no test performance was evaluated. This was file-content access, not metadata-only.', '',
        '## Main Findings', '',
        '### 1. A concrete backhaul task-contract discrepancy, not a cost-replay failure', '',
        f'Independently checked 300,000 saved training routes and 1,920 fixed validation diagnostic routes. '
        f'Under classical B (deliveries before pickups, separate per-route delivery/pickup capacities), '
        f'{train_fail:,}/{b_total:,} training B routes and {val_fail:,}/1,024 validation B routes fail.',
        'All routes pass the independently simulated native signed-residual-load rule. No customer-TW or route-length '
        'violations were found; maximum training cost error is '
        f"{max(float(r['max_abs_cost_error']) for r in train_routes):.9g} (original replay tolerance: atol=5e-5, rtol=2e-6).",
        'The existing ReLD B environments allow interleaved deliveries/pickups and permit pickup demand to restore '
        'residual load. RF/MoSES default to classical backhaul class1; the original adapter does not override it. '
        'These are different feasible-route contracts, not alternative names for the same capacity test.',
        'Raw data record signed demand but no backhaul_class. Therefore first decide and version the intended task '
        'contract; do not silently pick one, overwrite costs, or declare every historical label corrupt.',
        f'The eight affected problem types account for {b_budget/full_regret*100:.2f}% of R45A old-validation regret. '
        f'Cross-current-contract selected/winner comparisons account for {cross_budget/full_regret*100:.2f}%. '
        'These are scope budgets, NOT recoverable improvement estimates. Full historical RF/MoSES deployments remain unclosed.',
        '**This does not explain the MOEL/MTL binary plateau by itself: both ReLD methods use the same signed-load contract.**',
        'See route_check_summary.csv, route_checks.csv.gz, route_violation_examples.json and contract_scope_budget.csv.', '',
        '### 2. The baseline does use instance information beyond problem/size priors', '',
        '| Strategy | Val Top1 (%) | Actual regret (%) | Mean raw cost | Net cost vs R45A |',
        '|---|---:|---:|---:|---:|']
    for r in all_rows:
        text.append(f"| {r['strategy']} | {100*float(r['top1']):.4f} | {float(r['actual_regret_pct']):.6f} | "
                    f"{float(r['mean_cost']):.8f} | {float(r['net_cost_vs_R45A']):+.8f} |")
    text.extend(['', 'Priors are fit only on train: exact actual size, any varying scalar global constraints, '
        'and fixed 20-observation problem-global smoothing. In these actual datasets capacity=1 and route_limit=3 '
        'are constant; no coordinate distribution was guessed from indices. Majority native winner and minimum '
        'mean relative gap are different objectives and are reported separately.',
        '100 fixed-seed score-row derangements are within problem x exact size. Singleton rows remain unchanged '
        'in full-scope metrics and are excluded in the separately reported permutable_only comparison. '
        'The sampler uses randomized cyclic shifts, yielding valid but not uniformly sampled derangements; '
        'no uniform-permutation-test p-value is claimed. '
        'Coverage: TSP 88.9%, CVRP 89.3%, other tasks100%; equal-task macro98.7889%. '
        'Reported permutation quantiles describe this conditional randomization, not a population confidence interval or accuracy ceiling.',
        'Per-problem results are below. Variation across problems matters: a common ALL plateau is not proof '
        'of a common information-free representation or one universal network bug.', '',
        '| Problem | R45A Top1 | Majority Top1 | Gap-prior Top1 | Permuted Top1 | R45A regret | Permuted regret |',
        '|---|---:|---:|---:|---:|---:|---:|'])
    for problem in PROBLEMS:
        values = {r['strategy']: r for r in signal if r['problem'] == problem and r['scope'] == 'full'}
        top = [100 * float(values[name]['top1']) for name in ('R45A', 'conditional_majority', 'conditional_mean_gap', 'permuted_R45A')]
        text.append(f"| {problem} | " + ' | '.join(f'{v:.2f}%' for v in top) +
                    f" | {float(values['R45A']['actual_regret_pct']):.4f}% | {float(values['permuted_R45A']['actual_regret_pct']):.4f}% |")
    text.extend(['', '### 3. Runtime verification is explicitly partial', '',
        f"Original-input runtime completed for {sum(r['original_runtime_completed'] for r in coverage)}/128 legal pairs. "
        'The full matrix remains covered by static input/deployment/label inspection. No row is called fully historically certified.',
        'EasyNCO GLOP/MATNET/MATPOENET/ICAM preflights fail in the existing environment at TorchRL native-library ABI '
        '(undefined ATen clamp symbol). We did not install another dependency set and present it as the historical environment.',
        'NSS import recipes, MTPOMO/MVMOE effective decoder, and RF/MoSES runtime/import bindings are still unclosed. '
        'ICAM_ATSP/UniCO have current bridge-derived settings but no historical resolved snapshot; no default substitution was performed.',
        'Original runs are independent processes. Permutations synchronize node attributes and keep the depot fixed. '
        'For B ReLD, preserve_roles keeps the physical truncated positive-demand start set, while reassign_roles may change it. '
        'For single-task ReLD_CVRP starts are learned top-k, NOT first100 IDs; its start-set field is left unverified.',
        f"All {len(originals)} original solver-instance costs reproduce the historical scalar costs within "
        'atol=5e-5, rtol=2e-6. This does not certify unrecorded historical source/dependency revisions.',
        'No full-pool winner change is inferred from partial candidates. ReLD pair changes are explicitly labeled pair-only.',
        'See runtime_coverage.csv, runtime_summary.csv, selector_permutation.csv, permutation_audit.csv and runtime/* logs.', '',
        '| Probe | Solver-instance rows | Cost changed beyond tolerance | Pair comparisons | Pair winner flips (strict / numeric) |',
        '|---|---:|---:|---:|---:|'])
    for variant in ('original_1', 'preserve_roles', 'reassign_roles'):
        these = [r for r in runtime if r['variant'] == variant]
        pair = [r for r in pair_rows if r['variant'] == variant]
        text.append(f"| {variant} | {len(these)} | {sum(r['cost_changed_beyond_tolerance'] for r in these)} | {len(pair)} | "
            f"{sum(r['pair_winner_changed_strict'] for r in pair)} / {sum(r['pair_winner_changed_tolerant'] for r in pair)} |")
    if selector:
        text.extend(['', f"Frozen R45A: {len(selector)} synchronized node-permutation comparisons; "
            f"{sum(r['argmax_changed'] for r in selector)} greedy choices change; maximum absolute logit difference "
            f"{max(r['max_abs_score_difference'] for r in selector):.9g}. "
            'ATSP is measured rather than assumed invariant. Runtime affected historical-regret budgets '
            'are in runtime_affected_budget.csv, limited to actually inspected validation rows; not extrapolated.', ''])
    text.extend(['### 4. A measured execution-role mismatch, with limited observed scope', '',
        'Unlike a pure renumbering, reassign_roles can change the physical subset of positive-demand '
        'POMO starts when the B environment truncates that set. The selector has no corresponding marker. '
        'The following pair winners actually flip beyond numerical tolerance; these are not hypothetical examples.', '',
        '| Problem / split / index | Original MOEL / MTL cost | Reassigned MOEL / MTL cost |',
        '|---|---:|---:|'])
    for row in flipped:
        text.append(f"| {row['problem']} / {row['split']} / {row['index']} | "
                    f"{row['original_moel_cost']:.7f} / {row['original_mtl_cost']:.7f} | "
                    f"{row['moel_cost']:.7f} / {row['mtl_cost']:.7f} |")
    text.extend(['',
        'Both observed flips are training instances; no ReLD pair winner flips in the sampled validation '
        'instances. Preserving the physical start set gives zero pair flips. These small probes establish '
        'that execution roles can matter, not that this effect explains the overall plateau. '
        'Original-input repeats remain stable. Costs from different start subsets are not substituted into '
        'the original labels, and no full-pool winner is claimed.', ''])
    text.extend([
        '## Next Decision', '',
        'First settle the backhaul class/capacity contract and bind all deployment protocols, retaining old data as a separate scenario. '
        'Only then decide affected columns needing re-evaluation. Do not begin another attention-model experiment on the claim '
        'that all current costs are already semantically comparable.',
        'The prior/permutation evidence supports genuine instance-dependent baseline signal. It does not prove unused signal '
        'is easy to learn, labels are random, or 47% is an upper bound. Missing historical TSP role/recipe information remains '
        'a concrete audit gap, not a demonstrated cause of its errors.', '',
        '## Literature Boundary', '',
        '[NSS Appendix A.9/Table8](https://arxiv.org/html/2410.09693v2#A9) reports greedy TSP classification36%/ranking35%, '
        'CVRP61%/62%; its candidate pool is different. This contextualizes Top1, not a target or ceiling for this project. '
        'Multi-solver Top-k/rejection/Top-p changes runtime budget and is not a single-method fix.',
        '[ASlib](https://arxiv.org/abs/1506.02465) motivates explicit algorithm-selection scenarios and execution records. '
        '[Alissa et al.](https://link.springer.com/article/10.1007/s10732-022-09505-4) and '
        '[Smith-Miles and Bowly](https://research.monash.edu/en/publications/generating-new-test-instances-by-evolving-in-instance-space/) '
        'motivate examining instance/algorithm structure and coverage; neither establishes a property of these local labels.', '',
        '## Reproduction', '',
        'Commands and environment: config.json and run_v4_r57.sh. Source evidence: source_findings.md. '
        'Prediction replay: prediction_replay.json. Focused tests: test_r57.py. '
        'Large existing datasets, weights and route caches remain local; hashes/paths are recorded, not duplicated.', ''])
    (root / 'RESULTS.md').write_text('\n'.join(text))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    run(parser.parse_args().root)

"""Instance-level best-of-many mechanism diagnostics; no labels are replaced."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .multitask_probe import dump, file_hash
from .r48_common import PAIR, write_csv
from .r52_error_budget import ROOT


def winner_of(costs):
    costs = np.asarray(costs)
    return np.where(costs[..., 0] < costs[..., 1], 0,
                    np.where(costs[..., 1] < costs[..., 0], 1, -1))


def start_diagnostics(costs, index, seed=2, repeats=20, fraction=.8):
    """Both methods use the same retained start positions, sampled without replacement."""
    costs = np.asarray(costs)
    if costs.ndim != 2 or costs.shape[0] != 2 or not np.isfinite(costs).all():
        raise ValueError('Expected two complete, finite start-cost vectors')
    best = costs.min(1)
    winner = int(winner_of(best))
    support = int((costs[winner] < best[1 - winner]).sum()) if winner >= 0 else 0
    count = costs.shape[1]
    quantiles = np.quantile(costs, [.1, .25, .5], axis=1, method='linear')
    quantile_winners = winner_of(quantiles)
    retained_count = int(np.ceil(fraction * count))
    rng = np.random.default_rng(np.random.SeedSequence([seed, int(index)]))
    retained = np.zeros((repeats, count), dtype=bool)
    subset_winners = np.empty(repeats, dtype=np.int64)
    for repeat in range(repeats):
        retained[repeat, rng.choice(count, retained_count, replace=False)] = True
        subset_winners[repeat] = winner_of(costs[:, retained[repeat]].min(1))
    flips = (subset_winners >= 0) & (subset_winners != winner) & (winner >= 0)
    ties = subset_winners == -1
    return dict(min_winner=winner, min_costs=best, support_count=support,
        support_fraction=support / count, quantile_costs=quantiles,
        quantile_winners=quantile_winners, retained_count=retained_count,
        subset_winners=subset_winners, retained_masks=retained,
        strict_flip_count=int(flips.sum()), subset_tie_count=int(ties.sum()),
        strict_flip_fraction=float(flips.mean()), sensitive=bool(flips.any()))


def read_csv(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def table(rows, fields, headings=None):
    headings = headings or fields
    lines = ['| ' + ' | '.join(headings) + ' |', '|' + '|'.join(['---'] * len(fields)) + '|']
    for row in rows:
        values = []
        for field in fields:
            value = row[field]
            values.append(f'{value:.4f}' if isinstance(value, (float, np.floating)) else str(value))
        lines.append('| ' + ' | '.join(values) + ' |')
    return '\n'.join(lines)


def summarize_group(rows, all_rows, label, overall_regret):
    loss = sum(r['selector_actual_regret_pct'] for r in rows)
    total_loss = sum(r['selector_actual_regret_pct'] for r in all_rows)
    pair_errors = [r for r in rows if r['selector_target_pair_error']]
    all_pair_errors = [r for r in all_rows if r['selector_target_pair_error']]
    pair_loss = sum(r['selector_actual_regret_pct'] for r in pair_errors)
    all_pair_loss = sum(r['selector_actual_regret_pct'] for r in all_pair_errors)
    macro_contribution = loss / len(all_rows) / 18
    return dict(group=label, instances=len(rows), instance_share_pct=100 * len(rows) / len(all_rows),
        pool_errors=sum(r['selector_wrong'] for r in rows), target_pair_errors=len(pair_errors),
        mean_pair_gap_pct=np.mean([r['pair_gap_pct'] for r in rows]) if rows else 0.,
        median_pair_gap_pct=np.median([r['pair_gap_pct'] for r in rows]) if rows else 0.,
        target_pair_error_mean_regret_pct=pair_loss / len(pair_errors) if pair_errors else 0.,
        selector_mean_actual_regret_pct=loss / len(rows) if rows else 0.,
        selected_problem_regret_share_pct=100 * loss / total_loss if total_loss else 0.,
        target_pair_regret_share_pct=100 * pair_loss / all_pair_loss if all_pair_loss else 0.,
        macro_regret_contribution_pp=macro_contribution,
        all18_regret_share_pct=100 * macro_contribution / overall_regret,
        mean_strict_subset_flip_pct=100 * np.mean([r['subset_strict_flip_fraction'] for r in rows]) if rows else 0.)


def analyze(root=ROOT):
    baseline = json.loads((root / 'frozen_baseline.json').read_text())
    plan = json.loads((root / 'audit_plan.json').read_text())
    protocol = json.loads((root / 'protocol.json').read_text())
    replay = json.loads((root / 'replay_receipt.json').read_text())
    problem = plan['problem']
    all_errors = read_csv(root / 'validation_instance_errors.csv')
    scoped = {int(r['index']): r for r in all_errors if r['problem'] == problem}
    settings = protocol['mechanism']
    repeats = settings['subset_repeats']
    rows = []
    with np.load(root / 'start_costs.npz', allow_pickle=False) as cache:
        if str(cache['split']) != 'val' or str(cache['problem']) != problem or list(cache['solver_names']) != list(PAIR):
            raise ValueError('Unexpected start-cache identity')
        indices = cache['indices']
        if not np.array_equal(indices, plan['indices']) or len(set(indices)) != len(indices):
            raise ValueError('Incomplete or reordered mechanism sample')
        counts = cache['valid_start_mask'].sum(1)
        retained_masks = np.zeros((len(indices), repeats, cache['costs'].shape[-1]), dtype=bool)
        subset_winners = np.empty((len(indices), repeats), dtype=np.int64)
        for i, index in enumerate(indices):
            valid = cache['valid_start_mask'][i]
            ids = cache['start_ids'][i, valid]
            if len(set(ids)) != len(ids) or (ids <= 0).any():
                raise ValueError('Invalid native starting customer IDs')
            values = cache['costs'][i][:, valid]
            d = start_diagnostics(values, index, settings['subset_seed'], repeats, settings['retain_fraction'])
            historical = cache['historical_costs'][i]
            old_winner = int(winner_of(historical))
            b = scoped[int(index)]
            winner = d['min_winner']
            row = dict(problem=problem, index=int(index), customers=int(cache['customer_counts'][i]),
                starts=int(counts[i]), historical_moel_cost=float(historical[0]),
                historical_mtl_cost=float(historical[1]), historical_pair_winner=old_winner,
                replay_moel_min=float(d['min_costs'][0]), replay_mtl_min=float(d['min_costs'][1]),
                replay_pair_winner=winner, historical_pair_winner_reproduced=old_winner == winner,
                pair_gap_pct=100 * abs(float(historical[0] - historical[1])) / float(min(historical)),
                winner_support_count=d['support_count'], winner_support_fraction=d['support_fraction'],
                sparse_support=winner >= 0 and d['support_count'] <= settings['sparse_support_max_starts'],
                retained_starts=d['retained_count'], subset_strict_flip_count=d['strict_flip_count'],
                subset_tie_count=d['subset_tie_count'], subset_strict_flip_fraction=d['strict_flip_fraction'],
                sensitive=d['sensitive'], selector_selected=b['selected'], selector_native_winner=b['winner'],
                selector_winner_rank=int(b['winner_rank']), selector_wrong=b['category'] != 'correct',
                selector_target_pair_error=b['category'] != 'correct' and b['selected'] in PAIR and b['winner'] in PAIR,
                selector_actual_regret_pct=100 * float(b['regret']))
            for q, label in enumerate(('q10', 'q25', 'median')):
                qw = int(d['quantile_winners'][q])
                row.update({f'moel_{label}_cost': float(d['quantile_costs'][q, 0]),
                            f'mtl_{label}_cost': float(d['quantile_costs'][q, 1]),
                            f'{label}_winner': qw,
                            f'{label}_opposes_min': winner >= 0 and qw >= 0 and qw != winner})
            rows.append(row)
            retained_masks[i, :, valid] = d['retained_masks'].T
            subset_winners[i] = d['subset_winners']
    rows.sort(key=lambda r: r['index'])
    overall_regret = baseline['metrics']['macro_actual_regret_pct']
    predicates = dict(all=lambda r: True,
        strict_pair_winner=lambda r: r['replay_pair_winner'] >= 0,
        exactly_one_supporting_start=lambda r: r['replay_pair_winner'] >= 0 and r['winner_support_count'] == 1,
        at_most_two_supporting_starts=lambda r: r['sparse_support'],
        more_than_two_supporting_starts=lambda r: r['replay_pair_winner'] >= 0 and not r['sparse_support'],
        subset_sensitive=lambda r: r['sensitive'],
        sparse_or_sensitive=lambda r: r['sparse_support'] or r['sensitive'],
        stable_more_than_two=lambda r: r['replay_pair_winner'] >= 0 and not r['sparse_support'] and not r['sensitive'],
        median_opposes_min=lambda r: r['median_opposes_min'],
        q10_opposes_min=lambda r: r['q10_opposes_min'],
        q25_opposes_min=lambda r: r['q25_opposes_min'],
        support_fraction_le5pct=lambda r: r['replay_pair_winner'] >= 0 and r['winner_support_fraction'] <= .05,
        support_fraction_5to10pct=lambda r: .05 < r['winner_support_fraction'] <= .10,
        support_fraction_10to25pct=lambda r: .10 < r['winner_support_fraction'] <= .25,
        support_fraction_gt25pct=lambda r: r['winner_support_fraction'] > .25)
    summaries = [summarize_group([r for r in rows if predicate(r)], rows, label, overall_regret)
                 for label, predicate in predicates.items()]
    write_csv(root / 'winner_mechanism.csv', rows)
    write_csv(root / 'winner_mechanism_summary.csv', summaries)
    np.savez_compressed(root / 'start_subset_diagnostics.npz', indices=indices,
        subset_winners=subset_winners, retained_start_mask=retained_masks,
        seed=np.array(settings['subset_seed']), repeats=np.array(repeats))
    strict = [r for r in rows if r['replay_pair_winner'] >= 0]
    totals = dict(instances=len(rows), strict_pair_winners=len(strict),
        historical_pair_ties=sum(r['historical_pair_winner'] < 0 for r in rows),
        replay_pair_ties=len(rows) - len(strict),
        historical_pair_winner_disagreements=sum(not r['historical_pair_winner_reproduced'] for r in rows),
        winner_support_count_quantiles=np.quantile([r['winner_support_count'] for r in strict], [0, .25, .5, .75, 1]).tolist(),
        winner_support_fraction_quantiles=np.quantile([r['winner_support_fraction'] for r in strict], [0, .25, .5, .75, 1]).tolist(),
        median_opposition_pct=100 * np.mean([r['median_opposes_min'] for r in strict]),
        q10_opposition_pct=100 * np.mean([r['q10_opposes_min'] for r in strict]),
        q25_opposition_pct=100 * np.mean([r['q25_opposes_min'] for r in strict]),
        any_subset_flip_instance_pct=100 * np.mean([r['sensitive'] for r in strict]),
        mean_strict_subset_flip_pct=100 * np.mean([r['subset_strict_flip_fraction'] for r in strict]),
        any_subset_tie_instances=sum(r['subset_tie_count'] > 0 for r in rows))
    dump(root / 'mechanism_metrics.json', dict(totals=totals, groups=summaries,
        analysis_unit='instance, not start', winner_encoding={'0': PAIR[0], '1': PAIR[1], '-1': 'exact tie'},
        subset_run_kind='20 offline subsets of the same full trajectories, not repeated solving'))
    render_report(root, baseline, plan, replay, totals, summaries)
    figures(root, rows, summaries)
    print(json.dumps(dict(totals=totals, groups=summaries), indent=2), flush=True)
    return rows, summaries


def render_report(root, baseline, plan, replay, totals, summaries):
    decomposition = [r for r in read_csv(root / 'topk_error_decomposition.csv') if r['scope'] == 'ALL']
    pairs = read_csv(root / 'ranked_problem_pairs.csv')
    global_pairs = read_csv(root / 'ranked_global_pairs.csv')
    bounds = read_csv(root / 'correction_bounds.csv')
    for collection in (decomposition, pairs, global_pairs, bounds):
        for row in collection:
            for key in row:
                try:
                    row[key] = float(row[key])
                    if key in ('count', 'number_of_pairs', 'corrected_errors'):
                        row[key] = int(row[key])
                except ValueError:
                    pass
    group = {r['group']: r for r in summaries}
    moel_mtl = next(r for r in global_pairs if (r['solver_a'], r['solver_b']) == tuple(PAIR))
    ovrptw = next(r for r in pairs if r['problem'] == 'OVRPTW' and (r['solver_a'], r['solver_b']) == tuple(PAIR))
    problem_bounds = [r for r in bounds if r['pair_scope'].startswith('problem') and r['number_of_pairs'] <= 10]
    global_bound = next(r for r in bounds if r['pair_scope'] == 'solver_a+solver_b' and r['number_of_pairs'] == 1)
    metrics = baseline['metrics']
    sparse = group['at_most_two_supporting_starts']
    sensitive = group['subset_sensitive']
    union = group['sparse_or_sensitive']
    stable = group['stable_more_than_two']
    greater_ten = group['support_fraction_10to25pct']['target_pair_regret_share_pct'] + group['support_fraction_gt25pct']['target_pair_regret_share_pct']
    text = f'''# R52: validation error sources and multi-start winner mechanism

## Main findings

No selector was trained. No test data or test predictions were read. Fixed seed2.
R45A validation-best epoch{baseline['human_epoch']} is the unchanged baseline: Top1 {100 * metrics['macro_top1']:.4f}%, actual regret {metrics['macro_actual_regret_pct']:.6f}%.

- Head-to-head ranking errors with the native winner at predicted rank2 account for {decomposition[0]['share_of_errors_pct']:.2f}% of errors and {decomposition[0]['share_of_total_regret_pct']:.2f}% of regret. Winners outside predicted Top3 account for only {decomposition[2]['share_of_total_regret_pct']:.2f}% of regret. Full-pool candidate identification remains relevant but is not the largest aggregate loss source.
- Across all problems, MOEL/MTL reversals account for {moel_mtl['count']} errors and {moel_mtl['share_of_total_regret_pct']:.2f}% of regret. OVRPTW alone explains {ovrptw['share_of_total_regret_pct']:.2f}%. The pair is important, but OVRPTW alone is not representative of its full cost budget.
- The highest-contribution specific problem/pair is {plan['problem']} / MOEL vs MTL: {pairs[0]['count']} errors, {pairs[0]['share_of_total_regret_pct']:.2f}% of total regret. Both methods were replayed on all1000 existing validation instances, not selected errors.
- At most two winning starts: {sparse['instances']}/1000 instances; these account for {sparse['target_pair_regret_share_pct']:.2f}% of the selected problem's MOEL/MTL error regret. At least one strict reversal in20 shared80%-start subsets: {sensitive['instances']}/1000, accounting for {sensitive['target_pair_regret_share_pct']:.2f}% of that pair's error regret. Their union accounts for {union['target_pair_regret_share_pct']:.2f}%; the complementary stable, more-than-two-supported group still accounts for {stable['target_pair_regret_share_pct']:.2f}%. Thus these particular extreme-start/sensitivity mechanisms are not the dominant measured cost explanation.

## 1. Full18-task error budget

Ranks and correctness use native ind. Costs are independently reread from raw_label.pkl in FP64 and match saved validation predictions exactly. Scores use greedy argmax, ties by global solver ID. Every problem contributes1/18, every instance within it1/N_problem. Percentages below are regret shares, not shares of summed raw route cost.

{table(decomposition, ['category', 'count', 'share_of_errors_pct', 'macro_regret_contribution_pp', 'share_of_total_regret_pct'], ['Error category', 'Errors', 'Error share %', 'Macro regret pp', 'Regret share %'])}

### Largest problem-specific unordered pairs

Opposite directions of the same pair are combined here; `error_budget_by_problem_pair.csv` retains the full problem x selected x native-winner directions. A row counts only actual selector errors, not every binary contest between those methods.

{table(pairs[:10], ['problem', 'solver_a', 'solver_b', 'count', 'share_of_total_regret_pct'], ['Problem', 'Solver A', 'Solver B', 'Errors', 'Regret share %'])}

### Diagnostic correction upper bounds (NOT deployable results)

If an oracle corrected just the existing errors in the largest problem-specific pairs to native winner, leaving all other decisions unchanged:

{table(problem_bounds, ['number_of_pairs', 'corrected_errors', 'diagnostic_macro_top1_pct', 'diagnostic_macro_actual_regret_pct', 'recovered_regret_share_pct'], ['Pairs', 'Corrected', 'Diagnostic Top1 %', 'Diagnostic regret %', 'Recovered loss %'])}

Correcting every MOEL/MTL reversal across all tasks would add {moel_mtl['macro_top1_recovery_pp']:.4f}pp Top1: diagnostic Top1 {global_bound['diagnostic_macro_top1_pct']:.4f}% and regret {global_bound['diagnostic_macro_actual_regret_pct']:.6f}%. This uses true winners offline, not an achieved model score or an accuracy ceiling. Native-winner-vs-FP64-min checks are in `baseline_by_problem.csv`; no winner relabeling was performed.

## 2. Full-trajectory audit on {plan['problem']}

Selection is ranking-first, not automatically OVRPTW. The highest-ranked pair itself has the original no-augmentation export and recoverable bridge recipe, so no lower-ranked fallback or substituted default was needed. Sampling is size-only, seed2, covers all1000 original validation instances with50..100 customers. Eight size-stratified instances were checked before the full replay.

The original bridge supplies FP32 depot/coordinates/signed demand/route limit/service time/customer time windows, normalized capacity1, no augmentation, one argmax construction, native environment dispatch. Both R48-pinned epoch5000 Train_ALL weights and strict architecture are reused; SHA256 bindings are in `audit_plan.json`. Historical complete CLI and source/environment revision are not archived: matching exports, explicit bridge parameters and successful present replay do not manufacture missing historical metadata.

Important native detail: declared pomo_size=N is capped by OVRPBLTWEnv to floor(0.8*N), using positive-demand customer IDs in original order. This audit preserves ALL native enabled starts ({40}..{80}), not all customers including invalid backhaul starts. Both methods' actual start IDs match exactly. No start or trajectory is chosen using final cost before solving. Open-route cost excludes every return-to-depot segment, exactly as the original environment.

Every complete trajectory cost is saved in `start_costs.npz` with original instance index, customer count, start customer ID, valid-start mask and historical pair costs. Padding is NaN and excluded from all reductions. Native reward computation is FP32; saving those values as FP64 does not recover precision. Historical label costs and all diagnostic arithmetic are FP64.

- Historical scalar reproduction: all2000 full-run costs pass atol{5e-5:g}, rtol{2e-6:g}; maximum absolute difference, including precheck, {replay['max_abs_history_difference']:.12g}. Exact pair-winner disagreements: {totals['historical_pair_winner_disagreements']}. Historical ties: {totals['historical_pair_ties']}; replay ties: {totals['replay_pair_ties']}.
- Actual additional solving: {replay['completed_solver_runs']} solver-instance runs including8-instance precheck for each method, full replay {replay['full_replay_seconds']:.3f}s, total replay stage {replay['total_seconds']:.3f}s, peak allocated GPU {replay['peak_allocated_gpu_mib']:.1f}MiB on only {replay['device']} ({replay['gpu_uuid']}). This is not zero-solve analysis.
- Number of starts beating the opponent's best (strict pair wins only): min/q25/median/q75/max = {totals['winner_support_count_quantiles']}. Support fractions = {totals['winner_support_fraction_quantiles']}.
- Min-winner opposite to median winner: {totals['median_opposition_pct']:.2f}%; opposite to q10 winner: {totals['q10_opposition_pct']:.2f}%; opposite to q25 winner: {totals['q25_opposition_pct']:.2f}%. Exact ties are separate, not counted as opposition.
- Offline subset sensitivity: same retained starting-customer IDs for both methods, ceil(0.8*actual_start_count),20 subsets per instance, seed2 with per-index streams. {totals['any_subset_flip_instance_pct']:.2f}% of strict-win instances have at least one strict reversal; mean reversal frequency {totals['mean_strict_subset_flip_pct']:.2f}%. {totals['any_subset_tie_instances']} instances have a subset tie. This changes only the reduction over previously computed full costs, NOT POMO execution size, decoding, or20 new solver runs.

### Association with actual selector loss

Groups below overlap; do not add their percentages. `Target-pair loss %` is the share of{plan['selection']['count']} actual MOEL/MTL selector-error losses in this problem. `Problem loss %` uses all seven-method-pool selector decisions. `ALL loss %` is the contribution to the original18-task macro loss. Neither support counts nor subset repetitions are independent data samples. The final four support-fraction bins are disjoint descriptive summaries, not additional prespecified hypothesis tests.

{table(summaries, ['group', 'instances', 'pool_errors', 'target_pair_errors', 'target_pair_regret_share_pct', 'selected_problem_regret_share_pct', 'all18_regret_share_pct'], ['Group', 'Instances', 'Pool errors', 'Pair errors', 'Target-pair loss %', 'Problem loss %', 'ALL loss %'])}

## Interpretation and next direction

The error budget supports focusing on competitive head-to-head decisions across MVRP, particularly MOEL/MTL, rather than another architecture search limited to OVRPTW. It does not support attributing the platform mainly to winners absent from Top3.

The replay supplies direct evidence for the distinction between average trajectory quality and best-of-many quality: the measured median/q10 disagreements above cannot be inferred from mean prefix features. However, this distinction does NOT explain most of the measured selection loss. The stable, more-than-two-supported share is {stable['target_pair_regret_share_pct']:.2f}%; more than10% of starts beat the opponent's best on instances contributing {greater_ten:.2f}% of pair-error regret. More than two does not automatically mean broad support, so the full counts/fractions are reported rather than equated with an easy decision.

The first priority supported here is transferable instance-method representation/supervision for the high-loss MVRP pairs, including their stable supported wins. Do not automatically escalate to a larger trajectory predictor on the claim that isolated exceptional starts cause most remaining cost. Per-start modeling may still be tested as a different information source, but it would need its own controlled evidence; neither another pooled prefix vector nor longer probing is justified by this audit alone. The two original questions are answered: the dominant pair is real at the full-system level, whereas this selected pair's loss is not primarily located in the prespecified rare/sensitive groups.

No label error, randomness ceiling, or global predictability limit is established. A winner change after removing starts is not a corrupted label. The original complete-start minimum remains the target. This is one problem, two methods,1000 instances, not a claim about every solver's mechanism. No training, relabeling, seed search, or test-based selection was done.

Paper context: [Leader Reward for POMO-Based Neural Combinatorial Optimization, section4.1](https://arxiv.org/html/2405.13947v1) distinguishes average generated quality from the best generated solution. R52 measures this distinction in existing ReLD outputs; it does not implement that paper's training method or assume its findings prove selector difficulty.

## Reproduction and files

Run from `/public/home/shiys/0selection2` in easynco:

```bash
bash code/V4/run_v4_r52.sh budget
R52_GPU_UUID={replay['gpu_uuid']} bash code/V4/run_v4_r52.sh replay
bash code/V4/run_v4_r52.sh analysis
bash code/V4/run_v4_r52.sh tests
```

Replay must run on the GPU node inside the existing allocation; the recorded srun/tmux command is in `execution.json`. Source scripts are `r52_error_budget.py`, `r52_start_audit.py`, `r52_analysis.py`, `test_r52.py`, `run_v4_r52.sh`. Neither old solver code nor historical experiment outputs were edited.

Required deliverables: `error_budget_by_problem_pair.csv`, `topk_error_decomposition.csv`, `start_costs.npz`, `winner_mechanism.csv`, this report. Supplemental pair rankings, correction bounds, per-instance errors, historical replay CSV, frozen checkpoint/recipe hashes, start subset masks/winners and mechanism summaries allow independent recalculation without loading a selector or rerunning solvers.
'''
    (root / 'RESULTS.md').write_text(text)
    dump(root / 'analysis_receipt.json', dict(
        start_costs_sha256=file_hash(root / 'start_costs.npz'),
        frozen_baseline_sha256=file_hash(root / 'frozen_baseline.json'),
        source_sha256=file_hash(__file__), test_read=False, selector_training=False))


def figures(root, rows, summaries):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    strict = [r for r in rows if r['replay_pair_winner'] >= 0]
    axes[0].hist([r['winner_support_count'] for r in strict], bins=np.arange(0, 82, 2), color='#28756d')
    axes[0].set(xlabel='Starts beating opponent best', ylabel='Instances', title='Winning-trajectory support')
    axes[1].hist([r['subset_strict_flip_count'] / 20 for r in strict], bins=np.linspace(0, 1, 21), color='#ab523b')
    axes[1].set(xlabel='Strict reversal fraction over 20 subsets', ylabel='Instances', title='Shared 80%-start subset sensitivity')
    fig.savefig(root / 'winner_mechanism.png', dpi=160)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT)
    analyze(parser.parse_args().output)

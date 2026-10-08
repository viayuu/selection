"""R44 validation screen, paired decision changes, replay, and result tables."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from code.unified_selector.registry import PROBLEMS
from .multitask_probe import dump
from .pairwise_selector import maximin_logits
from .performance_analysis import write_csv
from .r43_analysis import plot_methods, plotting
from .r43_experiment import policy_metrics
from .r44_experiment import ROOT, MODES, PRIMARY, SEEDS, run_directory


FIELDS = ('ce', 'top1', 'top2', 'top3', 'mean_cost', 'vs_sbs_pct', 'vs_oracle_pct', 'actual_regret_pct')


def read(path):
    return json.loads(path.read_text())


def plot_run(directory):
    plt = plotting()
    history = read(directory / 'history.json')
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, field in zip(axes.flat, ('top1', 'ce', 'mean_cost', 'actual_regret_pct')):
        for split in ('train', 'val'):
            rows = [row for row in history if row[split] is not None]
            ax.plot([row['successful_updates'] for row in rows], [row[split]['macro']['macro_'+field] for row in rows], marker='.', label=split)
        ax.set(title=field, xlabel='Successful optimizer updates')
        ax.grid(alpha=.2)
        ax.legend()
    fig.tight_layout()
    fig.savefig(directory / 'train_val_curves.png', dpi=150)
    plt.close(fig)
    rows = history[1:]
    if rows:
        fig, axes = plt.subplots(2, 3, figsize=(14, 8))
        for ax, fields in zip(axes.flat, (('loss',), ('ce', 'cmp'), ('true3', 'pred3', 'all_pairs'), ('risk',), ('kl',), ('base_lr', 'adapter_lr'))):
            for field in fields:
                ax.plot([row['successful_updates'] for row in rows], [row['optimization'][field] for row in rows], label=field)
            ax.set(title=' / '.join(fields), xlabel='Successful optimizer updates')
            ax.grid(alpha=.2)
            ax.legend()
        fig.tight_layout()
        fig.savefig(directory / 'optimization_curves.png', dpi=150)
        plt.close(fig)


def screen(root):
    manifest = read(root / 'manifest.json')
    runs = {arm: read(run_directory(root, arm, 2) / 'result.json') for arm in MODES}
    histories = {arm: read(run_directory(root, arm, 2) / 'history.json') for arm in MODES}
    common = sorted(set.intersection(*[{row['successful_updates'] for row in values if row['successful_updates'] > 0} for values in histories.values()]))[-5:]
    if not common:
        raise ValueError('The screen requires completed nonzero validation points')
    best = {arm: value['best']['val']['macro'][PRIMARY] for arm, value in runs.items()}
    means = {arm: float(np.mean([row['val']['macro'][PRIMARY] for row in values if row['successful_updates'] in common])) for arm, values in histories.items()}
    threshold = manifest['protocol']['delta_screen']
    gaps = {arm: best[arm]-best['C2'] for arm in ('C0', 'C1')}
    mean_gaps = {arm: means[arm]-means['C2'] for arm in ('C0', 'C1')}
    passed = all(gaps[arm] > 0 and gaps[arm] >= threshold and mean_gaps[arm] > 0 for arm in gaps)
    value = dict(passed=passed, primary=PRIMARY, delta_screen=threshold, units='percentage points; lower is better',
                 best=best, last_common_steps=common, mean_last_common=means, best_improvement=gaps, mean_improvement=mean_gaps,
                 action='Run locked seeds 17/42, then test locked best checkpoints' if passed else 'End R44 with validation-only negative/mixed results; no test, extra seeds or search',
                 operational_screen_not_significance=True, test_read=False)
    path = root / 'stage_a_screen.json'
    if path.exists() and read(path) != value:
        raise ValueError('The completed stage-A screen changed')
    dump(path, value)
    print('[stage A] ' + json.dumps(value), flush=True)
    return value


def metric_rows(name, selection, value, step):
    rows = [dict(run=name, selection=selection, update=step, problem='ALL', n=value['instances'],
                 **{field: value['macro']['macro_'+field] for field in FIELDS},
                 sbs_cost=float(np.mean([r['sbs_cost'] for r in value['per_problem'].values()])),
                 oracle_cost=float(np.mean([r['oracle_cost'] for r in value['per_problem'].values()])))]
    rows += [dict(run=name, selection=selection, update=step, problem=p, n=r['n'], **{field: r[field] for field in FIELDS},
                  sbs_cost=r['sbs_cost'], oracle_cost=r['oracle_cost']) for p, r in value['per_problem'].items()]
    return rows


def table(rows):
    lines = ['| Run | Point | Update | Top1 | Top2 | Top3 | mean_cost | vs_SBS | actual regret |',
             '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        lines.append(f'| {row["run"]} | {row["selection"]} | {row["update"]} | {row["top1"]:.4f} | {row["top2"]:.4f} | '
                     f'{row["top3"]:.4f} | {row["mean_cost"]:.6f} | {row["vs_sbs_pct"]:+.6f}% | {row["actual_regret_pct"]:.6f}% |')
    return lines


def replay(path, expected):
    checks = []
    for p in PROBLEMS:
        with np.load(path / f'{p}.npz') as saved:
            utility = torch.from_numpy(saved['utility'])
            mask = torch.from_numpy(saved['solver_mask'])
            margins = utility[:, :, None]-utility[:, None, :]
            margins = margins.masked_fill(~(mask[:, :, None] & mask[:, None, :]), 0.)
            np.testing.assert_array_equal(margins.numpy(), saved['pair_margin'])
            np.testing.assert_array_equal(maximin_logits(margins, mask).numpy(), saved['logits'])
            raw = dict(costs=saved['costs'], winner=saved['winner'], pool_ids=saved['pool_ids'].tolist(), pool=saved['pool'].tolist())
            metrics, pred = policy_metrics(saved['logits'], raw)
            np.testing.assert_array_equal(pred, saved['pred'])
            error = max(abs(metrics[key]-expected['per_problem'][p][key]) for key in FIELDS)
            if error > 1e-10:
                raise ValueError('Saved utilities/maximin predictions do not replay reported metrics')
            checks.append(dict(problem=p, count=len(pred), metric_error=error))
    return checks


def paired_changes(root, seed, control, selection, steps):
    rows = []
    old_path = run_directory(root, control, seed) / 'val_predictions' / f'u{steps[control]:06d}'
    new_path = run_directory(root, 'C2', seed) / 'val_predictions' / f'u{steps["C2"]:06d}'
    for p in PROBLEMS:
        with np.load(old_path / f'{p}.npz') as old, np.load(new_path / f'{p}.npz') as new:
            for key in ('indices', 'costs', 'winner', 'nodes', 'pool_ids', 'size_bucket'):
                np.testing.assert_array_equal(old[key], new[key])
            before, after, winner, costs = old['pred'], new['pred'], old['winner'], old['costs']
            idx = np.arange(len(winner))
            difference = costs[idx, after]-costs[idx, before]
            relative = difference/costs.min(-1)*100.
            both_wrong = (before != winner) & (after != winner)
            categories = dict(corrected=(before != winner) & (after == winner), harmed=(before == winner) & (after != winner),
                              both_wrong_cost_down=both_wrong & (difference < 0), both_wrong_cost_up=both_wrong & (difference > 0),
                              unchanged_or_equal=(before == after) | (both_wrong & (difference == 0)), all=np.ones(len(winner), dtype=bool))
            for size in ('ALL', *[str(j) for j in np.unique(old['size_bucket'])]):
                size_mask = np.ones(len(winner), dtype=bool) if size == 'ALL' else old['size_bucket'] == int(size)
                for category, valid in categories.items():
                    valid = valid & size_mask
                    rows.append(dict(seed=seed, control=control, selection=selection, problem=p, size_bucket=size, category=category,
                                     count=int(valid.sum()), cost_change_sum=float(difference[valid].sum()), regret_change_sum_pp=float(relative[valid].sum()),
                                     net_mean_cost_change=float(difference[valid].sum()/size_mask.sum()), net_actual_regret_change_pp=float(relative[valid].sum()/size_mask.sum())))
    return rows


def plot_comparison(root, seeds):
    plt = plotting()
    fig, axes = plt.subplots(len(seeds), 4, figsize=(18, 4*len(seeds)), squeeze=False)
    baseline = read(root / 'baseline_eval.json')['macro']
    for row, seed in zip(axes, seeds):
        for arm in MODES:
            history = read(run_directory(root, arm, seed) / 'history.json')
            for ax, field in zip(row, ('top1', 'ce', 'vs_sbs_pct', 'actual_regret_pct')):
                ax.plot([r['successful_updates'] for r in history], [r['val']['macro']['macro_'+field] for r in history], marker='.', label=arm)
        for ax, field in zip(row, ('top1', 'ce', 'vs_sbs_pct', 'actual_regret_pct')):
            ax.axhline(baseline['macro_'+field], linestyle=':', color='black', label='Fixed R43A (3090 replay)')
            ax.set(title=f'seed{seed}: {field}', xlabel='Successful optimizer updates')
            ax.legend(fontsize=8)
            ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(root / 'comparison_curves.png', dpi=160)
    plt.close(fig)


def summarize(root):
    manifest = read(root / 'manifest.json')
    outcome = read(root / 'stage_a_screen.json')
    seeds = [seed for seed in SEEDS if all((run_directory(root, arm, seed) / 'result.json').exists() for arm in MODES)]
    if not seeds:
        raise ValueError('No complete matched three-arm comparison')
    rows, group_rows, size_rows, changes, gates, resources, replays = [], [], [], [], [], [], []
    last_rows = []
    for seed in seeds:
        runs = {arm: read(run_directory(root, arm, seed) / 'result.json') for arm in MODES}
        for arm, run in runs.items():
            directory = run_directory(root, arm, seed)
            history = read(directory / 'history.json')
            for label, point in (('step0', history[0]), ('best', run['best']), ('final', run['final'])):
                rows.extend(metric_rows(directory.name, label, point['val'], point['successful_updates']))
                for family, value in point['val']['families'].items():
                    group_rows.append(dict(run=directory.name, selection=label, problem=family, update=point['successful_updates'],
                                           **{field: value['macro_'+field] for field in FIELDS}))
                for p, bins in point['val']['size_groups'].items():
                    for index, value in bins.items():
                        size_rows.append(dict(run=directory.name, selection=label, problem=p, size_bucket=index, count=value['count'],
                                              min_nodes=value['min_nodes'], max_nodes=value['max_nodes'], **{field: value[field] for field in FIELDS}))
                if label in ('best', 'final'):
                    path = directory / 'val_predictions' / f'u{point["successful_updates"]:06d}'
                    replays.extend([dict(run=directory.name, selection=label, **row) for row in replay(path, point['val'])])
                for p, value in point['val']['gates'].items():
                    for size, info in (('ALL', value['overall']), *value['by_size'].items()):
                        gates.append(dict(run=directory.name, selection=label, problem=p, size_bucket=size,
                                          **{k: v for k, v in info.items() if k != 'channel_mean'}))
            last_rows.append(dict(run=directory.name, **{field: run['last5_val']['macro_'+field] for field in FIELDS}))
            parameters = manifest['parameters'][arm]
            resources.append(dict(run=directory.name, original_parameters=parameters['original'], added_parameters=parameters['added'],
                                  successful_updates=run['successful_updates'], attempted_updates=run['attempted_updates'], amp_skipped=run['amp_skipped'],
                                  presentations=sum(run['samples_by_problem'].values()), wall_seconds=run['wall_seconds'], optimization_seconds=run['optimization_seconds'],
                                  peak_memory_gib=run['peak_memory_gib'], gpu=run['gpu_name'], best_update=run['best']['successful_updates'],
                                  best_val_inference_seconds=run['best']['val']['inference_seconds']))
            plot_methods(directory.name, run['best']['val'], directory / 'best_val_method_figures')
        for selection in ('best', 'final'):
            steps = {arm: run[selection]['successful_updates'] for arm, run in runs.items()}
            for control in ('C0', 'C1'):
                changes.extend(paired_changes(root, seed, control, selection, steps))
    write_csv(root / 'summary.csv', rows)
    write_csv(root / 'family_summary.csv', group_rows)
    write_csv(root / 'size_summary.csv', size_rows)
    write_csv(root / 'last5.csv', last_rows)
    write_csv(root / 'decision_changes.csv', changes)
    write_csv(root / 'gate_summary.csv', gates)
    write_csv(root / 'resource_summary.csv', resources)
    dump(root / 'prediction_replay.json', replays)
    plot_comparison(root, seeds)
    lines = ['# R44: Constraint/size-conditioned node residual', '',
             '## Experiment', '',
             f'- Baseline: R43A epoch {manifest["baseline"]["epoch"]}, SHA256 `{manifest["baseline"]["sha256"]}`.',
             '- One insertion: after the solver-independent InstanceEncoder and before the first node/solver joint block.',
             '- C0: continuation only. C1: shared bottleneck. C2: existing 16 semantic fields plus train-normalized log(real nodes).',
             '- 34 solver features, local geometry, maximin scores, native labels, FP64 cost targets, comparison loss and R-Drop are unchanged.',
             '- All original parameters trainable; fresh AdamW; original/new peak LR 1e-5/1e-4; 5% warmup then cosine to 10%.',
             f'- Each arm: {manifest["protocol"]["successful_update_budget"]:,} successful updates, 10 complete data epochs, batch128, full tails retained.',
             '- Borrowed from PEPNet: two-layer 2*sigmoid multiplicative gate. GELU, metadata-only gate, zero-initialized bottleneck residual are project adaptations, not full PEPNet.',
             '- The 3090 step0 differs from historical 4090 predictions on four near-boundary instances; data/state are identical and every R44 arm shares the same local step0. See baseline_replay.json.',
             '', '## Validation Summary', '', *table([row for row in rows if row['problem'] == 'ALL']), '',
             'Last five nonzero common validation points:', '',
             '| Run | Top1 | CE | mean_cost | vs_SBS | actual regret |', '|---|---:|---:|---:|---:|---:|']
    for row in last_rows:
        lines.append(f'| {row["run"]} | {row["top1"]:.4f} | {row["ce"]:.6f} | {row["mean_cost"]:.6f} | {row["vs_sbs_pct"]:+.6f}% | {row["actual_regret_pct"]:.6f}% |')
    lines += ['', '## Stage-A Decision', '', f'Passed: **{outcome["passed"]}**. Locked screen threshold: {outcome["delta_screen"]:.3f} percentage points (operational, not statistical significance).']
    for control in ('C0', 'C1'):
        lines.append(f'- C2 vs {control}: best cost improvement {outcome["best_improvement"][control]:+.6f} pp; last-five mean improvement {outcome["mean_improvement"][control]:+.6f} pp. Positive means C2 is better.')
    lines += ['', '## Attribution', '', '| Seed | Point | Control | Corrected | Harmed | Both wrong, cost down | Both wrong, cost up | Net mean cost change | Net actual regret change |',
              '|---:|---|---|---:|---:|---:|---:|---:|---:|']
    for seed in seeds:
        for selection in ('best', 'final'):
            for control in ('C0', 'C1'):
                subset = [r for r in changes if r['seed'] == seed and r['selection'] == selection and r['control'] == control and r['size_bucket'] == 'ALL']
                counts = {key: sum(r['count'] for r in subset if r['category'] == key) for key in ('corrected', 'harmed', 'both_wrong_cost_down', 'both_wrong_cost_up')}
                all_rows = [r for r in subset if r['category'] == 'all']
                cost = np.mean([r['net_mean_cost_change'] for r in all_rows])
                regret = np.mean([r['net_actual_regret_change_pp'] for r in all_rows])
                lines.append(f'| {seed} | {selection} | {control} | {counts["corrected"]} | {counts["harmed"]} | {counts["both_wrong_cost_down"]} | {counts["both_wrong_cost_up"]} | {cost:+.8f} | {regret:+.6f} pp |')
    lines += ['', 'Negative cost/regret changes favor C2. Native-winner corrections alone do not explain all cost changes.', '', '## Per-Problem Best Validation', '',
              '| Run | Problem | Top1 | Top2 | Top3 | mean_cost | SBS cost (vs_SBS) | Oracle cost (vs_Oracle) | actual regret |',
              '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        if row['selection'] == 'best':
            lines.append(f'| {row["run"]} | {row["problem"]} | {row["top1"]:.4f} | {row["top2"]:.4f} | {row["top3"]:.4f} | {row["mean_cost"]:.6f} | '
                         f'{row["sbs_cost"]:.6f} ({row["vs_sbs_pct"]:+.5f}%) | {row["oracle_cost"]:.6f} ({row["vs_oracle_pct"]:+.5f}%) | {row["actual_regret_pct"]:.5f}% |')
    lines += ['', '## Gate and Scope', '',
              'Gate statistics are in gate_summary.csv and each history.json. Same semantic descriptor and node count imply the same gate; this is expected. Train-only normalization/bins and exact fields are in manifest.json.',
              'Instance uncertainty was not estimated; seed variation is continuation randomness from one fixed R43A, not independent from-scratch training.',
              'Percentages are arithmetic macro means over problems; ALL mean_cost ratios are not used to replace those percentages. Size-group references are recomputed within each fixed group, identically for all arms.',
              'No claim that multi-domain negative transfer was proven. A negative result constrains this insertion, gate, recipe and budget, not every conditional architecture.', '']
    if not outcome['passed']:
        lines += ['## Conclusion', '', 'This conditional residual did not pass the prespecified comparison against both continued training and parameter-matched shared capacity. Retain R43A; do not keep C2 as an established improvement.',
                  'Per protocol, no test data, additional seeds, gate/rank/LR search were run. All results above are validation results.', '']
    elif (root / 'test_results.json').exists():
        tests = read(root / 'test_results.json')
        test_rows = [row for name, value in tests.items() for row in metric_rows(name, 'locked_test', value, read(root / 'locked_checkpoints.json')[name]['successful_updates'])]
        write_csv(root / 'test_summary.csv', test_rows)
        lines += ['## Locked Test', '', *table([row for row in test_rows if row['problem'] == 'ALL']), '',
                  'All models use their validation-locked checkpoint once. Test had been used historically; it is not a newly untouched confirmation set.', '']
        for control in ('C0', 'C1'):
            diff = [tests[f'{control}_seed{s}']['macro'][PRIMARY]-tests[f'C2_seed{s}']['macro'][PRIMARY] for s in seeds]
            lines.append(f'- Locked test C2 vs {control}: mean improvement {np.mean(diff):+.6f} pp; improves in {sum(d > 0 for d in diff)}/{len(diff)} continuation seeds.')
    else:
        lines += ['Stage A passed; the additional continuation seeds/test have not all completed yet. Do not claim a test gain from validation alone.', '']
    (root / 'comparison.md').write_text('\n'.join(lines)+'\n')
    print(f'[report] {root / "comparison.md"}', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--screen', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.screen:
        screen(args.root)
    summarize(args.root)


if __name__ == '__main__':
    main()

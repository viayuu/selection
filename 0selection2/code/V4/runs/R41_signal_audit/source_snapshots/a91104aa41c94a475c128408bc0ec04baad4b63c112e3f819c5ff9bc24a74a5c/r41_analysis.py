"""Frozen test tables and complete-pool solver-repeat diagnostics for R41."""

import argparse
import json
from pathlib import Path

import numpy as np

from code.unified_selector.registry import PROBLEMS
from .multitask_probe import dump
from .multitask_probe import file_hash
from .performance_analysis import write_csv
from .performance_targets import read_raw_costs
from .r40_experiment import classification_metrics


ROOT = Path('code/V4/runs/R41_signal_audit')
GAP_LABELS = ('<=0.1%', '(0.1%,0.5%]', '>0.5%')


def test_tables(root):
    results = json.loads((root / 'test_results.json').read_text())
    locked = json.loads((root / 'frozen_checkpoints.json').read_text())
    lines = ['# R41 frozen test comparison', '',
             'All four checkpoints were selected before R41 using validation cost. No training or test-based reselection.',
             'Native `ind` is the Top1 target; original raw costs are aggregated in FP64. No LIB datasets in this round.',
             'ALL percentages are the mean of per-problem percentages, not ratios of ALL average costs.', '',
             '| Model | Locked epoch | Top1 | Top2 | Top3 | Mean cost | vs SBS | vs Oracle | Actual regret | CE |',
             '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    rows = []
    for name, value in results.items():
        m = value['macro']
        lines.append(f'| {name} | {locked[name]["epoch"]} | {m["macro_top1"]:.4f} | {m["macro_top2"]:.4f} | '
                     f'{m["macro_top3"]:.4f} | {m["macro_mean_cost"]:.6f} | {m["macro_vs_sbs_pct"]:+.4f}% | '
                     f'{m["macro_vs_oracle_pct"]:+.4f}% | {m["macro_actual_regret_pct"]:.4f}% | {m["macro_ce"]:.4f} |')
        per = value['per_problem']
        all_row = {key: m['macro_' + key] for key in
                   ('top1', 'top2', 'top3', 'mean_cost', 'vs_sbs_pct', 'vs_oracle_pct', 'actual_regret_pct', 'ce')}
        all_row.update(sbs_cost=float(np.mean([r['sbs_cost'] for r in per.values()])),
                       oracle_cost=float(np.mean([r['oracle_cost'] for r in per.values()])), n=sum(r['n'] for r in per.values()))
        rows.append(dict(model=name, problem='ALL', **all_row))
        for problem, r in per.items():
            rows.append(dict(model=name, problem=problem, **{key: r[key] for key in all_row}))
    for name, value in results.items():
        lines.extend(['', f'## {name}', '', f'Checkpoint: `{locked[name]["path"]}`; epoch {locked[name]["epoch"]}.', '',
             '| Problem | Top1 | Top2 | Top3 | Mean cost | SBS cost (vs SBS) | Oracle cost (vs Oracle) | Actual regret |',
             '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |'])
        for row in [r for r in rows if r['model'] == name]:
            lines.append(f'| {row["problem"]} | {row["top1"]:.4f} | {row["top2"]:.4f} | {row["top3"]:.4f} | '
                         f'{row["mean_cost"]:.6f} | {row["sbs_cost"]:.6f} ({row["vs_sbs_pct"]:+.4f}%) | '
                         f'{row["oracle_cost"]:.6f} ({row["vs_oracle_pct"]:+.4f}%) | {row["actual_regret_pct"]:.4f}% |')
        lines.extend(['', '### Solver pick distribution', '', '| Problem | Solver | Pick fraction | Native winner fraction |',
                      '| --- | --- | ---: | ---: |'])
        for problem, r in value['per_problem'].items():
            for solver in r['pool']:
                lines.append(f'| {problem} | {solver} | {r["arm_distribution"][solver]:.2%} | {r["method_top1"][solver]:.2%} |')
    lines.extend(['', '## Interpretation', '',
          '1. These test results do not show a substantial R40 improvement. R34 remains the strongest strict Top1 of the four frozen policies.',
          '2. Raw mean cost and macro relative cost are different aggregates. R40A has a slightly smaller raw mean cost, but not a better macro vs SBS than R34.',
          '3. Do not tune models, checkpoints, or the label audit sample using this test comparison. Original sampling remains locked.',
          '4. Validation replay uses historical evaluation batch sizes: R34/R39A 640, R40A/R40B 256. '
          'An initial batch256 R34 replay changed a few near-equal FP32 choices; that failed pre-test attempt is preserved separately.',
          '5. These are four single-seed frozen models; small differences are not evidence of a reliable architectural advantage.', ''])
    (root / 'test_comparison.md').write_text('\n'.join(lines))
    write_csv(root / 'test_comparison.csv', rows)
    arms = [dict(model=name, problem=p, solver=s, pick_fraction=r['arm_distribution'][s],
                 oracle_fraction=r['method_top1'][s]) for name, value in results.items()
            for p, r in value['per_problem'].items() for s in r['pool']]
    write_csv(root / 'test_arm_distribution.csv', arms)
    families = [dict(model=name, family=family, **value) for name, result in results.items()
                for family, value in result['families'].items()]
    write_csv(root / 'test_family_comparison.csv', families)
    singles = []
    for name, value in results.items():
        for problem, r in value['per_problem'].items():
            for solver in r['pool']:
                singles.append(dict(model=name, problem=problem, solver=solver,
                     solver_top1=r['method_top1'][solver], solver_mean_cost=r['method_mean_cost'][solver],
                     selector_top1=r['top1'], selector_mean_cost=r['mean_cost'],
                     selector_top1_exceeds=r['top1'] > r['method_top1'][solver],
                     selector_cost_exceeds=r['mean_cost'] < r['method_mean_cost'][solver],
                     fixed_best_top1=r['best_top1'], fixed_best_top2=r['best_top2'], fixed_best_top3=r['best_top3']))
    write_csv(root / 'test_vs_single_solvers.csv', singles)
    return results


def test_plots(root, results):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = ['#327b88', '#aa673b', '#5b8b56', '#aa536d']
    for problem in PROBLEMS:
        reference = results['R34']['per_problem'][problem]
        names = reference['pool']
        directory = root / 'test_method_plots' / problem
        directory.mkdir(parents=True, exist_ok=True)
        for key, metric, filename in [('method_top1', 'top1', 'top1_vs_single_methods'),
                                      ('method_mean_cost', 'mean_cost', 'mean_cost_vs_single_methods')]:
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.bar(names, [reference[key][s] for s in names], color='#bdc6c9')
            for color, (name, value) in zip(colors, results.items()):
                ax.axhline(value['per_problem'][problem][metric], color=color, label=name)
            ax.set(title=f'{problem}: frozen test {metric}', ylabel=metric)
            ax.tick_params(axis='x', labelrotation=35, labelsize=8)
            ax.legend(ncol=4)
            fig.tight_layout()
            fig.savefig(directory / f'{filename}.png', dpi=140)
            plt.close(fig)
        fig, ax = plt.subplots(figsize=(10, 4))
        positions = np.arange(len(names))
        for j, (name, value) in enumerate(results.items()):
            ax.bar(positions + (j - 1.5) * .18, [value['per_problem'][problem]['arm_distribution'][s] for s in names],
                   width=.18, color=colors[j], label=name)
        ax.set(xticks=positions, xticklabels=names, title=f'{problem}: frozen test solver picks', ylabel='Pick fraction')
        ax.tick_params(axis='x', labelrotation=35, labelsize=8)
        ax.legend(ncol=4)
        fig.tight_layout()
        fig.savefig(directory / 'arm_distribution.png', dpi=140)
        plt.close(fig)


def aligned_repeats(costs, deterministic):
    """Only reuse a verified exact constant for the optional third scenario."""
    aligned = costs.copy()
    reused = np.zeros_like(costs, dtype=bool)
    for solver, is_deterministic in enumerate(deterministic):
        if not is_deterministic:
            continue
        first, second = costs[:2, :, solver]
        stable = np.isfinite(first) & np.isfinite(second) & (first == second)
        missing = ~np.isfinite(costs[2, :, solver])
        selected = stable & missing
        aligned[2, selected, solver] = first[selected]
        reused[2, selected, solver] = True
    return aligned, reused


def complete_pool_metrics(costs, historical, winner, pred, deterministic, rtol=1e-6, atol=1e-6):
    observed = np.isfinite(costs)
    expected = np.array([2 if value else 3 for value in deterministic])
    complete = np.stack([observed[:n, :, j].all(0) for j, n in enumerate(expected)], -1).all(1)
    aligned, reused = aligned_repeats(costs, deterministic)
    rows = []
    for index in np.flatnonzero(complete):
        vectors = aligned[:, index]
        valid = np.isfinite(vectors).all(1)
        vectors = vectors[valid]
        if len(vectors) < 2:
            raise ValueError('Complete audit requires two independent full cost vectors')
        winners = vectors.argmin(1)
        ties = np.isclose(vectors, vectors.min(1)[:, None], rtol=rtol, atol=atol)
        mean = np.array([costs[:n, index, j].mean() for j, n in enumerate(expected)])
        oracle = mean.min()
        old_order = np.argsort(historical[index], kind='stable')
        old_gap = (historical[index, old_order[1]] / historical[index, old_order[0]] - 1) * 100
        old_regret = (historical[index, pred[index]] / historical[index].min() - 1) * 100
        rows.append(dict(position=int(index), complete=True, old_gap_pct=float(old_gap),
             gap_bucket=GAP_LABELS[np.searchsorted([.1, .5], old_gap, side='left')],
             native_winner=int(winner[index]), r39a_pred=int(pred[index]),
             repeat_winners=json.dumps(winners.tolist()), aligned_vectors=len(vectors),
             observed_full_vectors=int(observed[:, index].all(1).sum()), reused_deterministic_costs=int(reused[:, index].sum()),
             strict_winner_consistent=bool((winners == winners[0]).all()),
             numerical_tie_consistent=bool(ties.all(0).any()),
             native_winner_changed=bool((winners != winner[index]).any()),
             native_winner_numerically_lost=bool((~ties[:, winner[index]]).any()),
             label_mean_regret_pct=float((mean[winner[index]] / oracle - 1) * 100),
             r39a_mean_regret_pct=float((mean[pred[index]] / oracle - 1) * 100),
             r39a_old_regret_pct=float(old_regret),
             old_top2_diff=json.dumps((vectors[:, old_order[1]] - vectors[:, old_order[0]]).tolist())))
    return rows, complete, reused


def stability_tables(root):
    path = root / 'repeated_costs.npz'
    if not path.exists():
        return None
    samples = json.loads((root / 'audit_indices.json').read_text())
    manifest = json.loads((root / 'solver_manifest.json').read_text())
    protocol = json.loads((root / 'protocol.json').read_text())
    rows, bucket_rows, summaries, methods = [], [], [], []
    with np.load(path) as saved:
        for problem, record in samples['problems'].items():
            for split, sample in record['splits'].items():
                prefix = f'{problem}_{split}'
                costs, old = saved[prefix + '_costs'], saved[prefix + '_historical']
                winner, pred = saved[prefix + '_winner'], saved[prefix + '_pred']
                deterministic = [manifest['solvers'][f'{problem}/{s}']['randomness'] == 'deterministic' for s in sample['pool']]
                local, complete, reused = complete_pool_metrics(costs, old, winner, pred, deterministic,
                     rtol=protocol['tie_rtol'], atol=protocol['tie_atol'])
                for row in local:
                    pos = row['position']
                    row.update(problem=problem, split=split, index=sample['indices'][pos], nodes=sample['nodes'][pos])
                    if problem == 'OVRPTW':
                        aligned, _ = aligned_repeats(costs, deterministic)
                        vectors = aligned[:, pos]
                        vectors = vectors[np.isfinite(vectors).all(1)]
                        moel, mtl = [sample['pool'].index(s) for s in ('RELD_MOEL', 'RELD_MTL')]
                        row['reld_moel_minus_mtl'] = json.dumps((vectors[:, moel] - vectors[:, mtl]).tolist())
                    else:
                        row['reld_moel_minus_mtl'] = ''
                rows.extend(local)
                summary = dict(problem=problem, split=split, requested=64, complete_instances=int(complete.sum()),
                               completed_candidate_values=int(np.isfinite(costs).sum()), total_candidate_values=costs.size,
                               reused_deterministic_costs=int(reused.sum()))
                for metric in ('strict_winner_consistent', 'numerical_tie_consistent', 'native_winner_changed',
                               'label_mean_regret_pct', 'r39a_mean_regret_pct', 'r39a_old_regret_pct'):
                    summary[metric] = float(np.mean([r[metric] for r in local])) if local else None
                summaries.append(summary)
                for label in GAP_LABELS:
                    grouped = [r for r in local if r['gap_bucket'] == label]
                    bucket_rows.append(dict(problem=problem, split=split, gap_bucket=label, n=len(grouped),
                         **{key: float(np.mean([r[key] for r in grouped])) if grouped else None for key in
                            ('native_winner_changed', 'strict_winner_consistent', 'label_mean_regret_pct',
                             'r39a_mean_regret_pct', 'r39a_old_regret_pct')}))
                for j, solver in enumerate(sample['pool']):
                    expected = 2 if deterministic[j] else 3
                    available = np.isfinite(costs[:expected, :, j]).all(0)
                    mean = costs[:expected, available, j].mean(0)
                    old_cost = old[available, j]
                    deviation = (mean / old_cost - 1) * 100
                    randomness = manifest['solvers'][f'{problem}/{solver}']['randomness']
                    methods.append(dict(problem=problem, split=split, solver=solver, complete_repeat_instances=int(available.sum()),
                         expected_repeats=expected if randomness != 'unknown' else None, randomness=randomness,
                         historical_mean_abs_error=float(np.abs(mean - old_cost).mean()) if len(mean) else None,
                         historical_mean_abs_deviation_pct=float(np.abs(deviation).mean()) if len(mean) else None,
                         repeat_mean_std=float(costs[:expected, available, j].std(0).mean()) if len(mean) else None))
    if rows:
        write_csv(root / 'label_stability.csv', rows)
    else:
        (root / 'label_stability.csv').write_text('problem,split,index,complete\n')
    write_csv(root / 'label_stability_by_gap.csv', bucket_rows)
    write_csv(root / 'solver_repeat_metrics.csv', methods)
    dump(root / 'label_stability_summary.json', summaries)
    lines = ['# R41 solver label stability', '',
             'Sampling was locked before test. Train/val remain separate. Only complete original-config candidate pools enter winner statistics.',
             'The raw NPZ keeps missing/unexecuted values as NaN. No minima across repeats replace historical labels.',
             'A third scenario may reuse a deterministic cost only after two independent runs return exactly the same value. '
             'Observed and reused counts are separate; this is not a third solver invocation.', '',
             '| Problem | Split | Complete / requested | Strict winner consistency | Numerical tie consistency | Old winner changed | Label regret | R39A regret |',
             '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for value in summaries:
        fmt = lambda key: 'not measured' if value[key] is None else f'{value[key]:.4f}'
        lines.append(f'| {value["problem"]} | {value["split"]} | {value["complete_instances"]}/64 | '
                     f'{fmt("strict_winner_consistent")} | {fmt("numerical_tie_consistent")} | '
                     f'{fmt("native_winner_changed")} | {fmt("label_mean_regret_pct")}% | {fmt("r39a_mean_regret_pct")}% |')
    if any(row['complete_instances'] == 0 for row in summaries):
        lines.extend(['', 'Some full-pool audits are blocked. Do not infer label noise or stable winners from incomplete candidates.',
                      'See `solver_manifest.json` and `blocked_dependencies.md` for concrete provenance/dependency gaps.'])
    (root / 'label_stability.md').write_text('\n'.join(lines) + '\n')
    head_pair_tables(root)
    return summaries


def head_pair_tables(root):
    """A clearly labeled two-solver audit, never a full-pool winner result."""
    samples = json.loads((root / 'audit_indices.json').read_text())['problems']['OVRPTW']['splits']
    rows, summaries = [], []
    with np.load(root / 'repeated_costs.npz') as saved:
        for split, sample in samples.items():
            prefix = f'OVRPTW_{split}'
            costs, old, pred = saved[prefix + '_costs'], saved[prefix + '_historical'], saved[prefix + '_pred']
            a, b = [sample['pool'].index(s) for s in ('RELD_MOEL', 'RELD_MTL')]
            complete = np.isfinite(costs[:2, :, [a, b]]).all((0, 2))
            if not complete.any():
                continue
            pair = costs[:2, complete][:, :, [a, b]]
            means = pair.mean(0)
            delta = pair[:, :, 0] - pair[:, :, 1]
            old_pair = old[complete][:, [a, b]]
            historical_delta = old_pair[:, 0] - old_pair[:, 1]
            selected = pred[complete]
            in_pair = np.isin(selected, [a, b])
            positions = np.flatnonzero(complete)
            pair_winner = np.array([a, b])[means.argmin(1)]
            old_oracle = old[complete].min(1)
            old_gap = np.sort(old[complete], axis=1)[:, 1] / old_oracle * 100 - 100
            for j, pos in enumerate(positions):
                rows.append(dict(split=split, index=sample['indices'][pos], nodes=sample['nodes'][pos],
                     repeat0_difference=float(delta[0, j]), repeat1_difference=float(delta[1, j]),
                     sign_stable=bool(np.sign(delta[0, j]) == np.sign(delta[1, j])),
                     historical_difference=float(historical_delta[j]),
                     historical_sign_reproduced=bool(np.sign(delta[0, j]) == np.sign(historical_delta[j])),
                     r39a_pred=sample['pool'][selected[j]], pair_winner=sample['pool'][pair_winner[j]],
                     r39a_pair_correct=bool(selected[j] == pair_winner[j]) if in_pair[j] else None,
                     r39a_pair_regret_pct=float((means[j, 0 if selected[j] == a else 1] / means[j].min() - 1) * 100) if in_pair[j] else None,
                     old_full_pool_gap_pct=float(old_gap[j]), full_pool_repeated=False))
            summaries.append(dict(split=split, n=int(complete.sum()), independent_repeats=2,
                 sign_stability=float((np.sign(delta[0]) == np.sign(delta[1])).mean()),
                 old_sign_reproduced=float((np.sign(delta[0]) == np.sign(historical_delta)).mean()),
                 max_repeat_cost_difference=float(np.abs(pair[0] - pair[1]).max()),
                 max_historical_cost_difference=float(np.abs(pair[0] - old_pair).max()),
                 r39a_picks_in_pair=int(in_pair.sum()),
                 r39a_pair_top1=float((selected[in_pair] == pair_winner[in_pair]).mean()) if in_pair.any() else None,
                 r39a_pair_regret_pct=float(np.mean([r['r39a_pair_regret_pct'] for r in rows if r['split'] == split
                                                     and r['r39a_pair_regret_pct'] is not None])) if in_pair.any() else None,
                 old_full_pool_winners_in_pair=int(np.isin(saved[prefix + '_winner'][complete], [a, b]).sum())))
    if rows:
        write_csv(root / 'head_pair_stability.csv', rows)
    dump(root / 'head_pair_stability_summary.json', summaries)
    lines = ['# OVRPTW RELD head-pair audit (partial pool)', '',
             'This is a measured RELD_MOEL/RELD_MTL comparison, not seven-solver winner stability.',
             'Original decoder: argmax, N starts, no augmentation, one sample; two independent processes per split/method.',
             'R39A accuracy/regret below uses the two-solver minimum only, and is conditional on its choice being in this pair.', '',
             '| Split | N | Sign stability | Historical sign reproduced | Max repeat cost difference | Max historical cost difference | R39A pair Top1 | R39A pair regret |',
             '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for r in summaries:
        top1 = 'not measured' if r['r39a_pair_top1'] is None else f'{r["r39a_pair_top1"]:.4f}'
        regret = 'not measured' if r['r39a_pair_regret_pct'] is None else f'{r["r39a_pair_regret_pct"]:.4f}%'
        lines.append(f'| {r["split"]} | {r["n"]} | {r["sign_stability"]:.4f} | {r["old_sign_reproduced"]:.4f} | '
                     f'{r["max_repeat_cost_difference"]:.3g} | {r["max_historical_cost_difference"]:.3g} | '
                     f'{top1} | {regret} |')
    lines.extend(['', 'Interpretation: the rerun head-pair signal is stable in this fixed sample. '
                  'R39A still fails to select its cheaper member on many instances. This favors representation/generalization '
                  'work for this particular pair, not a claim that the entire project is free of stochastic labels.', ''])
    (root / 'head_pair_stability.md').write_text('\n'.join(lines))


def verify_test_outputs(root):
    locked = json.loads((root / 'frozen_checkpoints.json').read_text())
    protocol = json.loads((root / 'protocol.json').read_text())
    results = json.loads((root / 'test_results.json').read_text())
    count = 0
    for name, record in locked.items():
        if file_hash(record['path']) != record['sha256']:
            raise ValueError('Frozen checkpoint changed after test')
        for problem in PROBLEMS:
            raw = read_raw_costs(problem, 'test')
            with np.load(root / 'test_predictions' / name / f'{problem}.npz') as saved:
                np.testing.assert_array_equal(saved['indices'], np.arange(len(raw['winner'])))
                np.testing.assert_array_equal(saved['winner'], raw['winner'])
                np.testing.assert_array_equal(saved['costs'], raw['costs'])
                np.testing.assert_array_equal(saved['pool_ids'], raw['pool_ids'])
                replay, pred = classification_metrics(saved['logits'], raw)
                np.testing.assert_array_equal(pred, saved['pred'])
            expected = results[name]['per_problem'][problem]
            for metric in ('top1', 'top2', 'top3', 'mean_cost', 'actual_regret_pct', 'vs_sbs_pct', 'vs_oracle_pct'):
                if abs(replay[metric] - expected[metric]) > 1e-12:
                    raise ValueError(f'Reported test metric disagrees with its saved predictions: {name}/{problem}/{metric}')
            count += len(pred)
    for name in ('V4Model.py', 'dual_stream.py', 'direct_selector.py', 'solver_features.py', 'local_geometry.py'):
        if file_hash(Path(__file__).parent / name) != protocol['source_hashes'][name]:
            raise ValueError('An existing model/feature implementation changed during R41')
    dump(root / 'test_prediction_replay.json', dict(passed=True, checked_instances=count,
         complete_models=4, complete_problems=18, source='original raw_label.pkl FP64 and native ind',
         checkpoints_unchanged=True, existing_model_sources_unchanged=True))


def log_wandb(root, results):
    import wandb
    run = wandb.init(project='selector', entity='yjkds-southern-university-of-science-technology',
                     name='R41_frozen_signal_audit', dir=str(root.resolve()), mode='offline',
                     config=json.loads((root / 'protocol.json').read_text()))
    for name, value in results.items():
        run.log({f'test/{name}/{key}': v for key, v in value['macro'].items()})
        for problem, row in value['per_problem'].items():
            run.log({f'test/{name}/{problem}/{key}': row[key] for key in
                     ('top1', 'top2', 'top3', 'mean_cost', 'actual_regret_pct', 'vs_sbs_pct')})
    dump(root / 'wandb_status.json', dict(mode='offline', run_id=run.id, dir=run.dir, online_sync=False))
    run.finish()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--plots', action='store_true')
    args = parser.parse_args()
    from .r41_label_stability import snapshot_stage
    snapshot_stage(args.root, 'analysis')
    results = test_tables(args.root)
    stability_tables(args.root)
    verify_test_outputs(args.root)
    if args.wandb:
        log_wandb(args.root, results)
    if args.plots:
        test_plots(args.root, results)


if __name__ == '__main__':
    main()

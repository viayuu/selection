"""Validation-only R59 replay, paired decisions and the precommitted screen."""

import argparse
import json
from pathlib import Path

import numpy as np

from code.unified_selector.registry import PROBLEMS
from .multitask_probe import dump
from .performance_analysis import write_csv
from .performance_targets import read_raw_costs
from .r43_analysis import plot_methods, plotting
from .r43_experiment import policy_metrics


ROOT = Path('code/V4/runs/R59_trainable_solver_encoder')
REFERENCE = Path('code/V4/runs/R45_solver_code_embeddings/A_handcrafted_seed2')
RUN_NAMES = dict(A='A_random_seed2', B='B_pretrained_seed2')
FIELDS = ('top1', 'top2', 'top3', 'ce', 'mean_cost', 'vs_sbs_pct',
          'vs_oracle_pct', 'actual_regret_pct', 'top1_tie_aware', 'vbs_gap_closed_pct')


def load_json(path):
    return json.loads(Path(path).read_text())


def aggregate(per_problem):
    return {'macro_' + key: float(np.mean([row[key] for row in per_problem.values()]))
            for key in FIELDS}


def grouped(per_problem):
    groups = {p: [p] for p in ('TSP', 'CVRP', 'ATSP') if p in per_problem}
    mvrp = [p for p in per_problem if p not in ('TSP', 'CVRP', 'ATSP')]
    if mvrp:
        groups['MVRP'] = mvrp
    return {name: aggregate({p: per_problem[p] for p in tasks})
            for name, tasks in groups.items()}


def replay_prediction(path, expected=None):
    with np.load(path, allow_pickle=False) as stored:
        saved = {key: stored[key].copy() for key in stored.files}
    costs, scores = saved['costs'], saved['logits']
    if costs.dtype != np.float64:
        raise ValueError('Prediction replay requires original FP64 costs')
    if scores.shape != costs.shape or not np.isfinite(scores).all():
        raise ValueError('Scores must cover the complete finite legal pool')
    np.testing.assert_array_equal(saved['indices'], np.arange(len(costs)))
    if expected is not None:
        for key in ('costs', 'winner', 'pool_ids', 'pool'):
            np.testing.assert_array_equal(saved[key], np.asarray(expected[key]))
    raw = dict(costs=costs, winner=saved['winner'], pool_ids=saved['pool_ids'].tolist(),
               pool=saved['pool'].tolist())
    metrics, prediction = policy_metrics(scores, raw)
    np.testing.assert_array_equal(prediction, saved['pred'])
    return metrics, saved


def replay_directory(directory, raw):
    per, predictions = {}, {}
    for problem, labels in raw.items():
        per[problem], predictions[problem] = replay_prediction(directory / (problem + '.npz'), labels)
    return dict(per_problem=per, macro=aggregate(per), families=grouped(per)), predictions


def comparison_rows(model, evaluation):
    per = evaluation['per_problem']
    rows = [dict(model=model, split='val', problem='ALL', n=sum(row['n'] for row in per.values()),
                 **{key: evaluation['macro']['macro_' + key] for key in FIELDS},
                 sbs_cost=float(np.mean([row['sbs_cost'] for row in per.values()])),
                 oracle_cost=float(np.mean([row['oracle_cost'] for row in per.values()])))]
    rows += [dict(model=model, split='val', problem=problem, n=row['n'],
                  **{key: row[key] for key in FIELDS},
                  sbs_cost=row['sbs_cost'], oracle_cost=row['oracle_cost'])
             for problem, row in per.items()]
    return rows


def changes(before, after, before_name, after_name):
    if set(before) != set(after):
        raise ValueError('Paired comparisons must retain the same problem scope')
    rows = []
    for problem, a in before.items():
        b = after[problem]
        for key in ('indices', 'winner', 'costs', 'pool_ids', 'pool', 'nodes'):
            np.testing.assert_array_equal(a[key], b[key])
        winner, costs = a['winner'], a['costs']
        left, right = a['pred'], b['pred']
        delta = costs[np.arange(len(left)), right] - costs[np.arange(len(left)), left]
        both_wrong = (left != winner) & (right != winner) & (left != right)
        categories = dict(corrected=(left != winner) & (right == winner),
                          harmed=(left == winner) & (right != winner),
                          both_wrong_cheaper=both_wrong & (delta < 0),
                          both_wrong_dearer=both_wrong & (delta > 0),
                          both_wrong_equal=both_wrong & (delta == 0),
                          unchanged=left == right, all=np.ones(len(left), dtype=bool))
        for category, take in categories.items():
            rows.append(dict(before=before_name, after=after_name, problem=problem,
                             category=category, n=int(take.sum()),
                             mean_cost_delta=float(delta[take].sum() / len(left)),
                             actual_regret_delta_pct=float(
                                 (delta[take] / costs.min(-1)[take]).sum() / len(left) * 100)))
    for category in categories:
        subset = [row for row in rows if row['category'] == category]
        rows.append(dict(before=before_name, after=after_name, problem='ALL', category=category,
                         n=sum(row['n'] for row in subset),
                         mean_cost_delta=float(np.mean([row['mean_cost_delta'] for row in subset])),
                         actual_regret_delta_pct=float(np.mean(
                             [row['actual_regret_delta_pct'] for row in subset]))))
    return rows


def advancement_screen(reference, random, pretrained):
    historical_gain = 100 * (pretrained['macro_top1'] - reference['macro_top1'])
    matched_gain = 100 * (pretrained['macro_top1'] - random['macro_top1'])
    no_worse_reference = pretrained['macro_actual_regret_pct'] <= reference['macro_actual_regret_pct'] + 1e-10
    no_worse_random = pretrained['macro_actual_regret_pct'] <= random['macro_actual_regret_pct'] + 1e-10
    return dict(passed=bool(historical_gain >= 3. - 1e-10 and matched_gain >= 2. - 1e-10
                           and no_worse_reference and no_worse_random),
                top1_gain_vs_R45A_pp=historical_gain, top1_gain_vs_random_pp=matched_gain,
                regret_no_worse_than_R45A=bool(no_worse_reference),
                regret_no_worse_than_random=bool(no_worse_random),
                required_gain_vs_R45A_pp=3., required_gain_vs_random_pp=2.,
                scope='validation screen only; no test access or 70% achievement claim')


def plot_histories(root, histories, reference):
    plt = plotting()
    curves = root / 'curves'
    curves.mkdir(exist_ok=True)
    fields = ('top1', 'ce', 'mean_cost', 'actual_regret_pct', 'vs_sbs_pct', 'top2')
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    curve_rows = []
    for arm, history in histories.items():
        for row in history:
            for split in ('train', 'val'):
                if row.get(split) is not None:
                    curve_rows.append(dict(arm=arm, epoch=row['epoch'], split=split,
                                           **row[split]['macro']))
        for ax, field in zip(axes.flat, fields):
            for split, style in (('train', '--'), ('val', '-')):
                points = [row for row in history if row.get(split) is not None]
                ax.plot([row['epoch'] for row in points],
                        [row[split]['macro']['macro_' + field] for row in points],
                        linestyle=style, label=arm + ' ' + split)
            ax.set(title=field, xlabel='Epoch')
            ax.grid(alpha=.2)
    for ax, field in zip(axes.flat, fields):
        ax.axhline(reference['macro_' + field], color='black', linestyle=':', label='R45A val')
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(root / 'learning_curves.png', dpi=150)
    plt.close(fig)
    write_csv(root / 'learning_curves.csv', curve_rows)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for arm, history in histories.items():
        for ax, keys in zip(axes.flat, (('loss',), ('ce', 'cmp'), ('risk',), ('kl',),
                                        ('grad_norm',), ('lr', 'native_lr'))):
            for key in keys:
                points = [row for row in history if key in row['optimization']]
                ax.plot([row['epoch'] for row in points],
                        [row['optimization'][key] for row in points], label=arm + ' ' + key)
            ax.set(xlabel='Epoch', title=' / '.join(keys))
            ax.grid(alpha=.2)
            ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(curves / 'optimization.png', dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, family in zip(axes.flat, ('TSP', 'CVRP', 'ATSP', 'MVRP')):
        for arm, history in histories.items():
            for split, style in (('train', '--'), ('val', '-')):
                points = [row for row in history if row.get(split) is not None]
                ax.plot([row['epoch'] for row in points],
                        [row[split]['families'][family]['macro_top1'] for row in points],
                        linestyle=style, label=arm + ' ' + split)
        ax.set(title=family, xlabel='Epoch', ylabel='Top1')
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(curves / 'family_top1.png', dpi=150)
    plt.close(fig)
    common = min(len(history) for history in histories.values())
    write_csv(root / 'common_epoch_validation.csv',
              [dict(arm=arm, epoch=row['epoch'], **row['val']['macro'])
               for arm, history in histories.items() for row in history[:common]])


def table(rows):
    lines = ['| Model | Problem | N | Top1 % | Top2 % | Top3 % | Mean cost | vs SBS % | Regret % |',
             '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        lines.append(f"| {row['model']} | {row['problem']} | {row['n']} | "
                     f"{100 * row['top1']:.4f} | {100 * row['top2']:.4f} | "
                     f"{100 * row['top3']:.4f} | {row['mean_cost']:.6f} | "
                     f"{row['vs_sbs_pct']:+.4f} | {row['actual_regret_pct']:.6f} |")
    return lines


def summarize(root=ROOT):
    root = Path(root)
    protocol = load_json(root / 'protocol.json')
    records = {arm: load_json(root / name / 'result.json') for arm, name in RUN_NAMES.items()}
    histories = {arm: load_json(root / name / 'history.json') for arm, name in RUN_NAMES.items()}
    raw = {problem: read_raw_costs(problem, 'val') for problem in PROBLEMS}
    historical = load_json(REFERENCE / 'best_eval.json')
    baseline_dir = REFERENCE / 'val_predictions' / f"epoch{historical['epoch']:03d}"
    reference, baseline_predictions = replay_directory(baseline_dir, raw)
    for field in FIELDS:
        np.testing.assert_allclose(reference['macro']['macro_' + field],
                                   historical['val']['macro']['macro_' + field], atol=1e-10, rtol=0)
    results, predictions, checks = {}, {}, []
    for arm, name in RUN_NAMES.items():
        result, saved = replay_directory(root / name / 'val_predictions' / 'best', raw)
        for problem in PROBLEMS:
            error = max(abs(result['per_problem'][problem][field]
                            - records[arm]['best']['val']['per_problem'][problem][field])
                        for field in FIELDS)
            if error > 1e-10:
                raise ValueError('Saved validation predictions disagree with the selected model metrics')
            checks.append(dict(arm=arm, problem=problem, n=len(saved[problem]['pred']),
                               maximum_metric_error=error))
        results[arm], predictions[arm] = result, saved
    dump(root / 'prediction_replay.json', checks)
    dump(root / 'validation_results.json', results)
    all_results = {'R45A historical': reference, **{'R59' + arm: result for arm, result in results.items()}}
    for name, result in all_results.items():
        plot_methods(name, result, root / 'curves' / name.replace(' ', '_') / 'per_problem')
    rows = [row for name, result in all_results.items() for row in comparison_rows(name, result)]
    write_csv(root / 'comparison.csv', rows)
    write_csv(root / 'family_comparison.csv', [dict(model=name, family=family, **scores)
              for name, result in all_results.items() for family, scores in result['families'].items()])
    distributions, single_methods = [], []
    for name, result in all_results.items():
        for problem, metrics in result['per_problem'].items():
            for solver in metrics['pool']:
                distributions.append(dict(model=name, problem=problem, solver=solver,
                    pick_fraction=metrics['arm_distribution'][solver],
                    native_winner_fraction=metrics['method_top1'][solver]))
                single_methods.append(dict(model=name, problem=problem, solver=solver,
                    fixed_solver_top1=metrics['method_top1'][solver],
                    fixed_solver_mean_cost=metrics['method_mean_cost'][solver],
                    selector_top1=metrics['top1'], selector_mean_cost=metrics['mean_cost'],
                    selector_top1_exceeds=solver in metrics['top1_exceeds_methods'],
                    selector_cost_exceeds=solver in metrics['cost_exceeds_methods']))
    write_csv(root / 'solver_distributions.csv', distributions)
    write_csv(root / 'single_solver_comparison.csv', single_methods)
    write_csv(root / 'fixed_pool_coverage.csv', [dict(model=name, problem=problem,
              **{key: metrics[key] for key in ('best_top1', 'best_top2', 'best_top3')})
              for name, result in all_results.items() for problem, metrics in result['per_problem'].items()])
    changes_rows = changes(baseline_predictions, predictions['A'], 'R45A', 'R59A')
    changes_rows += changes(baseline_predictions, predictions['B'], 'R45A', 'R59B')
    changes_rows += changes(predictions['A'], predictions['B'], 'R59A', 'R59B')
    write_csv(root / 'decision_changes.csv', changes_rows)
    screen = advancement_screen(reference['macro'], results['A']['macro'], results['B']['macro'])
    dump(root / 'advancement_screen.json', screen)
    plot_histories(root, histories, reference['macro'])
    lines = ['# R59: Trainable Routing-Pretrained Instance Encoding', '',
        'Original18 tasks and full candidate pools; seed2. Validation only: no R59 test access.',
        'Legacy deployment scenario: the unresolved B feasible-domain discrepancy is NOT repaired.',
        'R58 is deferred by user. No solver decoding, new cost tables, expansion data or relabeling.', '',
        '## Validation-selected outcomes', ''] + table([row for row in rows if row['problem'] == 'ALL'])
    lines += ['', 'A/B have the same trainable architecture and non-transferred initial parameters.',
        'A starts its native encoder randomly; B loads the existing ReLD-MTL encoder weights.',
        'Both encoders train jointly with the selector. R45A is a historical, not matched, control.',
        'The shared native path covers CVRP and15 MVRP tasks; TSP/ATSP retain the old encoder path.',
        'Raw/local geometry and condition features preserve fields omitted by the native encoder.', '',
        '## Training and selection', '',
        '| Arm | Best epoch | Stop epoch | Successful updates | Final val Top1 % | Last5 val Top1 % |',
        '|---|---:|---:|---:|---:|---:|']
    for arm, record in records.items():
        lines.append(f"| R59{arm} | {record['best']['epoch']} | {record['epochs_completed']} | "
                     f"{record['successful_updates']} | "
                     f"{100 * record['final']['val']['macro']['macro_top1']:.4f} | "
                     f"{100 * record['last5_val']['macro_top1']:.4f} |")
    lines += ['', '| Arm | Trainable parameters | Native parameters | Actual run wall hours | First epoch optimization seconds |',
              '|---|---:|---:|---:|---:|']
    for arm, record in records.items():
        initial = protocol['initialization_checks'][arm]
        lines.append(f"| R59{arm} | {initial['trainable_parameters']} | {initial['native_parameters']} | "
                     f"{record['wall_seconds'] / 3600:.4f} | "
                     f"{histories[arm][0]['optimization']['seconds']:.2f} |")
    lines += ['', 'Actual per-arm wall includes training and evaluations, not data preparation/preflight.',
              'First-epoch optimization time is measured; the prelaunch4--8GPUh estimate was not a cutoff.',
              '', '## Selected-checkpoint train and validation', '',
              '| Arm | Split | Top1 % | CE | Mean cost | Actual regret % |',
              '|---|---|---:|---:|---:|---:|']
    for arm, record in records.items():
        for split in ('train', 'val'):
            selected = record['best'][split]['macro']
            lines.append(f"| R59{arm} | {split} | {100 * selected['macro_top1']:.4f} | "
                         f"{selected['macro_ce']:.6f} | {selected['macro_mean_cost']:.6f} | "
                         f"{selected['macro_actual_regret_pct']:.6f} |")
    lines += ['', 'Selection is the strict minimum validation macro_vs_sbs_pct, as precommitted.',
        'Native encoder LR3e-5, other parameters1e-4 in both arms; identical scheduling rules,',
        'not necessarily identical realized LR trajectories or stopping epochs. Original R45A',
        'comparison/risk/R-Drop objectives are reused without a coefficient search.', '',
        '## Paired decision changes', '',
        '| Before | After | Category | N | Macro mean-cost delta | Macro regret delta pp |',
        '|---|---|---|---:|---:|---:|']
    for row in changes_rows:
        if row['problem'] == 'ALL':
            lines.append(f"| {row['before']} | {row['after']} | {row['category']} | {row['n']} | "
                         f"{row['mean_cost_delta']:+.8f} | {row['actual_regret_delta_pct']:+.6f} |")
    lines += ['', '## Precommitted decision', '',
        f"Validation advancement screen passed: **{screen['passed']}**.",
        f"B minus R45A Top1: {screen['top1_gain_vs_R45A_pp']:+.4f}pp (required>=3).",
        f"B minus A Top1: {screen['top1_gain_vs_random_pp']:+.4f}pp (required>=2).",
        f"Regret no worse than R45A/A: {screen['regret_no_worse_than_R45A']}/"
        f"{screen['regret_no_worse_than_random']}.", '',
        'Passing this screen is not achievement of70% or evidence from an unseen test.',
        'A negative result rejects this transfer configuration, not a hard accuracy ceiling.',
        'No extra epochs, fine-tuning variants, task gates or test-driven changes are licensed',
        'by this report. Subsequent mechanisms require a new precommitted review round.', '',
        '## Interpretation limits', '',
        'One seed and external solver pretraining; total pretraining data are not matched.',
        'Historical training-instance provenance of the native weights is incomplete; absence',
        'of pretraining overlap cannot be certified. Existing B-domain and deployment gaps',
        'remain open. This evaluates native-winner prediction in the legacy cost table only.',
        'The historical test has been seen in prior rounds and is not described as pristine.', '',
        '## Per problem', ''] + table([row for row in rows if row['problem'] != 'ALL'])
    (root / 'RESULTS.md').write_text('\n'.join(lines) + '\n')
    return screen


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    print(json.dumps(summarize(parser.parse_args().root), indent=2))

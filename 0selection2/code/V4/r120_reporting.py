"""R120 native-objective metrics and portable, saved-prediction PDF reports.

Semantics fixed before implementation (base 048664148): every task retains its
native min/max objective, exact ties, and missing-label mask. TRAIN chooses the
SBS and a fixed positive objective scale. ALL rows average dimensionless task
metrics, never raw objectives. Routing native-ind and any-best are distinct.
Canonical winners on new tasks are only a documented display convention.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


ROUTING_PROBLEMS = ('TSP', 'CVRP', 'ATSP', 'OVRP', 'VRPB', 'VRPL', 'VRPTW',
                    'OVRPTW', 'OVRPB', 'OVRPL', 'VRPBL', 'VRPBTW', 'VRPLTW',
                    'OVRPBL', 'OVRPBTW', 'OVRPLTW', 'VRPBLTW', 'OVRPBLTW')
NONROUTING_PROBLEMS = ('MIS', 'JSSP', 'FJSP', 'BPP', 'MKP', 'SMTWTP')
PROBLEMS = ROUTING_PROBLEMS + NONROUTING_PROBLEMS
LIBS = {'TSPLIB': 'TSP', 'CVRPLIB': 'CVRP'}


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name('.' + path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n',
                         encoding='utf-8')
    temporary.replace(path)


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        writer.writerows(rows)


def objective_arrays(objectives, observed, direction):
    values, mask = np.asarray(objectives), np.asarray(observed, dtype=bool)
    if values.dtype != np.float64 or values.ndim != 2 or values.shape != mask.shape:
        raise ValueError('Native objectives must be FP64 [rows,candidates] with an aligned mask')
    if direction not in ('min', 'max') or not mask.any(1).all():
        raise ValueError('Each instance requires a native min/max direction and an observation')
    if not np.isfinite(values[mask]).all():
        raise ValueError('Observed objectives must be finite')
    signed = values if direction == 'min' else -values
    best_signed = np.where(mask, signed, np.inf).min(1)
    best = mask & (signed == best_signed[:, None])
    oracle = best_signed if direction == 'min' else -best_signed
    return values, mask, signed, oracle, best


def fit_train_reference(objectives, observed, direction, solver_names, *, zero_scale=None):
    """Fit the fixed deployment and native unit from TRAIN, without compatibility offsets."""
    values, mask, _, _, _ = objective_arrays(objectives, observed, direction)
    names = list(solver_names)
    if len(names) != values.shape[1]:
        raise ValueError('Candidate identities must match objective columns')
    complete = mask.all(0)
    if not complete.any():
        raise ValueError('TRAIN has no fully observed fixed deployment for an honest SBS')
    means = np.where(mask, values, 0).sum(0) / mask.sum(0).clip(1)
    signed_means = means if direction == 'min' else -means
    slot = int(np.argmin(np.where(complete, signed_means, np.inf)))
    scale = float(means[slot])
    source = 'native mean objective of TRAIN-fitted SBS'
    if scale <= 0:
        if zero_scale is None or not np.isfinite(zero_scale) or zero_scale <= 0:
            raise ValueError('A zero/nonpositive TRAIN-SBS objective requires an explicit positive input scale')
        scale, source = float(zero_scale), 'TRAIN mean(sum processing times * sum weights)'
    return dict(sbs_slot=slot, sbs_solver=names[slot], selection_split='train',
                objective_direction=direction, objective_scale=scale, scale_source=source,
                mean_objective_by_solver={name: float(means[i]) if mask[:, i].any() else None
                                          for i, name in enumerate(names)},
                fully_observed_candidates=complete.tolist(), train_instances=len(values))


def native_task_metrics(scores, objectives, observed, direction, *, all_best_mask=None,
                        native_winner=None, reference=None):
    """Return one task's metrics and choices; a missing selected label stays explicit."""
    values, mask, signed, oracle, best = objective_arrays(objectives, observed, direction)
    scores = np.asarray(scores)
    if scores.shape != values.shape or not np.isfinite(scores).all():
        raise ValueError('Each predeclared candidate requires a finite model score')
    if all_best_mask is not None and not np.array_equal(best, np.asarray(all_best_mask, dtype=bool)):
        raise ValueError('Saved all-best labels disagree with original-precision objectives')
    rows, n = np.arange(len(values)), len(values)
    prediction = scores.argmax(1).astype(np.int64)
    canonical = best.argmax(1)
    selected_observed = mask[rows, prediction]
    any_correct = best[rows, prediction]
    canonical_correct = prediction == canonical
    native_correct = None
    if native_winner is not None:
        native = np.asarray(native_winner, dtype=np.int64)
        if native.shape != (n,) or (native < 0).any() or (native >= values.shape[1]).any():
            raise ValueError('Native winner shape/slot differs from candidate pool')
        if not best[rows, native].all():
            raise ValueError('Native winner is not a native-objective optimum')
        native_correct = prediction == native
    shifted = scores.astype(np.float64) - scores.max(1, keepdims=True).astype(np.float64)

    def logsumexp(array):
        maximum = array.max(1)
        return maximum + np.log(np.exp(array - maximum[:, None]).sum(1))

    log_normalizer = logsumexp(shifted)
    ce = (log_normalizer - shifted[rows, native] if native_correct is not None else
          log_normalizer - logsumexp(np.where(best, shifted, -np.inf)))
    chosen = values[rows, prediction]
    directed_excess = np.where(selected_observed, signed[rows, prediction] -
                               (oracle if direction == 'min' else -oracle), np.nan)
    positive = selected_observed & (oracle > 0)
    zero = selected_observed & (oracle == 0)
    valid = selected_observed
    mean = lambda a: float(np.mean(a, dtype=np.float64)) if len(a) else None
    relative = directed_excess[positive] / oracle[positive]
    scale = float(reference['objective_scale']) if reference is not None else None
    if scale is not None and (not np.isfinite(scale) or scale <= 0):
        raise ValueError('TRAIN objective scale must be finite and positive')
    result = dict(n=n, objective_direction=direction, ce=float(ce.mean()),
        ce_rule='native-winner CE' if native_correct is not None else 'exact best-set probability CE',
        any_best_correct=int(any_correct.sum()),
        any_best_top1_pct=100. * float(any_correct.mean()),
        canonical_correct=int(canonical_correct.sum()),
        canonical_top1_pct=100. * float(canonical_correct.mean()),
        native_correct=int(native_correct.sum()) if native_correct is not None else None,
        native_top1_pct=100. * float(native_correct.mean()) if native_correct is not None else None,
        top1_pct=100. * float((native_correct if native_correct is not None else any_correct).mean()),
        top1_rule='native/FP64 ind' if native_correct is not None else 'any native exact best',
        tied_instances=int((best.sum(1) > 1).sum()),
        selected_objective_observed=int(valid.sum()), selected_objective_missing=int((~valid).sum()),
        complete_pool_instances=int(mask.all(1).sum()),
        mean_objective=mean(chosen[valid]), mean_cost=mean(chosen[valid]) if direction == 'min' else None,
        oracle_objective=mean(oracle), oracle_cost=mean(oracle) if direction == 'min' else None,
        absolute_excess=mean(directed_excess[valid]),
        positive_oracle_count=int(positive.sum()), zero_oracle_count=int(zero.sum()),
        zero_oracle_total=int((oracle == 0).sum()),
        actual_regret_pct=100. * mean(relative) if len(relative) else None,
        relative_regret_definition='directed absolute excess / positive native pool-oracle only',
        zero_oracle_absolute_excess=mean(directed_excess[zero]),
        zero_oracle_positive_loss_pct=100. * float((directed_excess[zero] > 0).mean()) if zero.any() else None,
        train_scaled_excess=mean(directed_excess[valid] / scale) if scale is not None else None,
        objective_scale=scale, sbs_selection_split='train' if reference is not None else None,
        sbs_objective=None, sbs_cost=None, sbs_solver=None, vs_sbs_pct=None, vs_oracle_pct=None)
    if reference is not None:
        slot = int(reference['sbs_slot'])
        sbs_valid = mask[:, slot]
        sbs_mean = mean(values[sbs_valid, slot])
        result.update(sbs_objective=sbs_mean, sbs_cost=sbs_mean if direction == 'min' else None,
                      sbs_solver=reference['sbs_solver'], sbs_observed_instances=int(sbs_valid.sum()))
        if valid.all() and sbs_valid.all() and sbs_mean > 0:
            excess = result['mean_objective'] - sbs_mean
            result['vs_sbs_pct'] = 100. * excess / sbs_mean * (1 if direction == 'min' else -1)
            result['net_objective_gain_vs_sbs'] = -excess * (1 if direction == 'min' else -1)
        if valid.all() and result['oracle_objective'] > 0:
            excess = result['mean_objective'] - result['oracle_objective']
            result['vs_oracle_pct'] = 100. * excess / result['oracle_objective'] * (1 if direction == 'min' else -1)
    return result, prediction


def aggregate_metrics(per_problem):
    """Task macro/sample micro; historical raw-cost means only for ROUTING18."""
    result = {}
    groups = {'ROUTING18': ROUTING_PROBLEMS, 'NONROUTING6': NONROUTING_PROBLEMS,
              'ALL24': PROBLEMS}
    for group, expected in groups.items():
        tasks = [p for p in expected if p in per_problem]
        if not tasks:
            continue
        rows, total = [per_problem[p] for p in tasks], sum(per_problem[p]['n'] for p in tasks)
        macro = lambda key: (float(np.mean([row[key] for row in rows]))
                             if all(row.get(key) is not None for row in rows) else None)
        native_rows = [row for row in rows if row.get('native_correct') is not None]
        mixed_correct = sum(row['native_correct'] if row.get('native_correct') is not None
                            else row['any_best_correct'] for row in rows)
        relative_sum = sum((row.get('actual_regret_pct') or 0) * row['positive_oracle_count'] for row in rows)
        positive_count = sum(row['positive_oracle_count'] for row in rows)
        result[group] = dict(tasks=tasks, task_count=len(tasks), expected_task_count=len(expected),
            complete_task_scope=len(tasks) == len(expected), n=total,
            task_macro_ce=macro('ce'),
            sample_micro_ce=(sum(row['ce'] * row['n'] for row in rows) / total
                             if all(row.get('ce') is not None for row in rows) else None),
            task_macro_any_best_top1_pct=macro('any_best_top1_pct'),
            sample_micro_any_best_top1_pct=100. * sum(row['any_best_correct'] for row in rows) / total,
            task_macro_mixed_top1_pct=macro('top1_pct'), sample_micro_mixed_top1_pct=100. * mixed_correct / total,
            task_macro_native_top1_pct=macro('native_top1_pct') if len(native_rows) == len(rows) else None,
            task_macro_train_scaled_excess=macro('train_scaled_excess'),
            task_macro_positive_oracle_regret_pct=macro('actual_regret_pct'),
            sample_micro_positive_oracle_regret_pct=relative_sum / positive_count if positive_count else None,
            positive_oracle_count=positive_count, zero_oracle_count=sum(row['zero_oracle_total'] for row in rows),
            mean_objective=None, mean_cost=None, raw_objective_aggregation='not defined across different task units')
        if group == 'ROUTING18':
            result[group].update({key: macro(key) for key in (
                'mean_cost', 'sbs_cost', 'oracle_cost', 'vs_sbs_pct', 'vs_oracle_pct', 'actual_regret_pct')})
            result[group]['raw_objective_aggregation'] = 'historical routing-only equal-weight per-task cost means'
    return result


def method_rows(problem, split, data, metrics, prediction=None):
    """Per-method support includes ties; unique misses and replacement loss do not.

    Replacement excess compares the actual selected deployment with this best
    method, in native objective direction. Selecting another exact best costs
    zero. Tied-support losses may overlap across methods; they are not additive
    pieces of a system error budget. Missing selected labels stay undefined.
    """
    values = data['native_objective']
    best, names = data['all_best_mask'], data['pool'].tolist()
    pred = np.asarray(prediction if prediction is not None else data['logits'].argmax(1))
    canonical = np.asarray(data.get('native_winner_index', best.argmax(1)))
    observed = data['observed_mask']
    signed = values if metrics['objective_direction'] == 'min' else -values
    oracle = np.where(observed, signed, np.inf).min(1)
    selected = signed[np.arange(len(values)), pred]
    selected_observed = observed[np.arange(len(values)), pred]
    result = []
    for slot, name in enumerate(names):
        chose, won = pred == slot, best[:, slot]
        tp = int((chose & won).sum())
        false = chose & ~won & observed[:, slot]
        unique = won & (best.sum(1) == 1)
        unique_tp, unique_missed = int((unique & chose).sum()), unique & ~chose
        missed_best, missed_tied = won & ~chose, won & ~unique & ~chose

        def replacement_excess(subset):
            if np.any(subset & ~selected_observed):
                return None
            return float((selected[subset] - signed[subset, slot]).sum())

        missed_unique_excess = replacement_excess(unique_missed)
        result.append(dict(problem=problem, split=split, solver=name, n=len(values),
            selected_count=int(chose.sum()), selected_pct=100. * float(chose.mean()),
            displayed_winner_count=int((canonical == slot).sum()),
            displayed_winner_pct=100. * float((canonical == slot).mean()),
            all_best_support_count=int(won.sum()), unique_best_count=int(unique.sum()),
            unique_best_true_positive=unique_tp, unique_best_missed=int(unique_missed.sum()),
            unique_best_recall=unique_tp / int(unique.sum()) if unique.any() else None,
            missed_best_replacement_directed_excess=replacement_excess(missed_best),
            tied_best_replacement_directed_excess=replacement_excess(missed_tied),
            missed_unique_winner_train_scaled_excess=(missed_unique_excess / metrics['objective_scale']
                                                      if missed_unique_excess is not None else None),
            positive_cost_missed_best_count=int((missed_best & selected_observed &
                                                (selected > signed[:, slot])).sum()),
            missed_best_unobserved_replacements=int((missed_best & ~selected_observed).sum()),
            true_positive=tp, false_positive=int((chose & ~won).sum()), false_negative=int((~chose & won).sum()),
            precision=tp / int(chose.sum()) if chose.any() else None,
            recall=tp / int(won.sum()) if won.any() else None,
            strict_zero_selected=not bool(chose.any()), positive_winner_support=bool(won.any()),
            false_selection_absolute_excess=float((selected[false] - oracle[false]).sum()),
            false_selection_train_scaled_excess=float((selected[false] - oracle[false]).sum() /
                                                       metrics['objective_scale']),
            winner_display_rule=('native/FP64 ind' if 'native_winner_index' in data
                                 else 'first exact best in fixed candidate order; not a unique label')))
    return result


def _plotting():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    for path in (Path('/usr/share/fonts/google-noto-cjk/NotoSansCJK-Regular.ttc'),
                 Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')):
        if path.is_file():
            font_manager.fontManager.addfont(str(path))
            matplotlib.rcParams['font.family'] = font_manager.FontProperties(fname=path).get_name()
            break
    for font in font_manager.fontManager.ttflist:
        if any(name in font.name for name in ('Noto Sans CJK', 'WenQuanYi', 'Source Han Sans')):
            matplotlib.rcParams['font.family'] = font.name
            break
    matplotlib.rcParams['axes.unicode_minus'] = False
    return plt


def distribution_figure(problem, split, data, metrics, model_name, data_version):
    plt = _plotting()
    rows = method_rows(problem, split, data, metrics)
    fig, ax = plt.subplots(figsize=(12, max(5, .56 * len(rows) + 2.6)))
    y = np.arange(len(rows))
    for offset, key, color, label in ((-.18, 'selected_pct', '#167d85', '模型选择'),
                                      (.18, 'displayed_winner_pct', '#c34e56', '真实最优方法')):
        values = np.asarray([r[key] for r in rows])
        ax.barh(y + offset, values, height=.31, color=color, label=label)
        count_key = 'selected_count' if key == 'selected_pct' else 'displayed_winner_count'
        for yy, value, row in zip(y + offset, values, rows):
            ax.text(value + .7, yy, f'{value:.2f}% ({row[count_key]:,})', va='center', color=color, fontsize=9)
    limit = min(118, max(30, max(max(r['selected_pct'], r['displayed_winner_pct']) for r in rows) + 18))
    ax.set(xlim=(0, limit), yticks=y, yticklabels=[r['solver'] for r in rows],
           xlabel='Share of dataset instances (%)')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=.2)
    ax.set_axisbelow(True)
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(loc='upper right', frameon=False)
    ax.set_title(f'{model_name} | {problem} {split} | n={metrics["n"]:,} | 模型Top1={metrics["top1_pct"]:.2f}%',
                 loc='left', fontsize=15, pad=30)
    rule = rows[0]['winner_display_rule']
    fig.text(.02, .018, f'Data: {data_version}\nTop1: {metrics["top1_rule"]}; display: {rule}; ties={metrics["tied_instances"]:,}',
             fontsize=8, color='#38464d')
    fig.tight_layout(rect=(0, .07, 1, 1))
    return fig


def _table_page(pdf, title, columns, rows, note=''):
    plt = _plotting()
    if len(rows) > 25:
        for start in range(0, len(rows), 25):
            _table_page(pdf, f'{title} ({start + 1}-{min(start + 25, len(rows))})',
                        columns, rows[start:start + 25], note)
        return
    fig, ax = plt.subplots(figsize=(15, 9))
    ax.axis('off')
    ax.set_title(title, loc='left', fontsize=18, pad=25)
    cells = []
    for row in rows:
        cells.append(['missing' if row.get(key) is None else
                      f'{row[key]:.5f}' if isinstance(row[key], float) else str(row[key])
                      for key in columns])
    if cells:
        table = ax.table(cellText=cells, colLabels=columns, loc='upper center', cellLoc='left')
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.6)
        for (i, _), cell in table.get_celld().items():
            cell.set_edgecolor('#e0e6eb')
            if i == 0:
                cell.set_facecolor('#dbe8eb')
    else:
        ax.text(.03, .85, 'No instances or methods in this subset.', transform=ax.transAxes, fontsize=12)
    fig.text(.04, .04, note, fontsize=9, wrap=True)
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


def nss_rows(predictions, nss_root, model_name):
    """Reuse matching saved NSS predictions; never silently compare another release."""
    root = Path(nss_root) if nss_root else None
    provenance = read_json(root / 'protocol.json') if root and (root / 'protocol.json').is_file() else {}
    r79_reference = ('R79' in provenance.get('weights_description', '')
                     and provenance.get('retrained_on_new90k') is False)
    rows = []
    for name, problem, split in (('TSP TEST', 'TSP', 'test'), ('CVRP TEST', 'CVRP', 'test'),
                                 ('TSPLIB', 'TSP', 'TSPLIB'), ('CVRPLIB', 'CVRP', 'CVRPLIB')):
        current = predictions.get((split, problem))
        row = dict(dataset=name, current_model=model_name, nss_top1_pct=None, current_top1_pct=None,
                   nss_mean_cost=None, current_mean_cost=None, status='missing current predictions',
                   nss_weights_description=provenance.get('weights_description', 'missing NSS weight provenance'),
                   nss_historical_training_data_root=provenance.get('historical_training_data_root'),
                   nss_historical_train_rows_per_problem=70114 if r79_reference else None,
                   nss_retrained_on_new90k=provenance.get('retrained_on_new90k'),
                   training_protocol_match=False if provenance.get('retrained_on_new90k') is False else None)
        if current is not None:
            data, metric = current
            row.update(current_top1_pct=metric['native_top1_pct'], current_mean_cost=metric['mean_cost'],
                       n=metric['n'], status='missing same-population NSS predictions')
            path = root / problem / f'{split}_predictions.npz' if root else None
            if path is not None and path.is_file():
                with np.load(path, allow_pickle=False) as saved:
                    scores = saved['scores'] if 'scores' in saved else saved['logits']
                    costs = saved['costs']
                    winner = saved['winner']
                    pool = saved['pool']
                    exact = (np.array_equal(costs, data['native_objective']) and
                             np.array_equal(winner, data['native_winner_index']) and
                             np.array_equal(pool, data['pool']))
                    if exact:
                        pred = scores.argmax(1)
                        row.update(nss_top1_pct=100. * float((pred == winner).mean()),
                            nss_mean_cost=float(costs[np.arange(len(costs)), pred].mean()),
                            status='same EVAL costs/winners/pool; historical NSS TRAIN protocol retained')
                    else:
                        row['status'] = 'NSS population/cost/winner/pool mismatch; reference not reused'
        rows.append(row)
    return rows


def learning_curves(root, output):
    history = read_json(root / 'history.json') if (root / 'history.json').exists() else []
    if not history:
        return None
    plt = _plotting()
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.7))
    for axis, field, multiplier, label, style in (
            (axes[0], 'ce', 1., 'online TRAIN CE', '--'),
            (axes[1], 'train_any_best_accuracy', 100., 'online TRAIN any-best', '--'),
            (axes[1], 'train_mixed_accuracy', 100., 'online TRAIN mixed', ':'),
            (axes[2], 'train_hard_scaled_excess', 1., 'online TRAIN hard excess', '--')):
        values = [row.get('train_parts', {}).get(field) for row in history]
        if any(value is not None for value in values):
            axis.plot([row['successful_updates'] for row in history],
                      [np.nan if value is None else value * multiplier for value in values],
                      color='#26343c', linestyle=style, label=label)
    for group, color in (('ROUTING18', '#167d85'), ('NONROUTING6', '#c34e56'), ('ALL24', '#455f92')):
        rows = [row for row in history if group in row.get('validation', {}).get('aggregates', {})]
        x = [r['successful_updates'] for r in rows]
        for ax, field, title in zip(axes,
                ('task_macro_ce', 'task_macro_any_best_top1_pct', 'task_macro_train_scaled_excess'),
                ('CE: online TRAIN and full VAL', 'Top1 (%): online TRAIN and full VAL',
                 'Hard TRAIN-scaled excess')):
            y = [r['validation']['aggregates'][group].get(field) for r in rows]
            ax.plot(x, [np.nan if v is None else v for v in y], label='VAL ' + group, color=color)
            ax.set(xlabel='Successful optimizer updates', title=title)
            ax.grid(alpha=.2)
    for axis in axes:
        axis.legend(frameon=False, fontsize=8)
    fig.text(.015, .015, 'Online TRAIN: interval averages from the changing model in train/dropout mode; '
             'not a full-TRAIN evaluation of the locked checkpoint. VAL: full frozen heldout populations.', fontsize=8)
    fig.tight_layout(rect=(0, .07, 1, 1))
    fig.savefig(output, dpi=170)
    fig.savefig(Path(output).with_suffix('.pdf'))
    plt.close(fig)
    return str(output)


def report(root, *, nss_root=None):
    """CPU-only reporting from the single locked checkpoint's saved predictions."""
    from matplotlib.backends.backend_pdf import PdfPages
    from .r76_experiment import file_hash
    root = Path(root)
    config, lock = read_json(root / 'protocol.json'), read_json(root / 'final_test_lock.json')
    if lock['checkpoint_sha256'] != file_hash(root / 'best.pt') or not lock['selected_without_test']:
        raise ValueError('The unique VAL-selected checkpoint lock changed')
    predictions, metric_rows, methods, summaries = {}, [], [], {}
    for split in ('val', 'test', 'TSPLIB', 'CVRPLIB'):
        directory = root / 'evaluation' / split / 'selected_best'
        if not (directory / 'metrics.json').is_file():
            if split in ('val', 'test'):
                raise FileNotFoundError(f'Complete selected-best {split} evaluation first')
            continue
        result = read_json(directory / 'metrics.json')
        if result['identity']['checkpoint_sha256'] != lock['checkpoint_sha256']:
            raise ValueError('Predictions use a different checkpoint')
        summaries[split] = result
        for problem, metric in result['per_problem'].items():
            path = directory / 'by_problem' / f'{problem}.npz'
            with np.load(path, allow_pickle=False) as archive:
                data = {key: archive[key] for key in archive.files}
            predictions[split, problem] = data, metric
            metric_rows.append(dict(split=split, problem=problem, **metric))
            methods.extend(method_rows(problem, split, data, metric))
    output = root / 'reports'
    output.mkdir(exist_ok=True)
    write_csv(output / 'metrics.csv', metric_rows)
    write_csv(output / 'selection_distribution.csv', methods)
    comparisons = nss_rows(predictions, nss_root, config.get('model_name', 'R120'))
    write_csv(output / 'nss_comparison.csv', comparisons)
    write_json(output / 'aggregate_metrics.json', {split: value['aggregates'] for split, value in summaries.items()})
    learning_curves(root, output / 'learning_curves.png')
    plt = _plotting()
    with PdfPages(output / 'selection_distributions.pdf') as pdf:
        for problem in config['model']['problem_names']:
            for split in ('test',) + tuple(name for name, task in LIBS.items() if task == problem):
                if (split, problem) not in predictions:
                    continue
                data, metric = predictions[split, problem]
                fig = distribution_figure(problem, split.upper(), data, metric,
                    config.get('model_name', 'R120'), config['data_version'])
                pdf.savefig(fig)
                plt.close(fig)
    with PdfPages(output / 'RESULTS.pdf') as pdf:
        _table_page(pdf, 'R120 | Same-checkpoint VAL / TEST',
            ['split', 'group', 'task_count', 'task_macro_any_best_top1_pct',
             'task_macro_mixed_top1_pct', 'task_macro_train_scaled_excess'],
            [dict(split=split.upper(), group=group, **row) for split, result in summaries.items()
             for group, row in result.get('aggregates', {}).items()],
            'ALL24 is task-macro; raw costs/profits/makespans are never pooled. Routing native Top1 is separate. '
            'New-task ties use any-best Top1. TEST/LIB did not select this checkpoint. '
            'This is a new multidomain baseline; historical R94B is not a matched architectural control.')
        for split in ('val', 'test'):
            _table_page(pdf, f'R120 | {split.upper()} native objectives',
                ['problem', 'objective_direction', 'top1_pct', 'mean_objective', 'sbs_objective',
                 'oracle_objective', 'actual_regret_pct', 'train_scaled_excess'],
                [row for row in metric_rows if row['split'] == split],
                'SBS fitted on TRAIN. Relative regret is defined only for strictly positive native pool-oracles; '
                'zero-oracle count and absolute losses are in metrics.csv. Canonical tie display is not a unique ground truth.')
        _table_page(pdf, 'R120 | Historical NSS reference',
            ['dataset', 'nss_top1_pct', 'current_top1_pct', 'nss_mean_cost', 'current_mean_cost', 'status'],
            comparisons, 'Only same-population evaluation labels and candidates are compared. Missing references remain missing. '
            'This does not match the training populations: R120 routing uses 81,000 TRAIN instances/task.\n'
            'NSS weights: ' + comparisons[0]['nss_weights_description'] + '\nHistorical NSS TRAIN: ' +
            str(comparisons[0]['nss_historical_training_data_root']) + '; rows/task=' +
            str(comparisons[0]['nss_historical_train_rows_per_problem']))
        _table_page(pdf, 'R120 | Zero-choice methods with unique wins or positive replacement loss',
            ['problem', 'solver', 'unique wins', 'unique missed', 'unique recall',
             'unique missed loss / s_t', 'tied replacement excess'],
            [dict(problem=row['problem'], solver=row['solver'], **{
                'unique wins': row['unique_best_count'], 'unique missed': row['unique_best_missed'],
                'unique recall': row['unique_best_recall'],
                'unique missed loss / s_t': row['missed_unique_winner_train_scaled_excess'],
                'tied replacement excess': row['tied_best_replacement_directed_excess']})
             for row in methods if row['split'] == 'test' and row['strict_zero_selected']
             and (row['unique_best_missed'] or row['positive_cost_missed_best_count'])],
            'Existing false_negative/recall/positive_winner_support use all-best support, including ties. '
            'Unique-best recall and directed replacement loss measure missed opportunities. Selecting another exact '
            'best has zero replacement loss; tied methods receive no selection quotas. Per-method tied losses can overlap.')
    write_json(output / 'report_complete.json', dict(complete=True,
        checkpoint_sha256=lock['checkpoint_sha256'], selected_update=lock['selected_update'],
        task_scope=config['model']['problem_names'],
        complete24=set(summaries['test']['per_problem']) == set(PROBLEMS),
        lib_complete=all(name in summaries for name in LIBS),
        nss_reference_status=[row['status'] for row in comparisons]))
    return output


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--root', required=True, type=Path)
    result.add_argument('--nss-root', type=Path, default=Path(__file__).resolve().parent /
                        'runs/R84_new90k_attention/nss_reference')
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    print(report(args.root, nss_root=args.nss_root))


if __name__ == '__main__':
    main()

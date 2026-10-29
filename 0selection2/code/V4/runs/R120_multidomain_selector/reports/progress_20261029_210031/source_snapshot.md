# Snapshot report source

Source: `code/V4/r120_progress_report.py`

```python
"""Read-only interim R120 validation/ETA report; never performs TEST inference."""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .r120_reporting import (
    NONROUTING_PROBLEMS, _plotting, _table_page, distribution_figure,
    read_json, write_csv, write_json,
)


ROOT = Path(__file__).resolve().parent / 'runs/R120_multidomain_selector'
CST = timezone(timedelta(hours=8))


def short_name(solver):
    parts = str(solver).split('::')[-1].split('/')
    return (' '.join(parts[1:3]) if len(parts) > 2 and parts[1] in ('DAN', 'L2D')
            else parts[1] if len(parts) > 1 else parts[0])


def eta_snapshot(root, status, config, history):
    observed = datetime.fromisoformat(status['updated_utc']).timestamp()
    anchor = root / 'initialization.json'
    started = anchor.stat().st_mtime
    elapsed = max(observed - started, 1.)
    validation_seconds = sum(point['validation']['seconds'] for point in history)
    training_seconds = max(elapsed - validation_seconds, 1.)
    update = status['successful_updates']
    remaining = max(config['max_updates'] - update, 0)
    seconds_per_update = training_seconds / max(update, 1)
    mean_val_seconds = validation_seconds / max(len(history), 1)
    remaining_seconds = remaining * seconds_per_update
    remaining_seconds += math.ceil(remaining / config['eval_every']) * mean_val_seconds
    return dict(observed_utc=status['updated_utc'], successful_updates=update,
        max_updates=config['max_updates'], completed_validation_count=len(history),
        last_completed_validation_update=history[-1]['successful_updates'] if history else None,
        training_start_reference=str(anchor), elapsed_seconds=elapsed,
        measured_training_seconds_per_successful_update=seconds_per_update,
        remaining_training_and_validation_hours_if_full_budget=remaining_seconds / 3600.,
        projected_full_budget_finish_cst=datetime.fromtimestamp(observed + remaining_seconds, CST).isoformat(),
        final_test_lib_report_allowance_minutes=[10, 20],
        estimate_boundary='Observed-rate projection, not a deadline; early stopping may finish sooner. '
                          'Final TEST/LIB/report overhead is an allowance, not a measured R120 result.')


def run(root, output=None):
    root = Path(root).resolve()
    status, config = read_json(root / 'status.json'), read_json(root / 'protocol.json')
    history = read_json(root / 'history.json')
    if not history:
        raise ValueError('No completed validation exists')
    output = Path(output or root / 'reports' /
                  datetime.now(CST).strftime('progress_%Y%m%d_%H%M%S'))
    output.mkdir(parents=True, exist_ok=False)
    latest = history[-1]
    update = latest['successful_updates']
    path = root / 'evaluation/val' / f'update{update:07d}'
    eta = eta_snapshot(root, status, config, history)
    best_observed = max(history, key=lambda point: (
        point['validation']['aggregates']['ALL24']['task_macro_any_best_top1_pct'],
        -point['validation']['aggregates']['ALL24']['task_macro_train_scaled_excess']))
    eta['best_observed_validation_update'] = best_observed['successful_updates']
    eta['best_observed_is_final_lock'] = False
    write_json(output / 'eta.json', eta)
    rows, selection_rows, figures = [], [], []
    for problem in NONROUTING_PROBLEMS:
        with np.load(path / 'by_problem' / (problem + '.npz'), allow_pickle=False) as saved:
            data = {key: saved[key] for key in saved.files}
        metric = latest['validation']['per_problem'][problem]
        best, pred = data['all_best_mask'], data['pred']
        names = data['pool'].tolist()
        selected = np.bincount(pred, minlength=len(names))
        reference = config['references'][problem]
        slot = reference['sbs_slot']
        baseline_top1 = 100. * float(best[:, slot].mean())
        rows.append(dict(problem=problem, validation_update=update, split='val', n=metric['n'],
            objective_direction=metric['objective_direction'], top1_rule=metric['top1_rule'],
            model_top1_pct=metric['any_best_top1_pct'], canonical_top1_pct=metric['canonical_top1_pct'],
            train_fitted_sbs=names[slot], sbs_any_best_top1_pct=baseline_top1,
            top1_gain_pp=metric['any_best_top1_pct'] - baseline_top1,
            selected_deployment_count=int((selected > 0).sum()),
            dominant_selected_solver=names[int(selected.argmax())],
            dominant_selection_pct=100. * float(selected.max()) / len(pred),
            mean_objective=metric['mean_objective'], sbs_objective=metric['sbs_objective'],
            oracle_objective=metric['oracle_objective'],
            actual_regret_positive_oracle_pct=metric['actual_regret_pct'],
            zero_oracle_count=metric['zero_oracle_total'],
            zero_oracle_absolute_excess=metric['zero_oracle_absolute_excess'],
            train_scaled_excess=metric['train_scaled_excess']))
        for i, name in enumerate(names):
            selection_rows.append(dict(problem=problem, split='val', validation_update=update,
                solver=name, short_name=short_name(name), selected_count=int(selected[i]),
                selected_pct=100. * float(selected[i]) / len(pred),
                canonical_winner_count=int((best.argmax(1) == i).sum()),
                co_best_support_count=int(best[:, i].sum())))
        plotted = dict(data, pool=np.asarray([short_name(name) for name in names]))
        figures.append((problem, plotted, metric))
    write_csv(output / 'new_task_validation.csv', rows)
    write_csv(output / 'new_task_validation_selection.csv', selection_rows)
    curve = []
    for point in history:
        for problem, metric in point['validation']['per_problem'].items():
            curve.append(dict(update=point['successful_updates'], problem=problem,
                split='val', top1_pct=metric['top1_pct'], any_best_top1_pct=metric['any_best_top1_pct'],
                ce=metric['ce'], mean_objective=metric['mean_objective'],
                actual_regret_positive_oracle_pct=metric['actual_regret_pct']))
    write_csv(output / 'completed_validation_history.csv', curve)
    macro_rows = []
    for point in history:
        for group, metric in point['validation']['aggregates'].items():
            macro_rows.append(dict(update=point['successful_updates'], group=group,
                is_latest_completed_validation=point is latest,
                best_observed_checkpoint=point is best_observed,
                native_top1_pct=metric['task_macro_native_top1_pct'],
                any_best_top1_pct=metric['task_macro_any_best_top1_pct'],
                train_scaled_excess=metric['task_macro_train_scaled_excess']))
    write_csv(output / 'completed_validation_macros.csv', macro_rows)

    from matplotlib.backends.backend_pdf import PdfPages
    plt = _plotting()
    with PdfPages(output / 'PROGRESS.pdf') as pdf:
        summary = []
        for row in rows:
            summary.append({
                '问题': row['problem'], '例数': row['n'],
                '模型Top1(%)': round(row['model_top1_pct'], 3),
                '固定SBS Top1(%)': round(row['sbs_any_best_top1_pct'], 3),
                '差值(pp)': round(row['top1_gain_pp'], 3),
                '选择方法数': row['selected_deployment_count'],
                '模型目标': round(row['mean_objective'], 3),
                'SBS目标': round(row['sbs_objective'], 3),
                'Oracle目标': round(row['oracle_objective'], 3)})
        _table_page(pdf, f'R120 阶段结果 | 最新完整 validation: update {update:,}',
            list(summary[0]), summary,
            '不是最终选定模型，也不是 TEST 成绩。新六题 Top1=选择任一原生 exact-best；SBS 只由 TRAIN 的平均目标选定。\n'
            'MIS/MKP 目标越大越好，其余越小越好；不同任务的原生目标不能求混合平均。\n'
            f'状态采样: {status["updated_utc"]}; 训练更新: {status["successful_updates"]:,}; '
            f'若跑满预算预计剩余 {eta["remaining_training_and_validation_hours_if_full_budget"]:.2f} 小时，'
            '早停可能提前；另预留 10–20 分钟完成 TEST/LIB/PDF。')
        _table_page(pdf, '已完成的完整验证；当前最好不等于最终锁定',
            ['更新', '原18题 native Top1', '新增6题 any-best Top1', '全部24题 any-best Top1'],
            [{'更新': point['successful_updates'],
              '原18题 native Top1': point['validation']['aggregates']['ROUTING18']['task_macro_native_top1_pct'],
              '新增6题 any-best Top1': point['validation']['aggregates']['NONROUTING6']['task_macro_any_best_top1_pct'],
              '全部24题 any-best Top1': point['validation']['aggregates']['ALL24']['task_macro_any_best_top1_pct']}
             for point in history],
            f'按预先固定规则，当前最好为 update {best_observed["successful_updates"]:,}；训练仍在继续，未锁定最终 TEST。\n'
            '新六题目前各自固定选择一个部署；其中五题等于 TRAIN-SBS。\n'
            'FJSP 固定选择另一个 DAN 权重，Top1 更高但平均 makespan 更差，不能解释为实例选择收益。')
        for problem, data, metric in figures:
            fig = distribution_figure(problem, 'VAL', data, metric, f'R120 update {update}', config['data_version'])
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        for group in ('ROUTING18', 'NONROUTING6', 'ALL24'):
            points = [p['successful_updates'] for p in history]
            key = 'task_macro_native_top1_pct' if group == 'ROUTING18' else 'task_macro_any_best_top1_pct'
            label = group + (' / native' if group == 'ROUTING18' else ' / any-best')
            axes[0].plot(points, [p['validation']['aggregates'][group][key] for p in history],
                         marker='o', label=label)
            axes[1].plot(points, [p['validation']['aggregates'][group]['task_macro_train_scaled_excess']
                                  for p in history], marker='o', label=label)
        for ax in axes:
            ax.set_xlabel('成功更新次数')
            ax.grid(alpha=.2)
            ax.legend(fontsize=8)
        axes[0].set_ylabel('Validation Top1 (%)')
        axes[1].set_ylabel('固定 TRAIN 尺度的超额损失')
        fig.suptitle('R120 已完成 validation；不读取 TEST 预测，不重新选型')
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)
    lines = [f'# R120 interim snapshot: validation update {update}', '',
             'This is an interim validation report, not the final locked TEST result.', '',
             '| Problem | Model Top1 | TRAIN-fitted SBS Top1 | Selected deployments |',
             '| --- | ---: | ---: | ---: |']
    for row in rows:
        lines.append(f'| {row["problem"]} | {row["model_top1_pct"]:.3f}% | '
                     f'{row["sbs_any_best_top1_pct"]:.3f}% | {row["selected_deployment_count"]} |')
    lines += ['', 'All new-task Top1 values are native-objective any-best accuracy.',
              'Canonical winner frequency is only a fixed-order display of exact ties.',
              'The report reads saved validation predictions only; no TEST inference or GPU work.',
              '', 'ETA is an observed-rate full-budget projection; early stopping is unknown.',
              f'Observed status: {json.dumps(eta, ensure_ascii=False)}', '']
    (output / 'SUMMARY.md').write_text('\n'.join(lines), encoding='utf-8')
    source = Path(__file__)
    (output / 'source_snapshot.md').write_text(
        '# Snapshot report source\n\nSource: `code/V4/r120_progress_report.py`\n\n```python\n'
        + source.read_text(encoding='utf-8') + '\n```\n', encoding='utf-8')
    write_json(output / 'report_manifest.json', dict(
        report_kind='interim validation and ETA, not formal TEST',
        created_utc=datetime.now(timezone.utc).isoformat(), validation_update=update,
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        evaluation_source=str(path), protocol_state_hash=latest['validation']['identity']['state_hash'],
        model_training_modified=False, test_predictions_read=False, output=str(output)))
    print(json.dumps(dict(output=str(output), eta=eta, new_task_validation=rows), indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    run(args.root, args.output)

```

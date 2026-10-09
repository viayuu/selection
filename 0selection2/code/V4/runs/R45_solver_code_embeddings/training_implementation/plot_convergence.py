"""R45 convergence zoom and the required per-problem solver comparisons."""

import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
RUNS = {
    'A handcrafted': 'A_handcrafted_seed2',
    'B correct code': 'B_code_seed2',
    'C shuffled code': 'C_shuffled_code_seed2',
}
FIELDS = ('top1', 'ce', 'mean_cost', 'vs_sbs_pct', 'actual_regret_pct', 'true3_accuracy')


def solver_observations():
    if not (ROOT / 'test_results.json').exists():
        return
    from code.V4.performance_analysis import write_csv
    from code.V4.r43_analysis import plot_methods

    results = json.loads((ROOT / 'test_results.json').read_text())
    rows = []
    for name, directory in RUNS.items():
        group = name[0]
        best = json.loads((ROOT / directory / 'best_eval.json').read_text())['val']
        plot_methods(name + ' validation best', best, ROOT / 'curves' / group / 'val_best')
        for split, value in (('val_best', best), ('test', results[group])):
            for problem, metrics in value['per_problem'].items():
                row = dict(model=group, split=split, problem=problem)
                for field in ('n', 'top1', 'top2', 'top3', 'mean_cost', 'best_top1', 'best_top2',
                              'best_top3', 'sbs_cost', 'sbs_name', 'vs_sbs_pct', 'actual_regret_pct'):
                    row[field] = metrics[field]
                for field in ('top1_exceeds_methods', 'top1_not_exceeds_methods',
                              'cost_exceeds_methods', 'cost_not_exceeds_methods'):
                    row[field] = json.dumps(metrics[field])
                rows.append(row)
    write_csv(ROOT / 'observations.csv', rows)


def main():
    histories = {name: json.loads((ROOT / directory / 'history.json').read_text())
                 for name, directory in RUNS.items()}
    start = max(1, min(20, min(len(history) for history in histories.values()) - 4))
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for name, color in zip(RUNS, ('#237b83', '#c44e52', '#8c6d31')):
        for field, ax in zip(FIELDS, axes.flat):
            for split, style in (('train', '--'), ('val', '-')):
                points = [r for r in histories[name] if r['epoch'] >= start and r[split] is not None]
                ax.plot([r['epoch'] for r in points],
                        [r[split]['macro']['macro_' + field] for r in points],
                        color=color, linestyle=style, marker='.', label=f'{name}: {split}')
            ax.set(xlabel='Epoch', title=field)
            ax.grid(alpha=.2)
            ax.ticklabel_format(axis='y', style='plain', useOffset=False)
            ax.legend(fontsize=7)
    fig.suptitle(f'R45 convergence from epoch {start}; train metrics use eval() on the full split')
    fig.tight_layout()
    directory = ROOT / 'curves'
    directory.mkdir(exist_ok=True)
    fig.savefig(directory / 'convergence_zoom.png', dpi=160)
    plt.close(fig)
    solver_observations()


if __name__ == '__main__':
    main()

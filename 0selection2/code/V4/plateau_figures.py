"""CPU-only, rerunnable figures for the frozen plateau diagnosis.

Run in easynco: python -B code/V4/plateau_figures.py
Reads analysis CSVs and a few existing prediction NPZs; never imports a model,
fits a selector, reselects a checkpoint, or writes outside the figure directory.
Missing or incomplete inputs are recorded in figure_status.csv, not imputed.
"""

import os
import sys
from pathlib import Path

sys.dont_write_bytecode = True
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
              'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '2'

DEFAULT_ROOT = Path(__file__).resolve().parent / 'runs/Plateau_diagnosis_R70'
os.environ.setdefault('MPLCONFIGDIR', str(DEFAULT_ROOT / 'figures/.mplconfig'))

import argparse
import csv
from datetime import datetime, timezone
import json
import re
import textwrap

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator, PercentFormatter
import numpy as np
import pandas as pd


PROBLEMS = ('TSP', 'CVRP', 'ATSP', 'OVRP', 'VRPB', 'VRPL', 'VRPTW',
            'OVRPTW', 'OVRPB', 'OVRPL', 'VRPBL', 'VRPBTW', 'VRPLTW',
            'OVRPBL', 'OVRPBTW', 'OVRPLTW', 'VRPBLTW', 'OVRPBLTW')
TEAL, RED, GOLD = '#147D80', '#C44E52', '#B48A32'
INK, GRAY, LIGHT = '#30383C', '#818A90', '#E6E9E8'
MODEL_COLORS = {'R25': GRAY, 'R42A': '#808D89', 'R42B': GOLD,
                'R45A': GOLD, 'R60B': GRAY, 'R70B': TEAL, 'NSS': RED}
MODELS4 = ('R25', 'R42A', 'R42B', 'R45A', 'R70B', 'NSS')
GAPS = ('exact_tie', '(0,0.1]%', '(0.1,0.5]%', '(0.5,1]%', '(1,2]%', '>2%')
GAP_LABELS = ('Exact\ntie', '(0, 0.1]', '(0.1, 0.5]', '(0.5, 1]', '(1, 2]', '>2')
SEQUENTIAL = LinearSegmentedColormap.from_list('neutral_teal', ['#F2F4F3', TEAL])
DIVERGING = LinearSegmentedColormap.from_list('teal_neutral_red', [TEAL, '#F5F5F2', RED])


class Unavailable(Exception):
    """A panel cannot be supported by the current input snapshot."""


def style():
    plt.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': 10,
        'axes.titlesize': 11, 'axes.titleweight': 'semibold',
        'axes.labelsize': 10, 'axes.labelcolor': INK, 'text.color': INK,
        'xtick.labelsize': 9, 'ytick.labelsize': 9,
        'xtick.color': INK, 'ytick.color': INK,
        'axes.edgecolor': '#ADB4B6', 'axes.linewidth': .7,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.facecolor': '#FCFCFB', 'figure.facecolor': 'white',
        'legend.fontsize': 9, 'legend.frameon': False,
        'lines.linewidth': 1.7, 'lines.markersize': 4.5,
        'pdf.fonttype': 42, 'ps.fonttype': 42,
        'savefig.dpi': 150, 'figure.dpi': 110,
    })


def select(frame, **conditions):
    out = frame
    for column, value in conditions.items():
        if column not in out:
            return out.iloc[:0]
        out = out[out[column].isin(value) if isinstance(value, (tuple, list, set))
                  else out[column].eq(value)]
    return out.copy()


def require(frame, columns, reason='required table is not ready'):
    if frame.empty or not set(columns).issubset(frame.columns):
        raise Unavailable(reason + ': ' + ', '.join(columns))
    return frame


def task_mean(frame, keys, columns):
    """Only a complete, unique set of 18 tasks can be called ALL18."""
    require(frame, ['problem', *keys, *columns])
    data = frame[frame.problem.isin(PROBLEMS)]
    if data.duplicated([*keys, 'problem']).any():
        raise Unavailable('duplicate task rows in macro input')
    rows = []
    groups = data.groupby(keys[0] if len(keys) == 1 else keys, dropna=False)
    for key, group in groups:
        if set(group.problem) != set(PROBLEMS):
            continue
        if not np.isfinite(group[columns].to_numpy(dtype=float)).all():
            continue
        key = (key,) if len(keys) == 1 else key
        rows.append(dict(zip(keys, key), **group[columns].mean().to_dict()))
    return pd.DataFrame(rows, columns=[*keys, *columns])


def ordered_models(models):
    def key(model):
        match = re.fullmatch(r'R(\d+)(.*)', model)
        return (int(match[1]), match[2]) if match else (10000, model)
    return sorted(set(models), key=key)


def canvas(title, subtitle, size=(12.6, 7.8), bottom=.13, top=.81):
    fig = plt.figure(figsize=size)
    fig.suptitle(title, x=.06, y=.978, ha='left', fontsize=15, fontweight='semibold')
    fig.text(.06, .932, subtitle, ha='left', va='top', fontsize=9.5, color='#5F686C')
    fig.subplots_adjust(left=.11, right=.96, bottom=bottom, top=top,
                        wspace=.34, hspace=.42)
    return fig


def footer(fig, text):
    fig.text(.06, .023, textwrap.fill(text, 155), fontsize=8.2, color='#626A6D',
             va='bottom', ha='left', linespacing=1.4)


def legend(fig, handles, ncol=4):
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.53, .894),
               ncol=ncol, columnspacing=1.7, handlelength=2.1)


def line_key(label, color, marker='o', linestyle='-'):
    return Line2D([], [], color=color, marker=marker, linestyle=linestyle, label=label)


def clean(ax, axis='x'):
    ax.grid(axis=axis, color=LIGHT, linewidth=.65, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(length=3)


def pct(ax, axis='x', decimals=0):
    getattr(ax, axis + 'axis').set_major_formatter(PercentFormatter(100, decimals=decimals))


def tasks(ax, names=PROBLEMS, highlight=False):
    ax.set_yticks(np.arange(len(names)), names)
    ax.set_ylim(len(names) - .4, -.6)
    if highlight:
        for label, problem in zip(ax.get_yticklabels(), names):
            if problem == 'ATSP':
                label.set_color(RED)
                label.set_fontweight('bold')
            elif problem == 'CVRP' or 'TW' in problem:
                label.set_color(GOLD)
                label.set_fontweight('bold')


def task_matrix(frame, index, columns, value):
    require(frame, [index, columns, value])
    if frame.duplicated([index, columns]).any():
        raise Unavailable('duplicate cells in ' + value)
    return frame.pivot(index=index, columns=columns, values=value)


def heatmap(fig, ax, values, rownames, colnames, label, cmap=SEQUENTIAL,
            vmin=0, vmax=100, annotations=False, norm=None):
    data = np.ma.masked_invalid(np.asarray(values, dtype=float))
    kwargs = dict(cmap=cmap, aspect='auto', interpolation='nearest')
    kwargs.update(dict(norm=norm) if norm is not None else dict(vmin=vmin, vmax=vmax))
    im = ax.imshow(data, **kwargs)
    ax.set_yticks(np.arange(len(rownames)), rownames)
    ax.set_xticks(np.arange(len(colnames)), colnames)
    ax.tick_params(length=0, pad=5)
    for spine in ax.spines.values():
        spine.set_visible(False)
    if annotations:
        for (i, j), value in np.ndenumerate(np.asarray(values)):
            if np.isfinite(value):
                fraction = im.norm(value)
                ax.text(j, i, f'{value:.0f}', ha='center', va='center', fontsize=7.5,
                        color='white' if fraction > .62 else INK)
    colorbar = fig.colorbar(im, ax=ax, fraction=.045, pad=.025)
    colorbar.set_label(label, fontsize=9)
    colorbar.ax.tick_params(labelsize=8)
    return im


class Figures:
    def __init__(self, root, output):
        self.root, self.output = root, output
        self.tables, self.input_status, self.captions, self.status = {}, {}, [], []

    def table(self, name):
        if name not in self.tables:
            path = self.root / (name + '.csv')
            try:
                data = pd.read_csv(path)
                self.input_status[name] = {'rows': len(data), 'bytes': path.stat().st_size}
            except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
                data = pd.DataFrame()
                self.input_status[name] = {'unavailable': str(exc)}
            self.tables[name] = data
        return self.tables[name]

    def result(self, fig, title, point, method, sources):
        return fig, dict(title=title, main_point=point, method_notes=method,
                         sources=';'.join(sources))

    def trend(self):
        data = task_mean(select(self.table('metrics'), split=('val', 'test')),
                         ['model', 'split'], ['top1_pct', 'actual_regret_pct'])
        require(data, ['model'], 'no complete full-18 frozen evaluation')
        models = ordered_models(data.model)
        title = 'Frozen versions: full-18 validation and test trends'
        fig = canvas(title, 'ALL18 = equal weight per task; frozen evaluations only; versions are not independent repetitions.',
                     size=(12.8, 7.8), bottom=.16)
        axes = fig.subplots(2, 1, sharex=True)
        for ax, metric, label in zip(axes, ['top1_pct', 'actual_regret_pct'],
                                     ['Native Top1 (%)', 'Mean relative regret (%)']):
            for split, color, marker in [('val', GOLD, 's'), ('test', TEAL, 'o')]:
                series = select(data, split=split).set_index('model')[metric].reindex(models)
                ax.plot(np.arange(len(models)), series, color=color, marker=marker)
            clean(ax, 'y')
            ax.set_ylabel(label)
            ax.yaxis.set_major_locator(MaxNLocator(5))
            if metric == 'actual_regret_pct':
                ax.set_ylim(bottom=0)
            for model in ('R45A', 'R60B', 'R70B'):
                if model in models:
                    ax.axvspan(models.index(model) - .25, models.index(model) + .25,
                               color=LIGHT, alpha=.55, zorder=0)
        axes[-1].set_xticks(np.arange(len(models)), models, rotation=45, ha='right')
        axes[-1].set_xlabel('Frozen model version (chronological, not an optimization trajectory)')
        legend(fig, [line_key('Validation', GOLD, 's'), line_key('Test', TEAL)])
        footer(fig, 'Regret = mean[100 x (selected cost / instance-best cost - 1)]. Missing validation points remain gaps. '
               'No four-dataset aggregate and no fitted performance ceiling are shown.')
        return self.result(fig, title, 'Compare accuracy and cost across frozen versions without treating versions as replicates.',
                           'Recomputed arithmetic task means from exactly 18 unique problems per model/split; no imputation.', ['metrics'])

    def signal(self):
        data = select(self.table('instance_signal'), model='R70B', split=('val', 'test'), scope='full')
        require(data, ['problem', 'actual_top1_pct', 'shuffled_top1_pct', 'actual_regret_pct', 'shuffled_regret_pct'])
        available = [s for s in ('val', 'test') if not select(data, split=s).empty]
        title = 'R70B instance signal beyond problem and exact size'
        fig = canvas(title, 'Gold task labels: CVRP / time-window variants. Red: ATSP. Each row is one task; no cross-task pooling.',
                     size=(12.8, 10.8), bottom=.105)
        axes = fig.subplots(len(available), 2, squeeze=False)
        for row, split in enumerate(available):
            part = select(data, split=split).set_index('problem').reindex(PROBLEMS)
            for col, (actual, shuffled, xlabel) in enumerate([
                ('actual_top1_pct', 'shuffled_top1_pct', 'Native Top1 (%)'),
                ('actual_regret_pct', 'shuffled_regret_pct', 'Mean relative regret (%)')]):
                ax = axes[row, col]
                y = np.arange(18)
                ax.hlines(y, part[actual], part[shuffled], color='#B8C0C0', linewidth=1.5)
                ax.scatter(part[shuffled], y, c=GOLD, marker='s', s=23, zorder=3)
                ax.scatter(part[actual], y, c=TEAL, s=26, zorder=4)
                if col == 0 and {'shuffle_top1_p025', 'shuffle_top1_p975'}.issubset(part):
                    ax.hlines(y, part.shuffle_top1_p025, part.shuffle_top1_p975, color=GOLD, linewidth=2, alpha=.65)
                tasks(ax, highlight=True)
                ax.set_title(f'{split.title()} | {xlabel.split(" (")[0]}', loc='left')
                ax.set_xlabel(xlabel)
                ax.set_xlim(left=0)
                if col == 0:
                    ax.set_xlim(0, 100)
                    pct(ax)
                clean(ax)
        legend(fig, [line_key('Actual frozen predictions', TEAL),
                     line_key('Within-size permutation mean', GOLD, 's')])
        footer(fig, 'Permutation preserves task, exact true size and pick frequencies. Singleton size groups stay unchanged. '
               'Top1 whiskers: 2.5-97.5% of permutation draws, not a sampling confidence interval; full-split estimates.')
        return self.result(fig, title, 'Instance-conditioned selection is compared with the same picks reassigned within exact size.',
                           'R70B val/test, full scope; singleton dilution retained. Whiskers are empirical permutation quantiles.', ['instance_signal'])

    def margins(self):
        gaps = select(self.table('gap_error_budget'), model='R70B', split='test')
        tolerant = select(self.table('tolerance_curves'), model='R70B', split=('val', 'test'))
        require(gaps, ['problem', 'gap_bin', 'n', 'regret_share_pct'])
        totals = gaps.groupby('problem').n.transform('sum')
        gaps['instance_share_pct'] = gaps.n / totals * 100
        matrix = task_matrix(gaps, 'problem', 'gap_bin', 'instance_share_pct').reindex(index=PROBLEMS, columns=GAPS)
        title = 'Winner-runner margins and cost tolerance answer different questions'
        fig = canvas(title, 'R70B | Left and upper right: test. Lower right: validation / test. Native Top1 is not a cost-tolerant hit.',
                     size=(12.8, 8.6), bottom=.15, top=.83)
        grid = fig.add_gridspec(2, 2, width_ratios=[1.1, 1], hspace=.57, wspace=.45)
        ax = fig.add_subplot(grid[:, 0])
        heatmap(fig, ax, matrix.values, PROBLEMS, GAP_LABELS, 'Task instances (%)', annotations=True)
        ax.set_xlabel('Best-to-second-best cost gap (%)')
        ax.set_title('A  |  Margin distribution by task', loc='left', pad=13)
        ax = fig.add_subplot(grid[0, 1])
        macro = task_mean(gaps, ['gap_bin'], ['instance_share_pct', 'regret_share_pct']).set_index('gap_bin').reindex(GAPS)
        x = np.arange(6)
        ax.bar(x - .18, macro.instance_share_pct, width=.34, color=GOLD, label='Instances')
        ax.bar(x + .18, macro.regret_share_pct, width=.34, color=RED, label='Regret budget')
        ax.set_xticks(x, GAP_LABELS, fontsize=8)
        ax.set_ylabel('Equal-task mean share (%)')
        ax.set_title('B  |  ALL18 margin / regret budget', loc='left')
        legend(fig, [Patch(color=GOLD, label='Margin-bin instances'),
                     Patch(color=RED, label='Margin-bin regret budget')], 2)
        clean(ax, 'y')
        ax = fig.add_subplot(grid[1, 1])
        require(tolerant, ['tolerance_pct', 'acceptable_pct'])
        macro = task_mean(tolerant, ['split', 'tolerance_pct'], ['acceptable_pct', 'strict_top1_pct'])
        for split, color, marker in [('val', GOLD, 's'), ('test', TEAL, 'o')]:
            part = select(macro, split=split).sort_values('tolerance_pct')
            if part.empty:
                continue
            ax.plot(part.tolerance_pct, part.acceptable_pct, color=color, marker=marker, label=split.title())
            ax.axhline(part.strict_top1_pct.iloc[0], color=color, linestyle=':', linewidth=1)
        ax.set_xscale('symlog', linthresh=.01)
        ax.set_xlim(0, 5.2)
        ticks = [0, .01, .1, 1, 5]
        ax.set_xticks(ticks, ['0', '0.01', '0.1', '1', '5'])
        ax.set_ylim(0, 101)
        ax.set_xlabel('Allowed selected-cost regret (%)')
        ax.set_ylabel('Cost-tolerant hit (%)')
        ax.set_title('C  |  ALL18 cost tolerance', loc='left')
        ax.legend(loc='lower right', fontsize=8)
        clean(ax, 'y')
        footer(fig, 'Margin is 100 x (second-best / best - 1); selected-cost regret may exceed this margin. '
               'Panel B averages per-task shares, not raw cross-task cost. Dotted lines in C: native Top1; zero tolerance permits exact-cost ties.')
        return self.result(fig, title, 'Small winner margins do not imply that a wrong selected solver is inexpensive.',
                           'R70B test margin counts; task-normalized regret shares. Tolerance curves use complete 18-task equal means.',
                           ['gap_error_budget', 'tolerance_curves'])

    def topk(self):
        data = select(self.table('topk_error_decomposition'), model='R70B', split='test')
        require(data, ['problem', 'category', 'count', 'regret_contribution_pp'])
        families = select(self.table('error_families'), model='R70B', split='test')
        family_macro = pd.DataFrame()
        if {'category', 'count', 'regret_contribution_pp', 'problem'}.issubset(families):
            family_macro = task_mean(families, ['category'], ['count', 'regret_contribution_pp'])
        has_families = len(family_macro) == 3
        title = ('R70B errors: winner rank and named-family cost budgets' if has_families else
                 'R70B top-k misses: error counts versus regret contributions')
        fig = canvas(title, 'Test | Winner ranks refer to the frozen model score order. Cost contributions retain the full task denominator.',
                     size=(12.8, 10.8) if has_families else (12.8, 8.1), bottom=.12)
        if has_families:
            grid = fig.add_gridspec(2, 2, height_ratios=[2.5, 1], hspace=.55)
            axes = [fig.add_subplot(grid[0, col]) for col in range(2)]
        else:
            axes = fig.subplots(1, 2)
        categories = [('rank2', 'Winner rank 2', TEAL), ('rank3', 'Winner rank 3', GOLD),
                      ('beyond3', 'Winner rank >3', RED)]
        for ax, metric, xlabel in zip(axes, ['count', 'regret_contribution_pp'],
                                      ['Number of wrong selections (instances)', 'Contribution to task mean regret (pp)']):
            left = np.zeros(18)
            for category, label, color in categories:
                part = select(data, category=category).set_index('problem').reindex(PROBLEMS)
                values = part[metric].to_numpy()
                ax.barh(np.arange(18), values, left=left, height=.72, color=color)
                left += values
            tasks(ax)
            ax.set_xlabel(xlabel)
            clean(ax)
        legend(fig, [Patch(color=c, label=l) for _, l, c in categories], 3)
        notes = 'Each task uses its own test denominator. Right: category regret sum / all task instances. Stack totals equal task mean regret, not wrong-only regret; pp = percentage points.'
        sources = ['topk_error_decomposition']
        if has_families:
            names = ['MOEL_MTL', 'other_within_named_family', 'between_named_families']
            family_macro = family_macro.set_index('category').reindex(names)
            for col, (metric, xlabel) in enumerate([
                ('count', 'Mean wrong selections per task (instances)'),
                ('regret_contribution_pp', 'Mean task regret contribution (pp)')]):
                ax = fig.add_subplot(grid[1, col])
                ax.barh(np.arange(3), family_macro[metric], height=.55, color=[TEAL, GOLD, RED])
                tasks(ax, ['RELD MOEL / MTL', 'Other same prefix', 'Between prefixes'])
                ax.set_xlabel(xlabel)
                ax.set_title('ALL18 | Named-family error decomposition', loc='left', fontsize=10)
                clean(ax)
            notes += ' Lower panels: equal-task means, complete ALL18 only. Families are descriptive RELD, MoSES, DIFUSCO and T2T name prefixes, not learned architectural groups.'
            sources.append('error_families')
        footer(fig, notes)
        return self.result(fig, title, 'A frequent rank-2 error can contribute less total regret than a rarer rank >3 error.',
                           'Frozen R70B test; native winner rank; additive regret contributions retain the full task denominator. Optional family summaries use complete equal-task means.', sources)

    def frequency(self):
        data = select(self.table('solver_distributions'), split='test', problem=('TSP', 'CVRP'),
                      model=('R45A', 'R70B', 'NSS'))
        require(data, ['problem', 'solver', 'model', 'winner_pct', 'pick_pct'])
        models = [m for m in ('R45A', 'R70B', 'NSS') if m in set(data.model)]
        title = 'Winner frequencies versus solver picks on TSP and CVRP'
        fig = canvas(title, 'Test | NSS, when present, is the local frozen checkpoint export, not a reconstructed paper result.', size=(12.8, 7.6))
        axes = fig.subplots(1, 2)
        for ax, problem in zip(axes, ('TSP', 'CVRP')):
            part = select(data, problem=problem)
            base = select(part, model='R70B')
            if base.empty:
                base = select(part, model=models[0])
            base = base.sort_values('winner_pct', ascending=False).set_index('solver')
            names = list(base.index)
            for _, group in part.groupby('model'):
                aligned = group.set_index('solver').winner_pct.reindex(names)
                if not np.allclose(aligned, base.winner_pct, atol=1e-7, equal_nan=False):
                    raise Unavailable('winner frequencies disagree across model exports')
            width = .78 / (len(models) + 1)
            y = np.arange(len(names))
            ax.barh(y - .39 + width / 2, base.winner_pct, height=width, color=GRAY)
            for j, model in enumerate(models, 1):
                aligned = select(part, model=model).set_index('solver').pick_pct.reindex(names)
                ax.barh(y - .39 + (j + .5) * width, aligned, height=width, color=MODEL_COLORS[model])
            tasks(ax, names)
            ax.set_title(problem, loc='left')
            ax.set_xlabel('Task test instances (%)')
            ax.set_xlim(0, max(50, float(part[['winner_pct', 'pick_pct']].max().max()) * 1.12))
            pct(ax)
            clean(ax)
        legend(fig, [Patch(color=GRAY, label='Native winner frequency'),
                     *[Patch(color=MODEL_COLORS[m], label=m + ' picks') for m in models]], 4)
        footer(fig, 'Solver rows are ordered by test winner frequency for display only, not fitted into a selector. '
               'A matching marginal pick distribution does not imply correct instance-level assignments. Missing NSS exports are omitted.')
        return self.result(fig, title, 'Compare concentration and under/over-selection without mistaking marginal frequency for instance signal.',
                           'Per-dataset test frequencies; all models must agree on native winner frequencies and pool identities.', ['solver_distributions'])

    def agreement(self):
        data = select(self.table('model_agreement'), split='test')
        require(data, ['model_a', 'model_b', 'problem', 'prediction_agreement_pct', 'error_jaccard'])
        macro = task_mean(data, ['model_a', 'model_b'], ['prediction_agreement_pct', 'error_jaccard'])
        available = set(macro.model_a) | set(macro.model_b)
        models = [m for m in ('R34', 'R42A', 'R43A', 'R45A', 'R46A', 'R47B',
                              'R60B', 'R64B', 'R65B', 'R69B', 'R70B') if m in available]
        if len(models) < 2:
            raise Unavailable('no complete full-18 model pairs')
        shared = select(self.table('shared_errors'), split='test', problem=PROBLEMS)
        consensus = select(self.table('unweighted_consensus'), split='test', problem=PROBLEMS)
        universal = select(self.table('universal_error_solvers'), split='test', problem=PROBLEMS)
        summary_ready = (
            {'models', 'model_count', 'hindsight_union_accuracy_pct', 'problem'}.issubset(shared) and
            {'top1_pct', 'problem'}.issubset(consensus) and
            {'solver', 'count', 'R70_regret_contribution_pp', 'problem'}.issubset(universal) and
            set(shared.problem) == set(PROBLEMS) and shared.models.nunique() == 1 and
            set(consensus.problem) == set(PROBLEMS) and
            'model_count' in consensus and
            set(consensus.model_count) == set(shared.model_count))
        title = 'Frozen models agree on predictions and share error sets'
        fig = canvas(title, 'Test | Every off-diagonal entry is an equal-task mean over all 18 problems; NSS is not a full-18 model.',
                     size=(12.8, 10.8) if summary_ready else (12.8, 7.7),
                     bottom=.14 if summary_ready else .18, top=.83)
        if summary_ready:
            grid = fig.add_gridspec(2, 2, height_ratios=[1.6, 1], hspace=.7, wspace=.55)
            axes = [fig.add_subplot(grid[0, col]) for col in range(2)]
        else:
            axes = fig.subplots(1, 2)
        for ax, metric, heading in zip(axes, ['prediction_agreement_pct', 'error_jaccard'],
                                       ['Prediction agreement (%)', 'Shared wrong sets: Jaccard (%)']):
            matrix = np.full((len(models), len(models)), np.nan)
            np.fill_diagonal(matrix, 100)
            for _, row in macro.iterrows():
                if row.model_a in models and row.model_b in models:
                    i, j = models.index(row.model_a), models.index(row.model_b)
                    matrix[i, j] = matrix[j, i] = row[metric] * (100 if metric == 'error_jaccard' else 1)
            heatmap(fig, ax, matrix, models, models, '%', annotations=True)
            ax.set_xticklabels(models, rotation=50, ha='right')
            ax.set_title(heading, loc='left', pad=12)
        notes = 'Error Jaccard = wrong for both / wrong for either, within each task before averaging; not joint error rate. Blank matrix cells denote incomplete task pairs.'
        sources = ['model_agreement']
        if summary_ready:
            reference = select(self.table('metrics'), model='R70B', split='test', problem=PROBLEMS)
            r70 = task_mean(reference, ['model'], ['top1_pct'])
            vote = task_mean(consensus.assign(group='ALL18'), ['group'], ['top1_pct'])
            union = task_mean(shared.assign(group='ALL18'), ['group'], ['hindsight_union_accuracy_pct'])
            ax = fig.add_subplot(grid[1, 0])
            if not r70.empty and not vote.empty and not union.empty:
                values = [r70.top1_pct.iloc[0], vote.top1_pct.iloc[0], union.hindsight_union_accuracy_pct.iloc[0]]
                ax.bar(np.arange(3), values, color=[TEAL, GOLD, GRAY], width=.6)
                ax.set_xticks(np.arange(3), ['R70B', 'Hard vote', 'Hindsight\nunion'])
                ax.set_ylim(0, 100)
                ax.set_ylabel('Native Top1 / union coverage (%)')
                for i, value in enumerate(values):
                    ax.text(i, value + 1.5, f'{value:.1f}', ha='center', fontsize=9)
                ax.set_title(f'ALL18 | {int(shared.model_count.iloc[0])}-model cohort, NSS excluded', loc='left', fontsize=10)
                clean(ax, 'y')
            else:
                ax.set_visible(False)
            ax = fig.add_subplot(grid[1, 1])
            winners = universal[(universal['count'] > 0) & np.isfinite(universal.R70_regret_contribution_pp)]
            winners = winners.nlargest(6, 'R70_regret_contribution_pp')
            ax.barh(np.arange(len(winners)), winners.R70_regret_contribution_pp, color=RED, height=.6)
            tasks(ax, [f'{row.problem} | {row.solver}' for _, row in winners.iterrows()])
            ax.tick_params(axis='y', labelsize=8)
            ax.set_xlabel('Contribution to that task\'s R70B regret (pp)')
            ax.set_title('Common-wrong winners | six largest contributions', loc='left', fontsize=9.5)
            clean(ax)
            notes += ' Bottom cohort excludes NSS and can exceed the matrix subset. Union is hindsight coverage, not a deployable selector or a ceiling. Hard vote has unmeasured multi-model overhead; right rows are test-sorted diagnostic task/solver pairs, not a pooled macro.'
            sources.extend(['shared_errors', 'unweighted_consensus', 'universal_error_solvers', 'metrics'])
        footer(fig, notes)
        return self.result(fig, title, 'Prediction diversity and shared wrong-instance sets are distinct diagnostics.',
                           '11-version matrix subset where available; complete 18-problem pairs only, equal task weights. Optional cohort vote/union are ALL18-only; hindsight union is not an accuracy ceiling or selector.', sources)

    def corrections(self):
        data = select(self.table('model_comparisons'), split='test', reference='R45A', model=('R60B', 'R70B'))
        require(data, ['problem', 'model', 'corrected', 'harmed', 'delta_regret_pp'])
        models = [m for m in ('R60B', 'R70B') if m in set(data.model)]
        title = 'Changes from R45A: corrected, harmed, and net regret'
        fig = canvas(title, 'Test | Same instances, frozen versions. Net regret = newer minus R45A; negative is cheaper.',
                     size=(12.8, 10.8), bottom=.105)
        axes = fig.subplots(len(models), 2, squeeze=False)
        for row, model in enumerate(models):
            part = select(data, model=model).set_index('problem').reindex(PROBLEMS)
            ax = axes[row, 0]
            ax.barh(np.arange(18), part.corrected, height=.67, color=TEAL)
            ax.barh(np.arange(18), -part.harmed, height=.67, color=RED)
            ax.axvline(0, color=GRAY, linewidth=.8)
            tasks(ax)
            ax.set_xlabel('Harmed (-) / corrected (+) instances')
            ax.set_title(model + ' vs R45A', loc='left')
            clean(ax)
            ax = axes[row, 1]
            values = part.delta_regret_pp
            ax.barh(np.arange(18), values, height=.67, color=[TEAL if v < 0 else RED for v in values])
            if {'regret_boot_lo', 'regret_boot_hi'}.issubset(part):
                ax.hlines(np.arange(18), part.regret_boot_lo, part.regret_boot_hi, color=INK, linewidth=.9)
            ax.axvline(0, color=GRAY, linewidth=.8)
            tasks(ax)
            ax.set_xlabel('Net mean relative regret (pp; lower is better)')
            ax.set_title(model + ' vs R45A', loc='left')
            clean(ax)
        legend(fig, [Patch(color=TEAL, label='Corrected / cheaper'), Patch(color=RED, label='Harmed / dearer')], 2)
        footer(fig, 'Corrected: R45A wrong, newer right. Harmed: R45A right, newer wrong. Net regret also includes cost changes when both are wrong. '
               'Whiskers are descriptive 95% paired-instance bootstrap ranges; one seed, repeated test exposure, no confirmatory significance claim.')
        return self.result(fig, title, 'Accuracy gains and losses need not have symmetric cost consequences.',
                           'Paired frozen predictions against R45A on each task test split; sign of regret retained; exported bootstrap ranges.', ['model_comparisons'])

    def four_datasets(self, cost=False):
        data = self.table('metrics')
        require(data, ['model', 'split', 'problem', 'top1_pct', 'top3_pct', 'mean_cost', 'oracle_cost'])
        settings = [('TSP', 'test', 'TSP test'), ('CVRP', 'test', 'CVRP test'),
                    ('TSP', 'LIB', 'TSPLIB'), ('CVRP', 'LIB', 'CVRPLIB')]
        title = ('Four datasets: selected cost relative to the instance oracle' if cost else
                 'Four datasets: native Top1 and winner coverage in Top3')
        fig = canvas(title, 'Each dataset stands alone. NSS = local frozen checkpoint export, when available. No four-dataset ALL aggregate.',
                     size=(12.8, 8.4), bottom=.13)
        axes = fig.subplots(2, 2)
        handles = ([Patch(color=TEAL, label='Mean selected cost'),
                    Patch(facecolor='white', edgecolor=GOLD, hatch='///', label='Mean oracle cost')]
                   if cost else [Patch(color=TEAL, label='Top1'),
                                 Patch(facecolor='white', edgecolor=GOLD, hatch='///', label='Top3 coverage')])
        legend(fig, handles, 2)
        any_panel = False
        for ax, (problem, split, name) in zip(axes.flat, settings):
            part = select(data, problem=problem, split=split, model=MODELS4)
            if part.empty:
                ax.set_visible(False)
                continue
            any_panel = True
            if part.duplicated(['model']).any():
                raise Unavailable('duplicate four-dataset model result')
            models = [m for m in MODELS4 if m in set(part.model)]
            part = part.set_index('model').reindex(models)
            x = np.arange(len(models))
            first, second = ('mean_cost', 'oracle_cost') if cost else ('top1_pct', 'top3_pct')
            ax.bar(x - .18, part[first], width=.34, color=TEAL)
            ax.bar(x + .18, part[second], width=.34, facecolor='white', edgecolor=GOLD, hatch='///', linewidth=.8)
            ax.set_xticks(x, [m if m != 'NSS' else 'NSS\n(local)' for m in models])
            ax.set_title(f'{name} | n={int(part.n.iloc[0]):,}', loc='left')
            if cost:
                ymax = part[first].max() * 1.19
                ax.set_ylim(0, ymax)
                ax.set_ylabel('Mean cost (dataset native units)')
                for j, (_, record) in enumerate(part.iterrows()):
                    ax.text(j, record[first] + ymax * .025,
                            f'{record[first]:.2f}\n+{record.mean_cost_vs_oracle_pct:.2f}%',
                            ha='center', va='bottom', fontsize=8)
            else:
                ax.set_ylim(0, 109)
                ax.set_ylabel('Task instances (%)')
                for j, (_, record) in enumerate(part.iterrows()):
                    ax.text(j - .18, record[first] + 1.6, f'{record[first]:.1f}', ha='center', fontsize=8)
                    ax.text(j + .18, record[second] + 1.6, f'{record[second]:.1f}', ha='center', fontsize=8)
            clean(ax, 'y')
        if not any_panel:
            raise Unavailable('none of the four evaluation datasets is ready')
        footer(fig, ('Annotations: raw mean cost and 100 x (mean selected / mean oracle - 1), a ratio of means, NOT mean instance regret. '
                     'Cost scales differ across datasets; no pooled raw-cost comparison. ' if cost else
                     'Top3 is winner-in-shortlist coverage, not single-choice accuracy or a free three-solver result. '
                     'No original-paper NSS Top3 is inferred. ') +
               'R70B is omitted from LIB unless actual LIB results exist; unavailable model/dataset cells are not filled.')
        return self.result(fig, title,
                           'Compare available frozen methods separately on TSP test, CVRP test, TSPLIB, and CVRPLIB.',
                           ('Raw dataset-native selected/oracle means; annotation uses ratio of means, not average relative regret.' if cost else
                            'Native Top1 and native winner Top3 coverage; local NSS only; missing cells excluded without substitution.'), ['metrics'])

    def wrong_cost_bins(self, model, problem):
        manifest = select(self.table('prediction_manifest'), model=model, split='test', problem=problem)
        if len(manifest) != 1 or 'path' not in manifest:
            return {}
        path = Path(manifest.path.iloc[0])
        if not path.is_absolute():
            path = Path(__file__).resolve().parents[2] / path
        if not path.is_file():
            return {}
        with np.load(path, allow_pickle=False) as archive:
            if not {'costs', 'logits', 'winner', 'pool_ids'}.issubset(archive.files):
                return {}
            costs = np.asarray(archive['costs'], dtype=float)
            scores = np.asarray(archive['logits'], dtype=float)
            winner = archive['winner']
            ids = archive['pool_ids']
        if (costs.shape != scores.shape or len(winner) != len(scores) or
                scores.shape[1] != len(ids) or not np.isfinite(costs).all() or
                np.isnan(scores).any() or np.isposinf(scores).any() or
                not np.isfinite(scores).any(1).all() or (costs <= 0).any()):
            return {}
        order = np.lexsort((np.broadcast_to(ids, scores.shape), -scores), axis=1)
        pred = order[:, 0]
        exponent = np.exp(scores - scores.max(1, keepdims=True))
        confidence = exponent.max(1) / exponent.sum(1)
        regret = (costs[np.arange(len(pred)), pred] / costs.min(1) - 1) * 100
        wrong = pred != winner
        bins = {}
        for i in range(10):
            mask = (confidence > i / 10) & (confidence <= (i + 1) / 10)
            errors = mask & wrong
            if mask.sum() >= 10 and errors.sum() >= 5:
                bins[i] = float(regret[errors].mean())
        return bins

    def confidence(self):
        data = select(self.table('confidence_bins'), split='test', problem=('TSP', 'CVRP', 'ATSP'),
                      model=('R45A', 'R70B', 'NSS'))
        require(data, ['model', 'problem', 'bin', 'mean_confidence', 'accuracy', 'n'])
        models = [m for m in ('R45A', 'R70B', 'NSS') if m in set(data.model)]
        title = 'Score confidence, empirical correctness, and cost of wrong picks'
        fig = canvas(title, 'Test | Max-softmax scores are descriptive rankings, not validated or calibrated probabilities. R25 is excluded.',
                     size=(12.8, 8.2), bottom=.13)
        axes = fig.subplots(2, 3)
        for col, problem in enumerate(('TSP', 'CVRP', 'ATSP')):
            for model in models:
                part = select(data, model=model, problem=problem).sort_values('bin')
                part = part[part.n >= 10]
                if part.empty:
                    continue
                color = MODEL_COLORS[model]
                axes[0, col].plot(part.mean_confidence * 100, part.accuracy * 100,
                                  color=color, marker='o', linewidth=1.4)
                wrong = self.wrong_cost_bins(model, problem)
                supported = part[part.bin.isin(wrong)]
                if not supported.empty:
                    axes[1, col].plot(supported.mean_confidence * 100,
                                      [wrong[b] for b in supported.bin], color=color, marker='o', linewidth=1.4)
            axes[0, col].plot([0, 100], [0, 100], color='#AFB5B5', linestyle=':', linewidth=1)
            axes[0, col].set_title(problem, loc='left')
            axes[0, col].set_ylim(0, 102)
            axes[0, col].set_ylabel('Observed native Top1 (%)')
            axes[1, col].set_ylabel('Mean regret among wrong picks (%)')
            axes[1, col].set_ylim(bottom=0)
            for ax in axes[:, col]:
                ax.set_xlim(0, 100)
                ax.set_xlabel('Mean max-softmax score in bin (%)')
                clean(ax, 'y')
        legend(fig, [line_key(m + (' (local)' if m == 'NSS' else ''), MODEL_COLORS[m]) for m in models], 3)
        footer(fig, 'Ten fixed score bins; top row requires >=10 instances, bottom also >=5 wrong picks. Wrong-only costs are calculated directly '
               'from existing prediction NPZs, not inferred by dividing bin-average regret by error rate. Dotted diagonal is a visual reference, not a calibration claim.')
        return self.result(fig, title, 'A high score need not imply either frequent correctness or a cheap error.',
                           'Test TSP/CVRP/ATSP only, no macro. Ten score bins, minimum n=10 and wrong n=5; wrong costs use cached FP64 costs/native winners.',
                           ['confidence_bins', 'prediction_manifest'])

    def portfolios(self):
        fixed = select(self.table('fixed_portfolios'), split='test', k=3,
                       objective=('winner_coverage', 'mean_relative_gap'))
        adaptive = select(self.table('metrics'), split='test', model='R70B')
        require(fixed, ['problem', 'objective', 'winner_coverage_pct', 'regret_pct'])
        require(adaptive, ['problem', 'top1_pct', 'top3_pct', 'actual_regret_pct'])
        title = 'Train-fixed Top3 portfolios versus adaptive R70B coverage'
        fig = canvas(title, 'Test | All fixed members were chosen on train only. Three-solver oracle cost is a multi-execution diagnostic.',
                     size=(12.8, 8.4), bottom=.13)
        axes = fig.subplots(1, 2)
        y = np.arange(18)
        adaptive = adaptive.set_index('problem').reindex(PROBLEMS)
        has_adaptive_cost = ('top3_min_regret_pct' in adaptive and
                             np.isfinite(adaptive.top3_min_regret_pct).any())
        for objective, color, marker in [('winner_coverage', GOLD, 's'), ('mean_relative_gap', RED, '^')]:
            part = select(fixed, objective=objective).set_index('problem').reindex(PROBLEMS)
            axes[0].scatter(part.winner_coverage_pct, y, color=color, marker=marker, s=27, zorder=3)
            axes[1].scatter(part.regret_pct, y, color=color, marker=marker, s=27, zorder=3)
        axes[0].scatter(adaptive.top3_pct, y, color=TEAL, marker='o', s=27, zorder=3)
        axes[0].scatter(adaptive.top1_pct, y, facecolors='none', edgecolors=GRAY, marker='D', s=24)
        axes[1].scatter(adaptive.actual_regret_pct, y, facecolors='none', edgecolors=GRAY, marker='D', s=24, zorder=3)
        if has_adaptive_cost:
            axes[1].scatter(adaptive.top3_min_regret_pct, y, color=TEAL, marker='o', s=27, zorder=4)
        for ax in axes:
            tasks(ax)
            clean(ax)
        axes[0].set_xlim(0, 101)
        axes[0].set_xlabel('Native winner coverage (%)')
        axes[0].set_title('A  |  Coverage; shortlist is not Top1 accuracy', loc='left')
        axes[1].set_xlim(left=0)
        axes[1].set_xlabel('Mean relative regret (%)')
        axes[1].set_title('B  |  Best-of-3 cost; single-pick reference' if has_adaptive_cost else
                         'B  |  Fixed best-of-3 vs adaptive single pick', loc='left')
        legend(fig, [line_key('Fixed 3: train winner frequency', GOLD, 's', 'None'),
                     line_key('Fixed 3: train relative gap', RED, '^', 'None'),
                     line_key('R70B: Top3 coverage / best-of-3 cost' if has_adaptive_cost else
                              'R70B: Top3 coverage (left only)', TEAL, 'o', 'None'),
                     Line2D([], [], marker='D', color=GRAY, markerfacecolor='none', linestyle='None',
                            label='R70B: Top1 / single-pick cost')], 2)
        footer(fig, ('Fixed and adaptive Top3 costs are the minimum actual cost among three candidates, requiring three solver executions. '
                     'Open diamonds are the single-pick reference, not the same execution budget. Solver runtime and selector overhead are not compared.'
                     if has_adaptive_cost else
                     'Fixed portfolio costs require all three solver runs; open diamonds show R70B single-pick cost. '
                     'Adaptive Top3 cost is not exported yet and is omitted. This is not an equal-compute deployment comparison.'))
        return self.result(fig, title, 'High shortlist coverage can come from a train-fixed set, while single-pick regret remains a different objective.',
                           'Train-selected k=3 fixed portfolios, held-out test evaluation; adaptive best-of-3 actual cost included when exported. Native Top1/Top3 remain distinct; single-pick cost has a different execution budget.',
                           ['fixed_portfolios', 'metrics'])

    def learning(self):
        data = select(self.table('learning_curves'), model=('R45A', 'R60B', 'R70B'), split=('train', 'val'))
        require(data, ['model', 'split', 'problem', 'epoch', 'ce', 'top1_pct', 'selected_epoch'])
        macro = task_mean(data, ['model', 'split', 'epoch'], ['ce', 'top1_pct'])
        choices = select(self.table('validation_selection_sensitivity'), model=('R45A', 'R60B', 'R70B'))
        models = [m for m in ('R45A', 'R60B', 'R70B') if m in set(macro.model)]
        if not models:
            raise Unavailable('no complete per-task learning histories')
        title = 'Training histories and validation-only checkpoint diagnostics'
        fig = canvas(title, 'ALL18 = equal task means from logged task metrics. Retrospective validation optima do not change the frozen checkpoint.',
                     size=(12.8, 10.8), bottom=.11, top=.80)
        axes = fig.subplots(len(models), 2, squeeze=False)
        markers = [('ce', GOLD, 's', 'Min validation CE'),
                   ('regret_pct', RED, '^', 'Min validation regret'),
                   ('top1_pct', GRAY, 'o', 'Max validation Top1')]
        for row, model in enumerate(models):
            locked = select(data, model=model).selected_epoch.dropna().unique()
            for col, (metric, ylabel) in enumerate([('ce', 'Cross entropy (nats)'), ('top1_pct', 'Native Top1 (%)')]):
                ax = axes[row, col]
                for split, color, linestyle in [('train', GRAY, '--'), ('val', TEAL, '-')]:
                    part = select(macro, model=model, split=split).sort_values('epoch')
                    ax.plot(part.epoch, part[metric], color=color, linestyle=linestyle, linewidth=1.6)
                val = select(macro, model=model, split='val').set_index('epoch')
                if len(locked) == 1 and locked[0] in val.index:
                    ax.axvline(locked[0], color=INK, linewidth=.9, linestyle=':')
                    ax.scatter([locked[0]], [val.loc[locked[0], metric]], marker='D', s=66,
                               facecolors='white', edgecolors=INK, linewidths=1.2, zorder=6)
                for criterion, color, marker, _ in markers:
                    match = select(choices, model=model, criterion=criterion)
                    if len(match) == 1 and match.epoch.iloc[0] in val.index:
                        epoch = match.epoch.iloc[0]
                        ax.scatter([epoch], [val.loc[epoch, metric]], marker=marker, s=27,
                                   c=color, edgecolors='white', linewidths=.5, zorder=7)
                ax.set_title(model + (' | frozen epoch not logged' if len(locked) == 0 else ''), loc='left')
                ax.set_xlabel('Stage epoch (within this run)')
                ax.set_ylabel(ylabel)
                clean(ax, 'y')
        legend(fig, [line_key('Train', GRAY, marker='', linestyle='--'), line_key('Validation', TEAL, marker=''),
                     Line2D([], [], color=INK, marker='D', markerfacecolor='white', linestyle=':', label='Frozen checkpoint'),
                     *[line_key(label, color, marker, 'None') for _, color, marker, label in markers]], 3)
        footer(fig, 'Only training and validation histories appear. Frozen markers come from the logged selected_epoch, never the best test result. '
               'R60B has no exported selected_epoch and is not guessed. CE comparisons are within a run, not a claim that distinct losses have identical score semantics.')
        return self.result(fig, title, 'Validation CE, winner accuracy, and cost can prefer different epochs within a frozen run.',
                           'Complete task-macro histories, local stage epochs; selected_epoch is independent of retrospective validation_selection_sensitivity optima.',
                           ['learning_curves', 'validation_selection_sensitivity'])

    def cost_weighting(self):
        scale = select(self.table('cost_scale_weighting'), model='R70B', split='test')
        trade = select(self.table('solver_tradeoffs'), split='test', problem=('TSP', 'CVRP'))
        require(scale, ['problem', 'cost_quartile', 'absolute_error_share_pct', 'relative_error_share_pct'])
        require(trade, ['problem', 'solver', 'native_winner_pct', 'recoverable_winner_gain_pp'])
        title = 'Cost scale and solver rarity explain different parts of the error budget'
        fig = canvas(title, 'Test | Upper: R70B, all 18 tasks. Lower: TSP / CVRP solver properties, independent of the selector.',
                     size=(12.8, 10.5), bottom=.105, top=.83)
        grid = fig.add_gridspec(2, 2, height_ratios=[1.5, 1], hspace=.5, wspace=.42)
        for col, (metric, heading) in enumerate([
            ('absolute_error_share_pct', 'A  |  Absolute excess-cost share'),
            ('relative_error_share_pct', 'B  |  Relative-regret share')]):
            ax = fig.add_subplot(grid[0, col])
            matrix = task_matrix(scale, 'problem', 'cost_quartile', metric).reindex(index=PROBLEMS, columns=[1, 2, 3, 4])
            heatmap(fig, ax, matrix.values, PROBLEMS, ['Q1 (low)', 'Q2', 'Q3', 'Q4 (high)'], 'Task error budget (%)', annotations=True)
            ax.set_title(heading, loc='left', pad=11)
            ax.set_xlabel('Instance-oracle cost quartile, within task')
        for col, problem in enumerate(('TSP', 'CVRP')):
            ax = fig.add_subplot(grid[1, col])
            part = select(trade, problem=problem).sort_values('recoverable_winner_gain_pp')
            y = np.arange(len(part))
            ax.barh(y - .17, part.native_winner_pct, height=.3, color=GOLD)
            ax.set_yticks(y, part.solver)
            ax.set_xlabel('Native winner frequency (%)', color=GOLD)
            ax.tick_params(axis='x', colors=GOLD)
            ax.set_xlim(left=0)
            twin = ax.twiny()
            twin.scatter(part.recoverable_winner_gain_pp, y + .17, marker='D', color=TEAL, s=27, zorder=4)
            twin.set_xlabel('Unique oracle contribution (pp)', color=TEAL, fontsize=9, labelpad=6)
            twin.tick_params(axis='x', colors=TEAL, labelsize=8)
            twin.set_xlim(left=0)
            twin.spines['top'].set_visible(True)
            twin.spines['right'].set_visible(False)
            ax.set_title(f'{problem}: frequency vs unique cost contribution', loc='left', y=1.32, fontsize=10)
            clean(ax)
        footer(fig, 'Quartiles are defined on held-out oracle costs for description only. Absolute cost and instance-relative regret use different weights. '
               'Unique contribution = mean[max(0, (best cost without solver - solver cost) / oracle) x 100]; ties contribute zero. Bottom axes have different units, not aligned magnitudes.')
        return self.result(fig, title, 'Large-scale instances can dominate absolute excess cost; rare winners can still have unique cost value.',
                           'Within-task evaluation cost quartiles; no fitted policy. Solver contribution measures oracle cost increase if that solver were removed; native frequency can include ties.',
                           ['cost_scale_weighting', 'solver_tradeoffs'])

    def regions(self):
        # Region exports are a separate agent's contract; accept only explicit
        # held-out metrics, never train-leaf purity as transfer performance.
        candidates = ('regions/regions', 'regions/region_transfer', 'regions/region_summary', 'regions/region_metrics')
        data = pd.DataFrame()
        source = ''
        for name in candidates:
            candidate = self.table(name)
            if not candidate.empty:
                data, source = candidate, name
                if name == 'regions/regions':
                    data=select(data, scope='winner_tree_leaf').rename(columns={'region':'region_id'})
                    data['top1_pct']=data['majority_top1']*100
                break
        require(data, ['problem', 'split', 'region_id', 'n'], 'advantage-region CSV contract is not ready')
        aliases = {'winner_coverage_pct': 'coverage_pct', 'region_regret_pct': 'regret_pct'}
        data = data.rename(columns={k: v for k, v in aliases.items() if k in data and v not in data})
        metric = next((m for m in ('top1_pct', 'coverage_pct', 'regret_pct') if m in data), None)
        if metric is None:
            raise Unavailable('region table lacks an explicit held-out performance metric')
        data = select(data, split=('train', 'val', 'test'))
        if set(data.split) <= {'train'}:
            raise Unavailable('region transfer needs a held-out split')
        problems = [p for p in ('TSP', 'CVRP', 'ATSP', 'VRPTW') if p in set(data.problem)]
        if not problems:
            raise Unavailable('region table lacks the representative problem panels')
        title = 'Train-defined advantage regions: held-out transfer'
        fig = canvas(title, 'Regions must be fitted on train only. Observed holdout metrics are not a new trained selector.',
                     size=(12.8, 8.4), bottom=.15)
        axes = fig.subplots(2, 2)
        for ax, problem in zip(axes.flat, problems):
            part = select(data, problem=problem)
            if part.duplicated(['region_id', 'split']).any():
                raise Unavailable('region table has multiple methods; explicit region method schema required')
            train = select(part, split='train').sort_values('n', ascending=False)
            if train.empty:
                ax.set_visible(False)
                continue
            ids = list(train.region_id.iloc[:8])
            for split, color, marker in [('train', GRAY, 'o'), ('val', GOLD, 's'), ('test', TEAL, '^')]:
                values = select(part, split=split).set_index('region_id').reindex(ids)
                if not values.empty:
                    ax.plot(np.arange(len(ids)), values[metric], color=color, marker=marker)
            ax.set_xticks(np.arange(len(ids)), [str(i) for i in ids])
            ax.set_xlabel('Train-defined region ID (largest train support first)')
            ax.set_ylabel(metric.replace('_pct', '').replace('_', ' ').title() + ' (%)')
            ax.set_title(problem, loc='left')
            clean(ax, 'y')
        for ax in list(axes.flat)[len(problems):]:
            ax.set_visible(False)
        legend(fig, [line_key('Train', GRAY), line_key('Validation', GOLD, 's'), line_key('Test', TEAL, '^')], 3)
        footer(fig, 'Up to eight regions per task, chosen by train support only, with held-out gaps retained. '
               'This plot is conditional on the exported region definition and metric; absence of export is recorded as skipped, not treated as absence of signal.')
        return self.result(fig, title, 'Compare train region behavior with validation and test without refitting regions on holdouts.',
                           'Explicit region_id/split metrics; top eight by train n, no holdout ranking; no mixed-task aggregate.', [source])

    def residual(self):
        data=select(self.table('residual_signal'),problem='ALL18')
        require(data,['split','representation','objective','top1_pct','actual_regret_pct'])
        title='Score-conditioned decisions recover a little; added features are mixed'
        fig=canvas(title,'Train-only conditional tables; same 18 tasks, no neural training, no test-selected rule.',size=(12.8,8.7),bottom=.20)
        axes=fig.subplots(2,2)
        labels=['Frozen\nR70B','Score\npair vote','Score\nmean gap','+ Feature\npair vote','+ Feature\nmean gap']
        choices=[('score_summary','pair_accuracy'),('score_summary','pair_relative_cost'),
                 ('score_summary_plus_feature','pair_accuracy'),('score_summary_plus_feature','pair_relative_cost')]
        for row,split in enumerate(('val','test')):
            part=select(data,split=split)
            for col,(metric,base,unit) in enumerate((('top1_pct','frozen_top1_pct','Native Top1 (%)'),('actual_regret_pct','frozen_regret_pct','Mean relative regret (%)'))):
                ax=axes[row,col];values=[part[base].iloc[0]]
                values += [select(part,representation=r,objective=o)[metric].iloc[0] for r,o in choices]
                ax.axhline(values[0],color=GRAY,linestyle='--',lw=1)
                ax.scatter(range(5),values,c=[INK,GOLD,TEAL,GOLD,TEAL],s=32,zorder=3)
                for i,value in enumerate(values):ax.annotate(f'{value:.3f}',(i,value),xytext=(0,9),textcoords='offset points',ha='center',fontsize=8)
                ax.set_xticks(range(5),labels);ax.set_xlim(-.4,4.4)
                lo,hi=min(values),max(values);pad=max((hi-lo)*.45,.005 if col else .2)
                ax.set_ylim(lo-pad,hi+pad);ax.set_ylabel(unit);ax.set_title(split.title(),loc='left');clean(ax,'y')
        coverage=select(data,split='test',representation='score_summary_plus_feature').own_cell_support_pct.iloc[0]
        footer(fig,'Score cells preserve ordered Top2, train-size quartile and margin tercile. Score/feature tables shrink to the same size/pair parent; a constant feature leaves decisions unchanged. '
               f'Minimum support 40, pseudocount 50. Test feature-cell coverage {coverage:.1f}%. Pair vote counts pairwise cost wins, not native full-pool winners. No test rule selection.')
        return self.result(fig,title,'Coarse score-conditioned mean-gap decisions recover a small part of R70 loss; added descriptor gains do not agree across metrics/splits.',
                           'Historical holdouts exploratory; still below R45A reference; no claim of absent residual information; point axes are zoomed.', ['residual_signal'])

    def regional_advantage(self):
        data=self.table('regions/regions')
        require(data,['problem','scope','region','split','ref_advantage_pp_mean'])
        title='Concrete solver preferences transfer across observed instance regions'
        fig=canvas(title,'Positive values favor the first named method; regions and boundaries are fixed using train only.',size=(12.8,7.4),bottom=.21)
        axes=fig.subplots(1,2)
        for ax,(problem,scope,pair) in zip(axes,[('CVRP','demand_quantile','OMNI vs RELD_CVRP'),('ATSP','size_bin','UNICO_MatPOENet vs ICAM_ATSP')]):
            part=select(data,problem=problem,scope=scope);part=part[part.region>=0]
            ids=sorted(part.region.unique())
            ax.axhline(0,color=GRAY,lw=1)
            for split,color,marker in [('train',GRAY,'o'),('val',GOLD,'s'),('test',TEAL,'^')]:
                row=select(part,split=split).set_index('region').reindex(ids)
                ax.plot(range(len(ids)),row.ref_advantage_pp_mean,color=color,marker=marker,label=split)
            ax.set_xticks(range(len(ids)),[str(int(i)) for i in ids]);ax.set_xlabel('Train-defined region ID')
            ax.set_ylabel('Mean (cost B - cost A) / oracle (%)');ax.set_title(problem+': '+pair,loc='left',fontsize=10);clean(ax,'y')
        legend(fig,[line_key('Train',GRAY),line_key('Validation',GOLD,'s'),line_key('Test',TEAL,'^')],3)
        footer(fig,'CVRP IDs 0/1, 2/3, 4/5, 6/7 are size quartiles crossed with low/high normalized demand (train median 0.0468134). '
               'ATSP IDs 0-3 are train-size quartiles. All regions shown, not only significant ones. Means do not imply reliable per-instance prediction; these are historical descriptive transfer checks.')
        return self.result(fig,title,'Preferences can reverse with observed size/demand and recur on held-out instances.',
                           'No holdout threshold selection; no causal attribution to the features; exact rules and supports in regions/regions.csv.', ['regions/regions'])

    def run(self):
        self.output.mkdir(parents=True, exist_ok=True)
        plots = [('01_frozen_full18_trends', self.trend), ('02_within_size_signal', self.signal),
                 ('03_margins_and_tolerance', self.margins), ('04_topk_error_budget', self.topk),
                 ('05_winners_vs_picks', self.frequency), ('06_model_agreement', self.agreement),
                 ('07_corrections_vs_r45', self.corrections), ('08_four_dataset_accuracy', self.four_datasets),
                 ('09_four_dataset_cost', lambda: self.four_datasets(cost=True)),
                 ('10_confidence_and_wrong_cost', self.confidence), ('11_fixed_vs_adaptive_top3', self.portfolios),
                 ('12_learning_and_selection', self.learning), ('13_cost_scale_and_solver_value', self.cost_weighting),
                 ('14_advantage_region_transfer', self.regions),('15_residual_signal',self.residual),
                 ('16_region_advantage',self.regional_advantage)]
        with PdfPages(self.output / 'charts.pdf', metadata={
                'Title': 'Frozen plateau diagnosis through R70',
                'Subject': 'Exploratory CPU-only diagnostics; no training or checkpoint reselection',
                'Author': 'Plateau diagnosis'}) as combined:
            for name, draw in plots:
                try:
                    fig, caption = draw()
                except Unavailable as exc:
                    self.status.append(dict(file=name, status='skipped', reason=str(exc)))
                    for suffix in ('.png', '.pdf'):
                        (self.output / (name + suffix)).unlink(missing_ok=True)
                    plt.close('all')
                    print(f'[skip] {name}: {exc}', flush=True)
                    continue
                if max(fig.get_size_inches()) > 13.0:
                    raise ValueError('unbounded figure dimensions: ' + name)
                fig.savefig(self.output / (name + '.png'), dpi=150)
                fig.savefig(self.output / (name + '.pdf'))
                combined.savefig(fig)
                plt.close(fig)
                self.captions.append(dict(file=name + '.png', pdf_file=name + '.pdf', **caption))
                self.status.append(dict(file=name, status='generated', reason=''))
                print(f'[saved] {name}', flush=True)
        self.write_csv('captions.csv', self.captions,
                       ['file', 'pdf_file', 'title', 'main_point', 'method_notes', 'sources'])
        self.write_csv('figure_status.csv', self.status, ['file', 'status', 'reason'])
        manifest = dict(created_utc=datetime.now(timezone.utc).isoformat(), root=str(self.root),
                        output=str(self.output), figures=len(self.captions), png_dpi=150,
                        cpu_only=True, training=False, checkpoint_reselection=False,
                        macro='equal task weights; exactly 18 unique tasks only',
                        missing_policy='skip unsupported panels/figures; never impute',
                        input_snapshot=self.input_status, status=self.status)
        (self.output / 'render_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        print(f'{len(self.captions)} PNG + {len(self.captions)} PDF; combined: {self.output / "charts.pdf"}', flush=True)

    def write_csv(self, filename, rows, fields):
        with (self.output / filename).open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--output', type=Path, help='Default: ROOT/figures')
    args = parser.parse_args()
    root = args.root.resolve()
    output = (args.output or root / 'figures').resolve()
    style()
    Figures(root, output).run()


if __name__ == '__main__':
    main()

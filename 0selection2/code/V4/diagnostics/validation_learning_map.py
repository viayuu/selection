"""R96: VAL-only common-error map of already locked, completed NEW90k models.

This is a descriptive catalog, not a new model, deployable oracle, ensemble,
checkpoint selector, or accuracy ceiling. No TEST/LIB predictions are opened.
"""

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from code.unified_selector.registry import PROBLEMS


REPO = Path(__file__).resolve().parents[3]
RUNS = REPO / 'code/V4/runs'
ROOT = RUNS / 'R96_validation_learning_map'
DATA = Path('/public/home/shiys/selection_data_new90k_B_filtered_single_reld_mt_seed2_20261018')
TARGETS = RUNS / 'R84_new90k_attention/comparison_artifacts/target_list.json'
REFERENCE = 'R89B'
CATALOG = (
    ('R84A', 'R84_new90k_attention/A_baseline', 'neural'),
    ('R84B', 'R84_new90k_attention/B_attention_direct', 'neural'),
    ('R84C', 'R84_new90k_attention/C_focal', 'neural'),
    ('R84D', 'R84_new90k_attention/D_attention_focal', 'neural'),
    ('R85A', 'R85_zero_selection_research/A_ovr_control', 'neural'),
    ('R85B', 'R85_zero_selection_research/B_gradient_balance', 'neural'),
    ('R85C', 'R85_zero_selection_research/C_cost_profile', 'neural'),
    ('R85D', 'R85_zero_selection_research/D_candidate_subset', 'neural'),
    ('R87A', 'R87_private_method_residual/A_shared_residual', 'neural'),
    ('R87B', 'R87_private_method_residual/B_private_residual', 'neural'),
    ('R88A', 'R88_execution_role_inputs/A_canonical_anchor', 'neural'),
    ('R88B', 'R88_execution_role_inputs/B_execution_anchor', 'neural'),
    ('R89A', 'R89_optimizer_geometry/A_adamw', 'neural'),
    ('R89B', 'R89_optimizer_geometry/B_muon_hybrid', 'neural'),
    ('R91A', 'R91_probability_link_control/A_utility_ce', 'neural'),
    ('R91B', 'R91_probability_link_control/B_utility_probabilities', 'neural'),
    ('R92B', 'R92_broad_learning/B', 'neural'),
    ('R92C', 'R92_broad_learning/C', 'tree'),
    ('R92D', 'R92_broad_learning/D', 'tree'),
    ('R93A', 'R93_plain_classification/A', 'neural'),
    ('R93B', 'R93_plain_classification/B', 'neural'),
)
BOUNDARY = ('VAL-only descriptive overlap of 21 previously locked completed models. '
    'An instance counts as covered if any saved model predicted its native winner. '
    'Identifying that model uses the answer: this is NOT deployable selection, '
    'a new ensemble result, or an accuracy ceiling. No TEST/LIB data, new fit, '
    'new checkpoint, per-task model assembly or inference-rule changes.')


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda: stream.read(2 ** 20), b''):
            value.update(data)
    return value.hexdigest()


def write_csv(path, rows):
    if not rows:
        raise ValueError('A diagnostic table cannot be empty')
    with Path(path).open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def ranked(logits, ids):
    """Same stable descending score, ascending global solver-ID tie rule."""
    logits, ids = np.asarray(logits), np.asarray(ids)
    if logits.ndim != 2 or ids.shape != (logits.shape[1],):
        raise ValueError('Expected row logits and one shared candidate-ID vector')
    if not np.isfinite(logits).all() or len(np.unique(ids)) != len(ids):
        raise ValueError('Legal candidate logits must be finite with distinct IDs')
    order = np.argsort(ids, kind='stable')
    return order[np.argsort(-logits[:, order], axis=-1, kind='stable')]


def overlap(predictions, costs, winner, reference_index):
    predictions, costs, winner = map(np.asarray, (predictions, costs, winner))
    if (predictions.ndim != 2 or costs.ndim != 2 or winner.shape != (costs.shape[0],)
            or predictions.shape[1] != len(winner) or costs.dtype != np.float64
            or not np.isfinite(costs).all() or (costs.min(1) <= 0).any()
            or not np.array_equal(winner, costs.argmin(1))):
        raise ValueError('Native FP64 rows and model predictions must align')
    correct = predictions == winner[None]
    oracle = costs.min(1)
    selected = costs[np.arange(len(winner))[None], predictions]
    regret = (selected - oracle[None]) / oracle[None]
    common = ~correct.any(0)
    base_wrong = ~correct[reference_index]
    base_regret = regret[reference_index]
    total = float(base_regret.sum())
    summary = dict(n=len(winner), reference_top1_pct=float(100 * correct[reference_index].mean()),
        reference_errors=int(base_wrong.sum()), any_model_correct_count=int((~common).sum()),
        any_model_correct_pct=float(100 * (~common).mean()), common_misses=int(common.sum()),
        common_misses_of_reference_errors_pct=float(100 * common.sum() / max(1, base_wrong.sum())),
        common_misses_with_zero_reference_regret=int(np.count_nonzero(common & (base_regret == 0))),
        reference_actual_regret_pct=float(100 * base_regret.mean()),
        reference_regret_on_common_misses_pp=float(100 * base_regret[common].sum() / len(winner)),
        reference_regret_share_common_misses_pct=float(100 * base_regret[common].sum() / total) if total else 0.)
    return summary, correct, regret, selected


def catalog_sources(runs=RUNS):
    records = []
    for name, relative, family in CATALOG:
        directory = Path(runs) / relative
        protocol, lock = read(directory / 'protocol.json'), read(directory / 'final_test_lock.json')
        if (Path(protocol['data_root']).resolve() != DATA.resolve()
                or protocol.get('rows_per_problem') != dict(train=81000, val=4500, test=4500)
                or lock.get('selected_without_test') is not True
                or lock.get('per_problem_assembly') is not False):
            raise ValueError('Catalog requires one locked full18 model on the frozen NEW90k split: ' + name)
        archive = directory / 'evaluation/val/selected_best/rank000.npz'
        if '/evaluation/val/' not in str(archive.resolve()):
            raise PermissionError('Only validation predictions may be opened')
        records.append(dict(model=name, family=family, run=str(directory.resolve()),
            selected_update=lock['selected_update'], checkpoint_sha256=lock['checkpoint_sha256'],
            state_hash=lock['state_hash'], val_prediction_path=str(archive.resolve()),
            val_prediction_sha256=digest(archive), lock_sha256=digest(directory / 'final_test_lock.json'),
            protocol_sha256=digest(directory / 'protocol.json'),
            selected_val_top1_pct=100 * lock['val']['macro_top1']))
    return records


def run(output=ROOT):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    sources = catalog_sources()
    names = [r['model'] for r in sources]
    reference_index = names.index(REFERENCE)
    families = np.asarray([r['family'] for r in sources])
    targets = read(TARGETS)
    if Path(targets['identity']['data_root']).resolve() != DATA.resolve():
        raise ValueError('Use the frozen same-data target36 definition')
    target_set = {(x['problem'], x['method']) for x in targets['targets']}
    if len(target_set) != 36:
        raise ValueError('The prior fixed target list must retain all36 groups')
    source_record = dict(schema_version=1, experiment='R96', kind='CPU-only descriptive diagnostic',
        created_utc=datetime.now(timezone.utc).isoformat(), models=sources, reference=REFERENCE,
        data_root=str(DATA), target_list_sha256=digest(TARGETS), target_definition=targets['identity'],
        source_script_sha256=digest(Path(__file__)), boundary=BOUNDARY, test_or_lib_read=False,
        fitting=False, inference_rule_changed=False, unfinished_models_included=False)
    save(output / 'sources.json', source_record)
    all_correct, all_regret, all_selected, all_target = [], [], [], []
    model_rows, target_rows, target_model_rows, problem_rows = [], [], [], []
    for problem in PROBLEMS:
        truth, predictions, scores = None, [], []
        for source in sources:
            with np.load(source['val_prediction_path'], allow_pickle=False) as archive:
                fields = ('costs', 'winner', 'pool', 'pool_ids', 'indices', 'release_source_indices', 'nodes', 'logits', 'pred')
                data = {key: archive[problem + '__' + key].copy() for key in fields}
            if truth is None:
                truth = {key: data[key] for key in fields if key not in ('logits', 'pred')}
            elif any(not np.array_equal(data[key], truth[key]) for key in truth):
                raise ValueError('Different validation rows/candidates/labels in ' + source['model'] + '/' + problem)
            if len(data['winner']) != 4500 or not np.array_equal(data['indices'], np.arange(4500)):
                raise ValueError('Keep the complete frozen4500 validation rows')
            order = ranked(data['logits'], data['pool_ids'])
            if not np.array_equal(order[:, 0], data['pred']):
                raise ValueError('Saved predictions disagree with the locked ranking rule')
            predictions.append(order[:, 0])
            scores.append(order)
        pred = np.stack(predictions)
        info, correct, regret, selected = overlap(pred, truth['costs'], truth['winner'], reference_index)
        info['problem'] = problem
        info['neural_catalog_covered_pct'] = float(100 * correct[families == 'neural'].any(0).mean())
        info['tree_catalog_covered_pct'] = float(100 * correct[families == 'tree'].any(0).mean())
        common = ~correct.any(0)
        reference_rank = np.argmax(scores[reference_index] == truth['winner'][:, None], axis=1) + 1
        info['common_miss_reference_rank2_count'] = int(np.count_nonzero(common & (reference_rank == 2)))
        info['common_miss_reference_rank3_count'] = int(np.count_nonzero(common & (reference_rank == 3)))
        info['common_miss_reference_rank_after3_count'] = int(np.count_nonzero(common & (reference_rank > 3)))
        target_mask = np.zeros(4500, dtype=bool)
        for slot, method in enumerate(truth['pool'].tolist()):
            if (problem, method) not in target_set:
                continue
            cases = truth['winner'] == slot
            target_mask |= cases
            total = int(cases.sum())
            recovered = int(np.count_nonzero(correct.any(0) & cases))
            target_rows.append(dict(problem=problem, method=method, native_winners=total,
                reference_correct=int(np.count_nonzero(correct[reference_index] & cases)),
                any_model_correct=recovered, all_models_missed=total - recovered,
                catalog_coverage_pct=100 * recovered / max(1, total),
                models_with_recovery=int((correct[:, cases].sum(1) > 0).sum()),
                zero_recall_models=int((correct[:, cases].sum(1) == 0).sum()),
                reference_regret_on_all_missed_pp=float(100 * regret[reference_index, cases & common].sum() / 4500)))
            for model, row_pred, row_correct, row_regret in zip(names, pred, correct, regret):
                picks = row_pred == slot
                true_positive = int(np.count_nonzero(picks & cases))
                target_model_rows.append(dict(model=model, problem=problem, method=method,
                    native_winners=total, selected_count=int(picks.sum()), true_positive=true_positive,
                    false_picks=int(np.count_nonzero(picks & ~cases)),
                    recall_pct=100 * true_positive / max(1, total),
                    precision_pct=100 * true_positive / max(1, int(picks.sum())),
                    false_pick_regret_contribution_pp=float(100 * row_regret[picks & ~cases].sum() / 4500)))
        info['target_native_winners'] = int(target_mask.sum())
        info['target_any_model_correct'] = int(np.count_nonzero(target_mask & correct.any(0)))
        info['target_all_models_missed'] = int(np.count_nonzero(target_mask & common))
        problem_rows.append(info)
        all_correct.append(correct)
        all_regret.append(regret)
        all_selected.append(selected)
        all_target.append(target_mask)
        model_rows.extend(model_statistics(problem, names, correct, regret, selected, reference_index))
    correct, regret, selected = map(lambda parts: np.concatenate(parts, axis=1),
                                   (all_correct, all_regret, all_selected))
    target_mask = np.concatenate(all_target)
    common = ~correct.any(0)
    model_rows.extend(model_statistics('ALL', names, correct, regret, selected, reference_index))
    count_fields = ('n', 'reference_errors', 'any_model_correct_count', 'common_misses',
        'common_misses_with_zero_reference_regret', 'common_miss_reference_rank2_count',
        'common_miss_reference_rank3_count', 'common_miss_reference_rank_after3_count',
        'target_native_winners', 'target_any_model_correct', 'target_all_models_missed')
    overall = {key: sum(row[key] for row in problem_rows) for key in count_fields}
    mean_fields = ('reference_top1_pct', 'any_model_correct_pct', 'reference_actual_regret_pct',
        'reference_regret_on_common_misses_pp', 'neural_catalog_covered_pct', 'tree_catalog_covered_pct')
    overall.update({key: float(np.mean([row[key] for row in problem_rows])) for key in mean_fields})
    overall.update(problem='ALL', common_misses_of_reference_errors_pct=100 * overall['common_misses'] / max(1, overall['reference_errors']),
        reference_regret_share_common_misses_pct=100 * overall['reference_regret_on_common_misses_pp'] / overall['reference_actual_regret_pct'])
    pair_rows = []
    for i, first in enumerate(names):
        for j, second in enumerate(names):
            first_wrong, second_wrong = ~correct[i], ~correct[j]
            pair_rows.append(dict(first=first, second=second,
                both_wrong=int(np.count_nonzero(first_wrong & second_wrong)),
                error_jaccard_pct=float(100 * np.count_nonzero(first_wrong & second_wrong) /
                    max(1, np.count_nonzero(first_wrong | second_wrong))),
                first_correct_second_wrong=int(np.count_nonzero(correct[i] & second_wrong)),
                second_correct_first_wrong=int(np.count_nonzero(correct[j] & first_wrong))))
    summary = dict(experiment='R96', complete=True, model_count=len(names), neural_models=int((families == 'neural').sum()),
        tree_models=int((families == 'tree').sum()), reference=REFERENCE, full18=overall,
        fixed36=dict(native_winners=int(target_mask.sum()), reference_correct=int(np.count_nonzero(correct[reference_index] & target_mask)),
            any_model_correct=int(np.count_nonzero(correct.any(0) & target_mask)),
            all_models_missed=int(np.count_nonzero(common & target_mask))),
        boundary=BOUNDARY, test_or_lib_read=False, fitting=False, model_training=False,
        raw_mean_cost_aggregation='all problems have4500 rows; equal task mean, descriptive across units',
        source_record_sha256=digest(output / 'sources.json'))
    write_csv(output / 'model_metrics.csv', model_rows)
    write_csv(output / 'common_errors_by_problem.csv', [overall, *problem_rows])
    write_csv(output / 'fixed36_coverage.csv', target_rows)
    write_csv(output / 'fixed36_model_recall_and_false_picks.csv', target_model_rows)
    write_csv(output / 'pairwise_error_overlap.csv', pair_rows)
    save(output / 'summary.json', summary)
    markdown(output, summary, model_rows, target_rows)
    render(output, summary, model_rows, problem_rows, target_rows)
    print(json.dumps(summary, indent=2))
    return summary


def model_statistics(problem, names, correct, regret, selected, reference_index):
    baseline, base_regret = correct[reference_index], regret[reference_index]
    rows = []
    for i, model in enumerate(names):
        both_wrong = ~baseline & ~correct[i]
        rows.append(dict(model=model, problem=problem, n=correct.shape[1],
            top1_pct=float(100 * correct[i].mean()), mean_cost=float(selected[i].mean()),
            actual_regret_pct=float(100 * regret[i].mean()),
            correct_count=int(correct[i].sum()), rescued_vs_reference=int(np.count_nonzero(correct[i] & ~baseline)),
            harmed_vs_reference=int(np.count_nonzero(~correct[i] & baseline)),
            both_wrong_cheaper=int(np.count_nonzero(both_wrong & (regret[i] < base_regret))),
            both_wrong_more_expensive=int(np.count_nonzero(both_wrong & (regret[i] > base_regret))),
            regret_delta_vs_reference_pp=float(100 * (regret[i] - base_regret).mean())))
    return rows


def markdown(output, summary, model_rows, target_rows):
    overall, target = summary['full18'], summary['fixed36']
    text = ['# R96: Validation Learning Map', '', BOUNDARY, '',
        'No new neural/tree model was trained. No TEST/LIB predictions or metrics were used.',
        'All21 catalog entries were locked before this analysis. R92A/R94/R95 are excluded because they were unfinished when the catalog was fixed.',
        '', '## Observations', '',
        f"- Reference R89B VAL Top1: {overall['reference_top1_pct']:.6f}%; actual regret: {overall['reference_actual_regret_pct']:.6f}%.",
        f"- {overall['common_misses']:,}/{overall['reference_errors']:,} reference errors ({overall['common_misses_of_reference_errors_pct']:.2f}%) are missed by every catalog model.",
        f"- These common misses contribute {overall['reference_regret_share_common_misses_pct']:.2f}% of R89B VAL regret, not an estimate of irreducible loss.",
        f"- Fixed36 winner cases: {target['native_winners']:,}; R89B correct {target['reference_correct']:,}; any catalog model correct {target['any_model_correct']:,}; all21 miss {target['all_models_missed']:,}.",
        f"- Only {100 * target['any_model_correct'] / target['native_winners']:.2f}% of those target winner cases were recovered by even one catalog entry; {100 * target['all_models_missed'] / target['native_winners']:.2f}% were missed by all21. This is a catalog finding, not an accuracy ceiling.",
        '- Catalog coverage is answer-dependent descriptive evidence. Weak models can add rare recall while harming overall cost; per-model false picks and net regret are reported separately.',
        '- Correlated failure does not prove random labels, a Bayes accuracy ceiling, insufficient inputs, or a universal Encoder defect.',
        '', '## Locked Model Metrics', '',
        '| Model | VAL Top1 (%) | Actual regret (%) | Rescued / Harmed vs R89B |',
        '| --- | ---: | ---: | ---: |']
    for row in model_rows:
        if row['problem'] == 'ALL':
            text.append(f"| {row['model']} | {row['top1_pct']:.4f} | {row['actual_regret_pct']:.6f} | {row['rescued_vs_reference']} / {row['harmed_vs_reference']} |")
    text += ['', '## Interpretation Boundary', '',
        'This catalog asks where existing learned mappings disagree or repeatedly fail. It does not construct a deployable ensemble or choose a model per problem/instance. Current R94 representation-transfer and R95 exposure experiments continue unchanged. Do not expand a small calibration/penalty grid on the strength of this descriptive map.',
        '', 'Winner labels are native FP64 argmin. Exact ties may count as classification misses with zero regret; the separate zero-regret common-miss count prevents confusing those losses.',
        '', 'Reproduce (CPU only, easynco):', '', '```bash',
        'python -B -m code.V4.diagnostics.validation_learning_map', '```', '',
        'Primary report: `VALIDATION_LEARNING_MAP.pdf`; machine-readable tables and source hashes are adjacent.']
    (output / 'RESULTS.md').write_text('\n'.join(text) + '\n')


def render(output, summary, model_rows, problem_rows, target_rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'pdf.fonttype': 42,
        'axes.spines.top': False, 'axes.spines.right': False})
    overall, target = summary['full18'], summary['fixed36']
    footer = ('VAL only / native FP64 winners / frozen NEW90k / no new fit or inference rule.\n'
              'Coverage uses known answers across saved models: NOT deployable performance or an accuracy ceiling.')
    with PdfPages(output / 'VALIDATION_LEARNING_MAP.pdf') as pdf:
        fig = plt.figure(figsize=(11.7, 8.3))
        fig.text(.055, .94, 'R96 | Where do completed models still fail?', fontsize=19, weight='bold', color='#064a4c')
        fig.text(.055, .90, f"21 fixed models | 18 tasks x 4,500 VAL rows | Reference R89B Model Top1={overall['reference_top1_pct']:.2f}%", fontsize=11)
        ax = fig.add_axes([.045, .16, .66, .69]); ax.axis('off')
        rows = [r for r in model_rows if r['problem'] == 'ALL']
        body = [[r['model'], f"{r['top1_pct']:.3f}", f"{r['actual_regret_pct']:.5f}",
                 f"{r['rescued_vs_reference']:,}", f"{r['harmed_vs_reference']:,}"] for r in rows]
        table = ax.table(cellText=body, colLabels=['Model', 'Top1 (%)', 'Regret (%)', 'Rescued', 'Harmed'],
                         cellLoc='right', colLoc='right', bbox=[0, 0, 1, 1])
        table.auto_set_font_size(False); table.set_fontsize(8)
        for (i, j), cell in table.get_celld().items():
            cell.set_edgecolor('#dce5e5')
            if i == 0:
                cell.set_facecolor('#dcebea'); cell.set_text_props(weight='bold')
        note = (f"Common misses across all21\n{overall['common_misses']:,}\n\n"
                f"Share of reference errors\n{overall['common_misses_of_reference_errors_pct']:.2f}%\n\n"
                f"Share of reference regret\n{overall['reference_regret_share_common_misses_pct']:.2f}%\n\n"
                f"Fixed36 winner cases\n{target['native_winners']:,}\n"
                f"Reference correct: {target['reference_correct']:,}\n"
                f"Any model correct: {target['any_model_correct']:,} ({100 * target['any_model_correct'] / target['native_winners']:.2f}%)\n"
                f"All models miss: {target['all_models_missed']:,} ({100 * target['all_models_missed'] / target['native_winners']:.2f}%)")
        fig.text(.75, .82, note, va='top', fontsize=11, linespacing=1.4)
        fig.text(.055, .055, footer, fontsize=8, color='#485858')
        pdf.savefig(fig); plt.close(fig)
        fig, axes = plt.subplots(1, 2, figsize=(11.7, 8.3), gridspec_kw={'width_ratios': [1.5, 1]})
        fig.subplots_adjust(left=.11, right=.96, top=.85, bottom=.17, wspace=.42)
        fig.suptitle('Common failure is measured per task, not just ALL Top1', fontsize=17, weight='bold', color='#064a4c')
        y = np.arange(len(problem_rows))
        common_pct = [100 * r['common_misses'] / r['n'] for r in problem_rows]
        other_pct = [100 * (r['reference_errors'] - r['common_misses']) / r['n'] for r in problem_rows]
        axes[0].barh(y, common_pct, color='#b64e57', label='Missed by all21')
        axes[0].barh(y, other_pct, left=common_pct, color='#16838a', label='At least one model correct')
        axes[0].set_yticks(y, [r['problem'] for r in problem_rows]); axes[0].invert_yaxis()
        axes[0].set_xlabel('Share of all VAL instances (%)'); axes[0].legend(fontsize=8, loc='lower right')
        axes[0].set_title('Partition of R89B classification errors')
        axes[1].barh(y, [r['reference_regret_share_common_misses_pct'] for r in problem_rows], color='#b64e57')
        axes[1].set_yticks(y, []); axes[1].invert_yaxis(); axes[1].set_xlim(0, 100)
        axes[1].set_xlabel('Share of R89B regret (%)'); axes[1].set_title('Regret from common misses')
        fig.text(.055, .055, footer, fontsize=8, color='#485858')
        pdf.savefig(fig); plt.close(fig)
        fig, axes = plt.subplots(1, 2, figsize=(11.7, 8.3))
        fig.subplots_adjust(left=.18, right=.98, top=.85, bottom=.17, wspace=.58)
        fig.suptitle('Fixed36 missed-method groups: coverage is not a new selector', fontsize=16, weight='bold', color='#064a4c')
        for ax, subset in zip(axes, (target_rows[:18], target_rows[18:])):
            y = np.arange(len(subset)); covered = [r['catalog_coverage_pct'] for r in subset]
            ax.barh(y, covered, color='#16838a', label='At least one model correct')
            ax.barh(y, 100 - np.asarray(covered), left=covered, color='#b64e57', label='All21 missed')
            ax.set_yticks(y, [r['problem'] + '/' + r['method'] + f" (n={r['native_winners']})" for r in subset], fontsize=7)
            ax.invert_yaxis(); ax.set_xlim(0, 100); ax.set_xlabel('Share of native-winner cases (%)')
        axes[0].legend(fontsize=7, loc='lower right')
        fig.text(.055, .055, footer + '\nWeak models can add coverage with many false picks; see per-model recall/false-pick CSV.', fontsize=8, color='#485858')
        pdf.savefig(fig); plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT)
    args = parser.parse_args(argv)
    run(args.output)


if __name__ == '__main__':
    main()

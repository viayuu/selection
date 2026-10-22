"""CPU-only saved-prediction reports for R98 expected-relative-cost regression.

No torch, runner, TabICL, context features, targets or query labels are loaded.
R97 pure alignment/paired-recovery helpers are reused without rebinding globals.
Raw predicted gaps, not the descriptive exp(-gap/.01), determine every ranking.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import fcntl
import json
import os
from pathlib import Path
import shutil
import textwrap

import numpy as np

from . import r97_reporting as previous


PROBLEMS = previous.PROBLEMS
LIBS = previous.LIBS
common = previous.common
DEFAULT_ROOT = Path(__file__).resolve().parent / 'runs/R98_performance_context'
SELECTION_RULE = previous.SELECTION_RULE
WINNER_RULE = previous.WINNER_RULE
INFERENCE_RULE = 'argmin raw predicted mean relative gap; ascending global-ID exact ties'
TEMPERATURE = .01
MODEL = 'R98'
METRICS = previous.METRICS
SAVED_KEYS = previous.SAVED_KEYS
NOTES = (
    'One predeclared full18 R98 expected-cost context bank; no SGD, fine-tuning, VAL grid or task-wise assembly.',
    'Official frozen TabICLRegressor: each legal method predicts the mean of raw FP64 relative gap (c-cmin)/cmin.',
    'Every nonconstant method fits all 81,000 historical TRAIN rows in natural order/frequency, sequentially; exact constants alone skip fitting.',
    'Decisions and Top-k use argmin RAW predicted mean gap, with ascending global-ID exact ties. Negative predictions are NOT clipped.',
    'Normalized exp(-predicted gap/0.01) is formal descriptive output only: NOT calibrated probabilities and NEVER the ranking rule.',
    'Fixed n_estimators=1, batch_size=1, norm_methods=[none], kv_cache=False, GPU-only/no CPU offload, Flash attention only.',
    'Input-only canonical R92 357 statistics; immutable R97 task-TRAIN-constant drop. No query labels/costs/winners enter forward.',
    'TRAIN self-context accuracy, epochs, SGD curves and training loss: N/A. Historical TRAIN outcomes are authorized context only.',
    'SBS is fixed on R84 full 81,000 TRAIN rows. Oracle is the per-instance minimum over the unchanged legal candidate pool.',
    'ALL is equal-problem macro across full18. Mixed raw mean_cost is descriptive, never a pooled optimization objective.',
    'vs_SBS / vs_Oracle = 100*(ratio of mean costs - 1); actual regret = mean_i[100*(selected_i-oracle_i)/oracle_i].',
    'NSS reuses saved R79 historical70,114 TRAIN weights; not a new81,000 retrain. Missing stays missing; incompatible evidence fails.',
    'R89B is the main historical control; R84A is secondary. External public pretraining is NOT matched compute.',
    'Old VAL/TEST members informed development. These results are NOT a wholly independent blind TEST.',
    'Fixed36 recovery must include rescued/lost true winners, new false picks, net recovery and cost/regret, not just fewer zero groups.',
    'Timing includes K sequential full-context fits/mean calls. Nested per-method times are breakdowns, NOT added twice to the enclosing call.',
    'Context/target loading and cached-query fingerprint verification are separate. Fresh raw-graph-to-statistics/cache-build latency is excluded for VAL/TEST.',
    'LIB native_input_to_statistics includes input loading/geometry/357 statistics, without opening labels before prediction. Recorded sums are not end-to-end latency.',
)


@dataclass(frozen=True)
class Predictions(previous.Predictions):
    expected_gaps: np.ndarray | None = None


def stable_gap_rank(expected_gaps, pool_ids):
    gaps, ids = np.asarray(expected_gaps), np.asarray(pool_ids)
    if (gaps.ndim != 2 or not len(gaps) or not gaps.shape[1]
            or gaps.dtype.kind != 'f' or not np.isfinite(gaps).all()
            or ids.shape != (gaps.shape[1],) or ids.dtype.kind not in 'iu'
            or np.any(ids < 0) or len(np.unique(ids)) != len(ids)):
        raise ValueError('Require finite raw predicted gaps and distinct stable global IDs')
    order = np.argsort(ids, kind='stable')
    return order[np.argsort(gaps[:, order], axis=1, kind='stable')]


def descriptive_gap_view(expected_gaps):
    """Stable formal exp(-gap/.01); never clip gaps or use this to rank."""
    gaps = np.asarray(expected_gaps, dtype=np.float64)
    if gaps.ndim != 2 or not len(gaps) or not gaps.shape[1] or not np.isfinite(gaps).all():
        raise ValueError('Descriptive gap view requires a nonempty finite matrix')
    with np.errstate(over='ignore', under='ignore'):
        shifted = -(gaps - gaps.min(1, keepdims=True)) / TEMPERATURE
        logits = shifted - np.log(np.exp(shifted).sum(1, keepdims=True))
        probabilities = np.exp(logits)
    return logits, probabilities


def load_predictions(archive, path, prefix=''):
    """Validate raw GAP decisions independently of the descriptive log scores."""
    data = previous.load_predictions(archive, path, prefix, legacy=True)
    required = ('expected_gaps', 'logits', 'scores', 'probabilities', 'pool_ids',
                'nodes', 'indices', 'source_indices', 'release_source_indices')
    if any(prefix + key not in archive for key in required):
        raise ValueError('Missing R98 raw-gap or R97-compatible prediction arrays')
    gaps = np.asarray(archive[prefix + 'expected_gaps'])
    if gaps.dtype != np.float64 or gaps.shape != data.costs.shape:
        raise ValueError('Expected full legal-pool FP64 predicted relative gaps')
    rank = stable_gap_rank(gaps, data.pool_ids)
    np.testing.assert_array_equal(data.pred, rank[:, 0], err_msg='Prediction must argmin raw mean gap')
    if prefix + 'predG' in archive:
        np.testing.assert_array_equal(archive[prefix + 'predG'], gaps)
    logits, probabilities = descriptive_gap_view(gaps)
    for key, expected in (('logits', logits), ('scores', logits), ('probabilities', probabilities)):
        actual = np.asarray(archive[prefix + key])
        if actual.shape != gaps.shape or actual.dtype != np.float64 or np.isnan(actual).any():
            raise ValueError('Invalid formal descriptive gap view: ' + key)
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12, err_msg=key)
    return Predictions(**data.__dict__, expected_gaps=gaps)


def metric_row(data, baseline, problem, split):
    # Negation preserves raw ordering exactly; descriptive softmax may lose near ties.
    row = previous.metric_row(replace(data, logits=-data.expected_gaps), baseline, problem, split)
    return dict(row, model=MODEL, inference_rule=INFERENCE_RULE,
                probabilities_calibrated=False, ranking='raw predicted mean relative gap')


def macro_row(rows, split):
    return dict(previous.macro_row(rows, split), model=MODEL)


def locked_inputs(root):
    """Completion and unchanged bank precede access to saved TEST/LIB labels."""
    root = Path(root).resolve()
    complete = common.read_json(root / 'run_complete.json')
    if complete.get('complete') is not True:
        raise ValueError('Runner is not complete; no final R98 report may be built')
    config = common.read_json(root / 'protocol.json')
    lock = common.read_json(root / 'final_test_lock.json')
    checkpoint = root / 'checkpoint.json'
    bank = common.read_json(checkpoint)
    state, digest = common.json_hash(config), common.file_hash(checkpoint)
    if (config.get('experiment') != 'R98' or config.get('arm') != 'A'
            or lock.get('state_hash') != state or lock.get('checkpoint_sha256') != digest
            or complete.get('checkpoint_sha256') != digest
            or common.resolve_record_path(lock['checkpoint'], root) != checkpoint
            or lock.get('selected_without_test') is not True or lock.get('per_problem_assembly') is not False
            or lock.get('selection_rule') != SELECTION_RULE or bank.get('selection_rule') != SELECTION_RULE
            or bank.get('protocol_sha256') != state or bank.get('weights_updated') is not False
            or any(bank.get(key) != config.get(key) for key in
                   ('contexts', 'targets', 'public_checkpoint_sha256', 'inference_rule'))):
        raise ValueError('Require the unchanged unique frozen R98 bank and pre-TEST lock')
    params, view = config.get('regressor_parameters', {}), config.get('descriptive_probabilities', {})
    if (config.get('optimizer_updates') != 0 or config.get('fine_tuning') is not False
            or config.get('historical_TRAIN_labels_in_context') is not True
            or config.get('current_query_labels_in_forward') is not False
            or config.get('inference_rule') != INFERENCE_RULE
            or config.get('prediction_statistic') != 'official TabICLRegressor mean'
            or config.get('constant_shortcut') != 'exact constant only'
            or params.get('n_estimators') != 1 or params.get('batch_size') != 1
            or params.get('norm_methods') != ['none'] or config.get('kv_cache') is not False
            or config.get('allow_cpu_offload') is not False or config.get('flash_attention_only') is not True
            or view.get('temperature') != TEMPERATURE or view.get('calibrated') is not False
            or view.get('used_for_decisions') is not False):
        raise ValueError('Require one frozen mean-gap regressor bank, not classification or posthoc selection')
    for key in ('contexts', 'targets', 'baselines', 'candidate_pools', 'candidate_pool_ids'):
        if set(config.get(key, {})) != set(PROBLEMS):
            raise ValueError('Incomplete full18 protocol: ' + key)
    for problem in PROBLEMS:
        baseline, context, target = (config[key][problem] for key in ('baselines', 'contexts', 'targets'))
        if (baseline.get('selection_split') != 'train' or baseline.get('train_rows') != 81000
                or context['rows'] != 81000 or target['rows'] != 81000
                or target['methods'] != len(config['candidate_pools'][problem])):
            raise ValueError('Require all81000 historical TRAIN contexts/targets and frozen R84 TRAIN SBS')
    bound = common.read_json(root / 'test_input_lock.json')
    if (bound.get('checkpoint_sha256') != digest or bound.get('labels_loaded') is not False
            or bound.get('created_after_unique_VAL_lock') is not True
            or set(bound.get('inputs', {})) != set(PROBLEMS)):
        raise ValueError('TEST input fingerprints must follow the unique VAL bank lock')
    results = {}
    for split in ('val', 'test'):
        result = common.read_json(root / (split + '_results.json'))
        if result.get('complete') is not True or set(result['per_problem']) != set(PROBLEMS):
            raise ValueError('Require completed full18 ' + split)
        previous._check_state(result, lock, split)
        results[split.upper()] = result
    libs = common.read_json(root / 'lib_results.json')
    for dataset, problem in LIBS.items():
        if libs[dataset].get('complete') is not True or libs[dataset].get('problem') != problem:
            raise ValueError('Require completed native LIB: ' + dataset)
        previous._check_state(libs[dataset], lock, dataset)
    return config, lock, results, libs


def saved_evaluations(root, config, lock, results, libs):
    predictions, metrics, sources = {}, [], []
    for split, result in results.items():
        path = root / 'evaluation' / split.lower() / 'selected_best/rank000.npz'
        sources.append(path)
        rows = []
        with np.load(path, allow_pickle=False) as archive:
            for problem in PROBLEMS:
                data = load_predictions(archive, path, problem + '__')
                previous.validate_population(data, problem, split, config)
                row = metric_row(data, config['baselines'][problem], problem, split)
                previous._check_saved(row, result['per_problem'][problem])
                predictions[split, problem] = data
                rows.append(row)
        overall = macro_row(rows, split)
        previous._check_saved(overall, result['macro'], macro=True)
        metrics.extend([overall, *rows])
    for dataset, problem in LIBS.items():
        path = root / 'lib_evaluation/predictions' / (dataset + '.npz')
        sources.append(path)
        with np.load(path, allow_pickle=False) as archive:
            data = load_predictions(archive, path)
        previous.validate_population(data, problem, dataset, config)
        row = metric_row(data, config['baselines'][problem], problem, dataset)
        previous._check_saved(row, libs[dataset])
        predictions[dataset, problem] = data
        metrics.append(row)
    methods, gaps = [], []
    for row in metrics:
        row.update(checkpoint_sha256=lock['checkpoint_sha256'], data_version=Path(config['data_root']).name,
                   sbs_train_rows=81000)
        if row['problem'] == 'ALL':
            continue
        data = predictions[row['split'], row['problem']]
        for slot, solver in enumerate(data.pool):
            top1, cost = float((data.winner == slot).mean()*100.), float(data.costs[:, slot].mean())
            methods.append(dict(problem=row['problem'], split=row['split'], solver=solver, model=MODEL,
                method_top1_pct=top1, method_mean_cost=cost, model_top1_pct=row['top1_pct'],
                model_mean_cost=row['mean_cost'], model_exceeds_top1=row['top1_pct'] > top1,
                model_cheaper=row['mean_cost'] < cost, winner_rule=WINNER_RULE))
            values = data.expected_gaps[:, slot]
            gaps.append(dict(model=MODEL, problem=row['problem'], split=row['split'], solver=solver,
                n=data.n, predicted_mean_gap=float(values.mean()), predicted_min_gap=float(values.min()),
                predicted_max_gap=float(values.max()), negative_predictions=int((values < 0).sum()),
                clipped=False, calibrated=False, inference_rule=INFERENCE_RULE))
    return predictions, metrics, methods, gaps, sources


def relabel(rows):
    return [dict(row, **{key: MODEL for key in ('model', 'current_model') if key in row}) for row in rows]


def nss_comparisons(config, lock, predictions, root):
    rows, sources = previous.nss_comparisons(config, lock, predictions, root)
    return relabel(rows), sources


def historical_comparisons(config, predictions, r84_root, r89_root, target_path):
    pairs, recovery, summaries, sources = previous.historical_comparisons(
        config, predictions, r84_root, r89_root, target_path)
    return relabel(pairs), relabel(recovery), relabel(summaries), sources


def distribution_rows(predictions, config, lock):
    return relabel(previous.distribution_rows(predictions, config, lock))


def context_resources(root, config):
    """Nested method times are diagnostic; enclosing call is counted once."""
    path, witness_path = root / 'timing.json', root / 'resource_witness.json'
    records, witness = common.read_json(path)['records'], common.read_json(witness_path)
    if (witness.get('complete') is not True or witness.get('labels_loaded') is not False
            or witness.get('accuracy_computed') is not False or witness.get('context_rows') != 81000
            or witness.get('checkpoint_sha256') != config['public_checkpoint_sha256']
            or witness.get('extra_subset_call') is not False):
        raise ValueError('Require the first full TSP VAL unlabeled resource receipt')

    def one(problem, split, stage):
        chosen = [r for r in records if (r['problem'], r['split'], r['stage']) == (problem, split, stage)]
        if len(chosen) != 1:
            raise ValueError(f'Require one {stage} timing: {problem}/{split}')
        return chosen[0]

    for record in records:
        if (not isinstance(record.get('seconds'), (int, float))
                or not np.isfinite(record['seconds']) or record['seconds'] < 0):
            raise ValueError('Invalid recorded context/call timing')
    rows = []
    for problem in PROBLEMS:
        row = dict(model=MODEL, problem=problem, context_rows=81000,
            retained_columns=config['contexts'][problem]['columns'], raw_columns=357,
            methods=len(config['candidate_pools'][problem]), optimizer_updates=0,
            train_self_accuracy='N/A', epochs='N/A', kv_cache=False,
            context_loading_seconds=0., context_read_and_verify_seconds=0., target_read_and_verify_seconds=0.,
            fit_method_seconds=0., predict_method_seconds=0., predict_including_fits_seconds=0.,
            cached_query_inputs_seconds=0., lib_native_input_to_statistics_seconds=None,
            val_test_fresh_graph_to_statistics_seconds=None, recorded_call_seconds=0.,
            sequential_fullcontext_fits=0, sequential_mean_calls=0, exact_constant_shortcuts=0)
        splits = ['val', 'test', *[d for d, p in LIBS.items() if p == problem]]
        for split in splits:
            loading, call = one(problem, split, 'load_context'), one(problem, split, 'predict')
            input_stage = 'native_input_to_statistics' if split in LIBS else 'cached_query_inputs'
            inputs = one(problem, split, input_stage)
            if (loading.get('train_rows') != 81000 or call.get('train_rows') != 81000
                    or call.get('kv_cache') is not False or call.get('context_reencoded') is not True
                    or call.get('excludes_graph_to_statistics_time') is not True
                    or call.get('includes_all_sequential_method_fits') is not True
                    or call.get('methods') != row['methods']):
                raise ValueError('Require full-context timing including all sequential method calls')
            constants = call.get('exact_constant_method_slots', [])
            if (len(set(constants)) != len(constants)
                    or any(not isinstance(v, int) or v < 0 or v >= row['methods'] for v in constants)
                    or constants != loading.get('exact_constant_method_slots')):
                raise ValueError('Invalid exact-constant shortcut accounting')
            expected = set(range(row['methods'])) - set(constants)
            if (call.get('sequential_fullcontext_fits') != len(expected)
                    or call.get('sequential_mean_calls') != len(expected)):
                raise ValueError('Sequential fit/mean count differs from legal nonconstant methods')
            for stage in ('fit_method', 'predict_method'):
                calls = [r for r in records if (r['problem'], r['split'], r['stage']) == (problem, split, stage)]
                slots = [r['method_slot'] for r in calls]
                if len(slots) != len(expected) or set(slots) != expected or any(r['train_rows'] != 81000 for r in calls):
                    raise ValueError('Missing/duplicate per-method full-context timing')
                row[stage + '_seconds'] += sum(r['seconds'] for r in calls)
            row['context_loading_seconds'] += loading['seconds']
            for key in ('context_read_and_verify_seconds', 'target_read_and_verify_seconds'):
                value = loading[key]
                if not np.isfinite(value) or value < 0 or value > loading['seconds']:
                    raise ValueError('Invalid context/target loading breakdown')
                row[key] += value
            row['predict_including_fits_seconds'] += call['seconds']
            row['sequential_fullcontext_fits'] += len(expected)
            row['sequential_mean_calls'] += len(expected)
            row['exact_constant_shortcuts'] += len(constants)
            row['recorded_call_seconds'] += loading['seconds'] + inputs['seconds'] + call['seconds']
            if split in LIBS:
                if inputs.get('labels_loaded') is not False or inputs.get('includes_native_input_loading') is not True:
                    raise ValueError('LIB feature timing must be input-only including native loading')
                row['lib_native_input_to_statistics_seconds'] = inputs['seconds']
            else:
                if inputs.get('fresh_graph_to_statistics_measured') is not False:
                    raise ValueError('VAL/TEST cached features cannot claim fresh statistics latency')
                row['cached_query_inputs_seconds'] += inputs['seconds']
        rows.append(row)
    return rows, records, witness, [path, witness_path]


def plot_distribution(problem, datasets, pool, version, lock):
    """Two series on one axis per dataset; TEST precedes native LIB."""
    from matplotlib.font_manager import FontProperties
    from matplotlib.ticker import PercentFormatter
    font_path = Path('/usr/share/fonts/google-noto-cjk/NotoSansCJK-Regular.ttc')
    font = FontProperties(fname=str(font_path)) if font_path.is_file() else None
    fig = previous._figure((20 if len(datasets) == 2 else 12, 9))
    axes = np.atleast_1d(fig.subplots(1, len(datasets)))
    fig.subplots_adjust(left=.16, right=.95, top=.78, bottom=.18, wspace=.5)
    fig.suptitle('R98 | ' + problem + ' | Frozen Expected-Cost Context', fontsize=18, y=.97)
    fig.text(.035, .91, f'Data: {version}\nBank SHA256: {lock["checkpoint_sha256"]}', fontsize=10)
    for ax, (split, data) in zip(axes, datasets):
        ax.set_title(f'{problem + " TEST" if split == "TEST" else split} | n={data.n:,} | '
            f'\u6a21\u578bTop1={data.top1_pct:.2f}%', fontsize=14, fontproperties=font, pad=18)
        mapping = [data.pool.index(name) for name in pool]
        for field, offset, color, label in (('pred', -.17, '#197d85', 'R98 model selection'),
                                           ('winner', .17, '#c44f58', 'True native winner')):
            counts = np.bincount(getattr(data, field), minlength=len(pool))[mapping]
            percentages = counts / data.n * 100.
            ax.barh(np.arange(len(pool))+offset, percentages, height=.29, color=color, label=label)
            for i, (count, pct) in enumerate(zip(counts, percentages)):
                ax.text(pct+.6, i+offset, f'{count:,} ({pct:.2f}%)', va='center', fontsize=8, color=color)
        ax.set_yticks(np.arange(len(pool)), labels=pool, fontsize=10)
        ax.invert_yaxis()
        ax.set_xlim(0, 122)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.xaxis.set_major_formatter(PercentFormatter(xmax=100))
        ax.set_xlabel('Share of instances (%)')
        ax.grid(axis='x', alpha=.2)
        ax.set_axisbelow(True)
        ax.legend(loc='lower right', fontsize=9)
    fig.text(.035, .10, 'Top1 = 100 * mean(pred == native winner), NOT the largest selection share.\n'
        'Saved native/FP64 ind; argmin RAW predicted mean gap, global-ID ties; LIB original native data.', fontsize=10)
    fig.text(.035, .04, 'Formal exp(-gap/0.01) is descriptive only, not calibrated. No report-time inference or reselection.', fontsize=9)
    return fig


def notes_page(config, lock):
    fig = previous._figure()
    fig.text(.04, .95, 'R98 | Evidence And Expected-Cost Method', fontsize=20, weight='bold', color='#093f45')
    fig.text(.04, .89, '\n\n'.join(textwrap.fill(note, 145) for note in NOTES),
             va='top', fontsize=9, linespacing=1.1)
    fig.text(.04, .075, f'Package: {config["package_version"]} | source: {config["source_revision"]}\n'
        f'Public regressor: {config["public_checkpoint_sha256"]}\nBank: {lock["checkpoint_sha256"]}', fontsize=8)
    return fig


def build_pdfs(root, output, config, lock, predictions, metrics, nss, pairs, recovery, summaries, resources):
    from matplotlib.backends.backend_pdf import PdfPages
    paths = dict(model_statistics_pdf=output / 'MODEL_STATISTICS.pdf',
        selection_distributions_pdf=output / 'selection_distributions.pdf',
        target_recovery_pdf=output / 'target_recovery.pdf')
    temporary = {key: path.with_name(f'.{path.stem}.{os.getpid()}.pdf') for key, path in paths.items()}
    try:
        with PdfPages(temporary['model_statistics_pdf']) as pdf, \
                PdfPages(temporary['selection_distributions_pdf']) as dist, \
                PdfPages(temporary['target_recovery_pdf']) as targets:
            def save(fig, *destinations):
                for destination in destinations:
                    destination.savefig(fig)
                fig.clear()

            save(notes_page(config, lock), pdf)
            keys = ['problem', 'n', 'top1_pct', 'top2_pct', 'top3_pct', 'mean_cost', 'sbs_cost',
                    'oracle_cost', 'vs_sbs_pct', 'vs_oracle_pct', 'actual_regret_pct']
            headers = ['Task', 'n', 'Top1 %', 'Top2 %', 'Top3 %', 'mean_cost', 'SBS cost',
                       'Oracle', 'vs_SBS %', 'vs_Oracle %', 'Actual regret %']
            for split in ('VAL', 'TEST'):
                rows = [r for r in metrics if r['split'] == split]
                save(previous._table_page('R98 | ' + split + ' | ALL And Full18', headers,
                    [[r[key] for key in keys] for r in rows],
                    ['Native/FP64 ind. Same bank on VAL/TEST/LIB. Raw predicted mean-gap ranking, no calibration.',
                     'SBS: R84 full81000 TRAIN. ALL: equal-problem macro; mixed raw costs descriptive.',
                     'vs_* use ratios of means; actual regret averages per-instance relative regrets.']), pdf)
            rows = [r for r in metrics if r['split'] in LIBS]
            save(previous._table_page('R98 | Native LIB Metrics', ['Dataset', *headers[1:]],
                [[r['split'], *[r[key] for key in keys[1:]]] for r in rows],
                ['Same full18 bank; no per-dataset tuning; native/FP64 winner labels.']), pdf)
            save(previous._table_page('R98 | NSS Four-Dataset Historical Comparison',
                ['Dataset', 'Status', 'n', 'NSS Top1 %', 'R98 Top1 %', 'NSS cost', 'R98 cost'],
                [[r['dataset'], r['status'], r['n'], r['nss_top1_native_pct'], r['current_top1_native_pct'],
                  r['nss_mean_cost'], r['current_mean_cost']] for r in nss],
                ['NSS saved R79 historical70114 TRAIN weights, not a new81000 retrain. Same current R98 bank for all four.',
                 'Rows/costs/winner labels align by instance/candidate; missing is explicit, incompatible evidence fails.']), pdf)
            for baseline in ('R89B', 'R84A'):
                pair = next(r for r in pairs if r['baseline'] == baseline and r['problem'] == 'ALL')
                summary = next(r for r in summaries if r['baseline'] == baseline)
                save(previous._table_page('R98 | Full18 TEST vs ' + baseline,
                    ['Status', 'Corrected', 'Harmed', 'Net correct', 'Top1 delta pp', 'Cost delta', 'Regret delta pp'],
                    [[pair['status'], pair.get('corrected'), pair.get('harmed'), pair.get('net_correct'),
                      pair.get('top1_delta_pp'), pair.get('mean_cost_delta'), pair.get('actual_regret_delta_pp')]],
                    ['R89B is main; R84A secondary. Historical, not newly compute-matched controls.',
                     'Positive cost/regret delta is worse; ALL equal-problem macro. Fixed36: ' + summary['status']]), pdf, targets)
                rows = [r for r in recovery if r['baseline'] == baseline]
                for start in range(0, len(rows), 18):
                    save(previous._table_page('R98 | Fixed36 TEST Recovery vs ' + baseline,
                        ['Task / Method', 'Winners', 'Rescued', 'Lost', 'Net', 'New false', 'Recall %', 'Regret delta pp'],
                        [[r['problem']+'/'+r['solver'], r['winner_count'], r['rescued'], r['lost'], r['net_recovery'],
                          r['new_false_pick_count'], r['recall_pct'], r['target_regret_delta_pp']]
                         for r in rows[start:start+18]],
                        ['Fixed36 R84A VAL targets, never redefined from current TEST picks.',
                         'CSV includes precision/total false/new-false regret; contributions normalized by whole task n.']), pdf, targets)
            save(previous._table_page('R98 | Sequential Full-Context Resource Cost',
                ['Task', 'TRAIN n', 'Methods', 'Fits', 'Mean calls', 'Load s', 'Fit s', 'Mean s', 'Outer call s', 'Recorded sum s'],
                [[r['problem'], r['context_rows'], r['methods'], r['sequential_fullcontext_fits'],
                  r['sequential_mean_calls'], r['context_loading_seconds'], r['fit_method_seconds'],
                  r['predict_method_seconds'], r['predict_including_fits_seconds'], r['recorded_call_seconds']] for r in resources],
                ['Fit/mean breakdowns are nested in outer calls: do NOT add them twice. Exact constants alone skip fitting.',
                 'Recorded sum = context/target load + cached or native query inputs + all sequential fits/mean calls.',
                 'VAL/TEST exclude fresh statistics/cache build; LIB includes native loading/geometry/statistics. Not end-to-end.']), pdf)
            for problem in PROBLEMS:
                datasets = [('TEST', predictions['TEST', problem])]
                datasets += [(d, predictions[d, problem]) for d, p in LIBS.items() if p == problem]
                save(plot_distribution(problem, datasets, config['candidate_pools'][problem],
                    Path(config['data_root']).name, lock), pdf, dist)
        for key, path in paths.items():
            temporary[key].replace(path)
    finally:
        for path in temporary.values():
            path.unlink(missing_ok=True)
    destination = root / f'.RESULTS.{os.getpid()}.pdf'
    try:
        shutil.copyfile(paths['model_statistics_pdf'], destination)
        destination.replace(root / 'RESULTS.pdf')
    finally:
        destination.unlink(missing_ok=True)
    return dict(paths, results_pdf=root / 'RESULTS.pdf')


def write_markdown(root, config, lock, metrics, nss, pairs, summaries):
    lines = ['# R98 Frozen Expected-Cost Context', '',
        'Primary delivery: [RESULTS.pdf](RESULTS.pdf). CSV and provenance: full_model_statistics/.', '',
        'Bank SHA256: `' + lock['checkpoint_sha256'] + '`', '',
        '## Method And Evidence Boundaries', '', *['- ' + note for note in NOTES], '',
        '## Full Metrics', '']
    keys = ['problem', 'split', 'n', 'top1_pct', 'mean_cost', 'sbs_cost', 'oracle_cost',
            'vs_sbs_pct', 'vs_oracle_pct', 'actual_regret_pct']
    lines.extend(previous._md_table(keys, [[r[key] for key in keys] for r in metrics]))
    lines.extend(['', '## NSS Four Datasets', ''])
    lines.extend(previous._md_table(
        ['Dataset', 'Status', 'NSS Top1 %', 'R98 Top1 %', 'NSS cost', 'R98 cost'],
        [[r['dataset'], r['status'], r['nss_top1_native_pct'], r['current_top1_native_pct'],
          r['nss_mean_cost'], r['current_mean_cost']] for r in nss]))
    lines.extend(['', '## Historical Full18 TEST', ''])
    lines.extend(previous._md_table(
        ['Reference', 'Status', 'Top1 delta pp', 'Cost delta', 'Regret delta pp'],
        [[r['baseline'], r['status'], r.get('top1_delta_pp'), r.get('mean_cost_delta'),
          r.get('actual_regret_delta_pp')] for r in pairs if r['problem'] == 'ALL']))
    lines.extend(['', 'Fixed36 summaries: ' + json.dumps(summaries, sort_keys=True), ''])
    path = root / 'RESULTS.md'
    temporary = root / f'.RESULTS.{os.getpid()}.md'
    try:
        temporary.write_text('\n'.join(lines), encoding='utf-8')
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def report(args):
    if str(getattr(args, 'device', 'cpu')) != 'cpu':
        raise ValueError('R98 final reporting is CPU-only; use --device cpu')
    root = Path(args.root).resolve()
    config, lock, results, libs = locked_inputs(root)
    output, status = root / 'full_model_statistics', root / 'postprocess_status.json'
    with (root / 'postprocess.lock').open('a') as claim:
        fcntl.flock(claim, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            sources = [root / name for name in ('protocol.json', 'checkpoint.json', 'final_test_lock.json',
                'test_input_lock.json', 'val_results.json', 'test_results.json', 'lib_results.json', 'run_complete.json',
                'timing.json', 'resource_witness.json', 'evaluation/val/selected_best/rank000.npz',
                'evaluation/test/selected_best/rank000.npz', 'lib_evaluation/predictions/TSPLIB.npz',
                'lib_evaluation/predictions/CVRPLIB.npz', 'wandb_receipt.json')]
            hashes = {str(path): common.file_hash(path) for path in sources}
            common.write_json(status, dict(experiment='R98', complete=False, phase='loading_saved_predictions',
                                           output=str(output), updated_utc=common.utc()))
            predictions, metrics, methods, gap_rows, prediction_sources = saved_evaluations(root, config, lock, results, libs)
            nss_root = Path(getattr(args, 'nss_root', None) or previous.DEFAULT_R84 / 'nss_reference').resolve()
            r84_root = Path(getattr(args, 'r84_root', None) or previous.DEFAULT_R84 / 'A_baseline').resolve()
            r89_root = Path(getattr(args, 'r89_root', None) or previous.DEFAULT_R89).resolve()
            target_path = Path(getattr(args, 'target_list', None) or previous.DEFAULT_R84 / 'comparison_artifacts/target_list.json').resolve()
            nss, nss_sources = nss_comparisons(config, lock, predictions, nss_root)
            pairs, recovery, summaries, history_sources = historical_comparisons(
                config, predictions, r84_root, r89_root, target_path)
            resources, timing, witness, resource_sources = context_resources(root, config)
            wandb = common.read_json(root / 'wandb_receipt.json')
            if wandb.get('enabled') is not True or wandb.get('mode') != 'offline':
                raise ValueError('Require the declared offline W&B receipt')
            for path in dict.fromkeys([*prediction_sources, *nss_sources, *history_sources, *resource_sources]):
                hashes.setdefault(str(path), common.file_hash(path))
            output.mkdir(parents=True, exist_ok=True)
            artifacts = {}
            tables = dict(metrics_by_problem=metrics, selection_distribution=distribution_rows(predictions, config, lock),
                single_method_comparison=methods, expected_gap_summary=gap_rows, nss_comparison=nss,
                paired_comparison=pairs, target_recovery=recovery or [dict(status='missing', reason='See target_recovery_summary.csv')],
                target_recovery_summary=summaries, context_resource_summary=resources, timing_records=timing)
            common.write_json(status, dict(experiment='R98', complete=False, phase='rendering_cpu_reports',
                                           output=str(output), updated_utc=common.utc()))
            for name, rows in tables.items():
                path = output / (name + '.csv')
                common.write_csv(path, rows)
                artifacts[name + '_csv'] = str(path)
            artifacts.update({key: str(path) for key, path in build_pdfs(root, output, config, lock,
                predictions, metrics, nss, pairs, recovery, summaries, resources).items()})
            artifacts['results_md'] = str(write_markdown(root, config, lock, metrics, nss, pairs, summaries))
            for path, digest in hashes.items():
                if common.file_hash(path) != digest:
                    raise ValueError('Report source changed during rendering: ' + path)
            if locked_inputs(root) != (config, lock, results, libs):
                raise ValueError('Locked R98 bank changed during reporting')
            provenance = dict(experiment='R98', arm='A', state_hash=lock['state_hash'],
                checkpoint_sha256=lock['checkpoint_sha256'], selection_rule=SELECTION_RULE,
                input_sha256=hashes, regressor_parameters=config['regressor_parameters'],
                contexts=config['contexts'], targets=config['targets'], inference_rule=INFERENCE_RULE,
                public_checkpoint_sha256=config['public_checkpoint_sha256'],
                package_version=config['package_version'], source_revision=config['source_revision'],
                public_checkpoint_revision=config.get('public_checkpoint_revision'), runtime_wheels=config.get('runtime_wheels', {}),
                resource_witness=witness, wandb_receipt=wandb, notes=list(NOTES),
                inference_performed=False, reference_retraining=False, train_self_accuracy=None,
                learning_curves=None, probabilities_calibrated=False, ranking_uses_raw_expected_gaps=True)
            path = output / 'provenance.json'
            common.write_json(path, provenance)
            artifacts['provenance_json'] = str(path)
            result = dict(experiment='R98', arm='A', complete=True, phase='complete', output=str(output),
                artifacts=artifacts, checkpoint_sha256=lock['checkpoint_sha256'], state_hash=lock['state_hash'],
                updated_utc=common.utc())
            common.write_json(status, result)
            return result
        except Exception as error:
            common.write_json(status, dict(experiment='R98', complete=False, phase='failed', error=str(error),
                                           output=str(output), updated_utc=common.utc()))
            raise


def parser():
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument('command', choices=('report',))
    value.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    value.add_argument('--device', choices=('cpu',), default='cpu')
    value.add_argument('--context-root', type=Path)
    value.add_argument('--target-root', type=Path)
    value.add_argument('--feature-root', type=Path)
    value.add_argument('--data-root', type=Path)
    value.add_argument('--prepared-root', type=Path)
    value.add_argument('--protocol-lock', type=Path)
    value.add_argument('--nss-root', type=Path)
    value.add_argument('--r84-root', type=Path)
    value.add_argument('--r89-root', type=Path)
    value.add_argument('--target-list', type=Path)
    return value


def main(argv=None):
    result = report(parser().parse_args(argv))
    print(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    main()

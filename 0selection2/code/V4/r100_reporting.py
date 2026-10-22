"""CPU-only R100 reports from saved predictions, without inference or reselection.

Only R97's pure native-metric, alignment, NSS and recovery helpers are reused.
No monkeypatches, old result writes, context features or query-label sources.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import fcntl
import json
import os
from pathlib import Path
import shutil
import textwrap

import numpy as np

from . import r97_reporting as previous
from .r100_data import LABEL_RULE, legal_pairs
from .r100_model import INFERENCE_RULE, PROBABILITY_SEMANTICS, formal_score_view, mean_opponent_win, stable_mean_win_rank


PROBLEMS, LIBS, common = previous.PROBLEMS, previous.LIBS, previous.common
MODEL = 'R100'
SELECTION_RULE = previous.SELECTION_RULE
WINNER_RULE = previous.WINNER_RULE
NOTES = (
    'One predeclared full18 R100 pairwise-binary frozen TabICL bank. No SGD, fine-tuning, epochs, TRAIN accuracy or VAL aggregation search.',
    'For every legal pair: ALL historical TRAIN rows in natural frequency/order, including third-method full-pool winners. Only exact native FP64 pair ties are excluded.',
    'Binary labels: 0 first method wins, 1 second wins. Pair orientation follows ascending stable global IDs; no FP32 reconstruction or full-winner filtering.',
    'Constant shortcut ONLY for exactly one strict class, or no strict rows (.5 neutral). Support/tie/class counts are identity-bound and reported.',
    'P[i,j] predicts i beating j conditional on a strict TRAIN relation; P[j,i]=1-P[i,j]. These are NOT full-pool winner probabilities.',
    'Decision: argmax mean opponent win probability, fixed expected-rank/Borda; ascending global-ID exact ties. No coupling, calibration, heuristic threshold or cost-sensitive claim.',
    'Formal log-softmax(mean-win) is for descriptive CE/report interoperability only, not calibrated probability or the ranking rule.',
    'Each nonconstant pair independently fits its strict TRAIN context and predicts the whole named dataset; only one pair classifier is resident. Runtime scales with K*(K-1)/2 contexts.',
    'Original R97 public classifier weights/runtime: max_classes=10 is sufficient for binary2. n_estimators=1, norm=none, batch_size=1, seed2, AMP, kv_cache=False, GPU-only, Flash-only.',
    'Immutable R97 input-only statistics and TRAIN-constant drop. Query labels/winners/costs never enter fit/forward; any query base ID in complete TRAIN is rejected BEFORE pair filtering.',
    'SBS fixed on R84 full81000 TRAIN. Oracle is the per-instance legal-pool minimum. ALL is equal-problem macro; mixed raw costs descriptive only.',
    'vs_SBS/vs_Oracle are percent ratios of mean costs; actual regret is mean per-instance percent regret. Native/FP64 ind is never replaced by FP32 argmin.',
    'Same current bank on VAL, full TEST and native LIB. Historical TEST members informed research, not wholly blind. Public pretraining compute is unmatched.',
    'NSS reuses saved R79 historical70114 TRAIN weights, not a new81000 retrain. Missing stays missing; incompatible saved evidence fails.',
    'R89B main, R84A secondary; fixed36 targets frozen from R84A VAL. Report rescued/lost true winners, false picks, raw false-pick cost and regret, not zero-count changes alone.',
    'Per-pair fit/predict times are NESTED in enclosing calls and are not added twice. VAL/TEST use cached inputs: fresh raw-graph statistics/cache-build latency is NOT measured.',
    'LIB native_input_to_statistics includes native loading/geometry/statistics without query-label loading. Recorded sums are not full end-to-end latency.',
)


@dataclass(frozen=True)
class Predictions(previous.Predictions):
    mean_win: np.ndarray | None = None
    pair_probabilities: np.ndarray | None = None


def load_predictions(archive, path, prefix=''):
    data = previous.load_predictions(archive, path, prefix, legacy=True)
    required = ('pair_win_probabilities', 'mean_opponent_win', 'scores', 'logits', 'formal_score_softmax',
        'pool_ids', 'nodes', 'indices', 'source_indices', 'release_source_indices')
    if any(prefix+key not in archive for key in required):
        raise ValueError('Missing R100 raw pair/mean-win arrays or immutable row IDs')
    p = np.asarray(archive[prefix+'pair_win_probabilities'])
    mean = mean_opponent_win(p)
    saved = np.asarray(archive[prefix+'mean_opponent_win'])
    if p.shape != (data.n, len(data.pool), len(data.pool)) or saved.dtype != np.float64:
        raise ValueError('R100 pair arrays must cover every legal pair and query')
    np.testing.assert_array_equal(saved, mean)
    np.testing.assert_array_equal(np.asarray(archive[prefix+'scores']), mean)
    np.testing.assert_array_equal(data.pred, stable_mean_win_rank(mean, data.pool_ids)[:, 0])
    logits, formal = formal_score_view(mean)
    for key, expected in (('logits', logits), ('formal_score_softmax', formal)):
        actual = np.asarray(archive[prefix+key])
        if actual.shape != expected.shape or actual.dtype != np.float64 or not np.isfinite(actual).all():
            raise ValueError('Invalid formal mean-win view: ' + key)
        np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-13)
    if prefix+'probabilities' in archive:
        raise ValueError('Do not store formal score views as winner probabilities')
    return Predictions(**data.__dict__, mean_win=mean, pair_probabilities=p)


def metric_row(data, baseline, problem, split):
    row = previous.metric_row(replace(data, logits=data.mean_win), baseline, problem, split)
    return dict(row, model=MODEL, inference_rule=INFERENCE_RULE,
        probability_semantics=PROBABILITY_SEMANTICS, calibrated_winner_probability=False,
        formal_ce=float(-data.logits[np.arange(data.n), data.winner].mean()),
        formal_ce_semantics='descriptive logsoftmax(mean opponent win), not winner-probability CE')


def macro_row(rows, split):
    return dict(previous.macro_row(rows, split), model=MODEL,
        formal_ce=float(np.mean([r['formal_ce'] for r in rows])))


def locked_inputs(root):
    root = Path(root)
    config, lock, bank, complete = (common.read_json(root / name) for name in
        ('protocol.json', 'final_test_lock.json', 'checkpoint.json', 'run_complete.json'))
    digest, state = common.file_hash(root / 'checkpoint.json'), common.json_hash(config)
    if (complete.get('complete') is not True or config.get('experiment') != MODEL or config.get('arm') != 'A'
            or common.read_json(root / 'prepare_identity.json') != config
            or lock.get('state_hash') != state or lock.get('checkpoint_sha256') != digest
            or complete.get('checkpoint_sha256') != digest or complete.get('full18_VAL') is not True
            or complete.get('full18_TEST') is not True or set(complete.get('LIB', [])) != set(LIBS)
            or common.resolve_record_path(lock['checkpoint'], root) != root / 'checkpoint.json'
            or lock.get('selected_without_test') is not True or lock.get('per_problem_assembly') is not False
            or lock.get('selection_rule') != SELECTION_RULE or bank.get('selection_rule') != SELECTION_RULE
            or bank.get('protocol_sha256') != state or bank.get('weights_updated') is not False
            or any(bank.get(key) != config.get(key) for key in
                   ('contexts', 'relations', 'public_checkpoint_sha256', 'inference_rule', 'source_sha256'))):
        raise ValueError('Require completed unchanged full18 R100 bank and unique pre-TEST lock')
    params = config.get('classifier_parameters', {})
    if (config.get('inference_rule') != INFERENCE_RULE or config.get('label_rule') != LABEL_RULE
            or config.get('probability_semantics') != PROBABILITY_SEMANTICS
            or config.get('optimizer_updates') != 0 or config.get('fine_tuning') is not False
            or config.get('historical_TRAIN_labels_in_context') is not True
            or config.get('current_query_labels_in_forward') is not False
            or config.get('kv_cache') is not False or config.get('allow_cpu_offload') is not False
            or config.get('flash_attention_only') is not True
            or any(config.get(key) is not False for key in ('aggregate_search', 'calibration', 'coupling', 'cost_sensitive_claim'))
            or params.get('n_estimators') != 1 or params.get('batch_size') != 1
            or params.get('norm_methods') != ['none'] or params.get('random_state') != 2):
        raise ValueError('Require fixed binary ICL and mean-opponent-win aggregation, no posthoc selection')
    for key in ('contexts', 'relations', 'candidate_pools', 'candidate_pool_ids', 'baselines'):
        if set(config.get(key, {})) != set(PROBLEMS):
            raise ValueError('Incomplete full18 protocol: ' + key)
    for problem in PROBLEMS:
        baseline, context, relation = (config[key][problem] for key in ('baselines', 'contexts', 'relations'))
        if (baseline.get('selection_split') != 'train' or baseline.get('train_rows') != 81000
                or context['rows'] != 81000 or relation['rows'] != 81000
                or relation['pairs'] != len(legal_pairs(config['candidate_pool_ids'][problem]))):
            raise ValueError('Require full81000 TRAIN base contexts and all legal pairs')
    bound = common.read_json(root / 'test_input_lock.json')
    if (bound.get('checkpoint_sha256') != digest or bound.get('labels_loaded') is not False
            or bound.get('created_after_unique_VAL_lock') is not True or set(bound['inputs']) != set(PROBLEMS)):
        raise ValueError('TEST input binding must follow unique full18 VAL lock')
    results = {}
    for split in ('val', 'test'):
        result = common.read_json(root / (split+'_results.json'))
        if result.get('complete') is not True or set(result['per_problem']) != set(PROBLEMS):
            raise ValueError('Incomplete full18 ' + split)
        previous._check_state(result, lock, split)
        results[split.upper()] = result
    libs = common.read_json(root / 'lib_results.json')
    for dataset, problem in LIBS.items():
        if libs[dataset].get('complete') is not True or libs[dataset].get('problem') != problem:
            raise ValueError('Incomplete native LIB ' + dataset)
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
                data = load_predictions(archive, path, problem+'__')
                previous.validate_population(data, problem, split, config)
                row = metric_row(data, config['baselines'][problem], problem, split)
                previous._check_saved(row, result['per_problem'][problem])
                predictions[split, problem] = data
                rows.append(row)
        overall = macro_row(rows, split)
        previous._check_saved(overall, result['macro'], macro=True)
        metrics.extend([overall, *rows])
    for dataset, problem in LIBS.items():
        path = root / 'lib_evaluation/predictions' / (dataset+'.npz')
        sources.append(path)
        with np.load(path, allow_pickle=False) as archive:
            data = load_predictions(archive, path)
        previous.validate_population(data, problem, dataset, config)
        row = metric_row(data, config['baselines'][problem], problem, dataset)
        previous._check_saved(row, libs[dataset])
        predictions[dataset, problem] = data
        metrics.append(row)
    for row in metrics:
        row.update(checkpoint_sha256=lock['checkpoint_sha256'], data_version=Path(config['data_root']).name)
    return predictions, metrics, sources


def relabel(rows):
    return [dict(row, **{key: MODEL for key in ('model', 'current_model') if key in row}) for row in rows]


def historical_comparisons(config, predictions, r84_root, r89_root, target_path):
    pairs, recovery, summaries, sources = previous.historical_comparisons(
        config, predictions, r84_root, r89_root, target_path)
    for baseline, root in (('R89B', r89_root), ('R84A', r84_root)):
        rows = [row for row in recovery if row['baseline'] == baseline]
        if not rows:
            continue
        path = root / 'evaluation/test/selected_best/rank000.npz'
        with np.load(path, allow_pickle=False) as archive:
            old = {p: previous.load_predictions(archive, path, p+'__', legacy=True) for p in PROBLEMS}
        for row in rows:
            data = predictions['TEST', row['problem']]
            reference = previous.align_reference(data, old[row['problem']], require_source=True)
            slot = data.pool.index(row['solver'])
            false = (data.pred == slot) & (data.winner != slot)
            new_false = false & (reference.pred != slot)
            selected = data.selected_costs
            row.update(false_pick_cost_sum=float(selected[false].sum()),
                new_false_pick_cost_sum=float(selected[new_false].sum()),
                new_false_pick_cost_contribution=float(selected[new_false].sum()/data.n),
                new_false_pick_cost_delta_contribution=float((selected-reference.selected_costs)[new_false].sum()/data.n))
        summary = next(s for s in summaries if s['baseline'] == baseline)
        for key in ('new_false_pick_cost_contribution', 'new_false_pick_cost_delta_contribution'):
            summary[key] = sum(row[key] for row in rows)/len(PROBLEMS)
    return relabel(pairs), relabel(recovery), relabel(summaries), sources


def context_resources(root, config):
    path, witness_path = root / 'timing.json', root / 'resource_witness.json'
    records, witness = common.read_json(path)['records'], common.read_json(witness_path)
    if (witness.get('complete') is not True or witness.get('labels_loaded') is not False
            or witness.get('accuracy_computed') is not False or witness.get('context_rows') != 81000
            or witness.get('query_rows') != 4500 or witness.get('extra_subset_call') is not False
            or witness.get('checkpoint_sha256') != config['public_checkpoint_sha256']):
        raise ValueError('Require first full TSP VAL unlabeled resource witness')
    for record in records:
        if not isinstance(record.get('seconds'), (int, float)) or not np.isfinite(record['seconds']) or record['seconds'] < 0:
            raise ValueError('Invalid recorded pair-context timing')
    resources, supports = [], []
    for problem in PROBLEMS:
        support = config['relations'][problem]['pair_support']
        supports.extend(dict(model=MODEL, problem=problem, **row) for row in support)
        expected = {r['pair_slot'] for r in support if r['shortcut'] == 'none'}
        constants = {r['pair_slot'] for r in support if r['shortcut'] != 'none'}
        row = dict(model=MODEL, problem=problem, full_train_rows=81000,
            retained_columns=config['contexts'][problem]['columns'], methods=len(config['candidate_pools'][problem]),
            legal_pairs=len(support), sequential_pair_fits=0, sequential_pair_predictions=0, constant_pair_shortcuts=0,
            load_seconds=0., pair_fit_seconds=0., pair_predict_seconds=0., outer_predict_seconds=0.,
            cached_query_seconds=0., native_LIB_input_to_statistics_seconds=0., recorded_sum_seconds=0.,
            fresh_VAL_TEST_graph_statistics_seconds=None, optimizer_updates=0, training_accuracy='N/A', epochs='N/A')
        splits = ['val', 'test', *[d for d, p in LIBS.items() if p == problem]]
        for split in splits:
            selected = [r for r in records if (r['problem'], r['split']) == (problem, split)]
            def one(stage):
                chosen = [r for r in selected if r['stage'] == stage]
                if len(chosen) != 1:
                    raise ValueError(f'Missing/duplicate {stage}: {problem}/{split}')
                return chosen[0]
            loading, call = one('load_context'), one('predict')
            inputs = one('native_input_to_statistics' if split in LIBS else 'cached_query_inputs')
            if (call.get('pairs') != len(support) or call.get('train_rows') != 81000
                    or call.get('sequential_pair_fits') != len(expected) or call.get('sequential_pair_predictions') != len(expected)
                    or call.get('constant_pair_shortcuts') != len(constants) or call.get('kv_cache') is not False
                    or call.get('context_reencoded') is not True or call.get('includes_all_sequential_pair_fits') is not True
                    or call.get('excludes_graph_to_statistics_time') is not True):
                raise ValueError('Require all K(K-1)/2 independent pair contexts in timing')
            for stage, target in (('fit_pair', 'pair_fit_seconds'), ('predict_pair', 'pair_predict_seconds'),
                                  ('constant_pair', None)):
                calls = [r for r in selected if r['stage'] == stage]
                slots = [r['pair_slot'] for r in calls]
                required = constants if stage == 'constant_pair' else expected
                if len(slots) != len(required) or set(slots) != required:
                    raise ValueError('Missing/duplicate independent pair call timing')
                for item in calls:
                    declared = support[item['pair_slot']]
                    if any(item[key] != declared[key] for key in ('strict_rows', 'tie_rows', 'label0_rows', 'label1_rows')):
                        raise ValueError('Pair runtime support differs from pre-VAL scientific lock')
                if target:
                    row[target] += sum(item['seconds'] for item in calls)
            row['load_seconds'] += loading['seconds']
            row['outer_predict_seconds'] += call['seconds']
            row['recorded_sum_seconds'] += loading['seconds']+inputs['seconds']+call['seconds']
            row['sequential_pair_fits'] += len(expected)
            row['sequential_pair_predictions'] += len(expected)
            row['constant_pair_shortcuts'] += len(constants)
            if split in LIBS:
                if inputs.get('labels_loaded') is not False or inputs.get('includes_native_input_loading') is not True:
                    raise ValueError('Native LIB input timing must exclude labels')
                row['native_LIB_input_to_statistics_seconds'] += inputs['seconds']
            else:
                if inputs.get('fresh_graph_to_statistics_measured') is not False:
                    raise ValueError('Cached TEST inputs cannot claim fresh graph statistics latency')
                row['cached_query_seconds'] += inputs['seconds']
        resources.append(row)
    return resources, supports, records, witness, [path, witness_path]


def plot_distribution(problem, datasets, pool, version, lock):
    from matplotlib.font_manager import FontProperties
    from matplotlib.ticker import PercentFormatter
    font_path = Path('/usr/share/fonts/google-noto-cjk/NotoSansCJK-Regular.ttc')
    font = FontProperties(fname=str(font_path)) if font_path.is_file() else None
    fig = previous._figure((20 if len(datasets) == 2 else 12, 9))
    axes = np.atleast_1d(fig.subplots(1, len(datasets)))
    fig.subplots_adjust(left=.16, right=.95, top=.78, bottom=.18, wspace=.5)
    fig.suptitle('R100 | '+problem+' | Frozen Pairwise Binary Context', fontsize=18, y=.97)
    fig.text(.035, .91, f'Data: {version}\nBank SHA256: {lock["checkpoint_sha256"]}', fontsize=10)
    for ax, (split, data) in zip(axes, datasets):
        ax.set_title(f'{problem+" TEST" if split == "TEST" else split} | n={data.n:,} | '
            f'\u6a21\u578bTop1={data.top1_pct:.2f}%', fontsize=14, fontproperties=font, pad=18)
        mapping = [data.pool.index(name) for name in pool]
        for field, offset, color, label in (('pred', -.17, '#197d85', 'R100 model selection'),
                                           ('winner', .17, '#c44f58', 'True native winner')):
            counts = np.bincount(getattr(data, field), minlength=len(pool))[mapping]
            percentages = counts/data.n*100.
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
    fig.text(.035, .10, 'Top1 = 100*mean(pred == saved native/FP64 ind), NOT the largest selection share.\n'
        'Argmax raw mean opponent win; ascending global-ID ties. TEST precedes original native LIB.', fontsize=10)
    fig.text(.035, .04, 'Pair win probabilities and formal exp(mean-win) views are NOT full-pool winner probabilities.', fontsize=9)
    return fig


def build_pdfs(root, output, config, lock, predictions, metrics, nss, pairs, recovery, summaries, resources):
    from matplotlib.backends.backend_pdf import PdfPages
    paths = dict(model_statistics_pdf=output / 'MODEL_STATISTICS.pdf',
        selection_distributions_pdf=output / 'selection_distributions.pdf', target_recovery_pdf=output / 'target_recovery.pdf')
    temporary = {key: path.with_name(f'.{path.stem}.{os.getpid()}.pdf') for key, path in paths.items()}
    try:
        with PdfPages(temporary['model_statistics_pdf']) as pdf, PdfPages(temporary['selection_distributions_pdf']) as dist, \
                PdfPages(temporary['target_recovery_pdf']) as targets:
            def save(fig, *destinations):
                for destination in destinations:
                    destination.savefig(fig)
                fig.clear()
            fig = previous._figure()
            fig.text(.04, .95, 'R100 | Pairwise Context And Evidence Boundaries', fontsize=20, weight='bold')
            fig.text(.04, .89, '\n\n'.join(textwrap.fill(note, 145) for note in NOTES), va='top', fontsize=8.5)
            save(fig, pdf)
            keys = ['problem', 'n', 'top1_pct', 'top2_pct', 'top3_pct', 'mean_cost', 'sbs_cost',
                'oracle_cost', 'vs_sbs_pct', 'vs_oracle_pct', 'actual_regret_pct']
            headers = ['Task', 'n', 'Top1 %', 'Top2 %', 'Top3 %', 'mean_cost', 'SBS cost',
                'Oracle', 'vs_SBS %', 'vs_Oracle %', 'Actual regret %']
            for split in ('VAL', 'TEST'):
                rows = [r for r in metrics if r['split'] == split]
                save(previous._table_page('R100 | '+split+' | ALL And Full18', headers,
                    [[r[key] for key in keys] for r in rows],
                    ['Native/FP64 ind; fixed expected-rank/Borda, no calibration/aggregation search.',
                     'SBS: R84 full81000 TRAIN. ALL equal-problem macro; mixed raw cost descriptive.',
                     'vs_* ratios of means differ from mean per-instance actual regret.']), pdf)
            rows = [r for r in metrics if r['split'] in LIBS]
            save(previous._table_page('R100 | Native LIB Metrics', ['Dataset', *headers[1:]],
                [[r['split'], *[r[key] for key in keys[1:]]] for r in rows],
                ['Same unique context/weight bank as full18 VAL/TEST; original native LIB, no per-dataset tuning.']), pdf)
            save(previous._table_page('R100 | NSS Four-Dataset Historical Comparison',
                ['Dataset', 'Status', 'n', 'NSS Top1 %', 'R100 Top1 %', 'NSS cost', 'R100 cost'],
                [[r['dataset'], r['status'], r['n'], r['nss_top1_native_pct'], r['current_top1_native_pct'],
                  r['nss_mean_cost'], r['current_mean_cost']] for r in nss],
                ['NSS: saved R79 historical70114 TRAIN weights, not a new81000 retrain. Current R100 bank fixed across all four.',
                 'Native rows/costs/winner/pool aligned; CSV records versions, pools, checkpoints and missing evidence.']), pdf)
            for baseline in ('R89B', 'R84A'):
                pair = next(r for r in pairs if r['baseline'] == baseline and r['problem'] == 'ALL')
                summary = next(r for r in summaries if r['baseline'] == baseline)
                save(previous._table_page('R100 | Full18 TEST vs '+baseline,
                    ['Status', 'Rescued', 'Harmed', 'Net correct', 'Top1 delta pp', 'Cost delta', 'Regret delta pp'],
                    [[pair['status'], pair.get('corrected'), pair.get('harmed'), pair.get('net_correct'),
                      pair.get('top1_delta_pp'), pair.get('mean_cost_delta'), pair.get('actual_regret_delta_pp')]],
                    ['R89B main, R84A secondary, historical and not compute matched. Positive cost/regret delta is worse.',
                     'Fixed36 status: '+summary['status']+'; targets frozen from R84A VAL, not R100 TEST.']), pdf, targets)
                rows = [r for r in recovery if r['baseline'] == baseline]
                for start in range(0, len(rows), 18):
                    save(previous._table_page('R100 | Fixed36 TEST Recovery vs '+baseline,
                        ['Task / Method', 'Winners', 'Rescued', 'Lost', 'Net', 'New false', 'False cost delta', 'Regret delta pp'],
                        [[r['problem']+'/'+r['solver'], r['winner_count'], r['rescued'], r['lost'], r['net_recovery'],
                          r['new_false_pick_count'], r['new_false_pick_cost_delta_contribution'], r['target_regret_delta_pp']]
                         for r in rows[start:start+18]],
                        ['New false-pick cost delta is relative to reference selections, normalized by whole task n.',
                         'CSV also includes false raw cost sums/contributions, precision, recall and new-false regret.']), pdf, targets)
            save(previous._table_page('R100 | Independent Pair Context Runtime',
                ['Task', 'Methods', 'Pairs', 'Fits', 'Predicts', 'Load s', 'Fit s', 'Predict s', 'Outer s', 'Recorded sum s'],
                [[r[key] for key in ('problem', 'methods', 'legal_pairs', 'sequential_pair_fits', 'sequential_pair_predictions',
                    'load_seconds', 'pair_fit_seconds', 'pair_predict_seconds', 'outer_predict_seconds', 'recorded_sum_seconds')]
                 for r in resources],
                ['K(K-1)/2 contexts per named dataset; only exact one-class/no-strict-row shortcuts. One classifier resident.',
                 'Pair breakdowns are nested in outer calls, not added twice. Cached VAL/TEST exclude fresh graph statistics.',
                 'LIB includes native input/geometry/statistics; recorded sum is not full end-to-end runtime.']), pdf)
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
    shutil.copyfile(paths['model_statistics_pdf'], root / 'RESULTS.pdf')
    return dict(paths, results_pdf=root / 'RESULTS.pdf')


def report(args):
    if str(getattr(args, 'device', 'cpu')) != 'cpu':
        raise ValueError('R100 final reporting is CPU-only; use --device cpu')
    root = Path(args.root).resolve()
    config, lock, results, libs = locked_inputs(root)
    output, status = root / 'full_model_statistics', root / 'postprocess_status.json'
    with (root / 'postprocess.lock').open('a') as claim:
        fcntl.flock(claim, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            names = ('prepare_identity.json', 'protocol.json', 'checkpoint.json', 'final_test_lock.json',
                'test_input_lock.json', 'val_results.json', 'test_results.json', 'lib_results.json', 'run_complete.json',
                'timing.json', 'resource_witness.json', 'wandb_receipt.json')
            hashes = {str(root / name): common.file_hash(root / name) for name in names}
            common.write_json(status, dict(experiment=MODEL, complete=False, phase='loading_saved_predictions'))
            predictions, metrics, prediction_sources = saved_evaluations(root, config, lock, results, libs)
            nss_root = Path(getattr(args, 'nss_root', None) or previous.DEFAULT_R84 / 'nss_reference').resolve()
            r84_root = Path(getattr(args, 'r84_root', None) or previous.DEFAULT_R84 / 'A_baseline').resolve()
            r89_root = Path(getattr(args, 'r89_root', None) or previous.DEFAULT_R89).resolve()
            target_path = Path(getattr(args, 'target_list', None) or previous.DEFAULT_R84 / 'comparison_artifacts/target_list.json').resolve()
            nss, nss_sources = previous.nss_comparisons(config, lock, predictions, nss_root)
            nss = relabel(nss)
            pairs, recovery, summaries, history_sources = historical_comparisons(config, predictions, r84_root, r89_root, target_path)
            resources, support, timing, witness, resource_sources = context_resources(root, config)
            wandb = common.read_json(root / 'wandb_receipt.json')
            if wandb.get('enabled') is not True or wandb.get('mode') != 'offline':
                raise ValueError('Require declared offline W&B receipt')
            for path in dict.fromkeys([*prediction_sources, *nss_sources, *history_sources, *resource_sources]):
                hashes.setdefault(str(path), common.file_hash(path))
            output.mkdir(parents=True, exist_ok=True)
            distributions = relabel(previous.distribution_rows(predictions, config, lock))
            methods = []
            for (split, problem), data in predictions.items():
                for slot, solver in enumerate(data.pool):
                    methods.append(dict(model=MODEL, split=split, problem=problem, solver=solver,
                        method_top1_pct=float((data.winner == slot).mean()*100.), method_mean_cost=float(data.costs[:, slot].mean()),
                        model_top1_pct=data.top1_pct, model_mean_cost=float(data.selected_costs.mean()), winner_rule=WINNER_RULE))
            tables = dict(metrics_by_problem=metrics, selection_distribution=distributions, single_method_comparison=methods,
                nss_comparison=nss, paired_comparison=pairs, target_recovery=recovery or [dict(status='missing')],
                target_recovery_summary=summaries, pair_support=support, context_resource_summary=resources, timing_records=timing)
            artifacts = {}
            for name, rows in tables.items():
                path = output / (name+'.csv')
                common.write_csv(path, rows)
                artifacts[name+'_csv'] = str(path)
            common.write_json(status, dict(experiment=MODEL, complete=False, phase='rendering_cpu_reports'))
            artifacts.update({key: str(path) for key, path in build_pdfs(root, output, config, lock,
                predictions, metrics, nss, pairs, recovery, summaries, resources).items()})
            keys = ['problem', 'split', 'n', 'top1_pct', 'mean_cost', 'sbs_cost', 'oracle_cost',
                'vs_sbs_pct', 'vs_oracle_pct', 'actual_regret_pct']
            lines = ['# R100 Frozen Pairwise Binary Context', '', 'Primary delivery: [RESULTS.pdf](RESULTS.pdf).', '',
                *['- '+note for note in NOTES], '',
                *previous._md_table(keys, [[r[key] for key in keys] for r in metrics]), '',
                'NSS/fixed36/support/timing/provenance: full_model_statistics/.', '']
            (root / 'RESULTS.md').write_text('\n'.join(lines), encoding='utf-8')
            artifacts['results_md'] = str(root / 'RESULTS.md')
            for path, digest in hashes.items():
                if common.file_hash(path) != digest:
                    raise ValueError('Report source changed during rendering: '+path)
            if locked_inputs(root) != (config, lock, results, libs):
                raise ValueError('Unique frozen bank changed during reporting')
            provenance = dict(experiment=MODEL, checkpoint_sha256=lock['checkpoint_sha256'], state_hash=lock['state_hash'],
                input_sha256=hashes, classifier_parameters=config['classifier_parameters'], contexts=config['contexts'],
                relations=config['relations'], public_checkpoint_sha256=config['public_checkpoint_sha256'],
                source_sha256=config['source_sha256'], runtime_source_sha256=config['runtime_source_sha256'],
                inference_rule=INFERENCE_RULE, probability_semantics=PROBABILITY_SEMANTICS, notes=list(NOTES),
                resource_witness=witness, wandb_receipt=wandb, inference_performed=False, reference_retraining=False,
                train_self_accuracy=None, learning_curves=None, winner_probabilities_claimed=False,
                fresh_TEST_graph_statistics_measured=False)
            common.write_json(output / 'provenance.json', provenance)
            artifacts['provenance_json'] = str(output / 'provenance.json')
            result = dict(experiment=MODEL, complete=True, phase='complete', artifacts=artifacts,
                checkpoint_sha256=lock['checkpoint_sha256'], state_hash=lock['state_hash'], updated_utc=common.utc())
            common.write_json(status, result)
            return result
        except Exception as error:
            common.write_json(status, dict(experiment=MODEL, complete=False, phase='failed', error=str(error)))
            raise

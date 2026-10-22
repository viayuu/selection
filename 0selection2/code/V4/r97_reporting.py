"""Offline, CPU-only final reporting for the single frozen R97/A bank.

API: ``report(args) -> dict`` needs only ``args.root`` and ``args.device='cpu'``.
The CLI is ``python -B -m code.V4.r97_reporting report --root PATH --device cpu``.
Optional reference paths: nss_root, r84_root, r89_root, target_list (see parser).
No runner, torch, TabICL, weights, context labels or query features are imported
or loaded. Importing does not change environment variables or shared globals.

Runner contract (root is the actual run directory, not its parent):
* protocol.json: experiment=R97, arm=A, rows_per_problem, baselines, pools/IDs,
  contexts[problem]={rows,columns,metadata_sha256}, classifier_parameters,
  public_checkpoint_sha256, package_version, source_revision, inference_rule.
* checkpoint.json: protocol_sha256, public_checkpoint_sha256, contexts,
  inference_rule, weights_updated=false, selection_rule=SELECTION_RULE.
* final_test_lock.json: checkpoint path/hash, state_hash=canonical protocol
  JSON hash, selected_without_test=true, per_problem_assembly=false,
  selection_rule=SELECTION_RULE. No epoch/selected_update is expected.
* run_complete.json: complete=true and checkpoint_sha256 of checkpoint.json.
* val_results.json / test_results.json: complete=true, state_hash, and r80
  summarize's per_problem + macro. LIB results map TSPLIB/CVRPLIB directly to
  metric dicts with complete=true, state_hash, problem, source_directory.
* evaluation/{val,test}/selected_best/rank000.npz: PROBLEM__{logits,pred,
  winner,costs,pool,pool_ids,nodes,indices,source_indices,release_source_indices}.
  LIB: lib_evaluation/predictions/{TSPLIB,CVRPLIB}.npz, unprefixed. Optional
  scores and probabilities are checked, not used to rerun inference.
* timing.json: records[{problem,split,stage='fit_context'|'predict',seconds,
  train_rows,...}]. Fit records also contain total_seconds (includes loading)
  and context_read_and_verify_seconds; predict records declare kv_cache=false,
  context_reencoded=true, excludes_graph_to_statistics_time=true. Each LIB has
  an additional native_input_to_statistics record (seconds includes native
  loading, geometry, and 357 statistics). VAL/TEST use canonical statistics
  caches: fresh raw-graph-to-statistics time is NOT measured.
  resource_witness.json: complete, labels_loaded=false,
  accuracy_computed=false, context_rows, checkpoint_sha256 (PUBLIC weights),
  fit/predict timing dicts. Witness fit is reused in TSP VAL: never count twice.

Artifacts: ROOT/RESULTS.{md,pdf}; ROOT/full_model_statistics contains
MODEL_STATISTICS.pdf, selection_distributions.pdf, target_recovery.pdf,
metrics_by_problem.csv, selection_distribution.csv, single_method_comparison.csv,
nss_comparison.csv, paired_comparison.csv, target_recovery{,_summary}.csv,
context_resource_summary.csv, timing_records.csv, and provenance.json.
Return value and ROOT/postprocess_status.json contain complete, phase, output,
and artifacts (absolute paths). Missing historical references are explicit;
present but incompatible references fail instead of silently changing cohorts.
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

from code.unified_selector.registry import PROBLEMS
from . import r80_reporting as common


V4 = Path(__file__).resolve().parent
DEFAULT_ROOT = V4 / 'runs/R97_tabular_in_context'
DEFAULT_R84 = V4 / 'runs/R84_new90k_attention'
DEFAULT_R89 = V4 / 'runs/R89_optimizer_geometry/B_muon_hybrid'
LIBS = {'TSPLIB': 'TSP', 'CVRPLIB': 'CVRP'}
SELECTION_RULE = 'frozen_single_predeclared_bank_no_VAL_search'
WINNER_RULE = 'saved native/FP64 ind; never FP32 argmin'
METRICS = ('top1_pct', 'top2_pct', 'top3_pct', 'sbs_top1_pct', 'mean_cost',
           'sbs_cost', 'oracle_cost', 'vs_sbs_pct', 'vs_oracle_pct', 'actual_regret_pct')
SAVED_KEYS = {'top1_pct': 'top1', 'top2_pct': 'top2', 'top3_pct': 'top3',
              **{key: key for key in METRICS[4:]}}
NOTES = (
    'One prelocked full18 R97/A context/weight bank; no VAL grid, no task-wise model assembly.',
    'TabICL v2.0.2 classifier, official v2 public synthetic checkpoint; external pretraining is NOT matched compute.',
    'Resource preset: n_estimators=1; norm_methods=[none]; full historical 81,000 TRAIN rows/task at natural frequency/order.',
    'The full-context resource budget is predeclared, not tuned via VAL. No selector SGD or fine-tuned weights.',
    'Input-only R92 statistics: 357 raw columns, drop only task-TRAIN constants (100-217 retained in this release).',
    'Historical TRAIN labels are authorized context. Current query labels/winners/costs never enter forward inputs.',
    'TRAIN self-context accuracy, epochs, SGD learning curves and training loss: N/A; no generalization claim from self labels.',
    'SBS is fixed on R84 full 81,000-row TRAIN. Oracle is each instance\'s minimum in its unchanged legal candidate pool.',
    'ALL is the equal-problem mean across 18 tasks; mixed raw mean_cost is descriptive, not a pooled objective.',
    'vs_SBS / vs_Oracle = 100*(ratio of mean costs - 1); actual regret = mean_i[100*(selected_i-oracle_i)/oracle_i].',
    'NSS reuses R79 historical 70,114-row TRAIN weights and saved new90k/LIB predictions; NOT a new 81,000-row retrain.',
    'R89B is the main historical reference; R84A is secondary. Neither is a newly compute-matched control.',
    'Retained old VAL/TEST members informed development; this is not a fully independent blind TEST.',
    'Fewer zero-selected groups are not success: inspect true recovery, lost winners, new false picks, cost and net regret.',
    'kv_cache=False: every predict re-encodes all 81,000 TRAIN context rows. No cached low-latency claim.',
    'Fit total_seconds includes context read/verification; fit seconds excludes that loading. Do not add read time twice.',
    'VAL/TEST use canonical R92 statistics caches. Timings EXCLUDE fresh raw-graph-to-statistics/cache construction.',
    'LIB native_input_to_statistics includes native loading, geometry and 357 statistics. Recorded sums are not end-to-end latency.',
)


@dataclass(frozen=True)
class Predictions:
    path: Path
    pool: list
    pool_ids: np.ndarray | None
    pred: np.ndarray
    winner: np.ndarray
    costs: np.ndarray
    indices: np.ndarray
    source_indices: np.ndarray | None = None
    release_source_indices: np.ndarray | None = None
    nodes: np.ndarray | None = None
    logits: np.ndarray | None = None

    @property
    def n(self):
        return len(self.winner)

    @property
    def selected_costs(self):
        return self.costs[np.arange(self.n), self.pred]

    @property
    def top1_pct(self):
        return float((self.pred == self.winner).mean() * 100.)


def load_predictions(archive, path, prefix='', *, legacy=False):
    """Keep native local labels; stable GLOBAL IDs resolve current score ties."""
    def get(key, optional=False):
        name = prefix + key
        return None if optional and name not in archive else np.asarray(archive[name])

    pool_array, costs, winner, pred = (get(key) for key in ('pool', 'costs', 'winner', 'pred'))
    if pool_array.ndim != 1 or pool_array.dtype.kind not in 'US':
        raise ValueError(f'Expected distinct string candidate names: {path} {prefix}')
    pool = pool_array.tolist()
    n, k = costs.shape if costs.ndim == 2 else (0, 0)
    if (not n or not k or len(pool) != k or len(set(pool)) != k
            or costs.dtype != np.float64 or not np.isfinite(costs).all() or np.any(costs <= 0)):
        raise ValueError(f'Expected nonempty positive FP64 costs/full pool: {path} {prefix}')
    for key, values in (('winner', winner), ('pred', pred)):
        if (values.shape != (n,) or values.dtype != np.int64
                or np.any(values < 0) or np.any(values >= k)):
            raise ValueError(f'Expected local int64 {key}: {path} {prefix}')
    # Checking the saved native winner is not replacing it with an FP32 argmin.
    if not np.array_equal(costs[np.arange(n), winner], costs.min(1)):
        raise ValueError(f'Native winner is not an FP64 minimum: {path} {prefix}')
    vectors = {}
    for key in ('indices', 'source_indices', 'release_source_indices', 'nodes', 'pool_ids'):
        values = get(key, optional=legacy and key != 'indices')
        if values is not None and (values.dtype.kind not in 'iu' or values.shape != ((k,) if key == 'pool_ids' else (n,))):
            raise ValueError(f'Invalid {key}: {path} {prefix}')
        vectors[key] = values
    if len(np.unique(vectors['indices'])) != n:
        raise ValueError('Duplicate saved instance indices')
    ids = vectors['pool_ids']
    if ids is not None and (np.any(ids < 0) or len(np.unique(ids)) != k):
        raise ValueError('Invalid stable global candidate IDs')
    logits = get('logits', optional=legacy)
    if not legacy:
        if (logits.shape != costs.shape or np.isnan(logits).any() or np.isposinf(logits).any()
                or not np.isfinite(logits.max(1)).all()
                or not np.allclose(np.exp(logits).sum(1), 1., rtol=1e-5, atol=1e-7)):
            raise ValueError('Expected local-K log probabilities')
        order = np.argsort(ids, kind='stable')
        np.testing.assert_array_equal(pred, order[np.argmax(logits[:, order], axis=1)],
                                      err_msg='Saved prediction differs from stable global-ID argmax')
        if prefix + 'scores' in archive:
            np.testing.assert_array_equal(get('scores'), logits)
        if prefix + 'probabilities' in archive:
            probabilities = get('probabilities')
            if (probabilities.shape != costs.shape or not np.isfinite(probabilities).all()
                    or np.any(probabilities < 0)):
                raise ValueError('Invalid optional probabilities')
            np.testing.assert_allclose(probabilities, np.exp(logits), rtol=1e-5, atol=1e-7)
    return Predictions(Path(path), pool, ids, pred, winner, costs, logits=logits, **{
        key: value for key, value in vectors.items() if key != 'pool_ids'})


def _check_state(record, lock, label):
    if record.get('state_hash') != lock['state_hash']:
        raise ValueError(f'{label} differs from the frozen protocol/context/weight bank')
    if 'checkpoint_sha256' in record and record['checkpoint_sha256'] != lock['checkpoint_sha256']:
        raise ValueError(f'{label} differs from the checkpoint manifest SHA256')


def locked_inputs(root):
    """Read completion FIRST; never create artifacts or inspect incomplete TEST."""
    root = Path(root).resolve()
    complete = common.read_json(root / 'run_complete.json')
    if complete.get('complete') is not True:
        raise ValueError('Runner is not complete; no real report may be built')
    config = common.read_json(root / 'protocol.json')
    lock = common.read_json(root / 'final_test_lock.json')
    checkpoint = root / 'checkpoint.json'
    bank = common.read_json(checkpoint)
    state, digest = common.json_hash(config), common.file_hash(checkpoint)
    if (config.get('experiment') != 'R97' or config.get('arm') != 'A'
            or lock.get('state_hash') != state or lock.get('checkpoint_sha256') != digest
            or complete.get('checkpoint_sha256') != digest
            or common.resolve_record_path(lock['checkpoint'], root) != checkpoint
            or lock.get('selected_without_test') is not True or lock.get('per_problem_assembly') is not False
            or lock.get('selection_rule') != SELECTION_RULE
            or bank.get('protocol_sha256') != state or bank.get('selection_rule') != SELECTION_RULE
            or bank.get('weights_updated') is not False
            or any(bank.get(key) != config[key] for key in ('contexts', 'public_checkpoint_sha256', 'inference_rule'))):
        raise ValueError('Require the unchanged unique frozen R97 bank and pre-TEST lock')
    if (config.get('optimizer_updates') != 0 or config.get('fine_tuning') is not False
            or config.get('historical_TRAIN_labels_in_context') is not True
            or config.get('current_query_labels_in_forward') is not False
            or config['classifier_parameters'].get('n_estimators') != 1
            or config['classifier_parameters'].get('norm_methods') != ['none']):
        raise ValueError('Require the single predeclared frozen TRAIN-context classifier')
    for key in ('contexts', 'baselines', 'candidate_pools', 'candidate_pool_ids'):
        if set(config[key]) != set(PROBLEMS):
            raise ValueError(f'Incomplete full18 protocol: {key}')
    for problem in PROBLEMS:
        baseline, context = config['baselines'][problem], config['contexts'][problem]
        if (baseline.get('selection_split') != 'train'
                or baseline.get('train_rows') != config['rows_per_problem']['train']
                or context['rows'] != config['rows_per_problem']['train']):
            raise ValueError('Require full historical TRAIN context and fixed R84 full-TRAIN SBS')
    results = {}
    for split in ('val', 'test'):
        result = common.read_json(root / (split + '_results.json'))
        if result.get('complete') is not True or set(result['per_problem']) != set(PROBLEMS):
            raise ValueError('Require completed full18 ' + split)
        _check_state(result, lock, split)
        results[split.upper()] = result
    libs = common.read_json(root / 'lib_results.json')
    for dataset, problem in LIBS.items():
        if libs[dataset].get('complete') is not True or libs[dataset].get('problem') != problem:
            raise ValueError('Require completed native LIB: ' + dataset)
        _check_state(libs[dataset], lock, dataset)
    return config, lock, results, libs


def validate_population(data, problem, split, config):
    ids = config['candidate_pool_ids'][problem]
    if (data.pool != config['candidate_pools'][problem] or ids != sorted(set(ids))
            or data.pool_ids is None or data.pool_ids.tolist() != ids):
        raise ValueError('Saved pool differs from the natural frozen candidate pool: ' + problem)
    if split in ('VAL', 'TEST') and data.n != config['rows_per_problem'][split.lower()]:
        raise ValueError('Saved prediction population differs: ' + split + '/' + problem)
    np.testing.assert_array_equal(data.indices, np.arange(data.n))


def metric_row(data, baseline, problem, split):
    if baseline['selection_split'] != 'train':
        raise ValueError('Main SBS must be fixed on full TRAIN, not this evaluation split')
    slot = data.pool.index(baseline['sbs_solver'])
    if baseline.get('sbs_slot', slot) != slot:
        raise ValueError('Frozen SBS local slot differs from candidate order')
    oracle, selected = data.costs.min(1), data.selected_costs
    mean, sbs, best = float(selected.mean()), float(data.costs[:, slot].mean()), float(oracle.mean())
    order = np.argsort(data.pool_ids, kind='stable')
    rank = order[np.argsort(-data.logits[:, order], axis=1, kind='stable')]
    return dict(problem=problem, split=split, n=data.n, model='R97/A',
        top1_pct=data.top1_pct,
        top2_pct=float((rank[:, :min(2, len(data.pool))] == data.winner[:, None]).any(1).mean() * 100.),
        top3_pct=float((rank[:, :min(3, len(data.pool))] == data.winner[:, None]).any(1).mean() * 100.),
        sbs_top1_pct=float((data.winner == slot).mean() * 100.), mean_cost=mean,
        sbs_cost=sbs, oracle_cost=best, vs_sbs_pct=100. * (mean / sbs - 1.),
        vs_oracle_pct=100. * (mean / best - 1.), actual_regret_pct=float(((selected - oracle) / oracle * 100.).mean()),
        native_correct=int((data.pred == data.winner).sum()), sbs_name=data.pool[slot],
        sbs_selection_split='R84 FULL TRAIN', winner_rule=WINNER_RULE,
        top1_fp32_argmin_pct=float((data.pred == data.costs.astype(np.float32).argmin(1)).mean() * 100.),
        aggregation='single problem', candidate_pool=json.dumps(data.pool), prediction_path=str(data.path))


def macro_row(rows, split):
    result = dict(rows[0], problem='ALL', split=split, n=sum(row['n'] for row in rows),
        native_correct=sum(row['native_correct'] for row in rows), sbs_name='per-problem fixed TRAIN SBS',
        candidate_pool='per-problem natural legal pool', prediction_path='see per-problem rows',
        aggregation='equal weight per problem; mixed raw costs descriptive only')
    for key in (*METRICS, 'top1_fp32_argmin_pct'):
        result[key] = float(np.mean([row[key] for row in rows]))
    return result


def _check_saved(row, saved, *, macro=False):
    for key, name in SAVED_KEYS.items():
        value = row[key] / 100. if key.startswith('top') else row[key]
        np.testing.assert_allclose(value, saved[('macro_' if macro else '') + name],
                                   rtol=1e-10, atol=1e-9, err_msg=key)
    if not macro and (row['n'] != saved['n'] or row['sbs_name'] != saved['sbs_name']
                      or saved.get('sbs_selection_split') != 'train'):
        raise ValueError('Saved metrics differ from full TRAIN SBS/population')


def saved_evaluations(root, config, lock, results, libs):
    predictions, metrics, methods = {}, [], []
    sources = []
    for split, result in results.items():
        path = root / 'evaluation' / split.lower() / 'selected_best/rank000.npz'
        sources.append(path)
        rows = []
        with np.load(path, allow_pickle=False) as archive:
            for problem in PROBLEMS:
                data = load_predictions(archive, path, problem + '__')
                validate_population(data, problem, split, config)
                row = metric_row(data, config['baselines'][problem], problem, split)
                _check_saved(row, result['per_problem'][problem])
                predictions[split, problem] = data
                rows.append(row)
        overall = macro_row(rows, split)
        _check_saved(overall, result['macro'], macro=True)
        metrics.extend([overall, *rows])
    for dataset, problem in LIBS.items():
        path = root / 'lib_evaluation/predictions' / (dataset + '.npz')
        sources.append(path)
        with np.load(path, allow_pickle=False) as archive:
            data = load_predictions(archive, path)
        validate_population(data, problem, dataset, config)
        row = metric_row(data, config['baselines'][problem], problem, dataset)
        _check_saved(row, libs[dataset])
        predictions[dataset, problem] = data
        metrics.append(row)
    for row in metrics:
        row.update(checkpoint_sha256=lock['checkpoint_sha256'], data_version=Path(config['data_root']).name,
                   sbs_train_rows=config['rows_per_problem']['train'])
        if row['problem'] == 'ALL':
            continue
        data = predictions[row['split'], row['problem']]
        for slot, solver in enumerate(data.pool):
            top1, cost = float((data.winner == slot).mean() * 100.), float(data.costs[:, slot].mean())
            methods.append(dict(problem=row['problem'], split=row['split'], solver=solver,
                method_top1_pct=top1, method_mean_cost=cost, model_top1_pct=row['top1_pct'],
                model_mean_cost=row['mean_cost'], model_exceeds_top1=row['top1_pct'] > top1,
                model_cheaper=row['mean_cost'] < cost, winner_rule=WINNER_RULE))
    return predictions, metrics, methods, sources


def align_reference(current, reference, *, require_source=False):
    """Align ALL rows by split-local ID and ALL columns/labels by solver name."""
    if current.n != reference.n or set(current.pool) != set(reference.pool):
        raise ValueError('Reference population/candidate pool mismatch')
    if set(current.indices.tolist()) != set(reference.indices.tolist()):
        raise ValueError('Reference instance IDs differ')
    lookup = {int(value): i for i, value in enumerate(reference.indices)}
    rows = np.asarray([lookup[int(value)] for value in current.indices])
    columns = [reference.pool.index(name) for name in current.pool]
    inverse = np.argsort(columns)
    np.testing.assert_array_equal(current.costs, reference.costs[rows][:, columns],
                                  err_msg='Reference original FP64 costs differ')
    np.testing.assert_array_equal(current.winner, inverse[reference.winner[rows]],
                                  err_msg='Reference native winner names differ')
    if current.pool_ids is not None and reference.pool_ids is not None:
        np.testing.assert_array_equal(current.pool_ids, reference.pool_ids[columns])
    if current.nodes is not None and reference.nodes is not None:
        np.testing.assert_array_equal(current.nodes, reference.nodes[rows])
    for name in ('source_indices', 'release_source_indices') if require_source else ():
        left, right = getattr(current, name), getattr(reference, name)
        if left is not None and right is not None:
            np.testing.assert_array_equal(left, right[rows])
    return replace(current, path=reference.path, pred=inverse[reference.pred[rows]], logits=None)


def nss_comparisons(config, lock, predictions, root):
    plan_path = root / 'protocol.json'
    plan = common.read_json(plan_path) if plan_path.is_file() else None
    if plan is not None and (Path(plan['data_root']).resolve() != Path(config['data_root']).resolve()
                             or plan.get('retrained_on_new90k') is not False):
        raise ValueError('NSS must be the saved historical R79 reference on this exact new90k release')
    rows, sources = [], [plan_path] if plan is not None else []
    for dataset, problem, split in (('TSP TEST', 'TSP', 'TEST'), ('CVRP TEST', 'CVRP', 'TEST'),
                                    ('TSPLIB', 'TSP', 'TSPLIB'), ('CVRPLIB', 'CVRP', 'CVRPLIB')):
        data = predictions[split, problem]
        name = 'test' if split == 'TEST' else split
        path = root / problem / (name + '_predictions.npz')
        metadata_path = path.with_name(name + '_metrics.json')
        row = dict(dataset=dataset, problem=problem, n=data.n, current_model='R97/A',
            current_top1_native_pct=data.top1_pct, current_mean_cost=float(data.selected_costs.mean()),
            nss_model='R79 historical70114 TRAIN weights; not new81000 retrained',
            nss_top1_native_pct=None, nss_mean_cost=None, nss_checkpoint_sha256=None,
            current_checkpoint_sha256=lock['checkpoint_sha256'], data_version=Path(config['data_root']).name,
            candidate_pool=json.dumps(data.pool), winner_rule=WINNER_RULE,
            nss_prediction_path=str(path), checked_rows=0, status='missing')
        if plan is None or not path.is_file() or not metadata_path.is_file():
            row['reason'] = 'Missing saved NSS protocol/prediction/metrics; no retraining or inference attempted'
        else:
            metadata = common.read_json(metadata_path)
            expected = plan['checkpoints'][problem]['sha256']
            if metadata.get('checkpoint', {}).get('sha256') != expected:
                raise ValueError('NSS weights identity differs from historical lock')
            with np.load(path, allow_pickle=False) as archive:
                reference = load_predictions(archive, path, legacy=True)
            aligned = align_reference(data, reference)
            np.testing.assert_allclose(aligned.top1_pct, metadata['top1_native_ind'] * 100., atol=1e-9, rtol=0)
            np.testing.assert_allclose(aligned.selected_costs.mean(), metadata['mean_cost'], atol=1e-9, rtol=0)
            row.update(nss_top1_native_pct=aligned.top1_pct, nss_mean_cost=float(aligned.selected_costs.mean()),
                nss_checkpoint_sha256=expected, checked_rows=data.n, status='complete', reason='',
                prediction_sha256=common.file_hash(path))
            sources.extend([path, metadata_path])
        rows.append(row)
    return rows, sources


def paired_comparison(current, old, baseline):
    rows = []
    for problem, data in current.items():
        reference = align_reference(data, old[problem], require_source=True)
        before, after = reference.pred == data.winner, data.pred == data.winner
        delta, oracle = data.selected_costs - reference.selected_costs, data.costs.min(1)
        rows.append(dict(model='R97/A', baseline=baseline, role='main' if baseline == 'R89B' else 'secondary',
            split='TEST', problem=problem, n=data.n, corrected=int((~before & after).sum()),
            harmed=int((before & ~after).sum()), net_correct=int(after.sum() - before.sum()),
            top1_delta_pp=float((after.mean() - before.mean()) * 100.),
            mean_cost_delta=float(delta.mean()), actual_regret_delta_pp=float((delta / oracle).mean() * 100.),
            status='complete'))
    total = dict(rows[0], problem='ALL', n=sum(row['n'] for row in rows))
    for name in ('corrected', 'harmed', 'net_correct'):
        total[name] = sum(row[name] for row in rows)
    for name in ('top1_delta_pp', 'mean_cost_delta', 'actual_regret_delta_pp'):
        total[name] = float(np.mean([row[name] for row in rows]))
    return [total, *rows]


def target_recovery(current, old, targets, baseline):
    aligned = {p: align_reference(data, old[p], require_source=True) for p, data in current.items()}
    rows = []
    for target in targets:
        problem, solver = target['problem'], target['method']
        data, reference = current[problem], aligned[problem]
        slot = data.pool.index(solver)
        truth, was, chosen = data.winner == slot, reference.pred == slot, data.pred == slot
        oracle = data.costs.min(1)
        regret, before = (data.selected_costs - oracle) / oracle, (reference.selected_costs - oracle) / oracle
        false = chosen & ~truth & ~was
        support, correct, picks = int(truth.sum()), int((truth & chosen).sum()), int(chosen.sum())
        rescued, lost = int((truth & ~was & chosen).sum()), int((truth & was & ~chosen).sum())
        rows.append(dict(model='R97/A', baseline=baseline, split='TEST', problem=problem, solver=solver,
            category=target.get('category', ''), winner_count=support, correct_count=correct, selected_count=picks,
            rescued=rescued, lost=lost, net_recovery=rescued-lost,
            false_pick_count=int((chosen & ~truth).sum()), new_false_pick_count=int(false.sum()),
            recall_pct=100. * correct / support if support else None,
            precision_pct=100. * correct / picks if picks else None,
            new_false_pick_regret_contribution_pp=float(regret[false].sum() / data.n * 100.),
            target_regret_delta_pp=float((regret-before)[truth].sum() / data.n * 100.), status='complete'))
    return rows


def historical_comparisons(config, predictions, r84_root, r89_root, target_path):
    target_record = common.read_json(target_path) if target_path.is_file() else None
    targets = target_record['targets'] if target_record is not None else None
    if targets is not None:
        pairs = [(row['problem'], row['method']) for row in targets]
        if (len(pairs) != 36 or len(set(pairs)) != 36
                or Path(target_record['identity']['data_root']).resolve() != Path(config['data_root']).resolve()):
            raise ValueError('Use the fixed36 R84 target list on the exact same release')
    current = {p: predictions['TEST', p] for p in PROBLEMS}
    comparisons, recovery, summaries, sources = [], [], [], []
    if target_record is not None:
        sources.append(target_path)
    for name, root, experiment, arm in (('R89B', r89_root, 'R89', 'B_muon_hybrid'),
                                        ('R84A', r84_root, 'R84', 'A_baseline')):
        paths = [root / filename for filename in ('protocol.json', 'final_test_lock.json', 'test_results.json')]
        path = root / 'evaluation/test/selected_best/rank000.npz'
        if not all(p.is_file() for p in [*paths, path]):
            missing = dict(model='R97/A', baseline=name, role='main' if name == 'R89B' else 'secondary',
                split='TEST', problem='ALL', status='missing', reason='Missing saved full TEST historical reference')
            comparisons.append(missing)
            summaries.append(missing)
            continue
        old_config, lock, saved = map(common.read_json, paths)
        if (old_config.get('experiment') != experiment or old_config.get('arm') != arm
                or Path(old_config['data_root']).resolve() != Path(config['data_root']).resolve()
                or any(old_config[key] != config[key] for key in
                       ('rows_per_problem', 'candidate_pools', 'candidate_pool_ids', 'baselines'))
                or lock.get('state_hash') != common.json_hash(old_config)
                or lock.get('selected_without_test') is not True or lock.get('per_problem_assembly') is not False
                or saved.get('complete') is not True or saved.get('checkpoint_sha256') != lock['checkpoint_sha256']):
            raise ValueError('Historical reference differs from its unique same-data lock: ' + name)
        result = saved.get('result', saved)
        if set(result['per_problem']) != set(PROBLEMS):
            raise ValueError('Historical TEST is not full18: ' + name)
        if name == 'R84A' and target_record is not None:
            if target_record['identity']['baseline_checkpoint_sha256'] != lock['checkpoint_sha256']:
                raise ValueError('Fixed36 targets refer to a different R84A checkpoint')
            state = target_record['identity'].get('baseline_state_hash', lock['state_hash'])
            if state != lock['state_hash']:
                raise ValueError('Fixed36 targets refer to a different R84A protocol')
        with np.load(path, allow_pickle=False) as archive:
            old = {p: load_predictions(archive, path, p + '__', legacy=True) for p in PROBLEMS}
        for problem, data in old.items():
            np.testing.assert_allclose(data.top1_pct, result['per_problem'][problem]['top1'] * 100., atol=1e-9, rtol=0)
            np.testing.assert_allclose(data.selected_costs.mean(), result['per_problem'][problem]['mean_cost'], atol=1e-9, rtol=0)
        comparisons.extend(paired_comparison(current, old, name))
        if targets is None:
            summaries.append(dict(baseline=name, split='TEST', status='missing', reason='Fixed36 target list missing'))
        else:
            rows = target_recovery(current, old, targets, name)
            recovery.extend(rows)
            summary = dict(baseline=name, split='TEST', target_groups=36, status='complete')
            for key in ('winner_count', 'correct_count', 'selected_count', 'rescued', 'lost', 'net_recovery',
                        'false_pick_count', 'new_false_pick_count'):
                summary[key] = sum(row[key] for row in rows)
            for key in ('new_false_pick_regret_contribution_pp', 'target_regret_delta_pp'):
                summary[key] = sum(row[key] for row in rows) / len(PROBLEMS)
            summary['recall_pct'] = (100. * summary['correct_count'] / summary['winner_count']
                                     if summary['winner_count'] else None)
            summary['all_test_regret_delta_pp'] = next(row['actual_regret_delta_pp'] for row in comparisons
                if row['baseline'] == name and row['problem'] == 'ALL')
            summaries.append(summary)
        sources.extend([*paths, path])
    return comparisons, recovery, summaries, sources


def context_resources(root, config):
    timing_path, witness_path = root / 'timing.json', root / 'resource_witness.json'
    records = common.read_json(timing_path)['records']
    witness = common.read_json(witness_path)
    if (witness.get('complete') is not True or witness.get('labels_loaded') is not False
            or witness.get('accuracy_computed') is not False
            or witness.get('context_rows') != config['contexts']['TSP']['rows']
            or witness.get('checkpoint_sha256') != config['public_checkpoint_sha256']):
        raise ValueError('Require the unlabeled resource witness on the frozen PUBLIC checkpoint')
    rows = []
    for problem in PROBLEMS:
        calls = [record for record in records if record['problem'] == problem]
        expected = ['val', 'test', *[dataset for dataset, p in LIBS.items() if p == problem]]
        row = dict(problem=problem, context_rows=config['contexts'][problem]['rows'],
            retained_columns=config['contexts'][problem]['columns'], raw_columns=357,
            optimizer_updates=0, train_self_accuracy='N/A', epochs='N/A',
            val_test_fresh_graph_to_statistics_seconds=None, statistics_cache_build_seconds=None,
            fit_context_seconds=0., fit_context_total_seconds=0., context_read_and_verify_seconds=0.,
            predict_seconds=0., lib_native_input_to_statistics_seconds=None,
            kv_cache=False, context_reencoded=True, predict_excludes_graph_to_statistics_time=True)
        for split in expected:
            for stage in ('fit_context', 'predict'):
                selected = [r for r in calls if r['split'] == split and r['stage'] == stage]
                if len(selected) != 1:
                    raise ValueError(f'Require recorded {stage} cost, not cached-only timing: {problem}/{split}')
                record = selected[0]
                seconds = record['seconds']
                if (not isinstance(seconds, (int, float)) or not np.isfinite(seconds) or seconds < 0
                        or record['train_rows'] != row['context_rows']):
                    raise ValueError('Invalid recorded full-context timing')
                row[f'{split}_{stage}_seconds'] = seconds
                row[stage + '_seconds'] += seconds
                if stage == 'fit_context':
                    for key in ('total_seconds', 'context_read_and_verify_seconds'):
                        value = record[key]
                        if not isinstance(value, (int, float)) or not np.isfinite(value) or value < 0:
                            raise ValueError('Invalid full-context loading/fit timing')
                    if record['total_seconds'] + 1e-6 < seconds + record['context_read_and_verify_seconds']:
                        raise ValueError('Fit total_seconds must include the fit and context loading')
                    row[f'{split}_fit_context_total_seconds'] = record['total_seconds']
                    row[f'{split}_context_read_and_verify_seconds'] = record['context_read_and_verify_seconds']
                    row['fit_context_total_seconds'] += record['total_seconds']
                    row['context_read_and_verify_seconds'] += record['context_read_and_verify_seconds']
                else:
                    for key, expected_value in (('kv_cache', False), ('context_reencoded', True),
                                                ('excludes_graph_to_statistics_time', True)):
                        if key in record and record[key] is not expected_value:
                            raise ValueError('Prediction timing boundary differs from the frozen protocol: ' + key)
            if split in LIBS:
                selected = [r for r in calls if r['split'] == split and r['stage'] == 'native_input_to_statistics']
                if len(selected) != 1:
                    raise ValueError('Require recorded native LIB input-to-statistics time: ' + split)
                seconds = selected[0]['seconds']
                if not isinstance(seconds, (int, float)) or not np.isfinite(seconds) or seconds < 0:
                    raise ValueError('Invalid native LIB input-to-statistics timing')
                row[f'{split}_native_input_to_statistics_seconds'] = seconds
                row['lib_native_input_to_statistics_seconds'] = seconds
        row['recorded_call_seconds'] = (row['fit_context_total_seconds'] + row['predict_seconds']
                                        + (row['lib_native_input_to_statistics_seconds'] or 0.))
        memory_calls = calls + ([witness['predict']] if problem == 'TSP' else [])
        for key in ('peak_sampled_rss_bytes', 'process_maxrss_bytes', 'peak_cuda_allocated_bytes', 'peak_cuda_reserved_bytes'):
            available = [r[key] for r in memory_calls if key in r]
            row[key] = max(available) if available else None
        row.update(witness_query_seconds=None,
            timing_scope='fit total + predict + LIB native input/statistics + witness query; not end-to-end; '
                         'excludes fresh VAL/TEST raw-graph-to-statistics, cache build and reporting')
        if problem == 'TSP':
            # The witness fit is also the first VAL fit in timing.json.
            row['witness_query_seconds'] = witness['predict']['seconds']
            row['recorded_call_seconds'] += witness['predict']['seconds']
            row['witness_fit_counted_once'] = True
        rows.append(row)
    return rows, records, witness, [timing_path, witness_path]


def distribution_rows(predictions, config, lock):
    rows = []
    for problem in PROBLEMS:
        datasets = [('TEST', predictions['TEST', problem])]
        datasets += [(name, predictions[name, problem]) for name, p in LIBS.items() if p == problem]
        for split, data in datasets:
            for field, kind in (('pred', 'model_selection'), ('winner', 'true_winner')):
                counts = np.bincount(getattr(data, field), minlength=len(data.pool))
                for index, solver in enumerate(config['candidate_pools'][problem]):
                    count = int(counts[data.pool.index(solver)])
                    rows.append(dict(model='R97/A', problem=problem, split=split, series=kind,
                        candidate_order=index, candidate=solver, n=data.n, count=count,
                        percentage=count / data.n * 100., model_top1_native_pct=data.top1_pct,
                        winner_rule=WINNER_RULE, checkpoint_sha256=lock['checkpoint_sha256'],
                        data_version=Path(config['data_root']).name if split == 'TEST' else 'original native ' + split))
    return rows


def _figure(figsize=(16, 10)):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    figure = Figure(figsize=figsize)
    FigureCanvasAgg(figure)
    return figure


def plot_distribution(problem, datasets, pool, version, lock):
    """Two horizontal series on each dataset's SAME axis, TEST before LIB."""
    from matplotlib.font_manager import FontProperties
    from matplotlib.ticker import PercentFormatter
    font_path = Path('/usr/share/fonts/google-noto-cjk/NotoSansCJK-Regular.ttc')
    font = FontProperties(fname=str(font_path)) if font_path.is_file() else None
    fig = _figure((20 if len(datasets) == 2 else 12, 9))
    axes = np.atleast_1d(fig.subplots(1, len(datasets)))
    fig.subplots_adjust(left=.16, right=.95, top=.78, bottom=.18, wspace=.5)
    fig.suptitle('R97/A | ' + problem + ' | Frozen Historical TRAIN Context', fontsize=18, y=.97)
    fig.text(.035, .91, f'Data: {version}\nBank SHA256: {lock["checkpoint_sha256"]}', fontsize=10)
    for ax, (split, data) in zip(axes, datasets):
        top1 = data.top1_pct
        ax.set_title(f'{problem + " TEST" if split == "TEST" else split} | n={data.n:,} | '
                     f'\u6a21\u578bTop1={top1:.2f}%', fontsize=14, fontproperties=font, pad=18)
        mapping = [data.pool.index(name) for name in pool]
        for field, offset, color, label in (('pred', -.17, '#197d85', 'Model selection'),
                                             ('winner', .17, '#c44f58', 'True native winner')):
            counts = np.bincount(getattr(data, field), minlength=len(pool))[mapping]
            percentages = counts / data.n * 100.
            ax.barh(np.arange(len(pool)) + offset, percentages, height=.29, color=color, label=label)
            for index, (count, pct) in enumerate(zip(counts, percentages)):
                ax.text(pct + .6, index + offset, f'{count:,} ({pct:.2f}%)', va='center', fontsize=8, color=color)
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
             'Saved native/FP64 ind; stable global-ID score ties; TEST release above, LIB original native data.', fontsize=10)
    fig.text(.035, .04, 'One frozen bank; no epochs, SGD, query-label inputs, inference, or checkpoint reselection in this reporter.', fontsize=9)
    return fig


def _format(value):
    if value is None:
        return 'N/A'
    if isinstance(value, (float, np.floating)):
        return f'{value:.6f}'
    return str(value)


def _table_page(title, headers, cells, notes=()):
    fig = _figure()
    fig.text(.04, .95, title, fontsize=18, weight='bold', color='#093f45')
    ax = fig.add_axes([.04, .21, .92, .67])
    ax.set_axis_off()
    table = ax.table(cellText=[[_format(value) for value in row] for row in cells],
                     colLabels=headers, cellLoc='right', loc='upper center', bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(8 if len(headers) > 8 else 10)
    for (row, column), cell in table.get_celld().items():
        cell.set_edgecolor('#dfe6e9')
        if row == 0:
            cell.set_facecolor('#093f45')
            cell.set_text_props(color='white', weight='bold')
        elif row % 2:
            cell.set_facecolor('#eef4f4')
    for i, note in enumerate(notes):
        fig.text(.04, .16 - .035*i, note, fontsize=9)
    return fig


def _notes_page(config, lock):
    fig = _figure()
    fig.text(.04, .95, 'R97/A | Evidence And Method Boundaries', fontsize=20, weight='bold', color='#093f45')
    text = '\n\n'.join(textwrap.fill(note, 135) for note in NOTES)
    fig.text(.04, .89, text, va='top', fontsize=10, linespacing=1.25)
    fig.text(.04, .09, f'Package: {config["package_version"]} | source revision: {config["source_revision"]}\n'
             f'Public weights: {config["public_checkpoint_sha256"]}\nBank: {lock["checkpoint_sha256"]}', fontsize=9)
    return fig


def recovery_pages(pairs, recovery, summaries):
    pages = []
    for baseline in ('R89B', 'R84A'):
        summary = next(row for row in summaries if row['baseline'] == baseline)
        pair = next(row for row in pairs if row['baseline'] == baseline and row['problem'] == 'ALL')
        headers = ['Baseline', 'Status', 'Corrected', 'Lost correct', 'Top1 delta pp', 'Cost delta', 'Regret delta pp']
        pages.append(_table_page('R97/A | Full18 TEST vs ' + baseline + ' (historical)', headers,
            [[baseline, pair['status'], pair.get('corrected'), pair.get('harmed'), pair.get('top1_delta_pp'),
              pair.get('mean_cost_delta'), pair.get('actual_regret_delta_pp')]],
            ['Main historical comparison: R89B. Secondary: R84A. Not newly compute-matched controls.',
             'Positive cost/regret delta is worse. ALL uses equal problem weight.',
             'Fixed36 recovery status: ' + summary['status'] + '. Fewer zero-selected groups do not prove success.']))
        rows = [row for row in recovery if row['baseline'] == baseline]
        for start in range(0, len(rows), 18):
            chunk = rows[start:start+18]
            headers = ['Task / Target', 'Winners', 'Correct', 'Rescued', 'Lost', 'New false', 'Recall %', 'Regret delta pp']
            pages.append(_table_page(f'R97/A | Fixed36 TEST Targets vs {baseline} | {start+1}-{start+len(chunk)}', headers,
                [[r['problem'] + '/' + r['solver'], r['winner_count'], r['correct_count'], r['rescued'], r['lost'],
                  r['new_false_pick_count'], r['recall_pct'], r['target_regret_delta_pp']] for r in chunk],
                ['Targets were frozen from R84A VAL, not redefined from R97 TEST selections.',
                 'Regret contribution is normalized by the whole task population, not just the target support.',
                 'Target CSV also includes precision, total false picks, net recovery, and new-false-pick regret.']))
    return pages


def build_pdfs(root, output, config, lock, predictions, metrics, nss, pairs, recovery, summaries, resources):
    from matplotlib.backends.backend_pdf import PdfPages
    paths = dict(model_statistics_pdf=output / 'MODEL_STATISTICS.pdf',
                 selection_distributions_pdf=output / 'selection_distributions.pdf',
                 target_recovery_pdf=output / 'target_recovery.pdf')
    temporaries = {name: path.with_name(f'.{path.stem}.{os.getpid()}.pdf') for name, path in paths.items()}
    try:
        with PdfPages(temporaries['model_statistics_pdf']) as pdf, \
                PdfPages(temporaries['selection_distributions_pdf']) as dist, \
                PdfPages(temporaries['target_recovery_pdf']) as target:
            def save(figure, *destinations):
                for destination in destinations:
                    destination.savefig(figure)
                figure.clear()

            save(_notes_page(config, lock), pdf)
            keys = ['problem', 'n', 'top1_pct', 'top2_pct', 'top3_pct', 'mean_cost', 'sbs_cost', 'oracle_cost',
                    'vs_sbs_pct', 'vs_oracle_pct', 'actual_regret_pct']
            headers = ['Problem', 'n', 'Top1 %', 'Top2 %', 'Top3 %', 'mean_cost', 'SBS cost', 'Oracle',
                       'vs_SBS %', 'vs_Oracle %', 'Actual regret %']
            for split in ('VAL', 'TEST'):
                rows = [row for row in metrics if row['split'] == split]
                save(_table_page('R97/A | ' + split + ' | ALL And 18 Tasks', headers,
                    [[row[key] for key in keys] for row in rows],
                    ['Native/FP64 ind. Same frozen context/public weights and candidate pools on VAL/TEST/LIB.',
                     'SBS fitted on R84 full TRAIN; ALL is equal-problem, raw costs descriptive only.',
                     'vs_* are ratios of mean costs; actual regret is the mean of instance-relative regrets.']), pdf)
            lib_rows = [row for row in metrics if row['split'] in LIBS]
            save(_table_page('R97/A | Native LIB Metrics', ['Dataset', *headers[1:]],
                [[row['split'], *[row[key] for key in keys[1:]]] for row in lib_rows],
                ['Same unique R97 bank as full18 VAL/TEST. Native/FP64 labels; no per-dataset tuning.']), pdf)
            save(_table_page('R97/A | NSS Four-Dataset Comparison',
                ['Dataset', 'Status', 'n', 'NSS Top1 %', 'R97 Top1 %', 'NSS cost', 'R97 cost'],
                [[r['dataset'], r['status'], r['n'], r['nss_top1_native_pct'], r['current_top1_native_pct'],
                  r['nss_mean_cost'], r['current_mean_cost']] for r in nss],
                ['NSS: saved R79 historical70,114 TRAIN weights. No new81,000 retrain or reporter inference.',
                 'All rows and native cost columns align by instance ID / candidate name. Missing stays missing.']), pdf)
            for figure in recovery_pages(pairs, recovery, summaries):
                save(figure, pdf, target)
            for start in range(0, len(resources), 18):
                save(_table_page('R97/A | Full-Context Resource And Call Timing',
                    ['Task', 'Context n', 'Columns', 'Fit total s', 'Read/verify s', 'Predict s', 'LIB input s', 'Recorded sum s'],
                    [[r['problem'], r['context_rows'], r['retained_columns'], r['fit_context_total_seconds'],
                      r['context_read_and_verify_seconds'], r['predict_seconds'],
                      r['lib_native_input_to_statistics_seconds'], r['recorded_call_seconds']]
                     for r in resources[start:start+18]],
                    ['Fit total includes read/verify (do not add it twice); TSP witness fit reused in VAL, counted once.',
                     'kv_cache=False: every predict re-encodes full context; LIB input includes native loading/geometry/statistics.',
                     'VAL/TEST use cached statistics, excluding fresh raw-graph-to-statistics time. Not end-to-end latency.']), pdf)
            for problem in PROBLEMS:
                datasets = [('TEST', predictions['TEST', problem])]
                datasets += [(name, predictions[name, problem]) for name, p in LIBS.items() if p == problem]
                save(plot_distribution(problem, datasets, config['candidate_pools'][problem],
                     Path(config['data_root']).name, lock), pdf, dist)
        for name, path in paths.items():
            temporaries[name].replace(path)
    finally:
        for path in temporaries.values():
            path.unlink(missing_ok=True)
    results = root / 'RESULTS.pdf'
    temporary = root / f'.RESULTS.{os.getpid()}.pdf'
    try:
        shutil.copyfile(paths['model_statistics_pdf'], temporary)
        temporary.replace(results)
    finally:
        temporary.unlink(missing_ok=True)
    return dict(paths, results_pdf=results)


def _md_table(headers, rows):
    safe = lambda value: _format(value).replace('|', '/').replace('\n', ' ')
    return ['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join('---' for _ in headers) + ' |',
            *['| ' + ' | '.join(safe(value) for value in row) + ' |' for row in rows]]


def write_markdown(root, config, lock, metrics, nss, pairs, summaries, resources):
    lines = ['# R97/A Final Results', '', '## Evidence Boundaries', '',
             *['- ' + note for note in NOTES], '',
             f'Data: `{config["data_root"]}`', f'Bank SHA256: `{lock["checkpoint_sha256"]}`',
             f'Public weights: `{config["public_checkpoint_sha256"]}`',
             f'Package/source: `{config["package_version"]}` / `{config["source_revision"]}`', '',
             'Main PDF: [RESULTS.pdf](RESULTS.pdf). Detailed CSV/PDF: `full_model_statistics/`.', '']
    keys = ['problem', 'n', 'top1_pct', 'top2_pct', 'top3_pct', 'mean_cost', 'sbs_cost', 'oracle_cost',
            'vs_sbs_pct', 'vs_oracle_pct', 'actual_regret_pct']
    headers = ['Problem', 'n', 'Top1 %', 'Top2 %', 'Top3 %', 'mean_cost', 'SBS cost', 'Oracle cost',
               'vs_SBS %', 'vs_Oracle %', 'actual regret %']
    for split in ('VAL', 'TEST', 'TSPLIB', 'CVRPLIB'):
        lines += ['## ' + split, '', *_md_table(headers,
                  [[row[key] for key in keys] for row in metrics if row['split'] == split]), '']
    lines += ['## NSS Reference', '', *_md_table(
        ['Dataset', 'Status', 'NSS Top1 %', 'R97 Top1 %', 'NSS mean_cost', 'R97 mean_cost'],
        [[r['dataset'], r['status'], r['nss_top1_native_pct'], r['current_top1_native_pct'],
          r['nss_mean_cost'], r['current_mean_cost']] for r in nss]), '',
        '## Historical Full-TEST Comparison', '', *_md_table(
        ['Baseline', 'Role', 'Status', 'Corrected', 'Harmed', 'Top1 delta pp', 'mean_cost delta', 'actual regret delta pp'],
        [[r['baseline'], r['role'], r['status'], r.get('corrected'), r.get('harmed'), r.get('top1_delta_pp'),
          r.get('mean_cost_delta'), r.get('actual_regret_delta_pp')] for r in pairs if r['problem'] == 'ALL']), '',
        '## Fixed36 Target Recovery', '', *_md_table(
        ['Baseline', 'Status', 'True rescued', 'True lost', 'Net recovery', 'New false picks', 'ALL regret delta pp'],
        [[r['baseline'], r['status'], r.get('rescued'), r.get('lost'), r.get('net_recovery'),
          r.get('new_false_pick_count'), r.get('all_test_regret_delta_pp')] for r in summaries]), '',
        'Positive cost/regret deltas are worse. Fixed36 comes from the frozen R84 VAL target list, not R97 TEST.', '',
        '## Context And Resource Summary', '', *_md_table(
        ['Task', 'Context n', 'Columns', 'Fit total s', 'Read/verify s', 'Predict s', 'LIB input/statistics s', 'Recorded sum s'],
        [[r['problem'], r['context_rows'], r['retained_columns'], r['fit_context_total_seconds'],
          r['context_read_and_verify_seconds'], r['predict_seconds'], r['lib_native_input_to_statistics_seconds'],
          r['recorded_call_seconds']] for r in resources]), '',
        'Fit total includes context read/verification and the fit call. Read/verify is a component, not added again.',
        '`kv_cache=False`, `context_reencoded=True`: **every prediction re-encodes all 81,000 context rows**.',
        '**VAL/TEST use canonical R92 statistics caches. Fresh raw-graph-to-statistics/cache-build latency is excluded (N/A).**',
        'LIB input/statistics time includes the native loader, geometry and all 357 statistics. This is not full end-to-end latency.',
        'TSP witness fit is also its VAL fit and is counted once. Its separate unlabeled witness query is added once.',
        'Epochs, training loss, TRAIN self-context accuracy, and learning curves: **N/A**.', '']
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
        raise ValueError('R97 final reporting is CPU-only; use --device cpu')
    root = Path(args.root).resolve()
    config, lock, results, libs = locked_inputs(root)
    output = root / 'full_model_statistics'
    status_path = root / 'postprocess_status.json'
    with (root / 'postprocess.lock').open('a') as claim:
        fcntl.flock(claim, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            primary_sources = [root / name for name in ('protocol.json', 'checkpoint.json', 'final_test_lock.json',
                'val_results.json', 'test_results.json', 'lib_results.json', 'run_complete.json',
                'timing.json', 'resource_witness.json', 'evaluation/val/selected_best/rank000.npz',
                'evaluation/test/selected_best/rank000.npz', 'lib_evaluation/predictions/TSPLIB.npz',
                'lib_evaluation/predictions/CVRPLIB.npz')]
            hashes = {str(path): common.file_hash(path) for path in primary_sources}
            common.write_json(status_path, dict(experiment='R97', complete=False, phase='loading_saved_predictions',
                                               output=str(output), updated_utc=common.utc()))
            predictions, metrics, methods, sources = saved_evaluations(root, config, lock, results, libs)
            nss_root = Path(getattr(args, 'nss_root', None) or DEFAULT_R84 / 'nss_reference').resolve()
            r84_root = Path(getattr(args, 'r84_root', None) or DEFAULT_R84 / 'A_baseline').resolve()
            r89_root = Path(getattr(args, 'r89_root', None) or DEFAULT_R89).resolve()
            target_path = Path(getattr(args, 'target_list', None) or DEFAULT_R84 / 'comparison_artifacts/target_list.json').resolve()
            nss, nss_sources = nss_comparisons(config, lock, predictions, nss_root)
            pairs, recovery, summaries, old_sources = historical_comparisons(config, predictions, r84_root, r89_root, target_path)
            resources, timing, witness, resource_sources = context_resources(root, config)
            sources += [root / name for name in ('protocol.json', 'checkpoint.json', 'final_test_lock.json',
                        'val_results.json', 'test_results.json', 'lib_results.json', 'run_complete.json')]
            sources += nss_sources + old_sources + resource_sources
            for path in dict.fromkeys(sources):
                if str(path) not in hashes:
                    hashes[str(path)] = common.file_hash(path)
            output.mkdir(parents=True, exist_ok=True)
            common.write_json(status_path, dict(experiment='R97', complete=False, phase='rendering_cpu_reports',
                                               output=str(output), updated_utc=common.utc()))
            artifacts = {}
            tables = dict(metrics_by_problem=metrics, selection_distribution=distribution_rows(predictions, config, lock),
                single_method_comparison=methods, nss_comparison=nss, paired_comparison=pairs,
                target_recovery=recovery or [dict(status='missing', reason='See target_recovery_summary.csv')],
                target_recovery_summary=summaries, context_resource_summary=resources, timing_records=timing)
            for name, rows in tables.items():
                path = output / (name + '.csv')
                common.write_csv(path, rows)
                artifacts[name + '_csv'] = str(path)
            artifacts.update({name: str(path) for name, path in build_pdfs(root, output, config, lock,
                predictions, metrics, nss, pairs, recovery, summaries, resources).items()})
            artifacts['results_md'] = str(write_markdown(root, config, lock, metrics, nss, pairs, summaries, resources))
            for path, digest in hashes.items():
                if common.file_hash(path) != digest:
                    raise ValueError('Report input changed during rendering: ' + path)
            if locked_inputs(root) != (config, lock, results, libs):
                raise ValueError('Frozen bank/results changed after initial completion check')
            provenance = dict(experiment='R97', arm='A', checkpoint_sha256=lock['checkpoint_sha256'],
                state_hash=lock['state_hash'], selection_rule=SELECTION_RULE, input_sha256=hashes,
                classifier_parameters=config['classifier_parameters'], contexts=config['contexts'],
                public_checkpoint_sha256=config['public_checkpoint_sha256'], package_version=config['package_version'],
                public_checkpoint_revision=config.get('public_checkpoint_revision'),
                source_revision=config['source_revision'], runtime_wheels=config.get('runtime_wheels', {}),
                resource_witness=witness, notes=list(NOTES),
                inference_performed=False, reference_retraining=False, train_self_accuracy=None, learning_curves=None)
            provenance_path = output / 'provenance.json'
            common.write_json(provenance_path, provenance)
            artifacts['provenance_json'] = str(provenance_path)
            result = dict(experiment='R97', arm='A', complete=True, phase='complete', output=str(output), artifacts=artifacts,
                checkpoint_sha256=lock['checkpoint_sha256'], state_hash=lock['state_hash'], updated_utc=common.utc())
            common.write_json(status_path, result)
            return result
        except Exception as error:
            common.write_json(status_path, dict(experiment='R97', complete=False, phase='failed', output=str(output),
                error=str(error), updated_utc=common.utc()))
            raise


def parser():
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument('command', choices=('report',))
    value.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    value.add_argument('--device', choices=('cpu',), default='cpu')
    value.add_argument('--nss-root', type=Path, default=DEFAULT_R84 / 'nss_reference')
    value.add_argument('--r84-root', type=Path, default=DEFAULT_R84 / 'A_baseline')
    value.add_argument('--r89-root', type=Path, default=DEFAULT_R89)
    value.add_argument('--target-list', type=Path, default=DEFAULT_R84 / 'comparison_artifacts/target_list.json')
    return value


def main(argv=None):
    result = report(parser().parse_args(argv))
    print(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    main()

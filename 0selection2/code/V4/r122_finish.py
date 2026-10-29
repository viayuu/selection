"""Post-TEST R122 delivery: one locked BASE, native LIB, saved-only PDFs/CSVs.

Only explicit ``lib``/``finish --device cuda:N`` performs model inference, and
only on TSPLIB/CVRPLIB after the full runner TEST. No training, solver, NSS
inference, checkpoint/rule selection, queueing, copying, or Git operations.
"""

from __future__ import annotations

import argparse
import fcntl
import importlib
import json
import os
from pathlib import Path
import sys
import textwrap
import traceback

import numpy as np

from code.unified_selector.registry import GLOBAL_SOLVERS, PROBLEMS
from . import r80_reporting as common


V4 = Path(__file__).resolve().parent
DEFAULT_ROOT = V4 / 'runs/R122_hard_decision_constraints'
DEFAULT_R84 = V4 / 'runs/R84_new90k_attention/A_baseline'
DEFAULT_R94 = V4 / 'runs/R94_pretrained_node_fusion/B'
DEFAULT_NSS = DEFAULT_R84.parent / 'nss_reference'
TARGET_LIST = DEFAULT_R84.parent / 'comparison_artifacts/target_list.json'
FONT = Path('/usr/share/fonts/google-noto-cjk/NotoSansCJK-Regular.ttc')
PORTABLE_FONT = Path('/public/home/shiys/.cache/selector_fonts/20261026/NotoSansCJK-Regular.ttc')
MAX_UPDATES, EVAL_EVERY = 14400, 720
ROWS = dict(train=81000, val=4500, test=4500)
LIBS = dict(TSPLIB='TSP', CVRPLIB='CVRP')
MODEL = 'R122/model'
RUNNER_FACTORY = 'code.V4.r122_experiment'
LIB_FACTORY = 'code.V4.r122_finish'
WINNER_RULE = 'native int64 ind verified against original FP64 costs; never FP32 argmin'
CORE_METRICS = ('top1_pct', 'mean_cost', 'sbs_cost', 'oracle_cost',
                'vs_sbs_pct', 'vs_oracle_pct', 'actual_regret_pct')
EXTRA_METRICS = ('top2_pct', 'top3_pct', 'sbs_top1_pct', 'top1_fp32_argmin_pct',
                 'posthoc_best_top1_pct', 'posthoc_best_top2_pct', 'posthoc_best_top3_pct')
REQUIRED_PDFS = ('MODEL_STATISTICS.pdf', 'selection_distributions.pdf',
                 'learning_curves.pdf', 'training_and_method_diagnostics.pdf', 'fixed36_recovery.pdf')
BASELINE_SHA = '23fb9a5cf621045138fcdcfe5e6613a1bfd061c8d4dfeca108db3548dbf8031f'
FINISH_SOURCES = ('r122_finish.py', 'r80_reporting.py', 'r77_lib.py', 'nss_comparison.py',
    'r41_frozen_evaluation.py', 'r76_experiment.py', 'r116_mechanism_finish.py', 'performance_evaluation.py')


class NotReady(RuntimeError):
    """A prerequisite has not completed; never start or retry its work here."""


def runner():
    return importlib.import_module(RUNNER_FACTORY)


def remember(inputs, path):
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    digest = common.file_hash(path)
    if str(path) in inputs and inputs[str(path)] != digest:
        raise ValueError('Input changed during finish: ' + str(path))
    inputs[str(path)] = digest
    return path


def read(inputs, path):
    return common.read_json(remember(inputs, path))


def helper_sources(inputs):
    return {name: inputs[str(remember(inputs, V4 / name))] for name in FINISH_SOURCES}


def same_identity(record, lock, *, epoch_optional=True):
    for key in ('checkpoint_sha256', 'state_hash', 'selected_update'):
        if record.get(key) != lock[key]:
            raise ValueError('Receipt differs from unique R122 VAL lock: ' + key)
    if (not epoch_optional or 'epoch' in record) and record.get('epoch') != lock['epoch']:
        raise ValueError('Receipt epoch differs from the locked update-derived VAL point')


def legacy_epoch_state(state, lock):
    """An in-memory legacy API field, never a checkpoint/model modification."""
    update = state.get('successful_updates', 0)
    if (not 0 < update <= MAX_UPDATES or update % EVAL_EVERY
            or update != lock['selected_update'] or lock['epoch'] != update // EVAL_EVERY
            or state.get('state_hash') != lock['state_hash']
            or 'epoch' in state and state['epoch'] != lock['epoch']):
        raise ValueError('Cannot adapt an epoch not implied by the unique selected update')
    return dict(state, epoch=update // EVAL_EVERY)


def load_checkpoint(path, device='cpu'):
    """Typed r80_lib adapter; the unchanged R122 factory restores the model."""
    path = Path(path).resolve()
    lock = common.read_json(path.parent / 'final_test_lock.json')
    if (path != (path.parent / 'best.pt').resolve()
            or common.file_hash(path) != lock['checkpoint_sha256']
            or lock.get('experiment') != 'R122' or lock.get('arm') != 'model'
            or lock.get('per_problem_assembly') is not False):
        raise ValueError('LIB adapter accepts only the actual unique R122 best.pt')
    model, state = runner().load_checkpoint(path, device=device)
    state = legacy_epoch_state(state, lock)
    from .r77_data import parameter_hash
    if (state['args'] != common.read_json(path.parent / 'protocol.json')
            or parameter_hash(model.state_dict()) != lock['model_state_sha256']):
        raise ValueError('LIB adapter changed model/protocol identity')
    return model, state


def locked_inputs(root, inputs):
    """Receipt-only gate before prediction reads; caches and GPUs are not needed."""
    run = Path(root).resolve() / 'model'
    required = ('protocol.json', 'training_complete.json', 'final_test_lock.json',
                'test_results.json', 'test_complete.json', 'final_metrics.json', 'history.json')
    missing = [name for name in required if not (run / name).is_file()]
    if missing:
        raise NotReady('Wait for parent full train/TEST: ' + ', '.join(missing))
    values = {name: read(inputs, run / name) for name in required}
    config, lock = values['protocol.json'], values['final_test_lock.json']
    training, test, final = (values[name] for name in
                             ('training_complete.json', 'test_results.json', 'final_metrics.json'))
    runner().validate_config(config)
    update = lock.get('selected_update', 0)
    if (training.get('successful_updates') != MAX_UPDATES
            or training.get('updates_by_problem') != dict.fromkeys(PROBLEMS, 800)
            or lock.get('experiment') != 'R122' or lock.get('arm') != 'model'
            or lock.get('selected_without_test') is not True or lock.get('per_problem_assembly') is not False
            or not 0 < update <= MAX_UPDATES or update % EVAL_EVERY
            or lock.get('epoch') != update // EVAL_EVERY
            or lock.get('state_hash') != common.json_hash(config)
            or lock.get('rule') != config['checkpoint_rule']
            or lock.get('inference_rule') != config['inference_rule']
            or training.get('selected_best') != lock
            or test.get('complete') is not True or final.get('complete') is not True
            or final.get('test_read') is not True):
        raise ValueError('Require complete14400/full18 TEST and the unchanged single BASE VAL lock')
    for record in (test, final):
        same_identity(record, lock)
    if values['test_complete.json'].get('checkpoint_sha256') != lock['checkpoint_sha256']:
        raise ValueError('TEST completion belongs to another checkpoint')
    checkpoint = remember(inputs, run / 'best.pt')
    if (common.resolve_record_path(lock['checkpoint'], run) != checkpoint
            or inputs[str(checkpoint)] != lock['checkpoint_sha256']):
        raise ValueError('Locked checkpoint file/path changed')
    for name, digest in config['source_sha256'].items():
        if common.file_hash(V4 / name) != digest:
            raise ValueError('Frozen training source changed: ' + name)
    prepared = read(inputs, Path(config['preparation_root']) / 'prepared.json')
    if prepared != config['preparation_lock']:
        raise ValueError('Preparation lock changed after training')
    reference = prepared['constraint_reference']
    frozen = read(inputs, Path(reference['path']))
    if inputs[str(Path(reference['path']).resolve())] != reference['sha256'] or frozen != config['constraint_reference']:
        raise ValueError('TRAIN budgets/VAL thresholds changed after training')
    results = dict(TRAIN=final.get('train'), VAL=final.get('val'), TEST=test['result'])
    for split, result in results.items():
        if (not result or set(result['per_problem']) != set(PROBLEMS)
                or any(result['per_problem'][p]['n'] != ROWS[split.lower()] for p in PROBLEMS)):
            raise ValueError('Missing full same-checkpoint ' + split + ' population')
        identity = result['identity']
        expected = dict(experiment='R122', arm='model', split=split.lower(), successful_updates=update,
            state_hash=lock['state_hash'], model_state_sha256=lock['model_state_sha256'],
            checkpoint_sha256=lock['checkpoint_sha256'])
        if any(identity.get(key) != value for key, value in expected.items()):
            raise ValueError('Selected ' + split + ' is not the same locked BASE state')
    audit = selection_audit(config, lock, values['history.json'])
    return dict(run=run, config=config, lock=lock, results=results,
                history=values['history.json'], selection_audit=audit)


def selection_audit(config, lock, history):
    if [r['successful_updates'] for r in history] != list(range(EVAL_EVERY, MAX_UPDATES + 1, EVAL_EVERY)):
        raise ValueError('Require all20 actual R122 VAL evaluations; never fabricate history')
    reference = config['constraint_reference']['val_selection']['macro']
    keys = []
    for row in history:
        key = runner().selection_key(row['val'], reference, row['successful_updates'])
        if row.get('selection_key') != list(key) or row.get('feasible') != (key[0] == 0):
            raise ValueError('Saved VAL feasibility/ranking differs from the predeclared rule')
        keys.append(key)
    selected = min(keys)
    if (selected[-1] != lock['selected_update'] or lock.get('selection_key') != list(selected)
            or lock.get('feasible') != (selected[0] == 0)):
        raise ValueError('The existing lock does not implement its original VAL-only ranking')
    return dict(audit_only=True, checkpoint_reselected=False, selected_feasible=selected[0] == 0,
        feasible_checkpoints=sum(key[0] == 0 for key in keys),
        fallback_used=selected[0] != 0, selected_update=lock['selected_update'],
        selected_key=list(selected), val_reference=reference,
        failure_disclosure='No feasible VAL checkpoint: constraints NOT satisfied' if selected[0] else 'VAL feasible')


def read_prediction(archive, prefix, config, problem, *, n=None, nss=False, path=None):
    costs = np.asarray(archive[prefix + 'costs'])
    winner = np.asarray(archive[prefix + 'winner'])
    pool = archive[prefix + 'pool'].astype(str).tolist()
    indices = np.asarray(archive[prefix + 'indices'])
    n = len(winner) if n is None else n
    if (n <= 0 or costs.dtype != np.float64 or winner.dtype != np.int64
            or costs.shape != (n, len(pool)) or winner.shape != (n,) or indices.shape != (n,)
            or not np.isfinite(costs).all() or (costs <= 0).any()
            or len(pool) != len(set(pool)) or (winner < 0).any() or (winner >= len(pool)).any()
            or not np.array_equal(costs[np.arange(n), winner], costs.min(1))
            or len(np.unique(indices)) != n):
        raise ValueError('Require native FP64 costs/int64 winners and complete unique rows')
    result = dict(costs=costs, winner=winner, pool=pool, indices=indices)
    if not nss:
        ids = np.asarray(archive[prefix + 'pool_ids'])
        if (pool != config['candidate_pools'][problem] or ids.tolist() != config['candidate_pool_ids'][problem]
                or ids.dtype != np.int64 or ids.tolist() != sorted(set(ids.tolist()))):
            raise ValueError('Candidate pool/stable IDs differ from the frozen current model')
        score_key = prefix + ('logits' if prefix + 'logits' in archive else 'scores')
        scores = np.asarray(archive[score_key])
        if scores.shape != costs.shape or not np.isfinite(scores).all():
            raise ValueError('Missing finite original maximin scores')
        result.update(scores=scores, pool_ids=ids)
        result['pred'] = np.argsort(ids, kind='stable')[np.argmax(scores[:, np.argsort(ids, kind='stable')], axis=1)]
        if prefix + 'pred' in archive:
            np.testing.assert_array_equal(result['pred'], archive[prefix + 'pred'])
    else:
        result['pred'] = np.asarray(archive[prefix + 'pred'])
        if (result['pred'].dtype.kind not in 'iu' or result['pred'].shape != (n,)
                or (result['pred'] < 0).any() or (result['pred'] >= len(pool)).any()):
            raise ValueError('NSS must use its original saved selection/tie rule')
    for name in ('source_indices', 'release_source_indices'):
        if prefix + name in archive:
            result[name] = np.asarray(archive[prefix + name])
    result['path'] = str(path) if path else ''
    return result


def natural_split(model, split, inputs):
    config, lock, result = model['config'], model['lock'], model['results'][split]
    path = common.resolve_record_path(result['prediction_archive'], model['run'])
    identity = read(inputs, path.parent / 'identity.json')
    if identity not in (result['identity'], result.get('archive_capture_identity')):
        raise ValueError('Saved natural archive has an unattributed identity')
    required = dict(experiment=config['experiment'], arm=config['arm'], split=split.lower(),
        successful_updates=lock['selected_update'], state_hash=lock['state_hash'],
        model_state_sha256=model['results']['TEST']['identity']['model_state_sha256'])
    if (any(identity.get(k) != v for k, v in required.items())
            or identity.get('checkpoint_sha256') not in (None, lock['checkpoint_sha256'])
            or config['experiment'] == 'R122' and identity.get('checkpoint_sha256') != lock['checkpoint_sha256']):
        raise ValueError('Natural predictions belong to another model/protocol/update')
    remember(inputs, path)
    with np.load(path, allow_pickle=False) as archive:
        data = {p: read_prediction(archive, p + '__', config, p, n=ROWS[split.lower()], path=path) for p in PROBLEMS}
    for problem, item in data.items():
        np.testing.assert_array_equal(item['indices'], np.arange(ROWS[split.lower()]))
        metadata = Path(config['data_root']) / (problem + split.lower()) / 'input_metadata.npz'
        if 'release_source_indices' not in item:
            raise ValueError('Natural capture must retain release source indices')
        with np.load(remember(inputs, metadata), allow_pickle=False) as source:
            np.testing.assert_array_equal(item['release_source_indices'], source['source_index'])
    return data


def metric_row(data, baseline, *, model, split, problem, checkpoint, version, saved=None):
    from .r76_experiment import fixed_sbs_metrics
    raw = {k: data[k] for k in ('costs', 'winner', 'pool', 'pool_ids')}
    measured, pred = fixed_sbs_metrics(data['scores'], raw, baseline)
    np.testing.assert_array_equal(pred, data['pred'])
    if saved is not None:
        for key in ('top1', 'top2', 'top3', 'mean_cost', 'sbs_cost', 'oracle_cost',
                    'vs_sbs_pct', 'vs_oracle_pct', 'actual_regret_pct'):
            np.testing.assert_allclose(measured[key], saved[key], rtol=0., atol=1e-9,
                                       err_msg=split + '/' + problem + '/' + key)
    slot = data['pool'].index(baseline['sbs_solver'])
    row = dict(model=model, split=split, problem=problem, n=len(pred),
        native_correct=int((pred == data['winner']).sum()),
        top1_pct=100. * measured['top1'], top2_pct=100. * measured['top2'], top3_pct=100. * measured['top3'],
        sbs_top1_pct=100. * float(np.mean(data['winner'] == slot)),
        top1_fp32_argmin_pct=100. * float(np.mean(pred == data['costs'].astype(np.float32).argmin(1))),
        **{key: float(measured[key]) for key in CORE_METRICS if key != 'top1_pct'},
        **{'posthoc_best_top' + str(k) + '_pct': 100. * measured['best_top' + str(k)] for k in (1, 2, 3)},
        sbs_name=baseline['sbs_solver'], sbs_selection_split='frozen81000 TRAIN',
        best_method_diagnostic='posthoc split mean-cost ranking; NOT the locked TRAIN SBS',
        winner_rule=WINNER_RULE, data_version=version, candidate_pool=json.dumps(data['pool']),
        checkpoint_sha256=checkpoint, aggregation='one problem', prediction_archive=data['path'])
    return row, measured


def macro_rows(rows, metrics=CORE_METRICS + EXTRA_METRICS, counts=('native_correct',)):
    if len(rows) != len(PROBLEMS) or {r['problem'] for r in rows} != set(PROBLEMS):
        raise ValueError('ALL requires all18 tasks, never an available subset')
    total = dict(rows[0], problem='ALL', n=sum(r['n'] for r in rows),
        aggregation='equal-task mean; raw macro costs descriptive only', sbs_name='per-problem TRAIN SBS',
        candidate_pool='per-problem legal pools')
    total.update({k: float(np.mean([r[k] for r in rows])) for k in metrics})
    total.update({k: sum(r[k] for r in rows) for k in counts})
    return total


def load_libs(model, inputs):
    run, config, lock = model['run'], model['config'], model['lock']
    receipt = read(inputs, run / 'lib_stage_complete.json')
    if (receipt.get('complete') is not True or receipt.get('exit_status') != 0
            or set(receipt.get('datasets', [])) != set(LIBS)):
        raise ValueError('Require both completed same-checkpoint LIB datasets')
    same_identity(receipt, lock)
    data, results = {}, {}
    for dataset, problem in LIBS.items():
        result = read(inputs, run / 'lib_evaluation' / (dataset + '_results.json'))
        identity = result['protocol']
        same_identity(identity, lock)
        if (identity.get('experiment') != 'R122' or identity.get('arm') != 'model'
                or identity.get('model_state_sha256') != lock['model_state_sha256']
                or identity.get('dataset') != dataset or identity.get('checkpoint_reselection') is not False
                or identity.get('locked_inference_rule') != lock['inference_rule']
                or identity.get('model_factory') != LIB_FACTORY + '.load_checkpoint'):
            raise ValueError('LIB capture is not the unchanged current BASE')
        path = remember(inputs, run / 'lib_evaluation/predictions' / (dataset + '.npz'))
        with np.load(path, allow_pickle=False) as archive:
            if (str(archive['checkpoint_sha256'].item()) != lock['checkpoint_sha256']
                    or int(archive['checkpoint_update']) != lock['selected_update']
                    or int(archive['checkpoint_epoch']) != lock['epoch']):
                raise ValueError('LIB archive checkpoint differs from the unique lock')
            data[dataset] = read_prediction(archive, '', config, problem,
                                            n=result['train_sbs_reference']['n'], path=path)
        for source in identity['sources'].values():
            source_path = remember(inputs, Path(source['path']))
            if inputs[str(source_path)] != source['sha256']:
                raise ValueError('Original native LIB source changed')
        results[dataset] = result
    return data, results


def load_targets(config, inputs):
    record = read(inputs, TARGET_LIST)
    reference = config['constraint_reference']
    pairs = [(r['problem'], r['method']) for r in record['targets']]
    if (record['targets'] != reference['fixed_targets'] or len(set(pairs)) != 36 or len(pairs) != 36
            or record['identity']['baseline_checkpoint_sha256'] != BASELINE_SHA
            or inputs[str(TARGET_LIST.resolve())] != reference['fixed_target_file']['sha256']
            or any(p not in PROBLEMS or s not in config['candidate_pools'][p] for p, s in pairs)):
        raise ValueError('Require the frozen historical fixed36; never reselect the target population')
    return pairs


def error_partition(data, slots):
    n, winner, pred, costs = len(data['winner']), data['winner'], data['pred'], data['costs']
    truth = np.isin(winner, slots)
    wrong = pred != winner
    selected = costs[np.arange(n), pred]
    oracle = costs.min(1)
    count, missed = int(truth.sum()), int((truth & wrong).sum())
    return dict(n=n, winner_count=count, missed_count=missed, population_frequency=count / n,
        hard_error_frequency=missed / n, hard_regret_contribution=float(((selected - oracle) / oracle)[truth].sum() / n),
        mean_cost_contribution=float(selected[truth].sum() / n),
        conditional_native_top1_pct=100. * (count - missed) / count if count else None)


def error_budgets(predictions, targets):
    rows = []
    for split in ('TRAIN', 'VAL', 'TEST'):
        for problem in PROBLEMS:
            data = predictions[split][problem]
            fixed = [data['pool'].index(s) for p, s in targets if p == problem]
            groups = dict(FIXED36=fixed, OUTSIDE36=[i for i in range(len(data['pool'])) if i not in fixed],
                          FULL18=list(range(len(data['pool']))))
            parts = {}
            for name, slots in groups.items():
                point = dict(split=split, problem=problem, group=name, **error_partition(data, slots),
                    aggregation='full problem denominator; native-winner partition, not chosen-method quota')
                rows.append(point)
                parts[name] = point
            for key in ('winner_count', 'missed_count', 'hard_error_frequency',
                        'hard_regret_contribution', 'mean_cost_contribution'):
                np.testing.assert_allclose(parts['FIXED36'][key] + parts['OUTSIDE36'][key], parts['FULL18'][key],
                                           rtol=0., atol=1e-12)
        for group in ('FIXED36', 'OUTSIDE36', 'FULL18'):
            parts = [r for r in rows if r['split'] == split and r['group'] == group and r['problem'] != 'ALL']
            total = macro_rows(parts, ('population_frequency', 'hard_error_frequency',
                'hard_regret_contribution', 'mean_cost_contribution'), ('winner_count', 'missed_count'))
            count = total['winner_count']
            total['conditional_native_top1_pct'] = 100. * (count - total['missed_count']) / count if count else None
            total['rate_aggregation'] = 'conditional Top1 from pooled counts; unconditional constraints task-equal'
            rows.append(total)
    return rows


def constraint_rows(error_rows, config, lock):
    result = []
    reference = config['constraint_reference']
    for split in ('TRAIN', 'VAL', 'TEST'):
        source = 'train' if split == 'TRAIN' else 'val_selection'
        budget = reference[source]['macro']
        full = next(r for r in error_rows if (r['split'], r['problem'], r['group']) == (split, 'ALL', 'FULL18'))
        fixed = next(r for r in error_rows if (r['split'], r['problem'], r['group']) == (split, 'ALL', 'FIXED36'))
        for name, value, limit in (('hard_macro_actual_regret', full['hard_regret_contribution'], budget['hard_regret']),
                                  ('fixed36_unconditional_missed_frequency', fixed['hard_error_frequency'], budget['hard_fixed_miss'])):
            result.append(dict(split=split, constraint=name, observed=value, threshold=limit,
                observed_pct=100. * value, threshold_pct=100. * limit, violation=value - limit,
                satisfied=value <= limit, threshold_source_split='TRAIN' if split == 'TRAIN' else 'VAL',
                role='TRAIN controller budget' if split == 'TRAIN' else
                    'VAL checkpoint selection only; no dual gradients' if split == 'VAL' else
                    'descriptive TEST vs frozen VAL reference; NOT a selection gate',
                selected_val_feasible=lock['feasible'], aggregation='equal-task macro; unscaled fractions'))
    return result


def historical_model(path, experiment, arm, config, inputs):
    from . import r84_reporting as shared
    if path.name != arm:
        raise ValueError('Historical root must name the explicit whole-model arm: ' + arm)
    run, old, lock, test, val = shared.locked_inputs(path.parent, arm, experiment=experiment, arms={arm: 57600})
    for name in ('protocol.json', 'final_test_lock.json', 'training_complete.json',
                 'test_results.json', 'final_metrics.json', 'best.pt'):
        remember(inputs, run / name)
    for key in ('data_root', 'rows_per_problem', 'candidate_pools', 'candidate_pool_ids', 'baselines', 'winner_rule'):
        if old.get(key) != config.get(key):
            raise ValueError('Historical same90k reference differs: ' + key)
    return dict(run=run, config=old, lock=lock, results=dict(VAL=val, TEST=test))


def aligned(old, new):
    for key in ('pool', 'pool_ids', 'indices', 'source_indices', 'release_source_indices', 'costs', 'winner'):
        if (key in old) != (key in new):
            raise ValueError('Historical paired alignment field missing: ' + key)
        if key in old:
            np.testing.assert_array_equal(old[key], new[key], err_msg='Historical same-instance alignment: ' + key)


def paired_row(old, new, *, comparison, split, problem):
    aligned(old, new)
    a, b, costs, winner = old['pred'], new['pred'], new['costs'], new['winner']
    before, after = costs[np.arange(len(a)), a], costs[np.arange(len(b)), b]
    delta, oracle = after - before, costs.min(1)
    old_ok, new_ok = a == winner, b == winner
    return dict(comparison=comparison, split=split, problem=problem, n=len(a),
        corrected=int((~old_ok & new_ok).sum()), harmed=int((old_ok & ~new_ok).sum()),
        net_correct=int(new_ok.sum() - old_ok.sum()), changed=int((a != b).sum()),
        both_wrong_cheaper=int((~old_ok & ~new_ok & (delta < 0)).sum()),
        both_wrong_dearer=int((~old_ok & ~new_ok & (delta > 0)).sum()),
        top1_delta_pp=100. * float(new_ok.mean() - old_ok.mean()), mean_cost_delta=float(delta.mean()),
        actual_regret_delta_pp=100. * float((delta / oracle).mean()),
        aggregation='delta=current-baseline; problem denominator')


def recovery_tables(old, new, targets, comparison, split):
    """Typed pure-array adapter; no old selection rule or event dump is reused."""
    from .r116_mechanism_finish import _augment, _rates, COUNTS, CONTRIBUTIONS
    details, summary = [], []
    for problem in PROBLEMS:
        aligned(old[problem], new[problem])
        pool = new[problem]['pool']
        fixed = [pool.index(s) for p, s in targets if p == problem]
        for solver in (s for p, s in targets if p == problem):
            details.append(_augment(dict(comparison=comparison, split=split, problem=problem,
                solver=solver, scope='fixed36_method', target_definition='frozen R84A VAL <=0.1%; never dynamic'),
                old[problem], new[problem], [pool.index(solver)]))
        groups = dict(FIXED36=fixed, OUTSIDE36=[i for i in range(len(pool)) if i not in fixed],
                      FULL18=list(range(len(pool))))
        partitions = {}
        for group, slots in groups.items():
            row = _augment(dict(comparison=comparison, split=split, problem=problem, solver=group,
                scope='native-winner partition; false picks separately grouped by predicted method'),
                old[problem], new[problem], slots)
            summary.append(row)
            partitions[group] = row
        for key in (*COUNTS, *CONTRIBUTIONS):
            np.testing.assert_allclose(partitions['FIXED36'][key] + partitions['OUTSIDE36'][key],
                                       partitions['FULL18'][key], rtol=1e-11, atol=1e-12)
    for group in ('FIXED36', 'OUTSIDE36', 'FULL18'):
        parts = [r for r in summary if r['solver'] == group and r['problem'] != 'ALL']
        summary.append(_rates(macro_rows(parts, CONTRIBUTIONS, COUNTS)))
    return details, summary


def nss_comparisons(predictions, libs, model, nss_root, inputs):
    """All four rows, no NSS inference; missing stays missing, corrupt fails."""
    plan_path = nss_root / 'protocol.json'
    plan = read(inputs, plan_path) if plan_path.is_file() else None
    config, lock = model['config'], model['lock']
    if plan and (Path(plan['data_root']).resolve() != Path(config['data_root']).resolve()
            or plan.get('retrained_on_new90k') is not False or plan.get('expected_test_rows_per_task') != 4500):
        raise ValueError('NSS must be the frozen historical70114 model on this exact new4500 release')
    rows, packed = [], {}
    for dataset, problem, split in (('TSP TEST', 'TSP', 'TEST'), ('CVRP TEST', 'CVRP', 'TEST'),
                                    ('TSPLIB', 'TSP', 'TSPLIB'), ('CVRPLIB', 'CVRP', 'CVRPLIB')):
        current = predictions['TEST'][problem] if split == 'TEST' else libs[split]
        name = 'test' if split == 'TEST' else split
        path = nss_root / problem / (name + '_predictions.npz')
        metadata_path = path.with_name(name + '_metrics.json')
        row = dict(dataset=dataset, problem=problem, split=split, n=len(current['winner']),
            current_model=MODEL, current_top1_native_pct=100. * float(np.mean(current['pred'] == current['winner'])),
            current_mean_cost=float(current['costs'][np.arange(len(current['pred'])), current['pred']].mean()),
            current_checkpoint_sha256=lock['checkpoint_sha256'], nss_top1_native_pct=None, nss_mean_cost=None,
            nss_checkpoint_sha256=None, nss_model='R79 historical70114 TRAIN NSS; NOT new90k retraining',
            current_training_data='new81000 TRAIN / task', evaluation_data=Path(config['data_root']).name
                if split == 'TEST' else 'original native ' + split,
            candidate_pool=json.dumps(current['pool']), winner_rule=WINNER_RULE,
            status='missing_saved_NSS', retrained=False, new_nss_inference=False)
        if plan and path.is_file() and metadata_path.is_file():
            metadata = read(inputs, metadata_path)
            expected = plan['checkpoints'][problem]['sha256']
            if metadata.get('checkpoint', {}).get('sha256') != expected:
                raise ValueError('Saved NSS checkpoint identity changed')
            if split == 'TEST':
                once = read(inputs, path.with_name('test_once.json'))
                if once.get('state') != 'complete' or once.get('predictions_sha256') != common.file_hash(path):
                    raise ValueError('Saved new4500 NSS predictions changed')
            remember(inputs, path)
            with np.load(path, allow_pickle=False) as archive:
                data = read_prediction(archive, '', config, problem, n=len(current['winner']), nss=True, path=path)
            if set(data['pool']) != set(current['pool']) or set(data['indices']) != set(current['indices']):
                raise ValueError('NSS instances/candidates are not the current exact datasets')
            row_lookup = {int(index): i for i, index in enumerate(data['indices'])}
            order = np.array([row_lookup[int(i)] for i in current['indices']])
            columns = [data['pool'].index(s) for s in current['pool']]
            inverse = np.argsort(columns)
            data.update(costs=data['costs'][order][:, columns], winner=inverse[data['winner'][order]],
                pred=inverse[data['pred'][order]], indices=data['indices'][order], pool=list(current['pool']))
            np.testing.assert_array_equal(data['costs'], current['costs'])
            np.testing.assert_array_equal(data['winner'], current['winner'])
            top1 = 100. * float(np.mean(data['pred'] == data['winner']))
            cost = float(data['costs'][np.arange(len(order)), data['pred']].mean())
            np.testing.assert_allclose(top1, 100. * metadata['top1_native_ind'], rtol=0., atol=1e-9)
            np.testing.assert_allclose(cost, metadata['mean_cost'], rtol=0., atol=1e-9)
            row.update(status='complete', nss_top1_native_pct=top1, nss_mean_cost=cost,
                nss_checkpoint_sha256=expected, aligned_rows=len(order),
                source_index_note='split-local indices plus ALL FP64 costs/native winners aligned; NSS source IDs may be pre-resplit')
            packed[split, problem] = data
        rows.append(row)
    return rows, packed


def lib_stage(args):
    if not str(args.device).startswith('cuda:'):
        raise ValueError('LIB requires explicit parent-assigned --device cuda:N; report is CPU-only')
    inputs = {}
    model = locked_inputs(args.root, inputs)
    run, config, lock = model['run'], model['config'], model['lock']
    sources = helper_sources(inputs)
    with (run / 'lib_stage.lock').open('a') as claim:
        fcntl.flock(claim, fcntl.LOCK_EX | fcntl.LOCK_NB)

        def phase(name, **fields):
            common.write_json(run / 'lib_stage_status.json', dict(experiment='R122', arm='model', phase=name,
                updated_utc=common.utc(), checkpoint_sha256=lock['checkpoint_sha256'], **fields))

        try:
            results = common.r80_lib(args.root, run, config, lock, args.device, args.lib_batch_size,
                phase, lock['model_state_sha256'], factory_module=LIB_FACTORY)
            if set(results) != set(LIBS):
                raise ValueError('Both native LIB datasets must finish using the same BASE')
            for path, digest in inputs.items():
                if common.file_hash(Path(path)) != digest:
                    raise ValueError('LIB input/helper changed during inference: ' + path)
            receipt = dict(experiment='R122', arm='model', complete=True, exit_status=0,
                completed_utc=common.utc(), checkpoint_sha256=lock['checkpoint_sha256'],
                state_hash=lock['state_hash'], selected_update=lock['selected_update'], epoch=lock['epoch'],
                model_state_sha256=lock['model_state_sha256'], datasets=list(results),
                model_factory=RUNNER_FACTORY + '.load_checkpoint', legacy_api_adapter=LIB_FACTORY + '.load_checkpoint',
                source_sha256=sources,
                epoch_field='derived in memory from selected_update/720; checkpoint bytes/model unchanged',
                nss_inference=False, solver_execution=False, checkpoint_reselection=False)
            common.write_json(run / 'lib_stage_complete.json', receipt)
            phase('complete', exit_status=0)
            return receipt
        except BaseException as error:
            phase('failed', exit_status=1, error=str(error), automatic_retry=False)
            raise


def set_style(font=FONT):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import font_manager
    if Path(font) == FONT and not FONT.is_file() and PORTABLE_FONT.is_file():
        font = PORTABLE_FONT
    if not Path(font).is_file():
        raise FileNotFoundError('Install/pass an existing CJK font for the mandatory model Top1 label: ' + str(font))
    font_manager.fontManager.addfont(str(font))
    matplotlib.rcParams.update({'font.family': font_manager.FontProperties(fname=str(font)).get_name(),
        'font.size': 10, 'pdf.fonttype': 42, 'axes.unicode_minus': False, 'figure.facecolor': 'white'})


def distribution_figure(problem, datasets, pool, lock, *, label=MODEL, data_version=''):
    from matplotlib import pyplot as plt
    fig, axes = plt.subplots(1, len(datasets), figsize=(20 if len(datasets) == 2 else 13, 8.7), squeeze=False)
    fig.subplots_adjust(left=.11 if len(datasets) == 2 else .19, right=.97, top=.74, bottom=.19, wspace=.46)
    fig.text(.035, .96, label + ' | ' + problem + ' | method choice vs native winner', fontsize=17, weight='bold')
    fig.text(.035, .905, 'Same checkpoint ' + lock['checkpoint_sha256'][:20] + ' | selected update ' + str(lock['selected_update']), fontsize=10)
    fig.text(.035, .86, 'TEST data: ' + data_version, fontsize=8.5)
    lib_names = ', '.join(name for name, _ in datasets if name != 'TEST')
    fig.text(.035, .815, ('LIB=original native ' + lib_names + ' | ' if lib_names else '') + WINNER_RULE, fontsize=9)
    for ax, (dataset, data) in zip(axes[0], datasets):
        if data is None:
            ax.set_title(dataset + ' | \u6a21\u578bTop1=MISSING')
            ax.text(.5, .5, 'MISSING saved predictions; no substitute inference', transform=ax.transAxes, ha='center')
            ax.set_axis_off()
            continue
        n = len(data['winner'])
        order = [data['pool'].index(name) for name in pool]
        top1 = 100. * float(np.mean(data['pred'] == data['winner']))
        ax.set_title(f'{dataset} | n={n:,} | \u6a21\u578bTop1={top1:.2f}%', fontsize=14, pad=20, weight='bold')
        for key, offset, color, title in (('pred', -.17, '#156f7c', 'Model selection'),
                                         ('winner', .17, '#bd5148', 'True native winner')):
            counts = np.bincount(data[key], minlength=len(pool))[order]
            if counts.sum() != n:
                raise ValueError('Distribution counts do not reconstruct the actual dataset')
            pct = counts * 100. / n
            ax.barh(np.arange(len(pool)) + offset, pct, height=.28, color=color, label=title, zorder=3)
            for i, (value, count) in enumerate(zip(pct, counts)):
                inside = value >= 80.
                ax.text(value - 1. if inside else value + .7, i + offset, f'{value:.2f}% ({count:,})',
                    va='center', ha='right' if inside else 'left', color='white' if inside else color, fontsize=9)
        ax.set_yticks(np.arange(len(pool)), labels=pool)
        ax.set_ylim(len(pool) - .4, -.65)
        ax.set_xlim(0, 100)
        ax.set_xlabel('Fraction of this dataset (%)')
        ax.grid(axis='x', alpha=.2, zorder=0)
        ax.spines[['top', 'right', 'left']].set_visible(False)
        ax.legend(loc='upper center', bbox_to_anchor=(.5, -.12), ncol=2, frameon=False, fontsize=9)
    fig.text(.035, .07, 'Paired bars share one axis and fixed candidate order. Labels show percent (count), including zeros.', fontsize=9)
    fig.text(.035, .035, 'Model Top1 = per-instance native winner agreement, NOT the largest selection frequency. TEST precedes LIB.', fontsize=9)
    return fig


def table_page(title, headers, rows, notes, lock, *, fontsize=8.5):
    from matplotlib import pyplot as plt
    fig, ax = plt.subplots(figsize=(16.54, 11.69))
    ax.axis('off')
    fig.text(.04, .955, title, fontsize=17, weight='bold')
    fig.text(.04, .92, MODEL + ' | same BASE ' + lock['checkpoint_sha256'][:20]
             + ' | update ' + str(lock['selected_update']), fontsize=10)
    cells = rows or [['MISSING reference; see provenance', *[''] * (len(headers) - 1)]]
    width = max(16, 170 // len(headers))
    wrap = lambda value: textwrap.fill(str(value), width=width, break_long_words=True, break_on_hyphens=False)
    table = ax.table(cellText=[[wrap(cell) for cell in row] for row in cells],
        colLabels=[wrap(h) for h in headers], bbox=[0., .24, 1., .64], cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(fontsize)
    for (r, _), cell in table.get_celld().items():
        if r == 0:
            cell.set_facecolor('#e5eff0')
            cell.set_text_props(weight='bold')
        cell.set_edgecolor('#d8e1e1')
    lines = [line for text in notes for line in textwrap.wrap(str(text), 150)]
    for i, line in enumerate(lines[:12]):
        fig.text(.04, .22 - i * .015, line, fontsize=8)
    return fig


def fmt(value, digits=5):
    if value is None:
        return 'MISSING'
    return format(value, '.' + str(digits) + 'f') if isinstance(value, (float, np.floating)) else str(value)


def history_rows(model):
    rows = []
    for row in model['history']:
        opt = row.get('optimization', {})
        point = dict(successful_updates=row['successful_updates'], selected=row['successful_updates'] == model['lock']['selected_update'],
            val_feasible=row['feasible'], training_loss=opt.get('loss'), training_classification_proxy_bits=opt.get('ce'),
            training_regret_proxy=opt.get('risk'), train_sampled_top1_pct=None if 'hard_error' not in opt else 100. * (1. - opt['hard_error']),
            train_sampled_actual_regret_pct=None if 'hard_regret' not in opt else 100. * opt['hard_regret'],
            train_sampled_fixed_miss_pct=None if 'hard_fixed_miss' not in opt else 100. * opt['hard_fixed_miss'])
        for split in ('train', 'val'):
            saved = row.get(split)
            for key in ('macro_top1', 'macro_actual_regret_pct', 'macro_mean_cost', 'macro_vs_sbs_pct', 'macro_ce'):
                point[split + '_full_' + key] = None if saved is None else saved['macro'][key] * (100. if key == 'macro_top1' else 1.)
            point[split + '_full_fixed_miss_pct'] = (None if saved is None or 'hard_decision' not in saved else
                100. * saved['hard_decision']['macro']['hard_fixed_miss'])
        rows.append(point)
    return rows


def curve_figure(model, rows):
    from matplotlib import pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(16.5, 9.5))
    fig.suptitle('D2: actual TRAIN optimization feedback and full VAL history; one locked BASE', fontsize=15)
    panels = (
        (axes[0, 0], ('training_loss', 'training_classification_proxy_bits', 'training_regret_proxy'), 'TRAIN proxy objective; CE in bits'),
        (axes[0, 1], ('train_sampled_top1_pct', 'train_full_macro_top1', 'val_full_macro_top1'), 'Native Top1 (%)'),
        (axes[0, 2], ('train_sampled_actual_regret_pct', 'train_full_macro_actual_regret_pct', 'val_full_macro_actual_regret_pct'), 'Hard actual regret (%)'),
        (axes[1, 0], ('train_sampled_fixed_miss_pct', 'train_full_fixed_miss_pct', 'val_full_fixed_miss_pct'), 'Unconditional fixed36 missed frequency (%)'),
        (axes[1, 1], ('train_full_macro_vs_sbs_pct', 'val_full_macro_vs_sbs_pct'), 'vs frozen TRAIN SBS (%)'),
        (axes[1, 2], ('train_full_macro_ce', 'val_full_macro_ce'), 'Full evaluation maximin CE (nats)'))
    for ax, names, title in panels:
        for name in names:
            points = [(r['successful_updates'], r[name]) for r in rows if r.get(name) is not None]
            if points:
                ax.plot(*zip(*points), marker='o' if len(points) == 1 else '.', label=name.replace('_', ' '), markersize=4)
        ax.set_title(title, fontsize=10)
        ax.axvline(model['lock']['selected_update'], color='#bd5148', linestyle='--', linewidth=1.)
        ax.set_xlabel('Successful logical1024 updates')
        ax.grid(alpha=.2)
        if ax.lines:
            ax.legend(fontsize=6)
    reference = model['config']['constraint_reference']
    for ax, field in ((axes[0, 2], 'hard_regret'), (axes[1, 0], 'hard_fixed_miss')):
        for source, color, label in (('train', '#156f7c', 'TRAIN budget'), ('val_selection', '#bd5148', 'VAL selection threshold')):
            ax.axhline(100. * reference[source]['macro'][field], color=color, linestyle=':', linewidth=1., label=label)
        ax.legend(fontsize=6)
    fig.text(.035, .025, 'Sampled TRAIN feedback is NOT full TRAIN evaluation. Full TRAIN points exist only where actually saved. No interpolation or TEST/LIB selection.', fontsize=9)
    fig.tight_layout(rect=(0, .055, 1, .95))
    return fig


def task_curve_figure(model, problem):
    from matplotlib import pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(16.5, 9.5))
    fig.suptitle(MODEL + ' | ' + problem + ' | actual saved full TRAIN/VAL evaluations', fontsize=15)
    records = []
    for split in ('train', 'val'):
        points = [(r['successful_updates'], r[split]['per_problem'][problem])
                  for r in model['history'] if r.get(split) is not None]
        for update, saved in points:
            records.append(dict(problem=problem, split=split.upper(), successful_updates=update, n=saved['n'],
                **{'top' + str(k) + '_pct': 100. * saved['top' + str(k)] for k in (1, 2, 3)},
                **{k: saved[k] for k in ('mean_cost', 'actual_regret_pct', 'vs_sbs_pct')},
                source='actual full evaluation in own history; not sampled optimization feedback'))
        for k in (1, 2, 3):
            if points:
                axes[0, 0].plot([u for u, _ in points], [100. * v['top' + str(k)] for _, v in points],
                    marker='o' if len(points) == 1 else '.', label=split.upper() + ' Top' + str(k))
        for ax, key, title in ((axes[0, 1], 'mean_cost', 'Mean original cost'),
                               (axes[1, 0], 'actual_regret_pct', 'Actual regret (%)'),
                               (axes[1, 1], 'vs_sbs_pct', 'vs frozen TRAIN SBS (%)')):
            if points:
                ax.plot([u for u, _ in points], [v[key] for _, v in points],
                    marker='o' if len(points) == 1 else '.', label=split.upper())
            ax.set_title(title)
    axes[0, 0].set_title('Native Top1/2/3 (%)')
    for ax in axes.flat:
        ax.axvline(model['lock']['selected_update'], color='#bd5148', linestyle='--', linewidth=1.)
        ax.set_xlabel('Successful logical1024 updates')
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.text(.035, .025, 'Full TRAIN points exist only where saved; no fabricated dense TRAIN curve. Per-task TRAIN proxy losses were not logged; see global optimization page.', fontsize=9)
    fig.tight_layout(rect=(0, .055, 1, .95))
    return fig, records


def method_figure(problem, predictions, metrics, lock):
    from matplotlib import pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(16.5, 10))
    fig.suptitle(MODEL + ' | ' + problem + ' | native method observables', fontsize=15)
    for column, split in enumerate(('VAL', 'TEST')):
        data = predictions[split][problem]
        row = next(r for r in metrics if r['split'] == split and r['problem'] == problem)
        y = np.arange(len(data['pool']))
        accuracy = 100. * np.bincount(data['winner'], minlength=len(y)) / len(data['winner'])
        mean_cost = data['costs'].mean(0)
        for ax, values, selected, title in ((axes[0, column], accuracy, row['top1_pct'], 'Native Top1 (%)'),
                                             (axes[1, column], mean_cost, row['mean_cost'], 'Mean original cost')):
            ax.barh(y, values, color='#87b9bf')
            ax.axvline(selected, color='#bd5148', label='Selector same BASE')
            ax.set_yticks(y, labels=data['pool'], fontsize=8)
            ax.invert_yaxis()
            ax.set_title(split + ' | ' + title)
            ax.legend(fontsize=8)
        axes[0, column].text(0., -.2, 'Selector Top1/2/3: ' + '/'.join(fmt(row['top' + str(k) + '_pct'], 2) for k in (1, 2, 3))
            + '%; posthoc best1/2/3: ' + '/'.join(fmt(row['posthoc_best_top' + str(k) + '_pct'], 2) for k in (1, 2, 3)) + '%',
            transform=axes[0, column].transAxes, fontsize=8)
    fig.text(.035, .025, 'Posthoc best-method ranking is diagnostic only, not the fixed TRAIN SBS. Selection frequencies are not model accuracy. SHA ' + lock['checkpoint_sha256'][:16], fontsize=9)
    fig.tight_layout(rect=(0, .065, 1, .95), h_pad=4.)
    return fig


def distribution_rows(problem, split, data, lock, *, label=MODEL, version=''):
    rows = []
    n = len(data['winner'])
    for slot, name in enumerate(data['pool']):
        picked, truth = data['pred'] == slot, data['winner'] == slot
        rows.append(dict(model=label, problem=problem, split=split, candidate_order=slot, candidate=name, n=n,
            selected_count=int(picked.sum()), winner_count=int(truth.sum()), correct_count=int((picked & truth).sum()),
            selected_pct=100. * float(picked.mean()), winner_pct=100. * float(truth.mean()),
            model_top1_native_pct=100. * float(np.mean(data['pred'] == data['winner'])),
            checkpoint_sha256=lock['checkpoint_sha256'], winner_rule=WINNER_RULE, data_version=version))
    return rows


def render_reports(output, model, predictions, libs, metrics, methods, nss_rows, nss_data,
                   constraints, budgets, comparisons, recovery, refs, claims, font):
    set_style(font)
    from matplotlib import pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    lock, config = model['lock'], model['config']
    version = Path(config['data_root']).name
    rows = history_rows(model)
    common.write_csv(output / 'learning_curves.csv', rows)
    task_history = []
    with PdfPages(output / 'learning_curves.pdf') as pdf:
        fig = curve_figure(model, rows)
        pdf.savefig(fig)
        plt.close(fig)
        for problem in PROBLEMS:
            fig, actual = task_curve_figure(model, problem)
            task_history.extend(actual)
            pdf.savefig(fig)
            plt.close(fig)
    common.write_csv(output / 'learning_curves_by_problem.csv', task_history)
    with PdfPages(output / 'training_and_method_diagnostics.pdf') as pdf:
        for problem in PROBLEMS:
            fig = method_figure(problem, predictions, metrics, lock)
            pdf.savefig(fig)
            plt.close(fig)
    recovery_all = [r for r in recovery if r['problem'] == 'ALL']
    with PdfPages(output / 'fixed36_recovery.pdf') as pdf:
        if not recovery_all:
            fig = table_page('Fixed36 recovery | missing historical reference', ['Status'], [['MISSING']],
                ['No recovery/false-pick effect is claimed without aligned historical predictions.'], lock)
            pdf.savefig(fig)
            plt.close(fig)
        for comparison in sorted({r['comparison'] for r in recovery_all}):
            for split in ('VAL', 'TEST'):
                chosen = [r for r in recovery_all if r['comparison'] == comparison and r['split'] == split]
                cells = [[r['solver'], r['winner_count'], r['true_rescues'], r['lost_true_picks'], r['net_correct'],
                    r['current_true_tp'], r['false_pick_count'], r['new_false_pick_count'],
                    fmt(r['actual_regret_delta_contribution_pp'])] for r in chosen]
                fig = table_page(comparison + ' | ' + split + ': genuine recovery and false picks',
                    ['Group', 'Winners', 'Rescued', 'Lost TP', 'Net TP', 'Current TP', 'False picks', 'New false', 'dRegret pp'], cells,
                    ['Fixed/outside regret partitions use the native winner and full-problem denominator; ALL task-equal.',
                     'False picks are grouped by the predicted method; they are NOT true recovery or evidence from a nonzero selection count.',
                     'Historical comparisons are not budget-matched new controls. Positive dRegret is worse.'], lock)
                pdf.savefig(fig)
                plt.close(fig)
    dist_rows = []
    with PdfPages(output / 'MODEL_STATISTICS.pdf') as main, PdfPages(output / 'selection_distributions.pdf') as dist:
        overall = [r for r in metrics if r['problem'] == 'ALL']
        fig = table_page('R122 D2 | full18 locked outcome',
            ['Split', 'Top1 %', 'mean_cost', 'SBS', 'Oracle', 'vsSBS %', 'vsOracle %', 'Regret %'],
            [[r['split'], *[fmt(r[k]) for k in CORE_METRICS]] for r in overall],
            [model['selection_audit']['failure_disclosure'],
             'Actual full TRAIN constraints satisfied: ' + str(claims['train_constraints_satisfied'])
                + '; selected VAL constraints satisfied: ' + str(claims['val_constraints_satisfied']),
             'TRAIN budgets update the controller; separate VAL thresholds select the checkpoint only. No dual gradient uses VAL.',
             'Error/regret ALL is task-equal. Mean raw cost is descriptive; vsSBS/Oracle ratios differ from per-instance actual regret.',
             'TEST was development-exposed. No current TEST/LIB checkpoint/rule selection. No solver quotas.',
             'NSS uses historical70114-trained weights, not a new90k retrain. Histories and missing references are disclosed.',
             'Observed D2 joint TEST evidence vs R84A: ' + str(claims.get('joint_observed_test_improvement_vs_R84A'))], lock)
        main.savefig(fig)
        plt.close(fig)
        fig = table_page('What D2 optimizes, constrains and selects', ['Component', 'Frozen meaning'],
            [['Primal objective', 'Native winner CE/log(2) plus multiplier-weighted normalized full-pool regret proxy and fixed36 CE proxy'],
             ['TRAIN hard feedback', 'Equal-task actual hard-action regret and unconditional fixed36 missed frequency; one commit after each successful logical update'],
             ['TRAIN budgets', 'Saved selected R84A TRAIN only; fixed historical36 targets, not dynamic zero groups or quotas'],
             ['VAL selection', 'Separate saved R84A VAL thresholds; no VAL dual updates. Feasible: native Top1, regret, earliest. Infeasible fallback: normalized violations first'],
             ['Inference', 'One unchanged original R45A selector; maximin; stable global-ID score ties; no current labels/costs in forward'],
             ['Evidence boundary', 'Proxy/controller heuristic, no deterministic nonconvex feasibility guarantee; TEST development-exposed; no per-task assembly']],
            ['The actual objective/controller parameters are recorded in protocol.json without modification.',
             'Selected feasibility/fallback is reported even when constraints fail; completing PDFs is not a scientific success claim.'], lock, fontsize=8)
        main.savefig(fig)
        plt.close(fig)
        for split in ('TRAIN', 'VAL', 'TEST'):
            selected = [r for r in metrics if r['split'] == split]
            fig = table_page(split + ' | ALL and18 tasks',
                ['Task', 'Top1 %', 'Cost', 'SBS', 'Oracle', 'vsSBS %', 'vsOracle %', 'Regret %'],
                [[r['problem'], *[fmt(r[k]) for k in CORE_METRICS]] for r in selected],
                [WINNER_RULE, 'SBS is fitted on full81000 TRAIN only; Oracle is the per-instance legal-pool minimum.',
                 'vsSBS=100*(mean_cost/SBS_cost-1); vsOracle=100*(mean_cost/Oracle_cost-1).',
                 'Actual regret=mean_i[100*(selected_i-oracle_i)/oracle_i]. No per-task model assembly.'], lock, fontsize=8)
            main.savefig(fig)
            plt.close(fig)
        fig = table_page('TRAIN constraints vs separate VAL thresholds',
            ['Split', 'Constraint', 'Observed %', 'Reference %', 'Satisfied', 'Role'],
            [[r['split'], r['constraint'].replace('fixed36_unconditional_', 'F36 '), fmt(r['observed_pct']),
              fmt(r['threshold_pct']), r['satisfied'], r['role'][:45]] for r in constraints],
            ['Selected VAL feasible=' + str(lock['feasible']) + '; fallback=' + str(model['selection_audit']['fallback_used']),
             'TEST threshold comparisons are descriptive, NOT a selection gate or dual-update target.',
             'Proxy-Lagrangian organization is a heuristic here, not a deterministic nonconvex feasibility guarantee.'], lock, fontsize=7)
        main.savefig(fig)
        plt.close(fig)
        fig = table_page('Natural versus fixed36 native-winner error budgets',
            ['Split', 'Group', 'Native winners', 'Missed', 'Population %', 'Error frequency %', 'Regret contribution %'],
            [[r['split'], r['group'], r['winner_count'], r['missed_count'], fmt(100. * r['population_frequency']),
              fmt(100. * r['hard_error_frequency']), fmt(100. * r['hard_regret_contribution'])]
             for r in budgets if r['problem'] == 'ALL'],
            ['F36 missed frequency is unconditional over natural instances, NOT conditional recall.',
             'FIXED36 + OUTSIDE36 reconstruct FULL18; task-equal macro includes tasks with zero fixed population.'], lock)
        main.savefig(fig)
        plt.close(fig)
        fig = table_page('One current BASE versus historical NSS | four exact datasets',
            ['Dataset', 'n', 'NSS Top1 %', 'R122 Top1 %', 'NSS mean_cost', 'R122 mean_cost', 'Status'],
            [[r['dataset'], r['n'], fmt(r['nss_top1_native_pct']), fmt(r['current_top1_native_pct']),
              fmt(r['nss_mean_cost']), fmt(r['current_mean_cost']), r['status']] for r in nss_rows],
            ['NSS: historical70114 TRAIN/old VAL selection; stored predictions on the exact new4500 TEST and original LIBs.',
             'No NSS retraining or new inference. Native labels are primary; old3895 TEST metrics are never reused.',
             'All four current cells use the same checkpoint SHA ' + lock['checkpoint_sha256']], lock)
        main.savefig(fig)
        plt.close(fig)
        lib_metrics = [r for r in metrics if r['split'] in LIBS]
        fig = table_page('Native LIB | same current checkpoint',
            ['Dataset', 'Top1 %', 'Cost', 'TRAIN SBS', 'Oracle', 'vsSBS %', 'vsOracle %', 'Regret %'],
            [[r['split'], *[fmt(r[k]) for k in CORE_METRICS]] for r in lib_metrics],
            ['Standard raw input/geometry pipeline; full selector factory and frozen TRAIN buffers; no LIB refit or selection.',
             'Legacy epoch is derived in memory from the unique update; best.pt and model parameters/buffers are never rewritten.'], lock)
        main.savefig(fig)
        plt.close(fig)
        fig = table_page('Same90k historical comparisons | not newly matched training controls',
            ['Reference / split', 'Corrected', 'Harmed', 'Net TP', 'dTop1 pp', 'dCost', 'dRegret pp'],
            [[r['comparison'] + ' ' + r['split'], r['corrected'], r['harmed'], r['net_correct'],
              fmt(r['top1_delta_pp']), fmt(r['mean_cost_delta']), fmt(r['actual_regret_delta_pp'])]
             for r in comparisons if r['problem'] == 'ALL'],
            ['R84A model/architecture matched, but budget/loss/R-Drop not matched; R94B model_matched=false (pretrained node fusion).',
             'Positive dTop1 is better; positive dRegret/dCost is worse. Selection counts are not the claim.',
             'Reference status: ' + '; '.join(r['model'] + '=' + r['status'] for r in refs)], lock, fontsize=8)
        main.savefig(fig)
        plt.close(fig)
        for problem in PROBLEMS:
            datasets = [('TEST', predictions['TEST'][problem])]
            datasets += [(name, libs[name]) for name, p in LIBS.items() if p == problem]
            fig = distribution_figure(problem, datasets, model['config']['candidate_pools'][problem], lock, data_version=version)
            main.savefig(fig)
            dist.savefig(fig)
            plt.close(fig)
            for split, data in datasets:
                dist_rows.extend(distribution_rows(problem, split, data, lock, version=version if split == 'TEST' else 'original native ' + split))
        fig = curve_figure(model, rows)
        main.savefig(fig)
        plt.close(fig)
    common.write_csv(output / 'selection_distribution.csv', dist_rows)
    if nss_data:
        with PdfPages(output / 'nss_selection_distributions.pdf') as pdf:
            nss_distributions = []
            for problem in ('TSP', 'CVRP'):
                lib = next(ds for ds, p in LIBS.items() if p == problem)
                datasets = [(split, nss_data.get((split, problem))) for split in ('TEST', lib)]
                checkpoint = next((r['nss_checkpoint_sha256'] for r in nss_rows
                    if r['problem'] == problem and r['status'] == 'complete'), 'MISSING')
                nss_lock = dict(lock, checkpoint_sha256=checkpoint)
                fig = distribution_figure(problem, datasets, config['candidate_pools'][problem], nss_lock,
                    label='NSS historical70114 (not retrained)', data_version=version)
                # NSS epochs/updates are not R122's, so never inherit the R122 footer.
                fig.texts[1].set_text('Frozen historical NSS checkpoint ' + checkpoint[:20] + ' | original stored inference')
                pdf.savefig(fig)
                plt.close(fig)
                for split, data in datasets:
                    if data is not None:
                        nss_distributions.extend(distribution_rows(problem, split, data, nss_lock,
                            label='NSS historical70114', version=version if split == 'TEST' else 'original native ' + split))
            common.write_csv(output / 'nss_selection_distribution.csv', nss_distributions)


def report(args):
    if str(args.device) != 'cpu':
        raise ValueError('Saved-only report requires --device cpu; it never performs inference')
    inputs = {}
    model = locked_inputs(args.root, inputs)
    sources = helper_sources(inputs)
    import torch
    torch.set_num_threads(2)
    run, config, lock = model['run'], model['config'], model['lock']
    with (run / 'postprocess.lock').open('a') as claim:
        fcntl.flock(claim, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            output = run / 'full_model_statistics'
            output.mkdir(exist_ok=True)
            common.write_json(run / 'postprocess_status.json', dict(experiment='R122', phase='saved_only_reporting'))
            predictions = {split: natural_split(model, split, inputs) for split in ('TRAIN', 'VAL', 'TEST')}
            libs, lib_results = load_libs(model, inputs)
            targets = load_targets(config, inputs)
            metrics, methods = [], []
            version = Path(config['data_root']).name
            for split in ('TRAIN', 'VAL', 'TEST'):
                rows = []
                for problem, data in predictions[split].items():
                    row, measured = metric_row(data, config['baselines'][problem], model=MODEL, split=split,
                        problem=problem, checkpoint=lock['checkpoint_sha256'], version=version,
                        saved=model['results'][split]['per_problem'][problem])
                    rows.append(row)
                    for slot, solver in enumerate(data['pool']):
                        top1, cost = float(np.mean(data['winner'] == slot)) * 100., float(data['costs'][:, slot].mean())
                        methods.append(dict(model=MODEL, split=split, problem=problem, candidate=solver,
                            method_native_top1_pct=top1, method_mean_cost=cost, selector_top1_pct=row['top1_pct'],
                            selector_mean_cost=row['mean_cost'], selector_better_top1=row['top1_pct'] > top1,
                            selector_better_cost=row['mean_cost'] < cost, sbs_solver=config['baselines'][problem]['sbs_solver']))
                metrics.extend([macro_rows(rows), *rows])
            for dataset, data in libs.items():
                problem = LIBS[dataset]
                metrics.append(metric_row(data, config['baselines'][problem], model=MODEL, split=dataset,
                    problem=problem, checkpoint=lock['checkpoint_sha256'], version='original native ' + dataset,
                    saved=lib_results[dataset]['train_sbs_reference'])[0])
            budgets = error_budgets(predictions, targets)
            constraints = constraint_rows(budgets, config, lock)
            val_ok = all(r['satisfied'] for r in constraints if r['split'] == 'VAL')
            if val_ok != lock['feasible']:
                raise ValueError('Recomputed VAL hard constraints disagree with the selected feasible flag')
            nss_rows, nss_data = nss_comparisons(predictions, libs, model, args.nss_root, inputs)
            comparisons, target_details, target_summary, refs, old_metrics = [], [], [], [], []
            for label, path, experiment, arm in (('R84A', args.historical_root, 'R84', 'A_baseline'),
                                                  ('R94B', args.r94_root, 'R94', 'B')):
                try:
                    old = historical_model(path, experiment, arm, config, inputs)
                except FileNotFoundError as error:
                    refs.append(dict(model=label, status='missing', reason=str(error), newly_matched_control=False,
                                     model_matched=label == 'R84A', training_protocol_matched=False))
                    continue
                if label == 'R84A' and old['lock']['checkpoint_sha256'] != BASELINE_SHA:
                    raise ValueError('Historical R84A changed from the TRAIN budget/fixed36 baseline')
                if label == 'R84A' and old['config']['model_params'] != config['model_params']:
                    raise ValueError('R84A architecture-matched status requires the exact original model parameters')
                refs.append(dict(model=label, status='complete', checkpoint_sha256=old['lock']['checkpoint_sha256'],
                    run=str(old['run']),
                    model_matched=label == 'R84A', architecture_matched=label == 'R84A',
                    training_protocol_matched=False, newly_matched_control=False, same90k=True,
                    reference_budget=57600, current_budget=MAX_UPDATES,
                    boundary='R84A same original model; objective/budget differ' if label == 'R84A'
                        else 'R94B model_matched=false; frozen pretrained node fusion / extra donor compute'))
                for split in ('VAL', 'TEST'):
                    previous = natural_split(old, split, inputs)
                    comparison = MODEL + ' vs ' + label
                    paired = [paired_row(previous[p], predictions[split][p], comparison=comparison,
                                         split=split, problem=p) for p in PROBLEMS]
                    comparisons.extend([*paired, macro_rows(paired, ('top1_delta_pp', 'mean_cost_delta', 'actual_regret_delta_pp'),
                        ('corrected', 'harmed', 'net_correct', 'changed', 'both_wrong_cheaper', 'both_wrong_dearer'))])
                    detail, summary = recovery_tables(previous, predictions[split], targets, comparison, split)
                    target_details.extend(detail)
                    target_summary.extend(summary)
                    rows = [metric_row(previous[p], old['config']['baselines'][p], model=label, split=split,
                        problem=p, checkpoint=old['lock']['checkpoint_sha256'], version=version,
                        saved=old['results'][split]['per_problem'][p])[0] for p in PROBLEMS]
                    old_metrics.extend([macro_rows(rows), *rows])
            base = next((r for r in comparisons if r['problem'] == 'ALL' and r['split'] == 'TEST'
                         and r['comparison'].endswith('R84A')), None)
            fixed = next((r for r in target_summary if r['problem'] == 'ALL' and r['split'] == 'TEST'
                          and r['solver'] == 'FIXED36' and r['comparison'].endswith('R84A')), None)
            claims = dict(selected_val_feasible=lock['feasible'], fallback_used=model['selection_audit']['fallback_used'],
                train_constraints_satisfied=all(r['satisfied'] for r in constraints if r['split'] == 'TRAIN'),
                val_constraints_satisfied=val_ok,
                failed_train_val_constraints=[r['split'] + '/' + r['constraint'] for r in constraints
                    if r['split'] in ('TRAIN', 'VAL') and not r['satisfied']],
                test_development_exposed=True, new_blind_confirmation=False, automatic_promotion=False,
                quotas=False, selection_counts_are_success=False,
                joint_observed_test_improvement_vs_R84A=None if base is None else bool(
                    lock['feasible'] and base['top1_delta_pp'] > 0. and base['actual_regret_delta_pp'] <= 0.
                    and fixed['net_correct'] >= 0),
                boundary='Descriptive full18 Top1/regret and fixed36 true misses; not a causal matched-control or statistical generalization claim',
                true_fixed36_net_tp_vs_R84A=None if fixed is None else fixed['net_correct'],
                new_false_fixed36_picks_vs_R84A=None if fixed is None else fixed['new_false_pick_count'])
            for name, rows in (('metrics_by_problem.csv', metrics), ('single_method_comparison.csv', methods),
                ('natural_fixed36_error_budgets.csv', budgets), ('constraint_status.csv', constraints),
                ('nss_comparison.csv', nss_rows), ('historical_references.csv', refs), ('historical_metrics.csv', old_metrics),
                ('paired_changes.csv', comparisons), ('target_recovery.csv', target_details),
                ('target_recovery_summary.csv', target_summary)):
                common.write_csv(output / name, rows)
            common.write_json(output / 'science_claims.json', claims)
            render_reports(output, model, predictions, libs, metrics, methods, nss_rows, nss_data,
                           constraints, budgets, comparisons, target_summary, refs, claims, args.font)
            notes = '\n'.join(('# R122 Locked Full18 Delivery', '',
                'Selected VAL feasible: ' + str(lock['feasible']) + '; fallback: ' + str(claims['fallback_used']),
                'Actual full TRAIN constraints satisfied: ' + str(claims['train_constraints_satisfied']),
                'Checkpoint SHA256: ' + lock['checkpoint_sha256'],
                'TRAIN budgets and VAL-only thresholds are separate; see constraint_status.csv.',
                'Primary claim is full18 native Top1 / actual regret, not selection quotas or zero-count reduction.',
                'R84A is architecture-matched only; R94B model_matched=false. Both are historical same90k references.',
                'NSS historical70114-trained weights are reused; no new90k retrain or inference.',
                'TEST was development-exposed. No TEST/LIB reselection or per-problem checkpoint assembly.',
                'See MODEL_STATISTICS.pdf, selection_distributions.pdf and fixed36_recovery.pdf.', ''))
            (output / 'RESULTS.md').write_text(notes, encoding='utf-8')
            for name in REQUIRED_PDFS:
                path = output / name
                with path.open('rb') as stream:
                    if path.stat().st_size < 100 or stream.read(5) != b'%PDF-':
                        raise ValueError('Missing/invalid required PDF: ' + name)
            for path, digest in inputs.items():
                if common.file_hash(Path(path)) != digest:
                    raise ValueError('Input changed before report completion: ' + path)
            outputs = {str(p.relative_to(output)): common.file_hash(p) for p in output.iterdir() if p.is_file() and p.name != 'protocol.json'}
            protocol = dict(experiment='R122', arm='model', complete=True, checkpoint_lock=lock,
                model_factory=RUNNER_FACTORY + '.load_checkpoint', lib_adapter=LIB_FACTORY + '.load_checkpoint',
                data_root=config['data_root'], rows_per_problem=config['rows_per_problem'],
                candidate_pools=config['candidate_pools'], candidate_pool_ids=config['candidate_pool_ids'],
                winner_rule=WINNER_RULE, all_aggregation='equal-task; raw macro costs descriptive',
                train_constraint_reference=config['constraint_reference']['train'],
                val_selection_only_reference=config['constraint_reference']['val_selection'],
                objective=config['loss'], controller=config['controller'],
                checkpoint_rule=config['checkpoint_rule'], inference_rule=config['inference_rule'],
                selection_audit=model['selection_audit'], scientific_claims=claims, historical_references=refs,
                nss_reference='historical70114-trained R79; saved exact4500 TEST/native LIB; no retraining',
                nss_status=nss_rows, distribution_pages=18, current_dataset_panels=20,
                source_sha256=sources,
                inputs=inputs, outputs=outputs, completed_utc=common.utc(),
                training=False, inference=False, test_rerun=False, solver_execution=False,
                checkpoint_reselection=False, per_problem_assembly=False)
            common.write_json(output / 'protocol.json', protocol)
            complete = dict(experiment='R122', arm='model', complete=True, phase='complete', exit_status=0,
                state_hash=lock['state_hash'], checkpoint_sha256=lock['checkpoint_sha256'], selected_update=lock['selected_update'],
                selected_val_feasible=lock['feasible'], scientific_claims=claims,
                report_protocol_sha256=common.file_hash(output / 'protocol.json'), outputs=outputs, completed_utc=common.utc())
            common.write_json(run / 'report_complete.json', complete)
            common.write_json(run / 'postprocess_status.json', complete)
            return complete
        except BaseException as error:
            common.write_json(run / 'postprocess_status.json', dict(experiment='R122', phase='failed', exit_status=1,
                error=str(error), traceback=traceback.format_exc(), automatic_retry=False,
                preserved='Training/TEST/checkpoint are untouched; rerun only finish/report after repair'))
            raise


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('command', choices=('describe', 'status', 'lib', 'report', 'finish'))
    result.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    result.add_argument('--arm', choices=('model',), default='model')
    result.add_argument('--device', default='cpu')
    result.add_argument('--lib-batch-size', type=int, choices=(1, 2), default=2)
    result.add_argument('--nss-root', type=Path, default=DEFAULT_NSS)
    result.add_argument('--historical-root', type=Path, default=DEFAULT_R84)
    result.add_argument('--r94-root', type=Path, default=DEFAULT_R94)
    result.add_argument('--font', type=Path, default=FONT)
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    for name in ('root', 'nss_root', 'historical_root', 'r94_root', 'font'):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    if args.command == 'describe':
        result = dict(experiment='R122', model=MODEL, commands=['lib', 'report', 'finish'],
            gate='complete14400 + full18 TEST + one unchanged VAL lock before any LIB/report',
            lib='parent-assigned CUDA; standard input/geometry; epoch-only typed adapter',
            report='CPU saved-only; native FP64, full18, TEST then LIB, four NSS datasets, historical R84A/R94B',
            outputs=list(REQUIRED_PDFS), training=False, test_rerun=False, nss_inference=False, git=False)
    elif args.command == 'status':
        path = args.root / 'model/report_complete.json'
        result = common.read_json(path) if path.is_file() else dict(experiment='R122', phase='not_finished')
    elif args.command == 'lib':
        result = lib_stage(args)
    elif args.command == 'report':
        result = report(args)
    else:
        # No waiting, automatic retries or training/TEST invocation are hidden here.
        lib_stage(args)
        result = report(argparse.Namespace(**dict(vars(args), device='cpu')))
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)
    return result


if __name__ == '__main__':
    try:
        main()
    except NotReady as error:
        print('NOT READY (no work launched): ' + str(error), file=sys.stderr)
        raise SystemExit(75)

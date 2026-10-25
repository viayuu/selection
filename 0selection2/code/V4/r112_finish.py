"""Saved-prediction R112 analysis and explicit, single-attempt Git publication.

No torch, model factories, training, evaluation, TEST tuning or runtime edits.
`wait` watches completion receipts only. `wait --publish` explicitly opts into
the same publication operation as `publish`; neither command restarts training.
"""

import argparse
import ast
from contextlib import contextmanager
import csv
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime, timezone

import numpy as np

from code.unified_selector.registry import PROBLEMS


V4 = Path(__file__).resolve().parent
ROOT = V4 / 'runs/R112_region_interfaces'
GIT_ROOT = Path('/public/home/shiys')
ARMS = ('A_collapsed', 'B_interfaces')
STAGES = ('train', 'test', 'timing', 'lib', 'report')
REFERENCES = {'R94B': V4 / 'runs/R94_pretrained_node_fusion/B',
              'R89B': V4 / 'runs/R89_optimizer_geometry/B_muon_hybrid'}
METRICS = ('top1_pct', 'mean_cost', 'sbs_cost', 'oracle_cost',
           'vs_sbs_pct', 'vs_oracle_pct', 'actual_regret_pct')
COUNTS = ('corrected', 'harmed', 'net_correct', 'changed', 'cost_wins', 'cost_losses',
          'cost_ties', 'both_wrong_cheaper', 'both_wrong_dearer')
DELTAS = ('top1_delta_pp', 'mean_cost_delta', 'actual_regret_delta_pp', 'W_pct', 'G_pct',
          'gross_gain_cost', 'gross_harm_cost', 'both_wrong_cost_delta', 'both_wrong_regret_delta_pp')
EVENT_FIELDS = ('comparison', 'split', 'problem', 'index', 'source_index', 'event',
                'baseline_solver', 'current_solver', 'winner_solver', 'baseline_cost',
                'current_cost', 'cost_delta', 'actual_regret_delta_pp')
CODE_NAMES = ('r112_model.py', 'r112_experiment.py', 'r112_reporting.py', 'r112_launch.py',
              'r112_witness.py', 'run_v4_r112.sh', 'r112_finish.py', 'test_r112_model.py',
              'test_r112_experiment.py', 'test_r112_launch.py', 'test_r112_finish.py')
OUTPUTS = ('RESULTS.md', 'RESULTS.pdf', 'comparison.csv', 'metrics.csv', 'events.csv',
           'historical_references.csv', 'science_boundary.json')
SMALL_LIMIT = 8 * 1024 * 1024
TOTAL_LIMIT = 64 * 1024 * 1024
RESEARCH_FILES = {
    'comprehensive_direction_review_20261025': (
        'REQUEST.md', 'REPORT.md', 'REPORT.pdf', 'RESEARCH_FULL.pdf', 'SYNTHESIS_DIALOGUE.md',
        '01_decision_and_identifiability.md', '01_decision_and_identifiability.pdf', '01_sources.csv',
        '02_representation_and_computation.md', '02_representation_and_computation.pdf', '02_sources.csv',
        '03_learning_and_optimization.md', '03_learning_and_optimization.pdf', '03_sources.csv',
        'research_directions.csv', 'source_registry.csv', 'render_notes.json', 'render_report.py',
        'collect_evidence.py', 'evidence_manifest.json', 'artifact_index.csv', 'reported_all_metrics.csv',
        'running_snapshot.csv', 'DELIVERY.md'),
    'major_paradigm_review_20261025': (
        'REQUEST.md', 'REPORT.md', 'REPORT.pdf', 'RESEARCH_FULL.pdf', 'SYNTHESIS_DIALOGUE.md',
        '01_computational_architectures.md', '01_computational_architectures.pdf', '01_sources.csv',
        '02_representation_learning.md', '02_representation_learning.pdf', '02_sources.csv',
        '03_prediction_objects.md', '03_prediction_objects.pdf', '03_sources.csv',
        '04_data_and_learning_paradigms.md', '04_data_and_learning_paradigms.pdf', '04_sources.csv',
        'research_directions.csv', 'source_registry.csv', 'render_notes.json', 'render_report.py', 'DELIVERY.md')}
REPORT_LAYOUT = 'deep_direction_review_v2_20261023/render_report.py'


class NotReady(RuntimeError):
    pass


def utc():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def write(path, value):
    path = Path(path)
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    try:
        temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def write_csv(path, rows, fields=None):
    path = Path(path)
    fields = fields or list(dict.fromkeys(k for row in rows for k in row))
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    try:
        with temporary.open('w', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def launch_receipt(root):
    return Path(root) / 'preparation/dispatch/launch.json'


def completed(root):
    """This is the only wait condition; no polling of training/model internals."""
    path = launch_receipt(root)
    if not path.is_file():
        raise NotReady('Awaiting ' + str(path))
    row = read(path)
    if row.get('state') == 'failed':
        raise RuntimeError('R112 launch failed; no restart: ' + str(row.get('error')))
    if row.get('state') != 'complete':
        raise NotReady('R112 launch state=' + str(row.get('state')))
    if row.get('exit_status') != 0 or any(s.get('exit_status') != 0 or s.get('state') != 'complete'
                                       for s in row.get('stages', [])):
        raise ValueError('Completed launch must have rc0 stages')
    inputs = {str(path.resolve()): digest(path)}
    for arm in ARMS:
        path = Path(root) / 'preparation/dispatch' / (arm + '.json')
        item = read(path)
        if (item.get('state') != 'complete' or item.get('exit_status') != 0
                or [s.get('name') for s in item.get('stages', [])] != list(STAGES)
                or any(s.get('state') != 'complete' or s.get('exit_status') != 0 for s in item['stages'])):
            raise ValueError('Both independent R112 pipelines must finish all five stages at rc0')
        inputs[str(path.resolve())] = digest(path)
    return inputs


def remember(inputs, path):
    path = Path(path).resolve()
    inputs[str(path)] = digest(path)
    return path


def inside(value, run):
    path = Path(value)
    if not path.is_absolute():
        path = Path(run) / path
    path = path.resolve()
    if not path.is_relative_to(Path(run).resolve()):
        raise ValueError('Prediction path escapes its own run: ' + str(path))
    return path


def load_run(run, inputs, *, current=False, expected=None):
    """Receipt/hash validation only; a .pt file is hashed, never deserialized."""
    run = Path(run).resolve()
    values = {}
    for name in ('protocol.json', 'final_test_lock.json', 'training_complete.json',
                 'test_results.json', 'final_metrics.json', 'full_model_statistics/protocol.json'):
        values[name] = read(remember(inputs, run / name))
    config, lock = values['protocol.json'], values['final_test_lock.json']
    training, test, final = (values[n] for n in ('training_complete.json', 'test_results.json', 'final_metrics.json'))
    if (expected is not None and (config.get('experiment'), config.get('arm')) != expected
            or lock.get('experiment') != config.get('experiment') or lock.get('arm') != config.get('arm')
            or lock.get('state_hash') != json_hash(config) or not lock.get('selected_without_test')
            or lock.get('per_problem_assembly') is not False
            or lock.get('rule') != config.get('checkpoint_rule')
            or lock.get('inference_rule') != config.get('inference_rule')
            or not 0 < lock.get('selected_update', 0) <= config['max_successful_updates']
            or training.get('successful_updates') != config['max_successful_updates']
            or test.get('complete') is not True or final.get('complete') is not True
            or set(test['result']['per_problem']) != set(PROBLEMS)
            or set(final['val']['per_problem']) != set(PROBLEMS)
            or config.get('rows_per_problem') != dict(train=81000, val=4500, test=4500)):
        raise ValueError('Incomplete or inconsistent full18 frozen model: ' + str(run))
    if any(test['result']['per_problem'][p]['n'] != 4500 or final['val']['per_problem'][p]['n'] != 4500
           or config['baselines'][p].get('selection_split') != 'train' for p in PROBLEMS):
        raise ValueError('Require exact NEW4500 and full-TRAIN SBS, never old heldout metrics')
    for item in (test, final):
        if any(item.get(k) != lock[k] for k in ('state_hash', 'selected_update', 'checkpoint_sha256')):
            raise ValueError('Saved TEST/final results differ from the unique VAL checkpoint')
    if digest(remember(inputs, run / 'best.pt')) != lock['checkpoint_sha256']:
        raise ValueError('Selected checkpoint file changed')
    if inside(lock['checkpoint'], run) != run / 'best.pt':
        raise ValueError('Checkpoint path differs from the unique best.pt')
    report = values['full_model_statistics/protocol.json']
    if report.get('complete') is not True or report.get('checkpoint_lock') != lock:
        raise ValueError('Full model statistics must use the same unique lock')
    for name in ('MODEL_STATISTICS.pdf', 'selection_distributions.pdf', 'learning_curves.pdf',
                 'training_and_method_diagnostics.pdf'):
        path = remember(inputs, run / 'full_model_statistics' / name)
        with path.open('rb') as stream:
            signature = stream.read(5)
        if path.stat().st_size < 16 or signature != b'%PDF-':
            raise ValueError('Missing/invalid required model PDF: ' + str(path))
    if current:
        receipt = read(remember(inputs, run / 'report_complete.json'))
        if (receipt.get('complete') is not True or receipt.get('experiment') != 'R112'
                or receipt.get('arm') != config['arm']
                or any(receipt.get(k) != lock[k] for k in ('state_hash', 'selected_update', 'checkpoint_sha256'))):
            raise ValueError('Both R112 report_complete receipts must bind to final TEST locks')
    return dict(run=run, config=config, lock=lock, final=final, test=test['result'])


def same_data(left, right):
    for field in ('data_root', 'preparation_sha256', 'rows_per_problem', 'candidate_pools',
                  'candidate_pool_ids', 'baselines', 'winner_rule'):
        if field not in left or left[field] != right.get(field):
            raise ValueError('Whole-model comparison requires identical ' + field)


def load_predictions(archive, prefix, config, problem, expected_rows):
    def get(name, optional=False):
        key = prefix + name
        return np.asarray(archive[key]) if key in archive else None if optional else archive[key]

    costs, winner = get('costs'), get('winner')
    pool, ids, indices = get('pool').astype(str).tolist(), get('pool_ids'), get('indices')
    logits = get('logits', True)
    if logits is None:
        logits = get('scores')
    n = len(winner)
    if (costs.dtype != np.float64 or winner.dtype != np.int64 or costs.shape != (n, len(pool))
            or logits.shape != costs.shape or winner.shape != (n,) or n != expected_rows
            or pool != config['candidate_pools'][problem] or ids.tolist() != config['candidate_pool_ids'][problem]
            or ids.tolist() != sorted(set(ids.tolist())) or len(np.unique(indices)) != n
            or not np.isfinite(costs).all() or (costs <= 0).any() or not np.isfinite(logits).all()
            or not np.array_equal(winner, costs.argmin(1))):
        raise ValueError('Invalid native FP64 prediction archive: ' + problem)
    pred, saved = logits.argmax(1), get('pred', True)
    if saved is not None and not np.array_equal(pred, saved):
        raise ValueError('Saved predictions differ from locked stable-global-ID argmax')
    result = dict(costs=costs, winner=winner, pool=pool, pool_ids=ids, indices=indices, pred=pred)
    for name in ('source_indices', 'release_source_indices'):
        value = get(name, True)
        if value is not None:
            result[name] = value
    return result


def saved_split(model, split, inputs):
    run, lock = model['run'], model['lock']
    result = model['test'] if split == 'TEST' else model['final']['val']
    path = remember(inputs, inside(result['prediction_archive'], run))
    capture = read(remember(inputs, path.parent / 'identity.json'))
    official = result.get('archive_capture_identity', result['identity'])
    expected = dict(experiment=model['config']['experiment'], arm=model['config']['arm'],
        state_hash=lock['state_hash'], split=split.lower(), successful_updates=lock['selected_update'],
        model_state_sha256=model['test']['identity']['model_state_sha256'])
    if (capture != official or any(capture.get(k) != v for k, v in expected.items())
            or capture.get('checkpoint_sha256') not in (None, lock['checkpoint_sha256'])
            or split == 'TEST' and capture.get('checkpoint_sha256') != lock['checkpoint_sha256']):
        raise ValueError('Prediction archive is not from the unique selected checkpoint')
    with np.load(path, allow_pickle=False) as archive:
        data = {p: load_predictions(archive, p + '__', model['config'], p, 4500) for p in PROBLEMS}
    for item in data.values():
        np.testing.assert_array_equal(item['indices'], np.arange(4500))
    return data


def saved_lib(model, dataset, inputs):
    problem = 'TSP' if dataset == 'TSPLIB' else 'CVRP'
    run, lock = model['run'], model['lock']
    result = read(remember(inputs, run / 'lib_evaluation' / (dataset + '_results.json')))
    identity = result['protocol']
    required = dict(experiment='R112', arm=model['config']['arm'], dataset=dataset,
        checkpoint_sha256=lock['checkpoint_sha256'], state_hash=lock['state_hash'],
        selected_update=lock['selected_update'], model_state_sha256=model['test']['identity']['model_state_sha256'])
    if any(identity.get(k) != v for k, v in required.items()):
        raise ValueError('LIB differs from the same full-model checkpoint')
    path = remember(inputs, run / 'lib_evaluation/predictions' / (dataset + '.npz'))
    with np.load(path, allow_pickle=False) as archive:
        if (str(archive['checkpoint_sha256'].item()) != lock['checkpoint_sha256']
                or int(archive['checkpoint_update']) != lock['selected_update']):
            raise ValueError('LIB prediction checkpoint binding differs')
        return problem, load_predictions(archive, '', model['config'], problem,
                                        result['train_sbs_reference']['n'])


def metric_row(data, baseline, *, model, split, problem, checkpoint):
    costs, pred, winner = data['costs'], data['pred'], data['winner']
    chosen, oracle = costs[np.arange(len(pred)), pred], costs.min(1)
    if baseline.get('selection_split') != 'train':
        raise ValueError('SBS must be fitted on frozen TRAIN')
    sbs = costs[:, data['pool'].index(baseline['sbs_solver'])].mean()
    return dict(model=model, split=split, problem=problem, n=len(pred),
        native_correct=int((pred == winner).sum()), top1_pct=100 * float((pred == winner).mean()),
        mean_cost=float(chosen.mean()), sbs_cost=float(sbs), oracle_cost=float(oracle.mean()),
        vs_sbs_pct=100 * float(chosen.mean() / sbs - 1),
        vs_oracle_pct=100 * float(chosen.mean() / oracle.mean() - 1),
        actual_regret_pct=100 * float(((chosen - oracle) / oracle).mean()),
        checkpoint_sha256=checkpoint, winner_rule='native/FP64 ind; never FP32 argmin',
        candidate_pool=json.dumps(data['pool']), sbs_selection_split='frozen full TRAIN',
        aggregation='single problem; frozen full-TRAIN SBS')


def macro(rows, fields, counts=()):
    if {r['problem'] for r in rows} != set(PROBLEMS):
        raise ValueError('ALL requires all18, never an available-problem subset')
    result = dict(rows[0], problem='ALL', n=sum(r['n'] for r in rows),
                  aggregation='equal-problem mean; raw cost descriptive only')
    result.update({k: sum(r[k] for r in rows) for k in counts})
    result.update({k: float(np.mean([r[k] for r in rows])) for k in fields})
    if 'candidate_pool' in result:
        result['candidate_pool'] = 'per-problem legal pools; see individual rows'
    return result


def paired(old, new, comparison, split, problem):
    for name in ('pool', 'pool_ids', 'indices', 'source_indices', 'release_source_indices', 'costs', 'winner'):
        if (name in old) != (name in new):
            raise ValueError('Prediction alignment field missing: ' + name)
        if name in old:
            np.testing.assert_array_equal(old[name], new[name], err_msg='unaligned ' + name)
    costs, left, right, winner = new['costs'], old['pred'], new['pred'], new['winner']
    index, n = np.arange(len(winner)), len(winner)
    before, after, oracle = costs[index, left], costs[index, right], costs.min(1)
    delta = after - before
    a, b = left == winner, right == winner
    masks = dict(corrected=~a & b, harmed=a & ~b,
        both_wrong_cheaper=~a & ~b & (delta < 0), both_wrong_dearer=~a & ~b & (delta > 0),
        both_wrong_equal_changed=~a & ~b & (delta == 0) & (left != right))
    row = dict(comparison=comparison, split=split, problem=problem, n=n,
        corrected=int(masks['corrected'].sum()), harmed=int(masks['harmed'].sum()),
        net_correct=int(b.sum() - a.sum()), changed=int((left != right).sum()),
        cost_wins=int((delta < 0).sum()), cost_losses=int((delta > 0).sum()), cost_ties=int((delta == 0).sum()),
        both_wrong_cheaper=int(masks['both_wrong_cheaper'].sum()),
        both_wrong_dearer=int(masks['both_wrong_dearer'].sum()),
        top1_delta_pp=100 * float(b.mean() - a.mean()), mean_cost_delta=float(delta.mean()),
        actual_regret_delta_pp=100 * float((delta / oracle).mean()),
        W_pct=100 * float((delta < 0).mean()), G_pct=100 * float((-delta / before).mean()),
        gross_gain_cost=float(np.maximum(-delta, 0).mean()),
        gross_harm_cost=float(np.maximum(delta, 0).mean()),
        both_wrong_cost_delta=float(np.where(~a & ~b, delta, 0).mean()),
        both_wrong_regret_delta_pp=100 * float(np.where(~a & ~b, delta / oracle, 0).mean()),
        aggregation='single problem; delta=current-baseline, positive G is improvement')
    events = []
    for event, mask in masks.items():
        for i in np.flatnonzero(mask):
            events.append(dict(comparison=comparison, split=split, problem=problem,
                index=int(new['indices'][i]), source_index=int(new.get('release_source_indices',
                    new.get('source_indices', new['indices']))[i]), event=event,
                baseline_solver=new['pool'][left[i]], current_solver=new['pool'][right[i]],
                winner_solver=new['pool'][winner[i]], baseline_cost=float(before[i]), current_cost=float(after[i]),
                cost_delta=float(delta[i]), actual_regret_delta_pp=100 * float(delta[i] / oracle[i])))
    if len(events) != row['changed']:
        raise ValueError('Paired event partition failed')
    return row, events


def science_boundary(models, inputs):
    left, right = (models[a]['config'] for a in ARMS)
    same_data(left, right)
    common = ('seed', 'global_batch_size', 'logical_global_batch', 'task_order_sha256',
              'max_successful_updates', 'lr', 'optimizer', 'weight_decay', 'loss', 'dropout',
              'microbatch_caps', 'eval_batch_size', 'checkpoint_rule', 'inference_rule',
              'model_spec', 'main_protocol_lock', 'source_sha256')
    if any(k not in left or left[k] != right.get(k) for k in common):
        raise ValueError('R112 A/B training or inference protocols differ beyond the intended arm')
    receipts = {a: read(remember(inputs, models[a]['run'] / 'initialization.json')) for a in ARMS}
    keys = ('trainable_sha256', 'parameter_layout_sha256', 'trainable_parameter_count', 'fresh_reference_sha256')
    if any(k not in receipts[ARMS[0]] or receipts[ARMS[0]][k] != receipts[ARMS[1]].get(k) for k in keys):
        raise ValueError('R112 arms must have identical complete fresh parameter tensors/shapes')
    for arm in ARMS:
        if (receipts[arm]['trainable_parameter_count'] != models[arm]['config']['trainable_parameter_count']
                or receipts[arm]['trainable_sha256'] != models[arm]['config']['initial_trainable_sha256']):
            raise ValueError('Initial receipt differs from frozen model protocol')
    return dict(initialization={a: {k: receipts[a][k] for k in keys} for a in ARMS},
        models={a: dict(selected_update=models[a]['lock']['selected_update'],
            checkpoint_sha256=models[a]['lock']['checkpoint_sha256'],
            data_version=Path(models[a]['config']['data_root']).name,
            trainable_parameter_count=models[a]['config']['trainable_parameter_count']) for a in ARMS},
        common_training_fields=list(common), same_logical_batch=left['logical_global_batch'],
        common_original_reference_hash_is_not_retained_subset_hash=True,
        order_boundary='same seeded task/shuffle protocol; actual per-update sample IDs not independently archived',
        statistical_boundary='single seed2; no multi-seed significance or automatic promotion',
        historical_boundary='whole saved historical models; not fresh parameter/budget matched controls',
        test_boundary='TEST participated in earlier R&D; not a pristine blind test; no finish-stage reselection',
        metric_boundary='ALL equal-problem; ratios of means differ from actual per-instance regret',
        W_definition='100*mean(current_cost < baseline_cost), cost win rate',
        G_definition='100*mean((baseline_cost-current_cost)/baseline_cost), mean relative cost improvement',
        both_wrong_definition='contribution divided by all rows in that problem, not only both-wrong rows')


def render(root, metrics, comparisons, references, boundary):
    """Small self-contained PDF; full-precision per-problem numbers stay in CSV."""
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    def table(headers, rows):
        return '\n'.join(['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |']
                         + ['| ' + ' | '.join(str(v) for v in r) + ' |' for r in rows])

    overall = [r for r in metrics if r['problem'] == 'ALL']
    pair_all = [r for r in comparisons if r['problem'] == 'ALL']
    metric_cells = lambda r: [r['model'], r['split'], *[f'{r[k]:.6f}' for k in METRICS]]
    pair_cells = lambda r: [r['comparison'], r['split'], r['corrected'], r['harmed'],
        r['both_wrong_cheaper'], r['both_wrong_dearer'], f"{r['top1_delta_pp']:.6f}",
        f"{r['actual_regret_delta_pp']:.6f}", f"{r['W_pct']:.4f}", f"{r['G_pct']:.6f}"]
    mh = ['Model', 'Split', 'Top1 %', 'mean cost', 'SBS', 'Oracle', 'vs SBS %', 'vs Oracle %', 'Regret %']
    ph = ['Comparison', 'Split', 'Corrected', 'Harmed', 'Wrong cheaper', 'Wrong dearer', 'dTop1 pp', 'dRegret pp', 'W %', 'G %']
    lines = ['# R112 Complete Models', '', '## What Completed?',
        'Both independent full18 pipelines completed train, locked TEST, timing, LIB and reports at rc0.',
        'One whole-model VAL-selected checkpoint per arm; saved predictions only, no model execution or TEST tuning.',
        table(['Arm', 'Selected update', 'Checkpoint SHA256'], [[a, boundary['models'][a]['selected_update'],
            boundary['models'][a]['checkpoint_sha256']] for a in ARMS]),
        'Data version: ' + boundary['models'][ARMS[0]]['data_version'] + '.',
        '', '## What Are The Results?', table(mh, [metric_cells(r) for r in overall]),
        '', 'ALL gives every problem equal weight. Cross-problem raw costs are descriptive only.',
        'SBS is frozen full-TRAIN fitted; Oracle is the per-instance legal-pool minimum. Native/FP64 ind, not FP32 argmin.',
        '', '## What Changed?', table(ph, [pair_cells(r) for r in pair_all]),
        '', boundary['W_definition'] + '.', boundary['G_definition'] + '.',
        'Cost/regret deltas are current minus baseline; G is positive for improvement.',
        'Both-wrong cost/regret contributions use all problem rows as denominator; gross gain/loss are separate CSV columns.',
        '', '## Historical References', table(['Model', 'Status', 'Reason'],
            [[r['model'], r['status'], r.get('reason', '')] for r in references]),
        '', '## Scientific Boundary',
        'Complete A/B fresh parameter hashes and shapes match; logical batch=' + str(boundary['same_logical_batch']) + '.',
        'The original full-reference hash is provenance, not the hash of the retained original-parameter subset.',
        boundary['order_boundary'] + '.', boundary['statistical_boundary'] + '.',
        boundary['historical_boundary'] + '.', boundary['test_boundary'] + '.',
        '', 'Per-problem and native LIB metrics: metrics.csv. Pair decompositions: comparison.csv.',
        'Instance-level events: events.csv. Input/output hashes: finish_complete.json.',
        'Per-arm method-distribution, NSS and learning-curve PDFs remain in each full_model_statistics directory.', '']
    (root / 'RESULTS.md').write_text('\n'.join(lines), encoding='utf-8')
    temporary = root / '.RESULTS.finish.pdf'
    try:
        with PdfPages(temporary) as pdf:
            def page(title, headers, rows, note):
                fig, ax = plt.subplots(figsize=(16.5, 8.3))
                ax.axis('off')
                fig.suptitle(title, fontsize=15)
                fig.text(.04, .89, note, fontsize=8)
                tab = ax.table(cellText=rows, colLabels=headers, loc='center', cellLoc='center')
                tab.auto_set_font_size(False)
                tab.set_fontsize(7)
                tab.scale(1, 1.4)
                fig.text(.04, .035, 'R112 | native/FP64 winners | TRAIN SBS | equal-task ALL | same locked weights; no TEST/LIB reselection', fontsize=8)
                pdf.savefig(fig)
                plt.close(fig)

            page('R112: Full18 Locked Results', mh, [metric_cells(r) for r in overall],
                 'Single-seed descriptive evidence. Historical models are not newly matched controls.')
            page('R112: Corrections, Harms And Cost Gains', ph, [pair_cells(r) for r in pair_all],
                 'W=cost win rate; G=mean relative cost improvement. dRegret=current-baseline; lower is better.')
            for arm in ARMS:
                for split in ('VAL', 'TEST'):
                    rows = [r for r in metrics if r['model'] == 'R112/' + arm and r['split'] == split]
                    page(arm + ': ' + split + ' Per-Problem Metrics', ['Problem', *mh[2:]],
                         [[r['problem'], *metric_cells(r)[2:]] for r in rows], boundary['metric_boundary'])
            for comparison in dict.fromkeys(r['comparison'] for r in comparisons):
                rows = [r for r in comparisons if r['comparison'] == comparison and r['split'] == 'TEST']
                page(comparison + ': TEST Paired Changes', ['Problem', *ph[2:]],
                     [[r['problem'], *pair_cells(r)[2:]] for r in rows],
                     'Both-wrong contributions are normalized by all problem rows; see comparison.csv for full precision.')
            libs = [r for r in metrics if r['split'] in ('TSPLIB', 'CVRPLIB')]
            page('R112: Same-Checkpoint Native LIB', mh, [metric_cells(r) for r in libs], boundary['test_boundary'])
            page('R112: Provenance And Scientific Boundary', ['Item', 'Evidence'],
                 [['Logical batch', boundary['same_logical_batch']],
                  ['Initial trainable SHA', boundary['initialization'][ARMS[0]]['trainable_sha256']],
                  *[[a + ' checkpoint', boundary['models'][a]['checkpoint_sha256']] for a in ARMS],
                  ['Data version', boundary['models'][ARMS[0]]['data_version']],
                  *[[r['model'], r['status'] + ': ' + r.get('reason', '')[:120]] for r in references],
                  ['Seed', '2 only; no multi-seed significance claim'],
                  ['Mechanism', 'identical fresh tensors/shapes; collapse versus retained interface conditions'],
                  ['Source of metrics', 'immutable saved predictions; no new forward passes or calibration']],
                 boundary['order_boundary'])
        temporary.replace(root / 'RESULTS.pdf')
    finally:
        temporary.unlink(missing_ok=True)


def analyze(root, references=None):
    root = Path(root).resolve()
    inputs = completed(root)
    with (root / 'finish.lock').open('a') as claim:
        fcntl.flock(claim, fcntl.LOCK_EX | fcntl.LOCK_NB)
        models = {arm: load_run(root / arm, inputs, current=True, expected=('R112', arm)) for arm in ARMS}
        main = read(remember(inputs, root / 'protocol_lock.json'))
        for model in models.values():
            frozen = model['config']['main_protocol_lock']
            if frozen['record'] != main or frozen['sha256'] != digest(root / 'protocol_lock.json'):
                raise ValueError('Current main protocol lock changed after training')
        boundary = science_boundary(models, inputs)
        metrics, comparisons, events, historical = [], [], [], []

        def add_metrics(model, data, split, label):
            rows = [metric_row(data[p], model['config']['baselines'][p], model=label,
                split=split, problem=p, checkpoint=model['lock']['checkpoint_sha256']) for p in PROBLEMS]
            for row in rows:
                row['data_version'] = Path(model['config']['data_root']).name
                saved = model['test']['per_problem'][row['problem']] if split == 'TEST' else model['final']['val']['per_problem'][row['problem']]
                for key in METRICS:
                    expected = saved['top1'] * 100 if key == 'top1_pct' else saved[key]
                    if not math.isclose(row[key], expected, rel_tol=1e-10, abs_tol=1e-9):
                        raise ValueError('Saved metric differs from locked predictions: ' + key)
            metrics.extend([*rows, macro(rows, METRICS, ('native_correct',))])
            return rows

        def compare(old, new, label, split):
            rows = []
            for p in PROBLEMS:
                row, details = paired(old[p], new[p], label, split, p)
                rows.append(row)
                events.extend(details)
            comparisons.extend([*rows, macro(rows, DELTAS, COUNTS)])

        test_data = None
        for split in ('VAL', 'TEST'):
            data = {a: saved_split(models[a], split, inputs) for a in ARMS}
            for arm in ARMS:
                add_metrics(models[arm], data[arm], split, 'R112/' + arm)
            compare(data[ARMS[0]], data[ARMS[1]], 'B_interfaces vs A_collapsed', split)
            if split == 'TEST':
                test_data = data
        for arm in ARMS:
            for dataset in ('TSPLIB', 'CVRPLIB'):
                problem, data = saved_lib(models[arm], dataset, inputs)
                metrics.append(metric_row(data, models[arm]['config']['baselines'][problem], model='R112/' + arm,
                    split=dataset, problem=problem, checkpoint=models[arm]['lock']['checkpoint_sha256']))
                metrics[-1]['data_version'] = 'original native ' + dataset
        for name, path in (REFERENCES if references is None else references).items():
            reference_inputs = {}
            try:
                old = load_run(path, reference_inputs, expected=('R94', 'B') if name == 'R94B' else ('R89', 'B_muon_hybrid'))
                same_data(models[ARMS[0]]['config'], old['config'])
                data = saved_split(old, 'TEST', reference_inputs)
                # Validate every alignment before accepting any of this reference.
                for arm in ARMS:
                    for p in PROBLEMS:
                        paired(data[p], test_data[arm][p], name, 'TEST', p)
            except (OSError, ValueError, KeyError, AssertionError) as error:
                historical.append(dict(model=name, status='missing_or_incompatible', run=str(path),
                    reason=str(error), whole_model_only=True, stitched=False))
                inputs.update({p: h for p, h in reference_inputs.items() if Path(p).is_file()})
            else:
                inputs.update(reference_inputs)
                add_metrics(old, data, 'TEST', name)
                for arm in ARMS:
                    compare(data, test_data[arm], arm + ' vs ' + name, 'TEST')
                historical.append(dict(model=name, status='complete', run=str(path),
                    checkpoint_sha256=old['lock']['checkpoint_sha256'], reason='', whole_model_only=True, stitched=False))
        write_csv(root / 'metrics.csv', metrics)
        write_csv(root / 'comparison.csv', comparisons)
        write_csv(root / 'events.csv', events, EVENT_FIELDS)
        write_csv(root / 'historical_references.csv', historical, list(dict.fromkeys(
            k for row in historical for k in row)) or ['model', 'status', 'reason'])
        write(root / 'science_boundary.json', boundary)
        render(root, metrics, comparisons, historical, boundary)
        for path, expected in inputs.items():
            if digest(path) != expected:
                raise ValueError('Analysis input changed during collection: ' + path)
        value = dict(experiment='R112', complete=True, analysis_rc=0, completed_utc=utc(),
            model_reselection=False, model_inference=False, inputs=inputs,
            outputs={name: digest(root / name) for name in OUTPUTS},
            checkpoints={a: models[a]['lock'] for a in ARMS},
            generator_sha256=digest(Path(__file__)), historical=historical,
            event_count=len(events), metric_rows=len(metrics), comparison_rows=len(comparisons))
        write(root / 'finish_complete.json', value)
        return value


def verify_delivery(root):
    root = Path(root).resolve()
    completed(root)
    value = read(root / 'finish_complete.json')
    if (value.get('complete') is not True or value.get('analysis_rc') != 0
            or value.get('experiment') != 'R112' or set(value.get('outputs', {})) != set(OUTPUTS)
            or value.get('generator_sha256') != digest(Path(__file__))):
        raise ValueError('Run the current analyze command successfully before publication')
    for path, expected in value['inputs'].items():
        if digest(path) != expected:
            raise ValueError('Frozen analysis input changed: ' + path)
    for name, expected in value['outputs'].items():
        path = root / name
        if not path.is_file() or digest(path) != expected:
            raise ValueError('Required analysis delivery missing/changed: ' + name)
    pdf = (root / 'RESULTS.pdf').read_bytes()
    if not pdf.startswith(b'%PDF-') or b'%%EOF' not in pdf[-1024:]:
        raise ValueError('RESULTS.pdf is incomplete')
    return value


def publication_paths(root):
    """Exact file names only; no globs, directory adds, arrays, weights or logs."""
    root = Path(root).resolve()
    expected = GIT_ROOT / '0selection2/code/V4/runs/R112_region_interfaces'
    if root != expected.resolve():
        raise ValueError('Publication owns only the named R112 root inside /public/home/shiys')
    v4 = root.parent.parent
    paths = [v4 / name for name in CODE_NAMES]
    research = GIT_ROOT / '0selection2/docs/research'
    research_paths = [research / group / name for group, names in RESEARCH_FILES.items() for name in names]
    # The two renderers dynamically import this small layout implementation.
    research_paths.append(research / REPORT_LAYOUT)
    paths += research_paths
    paths += [root / name for name in (*OUTPUTS, 'finish_complete.json', 'protocol_lock.json', 'PLAN.md')]
    paths += [root / 'preparation/dispatch' / name for name in
              ('launch.json', 'worker.json', 'A_collapsed.json', 'B_interfaces.json')]
    for arm in ARMS:
        paths += [root / arm / name for name in ('protocol.json', 'initialization.json', 'final_test_lock.json',
            'training_complete.json', 'report_complete.json', 'lib_stage_complete.json', 'inference_timing.json')]
        paths += [root / arm / 'full_model_statistics' / name for name in
            ('MODEL_STATISTICS.pdf', 'selection_distributions.pdf', 'learning_curves.pdf',
             'training_and_method_diagnostics.pdf', 'RESULTS.md', 'protocol.json', 'metrics_by_problem.csv',
             'nss_comparison.csv', 'selection_distribution.csv', 'inference_timing.csv')]
    included, excluded = [], []
    required = ({v4 / name for name in CODE_NAMES} | set(research_paths)
                | {root / name for name in (*OUTPUTS, 'finish_complete.json', 'protocol_lock.json')})
    for path in paths:
        if not path.exists():
            if path in required:
                raise FileNotFoundError('Required R112 publication file: ' + str(path))
            continue
        if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(GIT_ROOT.resolve()):
            raise ValueError('Publication file must be an ordinary in-repository file: ' + str(path))
        if path.stat().st_size > SMALL_LIMIT:
            if path == root / 'events.csv':
                excluded.append(dict(path=str(path.relative_to(GIT_ROOT)), reason='large event CSV retained locally',
                                     sha256=digest(path), bytes=path.stat().st_size))
                continue
            raise ValueError('R112 small-file limit exceeded: ' + str(path))
        included.append(path)
    if sum(p.stat().st_size for p in included) > TOTAL_LIMIT:
        raise ValueError('R112 publication exceeds the bounded total size; no staging')
    return included, excluded


def dependency_files(roots):
    """Local Python import closure, including literal lazy factory names."""
    repository = GIT_ROOT / '0selection2'

    def locate(name):
        path = repository.joinpath(*name.split('.'))
        return next((p for p in (path.with_suffix('.py'), path / '__init__.py') if p.is_file()), None)

    pending, seen = list(roots), set()
    while pending:
        path = Path(pending.pop()).resolve()
        if path in seen:
            continue
        seen.add(path)
        module = '.'.join(path.relative_to(repository).with_suffix('').parts)
        package = module.rsplit('.', 1)[0]
        if path.name == '__init__.py':
            package = package.rsplit('.', 1)[0] if package.endswith('.__init__') else package
        dependencies = set()
        for node in ast.walk(ast.parse(path.read_text(encoding='utf-8'), filename=str(path))):
            if isinstance(node, ast.Import):
                dependencies.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                base = importlib.util.resolve_name('.' * node.level + (node.module or ''), package) if node.level else node.module
                if base:
                    dependencies.add(base)
                    dependencies.update(base + '.' + alias.name for alias in node.names)
            elif isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value.startswith('code.'):
                if all(part.isidentifier() for part in node.value.split('.')):
                    dependencies.add(node.value)
        for name in dependencies:
            if any(part.startswith(('r109', 'r111', '_r109', '_r111', 'test_r109', 'test_r111')) for part in name.split('.')):
                raise ValueError('Unpublished R109/R111 hard dependency: ' + name)
            if name.startswith('code.'):
                while locate(name) is None and '.' in name:
                    name = name.rsplit('.', 1)[0]
                target = locate(name)
                if target is not None and target.resolve() not in seen:
                    pending.append(target)
    return sorted(seen)


def git(*args, timeout=120):
    result = subprocess.run(['git', '-C', str(GIT_ROOT), *args], capture_output=True, timeout=timeout)
    if result.returncode:
        raise RuntimeError('git ' + args[0] + ' failed: ' + result.stderr.decode(errors='replace')[-1500:])
    return result.stdout.decode(errors='strict').strip()


def publication_preflight(root):
    verify_delivery(root)
    if Path(git('rev-parse', '--show-toplevel')).resolve() != GIT_ROOT.resolve():
        raise ValueError('Expected Git root /public/home/shiys')
    paths, excluded = publication_paths(root)
    names = [str(path.relative_to(GIT_ROOT)) for path in paths]
    staged = set(filter(None, git('diff', '--cached', '--name-only', '-z').split('\0')))
    if staged:
        raise ValueError('Pre-existing staged changes; index untouched: ' + ', '.join(sorted(staged)))
    # Frozen launch-owned code must remain byte-identical, even though finish itself is new.
    claim = read(Path(root) / 'preparation/dispatch/claim.json')
    v4 = Path(root).parent.parent
    frozen_names = {str(v4 / n) for n in ('r112_model.py', 'r112_experiment.py', 'r112_reporting.py',
                                        'r112_launch.py', 'run_v4_r112.sh')} | {str(Path(root) / 'protocol_lock.json')}
    if not frozen_names.issubset(claim.get('source_sha256', {})):
        raise ValueError('Missing launch-frozen runtime source hashes')
    for path, expected in claim['source_sha256'].items():
        if digest(path) != expected:
            raise ValueError('Launch-frozen source changed; no publication: ' + path)
    dependencies = dependency_files([p for p in paths if p.suffix == '.py' and p.parent == Path(root).parent.parent])
    dep_names = [str(p.relative_to(GIT_ROOT)) for p in dependencies]
    tracked = set(filter(None, git('ls-files', '-z', '--', *dep_names).split('\0')))
    missing = set(dep_names) - tracked - set(names)
    if missing:
        raise ValueError('Unpublished dependencies outside R112 allowlist; never auto-add: ' + ', '.join(sorted(missing)))
    branch = git('symbolic-ref', '--quiet', '--short', 'HEAD')
    remote = git('config', '--get', 'branch.' + branch + '.remote')
    merge = git('config', '--get', 'branch.' + branch + '.merge')
    if remote == '.' or not merge.startswith('refs/heads/'):
        raise ValueError('Require an existing named remote/upstream branch; no inferred destination')
    tip = git('rev-parse', 'HEAD')
    rows = git('ls-remote', '--heads', remote, merge).splitlines()
    if len(rows) != 1 or rows[0].split()[0] != tip:
        raise ValueError('Local HEAD must equal remote upstream before the one R112 commit; fetch/reconcile manually')
    return dict(paths=names, excluded=excluded, remote=remote, merge=merge, branch=branch, parent=tip,
        sha256={name: digest(GIT_ROOT / name) for name in names},
        dependency_paths=dep_names, dependencies_tracked_or_in_exact_allowlist=True)


@contextmanager
def publication_mutex():
    """Same Git-directory flock as r98/r109_goal; held through commit and push."""
    git_dir = Path(git('rev-parse', '--absolute-git-dir')).resolve()
    path = git_dir / 'codex-publication.lock'
    with path.open('a') as mutex:
        fcntl.flock(mutex, fcntl.LOCK_EX)
        yield path


def publish(root, message='Publish complete R112 region-interface models and analysis'):
    """One explicit attempt. Failed/uncertain commits or pushes are never retried."""
    root = Path(root).resolve()
    with publication_mutex() as global_mutex, (root / 'finish_publish.lock').open('a') as claim:
        fcntl.flock(claim, fcntl.LOCK_EX | fcntl.LOCK_NB)
        receipt_path = root / 'publish_receipt.json'
        if receipt_path.exists():
            receipt = read(receipt_path)
            if receipt.get('state') == 'complete':
                return receipt
            raise RuntimeError('A publication attempt already exists; inspect manually, no automatic retry: ' + str(receipt_path))
        plan = publication_preflight(root)
        manifest = root / 'publish_manifest.json'
        write(manifest, dict(experiment='R112', prepared_utc=utc(), **plan))
        names = [*plan['paths'], str(manifest.relative_to(GIT_ROOT))]
        receipt = dict(experiment='R112', state='prepared', started_utc=utc(), parent=plan['parent'],
            remote=plan['remote'], merge=plan['merge'], commit_attempts=0, push_attempts=0,
            automatic_retry=False, allowlist=names, global_mutex=str(global_mutex))
        with receipt_path.open('x', encoding='utf-8') as stream:
            json.dump(receipt, stream, indent=2)
        try:
            for path, expected in plan['sha256'].items():
                if digest(GIT_ROOT / path) != expected:
                    raise ValueError('Publication input changed before staging: ' + path)
            if git('diff', '--cached', '--name-only', '-z'):
                raise ValueError('Pre-existing staged changes appeared; no stage/commit')
            git('add', '-f', '--', *names)
            staged = set(filter(None, git('diff', '--cached', '--name-only', '-z').split('\0')))
            if not staged or staged - set(names):
                raise ValueError('Staged set must be a nonempty subset of the exact R112 allowlist')
            blobs = {name: git('rev-parse', ':' + name) for name in names}
            receipt.update(state='commit_started', commit_attempts=1)
            write(receipt_path, receipt)
            # Explicit paths protect unrelated index entries if another writer stages mid-operation.
            git('commit', '-m', message, '--only', '--', *names)
            commit = git('rev-parse', 'HEAD')
            receipt.update(state='committed', commit=commit)
            write(receipt_path, receipt)
            changed = set(filter(None, git('diff-tree', '--no-commit-id', '--name-only', '-r', commit).splitlines()))
            if (git('rev-parse', 'HEAD^') != plan['parent'] or changed - set(names)
                    or any(git('rev-parse', commit + ':' + n) != blob for n, blob in blobs.items())):
                raise ValueError('Commit differs from the staged R112-only payload; no push')
            receipt.update(state='push_started', push_attempts=1)
            write(receipt_path, receipt)
            git('push', '--porcelain', plan['remote'], commit + ':' + plan['merge'])
            receipt.update(state='complete', exit_status=0, completed_utc=utc())
            write(receipt_path, receipt)
            return receipt
        except BaseException as error:
            receipt.update(state='failed', exit_status=1, error_type=type(error).__name__, error=str(error),
                           failed_utc=utc(), manual_recovery_required=True)
            write(receipt_path, receipt)
            raise


def wait(root, *, timeout_hours=72., poll_seconds=60., publish_after=False, message=None, references=None):
    if not math.isfinite(timeout_hours) or timeout_hours <= 0 or not math.isfinite(poll_seconds) or poll_seconds <= 0:
        raise ValueError('Wait timeout and poll interval must be finite and positive')
    deadline, previous = time.monotonic() + timeout_hours * 3600, None
    while True:
        try:
            completed(root)
            break
        except NotReady as error:
            if time.monotonic() >= deadline:
                raise TimeoutError('R112 finish wait timed out; no training restart')
            if str(error) != previous:
                print(str(error), flush=True)
                previous = str(error)
            time.sleep(min(poll_seconds, max(0., deadline - time.monotonic())))
    value = analyze(root, references)
    return publish(root, message or 'Publish complete R112 region-interface models and analysis') if publish_after else value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('analyze', 'wait', 'publish'))
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--r94b-root', type=Path, default=REFERENCES['R94B'])
    parser.add_argument('--r89b-root', type=Path, default=REFERENCES['R89B'])
    parser.add_argument('--timeout-hours', type=float, default=72.)
    parser.add_argument('--poll-seconds', type=float, default=60.)
    parser.add_argument('--publish', action='store_true', help='Only with wait: explicitly opt into one commit/push')
    parser.add_argument('--message', default='Publish complete R112 region-interface models and analysis')
    args = parser.parse_args(argv)
    root = args.root.expanduser().resolve()
    references = {name: getattr(args, key).expanduser().resolve() for name, key in
                  (('R94B', 'r94b_root'), ('R89B', 'r89b_root'))}
    if args.publish and args.command != 'wait':
        parser.error('--publish is an explicit wait-only opt-in; otherwise use the publish command')
    try:
        if args.command == 'analyze':
            result = analyze(root, references)
        elif args.command == 'publish':
            result = publish(root, args.message)
        else:
            result = wait(root, timeout_hours=args.timeout_hours, poll_seconds=args.poll_seconds,
                          publish_after=args.publish, message=args.message, references=references)
    except NotReady as error:
        print(str(error), file=sys.stderr)
        return 2
    except TimeoutError as error:
        print(str(error), file=sys.stderr)
        return 124
    except Exception as error:
        print(type(error).__name__ + ': ' + str(error), file=sys.stderr)
        return 1
    print(json.dumps({k: result[k] for k in ('complete', 'state', 'commit', 'metric_rows', 'comparison_rows',
                                           'event_count') if k in result}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

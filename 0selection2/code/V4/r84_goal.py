"""CPU-only R84 progress and matched locked-prediction comparison.

The status path imports neither torch nor numpy. The watcher only reads small
receipts until A's VAL lock can freeze targets, or all four reports are ready.
"""

import argparse
import csv
import fcntl
import hashlib
import json
import os
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from code.unified_selector.registry import PROBLEMS


DEFAULT_ROOT = Path(__file__).resolve().parent / 'runs/R84_new90k_attention'
DEFAULT_LAUNCH_ROOT = Path(__file__).resolve().parent / 'runs/R84_new90k_launch'
ARMS = ('A_baseline', 'B_attention_direct', 'C_focal', 'D_attention_focal')
ROWS = dict(train=81000, val=4500, test=4500)
BUDGET = 57600
NEAR_ZERO_PCT = .1
WINNER_RULE = 'saved native/FP64 winner; not FP32 argmin'
TARGET_RULE = 'A locked VAL: chosen percentage <= 0.1 and native winner count > 0'
METRICS = ('top1_pct', 'mean_cost', 'actual_regret_pct')


def utc():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    with Path(path).open(encoding='utf-8') as stream:
        return json.load(stream)


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    try:
        with temporary.open('w', encoding='utf-8') as stream:
            json.dump(value, stream, indent=2, ensure_ascii=True, allow_nan=False)
            stream.write('\n')
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
        allow_nan=False).encode()).hexdigest()


def file_hash(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(k for row in rows for k in row)))
        writer.writeheader()
        writer.writerows(rows)


def optional(path):
    return read(path) if Path(path).is_file() else {}


def selected_val_path(run, lock):
    fixed = run / 'evaluation/val/selected_best/rank000.npz'
    fallback = run / f'evaluation/val/update{lock["selected_update"]:07d}/rank000.npz'
    return fixed if fixed.is_file() else fallback


def arm_snapshot(root, launch_root, arm):
    run = root / arm
    status = optional(run / 'status.json')
    training = optional(run / 'training_complete.json')
    lock = optional(run / 'final_test_lock.json')
    test = optional(run / 'test_results.json')
    lib = optional(run / 'lib_stage_complete.json')
    post = optional(run / 'postprocess_status.json')
    report = optional(run / 'full_model_statistics/protocol.json')
    controller = optional(launch_root / arm / 'controller.json')
    worker = optional(launch_root / arm / 'worker.json')
    missing = []
    requirements = {
        'training_complete': training.get('successful_updates') == BUDGET,
        'unique_VAL_lock': bool(lock.get('selected_without_test')) and lock.get('per_problem_assembly') is False,
        'full_TEST': test.get('complete') is True,
        'LIB': lib.get('complete') is True,
        'arm_PDF': post.get('complete') is True and post.get('phase') == 'complete'
            and report.get('complete') is True,
    }
    for name, valid in requirements.items():
        if not valid:
            missing.append(name)
    for name in ('MODEL_STATISTICS.pdf', 'selection_distributions.pdf'):
        path = run / 'full_model_statistics' / name
        if not path.is_file() or path.stat().st_size == 0:
            missing.append(name)
    errors = []
    for path, record in ((run / 'status.json', status), (run / 'postprocess_status.json', post),
                         (run / 'lib_stage_status.json', optional(run / 'lib_stage_status.json')),
                         (launch_root / arm / 'controller.json', controller),
                         (launch_root / arm / 'worker.json', worker)):
        phase = str(record.get('phase', record.get('state', ''))).lower()
        if 'failed' in phase or phase == 'error' or record.get('state') == 'failed':
            errors.append(dict(path=str(path), error=record.get('error', record.get('message', phase)),
                action='Inspect this receipt and its stage log; retain checkpoints and predictions. '
                       'Repair only the failed stage manually. This watcher does not retry.'))
    updates = status.get('successful_updates', training.get('successful_updates', 0))
    return dict(arm=arm, phase=status.get('phase', controller.get('phase', 'not_started')),
        successful_updates=updates, budget=BUDGET, progress_pct=100. * updates / BUDGET,
        updated_utc=status.get('updated_utc'), host=status.get('host'), gpu=status.get('gpu'),
        eta_seconds=status.get('eta_seconds'), selected_update=lock.get('selected_update'),
        latest_val=status.get('latest_val'), reporting_phase=post.get('phase', 'not_started'),
        launcher_state=controller.get('state'), ready_for_comparison=not missing,
        missing=missing, errors=errors)


def status(root, launch_root=DEFAULT_LAUNCH_ROOT):
    """One cheap receipt snapshot, without opening prediction or checkpoint files."""
    root, launch_root = Path(root).resolve(), Path(launch_root).resolve()
    rows = [arm_snapshot(root, launch_root, arm) for arm in ARMS]
    errors = [dict(arm=row['arm'], **error) for row in rows for error in row['errors']]
    comparison = optional(root / 'comparison_artifacts/protocol.json')
    failed_report = optional(root / 'comparison_artifacts/error.json')
    if failed_report and not comparison.get('complete'):
        errors.append(dict(arm='comparison', path=str(root / 'comparison_artifacts/error.json'),
            error=failed_report.get('error'), action='Fix comparison inputs or code, then run the report '
                'command manually; do not restart training.'))
    all_ready = all(row['ready_for_comparison'] for row in rows)
    complete = all_ready and comparison.get('complete') is True
    result = dict(experiment='R84', checked_utc=utc(), poll_seconds=60,
        phase='failed' if errors else 'comparison_complete' if complete else
              'ready_for_comparison' if all_ready else 'waiting_for_arms',
        goal_status='unresolved', goal_complete=False, comparison_complete=complete,
        all_arms_ready=all_ready, arms=rows, errors=errors, cpu_only=True,
        training_started_by_monitor=False, automatic_retry=False,
        target_rule=TARGET_RULE, target_scope='Descriptive posthoc mechanism list from A VAL; '
            'never a TRAIN sampling or checkpoint-selection rule.',
        acceptance_rule='Unrounded VAL macro Top1 > A and macro mean_cost <= A and '
            'macro actual regret <= A. TEST is reported for locked versions only.',
        next_action='manual_failure_review' if errors else comparison.get('next_action') if complete else
            'derive_saved_prediction_comparison' if all_ready else 'wait_for_committed_arms',
        comparison_pdf=str(root / 'comparison_artifacts/COMPARISON.pdf') if complete else None)
    save(root / 'goal_progress.json', result)
    return result


@dataclass
class Prediction:
    pool: list
    pool_ids: object
    costs: object
    winner: object
    pred: object
    indices: object
    source: object


def load_prediction(archive, problem, config, split):
    import numpy as np
    prefix = problem + '__'
    get = lambda name: archive[prefix + name]
    pool, ids = get('pool').tolist(), get('pool_ids')
    costs, winner, logits, indices = (get(name) for name in ('costs', 'winner', 'logits', 'indices'))
    expected = ROWS[split.lower()]
    if (pool != config['candidate_pools'][problem] or ids.tolist() != config['candidate_pool_ids'][problem]
            or costs.shape != (expected, len(pool)) or costs.dtype != np.float64
            or not np.isfinite(costs).all() or np.any(costs <= 0)
            or logits.shape != costs.shape or not np.isfinite(logits).all()
            or winner.shape != (expected,) or winner.dtype.kind not in 'iu'
            or len(set(ids.tolist())) != len(pool) or len(set(pool)) != len(pool)):
        raise ValueError(f'Incomplete FP64 {split}/{problem} population or candidate identity')
    np.testing.assert_array_equal(indices, np.arange(expected))
    np.testing.assert_array_equal(winner, costs.argmin(1))
    order = np.argsort(ids, kind='stable')
    pred = order[np.argmax(logits[:, order], axis=1)]
    if prefix + 'pred' in archive:
        np.testing.assert_array_equal(pred, get('pred'))
    source = get('release_source_indices') if prefix + 'release_source_indices' in archive else None
    return Prediction(pool, ids, costs, winner, pred, indices, source)


def load_locked_val(root, arm):
    run = Path(root) / arm
    config, lock = read(run / 'protocol.json'), read(run / 'final_test_lock.json')
    if (config.get('experiment') != 'R84' or config.get('arm') != arm
            or lock.get('experiment') != 'R84' or lock.get('arm') != arm
            or not lock.get('selected_without_test') or lock.get('per_problem_assembly') is not False
            or lock.get('state_hash') != digest(config) or config.get('rows_per_problem') != ROWS
            or config.get('seed') != 2 or config.get('logical_global_batch') != 1024
            or config.get('max_successful_updates') != BUDGET
            or not 0 < lock.get('selected_update', 0) <= BUDGET):
        raise ValueError(f'{arm}: require the committed R84 protocol and unique VAL lock')
    path = selected_val_path(run, lock)
    identity = read(path.parent / 'identity.json')
    check_prediction_identity(identity, lock, arm, 'val')
    return config, lock, path, identity


def check_prediction_identity(identity, lock, arm, split):
    if (identity.get('experiment') != 'R84' or identity.get('arm') != arm
            or identity.get('split') != split or identity.get('state_hash') != lock['state_hash']
            or identity.get('successful_updates') != lock['selected_update']
            or not identity.get('model_state_sha256')
            or split == 'test' and identity.get('checkpoint_sha256') != lock['checkpoint_sha256']):
        raise ValueError(f'{arm}/{split}: saved predictions differ from the unique VAL lock')


def freeze_targets(root):
    """Freeze from A VAL before reading candidate-arm predictions or any TEST."""
    import numpy as np
    root = Path(root)
    config, lock, path, _ = load_locked_val(root, ARMS[0])
    identity = dict(baseline_checkpoint_sha256=lock['checkpoint_sha256'],
        baseline_state_hash=lock['state_hash'], baseline_val_prediction_sha256=file_hash(path),
        data_root=config['data_root'], threshold_chosen_pct=NEAR_ZERO_PCT, rule=TARGET_RULE)
    destination = root / 'comparison_artifacts/target_list.json'
    if destination.is_file():
        frozen = read(destination)
        if frozen['identity'] != identity:
            raise ValueError('The frozen A-VAL target list belongs to different predictions or rule')
        return frozen
    targets = []
    with np.load(path, allow_pickle=False) as archive:
        for problem in PROBLEMS:
            data = load_prediction(archive, problem, config, 'VAL')
            for m, method in enumerate(data.pool):
                wins, chosen = int((data.winner == m).sum()), int((data.pred == m).sum())
                if wins > 0 and chosen * 100 <= NEAR_ZERO_PCT * len(data.pred):
                    targets.append(dict(problem=problem, method=method, baseline_val_winners=wins,
                        baseline_val_chosen=chosen, baseline_val_chosen_pct=100. * chosen / len(data.pred),
                        category='zero' if chosen == 0 else 'near_zero'))
    result = dict(frozen_utc=utc(), identity=identity, targets=targets,
        mechanism_only=True, affects_training=False, affects_selection=False,
        posthoc_note='A VAL outcome defines this descriptive list. The threshold was predeclared. '
                     'TEST never defines or expands the list.')
    save(destination, result)
    write_csv(destination.with_suffix('.csv'), targets)
    return result


def metrics(data, baseline):
    import numpy as np
    n = len(data.pred)
    oracle = data.costs.min(1)
    selected = data.costs[np.arange(n), data.pred]
    sbs = data.costs[:, data.pool.index(baseline['sbs_solver'])]
    return dict(n=n, native_correct=int((data.pred == data.winner).sum()),
        top1_pct=100. * float(np.mean(data.pred == data.winner)), mean_cost=float(selected.mean()),
        actual_regret_pct=100. * float(np.mean((selected - oracle) / oracle)),
        sbs_cost=float(sbs.mean()), oracle_cost=float(oracle.mean()),
        vs_sbs_pct=100. * float(selected.mean() / sbs.mean() - 1),
        vs_oracle_pct=100. * float(selected.mean() / oracle.mean() - 1))


def paired(data, current):
    import numpy as np
    for name in ('pool_ids', 'costs', 'winner', 'indices'):
        np.testing.assert_array_equal(getattr(data, name), getattr(current, name),
            err_msg='Matched arms must use identical instances, FP64 costs and native labels')
    if data.pool != current.pool or (data.source is None) != (current.source is None):
        raise ValueError('Matched arm candidate/source identities differ')
    if data.source is not None:
        np.testing.assert_array_equal(data.source, current.source)
    old_ok, new_ok = data.pred == data.winner, current.pred == data.winner
    row = np.arange(len(data.pred))
    old, new = data.costs[row, data.pred], data.costs[row, current.pred]
    delta, oracle = new - old, data.costs.min(1)
    groups = dict(corrected=(~old_ok & new_ok), harmed=(old_ok & ~new_ok),
        both_wrong=(~old_ok & ~new_ok), both_correct=(old_ok & new_ok))
    result = dict(n=len(row), changed=int((data.pred != current.pred).sum()))
    for label, mask in groups.items():
        result[label + '_n'] = int(mask.sum())
        result[label + '_raw_cost_delta_sum'] = float(delta[mask].sum())
        result[label + '_regret_delta_pp'] = 100. * float(np.sum(delta[mask] / oracle[mask]) / len(row))
    wrong = groups['both_wrong']
    result.update(both_wrong_cheaper_n=int((wrong & (delta < 0)).sum()),
        both_wrong_dearer_n=int((wrong & (delta > 0)).sum()),
        both_wrong_equal_cost_n=int((wrong & (delta == 0)).sum()),
        net_mean_cost_delta=float(delta.mean()), net_regret_delta_pp=100. * float(np.mean(delta / oracle)))
    return result


def method_metrics(baseline, current, m):
    import numpy as np
    truth, selected, old = current.winner == m, current.pred == m, baseline.pred == m
    row = np.arange(len(truth))
    oracle = current.costs.min(1)
    regret = (current.costs[row, current.pred] - oracle) / oracle
    tp, count, winners = int((truth & selected).sum()), int(selected.sum()), int(truth.sum())
    rescued, lost = truth & ~old & selected, truth & old & ~selected
    new_false = selected & ~truth & ~old
    old_regret = (baseline.costs[row, baseline.pred] - oracle) / oracle
    return dict(n=len(truth), winner_n=winners, selected_n=count, true_positive_n=tp,
        recall_pct=100. * tp / winners if winners else None,
        precision_pct=100. * tp / count if count else None, false_picks_n=int((selected & ~truth).sum()),
        new_false_picks_n=int(new_false.sum()), rescued_winner_n=int(rescued.sum()),
        lost_winner_n=int(lost.sum()), false_pick_regret_contribution_pp=100. * float(regret[selected & ~truth].sum() / len(truth)),
        new_false_pick_regret_contribution_pp=100. * float(regret[new_false].sum() / len(truth)),
        target_winner_net_regret_delta_pp=100. * float((regret[truth] - old_regret[truth]).sum() / len(truth)))


def val_acceptance(current, baseline):
    return bool(current['top1_pct'] > baseline['top1_pct'] and
        current['mean_cost'] <= baseline['mean_cost'] and
        current['actual_regret_pct'] <= baseline['actual_regret_pct'])


def matched_config(configs):
    reference = configs[ARMS[0]]
    keys = ('seed', 'data_root', 'rows_per_problem', 'candidate_pools', 'candidate_pool_ids',
        'scenario_solver_aliases', 'preparation_sha256', 'common_trainable_sha256', 'task_order_sha256',
        'logical_global_batch', 'max_successful_updates', 'eval_every_updates', 'warmup_updates',
        'lr', 'dropout', 'optimizer', 'weight_decay', 'grad_clip', 'checkpoint_rule', 'baselines')
    for arm in ARMS[1:]:
        for key in keys:
            if configs[arm].get(key) != reference.get(key):
                raise ValueError(f'{arm}: unmatched experiment field {key}')


def derive(root, frozen):
    import numpy as np
    configs, locks, inputs, provenance = {}, {}, {}, []
    for arm in ARMS:
        config, lock, val_path, val_identity = load_locked_val(root, arm)
        test_path = root / arm / 'evaluation/test/selected_best/rank000.npz'
        test_identity = read(test_path.parent / 'identity.json')
        check_prediction_identity(test_identity, lock, arm, 'test')
        if test_identity['model_state_sha256'] != val_identity['model_state_sha256']:
            raise ValueError(f'{arm}: VAL and TEST are not the same locked model')
        config_baselines = config['baselines']
        if set(config_baselines) != set(PROBLEMS) or any(
                b.get('selection_split') != 'train' for b in config_baselines.values()):
            raise ValueError('Require TRAIN-fitted SBS for all18 tasks')
        configs[arm], locks[arm] = config, lock
        inputs[arm] = dict(VAL=val_path, TEST=test_path)
        provenance.append(dict(arm=arm, selected_update=lock['selected_update'],
            checkpoint_sha256=lock['checkpoint_sha256'], state_hash=lock['state_hash'],
            model_state_sha256=val_identity['model_state_sha256'],
            VAL_prediction_sha256=file_hash(val_path), TEST_prediction_sha256=file_hash(test_path)))
    matched_config(configs)
    target_keys = {(r['problem'], r['method']) for r in frozen['targets']}
    metric_rows, pair_rows, method_rows = [], [], []
    for split in ('VAL', 'TEST'):
        archives = {arm: np.load(inputs[arm][split], allow_pickle=False) for arm in ARMS}
        try:
            for problem in PROBLEMS:
                data = {arm: load_prediction(archives[arm], problem, configs[arm], split) for arm in ARMS}
                baseline = data[ARMS[0]]
                for arm in ARMS:
                    current = data[arm]
                    delta = paired(baseline, current)
                    pair_rows.append(dict(split=split, arm=arm, problem=problem, **delta))
                    metric_rows.append(dict(split=split, arm=arm, problem=problem,
                        **metrics(current, configs[arm]['baselines'][problem])))
                    for m, method in enumerate(current.pool):
                        method_rows.append(dict(split=split, arm=arm, problem=problem, method=method,
                            frozen_target=(problem, method) in target_keys,
                            **method_metrics(baseline, current, m)))
        finally:
            for archive in archives.values():
                archive.close()
    for split in ('VAL', 'TEST'):
        for arm in ARMS:
            rows = [r for r in metric_rows if r['split'] == split and r['arm'] == arm]
            macro = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]
                     if k not in ('split', 'arm', 'problem', 'n', 'native_correct')}
            metric_rows.append(dict(split=split, arm=arm, problem='ALL',
                n=sum(r['n'] for r in rows), native_correct=sum(r['native_correct'] for r in rows), **macro))
    gates = []
    for arm in ARMS[1:]:
        row = dict(arm=arm, acceptance_split='VAL', test_used_for_acceptance=False)
        for split in ('VAL', 'TEST'):
            base = next(r for r in metric_rows if (r['split'], r['arm'], r['problem']) == (split, ARMS[0], 'ALL'))
            curr = next(r for r in metric_rows if (r['split'], r['arm'], r['problem']) == (split, arm, 'ALL'))
            for key in METRICS:
                row[split + '_' + key + '_delta'] = curr[key] - base[key]
            if split == 'VAL':
                row['accepted_on_VAL'] = val_acceptance(curr, base)
        gates.append(row)
    targets = []
    for split in ('VAL', 'TEST'):
        for arm in ARMS:
            rows = [r for r in method_rows if r['split'] == split and r['arm'] == arm and r['frozen_target']]
            winners, tp, chosen = (sum(r[k] for r in rows) for k in ('winner_n', 'true_positive_n', 'selected_n'))
            baseline_rows = [r for r in method_rows if r['split'] == split and r['arm'] == ARMS[0] and r['frozen_target']]
            baseline_tp = sum(r['true_positive_n'] for r in baseline_rows)
            missed = winners - baseline_tp
            rescued = sum(r['rescued_winner_n'] for r in rows)
            targets.append(dict(split=split, arm=arm, frozen_target_groups=len(target_keys),
                target_winner_n=winners, selected_target_method_n=chosen, true_positive_n=tp,
                total_winner_recall_pct=100. * tp / winners if winners else None,
                selection_precision_pct=100. * tp / chosen if chosen else None,
                baseline_missed_target_winner_n=missed, rescued_winner_n=rescued,
                rescued_of_baseline_misses_pct=100. * rescued / missed if missed else None,
                lost_winner_n=sum(r['lost_winner_n'] for r in rows), net_winner_rescue_n=tp - baseline_tp,
                false_picks_n=sum(r['false_picks_n'] for r in rows),
                new_false_picks_n=sum(r['new_false_picks_n'] for r in rows),
                false_pick_macro_regret_contribution_pp=sum(r['false_pick_regret_contribution_pp'] for r in rows) / len(PROBLEMS),
                new_false_pick_macro_regret_contribution_pp=sum(r['new_false_pick_regret_contribution_pp'] for r in rows) / len(PROBLEMS),
                target_winner_macro_regret_delta_pp=sum(r['target_winner_net_regret_delta_pp'] for r in rows) / len(PROBLEMS)))
    return dict(metrics=metric_rows, paired=pair_rows, methods=method_rows, targets=targets,
        gates=gates, provenance=provenance, data_root=configs[ARMS[0]]['data_root'])


def make_pdf(output, result):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'pdf.fonttype': 42})

    def table(pdf, title, header, rows, note):
        fig = plt.figure(figsize=(15, 9))
        fig.text(.045, .94, title, fontsize=18, weight='bold', color='#104f55')
        ax = fig.add_axes([.045, .20, .91, .64])
        ax.axis('off')
        artist = ax.table(cellText=rows, colLabels=header, loc='upper center', cellLoc='center')
        artist.auto_set_font_size(False)
        artist.set_fontsize(9 if len(rows) > 10 else 11)
        artist.scale(1, 1.5)
        for (row, _), cell in artist.get_celld().items():
            cell.set_edgecolor('#d9e2e3')
            cell.set_facecolor('#d6eaeb' if row == 0 else '#f3f7f7' if row % 2 else 'white')
        for i, line in enumerate(note):
            fig.text(.045, .14 - i * .026, line, fontsize=9)
        pdf.savefig(fig)
        plt.close(fig)

    def fmt(v, digits=4):
        return 'n/a' if v is None else f'{v:.{digits}f}'

    destination = output / 'COMPARISON.pdf'
    temporary = output / f'.comparison.{os.getpid()}.pdf'
    try:
        with PdfPages(temporary) as pdf:
            overall = [r for r in result['metrics'] if r['problem'] == 'ALL']
            table(pdf, 'R84 | Four locked models: VAL and TEST separately',
                ['Split', 'Arm', 'Model Top1 (%)', 'mean_cost', 'Actual regret (%)'],
                [[r['split'], r['arm'], fmt(r['top1_pct']), fmt(r['mean_cost'], 9),
                  fmt(r['actual_regret_pct'], 6)] for r in overall],
                ['ALL = equal weight across18 tasks. Mixed raw mean_cost is descriptive.',
                 'Native/FP64 winners. One VAL-selected checkpoint per arm. No TEST-driven rule/checkpoint changes.',
                 'Previously published TEST rows have influenced research. These are not fully blind final tests.',
                 'Acceptance below uses unrounded VAL values only. The continuing research goal remains unresolved.'])
            table(pdf, 'R84 | Paired acceptance gates versus A_baseline',
                ['Arm', 'VAL dTop1 (pp)', 'VAL dCost', 'VAL dRegret (pp)', 'VAL accepted', 'TEST dTop1 (pp)'],
                [[r['arm'], fmt(r['VAL_top1_pct_delta'], 6), fmt(r['VAL_mean_cost_delta'], 9),
                  fmt(r['VAL_actual_regret_pct_delta'], 6), str(r['accepted_on_VAL']),
                  fmt(r['TEST_top1_pct_delta'], 6)] for r in result['gates']],
                ['VAL acceptance: Top1 strictly improves, mean_cost does not increase, actual regret does not increase.',
                 'All comparisons use the recorded checkpoints; this report does not select a winning deployment.',
                 'A smaller number of zero-selected methods is not an acceptance criterion.'])
            table(pdf, 'R84 | Frozen A-VAL zero / near-zero method targets',
                ['Split', 'Arm', 'Winners', 'Recall (%)', 'Rescued / lost', 'New false picks', 'New-FP regret (pp)'],
                [[r['split'], r['arm'], str(r['target_winner_n']), fmt(r['total_winner_recall_pct'], 2),
                  f'{r["rescued_winner_n"]} / {r["lost_winner_n"]}', str(r['new_false_picks_n']),
                  fmt(r['new_false_pick_macro_regret_contribution_pp'], 6)] for r in result['targets']],
                ['Frozen definition: A locked VAL chosen share <=0.1%, with at least one native winner.',
                 'This baseline-dependent target list is descriptive/posthoc; it never changes TRAIN sampling.',
                 'TEST uses exactly the frozen method/task list. Rescues and false picks are both reported.'])
            for split in ('VAL', 'TEST'):
                for arm in ARMS[1:]:
                    deltas = [r for r in result['paired'] if r['split'] == split and r['arm'] == arm]
                    current = {r['problem']: r for r in result['metrics'] if r['split'] == split and r['arm'] == arm}
                    baseline = {r['problem']: r for r in result['metrics'] if r['split'] == split and r['arm'] == ARMS[0]}
                    table(pdf, f'R84 | {split}: {arm} versus A_baseline',
                        ['Task', 'dTop1 (pp)', 'Corrected', 'Harmed', 'Both wrong: cheaper/dearer', 'dRegret (pp)'],
                        [[r['problem'], fmt(current[r['problem']]['top1_pct'] - baseline[r['problem']]['top1_pct'], 3),
                          str(r['corrected_n']), str(r['harmed_n']),
                          f'{r["both_wrong_cheaper_n"]} / {r["both_wrong_dearer_n"]}', fmt(r['net_regret_delta_pp'], 5)]
                         for r in deltas],
                        ['Negative dRegret means lower selected cost. Per-task raw deltas are available in the paired CSV.',
                         'Corrected/harmed use native-winner identity, even if exact-cost ties have zero cost change.'])
            table(pdf, 'R84 | Locked checkpoint identities', ['Arm', 'Selected update', 'SHA256 prefix'],
                [[r['arm'], str(r['selected_update']), r['checkpoint_sha256'][:24]] for r in result['provenance']],
                ['Full hashes and prediction hashes: comparison_artifacts/protocol.json.',
                 'Same81000/4500/4500 population; fresh shared initialization; seed2; logical batch1024.',
                 'This CPU report reads saved predictions. It starts no inference, training, solver or retry.'])
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def report(root, launch_root=DEFAULT_LAUNCH_ROOT):
    root = Path(root).resolve()
    snapshot = status(root, launch_root)
    arm_errors = [error for error in snapshot['errors'] if error['arm'] != 'comparison']
    if arm_errors or not snapshot['all_arms_ready']:
        raise RuntimeError('All four locked TEST, LIB and per-arm PDF completions are required; see goal_progress.json')
    output = root / 'comparison_artifacts'
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'report.lock').open('a') as claim:
        fcntl.flock(claim, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            frozen = freeze_targets(root)
            result = derive(root, frozen)
            fingerprint = digest(result['provenance'])
            previous = optional(output / 'protocol.json')
            if previous.get('complete') and previous.get('input_fingerprint') == fingerprint:
                return previous
            for filename, key in (('metrics_by_problem.csv', 'metrics'), ('paired_changes.csv', 'paired'),
                                  ('method_precision_recall.csv', 'methods'), ('target_summary.csv', 'targets'),
                                  ('paired_gates.csv', 'gates')):
                write_csv(output / filename, result[key])
            pdf = make_pdf(output, result)
            accepted = [r['arm'] for r in result['gates'] if r['accepted_on_VAL']]
            next_action = 'review_validation_gain' if accepted else 'research_handoff_needed'
            protocol = dict(experiment='R84', complete=True, completed_utc=utc(),
                input_fingerprint=fingerprint, provenance=result['provenance'], data_root=result['data_root'],
                rows_per_problem=ROWS, winner_rule=WINNER_RULE, macro_aggregation='equal18-task weights',
                target_identity=frozen['identity'], target_groups=len(frozen['targets']),
                target_list_posthoc_descriptive=True, train_sampling_changed=False,
                validation_gates=result['gates'], accepted_on_VAL=accepted, test_used_for_acceptance=False,
                goal_status='unresolved', goal_complete=False, next_action=next_action,
                training_started=False, inference_started=False, automatic_retry=False, comparison_pdf=str(pdf))
            lines = ['# R84 matched comparison', '', 'Goal: unresolved. Next action: `' + next_action + '`.', '',
                'Each arm uses one checkpoint selected only by its recorded full validation rule.',
                'Acceptance requires unrounded VAL macro Top1 > A, mean_cost <= A and actual regret <= A.',
                'TEST is reported separately for locked versions and does not select a checkpoint or rule.', '',
                '| Split | Arm | Top1 (%) | mean_cost | Actual regret (%) |',
                '| --- | --- | ---: | ---: | ---: |']
            for r in result['metrics']:
                if r['problem'] == 'ALL':
                    lines.append(f'| {r["split"]} | {r["arm"]} | {r["top1_pct"]:.6f} | '
                        f'{r["mean_cost"]:.9f} | {r["actual_regret_pct"]:.6f} |')
            lines += ['', 'Frozen targets: A locked VAL chosen share <=0.1% and native winners >0. '
                'This is a posthoc mechanism analysis, not a sampling change or success criterion.',
                'Fewer zero-selection groups alone does not establish improvement.',
                'All18 problems are equally weighted; mixed raw mean_cost is descriptive. '
                'SBS is fitted on TRAIN; oracle is each legal-pool FP64 minimum.',
                'Previously reported TEST rows have participated in research; no fresh blind-test claim.',
                'PDF: [COMPARISON.pdf](COMPARISON.pdf). CSVs include all methods, per-task corrections, '
                'harms, both-wrong cost changes, target rescues and new false picks.',
                'A negative result requests a research handoff. No new training is launched automatically.', '']
            (output / 'RESULTS.md').write_text('\n'.join(lines), encoding='utf-8')
            save(output / 'protocol.json', protocol)
            (output / 'error.json').unlink(missing_ok=True)
            status(root, launch_root)
            return protocol
        except BaseException as error:
            save(output / 'error.json', dict(updated_utc=utc(), error=f'{type(error).__name__}: {error}',
                traceback=traceback.format_exc(), action='Inspect this comparison error and source receipts; '
                    'repair only the failed comparison stage, then invoke report manually.', automatic_retry=False))
            status(root, launch_root)
            raise


def watch(root, launch_root=DEFAULT_LAUNCH_ROOT, interval=60, max_polls=None):
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    with (root / 'goal_watch.lock').open('a') as claim:
        fcntl.flock(claim, fcntl.LOCK_EX | fcntl.LOCK_NB)
        save(root / 'goal_watch_receipt.json', dict(started_utc=utc(), pid=os.getpid(),
            interval_seconds=interval, cpu_only=True, automatic_retry=False, autonomous_training=False))
        polls = 0
        while True:
            try:
                snapshot = status(root, launch_root)
                if snapshot['errors']:
                    return 1
                if snapshot['comparison_complete']:
                    return 0
                # Target freezing is a one-time A-VAL read, separate from TEST.
                a = root / ARMS[0]
                if ((a / 'training_complete.json').is_file() and (a / 'final_test_lock.json').is_file()
                        and not (root / 'comparison_artifacts/target_list.json').is_file()):
                    freeze_targets(root)
                if snapshot['all_arms_ready']:
                    report(root, launch_root)
                    return 0
            except BaseException as error:
                save(root / 'comparison_artifacts/error.json', dict(updated_utc=utc(),
                    error=f'{type(error).__name__}: {error}', traceback=traceback.format_exc(),
                    action='Inspect inputs and fix the monitor/report manually. No automatic retry or training.',
                    automatic_retry=False))
                try:
                    status(root, launch_root)
                except (OSError, ValueError):
                    pass
                return 1
            polls += 1
            if max_polls is not None and polls >= max_polls:
                return 0
            time.sleep(interval)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('command', choices=('status', 'watch', 'report'))
    result.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    result.add_argument('--launch-root', type=Path, default=DEFAULT_LAUNCH_ROOT)
    result.add_argument('--interval', type=int, default=60)
    result.add_argument('--max-polls', type=int)
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    if args.interval < 1 or args.max_polls is not None and args.max_polls < 1:
        raise ValueError('Poll interval and bounded poll count must be positive')
    if args.command == 'watch':
        return watch(args.root, args.launch_root, args.interval, args.max_polls)
    value = report(args.root, args.launch_root) if args.command == 'report' else status(args.root, args.launch_root)
    print(json.dumps(value, indent=2, allow_nan=False))
    return 0


if __name__ == '__main__':
    sys.exit(main())

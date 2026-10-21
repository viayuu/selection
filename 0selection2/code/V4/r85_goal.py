"""Quiet CPU receipt watcher and final saved-prediction R85 comparison.

Only the final report lazily imports numerical/reporting libraries. This module
never launches inference/training, retries stages, or completes the research goal.
"""

import argparse
import fcntl
import os
import sys
import time
import traceback
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')

from . import r84_goal as receipts
from code.unified_selector.registry import PROBLEMS


ROOT = Path(__file__).resolve().parent / 'runs/R85_zero_selection_research'
LAUNCH = ROOT.parent / 'R85_launch'
HISTORICAL = ROOT.parent / 'R84_new90k_attention/A_baseline'
ARMS = ('A_ovr_control', 'B_gradient_balance', 'C_cost_profile', 'D_candidate_subset')
METRICS = ('top1_pct', 'mean_cost', 'actual_regret_pct')
read, save, optional = receipts.read, receipts.save, receipts.optional


def status(root=ROOT, launch_root=LAUNCH):
    root, launch_root = Path(root).resolve(), Path(launch_root).resolve()
    arms = [receipts.arm_snapshot(root, launch_root, arm) for arm in ARMS]
    errors = [dict(arm=row['arm'], **error) for row in arms for error in row['errors']]
    complete = optional(root / 'comparison_artifacts/protocol.json')
    failed = optional(root / 'comparison_artifacts/error.json')
    if failed and not complete.get('complete'):
        errors.append(dict(arm='comparison', path=str(root / 'comparison_artifacts/error.json'),
                           error=failed.get('error'), action='Repair this stage manually; no automatic retry.'))
    ready = all(row['ready_for_comparison'] for row in arms)
    done = ready and complete.get('complete') is True
    for row in arms:
        lock = optional(root / row['arm'] / 'final_test_lock.json')
        row['locked_val'] = lock.get('val')
        row['latest_val_is_checkpoint_selection'] = False
    result = dict(experiment='R85', checked_utc=receipts.utc(), poll_seconds=60,
                  phase='failed' if errors else 'comparison_complete' if done else
                        'ready_for_comparison' if ready else 'waiting_for_arms',
                  arms=arms, errors=errors, error_flags=bool(errors), all_arms_ready=ready,
                  comparison_complete=done, cpu_only=True, automatic_retry=False,
                  training_started_by_monitor=False, goal_status='active_unresolved', goal_complete=False,
                  target_source=str(HISTORICAL.parent / 'comparison_artifacts/target_list.json'),
                  target_scope='R84A locked VAL zero/near-zero list, descriptive/posthoc; never a training rule',
                  validation_screen='unrounded full18 VAL Top1 > R84A; mean_cost and actual regret <= R84A',
                  test_used_for_acceptance=False,
                  next_action='manual_failure_review' if errors else complete.get('next_action') if done else
                              'aggregate_completed_saved_predictions' if ready else 'wait_for_committed_arms',
                  comparison_pdf=str(root / 'comparison_artifacts/COMPARISON.pdf') if done else None,
                  liveness_note='Small saved receipts only; no GPU/process liveness inference from stale timestamps.')
    save(root / 'goal_progress.json', result)
    return result


def freeze_targets(root=ROOT, historical=HISTORICAL):
    """Reuse the published R84 list, never expand it using R85 or TEST outcomes."""
    root, historical = Path(root), Path(historical)
    source = historical.parent / 'comparison_artifacts/target_list.json'
    frozen = read(source)
    lock, config = read(historical / 'final_test_lock.json'), read(historical / 'protocol.json')
    identity = frozen['identity']
    if (identity['baseline_checkpoint_sha256'] != lock['checkpoint_sha256']
            or identity['baseline_state_hash'] != lock['state_hash']
            or identity['data_root'] != config['data_root']):
        raise ValueError('Published targets are not from the historical R84A VAL lock')
    keys = [(row['problem'], row['method']) for row in frozen['targets']]
    if len(set(keys)) != len(keys):
        raise ValueError('Duplicate frozen target identities')
    result = dict(source=str(source.resolve()), source_sha256=receipts.file_hash(source),
                  identity=identity, targets=frozen['targets'], mechanism_only=True,
                  affects_training=False, affects_selection=False,
                  baseline_val_winners=sum(row['baseline_val_winners'] for row in frozen['targets']),
                  groups=len(keys), posthoc_descriptive=True)
    path = root / 'comparison_artifacts/target_list.json'
    if path.is_file() and read(path) != result:
        raise ValueError('Existing R85 target list cannot be redefined')
    if not path.is_file():
        save(path, result)
        receipts.write_csv(path.with_suffix('.csv'), frozen['targets'])
    return result


def passes(current, baseline):
    return (current['top1_pct'] > baseline['top1_pct'] and
            current['mean_cost'] <= baseline['mean_cost'] and
            current['actual_regret_pct'] <= baseline['actual_regret_pct'])


def summarize_targets(rows, model, baseline, split, definition):
    total = dict(model=model, baseline=baseline, split=split, target_definition=definition,
                 target_groups=len(rows))
    for key in ('winner_count', 'baseline_missed', 'correct_count', 'selected_count', 'rescued',
                'lost', 'false_pick_count', 'new_false_pick_count'):
        total[key] = sum(row[key] for row in rows)
    for key in ('new_false_pick_regret_contribution_pp', 'target_regret_delta_pp'):
        total[key] = sum(row[key] for row in rows) / len(PROBLEMS)
    total.update(recall_pct=100. * total['correct_count'] / total['winner_count'] if total['winner_count'] else None,
                 precision_pct=100. * total['correct_count'] / total['selected_count'] if total['selected_count'] else None,
                 rescued_of_baseline_misses_pct=100. * total['rescued'] / total['baseline_missed'] if total['baseline_missed'] else None)
    return total


def derive(root, frozen, historical=HISTORICAL):
    from . import r85_reporting as api
    root, historical = Path(root), Path(historical)
    previous = api.shared.locked_inputs(historical.parent, historical.name)
    old_run, old_config, old_lock, old_test, old_val = previous
    old_predictions, old_metrics, _ = api.shared.load_saved_splits(old_run, old_config, old_lock, old_test, old_val)
    metric_rows = [dict(row, model='R84A') for row in old_metrics if row['split'] in ('VAL', 'TEST')]
    packed = {'R84A': old_predictions}
    provenance = [dict(model='R84A', checkpoint_sha256=old_lock['checkpoint_sha256'],
                       selected_update=old_lock['selected_update'], state_hash=old_lock['state_hash'])]
    controls, train_rows = None, []
    for arm in ARMS:
        run, config, lock, test, val = api.locked_inputs(root, arm)
        if (Path(config['data_root']).resolve() != Path(old_config['data_root']).resolve()
                or config['rows_per_problem'] != dict(train=81000, val=4500, test=4500)):
            raise ValueError('R85 comparison requires the exact NEW90k release and 4500 held-out rows/task')
        keys = ('seed', 'data_root', 'rows_per_problem', 'candidate_pools', 'candidate_pool_ids',
                'common_trainable_sha256', 'task_order_sha256', 'logical_global_batch',
                'max_successful_updates', 'warmup_updates', 'lr', 'checkpoint_rule', 'baselines')
        control = {key: config[key] for key in keys}
        if controls is not None and controls != control:
            raise ValueError('The four R85 arms do not share the fixed controls')
        controls = control
        predictions, metrics, _ = api.shared.load_saved_splits(run, config, lock, test, val)
        packed[arm] = predictions
        metric_rows.extend(dict(row, model=arm) for row in metrics if row['split'] in ('VAL', 'TEST'))
        provenance.append(dict(model=arm, selected_update=lock['selected_update'],
                               checkpoint_sha256=lock['checkpoint_sha256'], state_hash=lock['state_hash'],
                               model_state_sha256=test['identity']['model_state_sha256']))
        final = read(run / 'final_metrics.json')
        train = final.get('train')
        train_rows.append(dict(model=arm, selected_update=lock['selected_update'],
                               train_top1_pct=100. * train['macro']['macro_top1'] if train else None,
                               val_top1_pct=100. * val['macro']['macro_top1'],
                               train_prediction_status='saved selected-model metrics' if train else 'MISSING'))
    data = {model: {split: {problem: predictions[split, problem] for problem in PROBLEMS}
                           for split in ('VAL', 'TEST')} for model, predictions in packed.items()}
    target_sets = dict(zero_near_zero=[(r['problem'], r['method']) for r in frozen['targets']],
                       strict_zero=[(r['problem'], r['method']) for r in frozen['targets'] if r['category'] == 'zero'])
    pairs, target_methods, targets, methods = [], [], [], []
    comparisons = [(arm, 'R84A') for arm in ARMS] + [('B_gradient_balance', 'A_ovr_control')]
    for model, baseline in comparisons:
        for split in ('VAL', 'TEST'):
            pairs.extend(api.paired_comparison(data[baseline][split], data[model][split], model, baseline, split))
    for model in ('R84A', *ARMS):
        for split in ('VAL', 'TEST'):
            for definition, keys in target_sets.items():
                rows = api.fixed_target_recovery(data['R84A'][split], data[model][split], keys, model, 'R84A', split)
                for row in rows:
                    row['target_definition'] = definition + ' from published R84A locked VAL list'
                target_methods.extend(rows)
                targets.append(summarize_targets(rows, model, 'R84A', split, definition))
            keys = [(problem, solver) for problem in PROBLEMS for solver in data[model][split][problem].pool]
            rows = api.fixed_target_recovery(data['R84A'][split], data[model][split], keys, model, 'R84A', split)
            for row in rows:
                row['target_definition'] = 'all legal methods; descriptive only'
                row['frozen_zero_near_target'] = (row['problem'], row['solver']) in target_sets['zero_near_zero']
            methods.extend(rows)
    overall = {(row['model'], row['split']): row for row in metric_rows if row['problem'] == 'ALL'}
    gates = []
    for model, baseline in comparisons:
        row = dict(model=model, baseline=baseline, acceptance_split='VAL', test_used_for_acceptance=False,
                   passed_VAL_screen=passes(overall[model, 'VAL'], overall[baseline, 'VAL']))
        for split in ('VAL', 'TEST'):
            for key in METRICS:
                row[split + '_' + key + '_delta'] = overall[model, split][key] - overall[baseline, split][key]
        gates.append(row)
    return dict(metrics=metric_rows, pairs=pairs, target_methods=target_methods, targets=targets,
                methods=methods, gates=gates, provenance=provenance, train=train_rows,
                data_root=old_config['data_root'])


def make_pdf(output, result):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'pdf.fonttype': 42})
    def fmt(value, places=4):
        return 'MISSING' if value is None else f'{value:.{places}f}'
    def label(model):
        return 'R85' + model[0] if model in ARMS else model
    def table(pdf, title, headers, rows, notes):
        fig = plt.figure(figsize=(15, 9))
        fig.text(.045, .94, title, fontsize=18, weight='bold', color='#104f55')
        axis = fig.add_axes([.045, .23, .91, .62]); axis.axis('off')
        artist = axis.table(cellText=rows, colLabels=headers, bbox=[0, 0, 1, 1], cellLoc='left')
        artist.auto_set_font_size(False); artist.set_fontsize(9.5)
        for (row, _), cell in artist.get_celld().items():
            cell.set_edgecolor('#d9e2e3')
            cell.set_facecolor('#d6eaeb' if row == 0 else '#f3f7f7' if row % 2 else 'white')
        for i, note in enumerate(notes): fig.text(.045, .16 - .032*i, note, fontsize=9)
        pdf.savefig(fig); plt.close(fig)
    destination = output / 'COMPARISON.pdf'
    temporary = output / f'.comparison.{os.getpid()}.pdf'
    try:
        with PdfPages(temporary) as pdf:
            rows = [r for r in result['metrics'] if r['problem'] == 'ALL']
            table(pdf, 'R85 | Locked full18 results', ['Split', 'Model', 'Top1 (%)', 'mean_cost', 'Actual regret (%)'],
                  [[r['split'], label(r['model']), fmt(r['top1_pct']), fmt(r['mean_cost'], 9), fmt(r['actual_regret_pct'], 6)] for r in rows],
                  ['Same NEW90k release and native/FP64 winner labels. SBS fitted on TRAIN; ALL = equal18-task mean.',
                   'Each model has one previously locked VAL checkpoint. TEST never changes a checkpoint or rule.',
                   'The long-term research goal remains active even if an empirical VAL screen passes.'])
            table(pdf, 'R85 | Validation screens and matched EQL control',
                  ['Model', 'Baseline', 'VAL dTop1 pp', 'VAL dCost', 'VAL dRegret pp', 'VAL screen', 'TEST dTop1 pp'],
                  [[label(r['model']), label(r['baseline']), fmt(r['VAL_top1_pct_delta'], 6), fmt(r['VAL_mean_cost_delta'], 9),
                    fmt(r['VAL_actual_regret_pct_delta'], 6), str(r['passed_VAL_screen']), fmt(r['TEST_top1_pct_delta'], 6)] for r in result['gates']],
                  ['Screen: unrounded VAL Top1 strictly higher; mean_cost and actual regret not higher.',
                   'B versus R85A separates EQL from the common OVR classification change.',
                   'A small screen pass is evidence for review, not completion or deployment selection.'])
            table(pdf, 'R85 | Published R84A zero / near-zero targets',
                  ['Split', 'Model', 'Winners', 'Recall %', 'Rescued / lost', 'New false picks', 'New-FP regret pp'],
                  [[r['split'], label(r['model']), str(r['winner_count']), fmt(r['recall_pct'], 3),
                    f"{r['rescued']} / {r['lost']}", str(r['new_false_pick_count']),
                    fmt(r['new_false_pick_regret_contribution_pp'], 6)] for r in result['targets'] if r['target_definition'] == 'zero_near_zero'],
                  ['Frozen from R84A locked VAL: 36 groups / 4,625 winners, choice share <=0.1%.',
                   'Posthoc mechanism targets, not a training rule. Strict-zero results and precision are in target_summary.csv.',
                   'One false choice can shrink the zero list; true rescue and new false picks are reported together.'])
            table(pdf, 'R85 | Same-instance corrections and harms',
                  ['Split', 'Model / baseline', 'Corrected', 'Harmed', 'Both wrong cheaper / dearer', 'Cost delta', 'Regret delta pp'],
                  [[r['split'], label(r['model'])+' / '+label(r['baseline']), str(r['corrected']), str(r['harmed']),
                    f"{r['both_wrong_cheaper']} / {r['both_wrong_dearer']}", fmt(r['mean_cost_delta'], 9),
                    fmt(r['actual_regret_delta_pp'], 6)] for r in result['pairs'] if r['problem'] == 'ALL'],
                  ['Every pair uses identical saved instances, candidates, FP64 costs and native winners.',
                   'Native-winner errors can have zero extra cost in exact ties. Negative cost/regret delta is better.',
                   'Per-task results, all-method recall/precision, target effects and full checkpoint hashes accompany this PDF.'])
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def report(root=ROOT, launch_root=LAUNCH, historical=HISTORICAL):
    root = Path(root).resolve(); output = root / 'comparison_artifacts'
    snapshot = status(root, launch_root)
    if not snapshot['all_arms_ready'] or any(e['arm'] != 'comparison' for e in snapshot['errors']):
        raise RuntimeError('Wait for all four training, unique lock, full TEST, LIB and per-arm PDF completions')
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'report.lock').open('a') as claim:
        fcntl.flock(claim, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = derive(root, freeze_targets(root, historical), historical)
        for filename, key in (('metrics_by_problem.csv', 'metrics'), ('paired_changes.csv', 'pairs'),
                              ('method_precision_recall.csv', 'methods'), ('target_summary.csv', 'targets'),
                              ('target_recovery_by_method.csv', 'target_methods'), ('validation_gates.csv', 'gates'),
                              ('selected_train_summary.csv', 'train')):
            receipts.write_csv(output / filename, result[key])
        pdf = make_pdf(output, result)
        passed = [r['model'] for r in result['gates'] if r['baseline'] == 'R84A' and r['passed_VAL_screen']]
        next_action = 'review_validation_signal_and_target_recall' if passed else 'research_handoff_needed'
        protocol = dict(experiment='R85', complete=True, completed_utc=receipts.utc(), data_root=result['data_root'],
                        winner_rule='saved native/FP64 ind; never FP32 argmin', macro_aggregation='equal18-task weights',
                        provenance=result['provenance'], validation_screens=result['gates'],
                        passed_empirical_VAL_screen=passed, test_used_for_acceptance=False,
                        target_list_posthoc_descriptive=True, training_rule_changed=False,
                        goal_status='active_unresolved', goal_complete=False, next_action=next_action,
                        cpu_only=True, training_started=False, inference_started=False, automatic_retry=False,
                        comparison_pdf=str(pdf))
        lines = ['# R85 comparison', '', 'Goal remains active/unresolved. Next action: `'+next_action+'`.', '',
                 '| Split | Model | Top1 (%) | mean_cost | actual regret (%) |', '|---|---|---:|---:|---:|']
        lines += [f"| {r['split']} | {r['model']} | {r['top1_pct']:.6f} | {r['mean_cost']:.9f} | {r['actual_regret_pct']:.6f} |"
                  for r in result['metrics'] if r['problem'] == 'ALL']
        lines += ['', 'VAL screens use unrounded Top1 up, cost/regret not worse versus R84A; B versus R85A is separately reported.',
                  'TEST belongs to the same previously locked checkpoint, never checkpoint/rule selection.',
                  'Targets reuse the published R84A locked-VAL zero/near-zero list (36 groups, 4625 winners), descriptively only.',
                  'Recall, precision, rescues, lost winners, new false picks and net regret are in the CSVs.',
                  'Native FP64 winners; equal18-task macros; TRAIN-fitted SBS. Prior TEST results have informed research.',
                  'A small pass does not complete the goal. No automatic retry, training, GPU inference or notification.',
                  'PDF: [COMPARISON.pdf](COMPARISON.pdf).', '']
        (output / 'RESULTS.md').write_text('\n'.join(lines), encoding='utf-8')
        save(output / 'protocol.json', protocol)
        (output / 'error.json').unlink(missing_ok=True)
        status(root, launch_root)
        return protocol


def watch(root=ROOT, launch_root=LAUNCH, historical=HISTORICAL, interval=60, max_polls=None):
    root = Path(root).resolve(); root.mkdir(parents=True, exist_ok=True)
    with (root / 'goal_watch.lock').open('a') as claim:
        fcntl.flock(claim, fcntl.LOCK_EX | fcntl.LOCK_NB)
        receipt = dict(started_utc=receipts.utc(), pid=os.getpid(), interval_seconds=interval,
                       state='watching', cpu_only=True, automatic_retry=False, autonomous_training=False)
        save(root / 'goal_watch_receipt.json', receipt)
        polls = 0
        while True:
            try:
                snapshot = status(root, launch_root)
                if snapshot['errors']:
                    receipt.update(state='stopped_on_error', completed_utc=receipts.utc())
                    save(root / 'goal_watch_receipt.json', receipt)
                    return 1
                if snapshot['comparison_complete']:
                    receipt.update(state='comparison_complete_goal_active', completed_utc=receipts.utc())
                    save(root / 'goal_watch_receipt.json', receipt)
                    return 0
                if not (root / 'comparison_artifacts/target_list.json').is_file():
                    freeze_targets(root, historical)
                if snapshot['all_arms_ready']:
                    report(root, launch_root, historical)
                    continue
            except BaseException as error:
                save(root / 'comparison_artifacts/error.json', dict(updated_utc=receipts.utc(),
                     error=f'{type(error).__name__}: {error}', traceback=traceback.format_exc(), automatic_retry=False,
                     action='Inspect and repair only the failed monitor/report stage manually; retain all experiments.'))
                status(root, launch_root)
                receipt.update(state='stopped_on_error', completed_utc=receipts.utc())
                save(root / 'goal_watch_receipt.json', receipt)
                return 1
            polls += 1
            if max_polls is not None and polls >= max_polls:
                return 0
            time.sleep(interval)


def main(argv=None):
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('command', choices=('status', 'watch', 'report'))
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--launch-root', type=Path, default=LAUNCH)
    parser.add_argument('--historical-root', type=Path, default=HISTORICAL)
    parser.add_argument('--interval', type=int, default=60)
    parser.add_argument('--max-polls', type=int)
    args = parser.parse_args(argv)
    if args.interval < 1 or (args.max_polls is not None and args.max_polls < 1):
        parser.error('Intervals and bounded poll counts must be positive')
    if args.command == 'watch':
        return watch(args.root, args.launch_root, args.historical_root, args.interval, args.max_polls)
    if args.command == 'report':
        report(args.root, args.launch_root, args.historical_root)
    else:
        import json
        print(json.dumps(status(args.root, args.launch_root), indent=2))
    return 0


if __name__ == '__main__':
    sys.exit(main())

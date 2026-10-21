"""R87 locked-model PDF/CSV reports; reuse saved R84 NSS4500 predictions."""

import argparse
import fcntl
import json
import os
import traceback
from pathlib import Path

import numpy as np

from code.unified_selector.registry import PROBLEMS
from . import r84_reporting as shared
from . import r85_reporting as previous
from .r76_experiment import read_json, utc, write_json
from .r87_experiment import ARMS, DEFAULT_ROOT, MAX_UPDATES, WINNER_RULE, zero_summary


common, LIBS = shared.common, shared.LIBS
FACTORY = 'code.V4.r87_experiment'
BUDGETS = dict.fromkeys(ARMS, MAX_UPDATES)
DEFAULT_R84, DEFAULT_NSS = previous.DEFAULT_R84, previous.DEFAULT_NSS


def locked_inputs(root, arm):
    return shared.locked_inputs(root, arm, experiment='R87', arms=BUDGETS)


def lib_stage(args):
    root = args.root.resolve()
    run, config, lock, test, _ = locked_inputs(root, args.arm)
    with (run / 'lib_stage.lock').open('a') as claim:
        fcntl.flock(claim, fcntl.LOCK_EX | fcntl.LOCK_NB)

        def phase(name, **fields):
            write_json(run / 'lib_stage_status.json', dict(experiment='R87', arm=args.arm,
                phase=name, updated_utc=utc(), **fields))

        reference = previous.validate_nss_reference(config, args.nss_root)
        results = common.r80_lib(root, run, config, lock, args.device, args.lib_batch_size,
            phase, test['identity']['model_state_sha256'], factory_module=FACTORY)
        write_json(run / 'lib_stage_complete.json', dict(experiment='R87', complete=True,
            updated_utc=utc(), checkpoint_sha256=lock['checkpoint_sha256'],
            nss_reference=reference, datasets=list(results)))
        phase('complete', checkpoint_sha256=lock['checkpoint_sha256'])
        return results


def selected_predictions(run, config, lock, val, test):
    train, train_path = previous.selected_train(run, config, lock)
    if train['identity']['model_state_sha256'] != test['identity']['model_state_sha256']:
        raise ValueError('R87 TRAIN and held-out diagnostics must use the same locked state')
    predictions, observations, summaries = {}, [], []
    paths = dict(TRAIN=train_path, VAL=common.selected_val_path(run, lock),
        TEST=run / 'evaluation/test/selected_best/rank000.npz')
    results = dict(TRAIN=train, VAL=val, TEST=test)
    for split, path in paths.items():
        packed, rows = previous.read_split(path, config, split)
        predictions[split] = packed
        observations.extend(dict(row, model='R87/' + config['arm']) for row in rows)
        summary = zero_summary(rows)
        for problem, counts in [*summary['per_problem'].items(), ('ALL', summary['all'])]:
            summaries.append(dict(model='R87/' + config['arm'], split=split, problem=problem, **counts))
    return predictions, observations, summaries, results, paths


def load_targets(config, historical_config, historical_lock):
    fallback = DEFAULT_R84 / 'comparison_artifacts/target_list.json'
    path = Path(config['main_protocol_lock']['record'].get('target_list', fallback))
    record = read_json(path)
    identity = record['identity']
    if (identity.get('baseline_checkpoint_sha256') != historical_lock['checkpoint_sha256']
            or Path(identity['data_root']).resolve() != Path(config['data_root']).resolve()
            or Path(historical_config['data_root']).resolve() != Path(config['data_root']).resolve()
            or historical_config['rows_per_problem'] != config['rows_per_problem']):
        raise ValueError('Frozen R84 target list is not bound to this exact data/reference model')
    targets = [(row['problem'], row['method']) for row in record['targets']]
    if len(set(targets)) != len(targets):
        raise ValueError('The frozen target list must not duplicate problem/method pairs')
    return targets, dict(path=str(path.resolve()), identity=identity,
        categories={row['problem'] + '/' + row['method']: row['category'] for row in record['targets']})


def recovery_summary(rows, model, baseline):
    summaries = []
    for split in ('TRAIN', 'VAL', 'TEST'):
        current = [row for row in rows if row['split'] == split]
        if not current:
            continue
        result = dict(model=model, baseline=baseline, split=split, target_groups=len(current))
        for name in ('winner_count', 'baseline_missed', 'correct_count', 'selected_count',
                     'rescued', 'lost', 'false_pick_count', 'new_false_pick_count'):
            result[name] = sum(row[name] for row in current)
        for name in ('new_false_pick_regret_contribution_pp', 'target_regret_delta_pp'):
            result[name] = sum(row[name] for row in current) / len(PROBLEMS)
        result.update(recall_pct=100. * result['correct_count'] / result['winner_count'] if result['winner_count'] else None,
            precision_pct=100. * result['correct_count'] / result['selected_count'] if result['selected_count'] else None,
            rescued_of_baseline_misses_pct=100. * result['rescued'] / result['baseline_missed'] if result['baseline_missed'] else None)
        summaries.append(result)
    return summaries


def comparison_tables(old, new, targets, target_identity, model, baseline):
    pairs, recovery = [], []
    for split in ('TRAIN', 'VAL', 'TEST'):
        pairs.extend(previous.paired_comparison(old[split], new[split], model, baseline, split))
        rows = previous.fixed_target_recovery(old[split], new[split], targets, model, baseline, split)
        for row in rows:
            row['target_definition'] = target_identity['identity']['rule']
            row['target_category'] = target_identity['categories'][row['problem'] + '/' + row['solver']]
        recovery.extend(rows)
    return pairs, recovery, recovery_summary(recovery, model, baseline)


def historical_tables(run, config, lock, val, test, historical_root):
    packed, methods, summaries, results, paths = selected_predictions(run, config, lock, val, test)
    old_run, old_config, old_lock, old_test, old_val = shared.locked_inputs(
        historical_root.parent, historical_root.name)
    old, _, _, _, _ = selected_predictions(old_run, old_config, old_lock, old_val, old_test)
    targets, target_identity = load_targets(config, old_config, old_lock)
    pairs, recovery, totals = comparison_tables(old, packed, targets, target_identity,
        'R87/' + config['arm'], 'R84A historical')
    return dict(predictions=packed, methods=methods, summaries=summaries, pairs=pairs,
        recovery=recovery, recovery_summary=totals, target_identity=target_identity,
        historical=dict(model='R84A', checkpoint_sha256=old_lock['checkpoint_sha256'],
            run=str(old_run), same_data=True, retrained=False), results=results, paths=paths)


def diagnostics_pdf(output, config, lock, tables):
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    helper = common.reporting_helpers()
    model = 'R87/' + config['arm']
    path = output / 'target_recovery_and_train.pdf'
    temporary = path.with_name(f'.{path.stem}.{os.getpid()}.pdf')
    try:
        with PdfPages(temporary) as pdf:
            fig = common._page(helper, model + ' | Fixed target recovery',
                'Frozen R84A VAL zero/near-zero target list; TRAIN/VAL/TEST from the same selected checkpoint.', model, lock, 1)
            helper.add_table(fig, ['Split', 'Winners', 'Correct', 'Rescued', 'Lost', 'New false picks', 'Recall (%)'],
                [[row['split'], str(row['winner_count']), str(row['correct_count']), str(row['rescued']),
                  str(row['lost']), str(row['new_false_pick_count']),
                  '-' if row['recall_pct'] is None else f"{row['recall_pct']:.4f}"]
                 for row in tables['recovery_summary']], [.05, .61, .9, .22], font=11)
            helper.add_table(fig, ['Split', 'Top1 delta pp', 'Corrected', 'Harmed', 'Cost delta', 'Regret delta pp'],
                [[row['split'], f"{row['top1_delta_pp']:.6f}", str(row['corrected']), str(row['harmed']),
                  f"{row['mean_cost_delta']:.9f}", f"{row['actual_regret_delta_pp']:.6f}"]
                 for row in tables['pairs'] if row['problem'] == 'ALL'], [.05, .31, .9, .2], font=10)
            fig.text(.05, .24, 'Historical R84A is a same-data reference, not a newly retrained no-residual control.', fontsize=10)
            fig.text(.05, .19, 'Fewer zero groups alone do not prove recovery: report recall, false picks and net regret.', fontsize=10)
            fig.text(.05, .14, 'Matched A/B comparison is generated only after both unique locks have completed.', fontsize=10)
            pdf.savefig(fig)
            plt.close(fig)
            for page, split in enumerate(('TRAIN', 'VAL', 'TEST'), 2):
                fig = common._page(helper, model + ' | ' + split + ' selection diagnostics',
                    'Native FP64 labels; zero winner mass differs from all missed winners.', model, lock, page)
                rows = [row for row in tables['summaries'] if row['split'] == split]
                helper.add_table(fig, ['Problem', 'Zero groups', 'Zero winners', 'Near-zero winners', 'All misses'],
                    [[row['problem'], str(row['zero_with_winner_groups']), str(row['zero_winner_mass']),
                      str(row['near_zero_winner_mass']), str(row['total_missed_winner_count'])] for row in rows],
                    [.05, .2, .9, .64], font=11)
                pdf.savefig(fig)
                plt.close(fig)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return str(path)


def train_metric_rows(config, lock, tables):
    rows, result = [], tables['results']['TRAIN']
    for problem in PROBLEMS:
        rows.append(common.metric_row(problem, 'TRAIN', tables['predictions']['TRAIN'][problem],
            result['per_problem'][problem], config['baselines'][problem], config, lock, tables['paths']['TRAIN']))
    return [common.macro_row('TRAIN', rows, result), *rows]


def report(args):
    run, config, lock, test, val = locked_inputs(args.root, args.arm)
    with (run / 'postprocess.lock').open('a') as claim:
        fcntl.flock(claim, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            write_json(run / 'postprocess_status.json', dict(experiment='R87', phase='loading_saved_predictions', updated_utc=utc()))
            reference = previous.validate_nss_reference(config, args.nss_root)
            predictions, metrics, methods = shared.load_saved_splits(run, config, lock, test, val)
            shared.saved_libs(run, config, lock, test, predictions, metrics, methods)
            comparisons, alignment = shared.nss_comparisons(config, lock, predictions, args.nss_root)
            tables = historical_tables(run, config, lock, val, test, args.historical_root)
            result = shared.build_reports(run, config, lock, test, val, predictions, metrics, methods, comparisons, alignment)
            output = run / 'full_model_statistics'
            for name, rows in (('method_selection_train_val_test.csv', tables['methods']),
                    ('zero_selection_summary.csv', tables['summaries']),
                    ('paired_changes_vs_R84A.csv', tables['pairs']),
                    ('target_recovery.csv', tables['recovery']),
                    ('target_recovery_summary.csv', tables['recovery_summary']),
                    ('train_metrics_by_problem.csv', train_metric_rows(config, lock, tables))):
                common.write_csv(output / name, rows)
            result.update(target_recovery_pdf=diagnostics_pdf(output, config, lock, tables),
                historical_reference=tables['historical'], nss_reference=reference)
            provenance = read_json(output / 'protocol.json')
            provenance.update(target_identity=tables['target_identity'], historical_reference=tables['historical'],
                nss_reference=reference, residual=config['residual'], objective=config['objective'])
            write_json(output / 'protocol.json', provenance)
            write_json(run / 'postprocess_status.json', dict(experiment='R87', phase='complete', completed_utc=utc(), **result))
            return result
        except BaseException as error:
            write_json(run / 'postprocess_status.json', dict(experiment='R87', phase='postprocess_failed',
                updated_utc=utc(), error=str(error), traceback=traceback.format_exc(),
                retry='report only; retain all training/TEST/LIB records'))
            raise


def compare(args):
    """Read two locked runs; emit paired results without choosing a new model."""
    selected, locks, configs = {}, {}, {}
    for arm in ARMS:
        run, config, lock, test, val = locked_inputs(args.root, arm)
        selected[arm], _, _, results, _ = selected_predictions(run, config, lock, val, test)
        locks[arm], configs[arm] = lock, config
    a, b = ARMS
    if any(configs[a][name] != configs[b][name] for name in
            ('data_root', 'rows_per_problem', 'baselines', 'candidate_pools', 'common_trainable_sha256')):
        raise ValueError('R87 matched runs must share data, pools, SBS and common fresh initialization')
    old_run, old_config, old_lock, _, _ = shared.locked_inputs(args.historical_root.parent, args.historical_root.name)
    targets, target_identity = load_targets(configs[b], old_config, old_lock)
    pairs, recovery, totals = comparison_tables(selected[a], selected[b], targets,
        target_identity, 'R87/' + b, 'R87/' + a)
    output = args.root / 'comparison_artifacts'
    output.mkdir(parents=True, exist_ok=True)
    for name, rows in (('paired_changes_B_vs_A.csv', pairs), ('target_recovery_B_vs_A.csv', recovery),
                       ('target_recovery_summary_B_vs_A.csv', totals)):
        common.write_csv(output / name, rows)
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    helper = common.reporting_helpers()
    helper.setup_style()
    path = output / 'COMPARISON.pdf'
    with PdfPages(path) as pdf:
        fig = common._page(helper, 'R87 | Matched private minus shared residual',
            'One independently VAL-selected checkpoint per arm; saved TRAIN/VAL/TEST predictions only.',
            'R87/B vs A', locks[b], 1)
        helper.add_table(fig, ['Split', 'Top1 delta pp', 'Corrected', 'Harmed', 'Cost delta', 'Regret delta pp'],
            [[row['split'], f"{row['top1_delta_pp']:.6f}", str(row['corrected']), str(row['harmed']),
              f"{row['mean_cost_delta']:.9f}", f"{row['actual_regret_delta_pp']:.6f}"]
             for row in pairs if row['problem'] == 'ALL'], [.05, .56, .9, .24], font=11)
        helper.add_table(fig, ['Split', 'Target correct', 'Rescued', 'Lost', 'New false picks', 'Recall (%)'],
            [[row['split'], str(row['correct_count']), str(row['rescued']), str(row['lost']),
              str(row['new_false_pick_count']), '-' if row['recall_pct'] is None else f"{row['recall_pct']:.4f}"]
             for row in totals], [.05, .25, .9, .23], font=11)
        fig.text(.05, .17, 'All costs use the same legal FP64 labels. Macro deltas weight each of18 tasks equally.', fontsize=10)
        fig.text(.05, .12, 'This report does not select a new checkpoint, threshold, inference rule or per-task model.', fontsize=10)
        pdf.savefig(fig)
        plt.close(fig)
    result = dict(experiment='R87', complete=True, checkpoints=locks, target_identity=target_identity,
        comparison_pdf=str(path), training=False, inference=False, checkpoint_reselection=False)
    write_json(output / 'protocol.json', result)
    return result


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('command', choices=('lib', 'report', 'status', 'compare'))
    result.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    result.add_argument('--arm', choices=ARMS)
    result.add_argument('--device', default='cpu')
    result.add_argument('--lib-batch-size', type=int, choices=(1, 2), default=2)
    result.add_argument('--nss-root', type=Path, default=DEFAULT_NSS)
    result.add_argument('--historical-root', type=Path, default=DEFAULT_R84 / 'A_baseline')
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    for name in ('root', 'nss_root', 'historical_root'):
        setattr(args, name, getattr(args, name).resolve())
    if args.command == 'compare':
        return compare(args)
    if args.arm is None:
        raise ValueError('--arm is required for lib, report and status')
    if args.command == 'status':
        path = args.root / args.arm / 'postprocess_status.json'
        result = read_json(path) if path.exists() else dict(experiment='R87', phase='not_started')
        print(json.dumps(result, indent=2))
        return result
    return lib_stage(args) if args.command == 'lib' else report(args)


if __name__ == '__main__':
    main()

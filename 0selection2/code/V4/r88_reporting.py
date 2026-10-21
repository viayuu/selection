"""R88 same-checkpoint full18/NSS/LIB PDFs and fixed-target paired recovery."""

import argparse
import fcntl
import json
import os
import traceback
from pathlib import Path

from . import r87_reporting as helpers
from .r76_experiment import read_json, utc, write_json
from .r88_experiment import ARMS, DEFAULT_ROOT, MAX_UPDATES


shared, previous, common = helpers.shared, helpers.previous, helpers.common
LIBS, DEFAULT_R84, DEFAULT_NSS = helpers.LIBS, helpers.DEFAULT_R84, helpers.DEFAULT_NSS
FACTORY, BUDGETS = 'code.V4.r88_experiment', dict.fromkeys(ARMS, MAX_UPDATES)


def locked_inputs(root, arm):
    return shared.locked_inputs(root, arm, experiment='R88', arms=BUDGETS)


def lib_stage(args):
    run, config, lock, test, _ = locked_inputs(args.root, args.arm)
    with (run / 'lib_stage.lock').open('a') as claim:
        fcntl.flock(claim, fcntl.LOCK_EX | fcntl.LOCK_NB)

        def phase(name, **fields):
            write_json(run / 'lib_stage_status.json', dict(experiment='R88', arm=args.arm,
                phase=name, updated_utc=utc(), **fields))

        reference = previous.validate_nss_reference(config, args.nss_root)
        results = common.r80_lib(args.root, run, config, lock, args.device,
            args.lib_batch_size, phase, test['identity']['model_state_sha256'], factory_module=FACTORY)
        write_json(run / 'lib_stage_complete.json', dict(experiment='R88', complete=True,
            updated_utc=utc(), checkpoint_sha256=lock['checkpoint_sha256'],
            nss_reference=reference, datasets=list(results)))
        phase('complete', checkpoint_sha256=lock['checkpoint_sha256'])
        return results


def selected_predictions(run, config, lock, val, test):
    packed, methods, summaries, results, paths = helpers.selected_predictions(run, config, lock, val, test)
    model = config['experiment'] + '/' + config['arm']
    methods = [dict(row, model=model) for row in methods]
    summaries = [dict(row, model=model) for row in summaries]
    return packed, methods, summaries, results, paths


def historical_tables(run, config, lock, val, test, historical_root):
    packed, methods, summaries, results, paths = selected_predictions(run, config, lock, val, test)
    old_run, old_config, old_lock, old_test, old_val = shared.locked_inputs(
        historical_root.parent, historical_root.name)
    old, _, _, _, _ = selected_predictions(old_run, old_config, old_lock, old_val, old_test)
    targets, target_identity = helpers.load_targets(config, old_config, old_lock)
    pairs, recovery, totals = helpers.comparison_tables(old, packed, targets, target_identity,
        'R88/' + config['arm'], 'R84A historical')
    return dict(predictions=packed, methods=methods, summaries=summaries, results=results,
        paths=paths, pairs=pairs, recovery=recovery, recovery_summary=totals,
        target_identity=target_identity,
        historical=dict(model='R84A', checkpoint_sha256=old_lock['checkpoint_sha256'],
            run=str(old_run), same_data=True, retrained=False))


def diagnostics_pdf(output, config, lock, tables):
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    helper, model = common.reporting_helpers(), 'R88/' + config['arm']
    path = output / 'target_recovery_and_train.pdf'
    temporary = path.with_name(f'.{path.stem}.{os.getpid()}.pdf')
    try:
        with PdfPages(temporary) as pdf:
            for page, split in enumerate(('TRAIN', 'VAL', 'TEST'), 1):
                fig = common._page(helper, model + ' | ' + split + ': execution-role input comparison',
                    'Same selected checkpoint; original score head/loss. Frozen R84 zero/near-zero targets.', model, lock, page)
                rows = [row for row in tables['summaries'] if row['split'] == split]
                helper.add_table(fig, ['Problem', 'Zero groups', 'Zero winner mass', 'Near-zero mass', 'All misses'],
                    [[row['problem'], str(row['zero_with_winner_groups']), str(row['zero_winner_mass']),
                      str(row['near_zero_winner_mass']), str(row['total_missed_winner_count'])] for row in rows],
                    [.05, .29, .9, .54], font=10)
                target = [row for row in tables['recovery_summary'] if row['split'] == split]
                helper.add_table(fig, ['Target winners', 'Correct', 'Rescued', 'Lost', 'New false picks', 'Recall (%)'],
                    [[str(row['winner_count']), str(row['correct_count']), str(row['rescued']),
                      str(row['lost']), str(row['new_false_pick_count']),
                      '-' if row['recall_pct'] is None else f"{row['recall_pct']:.4f}"] for row in target],
                    [.05, .16, .9, .08], font=10)
                fig.text(.05, .11, 'TSP input slot3: canonical anchor vs physical node0. The remaining17 inputs are unchanged.', fontsize=10)
                fig.text(.05, .07, 'Historical R84A is a same-data reference. A/B is the matched comparison; TEST does not select a rule.', fontsize=9.5)
                pdf.savefig(fig)
                plt.close(fig)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return str(path)


def report(args):
    run, config, lock, test, val = locked_inputs(args.root, args.arm)
    with (run / 'postprocess.lock').open('a') as claim:
        fcntl.flock(claim, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            write_json(run / 'postprocess_status.json', dict(experiment='R88', phase='loading_saved_predictions', updated_utc=utc()))
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
                    ('train_metrics_by_problem.csv', helpers.train_metric_rows(config, lock, tables))):
                common.write_csv(output / name, rows)
            result.update(target_recovery_pdf=diagnostics_pdf(output, config, lock, tables),
                historical_reference=tables['historical'], nss_reference=reference)
            provenance = read_json(output / 'protocol.json')
            provenance.update(target_identity=tables['target_identity'], historical_reference=tables['historical'],
                nss_reference=reference, input_roles=config['input_roles'], objective=config['objective'])
            write_json(output / 'protocol.json', provenance)
            write_json(run / 'postprocess_status.json', dict(experiment='R88', phase='complete', completed_utc=utc(), **result))
            return result
        except BaseException as error:
            write_json(run / 'postprocess_status.json', dict(experiment='R88', phase='postprocess_failed',
                updated_utc=utc(), error=str(error), traceback=traceback.format_exc(),
                retry='report only; retain all training/TEST/LIB records'))
            raise


def compare(args):
    selected, locks, configs = {}, {}, {}
    for arm in ARMS:
        run, config, lock, test, val = locked_inputs(args.root, arm)
        selected[arm], _, _, _, _ = selected_predictions(run, config, lock, val, test)
        locks[arm], configs[arm] = lock, config
    a, b = ARMS
    if any(configs[a][name] != configs[b][name] for name in
            ('data_root', 'rows_per_problem', 'baselines', 'candidate_pools', 'common_trainable_sha256')):
        raise ValueError('Matched R88 runs must share data, labels, pools, SBS and common initialization')
    _, old_config, old_lock, _, _ = shared.locked_inputs(args.historical_root.parent, args.historical_root.name)
    targets, target_identity = helpers.load_targets(configs[b], old_config, old_lock)
    pairs, recovery, totals = helpers.comparison_tables(selected[a], selected[b], targets,
        target_identity, 'R88/' + b, 'R88/' + a)
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
        fig = common._page(helper, 'R88 | Execution anchor minus canonical anchor',
            'Matched full18 training; unique VAL checkpoint per arm. No additional inference or checkpoint selection.',
            'R88/B vs A', locks[b], 1)
        helper.add_table(fig, ['Split', 'Top1 delta pp', 'Corrected', 'Harmed', 'Cost delta', 'Regret delta pp'],
            [[row['split'], f"{row['top1_delta_pp']:.6f}", str(row['corrected']), str(row['harmed']),
              f"{row['mean_cost_delta']:.9f}", f"{row['actual_regret_delta_pp']:.6f}"]
             for row in pairs if row['problem'] == 'ALL'], [.05, .56, .9, .24], font=11)
        helper.add_table(fig, ['Split', 'Target correct', 'Rescued', 'Lost', 'New false picks', 'Recall (%)'],
            [[row['split'], str(row['correct_count']), str(row['rescued']), str(row['lost']),
              str(row['new_false_pick_count']), '-' if row['recall_pct'] is None else f"{row['recall_pct']:.4f}"]
             for row in totals], [.05, .25, .9, .23], font=11)
        fig.text(.05, .17, 'Same FP64 labels and legal pools;18 tasks have equal macro weight. Original MLP and loss retained.', fontsize=10)
        fig.text(.05, .12, 'The only controlled change is TSP anchor identity in input slot3; this is not a global plateau explanation.', fontsize=10)
        pdf.savefig(fig)
        plt.close(fig)
    result = dict(experiment='R88', complete=True, checkpoints=locks, target_identity=target_identity,
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
        result = read_json(path) if path.exists() else dict(experiment='R88', phase='not_started')
        print(json.dumps(result, indent=2))
        return result
    return lib_stage(args) if args.command == 'lib' else report(args)


if __name__ == '__main__':
    main()

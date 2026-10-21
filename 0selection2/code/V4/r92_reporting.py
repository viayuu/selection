"""R92 unique-lock full18/LIB/NSS PDFs and frozen missed-winner diagnostics."""

import argparse
import fcntl
import json
import os
import traceback
from pathlib import Path

from . import r87_reporting as helpers
from .r76_experiment import file_hash, read_json, utc, write_json
from .r92_experiment import ARMS, BUDGETS, DEFAULT_ROOT, validate_checkpoint_config


shared, previous, common = helpers.shared, helpers.previous, helpers.common
DEFAULT_NSS, DEFAULT_R84 = helpers.DEFAULT_NSS, helpers.DEFAULT_R84
DEFAULT_R89B = DEFAULT_ROOT.parent / 'R89_optimizer_geometry/B_muon_hybrid'
FACTORY = 'code.V4.r92_experiment'
BOUNDARY = ('Historical same-data R84A/R89B references. Tree fitting is not an equal-compute neural control; '
            'all18 task trees use one common VAL-selected tree count. No TEST selection or per-task best assembly.')


def locked_inputs(root, arm):
    result = shared.locked_inputs(root, arm, experiment='R92', arms=BUDGETS)
    validate_checkpoint_config(result[1])
    return result


def lib_stage(args):
    run, config, lock, test, _ = locked_inputs(args.root, args.arm)
    with (run / 'lib_stage.lock').open('a') as claim:
        fcntl.flock(claim, fcntl.LOCK_EX | fcntl.LOCK_NB)

        def phase(name, **fields):
            write_json(run / 'lib_stage_status.json', dict(experiment='R92', arm=args.arm,
                phase=name, updated_utc=utc(), **fields))

        reference = previous.validate_nss_reference(config, args.nss_root)
        result = common.r80_lib(args.root, run, config, lock, args.device,
            args.lib_batch_size, phase, test['identity']['model_state_sha256'], factory_module=FACTORY)
        write_json(run / 'lib_stage_complete.json', dict(experiment='R92', complete=True,
            updated_utc=utc(), checkpoint_sha256=lock['checkpoint_sha256'],
            nss_reference=reference, datasets=list(result)))
        phase('complete', checkpoint_sha256=lock['checkpoint_sha256'])
        return result


def same_data(left, right):
    fields = ('data_root', 'rows_per_problem', 'baselines', 'candidate_pools',
              'candidate_pool_ids', 'winner_rule')
    if any(left[name] != right[name] for name in fields):
        raise ValueError('R92 comparisons require the identical frozen data, pools and TRAIN SBS')


def references(config, args):
    result = {}
    for name, path, experiment, budgets in (
            ('R84A', args.historical_root, 'R84', shared.BUDGETS),
            ('R89B', args.r89_root, 'R89', dict(A_adamw=57600, B_muon_hybrid=57600))):
        run, old_config, lock, test, val = shared.locked_inputs(path.parent, path.name,
            experiment=experiment, arms=budgets)
        same_data(config, old_config)
        result[name] = dict(run=run, config=old_config, lock=lock, test=test, val=val,
            provenance=dict(model=name, run=str(run), checkpoint_sha256=lock['checkpoint_sha256'],
                same_data=True, retrained=False, newly_matched_control=False))
    return result


def val_screen(config, lock, val, refs):
    rows = []
    for name, ref in refs.items():
        current, old = val['macro'], ref['val']['macro']
        top1 = 100. * (current['macro_top1'] - old['macro_top1'])
        cost = current['macro_mean_cost'] - old['macro_mean_cost']
        regret = current['macro_actual_regret_pct'] - old['macro_actual_regret_pct']
        rows.append(dict(model='R92/' + config['arm'], baseline=name, split='VAL',
            top1_delta_pp=top1, mean_cost_delta=cost, actual_regret_delta_pp=regret,
            pass_reference=top1 > 0 and cost <= 0 and regret <= 0,
            checkpoint_sha256=lock['checkpoint_sha256'], checkpoint_reselection=False))
    return [dict(row, joint_val_pass=all(r['pass_reference'] for r in rows)) for row in rows]


def diagnostic_tables(run, config, lock, val, test, args):
    packed, methods, summaries, results, paths = helpers.selected_predictions(run, config, lock, val, test)
    name = 'R92/' + config['arm']
    methods, summaries = ([dict(row, model=name) for row in data] for data in (methods, summaries))
    refs = references(config, args)
    targets, identity = helpers.load_targets(config, refs['R84A']['config'], refs['R84A']['lock'])
    pairs, recovery, totals = {}, {}, {}
    for ref_name, ref in refs.items():
        old = helpers.selected_predictions(ref['run'], ref['config'], ref['lock'], ref['val'], ref['test'])[0]
        pairs[ref_name], recovery[ref_name], totals[ref_name] = helpers.comparison_tables(
            old, packed, targets, identity, name, ref_name)
    return dict(predictions=packed, methods=methods, summaries=summaries, results=results,
        paths=paths, pairs=pairs, recovery=recovery, recovery_summary=totals,
        target_identity=identity, historical={name: ref['provenance'] for name, ref in refs.items()},
        val_screen=val_screen(config, lock, val, refs))


def diagnostics_pdf(output, config, lock, tables):
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    helper = common.reporting_helpers()
    helper.setup_style()
    path = output / 'target_recovery_and_train.pdf'
    temporary = path.with_name(f'.{path.stem}.{os.getpid()}.pdf')
    try:
        with PdfPages(temporary) as pdf:
            for page, ref_name in enumerate(('R84A', 'R89B'), 1):
                fig = common._page(helper, 'R92/' + config['arm'] + ' | Fixed target recovery vs ' + ref_name,
                    'One unique VAL lock; fixed R84 target36; true recovery, false picks and net cost.',
                    'R92/' + config['arm'], lock, page)
                helper.add_table(fig,
                    ['Split', 'Winners', 'Correct', 'Rescued', 'Lost', 'New false picks', 'Recall (%)'],
                    [[row['split'], str(row['winner_count']), str(row['correct_count']),
                      str(row['rescued']), str(row['lost']), str(row['new_false_pick_count']),
                      '-' if row['recall_pct'] is None else f"{row['recall_pct']:.4f}"]
                     for row in tables['recovery_summary'][ref_name]], [.05, .62, .9, .2], font=10)
                helper.add_table(fig,
                    ['Split', 'Top1 delta pp', 'Corrected', 'Harmed', 'Cost delta', 'Regret delta pp'],
                    [[row['split'], f"{row['top1_delta_pp']:.6f}", str(row['corrected']),
                      str(row['harmed']), f"{row['mean_cost_delta']:.9f}",
                      f"{row['actual_regret_delta_pp']:.6f}"] for row in tables['pairs'][ref_name]
                     if row['problem'] == 'ALL'], [.05, .32, .9, .2], font=10)
                fig.text(.05, .24, 'Independent family/architecture comparison; no claim of equal tree/NN update budget.', fontsize=9)
                fig.text(.05, .18, 'All18 have equal macro weight. Fewer zero groups alone are not success.', fontsize=9)
                pdf.savefig(fig)
                plt.close(fig)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return str(path)


def learning_curves(run, config, lock):
    import matplotlib.pyplot as plt
    history = read_json(run / 'history.json')
    tree = config['arm'] in ('C', 'D')
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    unit = 'Common trees per task' if tree else 'Successful neural updates'
    fig.suptitle('R92/' + config['arm'] + ' | ' + unit + '; unique VAL lock', fontsize=15)
    if tree:
        axes[0, 0].text(.05, .65, 'Full-dataset GPU boosting, not SGD.\n'
            'No fabricated minibatch loss or neural sample exposure.\n'
            'TRAIN is evaluated only at the locked common tree count.',
            transform=axes[0, 0].transAxes, fontsize=11)
        axes[0, 0].set_title('Budget and evidence boundary')
    else:
        for name in ('loss', 'ce', 'cmp', 'risk', 'kl', 'hierarchy_auxiliary',
                     'hierarchy_link', 'hierarchy_entropy'):
            points = [(row['successful_updates'], row['optimization'][name]) for row in history
                      if name in row.get('optimization', {})]
            if points:
                axes[0, 0].plot(*zip(*points), label=name)
        axes[0, 0].legend(fontsize=8)
        axes[0, 0].set_title('TRAIN selection / hierarchy objectives')
    for axis, key, multiplier, title in (
            (axes[0, 1], 'macro_top1', 100., 'Native Top1 (%)'),
            (axes[1, 0], 'macro_actual_regret_pct', 1., 'Actual regret (%)'),
            (axes[1, 1], 'macro_vs_sbs_pct', 1., 'vs fixed TRAIN SBS (%)')):
        for split in ('train', 'val'):
            points = [(row['successful_updates'], row[split]['macro'][key] * multiplier)
                      for row in history if row.get(split) is not None]
            if points:
                axis.plot(*zip(*points), label=split, marker='o' if len(points) == 1 else None)
        axis.set_title(title)
        axis.legend()
    for axis in axes.flat:
        axis.axvline(lock['selected_update'], color='#c45358', linestyle='--', linewidth=1)
        axis.set_xlabel(unit)
        axis.grid(alpha=.2)
    fig.tight_layout(rect=(0, 0, 1, .96))
    output = run / 'full_model_statistics'
    fig.savefig(output / 'learning_curves.pdf')
    fig.savefig(output / 'learning_curves.png', dpi=160)
    plt.close(fig)


def report(args):
    run, config, lock, test, val = locked_inputs(args.root, args.arm)
    with (run / 'postprocess.lock').open('a') as claim:
        fcntl.flock(claim, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            write_json(run / 'postprocess_status.json', dict(experiment='R92', phase='loading_saved_predictions', updated_utc=utc()))
            reference = previous.validate_nss_reference(config, args.nss_root)
            predictions, metrics, methods = shared.load_saved_splits(run, config, lock, test, val)
            shared.saved_libs(run, config, lock, test, predictions, metrics, methods)
            comparisons, alignment = shared.nss_comparisons(config, lock, predictions, args.nss_root)
            tables = diagnostic_tables(run, config, lock, val, test, args)
            result = shared.build_reports(run, config, lock, test, val, predictions, metrics, methods, comparisons, alignment)
            learning_curves(run, config, lock)
            output = run / 'full_model_statistics'
            for name, rows in (
                    ('method_selection_train_val_test.csv', tables['methods']),
                    ('zero_selection_summary.csv', tables['summaries']),
                    ('train_metrics_by_problem.csv', helpers.train_metric_rows(config, lock, tables)),
                    ('val_empirical_screen.csv', tables['val_screen'])):
                common.write_csv(output / name, rows)
            for ref_name in tables['historical']:
                for stem, values in (('paired_changes', tables['pairs']),
                        ('target_recovery', tables['recovery']),
                        ('target_recovery_summary', tables['recovery_summary'])):
                    common.write_csv(output / (stem + '_vs_' + ref_name + '.csv'), values[ref_name])
            common.write_csv(output / 'target_recovery.csv', tables['recovery']['R84A'])
            common.write_csv(output / 'target_recovery_summary.csv', tables['recovery_summary']['R84A'])
            result.update(target_recovery_pdf=diagnostics_pdf(output, config, lock, tables),
                historical_reference=tables['historical'], nss_reference=reference,
                joint_val_pass=all(row['joint_val_pass'] for row in tables['val_screen']))
            provenance = read_json(output / 'protocol.json')
            provenance.update(target_identity=tables['target_identity'], historical_reference=tables['historical'],
                objective=config['objective'], evidence_boundary=BOUNDARY,
                val_empirical_screen=tables['val_screen'], training_unit=config['training_unit'])
            write_json(output / 'protocol.json', provenance)
            write_json(run / 'postprocess_status.json', dict(experiment='R92', phase='complete', completed_utc=utc(), **result))
            return result
        except BaseException as error:
            write_json(run / 'postprocess_status.json', dict(experiment='R92', phase='postprocess_failed',
                updated_utc=utc(), error=str(error), traceback=traceback.format_exc(),
                retry='report only; retain all original training/TEST/LIB records'))
            raise


def compare(args):
    rows, locks, model_results = [], {}, {}
    for arm in ARMS:
        run, config, lock, test, val = locked_inputs(args.root, arm)
        locks[arm] = lock
        tables = diagnostic_tables(run, config, lock, val, test, args)
        model_results[arm] = tables
        for split, result in (('VAL', val), ('TEST', test)):
            macro = result['macro']
            rows.append(dict(model='R92/' + arm, split=split,
                top1_pct=100. * macro['macro_top1'], mean_cost=macro['macro_mean_cost'],
                actual_regret_pct=macro['macro_actual_regret_pct'], selected_update=lock['selected_update'],
                training_unit=config['training_unit'], checkpoint_sha256=lock['checkpoint_sha256']))
    output = args.root / 'comparison_artifacts'
    output.mkdir(parents=True, exist_ok=True)
    common.write_csv(output / 'comparison.csv', rows)
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    helper = common.reporting_helpers()
    helper.setup_style()
    path = output / 'COMPARISON.pdf'
    with PdfPages(path) as pdf:
        fig = common._page(helper, 'R92 | Broad learning mechanisms',
            'A hard hierarchy; B learned hierarchy; C winner trees; D complete-regret trees.',
            'R92 full18', locks['A'], 1)
        helper.add_table(fig, ['Model', 'Split', 'Top1 (%)', 'Mean cost', 'Actual regret (%)', 'Selected step/trees'],
            [[row['model'], row['split'], f"{row['top1_pct']:.4f}", f"{row['mean_cost']:.9f}",
              f"{row['actual_regret_pct']:.6f}", str(row['selected_update'])] for row in rows],
            [.05, .39, .9, .42], font=10)
        fig.text(.05, .29, 'Historical references are not newly matched controls. Tree iterations are not SGD updates.', fontsize=9)
        fig.text(.05, .23, 'No TEST checkpoint selection or per-task best-model assembly; all18 task-equal macro.', fontsize=9)
        pdf.savefig(fig)
        plt.close(fig)
    result = dict(experiment='R92', complete=True, checkpoints=locks,
        comparison_pdf=str(path), evidence_boundary=BOUNDARY,
        training=False, inference=False, checkpoint_reselection=False)
    write_json(output / 'protocol.json', result)
    lines = ['# R92: Full18 Broad Learning Mechanisms', '',
        'A: hard graph hierarchy; B: learned graph hierarchy; C: winner trees; D: relative-regret trees.',
        'One validation-selected lock per version; no test-based reselection or task-wise best-model assembly.', '',
        '| Model | Split | Top1 (%) | Mean cost | Actual regret (%) | Selected step / trees |',
        '| --- | --- | --- | --- | --- | --- |']
    for row in rows:
        lines.append(f"| {row['model']} | {row['split']} | {row['top1_pct']:.4f} | "
            f"{row['mean_cost']:.9f} | {row['actual_regret_pct']:.6f} | {row['selected_update']} |")
    lines.extend(['', '## Evidence Boundary', '', BOUNDARY,
        'Neural arms use57600 successful logical-batch1024 updates; tree arms use3000 trees per task.',
        'All18 tasks have equal macro weight. Historical TEST has informed development and is not blind.',
        'Each arm full_model_statistics contains SBS/Oracle, all-problem metrics, NSS4, paired method distributions,',
        'fixed target36 recovery, lost correct predictions, new false picks and net costs.',
        'Fewer zero-selected methods alone is not successful recovery, and no outcome automatically completes the goal.'])
    (args.root / 'RESULTS.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    import shutil
    shutil.copyfile(path, args.root / 'RESULTS.pdf')
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
    result.add_argument('--r89-root', type=Path, default=DEFAULT_R89B)
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    for name in ('root', 'nss_root', 'historical_root', 'r89_root'):
        setattr(args, name, getattr(args, name).resolve())
    if args.command == 'compare':
        return compare(args)
    if args.arm is None:
        raise ValueError('--arm is required for lib, report and status')
    if args.command == 'status':
        path = args.root / args.arm / 'postprocess_status.json'
        value = read_json(path) if path.exists() else dict(experiment='R92', phase='not_started')
        print(json.dumps(value, indent=2))
        return value
    return lib_stage(args) if args.command == 'lib' else report(args)


if __name__ == '__main__':
    main()

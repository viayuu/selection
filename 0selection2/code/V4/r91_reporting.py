"""R91 locked full18/NSS4 PDFs, frozen target36 recovery, and historical VAL screen."""

import argparse
import fcntl
import json
import math
import os
import traceback
from pathlib import Path

from . import r87_reporting as helpers
from .r76_experiment import file_hash, read_json, utc, write_json
from .r91_experiment import (ARMS, DEFAULT_ROOT, MAX_UPDATES, MODEL_ROLE_ARM,
    validate_checkpoint_config)


shared, previous, common = helpers.shared, helpers.previous, helpers.common
LIBS, DEFAULT_R84, DEFAULT_NSS = helpers.LIBS, helpers.DEFAULT_R84, helpers.DEFAULT_NSS
DEFAULT_R88B = DEFAULT_ROOT.parent / 'R88_execution_role_inputs/B_execution_anchor'
FACTORY, BUDGETS = 'code.V4.r91_experiment', dict.fromkeys(ARMS, MAX_UPDATES)
REFERENCE_BOUNDARY = 'Historical R84A and same-role old-loss R88B; neither is a newly matched original-loss control.'


def locked_inputs(root, arm):
    result = shared.locked_inputs(root, arm, experiment='R91', arms=BUDGETS)
    validate_checkpoint_config(result[1])
    return result


def lib_stage(args):
    run, config, lock, test, _ = locked_inputs(args.root, args.arm)
    with (run / 'lib_stage.lock').open('a') as claim:
        fcntl.flock(claim, fcntl.LOCK_EX | fcntl.LOCK_NB)

        def phase(name, **fields):
            write_json(run / 'lib_stage_status.json', dict(experiment='R91', arm=args.arm,
                phase=name, updated_utc=utc(), **fields))

        reference = previous.validate_nss_reference(config, args.nss_root)
        results = common.r80_lib(args.root, run, config, lock, args.device,
            args.lib_batch_size, phase, test['identity']['model_state_sha256'], factory_module=FACTORY)
        write_json(run / 'lib_stage_complete.json', dict(experiment='R91', complete=True,
            updated_utc=utc(), checkpoint_sha256=lock['checkpoint_sha256'],
            nss_reference=reference, datasets=list(results)))
        phase('complete', checkpoint_sha256=lock['checkpoint_sha256'])
        return results


def selected_predictions(run, config, lock, val, test):
    if val['identity']['model_state_sha256'] != test['identity']['model_state_sha256']:
        raise ValueError('R91 diagnostics require the same unique VAL/TEST state')
    packed, methods, summaries, results, paths = helpers.selected_predictions(run, config, lock, val, test)
    model = config['experiment'] + '/' + config['arm']
    methods = [dict(row, model=model) for row in methods]
    summaries = [dict(row, model=model) for row in summaries]
    return packed, methods, summaries, results, paths


def same_data(left, right, *, matched=False):
    names = ['data_root', 'rows_per_problem', 'baselines', 'candidate_pools',
             'candidate_pool_ids', 'winner_rule']
    if matched:
        names.extend(('common_trainable_sha256', 'preparation_sha256', 'model_params',
            'role_arm', 'global_batch_size', 'max_successful_updates', 'warmup_updates',
            'eval_every_updates', 'task_order_sha256', 'lr', 'optimizer', 'weight_decay',
            'dropout', 'grad_clip', 'microbatch_caps', 'gpu_type', 'eval_batch_size',
            'checkpoint_rule', 'inference_rule', 'main_protocol_lock'))
    if any(left[name] != right[name] for name in names):
        raise ValueError('R91 comparison requires identical frozen data/labels/pools/SBS and matched controls where applicable')


def load_targets(config, historical_config, historical_lock):
    targets, identity = helpers.load_targets(config, historical_config, historical_lock)
    if len(targets) != 36:
        raise ValueError('R91 must use the existing frozen R84 target36, without reselection')
    return targets, dict(identity, sha256=file_hash(Path(identity['path'])), target_count=36)


def historical_inputs(config, historical_root, same_role_root):
    references = {}
    for name, path, experiment, budgets in (
            ('R84A', historical_root, 'R84', shared.BUDGETS),
            ('R88B', same_role_root, 'R88', dict.fromkeys(('A_canonical_anchor', MODEL_ROLE_ARM), MAX_UPDATES))):
        run, old_config, lock, test, val = shared.locked_inputs(
            path.parent, path.name, experiment=experiment, arms=budgets)
        same_data(config, old_config)
        if name == 'R88B' and (old_config['model_params'].get('role_arm') != MODEL_ROLE_ARM
                or old_config['common_trainable_sha256'] != config['common_trainable_sha256']):
            raise ValueError('R88B reference must retain the identical fixed role and fresh initialization')
        references[name] = dict(run=run, config=old_config, lock=lock, test=test, val=val,
            provenance=dict(model=name, checkpoint_sha256=lock['checkpoint_sha256'],
                run=str(run), same_data=True, retrained=False, newly_matched_control=False,
                evidence='main historical reference' if name == 'R84A' else 'same-role historical old-loss reference'))
    return references


def val_screen(config, lock, val, references):
    """Evaluate the prelocked empirical screen; never select a checkpoint or rule."""
    current = val['macro']
    rows = []
    for name, reference in references.items():
        baseline = reference['val']['macro']
        values = [float(record[key]) for record in (current, baseline)
                  for key in ('macro_top1', 'macro_mean_cost', 'macro_actual_regret_pct')]
        if not all(math.isfinite(value) for value in values):
            raise ValueError('R91 joint VAL screen requires finite saved macro metrics')
        top1_delta = 100. * (current['macro_top1'] - baseline['macro_top1'])
        cost_delta = current['macro_mean_cost'] - baseline['macro_mean_cost']
        regret_delta = current['macro_actual_regret_pct'] - baseline['macro_actual_regret_pct']
        rows.append(dict(model='R91/' + config['arm'], baseline=name, split='VAL',
            top1_delta_pp=top1_delta, mean_cost_delta=cost_delta,
            actual_regret_delta_pp=regret_delta, top1_strictly_improves=top1_delta > 0,
            mean_cost_nonincreasing=cost_delta <= 0, actual_regret_nonincreasing=regret_delta <= 0,
            pass_reference=top1_delta > 0 and cost_delta <= 0 and regret_delta <= 0,
            checkpoint_sha256=lock['checkpoint_sha256'],
            baseline_checkpoint_sha256=reference['lock']['checkpoint_sha256'],
            macro_weighting='equal weight across all18 problems',
            newly_matched_control=False, checkpoint_reselection=False))
    passed = len(rows) == 2 and all(row['pass_reference'] for row in rows)
    return [dict(row, joint_val_pass=passed) for row in rows]


def historical_tables(run, config, lock, val, test, historical_root, same_role_root):
    packed, methods, summaries, results, paths = selected_predictions(run, config, lock, val, test)
    references = historical_inputs(config, historical_root, same_role_root)
    r84 = references['R84A']
    targets, identity = load_targets(config, r84['config'], r84['lock'])
    pairs, recovery, totals = {}, {}, {}
    for name, reference in references.items():
        old, _, _, _, _ = selected_predictions(reference['run'], reference['config'],
            reference['lock'], reference['val'], reference['test'])
        pairs[name], recovery[name], totals[name] = helpers.comparison_tables(
            old, packed, targets, identity, 'R91/' + config['arm'], name + ' historical')
    return dict(predictions=packed, methods=methods, summaries=summaries, results=results,
        paths=paths, pairs=pairs, recovery=recovery, recovery_summary=totals,
        target_identity=identity, val_screen=val_screen(config, lock, val, references),
        historical={name: reference['provenance'] for name, reference in references.items()})


def diagnostics_pdf(output, config, lock, tables):
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    helper, model = common.reporting_helpers(), 'R91/' + config['arm']
    path = output / 'target_recovery_and_train.pdf'
    temporary = path.with_name(f'.{path.stem}.{os.getpid()}.pdf')
    try:
        with PdfPages(temporary) as pdf:
            for page, split in enumerate(('TRAIN', 'VAL', 'TEST'), 1):
                fig = common._page(helper, model + ' | ' + split + ': frozen target36 recovery',
                    'Same selected checkpoint; original maximin inference. Native FP64 labels; equal task weights.', model, lock, page)
                rows = [row for row in tables['summaries'] if row['split'] == split]
                helper.add_table(fig, ['Problem', 'Zero groups', 'Zero winner mass', 'Near-zero mass', 'All misses'],
                    [[row['problem'], str(row['zero_with_winner_groups']), str(row['zero_winner_mass']),
                      str(row['near_zero_winner_mass']), str(row['total_missed_winner_count'])] for row in rows],
                    [.05, .34, .9, .5], font=10)
                target = [(name, row) for name, rows in tables['recovery_summary'].items()
                          for row in rows if row['split'] == split]
                helper.add_table(fig, ['Reference', 'Winners', 'Correct', 'Rescued', 'Lost', 'New false picks', 'Recall (%)'],
                    [[name, str(row['winner_count']), str(row['correct_count']), str(row['rescued']),
                      str(row['lost']), str(row['new_false_pick_count']),
                      '-' if row['recall_pct'] is None else f"{row['recall_pct']:.4f}"] for name, row in target],
                    [.05, .19, .9, .09], font=10)
                fig.text(.05, .13, REFERENCE_BOUNDARY, fontsize=9)
                fig.text(.05, .08, 'TRAIN CE: utility. Evaluation macro_ce: original maximin; these are different probability links.', fontsize=9)
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
            write_json(run / 'postprocess_status.json', dict(experiment='R91', phase='loading_saved_predictions', updated_utc=utc()))
            reference = previous.validate_nss_reference(config, args.nss_root)
            predictions, metrics, methods = shared.load_saved_splits(run, config, lock, test, val)
            shared.saved_libs(run, config, lock, test, predictions, metrics, methods)
            comparisons, alignment = shared.nss_comparisons(config, lock, predictions, args.nss_root)
            tables = historical_tables(run, config, lock, val, test, args.historical_root, args.same_role_root)
            result = shared.build_reports(run, config, lock, test, val, predictions, metrics, methods, comparisons, alignment)
            output = run / 'full_model_statistics'
            rows_by_file = [('method_selection_train_val_test.csv', tables['methods']),
                ('zero_selection_summary.csv', tables['summaries']),
                ('train_metrics_by_problem.csv', helpers.train_metric_rows(config, lock, tables)),
                ('val_empirical_screen.csv', tables['val_screen'])]
            for name in tables['historical']:
                rows_by_file.extend((('paired_changes_vs_' + name + '.csv', tables['pairs'][name]),
                    ('target_recovery_vs_' + name + '.csv', tables['recovery'][name]),
                    ('target_recovery_summary_vs_' + name + '.csv', tables['recovery_summary'][name])))
            # Keep the established R84-target filenames for downstream consumers.
            rows_by_file.extend((('target_recovery.csv', tables['recovery']['R84A']),
                ('target_recovery_summary.csv', tables['recovery_summary']['R84A'])))
            for name, rows in rows_by_file:
                common.write_csv(output / name, rows)
            result.update(target_recovery_pdf=diagnostics_pdf(output, config, lock, tables),
                historical_reference=tables['historical'], nss_reference=reference,
                joint_val_pass=all(row['joint_val_pass'] for row in tables['val_screen']))
            provenance = read_json(output / 'protocol.json')
            provenance.update(target_identity=tables['target_identity'], historical_reference=tables['historical'],
                nss_reference=reference, input_roles=config['input_roles'], objective=config['objective'],
                train_ce_probability_link='utility', evaluation_ce_probability_link='maximin',
                val_empirical_screen=tables['val_screen'], evidence_boundary=REFERENCE_BOUNDARY)
            write_json(output / 'protocol.json', provenance)
            write_json(run / 'postprocess_status.json', dict(experiment='R91', phase='complete', completed_utc=utc(), **result))
            return result
        except BaseException as error:
            write_json(run / 'postprocess_status.json', dict(experiment='R91', phase='postprocess_failed',
                updated_utc=utc(), error=str(error), traceback=traceback.format_exc(),
                retry='report only; retain all training/TEST/LIB records'))
            raise


def compare(args):
    selected, locks, configs, screens = {}, {}, {}, []
    for arm in ARMS:
        run, config, lock, test, val = locked_inputs(args.root, arm)
        selected[arm], _, _, _, _ = selected_predictions(run, config, lock, val, test)
        locks[arm], configs[arm] = lock, config
        references = historical_inputs(config, args.historical_root, args.same_role_root)
        screens.extend(val_screen(config, lock, val, references))
    a, b = ARMS
    same_data(configs[a], configs[b], matched=True)
    r84 = references['R84A']
    targets, identity = load_targets(configs[b], r84['config'], r84['lock'])
    pairs, recovery, totals = helpers.comparison_tables(selected[a], selected[b], targets,
        identity, 'R91/' + b, 'R91/' + a)
    output = args.root / 'comparison_artifacts'
    output.mkdir(parents=True, exist_ok=True)
    for name, rows in (('paired_changes_B_vs_A.csv', pairs), ('target_recovery_B_vs_A.csv', recovery),
            ('target_recovery_summary_B_vs_A.csv', totals), ('val_empirical_screen.csv', screens)):
        common.write_csv(output / name, rows)
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    helper = common.reporting_helpers()
    helper.setup_style()
    path = output / 'COMPARISON.pdf'
    temporary = path.with_name(f'.{path.stem}.{os.getpid()}.pdf')
    try:
        with PdfPages(temporary) as pdf:
            fig = common._page(helper, 'R91 | Utility risk/KL links minus utility-CE-only links',
                'Identical R88B model; unique VAL checkpoint per arm. No new inference or checkpoint selection.',
                'R91/B vs A', locks[b], 1)
            helper.add_table(fig, ['Split', 'Top1 delta pp', 'Corrected', 'Harmed', 'Cost delta', 'Regret delta pp'],
                [[row['split'], f"{row['top1_delta_pp']:.6f}", str(row['corrected']), str(row['harmed']),
                  f"{row['mean_cost_delta']:.9f}", f"{row['actual_regret_delta_pp']:.6f}"]
                 for row in pairs if row['problem'] == 'ALL'], [.05, .59, .9, .21], font=10)
            helper.add_table(fig, ['Arm / VAL reference', 'Top1 delta pp', 'Cost delta', 'Regret delta pp', 'Joint VAL pass'],
                [[row['model'].split('/')[-1] + ' / ' + row['baseline'], f"{row['top1_delta_pp']:.6f}",
                  f"{row['mean_cost_delta']:.9f}", f"{row['actual_regret_delta_pp']:.6f}",
                  str(row['joint_val_pass'])] for row in screens], [.05, .34, .9, .19], font=9)
            fig.text(.05, .25, REFERENCE_BOUNDARY, fontsize=9)
            fig.text(.05, .19, 'VAL screen: Top1 strictly improves; macro mean cost and actual regret do not increase against both.', fontsize=9)
            fig.text(.05, .13, 'All18 tasks have equal macro weight. Original pair comparison, averaged-maximin focus and inference retained.', fontsize=9)
            pdf.savefig(fig)
            plt.close(fig)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    result = dict(experiment='R91', complete=True, checkpoints=locks, target_identity=identity,
        comparison_pdf=str(path), val_empirical_screen=screens,
        historical_reference={name: ref['provenance'] for name, ref in references.items()},
        evidence_boundary=REFERENCE_BOUNDARY, training=False, inference=False, checkpoint_reselection=False)
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
    result.add_argument('--same-role-root', type=Path, default=DEFAULT_R88B)
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    for name in ('root', 'nss_root', 'historical_root', 'same_role_root'):
        setattr(args, name, getattr(args, name).resolve())
    if args.command == 'compare':
        return compare(args)
    if args.arm is None:
        raise ValueError('--arm is required for lib, report and status')
    if args.command == 'status':
        path = args.root / args.arm / 'postprocess_status.json'
        result = read_json(path) if path.exists() else dict(experiment='R91', phase='not_started')
        print(json.dumps(result, indent=2))
        return result
    return lib_stage(args) if args.command == 'lib' else report(args)


if __name__ == '__main__':
    main()

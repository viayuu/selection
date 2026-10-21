"""R93 locked full18/LIB/NSS4 reports and matched dropout-only A/B comparison.

Only ``lib`` performs inference, on the launcher-owned GPU. ``report`` and
``compare`` are CPU administration over saved predictions, never selection.
"""

import argparse
import copy
import fcntl
import importlib
import json
import math
import os
import shutil
import traceback
from pathlib import Path

from . import r87_reporting as helpers
from .r76_experiment import file_hash, read_json, utc, write_json


ARMS = ('A', 'B')
MAX_UPDATES = 57600
BUDGETS = dict.fromkeys(ARMS, MAX_UPDATES)
DROPOUT = dict(A=.1, B=0.)
DEFAULT_ROOT = Path(__file__).resolve().parent / 'runs/R93_plain_classification'
shared, previous, common = helpers.shared, helpers.previous, helpers.common
DEFAULT_NSS, DEFAULT_R84 = helpers.DEFAULT_NSS, helpers.DEFAULT_R84
DEFAULT_R89B = DEFAULT_ROOT.parent / 'R89_optimizer_geometry/B_muon_hybrid'
FACTORY = 'code.V4.r93_experiment'
MODEL_FACTORY = FACTORY + '.load_checkpoint'
REFERENCE_BOUNDARY = (
    'R89B is the main historical same-data reference; R84A is secondary. '
    'Neither is a newly matched original-loss control. The A/B algorithmic difference is dropout '
    '(A=0.1, B=0); both use one-forward utility CE only, no R-Drop, WD=1e-4 retained. '
    'A uses released3090_1; B uses4090_1: common code/data/fresh initialization/update budget, '
    'not bitwise hardware-matched. Tiny A/B differences are not causal proof of dropout. '
    'Engineering control, not architectural novelty or root-cause proof. '
    'Development-exposed TEST; no TEST/LIB selection or per-task assembly.')
BOUNDARY = REFERENCE_BOUNDARY


def locked_inputs(root, arm):
    if arm not in ARMS:
        raise ValueError('R93 requires arm A or B')
    result = shared.locked_inputs(root, arm, experiment='R93', arms=BUDGETS)
    # Keep the CLI importable while the experiment owner finishes its namespace.
    importlib.import_module(FACTORY).validate_checkpoint_config(result[1])
    if result[1].get('model_factory') != MODEL_FACTORY:
        raise ValueError('R93 reports require the original R93 checkpoint factory')
    return result


def require_cpu(args):
    if str(args.device) != 'cpu':
        raise ValueError('R93 PDF/report/compare administration must use --device cpu')


def completed_arm_report(run, lock):
    post = read_json(run / 'postprocess_status.json')
    protocol = read_json(run / 'full_model_statistics/protocol.json')
    lib = read_json(run / 'lib_stage_complete.json')
    checkpoint = lock['checkpoint_sha256']
    if (post.get('complete') is not True or post.get('phase') != 'complete'
            or post.get('exit_status') != 0 or protocol.get('complete') is not True
            or protocol.get('checkpoint_lock', {}).get('checkpoint_sha256') != checkpoint
            or lib.get('complete') is not True or lib.get('exit_status') != 0
            or lib.get('checkpoint_sha256') != checkpoint
            or set(lib.get('datasets', [])) != {'TSPLIB', 'CVRPLIB'}):
        raise ValueError('R93 comparison requires same-lock TEST/LIB/PDF completion with rc0')
    for name in ('MODEL_STATISTICS.pdf', 'selection_distributions.pdf', 'target_recovery_and_train.pdf'):
        path = run / 'full_model_statistics' / name
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError('R93 arm PDF missing: ' + name)


def lib_stage(args):
    if not str(args.device).startswith('cuda'):
        raise ValueError('R93 LIB inference belongs to the launcher-owned GPU stage')
    run, config, lock, test, _ = locked_inputs(args.root, args.arm)
    with (run / 'lib_stage.lock').open('a') as claim:
        fcntl.flock(claim, fcntl.LOCK_EX | fcntl.LOCK_NB)

        def phase(name, **fields):
            write_json(run / 'lib_stage_status.json', dict(experiment='R93', arm=args.arm,
                phase=name, updated_utc=utc(), **fields))

        try:
            reference = previous.validate_nss_reference(config, args.nss_root)
            results = common.r80_lib(args.root, run, config, lock, args.device,
                args.lib_batch_size, phase, test['identity']['model_state_sha256'],
                factory_module=FACTORY)
            write_json(run / 'lib_stage_complete.json', dict(experiment='R93', arm=args.arm,
                complete=True, exit_status=0, updated_utc=utc(),
                checkpoint_sha256=lock['checkpoint_sha256'], nss_reference=reference,
                datasets=list(results)))
            phase('complete', exit_status=0, checkpoint_sha256=lock['checkpoint_sha256'])
            return results
        except BaseException as error:
            phase('lib_failed', exit_status=1, error=str(error), automatic_retry=False)
            raise


def same_data(left, right, *, matched=False):
    fields = ('data_root', 'rows_per_problem', 'baselines', 'candidate_pools',
              'candidate_pool_ids', 'winner_rule')
    if any(left[name] != right[name] for name in fields):
        raise ValueError('R93 comparison requires identical frozen data/pools/native labels/TRAIN SBS')
    if not matched:
        return
    fixed = ('common_trainable_sha256', 'preparation_sha256', 'initialization', 'seed',
             'global_batch_size', 'logical_global_batch', 'max_successful_updates',
             'warmup_updates', 'eval_every_updates', 'eval_batch_size', 'task_order_sha256',
             'lr', 'optimizer', 'weight_decay', 'grad_clip', 'role_arm', 'input_roles',
             'checkpoint_rule', 'inference_rule', 'main_protocol_lock', 'objective',
             'microbatch_caps', 'loss')
    if any(left.get(name) != right.get(name) for name in fixed):
        raise ValueError('Matched R93 A/B may differ only in dropout, not training controls')
    expected = dict(seed=2, global_batch_size=1024, max_successful_updates=MAX_UPDATES,
        warmup_updates=4320, eval_every_updates=1440, eval_batch_size=256, weight_decay=1e-4)
    for config in (left, right):
        if any(config.get(name) != value for name, value in expected.items()):
            raise ValueError('R93 matched comparison changed the prelocked training budget')
        if not config.get('common_trainable_sha256') or not config.get('preparation_sha256'):
            raise ValueError('R93 matched comparison needs fresh R84 initialization/preparation identities')
    params = []
    if (left.get('arm'), right.get('arm')) != ARMS:
        raise ValueError('R93 matched comparison must be B versus A')
    for config in (left, right):
        arm = config['arm']
        current = copy.deepcopy(config['model_params'])
        if config.get('dropout') != DROPOUT[arm] or current.get('dropout') != DROPOUT[arm]:
            raise ValueError('R93 dropout must be A=0.1, B=0')
        if current.get('r93_arm', arm) != arm:
            raise ValueError('R93 model-parameter arm differs from the locked run')
        current.pop('dropout')
        current.pop('r93_arm', None)
        params.append(current)
    if params[0] != params[1]:
        raise ValueError('R93 matched A/B model differs beyond dropout')


def selected_predictions(run, config, lock, val, test):
    if val['identity']['model_state_sha256'] != test['identity']['model_state_sha256']:
        raise ValueError('R93 diagnostics require identical locked VAL/TEST model states')
    packed, methods, summaries, results, paths = helpers.selected_predictions(run, config, lock, val, test)
    name = config['experiment'] + '/' + config['arm']
    return (packed, [dict(r, model=name) for r in methods],
            [dict(r, model=name) for r in summaries], results, paths)


def references(config, args):
    result = {}
    for name, path, experiment, budgets in (
            ('R89B', args.r89_root, 'R89', dict(A_adamw=57600, B_muon_hybrid=57600)),
            ('R84A', args.historical_root, 'R84', shared.BUDGETS)):
        path = Path(path)
        run, old_config, lock, test, val = shared.locked_inputs(
            path.parent, path.name, experiment=experiment, arms=budgets)
        same_data(config, old_config)
        result[name] = dict(run=run, config=old_config, lock=lock, test=test, val=val,
            provenance=dict(model=name, run=str(run), checkpoint_sha256=lock['checkpoint_sha256'],
                same_data=True, retrained=False, newly_matched_control=False,
                role='main historical reference' if name == 'R89B' else 'secondary historical reference'))
    return result


def load_targets(config, historical_config, historical_lock):
    targets, identity = helpers.load_targets(config, historical_config, historical_lock)
    if len(targets) != 36:
        raise ValueError('R93 must retain the fixed R84 target36 without reselection')
    return targets, dict(identity, sha256=file_hash(Path(identity['path'])), target_count=36)


def val_screen(config, lock, val, refs):
    rows = []
    for name, ref in refs.items():
        current, old = val['macro'], ref['val']['macro']
        keys = ('macro_top1', 'macro_mean_cost', 'macro_actual_regret_pct')
        if not all(math.isfinite(float(r[key])) for r in (current, old) for key in keys):
            raise ValueError('R93 historical VAL diagnostics require finite saved metrics')
        top1 = 100. * (current[keys[0]] - old[keys[0]])
        cost, regret = (current[key] - old[key] for key in keys[1:])
        rows.append(dict(model='R93/' + config['arm'], baseline=name, split='VAL',
            reference_role='main' if name == 'R89B' else 'secondary', top1_delta_pp=top1,
            mean_cost_delta=cost, actual_regret_delta_pp=regret,
            pass_reference=top1 > 0 and cost <= 0 and regret <= 0,
            checkpoint_sha256=lock['checkpoint_sha256'], checkpoint_reselection=False,
            newly_matched_control=False, macro_weighting='equal weight across all18 problems'))
    return [dict(row, joint_val_pass=all(r['pass_reference'] for r in rows)) for row in rows]


def diagnostic_tables(run, config, lock, val, test, args):
    packed, methods, summaries, results, paths = selected_predictions(run, config, lock, val, test)
    refs = references(config, args)
    targets, identity = load_targets(config, refs['R84A']['config'], refs['R84A']['lock'])
    pairs, recovery, totals = {}, {}, {}
    for name, ref in refs.items():
        old = selected_predictions(ref['run'], ref['config'], ref['lock'], ref['val'], ref['test'])[0]
        pairs[name], recovery[name], totals[name] = helpers.comparison_tables(
            old, packed, targets, identity, 'R93/' + config['arm'], name)
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
            for page, ref in enumerate(('R89B', 'R84A'), 1):
                fig = common._page(helper, 'R93/' + config['arm'] + ' | Fixed target36 vs ' + ref,
                    'Same unique VAL checkpoint; native/FP64 winner; no reselection.', 'R93/' + config['arm'], lock, page)
                helper.add_table(fig,
                    ['Split', 'Winners', 'Correct', 'Rescued', 'Lost', 'New false picks', 'Recall (%)'],
                    [[r['split'], str(r['winner_count']), str(r['correct_count']), str(r['rescued']),
                      str(r['lost']), str(r['new_false_pick_count']),
                      '-' if r['recall_pct'] is None else f"{r['recall_pct']:.4f}"]
                     for r in tables['recovery_summary'][ref]], [.05, .62, .9, .2], font=10)
                helper.add_table(fig,
                    ['Split', 'Top1 delta pp', 'Corrected', 'Harmed', 'Cost delta', 'Regret delta pp'],
                    [[r['split'], f"{r['top1_delta_pp']:.6f}", str(r['corrected']), str(r['harmed']),
                      f"{r['mean_cost_delta']:.9f}", f"{r['actual_regret_delta_pp']:.6f}"]
                     for r in tables['pairs'][ref] if r['problem'] == 'ALL'], [.05, .32, .9, .2], font=10)
                fig.text(.05, .24, 'Historical R89B main / R84A secondary; not newly matched original-loss controls.', fontsize=9)
                fig.text(.05, .18, 'TRAIN CE = raw utility only. Eval macro_ce = original maximin diagnostic, not the optimized loss.', fontsize=9)
                fig.text(.05, .12, 'No comparison/risk/R-Drop; WD=1e-4 retained. Zero-count movement alone is not success.', fontsize=9)
                fig.text(.05, .06, 'A: released3090_1; B:4090_1. Not bitwise hardware-matched; tiny differences are not dropout causal proof.', fontsize=9)
                pdf.savefig(fig)
                plt.close(fig)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return str(path)


def learning_curves(run, config, lock):
    import matplotlib.pyplot as plt
    history = read_json(run / 'history.json')
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle('R93/' + config['arm'] + ' | one-forward utility CE; original maximin evaluation', fontsize=14)
    for key in ('loss', 'ce'):
        points = [(r['successful_updates'], r['optimization'][key]) for r in history
                  if key in r.get('optimization', {})]
        if points:
            axes[0, 0].plot(*zip(*points), label=key + ' (TRAIN raw utility)')
    axes[0, 0].set_title('TRAIN utility CE only; not eval macro_ce')
    axes[0, 0].legend(fontsize=8)
    for axis, key, factor, title in (
            (axes[0, 1], 'macro_ce', 1., 'Evaluation CE: original maximin diagnostic'),
            (axes[0, 2], 'macro_top1', 100., 'Native Top1 (%)'),
            (axes[1, 0], 'macro_mean_cost', 1., 'Task-equal raw mean cost (descriptive)'),
            (axes[1, 1], 'macro_actual_regret_pct', 1., 'Actual regret (%)'),
            (axes[1, 2], 'macro_vs_sbs_pct', 1., 'vs fixed TRAIN SBS (%)')):
        for split in ('train', 'val'):
            points = [(r['successful_updates'], r[split]['macro'][key] * factor)
                      for r in history if r.get(split) is not None and key in r[split]['macro']]
            if points:
                axis.plot(*zip(*points), label=split)
        axis.set_title(title)
        if axis.lines:
            axis.legend()
    for axis in axes.flat:
        axis.axvline(lock['selected_update'], color='#c45358', linestyle='--', linewidth=1)
        axis.set_xlabel('Successful logical-batch1024 updates')
        axis.grid(alpha=.2)
    fig.tight_layout(rect=(0, 0, 1, .95))
    for suffix in ('pdf', 'png'):
        fig.savefig(run / 'full_model_statistics' / ('learning_curves.' + suffix))
    plt.close(fig)


def report(args):
    require_cpu(args)
    run, config, lock, test, val = locked_inputs(args.root, args.arm)
    with (run / 'postprocess.lock').open('a') as claim:
        fcntl.flock(claim, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            write_json(run / 'postprocess_status.json', dict(experiment='R93', arm=args.arm,
                phase='loading_saved_predictions', updated_utc=utc()))
            lib = read_json(run / 'lib_stage_complete.json')
            if lib.get('complete') is not True or lib.get('checkpoint_sha256') != lock['checkpoint_sha256']:
                raise ValueError('R93 CPU report requires completed same-checkpoint GPU LIB')
            reference = previous.validate_nss_reference(config, args.nss_root)
            predictions, metrics, methods = shared.load_saved_splits(run, config, lock, test, val)
            shared.saved_libs(run, config, lock, test, predictions, metrics, methods)
            comparisons, alignment = shared.nss_comparisons(config, lock, predictions, args.nss_root)
            tables = diagnostic_tables(run, config, lock, val, test, args)
            # The shared renderer reads this field only for its printed caption.
            # Keep the actual locked objective/config untouched for all provenance.
            display = dict(config, objective=dict(config['objective'],
                classification='TRAIN raw utility CE only; eval CE is original maximin diagnostic'))
            result = shared.build_reports(run, display, lock, test, val, predictions, metrics, methods, comparisons, alignment)
            learning_curves(run, config, lock)
            output = run / 'full_model_statistics'
            for name, rows in (
                    ('method_selection_train_val_test.csv', tables['methods']),
                    ('zero_selection_summary.csv', tables['summaries']),
                    ('train_metrics_by_problem.csv', helpers.train_metric_rows(config, lock, tables)),
                    ('val_empirical_screen.csv', tables['val_screen'])):
                common.write_csv(output / name, rows)
            for ref in tables['historical']:
                for stem, data in (('paired_changes', tables['pairs']), ('target_recovery', tables['recovery']),
                                   ('target_recovery_summary', tables['recovery_summary'])):
                    common.write_csv(output / (stem + '_vs_' + ref + '.csv'), data[ref])
            for stem, data in (('target_recovery', tables['recovery']), ('target_recovery_summary', tables['recovery_summary'])):
                common.write_csv(output / (stem + '.csv'), data['R84A'])
            result.update(target_recovery_pdf=diagnostics_pdf(output, config, lock, tables),
                historical_reference=tables['historical'], nss_reference=reference,
                joint_val_pass=all(row['joint_val_pass'] for row in tables['val_screen']))
            provenance = read_json(output / 'protocol.json')
            provenance.update(target_identity=tables['target_identity'], historical_reference=tables['historical'],
                objective=config['objective'], evidence_boundary=REFERENCE_BOUNDARY,
                train_ce_probability_link='utility', evaluation_ce_probability_link='maximin',
                one_forward=True, rdrop=False, weight_decay_retained=True,
                val_empirical_screen=tables['val_screen'])
            write_json(output / 'protocol.json', provenance)
            notes = output / 'RESULTS.md'
            with notes.open('a', encoding='utf-8') as stream:
                stream.write('\nTRAIN CE: raw utility only; eval macro_ce: original maximin diagnostic.\n\n' + REFERENCE_BOUNDARY + '\n')
            write_json(run / 'postprocess_status.json', dict(experiment='R93', arm=args.arm,
                phase='complete', exit_status=0, completed_utc=utc(), **result))
            return result
        except BaseException as error:
            write_json(run / 'postprocess_status.json', dict(experiment='R93', arm=args.arm,
                phase='postprocess_failed', exit_status=1, updated_utc=utc(), error=str(error),
                traceback=traceback.format_exc(), automatic_retry=False))
            raise


def compare(args):
    require_cpu(args)
    selected, locks, configs, metrics, screens = {}, {}, {}, [], []
    for arm in ARMS:
        run, config, lock, test, val = locked_inputs(args.root, arm)
        completed_arm_report(run, lock)
        selected[arm] = selected_predictions(run, config, lock, val, test)[0]
        locks[arm], configs[arm] = lock, config
        refs = references(config, args)
        screens.extend(val_screen(config, lock, val, refs))
        for split, result in (('VAL', val), ('TEST', test)):
            metrics.append(dict(model='R93/' + arm, split=split, dropout=DROPOUT[arm],
                selected_update=lock['selected_update'], checkpoint_sha256=lock['checkpoint_sha256'],
                **{key: result['macro'][key] for key in ('macro_top1', 'macro_mean_cost',
                   'macro_sbs_cost', 'macro_oracle_cost', 'macro_vs_sbs_pct',
                   'macro_vs_oracle_pct', 'macro_actual_regret_pct')}))
    same_data(configs['A'], configs['B'], matched=True)
    targets, identity = load_targets(configs['B'], refs['R84A']['config'], refs['R84A']['lock'])
    pairs, recovery, totals = helpers.comparison_tables(selected['A'], selected['B'], targets,
        identity, 'R93/B', 'R93/A')
    output = args.root / 'comparison_artifacts'
    output.mkdir(parents=True, exist_ok=True)
    for name, rows in (('comparison.csv', metrics), ('paired_changes_B_vs_A.csv', pairs),
                       ('target_recovery_B_vs_A.csv', recovery), ('target_recovery_summary_B_vs_A.csv', totals),
                       ('val_empirical_screen.csv', screens)):
        common.write_csv(output / name, rows)
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    helper = common.reporting_helpers()
    helper.setup_style()
    path = output / 'COMPARISON.pdf'
    temporary = path.with_name(f'.{path.stem}.{os.getpid()}.pdf')
    try:
        with PdfPages(temporary) as pdf:
            fig = common._page(helper, 'R93 | Plain classification: dropout B=0 vs A=0.1',
                'Original R88B full18 model. One-forward raw-utility CE only; WD=1e-4 retained.', 'R93 B vs A', locks['B'], 1)
            helper.add_table(fig, ['Split', 'Top1 delta pp', 'Corrected', 'Harmed', 'Cost delta', 'Regret delta pp'],
                [[r['split'], f"{r['top1_delta_pp']:.6f}", str(r['corrected']), str(r['harmed']),
                  f"{r['mean_cost_delta']:.9f}", f"{r['actual_regret_delta_pp']:.6f}"]
                 for r in pairs if r['problem'] == 'ALL'], [.05, .61, .9, .2], font=10)
            helper.add_table(fig, ['Arm / VAL ref', 'Top1 delta pp', 'Cost delta', 'Regret delta pp', 'Pass'],
                [[r['model'] + ' / ' + r['baseline'], f"{r['top1_delta_pp']:.6f}",
                  f"{r['mean_cost_delta']:.9f}", f"{r['actual_regret_delta_pp']:.6f}", str(r['pass_reference'])]
                 for r in screens], [.05, .32, .9, .22], font=10)
            fig.text(.05, .25, 'A/B code-level dropout contrast. Both omit comparison, risk and R-Drop; neither removes WD.', fontsize=9)
            fig.text(.05, .19, 'A: released3090_1; B:4090_1. Not bitwise hardware-matched; tiny differences are not dropout causal proof.', fontsize=9)
            fig.text(.05, .13, 'No architectural novelty/root-cause proof; no TEST selection; no second weight grid.', fontsize=9)
            fig.text(.05, .07, 'R89B main / R84A secondary are historical, not newly matched original-loss controls.', fontsize=9)
            pdf.savefig(fig)
            plt.close(fig)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    result = dict(experiment='R93', complete=True, exit_status=0, checkpoints=locks,
        comparison_pdf=str(path), evidence_boundary=REFERENCE_BOUNDARY,
        matched_difference='code-level dropout contrast: B=0 versus A=0.1', target_identity=identity,
        hardware=dict(A='released3090_1', B='4090_1'), bitwise_hardware_matched=False,
        dropout_causal_proof=False,
        training=False, inference=False, checkpoint_reselection=False)
    write_json(output / 'protocol.json', result)
    lines = ['# R93: Plain Classification Control', '', REFERENCE_BOUNDARY, '',
        'TRAIN: one-forward raw-utility CE. Eval macro_ce: original maximin diagnostic.',
        'Original maximin inference (argmax order equivalent to raw utility), unchanged ties.', '',
        '| Model | Split | Top1 (%) | mean_cost | vs_SBS (%) | vs_Oracle (%) | actual regret (%) |',
        '| --- | --- | --- | --- | --- | --- | --- |']
    for row in metrics:
        lines.append(f"| {row['model']} | {row['split']} | {100 * row['macro_top1']:.6f} | "
            f"{row['macro_mean_cost']:.9f} | {row['macro_vs_sbs_pct']:.6f} | "
            f"{row['macro_vs_oracle_pct']:.6f} | {row['macro_actual_regret_pct']:.6f} |")
    (args.root / 'RESULTS.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
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
        value = read_json(path) if path.exists() else dict(experiment='R93', phase='not_started')
        print(json.dumps(value, indent=2))
        return value
    return lib_stage(args) if args.command == 'lib' else report(args)


if __name__ == '__main__':
    main()

"""Single-arm R111 LIB inference and CPU-only saved full18 PDF/CSV reports.

lib --device cuda:0 uses the R111 checkpoint factory. report --device cpu
never constructs a model, runs TEST inference, or selects a checkpoint.
"""

import argparse
import fcntl
import importlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import textwrap

import numpy as np

from . import r80_reporting, r84_reporting, r85_reporting, r87_reporting, r109_reporting
from .r76_experiment import file_hash, read_json, utc, write_json


ARMS = ('A_family_conditional',)
BUDGETS = dict.fromkeys(ARMS, 57600)
DEFAULT_ROOT = Path(__file__).resolve().parent / 'runs/R111_family_conditional_features'
FACTORY = 'code.V4.r111_experiment'
MODEL_FACTORY = FACTORY + '.load_checkpoint'
PROBLEMS = tuple(r84_reporting.PROBLEMS)
DEFAULT_R84, DEFAULT_NSS = r87_reporting.DEFAULT_R84, r87_reporting.DEFAULT_NSS
DEFAULT_R94B = DEFAULT_ROOT.parent / 'R94_pretrained_node_fusion/B'
DEFAULT_R110 = DEFAULT_ROOT.parent / 'R110_method_conditioned_ce'
REQUIRED_PDFS = r109_reporting.REQUIRED_PDFS
BOUNDARY = (
    'One new full18 R111 model, not an arm comparison. Fixed source-lineage families; '
    'P(method)=P(family)*P(method|family), normalized over the complete legal pool. '
    'Each family has two private full-node EncoderLayer self-attention/FFN layers '
    'before pooling, not single-query readout after pooling. Root reads only shared '
    'h/context and group-mean method tokens (plus family identity); private features '
    'do not feed back into root/shared joint/context. One unweighted NLL on normalized '
    'log probabilities; no second softmax, donor, teacher or auxiliary objective. '
    'Wrong-family private classification gradients are zero, not necessarily AdamW '
    'weight-decay/momentum updates. Shared features are not gradient-isolated. '
    'Singleton conditional probability is one; its private features may be skipped. '
    'All legal joint probabilities compete at inference; no hard family routing or '
    'winner-driven forward. Two4090 DDP, one group, global1024/local512; 57600 updates, '
    'warmup4320, AdamW1e-4 to1e-6, seed2. Original R84 frozen NEW90k data/cache and '
    'legal pools, TRAIN81000/VAL4500/TEST4500 per task. TRAIN-only SBS and per-instance '
    'legal-pool Oracle; ALL task-equal, mixed raw cost descriptive only. Native FP64 '
    'ind is the main winner, FP32 argmin only compatibility. One strict-first minimum '
    'full18 VAL macro_vs_sbs checkpoint for TRAIN/VAL/TEST/LIB, no reselection. '
    'Historical R84A/R94B and any R110 summaries are not newly trained, parameter- '
    'or FLOP-matched controls. NSS is saved R79 historical weights on exact NEW4500 '
    'TEST/native LIB, not new90k retraining. TEST has participated in development. '
    'Mechanism approval is not evidence of performance: single-arm historical '
    'comparisons remain provisional, with no causal or performance-gain claim. '
    'Optional timing is included only from an existing same-lock receipt; no benchmark '
    'is launched, and GPU-resident timing is not end-to-end dataset/solver latency.')


def _private_module(source, name):
    spec = importlib.util.spec_from_file_location('code.V4._r111_' + name, source.__file__)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# Rebind only private function globals, never another reporter's module globals.
common = _private_module(r80_reporting, 'common')
shared = _private_module(r84_reporting, 'shared')
previous = _private_module(r85_reporting, 'previous')
helpers = _private_module(r87_reporting, 'helpers')
shared.common = common
previous.shared, previous.common = shared, common
previous.validate_population = shared.validate_population
helpers.shared, helpers.previous, helpers.common = shared, previous, common
for module in (common, shared, previous, helpers):
    module.DEFAULT_ROOT, module.FACTORY = DEFAULT_ROOT, FACTORY
    module.ARMS, module.BUDGETS = BUDGETS, BUDGETS
LIBS = shared.LIBS
write_csv, read_csv = r109_reporting.write_csv, r109_reporting.read_csv
common.write_csv = write_csv
_historical_selected_train = previous.selected_train


def _runner():
    runner = importlib.import_module(FACTORY)
    if (tuple(runner.ARMS) != ARMS or runner.BUDGETS != BUDGETS
            or runner.MODEL_FACTORY != MODEL_FACTORY):
        raise ValueError('R111 reporter and runner interfaces differ')
    return runner


def _bound_split(run, config, lock, result, split, model_hash):
    path = Path(run) / 'evaluation' / split / 'selected_best/rank000.npz'
    expected = dict(experiment='R111', arm=config['arm'], split=split, tag='selected_best',
        checkpoint_sha256=lock['checkpoint_sha256'], state_hash=lock['state_hash'],
        successful_updates=lock['selected_update'], model_state_sha256=model_hash)
    identity = result.get('identity', {})
    if (not model_hash or any(identity.get(k) != v for k, v in expected.items())
            or set(result.get('per_problem', {})) != set(PROBLEMS)
            or common.resolve_record_path(result['prediction_archive'], run) != path.resolve()
            or read_json(path.parent / 'identity.json') != identity):
        raise ValueError('R111 requires file-bound selected_best ' + split.upper())
    return result, path


def locked_inputs(root, arm):
    run, config, lock, test, val = shared.locked_inputs(Path(root), arm,
        experiment='R111', arms=BUDGETS)
    _runner().validate_checkpoint_config(config)
    required = dict(model_factory=MODEL_FACTORY, rows_per_problem=dict(train=81000, val=4500, test=4500),
        local_batch_size=512, warmup_updates=4320, eval_batch_size=256, optimizer='AdamW',
        selection_head='normalized_family_times_conditional', rdrop_weight=0.)
    caps = {p: 256 if p in ('TSP', 'CVRP') else 128 if p == 'ATSP' else 512 for p in PROBLEMS}
    if (any(config.get(k) != v for k, v in required.items())
            or config.get('microbatch_caps') != caps
            or config.get('lr', {}).get('peak') != 1e-4 or config.get('lr', {}).get('end') != 1e-6
            or config.get('loss') != dict(nll=1., comparison=0., risk=0., rdrop=0.)):
        raise ValueError('R111 report requires the frozen NLL/data/budget/microbatch protocol')
    final = read_json(run / 'final_metrics.json')
    if (final.get('complete') is not True or final.get('test') != test or final.get('val') != val):
        raise ValueError('R111 final_metrics and test_results must describe the same complete TEST')
    model_hash = test.get('identity', {}).get('model_state_sha256')
    for split, result in (('train', final['train']), ('val', val), ('test', test)):
        _bound_split(run, config, lock, result, split, model_hash)
    return run, config, lock, test, val


def selected_train(run, config, lock):
    if config['experiment'] != 'R111':
        return _historical_selected_train(run, config, lock)
    final = read_json(Path(run) / 'final_metrics.json')
    return _bound_split(run, config, lock, final['train'], 'train',
        final['test']['identity']['model_state_sha256'])


previous.selected_train = selected_train


def selected_predictions(run, config, lock, val, test):
    packed, methods, summaries, results, paths = helpers.selected_predictions(run, config, lock, val, test)
    model = config['experiment'] + '/' + config['arm']
    return packed, [dict(r, model=model) for r in methods], [dict(r, model=model) for r in summaries], results, paths


def nss_comparisons(config, lock, predictions, nss_root):
    # Reuse the four-dataset native-NSS schema fix, including data/pool versions.
    rows, alignment = r109_reporting.nss_comparisons(config, lock, predictions, nss_root)
    for row in rows:
        row.update(current_model='R111/' + lock['arm'], current_model_factory=MODEL_FACTORY,
            current_state_hash=lock['state_hash'], current_selected_update=lock['selected_update'])
    return rows, alignment


def nss_reference(config, args):
    """Prefer completed R110/R94 receipts, but never run NSS from this reporter."""
    candidates = []
    if args.nss_root is not None:
        candidates.append((args.nss_root, 'explicit --nss-root'))
    else:
        receipts = [args.r110_root / arm / 'lib_stage_complete.json'
                    for arm in ('A_tied_ce', 'B_private_ce')]
        receipts.sort(key=lambda p: p.stat().st_mtime if p.is_file() else 0, reverse=True)
        for path in [*receipts, args.r94b_root / 'lib_stage_complete.json']:
            receipt = common._optional(path)
            root = receipt.get('nss_reference', {}).get('root')
            if receipt.get('complete') is True and root:
                candidates.append((Path(root), str(path)))
        candidates.append((DEFAULT_NSS, 'saved R84 exact NEW4500 reference'))
    errors = []
    for root, source in candidates:
        try:
            reference = previous.validate_nss_reference(config, root)
        except (OSError, ValueError, KeyError) as error:
            errors.append(str(root) + ': ' + str(error))
        else:
            return root, dict(reference, inherited_from=source)
    raise ValueError('No saved same-pool NEW4500 NSS reference; no implicit NSS inference: ' + '; '.join(errors))


def diagnostic_tables(run, config, lock, val, test, args):
    packed, methods, summaries, results, paths = selected_predictions(run, config, lock, val, test)
    tables = dict(predictions=packed, methods=methods, summaries=summaries, results=results, paths=paths,
        pairs={}, recovery={}, recovery_summary={}, historical={}, historical_status={})
    target_config = dict(config, main_protocol_lock=dict(record=dict(target_list=str(args.target_list))))
    targets, target_identity = None, None
    for name, path, experiment, budgets in (
            ('R84A', args.historical_root, 'R84', dict(A_baseline=57600)),
            ('R94B', args.r94b_root, 'R94', dict(A=57600, B=57600))):
        try:
            old_run, old_config, old_lock, old_test, old_val = shared.locked_inputs(
                path.parent, path.name, experiment=experiment, arms=budgets)
            r109_reporting.same_data(config, old_config)
            if name == 'R84A':
                targets, target_identity = helpers.load_targets(target_config, old_config, old_lock)
                if len(targets) != 36:
                    raise ValueError('R111 retains the original fixed36 targets without reselection')
                target_identity.update(target_count=36, sha256=file_hash(args.target_list))
            old = selected_predictions(old_run, old_config, old_lock, old_val, old_test)[0]
            pairs, recovery, totals = r109_reporting.comparison_tables(old, packed, targets,
                target_identity, 'R111/' + config['arm'], name)
        except (OSError, ValueError, KeyError, AssertionError) as error:
            if name == 'R84A':
                raise
            tables['historical_status'][name] = dict(status='unavailable', run=str(path), error=str(error))
            continue
        tables['pairs'][name], tables['recovery'][name], tables['recovery_summary'][name] = pairs, recovery, totals
        tables['historical'][name] = dict(run=str(path), checkpoint_sha256=old_lock['checkpoint_sha256'],
            same_data=True, retrained=False, newly_matched_control=False, parameter_or_flop_matched=False)
        tables['historical_status'][name] = dict(status='included', run=str(path))
    tables['target_identity'] = target_identity
    return tables


def r110_summary(config, root):
    """Read existing history only; unfinished arms are explicitly provisional."""
    rows, status = [], {}
    for arm in ('A_tied_ce', 'B_private_ce'):
        run = Path(root) / arm
        try:
            old = read_json(run / 'protocol.json')
            r109_reporting.same_data(config, old)
            if (run / 'test_results.json').is_file():
                _, _, lock, test, val = shared.locked_inputs(root, arm,
                    experiment='R110', arms=dict(A_tied_ce=57600, B_private_ce=57600))
                results = dict(VAL=val, TEST=test)
                state, update, digest = 'complete_saved_historical', lock['selected_update'], lock['checkpoint_sha256']
            else:
                best = read_json(run / 'best_eval.json')
                results = dict(VAL=best['val'])
                state, update, digest = 'provisional_VAL_only_no_final_TEST', best['successful_updates'], None
            for split, result in results.items():
                macro = result['macro']
                rows.append(dict(model='R110/' + arm, split=split, problem='ALL', status=state,
                    top1_pct=100 * macro['macro_top1'], mean_cost=macro['macro_mean_cost'],
                    vs_sbs_pct=macro['macro_vs_sbs_pct'], actual_regret_pct=macro['macro_actual_regret_pct'],
                    selected_update=update, checkpoint_sha256=digest, source=str(run), retrained=False))
            status[arm] = dict(status=state, run=str(run))
        except (OSError, ValueError, KeyError, AssertionError) as error:
            status[arm] = dict(status='unavailable', run=str(run), error=str(error))
    return rows, status


def optional_timing(run, lock):
    path = Path(run) / 'inference_timing.json'
    if not path.is_file():
        return dict(status='missing_not_benchmarked', path=str(path))
    try:
        value = read_json(path)
        common.check_identity(value, lock, 'optional timing', require_epoch=False)
        if value.get('complete') is not True or value.get('model_factory', MODEL_FACTORY) != MODEL_FACTORY:
            raise ValueError('Incomplete or different-factory timing receipt')
        model_hash = read_json(Path(run) / 'test_results.json')['result']['identity']['model_state_sha256']
        if value.get('model_state_sha256') != model_hash:
            raise ValueError('Timing model state differs from saved TEST')
        rows = value.get('per_problem', [])
        if (value.get('repetitions') != 3 or len(rows) != 18
                or {r['problem'] for r in rows} != set(PROBLEMS)
                or any(r['instances'] != 128 or not np.isfinite(r['standalone_model']['mean_seconds_per_instance'])
                       or r['standalone_model']['mean_seconds_per_instance'] <= 0 for r in rows)):
            raise ValueError('Timing requires saved all18 / 128 inputs / three repetitions')
        return dict(status='included_saved_receipt', path=str(path), sha256=file_hash(path), receipt=value)
    except (OSError, ValueError, KeyError) as error:
        return dict(status='unavailable', path=str(path), error=str(error))


def lib_stage(args):
    run, config, lock, test, _ = locked_inputs(args.root, args.arm)
    with (run / 'lib_stage.lock').open('a') as claim:
        fcntl.flock(claim, fcntl.LOCK_EX | fcntl.LOCK_NB)
        def phase(name, **fields):
            write_json(run / 'lib_stage_status.json', dict(experiment='R111', arm=args.arm,
                phase=name, updated_utc=utc(), **fields))
        try:
            _, reference = nss_reference(config, args)
            results = common.r80_lib(args.root, run, config, lock, args.device, args.lib_batch_size,
                phase, test['identity']['model_state_sha256'], factory_module=FACTORY)
            write_json(run / 'lib_stage_complete.json', dict(experiment='R111', arm=args.arm, complete=True,
                updated_utc=utc(), checkpoint_sha256=lock['checkpoint_sha256'], state_hash=lock['state_hash'],
                selected_update=lock['selected_update'], model_factory=MODEL_FACTORY,
                model_state_sha256=test['identity']['model_state_sha256'],
                datasets=list(results), nss_reference=reference))
            phase('complete')
            return results
        except Exception as error:
            phase('failed', error=str(error), automatic_retry=False)
            raise


def diagnostics_pdf(output, config, lock, tables, history, history_status, timing):
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    helper, model = common.reporting_helpers(), 'R111/' + config['arm']
    path = output / 'target_recovery_and_train.pdf'
    temporary = path.with_name(f'.{path.stem}.{os.getpid()}.pdf')
    def save(pdf, fig):
        pdf.savefig(fig)
        plt.close(fig)
    try:
        with PdfPages(temporary) as pdf:
            fig = common._page(helper, model + ' | Evidence boundary',
                'Single new arm; historical comparisons are provisional, not a performance claim.', model, lock, 1)
            fig.text(.05, .83, '\n'.join(textwrap.wrap(BOUNDARY, 125)), fontsize=9, va='top')
            fig.text(.05, .29, 'Historical controls: ' + '; '.join(k + ': ' + v['status']
                for k, v in tables['historical_status'].items()), fontsize=9)
            fig.text(.05, .20, 'R110 (not awaited): ' + '; '.join(k + ': ' + v['status']
                for k, v in history_status.items()), fontsize=9)
            fig.text(.05, .11, 'Optional timing: ' + timing['status'], fontsize=9)
            save(pdf, fig)
            fig = common._page(helper, model + ' | Same checkpoint full TRAIN',
                'TRAIN is diagnostic, not checkpoint selection. ALL is task-equal.', model, lock, 2)
            train = helpers.train_metric_rows(config, lock, tables)
            helper.add_table(fig, ['Problem', 'Top1 (%)', 'SBS Top1 (%)', 'mean_cost', 'SBS cost',
                'Oracle cost', 'vs_SBS (%)', 'vs_Oracle (%)', 'actual regret (%)'],
                [helper.metric_cells(r, r['problem']) for r in train], [.045, .21, .91, .62], font=9)
            save(pdf, fig)
            for page, name in enumerate(tables['historical'], 3):
                fig = common._page(helper, model + ' | Fixed36 vs ' + name,
                    'Original R84A target list; true correct/rescued/lost and false picks are distinct.', model, lock, page)
                helper.add_table(fig, ['Split', 'Winners', 'Correct', 'Rescued', 'Lost', 'False', 'New false', 'Net cost'],
                    [[r['split'], str(r['winner_count']), str(r['correct_count']), str(r['rescued']), str(r['lost']),
                      str(r['false_pick_count']), str(r['new_false_pick_count']), f"{r['full18_net_mean_cost_delta']:.9f}"]
                     for r in tables['recovery_summary'][name]], [.05, .56, .9, .24], font=9)
                helper.add_table(fig, ['Split', 'Top1 delta pp', 'Corrected', 'Harmed', 'Cost delta', 'Regret delta pp'],
                    [[r['split'], f"{r['top1_delta_pp']:.6f}", str(r['corrected']), str(r['harmed']),
                      f"{r['mean_cost_delta']:.9f}", f"{r['actual_regret_delta_pp']:.6f}"]
                     for r in tables['pairs'][name] if r['problem'] == 'ALL'], [.05, .25, .9, .21], font=9)
                fig.text(.05, .16, 'Net cost/regret uses the full18 aligned population, not only target winners.', fontsize=10)
                fig.text(.05, .11, 'Raw cost deltas are descriptive; no causal, matched-budget or efficacy conclusion.', fontsize=9)
                save(pdf, fig)
            if history:
                fig = common._page(helper, model + ' | Saved R110 historical summary',
                    'VAL-only rows are provisional; no waiting, rerun, cross-version selection or arm comparison.', model, lock, 5)
                helper.add_table(fig, ['Model / split', 'Status', 'Top1 (%)', 'mean_cost', 'vs_SBS (%)', 'regret (%)'],
                    [[r['model'] + ' ' + r['split'], r['status'], f"{r['top1_pct']:.4f}", f"{r['mean_cost']:.9f}",
                      f"{r['vs_sbs_pct']:.6f}", f"{r['actual_regret_pct']:.6f}"] for r in history],
                    [.04, .4, .92, .35], font=8)
                save(pdf, fig)
            if timing['status'] == 'included_saved_receipt':
                fig = common._page(helper, model + ' | Saved timing receipt',
                    'No benchmark run by this reporter. Scope/exclusions are those of the saved measurement.', model, lock, 6)
                receipt = timing['receipt']
                helper.add_table(fig, ['Problem', 'Inputs', 'Repetitions', 'Milliseconds / input'],
                    [[r['problem'], str(r['instances']), str(receipt['repetitions']),
                      f"{1000 * r['standalone_model']['mean_seconds_per_instance']:.6f}"]
                     for r in receipt['per_problem']], [.05, .27, .9, .55], font=9)
                fig.text(.05, .20, 'Scope: ' + str(receipt.get('scope', 'not recorded')), fontsize=9)
                fig.text(.05, .15, 'Excludes: ' + str(receipt.get('excludes', 'not recorded')), fontsize=9)
                fig.text(.05, .10, 'GPU-resident standalone model, not dataset/solver end-to-end latency.', fontsize=9)
                save(pdf, fig)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return str(path)


def report(args):
    if args.device != 'cpu':
        raise ValueError('R111 report is CPU-only saved-evidence processing; use lib for LIB inference')
    run = args.root / args.arm
    with (run / 'postprocess.lock').open('a') as claim:
        fcntl.flock(claim, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            write_json(run / 'postprocess_status.json', dict(experiment='R111', phase='loading_saved_evidence', updated_utc=utc()))
            run, config, lock, test, val = locked_inputs(args.root, args.arm)
            predictions, metrics, methods = shared.load_saved_splits(run, config, lock, test, val)
            shared.saved_libs(run, config, lock, test, predictions, metrics, methods)
            nss_root, reference = nss_reference(config, args)
            comparisons, alignment = nss_comparisons(config, lock, predictions, nss_root)
            tables = diagnostic_tables(run, config, lock, val, test, args)
            train_rows = helpers.train_metric_rows(config, lock, tables)
            history, history_status = r110_summary(config, args.r110_root)
            timing = optional_timing(run, lock)
            render_config = dict(config, objective=dict(classification='unweighted NLL on normalized joint logP'))
            result = shared.build_reports(run, render_config, lock, test, val, predictions,
                [*train_rows, *metrics], methods, comparisons, alignment)
            output = run / 'full_model_statistics'
            nss_proof = r109_reporting.validate_nss_csv(output / 'nss_comparison.csv', config, lock, predictions)
            for name, rows in (('method_selection_train_val_test.csv', tables['methods']),
                    ('zero_selection_summary.csv', tables['summaries']),
                    ('train_metrics_by_problem.csv', train_rows), ('historical_R110_summary.csv', history)):
                write_csv(output / name, rows)
            for name in tables['historical']:
                for stem, key in (('paired_changes', 'pairs'), ('target_recovery', 'recovery'),
                                  ('target_recovery_summary', 'recovery_summary')):
                    write_csv(output / (stem + '_vs_' + name + '.csv'), tables[key][name])
            supplement = diagnostics_pdf(output, config, lock, tables, history, history_status, timing)
            for name in REQUIRED_PDFS:
                if not (output / name).is_file() or not (output / name).stat().st_size:
                    raise ValueError('Missing required R111 PDF: ' + name)
            provenance = read_json(output / 'protocol.json')
            provenance.update(evidence_boundary=BOUNDARY, scientific_status='provisional_single_arm_historical_controls',
                selection_head=config['selection_head'], objective=config.get('objective', config['loss']),
                conditional_model=config.get('conditional_model'), source_sha256=config['source_sha256'],
                selected_archives_file_bound=True, target_identity=tables['target_identity'],
                historical_reference=tables['historical'], historical_status=tables['historical_status'],
                historical_R110_status=history_status, nss_reference=reference, nss_csv_delivery_proof=nss_proof,
                inference_timing=timing, checkpoint_reselection=False, inference=False)
            write_json(output / 'protocol.json', provenance)
            result.update(target_recovery_pdf=supplement, scientific_status=provenance['scientific_status'],
                historical_status=tables['historical_status'], historical_R110_status=history_status,
                inference_timing_status=timing['status'])
            write_json(run / 'postprocess_status.json', dict(experiment='R111', arm=args.arm, phase='complete',
                completed_utc=utc(), checkpoint_sha256=lock['checkpoint_sha256'], **result))
            return result
        except Exception as error:
            write_json(run / 'postprocess_status.json', dict(experiment='R111', arm=args.arm, phase='postprocess_failed',
                updated_utc=utc(), error=str(error), retry='report only; preserve all training/TEST/LIB evidence'))
            raise


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('command', choices=('lib', 'report', 'status'))
    result.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    result.add_argument('--arm', choices=ARMS, default=ARMS[0])
    result.add_argument('--device', default=None, help='Defaults: lib cuda:0; report cpu')
    result.add_argument('--lib-batch-size', type=int, choices=(1, 2), default=2)
    result.add_argument('--nss-root', type=Path, default=None,
        help='Optional explicit reference; otherwise reuse completed R110/R94 exact NEW4500 NSS')
    result.add_argument('--historical-root', type=Path, default=DEFAULT_R84 / 'A_baseline')
    result.add_argument('--r94b-root', type=Path, default=DEFAULT_R94B)
    result.add_argument('--r110-root', type=Path, default=DEFAULT_R110)
    result.add_argument('--target-list', type=Path, default=DEFAULT_R84 / 'comparison_artifacts/target_list.json')
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    args.device = args.device or ('cuda:0' if args.command == 'lib' else 'cpu')
    for name in ('root', 'nss_root', 'historical_root', 'r94b_root', 'r110_root', 'target_list'):
        if getattr(args, name) is not None:
            setattr(args, name, getattr(args, name).expanduser().resolve())
    if args.command == 'status':
        run = args.root / args.arm
        value = dict(experiment='R111', arm=args.arm, runner=common._optional(run / 'status.json'),
            postprocess=common._optional(run / 'postprocess_status.json'), inference=False)
        print(json.dumps(value, indent=2))
        return value
    return lib_stage(args) if args.command == 'lib' else report(args)


if __name__ == '__main__':
    main()

"""R85 locked full18 reports and saved-prediction zero-selection diagnostics.

GPU LIB inference uses the R85 checkpoint factory. Historical R84 NSS4500
predictions are read-only references; this reporter never trains or reruns NSS.
"""

import argparse
import fcntl
import json
import os
import traceback
from pathlib import Path

import numpy as np

from code.unified_selector.registry import PROBLEMS
from . import r84_reporting as shared
from .r76_experiment import file_hash, read_json, utc, write_json
from .r85_experiment import (
    ARMS, DEFAULT_ROOT, MAX_UPDATES, WINNER_RULE, selection_counts, zero_summary,
)


common, LIBS = shared.common, shared.LIBS
FACTORY = 'code.V4.r85_experiment'
BUDGETS = dict.fromkeys(ARMS, MAX_UPDATES)
DEFAULT_R84 = DEFAULT_ROOT.parent / 'R84_new90k_attention'
DEFAULT_NSS = DEFAULT_R84 / 'nss_reference'
validate_population = shared.validate_population


def locked_inputs(root, arm):
    return shared.locked_inputs(root, arm, experiment='R85', arms=BUDGETS)


def validate_nss_reference(config, nss_root):
    """Check the two saved prediction hashes and their exact release/pool."""
    plan = read_json(nss_root / 'protocol.json')
    if (Path(plan['data_root']).resolve() != Path(config['data_root']).resolve()
            or plan.get('expected_test_rows_per_task') != 4500
            or plan.get('retrained_on_new90k') is not False):
        raise ValueError('Reuse only the complete R84 NSS reference on the exact new4500 TEST')
    hashes = {}
    for problem in ('TSP', 'CVRP'):
        path = nss_root / problem / 'test_predictions.npz'
        receipt = read_json(path.with_name('test_once.json'))
        metadata = read_json(path.with_name('test_metrics.json'))
        digest = file_hash(path)
        if (receipt.get('state') != 'complete' or receipt.get('predictions_sha256') != digest
                or metadata.get('n') != 4500
                or metadata.get('checkpoint', {}).get('sha256') != plan['checkpoints'][problem]['sha256']):
            raise ValueError('Changed/incomplete historical NSS predictions: ' + problem)
        with np.load(path, allow_pickle=False) as archive:
            pool = archive['pool'].astype(str).tolist()
            if (pool != config['candidate_pools'][problem] or len(archive['winner']) != 4500
                    or archive['costs'].dtype != np.float64):
                raise ValueError('NSS candidate pool or original cost precision differs: ' + problem)
        hashes[problem] = digest
    return dict(root=str(nss_root.resolve()), prediction_sha256=hashes,
        retrained=False, new_inference=False, exact_test_rows=4500)


def lib_stage(args):
    root = args.root.resolve()
    run, config, lock, test, _ = locked_inputs(root, args.arm)
    with (run / 'lib_stage.lock').open('a') as claim:
        fcntl.flock(claim, fcntl.LOCK_EX | fcntl.LOCK_NB)

        def phase(name, **fields):
            write_json(run / 'lib_stage_status.json', dict(experiment='R85', arm=args.arm,
                phase=name, updated_utc=utc(), **fields))

        phase('reusing_exact4500_NSS_reference')
        reference = validate_nss_reference(config, args.nss_root)
        results = common.r80_lib(root, run, config, lock, args.device, args.lib_batch_size,
            phase, test['identity']['model_state_sha256'], factory_module=FACTORY)
        write_json(run / 'lib_stage_complete.json', dict(experiment='R85', complete=True,
            updated_utc=utc(), checkpoint_sha256=lock['checkpoint_sha256'],
            nss_reference=reference, datasets=list(results)))
        phase('complete', checkpoint_sha256=lock['checkpoint_sha256'])
        return results


def read_split(path, config, split):
    helper = common.reporting_helpers()
    predictions, rows = {}, []
    with np.load(path, allow_pickle=False) as archive:
        for problem in PROBLEMS:
            data = helper.load_predictions(archive, path, problem + '__')
            validate_population(data, problem, split, config)
            predictions[problem] = data
            rows.extend(selection_counts(archive, problem + '__', problem, split))
    return predictions, rows


def selected_train(run, config, lock):
    best = read_json(run / 'best_eval.json')
    result = best.get('train')
    if result is None or best.get('successful_updates') != lock['selected_update']:
        raise ValueError('Report requires full TRAIN predictions from the same selected checkpoint')
    identity = result['identity']
    if (identity.get('experiment') != config['experiment'] or identity.get('arm') != config['arm']
            or identity.get('state_hash') != lock['state_hash']
            or identity.get('successful_updates') != lock['selected_update']):
        raise ValueError('Selected TRAIN provenance differs from the locked model')
    path = Path(result['prediction_archive'])
    stored_identity = read_json(path.parent / 'identity.json')
    if stored_identity != identity:
        raise ValueError('Selected TRAIN archive identity changed')
    return result, path


def paired_comparison(old, new, model, baseline, split):
    rows = []
    for problem in PROBLEMS:
        left, right = old[problem], new[problem]
        if left.pool != right.pool:
            raise ValueError('Historical comparison candidate pool differs: ' + problem)
        for name in ('indices', 'source_indices', 'pool_ids', 'costs', 'winner'):
            np.testing.assert_array_equal(getattr(left, name), getattr(right, name))
        old_ok, new_ok = left.pred == left.winner, right.pred == right.winner
        index, oracle = np.arange(left.n), left.costs.min(1)
        delta = right.costs[index, right.pred] - left.costs[index, left.pred]
        both_wrong = ~old_ok & ~new_ok
        rows.append(dict(model=model, baseline=baseline, split=split, problem=problem, n=right.n,
            corrected=int((~old_ok & new_ok).sum()), harmed=int((old_ok & ~new_ok).sum()),
            net_correct=int(new_ok.sum() - old_ok.sum()), changed=int((left.pred != right.pred).sum()),
            both_wrong_cheaper=int((both_wrong & (delta < 0)).sum()),
            both_wrong_dearer=int((both_wrong & (delta > 0)).sum()),
            top1_delta_pp=100. * float(new_ok.mean() - old_ok.mean()),
            mean_cost_delta=float(delta.mean()), actual_regret_delta_pp=100. * float((delta / oracle).mean())))
    total = dict(rows[0], problem='ALL', n=sum(row['n'] for row in rows))
    for name in ('corrected', 'harmed', 'net_correct', 'changed', 'both_wrong_cheaper', 'both_wrong_dearer'):
        total[name] = sum(row[name] for row in rows)
    for name in ('top1_delta_pp', 'mean_cost_delta', 'actual_regret_delta_pp'):
        total[name] = float(np.mean([row[name] for row in rows]))
    return rows + [total]


def fixed_target_recovery(old, new, targets, model, baseline, split):
    rows = []
    for problem, solver in targets:
        left, right = old[problem], new[problem]
        slot = right.pool.index(solver)
        truth, was, chosen = right.winner == slot, left.pred == slot, right.pred == slot
        oracle = right.costs.min(1)
        index = np.arange(right.n)
        new_regret = (right.costs[index, right.pred] - oracle) / oracle
        old_regret = (left.costs[index, left.pred] - oracle) / oracle
        correct, support, picks = int((truth & chosen).sum()), int(truth.sum()), int(chosen.sum())
        false = chosen & ~truth & ~was
        rows.append(dict(model=model, baseline=baseline, split=split, problem=problem, solver=solver,
            target_definition='R84A selected VAL winner_count>0 and selected_count==0',
            winner_count=support, baseline_missed=int((truth & ~was).sum()), correct_count=correct,
            selected_count=picks, recall_pct=100. * correct / support if support else None,
            precision_pct=100. * correct / picks if picks else None,
            rescued=int((truth & ~was & chosen).sum()), lost=int((truth & was & ~chosen).sum()),
            false_pick_count=int((chosen & ~truth).sum()), new_false_pick_count=int(false.sum()),
            new_false_pick_regret_contribution_pp=100. * float(new_regret[false].sum() / right.n),
            target_regret_delta_pp=100. * float((new_regret - old_regret)[truth].sum() / right.n)))
    return rows


def diagnostic_tables(run, config, lock, val, test, predictions, historical_root):
    model = 'R85/' + config['arm']
    train, train_path = selected_train(run, config, lock)
    if train['identity']['model_state_sha256'] != test['identity']['model_state_sha256']:
        raise ValueError('TRAIN/VAL/TEST diagnostics must use the same selected state')
    all_predictions, methods, summaries = {}, [], []
    for split, result, path in (('TRAIN', train, train_path),
            ('VAL', val, common.selected_val_path(run, lock)),
            ('TEST', test, run / 'evaluation/test/selected_best/rank000.npz')):
        packed, observations = read_split(path, config, split)
        all_predictions[split] = packed
        methods.extend(dict(row, model=model) for row in observations)
        summary = zero_summary(observations)
        for problem, counts in [*summary['per_problem'].items(), ('ALL', summary['all'])]:
            summaries.append(dict(model=model, split=split, problem=problem, **counts))
    historical_root = historical_root.resolve()
    old_run, old_config, old_lock, old_test, old_val = shared.locked_inputs(
        historical_root.parent, historical_root.name)
    if (Path(old_config['data_root']).resolve() != Path(config['data_root']).resolve()
            or old_config['rows_per_problem'] != config['rows_per_problem']):
        raise ValueError('R84A historical reference must use the exact same NEW90k split')
    baseline_predictions, pairs, recovery, targets = {}, [], [], []
    for split, path in (('VAL', common.selected_val_path(old_run, old_lock)),
            ('TEST', old_run / 'evaluation/test/selected_best/rank000.npz')):
        previous, observations = read_split(path, old_config, split)
        baseline_predictions[split] = previous
        if split == 'VAL':
            targets = [(row['problem'], row['solver']) for row in observations
                       if row['winner_count'] > 0 and row['selected_count'] == 0]
        pairs.extend(paired_comparison(previous, all_predictions[split], model, 'R84A', split))
        recovery.extend(fixed_target_recovery(previous, all_predictions[split], targets, model, 'R84A', split))
    return dict(methods=methods, summaries=summaries, pairs=pairs, recovery=recovery,
        historical=dict(model='R84A', run=str(old_run), checkpoint_sha256=old_lock['checkpoint_sha256'],
            evidence='historical reference, not a newly retrained control',
            same_data_root=True, fixed_targets=[dict(problem=p, solver=s) for p, s in targets]),
        train_result=train, train_path=train_path)


def diagnostic_pdf(run, output, config, lock, tables):
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    helper = common.reporting_helpers()
    model = 'R85/' + config['arm']
    path = output / 'zero_selection_and_baseline.pdf'
    temporary = path.with_name(f'.{path.stem}.{os.getpid()}.pdf')
    try:
        with PdfPages(temporary) as pdf:
            fig = common._page(helper, model + ' | TRAIN and held-out selection',
                'Same locked checkpoint; zero groups do not substitute for actual corrected winners.', model, lock, 1)
            rows = [row for row in tables['summaries'] if row['problem'] == 'ALL']
            helper.add_table(fig, ['Split', 'n', 'Zero groups', 'Zero winner mass', 'All missed winners'],
                [[row['split'], f"{row['n']:,}", str(row['zero_with_winner_groups']),
                  f"{row['zero_winner_mass']:,}", f"{row['total_missed_winner_count']:,}"] for row in rows],
                [.05, .64, .9, .19], font=12)
            helper.add_table(fig, ['Split', 'Top1 delta pp', 'Corrected', 'Harmed', 'Cost delta', 'Regret delta pp'],
                [[row['split'], f"{row['top1_delta_pp']:.6f}", str(row['corrected']), str(row['harmed']),
                  f"{row['mean_cost_delta']:.9f}", f"{row['actual_regret_delta_pp']:.6f}"]
                 for row in tables['pairs'] if row['problem'] == 'ALL'], [.05, .37, .9, .17], font=10)
            for i, text in enumerate((
                    'Deltas compare against historical R84A on exactly the same instances/costs/candidate pool.',
                    'Winner mass counts true winners of zero-selected methods; one false pick can shrink it.',
                    'Fixed target recovery, precision, false picks and net regret are in the companion CSV.',
                    'VAL selects the checkpoint. TEST and LIB never select a new rule or checkpoint.')):
                fig.text(.05, .29 - i * .045, text, fontsize=10)
            pdf.savefig(fig)
            plt.close(fig)
            for page, split in enumerate(('TRAIN', 'VAL', 'TEST'), 2):
                fig = common._page(helper, model + ' | ' + split + ': zero-selection evidence',
                    'Native FP64 winner labels. ALL counts sum tasks; all cost deltas use equal task weights.', model, lock, page)
                rows = [row for row in tables['summaries'] if row['split'] == split]
                helper.add_table(fig, ['Problem', 'Zero groups', 'Zero winner mass', 'Near-zero mass', 'All misses'],
                    [[row['problem'], str(row['zero_with_winner_groups']), str(row['zero_winner_mass']),
                      str(row['near_zero_winner_mass']), str(row['total_missed_winner_count'])] for row in rows],
                    [.05, .2, .9, .63], font=11)
                pdf.savefig(fig)
                plt.close(fig)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return str(path)


def report(args):
    root = args.root.resolve()
    run, config, lock, test, val = locked_inputs(root, args.arm)
    with (run / 'postprocess.lock').open('a') as claim:
        fcntl.flock(claim, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            write_json(run / 'postprocess_status.json', dict(experiment='R85', phase='loading_saved_predictions', updated_utc=utc()))
            reference = validate_nss_reference(config, args.nss_root)
            predictions, metrics, methods = shared.load_saved_splits(run, config, lock, test, val)
            shared.saved_libs(run, config, lock, test, predictions, metrics, methods)
            comparisons, alignment = shared.nss_comparisons(config, lock, predictions, args.nss_root)
            tables = diagnostic_tables(run, config, lock, val, test, predictions, args.historical_root)
            result = shared.build_reports(run, config, lock, test, val, predictions, metrics, methods, comparisons, alignment)
            output = run / 'full_model_statistics'
            for name, rows in (('method_selection_train_val_test.csv', tables['methods']),
                    ('zero_selection_summary.csv', tables['summaries']),
                    ('paired_changes_vs_R84A.csv', tables['pairs']),
                    ('fixed_zero_target_recovery_vs_R84A.csv', tables['recovery'])):
                common.write_csv(output / name, rows)
            helper, train_metrics = common.reporting_helpers(), []
            for problem in PROBLEMS:
                data = next(row for row in tables['summaries'] if row['split'] == 'TRAIN' and row['problem'] == problem)
                saved = tables['train_result']['per_problem'][problem]
                train_metrics.append(dict(problem=problem, split='TRAIN', n=data['n'],
                    top1_pct=100. * saved['top1'], mean_cost=saved['mean_cost'], sbs_cost=saved['sbs_cost'],
                    oracle_cost=saved['oracle_cost'], vs_sbs_pct=saved['vs_sbs_pct'],
                    vs_oracle_pct=saved['vs_oracle_pct'], actual_regret_pct=saved['actual_regret_pct'],
                    sbs_selection_split='TRAIN', winner_rule=WINNER_RULE))
            total = dict(train_metrics[0], problem='ALL', n=sum(row['n'] for row in train_metrics))
            for key in ('top1_pct', 'mean_cost', 'sbs_cost', 'oracle_cost', 'vs_sbs_pct', 'vs_oracle_pct', 'actual_regret_pct'):
                total[key] = float(np.mean([row[key] for row in train_metrics]))
            common.write_csv(output / 'train_metrics_by_problem.csv', [total, *train_metrics])
            result.update(zero_selection_pdf=diagnostic_pdf(run, output, config, lock, tables),
                historical_reference=tables['historical'], nss_reference=reference)
            provenance = read_json(output / 'protocol.json')
            provenance.update(historical_reference=tables['historical'], nss_reference=reference,
                zero_selection_pdf=result['zero_selection_pdf'], objective=config['objective'])
            write_json(output / 'protocol.json', provenance)
            write_json(run / 'postprocess_status.json', dict(experiment='R85', phase='complete', completed_utc=utc(), **result))
            return result
        except BaseException as error:
            write_json(run / 'postprocess_status.json', dict(experiment='R85', phase='postprocess_failed',
                updated_utc=utc(), error=str(error), traceback=traceback.format_exc(),
                retry='report only; retain training/TEST/LIB'))
            raise


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('command', choices=('lib', 'report', 'status'))
    result.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    result.add_argument('--arm', choices=ARMS, required=True)
    result.add_argument('--device', default='cpu')
    result.add_argument('--lib-batch-size', type=int, choices=(1, 2), default=2)
    result.add_argument('--nss-root', type=Path, default=DEFAULT_NSS)
    result.add_argument('--historical-root', type=Path, default=DEFAULT_R84 / 'A_baseline')
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    for key in ('root', 'nss_root', 'historical_root'):
        setattr(args, key, getattr(args, key).resolve())
    if args.command == 'status':
        path = args.root / args.arm / 'postprocess_status.json'
        result = read_json(path) if path.exists() else dict(experiment='R85', phase='not_started')
        print(json.dumps(result, indent=2))
        return result
    return lib_stage(args) if args.command == 'lib' else report(args)


if __name__ == '__main__':
    main()

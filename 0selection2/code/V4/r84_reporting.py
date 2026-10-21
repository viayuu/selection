"""R84 locked TEST/LIB/NSS reports, with grouped selection/winner PDF plots.

The GPU `lib` stage runs the current checkpoint on both native LIBs and invokes
the separate historical-NSS reference helper on the same new4500 TEST rows.
The CPU `report` stage never starts model inference or checkpoint selection.
"""

import argparse
import fcntl
import json
import os
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from code.unified_selector.registry import GLOBAL_SOLVERS, PROBLEMS
from . import r80_reporting as common
from .r76_experiment import file_hash, read_json, utc, write_json
from .r84_experiment import ARMS, MAX_UPDATES, WINNER_RULE, DEFAULT_ROOT


LIBS = dict(TSPLIB='TSP', CVRPLIB='CVRP')
BUDGETS = dict.fromkeys(ARMS, MAX_UPDATES)
FACTORY = 'code.V4.r84_experiment'
DEFAULT_R79 = Path(__file__).resolve().parent / 'runs/R79_NSS_new_data_batch1024'


def locked_inputs(root, arm, *, experiment='R84', arms=BUDGETS):
    return common.locked_inputs(Path(root), arm, experiment=experiment, arms=arms)


def validate_population(data, problem, split, config):
    """Use this release's row count, aliases and actual stable candidate order."""
    expected = config['rows_per_problem'][split.lower()]
    names, ids = config['candidate_pools'][problem], config['candidate_pool_ids'][problem]
    aliases = config.get('scenario_solver_aliases', {})
    canonical = [aliases.get(name, name) for name in data.pool]
    if (data.n != expected or data.pool != names or data.pool_ids.tolist() != ids
            or ids != sorted(set(ids)) or canonical != [GLOBAL_SOLVERS[index] for index in ids]):
        raise ValueError(f'Invalid released population or candidate identities: {split}/{problem}')
    np.testing.assert_array_equal(data.indices, np.arange(expected))


def load_saved_splits(run, config, lock, test, val):
    helper = common.reporting_helpers()
    predictions, metrics, methods = {}, [], []
    model_hash = test['identity']['model_state_sha256']
    if val['identity']['model_state_sha256'] != model_hash:
        raise ValueError('Selected VAL and TEST used different model states')
    for split, result, path in (
            ('VAL', val, common.selected_val_path(run, lock)),
            ('TEST', test, run / 'evaluation/test/selected_best/rank000.npz')):
        identity = read_json(path.parent / 'identity.json')
        if (identity != result['identity'] or identity.get('experiment') != config['experiment']
                or identity.get('arm') != config['arm'] or identity.get('split') != split.lower()
                or identity.get('successful_updates') != lock['selected_update']
                or identity.get('state_hash') != lock['state_hash']
                or identity.get('model_state_sha256') != model_hash
                or split == 'TEST' and identity.get('checkpoint_sha256') != lock['checkpoint_sha256']):
            raise ValueError('Saved evaluation identity differs from the unique locked checkpoint')
        rows = []
        with np.load(path, allow_pickle=False) as archive:
            for problem in PROBLEMS:
                data = helper.load_predictions(archive, path, problem + '__')
                validate_population(data, problem, split, config)
                if data.n != result['per_problem'][problem]['n']:
                    raise ValueError('Saved metric population differs from predictions')
                predictions[split, problem] = data
                rows.append(common.metric_row(problem, split, data, result['per_problem'][problem],
                    config['baselines'][problem], config, lock, path))
                methods.extend(dict(row, model=config['experiment'] + '/' + config['arm']) for row in helper.method_rows(
                    problem, split, result['per_problem'][problem], data, Path(config['data_root']).name))
        metrics.extend([common.macro_row(split, rows, result), *rows])
    return predictions, metrics, methods


def lib_stage(args):
    """Inference only; PDF rendering is delegated to the CPU stage."""
    root = args.root.resolve()
    run, config, lock, test, val = locked_inputs(root, args.arm)
    with (run / 'lib_stage.lock').open('a') as claim:
        fcntl.flock(claim, fcntl.LOCK_EX | fcntl.LOCK_NB)

        def phase(name, **fields):
            write_json(run / 'lib_stage_status.json', dict(phase=name, updated_utc=utc(), **fields))

        # This helper validates/restores the actual R84 model factory and does not
        # contain the old heldout-count/alias check used by the R80 PDF loader.
        results = common.r80_lib(root, run, config, lock, args.device, args.lib_batch_size,
            phase, test['identity']['model_state_sha256'], factory_module=FACTORY)
        from .r84_nss_reference import run_reference
        phase('new_test_historical_NSS_reference')
        run_reference(SimpleNamespace(model_run=run, root=args.nss_root, data_root=Path(config['data_root']),
            r79_root=args.r79_root, device=args.device, workers=0, batch_size=8))
        write_json(run / 'lib_stage_complete.json', dict(complete=True, updated_utc=utc(),
            checkpoint_sha256=lock['checkpoint_sha256'], nss_root=str(args.nss_root), datasets=list(results)))
        phase('complete', checkpoint_sha256=lock['checkpoint_sha256'])
        return results


def saved_libs(run, config, lock, test, predictions, metrics, methods):
    helper = common.reporting_helpers()
    results = {}
    for dataset, problem in LIBS.items():
        path = run / 'lib_evaluation/predictions' / (dataset + '.npz')
        result = read_json(run / 'lib_evaluation' / (dataset + '_results.json'))
        identity = result['protocol']
        if (identity.get('experiment') != config['experiment'] or identity.get('arm') != config['arm']
                or identity.get('checkpoint_sha256') != lock['checkpoint_sha256']
                or identity.get('state_hash') != lock['state_hash']
                or identity.get('model_state_sha256') != test['identity']['model_state_sha256']
                or identity.get('selected_update') != lock['selected_update']):
            raise ValueError('Saved LIB results differ from the locked ' + config['experiment'] + ' checkpoint')
        with np.load(path, allow_pickle=False) as archive:
            data = helper.load_predictions(archive, path)
            if (str(archive['checkpoint_sha256'].item()) != lock['checkpoint_sha256']
                    or int(archive['checkpoint_update']) != lock['selected_update']):
                raise ValueError('LIB prediction provenance differs from current lock')
        predictions[dataset, problem] = data
        fixed = result['train_sbs_reference']
        metrics.append(common.metric_row(problem, dataset, data, fixed, config['baselines'][problem],
            config, lock, path))
        methods.extend(dict(row, model=config['experiment'] + '/' + config['arm']) for row in helper.method_rows(
            problem, dataset, fixed, data, 'original native LIB'))
        results[dataset] = result
    return results


def nss_comparisons(config, lock, predictions, nss_root):
    helper = common.reporting_helpers()
    plan = read_json(nss_root / 'protocol.json')
    if (Path(plan['data_root']).resolve() != Path(config['data_root']).resolve()
            or plan.get('retrained_on_new90k') is not False):
        raise ValueError('NSS reference must be locked historical weights on this new TEST')
    rows, alignment = [], {}
    for dataset, problem, split in (('TSP TEST', 'TSP', 'TEST'), ('CVRP TEST', 'CVRP', 'TEST'),
                                    ('TSPLIB', 'TSP', 'TSPLIB'), ('CVRPLIB', 'CVRP', 'CVRPLIB')):
        current = predictions[split, problem]
        name = 'test' if split == 'TEST' else split
        path = nss_root / problem / (name + '_predictions.npz')
        metadata = read_json(path.with_name(name + '_metrics.json'))
        expected = plan['checkpoints'][problem]['sha256']
        if metadata.get('checkpoint', {}).get('sha256') != expected:
            raise ValueError('NSS reference checkpoint differs from historical lock')
        with np.load(path, allow_pickle=False) as archive:
            nss = helper.load_predictions(archive, path, r77=False)
        checks = helper.check_comparison_alignment(current, nss)
        if split == 'TEST' and nss.n != config['rows_per_problem']['test']:
            raise ValueError('Do not reuse the old3895-row NSS TEST metrics')
        np.testing.assert_allclose(nss.top1_pct, metadata['top1_native_ind'] * 100, rtol=0, atol=1e-9)
        np.testing.assert_allclose(nss.mean_cost, metadata['mean_cost'], rtol=0, atol=1e-9)
        rows.append(dict(dataset=dataset, problem=problem, n=current.n,
            nss_model='R79 locked historical weights; NEW4500 TEST inference; no new90k retraining',
            current_model=config['experiment'] + '/' + config['arm'], nss_top1_native_pct=nss.top1_pct,
            current_top1_native_pct=current.top1_pct, nss_mean_cost=nss.mean_cost,
            current_mean_cost=current.mean_cost, nss_checkpoint_sha256=expected,
            current_checkpoint_sha256=lock['checkpoint_sha256'], winner_rule=WINNER_RULE,
            nss_prediction_path=str(path), current_prediction_path=str(current.path), status='complete'))
        alignment[dataset] = checks
    return rows, alignment


def learning_curves(run, output, model, lock, *, ovr=False):
    import matplotlib.pyplot as plt
    history = read_json(run / 'history.json')
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(model + ' | successful updates; one locked VAL checkpoint', fontsize=15)
    for name in ('loss', 'ce', 'cmp', 'risk', 'kl', 'focal_extra', 'ovr', 'ovr_extra',
                 'cost_KL', 'cost_KL_extra', 'subsetCE', 'subsetCE_extra'):
        points = [(r['successful_updates'], r['optimization'][name]) for r in history
                  if name in r.get('optimization', {})]
        if points:
            label = 'native maximin CE (diagnostic)' if ovr and name == 'ce' else name
            axes[0, 0].plot(*zip(*points), label=label)
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].set_title('TRAIN objectives and native CE diagnostic' if ovr else 'TRAIN interval mean objectives')
    for ax, key, factor, title in ((axes[0, 1], 'macro_top1', 100., 'Native Top1 (%)'),
                                  (axes[1, 0], 'macro_actual_regret_pct', 1., 'Actual regret (%)'),
                                  (axes[1, 1], 'macro_vs_sbs_pct', 1., 'vs fixed TRAIN SBS (%)')):
        for split in ('train', 'val'):
            points = [(r['successful_updates'], r[split]['macro'][key] * factor)
                      for r in history if r.get(split) is not None]
            if points:
                ax.plot(*zip(*points), label=split)
        ax.set_title(title)
        ax.legend()
    for ax in axes.flat:
        ax.axvline(lock['selected_update'], color='#c45358', linestyle='--', linewidth=1)
        ax.set_xlabel('Successful optimizer updates')
        ax.grid(alpha=.2)
    fig.tight_layout(rect=(0, 0, 1, .96))
    fig.savefig(output / 'learning_curves.pdf')
    fig.savefig(output / 'learning_curves.png', dpi=160)
    plt.close(fig)


def build_reports(run, config, lock, test, val, predictions, metrics, methods, comparisons, alignment):
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    helper = common.reporting_helpers()
    helper.setup_style()
    output = run / 'full_model_statistics'
    figures = output / 'figures'
    figures.mkdir(parents=True, exist_ok=True)
    experiment = config['experiment']
    model, version = experiment + '/' + config['arm'], Path(config['data_root']).name
    for name, rows in (('metrics_by_problem.csv', metrics), ('single_method_comparison.csv', methods),
                       ('nss_comparison.csv', comparisons)):
        common.write_csv(output / name, rows)
    panel_sets, distributions = {}, []
    for problem in PROBLEMS:
        data = predictions['TEST', problem]
        dataset = next((name for name, p in LIBS.items() if p == problem), None)
        panels = helper.make_panels(problem, data, predictions.get((dataset, problem)), data.pool)
        panel_sets[problem] = panels
        distributions.extend(dict(row, model=model, checkpoint_sha256=lock['checkpoint_sha256'])
            for row in helper.distribution_rows(problem, panels, data.pool, version))
    common.write_csv(output / 'selection_distribution.csv', distributions)
    pdf, dist_pdf = output / 'MODEL_STATISTICS.pdf', output / 'selection_distributions.pdf'
    pdf_tmp, dist_tmp = output / f'.model.{os.getpid()}.pdf', output / f'.distributions.{os.getpid()}.pdf'
    headings = ['Split / Problem', 'Top1 (%)', 'SBS Top1 (%)', 'mean_cost', 'SBS cost',
                'Oracle cost', 'vs_SBS (%)', 'vs_Oracle (%)', 'actual regret (%)']
    try:
        with PdfPages(pdf_tmp) as report:
            fig = common._page(helper, model + ' | Locked full18 results',
                'Same unique VAL checkpoint on VAL/TEST/LIB; native/FP64 ind; historical NSS weights disclosed.', model, lock, 1)
            fig.text(.045, .85, f'epoch {lock["epoch"]} | update {lock["selected_update"]} | seed2 | fresh initialization', fontsize=12)
            fig.text(.045, .81, 'Data: ' + version, fontsize=10)
            overall = [next(r for r in metrics if r['split'] == split and r['problem'] == 'ALL') for split in ('VAL', 'TEST')]
            helper.add_table(fig, headings, [helper.metric_cells(r, r['split'] + ' ALL') for r in overall],
                [.045, .63, .91, .13], font=10)
            helper.add_table(fig, ['Dataset', 'n', 'NSS Top1 (%)', experiment + ' Top1 (%)', 'NSS cost', experiment + ' cost'],
                [[r['dataset'], str(r['n']), f'{r["nss_top1_native_pct"]:.6f}',
                  f'{r["current_top1_native_pct"]:.6f}', f'{r["nss_mean_cost"]:.9f}',
                  f'{r["current_mean_cost"]:.9f}'] for r in comparisons], [.045, .33, .91, .23], font=11)
            notes = [
                'NSS: R79 weights trained on historical70114 TRAIN, selected using historical VAL; not retrained on new90k.',
                'NSS TEST predictions were freshly computed on these4500 instances; old3895 TEST metrics are not reused.',
                'SBS is fitted on the new81000 TRAIN only. Oracle is the per-instance legal-pool minimum.',
                'ALL is the equal-problem mean; mixed raw cost is descriptive. TEST informed prior research.',
                'Method selection and true winner use separate colors on the same axes; Top1 is per-instance accuracy.',
                'Selection head: ' + config['selection_head'] + ' | objective: ' + str(config.get('objective',
                    config['loss'].get('focal_extra', {})).get('classification',
                    config['loss'].get('focal_extra', {}).get('enabled', False)))]
            for index, text in enumerate(notes):
                fig.text(.045, .27 - index * .035, text, fontsize=10)
            report.savefig(fig)
            plt.close(fig)
            for page, split in enumerate(('VAL', 'TEST'), 2):
                fig = common._page(helper, model + ' | ' + split + ': ALL and18 tasks',
                    'TRAIN-fitted SBS and unchanged native/FP64 costs; no per-task checkpoint assembly.', model, lock, page)
                rows = [r for r in metrics if r['split'] == split]
                helper.add_table(fig, headings + ['TRAIN SBS'],
                    [helper.metric_cells(r, r['problem']) + [r.get('sbs_name', 'per-problem')] for r in rows],
                    [.045, .22, .91, .62], font=10, bold_first=True)
                for index, text in enumerate(('vs_SBS=100*(mean_cost/SBS_cost-1); vs_Oracle=100*(mean_cost/Oracle_cost-1).',
                        'actual regret=mean_i[100*(selected_i-oracle_i)/oracle_i]; not a ratio of means.',
                        'TRAIN81000 / VAL4500 / TEST4500 per problem. All18 problems have equal macro weight.')):
                    fig.text(.045, .17 - index * .04, text, fontsize=10.5)
                report.savefig(fig)
                plt.close(fig)
            fig = common._page(helper, model + ' | LIB and locked provenance',
                'Current ' + experiment + ' factory, same buffers and selected checkpoint; no solver execution or LIB tuning.', model, lock, 4)
            lib_rows = [r for r in metrics if r['split'] in LIBS]
            helper.add_table(fig, headings, [helper.metric_cells(r, r['split']) for r in lib_rows], [.045, .69, .91, .15], font=10)
            fig.text(.045, .62, 'SHA256: ' + lock['checkpoint_sha256'], fontsize=10)
            fig.text(.045, .57, 'Unique selection: ' + lock['rule'], fontsize=10)
            fig.text(.045, .52, 'Inference: ' + config['inference_rule'], fontsize=10)
            fig.text(.045, .47, 'Normalization: new81000 TRAIN only; no TEST/LIB refit.', fontsize=11)
            report.savefig(fig)
            plt.close(fig)
            with PdfPages(dist_tmp) as distribution:
                for problem in PROBLEMS:
                    fig = common._distribution(helper, problem, predictions['TEST', problem].pool,
                        panel_sets[problem], version, model, lock)
                    fig.savefig(figures / (problem + '_selection_distribution.pdf'))
                    fig.savefig(figures / (problem + '_selection_distribution.png'), dpi=140)
                    report.savefig(fig)
                    distribution.savefig(fig)
                    plt.close(fig)
        pdf_tmp.replace(pdf)
        dist_tmp.replace(dist_pdf)
    finally:
        pdf_tmp.unlink(missing_ok=True)
        dist_tmp.unlink(missing_ok=True)
    ovr = config.get('objective', {}).get('classification') == 'OVR BCE on raw utility'
    learning_curves(run, output, model, lock, ovr=ovr)
    diagnostic_options = dict(loss_labels={'ce': 'native maximin CE (diagnostic)'}) if ovr else {}
    diagnostics = common.diagnostic_pdf(run, output, predictions, methods, model, lock, **diagnostic_options)
    write_json(output / 'protocol.json', dict(experiment=experiment, arm=config['arm'], complete=True,
        created_utc=utc(), checkpoint_lock=lock, model_factory=config.get('model_factory', FACTORY + '.load_checkpoint'),
        data_root=config['data_root'], rows_per_problem=config['rows_per_problem'],
        candidate_pools=config['candidate_pools'], scenario_solver_aliases=config.get('scenario_solver_aliases', {}),
        winner_rule=WINNER_RULE, nss_reference='R79 historical weights; fresh new4500 TEST inference; existing native LIB predictions',
        nss_alignment=alignment, training=False, inference=False, checkpoint_reselection=False,
        distribution_pages=18, main_pdf_pages=22, diagnostics=diagnostics))
    overall = {split: next(r for r in metrics if r['split'] == split and r['problem'] == 'ALL') for split in ('VAL', 'TEST')}
    lines = ['# ' + model, '', 'Unique VAL checkpoint: update ' + str(lock['selected_update']), '',
             '| Split | Top1 (%) | mean_cost | actual regret (%) |', '| --- | ---: | ---: | ---: |']
    for split, row in overall.items():
        lines.append(f'| {split} | {row["top1_pct"]:.6f} | {row["mean_cost"]:.9f} | {row["actual_regret_pct"]:.6f} |')
    lines += ['', 'SBS was fitted on new81000 TRAIN. Native FP64 winner labels; equal task weights.',
              'NSS uses historical R79 weights, not a matched new90k training control.',
              'No checkpoint, temperature, or per-problem assembly was selected using TEST/LIB.',
              'PDF: MODEL_STATISTICS.pdf; distributions: selection_distributions.pdf.', '']
    (output / 'RESULTS.md').write_text('\n'.join(lines))
    return dict(complete=True, output=str(output), model_statistics_pdf=str(pdf), selection_distributions_pdf=str(dist_pdf))


def report(args):
    root = args.root.resolve()
    run, config, lock, test, val = locked_inputs(root, args.arm)
    with (run / 'postprocess.lock').open('a') as claim:
        fcntl.flock(claim, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            write_json(run / 'postprocess_status.json', dict(phase='loading_saved_predictions', updated_utc=utc()))
            predictions, metrics, methods = load_saved_splits(run, config, lock, test, val)
            saved_libs(run, config, lock, test, predictions, metrics, methods)
            comparisons, alignment = nss_comparisons(config, lock, predictions, args.nss_root)
            result = build_reports(run, config, lock, test, val, predictions, metrics, methods, comparisons, alignment)
            write_json(run / 'postprocess_status.json', dict(phase='complete', completed_utc=utc(), **result))
            return result
        except BaseException as error:
            write_json(run / 'postprocess_status.json', dict(phase='postprocess_failed', updated_utc=utc(),
                error=str(error), traceback=traceback.format_exc(), retry='report only; retain training/TEST/LIB'))
            raise


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('command', choices=('lib', 'report', 'status'))
    result.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    result.add_argument('--arm', choices=ARMS, required=True)
    result.add_argument('--device', default='cpu')
    result.add_argument('--lib-batch-size', type=int, choices=(1, 2), default=2)
    result.add_argument('--nss-root', type=Path)
    result.add_argument('--r79-root', type=Path, default=DEFAULT_R79)
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    args.root = args.root.resolve()
    args.nss_root = (args.nss_root or args.root / 'nss_reference').resolve()
    if args.command == 'status':
        path = args.root / args.arm / 'postprocess_status.json'
        result = read_json(path) if path.exists() else dict(phase='not_started')
        print(json.dumps(result, indent=2))
        return result
    return lib_stage(args) if args.command == 'lib' else report(args)


if __name__ == '__main__':
    main()

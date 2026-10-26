"""R113 same-checkpoint LIB and offline PDF/CSV; no new NSS training."""

import argparse
import copy
import csv
import fcntl
import json
import math
import os
from pathlib import Path

import numpy as np
import torch

from . import r80_reporting, r84_reporting
from . import r113_experiment as runner
from .r76_experiment import file_hash, read_json, utc, write_json
from .r77_data import parameter_hash


ARMS, BUDGETS = runner.ARMS, runner.BUDGETS
DEFAULT_ROOT, FACTORY = runner.DEFAULT_ROOT, 'code.V4.r113_experiment'
PROBLEMS = tuple(runner.PROBLEMS)
common = runner.private_module(r80_reporting, 'r113_report_common')
shared = runner.private_module(r84_reporting, 'r113_report_shared')
shared.common = common
for module in (common, shared):
    module.DEFAULT_ROOT, module.ARMS, module.BUDGETS, module.FACTORY = DEFAULT_ROOT, ARMS, BUDGETS, FACTORY
LIBS = shared.LIBS
NSS_DATASETS = (('TSP TEST', 'TSP', 'TEST'), ('CVRP TEST', 'CVRP', 'TEST'),
                ('TSPLIB', 'TSP', 'TSPLIB'), ('CVRPLIB', 'CVRP', 'CVRPLIB'))


def write_csv(path, rows):
    if not rows:
        return
    path = Path(path)
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    try:
        with temporary.open('w', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


common.write_csv = write_csv


def locked_inputs(root, arm):
    run, config, lock, test, val = shared.locked_inputs(Path(root), arm, experiment='R113', arms=BUDGETS)
    runner.validate_checkpoint_config(config)
    state = torch.load(run / 'best.pt', map_location='cpu', weights_only=False)
    final = read_json(run / 'final_metrics.json')
    if (final.get('complete') is not True or final.get('test') != test
            or test['identity'].get('model_state_sha256') != parameter_hash(state['model'])
            or test['identity'].get('checkpoint_sha256') != lock['checkpoint_sha256']):
        raise ValueError('R113 report requires completed same-model full TEST')
    val = copy.deepcopy(val)
    val['identity'] = runner.selected_capture_proof(run, val, lock, 'val', state)
    runner.selected_capture_proof(run, final['train'], lock, 'train', state)
    return run, config, lock, test, val


def inference_timing(run, config, lock, model_hash):
    value = read_json(Path(run) / 'inference_timing.json')
    expected = dict(experiment='R113', arm=config['arm'], complete=True,
        checkpoint_sha256=lock['checkpoint_sha256'], state_hash=lock['state_hash'],
        selected_update=lock['selected_update'], model_state_sha256=model_hash,
        model_factory=runner.MODEL_FACTORY, inference_rule=runner.INFERENCE_RULE,
        parameter_count=config['trainable_parameter_count'], eval_batch_size=config['eval_batch_size'],
        repetitions=3, warmup_batches=1, all_current_model_included=True,
        scope='complete_standalone_model_original_GPU_resident_inputs',
        sample_policy=f'first{config["eval_batch_size"]} frozen TEST inputs per problem',
        end_to_end_measured=False, input_labels_read=False, model_reselection=False,
        excludes='dataset I/O, CPU-to-GPU transfer, training and solver execution')
    if any(value.get(k) != v for k, v in expected.items()) or not str(value.get('device')).startswith('cuda'):
        raise ValueError('R113 timing must cover the complete same-checkpoint input-only model')
    rows, count = value.get('per_problem', []), config['eval_batch_size']
    if len(rows) != len(PROBLEMS) or {r['problem'] for r in rows} != set(PROBLEMS):
        raise ValueError('R113 timing must cover all18 tasks')
    for row in rows:
        samples = row.get('standalone_model', {}).get('batch_seconds', [])
        mean = row.get('standalone_model', {}).get('mean_seconds_per_instance')
        if (row.get('instances') != count or len(samples) != 3
                or any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in [*samples, mean])
                or not math.isclose(mean, sum(samples) / (3 * count), rel_tol=1e-10)):
            raise ValueError('Invalid R113 timing denominator/sample count')
    return value


def lib_stage(args):
    run, config, lock, test, _ = locked_inputs(args.root, args.arm)
    model_hash = test['identity']['model_state_sha256']
    inference_timing(run, config, lock, model_hash)
    with (run / 'lib_stage.lock').open('a') as claim:
        fcntl.flock(claim, fcntl.LOCK_EX | fcntl.LOCK_NB)

        def phase(name, **fields):
            write_json(run / 'lib_stage_status.json', dict(experiment='R113', arm=args.arm,
                phase=name, updated_utc=utc(), **fields))

        batch = min(config['eval_batch_size'], config['microbatch_caps']['TSP'],
            config['microbatch_caps']['CVRP']) if args.lib_batch_size is None else args.lib_batch_size
        if batch < 1:
            raise ValueError('LIB batch size must be positive')
        results = common.r80_lib(args.root, run, config, lock, args.device, batch, phase,
            model_hash, factory_module=FACTORY)
        if file_hash(run / 'best.pt') != lock['checkpoint_sha256']:
            raise ValueError('R113 checkpoint changed during LIB inference')
        write_json(run / 'lib_stage_complete.json', dict(experiment='R113', arm=args.arm,
            complete=True, completed_utc=utc(), checkpoint_sha256=lock['checkpoint_sha256'],
            state_hash=lock['state_hash'], selected_update=lock['selected_update'],
            model_factory=runner.MODEL_FACTORY, model_state_sha256=model_hash, datasets=list(results)))
        phase('complete')
        return results


def nss_comparisons(config, lock, predictions, nss_root):
    helper, rows, alignment = common.reporting_helpers(), [], {}
    root = Path(nss_root)
    plan = read_json(root / 'protocol.json') if (root / 'protocol.json').is_file() else None
    if plan is not None and (Path(plan['data_root']).resolve() != Path(config['data_root']).resolve()
                             or plan.get('retrained_on_new90k') is not False):
        raise ValueError('NSS must be saved historical weights on the same evaluation data')
    for dataset, problem, split in NSS_DATASETS:
        current = predictions[split, problem]
        name = 'test' if split == 'TEST' else split
        path = root / problem / (name + '_predictions.npz')
        row = dict(dataset=dataset, problem=problem, n=current.n,
            current_model='R113/' + config['arm'], current_model_factory=runner.MODEL_FACTORY,
            current_top1_native_pct=current.top1_pct, current_mean_cost=current.mean_cost,
            current_checkpoint_sha256=lock['checkpoint_sha256'], current_state_hash=lock['state_hash'],
            current_selected_update=lock['selected_update'], current_prediction_path=str(current.path),
            nss_model='R79 saved historical weights; not retrained on new90k',
            data_version=Path(config['data_root']).name if split == 'TEST' else 'original native ' + dataset,
            candidate_pool=json.dumps(list(current.pool)), winner_rule=runner.WINNER_RULE)
        try:
            if plan is None or problem not in plan.get('checkpoints', {}):
                raise FileNotFoundError('Saved NSS protocol/checkpoint lock missing: ' + problem)
            metadata = read_json(path.with_name(name + '_metrics.json'))
            expected = plan['checkpoints'][problem]['sha256']
            if metadata.get('checkpoint', {}).get('sha256') != expected:
                raise ValueError('NSS reference checkpoint differs: ' + dataset)
            with np.load(path, allow_pickle=False) as archive:
                nss = helper.load_predictions(archive, path, r77=False)
            checks = helper.check_comparison_alignment(current, nss)
            np.testing.assert_allclose(nss.top1_pct, metadata['top1_native_ind'] * 100, rtol=0, atol=1e-9)
            np.testing.assert_allclose(nss.mean_cost, metadata['mean_cost'], rtol=0, atol=1e-9)
        except FileNotFoundError as error:
            row.update(status='missing', nss_top1_native_pct=None, nss_mean_cost=None,
                nss_checkpoint_sha256=None, nss_prediction_path=str(path), missing_reason=str(error))
            alignment[dataset] = dict(status='missing', reason=str(error))
        else:
            row.update(status='complete', nss_top1_native_pct=nss.top1_pct, nss_mean_cost=nss.mean_cost,
                nss_checkpoint_sha256=expected, nss_prediction_path=str(path))
            alignment[dataset] = checks
        rows.append(row)
    return rows, alignment


def display(value, precision):
    return 'MISSING' if value is None else format(value, '.' + str(precision) + 'f')


def build_reports(run, config, lock, test, val, predictions, metrics, methods, comparisons, alignment, timing):
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from .r113_diagnostics import append_pdf_pages, saved_diagnostics
    helper = common.reporting_helpers()
    helper.setup_style()
    process_evidence = saved_diagnostics(run, config, lock)
    process_summary = {k: v for k, v in process_evidence.items() if k != 'value'}
    output, model = run / 'full_model_statistics', 'R113/' + config['arm']
    figures, version = output / 'figures', Path(config['data_root']).name
    figures.mkdir(parents=True, exist_ok=True)
    for name, rows in (('metrics_by_problem.csv', metrics), ('single_method_comparison.csv', methods),
                       ('nss_comparison.csv', comparisons)):
        write_csv(output / name, rows)
    panel_sets, distributions = {}, []
    for problem in PROBLEMS:
        current = predictions['TEST', problem]
        dataset = next((name for name, p in LIBS.items() if p == problem), None)
        panels = helper.make_panels(problem, current, predictions.get((dataset, problem)), current.pool)
        panel_sets[problem] = panels
        distributions.extend(dict(row, model=model, checkpoint_sha256=lock['checkpoint_sha256'])
            for row in helper.distribution_rows(problem, panels, current.pool, version))
    write_csv(output / 'selection_distribution.csv', distributions)
    pdf, distribution_pdf = output / 'MODEL_STATISTICS.pdf', output / 'selection_distributions.pdf'
    temporary = output / f'.model.{os.getpid()}.pdf'
    dist_temporary = output / f'.distribution.{os.getpid()}.pdf'
    headings = ['Split / Problem', 'Top1 (%)', 'SBS Top1 (%)', 'mean_cost', 'SBS cost',
                'Oracle cost', 'vs_SBS (%)', 'vs_Oracle (%)', 'actual regret (%)']
    try:
        with PdfPages(temporary) as report:
            fig = common._page(helper, model + ' | Locked full18 results',
                'Unique VAL-selected checkpoint; native/FP64 ind; equal-problem ALL.', model, lock, 1)
            fig.text(.045, .85, f'update {lock["selected_update"]} | seed2 | parameters {config["trainable_parameter_count"]:,}', fontsize=12)
            fig.text(.045, .81, 'Data: ' + version, fontsize=10)
            overall = [next(r for r in metrics if r['split'] == s and r['problem'] == 'ALL') for s in ('VAL', 'TEST')]
            helper.add_table(fig, headings, [helper.metric_cells(r, r['split'] + ' ALL') for r in overall],
                [.045, .63, .91, .13], font=10)
            helper.add_table(fig, ['Dataset', 'n', 'NSS Top1 (%)', 'R113 Top1 (%)', 'NSS cost', 'R113 cost'],
                [[r['dataset'], str(r['n']), display(r.get('nss_top1_native_pct'), 6),
                  display(r['current_top1_native_pct'], 6), display(r.get('nss_mean_cost'), 9),
                  display(r['current_mean_cost'], 9)] for r in comparisons], [.045, .33, .91, .23], font=11)
            notes = ['NSS: saved R79 weights; no new training. Missing same-population/native references remain MISSING.',
                'SBS: minimum mean cost on frozen81000 TRAIN; Oracle: per-instance legal-pool minimum.',
                'ALL: equal problem weight; raw-cost average is descriptive, never a pooled optimization score.',
                'TEST was development-exposed; only VAL chose this checkpoint. No per-problem assembly.',
                'Distribution Top1 is instance accuracy, not the largest method frequency.',
                'Same final recursive architecture; A scratch, B snapshots, C predicted transition pretraining.',
                'Post-lock process diagnostics: ' + process_evidence['status'].upper()
                + ('; saved PDF/CSV/JSON included below.' if process_evidence['status'] == 'complete'
                   else '; no process-generalization evidence claimed.')]
            for index, note in enumerate(notes):
                fig.text(.045, .27 - index * .035, note, fontsize=10)
            report.savefig(fig)
            plt.close(fig)
            for page, split in enumerate(('VAL', 'TEST'), 2):
                fig = common._page(helper, model + ' | ' + split + ': ALL and18 tasks',
                    'Native/FP64 winners, frozen pools, TRAIN-fitted SBS.', model, lock, page)
                rows = [r for r in metrics if r['split'] == split]
                helper.add_table(fig, headings + ['TRAIN SBS'],
                    [helper.metric_cells(r, r['problem']) + [r.get('sbs_name', 'per-problem')] for r in rows],
                    [.045, .22, .91, .62], font=10, bold_first=True)
                for i, note in enumerate(('vs_SBS / vs_Oracle:100*(ratio of mean costs-1).',
                    'actual regret: mean_i[100*(selected_i-oracle_i)/oracle_i], not a ratio of means.',
                    'TRAIN81000 / VAL4500 / TEST4500 per task; no label/cost in query forward.')):
                    fig.text(.045, .17 - i * .04, note, fontsize=10.5)
                report.savefig(fig)
                plt.close(fig)
            fig = common._page(helper, model + ' | LIB and provenance',
                'Same checkpoint; no LIB selection or solver execution.', model, lock, 4)
            rows = [r for r in metrics if r['split'] in LIBS]
            helper.add_table(fig, headings, [helper.metric_cells(r, r['split']) for r in rows], [.045, .69, .91, .15], font=10)
            notes = ['Checkpoint SHA256: ' + lock['checkpoint_sha256'],
                'VAL rule: maximum macroTop1; exact ties minimum actual regret; then first.',
                'Selector:512 x57600 successful updates; AdamW1e-4 to1e-6; warmup4320.',
                'Pretraining B/C:512 x8640; AdamW3e-4 to1e-5; warmup432; same targets/order.',
                'All deployment: self-initialized predicted-state feedback, transition mode.',
                'Candidates read global/node trajectory and final directed edge-state tokens.',
                'Raw two-layer stem enters processor context only; no old-score bypass.',
                'Fresh state SHA256: ' + config['fresh_state_sha256']]
            for i, note in enumerate(notes):
                fig.text(.045, .62 - i * .045, note, fontsize=10)
            pretrain = config.get('processor_pretraining')
            if pretrain:
                fig.text(.045, .22, 'Process manifest: ' + pretrain['dataset']['manifest_sha256'], fontsize=9)
                fig.text(.045, .18, 'Pretraining mode: ' + pretrain['mode'] + '; processor-only transfer; newly authorized TRAIN states.', fontsize=10)
                fig.text(.045, .14, 'Last TRAIN process losses are optimization evidence only, not generalization or selector gain.', fontsize=10)
            report.savefig(fig)
            plt.close(fig)
            process_pages = append_pdf_pages(report, process_evidence)
            with PdfPages(dist_temporary) as distribution:
                for problem in PROBLEMS:
                    fig = common._distribution(helper, problem, predictions['TEST', problem].pool,
                        panel_sets[problem], version, model, lock)
                    fig.savefig(figures / (problem + '_selection_distribution.pdf'))
                    fig.savefig(figures / (problem + '_selection_distribution.png'), dpi=140)
                    report.savefig(fig)
                    distribution.savefig(fig)
                    plt.close(fig)
        temporary.replace(pdf)
        dist_temporary.replace(distribution_pdf)
    finally:
        temporary.unlink(missing_ok=True)
        dist_temporary.unlink(missing_ok=True)
    shared.learning_curves(run, output, model, lock)
    diagnostics = common.diagnostic_pdf(run, output, predictions, methods, model, lock)
    write_csv(output / 'inference_timing.csv', [dict(problem=r['problem'], instances=r['instances'],
        milliseconds_per_instance=1000 * r['standalone_model']['mean_seconds_per_instance']) for r in timing['per_problem']])
    write_json(output / 'protocol.json', dict(experiment='R113', arm=config['arm'], complete=True,
        created_utc=utc(), checkpoint_lock=lock, model_factory=runner.MODEL_FACTORY,
        main_protocol_lock=config['main_protocol_lock'], model_params=config['model_params'],
        trainable_parameter_count=config['trainable_parameter_count'], fresh_state_sha256=config['fresh_state_sha256'],
        processor_pretraining=config.get('processor_pretraining'), data_root=config['data_root'],
        rows_per_problem=config['rows_per_problem'], candidate_pools=config['candidate_pools'],
        scenario_solver_aliases=config.get('scenario_solver_aliases', {}), winner_rule=runner.WINNER_RULE,
        checkpoint_rule=runner.CHECKPOINT_RULE, nss_reference=comparisons, nss_alignment=alignment,
        inference_timing=timing, training=False, inference=False, checkpoint_reselection=False,
        distribution_pages=18, main_pdf_pages=22 + process_pages, diagnostics=diagnostics,
        process_diagnostics=process_summary,
        evidence_boundary='development-exposed TEST; unique VAL checkpoint; ALL problem-equal; TRAIN SBS; no NSS retraining'))
    overall = {s: next(r for r in metrics if r['split'] == s and r['problem'] == 'ALL') for s in ('VAL', 'TEST')}
    lines = ['# ' + model, '', 'Unique VAL checkpoint: update ' + str(lock['selected_update']), '',
        '| Split | Top1 (%) | mean_cost | actual regret (%) |', '| --- | ---: | ---: | ---: |']
    lines += [f'| {s} | {r["top1_pct"]:.6f} | {r["mean_cost"]:.9f} | {r["actual_regret_pct"]:.6f} |' for s, r in overall.items()]
    lines += ['', 'Native FP64 winners; problem-equal ALL; TRAIN-fitted SBS. TEST was development-exposed.',
        'NSS references are saved historical weights, not newly trained controls; missing entries remain MISSING.',
        'PDF: MODEL_STATISTICS.pdf; paired selection/winner bars: selection_distributions.pdf.', '']
    (output / 'RESULTS.md').write_text('\n'.join(lines))
    return dict(complete=True, output=str(output), model_statistics_pdf=str(pdf),
        selection_distributions_pdf=str(distribution_pdf), process_diagnostics=process_summary)


def report(args):
    """Saved predictions only. No model construction, external pretrain, or tuning."""
    run, config, lock, test, val = locked_inputs(args.root, args.arm)
    with (run / 'report.lock').open('a') as claim:
        fcntl.flock(claim, fcntl.LOCK_EX | fcntl.LOCK_NB)
        lib = read_json(run / 'lib_stage_complete.json')
        expected = dict(experiment='R113', arm=args.arm, complete=True,
            checkpoint_sha256=lock['checkpoint_sha256'], state_hash=lock['state_hash'],
            selected_update=lock['selected_update'], model_factory=runner.MODEL_FACTORY,
            model_state_sha256=test['identity']['model_state_sha256'])
        if any(lib.get(k) != v for k, v in expected.items()) or set(lib.get('datasets', [])) != set(LIBS):
            raise ValueError('R113 PDF requires both LIBs from the unique selected model')
        timing = inference_timing(run, config, lock, test['identity']['model_state_sha256'])
        predictions, metrics, methods = shared.load_saved_splits(run, config, lock, test, val)
        shared.saved_libs(run, config, lock, test, predictions, metrics, methods)
        training, train_rows = read_json(run / 'final_metrics.json')['train'], []
        helper, path = common.reporting_helpers(), Path(training['prediction_archive'])
        with np.load(path, allow_pickle=False) as archive:
            for problem in PROBLEMS:
                value = helper.load_predictions(archive, path, problem + '__')
                shared.validate_population(value, problem, 'TRAIN', config)
                train_rows.append(common.metric_row(problem, 'TRAIN', value, training['per_problem'][problem],
                    config['baselines'][problem], config, lock, path))
        metrics = [common.macro_row('TRAIN', train_rows, training), *train_rows, *metrics]
        comparisons, alignment = nss_comparisons(config, lock, predictions, args.nss_root)
        result = build_reports(run, config, lock, test, val, predictions, metrics, methods, comparisons, alignment, timing)
        write_json(run / 'report_complete.json', dict(result, experiment='R113', arm=args.arm, complete=True,
            checkpoint_sha256=lock['checkpoint_sha256'], state_hash=lock['state_hash'],
            selected_update=lock['selected_update'], completed_utc=utc()))
        return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('lib', 'report'))
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--arm', choices=ARMS, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--lib-batch-size', type=int)
    parser.add_argument('--nss-root', type=Path, default=runner.V4 / 'runs/R84_new90k_attention/nss_reference')
    args = parser.parse_args(argv)
    args.root, args.nss_root = args.root.expanduser().resolve(), args.nss_root.expanduser().resolve()
    return lib_stage(args) if args.command == 'lib' else report(args)


if __name__ == '__main__':
    main()

"""R107 frozen-factory LIB and saved-evidence full18 PDF/CSV reporting."""

import argparse
import copy
import fcntl
import importlib
import importlib.util
import json
import math
import os
import sys
import textwrap
from pathlib import Path

import numpy as np

from . import r80_reporting, r84_reporting, r85_reporting, r87_reporting
from .r76_experiment import file_hash, read_json, utc, write_json
from .r77_data import parameter_hash


ARMS = ('A_edge_factorized', 'B_connected_paths')
BUDGETS = dict.fromkeys(ARMS, 57600)
DEFAULT_ROOT = Path(__file__).resolve().parent / 'runs/R107_explicit_path_relations'
FACTORY = 'code.V4.r107_experiment'
PROBLEMS = tuple(r84_reporting.PROBLEMS)
DEFAULT_R84, DEFAULT_NSS = r87_reporting.DEFAULT_R84, r87_reporting.DEFAULT_NSS
DEFAULT_R89 = DEFAULT_ROOT.parent / 'R89_optimizer_geometry'
REQUIRED_PDFS = ('MODEL_STATISTICS.pdf', 'selection_distributions.pdf', 'learning_curves.pdf',
                 'training_and_method_diagnostics.pdf', 'target_recovery_and_train.pdf')
ARM_DESCRIPTIONS = dict(
    A_edge_factorized='factorized edge/path evidence',
    B_connected_paths='joint connected short-path evidence')
BOUNDARY = ('R107 adds an explicit short-path module to the fresh R84/R88B full18 selector. '
    'A_edge_factorized uses factorized evidence; B_connected_paths uses joint connected-path '
    'computation. This is not a MoE FFN or learned expert-routing experiment. Same complete '
    'fresh tensors and total parameter counts require actual factory/initialization proof; '
    'they do not imply equal active functional freedom or equal FLOPs. Actual counts and '
    'path specification come from the locked model, never estimates. Factorization is '
    'conditional on each round incoming node states, not global selector additivity. '
    'A retains trunk interactions; B also receives explicit composed fields, so this is not '
    'an isolated topology-only causal comparison. Original R88B physical '
    'TSP node0 role, methodID+34, geometry/distance, score MLP/maximin and original '
    'CE/comparison/risk/two-forward RDropObjective remain; 57,600 updates, logical batch1024. '
    'No donor, teacher, context labels, new targets, calibration, quota or auxiliary loss. '
    'R89A AdamW, R89B Muon and R84A are saved historical references, not rerun or newly '
    'parameter/FLOP-matched controls; R94B is optional, never a prerequisite. Fixed36 rescues, '
    'losses, false picks and aligned net costs/regret are separate from choice frequency. '
    'Native/FP64 ind and legal pools; TRAIN-fitted SBS; ALL task-equal. TEST is '
    'development-exposed; one whole-model full18 VAL checkpoint for full TRAIN/VAL/TEST/LIB; '
    'no TEST/LIB reselection. Actual standalone GPU-resident-input timing includes complete '
    'model graph/path construction and computation, excludes dataset I/O, CPU transfer and '
    'solvers, and is not end-to-end dataset latency. No routing-summary requirement. '
    'No online W&B equality/availability is asserted.')


def _runner():
    """Import the real factory only when needed, without a historical fallback."""
    module = importlib.import_module(FACTORY)
    if (tuple(module.ARMS) != ARMS or module.BUDGETS != BUDGETS
            or module.MODEL_FACTORY != FACTORY + '.load_checkpoint'):
        raise ValueError('R107 reporting and experiment interfaces differ')
    return module


def _private_module(source, name):
    spec = importlib.util.spec_from_file_location('code.V4._r107_' + name, source.__file__)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# Function globals belong to these private copies, never the historical modules.
common = _private_module(r80_reporting, 'report_common')
shared = _private_module(r84_reporting, 'report_shared')
previous = _private_module(r85_reporting, 'report_previous')
helpers = _private_module(r87_reporting, 'report_helpers')
shared.common = common
previous.shared, previous.common = shared, common
helpers.shared, helpers.previous, helpers.common = shared, previous, common
LIBS = shared.LIBS
for module in (common, shared, previous, helpers):
    module.DEFAULT_ROOT, module.ARMS, module.BUDGETS, module.FACTORY = DEFAULT_ROOT, ARMS, BUDGETS, FACTORY
_template_selected_train = previous.selected_train


def selected_train(run, config, lock):
    result, path = _template_selected_train(run, config, lock)
    if config['experiment'] == 'R107':
        _runner().selected_capture_proof(run, result, lock, 'train')
    return result, path


previous.selected_train = selected_train


def capture_val(val, lock):
    """Use preserved capture identity only after checking the official binding."""
    binding = val.get('checkpoint_binding', {})
    expected = dict(rule=_runner().VAL_BINDING_RULE, checkpoint_sha256=lock['checkpoint_sha256'],
        state_hash=lock['state_hash'], selected_update=lock['selected_update'], archive_identity_unchanged=True)
    capture, identity = val.get('archive_capture_identity'), val.get('identity', {})
    if (not isinstance(capture, dict) or any(binding.get(k) != v for k, v in expected.items())
            or capture.get('experiment') != 'R107' or capture.get('arm') != lock['arm']
            or capture.get('split') != 'val'
            or capture.get('checkpoint_sha256') not in (None, lock['checkpoint_sha256'])
            or binding.get('archive_capture_checkpoint_sha256') != capture.get('checkpoint_sha256')
            or identity != dict(capture, checkpoint_sha256=lock['checkpoint_sha256'])
            or capture.get('state_hash') != lock['state_hash']
            or capture.get('successful_updates') != lock['selected_update']
            or not capture.get('model_state_sha256')
            or capture['model_state_sha256'] != binding.get('model_state_sha256')):
        raise ValueError('R107 selected VAL summary lacks its exact-state official binding')
    result = copy.deepcopy(val)
    result['identity'] = copy.deepcopy(capture)
    return result


def locked_inputs(root, arm):
    run, config, lock, test, val = shared.locked_inputs(Path(root), arm, experiment='R107', arms=BUDGETS)
    runner = _runner()
    model, state = runner.load_checkpoint(run / 'best.pt', 'cpu')
    if (state['args'] != config or state['successful_updates'] != lock['selected_update']
            or state.get('epoch') != lock['epoch'] or config['model_factory'] != runner.MODEL_FACTORY
            or sum(p.numel() for p in model.parameters() if p.requires_grad)
                != config['trainable_parameter_count']
            or config['trainable_parameter_count'] != runner.PARAMETER_COUNTS[arm]):
        raise ValueError('R107 reporter factory differs from the unique per-arm model lock')
    model_hash = parameter_hash(model.state_dict())
    _, binding = runner.selected_capture_proof(run, val, lock, 'val', state)
    if binding != val.get('checkpoint_binding'):
        raise ValueError('R107 official VAL binding differs from the actual factory/checkpoint proof')
    val = capture_val(val, lock)
    if model_hash != test['identity']['model_state_sha256'] or model_hash != val['identity']['model_state_sha256']:
        raise ValueError('R107 VAL/TEST/frozen-factory model states differ')
    del model, state
    return run, config, lock, test, val


def same_data(left, right):
    for key in ('data_root', 'rows_per_problem', 'candidate_pools', 'candidate_pool_ids', 'baselines', 'winner_rule'):
        if key not in left or left[key] != right.get(key):
            raise ValueError('R107 comparison requires identical ' + key)


def selected_predictions(run, config, lock, val, test):
    packed, methods, summaries, results, paths = helpers.selected_predictions(run, config, lock, val, test)
    name = config['experiment'] + '/' + config['arm']
    return packed, [dict(r, model=name) for r in methods], [dict(r, model=name) for r in summaries], results, paths


def historical_reference(config, name, path, experiment, budgets, optimizer):
    path = Path(path)
    run, old, lock, test, val = shared.locked_inputs(path.parent, path.name, experiment=experiment, arms=budgets)
    same_data(config, old)
    return dict(run=run, config=old, lock=lock, test=test, val=val,
        provenance=dict(model=name, run=str(run), checkpoint_sha256=lock['checkpoint_sha256'],
            same_data=True, retrained=False, newly_matched_control=False, optimizer=optimizer,
            parameter_matched=False, equal_flops=False, identical_full_initialization=False))


def references(config, args):
    result = {}
    for name, path, experiment, budgets, optimizer in (
            ('R89A', args.r89a_root, 'R89', dict(A_adamw=57600, B_muon_hybrid=57600), 'AdamW'),
            ('R89B', args.r89_root, 'R89', dict(A_adamw=57600, B_muon_hybrid=57600), 'Muon hybrid'),
            ('R84A', args.historical_root, 'R84', dict(A_baseline=57600), 'AdamW')):
        result[name] = historical_reference(config, name, path, experiment, budgets, optimizer)
    return result


def load_targets(config, old_config, old_lock):
    targets, identity = helpers.load_targets(config, old_config, old_lock)
    if len(targets) != 36:
        raise ValueError('R107 retains frozen R84 target36 without reselection')
    return targets, dict(identity, target_count=36, sha256=file_hash(Path(identity['path'])))


def comparison_tables(old, new, targets, identity, model, baseline):
    pairs, recovery, totals = helpers.comparison_tables(old, new, targets, identity, model, baseline)
    for row in recovery:
        left, right = old[row['split']][row['problem']], new[row['split']][row['problem']]
        slot, index = right.pool.index(row['solver']), np.arange(right.n)
        truth, was, chosen = right.winner == slot, left.pred == slot, right.pred == slot
        incoming, outgoing = ~truth & ~was & chosen, ~truth & was & ~chosen
        delta = right.costs[index, right.pred] - left.costs[index, left.pred]
        row.update(net_rescues=row['rescued'] - row['lost'],
            new_false_pick_cost_delta_contribution=float(delta[incoming].sum() / right.n),
            removed_false_pick_count=int(outgoing.sum()),
            removed_false_pick_cost_delta_contribution=float(delta[outgoing].sum() / right.n),
            target_winner_cost_delta_contribution=float(delta[truth].sum() / right.n),
            full_problem_mean_cost_delta=float(delta.mean()))
    for row in totals:
        row['net_rescues'] = row['rescued'] - row['lost']
        for key in ('new_false_pick_cost_delta_contribution', 'removed_false_pick_cost_delta_contribution',
                    'target_winner_cost_delta_contribution'):
            row[key] = sum(r[key] for r in recovery if r['split'] == row['split']) / len(PROBLEMS)
        row['full18_net_mean_cost_delta'] = next(r['mean_cost_delta'] for r in pairs
            if r['split'] == row['split'] and r['problem'] == 'ALL')
    return pairs, recovery, totals


def diagnostic_tables(run, config, lock, val, test, args):
    packed, methods, summaries, results, paths = selected_predictions(run, config, lock, val, test)
    refs = references(config, args)
    targets, identity = load_targets(config, refs['R84A']['config'], refs['R84A']['lock'])
    pairs, recovery, totals = {}, {}, {}
    for name, ref in refs.items():
        old = selected_predictions(ref['run'], ref['config'], ref['lock'], ref['val'], ref['test'])[0]
        pairs[name], recovery[name], totals[name] = comparison_tables(old, packed, targets, identity,
            'R107/' + config['arm'], name)
    optional = dict(R94B=dict(required=False, status='not_requested'))
    if args.r94b_root is not None:
        optional['R94B']['run'] = str(args.r94b_root)
        try:
            ref = historical_reference(config, 'R94B', args.r94b_root, 'R94', dict(A=57600, B=57600), 'AdamW')
            old = selected_predictions(ref['run'], ref['config'], ref['lock'], ref['val'], ref['test'])[0]
            paired, recovered, total = comparison_tables(old, packed, targets, identity,
                'R107/' + config['arm'], 'R94B')
        except Exception as error:
            optional['R94B'].update(status='unavailable', error_type=type(error).__name__, error=str(error))
        else:
            refs['R94B'] = ref
            pairs['R94B'], recovery['R94B'], totals['R94B'] = paired, recovered, total
            optional['R94B']['status'] = 'included'
    return dict(predictions=packed, methods=methods, summaries=summaries, results=results, paths=paths,
        pairs=pairs, recovery=recovery, recovery_summary=totals, target_identity=identity,
        historical={name: ref['provenance'] for name, ref in refs.items()}, optional_historical=optional)


def _receipt_identity(value, lock):
    required = dict(experiment='R107', arm=lock['arm'], complete=True,
        checkpoint_sha256=lock['checkpoint_sha256'], state_hash=lock['state_hash'],
        selected_update=lock['selected_update'], model_factory=_runner().MODEL_FACTORY,
        inference_rule=_runner().INFERENCE_RULE, model_reselection=False, routing_calibration=False)
    if (any(value.get(k) != v for k, v in required.items()) or lock['arm'] not in ARMS
            or not isinstance(value.get('model_state_sha256'), str) or not value['model_state_sha256']):
        raise ValueError('R107 diagnostic/timing identity differs from the unique per-arm model')


def inference_timing(run, lock):
    value = read_json(Path(run) / 'inference_timing.json')
    _receipt_identity(value, lock)
    required = dict(repetitions=3, warmup_batches=1,
        scope='complete_standalone_model_original_GPU_resident_inputs', all_current_model_included=True,
        explicit_path_computation_included=True, donor_present=False, cached_feature_latency_used=False,
        end_to_end_measured=False, input_labels_read=False,
        sample_policy='first128 frozen TEST inputs per problem')
    if (any(value.get(k) != v for k, v in required.items())
            or value.get('parameter_count') != _runner().PARAMETER_COUNTS[lock['arm']]
            or not str(value.get('device', '')).startswith('cuda')
            or value.get('excludes') != 'dataset I/O, CPU-to-GPU transfer, training and solver execution'):
        raise ValueError('R107 requires actual complete standalone GPU-resident model timing')
    rows = value.get('per_problem', [])
    if len(rows) != len(PROBLEMS) or {r['problem'] for r in rows} != set(PROBLEMS):
        raise ValueError('R107 timing must cover all18 tasks')
    for row in rows:
        samples = row.get('standalone_model', {}).get('batch_seconds', [])
        mean = row.get('standalone_model', {}).get('mean_seconds_per_instance')
        if (row.get('instances') != 128 or len(samples) != 3
                or any(isinstance(s, bool) or not isinstance(s, (int, float))
                       or not math.isfinite(s) or s <= 0 for s in samples)
                or isinstance(mean, bool) or not isinstance(mean, (int, float)) or not math.isfinite(mean) or mean <= 0
                or not math.isclose(mean, sum(samples) / (3 * 128), rel_tol=1e-10)):
            raise ValueError('Invalid R107 latency samples/denominator')
    return value


def lib_stage(args):
    run, config, lock, test, _ = locked_inputs(args.root, args.arm)
    with (run / 'lib_stage.lock').open('a') as claim:
        fcntl.flock(claim, fcntl.LOCK_EX | fcntl.LOCK_NB)
        timing = inference_timing(run, lock)
        expected_hash = test['identity']['model_state_sha256']
        if timing['model_state_sha256'] != expected_hash:
            raise ValueError('R107 LIB requires timing from the same locked TEST model')
        reference = previous.validate_nss_reference(config, args.nss_root)
        def phase(name, **fields):
            write_json(run / 'lib_stage_status.json', dict(experiment='R107', arm=args.arm,
                phase=name, updated_utc=utc(), **fields))
        results = common.r80_lib(args.root, run, config, lock, args.device, args.lib_batch_size,
            phase, test['identity']['model_state_sha256'], factory_module=FACTORY)
        if set(results) != set(LIBS):
            raise ValueError('R107 LIB must complete both TSPLIB and CVRPLIB')
        write_json(run / 'lib_stage_complete.json', dict(experiment='R107', arm=args.arm, complete=True,
            checkpoint_sha256=lock['checkpoint_sha256'], state_hash=lock['state_hash'],
            selected_update=lock['selected_update'], model_state_sha256=expected_hash,
            model_factory=_runner().MODEL_FACTORY, nss_reference=reference,
            datasets=list(results), completed_utc=utc()))
        phase('complete')
        return results


def diagnostics_pdf(output, config, lock, tables, timing):
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    helper, model = common.reporting_helpers(), 'R107/' + config['arm']
    path = output / 'target_recovery_and_train.pdf'
    temporary = path.with_name(f'.{path.stem}.{os.getpid()}.pdf')
    try:
        with PdfPages(temporary) as pdf:
            fig = common._page(helper, model + ' | Explicit paths and actual latency',
                'Complete standalone model including graph/path construction; no equal-FLOPs claim.', model, lock, 1)
            rows = [[r['problem'], '128', f"{1000 * r['standalone_model']['mean_seconds_per_instance']:.6f}"]
                    for r in timing['per_problem']]
            helper.add_table(fig, ['Problem', 'Instances', 'Milliseconds / instance'], rows, [.05, .37, .9, .46], font=9)
            fig.text(.05, .32, f"Total trainable parameters: {config['trainable_parameter_count']:,}; " + ARM_DESCRIPTIONS[config['arm']], fontsize=10)
            fig.text(.05, .29, 'Optional R94B: ' + tables['optional_historical']['R94B']['status'], fontsize=9)
            fig.text(.05, .25, '\n'.join(textwrap.wrap(BOUNDARY, 145)), fontsize=6.5, va='top')
            pdf.savefig(fig)
            plt.close(fig)
            headings = ['Problem', 'Top1 (%)', 'SBS Top1 (%)', 'mean_cost', 'SBS cost',
                        'Oracle cost', 'vs_SBS (%)', 'vs_Oracle (%)', 'actual regret (%)']
            fig = common._page(helper, model + ' | Same selected model: full TRAIN',
                'Diagnostic only; TRAIN was never used to select the checkpoint; full TRAIN-fitted SBS.', model, lock, 2)
            rows = helpers.train_metric_rows(config, lock, tables)
            helper.add_table(fig, headings, [helper.metric_cells(r, r['problem']) for r in rows],
                [.045, .22, .91, .62], font=9, bold_first=True)
            fig.text(.05, .15, 'ALL task-equal; ratios of mean costs differ from per-instance actual regret.', fontsize=10)
            pdf.savefig(fig)
            plt.close(fig)
            for page, ref in enumerate(tables['historical'], 3):
                fig = common._page(helper, model + ' | Fixed target36 vs ' + ref,
                    'True rescues/losses, false picks and aligned net cost; fixed targets, no reselection.', model, lock, page)
                helper.add_table(fig, ['Split', 'Winners', 'Rescued', 'Lost', 'New false', 'Net cost delta'],
                    [[r['split'], str(r['winner_count']), str(r['rescued']), str(r['lost']),
                      str(r['new_false_pick_count']), f"{r['full18_net_mean_cost_delta']:.9f}"]
                     for r in tables['recovery_summary'][ref]], [.05, .57, .9, .25], font=10)
                helper.add_table(fig, ['Split', 'Top1 delta pp', 'Corrected', 'Harmed', 'Cost delta', 'Regret delta pp'],
                    [[r['split'], f"{r['top1_delta_pp']:.6f}", str(r['corrected']), str(r['harmed']),
                      f"{r['mean_cost_delta']:.9f}", f"{r['actual_regret_delta_pp']:.6f}"]
                     for r in tables['pairs'][ref] if r['problem'] == 'ALL'], [.05, .25, .9, .22], font=10)
                fig.text(.05, .16, 'Historical only, not retrained/matched; R89A AdamW, R89B Muon, R84A AdamW.', fontsize=9)
                fig.text(.05, .11, 'Fixed-target-only deltas do not replace the full18 population net cost/regret above.', fontsize=9)
                pdf.savefig(fig)
                plt.close(fig)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return str(path)


def initialization_evidence(run, config):
    value = read_json(Path(run) / 'initialization.json')
    if (not value.get('common_fresh_model_state_sha256')
            or value.get('cross_arm_complete_state_verified') is not True
            or value.get('trainable_parameter_count') != config['trainable_parameter_count']):
        raise ValueError('R107 requires actual complete fresh initialization and parameter-count proof')
    return value


def _render_metadata(path, value):
    # The template renders several PDFs; only this reporter verifies the full set.
    if Path(path).name == 'protocol.json':
        value = dict(value, complete=False, phase='rendered_pending_R107_verification')
    return write_json(path, value)


shared.write_json = _render_metadata


def report(args):
    if args.device != 'cpu':
        raise ValueError('R107 report consumes saved evidence on CPU; no inference/reselection')
    run, config, lock, test, val = locked_inputs(args.root, args.arm)
    with (run / 'postprocess.lock').open('a') as claim:
        fcntl.flock(claim, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            write_json(run / 'postprocess_status.json', dict(experiment='R107', phase='loading_saved_predictions'))
            lib = read_json(run / 'lib_stage_complete.json')
            if (lib.get('experiment') != 'R107' or lib.get('complete') is not True or lib.get('arm') != args.arm
                    or lib.get('checkpoint_sha256') != lock['checkpoint_sha256']
                    or lib.get('state_hash') != lock['state_hash'] or lib.get('model_factory') != _runner().MODEL_FACTORY
                    or lib.get('selected_update') != lock['selected_update']
                    or lib.get('model_state_sha256') != test['identity']['model_state_sha256']
                    or set(lib.get('datasets', [])) != set(LIBS)):
                raise ValueError('R107 report needs complete same-checkpoint/factory LIB evidence')
            timing = inference_timing(run, lock)
            expected_hash = test['identity']['model_state_sha256']
            if timing['model_state_sha256'] != expected_hash:
                raise ValueError('R107 timing and locked predictions use different models')
            reference = previous.validate_nss_reference(config, args.nss_root)
            predictions, metrics, methods = shared.load_saved_splits(run, config, lock, test, val)
            shared.saved_libs(run, config, lock, test, predictions, metrics, methods)
            comparisons, alignment = shared.nss_comparisons(config, lock, predictions, args.nss_root)
            tables = diagnostic_tables(run, config, lock, val, test, args)
            initialization = initialization_evidence(run, config)
            train_rows = helpers.train_metric_rows(config, lock, tables)
            result = shared.build_reports(run, config, lock, test, val,
                predictions, [*train_rows, *metrics], methods, comparisons, alignment)
            output = run / 'full_model_statistics'
            for name, rows in (('method_selection_train_val_test.csv', tables['methods']),
                    ('zero_selection_summary.csv', tables['summaries']),
                    ('train_metrics_by_problem.csv', train_rows),
                    ('inference_timing.csv', [dict(problem=r['problem'], instances=r['instances'],
                        milliseconds_per_instance=1000 * r['standalone_model']['mean_seconds_per_instance'],
                        parameter_count=config['trainable_parameter_count']) for r in timing['per_problem']])):
                common.write_csv(output / name, rows)
            for ref in tables['historical']:
                for stem, key in (('paired_changes', 'pairs'), ('target_recovery', 'recovery'),
                                  ('target_recovery_summary', 'recovery_summary')):
                    common.write_csv(output / (stem + '_vs_' + ref + '.csv'), tables[key][ref])
            result.update(target_recovery_pdf=diagnostics_pdf(output, config, lock, tables, timing),
                inference_timing=timing,
                historical_reference=tables['historical'], optional_historical=tables['optional_historical'],
                nss_reference=reference)
            _, train_proof = _runner().selected_capture_proof(run, tables['results']['TRAIN'], lock, 'train')
            provenance = read_json(output / 'protocol.json')
            provenance.update(model_spec=config['model_spec'], path_spec=config['path_spec'],
                trainable_parameter_count=config['trainable_parameter_count'],
                initialization=initialization, target_identity=tables['target_identity'],
                historical_reference=tables['historical'], optional_historical=tables['optional_historical'],
                arm_description=ARM_DESCRIPTIONS[args.arm], inference_timing=timing,
                evidence_boundary=BOUNDARY,
                selected_train_capture_proof=train_proof,
                selected_val_binding=read_json(run / 'final_metrics.json')['val']['checkpoint_binding'])
            for name in REQUIRED_PDFS:
                if not (output / name).is_file() or not (output / name).stat().st_size:
                    raise ValueError('R107 required PDF missing: ' + name)
            provenance.update(complete=True, phase='complete')
            write_json(output / 'protocol.json', provenance)
            write_json(run / 'postprocess_status.json', dict(experiment='R107', phase='complete',
                exit_status=0, completed_utc=utc(), **result))
            return result
        except BaseException as error:
            write_json(run / 'postprocess_status.json', dict(experiment='R107', phase='postprocess_failed',
                exit_status=1, error=str(error), automatic_retry=False, updated_utc=utc()))
            raise


def compare(args):
    if args.device != 'cpu':
        raise ValueError('R107 comparison is saved-evidence CPU only')
    selected, records, initializations = {}, {}, {}
    for arm in ARMS:
        run, config, lock, test, val = locked_inputs(args.root, arm)
        selected[arm] = selected_predictions(run, config, lock, val, test)[0]
        records[arm], initializations[arm] = (config, lock), initialization_evidence(run, config)
    same_data(records[ARMS[0]][0], records[ARMS[1]][0])
    a, b = (initializations[arm] for arm in ARMS)
    if (a.get('common_fresh_model_state_sha256') != b.get('common_fresh_model_state_sha256')
            or a['trainable_parameter_count'] != b['trainable_parameter_count']):
        raise ValueError('R107 compare requires verified complete fresh tensor equality, not just counts')
    refs = references(records[ARMS[0]][0], args)
    targets, identity = load_targets(records[ARMS[0]][0], refs['R84A']['config'], refs['R84A']['lock'])
    pairs, recovery, totals = comparison_tables(selected[ARMS[0]], selected[ARMS[1]],
        targets, identity, 'R107/' + ARMS[1], 'R107/' + ARMS[0])
    output = args.root / 'comparison_artifacts'
    output.mkdir(parents=True, exist_ok=True)
    for name, rows in (('paired_changes_B_vs_A.csv', pairs), ('target_recovery_B_vs_A.csv', recovery),
                       ('target_recovery_summary_B_vs_A.csv', totals)):
        common.write_csv(output / name, rows)
    result = dict(experiment='R107', complete=True, total_parameter_matched=True,
        active_parameter_matched=False, equal_flops_claim=False, identical_complete_fresh_parameters=True,
        trainable_parameter_count=a['trainable_parameter_count'],
        common_fresh_model_state_sha256=a['common_fresh_model_state_sha256'],
        arm_descriptions=ARM_DESCRIPTIONS, target_identity=identity, evidence_boundary=BOUNDARY,
        checkpoints={arm: records[arm][1] for arm in ARMS}, training=False, inference=False)
    write_json(output / 'protocol.json', result)
    return result


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('command', choices=('lib', 'report', 'compare', 'status'))
    result.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    result.add_argument('--arm', choices=ARMS)
    result.add_argument('--device', default='cpu')
    result.add_argument('--lib-batch-size', type=int, choices=(1, 2), default=2)
    result.add_argument('--nss-root', type=Path, default=DEFAULT_NSS)
    result.add_argument('--historical-root', type=Path, default=DEFAULT_R84 / 'A_baseline')
    result.add_argument('--r89-root', type=Path, default=DEFAULT_R89 / 'B_muon_hybrid')
    result.add_argument('--r89a-root', type=Path, default=DEFAULT_R89 / 'A_adamw')
    result.add_argument('--r94b-root', type=Path, default=None,
        help='Optional saved R94B arm directory; absent/invalid evidence never blocks the base report')
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    for name in ('root', 'nss_root', 'historical_root', 'r89_root', 'r89a_root', 'r94b_root'):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, value.expanduser().resolve())
    if args.command == 'compare':
        return compare(args)
    if args.arm is None:
        raise ValueError('--arm is required for R107 lib/report/status')
    if args.command == 'status':
        path = args.root / args.arm / 'postprocess_status.json'
        value = read_json(path) if path.exists() else dict(experiment='R107', phase='not_started')
        print(json.dumps(value, indent=2))
        return value
    return lib_stage(args) if args.command == 'lib' else report(args)


if __name__ == '__main__':
    main()

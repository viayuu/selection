"""R110 one-forward utility-CE frozen-factory LIB and full18 PDF/CSV reports."""

import argparse
import copy
import csv
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


ARMS = ('A_tied_ce', 'B_private_ce')
BUDGETS = dict.fromkeys(ARMS, 57600)
DEFAULT_ROOT = Path(__file__).resolve().parent / 'runs/R110_method_conditioned_ce'
FACTORY = 'code.V4.r110_experiment'
PROBLEMS = tuple(r84_reporting.PROBLEMS)
DEFAULT_R84, DEFAULT_NSS = r87_reporting.DEFAULT_R84, r87_reporting.DEFAULT_NSS
DEFAULT_R89 = DEFAULT_ROOT.parent / 'R89_optimizer_geometry'
REQUIRED_PDFS = ('MODEL_STATISTICS.pdf', 'selection_distributions.pdf', 'learning_curves.pdf',
                 'training_and_method_diagnostics.pdf', 'target_recovery_and_train.pdf')
ARM_DESCRIPTIONS = dict(
    A_tied_ce='candidate-conditioned nodes, tied two-layer graph weights',
    B_private_ce='candidate-conditioned nodes, 21 global-ID-private two-layer graph banks')
MODEL_ARM_MAP = dict(A_tied_ce='A_tied_late', B_private_ce='B_private_late')
CE_DIAGNOSTICS = dict(training='one-forward unweighted legal-pool raw-utility CE, weight 1',
    evaluation='original maximin CE diagnostic, not the training raw-utility CE objective',
    comparison_weight=0., relative_risk_weight=0., RDrop_KL_weight=0., forward_passes=1)
BOUNDARY = ('R110 refines each candidate node graph through two layers on shared original H4 '
    'before pooling. A_tied_ce ties these weights; B_private_ce has 21 banks keyed by '
    'global method ID, never candidate column, winner or outcome. Both retain candidate '
    'conditioning, original two joint layers and score MLP/maximin. TRAIN optimizes '
    'one-forward unweighted legal-pool raw-utility CE, weight 1; comparison, relative-risk '
    'and RDrop/KL weights are zero. Evaluation maximin CE is diagnostic, not the '
    'optimized raw-utility CE. A has 2,305,033 trainables; B has '
    '10,220,553. The original 1,842,953-parameter base is fresh matched, and all late '
    'branches start from identical original-layer prototypes. The initial eval function '
    'is identical, but complete state hashes/dimensions are not identical. This is not '
    'parameter/FLOP-matched, a routing experiment, or an isolated capacity-only control. '
    'Original R88B physical TSP node0 role, methodID+34 and geometry/distance remain; '
    '57,600 updates, logical batch1024, seed2. No new route/path fields, labels, donor, '
    'teacher, context, targets, calibration, quotas or auxiliary loss. '
    'R89A AdamW, R89B Muon and R84A are saved historical references, not rerun or newly '
    'parameter/FLOP-matched controls; R94B, R93A and R109 arms are explicit optional '
    'saved references, never implicit dependencies. NSS uses saved historical R79 '
    'weights, not retrained on new90k. Fixed36 true rescues/losses, false picks and '
    'aligned population net costs/regret are separate from selection frequency. '
    'Native/FP64 ind and legal pools; TRAIN-fitted SBS; ALL task-equal. TEST is '
    'development-exposed; one whole-model full18 VAL checkpoint per arm for full '
    'TRAIN/VAL/TEST/LIB; no TEST/LIB reselection. Actual standalone GPU-resident-input '
    'timing includes complete method-conditioned node refinement, excludes dataset I/O, '
    'CPU transfer and solvers, and is not end-to-end dataset latency. Microcaps/eval '
    'batch come from the main-witness locked runner, not this reporter. No routing '
    'summary is required. No online W&B equality/availability is asserted.')


def _runner():
    """Import the real factory only when needed, without a historical fallback."""
    module = importlib.import_module(FACTORY)
    if (tuple(module.ARMS) != ARMS or module.BUDGETS != BUDGETS
            or module.MODEL_FACTORY != FACTORY + '.load_checkpoint'
            or module.MODEL_ARM_MAP != MODEL_ARM_MAP):
        raise ValueError('R110 reporting and experiment interfaces differ')
    return module


def _private_module(source, name):
    spec = importlib.util.spec_from_file_location('code.V4._r110_' + name, source.__file__)
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
_template_nss_comparisons = shared.nss_comparisons


def write_csv(path, rows):
    """BOM-free UTF-8 output also works with strict delivery-gate readers."""
    if not rows:
        return
    path = Path(path)
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    fields = list(dict.fromkeys(key for row in rows for key in row))
    try:
        with temporary.open('w', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


common.write_csv = write_csv
NSS_DATASETS = (('TSP TEST', 'TSP', 'TEST'), ('CVRP TEST', 'CVRP', 'TEST'),
                ('TSPLIB', 'TSP', 'TSPLIB'), ('CVRPLIB', 'CVRP', 'CVRPLIB'))


def _nss_current_identity(lock):
    return dict(current_model='R110/' + lock['arm'],
        current_checkpoint_sha256=lock['checkpoint_sha256'], current_state_hash=lock['state_hash'],
        current_selected_update=lock['selected_update'], current_model_factory=FACTORY + '.load_checkpoint')


def _nss_metadata(config, predictions, dataset, problem, split):
    current = predictions[split, problem]
    return dict(data_version=(Path(config['data_root']).name if split == 'TEST'
                             else dataset + ' original native LIB / historical R79 NSS comparison'),
        candidate_pool=json.dumps(list(current.pool)), winner_rule=common.WINNER_RULE)


def nss_comparisons(config, lock, predictions, nss_root):
    rows, alignment = _template_nss_comparisons(config, lock, predictions, nss_root)
    if len(rows) != 4 or [r['dataset'] for r in rows] != [d for d, _, _ in NSS_DATASETS]:
        raise ValueError('R110 requires all four aligned historical NSS datasets')
    for row, (dataset, problem, split) in zip(rows, NSS_DATASETS):
        if (row.get('current_checkpoint_sha256') != lock['checkpoint_sha256']
                or row.get('current_model') != 'R110/' + lock['arm']):
            raise ValueError('R110 NSS rows differ from the unique current model lock')
        row.update(_nss_metadata(config, predictions, dataset, problem, split))
        row.update(_nss_current_identity(lock))
    return rows, alignment


shared.nss_comparisons = nss_comparisons


def validate_nss_csv(path, config, lock, predictions):
    """Check real serialized headers/values, not just the in-memory rows."""
    rows = read_csv(path)
    if len(rows) != 4 or {r.get('dataset') for r in rows} != {d for d, _, _ in NSS_DATASETS}:
        raise ValueError('R110 NSS CSV requires exactly TSP/CVRP TEST and both native LIB datasets')
    by_dataset = {row['dataset']: row for row in rows}
    numeric = ('current_top1_native_pct', 'current_mean_cost', 'nss_top1_native_pct', 'nss_mean_cost')
    for dataset, problem, split in NSS_DATASETS:
        row = by_dataset[dataset]
        expected = _nss_metadata(config, predictions, dataset, problem, split)
        if (row.get('current_checkpoint_sha256') != lock['checkpoint_sha256']
                or row.get('status') != 'complete' or not row.get('nss_checkpoint_sha256')
                or any(row.get(k) != v for k, v in expected.items())
                or any(row.get(k) != str(v) for k, v in _nss_current_identity(lock).items())):
            raise ValueError('R110 NSS CSV lacks same-lock data-version/candidate-pool/native-winner provenance')
        for key in numeric:
            try:
                number = float(row[key])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError('R110 NSS CSV lacks finite ' + key) from error
            if not math.isfinite(number) or ('top1' in key and not 0 <= number <= 100):
                raise ValueError('Invalid R110 NSS CSV metric: ' + key)
    return dict(datasets=[d for d, _, _ in NSS_DATASETS], rows=4,
                encoding='UTF-8; BOM-tolerant reader', same_current_checkpoint=True)


def selected_train(run, config, lock):
    result, path = _template_selected_train(run, config, lock)
    if config['experiment'] == 'R110':
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
            or capture.get('experiment') != 'R110' or capture.get('arm') != lock['arm']
            or capture.get('split') != 'val'
            or capture.get('checkpoint_sha256') not in (None, lock['checkpoint_sha256'])
            or binding.get('archive_capture_checkpoint_sha256') != capture.get('checkpoint_sha256')
            or identity != dict(capture, checkpoint_sha256=lock['checkpoint_sha256'])
            or capture.get('state_hash') != lock['state_hash']
            or capture.get('successful_updates') != lock['selected_update']
            or not capture.get('model_state_sha256')
            or capture['model_state_sha256'] != binding.get('model_state_sha256')):
        raise ValueError('R110 selected VAL summary lacks its exact-state official binding')
    result = copy.deepcopy(val)
    result['identity'] = copy.deepcopy(capture)
    return result


def locked_inputs(root, arm):
    run, config, lock, test, val = shared.locked_inputs(Path(root), arm, experiment='R110', arms=BUDGETS)
    runner = _runner()
    model, state = runner.load_checkpoint(run / 'best.pt', 'cpu')
    if (state['args'] != config or state['successful_updates'] != lock['selected_update']
            or state.get('epoch') != lock['epoch'] or config['model_factory'] != runner.MODEL_FACTORY
            or sum(p.numel() for p in model.parameters() if p.requires_grad)
                != config['trainable_parameter_count']
            or config['trainable_parameter_count'] != runner.PARAMETER_COUNTS[arm]
            or config.get('model_arm') != MODEL_ARM_MAP[arm]
            or config.get('model_params', {}).get('r109_arm') != MODEL_ARM_MAP[arm]
            or 'r110_arm' in config.get('model_params', {})):
        raise ValueError('R110 reporter factory differs from the unique per-arm model lock')
    model_hash = parameter_hash(model.state_dict())
    _, binding = runner.selected_capture_proof(run, val, lock, 'val', state)
    if binding != val.get('checkpoint_binding'):
        raise ValueError('R110 official VAL binding differs from the actual factory/checkpoint proof')
    val = capture_val(val, lock)
    if model_hash != test['identity']['model_state_sha256'] or model_hash != val['identity']['model_state_sha256']:
        raise ValueError('R110 VAL/TEST/frozen-factory model states differ')
    del model, state
    return run, config, lock, test, val


def same_data(left, right):
    for key in ('data_root', 'rows_per_problem', 'candidate_pools', 'candidate_pool_ids', 'baselines', 'winner_rule'):
        if key not in left or left[key] != right.get(key):
            raise ValueError('R110 comparison requires identical ' + key)


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
        raise ValueError('R110 retains frozen R84 target36 without reselection')
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
            'R110/' + config['arm'], name)
    optional = {}
    for name, field, experiment, budgets in (
            ('R94B', 'r94b_root', 'R94', dict(A=57600, B=57600)),
            ('R93A', 'r93a_root', 'R93', dict(A=57600, B=57600)),
            ('R109A', 'r109a_root', 'R109', dict(A_tied_late=57600, B_private_late=57600)),
            ('R109B', 'r109b_root', 'R109', dict(A_tied_late=57600, B_private_late=57600))):
        optional[name] = dict(required=False, status='not_requested')
        path = getattr(args, field, None)
        if path is None:
            continue
        optional[name]['run'] = str(path)
        try:
            ref = historical_reference(config, name, path, experiment, budgets, 'AdamW')
            old = selected_predictions(ref['run'], ref['config'], ref['lock'], ref['val'], ref['test'])[0]
            paired, recovered, total = comparison_tables(old, packed, targets, identity,
                'R110/' + config['arm'], name)
        except Exception as error:
            optional[name].update(status='unavailable', error_type=type(error).__name__, error=str(error))
        else:
            refs[name] = ref
            pairs[name], recovery[name], totals[name] = paired, recovered, total
            optional[name]['status'] = 'included'
    return dict(predictions=packed, methods=methods, summaries=summaries, results=results, paths=paths,
        pairs=pairs, recovery=recovery, recovery_summary=totals, target_identity=identity,
        historical={name: ref['provenance'] for name, ref in refs.items()}, optional_historical=optional)


def _receipt_identity(value, lock):
    required = dict(experiment='R110', arm=lock['arm'], complete=True,
        checkpoint_sha256=lock['checkpoint_sha256'], state_hash=lock['state_hash'],
        selected_update=lock['selected_update'], model_factory=_runner().MODEL_FACTORY,
        inference_rule=_runner().INFERENCE_RULE, model_reselection=False, routing_calibration=False)
    if (any(value.get(k) != v for k, v in required.items()) or lock['arm'] not in ARMS
            or not isinstance(value.get('model_state_sha256'), str) or not value['model_state_sha256']):
        raise ValueError('R110 diagnostic/timing identity differs from the unique per-arm model')


def inference_timing(run, lock):
    value = read_json(Path(run) / 'inference_timing.json')
    _receipt_identity(value, lock)
    _runner().validate_timing_record(value, lock, lock['arm'], value['model_state_sha256'])
    rows = value.get('per_problem', [])
    count = rows[0].get('instances') if rows else None
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError('R110 timing needs the actual positive standalone sample count')
    required = dict(repetitions=3, warmup_batches=1,
        scope='complete_standalone_model_original_GPU_resident_inputs', all_current_model_included=True,
        method_conditioned_node_refinement_included=True, donor_present=False, cached_feature_latency_used=False,
        end_to_end_measured=False, input_labels_read=False,
        sample_policy=f'first{count} frozen TEST inputs per problem')
    if (any(value.get(k) != v for k, v in required.items())
            or value.get('parameter_count') != _runner().PARAMETER_COUNTS[lock['arm']]
            or not str(value.get('device', '')).startswith('cuda')
            or value.get('excludes') != 'dataset I/O, CPU-to-GPU transfer, training and solver execution'):
        raise ValueError('R110 requires actual complete standalone GPU-resident model timing')
    rows = value.get('per_problem', [])
    if len(rows) != len(PROBLEMS) or {r['problem'] for r in rows} != set(PROBLEMS):
        raise ValueError('R110 timing must cover all18 tasks')
    for row in rows:
        samples = row.get('standalone_model', {}).get('batch_seconds', [])
        mean = row.get('standalone_model', {}).get('mean_seconds_per_instance')
        if (row.get('instances') != count or len(samples) != 3
                or any(isinstance(s, bool) or not isinstance(s, (int, float))
                       or not math.isfinite(s) or s <= 0 for s in samples)
                or isinstance(mean, bool) or not isinstance(mean, (int, float)) or not math.isfinite(mean) or mean <= 0
                or not math.isclose(mean, sum(samples) / (3 * count), rel_tol=1e-10)):
            raise ValueError('Invalid R110 latency samples/denominator')
    return value


def lib_stage(args):
    run, config, lock, test, _ = locked_inputs(args.root, args.arm)
    with (run / 'lib_stage.lock').open('a') as claim:
        fcntl.flock(claim, fcntl.LOCK_EX | fcntl.LOCK_NB)
        timing = inference_timing(run, lock)
        expected_hash = test['identity']['model_state_sha256']
        if timing['model_state_sha256'] != expected_hash:
            raise ValueError('R110 LIB requires timing from the same locked TEST model')
        reference = previous.validate_nss_reference(config, args.nss_root)
        def phase(name, **fields):
            write_json(run / 'lib_stage_status.json', dict(experiment='R110', arm=args.arm,
                phase=name, updated_utc=utc(), **fields))
        results = common.r80_lib(args.root, run, config, lock, args.device, args.lib_batch_size,
            phase, test['identity']['model_state_sha256'], factory_module=FACTORY)
        if set(results) != set(LIBS):
            raise ValueError('R110 LIB must complete both TSPLIB and CVRPLIB')
        write_json(run / 'lib_stage_complete.json', dict(experiment='R110', arm=args.arm, complete=True,
            checkpoint_sha256=lock['checkpoint_sha256'], state_hash=lock['state_hash'],
            selected_update=lock['selected_update'], model_state_sha256=expected_hash,
            model_factory=_runner().MODEL_FACTORY, nss_reference=reference,
            datasets=list(results), completed_utc=utc()))
        phase('complete')
        return results


def diagnostics_pdf(output, config, lock, tables, timing):
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    helper, model = common.reporting_helpers(), 'R110/' + config['arm']
    path = output / 'target_recovery_and_train.pdf'
    temporary = path.with_name(f'.{path.stem}.{os.getpid()}.pdf')
    try:
        with PdfPages(temporary) as pdf:
            fig = common._page(helper, model + ' | Method-conditioned nodes and actual latency',
                'Complete standalone model including candidate node refinement; unequal parameter counts and no equal-FLOPs claim.', model, lock, 1)
            rows = [[r['problem'], str(r['instances']), f"{1000 * r['standalone_model']['mean_seconds_per_instance']:.6f}"]
                    for r in timing['per_problem']]
            helper.add_table(fig, ['Problem', 'Instances', 'Milliseconds / instance'], rows, [.05, .37, .9, .46], font=9)
            fig.text(.05, .32, f"Total trainable parameters: {config['trainable_parameter_count']:,}; " + ARM_DESCRIPTIONS[config['arm']], fontsize=10)
            fig.text(.05, .29, 'Optional saved references: ' + ', '.join(
                name + '=' + record['status'] for name, record in tables['optional_historical'].items()), fontsize=8)
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
    """Validate actual scratch-base evidence without equating unequal whole states."""
    value = read_json(Path(run) / 'initialization.json')
    runner, arm = _runner(), config['arm']
    count, original = runner.PARAMETER_COUNTS[arm], runner.PARAMETER_COUNTS['original']
    required = dict(common_parameter_sha256=runner.ORIGINAL_FRESH_SHA256,
        r84_fresh_trainable_sha256=runner.ORIGINAL_FRESH_SHA256,
        common_parameter_count=original, original_parameter_count=original,
        parameter_count=count, trainable_parameter_count=count, new_parameter_count=count - original,
        seed=2, branch_init_seed=runner.BRANCH_INIT_SEED, fresh_initialization=True,
        historical_checkpoint_loaded=False, state_dict_restored=False,
        identical_complete_fresh_parameters=False, total_parameter_matched=False,
        equal_total_parameters=False, equal_flops_claim=False)
    if (config.get('trainable_parameter_count') != count
            or any(value.get(k) != v for k, v in required.items())
            or not value.get('fresh_state_sha256') or not value.get('fresh_trainable_sha256')):
        raise ValueError('R110 requires matched fresh original base/prototype evidence and actual unequal counts')
    return value


def _render_metadata(path, value):
    # The template renders several PDFs; only this reporter verifies the full set.
    if Path(path).name == 'protocol.json':
        value = dict(value, complete=False, phase='rendered_pending_R110_verification')
    return write_json(path, value)


shared.write_json = _render_metadata

LOSS_LABELS = dict(loss='TRAIN raw-utility CE objective (weight 1; one forward)',
    ce='TRAIN raw-utility CE; EVAL maximin CE is diagnostic',
    cmp='comparison (inactive; weight 0)', risk='relative risk (inactive; weight 0)',
    kl='RDrop/KL (inactive; weight 0)')
_template_diagnostic_pdf = common.diagnostic_pdf


def diagnostic_pdf(*args, **kwargs):
    return _template_diagnostic_pdf(*args, **dict(kwargs, loss_labels=LOSS_LABELS))


def learning_curves(run, output, model, lock, *, ovr=False):
    """Plot optimized utility CE without relabeling evaluation maximin CE."""
    import matplotlib.pyplot as plt
    history = read_json(run / 'history.json')
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(model + ' | TRAIN raw-utility CE; EVAL maximin CE is diagnostic', fontsize=13)
    for name in ('loss', 'ce'):
        points = [(r['successful_updates'], r['optimization'][name]) for r in history
                  if name in r.get('optimization', {})]
        if points:
            axes[0, 0].plot(*zip(*points), label=LOSS_LABELS[name])
    axes[0, 0].set_title('TRAIN: one-forward utility CE only; other objective weights zero')
    if axes[0, 0].lines:
        axes[0, 0].legend(fontsize=8)
    for axis, key, factor, title in ((axes[0, 1], 'macro_top1', 100., 'Native Top1 (%)'),
            (axes[1, 0], 'macro_actual_regret_pct', 1., 'Actual regret (%)'),
            (axes[1, 1], 'macro_vs_sbs_pct', 1., 'vs fixed TRAIN SBS (%)')):
        for split in ('train', 'val'):
            points = [(r['successful_updates'], r[split]['macro'][key] * factor) for r in history
                      if r.get(split) is not None]
            if points:
                axis.plot(*zip(*points), label=split)
        axis.set_title(title)
        if axis.lines:
            axis.legend()
    for axis in axes.flat:
        axis.axvline(lock['selected_update'], color='#c45358', linestyle='--', linewidth=1)
        axis.set_xlabel('Successful optimizer updates')
        axis.grid(alpha=.2)
    fig.tight_layout(rect=(0, 0, 1, .96))
    fig.savefig(output / 'learning_curves.pdf')
    fig.savefig(output / 'learning_curves.png', dpi=160)
    plt.close(fig)


common.diagnostic_pdf = diagnostic_pdf
shared.learning_curves = learning_curves


def report(args):
    if args.device != 'cpu':
        raise ValueError('R110 report consumes saved evidence on CPU; no inference/reselection')
    run, config, lock, test, val = locked_inputs(args.root, args.arm)
    with (run / 'postprocess.lock').open('a') as claim:
        fcntl.flock(claim, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            write_json(run / 'postprocess_status.json', dict(experiment='R110', phase='loading_saved_predictions'))
            lib = read_json(run / 'lib_stage_complete.json')
            if (lib.get('experiment') != 'R110' or lib.get('complete') is not True or lib.get('arm') != args.arm
                    or lib.get('checkpoint_sha256') != lock['checkpoint_sha256']
                    or lib.get('state_hash') != lock['state_hash'] or lib.get('model_factory') != _runner().MODEL_FACTORY
                    or lib.get('selected_update') != lock['selected_update']
                    or lib.get('model_state_sha256') != test['identity']['model_state_sha256']
                    or set(lib.get('datasets', [])) != set(LIBS)):
                raise ValueError('R110 report needs complete same-checkpoint/factory LIB evidence')
            timing = inference_timing(run, lock)
            expected_hash = test['identity']['model_state_sha256']
            if timing['model_state_sha256'] != expected_hash:
                raise ValueError('R110 timing and locked predictions use different models')
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
            nss_csv_proof = validate_nss_csv(output / 'nss_comparison.csv', config, lock, predictions)
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
            provenance.update(model_spec=config['model_spec'], branch_init_seed=config['branch_init_seed'],
                trainable_parameter_count=config['trainable_parameter_count'],
                model_arm_map=MODEL_ARM_MAP, model_arm=MODEL_ARM_MAP[args.arm],
                objective=config['objective'], loss=config['loss'], ce_diagnostics=CE_DIAGNOSTICS,
                initialization=initialization, target_identity=tables['target_identity'],
                historical_reference=tables['historical'], optional_historical=tables['optional_historical'],
                arm_description=ARM_DESCRIPTIONS[args.arm], inference_timing=timing,
                evidence_boundary=BOUNDARY,
                selected_train_capture_proof=train_proof,
                nss_csv_delivery_proof=nss_csv_proof,
                selected_val_binding=read_json(run / 'final_metrics.json')['val']['checkpoint_binding'])
            for name in REQUIRED_PDFS:
                if not (output / name).is_file() or not (output / name).stat().st_size:
                    raise ValueError('R110 required PDF missing: ' + name)
            provenance.update(complete=True, phase='complete')
            write_json(output / 'protocol.json', provenance)
            write_json(run / 'postprocess_status.json', dict(experiment='R110', phase='complete',
                exit_status=0, completed_utc=utc(), **result))
            return result
        except BaseException as error:
            write_json(run / 'postprocess_status.json', dict(experiment='R110', phase='postprocess_failed',
                exit_status=1, error=str(error), automatic_retry=False, updated_utc=utc()))
            raise


COMPARISON_METRICS = ('top1_pct', 'mean_cost', 'sbs_cost', 'oracle_cost',
                      'vs_sbs_pct', 'vs_oracle_pct', 'actual_regret_pct')


def completed_report_metrics(run, config, lock):
    """Only combine fully delivered arms, never partially trained arm snapshots."""
    output = run / 'full_model_statistics'
    receipt = read_json(output / 'protocol.json')
    if (receipt.get('complete') is not True or receipt.get('experiment') != 'R110'
            or receipt.get('arm') != config['arm'] or receipt.get('checkpoint_lock') != lock
            or receipt.get('model_factory') != _runner().MODEL_FACTORY):
        raise ValueError('R110 root comparison requires both completed same-lock per-arm reports')
    for name in REQUIRED_PDFS:
        path = output / name
        if not path.is_file() or not path.stat().st_size:
            raise ValueError('R110 root comparison requires completed per-arm PDF: ' + name)
    rows = read_csv(output / 'metrics_by_problem.csv')
    indexed = {}
    for row in rows:
        key = row.get('split'), row.get('problem')
        if key in indexed or row.get('checkpoint_sha256') != lock['checkpoint_sha256']:
            raise ValueError('R110 combined metrics duplicate rows or mix checkpoint locks')
        for metric in COMPARISON_METRICS:
            try:
                row[metric] = float(row[metric])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError('R110 combined metrics require ' + metric) from error
            if not math.isfinite(row[metric]):
                raise ValueError('R110 combined metrics must be finite: ' + metric)
        indexed[key] = row
    expected = {(s, p) for s in ('TRAIN', 'VAL', 'TEST') for p in ('ALL', *PROBLEMS)}
    expected.update((s, p) for s, p in LIBS.items())
    if set(indexed) != expected:
        raise ValueError('R110 combined metrics require full TRAIN/VAL/TEST ALL+18 and both LIB datasets')
    return indexed


def comparison_reports(root, records, metrics, pairs, totals, target_identity):
    """Small whole-version PDF: matched rows, not a parameter-matched claim."""
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    helper = common.reporting_helpers()
    helper.setup_style()
    root = Path(root)
    path, markdown = root / 'MODEL_COMPARISON.pdf', root / 'RESULTS.md'
    temporary = path.with_name(f'.{path.stem}.{os.getpid()}.pdf')
    lock = records[ARMS[0]][1]
    summary, target_rows = [], []
    for split in ('TRAIN', 'VAL', 'TEST'):
        for label, arm in zip(('A', 'B'), ARMS):
            row = metrics[arm][split, 'ALL']
            summary.append([split, label, *[f'{row[m]:.6f}' for m in COMPARISON_METRICS]])
    for row in totals:
        support = row['winner_count']
        a_correct = row['correct_count'] - row['rescued'] + row['lost']
        recall_a = f'{100. * a_correct / support:.6f}' if support else '-'
        recall_b = f'{100. * row["correct_count"] / support:.6f}' if support else '-'
        paired = next(r for r in pairs if r['split'] == row['split'] and r['problem'] == 'ALL')
        target_rows.append([row['split'], str(support), recall_a, recall_b,
            str(row['rescued']), str(row['lost']), str(row['new_false_pick_count']),
            f'{paired["mean_cost_delta"]:.9f}', f'{paired["actual_regret_delta_pp"]:.6f}'])
    try:
        with PdfPages(temporary) as pdf:
            pdf.infodict().update(Title='R110 whole-version A/B comparison',
                Subject='Same selected per-arm models; unequal parameter counts; native FP64 labels')
            fig = common._page(helper, 'R110 | Whole-version A/B comparison',
                'Same frozen data, common base and prototype initialization; NOT parameter/FLOP-matched.',
                'R110 A/B', lock, 1)
            helper.add_table(fig, ['Split', 'Arm', 'Top1 (%)', 'mean_cost', 'SBS cost',
                'Oracle cost', 'vs_SBS (%)', 'vs_Oracle (%)', 'regret (%)'],
                summary, [.035, .48, .93, .34], font=8)
            for index, arm in enumerate(ARMS):
                config, arm_lock = records[arm]
                fig.text(.045, .41 - index * .08,
                    arm + f' | parameters={config["trainable_parameter_count"]:,}'
                    + f' | selected update={arm_lock["selected_update"]}', fontsize=10)
                fig.text(.045, .38 - index * .08, 'SHA256: ' + arm_lock['checkpoint_sha256'], fontsize=8)
            fig.text(.045, .20, '\n'.join(textwrap.wrap(BOUNDARY, 150)), fontsize=6.5, va='top')
            pdf.savefig(fig)
            plt.close(fig)
            for page, split in enumerate(('TRAIN', 'VAL', 'TEST'), 2):
                fig = common._page(helper, 'R110 | ' + split + ': ALL and18 tasks',
                    'Native/FP64 ind; same per-arm lock; mean cost and actual regret remain distinct.',
                    'R110 A/B', lock, page)
                rows = []
                for problem in ('ALL', *PROBLEMS):
                    a, b = (metrics[arm][split, problem] for arm in ARMS)
                    rows.append([problem, f'{a["top1_pct"]:.4f}', f'{b["top1_pct"]:.4f}',
                        f'{a["mean_cost"]:.7f}', f'{b["mean_cost"]:.7f}',
                        f'{a["actual_regret_pct"]:.5f}', f'{b["actual_regret_pct"]:.5f}',
                        f'{b["mean_cost"] - a["mean_cost"]:.7f}'])
                helper.add_table(fig, ['Problem', 'A Top1 (%)', 'B Top1 (%)', 'A mean_cost',
                    'B mean_cost', 'A regret (%)', 'B regret (%)', 'B-A cost'],
                    rows, [.035, .22, .93, .62], font=8, bold_first=True)
                fig.text(.045, .16, 'ALL task-equal; raw costs across different problems are descriptive, not a common-unit objective.', fontsize=9)
                fig.text(.045, .12, 'SBS fitted on frozen TRAIN; Oracle is each instance legal-pool minimum. No TEST/LIB reselection.', fontsize=9)
                pdf.savefig(fig)
                plt.close(fig)
            fig = common._page(helper, 'R110 | Frozen36 recall and population cost',
                'True rescues/losses and false picks, not selection counts; B versus A.', 'R110 A/B', lock, 5)
            helper.add_table(fig, ['Split', 'Winners', 'A recall (%)', 'B recall (%)', 'Rescued',
                'Lost', 'New false', 'ALL cost delta', 'regret delta pp'],
                target_rows, [.035, .55, .93, .25], font=8)
            helper.add_table(fig, ['Split', 'New false cost', 'Removed false cost', 'Target winner cost', 'Full18 net cost'],
                [[r['split'], *[f'{r[k]:.9f}' for k in (
                    'new_false_pick_cost_delta_contribution', 'removed_false_pick_cost_delta_contribution',
                    'target_winner_cost_delta_contribution', 'full18_net_mean_cost_delta')]] for r in totals],
                [.045, .28, .91, .20], font=8)
            fig.text(.045, .21, 'Target36 SHA256: ' + target_identity['sha256'], fontsize=8)
            fig.text(.045, .17, 'Target cost contributions are task-equal; fixed-target changes are not full-population net gain.', fontsize=9)
            fig.text(.045, .13, 'R89A/R89B/R84A and historical R79 NSS remain saved references in each arm PDF.', fontsize=9)
            pdf.savefig(fig)
            plt.close(fig)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    lines = ['# R110 Whole-Version Comparison', '',
        'A = A_tied_ce; B = B_private_ce. Common fresh original base and identical branch prototypes,',
        'but NOT parameter/FLOP-matched and NOT identical complete state hashes.', '',
        'TRAIN: one-forward unweighted raw-utility CE (weight 1). Comparison/risk/RDrop weights are 0.',
        'Evaluation maximin CE is diagnostic, not the optimized raw-utility CE.', '',
        '| Split | Arm | Top1 (%) | mean_cost | SBS cost | Oracle cost | vs_SBS (%) | vs_Oracle (%) | actual regret (%) |',
        '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    lines.extend('| ' + ' | '.join(row) + ' |' for row in summary)
    lines.extend(['', '## Frozen36 True Recovery', '',
        '| Split | Winners | A recall (%) | B recall (%) | Rescued | Lost | New false picks | ALL cost delta | regret delta pp |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |'])
    lines.extend('| ' + ' | '.join(row) + ' |' for row in target_rows)
    lines.extend(['', 'Native/FP64 ind; legal candidate pools; TRAIN-fitted SBS; task-equal ALL.',
        'Fixed36 recovery is not selection frequency. Population cost/regret deltas are B minus A.',
        'TEST has participated in development; no TEST/LIB checkpoint or inference-rule reselection.', '',
        '[Whole-version PDF](MODEL_COMPARISON.pdf)', ''])
    for arm in ARMS:
        config, arm_lock = records[arm]
        lines.append(f'{arm}: {config["trainable_parameter_count"]:,} parameters; '
            f'update {arm_lock["selected_update"]}; SHA256 {arm_lock["checkpoint_sha256"]}.')
        lines.append(f'[Per-arm PDF]({arm}/full_model_statistics/MODEL_STATISTICS.pdf)')
    text_path = markdown.with_name(f'.{markdown.name}.{os.getpid()}.tmp')
    try:
        text_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
        text_path.replace(markdown)
    finally:
        text_path.unlink(missing_ok=True)
    return dict(comparison_pdf=str(path), results_markdown=str(markdown))


def compare(args):
    if args.device != 'cpu':
        raise ValueError('R110 comparison is saved-evidence CPU only')
    selected, records, initializations, metrics = {}, {}, {}, {}
    for arm in ARMS:
        run, config, lock, test, val = locked_inputs(args.root, arm)
        selected[arm] = selected_predictions(run, config, lock, val, test)[0]
        records[arm], initializations[arm] = (config, lock), initialization_evidence(run, config)
        metrics[arm] = completed_report_metrics(run, config, lock)
    same_data(records[ARMS[0]][0], records[ARMS[1]][0])
    a, b = (initializations[arm] for arm in ARMS)
    if (a['common_parameter_sha256'] != b['common_parameter_sha256']
            or a['branch_init_seed'] != b['branch_init_seed']
            or a['trainable_parameter_count'] == b['trainable_parameter_count']
            or a['fresh_state_sha256'] == b['fresh_state_sha256']):
        raise ValueError('R110 compare requires matched original base/prototypes, not identical whole states/counts')
    refs = references(records[ARMS[0]][0], args)
    targets, identity = load_targets(records[ARMS[0]][0], refs['R84A']['config'], refs['R84A']['lock'])
    pairs, recovery, totals = comparison_tables(selected[ARMS[0]], selected[ARMS[1]],
        targets, identity, 'R110/' + ARMS[1], 'R110/' + ARMS[0])
    output = args.root / 'comparison_artifacts'
    output.mkdir(parents=True, exist_ok=True)
    for name, rows in (('paired_changes_B_vs_A.csv', pairs), ('target_recovery_B_vs_A.csv', recovery),
                       ('target_recovery_summary_B_vs_A.csv', totals)):
        common.write_csv(output / name, rows)
    delivery = comparison_reports(args.root, records, metrics, pairs, totals, identity)
    for name in ('comparison_pdf', 'results_markdown'):
        path = Path(delivery[name])
        if not path.is_file() or not path.stat().st_size:
            raise ValueError('R110 combined delivery missing: ' + name)
    result = dict(experiment='R110', complete=True, total_parameter_matched=False,
        active_parameter_matched=False, equal_flops_claim=False, identical_complete_fresh_parameters=False,
        trainable_parameters={arm: initializations[arm]['trainable_parameter_count'] for arm in ARMS},
        common_fresh_trainable_sha256=a['common_parameter_sha256'],
        identical_late_prototype_initialization=True, branch_init_seed=a['branch_init_seed'],
        fresh_state_sha256={arm: initializations[arm]['fresh_state_sha256'] for arm in ARMS},
        arm_descriptions=ARM_DESCRIPTIONS, target_identity=identity, evidence_boundary=BOUNDARY,
        model_arm_map=MODEL_ARM_MAP, ce_diagnostics=CE_DIAGNOSTICS,
        checkpoints={arm: records[arm][1] for arm in ARMS}, training=False, inference=False,
        **delivery)
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
    result.add_argument('--r93a-root', type=Path, default=None,
        help='Optional saved R93/A pure-CE arm directory; never retrain or select a checkpoint')
    result.add_argument('--r109a-root', type=Path, default=None,
        help='Optional saved R109/A_tied_late arm directory, not an implicit dependency')
    result.add_argument('--r109b-root', type=Path, default=None,
        help='Optional saved R109/B_private_late arm directory, not an implicit dependency')
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    for name in ('root', 'nss_root', 'historical_root', 'r89_root', 'r89a_root', 'r94b_root',
                 'r93a_root', 'r109a_root', 'r109b_root'):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, value.expanduser().resolve())
    if args.command == 'compare':
        return compare(args)
    if args.arm is None:
        raise ValueError('--arm is required for R110 lib/report/status')
    if args.command == 'status':
        path = args.root / args.arm / 'postprocess_status.json'
        value = read_json(path) if path.exists() else dict(experiment='R110', phase='not_started')
        print(json.dumps(value, indent=2))
        return value
    return lib_stage(args) if args.command == 'lib' else report(args)


if __name__ == '__main__':
    main()

"""R108 frozen-factory LIB and saved-evidence full18 PDF/CSV reporting."""

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


ARMS = ('A_prior_ce', 'B_density_ratio')
BUDGETS = dict.fromkeys(ARMS, 57600)
DEFAULT_ROOT = Path(__file__).resolve().parent / 'runs/R108_conditional_density_ratio'
FACTORY = 'code.V4.r108_experiment'
PROBLEMS = tuple(r84_reporting.PROBLEMS)
DEFAULT_R84, DEFAULT_NSS = r87_reporting.DEFAULT_R84, r87_reporting.DEFAULT_NSS
DEFAULT_R89 = DEFAULT_ROOT.parent / 'R89_optimizer_geometry'
REQUIRED_PDFS = ('MODEL_STATISTICS.pdf', 'selection_distributions.pdf', 'learning_curves.pdf',
                 'training_and_method_diagnostics.pdf', 'target_recovery_and_train.pdf')
ARM_DESCRIPTIONS = dict(
    A_prior_ce='pure cross entropy on s=logpi+t',
    B_density_ratio='product-noise binary density ratio on residual t')
BOUNDARY = ('R108 uses the original scratch R84/R88B full18 selector with 1,842,953 trainables. '
    'Both arms share the same complete fresh tensors, fixed TRAIN-only (problem, exactN) '
    'prior buffers, forward computation and original maximin inference on s=logpi+t. '
    'Active gradient freedom is not identical: CE common-shift gauge cancels the shared '
    'last-bias data gradient, whereas the ratio anchors the utility origin. '
    't is the original learned utility. A trains pure CE(s); B trains the canonical '
    'product-noise binary ratio: half positive softplus(-t_y) plus half pi-weighted '
    'softplus(t). Its normalization differs from CE; no equal gradient-scale claim. '
    'The ratio sigmoid is not a champion probability. Training classification losses and '
    'evaluation maximin CE are different diagnostics and are not compared as equal losses. '
    'Single forward, dropout0.1; comparison/risk/KL/RDrop weights and RDrop ramp are zero. '
    '57,600 updates, logical batch1024, seed2, AdamW1e-4 to1e-6. '
    'Prior tables are small frozen aggregate TRAIN-fit classifier parameters, not '
    'per-instance features or query labels. Priors alone do not demonstrate instance '
    'learning: the saved VAL/TEST t=0 reference is a TRAIN '
    'winner-frequency conditional prior, not a cost-optimal SBS. Model gains versus '
    'that reference are reported separately from gains versus SBS. No '
    'new feature cache, donor, additional '
    'path module, teacher, context labels, new targets, calibration, quota or auxiliary loss. '
    'R89A AdamW, R89B Muon and R84A are saved historical references, not rerun or newly '
    'parameter/FLOP-matched controls; R94B is optional, never a prerequisite. Fixed36 rescues, '
    'losses, false picks and aligned net costs/regret are separate from choice frequency. '
    'Native/FP64 ind and legal pools; TRAIN-fitted SBS; ALL task-equal. TEST is '
    'development-exposed; one whole-model full18 VAL checkpoint for full TRAIN/VAL/TEST/LIB; '
    'no TEST/LIB reselection. Actual standalone GPU-resident-input timing includes complete '
    'model prior lookup and residual computation, excludes dataset I/O, CPU transfer and '
    'solvers, and is not end-to-end dataset latency. No routing-summary requirement. '
    'No online W&B equality/availability is asserted.')


def _runner():
    """Import the real factory only when needed, without a historical fallback."""
    module = importlib.import_module(FACTORY)
    if (tuple(module.ARMS) != ARMS or module.BUDGETS != BUDGETS
            or module.MODEL_FACTORY != FACTORY + '.load_checkpoint'):
        raise ValueError('R108 reporting and experiment interfaces differ')
    return module


def _private_module(source, name):
    spec = importlib.util.spec_from_file_location('code.V4._r108_' + name, source.__file__)
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
    if config['experiment'] == 'R108':
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
            or capture.get('experiment') != 'R108' or capture.get('arm') != lock['arm']
            or capture.get('split') != 'val'
            or capture.get('checkpoint_sha256') not in (None, lock['checkpoint_sha256'])
            or binding.get('archive_capture_checkpoint_sha256') != capture.get('checkpoint_sha256')
            or identity != dict(capture, checkpoint_sha256=lock['checkpoint_sha256'])
            or capture.get('state_hash') != lock['state_hash']
            or capture.get('successful_updates') != lock['selected_update']
            or not capture.get('model_state_sha256')
            or capture['model_state_sha256'] != binding.get('model_state_sha256')):
        raise ValueError('R108 selected VAL summary lacks its exact-state official binding')
    result = copy.deepcopy(val)
    result['identity'] = copy.deepcopy(capture)
    return result


def locked_inputs(root, arm):
    run, config, lock, test, val = shared.locked_inputs(Path(root), arm, experiment='R108', arms=BUDGETS)
    runner = _runner()
    model, state = runner.load_checkpoint(run / 'best.pt', 'cpu')
    if (state['args'] != config or state['successful_updates'] != lock['selected_update']
            or state.get('epoch') != lock['epoch'] or config['model_factory'] != runner.MODEL_FACTORY
            or sum(p.numel() for p in model.parameters() if p.requires_grad)
                != config['trainable_parameter_count']
            or config['trainable_parameter_count'] != runner.PARAMETER_COUNTS[arm]):
        raise ValueError('R108 reporter factory differs from the unique per-arm model lock')
    model_hash = parameter_hash(model.state_dict())
    _, binding = runner.selected_capture_proof(run, val, lock, 'val', state)
    if binding != val.get('checkpoint_binding'):
        raise ValueError('R108 official VAL binding differs from the actual factory/checkpoint proof')
    val = capture_val(val, lock)
    if model_hash != test['identity']['model_state_sha256'] or model_hash != val['identity']['model_state_sha256']:
        raise ValueError('R108 VAL/TEST/frozen-factory model states differ')
    del model, state
    return run, config, lock, test, val


def same_data(left, right):
    for key in ('data_root', 'rows_per_problem', 'candidate_pools', 'candidate_pool_ids', 'baselines', 'winner_rule'):
        if key not in left or left[key] != right.get(key):
            raise ValueError('R108 comparison requires identical ' + key)


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
        raise ValueError('R108 retains frozen R84 target36 without reselection')
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
            'R108/' + config['arm'], name)
    optional = dict(R94B=dict(required=False, status='not_requested'))
    if args.r94b_root is not None:
        optional['R94B']['run'] = str(args.r94b_root)
        try:
            ref = historical_reference(config, 'R94B', args.r94b_root, 'R94', dict(A=57600, B=57600), 'AdamW')
            old = selected_predictions(ref['run'], ref['config'], ref['lock'], ref['val'], ref['test'])[0]
            paired, recovered, total = comparison_tables(old, packed, targets, identity,
                'R108/' + config['arm'], 'R94B')
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
    required = dict(experiment='R108', arm=lock['arm'], complete=True,
        checkpoint_sha256=lock['checkpoint_sha256'], state_hash=lock['state_hash'],
        selected_update=lock['selected_update'], model_factory=_runner().MODEL_FACTORY,
        inference_rule=_runner().INFERENCE_RULE, model_reselection=False, routing_calibration=False)
    if (any(value.get(k) != v for k, v in required.items()) or lock['arm'] not in ARMS
            or not isinstance(value.get('model_state_sha256'), str) or not value['model_state_sha256']):
        raise ValueError('R108 diagnostic/timing identity differs from the unique per-arm model')


def inference_timing(run, lock):
    value = read_json(Path(run) / 'inference_timing.json')
    _receipt_identity(value, lock)
    required = dict(repetitions=3, warmup_batches=1,
        scope='complete_standalone_model_original_GPU_resident_inputs', all_current_model_included=True,
        prior_lookup_and_residual_computation_included=True, donor_present=False, cached_feature_latency_used=False,
        end_to_end_measured=False, input_labels_read=False,
        sample_policy='first128 frozen TEST inputs per problem')
    if (any(value.get(k) != v for k, v in required.items())
            or value.get('parameter_count') != _runner().PARAMETER_COUNTS[lock['arm']]
            or not str(value.get('device', '')).startswith('cuda')
            or value.get('excludes') != 'dataset I/O, CPU-to-GPU transfer, training and solver execution'):
        raise ValueError('R108 requires actual complete standalone GPU-resident model timing')
    rows = value.get('per_problem', [])
    if len(rows) != len(PROBLEMS) or {r['problem'] for r in rows} != set(PROBLEMS):
        raise ValueError('R108 timing must cover all18 tasks')
    for row in rows:
        samples = row.get('standalone_model', {}).get('batch_seconds', [])
        mean = row.get('standalone_model', {}).get('mean_seconds_per_instance')
        if (row.get('instances') != 128 or len(samples) != 3
                or any(isinstance(s, bool) or not isinstance(s, (int, float))
                       or not math.isfinite(s) or s <= 0 for s in samples)
                or isinstance(mean, bool) or not isinstance(mean, (int, float)) or not math.isfinite(mean) or mean <= 0
                or not math.isclose(mean, sum(samples) / (3 * 128), rel_tol=1e-10)):
            raise ValueError('Invalid R108 latency samples/denominator')
    return value


def lib_stage(args):
    run, config, lock, test, _ = locked_inputs(args.root, args.arm)
    with (run / 'lib_stage.lock').open('a') as claim:
        fcntl.flock(claim, fcntl.LOCK_EX | fcntl.LOCK_NB)
        timing = inference_timing(run, lock)
        expected_hash = test['identity']['model_state_sha256']
        if timing['model_state_sha256'] != expected_hash:
            raise ValueError('R108 LIB requires timing from the same locked TEST model')
        reference = previous.validate_nss_reference(config, args.nss_root)
        def phase(name, **fields):
            write_json(run / 'lib_stage_status.json', dict(experiment='R108', arm=args.arm,
                phase=name, updated_utc=utc(), **fields))
        results = common.r80_lib(args.root, run, config, lock, args.device, args.lib_batch_size,
            phase, test['identity']['model_state_sha256'], factory_module=FACTORY)
        if set(results) != set(LIBS):
            raise ValueError('R108 LIB must complete both TSPLIB and CVRPLIB')
        write_json(run / 'lib_stage_complete.json', dict(experiment='R108', arm=args.arm, complete=True,
            checkpoint_sha256=lock['checkpoint_sha256'], state_hash=lock['state_hash'],
            selected_update=lock['selected_update'], model_state_sha256=expected_hash,
            model_factory=_runner().MODEL_FACTORY, nss_reference=reference,
            datasets=list(results), completed_utc=utc()))
        phase('complete')
        return results


def diagnostics_pdf(output, config, lock, tables, timing):
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    helper, model = common.reporting_helpers(), 'R108/' + config['arm']
    path = output / 'target_recovery_and_train.pdf'
    temporary = path.with_name(f'.{path.stem}.{os.getpid()}.pdf')
    try:
        with PdfPages(temporary) as pdf:
            fig = common._page(helper, model + ' | TRAIN priors and actual latency',
                'Complete forward including prior lookup and residual; no equal-gradient-scale or FLOPs claim.', model, lock, 1)
            rows = [[r['problem'], '128', f"{1000 * r['standalone_model']['mean_seconds_per_instance']:.6f}"]
                    for r in timing['per_problem']]
            helper.add_table(fig, ['Problem', 'Instances', 'Milliseconds / instance'], rows, [.05, .37, .9, .46], font=9)
            fig.text(.05, .32, f"Total trainable parameters: {config['trainable_parameter_count']:,}; " + ARM_DESCRIPTIONS[config['arm']], fontsize=10)
            fig.text(.05, .29, 'Optional R94B: ' + tables['optional_historical']['R94B']['status'], fontsize=9)
            fig.text(.05, .265, 'TRAIN-only prior SHA256: ' + config['priors_sha256'], fontsize=7)
            fig.text(.05, .24, '\n'.join(textwrap.wrap(BOUNDARY, 145)), fontsize=6.5, va='top')
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
            for index, row in enumerate(r for r in tables['prior_reference'] if r['problem'] == 'ALL'):
                fig.text(.05, .105 - index * .035,
                    f"{row['split']} ALL vs TRAIN frequency prior (not SBS), positive = better: "
                    f"Top1 gain {row['model_gain_top1_pp']:+.4f} pp; "
                    f"cost gain {row['model_gain_mean_cost']:+.7f}; "
                    f"regret gain {row['model_gain_actual_regret_pp']:+.4f} pp.", fontsize=8)
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
        raise ValueError('R108 requires actual complete fresh initialization and parameter-count proof')
    return value


def prior_provenance(config):
    path = Path(config['priors_path'])
    locked = config['main_protocol_lock']['record']
    if (config.get('priors_sha256') != locked.get('priors_sha256')
            or str(path) != locked.get('priors_path') or not config.get('prior_spec')
            or file_hash(path) != config['priors_sha256']):
        raise ValueError('R108 requires the unchanged main-locked TRAIN-only prior table')
    table = read_json(path)
    if (table.get('split') != 'train' or table.get('rows_per_problem') != 81000
            or set(table.get('tasks', {})) != set(PROBLEMS)
            or table.get('prior_spec') != config['prior_spec']
            or table.get('source_identity', {}).get('data_root') != config['data_root']
            or not table.get('identity_sha256')):
        raise ValueError('R108 prior provenance must cover only frozen full18 TRAIN')
    return dict(path=str(path), sha256=config['priors_sha256'], fit_split='train',
        identity_sha256=table['identity_sha256'], rows_per_problem=81000,
        source_identity=table['source_identity'],
        conditioning=['problem', 'exactN'], fixed_buffers=True, feature_cache=False,
        current_query_labels_used=False, specification=config['prior_spec'])


def conditional_prior_reference(config, lock, tables):
    """Evaluate t=0 from frozen TRAIN statistics and saved held-out nodes/costs."""
    prior = read_json(config['priors_path'])
    rows = []
    metric_keys = ('prior_top1_pct', 'prior_mean_cost', 'prior_actual_regret_pct',
        'model_top1_pct', 'model_mean_cost', 'model_actual_regret_pct',
        'model_gain_top1_pp', 'model_gain_mean_cost', 'model_gain_actual_regret_pp')
    for split in ('VAL', 'TEST'):
        split_rows = []
        with np.load(tables['paths'][split], allow_pickle=False) as archive:
            for problem in PROBLEMS:
                data, task = tables['predictions'][split][problem], prior['tasks'][problem]
                nodes = archive[problem + '__nodes']
                if nodes.shape != (data.n,) or nodes.dtype.kind not in 'iu' or (nodes <= 0).any():
                    raise ValueError('Invalid saved exact N for prior-only reference: ' + problem)
                columns = [task['pool_ids'].index(int(i)) for i in data.pool_ids]
                if len(columns) != len(task['pool_ids']) or [task['pool'][i] for i in columns] != data.pool:
                    raise ValueError('Prior-only reference must use the same frozen legal pool')
                values = np.asarray(task['n_values'])
                positions = np.searchsorted(values, nodes).clip(max=len(values) - 1)
                seen = values[positions] == nodes
                # Match the fixed FP32 buffers, unseen-N fallback and global-ID ties.
                probabilities = np.broadcast_to(np.asarray(task['task_prior'], dtype=np.float32),
                    (data.n, len(columns))).copy()
                probabilities[seen] = np.asarray(task['conditional_prior'], dtype=np.float32)[positions[seen]]
                order = np.argsort(data.pool_ids, kind='stable')
                scores = np.log(probabilities[:, columns])
                pred = order[scores[:, order].argmax(1)]
                oracle, index = data.costs.min(1), np.arange(data.n)
                def metrics(chosen):
                    cost = data.costs[index, chosen]
                    return (float(100. * np.mean(chosen == data.winner)),
                        float(cost.mean(dtype=np.float64)), float(np.mean(100. * (cost - oracle) / oracle)))
                p_top1, p_cost, p_regret = metrics(pred)
                m_top1, m_cost, m_regret = metrics(data.pred)
                split_rows.append(dict(split=split, problem=problem, n=data.n,
                    reference='TRAIN winner-frequency conditional prior (t=0); not cost-optimal SBS',
                    fit_split='train', inference_rule='argmax logpi(problem, exactN); unseenN task prior; global-ID ties',
                    model='R108/' + config['arm'], checkpoint_sha256=lock['checkpoint_sha256'],
                    priors_sha256=config['priors_sha256'], winner_rule=config['winner_rule'],
                    aggregation='per problem', gain_direction='positive is better',
                    prior_top1_pct=p_top1, prior_mean_cost=p_cost, prior_actual_regret_pct=p_regret,
                    model_top1_pct=m_top1, model_mean_cost=m_cost, model_actual_regret_pct=m_regret,
                    model_gain_top1_pp=m_top1 - p_top1, model_gain_mean_cost=p_cost - m_cost,
                    model_gain_actual_regret_pp=p_regret - m_regret))
        overall = dict(split_rows[0], problem='ALL', n=sum(r['n'] for r in split_rows),
            aggregation='equal weight across 18 problems; raw costs descriptive only')
        overall.update({key: float(np.mean([r[key] for r in split_rows])) for key in metric_keys})
        rows.extend([*split_rows, overall])
    return rows


LOSS_LABELS = dict(loss='TRAIN classification objective (arm-specific)',
    ce='TRAIN CE(s) diagnostic; evaluation CE uses maximin')
_template_diagnostic_pdf = common.diagnostic_pdf


def diagnostic_pdf(*args, **kwargs):
    return _template_diagnostic_pdf(*args, **dict(kwargs, loss_labels=LOSS_LABELS))


def learning_curves(run, output, model, lock, *, ovr=False):
    """R107 layout with the two R108 loss meanings explicitly distinguished."""
    import matplotlib.pyplot as plt
    history = read_json(run / 'history.json')
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(model + ' | one VAL lock; TRAIN CE(s) is not evaluation maximin CE', fontsize=13)
    for name, label in LOSS_LABELS.items():
        points = [(r['successful_updates'], r['optimization'][name]) for r in history
            if name in r.get('optimization', {})]
        if points:
            axes[0, 0].plot(*zip(*points), label=label)
    axes[0, 0].set_title('TRAIN interval means; CE and ratio normalization differ')
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


def _render_metadata(path, value):
    # The template renders several PDFs; only this reporter verifies the full set.
    if Path(path).name == 'protocol.json':
        value = dict(value, complete=False, phase='rendered_pending_R108_verification')
    return write_json(path, value)


shared.write_json = _render_metadata


def report(args):
    if args.device != 'cpu':
        raise ValueError('R108 report consumes saved evidence on CPU; no inference/reselection')
    run, config, lock, test, val = locked_inputs(args.root, args.arm)
    with (run / 'postprocess.lock').open('a') as claim:
        fcntl.flock(claim, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            write_json(run / 'postprocess_status.json', dict(experiment='R108', phase='loading_saved_predictions'))
            lib = read_json(run / 'lib_stage_complete.json')
            if (lib.get('experiment') != 'R108' or lib.get('complete') is not True or lib.get('arm') != args.arm
                    or lib.get('checkpoint_sha256') != lock['checkpoint_sha256']
                    or lib.get('state_hash') != lock['state_hash'] or lib.get('model_factory') != _runner().MODEL_FACTORY
                    or lib.get('selected_update') != lock['selected_update']
                    or lib.get('model_state_sha256') != test['identity']['model_state_sha256']
                    or set(lib.get('datasets', [])) != set(LIBS)):
                raise ValueError('R108 report needs complete same-checkpoint/factory LIB evidence')
            timing = inference_timing(run, lock)
            expected_hash = test['identity']['model_state_sha256']
            if timing['model_state_sha256'] != expected_hash:
                raise ValueError('R108 timing and locked predictions use different models')
            reference = previous.validate_nss_reference(config, args.nss_root)
            predictions, metrics, methods = shared.load_saved_splits(run, config, lock, test, val)
            shared.saved_libs(run, config, lock, test, predictions, metrics, methods)
            comparisons, alignment = shared.nss_comparisons(config, lock, predictions, args.nss_root)
            tables = diagnostic_tables(run, config, lock, val, test, args)
            initialization = initialization_evidence(run, config)
            prior = prior_provenance(config)
            prior_rows = conditional_prior_reference(config, lock, tables)
            tables['prior_reference'] = prior_rows
            train_rows = helpers.train_metric_rows(config, lock, tables)
            result = shared.build_reports(run, config, lock, test, val,
                predictions, [*train_rows, *metrics], methods, comparisons, alignment)
            output = run / 'full_model_statistics'
            for name, rows in (('method_selection_train_val_test.csv', tables['methods']),
                    ('zero_selection_summary.csv', tables['summaries']),
                    ('train_metrics_by_problem.csv', train_rows),
                    ('conditional_prior_reference.csv', prior_rows),
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
            result['conditional_prior_reference_csv'] = str(output / 'conditional_prior_reference.csv')
            _, train_proof = _runner().selected_capture_proof(run, tables['results']['TRAIN'], lock, 'train')
            provenance = read_json(output / 'protocol.json')
            provenance.update(model_spec=config['model_spec'], prior_spec=config['prior_spec'],
                prior_provenance=prior, objective=config['objective'], loss=config['loss'],
                conditional_prior_reference=dict(csv=result['conditional_prior_reference_csv'],
                    reference='TRAIN winner-frequency prior, not cost-optimal SBS; t=0; no fit or model inference',
                    gain_direction='positive is better',
                    all_by_split={r['split']: r for r in prior_rows if r['problem'] == 'ALL'}),
                training_evaluation_ce_differ=True, equal_gradient_scale_claim=False,
                total_parameter_matched=True, active_parameter_matched=False, equal_flops_claim=False,
                trainable_parameter_count=config['trainable_parameter_count'],
                initialization=initialization, target_identity=tables['target_identity'],
                historical_reference=tables['historical'], optional_historical=tables['optional_historical'],
                arm_description=ARM_DESCRIPTIONS[args.arm], inference_timing=timing,
                evidence_boundary=BOUNDARY,
                selected_train_capture_proof=train_proof,
                selected_val_binding=read_json(run / 'final_metrics.json')['val']['checkpoint_binding'])
            for name in REQUIRED_PDFS:
                if not (output / name).is_file() or not (output / name).stat().st_size:
                    raise ValueError('R108 required PDF missing: ' + name)
            provenance.update(complete=True, phase='complete')
            write_json(output / 'protocol.json', provenance)
            write_json(run / 'postprocess_status.json', dict(experiment='R108', phase='complete',
                exit_status=0, completed_utc=utc(), **result))
            return result
        except BaseException as error:
            write_json(run / 'postprocess_status.json', dict(experiment='R108', phase='postprocess_failed',
                exit_status=1, error=str(error), automatic_retry=False, updated_utc=utc()))
            raise


def compare(args):
    if args.device != 'cpu':
        raise ValueError('R108 comparison is saved-evidence CPU only')
    selected, records, initializations = {}, {}, {}
    for arm in ARMS:
        run, config, lock, test, val = locked_inputs(args.root, arm)
        selected[arm] = selected_predictions(run, config, lock, val, test)[0]
        records[arm], initializations[arm] = (config, lock), initialization_evidence(run, config)
    same_data(records[ARMS[0]][0], records[ARMS[1]][0])
    priors = [prior_provenance(records[arm][0]) for arm in ARMS]
    if priors[0] != priors[1]:
        raise ValueError('R108 compare requires the same fixed TRAIN-only prior buffers')
    a, b = (initializations[arm] for arm in ARMS)
    if (a.get('common_fresh_model_state_sha256') != b.get('common_fresh_model_state_sha256')
            or a['trainable_parameter_count'] != b['trainable_parameter_count']):
        raise ValueError('R108 compare requires verified complete fresh tensor equality, not just counts')
    refs = references(records[ARMS[0]][0], args)
    targets, identity = load_targets(records[ARMS[0]][0], refs['R84A']['config'], refs['R84A']['lock'])
    pairs, recovery, totals = comparison_tables(selected[ARMS[0]], selected[ARMS[1]],
        targets, identity, 'R108/' + ARMS[1], 'R108/' + ARMS[0])
    output = args.root / 'comparison_artifacts'
    output.mkdir(parents=True, exist_ok=True)
    for name, rows in (('paired_changes_B_vs_A.csv', pairs), ('target_recovery_B_vs_A.csv', recovery),
                       ('target_recovery_summary_B_vs_A.csv', totals)):
        common.write_csv(output / name, rows)
    result = dict(experiment='R108', complete=True, total_parameter_matched=True,
        active_parameter_matched=False, equal_flops_claim=False, equal_gradient_scale_claim=False,
        identical_complete_fresh_parameters=True, priors_sha256=priors[0]['sha256'],
        prior_provenance=priors[0],
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
        raise ValueError('--arm is required for R108 lib/report/status')
    if args.command == 'status':
        path = args.root / args.arm / 'postprocess_status.json'
        value = read_json(path) if path.exists() else dict(experiment='R108', phase='not_started')
        print(json.dumps(value, indent=2))
        return value
    return lib_stage(args) if args.command == 'lib' else report(args)


if __name__ == '__main__':
    main()

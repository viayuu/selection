"""R94 locked node-fusion reports, never training or checkpoint selection.

``report``/``compare`` are CPU-only saved-prediction administration. Only an
explicit ``lib --device cuda:N`` may infer, on the launcher's allocated GPU.
The R94 runner/factory owns checkpoint validation; its import is late-bound.
No imported R87/R93/shared module globals are rebound by this adaptation.
"""

import argparse
import copy
import fcntl
import importlib
import json
import math
import os
from pathlib import Path
import shutil
import traceback

from . import r87_reporting as helpers
from . import r93_reporting as r93
from .r76_experiment import file_hash, read_json, utc, write_json


ARMS = ('A', 'B')
MAX_UPDATES = 57600
BUDGETS = dict.fromkeys(ARMS, MAX_UPDATES)
VARIANTS = dict(A='random', B='pretrained')
MODEL_ROOT = Path(__file__).resolve().parent / 'runs/R94_pretrained_node_fusion'
DEFAULT_ROOT = MODEL_ROOT
FACTORY = 'code.V4.r94_experiment'
MODEL_FACTORY = FACTORY + '.load_checkpoint'
INITIALIZATION = 'scratch_shared_r77_plus_node_fusion'
shared, previous, common = helpers.shared, helpers.previous, helpers.common
DEFAULT_NSS, DEFAULT_R84 = helpers.DEFAULT_NSS, helpers.DEFAULT_R84
DEFAULT_R89B = MODEL_ROOT.parent / 'R89_optimizer_geometry/B_muon_hybrid'
PROBLEMS = tuple(helpers.PROBLEMS)
REQUIRED_PDFS = ('MODEL_STATISTICS.pdf', 'selection_distributions.pdf',
                 'target_recovery_and_train.pdf', 'learning_curves.pdf',
                 'training_and_method_diagnostics.pdf')
REFERENCE_BOUNDARY = (
    'A is the matched random-frozen ReLD128 donor control; B uses pinned pretrained '
    'weights in the same frozen donor. Both use identical fresh trainable selectors '
    'and early node-fusion projections before the original four inter-node layers. '
    'Full18 raw inputs; donor input coverage is16 routing tasks, not16-task pretraining. '
    'TSP/ATSP raw paths are unchanged. Original R77 CE=.35, comparison=.35, risk=.02, '
    'R-Drop=.35 and dropout=.1 are retained, not R93 pure CE. R89B is the main '
    'historical reference; R84A is secondary, neither newly matched. External donor '
    'pretraining compute/data are not budget-matched to scratch; provenance/overlap '
    'limits remain. R50 is negative evidence for pool-first readout, not proof that '
    'early trainable inter-node recomputation succeeds. No solver decoder, routes, '
    'cost-derived new labels, or restored ReLD MTL. Hardware differences must be '
    'disclosed; matching code/batches is not bitwise hardware matching. '
    'Development-exposed TEST; unique full18 VAL macro_vs_sbs lock, no TEST/LIB '
    'selection or per-task model assembly. Cached-feature selector latency is not '
    'full-system inference latency.')
BOUNDARY = REFERENCE_BOUNDARY


def validate_report_config(config):
    """Cheap report-side schema gate, in addition to the runner's exact validator."""
    arm = config.get('arm')
    expected = dict(experiment='R94', r94_schema_version=1, model_factory=MODEL_FACTORY,
        initialization=INITIALIZATION, seed=2, global_batch_size=1024,
        logical_global_batch=1024, max_successful_updates=MAX_UPDATES,
        warmup_updates=4320, eval_every_updates=1440, eval_batch_size=256,
        optimizer='AdamW', weight_decay=1e-4, grad_clip=1., dropout=.1,
        rdrop_weight=.35, rdrop_ramp_updates=4320,
        rows_per_problem=dict(train=81000, val=4500, test=4500))
    if arm not in ARMS or any(config.get(k) != v for k, v in expected.items()):
        raise ValueError('R94 schema, fresh initialization or original training controls changed')
    donor = config.get('frozen_donor', {})
    if not isinstance(donor, dict) or donor.get('variant') != VARIANTS[arm]:
        raise ValueError('R94 frozen_donor.variant must be A=random, B=pretrained')
    spec = config.get('model_spec', {})
    if (not isinstance(spec, dict) or spec.get('r94_schema_version') != 1
            or spec.get('donor_width') != 128 or spec.get('encoder_layer_num') != 4
            or spec.get('fusion_location') != 'before_first_original_instance_encoder_layer'
            or set(spec.get('covered_problems', [])) != set(PROBLEMS) - {'TSP', 'ATSP'}
            or spec.get('unsupported_problems') != ['TSP', 'ATSP']):
        raise ValueError('R94 requires the locked early-fusion model_spec')
    loss = config.get('loss', {})
    if any(loss.get(k) != v for k, v in dict(ce=.35, comparison=.35, risk=.02).items()):
        raise ValueError('R94 retains original R77 CE/comparison/risk/R-Drop, not R93 pure CE')
    lr = config.get('lr', {})
    if lr.get('peak') != 1e-4 or lr.get('end') != 1e-6:
        raise ValueError('R94 requires the prelocked AdamW 1e-4 -> 1e-6 schedule')


def locked_inputs(root, arm):
    if arm not in ARMS:
        raise ValueError('R94 requires arm A or B')
    result = shared.locked_inputs(Path(root), arm, experiment='R94', arms=BUDGETS)
    validate_report_config(result[1])
    importlib.import_module(FACTORY).validate_checkpoint_config(result[1])
    return result


def require_cpu(args):
    if str(args.device) != 'cpu':
        raise ValueError('R94 report/compare administration requires --device cpu')


def _donor_controls(donor):
    result = copy.deepcopy(donor)
    # Only donor weight provenance/variant can differ, never the donor interface.
    for key in ('variant', 'state_sha256', 'encoder_state_sha256', 'weights_sha256',
                'checkpoint_sha256', 'checkpoint', 'checkpoint_path', 'checkpoint_loaded', 'random_seed',
                'pretraining_source', 'known_unknowns'):
        result.pop(key, None)
    if 'provenance' in result:
        result['provenance'] = _donor_controls(result['provenance'])
    if isinstance(result.get('known_unknown_pretraining_source'), dict):
        result['known_unknown_pretraining_source'].pop('applies_to_loaded_weights', None)
    return result


def _core_params(config):
    params = copy.deepcopy(config.get('model_params', {}))
    arm = params.pop('r94_arm', config['arm'])
    if arm != config['arm']:
        raise ValueError('R94 parameter arm differs from its locked protocol')
    if 'frozen_donor' in params:
        if params['frozen_donor'].get('variant') != VARIANTS[arm]:
            raise ValueError('R94 parameter donor variant differs from the arm')
        params['frozen_donor'] = _donor_controls(params['frozen_donor'])
    if 'donor_variant' in params and params.pop('donor_variant') != VARIANTS[arm]:
        raise ValueError('R94 parameter donor variant differs from the arm')
    return params


def same_data(left, right, *, matched=False):
    fields = ('data_root', 'rows_per_problem', 'baselines', 'candidate_pools',
              'candidate_pool_ids', 'winner_rule')
    if any(left.get(k) != right.get(k) or k not in left or k not in right for k in fields):
        raise ValueError('R94 requires identical frozen data/pools/native winners/TRAIN SBS')
    if not matched:
        return
    if (left.get('arm'), right.get('arm')) != ARMS:
        raise ValueError('R94 matched contrast must be B pretrained versus A random')
    for config in (left, right):
        validate_report_config(config)
        for field in ('common_trainable_sha256', 'node_fusion_initialization_sha256',
                      'preparation_sha256', 'task_order_sha256'):
            if not config.get(field):
                raise ValueError('R94 matched comparison needs fresh init/data/batch hashes: ' + field)
        if config['model_params'].get('r94_fresh_trainable_sha256') != config['node_fusion_initialization_sha256']:
            raise ValueError('R94 full fresh trainable initialization hash differs from model_params')
        if not config.get('microbatch_caps') or not config.get('checkpoint_rule') or not config.get('inference_rule'):
            raise ValueError('R94 matched comparison needs locked microbatches and VAL/inference rules')
    # Fail closed on newly introduced control fields. Runtime placement may differ.
    runtime = {'arm', 'gpu_type', 'hardware', 'device', 'device_name', 'gpu_uuid',
               'run_root', 'output_root', 'created_utc', 'feature_cache_root', 'frozen_donor', 'model_params'}
    a = {k: v for k, v in left.items() if k not in runtime}
    b = {k: v for k, v in right.items() if k not in runtime}
    if (a != b or _core_params(left) != _core_params(right)
            or _donor_controls(left['frozen_donor']) != _donor_controls(right['frozen_donor'])):
        raise ValueError('Matched R94 A/B may differ only in donor weights/variant and disclosed hardware')


def selected_predictions(run, config, lock, val, test):
    if val['identity']['model_state_sha256'] != test['identity']['model_state_sha256']:
        raise ValueError('R94 diagnostics require same locked TRAIN/VAL/TEST model state')
    packed, methods, summaries, results, paths = helpers.selected_predictions(run, config, lock, val, test)
    model = config['experiment'] + '/' + config['arm']
    return packed, [dict(r, model=model) for r in methods], [dict(r, model=model) for r in summaries], results, paths


def references(config, args):
    result = {}
    for name, path, experiment, budgets in (
            ('R89B', args.r89_root, 'R89', dict(A_adamw=57600, B_muon_hybrid=57600)),
            ('R84A', args.historical_root, 'R84', shared.BUDGETS)):
        path = Path(path)
        run, old, lock, test, val = shared.locked_inputs(path.parent, path.name, experiment=experiment, arms=budgets)
        same_data(config, old)
        result[name] = dict(run=run, config=old, lock=lock, test=test, val=val,
            provenance=dict(model=name, run=str(run), checkpoint_sha256=lock['checkpoint_sha256'],
                same_data=True, retrained=False, newly_matched_control=False,
                role='main historical reference' if name == 'R89B' else 'secondary historical reference'))
    return result


def load_targets(config, historical_config, historical_lock):
    targets, identity = helpers.load_targets(config, historical_config, historical_lock)
    if len(targets) != 36:
        raise ValueError('R94 must retain the fixed R84 target36 without reselection')
    return targets, dict(identity, sha256=file_hash(Path(identity['path'])), target_count=36)


def val_screen(config, lock, val, refs):
    return [dict(row, model='R94/' + config['arm']) for row in r93.val_screen(config, lock, val, refs)]


def diagnostic_tables(run, config, lock, val, test, args):
    packed, methods, summaries, results, paths = selected_predictions(run, config, lock, val, test)
    refs = references(config, args)
    targets, identity = load_targets(config, refs['R84A']['config'], refs['R84A']['lock'])
    pairs, recovery, totals = {}, {}, {}
    for name, ref in refs.items():
        old = selected_predictions(ref['run'], ref['config'], ref['lock'], ref['val'], ref['test'])[0]
        pairs[name], recovery[name], totals[name] = helpers.comparison_tables(
            old, packed, targets, identity, 'R94/' + config['arm'], name)
    return dict(predictions=packed, methods=methods, summaries=summaries, results=results,
        paths=paths, pairs=pairs, recovery=recovery, recovery_summary=totals,
        target_identity=identity, historical={k: v['provenance'] for k, v in refs.items()},
        val_screen=val_screen(config, lock, val, refs))


def inference_timing(run, lock):
    """Consume required runner timing, never benchmark or claim I/O-inclusive E2E."""
    path = Path(run) / 'inference_timing.json'
    if not path.is_file():
        raise ValueError('R94 report requires complete same-checkpoint inference_timing.json')
    record = read_json(path)
    if (record.get('complete') is not True or record.get('checkpoint_sha256') != lock['checkpoint_sha256']
            or record.get('state_hash') != lock['state_hash'] or record.get('repetitions') != 3
            or record.get('includes_native_encoder') is not True
            or record.get('inference_rule') != lock['inference_rule']
            or not record.get('scope') or not record.get('excludes')):
        raise ValueError('R94 inference timing is incomplete or differs from the locked checkpoint/rule')
    raw = record.get('per_problem', [])
    if isinstance(raw, dict):
        raw = [dict(value, problem=key) for key, value in raw.items()]
    if (not isinstance(raw, list) or len(raw) != len(PROBLEMS)
            or any(not isinstance(row, dict) for row in raw)
            or {row.get('problem') for row in raw} != set(PROBLEMS)):
        raise ValueError('R94 timing must cover all18 tasks once')
    rows = []
    for item in raw:
        if item.get('instances') != 128:
            raise ValueError('R94 timing requires the fixed first128 TEST input rows per task')
        row = dict(problem=item['problem'], instances=128, device=record.get('device', 'unspecified'), repetitions=3)
        for scope in ('donor_plus_selector', 'cache_hit_selector'):
            measurement = item.get(scope, {})
            samples = measurement.get('batch_seconds', [])
            value = measurement.get('mean_seconds_per_instance')
            if (not isinstance(samples, list) or len(samples) != 3 or value is None
                    or any(isinstance(x, bool) or not math.isfinite(float(x)) or float(x) <= 0 for x in [*samples, value])
                    or not math.isclose(float(value), sum(samples) / (3 * 128), rel_tol=1e-6, abs_tol=1e-12)):
                raise ValueError('R94 timing requires three finite synchronized repetitions: ' + scope)
            row[scope + '_ms_per_instance'] = 1000 * float(value)
        rows.append(row)
    order = {p: i for i, p in enumerate(PROBLEMS)}
    return dict(path=str(path), status='complete', sha256=file_hash(path),
        measurements=sorted(rows, key=lambda row: order[row['problem']]),
        donor_plus_selector_measured=True, full_system_measured=False, end_to_end_measured=False,
        pdf_panel=True, scope=record['scope'], excludes=record['excludes'],
        note='GPU-resident donor+selector excludes I/O/one-time preparation; cache-hit selector is not full-system latency.')


def hardware_disclosure(configs):
    devices = {arm: {k: configs[arm][k] for k in ('hardware', 'gpu_type', 'device_name', 'gpu_uuid')
                     if k in configs[arm]} for arm in ARMS}
    known = all(devices.values())
    return dict(arms=devices, same_reported_hardware=(devices['A'] == devices['B']) if known else None,
        bitwise_hardware_matched=False,
        note='Hardware differences are disclosed; identical code/batches does not establish bitwise matching.')


def validate_matched_configs(left, right):
    """CPU/dictionary-only integration API; no checkpoint, donor/model or asset reads."""
    same_data(left, right, matched=True)
    return dict(experiment='R94', matched=True, arms=list(ARMS), budgets=BUDGETS.copy(),
        initialization=INITIALIZATION, common_trainable_sha256=left['common_trainable_sha256'],
        node_fusion_initialization_sha256=left['node_fusion_initialization_sha256'],
        task_order_sha256=left['task_order_sha256'], preparation_sha256=left['preparation_sha256'],
        donor_variants=VARIANTS.copy(), hardware=hardware_disclosure(dict(A=left, B=right)),
        external_pretraining_budget_matched=False, training=False, inference=False)


def _timing_panel(pdf, helper, model, lock, timings, page, *, locks=None):
    import matplotlib.pyplot as plt
    for index, (arm, timing) in enumerate(timings.items()):
        fig = common._page(helper, model + ' | ' + arm + ' Inference Timing',
            'First128 locked TEST inputs/task; synchronized GPU wall time, three repetitions; saved evidence only.',
            model, lock if locks is None else locks[arm], page + index)
        cells = [[r['problem'], str(r['instances']), f"{r['donor_plus_selector_ms_per_instance']:.6f}",
                  f"{r['cache_hit_selector_ms_per_instance']:.6f}"] for r in timing['measurements']]
        helper.add_table(fig, ['Problem', 'Batch', 'Donor + selector ms / instance', 'Cache-hit selector ms / instance'],
            cells, [.05, .26, .9, .55], font=9)
        fig.text(.05, .20, 'Inputs already on GPU. Excludes dataset I/O and one-time feature-cache preparation.', fontsize=9)
        fig.text(.05, .15, 'Cached-feature selector latency is NOT full-system latency; donor+selector is not I/O-inclusive E2E.', fontsize=9)
        fig.text(.05, .10, 'TSP/ATSP have unchanged raw paths and no donor; external pretraining budget is unmatched.', fontsize=9)
        fig.text(.05, .05, 'Device: ' + timing['measurements'][0]['device'], fontsize=9)
        pdf.savefig(fig)
        plt.close(fig)


def diagnostics_pdf(output, config, lock, tables, timing):
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    helper = common.reporting_helpers()
    helper.setup_style()
    model = 'R94/' + config['arm']
    path = output / 'target_recovery_and_train.pdf'
    temporary = path.with_name(f'.{path.stem}.{os.getpid()}.pdf')
    try:
        with PdfPages(temporary) as pdf:
            for page, ref in enumerate(('R89B', 'R84A'), 1):
                fig = common._page(helper, model + ' | Fixed target36 vs ' + ref,
                    'Native FP64 winners; same unique full18 VAL checkpoint on TRAIN/VAL/TEST.', model, lock, page)
                helper.add_table(fig, ['Split', 'Winners', 'Correct', 'Rescued', 'Lost', 'New false', 'Recall (%)'],
                    [[r['split'], str(r['winner_count']), str(r['correct_count']), str(r['rescued']),
                      str(r['lost']), str(r['new_false_pick_count']),
                      '-' if r['recall_pct'] is None else f"{r['recall_pct']:.4f}"]
                     for r in tables['recovery_summary'][ref]], [.05, .62, .9, .2], font=10)
                helper.add_table(fig, ['Split', 'Top1 delta pp', 'Corrected', 'Harmed', 'Net cost delta', 'Regret delta pp'],
                    [[r['split'], f"{r['top1_delta_pp']:.6f}", str(r['corrected']), str(r['harmed']),
                      f"{r['mean_cost_delta']:.9f}", f"{r['actual_regret_delta_pp']:.6f}"]
                     for r in tables['pairs'][ref] if r['problem'] == 'ALL'], [.05, .32, .9, .2], font=10)
                for y, text in ((.25, 'R89B main / R84A secondary historical; A is the matched random-donor control.'),
                        (.20, 'Original R77 CE .35 / comparison .35 / risk .02 / R-Drop .35; dropout .1.'),
                        (.15, 'Early trainable inter-node fusion, not R50 pool-first readout. Improvement remains unproven.'),
                        (.10, '16-task donor input coverage, not16-task pretraining; TSP/ATSP raw paths unchanged.'),
                        (.05, 'External donor pretraining budget unmatched. No decoder/routes/new cost labels/MTL.')):
                    fig.text(.05, y, text, fontsize=9)
                pdf.savefig(fig)
                plt.close(fig)
            _timing_panel(pdf, helper, model, lock, {config['arm']: timing}, 3)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return str(path)


def learning_curves(run, config, lock):
    import matplotlib.pyplot as plt
    history = read_json(run / 'history.json')
    if not history or not any(row.get('train') for row in history) or not any(row.get('val') for row in history):
        raise ValueError('R94 requires saved TRAIN and VAL learning curves, not fabricated traces')
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle('R94/' + config['arm'] + ' | original R77 loss; one locked VAL checkpoint', fontsize=14)
    for key in ('loss', 'ce', 'cmp', 'risk', 'kl'):
        points = [(r['successful_updates'], r['optimization'][key]) for r in history if key in r.get('optimization', {})]
        if points:
            axes[0, 0].plot(*zip(*points), label=key)
    axes[0, 0].set_title('TRAIN original R77 loss components (R-Drop retained)')
    if axes[0, 0].lines:
        axes[0, 0].legend(fontsize=8)
    for axis, key, factor, title in (
            (axes[0, 1], 'macro_ce', 1., 'Evaluation CE: original maximin'),
            (axes[0, 2], 'macro_top1', 100., 'Native Top1 (%)'),
            (axes[1, 0], 'macro_mean_cost', 1., 'Task-equal mean cost (descriptive)'),
            (axes[1, 1], 'macro_actual_regret_pct', 1., 'Actual regret (%)'),
            (axes[1, 2], 'macro_vs_sbs_pct', 1., 'vs fixed TRAIN SBS (%)')):
        for split in ('train', 'val'):
            points = [(r['successful_updates'], r[split]['macro'][key] * factor)
                      for r in history if r.get(split) is not None and key in r[split]['macro']]
            if points:
                axis.plot(*zip(*points), label=split.upper())
        axis.set_title(title)
        if axis.lines:
            axis.legend()
    for axis in axes.flat:
        axis.axvline(lock['selected_update'], color='#c45358', linestyle='--', linewidth=1)
        axis.set_xlabel('Successful logical-batch1024 updates')
        axis.grid(alpha=.2)
    fig.tight_layout(rect=(0, 0, 1, .95))
    try:
        for suffix in ('pdf', 'png'):
            fig.savefig(run / 'full_model_statistics' / ('learning_curves.' + suffix))
    finally:
        plt.close(fig)


def _completed_lib(run, lock):
    lib = read_json(run / 'lib_stage_complete.json')
    if (lib.get('complete') is not True or lib.get('exit_status') != 0
            or lib.get('checkpoint_sha256') != lock['checkpoint_sha256']
            or set(lib.get('datasets', [])) != {'TSPLIB', 'CVRPLIB'}):
        raise ValueError('R94 requires same-checkpoint TSPLIB/CVRPLIB completion with rc0')


def completed_arm_report(run, lock):
    _completed_lib(run, lock)
    post = read_json(run / 'postprocess_status.json')
    protocol = read_json(run / 'full_model_statistics/protocol.json')
    if (post.get('complete') is not True or post.get('phase') != 'complete' or post.get('exit_status') != 0
            or protocol.get('complete') is not True
            or protocol.get('checkpoint_lock', {}).get('checkpoint_sha256') != lock['checkpoint_sha256']):
        raise ValueError('R94 comparison requires completed same-lock TRAIN/VAL/TEST/LIB/PDF reports')
    for name in REQUIRED_PDFS:
        path = run / 'full_model_statistics' / name
        if not path.is_file() or not path.stat().st_size:
            raise ValueError('R94 arm PDF missing: ' + name)
    if protocol.get('inference_timing', {}).get('sha256') != inference_timing(run, lock)['sha256']:
        raise ValueError('R94 timing changed after PDF delivery; refresh the CPU arm report')


def lib_stage(args):
    if not str(args.device).startswith('cuda:'):
        raise ValueError('R94 LIB requires an explicit launcher-allocated --device cuda:N')
    run, config, lock, test, _ = locked_inputs(args.root, args.arm)
    with (run / 'lib_stage.lock').open('a') as claim:
        fcntl.flock(claim, fcntl.LOCK_EX | fcntl.LOCK_NB)

        def phase(name, **fields):
            write_json(run / 'lib_stage_status.json', dict(experiment='R94', arm=args.arm,
                phase=name, updated_utc=utc(), **fields))

        try:
            reference = previous.validate_nss_reference(config, args.nss_root)
            results = common.r80_lib(args.root, run, config, lock, args.device,
                args.lib_batch_size, phase, test['identity']['model_state_sha256'], factory_module=FACTORY)
            if set(results) != {'TSPLIB', 'CVRPLIB'}:
                raise ValueError('R94 LIB factory must finish both native LIB datasets')
            write_json(run / 'lib_stage_complete.json', dict(experiment='R94', arm=args.arm,
                complete=True, exit_status=0, updated_utc=utc(), state_hash=lock['state_hash'],
                checkpoint_sha256=lock['checkpoint_sha256'], model_state_sha256=test['identity']['model_state_sha256'],
                nss_reference=reference, datasets=list(results)))
            phase('complete', exit_status=0, checkpoint_sha256=lock['checkpoint_sha256'])
            return results
        except BaseException as error:
            phase('lib_failed', exit_status=1, error=str(error), automatic_retry=False)
            raise


def report(args):
    require_cpu(args)
    run, config, lock, test, val = locked_inputs(args.root, args.arm)
    with (run / 'postprocess.lock').open('a') as claim:
        fcntl.flock(claim, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            write_json(run / 'postprocess_status.json', dict(experiment='R94', arm=args.arm,
                phase='loading_saved_predictions', updated_utc=utc()))
            _completed_lib(run, lock)
            timing = inference_timing(run, lock)
            reference = previous.validate_nss_reference(config, args.nss_root)
            predictions, metrics, methods = shared.load_saved_splits(run, config, lock, test, val)
            shared.saved_libs(run, config, lock, test, predictions, metrics, methods)
            comparisons, alignment = shared.nss_comparisons(config, lock, predictions, args.nss_root)
            tables = diagnostic_tables(run, config, lock, val, test, args)
            result = shared.build_reports(run, config, lock, test, val,
                predictions, metrics, methods, comparisons, alignment)
            output = run / 'full_model_statistics'
            learning_curves(run, config, lock)
            for name, rows in (
                    ('method_selection_train_val_test.csv', tables['methods']),
                    ('zero_selection_summary.csv', tables['summaries']),
                    ('train_metrics_by_problem.csv', helpers.train_metric_rows(config, lock, tables)),
                    ('val_empirical_screen.csv', tables['val_screen']),
                    ('inference_timing.csv', timing['measurements'])):
                common.write_csv(output / name, rows)
            for ref in tables['historical']:
                for stem, data in (('paired_changes', tables['pairs']), ('target_recovery', tables['recovery']),
                                   ('target_recovery_summary', tables['recovery_summary'])):
                    common.write_csv(output / (stem + '_vs_' + ref + '.csv'), data[ref])
            for stem, data in (('target_recovery', tables['recovery']), ('target_recovery_summary', tables['recovery_summary'])):
                common.write_csv(output / (stem + '.csv'), data['R84A'])
            result.update(target_recovery_pdf=diagnostics_pdf(output, config, lock, tables, timing),
                historical_reference=tables['historical'], nss_reference=reference,
                inference_timing=timing, joint_val_pass=all(r['joint_val_pass'] for r in tables['val_screen']))
            provenance = read_json(output / 'protocol.json')
            provenance.update(r94_schema_version=1, model_spec=config['model_spec'],
                frozen_donor=config['frozen_donor'], initialization=config['initialization'],
                target_identity=tables['target_identity'], historical_reference=tables['historical'],
                nss_reference=reference, objective=config.get('objective'), loss=config['loss'],
                evidence_boundary=REFERENCE_BOUNDARY, inference_timing=timing,
                rdrop=True, external_pretraining_budget_matched=False,
                val_empirical_screen=tables['val_screen'])
            write_json(output / 'protocol.json', provenance)
            notes = output / 'RESULTS.md'
            notes.write_text(notes.read_text(encoding='utf-8') + '\n' + REFERENCE_BOUNDARY + '\n\n'
                + 'Timing: ' + timing['status'] + '; see target_recovery_and_train.pdf.\n', encoding='utf-8')
            write_json(run / 'postprocess_status.json', dict(experiment='R94', arm=args.arm,
                phase='complete', exit_status=0, completed_utc=utc(), **result))
            return result
        except BaseException as error:
            write_json(run / 'postprocess_status.json', dict(experiment='R94', arm=args.arm,
                phase='postprocess_failed', exit_status=1, updated_utc=utc(), error=str(error),
                traceback=traceback.format_exc(), automatic_retry=False))
            raise


def compare(args):
    require_cpu(args)
    # Check both protocols before expensive saved-array reads; never infer/select.
    inputs = {arm: locked_inputs(args.root, arm) for arm in ARMS}
    configs = {arm: inputs[arm][1] for arm in ARMS}
    matched_controls = validate_matched_configs(configs['A'], configs['B'])
    selected, locks, metrics, screens, timings = {}, {}, [], [], {}
    for arm, (run, config, lock, test, val) in inputs.items():
        completed_arm_report(run, lock)
        selected[arm], _, _, results, _ = selected_predictions(run, config, lock, val, test)
        locks[arm] = lock
        refs = references(config, args)
        screens.extend(val_screen(config, lock, val, refs))
        timings[arm] = inference_timing(run, lock)
        for split, result in results.items():
            for problem, saved in [('ALL', result['macro']), *result['per_problem'].items()]:
                prefix = 'macro_' if problem == 'ALL' else ''
                keys = ('top1', 'mean_cost', 'sbs_cost', 'oracle_cost', 'vs_sbs_pct',
                        'vs_oracle_pct', 'actual_regret_pct')
                metrics.append(dict(model='R94/' + arm, split=split, problem=problem,
                    donor_variant=VARIANTS[arm], selected_update=lock['selected_update'],
                    checkpoint_sha256=lock['checkpoint_sha256'],
                    **{key: saved[prefix + key] for key in keys}))
    targets, identity = load_targets(configs['B'], refs['R84A']['config'], refs['R84A']['lock'])
    pairs, recovery, totals = helpers.comparison_tables(selected['A'], selected['B'], targets,
        identity, 'R94/B', 'R94/A')
    output = Path(args.root) / 'comparison_artifacts'
    output.mkdir(parents=True, exist_ok=True)
    for name, rows in (('comparison.csv', metrics), ('paired_changes_B_vs_A.csv', pairs),
            ('target_recovery_B_vs_A.csv', recovery), ('target_recovery_summary_B_vs_A.csv', totals),
            ('val_empirical_screen.csv', screens)):
        common.write_csv(output / name, rows)
    hardware = hardware_disclosure(configs)
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    helper = common.reporting_helpers()
    helper.setup_style()
    path = output / 'COMPARISON.pdf'
    temporary = path.with_name(f'.{path.stem}.{os.getpid()}.pdf')
    try:
        with PdfPages(temporary) as pdf:
            fig = common._page(helper, 'R94 | Pretrained B versus random frozen donor A',
                'Same fresh selector + early node fusion; original R77 four inter-node layers and full loss.',
                'R94 B vs A', locks['B'], 1)
            helper.add_table(fig, ['Split', 'Top1 delta pp', 'Corrected', 'Harmed', 'Net cost delta', 'Regret delta pp'],
                [[r['split'], f"{r['top1_delta_pp']:.6f}", str(r['corrected']), str(r['harmed']),
                  f"{r['mean_cost_delta']:.9f}", f"{r['actual_regret_delta_pp']:.6f}"]
                 for r in pairs if r['problem'] == 'ALL'], [.05, .62, .9, .2], font=10)
            helper.add_table(fig, ['Split', 'Target correct', 'Rescued', 'Lost', 'New false', 'Recall (%)'],
                [[r['split'], str(r['correct_count']), str(r['rescued']), str(r['lost']),
                  str(r['new_false_pick_count']), '-' if r['recall_pct'] is None else f"{r['recall_pct']:.4f}"]
                 for r in totals], [.05, .35, .9, .2], font=10)
            for y, text in ((.29, 'A/B differ only in frozen donor initialization; early fusion + selector freshly initialized identically.'),
                    (.24, 'Original CE .35 / comparison .35 / risk .02 / R-Drop .35; neither arm is R93 pure CE.'),
                    (.19, 'External pretraining budget unmatched; input coverage16, not16-task pretraining; TSP/ATSP raw.'),
                    (.14, 'R50 pool-first is negative evidence. No decoder/routes/new cost labels/MTL restoration.'),
                    (.09, 'R89B main / R84A secondary historical. Hardware identities are disclosed on the next page.'),
                    (.04, 'ALL deltas equal-weight18 tasks; native FP64 winners, TRAIN SBS; no TEST/LIB model selection.')):
                fig.text(.05, y, text, fontsize=8.5)
            pdf.savefig(fig)
            plt.close(fig)
            fig = common._page(helper, 'R94 | Historical VAL screens and hardware',
                'Descriptive locked-model diagnostics, not checkpoint/model reselection.', 'R94 B vs A', locks['B'], 2)
            helper.add_table(fig, ['Arm / reference', 'Top1 delta pp', 'Cost delta', 'Regret delta pp', 'Pass'],
                [[r['model'] + ' / ' + r['baseline'], f"{r['top1_delta_pp']:.6f}",
                  f"{r['mean_cost_delta']:.9f}", f"{r['actual_regret_delta_pp']:.6f}", str(r['pass_reference'])]
                 for r in screens], [.05, .49, .9, .3], font=10)
            import textwrap
            for index, arm in enumerate(ARMS):
                text = arm + ': ' + json.dumps(hardware['arms'][arm], sort_keys=True)
                fig.text(.05, .39 - index * .12, '\n'.join(textwrap.wrap(text, 110)), fontsize=9)
            fig.text(.05, .10, 'Hardware differences are disclosed; no claim of bitwise hardware matching.', fontsize=10)
            pdf.savefig(fig)
            plt.close(fig)
            _timing_panel(pdf, helper, 'R94 B vs A', locks['B'], timings, 3, locks=locks)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    result = dict(experiment='R94', r94_schema_version=1, complete=True, exit_status=0,
        checkpoints=locks, comparison_pdf=str(path), evidence_boundary=REFERENCE_BOUNDARY,
        matched_difference='B pinned pretrained versus A random weights in the same frozen ReLD128 donor',
        target_identity=identity, hardware=hardware, inference_timing=timings,
        matched_controls=matched_controls,
        external_pretraining_budget_matched=False, training=False, inference=False, checkpoint_reselection=False)
    # The manifest is last: a failed PDF/copy/notes operation cannot claim completion.
    lines = ['# R94: Frozen Donor Early Node Fusion', '', REFERENCE_BOUNDARY, '',
        '| Model | Split | Top1 (%) | mean_cost | SBS cost | Oracle cost | vs_SBS (%) | vs_Oracle (%) | actual regret (%) |',
        '| --- | --- | --- | --- | --- | --- | --- | --- | --- |']
    for r in metrics:
        if r['problem'] == 'ALL':
            lines.append(f"| {r['model']} | {r['split']} | {100*r['top1']:.6f} | {r['mean_cost']:.9f} | "
                f"{r['sbs_cost']:.9f} | {r['oracle_cost']:.9f} | {r['vs_sbs_pct']:.6f} | "
                f"{r['vs_oracle_pct']:.6f} | {r['actual_regret_pct']:.6f} |")
    (Path(args.root) / 'RESULTS.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    shutil.copyfile(path, Path(args.root) / 'RESULTS.pdf')
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
        value = read_json(path) if path.is_file() else dict(experiment='R94', phase='not_started')
        print(json.dumps(value, indent=2))
        return value
    return lib_stage(args) if args.command == 'lib' else report(args)


if __name__ == '__main__':
    main()

"""R99 locked CPU reports over saved predictions, with a separate GPU LIB stage.

Reuse the native R80/R84 report and R87/R93 paired-recovery functions, not
R94 donor metadata. No historical module globals are modified. A/B differences
are limited to input column competition and disclosed runtime placement.
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
from . import r93_reporting as r93
from .r76_experiment import file_hash, read_json, utc, write_json


ARMS = ('A', 'B')
BUDGETS = dict.fromkeys(ARMS, 57600)
VARIANTS = dict(A='row', B='balanced')
DEFAULT_ROOT = Path(__file__).resolve().parent / 'runs/R99_global_competition_input'
FACTORY = 'code.V4.r99_experiment'
MODEL_FACTORY = FACTORY + '.load_checkpoint'
INITIALIZATION = 'scratch_shared_r77_plus_input_competition'
shared, previous, common = helpers.shared, helpers.previous, helpers.common
DEFAULT_NSS, DEFAULT_R84 = helpers.DEFAULT_NSS, helpers.DEFAULT_R84
DEFAULT_R89B = DEFAULT_ROOT.parent / 'R89_optimizer_geometry/B_muon_hybrid'
PROBLEMS = tuple(helpers.PROBLEMS)
REQUIRED_PDFS = ('MODEL_STATISTICS.pdf', 'selection_distributions.pdf',
    'target_recovery_and_train.pdf', 'learning_curves.pdf', 'training_and_method_diagnostics.pdf')
TIMING_SCOPES = ('transform_only', 'transform_plus_selector', 'cache_hit_selector')
REFERENCE_BOUNDARY = (
    'A uses row-softmax distance messages; B uses fixed64 Sinkhorn row/column competition. '
    'Both use the same input-only32 outgoing/incoming one/two-hop raw-attribute channels, '
    'projected before the original four instance layers, including directed matrix ATSP. '
    'Identical fresh R84 seed2 parameters plus shared new projection seed990003; no pretrained '
    'donor, new target or auxiliary loss. Original R88B physical TSP node0, hand34/methodID, '
    'joint2 and score MLP/maximin remain. Original R77 CE=.35, comparison=.35, risk=.02, '
    'R-Drop=.35, dropout=.1. Input-only representational accessibility, not new information, '
    'an assignment solver, VRP bound, feasibility test or B-semantics repair. Finite64 '
    'marginal residuals do not establish exact convergence. R89B main / R84A secondary are '
    'historical same-data references, not newly matched controls. Fixed R84 target36; '
    'zero-count movement alone is not success. Hardware differences are disclosed, not '
    'bitwise hardware matching. Development-exposed TEST; unique full18 minimum VAL '
    'macro_vs_sbs checkpoint; no TEST/LIB tuning or per-task assembly. GPU-resident '
    'transform+selector excludes I/O and host transfer; cache-hit selector is not full inference.')
BOUNDARY = REFERENCE_BOUNDARY


def validate_report_config(config):
    """Dictionary-only schema checks; never instantiate a model or open its weights."""
    arm = config.get('arm')
    expected = dict(experiment='R99', r99_schema_version=1, model_factory=MODEL_FACTORY,
        initialization=INITIALIZATION, seed=2, global_batch_size=1024, logical_global_batch=1024,
        max_successful_updates=57600, warmup_updates=4320, eval_every_updates=1440,
        train_eval_every_updates=7200, eval_batch_size=256, optimizer='AdamW',
        weight_decay=1e-4, grad_clip=1., dropout=.1, rdrop_weight=.35, rdrop_ramp_updates=4320,
        projection_seed=990003, relation_variants=VARIANTS,
        rows_per_problem=dict(train=81000, val=4500, test=4500), external_pretraining=False,
        new_target=False, auxiliary_loss=False)
    if arm not in ARMS or any(config.get(key) != value for key, value in expected.items()):
        raise ValueError('R99 fresh initialization or original full18 training controls changed')
    spec = config.get('model_spec', {})
    transform = config.get('relation_transform', {})
    if (config.get('relation_variant') != VARIANTS[arm]
            or spec.get('projection_seed') != 990003 or spec.get('external_pretraining') is not False
            or spec.get('fusion_location') != 'before_original_four_instance_layers'
            or spec.get('objective') != 'original_R77'
            or spec.get('transform') != transform
            or transform.get('width') != 32 or transform.get('iterations') != 64
            or transform.get('temperature') != .05 or transform.get('labels_used') is not False
            or transform.get('precision') != 'FP32_transform_then_FP16'):
        raise ValueError('R99 requires fixed input-only32 pre-layer messages for both arms')
    loss = config.get('loss', {})
    if any(loss.get(k) != v for k, v in dict(ce=.35, comparison=.35, risk=.02).items()):
        raise ValueError('R99 retains the original R77 objective, not R93 pure CE or a new loss')
    lr = config.get('lr', {})
    if lr.get('peak') != 1e-4 or lr.get('end') != 1e-6:
        raise ValueError('R99 requires AdamW 1e-4 to1e-6')
    if config.get('microbatch_caps') != {p: 352 if p in ('TSP', 'CVRP') else 1024 for p in PROBLEMS}:
        raise ValueError('R99 requires TSP/CVRP352 and other-task1024 micros')


def locked_inputs(root, arm):
    if arm not in ARMS:
        raise ValueError('R99 requires arm A or B')
    result = shared.locked_inputs(Path(root), arm, experiment='R99', arms=BUDGETS)
    validate_report_config(result[1])
    importlib.import_module(FACTORY).validate_checkpoint_config(result[1])
    return result


def require_cpu(args):
    if str(args.device) != 'cpu':
        raise ValueError('R99 report/compare administration requires --device cpu')


def same_data(left, right, *, matched=False):
    fields = ('data_root', 'rows_per_problem', 'baselines', 'candidate_pools',
              'candidate_pool_ids', 'winner_rule')
    if any(key not in left or key not in right or left[key] != right[key] for key in fields):
        raise ValueError('R99 requires identical frozen data, legal pools, native winners and TRAIN SBS')
    if not matched:
        return
    if (left.get('arm'), right.get('arm')) != ARMS:
        raise ValueError('R99 matched contrast is balanced B versus row-softmax A')
    params = []
    for config in (left, right):
        validate_report_config(config)
        for key in ('common_trainable_sha256', 'relation_initialization_sha256',
                    'retained_parameter_sha256', 'new_parameter_sha256',
                    'preparation_sha256', 'task_order_sha256', 'checkpoint_rule', 'inference_rule'):
            if not config.get(key):
                raise ValueError('R99 matched comparison requires identity: ' + key)
        current = copy.deepcopy(config.get('model_params', {}))
        if current.pop('r99_arm', None) != config['arm']:
            raise ValueError('R99 model parameter arm differs from locked protocol')
        for key, param in (('relation_initialization_sha256', 'r99_fresh_trainable_sha256'),
                ('retained_parameter_sha256', 'r99_common_parameter_sha256'),
                ('new_parameter_sha256', 'r99_new_parameter_sha256')):
            if config[key] != current.get(param):
                raise ValueError('R99 common/new parameter hashes must be explicit and match')
        if config['retained_parameter_sha256'] != config['common_trainable_sha256']:
            raise ValueError('R99 retained initialization differs from the full fresh R84 state')
        params.append(current)
    runtime = {'arm', 'relation_variant', 'feature_cache_root', 'model_params', 'gpu_type',
        'hardware', 'device', 'device_name', 'gpu_uuid', 'run_root', 'output_root', 'created_utc'}
    if ({k: v for k, v in left.items() if k not in runtime}
            != {k: v for k, v in right.items() if k not in runtime} or params[0] != params[1]):
        raise ValueError('Matched R99 A/B may differ only in input column competition and disclosed placement')


def hardware_disclosure(configs):
    devices = {arm: {key: configs[arm][key] for key in ('hardware', 'gpu_type', 'device_name', 'gpu_uuid')
                    if key in configs[arm]} for arm in ARMS}
    return dict(arms=devices, same_reported_hardware=(devices['A'] == devices['B']) if all(devices.values()) else None,
        bitwise_hardware_matched=False, note='Disclosed placement is not bitwise hardware matching.')


def validate_matched_configs(left, right):
    same_data(left, right, matched=True)
    return dict(experiment='R99', matched=True, arms=list(ARMS), budgets=BUDGETS.copy(),
        initialization=INITIALIZATION, relation_variants=VARIANTS.copy(),
        common_trainable_sha256=left['common_trainable_sha256'],
        relation_initialization_sha256=left['relation_initialization_sha256'],
        new_parameter_sha256=left['new_parameter_sha256'], projection_seed=990003,
        task_order_sha256=left['task_order_sha256'], preparation_sha256=left['preparation_sha256'],
        hardware=hardware_disclosure(dict(A=left, B=right)), training=False, inference=False)


def selected_predictions(run, config, lock, val, test):
    if val['identity']['model_state_sha256'] != test['identity']['model_state_sha256']:
        raise ValueError('R99 diagnostics require the same locked TRAIN/VAL/TEST model state')
    packed, methods, summaries, results, paths = helpers.selected_predictions(run, config, lock, val, test)
    model = config['experiment'] + '/' + config['arm']
    return packed, [dict(row, model=model) for row in methods], [dict(row, model=model) for row in summaries], results, paths


def references(config, args):
    # This R93 function checks only shared data/pools/SBS, not R93 objectives.
    return r93.references(config, args)


def load_targets(config, historical_config, historical_lock):
    targets, identity = helpers.load_targets(config, historical_config, historical_lock)
    if len(targets) != 36:
        raise ValueError('R99 requires the fixed R84 target36 without reselection')
    return targets, dict(identity, sha256=file_hash(Path(identity['path'])), target_count=36)


def val_screen(config, lock, val, refs):
    return [dict(row, model='R99/' + config['arm']) for row in r93.val_screen(config, lock, val, refs)]


def diagnostic_tables(run, config, lock, val, test, args):
    packed, methods, summaries, results, paths = selected_predictions(run, config, lock, val, test)
    refs = references(config, args)
    targets, identity = load_targets(config, refs['R84A']['config'], refs['R84A']['lock'])
    pairs, recovery, totals = {}, {}, {}
    for name, ref in refs.items():
        old = selected_predictions(ref['run'], ref['config'], ref['lock'], ref['val'], ref['test'])[0]
        pairs[name], recovery[name], totals[name] = helpers.comparison_tables(
            old, packed, targets, identity, 'R99/' + config['arm'], name)
    return dict(predictions=packed, methods=methods, summaries=summaries, results=results, paths=paths,
        pairs=pairs, recovery=recovery, recovery_summary=totals, target_identity=identity,
        historical={key: value['provenance'] for key, value in refs.items()},
        val_screen=val_screen(config, lock, val, refs))


def inference_timing(run, lock):
    """Require real online transforms, separate preprocessing and all18 timing rows."""
    path = Path(run) / 'inference_timing.json'
    record = read_json(path)
    required = dict(experiment='R99', complete=True, repetitions=3, warmup_batches=1,
        includes_input_transform=True, raw_input_source='original_base_cache_without_relation_fields',
        online_transform_verified=True, online_fp16_matches_cache=True,
        full_system_measured=False, end_to_end_measured=False, model_reselection=False,
        checkpoint_sha256=lock['checkpoint_sha256'], state_hash=lock['state_hash'],
        inference_rule=lock['inference_rule'])
    arm = record.get('arm')
    if (arm not in ARMS or (lock.get('arm') is not None and arm != lock['arm'])
            or record.get('variant') != VARIANTS[arm]
            or any(record.get(key) != value for key, value in required.items())
            or not record.get('scope') or not record.get('excludes')):
        raise ValueError('R99 timing requires same-lock verified online transform, not cached features as raw')
    raw, prep = record.get('per_problem'), record.get('preprocessing', {})
    test_prep = record.get('test_preprocessing', {})
    if (not isinstance(raw, list) or len(raw) != len(PROBLEMS)
            or any(not isinstance(row, dict) for row in raw)
            or {row.get('problem') for row in raw} != set(PROBLEMS)):
        raise ValueError('R99 timing must cover all18 tasks exactly once')
    def positive(value, *, zero=False):
        return (not isinstance(value, bool) and isinstance(value, (int, float))
                and math.isfinite(value) and (value >= 0 if zero else value > 0))
    if (prep.get('complete') is not True or prep.get('experiment') != 'R99'
            or prep.get('arm') != arm or prep.get('variant') != VARIANTS[arm]
            or prep.get('splits') != ['train', 'val'] or not positive(prep.get('seconds'), zero=True)):
        raise ValueError('R99 timing needs a separate arm-local TRAIN/VAL preprocessing receipt')
    if (test_prep.get('complete') is not True or test_prep.get('experiment') != 'R99'
            or test_prep.get('arm') != arm or test_prep.get('variant') != VARIANTS[arm]
            or test_prep.get('splits') != ['test'] or not positive(test_prep.get('seconds'), zero=True)
            or test_prep.get('checkpoint_sha256') != lock['checkpoint_sha256']
            or test_prep.get('state_hash') != lock['state_hash']):
        raise ValueError('R99 timing needs the same-lock TEST preprocessing receipt')
    rows = []
    for item in raw:
        residual = item.get('max_marginal_residual')
        if item.get('instances') != 128 or not positive(residual, zero=True):
            raise ValueError('R99 requires first128 rows and finite marginal residuals, not exact convergence')
        row = dict(problem=item['problem'], instances=128, device=record.get('device', 'unspecified'),
            repetitions=3, max_marginal_residual=residual)
        for scope in TIMING_SCOPES:
            value = item.get(scope, {})
            samples, mean = value.get('batch_seconds'), value.get('mean_seconds_per_instance')
            if (not isinstance(samples, list) or len(samples) != 3
                    or not all(positive(x) for x in samples) or not positive(mean)
                    or not math.isclose(mean, sum(samples) / (3 * 128), rel_tol=1e-6, abs_tol=1e-12)):
                raise ValueError('R99 timing requires three finite synchronized measurements: ' + scope)
            row[scope + '_ms_per_instance'] = 1000 * mean
        rows.append(row)
    rows.sort(key=lambda row: PROBLEMS.index(row['problem']))
    return dict(path=str(path), status='complete', sha256=file_hash(path), measurements=rows,
        preprocessing_seconds=prep['seconds'], preprocessing=prep,
        test_preprocessing_seconds=test_prep['seconds'], test_preprocessing=test_prep,
        transform_plus_selector_measured=True, full_system_measured=False, end_to_end_measured=False,
        pdf_panel=True, scope=record['scope'], excludes=record['excludes'],
        note='GPU-resident raw transform+selector includes FP16 quantization, excludes I/O/host transfer; cache hit is not full inference.')


def _timing_panel(pdf, helper, model, lock, timings, page, *, locks=None):
    import matplotlib.pyplot as plt
    for index, (arm, timing) in enumerate(timings.items()):
        fig = common._page(helper, model + ' | ' + arm + ' Input Transform / Inference Timing',
            'First128 locked TEST input rows/task; synchronized GPU wall time, three repetitions; saved evidence.',
            model, lock if locks is None else locks[arm], page + index)
        cells = [[r['problem'], f"{r['transform_only_ms_per_instance']:.6f}",
            f"{r['transform_plus_selector_ms_per_instance']:.6f}",
            f"{r['cache_hit_selector_ms_per_instance']:.6f}", f"{r['max_marginal_residual']:.6g}"]
            for r in timing['measurements']]
        helper.add_table(fig, ['Problem', 'Transform ms/row', 'Transform + selector ms/row',
            'Cache-hit selector ms/row', 'Marginal residual'], cells, [.04, .28, .92, .54], font=8.5)
        for y, text in ((.22, f"Feature preparation stage wall time: TRAIN/VAL {timing['preprocessing_seconds']:.3f}s; TEST {timing['test_preprocessing_seconds']:.3f}s (cache reuse allowed)."),
                (.17, 'Original inputs already on GPU; online raw path has no relation_nodes/relation_node_mask.'),
                (.12, 'Transform includes FP16 quantization. All18 tasks include ATSP; no solver decoder.'),
                (.07, 'Excludes dataset I/O and host transfer. Cache-hit selector is not full inference or I/O-inclusive E2E.'),
                (.02, 'Finite64 residuals are recorded, not proof of exact Sinkhorn convergence. Device: '
                    + timing['measurements'][0]['device'])):
            fig.text(.04, y, text, fontsize=9)
        pdf.savefig(fig)
        plt.close(fig)


def diagnostics_pdf(output, config, lock, tables, timing):
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    helper = common.reporting_helpers()
    helper.setup_style()
    model = 'R99/' + config['arm']
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
                for y, text in ((.25, 'Matched A/B: row-softmax versus fixed64 Sinkhorn, same input-only32 before four instance layers.'),
                        (.20, 'Original R77 CE .35 / comparison .35 / risk .02 / R-Drop .35, dropout .1; no new target/loss.'),
                        (.15, 'R89B main / R84A secondary historical, neither a newly matched control. Fixed R84 target36.'),
                        (.10, 'Not an assignment solver, VRP bound, feasibility test or B-semantics repair.'),
                        (.05, 'Zero-count movement alone is not success. Development-exposed TEST; no TEST/LIB selection.')):
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
        raise ValueError('R99 requires saved TRAIN/VAL curves, never fabricated traces')
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle('R99/' + config['arm'] + ' | unchanged R77 objective; one locked VAL checkpoint', fontsize=14)
    for key in ('loss', 'ce', 'cmp', 'risk', 'kl'):
        points = [(r['successful_updates'], r['optimization'][key]) for r in history if key in r.get('optimization', {})]
        if points:
            axes[0, 0].plot(*zip(*points), label=key)
    axes[0, 0].set_title('TRAIN original loss components (R-Drop retained)')
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
            or lib.get('state_hash') != lock['state_hash']
            or set(lib.get('datasets', [])) != {'TSPLIB', 'CVRPLIB'}):
        raise ValueError('R99 requires same-checkpoint TSPLIB/CVRPLIB completion with rc0')


def completed_arm_report(run, lock):
    _completed_lib(run, lock)
    post = read_json(run / 'postprocess_status.json')
    protocol = read_json(run / 'full_model_statistics/protocol.json')
    if (post.get('complete') is not True or post.get('phase') != 'complete' or post.get('exit_status') != 0
            or protocol.get('complete') is not True or protocol.get('experiment') != 'R99'
            or protocol.get('checkpoint_lock', {}).get('checkpoint_sha256') != lock['checkpoint_sha256']):
        raise ValueError('R99 comparison requires completed same-lock TRAIN/VAL/TEST/LIB/PDF reports')
    for name in REQUIRED_PDFS:
        path = run / 'full_model_statistics' / name
        if not path.is_file() or not path.stat().st_size:
            raise ValueError('R99 arm PDF missing: ' + name)
    if protocol.get('inference_timing', {}).get('sha256') != inference_timing(run, lock)['sha256']:
        raise ValueError('R99 timing changed after delivery; refresh the CPU report')


def lib_stage(args):
    if not str(args.device).startswith('cuda:'):
        raise ValueError('R99 LIB requires an explicit launcher-allocated --device cuda:N')
    run, config, lock, test, _ = locked_inputs(args.root, args.arm)
    with (run / 'lib_stage.lock').open('a') as claim:
        fcntl.flock(claim, fcntl.LOCK_EX | fcntl.LOCK_NB)
        def phase(name, **fields):
            write_json(run / 'lib_stage_status.json', dict(experiment='R99', arm=args.arm,
                phase=name, updated_utc=utc(), **fields))
        try:
            reference = previous.validate_nss_reference(config, args.nss_root)
            results = common.r80_lib(args.root, run, config, lock, args.device,
                args.lib_batch_size, phase, test['identity']['model_state_sha256'], factory_module=FACTORY)
            if set(results) != {'TSPLIB', 'CVRPLIB'}:
                raise ValueError('R99 LIB must finish both native datasets')
            write_json(run / 'lib_stage_complete.json', dict(experiment='R99', arm=args.arm,
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
            write_json(run / 'postprocess_status.json', dict(experiment='R99', arm=args.arm,
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
            for name, rows in (('method_selection_train_val_test.csv', tables['methods']),
                    ('zero_selection_summary.csv', tables['summaries']),
                    ('train_metrics_by_problem.csv', helpers.train_metric_rows(config, lock, tables)),
                    ('val_empirical_screen.csv', tables['val_screen']),
                    ('inference_timing.csv', timing['measurements']),
                    ('preprocessing.csv', [dict(arm=args.arm, splits='TRAIN/VAL',
                        seconds=timing['preprocessing_seconds'], feature_root=config['feature_cache_root'],
                        runtime_placement_only=True, end_to_end=False), dict(arm=args.arm, splits='TEST',
                        seconds=timing['test_preprocessing_seconds'], feature_root=config['feature_cache_root'],
                        runtime_placement_only=True, end_to_end=False)])):
                common.write_csv(output / name, rows)
            for ref in tables['historical']:
                for stem, data in (('paired_changes', tables['pairs']), ('target_recovery', tables['recovery']),
                        ('target_recovery_summary', tables['recovery_summary'])):
                    common.write_csv(output / (stem + '_vs_' + ref + '.csv'), data[ref])
            for stem, data in (('target_recovery', tables['recovery']), ('target_recovery_summary', tables['recovery_summary'])):
                common.write_csv(output / (stem + '.csv'), data['R84A'])
            result.update(target_recovery_pdf=diagnostics_pdf(output, config, lock, tables, timing),
                historical_reference=tables['historical'], nss_reference=reference,
                inference_timing=timing, joint_val_pass=all(row['joint_val_pass'] for row in tables['val_screen']))
            provenance = read_json(output / 'protocol.json')
            provenance.update(r99_schema_version=1, model_spec=config['model_spec'],
                relation_variant=config['relation_variant'], relation_transform=config['relation_transform'],
                initialization=config['initialization'], target_identity=tables['target_identity'],
                historical_reference=tables['historical'], nss_reference=reference,
                objective=config['objective'], loss=config['loss'], evidence_boundary=REFERENCE_BOUNDARY,
                inference_timing=timing, rdrop=True, external_pretraining=False,
                val_empirical_screen=tables['val_screen'], projection_seed=990003)
            write_json(output / 'protocol.json', provenance)
            notes = output / 'RESULTS.md'
            notes.write_text(notes.read_text(encoding='utf-8') + '\n' + REFERENCE_BOUNDARY + '\n\n'
                + 'Timing: ' + timing['status'] + '; see target_recovery_and_train.pdf.\n', encoding='utf-8')
            for name in REQUIRED_PDFS:
                if not (output / name).is_file() or not (output / name).stat().st_size:
                    raise ValueError('R99 report PDF missing: ' + name)
            write_json(run / 'postprocess_status.json', dict(experiment='R99', arm=args.arm,
                phase='complete', exit_status=0, completed_utc=utc(), **result))
            return result
        except BaseException as error:
            write_json(run / 'postprocess_status.json', dict(experiment='R99', arm=args.arm,
                phase='postprocess_failed', exit_status=1, updated_utc=utc(), error=str(error),
                traceback=traceback.format_exc(), automatic_retry=False))
            raise


def compare(args):
    require_cpu(args)
    inputs = {arm: locked_inputs(args.root, arm) for arm in ARMS}
    configs = {arm: inputs[arm][1] for arm in ARMS}
    matched = validate_matched_configs(configs['A'], configs['B'])
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
                keys = ('top1', 'mean_cost', 'sbs_cost', 'oracle_cost', 'vs_sbs_pct', 'vs_oracle_pct', 'actual_regret_pct')
                metrics.append(dict(model='R99/' + arm, split=split, problem=problem,
                    relation_variant=VARIANTS[arm], selected_update=lock['selected_update'],
                    checkpoint_sha256=lock['checkpoint_sha256'], **{key: saved[prefix + key] for key in keys}))
    targets, identity = load_targets(configs['B'], refs['R84A']['config'], refs['R84A']['lock'])
    pairs, recovery, totals = helpers.comparison_tables(selected['A'], selected['B'], targets, identity, 'R99/B', 'R99/A')
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
            fig = common._page(helper, 'R99 | Global balanced B versus row-softmax A',
                'Same fresh selector + input-only32 early fusion; only column competition differs.', 'R99 B vs A', locks['B'], 1)
            helper.add_table(fig, ['Split', 'Top1 delta pp', 'Corrected', 'Harmed', 'Net cost delta', 'Regret delta pp'],
                [[r['split'], f"{r['top1_delta_pp']:.6f}", str(r['corrected']), str(r['harmed']),
                  f"{r['mean_cost_delta']:.9f}", f"{r['actual_regret_delta_pp']:.6f}"]
                 for r in pairs if r['problem'] == 'ALL'], [.05, .62, .9, .2], font=10)
            helper.add_table(fig, ['Split', 'Target correct', 'Rescued', 'Lost', 'New false', 'Recall (%)'],
                [[r['split'], str(r['correct_count']), str(r['rescued']), str(r['lost']),
                  str(r['new_false_pick_count']), '-' if r['recall_pct'] is None else f"{r['recall_pct']:.4f}"]
                 for r in totals], [.05, .35, .9, .2], font=10)
            for y, text in ((.29, 'A row-softmax / B fixed64 Sinkhorn. Same raw8 attributes, input-only32 and projection seed990003.'),
                    (.24, 'Original CE .35 / comparison .35 / risk .02 / R-Drop .35, dropout .1. No new target/loss.'),
                    (.19, 'Original four instance layers, R88B physical TSP anchor, hand34/methodID, joint2, score MLP/maximin.'),
                    (.14, 'Representational accessibility, not an assignment solver, VRP bound or B-semantics repair.'),
                    (.09, 'R89B main / R84A secondary historical; no newly matched historical controls. Fixed target36.'),
                    (.04, 'ALL deltas equal-weight18 tasks; native FP64 winners, TRAIN SBS; no TEST/LIB model selection.')):
                fig.text(.05, y, text, fontsize=8.5)
            pdf.savefig(fig)
            plt.close(fig)
            fig = common._page(helper, 'R99 | Historical VAL screens and hardware',
                'Descriptive locked-model diagnostics, not checkpoint/model reselection.', 'R99 B vs A', locks['B'], 2)
            helper.add_table(fig, ['Arm / reference', 'Top1 delta pp', 'Cost delta', 'Regret delta pp', 'Pass'],
                [[r['model'] + ' / ' + r['baseline'], f"{r['top1_delta_pp']:.6f}",
                  f"{r['mean_cost_delta']:.9f}", f"{r['actual_regret_delta_pp']:.6f}", str(r['pass_reference'])]
                 for r in screens], [.05, .49, .9, .3], font=10)
            import textwrap
            for index, arm in enumerate(ARMS):
                text = arm + ': ' + json.dumps(hardware['arms'][arm], sort_keys=True)
                fig.text(.05, .39 - index * .12, '\n'.join(textwrap.wrap(text, 110)), fontsize=9)
            fig.text(.05, .10, 'Disclosed hardware is not bitwise hardware matching. Development-exposed TEST.', fontsize=10)
            pdf.savefig(fig)
            plt.close(fig)
            _timing_panel(pdf, helper, 'R99 B vs A', locks['B'], timings, 3, locks=locks)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    result = dict(experiment='R99', r99_schema_version=1, complete=True, exit_status=0, checkpoints=locks,
        comparison_pdf=str(path), evidence_boundary=REFERENCE_BOUNDARY,
        matched_difference='B fixed64 Sinkhorn row/column competition versus A row-softmax input messages',
        target_identity=identity, hardware=hardware, inference_timing=timings, matched_controls=matched,
        external_pretraining=False, training=False, inference=False, checkpoint_reselection=False)
    lines = ['# R99: Input-Only Global Competition', '', REFERENCE_BOUNDARY, '',
        '| Model | Split | Top1 (%) | mean_cost | SBS cost | Oracle cost | vs_SBS (%) | vs_Oracle (%) | actual regret (%) |',
        '| --- | --- | --- | --- | --- | --- | --- | --- | --- |']
    for row in metrics:
        if row['problem'] == 'ALL':
            lines.append(f"| {row['model']} | {row['split']} | {100*row['top1']:.6f} | {row['mean_cost']:.9f} | "
                f"{row['sbs_cost']:.9f} | {row['oracle_cost']:.9f} | {row['vs_sbs_pct']:.6f} | "
                f"{row['vs_oracle_pct']:.6f} | {row['actual_regret_pct']:.6f} |")
    (Path(args.root) / 'RESULTS.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    shutil.copyfile(path, Path(args.root) / 'RESULTS.pdf')
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
    result.add_argument('--r89-root', type=Path, default=DEFAULT_R89B)
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    for name in ('root', 'nss_root', 'historical_root', 'r89_root'):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    if args.command == 'compare':
        return compare(args)
    if args.arm is None:
        raise ValueError('--arm is required for lib, report and status')
    if args.command == 'status':
        path = args.root / args.arm / 'postprocess_status.json'
        value = read_json(path) if path.is_file() else dict(experiment='R99', phase='not_started')
        print(json.dumps(value, indent=2))
        return value
    return lib_stage(args) if args.command == 'lib' else report(args)


if __name__ == '__main__':
    main()

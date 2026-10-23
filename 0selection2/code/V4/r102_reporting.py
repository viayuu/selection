"""R102 locked-evidence CPU reporting and explicitly separate future LIB stage.

No import, report or compare constructs a donor/model. Historical report
helpers are loaded privately; R102 supplies its own schema and matched gates.
"""

import argparse
import copy
import fcntl
import importlib
import importlib.util
import json
import math
import os
import shutil
import sys
import traceback
from pathlib import Path

from .r76_experiment import file_hash, read_json, utc, write_json


V4 = Path(__file__).resolve().parent
DEFAULT_ROOT = V4 / 'runs/R102_full_type_pretrained_fusion'
ARMS = ('A', 'B')
BUDGETS = dict.fromkeys(ARMS, 57600)
ARM_VARIANTS = VARIANTS = dict(A='extra_random', B='full_pretrained')
FACTORY = 'code.V4.r102_experiment'
MODEL_FACTORY = FACTORY + '.load_checkpoint'
INITIALIZATION = 'fresh_r84_plus_full_type_fusion'
REQUIRED_PDFS = ('MODEL_STATISTICS.pdf', 'selection_distributions.pdf',
    'target_recovery_and_train.pdf', 'learning_curves.pdf', 'training_and_method_diagnostics.pdf')
LIB_COUNTS = dict(TSPLIB=49, CVRPLIB=100)
REFERENCE_BOUNDARY = (
    'R102 extends R94 donor coverage, not a new architecture family. Matched A/B differ ONLY '
    'in frozen TSP ELG / ATSP MatNet initialization: A extra_random, B full_pretrained. '
    'BOTH use the identical PRETRAINED R94B routing donor and existing routing cache for '
    'CVRP+15MVRP. All18 tasks; TSP128, ATSP FROM256+TO256=512, routing128; separate '
    'bias-free projections and affine-free norms;1941257 trainables=1842953 original+98304. '
    'Fresh R84 seed2 initialization/geometry, physical TSP node0 raw role, original four '
    'instance layers/two joint layers/hand34+ID/score MLP/maximin. Fixed ATSP identity '
    'onehot firstN roles are NOT historical RNG certification. Original R77 CE=.35, '
    'comparison=.35, risk=.02, R-Drop=.35, dropout=.1;57600 successful logical1024 updates, '
    'warm4320, VAL1440, TRAIN7200, AdamW1e-4 cosine1e-6, WD1e-4/clip1. Frozen donors '
    'are embedded in checkpoint; no external weight reads on restore. No decoder/routes, '
    'new labels or auxiliary loss. Native/FP64 ind winners, all legal pools, frozen R84 '
    'TRAIN SBS; ALL is equal-problem macro. Fixed R84 target36: genuine rescue/loss/new '
    'false picks and original-cost contributions, not selection-frequency improvement. '
    'R94B is a historical reference, not matched; unknown external-pretraining data/overlap '
    'and unmatched source-pretraining compute remain disclosed. Development-exposed TEST; '
    'one unique minimum full18 VAL macro_vs_sbs_pct checkpoint, first ties; no TEST/LIB '
    'tuning or per-problem checkpoint assembly. Native TSPLIB49/CVRPLIB100 and same-bank '
    'NSS4. Timing is GPU-resident batch128 amortized, NOT singleton or I/O-inclusive '
    'end-to-end: BOTH cached nco fields removed on every online call, donor hook proves '
    'execution for all18; cached-selector timing is a separate scope, never actual total.')
BOUNDARY = REFERENCE_BOUNDARY


def _private_helpers():
    modules = {}
    for stem in ('r80', 'r84', 'r85', 'r87'):
        name = 'code.V4._r102_' + stem + '_reporting'
        spec = importlib.util.spec_from_file_location(name, V4 / (stem + '_reporting.py'))
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        modules[stem] = module
    common, shared, previous, helpers = (modules[k] for k in ('r80', 'r84', 'r85', 'r87'))
    shared.common = common
    previous.common, previous.shared = common, shared
    helpers.common, helpers.shared, helpers.previous = common, shared, previous
    return helpers, shared, previous, common


helpers, shared, previous, common = _private_helpers()
PROBLEMS = tuple(helpers.PROBLEMS)
DEFAULT_NSS, DEFAULT_R84 = helpers.DEFAULT_NSS, helpers.DEFAULT_R84
DEFAULT_R94B = DEFAULT_ROOT.parent / 'R94_pretrained_node_fusion/B'


def validate_report_config(config):
    importlib.import_module(FACTORY).validate_checkpoint_config(config)


def locked_inputs(root, arm):
    if arm not in ARMS:
        raise ValueError('R102 requires arm A or B')
    result = shared.locked_inputs(Path(root), arm, experiment='R102', arms=BUDGETS)
    validate_report_config(result[1])
    if result[2].get('selection_split') != 'val':
        raise ValueError('R102 report requires the explicit unique VAL-only lock')
    return result


def require_cpu(args):
    if str(args.device) != 'cpu':
        raise ValueError('R102 report/compare requires --device cpu over saved evidence')


def _extra_donor_controls(component):
    result = copy.deepcopy(component)
    for field in ('variant', 'checkpoint_sha256', 'checkpoint_path', 'random_seed',
                  'state_sha256', 'encoder_state_sha256', 'weights_sha256', 'checkpoint_loaded'):
        result.pop(field, None)
    unknown = result.get('known_unknown_pretraining_source')
    if isinstance(unknown, dict):
        unknown.pop('applies_to_loaded_weights', None)
    return result


def _donor_controls(provenance):
    arm = provenance.get('arm')
    flags = provenance.get('external_pretraining_by_component')
    expected = dict(TSP=arm == 'B', ATSP=arm == 'B', VRP=True)
    if (arm not in ARMS or not isinstance(flags, dict) or flags != expected
            or any(type(value) is not bool for value in flags.values())):
        raise ValueError('R102 external_pretraining_by_component must exactly match the arm; VRP must be true')
    result = copy.deepcopy(provenance)
    result.pop('arm', None)
    result.pop('variant', None)
    for component in ('TSP', 'ATSP'):
        result['external_pretraining_by_component'].pop(component)
        result['components'][component] = _extra_donor_controls(result['components'][component])
    # VRP is deliberately NOT normalized: even provenance differences fail closed.
    return result


def same_data(left, right, *, matched=False):
    fields = ('data_root', 'rows_per_problem', 'baselines', 'candidate_pools', 'candidate_pool_ids', 'winner_rule')
    if any(k not in left or k not in right or left[k] != right[k] for k in fields):
        raise ValueError('R102 comparison needs identical data/native winners/legal pools/TRAIN SBS')
    if not matched:
        return
    if (left.get('arm'), right.get('arm')) != ARMS:
        raise ValueError('R102 matched comparison is whole B full_pretrained versus A extra_random')
    normalized = []
    for config in (left, right):
        validate_report_config(config)
        for field in ('common_trainable_sha256', 'node_fusion_initialization_sha256',
                      'new_parameter_sha256', 'preparation_sha256', 'task_order_sha256',
                      'routing_feature_cache_root', 'checkpoint_rule', 'inference_rule'):
            if not config.get(field):
                raise ValueError('R102 matched comparison requires identity: ' + field)
        current = copy.deepcopy(config)
        for key in ('arm', 'donor_variant', 'feature_cache_root', 'gpu_type', 'hardware',
                    'device', 'device_name', 'gpu_uuid', 'run_root', 'output_root', 'created_utc',
                    'extra_type_pretraining'):
            current.pop(key, None)
        current['donor_provenance'] = _donor_controls(current['donor_provenance'])
        frozen = current['model_params'].get('frozen_donor', {})
        if (frozen.get('variant') != ARM_VARIANTS[config['arm']]
                or frozen.get('provenance') != config['donor_provenance']):
            raise ValueError('R102 model donor metadata differs from the locked protocol')
        frozen.pop('variant')
        frozen['provenance'] = _donor_controls(frozen['provenance'])
        normalized.append(current)
    if normalized[0] != normalized[1]:
        raise ValueError('Matched R102 controls differ beyond TSP/ATSP weights and disclosed runtime placement')


def hardware_disclosure(configs):
    devices = {arm: {k: configs[arm][k] for k in ('hardware', 'gpu_type', 'device_name', 'gpu_uuid')
                     if k in configs[arm]} for arm in ARMS}
    return dict(arms=devices, same_reported_hardware=(devices['A'] == devices['B']) if all(devices.values()) else None,
        bitwise_hardware_matched=False, note='Identical controls do not imply bitwise hardware matching.')


def validate_matched_configs(left, right):
    same_data(left, right, matched=True)
    return dict(experiment='R102', matched=True, arms=list(ARMS), budgets=BUDGETS.copy(),
        initialization=INITIALIZATION, donor_variants=ARM_VARIANTS.copy(),
        common_trainable_sha256=left['common_trainable_sha256'],
        node_fusion_initialization_sha256=left['node_fusion_initialization_sha256'],
        new_parameter_sha256=left['new_parameter_sha256'],
        task_order_sha256=left['task_order_sha256'], preparation_sha256=left['preparation_sha256'],
        routing_donor_identical=True, routing_provenance=copy.deepcopy(left['donor_provenance']['components']['VRP']),
        routing_feature_cache_root=left['routing_feature_cache_root'],
        hardware=hardware_disclosure(dict(A=left, B=right)),
        external_pretraining_budget_matched=False, historical_R94B_matched=False,
        training=False, inference=False, checkpoint_reselection=False)


def selected_predictions(run, config, lock, val, test):
    if val['identity']['model_state_sha256'] != test['identity']['model_state_sha256']:
        raise ValueError('R102 diagnostics need the same unique VAL/TEST model state')
    packed, methods, summaries, results, paths = helpers.selected_predictions(run, config, lock, val, test)
    model = config['experiment'] + '/' + config['arm']
    return packed, [dict(r, model=model) for r in methods], [dict(r, model=model) for r in summaries], results, paths


def references(config, args):
    result = {}
    for name, path, experiment, budgets in (
            ('R84A', args.historical_root, 'R84', shared.BUDGETS),
            ('R94B', args.r94_root, 'R94', dict(A=57600, B=57600))):
        path = Path(path)
        run, old, lock, test, val = shared.locked_inputs(path.parent, path.name, experiment=experiment, arms=budgets)
        same_data(config, old)
        result[name] = dict(run=run, config=old, lock=lock, test=test, val=val,
            provenance=dict(model=name, run=str(run), checkpoint_sha256=lock['checkpoint_sha256'],
                same_data=True, retrained=False, newly_matched_control=False,
                role='fixed target36 historical anchor' if name == 'R84A' else 'historical coverage reference; NOT matched'))
    return result


def load_targets(config, historical_config, historical_lock):
    targets, identity = helpers.load_targets(config, historical_config, historical_lock)
    if len(targets) != 36:
        raise ValueError('R102 retains exactly the fixed R84 target36, never reselects targets')
    return targets, dict(identity, sha256=file_hash(Path(identity['path'])), target_count=36)


def comparison_tables(old, new, targets, identity, model, baseline):
    """Genuine per-instance recovery plus FP64 new-false-pick cost impact."""
    import numpy as np
    pairs, recovery, totals = helpers.comparison_tables(old, new, targets, identity, model, baseline)
    for row in recovery:
        left, right = old[row['split']][row['problem']], new[row['split']][row['problem']]
        slot = right.pool.index(row['solver'])
        false = (right.winner != slot) & (left.pred != slot) & (right.pred == slot)
        indices = np.arange(right.n)
        new_cost, old_cost = right.costs[indices, right.pred], left.costs[indices, left.pred]
        row.update(new_false_pick_mean_cost=float(new_cost[false].mean()) if false.any() else None,
            new_false_pick_cost_contribution=float(new_cost[false].sum() / right.n),
            new_false_pick_cost_delta_contribution=float((new_cost - old_cost)[false].sum() / right.n))
    for row in totals:
        for key in ('new_false_pick_cost_contribution', 'new_false_pick_cost_delta_contribution'):
            row[key] = sum(r[key] for r in recovery if r['split'] == row['split']) / len(PROBLEMS)
    return pairs, recovery, totals


def diagnostic_tables(run, config, lock, val, test, args):
    packed, methods, summaries, results, paths = selected_predictions(run, config, lock, val, test)
    refs = references(config, args)
    targets, identity = load_targets(config, refs['R84A']['config'], refs['R84A']['lock'])
    pairs, recovery, totals = {}, {}, {}
    for name, ref in refs.items():
        old = selected_predictions(ref['run'], ref['config'], ref['lock'], ref['val'], ref['test'])[0]
        pairs[name], recovery[name], totals[name] = comparison_tables(
            old, packed, targets, identity, 'R102/' + config['arm'], name)
    return dict(predictions=packed, methods=methods, summaries=summaries, results=results, paths=paths,
        pairs=pairs, recovery=recovery, recovery_summary=totals, target_identity=identity,
        historical={k: v['provenance'] for k, v in refs.items()})


def _positive(value, *, zero=False):
    return (not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)
            and (value >= 0 if zero else value > 0))


def inference_timing(run, lock):
    path = Path(run) / 'inference_timing.json'
    record = read_json(path)
    expected = dict(experiment='R102', arm=lock.get('arm'), complete=True,
        checkpoint_sha256=lock['checkpoint_sha256'], state_hash=lock['state_hash'],
        inference_rule=lock['inference_rule'], repetitions=3, warmup_batches=1,
        raw_input_source='original_base_cache_without_both_nco_fields',
        includes_native_encoder=True, online_donor_verified=True, all18_donor_coverage=True,
        includes_solver_decoder=False, full_system_measured=False, end_to_end_measured=False,
        singleton_measured=False, cache_hit_is_total=False, model_reselection=False)
    if (record.get('arm') not in ARMS or record.get('variant') != ARM_VARIANTS[record.get('arm')]
            or any(record.get(k) != v for k, v in expected.items())
            or not str(record.get('device', '')).startswith('cuda:')
            or not record.get('scope') or not record.get('excludes')):
        raise ValueError('R102 requires actual same-lock all18 online donor+selector timing, not cache-only latency')
    raw = record.get('per_problem')
    if (not isinstance(raw, list) or len(raw) != 18 or any(not isinstance(r, dict) for r in raw)
            or {r.get('problem') for r in raw} != set(PROBLEMS)):
        raise ValueError('R102 timing must contain all18 problems exactly once')
    rows = []
    for item in raw:
        if (item.get('instances') != 128 or item.get('online_donor_verified') is not True
                or item.get('online_fields_removed') != ['nco_nodes', 'nco_node_mask']):
            raise ValueError('R102 online timing must strip BOTH fields and prove real donor invocation')
        row = dict(problem=item['problem'], instances=128, device=record['device'], repetitions=3)
        for scope, calls in (('donor_plus_selector', 4), ('cache_hit_selector', 0)):
            measurement = item.get(scope, {})
            samples, mean = measurement.get('batch_seconds'), measurement.get('mean_seconds_per_instance')
            if (measurement.get('donor_forward_calls') != calls or measurement.get('warmup_batches') != 1
                    or not isinstance(samples, list) or len(samples) != 3
                    or not all(_positive(v) for v in samples) or not _positive(mean)
                    or not math.isclose(mean, sum(samples) / (3 * 128), rel_tol=1e-6, abs_tol=1e-12)):
                raise ValueError('R102 timing needs three synchronized batches and exact donor hook counts: ' + scope)
            row[scope + '_ms_per_instance'] = mean * 1000
            row[scope + '_donor_forward_calls'] = calls
        rows.append(row)
    return dict(path=str(path.resolve()), sha256=file_hash(path), status='measured_online_all18_donor_plus_selector',
        measurements=rows, scope=record['scope'], excludes=record['excludes'], online_donor_verified=True,
        donor_plus_selector_measured=True, full_system_measured=False, end_to_end_measured=False,
        singleton_measured=False, cache_hit_is_total=False,
        preprocessing=record.get('preprocessing'), test_preprocessing=record.get('test_preprocessing'))


def validate_lib_population(predictions, config):
    for dataset, problem in (('TSPLIB', 'TSP'), ('CVRPLIB', 'CVRP')):
        data = predictions[dataset, problem]
        if (data.n != LIB_COUNTS[dataset] or data.pool != config['candidate_pools'][problem]
                or data.pool_ids.tolist() != config['candidate_pool_ids'][problem]):
            raise ValueError('R102 requires native TSPLIB49/CVRPLIB100 on the same candidate bank')


def learning_curves(run, config, lock):
    import matplotlib.pyplot as plt
    history = read_json(run / 'history.json')
    steps = [r.get('successful_updates') for r in history]
    if (steps != list(range(1440, 57601, 1440))
            or any(not r.get('val') for r in history)
            or any(not r.get('train') for r in history if r['successful_updates'] % 7200 == 0)
            or not any(r.get('train') and r['successful_updates'] == lock['selected_update'] for r in history)):
        raise ValueError('R102 learning curves require the complete saved40 VAL points, TRAIN7200 and selected TRAIN')
    output, rows = run / 'full_model_statistics', []
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle('R102/' + config['arm'] + ' | original R77; one unique VAL checkpoint')
    for record in history:
        row = dict(successful_updates=record['successful_updates'])
        for prefix, values in [('optimization', record.get('optimization', {})),
                *[(s, record[s]['macro']) for s in ('train', 'val') if record.get(s)]]:
            row.update({prefix + '_' + k: v for k, v in values.items() if isinstance(v, (int, float))})
        rows.append(row)
    for key in ('loss', 'ce', 'cmp', 'risk', 'kl'):
        points = [(r['successful_updates'], r['optimization'][key]) for r in history if key in r.get('optimization', {})]
        if not points or not all(math.isfinite(y) for _, y in points):
            raise ValueError('R102 cannot fabricate missing/nonfinite original loss curves')
        axes[0, 0].plot(*zip(*points), label=key)
    axes[0, 0].set_title('TRAIN interval-mean original loss, R-Drop retained')
    axes[0, 0].legend(fontsize=8)
    for axis, key, factor, title in (
            (axes[0, 1], 'macro_ce', 1., 'Evaluation CE'),
            (axes[0, 2], 'macro_top1', 100., 'Native Top1 (%)'),
            (axes[1, 0], 'macro_mean_cost', 1., 'Task-equal cost (descriptive)'),
            (axes[1, 1], 'macro_actual_regret_pct', 1., 'Actual regret (%)'),
            (axes[1, 2], 'macro_vs_sbs_pct', 1., 'vs fixed TRAIN SBS (%)')):
        for split in ('train', 'val'):
            points = [(r['successful_updates'], r[split]['macro'][key] * factor)
                      for r in history if r.get(split) and key in r[split]['macro']]
            if not points or not all(math.isfinite(y) for _, y in points):
                raise ValueError('Missing/nonfinite saved R102 TRAIN/VAL metric curve: ' + key)
            axis.plot(*zip(*points), label=split.upper())
        axis.set_title(title)
        axis.legend()
    for axis in axes.flat:
        axis.axvline(lock['selected_update'], color='#c45358', linestyle='--', linewidth=1)
        axis.set_xlabel('Successful logical-batch1024 updates')
        axis.grid(alpha=.2)
    fig.tight_layout(rect=(0, 0, 1, .95))
    try:
        fig.savefig(output / 'learning_curves.pdf')
    finally:
        plt.close(fig)
    common.write_csv(output / 'learning_curves.csv', rows)


def _timing_panel(pdf, helper, model, lock, timing, page):
    import matplotlib.pyplot as plt
    fig = common._page(helper, model + ' | Online Timing (all18)',
        'GPU-resident batch128 amortized; one warmup + three synchronized repeats; donor hooks verified.', model, lock, page)
    helper.add_table(fig, ['Problem', 'n', 'Donor + selector ms / instance', 'Cache-hit selector ms / instance'],
        [[r['problem'], '128', f"{r['donor_plus_selector_ms_per_instance']:.6f}",
          f"{r['cache_hit_selector_ms_per_instance']:.6f}"] for r in timing['measurements']],
        [.05, .26, .9, .55], font=9)
    for y, text in ((.20, 'ALL18 include a real donor: TSP ELG128, ATSP row256+col256, routing frozen ReLD128.'),
            (.15, 'Both nco_nodes and nco_node_mask removed each online call; hook counts4 online,0 cache.'),
            (.10, 'Excludes dataset I/O, host transfer and one-time preparation; not singleton or end-to-end.'),
            (.05, 'Cache-hit selector is NOT actual total inference. Device: ' + timing['measurements'][0]['device'])):
        fig.text(.05, y, text, fontsize=9)
    pdf.savefig(fig)
    plt.close(fig)


def _recovery_page(pdf, helper, model, lock, baseline, pairs, totals, page):
    import matplotlib.pyplot as plt
    fig = common._page(helper, model + ' | Fixed target36 versus ' + baseline,
        'Native FP64 true-winner recovery; ALL deltas and cost contributions are task-equal.', model, lock, page)
    helper.add_table(fig, ['Split', 'Correct', 'Rescued', 'Lost', 'New false', 'Recall (%)'],
        [[r['split'], str(r['correct_count']), str(r['rescued']), str(r['lost']), str(r['new_false_pick_count']),
          '-' if r['recall_pct'] is None else f"{r['recall_pct']:.4f}"] for r in totals], [.05, .65, .9, .16], font=10)
    helper.add_table(fig, ['Split', 'Top1 delta pp', 'Corrected', 'Harmed', 'Cost delta', 'Regret delta pp'],
        [[r['split'], f"{r['top1_delta_pp']:.6f}", str(r['corrected']), str(r['harmed']),
          f"{r['mean_cost_delta']:.9f}", f"{r['actual_regret_delta_pp']:.6f}"]
         for r in pairs if r['problem'] == 'ALL'], [.05, .41, .9, .16], font=10)
    helper.add_table(fig, ['Split', 'New false-pick cost contribution', 'New false-pick cost DELTA contribution'],
        [[r['split'], f"{r['new_false_pick_cost_contribution']:.9f}",
          f"{r['new_false_pick_cost_delta_contribution']:.9f}"] for r in totals], [.05, .18, .9, .16], font=10)
    fig.text(.05, .11, 'False-pick costs use original FP64 selected costs / all instances, then equal-problem mean.', fontsize=9)
    fig.text(.05, .06, 'R84A and R94B are historical references, not matched controls. No TEST/LIB selection.', fontsize=9)
    pdf.savefig(fig)
    plt.close(fig)


def diagnostics_pdf(output, config, lock, tables, timing):
    from matplotlib.backends.backend_pdf import PdfPages
    helper = common.reporting_helpers()
    helper.setup_style()
    model, path = 'R102/' + config['arm'], output / 'target_recovery_and_train.pdf'
    temporary = path.with_name('.' + path.stem + '.' + str(os.getpid()) + '.pdf')
    try:
        with PdfPages(temporary) as pdf:
            for page, ref in enumerate(('R84A', 'R94B'), 1):
                _recovery_page(pdf, helper, model, lock, ref, tables['pairs'][ref], tables['recovery_summary'][ref], page)
            _timing_panel(pdf, helper, model, lock, timing, 3)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return str(path)


def _completed_lib(run, lock):
    value = read_json(run / 'lib_stage_complete.json')
    if (value.get('experiment') != 'R102' or value.get('arm') != lock.get('arm')
            or value.get('complete') is not True or value.get('exit_status') != 0
            or value.get('checkpoint_sha256') != lock['checkpoint_sha256']
            or value.get('state_hash') != lock['state_hash'] or value.get('native_counts') != LIB_COUNTS
            or set(value.get('datasets', [])) != set(LIB_COUNTS)):
        raise ValueError('R102 requires same-lock native TSPLIB49/CVRPLIB100 completion with rc0')


def completed_arm_report(run, lock):
    _completed_lib(run, lock)
    post = read_json(run / 'postprocess_status.json')
    protocol = read_json(run / 'full_model_statistics/protocol.json')
    if (post.get('complete') is not True or post.get('phase') != 'complete' or post.get('exit_status') != 0
            or protocol.get('complete') is not True or protocol.get('experiment') != 'R102'
            or protocol.get('checkpoint_lock', {}).get('checkpoint_sha256') != lock['checkpoint_sha256']
            or protocol.get('target_identity', {}).get('target_count') != 36
            or protocol.get('inference_timing', {}).get('sha256') != inference_timing(run, lock)['sha256']):
        raise ValueError('R102 compare requires completed same-lock TRAIN/VAL/TEST/LIB/PDF/timing/target36 reports')
    for name in REQUIRED_PDFS:
        path = run / 'full_model_statistics' / name
        if not path.is_file() or not path.stat().st_size:
            raise ValueError('Missing required R102 PDF: ' + name)


def lib_stage(args):
    if not str(args.device).startswith('cuda:'):
        raise ValueError('Future R102 LIB requires an explicit launcher-assigned --device cuda:N')
    run, config, lock, test, _ = locked_inputs(args.root, args.arm)
    with (run / 'lib_stage.lock').open('a') as claim:
        fcntl.flock(claim, fcntl.LOCK_EX | fcntl.LOCK_NB)

        def phase(name, **fields):
            write_json(run / 'lib_stage_status.json', dict(experiment='R102', arm=args.arm,
                phase=name, updated_utc=utc(), **fields))

        try:
            reference = previous.validate_nss_reference(config, args.nss_root)
            results = common.r80_lib(args.root, run, config, lock, args.device, args.lib_batch_size,
                phase, test['identity']['model_state_sha256'], factory_module=FACTORY)
            if (set(results) != set(LIB_COUNTS) or any(results[name]['train_sbs_reference']['n'] != n
                    for name, n in LIB_COUNTS.items())):
                raise ValueError('R102 LIB must retain native49/100 instances')
            write_json(run / 'lib_stage_complete.json', dict(experiment='R102', arm=args.arm,
                complete=True, exit_status=0, updated_utc=utc(), state_hash=lock['state_hash'],
                checkpoint_sha256=lock['checkpoint_sha256'], model_state_sha256=test['identity']['model_state_sha256'],
                native_counts=LIB_COUNTS, datasets=list(results), nss_reference=reference))
            phase('complete', exit_status=0, checkpoint_sha256=lock['checkpoint_sha256'])
            return results
        except BaseException as error:
            phase('lib_failed', exit_status=1, error=str(error), automatic_retry=False)
            raise


def _verify_pdfs(output):
    for name in REQUIRED_PDFS:
        path = output / name
        if not path.is_file() or not path.stat().st_size:
            raise ValueError('R102 report missing PDF: ' + name)


def render_environment():
    """Rendering is a host CPU operation with a real installed CJK font."""
    if any(os.environ.get(key) for key in ('APPTAINER_CONTAINER', 'SINGULARITY_CONTAINER')):
        raise ValueError('Render R102 PDFs on the CPU host, outside the experiment container')
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import font_manager
    from matplotlib.ft2font import FT2Font
    path = common.reporting_helpers().FONT
    if not path.is_file() or not FT2Font(str(path)).get_char_index(0x6a21):
        raise ValueError('R102 PDF rendering requires an installed CJK font on the CPU host')
    font_manager.fontManager.addfont(str(path))
    return dict(backend='Agg', font=font_manager.FontProperties(fname=str(path)).get_name(),
                font_path=str(path), device='cpu')


def report(args):
    require_cpu(args)
    run, config, lock, test, val = locked_inputs(args.root, args.arm)
    with (run / 'postprocess.lock').open('a') as claim:
        fcntl.flock(claim, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            write_json(run / 'postprocess_status.json', dict(experiment='R102', arm=args.arm,
                phase='loading_saved_predictions', updated_utc=utc(), complete=False))
            render = render_environment()
            _completed_lib(run, lock)
            timing = inference_timing(run, lock)
            reference = previous.validate_nss_reference(config, args.nss_root)
            predictions, metrics, methods = shared.load_saved_splits(run, config, lock, test, val)
            shared.saved_libs(run, config, lock, test, predictions, metrics, methods)
            validate_lib_population(predictions, config)
            comparisons, alignment = shared.nss_comparisons(config, lock, predictions, args.nss_root)
            tables = diagnostic_tables(run, config, lock, val, test, args)
            result = shared.build_reports(run, config, lock, test, val, predictions, metrics, methods, comparisons, alignment)
            output = run / 'full_model_statistics'
            learning_curves(run, config, lock)
            train_metrics = helpers.train_metric_rows(config, lock, tables)
            for name, rows in (('metrics_by_problem.csv', [*train_metrics, *metrics]),
                    ('method_selection_train_val_test.csv', tables['methods']),
                    ('zero_selection_summary.csv', tables['summaries']),
                    ('train_metrics_by_problem.csv', train_metrics), ('inference_timing.csv', timing['measurements'])):
                common.write_csv(output / name, rows)
            for ref in tables['historical']:
                for stem, data in (('paired_changes', tables['pairs']), ('target_recovery', tables['recovery']),
                                   ('target_recovery_summary', tables['recovery_summary'])):
                    common.write_csv(output / (stem + '_vs_' + ref + '.csv'), data[ref])
            for stem, data in (('target_recovery', tables['recovery']), ('target_recovery_summary', tables['recovery_summary'])):
                common.write_csv(output / (stem + '.csv'), data['R84A'])
            result.update(target_recovery_pdf=diagnostics_pdf(output, config, lock, tables, timing),
                historical_reference=tables['historical'], nss_reference=reference, inference_timing=timing)
            provenance = read_json(output / 'protocol.json')
            provenance.update(r102_schema_version=1, initialization=INITIALIZATION,
                model_spec=config['model_spec'], donor_provenance=config['donor_provenance'],
                target_identity=tables['target_identity'], historical_reference=tables['historical'],
                nss_reference=reference, inference_timing=timing, evidence_boundary=REFERENCE_BOUNDARY,
                objective=config['objective'], loss=config['loss'], rdrop=True,
                trainable_parameter_count=config['trainable_parameter_count'], routing_donor_identical_by_design=True,
                external_solver_pretraining=True, extra_type_pretraining=config['extra_type_pretraining'],
                external_pretraining_budget_matched=False, historical_R94B_matched=False,
                native_lib_counts=LIB_COUNTS, rendering=render)
            _verify_pdfs(output)
            write_json(output / 'protocol.json', provenance)
            notes = output / 'RESULTS.md'
            notes.write_text(notes.read_text(encoding='utf-8') + '\n' + REFERENCE_BOUNDARY + '\n', encoding='utf-8')
            write_json(run / 'postprocess_status.json', dict(experiment='R102', arm=args.arm,
                phase='complete', exit_status=0, completed_utc=utc(), **result))
            return result
        except BaseException as error:
            write_json(run / 'postprocess_status.json', dict(experiment='R102', arm=args.arm,
                phase='postprocess_failed', complete=False, exit_status=1, updated_utc=utc(), error=str(error),
                traceback=traceback.format_exc(), automatic_retry=False))
            raise


def compare(args):
    require_cpu(args)
    root = Path(args.root)
    with (root / 'comparison.lock').open('a') as claim:
        fcntl.flock(claim, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _compare(args)


def _compare(args):
    inputs = {arm: locked_inputs(args.root, arm) for arm in ARMS}
    configs = {arm: value[1] for arm, value in inputs.items()}
    matched = validate_matched_configs(configs['A'], configs['B'])
    render = render_environment()
    selected, locks, timings, metrics = {}, {}, {}, []
    for arm, (run, config, lock, test, val) in inputs.items():
        completed_arm_report(run, lock)
        selected[arm], _, _, results, paths = selected_predictions(run, config, lock, val, test)
        predictions, arm_metrics, methods = shared.load_saved_splits(run, config, lock, test, val)
        shared.saved_libs(run, config, lock, test, predictions, arm_metrics, methods)
        validate_lib_population(predictions, config)
        # Same locked model for all datasets, not separately selected best rows.
        arm_tables = dict(predictions=selected[arm], results=results, paths=paths)
        metrics.extend(dict(row, model='R102/' + arm) for row in
                       [*helpers.train_metric_rows(config, lock, arm_tables), *arm_metrics])
        locks[arm], timings[arm] = lock, inference_timing(run, lock)
    refs = references(configs['B'], args)
    targets, identity = load_targets(configs['B'], refs['R84A']['config'], refs['R84A']['lock'])
    pairs, recovery, totals = comparison_tables(selected['A'], selected['B'], targets, identity, 'R102/B', 'R102/A')
    output = Path(args.root) / 'comparison_artifacts'
    output.mkdir(parents=True, exist_ok=True)
    for name, rows in (('comparison.csv', metrics), ('paired_changes_B_vs_A.csv', pairs),
            ('target_recovery_B_vs_A.csv', recovery), ('target_recovery_summary_B_vs_A.csv', totals)):
        common.write_csv(output / name, rows)
    from matplotlib.backends.backend_pdf import PdfPages
    helper = common.reporting_helpers()
    helper.setup_style()
    path = output / 'COMPARISON.pdf'
    temporary = path.with_name('.comparison.' + str(os.getpid()) + '.pdf')
    try:
        with PdfPages(temporary) as pdf:
            _comparison_overview(pdf, helper, metrics, locks, matched)
            _recovery_page(pdf, helper, 'R102/B versus R102/A', locks['B'], 'matched R102/A', pairs, totals, 2)
            for page, arm in enumerate(ARMS, 3):
                _timing_panel(pdf, helper, 'R102/' + arm, locks[arm], timings[arm], page)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    result = dict(experiment='R102', r102_schema_version=1, complete=True, exit_status=0,
        checkpoints=locks, comparison_pdf=str(path), evidence_boundary=REFERENCE_BOUNDARY,
        target_identity=identity, matched_controls=matched, inference_timing=timings,
        hardware=hardware_disclosure(configs), historical_reference={k: v['provenance'] for k, v in refs.items()},
        rendering=render,
        matched_difference='ONLY frozen TSP/ATSP donor initialization; identical PRETRAINED R94B routing in BOTH arms',
        external_pretraining_budget_matched=False, historical_R94B_matched=False,
        training=False, inference=False, checkpoint_reselection=False)
    lines = ['# R102: Whole Matched A/B', '', REFERENCE_BOUNDARY, '',
        '| Model | Split | Top1 (%) | mean_cost | SBS cost | Oracle cost | vs_SBS (%) | vs_Oracle (%) | actual regret (%) |',
        '| --- | --- | --- | --- | --- | --- | --- | --- | --- |']
    for row in metrics:
        if row['problem'] == 'ALL':
            lines.append(f"| {row['model']} | {row['split']} | {row['top1_pct']:.6f} | {row['mean_cost']:.9f} | "
                f"{row['sbs_cost']:.9f} | {row['oracle_cost']:.9f} | {row['vs_sbs_pct']:.6f} | "
                f"{row['vs_oracle_pct']:.6f} | {row['actual_regret_pct']:.6f} |")
    (Path(args.root) / 'RESULTS.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    shutil.copyfile(path, Path(args.root) / 'RESULTS.pdf')
    write_json(output / 'protocol.json', result)
    return result


def _comparison_overview(pdf, helper, metrics, locks, matched):
    import matplotlib.pyplot as plt
    fig = common._page(helper, 'R102 | Whole Matched A/B',
        'Only TSP/ATSP donor initialization differs. BOTH routing donors are the identical pretrained R94B.',
        'R102/B versus R102/A', locks['B'], 1)
    overall = [row for row in metrics if row['problem'] == 'ALL']
    helper.add_table(fig, ['Arm / split', 'Top1 (%)', 'mean_cost', 'SBS', 'Oracle', 'vs_SBS (%)', 'Regret (%)'],
        [[r['model'] + ' / ' + r['split'], f"{r['top1_pct']:.4f}", f"{r['mean_cost']:.8f}",
          f"{r['sbs_cost']:.8f}", f"{r['oracle_cost']:.8f}", f"{r['vs_sbs_pct']:.4f}",
          f"{r['actual_regret_pct']:.4f}"] for r in overall], [.05, .48, .9, .33], font=9)
    libs = [r for r in metrics if r['split'] in LIB_COUNTS]
    helper.add_table(fig, ['Arm / native LIB', 'n', 'Top1 (%)', 'mean_cost', 'vs_SBS (%)', 'Regret (%)'],
        [[r['model'] + ' / ' + r['split'], str(r['n']), f"{r['top1_pct']:.4f}", f"{r['mean_cost']:.8f}",
          f"{r['vs_sbs_pct']:.4f}", f"{r['actual_regret_pct']:.4f}"] for r in libs], [.05, .23, .9, .19], font=9)
    for y, text in ((.18, 'Same original seed2/R84 state + geometry, three fresh projections; routing projection seed940003.'),
            (.14, 'A update ' + str(locks['A']['selected_update']) + '; B update ' + str(locks['B']['selected_update'])
                + '. Each is its own unique full18 minimum-VAL checkpoint.'),
            (.10, 'Native/FP64 ind; TRAIN SBS; ALL is task-equal. No TEST/LIB tuning or per-task assembly.'),
            (.06, 'R94B is historical, not matched. Source-pretraining compute/data overlap is not controlled.'),
            (.02, 'Routing provenance identical: ' + str(matched['routing_donor_identical']) + '; all18 metrics and NSS4 in arm PDFs.')):
        fig.text(.05, y, text, fontsize=8.5)
    pdf.savefig(fig)
    plt.close(fig)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('command', choices=('lib', 'report', 'compare', 'status'))
    result.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    result.add_argument('--arm', choices=ARMS)
    result.add_argument('--device', default='cpu')
    result.add_argument('--lib-batch-size', type=int, choices=(1, 2), default=2)
    result.add_argument('--nss-root', type=Path, default=DEFAULT_NSS)
    result.add_argument('--historical-root', type=Path, default=DEFAULT_R84 / 'A_baseline')
    result.add_argument('--r94-root', type=Path, default=DEFAULT_R94B)
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    for name in ('root', 'nss_root', 'historical_root', 'r94_root'):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    if args.command == 'compare':
        return compare(args)
    if args.arm is None:
        raise ValueError('--arm required for lib, report and status')
    if args.command == 'status':
        path = args.root / args.arm / 'postprocess_status.json'
        value = read_json(path) if path.is_file() else dict(experiment='R102', phase='not_started')
        print(json.dumps(value, indent=2))
        return value
    return lib_stage(args) if args.command == 'lib' else report(args)


if __name__ == '__main__':
    main()

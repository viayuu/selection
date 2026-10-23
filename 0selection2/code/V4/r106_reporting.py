"""R106 native LIB and CPU-only, same-lock full18/NSS4/fixed36 PDF reports."""

import argparse
import copy
import csv
import fcntl
import json
import math
import os
import textwrap
from pathlib import Path

import numpy as np
import torch

from . import r80_reporting, r84_reporting, r85_reporting, r87_reporting
from . import r106_experiment as runner
from .r76_experiment import file_hash, read_json, utc, write_json
from .r77_data import parameter_hash


ARMS, BUDGETS, DEFAULT_ROOT = runner.ARMS, runner.BUDGETS, runner.DEFAULT_ROOT
FACTORY = 'code.V4.r106_experiment'
PROBLEMS = tuple(runner.PROBLEMS)
DEFAULT_R84, DEFAULT_NSS = r87_reporting.DEFAULT_R84, r87_reporting.DEFAULT_NSS
DEFAULT_R94B = DEFAULT_ROOT.parent / 'R94_pretrained_node_fusion/B'
DEFAULT_R102B = (DEFAULT_ROOT.parent / 'R102_full_type_pretrained_fusion'
    / 'preparation/postprocess_recovery/report_view/B')
LIB_COUNTS = dict(TSPLIB=49, CVRPLIB=100)
REQUIRED_PDFS = ('MODEL_STATISTICS.pdf', 'selection_distributions.pdf', 'learning_curves.pdf',
                 'training_and_method_diagnostics.pdf', 'target_recovery_and_train.pdf')
BOUNDARY = ('R106 compares frozen A with trainable B native encoders under the same final '
    'original selector supervision. Both start from identical complete fresh R84 seed2 selector '
    'tensors, three projections and original R102B pretrained ELG/MatNet/ReLD weights. Native '
    'pretraining is not selector continuation; its data/overlap remains unknown. Total tensors '
    'are identical at initialization, but trainable freedom and training compute are NOT matched. '
    'Native stays eval in both arms; B receives gradients from update1 with effective LR0.1 '
    'times the common warmup/cosine LR. A does not update native parameters. One input-only '
    'native graph per microbatch is shared by the original two dropout selector forwards, '
    'with the unchanged CE/comparison/risk/R-Drop and real-row normalization. No auxiliary head '
    'or target. All TRAIN/VAL/TEST/LIB calls use online native encoders, never native-feature '
    'cache substitution. Native FP32 output precedes normal fusion/selector precision. '
    'Raw node4/joint2, physical TSP node0 role, methodID+hand34 and score MLP/maximin remain. '
    'Padding-adapted native execution is shared across arms; numerical fixture agreement is '
    'not bitwise equivalence or efficacy. Native/FP64 ind, legal-pool Oracle, full-TRAIN SBS; '
    'ALL is equal-task macro, mixed raw cost descriptive only. The fixed R84 target36 reports '
    'true rescues/losses, false picks and aligned net cost, not choice frequency as recovery. '
    'R94B and completed R102B recovery report view are historical, NOT newly matched controls; '
    'external pretraining and historical compute are not controlled. Development-exposed '
    'TEST; one whole18 minimum-VAL/first-tie checkpoint per arm, no TEST/LIB reselection or '
    'per-problem assembly. Actual all18 online native+selector timing is GPU-resident batch128 '
    'amortized, NOT singleton or I/O-inclusive end-to-end. Dataset I/O, CPU transfer, training '
    'and solver decoding/execution are excluded. Gradient/state witnesses are implementation '
    'checks, not selection-effectiveness evidence.')


common = runner.private_module(r80_reporting, 'report_common')
shared = runner.private_module(r84_reporting, 'report_shared')
previous = runner.private_module(r85_reporting, 'report_previous')
helpers = runner.private_module(r87_reporting, 'report_helpers')
shared.common = common
previous.shared, previous.common = shared, common
helpers.shared, helpers.previous, helpers.common = shared, previous, common
LIBS = shared.LIBS
for module in (common, shared, previous, helpers):
    module.DEFAULT_ROOT, module.ARMS, module.BUDGETS, module.FACTORY = DEFAULT_ROOT, ARMS, BUDGETS, FACTORY
_template_selected_train = previous.selected_train
_template_write_csv = common.write_csv


def write_csv(path, rows):
    if Path(path).name != 'nss_comparison.csv':
        return _template_write_csv(path, rows)
    # Administrative readers use plain UTF-8, so do not hide the first header
    # behind a BOM. Other stable report CSVs keep their inherited encoding.
    rows = list(rows)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


common.write_csv = write_csv


def selected_train(run, config, lock):
    result, path = _template_selected_train(run, config, lock)
    if config['experiment'] == 'R106':
        _, binding = runner.selected_capture_proof(run, result, lock, 'train')
        result = dict(result, checkpoint_binding=binding)
    return result, path


previous.selected_train = selected_train


def capture_val(val, lock):
    binding = val.get('checkpoint_binding', {})
    expected = dict(rule=runner.VAL_BINDING_RULE, checkpoint_sha256=lock['checkpoint_sha256'],
        state_hash=lock['state_hash'], selected_update=lock['selected_update'], archive_identity_unchanged=True)
    capture, identity = val.get('archive_capture_identity'), val.get('identity', {})
    if (not isinstance(capture, dict) or any(binding.get(k) != v for k, v in expected.items())
            or capture.get('experiment') != 'R106' or capture.get('arm') != lock['arm']
            or capture.get('split') != 'val'
            or capture.get('checkpoint_sha256') not in (None, lock['checkpoint_sha256'])
            or binding.get('archive_capture_checkpoint_sha256') != capture.get('checkpoint_sha256')
            or identity != dict(capture, checkpoint_sha256=lock['checkpoint_sha256'])
            or capture.get('state_hash') != lock['state_hash']
            or capture.get('successful_updates') != lock['selected_update']
            or not capture.get('model_state_sha256')
            or capture['model_state_sha256'] != binding.get('model_state_sha256')):
        raise ValueError('R106 selected VAL requires its exact-state parent-lock binding')
    result = copy.deepcopy(val)
    result['identity'] = copy.deepcopy(capture)
    return result


def locked_inputs(root, arm):
    run, config, lock, test, val = shared.locked_inputs(Path(root), arm, experiment='R106', arms=BUDGETS)
    runner.validate_checkpoint_config(config)
    state = torch.load(run / 'best.pt', map_location='cpu', weights_only=False)
    if (state['args'] != config or state.get('state_hash') != lock['state_hash']
            or state.get('successful_updates') != lock['selected_update'] or state.get('epoch') != lock['epoch']
            or config['model_factory'] != runner.MODEL_FACTORY):
        raise ValueError('R106 saved checkpoint differs from the unique per-arm parent lock')
    _, binding = runner.selected_capture_proof(run, val, lock, 'val', state)
    if binding != val.get('checkpoint_binding'):
        raise ValueError('R106 VAL binding differs from actual complete checkpoint state')
    val = capture_val(val, lock)
    model_hash = parameter_hash(state['model'])
    if model_hash != test['identity']['model_state_sha256'] or model_hash != val['identity']['model_state_sha256']:
        raise ValueError('R106 native-inclusive selected VAL/TEST model states differ')
    del state
    return run, config, lock, test, val


def same_data(left, right):
    for key in ('data_root', 'rows_per_problem', 'candidate_pools', 'candidate_pool_ids', 'baselines', 'winner_rule'):
        if key not in left or left[key] != right.get(key):
            raise ValueError('R106 comparison requires identical ' + key)


def selected_predictions(run, config, lock, val, test):
    if val['identity']['model_state_sha256'] != test['identity']['model_state_sha256']:
        raise ValueError('Selected VAL and TEST must use the same whole model')
    packed, methods, summaries, results, paths = helpers.selected_predictions(run, config, lock, val, test)
    name = config['experiment'] + '/' + config['arm']
    return packed, [dict(r, model=name) for r in methods], [dict(r, model=name) for r in summaries], results, paths


def historical_val(run, config, lock, test, val):
    """Old null/admin capture fields are not absence of a real selected VAL."""
    path = common.selected_val_path(run, lock)
    captured = read_json(path.parent / 'identity.json')
    identity = val.get('archive_capture_identity', val['identity'])
    state = torch.load(run / 'best.pt', map_location='cpu', weights_only=False)
    model_hash = parameter_hash(state['model'])
    if (identity != captured or captured.get('experiment') != config['experiment']
            or captured.get('arm') != config['arm'] or captured.get('split') != 'val'
            or captured.get('checkpoint_sha256') not in (None, lock['checkpoint_sha256'])
            or captured.get('state_hash') != lock['state_hash']
            or captured.get('successful_updates') != lock['selected_update']
            or captured.get('model_state_sha256') != model_hash
            or test['identity'].get('model_state_sha256') != model_hash
            or state.get('state_hash') != lock['state_hash'] or state['args'] != config
            or state.get('successful_updates') != lock['selected_update']
            or (lock.get('val') is not None and val['macro'] != lock['val'])):
        raise ValueError('Historical nullable VAL capture differs from exact parent state/update')
    result = copy.deepcopy(val)
    result['identity'] = copy.deepcopy(captured)
    return result


def references(config, args):
    result = {}
    for name, path, experiment, budgets in (
            ('R84A', args.historical_root, 'R84', dict(A_baseline=57600)),
            ('R94B', args.r94_root, 'R94', dict(A=57600, B=57600)),
            ('R102B', args.r102_root, 'R102', dict(A=57600, B=57600))):
        path = Path(path)
        run, old, lock, test, val = shared.locked_inputs(path.parent, path.name, experiment=experiment, arms=budgets)
        same_data(config, old)
        val = historical_val(run, old, lock, test, val)
        result[name] = dict(run=run, config=old, lock=lock, test=test, val=val,
            provenance=dict(model=name, run=str(run), checkpoint_sha256=lock['checkpoint_sha256'],
                selected_update=lock['selected_update'], same_data=True, retrained=False,
                newly_matched_control=False, parameter_matched=False, equal_training_compute_claim=False,
                pretraining_budget_matched=False, administrative_empty_fields_are_not_missing_VAL=True,
                role='fixed target36 anchor' if name == 'R84A' else 'historical non-matched reference'))
    return result


def load_targets(config, old_config, old_lock):
    targets, identity = helpers.load_targets(config, old_config, old_lock)
    if len(targets) != 36:
        raise ValueError('R106 retains exactly frozen R84 target36, never reselects targets')
    return targets, dict(identity, target_count=36, sha256=file_hash(Path(identity['path'])))


def comparison_tables(old, new, targets, identity, model, baseline):
    pairs, recovery, totals = helpers.comparison_tables(old, new, targets, identity, model, baseline)
    for row in recovery:
        left, right = old[row['split']][row['problem']], new[row['split']][row['problem']]
        slot, index = right.pool.index(row['solver']), np.arange(right.n)
        truth, was, chosen = right.winner == slot, left.pred == slot, right.pred == slot
        incoming, outgoing = ~truth & ~was & chosen, ~truth & was & ~chosen
        new_cost, old_cost = right.costs[index, right.pred], left.costs[index, left.pred]
        delta = new_cost - old_cost
        row.update(net_rescues=row['rescued'] - row['lost'],
            new_false_pick_mean_cost=float(new_cost[incoming].mean()) if incoming.any() else None,
            new_false_pick_cost_contribution=float(new_cost[incoming].sum() / right.n),
            new_false_pick_cost_delta_contribution=float(delta[incoming].sum() / right.n),
            removed_false_pick_count=int(outgoing.sum()),
            removed_false_pick_cost_delta_contribution=float(delta[outgoing].sum() / right.n),
            target_winner_cost_delta_contribution=float(delta[truth].sum() / right.n),
            full_problem_mean_cost_delta=float(delta.mean()))
    for row in totals:
        row['net_rescues'] = row['rescued'] - row['lost']
        for key in ('new_false_pick_cost_contribution', 'new_false_pick_cost_delta_contribution',
                    'removed_false_pick_cost_delta_contribution', 'target_winner_cost_delta_contribution'):
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
            'R106/' + config['arm'], name)
    return dict(predictions=packed, methods=methods, summaries=summaries, results=results, paths=paths,
        pairs=pairs, recovery=recovery, recovery_summary=totals, target_identity=identity,
        historical={name: ref['provenance'] for name, ref in refs.items()})


def inference_timing(run, lock):
    value = read_json(Path(run) / 'inference_timing.json')
    required = dict(experiment='R106', arm=lock['arm'], complete=True,
        checkpoint_sha256=lock['checkpoint_sha256'], state_hash=lock['state_hash'],
        selected_update=lock['selected_update'], model_factory=runner.MODEL_FACTORY,
        inference_rule=runner.INFERENCE_RULE, model_reselection=False, repetitions=3, warmup_batches=1,
        scope='online_native_plus_selector_GPU_resident_inputs', all_current_model_included=True,
        donor_present=True, cached_feature_latency_used=False, batch128_amortized=True,
        singleton_measured=False, end_to_end_measured=False, input_labels_read=False,
        sample_policy='first128 frozen TEST inputs per problem',
        excludes='dataset I/O, CPU-to-GPU transfer, training and solver decoding/execution')
    if (any(value.get(k) != v for k, v in required.items()) or lock['arm'] not in ARMS
            or not value.get('model_state_sha256') or not str(value.get('device', '')).startswith('cuda:')
            or value.get('core_parameter_count') != runner.CORE_PARAMETER_COUNT):
        raise ValueError('R106 timing requires actual same-lock online native+selector batch128, not cached latency')
    native, total, trainable = (value.get(k) for k in
        ('native_parameter_count', 'total_parameter_count', 'trainable_parameter_count'))
    if (type(native) is not int or native <= 0 or total != native + runner.CORE_PARAMETER_COUNT
            or trainable != (runner.CORE_PARAMETER_COUNT if lock['arm'] == ARMS[0] else total)
            or value.get('native_trainable_parameter_count') != (0 if lock['arm'] == ARMS[0] else native)):
        raise ValueError('R106 timing must distinguish total, core and native trainable counts')
    rows = value.get('per_problem', [])
    if len(rows) != len(PROBLEMS) or {r['problem'] for r in rows} != set(PROBLEMS):
        raise ValueError('R106 online timing must cover all18 tasks exactly once')
    for row in rows:
        measured = row.get('online_native_plus_selector', {})
        samples, mean = measured.get('batch_seconds', []), measured.get('mean_seconds_per_instance')
        if (row.get('instances') != 128 or len(samples) != 3
                or any(not isinstance(s, (float, int)) or not math.isfinite(s) or s <= 0 for s in samples)
                or not isinstance(mean, (float, int)) or not math.isfinite(mean) or mean <= 0
                or not math.isclose(mean, sum(samples) / (3 * 128), rel_tol=1e-10)
                or measured.get('donor_calls') != [1, 1, 1] or measured.get('warmup_donor_calls') != 1
                or measured.get('native_output_dtype') != ['torch.float32']):
            raise ValueError('R106 timing requires real donor hooks, FP32 output and exact batch128 denominator')
    return value


def validate_lib_population(predictions, config):
    for dataset, problem in LIBS.items():
        data = predictions[dataset, problem]
        if (data.n != LIB_COUNTS[dataset] or data.pool != config['candidate_pools'][problem]
                or data.pool_ids.tolist() != config['candidate_pool_ids'][problem]):
            raise ValueError('R106 must retain native TSPLIB49/CVRPLIB100 and the same legal candidate bank')


def _completed_lib(run, lock):
    value = read_json(Path(run) / 'lib_stage_complete.json')
    expected = dict(experiment='R106', arm=lock['arm'], complete=True, exit_status=0,
        checkpoint_sha256=lock['checkpoint_sha256'], state_hash=lock['state_hash'],
        selected_update=lock['selected_update'], model_factory=runner.MODEL_FACTORY,
        native_counts=LIB_COUNTS, native_execution='online_all_splits', native_feature_cache=False)
    if any(value.get(k) != v for k, v in expected.items()) or set(value.get('datasets', [])) != set(LIBS):
        raise ValueError('R106 requires complete same-lock online native49/100 LIB evidence')
    return value


def lib_stage(args):
    if not str(args.device).startswith('cuda:'):
        raise ValueError('R106 LIB needs the explicit launcher-assigned CUDA device')
    run, config, lock, test, _ = locked_inputs(args.root, args.arm)
    with (run / 'lib_stage.lock').open('a') as claim:
        fcntl.flock(claim, fcntl.LOCK_EX | fcntl.LOCK_NB)
        def phase(name, **fields):
            write_json(run / 'lib_stage_status.json', dict(experiment='R106', arm=args.arm,
                phase=name, updated_utc=utc(), **fields))
        try:
            reference = previous.validate_nss_reference(config, args.nss_root)
            results = common.r80_lib(args.root, run, config, lock, args.device, args.lib_batch_size,
                phase, test['identity']['model_state_sha256'], factory_module=FACTORY)
            if set(results) != set(LIBS) or any(results[d]['train_sbs_reference']['n'] != n for d, n in LIB_COUNTS.items()):
                raise ValueError('R106 LIB must retain every native49/100 instance')
            write_json(run / 'lib_stage_complete.json', dict(experiment='R106', arm=args.arm,
                complete=True, exit_status=0, completed_utc=utc(), checkpoint_sha256=lock['checkpoint_sha256'],
                state_hash=lock['state_hash'], selected_update=lock['selected_update'], model_factory=runner.MODEL_FACTORY,
                model_state_sha256=test['identity']['model_state_sha256'], datasets=list(results),
                native_counts=LIB_COUNTS, native_execution='online_all_splits', native_feature_cache=False,
                nss_reference=reference))
            phase('complete', exit_status=0)
            return results
        except BaseException as error:
            phase('lib_failed', exit_status=1, error=str(error), automatic_retry=False)
            raise


def render_environment():
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib.ft2font import FT2Font
    path = Path(common.reporting_helpers().FONT)
    if not path.is_file() or any(not FT2Font(str(path)).get_char_index(ord(c)) for c in '\u6a21\u578b\u9009\u62e9\u771f\u5b9e'):
        raise ValueError('R106 PDFs require an installed CJK font; do not silently render missing glyphs')
    return dict(backend='Agg', CJK_font=str(path), CJK_font_sha256=file_hash(path))


def validated_history(run, lock):
    history = read_json(Path(run) / 'history.json')
    if ([r.get('successful_updates') for r in history] != list(range(1440, 57601, 1440))
            or any(not r.get('val') for r in history)
            or any(not r.get('train') for r in history if r['successful_updates'] % 7200 == 0)
            or not any(r.get('train') and r['successful_updates'] == lock['selected_update'] for r in history)):
        raise ValueError('R106 needs all40 VAL points, TRAIN7200 and selected TRAIN')
    costs = [r['val']['macro']['macro_vs_sbs_pct'] for r in history]
    if (not all(math.isfinite(c) for c in costs)
            or history[min(range(len(costs)), key=costs.__getitem__)]['successful_updates'] != lock['selected_update']):
        raise ValueError('R106 selected checkpoint is not strict whole18 minimum VAL, first exact tie')
    return history


def learning_curves(run, output, model, lock, **kwargs):
    import matplotlib.pyplot as plt
    history = validated_history(run, lock)
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle(model + ' | original R77; online native; one VAL checkpoint')
    rows = []
    try:
        for record in history:
            row = dict(successful_updates=record['successful_updates'])
            for prefix, values in [('optimization', record.get('optimization', {})),
                    *[(s, record[s]['macro']) for s in ('train', 'val') if record.get(s)]]:
                row.update({prefix + '_' + k: v for k, v in values.items() if isinstance(v, (int, float))})
            rows.append(row)
        for key in ('loss', 'ce', 'cmp', 'risk', 'kl'):
            points = [(r['successful_updates'], r['optimization'][key]) for r in history if key in r.get('optimization', {})]
            if len(points) != 40 or not all(math.isfinite(y) for _, y in points):
                raise ValueError('Missing/nonfinite original R106 loss curve: ' + key)
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
                    raise ValueError('Missing/nonfinite R106 TRAIN/VAL metric curve: ' + key)
                axis.plot(*zip(*points), label=split.upper())
            axis.set_title(title)
            axis.legend()
        for axis in axes.flat:
            axis.axvline(lock['selected_update'], color='#c45358', linestyle='--', linewidth=1)
            axis.set_xlabel('Successful logical-batch1024 updates')
            axis.grid(alpha=.2)
        fig.tight_layout(rect=(0, 0, 1, .95))
        fig.savefig(Path(output) / 'learning_curves.pdf')
    finally:
        plt.close(fig)
    common.write_csv(Path(output) / 'learning_curves.csv', rows)


shared.learning_curves = learning_curves


def val_test_rows(metrics):
    rows = []
    for problem in ('ALL', *PROBLEMS):
        row = dict(problem=problem)
        for split in ('VAL', 'TEST'):
            source = next(r for r in metrics if r['split'] == split and r['problem'] == problem)
            row.update({split.lower() + '_' + k: source[k] for k in
                ('top1_pct', 'mean_cost', 'sbs_cost', 'oracle_cost', 'vs_sbs_pct', 'vs_oracle_pct', 'actual_regret_pct')})
        rows.append(row)
    return rows


def nss_comparison_rows(config, rows):
    expected = {'TSP TEST', 'CVRP TEST', 'TSPLIB', 'CVRPLIB'}
    if len(rows) != 4 or {r['dataset'] for r in rows} != expected:
        raise ValueError('R106 NSS comparison must contain the same current model on exactly four datasets')
    return [dict(r, data_version=('NEW90k frozen TEST4500' if r['dataset'].endswith(' TEST')
                    else 'original native LIB / historical NSS comparison'),
        data_root=config['data_root'], candidate_pool=json.dumps(config['candidate_pools'][r['problem']]),
        candidate_pool_ids=json.dumps(config['candidate_pool_ids'][r['problem']])) for r in rows]


def diagnostics_pdf(output, config, lock, tables, timing, initialization):
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    helper, model = common.reporting_helpers(), 'R106/' + config['arm']
    path = output / 'target_recovery_and_train.pdf'
    temporary = path.with_name(f'.{path.stem}.{os.getpid()}.pdf')
    try:
        with PdfPages(temporary) as pdf:
            fig = common._page(helper, model + ' | Native adaptation and actual latency',
                'Real online native+selector; GPU-resident batch128 amortized, not singleton or I/O end-to-end.', model, lock, 1)
            helper.add_table(fig, ['Problem', 'Instances', 'Online milliseconds / instance'],
                [[r['problem'], '128', f"{1000 * r['online_native_plus_selector']['mean_seconds_per_instance']:.6f}"]
                 for r in timing['per_problem']], [.05, .40, .9, .43], font=9)
            fig.text(.05, .35, f"Total: {initialization['total_parameter_count']:,}; trainable: {initialization['trainable_parameter_count']:,}; native trainable: {initialization['native_trainable_parameter_count']:,}", fontsize=10)
            fig.text(.05, .30, '\n'.join(textwrap.wrap(BOUNDARY, 128)), fontsize=7, va='top')
            pdf.savefig(fig)
            plt.close(fig)
            fig = common._page(helper, model + ' | Same selected checkpoint: full TRAIN',
                'Diagnostic only; checkpoint selected on whole18 VAL, SBS fitted on full frozen TRAIN.', model, lock, 2)
            helper.add_table(fig, ['Problem', 'Top1 (%)', 'SBS Top1 (%)', 'mean_cost', 'SBS cost',
                'Oracle cost', 'vs_SBS (%)', 'vs_Oracle (%)', 'actual regret (%)'],
                [helper.metric_cells(r, r['problem']) for r in helpers.train_metric_rows(config, lock, tables)],
                [.045, .22, .91, .62], font=9, bold_first=True)
            fig.text(.05, .15, 'ALL equal-task; ratios of mean costs differ from per-instance actual regret.', fontsize=10)
            pdf.savefig(fig)
            plt.close(fig)
            for page, ref in enumerate(('R84A', 'R94B', 'R102B'), 3):
                fig = common._page(helper, model + ' | Fixed target36 vs ' + ref,
                    'True rescues/losses, false picks and aligned net cost. Historical references are NOT matched.', model, lock, page)
                helper.add_table(fig, ['Split', 'Winners', 'Rescued', 'Lost', 'New false', 'Net cost delta'],
                    [[r['split'], str(r['winner_count']), str(r['rescued']), str(r['lost']),
                      str(r['new_false_pick_count']), f"{r['full18_net_mean_cost_delta']:.9f}"]
                     for r in tables['recovery_summary'][ref]], [.05, .57, .9, .25], font=10)
                helper.add_table(fig, ['Split', 'Top1 delta pp', 'Corrected', 'Harmed', 'Cost delta', 'Regret delta pp'],
                    [[r['split'], f"{r['top1_delta_pp']:.6f}", str(r['corrected']), str(r['harmed']),
                      f"{r['mean_cost_delta']:.9f}", f"{r['actual_regret_delta_pp']:.6f}"]
                     for r in tables['pairs'][ref] if r['problem'] == 'ALL'], [.05, .25, .9, .22], font=10)
                fig.text(.05, .17, 'Raw cost delta is aligned whole-population, equal-task macro; not target-choice frequency.', fontsize=9)
                fig.text(.05, .12, 'No matched historical pretraining/compute claim; TEST development-exposed; no rule reselection.', fontsize=9)
                pdf.savefig(fig)
                plt.close(fig)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return str(path)


def _verify_pdfs(output):
    for name in REQUIRED_PDFS:
        if not (output / name).is_file() or not (output / name).stat().st_size:
            raise ValueError('R106 report PDF missing: ' + name)


def report(args):
    if str(args.device) != 'cpu':
        raise ValueError('R106 report uses saved evidence on CPU, never model inference/reselection')
    run, config, lock, test, val = locked_inputs(args.root, args.arm)
    with (run / 'postprocess.lock').open('a') as claim:
        fcntl.flock(claim, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            write_json(run / 'postprocess_status.json', dict(experiment='R106', arm=args.arm,
                phase='loading_saved_predictions', complete=False, updated_utc=utc()))
            rendering = render_environment()
            lib, timing = _completed_lib(run, lock), inference_timing(run, lock)
            expected_hash = test['identity']['model_state_sha256']
            if timing['model_state_sha256'] != expected_hash or lib['model_state_sha256'] != expected_hash:
                raise ValueError('R106 timing/LIB differs from selected TEST model')
            reference = previous.validate_nss_reference(config, args.nss_root)
            predictions, metrics, methods = shared.load_saved_splits(run, config, lock, test, val)
            shared.saved_libs(run, config, lock, test, predictions, metrics, methods)
            validate_lib_population(predictions, config)
            comparisons, alignment = shared.nss_comparisons(config, lock, predictions, args.nss_root)
            comparisons = nss_comparison_rows(config, comparisons)
            tables = diagnostic_tables(run, config, lock, val, test, args)
            initialization = read_json(run / 'initialization.json')
            for k in ('total_parameter_count', 'trainable_parameter_count', 'native_parameter_count',
                      'native_trainable_parameter_count', 'core_parameter_count'):
                if initialization.get(k) != timing.get(k):
                    raise ValueError('R106 initialization and locked model count differ: ' + k)
            result = shared.build_reports(run, config, lock, test, val,
                predictions, metrics, methods, comparisons, alignment)
            output = run / 'full_model_statistics'
            train_metrics = helpers.train_metric_rows(config, lock, tables)
            for name, rows in (('metrics_by_problem.csv', [*train_metrics, *metrics]),
                    ('val_test_side_by_side.csv', val_test_rows(metrics)),
                    ('method_selection_train_val_test.csv', tables['methods']),
                    ('zero_selection_summary.csv', tables['summaries']),
                    ('train_metrics_by_problem.csv', train_metrics),
                    ('inference_timing.csv', [dict(problem=r['problem'], instances=128,
                        online_milliseconds_per_instance=1000 * r['online_native_plus_selector']['mean_seconds_per_instance'],
                        batch128_amortized=True, singleton_measured=False, end_to_end_measured=False)
                        for r in timing['per_problem']])):
                common.write_csv(output / name, rows)
            for ref in tables['historical']:
                for stem, key in (('paired_changes', 'pairs'), ('target_recovery', 'recovery'),
                                  ('target_recovery_summary', 'recovery_summary')):
                    common.write_csv(output / (stem + '_vs_' + ref + '.csv'), tables[key][ref])
            for stem, key in (('target_recovery', 'recovery'), ('target_recovery_summary', 'recovery_summary')):
                common.write_csv(output / (stem + '.csv'), tables[key]['R84A'])
            result.update(target_recovery_pdf=diagnostics_pdf(output, config, lock, tables, timing, initialization),
                inference_timing=timing, historical_reference=tables['historical'], nss_reference=reference)
            provenance = read_json(output / 'protocol.json')
            provenance.update(r106_schema_version=1, model_spec=config['model_spec'], initialization=initialization,
                target_identity=tables['target_identity'], historical_reference=tables['historical'],
                inference_timing=timing, evidence_boundary=BOUNDARY, rendering=rendering,
                native_execution='online_all_splits', native_feature_cache=False, native_lr_scale=.1,
                native_pair_context=runner.PAIR_CONTEXT_RULE, trainable_parameter_matched=False,
                equal_training_compute_claim=False, external_pretraining_data_overlap='unknown',
                native_lib_counts=LIB_COUNTS, nss_reference=reference,
                selected_train_capture_proof=tables['results']['TRAIN']['checkpoint_binding'],
                selected_val_binding=read_json(run / 'final_metrics.json')['val']['checkpoint_binding'])
            _verify_pdfs(output)
            write_json(output / 'protocol.json', provenance)
            write_json(run / 'postprocess_status.json', dict(experiment='R106', arm=args.arm,
                phase='complete', exit_status=0, completed_utc=utc(), **result))
            return result
        except BaseException as error:
            write_json(run / 'postprocess_status.json', dict(experiment='R106', arm=args.arm,
                phase='postprocess_failed', complete=False, exit_status=1, updated_utc=utc(),
                error=str(error), automatic_retry=False))
            raise


def validate_matched_configs(left, right, initializations):
    same_data(left, right)
    if (left.get('arm'), right.get('arm')) != ARMS:
        raise ValueError('R106 comparison is complete frozen A versus finetuned B')
    for config in (left, right):
        runner.validate_checkpoint_config(config)
    for key in ('main_protocol_lock', 'model_spec', 'common_trainable_sha256', 'preparation_sha256',
                'geometry_stats', 'task_order_sha256', 'source_plan', 'source_sha256', 'initial_model_state_sha256',
                'microbatch_caps', 'eval_batch_size', 'loss', 'objective', 'lr', 'optimizer_groups',
                'native_pair_context', 'checkpoint_rule', 'inference_rule'):
        if key not in left or left[key] != right.get(key):
            raise ValueError('R106 matched scientific control differs: ' + key)
    a, b = (initializations[arm] for arm in ARMS)
    if (not a.get('common_fresh_model_state_sha256')
            or a['common_fresh_model_state_sha256'] != b.get('common_fresh_model_state_sha256')
            or a['common_fresh_model_state_sha256'] != left['initial_model_state_sha256']
            or b['common_fresh_model_state_sha256'] != right['initial_model_state_sha256']
            or any(not r.get('complete_state_equality_required') for r in (a, b))
            or a['total_parameter_count'] != b['total_parameter_count']
            or a['core_parameter_count'] != b['core_parameter_count']
            or a['native_parameter_count'] != b['native_parameter_count']
            or a['native_trainable_parameter_count'] != 0
            or b['native_trainable_parameter_count'] != b['native_parameter_count']):
        raise ValueError('R106 requires identical complete initial native/selector/buffer tensors, not matched counts alone')
    return dict(identical_complete_fresh_parameters=True, cross_arm_complete_state_verified=True,
        common_fresh_model_state_sha256=a['common_fresh_model_state_sha256'],
        total_tensor_count_matched=True, trainable_parameter_matched=False, equal_training_compute_claim=False,
        native_pretraining_identical_between_current_arms=True, external_pretraining_data_overlap='unknown')


def completed_arm_report(run, lock):
    _completed_lib(run, lock)
    output = Path(run) / 'full_model_statistics'
    status, protocol = read_json(Path(run) / 'postprocess_status.json'), read_json(output / 'protocol.json')
    if (status.get('complete') is not True or status.get('phase') != 'complete' or status.get('exit_status') != 0
            or protocol.get('complete') is not True or protocol.get('checkpoint_lock') != lock
            or protocol.get('target_identity', {}).get('target_count') != 36
            or protocol.get('native_lib_counts') != LIB_COUNTS):
        raise ValueError('R106 compare requires complete same-lock TRAIN/VAL/TEST/LIB/PDF/fixed36 evidence')
    _verify_pdfs(output)


def comparison_pdf(output, locks, metrics, pairs, totals, matched):
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    helper = common.reporting_helpers()
    helper.setup_style()
    path = output / 'COMPARISON.pdf'
    temporary = path.with_name(f'.{path.stem}.{os.getpid()}.pdf')
    try:
        with PdfPages(temporary) as pdf:
            fig = common._page(helper, 'R106 | Native adaptation: whole A/B',
                'Identical initial complete state, different trainable freedom/compute; no TEST/LIB model selection.',
                'R106/B_finetune vs A_frozen', locks[ARMS[1]], 1)
            helper.add_table(fig, ['Model', 'Split / Problem', 'Top1 (%)', 'mean_cost', 'vs_SBS (%)', 'regret (%)'],
                [[r['model'], r['split'] + '/' + r['problem'], f"{r['top1_pct']:.5f}", f"{r['mean_cost']:.8f}",
                  f"{r['vs_sbs_pct']:.5f}", f"{r['actual_regret_pct']:.5f}"]
                 for r in metrics if r['problem'] == 'ALL' or r['split'] in LIBS], [.04, .43, .92, .39], font=9)
            fig.text(.05, .36, 'Native LR B = common x0.1 from first update; A frozen; both native eval and all splits online.', fontsize=10)
            fig.text(.05, .31, 'Initial state SHA256: ' + matched['common_fresh_model_state_sha256'], fontsize=8)
            fig.text(.05, .25, '\n'.join(textwrap.wrap(BOUNDARY, 128)), fontsize=7, va='top')
            pdf.savefig(fig)
            plt.close(fig)
            fig = common._page(helper, 'R106 | Fixed target36 B versus A',
                'True recovery, false picks and aligned full-population net cost; not method-frequency changes.',
                'R106/B_finetune vs A_frozen', locks[ARMS[1]], 2)
            helper.add_table(fig, ['Split', 'Winners', 'Rescued', 'Lost', 'New false', 'Full18 net cost delta'],
                [[r['split'], str(r['winner_count']), str(r['rescued']), str(r['lost']),
                  str(r['new_false_pick_count']), f"{r['full18_net_mean_cost_delta']:.9f}"] for r in totals],
                [.05, .56, .9, .24], font=10)
            helper.add_table(fig, ['Split', 'Top1 delta pp', 'Corrected', 'Harmed', 'Cost delta', 'Regret delta pp'],
                [[r['split'], f"{r['top1_delta_pp']:.6f}", str(r['corrected']), str(r['harmed']),
                  f"{r['mean_cost_delta']:.9f}", f"{r['actual_regret_delta_pp']:.6f}"]
                 for r in pairs if r['problem'] == 'ALL'], [.05, .25, .9, .21], font=10)
            fig.text(.05, .16, 'ALL equal-task; raw mean cost is descriptive only. No per-problem checkpoint assembly.', fontsize=10)
            pdf.savefig(fig)
            plt.close(fig)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return str(path)


def compare(args):
    if str(args.device) != 'cpu':
        raise ValueError('R106 comparison uses saved evidence on CPU')
    selected, configs, locks, initializations, timings, metrics = {}, {}, {}, {}, {}, []
    for arm in ARMS:
        run, config, lock, test, val = locked_inputs(args.root, arm)
        completed_arm_report(run, lock)
        selected[arm] = selected_predictions(run, config, lock, val, test)[0]
        predictions, rows, methods = shared.load_saved_splits(run, config, lock, test, val)
        shared.saved_libs(run, config, lock, test, predictions, rows, methods)
        validate_lib_population(predictions, config)
        metrics.extend(dict(row, model='R106/' + arm) for row in rows)
        configs[arm], locks[arm] = config, lock
        initializations[arm], timings[arm] = read_json(run / 'initialization.json'), inference_timing(run, lock)
    matched = validate_matched_configs(configs[ARMS[0]], configs[ARMS[1]], initializations)
    refs = references(configs[ARMS[0]], args)
    targets, identity = load_targets(configs[ARMS[0]], refs['R84A']['config'], refs['R84A']['lock'])
    pairs, recovery, totals = comparison_tables(selected[ARMS[0]], selected[ARMS[1]],
        targets, identity, 'R106/' + ARMS[1], 'R106/' + ARMS[0])
    output = args.root / 'comparison_artifacts'
    output.mkdir(parents=True, exist_ok=True)
    for name, rows in (('paired_changes_B_vs_A.csv', pairs), ('target_recovery_B_vs_A.csv', recovery),
                       ('target_recovery_summary_B_vs_A.csv', totals), ('whole_model_metrics.csv', metrics)):
        common.write_csv(output / name, rows)
    render_environment()
    result = dict(experiment='R106', complete=True, exit_status=0, **matched,
        comparison_pdf=comparison_pdf(output, locks, metrics, pairs, totals, matched),
        checkpoints=locks, target_identity=identity, inference_timing=timings,
        historical_reference={k: r['provenance'] for k, r in refs.items()}, historical_references_matched=False,
        evidence_boundary=BOUNDARY, training=False, inference=False, checkpoint_reselection=False)
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
    result.add_argument('--r94-root', type=Path, default=DEFAULT_R94B)
    result.add_argument('--r102-root', type=Path, default=DEFAULT_R102B)
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    for name in ('root', 'nss_root', 'historical_root', 'r94_root', 'r102_root'):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    if args.command == 'compare':
        return compare(args)
    if args.arm is None:
        raise ValueError('--arm is required for R106 lib/report/status')
    if args.command == 'status':
        path = args.root / args.arm / 'postprocess_status.json'
        value = read_json(path) if path.exists() else dict(experiment='R106', phase='not_started')
        print(json.dumps(value, indent=2))
        return value
    return lib_stage(args) if args.command == 'lib' else report(args)


if __name__ == '__main__':
    main()

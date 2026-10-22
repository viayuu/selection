"""R101 saved-evidence CPU reports and separately authorized native LIB inference.

Private copies of R99/R93's report helpers retain native winner alignment,
same-axis distributions, NSS4 and fixed R84 target36 without an NCO branch.
"""

import argparse
import copy
import fcntl
import importlib.util
import json
import math
import os
import shutil
import subprocess
import sys
import traceback
from pathlib import Path

from .r76_experiment import file_hash, read_json, utc, write_json


V4 = Path(__file__).resolve().parent
DEFAULT_ROOT = V4 / 'runs/R101_hint_pretraining'
ARMS, BUDGETS = ('A', 'B'), dict(A=57600, B=57600)
VARIANTS = dict(A='random', B='pretrained')
FACTORY = 'code.V4.r101_experiment'
MODEL_FACTORY = FACTORY + '.load_checkpoint'
REQUIRED_PDFS = ('MODEL_STATISTICS.pdf', 'selection_distributions.pdf',
    'target_recovery_and_train.pdf', 'learning_curves.pdf',
    'training_and_method_diagnostics.pdf', 'pretraining_curves.pdf')
REFERENCE_BOUNDARY = (
    'Matched A/B: random R94A versus pinned pretrained R94B TRAIN node representations; '
    'both12800 hint updates +57600 original full18 R77 updates, seed2/logical batch1024. '
    'Same original fresh R84 initialization and TRAIN geometry; transfer instance_encoder only. '
    'Original R88B physical TSP node0, hand34/methodID, joint2, score MLP/maximin,1842953 '
    'parameters. No added head/adapter, teacher input or hint MSE in Stage II or deployment. '
    'Original CE=.35, comparison=.35, risk=.02, R-Drop=.35, dropout=.1. '
    'Stage I uses TRAIN81000/task over CVRP+15MVRP, no outcome labels or teacher VAL/TEST; '
    'fixed endpoint, no VAL selection. Stage II resets optimizer/scaler/RNG/task/sample streams. '
    'R89B main / R84A secondary are historical same-data references, not total-budget-matched '
    'controls. R94B has additional external donor pretraining history of partly unknown provenance; '
    '16-task interface coverage is not16-task source pretraining. The paper memo is conditional '
    'motivation, not measured success or an algorithmic novelty claim. Fixed R84 target36; '
    'selection proportions and zero-count movement are not winner accuracy or true recovery. '
    'Native/FP64 ind winners, original legal pools, TRAIN-fitted SBS; ALL is task-equal. '
    'Development-exposed TEST; one unique minimum full18 VAL macro_vs_sbs checkpoint; '
    'no TEST/LIB tuning or per-task assembly. Latency measures only the original deployed student '
    'on GPU-resident original inputs; excludes I/O/host transfer; no speed claim from R94 cached latency.')
BOUNDARY = REFERENCE_BOUNDARY


def _private_helpers():
    modules = {}
    for stem in ('r80', 'r84', 'r85', 'r87', 'r93', 'r99'):
        name = 'code.V4._r101_' + stem + '_reporting'
        spec = importlib.util.spec_from_file_location(name, V4 / (stem + '_reporting.py'))
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        modules[stem] = module
    common, shared, previous, helpers, r93, reuse = (modules[k] for k in ('r80', 'r84', 'r85', 'r87', 'r93', 'r99'))
    shared.common = common
    previous.shared, previous.common = shared, common
    helpers.shared, helpers.previous, helpers.common = shared, previous, common
    for module in (r93, reuse):
        module.helpers, module.shared, module.previous, module.common = helpers, shared, previous, common
    reuse.r93 = r93
    return reuse, r93, helpers, shared, previous, common


reuse, r93, helpers, shared, previous, common = _private_helpers()
PROBLEMS = tuple(helpers.PROBLEMS)
DEFAULT_NSS, DEFAULT_R84 = helpers.DEFAULT_NSS, helpers.DEFAULT_R84
DEFAULT_R89B = DEFAULT_ROOT.parent / 'R89_optimizer_geometry/B_muon_hybrid'


def validate_report_config(config):
    # Dictionary validation imports no feature branch and never loads a model.
    importlib.import_module(FACTORY).validate_checkpoint_config(config)


def locked_inputs(root, arm):
    if arm not in ARMS:
        raise ValueError('R101 requires arm A or B')
    result = shared.locked_inputs(Path(root), arm, experiment='R101', arms=BUDGETS)
    validate_report_config(result[1])
    return result


def require_cpu(args):
    if str(args.device) != 'cpu':
        raise ValueError('R101 report/compare must use --device cpu over saved evidence')


def same_data(left, right, *, matched=False):
    fields = ('data_root', 'rows_per_problem', 'baselines', 'candidate_pools',
              'candidate_pool_ids', 'winner_rule')
    if any(k not in left or k not in right or left[k] != right[k] for k in fields):
        raise ValueError('R101 requires identical frozen data/native labels/legal pools/TRAIN SBS')
    if not matched:
        return
    if (left.get('arm'), right.get('arm')) != ARMS:
        raise ValueError('R101 matched comparison is pretrained B versus random A')
    normalized = []
    for config in (left, right):
        validate_report_config(config)
        for key in ('common_trainable_sha256', 'preparation_sha256', 'task_order_sha256',
                    'unchanged_non_encoder_sha256', 'checkpoint_rule', 'inference_rule'):
            if not config.get(key):
                raise ValueError('R101 matched comparison needs identity: ' + key)
        current = copy.deepcopy(config)
        params = current['model_params']
        if params.pop('r101_arm', None) != config['arm']:
            raise ValueError('R101 model arm differs from protocol')
        lineage = current.pop('pretraining_lineage')
        if lineage['initial_trainable_sha256'] != current['common_trainable_sha256']:
            raise ValueError('R101 A/B must share the original fresh R84 initialization')
        for key in ('arm', 'pretrained_encoder_sha256', 'external_pretraining', 'gpu_type',
                    'hardware', 'device', 'device_name', 'gpu_uuid', 'run_root', 'output_root', 'created_utc'):
            current.pop(key, None)
        normalized.append(current)
    if normalized[0] != normalized[1]:
        raise ValueError('Matched R101 controls differ beyond teacher-derived encoder and disclosed hardware')


def validate_matched_configs(left, right):
    same_data(left, right, matched=True)
    return dict(experiment='R101', matched=True, arms=list(ARMS),
        teacher_variants=VARIANTS.copy(), pretraining_updates=12800, downstream_updates=57600,
        common_trainable_sha256=left['common_trainable_sha256'],
        unchanged_non_encoder_sha256=left['unchanged_non_encoder_sha256'],
        task_order_sha256=left['task_order_sha256'], preparation_sha256=left['preparation_sha256'],
        teacher_lineage={c['arm']: c['pretraining_lineage'] for c in (left, right)},
        hardware=reuse.hardware_disclosure(dict(A=left, B=right)),
        historical_references_total_budget_matched=False, external_teacher_history_matched=False,
        training=False, inference=False)


def selected_predictions(run, config, lock, val, test):
    return reuse.selected_predictions(run, config, lock, val, test)


def references(config, args):
    result = r93.references(config, args)
    for ref in result.values():
        ref['provenance'].update(total_budget_matched=False,
            reason='Historical selector lacks the extra12800 TRAIN hint updates')
    return result


def load_targets(config, historical_config, historical_lock):
    return reuse.load_targets(config, historical_config, historical_lock)


def val_screen(config, lock, val, refs):
    return [dict(row, model='R101/' + config['arm'], total_budget_matched=False)
            for row in r93.val_screen(config, lock, val, refs)]


def comparison_tables(old, new, targets, identity, model, baseline):
    """Retain real rescues/losses and add original-cost impact of false picks."""
    import numpy as np
    pairs, recovery, totals = helpers.comparison_tables(old, new, targets, identity, model, baseline)
    for row in recovery:
        left, right = old[row['split']][row['problem']], new[row['split']][row['problem']]
        slot = right.pool.index(row['solver'])
        truth, was, chosen = right.winner == slot, left.pred == slot, right.pred == slot
        false = ~truth & ~was & chosen
        index = np.arange(right.n)
        new_cost, old_cost = right.costs[index, right.pred], left.costs[index, left.pred]
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
            old, packed, targets, identity, 'R101/' + config['arm'], name)
    return dict(predictions=packed, methods=methods, summaries=summaries, results=results, paths=paths,
        pairs=pairs, recovery=recovery, recovery_summary=totals, target_identity=identity,
        historical={k: v['provenance'] for k, v in refs.items()},
        val_screen=val_screen(config, lock, val, refs))


def _positive(value, *, zero=False):
    return (not isinstance(value, bool) and isinstance(value, (int, float))
            and math.isfinite(value) and (value >= 0 if zero else value > 0))


def inference_timing(run, lock):
    path = Path(run) / 'inference_timing.json'
    record = read_json(path)
    expected = dict(experiment='R101', complete=True, repetitions=3, warmup_batches=1,
        checkpoint_sha256=lock['checkpoint_sha256'], state_hash=lock['state_hash'],
        inference_rule=lock['inference_rule'], raw_input_source='original_base_cache',
        student_only=True, teacher_arrays_opened=False, online_teacher=False,
        r94_cached_latency_used=False, full_system_measured=False, end_to_end_measured=False,
        model_reselection=False)
    if (record.get('arm') not in ARMS or record.get('arm') != lock.get('arm')
            or any(record.get(k) != v for k, v in expected.items())
            or not record.get('scope') or not record.get('excludes')):
        raise ValueError('R101 latency must measure only this locked original student, not R94 cached latency')
    raw = record.get('per_problem')
    if (not isinstance(raw, list) or len(raw) != len(PROBLEMS)
            or any(not isinstance(r, dict) for r in raw)
            or {r.get('problem') for r in raw} != set(PROBLEMS)):
        raise ValueError('R101 student timing needs all18 problems exactly once')
    rows = []
    for row in raw:
        values = row.get('student', {})
        samples, mean = values.get('batch_seconds'), values.get('mean_seconds_per_instance')
        if (row.get('instances') != 128 or not isinstance(samples, list) or len(samples) != 3
                or not all(_positive(v) for v in samples) or not _positive(mean)
                or not math.isclose(mean, sum(samples) / (3 * 128), rel_tol=1e-6, abs_tol=1e-12)):
            raise ValueError('R101 timing needs three synchronized positive original-student measurements')
        rows.append(dict(problem=row['problem'], instances=128, device=record.get('device'),
            student_ms_per_instance=1000 * mean, repetitions=3))
    return dict(path=str(path.resolve()), sha256=file_hash(path), status='measured_original_student_only',
        measurements=rows, scope=record['scope'], excludes=record['excludes'],
        student_only=True, teacher_arrays_opened=False, r94_cached_latency_used=False,
        full_system_measured=False, end_to_end_measured=False)


def pretraining_evidence(run, config):
    """Read JSON/JSONL only; endpoints and teacher arrays need not be present."""
    from .r101_pretrain import learning_rate
    lineage = config['pretraining_lineage']
    receipt_path, path = run / 'pretrain_complete.json', run / 'pretrain_updates.jsonl'
    receipt = read_json(receipt_path)
    for key, value in lineage.items():
        if receipt.get(key) != value:
            raise ValueError('R101 pretraining report receipt differs from final lineage: ' + key)
    covered = {p for p in PROBLEMS if p not in ('TSP', 'ATSP')}
    if (receipt.get('selected_without_validation') is not True
            or receipt.get('updates_by_problem') != dict.fromkeys(covered, 800)
            or not _positive(receipt.get('seconds'))):
        raise ValueError('R101 Stage I must finish all16 tasks at the fixed endpoint without VAL selection')
    with path.open(encoding='utf-8') as stream:
        records = [json.loads(line) for line in stream if line.strip()]
    if [r.get('successful_updates') for r in records] != [1, *range(16, 12801, 16)]:
        raise ValueError('R101 requires the saved sparse pretraining trace through update12800')
    rows = []
    for record in records:
        update = record['successful_updates']
        if (record.get('experiment') != 'R101' or record.get('arm') != config['arm']
                or record.get('problem') not in covered or record.get('max_updates') != 12800
                or record.get('sample_exposures') != update * 1024
                or not _positive(record.get('loss'), zero=True)
                or not _positive(record.get('grad_norm'), zero=True)
                or not _positive(record.get('seconds_per_update'))
                or not _positive(record.get('lr'))
                or not math.isclose(record['lr'], learning_rate(update), rel_tol=1e-9, abs_tol=1e-14)):
            raise ValueError('R101 pretraining trace is incompatible or nonfinite')
        rows.append(dict(model='R101/' + config['arm'], teacher_variant=VARIANTS[config['arm']],
            successful_updates=update, problem=record['problem'], loss=record['loss'],
            grad_norm=record['grad_norm'], lr=record['lr'], sample_exposures=update * 1024,
            amp_retries=record.get('amp_retries'), seconds_per_update=record['seconds_per_update'],
            loss_scope='logged last-update per-instance/per-node LN-MSE, not interval mean'))
    return dict(receipt_path=str(receipt_path), receipt_sha256=file_hash(receipt_path),
        trace_path=str(path), trace_sha256=file_hash(path), rows=rows,
        endpoint_updates=12800, teacher_split='train', teacher_validation_used=False,
        endpoint_selected_without_validation=True, lineage=copy.deepcopy(lineage))


def pretraining_curves(output, config, lock, evidence):
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    helper = common.reporting_helpers()
    helper.setup_style()
    model = 'R101/' + config['arm']
    path = output / 'pretraining_curves.pdf'
    temporary = path.with_name(f'.{path.stem}.{os.getpid()}.pdf')
    try:
        with PdfPages(temporary) as pdf:
            fig = common._page(helper, model + ' | Two-stage provenance',
                'Fixed TRAIN hint endpoint, then independently reset original full18 learning.', model, lock, 1)
            import textwrap
            fig.text(.05, .82, '\n\n'.join(textwrap.fill(t, 110) for t in (
                'Teacher: ' + VARIANTS[config['arm']] + '. Stage I:12800 updates, logical batch1024, seed2; CVRP+15MVRP TRAIN81000/task only.',
                'Separate non-affine per-node LayerNorm MSE; one dropout forward. No outcome labels, learned adapter or teacher VAL/TEST. Fixed endpoint, not validation selected.',
                'Stage II: fresh original R84/R88B student; ONLY instance_encoder transferred. All non-encoder parameters remain identical fresh R84. Optimizer, scaler, RNG and task/sample streams reset.',
                'Original1842953 parameters; no added heads or features. R77 CE .35 / comparison .35 / risk .02 / R-Drop .35. No hint loss, teacher arrays or online donor in Stage II or deployment.',
                'Both A/B have12800+57600 updates; R89B/R84A are NOT total-budget-matched. B teacher has additional external solver pretraining history. The paper memo provides conditional motivation only.',
                'Native/FP64 ind winners; legal-pool Oracle and TRAIN-fitted SBS; ALL is task-equal. Development-exposed TEST; no TEST/LIB checkpoint selection.',
                'Stage-I encoder SHA256: ' + config['pretrained_encoder_sha256'],
                'Unchanged non-encoder SHA256: ' + config['unchanged_non_encoder_sha256'])),
                va='top', fontsize=10)
            pdf.savefig(fig)
            plt.close(fig)
            rows = evidence['rows']
            fig, axes = plt.subplots(2, 2, figsize=(14, 9))
            fig.suptitle(model + ' | TRAIN-only ' + VARIANTS[config['arm']] + ' node hints; fixed12800 endpoint')
            for axis, key, title in zip(axes.flat, ('loss', 'lr', 'grad_norm', 'seconds_per_update'),
                    ('Logged LN-MSE (last update, not interval mean)', 'AdamW learning rate',
                     'Gradient norm before clipping', 'Cumulative wall seconds / successful update')):
                axis.plot([r['successful_updates'] for r in rows], [r[key] for r in rows], color='#007f79')
                axis.set_title(title)
                axis.set_xlabel('Stage-I successful logical-batch1024 updates')
                axis.grid(alpha=.2)
            fig.tight_layout(rect=(0, 0, 1, .94))
            pdf.savefig(fig)
            plt.close(fig)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    common.write_csv(output / 'pretraining_curves.csv', evidence['rows'])
    return str(path)


def learning_curves(run, config, lock):
    import matplotlib.pyplot as plt
    history = read_json(run / 'history.json')
    if not history or not any(r.get('train') for r in history) or not any(r.get('val') for r in history):
        raise ValueError('R101 needs actual saved TRAIN/VAL learning curves')
    output = run / 'full_model_statistics'
    rows = []
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle('R101/' + config['arm'] + ' | Stage II: original R77, no hint loss or teacher inputs')
    for record in history:
        row = dict(successful_updates=record['successful_updates'])
        row.update({'optimization_' + k: v for k, v in record.get('optimization', {}).items()
                    if isinstance(v, (int, float))})
        for split in ('train', 'val'):
            if record.get(split):
                row.update({split + '_' + k: v for k, v in record[split]['macro'].items()
                            if isinstance(v, (int, float))})
        rows.append(row)
    for key in ('loss', 'ce', 'cmp', 'risk', 'kl'):
        points = [(r['successful_updates'], r['optimization'][key]) for r in history
                  if key in r.get('optimization', {})]
        if points:
            axes[0, 0].plot(*zip(*points), label=key)
    axes[0, 0].set_title('TRAIN original loss components, R-Drop retained')
    if axes[0, 0].lines:
        axes[0, 0].legend(fontsize=8)
    for axis, key, factor, title in (
            (axes[0, 1], 'macro_ce', 1., 'Evaluation native CE'),
            (axes[0, 2], 'macro_top1', 100., 'Native Top1 (%)'),
            (axes[1, 0], 'macro_mean_cost', 1., 'Task-equal mean cost (descriptive)'),
            (axes[1, 1], 'macro_actual_regret_pct', 1., 'Actual regret (%)'),
            (axes[1, 2], 'macro_vs_sbs_pct', 1., 'vs fixed TRAIN SBS (%)')):
        for split in ('train', 'val'):
            points = [(r['successful_updates'], r[split]['macro'][key] * factor) for r in history
                      if r.get(split) and key in r[split]['macro']]
            if points:
                axis.plot(*zip(*points), label=split.upper())
        axis.set_title(title)
        if axis.lines:
            axis.legend()
    for axis in axes.flat:
        axis.axvline(lock['selected_update'], color='#c45358', linestyle='--', linewidth=1)
        axis.set_xlabel('Stage-II successful logical-batch1024 updates')
        axis.grid(alpha=.2)
    fig.tight_layout(rect=(0, 0, 1, .95))
    try:
        fig.savefig(output / 'learning_curves.pdf')
    finally:
        plt.close(fig)
    common.write_csv(output / 'learning_curves.csv', rows)


def _timing_panel(pdf, helper, model, lock, timings, page):
    import matplotlib.pyplot as plt
    fig = common._page(helper, model + ' | Original student latency',
        'First128 GPU-resident original TEST inputs/task; three synchronized repetitions.', model, lock, page)
    helper.add_table(fig, ['Problem', *[arm + ' student ms/row' for arm in timings]],
        [[problem, *[f"{next(r for r in timing['measurements'] if r['problem'] == problem)['student_ms_per_instance']:.6f}"
                     for timing in timings.values()]] for problem in PROBLEMS], [.05, .26, .9, .58], font=9)
    for y, text in ((.20, 'Original1842953-parameter student only; no online donor, teacher feature I/O or solver decoder.'),
            (.15, 'Excludes dataset I/O, host transfer and Stage-I pretraining; not single-instance/E2E latency.'),
            (.10, 'No latency or speedup claim is inferred from R94 cached-selector numbers.'),
            (.05, 'Hardware is disclosed, not bitwise matched; no model reselection based on timing.')):
        fig.text(.05, y, text, fontsize=9)
    pdf.savefig(fig)
    plt.close(fig)


def _recovery_page(pdf, helper, model, lock, baseline, pairs, totals, page):
    import matplotlib.pyplot as plt
    fig = common._page(helper, model + ' | Fixed target36 vs ' + baseline,
        'Native/FP64 winners; true rescues/losses and false-pick cost, not selection-frequency accuracy.', model, lock, page)
    helper.add_table(fig, ['Split', 'True winners', 'Correct', 'Rescued', 'Lost', 'New false', 'Recall (%)'],
        [[r['split'], str(r['winner_count']), str(r['correct_count']), str(r['rescued']), str(r['lost']),
          str(r['new_false_pick_count']), '-' if r['recall_pct'] is None else f"{r['recall_pct']:.4f}"] for r in totals],
        [.05, .64, .9, .19], font=10)
    helper.add_table(fig, ['Split', 'Top1 delta pp', 'Corrected', 'Harmed', 'Cost delta', 'Regret delta pp'],
        [[r['split'], f"{r['top1_delta_pp']:.6f}", str(r['corrected']), str(r['harmed']),
          f"{r['mean_cost_delta']:.9f}", f"{r['actual_regret_delta_pp']:.6f}"] for r in pairs if r['problem'] == 'ALL'],
        [.05, .40, .9, .19], font=10)
    helper.add_table(fig, ['Split', 'New-false cost contribution', 'New-false cost delta', 'New-false regret pp'],
        [[r['split'], f"{r['new_false_pick_cost_contribution']:.9f}",
          f"{r['new_false_pick_cost_delta_contribution']:.9f}",
          f"{r['new_false_pick_regret_contribution_pp']:.6f}"] for r in totals], [.05, .18, .9, .16], font=10)
    fig.text(.05, .12, 'Cost contributions: sum over newly false-picked instances / task n; ALL averages18 tasks.', fontsize=9)
    fig.text(.05, .07, 'R89B/R84A are historical, not12800+57600 budget-matched; zero-count movement is not success.', fontsize=9)
    pdf.savefig(fig)
    plt.close(fig)


def diagnostics_pdf(output, config, lock, tables, timing):
    from matplotlib.backends.backend_pdf import PdfPages
    helper = common.reporting_helpers()
    helper.setup_style()
    model = 'R101/' + config['arm']
    path = output / 'target_recovery_and_train.pdf'
    temporary = path.with_name(f'.{path.stem}.{os.getpid()}.pdf')
    try:
        with PdfPages(temporary) as pdf:
            for page, ref in enumerate(('R89B', 'R84A'), 1):
                _recovery_page(pdf, helper, model, lock, ref, tables['pairs'][ref], tables['recovery_summary'][ref], page)
            _timing_panel(pdf, helper, model, lock, {config['arm']: timing}, 3)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return str(path)


def merge_pdfs(destination, sources):
    executable = shutil.which('pdfunite')
    if executable is None:
        raise RuntimeError('pdfunite is required to assemble the single offline full-version PDF')
    temporary = destination.with_name(f'.{destination.stem}.{os.getpid()}.pdf')
    try:
        subprocess.run([executable, *map(str, sources), str(temporary)], check=True,
                       capture_output=True, text=True)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def _completed_lib(run, lock):
    return reuse._completed_lib(run, lock)


def completed_arm_report(run, lock):
    _completed_lib(run, lock)
    post = read_json(run / 'postprocess_status.json')
    protocol = read_json(run / 'full_model_statistics/protocol.json')
    if (post.get('complete') is not True or post.get('phase') != 'complete'
            or post.get('exit_status') != 0 or protocol.get('complete') is not True
            or protocol.get('experiment') != 'R101'
            or protocol.get('checkpoint_lock', {}).get('checkpoint_sha256') != lock['checkpoint_sha256']
            or protocol.get('inference_timing', {}).get('sha256') != inference_timing(run, lock)['sha256']
            or protocol.get('pretraining_evidence', {}).get('trace_sha256') != file_hash(run / 'pretrain_updates.jsonl')
            or protocol.get('pretraining_evidence', {}).get('receipt_sha256') != file_hash(run / 'pretrain_complete.json')):
        raise ValueError('R101 comparison requires same-lock completed TRAIN/VAL/TEST/LIB/PDF evidence')
    for name in REQUIRED_PDFS:
        if not (run / 'full_model_statistics' / name).is_file() or not (run / 'full_model_statistics' / name).stat().st_size:
            raise ValueError('R101 arm PDF missing: ' + name)


def lib_stage(args):
    if not str(args.device).startswith('cuda:'):
        raise ValueError('R101 LIB requires an explicit allocated --device cuda:N')
    run, config, lock, test, _ = locked_inputs(args.root, args.arm)
    with (run / 'lib_stage.lock').open('a') as claim:
        fcntl.flock(claim, fcntl.LOCK_EX | fcntl.LOCK_NB)
        def phase(name, **fields):
            write_json(run / 'lib_stage_status.json', dict(experiment='R101', arm=args.arm,
                phase=name, updated_utc=utc(), **fields))
        try:
            reference = previous.validate_nss_reference(config, args.nss_root)
            results = common.r80_lib(args.root, run, config, lock, args.device,
                args.lib_batch_size, phase, test['identity']['model_state_sha256'], factory_module=FACTORY)
            if set(results) != {'TSPLIB', 'CVRPLIB'}:
                raise ValueError('R101 LIB must complete both native datasets using the same student')
            write_json(run / 'lib_stage_complete.json', dict(experiment='R101', arm=args.arm,
                complete=True, exit_status=0, updated_utc=utc(), state_hash=lock['state_hash'],
                checkpoint_sha256=lock['checkpoint_sha256'], model_state_sha256=test['identity']['model_state_sha256'],
                nss_reference=reference, datasets=list(results), online_teacher=False))
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
            write_json(run / 'postprocess_status.json', dict(experiment='R101', arm=args.arm,
                phase='loading_saved_predictions', updated_utc=utc()))
            _completed_lib(run, lock)
            timing, pretraining = inference_timing(run, lock), pretraining_evidence(run, config)
            reference = previous.validate_nss_reference(config, args.nss_root)
            predictions, metrics, methods = shared.load_saved_splits(run, config, lock, test, val)
            shared.saved_libs(run, config, lock, test, predictions, metrics, methods)
            comparisons, alignment = shared.nss_comparisons(config, lock, predictions, args.nss_root)
            if {r['dataset'] for r in comparisons} != {'TSP TEST', 'CVRP TEST', 'TSPLIB', 'CVRPLIB'}:
                raise ValueError('R101 NSS4 must use this one current checkpoint on all four datasets')
            tables = diagnostic_tables(run, config, lock, val, test, args)
            result = shared.build_reports(run, config, lock, test, val,
                predictions, metrics, methods, comparisons, alignment)
            output = run / 'full_model_statistics'
            learning_curves(run, config, lock)
            pretraining_curves(output, config, lock, pretraining)
            # Prepend the two-stage disclosure to the generic inherited report.
            merge_pdfs(output / 'MODEL_STATISTICS.pdf', [output / 'pretraining_curves.pdf', output / 'MODEL_STATISTICS.pdf'])
            for name, rows in (('method_selection_train_val_test.csv', tables['methods']),
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
            pretraining.pop('rows')
            provenance.update(r101_schema_version=1, model_spec=config['model_spec'],
                initialization=config['initialization'], pretraining_lineage=config['pretraining_lineage'],
                hint_pretraining=config['hint_pretraining'], pretraining_evidence=pretraining,
                target_identity=tables['target_identity'], historical_reference=tables['historical'],
                nss_reference=reference, objective=config['objective'], loss=config['loss'],
                evidence_boundary=REFERENCE_BOUNDARY, inference_timing=timing, rdrop=True,
                deployment_teacher=False, external_pretraining=config['external_pretraining'],
                val_empirical_screen=tables['val_screen'], main_pdf_pages=24)
            write_json(output / 'protocol.json', provenance)
            notes = output / 'RESULTS.md'
            notes.write_text(notes.read_text(encoding='utf-8') + '\n' + REFERENCE_BOUNDARY + '\n', encoding='utf-8')
            for name in REQUIRED_PDFS:
                if not (output / name).is_file() or not (output / name).stat().st_size:
                    raise ValueError('R101 report PDF missing: ' + name)
            write_json(run / 'postprocess_status.json', dict(experiment='R101', arm=args.arm,
                phase='complete', exit_status=0, completed_utc=utc(), **result))
            return result
        except BaseException as error:
            write_json(run / 'postprocess_status.json', dict(experiment='R101', arm=args.arm,
                phase='postprocess_failed', exit_status=1, updated_utc=utc(), error=str(error),
                traceback=traceback.format_exc(), automatic_retry=False))
            raise


def compare(args):
    require_cpu(args)
    # A single publication is possible only after BOTH complete model deliveries.
    with (Path(args.root) / 'comparison.lock').open('a') as claim:
        fcntl.flock(claim, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _compare(args)


def _compare(args):
    inputs = {arm: locked_inputs(args.root, arm) for arm in ARMS}
    configs = {arm: inputs[arm][1] for arm in ARMS}
    matched = validate_matched_configs(configs['A'], configs['B'])
    for run, _, lock, _, _ in inputs.values():
        completed_arm_report(run, lock)
    selected, locks, metrics, screens, timings = {}, {}, [], [], {}
    for arm, (run, config, lock, test, val) in inputs.items():
        selected[arm], _, _, results, _ = selected_predictions(run, config, lock, val, test)
        locks[arm] = lock
        refs = references(config, args)
        screens.extend(val_screen(config, lock, val, refs))
        timings[arm] = inference_timing(run, lock)
        for split, result in results.items():
            for problem, saved in [('ALL', result['macro']), *result['per_problem'].items()]:
                prefix = 'macro_' if problem == 'ALL' else ''
                keys = ('top1', 'mean_cost', 'sbs_cost', 'oracle_cost', 'vs_sbs_pct', 'vs_oracle_pct', 'actual_regret_pct')
                metrics.append(dict(model='R101/' + arm, split=split, problem=problem,
                    teacher_variant=VARIANTS[arm], selected_update=lock['selected_update'],
                    checkpoint_sha256=lock['checkpoint_sha256'], **{k: saved[prefix + k] for k in keys}))
    targets, identity = load_targets(configs['B'], refs['R84A']['config'], refs['R84A']['lock'])
    pairs, recovery, totals = comparison_tables(selected['A'], selected['B'], targets, identity, 'R101/B', 'R101/A')
    output = Path(args.root) / 'comparison_artifacts'
    output.mkdir(parents=True, exist_ok=True)
    for name, rows in (('comparison.csv', metrics), ('paired_changes_B_vs_A.csv', pairs),
            ('target_recovery_B_vs_A.csv', recovery), ('target_recovery_summary_B_vs_A.csv', totals),
            ('val_empirical_screen.csv', screens)):
        common.write_csv(output / name, rows)
    from matplotlib.backends.backend_pdf import PdfPages
    helper = common.reporting_helpers()
    helper.setup_style()
    path = output / 'COMPARISON.pdf'
    temporary = path.with_name(f'.{path.stem}.{os.getpid()}.pdf')
    try:
        with PdfPages(temporary) as pdf:
            _recovery_page(pdf, helper, 'R101 pretrained B vs random A', locks['B'], 'R101/A', pairs, totals, 1)
            _timing_panel(pdf, helper, 'R101 B vs A', locks['B'], timings, 2)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    result = dict(experiment='R101', r101_schema_version=1, complete=True, exit_status=0,
        checkpoints=locks, comparison_pdf=str(path), evidence_boundary=REFERENCE_BOUNDARY,
        matched_difference='pinned pretrained R94B versus random R94A teacher, same12800+57600 updates',
        matched_controls=matched, target_identity=identity, inference_timing=timings,
        training=False, inference=False, checkpoint_reselection=False, github_published=False)
    sources = [path]
    for arm in ARMS:
        folder = inputs[arm][0] / 'full_model_statistics'
        sources.extend(folder / name for name in REQUIRED_PDFS if name not in ('selection_distributions.pdf', 'pretraining_curves.pdf'))
    merge_pdfs(Path(args.root) / 'RESULTS.pdf', sources)
    lines = ['# R101: TRAIN Hint Pretraining, Original Student Deployment', '', REFERENCE_BOUNDARY, '',
        '| Model | Split | Top1 (%) | mean_cost | SBS cost | Oracle cost | vs_SBS (%) | vs_Oracle (%) | actual regret (%) |',
        '| --- | --- | --- | --- | --- | --- | --- | --- | --- |']
    for row in metrics:
        if row['problem'] == 'ALL':
            lines.append(f"| {row['model']} | {row['split']} | {100*row['top1']:.6f} | {row['mean_cost']:.9f} | "
                f"{row['sbs_cost']:.9f} | {row['oracle_cost']:.9f} | {row['vs_sbs_pct']:.6f} | "
                f"{row['vs_oracle_pct']:.6f} | {row['actual_regret_pct']:.6f} |")
    (Path(args.root) / 'RESULTS.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    result.update(full_version_pdf=str(Path(args.root) / 'RESULTS.pdf'), both_arms_complete=True)
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
        value = read_json(path) if path.is_file() else dict(experiment='R101', phase='not_started')
        print(json.dumps(value, indent=2))
        return value
    return lib_stage(args) if args.command == 'lib' else report(args)


if __name__ == '__main__':
    main()

"""R95 original-R77 full18/LIB/NSS4 reports and matched TRAIN-sampling contrast.

Only ``lib`` performs inference, on the launcher-owned GPU. ``report`` and
``compare`` are CPU administration over saved predictions, never selection.
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
from .r76_experiment import file_hash, read_json, utc, write_json


ARMS = ('A', 'B')
MAX_UPDATES = 57600
BUDGETS = dict.fromkeys(ARMS, MAX_UPDATES)
DROPOUT = dict(A=.1, B=.1)
DEFAULT_ROOT = Path(__file__).resolve().parent / 'runs/R95_adaptive_example_sampling'
shared, previous, common = helpers.shared, helpers.previous, helpers.common
DEFAULT_NSS, DEFAULT_R84 = helpers.DEFAULT_NSS, helpers.DEFAULT_R84
DEFAULT_R89B = DEFAULT_ROOT.parent / 'R89_optimizer_geometry/B_muon_hybrid'
FACTORY = 'code.V4.r95_experiment'
MODEL_FACTORY = FACTORY + '.load_checkpoint'
PROBLEMS = tuple(helpers.PROBLEMS)
SAMPLING = dict(warm_natural_updates=7200, refresh_every_updates=7200,
    proposal_uniform_mixture=.5, proxy='deterministic_full_loss_final_affine_gradient_norm',
    correction='1/(N*q)', normalization='arithmetic_batch_mean_no_self_normalization')
REQUIRED_PDFS = ('MODEL_STATISTICS.pdf', 'selection_distributions.pdf',
                 'target_recovery_and_train.pdf', 'learning_curves.pdf',
                 'sampling_diagnostics.pdf')
LOSS_LABEL = 'Original R77 CE .35 / comparison .35 / risk .02 / R-Drop ramp .35; dropout .1 in both arms.'
PROXY_BOUNDARY = ('Deterministic eval final-affine gradient norm of CE+comparison+risk, including cancellation; '
    'eval R-Drop/KL is zero. Not full-parameter or stochastic-training gradient norm. '
    'Inverse correction is before global clipping/AdamW, not an unbiased optimizer-update claim.')
REFERENCE_BOUNDARY = (
    'R89B is the main historical same-data reference; R84A is secondary. '
    'Neither is a newly matched control. Both keep the original R88B execution role and original R77 loss/R-Drop. '
    'Both use the same natural rows for first7200 updates; then A uses uniform IID replacement, '
    'B uses an own-TRAIN adaptive proposal with .5 uniform mixture, refreshed every7200 updates. '
    'The entire row objective is corrected by 1/(N*q), arithmetic batch mean, no self-normalization. '
    'A uses3090_0 after R92A full delivery; B uses4090_0 after R92B full delivery. '
    'Common code/data/fresh initialization/update budget, not bitwise hardware-matched. '
    'No guaranteed variance reduction, causal claim, architectural novelty or root-cause proof. '
    'Development-exposed TEST; no TEST/LIB selection or per-task assembly. ' + PROXY_BOUNDARY)
BOUNDARY = REFERENCE_BOUNDARY


def validate_report_config(config):
    arm = config.get('arm')
    expected = dict(experiment='R95', r95_schema_version=1, model_factory=MODEL_FACTORY,
        initialization='scratch_shared_r77', seed=2, global_batch_size=1024,
        logical_global_batch=1024, max_successful_updates=MAX_UPDATES,
        warmup_updates=4320, eval_every_updates=1440, eval_batch_size=256,
        optimizer='AdamW', weight_decay=1e-4, grad_clip=1., dropout=.1,
        rdrop_weight=.35, rdrop_ramp_updates=4320, sampling=SAMPLING,
        selection_head='original_score_mlp_maximin',
        rows_per_problem=dict(train=81000, val=4500, test=4500))
    if arm not in ARMS or any(config.get(key) != value for key, value in expected.items()):
        raise ValueError('R95 requires unchanged original R77 model/loss and matched sampling controls')
    if any(config.get('loss', {}).get(key) != value for key, value in
            dict(ce=.35, comparison=.35, risk=.02).items()):
        raise ValueError('R95 retains original R77 CE/comparison/risk/R-Drop, not plain CE')
    params, roles = config.get('model_params', {}), config.get('input_roles', {})
    if (params.get('r95_arm') != arm or params.get('dropout') != .1
            or params.get('role_arm') != 'B_execution_anchor'
            or roles.get('arm') != 'B_execution_anchor' or roles.get('node_column') != 3
            or config.get('objective', {}).get('forward_passes') != 2):
        raise ValueError('R95 keeps the original execution role and two-pass R-Drop')


def finalized_test(run, config, lock, test):
    """Do not report a TEST result saved before the runner finalized its gate."""
    final = read_json(run / 'final_metrics.json')
    complete = read_json(run / 'test_complete.json')
    checkpoint = lock['checkpoint_sha256']
    if (final.get('complete') is not True or final.get('test_read') is not True
            or final.get('test') != test or final.get('checkpoint_sha256') != checkpoint
            or final.get('state_hash') != lock['state_hash']
            or final.get('selected_update') != lock['selected_update']
            or complete.get('full_tests') != 1 or complete.get('checkpoint_sha256') != checkpoint
            or complete.get('saved_predictions_reusable') is not True):
        raise ValueError('R95 requires finalized same-lock full TEST, not an intermediate result')
    model_hash = test.get('identity', {}).get('model_state_sha256')
    for split, result in (('test', test), ('val', final.get('val', {})), ('train', final.get('train', {}))):
        identity = result.get('identity', {})
        rows = result.get('per_problem', {})
        if (set(rows) != set(PROBLEMS) or not model_hash
                or any(row.get('n') != config['rows_per_problem'][split] for row in rows.values())
                or identity.get('experiment') != 'R95' or identity.get('arm') != config['arm']
                or identity.get('split') != split or identity.get('state_hash') != lock['state_hash']
                or identity.get('successful_updates') != lock['selected_update']
                or identity.get('model_state_sha256') != model_hash
                or identity.get('checkpoint_sha256') not in (None, checkpoint)
                or split == 'test' and identity.get('checkpoint_sha256') != checkpoint):
            raise ValueError('R95 finalized TRAIN/VAL/TEST must cover all18 rows with one model state')


def locked_inputs(root, arm):
    if arm not in ARMS:
        raise ValueError('R95 requires arm A or B')
    result = shared.locked_inputs(root, arm, experiment='R95', arms=BUDGETS)
    validate_report_config(result[1])
    importlib.import_module(FACTORY).validate_checkpoint_config(result[1])
    finalized_test(result[0], result[1], result[2], result[3])
    return result


def require_cpu(args):
    if str(args.device) != 'cpu':
        raise ValueError('R95 PDF/report/compare administration must use --device cpu')


def completed_arm_report(run, lock):
    post = read_json(run / 'postprocess_status.json')
    protocol = read_json(run / 'full_model_statistics/protocol.json')
    lib = read_json(run / 'lib_stage_complete.json')
    checkpoint = lock['checkpoint_sha256']
    if (post.get('complete') is not True or post.get('phase') != 'complete'
            or post.get('exit_status') != 0 or protocol.get('complete') is not True
            or protocol.get('checkpoint_lock', {}).get('checkpoint_sha256') != checkpoint
            or lib.get('complete') is not True or lib.get('exit_status') != 0
            or lib.get('checkpoint_sha256') != checkpoint
            or set(lib.get('datasets', [])) != {'TSPLIB', 'CVRPLIB'}):
        raise ValueError('R95 comparison requires same-lock TEST/LIB/PDF completion with rc0')
    for name in REQUIRED_PDFS:
        path = run / 'full_model_statistics' / name
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError('R95 arm PDF missing: ' + name)


def lib_stage(args):
    if not str(args.device).startswith('cuda'):
        raise ValueError('R95 LIB inference belongs to the launcher-owned GPU stage')
    run, config, lock, test, _ = locked_inputs(args.root, args.arm)
    with (run / 'lib_stage.lock').open('a') as claim:
        fcntl.flock(claim, fcntl.LOCK_EX | fcntl.LOCK_NB)

        def phase(name, **fields):
            write_json(run / 'lib_stage_status.json', dict(experiment='R95', arm=args.arm,
                phase=name, updated_utc=utc(), **fields))

        try:
            reference = previous.validate_nss_reference(config, args.nss_root)
            results = common.r80_lib(args.root, run, config, lock, args.device,
                args.lib_batch_size, phase, test['identity']['model_state_sha256'],
                factory_module=FACTORY)
            write_json(run / 'lib_stage_complete.json', dict(experiment='R95', arm=args.arm,
                complete=True, exit_status=0, updated_utc=utc(),
                checkpoint_sha256=lock['checkpoint_sha256'], state_hash=lock['state_hash'],
                model_state_sha256=test['identity']['model_state_sha256'], nss_reference=reference,
                datasets=list(results)))
            phase('complete', exit_status=0, checkpoint_sha256=lock['checkpoint_sha256'])
            return results
        except BaseException as error:
            phase('lib_failed', exit_status=1, error=str(error), automatic_retry=False)
            raise


def same_data(left, right, *, matched=False):
    fields = ('data_root', 'rows_per_problem', 'baselines', 'candidate_pools',
              'candidate_pool_ids', 'winner_rule')
    if any(name not in left or name not in right or left[name] != right[name] for name in fields):
        raise ValueError('R95 comparison requires identical frozen data/pools/native labels/TRAIN SBS')
    if not matched:
        return
    if (left.get('arm'), right.get('arm')) != ARMS:
        raise ValueError('R95 matched contrast must be B adaptive versus A uniform')
    for config in (left, right):
        validate_report_config(config)
        if any(not config.get(name) for name in
                ('common_trainable_sha256', 'preparation_sha256', 'task_order_sha256',
                 'microbatch_caps', 'checkpoint_rule', 'inference_rule', 'main_protocol_lock')):
            raise ValueError('R95 matched comparison requires fresh init/data/task-order and locked rules')
    runtime = {'arm', 'gpu_type', 'hardware', 'device', 'device_name', 'gpu_uuid',
               'run_root', 'output_root', 'created_utc', 'model_params'}
    params = [copy.deepcopy(config['model_params']) for config in (left, right)]
    for current in params:
        current.pop('r95_arm')
    if ({k: v for k, v in left.items() if k not in runtime}
            != {k: v for k, v in right.items() if k not in runtime} or params[0] != params[1]):
        raise ValueError('Matched R95 A/B may differ only in sampling arm and disclosed hardware')


def validate_matched_configs(left, right):
    same_data(left, right, matched=True)
    return dict(experiment='R95', matched=True, arms=list(ARMS), budgets=BUDGETS.copy(),
        sampling=dict(SAMPLING), dropout=dict(DROPOUT), training=False, inference=False,
        matched_difference='A uniform replacement versus B inverse-corrected adaptive sampling',
        bitwise_hardware_matched=False, causal_claim=False)


def selected_predictions(run, config, lock, val, test):
    if val['identity']['model_state_sha256'] != test['identity']['model_state_sha256']:
        raise ValueError('R95 diagnostics require identical locked VAL/TEST model states')
    packed, methods, summaries, results, paths = helpers.selected_predictions(run, config, lock, val, test)
    name = config['experiment'] + '/' + config['arm']
    return (packed, [dict(r, model=name) for r in methods],
            [dict(r, model=name) for r in summaries], results, paths)


def references(config, args):
    result = {}
    for name, path, experiment, budgets in (
            ('R89B', args.r89_root, 'R89', dict(A_adamw=57600, B_muon_hybrid=57600)),
            ('R84A', args.historical_root, 'R84', shared.BUDGETS)):
        path = Path(path)
        run, old_config, lock, test, val = shared.locked_inputs(
            path.parent, path.name, experiment=experiment, arms=budgets)
        same_data(config, old_config)
        result[name] = dict(run=run, config=old_config, lock=lock, test=test, val=val,
            provenance=dict(model=name, run=str(run), checkpoint_sha256=lock['checkpoint_sha256'],
                same_data=True, retrained=False, newly_matched_control=False,
                role='main historical reference' if name == 'R89B' else 'secondary historical reference'))
    return result


def load_targets(config, historical_config, historical_lock):
    targets, identity = helpers.load_targets(config, historical_config, historical_lock)
    if len(targets) != 36:
        raise ValueError('R95 must retain the fixed R84 target36 without reselection')
    return targets, dict(identity, sha256=file_hash(Path(identity['path'])), target_count=36)


def val_screen(config, lock, val, refs):
    rows = []
    for name, ref in refs.items():
        current, old = val['macro'], ref['val']['macro']
        keys = ('macro_top1', 'macro_mean_cost', 'macro_actual_regret_pct')
        if not all(math.isfinite(float(r[key])) for r in (current, old) for key in keys):
            raise ValueError('R95 historical VAL diagnostics require finite saved metrics')
        top1 = 100. * (current[keys[0]] - old[keys[0]])
        cost, regret = (current[key] - old[key] for key in keys[1:])
        rows.append(dict(model='R95/' + config['arm'], baseline=name, split='VAL',
            reference_role='main' if name == 'R89B' else 'secondary', top1_delta_pp=top1,
            mean_cost_delta=cost, actual_regret_delta_pp=regret,
            pass_reference=top1 > 0 and cost <= 0 and regret <= 0,
            checkpoint_sha256=lock['checkpoint_sha256'], checkpoint_reselection=False,
            newly_matched_control=False, macro_weighting='equal weight across all18 problems'))
    return [dict(row, joint_val_pass=all(r['pass_reference'] for r in rows)) for row in rows]


def diagnostic_tables(run, config, lock, val, test, args):
    packed, methods, summaries, results, paths = selected_predictions(run, config, lock, val, test)
    refs = references(config, args)
    targets, identity = load_targets(config, refs['R84A']['config'], refs['R84A']['lock'])
    pairs, recovery, totals = {}, {}, {}
    for name, ref in refs.items():
        old = selected_predictions(ref['run'], ref['config'], ref['lock'], ref['val'], ref['test'])[0]
        pairs[name], recovery[name], totals[name] = helpers.comparison_tables(
            old, packed, targets, identity, 'R95/' + config['arm'], name)
    return dict(predictions=packed, methods=methods, summaries=summaries, results=results,
        paths=paths, pairs=pairs, recovery=recovery, recovery_summary=totals,
        target_identity=identity, historical={name: ref['provenance'] for name, ref in refs.items()},
        val_screen=val_screen(config, lock, val, refs))


def sampling_diagnostics(run, config, lock):
    """Consume only own-TRAIN scalar proposal receipts, never proposal arrays."""
    rows, sources = [], {}
    metrics = ('q_min', 'q_max', 'max_inverse_weight', 'proposal_effective_population',
               'gradient_norm_mean', 'gradient_norm_max', 'deterministic_last_layer_second_moment_ratio')
    for step in range(7200, MAX_UPDATES + 1, 7200):
        path = run / 'proposals' / f'update{step:07d}.json'
        record = read_json(path)
        source = run / 'evaluation/train' / f'update{step:07d}' / 'identity.json'
        identity = read_json(source)
        if (record.get('experiment') != 'R95' or record.get('arm') != config['arm']
                or record.get('complete') is not True or record.get('split') != 'train'
                or record.get('successful_updates') != step or record.get('rows_per_problem') != 81000
                or record.get('state_hash') != lock['state_hash'] or record.get('sampling') != SAMPLING
                or set(record.get('per_problem', {})) != set(PROBLEMS)
                or not record.get('proposal_sha256') or not record.get('source_prediction_sha256')
                or record.get('source_identity_sha256') != file_hash(source)
                or not record.get('model_state_sha256')
                or identity.get('experiment') != 'R95' or identity.get('arm') != config['arm']
                or identity.get('split') != 'train' or identity.get('successful_updates') != step
                or identity.get('state_hash') != lock['state_hash']
                or identity.get('model_state_sha256') != record['model_state_sha256']):
            raise ValueError('R95 diagnostics require each own full-TRAIN7200 proposal receipt')
        sources[str(path)] = file_hash(path)
        for problem in PROBLEMS:
            values = record['per_problem'][problem]
            if (values.get('n') != 81000 or any(isinstance(values.get(k), bool)
                    or not isinstance(values.get(k), (int, float))
                    or not math.isfinite(values[k]) or values[k] < 0 for k in metrics)):
                raise ValueError('Invalid R95 scalar proposal diagnostics: ' + problem)
            if (values['q_min'] < .5 / 81000 - 1e-15
                    or not values['q_min'] <= values['q_max'] <= 1
                    or not 0 < values['max_inverse_weight'] <= 2 + 1e-10
                    or not 0 < values['proposal_effective_population'] <= 81000 + 1e-6
                    or not math.isclose(values['max_inverse_weight'],
                        1 / (81000 * values['q_min']), rel_tol=1e-8)):
                raise ValueError('R95 proposal violates the locked half-uniform correction bounds')
            rows.append(dict(model='R95/' + config['arm'], arm=config['arm'], problem=problem,
                successful_updates=step, n=81000,
                applied_to_draws=config['arm'] == 'B' and step < MAX_UPDATES,
                proposal_kind='adaptive' if config['arm'] == 'B' else 'diagnostic_only_A_draws_uniform',
                **{k: values[k] for k in metrics}))
    history = read_json(run / 'history.json')
    optimization = []
    keys = ('sampling_mean_weight', 'sampling_min_weight', 'sampling_max_weight', 'sampling_unique_fraction')
    for item in history:
        values = item.get('optimization', {})
        if any(not isinstance(values.get(k), (int, float)) or isinstance(values.get(k), bool)
                or not math.isfinite(values[k]) for k in keys):
            raise ValueError('R95 history must retain finite sampled-weight/coverage diagnostics')
        optimization.append(dict(model='R95/' + config['arm'],
            successful_updates=item['successful_updates'], **{k: values[k] for k in keys}))
    if [row['successful_updates'] for row in optimization] != list(range(1440, MAX_UPDATES + 1, 1440)):
        raise ValueError('R95 sampling diagnostics require all40 fixed evaluation intervals')
    return dict(complete=True, proposal_rows=rows, optimization_rows=optimization,
        source_sha256=sources, sampling=dict(SAMPLING), proxy_boundary=PROXY_BOUNDARY,
        applied_to_draws=config['arm'] == 'B', raw_arrays_read=False,
        training=False, inference=False, causal_claim=False)


def sampling_pdf(output, config, lock, diagnostics):
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    path = output / 'sampling_diagnostics.pdf'
    temporary = path.with_name(f'.{path.stem}.{os.getpid()}.pdf')
    try:
        with PdfPages(temporary) as pdf:
            fig, axes = plt.subplots(2, 2, figsize=(14, 9))
            fig.suptitle('R95/' + config['arm'] + ' | TRAIN-only sampling diagnostics')
            for axis, key, title in ((axes[0, 0], 'proposal_effective_population', 'Proposal effective population'),
                    (axes[0, 1], 'deterministic_last_layer_second_moment_ratio', 'Proxy second moment / uniform (not network variance)')):
                for problem in PROBLEMS:
                    points = [(r['successful_updates'], r[key]) for r in diagnostics['proposal_rows'] if r['problem'] == problem]
                    axis.plot(*zip(*points), label=problem, linewidth=1)
                axis.set_title(title)
            for axis, keys, title in ((axes[1, 0], ('sampling_mean_weight', 'sampling_min_weight', 'sampling_max_weight'),
                    'Interval means of sampled weight mean/min/max'),
                    (axes[1, 1], ('sampling_unique_fraction',), 'Mean unique fraction per logical batch')):
                for key in keys:
                    points = [(r['successful_updates'], r[key]) for r in diagnostics['optimization_rows']]
                    axis.plot(*zip(*points), label=key)
                axis.set_title(title)
                axis.legend(fontsize=8)
            for axis in axes.flat:
                axis.axvline(7200, color='#c45358', linestyle='--')
                axis.set_xlabel('Successful logical1024 updates')
                axis.grid(alpha=.2)
            fig.text(.05, .035, 'A: adaptive proposals are diagnostic only, draws remain uniform. Final57600 proposal is not used for training.', fontsize=9)
            fig.text(.05, .015, 'Final-affine eval proxy only; R-Drop/KL=0 in eval. No unbiased AdamW, guaranteed variance reduction or causal claim.', fontsize=9)
            fig.tight_layout(rect=(0, .06, 1, .95))
            pdf.savefig(fig)
            plt.close(fig)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return str(path)


def diagnostics_pdf(output, config, lock, tables):
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    helper = common.reporting_helpers()
    helper.setup_style()
    path = output / 'target_recovery_and_train.pdf'
    temporary = path.with_name(f'.{path.stem}.{os.getpid()}.pdf')
    try:
        with PdfPages(temporary) as pdf:
            for page, ref in enumerate(('R89B', 'R84A'), 1):
                fig = common._page(helper, 'R95/' + config['arm'] + ' | Fixed target36 vs ' + ref,
                    'Same unique VAL checkpoint; native/FP64 winner; no reselection.', 'R95/' + config['arm'], lock, page)
                helper.add_table(fig,
                    ['Split', 'Winners', 'Correct', 'Rescued', 'Lost', 'New false picks', 'Recall (%)'],
                    [[r['split'], str(r['winner_count']), str(r['correct_count']), str(r['rescued']),
                      str(r['lost']), str(r['new_false_pick_count']),
                      '-' if r['recall_pct'] is None else f"{r['recall_pct']:.4f}"]
                     for r in tables['recovery_summary'][ref]], [.05, .62, .9, .2], font=10)
                helper.add_table(fig,
                    ['Split', 'Top1 delta pp', 'Corrected', 'Harmed', 'Cost delta', 'Regret delta pp'],
                    [[r['split'], f"{r['top1_delta_pp']:.6f}", str(r['corrected']), str(r['harmed']),
                      f"{r['mean_cost_delta']:.9f}", f"{r['actual_regret_delta_pp']:.6f}"]
                     for r in tables['pairs'][ref] if r['problem'] == 'ALL'], [.05, .32, .9, .2], font=10)
                fig.text(.05, .24, 'Historical R89B main / R84A secondary; not newly matched original-loss controls.', fontsize=9)
                fig.text(.05, .18, LOSS_LABEL, fontsize=9)
                fig.text(.05, .12, 'Same natural first7200; A uniform replacement / B whole-row inverse-corrected adaptive q. No causal claim.', fontsize=9)
                fig.text(.05, .06, 'A:3090_0; B:4090_0. Not bitwise hardware-matched. Zero-count movement alone is not success.', fontsize=9)
                pdf.savefig(fig)
                plt.close(fig)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return str(path)


def learning_curves(run, config, lock):
    import matplotlib.pyplot as plt
    history = read_json(run / 'history.json')
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle('R95/' + config['arm'] + ' | Original R77 loss and two-pass R-Drop', fontsize=14)
    for key in ('loss', 'ce', 'cmp', 'risk', 'kl'):
        points = [(r['successful_updates'], r['optimization'][key]) for r in history
                  if key in r.get('optimization', {})]
        if points:
            axes[0, 0].plot(*zip(*points), label=key + ' (inverse-corrected TRAIN)')
    axes[0, 0].set_title('Original CE/comparison/risk/R-Drop; not plain CE')
    axes[0, 0].legend(fontsize=8)
    for axis, key, factor, title in (
            (axes[0, 1], 'macro_ce', 1., 'Evaluation CE: original maximin diagnostic'),
            (axes[0, 2], 'macro_top1', 100., 'Native Top1 (%)'),
            (axes[1, 0], 'macro_mean_cost', 1., 'Task-equal raw mean cost (descriptive)'),
            (axes[1, 1], 'macro_actual_regret_pct', 1., 'Actual regret (%)'),
            (axes[1, 2], 'macro_vs_sbs_pct', 1., 'vs fixed TRAIN SBS (%)')):
        for split in ('train', 'val'):
            points = [(r['successful_updates'], r[split]['macro'][key] * factor)
                      for r in history if r.get(split) is not None and key in r[split]['macro']]
            if points:
                axis.plot(*zip(*points), label=split)
        axis.set_title(title)
        if axis.lines:
            axis.legend()
    for axis in axes.flat:
        axis.axvline(lock['selected_update'], color='#c45358', linestyle='--', linewidth=1)
        axis.set_xlabel('Successful logical-batch1024 updates')
        axis.grid(alpha=.2)
    fig.tight_layout(rect=(0, 0, 1, .95))
    for suffix in ('pdf', 'png'):
        fig.savefig(run / 'full_model_statistics' / ('learning_curves.' + suffix))
    plt.close(fig)


def report(args):
    require_cpu(args)
    run, config, lock, test, val = locked_inputs(args.root, args.arm)
    with (run / 'postprocess.lock').open('a') as claim:
        fcntl.flock(claim, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            write_json(run / 'postprocess_status.json', dict(experiment='R95', arm=args.arm,
                phase='loading_saved_predictions', updated_utc=utc()))
            lib = read_json(run / 'lib_stage_complete.json')
            if lib.get('complete') is not True or lib.get('checkpoint_sha256') != lock['checkpoint_sha256']:
                raise ValueError('R95 CPU report requires completed same-checkpoint GPU LIB')
            reference = previous.validate_nss_reference(config, args.nss_root)
            predictions, metrics, methods = shared.load_saved_splits(run, config, lock, test, val)
            shared.saved_libs(run, config, lock, test, predictions, metrics, methods)
            comparisons, alignment = shared.nss_comparisons(config, lock, predictions, args.nss_root)
            tables = diagnostic_tables(run, config, lock, val, test, args)
            sampling = sampling_diagnostics(run, config, lock)
            # The shared renderer reads this field only for its printed caption.
            # Keep the actual locked objective/config untouched for all provenance.
            display = dict(config, objective=dict(config['objective'],
                classification=LOSS_LABEL))
            result = shared.build_reports(run, display, lock, test, val, predictions, metrics, methods, comparisons, alignment)
            learning_curves(run, config, lock)
            output = run / 'full_model_statistics'
            for name, rows in (
                    ('method_selection_train_val_test.csv', tables['methods']),
                    ('zero_selection_summary.csv', tables['summaries']),
                    ('train_metrics_by_problem.csv', helpers.train_metric_rows(config, lock, tables)),
                    ('val_empirical_screen.csv', tables['val_screen']),
                    ('sampling_diagnostics.csv', sampling['proposal_rows']),
                    ('sampling_optimization.csv', sampling['optimization_rows'])):
                common.write_csv(output / name, rows)
            for ref in tables['historical']:
                for stem, data in (('paired_changes', tables['pairs']), ('target_recovery', tables['recovery']),
                                   ('target_recovery_summary', tables['recovery_summary'])):
                    common.write_csv(output / (stem + '_vs_' + ref + '.csv'), data[ref])
            for stem, data in (('target_recovery', tables['recovery']), ('target_recovery_summary', tables['recovery_summary'])):
                common.write_csv(output / (stem + '.csv'), data['R84A'])
            result.update(target_recovery_pdf=diagnostics_pdf(output, config, lock, tables),
                sampling_diagnostics_pdf=sampling_pdf(output, config, lock, sampling),
                historical_reference=tables['historical'], nss_reference=reference,
                joint_val_pass=all(row['joint_val_pass'] for row in tables['val_screen']))
            provenance = read_json(output / 'protocol.json')
            provenance.update(target_identity=tables['target_identity'], historical_reference=tables['historical'],
                objective=config['objective'], evidence_boundary=REFERENCE_BOUNDARY,
                train_ce_probability_link='maximin', evaluation_ce_probability_link='maximin',
                one_forward=False, rdrop=True, weight_decay_retained=True,
                sampling_diagnostics={key: value for key, value in sampling.items()
                    if key not in ('proposal_rows', 'optimization_rows')},
                val_empirical_screen=tables['val_screen'])
            write_json(output / 'protocol.json', provenance)
            notes = output / 'RESULTS.md'
            with notes.open('a', encoding='utf-8') as stream:
                stream.write('\n' + LOSS_LABEL + '\n\n' + REFERENCE_BOUNDARY + '\n')
            write_json(run / 'postprocess_status.json', dict(experiment='R95', arm=args.arm,
                phase='complete', exit_status=0, completed_utc=utc(), **result))
            return result
        except BaseException as error:
            write_json(run / 'postprocess_status.json', dict(experiment='R95', arm=args.arm,
                phase='postprocess_failed', exit_status=1, updated_utc=utc(), error=str(error),
                traceback=traceback.format_exc(), automatic_retry=False))
            raise


def compare(args):
    require_cpu(args)
    selected, locks, configs, diagnostics, metrics, screens = {}, {}, {}, {}, [], []
    for arm in ARMS:
        run, config, lock, test, val = locked_inputs(args.root, arm)
        completed_arm_report(run, lock)
        selected[arm] = selected_predictions(run, config, lock, val, test)[0]
        locks[arm], configs[arm] = lock, config
        diagnostics[arm] = sampling_diagnostics(run, config, lock)
        refs = references(config, args)
        screens.extend(val_screen(config, lock, val, refs))
        for split, result in (('VAL', val), ('TEST', test)):
            metrics.append(dict(model='R95/' + arm, split=split, dropout=DROPOUT[arm],
                sampler='uniform_replacement' if arm == 'A' else 'inverse_corrected_adaptive',
                selected_update=lock['selected_update'], checkpoint_sha256=lock['checkpoint_sha256'],
                **{key: result['macro'][key] for key in ('macro_top1', 'macro_mean_cost',
                   'macro_sbs_cost', 'macro_oracle_cost', 'macro_vs_sbs_pct',
                   'macro_vs_oracle_pct', 'macro_actual_regret_pct')}))
    matching = validate_matched_configs(configs['A'], configs['B'])
    targets, identity = load_targets(configs['B'], refs['R84A']['config'], refs['R84A']['lock'])
    pairs, recovery, totals = helpers.comparison_tables(selected['A'], selected['B'], targets,
        identity, 'R95/B', 'R95/A')
    output = args.root / 'comparison_artifacts'
    output.mkdir(parents=True, exist_ok=True)
    for name, rows in (('comparison.csv', metrics), ('paired_changes_B_vs_A.csv', pairs),
                       ('target_recovery_B_vs_A.csv', recovery), ('target_recovery_summary_B_vs_A.csv', totals),
                       ('val_empirical_screen.csv', screens),
                       ('sampling_diagnostics.csv', [row for arm in ARMS for row in diagnostics[arm]['proposal_rows']]),
                       ('sampling_optimization.csv', [row for arm in ARMS for row in diagnostics[arm]['optimization_rows']])):
        common.write_csv(output / name, rows)
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    helper = common.reporting_helpers()
    helper.setup_style()
    path = output / 'COMPARISON.pdf'
    temporary = path.with_name(f'.{path.stem}.{os.getpid()}.pdf')
    try:
        with PdfPages(temporary) as pdf:
            fig = common._page(helper, 'R95 | Adaptive corrected sampling B versus uniform A',
                'Original R88B/R77 full18 model and CE/comparison/risk/two-pass R-Drop retained.', 'R95 B vs A', locks['B'], 1)
            helper.add_table(fig, ['Split', 'Top1 delta pp', 'Corrected', 'Harmed', 'Cost delta', 'Regret delta pp'],
                [[r['split'], f"{r['top1_delta_pp']:.6f}", str(r['corrected']), str(r['harmed']),
                  f"{r['mean_cost_delta']:.9f}", f"{r['actual_regret_delta_pp']:.6f}"]
                 for r in pairs if r['problem'] == 'ALL'], [.05, .61, .9, .2], font=10)
            helper.add_table(fig, ['Arm / VAL ref', 'Top1 delta pp', 'Cost delta', 'Regret delta pp', 'Pass'],
                [[r['model'] + ' / ' + r['baseline'], f"{r['top1_delta_pp']:.6f}",
                  f"{r['mean_cost_delta']:.9f}", f"{r['actual_regret_delta_pp']:.6f}", str(r['pass_reference'])]
                 for r in screens], [.05, .32, .9, .22], font=10)
            fig.text(.05, .25, LOSS_LABEL, fontsize=9)
            fig.text(.05, .19, 'A:3090_0; B:4090_0. Not bitwise hardware-matched. No causal claim or guaranteed variance reduction.', fontsize=9)
            fig.text(.05, .13, 'Same natural first7200; then A uniform IID / B own-TRAIN adaptive q, .5 uniform mix, whole-row 1/(N*q).', fontsize=9)
            fig.text(.05, .07, 'R89B main / R84A secondary historical; no TEST/LIB checkpoint or inference-rule selection.', fontsize=9)
            pdf.savefig(fig)
            plt.close(fig)
            fig = common._page(helper, 'R95 | Fixed target36 genuine recovery and false picks',
                'B versus A on the unchanged historical36 targets; native FP64 winners.', 'R95 B vs A', locks['B'], 2)
            helper.add_table(fig, ['Split', 'Winners', 'Correct', 'Rescued', 'Lost', 'New false picks', 'Recall (%)'],
                [[r['split'], str(r['winner_count']), str(r['correct_count']), str(r['rescued']),
                  str(r['lost']), str(r['new_false_pick_count']),
                  '-' if r['recall_pct'] is None else f"{r['recall_pct']:.4f}"] for r in totals],
                [.05, .48, .9, .25], font=10)
            fig.text(.05, .32, 'Detailed per-task recovery, false-pick costs and regret remain in the paired/target CSVs.', fontsize=9)
            fig.text(.05, .24, 'Changing zero-selection counts alone is not success. Full18 cost, regret and native Top1 all matter.', fontsize=9)
            fig.text(.05, .16, 'No causal claim; the independent arms run on different GPU types.', fontsize=9)
            pdf.savefig(fig)
            plt.close(fig)
            fig = common._page(helper, 'R95 | Own-TRAIN sampling proposal diagnostics',
                'Summaries only; never full-network/stochastic gradient variance or unbiased optimizer updates.', 'R95 B vs A', locks['B'], 3)
            cells = []
            for arm in ARMS:
                for step in range(7200, MAX_UPDATES + 1, 7200):
                    rows = [r for r in diagnostics[arm]['proposal_rows'] if r['successful_updates'] == step]
                    cells.append([arm, str(step), str(rows[0]['applied_to_draws']),
                        f"{sum(r['proposal_effective_population'] for r in rows) / len(rows):.2f}",
                        f"{max(r['max_inverse_weight'] for r in rows):.6f}",
                        f"{sum(r['deterministic_last_layer_second_moment_ratio'] for r in rows) / len(rows):.6f}"])
            helper.add_table(fig, ['Arm', 'Update', 'Used for draws', 'Task-equal proposal ESS', 'Max inverse weight', 'Proxy second-moment ratio'],
                cells, [.05, .30, .9, .54], font=9)
            fig.text(.05, .22, 'A proposal summaries are diagnostic only; A draws stay uniform. Final57600 proposal has no next interval.', fontsize=9)
            fig.text(.05, .15, 'Final-affine CE+comparison+risk eval proxy; KL=0 in eval, not full-parameter norm.', fontsize=9)
            fig.text(.05, .08, 'Whole-row correction, arithmetic mean, no self-normalization. No guaranteed variance reduction or causal claim.', fontsize=9)
            pdf.savefig(fig)
            plt.close(fig)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    result = dict(experiment='R95', complete=True, exit_status=0, checkpoints=locks,
        comparison_pdf=str(path), evidence_boundary=REFERENCE_BOUNDARY,
        matched_difference=matching['matched_difference'], target_identity=identity,
        matched_controls=matching, sampling=dict(SAMPLING),
        sampling_diagnostics={arm: {key: value for key, value in diagnostics[arm].items()
            if key not in ('proposal_rows', 'optimization_rows')} for arm in ARMS},
        hardware=dict(A='3090_0', B='4090_0'), bitwise_hardware_matched=False,
        causal_claim=False, guaranteed_variance_reduction=False,
        training=False, inference=False, checkpoint_reselection=False)
    write_json(output / 'protocol.json', result)
    lines = ['# R95: Adaptive Example Sampling', '', REFERENCE_BOUNDARY, '', LOSS_LABEL,
        'Original maximin inference (argmax order equivalent to raw utility), unchanged ties.', '',
        '| Model | Split | Top1 (%) | mean_cost | vs_SBS (%) | vs_Oracle (%) | actual regret (%) |',
        '| --- | --- | --- | --- | --- | --- | --- |']
    for row in metrics:
        lines.append(f"| {row['model']} | {row['split']} | {100 * row['macro_top1']:.6f} | "
            f"{row['macro_mean_cost']:.9f} | {row['macro_vs_sbs_pct']:.6f} | "
            f"{row['macro_vs_oracle_pct']:.6f} | {row['macro_actual_regret_pct']:.6f} |")
    (args.root / 'RESULTS.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    shutil.copyfile(path, args.root / 'RESULTS.pdf')
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
        value = read_json(path) if path.exists() else dict(experiment='R95', phase='not_started')
        print(json.dumps(value, indent=2))
        return value
    return lib_stage(args) if args.command == 'lib' else report(args)


if __name__ == '__main__':
    main()

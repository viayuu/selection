"""Post-lock, saved-only fixed36 supplement to the frozen R116 finisher.

Only a private source-loaded finisher is rebound. No selector imports, forward
passes, target selection, checkpoint changes, retries, or automatic new arms.
The original hard receipts gate all current VAL/TEST reads and publication.
"""

import argparse
from pathlib import Path
import sys
import types
import uuid

import numpy as np

from . import r112_finish as common
from . import r113_finish as recovery


V4 = Path(__file__).resolve().parent
ROOT = V4 / 'runs/R116_subset_elimination'
FROZEN_FINISHER = V4 / 'r116_finish.py'
TARGET_LIST = V4 / 'runs/R84_new90k_attention/comparison_artifacts/target_list.json'
BASELINE = V4 / 'runs/R84_new90k_attention/A_baseline'
BASELINE_SHA256 = '23fb9a5cf621045138fcdcfe5e6613a1bfd061c8d4dfeca108db3548dbf8031f'
FONT = Path('/public/home/shiys/.cache/selector_fonts/20261026/NotoSansCJK-Regular.ttc')
EXTRA_OUTPUTS = ('target_recovery.csv', 'target_recovery_summary.csv', 'target_status.json', 'ZERO_SELECTION.pdf')
EXTRA_CODE = ('r116_mechanism_finish.py', 'test_r116_mechanism_finish.py')
LOADED_SHA256 = common.digest(Path(__file__))
COUNTS = ('winner_count', 'selected_count', 'baseline_selected_count', 'baseline_true_tp',
    'current_true_tp', 'baseline_false_pick_count', 'false_pick_count', 'new_false_pick_count',
    'removed_false_pick_count', 'corrected', 'harmed', 'net_correct', 'both_wrong_count',
    'methods_in_group', 'baseline_zero_selected_methods', 'current_zero_selected_methods')
CONTRIBUTIONS = ('baseline_mean_cost_contribution', 'current_mean_cost_contribution',
    'mean_cost_delta_contribution', 'baseline_actual_regret_contribution_pp',
    'current_actual_regret_contribution_pp', 'actual_regret_delta_contribution_pp',
    'both_wrong_cost_delta_contribution', 'both_wrong_regret_delta_contribution_pp',
    'new_false_pick_cost_delta_contribution', 'new_false_pick_regret_delta_contribution_pp',
    'new_false_pick_current_regret_contribution_pp', 'true_rescue_cost_delta_contribution',
    'lost_true_pick_cost_delta_contribution')


def _rates(row):
    def pct(numerator, denominator):
        return 100. * numerator / denominator if denominator else None

    row.update(baseline_recall_pct=pct(row['baseline_true_tp'], row['winner_count']),
        current_recall_pct=pct(row['current_true_tp'], row['winner_count']),
        baseline_selection_precision_pct=pct(row['baseline_true_tp'], row['baseline_selected_count']),
        current_selection_precision_pct=pct(row['current_true_tp'], row['selected_count']),
        true_rescues=row['corrected'], lost_true_picks=row['harmed'],
        zero_count_reduction=row['baseline_zero_selected_methods'] - row['current_zero_selected_methods'])
    return row


def _augment(row, old, new, slots):
    costs, winner, before, after = new['costs'], new['winner'], old['pred'], new['pred']
    n, indices = len(winner), np.arange(len(winner))
    truth, was, chosen = (np.isin(values, slots) for values in (winner, before, after))
    correct_before, correct_after = before == winner, after == winner
    baseline_cost, current_cost, oracle = costs[indices, before], costs[indices, after], costs.min(1)
    delta = current_cost - baseline_cost
    old_regret, new_regret = (baseline_cost - oracle) / oracle, (current_cost - oracle) / oracle
    rescued, lost = truth & ~correct_before & correct_after, truth & correct_before & ~correct_after
    both_wrong = truth & ~correct_before & ~correct_after
    incoming = chosen & ~correct_after & (after != before)
    removed = was & ~correct_before & (after != before)

    def contribution(values, mask):
        return float(values[mask].sum() / n)

    row.update(n=n, winner_count=int(truth.sum()), selected_count=int(chosen.sum()),
        baseline_selected_count=int(was.sum()), baseline_true_tp=int((truth & correct_before).sum()),
        current_true_tp=int((truth & correct_after).sum()),
        baseline_false_pick_count=int((was & ~correct_before).sum()),
        false_pick_count=int((chosen & ~correct_after).sum()),
        new_false_pick_count=int(incoming.sum()), removed_false_pick_count=int(removed.sum()),
        corrected=int(rescued.sum()), harmed=int(lost.sum()), net_correct=int(rescued.sum() - lost.sum()),
        both_wrong_count=int(both_wrong.sum()), methods_in_group=len(slots),
        baseline_zero_selected_methods=sum(not np.any(before == slot) for slot in slots),
        current_zero_selected_methods=sum(not np.any(after == slot) for slot in slots),
        baseline_mean_cost_contribution=contribution(baseline_cost, truth),
        current_mean_cost_contribution=contribution(current_cost, truth),
        mean_cost_delta_contribution=contribution(delta, truth),
        baseline_actual_regret_contribution_pp=100 * contribution(old_regret, truth),
        current_actual_regret_contribution_pp=100 * contribution(new_regret, truth),
        actual_regret_delta_contribution_pp=100 * contribution(delta / oracle, truth),
        both_wrong_cost_delta_contribution=contribution(delta, both_wrong),
        both_wrong_regret_delta_contribution_pp=100 * contribution(delta / oracle, both_wrong),
        new_false_pick_cost_delta_contribution=contribution(delta, incoming),
        new_false_pick_regret_delta_contribution_pp=100 * contribution(delta / oracle, incoming),
        new_false_pick_current_regret_contribution_pp=100 * contribution(new_regret, incoming),
        true_rescue_cost_delta_contribution=contribution(delta, rescued),
        lost_true_pick_cost_delta_contribution=contribution(delta, lost),
        aggregation='cost/regret: full problem denominator, native-winner partition; false picks: predicted-method group',
        rate_aggregation='group TP/winners and TP/selections; undefined denominators are blank')
    # The old helper's similarly named regret column is absolute current regret, not a delta.
    row['new_false_pick_regret_contribution_pp'] = row['new_false_pick_current_regret_contribution_pp']
    return _rates(row)


def recovery_tables(old, new, targets, comparison, split):
    """Augment the frozen helper; only FIXED36/OUTSIDE36 partitions are additive."""
    if len(targets) != 36 or len(set(targets)) != 36:
        raise ValueError('Use exactly the frozen 36 problem/method pairs, never reselect targets')
    rows, _ = recovery.target_recovery(old, new, targets, comparison, split)
    details, summary = [], []
    for problem in common.PROBLEMS:
        pool = new[problem]['pool']
        fixed = [pool.index(method) for p, method in targets if p == problem]
        outside = [i for i in range(len(pool)) if i not in fixed]
        partitions = []
        for row in (r for r in rows if r['problem'] == problem):
            slots = ([pool.index(row['solver'])] if row['scope'] == 'fixed36_method'
                     else fixed if row['solver'] == 'FIXED36' else outside)
            _augment(row, old[problem], new[problem], slots)
            (details if row['scope'] == 'fixed36_method' else partitions).append(row)
        full = _augment(dict(comparison=comparison, split=split, problem=problem,
            solver='FULL18', scope='full_problem'), old[problem], new[problem], list(range(len(pool))))
        for field in (*COUNTS, *CONTRIBUTIONS):
            if not np.isclose(sum(r[field] for r in partitions), full[field], rtol=1e-11, atol=1e-12):
                raise ValueError('Fixed/outside partition does not reconstruct full18: ' + problem + '/' + field)
        summary.extend([*partitions, full])
    for name in ('FIXED36', 'OUTSIDE36', 'FULL18'):
        group = [r for r in summary if r['solver'] == name]
        overall = common.macro(group, CONTRIBUTIONS, COUNTS)
        overall['aggregation'] = 'ALL cost/regret: equal mean over all18 problems; raw costs descriptive only'
        overall['rate_aggregation'] = 'ALL rates from pooled group counts; cost/regret are problem-equal'
        summary.append(_rates(overall))
    return details, summary


def _baseline(current_config, inputs):
    model = common.load_run(BASELINE, inputs, expected=('R84', 'A_baseline'))
    common.same_data(current_config, model['config'])
    identity = common.read(TARGET_LIST)['identity']
    if (identity['baseline_checkpoint_sha256'] != BASELINE_SHA256
            or model['lock']['checkpoint_sha256'] != BASELINE_SHA256
            or model['lock']['state_hash'] != identity['baseline_state_hash']):
        raise ValueError('Fixed36 must bind the specified frozen R84/A_baseline checkpoint')
    return model


def _fmt(value, digits=3):
    return '-' if value is None else format(value, '.' + str(digits) + 'f')


def zero_selection_pdf(root, details, summary, status, gate):
    """Small vector PDF, using an existing font and subset embedding only."""
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt, font_manager
    from matplotlib.backends.backend_pdf import PdfPages

    font_manager.fontManager.addfont(str(FONT))
    family = font_manager.FontProperties(fname=str(FONT)).get_name()
    temporary = Path(root) / '.ZERO_SELECTION.mechanism.pdf'
    note = ('\u9009\u62e9\u6b21\u6570\u589e\u52a0\u6216\u96f6\u9009\u62e9\u6570\u51cf\u5c11\u4e0d\u7b49\u4e8e\u6210\u529f; '
            '\u4e3b\u5224\u636e\u4ecd\u662f\u539f\u9884\u6ce8\u518c VAL gate\u3002')
    with matplotlib.rc_context({'font.family': family, 'pdf.fonttype': 42, 'axes.unicode_minus': False}):
        try:
            with PdfPages(temporary) as pdf:
                def page(title, headers, cells, explanation):
                    fig, ax = plt.subplots(figsize=(17., 10.5))
                    ax.axis('off')
                    fig.suptitle(title, fontsize=15)
                    fig.text(.035, .91, explanation, fontsize=8)
                    table = ax.table(cellText=cells, colLabels=headers, bbox=[0, .05, 1, .88], cellLoc='center')
                    table.auto_set_font_size(False)
                    table.set_fontsize(6.5 if len(cells) > 25 else 8)
                    fig.text(.035, .035, note, fontsize=9)
                    fig.text(.035, .018, 'Native FP64 ind; delta=current-baseline; no forward passes or model/target reselection.', fontsize=8)
                    pdf.savefig(fig)
                    plt.close(fig)

                page('R116 | Original Decision Gate / \u539f\u59cb\u5224\u636e', ['Original field', 'Actual result'], [
                    ['scientific_gate_passed', str(gate['scientific_gate_passed'])],
                    ['deployment_p95_gate_passed', str(gate['deployment_p95_gate_passed'])],
                    ['recommend_deployment', str(gate['recommend_deployment'])],
                    ['VAL B-A Top1 (pp)', _fmt(gate['top1_delta_pp'], 6)],
                    ['VAL B-A actual regret (pp)', _fmt(gate['actual_regret_delta_pp'], 6)],
                    ['VAL B-A vsSBS (pp)', _fmt(gate['vs_sbs_delta_pp'], 6)],
                    ['Worst task regret harm (pp)', _fmt(gate['worst_regret_harm_pp'], 6)],
                    ['Original thresholds', str(gate['thresholds'])],
                    ['Measured complete-DP p95 / point baseline', str(gate['max_task_p95_ratio'])],
                    ['Fixed target source', 'historical R84/A_baseline VAL; 36 problem/method pairs'],
                    ['Frozen baseline checkpoint', BASELINE_SHA256],
                    ['Supplement generator SHA256', status['supplement_generator_sha256']],
                    ['Scope', 'post-lock reporting only; no checkpoint, model or protocol change']],
                    'Actual decision_gate.json from frozen r116_finish, not a new success criterion. TEST never chooses targets or locks.')
                comparisons = list(dict.fromkeys(r['comparison'] for r in details))
                for comparison in comparisons:
                    rows = [r for r in details if r['comparison'] == comparison and r['split'] == 'TEST']
                    page('Fixed36 TEST | ' + comparison,
                        ['Task', 'Method', 'Winners', 'TP old', 'TP new', 'Recall old%', 'Recall new%',
                         'Prec old%', 'Prec new%', 'New FP', 'Rescue', 'Lost', 'dCost*', 'dRegret pp*'],
                        [[r['problem'], r['solver'], r['winner_count'], r['baseline_true_tp'], r['current_true_tp'],
                          _fmt(r['baseline_recall_pct']), _fmt(r['current_recall_pct']),
                          _fmt(r['baseline_selection_precision_pct']), _fmt(r['current_selection_precision_pct']),
                          r['new_false_pick_count'], r['true_rescues'], r['lost_true_picks'],
                          format(r['mean_cost_delta_contribution'], '.5g'),
                          format(r['actual_regret_delta_contribution_pp'], '.5g')] for r in rows],
                        '*Cost/regret charged to native-winner blocks using full task denominator; not a conditional mean. - means undefined denominator.')
                for split in ('VAL', 'TEST'):
                    rows = [r for r in summary if r['problem'] == 'ALL' and r['split'] == split]
                    page(split + ' | Full18 Fixed36 / Outside36 Summary',
                        ['Comparison', 'Winner block', 'TP old', 'TP new', 'Recall new%', 'Prec new%',
                         'New FP', 'Rescue', 'Lost', 'dCost', 'dRegret pp', 'Wrong dCost', 'Wrong dReg pp'],
                        [[r['comparison'], r['solver'], r['baseline_true_tp'], r['current_true_tp'],
                          _fmt(r['current_recall_pct']), _fmt(r['current_selection_precision_pct']),
                          r['new_false_pick_count'], r['true_rescues'], r['lost_true_picks'],
                          format(r['mean_cost_delta_contribution'], '.5g'),
                          format(r['actual_regret_delta_contribution_pp'], '.5g'),
                          format(r['both_wrong_cost_delta_contribution'], '.5g'),
                          format(r['both_wrong_regret_delta_contribution_pp'], '.5g')] for r in rows],
                        'FIXED36 + OUTSIDE36 = FULL18 cost/regret. ALL gives every task equal weight; raw cost descriptive. Historical references are not matched ablations.')
            temporary.replace(Path(root) / 'ZERO_SELECTION.pdf')
        finally:
            temporary.unlink(missing_ok=True)


def _supplement(root, models, saved, inputs, gate):
    current = models[str((root / 'A_static').resolve())]
    targets, status = recovery.target_groups(TARGET_LIST, current['config'], inputs)
    target_record = common.read(TARGET_LIST)
    baseline = _baseline(current['config'], inputs)
    baseline_data = {split: common.saved_split(baseline, split, inputs) for split in ('VAL', 'TEST')}
    val_path = Path(baseline['final']['val']['prediction_archive'])
    if common.digest(val_path) != status['identity']['baseline_val_prediction_sha256']:
        raise ValueError('Frozen target36 VAL prediction fingerprint differs from its R84 baseline')
    categories = {(r['problem'], r['method']): r['category'] for r in target_record['targets']}
    for item in target_record['targets']:
        data = baseline_data['VAL'][item['problem']]
        slot = data['pool'].index(item['method'])
        if (int((data['winner'] == slot).sum()) != item['baseline_val_winners']
                or int((data['pred'] == slot).sum()) != item['baseline_val_chosen']):
            raise ValueError('Fixed target metadata is not from the specified frozen R84 VAL predictions')
    plans = [('B_subset vs A_static', 'A_static', 'B_subset')]
    plans += [(arm + ' vs R84/A_baseline', 'R84/A_baseline', arm) for arm in ('A_static', 'B_subset')]
    references = {'R84/A_baseline': dict(status='complete', checkpoint_sha256=BASELINE_SHA256)}
    historical = next((m for m in models.values() if (m['config']['experiment'], m['config']['arm']) == ('R94', 'B')), None)
    historical_data = None
    if historical is not None:
        try:
            common.same_data(current['config'], historical['config'])
            historical_data = {s: saved.get((str(historical['run']), s)) or common.saved_split(historical, s, inputs)
                               for s in ('VAL', 'TEST')}
            for split in ('VAL', 'TEST'):
                for p in common.PROBLEMS:
                    common.paired(historical_data[split][p], saved[(str(root / 'A_static'), split)][p], 'R94B', split, p)
        except (OSError, KeyError, ValueError, AssertionError) as error:
            historical_data = None
            references['R94B'] = dict(status='missing_or_incompatible', reason=str(error))
        else:
            plans += [(arm + ' vs R94B', 'R94B', arm) for arm in ('A_static', 'B_subset')]
            references['R94B'] = dict(status='complete', checkpoint_sha256=historical['lock']['checkpoint_sha256'])
    else:
        references['R94B'] = dict(status='not_available_in_original_finisher', optional=True)
    details, summary = [], []
    for split in ('VAL', 'TEST'):
        data = {arm: saved[(str(root / arm), split)] for arm in ('A_static', 'B_subset')}
        data['R84/A_baseline'] = baseline_data[split]
        if historical_data is not None:
            data['R94B'] = historical_data[split]
        for label, left, right in plans:
            rows, totals = recovery_tables(data[left], data[right], targets, label, split)
            for row in rows:
                row['frozen_target_category'] = categories[row['problem'], row['solver']]
            details.extend(rows)
            summary.extend(totals)
    common.remember(inputs, FONT)
    status.update(post_lock_report_only=True, model_change=False, checkpoint_change=False, protocol_change=False,
        target_reselection=False, inference=False, original_gate_unchanged=True, original_decision_gate=gate,
        supplement_generator=str(Path(__file__).resolve()), supplement_generator_sha256=LOADED_SHA256,
        baseline_run=str(BASELINE), baseline_checkpoint_sha256=BASELINE_SHA256,
        comparisons=[p[0] for p in plans], historical_references=references,
        font=dict(path=str(FONT), sha256=common.digest(FONT), embedding='PDF subset only; never embed entire font file'),
        denominator='native TP / native winner count; precision = native TP / chosen count; zero denominators undefined',
        false_pick_definition='new FP: current predicted method is in group, wrong native winner, and differs from baseline prediction',
        cost_partition='cost/regret and both-wrong changes partition by native winner; new-FP columns partition by current prediction',
        ALL='cost/regret equal weight over all18; recall/precision from pooled group counts; mixed raw cost descriptive',
        interpretation='selected counts and zero-count reduction are not success; original predeclared VAL gate is primary')
    common.write_csv(root / 'target_recovery.csv', details)
    common.write_csv(root / 'target_recovery_summary.csv', summary)
    common.write(root / 'target_status.json', status)
    zero_selection_pdf(root, details, summary, status, gate)
    return status


def private_finisher():
    """Create a fresh isolated namespace, preserving frozen generator/source hashes."""
    module = types.ModuleType('code.V4._r116_mechanism_' + uuid.uuid4().hex)
    module.__file__, module.__package__ = str(FROZEN_FINISHER), 'code.V4'
    source = FROZEN_FINISHER.read_text(encoding='utf-8')
    old = "'code.V4._r116_finish_callbacks'"
    if source.count(old) != 1:
        raise RuntimeError('Frozen finisher namespace template drift')
    source = source.replace(old, repr(module.__name__ + '_callbacks'))
    sys.modules[module.__name__] = module
    exec(compile(source, str(FROZEN_FINISHER) + ':mechanism_private', 'exec'), module.__dict__)
    callbacks, context = module._callbacks, {}
    module.OUTPUTS = (*module.OUTPUTS, *EXTRA_OUTPUTS)
    module.CODE_NAMES = (*module.CODE_NAMES, *EXTRA_CODE)
    callbacks.OUTPUTS, callbacks.CODE_NAMES = module.OUTPUTS, module.CODE_NAMES
    completed, load_run, saved_split = callbacks.completed, callbacks.load_run, callbacks.saved_split
    core_render, core_write, publication_paths = callbacks.render, callbacks.write, module.publication_paths
    frozen_hash = common.digest(FROZEN_FINISHER)

    def completed_with_inputs(root):
        # No supervision/archive read can happen until both hard pipelines complete.
        inputs = completed(root)
        if common.digest(Path(__file__)) != LOADED_SHA256 or common.digest(FROZEN_FINISHER) != frozen_hash:
            raise ValueError('Loaded supplement/frozen finisher source changed; retire this CPU watcher explicitly')
        for path in (Path(__file__), V4 / EXTRA_CODE[1], FROZEN_FINISHER, TARGET_LIST,
                     Path(common.__file__), Path(recovery.__file__)):
            common.remember(inputs, path)
        context.clear()
        context.update(root=Path(root).resolve(), inputs=inputs, models={}, saved={})
        return inputs

    def remember_model(*args, **kwargs):
        model = load_run(*args, **kwargs)
        context['models'][str(model['run'])] = model
        return model

    def remember_split(model, split, inputs):
        data = saved_split(model, split, inputs)
        context['saved'][str(model['run']), split] = data
        return data

    def render_with_targets(root, metrics, comparisons, references, boundary):
        if context.get('root') != Path(root).resolve():
            raise RuntimeError('Supplement render requires this complete hard-receipt analysis context')
        core_render(root, metrics, comparisons, references, boundary)
        gate = common.read(Path(root) / 'decision_gate.json')
        if gate != module.decision_gate(root):
            raise ValueError('Supplement must show the actual unchanged frozen decision gate')
        for arm in module.ARMS:
            common.remember(context['inputs'], Path(root) / arm / 'inference_timing.json')
        _supplement(Path(root).resolve(), context['models'], context['saved'], context['inputs'], gate)

    def write_with_provenance(path, value):
        if Path(path).name == 'finish_complete.json':
            value = dict(value, postlock_reporting_augmentation=dict(generator=str(Path(__file__).resolve()),
                generator_sha256=LOADED_SHA256, outputs=list(EXTRA_OUTPUTS), report_only=True,
                scientific_source_hashes_unchanged=True, model_change=False, checkpoint_change=False,
                protocol_change=False, targets_sha256=common.digest(TARGET_LIST)))
        return core_write(path, value)

    def publication_with_targets(root):
        paths, excluded = publication_paths(root)
        if TARGET_LIST not in paths:
            paths.append(TARGET_LIST)
        if TARGET_LIST.is_symlink() or not TARGET_LIST.is_file() or not TARGET_LIST.is_relative_to(common.GIT_ROOT):
            raise ValueError('Target provenance must be an ordinary in-repository file')
        if TARGET_LIST.stat().st_size > common.SMALL_LIMIT or sum(p.stat().st_size for p in paths) > common.TOTAL_LIMIT:
            raise ValueError('Supplement must respect original bounded curated publication limits')
        return paths, excluded

    callbacks.completed, callbacks.load_run, callbacks.saved_split = completed_with_inputs, remember_model, remember_split
    callbacks.render, callbacks.write = render_with_targets, write_with_provenance
    module.publication_paths = callbacks.publication_paths = publication_with_targets
    return module


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('wait', 'analyze', 'publish'))
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--publish-after', action='store_true')
    parser.add_argument('--timeout-hours', type=float, default=32.)
    parser.add_argument('--poll-seconds', type=float, default=60.)
    parser.add_argument('--message', default='Publish complete R116 matched models and fixed36 reporting')
    args = parser.parse_args(argv)
    root, finisher = args.root.expanduser().resolve(), private_finisher()
    if args.command == 'wait':
        return finisher.wait(root, timeout_hours=args.timeout_hours, poll_seconds=args.poll_seconds,
                             publish_after=args.publish_after, message=args.message)
    if args.publish_after and args.command != 'wait':
        raise ValueError('--publish-after belongs to the finite wait command only')
    return finisher.analyze(root) if args.command == 'analyze' else finisher.publish(root, args.message)


if __name__ == '__main__':
    main()

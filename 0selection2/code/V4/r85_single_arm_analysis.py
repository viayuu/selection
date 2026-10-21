"""CPU-only R85C analysis of locked, already saved evaluation artifacts."""

import csv
import json
import os
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = ''
for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[variable] = '2'

import numpy as np

from code.unified_selector.registry import PROBLEMS
from . import r85_reporting as api
from .r85_goal import passes, summarize_targets


ROOT = Path(__file__).resolve().parent / 'runs/R85_zero_selection_research'
BASELINE = ROOT.parent / 'R84_new90k_attention/A_baseline'
OUTPUT = ROOT / 'C_cost_profile_analysis'
ARM = 'C_cost_profile'
MODEL = 'R85C'
TARGET_DEFINITION = '36 published R84A locked VAL zero/near-zero (chosen <= 0.1%) groups'
METRIC_NAMES = ('top1_pct', 'mean_cost', 'actual_regret_pct')


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def save_json(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')


def category_changes(old, new, split):
    rows = []
    for problem in PROBLEMS:
        left, right = old[problem], new[problem]
        index, oracle = np.arange(left.n), left.costs.min(1)
        left_cost, right_cost = left.costs[index, left.pred], right.costs[index, right.pred]
        delta = right_cost - left_cost
        old_ok, new_ok = left.pred == left.winner, right.pred == right.winner
        both_wrong = ~old_ok & ~new_ok
        masks = dict(corrected=~old_ok & new_ok, harmed=old_ok & ~new_ok,
                     both_wrong_cheaper=both_wrong & (delta < 0),
                     both_wrong_dearer=both_wrong & (delta > 0),
                     both_wrong_equal=both_wrong & (delta == 0), both_correct=old_ok & new_ok)
        for category, mask in masks.items():
            count = int(mask.sum())
            rows.append(dict(split=split, problem=problem, category=category, n=left.n, count=count,
                old_category_mean_cost=float(left_cost[mask].mean()) if count else None,
                new_category_mean_cost=float(right_cost[mask].mean()) if count else None,
                raw_cost_delta_sum=float(delta[mask].sum()),
                mean_cost_delta_contribution=float(delta[mask].sum() / left.n),
                actual_regret_delta_contribution_pp=float(100. * (delta[mask] / oracle[mask]).sum() / left.n)))
    for category in masks:
        current = [r for r in rows if r['category'] == category]
        rows.append(dict(split=split, problem='ALL', category=category,
            n=sum(r['n'] for r in current), count=sum(r['count'] for r in current),
            old_category_mean_cost=None, new_category_mean_cost=None, raw_cost_delta_sum=None,
            mean_cost_delta_contribution=float(np.mean([r['mean_cost_delta_contribution'] for r in current])),
            actual_regret_delta_contribution_pp=float(np.mean([r['actual_regret_delta_contribution_pp'] for r in current]))))
    return rows


def typed_methods(rows, model):
    result = []
    for row in rows:
        counts = {key: int(row[key]) for key in ('n', 'winner_count', 'selected_count', 'correct_count',
                  'missed_winner_count', 'false_pick_count')}
        chosen, support, correct = counts['selected_count'], counts['winner_count'], counts['correct_count']
        result.append(dict(model=model, problem=row['problem'], split=row['split'], solver=row['solver'],
            **counts, selected_share_pct=100. * chosen / counts['n'],
            recall_pct=100. * correct / support if support else None,
            precision_pct=100. * correct / chosen if chosen else None))
    return result


def method_totals(rows, targets, model, split):
    selected = [r for r in rows if r['split'] == split]
    summary = dict(api.zero_summary(selected)['all'], model=model, split=split)
    current = [r for r in selected if (r['problem'], r['solver']) in targets]
    for key in ('winner_count', 'selected_count', 'correct_count', 'false_pick_count'):
        summary['target_' + key] = sum(r[key] for r in current)
    winners, chosen, correct = (summary['target_' + k] for k in ('winner_count', 'selected_count', 'correct_count'))
    summary.update(target_recall_pct=100. * correct / winners if winners else None,
                   target_precision_pct=100. * correct / chosen if chosen else None)
    return summary


def collect():
    run, config, lock, test, val = api.locked_inputs(ROOT, ARM)
    old_run, old_config, old_lock, old_test, old_val = api.shared.locked_inputs(BASELINE.parent, BASELINE.name)
    if config['data_root'] != old_config['data_root'] or config['candidate_pools'] != old_config['candidate_pools']:
        raise ValueError('R85C and historical R84A must use the same frozen NEW90k release and pools')
    frozen = api.read_json(ROOT / 'comparison_artifacts/target_list.json')
    if (frozen['identity']['baseline_checkpoint_sha256'] != old_lock['checkpoint_sha256']
            or frozen['identity']['data_root'] != config['data_root']
            or frozen['groups'] != 36 or frozen['baseline_val_winners'] != 4625):
        raise ValueError('Published fixed target identity changed')
    targets = [(r['problem'], r['method']) for r in frozen['targets']]
    packed, metrics, pairs, costs, methods, target_rows, target_summary = {}, [], [], [], [], [], []
    sources = []
    for model, args in (('R84A', (old_run, old_config, old_lock, old_test, old_val)),
                        (MODEL, (run, config, lock, test, val))):
        predictions, current_metrics, _ = api.shared.load_saved_splits(*args)
        metrics.extend(dict(row, model=model) for row in current_metrics)
        packed[model] = {split: {p: predictions[split, p] for p in PROBLEMS} for split in ('VAL', 'TEST')}
        sources.append(dict(model=model, run=str(args[0]), selected_update=args[2]['selected_update'],
            checkpoint_sha256=args[2]['checkpoint_sha256'], state_hash=args[2]['state_hash'],
            model_state_sha256=args[3]['identity']['model_state_sha256']))
    for split in ('VAL', 'TEST'):
        old, new = packed['R84A'][split], packed[MODEL][split]
        pairs.extend(api.paired_comparison(old, new, MODEL, 'R84A', split))
        costs.extend(category_changes(old, new, split))
        for model in ('R84A', MODEL):
            current = api.fixed_target_recovery(old, packed[model][split], targets, model, 'R84A', split)
            for row in current:
                row['target_definition'] = TARGET_DEFINITION
            target_rows.extend(current)
            target_summary.append(summarize_targets(current, model, 'R84A', split, TARGET_DEFINITION))

    # Reuse selected-state TRAIN tables; never run another forward pass.
    current_train, current_path = api.selected_train(run, config, lock)
    baseline_train, baseline_path = api.selected_train(old_run, old_config, old_lock)
    for result, path, expected in ((current_train, current_path, test), (baseline_train, baseline_path, old_test)):
        if not path.is_file() or result['identity']['model_state_sha256'] != expected['identity']['model_state_sha256']:
            raise ValueError('Missing TRAIN archive or TRAIN is not the locked selected state')
    current_methods_path = run / 'full_model_statistics/method_selection_train_val_test.csv'
    old_methods_path = ROOT / 'prior_experiments/method_diagnostics.csv'
    current_methods = typed_methods(read_csv(current_methods_path), MODEL)
    old_saved = [r for r in read_csv(old_methods_path) if r['model'] == 'R84A' and r['split'] == 'TRAIN']
    if not old_saved or any(Path(r['prediction_path']).resolve() != baseline_path.resolve() for r in old_saved):
        raise ValueError('Existing baseline TRAIN method table is not from the selected archive')
    old_methods = typed_methods(old_saved, 'R84A')
    methods = old_methods + current_methods
    train_metrics, train_summary = [], []
    for model, result, counts in (('R84A', baseline_train, old_methods), (MODEL, current_train, current_methods)):
        rows = []
        for problem in PROBLEMS:
            saved = result['per_problem'][problem]
            method_rows = [r for r in counts if r['problem'] == problem and r['split'] == 'TRAIN']
            if sum(r['correct_count'] for r in method_rows) != round(saved['top1'] * saved['n']):
                raise ValueError('Selected TRAIN method counts and selected TRAIN metrics differ')
            rows.append(dict(model=model, split='TRAIN', problem=problem, n=saved['n'],
                top1_pct=100. * saved['top1'], mean_cost=saved['mean_cost'], sbs_cost=saved['sbs_cost'],
                oracle_cost=saved['oracle_cost'], actual_regret_pct=saved['actual_regret_pct'],
                vs_sbs_pct=saved['vs_sbs_pct'], vs_oracle_pct=saved['vs_oracle_pct'], ce=saved['ce']))
        total = dict(model=model, split='TRAIN', problem='ALL', n=sum(r['n'] for r in rows))
        for key in (*METRIC_NAMES, 'sbs_cost', 'oracle_cost', 'vs_sbs_pct', 'vs_oracle_pct', 'ce'):
            total[key] = float(np.mean([r[key] for r in rows]))
        train_metrics.extend([total, *rows])
        train_summary.append(method_totals(counts, targets, model, 'TRAIN'))

    all_metrics = {(r['model'], r['split']): r for r in metrics if r['problem'] == 'ALL'}
    gate = dict(split='VAL', pass_joint_gate=passes(all_metrics[MODEL, 'VAL'], all_metrics['R84A', 'VAL']),
        top1_improves=all_metrics[MODEL, 'VAL']['top1_pct'] > all_metrics['R84A', 'VAL']['top1_pct'],
        mean_cost_no_worse=all_metrics[MODEL, 'VAL']['mean_cost'] <= all_metrics['R84A', 'VAL']['mean_cost'],
        actual_regret_no_worse=all_metrics[MODEL, 'VAL']['actual_regret_pct'] <= all_metrics['R84A', 'VAL']['actual_regret_pct'],
        test_used_for_acceptance=False)
    lib_source = run / 'full_model_statistics/metrics_by_problem.csv'
    libs = [r for r in read_csv(lib_source) if r['split'] in ('TSPLIB', 'CVRPLIB')]
    if len(libs) != 2 or any(r['checkpoint_sha256'] != lock['checkpoint_sha256'] for r in libs):
        raise ValueError('Expected two already completed LIB rows from the same lock')
    train_discrimination = []
    old_index = {(r['problem'], r['solver']): r for r in old_methods}
    for row in (r for r in current_methods if r['split'] == 'TRAIN'):
        old = old_index[row['problem'], row['solver']]
        if row['n'] != old['n'] or row['winner_count'] != old['winner_count']:
            raise ValueError('Baseline and C TRAIN supports differ')
        train_discrimination.append(dict(row, baseline_selected_count=old['selected_count'],
            baseline_correct_count=old['correct_count'], baseline_recall_pct=old['recall_pct'],
            correct_count_delta=row['correct_count'] - old['correct_count'],
            frozen_target=(row['problem'], row['solver']) in targets))
    evidence = dict(analysis='R85 C_cost_profile only vs historical R84A', cpu_only=True,
        inference_performed=False, training_performed=False, goal_complete=False,
        core_modified=False, new_gpu_started=False, sources=sources, data_root=config['data_root'],
        rows_per_problem=config['rows_per_problem'], target_source=str(ROOT / 'comparison_artifacts/target_list.json'),
        fixed_groups=len(targets), baseline_val_target_winners=frozen['baseline_val_winners'],
        gate=gate, missing_items=[], train_sources=dict(C=str(current_path), R84A=str(baseline_path),
            C_method_table=str(current_methods_path), R84A_method_table=str(old_methods_path)),
        train_method_comparison='Already saved selected-state counts; no new inference or TRAIN pairwise rescue calculation',
        probability_note='Only native FP64 winners and argmax decision counts; no softmax probability is called calibrated',
        aggregation='Equal weights for all 18 tasks; ALL raw-cost average is descriptive, raw cost sums never pooled across tasks',
        limitation='One seed, historical R84A comparator; TEST/LIB diagnostic only and already used in prior research',
        next_action='Wait for other R85 arms; do not promote C from TEST or launch a follow-up based on this analysis')
    for split in ('VAL', 'TEST'):
        total = next(r for r in pairs if r['split'] == split and r['problem'] == 'ALL')
        groups = [r for r in costs if r['split'] == split and r['problem'] == 'ALL']
        np.testing.assert_allclose(sum(r['actual_regret_delta_contribution_pp'] for r in groups),
                                   total['actual_regret_delta_pp'], atol=1e-12, rtol=0)
        if sum(r['count'] for r in groups) != total['n']:
            raise ValueError('Error categories must partition the evaluation population')
    return dict(metrics=metrics, pairs=pairs, costs=costs, targets=target_rows, target_summary=target_summary,
                train_metrics=train_metrics, train_summary=train_summary, train_methods=train_discrimination,
                libs=libs, evidence=evidence, lock=lock)


def fmt(value, digits=4):
    return 'undefined' if value is None or value == '' else f'{float(value):.{digits}f}'


def write_results(data):
    gate, lock = data['evidence']['gate'], data['lock']
    text = ['# R85C: locked single-arm analysis', '',
        f"C: epoch {lock['epoch']}, update {lock['selected_update']}; SHA256 `{lock['checkpoint_sha256']}`.",
        'Historical R84A: update 37440. Same frozen NEW90k instances, candidate pools and native/FP64 winners.', '',
        '## Full-system result', '',
        '| Split | Model | Top1 (%) | mean_cost | actual regret (%) |',
        '| --- | --- | ---: | ---: | ---: |']
    for split in ('VAL', 'TEST'):
        for model in ('R84A', MODEL):
            row = next(r for r in data['metrics'] if (r['model'], r['split'], r['problem']) == (model, split, 'ALL'))
            text.append(f"| {split} | {model} | {fmt(row['top1_pct'])} | {fmt(row['mean_cost'],9)} | {fmt(row['actual_regret_pct'],6)} |")
    text += ['', f"**Joint VAL gate: {'PASS' if gate['pass_joint_gate'] else 'FAIL'}.** "
        'Top1 must improve while mean_cost and actual regret do not worsen. C improves cost/regret but not VAL Top1. '
        'A better TEST Top1 does not override this gate.', '', '## Fixed zero/near-zero targets', '',
        '36 groups were frozen from R84A VAL before this analysis (4,625 VAL winners). '
        'The per-arm strict-zero-only CSV has a smaller scope; this report consistently uses the full published list.', '',
        '| Split | Model | True winners | Correct | Recall (%) | Picks | Precision (%) | Rescued | Lost | New false picks |',
        '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for row in data['target_summary']:
        text.append(f"| {row['split']} | {row['model']} | {row['winner_count']} | {row['correct_count']} | "
            f"{fmt(row['recall_pct'])} | {row['selected_count']} | {fmt(row['precision_pct'])} | "
            f"{row['rescued']} | {row['lost']} | {row['new_false_pick_count']} |")
    text += ['', 'Rescued/lost/new false picks compare the same instances with R84A. '
        'A new false pick means C newly selects a target on which that target is not the native winner. '
        'Undefined support or selection denominators remain undefined, never zero.', '', '## Paired net changes', '',
        '| Split | Corrected | Harmed | Both wrong: cheaper / dearer | Net correct | Regret delta (pp) |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for row in (r for r in data['pairs'] if r['problem'] == 'ALL'):
        text.append(f"| {row['split']} | {row['corrected']} | {row['harmed']} | {row['both_wrong_cheaper']} / "
            f"{row['both_wrong_dearer']} | {row['net_correct']} | {fmt(row['actual_regret_delta_pp'],6)} |")
    text += ['', '`cost_change_categories.csv` partitions every instance, including both-wrong/equal-cost and both-correct. '
        'Each regret contribution uses the full problem population as denominator, then equal task weights; '
        'negative deltas favor C. Raw sums are retained per problem only.', '', '## TRAIN: selected checkpoint, not final epoch', '',
        '| Model | TRAIN Top1 (%) | Zero groups | Zero winner mass | Target correct / winners | Target recall (%) |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for row in data['train_summary']:
        metric = next(r for r in data['train_metrics'] if r['problem'] == 'ALL' and r['model'] == row['model'])
        text.append(f"| {row['model']} | {fmt(metric['top1_pct'])} | {row['zero_with_winner_groups']} | "
            f"{row['zero_winner_mass']} | {row['target_correct_count']} / {row['target_winner_count']} | {fmt(row['target_recall_pct'])} |")
    text += ['', 'All TRAIN results already existed for the locked states. No GPU or new inference was used. '
        'TRAIN zero selection remains for well-supported methods, including TSP LEHD/DIFUSCO and ATSP MATPOENET. '
        'This is decision-level suppression already on TRAIN, not evidence that the only issue is unseen-instance overfitting. '
        'Counts do not identify its unique cause or establish a performance ceiling.', '', '## Existing LIB results (C only)', '',
        '| Dataset | n | Top1 (%) | mean_cost | actual regret (%) |', '| --- | ---: | ---: | ---: | ---: |']
    for row in data['libs']:
        text.append(f"| {row['split']} | {row['n']} | {fmt(row['top1_pct'])} | {fmt(row['mean_cost'],9)} | {fmt(row['actual_regret_pct'],6)} |")
    text += ['', 'LIB rows quote the completed same-lock report only; neither LIB nor TEST selects a model here.', '',
        '## Scope and next action', '',
        'No missing requested TRAIN/VAL/TEST/LIB evidence. Full18 per-task VAL and TEST metrics, TRAIN selected-state '
        'counts, fixed-target rows, and category cost changes are in companion CSVs. TRAIN paired rescues/AUC are not '
        'newly computed; only saved selected-state decision counts are compared.', '',
        'C does not meet the predeclared joint VAL gate and does not establish a substantial repair of missed methods. '
        'Wait for the other committed R85 arms before a research handoff. No follow-up experiment is authorized by this report.', '',
        'Limits: one seed; R84A is a historical comparator, not a freshly matched R85A control; '
        'prior TEST/LIB have participated in research. ALL cost is a descriptive equal-task mean, not a pooled '
        'cross-unit objective. This analysis does not complete the active research goal.', '']
    (OUTPUT / 'RESULTS.md').write_text('\n'.join(text), encoding='utf-8')


def write_pdf(data):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    helper, common = api.common.reporting_helpers(), api.common
    plt.rcParams.update({'pdf.fonttype': 42, 'font.family': 'DejaVu Sans'})
    page = lambda title, subtitle, n: common._page(helper, title, subtitle, MODEL, data['lock'], n)
    table = helper.add_table
    with PdfPages(OUTPUT / 'ANALYSIS.pdf') as pdf:
        fig = page('R85C | Locked result and missed-method recovery',
                   'Epoch 25 / update 36000; historical R84A update 37440. Same NEW90k release; no new inference.', 1)
        rows = [r for r in data['metrics'] if r['problem'] == 'ALL']
        table(fig, ['Split', 'Model', 'Top1 (%)', 'mean_cost', 'Actual regret (%)'],
              [[r['split'], r['model'], fmt(r['top1_pct']), fmt(r['mean_cost'],9), fmt(r['actual_regret_pct'],6)] for r in rows],
              [.045,.71,.91,.14], font=12)
        fig.text(.045,.672,'Joint VAL gate: FAIL. Lower cost/regret do not compensate for lower VAL Top1.',
                 fontsize=13, color='#9b3039', weight='bold')
        fig.text(.045,.633,'Published 36 zero/near-zero targets: 4,625 VAL winners. No target redefinition using C or TEST.',fontsize=11)
        table(fig, ['Split/model', 'Winners', 'Correct', 'Recall %', 'Picks', 'Precision %', 'Rescue/lost', 'New false'],
              [[r['split']+'/'+r['model'], str(r['winner_count']), str(r['correct_count']), fmt(r['recall_pct'],3),
                str(r['selected_count']),fmt(r['precision_pct'],2),f"{r['rescued']}/{r['lost']}",str(r['new_false_pick_count'])]
               for r in data['target_summary']], [.045,.445,.91,.155],font=10.5)
        cats = [r for r in data['costs'] if r['problem'] == 'ALL' and r['category'] in
                ('corrected','harmed','both_wrong_cheaper','both_wrong_dearer')]
        table(fig,['Split','Decision change','Count','Regret delta (pp)','Mean cost delta'],
              [[r['split'],r['category'],f"{r['count']:,}",fmt(r['actual_regret_delta_contribution_pp'],6),
                fmt(r['mean_cost_delta_contribution'],9)] for r in cats],[.045,.117,.91,.28],font=10.5)
        fig.text(.045,.078,'Negative deltas favor C. Category contributions use all instances per task, then equal task weights.',fontsize=10)
        fig.text(.045,.05,'ALL mean_cost is descriptive only; cross-task raw-cost sums are not pooled. TEST cannot override VAL.',fontsize=10)
        pdf.savefig(fig); plt.close(fig)

        fig = page('R85C | Full18: VAL and TEST kept separate',
                   'Each task: 4,500 instances per split; native/FP64 winner. Deltas = C minus historical R84A.',2)
        pair_index = {(r['split'],r['problem']):r for r in data['pairs']}
        cells = []
        for problem in (*PROBLEMS,'ALL'):
            left,right = pair_index['VAL',problem],pair_index['TEST',problem]
            cells.append([problem,fmt(left['top1_delta_pp'],4),fmt(left['actual_regret_delta_pp'],6),
                          fmt(left['mean_cost_delta'],7),f"{left['corrected']}/{left['harmed']}",
                          fmt(right['top1_delta_pp'],4),fmt(right['actual_regret_delta_pp'],6),
                          fmt(right['mean_cost_delta'],7),f"{right['corrected']}/{right['harmed']}"])
        table(fig,['Task','VAL Top1 pp','VAL regret pp','VAL cost','VAL fix/harm',
                   'TEST Top1 pp','TEST regret pp','TEST cost','TEST fix/harm'],cells,
              [.045,.22,.91,.63],widths=[.12,.105,.12,.11,.105,.105,.12,.11,.105],font=9.5)
        fig.text(.045,.16,'Complete absolute metrics, TRAIN-fitted SBS, Oracle, vs_SBS and vs_Oracle are in metrics_by_problem.csv.',fontsize=11)
        fig.text(.045,.119,'Top1 improvements and regret reductions need not occur on the same tasks. No per-task model assembly.',fontsize=11)
        fig.text(.045,.078,'Only the prelocked VAL screen can qualify C; this report does not select another checkpoint or inference rule.',fontsize=11)
        pdf.savefig(fig); plt.close(fig)

        fig = page('R85C | TRAIN suppression and existing LIB evidence',
                   'TRAIN predictions are already saved for each selected checkpoint; this analysis does not occupy a GPU.',3)
        cells=[]
        for r in data['train_summary']:
            metric=next(m for m in data['train_metrics'] if m['problem']=='ALL' and m['model']==r['model'])
            cells.append([r['model'],fmt(metric['top1_pct']),str(r['zero_with_winner_groups']),f"{r['zero_winner_mass']:,}",
                          f"{r['target_correct_count']}/{r['target_winner_count']}",fmt(r['target_recall_pct'],3)])
        table(fig,['Model','TRAIN Top1 %','Zero groups','Zero winner mass','Target correct/support','Recall %'],cells,
              [.045,.73,.91,.12],font=11.5)
        wanted=[('TSP','LEHD'),('TSP','DIFUSCO'),('ATSP','MATPOENET'),('CVRP','RouteFinder')]
        selected=[next(r for r in data['train_methods'] if (r['problem'],r['solver'])==key) for key in wanted]
        table(fig,['TRAIN task/method','Winners','R84A picks/correct','C picks/correct','C recall %','C precision %'],
              [[r['problem']+'/'+r['solver'],f"{r['winner_count']:,}",f"{r['baseline_selected_count']}/{r['baseline_correct_count']}",
                f"{r['selected_count']}/{r['correct_count']}",fmt(r['recall_pct'],3),fmt(r['precision_pct'],3)] for r in selected],
              [.045,.50,.91,.18],font=11)
        fig.text(.045,.456,'Decision suppression persists on TRAIN; unseen-instance overfitting alone is not an adequate explanation.',fontsize=11.5)
        table(fig,['Same locked C model','n','Top1 (%)','mean_cost','Actual regret (%)'],
              [[r['split'],r['n'],fmt(r['top1_pct']),fmt(r['mean_cost'],9),fmt(r['actual_regret_pct'],6)] for r in data['libs']],
              [.045,.30,.91,.11],font=11.5)
        notes=[
            'LIB rows quote the existing completed report; no new inference or LIB-based model selection.',
            'Missing evidence: none required here. TRAIN comparison reuses selected-state counts, not newly computed AUC.',
            'One seed, historical R84A reference; R85A is still needed as the newly matched control.',
            'No causal proof, no accuracy ceiling claim, no acceptance from fewer zero groups or a better TEST Top1.',
            'Next: wait for other R85 arms. The research goal remains active and unresolved.'
        ]
        for i,note in enumerate(notes):
            fig.text(.045,.25-i*.038,note,fontsize=10.5)
        pdf.savefig(fig); plt.close(fig)


def main():
    OUTPUT.mkdir(parents=True,exist_ok=True)
    data=collect()
    tables={'metrics_by_problem.csv':data['metrics'],'paired_changes.csv':data['pairs'],
        'cost_change_categories.csv':data['costs'],'target_summary.csv':data['target_summary'],
        'target_recovery_by_method.csv':data['targets'],'train_metrics.csv':data['train_metrics'],
        'train_summary.csv':data['train_summary'],'train_method_discrimination.csv':data['train_methods'],
        'lib_metrics.csv':data['libs']}
    for name,rows in tables.items():
        api.common.write_csv(OUTPUT/name,rows)
    for split in ('VAL','TEST'):
        api.common.write_csv(OUTPUT/(split.lower()+'_metrics.csv'),[r for r in data['metrics'] if r['split']==split])
    save_json(OUTPUT/'evidence.json',dict(data['evidence'],target_summary=data['target_summary'],train_summary=data['train_summary']))
    write_results(data)
    print(json.dumps(dict(phase='saved_evidence_ready',output=str(OUTPUT),gate=data['evidence']['gate'],
                         target_summary=data['target_summary'],train_summary=data['train_summary']),indent=2),flush=True)
    write_pdf(data)
    print(json.dumps(dict(phase='complete',pdf=str(OUTPUT/'ANALYSIS.pdf'),cpu_only=True,inference=False)),flush=True)


if __name__=='__main__':
    main()

"""Replay integrated R53 decisions, separate corrections from expensive harms."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from code.unified_selector.registry import PROBLEMS
from .multitask_probe import dump
from .pair_specialist import ordinal_scores
from .performance_evaluation import decision_metrics
from .r48_common import write_csv
from .r53_data import MVRP, ROOT, baseline_metrics, read_baseline


ARM_NAMES = dict(A='A_ce_seed2', B='B_cost_sensitive_seed2')


def changes(saved, problem, split, arm):
    cost, winner = saved['costs'], saved['winner']
    old, new = saved['baseline_pred'], saved['pred']
    n = len(old)
    old_ok, new_ok = old == winner, new == winner
    delta = cost[np.arange(n), new]-cost[np.arange(n), old]
    relative = delta/cost.min(1)*100
    categories = dict(corrected=(~old_ok & new_ok), harmed=(old_ok & ~new_ok),
        both_wrong_cheaper=(~old_ok & ~new_ok & (delta < 0)),
        both_wrong_more_expensive=(~old_ok & ~new_ok & (delta > 0)),
        both_wrong_equal_cost_switch=(~old_ok & ~new_ok & (delta == 0) & (old != new)),
        unchanged=(old == new))
    rows = []
    for category, mask in categories.items():
        rows.append(dict(split=split, arm=arm, problem=problem, category=category, count=int(mask.sum()),
            sum_cost_delta=float(delta[mask].sum()), mean_cost_delta_contribution=float(delta[mask].sum()/n),
            macro_regret_delta_contribution_pp=float(relative[mask].sum()/n/len(PROBLEMS)),
            mean_regret_change_within_category_pct=float(relative[mask].mean()) if mask.any() else 0.))
    return rows


def plot_curves(root, histories):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    colors = dict(A='#007c91', B='#bb4c2d')
    for arm, history in histories.items():
        x = [r['successful_updates'] for r in history]
        color = colors[arm]
        for ax, key, title, factor in (
                (axes[0, 0], 'ce', 'Specialist unweighted BCE', 1),
                (axes[0, 1], 'accuracy', 'Specialist binary accuracy (%)', 100),
                (axes[0, 2], 'pair_regret_pct', 'Specialist two-method regret (%)', 1)):
            ax.plot(x, [r['binary_val']['macro'][key]*factor for r in history], color=color, label=arm+' val')
            train = [r for r in history if r['binary_train'] is not None]
            ax.plot([r['successful_updates'] for r in train],
                [r['binary_train']['macro'][key]*factor for r in train], '--', color=color, label=arm+' train_eval')
            ax.set_title(title)
        axes[1, 0].plot(x, [r['val']['groups']['ALL']['top1']*100 for r in history], color=color, label=arm)
        axes[1, 1].plot(x, [r['val']['groups']['ALL']['actual_regret_pct'] for r in history], color=color, label=arm)
        axes[1, 2].plot(x, [r['learning_rate'] for r in history], color=color, label=arm)
    baseline = json.loads((root/'gate_budget.json').read_text())['baseline']
    axes[1, 0].axhline(baseline['top1']*100, color='#555555', linestyle=':', label='R45A')
    axes[1, 1].axhline(baseline['actual_regret_pct'], color='#555555', linestyle=':', label='R45A')
    for ax, title in zip(axes[1], ('Integrated 18-task val Top1 (%)', 'Integrated 18-task val actual regret (%)', 'Learning rate')):
        ax.set_title(title)
    for ax in axes.flat:
        ax.set_xlabel('Successful optimizer updates')
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.savefig(root/'learning_curves.png', dpi=160)
    plt.close(fig)


def analyze(root=ROOT):
    root = Path(root)
    protocol = json.loads((root/'protocol.json').read_text())
    budget = json.loads((root/'gate_budget.json').read_text())
    results = {arm:json.loads((root/name/'result.json').read_text()) for arm, name in ARM_NAMES.items()}
    histories = {arm:json.loads((root/name/'history.json').read_text()) for arm, name in ARM_NAMES.items()}
    if results['A']['initialization_sha256'] != results['B']['initialization_sha256']:
        raise ValueError('A/B were not initialized identically')
    rows, correction_rows, selections, scope_rows, replay = [], [], [], [], {}
    for split in ('val', 'test'):
        raw, baseline = read_baseline(split, root)
        base = baseline_metrics(raw, baseline)
        replay[split] = dict(R45A=base)
        for name, evaluated in [('R45A', base)] + [
                ('R53'+arm, results[arm]['best_validation'] if split == 'val' else
                 json.loads((root/ARM_NAMES[arm]/'test_result.json').read_text())['integrated']) for arm in ('A', 'B')]:
            for scope, values in dict(ALL=evaluated['groups']['ALL'], MVRP=evaluated['groups']['MVRP'],
                                      **evaluated['per_problem']).items():
                rows.append(dict(split=split, model=name, scope=scope,
                    top1_pct=values['top1']*100, top2_pct=values['top2']*100, top3_pct=values['top3']*100,
                    mean_cost=values['mean_cost'], vs_sbs_pct=values['vs_sbs_pct'],
                    vs_oracle_pct=values['vs_oracle_pct'], actual_regret_pct=values['actual_regret_pct']))
        for arm, directory in ARM_NAMES.items():
            expected = results[arm]['best_validation'] if split == 'val' else json.loads((root/directory/'test_result.json').read_text())['integrated']
            for p in PROBLEMS:
                with np.load(root/directory/'predictions'/split/(p+'.npz'), allow_pickle=False) as saved:
                    np.testing.assert_array_equal(saved['costs'], raw[p]['costs'])
                    np.testing.assert_array_equal(saved['winner'], raw[p]['winner'])
                    metrics, pred = decision_metrics(ordinal_scores(saved['order']), raw[p])
                    np.testing.assert_array_equal(pred, saved['pred'])
                    for key in ('top1', 'top2', 'top3', 'mean_cost', 'actual_regret_pct'):
                        np.testing.assert_allclose(metrics[key], expected['per_problem'][p][key], rtol=0, atol=1e-10)
                    correction_rows.extend(changes(saved, p, split, arm))
                    for solver, fraction in metrics['arm_distribution'].items():
                        selections.append(dict(split=split, arm=arm, problem=p, solver=solver, selected_pct=fraction*100))
                    if p in MVRP:
                        for i in range(len(pred)):
                            scope_rows.append(dict(split=split, arm=arm, problem=p, instance_index=i,
                                gate=bool(saved['gate'][i]), baseline=raw[p]['pool'][int(saved['baseline_pred'][i])],
                                selected=raw[p]['pool'][int(pred[i])], winner=raw[p]['pool'][int(raw[p]['winner'][i])],
                                expert_logit=float(saved['expert_logit'][i]),
                                selected_cost=float(raw[p]['costs'][i, pred[i]]),
                                baseline_cost=float(raw[p]['costs'][i, saved['baseline_pred'][i]])))
            replay[split]['R53'+arm] = expected
    write_csv(root/'comparison.csv', rows)
    write_csv(root/'corrections_and_harms.csv', correction_rows)
    write_csv(root/'solver_selection.csv', selections)
    write_csv(root/'instance_decisions.csv', scope_rows)
    binary_rows = []
    for arm, result in results.items():
        for split, metrics in (('train', result['best_binary_train']), ('val', result['best_binary_val']),
                ('test', json.loads((root/ARM_NAMES[arm]/'test_result.json').read_text())['binary'])):
            for scope, values in dict(MVRP=metrics['macro'], **metrics['per_problem']).items():
                binary_rows.append(dict(arm=arm, split=split, scope=scope, **values))
    write_csv(root/'binary_comparison.csv', binary_rows)
    curve_rows = []
    for arm, history in histories.items():
        for row in history:
            curve_rows.append(dict(arm=arm, epoch=row['epoch'], successful_updates=row['successful_updates'],
                training_objective=row['loss'], grad_norm=row['grad_norm'], learning_rate=row['learning_rate'],
                integrated_val_top1_pct=row['val']['groups']['ALL']['top1']*100,
                integrated_val_actual_regret_pct=row['val']['groups']['ALL']['actual_regret_pct'],
                binary_val_ce=row['binary_val']['macro']['ce'], binary_val_accuracy=row['binary_val']['macro']['accuracy'],
                binary_train_ce=row['binary_train']['macro']['ce'] if row['binary_train'] else '',
                binary_train_accuracy=row['binary_train']['macro']['accuracy'] if row['binary_train'] else ''))
    write_csv(root/'learning_curves.csv', curve_rows)
    plot_curves(root, histories)
    screens = {}
    for split in ('val', 'test'):
        base = replay[split]['R45A']['groups']['ALL']
        screens[split] = {}
        for arm in ('A', 'B'):
            current = replay[split]['R53'+arm]['groups']['ALL']
            top1_delta = (current['top1']-base['top1'])*100
            decrease = (base['actual_regret_pct']-current['actual_regret_pct'])/base['actual_regret_pct']*100
            screens[split][arm] = dict(top1_delta_pp=top1_delta, regret_relative_reduction_pct=decrease,
                meets_predeclared_screen=bool((top1_delta >= 5 and decrease >= 0) or (decrease >= 15 and top1_delta >= 0)))
    dump(root/'screen.json', screens)
    lines = ['# R53: Independent MOEL/MTL specialists', '',
        '## Protocol',
        'Frozen R45A supplies the full ranking. Independent four-layer, d=128 encoders were trained from scratch on all 15 MVRP tasks (seed=2). Only original Top2={MOEL,MTL} decisions can change. No solver features/ID, retrieval, solver execution, risk, pairwise objective or R-Drop enters the expert.', '',
        'A uses unweighted BCE; B uses actual pair misselection cost divided by a fixed training-only mean. Both include all non-tied instances, including those with another full-pool winner.',
        f"Fixed global mean weight: {protocol['training_weight_mean']:.12g}. A/B initialization hashes match; sampling plans match. Schedules follow the same rules, not necessarily the same actual LR trajectory or stop epoch.", '',
        'Selection uses the strict minimum **integrated full18 validation actual regret**; plateau/early-stop min_delta=0.001 percentage points, warmup=3, min15/max40 epochs. Both best checkpoints were locked before test. Test never selects an epoch.', '',
        'The [SATzilla2012 solver description](https://www.cs.ubc.ca/~kevinlb/papers/2012-SATzilla2012-Solver-Description.pdf) motivates independent cost-sensitive pair classification. This experiment does not reproduce its forests, portfolio voting or runtime prediction.', '',
        '## Pretraining Gate Budget (Validation)',
        f"Gate: {budget['triggered']}/{budget['n']} instances. Recoverable native-winner errors: {budget['correctable_errors']}; all MOEL/MTL mutual errors: {budget['all_pair_errors']}, gate-covered: {budget['gated_pair_errors']}.",
        f"Diagnostic upper bound (not a deployable score): +{budget['diagnostic_top1_recovery_pp']:.4f} Top1 points and -{budget['diagnostic_regret_recovery_pp']:.6f} regret points ({budget['diagnostic_regret_recovery_relative_pct']:.2f}% relative reduction).", '',
        '## Integrated Results',
        '| Split | Model | Best epoch | Top1 % | Mean cost | Actual regret % | Delta Top1 pp | Regret reduction % |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for split in ('val', 'test'):
        for name in ('R45A', 'R53A', 'R53B'):
            m = replay[split][name]['groups']['ALL']
            arm = name[-1]
            epoch = protocol['frozen_baseline']['human_epoch'] if name == 'R45A' else results[arm]['best_epoch']
            screen = dict(top1_delta_pp=0, regret_relative_reduction_pct=0) if name == 'R45A' else screens[split][arm]
            lines.append(f"| {split} | {name} | {epoch} | {m['top1']*100:.4f} | {m['mean_cost']:.6f} | {m['actual_regret_pct']:.6f} | {screen['top1_delta_pp']:+.4f} | {screen['regret_relative_reduction_pct']:+.3f} |")
    lines += ['', '## Corrections and Harms']
    for split in ('val', 'test'):
        for arm in ('A', 'B'):
            selected = [r for r in correction_rows if r['split'] == split and r['arm'] == arm]
            counts = {key:sum(r['count'] for r in selected if r['category'] == key) for key in
                      ('corrected', 'harmed', 'both_wrong_cheaper', 'both_wrong_more_expensive')}
            net_cost = sum(r['mean_cost_delta_contribution'] for r in selected)/len(PROBLEMS)
            net_regret = sum(r['macro_regret_delta_contribution_pp'] for r in selected)
            improved_problems = sum(replay[split]['R53'+arm]['per_problem'][p]['actual_regret_pct'] <
                                    replay[split]['R45A']['per_problem'][p]['actual_regret_pct'] for p in MVRP)
            lines.append(f"- {split} R53{arm}: corrected {counts['corrected']}, harmed {counts['harmed']}; both-wrong cheaper/more expensive {counts['both_wrong_cheaper']}/{counts['both_wrong_more_expensive']}; net mean-cost delta {net_cost:+.8f}, regret delta {net_regret:+.6f} pp; cost improved on {improved_problems}/15 MVRP tasks.")
    lines += ['', '## Stopping and Generalization']
    for arm, result in results.items():
        best_train, best_val = result['best_binary_train']['macro'], result['best_binary_val']['macro']
        last = result['final']
        lines.append(f"- R53{arm}: best epoch {result['best_epoch']}, stop {result['final_epoch']}, successful updates {result['successful_updates']}. At best: expert train/val accuracy {best_train['accuracy']*100:.3f}%/{best_val['accuracy']*100:.3f}%, unweighted BCE {best_train['ce']:.6f}/{best_val['ce']:.6f}. Last5 integrated val: Top1 {result['last5_val']['top1']*100:.4f}%, regret {result['last5_val']['actual_regret_pct']:.6f}%.")
    passed = [f"{split} R53{arm}" for split in ('val', 'test') for arm in ('A', 'B') if screens[split][arm]['meets_predeclared_screen']]
    lines += ['', '## Conclusion',
        ('Predeclared screen passed: '+', '.join(passed)+'. Examine validation and test jointly before further investment.' if passed else
         'Neither expert meets the predeclared meaningful-improvement screen. Independent cross-problem pair learning and this cost-weighted objective did not produce the intended system-level breakthrough; do not expand this configuration into further seeds or threshold searches.'), '',
        'This is a fixed-seed direction screen, not proof that the pair is unpredictable or that labels are noisy. The gate limits what can improve. Top2/Top3 coverage is unchanged by construction; ranking-only combined outputs are not treated as calibrated probabilities or assigned a fabricated full-system CE.', '',
        '## Artifacts and Runtime',
        'See comparison.csv (ALL/MVRP/all18), binary_comparison.csv, corrections_and_harms.csv, gate_budget.csv, learning_curves.csv/png, instance_decisions.csv, locked_checkpoints.json and each arm predictions/{val,test}. Checkpoint and sampling-plan .pt files remain local, following existing repository rules.',
        'Inference adds a second instance encoder only where the fixed gate triggers. Recorded specialist-only timings do not pretend that cached baseline predictions are a free end-to-end deployment. No underlying solver is executed.', '',
        'W&B logs are retained locally in offline mode (project selector); no unrelated historical artifacts were changed.']
    (root/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(screens=screens, initialization_matched=True, prediction_replay=True), indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    analyze(parser.parse_args().root)

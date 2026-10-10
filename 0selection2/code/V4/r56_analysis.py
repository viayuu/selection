"""Replay R56 full-system decisions against fixed historical references."""

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
from .r53_analysis import changes
from .r53_data import MVRP, baseline_metrics, read_baseline


ROOT = Path('code/V4/runs/R56_relational_encoder')
ARM = 'relational_seed2'
HISTORICAL = dict(R53A=Path('code/V4/runs/R53_pair_specialist/A_ce_seed2'),
                  R54=Path('code/V4/runs/R54_route_supervision/route_supervised_seed2'))


def plot(root, history):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    x = [r['successful_updates'] for r in history]
    train = [r for r in history if r['binary_train']]
    for ax, key, factor, title in ((axes[0, 0], 'ce', 1, 'Expert unweighted BCE'),
            (axes[0, 1], 'accuracy', 100, 'Expert pair accuracy (%)'),
            (axes[0, 2], 'pair_regret_pct', 1, 'Expert pair regret (%)')):
        ax.plot(x, [r['binary_val']['macro'][key]*factor for r in history], color='#007c91', label='val')
        ax.plot([r['successful_updates'] for r in train],
                [r['binary_train']['macro'][key]*factor for r in train], '--', color='#bb4c2d', label='full train_eval')
        ax.set_title(title)
    baseline = json.loads((root/'gate_budget.json').read_text())['baseline']
    for ax, key, factor, title in ((axes[1, 0], 'top1', 100, 'Integrated 18-task val Top1 (%)'),
            (axes[1, 1], 'actual_regret_pct', 1, 'Integrated 18-task val actual regret (%)')):
        ax.plot(x, [r['val']['groups']['ALL'][key]*factor for r in history], color='#007c91', label='R56')
        ax.axhline(baseline[key]*factor, linestyle=':', color='#555555', label='R45A')
        ax.set_title(title)
    axes[1, 2].plot(x, [r['learning_rate'] for r in history], label='LR', color='#007c91')
    axes[1, 2].set_title('Learning rate')
    for ax in axes.flat:
        ax.grid(alpha=.2)
        ax.set_xlabel('Successful optimizer updates')
        ax.legend(fontsize=8)
    fig.savefig(root/'learning_curves.png', dpi=160)
    plt.close(fig)


def analyze(root=ROOT):
    root = Path(root)
    directory = root/ARM
    result = json.loads((directory/'result.json').read_text())
    history = json.loads((directory/'history.json').read_text())
    test = json.loads((directory/'test_result.json').read_text())
    protocol = json.loads((root/'protocol.json').read_text())
    latency = json.loads((root/'inference_latency.json').read_text())
    rows, corrections, binary_rows, instance_rows, selection_rows = [], [], [], [], []
    replay, screens = {}, {}
    for split in ('val', 'test'):
        raw, baseline = read_baseline(split, root)
        current = result['best_validation'] if split == 'val' else test['integrated']
        reference = baseline_metrics(raw, baseline)
        all_models = dict(R45A=reference, R56=current)
        for name, historical in HISTORICAL.items():
            if (historical/'result.json').exists():
                old = json.loads((historical/('result.json' if split == 'val' else 'test_result.json')).read_text())
                all_models[name] = old['best_validation'] if split == 'val' else old['integrated']
        for name, measured in all_models.items():
            for scope, values in dict(ALL=measured['groups']['ALL'], MVRP=measured['groups']['MVRP'],
                                       **measured['per_problem']).items():
                rows.append(dict(split=split, model=name, scope=scope, top1_pct=values['top1']*100,
                    top2_pct=values['top2']*100, top3_pct=values['top3']*100, mean_cost=values['mean_cost'],
                    vs_sbs_pct=values['vs_sbs_pct'], vs_oracle_pct=values['vs_oracle_pct'],
                    actual_regret_pct=values['actual_regret_pct']))
        for p in PROBLEMS:
            with np.load(directory/'predictions'/split/(p+'.npz'), allow_pickle=False) as saved:
                np.testing.assert_array_equal(saved['costs'], raw[p]['costs'])
                np.testing.assert_array_equal(saved['winner'], raw[p]['winner'])
                metrics, pred = decision_metrics(ordinal_scores(saved['order']), raw[p])
                np.testing.assert_array_equal(pred, saved['pred'])
                for key in ('top1', 'top2', 'top3', 'mean_cost', 'actual_regret_pct'):
                    np.testing.assert_allclose(metrics[key], current['per_problem'][p][key], rtol=0, atol=1e-10)
                corrections.extend(changes(saved, p, split, 'R56'))
                for solver, fraction in metrics['arm_distribution'].items():
                    selection_rows.append(dict(split=split, problem=p, solver=solver, selected_pct=fraction*100))
                for i in range(len(pred)):
                    instance_rows.append(dict(split=split, problem=p, instance_index=i,
                        gate=bool(saved['gate'][i]), expert_logit=float(saved['expert_logit'][i]),
                        baseline=raw[p]['pool'][int(saved['baseline_pred'][i])],
                        selected=raw[p]['pool'][int(pred[i])], winner=raw[p]['pool'][int(raw[p]['winner'][i])],
                        selected_cost=float(raw[p]['costs'][i, pred[i]]),
                        baseline_cost=float(raw[p]['costs'][i, saved['baseline_pred'][i]])))
        base, new = reference['groups']['ALL'], current['groups']['ALL']
        delta = (new['top1']-base['top1'])*100
        reduction = (1-new['actual_regret_pct']/base['actual_regret_pct'])*100
        screens[split] = dict(top1_delta_pp=delta, regret_relative_reduction_pct=reduction,
            meets_predeclared_screen=bool((delta >= 5 and reduction >= 0) or (reduction >= 15 and delta >= 0)))
        replay[split] = all_models
    for split, summary in (('train', result['best_binary_train']), ('val', result['best_binary_val']), ('test', test['binary'])):
        for scope, values in dict(MVRP=summary['macro'], **summary['per_problem']).items():
            binary_rows.append(dict(model='R56', split=split, scope=scope, **values))
    for name, historical in HISTORICAL.items():
        if not (historical/'result.json').exists():
            continue
        r = json.loads((historical/'result.json').read_text())
        t = json.loads((historical/'test_result.json').read_text())
        for split, summary in (('train', r['best_binary_train']), ('val', r['best_binary_val']), ('test', t['binary'])):
            for scope, values in dict(MVRP=summary['macro'], **summary['per_problem']).items():
                binary_rows.append(dict(model=name, split=split, scope=scope, **values))
    write_csv(root/'comparison.csv', rows)
    write_csv(root/'binary_comparison.csv', binary_rows)
    write_csv(root/'corrections_and_harms.csv', corrections)
    write_csv(root/'instance_decisions.csv', instance_rows)
    write_csv(root/'solver_selection.csv', selection_rows)
    curve = []
    for row in history:
        curve.append(dict(epoch=row['epoch'], successful_updates=row['successful_updates'],
            training_objective=row['loss'], grad_norm=row['grad_norm'], learning_rate=row['learning_rate'],
            amp_retries=row['amp_retries'], epoch_seconds=row['epoch_seconds'],
            val_top1_pct=row['val']['groups']['ALL']['top1']*100,
            val_actual_regret_pct=row['val']['groups']['ALL']['actual_regret_pct'],
            binary_val_ce=row['binary_val']['macro']['ce'], binary_val_accuracy=row['binary_val']['macro']['accuracy'],
            binary_train_ce=row['binary_train']['macro']['ce'] if row['binary_train'] else '',
            binary_train_accuracy=row['binary_train']['macro']['accuracy'] if row['binary_train'] else ''))
    write_csv(root/'learning_curves.csv', curve)
    plot(root, history)
    dump(root/'screen.json', screens)
    dump(root/'prediction_replay.json', dict(all18_metrics_reproduced=True, cost_precision='FP64'))
    lines = ['# R56: Directed Node-Relation Encoder', '', '## Protocol',
        'One seed=2 specialist is trained from scratch on all 15 MVRP tasks, with ordinary unweighted BCE. '
        'The original mean/max pooling and binary head remain; the original four node-only layers and fixed distance bias '
        'are replaced by four d128/e32/four-head joint node/relation layers. '
        'Fixed R45A supplies the ranking; only original Top2={MOEL,MTL} decisions may change.', '',
        'No solver reruns, new labels, R54 routes, R55 edge targets, source embeddings, retrieval, solver pretraining, '
        'behavior probe, regression, weighted BCE, risk or R-Drop are used. Original FP64 costs determine pair labels '
        'and evaluation. All non-ties are trained, including instances won by another full-pool solver.', '',
        '## EGT Adoption and Limits',
        '[EGT, KDD 2022, section3.2 equations3-7](https://arxiv.org/html/2108.03348v3) motivates persistent edge channels, '
        'clipped node QK interactions plus learned edge bias, post-softmax sigmoid gates and edge updates from '
        'the pre-softmax multihead interaction. Both streams have pre-normalization, residuals and FFNs. '
        'This is EGT-inspired, not a full reproduction.',
        'Relations use the original ordered endpoint attributes (including coordinates, demand, depot, route limit, '
        'service and time windows), raw/mean-normalized distance, original constraints and real scale. '
        'Independent source/target projections allow E_ij != E_ji. '
        'All valid ordered pairs, self-pairs and the original three special tokens are retained; only padding is masked. '
        'No local-feasibility heuristic or label screens relations.',
        'We use GELU/probability dropout0.1 rather than ELU/random attention masking; dynamic centrality scalers and '
        'SVD encodings are omitted. Terminal edge-only updates are computed for diagnosis but are not supervised '
        'by a node-only graph readout. Earlier edge residual updates and all four bias/gate projections do receive BCE gradients; '
        'no extra edge pooling or auxiliary head is added to hide this boundary.', '',
        '## Integrated Results', '| Split | Model | Top1 % | Mean cost | Actual regret % |', '|---|---|---:|---:|---:|']
    for split, models in replay.items():
        for name in ('R45A', 'R53A', 'R54', 'R56'):
            if name in models:
                r = models[name]['groups']['ALL']
                lines.append(f"| {split} | {name} | {r['top1']*100:.4f} | {r['mean_cost']:.6f} | {r['actual_regret_pct']:.6f} |")
    lines += ['', 'R53A is a historical reference, not a newly rerun matched control. '
        'All percentages retain equal problem weighting; ALL raw mean-cost is not used to reconstruct macro ratios.', '',
        '## Training and Selection',
        f"Best epoch {result['best_epoch']}; final epoch {result['final_epoch']}; {result['successful_updates']} successful updates. "
        f"Last5 validation Top1 {result['last5_val']['top1']*100:.4f}%, actual regret {result['last5_val']['actual_regret_pct']:.6f}%.",
        'R53A batch128, AdamW LR1e-4/WD1e-4, dropout0.1, clip1, warmup3 and min15/max40 control are reused. '
        'All forty pre-generated epoch/task/index hashes match the R53 plan. '
        'Every successful epoch covers all non-tied training instances including tails. '
        'AMP overflow retries the same batch/RNG without consuming another planned update.',
        'Checkpoint selection uses any strict new minimum full18 validation actual regret; '
        'LR/early stopping require improvement of at least0.001 percentage points. '
        'A tiny strict improvement need not reset the plateau counter. One best hash is locked before the one full test.', '',
        '## Specialist Generalization']
    for split, summary in (('train', result['best_binary_train']), ('val', result['best_binary_val']), ('test', test['binary'])):
        r = summary['macro']
        lines.append(f"- {split}: pair accuracy {r['accuracy']*100:.4f}%, BCE {r['ce']:.6f}, two-method regret {r['pair_regret_pct']:.6f}%.")
    lines += ['', '## Corrections and Harms']
    for split in ('val', 'test'):
        selected = [r for r in corrections if r['split'] == split]
        counts = {category:sum(r['count'] for r in selected if r['category'] == category)
                  for category in ('corrected', 'harmed', 'both_wrong_cheaper', 'both_wrong_more_expensive')}
        net = sum(r['macro_regret_delta_contribution_pp'] for r in selected)
        improved = sum(replay[split]['R56']['per_problem'][p]['actual_regret_pct'] <
                       replay[split]['R45A']['per_problem'][p]['actual_regret_pct'] for p in MVRP)
        lines.append(f"- {split} vs R45A: corrected {counts['corrected']}, harmed {counts['harmed']}; "
            f"both-wrong cheaper/more expensive {counts['both_wrong_cheaper']}/{counts['both_wrong_more_expensive']}; "
            f"net macro regret change {net:+.6f} pp; {improved}/15 MVRP tasks improve regret.")
    lines += ['', '## Inference Cost',
        f"All15000 test expert forwards: {test['specialist_all_15000_forward_seconds']:.3f} seconds. "
        f"Synchronized same-batch FP32 benchmark, batch{latency['batch']}/max nodes{latency['max_nodes']}: "
        f"R53A {latency['milliseconds']['R53A']:.3f}ms, R56 {latency['milliseconds']['R56']:.3f}ms "
        f"({latency['relative_overhead_pct']:+.1f}%).",
        'These include each expert Encoder/head, not the common frozen baseline ranking, data I/O or solver run. '
        'The whole test timing includes ungated examples for reporting; deployment only needs the expert on triggered inputs. '
        'One batch benchmark is not a full-node-size or full-system latency claim.', '', '## Conclusion']
    for split, screen in screens.items():
        lines.append(f"- {split} vs R45A: Top1 {screen['top1_delta_pp']:+.4f}pp; regret relative reduction "
            f"{screen['regret_relative_reduction_pct']:+.3f}%; predeclared advancement screen: {screen['meets_predeclared_screen']}.")
    if not all(s['meets_predeclared_screen'] for s in screens.values()):
        lines.append('This single configuration did not meet the joint validation/test meaningful-improvement screen. '
                     'Do not claim success from evolving edges or training accuracy; no extra seeds/architectures were started.')
    else:
        lines.append('The predeclared screen is met; matched independent replication is still needed before a broad mechanism claim.')
    lines += ['This does not prove a hard accuracy ceiling, random labels or the absence of useful instance information.', '',
        '## Artifacts', 'See protocol.json, precheck.json, runtime.json, sampling_plan.json, training_counts.csv, '
        'comparison.csv (ALL/MVRP/all18), binary_comparison.csv, corrections_and_harms.csv, '
        'learning_curves.csv/png, inference_latency.json, locked_checkpoint.json and per-instance predictions/{val,test}. '
        'Checkpoints, sampling plan tensors and offline W&B logs stay local. Historical models/data are unchanged.']
    (root/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(screens=screens, full18_prediction_replay=True), indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    analyze(parser.parse_args().root)

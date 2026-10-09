"""R48 train/validation-only conclusions and replayable binary metrics."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from .multitask_probe import dump
from .r48_common import ROOT, PAIR, binary_metrics, close_cost, load_split, read_csv, write_csv


def analyze(root):
    audit = json.loads((root/'solver_audit_summary.json').read_text())
    solver_rows = read_csv(root/'solver_audit.csv')
    originals = {(r['split'],int(r['instance_index']),r['solver'],r['variant']):r for r in solver_rows}
    repeat_rows=[]
    for split in ('train','val'):
        indices=sorted({int(r['instance_index']) for r in solver_rows if r['split']==split})
        for name in PAIR:
            changes=[]
            for index in indices:
                first=originals.get((split,index,name,'original_0'))
                second=originals.get((split,index,name,'original_1'))
                if first and second:
                    changes.append(float(second['rerun_cost'])-float(first['rerun_cost']))
            if changes:
                repeat_rows.append(dict(split=split,solver=name,instances=len(changes),
                    exact_cost_changes=int(np.count_nonzero(changes)),max_abs_cost_change=float(np.abs(changes).max())))
    write_csv(root/'original_repeat_summary.csv',repeat_rows)
    lines=['# R48: core two-solver diagnosis','',
        'Scope: OVRPTW, RELD_MOEL (class0) versus RELD_MTL (class1), seed2. Train/val only.',
        'No main-model edit, no 18-task retraining, no test evaluation. Only the authorized idle RTX3090 was used.','',
        '## 1. Historical labels and independent route validation','',
        f'Captured route rows: {audit["rows"]}; expected: {audit["expected_rows"]}. '
        f'Blocking anomalies: {audit["anomaly_count"]}.',
        f'Maximum absolute historical-cost error on original inputs: {audit["max_abs_historical_error"]}.',
        f'Maximum absolute independent FP64 route-cost error versus solver scalar: {audit["max_abs_recomputed_error"]}.','',
        'The exact R41 two solver checkpoints, argmax decoding, all-customer POMO starts, no augmentation, '
        'sample_size=1 and minimum over starts are retained. All six solver input fields are checked after '
        'the historical FP32 conversion. Actual routes, winning start IDs and customer remappings are saved in solver_runs/.',
        'The independent NumPy checker checks each customer exactly once, normalized capacity=1, speed=1, '
        'waiting/service/customer time windows, and open route distance. It does not charge customer-to-depot '
        'returns or treat the depot placeholder time window as a real deadline. Feasibility tolerance=1e-5; '
        'cost agreement uses atol=2e-5, rtol=2e-6. Both original-input and runtime-FP32-input FP64 costs are recorded.','',
        '| Split | Solver | Repeated instances | Exact scalar changes | Maximum absolute change |',
        '|---|---|---:|---:|---:|']
    for row in repeat_rows:
        lines.append(f'| {row["split"]} | {row["solver"]} | {row["instances"]} | {row["exact_cost_changes"]} | {row["max_abs_cost_change"]:.9g} |')
    lines += ['', 'Checkpoint/path/config evidence applies to the pinned, locally available R41 weights; '
        'the unrecorded original historical checkpoint digest cannot be retroactively recovered. Matching reruns '
        'support the sampled rows, not a claim that every historical solver deployment has been audited.']
    if (root/'permutation_audit.csv').exists():
        permutation_rows=read_csv(root/'permutation_audit.csv')
        summaries=[]
        for split in ('train','val'):
            selected=[r for r in permutation_rows if r['split']==split and r['variant'].startswith('permutation')]
            if not selected:
                continue
            valid=[r for r in selected if int(r['historical_binary_label'])>=0]
            flips=[r for r in valid if int(r['rerun_binary_winner'])>=0 and int(r['rerun_binary_winner'])!=int(r['historical_binary_label'])]
            changed_prediction=sum(int(r['R47_binary_prediction']) != int(
                next(x for x in permutation_rows if x['split']==split and x['instance_index']==r['instance_index']
                     and x['variant']=='original_0')['R47_binary_prediction']) for r in selected)
            summary=dict(split=split,instances=len({r['instance_index'] for r in selected}),permutations=len(selected),
                strict_winner_flips=len(flips),strict_non_tie_originals=len(valid),
                examples_with_strict_flip=len({r['instance_index'] for r in flips}),
                new_exact_ties=sum(int(r['rerun_binary_winner'])<0 for r in selected),
                R47_binary_prediction_changes=changed_prediction,
                R47_max_abs_score_change=max(float(r['R47_max_score_change_from_original']) for r in selected),
                R47_max_abs_margin_change=max(abs(float(r['R47_margin_change_from_original'])) for r in selected))
            for name,suffix in zip(PAIR,('MOEL','MTL')):
                changes=np.array([float(r['rerun_'+suffix])-float(r['historical_'+suffix]) for r in selected])
                summary[suffix+'_exact_cost_changes']=int(np.count_nonzero(changes))
                summary[suffix+'_material_cost_changes']=sum(not close_cost(float(r['rerun_'+suffix]),float(r['historical_'+suffix])) for r in selected)
                summary[suffix+'_mean_abs_cost_change']=float(np.abs(changes).mean())
                summary[suffix+'_max_abs_cost_change']=float(np.abs(changes).max())
            summaries.append(summary)
        write_csv(root/'permutation_summary.csv',summaries)
        lines += ['', '## 2. Customer-node permutation','',
            'Four fixed customer permutations per R41 instance; depot stays fixed and demand/service/time-window fields '
            'move with coordinates. Routes are mapped back to original customer IDs before the same independent check.',
            'Frozen R47 keeps the original full seven-method pool and train-only memory. Its two actual maximin scores '
            'are compared without rebuilding a smaller candidate pool. No solver labels enter its query inputs.','',
            '| Split | Instances | Permutations | Strict winner flips | Instances with flips | R47 binary changes | Max score change |',
            '|---|---:|---:|---:|---:|---:|---:|']
        for row in summaries:
            lines.append(f'| {row["split"]} | {row["instances"]} | {row["permutations"]} | {row["strict_winner_flips"]} | '
                         f'{row["examples_with_strict_flip"]} | {row["R47_binary_prediction_changes"]} | {row["R47_max_abs_score_change"]:.9g} |')
        lines += ['', 'Exact scalar differences and changes beyond the predeclared FP32 tolerance are separately saved '
            'in permutation_summary.csv. A strict winner flip caused by numerical closeness is not silently relabeled as a tie. '
            'Order-sensitive solver outcomes are recorded as behavior, not automatically classified as implementation bugs.']
    else:
        summaries=[]
    if not (root/'binary_complete.json').exists():
        lines += ['', '## Binary experiment not executed','',
                  'The solver audit did not clear training, or binary execution is incomplete. No synthetic result is substituted.']
        (root/'RESULTS.md').write_text('\n'.join(lines)+'\n')
        return
    config=json.loads((root/'config.json').read_text())
    selection=json.loads((root/'neural_seed2/selection.json').read_text())
    metrics=read_csv(root/'binary_comparison.csv')
    indexed={(r['model'],r['split']):r for r in metrics}
    lines += ['', '## 3. Clean binary learning','',
        'All original FP64 non-ties are used, even if neither RELD solver is the seven-pool winner. '
        'Exact ties are retained in predictions with label=-1 but excluded from fitting and binary metrics.',
        'Regret here is relative to min(MOEL_cost, MTL_cost), not the seven-method Oracle. '
        'Binary accuracy is not comparable to the old seven-method Top1.','',
        '| Split | Total | Strict comparisons | Exact ties | MOEL wins | MTL wins |',
        '|---|---:|---:|---:|---:|---:|']
    for split,counts in config['counts'].items():
        lines.append(f'| {split} | {counts["total"]} | {counts["non_ties"]} | {counts["ties"]} | {counts["class0"]} | {counts["class1"]} |')
    lines += ['', '| Model | Split | N | Binary accuracy | CE | Selected cost | Two-method regret |',
              '|---|---|---:|---:|---:|---:|---:|']
    for row in metrics:
        lines.append(f'| {row["model"]} | {row["split"]} | {row["n"]} | {float(row["accuracy"])*100:.3f}% | '
                     f'{float(row["ce"]):.6f} | {float(row["mean_cost"]):.6f} | {float(row["pair_regret_pct"]):.6f}% |')
    lines += ['', f'Neural stop epoch={selection["stop_epoch"]}; validation-CE-best epoch={selection["best_epoch"]}; '
        f'successful updates={selection["successful_updates"]}. All table entries for the neural model replay the same best checkpoint.',
        'Neural protocol: scratch Encoder + binary head, CE only, batch128, fixed LR1e-4, AdamW WD1e-4, '
        'dropout0.1, FP32/TF32 off, at most60 epochs, full train/val eval every epoch, strict validation CE selection '
        'and ten nonimproving epochs to stop. No solver encoder, retrieval, pair/risk/R-Drop or augmentation.',
        'Tree protocol: one fixed HistGradientBoosting configuration, 300 iterations and 15 leaves, no parameter sweep '
        'or validation selection. Input statistics are explicitly listed in config.json and r48_common.tree_features. '
        'Size-majority probabilities are train-fitted with add-one smoothing solely to make CE finite.']
    replay=[]
    for split in ('train','val'):
        predictions=read_csv(root/(split+'_predictions.csv'))
        _,raw,costs,_=load_split(split)
        for name in ('neural','tree','size_prior','frozen_R47'):
            scores=np.array([[float(r[name+'_score0']),float(r[name+'_score1'])] for r in predictions])
            result=binary_metrics(scores,costs)
            errors={k:abs(float(result[k])-float(indexed[(name,split)][k])) for k in result}
            if max(errors.values())>1e-10:
                raise ValueError('Saved CSV does not reproduce binary metrics')
            replay.append(dict(model=name,split=split,max_metric_error=max(errors.values())))
    dump(root/'prediction_replay.json',replay)
    history=json.loads((root/'neural_seed2/history.json').read_text())
    plot_curves(root,history,indexed,selection['best_epoch'])
    lines += ['', '## Findings and limits','']
    if audit['anomaly_count']==0:
        lines.append('1. No input/constraint mismatch, missing route, infeasible selected route or material cost-calculation '
                     'error was found in this fixed audit. This is sample-level evidence, not proof about all labels.')
    if summaries:
        flips=sum(r['strict_winner_flips'] for r in summaries)
        total=sum(r['permutations'] for r in summaries)
        lines.append(f'2. Customer reorderings changed the strict two-solver winner in {flips}/{total} trials. '
                     'See per-instance costs and score changes rather than inferring a data error from a flip alone.')
    prior_acc=float(indexed[('size_prior','val')]['accuracy'])
    r47_acc=float(indexed[('frozen_R47','val')]['accuracy'])
    for number,name in enumerate(('neural','tree'),start=3):
        row=indexed[(name,'val')]
        lines.append(f'{number}. {name} validation accuracy={float(row["accuracy"])*100:.3f}%; '
            f'difference versus size prior={(float(row["accuracy"])-prior_acc)*100:+.3f}pp, '
            f'versus frozen R47={(float(row["accuracy"])-r47_acc)*100:+.3f}pp; '
            f'CE={float(row["ce"]):.6f}, two-method regret={float(row["pair_regret_pct"]):.6f}%. '
            'Accuracy, probability quality and selection cost must be read together.')
    lines += ['5. This diagnostic removes seven-way competition, shared-task updates and auxiliary losses. '
        'It cannot by itself prove random labels, a universal predictability ceiling, or that one architecture is optimal. '
        'The training/validation curves and an independently constructed tree provide separate evidence about the remaining signal.',
        '', '## Execution and artifacts','',
        'Actual compute route: srun --jobid=1465 --overlap --exact --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=10G; '
        'only GPU-cb24f168-9f18-f0d5-6f4a-6d2406958f50 (RTX3090) is visible to each process.',
        '```bash',
        'source /public/home/shiys/miniconda3/etc/profile.d/conda.sh',
        'conda activate easynco',
        'cd /public/home/shiys/0selection2',
        'R48_GPU_UUID=GPU-cb24f168-9f18-f0d5-6f4a-6d2406958f50 bash code/V4/run_v4_r48.sh audit',
        'R48_GPU_UUID=GPU-cb24f168-9f18-f0d5-6f4a-6d2406958f50 bash code/V4/run_v4_r48.sh binary',
        'R48_GPU_UUID=GPU-cb24f168-9f18-f0d5-6f4a-6d2406958f50 bash code/V4/run_v4_r48.sh analysis',
        '```','',
        'Required outputs: solver_audit.csv, permutation_audit.csv, binary_comparison.csv, learning_curves.png, '
        'val_predictions.csv, config.json. Additional train predictions, route JSONL, audit/permutation summaries, '
        'best/last neural checkpoints, tree pickle, exact sample/permutations and launch sources are retained.',
        'The first precheck exposed a duplicate-key result-recording error, fixed before any accepted solver output. '
        'It was a diagnostic harness error, not a solver/data anomaly. Logs retain the failed launch.',
        'All saved binary prediction CSVs replay their metrics to numerical tolerance. W&B logs are offline; '
        'no API keys or external embedding requests were used.']
    (root/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    dump(root/'completion.json',dict(complete=True,test_read=False,neural_best_epoch=selection['best_epoch'],
        neural_stop_epoch=selection['stop_epoch'],solver_audit=audit,permutation_summary=summaries))
    print('\n'.join(lines[-30:]),flush=True)


def plot_curves(root,history,indexed,best_epoch):
    fig,axes=plt.subplots(1,2,figsize=(13,4.4))
    epochs=[r['epoch'] for r in history]
    for axis,key,title,scale in ((axes[0],'ce','Unweighted binary CE',1.),
                                  (axes[1],'accuracy','Binary accuracy (Top1)',100.)):
        for split,color in (('train','#2879b9'),('val','#d45142')):
            axis.plot(epochs,[r[split][key]*scale for r in history],label='Neural '+split,color=color,lw=1.8)
        for name,color in (('size_prior','#787878'),('frozen_R47','#3c9369'),('tree','#9d72b2')):
            axis.axhline(float(indexed[(name,'val')][key])*scale,label=name+' val',color=color,ls=':',lw=1.2)
        axis.axvline(best_epoch,color='#444444',ls='--',lw=.8,label='Best val CE epoch')
        axis.set(title=title,xlabel='Epoch',ylabel='Accuracy (%)' if key=='accuracy' else 'CE')
        axis.grid(alpha=.2)
        axis.legend(fontsize=8)
    fig.suptitle('R48: OVRPTW MOEL versus MTL; exact cost ties excluded')
    fig.tight_layout()
    fig.savefig(root/'learning_curves.png',dpi=180)
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,default=ROOT)
    args=parser.parse_args()
    analyze(args.root)


if __name__=='__main__':
    main()

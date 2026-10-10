"""R49 fixed-checkpoint comparisons, distribution summaries and update curves."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from .multitask_probe import dump, file_hash
from .r48_common import binary_metrics, read_csv, write_csv
from .r49_data import ROOT, MODELS


def instance_metrics(archive):
    scores=archive['scores'].astype(np.float64)
    labels=archive['labels']
    keep=labels>=0
    stable=scores-scores.max(1,keepdims=True)
    logp=stable-np.log(np.exp(stable).sum(1,keepdims=True))
    costs=archive['costs']
    pred=scores.argmax(1)
    return np.stack([(pred[keep]==labels[keep]).astype(float)*100,
        -logp[keep,labels[keep]],
        (costs[np.arange(len(costs)),pred]/costs.min(1)-1)[keep]*100],axis=1)


def bootstrap_difference(first,second,paired,seed):
    rng=np.random.default_rng(seed)
    draws=[]
    for _ in range(2000):
        left=rng.integers(len(first),size=len(first))
        right=left if paired else rng.integers(len(second),size=len(second))
        draws.append(first[left].mean(0)-second[right].mean(0))
    interval=np.quantile(draws,[.025,.975],axis=0)
    delta=first.mean(0)-second.mean(0)
    return [dict(metric=metric,difference=float(delta[j]),lower95=float(interval[0,j]),upper95=float(interval[1,j]),
                 paired=paired,unit='percentage_points' if metric!='ce' else 'unweighted CE')
        for j,metric in enumerate(('accuracy','ce','pair_regret_pct'))]


def plot_curves(root,histories,selections):
    fig,axes=plt.subplots(2,3,figsize=(16,8),sharex=True)
    labels=dict(train='Train subset',development='Development (selection)',internal='Internal holdout',original_val='Original validation')
    colors=dict(train='#2879b9',development='#d99b22',internal='#21836c',original_val='#ce4e4b')
    for i,name in enumerate(MODELS):
        history=histories[name]
        x=[r['updates'] for r in history]
        for j,(field,title,scale) in enumerate((('ce','Unweighted CE',1),('accuracy','Binary accuracy (%)',100),('pair_regret_pct','Two-method regret (%)',1))):
            ax=axes[i,j]
            for split in labels:
                ax.plot(x,[r['metrics'][split][field]*scale for r in history],color=colors[split],lw=1.6,label=labels[split])
            ax.axvline(selections[name]['best_updates'],color='#444444',ls='--',lw=.9,label='Best development CE')
            ax.set_title(name+': '+title,fontsize=11)
            ax.set_xlabel('Successful optimizer updates')
            ax.grid(alpha=.2)
            ax.legend(fontsize=8)
    fig.suptitle('R49: scratch OVRPTW MOEL vs MTL; 2,000 vs 8,000 existing instances',fontsize=13)
    fig.tight_layout()
    fig.savefig(root/'learning_curves.png',dpi=180)
    plt.close(fig)


def analyze(root):
    config=json.loads((root/'config.json').read_text())
    manifest=json.loads((root/'split_manifest.json').read_text())
    histories={name:json.loads((root/name/'history.json').read_text()) for name in MODELS}
    selections={name:json.loads((root/name/'selection.json').read_text()) for name in MODELS}
    archives={name:{split:dict(np.load(root/name/(split+'_predictions.npz')))
        for split in ('train','development','internal','original_val')} for name in MODELS}
    execution_checks={}
    if file_hash(root/'split_manifest.json')!=config['split_manifest_sha256']:
        raise ValueError('Locked split manifest changed after launch')
    for name in MODELS:
        train_key='A_2000' if name==MODELS[0] else 'training_pool'
        nominal=np.asarray(manifest['sets'][train_key]['indices'])
        train_archive=archives[name]['train']
        np.testing.assert_array_equal(train_archive['indices'],nominal)
        expected=nominal[train_archive['labels']>=0]
        plan=np.load(root/name/'sampling_plan.npz')['indices']
        normalization=json.loads((root/name/'normalization.json').read_text())
        unique,counts=np.unique(plan,return_counts=True)
        np.testing.assert_array_equal(unique,expected)
        np.testing.assert_array_equal(normalization['training_indices'],expected)
        if plan.shape!=(4000,128) or counts.max()-counts.min()>1:
            raise ValueError('Update/presentation coverage differs from the fixed protocol')
        execution_checks[name]=dict(updates=plan.shape[0],batch=plan.shape[1],presentations=plan.size,
            unique_training_rows=len(unique),minimum_presentations_per_row=int(counts.min()),
            maximum_presentations_per_row=int(counts.max()),
            normalization_uses_only_actual_training_rows=True)
    for split in ('development','internal','original_val'):
        if histories[MODELS[0]][0]['metrics'][split] != histories[MODELS[1]][0]['metrics'][split]:
            raise ValueError('Shared initial held-out evaluations differ')
    dump(root/'execution_checks.json',execution_checks)
    results=[]
    replay=[]
    for name in MODELS:
        selected=next(r for r in histories[name] if r['updates']==selections[name]['best_updates'])
        if selections[name]['initialization_sha256']!=config['initial_trainable_parameter_sha256']:
            raise ValueError('Shared parameter initialization differs')
        for split,archive in archives[name].items():
            metrics=binary_metrics(archive['scores'],archive['costs'])
            error=max(abs(float(metrics[k])-float(selected['metrics'][split][k])) for k in metrics)
            if error>1e-10:
                raise ValueError('Saved prediction replay failed')
            results.append(dict(model=name,split=split,best_updates=selections[name]['best_updates'],**metrics))
            replay.append(dict(model=name,split=split,max_metric_error=error))
    write_csv(root/'comparison.csv',results)
    dump(root/'prediction_replay.json',replay)
    indexed={(r['model'],r['split']):r for r in results}
    for split,outname in (('internal','internal_predictions.csv'),('original_val','val_predictions.csv')):
        first=archives[MODELS[0]][split]
        second=archives[MODELS[1]][split]
        for field in ('indices','costs','labels','base_ids','customers'):
            np.testing.assert_array_equal(first[field],second[field])
        rows=[]
        for i in range(len(first['indices'])):
            row=dict(source_split='train' if split=='internal' else 'val',original_index=int(first['indices'][i]),
                base_id=int(first['base_ids'][i]),customers=int(first['customers'][i]),
                MOEL_cost=float(first['costs'][i,0]),MTL_cost=float(first['costs'][i,1]),
                binary_label=int(first['labels'][i]),included_in_metrics=bool(first['labels'][i]>=0))
            for name,short in zip(MODELS,('A_2000','B_8000')):
                scores=archives[name][split]['scores'][i].astype(np.float64)
                probability=np.exp(scores-scores.max())
                row.update({short+'_prediction':int(scores.argmax()),short+'_score0':float(scores[0]),
                    short+'_score1':float(scores[1]),short+'_probability1':float(probability[1]/probability.sum())})
            rows.append(row)
        write_csv(root/outname,rows)
        saved=read_csv(root/outname)
        for name,short in zip(MODELS,('A_2000','B_8000')):
            scores=np.array([[float(r[short+'_score0']),float(r[short+'_score1'])] for r in saved])
            costs=np.array([[float(r['MOEL_cost']),float(r['MTL_cost'])] for r in saved])
            replayed=binary_metrics(scores,costs)
            reference=indexed[(name,split)]
            error=max(abs(float(replayed[k])-float(reference[k])) for k in replayed)
            if error>1e-10:
                raise ValueError('CSV prediction replay failed')
            replay.append(dict(model=name,split=split,format='CSV',max_metric_error=error))
    dump(root/'prediction_replay.json',replay)
    bootstrap=[]
    for split in ('internal','original_val'):
        for row in bootstrap_difference(instance_metrics(archives[MODELS[1]][split]),instance_metrics(archives[MODELS[0]][split]),True,4902):
            bootstrap.append(dict(comparison='B minus A',split=split,**row))
    for name in MODELS:
        for row in bootstrap_difference(instance_metrics(archives[name]['internal']),instance_metrics(archives[name]['original_val']),False,4902):
            bootstrap.append(dict(comparison='internal minus original validation',model=name,**row))
    write_csv(root/'comparison_uncertainty.csv',bootstrap)
    plot_curves(root,histories,selections)
    distribution=read_csv(root/'split_comparison.csv')
    lines=['# R49: existing-data coverage and sample-count diagnosis','',
        'Scope: OVRPTW RELD_MOEL (class0) vs RELD_MTL (class1), seed2. Existing train/val only; no test, new instances or solver reruns.',
        'Both clean R48 classifiers start from the same fresh trainable initialization. No solver encoding, retrieval, pair/risk/R-Drop or auxiliary loss.',
        '', '## Locked protocol','',
        'Split original 10,000 training rows into 8,000 training candidates, 1,000 development and 1,000 internal holdout. '
        'A uses a fixed size-stratified 2,000-row subset of B; original validation is unchanged. '
        'Splitting is by input-derived base group and exact customer count, never label/winner or index-derived distribution.',
        f'Original training unique base groups: {manifest["unique_original_train_groups"]}; original validation: {manifest["unique_original_val_groups"]}. '
        'No detected related-group overlap between train/development/internal/original validation under the input-canonicalization rule. '
        'No reliable per-instance distribution or historical base-ID metadata is present.',
        'Each group uses 4,000 successful updates of exactly128 examples: 512,000 presentations, shuffled complete passes with tails carried forward. '
        'There is no early stopping or schedule; LR1e-4, AdamW WD1e-4, dropout0.1, gradient clip1.0, FP32/TF32 off match R48. '
        'A and B each fit geometry normalization on their own actual non-tie training nodes only.',
        'Evaluate all four sets at update0 and every200 updates. Strict minimum development CE chooses best; internal and original validation are observations only. '
        'All metrics exclude exact FP64 cost ties; regret uses the two-method Oracle, not the seven-method pool.',
        '', '| Set | Nominal rows | Non-ties | Exact ties |', '|---|---:|---:|---:|']
    for name,values in manifest['sets'].items():
        lines.append(f'| {name} | {values["n_total"]} | {values["n_non_ties"]} | {values["n_exact_ties"]} |')
    lines += ['', '## Main results','',
        '| Model | Best update | Train accuracy | Development CE | Internal accuracy / regret | Original validation accuracy / regret |',
        '|---|---:|---:|---:|---:|---:|']
    for name in MODELS:
        train,dev,internal,val=[indexed[(name,s)] for s in ('train','development','internal','original_val')]
        lines.append(f'| {name} | {selections[name]["best_updates"]} | {train["accuracy"]*100:.3f}% | {dev["ce"]:.6f} | '
            f'{internal["accuracy"]*100:.3f}% / {internal["pair_regret_pct"]:.6f}% | {val["accuracy"]*100:.3f}% / {val["pair_regret_pct"]:.6f}% |')
    lines += ['', 'Full same-checkpoint metrics:', '', '| Model | Set | N | Accuracy | CE | Mean cost | Two-method regret |',
              '|---|---|---:|---:|---:|---:|---:|']
    for row in results:
        lines.append(f'| {row["model"]} | {row["split"]} | {row["n"]} | {row["accuracy"]*100:.3f}% | {row["ce"]:.6f} | '
            f'{row["mean_cost"]:.6f} | {row["pair_regret_pct"]:.6f}% |')
    lines += ['', '## Diagnostic differences','',
        'Intervals below resample held-out instances conditional on these fixed models; they do not measure training-seed or split uncertainty.',
        '', '| Contrast | Set / model | Metric | Difference | 95% interval |', '|---|---|---|---:|---|']
    for row in bootstrap:
        lines.append(f'| {row["comparison"]} | {row.get("split",row.get("model"))} | {row["metric"]} | {row["difference"]:+.6f} | '
            f'[{row["lower95"]:+.6f}, {row["upper95"]:+.6f}] |')
    lines += ['', 'Final update4,000 is reported separately, never used instead of the development-selected checkpoint:',
        '', '| Model | Train accuracy / CE | Development accuracy / CE | Internal accuracy / CE | Original validation accuracy / CE |',
        '|---|---:|---:|---:|---:|']
    for name in MODELS:
        final=histories[name][-1]
        if final['updates']!=4000:
            raise ValueError('Successful-update budget not completed')
        values=[final['metrics'][split] for split in ('train','development','internal','original_val')]
        lines.append('| '+name+' | '+' | '.join(f'{r["accuracy"]*100:.3f}% / {r["ce"]:.6f}' for r in values)+' |')
    lines += ['', '## Input distribution comparison','',
        'All R48 hand-computed input statistics are in split_comparison.csv and input_statistics.npz. '
        'They are not added to the network. Standardized mean differences use pooled per-feature standard deviations against the 8,000-row training pool.',
        '', '| Set | Mean customers | MOEL win fraction | Mean demand total | Mean TW width | Mean pair distance |',
        '|---|---:|---:|---:|---:|---:|']
    distribution_index={(r['set'],r['feature']):r for r in distribution}
    for name in manifest['sets']:
        columns=[float(distribution_index[(name,key)]['mean']) for key in
            ('customers','MOEL_win_fraction','total_demand','tw_width_mean','distance_mean')]
        lines.append('| '+name+' | '+' | '.join(f'{v:.6f}' for v in columns)+' |')
    lines += ['', 'Largest absolute input-statistic standardized mean differences:', '']
    for name in ('development','internal','original_val','A_2000'):
        shifted=[r for r in distribution if r['set']==name and r.get('standardized_mean_difference_vs_training_pool') not in (None,'')]
        largest=sorted(shifted,key=lambda r:abs(float(r['standardized_mean_difference_vs_training_pool'])),reverse=True)[:5]
        lines.append(name+': '+', '.join(r['feature']+'='+f'{float(r["standardized_mean_difference_vs_training_pool"]):+.3f}' for r in largest)+'.')
    lines += ['', '## Observations and interpretation','']
    coverage_supported=False
    for name in MODELS:
        internal=indexed[(name,'internal')]
        val=indexed[(name,'original_val')]
        interval=next(r for r in bootstrap if r['comparison']=='internal minus original validation'
            and r['model']==name and r['metric']=='accuracy')
        coverage_supported |= interval['lower95']>0
        lines.append(f'- {name}: internal accuracy {internal["accuracy"]*100:.3f}% versus original validation '
            f'{val["accuracy"]*100:.3f}% ({interval["difference"]:+.3f}pp; conditional95% interval '
            f'[{interval["lower95"]:+.3f}, {interval["upper95"]:+.3f}]). Internal/original regret is '
            f'{internal["pair_regret_pct"]:.6f}% / {val["pair_regret_pct"]:.6f}%.')
    sample_supported=False
    for split in ('internal','original_val'):
        first=indexed[(MODELS[0],split)]
        second=indexed[(MODELS[1],split)]
        interval=next(r for r in bootstrap if r['comparison']=='B minus A' and r['split']==split and r['metric']=='accuracy')
        sample_supported |= interval['lower95']>0
        lines.append(f'- Increasing distinct training rows2,000->8,000 changes {split} accuracy by '
            f'{interval["difference"]:+.3f}pp (conditional95% interval [{interval["lower95"]:+.3f}, {interval["upper95"]:+.3f}]), '
            f'CE by {second["ce"]-first["ce"]:+.6f}, and regret by '
            f'{second["pair_regret_pct"]-first["pair_regret_pct"]:+.6f}pp. Negative CE/regret changes are better.')
    if not coverage_supported:
        lines.append('- Neither fixed model provides clear accuracy evidence that same-source internal holdout is '
            'easier than original validation. A large original-validation source mismatch is not supported by this probe; '
            'this does not exclude a smaller, conditional or unobserved distribution shift.')
    else:
        lines.append('- At least one fixed model has a positive internal-versus-original accuracy interval. '
            'Investigate whether this also holds for CE/regret and is robust to the different binary class proportions '
            'before attributing it to a generation-source mismatch.')
    if not sample_supported:
        lines.append('- Neither accuracy gain from2,000->8,000 excludes zero in the conditional instance-bootstrap interval. '
            'This experiment does not establish a large sample-count benefit; any cost/CE improvements are reported '
            'rather than discarded. Do not extrapolate these two points to an irreducible ceiling or to much larger datasets.')
    else:
        lines.append('- At least one accuracy gain from2,000->8,000 has a positive conditional interval. '
            'This is evidence of a sample-count effect on that holdout for this fixed split/seed; '
            'the other holdout and cost directions determine whether it is a broad benefit.')
    for name in MODELS:
        first=next((r for r in histories[name] if r['metrics']['train']['accuracy']>=.99),None)
        if first is not None:
            lines.append(f'- {name} first reaches>=99% training accuracy at update{first["updates"]}; '
                f'the corresponding internal/original accuracy is {first["metrics"]["internal"]["accuracy"]*100:.3f}% / '
                f'{first["metrics"]["original_val"]["accuracy"]*100:.3f}%. Strong training fit is not itself evidence of transferable solver-win signal.')
    lines += ['', '## Limits and next step','',
        '1. If the two held-out sets remain similarly difficult and larger subsets do not provide a clear joint benefit, '
        'the next targeted question is whether the instance Encoder captures transferable relationships governing '
        'relative solver performance, rather than another solver embedding or decoder replacement. '
        'This run does not train any additional architecture or use held-out observations to tune the two runs.',
        '2. This is two sample counts at one optimization budget and one split/seed, not an asymptotic learning curve. '
        'Normalization changes with the actual training subset as predeclared, and may contribute to A/B differences. '
        'A repeatedly sees fewer unique instances; B has fewer passes per instance despite equal updates and presentations. '
        'Development selection can choose different update counts; equal budgets describe the complete runs, not necessarily the selected checkpoints.',
        '3. Similar observed input statistics do not establish equal joint distributions or equal solver behavior; '
        'a lack of improvement within 2,000-to-8,000 cannot establish a predictability ceiling or rule out much larger independent datasets.',
        '4. The internal and original validation results never choose checkpoints, stop training or change hyperparameters. '
        'No historical learned weights are used, because they would have seen the new internal holdout.',
        '', '## Execution','',
        'Only the idle RTX3090 selected by R49_GPU_UUID was used; the busy other3090 and4090 were not used.',
        '```bash', 'source /public/home/shiys/miniconda3/etc/profile.d/conda.sh', 'conda activate easynco',
        'cd /public/home/shiys/0selection2',
        f'R49_GPU_UUID={config["runtime"]["visible_gpu"]} bash code/V4/run_v4_r49.sh all', '```',
        'On this launch the GPU is reached within the existing allocation with '
        'srun --jobid=1465 --overlap --exact --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=10G; long tasks use tmux.',
        'Checkpoints, per-run history/normalization/sampling plans and W&B offline logs are saved. '
        'Prediction artifacts reproduce the reported metrics. No API or embedding generation is needed.',
        '', 'Learning-curve context: [The Shape of Learning Curves: a Review](https://arxiv.org/abs/2103.10948). '
        'This experiment studies distinct training sample count, not only repeated optimization epochs; it does not assume more data always helps.']
    (root/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    dump(root/'analysis_provenance.json',dict(source_sha256=file_hash(Path(__file__)),
        launch_source_sha256=file_hash(root/'source_launch/r49_analysis.py'),
        changes='Reporting only: final-update table, numerical findings and conditional uncertainty; training unchanged.'))
    dump(root/'completion.json',dict(complete=True,models=list(MODELS),test_read=False,
        successful_updates_each=4000,best_updates={name:selections[name]['best_updates'] for name in MODELS},
        original_validation_used_for_selection=False,internal_used_for_selection=False))
    print('\n'.join(lines[lines.index('## Main results'):lines.index('Full same-checkpoint metrics:')]),flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,default=ROOT)
    args=parser.parse_args()
    analyze(args.root)


if __name__=='__main__':
    main()

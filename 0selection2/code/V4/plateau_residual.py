"""Train-defined tabular probes of signal beyond frozen R70 score summaries."""

import argparse
from collections import defaultdict
import json
import shutil

import numpy as np

from code.V4.plateau_analysis import ROOT, PROBLEMS, load, locate, sources, write_csv, dump


MIN_SUPPORT = 40
SHRINKAGE = 50.


def cuts(values, quantiles):
    return np.unique(np.quantile(values, quantiles))


def feature(problem, archive):
    names = list(archive['names'].astype(str))
    if problem == 'ATSP':
        name = 'matrix_asymmetry_mean'
        value = archive['X'][:, names.index(name)]
    elif 'TW' in problem:
        name = 'directed_slack_mean'
        value = archive['X'][:, names.index(name)]
    else:
        name = 'sqrt_size_times_nn1_mean'
        value = archive['X'][:, names.index('nn1_mean')] * np.sqrt(archive['sizes'])
    return name, value


def rows_for(a, f, boundaries):
    pair = a['order'][:, :2]
    margin = np.sort(a['probs'], axis=1)[:, -1] - np.sort(a['probs'], axis=1)[:, -2]
    bins = [np.searchsorted(b, x, side='right') for b, x in zip(boundaries, (a['size'], margin, f))]
    pair_keys = [tuple(x) for x in pair]
    size_keys = [(int(s), *p) for s, p in zip(bins[0], pair_keys)]
    score_keys = [(int(s), *p, int(m)) for s, p, m in zip(bins[0], pair_keys, bins[1])]
    feature_keys = [(*s, int(v)) for s, v in zip(score_keys, bins[2])]
    return pair, (pair_keys, size_keys, score_keys, feature_keys)


def fit(keys, targets):
    output=[]
    for level in keys:
        sums=defaultdict(lambda: np.zeros(4, dtype=np.float64))
        for key, target in zip(level, targets):
            sums[key] += [target, float(target < 0), float(target != 0), 1.]
        output.append(sums)
    return output


def estimates(keys, tables, prior):
    means=np.tile(prior, (len(keys[0]), 1))
    outputs=[]
    support=[]
    used=[]
    for depth,(level, table) in enumerate(zip(keys, tables)):
        # Both score and descriptor cells shrink to the same size/pair parent.
        parent=outputs[1] if depth==3 else means.copy()
        counts=np.zeros(len(level),int)
        eligible=np.zeros(len(level),bool)
        for i,key in enumerate(level):
            value=table.get(key)
            if value is None:continue
            counts[i]=int(value[3])
            if value[3] < MIN_SUPPORT:continue
            eligible[i]=True
            means[i,0]=(value[0]+SHRINKAGE*parent[i,0])/(value[3]+SHRINKAGE)
            means[i,1]=(value[1]+SHRINKAGE*parent[i,1])/(value[2]+SHRINKAGE)
        outputs.append(means.copy());support.append(counts);used.append(eligible)
    return outputs,support,used


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',default=str(ROOT));args=parser.parse_args()
    from pathlib import Path
    root=Path(args.root);spec=sources()['R70B']
    protocol=root/'residual_protocol.json'
    if protocol.exists() and json.loads(protocol.read_text()).get('version',1)<2:
        old=root/'residual_initial';old.mkdir(exist_ok=True)
        for path in root.glob('residual_*.csv'):shutil.copy2(path,old/path.name)
        shutil.copy2(protocol,old/protocol.name)
    results=[];transfers=[];missing=[];settings=[]
    for p in PROBLEMS:
        paths={s:root/'regions/feature_cache'/f'{p}_{s}.npz' for s in ('train','val','test')}
        if not all(v.exists() for v in paths.values()):
            missing.append(dict(problem=p,reason='feature cache pending'));continue
        arrays={s:load(locate(spec[s],p,s),p,s) for s in paths}
        features={}
        for s,path in paths.items():
            with np.load(path,allow_pickle=False) as f:
                np.testing.assert_array_equal(f['sizes'],arrays[s]['size'])
                name,features[s]=feature(p,f)
        train=arrays['train']
        prob=np.sort(train['probs'],axis=1)
        boundaries=[cuts(train['size'],[.25,.5,.75]),cuts(prob[:,-1]-prob[:,-2],[1/3,2/3]),cuts(features['train'],[.25,.5,.75])]
        pair,keys=rows_for(train,features['train'],boundaries)
        c=train['costs'];ix=np.arange(len(c))
        delta=(c[ix,pair[:,0]]-c[ix,pair[:,1]])/c.min(1)*100
        prior=np.asarray([delta.mean(),(delta[delta!=0]<0).mean()])
        tables=fit(keys,delta)
        constant_keys=(*keys[:3],[(*key,0) for key in keys[2]])
        constant_tables=fit(constant_keys,delta)
        constant_estimates=estimates(constant_keys,constant_tables,prior)[0]
        np.testing.assert_allclose(constant_estimates[2],constant_estimates[3],rtol=0,atol=0)
        settings.append(dict(problem=p,feature=name,train_boundaries=[b.tolist() for b in boundaries]))
        for s in ('val','test'):
            a=arrays[s];c=a['costs'];n=len(c);ix=np.arange(n);oracle=c.min(1)
            pair,keys=rows_for(a,features[s],boundaries)
            truth=(c[ix,pair[:,0]]-c[ix,pair[:,1]])/oracle*100
            estimates_,support,used=estimates(keys,tables,prior)
            substantial=np.abs(truth)>.5
            for level,label in [(2,'score_summary'),(3,'score_summary_plus_feature')]:
                estimate=estimates_[level]
                for objective in ('pair_accuracy','pair_relative_cost'):
                    choose_first=estimate[:,1]>.5 if objective=='pair_accuracy' else estimate[:,0]<0
                    pred=np.where(choose_first,pair[:,0],pair[:,1])
                    regret=(c[ix,pred]/oracle-1)*100
                    results.append(dict(problem=p,split=s,n=n,representation=label,objective=objective,
                        feature=name,top1_pct=(pred==a['y']).mean()*100,actual_regret_pct=regret.mean(),
                        frozen_top1_pct=a['correct'].mean()*100,frozen_regret_pct=a['regret'].mean(),
                        changed_count=int((pred!=a['pred']).sum()),
                        corrected=int(((pred==a['y'])&~a['correct']).sum()),
                        harmed=int(((pred!=a['y'])&a['correct']).sum()),
                        own_cell_support_pct=used[level].mean()*100,
                        mean_cell_train_count=support[level].mean(),fallback_pct=(~used[level]).mean()*100,
                        signed_delta_mse=float(np.square(estimate[:,0]-truth).mean()),
                        substantial_pair_n=int(substantial.sum()),
                        substantial_pair_sign_accuracy_pct=((estimate[substantial,0]<0)==(truth[substantial]<0)).mean()*100 if substantial.any() else np.nan))
            for level,label in [(2,'score_summary'),(3,'score_summary_plus_feature')]:
                groups=defaultdict(list)
                for i,key in enumerate(keys[level]):groups[key].append(i)
                for key,ii in groups.items():
                    if len(ii)<20 or key not in tables[level] or tables[level][key][3]<MIN_SUPPORT:continue
                    observed=truth[ii].mean();training=tables[level][key][0]/tables[level][key][3]
                    transfers.append(dict(problem=p,split=s,representation=label,feature=name,cell=str(key),
                        train_n=int(tables[level][key][3]),eval_n=len(ii),train_mean_signed_gap_pct=training,
                        eval_mean_signed_gap_pct=observed,same_sign=bool(np.sign(training)==np.sign(observed))))
    for split in ('val','test'):
        for representation in ('score_summary','score_summary_plus_feature'):
            for objective in ('pair_accuracy','pair_relative_cost'):
                subset=[r for r in results if r['split']==split and r['representation']==representation and r['objective']==objective]
                if len(subset)!=len(PROBLEMS):continue
                row=dict(problem='ALL18',split=split,representation=representation,objective=objective,n=sum(r['n'] for r in subset))
                for key in ('top1_pct','actual_regret_pct','frozen_top1_pct','frozen_regret_pct','own_cell_support_pct','fallback_pct','signed_delta_mse'):
                    row[key]=float(np.mean([r[key] for r in subset]))
                results.append(row)
    write_csv(root/'residual_signal.csv',results)
    write_csv(root/'residual_region_transfer.csv',transfers)
    write_csv(root/'residual_missing.csv',missing)
    dump(root/'residual_protocol.json',dict(version=2,model='R70B locked checkpoint',minimum_cell_train_n=MIN_SUPPORT,
        parent_pseudocount=SHRINKAGE,settings=settings,
        meaning='train-only conditional tables, not neural retraining; ordered predicted top2 preserves frozen choice',
        shrinkage='score and descriptor cells use same size/pair parent; unsupported descriptor falls back to score; constant descriptor exactly invariant',
        pair_accuracy='pairwise non-tied cost-win probability, NOT native full-pool winner frequency',
        review_correction='v1 discarded predicted direction and duplicated shrinkage; archived as residual_initial, not used in final claims',
        limitations=['training scores are in-sample, hence conditional overfit can occur',
                     'historical test exploratory; not a released model or test-tuned policy',
                     'a null is only about these coarse summaries and one predeclared descriptor',
                     'features use existing inputs, no winner-dependent neighborhood or routing labels']))
    print(f'[residual] {len(settings)} problems; missing {len(missing)}',flush=True)


if __name__=='__main__':main()

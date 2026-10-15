"""Read-only prediction analysis; no selector fitting or checkpoint selection."""

import argparse
import csv
import itertools
import json
import pickle
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.special import logsumexp, softmax

from code.unified_selector.registry import DATA_ROOT, POOLS, PROBLEMS, problem_to_pool_mask


RUNS = Path(__file__).resolve().parent / 'runs'
ROOT = RUNS / 'Plateau_diagnosis_R70'
PRIMARY = ('R45A', 'R60B', 'R70B', 'NSS')
MARGINS = (0., .1, .5, 1., 2., np.inf)
MARGIN_NAMES = ('exact_tie', '(0,0.1]%', '(0.1,0.5]%', '(0.5,1]%', '(1,2]%', '>2%')


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


@lru_cache(None)
def raw(problem, split):
    with (DATA_ROOT / (problem + split) / 'raw_label.pkl').open('rb') as stream:
        rows = pickle.load(stream)
    values = [rows[str(i)] for i in range(len(rows))]
    c = np.asarray([r['cost'] for r in values], dtype=np.float64)
    y = np.asarray([r['ind'] for r in values], dtype=np.int64)
    assert c.shape == (len(values), len(POOLS[problem]))
    assert np.isfinite(c).all() and (c > 0).all()
    return c, y


def sources():
    specs = {}
    for model in ('R34', 'R39A', 'R40A', 'R40B'):
        specs[model] = {'test': RUNS / 'R41_signal_audit/test_predictions' / model}
    rounds = (
        ('R42_relation_query', 'R42A', 'A', 'dual_stream_rdrop_seed2'),
        ('R42_relation_query', 'R42B', 'B', 'relation_query_rdrop_seed2'),
        ('R43_pairwise_selection', 'R43A', 'A', 'score_difference_seed2'),
        ('R43_pairwise_selection', 'R43B', 'B', 'explicit_pair_seed2'),
        ('R45_solver_code_embeddings', 'R45A', 'A', 'A_handcrafted_seed2'),
        ('R45_solver_code_embeddings', 'R45B', 'B', 'B_code_seed2'),
        ('R45_solver_code_embeddings', 'R45C', 'C', 'C_shuffled_code_seed2'),
        ('R46_code_token_interaction', 'R46A', 'A', 'A_code_tokens_seed2'),
        ('R47_behavior_memory', 'R47B', 'B', 'B_behavior_memory_seed2'),
    )
    for folder, model, arm, run in rounds:
        root = RUNS / folder
        lock = json.loads((root / 'locked_checkpoints.json').read_text())[arm]
        specs[model] = {'test': root / 'test_predictions' / arm,
                        'val': root / run / 'val_predictions' / f"epoch{lock['epoch']:03d}",
                        'history': root / run / 'history.json', 'selected_epoch': lock['epoch']}
    specs['R60B'] = {
        'test': RUNS / 'R63_recurrent_workspaces/reference/R60B/test_predictions',
        'val': RUNS / 'R60_expanded_partial_supervision/B_expanded_seed2/val_predictions/best',
        'history': RUNS / 'R60_expanded_partial_supervision/B_expanded_seed2/history.json'}
    for folder, model, run in (
        ('R64_pcgrad_full18', 'R64A', 'A_mean_gradient_seed2'),
        ('R64_pcgrad_full18', 'R64B', 'B_pcgrad_seed2'),
        ('R65_schedule_free_full18', 'R65B', 'B_schedule_free_seed2'),
        ('R69_task_specific_finetuning', 'R69B', 'B_task_specific_seed2'),
        ('R70_latent_mixup', 'R70B', 'B_latent_mixup_seed2'),
    ):
        root = RUNS / folder
        arm = 'A' if model.endswith('A') else 'B'
        lock = json.loads((root / 'final_test_lock.json').read_text())['models'][arm]
        val = root / run / 'val_predictions' / f"epoch{lock['epoch']:03d}"
        if not val.exists():
            val = root / run / 'val_predictions/best'
        specs[model] = {'test': root / run / 'test_predictions', 'val': val,
                        'train': root / run / 'train_predictions/best',
                        'history': root / run / 'history.json', 'selected_epoch': lock['epoch']}
    specs['R25'] = {'test': RUNS / 'NSS_ppt_comparison/predictions/R25',
                    'LIB': RUNS / 'NSS_ppt_comparison/predictions/R25'}
    for model in ('R42A', 'R42B'):
        specs[model]['LIB'] = RUNS / 'NSS_ppt_comparison/predictions' / model
    specs['NSS'] = {s: ROOT / 'nss_predictions' / s for s in ('train', 'train_sample', 'val', 'test', 'LIB')}
    return specs


def locate(root, p, split):
    names = [f'{p}.npz', f'{p}_{split}.npz', f'{p}{split}.npz']
    if split == 'LIB':
        names = [f'{p}LIB.npz', f'{p}.npz']
    for name in names:
        path = root / name
        if path.exists():
            return path
    return None


def load(path, p, split):
    with np.load(path, allow_pickle=False) as f:
        a = {k: f[k] for k in f.files}
    c, y = raw(p, 'train' if split=='train_sample' else split)
    idx = a.get('indices', np.arange(len(a['logits'])))
    assert len(np.unique(idx)) == len(idx), path
    reorder = np.argsort(idx)
    if split=='train_sample':
        assert (idx>=0).all() and (idx<len(c)).all()
        c,y=c[idx[reorder]],y[idx[reorder]]
    else:
        assert np.array_equal(idx[reorder], np.arange(len(c))), (path, 'incomplete split')
    pool = list(map(str, a['pool']))
    columns = [pool.index(s) for s in POOLS[p]]
    z = np.asarray(a['logits'][reorder][:, columns], np.float64)
    assert not np.isnan(z).any() and not np.isposinf(z).any(), path
    assert np.isfinite(z).any(1).all(), path
    if 'costs' in a:
        np.testing.assert_allclose(a['costs'][reorder][:, columns], c, rtol=1e-6, atol=1e-7)
    nodes = a.get('nodes', a.get('sizes'))
    size = None if nodes is None else np.asarray(nodes)[reorder].astype(int)
    if size is not None and p not in ('TSP', 'ATSP') and 'nodes' in a:
        size = size - 1
    ids = np.asarray(problem_to_pool_mask(p)[0])
    order = np.lexsort((np.broadcast_to(ids, z.shape), -z), axis=1)
    pred = order[:, 0]
    oracle = c.min(1)
    regret = (c[np.arange(len(y)), pred] / oracle - 1) * 100
    rank = np.argmax(order == y[:, None], axis=1) + 1
    probs = softmax(z, axis=1)
    return dict(costs=c, y=y, z=z, order=order, pred=pred, regret=regret, rank=rank,
                probs=probs, size=size, indices=idx[reorder], correct=pred == y, path=str(path), pool=list(POOLS[p]))


def metric(a):
    c, y, pred = a['costs'], a['y'], a['pred']
    n = len(y)
    oracle = c.min(1)
    selected = c[np.arange(n), pred]
    train, ty = raw(a['problem'], 'train')
    sbs = int(train.mean(0).argmin())
    prior = int(np.bincount(ty, minlength=c.shape[1]).argmax())
    relative_sbs = int((train/train.min(1,keepdims=True)-1).mean(0).argmin())
    fixed_regret = (c[:, sbs] / oracle - 1).mean() * 100
    confidence = a['probs'].max(1)
    ece = 0.
    for lo, hi in zip(np.linspace(0, 1, 11)[:-1], np.linspace(0, 1, 11)[1:]):
        m = (confidence > lo) & (confidence <= hi)
        if m.any():
            ece += m.mean() * abs(confidence[m].mean() - a['correct'][m].mean())
    out = dict(n=n, top1_pct=a['correct'].mean()*100, top2_pct=(a['rank']<=2).mean()*100,
               top3_pct=(a['rank']<=3).mean()*100, actual_regret_pct=a['regret'].mean(),
               mean_cost=selected.mean(), oracle_cost=oracle.mean(), ce=(logsumexp(a['z'], axis=1)-a['z'][np.arange(n),y]).mean(),
               top1_fp32_pct=(pred==c.astype(np.float32).argmin(1)).mean()*100,
               top2_fp32_pct=(a['order'][:,:2]==c.astype(np.float32).argmin(1)[:,None]).any(1).mean()*100,
               top3_fp32_pct=(a['order'][:,:3]==c.astype(np.float32).argmin(1)[:,None]).any(1).mean()*100,
               top2_min_regret_pct=(np.take_along_axis(c,a['order'][:,:2],axis=1).min(1)/oracle-1).mean()*100,
               top3_min_regret_pct=(np.take_along_axis(c,a['order'][:,:3],axis=1).min(1)/oracle-1).mean()*100,
               unavailable_winner_pct=np.isneginf(a['z'][np.arange(n),y]).mean()*100,
               exact_cost_tie_hit_pct=(selected==oracle).mean()*100, ece_pct=ece*100,
               brier=((a['probs']-np.eye(c.shape[1])[y])**2).sum(1).mean(),
               regret_p95_pct=np.percentile(a['regret'],95), regret_max_pct=a['regret'].max(),
               mean_cost_vs_oracle_pct=(selected.mean()/oracle.mean()-1)*100,
               train_sbs_regret_pct=fixed_regret, train_majority_accuracy_pct=(y==prior).mean()*100,
               train_relative_sbs_regret_pct=(c[:,relative_sbs]/oracle-1).mean()*100,
               regret_gap_closed_pct=(1-a['regret'].mean()/fixed_regret)*100 if fixed_regret>0 else np.nan,
               tail_5pct_regret_share_pct=np.sort(a['regret'])[-max(1,int(np.ceil(n*.05))):].sum()/max(a['regret'].sum(),1e-15)*100)
    return {k: (int(v) if k=='n' else float(v)) for k,v in out.items()}


def append_macros(rows, keys, numeric, scope_name='ALL18'):
    groups = defaultdict(list)
    for row in rows:
        if row['problem'] in PROBLEMS:
            groups[tuple(row[k] for k in keys)].append(row)
    extra = []
    for key, values in groups.items():
        if {r['problem'] for r in values} != set(PROBLEMS):
            continue
        row = dict(zip(keys, key), problem=scope_name, n=sum(r.get('n',0) for r in values))
        for col in numeric:
            row[col] = float(np.mean([r[col] for r in values]))
        extra.append(row)
    rows.extend(extra)


def fixed_portfolios(p, c, y):
    train, ty = raw(p, 'train')
    gap = train/train.min(1,keepdims=True)-1
    winners = np.bincount(ty, minlength=c.shape[1])
    rows=[]
    for k in (1,2,3):
        if k>c.shape[1]:continue
        majority = np.argsort(-winners, kind='stable')[:k]
        subsets=list(itertools.combinations(range(c.shape[1]),k))
        best=min(subsets,key=lambda s:gap[:,s].min(1).mean())
        for objective, members in [('winner_coverage',majority),('mean_relative_gap',best)]:
            members=np.asarray(members)
            rows.append(dict(k=k,objective=objective,methods='|'.join(POOLS[p][j] for j in members),
                             winner_coverage_pct=np.isin(y,members).mean()*100,
                             regret_pct=(c[:,members].min(1)/c.min(1)-1).mean()*100,
                             diagnostic='k>1 requires multiple solver runs; not single-choice accuracy'))
    return rows


def data_tables(loaded):
    profiles=[]; portfolios=[]; distributions=[]
    for (model,split,p),a in loaded.items():
        n=len(a['y']);c=a['costs'];k=c.shape[1]
        for j,name in enumerate(POOLS[p]):
            m=a['y']==j; chosen=a['pred']==j
            distributions.append(dict(model=model,split=split,problem=p,solver=name,n=n,
                winner_count=int(m.sum()),pick_count=int(chosen.sum()),
                winner_pct=m.mean()*100,pick_pct=chosen.mean()*100,
                winner_recall_pct=(a['pred'][m]==j).mean()*100 if m.any() else np.nan,
                pick_precision_pct=(a['y'][chosen]==j).mean()*100 if chosen.any() else np.nan,
                fixed_method_regret_pct=(c[:,j]/c.min(1)-1).mean()*100,
                missed_winner_regret_sum=float(a['regret'][m & ~a['correct']].sum())))
    seen=set()
    for (_,split,p),a in loaded.items():
        if (split,p) in seen: continue
        seen.add((split,p));c=a['costs'];y=a['y']; sc=np.sort(c,axis=1)
        gap=(sc[:,1]/sc[:,0]-1)*100; counts=np.bincount(y,minlength=c.shape[1]); freq=counts/len(y)
        ent=-np.sum(freq[freq>0]*np.log(freq[freq>0]))
        profiles.append(dict(problem=p,split=split,n=len(y),candidate_count=c.shape[1],
            oracle_mean=c.min(1).mean(), winner_entropy_nats=ent,effective_winner_count=np.exp(ent),
            modal_winner=POOLS[p][counts.argmax()],modal_winner_pct=freq.max()*100,
            top3_marginal_coverage_pct=np.sort(freq)[-3:].sum()*100,
            median_best2_gap_pct=np.median(gap),best2_gap_le01_pct=(gap<=.1).mean()*100,
            best2_gap_le05_pct=(gap<=.5).mean()*100,best2_gap_gt1_pct=(gap>1).mean()*100,
            fp32_native_mismatch_count=int((c.astype(np.float32).argmin(1)!=y).sum()),
            exact_tie_count=int((sc[:,0]==sc[:,1]).sum()),
            hindsight_best_constant_regret_pct=float(((c/c.min(1,keepdims=True)-1)*100).mean(0).min()),
            hindsight_best_constant_solver=POOLS[p][((c/c.min(1,keepdims=True)-1)*100).mean(0).argmin()],
            winner_semantics='legacy_signed_B' if p not in ('TSP','CVRP','ATSP') and 'B' in p else 'not_B_conflict_scope'))
        for row in fixed_portfolios(p,c,y):
            portfolios.append(dict(problem=p,split=split,n=len(y),**row))
    return profiles,portfolios,distributions


def errors(loaded):
    topk=[]; margin=[]; pair=[]; tolerant=[]; confidence=[]; samples=[];families=[]
    for (model,split,p),a in loaded.items():
        if model not in PRIMARY or split in ('train','train_sample'):continue
        n=len(a['y']); r=a['regret']; total=max(r.sum(),1e-15); error=~a['correct']; ec=max(error.sum(),1)
        for category,m in [('rank2',a['rank']==2),('rank3',a['rank']==3),('beyond3',a['rank']>3)]:
            topk.append(dict(model=model,split=split,problem=p,n=n,category=category,
                count=int(m.sum()),fraction_all_pct=m.mean()*100,error_share_pct=m.sum()/ec*100,
                regret_contribution_pp=r[m].sum()/n,regret_share_pct=r[m].sum()/total*100))
        sc=np.sort(a['costs'],axis=1); gap=(sc[:,1]/sc[:,0]-1)*100
        bucket=np.searchsorted(MARGINS,gap,side='left')
        for b,name in enumerate(MARGIN_NAMES):
            m=bucket==b
            margin.append(dict(model=model,split=split,problem=p,n=int(m.sum()),gap_bin=name,
                top1_pct=a['correct'][m].mean()*100 if m.any() else np.nan,
                regret_pct=r[m].mean() if m.any() else np.nan,
                error_count=int((m&error).sum()),error_share_pct=(m&error).sum()/ec*100,
                regret_share_pct=r[m].sum()/total*100,regret_contribution_pp=r[m].sum()/n))
        for selected,winner in sorted(set(zip(a['pred'][error],a['y'][error]))):
            m=(a['pred']==selected)&(a['y']==winner)
            pair.append(dict(model=model,split=split,problem=p,n=n,
                selected=POOLS[p][selected],winner=POOLS[p][winner],count=int(m.sum()),
                regret_sum_pct=r[m].sum(),regret_contribution_pp=r[m].sum()/n,
                error_share_pct=m.sum()/ec*100,regret_share_pct=r[m].sum()/total*100,
                mean_pair_gap_pct=r[m].mean()))
        def family(name):
            for prefix in ('RELD_','MoSES_','DIFUSCO','T2T'):
                if name.startswith(prefix):return prefix.rstrip('_')
            return name
        sf=np.asarray([family(POOLS[p][j]) for j in a['pred']])
        yf=np.asarray([family(POOLS[p][j]) for j in a['y']])
        reld_pair=np.asarray([{POOLS[p][i],POOLS[p][j]}=={'RELD_MOEL','RELD_MTL'} for i,j in zip(a['pred'],a['y'])])
        for cat,mask in [('MOEL_MTL',error&reld_pair),('other_within_named_family',error&~reld_pair&(sf==yf)),('between_named_families',error&(sf!=yf))]:
            families.append(dict(model=model,split=split,problem=p,n=n,category=cat,count=int(mask.sum()),
                regret_contribution_pp=r[mask].sum()/n,regret_share_pct=r[mask].sum()/total*100,
                definition='RELD, MoSES, DIFUSCO, T2T prefixes only; descriptive, no family-trained model'))
        for tolerance in [0.,.01,.05,.1,.25,.5,1.,2.,5.]:
            tolerant.append(dict(model=model,split=split,problem=p,n=n,tolerance_pct=tolerance,
                acceptable_pct=(r<=tolerance+1e-12).mean()*100,
                strict_top1_pct=a['correct'].mean()*100,
                label='cost-tolerant hit, NOT native Top1'))
        cf=a['probs'].max(1)
        for i in range(10):
            m=(cf>i/10)&(cf<=(i+1)/10)
            if m.any(): confidence.append(dict(model=model,split=split,problem=p,n=int(m.sum()),
                bin=i,mean_confidence=cf[m].mean(),accuracy=a['correct'][m].mean(),regret_pct=r[m].mean()))
        for idx in np.argsort(-r)[:10]:
            samples.append(dict(model=model,split=split,problem=p,index=int(idx),
                true_size=int(a['size'][idx]) if a['size'] is not None else '',
                selected=POOLS[p][a['pred'][idx]],winner=POOLS[p][a['y'][idx]],
                winner_rank=int(a['rank'][idx]),regret_pct=float(r[idx]),
                best2_gap_pct=float(gap[idx]),confidence=float(cf[idx]),diagnostic_only=True))
    return dict(topk_error_decomposition=topk,gap_error_budget=margin,error_pairs=pair,
                tolerance_curves=tolerant,confidence_bins=confidence,largest_losses=samples,error_families=families)


def permutations(loaded,repeats):
    rows=[]; draws={};rng=np.random.default_rng(2)
    for (model,split,p),a in loaded.items():
        if model not in PRIMARY or split not in ('val','test','LIB'):continue
        if a['size'] is None:continue
        n=len(a['y']); groups=[np.flatnonzero(a['size']==v) for v in np.unique(a['size'])]
        groups=[g for g in groups if len(g)>1];eligible=np.zeros(n,bool)
        for g in groups:eligible[g]=True
        acc=[];reg=[];eligible_acc=[];eligible_reg=[]
        for _ in range(repeats):
            mapping=np.arange(n)
            for g in groups:
                perm=rng.permutation(g)
                mapping[perm]=np.roll(perm,int(rng.integers(1,len(g))))
            pred=a['pred'][mapping];rs=(a['costs'][np.arange(n),pred]/a['costs'].min(1)-1)*100
            correct=pred==a['y'];acc.append(correct.mean()*100);reg.append(rs.mean())
            if eligible.any():
                eligible_acc.append(correct[eligible].mean()*100);eligible_reg.append(rs[eligible].mean())
        coverage=eligible.mean()*100
        for scope in ('full','eligible_only'):
            mask=np.ones(n,bool) if scope=='full' else eligible
            if not mask.any():continue
            aa=acc if scope=='full' else eligible_acc;rr=reg if scope=='full' else eligible_reg
            rows.append(dict(model=model,split=split,problem=p,scope=scope,n=int(mask.sum()),coverage_pct=coverage,
                actual_top1_pct=a['correct'][mask].mean()*100,shuffled_top1_pct=np.mean(aa),
                top1_instance_gain_pp=a['correct'][mask].mean()*100-np.mean(aa),
                actual_regret_pct=a['regret'][mask].mean(),shuffled_regret_pct=np.mean(rr),
                regret_instance_gain_pp=np.mean(rr)-a['regret'][mask].mean(),repeats=repeats,
                shuffle_top1_p025=np.percentile(aa,2.5),shuffle_top1_p975=np.percentile(aa,97.5)))
        draws[(model,split,p)] = np.asarray([acc,reg])
    for model in PRIMARY:
        for split in ('val','test'):
            keys=[(model,split,p) for p in PROBLEMS]
            if not all(k in draws for k in keys):continue
            arr=np.mean([draws[k] for k in keys],axis=0)
            subset=[r for r in rows if r['model']==model and r['split']==split and r['scope']=='full']
            rows.append(dict(model=model,split=split,problem='ALL18',scope='full',n=sum(r['n'] for r in subset),
                coverage_pct=np.mean([r['coverage_pct'] for r in subset]),
                actual_top1_pct=np.mean([r['actual_top1_pct'] for r in subset]),shuffled_top1_pct=arr[0].mean(),
                top1_instance_gain_pp=np.mean([r['top1_instance_gain_pp'] for r in subset]),
                actual_regret_pct=np.mean([r['actual_regret_pct'] for r in subset]),shuffled_regret_pct=arr[1].mean(),
                regret_instance_gain_pp=np.mean([r['regret_instance_gain_pp'] for r in subset]),repeats=repeats,
                shuffle_top1_p025=np.percentile(arr[0],2.5),shuffle_top1_p975=np.percentile(arr[0],97.5)))
    return rows


def comparisons(loaded):
    pair_rows=[];sample_rows=[]; common_rows=[];disagreement=[];universal=[];consensus=[]
    boot_draws={}
    rng=np.random.default_rng(502)
    for split in ('val','test','LIB'):
        for p in PROBLEMS:
            models=[m for m in ('R34','R39A','R42A','R42B','R43A','R43B','R45A','R45B','R46A','R47B','R60B','R64B','R65B','R69B','R70B','NSS') if (m,split,p) in loaded]
            if len(models)<2:continue
            aa=[loaded[(m,split,p)] for m in models];n=len(aa[0]['y'])
            preds=np.stack([a['pred'] for a in aa]); correct=np.stack([a['correct'] for a in aa]);reg=np.stack([a['regret'] for a in aa])
            # The union is a hindsight diagnostic, never a deployable selector.
            common_ix=[i for i,m in enumerate(models) if m!='NSS']
            cc=correct[common_ix];cr=reg[common_ix];cp=preds[common_ix]
            common_rows.append(dict(problem=p,split=split,n=n,models='|'.join(models[i] for i in common_ix),model_count=len(common_ix),
                all_wrong_pct=(~cc.any(0)).mean()*100,all_right_pct=cc.all(0).mean()*100,
                hindsight_union_accuracy_pct=cc.any(0).mean()*100,
                hindsight_min_selected_regret_pct=cr.min(0).mean(),
                all_wrong_R70_regret_share_pct=(loaded[('R70B',split,p)]['regret'][~cc.any(0)].sum()/max(loaded[('R70B',split,p)]['regret'].sum(),1e-15)*100) if ('R70B',split,p) in loaded else np.nan,
                all_wrong_R70_regret_contribution_pp=(loaded[('R70B',split,p)]['regret'][~cc.any(0)].sum()/n) if ('R70B',split,p) in loaded else np.nan,
                all_agree_pct=(cp==cp[0]).all(0).mean()*100))
            counts=np.zeros((n,aa[0]['costs'].shape[1]),int)
            for pp in cp:counts[np.arange(n),pp]+=1
            ids=np.asarray(problem_to_pool_mask(p)[0])
            vote=np.lexsort((np.broadcast_to(ids,counts.shape),-counts),axis=1)[:,0]
            regvote=(aa[0]['costs'][np.arange(n),vote]/aa[0]['costs'].min(1)-1)*100
            consensus.append(dict(problem=p,split=split,n=n,model_count=len(common_ix),
                top1_pct=(vote==aa[0]['y']).mean()*100,actual_regret_pct=regvote.mean(),
                rule='unweighted hard vote, global-ID tie; diagnostic multi-model overhead not measured'))
            for j,name in enumerate(POOLS[p]):
                mask=(aa[0]['y']==j)&~cc.any(0)
                universal.append(dict(problem=p,split=split,solver=name,n=n,count=int(mask.sum()),
                    total_winner_count=int((aa[0]['y']==j).sum()),
                    R70_regret_contribution_pp=loaded[('R70B',split,p)]['regret'][mask].sum()/n if ('R70B',split,p) in loaded else np.nan))
            for i,j in itertools.combinations(range(len(models)),2):
                disagreement.append(dict(problem=p,split=split,model_a=models[i],model_b=models[j],n=n,
                    prediction_agreement_pct=(preds[i]==preds[j]).mean()*100,
                    error_jaccard=((~correct[i])&(~correct[j])).sum()/max(((~correct[i])|(~correct[j])).sum(),1)))
            reference='NSS' if 'NSS' in models else ('R45A' if 'R45A' in models else models[0])
            references=['R45A'] if 'R45A' in models else []
            if reference not in references:references.append(reference)
            for ref in references:
                a=loaded[(ref,split,p)]
                for model in models:
                    if model==ref:continue
                    b=loaded[(model,split,p)];dc=b['correct'].astype(float)-a['correct'];dr=b['regret']-a['regret']
                    boot=rng.integers(n,size=(400,n))
                    aboot=dc[boot].mean(1)*100;rboot=dr[boot].mean(1)
                    boot_draws[(split,ref,model,p)]=(aboot,rboot)
                    aci=np.percentile(aboot,[2.5,97.5]);rci=np.percentile(rboot,[2.5,97.5])
                    both_wrong=~a['correct']&~b['correct'];changed=a['pred']!=b['pred']
                    row=dict(problem=p,split=split,reference=ref,model=model,n=n,
                        changed_count=int(changed.sum()),corrected=int((~a['correct']&b['correct']).sum()),
                        harmed=int((a['correct']&~b['correct']).sum()),
                        both_wrong_cheaper=int((both_wrong&(dr<0)).sum()),both_wrong_dearer=int((both_wrong&(dr>0)).sum()),
                        delta_top1_pp=dc.mean()*100,delta_regret_pp=dr.mean(),delta_raw_cost=float((b['costs'][np.arange(n),b['pred']]-a['costs'][np.arange(n),a['pred']]).mean()),
                        top1_boot_lo=aci[0],top1_boot_hi=aci[1],regret_boot_lo=rci[0],regret_boot_hi=rci[1])
                    pair_rows.append(row)
                    for category,mask in [('corrected',~a['correct']&b['correct']),('harmed',a['correct']&~b['correct']),('both_wrong',both_wrong),('both_correct',a['correct']&b['correct'])]:
                        sample_rows.append(dict(problem=p,split=split,reference=ref,model=model,category=category,n=int(mask.sum()),
                            net_regret_contribution_pp=dr[mask].sum()/n))
    for split,ref,model in sorted({k[:3] for k in boot_draws}):
        keys=[(split,ref,model,p) for p in PROBLEMS]
        if not all(k in boot_draws for k in keys):continue
        subset=[r for r in pair_rows if r['split']==split and r['reference']==ref and r['model']==model]
        ab,rb=np.mean([boot_draws[k] for k in keys],axis=0)
        row=dict(problem='ALL18',split=split,reference=ref,model=model)
        for k in ('n','changed_count','corrected','harmed','both_wrong_cheaper','both_wrong_dearer'):row[k]=sum(r[k] for r in subset)
        for k in ('delta_top1_pp','delta_regret_pp','delta_raw_cost'):row[k]=float(np.mean([r[k] for r in subset]))
        row.update(top1_boot_lo=np.percentile(ab,2.5),top1_boot_hi=np.percentile(ab,97.5),
                   regret_boot_lo=np.percentile(rb,2.5),regret_boot_hi=np.percentile(rb,97.5))
        pair_rows.append(row)
    return dict(model_comparisons=pair_rows,correction_cost_decomposition=sample_rows,
                shared_errors=common_rows,model_agreement=disagreement,
                universal_error_solvers=universal,unweighted_consensus=consensus)


def cost_scale_tables(loaded):
    size_rows=[];tradeoffs=[]
    for (model,split,p),a in loaded.items():
        if model not in PRIMARY or split in ('train','train_sample'):continue
        c=a['costs'];n=len(c);oracle=c.min(1)
        absolute=a['regret']/100*oracle
        # These quartiles diagnose the weighting of observed errors, not a deployment policy.
        levels=np.searchsorted(np.quantile(oracle,[.25,.5,.75]),oracle,side='right')
        for q in range(4):
            m=levels==q
            if not m.any():continue
            size_rows.append(dict(model=model,split=split,problem=p,cost_quartile=q+1,n=int(m.sum()),
                mean_reference_cost=oracle[m].mean(),reference_cost_min=oracle[m].min(),reference_cost_max=oracle[m].max(),
                mean_true_size=a['size'][m].mean() if a['size'] is not None else np.nan,
                top1_pct=a['correct'][m].mean()*100,regret_pct=a['regret'][m].mean(),
                absolute_error_share_pct=absolute[m].sum()/max(absolute.sum(),1e-15)*100,
                relative_error_share_pct=a['regret'][m].sum()/max(a['regret'].sum(),1e-15)*100,
                relative_gap_weight_share_pct=oracle[m].sum()/oracle.sum()*100,
                note='descriptive evaluation quartiles, not a fitted selection rule'))
    for p in PROBLEMS:
        for split in ('train','val','test'):
            c,y=raw(p,split);oracle=c.min(1);reg=(c/oracle[:,None]-1)*100
            for j,name in enumerate(POOLS[p]):
                best_other=np.delete(c,j,axis=1).min(1);adv=(best_other-c[:,j])/oracle*100
                win=adv>0;lose=adv<0
                tradeoffs.append(dict(problem=p,split=split,solver=name,n=len(c),native_winner_pct=(y==j).mean()*100,
                    strict_cost_winner_pct=win.mean()*100,
                    mean_win_advantage_pct=adv[win].mean() if win.any() else 0.,
                    median_win_advantage_pct=np.median(adv[win]) if win.any() else 0.,
                    mean_loss_when_not_best_pct=reg[lose,j].mean() if lose.any() else 0.,
                    loss_p95_pct=np.percentile(reg[:,j],95),always_select_regret_pct=reg[:,j].mean(),
                    recoverable_winner_gain_pp=np.maximum(adv,0).mean()))
    return size_rows,tradeoffs


def histories(specs):
    rows=[];choice=[]
    for model,spec in specs.items():
        path=spec.get('history')
        if path is None or not path.exists():continue
        h=json.loads(path.read_text())
        if not isinstance(h,list):continue
        choices=[]
        for entry in h:
            for split in ('train','val'):
                result=entry.get(split)
                if not isinstance(result,dict):continue
                records=dict(result.get('per_problem',{}))
                records['ALL18']=result.get('macro',{})
                for p,d in records.items():
                    d={k.replace('macro_',''):v for k,v in d.items()}
                    if 'top1' not in d:continue
                    row=dict(model=model,split=split,problem=p,epoch=entry['epoch'],
                        global_epoch=entry.get('global_epoch',entry['epoch']),updates=entry.get('successful_updates'),
                        top1_pct=d['top1']*100,ce=d.get('ce'),regret_pct=d.get('actual_regret_pct'),mean_cost=d.get('mean_cost'),
                        selected_epoch=spec.get('selected_epoch'))
                    rows.append(row)
                    if split=='val' and p=='ALL18':choices.append(row)
        for metric,minimize in [('ce',True),('regret_pct',True),('mean_cost',True),('top1_pct',False)]:
            available=[r for r in choices if r.get(metric) is not None]
            if not available:continue
            best=min(available,key=lambda r:r[metric]*(1 if minimize else -1))
            choice.append(dict(model=model,criterion=metric,epoch=best['epoch'],val_top1_pct=best['top1_pct'],val_regret_pct=best['regret_pct'],
                note='validation-only retrospective metric comparison; no test reselection'))
    return rows,choice


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--permutations',type=int,default=100);args=parser.parse_args()
    root=args.root;root.mkdir(parents=True,exist_ok=True)
    specs=sources();loaded={};manifest=[];missing=[]
    for model,spec in specs.items():
        for split in ('val','test','train','train_sample','LIB'):
            if split not in spec:continue
            problems=('TSP','CVRP') if model in ('NSS','R25') or split=='LIB' else PROBLEMS
            for p in problems:
                path=locate(spec[split],p,split)
                if path is None:
                    missing.append(dict(model=model,split=split,problem=p,expected=str(spec[split])))
                    continue
                a=load(path,p,split);a['problem']=p;loaded[(model,split,p)]=a
                manifest.append(dict(model=model,split=split,problem=p,n=len(a['y']),path=str(path)))
    print(f'[plateau] loaded {len(loaded)} frozen problem/split prediction sets',flush=True)
    metric_rows=[dict(model=m,split=s,problem=p,**metric(a)) for (m,s,p),a in loaded.items()]
    numeric=[k for k in metric_rows[0] if k not in ('model','split','problem','n')]
    append_macros(metric_rows,['model','split'],numeric)
    write_csv(root/'metrics.csv',metric_rows)
    scope_rows=[]
    groups={'B8_legacy_semantics':[p for p in PROBLEMS if p not in ('TSP','CVRP','ATSP') and 'B' in p],
            'nonB10':[p for p in PROBLEMS if p in ('TSP','CVRP','ATSP') or 'B' not in p],
            'MVRP15':[p for p in PROBLEMS if p not in ('TSP','CVRP','ATSP')]}
    for model,split in sorted({(r['model'],r['split']) for r in metric_rows}):
        for scope,problems in groups.items():
            selected=[r for r in metric_rows if r['model']==model and r['split']==split and r['problem'] in problems]
            if len(selected)!=len(problems):continue
            scope_rows.append(dict(model=model,split=split,scope=scope,n=sum(r['n'] for r in selected),
                                  task_count=len(problems),**{k:float(np.mean([r[k] for r in selected])) for k in numeric}))
    write_csv(root/'scope_metrics.csv',scope_rows)
    profiles,portfolios,distributions=data_tables(loaded)
    for name,rows in [('data_profiles',profiles),('fixed_portfolios',portfolios),('solver_distributions',distributions)]:write_csv(root/f'{name}.csv',rows)
    for name,rows in errors(loaded).items():write_csv(root/f'{name}.csv',rows)
    print('[plateau] margins and error budgets complete',flush=True)
    write_csv(root/'instance_signal.csv',permutations(loaded,args.permutations))
    print('[plateau] exact-size permutations complete',flush=True)
    for name,rows in comparisons(loaded).items():write_csv(root/f'{name}.csv',rows)
    size_rows,tradeoffs=cost_scale_tables(loaded)
    write_csv(root/'cost_scale_weighting.csv',size_rows);write_csv(root/'solver_tradeoffs.csv',tradeoffs)
    curve,choices=histories(specs)
    write_csv(root/'learning_curves.csv',curve);write_csv(root/'validation_selection_sensitivity.csv',choices)
    write_csv(root/'prediction_manifest.csv',manifest);write_csv(root/'missing_predictions.csv',missing)
    dump(root/'analysis_protocol.json',dict(scope='existing frozen models through R70; no R71 data',seed=2,
        analysis_only=True,neural_training=False,checkpoint_reselection=False,cost_labels='original FP64, native ind',
        macro='equal task weights, ALL18 only for complete 18-problem coverage',
        permutation=dict(repeats=args.permutations,strata='same problem and exact true size',singleton='unchanged, eligible-only separately',
            operation='random single-cycle shifts of shuffled group rows; not all permutations uniformly'),
        bootstrap='400 paired instance resamples per problem; descriptive intervals, not across-seed significance or adaptive-test valid inference',
        portfolios='trained on train costs/frequencies only; k>1 diagnostics require multiple executions',
        limitations=['legacy B semantic conflict deferred by user','many previous test exposures; exploratory rather than confirmatory',
                     'one seed per model; epochs and versions are not independent repetitions','original NSS paper per-instance predictions unavailable',
                     'accuracy softmax calibration is descriptive; score semantics differ across losses',
                     'same data identity assumed from raw arrays/pool alignment; no new solver validity audit'],
        prediction_sets=len(manifest)))
    print(f'[plateau] wrote analysis to {root}',flush=True)


if __name__=='__main__':main()

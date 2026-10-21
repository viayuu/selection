"""R86: one CPU-only sampled diagnostic of a frozen shared readout vector."""

import argparse
import csv
import json
import os
import time
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = ''
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '2'

import numpy as np
import torch

from code.unified_selector.registry import PROBLEMS
from .multitask_probe import task_seed
from .pairwise_objective import inputs_only
from .r76_experiment import open_cache
from .r82_context import fit_context, metadata_values
from .r84_experiment import load_checkpoint


BASE = Path(__file__).resolve().parent / 'runs/R84_new90k_attention'
OUTPUT = BASE.parent / 'R86_readout_geometry_probe'
DATA = Path('/public/home/shiys/selection_data_new90k_B_filtered_single_reld_mt_seed2_20261018')
GUARDS = [
    'Frozen R84A shared final Linear: there are no method-specific classifier weight norms. Only hidden-feature radius and angle to the same w are measured.',
    'CPU FP32, 64 TRAIN + 64 VAL samples/task (or the recorded input/timing-only budget reduction), seed2. Not a formal AMP evaluation, fitted model, threshold search, TEST or LIB result.',
    'Maximin utilities permit common shifts. Angle-only argmax is a fixed-parameter counterfactual, not a calibrated/Bayes-equivalent inference rule.',
    'A favorable change is only priority evidence for a new end-to-end normalized-head experiment, not causal proof or an accepted inference change. A negative result does not rule out normalized architectures.',
    'No native winners in a sampled method group means undefined recall, never zero. All18 macros equally weight tasks; mixed raw mean_cost is descriptive only.',
]


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name('.'+path.name+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    temporary.replace(path)


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fields); writer.writeheader(); writer.writerows(rows)


def stratified_sample(groups, count, seed):
    """Proportional allocation by input strata, without any performance labels."""
    rng = np.random.default_rng(seed)
    groups = np.asarray(groups)
    axis, sizes = np.unique(groups, return_counts=True)
    desired = sizes * (count / len(groups))
    allocation = np.floor(desired).astype(int)
    # Random tie breaks avoid privileging the lexical order of a named stratum.
    order = np.lexsort((rng.random(len(axis)), -(desired-allocation)))
    allocation[order[:count-int(allocation.sum())]] += 1
    selected = np.concatenate([rng.choice(np.flatnonzero(groups==group), size=n, replace=False)
                               for group, n in zip(axis, allocation) if n])
    return rng.permutation(selected)


def stable_rank(scores, ids):
    ids = np.asarray(ids)
    return np.lexsort((np.broadcast_to(ids, scores.shape), -scores), axis=1)


def decompose(features, weight, bias):
    features, weight = features.astype(np.float64), weight.astype(np.float64)
    radius, wn = np.linalg.norm(features, axis=-1), np.linalg.norm(weight)
    cosine = (features @ weight) / np.maximum(radius * wn, np.finfo(np.float64).tiny)
    return radius, cosine, wn*radius*cosine+bias


def ratio(numerator, denominator):
    return 100.*numerator/denominator if denominator else None


def prepare_samples():
    plans = {}; specs = {}
    for problem in PROBLEMS:
        with np.load(DATA/(problem+'train')/'input_metadata.npz', allow_pickle=False) as source:
            train_metadata = {key:source[key] for key in ('scale','distribution','capacity_type')}
        _, spec = fit_context(train_metadata, problem, bins=10)
        specs[problem] = spec
        for split in ('train','val'):
            with np.load(DATA/(problem+split)/'input_metadata.npz', allow_pickle=False) as source:
                meta = {key:source[key] for key in ('scale','distribution','capacity_type','source_index')}
            size, distribution, capacity = metadata_values(meta, problem)
            bins = np.searchsorted(spec['log_size_cutpoints'], np.log(size), side='right')
            keys = np.asarray([f'{b}|{d}|{c}' for b,d,c in zip(bins,distribution,capacity)])
            seed = task_seed(2,problem)+(0 if split=='train' else 1000003)
            indices = stratified_sample(keys,64,seed)
            smaller = indices[stratified_sample(keys[indices],32,seed+17)]
            plans[problem,split] = dict(indices=indices, smaller=smaller, scale=meta['scale'],
                                        distribution=meta['distribution'], capacity_type=meta['capacity_type'],
                                        source_index=meta['source_index'], strata=keys,
                                        nodes=size+int(problem not in ('TSP','ATSP')))
    return plans, specs


def work_units(plan, problem, rows):
    indices = plan['indices'] if rows==64 else plan['smaller']
    lengths = np.sort(plan['nodes'][indices])
    cap = 8 if problem in ('TSP','CVRP') else 32
    return sum(len(lengths[i:i+cap])*float(lengths[i:i+cap].max())**2 for i in range(0,len(lengths),cap))


def run(output):
    started = time.monotonic(); output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    def progress(phase, **extra):
        save(output/'progress.json',dict(experiment='R86',phase=phase,pid=os.getpid(),cpu_threads=2,
             gpu_used=False,goal_complete=False,elapsed_seconds=time.monotonic()-started,**extra))
    progress('preparing_input_only_samples')
    print(f'R86 CPU diagnostic started PID={os.getpid()} threads=2 GPU=disabled',flush=True)
    lock = read(BASE/'A_baseline/final_test_lock.json')
    if lock['selected_update'] != 37440:
        raise ValueError('R86 is fixed to historical R84A update37440')
    model, checkpoint = load_checkpoint(BASE/'A_baseline/best.pt','cpu')
    if checkpoint['successful_updates'] != 37440 or checkpoint['state_hash'] != lock['state_hash']:
        raise ValueError('Frozen model differs from the requested R84A lock')
    model.eval().requires_grad_(False)
    del checkpoint
    linear = model.score_head[-1]
    if not isinstance(linear,torch.nn.Linear) or linear.out_features!=1:
        raise ValueError('Expected the shared scalar final Linear')
    weight = linear.weight.detach().numpy()[0].copy()
    bias = float(linear.bias.detach()[0]) if linear.bias is not None else 0.
    if np.linalg.norm(weight)==0:
        raise ValueError('Shared readout weight has zero norm')
    plans, contexts = prepare_samples()
    source_manifest = read(BASE/'prepared/source_cache_manifest.json')
    baseline = read(BASE/'prepared/common_full_train_sbs.json')
    frozen = read(BASE/'comparison_artifacts/target_list.json')
    targets = {(r['problem'],r['method']) for r in frozen['targets']}
    samples, chosen_rows, budget = {},64,dict(policy='Timing/input-shape decision before any outcome analysis',max_seconds=600)
    captured = []
    handle = linear.register_forward_pre_hook(lambda module,args:captured.append(args[0].detach().float().numpy().copy()))
    with torch.inference_mode():
        for problem in PROBLEMS:
            for split in ('train','val'):
                plan = plans[problem,split]
                indices = plan['indices'] if chosen_rows==64 else plan['smaller']
                indices = indices[np.argsort(plan['nodes'][indices],kind='stable')]
                cap = 8 if problem in ('TSP','CVRP') else 32
                progress('forward',problem=problem,split=split,rows_per_problem_split=chosen_rows,
                         completed_problem_splits=len(samples),total_problem_splits=36)
                cache = open_cache(BASE/'input_cache',split,problem,source_manifest,DATA)
                features, scores, utility = [],[],[]
                forward_started = time.monotonic()
                try:
                    for start in range(0,len(indices),cap):
                        captured.clear()
                        out = model(inputs_only(cache.batch(indices[start:start+cap],device='cpu')))
                        if len(captured)!=1:
                            raise ValueError('Expected one shared score-head call per batch')
                        features.append(captured[0]);scores.append(out['logits'].numpy().copy());utility.append(out['utility'].numpy().copy())
                    f, logits, u = np.concatenate(features),np.concatenate(scores),np.concatenate(utility)
                    if not samples:
                        elapsed = time.monotonic()-forward_started
                        total_units = sum(work_units(v,p,64) for (p,s),v in plans.items())
                        forecast = (time.monotonic()-started)+1.5*elapsed*total_units/work_units(plan,problem,64)
                        budget.update(initial_probe_seconds=elapsed,forecast_seconds=forecast,decision_before_outcome_analysis=True)
                        if forecast>600:
                            chosen_rows=32
                            keep=np.isin(indices,plan['smaller'])
                            indices,f,logits,u=indices[keep],f[keep],logits[keep],u[keep]
                            budget['reduction']='64 to 32, fixed from runtime and input shapes before labels/metrics were analyzed'
                        else: budget['reduction']=None
                        budget['actual_rows_per_problem_split']=chosen_rows
                    samples[problem,split]=dict(indices=indices,features=f,logits=logits,utility=u,
                        costs=np.asarray(cache.raw['costs'][indices]).copy(),winner=np.asarray(cache.raw['winner'][indices]).copy(),
                        pool=list(cache.pool),pool_ids=np.asarray(cache.pool_ids))
                finally:cache.close()
                print(f'forward {problem} {split} n={len(indices)} complete ({len(samples)}/36); no outcomes analyzed yet',flush=True)
    handle.remove()
    sample_manifest = dict(experiment='R86',diagnostic_only=True,seed=2,checkpoint_sha256=lock['checkpoint_sha256'],
        selected_update=37440,data_root=str(DATA),cache_root=str(BASE/'input_cache'),
        cpu_dtype='float32; no autocast/AMP',threads=2,batch_caps={'TSP':8,'CVRP':8,'others':32},
        sampling='Proportional TRAIN-log-size-quantile/distribution/capacity strata; no labels used',
        budget=budget,context_definitions=contexts,shared_weight_norm=float(np.linalg.norm(weight)),shared_bias=bias,
        tests_or_lib_read=False,fit_or_threshold_search=False,interpretation_guards=GUARDS,
        samples={p+'_'+s:[dict(index=int(i),source_index=int(plans[p,s]['source_index'][i]),
                   scale=int(plans[p,s]['scale'][i]),distribution=str(plans[p,s]['distribution'][i]),
                   capacity_type=str(plans[p,s]['capacity_type'][i])) for i in data['indices']]
                 for (p,s),data in samples.items()})
    save(output/'sample_manifest.json',sample_manifest)
    progress('analyzing_frozen_forward',rows_per_problem_split=chosen_rows)
    metrics, geometry, pair_diagnostics = [],[],[]
    for (problem,split),data in samples.items():
        n,k=data['costs'].shape; rows=np.arange(n); y=data['winner']
        radius,cosine,reconstructed=decompose(data['features'],weight,bias)
        error=float(np.abs(reconstructed-data['utility']).max())
        np.testing.assert_allclose(reconstructed,data['utility'],rtol=3e-5,atol=3e-5)
        original_rank,angle_rank=stable_rank(data['logits'],data['pool_ids']),stable_rank(cosine,data['pool_ids'])
        original,angle=original_rank[:,0],angle_rank[:,0]
        ranks=np.empty_like(original_rank);np.put_along_axis(ranks,original_rank,np.broadcast_to(np.arange(k)+1,(n,k)),axis=1)
        oracle=data['costs'].min(1);original_cost=data['costs'][rows,original]
        target_slots=np.array([m for m,name in enumerate(data['pool']) if (problem,name) in targets],dtype=int)
        truth_target=np.isin(y,target_slots)
        for rule,pred in [('original_maximin',original),('angle_only_counterfactual',angle)]:
            cost=data['costs'][rows,pred];good=pred==y;old_good=original==y;chosen_target=np.isin(pred,target_slots)
            target_tp=int((good & truth_target).sum());target_support=int(truth_target.sum())
            sbs=data['costs'][:,data['pool'].index(baseline[problem]['sbs_solver'])]
            metrics.append(dict(split=split.upper(),problem=problem,rule=rule,n=n,native_correct=int(good.sum()),
                top1_pct=100.*float(good.mean()),mean_cost=float(cost.mean()),sbs_cost=float(sbs.mean()),oracle_cost=float(oracle.mean()),
                vs_sbs_pct=100.*float(cost.mean()/sbs.mean()-1),vs_oracle_pct=100.*float(cost.mean()/oracle.mean()-1),
                actual_regret_pct=100.*float(((cost-oracle)/oracle).mean()),
                corrected=int((~old_good & good).sum()),harmed=int((old_good & ~good).sum()),
                target_winners=target_support,target_correct=target_tp,target_selected=int(chosen_target.sum()),
                target_recall_pct=ratio(target_tp,target_support),target_precision_pct=ratio(target_tp,int(chosen_target.sum())),
                new_target_false_picks=int((chosen_target & ~good & (pred!=original)).sum()),
                regret_delta_pp=100.*float(((cost-original_cost)/oracle).mean()),
                aggregation='one sampled problem; native FP64 labels',max_decomposition_abs_error=error))
        missed=original!=y
        angular=cosine[rows,y]-cosine[rows,original]
        radial_term=np.linalg.norm(weight)*(radius[rows,y]-radius[rows,original])*(cosine[rows,y]+cosine[rows,original])/2
        directional_term=np.linalg.norm(weight)*angular*(radius[rows,y]+radius[rows,original])/2
        pair_diagnostics.append(dict(problem=problem,split=split.upper(),missed=int(missed.sum()),
            winner_higher_cosine_but_lower_utility=int((missed & (angular>0)).sum()),
            median_radial_term_on_misses=float(np.median(radial_term[missed])) if missed.any() else None,
            median_directional_term_on_misses=float(np.median(directional_term[missed])) if missed.any() else None))
        for m,method in enumerate(data['pool']):
            truth=y==m;support=int(truth.sum());old_selected=original==m;new_selected=angle==m
            groups=[('all','ALL',np.ones(n,dtype=bool)),('true_winner','ALL',truth),('nonwinner','ALL',~truth)]
            groups += [('all',str(rank),ranks[:,m]==rank) for rank in range(1,k+1)]
            groups += [('true_winner',str(rank),truth & (ranks[:,m]==rank)) for rank in range(1,k+1)]
            for population,rank,mask in groups:
                count=int(mask.sum())
                geometry.append(dict(problem=problem,split=split.upper(),method=method,population=population,
                    original_predicted_rank=rank,count=count,native_winner_support=support,
                    feature_norm_mean=float(radius[mask,m].mean()) if count else None,
                    feature_norm_std=float(radius[mask,m].std()) if count else None,
                    cosine_to_shared_w_mean=float(cosine[mask,m].mean()) if count else None,
                    cosine_to_shared_w_std=float(cosine[mask,m].std()) if count else None,
                    utility_mean=float(data['utility'][mask,m].mean()) if count else None,
                    original_selected=int(old_selected.sum()),angle_selected=int(new_selected.sum()),
                    original_correct=int((truth & old_selected).sum()),angle_correct=int((truth & new_selected).sum()),
                    original_recall_pct=ratio(int((truth & old_selected).sum()),support),
                    angle_recall_pct=ratio(int((truth & new_selected).sum()),support),
                    angle_precision_pct=ratio(int((truth & new_selected).sum()),int(new_selected.sum())),
                    rescued=int((truth & ~old_selected & new_selected).sum()),
                    new_false_picks=int((new_selected & ~truth & ~old_selected).sum()),frozen_target=(problem,method) in targets))
    for split in ('TRAIN','VAL'):
        for rule in ('original_maximin','angle_only_counterfactual'):
            part=[r for r in metrics if r['split']==split and r['rule']==rule]
            total=dict(part[0],problem='ALL',aggregation='equal18-task means; raw cost descriptive only')
            for key in ('n','native_correct','corrected','harmed','target_winners','target_correct','target_selected','new_target_false_picks'):
                total[key]=sum(r[key] for r in part)
            for key in ('top1_pct','mean_cost','sbs_cost','oracle_cost','vs_sbs_pct','vs_oracle_pct','actual_regret_pct','regret_delta_pp'):
                total[key]=float(np.mean([r[key] for r in part]))
            total['target_recall_pct']=ratio(total['target_correct'],total['target_winners'])
            total['target_precision_pct']=ratio(total['target_correct'],total['target_selected'])
            total['max_decomposition_abs_error']=max(r['max_decomposition_abs_error'] for r in part)
            metrics.append(total)
    write_csv(output/'metrics.csv',metrics);write_csv(output/'method_geometry.csv',geometry)
    sample_manifest['pair_diagnostics']=pair_diagnostics
    sample_manifest['elapsed_forward_seconds']=time.monotonic()-started
    save(output/'sample_manifest.json',sample_manifest)
    render(output,metrics,geometry,pair_diagnostics,chosen_rows)
    progress('complete',complete=True,rows_per_problem_split=chosen_rows,total_evaluated=chosen_rows*36,
             pdf=str(output/'PROBE.pdf'),training=False,test_or_lib=False,interpretation='priority evidence only; no new inference rule')
    print(json.dumps(dict(phase='complete',rows_per_problem_split=chosen_rows,seconds=time.monotonic()-started,
                         overall=[r for r in metrics if r['problem']=='ALL'])),flush=True)


def render(output,metrics,geometry,pairs,count):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    plt.rcParams.update({'font.family':'DejaVu Sans','pdf.fonttype':42})
    overall=[r for r in metrics if r['problem']=='ALL']
    def fmt(value,digits=4):return 'undefined' if value is None else f'{value:.{digits}f}'
    lines=['# R86 frozen readout geometry diagnostic','',f'R84A update37440; seed2; {count} TRAIN + {count} VAL per task; CPU FP32, 2 threads.','',
           '| Split | Rule | Top1 % | Actual regret % | mean_cost | Target recall % | Corrected / harmed |',
           '|---|---|---:|---:|---:|---:|---:|']
    lines += [f"| {r['split']} | {r['rule']} | {r['top1_pct']:.4f} | {r['actual_regret_pct']:.6f} | {r['mean_cost']:.9f} | {fmt(r['target_recall_pct'])} | {r['corrected']} / {r['harmed']} |" for r in overall]
    lines += ['', '## Interpretation guards','']+['- '+line for line in GUARDS]
    lines += ['', 'The exact symmetric pair decomposition is radial = ||w||(r_y-r_s)(cos_y+cos_s)/2 and directional = ||w||(cos_y-cos_s)(r_y+r_s)/2. Their sum equals u_y-u_s. This is descriptive within the frozen representation, not causal attribution.','',
              'All group/rank feature statistics and undefined-recall groups are in method_geometry.csv. Sample IDs, recorded input strata, TRAIN-only boundaries, timing-only budget decision and pair summaries are in sample_manifest.json.','']
    (output/'RESULTS.md').write_text('\n'.join(lines))
    def table(pdf,title,headers,rows,notes):
        fig=plt.figure(figsize=(15,9));fig.text(.04,.94,title,fontsize=18,weight='bold',color='#104f55')
        ax=fig.add_axes([.04,.29,.92,.55]);ax.axis('off')
        t=ax.table(cellText=rows,colLabels=headers,bbox=[0,0,1,1],cellLoc='left');t.auto_set_font_size(False);t.set_fontsize(9)
        for (row,_),cell in t.get_celld().items():cell.set_edgecolor('#d9e2e3');cell.set_facecolor('#deedef' if row==0 else '#f4f7f8' if row%2 else 'white')
        import textwrap
        y=.23
        for note in notes:
            wrapped=textwrap.fill(note,155);fig.text(.04,y,wrapped,fontsize=9,va='top');y-=.029*(wrapped.count('\n')+1)+.009
        pdf.savefig(fig);plt.close(fig)
    with PdfPages(output/'PROBE.pdf') as pdf:
        table(pdf,'R86 | Frozen shared-readout geometry; diagnostic only',
              ['Split','Rule','Top1 %','Regret %','mean_cost','Target recall %','Corrected / harmed'],
              [[r['split'],'original' if r['rule']=='original_maximin' else 'angle-only',fmt(r['top1_pct']),fmt(r['actual_regret_pct'],6),fmt(r['mean_cost'],7),fmt(r['target_recall_pct']),f"{r['corrected']} / {r['harmed']}"] for r in overall],GUARDS)
        focus=[r for r in geometry if r['population']=='all' and r['original_predicted_rank']=='ALL' and r['frozen_target'] and r['native_winner_support']>0]
        focus=sorted(focus,key=lambda r:(r['split'],-r['native_winner_support'],r['problem'],r['method']))[:18]
        lookup={(r['problem'],r['split'],r['method']):r for r in geometry if r['population']=='true_winner' and r['original_predicted_rank']=='ALL'}
        table(pdf,'R86 | Sampled frozen-target method geometry',
              ['Split / task','Method','Winners','TP orig / angle','Feature norm on wins','Cosine on wins','Rescued / new FP'],
              [[r['split']+' / '+r['problem'],r['method'],str(r['native_winner_support']),f"{r['original_correct']} / {r['angle_correct']}",
                fmt(lookup[r['problem'],r['split'],r['method']]['feature_norm_mean']),fmt(lookup[r['problem'],r['split'],r['method']]['cosine_to_shared_w_mean']),f"{r['rescued']} / {r['new_false_picks']}"] for r in focus],
              ['At most18 nonempty method/split examples shown by sampled winner support. Every method and every original predicted rank is in CSV.',
               'Methods with no sampled native winners have undefined recall, including any zero-choice groups.',
               'One shared w is used for every method; these are hidden-feature statistics, not per-class weight norms.'])
        rows=[]
        for problem in PROBLEMS:
            cells=[problem]
            for split in ('TRAIN','VAL'):
                old=next(r for r in metrics if r['problem']==problem and r['split']==split and r['rule']=='original_maximin')
                new=next(r for r in metrics if r['problem']==problem and r['split']==split and r['rule']=='angle_only_counterfactual')
                pair=next(r for r in pairs if r['problem']==problem and r['split']==split)
                cells += [fmt(new['top1_pct']-old['top1_pct'],2),fmt(new['regret_delta_pp'],4),f"{pair['winner_higher_cosine_but_lower_utility']} / {pair['missed']}"]
            rows.append(cells)
        table(pdf,'R86 | Per-task sampled changes and fixed-representation tradeoffs',
              ['Task','TRAIN dTop1 pp','TRAIN dRegret pp','Better angle / missed','VAL dTop1 pp','VAL dRegret pp','Better angle / missed'],rows,
              ['Better angle / missed: native winner has higher cosine than the originally selected method but loses original utility. This need not make it angle-argmax.',
               'The utility bias/common-shift freedom prevents treating this decomposition as causal or uniquely identifiable.',
               'No TEST/LIB, fitting, threshold/coefficient search, solver execution or GPU use. The research goal remains active.'])


def self_test():
    features=np.array([[[1.,2.],[3.,4.]]]);weight=np.array([.3,-.2]);bias=.7
    radius,cosine,reconstructed=decompose(features,weight,bias)
    np.testing.assert_allclose(reconstructed,features@weight+bias,atol=1e-14)
    _,rescaled,_=decompose(features*np.array([[[2.],[5.]]]),weight,bias)
    np.testing.assert_allclose(cosine,rescaled)
    groups=np.repeat(np.arange(8),[10,20,30,40,50,60,70,80])
    ids=stratified_sample(groups,64,2)
    assert len(ids)==len(np.unique(ids))==64
    np.testing.assert_array_equal(ids,stratified_sample(groups,64,2))
    assert ratio(0,0) is None and ratio(0,2)==0.
    print('PASS: 3 focused checks (decomposition/radial invariance, input-only sampling, undefined support)')


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--output',type=Path,default=OUTPUT);parser.add_argument('--self-test',action='store_true')
    args=parser.parse_args()
    if args.self_test:self_test()
    else:
        try:run(args.output)
        except BaseException as error:
            save(args.output/'progress.json',dict(experiment='R86',phase='failed',pid=os.getpid(),error=str(error),gpu_used=False,goal_complete=False))
            raise

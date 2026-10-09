"""Clean OVRPTW two-solver prediction; no source features, retrieval or auxiliary loss."""

import argparse
import json
import os
import pickle
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .V4Model import InstanceEncoder, _masked_max, _masked_mean
from .behavior_memory import attach_query_ids, refresh_memory
from .local_geometry import FIELDS as GEOMETRY_FIELDS, local_geometry
from .multitask_probe import dump, file_hash, gather_batch, state_hash
from .r47_model import load_r47_checkpoint
from .r48_common import (ROOT, R47, PAIR, VARIANTS, binary_labels, binary_metrics,
                         load_split, pair_winner, plan_order, read_csv, tree_features, write_csv)
from .r48_common import read_indexed_costs
from .tensor_loader import TensorBatchLoader
from .train import configure_torch, make_loader, set_seed


INPUT_FIELDS = ('kind', 'problem_id', 'node', 'node_mask', 'matrix', 'n', 'cbits',
                'problem_desc', 'coord_dist', 'node_geom', 'pool_ids', 'solver_mask', 'base_id')


def clean_inputs(batch):
    return {k: batch[k] for k in INPUT_FIELDS if k in batch}


class BinaryInstanceClassifier(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.instance_encoder = InstanceEncoder(**params)
        d = params['embedding_dim']
        self.head = nn.Sequential(nn.Linear(2*d+1, d), nn.GELU(), nn.Dropout(params['dropout']), nn.Linear(d, 2))

    def forward(self, batch):
        nodes, mask = self.instance_encoder(clean_inputs(batch))
        feature = torch.cat([_masked_mean(nodes, mask), _masked_max(nodes, mask),
                             batch['n'].float().log1p()[:, None]/6.], -1)
        return self.head(feature)


def prepare_data(root, device):
    data, loaders = {}, {}
    geometry_stats = json.loads(Path('code/V4/runs/R39_performance_model/geometry_cache/stats.json').read_text())
    for split in ('train', 'val'):
        instances, raw, costs, nodes = load_split(split)
        loader = make_loader('OVRPTW', split, 128, 0, shuffle=False, cache_device=device)
        loader.drop_last = False
        np.testing.assert_array_equal(loader.batch['pool_ids'].cpu(), raw['pool_ids'])
        np.testing.assert_array_equal(loader.batch['costs'].cpu(), raw['costs'].astype(np.float32))
        np.testing.assert_array_equal(loader.lengths, nodes+1)
        np.testing.assert_array_equal(loader.batch['ind'].cpu(), raw['winner'])
        loader.batch['binary_label'] = torch.as_tensor(binary_labels(costs), device=device)
        loader.batch['instance_index'] = torch.arange(loader.size, device=device)
        cache = Path('code/V4/runs/R39_performance_model/geometry_cache') / ('OVRPTW_'+split+'.pt')
        if geometry_stats['sources'][split+'/OVRPTW'] != raw['data_hash'] or geometry_stats['definition_hash'] != file_hash(
                Path(__file__).parent/'local_geometry.py'):
            raise ValueError('Geometry preprocessing no longer matches the current dataset')
        geom = torch.load(cache, map_location='cpu', weights_only=True)
        if geom.shape != (*loader.batch['node'].shape[:2], 12):
            raise ValueError('Wrong geometry row/node correspondence')
        loader.batch['node_geom'] = geom.to(device)
        attach_query_ids({'OVRPTW': loader})
        data[split] = dict(instances=instances, raw=raw, costs=costs, nodes=nodes, labels=binary_labels(costs))
        loaders[split] = loader
    values = loaders['train'].batch['node_geom'][loaders['train'].batch['node_mask']].double()
    mean = values.mean(0)
    std = values.std(0, unbiased=False).clamp_min(1e-6)
    stats = dict(mean=mean.cpu().tolist(), std=std.cpu().tolist(), fields=list(GEOMETRY_FIELDS),
                 fitted_on='OVRPTW train valid nodes only', count=len(values))
    dump(root/'binary_geometry_stats.json', stats)
    return data, loaders, stats


def write_config(root, data, stats):
    import sklearn
    prior = json.loads((root/'audit_plan.json').read_text())
    old = json.loads(Path('code/V4/runs/R47_behavior_memory/B_behavior_memory_seed2/args.json').read_text())
    fields = ('embedding_dim', 'head_num', 'qkv_dim', 'ff_hidden_dim', 'encoder_layer_num',
              'dropout', 'stats_dim', 'rezero', 'ignore_coord_dist', 'local_geometry', 'geometry_mode')
    params = {k: old['model_params'][k] for k in fields}
    params.update(node_only=True, sdpa=True)
    config = dict(seed=2, problem='OVRPTW', pair=list(PAIR), only_splits=['train', 'val'], test_read=False,
        labels='0: raw FP64 MOEL<MTL; 1: MTL<MOEL; exact ties excluded from all binary fitting/metrics',
        counts={s: dict(total=len(d['labels']), non_ties=int((d['labels']>=0).sum()),
                ties=int((d['labels']<0).sum()), class0=int((d['labels']==0).sum()), class1=int((d['labels']==1).sum()),
                retained_non_ties_with_other_seven_pool_winner=int(((d['labels']>=0) & ~np.isin(d['raw']['winner'],
                    [d['raw']['pool'].index(name) for name in PAIR])).sum()))
                for s, d in data.items()},
        data={s: {k:d['raw'][k] for k in ('pool', 'pool_ids', 'data_hash', 'label_hash')} for s,d in data.items()},
        sampling='all non-ties, including examples whose seven-pool winner is a different method',
        solver_config=prior['solvers'], fixed_audit_sample=prior['splits'],
        route_checker=dict(implementation='r48_common.check_open_tw_route; independent NumPy FP64',
            open_routes=True, capacity=1., speed=1., service_begin_must_be_within_customer_tw=True,
            depot_deadline=False, route_return_cost=False, feasibility_tolerance=1e-5,
            cost_atol=2e-5, cost_rtol=2e-6,
            precision='Original input FP64 and exact runtime FP32 values independently recomputed in FP64'),
        neural=dict(model_params=params, architecture='current four-layer InstanceEncoder + masked mean/max/size + 257->128->2 MLP',
            initialization='scratch seed2; no previous selector weights', loss='unweighted CE only', batch_size=128,
            optimizer='fresh AdamW', lr=1e-4, weight_decay=1e-4, dropout=.1, gradient_clip=1.,
            maximum_epochs=60, early_stop_patience=10, selection='strict minimum full validation binary CE',
            schedule='fixed LR; no warmup or plateau adjustment', dtype='FP32; TF32 disabled; no AMP',
            drop_last=False, class_resampling=False, augmentation=False, geometry_stats=stats),
        tree=dict(algorithm='sklearn HistGradientBoostingClassifier', learning_rate=.1, max_iter=300,
            max_leaf_nodes=15, min_samples_leaf=40, l2_regularization=1., max_bins=255,
            early_stopping=False, random_state=2, tuning='one preset; no search/no validation selection',
            features=list(tree_features(data['train']['instances'][0])),
            definitions='Counts/sums; geometric/depot/nearest-neighbor distances; demands; service and time-window moments/quantiles; directed local time pressure. See tree_features source.',
            statistic_order=['mean', 'population_std', 'q10', 'q50', 'q90', 'max'], travel_speed=1.,
            excludes='No neural embedding, solver outputs, label, winner, instance index or input node order'),
        size_prior=dict(boundaries='train customer-count quartiles, repeated boundaries merged',
            label='binary outcome frequency among non-ties', smoothing='add-one Laplace for finite CE',
            ties='argmax probabilities; equal probabilities choose class0'),
        frozen_R47=dict(checkpoint=str(R47.resolve()), sha256=file_hash(R47), epoch=31,
            candidate_pool='original full seven-method OVRPTW pool; only output columns are compared',
            score='actual maximin logits; two selected logits normalized only for binary CE',
            numerics='Original R47 configure_torch(): TF32/high matmul precision; restored before reference replay only',
            memory='train only; same best weights rebuild the OVRPTW memory; self/base-copy exclusion'),
        runtime=dict(torch=torch.__version__, sklearn=sklearn.__version__, gpu=torch.cuda.get_device_name(0),
                     visible_gpu=os.environ.get('CUDA_VISIBLE_DEVICES'), slurm_job=os.environ.get('SLURM_JOB_ID')),
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip(),
        wandb=dict(project='selector', entity='yjkds-southern-university-of-science-technology', mode='offline'))
    source_dir = root/'source_launch'
    source_dir.mkdir(exist_ok=True)
    names = ('r48_common.py', 'r48_solver_audit.py', 'r48_binary.py', 'r48_analysis.py', 'test_r48.py', 'run_v4_r48.sh')
    config['source_hashes'] = {}
    for name in names:
        path = Path(__file__).parent/name
        if path.exists():
            config['source_hashes'][name] = file_hash(path)
            shutil.copy2(path, source_dir/name)
    config['dependency_hashes']={str(p.resolve()):file_hash(p) for p in
        [Path(__file__).parent/name for name in ('V4Model.py','local_geometry.py','r47_model.py','behavior_memory.py',
                                                'dual_stream.py','pairwise_selector.py','performance_targets.py')]
        + [Path(__file__).parents[1]/'unified_selector'/name for name in ('data.py','registry.py')]}
    cost_files=[]
    for split in data:
        for column,name in enumerate(PAIR):
            record=prior['solvers'][name]['cost_files'][split]
            if file_hash(record['path'])!=record['sha256']:
                raise ValueError('Original per-solver result file changed')
            values=read_indexed_costs(record['path'])
            np.testing.assert_array_equal(values,data[split]['costs'][:,column])
            cost_files.append(dict(split=split,solver=name,rows=len(values),raw_label_cost_column_exact_match=True,**record))
    config['raw_cost_file_alignment']=cost_files
    dump(root/'config.json', config)
    return config


@torch.no_grad()
def frozen_reference(root, data, loaders, device):
    model, checkpoint = load_r47_checkpoint(R47, device)
    original = model.memory['OVRPTW'].summary.clone()
    np.testing.assert_array_equal(model.memory['OVRPTW'].base_ids.cpu(), loaders['train'].batch['base_id'].cpu())
    np.testing.assert_array_equal(model.memory['OVRPTW'].pool_ids.cpu(), data['train']['raw']['pool_ids'])
    memory = refresh_memory(model, {'OVRPTW': loaders['train']}, 128)
    error = float((model.memory['OVRPTW'].summary-original).abs().max())
    if error > 1e-4:
        raise ValueError(f'Frozen R47 reference summaries do not reproduce its selected checkpoint; max_abs_error={error}')
    memory['rebuild_max_abs_error'] = error
    memory['checkpoint_epoch'] = checkpoint['epoch']
    dump(root/'frozen_memory_check.json', memory)
    pair_columns = [data['train']['raw']['pool'].index(name) for name in PAIR]
    scores = {}
    for split, loader in loaders.items():
        values = []
        for batch in TensorBatchLoader(loader.batch, 128, False, False):
            output = model(clean_inputs(batch))
            if (model.memory['OVRPTW'].base_ids[output['neighbor_indices']] == batch['base_id'][:, None]).any():
                raise ValueError('Frozen R47 query retrieved its own behavior')
            values.append(output['logits'].cpu().numpy())
        all_scores = np.concatenate(values)
        scores[split] = all_scores[:, pair_columns]
        np.savez_compressed(root/('r47_scores_'+split+'.npz'), scores=scores[split], all_scores=all_scores,
                            indices=np.arange(loader.size), pool=np.asarray(data[split]['raw']['pool']))
    plan = json.loads((root/'audit_plan.json').read_text())
    solver_rows = {(r['split'],int(r['instance_index']),r['variant'],r['solver']):r
                   for r in read_csv(root/'solver_audit.csv')}
    rows = []
    for split in ('train', 'val'):
        indices = torch.tensor(plan['splits'][split]['indices'])
        for variant in VARIANTS:
            batch = gather_batch(loaders[split], indices)
            batch['node'] = batch['node'].clone()
            batch['node_geom'] = batch['node_geom'].clone()
            for slot, index in enumerate(indices.tolist()):
                order = torch.as_tensor(plan_order(plan, split, index, variant), device=device) + 1
                count = len(order)
                batch['node'][slot, 1:count+1] = loaders[split].batch['node'][index, order]
                batch['node_geom'][slot, 1:count+1] = loaders[split].batch['node_geom'][index, order]
            current = model(clean_inputs(batch))['logits'][:, pair_columns].cpu().numpy()
            for index, logits in zip(indices.tolist(), current):
                first, second = [solver_rows[(split,index,variant,name)] for name in PAIR]
                costs = [float(r['rerun_cost']) for r in (first,second)]
                recomputed = [float(r['recomputed_cost']) for r in (first,second)]
                old = data[split]['costs'][index]
                rows.append(dict(split=split, instance_index=index, customers=int(data[split]['nodes'][index]),
                    variant=variant, historical_MOEL=old[0], historical_MTL=old[1],
                    historical_binary_label=pair_winner(*old), rerun_MOEL=costs[0], rerun_MTL=costs[1],
                    rerun_binary_winner=pair_winner(*costs), recomputed_MOEL=recomputed[0], recomputed_MTL=recomputed[1],
                    recomputed_binary_winner=pair_winner(*recomputed),
                    MOEL_feasible=first['feasible'], MTL_feasible=second['feasible'],
                    R47_MOEL_score=float(logits[0]), R47_MTL_score=float(logits[1]),
                    R47_margin=float(logits[0]-logits[1]), R47_binary_prediction=int(logits.argmax()),
                    R47_margin_change_from_original=float((logits[0]-logits[1])-(scores[split][index,0]-scores[split][index,1])),
                    R47_max_score_change_from_original=float(np.abs(logits-scores[split][index]).max())))
    write_csv(root/'permutation_audit.csv', rows)
    del model
    torch.cuda.empty_cache()
    return scores


@torch.no_grad()
def predict_neural(model, loader):
    model.eval()
    scores = [model(clean_inputs(batch)).cpu().numpy() for batch in TensorBatchLoader(loader.batch, 128, False, False)]
    return np.concatenate(scores)


def train_neural(root, config, data, loaders, stats, device):
    directory = root/'neural_seed2'
    directory.mkdir(exist_ok=True)
    if (directory/'last.pt').exists():
        raise ValueError('Preserve the existing binary experiment; do not silently retrain')
    set_seed(2)
    model = BinaryInstanceClassifier(config['neural']['model_params']).to(device)
    with torch.no_grad():
        branch = model.instance_encoder.geometry_residual
        branch.mean.copy_(torch.tensor(stats['mean'], device=device))
        branch.std.copy_(torch.tensor(stats['std'], device=device))
    if any('solver' in name or 'retriev' in name or 'joint' in name for name,_ in model.named_parameters()):
        raise ValueError('Clean classifier contains an unwanted solver/retrieval branch')
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    rows = torch.from_numpy(np.flatnonzero(data['train']['labels'] >= 0))
    generator = torch.Generator().manual_seed(2)
    best, bad_epochs, history, updates = float('inf'), 0, [], 0
    import wandb
    wandb_run = wandb.init(project='selector', entity='yjkds-southern-university-of-science-technology',
                           name='R48_clean_binary_seed2', dir=str(directory), config=config['neural'], mode='offline')
    try:
        for epoch in range(61):
            started = time.monotonic()
            train_loss, seen, gradient = 0., 0, 0.
            if epoch:
                model.train()
                order = rows[torch.randperm(len(rows), generator=generator)]
                for indices in order.split(128):
                    batch = gather_batch(loaders['train'], indices)
                    optimizer.zero_grad(set_to_none=True)
                    logits = model(clean_inputs(batch))
                    loss = F.cross_entropy(logits, batch['binary_label'])
                    if not torch.isfinite(loss):
                        raise ValueError('Nonfinite clean CE')
                    loss.backward()
                    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                    if epoch == 1 and seen == 0:
                        witness = {}
                        for name, module in (('encoder',model.instance_encoder),('head',model.head)):
                            values = [p.grad for p in module.parameters() if p.grad is not None]
                            witness[name] = float(sum(g.abs().sum() for g in values))
                            if not witness[name] > 0:
                                raise ValueError('No gradient in '+name)
                        dump(directory/'gradient_witness.json', witness)
                    optimizer.step()
                    updates += 1
                    seen += len(indices)
                    train_loss += float(loss.detach())*len(indices)
                    gradient += float(norm)*len(indices)
                if seen != len(rows):
                    raise ValueError('Training did not cover every non-tie')
            predictions = {s:predict_neural(model,loader) for s,loader in loaders.items()}
            result = dict(epoch=epoch, updates=updates,
                train=binary_metrics(predictions['train'],data['train']['costs']),
                val=binary_metrics(predictions['val'],data['val']['costs']),
                optimization_ce=train_loss/seen if seen else None,
                grad_norm=gradient/seen if seen else None, seconds=time.monotonic()-started, lr=1e-4)
            improved = result['val']['ce'] < best
            if improved:
                best, bad_epochs = result['val']['ce'], 0
            elif epoch:
                bad_epochs += 1
            history.append(result)
            checkpoint = dict(model=model.state_dict(), epoch=epoch, updates=updates,
                model_params=config['neural']['model_params'], geometry_stats=stats,
                config=config['neural'], metrics=result, initialization='scratch seed2')
            if improved:
                torch.save(checkpoint, directory/'best.pt')
            if epoch:
                torch.save(dict(checkpoint, optimizer=optimizer.state_dict()), directory/'last.pt')
            dump(directory/'history.json',history)
            print(f'[R48 neural] epoch={epoch} train acc={result["train"]["accuracy"]:.5f} '
                  f'CE={result["train"]["ce"]:.5f}; val acc={result["val"]["accuracy"]:.5f} '
                  f'CE={result["val"]["ce"]:.5f} regret={result["val"]["pair_regret_pct"]:.5f}% '
                  f'best={improved} bad={bad_epochs} seconds={result["seconds"]:.1f}',flush=True)
            wandb_run.log({'epoch':epoch, 'updates':updates, 'train/ce':result['train']['ce'],
                'train/accuracy':result['train']['accuracy'], 'val/ce':result['val']['ce'],
                'val/accuracy':result['val']['accuracy'], 'val/pair_regret_pct':result['val']['pair_regret_pct'],
                'lr':1e-4, 'grad_norm':result['grad_norm']})
            if epoch and bad_epochs >= 10:
                break
        checkpoint = torch.load(directory/'best.pt', map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model'],strict=True)
        predictions = {s:predict_neural(model,loader) for s,loader in loaders.items()}
        dump(directory/'selection.json',dict(best_epoch=checkpoint['epoch'], stop_epoch=history[-1]['epoch'],
            successful_updates=updates, trainable_parameters=sum(p.numel() for p in model.parameters()),
            checkpoint_sha256=file_hash(directory/'best.pt'), validation_ce=best))
        for split in predictions:
            np.savez_compressed(directory/(split+'_predictions.npz'), scores=predictions[split],
                indices=np.arange(len(predictions[split])), costs=data[split]['costs'], labels=data[split]['labels'])
        return predictions
    finally:
        wandb_run.finish()


def fit_tree_and_prior(root,data,config):
    from sklearn.ensemble import HistGradientBoostingClassifier
    features = {}
    fields = config['tree']['features']
    for split, values in data.items():
        dictionaries = [tree_features(instance) for instance in values['instances']]
        if any(list(d) != fields for d in dictionaries):
            raise ValueError('Independent tree feature order changed')
        features[split] = np.array([[row[k] for k in fields] for row in dictionaries], dtype=np.float64)
    keep = data['train']['labels'] >= 0
    tree = HistGradientBoostingClassifier(learning_rate=.1,max_iter=300,max_leaf_nodes=15,
        min_samples_leaf=40,l2_regularization=1.,early_stopping=False,random_state=2)
    tree.fit(features['train'][keep], data['train']['labels'][keep])
    with (root/'tree_seed2.pkl').open('wb') as stream:
        pickle.dump(tree,stream,pickle.HIGHEST_PROTOCOL)
    tree_scores = {s:np.log(tree.predict_proba(x).clip(1e-15,1)) for s,x in features.items()}
    cuts = np.unique(np.quantile(data['train']['nodes'], [.25,.5,.75]))
    buckets = {s:np.searchsorted(cuts,d['nodes'],side='left') for s,d in data.items()}
    probabilities, records = [], []
    for bucket in range(len(cuts)+1):
        labels = data['train']['labels'][(buckets['train']==bucket)&keep]
        counts = np.bincount(labels,minlength=2)
        probability = (counts+1)/(len(labels)+2)
        probabilities.append(probability)
        records.append(dict(bucket=bucket,class_counts=counts.tolist(),n=len(labels),
            probability=probability.tolist(),prediction=int(probability.argmax())))
    dump(root/'size_prior.json',dict(cuts=cuts.tolist(),buckets=records,fitted_on='train non-ties',smoothing='Laplace +1'))
    prior_scores = {s:np.log(np.asarray(probabilities)[b]) for s,b in buckets.items()}
    return tree_scores, prior_scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--stage',choices=('all','reference','train'),default='all')
    args=parser.parse_args()
    status=json.loads((args.root/'solver_audit_summary.json').read_text())
    if not status['train_allowed'] or (args.root/'blocking_anomalies.json').exists():
        raise ValueError('Input/route audit has not cleared binary training')
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.set_float32_matmul_precision('highest')
    data,loaders,stats=prepare_data(args.root,args.device)
    config=write_config(args.root,data,stats)
    if args.stage in ('all','reference'):
        configure_torch()
        frozen=frozen_reference(args.root,data,loaders,args.device)
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
        torch.set_float32_matmul_precision('highest')
    else:
        frozen={s:np.load(args.root/('r47_scores_'+s+'.npz'))['scores'] for s in data}
    if args.stage=='reference':
        return
    neural=train_neural(args.root,config,data,loaders,stats,args.device)
    tree,prior=fit_tree_and_prior(args.root,data,config)
    predictions=dict(neural=neural,tree=tree,size_prior=prior,frozen_R47=frozen)
    rows=[]
    for name,values in predictions.items():
        for split,scores in values.items():
            rows.append(dict(model=name,split=split,**binary_metrics(scores,data[split]['costs'])))
    write_csv(args.root/'binary_comparison.csv',rows)
    for split,values in data.items():
        records=[]
        for index,(costs,label,n) in enumerate(zip(values['costs'],values['labels'],values['nodes'])):
            record=dict(instance_index=index,customers=int(n),MOEL_cost=float(costs[0]),MTL_cost=float(costs[1]),
                binary_label=int(label),included_in_binary_metrics=bool(label>=0),
                old_seven_pool_winner=values['raw']['pool'][values['raw']['winner'][index]])
            for name in predictions:
                scores=predictions[name][split][index]
                exp=np.exp(scores-scores.max())
                record.update({name+'_prediction':int(scores.argmax()),name+'_score0':float(scores[0]),
                               name+'_score1':float(scores[1]),name+'_probability1':float(exp[1]/exp.sum())})
            records.append(record)
        write_csv(args.root/(split+'_predictions.csv'),records)
    dump(args.root/'binary_complete.json',dict(complete=True,models=list(predictions),test_read=False,
        n_train=len(data['train']['labels']),n_val=len(data['val']['labels'])))


if __name__=='__main__':
    main()

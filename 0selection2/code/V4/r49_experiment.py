"""Two scratch, CE-only binary runs; select solely with the new development CE."""

import argparse
import copy
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .multitask_probe import dump, file_hash, gather_batch, state_hash
from .r48_binary import BinaryInstanceClassifier, clean_inputs
from .r48_common import PAIR, binary_metrics, write_csv
from .r49_data import (ROOT, MODELS, GROUPS, ShuffledIndexStream, fit_geometry_stats,
                       load_data, load_manifest, prepare)
from .train import set_seed


def configure_numeric():
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.set_float32_matmul_precision('highest')


def parameter_hash(model):
    return state_hash(dict(model.named_parameters()))


def development_improved(metrics, best):
    return metrics['development']['ce'] < best


def make_config(root, manifest):
    previous=json.loads(Path('code/V4/runs/R48_core_diagnosis/config.json').read_text())
    config=dict(seed=2, problem='OVRPTW', pair=list(PAIR), models=list(MODELS),
        model_params=previous['neural']['model_params'],
        architecture='Unchanged R48 BinaryInstanceClassifier: four-layer InstanceEncoder + binary MLP',
        initialization='One freshly initialized seed2 template copied to both models; no trained weights loaded',
        loss='unweighted CE only; original FP64 pair labels; exact ties excluded',
        batch_size=128, successful_updates=4000, evaluate_every=200,
        sampling='Independent seeded shuffled passes; tails carried forward; every update uses 128 examples',
        presentations_per_model=512000, optimizer='fresh AdamW', learning_rate=1e-4,
        weight_decay=1e-4, dropout=.1, gradient_clip=1., amp=False, tf32=False,
        scheduler=None, early_stopping=False, augmentation=False, class_resampling=False,
        selection='Strictly lowest full development CE among fixed update evaluations, including u0000',
        original_validation_used_for_selection=False, internal_used_for_selection=False,
        normalization='Geometry mean/std fit separately on each actual non-tie training subset, padding excluded',
        other_input_statistics='Existing per-instance statistics only; no additional dataset-fitted normalization',
        distribution_statistics='R48 explicit statistics are descriptive only, not extra network inputs',
        bootstrap='2000 instance resamples, seed4902, conditional on fixed trained models; not seed uncertainty',
        split_manifest_sha256=file_hash(root/'split_manifest.json'),
        raw_source_hashes=manifest['sources'],
        reference_r48_config_sha256=file_hash(Path('code/V4/runs/R48_core_diagnosis/config.json')),
        git_parent=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        runtime=dict(torch=torch.__version__,gpu=torch.cuda.get_device_name(0),
            visible_gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),slurm_job=os.environ.get('SLURM_JOB_ID')),
        wandb=dict(project='selector',entity='yjkds-southern-university-of-science-technology',mode='offline'),
        test_read=False, sources={})
    snapshot=root/'source_launch'
    snapshot.mkdir(exist_ok=True)
    for name in ('r49_data.py','r49_experiment.py','r49_analysis.py','test_r49.py','run_v4_r49.sh'):
        path=Path(__file__).parent/name
        config['sources'][name]=file_hash(path)
        shutil.copy2(path,snapshot/name)
    config['dependencies']={name:file_hash(Path(__file__).parent/name) for name in
        ('r48_binary.py','r48_common.py','V4Model.py','local_geometry.py','tensor_loader.py','behavior_memory.py')}
    dump(root/'config.json',config)
    return config


@torch.no_grad()
def evaluate(model, data, loaders, sets):
    model.eval()
    metrics,scores={},{}
    for name,(source,rows) in sets.items():
        logits=[]
        for indices in torch.as_tensor(rows).split(128):
            logits.append(model(clean_inputs(gather_batch(loaders[source],indices))).cpu().numpy())
        scores[name]=np.concatenate(logits)
        metrics[name]=binary_metrics(scores[name],data[source]['costs'][rows])
    return metrics,scores


def save_predictions(directory, scores, sets, data):
    for name,logits in scores.items():
        source,rows=sets[name]
        np.savez_compressed(directory/(name+'_predictions.npz'), scores=logits, indices=rows,
            labels=data[source]['labels'][rows], costs=data[source]['costs'][rows],
            customers=data[source]['sizes'][rows],base_ids=data[source]['base_ids'][rows])


def train_one(root,name,config,template,data,loaders,manifest):
    directory=root/name
    directory.mkdir(exist_ok=True)
    if (directory/'last.pt').exists():
        raise ValueError('Do not overwrite an existing R49 training run')
    train_key='A_2000' if name==MODELS[0] else 'training_pool'
    nominal=np.asarray(manifest['sets'][train_key]['indices'])
    rows=nominal[data['train']['labels'][nominal]>=0]
    stats=fit_geometry_stats(loaders['train'],rows)
    dump(directory/'normalization.json',stats)
    device=loaders['train'].batch['node'].device
    model=BinaryInstanceClassifier(config['model_params']).to(device)
    model.load_state_dict(copy.deepcopy(template),strict=True)
    initial_hash=parameter_hash(model)
    with torch.no_grad():
        geometry=model.instance_encoder.geometry_residual
        geometry.mean.copy_(torch.tensor(stats['mean'],device=device))
        geometry.std.copy_(torch.tensor(stats['std'],device=device))
    if parameter_hash(model)!=config['initial_trainable_parameter_sha256']:
        raise ValueError('A/B trainable initializations differ')
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-4)
    stream=ShuffledIndexStream(rows,seed=2)
    schedule=torch.stack([stream.next_batch(128) for _ in range(4000)])
    np.savez_compressed(directory/'sampling_plan.npz',indices=schedule.numpy())
    sets={'train':('train',nominal)}
    sets.update({key:(manifest['sets'][key]['source_split'],np.asarray(manifest['sets'][key]['indices']))
                 for key in GROUPS[1:]})
    args=dict(config,model=name,training_nominal=len(nominal),training_strict=len(rows),
        initial_trainable_parameter_sha256=initial_hash,normalization_sha256=file_hash(directory/'normalization.json'),
        sampling_plan_sha256=file_hash(directory/'sampling_plan.npz'),
        sampling_unique_instances=len(torch.unique(schedule)),passes_started=stream.passes_started)
    dump(directory/'args.json',args)
    import wandb
    run=wandb.init(project='selector',entity='yjkds-southern-university-of-science-technology',
        name='R49_'+name,dir=str(directory),config=args,mode='offline')
    set_seed(2)
    best=float('inf')
    history=[]
    loss_sum=grad_sum=0.
    start=time.monotonic()
    try:
        for updates in range(4001):
            if updates:
                model.train()
                batch=gather_batch(loaders['train'],schedule[updates-1])
                optimizer.zero_grad(set_to_none=True)
                logits=model(clean_inputs(batch))
                loss=F.cross_entropy(logits,batch['binary_label'])
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite CE')
                loss.backward()
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
                if updates==1:
                    witness={key:float(sum(p.grad.abs().sum() for p in module.parameters() if p.grad is not None))
                        for key,module in (('encoder',model.instance_encoder),('head',model.head))}
                    if min(witness.values())<=0:
                        raise ValueError('Missing encoder/head gradient')
                    dump(directory/'gradient_witness.json',witness)
                optimizer.step()
                loss_sum+=float(loss.detach())
                grad_sum+=float(norm)
            if updates%200:
                continue
            metrics,scores=evaluate(model,data,loaders,sets)
            improved=development_improved(metrics,best)
            if improved:
                best=metrics['development']['ce']
            record=dict(updates=updates,metrics=metrics,lr=1e-4,
                optimization_ce=loss_sum/200 if updates else None,
                grad_norm=grad_sum/200 if updates else None,seconds=time.monotonic()-start,
                selection_improved=improved)
            history.append(record)
            checkpoint=dict(model=model.state_dict(),updates=updates,model_params=config['model_params'],
                geometry_stats=stats,config=args,metrics=metrics,
                selection='development CE only; original validation and internal holdout never select')
            if improved:
                torch.save(checkpoint,directory/'best.pt')
            torch.save(dict(checkpoint,optimizer=optimizer.state_dict()),directory/'last.pt')
            dump(directory/'history.json',history)
            summary='; '.join(f'{key}:acc={value["accuracy"]:.4f} CE={value["ce"]:.5f} '
                f'regret={value["pair_regret_pct"]:.4f}%' for key,value in metrics.items())
            print(f'[R49 {name}] update={updates} {summary} dev_best={improved}',flush=True)
            log={'updates':updates,'lr':1e-4,'grad_norm':record['grad_norm'],'optimization_ce':record['optimization_ce']}
            for key,values in metrics.items():
                log.update({key+'/'+field:values[field] for field in ('accuracy','ce','pair_regret_pct','mean_cost')})
            run.log(log,step=updates)
            loss_sum=grad_sum=0.
        best_checkpoint=torch.load(directory/'best.pt',map_location=device,weights_only=False)
        model.load_state_dict(best_checkpoint['model'],strict=True)
        metrics,scores=evaluate(model,data,loaders,sets)
        for key in metrics:
            for field in metrics[key]:
                if not np.isclose(metrics[key][field],best_checkpoint['metrics'][key][field],atol=1e-12,rtol=0):
                    raise ValueError('Best checkpoint evaluation did not replay')
        save_predictions(directory,scores,sets,data)
        selection=dict(model=name,best_updates=best_checkpoint['updates'],successful_updates=4000,
            presentations=4000*128,checkpoint_sha256=file_hash(directory/'best.pt'),
            development_ce=metrics['development']['ce'],selection_uses_internal_or_original_validation=False,
            initialization_sha256=initial_hash,trainable_parameters=sum(p.numel() for p in model.parameters()))
        dump(directory/'selection.json',selection)
        return metrics
    finally:
        run.finish()


def train(root):
    if any((root/name/'last.pt').exists() for name in MODELS):
        raise ValueError('R49 already has trained outputs; refusing an accidental rerun')
    configure_numeric()
    data,loaders=load_data('cuda:0')
    manifest=load_manifest(root,data)
    config=make_config(root,manifest)
    set_seed(2)
    template_model=BinaryInstanceClassifier(config['model_params'])
    config['initial_trainable_parameter_sha256']=parameter_hash(template_model)
    template=copy.deepcopy(template_model.state_dict())
    dump(root/'config.json',config)
    results=[]
    for name in MODELS:
        metrics=train_one(root,name,config,template,data,loaders,manifest)
        for split,values in metrics.items():
            results.append(dict(model=name,split=split,**values))
    write_csv(root/'comparison.csv',results)
    dump(root/'training_complete.json',dict(complete=True,models=list(MODELS),
        successful_updates_each=4000,presentations_each=512000,test_read=False))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--stage',choices=('prepare','train','all'),default='all')
    args=parser.parse_args()
    if args.stage in ('prepare','all'):
        prepare(args.root)
    if args.stage in ('train','all'):
        train(args.root)


if __name__=='__main__':
    main()

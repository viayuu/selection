"""R53: paired independent specialists selected by integrated validation regret."""

import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import torch

from code.unified_selector.registry import PROBLEMS
from .multitask_probe import dump, file_hash, gather_batch, state_hash
from .pair_specialist import (PAIR, PairSpecialist, apply_specialist, binary_objective,
                             ordinal_scores, specialist_metrics)
from .performance_evaluation import decision_metrics
from .r42_experiment import CostController
from .r48_common import write_csv
from .r53_data import (BASELINE, MVRP, ROOT, aggregate, baseline_metrics, build_loaders,
                      fit_geometry, gate_budget, grouped, make_plan, raw_training, read_baseline)
from .tensor_loader import TensorBatchLoader
from .train import set_seed
from .training_monitor import capture_rng, restore_rng


ARM_NAMES = dict(A='A_ce_seed2', B='B_cost_sensitive_seed2')
SOURCES = ('pair_specialist.py', 'r53_data.py', 'r53_experiment.py', 'r53_analysis.py',
           'test_r53.py', 'run_v4_r53.sh')


def model_params():
    previous = json.loads(Path('code/V4/runs/R48_core_diagnosis/config.json').read_text())
    return dict(previous['neural']['model_params'])


def cpu_prepare(root):
    root.mkdir(parents=True, exist_ok=True)
    train_raw, mean_weight = raw_training(root)
    val_raw, baseline = read_baseline('val', root)
    budget = gate_budget(val_raw, baseline, root)
    best = json.loads((BASELINE/'best_eval.json').read_text())
    observed = baseline_metrics(val_raw, baseline)['groups']['ALL']
    for key, value in observed.items():
        np.testing.assert_allclose(value, best['val']['macro']['macro_'+key], rtol=0, atol=1e-10)
    protocol = dict(experiment='R53', seed=2, arms=ARM_NAMES, problems=list(MVRP),
        frozen_baseline=dict(checkpoint=str((BASELINE/'best.pt').resolve()),
            sha256=file_hash(BASELINE/'best.pt'), human_epoch=best['epoch'],
            original_selection='validation macro_vs_sbs_pct', val=observed),
        model_params=model_params(), label='1: MOEL<MTL; 0: MTL<MOEL; exact ties excluded',
        weights='abs(MOEL-MTL)/min(full legal pool), original FP64; no clipping or class balancing',
        training_weight_mean=mean_weight,
        weight_mean_population='all non-tied training instances across all 15 MVRP tasks',
        losses=dict(A='mean BCEWithLogits', B='mean(weight / fixed training mean * BCEWithLogits)'),
        gate='frozen R45A predicted Top2 names are exactly RELD_MOEL and RELD_MTL',
        score_rule='positive specialist logit -> MOEL; zero -> lower global ID (MOEL)',
        ranking='swap original Top2 only; all remaining ranks and untriggered decisions unchanged',
        checkpoint_selection='strict minimum integrated full 18-task validation actual_regret_pct, trained epochs only',
        controller=dict(metric='integrated validation actual_regret_pct', min_delta=.001,
            lr_patience=3, lr_factor=.5, minimum_lr=1e-6, min_epochs=15, early_stop_patience=8),
        batch=128, maximum_epochs=40, warmup_epochs=3, learning_rate=1e-4, weight_decay=1e-4,
        dropout=.1, optimizer='fresh AdamW', gradient_clip=1., accumulation=False,
        sampling='all non-ties, complete shuffled coverage with tail; interleaved tasks',
        initialization='fresh independent instance encoder/head; exactly identical A/B parameter values',
        supervision_excludes=['full-pool CE', 'pairwise', 'risk', 'R-Drop', 'R52 start costs'],
        precision='FP16 AMP + existing SDPA for training; FP32 eval, TF32 off; original FP64 cost accounting',
        baseline_training=False, test_policy='one full test replay per locked validation-best; no test-driven changes',
        development_splits=['train', 'val'], test_used_for_selection=False,
        success_screen=dict(top1_improvement_pp=5., no_regret_increase=True,
            alternative_regret_relative_reduction_pct=15., alternative_no_top1_decline=True),
        reference='https://www.cs.ubc.ca/~kevinlb/papers/2012-SATzilla2012-Solver-Description.pdf',
        reference_boundary='independent cost-sensitive pair classification only; not SATzilla voting/forests',
        git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        source_hashes={name:file_hash(Path(__file__).parent/name) for name in SOURCES},
        dependency_hashes={name:file_hash(Path(__file__).parent/name) for name in
                          ('V4Model.py', 'local_geometry.py', 'tensor_loader.py', 'performance_targets.py')},
        wandb=dict(project='selector', entity='yjkds-southern-university-of-science-technology', mode='offline'))
    path = root/'protocol.json'
    if path.exists():
        old = json.loads(path.read_text())
        for key in ('training_weight_mean', 'frozen_baseline', 'model_params', 'source_hashes'):
            if old[key] != protocol[key]:
                raise ValueError(f'R53 preparation changed: {key}; preserve existing experiment')
    dump(path, protocol)
    print('[R53 gate budget] '+json.dumps(budget), flush=True)
    return protocol, train_raw, val_raw, baseline


@torch.no_grad()
def predict(model, loaders):
    model.eval()
    return {p: np.concatenate([model(batch).float().cpu().numpy()
                for batch in TensorBatchLoader(loader.batch, 256, False, False)])
            for p, loader in loaders.items()}


def binary_summary(scores, raw, training_mean):
    per_problem = {p:specialist_metrics(scores[p], raw[p], training_mean) for p in MVRP}
    keys = ('accuracy', 'ce', 'weighted_ce', 'pair_regret_pct', 'extra_full_pool_regret_pct')
    macro = {key:float(np.mean([r[key] for r in per_problem.values()])) for key in keys}
    macro['n'] = sum(r['n'] for r in per_problem.values())
    macro['ties'] = sum(r['ties'] for r in per_problem.values())
    return dict(macro=macro, per_problem=per_problem)


def integrated_evaluation(scores, raw, baseline, directory=None):
    if directory is not None:
        directory.mkdir(parents=True, exist_ok=True)
    per_problem = {}
    for p in PROBLEMS:
        r, base = raw[p], baseline[p]
        z = scores[p] if p in MVRP else np.zeros(len(r['winner']), dtype=np.float32)
        order, gate = apply_specialist(base['order'], r['pool'], z)
        metrics, pred = decision_metrics(ordinal_scores(order), r)
        metrics['triggered'] = int(gate.sum())
        per_problem[p] = metrics
        np.testing.assert_array_equal(np.sort(order[:, :2], axis=1), np.sort(base['order'][:, :2], axis=1))
        np.testing.assert_array_equal(order[:, 2:], base['order'][:, 2:])
        np.testing.assert_array_equal(order[~gate], base['order'][~gate])
        if directory is not None:
            np.savez_compressed(directory/(p+'.npz'), indices=np.arange(len(pred)), pred=pred,
                order=order, baseline_order=base['order'], baseline_logits=base['logits'],
                baseline_pred=base['order'][:, 0], gate=gate, expert_logit=z, winner=r['winner'],
                costs=r['costs'], pool=np.asarray(r['pool']), pool_ids=np.asarray(r['pool_ids']))
    return dict(per_problem=per_problem, groups=grouped(per_problem))


def update(model, optimizer, scaler, batch, weighted, mean_weight):
    before = capture_rng()
    for retry in range(20):
        restore_rng(before)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.float16):
            z = model(batch)
        loss = binary_objective(z, batch['binary_label'], batch['cost_weight'], weighted, mean_weight)
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite specialist loss')
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        old_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        if scaler.get_scale() >= old_scale:
            if not torch.isfinite(norm):
                raise RuntimeError('Nonfinite gradient was not skipped')
            return dict(loss=float(loss.detach()), grad_norm=float(norm), retries=retry)
        print(f'[AMP retry] same batch, scale {old_scale}->{scaler.get_scale()}', flush=True)
    raise RuntimeError('Persistent overflow on unchanged specialist batch')


def real_precheck(model, loaders, root):
    model.eval()
    sample = gather_batch(loaders['OVRPTW'], torch.arange(4))
    with torch.no_grad():
        original = model(sample)
        changed = dict(sample, costs=sample['costs'].flip(-1), ind=sample['ind']*0,
                       binary_label=sample['binary_label']*0, cost_weight=sample['cost_weight']+10)
        np.testing.assert_array_equal(original.cpu(), model(changed).cpu())
        padded = dict(sample)
        padded['node'] = torch.cat((sample['node'], torch.full_like(sample['node'][:, :3], 777)), dim=1)
        padded['node_mask'] = torch.cat((sample['node_mask'], torch.zeros_like(sample['node_mask'][:, :3])), dim=1)
        padded['node_geom'] = torch.cat((sample['node_geom'], torch.full_like(sample['node_geom'][:, :3], 777)), dim=1)
        padding_error = float((original-model(padded)).abs().max())
        if padding_error > 2e-5:
            raise ValueError(f'Padding changed specialist output: {padding_error}')
    valid = sample['binary_label'] >= 0
    if not valid.all():
        indices = torch.where(loaders['OVRPTW'].batch['binary_label'].cpu() >= 0)[0][:4]
        sample = gather_batch(loaders['OVRPTW'], indices)
    loss = binary_objective(model(sample), sample['binary_label'], sample['cost_weight'], True, .01)
    loss.backward()
    norm = lambda part: float(sum(p.grad.float().square().sum() for p in part.parameters() if p.grad is not None).sqrt())
    witness = dict(label_isolation=True, padding_max_abs_error=padding_error,
        encoder_grad_norm=norm(model.instance_encoder), head_grad_norm=norm(model.head),
        parameters=sum(p.numel() for p in model.parameters()),
        no_solver_parameters=not any('solver' in n or 'retriev' in n for n, _ in model.named_parameters()))
    if not witness['no_solver_parameters'] or not witness['encoder_grad_norm'] > 0 or not witness['head_grad_norm'] > 0:
        raise ValueError('Specialist precheck failed')
    model.zero_grad(set_to_none=True)
    dump(root/'precheck.json', witness)


def train_arm(arm, root, protocol, raw, loaders, baseline, geometry, plan, initial):
    directory = root/ARM_NAMES[arm]
    directory.mkdir(parents=True, exist_ok=True)
    if (directory/'result.json').exists():
        print(f'[R53{arm}] completed arm retained', flush=True)
        return
    set_seed(2)
    model = PairSpecialist(protocol['model_params']).cuda()
    model.load_state_dict(initial, strict=True)
    initial_hash = state_hash(model.state_dict())
    optimizer = torch.optim.AdamW(model.parameters(), lr=protocol['learning_rate'], weight_decay=protocol['weight_decay'])
    scaler = torch.amp.GradScaler('cuda', init_scale=1024.)
    controller = CostController(**{k:protocol['controller'][k] for k in ('min_delta', 'min_epochs', 'lr_patience')},
                                patience=protocol['controller']['early_stop_patience'])
    history, step, start_epoch, best_cost, peak = [], 0, 0, float('inf'), protocol['learning_rate']
    args = dict(protocol, arm=arm, geometry=geometry, initialization_sha256=initial_hash,
                trainable_parameters=sum(p.numel() for p in model.parameters()))
    dump(directory/'config.json', args)
    if (directory/'last.pt').exists():
        old = torch.load(directory/'last.pt', map_location='cpu', weights_only=False)
        if old['initialization_sha256'] != initial_hash:
            raise ValueError('Resume initialization changed')
        model.load_state_dict(old['model'])
        optimizer.load_state_dict(old['optimizer'])
        scaler.load_state_dict(old['scaler'])
        controller.__dict__.update(old['controller'])
        history, step, start_epoch, best_cost, peak = old['history'], old['step'], old['epoch'], old['best_cost'], old['peak']
        restore_rng(old['rng'])
    else:
        set_seed(2)
        initial_scores = predict(model, loaders['val'])
        initial_eval = dict(binary=binary_summary(initial_scores, raw['val'], protocol['training_weight_mean']),
            integrated=integrated_evaluation(initial_scores, raw['val'], baseline))
        dump(directory/'initial_eval.json', initial_eval)
        set_seed(2)
    import wandb
    run = wandb.init(project=protocol['wandb']['project'], entity=protocol['wandb']['entity'],
                     name='R53'+arm+'_seed2', dir=str(directory.resolve()), mode='offline', config=args)
    started = time.monotonic()
    warmup = 3*len(plan['orders'][0])
    for epoch in range(start_epoch, protocol['maximum_epochs']):
        human_epoch = epoch+1
        model.train()
        offsets = {p:0 for p in MVRP}
        seen = {p:0 for p in MVRP}
        counts = {p:0 for p in MVRP}
        loss_total, examples, grad_total, retries = 0., 0, 0., 0
        epoch_start = time.monotonic()
        for p in plan['orders'][epoch]:
            perm = plan['permutations'][epoch][p]
            positions = perm[offsets[p]:offsets[p]+128]
            indices = plan['valid_indices'][p][positions]
            if step < warmup:
                lr = protocol['learning_rate']*(step+1)/warmup
            else:
                lr = peak
            for group in optimizer.param_groups:
                group['lr'] = lr
            batch = gather_batch(loaders['train'][p], indices)
            stats = update(model, optimizer, scaler, batch, arm == 'B', protocol['training_weight_mean'])
            offsets[p] += len(indices)
            seen[p] += len(indices)
            counts[p] += 1
            step += 1
            loss_total += stats['loss']*len(indices)
            examples += len(indices)
            grad_total += stats['grad_norm']
            retries += stats['retries']
            if step % 200 == 0:
                print(f'[R53{arm}] epoch={human_epoch} successful_update={step} lr={lr:.3g} loss={stats["loss"]:.5f}', flush=True)
        for p in MVRP:
            if seen[p] != len(plan['valid_indices'][p]):
                raise ValueError(f'Incomplete non-tie training coverage: {p}')
        scores = predict(model, loaders['val'])
        binary_val = binary_summary(scores, raw['val'], protocol['training_weight_mean'])
        val = integrated_evaluation(scores, raw['val'], baseline)
        cost = val['groups']['ALL']['actual_regret_pct']
        strict_best = cost < best_cost
        if strict_best:
            best_cost = cost
        halve, stop = controller.observe(cost, human_epoch)
        if halve:
            peak = max(protocol['controller']['minimum_lr'], peak*.5)
        train_eval = None
        if human_epoch % 5 == 0 or stop or human_epoch == protocol['maximum_epochs']:
            train_scores = predict(model, loaders['train'])
            train_eval = binary_summary(train_scores, raw['train'], protocol['training_weight_mean'])
        row = dict(epoch=human_epoch, successful_updates=step, loss=loss_total/examples,
            grad_norm=grad_total/len(plan['orders'][epoch]), amp_retries=retries, learning_rate=lr,
            next_peak_lr=peak, training_examples=examples, per_problem_updates=counts, per_problem_seen=seen,
            epoch_seconds=time.monotonic()-epoch_start, val=val, binary_val=binary_val,
            binary_train=train_eval, new_strict_best=strict_best, early_stop=stop)
        history.append(row)
        checkpoint = dict(model={k:v.detach().cpu() for k,v in model.state_dict().items()},
            optimizer=optimizer.state_dict(), scaler=scaler.state_dict(), rng=capture_rng(),
            controller=dict(controller.__dict__), epoch=human_epoch, step=step, history=history,
            best_cost=best_cost, peak=peak, initialization_sha256=initial_hash, config=args)
        if strict_best:
            torch.save(checkpoint, directory/'best.pt')
            dump(directory/'best_eval.json', row)
        torch.save(checkpoint, directory/'last.pt')
        dump(directory/'history.json', history)
        run.log(dict(epoch=human_epoch, successful_updates=step, loss=row['loss'], learning_rate=lr,
            grad_norm=row['grad_norm'], val_actual_regret_pct=cost,
            val_top1=val['groups']['ALL']['top1'], specialist_val_ce=binary_val['macro']['ce'],
            specialist_val_accuracy=binary_val['macro']['accuracy']))
        print(f'[R53{arm}] epoch={human_epoch} val Top1={val["groups"]["ALL"]["top1"]*100:.4f}% '
              f'actual_regret={cost:.6f}% binary_acc={binary_val["macro"]["accuracy"]*100:.3f}% '
              f'best={best_cost:.6f}% time={row["epoch_seconds"]:.1f}s stop={stop}', flush=True)
        if stop:
            break
    best = torch.load(directory/'best.pt', map_location='cpu', weights_only=False)
    model.load_state_dict(best['model'])
    best_scores = predict(model, loaders['val'])
    best_val = integrated_evaluation(best_scores, raw['val'], baseline, directory/'predictions'/'val')
    np.testing.assert_allclose(best_val['groups']['ALL']['actual_regret_pct'], best['best_cost'], rtol=0, atol=1e-10)
    best_binary = binary_summary(best_scores, raw['val'], protocol['training_weight_mean'])
    best_train = binary_summary(predict(model, loaders['train']), raw['train'], protocol['training_weight_mean'])
    result = dict(arm=arm, best_epoch=best['epoch'], final_epoch=history[-1]['epoch'],
        best_validation=best_val, best_binary_val=best_binary, best_binary_train=best_train,
        final=history[-1], last5_val={k:float(np.mean([r['val']['groups']['ALL'][k] for r in history[-5:]]))
                                  for k in best_val['groups']['ALL']},
        elapsed_seconds=time.monotonic()-started, initialization_sha256=initial_hash,
        best_checkpoint_sha256=file_hash(directory/'best.pt'), successful_updates=step)
    dump(directory/'result.json', result)
    run.summary.update(dict(best_epoch=best['epoch'], best_val_actual_regret_pct=best['best_cost'], final_epoch=history[-1]['epoch']))
    run.finish()
    del model, optimizer
    torch.cuda.empty_cache()


def locked_test(root, protocol):
    lock = {}
    for arm, name in ARM_NAMES.items():
        directory = root/name
        result = json.loads((directory/'result.json').read_text())
        actual_hash = file_hash(directory/'best.pt')
        if actual_hash != result['best_checkpoint_sha256']:
            raise ValueError('Best changed after validation selection')
        lock[arm] = dict(checkpoint=str((directory/'best.pt').resolve()), sha256=actual_hash,
                         epoch=result['best_epoch'], validation_actual_regret=result['best_validation']['groups']['ALL']['actual_regret_pct'])
    path = root/'locked_checkpoints.json'
    if path.exists() and json.loads(path.read_text()) != lock:
        raise ValueError('Cannot reselect a checkpoint after test')
    dump(path, lock)
    # Test is first opened only after both validation-selected weights are locked.
    raw, baseline = read_baseline('test', root)
    loaders = build_loaders(raw, 'test', 'cuda')
    dump(root/'baseline_test_metrics.json', baseline_metrics(raw, baseline))
    for arm, name in ARM_NAMES.items():
        directory = root/name
        if (directory/'test_result.json').exists():
            continue
        checkpoint = torch.load(directory/'best.pt', map_location='cpu', weights_only=False)
        model = PairSpecialist(protocol['model_params']).cuda()
        model.load_state_dict(checkpoint['model'])
        start = time.monotonic()
        scores = predict(model, loaders)
        torch.cuda.synchronize()
        seconds = time.monotonic()-start
        metrics = integrated_evaluation(scores, raw, baseline, directory/'predictions'/'test')
        dump(directory/'test_result.json', dict(integrated=metrics,
            binary=binary_summary(scores, raw, protocol['training_weight_mean']),
            checkpoint=lock[arm], specialist_all_15000_forward_seconds=seconds,
            timing_boundary='specialist inference only; not including frozen R45A inference (reused predictions)'))
        del model
        torch.cuda.empty_cache()


def run(root, stage):
    protocol, train_raw, val_raw, baseline = cpu_prepare(root)
    if stage == 'prepare':
        return
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1 or '3090' not in torch.cuda.get_device_name(0):
        raise RuntimeError('R53 must be bound to the authorized single idle RTX3090')
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    if stage in ('all', 'train'):
        raw = dict(train=train_raw, val=val_raw)
        loaders = dict(train=build_loaders(train_raw, 'train', 'cuda'), val=build_loaders(val_raw, 'val', 'cuda'))
        geometry = fit_geometry(loaders['train'], root)
        plan, schedule = make_plan(train_raw, root)
        set_seed(2)
        template = PairSpecialist(protocol['model_params']).cuda()
        with torch.no_grad():
            template.instance_encoder.geometry_residual.mean.copy_(torch.tensor(geometry['mean'], device='cuda'))
            template.instance_encoder.geometry_residual.std.copy_(torch.tensor(geometry['std'], device='cuda'))
        initial = {k:v.detach().cpu().clone() for k,v in template.state_dict().items()}
        real_precheck(template, loaders['train'], root)
        torch.save(initial, root/'shared_initialization.pt')
        dump(root/'runtime.json', dict(gpu=torch.cuda.get_device_name(0), visible_gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),
            torch=torch.__version__, slurm_job=os.environ.get('SLURM_JOB_ID'), sampling=schedule,
            shared_initialization_sha256=state_hash(initial)))
        del template
        for arm in ('A', 'B'):
            train_arm(arm, root, protocol, raw, loaders, baseline, geometry, plan, initial)
        del loaders
        torch.cuda.empty_cache()
    if stage in ('all', 'evaluate'):
        locked_test(root, protocol)
        from .r53_analysis import analyze
        analyze(root)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--stage', choices=('prepare', 'train', 'evaluate', 'all'), default='all')
    args = parser.parse_args()
    run(args.root, args.stage)


if __name__ == '__main__':
    main()

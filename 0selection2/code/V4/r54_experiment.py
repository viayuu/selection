"""R54: learn the two ReLD solution structures while training the pair specialist."""

import argparse
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from .multitask_probe import dump, file_hash, gather_batch, state_hash
from .pair_specialist import PairSpecialist, binary_objective
from .r42_experiment import CostController
from .r53_data import (MVRP, baseline_metrics, build_loaders, fit_geometry, gate_budget,
                      make_plan, raw_training, read_baseline)
from .r53_experiment import binary_summary, integrated_evaluation, predict
from .r54_model import RouteSupervisedSpecialist, successor_objective
from .r54_routes import PAIR, ROOT
from .train import set_seed
from .training_monitor import capture_rng, restore_rng


ARM = 'route_supervised_seed2'
SOURCES = ('r54_model.py', 'r54_routes.py', 'r54_experiment.py', 'r54_analysis.py', 'test_r54.py', 'run_v4_r54.sh')


def prepare(root):
    root.mkdir(parents=True, exist_ok=True)
    train, mean_weight = raw_training(root)
    val, baseline = read_baseline('val', root)
    budget = gate_budget(val, baseline, root)
    old = json.loads(Path('code/V4/runs/R53_pair_specialist/protocol.json').read_text())
    protocol = {key: old[key] for key in ('seed', 'frozen_baseline', 'model_params', 'gate', 'score_rule', 'ranking',
        'controller', 'batch', 'maximum_epochs', 'warmup_epochs', 'learning_rate', 'weight_decay', 'dropout',
        'optimizer', 'gradient_clip', 'accumulation', 'sampling', 'precision', 'success_screen', 'wandb')}
    protocol.update(experiment='R54', arm=ARM, training_weight_mean=mean_weight,
        losses='unweighted BCE + 0.5*(MOEL normalized successor CE + MTL normalized successor CE)',
        successor_normalization='per instance/method: mean valid-customer CE / max(log(Ncustomers), 1); N-1 alternatives plus END',
        target='next customer ID or END, from the native best POMO route; never an input feature',
        training_population='same non-tied 15-task training queries and batch plan as R53A; all 150000 train routes stored',
        historical_reference='R53A reused, not retrained; no strictly matched same-run ablation',
        structural_validation='fixed 64 original validation instances/task, chosen by size before any R54 result; only diagnostic',
        checkpoint_selection='strict minimum integrated full18 validation actual_regret_pct; auxiliary metrics never select',
        inference='R53A classification path only; successor heads skipped; no solver execution or probing',
        reference='https://arxiv.org/html/1906.01227v2',
        reference_boundary='solution edge supervision idea only; not the paper GCN, undirected binary heatmap, beam search or weighted edge BCE',
        annotation_authorized=True, test_used_for_selection=False,
        source_hashes={name: file_hash(Path(__file__).parent / name) for name in SOURCES})
    path = root / 'protocol.json'
    if path.exists() and json.loads(path.read_text()) != protocol and (root / ARM / 'last.pt').exists():
        raise ValueError('R54 protocol changed; preserve existing run')
    dump(path, protocol)
    dump(root / 'baseline_val_metrics.json', baseline_metrics(val, baseline))
    return protocol, dict(train=train, val=val), baseline, budget


def attach_routes(loaders, raw, split, root):
    receipt = json.loads((root / 'route_receipt.json').read_text())
    records = {(item['problem'], item['split']): item for item in receipt['rows']}
    result = {}
    for p in MVRP:
        path = root / 'route_cache' / (p + '_' + split + '.npz')
        if file_hash(path) != records[(p, split)]['sha256']:
            raise ValueError('Route archive changed after replay receipt')
        with np.load(path, allow_pickle=False) as saved:
            if str(saved['data_hash']) != raw[p]['data_hash'] or str(saved['label_hash']) != raw[p]['label_hash']:
                raise ValueError('Auxiliary routes and selector inputs have different hashes')
            np.testing.assert_array_equal(saved['pool'], PAIR)
            selected = saved['selected_indices'].copy()
            if not saved['complete'][selected].all():
                raise ValueError('Missing selected route target')
            if split == 'train':
                np.testing.assert_array_equal(selected, np.arange(loaders[p].size))
            successor = torch.as_tensor(saved['successor'].copy(), dtype=torch.long,
                                        device=loaders[p].batch['node'].device)
        loaders[p].batch['route_successor'] = successor
        result[p] = torch.from_numpy(selected)
    return result


def train_update(model, optimizer, scaler, batch):
    before = capture_rng()
    for retry in range(20):
        restore_rng(before)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.float16):
            out = model(batch, return_aux=True)
        binary = binary_objective(out['logit'], batch['binary_label'], batch['cost_weight'], False, 1.)
        structure, stats = successor_objective(out['successor_logits'], batch['route_successor'], out['customer_mask'])
        loss = binary + structure
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite R54 objective')
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        if scaler.get_scale() >= scale:
            if not torch.isfinite(norm):
                raise RuntimeError('Unskipped nonfinite R54 gradient')
            return dict(loss=float(loss.detach()), binary_loss=float(binary.detach()),
                successor_loss=float(structure.detach()), grad_norm=float(norm), retries=retry,
                **{key: float(value.detach()) for key, value in stats.items()})
        print(f'[R54 AMP] retry unchanged batch, scaler {scale}->{scaler.get_scale()}', flush=True)
    raise RuntimeError('Persistent R54 AMP overflow')


@torch.no_grad()
def structure_eval(model, loaders, indices):
    model.eval()
    rows = {}
    for p in MVRP:
        accum = dict(moel_loss=0., mtl_loss=0., moel_accuracy=0., mtl_accuracy=0.)
        examples, customers = 0, 0
        for ids in indices[p].split(128):
            batch = gather_batch(loaders[p], ids)
            out = model(batch, return_aux=True)
            _, metrics = successor_objective(out['successor_logits'], batch['route_successor'], out['customer_mask'])
            count = int(metrics['customers'])
            for key in accum:
                accum[key] += float(metrics[key]) * (count if 'accuracy' in key else len(ids))
            examples += len(ids)
            customers += count
        rows[p] = {key: value / (customers if 'accuracy' in key else examples) for key, value in accum.items()}
        rows[p].update(n=examples, customers=customers)
    return dict(per_problem=rows, macro={key: float(np.mean([r[key] for r in rows.values()]))
                for key in ('moel_loss', 'mtl_loss', 'moel_accuracy', 'mtl_accuracy')})


@torch.no_grad()
def save_structure_predictions(model, loaders, indices, folder):
    model.eval()
    folder.mkdir(parents=True, exist_ok=True)
    for p in MVRP:
        target, predictions, valid = [], [], []
        for ids in indices[p].split(128):
            batch = gather_batch(loaders[p], ids)
            out = model(batch, return_aux=True)
            count = out['successor_logits'].shape[-2]
            classes = out['successor_logits'].argmax(-1)
            original_ids = torch.where(classes == count, 0, classes + 1)
            original_ids = original_ids.masked_fill(~out['customer_mask'][:, None], -100)
            width = loaders[p].batch['route_successor'].shape[-1]
            predictions.append(torch.nn.functional.pad(original_ids, (0, width - count), value=-100).cpu())
            target.append(batch['route_successor'].cpu())
            valid.append(torch.nn.functional.pad(out['customer_mask'], (0, width - count), value=False).cpu())
        np.savez_compressed(folder / (p + '.npz'), indices=indices[p].numpy(),
            predicted_successor=torch.cat(predictions).numpy(), target_successor=torch.cat(target).numpy(),
            customer_mask=torch.cat(valid).numpy(), solver_names=np.asarray(PAIR))


def precheck(model, loaders, root):
    model.eval()
    batch = gather_batch(loaders['OVRPTW'], torch.arange(4))
    with torch.no_grad():
        original = model(batch, return_aux=True)
        legacy = PairSpecialist(model.params).to(batch['node'].device).eval()
        legacy.load_state_dict({k: v for k, v in model.state_dict().items() if not k.startswith('successor_heads.')})
        torch.testing.assert_close(original['logit'], legacy(batch), rtol=0, atol=0)
        changed = dict(batch, costs=batch['costs'].flip(-1), ind=batch['ind'] * 0,
            binary_label=batch['binary_label'] * 0, route_successor=batch['route_successor'].flip(1))
        second = model(changed, return_aux=True)
        torch.testing.assert_close(original['logit'], second['logit'], rtol=0, atol=0)
        torch.testing.assert_close(original['successor_logits'], second['successor_logits'], rtol=0, atol=0)
        torch.testing.assert_close(original['logit'], model(batch), rtol=0, atol=0)
    aux, _ = successor_objective(original['successor_logits'], batch['route_successor'], original['customer_mask'])
    # Recompute with a graph: auxiliary supervision alone must reach the encoder.
    out = model(batch, return_aux=True)
    loss, _ = successor_objective(out['successor_logits'], batch['route_successor'], out['customer_mask'])
    loss.backward()
    grad = lambda part: float(sum(p.grad.float().square().sum() for p in part.parameters() if p.grad is not None).sqrt())
    result = dict(label_isolation=True, inference_auxiliary_bypass=True, original_R53A_classification_path_identical=True,
        auxiliary_encoder_grad_norm=grad(model.instance_encoder),
        auxiliary_head_grad_norms=[grad(head) for head in model.successor_heads], initial_aux_loss=float(aux),
        total_parameters=sum(p.numel() for p in model.parameters()),
        inference_parameters=sum(p.numel() for n, p in model.named_parameters() if not n.startswith('successor_heads.')))
    if not result['auxiliary_encoder_grad_norm'] > 0 or not all(x > 0 for x in result['auxiliary_head_grad_norms']):
        raise ValueError('Route-supervision gradients are not connected')
    model.zero_grad(set_to_none=True)
    dump(root / 'model_precheck.json', result)


def train(root, protocol, raw, baseline):
    directory = root / ARM
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / 'result.json').exists():
        return
    loaders = {split: build_loaders(raw[split], split, 'cuda') for split in ('train', 'val')}
    route_indices = {split: attach_routes(loaders[split], raw[split], split, root) for split in ('train', 'val')}
    geometry = fit_geometry(loaders['train'], root)
    plan, schedule = make_plan(raw['train'], root)
    previous_schedule = json.loads(Path('code/V4/runs/R53_pair_specialist/sampling_plan.json').read_text())
    if schedule['epochs_hashes'] != previous_schedule['epochs_hashes']:
        raise ValueError('R54 and R53A sample plans differ')
    set_seed(2)
    model = RouteSupervisedSpecialist(protocol['model_params']).cuda()
    with torch.no_grad():
        model.instance_encoder.geometry_residual.mean.copy_(torch.tensor(geometry['mean'], device='cuda'))
        model.instance_encoder.geometry_residual.std.copy_(torch.tensor(geometry['std'], device='cuda'))
    common = {k: v.detach().cpu() for k, v in model.state_dict().items() if not k.startswith('successor_heads.')}
    previous = torch.load('code/V4/runs/R53_pair_specialist/shared_initialization.pt', map_location='cpu', weights_only=False)
    if state_hash(common) != state_hash(previous):
        raise ValueError('Common R53A/R54 initialization is not identical')
    initial_hash = state_hash(model.state_dict())
    precheck(model, loaders['train'], root)
    torch.save(model.state_dict(), root / 'initialization.pt')
    args = dict(protocol, geometry=geometry, sampling=schedule, common_initialization_matches_R53A=True,
        initialization_sha256=initial_hash, route_receipt_sha256=file_hash(root / 'route_receipt.json'))
    dump(directory / 'config.json', args)
    optimizer = torch.optim.AdamW(model.parameters(), lr=protocol['learning_rate'], weight_decay=protocol['weight_decay'])
    scaler = torch.amp.GradScaler('cuda', init_scale=1024.)
    c = protocol['controller']
    controller = CostController(min_delta=c['min_delta'], min_epochs=c['min_epochs'],
                                patience=c['early_stop_patience'], lr_patience=c['lr_patience'])
    history, step, start_epoch, best_cost, peak = [], 0, 0, float('inf'), protocol['learning_rate']
    set_seed(2)
    if (directory / 'last.pt').exists():
        old = torch.load(directory / 'last.pt', map_location='cpu', weights_only=False)
        if old['config'] != args:
            raise ValueError('Cannot resume different R54 data/configuration')
        model.load_state_dict(old['model'])
        optimizer.load_state_dict(old['optimizer'])
        scaler.load_state_dict(old['scaler'])
        controller.__dict__.update(old['controller'])
        history, step, start_epoch, best_cost, peak = old['history'], old['step'], old['epoch'], old['best_cost'], old['peak']
        restore_rng(old['rng'])
    else:
        dump(directory / 'initial_structure_val.json', structure_eval(model, loaders['val'], route_indices['val']))
    import wandb
    run = wandb.init(project=protocol['wandb']['project'], entity=protocol['wandb']['entity'],
        mode='offline', name='R54_route_supervision_seed2', dir=str(directory.resolve()), config=args)
    start = time.monotonic()
    warmup = 3 * len(plan['orders'][0])
    for epoch in range(start_epoch, protocol['maximum_epochs']):
        model.train()
        offsets, seen = {p: 0 for p in MVRP}, {p: 0 for p in MVRP}
        sums, examples, begin = {}, 0, time.monotonic()
        for p in plan['orders'][epoch]:
            positions = plan['permutations'][epoch][p][offsets[p]:offsets[p] + 128]
            ids = plan['valid_indices'][p][positions]
            lr = protocol['learning_rate'] * (step + 1) / warmup if step < warmup else peak
            for group in optimizer.param_groups:
                group['lr'] = lr
            stats = train_update(model, optimizer, scaler, gather_batch(loaders['train'][p], ids))
            for key, value in stats.items():
                if key != 'customers':
                    sums[key] = sums.get(key, 0.) + value * len(ids)
            examples += len(ids)
            offsets[p] += len(ids)
            seen[p] += len(ids)
            step += 1
            if step % 200 == 0:
                print(f'[R54] epoch={epoch+1} update={step} loss={stats["loss"]:.4f} BCE={stats["binary_loss"]:.4f} succ={stats["successor_loss"]:.4f}', flush=True)
        if any(seen[p] != len(plan['valid_indices'][p]) for p in MVRP):
            raise ValueError('Incomplete training epoch')
        scores = predict(model, loaders['val'])
        val = integrated_evaluation(scores, raw['val'], baseline)
        binary = binary_summary(scores, raw['val'], protocol['training_weight_mean'])
        structure_val = structure_eval(model, loaders['val'], route_indices['val'])
        cost = val['groups']['ALL']['actual_regret_pct']
        strict = cost < best_cost
        if strict:
            best_cost = cost
        halve, stop = controller.observe(cost, epoch + 1)
        if halve:
            peak = max(c['minimum_lr'], peak * .5)
        binary_train = structure_train = None
        if (epoch + 1) % 5 == 0 or stop or epoch + 1 == protocol['maximum_epochs']:
            binary_train = binary_summary(predict(model, loaders['train']), raw['train'], protocol['training_weight_mean'])
            structure_train = structure_eval(model, loaders['train'], plan['valid_indices'])
        row = dict(epoch=epoch + 1, successful_updates=step, **{key: value / examples for key, value in sums.items()},
            learning_rate=lr, next_peak_lr=peak, epoch_seconds=time.monotonic() - begin, training_examples=examples,
            per_problem_seen=seen, val=val, binary_val=binary, binary_train=binary_train,
            structure_val=structure_val, structure_train=structure_train, new_strict_best=strict, early_stop=stop)
        history.append(row)
        checkpoint = dict(model={k: v.detach().cpu() for k, v in model.state_dict().items()},
            optimizer=optimizer.state_dict(), scaler=scaler.state_dict(), rng=capture_rng(), controller=dict(controller.__dict__),
            epoch=epoch + 1, step=step, history=history, best_cost=best_cost, peak=peak, config=args)
        if strict:
            torch.save(checkpoint, directory / 'best.pt')
            dump(directory / 'best_eval.json', row)
        torch.save(checkpoint, directory / 'last.pt')
        dump(directory / 'history.json', history)
        run.log(dict(epoch=epoch + 1, successful_updates=step, train_loss=row['loss'], train_binary_loss=row['binary_loss'],
            train_successor_loss=row['successor_loss'], val_actual_regret_pct=cost, val_top1=val['groups']['ALL']['top1'],
            val_binary_ce=binary['macro']['ce'], val_binary_accuracy=binary['macro']['accuracy'], learning_rate=lr,
            val_moel_successor_accuracy=structure_val['macro']['moel_accuracy'],
            val_mtl_successor_accuracy=structure_val['macro']['mtl_accuracy']))
        print(f'[R54] epoch={epoch+1} Top1={val["groups"]["ALL"]["top1"]*100:.4f}% regret={cost:.6f}% '
            f'BCE={binary["macro"]["ce"]:.5f} succ_acc={structure_val["macro"]["moel_accuracy"]:.3f}/'
            f'{structure_val["macro"]["mtl_accuracy"]:.3f} seconds={row["epoch_seconds"]:.1f} stop={stop}', flush=True)
        if stop:
            break
    checkpoint = torch.load(directory / 'best.pt', map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model'])
    selected = predict(model, loaders['val'])
    best = integrated_evaluation(selected, raw['val'], baseline, directory / 'predictions' / 'val')
    np.testing.assert_allclose(best['groups']['ALL']['actual_regret_pct'], checkpoint['best_cost'], rtol=0, atol=1e-10)
    result = dict(best_epoch=checkpoint['epoch'], final_epoch=history[-1]['epoch'], successful_updates=step,
        best_validation=best, best_binary_val=binary_summary(selected, raw['val'], protocol['training_weight_mean']),
        best_binary_train=binary_summary(predict(model, loaders['train']), raw['train'], protocol['training_weight_mean']),
        best_structure_val=structure_eval(model, loaders['val'], route_indices['val']),
        best_structure_train=structure_eval(model, loaders['train'], plan['valid_indices']), final=history[-1],
        last5_val={k: float(np.mean([r['val']['groups']['ALL'][k] for r in history[-5:]])) for k in best['groups']['ALL']},
        elapsed_seconds=time.monotonic() - start, best_checkpoint_sha256=file_hash(directory / 'best.pt'))
    save_structure_predictions(model, loaders['val'], route_indices['val'], directory / 'structure_predictions' / 'val')
    dump(directory / 'result.json', result)
    run.summary.update(dict(best_epoch=checkpoint['epoch'], best_validation_actual_regret_pct=checkpoint['best_cost']))
    run.finish()
    del model, loaders, optimizer
    torch.cuda.empty_cache()


def locked_test(root, protocol):
    directory = root / ARM
    result = json.loads((directory / 'result.json').read_text())
    lock = dict(checkpoint=str((directory / 'best.pt').resolve()), sha256=file_hash(directory / 'best.pt'),
        epoch=result['best_epoch'], validation_actual_regret=result['best_validation']['groups']['ALL']['actual_regret_pct'])
    if lock['sha256'] != result['best_checkpoint_sha256']:
        raise ValueError('Checkpoint changed after validation selection')
    path = root / 'locked_checkpoint.json'
    if path.exists() and json.loads(path.read_text()) != lock:
        raise ValueError('Cannot reselect checkpoint after test')
    dump(path, lock)
    if (directory / 'test_result.json').exists():
        return
    raw, baseline = read_baseline('test', root)
    loaders = build_loaders(raw, 'test', 'cuda')
    model = RouteSupervisedSpecialist(protocol['model_params']).cuda()
    model.load_state_dict(torch.load(directory / 'best.pt', map_location='cpu', weights_only=False)['model'])
    torch.cuda.synchronize()
    begin = time.monotonic()
    scores = predict(model, loaders)
    torch.cuda.synchronize()
    seconds = time.monotonic() - begin
    dump(directory / 'test_result.json', dict(integrated=integrated_evaluation(scores, raw, baseline, directory / 'predictions' / 'test'),
        binary=binary_summary(scores, raw, protocol['training_weight_mean']), checkpoint=lock,
        specialist_15000_inference_seconds=seconds, auxiliary_heads_executed=False,
        timing_boundary='classification expert only, including its encoder; frozen R45A ranking reused, no solver/probing'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--stage', choices=('prepare', 'train', 'evaluate', 'all'), default='all')
    args = parser.parse_args()
    protocol, raw, baseline, _ = prepare(args.root)
    if args.stage == 'prepare':
        return
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1 or '3090' not in torch.cuda.get_device_name(0):
        raise RuntimeError('R54 only uses the authorized idle RTX3090')
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dump(args.root / 'runtime.json', dict(device=torch.cuda.get_device_name(0), visible_gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),
        slurm_job=os.environ.get('SLURM_JOB_ID'), torch=torch.__version__))
    if args.stage in ('train', 'all'):
        train(args.root, protocol, raw, baseline)
    if args.stage in ('evaluate', 'all'):
        locked_test(args.root, protocol)
        from .r54_analysis import analyze
        analyze(args.root)


if __name__ == '__main__':
    main()

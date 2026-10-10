"""Train one scratch R55 bottleneck; select by integrated validation regret."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import torch

from code.unified_selector.registry import PROBLEMS
from .multitask_probe import dump, file_hash, gather_batch, state_hash
from .pair_specialist import PAIR, PairSpecialist, apply_specialist, ordinal_scores, pair_targets
from .performance_evaluation import decision_metrics
from .r42_experiment import CostController
from .r53_data import (MVRP, baseline_metrics, build_loaders, fit_geometry, gate_budget,
                      grouped, make_plan, raw_training, read_baseline)
from .r53_experiment import predict
from .r55_model import CostDifferenceSpecialist, analytic_cost_difference, difference_objective
from .r55_targets import ROOT, ROUTES, attach_targets, prepare as prepare_targets
from .train import set_seed
from .training_monitor import capture_rng, restore_rng


ARM = 'cost_difference_seed2'
SOURCES = ('r55_targets.py', 'r55_model.py', 'r55_experiment.py', 'r55_analysis.py', 'test_r55.py', 'run_v4_r55.sh')


def prepare(root):
    root.mkdir(parents=True, exist_ok=True)
    targets = prepare_targets(root)
    train, _ = raw_training(root)
    val, baseline = read_baseline('val', root)
    gate_budget(val, baseline, root)
    previous = json.loads(Path('code/V4/runs/R54_route_supervision/protocol.json').read_text())
    keys = ('seed', 'frozen_baseline', 'model_params', 'gate', 'ranking', 'controller', 'batch', 'maximum_epochs',
            'warmup_epochs', 'learning_rate', 'weight_decay', 'dropout', 'optimizer', 'gradient_clip',
            'accumulation', 'sampling', 'precision', 'success_screen', 'wandb')
    protocol = {k: previous[k] for k in keys}
    protocol.update(experiment='R55', arm=ARM, git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        score_rule='delta_hat=sum(valid undirected edge D_hat * input Euclidean distance); positive/zero -> MOEL',
        architecture=dict(instance_encoder='unchanged R53/R54 4x128, node_only',
            node_projection='LayerNorm(128), Linear(128,32), GELU',
            edge_features='sum/abs-difference/product of projected endpoints, raw/mean-normalized distance, explicit constraints and log(1+n)/6',
            condition_projection='Linear(K_CBITS+K_PROBLEM_DESC+1,16), GELU',
            edge_mlp='Linear(114,64), GELU, Dropout(0.1), Linear(64,1)',
            output_initialization='nonzero Normal(std=0.001) weight, zero bias; avoid a random quadratic-in-N cost offset',
            readout='fixed FP32 distance-weighted sum; no pooling classifier, post-sum calibration, solver ID or bypass'),
        losses='group-balanced difference-edge MSE + relative cost-difference MSE; weights 1 and 1',
        edge_normalization='positive/negative/zero groups each averaged within each instance; equal mean over nonempty groups, then batch mean',
        cost_normalization='((pred_delta-(rawFP64_MTL-rawFP64_MOEL))/(0.01*rawFP64_full_pool_min))^2, batch mean',
        target_receipt_sha256=file_hash(root / 'target_receipt.json'),
        training_population='same 149976 non-tied queries and shuffled interleaved plan as R53/R54; 24 exact ties excluded for continuity',
        structure_validation='unchanged 64 validation instances/task, diagnostics only; no route inputs or validation fitting',
        checkpoint_selection='strict minimum integrated full18 validation actual_regret_pct',
        classification_metrics='sign accuracy/regret only; raw cost differences are not calibrated BCE logits',
        inference='instance encoder plus all-valid-edge decoder and fixed sum; no solver/probing',
        references=['https://proceedings.mlr.press/v119/koh20a.html'],
        reference_boundary='supervised intermediate bottleneck must determine the output; signed billed counts and analytic distance readout are our adaptation',
        solver_runs=0, new_annotations=0, historical_costs_modified=False, test_used_for_selection=False,
        historical_references=['R45A', 'R53A', 'R54; not retrained or strictly same-run ablations'],
        source_hashes={name: file_hash(Path(__file__).parent / name) for name in SOURCES},
        dependency_hashes={name: file_hash(Path(__file__).parent / name) for name in
                          ('V4Model.py', 'r53_data.py', 'pair_specialist.py', 'performance_evaluation.py')})
    path = root / 'protocol.json'
    if path.exists():
        protocol['git_commit'] = json.loads(path.read_text())['git_commit']
    if path.exists() and json.loads(path.read_text()) != protocol and (root / ARM / 'last.pt').exists():
        raise ValueError('R55 protocol changed; do not overwrite the trained experiment')
    dump(path, protocol)
    dump(root / 'baseline_val_metrics.json', baseline_metrics(val, baseline))
    return protocol, dict(train=train, val=val), baseline


def pair_summary(scores, raw):
    per_problem = {}
    for p in MVRP:
        r, z = raw[p], np.asarray(scores[p], dtype=np.float64)
        y, w = pair_targets(r['costs'], r['pool'])
        valid = y >= 0
        a, b = [r['pool'].index(name) for name in PAIR]
        delta = r['costs'][:, b] - r['costs'][:, a]
        chosen = np.where(z >= 0, a, b)
        selected = r['costs'][np.arange(len(z)), chosen]
        minimum = r['costs'][:, [a, b]].min(1)
        oracle = r['costs'].min(1)
        per_problem[p] = dict(n=int(valid.sum()), ties=int((~valid).sum()),
            accuracy=float(((z[valid] >= 0) == y[valid]).mean()),
            pair_regret_pct=float(((selected[valid] - minimum[valid]) / minimum[valid] * 100).mean()),
            extra_full_pool_regret_pct=float((w[valid] * (chosen[valid] != np.where(y[valid], a, b)) * 100).mean()),
            relative_cost_mse=float((((z[valid] - delta[valid]) / (.01 * oracle[valid])) ** 2).mean()),
            zero_delta_relative_mse=float(((delta[valid] / (.01 * oracle[valid])) ** 2).mean()),
            cost_difference_mae=float(np.abs(z[valid] - delta[valid]).mean()),
            moel_prediction_pct=float((z[valid] >= 0).mean() * 100),
            moel_win_pct=float(y[valid].mean() * 100))
    keys = [k for k in per_problem[MVRP[0]] if k not in ('n', 'ties')]
    macro = {k: float(np.mean([r[k] for r in per_problem.values()])) for k in keys}
    macro.update(n=sum(r['n'] for r in per_problem.values()), ties=sum(r['ties'] for r in per_problem.values()))
    return dict(per_problem=per_problem, macro=macro)


def evaluate_selection(scores, raw, baseline, directory=None):
    per_problem = {}
    if directory is not None:
        directory.mkdir(parents=True, exist_ok=True)
    for p in PROBLEMS:
        r, base = raw[p], baseline[p]
        z = scores[p] if p in MVRP else np.zeros(len(r['winner']), dtype=np.float32)
        order, gate = apply_specialist(base['order'], r['pool'], z)
        metrics, pred = decision_metrics(ordinal_scores(order), r)
        metrics['triggered'] = int(gate.sum())
        per_problem[p] = metrics
        np.testing.assert_array_equal(order[~gate], base['order'][~gate])
        np.testing.assert_array_equal(order[:, 2:], base['order'][:, 2:])
        if directory is not None:
            np.savez_compressed(directory / (p + '.npz'), indices=np.arange(len(pred)), pred=pred, order=order,
                baseline_order=base['order'], baseline_logits=base['logits'], baseline_pred=base['order'][:, 0],
                gate=gate, predicted_cost_difference=z, winner=r['winner'], costs=r['costs'],
                pool=np.asarray(r['pool']), pool_ids=np.asarray(r['pool_ids']))
    return dict(per_problem=per_problem, groups=grouped(per_problem))


def train_update(model, optimizer, scaler, batch):
    before = capture_rng()
    for retry in range(20):
        restore_rng(before)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.float16):
            out = model(batch, return_edges=True)
        loss, metrics = difference_objective(out, batch['edge_difference_target'],
                                             batch['cost_difference_target'], batch['oracle_cost_target'])
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite R55 objective')
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        if scaler.get_scale() >= scale:
            if not torch.isfinite(norm):
                raise RuntimeError('Unskipped nonfinite R55 gradient')
            return dict(loss=float(loss.detach()), grad_norm=float(norm), retries=retry,
                        **{k: float(v.detach()) for k, v in metrics.items()})
        print(f'[R55 AMP] retry unchanged batch, scaler {scale}->{scaler.get_scale()}', flush=True)
    raise RuntimeError('Persistent R55 AMP overflow')


@torch.no_grad()
def structure_eval(model, loaders, indices, folder=None):
    model.eval()
    per_problem = {}
    if folder is not None:
        folder.mkdir(parents=True, exist_ok=True)
    for p in MVRP:
        sums, count, arrays = {}, 0, {}
        for ids in indices[p].split(128):
            batch = gather_batch(loaders[p], ids)
            out = model(batch, return_edges=True)
            _, stats = difference_objective(out, batch['edge_difference_target'],
                                            batch['cost_difference_target'], batch['oracle_cost_target'])
            for key, value in stats.items():
                sums[key] = sums.get(key, 0.) + float(value) * len(ids)
            count += len(ids)
            if folder is not None:
                for name, tensor in dict(edge_difference=out['edge_difference'], edge_distance=out['edge_distance'],
                    edge_mask=out['edge_mask'], target=batch['edge_difference_target'][:, out['row'], out['col']],
                    cost_difference=out['cost_difference'], true_cost_difference=batch['cost_difference_target']).items():
                    arrays.setdefault(name, []).append(tensor.cpu().numpy())
        per_problem[p] = dict(n=count, **{k: v / count for k, v in sums.items()})
        if folder is not None:
            # Diagnostic subsets are one batch/task, so their edge grids are aligned.
            np.savez_compressed(folder / (p + '.npz'), indices=indices[p].numpy(), row=out['row'].cpu().numpy(),
                col=out['col'].cpu().numpy(), **{k: np.concatenate(v) for k, v in arrays.items()})
    keys = [k for k in per_problem[MVRP[0]] if k != 'n']
    return dict(per_problem=per_problem, macro={k: float(np.mean([r[k] for r in per_problem.values()])) for k in keys})


def precheck(model, loaders, root):
    model.eval()
    checks = []
    for p in ('OVRP', 'VRPB', 'OVRPBLTW'):
        batch = gather_batch(loaders[p], torch.arange(4))
        with torch.no_grad():
            first = model(batch, return_edges=True)
            changed = dict(batch, costs=batch['costs'].flip(-1), ind=batch['ind'] * 0,
                edge_difference_target=-batch['edge_difference_target'], cost_difference_target=-batch['cost_difference_target'],
                oracle_cost_target=batch['oracle_cost_target'] + 100, binary_label=batch['binary_label'] * 0)
            torch.testing.assert_close(model(changed), first['cost_difference'], rtol=0, atol=0)
            torch.testing.assert_close(analytic_cost_difference(first['edge_difference'], first['edge_distance'], first['edge_mask']),
                                       first['cost_difference'], rtol=0, atol=0)
            xy = batch['node'][..., :2].double()
            distance64 = torch.linalg.vector_norm(xy[:, first['row']] - xy[:, first['col']], dim=-1)
            truth = batch['edge_difference_target'][:, first['row'], first['col']].double()
            recovered = (truth * distance64 * first['edge_mask']).sum(-1)
            torch.testing.assert_close(recovered, batch['cost_difference_target'].double(), rtol=0, atol=2e-4)
        out = model(batch, return_edges=True)
        loss, stats = difference_objective(out, batch['edge_difference_target'], batch['cost_difference_target'], batch['oracle_cost_target'])
        loss.backward()
        grad = lambda module: float(sum(x.grad.float().square().sum() for x in module.parameters() if x.grad is not None).sqrt())
        encoder, decoder = grad(model.instance_encoder), grad(model.edge_decoder)
        if not np.isfinite([float(loss), encoder, decoder]).all() or min(encoder, decoder) <= 0:
            raise ValueError('Bottleneck gradients are disconnected/nonfinite')
        checks.append(dict(problem=p, label_isolation=True, mandatory_analytic_readout=True,
            selector_input_cost_identity_error=float((recovered - batch['cost_difference_target']).abs().max()),
            encoder_gradient_norm=encoder, edge_decoder_gradient_norm=decoder, loss=float(loss),
            **{k: float(v) for k, v in stats.items()}))
        model.zero_grad(set_to_none=True)
    dump(root / 'model_precheck.json', dict(checks=checks, parameters=sum(x.numel() for x in model.parameters()),
        graph_classification_head=False, solver_parameters=False, all_valid_edges=True))


def train(root, protocol, raw, baseline):
    directory = root / ARM
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / 'result.json').exists():
        return
    loaders = {s: build_loaders(raw[s], s, 'cuda') for s in ('train', 'val')}
    target_indices = {s: attach_targets(loaders[s], raw[s], s, root) for s in ('train', 'val')}
    geometry = fit_geometry(loaders['train'], root)
    plan, schedule = make_plan(raw['train'], root)
    old_schedule = json.loads((ROUTES / 'sampling_plan.json').read_text())
    if schedule['epochs_hashes'] != old_schedule['epochs_hashes']:
        raise ValueError('R55 sample order differs from the fixed R53/R54 plan')
    set_seed(2)
    model = CostDifferenceSpecialist(protocol['model_params']).cuda()
    with torch.no_grad():
        model.instance_encoder.geometry_residual.mean.copy_(torch.tensor(geometry['mean'], device='cuda'))
        model.instance_encoder.geometry_residual.std.copy_(torch.tensor(geometry['std'], device='cuda'))
    old_init = torch.load('code/V4/runs/R53_pair_specialist/shared_initialization.pt', map_location='cpu', weights_only=False)
    encoder_init = {k: v for k, v in old_init.items() if k.startswith('instance_encoder.')}
    current_init = {k: v for k, v in model.state_dict().items() if k.startswith('instance_encoder.')}
    if state_hash(current_init) != state_hash(encoder_init):
        raise ValueError('Fresh encoder initialization differs from R53/R54')
    args = dict(protocol, geometry=geometry, sampling=schedule, fresh_encoder_matches_R53A=True,
                initialization_sha256=state_hash(model.state_dict()))
    precheck(model, loaders['train'], root)
    torch.save(model.state_dict(), root / 'initialization.pt')
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
            raise ValueError('Cannot resume changed R55 configuration')
        model.load_state_dict(old['model'])
        optimizer.load_state_dict(old['optimizer'])
        scaler.load_state_dict(old['scaler'])
        controller.__dict__.update(old['controller'])
        history, step, start_epoch, best_cost, peak = old['history'], old['step'], old['epoch'], old['best_cost'], old['peak']
        restore_rng(old['rng'])
    else:
        dump(directory / 'initial_structure_val.json', structure_eval(model, loaders['val'], target_indices['val']))
    import wandb
    run = wandb.init(project=protocol['wandb']['project'], entity=protocol['wandb']['entity'], mode='offline',
        name='R55_route_cost_difference_seed2', dir=str(directory.resolve()), config=args)
    start, warmup = time.monotonic(), 3 * len(plan['orders'][0])
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
                sums[key] = sums.get(key, 0.) + value * len(ids)
            examples += len(ids)
            offsets[p] += len(ids)
            seen[p] += len(ids)
            step += 1
            if step % 200 == 0:
                print(f'[R55] epoch={epoch+1} update={step} loss={stats["loss"]:.4f} edge={stats["edge_loss"]:.4f} relative={stats["relative_cost_loss"]:.4f}', flush=True)
        if any(seen[p] != len(plan['valid_indices'][p]) for p in MVRP):
            raise ValueError('Incomplete R55 epoch')
        scores = predict(model, loaders['val'])
        val, pair = evaluate_selection(scores, raw['val'], baseline), pair_summary(scores, raw['val'])
        structure = structure_eval(model, loaders['val'], target_indices['val'])
        cost = val['groups']['ALL']['actual_regret_pct']
        strict = cost < best_cost
        if strict:
            best_cost = cost
        halve, stop = controller.observe(cost, epoch + 1)
        if halve:
            peak = max(c['minimum_lr'], peak * .5)
        train_pair = train_structure = None
        if (epoch + 1) % 5 == 0 or stop or epoch + 1 == protocol['maximum_epochs']:
            train_pair = pair_summary(predict(model, loaders['train']), raw['train'])
            train_structure = structure_eval(model, loaders['train'], plan['valid_indices'])
        row = dict(epoch=epoch + 1, successful_updates=step, **{k: v / examples for k, v in sums.items()},
            learning_rate=lr, next_peak_lr=peak, epoch_seconds=time.monotonic() - begin, training_examples=examples,
            per_problem_seen=seen, val=val, pair_val=pair, pair_train=train_pair,
            structure_val=structure, structure_train=train_structure, new_strict_best=strict, early_stop=stop)
        history.append(row)
        checkpoint = dict(model={k: v.detach().cpu() for k, v in model.state_dict().items()}, optimizer=optimizer.state_dict(),
            scaler=scaler.state_dict(), rng=capture_rng(), controller=dict(controller.__dict__), epoch=epoch + 1, step=step,
            history=history, best_cost=best_cost, peak=peak, config=args)
        if strict:
            torch.save(checkpoint, directory / 'best.pt')
            dump(directory / 'best_eval.json', row)
        torch.save(checkpoint, directory / 'last.pt')
        dump(directory / 'history.json', history)
        run.log(dict(epoch=epoch + 1, successful_updates=step, train_loss=row['loss'], train_edge_loss=row['edge_loss'],
            train_relative_cost_loss=row['relative_cost_loss'], val_actual_regret_pct=cost, val_top1=val['groups']['ALL']['top1'],
            val_pair_accuracy=pair['macro']['accuracy'], val_relative_cost_mse=pair['macro']['relative_cost_mse'],
            val_edge_loss=structure['macro']['edge_loss'], learning_rate=lr))
        print(f'[R55] epoch={epoch+1} Top1={val["groups"]["ALL"]["top1"]*100:.4f}% regret={cost:.6f}% '
            f'pair_acc={pair["macro"]["accuracy"]*100:.2f}% edge={structure["macro"]["edge_loss"]:.5f} '
            f'seconds={row["epoch_seconds"]:.1f} stop={stop}', flush=True)
        if stop:
            break
    checkpoint = torch.load(directory / 'best.pt', map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model'])
    selected = predict(model, loaders['val'])
    best = evaluate_selection(selected, raw['val'], baseline, directory / 'predictions' / 'val')
    np.testing.assert_allclose(best['groups']['ALL']['actual_regret_pct'], checkpoint['best_cost'], rtol=0, atol=1e-10)
    result = dict(best_epoch=checkpoint['epoch'], final_epoch=history[-1]['epoch'], successful_updates=step,
        best_validation=best, best_pair_val=pair_summary(selected, raw['val']),
        best_pair_train=pair_summary(predict(model, loaders['train']), raw['train']),
        best_structure_val=structure_eval(model, loaders['val'], target_indices['val'], directory / 'bottleneck_predictions' / 'val'),
        best_structure_train=structure_eval(model, loaders['train'], plan['valid_indices']), final=history[-1],
        last5_val={k: float(np.mean([r['val']['groups']['ALL'][k] for r in history[-5:]])) for k in best['groups']['ALL']},
        elapsed_seconds=time.monotonic() - start, best_checkpoint_sha256=file_hash(directory / 'best.pt'))
    dump(directory / 'result.json', result)
    run.summary.update(dict(best_epoch=checkpoint['epoch'], best_validation_actual_regret_pct=checkpoint['best_cost']))
    run.finish()
    del model, optimizer, loaders
    torch.cuda.empty_cache()


@torch.no_grad()
def inference_benchmark(model, loaders, protocol):
    batch = gather_batch(loaders['OVRPBLTW'], torch.arange(128))
    legacy = PairSpecialist(protocol['model_params']).cuda().eval()
    old = torch.load('code/V4/runs/R53_pair_specialist/A_ce_seed2/best.pt', map_location='cpu', weights_only=False)
    legacy.load_state_dict(old['model'])
    clean = {k: batch[k] for k in ('kind', 'problem_id', 'node', 'node_mask', 'matrix', 'n', 'cbits', 'problem_desc', 'coord_dist', 'node_geom') if k in batch}
    nodes, mask = model.instance_encoder(clean)
    def measure(call):
        for _ in range(5):
            call()
        torch.cuda.synchronize()
        begin = time.perf_counter()
        for _ in range(30):
            call()
        torch.cuda.synchronize()
        return (time.perf_counter() - begin) * 1000 / 30
    return dict(batch_size=128, maximum_nodes=int(batch['node'].shape[1]), fp32=True, warmup=5, repeats=30,
        r53a_encoder_and_binary_ms=measure(lambda: legacy(batch)),
        r55_encoder_and_edge_sum_ms=measure(lambda: model(batch)),
        edge_decoder_only_ms=measure(lambda: model.edge_decoder(nodes, mask, clean['node'][..., :2], clean['cbits'], clean['problem_desc'], clean['n'])),
        boundary='synchronized single idle GPU, complete expert forward; R45A shared ranking and CPU I/O excluded; edge-only timing is not end-to-end selection')


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
    model = CostDifferenceSpecialist(protocol['model_params']).cuda().eval()
    model.load_state_dict(torch.load(directory / 'best.pt', map_location='cpu', weights_only=False)['model'])
    torch.cuda.synchronize()
    begin = time.monotonic()
    scores = predict(model, loaders)
    torch.cuda.synchronize()
    seconds = time.monotonic() - begin
    dump(directory / 'test_result.json', dict(integrated=evaluate_selection(scores, raw, baseline, directory / 'predictions' / 'test'),
        pair=pair_summary(scores, raw), checkpoint=lock, specialist_15000_inference_seconds=seconds,
        no_solver_execution=True, graph_classification_bypass=False,
        benchmark=inference_benchmark(model, loaders, protocol)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--stage', choices=('prepare', 'train', 'evaluate', 'all'), default='all')
    args = parser.parse_args()
    protocol, raw, baseline = prepare(args.root)
    if args.stage == 'prepare':
        return
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1 or '3090' not in torch.cuda.get_device_name(0):
        raise RuntimeError('R55 uses only the authorized idle RTX3090')
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dump(args.root / 'runtime.json', dict(device=torch.cuda.get_device_name(0), visible_gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),
        slurm_job=os.environ.get('SLURM_JOB_ID'), torch=torch.__version__))
    if args.stage in ('train', 'all'):
        train(args.root, protocol, raw, baseline)
    if args.stage in ('evaluate', 'all'):
        locked_test(args.root, protocol)
        from .r55_analysis import analyze
        analyze(args.root)


if __name__ == '__main__':
    main()

"""One scratch EGT-inspired specialist, selected by full-system validation regret."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import torch

from .multitask_probe import dump, file_hash, gather_batch, state_hash
from .pair_specialist import PairSpecialist, binary_objective
from .r42_experiment import CostController
from .r53_data import (BASELINE, MVRP, baseline_metrics, build_loaders, fit_geometry,
                       gate_budget, make_plan, raw_training, read_baseline)
from .r53_experiment import binary_summary, integrated_evaluation, model_params, predict, update
from .r56_model import RelationalPairSpecialist
from .train import set_seed
from .training_monitor import capture_rng, restore_rng


ROOT = Path('code/V4/runs/R56_relational_encoder')
ARM = 'relational_seed2'
SOURCES = ('r56_model.py', 'r56_experiment.py', 'r56_analysis.py', 'test_r56.py', 'run_v4_r56.sh')
DEPS = ('pair_specialist.py', 'r53_data.py', 'r53_experiment.py', 'V4Model.py',
        'local_geometry.py', 'r42_experiment.py', 'tensor_loader.py', 'performance_targets.py')


def prepare(root):
    root.mkdir(parents=True, exist_ok=True)
    train_raw, mean_weight = raw_training(root)
    val_raw, baseline = read_baseline('val', root)
    budget = gate_budget(val_raw, baseline, root)
    reference = json.loads(Path('code/V4/runs/R53_pair_specialist/protocol.json').read_text())
    config = dict(experiment='R56', seed=2, arm=ARM, problems=list(MVRP),
        model_params=dict(model_params(), relation_dim=32),
        frozen_baseline=reference['frozen_baseline'],
        historical_reference='R53A; not a newly matched control',
        training_weight_mean=mean_weight, loss='ordinary unweighted BCEWithLogits only',
        labels='original FP64 MOEL<MTL -> 1, MTL<MOEL -> 0; exact ties excluded',
        gate=reference['gate'], score_rule=reference['score_rule'],
        checkpoint_selection='strict minimum integrated full18 validation actual_regret_pct',
        controller=reference['controller'], batch=128, maximum_epochs=40, warmup_epochs=3,
        learning_rate=1e-4, weight_decay=1e-4, dropout=.1, optimizer='fresh AdamW',
        gradient_clip=1., accumulation=False,
        sampling='same seed2 R53 plan; all non-ties, interleaved tasks, shuffled full coverage including tails',
        initialization='all trainable parameters freshly seeded; no trained selector or solver weights',
        architecture=dict(node_dim=128, relation_dim=32, heads=4, layers=4,
            input='original 8 node fields, 12 geometry fields, condition/stats/CLS tokens',
            directed_relation='separate source and target projections of original 8 fields plus real-node marker; '
                              'raw/mean-normalized input distance, cbits, problem_desc, log1p(n)/6',
            connectivity='all valid ordered pairs including self; no route/label/feasibility pruning',
            update='pre-norm EGT clipped QK + learned edge bias; sigmoid gate after softmax; '
                   'unmasked pre-softmax interactions update persistent edge residual + FFN',
            readout='original node mean/max, log1p(n)/6, d128 binary MLP',
            omitted=['dynamic centrality scaler', 'SVD positional encoding', 'random attention masking',
                     'edge prediction head', 'route/cost decomposition'],
            substitutions=['GELU FFNs and probability dropout0.1; no ELU or random -inf masking',
                'coordinate/constraint relations rather than shortest-path integer embeddings',
                'terminal edge-only update is diagnostic, with no BCE gradient; earlier edge updates '
                'and every layer edge bias/gate feed the node readout']),
        precision='FP16 AMP training, FP32 softmax/gates, FP32 eval, TF32 off; raw FP64 cost metrics',
        new_solver_runs=0, new_annotations=0, uses_saved_r54_routes=False,
        success_screen=reference['success_screen'],
        test_policy='lock one validation-best checkpoint, then one full test; never tune on test',
        reference='https://arxiv.org/html/2108.03348v3',
        reference_sections='EGT section3.2 equations3-7; EGT-inspired, not complete reproduction',
        git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        source_hashes={f:file_hash(Path(__file__).parent/f) for f in SOURCES},
        dependency_hashes={f:file_hash(Path(__file__).parent/f) for f in DEPS},
        wandb=reference['wandb'])
    for key in ('batch', 'maximum_epochs', 'warmup_epochs', 'learning_rate', 'weight_decay', 'dropout'):
        if config[key] != reference[key]:
            raise ValueError('R53 protocol changed: '+key)
    actual = baseline_metrics(val_raw, baseline)['groups']['ALL']
    for key, value in actual.items():
        np.testing.assert_allclose(value, reference['frozen_baseline']['val'][key], atol=1e-10, rtol=0)
    if (root/'protocol.json').exists() and (root/'initialization.pt').exists():
        old = json.loads((root/'protocol.json').read_text())
        for key in ('source_hashes', 'dependency_hashes', 'model_params', 'frozen_baseline'):
            if old[key] != config[key]:
                raise ValueError('Preserve existing R56 artifacts; changed '+key)
    dump(root/'protocol.json', config)
    print('[R56 gate] '+json.dumps(budget), flush=True)
    return config, train_raw, val_raw, baseline


def real_precheck(model, loaders, root):
    rows = []
    for problem in ('OVRP', 'VRPB', 'OVRPBLTW'):
        model.eval()
        sample = gather_batch(loaders[problem], torch.arange(4))
        with torch.no_grad():
            original, states = model(sample, return_states=True)
            changed = dict(sample, costs=sample['costs'].flip(-1), ind=sample['ind']*0,
                           binary_label=sample['binary_label']*0, cost_weight=sample['cost_weight']+10)
            torch.testing.assert_close(original, model(changed), rtol=0, atol=0)
            padded = dict(sample)
            for key in ('node', 'node_geom'):
                padded[key] = torch.cat((sample[key], torch.full_like(sample[key][:, :3], 777)), dim=1)
            padded['node_mask'] = torch.cat((sample['node_mask'], torch.zeros_like(sample['node_mask'][:, :3])), dim=1)
            padding_error = float((model(padded)-original).abs().max())
            if padding_error > 5e-5:
                raise ValueError('R56 padding changed graph logits')
            permutation = torch.arange(sample['node'].shape[1], device='cuda')
            permutation[1:] = permutation[1:].flip(0).clone()
            reordered = dict(sample)
            for key in ('node', 'node_mask', 'node_geom'):
                reordered[key] = sample[key][:, permutation]
            permutation_error = float((model(reordered)-original).abs().max())
            if permutation_error > 5e-5:
                raise ValueError('R56 node order changed invariant graph logits')
            initial_nodes, initial_edges, mask = model.instance_encoder.initial_states(sample)
            edge_changes, previous = [], initial_edges
            for _, edges in states['states']:
                edge_changes.append(float((edges-previous).abs().mean()))
                previous = edges
            asymmetry = float((initial_edges-initial_edges.transpose(1, 2)).abs().mean())
        indices = torch.where(loaders[problem].batch['binary_label'].cpu() >= 0)[0][:4]
        sample = gather_batch(loaders[problem], indices)
        model.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.float16):
            logits = model(sample)
        loss = binary_objective(logits, sample['binary_label'], sample['cost_weight'], False, 1.)
        loss.backward()
        def grad(module):
            gs = [p.grad.float() for p in module.parameters() if p.grad is not None]
            if not gs or not all(torch.isfinite(g).all() for g in gs):
                raise ValueError('Missing/nonfinite BCE gradient')
            return float(sum(g.square().sum() for g in gs).sqrt())
        gradients = dict(encoder=grad(model.instance_encoder), head=grad(model.head),
            source=grad(model.instance_encoder.source_projection), target=grad(model.instance_encoder.target_projection),
            relation_bias=[grad(layer.edge_bias) for layer in model.instance_encoder.layers],
            relation_gate=[grad(layer.edge_gate) for layer in model.instance_encoder.layers],
            persistent_update=[grad(layer.edge_out) for layer in model.instance_encoder.layers[:-1]])
        if not all(value > 0 for value in (gradients['encoder'], gradients['head'], gradients['source'], gradients['target'],
                                          *gradients['relation_bias'], *gradients['relation_gate'], *gradients['persistent_update'])):
            raise ValueError('Disconnected R56 relation path')
        rows.append(dict(problem=problem, label_isolation=True, padding_error=padding_error,
            permutation_error=permutation_error, initial_relation_asymmetry=asymmetry,
            per_layer_relation_change=edge_changes, gradients=gradients))
    model.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats()
    model.train()
    # Verify a real full-size optimizer update fits; restore the fresh parameters/RNG afterwards.
    before = {k:v.detach().clone() for k,v in model.state_dict().items()}
    rng = capture_rng()
    largest = max(loaders, key=lambda p:int(loaders[p].batch['n'].max()))
    valid = torch.where(loaders[largest].batch['binary_label'] >= 0)[0]
    ns = loaders[largest].batch['n'][valid]
    indices = valid[torch.argsort(ns, descending=True)[:128]].cpu()
    batch = gather_batch(loaders[largest], indices)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scaler = torch.amp.GradScaler('cuda', init_scale=1024.)
    stats = update(model, optimizer, scaler, batch, False, 1.)
    torch.cuda.synchronize()
    witness = dict(checks=rows, maximum_size_update=dict(problem=largest,
        batch=len(indices), n_max=int(batch['n'].max()), **stats),
        peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3,
        parameters=sum(p.numel() for p in model.parameters()),
        no_solver_parameters=not any('solver' in n for n, _ in model.named_parameters()),
        terminal_edge_output_not_in_node_only_bce=True)
    model.load_state_dict(before)
    model.zero_grad(set_to_none=True)
    restore_rng(rng)
    dump(root/'precheck.json', witness)
    print('[R56 precheck] '+json.dumps(witness), flush=True)


def train(root, protocol, raw, loaders, baseline, geometry, plan, initial):
    directory = root/ARM
    directory.mkdir(parents=True, exist_ok=True)
    if (directory/'result.json').exists():
        return
    set_seed(2)
    model = RelationalPairSpecialist(protocol['model_params']).cuda()
    model.load_state_dict(initial)
    initial_hash = state_hash(model.state_dict())
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scaler = torch.amp.GradScaler('cuda', init_scale=1024.)
    controller = CostController(min_delta=.001, min_epochs=15, lr_patience=3, patience=8)
    history, step, start_epoch, best_cost, peak = [], 0, 0, float('inf'), 1e-4
    config = dict(protocol, geometry=geometry, initialization_sha256=initial_hash,
                  trainable_parameters=sum(p.numel() for p in model.parameters()))
    dump(directory/'config.json', config)
    if (directory/'last.pt').exists():
        old = torch.load(directory/'last.pt', map_location='cpu', weights_only=False)
        if old['initialization_sha256'] != initial_hash or old['config']['source_hashes'] != config['source_hashes']:
            raise ValueError('R56 resume configuration changed')
        model.load_state_dict(old['model'])
        optimizer.load_state_dict(old['optimizer'])
        scaler.load_state_dict(old['scaler'])
        controller.__dict__.update(old['controller'])
        history, step, start_epoch, best_cost, peak = old['history'], old['step'], old['epoch'], old['best_cost'], old['peak']
        restore_rng(old['rng'])
    else:
        scores = predict(model, loaders['val'])
        dump(directory/'initial_eval.json', dict(binary=binary_summary(scores, raw['val'], protocol['training_weight_mean']),
            integrated=integrated_evaluation(scores, raw['val'], baseline)))
        set_seed(2)
    import wandb
    run = wandb.init(project=protocol['wandb']['project'], entity=protocol['wandb']['entity'],
                     name='R56_relational_seed2', dir=str(directory.resolve()), mode='offline', config=config)
    started = time.monotonic()
    warmup = 3*len(plan['orders'][0])
    for epoch in range(start_epoch, 40):
        model.train()
        human_epoch = epoch+1
        offsets, seen, counts = ({p:0 for p in MVRP} for _ in range(3))
        loss_total, examples, grad_total, retries = 0., 0, 0., 0
        epoch_start = time.monotonic()
        for p in plan['orders'][epoch]:
            positions = plan['permutations'][epoch][p][offsets[p]:offsets[p]+128]
            indices = plan['valid_indices'][p][positions]
            lr = 1e-4*(step+1)/warmup if step < warmup else peak
            for group in optimizer.param_groups:
                group['lr'] = lr
            stats = update(model, optimizer, scaler, gather_batch(loaders['train'][p], indices), False, 1.)
            offsets[p] += len(indices)
            seen[p] += len(indices)
            counts[p] += 1
            step += 1
            loss_total += stats['loss']*len(indices)
            examples += len(indices)
            grad_total += stats['grad_norm']
            retries += stats['retries']
            if step % 200 == 0:
                print(f'[R56] epoch={human_epoch} successful_update={step} lr={lr:.3g} BCE={stats["loss"]:.5f}', flush=True)
        if any(seen[p] != len(plan['valid_indices'][p]) for p in MVRP):
            raise ValueError('Incomplete R56 training coverage')
        scores = predict(model, loaders['val'])
        binary_val = binary_summary(scores, raw['val'], protocol['training_weight_mean'])
        val = integrated_evaluation(scores, raw['val'], baseline)
        cost = val['groups']['ALL']['actual_regret_pct']
        strict_best = cost < best_cost
        if strict_best:
            best_cost = cost
        halve, stop = controller.observe(cost, human_epoch)
        if halve:
            peak = max(1e-6, peak*.5)
        train_eval = None
        if human_epoch % 5 == 0 or stop or human_epoch == 40:
            train_eval = binary_summary(predict(model, loaders['train']), raw['train'], protocol['training_weight_mean'])
        row = dict(epoch=human_epoch, successful_updates=step, loss=loss_total/examples,
            grad_norm=grad_total/len(plan['orders'][epoch]), amp_retries=retries,
            learning_rate=lr, next_peak_lr=peak, training_examples=examples,
            per_problem_updates=counts, per_problem_seen=seen, epoch_seconds=time.monotonic()-epoch_start,
            peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3,
            val=val, binary_val=binary_val, binary_train=train_eval, new_strict_best=strict_best, early_stop=stop)
        history.append(row)
        checkpoint = dict(model={k:v.detach().cpu() for k,v in model.state_dict().items()},
            optimizer=optimizer.state_dict(), scaler=scaler.state_dict(), rng=capture_rng(),
            controller=dict(controller.__dict__), epoch=human_epoch, step=step, history=history,
            best_cost=best_cost, peak=peak, initialization_sha256=initial_hash, config=config)
        if strict_best:
            torch.save(checkpoint, directory/'best.pt')
            dump(directory/'best_eval.json', row)
        torch.save(checkpoint, directory/'last.pt')
        dump(directory/'history.json', history)
        run.log(dict(epoch=human_epoch, successful_updates=step, loss=row['loss'], learning_rate=lr,
            grad_norm=row['grad_norm'], val_actual_regret_pct=cost, val_top1=val['groups']['ALL']['top1'],
            specialist_val_ce=binary_val['macro']['ce'], specialist_val_accuracy=binary_val['macro']['accuracy'],
            gpu_peak_allocated_gib=row['peak_allocated_gib'], samples_per_second=examples/row['epoch_seconds']))
        print(f'[R56] epoch={human_epoch} val Top1={val["groups"]["ALL"]["top1"]*100:.4f}% '
              f'actual_regret={cost:.6f}% binary_acc={binary_val["macro"]["accuracy"]*100:.3f}% '
              f'best={best_cost:.6f}% time={row["epoch_seconds"]:.1f}s stop={stop}', flush=True)
        if stop:
            break
    best = torch.load(directory/'best.pt', map_location='cpu', weights_only=False)
    model.load_state_dict(best['model'])
    scores = predict(model, loaders['val'])
    best_val = integrated_evaluation(scores, raw['val'], baseline, directory/'predictions'/'val')
    np.testing.assert_allclose(best_val['groups']['ALL']['actual_regret_pct'], best['best_cost'], rtol=0, atol=1e-10)
    result = dict(best_epoch=best['epoch'], final_epoch=history[-1]['epoch'],
        best_validation=best_val, best_binary_val=binary_summary(scores, raw['val'], protocol['training_weight_mean']),
        best_binary_train=binary_summary(predict(model, loaders['train']), raw['train'], protocol['training_weight_mean']),
        final=history[-1], last5_val={k:float(np.mean([r['val']['groups']['ALL'][k] for r in history[-5:]]))
                                   for k in best_val['groups']['ALL']},
        elapsed_seconds=time.monotonic()-started, initialization_sha256=initial_hash,
        best_checkpoint_sha256=file_hash(directory/'best.pt'), successful_updates=step,
        source_hashes_at_end={f:file_hash(Path(__file__).parent/f) for f in SOURCES})
    if result['source_hashes_at_end'] != protocol['source_hashes']:
        raise ValueError('R56 sources changed during training')
    dump(directory/'result.json', result)
    run.summary.update(dict(best_epoch=best['epoch'], final_epoch=history[-1]['epoch'],
                            best_val_actual_regret_pct=best['best_cost']))
    run.finish()


def benchmark(model, loaders, root):
    sample = gather_batch(loaders['OVRPBLTW'], torch.arange(128))
    old = PairSpecialist(model_params()).cuda().eval()
    old.load_state_dict(torch.load('code/V4/runs/R53_pair_specialist/A_ce_seed2/best.pt',
                                  map_location='cpu', weights_only=False)['model'])
    timings = {}
    with torch.no_grad():
        for name, tested in (('R53A', old), ('R56', model.eval())):
            for _ in range(5):
                tested(sample)
            torch.cuda.synchronize()
            start = time.monotonic()
            for _ in range(30):
                tested(sample)
            torch.cuda.synchronize()
            timings[name] = (time.monotonic()-start)/30*1000
    dump(root/'inference_latency.json', dict(milliseconds=timings,
        relative_overhead_pct=(timings['R56']/timings['R53A']-1)*100,
        batch=128, max_nodes=sample['node'].shape[1], dtype='FP32', warmup=5, repeats=30,
        boundary='expert Encoder+head only; common frozen R45A ranking/data loading excluded; '
                 'not full selector or solver latency; one same-batch comparison', gpu=torch.cuda.get_device_name(0)))


def evaluate(root, protocol):
    directory = root/ARM
    result = json.loads((directory/'result.json').read_text())
    digest = file_hash(directory/'best.pt')
    if digest != result['best_checkpoint_sha256']:
        raise ValueError('R56 best checkpoint changed')
    lock = dict(checkpoint=str((directory/'best.pt').resolve()), sha256=digest,
                epoch=result['best_epoch'], validation_actual_regret=result['best_validation']['groups']['ALL']['actual_regret_pct'])
    path = root/'locked_checkpoint.json'
    if path.exists() and json.loads(path.read_text()) != lock:
        raise ValueError('Cannot select another checkpoint after test')
    dump(path, lock)
    if not (directory/'test_result.json').exists():
        raw, baseline = read_baseline('test', root)
        loaders = build_loaders(raw, 'test', 'cuda')
        model = RelationalPairSpecialist(protocol['model_params']).cuda()
        model.load_state_dict(torch.load(directory/'best.pt', map_location='cpu', weights_only=False)['model'])
        torch.cuda.synchronize()
        start = time.monotonic()
        scores = predict(model, loaders)
        torch.cuda.synchronize()
        seconds = time.monotonic()-start
        metrics = integrated_evaluation(scores, raw, baseline, directory/'predictions'/'test')
        dump(directory/'test_result.json', dict(integrated=metrics,
            binary=binary_summary(scores, raw, protocol['training_weight_mean']), checkpoint=lock,
            specialist_all_15000_forward_seconds=seconds,
            timing_boundary='all15 expert inference only; reused baseline inference and loading excluded'))
        benchmark(model, loaders, root)
    from .r56_analysis import analyze
    analyze(root)


def run(root, stage):
    protocol, train_raw, val_raw, baseline = prepare(root)
    if stage == 'prepare':
        return
    if (not torch.cuda.is_available() or torch.cuda.device_count() != 1 or
            '3090' not in torch.cuda.get_device_name(0) or
            os.environ.get('CUDA_VISIBLE_DEVICES') != 'GPU-cb626bdb-4319-2f5e-b91f-17fcaa44a83d'):
        raise RuntimeError('Bind only the authorized idle RTX3090 UUID')
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    if stage in ('all', 'train'):
        raw = dict(train=train_raw, val=val_raw)
        loaders = dict(train=build_loaders(train_raw, 'train', 'cuda'), val=build_loaders(val_raw, 'val', 'cuda'))
        geometry = fit_geometry(loaders['train'], root)
        plan, schedule = make_plan(train_raw, root)
        old_plan = json.loads(Path('code/V4/runs/R53_pair_specialist/sampling_plan.json').read_text())
        if schedule['epochs_hashes'] != old_plan['epochs_hashes']:
            raise ValueError('R56 sample/task plan differs from R53')
        set_seed(2)
        model = RelationalPairSpecialist(protocol['model_params']).cuda()
        with torch.no_grad():
            model.instance_encoder.geometry_residual.mean.copy_(torch.tensor(geometry['mean'], device='cuda'))
            model.instance_encoder.geometry_residual.std.copy_(torch.tensor(geometry['std'], device='cuda'))
        initial = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        real_precheck(model, loaders['train'], root)
        torch.save(initial, root/'initialization.pt')
        dump(root/'runtime.json', dict(gpu=torch.cuda.get_device_name(0), visible_gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),
            torch=torch.__version__, slurm_job=os.environ.get('SLURM_JOB_ID'), sampling=schedule,
            initialization_sha256=state_hash(initial), historical_sampling_plan_matched=True))
        del model
        train(root, protocol, raw, loaders, baseline, geometry, plan, initial)
        del loaders
        torch.cuda.empty_cache()
    if stage in ('all', 'evaluate'):
        evaluate(root, protocol)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--stage', choices=('prepare', 'train', 'evaluate', 'all'), default='all')
    args = parser.parse_args()
    run(args.root, args.stage)

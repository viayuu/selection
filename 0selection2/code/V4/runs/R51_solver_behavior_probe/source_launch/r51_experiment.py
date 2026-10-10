"""R51: static versus short native behavior evidence, locked R49 splits."""

import argparse
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .multitask_probe import dump, file_hash, state_hash
from .r48_common import PAIR, binary_metrics, write_csv
from .r49_experiment import configure_numeric, development_improved
from .r50_experiment import load_locked_data, load_cache as load_static_cache, take, save_predictions
from .r50_solver_encoder import source_provenance, native_inputs, raw_nodes, encoder_output
from .r51_probe import (BEHAVIOR_DIM, FEATURE_NAMES, FIELDS, STATS, STEPS,
                        build_solvers, cache_behaviors, probe, initialize, finish)
from .r51_readout import BehaviorReadout, common_state, copy_common_initialization
from .train import set_seed


ROOT = Path('code/V4/runs/R51_solver_behavior_probe')
R50 = Path('code/V4/runs/R50_pretrained_solver_encoder')
MODELS = ('A_static_seed2', 'B_behavior_seed2')
SETS = ('train', 'development', 'internal', 'original_val')


def prepare(root):
    root.mkdir(parents=True, exist_ok=True)
    if (root / 'config.json').exists():
        raise ValueError('R51 already prepared; refusing to overwrite the protocol')
    data, _, sets = load_locked_data()
    for name in ('split_manifest.json', 'sampling_plan.npz'):
        shutil.copy2(R50 / name, root / name)
    provenance = source_provenance()
    dump(root / 'solver_provenance.json', provenance)
    set_seed(2)
    static, enhanced = BehaviorReadout(), BehaviorReadout(True)
    copy_common_initialization(static, enhanced)
    if state_hash(common_state(static)) != state_hash(common_state(enhanced)):
        raise ValueError('A/B shared parameter initialization differs')
    torch.save(static.state_dict(), root / 'A_initial.pt')
    torch.save(enhanced.state_dict(), root / 'B_initial.pt')
    timing_rows = []
    rng = np.random.default_rng(5102)
    train = sets['train'][1]
    for size in (int(data['train']['sizes'][train].min()),
                 int(np.median(data['train']['sizes'][train])),
                 int(data['train']['sizes'][train].max())):
        candidates = train[data['train']['sizes'][train] == size]
        timing_rows.extend(rng.choice(candidates, size=4, replace=False).tolist())
    config = dict(seed=2, models=list(MODELS), problem='OVRPTW', pair=list(PAIR),
        counts={name: dict(total=len(rows), non_ties=int((data[source]['labels'][rows] >= 0).sum()))
                for name, (source, rows) in sets.items()},
        split='Locked R49/R50 8000 train + 1000 development + 1000 internal + original1000 validation',
        supervision='Original FP64 two-method costs; exact ties excluded; unweighted binary CE only',
        test_read=False, new_cost_labels=False,
        solver_frozen=True, solver_eval=True, solver_numeric='FP32, no AMP, TF32 disabled',
        static_cache_root=str(R50), static_cache_model='B_pretrained_seed2',
        static_cache_manifest_sha256=file_hash(R50 / 'cache_manifest.json'),
        static_readout='Exact R50 node readout; fresh seed2 parameters, not R50 trained readout',
        behavior_readout='Same node MLP/graph385; behavior270->64 GELU Dropout ->64 GELU; concatenate; head449->128->2',
        behavior_normalization='Per-field mean/population std from actual8000 training rows only; std floor1e-6',
        common_initial_sha256=state_hash(common_state(static)),
        parameter_counts={MODELS[0]: sum(p.numel() for p in static.parameters()),
                          MODELS[1]: sum(p.numel() for p in enhanced.parameters())},
        initial_hashes={MODELS[0]: state_hash(static.state_dict()), MODELS[1]: state_hash(enhanced.state_dict())},
        batch_size=128, successful_updates=4000, evaluation_interval=200,
        optimizer='fresh AdamW', learning_rate=1e-3, weight_decay=1e-4, dropout=.1,
        gradient_clip=1., amp=False, tf32=False, scheduler=None, early_stop=False,
        selection='Strict lowest development CE, u0/200/.../4000; no internal/val selection',
        sampling_plan_sha256=file_hash(root / 'sampling_plan.npz'),
        split_manifest_sha256=file_hash(root / 'split_manifest.json'),
        timing=dict(rows=timing_rows, source='Original train, members of8000 training subset',
            selection='Seed5102, four randomly chosen rows at minimum/median/maximum input customer size; no labels',
            batch_size=1, warmup=1, repetitions=3, statistic='Synchronized wall time including native collation and transfer',
            resume=True, overhead_screen_max_ratio_to_static=1.25,
            caveat='Engineering screen; GPU shared with unrelated workload on the other card; not a throughput benchmark'),
        screen='B-A >=5pp accuracy on BOTH internal and original_val, lower BOTH regrets, total pipeline latency <=1.25x A',
        runtime=dict(torch=torch.__version__, gpu=torch.cuda.get_device_name(0),
            gpu_uuid=os.environ.get('CUDA_VISIBLE_DEVICES'), slurm_job=os.environ.get('SLURM_JOB_ID')),
        paper='https://arxiv.org/html/2401.12745v1',
        wandb=dict(project='selector', entity='yjkds-southern-university-of-science-technology', mode='offline'),
        git_parent=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip())
    snapshot = root / 'source_launch'
    snapshot.mkdir()
    config['sources'] = {}
    for name in ('r51_probe.py', 'r51_readout.py', 'r51_experiment.py', 'r51_analysis.py',
                 'test_r51.py', 'run_v4_r51.sh'):
        source = Path(__file__).parent / name
        shutil.copy2(source, snapshot / name)
        config['sources'][name] = file_hash(source)
    config['dependencies'] = {name: file_hash(Path(__file__).parent / name) for name in
        ('r50_readout.py', 'r50_experiment.py', 'r50_solver_encoder.py', 'r48_common.py')}
    dump(root / 'config.json', config)
    dump(root / 'feature_definition.json', dict(dimension=BEHAVIOR_DIM, names=FEATURE_NAMES,
        steps=list(STEPS), fields=list(FIELDS), pomo_statistics=list(STATS),
        shape='[solver2, network_step3, field9, statistic5], flattened in this order',
        initialization='Forced depot followed by ALL customer POMO starts, then exactly10 original greedy network actions',
        progress_cost_alignment='Progress/pressure/cost measured after the action; entropy/probability gap from that action distribution before the move',
        progress='Served customer/demand proportions include the forced first customer',
        depot_returns='Count of network-selected depot returns; forced initialization is excluded',
        distance='Sum of traveled segments whose DESTINATION is a customer; no return-to-depot cost; includes depot-to-POMO-start',
        waiting='Sum max(tw_start - arrival,0) at customer visits; no depot placeholder deadline',
        distance_scale='Mean ordered off-diagonal Euclidean distance across original depot+customers, floor1e-8',
        time_scale='Input mean distance / original environment speed1',
        capacity='Native capacity1 remaining load, reset at depot',
        feasibility='Post-action native finite-mask customers / unserved customers; no depot in numerator/denominator',
        entropy='Legal-action Shannon entropy / log(number of legal actions); zero when only one action is legal',
        probability_gap='Highest minus second-highest action probability; native masked illegal actions have zero probability',
        population_std=True, quantile_interpolation='linear', labels_or_final_costs_used=False))


def make_caches(root):
    config = json.loads((root / 'config.json').read_text())
    data, _, sets = load_locked_data()
    provenance = source_provenance()
    models, module = build_solvers(provenance, 'cuda:0')
    cache_dir = root / 'feature_cache'
    cache_dir.mkdir(exist_ok=True)
    records = {}
    start = time.monotonic()
    for split, values in data.items():
        path = cache_dir / (split + '_behavior.pt')
        if path.exists():
            raise ValueError('Refusing to replace a behavior cache')
        def progress(count, total, size):
            print(f'[R51 probe cache] {split} {count}/{total} size={size}', flush=True)
        features, witnesses = cache_behaviors(models, module, values['instances'], values['sizes'], progress)
        torch.save(dict(behavior=features, data_hash=values['raw']['data_hash']), path)
        torch.save(witnesses, cache_dir / (split + '_witness.pt'))
        records[split] = dict(path=str(path), sha256=file_hash(path), shape=list(features.shape),
            data_hash=values['raw']['data_hash'], witness_path=str(cache_dir / (split + '_witness.pt')))
        if split == 'train':
            train_features = features[sets['train'][1]]
            mean = train_features.mean(0)
            std = train_features.std(0, correction=0).clamp_min(1e-6)
            torch.save(dict(mean=mean, std=std), root / 'behavior_normalization.pt')
            dump(root / 'behavior_normalization.json', dict(names=FEATURE_NAMES,
                mean=mean.tolist(), std=std.tolist(), source_rows=sets['train'][1].tolist(),
                fit_set='8000 training subset only; development/internal/original_val excluded'))
    records['solvers'] = {name: dict(full_state_sha256=state_hash(model.state_dict()),
        frozen=True, eval=True) for name, model in zip(PAIR, models)}
    records['seconds'] = time.monotonic() - start
    dump(root / 'behavior_cache_manifest.json', records)
    static = load_static_cache(R50, 'B_pretrained_seed2', 'cpu')
    checks = []
    batch, size = native_inputs(module, data['train']['instances'], [0], 'cuda:0')
    for number, (name, model) in enumerate(zip(PAIR, models)):
        features, prefix, witness = probe(model, module, batch, keep_witness=True)
        torch.testing.assert_close(model.encoded_nodes.cpu(), static['train']['encoded'][[0], :size + 1, number],
                                   rtol=2e-4, atol=5e-5)
        cache = torch.load(cache_dir / 'train_behavior.pt', weights_only=True)
        feature_error = float((cache['behavior'][0, number * 135:(number + 1) * 135] - features[0].cpu()).abs().max())
        torch.testing.assert_close(cache['behavior'][0, number * 135:(number + 1) * 135], features[0].cpu(), rtol=2e-4, atol=1e-5)
        saved_nodes = prefix.env.selected_node_list.clone()
        resumed, continuation_actions = finish(model, prefix)
        direct = initialize(model, module, batch)
        full, _ = finish(model, direct)
        torch.testing.assert_close(resumed, full, rtol=0, atol=0)
        if not torch.equal(prefix.env.selected_node_list, direct.env.selected_node_list):
            raise ValueError('Resumed construction differs from uninterrupted original decoding')
        historical = float(data['train']['costs'][0, number])
        checks.append(dict(solver=name, instance_index=0, cache_feature_max_error=feature_error,
            forced_plus_network_actions_before_resume=saved_nodes.shape[-1], network_actions=10,
            all_pomo_starts=size, continuation_actions=continuation_actions,
            resumed_equals_uninterrupted=True, full_cost=float(full[0]), historical_cost=historical,
            historical_absolute_error=abs(float(full[0]) - historical)))
    dump(root / 'preflight.json', dict(checks=checks, solvers_frozen=True,
        static_cache_reused=True, inputs_from_six_original_native_fields=True,
        no_full_route_used_for_feature_generation=True,
        full_solve_witness='Only original train index0, after feature caching; used for state-continuation verification, never as an input/label',
        cost_and_label_isolation=True, test_read=False))
    print('[R51] Cached static and ten-action behavior evidence; continuation witness passed', flush=True)


def load_caches(root, device='cuda:0'):
    config = json.loads((root / 'config.json').read_text())
    if file_hash(R50 / 'cache_manifest.json') != config['static_cache_manifest_sha256']:
        raise ValueError('R50 frozen cache metadata changed')
    caches = load_static_cache(R50, 'B_pretrained_seed2', device)
    manifest = json.loads((root / 'behavior_cache_manifest.json').read_text())
    for split, cache in caches.items():
        record = manifest[split]
        if file_hash(record['path']) != record['sha256']:
            raise ValueError('Behavior cache changed')
        payload = torch.load(record['path'], map_location='cpu', weights_only=True)
        if payload['data_hash'] != record['data_hash']:
            raise ValueError('Wrong behavior data source')
        cache['behavior'] = payload['behavior'].to(device)
    return caches


def inputs(cache, rows, behavior):
    tensors = take(cache, rows)
    if behavior:
        rows = torch.as_tensor(rows, device=cache['encoded'].device)
        tensors += (cache['behavior'][rows],)
    return tensors


def make_readout(root, name, initial=False):
    behavior = name == MODELS[1]
    normalization = torch.load(root / 'behavior_normalization.pt', weights_only=True)
    model = BehaviorReadout(behavior, **normalization)
    if initial:
        template = torch.load(root / ('B_initial.pt' if behavior else 'A_initial.pt'), weights_only=True)
        if behavior:
            template.update(behavior_mean=normalization['mean'], behavior_std=normalization['std'])
        model.load_state_dict(template, strict=True)
    return model


@torch.no_grad()
def evaluate(model, caches, data, sets):
    model.eval()
    metrics, scores = {}, {}
    for name, (source, rows) in sets.items():
        scores[name] = np.concatenate([model(*inputs(caches[source], indices, model.behavior_enabled)).cpu().numpy()
                                      for indices in torch.as_tensor(rows).split(128)])
        metrics[name] = binary_metrics(scores[name], data[source]['costs'][rows])
    return metrics, scores


def train_one(root, name):
    directory = root / name
    directory.mkdir(exist_ok=True)
    if (directory / 'last.pt').exists():
        raise ValueError('R51 training output already exists')
    config = json.loads((root / 'config.json').read_text())
    data, _, sets = load_locked_data()
    caches = load_caches(root)
    model = make_readout(root, name, initial=True).cuda()
    if state_hash(common_state(model)) != config['common_initial_sha256']:
        raise ValueError('Shared fresh initialization changed')
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    if file_hash(root / 'sampling_plan.npz') != config['sampling_plan_sha256']:
        raise ValueError('Locked batch schedule changed')
    plan = np.load(root / 'sampling_plan.npz')['indices']
    if plan.shape != (4000, 128) or not np.isin(plan, sets['train'][1]).all():
        raise ValueError('Unexpected sample budget')
    labels = torch.as_tensor(data['train']['labels'], device='cuda:0')
    if (labels[torch.as_tensor(plan, device='cuda:0')] < 0).any():
        raise ValueError('An exact cost tie entered CE training')
    args = dict(config, model=name, behavior_enabled=model.behavior_enabled,
                behavior_cache_sha256=file_hash(root / 'behavior_cache_manifest.json'))
    dump(directory / 'args.json', args)
    import wandb
    run = wandb.init(project='selector', entity='yjkds-southern-university-of-science-technology',
        name='R51_' + name, dir=str(directory), config=args, mode='offline')
    set_seed(2)
    history, best, loss_sum, grad_sum = [], float('inf'), 0., 0.
    start = time.monotonic()
    try:
        for updates in range(4001):
            if updates:
                model.train()
                rows = torch.as_tensor(plan[updates - 1], device='cuda:0')
                optimizer.zero_grad(set_to_none=True)
                loss = F.cross_entropy(model(*inputs(caches['train'], rows, model.behavior_enabled)), labels[rows])
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite readout CE')
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                if updates == 1:
                    blocks = dict(node_mlp=model.node_mlp, head=model.head,
                                  moel_norm=model.moel_norm, mtl_norm=model.mtl_norm)
                    if model.behavior_enabled:
                        blocks['behavior_mlp'] = model.behavior_mlp
                    witness = {key: float(sum(p.grad.abs().sum() for p in block.parameters() if p.grad is not None))
                               for key, block in blocks.items()}
                    if min(witness.values()) <= 0:
                        raise ValueError('A readout module received no gradient')
                    dump(directory / 'gradient_witness.json', witness)
                optimizer.step()
                loss_sum += float(loss.detach())
                grad_sum += float(norm)
            if updates % 200:
                continue
            metrics, scores = evaluate(model, caches, data, sets)
            improved = development_improved(metrics, best)
            if improved:
                best = metrics['development']['ce']
            record = dict(updates=updates, metrics=metrics, lr=1e-3,
                optimization_ce=loss_sum / 200 if updates else None,
                grad_norm=grad_sum / 200 if updates else None,
                seconds=time.monotonic() - start, selection_improved=improved)
            history.append(record)
            checkpoint = dict(model=model.state_dict(), updates=updates, metrics=metrics,
                              config=args, selection='development CE only')
            if improved:
                torch.save(checkpoint, directory / 'best.pt')
            torch.save(dict(checkpoint, optimizer=optimizer.state_dict()), directory / 'last.pt')
            dump(directory / 'history.json', history)
            summary = '; '.join(f'{key}:acc={v["accuracy"]:.4f} CE={v["ce"]:.5f} regret={v["pair_regret_pct"]:.4f}%'
                                for key, v in metrics.items())
            message = f'[R51 {name}] u={updates} {summary} dev_best={improved}'
            print(message, flush=True)
            with (directory / 'train.log').open('a') as stream:
                stream.write(message + '\n')
            log = dict(updates=updates, lr=1e-3, optimization_ce=record['optimization_ce'], grad_norm=record['grad_norm'])
            for key, values in metrics.items():
                log.update({key + '/' + field: values[field] for field in ('accuracy', 'ce', 'pair_regret_pct', 'mean_cost')})
            run.log(log, step=updates)
            loss_sum = grad_sum = 0.
        selected = torch.load(directory / 'best.pt', map_location='cuda:0', weights_only=False)
        model.load_state_dict(selected['model'], strict=True)
        metrics, scores = evaluate(model, caches, data, sets)
        for split, values in metrics.items():
            for field, value in values.items():
                if not np.isclose(value, selected['metrics'][split][field], atol=1e-12, rtol=0):
                    raise ValueError('Best checkpoint metric replay failed')
        save_predictions(directory, scores, sets, data)
        dump(directory / 'selection.json', dict(model=name, best_updates=selected['updates'],
            successful_updates=4000, presentations=512000,
            best_checkpoint_sha256=file_hash(directory / 'best.pt'),
            common_initial_sha256=config['common_initial_sha256'],
            sampling_plan_sha256=config['sampling_plan_sha256'],
            training_seconds=time.monotonic() - start, development_ce=metrics['development']['ce'],
            trainable_parameters=sum(p.numel() for p in model.parameters()),
            selection_uses_internal_or_original_val=False, frozen_solvers_not_in_optimizer=True))
        return metrics
    finally:
        run.finish()


@torch.no_grad()
def measure_timing(root):
    data, _, _ = load_locked_data()
    config = json.loads((root / 'config.json').read_text())
    models, module = build_solvers(source_provenance(), 'cuda:0')
    hashes = [state_hash(model.state_dict()) for model in models]
    readouts = {}
    for name in MODELS:
        model = make_readout(root, name).cuda().eval()
        model.load_state_dict(torch.load(root / name / 'best.pt', map_location='cuda:0', weights_only=False)['model'])
        readouts[name] = model

    def mark():
        torch.cuda.synchronize()
        return time.perf_counter()

    def pipeline(index, name):
        start = mark()
        batch, size = native_inputs(module, data['train']['instances'], [index], 'cuda:0')
        prefixes, evidence, nodes = [], [], []
        if name == MODELS[1]:
            for model in models:
                part, prefix, _ = probe(model, module, batch)
                evidence.append(part)
                prefixes.append(prefix)
                nodes.append(model.encoded_nodes)
        else:
            for model in models:
                nodes.append(encoder_output(model.encoder, batch))
        feature_time = mark()
        encoded = torch.stack(nodes, 2)
        mask = torch.ones(1, size + 1, dtype=torch.bool, device='cuda:0')
        kwargs = (torch.cat(evidence, -1),) if evidence else ()
        scores = readouts[name](encoded, raw_nodes(batch), mask, *kwargs)
        choice = int(scores.argmax(1)[0])
        selection_time = mark()
        if name == MODELS[1]:
            prefix = prefixes[choice]
            before = prefix.env.selected_count
        else:
            models[choice].encoded_nodes = nodes[choice]
            models[choice].decoder.set_kv(nodes[choice])
            prefix = initialize(models[choice], module, batch, encode=False)
            before = 0
        costs, continuation = finish(models[choice], prefix)
        end = mark()
        return dict(model=name, original_index=index, customers=size, chosen_solver=PAIR[choice],
            feature_ms=(feature_time - start) * 1000, selection_ms=(selection_time - feature_time) * 1000,
            completion_ms=(end - selection_time) * 1000, total_ms=(end - start) * 1000,
            prefix_actions_reused=before, continuation_actions=continuation,
            final_solver_cost=float(costs[0]), historical_cost=float(data['train']['costs'][index, choice]),
            historical_absolute_error=abs(float(costs[0]) - float(data['train']['costs'][index, choice])))

    for name in MODELS:
        pipeline(config['timing']['rows'][0], name)
    records = []
    for index in config['timing']['rows']:
        for repetition in range(config['timing']['repetitions']):
            # Alternate order to reduce first/second systematic timing bias.
            order = MODELS if repetition % 2 == 0 else MODELS[::-1]
            for name in order:
                record = pipeline(index, name)
                records.append(dict(repetition=repetition, **record))
        print(f'[R51 timing] completed native selected routes: instance {index}', flush=True)
    if hashes != [state_hash(model.state_dict()) for model in models]:
        raise ValueError('Frozen full solver state changed')
    write_csv(root / 'inference_timing.csv', records)
    summary = {}
    for name in MODELS:
        rows = [r for r in records if r['model'] == name]
        summary[name] = {field: dict(mean=float(np.mean([r[field] for r in rows])),
                                    median=float(np.median([r[field] for r in rows])),
                                    p95=float(np.quantile([r[field] for r in rows], .95)))
                         for field in ('feature_ms', 'selection_ms', 'completion_ms', 'total_ms')}
    ratio = summary[MODELS[1]]['total_ms']['mean'] / summary[MODELS[0]]['total_ms']['mean']
    dump(root / 'inference_timing.json', dict(summary=summary, ratio_behavior_to_static=ratio,
        overhead_acceptable=ratio <= config['timing']['overhead_screen_max_ratio_to_static'],
        measured_instances=len(config['timing']['rows']), repeats=config['timing']['repetitions'],
        prefix_state_continuation_implemented=True, full_native_solvers_frozen_unchanged=True,
        resumed_prefix_includes_forced_depot_and_start_plus_ten_network_actions=True,
        max_historical_cost_error=max(r['historical_absolute_error'] for r in records),
        timing_scope='Native CPU collation/H2D + both encoders/probes + readout + full chosen native solve, all customer POMO starts',
        overhead_limit_predeclared=True, test_read=False, costs_not_used_as_new_training_labels=True))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--stage', choices=('all', 'prepare', 'cache', 'train', 'timing'), default='all')
    args = parser.parse_args()
    configure_numeric()
    if args.stage in ('all', 'prepare'):
        prepare(args.root)
    if args.stage in ('all', 'cache'):
        make_caches(args.root)
    if args.stage in ('all', 'train'):
        results = []
        for name in MODELS:
            for split, values in train_one(args.root, name).items():
                results.append(dict(model=name, split=split, **values))
            torch.cuda.empty_cache()
        write_csv(args.root / 'comparison.csv', results)
        dump(args.root / 'training_complete.json', dict(complete=True, models=list(MODELS), updates_each=4000, test_read=False))
    if args.stage in ('all', 'timing'):
        measure_timing(args.root)


if __name__ == '__main__':
    main()

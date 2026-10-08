"""R44: paired, budget-limited R43A continuations; test is stage-gated."""

import argparse
import copy
import json
import math
import shutil
import subprocess
import sys
import time
from collections import Counter
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import numpy as np
import torch

from code.unified_selector.registry import PROBLEMS, PROBLEM_DESCRIPTOR_FIELDS
from .conditional_node_adapter import CONDITION_FIELDS, MODES, ConditionalNodeSelector, load_baseline
from .multitask_probe import Tee, dump, file_hash, gather_batch, state_hash
from .pairwise_objective import inputs_only, predicted_top3, single_objective
from .pairwise_selector import make_r43_model, maximin_logits
from .performance_experiment import SOURCE_FILES
from .r40_experiment import aggregate, families
from .r42_experiment import paired_schedule, prepare_data
from .r43_experiment import attach_targets, policy_metrics, update_batch
from .tensor_loader import TensorBatchLoader
from .train import configure_torch, print_eval, set_seed
from .training_monitor import capture_rng, restore_rng


ROOT = Path('code/V4/runs/R44_conditional_node_adapter')
BASELINE = Path('code/V4/runs/R43_pairwise_selection/score_difference_seed2/best.pt')
NEW_SOURCES = ('conditional_node_adapter.py', 'r44_experiment.py', 'r44_analysis.py', 'test_r44.py', 'run_v4_r44.sh')
SEEDS = (2, 17, 42)
PRIMARY = 'macro_vs_sbs_pct'


def run_directory(root, arm, seed):
    return root / f'{arm}_seed{seed}'


def initialize(checkpoint, metadata, arm, seed, device):
    set_seed(seed)
    params = copy.deepcopy(checkpoint['args']['model_params'])
    params.update(node_adapter_mode=MODES[arm], log_n_mean=metadata['log_n_mean'], log_n_std=metadata['log_n_std'])
    model = ConditionalNodeSelector(**params).to(device)
    load_baseline(model, checkpoint['model'])
    return model, params


def optimizer_for(model, protocol):
    base = [p for name, p in model.named_parameters() if not name.startswith('node_adapter.')]
    groups = [dict(params=base, peak_lr=protocol['base_peak_lr'], name='original')]
    if model.node_adapter is not None:
        groups.append(dict(params=list(model.node_adapter.parameters()), peak_lr=protocol['adapter_peak_lr'], name='adapter'))
    if not all(p.requires_grad for group in groups for p in group['params']):
        raise ValueError('All original and added parameters must remain trainable')
    return torch.optim.AdamW(groups, lr=protocol['base_peak_lr'], weight_decay=protocol['weight_decay'])


def lr_factor(step, total, warmup):
    if step <= warmup:
        return step / max(warmup, 1)
    progress = (step - warmup) / max(total - warmup, 1)
    return .1 + .9 * (1. + math.cos(math.pi * progress)) / 2.


def fit_metadata(loaders):
    values = torch.cat([loader.lengths.double().log() for loader in loaders.values()])
    metadata = dict(fields=list(CONDITION_FIELDS), descriptor_source='existing batch[problem_desc], unchanged registry/data semantics',
                    size_source='node_mask.sum(-1); matches loader n; CVRP/MVRP includes the single depot',
                    log_n_mean=float(values.mean()), log_n_std=max(float(values.std(unbiased=False)), 1e-6),
                    normalization_fit='train metadata only; equal weight per task-instance', train_instances=len(values),
                    std_floor=1e-6, problem_descriptors={}, size_boundaries={})
    for p, loader in loaders.items():
        desc = loader.batch['problem_desc'].cpu()
        if desc.size(-1) != len(PROBLEM_DESCRIPTOR_FIELDS) or not torch.equal(desc, desc[:1].expand_as(desc)):
            raise ValueError('Expected the existing fixed semantic descriptor per training problem')
        np.testing.assert_array_equal(loader.batch['node_mask'].sum(-1).cpu(), loader.lengths)
        bits = loader.batch['cbits'][0].cpu()
        torch.testing.assert_close(desc[0, [0, 1, 2, 4, 5]], bits, atol=0, rtol=0)
        metadata['problem_descriptors'][p] = desc[0].tolist()
        metadata['size_boundaries'][p] = np.unique(np.quantile(loader.lengths.numpy(), [.25, .5, .75])).tolist()
    if values.std() == 0 and len({tuple(v) for v in metadata['problem_descriptors'].values()}) == 1:
        raise ValueError('A constant condition cannot test conditional adaptation')
    return metadata


def gate_summary(gates):
    return dict(finite=bool(np.isfinite(gates).all()), mean=float(gates.mean()), std=float(gates.std()),
                min=float(gates.min()), max=float(gates.max()), mean_abs_from_one=float(np.abs(gates-1).mean()),
                saturation_fraction=float(((gates < .05) | (gates > 1.95)).mean()),
                channel_mean=gates.mean(0).tolist())


@torch.no_grad()
def evaluate(model, loaders, raw, metadata, batch_size, prediction_dir=None):
    was_training = model.training
    model.eval()
    per, groups, gate_stats = {}, {}, {}
    if prediction_dir is not None:
        prediction_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    try:
        for p, loader in loaders.items():
            arrays = {key: [] for key in ('utility', 'logits', 'pair_margin', 'solver_mask')}
            gates = []
            for batch in TensorBatchLoader(loader.batch, batch_size, False, False):
                clean = inputs_only(batch)
                output = model(clean, return_utility=True)
                for key in arrays:
                    arrays[key].append(output[key].cpu())
                if model.params['node_adapter_mode'] == 'conditional':
                    gates.append(model.node_adapter.gate(clean, batch['node_mask']).float().cpu())
            arrays = {key: torch.cat(value).numpy() for key, value in arrays.items()}
            if len(arrays['logits']) != loader.size:
                raise ValueError('Evaluation must retain every instance, including the tail')
            result, pred = policy_metrics(arrays['logits'], raw[p])
            per[p] = result
            n = loader.lengths.numpy()
            bucket = np.digitize(n, metadata['size_boundaries'][p], right=True)
            groups[p] = {}
            for index in np.unique(bucket):
                take = bucket == index
                subset = dict(raw[p], costs=raw[p]['costs'][take], winner=raw[p]['winner'][take])
                row, _ = policy_metrics(arrays['logits'][take], subset)
                groups[p][str(index)] = dict(count=int(take.sum()), min_nodes=int(n[take].min()), max_nodes=int(n[take].max()), **row)
            if gates:
                values = torch.cat(gates).numpy()
                gate_stats[p] = dict(overall=gate_summary(values), by_size={str(j): gate_summary(values[bucket == j]) for j in np.unique(bucket)})
                arrays['gates'] = values
            if prediction_dir is not None:
                np.savez_compressed(prediction_dir / f'{p}.npz', **arrays, indices=np.arange(loader.size), task=np.array(p),
                                    pred=pred, winner=raw[p]['winner'], costs=raw[p]['costs'], pool=np.array(raw[p]['pool']),
                                    pool_ids=np.array(raw[p]['pool_ids']), nodes=n, size_bucket=bucket)
    finally:
        model.train(was_training)
    return dict(per_problem=per, macro=aggregate(per), families=families(per), size_groups=groups,
                gates=gate_stats, inference_seconds=time.monotonic()-started, instances=sum(x.size for x in loaders.values()))


def baseline_replay(root, checkpoint, actual, predictions):
    directory = BASELINE.parent / 'val_predictions' / f'epoch{checkpoint["epoch"]+1:03d}'
    changes, largest = [], 0.
    for p in PROBLEMS:
        with np.load(directory / f'{p}.npz') as old, np.load(predictions / f'{p}.npz') as new:
            for key in ('indices', 'pool_ids', 'winner', 'costs', 'nodes'):
                np.testing.assert_array_equal(old[key], new[key])
            differences = np.abs(new['logits']-old['logits'])
            largest = max(largest, float(differences.max()))
            for i in np.flatnonzero(new['pred'] != old['pred']):
                ranked = np.sort(old['logits'][i])
                margin = float(ranked[-1]-ranked[-2])
                delta = float(differences[i].max())
                if margin > 2*delta:
                    raise ValueError('Changed selection is not explained by numerical score differences')
                changes.append(dict(problem=p, index=int(i), historical_margin=margin, max_score_change=delta))
    delta = {k: actual['macro'][k]-checkpoint['macro'][k] for k in actual['macro']}
    if abs(delta[PRIMARY]) >= checkpoint['args']['cost_min_delta_pct'] or abs(delta['macro_top1']) > .001:
        raise ValueError(f'Historical baseline replay discrepancy requires investigation: {delta}')
    record = dict(recorded=checkpoint['macro'], current=actual['macro'], differences=delta, changed_choices=changes,
                  max_score_difference=largest, arrays_identical=True, strict_state_load=True,
                  note='Historical RTX4090 vs current RTX3090, same torch/TF32/SDPA. Near-boundary numerical changes are explicit; all R44 arms use the same RTX3090 evaluation.',
                  test_read=False)
    dump(root / 'baseline_replay.json', record)
    return record


def smoke_check(checkpoint, metadata, protocol, loaders, device):
    original = make_r43_model(checkpoint['args']['model_params']).to(device).eval()
    original.load_state_dict(checkpoint['model'], strict=True)
    reports = []
    for arm in MODES:
        model, _ = initialize(checkpoint, metadata, arm, 2, device)
        model.eval()
        with torch.no_grad():
            for p in ('TSP', 'CVRP', 'ATSP', 'OVRPTW'):
                loader = loaders['val'][p]
                idx = torch.tensor([int(loader.lengths.argmin()), int(loader.lengths.argmax())])
                batch = gather_batch(loader, idx)
                expected, actual = original(inputs_only(batch)), model(inputs_only(batch))
                for key in expected:
                    torch.testing.assert_close(actual[key], expected[key], atol=0, rtol=0)
                changed = dict(batch, costs=torch.full_like(batch['costs'], float('nan')), ind=batch['ind'].flip(0))
                torch.testing.assert_close(model(changed)['logits'], actual['logits'], atol=0, rtol=0)
                if model.node_adapter is not None:
                    nodes, mask = model.instance_encoder(inputs_only(batch))
                    model.node_adapter.up.weight.fill_(.001)
                    adapted = model.adapt_nodes(nodes, mask, inputs_only(batch))
                    torch.testing.assert_close(adapted[~mask], nodes[~mask], atol=0, rtol=0)
                    model.node_adapter.up.weight.zero_()
                if not torch.isfinite(actual['logits'][actual['solver_mask']]).all():
                    raise ValueError('Nonfinite valid candidate score')
        if arm != 'C0':
            optimizer = optimizer_for(model, protocol)
            for group in optimizer.param_groups:
                group['lr'] = group['peak_lr']
            scaler = torch.amp.GradScaler('cuda', enabled=device.startswith('cuda'))
            model.train()
            batch = gather_batch(loaders['train']['CVRP'], torch.arange(128))
            for _ in range(2):
                update_batch(model, optimizer, scaler, batch, .35)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=batch['ind'].device.type, dtype=torch.float16, enabled=scaler.is_enabled()):
                out = model(inputs_only(batch))
            focus = predicted_top3(out['logits'], out['logits'], batch['pool_ids'], out['solver_mask'])
            loss, _ = single_objective(out, batch, focus)
            loss.backward()
            gradients = {name: dict(finite=bool(p.grad is not None and torch.isfinite(p.grad).all()),
                                    absolute_sum=float(p.grad.abs().sum()) if p.grad is not None else None)
                         for name, p in model.node_adapter.named_parameters()}
            if not all(value['finite'] for value in gradients.values()):
                raise ValueError('The zero-initialized adapter gradient path did not become finite')
            if not torch.isfinite(model.instance_encoder.coord_proj.weight.grad).all():
                raise ValueError('Original encoder gradient path is not finite')
            reports.append(dict(arm=arm, third_backward=gradients, smoke_discarded=True))
            del optimizer, scaler
        del model
    del original
    return dict(identity_modes=list(MODES), representative_problems=['TSP', 'CVRP', 'ATSP', 'OVRPTW'],
                labels_excluded=True, padding_delta_zero=True, gradient_checks=reports, test_read=False)


def prepare(args):
    args.root.mkdir(parents=True, exist_ok=True)
    path = args.root / 'manifest.json'
    if path.exists():
        verify_manifest(args.root)
        print('[prepared] Reusing the locked R44 protocol', flush=True)
        return
    checkpoint = torch.load(BASELINE, map_location='cpu', weights_only=False)
    original = json.loads((BASELINE.parent.parent / 'protocol.json').read_text())
    result = json.loads((BASELINE.parent / 'result.json').read_text())
    if checkpoint['args']['model_params']['pair_mode'] != 'score_difference':
        raise ValueError('The specified checkpoint is not R43A')
    interval = sum(original['updates_per_epoch'].values())
    total = original['max_epochs'] * interval
    if total != result['successful_updates']:
        raise ValueError('Reference update budget could not be confirmed')
    budget = min(total, interval * math.ceil(max(.25 * total, 5 * interval) / interval))
    protocol = dict(reference_updates=total, validation_interval=interval, successful_update_budget=budget,
                    continuation_epochs=math.ceil(budget/interval), batch_size=original['batch_size'],
                    eval_batch_size=original['eval_batch_size'], base_peak_lr=.1*original['learning_rate'],
                    adapter_peak_lr=original['learning_rate'], warmup_updates=math.ceil(.05*budget),
                    lr_final_fraction=.1, weight_decay=original['weight_decay'], clip_norm=original['clip_norm'],
                    rdrop_weight=original['rdrop_weight'], rdrop_ramp_updates=original['rdrop_ramp_epochs']*interval,
                    delta_screen=original['cost_min_delta_pct'], primary=PRIMARY, primary_unit='percentage points; lower is better',
                    loss=original['loss'], full_coverage=True, drop_last=False, gradient_accumulation=1,
                    amp_dtype='fp16', dropout_passes=2, augmentation=False, sampling='natural distribution, interleaved task batches',
                    initialization='same R43A best, fresh AdamW/scaler, all original and adapter parameters trainable',
                    lr_rule='5% successful-update warmup, then fixed cosine; replaces validation-driven plateau',
                    best_rule='strict minimum validation macro_vs_sbs_pct, including step0',
                    stage_a=dict(seeds=[2], total_updates=3*budget), stage_b=dict(seeds=[17, 42], maximum_total_updates=9*budget),
                    screen='C2 best beats C0 and C1 by delta_screen, and mean of last up-to-five common nonzero validation points is lower than both',
                    test_rule='only after stage A passes and all stage B arms finish; one locked best per run')
    loaders, raw, geometry = prepare_data(args)
    for split in loaders:
        attach_targets(loaders[split], raw[split])
    metadata = fit_metadata(loaders['train'])
    if geometry['mean'] != checkpoint['args']['geometry_stats']['mean'] or geometry['std'] != checkpoint['args']['geometry_stats']['std']:
        raise ValueError('Existing local-geometry normalization changed')
    parameters, schedules = {}, {}
    for seed in SEEDS:
        schedule = paired_schedule(original['sizes'], protocol['continuation_epochs'], protocol['batch_size'], seed)
        torch.save(schedule, args.root / f'schedule_seed{seed}.pt')
        schedules[str(seed)] = state_hash(schedule)
    for arm in MODES:
        model, params = initialize(checkpoint, metadata, arm, 2, args.device)
        count = sum(p.numel() for p in model.parameters())
        added = 0 if model.node_adapter is None else sum(p.numel() for p in model.node_adapter.parameters())
        parameters[arm] = dict(original=count-added, added=added, total=count, rank=None if model.node_adapter is None else model.node_adapter.rank,
                               added_fraction=added/(count-added), model_params=params)
        del model
    base, _ = initialize(checkpoint, metadata, 'C0', 2, args.device)
    initial = evaluate(base, loaders['val'], raw['val'], metadata, protocol['eval_batch_size'], args.root / 'baseline_predictions/val')
    replay = baseline_replay(args.root, checkpoint, initial, args.root / 'baseline_predictions/val')
    dump(args.root / 'baseline_eval.json', initial)
    checks = smoke_check(checkpoint, metadata, protocol, loaders, args.device)
    del base
    sources = {}
    snapshot = args.root / 'source_launch'
    snapshot.mkdir(exist_ok=True)
    for name in set(SOURCE_FILES + NEW_SOURCES + ('pairwise_selector.py', 'pairwise_objective.py', 'r43_experiment.py', 'r42_experiment.py', 'r40_experiment.py')):
        source = Path(__file__).parent / name
        sources[str(source.resolve())] = file_hash(source)
        shutil.copy2(source, snapshot / name)
    for name in ('registry.py', 'data.py'):
        source = Path(__file__).parents[1] / 'unified_selector' / name
        sources[str(source.resolve())] = file_hash(source)
        shutil.copy2(source, snapshot / name)
    gpu = torch.cuda.get_device_properties(args.device)
    manifest = dict(protocol=protocol, baseline=dict(path=str(BASELINE), sha256=file_hash(BASELINE), epoch=checkpoint['epoch']+1,
                    training_seed=original['seed'], successful_updates=checkpoint['successful_updates'], recorded_val=checkpoint['macro'],
                    code_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()),
                    metadata=metadata, parameters=parameters, schedules=schedules, source_hashes=sources,
                    data_manifest=json.loads((args.root / 'data_manifest.json').read_text()),
                    baseline_replay_summary=dict(changed_choices=len(replay['changed_choices']), primary_difference=replay['differences'][PRIMARY]),
                    insertion='DualStreamSelector.encode_state: InstanceEncoder(node_only=True) output [B,N,128] -> adapt_nodes -> first joint_layers block; excludes CLS/condition/stats, includes real depot, masks padding',
                    existing_conditioning='additive problem/constraint/scale/stats condition token in instance attention; solver ID and 34 features enter solver stream; no equivalent multiplicative post-encoder node adapter',
                    paper=dict(url='https://arxiv.org/html/2302.01115v3', gate='Sec2.2.1 Eq2-3, two-layer bounded 2*sigmoid gate',
                               borrowed='EPNet/PPNet channel modulation; not a full reproduction',
                               project_adaptations='GELU, metadata-only gate, bottleneck residual, zero Up/GateOut, no content detach, one insertion'),
                    hardware=dict(host=subprocess.check_output(['hostname'], text=True).strip(), gpu=gpu.name, memory_gib=gpu.total_memory/2**30,
                                  torch=torch.__version__, tf32=torch.backends.cuda.matmul.allow_tf32, slurm_job='1465; two authorized RTX3090s',
                                  other_gpu='RTX4090 occupied by an existing job; leave untouched'), checks=checks, test_read=False)
    dump(args.root / 'checks.json', checks)
    dump(path, manifest)
    for seed in SEEDS:
        for arm in MODES:
            directory = run_directory(args.root, arm, seed)
            directory.mkdir(exist_ok=True)
            dump(directory / 'args.json', dict(protocol, arm=arm, seed=seed, model_params=parameters[arm]['model_params'],
                                             baseline=manifest['baseline'], metadata=metadata, schedule_hash=schedules[str(seed)], wandb_mode='offline'))
    print('[prepared] ' + json.dumps(dict(budget=budget, interval=interval,
          added_parameters={arm: row['added'] for arm, row in parameters.items()}, stage_a_only=True)), flush=True)


def verify_manifest(root):
    manifest = json.loads((root / 'manifest.json').read_text())
    if file_hash(manifest['baseline']['path']) != manifest['baseline']['sha256']:
        raise ValueError('Locked R43A baseline changed')
    for path, digest in manifest['source_hashes'].items():
        if file_hash(path) != digest:
            raise ValueError(f'R44 source changed after preparation: {path}')
    return manifest


def run(args):
    manifest = verify_manifest(args.root)
    protocol, metadata = manifest['protocol'], manifest['metadata']
    if args.seed != 2:
        screen = json.loads((args.root / 'stage_a_screen.json').read_text())
        if not screen['passed']:
            raise ValueError('Stage B is not authorized by the locked validation screen')
    directory = run_directory(args.root, args.arm, args.seed)
    if (directory / 'result.json').exists():
        print(f'[complete already] {directory}', flush=True)
        return
    if (directory / 'last.pt').exists() and not args.resume:
        raise ValueError('Partial run exists; use --resume to continue it')
    mode = 'a' if args.resume else 'w'
    with (directory / 'train.log').open(mode) as stream, redirect_stdout(Tee(sys.stdout, stream)), redirect_stderr(Tee(sys.stderr, stream)):
        _run(args, manifest, directory)


def _run(args, manifest, directory):
    protocol, metadata = manifest['protocol'], manifest['metadata']
    args.batch_size = protocol['batch_size']
    loaders, raw, _ = prepare_data(args)
    for split in loaders:
        attach_targets(loaders[split], raw[split])
    schedule = torch.load(args.root / f'schedule_seed{args.seed}.pt', map_location='cpu', weights_only=False)
    if state_hash(schedule) != manifest['schedules'][str(args.seed)]:
        raise ValueError('Paired task/sample schedule changed')
    checkpoint = torch.load(manifest['baseline']['path'], map_location='cpu', weights_only=False)
    model, _ = initialize(checkpoint, metadata, args.arm, args.seed, args.device)
    optimizer = optimizer_for(model, protocol)
    scaler = torch.amp.GradScaler('cuda', enabled=args.device.startswith('cuda'))
    config = json.loads((directory / 'args.json').read_text())
    history, successful, skipped, counts, samples = [], 0, 0, Counter(), Counter()
    torch.cuda.reset_peak_memory_stats()
    wandb_run = None
    if args.wandb:
        import wandb
        wandb_run = wandb.init(project='selector', entity='yjkds-southern-university-of-science-technology', name='R44_'+directory.name,
                               config=config, dir=str(directory), mode='offline')
    started_all = time.monotonic()

    def save(row):
        state = dict(model=model.state_dict(), optimizer=optimizer.state_dict(), scaler=scaler.state_dict(), args=config,
                     epoch=row['epoch']-1, successful_updates=successful, macro=row['val']['macro'], rng=capture_rng(),
                     updates_by_problem=dict(counts), samples_by_problem=dict(samples), amp_skipped=skipped)
        torch.save(state, directory / 'last.pt')
        best = json.loads((directory / 'best_eval.json').read_text()) if (directory / 'best_eval.json').exists() else None
        if best is None or row['val']['macro'][PRIMARY] < best['val']['macro'][PRIMARY]:
            torch.save(state, directory / 'best.pt')
            dump(directory / 'best_eval.json', row)
            print(f'[best] update={successful} vs_sbs={row["val"]["macro"][PRIMARY]:+.6f}% top1={row["val"]["macro"]["macro_top1"]:.5f}', flush=True)
        dump(directory / 'history.json', history)
        if wandb_run is not None:
            values = {'successful_updates': successful, 'epoch': row['epoch']}
            for split in ('train', 'val'):
                if row[split] is not None:
                    values.update({split+'/'+key: value for key, value in row[split]['macro'].items()})
            values.update({'optimization/'+key: value for key, value in (row['optimization'] or {}).items()})
            wandb_run.log(values)
        from .r44_analysis import plot_run
        plot_run(directory)

    if args.resume and (directory / 'last.pt').exists():
        saved = torch.load(directory / 'last.pt', map_location=args.device, weights_only=False)
        model.load_state_dict(saved['model'], strict=True)
        optimizer.load_state_dict(saved['optimizer'])
        scaler.load_state_dict(saved['scaler'])
        successful, skipped = saved['successful_updates'], saved['amp_skipped']
        counts, samples = Counter(saved['updates_by_problem']), Counter(saved['samples_by_problem'])
        history = json.loads((directory / 'history.json').read_text())
        restore_rng(saved['rng'])
        print(f'[resume] successful_updates={successful}', flush=True)
    else:
        initial = {s: evaluate(model, loaders[s], raw[s], metadata, protocol['eval_batch_size'], directory / f'{s}_predictions/u000000') for s in ('train', 'val')}
        reference = json.loads((args.root / 'baseline_eval.json').read_text())['macro']
        for key in reference:
            if abs(initial['val']['macro'][key]-reference[key]) > 1e-10:
                raise ValueError(f'All arms must share the identical local step0 result: {key}')
        set_seed(args.seed + 440000003)
        row = dict(epoch=0, successful_updates=0, train=initial['train'], val=initial['val'], optimization=None)
        history.append(row)
        save(row)
        print_eval('eval initial', initial['val']['per_problem'], initial['val']['macro'])
    epoch_start = successful // protocol['validation_interval']
    for epoch in range(epoch_start, protocol['continuation_epochs']):
        model.train()
        began, totals, cursor = time.monotonic(), Counter(), Counter()
        for index, p in enumerate(schedule['orders'][epoch]):
            offset = cursor[p]*protocol['batch_size']
            idx = schedule['permutations'][epoch][p][offset:offset+protocol['batch_size']]
            cursor[p] += 1
            factor = lr_factor(successful+1, protocol['successful_update_budget'], protocol['warmup_updates'])
            for group in optimizer.param_groups:
                group['lr'] = group['peak_lr']*factor
            kl_weight = protocol['rdrop_weight']*min(1., (successful+1)/protocol['rdrop_ramp_updates'])
            values = update_batch(model, optimizer, scaler, gather_batch(loaders['train'][p], idx), kl_weight)
            successful += 1
            skipped += values['skipped']
            counts[p] += 1
            samples[p] += len(idx)
            for key in ('loss', 'ce', 'cmp', 'risk', 'kl', 'true3', 'pred3', 'all_pairs'):
                totals[key] += values[key]*len(idx)
            totals['samples'] += len(idx)
            if (index+1) % 180 == 0 or index+1 == len(schedule['orders'][epoch]):
                print(f'[train epoch {epoch+1}] update={successful}/{protocol["successful_update_budget"]} '
                      f'loss={totals["loss"]/totals["samples"]:.5f} KL={totals["kl"]/totals["samples"]:.5f} '
                      f'base_lr={optimizer.param_groups[0]["lr"]:.3g} samples/s={totals["samples"]/(time.monotonic()-began):.0f}', flush=True)
        if totals['samples'] != sum(loader.size for loader in loaders['train'].values()):
            raise ValueError('A full epoch must use every training instance exactly once')
        optimization = {key: totals[key]/totals['samples'] for key in ('loss', 'ce', 'cmp', 'risk', 'kl', 'true3', 'pred3', 'all_pairs')}
        optimization.update(seconds=time.monotonic()-began, samples=totals['samples'], base_lr=optimizer.param_groups[0]['lr'],
                            adapter_lr=optimizer.param_groups[-1]['lr'] if len(optimizer.param_groups) > 1 else 0.,
                            consistency_weight=kl_weight, amp_skipped=skipped)
        name = f'u{successful:06d}'
        validation = evaluate(model, loaders['val'], raw['val'], metadata, protocol['eval_batch_size'], directory / 'val_predictions' / name)
        training = None
        if (epoch+1) % 5 == 0 or epoch+1 == protocol['continuation_epochs']:
            training = evaluate(model, loaders['train'], raw['train'], metadata, protocol['eval_batch_size'], directory / 'train_predictions' / name)
        row = dict(epoch=epoch+1, successful_updates=successful, optimization=optimization, train=training, val=validation,
                   updates_by_problem=dict(counts), samples_by_problem=dict(samples))
        history.append(row)
        print_eval(f'eval epoch {epoch+1}', validation['per_problem'], validation['macro'])
        save(row)
    if successful != protocol['successful_update_budget'] or len(set(counts.values())) != 1 or len(set(samples.values())) != 1:
        raise ValueError('Success budgets or per-task coverage are unequal')
    best = json.loads((directory / 'best_eval.json').read_text())
    last5 = history[1:][-5:]
    final = dict(best=best, final=history[-1], last5_val={key: float(np.mean([r['val']['macro'][key] for r in last5])) for key in last5[-1]['val']['macro']},
                 successful_updates=successful, attempted_updates=successful+skipped, amp_skipped=skipped,
                 updates_by_problem=dict(counts), samples_by_problem=dict(samples), wall_seconds=time.monotonic()-started_all,
                 optimization_seconds=sum(r['optimization']['seconds'] for r in history[1:]),
                 peak_memory_gib=torch.cuda.max_memory_allocated()/2**30, gpu_name=torch.cuda.get_device_name(args.device),
                 baseline_sha256=manifest['baseline']['sha256'], test_read=False)
    dump(directory / 'result.json', final)
    if wandb_run is not None:
        wandb_run.finish()
    print(f'[complete] {directory.name}, best_update={best["successful_updates"]}, successful={successful}', flush=True)


def lock_and_test(args):
    manifest = verify_manifest(args.root)
    screen = json.loads((args.root / 'stage_a_screen.json').read_text())
    if not screen['passed']:
        raise ValueError('Do not read test after a failed stage-A screen')
    if (args.root / 'test_results.json').exists():
        print('[test complete already]', flush=True)
        return
    records = {}
    for seed in SEEDS:
        for arm in MODES:
            directory = run_directory(args.root, arm, seed)
            if not (directory / 'result.json').exists():
                raise ValueError('All Stage-A/B runs must finish before test')
            path = directory / 'best.pt'
            checkpoint = torch.load(path, map_location='cpu', weights_only=False)
            records[directory.name] = dict(path=str(path), sha256=file_hash(path), successful_updates=checkpoint['successful_updates'], val=checkpoint['macro'])
    lock = args.root / 'locked_checkpoints.json'
    if lock.exists() and json.loads(lock.read_text()) != records:
        raise ValueError('Locked checkpoints changed')
    dump(lock, records)
    from .r41_frozen_evaluation import attach_geometry, checked_loader
    loaders, raw = {}, {}
    for p in PROBLEMS:
        loaders[p], raw[p] = checked_loader(p, 'test', manifest['protocol']['eval_batch_size'], args.device)
        attach_geometry(loaders[p], args.device)
    results = {}
    for name, record in records.items():
        checkpoint = torch.load(record['path'], map_location='cpu', weights_only=False)
        model = ConditionalNodeSelector(**checkpoint['args']['model_params']).to(args.device)
        model.load_state_dict(checkpoint['model'], strict=True)
        results[name] = evaluate(model, loaders, raw, manifest['metadata'], manifest['protocol']['eval_batch_size'], args.root / 'test_predictions' / name)
        print_eval(f'test {name} update {record["successful_updates"]}', results[name]['per_problem'], results[name]['macro'])
        del model
    dump(args.root / 'test_data_manifest.json', {p: {k: r[k] for k in ('pool', 'pool_ids', 'data_hash', 'label_hash')} for p, r in raw.items()})
    dump(args.root / 'test_results.json', results)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=('prepare', 'train', 'test'), required=True)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--reference', type=Path, default=Path('code/V4/runs/R39_performance_model'))
    parser.add_argument('--arm', choices=tuple(MODES))
    parser.add_argument('--seed', type=int, choices=SEEDS, default=2)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--wandb', action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    configure_torch()
    torch.set_num_threads(4)
    if args.stage == 'prepare':
        prepare(args)
    elif args.stage == 'train':
        if args.arm is None:
            parser.error('--arm is required for training')
        run(args)
    else:
        lock_and_test(args)


if __name__ == '__main__':
    main()

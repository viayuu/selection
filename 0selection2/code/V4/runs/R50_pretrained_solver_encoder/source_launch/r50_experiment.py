"""R50 random-vs-pretrained frozen solver encoders on the locked R49 split."""

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
from .r48_common import PAIR, binary_labels, binary_metrics, load_split, write_csv
from .r49_experiment import configure_numeric, development_improved
from .r50_readout import RAW_FIELDS, SolverEncoderReadout
from .r50_solver_encoder import build_pair, cache_split, native_inputs, raw_nodes, source_provenance
from .train import set_seed


ROOT = Path('code/V4/runs/R50_pretrained_solver_encoder')
R49 = Path('code/V4/runs/R49_existing_data_diagnosis')
MODELS = ('A_random_seed2', 'B_pretrained_seed2')
SETS = ('train', 'development', 'internal', 'original_val')


def load_locked_data():
    manifest = json.loads((R49 / 'split_manifest.json').read_text())
    data = {}
    for split in ('train', 'val'):
        instances, raw, costs, sizes = load_split(split)
        for field, value in manifest['sources'][split].items():
            if raw[field] != value:
                raise ValueError('Dataset or FP64 costs differ from the locked R49 sources')
        data[split] = dict(instances=instances, raw=raw, costs=costs, sizes=sizes,
                           labels=binary_labels(costs))
    sets = {'train': ('train', np.asarray(manifest['sets']['training_pool']['indices']))}
    sets.update({name: (manifest['sets'][name]['source_split'],
                       np.asarray(manifest['sets'][name]['indices'])) for name in SETS[1:]})
    for name, (source, rows) in sets.items():
        entry = manifest['sets']['training_pool' if name == 'train' else name]
        if int((data[source]['labels'][rows] >= 0).sum()) != entry['n_non_ties']:
            raise ValueError('Exact-tie membership differs from R49')
    return data, manifest, sets


def prepare(root):
    root.mkdir(parents=True, exist_ok=True)
    if (root / 'config.json').exists():
        raise ValueError('R50 is already prepared; refusing to overwrite the locked configuration')
    data, manifest, sets = load_locked_data()
    shutil.copy2(R49 / 'split_manifest.json', root / 'split_manifest.json')
    original_plan = R49 / 'B_8000_seed2/sampling_plan.npz'
    shutil.copy2(original_plan, root / 'sampling_plan.npz')
    schedule = np.load(original_plan)['indices']
    if schedule.shape != (4000, 128) or not np.isin(schedule, sets['train'][1]).all():
        raise ValueError('Original R49B sample sequence does not implement the R50 budget')
    if (data['train']['labels'][schedule] < 0).any():
        raise ValueError('A tied example entered training')
    provenance = source_provenance()
    dump(root / 'pretraining_provenance.json', provenance)
    set_seed(2)
    template = SolverEncoderReadout()
    torch.save(template.state_dict(), root / 'readout_initial.pt')
    config = dict(seed=2, models=list(MODELS), problem='OVRPTW', pair=list(PAIR),
        split='Exact locked R49B 8000/1000/1000 + original 1000 validation; no resplitting',
        counts={name: dict(total=len(rows), non_ties=int((data[source]['labels'][rows] >= 0).sum()))
                for name, (source, rows) in sets.items()},
        readout=dict(encoder_dim=128, hidden_dim=128, dropout=.1,
            node_mlp='263->128 GELU Dropout ->128 GELU',
            pooling='Customer-only masked mean/max + separately encoded depot0 + log1p(customer_count)/6',
            binary_mlp='385->128 GELU Dropout ->2', raw_fields=list(RAW_FIELDS),
            normalization='Two independent trainable channel LayerNorms; no dataset-fitted statistics',
            raw_depot='Original xy, zero demand/service/TW placeholders, is_depot=1'),
        representation='Two frozen eval native six-layer ReLD encoders; no route decoding/solver selection',
        encoder_input='Native depot xy and customer xy/demand/tw_start/tw_end; service is passed to readout, not added to native encoder',
        random_control='Original solver constructor with seed2; no reset/weight loading; freeze exactly the same encoder architectures',
        trainable_parameters=sum(p.numel() for p in template.parameters()),
        initial_readout_sha256=state_hash(template.state_dict()),
        optimizer='fresh AdamW', learning_rate=1e-3, weight_decay=1e-4, dropout=.1,
        batch_size=128, updates=4000, evaluation_interval=200, gradient_clip=1.,
        loss='unweighted binary CE; original FP64 pair cost labels; exact ties excluded',
        amp=False, tf32=False, scheduler=None, early_stop=False, data_augmentation=False,
        selection='Strict minimum development CE at update0 and each200 updates; one checkpoint for all evaluation sets',
        internal_and_original_val_select=False, test_read=False,
        sampling_plan_sha256=file_hash(root / 'sampling_plan.npz'),
        split_manifest_sha256=file_hash(root / 'split_manifest.json'),
        provenance_sha256=file_hash(root / 'pretraining_provenance.json'),
        r49_reference='B_8000_seed2, its already locked development-selected checkpoint; no retraining',
        advancement_screen='B gains >=5pp accuracy over R49B on BOTH internal and original validation, lowers BOTH pair regrets, and beats random A',
        runtime=dict(torch=torch.__version__, gpu=torch.cuda.get_device_name(0),
            gpu_uuid=os.environ.get('CUDA_VISIBLE_DEVICES'), slurm_job=os.environ.get('SLURM_JOB_ID')),
        wandb=dict(project='selector', entity='yjkds-southern-university-of-science-technology', mode='offline'),
        git_parent=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip())
    snapshot = root / 'source_launch'
    snapshot.mkdir(exist_ok=True)
    config['sources'] = {}
    for name in ('r50_readout.py', 'r50_solver_encoder.py', 'r50_experiment.py',
                 'r50_analysis.py', 'test_r50.py', 'run_v4_r50.sh'):
        path = Path(__file__).parent / name
        shutil.copy2(path, snapshot / name)
        config['sources'][name] = file_hash(path)
    config['dependencies'] = {name: file_hash(Path(__file__).parent / name) for name in
        ('r48_common.py', 'r41_label_stability.py', 'performance_targets.py')}
    dump(root / 'config.json', config)
    return config


def make_caches(root):
    data, _, _ = load_locked_data()
    provenance = json.loads((root / 'pretraining_provenance.json').read_text())
    cache_dir = root / 'feature_cache'
    cache_dir.mkdir(exist_ok=True)
    records = {}
    for name in MODELS:
        start = time.monotonic()
        pair, module, witness = build_pair(name == MODELS[1], 'cuda:0', provenance, data['train']['instances'])
        initial_hash = state_hash(pair.state_dict())
        if any(p.requires_grad for p in pair.parameters()) or pair.training:
            raise ValueError('Solver encoders are not frozen/eval')
        torch.save(pair.state_dict(), cache_dir / (name + '_encoders.pt'))
        entry = dict(encoder_sha256=initial_hash, witness=witness,
            frozen_parameter_count=sum(p.numel() for p in pair.parameters()), splits={})
        for split, values in data.items():
            path = cache_dir / (name + '_' + split + '.pt')
            if path.exists():
                raise ValueError('Refusing to replace an existing representation cache')
            cache = cache_split(pair, module, values['instances'], values['sizes'])
            cache.update(source_data_hash=values['raw']['data_hash'], encoder_sha256=initial_hash)
            torch.save(cache, path)
            entry['splits'][split] = dict(path=str(path), sha256=file_hash(path),
                shape=list(cache['encoded'].shape), rows=len(values['instances']),
                data_hash=values['raw']['data_hash'], padding_after_encoding=True)
            print(f'[R50 cache] {name}/{split} shape={cache["encoded"].shape}', flush=True)
            del cache
        if state_hash(pair.state_dict()) != initial_hash:
            raise ValueError('Frozen solver state changed while caching')
        entry.update(seconds=time.monotonic() - start, frozen_state_unchanged=True,
                     route_decoding_calls=0, label_input=False)
        records[name] = entry
        dump(root / 'cache_manifest.json', records)
        del pair
        torch.cuda.empty_cache()
    if records[MODELS[0]]['encoder_sha256'] == records[MODELS[1]]['encoder_sha256']:
        raise ValueError('Random and pretrained encoders have identical weights')
    dump(root / 'preflight.json', dict(
        pretrained_distinct_from_random=True,
        original_pre_forward_matches_encoder_only={name: records[name]['witness']['checks'] for name in MODELS},
        encoders_frozen_and_unchanged=True, label_input=False, route_decoding_calls=0,
        padding_introduced_only_after_encoding=True, native_six_field_input_checked_all_rows=True,
        readout_shared_initialization=True, original_r49b_sampling_plan=True,
        no_selector_pretrained_weights=True))


def load_cache(root, name, device):
    manifest = json.loads((root / 'cache_manifest.json').read_text())[name]
    caches = {}
    for split, record in manifest['splits'].items():
        path = Path(record['path'])
        if file_hash(path) != record['sha256']:
            raise ValueError('Frozen representation cache changed')
        cache = torch.load(path, map_location='cpu', weights_only=True)
        if cache['encoder_sha256'] != manifest['encoder_sha256']:
            raise ValueError('Wrong frozen encoder/cache pairing')
        caches[split] = {k: v.to(device) for k, v in cache.items() if torch.is_tensor(v)}
    return caches


def take(cache, rows):
    rows = torch.as_tensor(rows, device=cache['encoded'].device)
    return tuple(cache[name][rows] for name in ('encoded', 'raw_node', 'node_mask'))


@torch.no_grad()
def evaluate(model, caches, data, sets):
    model.eval()
    metrics, scores = {}, {}
    for name, (source, rows) in sets.items():
        scores[name] = np.concatenate([model(*take(caches[source], indices)).cpu().numpy()
            for indices in torch.as_tensor(rows).split(128)])
        metrics[name] = binary_metrics(scores[name], data[source]['costs'][rows])
    return metrics, scores


def save_predictions(directory, scores, sets, data):
    for name, logits in scores.items():
        source, rows = sets[name]
        np.savez_compressed(directory / (name + '_predictions.npz'), scores=logits, indices=rows,
            costs=data[source]['costs'][rows], labels=data[source]['labels'][rows],
            customers=data[source]['sizes'][rows])


def train_one(root, name):
    directory = root / name
    directory.mkdir(exist_ok=True)
    if (directory / 'last.pt').exists():
        raise ValueError('R50 output exists; refusing an accidental rerun')
    config = json.loads((root / 'config.json').read_text())
    data, _, sets = load_locked_data()
    caches = load_cache(root, name, 'cuda:0')
    model = SolverEncoderReadout().cuda()
    template = torch.load(root / 'readout_initial.pt', map_location='cpu', weights_only=True)
    model.load_state_dict(template, strict=True)
    if state_hash(model.state_dict()) != config['initial_readout_sha256']:
        raise ValueError('Readout initializations differ')
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    plan_path = root / 'sampling_plan.npz'
    if file_hash(plan_path) != config['sampling_plan_sha256']:
        raise ValueError('Locked sample order changed')
    plan = np.load(plan_path)['indices']
    labels = torch.as_tensor(data['train']['labels'], device='cuda:0')
    args = dict(config, model=name, representation='pretrained' if name == MODELS[1] else 'random',
                cache_manifest_sha256=file_hash(root / 'cache_manifest.json'))
    dump(directory / 'args.json', args)
    import wandb
    run = wandb.init(project='selector', entity='yjkds-southern-university-of-science-technology',
        name='R50_' + name, dir=str(directory), config=args, mode='offline')
    set_seed(2)
    history, best, loss_sum, grad_sum = [], float('inf'), 0., 0.
    start = time.monotonic()
    try:
        for updates in range(4001):
            if updates:
                model.train()
                rows = torch.as_tensor(plan[updates - 1], device='cuda:0')
                optimizer.zero_grad(set_to_none=True)
                loss = F.cross_entropy(model(*take(caches['train'], rows)), labels[rows])
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite readout CE')
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                if updates == 1:
                    witness = {key: float(sum(p.grad.abs().sum() for p in block.parameters() if p.grad is not None))
                        for key, block in (('moel_norm', model.moel_norm), ('mtl_norm', model.mtl_norm),
                                           ('node_mlp', model.node_mlp), ('binary_head', model.head))}
                    if min(witness.values()) <= 0:
                        raise ValueError('Readout block did not receive a gradient')
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
            checkpoint = dict(model=model.state_dict(), updates=updates, metrics=metrics, config=args,
                selection='development CE only', encoder_state_path=str(root / 'feature_cache' / (name + '_encoders.pt')))
            if improved:
                torch.save(checkpoint, directory / 'best.pt')
            torch.save(dict(checkpoint, optimizer=optimizer.state_dict()), directory / 'last.pt')
            dump(directory / 'history.json', history)
            summary = '; '.join(f'{key}:acc={v["accuracy"]:.4f} CE={v["ce"]:.5f} regret={v["pair_regret_pct"]:.4f}%'
                                 for key, v in metrics.items())
            message = f'[R50 {name}] u={updates} {summary} dev_best={improved}'
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
        for key, values in metrics.items():
            for field, value in values.items():
                if not np.isclose(value, selected['metrics'][key][field], atol=1e-12, rtol=0):
                    raise ValueError('Selected checkpoint failed metric replay')
        save_predictions(directory, scores, sets, data)
        dump(directory / 'selection.json', dict(model=name, best_updates=selected['updates'],
            successful_updates=4000, presentations=512000,
            best_checkpoint_sha256=file_hash(directory / 'best.pt'),
            initial_readout_sha256=config['initial_readout_sha256'],
            sampling_plan_sha256=config['sampling_plan_sha256'], development_ce=metrics['development']['ce'],
            training_seconds=time.monotonic() - start, trainable_parameters=config['trainable_parameters'],
            selection_uses_internal_or_original_val=False, frozen_encoders_not_in_optimizer=True))
        return metrics
    finally:
        run.finish()


@torch.no_grad()
def measure_timing(root):
    data, _, _ = load_locked_data()
    provenance = json.loads((root / 'pretraining_provenance.json').read_text())
    results = {}
    # A fixed input-derived size and rows, not chosen by cost or classifier outcome.
    sizes = data['train']['sizes']
    size = int(np.median(sizes))
    candidates = np.flatnonzero(sizes == size)
    for name in MODELS:
        pair, module, _ = build_pair(name == MODELS[1], 'cuda:0', provenance)
        saved = torch.load(root / 'feature_cache' / (name + '_encoders.pt'), map_location='cuda:0', weights_only=True)
        pair.load_state_dict(saved, strict=True)
        checkpoint = torch.load(root / name / 'best.pt', map_location='cuda:0', weights_only=False)
        model = SolverEncoderReadout().cuda().eval()
        model.load_state_dict(checkpoint['model'], strict=True)
        records = []
        for batch_size in (1, 128):
            rows = np.resize(candidates, batch_size).tolist()
            instances = data['train']['instances']
            batch, _ = native_inputs(module, instances, rows, 'cuda:0')
            mask = torch.ones(batch_size, size + 1, dtype=torch.bool, device='cuda:0')
            encoded, raw = pair(batch), raw_nodes(batch)
            def readout_only():
                return model(encoded, raw, mask)

            def encoder_readout():
                return model(pair(batch), raw_nodes(batch), mask)

            def end_to_end():
                native, _ = module.collate_problem_batch('OVRPTWtrain', instances, rows)
                native = tuple(value.to('cuda:0') for value in native)
                return model(pair(native), raw_nodes(native), mask)

            expected = encoder_readout()
            torch.testing.assert_close(end_to_end(), expected, rtol=0, atol=0)
            for label, function in (('cached_readout', readout_only),
                                    ('both_encoders_and_readout', encoder_readout),
                                    ('native_collation_transfer_both_encoders_readout', end_to_end)):
                for _ in range(5):
                    function()
                times = []
                for _ in range(30):
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    function()
                    torch.cuda.synchronize()
                    times.append((time.perf_counter() - start) * 1000)
                records.append(dict(component=label, batch_size=batch_size, customers=size,
                    mean_ms=float(np.mean(times)), median_ms=float(np.median(times)),
                    p95_ms=float(np.quantile(times, .95)), per_instance_mean_ms=float(np.mean(times) / batch_size),
                    repetitions=30, warmup=5, synchronized=True))
        results[name] = dict(records=records, frozen_state_sha256=state_hash(pair.state_dict()))
        dump(root / 'inference_timing.json', results)
        del pair, model, saved, checkpoint
        torch.cuda.empty_cache()


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
        dump(args.root / 'training_complete.json', dict(complete=True, updates_each=4000,
            models=list(MODELS), test_read=False))
    if args.stage in ('all', 'timing'):
        measure_timing(args.root)


if __name__ == '__main__':
    main()

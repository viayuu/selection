"""R41: locked checkpoints, full test coverage, native winners and raw FP64 costs."""

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch

from code.unified_selector.registry import GLOBAL_SOLVERS, PROBLEMS
from .direct_selector import make_r40_model
from .local_geometry import FIELDS, local_geometry
from .multitask_probe import dump, file_hash, gather_batch, state_hash
from .r40_experiment import aggregate, classification_metrics, families
from .performance_targets import read_raw_costs
from .tensor_loader import TensorBatchLoader
from .train import configure_torch, make_loader


ROOT = Path('code/V4/runs/R41_signal_audit')
CHECKPOINTS = {
    'R34': 'code/V4/runs/R34_winner_cost_seed2_4090/best.pt',
    'R39A': 'code/V4/runs/R39_performance_model/winner_cost_seed2/best.pt',
    'R40A': 'code/V4/runs/R40_discriminative_training/dual_stream_seed2/best.pt',
    'R40B': 'code/V4/runs/R40_discriminative_training/direct_seed2/best.pt',
}


def load_frozen(record, device):
    path = Path(record['path'])
    if file_hash(path) != record['sha256']:
        raise ValueError('A locked checkpoint changed')
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    params = checkpoint['args']['model_params']
    if params['solver_feature_spec']['solver_names'] != GLOBAL_SOLVERS:
        raise ValueError('Checkpoint/global solver identity order changed')
    if params.get('decision_head', 'classification') != 'classification':
        raise ValueError('R41 only evaluates the four preselected classification policies')
    model = make_r40_model(params).to(device)
    model.load_state_dict(checkpoint['model'], strict=True)
    model.eval()
    model.requires_grad_(False)
    return model


def attach_geometry(loader, device):
    """Cache raw features, never fit normalization; that remains in model buffers."""
    if loader.batch['kind'] != 'coord':
        return
    features = torch.zeros(loader.size, loader.batch['node'].size(1), len(FIELDS), device=device)
    for start in range(0, loader.size, 16):
        batch = gather_batch(loader, torch.arange(start, min(start + 16, loader.size)))
        geometry = local_geometry(batch['node'][..., :2], batch['node_mask']).float()
        if not torch.isfinite(geometry).all():
            raise ValueError('Nonfinite label-free local geometry')
        features[start:start + len(geometry), :geometry.size(1)] = geometry
    loader.batch['node_geom'] = features


def checked_loader(problem, split, batch_size, device):
    raw = read_raw_costs(problem, split)
    loader = make_loader(problem, split, batch_size, 0, shuffle=False, cache_device=device)
    np.testing.assert_array_equal(loader.batch['ind'].cpu(), raw['winner'])
    np.testing.assert_array_equal(loader.batch['costs'].cpu(), raw['costs'].astype(np.float32))
    np.testing.assert_array_equal(loader.batch['pool_ids'].cpu(), raw['pool_ids'])
    if loader.size != len(raw['winner']):
        raise ValueError('Dataset/raw label coverage changed')
    loader.drop_last = False
    return loader, raw


@torch.no_grad()
def predict(model, loader, raw, batch_size, path):
    outputs = []
    for batch in TensorBatchLoader(loader.batch, batch_size, False, False):
        inputs = {key: value for key, value in batch.items() if key not in ('costs', 'ind', 'performance_target')}
        outputs.append(model(inputs)['logits'].float().cpu())
    logits = torch.cat(outputs).numpy()
    if len(logits) != loader.size:
        raise ValueError('Evaluation lost the final partial batch')
    result, pred = classification_metrics(logits, raw)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, indices=np.arange(len(pred)), logits=logits, pred=pred,
                        winner=raw['winner'], costs=raw['costs'], pool_ids=np.array(raw['pool_ids']),
                        nodes=loader.lengths.numpy(), pool=np.array(raw['pool']))
    return result


def lock_protocol(root):
    from .r41_label_stability import prepare_indices
    root.mkdir(parents=True, exist_ok=True)
    path = root / 'frozen_checkpoints.json'
    records = {}
    for name, value in CHECKPOINTS.items():
        checkpoint = torch.load(value, map_location='cpu', weights_only=False)
        params = checkpoint['args']['model_params']
        state = checkpoint['model']
        geometry = {key.rsplit('.', 1)[-1]: tensor.tolist() for key, tensor in state.items()
                    if key in ('instance_encoder.geometry_residual.mean', 'instance_encoder.geometry_residual.std')}
        records[name] = dict(path=value, sha256=file_hash(value), epoch=checkpoint['epoch'] + 1,
                             architecture=params['architecture'], decision_head='classification',
                             original_val=checkpoint['macro'], geometry_buffers=geometry,
                             model_params=params, state_hash=state_hash(state))
    if path.exists():
        if json.loads(path.read_text()) != records:
            raise ValueError('Frozen model protocol changed; preserve the original R41 artifacts')
        return records
    if (root / 'test_results.json').exists():
        raise ValueError('Cannot lock models after test evaluation')
    prepare_indices(root)
    dump(path, records)
    dump(root / 'protocol.json', dict(seed=2, selector_training=False, problems=PROBLEMS,
         fixed_checkpoints=CHECKPOINTS, checkpoint_rule='historical minimum validation macro_vs_sbs_pct; no reselection',
         label_rule='native ind', cost_dtype='raw FP64, not recast cached FP32',
         inference_dtype='FP32; historical SDPA/TF32 settings', augmentation=False, drop_tail=False,
         decision='classification logits argmax for all four models', geometry_normalization='checkpoint buffers only',
         sbs_definition='best fixed solver by evaluation split mean cost, same historical metric convention',
         macro_definition='arithmetic mean of 18 per-problem metrics, including percentages',
         sample_lock_before_test=True, repeats={'deterministic': 2, 'stochastic_seeds': [2, 3, 4]},
         tie_rtol=1e-6, tie_atol=1e-6, old_gap_boundaries_pct=[.1, .5],
         source_hashes={name: file_hash(Path(__file__).parent / name) for name in
                        ('V4Model.py', 'dual_stream.py', 'direct_selector.py', 'local_geometry.py', 'solver_features.py',
                         'r41_frozen_evaluation.py', 'r41_label_stability.py')},
         timestamp=time.strftime('%Y-%m-%d %H:%M:%S %z')))
    return records


def evaluate_frozen(root, device, batch_size):
    records = lock_protocol(root)
    if (root / 'test_results.json').exists():
        raise ValueError('Frozen test evaluation already exists; do not repeat/reselect it silently')
    models = {name: load_frozen(record, device) for name, record in records.items()}
    batch_sizes = {}
    for name, record in records.items():
        args = torch.load(record['path'], map_location='cpu', weights_only=False)['args']
        batch_sizes[name] = args.get('eval_batch_size', args.get('batch_per_problem', batch_size))
    dump(root / 'evaluation_settings.json', dict(batch_sizes=batch_sizes, inference_dtype='FP32',
         note='Replay each historical evaluation batch shape; batch changes can flip near-equal FP32 logits.',
         source_hash=file_hash(__file__)))
    data_manifest, results, val_results = {}, {}, {}
    for split in ('val', 'test'):
        per_model = {name: {} for name in models}
        for problem in PROBLEMS:
            started = time.monotonic()
            loader, raw = checked_loader(problem, split, batch_size, device)
            data_manifest[f'{problem}/{split}'] = {key: raw[key] for key in ('pool', 'pool_ids', 'label_hash', 'data_hash')}
            attach_geometry(loader, device)
            for name, model in models.items():
                path = root / f'{split}_predictions' / name / f'{problem}.npz'
                per_model[name][problem] = predict(model, loader, raw, batch_sizes[name], path)
                r = per_model[name][problem]
                print(f'[{split} {name} {problem}] top1={r["top1"]:.4f} top2={r["top2"]:.4f} '
                      f'top3={r["top3"]:.4f} mean_cost={r["mean_cost"]:.6f} vs_sbs={r["vs_sbs_pct"]:+.4f}% '
                      f'actual_regret={r["actual_regret_pct"]:.4f}%', flush=True)
            print(f'[coverage] {split}/{problem}: {loader.size} instances, seconds={time.monotonic()-started:.1f}', flush=True)
        output = {name: dict(per_problem=per, macro=aggregate(per), families=families(per)) for name, per in per_model.items()}
        if split == 'val':
            val_results = output
            checks = {}
            for name, value in output.items():
                original = records[name]['original_val']
                current = value['macro']
                checks[name] = dict(top1_difference=current['macro_top1'] - original['macro_top1'],
                                    vs_sbs_difference_pct=current['macro_vs_sbs_pct'] - original['macro_vs_sbs_pct'])
                # R34 historical cost summaries used FP32; R41 explicitly recomputes from raw FP64.
                if abs(checks[name]['top1_difference']) > 1e-12 or abs(checks[name]['vs_sbs_difference_pct']) > 1e-4:
                    dump(root / 'validation_replay_failure.json', dict(model=name, checks=checks, results=output, test_read=False))
                    raise ValueError(f'{name}: historical validation replay failed: {checks[name]}')
            dump(root / 'validation_replay.json', dict(checks=checks, results=output, passed=True))
        else:
            results = output
            dump(root / 'test_results.json', results)
        dump(root / 'evaluation_data_manifest.json', data_manifest)
    for name, record in records.items():
        if state_hash(models[name].state_dict()) != record['state_hash'] or file_hash(record['path']) != record['sha256']:
            raise ValueError('Frozen model state changed during evaluation')
    dump(root / 'frozen_evaluation_integrity.json', dict(checkpoints_unchanged=True, model_states_unchanged=True,
         native_ind=True, full_test_instances=sum(r['n'] for value in results.values() for r in value['per_problem'].values()), validation_replay=True,
         label_free_forward=True, geometry_fitted_on_test=False, audit_indices_hash=file_hash(root / 'audit_indices.json')))
    for name, value in results.items():
        print('[test ALL] ' + name + ' ' + json.dumps(value['macro']), flush=True)


def predict_audit(root, device, batch_size):
    records = json.loads((root / 'frozen_checkpoints.json').read_text())
    model = load_frozen(records['R39A'], device)
    samples = json.loads((root / 'audit_indices.json').read_text())
    for problem, record in samples['problems'].items():
        for split, sample in record['splits'].items():
            loader, raw = checked_loader(problem, split, batch_size, device)
            indices = np.array(sample['indices'])
            path = root / 'audit_predictions' / f'{problem}_{split}.npz'
            full_path = root / 'val_predictions' / 'R39A' / f'{problem}.npz'
            if split == 'train':
                full_path = root / 'audit_reference_predictions' / f'{problem}_train.npz'
                if not full_path.exists():
                    cache_dir = Path('code/V4/runs/R39_performance_model/geometry_cache')
                    stats = json.loads((cache_dir / 'stats.json').read_text())
                    if stats['sources'][f'train/{problem}'] != raw['data_hash']:
                        raise ValueError('Historical geometry cache no longer matches the train instances')
                    loader.batch['node_geom'] = torch.load(cache_dir / f'{problem}_train.pt',
                                                          map_location=device, weights_only=True)
                    predict(model, loader, raw, 640, full_path)
            with np.load(full_path) as saved:
                values = dict(saved)
            for key in ('indices', 'logits', 'pred', 'winner', 'costs', 'nodes'):
                values[key] = values[key][indices]
            values['indices'] = indices
            values['reference_path'] = np.array(str(full_path))
            values['reference_hash'] = np.array(file_hash(full_path))
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists() and not (path.parent / (path.stem + '_initial_subsetbatch.npz')).exists():
                shutil.copy2(path, path.parent / (path.stem + '_initial_subsetbatch.npz'))
            np.savez_compressed(path, **values)
            print(f'[audit selector] {problem}/{split}: {len(indices)} fixed R39A predictions', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--stage', choices=('prepare', 'evaluate', 'audit-predict'), default='evaluate')
    args = parser.parse_args()
    torch.set_num_threads(1)
    configure_torch()
    from .r41_label_stability import snapshot_stage
    args.root.mkdir(parents=True, exist_ok=True)
    snapshot_stage(args.root, 'frozen-' + args.stage)
    if args.stage == 'prepare':
        lock_protocol(args.root)
    elif args.stage == 'evaluate':
        evaluate_frozen(args.root, args.device, args.batch_size)
    else:
        predict_audit(args.root, args.device, args.batch_size)


if __name__ == '__main__':
    main()

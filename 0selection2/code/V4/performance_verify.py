"""Validate raw-label provenance, replay saved policies, then replay checkpoints."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from code.unified_selector.registry import PROBLEMS
from .V4Model import make_selector
from .multitask_probe import dump, file_hash
from .performance_evaluation import POLICY_METRICS, aggregate, decision_metrics, evaluate_performance, family_metrics, metrics_from_outputs
from .performance_experiment import SOURCE_FILES, prepare_data
from .performance_targets import read_raw_costs
from .tensor_loader import TensorBatchLoader
from .train import configure_torch


def declared_runs(root):
    budget = json.loads((root / 'run_budget.json').read_text())
    names = {'A': 'winner_cost', 'B': 'performance'}
    seeds, groups = budget['seeds'], budget['groups']
    if not seeds or not groups or len(set(seeds)) != len(seeds) or len(set(groups)) != len(groups):
        raise ValueError('Run budget must declare unique nonempty seeds and groups')
    expected = {f'{names[group]}_seed{seed}': (group, seed) for group in groups for seed in seeds}
    actual = {p.name for p in root.glob('*_seed*') if p.is_dir()}
    if actual != set(expected) or budget['expected_runs'] != len(expected):
        raise ValueError(f'Run inventory does not match the declared budget: expected={set(expected)}, actual={actual}')
    for name in expected:
        for filename in ('args.json', 'history.json', 'result.json', 'best.pt', 'last.pt'):
            if not (root / name / filename).is_file():
                raise ValueError(f'Incomplete declared run: {name}/{filename}')
    return budget, expected


def check_history(history, epochs, batch_size, train):
    if [r['epoch'] for r in history] != list(range(1, epochs + 1)):
        raise ValueError('Epoch coverage must be complete, ordered and unique')
    counts = {p: len(train[p]['winner']) // batch_size for p in PROBLEMS}
    for record in history:
        if set(record['per_problem']) != set(PROBLEMS):
            raise ValueError('History does not cover the full problem registry')
        if record['updates'] != record['epoch'] * sum(counts.values()) or record['updates_by_problem'] != counts:
            raise ValueError('Successful update counts do not match the declared full-pass protocol')


def assert_metrics(actual, expected):
    if set(actual) != set(expected):
        raise ValueError('Metric coverage differs')
    for key in actual:
        np.testing.assert_allclose(actual[key], expected[key], atol=1e-9, rtol=1e-9, err_msg=key)


def check_run_identity(config, result, group, seed, epochs):
    expected = (group, seed, epochs)
    for value in (config, result):
        if (value['group'], value['seed'], value['epochs']) != expected:
            raise ValueError('Run identity does not match the declared budget')


def check_checkpoint_record(checkpoint, filename, config, result):
    record = result['best' if filename == 'best.pt' else 'final']
    if checkpoint['epoch'] + 1 != record['epoch']:
        raise ValueError(f'{filename} does not contain the declared selected/final epoch')
    if checkpoint['successful_updates'] != record['updates'] or checkpoint['args'] != config:
        raise ValueError(f'{filename} update count or configuration differs from its run')
    assert_metrics(checkpoint['macro'], record['val'])
    return record


def check_provenance(root, raw, train, scales):
    manifest = json.loads((root / 'data_manifest.json').read_text())
    for split, entries in (('train', train), ('val', raw)):
        if set(manifest[split]) != set(PROBLEMS):
            raise ValueError('Data manifest omits problems')
        for p in PROBLEMS:
            for key in ('pool', 'pool_ids', 'label_hash', 'data_hash'):
                if manifest[split][p][key] != entries[p][key]:
                    raise ValueError(f'Data provenance mismatch: {split}/{p}/{key}')
    if set(scales) != set(PROBLEMS):
        raise ValueError('Training scales omit problems')
    for p in PROBLEMS:
        costs = train[p]['costs']
        centered = costs - costs.mean(1, keepdims=True)
        independent_scale = float(np.sqrt(np.square(centered).mean())) or 1.
        np.testing.assert_allclose(scales[p]['scale'], independent_scale, rtol=1e-12, atol=1e-12)
        if scales[p]['fitted_on'] != 'train' or scales[p]['instances'] != len(costs):
            raise ValueError('Scale fitting scope does not match the original training labels')
        for key in ('pool', 'pool_ids', 'label_hash', 'data_hash'):
            if scales[p][key] != train[p][key]:
                raise ValueError(f'Scale provenance mismatch: {p}/{key}')
    sources = json.loads((root / 'source_launch' / 'hashes.json').read_text())
    if set(sources) != set(SOURCE_FILES) | {'data.py', 'registry.py'}:
        raise ValueError('Launch source hash coverage is incomplete')
    exceptions = json.loads((root / 'post_training_budget_change.json').read_text())['changes'] if (root / 'post_training_budget_change.json').exists() else {}
    for name, expected in sources.items():
        if file_hash(root / 'source_launch' / name) != expected:
            raise ValueError(f'Launch snapshot changed: {name}')
        current = Path(__file__).parents[1] / 'unified_selector' / name if name in ('data.py', 'registry.py') else Path(__file__).parent / name
        observed = file_hash(current)
        if observed != expected:
            if name not in ('evaluate.py', 'performance_analysis.py', 'performance_experiment.py') or exceptions.get(name) != dict(launch_sha256=expected, current_sha256=observed):
                raise ValueError(f'Undocumented source change: {name}')
    return sources


def verify_predictions(root):
    budget, expected_runs = declared_runs(root)
    train = {p: read_raw_costs(p, 'train') for p in PROBLEMS}
    raw = {p: read_raw_costs(p, 'val') for p in PROBLEMS}
    scales = json.loads((root/'performance_scales.json').read_text())
    sources = check_provenance(root, raw, train, scales)
    reports = []
    for name, (group, seed) in sorted(expected_runs.items()):
        directory = root / name
        config = json.loads((directory/'args.json').read_text())
        history = json.loads((directory/'history.json').read_text())
        result = json.loads((directory/'result.json').read_text())
        epochs = budget['epochs_per_run']
        check_run_identity(config, result, group, seed, epochs)
        if config['source_hashes'] != sources or result['source_hashes'] != sources or config['performance_scales'] != scales:
            raise ValueError('Run provenance differs from the launch source/training scales')
        if config['decision_head'] != ('classification' if group == 'A' else 'performance'):
            raise ValueError('Run uses the wrong declared decision policy')
        check_history(history, epochs, config['batch_per_problem'], train)
        if result['epochs'] != epochs or result['updates'] != history[-1]['updates']:
            raise ValueError('Result completion disagrees with history')
        for record in history:
            per = {}
            prediction_dir = directory/'predictions'/f"val_e{record['epoch']:03d}"
            if {p.stem for p in prediction_dir.glob('*.npz')} != set(PROBLEMS):
                raise ValueError('Saved predictions must cover every problem exactly once')
            for problem in PROBLEMS:
                labels = raw[problem]
                path = prediction_dir/f'{problem}.npz'
                with np.load(path) as values:
                    np.testing.assert_array_equal(values['indices'], np.arange(len(labels['winner'])))
                    np.testing.assert_array_equal(values['winner'], labels['winner'])
                    np.testing.assert_array_equal(values['costs'], labels['costs'])
                    np.testing.assert_array_equal(values['pool_ids'], labels['pool_ids'])
                    if float(values['scale']) != scales[problem]['scale']:
                        raise ValueError('Prediction scale differs from the saved training scale')
                    if str(values['decision_head']) != config['decision_head']:
                        raise ValueError('Saved prediction policy differs from the run configuration')
                    r, class_pred, perf_pred = metrics_from_outputs(values['logits'], values['pred_performance'], labels,
                                                                  float(values['scale']), config['decision_head'], values['solver_mask'])
                    np.testing.assert_array_equal(values['classification_pred'], class_pred)
                    np.testing.assert_array_equal(values['performance_pred'], perf_pred)
                    np.testing.assert_array_equal(values['pred'], class_pred if config['group'] == 'A' else perf_pred)
                keys = ('ce', 'performance_mse', 'performance_zero_mse', 'performance_r2', *POLICY_METRICS)
                keys += tuple(f'{key}_{head}' for head in ('classification', 'performance') for key in POLICY_METRICS)
                for key in keys:
                    np.testing.assert_allclose(r[key], record['per_problem'][problem][key], atol=1e-9, rtol=1e-9)
                for head in ('classification', 'performance'):
                    expected_policy = record['per_problem'][problem][head]
                    assert_metrics({key: r[head][key] for key in POLICY_METRICS}, {key: expected_policy[key] for key in POLICY_METRICS})
                np.testing.assert_allclose(r['pick_dist'], record['per_problem'][problem]['pick_dist'], atol=0, rtol=0)
                per[problem] = r
                reports.append(dict(run=directory.name, epoch=record['epoch'], problem=problem, n=r['n']))
            assert_metrics(aggregate(per), record['val'])
            families = family_metrics(per)
            if set(families) != set(record['families']):
                raise ValueError('Family metric coverage differs')
            for family in families:
                assert_metrics(families[family], record['families'][family])
        if result['best'] != min(history, key=lambda r: r['val']['macro_vs_sbs_pct']) or result['final'] != history[-1]:
            raise ValueError('Selected/final result does not match recorded validation history')
        assert_metrics({key: float(np.mean([r['val'][key] for r in history[-5:]])) for key in history[-1]['val']}, result['last5'])
        for family in history[-1]['families']:
            assert_metrics({key: float(np.mean([r['families'][family][key] for r in history[-5:]])) for key in history[-1]['families'][family]}, result['last5_families'][family])
    if len(reports) != len(expected_runs) * budget['epochs_per_run'] * len(PROBLEMS):
        raise ValueError('Incomplete prediction coverage')
    dump(root/'prediction_replay.json', dict(evaluations=len(reports), matched_original_FP64_costs=True,
                                            matched_native_winners=True, main_policy_and_metrics_consistent=True,
                                            diagnostic_policies_and_aggregates_consistent=True, independently_refit_train_scales=True,
                                            source_and_data_hashes_verified=True, expected_runs=sorted(expected_runs),
                                            test_read=False, records=reports))
    return reports


@torch.no_grad()
def verify_checkpoints(args):
    budget, expected_runs = declared_runs(args.root)
    loaders, raw, _, _ = prepare_data(args)
    checks = []
    for name, (group, seed) in sorted(expected_runs.items()):
        directory = args.root / name
        config = json.loads((directory / 'args.json').read_text())
        result = json.loads((directory / 'result.json').read_text())
        check_run_identity(config, result, group, seed, budget['epochs_per_run'])
        for filename in ('best.pt', 'last.pt'):
            checkpoint = torch.load(directory/filename, map_location='cpu', weights_only=False)
            record = check_checkpoint_record(checkpoint, filename, config, result)
            model = make_selector(checkpoint['args']['model_params']).to(args.device)
            model.load_state_dict(checkpoint['model'], strict=True)
            output = directory/('checkpoint_replay_'+filename[:-3])
            per, macro = evaluate_performance(model, PROBLEMS, 'val', args.batch_size, 0, args.device,
                                              loaders=loaders['val'], raw=raw['val'], prediction_dir=output)
            original = directory/'predictions'/f"val_e{record['epoch']:03d}"
            maximum = 0.
            for p in PROBLEMS:
                with np.load(output/f'{p}.npz') as a, np.load(original/f'{p}.npz') as b:
                    for key in ('logits', 'pred_performance'):
                        maximum = max(maximum, float(np.max(np.abs(a[key]-b[key]))))
                        np.testing.assert_allclose(a[key], b[key], atol=2e-5, rtol=2e-5)
                    np.testing.assert_array_equal(a['pred'], b['pred'])
            for key in checkpoint['macro']:
                np.testing.assert_allclose(macro[key], checkpoint['macro'][key], atol=2e-6, rtol=2e-6)
            checks.append(dict(run=directory.name, checkpoint=filename, epoch=checkpoint['epoch']+1,
                               max_output_error=maximum, strict_load=True, selections_match=True,
                               selected_record_and_run_identity_verified=True))
            del model, checkpoint
            torch.cuda.empty_cache()
    if len(checks) != 2 * len(expected_runs):
        raise ValueError('Checkpoint replay coverage is incomplete')
    dump(args.root/'checkpoint_replay.json', dict(checkpoints=checks, test_read=False))
    return loaders, raw


@torch.no_grad()
def reference_r34(args, loaders, raw):
    path = Path('code/V4/runs/R34_winner_cost_seed2_4090/best.pt')
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    model = make_selector(ckpt['args']['model_params']).to(args.device)
    model.load_state_dict(ckpt['model'], strict=True)
    model.eval()
    per = {}
    output = args.root/'reference_R34_predictions'
    output.mkdir(exist_ok=True)
    for problem in PROBLEMS:
        loader = loaders['val'][problem]
        logits = torch.cat([model(batch)['logits'].float().cpu() for batch in TensorBatchLoader(loader.batch, args.batch_size, False, False)])
        r, pred = decision_metrics(logits.numpy(), raw['val'][problem])
        r['ce'] = float(F.cross_entropy(logits, torch.from_numpy(raw['val'][problem]['winner'])))
        per[problem] = r
        np.savez_compressed(output/f'{problem}.npz', logits=logits.numpy(), pred=pred,
                            winner=raw['val'][problem]['winner'], costs=raw['val'][problem]['costs'])
    macro = {f'macro_{key}': float(np.mean([r[key] for r in per.values()])) for key in ('ce', *POLICY_METRICS)}
    dump(args.root/'reference_R34_val.json', dict(checkpoint=str(path), checkpoint_hash=file_hash(path),
                                                checkpoint_epoch=ckpt['epoch']+1, per_problem=per, macro=macro,
                                                test_read=False, precision='FP32 model / raw FP64 costs'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('code/V4/runs/R39_performance_model'))
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--batch-size', type=int, default=640)
    parser.add_argument('--predictions-only', action='store_true')
    args = parser.parse_args()
    configure_torch()
    records = verify_predictions(args.root)
    print(f'[prediction replay] verified {len(records)} validation problem/epoch records', flush=True)
    if not args.predictions_only:
        loaders, raw = verify_checkpoints(args)
        reference_r34(args, loaders, raw)
        print('[checkpoint replay] strict checkpoint and R34 reference evaluation complete; test_read=False', flush=True)


if __name__ == '__main__':
    main()

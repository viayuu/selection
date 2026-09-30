"""Replay R40 decisions, sample budgets and selected/final checkpoints."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from code.unified_selector.registry import PROBLEMS
from .direct_selector import make_r40_model
from .multitask_probe import dump, file_hash, state_hash
from .performance_targets import read_raw_costs
from .performance_verify import assert_metrics
from .r40_experiment import METRICS, ROOT, RUN_NAMES, aggregate, classification_metrics, evaluate, families, learning_rate, prepare_data
from .train import configure_torch


def verify_predictions(root):
    protocol = json.loads((root / 'protocol.json').read_text())
    actual = {p.name for p in root.glob('*_seed*') if p.is_dir()}
    if actual != set(RUN_NAMES.values()):
        raise ValueError('R40 requires exactly the two declared seed2 runs')
    for name, expected in protocol['source_hashes'].items():
        original = root / 'source_launch' / name
        source = Path(__file__).parents[1] / 'unified_selector' / name if name in ('data.py', 'registry.py') else Path(__file__).parent / name
        if file_hash(original) != expected or file_hash(source) != expected:
            raise ValueError(f'R40 source changed: {name}')
    raw = {s: {p: read_raw_costs(p, s) for p in PROBLEMS} for s in ('train', 'val')}
    manifest = json.loads((root / 'data_manifest.json').read_text())
    for s in raw:
        for p in PROBLEMS:
            for key in ('pool', 'pool_ids', 'data_hash', 'label_hash'):
                if raw[s][p][key] != manifest[s][p][key]:
                    raise ValueError('R40 ground truth/candidate identity changed')
    schedule = torch.load(root / 'schedule_seed2.pt', map_location='cpu', weights_only=False)
    if state_hash(schedule) != protocol['schedule_hash']:
        raise ValueError('R40 sample/task sequence changed')
    reports = []
    for group, name in RUN_NAMES.items():
        directory = root / name
        config = json.loads((directory / 'args.json').read_text())
        history = json.loads((directory / 'history.json').read_text())
        result = json.loads((directory / 'result.json').read_text())
        for value in (config, result):
            if (value['group'], value['seed'], value['epochs']) != (group, 2, 60):
                raise ValueError('R40 result/run identity differs from its declared configuration')
        if [r['epoch'] for r in history] != list(range(1, 61)):
            raise ValueError('R40 validation epoch coverage is incomplete or duplicated')
        if result['best'] != min(history, key=lambda r: r['val']['macro_vs_sbs_pct']) or result['final'] != history[-1]:
            raise ValueError('R40 selected/final point differs from validation history')
        assert_metrics(result['last5'], {key: float(np.mean([r['val'][key] for r in history[-5:]])) for key in history[-1]['val']})
        assert_metrics(result['train_final'], history[-1]['train_eval'])
        for record in history:
            if record['updates'] != record['epoch'] * 2700 or record['updates_by_problem'] != {p: record['epoch'] * 150 for p in PROBLEMS}:
                raise ValueError('R40 successful optimizer/task updates differ from the budget')
            per = {}
            folder = directory / 'predictions' / f"val_e{record['epoch']:03d}"
            if {p.stem for p in folder.glob('*.npz')} != set(PROBLEMS):
                raise ValueError('R40 validation predictions omit problems')
            for p in PROBLEMS:
                with np.load(folder / f'{p}.npz') as prediction:
                    labels = raw['val'][p]
                    np.testing.assert_array_equal(prediction['indices'], np.arange(len(labels['winner'])))
                    for key in ('winner', 'costs', 'pool_ids'):
                        np.testing.assert_array_equal(prediction[key], labels[key])
                    r, pred = classification_metrics(prediction['logits'], labels)
                    np.testing.assert_array_equal(prediction['pred'], pred)
                    assert_metrics({k: r[k] for k in METRICS}, {k: record['per_problem'][p][k] for k in METRICS})
                    per[p] = r
                    reports.append(dict(run=name, epoch=record['epoch'], problem=p))
            assert_metrics(aggregate(per), record['val'])
            for family, values in families(per).items():
                assert_metrics(values, record['families'][family])
        positions = {p: 0 for p in PROBLEMS}
        with (directory / 'update_trace.jsonl').open() as stream:
            step = 0
            for epoch, order in enumerate(schedule['orders']):
                positions = {p: 0 for p in PROBLEMS}
                for p in order:
                    row = json.loads(next(stream))
                    indices = schedule['batches'][p][epoch, positions[p]]
                    if (row['step'], row['epoch'], row['problem'], row['batch_size'], row['batch_index']) != (step + 1, epoch + 1, p, 64, positions[p]):
                        raise ValueError('R40 trace violates sample order, real batch size or successful step sequence')
                    if row['batch_hash'] != state_hash(indices):
                        raise ValueError('R40 optimizer update used a different sample batch')
                    np.testing.assert_allclose(row['lr'], learning_rate(step, 162000), atol=0, rtol=0)
                    step += 1
                    positions[p] += 1
            if stream.readline() or step != 162000:
                raise ValueError('R40 trace does not cover exactly the optimizer budget')
        train_per = {}
        for p in PROBLEMS:
            with np.load(directory / 'predictions' / 'train_final' / f'{p}.npz') as prediction:
                labels = raw['train'][p]
                np.testing.assert_array_equal(prediction['indices'], np.arange(len(labels['winner'])))
                for key in ('winner', 'costs', 'pool_ids'):
                    np.testing.assert_array_equal(prediction[key], labels[key])
                train_per[p], pred = classification_metrics(prediction['logits'], labels)
                np.testing.assert_array_equal(prediction['pred'], pred)
        assert_metrics(aggregate(train_per), result['train_final'])
    dump(root / 'prediction_replay.json', dict(validation_records=len(reports), full_train_instances_verified=360000,
                                              successful_optimizer_updates_verified=324000, source_data_and_pool_hashes_verified=True,
                                              original_FP64_costs=True, native_winners=True, test_read=False, records=reports))


@torch.no_grad()
def verify_checkpoints(args):
    loaders, raw, _ = prepare_data(args)
    reports = []
    for name in RUN_NAMES.values():
        directory = args.root / name
        config = json.loads((directory / 'args.json').read_text())
        result = json.loads((directory / 'result.json').read_text())
        for filename, point in (('best.pt', 'best'), ('last.pt', 'final')):
            checkpoint = torch.load(directory / filename, map_location='cpu', weights_only=False)
            expected = result[point]
            if checkpoint['epoch'] + 1 != expected['epoch'] or checkpoint['successful_updates'] != expected['updates'] or checkpoint['args'] != config:
                raise ValueError('R40 checkpoint is not bound to the declared selected/final record')
            assert_metrics(checkpoint['macro'], expected['val'])
            model = make_r40_model(config['model_params']).to(args.device)
            model.load_state_dict(checkpoint['model'], strict=True)
            folder = directory / f'checkpoint_replay_{point}'
            observed = evaluate(model, loaders['val'], raw['val'], args.eval_batch_size, folder)
            assert_metrics(observed['macro'], expected['val'])
            maximum = 0.
            for p in PROBLEMS:
                with np.load(folder / f'{p}.npz') as a, np.load(directory / 'predictions' / f"val_e{expected['epoch']:03d}" / f'{p}.npz') as b:
                    maximum = max(maximum, float(np.max(np.abs(a['logits'] - b['logits']))))
                    np.testing.assert_allclose(a['logits'], b['logits'], atol=2e-5, rtol=2e-5)
                    np.testing.assert_array_equal(a['pred'], b['pred'])
            reports.append(dict(run=name, checkpoint=filename, epoch=expected['epoch'], max_output_error=maximum,
                                selected_record_verified=True, strict_load=True, selections_match=True))
            del model, checkpoint
            torch.cuda.empty_cache()
    dump(args.root / 'checkpoint_replay.json', dict(checkpoints=reports, test_read=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--reference', type=Path, default=Path('code/V4/runs/R39_performance_model'))
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--eval-batch-size', type=int, default=256)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    configure_torch()
    torch.set_num_threads(1)
    verify_predictions(args.root)
    verify_checkpoints(args)
    print('[verified] two runs, 2160 validation records, full train, 324000 optimizer updates and four checkpoints; no test', flush=True)

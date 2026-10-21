"""Evaluate locked historical R79 NSS weights on R84's new 4,500-row TEST.

Pipeline API: ``run_reference(args)`` or the CLI below, AFTER the current R84
model's unique VAL lock exists. Nothing trains or chooses a checkpoint here.
The historical R79 weights were trained on the 20261014 release, not on R84's
20261018 release. Its old 3,895-row TEST metrics are never copied to new TEST.
Original LIB predictions are copied and re-summarized without model inference.

    python -m code.V4.r84_nss_reference --model-run /path/to/R84/group \
        --device cuda:0

Outputs deliberately follow the R79 reporter layout: ROOT/protocol.json,
ROOT/final_test_lock.json, ROOT/{TSP,CVRP}/{dataset}_predictions.npz and matching
``_metrics.json`` files. Pass ROOT as the R84 reporter's ``--nss-root``.
"""

import argparse
import csv
import fcntl
import importlib
import shutil
from pathlib import Path

import numpy as np
import torch

from .r80_reporting import file_hash, json_hash, read_json, resolve_record_path, utc, write_json


V4 = Path(__file__).resolve().parent
DEFAULT_ROOT = V4 / 'runs/R84_new90k_attention/nss_reference'
DEFAULT_R79_ROOT = V4 / 'runs/R79_NSS_new_data_batch1024'
DEFAULT_DATA = Path('/public/home/shiys/selection_data_new90k_B_filtered_single_reld_mt_seed2_20261018')
TASKS = ('TSP', 'CVRP')
TEST_ROWS = 4500
LIB_ROWS = {'TSP': 49, 'CVRP': 100}
WEIGHTS_DESCRIPTION = 'R79 historical VALID-selected NSS weights; not retrained on R84/new90k'
WINNER_RULE = 'saved native ind verified against original FP64 cost argmin; never FP32 argmin'


def require_r84_val_lock(model_run, data_root):
    """This gate must execute before any new TEST dataset or label is opened."""
    model_run, data_root = Path(model_run).resolve(), Path(data_root).resolve()
    config = read_json(model_run / 'protocol.json')
    lock = read_json(model_run / 'final_test_lock.json')
    checkpoint = model_run / 'best.pt'
    if (config.get('experiment') != 'R84' or lock.get('experiment') != 'R84'
            or lock.get('arm') != config.get('arm')
            or Path(config.get('data_root', '')).resolve() != data_root
            or lock.get('selected_without_test') is not True
            or lock.get('per_problem_assembly') is not False
            or lock.get('state_hash') != json_hash(config)
            or not 0 < int(lock.get('selected_update', 0)) <= int(config.get('max_successful_updates', 0))
            or int(lock.get('epoch', 0)) < 1
            or not lock.get('rule') or lock['rule'] != config.get('checkpoint_rule')
            or not lock.get('inference_rule') or lock['inference_rule'] != config.get('inference_rule')
            or resolve_record_path(lock['checkpoint'], model_run) != checkpoint.resolve()
            or file_hash(checkpoint) != lock.get('checkpoint_sha256')):
        raise PermissionError('Require the unchanged unique R84 VALID-selected checkpoint before NSS TEST access')
    return config, lock


def native_metrics(scores, predicted, costs, winner, pool, train_sbs):
    """Primary metrics use the native FP64 winner, including FP32 near ties."""
    scores, predicted = np.asarray(scores), np.asarray(predicted, dtype=np.int64)
    costs, winner = np.asarray(costs), np.asarray(winner, dtype=np.int64)
    n = len(winner)
    if (costs.dtype != np.float64 or costs.shape != (n, len(pool))
            or scores.shape != costs.shape or predicted.shape != (n,)
            or not np.isfinite(scores).all() or not np.isfinite(costs).all()
            or not np.array_equal(winner, costs.argmin(1))
            or np.any(predicted < 0) or np.any(predicted >= len(pool))):
        raise ValueError('NSS reference requires aligned finite scores and original FP64 native winners')
    oracle = costs.min(1)
    if not n or np.any(oracle <= 0):
        raise ValueError('NSS reference requires nonempty rows and positive Oracle costs')
    selected = costs[np.arange(n), predicted]
    fixed = pool.index(train_sbs)
    split_sbs = int(costs.mean(0).argmin())
    mean, sbs, best = float(selected.mean()), float(costs[:, fixed].mean()), float(oracle.mean())
    native = float((predicted == winner).mean())
    return dict(n=n, pool=list(pool), top1=native, top1_native_ind=native,
        top1_fp32_argmin=float((predicted == costs.astype(np.float32).argmin(1)).mean()),
        mean_cost=mean, oracle_cost=best, sbs_solver=train_sbs, sbs_cost=sbs,
        train_sbs_solver=train_sbs, train_sbs_cost=sbs, sbs_selection_split='R84 TRAIN',
        sbs_top1_native_ind=float((winner == fixed).mean()),
        posthoc_split_sbs_solver=pool[split_sbs], posthoc_split_sbs_cost=float(costs[:, split_sbs].mean()),
        vs_sbs_pct=(mean / sbs - 1.) * 100., vs_oracle_pct=(mean / best - 1.) * 100.,
        actual_regret_pct=float(((selected - oracle) / oracle).mean() * 100.),
        winner_rule=WINNER_RULE, weights_description=WEIGHTS_DESCRIPTION,
        fp32_native_winner_disagreements=int((costs.astype(np.float32).argmin(1) != winner).sum()),
        methods=[dict(solver=name, selected_count=int((predicted == i).sum()),
                      winner_count=int((winner == i).sum()), correct_count=int(((predicted == i) & (winner == i)).sum()))
                 for i, name in enumerate(pool)])


def load_test_cache(data_root, task, pool, nss, *, expected_rows=TEST_ROWS):
    """Original NSS native input conversion; no selector embeddings or augmentation."""
    directory = Path(data_root) / (task + 'test')
    metadata = read_json(directory / 'metadata.json')
    if metadata.get('instances') != expected_rows or metadata.get('methods') != pool:
        raise ValueError(f'{task}: new TEST count/candidate order differs from the locked NSS deployment')
    instances = nss.load_pickle(directory / 'dataset.pkl')
    labels = nss.load_pickle(directory / 'raw_label.pkl')
    if len(instances) != expected_rows or len(labels) != expected_rows:
        raise ValueError(f'{task}: incomplete new TEST population')
    if task == 'CVRP':
        instances = [nss.ORIGINAL_UTILS.process_instance_CVRP(x) for x in instances]
    lengths = torch.tensor([x.shape[1] for x in instances], dtype=torch.int64)
    points = torch.cat([x.squeeze(0) for x in instances], 0).float().contiguous()
    offsets = torch.cat((torch.zeros(1, dtype=torch.int64), lengths.cumsum(0)))
    costs = np.asarray([labels[str(i)]['cost'] for i in range(expected_rows)], dtype=np.float64)
    winner = np.asarray([labels[str(i)]['ind'] for i in range(expected_rows)], dtype=np.int64)
    with np.load(directory / 'input_metadata.npz', allow_pickle=False) as inputs:
        source = inputs['source_index'].astype(np.int64)
    if costs.shape != (expected_rows, len(pool)) or source.shape != (expected_rows,):
        raise ValueError(f'{task}: unaligned FP64 costs or instance identities')
    if not np.array_equal(winner, costs.argmin(1)) or not np.isfinite(costs).all():
        raise ValueError(f'{task}: native winners do not match original FP64 costs')
    return dict(points=points, offsets=offsets, lengths=lengths, costs=torch.from_numpy(costs),
        winner=torch.from_numpy(winner), source_indices=torch.from_numpy(source), pool=list(pool),
        gaps=torch.zeros_like(torch.from_numpy(costs)), times=torch.zeros_like(torch.from_numpy(costs)),
        task=task, split='test')


def check_or_write(path, value):
    if Path(path).exists():
        if read_json(path) != value:
            raise ValueError(f'Refusing changed NSS reference protocol: {path}')
    else:
        write_json(path, value)


def bind_reference_protocol(root, protocol):
    """Share fixed NSS predictions across arms while retaining every VAL gate."""
    path = root / 'protocol.json'
    if path.exists():
        previous = read_json(path)
        comparable = lambda value: {key: item for key, item in value.items() if key != 'current_model_lock'}
        if comparable(previous) != comparable(protocol):
            raise ValueError('Shared NSS reference data/weights/evaluation protocol changed')
    else:
        write_json(path, protocol)
        previous = protocol
    consumer = protocol['current_model_lock']
    check_or_write(root / 'val_lock_consumers' / (consumer['checkpoint_sha256'] + '.json'), consumer)
    return previous


def source_records(directory):
    return {name: dict(path=str((directory / name).resolve()), sha256=file_hash(directory / name))
            for name in ('metadata.json', 'dataset.pkl', 'raw_label.pkl', 'input_metadata.npz')}


def read_saved_prediction(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def evaluate_new_test(args, task, nss, frozen, config):
    run = args.root / task
    run.mkdir(parents=True, exist_ok=True)
    prediction = run / 'test_predictions.npz'
    metadata_path = run / 'test_metrics.json'
    receipt_path = run / 'test_once.json'
    identity = dict(task=task, split='test', checkpoint=frozen,
        data_root=str(args.data_root.resolve()), sources=source_records(args.data_root / (task + 'test')),
        n=TEST_ROWS, batch_size=args.batch_size, augmentation=False, training=False,
        weights_description=WEIGHTS_DESCRIPTION, new_test_inference=True)
    if receipt_path.exists():
        receipt = read_json(receipt_path)
        if (receipt.get('state') != 'complete' or receipt.get('identity') != identity
                or not metadata_path.exists() or not prediction.exists()
                or receipt.get('predictions_sha256') != file_hash(prediction)):
            raise RuntimeError(f'{task}: incomplete or changed NSS reference; no automatic inference retry')
        return read_json(metadata_path)
    if prediction.exists() or metadata_path.exists():
        raise RuntimeError(f'{task}: unattributed NSS reference output already exists')
    write_json(receipt_path, dict(state='started', identity=identity, started_utc=utc()))
    try:
        checkpoint_path = Path(frozen['path'])
        state = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        if state.get('original_sources') != nss.source_hashes():
            raise ValueError('Historical NSS checkpoint/source binding changed')
        model_config = state['config']
        if model_config['problem_type'] != task:
            raise ValueError('Historical NSS checkpoint problem differs')
        pool = read_json(nss.NSS / 'solver_order_ours.json')[task]
        cache = load_test_cache(args.data_root, task, pool, nss)
        device = torch.device(args.device)
        model = nss.make_model(model_config).to(device)
        model.load_state_dict(state['model_state_dict'], strict=True)
        model.eval().requires_grad_(False)
        del state
        loader, _ = nss.make_loader(nss.NSSDataset(cache, augment=False), args.batch_size, args.workers)
        with torch.inference_mode():
            scores, seconds = nss.infer(model, loader, device)
        # Retain the original NSS softmax/topk decision, including its exact-tie rule.
        predicted = scores.float().softmax(-1).topk(1, dim=-1).indices[:, 0].numpy()
        metrics = native_metrics(scores.numpy(), predicted, cache['costs'].numpy(),
                                 cache['winner'].numpy(), pool, config['baselines'][task]['sbs_solver'])
        nss.save_predictions(prediction, scores, cache, predicted)
        metrics.update(checkpoint=frozen, data_root=str(args.data_root.resolve()),
            task=task, split='test', inference_seconds=seconds, inference_device=str(device),
            historical_training_data_root=str(nss.DATA), retrained_on_new90k=False,
            identity=identity, completed_utc=utc())
        write_json(metadata_path, metrics)
        write_json(receipt_path, dict(state='complete', identity=identity,
            predictions_sha256=file_hash(prediction), completed_utc=utc(), inference_seconds=seconds))
        print(f'NSS historical weights / NEW {task} TEST n={metrics["n"]} '
              f'native Top1={100*metrics["top1_native_ind"]:.6f}% mean_cost={metrics["mean_cost"]:.9f}', flush=True)
        return metrics
    except BaseException as error:
        write_json(receipt_path, dict(state='failed', identity=identity, error=str(error), failed_utc=utc()))
        raise


def reuse_lib(root, r79_root, task, frozen, train_sbs):
    """Copy complete original LIB predictions; no loader, model, CUDA or inference."""
    dataset = task + 'LIB'
    source = r79_root / task / (dataset + '_predictions.npz')
    original = read_json(r79_root / task / (dataset + '_metrics.json'))
    receipt = read_json(r79_root / task / (dataset + '_once.json'))
    digest = file_hash(source)
    if (original.get('checkpoint', {}).get('sha256') != frozen['sha256']
            or receipt.get('state') != 'complete' or receipt.get('checkpoint_sha256') != frozen['sha256']
            or receipt.get('predictions_sha256') != digest):
        raise ValueError(f'{dataset}: saved original LIB predictions are not bound to the R79 lock')
    data = read_saved_prediction(source)
    pool = data['pool'].astype(str).tolist()
    metrics = native_metrics(data['scores'], data['pred'], data['costs'], data['winner'], pool, train_sbs)
    if metrics['n'] != LIB_ROWS[task]:
        raise ValueError(f'{dataset}: original LIB population changed')
    run = root / task
    run.mkdir(parents=True, exist_ok=True)
    destination = run / (dataset + '_predictions.npz')
    if destination.exists():
        if file_hash(destination) != digest:
            raise ValueError(f'{dataset}: reference copy changed')
    else:
        shutil.copyfile(source, destination)
    metrics.update(checkpoint=frozen, task=task, split=dataset, data_root='original native LIB',
        inference_reused=True, new_inference=False, retrained_on_new90k=False,
        source_prediction=dict(path=str(source.resolve()), sha256=digest),
        completed_utc=utc())
    write_json(run / (dataset + '_metrics.json'), metrics)
    return metrics


def run_reference(args):
    config, model_lock = require_r84_val_lock(args.model_run, args.data_root)
    # Importing NSS and all TEST reads happen only after the current-model gate.
    nss = importlib.import_module('code.V4.r79_nss_experiment')
    historical = read_json(args.r79_root / 'final_test_lock.json')
    if historical.get('selected_using') != 'VALID only' or historical['original_sources'] != nss.source_hashes():
        raise ValueError('Require unchanged historical R79 validation lock and original NSS source')
    checkpoints = {}
    for task in TASKS:
        record = historical['checkpoints'][task]
        path = (args.r79_root / task / 'best.pt').resolve()
        if file_hash(path) != record['sha256']:
            raise ValueError(f'{task}: historical locked NSS weights changed')
        checkpoints[task] = dict(record, path=str(path))
    protocol = dict(experiment='R84_NSS_historical_reference', data_root=str(args.data_root.resolve()),
        historical_training_data_root=historical['data_root'], weights_description=WEIGHTS_DESCRIPTION,
        retrained_on_new90k=False, expected_test_rows_per_task=TEST_ROWS, winner_rule=WINNER_RULE,
        original_sources=historical['original_sources'], checkpoints=checkpoints,
        current_model_lock=dict(path=str((args.model_run / 'final_test_lock.json').resolve()),
            checkpoint_sha256=model_lock['checkpoint_sha256'], state_hash=model_lock['state_hash'],
            selected_update=model_lock['selected_update']),
        evaluation=dict(batch_size=args.batch_size, augmentation=False,
                        inference_rule='original NSS softmax then torch.topk(1), without output adjustment'),
        sbs={task: config['baselines'][task]['sbs_solver'] for task in TASKS},
        validation_test_history='Retained old holdout members were used in prior research; not fully blind.')
    args.root.mkdir(parents=True, exist_ok=True)
    with (args.root / 'reference.lock').open('a') as claim:
        fcntl.flock(claim, fcntl.LOCK_EX)
        frozen_protocol = bind_reference_protocol(args.root, protocol)
        check_or_write(args.root / 'final_test_lock.json', dict(checkpoints=checkpoints,
            original_sources=historical['original_sources'], data_root=str(args.data_root.resolve()),
            selected_using='Historical R79 VALID only; current R84 unique VAL lock also required',
            current_model_lock=frozen_protocol['current_model_lock'], weights_description=WEIGHTS_DESCRIPTION))
        rows = []
        for task in TASKS:
            rows.append(evaluate_new_test(args, task, nss, checkpoints[task], config))
            rows.append(reuse_lib(args.root, args.r79_root, task, checkpoints[task], protocol['sbs'][task]))
        write_json(args.root / 'summary.json', dict(complete=True, protocol=frozen_protocol, results=rows, completed_utc=utc()))
        with (args.root / 'summary.csv').open('w', newline='') as stream:
            fields = ('task', 'split', 'n', 'top1_native_ind', 'mean_cost', 'actual_regret_pct', 'weights_description')
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(rows)
    return rows


def parser():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--model-run', type=Path, required=True, help='R84 group holding final_test_lock.json and best.pt')
    cli.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    cli.add_argument('--data-root', type=Path, default=DEFAULT_DATA)
    cli.add_argument('--r79-root', type=Path, default=DEFAULT_R79_ROOT)
    cli.add_argument('--device', default='cpu', help='CPU or the pipeline-owned CUDA device; no GPU allocation')
    cli.add_argument('--workers', type=int, default=0)
    cli.add_argument('--batch-size', type=int, choices=(8,), default=8, help='Original NSS evaluation batch')
    return cli


def main(argv=None):
    args = parser().parse_args(argv)
    torch.set_num_threads(2)
    run_reference(args)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

"""Evaluate or reuse the uniquely locked R77B checkpoint on the original LIBs."""

import argparse
import os
import socket
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from code.unified_selector.registry import GLOBAL_SOLVERS
from .nss_comparison import comparable_metrics, infer, read_costs
from .performance_evaluation import decision_metrics
from .r41_frozen_evaluation import attach_geometry
from .r76_experiment import file_hash, fixed_sbs_metrics, read_json, utc, write_json
from .r77_experiment import load_checkpoint
from .train import configure_torch, make_loader


RUN = Path('code/V4/runs/R77_sample_size_batch1024/B_full_seed2')
SHA256 = '31c0c75d66b1bde6ad533909a9c04b0036e80af4c4a5775b01b957ce310e6d1b'
DATASETS = {'TSPLIB': 'TSP', 'CVRPLIB': 'CVRP'}
SCHEMA = {
    'scores': 'float32 [n,k], actual maximin decision scores',
    'logits': 'float32 [n,k], model maximin logits (identical to scores)',
    'pred': 'int64 [n], local candidate slot; score ties use lowest global ID',
    'costs': 'float64 [n,k], unchanged original native costs',
    'winner': 'int64 [n], original native raw_label ind',
    'winner_fp32_argmin': 'int64 [n], historical NSS compatibility labels',
    'pool': 'unicode [k], original candidate names in column order',
    'pool_ids': 'int64 [k], stable global solver IDs',
    'nodes': 'int64 [n], input graph sizes',
    'indices': 'int64 [n], original zero-based instance order',
    'checkpoint_sha256': 'unicode scalar',
    'checkpoint_epoch': 'int64 scalar, one-based locked epoch',
    'checkpoint_update': 'int64 scalar, locked successful update',
}


def verify_checkpoint(run_root):
    checkpoint = run_root / 'best.pt'
    lock = read_json(run_root / 'final_test_lock.json')
    if (lock.get('experiment') != 'R77' or lock.get('arm') != 'B'
            or lock.get('selected_without_test') is not True
            or lock.get('per_problem_assembly') is not False
            or lock.get('epoch') != 28 or lock.get('selected_update') != 34776
            or lock.get('checkpoint_sha256') != SHA256
            or Path(lock['checkpoint']).resolve() != checkpoint.resolve()
            or file_hash(checkpoint) != SHA256):
        raise ValueError('Only the unchanged R77B epoch28/update34776 lock is accepted')
    return checkpoint, lock


def select_device():
    """Bind at most one genuinely idle GPU; never use an occupied GPU."""
    query = ['nvidia-smi', '--query-gpu=index,memory.used,utilization.gpu',
             '--format=csv,noheader,nounits']
    try:
        status = subprocess.run(query, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired) as error:
        return torch.device('cpu'), {'query': query, 'error': str(error)}
    evidence = {'query': query, 'returncode': status.returncode,
                'stdout': status.stdout.strip(), 'stderr': status.stderr.strip()}
    if status.returncode == 0 and torch.cuda.is_available():
        visible = os.environ.get('CUDA_VISIBLE_DEVICES')
        devices = None if visible is None else visible.split(',')
        for line in status.stdout.splitlines():
            index, used, utilization = (part.strip() for part in line.split(','))
            if int(used) >= 500 or int(utilization) != 0:
                continue
            if devices is not None and index not in devices:
                continue
            local_index = int(index) if devices is None else devices.index(index)
            device = torch.device(f'cuda:{local_index}')
            torch.cuda.set_device(device)
            evidence['selected_physical_gpu'] = int(index)
            return device, evidence
    return torch.device('cpu'), evidence


def checked_raw(problem, dataset, config):
    raw = read_costs(problem, dataset)
    ids, pool, costs = raw['pool_ids'], raw['pool'], raw['costs']
    if (ids != sorted(set(ids)) or [GLOBAL_SOLVERS[i] for i in ids] != pool
            or set(pool) != set(config['baselines'][problem]['mean_cost_by_solver'])
            or costs.dtype != np.float64 or np.any(costs <= 0)):
        raise ValueError('Native LIB candidates/costs do not match R77B deployment identities')
    return raw


def checked_predictions(path, raw, lock):
    with np.load(path, allow_pickle=False) as saved:
        data = {key: saved[key] for key in saved.files}
    if set(SCHEMA) - data.keys():
        raise ValueError('Existing predictions lack the documented R77B schema')
    if (str(data['checkpoint_sha256'].item()) != lock['checkpoint_sha256']
            or int(data['checkpoint_epoch']) != lock['epoch']
            or int(data['checkpoint_update']) != lock['selected_update']):
        raise ValueError('Existing LIB predictions belong to another checkpoint lock')
    for key in ('costs', 'winner', 'pool', 'pool_ids'):
        np.testing.assert_array_equal(data[key], raw[key])
    n, k = raw['costs'].shape
    if (data['costs'].dtype != np.float64 or data['scores'].shape != (n, k)
            or not np.isfinite(data['scores']).all()
            or data['nodes'].shape != (n,) or np.any(data['nodes'] <= 0)):
        raise ValueError('Invalid existing prediction shapes, precision, or graph sizes')
    np.testing.assert_array_equal(data['indices'], np.arange(n))
    np.testing.assert_array_equal(data['scores'], data['logits'])
    np.testing.assert_array_equal(data['winner_fp32_argmin'], raw['costs'].astype(np.float32).argmin(1))
    _, pred = decision_metrics(data['scores'], raw)
    np.testing.assert_array_equal(data['pred'], pred)
    return data


def summarize(data, raw, baseline):
    native, pred = decision_metrics(data['scores'], raw)
    native.update(correct=int(np.count_nonzero(pred == raw['winner'])), sbs_selection_split='LIB posthoc')
    nss, nss_pred = comparable_metrics(data['scores'], raw)
    nss.update(correct=int(np.count_nonzero(pred == data['winner_fp32_argmin'])),
               sbs_selection_split='LIB posthoc')
    train_sbs, train_pred = fixed_sbs_metrics(data['scores'], raw, baseline)
    np.testing.assert_array_equal(pred, nss_pred)
    np.testing.assert_array_equal(pred, train_pred)
    return dict(native=native, nss_comparable=nss, train_sbs_reference=train_sbs,
                winner_fp32_disagreements=int(np.count_nonzero(raw['winner'] != data['winner_fp32_argmin'])))


@torch.inference_mode()
def evaluate(args):
    started = time.monotonic()
    checkpoint_path, lock = verify_checkpoint(args.run_root)
    config = read_json(args.run_root / 'protocol.json')
    if (config.get('experiment') != 'R77' or config.get('arm') != 'B'
            or config.get('solver_names') != GLOBAL_SOLVERS):
        raise ValueError('The training protocol is not the full R77B model')
    problem = DATASETS[args.dataset]
    raw = checked_raw(problem, args.dataset, config)
    result_path = args.root / f'{args.dataset}_results.json'
    protocol_path = args.root / f'{args.dataset}_protocol.json'
    path = args.root / 'predictions' / f'{args.dataset}.npz'
    if path.exists():
        data = checked_predictions(path, raw, lock)
        protocol = read_json(protocol_path)
        if (protocol['checkpoint_sha256'] != lock['checkpoint_sha256']
                or protocol['dataset'] != args.dataset or protocol['candidate_pool'] != raw['pool']
                or protocol['pool_ids'] != raw['pool_ids']):
            raise ValueError('Existing prediction protocol does not match the R77B LIB lock')
        if result_path.exists():
            existing = read_json(result_path)
            if existing['protocol'] != protocol:
                raise ValueError('Existing results and prediction protocol disagree')
            print(f'[R77B {args.dataset}] Reused checked predictions/results; no inference', flush=True)
            return existing
        timing = dict(inference_reused=True, recompute_seconds=time.monotonic() - started,
                      inference_seconds=None, seconds=None)
        device_name = protocol['device']
    else:
        if result_path.exists() or protocol_path.exists():
            raise ValueError('Incomplete LIB artifacts exist; do not silently replace them')
        configure_torch()
        torch.set_num_threads(2)
        device, availability = select_device()
        model, checkpoint = load_checkpoint(checkpoint_path, device)
        # R77 stores zero-based checkpoint epochs; the final lock is one-based.
        if (checkpoint['args'] != config or checkpoint['epoch'] + 1 != lock['epoch']
                or checkpoint['successful_updates'] != lock['selected_update']):
            raise ValueError('Checkpoint metadata differs from the locked training protocol')
        geometry = model.instance_encoder.geometry_residual
        for key in ('mean', 'std'):
            value = getattr(geometry, key)
            torch.testing.assert_close(value, value.new_tensor(config['geometry_stats'][key]), rtol=0, atol=0)
        if not model.params.get('ignore_coord_dist'):
            raise ValueError('R77B must retain its locked ignore_coord_dist input policy')
        model.eval().requires_grad_(False)
        device_name = torch.cuda.get_device_name(device) if device.type == 'cuda' else 'cpu'
        directory = Path(raw['directory'])
        protocol = dict(experiment='R77', model_version='R77B', dataset=args.dataset, problem=problem,
            checkpoint=str(checkpoint_path), checkpoint_sha256=lock['checkpoint_sha256'],
            checkpoint_epoch=lock['epoch'], checkpoint_update=lock['selected_update'],
            selected_without_test=True, selection_rule=lock['rule'], per_problem_assembly=False,
            candidate_pool=raw['pool'], pool_ids=raw['pool_ids'], n=len(raw['winner']),
            data_version='original native LIB / historical NSS comparison, not R77 new-data TEST',
            sources={name: dict(path=str(directory / name), sha256=file_hash(directory / name))
                     for name in ('dataset.pkl', 'raw_label.pkl')},
            training_data_root=config['data_root'], normalization='R77B TRAIN buffers from best.pt; no LIB refitting',
            geometry_mean=geometry.mean.cpu().tolist(), geometry_std=geometry.std.cpu().tolist(),
            decision='single forward, greedy maximin-logit argmax; exact score ties use ascending stable global ID',
            top1_primary='native FP64 raw_label ind', top1_nss_comparison='FP32 cost argmin',
            topk_definition='native winner contained in score top-k; stable global-ID tie rule',
            sbs='LIB posthoc minimum mean FP64 cost, historical NSS comparison only',
            auxiliary_sbs='R77B FULL TRAIN-selected fixed solver evaluated on native LIB costs',
            train_sbs=config['baselines'][problem],
            metric_formulas=dict(vs_sbs_pct='100 * (mean_cost / sbs_cost - 1)',
                vs_oracle_pct='100 * (mean_cost / oracle_cost - 1)',
                actual_regret_pct='mean_i(100 * (selected_cost_i - min_j cost_ij) / min_j cost_ij)'),
            batch_size=args.batch_size, precision='FP32 model, FP64 costs and metrics; no autocast',
            augmentation=False, solver_execution=False, training=False, checkpoint_reselection=False,
            ignore_coord_dist=True, device=device_name, availability=availability,
            prediction_file=str(path), prediction_schema=SCHEMA, host=socket.gethostname(), created_utc=utc())
        loader = make_loader(problem, args.dataset, args.batch_size, 0, coord_augment=0,
                             shuffle=False, cache_device=device)
        if loader.size != len(raw['winner']):
            raise ValueError('LIB input and native label row counts differ')
        np.testing.assert_array_equal(loader.batch['pool_ids'].cpu().numpy(), raw['pool_ids'])
        np.testing.assert_array_equal(loader.batch['ind'].cpu().numpy(), raw['winner'])
        np.testing.assert_array_equal(loader.batch['costs'].cpu().numpy(), raw['costs'].astype(np.float32))
        write_json(protocol_path, protocol)
        inference_started = time.monotonic()
        attach_geometry(loader, device)
        infer(model, loader, args.batch_size, raw, path)
        inference_seconds = time.monotonic() - inference_started
        with np.load(path, allow_pickle=False) as saved:
            data = {key: saved[key] for key in saved.files}
        data.update(winner_fp32_argmin=raw['costs'].astype(np.float32).argmin(1),
                    checkpoint_sha256=np.array(lock['checkpoint_sha256']),
                    checkpoint_epoch=np.array(lock['epoch'], dtype=np.int64),
                    checkpoint_update=np.array(lock['selected_update'], dtype=np.int64))
        np.savez_compressed(path, **data)
        checked_predictions(path, raw, lock)
        timing = dict(inference_reused=False, inference_seconds=inference_seconds,
                      seconds=time.monotonic() - started)
    result = dict(**summarize(data, raw, config['baselines'][problem]), protocol=protocol,
                  host=socket.gethostname(), device=device_name, **timing, completed_utc=utc())
    write_json(result_path, result)
    native = result['native']
    print(f"[R77B {args.dataset}] n={native['n']} native Top1={100*native['top1']:.4f}% "
          f"NSS Top1={100*result['nss_comparable']['top1']:.4f}% mean_cost={native['mean_cost']:.9f}", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=tuple(DATASETS), required=True)
    parser.add_argument('--run-root', type=Path, default=RUN)
    parser.add_argument('--batch-size', type=int, default=2)
    args = parser.parse_args()
    if not 1 <= args.batch_size <= 2:
        parser.error('LIB inference uses a fixed small batch of 1 or 2 for large graphs')
    args.run_root = args.run_root.resolve()
    args.root = args.run_root / 'lib_evaluation'
    evaluate(args)


if __name__ == '__main__':
    main()

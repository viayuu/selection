"""Replay one validation-selected original ReLD recipe, retaining every start cost."""

import argparse
import csv
import json
import os
from pathlib import Path
import pickle
import subprocess
import time

import numpy as np
import torch

from code.unified_selector.registry import DATA_ROOT
from .multitask_probe import dump, file_hash, state_hash
from .performance_targets import read_raw_costs
from .r41_label_stability import proportional_indices
from .r48_common import PAIR, write_csv
from .r50_solver_encoder import RELD_ROOT, source_provenance
from .r51_probe import build_solvers
from .r52_error_budget import ROOT
from .tsp_learnability import size_boundaries


EXPORT_ROOT = Path('/public/home/shiys/easynco_v3_bridge/exports/reld_nss_like_no_aug_v1')
FIELDS = ('depot_xy', 'node_xy', 'node_demand', 'route_limit', 'service_time', 'tw_start', 'tw_end')
ATOL, RTOL = 5e-5, 2e-6


def load_validation(problem):
    with (DATA_ROOT / f'{problem}val' / 'dataset.pkl').open('rb') as stream:
        instances = pickle.load(stream)
    return instances, np.array([len(item['node_xy']) for item in instances], dtype=np.int64)


def prepare(root=ROOT):
    with (root / 'ranked_problem_pairs.csv').open() as stream:
        ranked = list(csv.DictReader(stream))
    selected = ranked[0]
    if (selected['solver_a'], selected['solver_b']) != tuple(PAIR):
        raise ValueError('Highest-regret pair is not the recoverable ReLD pair; inspect its original recipe first')
    problem = selected['problem']
    if problem != 'OVRPBLTW':
        raise ValueError('Review the selected constraint-specific input contract before replay')
    instances, sizes = load_validation(problem)
    raw = read_raw_costs(problem, 'val')
    cuts = size_boundaries(sizes)
    indices, groups, quotas = proportional_indices(sizes, cuts, min(1000, len(sizes)), 2)
    precheck, _, precheck_quotas = proportional_indices(sizes, cuts, 8, 2)
    provenance = source_provenance()
    provenance['source_files'][str(RELD_ROOT / f'envs/{problem}Env.py')] = file_hash(
        RELD_ROOT / f'envs/{problem}Env.py')
    ancestors = {}
    for name in PAIR:
        path = EXPORT_ROOT / f'{problem}val/results/result_{name}.txt'
        exported = np.loadtxt(path, delimiter=',', dtype=np.float64)
        if not np.array_equal(exported[:, 0], np.arange(len(instances))):
            raise ValueError('Original solver export indices do not match validation')
        column = raw['pool'].index(name)
        if not np.array_equal(exported[:, 1], raw['costs'][:, column]):
            raise ValueError('Raw labels do not equal the original no-augmentation export')
        ancestors[name] = dict(path=str(path), sha256=file_hash(path),
            equal_to_raw_label_column=True, max_abs_difference=0.)
    plan = dict(seed=2, problem=problem, split='val', solvers=list(PAIR),
        selection=selected, selection_rank=1,
        selection_rule='Highest validation macro-regret problem/pair; no fallback or default solver recipe',
        indices=indices.tolist(), customer_counts=sizes[indices].tolist(),
        sample_rule='Size-stratified, no error/winner conditioning; all 1000 original validation instances',
        size_boundaries=cuts.tolist(), size_quotas=quotas.tolist(),
        precheck_indices=precheck.tolist(), precheck_quotas=precheck_quotas.tolist(),
        dataset_path=str(DATA_ROOT / f'{problem}val/dataset.pkl'), dataset_sha256=raw['data_hash'],
        raw_label_sha256=raw['label_hash'], original_exports=ancestors,
        provenance=provenance,
        recipe=dict(input_fields=list(FIELDS), dtype='FP32', model_eval=True,
            eval_type='argmax', augmentation=1, sample_size=1, maximum_same_size_batch=128,
            declared_pomo_size='number of customers, as in original bridge',
            effective_pomo_size='native OVRPBLTWEnv: min(floor(0.8*N), declared_pomo_size)',
            start_nodes='positive-demand customer IDs in original node order, native cap; no new starts',
            constraints=[1, 1, 1, 1, 1], capacity=1., speed=1., depot_time_window=[0., 3.],
            open_route_objective='sum segments whose destination is not depot; no return-to-depot cost',
            normalization='No new normalization; original bridge collate_problem_batch'),
        history_comparison_tolerance=dict(atol=ATOL, rtol=RTOL),
        precision='Native trajectory rewards are FP32, stored as FP64 without claiming recovered precision; historical labels are original FP64',
        provenance_limitations=[
            'Original label exports, bridge entry, explicit decoding parameters and R48-pinned weights are available.',
            'Historical source/environment revision and original complete CLI are not archived; no default replacement is used.',
            'R48 runtime validation covered OVRPTW only; this audit newly checks OVRPBLTW by historical scalar replay.',
            'Only the selected problem and pair are rerun. Costs and winners remain unchanged.'])
    path = root / 'audit_plan.json'
    if path.exists() and json.loads(path.read_text()) != plan:
        raise ValueError('The locked R52 audit plan changed')
    dump(path, plan)
    print(json.dumps(dict(problem=problem, pair=list(PAIR), instances=len(indices),
        precheck=precheck.tolist(), rank=1, size_range=[int(sizes.min()), int(sizes.max())]), indent=2), flush=True)
    return plan


def native_batch(module, instances, rows, problem, device):
    batch, size = module.collate_problem_batch(problem + 'val', instances, list(map(int, rows)))
    if len(batch) != len(FIELDS) or len({len(instances[int(i)]['node_xy']) for i in rows}) != 1:
        raise ValueError('Native selected-problem inputs require all seven fields, no padding')
    for field, actual in zip(FIELDS, batch):
        expected = torch.stack([torch.as_tensor(instances[int(i)][field], dtype=torch.float32) for i in rows])
        if expected.shape != actual.shape and expected.numel() == actual.numel():
            expected = expected.reshape(actual.shape)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    if any(float(instances[int(i)]['capacity']) != 1. for i in rows):
        raise ValueError('Unexpected capacity convention')
    return tuple(value.to(device) for value in batch), size


@torch.no_grad()
def full_start_costs(model, module, batch, size, problem):
    env = module.get_env_class(problem)(problem_size=size, pomo_size=size, device=batch[0].device)
    env.load_problems(len(batch[0]), problems=batch, aug_factor=1)
    reset, _, _ = env.reset()
    if not torch.equal(reset.prob_emb, torch.ones(1, 5, device=batch[0].device)):
        raise ValueError('Native constraint flags differ from OVRPBLTW')
    starts = env.START_NODE.clone()
    if starts.shape[1] != int(size * .8) or not (batch[2].gather(1, starts - 1) > 0).all():
        raise ValueError('Native backhaul start restriction changed')
    model.pre_forward(reset)
    state, reward, done = env.pre_step()
    steps = 0
    while not done:
        selected, _ = model(state)
        if steps == 0 and not (selected == 0).all():
            raise ValueError('Native depot initialization changed')
        if steps == 1 and not torch.equal(selected, starts):
            raise ValueError('Native POMO initialization changed')
        state, reward, done = env.step(selected)
        steps += 1
        if steps > 2 * size + 5:
            raise ValueError('Native trajectory safety bound exceeded')
    costs = -reward
    if not torch.isfinite(costs).all() or (costs <= 0).any() or not env.finished.all():
        raise ValueError('Nonfinite or incomplete native trajectories')
    return costs.double().cpu().numpy(), starts.cpu().numpy(), steps


def replay(root=ROOT):
    plan = json.loads((root / 'audit_plan.json').read_text())
    if torch.cuda.device_count() != 1 or '3090' not in torch.cuda.get_device_name(0):
        raise ValueError('R52 must be bound to the idle RTX3090 only')
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    problem = plan['problem']
    instances, sizes = load_validation(problem)
    raw = read_raw_costs(problem, 'val')
    if raw['data_hash'] != plan['dataset_sha256'] or raw['label_hash'] != plan['raw_label_sha256']:
        raise ValueError('Locked validation inputs or labels changed')
    models, module = build_solvers(plan['provenance'], 'cuda:0')
    hashes = [state_hash(model.state_dict()) for model in models]
    pair_columns = [raw['pool'].index(name) for name in PAIR]
    historical = raw['costs'][:, pair_columns]
    check_rows, batch_records = [], []
    start_time = time.perf_counter()

    def solve(rows, stage):
        batch, size = native_batch(module, instances, rows, problem, 'cuda:0')
        torch.cuda.synchronize()
        begin = time.perf_counter()
        values, start_ids, steps = [], None, []
        for model in models:
            costs, ids, step_count = full_start_costs(model, module, batch, size, problem)
            if start_ids is not None and not np.array_equal(start_ids, ids):
                raise ValueError('Solvers did not use identical original start IDs')
            values.append(costs)
            start_ids = ids
            steps.append(step_count)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - begin
        values = np.stack(values, axis=1)
        best = values.min(-1)
        expected = historical[rows]
        matches = np.isclose(best, expected, atol=ATOL, rtol=RTOL)
        batch_records.append(dict(stage=stage, customers=int(size), rows=list(map(int, rows)),
            effective_starts=values.shape[-1], native_steps=steps, seconds=elapsed,
            input_sha256=state_hash(batch), max_abs_history_difference=float(np.abs(best - expected).max())))
        for position, index in enumerate(rows):
            for solver, name in enumerate(PAIR):
                check_rows.append(dict(stage=stage, index=int(index), customers=int(size), solver=name,
                    starts=values.shape[-1], historical_cost=float(expected[position, solver]),
                    replay_min_cost=float(best[position, solver]),
                    absolute_difference=float(abs(best[position, solver] - expected[position, solver])),
                    reproduced=bool(matches[position, solver])))
        write_csv(root / 'historical_cost_replay.csv', check_rows)
        dump(root / 'replay_batches.json', batch_records)
        if not matches.all():
            raise ValueError(f'{stage}: historical scalar not reproduced; preserve evidence, do not continue')
        return values, start_ids

    # An eight-instance check precedes the full selected population, not 20 repeated solves.
    precheck = np.array(plan['precheck_indices'], dtype=np.int64)
    for size in np.unique(sizes[precheck]):
        rows = precheck[sizes[precheck] == size]
        solve(rows, 'precheck')
    print('[R52] All eight precheck instances reproduce both historical costs.', flush=True)
    indices = np.array(plan['indices'], dtype=np.int64)
    maximum = int(max(sizes[indices]))
    costs = np.full((len(indices), 2, maximum), np.nan, dtype=np.float64)
    valid = np.zeros((len(indices), maximum), dtype=bool)
    start_ids = np.full((len(indices), maximum), -1, dtype=np.int64)
    positions = {int(index): i for i, index in enumerate(indices)}
    completed = 0
    for size in np.unique(sizes[indices]):
        selected = indices[sizes[indices] == size]
        for begin in range(0, len(selected), 128):
            rows = selected[begin:begin + 128]
            values, ids = solve(rows, 'full')
            slots = np.array([positions[int(i)] for i in rows])
            length = ids.shape[1]
            costs[slots, :, :length] = values
            start_ids[slots, :length] = ids
            valid[slots, :length] = True
            completed += len(rows)
            print(f'[R52] {completed}/{len(indices)} instances; N={size}, starts={length}, '
                  f'elapsed={time.perf_counter() - start_time:.1f}s', flush=True)
    if hashes != [state_hash(model.state_dict()) for model in models]:
        raise ValueError('Frozen solver parameters changed')
    if not valid.any(1).all() or not np.isfinite(costs[np.broadcast_to(valid[:, None], costs.shape)]).all():
        raise ValueError('Incomplete start-cost coverage')
    np.savez_compressed(root / 'start_costs.npz', indices=indices, customer_counts=sizes[indices],
        costs=costs, valid_start_mask=valid, start_ids=start_ids, solver_names=np.array(PAIR),
        historical_costs=historical[indices], problem=np.array(problem), split=np.array('val'))
    dump(root / 'replay_receipt.json', dict(problem=problem, instances=len(indices),
        completed_solver_runs=2 * (len(indices) + len(precheck)),
        total_seconds=time.perf_counter() - start_time,
        full_replay_seconds=sum(r['seconds'] for r in batch_records if r['stage'] == 'full'),
        precheck_seconds=sum(r['seconds'] for r in batch_records if r['stage'] == 'precheck'),
        max_abs_history_difference=max(r['absolute_difference'] for r in check_rows),
        all_history_costs_reproduced=True, solver_state_hashes_before_and_after=hashes,
        device=torch.cuda.get_device_name(0), gpu_uuid=os.environ.get('CUDA_VISIBLE_DEVICES'),
        peak_allocated_gpu_mib=torch.cuda.max_memory_allocated() / 2 ** 20,
        start_costs_sha256=file_hash(root / 'start_costs.npz'),
        git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        no_new_labels=True, no_selector_training=True, test_read=False,
        source_files={str(Path(__file__)): file_hash(__file__)}))
    print('[R52] Complete trajectories saved; all scalar replay checks passed.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['prepare', 'replay'], required=True)
    parser.add_argument('--output', type=Path, default=ROOT)
    args = parser.parse_args()
    (prepare if args.stage == 'prepare' else replay)(args.output)

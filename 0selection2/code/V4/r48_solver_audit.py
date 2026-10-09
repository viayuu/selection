"""Capture original RELD routes without changing its inference implementation."""

import argparse
import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from .multitask_probe import dump, file_hash, state_hash
from .r41_label_stability import OVRPTW_FIELDS, relocated_reld_module
from .r48_common import (ROOT, PAIR, VARIANTS, check_open_tw_route, close_cost,
                         load_split, original_route, permutation, plan_order,
                         prepare_plan, write_csv)


def worker(args):
    plan = json.loads((args.root/'audit_plan.json').read_text())
    entry = plan['solvers'][args.solver]
    if file_hash(entry['original_entry']) != entry['source_sha256'] or file_hash(
            entry['checkpoint']['path']) != entry['checkpoint']['sha256']:
        raise ValueError('R41 pinned source or solver weights changed')
    module = relocated_reld_module(Path(entry['original_entry']))
    key = 'reld_moel' if args.solver == 'RELD_MOEL' else 'reld_mtl'
    torch.set_num_threads(1)
    torch.manual_seed(2)
    torch.cuda.manual_seed_all(2)
    np.random.seed(2)
    device = torch.device('cuda:0')
    model, params = module.instantiate_reld_model(key, device)
    if params != entry['runtime_params_contract']:
        raise ValueError('Runtime config differs from the R41 contract')
    instances, raw, costs, _ = load_split(args.split)
    sample = plan['splits'][args.split]
    if raw['data_hash'] != sample['dataset_hash'] or raw['label_hash'] != sample['label_hash']:
        raise ValueError('Solver worker inputs changed')
    indices = sample['precheck_indices'] if args.precheck else sample['indices']
    selected_instances, orders = [], []
    for index in indices:
        order = plan_order(plan, args.split, index, args.variant)
        orders.append(order)
        selected_instances.append(permutation(instances[index], order))
    env_class = module.get_env_class('OVRPTW')
    observed = {}

    class CapturedEnv(env_class):
        def load_problems(self, batch_size, problems=None, aug_factor=1):
            super().load_problems(batch_size, problems, aug_factor)
            if self.speed != 1. or self.problem != 'OVRPTW' or aug_factor != 1:
                raise ValueError('Unexpected travel-time/constraint/augmentation semantics')
            torch.testing.assert_close(self.reset_state.prob_emb.cpu(), torch.tensor([[1., 1., 0., 0., 1.]]))

        def step(self, chosen):
            state, reward, done = super().step(chosen)
            if done:
                best = reward.argmax(1)
                batch_index = torch.arange(len(best), device=best.device)
                observed['routes'] = self.selected_node_list[batch_index, best].detach().cpu().tolist()
                observed['pomo'] = best.detach().cpu().tolist()
            return state, reward, done

    groups = defaultdict(list)
    for slot, instance in enumerate(selected_instances):
        groups[len(instance['node_xy'])].append(slot)
    rows, routes, witnesses = [], [], []
    started = time.monotonic()
    for scale, slots in sorted(groups.items()):
        for offset in range(0, len(slots), 128):
            part = slots[offset:offset+128]
            batch, resolved = module.collate_problem_batch('OVRPTW'+args.split, selected_instances, part)
            if resolved != scale:
                raise ValueError('Unexpected solver scale')
            for value, field in zip(batch, OVRPTW_FIELDS):
                expected = torch.stack([torch.as_tensor(selected_instances[i][field], dtype=torch.float32) for i in part])
                torch.testing.assert_close(value, expected, rtol=0, atol=0)
            values = module._run_no_aug_batch(model, CapturedEnv, batch, scale, device).numpy().astype(np.float64)
            if len(observed['routes']) != len(part):
                raise ValueError('No captured output route')
            witnesses.append(dict(original_indices=[int(indices[i]) for i in part], fields=list(OVRPTW_FIELDS),
                                  actual_input_sha256=state_hash(batch), customer_count=scale, exact_input_match=True))
            for slot, scalar, route, pomo in zip(part, values, observed['routes'], observed['pomo']):
                index = indices[slot]
                mapped = original_route(route, orders[slot])
                checked = check_open_tw_route(instances[index], mapped)
                runtime = check_open_tw_route(instances[index], mapped, runtime_float32=True)
                old = float(costs[index, PAIR.index(args.solver)])
                row = dict(split=args.split, instance_index=index, solver=args.solver,
                           variant=args.variant, historical_cost=old, rerun_cost=float(scalar),
                           **checked, runtime_recomputed_cost=runtime['recomputed_cost'],
                           rerun_minus_historical=float(scalar-old),
                           recomputed_minus_rerun=float(checked['recomputed_cost']-scalar),
                           cost_matches_recomputation=close_cost(scalar, runtime['recomputed_cost']),
                           historical_matches=(close_cost(scalar, old) if args.variant.startswith('original') else ''),
                           pid=os.getpid(), best_pomo_index=pomo, input_matches_original_fields=True)
                rows.append(row)
                routes.append(dict(split=args.split, instance_index=index, solver=args.solver,
                                   variant=args.variant, customer_permutation=orders[slot].tolist(),
                                   route_solver_ids=route, route_original_ids=mapped, best_pomo_index=pomo,
                                   scalar_cost=float(scalar), independent_check=checked))
    prefix = 'precheck' if args.precheck else 'run'
    directory = args.root/'solver_runs'/args.split/args.solver
    directory.mkdir(parents=True, exist_ok=True)
    output = directory/(prefix+'_'+args.variant)
    write_csv(output.with_suffix('.csv'), rows)
    with output.with_suffix('.routes.jsonl').open('w') as stream:
        for route in routes:
            stream.write(json.dumps(route)+'\n')
    dump(output.with_suffix('.json'), dict(solver=args.solver, split=args.split, variant=args.variant,
        pid=os.getpid(), independent_process=True, seed=2, seconds=time.monotonic()-started,
        model_eval=True, decode_budget=entry['decode_budget'], actual_params=params,
        checkpoint=entry['checkpoint'], source_sha256=entry['source_sha256'], input_witness=witnesses,
        instrumentation='Subclass only records completed routes; original _run_no_aug_batch is called unchanged',
        torch=torch.__version__, gpu=torch.cuda.get_device_name(0)))
    print(f'[R48 solver] {args.split}/{args.solver}/{args.variant}: {len(rows)} routes in {time.monotonic()-started:.1f}s', flush=True)


def execute(root):
    prepare_plan(root)
    commands = [(True, 'train', name, 'original_0') for name in PAIR]
    commands += [(False, split, name, variant) for split in ('train', 'val') for variant in VARIANTS for name in PAIR]
    for precheck, split, name, variant in commands:
        output = root/'solver_runs'/split/name/(('precheck_' if precheck else 'run_')+variant+'.csv')
        if output.exists() and output.with_suffix('.json').exists():
            continue
        command = [sys.executable, '-u', '-m', 'code.V4.r48_solver_audit', '--stage', 'worker',
                   '--root', str(root), '--split', split, '--solver', name, '--variant', variant]
        if precheck:
            command.append('--precheck')
        print('[R48 launch] ' + ' '.join(command), flush=True)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.with_suffix('.log').open('w') as stream:
            subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True)
        from .r48_common import read_csv
        rows = read_csv(output)
        errors = [r for r in rows if r['feasible'] != 'True' or r['cost_matches_recomputation'] != 'True'
                  or (variant.startswith('original') and r['historical_matches'] != 'True')]
        if errors:
            dump(root/'blocking_anomalies.json', dict(stage='solver_audit', rows=errors,
                action='Stop binary training; preserve evidence and locate mismatch first'))
            collect(root, partial=True)
            raise RuntimeError(f'Solver audit found {len(errors)} anomalies; inspect blocking_anomalies.json')
    return collect(root)


def collect(root, partial=False):
    from .r48_common import read_csv
    rows, pids = [], []
    for split in ('train', 'val'):
        for name in PAIR:
            for variant in VARIANTS:
                path = root/'solver_runs'/split/name/('run_'+variant+'.csv')
                if not path.exists():
                    if partial:
                        continue
                    raise ValueError('Incomplete solver audit: '+str(path))
                rows.extend(read_csv(path))
                pids.append(json.loads(path.with_suffix('.json').read_text())['pid'])
    write_csv(root/'solver_audit.csv', rows)
    failures = [r for r in rows if r['feasible'] != 'True' or r['cost_matches_recomputation'] != 'True'
                or (r['variant'].startswith('original') and r['historical_matches'] != 'True')]
    report = dict(rows=len(rows), expected_rows=128*6*2, all_routes_captured=all(r['route_available']=='True' for r in rows),
        independent_processes=len(pids), distinct_pids=len(set(pids)), anomaly_count=len(failures),
        complete=not partial and len(rows)==128*6*2, train_allowed=not partial and not failures,
        max_abs_recomputed_error=max((abs(float(r['recomputed_minus_rerun'])) for r in rows), default=None),
        max_abs_historical_error=max((abs(float(r['rerun_minus_historical'])) for r in rows
                                     if r['variant'].startswith('original')), default=None))
    dump(root/'solver_audit_summary.json', report)
    print('[R48 audit] '+json.dumps(report), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--stage', choices=('all', 'worker', 'collect'), default='all')
    parser.add_argument('--solver', choices=PAIR)
    parser.add_argument('--split', choices=('train', 'val'))
    parser.add_argument('--variant', choices=VARIANTS)
    parser.add_argument('--precheck', action='store_true')
    args = parser.parse_args()
    worker(args) if args.stage == 'worker' else collect(args.root) if args.stage == 'collect' else execute(args.root)


if __name__ == '__main__':
    main()

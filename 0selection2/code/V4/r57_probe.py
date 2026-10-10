"""Bounded, read-only runtime probes of recovered recipes on the idle RTX3090."""

import argparse
import copy
import importlib.util
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback

import numpy as np
import torch

from code.unified_selector.registry import GLOBAL_SOLVERS, POOLS, PROBLEMS
from .multitask_probe import dump, file_hash
from .performance_targets import read_raw_costs
from .r48_common import PAIR, write_csv
from .r53_data import BASELINE, MVRP
from .r54_routes import native_batch, solve
from .r57_signal import ROOT, load_instances, true_size


HOME = Path('/public/home/shiys')
BRIDGE = HOME / 'easynco_v3_bridge/scripts'
RECIPE_SOLVERS = ('GLOP', 'MATNET', 'MATPOENET', 'ICAM', 'ICAM_ATSP',
                  'UNICO_MatPOENet', 'RELD_CVRP', 'RELD_PAIR')
VARIANTS = ('original_0', 'original_1', 'preserve_roles', 'reassign_roles')


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def permutation(problem, instance, variant, index, solver=None):
    n = true_size(problem, instance)
    order = np.arange(n)
    if variant.startswith('original') or variant.startswith('rng'):
        return order
    rng = np.random.default_rng(np.random.SeedSequence([2, PROBLEMS.index(problem), int(index)]))
    if variant == 'preserve_roles':
        if problem == 'TSP':
            order[1:-1] = rng.permutation(order[1:-1])
        elif problem == 'CVRP':
            order = rng.permutation(order)
        elif problem in MVRP and 'B' in problem:
            demand = np.asarray(instance['node_demand'])
            positive, nonpositive = np.flatnonzero(demand > 0), np.flatnonzero(demand <= 0)
            count = int(n * .8)
            order[positive[:count]] = rng.permutation(positive[:count])
            order[positive[count:]] = rng.permutation(positive[count:])
            order[nonpositive] = rng.permutation(nonpositive)
        else:
            order = rng.permutation(order)
    else:
        order = rng.permutation(order)
        if n > 1 and np.array_equal(order, np.arange(n)):
            order = np.roll(order, 1)
    return order


def permute_instance(problem, instance, order):
    if problem == 'TSP':
        return instance[:, order, :].clone()
    if problem == 'ATSP':
        return instance[:, order, :][:, :, order].clone() if instance.ndim == 3 else instance[order][:, order].clone()
    result = copy.deepcopy(instance)
    if problem == 'CVRP':
        for name in ('loc', 'demand'):
            result[name] = instance[name][:, order].clone() if torch.is_tensor(instance[name]) else np.asarray(instance[name])[:, order].copy()
    else:
        for name in ('node_xy', 'node_demand', 'service_time', 'tw_start', 'tw_end'):
            if name in instance:
                result[name] = np.asarray(instance[name])[order].copy()
    return result


def physical_starts(problem, instance, order, solver):
    if problem == 'TSP':
        return [int(order[0])] if solver in ('BQ', 'LEHD') else None
    if solver == 'RELD_CVRP':
        # Single-task ReLD branches on learned top-k probabilities, not IDs 1..100.
        return None
    if solver in PAIR:
        if 'B' in problem:
            demand = np.asarray(instance['node_demand'])[order]
            return sorted((order[np.flatnonzero(demand > 0)[:int(len(order) * .8)]] + 1).tolist())
        return sorted((order + 1).tolist())
    if problem == 'ATSP' and solver != 'GLOP':
        return sorted(order.tolist())
    return None


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def torchrl_compat():
    import torchrl.data.tensor_specs as specs
    aliases = dict(CompositeSpec='Composite', BoundedTensorSpec='Bounded',
        UnboundedContinuousTensorSpec='Unbounded', UnboundedDiscreteTensorSpec='Unbounded',
        DiscreteTensorSpec='Categorical', OneHotDiscreteTensorSpec='OneHot',
        BinaryDiscreteTensorSpec='Binary', MultiDiscreteTensorSpec='MultiCategorical',
        MultiOneHotDiscreteTensorSpec='MultiOneHot')
    for old, new in aliases.items():
        if not hasattr(specs, old) and hasattr(specs, new):
            setattr(specs, old, getattr(specs, new))


def archived_config(solver):
    from .r57_audit import EVIDENCE
    evidence = json.loads(EVIDENCE.read_text())
    selected = []
    for item in evidence['archived_configs']:
        path = item['file']
        if solver == 'ICAM':
            match = '/icam_cvrp/CVRPtrain/eval.log' in path
        else:
            match = f'/label_v3/{solver.lower()}_atsp/scale_50/.hydra/config.yaml' in path
        if match:
            selected.append(item)
    if len(selected) != 1:
        raise ValueError(f'No unique archived recipe for {solver}: {len(selected)}')
    item = selected[0]
    if file_hash(item['file']) != item['sha256']:
        raise ValueError('Archived configuration changed')
    return item


class EasyReplay:
    def __init__(self, solver):
        sys.path.insert(0, str(HOME))
        torchrl_compat()
        from hydra.utils import instantiate
        from omegaconf import OmegaConf
        from EasyNCO.utils.utils import load_model
        item = archived_config(solver)
        self.cfg = OmegaConf.create(item['config'])
        self.solver = solver
        self.instantiate = instantiate
        self.policy = instantiate(self.cfg.settings.model)
        weight = HOME / 'EasyNCO' / self.cfg.settings.test_loader.model_dirpath / self.cfg.settings.test_loader.model_filename
        self.policy = load_model(self.policy, str(weight), device=0, model_name=self.cfg.model)
        self.policy = self.policy.cuda().eval()
        self.init = instantiate(self.cfg.settings.initialization, policy=self.policy)
        self.evidence = dict(archived_config=item['file'], config_sha256=item['sha256'],
                             checkpoint=str(weight), checkpoint_sha256=file_hash(weight),
                             scope='archived recipe with current source; historical source/weight-byte binding remains open')

    @torch.no_grad()
    def run(self, instances, n, seed):
        from omegaconf import OmegaConf
        settings = OmegaConf.to_container(self.cfg.settings.env, resolve=True)
        settings.update(problem_size=n, pomo_size=n, device='cuda:0', seed=seed)
        if self.solver == 'ICAM':
            xy = torch.cat((torch.stack([x['depot'].squeeze(0) for x in instances]),
                            torch.stack([x['loc'].squeeze(0) for x in instances])), 1)
            demand = torch.stack([x['demand'].squeeze(0) for x in instances])
            demand = torch.cat((torch.zeros(len(instances), 1), demand), 1)
            batch = torch.cat((xy, demand[..., None]), -1).cuda()
        else:
            batch = torch.cat([x if x.ndim == 3 else x[None] for x in instances]).float().cuda()
        with torch.device('cuda:0'):
            env = self.instantiate(settings)
            if self.solver == 'GLOP':
                state, out = self.init.run(env, batch, strategy='greedy', phase='eval')
                return np.asarray([float(out['no_aug_score'])]), state['selected_node_list'].cpu().numpy()
            env.load_problems(batch, batch_size=len(instances))
            state, out = self.init.play_episode(env, self.cfg.settings.module.decoder_strategy)
            reward = out['reward'].reshape(env.aug_factor, len(instances), env.pomo_size)[0]
            cost, best = (-reward).min(1)
            routes = getattr(env, 'selected_node_list', None)
            if routes is not None:
                routes = routes[:len(instances)][torch.arange(len(instances), device=best.device), best].detach().cpu().numpy()
            return cost.double().cpu().numpy(), routes


def worker(args):
    if torch.cuda.device_count() != 1 or '3090' not in torch.cuda.get_device_name(0):
        raise RuntimeError('Expose only the authorized idle RTX3090')
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    set_seed(2)
    plan = json.loads((args.root / 'audit_indices.json').read_text())['datasets']
    folder = args.root / 'runtime' / args.solver
    folder.mkdir(parents=True, exist_ok=True)
    record_path = folder / (args.variant + ('_preflight' if args.preflight else '') + '.jsonl')
    metadata = dict(solver=args.solver, variant=args.variant, seed=2, preflight=args.preflight,
        device=torch.cuda.get_device_name(0), cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
        historical_exact_replay_claimed=False, node_permutation='synchronized fields; depot stays fixed',
        started=time.time(), label_use='comparison after solving only; never supplied to the solver')
    if args.solver == 'SELECTOR':
        return selector_worker(args, plan, record_path, metadata)
    if args.solver == 'RELD_PAIR':
        from .r50_solver_encoder import source_provenance
        from .r51_probe import build_solvers
        provenance = source_provenance()
        models, module = build_solvers(provenance, 'cuda:0')
        metadata['provenance'] = provenance
        problems = MVRP
    elif args.solver in ('MATNET', 'MATPOENET', 'GLOP', 'ICAM'):
        runner = EasyReplay(args.solver)
        metadata['provenance'] = runner.evidence
        problems = ('CVRP',) if args.solver == 'ICAM' else ('ATSP',)
    elif args.solver == 'RELD_CVRP':
        sys.path.insert(0, str(BRIDGE))
        module = load_module('r57_single_bridge', BRIDGE / 'run_reld_cvrp_single_bridge.py')
        module.CVRP_ROOT = HOME / 'reld-nco-main/CVRP'
        model, model_params = module.instantiate_single_cvrp_model(torch.device('cuda:0'))
        metadata['provenance'] = dict(entry=str(BRIDGE / 'run_reld_cvrp_single_bridge.py'),
            path_relocation_only=True, model_params=model_params, eval_type='greedy', aug=1,
            checkpoint_sha256=file_hash(module.CVRP_ROOT / 'weights/ReLD/model_epoch_90.pt'))
        problems = ('CVRP',)
    else:
        # These bridges depend on unarchived imported settings and native cores.
        # Do not fill those settings with a guessed or current default recipe.
        raise RuntimeError('Bridge provenance remains partial: effective imported/native recipe needs certification before runtime replay')
    written = 0
    with record_path.open('w') as stream:
        for problem in problems:
            for split in ('train', 'val'):
                instances = load_instances(problem, split)
                raw = read_raw_costs(problem, split)
                source = plan[problem][split]
                verify_locked_inputs(source, raw)
                chosen = source['indices'][:1] if args.preflight else source['indices']
                # A singleton reproduces each method's existing batch1 contracts;
                # it also avoids padding a solver's original graph input.
                for i in chosen:
                    order = permutation(problem, instances[i], args.variant, i,
                                        'RELD_MOEL' if args.solver == 'RELD_PAIR' else args.solver)
                    changed = permute_instance(problem, instances[i], order)
                    n = true_size(problem, instances[i])
                    seed = int(args.variant[3:]) if args.variant.startswith('rng') else (2 if args.solver in ('RELD_PAIR', 'RELD_CVRP') else 1234)
                    set_seed(seed)
                    torch.cuda.synchronize()
                    started = time.perf_counter()
                    if args.solver == 'RELD_PAIR':
                        batch, _ = native_batch(module, [changed], [0], problem, split, 'cuda:0')
                        results = []
                        for name, model in zip(PAIR, models):
                            costs, routes, _, _, effective = solve(model, module, batch, problem)
                            results.append((name, float(costs[0]), routes[0].tolist(), effective))
                    elif args.solver == 'RELD_CVRP':
                        batch, scale = module.collate_batch([changed], [0])
                        batch = {k: v.cuda() for k, v in batch.items()}
                        costs = module.run_no_aug_scores(model, batch, scale, torch.device('cuda:0'))
                        results = [(args.solver, float(costs[0]), None, min(n, 100))]
                    else:
                        costs, routes = runner.run([changed], n, seed)
                        results = [(args.solver, float(costs[0]), None if routes is None else routes[0].tolist(), 1 if args.solver == 'GLOP' else n)]
                    torch.cuda.synchronize()
                    seconds = time.perf_counter() - started
                    for name, cost, route, effective in results:
                        starts = physical_starts(problem, instances[i], order, name)
                        original_starts = physical_starts(problem, instances[i], np.arange(n), name)
                        physical_route = None
                        if route is not None:
                            lookup = np.r_[0, order + 1] if problem != 'ATSP' else order
                            physical_route = lookup[np.asarray(route, np.int64)].tolist()
                        record = dict(problem=problem, split=split, index=i, solver=name, variant=args.variant,
                            true_size=n, seed=seed, cost=cost, historical_cost=float(raw['costs'][i, raw['pool'].index(name)]),
                            seconds=seconds / len(results), permutation=order.tolist(), effective_pomo=effective,
                            original_physical_starts=original_starts, current_physical_starts=starts,
                            physical_start_set_changed=None if starts is None else starts != original_starts,
                            route_original_ids=physical_route, historical_exact_recipe=False)
                        stream.write(json.dumps(record) + '\n')
                        stream.flush()
                        written += 1
                del instances
                print(f'[R57 probe] {args.solver}/{args.variant} {problem}/{split} rows={written}', flush=True)
    metadata.update(written=written, finished=time.time(), status='completed')
    dump(record_path.with_suffix('.metadata.json'), metadata)


@torch.no_grad()
def selector_worker(args, plan, record_path, metadata):
    from code.unified_selector.data import UnifiedProblemDataset, collate_single_problem
    from .local_geometry import local_geometry
    from .solver_code_encoder import load_r45_checkpoint
    from .pairwise_objective import inputs_only
    model, checkpoint = load_r45_checkpoint(BASELINE / 'best.pt', 'cuda:0')
    metadata.update(checkpoint=str((BASELINE / 'best.pt').resolve()),
                    checkpoint_sha256=file_hash(BASELINE / 'best.pt'), epoch=checkpoint['epoch'])
    written = 0
    with record_path.open('w') as stream:
        for problem in PROBLEMS:
            for split in ('train', 'val'):
                raw = read_raw_costs(problem, split)
                source = plan[problem][split]
                verify_locked_inputs(source, raw)
                dataset = UnifiedProblemDataset(problem, split)
                chosen = source['indices']
                original = [dataset[i] for i in chosen]
                for variant in ('original_0', 'preserve_roles', 'reassign_roles'):
                    samples = copy.deepcopy(original)
                    for sample, i in zip(samples, chosen):
                        instance = dataset.instances[i]
                        order = permutation(problem, instance, variant, i, 'RELD_MOEL' if problem in MVRP else 'RELD_CVRP')
                        if sample['node'] is not None:
                            order = np.r_[0, order + 1] if problem not in ('TSP',) else order
                            sample['node'] = sample['node'][order]
                        else:
                            sample['matrix'] = sample['matrix'][order][:, order]
                    batch = collate_single_problem(samples)
                    batch = {k: v.cuda() if torch.is_tensor(v) else v for k, v in batch.items()}
                    if batch['kind'] == 'coord':
                        batch['node_geom'] = local_geometry(batch['node'][..., :2], batch['node_mask'])
                    output = model(inputs_only(batch))
                    scores = output['logits'].float().cpu().numpy()
                    utilities = output['utility'].float().cpu().numpy() if 'utility' in output else None
                    for k, i in enumerate(chosen):
                        stream.write(json.dumps(dict(problem=problem, split=split, index=i, solver='R45A',
                            variant=variant, scores=scores[k].tolist(),
                            utility=None if utilities is None else utilities[k].tolist(), pool=POOLS[problem])) + '\n')
                        written += 1
                    stream.flush()
                print(f'[R57 selector] {problem}/{split}', flush=True)
                del dataset
    metadata.update(written=written, status='completed', finished=time.time())
    dump(record_path.with_suffix('.metadata.json'), metadata)


def verify_locked_inputs(source, raw):
    if any(source[name] != raw[name] for name in ('data_hash', 'label_hash')):
        raise ValueError('Locked runtime sample no longer matches inputs/labels')


def orchestrate(root, solvers=None):
    solvers = list(solvers or RECIPE_SOLVERS)
    records = []
    for solver in solvers:
        command = [sys.executable, '-m', 'code.V4.r57_probe', '--root', str(root), '--worker',
                   '--solver', solver, '--variant', 'original_0', '--preflight']
        folder = root / 'runtime' / solver
        folder.mkdir(parents=True, exist_ok=True)
        start = time.monotonic()
        with (folder / 'preflight.log').open('w') as stream:
            try:
                result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, timeout=120)
                status = 'passed' if result.returncode == 0 else 'blocked_runtime_dependency_or_recipe'
            except subprocess.TimeoutExpired:
                status = 'preflight_time_budget_exceeded'
        records.append(dict(solver=solver, status=status, seconds=time.monotonic() - start,
                            log=str((folder / 'preflight.log').resolve())))
        write_csv(root / 'runtime_preflight.csv', records)
    # Cover all recoverable method families before scheduling any full samples.
    for row in records:
        if row['status'] != 'passed':
            continue
        solver = row['solver']
        variants = list(VARIANTS) + (['rng2', 'rng3', 'rng4'] if solver == 'GLOP' else [])
        for variant in variants:
            command = [sys.executable, '-m', 'code.V4.r57_probe', '--root', str(root), '--worker',
                       '--solver', solver, '--variant', variant]
            path = root / 'runtime' / solver / (variant + '.log')
            with path.open('w') as stream:
                try:
                    result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, timeout=900)
                    status = 'completed' if result.returncode == 0 else 'failed'
                except subprocess.TimeoutExpired:
                    status = 'time_budget_exceeded'
            row[variant] = status
            dump(root / 'runtime_status.json', records)
            print(f'[R57 runtime] {solver}/{variant}: {status}', flush=True)
            if status != 'completed':
                break
    path = root / 'runtime/SELECTOR/selector.log'
    path.parent.mkdir(exist_ok=True)
    with path.open('w') as stream:
        result = subprocess.run([sys.executable, '-m', 'code.V4.r57_probe', '--worker', '--solver', 'SELECTOR',
                                 '--root', str(root)], stdout=stream, stderr=subprocess.STDOUT)
    dump(root / 'runtime_status.json', dict(solvers=records, selector_exit_code=result.returncode))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--preflight', action='store_true')
    parser.add_argument('--solver', choices=(*RECIPE_SOLVERS, 'SELECTOR'), default='SELECTOR')
    parser.add_argument('--variant', default='original_0')
    parser.add_argument('--solvers', nargs='+', choices=RECIPE_SOLVERS)
    args = parser.parse_args()
    if args.worker:
        try:
            worker(args)
        except Exception:
            traceback.print_exc()
            raise SystemExit(1)
    else:
        orchestrate(args.root, args.solvers)

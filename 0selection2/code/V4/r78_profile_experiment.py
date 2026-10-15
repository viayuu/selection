"""R78 round006 single fixed profile-representation trial; prepare is CPU-only."""

import argparse
from collections import Counter
import csv
import fcntl
import json
import os
from pathlib import Path
import socket
import time

import numpy as np
import torch

from code.unified_selector.registry import PROBLEMS
from .multitask_probe import task_seed
from .pairwise_objective import inputs_only
from .pairwise_selector import rank_by_solver_ids
from .r76_experiment import (PinnedPrefetch, file_hash, fixed_sbs_metrics, open_cache,
                             read_json, save_checkpoint, utc, write_json)
from .r77_experiment import ShuffleStream, load_checkpoint, metrics_summary, task_order
from .r78_profile_loss import PROFILE_KEY, ProfileObjective, cached_update, cost_profiles
from .train import configure_torch, set_seed


GOAL = Path(__file__).resolve().parent / 'runs/R78_zero_selection_goal'
ROOT = GOAL / 'round006'
DECISION = GOAL / 'review-stage/SOL_NEXT_EXPERIMENT_DECISION.md'
UPDATES, BATCH, SEED, WALL_SECONDS = 14904, 1024, 2, 4 * 3600
BASE_SHA = '31c0c75d66b1bde6ad533909a9c04b0036e80af4c4a5775b01b957ce310e6d1b'
GPU_NODES = {'GPU-71479e7e-f8bd-e430-66ed-f8db9f0d47be': 'gpu05',
             'GPU-597c682a-4b7f-7609-80a6-8ced791f5c5d': 'gpu05'}
FILES = ('r78_profile_loss.py', 'r78_profile_experiment.py', 'test_r78_profile.py',
         'run_v4_r78_profile.sh')


class BudgetExceeded(RuntimeError):
    pass


def workflow(root, phase, **values):
    record = dict(phase=phase, updated_utc=utc(), **values)
    write_json(root / 'status.json', record)
    path = root.parent / 'STATUS.json'
    with path.open() as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        state = json.load(handle)
        state['file_based_goal'].update(status='active', pending=True, complete=False, round=6, phase=phase)
        state['round006'] = dict(state.get('round006', {}), **record)
        state['next_action'] = record.get('next_action', phase)
        write_json(path, state)


def prepare(root):
    root.mkdir(parents=True, exist_ok=True)
    if (root / 'prepared.json').exists():
        raise ValueError('Preparation already exists; preserve the fixed recipe')
    previous = read_json(GOAL / 'round003/prepared.json')
    rates = np.load(GOAL / 'round003/late_lr.npy', allow_pickle=False)
    if rates.dtype != np.float64 or rates.shape != (UPDATES,):
        raise ValueError('Reuse the exact control late-LR array')
    np.save(root / 'late_lr.npy', rates, allow_pickle=False)
    config = {key: previous[key] for key in ('source_checkpoint', 'source_checkpoint_sha256',
        'source_r77_root', 'source_cache_manifest', 'data_root', 'model_params', 'microbatch_caps')}
    config.update(experiment='R78_round006', prepared_utc=utc(), seed=SEED, batch_size=BATCH,
        successful_updates=UPDATES, updates_per_task=828, train_rows_per_task=70114,
        val_rows_per_task=3895, tasks=list(PROBLEMS), wall_seconds=WALL_SECONDS,
        wall_scope='Entire allocated process, including update1, fitting, final TRAIN/VAL and conditional TEST',
        profile_weight=.10, temperature=1., feature='encode_pairs first embedding block: pooled h',
        target='FP64 q=log(C/min(C))-mean_K(log(C/min(C))); d=mean_K(abs(q_i-q_j))',
        profile='mean_i!=j(logsumexp_k!=i,d_ik>=d_ij(cos(h_i,h_k))-cos(h_i,h_j))',
        normalization='Global1024, symmetric anchor+key autograd, two original dropout views averaged',
        optimizer=dict(name='AdamW', betas=[.9, .999], eps=1e-8, weight_decay=1e-4, fresh=True),
        clip=1., amp='FP16 complete forwards; FP32 losses/features, one scale per combined micro loss',
        eval_batch_size=256, trainable='All existing B parameters; no new inference parameters',
        base_loss='Unchanged R77 .35CE+.35compare+.02originalRisk; two-view mean+.35RDropKL',
        endpoint='Only final14904; complete all18 and lock SHA before candidate VAL',
        test_authorization='Coordinator explicitly preauthorized one full18 TEST iff final fixed VAL gate passes',
        lib_authorized=False, scans=False, gpu_launch_approved=False,
        decision=str(DECISION), decision_sha256=file_hash(DECISION),
        code_sha256={name: file_hash(Path(__file__).parent / name) for name in FILES},
        late_lr_sha256=file_hash(root / 'late_lr.npy'),
        baseline_records=str(GOAL / 'round003/baseline_B_train_val.json'))
    if config['source_checkpoint_sha256'] != BASE_SHA:
        raise ValueError('The approved initialization is locked R77 fullB ep28/u34776 only')
    write_json(root / 'prepared.json', config)
    write_json(root / 'hardware_allocation.pending.json', dict(approved=False,
        r79_frozen_test_lib_delivered_and_gpu_released=False, release_receipt=None,
        gpu_uuid=None, node=None, job_id=None, cache_root='/tmp/shiys_R76_R45A_single_reld_mt_cache',
        cpus_per_task=4, mem='9G', slurm_gpu_count=1))
    workflow(root, 'round006_cpu_prepared_pending_hardware', successful_updates=0, gpu_started=False,
        next_action='Run only the two small CPU tests; await coordinator R79 release receipt and exact single4090 UUID/allocation. No GPU probe or reservation.')
    print(json.dumps(dict(prepared=str(root / 'prepared.json'), gpu_started=False,
                          updates=UPDATES, lr_first=float(rates[0]), lr_last=float(rates[-1]))), flush=True)


def allocation(path, require_step=False):
    value = read_json(path)
    if (value.get('approved') is not True
            or value.get('r79_frozen_test_lib_delivered_and_gpu_released') is not True
            or not value.get('release_receipt')):
        raise ValueError('Wait for explicit R79 complete delivery/GPU release and coordinator allocation')
    if (value.get('gpu_uuid') not in GPU_NODES or GPU_NODES[value['gpu_uuid']] != value.get('node')
            or not str(value.get('job_id', '')).isdigit() or value.get('cpus_per_task') != 4
            or value.get('mem') != '9G' or value.get('slurm_gpu_count') != 1):
        raise ValueError('Require the exact owner-approved single4090 UUID, allocation, CPU4/mem9G')
    if require_step and (os.environ.get('SLURM_JOB_ID') != str(value['job_id'])
            or not os.environ.get('SLURM_STEP_ID') or socket.gethostname().split('.')[0] != value['node']
            or os.environ.get('CUDA_VISIBLE_DEVICES') != value['gpu_uuid']):
        raise ValueError('Run only in the assigned exact Slurm step and UUID')
    return value


class ProfileCache:
    def __init__(self, source):
        self.source = source
        self.raw, self.size = source.raw, source.size

    def batch(self, rows, device='cpu'):
        if str(device) != 'cpu':
            raise ValueError('Use original CPU gather/pin/async-transfer path')
        batch = self.source.batch(rows, device='cpu')
        indices = rows.cpu().numpy() if torch.is_tensor(rows) else np.asarray(rows)
        costs = torch.from_numpy(np.array(self.raw['costs'][indices], copy=True))
        batch[PROFILE_KEY] = cost_profiles(costs, batch['solver_mask'])
        return batch

    def close(self):
        self.raw = {}
        self.source.close()


def choices(scores, pool_ids):
    return rank_by_solver_ids(torch.as_tensor(scores), torch.as_tensor(pool_ids))[:, 0].numpy()


def method_report(costs, winner, old, new, pool):
    oracle = costs.min(1)
    change = costs[np.arange(len(new)), new] - costs[np.arange(len(old)), old]
    groups = dict(corrected=(old != winner) & (new == winner),
                  harmed=(old == winner) & (new != winner),
                  wrong_to_wrong=(old != winner) & (new != winner) & (old != new))
    return dict(changed=int((old != new).sum()),
        groups={key: dict(n=int(mask.sum()), signed_cost_sum=float(change[mask].sum(dtype=np.float64)),
            signed_cost_mean=float(change[mask].mean(dtype=np.float64)) if mask.any() else 0.,
            signed_regret_sum_pct=float((change[mask] / oracle[mask]).sum(dtype=np.float64) * 100.))
            for key, mask in groups.items()},
        native_methods=[dict(method=str(name), winners=int((winner == j).sum()),
            B_selected=int((old == j).sum()), fitted_selected=int((new == j).sum()))
            for j, name in enumerate(pool)],
        zero_winner_methods_B=[str(name) for j, name in enumerate(pool)
                              if (winner == j).any() and not (old == j).any()],
        zero_winner_methods_fitted=[str(name) for j, name in enumerate(pool)
                                   if (winner == j).any() and not (new == j).any()])


@torch.no_grad()
def evaluate(model, root, ready, manifest, common_sbs, cache_root, split, check_deadline,
             baseline=None):
    if split == 'test' and not (root / 'test_started.json').exists():
        raise ValueError('TEST must be claimed once after the locked candidate passed VAL')
    directory = root / 'evaluation' / split / 'final14904'
    directory.mkdir(parents=True, exist_ok=True)
    model.eval()
    per_task, diagnostics = {}, {}
    shards = []
    if baseline is not None:
        shards = [np.load(path, allow_pickle=False) for path in sorted(
            Path(baseline['prediction_directories'][split]).glob('rank*.npz'))]
        if not shards:
            raise ValueError('Reuse existing B TRAIN/VAL predictions; never rerun baseline')
    try:
        for problem in PROBLEMS:
            check_deadline()
            cache = open_cache(cache_root, split, problem, manifest, Path(ready['data_root']))
            try:
                expected = manifest['splits']['train'][problem]
                if list(cache.pool) != expected['pool'] or list(cache.pool_ids) != expected['pool_ids']:
                    raise ValueError('Evaluation must retain the original native task pool')
                requests = ((problem, torch.arange(start, min(start + 256, cache.size)),
                    min(256, cache.size - start)) for start in range(0, cache.size, 256))
                values = []
                with PinnedPrefetch({problem: cache}, requests, next(model.parameters()).device) as batches:
                    for _, _, _, batch in batches:
                        check_deadline()
                        values.append(model(inputs_only(batch))['logits'].float().cpu().numpy())
                scores = np.concatenate(values)
                if len(scores) != cache.size or cache.raw['costs'].dtype != np.float64:
                    raise ValueError('Complete original FP64 rows required')
                per_task[problem], predicted = fixed_sbs_metrics(scores, cache.raw, common_sbs[problem])
                new = choices(scores, cache.pool_ids)
                if not np.array_equal(predicted, new):
                    raise ValueError('Native policy and global-ID tie semantics must agree')
                if baseline is not None:
                    prefix = problem + '__'
                    source = next(shard for shard in shards if prefix + 'logits' in shard)
                    if (not np.array_equal(source[prefix + 'indices'], np.arange(cache.size))
                            or not np.array_equal(source[prefix + 'pool_ids'], cache.pool_ids)
                            or not np.array_equal(source[prefix + 'costs'], cache.raw['costs'])
                            or not np.array_equal(source[prefix + 'winner'], cache.raw['winner'])):
                        raise ValueError('Cached B/native task rows, labels or pools do not align')
                    old = choices(source[prefix + 'logits'], cache.pool_ids)
                    diagnostics[problem] = method_report(cache.raw['costs'], cache.raw['winner'], old, new, cache.pool)
                np.savez(directory / f'{problem}.npz', logits=scores, predicted=new,
                    indices=np.arange(cache.size), winner=cache.raw['winner'], costs=cache.raw['costs'],
                    pool=np.asarray(cache.pool), pool_ids=np.asarray(cache.pool_ids))
                print(f'[evaluation] {split}/{problem}: Top1={per_task[problem]["top1"]:.9f}', flush=True)
            finally:
                cache.close()
    finally:
        for shard in shards:
            shard.close()
    result = metrics_summary(per_task)
    write_json(directory / 'metrics.json', result)
    if baseline is not None:
        write_json(directory / 'method_changes.json', diagnostics)
        with (directory / 'native_methods.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=['task', 'method', 'winners', 'B_selected', 'fitted_selected'])
            writer.writeheader()
            for problem, value in diagnostics.items():
                writer.writerows(dict(task=problem, **row) for row in value['native_methods'])
    return result


def run(args):
    root = args.root
    assigned = allocation(args.allocation, require_step=True)
    ready = read_json(root / 'prepared.json')
    if not (root / 'cpu_checks.json').is_file() or read_json(root / 'cpu_checks.json')['passed'] != 2:
        raise ValueError('The two planned CPU checks must pass before launch')
    for name, sha in ready['code_sha256'].items():
        if file_hash(Path(__file__).parent / name) != sha:
            raise ValueError(f'Prepared implementation changed: {name}')
    if file_hash(root / 'late_lr.npy') != ready['late_lr_sha256']:
        raise ValueError('Fixed original late-LR schedule changed')
    # Exclusive claim prevents a second trial or silently restarting a partial budget.
    with (root / 'training_started.json').open('x') as handle:
        json.dump(dict(utc=utc(), pid=os.getpid(), allocation=assigned), handle)
    started = time.monotonic()
    successful, counts, caches, tracker = 0, Counter(), {}, None

    def deadline():
        if time.monotonic() - started >= WALL_SECONDS:
            raise BudgetExceeded('Fixed4 GPU-hour ceiling reached; no extension/rescue authorized')

    try:
        configure_torch()
        torch.set_num_threads(2)
        device = torch.device('cuda:0')
        if torch.cuda.device_count() != 1:
            raise ValueError('Exactly one explicitly assigned CUDA device is allowed')
        actual_uuid = str(torch.cuda.get_device_properties(device).uuid)
        if actual_uuid.removeprefix('GPU-') != assigned['gpu_uuid'].removeprefix('GPU-'):
            raise ValueError('CUDA device UUID does not match the owner assignment')
        if file_hash(ready['source_checkpoint']) != BASE_SHA:
            raise ValueError('Locked B checkpoint SHA mismatch')
        model, source = load_checkpoint(ready['source_checkpoint'], device)
        if source['successful_updates'] != 34776 or source['args']['arm'] != 'B':
            raise ValueError('Only full-B ep28/u34776 weights initialize this independent run')
        del source
        set_seed(SEED)
        model.train()
        objective = ProfileObjective(model)
        rates = np.load(root / 'late_lr.npy', allow_pickle=False)
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(rates[0]),
            betas=(.9, .999), eps=1e-8, weight_decay=1e-4)
        scaler = torch.amp.GradScaler('cuda')
        manifest = read_json(ready['source_cache_manifest'])
        baseline = read_json(ready['baseline_records'])
        common_sbs = read_json(Path(ready['source_r77_root']) / 'common_full_train_sbs.json')
        for problem in PROBLEMS:
            cache = open_cache(args.cache_root, 'train', problem, manifest, Path(ready['data_root']))
            if cache.size != 70114:
                raise ValueError('All18 complete natural TRAIN populations must be retained')
            caches[problem] = ProfileCache(cache)
        streams = {p: ShuffleStream(70114, task_seed(SEED, p)) for p in PROBLEMS}
        order = task_order(UPDATES, seed=SEED)
        requests = ((p, streams[p].take(BATCH), BATCH) for p in order)
        import wandb
        tracker = wandb.init(project='selector', entity='yjkds-southern-university-of-science-technology',
            name='R78_round006_profile_seed2', mode='offline', dir=str(root), config=ready,
            settings=wandb.Settings(x_disable_stats=True, init_timeout=15))
        with PinnedPrefetch(caches, requests, device) as batches:
            for problem, n, total, batch in batches:
                deadline()
                if n != BATCH or total != BATCH:
                    raise ValueError('Every successful update is one complete logical1024 task batch')
                for group in optimizer.param_groups:
                    group['lr'] = float(rates[successful])
                cap = 352 if problem in ('TSP', 'CVRP') else BATCH
                parts, retries = cached_update(objective, optimizer, scaler, batch, cap,
                    check_replay=successful == 0, deadline_check=deadline)
                successful += 1
                counts[problem] += 1
                if successful == 1:
                    write_json(root / 'update1_receipt.json', dict(successful_updates=1, formal_budget=True,
                        finite=all(np.isfinite(v) for v in parts.values()), exact_feature_replay=True,
                        logical_batch=BATCH, physical_cap=cap, task=problem, cuda_uuid=actual_uuid,
                        pid=os.getpid(), retries=retries, metrics=parts))
                if successful == 1 or successful % 100 == 0 or successful == UPDATES:
                    elapsed = time.monotonic() - started
                    receipt = dict(successful_updates=successful, target_updates=UPDATES,
                        per_task_updates=dict(counts), seconds=elapsed,
                        eta_training_seconds=elapsed / successful * (UPDATES - successful),
                        peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                        pid=os.getpid(), cuda_uuid=actual_uuid, metrics=parts, retries=retries)
                    workflow(root, 'round006_training', **receipt)
                    print(json.dumps(receipt), flush=True)
                    tracker.log(dict(parts, successful_updates=successful, lr=float(rates[successful - 1])))
        if successful != UPDATES or any(counts[p] != 828 for p in PROBLEMS):
            raise ValueError('Final endpoint must contain exactly14904 successful updates,828/task')
        for cache in caches.values():
            cache.close()
        caches.clear()
        deadline()
        checkpoint = root / 'final14904.pt'
        save_checkpoint(checkpoint, dict(model=model.state_dict(), successful_updates=successful,
            per_task_updates=dict(counts), model_params=ready['model_params'], experiment='R78_round006',
            optimizer=optimizer.state_dict(), scaler=scaler.state_dict(), source_sha256=BASE_SHA))
        lock = dict(locked_utc=utc(), checkpoint=str(checkpoint), checkpoint_sha256=file_hash(checkpoint),
            successful_updates=successful, per_task_updates=dict(counts), preprocessing='Original fullB, unchanged',
            source_checkpoint_sha256=BASE_SHA, model_params=ready['model_params'],
            pools={p: dict(pool=manifest['splits']['train'][p]['pool'],
                           pool_ids=manifest['splits']['train'][p]['pool_ids']) for p in PROBLEMS},
            inference='Unchanged score_difference/maximin, ties by global solver ID',
            code_sha256=ready['code_sha256'], cache_manifest_sha256=file_hash(ready['source_cache_manifest']),
            candidate_VAL_previously_opened=False, test_authorization=ready['test_authorization'])
        write_json(root / 'candidate_lock.json', lock)
        workflow(root, 'round006_final_train_val', successful_updates=successful, checkpoint_sha256=lock['checkpoint_sha256'])
        fitted = {split: evaluate(model, root, ready, manifest, common_sbs, args.cache_root,
                                 split, deadline, baseline) for split in ('train', 'val')}
        c, b = fitted['val']['macro'], baseline['val']['macro']
        gate = dict(top1_strictly_improves=c['macro_top1'] > b['macro_top1'],
            raw_mean_cost_not_worse=c['macro_mean_cost'] <= b['macro_mean_cost'],
            actual_regret_not_worse=c['macro_actual_regret_pct'] <= b['macro_actual_regret_pct'])
        passed = all(gate.values())
        results = dict(successful_updates=successful, per_task_updates=dict(counts), fitted=fitted,
            B={split: baseline[split] for split in ('train', 'val')},
            fallback_B={split: baseline[split] for split in ('train', 'val')}, gate=gate, gate_passed=passed,
            selected_rule='locked_final14904' if passed else 'whole_B_fallback',
            historical_control_final14904_val=dict(macro_top1=.7081443446013407,
                macro_mean_cost=11.395855343252101, macro_actual_regret_pct=.41092598253845325),
            control_train_available=False, test_run=False, no_test_or_lib_for_design=True,
            scientific_status='exploratory_gate_pass' if passed else 'fixed_profile_route_closed_negative')
        write_json(root / 'results.json', results)
        if passed:
            release = dict(lock, val_gate=gate, val=fitted['val'], locked_before_TEST=True,
                test_runs_authorized=1, lib_authorized=False)
            write_json(root / 'release_manifest.json', release)
            deadline()
            with (root / 'test_started.json').open('x') as handle:
                json.dump(dict(utc=utc(), checkpoint_sha256=lock['checkpoint_sha256'],
                               gate_passed=True, authorization=ready['test_authorization']), handle)
            results['test'] = evaluate(model, root, ready, manifest, common_sbs, args.cache_root,
                                       'test', deadline)
            results['test_run'] = True
            write_json(root / 'results.json', results)
        elapsed = time.monotonic() - started
        write_json(root / 'execution_receipt.json', dict(complete=True, successful_updates=successful,
            per_task_updates=dict(counts), elapsed_seconds=elapsed, wall_seconds=WALL_SECONDS,
            test_completed=results['test_run'], gate_passed=passed, pid=os.getpid(),
            cuda_uuid=actual_uuid, checkpoint_sha256=lock['checkpoint_sha256']))
        workflow(root, 'round006_complete', successful_updates=successful, complete=True,
            elapsed_seconds=elapsed, gate_passed=passed, selected_rule=results['selected_rule'],
            next_action='Deliver fixed results. No rescue/extra budget; file goal remains active, app goals untouched.')
    except BaseException as error:
        write_json(root / 'execution_stop.json', dict(complete=False, successful_updates=successful,
            per_task_updates=dict(counts), elapsed_seconds=time.monotonic() - started,
            error=repr(error), scientific_status='incomplete_not_a_negative_result', no_extension=True))
        workflow(root, 'round006_execution_stopped_incomplete', successful_updates=successful,
            error=repr(error), next_action='Report incomplete execution; no automatic retry/new budget/rule change.')
        raise
    finally:
        for cache in caches.values():
            cache.close()
        if tracker is not None:
            tracker.finish()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('prepare', 'allocation', 'train'))
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--allocation', type=Path)
    parser.add_argument('--cache-root', type=Path, default=Path('/tmp/shiys_R76_R45A_single_reld_mt_cache'))
    args = parser.parse_args()
    if args.stage == 'prepare':
        prepare(args.root)
    elif args.stage == 'allocation':
        print(json.dumps(allocation(args.allocation)))
    else:
        with (args.root / 'execution.lock').open('a') as handle:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            run(args)


if __name__ == '__main__':
    main()

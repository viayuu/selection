"""One full18 family-conditional selector; two GPUs, global batch1024.

Reuses the frozen R84 inputs, natural streams, evaluation and AMP update.
Only rank0 writes checkpoints or chooses them; held-out labels are evaluation
only. No historical module globals are changed.
"""

import argparse
from collections import Counter
from datetime import timedelta
import fcntl
import json
import math
import os
from pathlib import Path
import socket
import time

import torch
from torch import nn
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel

from code.unified_selector.registry import PROBLEMS
from . import r84_experiment as base
from .multitask_probe import task_seed
from .pairwise_objective import inputs_only
from .r76_experiment import (PART_KEYS, Runtime, PinnedPrefetch, capture_rank_rng,
    file_hash, open_cache, read_json, restore_rank_rng, save_checkpoint,
    update_batch as amp_update, utc, write_json)
from .r77_experiment import ShuffleStream, task_order
from .r77_data import parameter_hash
from .r80_experiment import json_hash, lock_best, wandb_run, write_metrics_csv
from .train import configure_torch, set_seed


ROOT = Path(__file__).resolve().parent / 'runs/R111_family_conditional_features'
ARMS, BUDGETS = ('A_family_conditional',), {'A_family_conditional': 57600}
MAX_UPDATES, EVAL_EVERY, BATCH_SIZE = 57600, 1440, 1024
MODEL_FACTORY = 'code.V4.r111_experiment.load_checkpoint'
INFERENCE = 'eval, argmax normalized family-times-within-family probability; stable global-ID ties'


class ConditionalObjective(nn.Module):
    def __init__(self, selector):
        super().__init__()
        self.selector = selector

    def forward(self, batch, consistency_weight=0.):
        device = batch['ind'].device
        with torch.autocast(device_type=device.type, dtype=torch.float16,
                            enabled=device.type == 'cuda'):
            output = self.selector(inputs_only(batch))
        # Already normalized probabilities: a second softmax would change the
        # conditional-gradient interpretation and is deliberately not applied.
        loss = F.nll_loss(output['logits'].float(), batch['ind'])
        parts = loss.detach().new_zeros(len(PART_KEYS) - 1)
        parts[PART_KEYS.index('loss')] = parts[PART_KEYS.index('ce')] = loss.detach()
        return loss, parts


def validate_checkpoint_config(config):
    required = dict(experiment='R111', arm=ARMS[0], seed=2,
        global_batch_size=1024, logical_global_batch=1024,
        max_successful_updates=57600, world_size=2,
        model_factory=MODEL_FACTORY, inference_rule=INFERENCE)
    if any(config.get(k) != v for k, v in required.items()):
        raise ValueError('Use the frozen R111 whole18 conditional experiment')


def load_checkpoint(path, device='cpu'):
    from .r111_model import make_model
    state = torch.load(path, map_location='cpu', weights_only=False)
    validate_checkpoint_config(state['args'])
    model = make_model(state['args']['model_params'], ARMS[0], state_dict=state['model'])
    return model.to(device), state


def initialize(initial, geometry, device):
    from .r111_model import initialize_shared
    return initialize_shared(initial, geometry, ARMS[0], device)


def protocol(args, ready, manifest, geometry, baselines, params, order):
    config = base.protocol(args, ready, manifest, geometry, baselines, params, order,
        dict(TSP=256, CVRP=256, ATSP=128, **{p: 512 for p in PROBLEMS[3:]}))
    config.update(experiment='R111', model_factory=MODEL_FACTORY,
        initialization='scratch_shared_r77_family_conditional', world_size=2,
        local_batch_size=512, conditional_model=params.get('r111_spec'),
        selection_head='normalized_family_times_conditional', inference_rule=INFERENCE,
        loss=dict(nll=1., comparison=0., risk=0., rdrop=0.),
        rdrop_weight=0., train_eval='full TRAIN every7200 plus locked selected checkpoint',
        gradients='two ranks512 each; micro rows*world/global1024 before DDP mean; one clip/step; coordinated AMP replay',
        research_boundary='One new mechanism, historical controls only; no parameter/FLOP matched efficacy claim',
        source_sha256={name: file_hash(Path(__file__).parent / name) for name in
            ('r111_experiment.py', 'r111_model.py')})
    return config


def runtime():
    rank, world, local = (int(os.environ.get(k, d)) for k, d in
        (('RANK', '0'), ('WORLD_SIZE', '1'), ('LOCAL_RANK', '0')))
    if world != 2 or local not in (0, 1):
        raise ValueError('Formal R111 requires torchrun --nproc-per-node=2')
    torch.cuda.set_device(local)
    dist.init_process_group('nccl', timeout=timedelta(hours=2))
    return Runtime(rank, world, torch.device('cuda', local), local)


def train(args):
    rt = runtime()
    run = args.root / args.arm
    ready, manifest, geometry, baselines = base.prepared_config(args)
    order = task_order(MAX_UPDATES)
    initial = torch.load(args.prepared_root / 'identicalinit_seed2.pt', map_location='cpu', weights_only=False)
    model, params, receipt = initialize(initial, geometry, rt.device)
    del initial
    config = protocol(args, ready, manifest, geometry, baselines, params, order)
    state_hash = json_hash(config)
    last = run / 'last.pt'
    if (run / 'training_complete.json').exists():
        raise RuntimeError('Completed training is never silently relaunched; run finalize instead')
    if last.exists() and not args.resume:
        raise RuntimeError('Preserve existing run; use explicit --resume')
    if (run / 'protocol.json').exists() and read_json(run / 'protocol.json') != config:
        raise RuntimeError('Existing scientific protocol differs')
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scaler = torch.amp.GradScaler('cuda')
    history, best, successful, retries = [], None, 0, 0
    counts, cursors = Counter(), dict.fromkeys(PROBLEMS, 0)
    if args.resume:
        saved = torch.load(last, map_location='cpu', weights_only=False)
        if saved['args'] != config:
            raise RuntimeError('Resume protocol mismatch')
        model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        for state in optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value) and key != 'step':
                    state[key] = value.to(rt.device)
        scaler.load_state_dict(saved['scaler'])
        successful, retries = saved['successful_updates'], saved['amp_retries']
        history, best, counts = saved['history'], saved['best'], Counter(saved['updates_by_problem'])
        cursors, receipt = saved['stream_draws'], saved['initialization']
        restore_rank_rng(saved['rank_rng'][rt.rank], rt.device)
        del saved
    else:
        set_seed(2 + rt.rank)
    objective = DistributedDataParallel(ConditionalObjective(model),
        device_ids=[rt.local_rank], find_unused_parameters=True)
    streams = {p: ShuffleStream(81000, task_seed(2, p), cursors[p]) for p in PROBLEMS}
    caches = {p: open_cache(args.cache_root, 'train', p, manifest, args.data_root) for p in PROBLEMS}
    if any(cache.size != 81000 for cache in caches.values()):
        raise ValueError('All TRAIN rows must remain available')

    def snapshot():
        rank_rng = [None] * 2
        dist.all_gather_object(rank_rng, capture_rank_rng(rt.device))
        return dict(model=model.state_dict(), optimizer=optimizer.state_dict(), scaler=scaler.state_dict(),
            args=config, state_hash=state_hash, initialization=receipt, rank_rng=rank_rng,
            successful_updates=successful, sample_exposures=successful * 1024,
            updates_by_problem=dict(counts), stream_draws={p: s.draws for p, s in streams.items()},
            amp_retries=retries, history=history, best=best, epoch=successful // EVAL_EVERY)

    if rt.rank == 0:
        write_json(run / 'protocol.json', config)
        write_json(run / 'initialization.json', receipt)
    if not last.exists():
        state = snapshot()
        if rt.rank == 0:
            save_checkpoint(last, state)
        del state
    rt.barrier()
    before_tracking = capture_rank_rng(rt.device)
    tracker = wandb_run(args, config) if rt.rank == 0 else None
    restore_rank_rng(before_tracking, rt.device)
    try:
        while successful < MAX_UPDATES:
            start, end = successful, min(MAX_UPDATES, successful + EVAL_EVERY)
            requests = []
            for problem in order[start:end]:
                indices = streams[problem].take(1024)
                requests.append((problem, indices[rt.rank * 512:(rt.rank + 1) * 512], 1024))
            weighted = torch.zeros(len(PART_KEYS), dtype=torch.float64, device=rt.device)
            started = time.monotonic()
            model.train()
            with PinnedPrefetch(caches, requests, rt.device) as batches:
                for problem, n_rank, n_global, batch in batches:
                    if (n_rank, n_global) != (512, 1024):
                        raise ValueError('Global batch must remain1024 independent natural rows')
                    lr = base.learning_rate(successful + 1)
                    for group in optimizer.param_groups:
                        group['lr'] = lr
                    parts, retried = amp_update(objective, optimizer, scaler, batch, 0.,
                        n_rank, n_global, rt, microbatch_cap=config['microbatch_caps'][problem])
                    weighted.add_(parts.double())
                    retries += retried
                    successful += 1
                    counts[problem] += 1
                    if successful % args.log_every == 0 or successful == 1 or successful == end:
                        elapsed = time.monotonic() - started
                        values = dict(successful_updates=successful, max_updates=MAX_UPDATES,
                            sample_exposures=successful * 1024, problem=problem, lr=lr,
                            consistency_weight=0., amp_retries=retries, world_size=2,
                            seconds_per_update=elapsed / (successful - start),
                            samples_per_second=(successful - start) * 1024 / elapsed,
                            updates_by_problem=dict(counts),
                            **dict(zip(PART_KEYS, (weighted / (successful - start)).cpu().tolist())))
                        if rt.rank == 0:
                            base.status_update(args, 'training', successful, **{k: v for k, v in values.items() if k != 'successful_updates'})
                            with (run / 'updates.jsonl').open('a') as stream:
                                stream.write(json.dumps(dict(updated_utc=utc(), **values), allow_nan=False) + '\n')
                            print('[R111 train] ' + json.dumps(values), flush=True)
                            if tracker:
                                tracker.log(values, step=successful)
            dist.all_reduce(weighted)
            optimization = dict(zip(PART_KEYS, (weighted / (2 * (end - start))).cpu().tolist()))
            optimization.update(lr=lr, consistency_weight=0., seconds=time.monotonic() - started,
                samples=(end - start) * 1024, amp_retries=retries)
            if rt.rank == 0:
                tag = f'update{successful:07d}'
                val = base.evaluate(model, args, manifest, baselines, 'val', tag, successful, state_hash)
                training = base.evaluate(model, args, manifest, baselines, 'train', tag, successful, state_hash) if successful % 7200 == 0 else None
                metric = val['macro']['macro_vs_sbs_pct']
                if not math.isfinite(metric):
                    raise ValueError('VAL selection metric must be finite')
                row = dict(epoch=successful // EVAL_EVERY, evaluation_point=successful // EVAL_EVERY,
                    successful_updates=successful, sample_exposures=successful * 1024,
                    optimization=optimization, val=val, train=training, evaluation_tag=tag)
                history.append(row)
                improved = best is None or metric < best['val']['macro']['macro_vs_sbs_pct']
                if improved:
                    best = row
                if tracker:
                    tracker.log({**{'val/' + k: v for k, v in val['macro'].items()},
                        'successful_updates': successful}, step=successful)
            else:
                improved = False
            # All ranks reach this only after rank0's complete evaluation.
            state = snapshot()
            if rt.rank == 0:
                if improved:
                    save_checkpoint(run / 'best.pt', state)
                    write_json(run / 'best_eval.json', best)
                save_checkpoint(last, state)
                write_json(run / 'history.json', history)
                write_metrics_csv(run, history)
                base.status_update(args, 'evaluation_complete', successful,
                    selected_update=best['successful_updates'], latest_val=val['macro'])
            del state
            rt.barrier()
        if any(counts[p] != 3200 for p in PROBLEMS):
            raise ValueError('Every problem must have exactly3200 successful updates')
        if rt.rank == 0:
            lock = lock_best(run, best, config, state_hash)
            write_json(run / 'training_complete.json', dict(completed_utc=utc(),
                successful_updates=successful, updates_by_problem=dict(counts),
                selected_best=lock, state_hash=state_hash, test_read=False))
            base.status_update(args, 'training_complete', successful, selected_update=lock['selected_update'])
    finally:
        for cache in caches.values():
            cache.close()
        if tracker:
            tracker.finish()
        dist.destroy_process_group()


def authorize_test(args):
    lock, config = base.authorize_test(args)
    validate_checkpoint_config(config)
    return lock, config


def ensure_test_cache(args, lock):
    # Caches already exist. Do not fall back to generation/preprocessing.
    manifest = read_json(args.prepared_root / 'source_cache_manifest.json')
    for problem in PROBLEMS:
        path = args.cache_root / 'test' / problem / 'metadata.json'
        if not path.is_file() or read_json(path).get('complete') is not True:
            raise RuntimeError('Existing warm TEST cache missing; no implicit rebuild')
        if manifest['splits']['test'][problem]['rows'] != 4500:
            raise RuntimeError('Unexpected TEST membership')
    return manifest


evaluate = base.evaluate


def finalize(args, device):
    run = args.root / args.arm
    lock, config = authorize_test(args)
    _, manifest, _, baselines = base.prepared_config(args)
    model, state = load_checkpoint(run / 'best.pt', device)
    del state
    # Fresh, explicitly file-bound selected evaluations avoid rebinding an old
    # in-training archive whose checkpoint file did not yet exist.
    val = evaluate(model, args, manifest, baselines, 'val', 'selected_best',
        lock['selected_update'], lock['state_hash'], lock['checkpoint_sha256'])
    training = evaluate(model, args, manifest, baselines, 'train', 'selected_best',
        lock['selected_update'], lock['state_hash'], lock['checkpoint_sha256'])
    final_path = run / 'final_metrics.json'
    if not final_path.exists():
        write_json(final_path, dict(epoch=lock['epoch'], selected_update=lock['selected_update'],
            checkpoint_sha256=lock['checkpoint_sha256'], state_hash=lock['state_hash'],
            train=training, val=val, test_read=False))
    del model
    return base.test(args, device, integration=__import__(__name__, fromlist=['unused']))


@torch.no_grad()
def timing(args, device):
    from .r110_experiment import cached_input_batch, measure_inputs
    run = args.root / args.arm
    lock, config = authorize_test(args)
    path = run / 'inference_timing.json'
    if path.exists():
        record = read_json(path)
        if record.get('checkpoint_sha256') != lock['checkpoint_sha256']:
            raise ValueError('Saved timing belongs to another checkpoint')
        return record
    model, state = load_checkpoint(run / 'best.pt', device)
    del state
    model.eval()
    manifest = read_json(args.prepared_root / 'source_cache_manifest.json')
    rows = []
    for problem in PROBLEMS:
        batch = cached_input_batch(config, manifest, 'val', problem, device, count=128)
        rows.append(dict(problem=problem, instances=128,
            standalone_model=measure_inputs(model, batch, device, repetitions=3)))
        del batch
    record = dict(experiment='R111', arm=args.arm, complete=True,
        checkpoint_sha256=lock['checkpoint_sha256'], state_hash=lock['state_hash'],
        selected_update=lock['selected_update'], model_state_sha256=parameter_hash(model.state_dict()),
        inference_rule=INFERENCE, device=str(device), device_name=torch.cuda.get_device_name(device),
        parameter_count=sum(p.numel() for p in model.parameters()),
        scope='complete_standalone_model_GPU_resident_inputs', all_current_model_included=True,
        family_and_conditional_branches_included=True, end_to_end_measured=False,
        excludes=['dataset I/O', 'CPU transfer', 'routing solvers'],
        repetitions=3, warmup_batches=1, per_problem=rows, completed_utc=utc())
    write_json(path, record)
    return record


def parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('train', 'finalize', 'test', 'timing'))
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--arm', choices=ARMS, default=ARMS[0])
    parser.add_argument('--data-root', type=Path, default=base.DEFAULT_DATA_ROOT)
    parser.add_argument('--prepared-root', type=Path, default=base.DEFAULT_PREPARED_ROOT)
    parser.add_argument('--cache-root', type=Path, default=Path('/tmp/shiys_R84_new90k_cache'))
    parser.add_argument('--gpu-type', default='4090', choices=('4090',))
    parser.add_argument('--large-micro-cap', type=int, default=256)
    parser.add_argument('--eval-batch-size', type=int, default=256)
    parser.add_argument('--log-every', type=int, default=18)
    parser.add_argument('--wandb-mode', default='offline', choices=('offline', 'online', 'disabled'))
    parser.add_argument('--state-hash')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--resume', action='store_true')
    return parser


def main(argv=None):
    args = parser().parse_args(argv)
    args.experiment = 'R111'
    configure_torch()
    torch.set_num_threads(2)
    run = args.root / args.arm
    run.mkdir(parents=True, exist_ok=True)
    # Rank0 owns one file lock; rank1 joins the exact same torchrun group.
    claim = (run / 'runner.lock').open('a')
    if int(os.environ.get('RANK', '0')) == 0:
        fcntl.flock(claim, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        if args.command == 'train':
            train(args)
        elif args.command == 'timing':
            timing(args, torch.device(args.device))
        else:
            finalize(args, torch.device(args.device))
    except Exception as error:
        if int(os.environ.get('RANK', '0')) == 0:
            write_json(run / 'failure.json', dict(failed_utc=utc(), command=args.command,
                error=repr(error), host=socket.gethostname(), pid=os.getpid(), automatic_retry=False))
        raise
    finally:
        claim.close()


if __name__ == '__main__':
    main()

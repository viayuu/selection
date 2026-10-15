"""One approved R75 large-data run, validation freeze and one final TEST."""

import argparse
import gc
import hashlib
import json
import math
import os
import time
import traceback
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from code.unified_selector.registry import GLOBAL_SOLVERS, PROBLEMS
from .pairwise_objective import inputs_only
from .r40_experiment import aggregate, families
from .r43_experiment import policy_metrics
from .r64_experiment import dropout_seed
from .r64_model import INFERENCE_RULE, initialize
from .r65_official_schedulefree import AdamWScheduleFree
from .r75_capacity import DATA_ROOT, GPU_UUID, ROOT, availability, append_json, utc, write_json
from .r75_gradients import GradientLayout, update_group
from .training_monitor import capture_rng, restore_rng
from .train import configure_torch, set_seed


DIRECTORY = 'single_seed2'
BATCH = 384
EVAL_BATCH = 128
EPOCHS = 40


def atomic_save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    torch.save(value, temporary)
    temporary.replace(path)


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def status(root, phase, **values):
    previous = json.loads((root/'status.json').read_text()) if (root/'status.json').exists() else {}
    previous.update(phase=phase, heartbeat_utc=utc(), heartbeat_unix=time.time(), pid=os.getpid(), **values)
    write_json(root/'status.json', previous)


def authorize(args):
    record = json.loads((args.root/'user_authorization.json').read_text())
    if (not record.get('approved') or record['batch_size'] != BATCH or record['seed'] != 2
            or record['groups'] != 1 or record['data_root'] != str(args.data_root)):
        raise ValueError('The exact single-group seed2/batch384 run needs explicit user approval')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != GPU_UUID:
        raise RuntimeError('Only the approved GPU UUID is permitted')
    if os.environ.get('PYTORCH_CUDA_ALLOC_CONF') != 'expandable_segments:True':
        raise RuntimeError('Batch384 requires the measured expandable-segment allocator configuration')


def epoch_plan(sizes, epoch):
    generator = np.random.default_rng(np.random.SeedSequence([2, epoch, 75]))
    permutations = {p: generator.permutation(sizes[p]) for p in PROBLEMS}
    chunks = {p: math.ceil(sizes[p]/BATCH) for p in PROBLEMS}
    groups = []
    for cycle in range(max(chunks.values())):
        active = [p for p in PROBLEMS if cycle < chunks[p]]
        generator.shuffle(active)
        for start in range(0, len(active), 3):
            groups.append([(p, permutations[p][cycle*BATCH:min((cycle+1)*BATCH, sizes[p])])
                           for p in active[start:start+3]])
    return groups


def cache_loaders(cache_root, split):
    from .r75_cache import ProblemCache
    return {p: ProblemCache(cache_root/split/p) for p in PROBLEMS}


@torch.no_grad()
def evaluate(model, loaders, root, split, predictions=None):
    model.eval()
    started, results = time.monotonic(), {}
    if predictions is not None:
        predictions.mkdir(parents=True, exist_ok=True)
    for problem, loader in loaders.items():
        scores = []
        for begin in range(0, loader.size, EVAL_BATCH):
            indices = np.arange(begin, min(begin+EVAL_BATCH, loader.size))
            batch = loader.batch(indices, device='cuda:0')
            scores.append(model(inputs_only(batch))['logits'].float().cpu())
        logits = torch.cat(scores).numpy()
        raw = loader.raw
        result, pred = policy_metrics(logits, raw)
        result['native_correct'] = int(np.count_nonzero(pred == raw['winner']))
        results[problem] = result
        if predictions is not None:
            np.savez_compressed(predictions/(problem+'.npz'), indices=np.arange(loader.size),
                logits=logits, pred=pred, winner=raw['winner'], costs=raw['costs'],
                pool_ids=np.asarray(raw['pool_ids']), pool=np.asarray(raw['pool']), nodes=loader.lengths)
        status(root, 'evaluating_'+split, evaluating_problem=problem,
               evaluated_tasks=len(results), evaluation_seconds=time.monotonic()-started)
        del scores, batch
    correct, total = sum(r['native_correct'] for r in results.values()), sum(r['n'] for r in results.values())
    value = dict(per_problem=results, macro=aggregate(results), families=families(results),
                 native_correct=correct, native_total=total, seconds=time.monotonic()-started)
    if split in ('val', 'test') and (total != 18000 or any(r['n'] != 1000 for r in results.values())):
        raise ValueError('Original heldout evaluation must cover all18000 rows, with all tails')
    return value


def build_protocol(args, manifest, train_loaders):
    sizes = {p: int(train_loaders[p].size) for p in PROBLEMS}
    if sum(sizes.values()) != 1786887:
        raise ValueError('Use the approved complete-label snapshot, without truncating or oversampling')
    updates = len(epoch_plan(sizes, 0))
    sources = ('r75_experiment.py', 'r75_data.py', 'r75_cache.py', 'r75_gradients.py',
               'r75_analysis.py', 'r75_monitor.py', 'run_v4_r75.sh', 'r64_model.py',
               'r65_gradients.py', 'r65_official_schedulefree/adamw_schedulefree.py')
    return dict(seed=2, groups=1, data_root=str(args.data_root), cache_root=str(args.cache_root),
        problems=list(PROBLEMS), solver_names=list(GLOBAL_SOLVERS), sizes=sizes,
        train_rows=sum(sizes.values()), batch_size=BATCH, effective_batch_size=3*BATCH,
        eval_batch_size=EVAL_BATCH, max_epochs=EPOCHS, updates_per_epoch=updates,
        warmup_successful_updates=3*updates, rdrop_ramp_updates=3*updates,
        maximum_successful_updates=EPOCHS*updates, maximum_presentations=EPOCHS*sum(sizes.values()),
        optimizer=dict(implementation='official AdamWScheduleFree', lr=3e-4, weight_decay=1e-4,
            betas=[.9, .999], eps=1e-8, foreach=False, warmup_steps=3*updates,
            r=0., weight_lr_power=2., inner_momentum=0.),
        objective='mean two-dropout native CE + .35*ramp*legal symmetric KL',
        model='unchanged R65B NativeUtilitySelector; fresh all parameters seed2',
        no_inherited_trained_weights=True, geometry_stats=manifest['geometry_stats'],
        geometry_fit='all valid coordinate TRAIN nodes only; population moments of cached FP32 features',
        gradients='same-state FP32 arithmetic mean of actual distinct tasks, normally3; clip1; one update',
        partial_groups='late active-task cycles contain1/2 real tasks, mean over actual tasks; no dummy/repeated/dropped rows',
        sampling='per-epoch private seed2 permutation each task; shuffle active tasks each cycle; retain every tail',
        dropout=.1, rdrop_weight=.35, amp='FP16', allocator_configuration='expandable_segments:True',
        checkpoint_rule='strict maximum full18 validation integer correct; earliest exact tie',
        training_control='fixed40 full epochs, no LR/controller/early-stop parameter search',
        checkpoint_interval_updates=60, train_evaluation='full every5/final/selected, eval averaged x',
        validation_evaluation='full every epoch, eval averaged x',
        inference_rule=INFERENCE_RULE, test_rule='unique VAL-selected complete checkpoint, exactly one full18 TEST',
        wandb=dict(project='selector', entity='yjkds-southern-university-of-science-technology', mode='offline'),
        approved_gpu_uuid=GPU_UUID, old_goal='paused; this is a separate user-approved experiment',
        R74='terminated, disk files kept', legacy_B_semantics='R58 deferred/unfixed',
        limitations=['One seed and historical baseline, not matched data-only causality.',
                     'TSP/CVRP snapshot misses mainly large-size tails.'],
        source_hashes={p: digest(Path(__file__).parent/p) for p in sources})


def train(args, manifest, launch):
    directory = args.root/DIRECTORY
    directory.mkdir(parents=True, exist_ok=True)
    train_loaders, val_loaders = cache_loaders(args.cache_root, 'train'), cache_loaders(args.cache_root, 'val')
    config = build_protocol(args, manifest, train_loaders)
    if (args.root/'protocol.json').exists():
        existing = json.loads((args.root/'protocol.json').read_text())
        if existing != config:
            raise ValueError('Do not change the formal R75 protocol after launch')
    else:
        write_json(args.root/'protocol.json', config)
    write_json(directory/'args.json', config)
    model, params = initialize(config['geometry_stats'], device='cuda:0', seed=2)
    optimizer = AdamWScheduleFree(model.parameters(), **{k: v for k, v in config['optimizer'].items()
                                  if k != 'implementation'})
    scaler, layout = torch.amp.GradScaler('cuda', enabled=True), GradientLayout(model)
    history, best = [], None
    successful = presentations = amp_retries = growth = epoch_start = next_step = 0
    totals, per_task = Counter(), {}
    last_path = directory/'last.pt'
    checkpoint = None
    if last_path.exists():
        if not args.resume:
            raise ValueError('A formal checkpoint exists: do not silently duplicate training')
        checkpoint = torch.load(last_path, map_location='cuda:0', weights_only=False)
        if checkpoint['args'] != config or checkpoint['parameter_mode'] != 'averaged x':
            raise ValueError('Resume requires identical configuration and eval-mode averaged x')
        latest = None
        if (directory/'updates.jsonl').exists():
            with (directory/'updates.jsonl').open('rb') as stream:
                stream.seek(max(0, (directory/'updates.jsonl').stat().st_size-65536))
                latest = json.loads(stream.read().splitlines()[-1])
        if latest and latest['successful_update'] != checkpoint['successful_updates']:
            raise ValueError('Logged updates exceed durable checkpoint; explicit recovery accounting required')
        model.load_state_dict(checkpoint['model'], strict=True)
        optimizer.load_state_dict(checkpoint['optimizer'])
        scaler.load_state_dict(checkpoint['scaler'])
        history, best = checkpoint['history'], checkpoint['best']
        successful, presentations, amp_retries, growth = (checkpoint[k] for k in
            ('successful_updates', 'presentations', 'amp_retries', 'scale_growth_successes'))
        epoch_start, next_step = checkpoint['epoch_index'], checkpoint['next_step']
        totals, per_task = Counter(checkpoint['totals']), checkpoint['per_task']
        restore_rng(checkpoint['rng'])
    elif (directory/'updates.jsonl').exists():
        raise ValueError('Updates without a checkpoint must not be silently discarded')
    else:
        atomic_save(directory/'initial.pt', dict(model=model.state_dict(), model_params=params, seed=2,
            geometry_stats=config['geometry_stats'], no_trained_weights_loaded=True))
        write_json(directory/'initialization.json', dict(seed=2, trainable_parameters=sum(p.numel() for p in model.parameters()),
            source='fresh R65B architecture with new TRAIN normalization', inherited_trained_weights=False))
    started, run = time.monotonic(), None

    def save(epoch_index, cursor):
        was_training = model.training
        optimizer.eval()
        model.eval()
        atomic_save(last_path, dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
            scaler=scaler.state_dict(), args=config, model_params=params, parameter_mode='averaged x',
            epoch_index=epoch_index, next_step=cursor, successful_updates=successful,
            presentations=presentations, amp_retries=amp_retries, scale_growth_successes=growth,
            rng=capture_rng(), history=history, best=best, totals=dict(totals), per_task=per_task))
        status(args.root, 'training', durable_update=successful)
        if was_training:
            optimizer.train()
            model.train()

    try:
        import wandb
        run = wandb.init(**config['wandb'], name='R75_large_complete_labels_seed2', config=config,
                         dir=str(directory), id='r75-seed2-'+launch['id'], resume='allow')
        for epoch in range(epoch_start, EPOCHS):
            optimizer.train()
            model.train()
            plan = epoch_plan(config['sizes'], epoch)
            if len(plan) != config['updates_per_epoch']:
                raise ValueError('The actual task-cycle schedule changed size')
            first_step = next_step if epoch == epoch_start else 0
            if first_step == 0:
                totals, per_task = Counter(), {}
            epoch_started = time.monotonic()
            torch.cuda.reset_peak_memory_stats(0)
            for step in range(first_step, len(plan)):
                entries = plan[step]
                torch.cuda.synchronize(0)
                before = time.monotonic()
                batches = [train_loaders[p].batch(indices, device='cuda:0') for p, indices in entries]
                set_seed(dropout_seed(successful))
                weight = .35*min(1., (successful+1)/config['rdrop_ramp_updates'])
                values = update_group(model, optimizer, scaler, batches, weight, layout, growth)
                torch.cuda.synchronize(0)
                seconds = time.monotonic()-before
                growth = values['scale_growth_successes']
                rows = sum(len(indices) for _, indices in entries)
                successful += 1
                presentations += rows
                amp_retries += values['retries']
                lr = float(optimizer.param_groups[0]['scheduled_lr'])
                metrics = dict(utc=utc(), epoch=epoch+1, update_in_epoch=step+1,
                    updates_per_epoch=len(plan), successful_update=successful, presentations=presentations,
                    tasks=[p for p, _ in entries], task_rows=[len(i) for _, i in entries], rows=rows,
                    seconds=seconds, queries_per_second=rows/seconds, lr=lr,
                    consistency_weight=weight, peak_memory_gib=torch.cuda.max_memory_allocated(0)/2**30,
                    peak_reserved_gib=torch.cuda.max_memory_reserved(0)/2**30, **values)
                for key in ('loss', 'ce', 'kl'):
                    totals[key] += values[key]*rows
                totals.update(samples=rows, updates=1, grad_norm=values['grad_norm'],
                              clip_coefficient=values['clip_coefficient'], retries=values['retries'], seconds=seconds)
                for j, (problem, indices) in enumerate(entries):
                    item = per_task.setdefault(problem, dict(rows=0, batches=0, ce=0., kl=0.))
                    item['rows'] += len(indices)
                    item['batches'] += 1
                    item['ce'] += values['task_losses'][j]['ce']*len(indices)
                    item['kl'] += values['task_losses'][j]['kl']*len(indices)
                if successful == 1 or successful % 60 == 0:
                    save(epoch, step+1)
                append_json(directory/'updates.jsonl', metrics)
                if successful == 1:
                    write_json(directory/'first_formal_update.json', dict(metrics,
                        selected_indices={p: i.tolist() for p, i in entries}, launch=launch))
                if successful == 1 or successful % 60 == 0 or step+1 == len(plan):
                    status(args.root, 'training', epoch=epoch+1, update_in_epoch=step+1,
                        updates_per_epoch=len(plan), successful_updates=successful, presentations=presentations,
                        latest=metrics, best_epoch=best['epoch'] if best else None,
                        best_val_correct=best['val']['native_correct'] if best else None,
                        training_wall_seconds=time.monotonic()-started)
                    print('[R75 TRAIN] ' + json.dumps({k: metrics[k] for k in
                        ('epoch', 'update_in_epoch', 'successful_update', 'presentations', 'loss', 'ce', 'kl',
                         'lr', 'grad_norm', 'seconds', 'peak_memory_gib', 'peak_reserved_gib', 'retries')}), flush=True)
                    run.log({k: metrics[k] for k in ('epoch', 'successful_update', 'loss', 'ce', 'kl', 'lr',
                                                     'grad_norm', 'peak_memory_gib', 'queries_per_second')})
                batches = None
            if (totals['samples'] != config['train_rows'] or set(per_task) != set(PROBLEMS)
                    or any(per_task[p]['rows'] != config['sizes'][p] for p in PROBLEMS)):
                raise ValueError('Each real TRAIN instance must appear exactly once in this epoch')
            optimization = {k: totals[k]/totals['samples'] for k in ('loss', 'ce', 'kl')}
            optimization.update(grad_norm=totals['grad_norm']/totals['updates'], lr=lr,
                seconds=time.monotonic()-epoch_started, amp_retries=int(totals['retries']),
                samples=int(totals['samples']), updates=int(totals['updates']),
                peak_memory_gib=torch.cuda.max_memory_allocated(0)/2**30,
                peak_reserved_gib=torch.cuda.max_memory_reserved(0)/2**30)
            optimizer.eval()
            model.eval()
            validation = evaluate(model, val_loaders, args.root, 'val',
                                  directory/'val_predictions'/f'epoch{epoch+1:03d}')
            training = (evaluate(model, train_loaders, args.root, 'train')
                        if (epoch+1) % 5 == 0 or epoch+1 == EPOCHS else None)
            row = dict(epoch=epoch+1, successful_updates=successful, presentations=presentations,
                       optimization=optimization, train=training, val=validation, task_optimization=per_task)
            history.append(row)
            if best is None or validation['native_correct'] > best['val']['native_correct']:
                best = row
                atomic_save(directory/'best.pt', dict(model=model.state_dict(), args=config,
                    model_params=params, epoch=epoch+1, successful_updates=successful,
                    parameter_mode='averaged x', native_correct=validation['native_correct']))
                write_json(directory/'best_eval.json', best)
            write_json(directory/'history.json', history)
            totals, per_task = Counter(), {}
            save(epoch+1, 0)
            print('[R75 VAL] ' + json.dumps(dict(epoch=epoch+1, correct=validation['native_correct'],
                total=validation['native_total'], top1=validation['macro']['macro_top1'],
                ce=validation['macro']['macro_ce'], regret=validation['macro']['macro_actual_regret_pct'],
                best_epoch=best['epoch'], best_correct=best['val']['native_correct'])), flush=True)
            run.log(dict(epoch=epoch+1, successful_update=successful,
                **{'val/'+k: v for k, v in validation['macro'].items()},
                **({'train_eval/'+k: v for k, v in training['macro'].items()} if training else {})))
            status(args.root, 'training', epoch=epoch+1, epoch_complete=True,
                last_val=validation['macro'], best_epoch=best['epoch'], best_val_correct=best['val']['native_correct'])
        completion = dict(complete=True, completed_utc=utc(), epochs=EPOCHS, successful_updates=successful,
            presentations=presentations, expected_presentations=config['maximum_presentations'],
            amp_retries=amp_retries, training_wall_seconds=time.monotonic()-started,
            selected_epoch=best['epoch'], selected_val_correct=best['val']['native_correct'])
        if presentations != config['maximum_presentations'] or successful != config['maximum_successful_updates']:
            raise ValueError('Actual formal coverage differs from the fixed budget')
        write_json(args.root/'training_complete.json', completion)
        return config, best
    finally:
        if run:
            run.finish()
        del model, optimizer, scaler, layout, train_loaders, val_loaders
        gc.collect()
        torch.cuda.empty_cache()


def final_test(args, config, best):
    from .r75_cache import prepare_cache

    directory = args.root/DIRECTORY
    checkpoint_path = directory/'best.pt'
    lock_path = args.root/'final_test_lock.json'
    if lock_path.exists() or (args.root/'test_started.json').exists():
        raise ValueError('A final lock/test already exists; no automatic repeated TEST')
    lock = dict(locked_utc=utc(), checkpoint=str(checkpoint_path), checkpoint_sha256=digest(checkpoint_path),
        epoch=best['epoch'], val_correct=best['val']['native_correct'], val_total=18000,
        rule=config['checkpoint_rule'], inference_rule=INFERENCE_RULE, selected_without_test=True,
        groups=1, no_per_problem_assembly=True)
    write_json(lock_path, lock)
    status(args.root, 'preparing_locked_test', selected_epoch=best['epoch'])
    prepare_cache(args.root, args.data_root, args.cache_root, device='cuda:0', splits=('test',))
    model, _ = initialize(config['geometry_stats'], device='cuda:0', seed=2)
    checkpoint = torch.load(checkpoint_path, map_location='cuda:0', weights_only=False)
    model.load_state_dict(checkpoint['model'], strict=True)
    model.eval()
    train_loaders = cache_loaders(args.cache_root, 'train')
    selected_train = evaluate(model, train_loaders, args.root, 'selected_train')
    best = dict(best, train=selected_train)
    write_json(directory/'best_eval.json', best)
    del train_loaders
    loaders = cache_loaders(args.cache_root, 'test')
    write_json(args.root/'test_started.json', dict(started_utc=utc(), lock=lock,
        checkpoint_sha256=digest(checkpoint_path), previously_executed_tests=0))
    result = evaluate(model, loaders, args.root, 'test', directory/'test_predictions')
    write_json(args.root/'test_results.json', dict(model='R75', epoch=best['epoch'], checkpoint=str(checkpoint_path),
        checkpoint_sha256=lock['checkpoint_sha256'], result=result, complete=True, completed_utc=utc()))
    print('[R75 TEST] ' + json.dumps(dict(correct=result['native_correct'], total=result['native_total'],
        top1=result['macro']['macro_top1'], actual_regret=result['macro']['macro_actual_regret_pct'])), flush=True)
    del model, checkpoint, loaders
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--data-root', type=Path, default=DATA_ROOT)
    parser.add_argument('--cache-root', type=Path, default=Path('/tmp/shiys_r75_large_complete_labels_cache'))
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    args.root, args.data_root, args.cache_root = args.root.resolve(), args.data_root.resolve(), args.cache_root.resolve()
    authorize(args)
    if (args.root/'training_complete.json').exists() and not args.resume:
        raise ValueError('This single formal training already completed; do not start another group')
    device = availability()
    configure_torch()
    torch.cuda.set_device(0)
    launch = dict(id=utc().replace(':', '').replace('-', ''), started_utc=utc(), device=device,
                  batch=BATCH, effective_batch=3*BATCH, seed=2, groups=1, resume=args.resume)
    write_json(args.root/'formal_started.json', launch)
    status(args.root, 'preparing_train_val_cache', launch=launch, formal_training_approved=True,
           test_started=False, training_complete=False, old_goal_resumed=False)
    try:
        from .r75_cache import prepare_cache
        manifest = prepare_cache(args.root, args.data_root, args.cache_root, device='cuda:0', splits=('train', 'val'))
        torch.cuda.empty_cache()
        config, best = train(args, manifest, launch)
        result = final_test(args, config, best)
        status(args.root, 'analyzing', training_complete=True, test_complete=True)
        from .r75_analysis import analyze
        analyze(args.root)
        status(args.root, 'complete', complete=True, completed_utc=utc(),
               selected_epoch=best['epoch'], val_correct=best['val']['native_correct'],
               test_correct=result['native_correct'], test_top1=result['macro']['macro_top1'])
        write_json(args.root/'pipeline_complete.json', dict(complete=True, utc=utc(), launch=launch,
            selected_epoch=best['epoch'], validation_correct=best['val']['native_correct'],
            test_correct=result['native_correct'], official_test_passes=1, groups=1))
    except BaseException as error:
        failure = dict(utc=utc(), error=repr(error), traceback=traceback.format_exc(), launch=launch,
                       automatic_restart=False, change_batch=False)
        write_json(args.root/'failure.json', failure)
        status(args.root, 'failed', failed=True, error=repr(error))
        raise


if __name__ == '__main__':
    main()

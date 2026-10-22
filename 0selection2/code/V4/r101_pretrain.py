"""TRAIN-only per-node hint initialization of the deployed original encoder.

This stage never constructs a solver, uses outcome labels, or trains a side
head. The separate downstream runner resets the optimizer and has no hints.
"""

from collections import Counter
import math
from pathlib import Path
import time

import torch
import torch.nn.functional as F

from code.unified_selector.registry import PROBLEMS
from .multitask_probe import task_seed
from .r76_experiment import (PinnedPrefetch, capture_rank_rng, restore_rank_rng,
    save_checkpoint, read_json, write_json, file_hash, utc)
from .r77_data import parameter_hash
from .r77_experiment import ShuffleStream, task_order
from .r80_reporting import json_hash
from .r88_model import make_model, prepare_role_inputs
from .train import set_seed


COVERED = tuple(p for p in PROBLEMS if p not in ('TSP', 'ATSP'))
VARIANTS = dict(A='random', B='pretrained')
UPDATES, BATCH_SIZE, WARMUP = 12800, 1024, 960
SPEC = dict(stage='TRAIN_node_hint_pretraining', teacher_width=128,
    loss='separate non-affine per-node LayerNorm MSE; channel/node/instance mean',
    teacher_scope=list(COVERED), teacher_split='train', label_inputs=False,
    updates=UPDATES, logical_batch=BATCH_SIZE, warmup=WARMUP,
    optimizer='AdamW', peak_lr=1e-4, end_lr=1e-6, weight_decay=1e-4,
    grad_clip=1., dropout=.1, dropout_passes=1, learned_adapter=False,
    optimizer_scope='instance_encoder only', endpoint='fixed budget; no VAL selection')


def learning_rate(update):
    if not 1 <= update <= UPDATES:
        raise ValueError('Hint update is one-based and bounded')
    if update <= WARMUP:
        return 1e-4*update/WARMUP
    progress = (update-WARMUP)/(UPDATES-WARMUP)
    return 1e-6 + .5*(1e-4-1e-6)*(1+math.cos(math.pi*progress))


def initialize_fresh(initial, geometry, device='cpu'):
    if (initial.get('seed') != 2 or initial.get('initialization') != 'scratch'
            or parameter_hash(initial['trainable']) != initial.get('trainable_sha256')):
        raise ValueError('Use only the fresh R84 initialization, never a trained selector')
    set_seed(2)
    selector = make_model(initial['model_params'], 'B_execution_anchor')
    actual = dict(selector.named_parameters())
    if set(actual) != set(initial['trainable']):
        raise ValueError('Hint transfer must not add a head or change student capacity')
    with torch.no_grad():
        for name, value in actual.items():
            value.copy_(initial['trainable'][name])
        branch = selector.instance_encoder.geometry_residual
        branch.mean.copy_(torch.as_tensor(geometry['mean'], dtype=branch.mean.dtype))
        branch.std.copy_(torch.as_tensor(geometry['std'], dtype=branch.std.dtype))
    if parameter_hash(actual) != initial['trainable_sha256']:
        raise ValueError('Original fresh student parameters changed')
    return selector.to(device)


def non_encoder_hash(selector):
    return parameter_hash({name:value for name,value in selector.named_parameters()
                           if not name.startswith('instance_encoder.')})


def hint_loss(student, teacher, mask):
    if (student.shape != teacher.shape or student.shape[:2] != mask.shape
            or student.shape[-1] != 128 or mask.dtype != torch.bool
            or not mask.any(1).all() or not torch.isfinite(teacher[mask]).all()):
        raise ValueError('Teacher and student must align on every valid node/channel')
    target = F.layer_norm(teacher.detach().float(), (128,))
    prediction = F.layer_norm(student.float(), (128,))
    per_node = (prediction-target).square().mean(-1)
    return ((per_node*mask).sum(1)/mask.sum(1)).mean()


def raw_inputs(batch):
    names = ('node', 'node_mask', 'n', 'node_geom', 'kind', 'problem_id',
             'coord_dist', 'cbits', 'problem_desc', 'd_in')
    return prepare_role_inputs({name:batch[name] for name in names if name in batch},
                               'B_execution_anchor')


def step(encoder, optimizer, scaler, batch, cap, max_retries=20):
    count = len(batch['n'])
    device = batch['node_mask'].device
    rng = capture_rank_rng(device)
    if count < 1 or cap < 1 or batch['teacher_nodes'].shape[:2] != batch['node_mask'].shape:
        raise ValueError('Expected a nonempty aligned logical hint batch')
    for retry in range(max_retries):
        restore_rank_rng(rng, device)
        optimizer.zero_grad(set_to_none=True)
        weighted_loss = torch.zeros((), dtype=torch.float64, device=device)
        for start in range(0, count, cap):
            stop = min(start+cap, count)
            micro = {key:(value[start:stop] if torch.is_tensor(value) and value.ndim
                         and value.shape[0] == count else value) for key,value in batch.items()}
            teacher_mask = micro.get('teacher_node_mask')
            if not torch.is_tensor(teacher_mask) or not torch.equal(teacher_mask, micro['node_mask']):
                raise ValueError('Teacher node mask cannot differ from original inputs')
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == 'cuda'):
                hidden, mask = encoder(raw_inputs(micro))
            loss = hint_loss(hidden, micro['teacher_nodes'], mask)
            scaler.scale(loss*((stop-start)/count)).backward()
            weighted_loss.add_(loss.detach().double(), alpha=(stop-start)/count)
        scaler.unscale_(optimizer)
        norm = torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.)
        if not bool(torch.isfinite(norm) & torch.isfinite(weighted_loss)):
            if scaler.is_enabled():
                scaler.update(new_scale=scaler.get_scale()*.5)
            continue
        scaler.step(optimizer)
        scaler.update()
        return dict(loss=float(weighted_loss), grad_norm=float(norm), retries=retry)
    raise RuntimeError('Nonfinite hint gradients persisted; no successful update or row skip')


def pretrain(args, device):
    from .r101_features import HintInputCache
    device = torch.device(device)
    if device.type != 'cuda' or not torch.cuda.is_available():
        raise RuntimeError('Formal hint pretraining requires the assigned CUDA device')
    root = Path(args.root)/args.arm
    lock_path = Path(args.protocol_lock or Path(args.root)/'protocol_lock.json')
    lock = read_json(lock_path)
    if (args.arm not in VARIANTS or lock.get('hint_pretraining') != SPEC
            or lock.get('data_root') != str(Path(args.data_root).resolve())
            or lock.get('locked') is not True):
        raise ValueError('R101 hint source and full fixed pretraining spec must be locked')
    prepared_root = Path(args.prepared_root)
    prepared = read_json(prepared_root/'prepared.json')
    if (prepared.get('seed') != 2 or prepared.get('rows_per_problem') != dict(train=81000, val=4500, test=4500)
            or prepared.get('data_root') != str(Path(args.data_root).resolve())):
        raise ValueError('Use the frozen complete R84 NEW90k training preparation')
    for name in ('identicalinit_seed2.pt', 'geometry_stats.json'):
        if file_hash(prepared_root/name) != prepared.get('artifact_sha256', {}).get(name):
            raise ValueError('Fresh initialization or TRAIN geometry statistics changed')
    end_path, complete_path = root/'pretrain_end.pt', root/'pretrain_complete.json'
    if complete_path.exists():
        saved = read_json(complete_path)
        if (saved.get('complete') is not True or saved.get('successful_updates') != UPDATES
                or saved.get('checkpoint_sha256') != file_hash(end_path)
                or saved.get('protocol_sha256') != file_hash(lock_path)):
            raise ValueError('Preserve an incompatible pretraining endpoint')
        return saved
    if (root/'pretrain_attempt.json').exists() or end_path.exists():
        raise RuntimeError('Preserve interrupted hint pretraining; no implicit retry')
    initial = torch.load(Path(args.prepared_root)/'identicalinit_seed2.pt', map_location='cpu', weights_only=False)
    geometry = read_json(Path(args.prepared_root)/'geometry_stats.json')
    selector = initialize_fresh(initial, geometry, device)
    initial_hash = initial['trainable_sha256']
    del initial
    other_hash = non_encoder_hash(selector)
    selector.train()
    caches, tracker = {}, None
    started = time.monotonic()
    order = task_order(UPDATES, 2, COVERED)
    streams = {p:ShuffleStream(81000, task_seed(2, p)) for p in COVERED}
    counts, retries = Counter(), 0
    try:
        caches = {p:HintInputCache(args.cache_root, args.teacher_root, p, VARIANTS[args.arm]) for p in COVERED}
        if any(cache.size != 81000 for cache in caches.values()):
            raise ValueError('Hint pretraining uses every one of the81000 TRAIN instances per covered task')
        teacher_records = {p:read_json(Path(args.teacher_root)/'train'/p/'metadata.json') for p in COVERED}
        teacher_identity = dict(variant=VARIANTS[args.arm], split='train', rows_per_task=81000,
            metadata_sha256={p:json_hash(value) for p,value in teacher_records.items()},
            donor={p:value['donor'] for p,value in teacher_records.items()},
            interface_coverage_not_pretraining_task_coverage=True,
            no_teacher_in_final_model=True)
        write_json(root/'pretrain_attempt.json', dict(started_utc=utc(), automatic_retry=False,
            protocol_sha256=file_hash(lock_path), teacher_identity=teacher_identity,
            spec=SPEC, initial_trainable_sha256=initial_hash))
        import wandb
        tracker = wandb.init(project='selector', entity='yjkds-southern-university-of-science-technology',
            name='R101_'+args.arm+'_node_hint_pretraining', mode='offline', dir=str(root),
            config=dict(experiment='R101', arm=args.arm, spec=SPEC, teacher_identity=teacher_identity))
        write_json(root/'pretrain_wandb_receipt.json', dict(enabled=True, mode='offline', run_id=tracker.id,
            path=str(tracker.dir), project='selector', entity='yjkds-southern-university-of-science-technology'))
        optimizer = torch.optim.AdamW(selector.instance_encoder.parameters(), lr=1e-4, weight_decay=1e-4)
        scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda')
        set_seed(2)
        requests = ((problem, streams[problem].take(BATCH_SIZE), BATCH_SIZE) for problem in order)
        with (root/'pretrain_updates.jsonl').open('a', buffering=1) as log, PinnedPrefetch(caches, requests, device) as batches:
            for update, (problem, count, total, batch) in enumerate(batches, 1):
                if count != BATCH_SIZE or total != BATCH_SIZE:
                    raise ValueError('Hint batches must preserve all1024 real instances')
                for group in optimizer.param_groups:
                    group['lr'] = learning_rate(update)
                parts = step(selector.instance_encoder, optimizer, scaler, batch,
                             352 if problem == 'CVRP' else BATCH_SIZE)
                retries += parts['retries']
                counts[problem] += 1
                if update % 16 == 0 or update == 1:
                    record = dict(experiment='R101', arm=args.arm, phase='hint_pretraining',
                        successful_updates=update, max_updates=UPDATES, problem=problem,
                        loss=parts['loss'], grad_norm=parts['grad_norm'], amp_retries=retries,
                        lr=learning_rate(update), sample_exposures=update*BATCH_SIZE,
                        updates_by_problem=dict(counts), seconds_per_update=(time.monotonic()-started)/update,
                        updated_utc=utc())
                    import json
                    log.write(json.dumps(record)+'\n')
                    print('[R101 hint]', json.dumps(record), flush=True)
                    write_json(root/'pretrain_status.json', record)
                    tracker.log({'hint/'+key:value for key,value in record.items()
                                 if isinstance(value, (int, float))}, step=update)
                if update % 1600 == 0:
                    save_checkpoint(root/'pretrain_last.pt', dict(instance_encoder=selector.instance_encoder.state_dict(),
                        optimizer=optimizer.state_dict(), scaler=scaler.state_dict(), successful_updates=update,
                        arm=args.arm, protocol_sha256=file_hash(lock_path)))
        if non_encoder_hash(selector) != other_hash:
            raise RuntimeError('A non-encoder parameter changed during hint training')
        encoder_state = {key:value.detach().cpu() for key,value in selector.instance_encoder.state_dict().items()}
        encoder_hash = parameter_hash(encoder_state)
        endpoint = dict(experiment='R101', arm=args.arm, complete=True, successful_updates=UPDATES,
            instance_encoder=encoder_state, encoder_sha256=encoder_hash,
            initial_trainable_sha256=initial_hash, unchanged_non_encoder_sha256=other_hash,
            teacher_identity=teacher_identity, pretrain_spec=SPEC, protocol_sha256=file_hash(lock_path))
        save_checkpoint(end_path, endpoint)
        result = dict(experiment='R101', arm=args.arm, complete=True, successful_updates=UPDATES,
            checkpoint=str(end_path), checkpoint_sha256=file_hash(end_path), encoder_sha256=encoder_hash,
            protocol_sha256=file_hash(lock_path), initial_trainable_sha256=initial_hash,
            unchanged_non_encoder_sha256=other_hash, teacher_identity=teacher_identity,
            pretrain_spec=SPEC, updates_by_problem=dict(counts), amp_retries=retries,
            seconds=time.monotonic()-started, completed_utc=utc(), selected_without_validation=True)
        write_json(complete_path, result)
        write_json(root/'pretrain_status.json', dict(result, phase='hint_pretraining_complete'))
        return result
    except BaseException as error:
        write_json(root/'pretrain_status.json', dict(experiment='R101', arm=args.arm,
            phase='failed', successful_updates=sum(counts.values()), error=str(error),
            automatic_retry=False, updated_utc=utc()))
        raise
    finally:
        for cache in caches.values():
            cache.close()
        if tracker is not None:
            tracker.finish()

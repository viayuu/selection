"""TRAIN-only R113 processor pretraining, with an immutable full18 manifest.

The processor is extracted from the complete selector's shared initialization.
No teacher state, native winner, candidate cost, or trained selector is an input.
"""

import copy
import fcntl
import hashlib
import json
import math
import sys
import time
from collections import Counter, OrderedDict
from pathlib import Path

import numpy as np
import torch

from . import r113_process_data as data
from .r76_experiment import (capture_rank_rng, file_hash, read_json, restore_rank_rng,
                             save_checkpoint, utc, write_json)
from .r77_data import parameter_hash
from .r77_experiment import task_order
from .r80_experiment import json_hash, wandb_run
from .r113_processor import process_loss


ARMS = ('B_snapshot', 'C_transition')
BATCH_SIZE, MAX_UPDATES, WARMUP_UPDATES = 512, 8640, 432
PEAK_LR, END_LR, SAVE_EVERY = 3e-4, 1e-5, 432
LOSS_KEYS = ('scalar_state', 'scalar_transition', 'binary_state', 'pointer_state',
             'compatible', 'reached', 'frontier')
MODE = dict(B_snapshot='snapshot', C_transition='transition')
SOURCE_FILES = ('r113_pretrain.py', 'r113_experiment.py', 'r113_model.py', 'r113_processor.py', 'r113_process_data.py')
V4 = Path(__file__).resolve().parent
STREAM_RULE = 'seed2 task round-robin shuffle; per-task shuffled shards/rows; carry cycle tails'


def required_fields():
    return dict(arms=list(ARMS), logical_global_batch=BATCH_SIZE, max_successful_updates=MAX_UPDATES,
        warmup_updates=WARMUP_UPDATES, peak_lr=PEAK_LR, end_lr=END_LR, optimizer='AdamW',
        weight_decay=1e-4, grad_clip=1., checkpoint_every_updates=SAVE_EVERY,
        loss_weights=dict.fromkeys(LOSS_KEYS, 1.), stream_rule=STREAM_RULE,
        checkpoint_rule='last successful update; no validation or checkpoint selection',
        processor_only=True, teacher_forcing=False, context=None)


def learning_rate(update):
    if not 1 <= update <= MAX_UPDATES:
        raise ValueError('Use a one-based successful process update')
    if update <= WARMUP_UPDATES:
        return PEAK_LR * update / WARMUP_UPDATES
    fraction = (update - WARMUP_UPDATES) / (MAX_UPDATES - WARMUP_UPDATES)
    return END_LR + .5 * (PEAK_LR - END_LR) * (1. + math.cos(math.pi * fraction))


def tensor_schema(state):
    return {key: dict(shape=list(value.shape), dtype=str(value.dtype)) for key, value in state.items()}


class ProcessDataset:
    """Bounded NPZ reader. A building/subset manifest is never a training set.

    Shard hashes are verified on first access, not by rescanning the raw corpus.
    Source IDs are checked against frozen TRAIN input_metadata, including resume.
    """

    def __init__(self, manifest_path, data_root, *, rows_per_problem=81000, cache_shards=36):
        self.path = Path(manifest_path).resolve()
        self.root, self.manifest = self.path.parent, read_json(self.path)
        m, total = self.manifest, len(data.PROBLEMS) * rows_per_problem
        if (m.get('status') != 'complete' or m.get('split') != 'train' or m.get('seed') != 2
                or m.get('version') != data.VERSION or m.get('schema') != data.schema()
                or m.get('rows') != total or m.get('total_expected_rows') != total
                or m.get('native_solver_calls') != 0
                or m.get('spec') != dict(max_steps=12, neighbors=2)):
            raise ValueError('Require complete, fixed full18 x 81000 TRAIN process manifest')
        completion = read_json(self.root / 'completion_receipt.json')
        source = read_json(self.root / 'source_manifest.json')
        source_hash = file_hash(self.root / 'source_manifest.json')
        if (completion.get('status') != 'complete' or completion.get('manifest_sha256') != file_hash(self.path)
                or completion.get('source_manifest_sha256') != source_hash
                or m.get('source_manifest_sha256') != source_hash
                or source.get('sources') != m.get('sources') or source.get('seed') != 2
                or source.get('split') != 'train' or source.get('scenario_id') != data.SCENARIO
                or completion.get('per_problem_rows') != dict.fromkeys(data.PROBLEMS, rows_per_problem)
                or completion.get('schema_sha256') != file_hash(self.root / 'schema.json')
                or m.get('code_sha256', {}).get('r113_process_data.py') != file_hash(V4 / 'r113_process_data.py')):
            raise ValueError('Process completion/source/schema identity differs')
        self.rows, self.by_problem, self.source_ids = rows_per_problem, {}, {}
        identities = m['sources']
        if len(identities) != len(data.PROBLEMS) or {v['problem'] for v in identities} != set(data.PROBLEMS):
            raise ValueError('Process manifest must name every TRAIN source exactly once')
        for identity in identities:
            problem = identity['problem']
            source_path = Path(data_root).resolve() / (problem + 'train') / 'input_metadata.npz'
            if (identity.get('split') != 'train' or identity.get('rows') != rows_per_problem
                    or identity.get('scenario_id') != data.SCENARIO
                    or Path(identity['data_root']).resolve() != Path(data_root).resolve()
                    or file_hash(source_path) != identity.get('source_ids_sha256')):
                raise ValueError('Frozen TRAIN source identity differs: ' + problem)
            with np.load(source_path, allow_pickle=False) as archive:
                ids = np.array(archive['source_index'], dtype=np.int64, copy=True)
            if ids.shape != (rows_per_problem,) or len(np.unique(ids)) != rows_per_problem:
                raise ValueError('TRAIN source/base IDs are not unique and aligned')
            self.source_ids[problem] = ids
        paths = set()
        if len(m['shards']) != completion.get('shards'):
            raise ValueError('Process shard count differs from completion')
        for problem in data.PROBLEMS:
            shards = sorted((s for s in m['shards'] if s['problem'] == problem), key=lambda s: s['start_row'])
            stop = 0
            for shard in shards:
                path = (self.root / shard['path']).resolve()
                if (not path.is_relative_to((self.root / 'shards').resolve()) or path in paths
                        or shard.get('split') != 'train' or shard.get('seed') != 2
                        or shard.get('source_manifest_sha256') != source_hash
                        or shard['start_row'] != stop or not stop < shard['stop_row'] <= rows_per_problem
                        or shard['rows'] != shard['stop_row'] - stop
                        or not path.is_file() or path.stat().st_size != shard['bytes']):
                    raise ValueError('Incomplete/overlapping/foreign process shard: ' + str(path))
                paths.add(path)
                stop = shard['stop_row']
            if stop != rows_per_problem:
                raise ValueError('Missing TRAIN process rows: ' + problem)
            self.by_problem[problem] = shards
        if len(paths) != len(m['shards']):
            raise ValueError('Process manifest contains extra or unknown task shards')
        self.identity = dict(manifest=str(self.path), manifest_sha256=file_hash(self.path),
            source_manifest_sha256=source_hash, schema_sha256=json_hash(m['schema']),
            tensor_schema_sha256=file_hash(self.root / 'schema.json'),
            generator_sha256=m['code_sha256']['r113_process_data.py'], rows=total,
            rows_per_problem=rows_per_problem, split='train', program_version=data.VERSION,
            spec=m['spec'], source_policy=source.get('source_policy'))
        self.cache, self.verified = OrderedDict(), set()
        if cache_shards < 1:
            raise ValueError('Positive bounded shard-cache size required')
        self.cache_shards = cache_shards

    def _load(self, shard):
        path = (self.root / shard['path']).resolve()
        if path in self.cache:
            self.cache.move_to_end(path)
            return self.cache[path]
        if path not in self.verified and file_hash(path) != shard['sha256']:
            raise ValueError('Immutable process shard hash differs: ' + str(path))
        with np.load(path, allow_pickle=False) as archive:
            expected = ({'input__' + k for k in data.INPUT_KEYS} | {'target__' + k for k in data.TARGET_KEYS}
                | {'row_index', 'source_index', 'input_sha256', 'version', 'split', 'seed', 'source_manifest_sha256'})
            if (set(archive.files) != expected or str(archive['version']) != data.VERSION
                    or str(archive['split']) != 'train' or int(archive['seed']) != 2
                    or str(archive['source_manifest_sha256']) != self.identity['source_manifest_sha256']):
                raise ValueError('Process shard has foreign fields, labels or provenance')
            program = {k: torch.from_numpy(archive['input__' + k].copy()) for k in data.INPUT_KEYS}
            targets = {k: torch.from_numpy(archive['target__' + k].copy()) for k in data.TARGET_KEYS}
            ids = {k: archive[k].copy() for k in ('row_index', 'source_index', 'input_sha256')}
        rows = np.arange(shard['start_row'], shard['stop_row'], dtype=np.int64)
        p = shard['problem']
        if (not np.array_equal(ids['row_index'], rows)
                or not np.array_equal(ids['source_index'], self.source_ids[p][rows])
                or ids['input_sha256'].shape != rows.shape or ids['input_sha256'].dtype != np.dtype('S64')
                or any(len(bytes(v)) != 64 or any(c not in b'0123456789abcdef' for c in bytes(v))
                       for v in ids['input_sha256'])
                or any(v.shape[0] != len(rows) for v in (*program.values(), *targets.values()))
                or not torch.equal(program['problem_id'], torch.full((len(rows),), data.PROBLEMS.index(p)))):
            raise ValueError('Process rows/source base IDs/task tensors differ: ' + p)
        self._validate_tensors(program, targets, len(rows))
        self.verified.add(path)
        self.cache[path] = program, targets, ids
        while len(self.cache) > self.cache_shards:
            self.cache.popitem(last=False)
        return self.cache[path]

    @staticmethod
    def _validate_tensors(program, targets, b):
        k, g, f, h = 12, len(data.GRAPH_FIELDS), len(data.SCALAR_FIELDS), len(data.BINARY_FIELDS)
        shapes = dict(nodes=(b, k, len(data.NODE_FIELDS)), anchor=(b, len(data.NODE_FIELDS)),
            edge_features=(b, k, k, 2), topology=(b, k, k), node_mask=(b, k), sequence=(b, k),
            anchor_index=(b,), steps=(b,), n=(b,), problem_id=(b,), coord_dist=(b,),
            task_flags=(b, len(data.TASK_FIELDS)), graph_mask=(b, g), scalar_mask=(b, f), binary_mask=(b, h))
        float_keys = {'nodes', 'anchor', 'edge_features'}
        int_keys = {'sequence', 'anchor_index', 'steps', 'n', 'problem_id', 'coord_dist'}
        for key, shape in shapes.items():
            dtype = torch.float32 if key in float_keys else torch.int64 if key in int_keys else torch.bool
            if program[key].shape != shape or program[key].dtype != dtype:
                raise ValueError('Process input tensor schema differs: ' + key)
        shapes = dict(scalar=(b, k+1, f), binary=(b, k+1, h), pointer=(b, k+1),
            reached=(b, k+1, g, k), frontier=(b, k+1, g, k), compatible=(b, g, k, k),
            scalar_delta=(b, k, f), state_mask=(b, k+1), transition_mask=(b, k))
        for key, shape in shapes.items():
            dtype = torch.float32 if key in ('scalar', 'scalar_delta') else torch.int64 if key == 'pointer' else torch.bool
            if targets[key].shape != shape or targets[key].dtype != dtype:
                raise ValueError('Process target tensor schema differs: ' + key)
        if (not torch.equal(program['steps'], program['node_mask'].sum(-1))
                or not torch.equal(targets['transition_mask'], torch.arange(k)[None] < program['steps'][:, None])
                or not torch.equal(targets['state_mask'], torch.arange(k+1)[None] <= program['steps'][:, None])):
            raise ValueError('Teacher masks cannot define a different execution schedule')

    def batch(self, problem, rows, device='cpu'):
        rows = np.asarray(rows, dtype=np.int64)
        if rows.ndim != 1 or not len(rows) or (rows < 0).any() or (rows >= self.rows).any():
            raise ValueError('Nonempty TRAIN row indices required')
        fragments = []
        for shard in self.by_problem[problem]:
            slots = np.flatnonzero((rows >= shard['start_row']) & (rows < shard['stop_row']))
            if len(slots):
                program, targets, ids = self._load(shard)
                local = rows[slots] - shard['start_row']
                fragments.append((slots, {k: v[local] for k, v in program.items()},
                    {k: v[local] for k, v in targets.items()}, {k: v[local] for k, v in ids.items()}))
        joined = np.concatenate([part[0] for part in fragments])
        order = np.argsort(joined, kind='stable')
        program = {k: torch.cat([f[1][k] for f in fragments])[order].to(device) for k in data.INPUT_KEYS}
        targets = {k: torch.cat([f[2][k] for f in fragments])[order].to(device) for k in data.TARGET_KEYS}
        ids = {k: np.concatenate([f[3][k] for f in fragments])[order]
               for k in ('row_index', 'source_index', 'input_sha256')}
        # The input-derived sequence is retained in program; never substitute a route.
        return program, targets, ids


class ProcessStream:
    def __init__(self, dataset, problem, draws=0):
        if draws < 0:
            raise ValueError('Negative process cursor')
        self.dataset, self.problem, self.draws = dataset, problem, draws
        self.cycle, self.permutation = None, None

    def take(self, count=BATCH_SIZE):
        if count < 1:
            raise ValueError('Positive process batch required')
        parts = []
        while count:
            cycle, offset = divmod(self.draws, self.dataset.rows)
            if cycle != self.cycle:
                rng = np.random.default_rng(np.random.SeedSequence([2, data.PROBLEMS.index(self.problem), cycle]))
                shards = self.dataset.by_problem[self.problem]
                self.permutation = np.concatenate([rng.permutation(np.arange(shards[i]['start_row'],
                    shards[i]['stop_row'], dtype=np.int64)) for i in rng.permutation(len(shards))])
                self.cycle = cycle
            size = min(count, self.dataset.rows - offset)
            parts.append(self.permutation[offset:offset + size])
            self.draws += size
            count -= size
        return np.concatenate(parts)


def extract_processor(initial, geometry, arm, device='cpu'):
    from .r113_experiment import initialize_shared
    model, params, receipt = initialize_shared(initial, geometry, arm, 'cpu')
    processor = model.instance_encoder.processor
    receipt = dict(receipt, initial_processor_sha256=parameter_hash(processor.state_dict()),
        processor_parameter_schema=tensor_schema(processor.state_dict()), model_params=params)
    del model
    return processor.to(device), receipt


def update_batch(processor, optimizer, scaler, program, targets, mode, cap, *, max_retries=20):
    count, device = len(program['steps']), program['steps'].device
    if count < 1 or cap < 1 or mode not in MODE.values() or not processor.training:
        raise ValueError('Training-mode processor and positive real batch/cap required')
    rng = capture_rank_rng(device)
    for retry in range(max_retries):
        restore_rank_rng(rng, device)
        optimizer.zero_grad(set_to_none=True)
        sums = dict.fromkeys((*LOSS_KEYS, 'total'), None)
        bad = False
        for start in range(0, count, cap):
            stop = min(start + cap, count)
            p = {k: v[start:stop] for k, v in program.items()}
            y = {k: v[start:stop] for k, v in targets.items()}
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == 'cuda'):
                outputs = processor.run(p, context=None, mode=mode)
            losses = process_loss(outputs, y, p, weights=dict.fromkeys(LOSS_KEYS, 1.))
            factor = (stop - start) / count
            scaler.scale(losses['total'] * factor).backward()
            bad |= not bool(torch.isfinite(losses['total']))
            for key, value in losses.items():
                value = value.detach().double() * factor
                sums[key] = value if sums[key] is None else sums[key] + value
        scaler.unscale_(optimizer)
        norm = torch.nn.utils.clip_grad_norm_(processor.parameters(), 1.)
        if bad or not bool(torch.isfinite(norm)):
            if scaler.is_enabled():
                scaler.update(new_scale=scaler.get_scale() * .5)
            continue
        scaler.step(optimizer)
        scaler.update()
        return {k: float(v) for k, v in sums.items()}, retry, float(norm)
    raise RuntimeError('Nonfinite process gradients persisted; no update or sample skip accepted')


def config_for(args, dataset, receipt):
    from . import r113_experiment as runner
    lock = runner.protocol_lock(args)
    if (str(dataset.path) != str(Path(lock['record']['process_manifest']).resolve())
            or dataset.identity['manifest_sha256'] != lock['record']['process_manifest_sha256']):
        raise ValueError('Main lock must bind the complete fixed process manifest')
    return dict(experiment='R113_pretrain', arm=args.arm, seed=2, **required_fields(),
        mode=MODE[args.arm], dataset=dataset.identity, main_protocol_lock=lock,
        initial_full_state_sha256=receipt['fresh_state_sha256'],
        initial_processor_sha256=receipt['initial_processor_sha256'],
        fresh_reference_sha256=receipt['fresh_reference_sha256'],
        processor_parameter_schema=receipt['processor_parameter_schema'], r113_spec=receipt['r113_spec'],
        microbatch_caps=lock['record']['pretrain_microbatch_caps'],
        task_order_sha256=hashlib.sha256('\n'.join(task_order(MAX_UPDATES)).encode('ascii')).hexdigest(),
        source_sha256={name: file_hash(V4 / name) for name in SOURCE_FILES},
        model_source='full selector initialize_shared; extract instance_encoder.processor',
        query_labels_in_forward=False, native_solver_calls=0, wandb=True, wandb_mode=args.wandb_mode)


def validate_export(value, model, receipt, arm, *, manifest_sha256=None, main_lock=None):
    config = value.get('args', {})
    expected = dict(experiment='R113_pretrain', arm=arm, seed=2, mode=MODE.get(arm), **required_fields())
    processor = model.instance_encoder.processor
    if (arm not in ARMS or value.get('format') != 'r113_processor_only_v1'
            or value.get('complete') is not True or value.get('successful_updates') != MAX_UPDATES
            or value.get('state_hash') != json_hash(config)
            or any(config.get(k) != v for k, v in expected.items())
            or value.get('processor_sha256') != parameter_hash(value['processor'])
            or tensor_schema(value['processor']) != tensor_schema(processor.state_dict())
            or config.get('processor_parameter_schema') != tensor_schema(processor.state_dict())
            or config.get('initial_full_state_sha256') != receipt['fresh_state_sha256']
            or config.get('initial_processor_sha256') != parameter_hash(processor.state_dict())
            or config.get('fresh_reference_sha256') != receipt['fresh_reference_sha256']
            or config.get('r113_spec') != json.loads(json.dumps(receipt['r113_spec']))
            or value.get('updates_by_problem') != dict.fromkeys(data.PROBLEMS, MAX_UPDATES // len(data.PROBLEMS))
            or value.get('stream_draws') != dict.fromkeys(data.PROBLEMS, MAX_UPDATES // len(data.PROBLEMS) * BATCH_SIZE)
            or manifest_sha256 is not None and config.get('dataset', {}).get('manifest_sha256') != manifest_sha256
            or main_lock is not None and config.get('main_protocol_lock') != main_lock):
        raise ValueError('Processor export differs from full-model initial/schema/source/fixed process budget')
    if config.get('source_sha256') != {name: file_hash(V4 / name) for name in SOURCE_FILES}:
        raise ValueError('Processor export source hashes differ')
    return config


def completion(value, path):
    config = value['args']
    return dict(experiment='R113_pretrain', arm=config['arm'], complete=True, completed_utc=utc(),
        successful_updates=value['successful_updates'], processor_path=str(path), processor_file_sha256=file_hash(path),
        processor_sha256=value['processor_sha256'], initial_processor_sha256=config['initial_processor_sha256'],
        initial_full_state_sha256=config['initial_full_state_sha256'], dataset=config['dataset'],
        source_sha256=config['source_sha256'], state_hash=value['state_hash'],
        updates_by_problem=value['updates_by_problem'], last_update_losses=value.get('last_update_losses'),
        deployment_mode='transition', pretraining_seconds=value.get('pretraining_seconds'), model_reselection=False)


def fresh_assets(args):
    """Read fresh weights and TRAIN geometry only, not native labels/SBS costs."""
    ready = read_json(args.prepared_root / 'prepared.json')
    if (ready.get('experiment') != 'R84' or ready.get('seed') != 2
            or Path(ready['data_root']).resolve() != args.data_root.resolve()
            or ready.get('rows_per_problem') != dict(train=81000, val=4500, test=4500)
            or ready.get('scenario_sha256') != file_hash(args.data_root / 'scenario_manifest.json')):
        raise ValueError('Require frozen R84 TRAIN-only preparation identity')
    for name in ('identicalinit_seed2.pt', 'geometry_stats.json'):
        if ready.get('artifact_sha256', {}).get(name) != file_hash(args.prepared_root / name):
            raise ValueError('Frozen fresh/TRAIN geometry artifact differs: ' + name)
    initial = torch.load(args.prepared_root / 'identicalinit_seed2.pt', map_location='cpu', weights_only=False)
    if initial.get('trainable_sha256') != ready.get('trainable_sha256'):
        raise ValueError('Prepared initialization and fresh-weight identity differ')
    return initial, read_json(args.prepared_root / 'geometry_stats.json')


def run(args, device):
    from . import r113_experiment as runner
    if args.arm not in ARMS:
        raise ValueError('Scratch has no process pretraining')
    lock = runner.protocol_lock(args)
    dataset = ProcessDataset(lock['record']['process_manifest'], args.data_root)
    initial, geometry = fresh_assets(args)
    processor, receipt = extract_processor(initial, geometry, args.arm, device)
    del initial
    config = config_for(args, dataset, receipt)
    root = args.pretrain_root / args.arm
    root.mkdir(parents=True, exist_ok=True)
    with (root / 'runner.lock').open('a') as claim:
        fcntl.flock(claim, fcntl.LOCK_EX | fcntl.LOCK_NB)
        path, last, exported = root / 'protocol.json', root / 'last.pt', root / 'processor.pt'
        if path.exists() and read_json(path) != config:
            raise ValueError('Existing process protocol differs; preserve the old version')
        if exported.exists():
            value = torch.load(exported, map_location='cpu', weights_only=False)
            # Validation here uses a fresh complete model, not standalone manual seeding.
            fresh = runner.make_model(receipt['model_params'], args.arm)
            fresh.instance_encoder.processor.load_state_dict(processor.state_dict())
            validate_export(value, fresh, receipt, args.arm, manifest_sha256=dataset.identity['manifest_sha256'], main_lock=lock)
            if not (root / 'completion.json').exists():
                write_json(root / 'completion.json', completion(value, exported))
            return read_json(root / 'completion.json')
        if last.exists() and not args.resume:
            raise ValueError('Existing process training requires --resume')
        other = args.pretrain_root / next(a for a in ARMS if a != args.arm) / 'protocol.json'
        if other.exists():
            shared = read_json(other)
            for key in ('dataset', 'initial_full_state_sha256', 'initial_processor_sha256',
                        'processor_parameter_schema', 'task_order_sha256', 'main_protocol_lock'):
                if shared.get(key) != config[key]:
                    raise ValueError('B/C process initialization/sample stream differs: ' + key)
        write_json(path, config)
        optimizer = torch.optim.AdamW(processor.parameters(), lr=PEAK_LR, weight_decay=1e-4)
        scaler = torch.amp.GradScaler(device.type, enabled=device.type == 'cuda')
        successful, retries_total, counts, cursors, history = 0, 0, Counter(), dict.fromkeys(data.PROBLEMS, 0), []
        elapsed_prior = 0.
        order = task_order(MAX_UPDATES)
        if last.exists():
            saved = torch.load(last, map_location='cpu', weights_only=False)
            successful = saved['successful_updates']
            if (saved['args'] != config or saved['state_hash'] != json_hash(config)
                    or not 0 <= successful <= MAX_UPDATES or successful % SAVE_EVERY
                    or Counter(saved['updates_by_problem']) != Counter(order[:successful])
                    or any(saved['stream_draws'][p] != Counter(order[:successful])[p] * BATCH_SIZE for p in data.PROBLEMS)):
                raise ValueError('Process resume protocol/source cursors differ')
            processor.load_state_dict(saved['processor'], strict=True)
            optimizer.load_state_dict(saved['optimizer'])
            for state in optimizer.state.values():
                for key, value in state.items():
                    if torch.is_tensor(value):
                        state[key] = value.cpu() if key == 'step' else value.to(device)
            scaler.load_state_dict(saved['scaler'])
            counts, cursors = Counter(saved['updates_by_problem']), saved['stream_draws']
            history, retries_total = saved['history'], saved['amp_retries']
            elapsed_prior = saved.get('elapsed_seconds', 0.)
            restore_rank_rng(saved['rng'], device)
        else:
            runner.engine.set_seed(2)
        streams = {p: ProcessStream(dataset, p, cursors[p]) for p in data.PROBLEMS}

        def snapshot():
            return dict(args=config, state_hash=json_hash(config), processor=processor.state_dict(),
                optimizer=optimizer.state_dict(), scaler=scaler.state_dict(), rng=capture_rank_rng(device),
                successful_updates=successful, updates_by_problem=dict(counts),
                stream_draws={p: s.draws for p, s in streams.items()}, amp_retries=retries_total, history=history,
                elapsed_seconds=elapsed_prior + time.monotonic() - started)

        tracker, started = None, time.monotonic()
        process_args = copy.copy(args)
        process_args.root, process_args.experiment = args.pretrain_root, 'R113_pretrain'
        try:
            if not last.exists():
                save_checkpoint(last, snapshot())
            rng = capture_rank_rng(device)
            tracker = wandb_run(process_args, config)
            restore_rank_rng(rng, device)
            processor.train()
            while successful < MAX_UPDATES:
                problem = order[successful]
                rows = streams[problem].take(BATCH_SIZE)
                program, targets, ids = dataset.batch(problem, rows, device)
                for group in optimizer.param_groups:
                    group['lr'] = learning_rate(successful + 1)
                losses, retries, norm = update_batch(processor, optimizer, scaler, program, targets,
                    MODE[args.arm], config['microbatch_caps'][problem])
                successful += 1
                counts[problem] += 1
                retries_total += retries
                digest = hashlib.sha256(problem.encode('ascii') + ids['row_index'].tobytes()
                    + ids['source_index'].tobytes() + ids['input_sha256'].tobytes()
                    + program['sequence'].detach().cpu().contiguous().numpy().tobytes()).hexdigest()
                info = dict(successful_updates=successful, problem=problem, lr=learning_rate(successful),
                    losses=losses, grad_norm=norm, amp_retries=retries_total, sample_ids_sha256=digest,
                    stream_draws=streams[problem].draws, logical_rows=BATCH_SIZE,
                    seconds=elapsed_prior + time.monotonic() - started)
                if successful == 1 or successful % args.log_every == 0 or successful % SAVE_EVERY == 0:
                    write_json(root / 'status.json', dict(experiment='R113_pretrain', arm=args.arm,
                        phase='pretraining', max_successful_updates=MAX_UPDATES, updated_utc=utc(), **info))
                    print('[R113 pretrain] ' + json.dumps(dict(arm=args.arm, **info)), flush=True)
                    tracker.log({**{'process/' + k: v for k, v in losses.items()}, 'lr': info['lr'],
                        'successful_updates': successful}, step=successful)
                with (root / 'updates.jsonl').open('a') as stream:
                    stream.write(json.dumps(info, allow_nan=False) + '\n')
                if successful % SAVE_EVERY == 0 or successful == MAX_UPDATES:
                    history.append(info)
                    save_checkpoint(last, snapshot())
                del program, targets
            if any(counts[p] != MAX_UPDATES // len(data.PROBLEMS) for p in data.PROBLEMS):
                raise ValueError('Incomplete full18 process update budget')
            value = dict(format='r113_processor_only_v1', complete=True, args=config, state_hash=json_hash(config),
                processor={k: v.detach().cpu().clone() for k, v in processor.state_dict().items()},
                successful_updates=successful, updates_by_problem=dict(counts),
                stream_draws={p: s.draws for p, s in streams.items()},
                last_update_losses=history[-1]['losses'], pretraining_seconds=elapsed_prior + time.monotonic() - started)
            value['processor_sha256'] = parameter_hash(value['processor'])
            save_checkpoint(exported, value)
            result = completion(value, exported)
            write_json(root / 'completion.json', result)
            write_json(root / 'status.json', dict(result, phase='complete'))
            return result
        finally:
            if tracker is not None:
                tracker.finish()


def main(argv=None):
    from . import r113_experiment as runner
    values = sys.argv[1:] if argv is None else argv
    return runner.main(['pretrain', *values])


if __name__ == '__main__':
    main()

"""Post-TEST R123 diagnostics, live sparse LIB inference and portable PDF.

Nothing runs at import. All inference commands are explicit and post-lock;
report consumes fixed predictions and never selects a checkpoint or policy.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

from code.unified_selector.registry import PROBLEMS


ROOT = Path(__file__).parent / 'runs/R123_heterogeneous_moe'
REFERENCE_ROOT = Path(__file__).parent / 'runs/R84_new90k_attention'
EXPERTS = ('R77B', 'R80A', 'R83')
CORE = ('top1_pct', 'mean_cost', 'sbs_cost', 'oracle_cost', 'vs_sbs_pct',
        'vs_oracle_pct', 'actual_regret_pct')
WINNER_RULE = 'Saved native int64 winner / positive FP64 costs; NOT FP32 argmin'
FONT = Path('/usr/share/fonts/google-noto-cjk/NotoSansCJK-Regular.ttc')
SCHEMA = 'r123_post_test_finish_v1'


def read_json(path):
    with Path(path).open() as stream:
        return json.load(stream)


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.tmp.{os.getpid()}')
    with temporary.open('w') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    os.replace(temporary, path)


def save_npz(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.tmp.{os.getpid()}')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, **data)
    os.replace(temporary, path)


def read_npz(path):
    with np.load(path, allow_pickle=False) as stream:
        return {name: np.array(stream[name], copy=True) for name in stream.files}


def write_csv(path, rows, fields=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(fields or dict.fromkeys(k for row in rows for k in row))
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def remember(inputs, path):
    inputs[str(Path(path).resolve())] = file_hash(path)


def hard_actions(scores, pool_ids):
    """No cross-expert calibration; score ties use ascending global solver ID."""
    scores, ids = np.asarray(scores), np.asarray(pool_ids)
    if (scores.ndim < 2 or ids.ndim != 1 or scores.shape[-1] != len(ids)
            or len(np.unique(ids)) != len(ids) or not np.isfinite(scores).all()):
        raise ValueError('Finite aligned candidate scores and unique solver IDs required')
    order = np.argsort(ids, kind='stable')
    return order[np.argmax(scores[..., order], axis=-1)]


def validate_prediction(data):
    costs, winner, pred = data['costs'], data['winner'], data['pred']
    n, k = costs.shape
    pool = [str(x) for x in data['pool']]
    if (not n or costs.dtype != np.float64 or not np.isfinite(costs).all()
            or (costs <= 0).any() or len(pool) != k or len(set(pool)) != k
            or np.asarray(data['pool_ids']).shape != (k,)
            or len(np.unique(data['pool_ids'])) != k):
        raise ValueError('Native positive FP64 costs and complete legal pool required')
    for name, value in (('winner', winner), ('pred', pred)):
        if (value.dtype != np.int64 or value.shape != (n,)
                or (value < 0).any() or (value >= k).any()):
            raise ValueError('Invalid native/candidate index: ' + name)
    if not np.array_equal(costs[np.arange(n), winner], costs.min(axis=1)):
        raise ValueError('Saved native winner is not exactly cost-optimal')
    if not np.array_equal(hard_actions(data['logits'], data['pool_ids']), pred):
        raise ValueError('Saved prediction differs from fixed solver-ID argmax')
    indices = data['indices']
    if indices.dtype != np.int64 or indices.shape != (n,) or len(np.unique(indices)) != n:
        raise ValueError('Current row identity collision or noninteger index')
    return n


def check_bank_join(data, bank):
    """A row position is never substituted for the declared source identity."""
    validate_prediction(data)
    for name, expected in (('indices', bank.row_indices),
                           ('release_source_indices', bank.source_indices),
                           ('source_identity', bank.source_identity)):
        if name not in data or not np.array_equal(data[name], expected):
            raise ValueError('Current/source identity mismatch: ' + name)
    if len(np.unique(data['source_identity'])) != len(data['source_identity']):
        raise ValueError('Duplicate qualified source identity')
    chosen = data['chosen_expert']
    if (chosen.dtype != np.int64 or chosen.shape != data['pred'].shape
            or (chosen < 0).any() or (chosen >= len(EXPERTS)).any()):
        raise ValueError('Invalid hard expert route')
    selected = bank.logits[np.arange(len(chosen)), chosen]
    if not np.array_equal(data['logits'], selected):
        raise ValueError('Capture did not retain exact selected-expert FP32 scores')
    return hard_actions(bank.logits, data['pool_ids'])


def quality(data, pred=None):
    pred = data['pred'] if pred is None else np.asarray(pred)
    costs = data['costs']
    selected = costs[np.arange(len(pred)), pred]
    oracle = costs.min(axis=1)
    return selected, oracle, (selected - oracle) / oracle


def metric_row(data, baseline, *, problem, split, policy='R123'):
    if baseline['selection_split'] != 'train':
        raise ValueError('Fixed SBS must come from TRAIN')
    pool = list(data['pool'])
    slot = pool.index(baseline['sbs_solver'])
    selected, oracle, regret = quality(data)
    mean, sbs, optimal = (float(selected.mean()), float(data['costs'][:, slot].mean()),
                          float(oracle.mean()))
    return dict(problem=problem, split=split, policy=policy, n=len(selected),
        top1_pct=100. * float(np.mean(data['pred'] == data['winner'])),
        mean_cost=mean, sbs_cost=sbs, oracle_cost=optimal,
        vs_sbs_pct=100. * (mean / sbs - 1.), vs_oracle_pct=100. * (mean / optimal - 1.),
        actual_regret_pct=100. * float(regret.mean()), sbs_solver=baseline['sbs_solver'],
        sbs_fit='original current TRAIN81000, R84 fixed artifact', winner_rule=WINNER_RULE)


def macro_row(rows, fields, **labels):
    if len(rows) != len(PROBLEMS) or {r['problem'] for r in rows} != set(PROBLEMS):
        raise ValueError('ALL18 requires all18 exactly once; no row-weighted substitute')
    return dict(problem='ALL18', n=sum(r['n'] for r in rows),
        aggregation='equal problem weight; raw cost means are historical routing macro only',
        **labels, **{k: float(np.mean([r[k] for r in rows])) for k in fields})


def bank_decomposition(data, actions):
    n = len(data['winner'])
    if actions.shape != (n, len(EXPERTS)) or not (actions == data['pred'][:, None]).any(axis=1).all():
        raise ValueError('Router action must be supplied by the fixed complete bank')
    covered = (actions == data['winner'][:, None]).any(axis=1)
    wrong = data['pred'] != data['winner']
    selected, oracle, regret = quality(data)
    bank_cost = data['costs'][np.arange(n)[:, None], actions].min(axis=1)
    floor = (bank_cost - oracle) / oracle
    excess = (selected - bank_cost) / oracle
    if (floor < 0).any() or (excess < 0).any():
        raise ValueError('Negative bank support or selected routing excess')
    np.testing.assert_allclose(floor + excess, regret, rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal((~covered).astype(int) + (covered & wrong).astype(int), wrong)
    return dict(n=n, bank_native_coverage_pct=100. * float(covered.mean()),
        native_bank_miss_pp=100. * float((~covered).mean()),
        native_router_miss_pp=100. * float((covered & wrong).mean()),
        native_error_pct=100. * float(wrong.mean()),
        bank_quality_floor_pct=100. * float(floor.mean()),
        routing_quality_excess_pct=100. * float(excess.mean()),
        actual_regret_pct=100. * float(regret.mean()),
        all_experts_wrong_count=int((~covered).sum()),
        all_experts_correct_count=int((actions == data['winner'][:, None]).all(axis=1).sum()),
        native_error_zero_quality_count=int((wrong & (regret == 0)).sum()),
        bank_oracle_is_deployable=False)


def fit_train_references(training):
    """Only native TRAIN correctness fits these two constant expert policies."""
    rates = {}
    for problem, value in training.items():
        if value['split'] != 'train':
            raise ValueError('No VAL/TEST fitting or reference selection')
        actions, winner = value['actions'], value['winner']
        if actions.shape != (len(winner), len(EXPERTS)) or not len(winner):
            raise ValueError('Complete bank TRAIN actions required')
        rates[problem] = (actions == winner[:, None]).mean(axis=0).tolist()
    mean = np.mean(list(rates.values()), axis=0)
    return dict(fit_split='train', population='certified metaTRAIN10886/task; NOT full TRAIN81000',
        expert_order=list(EXPERTS), tie_rule='first fixed bank ID order',
        global_fixed_expert=int(np.argmax(mean)), equal_task_correctness_prior=mean.tolist(),
        task_correctness_prior=rates,
        task_prior_expert={p: int(np.argmax(v)) for p, v in rates.items()},
        conditional_prior_meaning='conditioned on task only, no per-query answers or fitted instance learner')


def method_rows(data, *, problem, split, near_zero_pct=.1):
    n = len(data['winner'])
    _, _, regret = quality(data)
    rows = []
    for slot, solver in enumerate(data['pool']):
        won, picked = data['winner'] == slot, data['pred'] == slot
        missed = won & ~picked
        count, support = int(picked.sum()), int(won.sum())
        pct = 100. * count / n
        rows.append(dict(problem=problem, split=split, solver=str(solver),
            solver_id=int(data['pool_ids'][slot]), n=n,
            selected_count=count, selected_pct=pct, true_winner_count=support,
            true_winner_pct=100. * support / n, correct_selected_count=int((won & picked).sum()),
            missed_winner_count=int(missed.sum()), false_pick_count=int((picked & ~won).sum()),
            zero_selected=count == 0, zero_selected_with_winner=count == 0 and support > 0,
            near_zero_selected_with_winner=0. < pct <= near_zero_pct and support > 0,
            missed_actual_regret_contribution_pp=100. * float(regret[missed].sum()) / n,
            missed_zero_quality_count=int((missed & (regret == 0)).sum()),
            population='metaTRAIN10886' if split == 'train' else 'frozen current4500'))
    np.testing.assert_allclose(sum(r['missed_actual_regret_contribution_pp'] for r in rows),
                               100. * regret.mean(), rtol=1e-12, atol=1e-12)
    return rows


def paired_reference(data, reference_pred):
    reference_ok, router_ok = reference_pred == data['winner'], data['pred'] == data['winner']
    _, _, regret = quality(data)
    _, _, reference_regret = quality(data, reference_pred)
    return dict(recovered_count=int((~reference_ok & router_ok).sum()),
        harmed_count=int((reference_ok & ~router_ok).sum()),
        native_top1_delta_pp=100. * float(router_ok.mean() - reference_ok.mean()),
        actual_regret_delta_pp=100. * float((regret - reference_regret).mean()))


def locked(args):
    from . import r123_experiment as runner
    import torch
    run = args.root / 'model'
    lock = runner.test_permission(run)
    config = read_json(run / 'protocol.json')
    state = torch.load(run / 'best.pt', map_location='cpu', weights_only=False)
    if state['args'] != config or config['expert_order'] != list(EXPERTS):
        raise ValueError('Locked complete router protocol differs')
    inputs = {}
    for path in (run / 'best.pt', run / 'protocol.json', run / 'final_test_lock.json'):
        remember(inputs, path)
    for name, digest in config['source_sha256'].items():
        path = Path(__file__).with_name(name)
        if file_hash(path) != digest:
            raise ValueError('Frozen inference source changed: ' + name)
        remember(inputs, path)
    helper = Path(__file__).with_name('r123_prediction_cache.py')
    if file_hash(helper) != config['shared_prediction_cache_sha256']:
        raise ValueError('Frozen shared prediction helper changed')
    remember(inputs, helper)
    for split in ('val', 'test'):
        path = run / 'evaluation' / split / 'selected_best/metrics.json'
        if not path.is_file():
            raise PermissionError('Finish waits for completed locked VAL and TEST; no new evaluation here')
        metrics = read_json(path)
        if (metrics.get('experiment') != 'R123' or metrics.get('split') != split
                or metrics.get('successful_updates') != lock['selected_update']
                or set(metrics['per_problem']) != set(PROBLEMS)):
            raise ValueError('Incomplete or wrong-update selected capture: ' + split)
        remember(inputs, path)
    return dict(run=run, lock=lock, config=config, state=state, inputs=inputs)


def capture_identity(model, kind):
    return dict(schema=SCHEMA, kind=kind, checkpoint_sha256=model['lock']['checkpoint_sha256'],
        protocol_sha256=model['lock']['protocol_sha256'], selected_update=model['lock']['selected_update'])


def require_identity(path, model, kind):
    value = read_json(path)
    for key, expected in capture_identity(model, kind).items():
        if value.get(key) != expected:
            raise ValueError('Post-lock artifact identity mismatch: ' + str(path))
    for name, digest in value['artifacts'].items():
        if file_hash(path.parent / name) != digest:
            raise ValueError('Post-lock artifact changed: ' + name)
    return value


def context(model):
    from .r123_experiment import source_context
    return source_context(model['config']['prepared_root'], model['config']['meta_preparation'])


def capture_train(args, model):
    import torch
    from . import r123_experiment as runner
    from .r123_model import ReliabilityGate, gate_inputs
    directory = args.root / 'finish/captures/train'
    receipt_path = directory / 'receipt.json'
    if receipt_path.is_file():
        return require_identity(receipt_path, model, 'metaTRAIN_cached_chosen_scores')
    ctx = context(model)
    runner.configure_precision()
    gate = ReliabilityGate(model['config']['model_params'])
    gate.load_state_dict(model['state']['model'], strict=True)
    gate = gate.to(args.device).eval().requires_grad_(False)
    artifacts, counts = {}, {}
    with torch.inference_mode():
        for problem in PROBLEMS:
            source = runner.open_source(ctx, 'train', problem, input_cache_root=args.input_cache_root)
            bank = None
            try:
                bank = runner.open_bank_cache(args.bank_root, 'train', problem, ctx, run_root=model['run'])
                scores, choices, reliability = [], [], []
                for start in range(0, len(bank.row_indices), args.predict_batch):
                    rows = np.arange(start, min(start + args.predict_batch, len(bank.row_indices)))
                    batch = source.batch(bank.row_indices[rows], device=args.device)
                    out = gate(gate_inputs(batch))
                    chosen = out['reliability_logits'].argmax(-1).cpu().numpy()
                    scores.append(bank.logits[rows, chosen].copy())
                    choices.append(chosen)
                    reliability.append(out['reliability'].cpu().numpy())
                scores = np.concatenate(scores)
                data = dict(logits=scores, pred=hard_actions(scores, source.pool_ids).astype(np.int64),
                    costs=np.array(source.raw['costs'][bank.row_indices], copy=True),
                    winner=np.array(source.raw['winner'][bank.row_indices], copy=True),
                    indices=bank.row_indices.copy(), release_source_indices=bank.source_indices.copy(),
                    source_identity=bank.source_identity.copy(), pool=np.asarray(source.pool),
                    pool_ids=np.asarray(source.pool_ids, dtype=np.int64),
                    chosen_expert=np.concatenate(choices), reliability=np.concatenate(reliability),
                    metadata_json=np.asarray(json.dumps(capture_identity(model, 'metaTRAIN_cached_chosen_scores'))))
                check_bank_join(data, bank)
                path = directory / (problem + '.npz')
                save_npz(path, data)
                artifacts[path.name], counts[problem] = file_hash(path), len(data['winner'])
            finally:
                if bank is not None:
                    bank.close()
                source.close()
    value = dict(capture_identity(model, 'metaTRAIN_cached_chosen_scores'), artifacts=artifacts,
        rows=counts, forward='raw inputs -> gate -> selected cached FP32 expert scores',
        deployment_timing=False, fitted_or_updated_parameters=False)
    write_json(receipt_path, value)
    return value


class ExpertExecutionCounter:
    """Hooks validate row dispatch, not just the number of Python calls."""

    def __init__(self, experts):
        self.rows = np.zeros(len(experts), dtype=np.int64)
        self.handles = [expert.register_forward_pre_hook(self._hook(i)) for i, expert in enumerate(experts)]

    def _hook(self, index):
        def count(_module, inputs):
            batch = inputs[0]
            if any(key in batch for key in ('costs', 'winner', 'ind', 'performance_target')):
                raise ValueError('Query supervision reached frozen expert forward')
            self.rows[index] += len(batch['n'])
        return count

    def verify(self, chosen):
        expected = np.bincount(chosen, minlength=len(self.rows))
        np.testing.assert_array_equal(self.rows, expected)
        if self.rows.sum() != len(chosen):
            raise ValueError('Live inference did not execute exactly one expert per query')
        return self.rows.tolist()

    def close(self):
        for handle in self.handles:
            handle.remove()


def synchronize(device):
    if str(device).startswith('cuda'):
        import torch
        torch.cuda.synchronize(device)


def live_forward(model, batches, device, *, warmup=True):
    """No expert_logits argument is accepted on this deployment path."""
    import torch
    from .r123_meta_data import model_inputs
    scores, choices, reliability = [], [], []
    seconds, transfer_seconds, gather_seconds = 0., 0., 0.
    counter = ExpertExecutionCounter(model.experts)
    first = True
    iterator = iter(batches)
    try:
        with torch.inference_mode():
            while True:
                start = time.perf_counter()
                try:
                    batch = next(iterator)
                except StopIteration:
                    break
                gather_seconds += time.perf_counter() - start
                clean = model_inputs(batch)
                synchronize(device)
                start = time.perf_counter()
                clean = {k: v.to(device) if torch.is_tensor(v) else v for k, v in clean.items()}
                synchronize(device)
                transfer_seconds += time.perf_counter() - start
                if first and warmup:
                    model(clean)
                    synchronize(device)
                    counter.rows[:] = 0
                first = False
                synchronize(device)
                start = time.perf_counter()
                output = model(clean)
                synchronize(device)
                seconds += time.perf_counter() - start
                scores.append(output['logits'].float().cpu().numpy())
                choices.append(output['chosen_expert'].cpu().numpy())
                reliability.append(output['reliability'].cpu().numpy())
        chosen = np.concatenate(choices)
        counts = counter.verify(chosen)
    finally:
        counter.close()
    return dict(logits=np.concatenate(scores), chosen_expert=chosen,
        reliability=np.concatenate(reliability)), dict(n=len(chosen), device=str(device),
        forward_seconds=seconds, forward_ms_per_query=1000. * seconds / len(chosen),
        gather_seconds=gather_seconds, host_transfer_seconds=transfer_seconds,
        expert_executed_rows=counts, total_expert_executed_rows=sum(counts),
        expected_experts_per_query=1, warmup_batches_excluded=int(warmup),
        precision='FP32, TF32 disabled, eval, inference_mode, no autocast',
        included='raw-input gate plus only selected complete expert, dispatch and execution hooks',
        excluded='input/geometry preparation, batch gather, host transfer, warmup, output CPU copy, bank cache generation')


def lib_sources(raw):
    directory = Path(raw['directory'])
    return {str(directory / name): file_hash(directory / name)
            for name in ('dataset.pkl', 'raw_label.pkl')}


def lib_capture_receipt(path, raw, model, *, allow_legacy=False):
    path = Path(path)
    data = read_npz(path)
    n = validate_prediction(data)
    for name, expected in (('costs', raw['costs']), ('winner', raw['winner']),
                           ('pool', raw['pool']), ('pool_ids', raw['pool_ids']),
                           ('indices', np.arange(len(raw['winner']), dtype=np.int64))):
        if not np.array_equal(data[name], expected):
            raise ValueError('Existing LIB capture/native identity mismatch: ' + name)
    chosen = data['chosen_expert']
    if (chosen.dtype != np.int64 or chosen.shape != (n,)
            or (chosen < 0).any() or (chosen >= len(EXPERTS)).any()):
        raise ValueError('Invalid saved LIB hard routes')
    sources = lib_sources(raw)
    if 'metadata_json' in data:
        metadata = json.loads(str(data['metadata_json']))
        for key, expected in capture_identity(model, 'live_sparse_LIB').items():
            if metadata.get(key) != expected:
                raise ValueError('Existing LIB capture/locked checkpoint mismatch: ' + key)
        if metadata['sources'] != sources or metadata['dataset'] != path.stem:
            raise ValueError('Existing LIB input provenance mismatch')
        timing = metadata['timing']
        attribution = 'Embedded locked checkpoint, dataset and source hashes verified'
    else:
        if not allow_legacy:
            raise ValueError('Unsigned existing LIB capture preserved; same-lock reuse requires '
                             '--reuse-legacy-lib-capture ' + path.stem)
        counts = np.bincount(chosen, minlength=len(EXPERTS)).tolist()
        timing = dict(n=n, batch_size=None, forward_seconds=None, forward_ms_per_query=None,
            expert_executed_rows=counts, total_expert_executed_rows=n, expected_experts_per_query=1,
            measurement_status='Elapsed timing lost before failed provenance write; NOT zero or remeasured',
            execution_count_basis='Saved selected routes; legacy producer checked live hooks before saving')
        attribution = ('Explicit operator same-lock legacy reuse; native arrays/pool verified; '
                       'original capture has no embedded checkpoint hash; no retroactive signature')
    return dict(capture_identity(model, 'live_sparse_LIB'), dataset=path.stem,
        artifacts={path.name: file_hash(path)}, sources=sources, timing=timing,
        capture_attribution=attribution, query_labels_used_in_forward=False)


def lib(args, model):
    from . import r123_experiment as runner
    from .nss_comparison import read_costs
    from .r41_frozen_evaluation import attach_geometry
    from .tensor_loader import TensorBatchLoader
    from .train import make_loader
    directory = args.root / 'finish/lib'
    receipt_path = directory / 'receipt.json'
    if receipt_path.is_file():
        return require_identity(receipt_path, model, 'live_sparse_LIB')
    router = None
    artifacts, timings, sources = {}, {}, {}
    for problem, dataset in (('TSP', 'TSPLIB'), ('CVRP', 'CVRPLIB')):
        start = time.perf_counter()
        raw = read_costs(problem, dataset)
        if (raw['pool'] != model['config']['candidate_pools'][problem]
                or list(raw['pool_ids']) != model['config']['candidate_pool_ids'][problem]):
            raise ValueError('Native LIB/current legal candidate pools differ')
        sources[dataset] = lib_sources(raw)
        path = directory / (dataset + '.npz')
        sidecar = directory / (dataset + '_receipt.json')
        if path.is_file():
            if sidecar.is_file():
                previous = require_identity(sidecar, model, 'live_sparse_LIB')
                if previous['sources'] != sources[dataset] or previous['dataset'] != dataset:
                    raise ValueError('Existing LIB receipt/input provenance mismatch')
            record = lib_capture_receipt(path, raw, model,
                allow_legacy=sidecar.is_file() or dataset in args.reuse_legacy_lib_capture)
            if not sidecar.is_file():
                write_json(sidecar, record)
            artifacts.update(record['artifacts'])
            artifacts[sidecar.name] = file_hash(sidecar)
            timings[dataset] = record['timing']
            continue
        if sidecar.is_file():
            raise ValueError('Existing LIB receipt has no capture; preserved for explicit recovery')
        if router is None:
            router, _ = runner.load_checkpoint(model['run'] / 'best.pt', device=args.device)
            router.requires_grad_(False)
        loader = make_loader(problem, dataset, args.lib_batch, 0,
            coord_augment=0, shuffle=False, cache_device='cpu')
        if loader.size != len(raw['winner']) or loader.batch['pool_ids'].tolist() != list(raw['pool_ids']):
            raise ValueError('LIB input/native label membership or candidate alignment differs')
        attach_geometry(loader, 'cpu')
        prepare_seconds = time.perf_counter() - start
        output, timing = live_forward(router, TensorBatchLoader(loader.batch, args.lib_batch, False, False), args.device)
        data = dict(output, costs=raw['costs'], winner=raw['winner'],
            indices=np.arange(loader.size, dtype=np.int64), pool=np.asarray(raw['pool']),
            pool_ids=np.asarray(raw['pool_ids'], dtype=np.int64))
        data['pred'] = hard_actions(data['logits'], data['pool_ids']).astype(np.int64)
        validate_prediction(data)
        timing.update(input_geometry_preparation_seconds=prepare_seconds, batch_size=args.lib_batch)
        metadata = dict(capture_identity(model, 'live_sparse_LIB'), dataset=dataset,
                        sources=sources[dataset], timing=timing)
        data['metadata_json'] = np.asarray(json.dumps(metadata, allow_nan=False))
        save_npz(path, data)
        artifacts[path.name] = file_hash(path)
        timings[dataset] = timing
        write_json(sidecar, lib_capture_receipt(path, raw, model))
        artifacts[sidecar.name] = file_hash(sidecar)
    value = dict(capture_identity(model, 'live_sparse_LIB'), artifacts=artifacts,
        timing=timings, sources=sources, query_labels_used_in_forward=False)
    write_json(receipt_path, value)
    return value


def timing(args, model):
    from . import r123_experiment as runner
    directory = args.root / 'finish/timing'
    receipt_path = directory / 'receipt.json'
    if receipt_path.is_file():
        return require_identity(receipt_path, model, 'live_sparse_TEST_fixed_prefix')
    ctx = context(model)
    router, _ = runner.load_checkpoint(model['run'] / 'best.pt', device=args.device)
    router.requires_grad_(False)
    results = {}
    for problem in PROBLEMS:
        source = runner.open_source(ctx, 'test', problem, run_root=model['run'],
                                    input_cache_root=args.input_cache_root)
        try:
            n = min(args.timing_rows, source.size)
            batches = (source.batch(np.arange(start, min(start + args.predict_batch, n)), device='cpu')
                       for start in range(0, n, args.predict_batch))
            output, result = live_forward(router, batches, args.device)
            path = model['run'] / 'evaluation/test/selected_best/by_problem' / (problem + '.npz')
            saved = read_npz(path)
            pred = hard_actions(output['logits'], source.pool_ids)
            np.testing.assert_array_equal(pred, saved['pred'][:n])
            np.testing.assert_array_equal(output['chosen_expert'], saved['chosen_expert'][:n])
            result.update(problem=problem, row_rule='fixed current TEST prefix, no performance-based selection',
                          rows=list(range(n)), live_hard_actions_match_locked_cache=True,
                          batch_size=args.predict_batch)
            results[problem] = result
        finally:
            source.close()
    value = dict(capture_identity(model, 'live_sparse_TEST_fixed_prefix'), artifacts={},
        by_problem=results, inference_rule_changed=False, performance_tuning=False)
    write_json(receipt_path, value)
    return value


def align_nss_prediction(other, current):
    """Align historical public metadata without replacing saved NSS actions."""
    pool = np.asarray(other['pool']).astype(str).tolist()
    indices = np.asarray(other['indices'])
    costs, winner, pred = other['costs'], other['winner'], other['pred']
    scores = other['logits'] if 'logits' in other else other['scores']
    n, k = current['costs'].shape
    if (len(pool) != k or len(set(pool)) != k or set(pool) != set(current['pool'])
            or indices.dtype != np.int64 or indices.shape != (n,)
            or len(np.unique(indices)) != n or set(indices) != set(current['indices'])
            or costs.dtype != np.float64 or costs.shape != (n, k)
            or scores.shape != (n, k) or not np.isfinite(scores).all()):
        raise ValueError('NSS reference instances/candidate names/native array schema differ')
    for value in (winner, pred):
        if (value.dtype != np.int64 or value.shape != (n,)
                or (value < 0).any() or (value >= k).any()):
            raise ValueError('Invalid saved NSS native winner or original selection')
    lookup = {int(index): i for i, index in enumerate(indices)}
    order = np.asarray([lookup[int(i)] for i in current['indices']])
    columns = [pool.index(str(s)) for s in current['pool']]
    inverse = np.argsort(columns)
    aligned_costs = costs[order][:, columns]
    aligned_winner = inverse[winner[order]]
    if (not np.array_equal(aligned_costs, current['costs'])
            or not np.array_equal(aligned_winner, current['winner'])):
        raise ValueError('NSS/current aligned native FP64 costs or winner differ')
    # Only after exact instance/name/cost/winner alignment can IDs be inherited.
    ids = np.asarray(current['pool_ids'], dtype=np.int64)
    if 'pool_ids' in other and not np.array_equal(np.asarray(other['pool_ids'])[columns], ids):
        raise ValueError('Saved NSS public solver IDs disagree with aligned names')
    aligned = dict(other, costs=aligned_costs, winner=aligned_winner,
        pred=inverse[pred[order]], indices=np.array(current['indices'], copy=True),
        pool=np.asarray(current['pool']), pool_ids=ids.copy())
    for key in ('scores', 'logits'):
        if key in other:
            aligned[key] = other[key][order][:, columns]
    if 'source_indices' in other:
        aligned['source_indices'] = other['source_indices'][order]
    return aligned


def nss_comparisons(predictions, libs, model, root, inputs):
    plan_path = root / 'protocol.json'
    plan = read_json(plan_path) if plan_path.is_file() else None
    if plan is not None:
        remember(inputs, plan_path)
        if (Path(plan['data_root']).resolve() != Path(model['config']['data_root']).resolve()
                or plan.get('retrained_on_new90k') is not False
                or plan.get('expected_test_rows_per_task') != 4500):
            raise ValueError('NSS reference is not the frozen same-release historical model')
    rows = []
    for dataset, problem in (('TSP TEST', 'TSP'), ('CVRP TEST', 'CVRP'),
                              ('TSPLIB', 'TSP'), ('CVRPLIB', 'CVRP')):
        is_test = dataset.endswith(' TEST')
        data = predictions['test'][problem] if is_test else libs[dataset]
        name = 'test' if is_test else dataset
        path = root / problem / (name + '_predictions.npz')
        metadata_path = path.with_name(name + '_metrics.json')
        selected, _, _ = quality(data)
        row = dict(dataset=dataset, n=len(selected), current_model='R123 unique locked complete router',
            current_top1_pct=100. * float(np.mean(data['pred'] == data['winner'])),
            current_mean_cost=float(selected.mean()), nss_top1_pct=None, nss_mean_cost=None,
            status='missing_saved_NSS', checkpoint_sha256=model['lock']['checkpoint_sha256'],
            nss_checkpoint_sha256=None, nss_retrained=False, new_nss_inference=False,
            current_train='certified metaTRAIN10886/task; bank historical TRAIN70114',
            nss_train='historical TRAIN70114; not new90k retraining',
            evaluation_data=model['config']['data_root'] if is_test else 'original native ' + dataset,
            pool=json.dumps(list(data['pool'])), winner_rule=WINNER_RULE)
        if plan is not None and path.is_file() and metadata_path.is_file():
            metadata = read_json(metadata_path)
            digest = plan['checkpoints'][problem]['sha256']
            if metadata.get('checkpoint', {}).get('sha256') != digest:
                raise ValueError('NSS reference checkpoint identity differs')
            if is_test:
                once_path = path.with_name('test_once.json')
                once = read_json(once_path)
                if once.get('state') != 'complete' or once.get('predictions_sha256') != file_hash(path):
                    raise ValueError('NSS saved new4500 completion identity differs')
                remember(inputs, once_path)
            other = align_nss_prediction(read_npz(path), data)
            pred = other['pred']
            correct = 100. * float(np.mean(pred == data['winner']))
            cost = float(data['costs'][np.arange(len(pred)), pred].mean())
            np.testing.assert_allclose(correct, 100. * metadata['top1_native_ind'], rtol=0, atol=1e-9)
            np.testing.assert_allclose(cost, metadata['mean_cost'], rtol=0, atol=1e-9)
            row.update(status='complete', nss_top1_pct=correct, nss_mean_cost=cost,
                       nss_checkpoint_sha256=digest)
            remember(inputs, path)
            remember(inputs, metadata_path)
        rows.append(row)
    return rows


def set_style(font=FONT):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import font_manager
    if not Path(font).is_file():
        raise FileNotFoundError('Existing CJK font required: ' + str(font))
    font_manager.fontManager.addfont(str(font))
    matplotlib.rcParams.update({'font.family': font_manager.FontProperties(fname=str(font)).get_name(),
        'font.size': 10, 'pdf.fonttype': 42, 'axes.unicode_minus': False})


def table_figure(title, headers, rows, notes):
    from matplotlib import pyplot as plt
    fig, ax = plt.subplots(figsize=(16.54, 11.69))
    ax.axis('off')
    fig.text(.04, .95, title, fontsize=17, weight='bold')
    table = ax.table(cellText=rows, colLabels=headers, cellLoc='center', bbox=[-.04, .18, 1.08, .68])
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    for (i, _), cell in table.get_celld().items():
        cell.set_edgecolor('#d3d9db')
        cell.set_linewidth(.4)
        if i == 0:
            cell.set_facecolor('#123e49')
            cell.set_text_props(color='white', weight='bold')
        elif i % 2:
            cell.set_facecolor('#edf4f2')
    fig.text(.04, .09, '\n'.join(notes), fontsize=9, va='top')
    return fig


def distribution_figure(problem, datasets, model):
    from matplotlib import pyplot as plt
    pool = list(datasets[0][1]['pool'])
    fig, axes = plt.subplots(1, len(datasets), figsize=(20 if len(datasets) == 2 else 13, 9), squeeze=False)
    fig.subplots_adjust(left=.12 if len(datasets) == 2 else .19, right=.97, top=.77, bottom=.2, wspace=.48)
    fig.text(.035, .95, 'R123 | ' + problem + ' | Model selection vs native winner', fontsize=18, weight='bold')
    fig.text(.035, .90, 'Same checkpoint ' + model['lock']['checkpoint_sha256'][:24]
             + ' | selected update ' + str(model['lock']['selected_update']), fontsize=10)
    fig.text(.035, .855, 'TEST: ' + Path(model['config']['data_root']).name + '; LIB: original native library', fontsize=9)
    fig.text(.035, .815, WINNER_RULE, fontsize=9)
    for ax, (dataset, data) in zip(axes[0], datasets):
        order = [list(data['pool']).index(name) for name in pool]
        n = len(data['winner'])
        accuracy = 100. * float(np.mean(data['pred'] == data['winner']))
        ax.set_title(f'{dataset} | n={n:,} | \u6a21\u578bTop1={accuracy:.2f}%', fontsize=14, pad=18)
        for key, offset, color, label in (('pred', -.17, '#177787', 'Model selection'),
                                         ('winner', .17, '#be5745', 'True native winner')):
            counts = np.bincount(data[key], minlength=len(pool))[order]
            pct = counts * 100. / n
            ax.barh(np.arange(len(pool)) + offset, pct, height=.28, color=color, label=label, zorder=3)
            for i, (value, count) in enumerate(zip(pct, counts)):
                inside = value >= 78
                ax.text(value - 1 if inside else value + .6, i + offset, f'{value:.2f}% ({count:,})',
                    va='center', ha='right' if inside else 'left', color='white' if inside else color, fontsize=9)
        ax.set_yticks(np.arange(len(pool)), labels=pool)
        ax.set_ylim(len(pool) - .4, -.6)
        ax.set_xlim(0, 100)
        ax.set_xlabel('Fraction of this dataset (%)')
        ax.grid(axis='x', alpha=.18, zorder=0)
        ax.spines[['top', 'right', 'left']].set_visible(False)
        ax.legend(loc='upper center', bbox_to_anchor=(.5, -.12), ncol=2, frameon=False, fontsize=9)
    fig.text(.035, .065, 'Paired bars use the same axis and fixed candidate order, including zeros. TEST precedes LIB.', fontsize=9)
    fig.text(.035, .03, 'Model Top1 is per-instance native-winner agreement, NOT the most frequently selected method.', fontsize=9)
    return fig


def render_pdf(path, predictions, libs, model, metrics, budgets, references, nss, timings, methods):
    from matplotlib import pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    notes = ['One locked full18 checkpoint; no TEST/LIB policy or checkpoint selection.',
        'TRAIN here = source-record-certified metaTRAIN10886/task, not full TRAIN81000; historical holdouts development-exposed.',
        'SBS = fixed R84 original TRAIN81000 choice. ALL18 = equal task weight; raw cost macro retained only for routing history.',
        'Native error and relative-cost regret are different budgets. An exact-cost tie may be a native error with zero regret.']
    path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(path, metadata={'Title': 'R123 locked full18 hard-router report', 'Author': 'Selector research'}) as pdf:
        def add(fig):
            pdf.savefig(fig)
            plt.close(fig)
        combined = []
        for p in (*PROBLEMS, 'ALL18'):
            val = next(r for r in metrics if r['problem'] == p and r['split'] == 'val' and r['policy'] == 'R123')
            test = next(r for r in metrics if r['problem'] == p and r['split'] == 'test' and r['policy'] == 'R123')
            combined.append([p, f"{val['top1_pct']:.2f}", f"{test['top1_pct']:.2f}",
                f"{val['mean_cost']:.5g}", f"{test['mean_cost']:.5g}",
                f"{val['actual_regret_pct']:.3f}", f"{test['actual_regret_pct']:.3f}"])
        add(table_figure('R123 | Unique VAL-selected checkpoint | VAL and TEST',
            ['Task', 'VAL Top1 %', 'TEST Top1 %', 'VAL cost', 'TEST cost', 'VAL regret %', 'TEST regret %'], combined, notes))
        for split in ('train', 'val', 'test'):
            rows = [r for r in metrics if r['split'] == split and r['policy'] == 'R123']
            add(table_figure('R123 | ' + split.upper() + ' | all required native metrics',
                ['Task', 'N', 'Top1 %', 'Mean cost', 'SBS cost', 'Oracle cost', 'vs SBS %', 'vs Oracle %', 'Actual regret %'],
                [[r['problem'], f"{r['n']:,}"] + [f'{r[k]:.5g}' for k in CORE] for r in rows], notes))
        rows = [r for r in metrics if r['split'] in ('TSPLIB', 'CVRPLIB')]
        add(table_figure('R123 | Original native LIB | all required metrics, same locked checkpoint',
            ['Dataset', 'N', 'Top1 %', 'Mean cost', 'SBS cost', 'Oracle cost', 'vs SBS %', 'vs Oracle %', 'Actual regret %'],
            [[r['split'], f"{r['n']:,}"] + [f'{r[k]:.5g}' for k in CORE] for r in rows], notes))
        for problem in PROBLEMS:
            datasets = [('TEST', predictions['test'][problem])]
            if problem in ('TSP', 'CVRP'):
                dataset = 'TSPLIB' if problem == 'TSP' else 'CVRPLIB'
                datasets.append((dataset, libs[dataset]))
            add(distribution_figure(problem, datasets, model))
        rows = [r for r in budgets if r['split'] == 'test']
        add(table_figure('R123 TEST | Bank coverage vs router errors; cost budget counted once',
            ['Task', 'Bank native cover %', 'Bank miss pp', 'Router miss pp', 'Native error %', 'Bank floor %', 'Route excess %', 'Regret %'],
            [[r['problem']] + [f'{r[k]:.3f}' for k in ('bank_native_coverage_pct', 'native_bank_miss_pp',
              'native_router_miss_pp', 'native_error_pct', 'bank_quality_floor_pct', 'routing_quality_excess_pct', 'actual_regret_pct')] for r in rows],
            ['Native error = bank miss + covered-but-missed route, with disjoint events.',
             'Regret = (best bank action cost - oracle)/oracle + (chosen cost - best bank action cost)/oracle.',
             'Bank best/coverage are answer-dependent DIAGNOSTICS, not deployable oracle policies.'] + notes[:2]))
        rows = []
        for p in (*PROBLEMS, 'ALL18'):
            choices = {r['policy']: r for r in references if r['split'] == 'test' and r['problem'] == p}
            router = next(r for r in metrics if r['split'] == 'test' and r['problem'] == p and r['policy'] == 'R123')
            rows.append([p, f"{router['top1_pct']:.2f}", f"{choices['TRAIN_global_fixed']['top1_pct']:.2f}",
                f"{choices['TRAIN_task_prior']['top1_pct']:.2f}", f"{router['actual_regret_pct']:.3f}",
                f"{choices['TRAIN_global_fixed']['actual_regret_pct']:.3f}", f"{choices['TRAIN_task_prior']['actual_regret_pct']:.3f}"])
        add(table_figure('R123 TEST | TRAIN-only fixed-expert and task-conditioned prior references',
            ['Task', 'Router Top1 %', 'Global fixed %', 'Task prior %', 'Router regret %', 'Global regret %', 'Prior regret %'], rows,
            ['References fitted only on the SAME certified metaTRAIN bank native-correctness labels.',
             'Global fixed maximizes equal-task TRAIN correctness; task prior maximizes within-task TRAIN correctness.',
             'Task prior is constant within a task, not another instance model; ties use fixed R77B/R80A/R83 order.'] + notes[:2]))
        add(table_figure('Same locked R123 vs saved NSS | TEST and native LIB',
            ['Dataset', 'N', 'R123 Top1 %', 'NSS Top1 %', 'R123 cost', 'NSS cost', 'NSS status'],
            [[r['dataset'], r['n'], f"{r['current_top1_pct']:.2f}",
              'MISSING' if r['nss_top1_pct'] is None else f"{r['nss_top1_pct']:.2f}",
              f"{r['current_mean_cost']:.5g}", 'MISSING' if r['nss_mean_cost'] is None else f"{r['nss_mean_cost']:.5g}", r['status']] for r in nss],
            ['Same instances, native FP64 labels, named legal pool and checkpoint verified when NSS is available.',
             'NSS is the historical70114 model, NOT a new90k retraining; no new NSS inference.'] + notes[:2]))
        add(table_figure('Actual live hard routing | Gate plus ONE selected complete expert per query',
            ['Dataset', 'N', 'Batch', 'FP32 forward sec', 'ms / query', 'Executed expert rows'],
            [[r['dataset'], r['n'], r['batch_size'] if r['batch_size'] is not None else 'UNKNOWN',
              'UNKNOWN' if r['forward_seconds'] is None else f"{r['forward_seconds']:.4f}",
              'UNKNOWN' if r['forward_ms_per_query'] is None else f"{r['forward_ms_per_query']:.4f}",
              str(r['expert_executed_rows'])] for r in timings],
            ['Direct live router calls have NO expert-score input. Dispatch hooks require total expert rows = query count.',
             'Forward includes gate, chosen expert and dispatch. One warmup batch excluded; no parameter update.',
             'Geometry/input preparation, gather/transfer/output copy and three-bank cache construction excluded and recorded separately.',
             'Natural timing uses fixed first TEST rows/task only, not a performance-selected sample or tuning.',
             'UNKNOWN: a legacy capture survived a failed provenance write but elapsed timing did not; capture reused, not remeasured.']))
        zero = []
        for split in ('train', 'val', 'test'):
            values = [r for r in methods if r['split'] == split]
            for flag in ('zero_selected_with_winner', 'near_zero_selected_with_winner'):
                part = [r for r in values if r[flag]]
                zero.append([split, flag, len(part), sum(r['true_winner_count'] for r in part),
                    sum(r['missed_winner_count'] for r in part),
                    f"{sum(r['missed_actual_regret_contribution_pp'] for r in part) / len(PROBLEMS):.4f}"])
        add(table_figure('Six descriptive zero / near-zero lists | Native winner support retained',
            ['Split', 'Group', 'Methods', 'Native wins', 'Missed native wins', 'ALL18 regret contribution pp'], zero,
            ['Strict zero: no picks and positive winner support. Near zero: (0, 0.1%] picks and positive winner support.',
             'Lists are disjoint within each split; full methods.csv retains ALL candidates, including zero-winner candidates.',
             'Regret is charged once to the missed native-winner row. No false-pick double charge; no frequency quotas.'] + notes[:2]))


def report(args, model):
    from . import r123_experiment as runner
    set_style(args.font)
    output = args.root / 'finish'
    train_receipt = require_identity(output / 'captures/train/receipt.json', model, 'metaTRAIN_cached_chosen_scores')
    lib_receipt = require_identity(output / 'lib/receipt.json', model, 'live_sparse_LIB')
    timing_receipt = require_identity(output / 'timing/receipt.json', model, 'live_sparse_TEST_fixed_prefix')
    inputs, ctx = dict(model['inputs']), context(model)
    for path in (output / 'captures/train/receipt.json', output / 'lib/receipt.json', output / 'timing/receipt.json'):
        remember(inputs, path)
    if set(train_receipt['rows']) != set(PROBLEMS) or set(timing_receipt['by_problem']) != set(PROBLEMS):
        raise ValueError('Require full18 metaTRAIN captures and direct live timing rows')
    target = read_json(args.fixed36)
    if len(target['targets']) != 36 or Path(target['identity']['data_root']).resolve() != Path(model['config']['data_root']).resolve():
        raise ValueError('Fixed36 must be the original current-release frozen target list')
    targets = {(r['problem'], r['method']) for r in target['targets']}
    if len(targets) != 36 or any(s not in model['config']['candidate_pools'][p] for p, s in targets):
        raise ValueError('Fixed36 duplicate or illegal candidate')
    remember(inputs, args.fixed36)
    predictions, actions_by_split = {}, {}
    for split in ('train', 'val', 'test'):
        predictions[split], actions_by_split[split] = {}, {}
        for problem in PROBLEMS:
            path = ((output / 'captures/train' / (problem + '.npz')) if split == 'train' else
                    model['run'] / 'evaluation' / split / 'selected_best/by_problem' / (problem + '.npz'))
            data = read_npz(path)
            bank = runner.open_bank_cache(args.bank_root, split, problem, ctx, run_root=model['run'])
            try:
                if (list(data['pool']) != model['config']['candidate_pools'][problem]
                        or list(data['pool_ids']) != model['config']['candidate_pool_ids'][problem]):
                    raise ValueError('Capture pool differs from unique complete-model protocol')
                actions_by_split[split][problem] = check_bank_join(data, bank)
            finally:
                bank.close()
            predictions[split][problem] = data
            remember(inputs, path)
            for expert in EXPERTS:
                remember(inputs, args.bank_root / 'prediction_cache' / split / problem / (expert + '.npz'))
    refs = fit_train_references({p: dict(split='train', actions=actions_by_split['train'][p],
        winner=predictions['train'][p]['winner']) for p in PROBLEMS})
    refs.update(capture_identity(model, 'TRAIN_only_references'))
    write_json(output / 'reports/train_references.json', refs)
    metrics, budgets, references, methods, fixed, paired = [], [], [], [], [], []
    budget_fields = ('bank_native_coverage_pct', 'native_bank_miss_pp', 'native_router_miss_pp',
        'native_error_pct', 'bank_quality_floor_pct', 'routing_quality_excess_pct', 'actual_regret_pct')
    for split in ('train', 'val', 'test'):
        for problem in PROBLEMS:
            data, actions = predictions[split][problem], actions_by_split[split][problem]
            metrics.append(metric_row(data, ctx['baselines'][problem], problem=problem, split=split))
            budgets.append(dict(problem=problem, split=split, **bank_decomposition(data, actions)))
            detail = method_rows(data, problem=problem, split=split)
            methods.extend(detail)
            _, _, regret = quality(data)
            for name, inside in (('FIXED36', True), ('OUTSIDE36', False)):
                rows = np.asarray([((problem, str(data['pool'][slot])) in targets) == inside for slot in data['winner']])
                fixed.append(dict(problem=problem, split=split, group=name, n=len(rows),
                    true_winner_count=int(rows.sum()),
                    missed_winner_count=int((rows & (data['pred'] != data['winner'])).sum()),
                    native_error_contribution_pp=100. * float((rows & (data['pred'] != data['winner'])).mean()),
                    actual_regret_contribution_pp=100. * float(regret[rows].sum()) / len(rows)))
            policies = [('TRAIN_global_fixed', refs['global_fixed_expert']),
                        ('TRAIN_task_prior', refs['task_prior_expert'][problem])]
            policies += [(name, i) for i, name in enumerate(EXPERTS)]
            for name, expert in policies:
                baseline_data = dict(data, pred=actions[:, expert])
                row = metric_row(baseline_data, ctx['baselines'][problem], problem=problem, split=split, policy=name)
                row['chosen_expert'] = EXPERTS[expert]
                references.append(row)
                paired.append(dict(problem=problem, split=split, reference=name,
                                   **paired_reference(data, actions[:, expert])))
        metrics.append(macro_row([r for r in metrics if r['split'] == split], CORE, split=split, policy='R123'))
        budgets.append(macro_row([r for r in budgets if r['split'] == split], budget_fields, split=split))
        for name in ('TRAIN_global_fixed', 'TRAIN_task_prior', *EXPERTS):
            references.append(macro_row([r for r in references if r['split'] == split and r['policy'] == name],
                                        CORE, split=split, policy=name))
        for group in ('FIXED36', 'OUTSIDE36'):
            values = [r for r in fixed if r['split'] == split and r['group'] == group]
            fixed.append(macro_row(values, ('native_error_contribution_pp', 'actual_regret_contribution_pp'), split=split, group=group))
    libs, timings = {}, []
    for problem, dataset in (('TSP', 'TSPLIB'), ('CVRP', 'CVRPLIB')):
        path = output / 'lib' / (dataset + '.npz')
        data = read_npz(path)
        validate_prediction(data)
        metrics.append(metric_row(data, ctx['baselines'][problem], problem=problem, split=dataset))
        libs[dataset] = data
        timings.append(dict(dataset=dataset, **lib_receipt['timing'][dataset]))
        remember(inputs, path)
        for raw_path, digest in lib_receipt['sources'][dataset].items():
            if file_hash(raw_path) != digest:
                raise ValueError('Native LIB input/label changed since capture')
            inputs[raw_path] = digest
    for p in PROBLEMS:
        timings.append(dict(dataset=p + ' TEST prefix', **timing_receipt['by_problem'][p]))
    nss = nss_comparisons(predictions, libs, model, args.nss_root, inputs)
    reports = output / 'reports'
    for filename, rows in (('metrics.csv', metrics), ('bank_decomposition.csv', budgets),
                           ('train_only_references.csv', references), ('paired_reference_deltas.csv', paired),
                           ('methods.csv', methods), ('fixed36_budget.csv', fixed),
                           ('nss_comparison.csv', nss), ('live_inference_timing.csv', timings)):
        write_csv(reports / filename, rows)
    fields = list(methods[0])
    for split in ('train', 'val', 'test'):
        for prefix, flag in (('zero', 'zero_selected_with_winner'), ('near_zero', 'near_zero_selected_with_winner')):
            write_csv(reports / f'{prefix}_methods_{split}.csv',
                [r for r in methods if r['split'] == split and r[flag]], fields)
    pdf = reports / 'R123_MODEL_REPORT.pdf'
    render_pdf(pdf, predictions, libs, model, metrics, budgets, references, nss, timings, methods)
    remember(inputs, Path(__file__))
    remember(inputs, Path(model['config']['meta_preparation']) / 'manifest.json')
    remember(inputs, Path(model['config']['meta_preparation']) / 'meta_train_indices.npz')
    remember(inputs, Path(model['config']['prepared_root']) / 'common_full_train_sbs.json')
    receipt = dict(capture_identity(model, 'complete_post_TEST_report'),
        complete=True, trained_by_finish=False, input_files_sha256=inputs,
        artifacts={str(path.relative_to(reports)): file_hash(path) for path in reports.iterdir()
                   if path.name != 'receipt.json' and path.suffix in ('.pdf', '.csv', '.json')},
        rows={s: {p: len(d['winner']) for p, d in predictions[s].items()} for s in predictions},
        lib_rows={s: len(d['winner']) for s, d in libs.items()},
        nss_status={r['dataset']: r['status'] for r in nss},
        six_zero_lists=True, row_certificate='source-record only; exhaustive content/base-copy novelty NOT certified',
        existing_runner_capture_attribution='locked run/update, pinned producer, exact source identities and selected bank scores; '
            'original VAL/TEST captures have no embedded checkpoint SHA; this is not a retroactive capture signature',
        inference='live gate plus one chosen expert; cached three-bank construction is NOT deployment timing',
        bank_precompute_query_rows={s: sum(len(d['winner']) for d in predictions[s].values()) for s in predictions},
        bank_precompute_expert_rows={s: len(EXPERTS) * sum(len(d['winner']) for d in predictions[s].values())
                                    for s in predictions},
        bank_precompute_accounting='full train+val+postlock test three-expert cache is required and not free; '
            'the helper does not record elapsed GPU time, so its numeric construction cost remains unavailable, not zero')
    write_json(reports / 'receipt.json', receipt)
    return receipt


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('command', choices=('describe', 'capture-train', 'lib', 'timing', 'report', 'finish'))
    result.add_argument('--root', required=True, type=Path, help='Literal R123 bank root; unique run is ROOT/model')
    result.add_argument('--bank-root', type=Path)
    result.add_argument('--input-cache-root', type=Path)
    result.add_argument('--device', default='cpu')
    result.add_argument('--predict-batch', type=int, default=128)
    result.add_argument('--lib-batch', type=int, default=1)
    result.add_argument('--reuse-legacy-lib-capture', action='append', choices=('TSPLIB', 'CVRPLIB'),
                        default=[], help='Explicit same-lock attribution for a surviving unsigned LIB capture; never overwrite it')
    result.add_argument('--timing-rows', type=int, default=128)
    result.add_argument('--font', type=Path, default=FONT)
    result.add_argument('--fixed36', type=Path, default=REFERENCE_ROOT / 'comparison_artifacts/target_list.json')
    result.add_argument('--nss-root', type=Path, default=REFERENCE_ROOT / 'nss_reference')
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    args.root = args.root.resolve()
    args.bank_root = (args.bank_root or args.root).resolve()
    if min(args.predict_batch, args.lib_batch, args.timing_rows) < 1:
        raise ValueError('Batch sizes and fixed timing-prefix length must be positive')
    if args.command == 'describe':
        result = dict(schema=SCHEMA, root=str(args.root), run=str(args.root / 'model'),
            output=str(args.root / 'finish'), commands=['capture-train', 'lib', 'timing', 'report'],
            required='unique VAL lock and completed selected VAL+TEST, all7200 updates; no selection here',
            new_training=False, expert_bank=list(EXPERTS), metaTRAIN_per_task=10886,
            reference_fit='metaTRAIN native correctness only; equal-task global fixed + task-conditioned constant prior',
            live_inference='raw gate then exactly one expert; no cached expert-score inputs',
            test_results_read=False, cuda_initialized=False)
    else:
        import torch
        torch.set_num_threads(2)
        model = locked(args)
        if args.command == 'finish':
            capture_train(args, model)
            lib(args, model)
            timing(args, model)
            result = report(args, model)
        else:
            result = {'capture-train': capture_train, 'lib': lib, 'timing': timing, 'report': report}[args.command](args, model)
        if file_hash(model['run'] / 'best.pt') != model['lock']['checkpoint_sha256']:
            raise ValueError('Locked checkpoint changed during postprocessing')
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False), flush=True)
    return result


if __name__ == '__main__':
    main()

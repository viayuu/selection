"""Same-lock R124 LIB inference and offline full18 PDF/CSV finish.

Queued by the parent AFTER locked TEST. No training, checkpoint selection,
frozen-input rebuild, historical3895 metric reuse or source/prediction rewrite.
All new outputs live under ROOT/model/finish. ``report`` never runs inference.

    python -m code.V4.r124_finish all --root /path/to/R124/launch_v2 --device cuda:0
    python -m code.V4.r124_finish report --root /path/to/R124/launch_v2
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
import fcntl
import hashlib
import json
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np

from code.unified_selector.registry import PROBLEMS


V4 = Path(__file__).resolve().parent
BASE = V4 / 'runs/R124_top2_reranking'
DEFAULT_ROOT = BASE / 'launch_v2'
DEFAULT_BANK = V4 / 'runs/R123_heterogeneous_moe'
DEFAULT_NSS = V4 / 'runs/R84_new90k_attention/nss_reference'
EXPERT_SHA = '31c0c75d66b1bde6ad533909a9c04b0036e80af4c4a5775b01b957ce310e6d1b'
LIBS = {'TSPLIB': ('TSP', 49), 'CVRPLIB': ('CVRP', 100)}
WINNER_RULE = 'saved native/FP64 ind; exact native ties retained; NOT FP32 argmin'
MODEL_FILES = ('r124_model.py', 'r124_experiment.py')
METRICS = ('top1_pct', 'mean_cost', 'sbs_cost', 'oracle_cost', 'vs_sbs_pct',
           'vs_oracle_pct', 'actual_regret_pct')


def read_json(path):
    return json.loads(Path(path).read_text())


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value, *, immutable=False):
    path = Path(path)
    if immutable and path.exists():
        if read_json(path) != value:
            raise ValueError(f'Existing finish identity changed: {path}')
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name('.' + path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def save_npz(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f'Never overwrite existing predictions: {path}')
    temporary = path.with_name('.' + path.name + '.tmp')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, **data)
    temporary.replace(path)


@contextmanager
def exclusive(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def archive(path, problem=None):
    prefix = problem + '__' if problem is not None else ''
    with np.load(path, allow_pickle=False) as stream:
        return {k[len(prefix):]: stream[k].copy() for k in stream.files if k.startswith(prefix)}


def stable_rank(scores, ids):
    scores, ids = np.asarray(scores), np.asarray(ids)
    if scores.ndim != 2 or ids.shape != scores.shape[1:] or len(np.unique(ids)) != len(ids):
        raise ValueError('Scores and distinct global candidate IDs must align')
    if not np.isfinite(scores).all():
        raise ValueError('All legal candidate scores must be finite')
    order = np.argsort(ids, kind='stable')
    return order[np.argsort(-scores[:, order], axis=1, kind='stable')]


def validate_data(data, *, expected_rows=None, expected_pool=None, expected_ids=None):
    costs, winner = data['costs'], data['winner']
    n, k = costs.shape
    if (not n or costs.dtype != np.float64 or winner.dtype != np.int64
            or winner.shape != (n,) or np.any(winner < 0) or np.any(winner >= k)
            or not np.isfinite(costs).all() or np.any(costs <= 0)
            or len(data['pool']) != k or len(set(data['pool'].tolist())) != k
            or not np.array_equal(costs[np.arange(n), winner], costs.min(1))
            or not np.array_equal(data['indices'], np.arange(n))):
        raise ValueError('Require original native FP64 costs, winner and complete row order')
    if expected_rows is not None and n != expected_rows:
        raise ValueError(f'Expected same NEW{expected_rows} population; got {n}, never reuse old3895 metrics')
    for name, expected in (('pool', expected_pool), ('pool_ids', expected_ids)):
        if expected is not None and not np.array_equal(data[name], expected):
            raise ValueError('Changed legal candidate deployments/order: ' + name)
    if data['logits'].shape != costs.shape:
        raise ValueError('Full-pool prediction/cost shape differs')
    pred = stable_rank(data['logits'], data['pool_ids'])[:, 0]
    if 'pred' in data and not np.array_equal(data['pred'], pred):
        raise ValueError('Saved action differs from stable global-ID decision')
    return pred


def locked_inputs(args):
    """No TEST/LIB data is read until the complete original gate has passed."""
    from . import r124_experiment as runner
    run = args.root / 'model'
    config = read_json(run / 'protocol.json')
    gate = SimpleNamespace(root=args.root, data_root=Path(config['data_root']), state_hash=None)
    lock, checked = runner.authorize_test(gate)
    if checked != config:
        raise PermissionError('R124 protocol changed while opening finish stage')
    method = read_json(args.root / 'preparation/method_config.json')
    source_hashes = {name: file_hash(V4 / name) for name in MODEL_FILES}
    if any(source_hashes[name] != method['software_source_sha256'][name]
           or source_hashes[name] != config['source_sha256'][name] for name in MODEL_FILES):
        raise PermissionError('Frozen model/runner SHA changed; finish never repairs those sources')
    saved = read_json(run / 'test_results.json')
    receipt = read_json(run / 'test_complete.json')
    final = read_json(run / 'final_metrics.json')
    if (saved.get('complete') is not True or receipt.get('full_tests') != 1
            or saved.get('checkpoint_sha256') != lock['checkpoint_sha256']
            or receipt.get('checkpoint_sha256') != lock['checkpoint_sha256']
            or saved.get('state_hash') != lock['state_hash'] or final.get('test_read') is not True
            or not final.get('complete') or final['test'] != saved['result']
            or config['first_stage']['checkpoint_sha256'] != EXPERT_SHA):
        raise PermissionError('Finish requires the completed SAME unique R124 TEST checkpoint')
    identity = dict(experiment='R124', checkpoint_sha256=lock['checkpoint_sha256'],
        selected_update=lock['selected_update'], state_hash=lock['state_hash'],
        first_stage_checkpoint_sha256=EXPERT_SHA, frozen_source_sha256=source_hashes,
        run_root=str(args.root), data_root=config['data_root'], winner_rule=WINNER_RULE,
        model_factory='code.V4.r124_experiment.load_checkpoint',
        finish_source_sha256=file_hash(__file__))
    return run, config, lock, final, identity


def native_metrics(data, prediction, baseline, aliases=None):
    costs, winner = data['costs'], data['winner']
    pool = data['pool'].tolist()
    aliases = aliases or {}
    if baseline['selection_split'] != 'train':
        raise ValueError('SBS must be fitted on frozen TRAIN81000, not VAL/TEST/LIB')
    names = [aliases.get(name, name) for name in pool]
    slot = names.index(aliases.get(baseline['sbs_solver'], baseline['sbs_solver']))
    oracle = costs.min(1)
    selected = costs[np.arange(len(costs)), prediction]
    mean, sbs, best = float(selected.mean()), float(costs[:, slot].mean()), float(oracle.mean())
    return dict(n=len(costs), top1_pct=float(np.mean(prediction == winner) * 100),
        mean_cost=mean, sbs_cost=sbs, oracle_cost=best, vs_sbs_pct=100. * (mean / sbs - 1.),
        vs_oracle_pct=100. * (mean / best - 1.),
        actual_regret_pct=float(np.mean((selected - oracle) / oracle) * 100),
        sbs_name=baseline['sbs_solver'], sbs_selection_split='frozen TRAIN81000',
        top1_fp32_argmin_pct=float(np.mean(prediction == costs.astype(np.float32).argmin(1)) * 100))


def reranking_diagnostics(data):
    pred = validate_data(data)
    costs, winner, ids = data['costs'], data['winner'], data['pool_ids']
    if data['stage1_scores'].shape != costs.shape:
        raise ValueError('First stage must score the same complete legal pool')
    pair = stable_rank(data['stage1_scores'], ids)[:, :2]
    if not np.array_equal(data['pair_indices'], pair):
        raise ValueError('Reranker did not use the actual whole-pool first-stage Top2')
    difference, quality = data['difference'], data['pair_quality']
    if (difference.shape != (len(costs),) or quality.shape != (len(costs),)
            or not np.isfinite(difference).all() or not np.isfinite(quality).all()):
        raise ValueError('Separate native/quality heads must preserve per-row finite outputs')
    first, second = pair.T
    expected = np.where(difference > 0, first, np.where(difference < 0, second,
                        np.where(ids[first] < ids[second], first, second)))
    if not np.array_equal(pred, expected):
        raise ValueError('Final action is not the native head decision; quality must NOT select')
    remaining = np.ones_like(costs, dtype=bool)
    remaining[np.arange(len(costs))[:, None], pair] = False
    if not np.array_equal(data['logits'][remaining], data['stage1_scores'][remaining]):
        raise ValueError('Non-Top2 scores changed')
    rest = np.where(remaining, data['logits'], -np.inf).max(1)
    if np.any(np.take_along_axis(data['logits'], pair, axis=1).min(1) <= rest):
        raise ValueError('Both Top2 candidates must remain strictly ahead of rank3')
    base = pair[:, 0]
    covered = (pair == winner[:, None]).any(1)
    correct = pred == winner
    corrected = (base != winner) & correct
    damaged = (base == winner) & ~correct
    oracle = costs.min(1)
    pair_cost = np.take_along_axis(costs, pair, axis=1).min(1)
    row = np.arange(len(costs))
    floor = (pair_cost - oracle) / oracle
    excess = (costs[row, pred] - pair_cost) / oracle
    base_excess = (costs[row, base] - pair_cost) / oracle
    actual = (costs[row, pred] - oracle) / oracle
    np.testing.assert_allclose(floor + excess, actual, rtol=0, atol=1e-12)
    np.testing.assert_allclose(correct.mean() - (base == winner).mean(),
                               corrected.mean() - damaged.mean(), rtol=0, atol=1e-12)
    target = (costs[row, second] - costs[row, first]) / oracle
    residual = (quality - target) / .01
    huber = np.where(np.abs(residual) < 1., .5 * residual ** 2, np.abs(residual) - .5)
    sign = np.where(winner == first, 1., -1.)
    native_ce = np.logaddexp(0., -sign * difference) * covered
    summary = dict(n=len(costs), native_covered_count=int(covered.sum()),
        coverage_native_pct=float(covered.mean() * 100), first_stage_top1_pct=float((base == winner).mean() * 100),
        final_top1_pct=float(correct.mean() * 100), native_correct_count=int(correct.sum()),
        conditional_native_top1_pct=float(correct.sum() / covered.sum() * 100) if covered.any() else None,
        corrected_count=int(corrected.sum()), damaged_count=int(damaged.sum()),
        correction_pct=float(corrected.mean() * 100), damage_pct=float(damaged.mean() * 100),
        net_top1_pp=float((corrected.mean() - damaged.mean()) * 100),
        recall_floor_regret_pct=float(floor.mean() * 100), reranking_excess_regret_pct=float(excess.mean() * 100),
        stage1_excess_regret_pct=float(base_excess.mean() * 100), actual_regret_pct=float(actual.mean() * 100),
        native_bce_all_rows=float(native_ce.mean()), pair_quality_huber_all_rows=float(huber.mean()),
        quality_loss_is_not_success=True)
    candidates = []
    for slot, name in enumerate(data['pool'].tolist()):
        native = winner == slot
        candidates.append(dict(method=name, global_id=int(ids[slot]), n=len(costs), native_winner_count=int(native.sum()),
            native_winner_pct=float(native.mean() * 100), stage1_selected=int((base == slot).sum()),
            reranker_selected=int((pred == slot).sum()), top2_occurrences=int((pair == slot).any(1).sum()),
            winner_in_top2=int((native & covered).sum()), winner_outside_top2=int((native & ~covered).sum()),
            native_candidate_coverage_pct=float(covered[native].mean() * 100) if native.any() else None,
            native_corrected=int((native & corrected).sum()), native_damaged=int((native & damaged).sum())))
    return summary, candidates


def macro_rows(rows, split):
    if len(rows) != len(PROBLEMS) or {r['problem'] for r in rows} != set(PROBLEMS):
        raise ValueError('ALL must contain each of the full18 tasks exactly once')
    result = dict(split=split, problem='ALL', n=sum(r['n'] for r in rows),
        aggregation='equal task weight; raw mean cost descriptive only')
    for key in rows[0]:
        values = [r.get(key) for r in rows]
        if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in values):
            result[key] = sum(values) if key.endswith('_count') or key == 'n' else float(np.mean(values))
    if 'coverage_native_pct' in result:
        cover = result['coverage_native_pct']
        result['conditional_native_top1_pct'] = result['final_top1_pct'] / cover * 100 if cover else None
    return result


def align_reference(current, other, aliases=None, *, require_source=False):
    """Full-row/full-cost alignment, never a three-row sample or old3895 summary."""
    aliases = aliases or {}
    left = [aliases.get(x, x) for x in current['pool'].tolist()]
    right = [aliases.get(x, x) for x in other['pool'].tolist()]
    if len(set(right)) != len(right) or set(left) != set(right) or len(current['costs']) != len(other['costs']):
        raise ValueError('Reference population or legal deployment set differs (old3895 is invalid)')
    order = other['indices']
    if len(np.unique(order)) != len(order) or set(order) != set(current['indices']):
        raise ValueError('Reference row identities differ')
    lookup = {int(value): i for i, value in enumerate(order)}
    rows = np.array([lookup[int(value)] for value in current['indices']])
    columns = [right.index(x) for x in left]
    if not np.array_equal(current['costs'], other['costs'][rows][:, columns]):
        raise ValueError('Reference does not use the SAME native FP64 cost source on ALL rows')
    if [left[i] for i in current['winner']] != [right[i] for i in other['winner'][rows]]:
        raise ValueError('Reference native winner identity differs')
    if require_source:
        original = other.get('release_source_indices', other.get('source_indices'))
        if original is None or not np.array_equal(current['release_source_indices'], original[rows]):
            raise ValueError('Reference release source IDs differ')
    pred = np.asarray(other['pred'])[rows]
    if pred.shape != (len(rows),) or not np.issubdtype(pred.dtype, np.integer) or np.any(pred < 0) or np.any(pred >= len(right)):
        raise ValueError('Invalid saved reference action; do not replace NSS tie handling')
    mapping = np.array([left.index(x) for x in right])
    return mapping[pred]


def load_selected(args, run, config, lock, final):
    predictions, identities = {}, {}
    expected_model_hash = final['test']['identity']['model_state_sha256']
    for split in ('val', 'test'):
        directory = run / 'evaluation' / split / 'selected_best'
        identity = read_json(directory / 'identity.json')
        if (identity != final[split]['identity'] or identity['experiment'] != 'R124'
                or identity['state_hash'] != lock['state_hash']
                or identity['successful_updates'] != lock['selected_update']
                or identity['model_state_sha256'] != expected_model_hash
                or identity['first_stage_checkpoint_sha256'] != EXPERT_SHA
                or split == 'test' and identity['checkpoint_sha256'] != lock['checkpoint_sha256']):
            raise ValueError('Selected VAL/TEST prediction identity differs from single lock')
        path = directory / 'rank000.npz'
        identities[split.upper()] = dict(path=str(path), sha256=file_hash(path), identity=identity)
        for problem in PROBLEMS:
            data = archive(path, problem)
            validate_data(data, expected_rows=4500, expected_pool=config['candidate_pools'][problem],
                          expected_ids=config['candidate_pool_ids'][problem])
            with np.load(Path(config['data_root']) / (problem + split) / 'input_metadata.npz', allow_pickle=False) as stream:
                source = stream['source_index']
            if not np.array_equal(data['release_source_indices'], source):
                raise ValueError('Saved prediction does not match frozen NEW4500 release membership')
            saved = final[split]['per_problem'][problem]
            pred = validate_data(data)
            metrics = native_metrics(data, pred, config['baselines'][problem], config['scenario_solver_aliases'])
            for key, actual in (('top1', metrics['top1_pct'] / 100.), ('mean_cost', metrics['mean_cost']),
                                ('actual_regret_pct', metrics['actual_regret_pct'])):
                np.testing.assert_allclose(actual, saved[key], rtol=0, atol=1e-9)
            data['inference_device'] = np.asarray(saved['inference_cost']['device'])
            data['device_name'] = np.asarray('not recorded in original forward archive; see launcher resource receipt')
            predictions[split.upper(), problem] = data
    return predictions, identities


def validate_val_reference(args, config, predictions):
    from . import r123_prediction_cache as bank
    context = bank.context(Path(config['prepared_root']))
    records = {}
    for problem in PROBLEMS:
        current = predictions['VAL', problem]
        data = bank.read_expert(args.shared_bank_root, 'val', problem, 'R77B', context)
        if (data['metadata']['checkpoint_sha256'] != EXPERT_SHA
                or len(data['logits']) != 4500 or not data['metadata']['complete']
                or not np.array_equal(data['row_indices'], current['indices'])
                or not np.array_equal(data['source_indices'], current['release_source_indices'])
                or not np.array_equal(data['pool'], current['pool'])
                or not np.array_equal(data['pool_ids'], current['pool_ids'])
                or not np.array_equal(data['logits'], current['stage1_scores'])):
            raise ValueError('R77B VAL reference must be the official shared NEW4500 predictions')
        path = bank.archive_path(args.shared_bank_root, 'val', problem, 'R77B')
        records[problem] = dict(path=str(path), sha256=file_hash(path), n=4500,
            checkpoint_sha256=EXPERT_SHA, cost_source='same frozen current native VAL4500 cost matrix')
    return records


def load_lib_raw(problem, dataset, config):
    from .nss_comparison import read_costs
    raw = read_costs(problem, dataset)
    ids = config['candidate_pool_ids'][problem]
    if set(raw['pool_ids']) != set(ids):
        raise ValueError('LIB and locked current model legal deployments differ')
    columns = [list(raw['pool_ids']).index(i) for i in ids]
    mapping = np.array([ids.index(i) for i in raw['pool_ids']])
    return dict(costs=raw['costs'][:, columns], winner=mapping[raw['winner']],
        pool=np.asarray(config['candidate_pools'][problem]), pool_ids=np.asarray(ids, dtype=np.int64),
        indices=np.arange(len(raw['winner']), dtype=np.int64), directory=raw['directory'],
        original_pool=raw['pool'], original_pool_ids=raw['pool_ids'],
        original_costs=raw['costs'], original_winner=raw['winner'])


def full_forward(model, loader, raw, batch_size, device):
    """One complete f0+reranker pass. Feature work is timed separately, never hidden."""
    import torch
    from .local_geometry import local_geometry
    from .r124_model import query_inputs
    from .tensor_loader import TensorBatchLoader
    values = {k: [] for k in ('logits', 'stage1_scores', 'pair_indices', 'difference', 'pair_quality')}
    forward_seconds, feature_seconds = 0., 0.
    for batch in TensorBatchLoader(loader.batch, batch_size, False, False):
        inputs = query_inputs(batch)
        inputs['pool_ids'] = torch.as_tensor(raw['pool_ids'], dtype=torch.long, device=device)
        inputs['solver_mask'] = torch.ones(len(inputs['n']), len(raw['pool']), dtype=torch.bool, device=device)
        if inputs['kind'] == 'coord' and 'node_geom' not in inputs:
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            started = time.perf_counter()
            inputs['node_geom'] = local_geometry(inputs['node'][..., :2], inputs['node_mask']).float()
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            feature_seconds += time.perf_counter() - started
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        output = model(inputs)
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        forward_seconds += time.perf_counter() - started
        for key in values:
            values[key].append(output[key].detach().cpu().numpy())
    data = {key: np.concatenate(parts) for key, parts in values.items()}
    data.update({key: raw[key] for key in ('costs', 'winner', 'pool', 'pool_ids', 'indices')})
    data.update(nodes=np.asarray(loader.lengths), forward_seconds=np.asarray(forward_seconds),
        feature_seconds=np.asarray(feature_seconds), forward_batch_cap=np.asarray(batch_size),
        full_pipeline_timed=np.asarray(True), inference_device=np.asarray(str(device)),
        device_name=np.asarray(torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU'))
    data['pred'] = validate_data(data)
    reranking_diagnostics(data)
    return data


def lib_stage(args):
    run, config, lock, final, identity = locked_inputs(args)
    output = run / 'finish'
    with exclusive(output / 'lib.lock'):
        write_json(output / 'identity.json', identity, immutable=True)
        model, results = None, {}
        for dataset, (problem, count) in LIBS.items():
            raw = load_lib_raw(problem, dataset, config)
            if len(raw['winner']) != count:
                raise ValueError('Original LIB instance population changed')
            directory = Path(raw['directory'])
            binding = dict(identity, dataset=dataset, problem=problem, expected_rows=count,
                source_sha256={name: file_hash(directory / name) for name in ('dataset.pkl', 'raw_label.pkl')},
                candidate_pool=raw['pool'].tolist(), pool_ids=raw['pool_ids'].tolist(),
                batch_size=args.lib_batch_size, requested_device=args.device, full_stage0_plus_reranker=True,
                quality_head_used_for_action=False)
            target = output / 'lib_evaluation'
            path = target / 'predictions' / (dataset + '.npz')
            protocol_path = target / (dataset + '_protocol.json')
            receipt_path = target / (dataset + '_results.json')
            write_json(protocol_path, binding, immutable=True)
            if path.exists():
                data = archive(path)
                if (str(data['identity_json'].item()) != json.dumps(binding, sort_keys=True)
                        or receipt_path.exists() and read_json(receipt_path)['predictions_sha256'] != file_hash(path)):
                    raise ValueError('Existing finish LIB archive differs from this lock/source')
            else:
                import torch
                from . import r124_experiment as runner
                from .r77_data import parameter_hash
                from .train import make_loader
                torch.set_num_threads(2)
                device = torch.device(args.device)
                if model is None:
                    model, state = runner.load_checkpoint(run / 'best.pt', device)
                    if (state['args'] != config or state['successful_updates'] != lock['selected_update']
                            or parameter_hash(model.state_dict()) != final['test']['identity']['model_state_sha256']):
                        raise ValueError('LIB factory did not restore the same locked complete inference model')
                    model.eval().requires_grad_(False)
                    buffers = {k: v.detach().cpu().clone() for k, v in model.named_buffers()}
                loader = make_loader(problem, dataset, args.lib_batch_size, 0, coord_augment=0,
                                     shuffle=False, cache_device=device)
                if loader.size != count:
                    raise ValueError('LIB loader lost instances')
                original_ids = np.asarray(raw['original_pool_ids'])
                if not np.array_equal(loader.batch['pool_ids'].cpu().numpy(), original_ids):
                    raise ValueError('LIB loader candidate IDs differ from original cost source')
                np.testing.assert_array_equal(loader.batch['ind'].cpu().numpy(), raw['original_winner'])
                np.testing.assert_array_equal(loader.batch['costs'].cpu().numpy(), raw['original_costs'].astype(np.float32))
                with torch.inference_mode():
                    data = full_forward(model, loader, raw, args.lib_batch_size, device)
                for key, value in model.named_buffers():
                    torch.testing.assert_close(value.detach().cpu(), buffers[key], rtol=0, atol=0)
                data['identity_json'] = np.asarray(json.dumps(binding, sort_keys=True))
                save_npz(path, data)
                del loader
            validate_data(data, expected_rows=count, expected_pool=raw['pool'], expected_ids=raw['pool_ids'])
            if not np.array_equal(data['costs'], raw['costs']) or not np.array_equal(data['winner'], raw['winner']):
                raise ValueError('Saved LIB costs/native winner differ from original source')
            result = dict(protocol=binding, predictions_sha256=file_hash(path), complete=True,
                full_stage0_plus_reranker_seconds=float(data['forward_seconds']),
                feature_preparation_seconds=float(data['feature_seconds']))
            write_json(receipt_path, result, immutable=True)
            results[dataset] = result
        locked_inputs(args)
        write_json(output / 'lib_complete.json', dict(identity=identity, datasets=results, complete=True), immutable=True)
        return results


def nss_comparisons(args, config, predictions, lock):
    plan_path = args.nss_root / 'protocol.json'
    if not plan_path.exists():
        if not args.allow_missing_nss:
            raise FileNotFoundError('Reuse complete same-new4500 NSS reference; do not retrain or use old3895 metrics')
        plan = None
    else:
        plan = read_json(plan_path)
        if (Path(plan['data_root']).resolve() != Path(config['data_root']).resolve()
                or plan.get('retrained_on_new90k') is not False
                or plan.get('expected_test_rows_per_task') != 4500):
            raise ValueError('NSS protocol does not describe the SAME NEW4500 population')
    rows = []
    for label, split, problem in (('TSP TEST', 'TEST', 'TSP'), ('CVRP TEST', 'TEST', 'CVRP'),
                                  ('TSPLIB', 'TSPLIB', 'TSP'), ('CVRPLIB', 'CVRPLIB', 'CVRP')):
        data = predictions[split, problem]
        pred = validate_data(data)
        base = stable_rank(data['stage1_scores'], data['pool_ids'])[:, 0]
        stem = 'test' if split == 'TEST' else split
        path = args.nss_root / problem / (stem + '_predictions.npz')
        row = dict(dataset=label, n=len(pred), r124_top1_native_pct=float(np.mean(pred == data['winner']) * 100),
            r77b_top1_native_pct=float(np.mean(base == data['winner']) * 100),
            r124_mean_cost=float(data['costs'][np.arange(len(pred)), pred].mean()),
            r77b_mean_cost=float(data['costs'][np.arange(len(base)), base].mean()),
            nss_top1_native_pct=None, nss_mean_cost=None, nss_status='MISSING',
            nss_weights='locked historical R79 weights; no retraining; NEW4500 TEST, original LIB',
            winner_rule=WINNER_RULE, nss_prediction_path=str(path), nss_checkpoint_sha256=None,
            r124_checkpoint_sha256=lock['checkpoint_sha256'], r77b_checkpoint_sha256=EXPERT_SHA,
            data_version=Path(config['data_root']).name if split == 'TEST' else 'original native ' + split,
            candidate_pool='|'.join(data['pool'].tolist()), candidate_global_ids='|'.join(map(str, data['pool_ids'])),
            cost_source='exact full native FP64 matrix shared by all three methods')
        if plan is not None and path.exists():
            info = read_json(path.with_name(stem + '_metrics.json'))
            expected = plan['checkpoints'][problem]['sha256']
            if info['checkpoint']['sha256'] != expected:
                raise ValueError('NSS prediction checkpoint differs from locked historical weights')
            reference = archive(path)
            other_pred = align_reference(data, reference, config['scenario_solver_aliases'], require_source=split == 'TEST')
            row.update(nss_top1_native_pct=float(np.mean(other_pred == data['winner']) * 100),
                nss_mean_cost=float(data['costs'][np.arange(len(pred)), other_pred].mean()),
                nss_checkpoint_sha256=expected, nss_status='complete', nss_prediction_sha256=file_hash(path))
        elif not args.allow_missing_nss:
            raise FileNotFoundError('Required NSS comparison missing: ' + str(path))
        rows.append(row)
    return rows


def build_tables(predictions, config):
    metrics, comparisons, diagnostics, candidates, distributions, timing = [], [], [], [], [], []
    for (split, problem), data in predictions.items():
        pred = validate_data(data)
        base = stable_rank(data['stage1_scores'], data['pool_ids'])[:, 0]
        baseline = config['baselines'][problem]
        current = native_metrics(data, pred, baseline, config['scenario_solver_aliases'])
        original = native_metrics(data, base, baseline, config['scenario_solver_aliases'])
        common = dict(split=split, problem=problem, winner_rule=WINNER_RULE,
                      data_version=Path(config['data_root']).name if split in ('VAL', 'TEST') else 'original native LIB')
        metrics.append(dict(common, **current))
        compare = dict(common, n=len(pred), first_stage_checkpoint_sha256=EXPERT_SHA,
            r77b_reference='official complete frozen R77B update34776; SAME NEW4500 costs, not old3895 metrics'
                if split in ('VAL', 'TEST') else 'same original LIB inputs and native costs; recorded complete f0 forward',
            reference_origin='independent shared R123 R77B VAL archive' if split == 'VAL'
                else 'stage1_scores recorded by the same post-lock full-composite forward')
        for key in METRICS:
            compare['r77b_' + key], compare['r124_' + key] = original[key], current[key]
        compare.update(top1_delta_pp=current['top1_pct'] - original['top1_pct'],
            actual_regret_delta_pp=current['actual_regret_pct'] - original['actual_regret_pct'])
        comparisons.append(compare)
        summary, candidate_rows = reranking_diagnostics(data)
        diagnostics.append(dict(common, **summary))
        candidates.extend(dict(common, **row) for row in candidate_rows)
        for slot, name in enumerate(data['pool'].tolist()):
            distributions.append(dict(common, method=name, global_id=int(data['pool_ids'][slot]), n=len(pred),
                model_selected_count=int((pred == slot).sum()), model_selected_pct=float((pred == slot).mean() * 100),
                native_winner_count=int((data['winner'] == slot).sum()),
                native_winner_pct=float((data['winner'] == slot).mean() * 100),
                r77b_selected_count=int((base == slot).sum()), r77b_selected_pct=float((base == slot).mean() * 100),
                model_top1_pct=current['top1_pct']))
        full = bool(data['full_pipeline_timed'])
        if split != 'VAL' and not full:
            raise ValueError('TEST/LIB cost must include the COMPLETE frozen first stage + reranker')
        seconds = float(data['forward_seconds'])
        if not np.isfinite(seconds) or seconds <= 0:
            raise ValueError('Actual inference time must be present; do not synthesize readout-only deployment cost')
        timing.append(dict(common, n=len(pred), seconds=seconds, milliseconds_per_instance=1000. * seconds / len(pred),
            full_stage0_plus_reranker=full, batch_cap=int(data['forward_batch_cap']),
            device=str(data.get('inference_device', 'not recorded')), device_name=str(data.get('device_name', 'not recorded')),
            feature_preparation_seconds=float(data.get('feature_seconds', 0.)) if split in LIBS else None,
            scope='complete frozen R77B + independent reranker' if full else 'reranker-only; cached f0 is NOT free',
            measurement='actual FP32 first-pass batched forward, CUDA synchronized if CUDA; no warmup',
            excludes='I/O, host-to-device, solver calls; LIB geometry preparation reported separately'))
    for split in ('VAL', 'TEST'):
        for rows in (metrics, comparisons, diagnostics):
            rows.insert(0, macro_rows([r for r in rows if r['split'] == split], split))
        rows = [r for r in timing if r['split'] == split]
        macro = macro_rows(rows, split)
        macro.pop('batch_cap', None)
        macro['seconds'] = sum(r['seconds'] for r in rows)
        macro['full_stage0_plus_reranker'] = split == 'TEST'
        macro['scope'] = 'full composite model forward' if split == 'TEST' else 'cached-f0 reranker evaluation, NOT deployment time'
        timing.insert(0, macro)
    return dict(metrics=metrics, r77b_comparison=comparisons, reranking=diagnostics,
                candidate_coverage=candidates, selection_distribution=distributions, inference_cost=timing)


def plotting():
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import font_manager
    import matplotlib.pyplot as plt
    for path in (Path('/usr/share/fonts/google-noto-cjk/NotoSansCJK-Regular.ttc'),
                 Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')):
        if path.exists():
            font_manager.fontManager.addfont(str(path))
            matplotlib.rcParams['font.family'] = font_manager.FontProperties(fname=path).get_name()
            break
    else:
        raise FileNotFoundError('Install/locate a CJK font before rendering required native-winner distribution titles')
    matplotlib.rcParams.update({'axes.unicode_minus': False, 'pdf.fonttype': 42, 'font.size': 10,
        'axes.spines.top': False, 'axes.spines.right': False})
    return plt


def fmt(value, digits=4):
    return 'MISSING' if value is None else f'{value:.{digits}f}' if isinstance(value, (float, np.floating)) else str(value)


def table_page(plt, title, columns, rows, note, *, font=9):
    fig = plt.figure(figsize=(15.5, 10.5))
    fig.text(.04, .95, title, fontsize=17, weight='bold')
    ax = fig.add_axes([.035, .16, .93, .73])
    ax.axis('off')
    height = min(1., .055 * (len(rows) + 1))
    table = ax.table(cellText=[[fmt(value) for value in row] for row in rows], colLabels=columns,
                     cellLoc='center', bbox=[0, 1 - height, 1, height])
    table.auto_set_font_size(False)
    table.set_fontsize(font)
    for (row, _), cell in table.get_celld().items():
        cell.set_linewidth(.3)
        if row == 0:
            cell.set_facecolor('#dce9e5')
            cell.set_text_props(weight='bold')
        elif row % 2 == 0:
            cell.set_facecolor('#f3f5f2')
    fig.text(.04, .11, note, fontsize=9, va='top', wrap=True)
    return fig


def distribution_figure(plt, problem, predictions, config, lock):
    splits = ['TEST'] + [dataset for dataset, (p, _) in LIBS.items() if p == problem]
    fig, axes = plt.subplots(1, len(splits), figsize=(15.5, 9), squeeze=False)
    for ax, split in zip(axes[0], splits):
        data = predictions[split, problem]
        pred = validate_data(data)
        pool = data['pool'].tolist()
        n = len(pred)
        selection = np.bincount(pred, minlength=len(pool))
        winners = np.bincount(data['winner'], minlength=len(pool))
        y = np.arange(len(pool))
        ax.barh(y - .18, selection / n * 100, .34, color='#187f78', label='\u6a21\u578b\u9009\u62e9')
        ax.barh(y + .18, winners / n * 100, .34, color='#d88f38', label='\u771f\u5b9e native winner')
        for offset, values in ((-.18, selection), (.18, winners)):
            for index, count in enumerate(values):
                ax.text(count / n * 100 + .7, index + offset, f'{count} ({count / n * 100:.2f}%)', va='center', fontsize=8)
        ax.set_yticks(y, pool)
        ax.invert_yaxis()
        ax.set_xlim(0, 121)
        ax.set_xlabel('Frequency (%) on the SAME dataset; count / n')
        ax.set_title(f'{problem} {split} | n={n}\n\u6a21\u578bTop1={np.mean(pred == data["winner"]) * 100:.2f}%')
        ax.legend(loc='lower right', fontsize=9)
        ax.grid(axis='x', alpha=.15)
    fig.suptitle(f'R124 | update {lock["selected_update"]} | {Path(config["data_root"]).name}', fontsize=12)
    fig.text(.04, .025, 'Winner: saved native/FP64 ind, NOT FP32 argmin. Fixed candidate order; true-winner frequency is NOT model accuracy.', fontsize=9)
    fig.tight_layout(rect=(.02, .05, .98, .93))
    return fig


def diagnostics_figure(plt, tables):
    rows = [next(r for r in tables['reranking'] if r['split'] == 'TEST' and r['problem'] == p) for p in PROBLEMS]
    fig, axes = plt.subplots(1, 3, figsize=(17, 10), sharey=True)
    y = np.arange(len(rows))
    axes[0].barh(y - .18, [r['correction_pct'] for r in rows], .34, label='Correction', color='#187f78')
    axes[0].barh(y + .18, [r['damage_pct'] for r in rows], .34, label='Damage', color='#c45a45')
    axes[0].set_xlabel('Unconditional instance rate (%)')
    axes[0].set_title('Net Top1 change = correction - damage')
    axes[1].barh(y, [r['coverage_native_pct'] for r in rows], color='#74969a')
    axes[1].set_xlim(0, 100)
    axes[1].set_xlabel('Saved native winner in actual Top2 (%)')
    axes[1].set_title('Coverage is a ceiling, not achieved accuracy')
    floor = np.array([r['recall_floor_regret_pct'] for r in rows])
    axes[2].barh(y, floor, label='Candidate-set recall floor', color='#d88f38')
    axes[2].barh(y, [r['reranking_excess_regret_pct'] for r in rows], left=floor,
                 label='Reranking excess', color='#187f78')
    axes[2].set_xlabel('Actual regret (%) = floor + excess')
    axes[2].set_title('Two exact terms, SAME full legal Oracle')
    axes[0].set_yticks(y, PROBLEMS)
    axes[0].invert_yaxis()
    axes[0].legend()
    axes[2].legend()
    fig.suptitle('Full18 TEST: what changed relative to official frozen R77B?')
    fig.tight_layout(rect=(.01, .02, .99, .95))
    return fig


def learning_figure(plt, history, lock):
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    steps = [r['successful_updates'] for r in history]
    for key in ('native_ce', 'pair_quality_huber'):
        values = [r['optimization'].get(key) for r in history]
        axes[0].plot(steps, values, label=key)
    axes[0].set_title('Separate TRAIN heads; quality loss is NOT success')
    axes[0].legend()
    for ax, key, scale, title in ((axes[1], 'macro_top1', 100., 'Full18 VAL native Top1 (%)'),
                                  (axes[2], 'macro_actual_regret_pct', 1., 'Full18 VAL actual regret (%)')):
        ax.plot(steps, [r['val']['macro'][key] * scale for r in history])
        ax.set_title(title)
    for ax in axes:
        ax.axvline(lock['selected_update'], color='#c45a45', linestyle='--')
        ax.set_xlabel('Successful updates')
        ax.grid(alpha=.2)
    fig.tight_layout(rect=(0, .07, 1, 1))
    fig.text(.03, .01, 'TRAIN loss = ALL-row mean [1(Y in Top2) native BCE] + Huber(quality/.01, pair-relative-quality/.01), coefficient 1.', fontsize=9)
    return fig


def render_reports(output, predictions, tables, nss, config, lock, history):
    from matplotlib.backends.backend_pdf import PdfPages
    plt = plotting()
    output.mkdir(parents=True, exist_ok=True)
    for name, rows in tables.items():
        write_csv(output / (name + '.csv'), rows)
    write_csv(output / 'nss_four_datasets.csv', nss)
    main = output / 'R124_FULL18.pdf'
    distribution = output / 'selection_distributions.pdf'
    main_temp, dist_temp = main.with_suffix('.tmp.pdf'), distribution.with_suffix('.tmp.pdf')
    with PdfPages(main_temp) as report, PdfPages(dist_temp) as dist:
        compare = [r for r in tables['r77b_comparison'] if r['problem'] == 'ALL']
        rows = [[r['split'], r['n'], r['r77b_top1_pct'], r['r124_top1_pct'], r['top1_delta_pp'],
                 r['r77b_actual_regret_pct'], r['r124_actual_regret_pct']] for r in compare]
        fig = table_page(plt, 'Did the native reranker improve over official frozen R77B?',
            ['Split', 'n', 'R77B Top1 %', 'R124 Top1 %', 'Delta pp', 'R77B regret %', 'R124 regret %'], rows,
            f'Single VAL lock update {lock["selected_update"]}; checkpoint {lock["checkpoint_sha256"][:16]}.\n'
            'ALL is equal-task. Both NEW4500 populations and FP64 cost sources match exactly; old3895 metrics are never reused.\n'
            'Success is native correction minus damage and actual regret, NOT auxiliary quality loss. TEST was exposed during research.\n'
            'R77B is a complete external pretrained selector; the new original4layer128 encoder is scratch-initialized.\n'
            'MetaTRAIN10886/task is source-record unseen only; no exhaustive content/base-copy certificate.')
        report.savefig(fig)
        plt.close(fig)
        fig = table_page(plt, 'NSS / official R77B / SAME locked R124: four matched datasets',
            ['Dataset', 'n', 'NSS Top1 %', 'R77B Top1 %', 'R124 Top1 %', 'NSS mean cost', 'R77B cost', 'R124 cost', 'NSS status'],
            [[r['dataset'], r['n'], r['nss_top1_native_pct'], r['r77b_top1_native_pct'], r['r124_top1_native_pct'],
              r['nss_mean_cost'], r['r77b_mean_cost'], r['r124_mean_cost'], r['nss_status']] for r in nss],
            'NSS: historical R79 checkpoints, no new90k retraining. TEST4500 is the current release; LIB is original native LIB.\n'
            'All columns use saved native winner and full FP64 cost alignment. Missing references remain MISSING, never synthesized.\n'
            'Release: ' + Path(config['data_root']).name + '\n'
            'TSP pool: ' + ', '.join(predictions['TEST', 'TSP']['pool'].tolist()) + '\n'
            'CVRP pool: ' + ', '.join(predictions['TEST', 'CVRP']['pool'].tolist()))
        report.savefig(fig)
        plt.close(fig)
        for split in ('VAL', 'TEST'):
            rows = [r for r in tables['metrics'] if r['split'] == split]
            rows.sort(key=lambda r: -1 if r['problem'] == 'ALL' else list(PROBLEMS).index(r['problem']))
            fig = table_page(plt, split + ': ALL and all18 tasks',
                ['Task', 'n', 'Top1 %', 'mean cost', 'SBS cost', 'Oracle', 'vs SBS %', 'vs Oracle %', 'actual regret %'],
                [[r['problem'], r['n'], *[r[k] for k in METRICS]] for r in rows],
                'SBS fitted on frozen TRAIN81000. Oracle = each instance legal-pool minimum.\n'
                'vs_SBS/vs_Oracle are ratios of mean costs; actual regret is the mean per-instance relative regret.\n'
                'ALL averages tasks equally; raw-cost macro means are descriptive, not cross-task optimization evidence.')
            report.savefig(fig)
            plt.close(fig)
            rows = [r for r in tables['r77b_comparison'] if r['split'] == split]
            fig = table_page(plt, split + ': official R77B reference on the SAME new4500',
                ['Task', 'R77B Top1 %', 'R124 Top1 %', 'Delta pp', 'R77B mean cost', 'R124 mean cost', 'R77B regret %', 'R124 regret %'],
                [[r['problem'], r['r77b_top1_pct'], r['r124_top1_pct'], r['top1_delta_pp'], r['r77b_mean_cost'],
                  r['r124_mean_cost'], r['r77b_actual_regret_pct'], r['r124_actual_regret_pct']] for r in rows],
                'VAL f0 scores verified against independent shared R123 R77B archives; TEST f0 scores from the same post-lock full forward.\n'
                'Official frozen R77B checkpoint update34776, SHA ' + EXPERT_SHA + '. No old3895 accuracy/cost summaries.')
            report.savefig(fig)
            plt.close(fig)
        fig = diagnostics_figure(plt, tables)
        report.savefig(fig)
        plt.close(fig)
        rows = [r for r in tables['reranking'] if r['split'] == 'TEST']
        fig = table_page(plt, 'Full18 TEST: exact coverage and two-term regret accounting',
            ['Task', 'Top2 cover %', 'Pair Top1 %', 'Corrected', 'Damaged', 'Net pp', 'Recall floor %', 'Excess %', 'Sum regret %'],
            [[r['problem'], r['coverage_native_pct'], r['conditional_native_top1_pct'], r['corrected_count'],
              r['damaged_count'], r['net_top1_pp'], r['recall_floor_regret_pct'], r['reranking_excess_regret_pct'],
              r['actual_regret_pct']] for r in rows],
            'Top1 = coverage * conditional accuracy. ALL conditional is macro joint-correct / macro coverage, NOT mean task conditional.\n'
            'Regret = mean[(min pair cost - Oracle)/Oracle] + mean[(selected cost - min pair cost)/Oracle].\n'
            'Native ties may cause a native error with zero cost regret. Pair-outside native winner is never a fake classification target.')
        report.savefig(fig)
        plt.close(fig)
        rows = [r for r in tables['inference_cost'] if r['split'] != 'VAL']
        fig = table_page(plt, 'Actual complete stage0 + reranker inference cost',
            ['Dataset', 'Task', 'n', 'Full model sec', 'ms / instance', 'Cap', 'Full chain', 'Device', 'Geometry sec'],
            [[r['split'], r['problem'], r['n'], r['seconds'], r['milliseconds_per_instance'], r.get('batch_cap', 'per-task'),
              r['full_stage0_plus_reranker'], r.get('device', 'per-task'), r.get('feature_preparation_seconds')] for r in rows],
            'Every TEST/LIB model timing includes COMPLETE frozen R77B + new encoder + pair-native/quality heads; no readout-only estimate.\n'
            'Actual first-pass FP32 batched forward, CUDA synchronized when applicable; no I/O, transfer or solver time.\n'
            'VAL cached-f0 timing is NOT deployment cost. Shared bank precomputation is not free; LIB geometry is reported separately.', font=8)
        report.savefig(fig)
        plt.close(fig)
        fig = learning_figure(plt, history, lock)
        report.savefig(fig)
        plt.close(fig)
        for problem in PROBLEMS:
            fig = distribution_figure(plt, problem, predictions, config, lock)
            report.savefig(fig)
            dist.savefig(fig)
            plt.close(fig)
    main_temp.replace(main)
    dist_temp.replace(distribution)
    return dict(report_pdf=str(main), distributions_pdf=str(distribution))


def report(args):
    run, config, lock, final, identity = locked_inputs(args)
    with exclusive(run / 'finish/report.lock'):
        write_json(run / 'finish/identity.json', identity, immutable=True)
        predictions, inputs = load_selected(args, run, config, lock, final)
        references = validate_val_reference(args, config, predictions)
        for dataset, (problem, count) in LIBS.items():
            target = run / 'finish/lib_evaluation'
            result = read_json(target / (dataset + '_results.json'))
            path = target / 'predictions' / (dataset + '.npz')
            binding = result['protocol']
            if (not result['complete'] or result['predictions_sha256'] != file_hash(path)
                    or any(binding[k] != v for k, v in identity.items())):
                raise ValueError('Saved finish LIB inference belongs to another lock/source')
            data = archive(path)
            if str(data['identity_json'].item()) != json.dumps(binding, sort_keys=True):
                raise ValueError('LIB archive attribution differs from immutable protocol')
            validate_data(data, expected_rows=count, expected_pool=config['candidate_pools'][problem],
                          expected_ids=config['candidate_pool_ids'][problem])
            predictions[dataset, problem] = data
            inputs[dataset] = dict(path=str(path), sha256=file_hash(path), protocol=binding)
        tables = build_tables(predictions, config)
        nss = nss_comparisons(args, config, predictions, lock)
        history = read_json(run / 'history.json')
        output = run / 'finish/reports'
        paths = render_reports(output, predictions, tables, nss, config, lock, history)
        locked_inputs(args)
        if (any(file_hash(Path(value['path'])) != value['sha256'] for value in inputs.values())
                or any(file_hash(value['path']) != value['sha256'] for value in references.values())
                or any(file_hash(row['nss_prediction_path']) != row['nss_prediction_sha256']
                       for row in nss if row['nss_status'] == 'complete')):
            raise ValueError('Original prediction archive changed while reporting')
        test_all = next(r for r in tables['r77b_comparison'] if r['split'] == 'TEST' and r['problem'] == 'ALL')
        receipt = dict(identity=identity, inputs=inputs, official_r77b_val_references=references,
            official_r77b_test_reference='same post-lock TEST4500 stage1_scores; native cost source verified for all rows',
            nss_references=nss, full18=True, native_and_quality_heads_separate=True, quality_loss_alone_is_not_success=True,
            descriptive_test_changes=dict(native_top1_delta_pp=test_all['top1_delta_pp'],
                actual_regret_delta_pp=test_all['actual_regret_delta_pp'],
                native_top1_improved=test_all['top1_delta_pp'] > 0,
                actual_regret_improved=test_all['actual_regret_delta_pp'] < 0,
                native_and_cost_joint_improvement=test_all['top1_delta_pp'] > 0 and test_all['actual_regret_delta_pp'] < 0,
                statistical_significance_not_assessed=True, fresh_blind_generalization_claim=False),
            nss_complete=all(r['nss_status'] == 'complete' for r in nss), outputs=paths,
            output_sha256={name: file_hash(path) for name, path in paths.items()},
            original_predictions_modified=False, complete=all(r['nss_status'] == 'complete' for r in nss))
        write_json(output / 'finish_receipt.json', receipt)
        print(json.dumps(paths, indent=2), flush=True)
        return receipt


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('command', choices=('describe', 'lib', 'report', 'all'))
    result.add_argument('--root', type=Path, default=DEFAULT_ROOT,
                        help='Actual experiment execution root, e.g. R124_top2_reranking/launch_v2; NOT its model subdir')
    result.add_argument('--shared-bank-root', type=Path, default=DEFAULT_BANK)
    result.add_argument('--nss-root', type=Path, default=DEFAULT_NSS)
    result.add_argument('--device', default='cuda:0', help='Launcher-assigned device; no GPU auto-selection')
    result.add_argument('--lib-batch-size', type=int, default=1)
    result.add_argument('--allow-missing-nss', action='store_true', help='Report explicitly MISSING entries; receipt remains incomplete')
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    for name in ('root', 'shared_bank_root', 'nss_root'):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    if args.lib_batch_size <= 0:
        raise ValueError('LIB batch size must be positive')
    if args.command == 'describe':
        value = dict(run_root=str(args.root), checkpoint=str(args.root / 'model/best.pt'),
            output=str(args.root / 'model/finish'), required_gate='complete unchanged single R124 TEST lock',
            lib_factory='code.V4.r124_experiment.load_checkpoint', tasks=list(PROBLEMS),
            nss_datasets=['TSP TEST4500', 'CVRP TEST4500', 'TSPLIB', 'CVRPLIB'],
            reference='official frozen R77B; shared VAL4500, same post-lock TEST4500 full-forward scores',
            inference_cost='actual full frozen R77B + independent reranker on TEST and LIB',
            training=False, solver=False, source_or_original_prediction_edits=False, inference_started=False)
        print(json.dumps(value, indent=2))
        return value
    if args.command in ('lib', 'all'):
        lib_stage(args)
    if args.command in ('report', 'all'):
        return report(args)


if __name__ == '__main__':
    main()

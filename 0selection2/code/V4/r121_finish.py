"""One-lock R121 full TRAIN capture and saved-only mechanism supplement.

Only capture-train may forward, using main's existing release/binding/device.
finish calls the original saved-only finish and adds, never replaces, reports.
No training, solver, allocation, collector, new variant or checkpoint selection.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
import fcntl
import hashlib
import importlib
import json
from pathlib import Path
import time

import numpy as np

from code.unified_selector.registry import PROBLEMS


V4 = Path(__file__).resolve().parent
PREPARATION = V4 / 'runs/R121_structural_families/preparation'
DEFAULT_ROOT = PREPARATION / 'learner_v1'
SCHEMA = 'r121_learner_finish_v1'
ARM = 'principal'
ROWS = dict(TRAIN=81000, VAL=4500, TEST=4500)
HISTORICAL = (('R84A', V4 / 'runs/R84_new90k_attention/A_baseline', 'R84', 'A_baseline'),
              ('R94B', V4 / 'runs/R94_pretrained_node_fusion/B', 'R94', 'B'))
FAMILY_BASELINES = ('zero_potential', 'family_train_mean_potential')
TARGET_SHA = '406ded5a21d8b73c2a41b87dc2bc3cc14330663ae1aa4f59ca6fbfde83770fe4'
REPORT_FILES = ('R121_MECHANISM.pdf', 'natural_metrics.csv', 'natural_fixed36_error_budgets.csv',
    'family_diagnostics.csv', 'target_recovery.csv', 'target_recovery_summary.csv',
    'paired_changes.csv', 'historical_references.csv', 'protocol.json')


def learner():
    return importlib.import_module('code.V4.r121_experiment')


def helpers():
    return importlib.import_module('code.V4.r122_finish')


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def tracked(inputs, path):
    path = Path(path).resolve()
    digest = sha(path)
    if str(path) in inputs and inputs[str(path)] != digest:
        raise ValueError('Input changed during the stage: ' + str(path))
    inputs[str(path)] = digest
    return path


def artifact(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=sha(path), bytes=path.stat().st_size)


def write_json(path, value, *, immutable=False):
    path = Path(path)
    if immutable and path.exists():
        if read(path) != value:
            raise ValueError('Preserve the existing stage identity: ' + str(path))
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name('.' + path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def write_csv(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def save_npz(path, values):
    path = Path(path)
    if path.exists():
        raise FileExistsError('Never overwrite prediction artifacts: ' + str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name('.' + path.name + '.tmp')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, **values)
    temporary.replace(path)


def load_npz(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


@contextmanager
def exclusive(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def verify_inputs(inputs):
    for path, digest in inputs.items():
        if sha(path) != digest:
            raise ValueError('Captured input changed: ' + path)


def check_identity(record, identity):
    for key in ('schema', 'experiment', 'arm', 'checkpoint_sha256', 'selected_update',
                'state_hash', 'model_state_sha256'):
        if record.get(key) != identity[key]:
            raise ValueError('Capture belongs to a different locked model: ' + key)


def completed_receipt(path, identity):
    result = read(path)
    check_identity(result, identity)
    if result.get('complete') is not True:
        raise ValueError('Incomplete saved capture: ' + str(path))
    verify_inputs(result['input_files_sha256'])
    for value in result.get('artifacts', {}).values():
        if artifact(value['path']) != value:
            raise ValueError('Saved capture artifact changed: ' + value['path'])
    return result


def locked(args, inputs, *, test=False):
    base = learner()
    run, config, lock = base.locked_training(args)
    training = read(tracked(inputs, run / 'training_complete.json'))
    history = read(tracked(inputs, run / 'history.json'))
    for name in ('protocol.json', 'final_test_lock.json', 'best.pt'):
        tracked(inputs, run / name)
    if (training.get('complete') is not True or training.get('selected_best') != lock
            or training.get('state_hash') != lock['state_hash'] or config.get('arm') != ARM
            or Path(lock['checkpoint']).resolve() != (run / 'best.pt').resolve()
            or lock['inference_rule'] != config['inference_rule']
            or lock['inference_rule'] != base.INFERENCE_RULE
            or [r['successful_updates'] for r in history] != list(range(base.EVAL_EVERY, base.MAX_UPDATES + 1, base.EVAL_EVERY))):
        raise PermissionError('Require the complete unchanged unique natural-VAL lock')
    chosen = min(history, key=lambda r: base.choose_key(r['val'], r['successful_updates']))
    if chosen['successful_updates'] != lock['selected_update'] or chosen['val']['macro'] != lock['val']:
        raise PermissionError('The existing unique lock differs from its original VAL-only rule')
    for name, digest in config['source_sha256'].items():
        if sha(tracked(inputs, V4 / name)) != digest:
            raise PermissionError('Frozen learner source changed: ' + name)
    for name in ('r121_finish.py', 'r122_finish.py', 'r116_mechanism_finish.py', 'r80_reporting.py', 'r84_reporting.py'):
        tracked(inputs, V4 / name)
    if test:
        from . import r80_reporting as common
        actual = common.locked_inputs(args.root, ARM, experiment='R121', arms={ARM: base.MAX_UPDATES})
        if actual[:3] != (run, config, lock):
            raise ValueError('Base TEST gate returned another checkpoint')
        for name in ('test_results.json', 'final_metrics.json'):
            tracked(inputs, run / name)
        return run, config, lock, history, actual[3], actual[4]
    return run, config, lock, history


def identity_for(lock, model_hash):
    return dict(schema=SCHEMA, experiment='R121', arm=ARM,
        checkpoint_sha256=lock['checkpoint_sha256'], selected_update=lock['selected_update'],
        state_hash=lock['state_hash'], model_state_sha256=model_hash)


def load_natural(run, config, lock, result, model_hash, split, inputs):
    helper = helpers()
    path = helper.common.resolve_record_path(result['prediction_archive'], run)
    identity = read(tracked(inputs, path.parent / 'identity.json'))
    expected = dict(experiment=config['experiment'], arm=config['arm'], split=split.lower(),
        successful_updates=lock['selected_update'], state_hash=lock['state_hash'], model_state_sha256=model_hash)
    if (identity not in (result['identity'], result.get('archive_capture_identity'))
            or any(identity.get(k) != v for k, v in expected.items())
            or identity.get('checkpoint_sha256') not in (None, lock['checkpoint_sha256'])
            or split in ('TRAIN', 'TEST') and config['experiment'] == 'R121'
                and identity.get('checkpoint_sha256') != lock['checkpoint_sha256']):
        raise ValueError('Natural capture is not from the selected unchanged model: ' + split)
    with np.load(tracked(inputs, path), allow_pickle=False) as archive:
        data = {p: helper.read_prediction(archive, p + '__', config, p, n=ROWS[split], path=path) for p in PROBLEMS}
    for p, item in data.items():
        np.testing.assert_array_equal(item['indices'], np.arange(ROWS[split]))
        with np.load(tracked(inputs, Path(config['data_root']) / (p + split.lower()) / 'input_metadata.npz'), allow_pickle=False) as metadata:
            np.testing.assert_array_equal(item['release_source_indices'], metadata['source_index'])
    return data


def eligible_indices(config, train, inputs):
    base = learner()
    record = config['natural_ce_isolation']
    path = tracked(inputs, record['path'])
    isolation = read(path)
    base.structural.verify_seal(isolation, 'isolation_sha256')
    if sha(path) != record['sha256'] or isolation['isolation_sha256'] != record['isolation_sha256']:
        raise ValueError('Existing gradient isolation changed')
    indices = {}
    for p, data in train.items():
        item = isolation['per_problem'][p]
        excluded = np.asarray(item['excluded_local_rows'], dtype=np.int64)
        if (len(np.unique(excluded)) != len(excluded) or (excluded < 0).any()
                or (excluded >= len(data['winner'])).any()):
            raise ValueError('Invalid saved excluded TRAIN rows')
        keep = np.setdiff1d(np.arange(len(data['winner'])), excluded)
        if (len(keep) != config['natural_ce_eligible_rows_per_problem'][p]
                or base.index_hash(keep) != item['eligible_local_rows_sha256']
                or base.index_hash(data['release_source_indices']) != item['source_index_array_sha256']):
            raise ValueError('Eligible-gradient population differs from actual full TRAIN')
        indices[p] = keep
    return indices


def subset(data, indices):
    result = dict(data)
    for key in ('scores', 'costs', 'winner', 'pred', 'indices', 'source_indices', 'release_source_indices'):
        if key in data:
            result[key] = data[key][indices]
    return result


def natural_metrics(predictions, config, lock, eligible):
    helper, base = helpers(), learner()
    rows, summaries = [], {}
    populations = dict(predictions, TRAIN_ELIGIBLE={p: subset(predictions['TRAIN'][p], eligible[p]) for p in PROBLEMS})
    for split, tasks in populations.items():
        points, measured = [], {}
        for p in PROBLEMS:
            row, measured[p] = helper.metric_row(tasks[p], config['baselines'][p], model='R121/principal',
                split=split, problem=p, checkpoint=lock['checkpoint_sha256'], version=Path(config['data_root']).name)
            row['gradient_population'] = 'saved eligible-gradient subset' if split == 'TRAIN_ELIGIBLE' else 'original frozen full split'
            points.append(row)
        rows.extend([helper.macro_rows(points), *points])
        summaries[split] = base.natural.summarize(measured)
    return rows, summaries


def huber(residual):
    absolute = np.abs(residual)
    return np.where(absolute <= 1., .5 * residual ** 2, absolute - .5)


def family_statistics(data):
    """Pure saved-array diagnostics; mean potential fits unique TRAIN endpoints only."""
    costs, prediction = data['costs'], data['advantage_units']
    n, k = costs.shape
    if (costs.dtype != np.float64 or prediction.shape != costs.shape or k < 2
            or not np.isfinite(costs).all() or (costs <= 0).any() or not np.isfinite(prediction).all()
            or data['winner'].dtype != np.int64
            or not np.array_equal(costs[np.arange(n), data['winner']], costs.min(1))
            or not np.array_equal(data['indices'], np.arange(n))
            or len(np.unique(data['endpoint_ids'])) != n):
        raise ValueError('Invalid original family endpoint prediction population')
    if 'logits' in data and (data['logits'].shape != costs.shape or not np.isfinite(data['logits']).all()):
        raise ValueError('Saved family native logits must also be complete and finite')
    target = np.log(costs)
    target = (target - target.mean(1, keepdims=True)) / .01
    partitions = data['partition']
    train = np.flatnonzero(partitions == 'family_train')
    val = np.flatnonzero(partitions == 'family_val')
    if not len(train) or not len(val) or len(train) + len(val) != n:
        raise ValueError('Both unique family TRAIN/VAL endpoint populations are required')
    if set(data['group_id'][train]) & set(data['group_id'][val]):
        raise ValueError('A family geometry group crosses partitions')
    mean = target[train].mean(0)
    models = dict(locked_model=prediction.astype(np.float64), zero_potential=np.zeros_like(target),
                  family_train_mean_potential=np.broadcast_to(mean, target.shape))
    rows = []
    for partition, indices, name in (('family_train', train, 'edges_train'), ('family_val', val, 'edges_val')):
        edges = data[name]
        if (edges.dtype.kind not in 'iu' or edges.ndim != 2 or edges.shape[1] != 2 or not len(edges)
                or (edges < 0).any() or (edges >= n).any() or not np.isin(edges, indices).all()
                or not np.all(data['cell'][edges[:, 0]] == 'original')
                or np.any(data['cell'][edges[:, 1]] == 'original')
                or not np.array_equal(data['family_id'][edges[:, 0]], data['family_id'][edges[:, 1]])
                or set(edges.reshape(-1)) != set(indices)):
            raise ValueError('Actual family edges must retain within-family original/changed pairs')
        for model, values in models.items():
            endpoint = huber(values - target).mean(1)
            response = huber((values[edges[:, 1]] - values[edges[:, 0]])
                            - (target[edges[:, 1]] - target[edges[:, 0]])).mean(1)
            rows.append(dict(partition=partition, model=model, n=len(indices), pairs=len(edges),
                endpoint_huber_unique=float(endpoint[indices].mean()),
                endpoint_huber_edge_slots=float(endpoint[edges.reshape(-1)].mean()), response_huber=float(response.mean()),
                baseline_fit_partition='unique family_train endpoints only', baseline_fit_endpoints=len(train),
                train_mean_potential_units=json.dumps(mean.tolist()) if model == 'family_train_mean_potential' else '',
                advantage_unit=.01, huber_delta=1., target_precision='center(log(native FP64 cost))/.01; saved predictions promoted to FP64',
                current_labels_in_forward=False, used_for_checkpoint=False))
    return rows


def family_summary(per_problem):
    rows = [dict(problem=p, **row) for p, data in per_problem.items() for row in family_statistics(data)]
    if set(per_problem) != set(PROBLEMS):
        raise ValueError('Family diagnostics require all18 tasks, not an available subset')
    for partition in ('family_train', 'family_val'):
        for model in ('locked_model', *FAMILY_BASELINES):
            group = [r for r in rows if r['partition'] == partition and r['model'] == model and r['problem'] != 'ALL']
            rows.append(dict(problem='ALL', partition=partition, model=model,
                n=sum(r['n'] for r in group), pairs=sum(r['pairs'] for r in group),
                **{key: float(np.mean([r[key] for r in group])) for key in
                   ('endpoint_huber_unique', 'endpoint_huber_edge_slots', 'response_huber')},
                aggregation='equal task weight; Huber improvement alone is not selection success'))
    return rows


def sync(device):
    if device.type == 'cuda':
        import torch
        torch.cuda.synchronize(device)


def family_forward(model, cache, cap, device):
    """The ONLY new forward path; query allowlist excludes labels and metadata."""
    import torch
    base = learner()
    values = dict(logits=[], advantage_units=[])
    seconds = 0.
    for start in range(0, cache.size, cap):
        batch = cache.batch(np.arange(start, min(start + cap, cache.size)), device)
        query = base.model_api.query_inputs(batch)
        sync(device)
        before = time.perf_counter()
        output = model(query)
        sync(device)
        seconds += time.perf_counter() - before
        for key in values:
            values[key].append(output[key].detach().float().cpu().numpy())
    return {**{key: np.concatenate(parts) for key, parts in values.items()},
            'forward_seconds': np.asarray(seconds), 'forward_batch_cap': np.asarray(cap), 'device': np.asarray(str(device))}


def family_capture(args, model, config, identity, inputs, output):
    base = learner()
    grouped, receipt, manifest = base.family_rows(args.family_labels)
    if (receipt != config['family_labels']
            or manifest['plan_sha256'] != read(args.family_input_manifest)['plan_sha256']):
        raise ValueError('Family labels differ from the immutable training population')
    tracked(inputs, args.family_labels)
    tracked(inputs, args.family_labels.with_suffix('.receipt.json'))
    tracked(inputs, receipt['family_input_manifest'])
    write_json(output / 'family_capture_protocol.json', dict(identity, family_export=receipt,
        candidate_pools=config['candidate_pools'], candidate_pool_ids=config['candidate_pool_ids']), immutable=True)
    tracked(inputs, output / 'family_capture_protocol.json')
    data, artifacts = {}, {}
    device = next(model.parameters()).device
    for p in PROBLEMS:
        target = output / 'family_predictions' / (p + '.npz')
        metadata = tracked(inputs, args.family_cache / p / 'complete.json')
        source_costs = np.asarray([row['targets']['native_cost_fp64'] for row in grouped[p]], dtype=np.float64)
        source_ids = np.asarray([row['endpoint']['endpoint_id'] for row in grouped[p]])
        if target.exists():
            current = load_npz(target)
            if json.loads(str(current['identity_json'].item())) != identity:
                raise ValueError('Preserve family predictions from another checkpoint')
        else:
            cache = base.FamilyCache(args.family_cache, p, grouped[p], receipt['sha256'])
            try:
                if cache.pool_ids != config['candidate_pool_ids'][p]:
                    raise ValueError('Family/native candidate pools differ')
                cap = min(args.eval_batch_size, base.micro_caps(args.gpu_type, args.micro_cap)[p])
                current = family_forward(model, cache, cap, device)
                current.update(costs=cache.arrays['costs'].copy(), winner=cache.arrays['winner'].copy(),
                    pool=np.asarray(cache.pool), pool_ids=np.asarray(cache.pool_ids, dtype=np.int64), indices=np.arange(cache.size),
                    endpoint_ids=source_ids, edges_train=cache.edges['family_train'], edges_val=cache.edges['family_val'],
                    identity_json=np.asarray(json.dumps(identity, sort_keys=True)))
                for key, field in (('partition', 'family_partition'), ('family_id', 'family_id'), ('cell', 'cell'), ('group_id', 'group_id')):
                    current[key] = np.asarray([row['endpoint'][field] for row in grouped[p]])
                family_statistics(current)
                save_npz(target, current)
            finally:
                cache.close()
        np.testing.assert_array_equal(current['costs'], source_costs)
        np.testing.assert_array_equal(current['endpoint_ids'], source_ids)
        np.testing.assert_array_equal(current['pool'], config['candidate_pools'][p])
        np.testing.assert_array_equal(current['pool_ids'], config['candidate_pool_ids'][p])
        for partition, key in (('family_train', 'edges_train'), ('family_val', 'edges_val')):
            np.testing.assert_array_equal(current[key], base.family_edges(grouped[p], partition))
        tracked(inputs, target)
        data[p], artifacts[p + '.npz'] = current, artifact(target)
    rows = family_summary(data)
    path = output / 'captured_family_diagnostics.csv'
    write_csv(path, rows)
    artifacts[path.name] = artifact(path)
    result = dict(identity, complete=True, phase='locked_family_capture', all18=True, model_inference=True,
        GPU_execution=device.type == 'cuda', solver_execution=False, training=False, checkpoint_reselection=False,
        family_baselines=list(FAMILY_BASELINES), input_files_sha256=dict(inputs), artifacts=artifacts,
        forward_seconds=sum(float(d['forward_seconds']) for d in data.values()),
        endpoints=sum(len(d['costs']) for d in data.values()), diagnostic_summary=[r for r in rows if r['problem'] == 'ALL'])
    write_json(output / 'family_capture_receipt.json', result)
    return result


def capture_train(args):
    stage_started = time.perf_counter()
    base, inputs = learner(), {}
    run, config, lock, _ = locked(args, inputs)
    authorization = base.authorized(args)
    if (str(args.data_root.resolve()) != config['data_root'] or str(args.prepared_root.resolve()) != config['prepared_root']
            or base.micro_caps(args.gpu_type, args.micro_cap) != config['microbatch_caps']
            or args.eval_batch_size != config['eval_batch_size']):
        raise ValueError('Capture arguments must retain the frozen population and evaluation capacities')
    tracked(inputs, args.release)
    tracked(inputs, args.binding)
    output = run / 'learner_finish'
    with exclusive(output / 'capture.lock'):
        existing = output / 'capture_train_receipt.json'
        if existing.exists():
            saved = read(existing)
            identity = identity_for(lock, saved['model_state_sha256'])
            result = completed_receipt(existing, identity)
            completed_receipt(output / 'family_capture_receipt.json', identity)
            return result
        import torch
        from .r77_data import parameter_hash
        _, manifest, _, baselines = base.natural.prepared_config(args)
        if baselines != config['baselines']:
            raise ValueError('Never refit TRAIN81000 SBS or replace input preparation')
        model, state = base.load_checkpoint(run / 'best.pt', args.device)
        if state['args'] != config or state['successful_updates'] != lock['selected_update']:
            raise ValueError('Full TRAIN/family diagnostics require the actual locked best.pt')
        model.eval().requires_grad_(False)
        model_hash = parameter_hash(model.state_dict())
        identity = identity_for(lock, model_hash)
        write_json(output / 'capture_protocol.json', dict(identity, full_train_rows=81000,
            eligible_rows=config['natural_ce_eligible_rows_per_problem'], resource_authorization=authorization), immutable=True)
        device = next(model.parameters()).device
        sync(device)
        started = time.perf_counter()
        reused = (run / 'evaluation/train/selected_best/rank000.npz').exists()
        with torch.inference_mode():
            result = base.evaluate(model, args, manifest, baselines, 'train', 'selected_best',
                lock['selected_update'], lock['state_hash'], lock['checkpoint_sha256'])
            sync(device)
            capture_seconds = time.perf_counter() - started
            train = load_natural(run, config, lock, result, model_hash, 'TRAIN', inputs)
            eligible = eligible_indices(config, train, inputs)
            rows, summaries = natural_metrics(dict(TRAIN=train), config, lock, eligible)
            family = family_capture(args, model, config, identity, inputs, output)
        if parameter_hash(model.state_dict()) != model_hash:
            raise ValueError('Inference changed locked weights or buffers')
        locked(args, inputs)
        verify_inputs(inputs)
        metrics_path = output / 'captured_train_metrics.csv'
        write_csv(metrics_path, rows)
        artifacts = {metrics_path.name: artifact(metrics_path), 'train_predictions': artifact(result['prediction_archive']),
            'train_metrics': artifact(run / 'evaluation/train/selected_best/metrics.json'),
            'train_identity': artifact(run / 'evaluation/train/selected_best/identity.json'),
            'family_capture_receipt': artifact(output / 'family_capture_receipt.json')}
        value = dict(identity, complete=True, phase='locked_full_train_capture', all18=True, train_complete=True,
            train_rows_per_problem=81000, train_eligible_rows_per_problem=config['natural_ce_eligible_rows_per_problem'],
            TRAIN=result['macro'], TRAIN_ELIGIBLE=summaries['TRAIN_ELIGIBLE']['macro'], result=result,
            natural_evaluation_seconds=result['seconds'], stage_call_elapsed_seconds=capture_seconds,
            capture_elapsed_seconds=time.perf_counter() - stage_started,
            timing_scope='original complete TRAIN evaluation elapsed, including cache/data/report I/O; not forward-only',
            reused_existing_train_capture=reused, family_forward_seconds=family['forward_seconds'],
            artifacts=artifacts, input_files_sha256=dict(inputs), source_sha256=config['source_sha256'],
            model_inference=not reused or bool(family['endpoints']), GPU_execution=device.type == 'cuda',
            solver_execution=False, training=False, checkpoint_reselection=False)
        write_json(existing, value)
        base.status(args, 'locked_train_and_family_capture_complete', base.MAX_UPDATES, selected_update=lock['selected_update'])
        return value


def load_family(output, config, identity, receipt, inputs):
    tasks = {}
    for p in PROBLEMS:
        path = output / 'family_predictions' / (p + '.npz')
        saved = receipt['artifacts'][p + '.npz']
        if artifact(path) != saved:
            raise ValueError('Family prediction artifact differs from the actual capture')
        current = load_npz(tracked(inputs, path))
        if (json.loads(str(current['identity_json'].item())) != identity
                or current['pool'].tolist() != config['candidate_pools'][p]
                or current['pool_ids'].tolist() != config['candidate_pool_ids'][p]):
            raise ValueError('Family diagnostic belongs to another model/pool')
        expected = config['family_population']['per_problem'][p]
        for part in ('family_train', 'family_val'):
            if int((current['partition'] == part).sum()) != expected[part]['endpoints']:
                raise ValueError('Family diagnostic omitted actual endpoints')
        tasks[p] = current
    return tasks


def fixed_targets(config, inputs):
    helper = helpers()
    path = tracked(inputs, helper.TARGET_LIST)
    record = read(path)
    pairs = [(r['problem'], r['method']) for r in record['targets']]
    if (sha(path) != TARGET_SHA or len(pairs) != 36 or len(set(pairs)) != 36
            or record['identity']['baseline_checkpoint_sha256'] != helper.BASELINE_SHA
            or any(p not in PROBLEMS or name not in config['candidate_pools'][p] for p, name in pairs)):
        raise ValueError('Use exactly the original frozen R84A VAL fixed36 list')
    return pairs, artifact(path), record['identity']


def mechanism_tables(config, predictions, inputs):
    helper = helpers()
    targets, target_file, target_identity = fixed_targets(config, inputs)
    details, summary, paired, references = [], [], [], []
    for label, path, experiment, arm in HISTORICAL:
        old = helper.historical_model(path, experiment, arm, config, inputs)
        if label == 'R84A' and (old['lock']['checkpoint_sha256'] != helper.BASELINE_SHA
                or old['lock']['state_hash'] != target_identity['baseline_state_hash']):
            raise ValueError('The fixed36 historical R84A reference changed')
        references.append(dict(model=label, status='complete', run=str(path),
            checkpoint_sha256=old['lock']['checkpoint_sha256'], same90k=True,
            original_raw_query_backbone=label == 'R84A', full_model_matched=False,
            training_protocol_matched=False, newly_matched_control=False,
            comparison_splits='VAL4500|TEST4500', historical_full_train_comparison='NOT ASSESSED; no sampled TRAIN substituted',
            boundary='R84A original raw-query backbone; R121 adds a potential head and different data/loss/exposures'
                if label == 'R84A' else 'R94B historical pretrained node fusion; extra donor compute; not architecture matched'))
        for split in ('VAL', 'TEST'):
            previous = helper.natural_split(old, split, inputs)
            comparison = 'R121/principal vs ' + label
            detail, groups = helper.recovery_tables(previous, predictions[split], targets, comparison, split)
            details.extend(detail)
            summary.extend(groups)
            points = [helper.paired_row(previous[p], predictions[split][p],
                comparison=comparison, split=split, problem=p) for p in PROBLEMS]
            paired.extend([*points, helper.macro_rows(points,
                ('top1_delta_pp', 'mean_cost_delta', 'actual_regret_delta_pp'),
                ('corrected', 'harmed', 'net_correct', 'changed', 'both_wrong_cheaper', 'both_wrong_dearer'))])
    budgets = helper.error_budgets(predictions, targets)
    return dict(target_recovery=details, target_recovery_summary=summary, paired_changes=paired,
                historical_references=references, natural_fixed36_error_budgets=budgets), target_file


def plots():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    for path in (Path('/usr/share/fonts/google-noto-cjk/NotoSansCJK-Regular.ttc'),
                 Path('/public/home/shiys/.cache/selector_fonts/20261026/NotoSansCJK-Regular.ttc')):
        if path.exists():
            font_manager.fontManager.addfont(str(path))
            matplotlib.rcParams['font.family'] = font_manager.FontProperties(fname=path).get_name()
            break
    matplotlib.rcParams.update({'pdf.fonttype': 42, 'axes.unicode_minus': False, 'font.size': 9})
    return plt


def page(plt, title, columns, rows, notes, lock, *, font=8):
    fig = plt.figure(figsize=(16, 10.5))
    fig.text(.04, .95, title, fontsize=16, weight='bold')
    ax = fig.add_axes([.035, .18, .93, .71])
    ax.axis('off')
    def display(value):
        if value is None:
            return 'undefined'
        return f'{value:.5f}' if isinstance(value, (float, np.floating)) else str(value)
    height = min(1., .055 * (len(rows) + 1))
    table = ax.table(cellText=[[display(v) for v in row] for row in rows], colLabels=columns,
        cellLoc='center', bbox=[0, 1. - height, 1, height])
    table.auto_set_font_size(False)
    table.set_fontsize(font)
    for (row, _), cell in table.get_celld().items():
        cell.set_linewidth(.25)
        if row == 0:
            cell.set_facecolor('#dce9e5')
            cell.set_text_props(weight='bold')
        elif row % 2 == 0:
            cell.set_facecolor('#f3f5f2')
    fig.text(.04, .13, notes, fontsize=9, va='top', wrap=True)
    fig.text(.04, .025, f'R121 principal | unique natural-VAL lock update {lock["selected_update"]} | SHA {lock["checkpoint_sha256"]}', fontsize=8)
    return fig


def render(output, tables, config, lock):
    from matplotlib.backends.backend_pdf import PdfPages
    plt = plots()
    output.mkdir(parents=True, exist_ok=True)
    for name, rows in tables.items():
        write_csv(output / (name + '.csv'), rows)
    target = output / 'R121_MECHANISM.pdf'
    temporary = target.with_suffix('.tmp.pdf')
    with PdfPages(temporary) as pdf:
        rows = [r for r in tables['natural_metrics'] if r['problem'] == 'ALL']
        fig = page(plt, 'R121: same locked weights, complete natural populations',
            ['Population', 'n', 'Native Top1 %', 'mean cost', 'TRAIN SBS', 'Oracle', 'vs SBS %', 'vs Oracle %', 'actual regret %'],
            [[r['split'], r['n'], *[r[k] for k in helpers().CORE_METRICS]] for r in rows],
            'Full TRAIN is original81000/task; TRAIN_ELIGIBLE is the existing gradient-eligible subset, NOT another release.\n'
            'ALL is task-equal. Raw macro cost is descriptive; ratio-of-mean costs is not mean instance actual regret.\n'
            'Saved native int64 ind / FP64 costs; exact native ties retained. All legal pools and TRAIN81000 SBS are unchanged.\n'
            'This supplement does not replace the original main PDFs. TEST was exposed during R&D; no blind/matched-control claim.', lock)
        pdf.savefig(fig)
        plt.close(fig)
        for split in ('TRAIN', 'TRAIN_ELIGIBLE', 'VAL', 'TEST'):
            rows = [r for r in tables['natural_metrics'] if r['split'] == split]
            fig = page(plt, split + ': all18 tasks at the SAME checkpoint',
                ['Task', 'n', 'Native Top1 %', 'mean cost', 'SBS cost', 'Oracle', 'vs SBS %', 'vs Oracle %', 'actual regret %'],
                [[r['problem'], r['n'], *[r[k] for k in helpers().CORE_METRICS]] for r in rows],
                'TRAIN SBS fitted on original81000/task, never refitted on eligible/VAL/TEST. Oracle is the actual legal-pool minimum.\n'
                'Normalization and original release membership are retained; known heldout source/geometry isolation is not exhaustive dedup.\n'
                'Data: ' + Path(config['data_root']).name, lock)
            pdf.savefig(fig)
            plt.close(fig)
        rows = [r for r in tables['family_diagnostics'] if r['problem'] == 'ALL']
        fig = page(plt, 'Family diagnostic: actual saved potentials versus TRAIN-only baselines',
            ['Partition', 'Prediction', 'Endpoints', 'Pairs', 'Endpoint unique Huber', 'Endpoint slot Huber', 'Response Huber'],
            [[r['partition'], r['model'], r['n'], r['pairs'], r['endpoint_huber_unique'],
              r['endpoint_huber_edge_slots'], r['response_huber']] for r in rows],
            'Target = center(log(native FP64 costs))/.01, Huber delta=1; no fabricated actual MAE. Predictions captured from locked weights.\n'
            'Mean potential fitted only on unique family_train endpoints per task/method. No family_val fitting or parameter updates.\n'
            'Unique endpoints and repeated original edge slots have different weights. Slot Huber uses the original training-loss weighting.\n'
            'Constant mean and zero baselines both have zero response. Lower auxiliary Huber alone is NOT selector success.', lock)
        pdf.savefig(fig)
        plt.close(fig)
        for partition in ('family_train', 'family_val'):
            data = {(r['problem'], r['model']): r for r in tables['family_diagnostics'] if r['partition'] == partition}
            rows = []
            for p in ('ALL', *PROBLEMS):
                models = [data[p, m] for m in ('locked_model', *FAMILY_BASELINES)]
                rows.append([p, models[0]['n'], *[r['endpoint_huber_edge_slots'] for r in models],
                             *[r['response_huber'] for r in models]])
            fig = page(plt, partition + ': complete all18 native family endpoints',
                ['Task', 'n', 'Model endpoint', 'Zero endpoint', 'Mean endpoint', 'Model response', 'Zero response', 'Mean response'],
                rows, 'Endpoint = edge-slot-weighted Huber; unique endpoint Huber is separately recorded in family_diagnostics.csv.\n'
                'Same potential predicts both endpoints; response = changed - original. Costs, labels and family metadata never enter forward.\n'
                'family_val is a TRAIN-derived known-source gradient holdout, not an independent benchmark or an exhaustive content-copy holdout.', lock)
            pdf.savefig(fig)
            plt.close(fig)
        budgets = [r for r in tables['natural_fixed36_error_budgets'] if r['problem'] == 'ALL']
        fig = page(plt, 'Fixed36: actual native-winner error partitions, including complete TRAIN',
            ['Split', 'Group', 'Winners', 'Missed', 'Recall %', 'Missed / full %', 'Regret contribution %'],
            [[r['split'], r['group'], r['winner_count'], r['missed_count'], r['conditional_native_top1_pct'],
              r['hard_error_frequency'] * 100., r['hard_regret_contribution'] * 100.] for r in budgets],
            'Fixed36 is the frozen historical R84A VAL target list, not a newly chosen group. Zero selection is NOT success.\n'
            'Historical paired rescue/cost comparisons below use complete aligned VAL/TEST4500 only; no sampled TRAIN is substituted.\n'
            'Winner-group cost contributions use the full problem denominator; ALL contributions average all18 tasks.', lock)
        pdf.savefig(fig)
        plt.close(fig)
        for label, _, _, _ in HISTORICAL:
            rows = [r for r in tables['target_recovery_summary'] if r['problem'] == 'ALL' and r['comparison'].endswith(label)]
            fig = page(plt, 'True recovery / damage / new false picks: R121 vs ' + label,
                ['Split', 'Group', 'Winners', 'Old recall %', 'New recall %', 'Rescued', 'Lost', 'Net TP', 'New false', 'dCost contrib', 'dRegret pp'],
                [[r['split'], r['solver'], r['winner_count'], r['baseline_recall_pct'], r['current_recall_pct'],
                  r['corrected'], r['harmed'], r['net_correct'], r['new_false_pick_count'],
                  r['mean_cost_delta_contribution'], r['actual_regret_delta_contribution_pp']] for r in rows],
                'Delta = current - historical baseline. New false pick = incorrect newly selected group member, including outside-group winners.\n'
                'Cost/regret columns use the native-winner group; false picks use the predicted-method group. These are NOT the same subset.\n'
                'Per-method fixed36 actual recall and effects are in target_recovery.csv; zero-group reduction is descriptive, never benefit.\n'
                'Historical architecture/objective/budget/exposure are not a newly matched control. No significance or blind confirmation claim.', lock, font=7)
            pdf.savefig(fig)
            plt.close(fig)
    temporary.replace(target)


def finish(args):
    if args.device != 'cpu':
        raise PermissionError('finish is saved-only CPU; forward diagnostics belong in capture-train')
    base, inputs = learner(), {}
    run, config, lock, history, test, val = locked(args, inputs, test=True)
    model_hash = test['identity']['model_state_sha256']
    identity = identity_for(lock, model_hash)
    output = run / 'learner_finish'
    with exclusive(output / 'finish.lock'):
        train_receipt_path = output / 'capture_train_receipt.json'
        family_receipt_path = output / 'family_capture_receipt.json'
        train_receipt = completed_receipt(train_receipt_path, identity)
        family_receipt = completed_receipt(family_receipt_path, identity)
        tracked(inputs, train_receipt_path)
        tracked(inputs, family_receipt_path)
        if (train_receipt.get('train_rows_per_problem') != 81000 or train_receipt.get('all18') is not True
                or train_receipt['train_eligible_rows_per_problem'] != config['natural_ce_eligible_rows_per_problem']):
            raise ValueError('Require the actual complete TRAIN and unchanged eligible-gradient population')
        base_delivery = base.finish(args)
        if base_delivery.get('complete') is not True:
            raise ValueError('Original saved-only R121 finish is incomplete')
        delivery_path = tracked(inputs, run / 'delivery_manifest.json')
        results = dict(TRAIN=train_receipt['result'], VAL=val, TEST=test)
        predictions = {split: load_natural(run, config, lock, result, model_hash, split, inputs)
                       for split, result in results.items()}
        eligible = eligible_indices(config, predictions['TRAIN'], inputs)
        natural_rows, summaries = natural_metrics(predictions, config, lock, eligible)
        for split in ('TRAIN', 'VAL', 'TEST'):
            if summaries[split]['macro'] != results[split]['macro']:
                raise ValueError('Recomputed native macro differs from same-checkpoint saved result: ' + split)
        family_data = load_family(output, config, identity, family_receipt, inputs)
        tables, target_file = mechanism_tables(config, predictions, inputs)
        tables.update(natural_metrics=natural_rows, family_diagnostics=family_summary(family_data))
        reports = output / 'reports'
        render(reports, tables, config, lock)
        protocol = dict(identity, complete=True, checkpoint_lock=lock,
            data_root=config['data_root'], candidate_pools=config['candidate_pools'], candidate_pool_ids=config['candidate_pool_ids'],
            full_train_rows_per_problem=81000, train_eligible_rows_per_problem=config['natural_ce_eligible_rows_per_problem'],
            natural_ce_isolation=config['natural_ce_isolation'], family_population=config['family_population'],
            family_baselines=list(FAMILY_BASELINES), family_mean_fit='unique family_train endpoints only, no family_val fit',
            fixed36_target_file=target_file, model_inference=False, checkpoint_reselection=False,
            selection_counts_are_not_success=True, actual_MAE_reported=False,
            baseline_comparison_splits=['VAL4500', 'TEST4500'], historical_training_protocol_matched=False,
            known_source_record_only_not_exhaustive_copy_novelty=True, independent_new_benchmark=False)
        write_json(reports / 'protocol.json', protocol)
        artifacts = {name: artifact(reports / name) for name in REPORT_FILES}
        locked(args, inputs, test=True)
        verify_inputs(inputs)
        for record in (train_receipt, family_receipt):
            verify_inputs(record['input_files_sha256'])
        all_inputs = dict(train_receipt['input_files_sha256'], **family_receipt['input_files_sha256'])
        for path, digest in inputs.items():
            if path in all_inputs and all_inputs[path] != digest:
                raise ValueError('Capture and saved-only report source/input identities differ')
            all_inputs[path] = digest
        value = dict(identity, complete=True, phase='complete_post_TEST_learner_finish', all18=True,
            train_complete=True, train_rows_per_problem=81000,
            train_eligible_rows_per_problem=config['natural_ce_eligible_rows_per_problem'],
            TRAIN=summaries['TRAIN']['macro'], TRAIN_ELIGIBLE=summaries['TRAIN_ELIGIBLE']['macro'],
            VAL=val['macro'], TEST=test['macro'], family_diagnostics_complete=True, family_baselines=list(FAMILY_BASELINES),
            fixed36_count=36, fixed36_target_file=target_file, historical_references=tables['historical_references'],
            base_delivery_manifest=artifact(delivery_path), capture_train_receipt=artifact(train_receipt_path),
            family_capture_receipt=artifact(family_receipt_path), artifacts=artifacts,
            input_files_sha256=all_inputs, source_sha256={Path(path).name: digest for path, digest in all_inputs.items()
                if Path(path).parent == V4 and Path(path).suffix == '.py'},
            model_inference=False, GPU_execution=False, solver_execution=False, training=False,
            checkpoint_reselection=False, research_goal_complete=False,
            resource_release='main-owned; saved-only report does not assert resource release',
            selection_counts_are_not_success=True, auxiliary_huber_alone_is_not_success=True,
            natural_evaluation_seconds=train_receipt['natural_evaluation_seconds'],
            family_forward_seconds=train_receipt['family_forward_seconds'])
        write_json(reports / 'receipt.json', value)
        return value


def parser():
    base = learner()
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('command', choices=('describe', 'capture-train', 'finish'))
    result.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    result.add_argument('--data-root', type=Path, default=base.DEFAULT_DATA_ROOT)
    result.add_argument('--prepared-root', type=Path, default=base.DEFAULT_PREPARED_ROOT)
    result.add_argument('--cache-root', type=Path)
    result.add_argument('--test-cache-root', type=Path)
    result.add_argument('--family-labels', type=Path, default=base.DEFAULT_FAMILY)
    result.add_argument('--family-cache', type=Path)
    result.add_argument('--family-input-manifest', type=Path, default=PREPARATION / 'full18_inputs_v1/manifest.json')
    result.add_argument('--natural-isolation', type=Path, default=base.DEFAULT_ISOLATION)
    result.add_argument('--device', default='cpu')
    result.add_argument('--release', type=Path)
    result.add_argument('--binding', type=Path)
    result.add_argument('--gpu-type', choices=('4090', '3090'), default='4090')
    result.add_argument('--micro-cap', type=int, default=128)
    result.add_argument('--eval-batch-size', type=int, default=128)
    result.add_argument('--lib-batch-size', type=int, default=1)
    result.add_argument('--wandb-mode', choices=('online', 'offline', 'disabled'), default='online')
    result.add_argument('--resume', action='store_true', help='Compatibility only; no training resume is performed')
    result.add_argument('--nss-root', type=Path, default=V4 / 'runs/R84_new90k_attention/nss_reference')
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    for key, value in vars(args).copy().items():
        if isinstance(value, Path):
            setattr(args, key, value.expanduser().resolve())
    args.arm, args.experiment = ARM, 'R121'
    if args.family_cache is None:
        args.family_cache = args.root / 'family_cache'
    if args.test_cache_root and args.test_cache_root.name == 'test' and (args.test_cache_root / 'TSP').is_dir():
        args.test_cache_root = args.test_cache_root.parent
    if args.micro_cap < 2 or args.eval_batch_size < 1 or args.lib_batch_size < 1:
        raise ValueError('Positive capacities and micro-cap>=2 required')
    if args.command == 'describe':
        value = dict(schema=SCHEMA, root=str(args.root), arm=ARM,
            capture='completed unique natural-VAL lock -> full81000/task + independent family diagnostics; main binding required',
            finish='saved-only original finish + separate mechanism supplement; no forward',
            receipt=str(args.root / ARM / 'learner_finish/reports/receipt.json'),
            GPU_execution=False, training=False, solver_execution=False, inference_started=False)
    elif args.command == 'capture-train':
        if args.release is None or args.binding is None:
            raise PermissionError('Reuse main release/binding; this wrapper never allocates a resource')
        value = capture_train(args)
    else:
        value = finish(args)
    print(json.dumps(value, sort_keys=True, allow_nan=False), flush=True)
    return value


if __name__ == '__main__':
    main()

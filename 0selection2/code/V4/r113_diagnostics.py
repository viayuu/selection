"""Bounded, post-lock R113 process generalization, never training or selection.

Freeze the shared input-only VAL subset before launch with ``python -B -m
code.V4.r113_diagnostics freeze --root ROOT``. Only the selector finalizer calls
``finalize_diagnostics``; ordinary TEST/report/deployment do not run a teacher.
"""

import argparse
import csv
import fcntl
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from . import r113_experiment as runner
from .r112_experiment import adapt_function, cached_input_batch
from .r113_process_data import BINARY_FIELDS, GRAPH_FIELDS, SCALAR_FIELDS, execute_program
from .r75_cache import _close, _open_arrays
from .r76_cache import _validate_record
from .r76_experiment import file_hash, read_json, utc, write_json
from .r77_data import parameter_hash


VERSION = 'r113_postlock_process_diagnostics_v1'
INPUTS_FILE = 'process_diagnostic_inputs.json'
COMPLETE_FILE = 'process_diagnostics_complete.json'
SAMPLES, SEED, BATCH_SIZE = 64, 2, 8
POLICY = dict(version=VERSION, split='val', seed=SEED, rows_per_problem=SAMPLES,
    stratification='true n only; proportional largest remainder; ties by ascending n',
    sampling='seed2 permutation within each size; sort selected cache row IDs',
    shared_across_all_arms=True, selection_labels_read=False,
    normalized_mse='per-field MSE / max(1, mean(target**2)); mean over active fields',
    binary_threshold=.5, exclude_t0=True, exclude_padding=True,
    directed_pairs='all valid ordered non-self pairs, not just topology edges',
    precision_when_no_predicted_positive=None, recall_when_no_true_positive=None)
COMPONENTS = ('scalar', 'binary', 'pointer', 'compatible', 'reached', 'frontier')


def stratified_rows(sizes, count=SAMPLES):
    sizes = np.asarray(sizes)
    if (sizes.ndim != 1 or not np.issubdtype(sizes.dtype, np.integer)
            or (sizes < 2).any() or type(count) is not int or not 1 <= count <= len(sizes)):
        raise ValueError('Require enough integer input sizes for the fixed subset')
    values, populations = np.unique(sizes, return_counts=True)
    quotas = count * populations / len(sizes)
    allocations = np.floor(quotas).astype(np.int64)
    order = np.lexsort((values, -(quotas - allocations)))
    allocations[order[:count - int(allocations.sum())]] += 1
    rng, selected = np.random.default_rng(SEED), []
    for size, amount in zip(values, allocations):
        selected.extend(rng.permutation(np.flatnonzero(sizes == size))[:amount].tolist())
    return np.array(sorted(selected), dtype=np.int64)


def _cache_directory(config, problem):
    root = Path(config['cache_root'])
    path = next((p for p in (root / 'val' / problem, root / (problem + 'val')) if p.is_dir()), None)
    if path is None:
        raise FileNotFoundError('Existing frozen VAL input cache required: ' + problem)
    return path


@lru_cache(maxsize=1)
def _row_reader():
    # R112 has a prefix-only reader. Adapt only its gathers in a private namespace,
    # retaining its provenance checks without changing/monkeypatching that module.
    namespace = SimpleNamespace(**cached_input_batch.__globals__)
    adapt_function(namespace, cached_input_batch, (
        ('device, count=None):', 'device, count=None, *, rows):'),
        ("arrays['n'][:count]", "arrays['n'][rows]"),
        ("arrays['coord_dist'][:count]", "arrays['coord_dist'][rows]"),
        ('arrays[key][:count, :maximum]', 'arrays[key][rows, :maximum]'),
        ("arrays['matrix'][:count, :maximum, :maximum]", "arrays['matrix'][rows, :maximum, :maximum]")))
    return namespace.cached_input_batch


def cached_input_rows(config, manifest, problem, rows, *, directory=None):
    rows = np.asarray(rows)
    if (rows.ndim != 1 or not np.issubdtype(rows.dtype, np.integer) or not len(rows)
            or (rows < 0).any() or (rows >= manifest['splits']['val'][problem]['rows']).any()):
        raise ValueError('Invalid fixed VAL input row IDs')
    if directory is not None:
        directory = Path(directory)
        root = directory.parent.parent if directory.name == problem and directory.parent.name == 'val' else directory.parent
        # Reader location only: never change the main protocol or checkpoint config.
        config = dict(config, cache_root=str(root))
    return _row_reader()(config, manifest, 'val', problem, torch.device('cpu'), count=len(rows), rows=rows)


def _input_row_hashes(config, manifest, problem, directory, rows):
    hashes = []
    for start in range(0, len(rows), BATCH_SIZE):
        batch = cached_input_rows(config, manifest, problem, rows[start:start + BATCH_SIZE], directory=directory)
        for index in range(len(batch['n'])):
            digest = hashlib.sha256()
            for key in sorted(batch):
                value = batch[key]
                digest.update(key.encode('ascii'))
                if torch.is_tensor(value):
                    value = value if key == 'pool_ids' else value[index]
                    array = value.contiguous().numpy()
                    digest.update(str(array.dtype).encode('ascii') + str(array.shape).encode('ascii') + array.tobytes())
                else:
                    digest.update(json.dumps(value, sort_keys=True).encode('ascii'))
            hashes.append(digest.hexdigest())
    return hashes


def _input_manifest(config, manifest, *, local=False):
    sources = {}
    for problem in runner.PROBLEMS:
        expected = manifest['splits']['val'][problem]
        authoritative = Path(expected['directory']) if expected.get('directory') else _cache_directory(config, problem)
        directory = _cache_directory(config, problem) if local else authoritative
        metadata = directory / 'metadata.json'
        record = read_json(metadata)
        specs = _validate_record(record, directory)
        if (record['problem'] != problem or record['split'] != 'val'
                or record['rows'] != runner.SPLIT_ROWS['val'] or record['rows'] != expected['rows']
                or record['pool'] != config['candidate_pools'][problem]
                or record['pool_ids'] != config['candidate_pool_ids'][problem]
                or record['pool'] != expected['pool'] or record['pool_ids'] != expected['pool_ids']
                or Path(record['data_root']).resolve() != Path(config['data_root']).resolve()):
            raise ValueError('Fixed diagnostic input cache provenance differs: ' + problem)
        arrays = _open_arrays(directory, {'n': specs['n']})
        try:
            sizes = np.array(arrays['n'], copy=True)
        finally:
            _close(arrays)
        ids_path = Path(config['data_root']) / (problem + 'val') / 'input_metadata.npz'
        with np.load(ids_path, allow_pickle=False) as archive:
            ids = np.array(archive['source_index'], copy=True)
        if (ids.shape != sizes.shape or not np.issubdtype(ids.dtype, np.integer)
                or len(np.unique(ids)) != len(ids)):
            raise ValueError('Frozen VAL source IDs must uniquely match cache rows')
        rows = stratified_rows(sizes)
        hashes = _input_row_hashes(config, manifest, problem, directory, rows)
        sources[problem] = dict(cache_directory=str(authoritative.resolve()),
            cache_metadata_sha256=file_hash(metadata), n_sha256=file_hash(directory / specs['n']['file']),
            source_ids_path=str(ids_path.resolve()), source_ids_sha256=file_hash(ids_path),
            population=len(sizes), input_byte_identity='metadata.json and n.npy SHA256; exact selected input tensor bytes including cropped padding',
            input_rows=[dict(cache_row_id=int(r), source_index=int(ids[r]),
                n=int(sizes[r]), input_sha256=digest) for r, digest in zip(rows, hashes)])
    value = dict(experiment='R113', policy=POLICY, complete=True,
        diagnostic_source_sha256=file_hash(Path(__file__)),
        main_protocol_lock_sha256=config['main_protocol_lock']['sha256'],
        source_cache_manifest_sha256=file_hash(Path(config['prepared_root']) / 'source_cache_manifest.json'),
        data_root=str(Path(config['data_root']).resolve()), sources=sources)
    return dict(value, subset_sha256=runner.json_hash(value))


def freeze_inputs(root, protocol_lock=None):
    """Input-only prelaunch operation; no weights, evaluation outputs or teacher."""
    root = Path(root).expanduser().resolve()
    path = Path(protocol_lock or root / 'protocol_lock.json').expanduser().resolve()
    record = read_json(path)
    if any(record.get(k) != v for k, v in runner.required_lock_fields().items()):
        raise PermissionError('Freeze diagnostics only against the main locked R113 protocol')
    manifest = read_json(Path(record['prepared_root']) / 'source_cache_manifest.json')
    config = dict(data_root=record['data_root'], cache_root=record['primary_cache'],
        prepared_root=record['prepared_root'], main_protocol_lock=dict(sha256=file_hash(path)),
        candidate_pools={p: manifest['splits']['val'][p]['pool'] for p in runner.PROBLEMS},
        candidate_pool_ids={p: manifest['splits']['val'][p]['pool_ids'] for p in runner.PROBLEMS})
    root.mkdir(parents=True, exist_ok=True)
    with (root / 'process_diagnostic_inputs.lock').open('a') as claim:
        fcntl.flock(claim, fcntl.LOCK_EX | fcntl.LOCK_NB)
        destination = root / INPUTS_FILE
        if not destination.exists() and any((root / arm / name).exists()
                for arm in runner.ARMS for name in ('history.json', 'best_eval.json', 'test_results.json',
                    'final_metrics.json', COMPLETE_FILE)):
            raise PermissionError('Freeze the common VAL inputs before any arm produces selection/results')
        value = _input_manifest(config, manifest)
        if destination.exists():
            if read_json(destination) != value:
                raise ValueError('Existing shared diagnostic subset differs; never replace it')
        else:
            write_json(destination, value)
    return value


def masked_counts(program, outputs, targets):
    """Sufficient statistics on active post-initial states, never padded entries."""
    p, o, y = ({k: v.detach().cpu() for k, v in values.items()}
               for values in (program, outputs, targets))
    batch, width = p['node_mask'].shape
    active = torch.arange(width)[None] < p['steps'][:, None]
    if not torch.equal(active, y['transition_mask']):
        raise ValueError('Teacher lengths disagree with input-only steps')
    rows = []

    def binary(component, field, predicted, target, mask):
        mask = torch.broadcast_to(mask, predicted.shape)
        pred, truth = predicted[mask], torch.broadcast_to(target, predicted.shape)[mask].bool()
        if not torch.isfinite(pred).all():
            raise ValueError('Non-finite active process prediction')
        pred = pred >= POLICY['binary_threshold']
        rows.append(dict(component=component, field=field, valid_count=int(mask.sum()),
            tp=int((pred & truth).sum()), tn=int((~pred & ~truth).sum()),
            fp=int((pred & ~truth).sum()), fn=int((~pred & truth).sum())))

    for index, field in enumerate(SCALAR_FIELDS):
        mask = active & p['scalar_mask'][:, index, None]
        pred, truth = o['scalar'][:, 1:, index][mask].double(), y['scalar'][:, 1:, index][mask].double()
        if not torch.isfinite(pred).all() or not torch.isfinite(truth).all():
            raise ValueError('Non-finite active scalar state')
        rows.append(dict(component='scalar', field=field, valid_count=int(mask.sum()),
            squared_error_sum=float((pred - truth).square().sum()), target_squared_sum=float(truth.square().sum())))
    for index, field in enumerate(BINARY_FIELDS):
        binary('binary', field, o['binary'][:, 1:, index], y['binary'][:, 1:, index],
            active & p['binary_mask'][:, index, None])
    pointer = o['pointer'][:, 1:]
    valid_pointer = torch.cat((p['node_mask'], torch.ones(batch, 1, dtype=torch.bool)), -1)
    pointer_mask = active[..., None] & valid_pointer[:, None]
    if not torch.isfinite(pointer[pointer_mask]).all():
        raise ValueError('Non-finite active pointer probability')
    prediction = pointer.masked_fill(~valid_pointer[:, None], float('-inf')).argmax(-1)
    truth = y['pointer'][:, 1:].long()
    if ((truth[active] < 0) | (truth[active] >= width)).any():
        raise ValueError('Post-initial teacher pointers must reference a valid program node')
    if not p['node_mask'].gather(1, truth.clamp(0, width - 1))[active].all():
        raise ValueError('Teacher points at padding')
    rows.append(dict(component='pointer', field='node_index', valid_count=int(active.sum()),
        correct=int((prediction[active] == truth[active]).sum())))
    pair = p['node_mask'][:, :, None] & p['node_mask'][:, None]
    pair &= ~torch.eye(width, dtype=torch.bool)[None]
    for index, field in enumerate(GRAPH_FIELDS):
        steps = active & p['graph_mask'][:, index, None]
        binary('compatible', field, o['compatible'][:, 1:, index], y['compatible'][:, index, None],
            steps[..., None, None] & pair[:, None])
        for component in ('reached', 'frontier'):
            binary(component, field, o[component][:, 1:, index], y[component][:, 1:, index],
                steps[..., None] & p['node_mask'][:, None])
    return rows


def summarize_counts(records):
    combined = {}
    for row in records:
        key = row['problem'], row['component'], row['field']
        current = combined.setdefault(key, {k: row[k] for k in ('problem', 'component', 'field')})
        for name, value in row.items():
            if name not in ('problem', 'component', 'field'):
                current[name] = current.get(name, 0) + value
    rows = list(combined.values())
    for problem in dict.fromkeys(r['problem'] for r in rows):
        for component in COMPONENTS:
            fields = [r for r in rows if r['problem'] == problem and r['component'] == component]
            aggregate = dict(problem=problem, component=component, field='ALL')
            for row in fields:
                for key, value in row.items():
                    if key not in ('problem', 'component', 'field'):
                        aggregate[key] = aggregate.get(key, 0) + value
            rows.append(aggregate)
    for row in rows:
        count, component = row['valid_count'], row['component']
        if component == 'scalar':
            row['mse'] = row['squared_error_sum'] / count if count else None
            row['normalized_mse'] = row['squared_error_sum'] / max(count, row['target_squared_sum']) if count else None
        elif component == 'pointer':
            row['accuracy'] = row['correct'] / count if count else None
        else:
            tp, tn, fp, fn = (row[k] for k in ('tp', 'tn', 'fp', 'fn'))
            row.update(accuracy=(tp + tn) / count if count else None,
                precision=tp / (tp + fp) if tp + fp else None,
                recall=tp / (tp + fn) if tp + fn else None,
                all_zero_accuracy=(tn + fp) / count if count else None,
                true_positive_count=tp + fn, predicted_positive_count=tp + fp)
    for row in rows:
        if row['component'] == 'scalar' and row['field'] == 'ALL':
            active = [r['normalized_mse'] for r in rows if r['problem'] == row['problem']
                and r['component'] == 'scalar' and r['field'] != 'ALL' and r['valid_count']]
            row['normalized_mse'] = sum(active) / len(active) if active else None
    return rows


def _completed_test(args):
    lock, config = runner.authorize_test(args)
    run = Path(args.root) / args.arm
    saved, final = read_json(run / 'test_results.json'), read_json(run / 'final_metrics.json')
    expected = dict(checkpoint_sha256=lock['checkpoint_sha256'], state_hash=lock['state_hash'],
        selected_update=lock['selected_update'])
    if (saved.get('complete') is not True or final.get('complete') is not True
            or any(saved.get(k) != v or final.get(k) != v for k, v in expected.items())
            or saved.get('result') != final.get('test')):
        raise PermissionError('Process teacher requires complete same-checkpoint locked full TEST')
    result = saved['result']
    if (set(result.get('per_problem', {})) != set(runner.PROBLEMS)
            or any(result['per_problem'][p].get('n') != runner.SPLIT_ROWS['test'] for p in runner.PROBLEMS)
            or result.get('identity', {}).get('checkpoint_sha256') != lock['checkpoint_sha256']):
        raise PermissionError('Process diagnostics cannot precede full18 locked TEST')
    if file_hash(run / 'best.pt') != lock['checkpoint_sha256']:
        raise ValueError('Diagnostic checkpoint SHA differs from the immutable lock')
    return run, config, lock, result['identity']['model_state_sha256']


def saved_diagnostics(run, config, lock):
    """Saved-only report integration: absent/partial artifacts are honestly missing."""
    run = Path(run)
    marker = run / COMPLETE_FILE
    if not marker.is_file():
        return dict(status='missing', reason='Post-lock process diagnostics not completed; no process evidence claimed.')
    receipt = read_json(marker)
    expected = dict(experiment='R113', arm=config['arm'], complete=True, version=VERSION,
        checkpoint_sha256=lock['checkpoint_sha256'], state_hash=lock['state_hash'],
        selected_update=lock['selected_update'], diagnostic_source_sha256=file_hash(Path(__file__)))
    if any(receipt.get(k) != v for k, v in expected.items()):
        raise ValueError('Saved process diagnostics name another checkpoint/protocol/source')
    files = {name: run / 'process_diagnostics' / ('process_diagnostics.' + name) for name in ('json', 'csv', 'pdf')}
    if any(not path.is_file() or receipt.get('files', {}).get(name, {}).get('sha256') != file_hash(path)
           for name, path in files.items()):
        return dict(status='missing', reason='Process completion receipt has missing/partial diagnostic files.')
    value = read_json(files['json'])
    frozen = read_json(run.parent / INPUTS_FILE)
    if (value.get('complete') is not True or value.get('policy') != POLICY
            or any(value.get(k) != v for k, v in expected.items())
            or value.get('inputs') != frozen or receipt.get('subset_sha256') != frozen['subset_sha256']
            or frozen['main_protocol_lock_sha256'] != config['main_protocol_lock']['sha256']):
        raise ValueError('Saved process diagnostics do not match the common frozen inputs')
    return dict(status='complete', files={k: str(v) for k, v in files.items()},
        complete_flag=str(marker), subset_sha256=frozen['subset_sha256'], value=value)


def append_pdf_pages(report, evidence):
    if evidence['status'] != 'complete':
        return 0
    import matplotlib.pyplot as plt
    value = evidence['value']
    lookup = {(r['problem'], r['component']): r for r in value['metrics'] if r['field'] == 'ALL'}
    problems = tuple(value['inputs']['sources'])

    def number(x):
        return 'N/A' if x is None else f'{x:.4f}'

    for component in ('overview', 'binary', 'compatible', 'reached', 'frontier'):
        fig = plt.figure(figsize=(11.7, 8.3))
        try:
            fig.suptitle('R113/' + value['arm'] + ' | Post-lock VAL process: ' + component, fontsize=16, x=.05, ha='left')
            fig.text(.05, .91, '64 fixed inputs/task; seed2, size-only strata; eval-only CPU teacher after complete locked TEST.', fontsize=10)
            if component == 'overview':
                headings = ['Problem', 'Scalar NMSE', 'Binary acc', 'Pointer acc', 'Directed acc', 'Reached acc', 'Frontier acc']
                cells = [[p, *[number(lookup[p, c].get('normalized_mse' if c == 'scalar' else 'accuracy'))
                    for c in COMPONENTS]] for p in problems]
            else:
                headings = ['Problem', 'Active entries', 'Accuracy', 'Precision', 'Recall', 'All-zero acc', 'True positives']
                cells = [[p, str(lookup[p, component]['valid_count']),
                    *[number(lookup[p, component][k]) for k in ('accuracy', 'precision', 'recall', 'all_zero_accuracy')],
                    str(lookup[p, component]['true_positive_count'])] for p in problems]
            axes = fig.add_axes([.05, .21, .9, .64])
            axes.axis('off')
            table = axes.table(cellText=cells, colLabels=headings, cellLoc='center', bbox=[0, 0, 1, 1])
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            notes = ['No t0, padded nodes, inactive types or diagonal pairs; directed source/target axes preserved.',
                'Scalar NMSE: per-field MSE/max(1, mean(target^2)), then mean active fields. Acc/P/R are fractions.',
                'Threshold=.5; undefined precision/recall=N/A. Sparse all-zero accuracy is not learned execution.',
                'Teacher never enters model inputs, training, checkpoint selection or deployment; no solver calls.',
                'Checkpoint SHA256: ' + value['checkpoint_sha256'],
                'Shared input subset SHA256: ' + value['inputs']['subset_sha256'],
                'Saved detail: process_diagnostics.pdf / process_diagnostics.csv / process_diagnostics.json']
            for index, note in enumerate(notes):
                fig.text(.05, .18 - .022 * index, note, fontsize=8)
            report.savefig(fig)
        finally:
            plt.close(fig)
    return 5


def _write_csv(path, rows):
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(k for row in rows for k in row)))
        writer.writeheader()
        writer.writerows(rows)


@torch.no_grad()
def finalize_diagnostics(args):
    """Called only by finalize, after locked full TEST; always bounded on CPU."""
    run, config, lock, model_hash = _completed_test(args)
    with (run / 'process_diagnostics.lock').open('a') as claim:
        fcntl.flock(claim, fcntl.LOCK_EX | fcntl.LOCK_NB)
        frozen_path = Path(args.root) / INPUTS_FILE
        if not frozen_path.is_file():
            raise PermissionError('Missing prelaunch ' + INPUTS_FILE + '; do not choose VAL samples after results')
        frozen = read_json(frozen_path)
        manifest = read_json(Path(config['prepared_root']) / 'source_cache_manifest.json')
        if _input_manifest(config, manifest, local=True) != frozen:
            raise ValueError('Frozen process diagnostic input identity changed')
        saved = saved_diagnostics(run, config, lock)
        if saved['status'] == 'complete':
            return {k: v for k, v in saved.items() if k != 'value'}
        model, state = runner.load_checkpoint(run / 'best.pt', torch.device('cpu'))
        if parameter_hash(state['model']) != model_hash or state['args'] != config:
            raise ValueError('Diagnostic model differs from the completed TEST checkpoint')
        del state
        model.eval()
        records, input_hashes = [], {}
        for problem in runner.PROBLEMS:
            identities = frozen['sources'][problem]['input_rows']
            input_hashes[problem] = []
            for start in range(0, SAMPLES, BATCH_SIZE):
                ids = identities[start:start + BATCH_SIZE]
                batch = cached_input_rows(config, manifest, problem, np.array([r['cache_row_id'] for r in ids]))
                if batch['n'].tolist() != [r['n'] for r in ids]:
                    raise ValueError('Fixed input row/size alignment changed')
                prepared = runner.models.prepare_role_inputs(batch, runner.models.SPEC['role_arm'])
                program, outputs = model.instance_encoder.process_state(prepared)
                # The teacher runs strictly after the input-only predicted rollout.
                targets = execute_program(program)
                records.extend(dict(row, problem=problem) for row in masked_counts(program, outputs, targets))
                for index, identity in enumerate(ids):
                    digest = hashlib.sha256()
                    for key in sorted(program):
                        tensor = program[key][index].detach().cpu().contiguous().numpy()
                        digest.update(key.encode('ascii') + str(tensor.dtype).encode('ascii'))
                        digest.update(str(tensor.shape).encode('ascii') + tensor.tobytes())
                    input_hashes[problem].append(dict(identity, compiled_input_sha256=digest.hexdigest()))
                del batch, prepared, program, outputs, targets
        if (file_hash(run / 'best.pt') != lock['checkpoint_sha256']
                or parameter_hash(model.state_dict()) != model_hash):
            raise ValueError('Process diagnostic changed the locked checkpoint/model')
        value = dict(experiment='R113', arm=args.arm, version=VERSION, complete=True, completed_utc=utc(),
            checkpoint_sha256=lock['checkpoint_sha256'], state_hash=lock['state_hash'],
            selected_update=lock['selected_update'], model_state_sha256=model_hash,
            diagnostic_source_sha256=file_hash(Path(__file__)), policy=POLICY, inputs=frozen,
            input_rows=input_hashes, metrics=summarize_counts(records), device='cpu',
            model_mode='eval; self-initialized predicted feedback; role preparation matches deployment',
            teacher_scope='input-native execute_program; post-lock evaluation only, after full TEST',
            model_reselection=False, teacher_in_model_input=False, teacher_used_for_training=False,
            cost_or_winner_arrays_opened=False, native_solver_calls=0)
        output = run / 'process_diagnostics'
        output.mkdir(parents=True, exist_ok=True)
        files = {name: output / ('process_diagnostics.' + name) for name in ('json', 'csv', 'pdf')}
        from matplotlib.backends.backend_pdf import PdfPages
        for name, destination in files.items():
            temporary = output / ('.' + destination.stem + f'.{os.getpid()}.' + name)
            try:
                if name == 'json':
                    write_json(temporary, value)
                elif name == 'csv':
                    _write_csv(temporary, [dict(r, checkpoint_sha256=lock['checkpoint_sha256'],
                        subset_sha256=frozen['subset_sha256'], split='VAL', instances=SAMPLES) for r in value['metrics']])
                else:
                    with PdfPages(temporary) as pdf:
                        append_pdf_pages(pdf, dict(status='complete', value=value))
                temporary.replace(destination)
            finally:
                temporary.unlink(missing_ok=True)
        write_json(run / COMPLETE_FILE, {k: value[k] for k in ('experiment', 'arm', 'version', 'complete',
            'completed_utc', 'checkpoint_sha256', 'state_hash', 'selected_update', 'diagnostic_source_sha256')} | dict(
                subset_sha256=frozen['subset_sha256'], files={k: dict(path=str(p), sha256=file_hash(p)) for k, p in files.items()}))
        return {k: v for k, v in saved_diagnostics(run, config, lock).items() if k != 'value'}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('freeze',))
    parser.add_argument('--root', type=Path, default=runner.DEFAULT_ROOT)
    parser.add_argument('--protocol-lock', type=Path)
    args = parser.parse_args(argv)
    value = freeze_inputs(args.root, args.protocol_lock)
    print(json.dumps(dict(complete=True, path=str(args.root.expanduser().resolve() / INPUTS_FILE),
        subset_sha256=value['subset_sha256'], rows_per_problem=SAMPLES, seed=SEED), indent=2))
    return value


if __name__ == '__main__':
    main()

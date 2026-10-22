"""Additive R94 timing correction. Never train, select, or overwrite R94 outputs.

CPU: preflight, cpu-smoke, report. GPU: measure --gpu-authorized --device cuda:0.
All writes are confined to R94_pretrained_node_fusion/timing_correction.
"""

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path


V4 = Path(__file__).resolve().parents[1]
ROOT = V4 / 'runs/R94_pretrained_node_fusion'
OUTPUT = ROOT / 'timing_correction'
FROZEN_COMMIT = 'cf2f06bc4'
PINS = dict(A='872315c1bded11e1de128c5006b683b47b9aa30b9314d793bb3d0e871e6b5c42',
            B='8e797150d22ea138db246536f36a1b7dfa6ef3b516ee065a3303a6b6e579c3ee')
PROBLEMS = ('TSP', 'CVRP', 'ATSP', 'OVRP', 'VRPB', 'VRPL', 'VRPTW', 'OVRPTW',
            'OVRPB', 'OVRPL', 'VRPBL', 'VRPBTW', 'VRPLTW', 'OVRPBL', 'OVRPBTW',
            'OVRPLTW', 'VRPBLTW', 'OVRPBLTW')
COUNT, REPETITIONS, WARMUPS = 128, 3, 1
SCOPES = ('native_donor', 'selector_only', 'donor_plus_selector')
INPUT_KEYS = ('problem_id', 'kind', 'd_in', 'n', 'node_mask', 'cbits', 'problem_desc',
              'coord_dist', 'pool_ids', 'mask', 'solver_mask', 'node', 'node_geom', 'matrix')
EXCLUDES = ('Excludes dataset/checkpoint I/O, hashing, parsing/collation, FP64 local-geometry '
            'preparation, CPU-to-GPU transfer, warmup and post-timing validation. '
            'No solver decoder or full-system/I/O-inclusive end-to-end latency.')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(name, value, *, exclusive=False):
    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT / name
    if path.parent != OUTPUT or path.suffix != '.json':
        raise ValueError('Only correction JSON files may be written')
    with path.open('x' if exclusive else 'w', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def utc():
    return datetime.now(timezone.utc).isoformat()


def raw_inputs(batch):
    # An allowlist, not the legacy label-only denylist that leaked nco_*.
    return {key: batch[key] for key in INPUT_KEYS if key in batch}


def locked_artifacts(arm):
    from code.V4.r94_experiment import authorize_test
    directory = ROOT / arm
    config = read_json(directory / 'protocol.json')
    args = argparse.Namespace(root=ROOT, arm=arm, experiment='R94', state_hash=None,
                              data_root=Path(config['data_root']))
    lock, config = authorize_test(args)
    if lock['checkpoint_sha256'] != PINS[arm]:
        raise PermissionError('Recovery is pinned to the completed cf2f06bc4 checkpoint')
    test = read_json(directory / 'test_results.json')
    if (test.get('complete') is not True or test.get('checkpoint_sha256') != PINS[arm]
            or test.get('state_hash') != lock['state_hash']):
        raise PermissionError('Require the completed same-checkpoint TEST receipt')
    for name, expected in config['source_sha256'].items():
        if sha256(V4 / name) != expected:
            raise ValueError('Frozen source differs from checkpoint protocol: ' + name)
    return lock, config


def protected_files():
    paths = list(V4.glob('r94*.py')) + list(V4.glob('test_r94*.py'))
    paths += [V4 / 'run_v4_r94_launch.sh', ROOT / 'protocol_lock.json',
              ROOT / 'RESULTS.md', ROOT / 'RESULTS.pdf']
    for arm in PINS:
        paths += [ROOT / arm / name for name in
                  ('best.pt', 'protocol.json', 'final_test_lock.json', 'test_results.json',
                   'training_complete.json', 'inference_timing.json')]
        paths += [p for subdir in ('full_model_statistics', 'lib_evaluation', 'evaluation')
                  for p in (ROOT / arm / subdir).rglob('*') if p.is_file()]
    paths += [p for p in (ROOT / 'comparison_artifacts').rglob('*') if p.is_file()]
    paths += [p for p in (ROOT / 'B').glob('ROUND_ANALYSIS.*') if p.is_file()]
    result = {}
    for path in sorted(set(paths)):
        if path.is_file():
            stat = path.stat()
            identity = dict(size_bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)
            # Prediction shards exceed 6GB; do not reread them for a timing erratum.
            if path.suffix not in ('.npz', '.npy'):
                identity['sha256'] = sha256(path)
            result[str(path)] = identity
    return result


def preservation_check():
    path = OUTPUT / 'preservation_before.json'
    if not path.exists():
        write_json(path.name, dict(created_utc=utc(), files=protected_files()), exclusive=True)
    prior = read_json(path)['files']
    changed = []
    for name, identity in prior.items():
        path = Path(name)
        if not path.is_file():
            changed.append(name)
            continue
        stat = path.stat()
        if ((stat.st_size, stat.st_mtime_ns) != (identity['size_bytes'], identity['mtime_ns'])
                or 'sha256' in identity and sha256(path) != identity['sha256']):
            changed.append(name)
    if changed:
        raise RuntimeError('Protected artifacts changed; do not revert: ' + ', '.join(changed))
    return dict(verified_utc=utc(), files_checked=len(prior), all_unchanged=True)


def input_record(config, problem):
    manifest_path = Path(config['prepared_root']) / 'source_cache_manifest.json'
    manifest = read_json(manifest_path)
    record = manifest['splits']['test'][problem]
    if (manifest.get('complete') is not True or record.get('complete') is not True
            or record.get('rows') != 4500 or record.get('problem') != problem
            or record.get('problem_id') != PROBLEMS.index(problem)
            or record.get('split') != 'test' or record.get('data_root') != config['data_root']):
        raise ValueError('Wrong frozen TEST input manifest: ' + problem)
    directory = Path(config['data_root']) / (problem + 'test')
    for name in ('dataset.pkl', 'input_metadata.npz', 'metadata.json'):
        path = directory / name
        source = next(p for p in record['source_files'] if p['path'] == str(path))
        stat = path.stat()
        if (stat.st_size, stat.st_mtime_ns) != (source['size_bytes'], source['mtime_ns']):
            raise ValueError('Frozen input identity changed: ' + str(path))
    return record


def input_tensor(instance, problem, bits):
    import torch
    from code.V4.r60_data import _node_input
    if problem == 'TSP':
        return torch.as_tensor(instance[0], dtype=torch.float32)
    if problem == 'ATSP':
        return torch.as_tensor(instance[0] if instance.ndim == 3 else instance, dtype=torch.float32)
    if problem == 'CVRP':
        xy = torch.as_tensor(instance['loc'][0], dtype=torch.float32)
        demand = torch.as_tensor(instance['demand'][0], dtype=torch.float32)
        depot = torch.as_tensor(instance['depot'][0], dtype=torch.float32)
        return torch.cat((torch.cat((depot, torch.zeros(1, 1), torch.ones(1, 1)), 1),
                          torch.cat((xy, demand[:, None], torch.zeros(len(xy), 1)), 1)), 0)
    return torch.from_numpy(_node_input(instance, bits))


def prepare_batch(config, problem, count=COUNT):
    """Read original inputs only; never open raw_label, costs, winner or nco arrays."""
    import numpy as np
    import torch
    from code.V4.local_geometry import FIELDS, local_geometry
    from code.V4.r75_data import _CPUUnpickler

    record = input_record(config, problem)
    directory = Path(config['data_root']) / (problem + 'test')
    with (directory / 'dataset.pkl').open('rb') as stream:
        instances = _CPUUnpickler(stream).load()
    if len(instances) != record['rows'] or not 0 < count <= COUNT:
        raise ValueError('Fixed TEST row selection changed')
    values = [input_tensor(instances[i], problem, record['cbits']) for i in range(count)]
    del instances
    n = torch.tensor([len(value) for value in values], dtype=torch.long)
    maximum = int(n.max())
    kind = record['kind']
    tensor = torch.zeros((count, maximum, maximum) if kind == 'matrix'
                         else (count, maximum, record['d_in']), dtype=torch.float32)
    for row, value in enumerate(values):
        if kind == 'matrix':
            tensor[row, :len(value), :len(value)] = value
        else:
            tensor[row, :len(value)] = value
    with np.load(directory / 'input_metadata.npz', allow_pickle=False) as metadata:
        distributions = metadata['coord_dist'][:count].copy()
        source_ids = metadata['source_index'][:count].tolist()
    batch = dict(problem_id=record['problem_id'], kind=kind, n=n,
        node_mask=torch.arange(maximum)[None] < n[:, None],
        cbits=torch.tensor(record['cbits'], dtype=torch.float32).repeat(count, 1),
        problem_desc=torch.tensor(record['problem_desc'], dtype=torch.float32).repeat(count, 1),
        coord_dist=torch.from_numpy(distributions),
        pool_ids=torch.tensor(record['pool_ids'], dtype=torch.long),
        mask=torch.tensor(record['mask'], dtype=torch.float32).repeat(count, 1),
        solver_mask=torch.ones(count, len(record['pool']), dtype=torch.bool))
    if kind == 'matrix':
        batch['matrix'] = tensor
    else:
        batch.update(node=tensor, d_in=record['d_in'])
        batch['node_geom'] = torch.zeros(count, maximum, len(FIELDS))
        # The frozen FP64 geometry formula, outside inference; bounded CPU RAM.
        for start in range(0, count, 4):
            stop = min(count, start + 4)
            width = int(n[start:stop].max())
            batch['node_geom'][start:stop, :width] = local_geometry(
                tensor[start:stop, :width, :2], batch['node_mask'][start:stop, :width]).float()
    fingerprint = hashlib.sha256()
    for key, value in sorted(batch.items()):
        fingerprint.update(key.encode('ascii'))
        if torch.is_tensor(value):
            fingerprint.update(str((str(value.dtype), tuple(value.shape))).encode('ascii'))
            fingerprint.update(value.contiguous().numpy().tobytes())
        else:
            fingerprint.update(json.dumps(value).encode('ascii'))
    identity = dict(problem=problem, instances=count, row_indices=list(range(count)),
                    source_indices=source_ids, input_sha256=fingerprint.hexdigest(),
                    padded_nodes=maximum, unique_lengths=int(n.unique().numel()),
                    preparation='original TEST inputs + frozen FP64 geometry on CPU; no labels/cache',
                    dataset_sha256=sha256(directory / 'dataset.pkl'),
                    input_metadata_sha256=sha256(directory / 'input_metadata.npz'))
    return batch, identity


@contextmanager
def native_calls(model):
    counts = dict(donor=0, encoder=0)

    def donor_hook(*unused):
        counts['donor'] += 1

    def encoder_hook(*unused):
        counts['encoder'] += 1

    handles = (model.donor.register_forward_hook(donor_hook),
               model.donor.encoder.register_forward_hook(encoder_hook))
    try:
        yield counts
    finally:
        for handle in handles:
            handle.remove()


def timed_call(fn, device, model, expected, validate, *, clock=time.perf_counter):
    import torch

    def synchronize():
        if device.type == 'cuda':
            torch.cuda.synchronize(device)

    samples, calls = [], []
    result = None
    with native_calls(model) as counters:
        for index in range(WARMUPS + REPETITIONS):
            result = None
            counters.update(donor=0, encoder=0)
            synchronize()
            start = clock()
            result = fn()
            synchronize()
            elapsed = clock() - start
            # Assertions and finite/output checks are outside the measured span.
            if counters != expected:
                raise AssertionError(f'Native execution mismatch: {counters} != {expected}')
            validate(result)
            if index >= WARMUPS:
                if not math.isfinite(elapsed) or elapsed <= 0:
                    raise ValueError('Invalid synchronized elapsed time')
                samples.append(elapsed)
                calls.append(dict(counters))
    return result, dict(batch_seconds=samples, mean_batch_seconds=statistics.mean(samples),
                       std_batch_seconds=statistics.pstdev(samples), native_calls=calls,
                       warmup_native_calls=dict(expected))


def benchmark_problem(model, batch, problem, device):
    """Three independently timed scopes on the same immutable input tensors."""
    import torch
    raw = raw_inputs(batch)
    count = len(raw['n'])
    covered = problem not in ('TSP', 'ATSP')
    expected = dict(donor=int(covered), encoder=int(raw['n'].unique().numel()) if covered else 0)

    def finite_nodes(value):
        if value.dtype != torch.float32 or not torch.isfinite(value).all():
            raise ValueError('Nonfinite/non-native donor output')

    def finite_logits(value):
        if not torch.isfinite(value['logits'][value['solver_mask']]).all():
            raise ValueError('Nonfinite selector output')

    with torch.no_grad(), torch.autocast(device_type=device.type, enabled=False):
        if covered:
            nodes, donor = timed_call(lambda: model.donor(raw), device, model, expected, finite_nodes)
            # Same FP16 quantization as frozen R94.prepare_inputs; not a disk sidecar.
            selector_input = dict(raw, nco_nodes=nodes.detach().to(torch.float16),
                                  nco_node_mask=raw['node_mask'])
        else:
            selector_input = raw
            donor = dict(batch_seconds=[0.] * REPETITIONS, mean_batch_seconds=0.,
                         std_batch_seconds=0., native_calls=[dict(expected) for _ in range(REPETITIONS)],
                         warmup_native_calls=dict(expected), status='not_applicable_raw_only')
        selector_result, selector = timed_call(lambda: model(selector_input), device, model,
                                               dict(donor=0, encoder=0), finite_logits)
        online_result, total = timed_call(lambda: model(raw), device, model, expected, finite_logits)
        torch.testing.assert_close(online_result['logits'], selector_result['logits'],
                                   rtol=1e-5, atol=1e-6)
    results = dict(native_donor=donor, selector_only=selector, donor_plus_selector=total)
    for value in results.values():
        value['mean_ms_per_instance'] = 1000. * value['mean_batch_seconds'] / count
    return dict(problem=problem, instances=count, donor_applicable=covered, measurements=results,
                online_matches_selector_only=True, donor_cache_read=False,
                raw_has_nco_fields=any(key.startswith('nco_') for key in raw))


def cpu_setup():
    import torch
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    if torch.cuda.is_initialized():
        raise RuntimeError('CPU mode must not initialize CUDA')


def preflight():
    preservation = preservation_check()
    arms = {}
    for arm in PINS:
        lock, config = locked_artifacts(arm)
        for problem in PROBLEMS:
            input_record(config, problem)
        arms[arm] = dict(checkpoint_sha256=lock['checkpoint_sha256'],
                         state_hash=lock['state_hash'], selected_update=lock['selected_update'],
                         inference_rule=lock['inference_rule'],
                         prepared_manifest_sha256=sha256(Path(config['prepared_root']) /
                                                         'source_cache_manifest.json'))
    value = dict(created_utc=utc(), complete=True, gpu_used=False, frozen_commit=FROZEN_COMMIT,
                 arms=arms, preservation=preservation, rows_per_task=COUNT,
                 repetitions=REPETITIONS, warmup_batches=WARMUPS, excludes=EXCLUDES)
    write_json('preflight.json', value)
    return value


def cpu_smoke():
    import torch
    from code.V4.r94_experiment import load_checkpoint
    from code.V4.r77_data import parameter_hash
    preflight()
    rows, inputs = [], {}
    for arm in PINS:
        _, config = locked_artifacts(arm)
        if arm == 'A':
            for problem in PROBLEMS:
                batch, identity = prepare_batch(config, problem)
                inputs[problem] = identity
                del batch
                print('CPU fixed128 input check:', problem, flush=True)
        model, state = load_checkpoint(ROOT / arm / 'best.pt', 'cpu')
        model.eval()
        original = parameter_hash(state['model'])
        del state
        for problem in ('CVRP', 'OVRPBLTW', 'TSP', 'ATSP'):
            batch, _ = prepare_batch(config, problem, count=1)
            result = benchmark_problem(model, batch, problem, torch.device('cpu'))
            rows.append(dict(arm=arm, problem=problem, instances=1,
                             equivalent=result['online_matches_selector_only'],
                             online_native_calls=result['measurements']['donor_plus_selector']['native_calls']))
        if parameter_hash(model.state_dict()) != original:
            raise AssertionError('CPU diagnostic mutated model state')
        del model
    if torch.cuda.is_initialized():
        raise AssertionError('CPU smoke initialized CUDA')
    value = dict(created_utc=utc(), passed=True, cuda_initialized=False,
                 official_checkpoints_loaded=True, cases=rows, not_a_latency_result=True,
                 checkpoint_sha256=PINS, fixed128_input_checks=inputs,
                 preservation=preservation_check())
    write_json('cpu_smoke.json', value)
    return value


def measure(args):
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if not args.gpu_authorized or not visible or ',' in visible or visible == '-1':
        raise PermissionError('Require user approval and exactly one CUDA_VISIBLE_DEVICES GPU')
    if args.device != 'cuda:0':
        raise ValueError('Use cuda:0 inside the single approved GPU visibility namespace')
    if len(args.arms) != len(set(args.arms)):
        raise ValueError('Duplicate arms are not a new timing experiment')
    preflight()
    smoke = read_json(OUTPUT / 'cpu_smoke.json')
    if smoke.get('passed') is not True or smoke.get('checkpoint_sha256') != PINS:
        raise PermissionError('Pass official-checkpoint CPU smoke before GPU execution')
    for arm in args.arms:
        if (OUTPUT / (arm + '_online_timing.json')).exists():
            raise FileExistsError('Preserve existing corrected measurements; no automatic overwrite')
    import torch
    from code.V4.r94_experiment import load_checkpoint
    from code.V4.r77_data import parameter_hash
    device = torch.device(args.device)
    if torch.cuda.device_count() != 1:
        raise RuntimeError('Expected exactly the approved single GPU')
    hardware = dict(visible_device=visible, name=torch.cuda.get_device_name(device),
                    torch_version=torch.__version__, cuda_version=torch.version.cuda,
                    cpu_threads=torch.get_num_threads(),
                    matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
                    cudnn_tf32=torch.backends.cudnn.allow_tf32,
                    selector_autocast=False, donor_dtype='float32', donor_autocast=False,
                    donor_tf32=False, fusion_feature_dtype='float16')
    for arm in args.arms:
        lock, config = locked_artifacts(arm)
        model, state = load_checkpoint(ROOT / arm / 'best.pt', device)
        model.eval()
        original = parameter_hash(state['model'])
        del state
        measurements = []
        for problem in PROBLEMS:
            batch, identity = prepare_batch(config, problem)
            if identity != smoke['fixed128_input_checks'][problem]:
                raise ValueError('Runtime inputs differ from CPU-audited fixed128 inputs')
            batch = {key: value.to(device) if torch.is_tensor(value) else value
                     for key, value in batch.items()}
            result = benchmark_problem(model, batch, problem, device)
            result['input_identity'] = identity
            measurements.append(result)
            del batch
            print(arm, problem, 'online ms/instance',
                  result['measurements']['donor_plus_selector']['mean_ms_per_instance'], flush=True)
        if parameter_hash(model.state_dict()) != original:
            raise AssertionError('Timing diagnostic mutated model state')
        preservation = preservation_check()
        write_json(arm + '_online_timing.json', dict(
            schema='r94_online_timing_recovery_v1', experiment='R94', arm=arm, complete=True,
            completed_utc=utc(), checkpoint_sha256=PINS[arm], state_hash=lock['state_hash'],
            model_state_sha256=original, selected_update=lock['selected_update'],
            inference_rule=lock['inference_rule'], rows_per_task=COUNT,
            repetitions=REPETITIONS, warmup_batches=WARMUPS, hardware=hardware,
            script_sha256=sha256(__file__), input_manifest_sha256=sha256(
                Path(config['prepared_root']) / 'source_cache_manifest.json'),
            synchronization='perf_counter wall clock; synchronize device before/after each call',
            native_donor_scope='unmodified donor wrapper including validation/packing and FP32 encoder; no FP16 cast',
            selector_scope='unmodified selector with resident FP16 nodes freshly encoded online, no disk donor cache',
            total_scope='unmodified model(raw), forcing donor execution and original FP16 quantization every call',
            overhead='two integer forward-hook counters included; assertions/validation excluded',
            excludes=EXCLUDES, model_reselection=False, predictions_saved=False,
            preservation=preservation, per_problem=measurements), exclusive=True)
        del model
    return dict(complete=True, arms=args.arms, cpu_report_command='report')


def corrected_rows(arm):
    path = OUTPUT / (arm + '_online_timing.json')
    if not path.exists():
        return {}
    receipt = read_json(path)
    if (receipt.get('schema') != 'r94_online_timing_recovery_v1'
            or receipt.get('complete') is not True or receipt.get('arm') != arm
            or receipt.get('checkpoint_sha256') != PINS[arm]
            or receipt.get('rows_per_task') != COUNT or receipt.get('repetitions') != REPETITIONS
            or receipt.get('warmup_batches') != WARMUPS):
        raise ValueError('Invalid corrected timing receipt: ' + str(path))
    lock = read_json(ROOT / arm / 'final_test_lock.json')
    if (receipt.get('state_hash') != lock['state_hash']
            or receipt.get('inference_rule') != lock['inference_rule']):
        raise ValueError('Corrected timing differs from the fixed checkpoint/rule')
    rows = receipt['per_problem']
    if [row['problem'] for row in rows] != list(PROBLEMS):
        raise ValueError('Corrected receipt requires the fixed full18 task order')
    for row in rows:
        covered = row['problem'] not in ('TSP', 'ATSP')
        identity = row['input_identity']
        if (row['instances'] != COUNT or identity['instances'] != COUNT
                or identity['problem'] != row['problem']
                or identity['row_indices'] != list(range(COUNT))
                or row['raw_has_nco_fields'] is not False or row['donor_cache_read'] is not False
                or row['online_matches_selector_only'] is not True):
            raise ValueError('Invalid fixed raw inputs/equivalence')
        expected = dict(donor=int(covered), encoder=identity['unique_lengths'] if covered else 0)
        for scope in SCOPES:
            value = row['measurements'][scope]
            calls = dict(donor=0, encoder=0) if scope == 'selector_only' else expected
            samples = value['batch_seconds']
            if (len(samples) != REPETITIONS or value['native_calls'] != [calls] * REPETITIONS
                    or value['warmup_native_calls'] != calls
                    or any(not math.isfinite(t) or t < 0 for t in samples)
                    or ((scope != 'native_donor' or covered) and any(t <= 0 for t in samples))
                    or (scope == 'native_donor' and not covered and any(t != 0 for t in samples))
                    or not math.isclose(value['mean_batch_seconds'], statistics.mean(samples), abs_tol=1e-12)
                    or not math.isclose(value['mean_ms_per_instance'],
                                        statistics.mean(samples) * 1000. / COUNT, abs_tol=1e-12)):
                raise ValueError('Invalid samples or missing native execution: ' + scope)
    return {row['problem']: row for row in rows}


def report():
    """CPU-only additive erratum, usable before or after the authorized GPU run."""
    preservation = preservation_check()
    rows, provenance, corrected = [], {}, {arm: corrected_rows(arm) for arm in PINS}
    for arm in PINS:
        original_path = ROOT / arm / 'inference_timing.json'
        original = read_json(original_path)
        provenance[arm] = dict(path=str(original_path), sha256=sha256(original_path),
                               original_receipt=original)
        if original['checkpoint_sha256'] != PINS[arm]:
            raise ValueError('Legacy timing must belong to the fixed R94 checkpoint')
        for old in original['per_problem']:
            problem = old['problem']
            row = dict(arm=arm, problem=problem, instances=COUNT,
                       donor_applicable=problem not in ('TSP', 'ATSP'),
                       legacy_nominal_ms=old['donor_plus_selector']['mean_seconds_per_instance'] * 1000.,
                       legacy_cached_ms=old['cache_hit_selector']['mean_seconds_per_instance'] * 1000.,
                       native_donor_ms='', selector_only_ms='', online_total_ms='',
                       corrected_status='PENDING_GPU_APPROVAL_AND_MEASUREMENT')
            if problem in corrected[arm]:
                new = corrected[arm][problem]['measurements']
                row.update(native_donor_ms=new['native_donor']['mean_ms_per_instance'],
                           selector_only_ms=new['selector_only']['mean_ms_per_instance'],
                           online_total_ms=new['donor_plus_selector']['mean_ms_per_instance'],
                           corrected_status='MEASURED_ONLINE')
            rows.append(row)
    if corrected['A'] and corrected['B']:
        for problem in PROBLEMS:
            if corrected['A'][problem]['input_identity'] != corrected['B'][problem]['input_identity']:
                raise ValueError('A/B did not use identical fixed input tensors')
    write_json('legacy_timing_preserved.json', provenance)
    with (OUTPUT / 'timing_correction.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    status = 'MEASURED_ONLINE' if all(corrected.values()) else 'ONLINE_TIMING_NOT_YET_ESTABLISHED'
    intro = [
        '# R94 Online Timing Erratum', '',
        'Status: ' + status + '. Additive correction to frozen version cf2f06bc4.', '',
        'The old nominal donor+selector branch retained nco_nodes/nco_node_mask from '
        'FrozenFeatureCache.batch. pairwise_objective.inputs_only removes labels but not '
        'those fields; R94.prepare_inputs calls the donor only when they are absent. '
        'Both old covered-task columns therefore timed cached-feature selector calls. '
        'Their difference is NOT donor overhead. The flags includes_native_encoder=true '
        'and donor_plus_selector_measured=true are withdrawn for that path.', '',
        'Old JSON/CSV/PDF files are preserved byte-for-byte, including old timing numbers '
        'below. Their online-donor timing claims are superseded by this erratum. '
        'TSP/ATSP are raw-only by design; donor time is not applicable (zero executed calls).', '',
        'No corrected GPU number exists unless marked MEASURED_ONLINE. CPU smoke numbers '
        'are never reported as GPU latency. Predictive metrics, checkpoint locks, '
        'selection rules and saved predictions are unchanged. A and B are both complete; '
        "the older B/ROUND_ANALYSIS.md statement that A was still running is historical. "
        'This erratum does not re-evaluate the pretrained-vs-random quality claim.', '',
        '## Fixed Measurement Protocol',
        'Each task: original TEST rows 0..127 in order, no label/winner/cost reads; '
        'one warmup and three repetitions per scope, same per-arm VAL-locked checkpoint. '
        'A update40320: ' + PINS['A'] + '; B update54720: ' + PINS['B'] + '.', '',
        'Native donor: unmodified frozen FP32 ReLD encoder wrapper, including its '
        'validation/packing but excluding the FP16 cast. Selector: unmodified selector '
        'given resident FP16 features freshly produced by that donor, not a disk sidecar. '
        'Total: separately measured unmodified model(raw), including donor, original '
        'FP16 quantization and selector. Total is not the sum of independent means.', '',
        'Synchronized perf_counter wall time; GPU inputs already resident. Integer '
        'donor and native-encoder forward-hook counters verify each warmup/repetition '
        '(one encoder call per unique real node length); hook overhead is included. '
        'Output/equivalence checks are outside the timed span. Batch time /128 is '
        'amortized throughput cost, NOT single-query latency. ' + EXCLUDES, '',
        'Preprocessing exclusion includes CPU reconstruction of the original input '
        'fields and frozen FP64 local geometry because the old container /tmp caches '
        'are not available here. No new donor feature cache is prepared or read.', '',
        '## Preserved And Corrected Values',
        'All values below are ms/instance. PENDING means unmeasured, not zero. '
        'Do not infer an A/B speed advantage from the old hardware-dependent numbers.', '',
        '| Arm | Problem | Old nominal (cached) | Old cached | Native donor | Selector | Online total |',
        '| --- | --- | --- | --- | --- | --- | --- |']
    for row in rows:
        values = [row['arm'], row['problem']]
        values += [f"{row[key]:.6f}" if row[key] != '' else 'PENDING' for key in
                   ('legacy_nominal_ms', 'legacy_cached_ms', 'native_donor_ms',
                    'selector_only_ms', 'online_total_ms')]
        intro.append('| ' + ' | '.join(values) + ' |')
    intro += ['', '## Scope And Provenance',
              'Instrumentation: only diagnostics/r94_online_timing_recovery.py and files '
              'inside timing_correction/. Frozen r94 sources are unchanged. No training, '
              'model selection, new predictions files, Git commit or publishing. '
              'legacy_timing_preserved.json retains both complete old receipts and hashes; '
              'preservation_before.json hashes the original sources/checkpoints/reports '
              'and checks size/mtime identities of prediction arrays without rereading their payloads.',
              'B/ROUND_ANALYSIS.md is reused only as evidence for the call-path error and '
              'unchanged quality-evidence boundary. No external reviewer/model was invoked; '
              'this script makes no claim about verified gpt-6.1-sol routing.']
    (OUTPUT / 'TIMING_CORRECTION.md').write_text('\n'.join(intro) + '\n', encoding='utf-8')
    render_pdf(rows, status)
    value = dict(created_utc=utc(), status=status, gpu_measurements_available={
        arm: bool(corrected[arm]) for arm in PINS}, old_timing_valid_as_online=False,
        predictive_metrics_changed=False, original_timing_overwritten=False,
        preservation=preservation, source_analysis=str(ROOT / 'B/ROUND_ANALYSIS.md'),
        old_receipt_sha256={arm: provenance[arm]['sha256'] for arm in PINS})
    write_json('correction_status.json', value)
    return value


def render_pdf(rows, status):
    import textwrap
    os.environ['MPLCONFIGDIR'] = str(OUTPUT / 'mplconfig')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    with PdfPages(OUTPUT / 'TIMING_CORRECTION.pdf') as pdf:
        fig = plt.figure(figsize=(11.7, 8.3))
        fig.text(.06, .93, 'R94 | Online Timing Erratum', size=20, weight='bold')
        paragraphs = [
            'Status: ' + status + '. Frozen version cf2f06bc4; original timing files preserved.',
            'CORRECTION: the old nominal donor+selector input retained nco_nodes and '
            'nco_node_mask. The model skipped its donor. Both old covered-task columns '
            'are cached-feature selector measurements, not online donor measurements. '
            'Column differences are not donor overhead. Old includes_native_encoder=true '
            '/ donor_plus_selector_measured=true claims are withdrawn.',
            'Protocol: each task TEST rows0..127; 1 warmup + 3 reps per scope; same locked '
            'weights and rule. Native FP32 donor wrapper, resident-feature selector, and '
            'independently timed online total. The total includes native donor + original '
            'FP16 feature quantization + selector. No disk donor features. Native calls '
            'are asserted for every warmup/rep, with one encoder call per real node length.',
            'Inputs already on GPU. ' + EXCLUDES,
            'The old container /tmp input caches are unavailable. Original inputs and '
            'frozen FP64 geometry are reconstructed on CPU before measurement. All '
            'preprocessing/transfer is excluded. Integer counting hooks are included. '
            'Batch wall time /128 is amortized cost, not independent query latency.',
            'TSP/ATSP: raw-only, donor not applicable. PENDING is unmeasured, not zero. '
            'CPU smoke outputs are not GPU performance evidence. Both arms are complete; '
            'the older B analysis described an earlier A-running snapshot.',
            'No retraining/reselection or saved prediction changes. Quality conclusions '
            'are not recomputed. Sources: original timing receipts, final_test_lock.json, '
            'frozen r94 sources and B/ROUND_ANALYSIS.md. Full hashes and samples accompany '
            'this PDF in JSON/CSV. Only the new diagnostics script and correction directory '
            'are written; no external reviewer or verified model-routing claim.']
        y = .86
        for paragraph in paragraphs:
            wrapped = textwrap.fill(paragraph, 123)
            fig.text(.06, y, wrapped, size=10, va='top', linespacing=1.45)
            y -= .027 * (len(wrapped.splitlines()) + 1)
        pdf.savefig(fig)
        plt.close(fig)
        for arm in PINS:
            fig, ax = plt.subplots(figsize=(11.7, 8.3))
            ax.axis('off')
            fig.suptitle('R94/' + arm + ' | Old And Corrected Timing', fontsize=18, y=.95)
            cells = []
            for row in rows:
                if row['arm'] == arm:
                    cells.append([row['problem']] + [f"{row[key]:.6f}" if row[key] != '' else 'PENDING'
                        for key in ('legacy_nominal_ms', 'legacy_cached_ms', 'native_donor_ms',
                                    'selector_only_ms', 'online_total_ms')])
            table = ax.table(cellText=cells, colLabels=['Task', 'Old nominal\n(cached)', 'Old cached',
                'Native donor', 'Selector', 'Online total'], cellLoc='center', bbox=[0, .14, 1, .81])
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            fig.text(.07, .11, 'ms/instance = batch/128; fixed first128 TEST; 1 warmup + 3 reps; no I/O.', size=10)
            fig.text(.07, .075, 'TSP/ATSP donor: not applicable. Old nominal is NOT online. PENDING: not measured.', size=10)
            fig.text(.07, .04, 'Locked checkpoint SHA256: ' + PINS[arm], size=9)
            pdf.savefig(fig)
            plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('preflight', 'cpu-smoke', 'measure', 'report'))
    parser.add_argument('--arms', nargs='+', choices=tuple(PINS), default=['A', 'B'])
    parser.add_argument('--device')
    parser.add_argument('--gpu-authorized', action='store_true')
    args = parser.parse_args(argv)
    if args.command != 'measure':
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    if args.command != 'report':
        cpu_setup()
    result = dict(preflight=preflight, **{'cpu-smoke': cpu_smoke}, report=report,
                  measure=lambda: measure(args))[args.command]()
    summary = {key: value for key, value in result.items()
               if key not in ('fixed128_input_checks', 'cases')}
    if args.command == 'cpu-smoke':
        summary.update(fixed128_tasks_checked=len(result['fixed128_input_checks']),
                       official_checkpoint_cases_passed=len(result['cases']))
    print(json.dumps(summary, indent=2))
    return result


if __name__ == '__main__':
    main()

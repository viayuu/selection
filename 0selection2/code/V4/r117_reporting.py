"""Same-lock R117 LIB, honest online/cache timing and saved-only PDF/CSV reports."""

import copy
import fcntl
import math
from pathlib import Path
import time

import numpy as np
import torch

from . import r117_experiment as runner
from .r76_experiment import file_hash, read_json, utc, write_json
from .r77_data import parameter_hash
from .r117_objective import inputs_only, raw_inputs


_callbacks = runner.private_module(runner.V4 / 'r112_reporting.py', 'report_callbacks')
ARMS, DEFAULT_ROOT = runner.ARMS, runner.DEFAULT_ROOT
PROBLEMS, LIBS = runner.PROBLEMS, _callbacks.LIBS
common, shared = _callbacks.common, _callbacks.shared
write_csv, locked_inputs = _callbacks.write_csv, _callbacks.locked_inputs
nss_comparisons = _callbacks.nss_comparisons
REPETITIONS, WARMUPS = 3, 1
TIMING_SCOPES = ('native_computation_plus_selector', 'cache_hit_selector')


def device_identity(device):
    device = torch.device(device)
    if device.type != 'cuda':
        raise ValueError('R117 inference requires the launcher-assigned RTX3090')
    props = torch.cuda.get_device_properties(device)
    if '3090' not in props.name:
        raise ValueError('R117 does not authorize 4090 or other GPU types')
    uuid = getattr(props, 'uuid', None)
    if uuid is None:
        raise RuntimeError('A physical GPU UUID is required to bind TEST/LIB/timing to the same GPU')
    return dict(name=props.name, uuid=str(uuid))


def bind_device(args, device, lock):
    identity = device_identity(device)
    path = args.root / args.arm / 'inference_device.json'
    expected = dict(experiment='R117', arm=args.arm, checkpoint_sha256=lock['checkpoint_sha256'],
        state_hash=lock['state_hash'], gpu=identity)
    if path.exists():
        if read_json(path) != expected:
            raise ValueError('Locked TEST, LIB and timing must use the same physical GPU')
    else:
        write_json(path, expected)
    return identity


@torch.no_grad()
def _measure(model, batch, device):
    def sync():
        torch.cuda.synchronize(device)

    for _ in range(WARMUPS):
        model(inputs_only(batch))
    sync()
    samples = []
    for _ in range(REPETITIONS):
        sync()
        started = time.perf_counter()
        result = model(inputs_only(batch))
        sync()
        samples.append(time.perf_counter() - started)
        if not bool(torch.isfinite(result['logits'][result['solver_mask']]).all()):
            raise ValueError('Nonfinite same-model timing prediction')
    count = len(batch['node_mask'])
    return dict(batch_seconds=samples, mean_seconds_per_instance=float(np.mean(samples)) / count,
        p95_seconds_per_instance=float(np.percentile(samples, 95)) / count)


@torch.no_grad()
def measure_timing(args, device):
    lock, config = runner.authorize_test(args)
    gpu = bind_device(args, device, lock)
    run = args.root / args.arm
    saved = read_json(run / 'test_results.json')
    lib = read_json(run / 'lib_stage_complete.json')
    if (saved.get('complete') is not True or saved.get('checkpoint_sha256') != lock['checkpoint_sha256']
            or saved.get('state_hash') != lock['state_hash'] or lib.get('complete') is not True
            or lib.get('checkpoint_sha256') != lock['checkpoint_sha256'] or lib.get('gpu') != gpu):
        raise PermissionError('Timing must follow completed same-lock TEST and same-GPU LIB')
    model_hash = saved['result']['identity']['model_state_sha256']
    if (run / 'inference_timing.json').exists():
        return inference_timing(run, config, lock, model_hash)
    model, state = runner.load_checkpoint(run / 'best.pt', device)
    if parameter_hash(state['model']) != model_hash:
        raise ValueError('Timing must use the complete TEST model state')
    del state
    model.eval()
    manifest, rows = runner.ensure_test_cache(args, lock), []
    for problem in PROBLEMS:
        count = min(config['eval_batch_size'], config['microbatch_caps'][problem])
        raw = raw_inputs(runner.cached_input_batch(config, manifest, 'test', problem, device, count=count))
        # Quantization and common-base reconstruction are the model's online path,
        # not an unquantized donor shortcut. No supervision arrays are opened here.
        cached = model.prepare_inputs(raw)
        first, second = model(raw)['logits'], model(cached)['logits']
        torch.testing.assert_close(first, second, rtol=0., atol=0.)
        online, hit = _measure(model, raw, device), _measure(model, cached, device)
        rows.append(dict(problem=problem, instances=count,
            native_computation_plus_selector=online, cache_hit_selector=hit,
            standalone_model=copy.deepcopy(online), online_cache_logits_identical=True))
        del raw, cached, first, second
    if file_hash(run / 'best.pt') != lock['checkpoint_sha256'] or parameter_hash(model.state_dict()) != model_hash:
        raise ValueError('Timing must not mutate the checkpoint or any model buffers')
    value = dict(experiment='R117', arm=args.arm, complete=True, completed_utc=utc(),
        checkpoint_sha256=lock['checkpoint_sha256'], state_hash=lock['state_hash'],
        selected_update=lock['selected_update'], model_state_sha256=model_hash,
        model_factory=runner.MODEL_FACTORY, inference_rule=runner.INFERENCE_RULE,
        parameter_count=config['trainable_parameter_count'], device=str(device), gpu=gpu,
        repetitions=REPETITIONS, warmup_batches=WARMUPS, timing_scopes=list(TIMING_SCOPES),
        scope='GPU-resident original inputs: native computation plus selector; cached selector separately',
        all_current_model_included=True, end_to_end_measured=False, input_labels_read=False,
        cache_is_not_deployed_latency=True, model_reselection=False, per_problem=rows,
        excludes='dataset I/O, CPU-to-GPU transfer, training and complete solver execution')
    write_json(run / 'inference_timing.json', value)
    return inference_timing(run, config, lock, model_hash)


def inference_timing(run, config, lock, model_hash):
    value = read_json(Path(run) / 'inference_timing.json')
    expected = dict(experiment='R117', arm=config['arm'], complete=True,
        checkpoint_sha256=lock['checkpoint_sha256'], state_hash=lock['state_hash'],
        selected_update=lock['selected_update'], model_state_sha256=model_hash,
        model_factory=runner.MODEL_FACTORY, inference_rule=runner.INFERENCE_RULE,
        parameter_count=config['trainable_parameter_count'], repetitions=REPETITIONS, warmup_batches=WARMUPS,
        timing_scopes=list(TIMING_SCOPES), all_current_model_included=True,
        end_to_end_measured=False, input_labels_read=False, cache_is_not_deployed_latency=True,
        model_reselection=False)
    if any(value.get(k) != v for k, v in expected.items()):
        raise ValueError('Timing identity must match the unique whole18 checkpoint')
    bound = read_json(Path(run) / 'inference_device.json')
    if value.get('gpu') != bound.get('gpu') or bound.get('checkpoint_sha256') != lock['checkpoint_sha256']:
        raise ValueError('Timing device differs from locked inference GPU')
    rows = value.get('per_problem', [])
    if len(rows) != len(PROBLEMS) or {r['problem'] for r in rows} != set(PROBLEMS):
        raise ValueError('Timing must cover all18 tasks in both scopes')
    for row in rows:
        count = min(config['eval_batch_size'], config['microbatch_caps'][row['problem']])
        if row.get('instances') != count or row.get('online_cache_logits_identical') is not True:
            raise ValueError('Timing must use actual cap-bounded rows and online-equivalent features')
        for scope in TIMING_SCOPES:
            timing = row[scope]
            samples = timing['batch_seconds']
            values = [*samples, timing['mean_seconds_per_instance'], timing['p95_seconds_per_instance']]
            if (len(samples) != REPETITIONS or any(type(v) not in (float, int) or not math.isfinite(v) or v <= 0 for v in values)
                    or not math.isclose(timing['mean_seconds_per_instance'], float(np.mean(samples)) / count, rel_tol=1e-10)
                    or not math.isclose(timing['p95_seconds_per_instance'], float(np.percentile(samples, 95)) / count, rel_tol=1e-10)):
                raise ValueError('Invalid timing samples or per-instance denominator')
        if row.get('standalone_model') != row['native_computation_plus_selector']:
            raise ValueError('Standalone timing cannot be replaced by cache-hit latency')
    return value


_callbacks.inference_timing = inference_timing


def lib_stage(args):
    runner.protocol_lock(args)
    run, config, lock, test, _ = locked_inputs(args.root, args.arm)
    gpu = bind_device(args, torch.device(args.device), lock)
    model_hash = test['identity']['model_state_sha256']
    with (run / 'lib_stage.lock').open('a') as claim:
        fcntl.flock(claim, fcntl.LOCK_EX | fcntl.LOCK_NB)

        def phase(name, **fields):
            write_json(run / 'lib_stage_status.json', dict(experiment='R117', arm=args.arm,
                phase=name, updated_utc=utc(), **fields))

        results = common.r80_lib(args.root, run, config, lock, args.device,
            args.lib_batch_size, phase, model_hash, factory_module='code.V4.r117_experiment')
        if file_hash(run / 'best.pt') != lock['checkpoint_sha256']:
            raise ValueError('Checkpoint changed during LIB inference')
        receipt = dict(experiment='R117', arm=args.arm, complete=True, completed_utc=utc(),
            checkpoint_sha256=lock['checkpoint_sha256'], state_hash=lock['state_hash'],
            selected_update=lock['selected_update'], model_factory=runner.MODEL_FACTORY,
            model_state_sha256=model_hash, datasets=list(results), gpu=gpu)
        write_json(run / 'lib_stage_complete.json', receipt)
        phase('complete')
        return results


def report(args):
    """Read saved predictions only; rendering never invokes a model or selects weights."""
    result = _callbacks.report(args)
    run = args.root / args.arm
    config, lock = read_json(run / 'protocol.json'), read_json(run / 'final_test_lock.json')
    final, output = read_json(run / 'final_metrics.json'), run / 'full_model_statistics'
    measured = inference_timing(run, config, lock, final['test']['identity']['model_state_sha256'])
    write_csv(output / 'inference_timing.csv', [dict(problem=r['problem'], scope=s,
        instances=r['instances'], mean_milliseconds=1000 * r[s]['mean_seconds_per_instance'],
        p95_milliseconds=1000 * r[s]['p95_seconds_per_instance'])
        for r in measured['per_problem'] for s in TIMING_SCOPES])
    rows = [dict(successful_updates=r['successful_updates'], split='TRAIN optimization',
        aggregation='successful-update interval mean; not a full-split evaluation',
        **{k: r['optimization'][k] for k in runner.PART_KEYS if k in r['optimization']})
        for r in read_json(run / 'history.json')]
    write_csv(output / 'objective_diagnostics.csv', rows)
    provenance = read_json(output / 'protocol.json')
    provenance.update(objective=config['objective'], val_gate=config['val_gate'],
        frozen_donor=config['frozen_donor'], native_timing=measured,
        common_base_pretrained_in_both_arms=True, cache_is_not_deployed_latency=True,
        historical_comparison=config['historical_comparison'], native_scope=config['native_scope'])
    write_json(output / 'protocol.json', provenance)
    return result


def main(argv=None):
    parser = runner.parser()
    args = parser.parse_args(argv)
    if args.command != 'finalize':
        raise ValueError('Use reporting finalize for saved-only CPU rendering')
    args.root, args.nss_root = args.root.expanduser().resolve(), args.nss_root.expanduser().resolve()
    return report(args)


if __name__ == '__main__':
    main()

"""Recover only R112 CPU reports from the original completed GPU artifacts.

Original dispatch receipts stay failed and byte-identical. No training, model
inference, GPU dispatch, checkpoint selection or Git operation is available.
"""

import argparse
import base64
import fcntl
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import shlex
import socket
import subprocess
import sys
from datetime import datetime, timezone


V4 = Path(__file__).resolve().parent
ROOT = V4 / 'runs/R112_region_interfaces'
RECOVERY = Path('preparation/recovery_20261026')
ARMS = ('A_collapsed', 'B_interfaces')
STAGES = ('train', 'test', 'timing', 'lib', 'report')
FROZEN = ('r112_model.py', 'r112_experiment.py', 'r112_reporting.py',
          'r112_launch.py', 'run_v4_r112.sh')
PDFS = ('MODEL_STATISTICS.pdf', 'selection_distributions.pdf', 'learning_curves.pdf',
        'training_and_method_diagnostics.pdf')
REPORT_FILES = (*PDFS, 'protocol.json', 'RESULTS.md', 'metrics_by_problem.csv',
                'selection_distribution.csv', 'nss_comparison.csv', 'inference_timing.csv')
FONT = Path('/usr/share/fonts/google-noto-cjk/NotoSansCJK-Regular.ttc')
UUIDS_3090 = ('GPU-cb24f168-9f18-f0d5-6f4a-6d2406958f50',
              'GPU-cb626bdb-4319-2f5e-b91f-17fcaa44a83d')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def utc():
    return datetime.now(timezone.utc).isoformat()


def write(path, value):
    path = Path(path)
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    try:
        temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def receipt_path(root):
    return Path(root) / RECOVERY / 'recovery_complete.json'


def snapshot(paths):
    result = {}
    for path in paths:
        path = Path(path)
        if path.is_symlink() or not path.is_file():
            raise ValueError('Evidence must be an ordinary existing file: ' + str(path))
        result[str(path.resolve())] = digest(path)
    return result


def check_hashes(values):
    for path, expected in values.items():
        if Path(path).is_symlink() or digest(path) != expected:
            raise ValueError('Recovery evidence changed: ' + path)


def inside(value, run):
    path = Path(value)
    path = (path if path.is_absolute() else run / path).resolve()
    if not path.is_relative_to(run.resolve()):
        raise ValueError('Saved artifact escapes its own R112 arm: ' + str(path))
    return path


def inspect_original(root):
    """Bounded receipt/identity checks; no model imports or prediction execution."""
    root = Path(root).resolve()
    dispatch = root / 'preparation/dispatch'
    launch = read(dispatch / 'launch.json')
    if launch.get('state') != 'failed' or launch.get('exit_status') != 1:
        raise ValueError('Recovery requires the preserved failed R112 launch')
    claim = read(dispatch / 'claim.json')
    required = {str(V4 / name) for name in FROZEN} | {str(root / 'protocol_lock.json')}
    sources = claim.get('source_sha256', {})
    if not required.issubset(sources):
        raise ValueError('Missing launch-frozen R112 source proof')
    check_hashes(sources)
    inputs = [dispatch / 'claim.json', dispatch / 'launch.json', *map(Path, sources)]
    models = {}
    for arm in ARMS:
        original = dispatch / (arm + '.json')
        row = read(original)
        stages = row.get('stages', [])
        if (row.get('arm') != arm or row.get('state') != 'failed' or row.get('exit_status') != 1
                or row.get('phase') != 'report' or [s.get('name') for s in stages] != list(STAGES)
                or any(s.get('state') != 'complete' or s.get('exit_status') != 0 for s in stages[:4])
                or stages[-1].get('state') != 'failed' or stages[-1].get('exit_status') != 1):
            raise ValueError('Only a report-only failure after four rc0 stages is recoverable: ' + arm)
        log = dispatch / (arm + '_report.log')
        if 'FileNotFoundError: ' + str(FONT) not in log.read_text(encoding='utf-8'):
            raise ValueError('Original first fatal error is not the known missing-font failure: ' + arm)
        inputs.extend([original, log])
        run = root / arm
        config, lock = read(run / 'protocol.json'), read(run / 'final_test_lock.json')
        training, test = read(run / 'training_complete.json'), read(run / 'test_results.json')
        final, timing, lib = (read(run / name) for name in
                             ('final_metrics.json', 'inference_timing.json', 'lib_stage_complete.json'))
        identity = {key: lock[key] for key in ('checkpoint_sha256', 'state_hash', 'selected_update')}
        if (config.get('experiment') != 'R112' or config.get('arm') != arm
                or lock.get('experiment') != 'R112' or lock.get('arm') != arm
                or not lock.get('selected_without_test') or lock.get('per_problem_assembly') is not False
                or training.get('successful_updates') != config.get('max_successful_updates')
                or training.get('successful_updates') != 57600
                or any(item.get('complete') is not True or any(item.get(k) != v for k, v in identity.items())
                       for item in (test, final, timing, lib))
                or digest(run / 'best.pt') != identity['checkpoint_sha256']
                or inside(lock['checkpoint'], run) != run / 'best.pt'
                or not str(timing.get('device')).startswith('cuda')
                or timing.get('device_name') != 'NVIDIA GeForce RTX 4090'
                or set(lib.get('datasets', [])) != {'TSPLIB', 'CVRPLIB'}):
            raise ValueError('R112 requires immutable full training, locked TEST, GPU timing and LIB: ' + arm)
        if hashlib.sha256(json.dumps(config, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest() != lock['state_hash']:
            raise ValueError('R112 protocol differs from its original checkpoint lock')
        check_hashes({str(V4 / name): value for name, value in config['source_sha256'].items()})
        inputs.extend(V4 / name for name in config['source_sha256'])
        inputs.extend(run / name for name in ('protocol.json', 'final_test_lock.json', 'best.pt',
            'training_complete.json', 'test_results.json', 'final_metrics.json', 'inference_timing.json',
            'lib_stage_complete.json', 'initialization.json', 'history.json'))
        for split, result in (('train', final['train']), ('val', final['val']), ('test', test['result'])):
            archive = inside(result['prediction_archive'], run)
            capture = read(archive.parent / 'identity.json')
            if (capture != result.get('archive_capture_identity', result['identity'])
                    or capture.get('split') != split or capture.get('state_hash') != lock['state_hash']
                    or capture.get('successful_updates') != lock['selected_update']
                    or capture.get('model_state_sha256') != test['result']['identity']['model_state_sha256']):
                raise ValueError('Saved selected prediction identity differs: ' + arm + '/' + split)
            inputs.extend([archive, archive.parent / 'identity.json'])
        for dataset in ('TSPLIB', 'CVRPLIB'):
            inputs.extend([run / 'lib_evaluation/predictions' / (dataset + '.npz'),
                           run / 'lib_evaluation' / (dataset + '_results.json')])
        models[arm] = dict(identity, original_receipt=str(original),
            original_gpu_uuid=row['gpu_uuid'], reused_stages=list(STAGES[:4]),
            timing_device=timing['device_name'], timing_remeasured=False)
    return snapshot(inputs), models


def cpu_environment():
    repo = str(V4.parents[1])
    compat = '/public/home/shiys/easynco_v3_bridge/exports/train_100k_20261002/python_compat'
    return dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1',
        PYTHONPATH=compat + ':' + repo, MPLBACKEND='Agg', OMP_NUM_THREADS='2',
        MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', NUMEXPR_NUM_THREADS='2', WANDB_MODE='offline')


def report_command(root, arm):
    return [sys.executable, '-B', '-u', '-m', 'code.V4.r112_reporting', 'report',
            '--root', str(root), '--arm', arm, '--device', 'cpu']


def report_outputs(root, arm):
    run = Path(root) / arm
    lock = read(run / 'final_test_lock.json')
    report = read(run / 'report_complete.json')
    if (report.get('experiment') != 'R112' or report.get('arm') != arm or report.get('complete') is not True
            or any(report.get(k) != lock[k] for k in ('checkpoint_sha256', 'state_hash', 'selected_update'))):
        raise ValueError('Recovered report must bind to the unique original checkpoint: ' + arm)
    output = run / 'full_model_statistics'
    protocol = read(output / 'protocol.json')
    if protocol.get('complete') is not True or protocol.get('checkpoint_lock') != lock:
        raise ValueError('Incomplete recovered model report: ' + arm)
    for name in PDFS:
        path = output / name
        with path.open('rb') as stream:
            signature = stream.read(5)
            stream.seek(-min(path.stat().st_size, 1024), os.SEEK_END)
            trailer = stream.read()
        if signature != b'%PDF-' or b'%%EOF' not in trailer:
            raise ValueError('Incomplete recovered PDF: ' + str(path))
    return snapshot([run / 'report_complete.json', *(output / name for name in REPORT_FILES)])


def verify_receipt(root):
    """Finisher gate: failed dispatch is never rewritten or treated as rc0."""
    root = Path(root).resolve()
    path = receipt_path(root)
    value = read(path)
    if (value.get('experiment') != 'R112' or value.get('kind') != 'cpu_report_recovery'
            or value.get('state') != 'complete' or value.get('exit_status') != 0
            or value.get('gpu_compute') is not False or value.get('model_inference') is not False
            or value.get('checkpoint_reselection') is not False
            or set(value.get('arms', {})) != set(ARMS)):
        raise ValueError('Invalid R112 report recovery completion receipt')
    provenance = {}
    if value.get('generator_sha256') != digest(Path(__file__)):
        source = root / RECOVERY / 'report_generator_v1.json'
        saved = read(source)
        raw = base64.b64decode(saved['source_base64'], validate=True)
        if saved.get('sha256') != value.get('generator_sha256') or hashlib.sha256(raw).hexdigest() != saved['sha256']:
            raise ValueError('Successful report generator source snapshot differs from its original receipt')
        provenance[str(source)] = digest(source)
    inputs, models = inspect_original(root)
    if value.get('inputs') != inputs:
        raise ValueError('Original R112 inputs changed after CPU recovery')
    outputs = {}
    for arm in ARMS:
        row = value['arms'][arm]
        if (row.get('state') != 'complete' or row.get('exit_status') != 0
                or row.get('argv') != report_command(root, arm)
                or any(row.get(k) != v for k, v in models[arm].items())):
            raise ValueError('Recovery arm differs from the original frozen model: ' + arm)
        outputs.update(report_outputs(root, arm))
    if value.get('outputs') != outputs:
        raise ValueError('Recovered report output changed')
    check_hashes(value['auxiliary_sha256'])
    return {**inputs, **outputs, **value['auxiliary_sha256'], **provenance,
            str(Path(__file__)): digest(Path(__file__)), str(path): digest(path)}


def plan(root):
    inputs, models = inspect_original(root)
    return dict(experiment='R112', first_fatal_stage='report',
        root_cause='Container lacks ' + str(FONT),
        remaining=['CPU per-arm report/PDF', 'saved-prediction finisher analysis', 'explicit mutex-guarded publication'],
        gpu_compute=False, model_inference=False, checkpoint_reselection=False,
        ownership=dict(owned_files=['r112_postprocess_recovery.py', 'test_r112_postprocess_recovery.py', str(RECOVERY)],
            protected_runtime=(*FROZEN, 'r112_finish.py'), original_dispatch_preserved=True,
            authorized_gpu_job=1465, authorized_gpu_node='gpu03', authorized_gpu_uuids=UUIDS_3090,
            gpu_4090_compute_allowed=False, current_gpu_action='none; existing original timing/LIB are complete'),
        frozen_source_bytes=[dict(path=p, bytes=Path(p).stat().st_size, sha256=h)
                             for p, h in inputs.items() if p.endswith(('.py', '.sh'))],
        inputs=inputs, models=models, cpu_commands={a: shlex.join(report_command(root, a)) for a in ARMS},
        font_available=FONT.is_file(), font_path=str(FONT),
        source_adapter='none; original frozen reporter, host-installed CJK font')


def write_readiness(root, details, complete=False):
    root = Path(root).resolve()
    folder = root / RECOVERY
    folder.mkdir(parents=True, exist_ok=True)
    environment = cpu_environment()
    prefix = ['/usr/bin/env', *(key + '=' + environment[key] for key in ('CUDA_VISIBLE_DEVICES',
        'PYTHONDONTWRITEBYTECODE', 'PYTHONPATH', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'))]
    verify_argv = [*prefix, sys.executable, '-B', '-m', 'code.V4.r112_postprocess_recovery',
                   'verify', '--root', str(root), '--device', 'cpu']
    watcher_argv = [*prefix, sys.executable, '-B', '-m', 'code.V4.r112_postprocess_recovery',
                    'wait', '--root', str(root), '--publish']
    source_paths = [V4 / name for name in (*FROZEN, 'r112_postprocess_recovery.py',
                    'test_r112_postprocess_recovery.py', 'r112_finish.py')]
    value = dict(experiment='R112', updated_utc=utc(),
        state='cpu_reports_complete' if complete else 'ready_for_cpu_reports',
        authoritative_receipt=str(receipt_path(root)), gpu_dispatch_ready=False,
        gpu_stages=[], gpu_compute=False, needs_3090_queue=False,
        gpu_authorization=details['ownership'], inputs=details['inputs'],
        source_sha256=snapshot(source_paths), verify_argv=verify_argv, watcher_argv=watcher_argv,
        finish_argv=watcher_argv,
        watcher_policy='Optional, main-ready gated, after recovery verifier rc0; never rerun recovery automatically',
        recovery_attempt=str(folder / 'attempt.json'), tmux_session='r112_cpu_recovery_20261026',
        cpu_environment={k: cpu_environment()[k] for k in ('CUDA_VISIBLE_DEVICES', 'PYTHONPATH',
            'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MPLBACKEND')},
        per_arm={a: dict(reused_stages=list(STAGES[:4]), remaining_stages=[] if complete else ['report'],
            argv=report_command(root, a), device='cpu', checkpoint=details['models'][a],
            completion_receipt=str(root / a / 'report_complete.json')) for a in ARMS},
        recovery_argv=[sys.executable, '-B', '-m', 'code.V4.r112_postprocess_recovery',
                       'recover', '--root', str(root), '--device', 'cpu'],
        completion_checks=[verify_argv,
            [*prefix, sys.executable, '-B', '-m', 'code.V4.r112_postprocess_recovery', 'analyze', '--root', str(root)]],
        publication='Main finisher only, after successful analyze, under its existing global Git mutex; one attempt')
    write(folder / 'readiness.json', value)
    return value


def prepare(root):
    details = plan(root)
    folder = Path(root) / RECOVERY
    folder.mkdir(parents=True, exist_ok=True)
    complete = receipt_path(root).is_file()
    if complete:
        verify_receipt(root)
    write_readiness(root, details, complete=complete)
    return details


def private_finisher(root):
    """Source-bound private globals; the original finisher/runtime are untouched."""
    root = Path(root).resolve()
    source = V4 / 'r112_finish.py'
    raw = source.read_bytes()
    source_hash = hashlib.sha256(raw).hexdigest()
    spec = importlib.util.spec_from_file_location('code.V4._r112_recovery_finish', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    old_text = 'Both independent full18 pipelines completed train, locked TEST, timing, LIB and reports at rc0.'
    new_text = ('Original dispatch failed only at CPU PDF rendering; independent CPU recovery completed both reports. '
                'Original train, locked TEST, GPU timing and LIB were reused at rc0; failed receipts are preserved.')
    renderer = inspect.getsource(module.render)
    if renderer.count(old_text) != 1:
        raise ValueError('Frozen finisher render template changed; inspect private adapter')
    renderer = renderer.replace(old_text, new_text)
    exec(compile(renderer, str(source) + ':R112_RECOVERY_PRIVATE', 'exec'), module.__dict__)
    folder = root / RECOVERY
    folder.mkdir(parents=True, exist_ok=True)
    adapter = folder / 'finish_adapter.json'
    extra_code = [V4 / n for n in ('r112_postprocess_recovery.py', 'test_r112_postprocess_recovery.py')]
    extra_metadata = [folder / name for name in ('README.md', 'plan.json', 'attempt.json', 'readiness.json',
        'recovery_complete.json', 'A_collapsed_report.log', 'B_interfaces_report.log', 'finish_adapter.json')]
    if (folder / 'report_generator_v1.json').is_file():
        extra_metadata.append(folder / 'report_generator_v1.json')
    evidence = dict(original_finisher=str(source), original_sha256=source_hash,
        original_source_base64=base64.b64encode(raw).decode('ascii'),
        recovery_generator_sha256=digest(Path(__file__)),
        bindings=dict(completed='verify preserved failed dispatch plus independent CPU recovery receipt',
            publication_paths='original exact allowlist plus only new recovery code/test/metadata',
            analyze='reuse a hash-verified finish_complete receipt instead of reanalyzing'),
        render_replacement=dict(old=old_text, new=new_text,
            adapted_function_sha256=hashlib.sha256(renderer.encode()).hexdigest()),
        publication_append=[str(p) for p in (*extra_code, *extra_metadata)],
        disk_runtime_edited=False, gpu_compute=False)
    if adapter.exists():
        if read(adapter) != evidence:
            raise ValueError('Private finisher adapter evidence changed; no implicit refreeze')
    else:
        write(adapter, evidence)

    def completed(value):
        if not receipt_path(value).is_file():
            raise module.NotReady('Awaiting independent R112 recovery receipt; original dispatch stays failed')
        inputs = verify_receipt(value)
        if digest(source) != source_hash or read(adapter) != evidence:
            raise ValueError('Private finisher source/adapter changed')
        return {**inputs, str(source): source_hash, str(adapter): digest(adapter)}

    original_paths = module.publication_paths

    def publication_paths(value):
        included, excluded = original_paths(value)
        for path in (*extra_code, *extra_metadata):
            if (path.is_symlink() or not path.is_file()
                    or not path.resolve().is_relative_to(module.GIT_ROOT.resolve())
                    or path.stat().st_size > module.SMALL_LIMIT):
                raise ValueError('Invalid recovery-only publication path: ' + str(path))
            if path not in included:
                included.append(path)
        if sum(p.stat().st_size for p in included) > module.TOTAL_LIMIT:
            raise ValueError('Recovery publication exceeds the original total limit')
        return included, excluded

    original_analyze = module.analyze

    def analyze(value, references=None):
        with (folder / 'finisher.lock').open('a') as claim:
            fcntl.flock(claim, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if (Path(value) / 'finish_complete.json').is_file():
                return module.verify_delivery(value)
            return original_analyze(value, references)

    module.completed, module.publication_paths, module.analyze = completed, publication_paths, analyze
    if digest(source) != source_hash:
        raise ValueError('Original finisher changed while loading its private adapter')
    return module


def recover(root):
    root = Path(root).resolve()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError("CPU recovery requires explicit CUDA_VISIBLE_DEVICES=''")
    folder = root / RECOVERY
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / 'recovery.lock').open('a') as claim:
        fcntl.flock(claim, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if receipt_path(root).exists():
            verify_receipt(root)
            return read(receipt_path(root))
        details = plan(root)
        if not details['font_available']:
            raise FileNotFoundError(FONT)
        attempt = folder / 'attempt.json'
        if attempt.exists():
            raise RuntimeError('Recovery already attempted; inspect attempt.json, no automatic retry')
        write(folder / 'plan.json', details)
        write_readiness(root, details)
        value = dict(experiment='R112', kind='cpu_report_recovery', state='running', started_utc=utc(),
            pid=os.getpid(), host=socket.gethostname(), generator_sha256=digest(Path(__file__)),
            gpu_compute=False, model_inference=False, checkpoint_reselection=False,
            original_dispatch_state='failed', original_dispatch_preserved=True, automatic_retry=False,
            font_path=str(FONT), font_sha256=digest(FONT), source_adapter='none',
            inputs=details['inputs'], arms={}, outputs={})
        write(attempt, value)
        try:
            for arm in ARMS:
                argv, log = report_command(root, arm), folder / (arm + '_report.log')
                row = dict(details['models'][arm], state='running', started_utc=utc(),
                           argv=argv, log=str(log), cuda_visible_devices='')
                value['arms'][arm] = row
                write(attempt, value)
                with log.open('x', encoding='utf-8') as stream:
                    print(shlex.join(argv), file=stream, flush=True)
                    result = subprocess.run(argv, cwd=V4.parents[1], env=cpu_environment(),
                                            stdout=stream, stderr=subprocess.STDOUT, timeout=1800)
                row.update(exit_status=result.returncode, finished_utc=utc(),
                           state='complete' if result.returncode == 0 else 'failed')
                write(attempt, value)
                if result.returncode:
                    raise RuntimeError('CPU report failed; inspect ' + str(log))
                value['outputs'].update(report_outputs(root, arm))
            check_hashes(value['inputs'])
            value.update(state='complete', exit_status=0, completed_utc=utc(),
                auxiliary_sha256=snapshot([folder / 'plan.json', FONT,
                    *(folder / (a + '_report.log') for a in ARMS)]))
            write(attempt, value)
            write(receipt_path(root), value)
            verify_receipt(root)
            write_readiness(root, details, complete=True)
            return value
        except BaseException as error:
            value.update(state='failed', exit_status=1, error_type=type(error).__name__,
                         error=str(error), failed_utc=utc())
            write(attempt, value)
            receipt_path(root).unlink(missing_ok=True)
            raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('plan', 'prepare', 'recover', 'verify', 'analyze', 'wait', 'publish'))
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--device', choices=('cpu',), default='cpu')
    parser.add_argument('--publish', action='store_true', help='Only with wait: one main-authorized mutex-guarded publication')
    parser.add_argument('--poll-seconds', type=float, default=60.)
    parser.add_argument('--timeout-hours', type=float, default=72.)
    args = parser.parse_args(argv)
    root = args.root.expanduser().resolve()
    if args.publish and args.command != 'wait':
        parser.error('--publish is wait-only; otherwise use the explicit publish command')
    if args.command in ('analyze', 'wait', 'publish'):
        if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
            parser.error("CPU finisher requires explicit CUDA_VISIBLE_DEVICES=''")
        module = private_finisher(root)
        result = (module.analyze(root) if args.command == 'analyze' else module.publish(root)
                  if args.command == 'publish' else module.wait(root, timeout_hours=args.timeout_hours,
                      poll_seconds=args.poll_seconds, publish_after=args.publish))
    else:
        result = (plan(root) if args.command == 'plan' else prepare(root) if args.command == 'prepare'
                  else recover(root) if args.command == 'recover' else verify_receipt(root))
    if args.command in ('plan', 'prepare'):
        print(json.dumps(result, indent=2))
    else:
        print(json.dumps(dict(command=args.command, complete=True, gpu_compute=False,
                              receipt=str(receipt_path(root))), indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

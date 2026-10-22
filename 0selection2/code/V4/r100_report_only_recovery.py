"""Explicit, single-attempt CPU recovery of the completed R100 report."""

import argparse
import copy
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / 'code/V4/runs/R100_pairwise_context'
LAUNCH = REPO / 'code/V4/runs/R98_parallel_dispatch'
RECOVERY = ROOT / 'preparation/report_only_recovery/20261022_cpu_render_01'
SOURCE = REPO / 'code/V4/r100_reporting.py'
SCOPE = 'manual explicit post-failure CPU render recovery; no inference, reselection or automatic retry'


def utc():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def identity(path):
    before = path.stat()
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    after = path.stat()
    fields = ('st_size', 'st_mtime_ns', 'st_ctime_ns', 'st_ino', 'st_dev', 'st_mode')
    if any(getattr(before, field) != getattr(after, field) for field in fields):
        raise RuntimeError('File changed while hashing: ' + str(path))
    return dict(sha256=digest.hexdigest(), **{field: getattr(after, field) for field in fields})


def csv_counts(directory):
    counts = {}
    for path in sorted(directory.glob('*.csv')):
        with path.open(newline='') as stream:
            counts[path.name] = sum(1 for _ in csv.DictReader(stream))
    return counts


def ready_plans():
    result = {}
    for arm in ('R101_B', 'R99_A', 'R99_B'):
        path = LAUNCH / arm / 'plan.json'
        keys = list(read(path)['source_sha256'])
        if any(Path(key).name == SOURCE.name for key in keys):
            raise RuntimeError('STOP: ' + arm + ' pins the reporting source')
        result[arm] = dict(plan_sha256=identity(path)['sha256'], source_keys=keys,
                           includes_r100_reporting=False)
    return result


def prepare():
    target = RECOVERY / 'before.json'
    if target.exists():
        raise RuntimeError('Preparation already recorded; do not overwrite')
    original = RECOVERY / 'original'
    if SOURCE.read_bytes() != (original / SOURCE.name).read_bytes():
        raise RuntimeError('Original source must be snapshotted before applying the fix')
    controller = read(original / 'controller.json')
    if controller != read(LAUNCH / 'R100_A/controller.json'):
        raise RuntimeError('Canonical controller changed since failure snapshot')
    if controller['state'] != 'failed' or controller['stages'][0]['exit_status'] != 0:
        raise RuntimeError('Require original successful GPU stage and failed CPU stage')
    if read(original / 'goal_watch_receipt.json')['state'] != 'stopped_on_error':
        raise RuntimeError('Require preserved stopped goal watcher')
    plans = ready_plans()
    names = ('protocol.json', 'protocol_lock.json', 'prepare_identity.json', 'checkpoint.json',
             'final_test_lock.json', 'test_input_lock.json', 'val_results.json', 'test_results.json',
             'lib_results.json', 'run_complete.json', 'timing.json', 'resource_witness.json',
             'wandb_receipt.json', 'status.json')
    paths = [ROOT / name for name in names]
    paths += list(ROOT.glob('evaluation/*/selected_best/*.npz'))
    paths += list((ROOT / 'lib_evaluation/predictions').glob('*.npz'))
    paths += list((ROOT / 'lib_input_locks').glob('*.json'))
    paths += [ROOT / 'goal_watch_receipt.json', LAUNCH / 'R100_A/worker.json',
              LAUNCH / 'R100_A/plan.json', LAUNCH / 'R100_A/ready.json', LAUNCH / 'R100_A/PDF.log']
    for arm in plans:
        paths += [LAUNCH / arm / name for name in ('plan.json', 'ready.json')]
    for key in read(ROOT / 'protocol.json')['source_sha256']:
        path = Path(key)
        if not path.is_absolute():
            path = REPO / path if (REPO / path).is_file() else SOURCE.parent / path.name
        if path != SOURCE:
            paths.append(path)
    protected = {str(path): identity(path) for path in sorted(set(paths))}
    snapshots = {str(path): identity(path) for path in sorted(original.rglob('*')) if path.is_file()}
    write(target, dict(scope=SCOPE, prepared_utc=utc(), protected=protected, snapshots=snapshots,
                       ready_plans=plans, original_reporting_sha256=identity(SOURCE)['sha256'],
                       csv_rows=csv_counts(ROOT / 'full_model_statistics'),
                       lock=read(ROOT / 'final_test_lock.json')))
    target.chmod(0o444)
    write(RECOVERY / 'failure_summary_redacted.json', dict(experiment='R100',
        scope=SCOPE, original_GPU_exit_status=0, original_PDF_exit_status=1,
        error=read(original / 'postprocess_status.json')['error'],
        original_reporting_sha256=identity(SOURCE)['sha256'],
        original_attempt_preserved=True, stopped_goalwatch_preserved=True,
        active_ready_plans_exclude_reporting=True, automatic_retry=False,
        sensitive_argv_and_tokens_omitted=True))
    print(json.dumps(dict(prepared=True, protected_files=len(protected), recovery=str(RECOVERY))))


def recover():
    before = read(RECOVERY / 'before.json')
    ready_plans()
    for path, expected in {**before['protected'], **before['snapshots']}.items():
        if identity(Path(path)) != expected:
            raise RuntimeError('Pre-recovery identity changed: ' + path)
    original = RECOVERY / 'original'
    expected_source = (original / SOURCE.name).read_text().replace(
        "*['- '+note for note in NOTES], '', previous._md_table(keys, [[r[key] for key in keys] for r in metrics]), '',",
        "*['- '+note for note in NOTES], '',\n                *previous._md_table(keys, [[r[key] for key in keys] for r in metrics]), '',")
    if SOURCE.read_text() != expected_source or expected_source == (original / SOURCE.name).read_text():
        raise RuntimeError('Recovery permits only the two-line Markdown list expansion')
    controller_path = LAUNCH / 'R100_A/controller.json'
    controller = read(controller_path)
    if controller != read(original / 'controller.json'):
        raise RuntimeError('Controller changed; recovery will not overwrite it')
    argv = [sys.executable, '-B', '-u', '-m', 'code.V4.r100_experiment', 'report',
            '--root', str(ROOT), '--device', 'cpu']
    started = utc()
    log = RECOVERY / 'CPU_report.log'
    # Exclusive creation consumes the single attempt even if interrupted or failed.
    with (RECOVERY / 'attempt.json').open('x') as marker:
        json.dump(dict(scope=SCOPE, started_utc=started, argv=argv, automatic_retry=False,
                       reporting_sha256=identity(SOURCE)['sha256']), marker, indent=2)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1',
               MPLBACKEND='Agg', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    with log.open('x') as stream:
        process = subprocess.Popen(argv, cwd=REPO, env=env, stdout=stream, stderr=subprocess.STDOUT)
        stage = dict(name='PDF', device='cpu', argv=argv, started_utc=started, pid=process.pid,
                     state='running', log=str(log), recovery_scope=SCOPE, automatic_retry=False)
        write(RECOVERY / 'cpu_stage.json', stage)
        rc = process.wait()
    stage.update(exit_status=rc, finished_utc=utc(), state='complete' if rc == 0 else 'failed')
    write(RECOVERY / 'cpu_stage.json', stage)
    after = {path: identity(Path(path)) for path in before['protected']}
    unchanged = after == before['protected']
    snapshots_unchanged = all(identity(Path(path)) == expected for path, expected in before['snapshots'].items())
    rows = csv_counts(ROOT / 'full_model_statistics')
    write(RECOVERY / 'after.json', dict(protected=after, unchanged=unchanged,
        snapshots_unchanged=snapshots_unchanged, csv_rows=rows, csv_rows_unchanged=rows == before['csv_rows']))
    if rc != 0 or not unchanged or not snapshots_unchanged or rows != before['csv_rows']:
        raise RuntimeError('CPU report failed or invariant changed; no retry and no controller advancement')
    post = read(ROOT / 'postprocess_status.json')
    lock = before['lock']
    if (post.get('complete') is not True or post.get('phase') != 'complete'
            or post.get('checkpoint_sha256') != lock['checkpoint_sha256']
            or post.get('state_hash') != lock['state_hash']):
        raise RuntimeError('Report did not complete with the same frozen bank')
    pdfs = {str(path): identity(path) for path in
            [ROOT / 'RESULTS.pdf', *sorted((ROOT / 'full_model_statistics').glob('*.pdf'))]}
    if len(pdfs) != 4 or any(Path(path).read_bytes()[:5] != b'%PDF-' for path in pdfs):
        raise RuntimeError('Expected all four completed PDF artifacts')
    receipt = dict(experiment='R100', complete=True, phase='complete', exit_status=0,
        scope=SCOPE, manual_explicit_recovery=True, original_attempt_succeeded=False,
        inference_performed=False, reselection_performed=False, automatic_retry=False,
        started_utc=started, completed_utc=utc(), cpu_stage=stage, PDFs=pdfs,
        checkpoint_sha256=lock['checkpoint_sha256'], state_hash=lock['state_hash'],
        original_reporting_sha256=before['original_reporting_sha256'],
        recovery_reporting_sha256=identity(SOURCE)['sha256'],
        historical_R100_ready_plan_unchanged=True, protected_metadata_and_sha256_unchanged=True,
        csv_rows_unchanged=True, csv_rows=rows, immutable_snapshots_unchanged=True,
        stopped_goalwatch_preserved=True, no_Git_or_R101_start=True)
    write(RECOVERY / 'recovery_complete.json', receipt)
    if read(controller_path) != controller:
        raise RuntimeError('Concurrent controller change; do not advance')
    recovered = copy.deepcopy(controller)
    old_pdf = next(s for s in controller['stages'] if s['name'] == 'PDF')
    stage['original_attempt'] = copy.deepcopy(old_pdf)
    stage['original_attempt_snapshot'] = str(original / 'controller.json')
    recovered['stages'] = [stage if s['name'] == 'PDF' else s for s in recovered['stages']]
    recovered['original_error'] = recovered.pop('error')
    recovered.update(state='complete', exit_status=0, finished_utc=stage['finished_utc'],
                     recovery_scope=SCOPE, recovery_receipt=str(RECOVERY / 'recovery_complete.json'))
    write(controller_path, recovered)
    print(json.dumps(dict(complete=True, cpu_pid=process.pid, exit_status=rc,
        checkpoint_sha256=lock['checkpoint_sha256'], recovery=str(RECOVERY))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'recover'))
    args = parser.parse_args()
    prepare() if args.action == 'prepare' else recover()

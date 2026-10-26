"""Quiet, restart-safe followthrough of the 2026-10-25 research experiments.

Polling reads saved receipts only. Model execution is opt-in, serialized, and
claimed on disk before launch. Historical interrupted GPU receipts are not an
authority for the migrated two-GPU queue. No model/GPU libraries are imported.
"""

import argparse
from contextlib import contextmanager
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import sys
import threading


REPO = Path(__file__).resolve().parents[2]
RUN_REL = Path('runs/research_followthrough_20261026')
RESEARCH_REL = Path('docs/research/post_major_paradigm_results_20261026')
CODEX = '/public/home/shiys/.local/bin/codex'
POLL_SECONDS = 300
SOL, ASTRA = 'gpt-6.1-sol', 'gpt-6-astra'
UUIDS = ('GPU-cb24f168-9f18-f0d5-6f4a-6d2406958f50',
         'GPU-cb626bdb-4319-2f5e-b91f-17fcaa44a83d')
DATA_VERSION = 'selection_data_new90k_B_filtered_single_reld_mt_seed2_20261018'
ARMS = {'R109': ('A_tied_late', 'B_private_late'),
        'R112': ('A_collapsed', 'B_interfaces'),
        'R113': ('A_scratch', 'B_snapshot', 'C_transition')}
ROOTS = {'R109': 'R109_method_conditioned_nodes', 'R112': 'R112_region_interfaces',
         'R113': 'R113_executable_process', 'R114': 'R114_paired_responses'}
PROBLEMS = ('TSP', 'CVRP', 'VRPB', 'VRPL', 'VRPTW', 'VRPBL', 'VRPBTW',
            'VRPLTW', 'VRPBLTW', 'OVRP', 'OVRPB', 'OVRPL', 'OVRPTW',
            'OVRPBL', 'OVRPBTW', 'OVRPLTW', 'OVRPBLTW', 'ATSP')
METRICS = ('top1_pct', 'mean_cost', 'sbs_cost', 'oracle_cost', 'vs_sbs_pct',
           'vs_oracle_pct', 'actual_regret_pct')
MODULE = 'code.V4.research_completion_monitor'
QUEUE_MODULE = 'code.V4.r113_3090_queue'
SHA = re.compile(r'[0-9a-f]{64}')
COMMIT = re.compile(r'[0-9a-f]{40,64}')
MAX_RECOVERIES = 4
MAX_JSON_BYTES = 8 * 1024 * 1024
MODEL_TIMEOUT_SECONDS = 6 * 60 * 60


class NeedAction(ValueError):
    pass


def utc():
    return datetime.now(timezone.utc).isoformat()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def json_hash(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def read(path, *, missing=None):
    path = Path(path)
    if not path.exists():
        return {} if missing is None else missing
    if path.is_symlink() or path.stat().st_size > MAX_JSON_BYTES:
        raise NeedAction('Not an ordinary bounded JSON receipt: ' + str(path))
    try:
        result = json.loads(path.read_text(encoding='utf-8'))
    except (ValueError, OSError) as error:
        raise NeedAction('Unreadable receipt ' + str(path) + ': ' + str(error)) from error
    if not isinstance(result, dict):
        raise NeedAction('Receipt must be an object: ' + str(path))
    return result


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name('.' + path.name + '.' + str(os.getpid()) + '.tmp')
    try:
        with temporary.open('w', encoding='utf-8') as stream:
            stream.write(json.dumps(value, indent=2, allow_nan=False) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
        descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def lock(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield stream


def inside(path, root):
    path, root = Path(path), Path(root).resolve()
    if not path.is_absolute():
        path = root / path
    if not path.resolve().is_relative_to(root):
        raise NeedAction('Receipt path escapes its owner scope: ' + str(path))
    if path.is_symlink() or any(p.is_symlink() for p in path.parents if p != root.parent):
        raise NeedAction('Symlink is not completion evidence: ' + str(path))
    return path.resolve()


def pdf_ready(path):
    path = Path(path)
    if not path.is_file() or path.is_symlink() or path.stat().st_size < 12:
        return False
    with path.open('rb') as stream:
        start = stream.read(5)
        stream.seek(max(0, path.stat().st_size - 1024))
        return start == b'%PDF-' and b'%%EOF' in stream.read()


def process_identity(pid):
    """Start ticks plus boot ID protect adoption/signals against PID reuse."""
    try:
        directory = Path('/proc') / str(int(pid))
        fields = (directory / 'stat').read_text().rsplit(')', 1)[1].split()
        if fields[0] == 'Z':
            return None
        return dict(pid=int(pid), start_ticks=fields[19],
                    boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                    argv=(directory / 'cmdline').read_bytes().decode(errors='replace').split('\0')[:-1])
    except (OSError, ValueError, IndexError, TypeError):
        return None


def alive(identity):
    current = process_identity(identity.get('pid')) if identity else None
    return bool(current and all(current.get(k) == identity.get(k)
                                for k in ('pid', 'start_ticks', 'boot_id', 'argv')))


def processes():
    for directory in Path('/proc').iterdir():
        if directory.name.isdigit():
            identity = process_identity(directory.name)
            if identity:
                yield identity


def module_running(module):
    script = module.replace('.', '/') + '.py'
    return any(any(argv[i:i + 2] == ['-m', module] for i in range(len(argv) - 1))
               or any(arg.endswith('/' + script) for arg in argv)
               for argv in (row['argv'] for row in processes()))


def path_running(path):
    return any(str(path) in row['argv'] for row in processes())


def published(value):
    return (value.get('state') == 'complete' and value.get('exit_status', 0) == 0
            and bool(COMMIT.fullmatch(str(value.get('commit', '')))))


def failure_signature(failure):
    """Do not turn changing timestamps, heartbeats or attempt numbers into retries."""
    if isinstance(failure, str):
        failure = dict(message=failure)
    result = {k: failure[k] for k in ('kind', 'code', 'stage', 'job', 'message', 'reason') if k in failure}
    for key in ('message', 'reason'):
        if key in result:
            message = re.sub(r'\b\d{4}-\d\d-\d\d[T ][0-9:.+Z-]+', '<time>', str(result[key]))
            result[key] = re.sub(r'(?:attempt[_ =-]*|pid[ =:]*)\d+', '<attempt/pid>', message,
                                 flags=re.IGNORECASE)
    if not result:
        raise NeedAction('New authoritative queue failure needs a stable first-fatal description')
    return json_hash(result)


class Sources:
    def __init__(self, repo, handoff):
        self.repo, self.handoff = Path(repo), Path(handoff)
        self.paths = set()

    def root(self, version):
        return self.repo / 'code/V4/runs' / ROOTS[version]

    def receipt(self, path):
        path = inside(path, self.repo)
        value = read(path)
        if value:
            self.paths.add(path)
        return value

    def queue(self):
        handoff = self.receipt(self.handoff)
        if handoff.get('ready') is not True:
            return dict(state='awaiting_handoff', handoff=handoff)
        if (handoff.get('owner') != 'Mencius' or handoff.get('queue_module') != QUEUE_MODULE
                or handoff.get('host') != 'gpu03' or handoff.get('allocation_job') != 1465
                or set(handoff.get('gpu_uuids', [])) != set(UUIDS)
                or len(handoff.get('gpu_uuids', [])) != 2):
            raise NeedAction('Queue handoff does not authorize exactly gpu03/job1465 and the two RTX3090 UUIDs')
        for arm in ('B_snapshot', 'C_transition'):
            resume = handoff.get('resumes', {}).get(arm, {})
            if (resume.get('kind') != 'resource_checkpoint_resume' or resume.get('update') != 20160
                    or resume.get('optimizer_retained') is not True or resume.get('rng_retained') is not True):
                raise NeedAction('B/C need documented update20160 optimizer/RNG-preserving resource resumes')
        if handoff.get('resumes', {}).get('A_scratch', {}).get('kind') != 'fresh':
            raise NeedAction('A_scratch must be a new fresh run, never a substituted trained initializer')
        owner_ready = self.receipt(inside(handoff['owner_ready_path'], self.root('R113')))
        if owner_ready.get('main_approved') is not True:
            return dict(state='waiting_main', handoff=handoff)
        state_path = inside(handoff['queue_state_path'], self.root('R113'))
        value = self.receipt(state_path)
        state = value.get(handoff.get('state_field', 'state'), 'awaiting_state')
        jobs = value.get('jobs', {})
        terminal = {'complete', 'blocked', 'failed'}
        models_complete = all(jobs.get(a, {}).get('state') == 'complete'
                              and jobs[a].get('exit_status') == 0 for a in ARMS['R113'])
        # A blocked acquisition pilot is a valid terminal outcome, not a failed NN arm.
        complete = (state in ('complete', 'blocked', 'failed') and models_complete
                    and jobs.get('R114_labels', {}).get('state') in terminal)
        failed_models = {a: jobs[a] for a in ARMS['R113'] if jobs.get(a, {}).get('state') in ('failed', 'blocked')}
        failed = bool(failed_models) or (not complete and state in handoff.get('failure_states', ['failed', 'RESOURCEERROR', 'resource_error']))
        failure = value.get('failure') or value.get('error') or {}
        if failed_models:
            arm, row = next(iter(failed_models.items()))
            failure = dict(job=arm, stage=row.get('stage', 'new_queue_execution'),
                           code=row.get('exit_status'), reason=row.get('reason') or row.get('error') or 'New model job failed')
        return dict(state=state, complete=complete, failed=failed, failure=failure,
                    path=str(state_path), value=value, handoff=handoff)

    def r109_report_failure(self):
        """Only actual rc0 compute + failed PDF permits a CPU-only recovery call."""
        root, originals, failed = self.root('R109'), {}, False
        for arm in ARMS['R109']:
            folder = root / 'preparation/dispatch' / ('R109_' + arm)
            worker = self.receipt(folder / 'worker.json')
            controller = self.receipt(folder / 'controller.json')
            stages = {s.get('name'): s for s in worker.get('stages', [])}
            if not all(stages.get(n, {}).get('state') == 'complete' and stages[n].get('exit_status') == 0
                       for n in ('train', 'TEST', 'timing', 'LIB')):
                return None
            pdf = next((s for s in controller.get('stages', []) if s.get('name') == 'PDF'), {})
            failed |= pdf.get('state') == 'failed' or pdf.get('exit_status') not in (None, 0)
            for name in ('worker.json', 'controller.json'):
                path = folder / name
                originals[str(path)] = digest(path)
        return originals if failed else None

    def models(self, version):
        """Small saved-stage gate; the one-shot Sol pass authenticates full reports."""
        root, locks = self.root(version), {}
        for arm in ARMS[version]:
            run = root / arm
            config = self.receipt(run / 'protocol.json')
            final_lock = self.receipt(run / 'final_test_lock.json')
            train = self.receipt(run / 'training_complete.json')
            test = self.receipt(run / 'test_results.json')
            lib = self.receipt(run / 'lib_stage_complete.json')
            report = self.receipt(run / ('postprocess_status.json' if version == 'R109' else 'report_complete.json'))
            protocol = self.receipt(run / 'full_model_statistics/protocol.json')
            maximum = config.get('max_successful_updates')
            if not (SHA.fullmatch(str(final_lock.get('checkpoint_sha256', '')))
                    and SHA.fullmatch(str(final_lock.get('state_hash', '')))
                    and final_lock.get('selection_split', 'val').lower() == 'val'
                    and final_lock.get('selected_without_test', True) is True
                    and final_lock.get('per_problem_assembly', False) is False
                    and type(final_lock.get('selected_update')) is int and final_lock['selected_update'] > 0
                    and 'val' in str(final_lock.get('rule', '')).lower()
                    and bool(config.get('inference_rule'))
                    and config['inference_rule'] == final_lock.get('inference_rule')
                    and type(maximum) is int and maximum > 0 and train.get('successful_updates') == maximum
                    and test.get('complete') is True
                    and test.get('checkpoint_sha256') == final_lock['checkpoint_sha256']
                    and lib.get('complete') is True and lib.get('checkpoint_sha256') == final_lock['checkpoint_sha256']
                    and report.get('complete') is True and protocol.get('complete') is True
                    and protocol.get('checkpoint_lock') == final_lock):
                return None
            for name in ('MODEL_STATISTICS.pdf', 'selection_distributions.pdf'):
                path = inside(run / 'full_model_statistics' / name, self.repo)
                if not pdf_ready(path):
                    return None
                self.paths.add(path)
            locks[arm] = final_lock
        return locks

    def experiment(self, version, queue):
        root = self.root(version)
        if version == 'R109':
            progress = self.receipt(root / 'goal_progress.json')
            watcher = self.receipt(root / 'goal_watch_receipt.json')
            publication = self.receipt(root / 'github_publication.json')
            if not published(publication):
                # A later additive publication does not rewrite the failed watcher.
                recovery_dir = root / 'preparation/publication_recovery_20261026'
                recovered_publication = self.receipt(recovery_dir / 'publication_receipt.json')
                verification = self.receipt(recovery_dir / 'final_verification.json')
                if (published(recovered_publication)
                        and recovered_publication.get('kind') == 'explicit_additive_publication_recovery'
                        and recovered_publication.get('original_failure_preserved') is True
                        and recovered_publication.get('canonical_main_remote_verified') is True
                        and recovered_publication.get('unrelated_changes_staged') is False
                        and verification.get('state') == 'complete'
                        and verification.get('publication_receipt_sha256')
                        == digest(recovery_dir / 'publication_receipt.json')
                        and verification.get('actual_commit') == recovered_publication['commit']
                        == verification.get('actual_origin_main')
                        == recovered_publication.get('remote_after')):
                    publication = recovered_publication
            ready = progress.get('complete') is True and progress.get('comparison_complete') is True
            recovery_root = self.repo / RUN_REL / 'recoveries/R109'
            recovery = self.receipt(recovery_root / 'recovery_complete.json')
            recovered = (recovery.get('complete') is True and recovery.get('experiment') == 'R109'
                         and recovery.get('kind') == 'cpu_report_recovery' and recovery.get('state') == 'complete'
                         and recovery.get('exit_status') == 0 and all(recovery.get(k) is False for k in
                                ('gpu_compute', 'model_inference', 'checkpoint_reselection')))
            if recovered:
                originals = self.r109_report_failure()
                if not originals or recovery.get('original_receipts') != originals:
                    raise NeedAction('R109 CPU recovery must preserve actual failed PDF and successful compute receipts')
                ready = True
                publication = self.receipt(recovery_root / 'github_publication.json')
            elif not ready:
                originals = self.r109_report_failure()
                if originals:
                    return dict(state='report_only_failure', reason='R109 rc0 compute; PDF-only failure permits CPU saved-report recovery',
                                signature=json_hash(originals), original_receipts=originals)
                if progress.get('phase') == 'failed' or watcher.get('state') in ('failed', 'stopped_on_error'):
                    return dict(state='needs_action', reason='Current R109 watcher stopped without a proved report-only recoverable failure')
            if progress.get('errors'):
                if not recovered:
                    return dict(state='needs_action', reason='Existing R109 goal reports an error; inspect its current owner',
                                failure=progress['errors'])
        else:
            finish = self.receipt(root / 'finish_complete.json')
            publication = self.receipt(root / 'publish_receipt.json')
            ready = finish.get('complete') is True and finish.get('analysis_rc') == 0
            if version == 'R112':
                recovery = self.receipt(root / 'preparation/recovery_20261026/recovery_complete.json')
                # Never read failed original PDF dispatch as current failure evidence.
                recovered = (recovery.get('experiment') == 'R112' and recovery.get('kind') == 'cpu_report_recovery'
                             and recovery.get('state') == 'complete' and recovery.get('exit_status') == 0
                             and all(recovery.get(k) is False for k in
                                     ('gpu_compute', 'model_inference', 'checkpoint_reselection')))
                if not recovered:
                    return dict(state='waiting_owner', reason='Await Turing authentic CPU recovery; snapshots are not a complete version')
                if not ready:
                    return dict(state='closeout', reason='Recovered R112 needs saved-only analysis/publication')
            elif queue.get('failed'):
                signature = failure_signature(queue.get('failure', {}))
                return dict(state='queue_failure', reason='New authoritative two-GPU queue stage failed',
                            failure=queue['failure'], signature=signature)
            elif not queue.get('complete'):
                return dict(state='waiting_queue', reason='Await new two-GPU queue, not the interrupted four-slot controller')
        if not ready:
            return dict(state='closeout' if version == 'R113' else 'running', reason='Await saved version closeout')
        locks = self.models(version)
        if not locks:
            return dict(state='needs_action', reason='Completion receipt lacks same-VAL-lock full saved TEST/LIB/PDF stages')
        if version == 'R109' and recovered and recovery.get('checkpoints') != locks:
            raise NeedAction('R109 recovered reports must match both original VAL locks')
        if not published(publication):
            if publication and publication.get('state') not in ('complete', 'prepared', 'running'):
                return dict(state='needs_action', reason='Uncertain/failed publication; reconcile commit/push, never blindly retry')
            return dict(state='closeout', reason='Saved complete version awaits existing mutex-guarded publication')
        if version != 'R109' and finish.get('checkpoints') != locks:
            return dict(state='needs_action', reason='Finisher checkpoints differ from unique whole-model VAL locks')
        return dict(state='complete', locks=locks, publication=publication)

    def pilot(self, queue):
        root = self.root('R114')
        required_finish = queue.get('handoff', {}).get('r114_required_finish_dir')
        if required_finish:
            required_finish = inside(required_finish, root / 'preparation/finish')
            if not (required_finish / 'finish_complete.json').is_file():
                return dict(state='waiting_owner', reason='Await actual corrected CVRP acquisition and combined R114 closeout; no native-paradigm conclusion from the v1 interface failure')
        progress_path = queue.get('handoff', {}).get('r114_progress_path')
        if not progress_path:
            return dict(state='waiting_queue', reason='Await exact R114 authoritative progress path in queue handoff')
        progress_path = inside(progress_path, root)
        progress = self.receipt(progress_path)
        terminal = progress.get('status') in ('complete', 'blocked', 'failed', 'interrupted')
        if not terminal:
            return dict(state='running', reason='R114 pilot still running')
        if not queue.get('complete') and queue.get('handoff', {}).get('r114_terminal_authoritative') is not True:
            return dict(state='waiting_queue', reason='Historical pilot interruption is not an authoritative new terminal closeout')
        if progress.get('execution') is not True:
            return dict(state='needs_action', reason='R114 dry-run is not terminal execution evidence')
        analysis = progress.get('analysis') or {}
        if analysis.get('status') != 'complete' or not analysis.get('output'):
            return dict(state='closeout', reason='Terminal R114 needs finite saved-label analysis, including explicit partial facts')
        analysis_dir = inside(analysis['output'], root)
        analysis_receipt = self.receipt(analysis_dir / 'analysis_complete.json')
        if analysis_receipt.get('complete') is not True:
            return dict(state='closeout', reason='Await saved pilot analysis completion')
        for path in sorted((root / 'preparation/finish').glob('*/finish_complete.json')):
            if required_finish and path.parent != required_finish:
                continue
            finish = self.receipt(path)
            if (finish.get('closeout_ready') is not True
                    or finish.get('inputs', {}).get(str(progress_path)) != digest(progress_path)):
                continue
            kind = finish.get('closeout_kind')
            if kind not in ('complete_labels', 'partial_negative_closeout'):
                continue
            if (kind == 'partial_negative_closeout' and finish.get('labeling_complete') is not False
                    or kind == 'complete_labels' and finish.get('labeling_complete') is not True):
                raise NeedAction('R114 complete/partial closeout must be explicit, never inferred from missing labels')
            publication = self.receipt(path.parent / 'publish_receipt.json')
            if not published(publication):
                shared_publication = self.receipt(root / 'preparation/finish/publish_receipt.json')
                if shared_publication.get('delivery') == str(path.parent):
                    publication = shared_publication
            if published(publication):
                return dict(state='complete', closeout_kind=kind, finish_dir=str(path.parent),
                            progress_path=str(progress_path), publication=publication,
                            labeling_complete=finish['labeling_complete'])
        return dict(state='closeout', reason='Pilot complete or explicit partial-negative analysis awaits closeout/publication')

    def observe(self):
        self.paths.clear()
        try:
            queue = self.queue()
        except (NeedAction, KeyError, TypeError) as error:
            queue = dict(state='needs_action', reason=str(error), handoff={})
        tasks = {}
        for version in ROOTS:
            try:
                tasks[version] = self.pilot(queue) if version == 'R114' else self.experiment(version, queue)
            except (NeedAction, KeyError, TypeError, OSError, AttributeError) as error:
                tasks[version] = dict(state='needs_action', reason=str(error))
        return dict(queue=queue, tasks=tasks,
                    complete=all(row['state'] == 'complete' for row in tasks.values()))

    def snapshot(self, observation):
        """One evidence snapshot after the receipt gate, not a recurrent giga-audit."""
        paths = set(self.paths)
        for version, arms in ARMS.items():
            root = self.root(version)
            for name in ('RESULTS.md', 'RESULTS.pdf', 'metrics.csv', 'comparison.csv', 'science_boundary.json'):
                if (root / name).is_file():
                    paths.add(root / name)
            for arm in arms:
                run = root / arm
                for name in ('final_metrics.json', 'inference_timing.json', 'full_model_statistics/RESULTS.md',
                             'full_model_statistics/metrics_by_problem.csv',
                             'full_model_statistics/selection_distribution.csv',
                             'full_model_statistics/nss_comparison.csv'):
                    path = inside(run / name, self.repo)
                    if not path.is_file():
                        raise NeedAction('Missing saved evidence-pack input: ' + str(path))
                    paths.add(path)
        pilot = Path(observation['tasks']['R114']['finish_dir'])
        for name in ('closeout.json', 'RESULTS.md', 'counts.csv'):
            path = inside(pilot / name, self.repo)
            if not path.is_file():
                raise NeedAction('Missing explicit pilot facts: ' + str(path))
            paths.add(path)
        return {str(inside(p, self.repo)): digest(p) for p in sorted(paths)}


def csv_rows(path):
    with Path(path).open(newline='', encoding='utf-8-sig') as stream:
        return list(csv.DictReader(stream))


def verify_pack(folder, snapshot, observation, repo):
    """Fail closed before Astra; no model, predictions, labels or Git are executed."""
    folder = Path(folder)
    value = read(folder / 'verification.json')
    required = dict(complete=True, model=SOL, reasoning_effort='max', saved_results_only=True,
                    unique_val_checkpoints=True, same_version_metrics=True, test_development_exposed=True)
    if (any(value.get(k) != v for k, v in required.items())
            or value.get('verified_inputs') != snapshot
            or value.get('scientific_completion_gate_sha256') != json_hash(snapshot)
            or value.get('r114_closeout_kind') != observation['tasks']['R114']['closeout_kind']
            or not value.get('negative_partial_facts')
            or set(value.get('implemented_hypothesis_limits', {})) != {'R112', 'R113', 'R114'}):
        raise NeedAction('Sol/max saved-results evidence attestation is incomplete or stale')
    for name in ('evidence_pack.md', 'checkpoints.csv', 'metrics.csv'):
        path = inside(folder / name, folder)
        if not path.is_file() or value.get('outputs', {}).get(name) != digest(path):
            raise NeedAction('Evidence output is missing or not hash-bound: ' + name)
    for path, expected in snapshot.items():
        if digest(inside(path, repo)) != expected:
            raise NeedAction('Evidence changed during Sol verification: ' + path)
    expected_models = {(v, arm): final_lock for v, task in observation['tasks'].items()
                       for arm, final_lock in task.get('locks', {}).items()}
    rows = csv_rows(folder / 'checkpoints.csv')
    if len(rows) != 7 or {(r['experiment'], r['arm']) for r in rows} != set(expected_models):
        raise NeedAction('Exactly seven whole-model checkpoint bindings are required')
    pools = {}
    for row in rows:
        key = row['experiment'], row['arm']
        final_lock = expected_models[key]
        if (any(str(final_lock[k]) != row.get(k) for k in ('checkpoint_sha256', 'state_hash', 'selected_update', 'inference_rule'))
                or row.get('selection_split') != 'VAL'):
            raise NeedAction('Evidence checkpoint does not match unique frozen VAL lock: ' + str(key))
        config = read(Path(repo) / 'code/V4/runs' / ROOTS[key[0]] / key[1] / 'protocol.json')
        pools[key] = json_hash(config['candidate_pools'])
    rows = csv_rows(folder / 'metrics.csv')
    keys = [(r['experiment'], r['arm'], r['split'], r['problem']) for r in rows]
    expected = {(v, a, s, p) for v, a in expected_models for s in ('VAL', 'TEST') for p in (*PROBLEMS, 'ALL')}
    expected |= {(v, a, s, p) for v, a in expected_models for s, p in (('TSPLIB', 'TSP'), ('CVRPLIB', 'CVRP'))}
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise NeedAction('Evidence must cover all18 + equal-problem ALL VAL/TEST and both LIBs for each same model')
    for row in rows:
        key = row['experiment'], row['arm']
        if (row['checkpoint_sha256'] != expected_models[key]['checkpoint_sha256']
                or row.get('candidate_pool_sha256') != pools[key]
                or row.get('sbs_fit_split') != 'TRAIN' or not row.get('winner_rule')
                or row.get('data_version') != (DATA_VERSION if row['split'] in ('VAL', 'TEST') else 'native_' + row['split'])
                or not all(math.isfinite(float(row[k])) for k in METRICS)
                or not 0 <= float(row['top1_pct']) <= 100
                or int(row['n']) <= 0):
            raise NeedAction('Mixed-version/missing/nonfinite metric metadata: ' + str(key))
        if row['split'] in ('VAL', 'TEST'):
            if int(row['n']) != (81000 if row['problem'] == 'ALL' else 4500):
                raise NeedAction('Wrong frozen evaluation counts')
            if row['problem'] == 'ALL':
                members = [r for r in rows if (r['experiment'], r['arm'], r['split']) == (*key, row['split']) and r['problem'] != 'ALL']
                if row.get('all_aggregation') != 'equal_problem' or any(not math.isclose(float(row[k]),
                        sum(float(r[k]) for r in members) / 18, rel_tol=1e-6, abs_tol=1e-6) for k in METRICS):
                    raise NeedAction('ALL must be the equal-problem mean, not mixed raw cost or stitched models')
    return value


def publication_text_hash(path):
    # Git's text materialization must not turn CRLF-only changes into new science.
    return hashlib.sha256(Path(path).read_bytes().replace(b'\r\n', b'\n')).hexdigest()


def reconciled_evidence(root):
    authority = read(Path(root) / 'evidence_csv_reconciliation.json')
    if not authority:
        return Path(root) / 'evidence', False
    folder = inside(authority['evidence_dir'], root)
    old = Path(root) / 'evidence'
    proof = read(folder / 'publication_csv_materialization.json')
    if (authority.get('main_approved') is not True or authority.get('kind') != 'git_csv_eol_only'
            or authority.get('original_action') != 'verify-results'
            or authority.get('original_action_preserved') is not True
            or authority.get('verification_sha256') != digest(folder / 'verification.json')
            or authority.get('normalization_sha256') != digest(folder / 'publication_csv_materialization.json')
            or authority.get('original_verification_sha256') != digest(old / 'verification.json')
            or authority.get('generator_sha256') != digest(authority['generator'])
            or proof.get('complete') is not True or proof.get('parsed_content_equivalence') is not True
            or proof.get('strict_byte_identity') is not False or proof.get('discrepancies') != 92
            or authority.get('source_snapshot_sha256') != json_hash(read(folder / 'source_snapshot.json'))
            or any(authority.get('original_' + name + '_sha256') != digest(old / (name + '.csv'))
                   or digest(old / (name + '.csv')) != digest(folder / (name + '.csv'))
                   for name in ('metrics', 'checkpoints'))):
        raise NeedAction('CSV materialization reconciliation is incomplete or changes scientific evidence')
    return folder, True


def verify_research(repo):
    folder = Path(repo) / RESEARCH_REL
    report = folder / 'REPORT.md'
    if not report.is_file() or not report.read_text(encoding='utf-8').strip():
        raise NeedAction('Astra has not saved the full research report')
    sources = csv_rows(folder / 'sources.csv')
    if len(sources) < 8 or any(r.get('primary') != 'true' or r.get('browsed') != 'true'
                               or not r.get('url', '').startswith('https://') or not r.get('sections') for r in sources):
        raise NeedAction('Research requires actually browsed, section-specific primary-paper sources')
    directions = csv_rows(folder / 'directions.csv')
    fields = ('hypothesis', 'scope', 'decisive_control', 'falsifier', 'finite_plan',
              'gpu_hours_2x3090', 'label_cost', 'stop_rule', 'counterevidence')
    if not 3 <= len(directions) <= 5 or any(not all(row.get(k) for k in fields) for row in directions):
        raise NeedAction('Research needs 3-5 broad, costed, falsifiable finite plans, not coefficient tuning')
    return {str(p): publication_text_hash(p) for p in (report, folder / 'sources.csv', folder / 'directions.csv')}


def codex_cwd(model, repo):
    if model not in (SOL, ASTRA):
        raise NeedAction('Requested model is outside the Sol/Astra contract; no fallback')
    return inside(RESEARCH_REL, repo) if model == ASTRA else Path(repo).resolve()


def codex_argv(model, repo, output, schema):
    cwd = codex_cwd(model, repo)
    argv = [CODEX, *(['--search'] if model == ASTRA else []), 'exec', '-m', model,
            '-c', 'model_reasoning_effort="max"', '-c', 'approval_policy="never"',
            '--sandbox', 'workspace-write' if model == ASTRA else 'danger-full-access']
    if model == ASTRA:
        # Keep inherited extra roots and temporary directories out of research writes.
        argv += ['-c', 'sandbox_workspace_write.writable_roots=[]',
                 '-c', 'sandbox_workspace_write.exclude_tmpdir_env_var=true',
                 '-c', 'sandbox_workspace_write.exclude_slash_tmp=true',
                 '-c', 'sandbox_workspace_write.network_access=true']
    return argv + ['-C', str(cwd), '--json', '--output-schema', str(schema), '-o', str(output), '-']


def worker(attempt):
    """Durable supervisor, independently adoptable after a controller restart."""
    attempt = Path(attempt).resolve()
    request = read(attempt / 'request.json')
    with lock(attempt / 'execution.lock'):
        if (attempt / 'exit.json').exists():
            return 0
        if (attempt / 'worker.json').exists():
            # Even a stale claim is ambiguous: never start a second model call.
            return 2
        save(attempt / 'worker.json', dict(identity=process_identity(os.getpid()), started_utc=utc()))
        output = attempt / 'response.json'
        environment = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1',
                           OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2')
        result = dict(action_id=request['action_id'], requested_model=request['model'],
                      requested_reasoning_effort='max',
                      model_identity_evidence='CLI requested-model flag only; self-report is not provider routing proof',
                      provider_route_verified=False, state='failed', exit_status=1)
        try:
            expected = ASTRA if request.get('stage') == 'research' else SOL
            if request['model'] != expected:
                raise NeedAction('Model/stage violates Sol execution and Astra research-only contract; no fallback')
            argv = codex_argv(request['model'], request['repo'], output, request['schema'])
            cwd = codex_cwd(request['model'], request['repo'])
            result.update(argv=argv, cwd=str(cwd))
            run_root = Path(request['repo']) / RUN_REL
            if read(run_root / 'control.json').get('paused') is True or read(run_root / 'monitor_plan.json').get('enabled') is False:
                raise NeedAction('User paused before model launch; no model call was started')
            if request['model'] == ASTRA:
                cwd.mkdir(parents=True, exist_ok=True)
            with (attempt / 'prompt.md').open('rb') as prompt, (attempt / 'events.jsonl').open('ab') as events, (attempt / 'stderr.log').open('ab') as errors:
                child = subprocess.Popen(argv, cwd=cwd, env=environment,
                                         stdin=prompt, stdout=events, stderr=errors)
                save(attempt / 'codex.json', dict(identity=process_identity(child.pid), argv=argv, cwd=str(cwd)))
                try:
                    rc = child.wait(timeout=MODEL_TIMEOUT_SECONDS)
                except subprocess.TimeoutExpired:
                    child.terminate()
                    try:
                        child.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        child.kill()
                        child.wait(timeout=10)
                    raise NeedAction('Requested model CLI exceeded six hours; terminated owned CLI, no automatic retry')
            response = read(output)
            valid = (response.get('action_id') == request['action_id'] and response.get('model') == request['model']
                     and response.get('reasoning_effort') == 'max' and response.get('status') == 'complete'
                     and response.get('made_progress') is True)
            unavailable = response.get('status') == 'model_unavailable'
            for name in ('stderr.log', 'events.jsonl'):
                with (attempt / name).open('rb') as stream:
                    stream.seek(max(0, (attempt / name).stat().st_size - 65536))
                    unavailable |= bool(re.search(rb'model_not_found|model[^\n]{0,100}(?:not available|not supported|does not exist)|unsupported model', stream.read(), re.I))
            result.update(state='complete' if rc == 0 and valid and not unavailable else 'needs_action',
                          exit_status=rc, model_unavailable=unavailable, response=response,
                          reason='Model unavailable; no fallback' if unavailable else response.get('summary', 'Missing/invalid result or no material progress'))
        except Exception as error:
            result.update(state='needs_action', reason=type(error).__name__ + ': ' + str(error))
        result['finished_utc'] = utc()
        save(attempt / 'exit.json', result)
        return 0 if result['state'] == 'complete' else 1


class Monitor:
    def __init__(self, repo=REPO, handoff=None):
        self.repo = Path(repo).resolve()
        self.root = self.repo / RUN_REL
        self.handoff = Path(handoff or self.root / 'queue_handoff.json').resolve()
        self.sources = Sources(self.repo, self.handoff)
        self.state = read(self.root / 'state.json') or dict(schema_version=1, actions={}, notified=[], phase='ready')
        self.children = []
        self.wakeup = threading.Event()

    def persist(self):
        path = self.root / 'state.json'
        if read(path) != self.state:
            save(path, self.state)

    def notify(self, kind, target, message):
        identifier = json_hash([kind, target, message])
        if identifier in self.state['notified']:
            return
        self.state['notified'].append(identifier)
        self.persist()
        value = dict(utc=utc(), kind=kind, target=target, message=message)
        with (self.root / 'events.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(canonical(value) + '\n')
        print(canonical(value), flush=True)

    def reconcile(self):
        for child in self.children[:]:
            if child.poll() is not None:
                self.children.remove(child)
        active = False
        for key, action in self.state['actions'].items():
            if action['state'] not in ('claimed', 'running'):
                continue
            attempt = Path(action['attempt'])
            exit_record = read(attempt / 'exit.json')
            if exit_record:
                action.update(state=exit_record['state'], result=exit_record)
                if action['state'] != 'complete':
                    self.notify('failure', key, exit_record.get('reason', 'Action failed; no automatic retry'))
                if exit_record.get('model_unavailable'):
                    self.state['model_unavailable'] = action['model']
                continue
            started = action.get('started_utc') or read(attempt / 'request.json').get('created_utc')
            if started and (datetime.now(timezone.utc) - datetime.fromisoformat(started)).total_seconds() >= MODEL_TIMEOUT_SECONDS:
                self.stop_owned(action)
                action.update(state='needs_action', reason='Six-hour action deadline exceeded, including orphaned CLI; no automatic retry')
                self.notify('failure', key, action['reason'])
                continue
            claims = [read(attempt / n).get('identity') for n in ('worker.json', 'codex.json')]
            # The controller could crash between Popen and persisting its child PID.
            found = any(alive(identity) for identity in claims if identity)
            if not found:
                token = str(attempt)
                found = any(token in p['argv'] or str(attempt / 'response.json') in p['argv'] for p in processes())
            if found:
                action['state'] = 'running'
                active = True
            else:
                action.update(state='needs_action', reason='Claimed worker/CLI disappeared without an exit receipt; inspect before any restart')
                self.notify('need_action', key, action['reason'])
        self.persist()
        return active

    def dispatch(self, key, stage, context):
        if key in self.state['actions']:
            action = self.state['actions'][key]
            if action['state'] == 'complete':
                self.notify('need_action', key, 'Action returned complete but required saved artifacts are still missing; no repeated preflights')
            return False
        recoveries = sum(a['stage'] == 'execution' and k.startswith('repair:') for k, a in self.state['actions'].items())
        if key.startswith('repair:') and recoveries >= MAX_RECOVERIES:
            self.notify('need_action', key, 'Total bounded recovery budget exhausted; main intervention required')
            return False
        model = ASTRA if stage == 'research' else SOL
        identifier = json_hash(key)[:24]
        attempt = self.root / 'actions' / identifier
        if attempt.exists():
            self.notify('need_action', key, 'Action directory exists without ledger entry; reconcile it, never overwrite/relaunch')
            return False
        attempt.mkdir(parents=True)
        request = dict(action_id=identifier, key=key, stage=stage, model=model, reasoning_effort='max', repo=str(self.repo),
                       schema=str(self.root / 'response.schema.json'), context=context, created_utc=utc())
        prompt = (self.root / 'prompts/common.md').read_text() + '\n' + (self.root / ('prompts/' + stage + '.md')).read_text()
        prompt += '\n\n## Invocation Context\n```json\n' + json.dumps(request, indent=2) + '\n```\n'
        (attempt / 'prompt.md').write_text(prompt, encoding='utf-8')
        save(attempt / 'request.json', request)
        self.state['actions'][key] = dict(state='claimed', stage=stage, model=model, attempt=str(attempt),
                                          started_utc=request['created_utc'])
        self.persist()
        try:
            if stage == 'research':
                codex_cwd(model, self.repo).mkdir(parents=True, exist_ok=True)
            with (attempt / 'supervisor.log').open('ab') as log:
                child = subprocess.Popen([sys.executable, '-B', '-m', MODULE, '_worker', '--attempt-dir', str(attempt)],
                                         cwd=self.repo, stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                                         start_new_session=True)
            self.children.append(child)
            self.state['actions'][key]['identity'] = process_identity(child.pid)
            self.persist()
        except Exception as error:
            self.state['actions'][key].update(state='needs_action', reason=str(error))
            self.notify('failure', key, 'Supervisor launch uncertain; no automatic retry: ' + str(error))
        return True

    def stop_owned(self, action):
        attempt = Path(action['attempt'])
        identities = [read(attempt / n).get('identity') for n in ('worker.json', 'codex.json')]
        identities.append(action.get('identity'))
        signalled = set()
        for identity in identities:
            if not identity or identity['pid'] in signalled or not alive(identity):
                continue
            signalled.add(identity['pid'])
            try:
                if os.getpgid(identity['pid']) == identity['pid']:
                    os.killpg(identity['pid'], signal.SIGTERM)
                else:
                    os.kill(identity['pid'], signal.SIGTERM)
            except ProcessLookupError:
                pass

    def pause_owned(self):
        for key, action in self.state['actions'].items():
            if action['state'] not in ('claimed', 'running'):
                continue
            self.stop_owned(action)
            action.update(state='needs_action', reason='User paused owned model action; inspect partial outputs before explicit recovery')
            self.notify('need_action', key, action['reason'])

    def tick(self):
        control = read(self.root / 'control.json')
        plan = read(self.root / 'monitor_plan.json')
        if control.get('paused') is True or plan.get('enabled') is False:
            self.pause_owned()
            self.state['phase'] = 'paused'
            self.persist()
            return False
        active = self.reconcile()
        observation = self.sources.observe()
        self.state['observation'] = observation
        self.state['phase'] = 'waiting'
        queue = observation['queue']
        if queue['state'] in ('awaiting_handoff', 'needs_action'):
            self.notify('need_action', 'queue_handoff', queue.get('reason', 'Await Mencius exact new two-GPU state-path handoff; no guessed path or GPU launch'))
        for version, task in observation['tasks'].items():
            if task['state'] == 'complete':
                self.notify('completion', version, 'Published saved version complete' +
                            (': ' + task['closeout_kind'] if version == 'R114' else '; scientific improvement is not implied'))
            elif task['state'] in ('needs_action', 'queue_failure', 'report_only_failure'):
                self.notify('failure' if task['state'] in ('queue_failure', 'report_only_failure') else 'need_action', version, task['reason'] +
                            (': ' + task.get('signature', '') if task['state'] in ('queue_failure', 'report_only_failure') else ''))
        if active:
            self.state['phase'] = 'model_action_running'
            self.persist()
            return False
        if self.state.get('model_unavailable'):
            self.state['phase'] = 'needs_action'
            self.persist()
            return False
        base = dict(observation=observation, queue_handoff=str(self.handoff), monitor_root=str(self.root),
                    allowed_gpu_uuids=list(UUIDS), allocation_job=1465, poll_seconds=POLL_SECONDS)
        for version, task in observation['tasks'].items():
            if task['state'] == 'report_only_failure':
                if module_running('code.V4.r109_goal'):
                    continue
                self.dispatch('repair:R109_pdf:' + task['signature'], 'execution',
                              dict(base, target=version, mode='report_only_recovery'))
                self.persist()
                return False
            if task['state'] == 'queue_failure':
                if module_running(QUEUE_MODULE):
                    continue
                self.dispatch('repair:' + task['signature'], 'execution', dict(base, target=version, mode='repair'))
                self.persist()
                return False
            if task['state'] == 'closeout':
                modules = {'R109': ('code.V4.r109_goal',),
                           'R112': ('code.V4.r112_postprocess_recovery', 'code.V4.r112_finish'),
                           'R113': (QUEUE_MODULE, 'code.V4.r113_finish'), 'R114': ('code.V4.r114_finish', QUEUE_MODULE)}
                if any(module_running(module) for module in modules[version]):
                    continue
                adapter = queue.get('handoff', {}).get('finish_adapter_path')
                if version == 'R113' and adapter and path_running(adapter):
                    continue
                if version == 'R113' and not queue['handoff'].get('closeout_argv'):
                    self.notify('need_action', version, 'Mencius must hand off a new-queue-compatible closeout argv; never rewrite old failed receipts')
                    continue
                self.dispatch('closeout:' + version, 'execution', dict(base, target=version, mode='closeout'))
                self.persist()
                return False
        if observation['complete']:
            self.advance_research(observation, base)
        self.persist()
        return self.state['phase'] == 'complete'

    def advance_research(self, observation, base):
        if self.state.get('research_gate_blocked'):
            self.state['phase'] = 'needs_action'
            return
        try:
            evidence, csv_reconciled = reconciled_evidence(self.root)
            snapshot_path = evidence / 'source_snapshot.json'
            snapshot = read(snapshot_path)
            if not snapshot:
                snapshot = self.sources.snapshot(observation)
                save(snapshot_path, snapshot)
            context = dict(base, source_snapshot=str(snapshot_path), source_snapshot_sha256=json_hash(snapshot),
                           evidence_dir=str(evidence), research_dir=str(self.repo / RESEARCH_REL))
            action = self.state['actions'].get('verify-results')
            if (not action or action['state'] != 'complete') and not csv_reconciled:
                self.dispatch('verify-results', 'verify', context)
                return
            verify_pack(evidence, snapshot, observation, self.repo)
            research = self.state['actions'].get('deep-research')
            if not research or research['state'] != 'complete':
                self.dispatch('deep-research', 'research', context)
                return
            research_hashes = verify_research(self.repo)
            if self.state.get('research_hashes') and self.state['research_hashes'] != research_hashes:
                raise NeedAction('Astra research prose/CSVs changed after handoff; Sol may render/publish, not replace the scientific content')
            self.state['research_hashes'] = research_hashes
            publication = self.state['actions'].get('publish-research')
            if not publication or publication['state'] != 'complete':
                self.dispatch('publish-research', 'publish', dict(context, research_hashes=research_hashes))
                return
            folder = self.repo / RESEARCH_REL
            receipt = read(folder / 'publication_receipt.json')
            expected_mutex = str(self.repo.parent / '.git/codex-publication.lock')
            required = dict(research_hashes, **{str(folder / 'REPORT.pdf'): digest(folder / 'REPORT.pdf')})
            if (not pdf_ready(folder / 'REPORT.pdf') or not published(receipt)
                    or receipt.get('model') != SOL or receipt.get('reasoning_effort') != 'max'
                    or receipt.get('global_mutex') != expected_mutex
                    or any(receipt.get('published_hashes', {}).get(p) != h for p, h in required.items())):
                raise NeedAction('Missing valid Sol PDF / exact global-mutex publication evidence')
            self.state['phase'] = 'complete'
            self.notify('completion', 'research', 'Completed research version published; PDF: ' + str(folder / 'REPORT.pdf'))
        except (NeedAction, KeyError, OSError, ValueError) as error:
            self.state['phase'] = 'needs_action'
            self.state['research_gate_blocked'] = str(error)
            self.notify('need_action', 'research_gate', str(error))

    def run(self):
        with lock(self.root / 'controller.lock'):
            self.terminate = False
            def terminate(*_):
                self.terminate = True
                self.wakeup.set()
            signal.signal(signal.SIGUSR1, lambda *_: self.wakeup.set())
            signal.signal(signal.SIGTERM, terminate)
            self.state.update(controller=process_identity(os.getpid()), main_approved=True,
                              queue_handoff=str(self.handoff), poll_seconds=POLL_SECONDS)
            self.persist()
            while True:
                self.wakeup.clear()
                try:
                    if self.tick():
                        return 0
                except Exception as error:
                    self.state['phase'] = 'needs_action'
                    self.notify('failure', 'controller', type(error).__name__ + ': ' + str(error))
                self.wakeup.wait(POLL_SECONDS)
                if self.terminate:
                    self.state['phase'] = 'controller_stopped_workers_adoptable'
                    self.persist()
                    return 0


def start(repo, handoff):
    repo = Path(repo).resolve()
    root = repo / RUN_REL
    with lock(root / 'start.lock'):
        state = read(root / 'state.json')
        if state.get('controller') and alive(state['controller']):
            return dict(state='already_running', controller=state['controller'])
        session = 'research_followthrough_20261026'
        existing = subprocess.run(['tmux', 'has-session', '-t', session], capture_output=True)
        if existing.returncode == 0:
            raise NeedAction('tmux session already exists; inspect/restart its claimed PID, do not spawn blindly')
        command = ['bash', str(repo / 'code/V4/run_research_completion_monitor.sh'), 'run',
                   '--main-approved', '--queue-handoff', str(handoff)]
        subprocess.run(['tmux', 'new-session', '-d', '-s', session, '-c', str(repo), shlex.join(command)], check=True)
        return dict(state='started', tmux_session=session, poll_seconds=POLL_SECONDS)


def set_pause(root, paused):
    root = Path(root)
    save(root / 'control.json', dict(paused=paused, source='explicit_user_command', utc=utc()))
    identity = read(root / 'state.json').get('controller')
    if alive(identity):
        os.kill(identity['pid'], signal.SIGUSR1)
    return dict(paused=paused)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('status', 'start', 'run', 'pause', 'resume', '_worker'))
    parser.add_argument('--main-approved', action='store_true')
    parser.add_argument('--queue-handoff', type=Path, default=REPO / RUN_REL / 'queue_handoff.json')
    parser.add_argument('--attempt-dir', type=Path)
    args = parser.parse_args(argv)
    if args.command in ('start', 'run') and not args.main_approved:
        parser.error('start/run require explicit --main-approved; this implementation does not start itself')
    try:
        if args.command == '_worker':
            if not args.attempt_dir:
                parser.error('_worker requires --attempt-dir')
            attempt = inside(args.attempt_dir, REPO / RUN_REL / 'actions')
            return worker(attempt)
        if args.command == 'status':
            state = read(REPO / RUN_REL / 'state.json')
            result = dict(phase=state.get('phase', 'not_started'), controller_alive=alive(state.get('controller')),
                          poll_seconds=POLL_SECONDS, control=read(REPO / RUN_REL / 'control.json'),
                          queue_handoff=str(args.queue_handoff), actions=state.get('actions', {}),
                          observation=state.get('observation', {}))
        elif args.command in ('pause', 'resume'):
            result = set_pause(REPO / RUN_REL, args.command == 'pause')
        elif args.command == 'start':
            result = start(REPO, args.queue_handoff)
        else:
            return Monitor(handoff=args.queue_handoff).run()
        print(json.dumps(result, indent=2))
        return 0
    except (NeedAction, BlockingIOError, OSError) as error:
        print(type(error).__name__ + ': ' + str(error), file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())

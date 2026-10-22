"""Explicit R100-only status/publication after the documented CPU report recovery.

The original stopped watcher stays unchanged. Only its exact superseded error
is removed from the active gate, and remains visible in superseded_errors.
No watcher, model, report, launch or Git retry is provided here.
"""

import argparse
import copy
from datetime import datetime
import importlib.util
import json
from pathlib import Path
import sys

from .r100_report_only_recovery import csv_counts, identity


REPO = Path(__file__).resolve().parents[2]
RECOVERY_RELATIVE = 'preparation/report_only_recovery/20261022_cpu_render_01'
WATCH_ERROR = 'Stage failure; preserve receipts and outputs; no retry or model completion'


def require(condition, message):
    if not condition:
        raise RuntimeError('R100 administrative recovery refused: ' + message)


def private_goal():
    spec = importlib.util.spec_from_file_location(
        'code.V4._r100_goal_recovery_private', REPO / 'code/V4/r98_goal.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class RecoveryGoal:
    def __init__(self, module):
        self.base = module
        self.root = module.model_root('R100').resolve()
        self.launch = module.LAUNCH.resolve()
        self.recovery = self.root / RECOVERY_RELATIVE
        self.original_status = module.status
        # Original publish() resolves this private global before and inside its
        # existing shared codex-publication.lock. The shared module is untouched.
        module.status = self.status

    def verify(self):
        base, root, recovery = self.base, self.root, self.recovery
        original = recovery / 'original'
        receipt_path = recovery / 'recovery_complete.json'
        receipt = base.optional(receipt_path)
        before, after = (base.optional(recovery / name) for name in ('before.json', 'after.json'))
        require(receipt.get('experiment') == 'R100' and receipt.get('complete') is True
                and receipt.get('phase') == 'complete' and type(receipt.get('exit_status')) is int
                and receipt['exit_status'] == 0, 'missing genuine complete/rc0 report recovery')
        for key, expected in dict(manual_explicit_recovery=True, original_attempt_succeeded=False,
                inference_performed=False, reselection_performed=False, automatic_retry=False,
                protected_metadata_and_sha256_unchanged=True, immutable_snapshots_unchanged=True,
                stopped_goalwatch_preserved=True, csv_rows_unchanged=True).items():
            require(receipt.get(key) is expected, 'invalid recovery flag: ' + key)
        require(bool(before.get('protected')) and bool(before.get('snapshots'))
                and before['protected'] == after.get('protected')
                and all(after.get(k) is True for k in ('unchanged', 'snapshots_unchanged', 'csv_rows_unchanged')),
                'before/after unchanged proof is absent or inconsistent')
        required = [root / name for name in ('protocol.json', 'checkpoint.json', 'final_test_lock.json',
                    'test_input_lock.json', 'val_results.json', 'test_results.json', 'lib_results.json',
                    'run_complete.json', 'goal_watch_receipt.json')]
        required += [self.launch / 'R100_A/worker.json', self.launch / 'R100_A/plan.json',
                     self.launch / 'R100_A/ready.json']
        required += [root / name for name in ('evaluation/val/selected_best/rank000.npz',
                    'evaluation/test/selected_best/rank000.npz', 'lib_evaluation/predictions/TSPLIB.npz',
                    'lib_evaluation/predictions/CVRPLIB.npz')]
        require(all(str(path) in before['protected'] for path in required), 'incomplete protected core proof')
        require(all(str(original / name) in before['snapshots'] for name in
                ('controller.json', 'goal_watch_receipt.json', 'r100_reporting.py',
                 'postprocess_status.json', 'PDF.log')), 'incomplete original failure archive')
        for group in ('protected', 'snapshots'):
            for name, expected in before[group].items():
                path = Path(name)
                if group == 'snapshots':
                    require(path.resolve().is_relative_to(original.resolve()), 'snapshot escapes archive')
                require(identity(path) == expected, 'preserved file identity changed: ' + str(path))
        watcher_path, archived_path = root / 'goal_watch_receipt.json', original / 'goal_watch_receipt.json'
        require(str(archived_path) in before['snapshots'], 'archived watcher is not identity-bound')
        require(watcher_path.read_bytes() == archived_path.read_bytes(), 'watcher differs from immutable archive')
        require(archived_path.stat().st_mode & 0o222 == 0, 'archived watcher is writable')
        watcher = base.optional(watcher_path)
        require(watcher.get('experiment') == 'R100' and watcher.get('state') == 'stopped_on_error'
                and type(watcher.get('exit_status')) is int and watcher['exit_status'] == 1
                and watcher.get('error') == WATCH_ERROR and watcher.get('automatic_retry') is False,
                'not the exact original stopped watcher failure')
        expected_error = dict(path=str(watcher_path), error=WATCH_ERROR)
        require(base.receipt_errors(watcher_path, watcher) == [expected_error], 'extra watcher errors')
        lock = base.optional(root / 'final_test_lock.json')
        require(lock == before.get('lock') and all(receipt.get(k) == lock.get(k)
                for k in ('checkpoint_sha256', 'state_hash'))
                and base.file_hash(root / 'checkpoint.json') == receipt['checkpoint_sha256'],
                'recovery is not bound to the same frozen bank')
        old = base.optional(original / 'controller.json')
        controller = base.optional(self.launch / 'R100_A/controller.json')
        require(old.get('state') == 'failed' and old.get('exit_status') == 1
                and base.rc0(controller) and 'error' not in controller
                and controller.get('original_error') == old.get('error')
                and controller.get('recovery_receipt') == str(receipt_path), 'controller failure history not preserved')
        old_stages, stages = old.get('stages', []), controller.get('stages', [])
        require([s.get('name') for s in old_stages] == ['allocation_GPU_pipeline', 'PDF']
                and [s.get('name') for s in stages] == ['allocation_GPU_pipeline', 'PDF'], 'unexpected stage layout')
        require(base.rc0(old_stages[0]) and stages[0] == old_stages[0], 'original GPU rc0 stage changed')
        cpu = copy.deepcopy(stages[1])
        require(cpu.pop('original_attempt', None) == old_stages[1]
                and cpu.pop('original_attempt_snapshot', None) == str(original / 'controller.json')
                and old_stages[1].get('state') == 'failed' and old_stages[1].get('exit_status') == 1,
                'original failed PDF attempt missing')
        require(cpu == receipt.get('cpu_stage') == base.optional(recovery / 'cpu_stage.json')
                and base.rc0(cpu) and cpu.get('device') == 'cpu'
                and type(cpu.get('pid')) is int and cpu['pid'] > 0, 'actual recovered CPU pid/rc0 not bound')
        argv = [sys.executable, '-B', '-u', '-m', 'code.V4.r100_experiment', 'report',
                '--root', str(root), '--device', 'cpu']
        attempt = base.optional(recovery / 'attempt.json')
        require(cpu.get('argv') == argv == attempt.get('argv') and attempt.get('automatic_retry') is False
                and cpu.get('automatic_retry') is False and cpu.get('started_utc') == attempt.get('started_utc'),
                'not the explicit single-attempt CPU report command')
        require(datetime.fromisoformat(old_stages[1]['finished_utc'])
                < datetime.fromisoformat(cpu['started_utc']) <= datetime.fromisoformat(cpu['finished_utc'])
                <= datetime.fromisoformat(receipt['completed_utc']), 'invalid recovery chronology')
        require(receipt.get('original_reporting_sha256') == before.get('original_reporting_sha256')
                == base.file_hash(original / 'r100_reporting.py')
                and receipt.get('recovery_reporting_sha256') == attempt.get('reporting_sha256')
                == base.file_hash(base.REPO / 'code/V4/r100_reporting.py'), 'report source hashes differ')
        provenance = base.optional(root / 'full_model_statistics/provenance.json')
        require(provenance.get('inference_performed') is False
                and all(provenance.get(k) == lock.get(k) for k in ('checkpoint_sha256', 'state_hash')),
                'report provenance lacks same-bank/no-inference evidence')
        pdfs = [root / 'RESULTS.pdf', *[root / 'full_model_statistics' / name for name in base.FROZEN_PDFS]]
        require(set(receipt.get('PDFs', {})) == {str(path) for path in pdfs}, 'incomplete PDF receipt')
        require(all(base.pdf_ready(path) and identity(path) == receipt['PDFs'][str(path)] for path in pdfs),
                'recovered PDF identity changed')
        require(csv_counts(root / 'full_model_statistics') == before.get('csv_rows')
                == after.get('csv_rows') == receipt.get('csv_rows'), 'dataframe counts changed')
        return dict(error=expected_error, archive=str(archived_path),
                    watcher_sha256=identity(watcher_path)['sha256'], report_recovery=str(receipt_path),
                    report_recovery_sha256=base.file_hash(receipt_path), cpu_pid=cpu['pid'], cpu_exit_status=0,
                    original_PDF_exit_status=1, original_GPU_exit_status=0,
                    checkpoint_sha256=lock['checkpoint_sha256'], state_hash=lock['state_hash'])

    def status(self, version='R100', root=None, launch_root=None, *, persist=True):
        require(version == 'R100' and Path(root or self.root).resolve() == self.root
                and Path(launch_root or self.launch).resolve() == self.launch, 'only canonical R100 is authorized')
        evidence = self.verify()
        original = self.original_status('R100', self.root, self.launch, persist=False)
        require(original.get('errors') == [evidence['error']], 'another, missing or duplicate active error')
        require(original.get('all_arms_ready') is True and original.get('comparison_complete') is True
                and original.get('arms') and all(row.get('ready') is True and not row.get('missing')
                    and all(row.get('checks', {}).values()) for row in original['arms']),
                'original full-pipeline checks are not all ready')
        current = copy.deepcopy(original)
        current.update(errors=[], complete=True, phase='complete',
            original_phase=original['phase'], superseded_errors=[evidence],
            administrative_recovery=dict(explicit=True, watcher_unchanged=True, watcher_restarted=False,
                original_first_attempt_succeeded=False, no_inference_or_launch=True,
                lifecycle_note='Historical report recovery only; current R101 A/B are managed by main.',
                helper_sha256=self.base.file_hash(Path(__file__)),
                shared_goal_sha256=self.base.file_hash(Path(self.base.__file__))))
        if persist:
            current['updated_utc'] = self.base.utc()
            self.base.save(self.root / 'goal_progress.json', current)
        return current


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--version', choices=('R100',), default='R100')
    parser.add_argument('--publish-github', action='store_true')
    args = parser.parse_args(argv)
    goal = RecoveryGoal(private_goal())
    current = goal.status(args.version, persist=False)
    goal.base.save(goal.recovery / 'goal_recovery_status.json', dict(status=current,
        checked_utc=goal.base.utc(), mode='publication_requested' if args.publish_github else 'status_only',
        publication_complete=False, canonical_goal_progress_written=False,
        original_watcher_preserved=True, no_watch_or_launch=True))
    if args.publish_github:
        # Do not replace the curated path builder, Git mutex, publisher or retry rules.
        try:
            published = goal.base.publish('R100', goal.root, goal.launch)
        except BaseException as error:
            goal.base.save(goal.recovery / 'goal_recovery_publication_result.json', dict(
                checked_utc=goal.base.utc(), complete=False, error=str(error), automatic_retry=False,
                original_watcher_preserved=True, original_publisher_receipt=str(goal.root / 'github_publication.json')))
            raise
        current = goal.status(persist=False)
        result = dict(status=current, publication=published)
        goal.base.save(goal.recovery / 'goal_recovery_publication_result.json', result)
        print(json.dumps(result, indent=2))
    else:
        print(json.dumps(current, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

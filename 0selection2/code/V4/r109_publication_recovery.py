"""CPU-only, exact-manifest recovery of the completed R109 publication.

The failed watcher is evidence, not something to reset or rerun. Planning never
stages files. Publishing requires main's approval of the final manifest digest,
and makes at most one commit attempt and one normal push attempt.
"""

import argparse
import ast
from contextlib import contextmanager
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys


PROJECT = Path(__file__).resolve().parents[2]
ORIGIN = 'git@github.com:viayuu/selection.git'
BASE = '446ac3c046b8c3dfb5722b529a4356b1ead80a61'
VERSION_PATH = 'code/V4/runs/R109_method_conditioned_nodes'
RECOVERY_PATH = 'preparation/publication_recovery_20261026'
ARMS = ('A_tied_late', 'B_private_late')
EVIDENCE = 'R109_main66CPU_twoView128_1024_seed2_full18'
WATCH_ERROR = 'Publication requires canonical roots'
WATCH_ENDED = '2026-10-25T19:52:49.452683+00:00'
MESSAGE = 'r109: publish completed full18 version (publication recovery)'
RECOVERY_FILES = ('README.md', 'completion_gate.json', 'source_closure.json',
                  'git_reconciliation.json', 'fixture_receipt.json', 'publication_plan.json',
                  'dispatch_summaries.json')
SOURCE_PREFIXES = ('r109_', 'test_r109_', 'run_v4_r109_')
EXTRA = 'docs/research/post_r108_learning_mechanisms_20261024.md'
ASSET_EXTENSIONS = {'.json', '.csv', '.pdf', '.md'}
PRUNED = {'sources', 'runtime', 'cache', 'input_cache', 'feature_cache', 'weights',
          'python', 'wheels', 'wandb', 'logs', '__pycache__', 'evaluation',
          'predictions', 'outcomes', 'checkpoints', 'site-packages', 'figures'}
MAX_BYTES = 32 * 1024**2


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def saved_gate_ready(current, saved, original, root):
    """Accept exactly the known administrative error, never rewrite its status."""
    require(original.get('experiment') == 'R109'
            and original.get('state') == 'stopped_on_error'
            and type(original.get('exit_status')) is int and original['exit_status'] == 1
            and original.get('error') == WATCH_ERROR
            and original.get('completed_utc') == WATCH_ENDED
            and original.get('main_ready_evidence') == EVIDENCE
            and original.get('automatic_retry') is False,
            'Original publication-only failure differs; no broader exception authorized')
    require(current.get('errors') == [dict(path=str(root / 'goal_watch_receipt.json'),
                                         error=WATCH_ERROR)],
            'Scientific gate has additional/missing errors; no recovery exception')
    for record in (current, saved):
        require(record.get('experiment') == 'R109'
                and record.get('all_arms_ready') is True
                and record.get('comparison_complete') is True
                and record.get('matched_controls') is True
                and record.get('common_original_initialization') is True
                and [row.get('arm') for row in record.get('arms', [])] == list(ARMS)
                and all(row.get('ready') is True and row.get('checks')
                        and all(v is True for v in row['checks'].values())
                        for row in record['arms']),
                'BOTH whole-budget arm pipelines and comparison must already be complete')
    require(saved.get('complete') is True and saved.get('errors') == [],
            'Saved prepublication completion gate was not successful')
    require(current.get('complete') is False and current.get('phase') == 'failed',
            'Original watcher must remain failed, not fabricated as successful')


class Recovery:
    def __init__(self, project=PROJECT, origin=ORIGIN, base=BASE):
        self.project = Path(project).resolve()
        self.git_root = self.project.parent
        self.root = self.project / VERSION_PATH
        self.dispatch = self.root / 'preparation/dispatch'
        self.here = self.root / RECOVERY_PATH
        self.manifest = self.here / 'publication_manifest.json'
        self.receipt = self.here / 'publication_receipt.json'
        self.origin, self.base = origin, base

    def relative(self, path):
        return str(Path(path).relative_to(self.git_root))

    def environment(self):
        return dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1',
                    OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                    NUMEXPR_NUM_THREADS='1', GIT_TERMINAL_PROMPT='0')

    def run(self, argv, *, data=None, timeout=60, check=True):
        return subprocess.run(argv, cwd=self.git_root, env=self.environment(), input=data,
                              capture_output=True, timeout=timeout, check=check)

    def git(self, *args, data=None, timeout=60):
        return self.run(['git', *args], data=data, timeout=timeout).stdout.decode().strip()

    def save(self, path, value, *, exclusive=False):
        path = Path(path)
        require(path.parent == self.here, 'Write outside owned recovery directory')
        self.here.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(value, indent=2, allow_nan=False, ensure_ascii=True) + '\n'
        if exclusive:
            with path.open('x', encoding='utf-8') as stream:
                stream.write(payload)
            return
        temporary = path.with_name('.' + path.name + '.tmp')
        temporary.write_text(payload, encoding='utf-8')
        temporary.replace(path)

    @contextmanager
    def locked(self):
        require(self.git('rev-parse', '--show-toplevel') == str(self.git_root),
                'Noncanonical Git root')
        git_dir = Path(self.git('rev-parse', '--absolute-git-dir')).resolve()
        require(git_dir == self.git_root / '.git', 'Unexpected Git directory/worktree')
        with (git_dir / 'codex-publication.lock').open('a') as mutex:
            fcntl.flock(mutex, fcntl.LOCK_EX | fcntl.LOCK_NB)
            yield git_dir

    def remote_head(self):
        fields = self.git('ls-remote', '--exit-code', 'origin', 'refs/heads/main').split()
        require(len(fields) == 2 and fields[1] == 'refs/heads/main', 'Unexpected remote ref')
        return fields[0]

    def no_previous_attempt(self):
        require(not self.receipt.exists() and not (self.root / 'github_publication.json').exists(),
                'Publication already attempted; preserve receipt, no commit/push retry')
        other = [p for p in self.root.rglob('*publication*.json')
                 if not p.is_relative_to(self.here)]
        require(not other, 'Other publication records require manual reconciliation')

    def preflight(self):
        self.no_previous_attempt()
        require(self.git('branch', '--show-current') == 'main', 'Expected existing main branch')
        require(not self.git('diff', '--cached', '--name-only', '-z'),
                'Pre-existing staged index; do not commit or unstage it')
        require(not (self.git_root / '.git/index.lock').exists(), 'Another Git writer is active')
        require(self.git('remote', 'get-url', 'origin') == self.origin, 'Origin changed')
        require(self.git('remote', 'get-url', '--push', 'origin') == self.origin,
                'Push origin differs from verified remote')
        head, remote = self.git('rev-parse', 'HEAD'), self.remote_head()
        require(head == remote == self.base, 'HEAD/remote/base mismatch; no merge/reset/force')
        pattern = '[Rr]109.*[Pp]ubli'
        require(not self.git('log', '--all', '--format=%H', '--extended-regexp', '--grep=' + pattern),
                'R109 publication commit already exists; no duplicate')
        require(not self.git('reflog', '--all', '--format=%H %gs', '--extended-regexp',
                             '--grep-reflog=' + pattern),
                'R109 publication in reflog; uncertain prior commit must be reconciled')
        critical = [self.relative(self.root / n) for n in
                    ('RESULTS.md', 'MODEL_COMPARISON.pdf', 'A_tied_late/final_test_lock.json',
                     'B_private_late/final_test_lock.json')]
        require(not self.git('log', '--all', '-1', '--format=%H', '--', *critical),
                'Completed R109 result paths already have history; inspect, do not republish')
        return dict(checked_utc=utc(), exclusive_global_mutex_acquired=True,
                    global_mutex=str(self.git_root / '.git/codex-publication.lock'),
                    git_root=str(self.git_root), branch='main', head=head, remote_main=remote,
                    stage_index_empty=True, no_prior_R109_publication_commit=True,
                    no_prior_publication_receipt=True, git_mutation_performed=False)

    def goal(self):
        path = self.project / 'code/V4/r109_goal.py'
        spec = importlib.util.spec_from_file_location('code.V4._r109_publication_evidence', path)
        goal = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(goal)
        require(goal.ROOT.resolve() == self.root and goal.LAUNCH.resolve() == self.dispatch,
                'R109 canonical scientific roots changed')
        return goal

    def completion_gate(self):
        goal = self.goal()
        original, saved = (read(self.root / n) for n in
                           ('goal_watch_receipt.json', 'goal_progress.json'))
        current = goal.status(root=self.root, launch_root=self.dispatch, persist=False)
        saved_gate_ready(current, saved, original, self.root)
        compared = read(self.root / 'goal_compare_receipt.json')
        require(compared.get('state') == 'complete' and compared.get('exit_status') == 0,
                'Original CPU comparison did not finish rc0')
        source = self.project / 'code/V4/r98_goal.py'
        tree = ast.parse(source.read_text(encoding='utf-8'))
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'publish')
        guard = function.body[1]
        require(isinstance(guard, ast.If) and isinstance(guard.body[0], ast.Raise)
                and WATCH_ERROR in ast.get_source_segment(source.read_text(), guard)
                and goal.adapter.LAUNCH.resolve() != self.dispatch,
                'Prior error is not the inherited pre-Git canonical-root guard')
        locks, protected, summary = {}, [], {}
        for arm in ARMS:
            run = self.root / arm
            lock = read(run / 'final_test_lock.json')
            locks[arm] = lock
            require(sha(run / 'best.pt') == lock['checkpoint_sha256'],
                    'Selected checkpoint bytes differ from VAL lock')
            protected.append(dict(path=self.relative(run / 'best.pt'), sha256=sha(run / 'best.pt')))
            with (run / 'full_model_statistics/metrics_by_problem.csv').open(
                    newline='', encoding='utf-8-sig') as stream:
                rows = list(csv.DictReader(stream))
            summary[arm] = {row['split']: row for row in rows if row['problem'] == 'ALL'}
            require(set(summary[arm]) == {'TRAIN', 'VAL', 'TEST'}, 'Missing saved ALL metrics')
        return dict(checked_utc=utc(), original_watcher=original, original_goal_status=current,
                    saved_prepublication_status=saved, recovered_pipeline_complete=True,
                    original_watcher_success=False, accepted_administrative_error=WATCH_ERROR,
                    error_occurs_before_receipt_lock_stage_commit=True,
                    inherited_publisher_source_sha256=sha(source), locks=locks,
                    selected_checkpoint_bytes=protected, ALL_saved_metrics=summary,
                    cpu_only=True, model_or_report_code_executed=False,
                    training=False, new_TEST=False, new_timing=False, new_LIB=False,
                    report_regenerated=False, model_reselected=False, goal_complete=False)

    def parent_tree(self):
        raw = self.git('ls-tree', '-r', '-z', self.base)
        result = {}
        for entry in filter(None, raw.split('\0')):
            header, name = entry.split('\t', 1)
            mode, kind, oid = header.split()
            result[name] = dict(mode=mode, kind=kind, oid=oid)
        return result

    def blob_oid(self, path):
        # Git's existing attributes/autocrlf may normalize CSV CRLF in the index.
        # Hash through those same filters without writing either the file or index.
        return self.git('hash-object', '--path=' + self.relative(path), '--stdin',
                        data=Path(path).read_bytes())

    def source_closure(self, tree):
        frozen = {}
        for arm in ARMS:
            output = self.dispatch / ('R109_' + arm)
            spec = read(self.dispatch / 'specs' / ('R109_' + arm + '.json'))
            plan, ready = read(output / 'plan.json'), read(output / 'ready.json')
            require(plan.get('spec') == spec and spec.get('run_dir') == str(self.root / arm)
                    and plan.get('evidence') == ready.get('evidence') == EVIDENCE
                    and ready.get('plan_sha256') == sha(output / 'plan.json')
                    and ready.get('source_freeze_ready') is True
                    and ready.get('main_thread_ready') is True,
                    'Source-frozen launch plan/readiness changed')
            for name, digest in plan['source_sha256'].items():
                require(name not in frozen or frozen[name] == digest, 'Arm source locks differ')
                frozen[name] = digest
        require(frozen, 'No frozen source dependency closure')
        rows, own = [], set()
        for name, digest in sorted(frozen.items()):
            path = Path(name)
            require(path.is_file() and sha(path) == digest,
                    'Frozen source/input changed: ' + name)
            relative = self.relative(path)
            publish = (path.is_relative_to(self.root)
                       or (path.parent == self.project / 'code/V4'
                        and path.name.startswith(SOURCE_PREFIXES))
                       or relative == self.relative(self.project / EXTRA))
            in_parent = relative in tree
            if path.suffix in ('.py', '.sh', '.md') and not publish:
                require(in_parent and tree[relative]['oid'] == self.blob_oid(path),
                        'Shared dependency absent/different in parent: ' + relative)
            if publish:
                own.add(path)
            rows.append(dict(path=relative, sha256=digest, already_in_parent=in_parent,
                             selected_for_publication=publish))
        return dict(frozen_sources_unchanged=True, parent_commit=self.base, sources=rows,
                    binary_input_and_checkpoint_files_intentionally_excluded=True), own

    def candidates(self, own):
        paths = set(own)
        for parent in (self.root, *(self.root / arm for arm in ARMS)):
            paths.update(p for p in parent.iterdir()
                         if p.suffix.lower() in ASSET_EXTENSIONS
                         and not p.name.startswith(('goal_', 'github_', 'wandb')))
        for arm in ARMS:
            run = self.root / arm
            paths.update(p for p in (run / 'full_model_statistics').iterdir()
                         if p.suffix.lower() in ASSET_EXTENSIONS)
            paths.update(run / 'lib_evaluation' / (dataset + '_' + kind + '.json')
                         for dataset in ('TSPLIB', 'CVRPLIB') for kind in ('protocol', 'results'))
        paths.update(p for p in (self.root / 'comparison_artifacts').iterdir()
                     if p.suffix.lower() in ASSET_EXTENSIONS)
        for parent, dirs, files in os.walk(self.root / 'preparation'):
            dirs[:] = [n for n in dirs if n not in PRUNED
                       and (Path(parent) / n) != self.here
                       and not (Path(parent) / n).is_symlink()]
            paths.update(Path(parent) / n for n in files if Path(n).suffix.lower() in ASSET_EXTENSIONS
                         and not (Path(parent).parent == self.dispatch
                                  and n in {'claim.json', 'controller.json', 'entry.json'}))
        paths.update(self.root / name for name in
                     ('goal_watch_receipt.json', 'goal_progress.json', 'goal_compare_receipt.json'))
        paths.update(self.project / 'code/V4' / name for name in
                     ('r109_publication_recovery.py', 'test_r109_publication_recovery.py'))
        paths.update(self.here / name for name in RECOVERY_FILES)
        return paths

    def asset(self, path, safe_json):
        path = Path(path)
        require(path.is_file() and not path.is_symlink()
                and not any(p.is_symlink() for p in path.parents)
                and path.resolve().is_relative_to(self.project), 'Unsafe asset: ' + str(path))
        owned_source = path.parent == self.project / 'code/V4' and path.name.startswith(SOURCE_PREFIXES)
        require(path.is_relative_to(self.root) or owned_source or path == self.project / EXTRA,
                'Unrelated asset: ' + str(path))
        require(path.suffix.lower() in ASSET_EXTENSIONS | {'.py', '.sh'}
                and path.stat().st_size <= MAX_BYTES, 'Noncurated/oversized asset: ' + str(path))
        require(not any(word in path.name.lower() for word in
                        ('wandb', 'credential', 'secret', 'prediction', 'outcome', 'raw_label', 'raw_cost')),
                'Sensitive/array asset rejected: ' + str(path))
        if path.suffix == '.json':
            require(safe_json(read(path)), 'Sensitive JSON rejected: ' + str(path))
        if path.suffix != '.pdf':
            content = path.read_text(encoding='utf-8')
            patterns = (r'-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----',
                        r'\b(?:ghp_|github_pat_)[A-Za-z0-9_]{24,}', r'\bsk-[A-Za-z0-9_-]{32,}')
            require(not any(re.search(pattern, content) for pattern in patterns),
                    'Possible credential rejected: ' + str(path))
        return dict(path=self.relative(path), sha256=sha(path), bytes=path.stat().st_size,
                    git_blob_oid=self.blob_oid(path))

    def plan(self):
        require(not self.manifest.exists(), 'Keep the existing manifest; do not silently regenerate')
        with self.locked():
            reconciliation = self.preflight()
            gate = self.completion_gate()
            tree = self.parent_tree()
            closure, own = self.source_closure(tree)
            self.save(self.here / 'completion_gate.json', gate, exclusive=True)
            self.save(self.here / 'source_closure.json', closure, exclusive=True)
            self.save(self.here / 'git_reconciliation.json', reconciliation, exclusive=True)
            fixture = read(self.here / 'fixture_receipt.json')
            require(fixture.get('complete') is True and fixture.get('exit_status') == 0
                    and fixture.get('helper_sha256') == sha(self.project / 'code/V4/r109_publication_recovery.py')
                    and fixture.get('test_sha256') == sha(self.project / 'code/V4/test_r109_publication_recovery.py'),
                    'Minimal isolated CPU fixture must pass before manifest creation')
            summaries, withheld = {}, []
            for arm in ARMS:
                output = self.dispatch / ('R109_' + arm)
                summaries[arm] = {}
                for name in ('controller.json', 'worker.json'):
                    path = output / name
                    record = read(path)
                    summaries[arm][name] = dict(
                        original_path=self.relative(path), original_sha256=sha(path),
                        state=record.get('state'), exit_status=record.get('exit_status'),
                        stages=[{k: s[k] for k in ('name', 'state', 'exit_status',
                                                  'started_utc', 'completed_utc') if k in s}
                                for s in record.get('stages', [])])
                withheld.extend(dict(path=self.relative(output / name), sha256=sha(output / name),
                                     reason='Raw dispatch command/claim receipt may contain access token')
                                for name in ('controller.json', 'claim.json', 'entry.json')
                                if (output / name).is_file())
            self.save(self.here / 'dispatch_summaries.json', dict(
                experiment='R109', command_argv_and_credentials_excluded=True,
                original_receipts_unchanged=True, arms=summaries, withheld_raw_receipts=withheld), exclusive=True)
            self.save(self.here / 'publication_plan.json', dict(
                experiment='R109', kind='precommit_intent_not_publication_success', created_utc=utc(),
                parent_commit=self.base, remote_before=self.base, origin=self.origin, branch='main',
                commit_message=MESSAGE, strict_manifest_approval_required=True, cpu_only=True,
                original_failure_preserved=True, only_publication_recovered=True,
                model_reselected=False, experiment_rerun=False, report_regenerated=False,
                single_commit=True, normal_push=True, automatic_retry=False,
                actual_receipt_intentionally_uncommitted=True, goal_complete=False), exclusive=True)
            safe_json = self.goal().adapter._safe_json
            candidates = self.candidates(own)
            records = [self.asset(p, safe_json) for p in sorted(candidates)]
            require(all(r['already_in_parent'] or r['selected_for_publication']
                        or Path(r['path']).suffix not in ('.py', '.sh', '.md')
                        for r in closure['sources']), 'Incomplete source closure')
            changed = [r['path'] for r in records
                       if tree.get(r['path'], {}).get('oid') != r['git_blob_oid']]
            manifest_relative = self.relative(self.manifest)
            require(manifest_relative not in tree, 'Recovery manifest already exists in parent')
            manifest = dict(schema_version=1, experiment='R109', created_utc=utc(),
                            parent_commit=self.base, remote_before=self.base,
                            records=records, total_asset_bytes=sum(r['bytes'] for r in records),
                            expected_changed_paths=sorted(changed + [manifest_relative]),
                            explicit_stage_paths=sorted([r['path'] for r in records] + [manifest_relative]),
                            withheld_raw_dispatch_receipts=withheld,
                            original_watch_receipt_unchanged=True, binary_weights_and_arrays_excluded=True,
                            actual_publication_receipt_intentionally_uncommitted=True)
            self.save(self.manifest, manifest, exclusive=True)
            return dict(state='ready_for_main_GO', manifest=str(self.manifest),
                        manifest_sha256=sha(self.manifest), parent_commit=self.base,
                        asset_count=len(records), changed_path_count=len(changed) + 1,
                        total_asset_bytes=manifest['total_asset_bytes'], git_mutation_performed=False)

    def publish(self, approved_digest, approval_evidence):
        require(approval_evidence and approval_evidence.strip(), 'Explicit main GO evidence required')
        require(self.manifest.is_file() and sha(self.manifest) == approved_digest,
                'Main GO digest differs from the final manifest')
        with self.locked():
            self.preflight()
            manifest = read(self.manifest)
            require(manifest['parent_commit'] == manifest['remote_before'] == self.base,
                    'Manifest parent changed')
            self.completion_gate()
            closure, _ = self.source_closure(self.parent_tree())
            require(closure == read(self.here / 'source_closure.json'), 'Frozen source closure changed')
            safe_json = self.goal().adapter._safe_json
            for row in manifest['records']:
                require(self.asset(self.git_root / row['path'], safe_json) == row,
                        'Manifest asset changed: ' + row['path'])
            files = manifest['explicit_stage_paths']
            require(set(files) == {r['path'] for r in manifest['records']} | {self.relative(self.manifest)},
                    'Stage paths differ from exact manifest')
            receipt = dict(experiment='R109', kind='explicit_additive_publication_recovery',
                           state='staging', complete=False, started_utc=utc(),
                           parent_commit=self.base, remote_before=self.base,
                           manifest=str(self.manifest), manifest_sha256=approved_digest,
                           approval_evidence=approval_evidence, cpu_only=True,
                           global_mutex=str(self.git_root / '.git/codex-publication.lock'),
                           original_failure_preserved=True, original_watcher_success=False,
                           automatic_retry=False, commit_attempts=0, push_attempts=0,
                           training=False, new_TEST=False, new_LIB=False, new_timing=False,
                           report_regenerated=False, model_reselected=False,
                           force_push=False, history_rewritten=False, goal_complete=False)
            self.save(self.receipt, receipt, exclusive=True)
            try:
                require(not self.git('diff', '--cached', '--name-only', '-z'), 'Index changed before staging')
                for start in range(0, len(files), 80):
                    self.git('add', '-f', '--', *files[start:start + 80])
                staged = set(filter(None, self.git('diff', '--cached', '--name-only', '-z').split('\0')))
                require(staged == set(manifest['expected_changed_paths']), 'Unexpected staged paths')
                expected = {r['path']: r['git_blob_oid'] for r in manifest['records']}
                expected[self.relative(self.manifest)] = self.blob_oid(self.manifest)
                for entry in filter(None, self.git('ls-files', '--stage', '-z', '--', *files).split('\0')):
                    header, name = entry.split('\t', 1)
                    mode, oid, stage = header.split()
                    require(stage == '0' and mode in ('100644', '100755') and oid == expected[name],
                            'Index blob/mode differs from approved manifest: ' + name)
                require(self.git('branch', '--show-current') == 'main'
                        and self.git('rev-parse', 'HEAD') == self.remote_head() == self.base,
                        'HEAD/branch/remote changed before commit; no commit')
                receipt.update(state='committing', commit_attempts=1)
                self.save(self.receipt, receipt)
                output = self.git('commit', '-m', MESSAGE, timeout=180)
                commit = self.git('rev-parse', 'HEAD')
                require(self.git('rev-parse', commit + '^') == self.base, 'Unexpected commit parent; no push')
                actual = set(filter(None, self.git('diff-tree', '--no-commit-id', '--name-only',
                                                   '-r', '-z', commit).split('\0')))
                require(actual == staged, 'Commit differs from verified index; no push')
                tree = {entry.split('\t', 1)[1]: entry.split('\t', 1)[0].split()[2]
                        for entry in filter(None, self.git('ls-tree', '-r', '-z', commit, '--', *files).split('\0'))}
                require(all(tree.get(name) == oid for name, oid in expected.items()),
                        'Committed blobs differ from manifest; no push')
                receipt.update(state='committed_pending_push', commit=commit,
                               changed_path_count=len(actual), commit_output=output)
                self.save(self.receipt, receipt)
                require(self.git('branch', '--show-current') == 'main'
                        and self.git('rev-parse', 'HEAD') == commit
                        and not self.git('diff', '--cached', '--name-only', '-z'),
                        'Post-commit branch/index changed; no push')
                require(self.remote_head() == self.base, 'Remote moved before push; no retry/force')
                receipt.update(state='pushing', push_attempts=1)
                self.save(self.receipt, receipt)
                pushed = self.run(['git', '-c', 'pack.threads=1', '-c', 'pack.windowMemory=64m',
                                   '-c', 'pack.deltaCacheSize=16m', 'push', '--porcelain',
                                   'origin', commit + ':refs/heads/main'], timeout=180)
                remote = self.remote_head()
                require(remote == commit and self.git('rev-parse', 'HEAD') == commit,
                        'Post-push remote/HEAD mismatch; preserve uncertainty, no retry')
                require(all(sha(self.git_root / row['path']) == row['sha256']
                            for row in manifest['records']), 'Asset changed during publication')
                require(sha(self.manifest) == approved_digest, 'Manifest changed during publication')
                receipt.update(state='complete', complete=True, exit_status=0, completed_utc=utc(),
                               remote_after=remote, canonical_main_remote_verified=True,
                               verification='Independent post-push git ls-remote origin refs/heads/main',
                               push_output=(pushed.stdout + pushed.stderr).decode(),
                               frozen_assets_unchanged=True, unrelated_changes_staged=False)
                self.save(self.receipt, receipt)
                return receipt
            except BaseException as error:
                receipt.update(state='stopped_on_error', exit_status=1, stopped_utc=utc(),
                               error=str(error), actual_HEAD_after_error=self.git('rev-parse', 'HEAD'),
                               no_retry_requires_manual_reconciliation=True)
                self.save(self.receipt, receipt)
                raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('plan', 'publish'))
    parser.add_argument('--approved-manifest-sha256')
    parser.add_argument('--approval-evidence')
    args = parser.parse_args(argv)
    require(PROJECT == Path('/public/home/shiys/0selection2'), 'CLI requires canonical workspace')
    recovery = Recovery()
    os.environ.update(recovery.environment())
    if hasattr(os, 'sched_getaffinity'):
        os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:2])
    result = (recovery.plan() if args.action == 'plan' else recovery.publish(
        args.approved_manifest_sha256, args.approval_evidence))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()

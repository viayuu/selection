"""One-shot, exact-manifest R110 publication; never run experiment stages."""

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys


HERE = Path(__file__).resolve().parent
PROJECT = Path('/public/home/shiys/0selection2')
GIT_ROOT = PROJECT.parent
ROOT = PROJECT / 'code/V4/runs/R110_method_conditioned_ce'
DISPATCH = ROOT / 'preparation/dispatch'
PREFIX = str(HERE.relative_to(GIT_ROOT)) + '/'
EVIDENCE = 'R110_main58CPU_singleView256_1024_seed2_full18'
ORIGIN = 'git@github.com:viayuu/selection.git'
MESSAGE = 'r110: publish completed full18 version (publication recovery)'
WATCH_ERROR = 'Publication requires canonical roots'
EXTRA = ('code/V4/r109_model.py', 'code/V4/test_r109_model.py',
         'code/V4/runs/R109_method_conditioned_nodes/MODEL_PREP.md',
         'docs/research/post_r108_learning_mechanisms_20261024.md')
RECOVERY_FILES = ('README.md', 'recover_publication.py', 'completion_gate.json',
                  'source_closure.json', 'publication_plan.json')
MANIFEST = HERE / 'publication_manifest.json'
RECEIPT = HERE / 'publication_receipt.json'
sys.path.insert(0, str(PROJECT))


def utc():
    return datetime.now(timezone.utc).isoformat()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def index_blob_matches(record, blob):
    if hashlib.sha256(blob).hexdigest() == record['sha256']:
        return True
    path = GIT_ROOT / record['path']
    return (path.suffix == '.csv' and sha(path) == record['sha256']
            and blob == path.read_bytes().replace(b'\r\n', b'\n'))


def save(path, value):
    path = Path(path)
    require(path.parent == HERE or path == ROOT / 'github_publication.json',
            'Write outside explicitly owned receipt paths')
    temporary = path.with_name('.' + path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=True) + '\n', encoding='utf-8')
    temporary.replace(path)


def run(argv, *, timeout=60, check=True, input=None):
    return subprocess.run(argv, cwd=GIT_ROOT, env=os.environ.copy(), input=input,
                          capture_output=True, text=True, timeout=timeout, check=check)


def git(*args, timeout=60):
    return run(['git', *args], timeout=timeout).stdout.strip()


def remote_head():
    result = git('ls-remote', '--exit-code', 'origin', 'refs/heads/main', timeout=60)
    fields = result.split()
    require(len(fields) == 2 and fields[1] == 'refs/heads/main', 'Unexpected remote ref')
    return fields[0]


def git_preflight(owned_staged=None):
    require(git('rev-parse', '--show-toplevel') == str(GIT_ROOT), 'Noncanonical Git root')
    require(git('branch', '--show-current') == 'main', 'Expected existing main branch')
    staged = set(filter(None, git('diff', '--cached', '--name-only', '-z').split('\0')))
    require(not staged or owned_staged is not None and staged.issubset(owned_staged),
            'BLOCKER: unrelated staged index; stop and ask main, do not commit or unstage')
    require(git('remote', 'get-url', 'origin') == ORIGIN, 'Origin changed; no remote rewrite')
    parent = git('rev-parse', 'HEAD')
    remote = remote_head()
    require(parent == remote, 'Local HEAD differs from origin/main; no merge, reset or force')
    require(not git('log', '--all', '--format=%H', '--grep=^r110: publish completed full18'),
            'An R110 completed-version commit already exists; do not duplicate')
    require(not git('log', '-1', '--format=%H', '--',
                    '0selection2/code/V4/r110_goal.py',
                    '0selection2/code/V4/runs/R110_method_conditioned_ce'),
            'R110 paths already have history; inspect instead of creating another version')
    return parent, remote


def gate():
    from code.V4 import r110_goal as goal

    require(goal.ROOT.resolve() == ROOT and goal.LAUNCH.resolve() == DISPATCH,
            'R110 helper canonical roots changed')
    original = read(ROOT / 'goal_watch_receipt.json')
    saved = read(ROOT / 'goal_progress.json')
    require(original.get('state') == 'stopped_on_error' and original.get('exit_status') == 1
            and original.get('error') == WATCH_ERROR
            and original.get('completed_utc') == '2026-10-25T03:22:04.961361+00:00'
            and original.get('main_ready_evidence') == EVIDENCE,
            'Original publication-only failure differs; no broader exception authorized')
    compared = read(ROOT / 'goal_compare_receipt.json')
    require(saved['all_arms_ready'] is True and saved['comparison_complete'] is True
            and all(arm['ready'] and all(arm['checks'].values()) for arm in saved['arms'])
            and saved.get('complete') is True and saved.get('errors') == []
            and compared.get('state') == 'complete' and compared.get('exit_status') == 0,
            'Existing saved completion/comparison evidence changed')
    return goal, saved


def source_closure(goal, parent):
    frozen = {}
    for arm in goal.ARMS:
        for name, digest in read(DISPATCH / ('R110_' + arm) / 'plan.json')['source_sha256'].items():
            require(name not in frozen or frozen[name] == digest, 'Arm source locks differ')
            frozen[name] = digest
    rows = []
    names = [str(Path(name).relative_to(GIT_ROOT)) for name in frozen
             if Path(name).suffix in ('.py', '.sh', '.md')]
    tracked = set(filter(None, git('ls-files', '-z', '--', *names).split('\0')))
    require(not git('diff', '--name-only', parent, '--', *sorted(tracked)),
            'Shared source dependency differs from parent; stop without editing')
    for name, digest in sorted(frozen.items()):
        path = Path(name)
        if path.suffix not in ('.py', '.sh', '.md'):
            continue
        relative = str(path.relative_to(GIT_ROOT))
        rows.append(dict(path=relative, frozen_plan_sha256=digest,
                         already_in_parent=relative in tracked))
    return rows


def safe_asset(goal, relative):
    path = GIT_ROOT / relative
    require(path.is_file() and not path.is_symlink()
            and not any(p.is_symlink() for p in path.parents)
            and path.resolve().is_relative_to(PROJECT), 'Unsafe asset path: ' + relative)
    require(path.suffix.lower() in ('.py', '.sh', '.json', '.csv', '.pdf', '.md', '.log'),
            'Noncurated extension: ' + relative)
    require(path.stat().st_size <= 32 * 1024**2, 'Asset exceeds 32 MiB: ' + relative)
    own = relative.startswith(str(ROOT.relative_to(GIT_ROOT)) + '/')
    source = path.parent == PROJECT / 'code/V4' and path.name.startswith(
        ('r110_', 'test_r110_', 'run_v4_r110_'))
    require(own or source or path in {PROJECT / p for p in EXTRA},
            'Unrelated asset outside exact R110 source closure: ' + relative)
    require('r111' not in relative.lower() and path.suffix.lower() not in ('.npz', '.pt', '.pth'),
            'Future source or model/data array rejected')
    if path.suffix == '.json':
        require(goal.adapter._safe_json(read(path)), 'Sensitive JSON rejected: ' + relative)
    if path.suffix != '.pdf':
        content = path.read_text(encoding='utf-8')
        patterns = (r'-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----',
                    r'\b(?:ghp_|github_pat_)[A-Za-z0-9_]{24,}',
                    r'\bsk-[A-Za-z0-9_-]{32,}', r'\bAKIA[A-Z0-9]{16}\b')
        require(not any(re.search(pattern, content) for pattern in patterns),
                'Possible credential found; withheld: ' + relative)
    return dict(path=relative, sha256=sha(path), bytes=path.stat().st_size)


def plan():
    require(not MANIFEST.exists() and not RECEIPT.exists()
            and not (ROOT / 'github_publication.json').exists(), 'Preserve existing recovery/publication')
    parent, remote = git_preflight()
    goal, current = gate()
    assets = set(goal.publication_paths('R110', ROOT, DISPATCH, GIT_ROOT))
    assets = {p for p in assets if not p.startswith(PREFIX) and '/full_model_statistics/figures/' not in p}
    assets.update(str((PROJECT / p).relative_to(GIT_ROOT)) for p in EXTRA)
    assets.update(str((ROOT / p).relative_to(GIT_ROOT)) for p in
                  ('goal_watch_receipt.json', 'goal_compare_receipt.json', 'goal_progress.json'))
    for arm in goal.ARMS:
        for dataset in ('TSPLIB', 'CVRPLIB'):
            for kind in ('protocol', 'results'):
                assets.add(str((ROOT / arm / 'lib_evaluation' / (dataset + '_' + kind + '.json')).relative_to(GIT_ROOT)))
    closure = source_closure(goal, parent)
    require(all(row['already_in_parent'] or row['path'] in assets for row in closure),
            'Missing frozen source dependency; inspect closure before publication')
    save(HERE / 'completion_gate.json', dict(
        checked_utc=utc(), saved_prepublication_gate_unchanged=current,
        original_failure=read(ROOT / 'goal_watch_receipt.json'),
        publication_only_exception=WATCH_ERROR, saved_prepublication_gate_complete=True,
        scientific_gate_rerun=False, cpu_model_tests_run=False,
        recovered_publication_gate=True, goal_complete=False, goal_status='active_unresolved'))
    save(HERE / 'source_closure.json', dict(parent_commit=parent, sources=closure,
        frozen_sources_unchanged=True, weights_and_data_intentionally_excluded=True))
    save(HERE / 'publication_plan.json', dict(experiment='R110', kind='precommit_publication_intent',
        created_utc=utc(), parent_commit=parent, remote_before=remote, branch='main', origin=ORIGIN,
        commit_message=MESSAGE, original_failure_preserved=True, frozen_science_unchanged=True,
        model_reselected=False, experiment_rerun=False, single_commit=True, normal_push=True,
        cpu_model_tests_run=False, initial_preflight_stopped_before_tests_or_git=True,
        global_mutex=str(GIT_ROOT / '.git/codex-publication.lock'),
        goal_complete=False, goal_status='active_unresolved',
        actual_receipt_intentionally_uncommitted=True,
        requested_executor=dict(model='gpt-6.1-sol', reasoning_effort='max',
                                backend_identity_independently_verified=False)))
    assets.update(PREFIX + name for name in RECOVERY_FILES)
    records = [safe_asset(goal, p) for p in sorted(assets)]
    save(MANIFEST, dict(experiment='R110', parent_commit=parent, records=records,
        total_asset_bytes=sum(r['bytes'] for r in records),
        explicit_stage_paths=sorted(assets | {str(MANIFEST.relative_to(GIT_ROOT))}),
        manifest_self_hash='Bound by Git commit; no recursive self hash'))
    print(json.dumps(dict(state='plan_ready', parent_commit=parent, asset_count=len(records),
                          asset_bytes=sum(r['bytes'] for r in records), cpu_model_tests_run=False)))


def publish(resume_staged=False):
    prior = read(RECEIPT) if resume_staged else None
    if resume_staged:
        require(prior.get('state') == 'stopped_on_error' and 'commit' not in prior
                and prior.get('push_attempts') == 0 and prior.get('error') == (
                    'Index blob differs from manifest: 0selection2/code/V4/runs/'
                    'R110_method_conditioned_ce/A_tied_ce/full_model_statistics/inference_timing.csv'),
                'Only the exact precommit CSV text-decoding error may be continued')
    require(MANIFEST.is_file() and (resume_staged or not RECEIPT.exists())
            and not (ROOT / 'github_publication.json').exists(),
            'One-shot recovery already attempted or manifest missing; no retry/new commit')
    git_dir = Path(git('rev-parse', '--absolute-git-dir'))
    with (git_dir / 'codex-publication.lock').open('a') as mutex:
        fcntl.flock(mutex, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest = read(MANIFEST)
        owned_staged = set(manifest['explicit_stage_paths']) if resume_staged else None
        parent, remote = git_preflight(owned_staged)
        require(parent == manifest['parent_commit'], 'HEAD changed since manifest preparation')
        goal, _ = gate()
        if resume_staged:
            # Verify the original index as bytes before rebinding only our helper.
            normalized_csv = []
            for record in manifest['records']:
                blob = subprocess.check_output(['git', 'show', ':' + record['path']],
                                               cwd=GIT_ROOT, timeout=30)
                require(index_blob_matches(record, blob),
                        'Precontinuation index changed: ' + record['path'])
                if hashlib.sha256(blob).hexdigest() != record['sha256']:
                    normalized_csv.append(dict(path=record['path'],
                                               index_sha256=hashlib.sha256(blob).hexdigest()))
            previous_path = HERE / 'staging_check_01.json'
            require(not previous_path.exists(), 'Precommit continuation already attempted')
            save(previous_path, prior)
            helper = str(Path(__file__).resolve().relative_to(GIT_ROOT))
            manifest['records'] = [safe_asset(goal, helper) if r['path'] == helper else r
                                   for r in manifest['records']]
            manifest['records'].append(safe_asset(goal, str(previous_path.relative_to(GIT_ROOT))))
            manifest['records'].sort(key=lambda r: r['path'])
            manifest['explicit_stage_paths'] = sorted(set(manifest['explicit_stage_paths'])
                | {str(previous_path.relative_to(GIT_ROOT))})
            manifest['total_asset_bytes'] = sum(r['bytes'] for r in manifest['records'])
            manifest['administrative_continuation'] = (
                'Explicit precommit continuation: existing core.autocrlf=input normalizes CSV CRLF '
                'to LF in Git only; retain original worktree SHA256 and verify exact normalized bytes; '
                'no tests, metrics, frozen-source changes or earlier commit')
            manifest['git_csv_normalization'] = normalized_csv
            save(MANIFEST, manifest)
        for record in manifest['records']:
            require(safe_asset(goal, record['path']) == record,
                    'Manifest asset changed: ' + record['path'])
        files = manifest['explicit_stage_paths']
        require(set(files) == {r['path'] for r in manifest['records']}
                | {str(MANIFEST.relative_to(GIT_ROOT))}, 'Stage list differs from exact hashed manifest')
        receipt = dict(experiment='R110', kind='explicit_additive_publication_recovery',
            complete=False, state='staging', started_utc=utc(), parent_commit=parent,
            remote_before=remote, global_mutex=str(git_dir / 'codex-publication.lock'),
            manifest=str(MANIFEST), manifest_sha256=sha(MANIFEST),
            branch='main', origin=ORIGIN, curated_only=True, original_failure_preserved=True,
            cpu_only=True, experiment_rerun=False, model_reselected=False,
            automatic_retry=False, force_push=False, history_rewritten=False,
            goal_complete=False, goal_status='active_unresolved',
            actual_receipt_intentionally_uncommitted=True, push_attempts=0)
        if resume_staged:
            receipt.update(explicit_precommit_continuation=True,
                           preserved_staging_check=str(HERE / 'staging_check_01.json'))
        save(RECEIPT, receipt)
        try:
            staged_before = set(filter(None, git('diff', '--cached', '--name-only', '-z').split('\0')))
            require(not staged_before or resume_staged and staged_before.issubset(owned_staged),
                    'Index changed before exact staging; stop and ask main')
            for start in range(0, len(files), 80):
                git('add', '--', *files[start:start + 80])
            staged = set(filter(None, git('diff', '--cached', '--name-only', '-z').split('\0')))
            require(staged and staged.issubset(set(files))
                    and git('branch', '--show-current') == 'main'
                    and git('rev-parse', 'HEAD') == parent,
                    'Unexpected staged/branch/HEAD mutation; no unstage/reset performed')
            for record in manifest['records']:
                if record['path'] in staged:
                    blob = subprocess.check_output(['git', 'show', ':' + record['path']],
                                                   cwd=GIT_ROOT, timeout=30)
                    require(index_blob_matches(record, blob),
                            'Index blob differs from manifest: ' + record['path'])
            commit_result = git('commit', '-m', MESSAGE, timeout=120)
            commit = git('rev-parse', 'HEAD')
            require(git('rev-parse', commit + '^') == parent, 'Unexpected commit parent')
            changed = set(filter(None, git('diff-tree', '--no-commit-id', '--name-only', '-r', '-z', commit).split('\0')))
            require(changed == staged, 'Commit differs from verified curated index; no push')
            receipt.update(state='committed_pending_push', commit=commit,
                           changed_path_count=len(changed), new_commit_created=True,
                           commit_output=commit_result)
            save(RECEIPT, receipt)
            require(git('branch', '--show-current') == 'main' and git('rev-parse', 'HEAD') == commit
                    and not git('diff', '--cached', '--name-only', '-z'), 'Post-commit state changed; no push')
            receipt['push_attempts'] = 1
            save(RECEIPT, receipt)
            pushed = run(['git', '-c', 'pack.threads=1', '-c', 'pack.windowMemory=64m',
                          '-c', 'pack.deltaCacheSize=16m', 'push', '--porcelain',
                          'origin', commit + ':refs/heads/main'], timeout=180)
            remote_after = remote_head()
            require(remote_after == commit, 'Remote main does not match completed-version commit')
            for record in manifest['records']:
                require(sha(GIT_ROOT / record['path']) == record['sha256'],
                        'Frozen asset changed during publication: ' + record['path'])
            receipt.update(complete=True, state='complete', exit_status=0, completed_utc=utc(),
                remote_after=remote_after, canonical_main_remote_verified=True,
                verification='Independent post-push git ls-remote origin refs/heads/main',
                push_porcelain=pushed.stdout + pushed.stderr, frozen_assets_unchanged=True,
                unrelated_changes_staged=False)
            save(RECEIPT, receipt)
            save(ROOT / 'github_publication.json', receipt)
            print(json.dumps({k: receipt[k] for k in ('state', 'commit', 'remote_after',
                                                     'changed_path_count', 'canonical_main_remote_verified')}))
        except BaseException as error:
            receipt.update(state='stopped_on_error', error=str(error), exit_status=1, stopped_utc=utc())
            save(RECEIPT, receipt)
            raise


def main():
    require(HERE == ROOT / 'preparation/publication_recovery_01', 'Noncanonical recovery directory')
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[key] = '1'
    os.environ.update(CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1', GIT_TERMINAL_PROMPT='0')
    if hasattr(os, 'sched_getaffinity'):
        os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:2])
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('plan', 'publish', 'continue-staged'))
    args = parser.parse_args()
    if args.action == 'plan':
        plan()
    else:
        publish(resume_staged=args.action == 'continue-staged')


if __name__ == '__main__':
    main()

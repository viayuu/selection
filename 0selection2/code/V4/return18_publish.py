"""CPU-only completed-version publication: read-only by default, explicit --apply.

No finish/inference/training commands are invoked. Git is only run with --apply.
"""

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace


V4 = Path(__file__).resolve().parent
REPO, GIT_ROOT = V4.parents[1], Path('/public/home/shiys')
RETURN18 = V4 / 'runs/Return18_post_R120'
CODE = {
    'R122': 'r122_experiment.py r122_objective.py r122_finish.py r122_test_recovery.py '
            'test_r122_experiment.py test_r122_objective.py test_r122_finish.py test_r122_test_recovery.py',
    'R123': 'r123_experiment.py r123_model.py r123_meta_data.py r123_prediction_cache.py r123_finish.py '
            'test_r123_experiment.py test_r123_model.py test_r123_meta_data.py test_r123_finish.py',
    'R124': 'r124_experiment.py r124_model.py r124_finish.py r123_meta_data.py r123_prediction_cache.py '
            'test_r124_experiment.py test_r124_model.py test_r124_finish.py',
}
COMMON = 'return18_publish.py test_return18_publish.py return18_launch.py test_return18_launch.py run_v4_return18.sh r120_closeout.py'
SUFFIXES = {'.py', '.sh', '.md', '.pdf', '.csv', '.json'}


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def completion_gate(root, experiment):
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    finish = importlib.import_module('code.V4.' + experiment.lower() + '_finish')
    run, extras = root / 'model', []
    if experiment == 'R122':
        model = finish.locked_inputs(root, {})
        finish.load_libs(model, {})
        lock, output, path = model['lock'], run / 'full_model_statistics', run / 'report_complete.json'
        receipt, protocol = read(path), read(output / 'protocol.json')
        finish.same_identity(receipt, lock)
        if (receipt['report_protocol_sha256'] != sha(output / 'protocol.json')
                or protocol['checkpoint_lock'] != lock or protocol['outputs'] != receipt['outputs']):
            raise ValueError('R122 report protocol/unique VAL lock changed')
        if any(sha(V4 / name) != digest for name, digest in protocol['source_sha256'].items()):
            raise ValueError('R122 report producer source changed')
        artifacts = {output / name: digest for name, digest in receipt['outputs'].items()}
        extras = [output / 'protocol.json', run / 'lib_stage_complete.json']
        recovery = run / 'test_recovery_v1/resource_adapter.json'
        if recovery.exists():
            if read(recovery).get('complete') is not True or read(recovery)['checkpoint_lock'] != lock:
                raise ValueError('R122 resource recovery is not complete for this lock')
            extras.append(recovery)
    elif experiment == 'R123':
        model = finish.locked(SimpleNamespace(root=root))
        lock, output = model['lock'], root / 'finish/reports'
        for relative, kind in (('captures/train', 'metaTRAIN_cached_chosen_scores'),
                               ('lib', 'live_sparse_LIB'), ('timing', 'live_sparse_TEST_fixed_prefix')):
            stage = root / 'finish' / relative / 'receipt.json'
            finish.require_identity(stage, model, kind)
            extras.append(stage)
        path = output / 'receipt.json'
        receipt = finish.require_identity(path, model, 'complete_post_TEST_report')
        if receipt['input_files_sha256'].get(str(Path(finish.__file__).resolve())) != sha(finish.__file__):
            raise ValueError('R123 report producer source changed')
        artifacts = {output / name: digest for name, digest in receipt['artifacts'].items()}
    else:
        run, _, lock, _, identity = finish.locked_inputs(SimpleNamespace(root=root))
        output, path = run / 'finish/reports', run / 'finish/reports/finish_receipt.json'
        receipt, lib = read(path), read(run / 'finish/lib_complete.json')
        if receipt['identity'] != identity or lib.get('complete') is not True or lib['identity'] != identity:
            raise ValueError('R124 report/LIB differs from the unique VAL/TEST identity')
        for dataset in ('TSPLIB', 'CVRPLIB'):
            value = lib['datasets'][dataset]
            if (not value['complete'] or any(value['protocol'][k] != v for k, v in identity.items())
                    or sha(run / 'finish/lib_evaluation/predictions' / (dataset + '.npz')) != value['predictions_sha256']):
                raise ValueError('R124 locked LIB capture changed')
        if any(sha(v['path']) != v['sha256'] for v in receipt['inputs'].values()):
            raise ValueError('R124 report prediction inputs changed')
        artifacts = {Path(value): receipt['output_sha256'][key] for key, value in receipt['outputs'].items()}
        extras = [run / 'finish/lib_complete.json', run / 'finish/identity.json']
    if receipt.get('complete') is not True or not artifacts:
        raise ValueError('Final report is incomplete; never start/retry missing stages here')
    if not {'.pdf', '.csv'} <= {p.suffix for p in artifacts}:
        raise ValueError('Completed report requires both PDF and CSV')
    for artifact, digest in artifacts.items():
        if artifact.resolve().parent != output.resolve() or sha(artifact) != digest:
            raise ValueError('Report artifact path/hash changed: ' + str(artifact))
    return lock, [path, *artifacts, *extras]


def curated_paths(root, experiment, reports):
    files = [*reports, *(V4 / name for name in (CODE[experiment] + ' ' + COMMON).split())]
    small_manifests = {root / 'preparation/manifest.json'} if experiment == 'R123' else set()
    files.extend(small_manifests)
    for name in ('protocol.json', 'training_complete.json', 'final_test_lock.json', 'test_results.json',
                 'test_complete.json', 'final_metrics.json', 'history.json', 'val_history.json'):
        files.extend((root / 'model').glob(name))
    for directory in (root / 'preparation', root / 'preparation/v2'):
        files.extend(directory.glob('*.md'))
        for name in ('prepared.json', 'constraint_reference.json', 'method_config.json', 'frozen_contract.json',
                     'experts.json', 'learner_protocol.json', 'CPU_CHECKS.json', 'FINISH_CPU_CHECKS.json'):
            files.extend(directory.glob(name))
    for name in ('README.md', 'ASSUMPTIONS.md', 'CACHE_PORTABILITY_V2.md', 'PUBLISHING.md',
                 'plan.schema.json', 'CPU_CHECKS.json', 'CPU_CHECKS_v2.json'):
        files.extend((RETURN18 / 'dispatch').glob(name))
    for name in ('SPEC.md', 'ASSUMPTIONS.md', 'BUILD_NOTE.md'):
        files.extend((RETURN18 / 'implement-stage').glob(name))
    files.extend(p for p in (RETURN18 / 'R120_closeout').glob('*') if p.suffix in {'.md', '.pdf', '.csv'})
    files.extend((RETURN18 / 'dispatch').glob('main_approved_' + experiment.lower() + '_*.json'))
    files.extend(RETURN18 / 'R120_closeout' / name for name in ('CHECKS.json', 'evidence.json'))
    science = REPO / 'docs/research/return18_major_routes_20261030'
    files.extend(science / name for name in ('REPORT.md', 'HANDOFF.md', 'sources.csv', 'directions.csv',
        'ROUTING.json', 'routing_receipt.json', 'astra_followup.md', 'astra_followup_sources.csv'))
    result = {}
    for path in files:
        relative = path.resolve().relative_to(GIT_ROOT)
        if (path.is_symlink() or path.suffix not in SUFFIXES or path.stat().st_size >= 8 * 1024**2
                or {'wandb', 'input_cache', 'prediction_cache', 'rawinputs', 'caches'} & set(relative.parts)
                or (path.name.endswith('manifest.json') and path not in small_manifests)
                or any(word in path.name for word in ('raw_label', 'raw_input'))):
            raise ValueError('Outside the small curated publication allowlist: ' + str(path))
        result[str(relative)] = sha(path)
    return dict(sorted(result.items()))


def git(*args):
    return subprocess.run(['git', *args], cwd=GIT_ROOT, check=True, text=True, capture_output=True,
                          env=dict(os.environ, GIT_TERMINAL_PROMPT='0')).stdout.strip()


def publish(root, experiment, *, apply=False):
    root = root.resolve()
    root.relative_to(V4 / 'runs')
    lock, reports = completion_gate(root, experiment)
    files = curated_paths(root, experiment, reports)
    identity = dict(experiment=experiment, root=str(root), checkpoint_sha256=lock['checkpoint_sha256'])
    if not apply:
        return dict(identity, ready=True, applied=False, files=files)
    path = root / 'github_publication.json'
    def save(value):
        temporary = path.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(value, indent=2) + '\n')
        temporary.replace(path)
    with (GIT_ROOT / '.git/codex-publication.lock').open('a+') as claim:
        fcntl.flock(claim, fcntl.LOCK_EX)
        if Path(git('rev-parse', '--show-toplevel')) != GIT_ROOT or git('branch', '--show-current') != 'main':
            raise RuntimeError('Publication requires the declared Git root and main branch')
        if git('diff', '--cached', '--name-only'):
            raise RuntimeError('Existing staged changes preserved; nothing added')
        record = read(path) if path.exists() else {}
        if record and any(record.get(k) != v for k, v in identity.items()):
            raise RuntimeError('Existing publication belongs to another identity')
        if record.get('complete') is True:
            return record
        if record and not record.get('commit'):
            raise RuntimeError('Prior staging/commit attempt preserved; manual inspection required, no extra auto-commit')
        if record and git('rev-parse', 'HEAD') != record['commit']:
            raise RuntimeError('Failed-push commit is no longer HEAD; preserve it for inspection')
        record = record or dict(identity, complete=False, publisher_sha256=sha(__file__), files=files)
        try:
            if not record.get('commit'):
                record.update(phase='staging', before=git('rev-parse', 'HEAD'))
                if record['before'] != git('rev-parse', 'refs/remotes/origin/main'):
                    raise RuntimeError('Local unpublished history exists; do not push unrelated commits')
                save(record)
                git('add', '-f', '--', *files)
                staged = set(git('diff', '--cached', '--name-only').splitlines())
                if not staged or not staged <= files.keys() or git('rev-parse', 'HEAD') != record['before']:
                    raise RuntimeError('Concurrent/empty index changed; preserve it, never include unrelated files')
                if any(sha(GIT_ROOT / name) != files[name] for name in staged):
                    raise RuntimeError('Curated source/report changed during staging')
                record.update(phase='committing', staged=sorted(staged))
                save(record)
                git('commit', '-m', experiment.lower() + ': deliver one locked full18 model and final reports')
                record.update(commit=git('rev-parse', 'HEAD'), phase='committed', commits_for_this_version=1)
                save(record)
            record.update(phase='pushing')
            save(record)
            git('push', 'origin', 'HEAD:refs/heads/main')
            record.update(complete=True, phase='complete', push_returncode=0,
                          published_utc=datetime.now(timezone.utc).isoformat())
        except BaseException as error:
            record.setdefault('failures', []).append(dict(phase=record.get('phase'), error=str(error),
                                                        returncode=getattr(error, 'returncode', None)))
            record.update(complete=False, phase='failed')
            raise
        finally:
            save(record)
        return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--experiment', choices=tuple(CODE), required=True)
    parser.add_argument('--apply', action='store_true', help='Commit/push once; omitted means no Git calls or writes')
    args = parser.parse_args(argv)
    value = publish(args.root, args.experiment, apply=args.apply)
    print(json.dumps(value, indent=2), flush=True)
    return value


if __name__ == '__main__':
    main()

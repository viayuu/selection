"""Explicit one-shot R99 publication after disclosed postprocessing recovery.

The original failed controllers and stopped watcher stay unchanged. This is
an administrative delivery gate, not a training/inference retry or success
claim for the research goal.
"""

import argparse
import importlib.util
import json
from pathlib import Path

from . import r104_launch


REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / 'code/V4/runs/R99_global_competition_input'
LAUNCH = REPO / 'code/V4/runs/R98_parallel_dispatch'
RECOVERY = ROOT / 'preparation/postprocess_recovery'


def private_goal():
    spec = importlib.util.spec_from_file_location('code.V4._r99_recovered_publication',
                                                   REPO / 'code/V4/r98_goal.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_file(base, record):
    path = Path(record['path']).resolve()
    if not path.is_relative_to(RECOVERY.resolve()) or base.file_hash(path) != record['sha256']:
        raise RuntimeError('Changed or foreign whole-version recovery artifact')
    return base.optional(path) if path.suffix == '.json' else path


def verified_status(base, version='R99', root=None, launch_root=None, *, persist=True):
    if (version != 'R99' or Path(root or ROOT).resolve() != ROOT.resolve()
            or Path(launch_root or LAUNCH).resolve() != LAUNCH.resolve()):
        raise ValueError('Only the canonical completed R99 recovery may be published')
    current = base.optional(RECOVERY / 'recovery_complete.json')
    expected = dict(schema='r99_postprocess_recovery_v1', experiment='R99',
        complete=True, exit_status=0, state='complete', original_failure_receipts_preserved=True,
        training_rerun=False, full_test_rerun=False, feature_rebuild=False, feature_copy=False,
        decision_rule_changed=False, efficacy_reused_as_exact_online=False)
    if any(current.get(key) != value for key, value in expected.items()):
        raise RuntimeError('Missing complete same-lock, disclosed whole-version recovery')
    if set(current.get('arms', {})) != {'A', 'B'}:
        raise RuntimeError('Both original arms must be complete')
    arms = []
    for arm, capacity_arm in (('A', 'A_width256'), ('B', 'B_depth8')):
        row = r104_launch.predecessor_gate(capacity_arm)
        recovered = row.get('recovery') or {}
        receipt = current['arms'][arm]
        record = verify_file(base, receipt)
        if (row.get('complete') is not True or row.get('errors')
                or recovered.get('sha256') != receipt.get('sha256')
                or record.get('checkpoint_lock') != base.optional(ROOT / arm / 'final_test_lock.json')):
            raise RuntimeError('Per-arm recovery or original locked model no longer matches')
        arms.append(dict(arm=arm, ready=True, missing=[], checks=row['checks'],
                         preserved_original_errors=row.get('preserved_original_errors', []), recovery=recovered))
    comparison = verify_file(base, current['comparison'])
    if comparison.get('complete') is not True or comparison.get('exit_status') != 0:
        raise RuntimeError('Whole-version CPU comparison is incomplete')
    verify_file(base, comparison['comparison_pdf'])
    verify_file(base, comparison['protocol'])
    watcher = base.optional(ROOT / 'goal_watch_receipt.json')
    if watcher.get('state') != 'stopped_on_error' or watcher.get('exit_status') != 1:
        raise RuntimeError('Original stopped watcher must remain an honest failed attempt')
    result = dict(experiment='R99', phase='complete_with_disclosed_postprocess_recovery',
        complete=True, all_arms_ready=True, comparison_complete=True, arms=arms, errors=[],
        updated_utc=base.utc(), recovery_receipt=str(RECOVERY / 'recovery_complete.json'),
        recovery_sha256=base.file_hash(RECOVERY / 'recovery_complete.json'),
        original_watcher_preserved=True, original_watcher_sha256=base.file_hash(ROOT / 'goal_watch_receipt.json'),
        original_inference_equivalence=False, cached_full_TEST_is_not_exact_online_efficacy=True,
        goal_complete=False, goal_status='active_unresolved')
    if persist:
        base.save(ROOT / 'goal_progress.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--publish-github', action='store_true')
    args = parser.parse_args()
    base = private_goal()
    original_paths = base.publication_paths

    def paths(version, root, launch_root, git_root):
        result = original_paths(version, root, launch_root, git_root)
        extra = ROOT / '.gitattributes'
        return sorted(set(result + [str(extra.relative_to(git_root))]))

    base.status = lambda version='R99', root=None, launch_root=None, **kwargs: verified_status(
        base, version, root, launch_root, **kwargs)
    base.publication_paths = paths
    result = verified_status(base)
    if args.publish_github:
        result['publication'] = base.publish('R99', ROOT, LAUNCH)
    print(json.dumps({key: value for key, value in result.items() if key != 'arms'}, ensure_ascii=False))


if __name__ == '__main__':
    main()

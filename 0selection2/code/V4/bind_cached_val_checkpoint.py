"""Late-bind a completed run's cached VAL summary to its unique final lock.

The original capture-time VAL identity and all prediction payloads stay intact.
Only a null file-hash field in final_metrics.json is completed, with a backup
and an explicit receipt. No model, metric, inference, or selection is rerun.
"""

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
RUNS = REPO / 'code/V4/runs'
DISPATCH = RUNS / 'R98_parallel_dispatch'
VERSIONS = {'R99': 'R99_global_competition_input', 'R101': 'R101_hint_pretraining'}


def digest(payload):
    return hashlib.sha256(payload).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, indent=2) + '\n').encode('utf-8')


def bind(version, arm, run=None):
    if version not in VERSIONS or arm not in ('A', 'B'):
        raise ValueError('Only completed R99/R101 arms are supported')
    root = Path(run).resolve() if run else RUNS / VERSIONS[version] / arm
    if not root.is_relative_to(RUNS.resolve()) or root.is_symlink():
        raise ValueError('Expected an ordinary run directory inside this repository')
    final_path = root / 'final_metrics.json'
    original = final_path.read_bytes()
    final = json.loads(original)
    lock = read(root / 'final_test_lock.json')
    test = read(root / 'test_results.json')
    report = read(root / 'full_model_statistics/protocol.json')
    dispatch = DISPATCH / (version + '_' + arm)
    for filename in ('worker.json', 'controller.json'):
        stage = read(dispatch / filename)
        if (stage.get('state') != 'complete' or stage.get('exit_status') != 0
                or not stage.get('stages') or any(s.get('state') != 'complete'
                    or s.get('exit_status') != 0 for s in stage['stages'])):
            raise ValueError('Wait for the original complete rc0 GPU/PDF pipeline')
    if (final.get('complete') is not True or test.get('complete') is not True
            or report.get('complete') is not True or report.get('checkpoint_lock') != lock):
        raise ValueError('Require complete same-lock final results and report')
    sha = lock.get('checkpoint_sha256')
    state = lock.get('state_hash')
    update = lock.get('selected_update')
    if not sha or not state or not isinstance(update, int) or update <= 0:
        raise ValueError('Unique checkpoint lock is incomplete')
    for row in (final, test):
        if any(row.get(k) != lock.get(k) for k in
               ('checkpoint_sha256', 'state_hash', 'selected_update')):
            raise ValueError('Final TRAIN/TEST summaries disagree with the lock')
    val = final['val']
    ids = [val['identity'], final['train']['identity'], test['result']['identity']]
    model_sha = ids[0].get('model_state_sha256')
    for identity, split in zip(ids, ('val', 'train', 'test')):
        if (identity.get('experiment') != version or identity.get('arm') != arm
                or identity.get('split') != split or identity.get('successful_updates') != update
                or identity.get('state_hash') != state or not model_sha
                or identity.get('model_state_sha256') != model_sha):
            raise ValueError('Capture-time VAL, final TRAIN and TEST are not the same model')
    if any(i.get('checkpoint_sha256') != sha for i in ids[1:]):
        raise ValueError('Final TRAIN and TEST do not bind the actual checkpoint file')
    problems = val.get('per_problem', {})
    if (len(problems) != 18 or set(problems) != set(test['result']['per_problem'])
            or any(p.get('n') != 4500 for p in problems.values())):
        raise ValueError('Require full18 cached VAL with 4500 rows per task')
    cached_path = root / val['prediction_archive'] / 'identity.json'
    if not cached_path.is_file():
        cached_path = root / 'evaluation/val/selected_best/identity.json'
    cached = read(cached_path)
    for key in ('experiment', 'arm', 'split', 'successful_updates', 'state_hash', 'model_state_sha256'):
        if cached.get(key) != ids[0].get(key):
            raise ValueError('Original VAL archive identity does not match its summary')
    out = root.parent / 'preparation/val_metadata_binding' / arm
    receipt_path = out / 'receipt.json'
    if ids[0].get('checkpoint_sha256') == sha:
        if receipt_path.exists():
            prior = read(receipt_path)
            if prior.get('bound_final_metrics_sha256') != digest(original):
                raise ValueError('Previously bound final summary changed unexpectedly')
            return prior
        return {'complete': True, 'changed': False, 'reason': 'already bound at capture time'}
    if ids[0].get('checkpoint_sha256') is not None or cached.get('checkpoint_sha256') is not None:
        raise ValueError('Only the documented capture-time null field may be completed')
    out.mkdir(parents=True, exist_ok=True)
    backup = out / 'original_final_metrics.json'
    if backup.exists() and backup.read_bytes() != original:
        raise ValueError('Preserved original differs; no overwrite')
    if not backup.exists():
        with backup.open('xb') as stream:
            stream.write(original)
    revised = copy.deepcopy(final)
    revised['val']['identity']['checkpoint_sha256'] = sha
    check = copy.deepcopy(revised)
    check['val']['identity']['checkpoint_sha256'] = None
    if check != final:
        raise AssertionError('A field other than the null VAL checkpoint SHA was changed')
    payload = encoded(revised)
    receipt = dict(experiment=version, arm=arm, complete=True, changed=True,
        bound_utc=datetime.now(timezone.utc).isoformat(),
        reason='Cached VAL preceded final checkpoint serialization; bind by identical model state, update and completed TEST',
        field='val.identity.checkpoint_sha256', before=None, after=sha,
        original_final_metrics_sha256=digest(original), bound_final_metrics_sha256=digest(payload),
        original_backup=str(backup), cached_val_identity=str(cached_path),
        cached_val_identity_sha256=digest(cached_path.read_bytes()),
        checkpoint_lock_sha256=digest((root / 'final_test_lock.json').read_bytes()),
        model_state_sha256=model_sha, state_hash=state, selected_update=update,
        numerical_metrics_changed=False, predictions_changed=False, model_changed=False,
        inference_rerun=False, checkpoint_reselected=False, source_changed=False)
    temporary = final_path.with_name('.final_metrics.val_binding.tmp')
    temporary.write_bytes(payload)
    temporary.replace(final_path)
    receipt_path.write_bytes(encoded(receipt))
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--version', required=True, choices=tuple(VERSIONS))
    parser.add_argument('--arm', required=True, choices=('A', 'B'))
    parser.add_argument('--run', type=Path)
    args = parser.parse_args()
    print(json.dumps(bind(args.version, args.arm, args.run), ensure_ascii=False))


if __name__ == '__main__':
    main()

"""Bind Git-normalized CSV bytes without changing any experiment or old receipt."""

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess

from . import research_completion_monitor as monitor


def reconcile(root, approved):
    if not approved:
        raise ValueError('Main approval is required for an additive evidence reconciliation')
    root = Path(root).resolve()
    old = root / 'evidence'
    new = root / 'evidence_csv_materialization'
    original = monitor.read(old / 'verification.json')
    publication = monitor.read(old / 'publication_byte_verification.json')
    snapshot = monitor.read(old / 'source_snapshot.json')
    action = monitor.read(root / 'state.json')['actions']['verify-results']
    if (action['state'] != 'needs_action' or original.get('complete') is not False
            or original.get('publication_byte_mismatch_count') != 92
            or not (Path(action['attempt']) / 'exit.json').is_file()):
        raise ValueError('Only the completed strict CSV-byte review may be reconciled')
    if original['verified_inputs'] != snapshot:
        raise ValueError('Original numeric verification is not bound to this source snapshot')
    for path, expected in snapshot.items():
        if monitor.digest(path) != expected:
            raise ValueError('Scientific input changed: ' + path)
    for name in ('evidence_pack.md', 'metrics.csv', 'checkpoints.csv'):
        if monitor.digest(old / name) != original['outputs'][name]:
            raise ValueError('Original Sol evidence output changed: ' + name)
    rows = []
    git_root = monitor.REPO.parent
    for row in publication['mismatches']:
        if not row['path'].endswith('.csv') or row.get('CRLF_to_LF_only') is not True:
            raise ValueError('Not a CSV line-ending-only discrepancy')
        work = (git_root / row['path']).read_bytes()
        blob = subprocess.check_output(['git', 'cat-file', 'blob', row['commit'] + ':' + row['path']],
                                       cwd=git_root)
        sha = lambda content: hashlib.sha256(content).hexdigest()
        if (sha(work) != row['working_sha256'] or sha(blob) != row['git_blob_sha256']
                or work.replace(b'\r\n', b'\n') != blob
                or list(csv.reader(io.StringIO(work.decode('utf-8'))))
                != list(csv.reader(io.StringIO(blob.decode('utf-8'))))):
            raise ValueError('CSV contains a material or unrecorded change: ' + row['path'])
        rows.append(dict(row, parsed_cells_identical=True, verified_now=True))
    if len(rows) != 92:
        raise ValueError('The supplied discrepancy set changed')
    new.mkdir(parents=True, exist_ok=False)
    proof = dict(complete=True, main_approved=True, kind='git_csv_eol_only',
        timestamp_utc=monitor.utc(), discrepancies=len(rows), records=rows,
        strict_byte_identity=False, parsed_content_equivalence=True,
        original_publication_check=dict(path=str(old / 'publication_byte_verification.json'),
            sha256=monitor.digest(old / 'publication_byte_verification.json')),
        original_numeric_verification=dict(path=str(old / 'verification.json'),
            sha256=monitor.digest(old / 'verification.json')),
        original_action='verify-results', original_action_preserved=True,
        original_sources_and_receipts_modified=False, git_mutations=False,
        scientific_metrics_changed=False, model_execution=False)
    monitor.save(new / 'publication_csv_materialization.json', proof)
    for name in ('checkpoints.csv', 'metrics.csv', 'source_snapshot.json',
                 'publication_byte_verification.json', 'saved_results_authentication.json'):
        shutil.copyfile(old / name, new / name)
    body = (old / 'evidence_pack.md').read_text(encoding='utf-8').split('## 实际完成与保留的失败', 1)[1]
    body = body.replace('当前有限核验受阻既不关闭目标，也不授权新的arm、求解或发布。',
                        '本次换行对账不关闭目标；后续仍按已有研究授权推进。')
    header = ('# 保存结果证据包：数值核验完成，CSV 发布格式已对账\n\n'
        '七个唯一 VAL checkpoint 和280行指标沿用原 Sol/max 的实际核验，数值与科学结论未改。'
        '主线程另行确认92个CSV差异全部仅为CRLF到LF，实际Git blob与规范化工作文件相等，'
        '逐单元格解析内容也完全一致。严格字节SHA仍不同，不把它写成相同。\n\n'
        '附加对账见 `publication_csv_materialization.json`；原严格核验、原发布清单、失败动作、'
        'checkpoint、预测、CSV和提交全部保留。当前完成口径是科学内容一致并明确记录发布格式转换，'
        '不是谎称原严格字节检查通过。不重训、不重测、不重新选型，也不为换行重复提交。\n\n'
        '## 实际完成与保留的失败')
    (new / 'evidence_pack.md').write_text(header + body, encoding='utf-8')
    verification = dict(original, complete=True, status='complete', blocking_reason=None,
        required_action=None, completion_authority='main_csv_materialization_reconciliation',
        original_review_was_strict_byte_blocked=True,
        strict_published_bytes_pass=False, publication_content_equivalence_pass=True,
        original_verification=proof['original_numeric_verification'],
        publication_materialization=dict(path=str(new / 'publication_csv_materialization.json'),
            sha256=monitor.digest(new / 'publication_csv_materialization.json')))
    verification['outputs'] = {name: monitor.digest(new / name)
        for name in ('evidence_pack.md', 'checkpoints.csv', 'metrics.csv')}
    verification['negative_partial_facts'] = [dict(fact=
        '92 CSV published/workspace byte hashes differ only by documented CRLF-to-LF; parsed numeric content is identical; original strict byte failure remains preserved.',
        sources=[str(new / 'publication_csv_materialization.json')]), *original['negative_partial_facts'][1:]]
    monitor.save(new / 'verification.json', verification)
    authority = dict(main_approved=True, kind='git_csv_eol_only', evidence_dir=str(new),
        original_action='verify-results', original_action_preserved=True,
        original_verification_sha256=monitor.digest(old / 'verification.json'),
        original_metrics_sha256=monitor.digest(old / 'metrics.csv'),
        original_checkpoints_sha256=monitor.digest(old / 'checkpoints.csv'),
        source_snapshot_sha256=monitor.json_hash(snapshot),
        verification_sha256=monitor.digest(new / 'verification.json'),
        normalization_sha256=monitor.digest(new / 'publication_csv_materialization.json'),
        generator=str(Path(__file__).resolve()), generator_sha256=monitor.digest(__file__))
    monitor.save(root / 'evidence_csv_reconciliation.json', authority)
    return dict(complete=True, evidence_dir=str(new), normalized_files=len(rows),
                retraining=False, git_mutations=False, original_failure_preserved=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=monitor.REPO / monitor.RUN_REL)
    parser.add_argument('--main-approved', action='store_true')
    args = parser.parse_args()
    print(json.dumps(reconcile(args.root, args.main_approved), indent=2))


if __name__ == '__main__':
    main()

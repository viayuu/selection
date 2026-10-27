"""Mechanical PDF delivery after the scientific agent actually completes."""

import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
TEMPLATE = ROOT.parent / 'major_paradigm_review_20261025/render_report.py'
SCIENTIFIC = ('REPORT.md', 'sources.csv', 'directions.csv')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def renderer():
    spec = importlib.util.spec_from_file_location('early_graph_context_layout', TEMPLATE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.ROOT = ROOT
    module.previous.MATH_DIR = ROOT / 'formula_assets'
    return module


def render():
    routing = json.loads((ROOT / 'ROUTING.json').read_text())
    if (routing.get('phase') != 'research_complete' or routing.get('requested_model') != 'gpt-6-astra'
            or routing.get('requested_reasoning_effort') != 'max'):
        raise RuntimeError('Render only after an actual Astra completion notification')
    before = {name: sha(ROOT / name) for name in SCIENTIFIC}
    layout = renderer()
    counts = {name: len(layout.read_csv(ROOT / name)[1]) for name in SCIENTIFIC[1:]}
    if not all(counts.values()):
        raise ValueError('Scientific CSVs must be nonempty')
    with (ROOT / 'sources.csv').open(newline='', encoding='utf-8') as stream:
        primary_papers = sum(row.get('source_type') == 'primary_paper' for row in csv.DictReader(stream))
    original_document = layout.document

    def furniture(canvas, doc):
        width, height = canvas._pagesize
        canvas.saveState()
        for text, y in (('SOLVER SELECTION / 早期图上下文的科学依据', height - 11 * layout.mm),
                        ('2026-10-27 | Astra 科学正文 / Sol 机械排版', 9.5 * layout.mm)):
            item = layout.paragraph(text, 'SmallCN')
            item.wrap(width - 2 * layout.MARGIN - 30, 20)
            item.drawOn(canvas, layout.MARGIN, y)
        canvas.setFont('Latin', 8)
        canvas.drawRightString(width - layout.MARGIN, 9.5 * layout.mm, str(doc.page))
        canvas.restoreState()

    def document(target):
        result = original_document(target)
        result.title = '历史TRAIN上下文参与图结构形成：科学依据与边界'
        result.author = 'gpt-6-astra scientific research; gpt-6.1-sol mechanical production'
        result.subject = ROOT.name
        return result

    layout.document, layout.furniture = document, furniture
    pdf = layout.build('REPORT.pdf', ['REPORT.md'])
    if layout.previous.MATH_WARNINGS:
        raise ValueError('Mechanical formula correction required; do not rewrite the scientific source')
    if before != {name: sha(ROOT / name) for name in SCIENTIFIC}:
        raise RuntimeError('Scientific sources changed during rendering')
    notes = dict(complete=True, scientific_source_changed=False, inputs=before,
        actual_agent_id=routing['research_agent_id'], requested_scientific_model='gpt-6-astra',
        requested_scientific_reasoning_effort='max', rendering_role='Sol mechanical only',
        template=dict(path=str(TEMPLATE), sha256=sha(TEMPLATE)), source_csv_rows=counts,
        primary_papers=primary_papers, pdf=pdf, unsupported_glyphs=sorted(layout.UNSUPPORTED_GLYPHS),
        layout_warnings=layout.LAYOUT_WARNINGS, generator_sha256=sha(__file__))
    (ROOT / 'render_notes.json').write_text(json.dumps(notes, ensure_ascii=False, indent=2) + '\n')
    (ROOT / 'DELIVERY.md').write_text(
        '# Scientific Follow-Up Delivery\n\n'
        'Astra/max authored the scientific text and source tables; Sol mechanically rendered them unchanged.\n\n'
        f"REPORT.pdf: {pdf['pages']} pages; sources.csv: {counts['sources.csv']} records "
        f"({primary_papers} primary papers; {counts['sources.csv'] - primary_papers} local project sources).\n\n"
        'This is not a new experiment result or an operational experiment plan. R117 was still running\n'
        'when this research was requested; no interim R117 score establishes its final outcome.\n'
        'Future finite protocols, implementation and execution remain Sol responsibilities.\n'
        'The overall zero-selection and whole18 improvement goal remains ACTIVE.\n')
    print(json.dumps(notes, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', action='store_true')
    args = parser.parse_args()
    if args.prepare:
        value = renderer()
        print(json.dumps(dict(layout_imported=True, font=str(value.layout.FONT), scientific_drafts_read=False)))
    else:
        render()

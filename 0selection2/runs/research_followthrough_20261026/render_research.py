"""Mechanical, CPU-only rendering of the immutable Astra research handoff."""

import csv
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
ACTION = 'f7ece9228eec9e192e59a6bb'
REQUEST = ROOT / 'actions' / ACTION / 'request.json'


def sha(payload):
    return hashlib.sha256(payload).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def check(condition, reason):
    if not condition:
        raise ValueError(reason)


def active():
    path = ROOT / 'control.json'
    check(not path.exists() or read(path).get('paused') is not True,
          'New explicit monitor pause: stop this publication continuation')


def verify_inputs():
    active()
    request = read(REQUEST)
    check((request['action_id'], request['model'], request['reasoning_effort']) ==
          (ACTION, 'gpt-6.1-sol', 'max'), 'Invocation identity mismatch')
    context = request['context']
    folder = Path(context['research_dir'])
    for path, expected in context['research_hashes'].items():
        check(sha(Path(path).read_bytes().replace(b'\r\n', b'\n')) == expected,
              'Scientific handoff changed: ' + path)
    snapshot = read(context['source_snapshot'])
    canonical = json.dumps(snapshot, sort_keys=True, separators=(',', ':')).encode()
    check(sha(canonical) == context['source_snapshot_sha256'], 'Snapshot identity changed')
    for path, expected in snapshot.items():
        check(sha(Path(path).read_bytes()) == expected, 'Source evidence changed: ' + path)
    evidence = Path(context['evidence_dir'])
    verification = read(evidence / 'verification.json')
    check(verification['complete'] is True and verification['blocking_reason'] is None,
          'Scientific evidence gate is incomplete or blocked')
    check(verification.get('publication_content_equivalence_pass') is True and
          verification.get('completion_authority') == 'main_csv_materialization_reconciliation',
          'Missing additive CSV materialization authority')
    for name, expected in verification['outputs'].items():
        check(sha((evidence / name).read_bytes()) == expected, 'Evidence output changed: ' + name)
    authority = read(ROOT / 'evidence_csv_reconciliation.json')
    for key, path in (
        ('verification_sha256', evidence / 'verification.json'),
        ('normalization_sha256', evidence / 'publication_csv_materialization.json'),
        ('original_verification_sha256', ROOT / 'evidence/verification.json'),
        ('generator_sha256', Path(authority['generator'])),
    ):
        check(sha(path.read_bytes()) == authority[key], 'CSV authority changed: ' + str(path))
    check(authority['main_approved'] is True, 'CSV reconciliation lacks main authority')
    for name, count in (('sources.csv', 10), ('directions.csv', 4)):
        rows = list(csv.reader(io.StringIO((folder / name).read_text(encoding='utf-8'))))
        check(len(rows) == count + 1 and all(len(r) == len(rows[0]) for r in rows),
              'Invalid research CSV: ' + name)
    return request, dict(research_hashes=context['research_hashes'],
                        source_snapshot_sha256=sha(canonical),
                        source_snapshot_file_sha256=sha(Path(context['source_snapshot']).read_bytes()),
                        immutable_source_files=len(snapshot), evidence_complete=True,
                        csv_materialization_authority_sha256=sha((ROOT / 'evidence_csv_reconciliation.json').read_bytes()))


def load_renderer():
    sys.path.insert(0, '/tmp/shiys_astra_pdf_dependencies_compat')
    source = REPO / 'docs/research/major_paradigm_review_20261025/render_report.py'
    spec = importlib.util.spec_from_file_location('saved_research_renderer', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_rendered_text(renderer, text, payload):
    extracted = subprocess.check_output(['pdftotext', '-raw', '-', '-'], input=payload).decode('utf-8')
    compact = lambda value: re.sub(r'\s+', '', value)
    pages = []
    for page in extracted.split('\f'):
        lines = page.splitlines()
        if not lines:
            continue
        check(lines[0] == 'SOLVER SELECTION / 完整结果之后的研究', 'Unexpected PDF page identity')
        check(lines[1].startswith('2026-10-26 | Astra 科学正文 / Sol 机械排版 '),
              'Unexpected PDF date/author identity')
        pages.append('\n'.join(lines[2:]))
    actual = compact('\n'.join(pages))
    checked = 0
    formula = False
    for line in text.splitlines():
        line = line.strip()
        if line.startswith(r'\['):
            formula = True
            continue
        if formula:
            if line.startswith(r'\]'):
                formula = False
            continue
        if not line:
            continue
        if line.startswith('|'):
            cells = renderer.table_cells(line)
            if all(re.fullmatch(r':?-+:?', x.replace(' ', '')) for x in cells):
                continue
        else:
            cells = [re.sub(r'^#{1,6}\s+', '', line)]
        for cell in cells:
            plain = re.sub(r'\[([^\]]+)\]\((?:[^()\n]|\([^()\n]*\))*\)', r'\1', cell)
            plain = plain.replace('**', '').replace('`', '')
            check(compact(plain) in actual, 'PDF lost source text/table cell: ' + plain[:140])
            checked += 1
    expressions = re.findall(r'\\\[(.*?)\\\]', text, re.S)
    for expression in expressions:
        check(compact(expression) in actual, 'PDF lost formula source: ' + expression[:100])
    return dict(source_text_fragments_checked=checked, all_source_table_cells_present=True,
                complete_prose_round_trip=True, formula_sources_checked=len(expressions))


def render():
    request, inputs = verify_inputs()
    folder = Path(request['context']['research_dir'])
    renderer = load_renderer()
    renderer.ROOT = folder
    work = ROOT / 'render_work'
    work.mkdir(exist_ok=True)
    renderer.previous.MATH_DIR = work / 'formula_assets'
    font = Path(renderer.layout.FONT)
    font_info = subprocess.check_output(['fc-scan', '--format', '%{family}\n%{file}\n', str(font)]).decode()
    check('Droid Sans' in font_info and font.is_file(), 'Inspected CJK font is unavailable')

    def furniture(canvas, doc):
        width, height = canvas._pagesize
        canvas.saveState()
        for label, x, y in (
            ('SOLVER SELECTION / 完整结果之后的研究', renderer.MARGIN, height - 11 * renderer.mm),
            ('2026-10-26 | Astra 科学正文 / Sol 机械排版', renderer.MARGIN, 9.5 * renderer.mm),
        ):
            item = renderer.paragraph(label, 'SmallCN')
            item.wrap(width - 2 * renderer.MARGIN - 30, 20)
            item.drawOn(canvas, x, y)
        canvas.setStrokeColor(renderer.layout.RULE)
        for y in (height - 14 * renderer.mm, 14 * renderer.mm):
            canvas.line(renderer.MARGIN, y, width - renderer.MARGIN, y)
        canvas.setFont('Latin', 8)
        canvas.drawRightString(width - renderer.MARGIN, 9.5 * renderer.mm, str(doc.page))
        canvas.restoreState()

    # Keep exact formula source available to machine readers while displaying
    # the existing renderer's formula artwork. No mathematical rewriting.
    class FormulaImage(renderer.Image):
        def draw(self):
            encoded = ('\ufeff' + self.source_expression).encode('utf-16-be').hex()
            self.canv._code.append('/Span << /ActualText <' + encoded + '> >> BDC')
            # Poppler emits ActualText for text objects, whereas image-only
            # marked content is skipped. This invisible anchor carries the
            # exact formula source without adding visible prose or glyphs.
            anchor = self.canv.beginText(0, 0)
            anchor.setFont('Latin', 1)
            anchor.setTextRenderMode(3)
            anchor.textOut('formula')
            self.canv.drawText(anchor)
            super().draw()
            self.canv._code.append('EMC')

    def display_formula(expression):
        result = []
        for line in re.split(r'\\\\(?:\[[^]]*\])?', expression):
            if not line.strip():
                continue
            rendered = renderer.render_formula(line.replace('&', ' '))
            check(rendered is not None, 'Formula rendering failed: ' + line)
            path, width, height = rendered
            item = FormulaImage(str(path), width=width, height=height)
            item.source_expression = line.strip()
            result.extend([renderer.Spacer(1, 4), item, renderer.Spacer(1, 7)])
        return result

    original_document = renderer.document

    def document(target):
        doc = original_document(target)
        doc.title = '完整结果之后：选择器平台期、响应学习缺口与下一轮大方向'
        doc.author = 'gpt-6-astra research; gpt-6.1-sol mechanical rendering'
        doc.subject = 'post_major_paradigm_results_20261026; immutable research handoff'
        return doc

    renderer.furniture = furniture
    renderer.display_formula = display_formula
    renderer.document = document
    result = renderer.build('REPORT.pdf', ['REPORT.md'])
    check(not renderer.previous.MATH_WARNINGS, 'Formula warning requires correction')
    payload = (folder / 'REPORT.pdf').read_bytes()
    text_checks = verify_rendered_text(renderer, (folder / 'REPORT.md').read_text(), payload)
    verify_inputs()
    notes = dict(action_id=ACTION, model='gpt-6.1-sol', reasoning_effort='max',
                 report_version=folder.name, scientific_text_changed=False,
                 font=dict(path=str(font), sha256=sha(font.read_bytes()), fc_scan=font_info),
                 reused_renderer=str(REPO / 'docs/research/major_paradigm_review_20261025/render_report.py'),
                 renderer_sha256=sha((REPO / 'docs/research/major_paradigm_review_20261025/render_report.py').read_bytes()),
                 reportlab_version=renderer.REPORTLAB_VERSION,
                 pdf=result, text_checks=text_checks, inputs=inputs,
                 unsupported_glyphs=sorted(renderer.UNSUPPORTED_GLYPHS),
                 formula_warnings=renderer.previous.MATH_WARNINGS,
                 formula_source_actualtext=True,
                 pdfinfo=subprocess.check_output(['pdfinfo', str(folder / 'REPORT.pdf')]).decode(),
                 pdffonts=subprocess.check_output(['pdffonts', str(folder / 'REPORT.pdf')]).decode())
    (ROOT / 'render_notes.json').write_text(json.dumps(notes, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(dict(pdf=result, text_checks=text_checks), ensure_ascii=False))


if __name__ == '__main__':
    render()

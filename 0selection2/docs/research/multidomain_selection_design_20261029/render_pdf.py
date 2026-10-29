"""Mechanical PDF rendering of completed Astra documents."""

import hashlib
import importlib.util
import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parent
TEMPLATE = ROOT.parent / 'major_paradigm_review_20261025/render_report.py'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    route = json.loads((ROOT / 'ROUTING.json').read_text())
    if route['phase'] != 'research_complete' or route['requested_model'] != 'gpt-6-astra':
        raise ValueError('Require the completed requested Astra research')
    sources = ('REPORT.md', 'HANDOFF.md', 'sources.csv', 'directions.csv')
    before = {name: digest(ROOT / name) for name in sources}
    spec = importlib.util.spec_from_file_location('multidomain_pdf_layout', TEMPLATE)
    layout = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(layout)
    layout.ROOT = ROOT
    layout.previous.MATH_DIR = ROOT / 'formula_assets'
    original_formula = layout.render_formula
    original_display = layout.display_formula

    def formula(expression, *, inline=False):
        normalized = re.sub(r'\\(?:big|Big|bigg|Bigg)[lr]?(?![A-Za-z])', '', expression)
        normalized = re.sub(r'\\sqrt\s+([A-Za-z0-9])', r'\\sqrt{\1}', normalized)
        return original_formula(normalized, inline=inline)

    layout.render_formula = formula

    def display(expression):
        if r'\begin{cases}' not in expression:
            return original_display(expression)
        prefix, cases = expression.split(r'\begin{cases}', 1)
        body, suffix = cases.split(r'\end{cases}', 1)
        result = []
        for row in re.split(r'\\\\', body):
            if row.strip():
                # Repeat the same left-hand side on each conditional row.
                result.extend(original_display(prefix + row.replace('&', r'\quad ')))
        if suffix.strip():
            result.extend(original_display(suffix))
        return result

    layout.display_formula = display

    def furniture(canvas, doc):
        width, height = canvas._pagesize
        canvas.saveState()
        for text, y in (
            ('MULTIDOMAIN SOLVER SELECTION / SCIENTIFIC DESIGN', height - 11 * layout.mm),
            ('2026-10-29 | Astra science / Sol mechanical production', 9.5 * layout.mm),
        ):
            item = layout.paragraph(text, 'SmallCN')
            item.wrap(width - 2 * layout.MARGIN - 30, 20)
            item.drawOn(canvas, layout.MARGIN, y)
        canvas.setFont('Latin', 8)
        canvas.drawRightString(width - layout.MARGIN, 9.5 * layout.mm, str(doc.page))
        canvas.restoreState()

    layout.furniture = furniture
    output = [layout.build('REPORT.pdf', ['REPORT.md']), layout.build('HANDOFF.pdf', ['HANDOFF.md'])]
    if before != {name: digest(ROOT / name) for name in sources}:
        raise RuntimeError('Scientific sources changed during rendering')
    note = dict(complete=not bool(layout.previous.MATH_WARNINGS), sources=before,
                outputs=output, math_warnings=layout.previous.MATH_WARNINGS,
                requested_model=route['requested_model'], requested_reasoning_effort='max',
                scientific_sources_changed=False)
    (ROOT / 'pdf_build.json').write_text(json.dumps(note, ensure_ascii=False, indent=2) + '\n')
    if layout.previous.MATH_WARNINGS:
        raise RuntimeError(str(layout.previous.MATH_WARNINGS))
    print(json.dumps(note, ensure_ascii=False))


if __name__ == '__main__':
    main()

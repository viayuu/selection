"""Render the Astra synthesis and appendices without touching experiments.

Run in easynco with the already available ReportLab target on PYTHONPATH:
PYTHONPATH=/tmp/shiys_astra_pdf_dependencies_compat python <this script> appendices
Use finalmain only after the user confirms the main report is ready.
"""

from __future__ import annotations

import importlib.util
import argparse
import csv
import hashlib
import json
import re
import unicodedata
from pathlib import Path

from matplotlib import mathtext
from matplotlib.font_manager import FontProperties
from PIL import Image as PILImage
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer

ROOT = Path(__file__).resolve().parent
PREVIOUS = ROOT.parent / "deep_direction_review_v2_20261023/render_report.py"
spec = importlib.util.spec_from_file_location("previous_research_pdf", PREVIOUS)
layout = importlib.util.module_from_spec(spec)
spec.loader.exec_module(layout)
ORIGINAL_INLINE = layout.inline
MATH_DIR = ROOT / "formula_assets"
MATH_WARNINGS = []
INLINE_LIMIT = layout.WIDTH * 0.78
ORIGINAL_TABLE = layout.table


def formula(expression, *, inline=False):
    original_expression = expression
    expression = re.sub(r"\\tag\{[^}]*\}", "", expression).strip()
    expression = re.sub(r"\\mathbf\s*1", r"\\mathbf{1}", expression)
    expression = " ".join(expression.split())
    expression = re.sub(r"\\(mathcal|mathbb|mathbf|mathrm|mathsf)\s+([A-Za-z0-9])",
                        lambda m: "\\" + m[1] + "{" + m[2] + "}", expression)
    expression = re.sub(r"\\(le|ge)(?![A-Za-z])", lambda m: "\\" + m[1] + "q", expression)
    expression = expression.replace(r"\tfrac", r"\frac").replace(r"\dfrac", r"\frac")
    expression = re.sub(r"\\frac\s*([A-Za-z0-9])\s*([A-Za-z0-9])",
                        lambda m: r"\frac{" + m[1] + "}{" + m[2] + "}", expression)
    expression = re.sub(r"\\frac\s*([A-Za-z0-9])\s*\{",
                        lambda m: r"\frac{" + m[1] + "}{", expression)
    key = hashlib.sha256((str(inline) + expression).encode()).hexdigest()[:20]
    path = MATH_DIR / (key + ".png")
    try:
        if not path.is_file():
            MATH_DIR.mkdir(exist_ok=True)
            mathtext.math_to_image("$" + expression + "$", str(path), dpi=220,
                prop=FontProperties(size=9 if inline else 11), format="png",
                color="#24313B")
        with PILImage.open(path) as raster:
            width, height = (value * 72 / 220 for value in raster.size)
        limit = INLINE_LIMIT if inline else layout.WIDTH * 0.98
        scale = min(1, limit / max(width, 1))
        return path, width * scale, height * scale
    except (ValueError, RuntimeError) as error:
        MATH_WARNINGS.append(dict(original_expression=original_expression,
                                  expression=expression, reason=str(error)))
        return None


def inline_math(text):
    images = []

    def replace(match):
        rendered = formula(match[1], inline=True)
        if rendered is None:
            return "[公式原文（排版回退）：" + match[1] + "]"
        path, width, height = rendered
        marker = "FORMULAPLACEHOLDER" + str(len(images)) + "END"
        images.append((marker, '<img src="%s" width="%.2f" height="%.2f" valign="-2"/>'
                       % (path, width, height)))
        return marker

    output = ORIGINAL_INLINE(re.sub(r"\\\((.*?)\\\)", replace, text))
    for marker, image in images:
        output = output.replace(marker, image)
    return output


layout.inline = inline_math


def paragraph(text, style="BodyCN"):
    content = inline_math(text)
    selected = layout.STYLES[style]
    if "<img " in content:
        selected = ParagraphStyle("Math" + style, parent=selected, wordWrap=None, autoLeading="max")
    return Paragraph(content, selected)


def table(rows):
    global INLINE_LIMIT
    original = INLINE_LIMIT
    INLINE_LIMIT = layout.WIDTH / len(rows[0]) * 0.68 - 12
    try:
        return ORIGINAL_TABLE(rows)
    finally:
        INLINE_LIMIT = original


layout.para = paragraph
layout.table = table


def markdown_text(text):
    references = dict(re.findall(r"^\[([^]]+)\]:\s*(\S+).*?$", text, re.MULTILINE))
    text = re.sub(r"^\[([^]]+)\]:\s*(\S+).*?$", "", text, flags=re.MULTILINE)
    text = re.sub(r"\[([^]]+)\]\[([^]]+)\]",
                  lambda m: "[%s](%s)" % (m[1], references[m[2]])
                  if m[2] in references else m[0], text)
    return re.sub(r"^#{4,6} ", "### ", text, flags=re.MULTILINE)


def parse(text):
    story = []
    # Formula images are embedded into the PDF, never external report dependencies.
    parts = re.split(r"(\\\[.*?\\\]|\$\$.*?\$\$)", markdown_text(text), flags=re.DOTALL)
    for part in parts:
        if part.startswith((r"\[", "$$")):
            expression = part[2:-2]
            expression = re.sub(r"\\(?:begin|end)\{(?:aligned|gathered)\}", "", expression)
            lines = re.split(r"\\\\(?:\[[^]]*\])?", expression)
            for line in lines:
                rendered = formula(line.replace("&", " "))
                if rendered:
                    path, width, height = rendered
                    story.extend([Spacer(1, 4), Image(str(path), width=width, height=height), Spacer(1, 7)])
                elif line.strip():
                    story.append(layout.para("公式排版回退：以下保留原式，未改数学含义。", "SmallCN"))
                    story.append(layout.para(line.strip(), "CodeCN"))
        elif part.strip():
            story.extend(layout.parse(part))
    return story


def furniture(canvas, doc):
    canvas.saveState()
    canvas.setFillColor(layout.MUTED)
    header = layout.para("SOLVER SELECTION / Astra 全面研究 III", "SmallCN")
    header.wrap(layout.WIDTH, 20)
    header.drawOn(canvas, layout.MARGIN, layout.PAGE_H - 11 * mm)
    canvas.setStrokeColor(layout.RULE)
    canvas.line(layout.MARGIN, layout.PAGE_H - 14 * mm,
                layout.PAGE_W - layout.MARGIN, layout.PAGE_H - 14 * mm)
    canvas.line(layout.MARGIN, 14 * mm, layout.PAGE_W - layout.MARGIN, 14 * mm)
    footer = layout.para("2026-10-25 | Astra max 调研与综合 | 未改动正在运行的实验", "SmallCN")
    footer.wrap(layout.WIDTH - 20, 20)
    footer.drawOn(canvas, layout.MARGIN, 9.5 * mm)
    canvas.setFont("Latin", 8)
    canvas.drawRightString(layout.PAGE_W - layout.MARGIN, 9.5 * mm, str(doc.page))
    canvas.restoreState()


def build(output, sources):
    story = []
    for index, source in enumerate(sources):
        if index:
            story.append(PageBreak())
        text = source.read_text(encoding="utf-8")
        story.extend(parse(text))
    doc = SimpleDocTemplate(str(output), pagesize=layout.A4,
        rightMargin=layout.MARGIN, leftMargin=layout.MARGIN,
        topMargin=20 * mm, bottomMargin=20 * mm,
        title="零选择与 Top1 平台：Astra 全面研究与后续方向",
        author="gpt-6-astra/max (research); gpt-6.1-sol/max (document production)")
    doc.build(story, onFirstPage=furniture, onLaterPages=furniture)
    print(f"{output} ({doc.page} pages)")
    return dict(file=output.name, pages=doc.page, sources=[source.name for source in sources])


def source_registry():
    records = []
    for path in sorted(ROOT.glob("0[123]_sources.csv")):
        with path.open(encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                kind = ("paper" if row.get("source_type", "primary_paper") == "primary_paper"
                        else "local_evidence")
                title = " ".join(unicodedata.normalize("NFKC", row["title"]).split()).casefold()
                records.append(dict(lens=path.stem[:2], source_csv=path.name,
                                    record_kind=kind, normalized_title=title, **row))
    fields = list(dict.fromkeys(key for record in records for key in record))
    with (ROOT / "source_registry.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    papers = [record for record in records if record["record_kind"] == "paper"]
    return dict(paper_reading_records=len(papers),
                normalized_unique_paper_titles=len({record["normalized_title"] for record in papers}),
                local_evidence_records=len(records) - len(papers),
                note="Reading scopes are copied from source CSVs, not independently reviewed.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", nargs="?", choices=("appendices", "finalmain"),
                        default="appendices")
    args = parser.parse_args()
    MATH_WARNINGS.clear()
    main_report = ROOT / "REPORT.md"
    appendices = [ROOT / name for name in (
        "01_decision_and_identifiability.md", "02_representation_and_computation.md",
        "03_learning_and_optimization.md")]
    dialogue = ROOT / "SYNTHESIS_DIALOGUE.md"
    required = appendices if args.stage == "appendices" else [main_report, *appendices, dialogue]
    for source in required:
        if not source.is_file():
            raise FileNotFoundError(source)
    outputs = []
    if args.stage == "finalmain":
        outputs.append(build(ROOT / "REPORT.pdf", [main_report]))
        outputs.append(build(ROOT / "RESEARCH_FULL.pdf", [main_report, *appendices, dialogue]))
    for source in appendices:
        outputs.append(build(source.with_suffix(".pdf"), [source]))
    registry = source_registry()
    (ROOT / "render_notes.json").write_text(json.dumps(
        dict(stage=args.stage, outputs=outputs, source_registry=registry,
             renderwarn_count=len(MATH_WARNINGS), formula_fallbacks=MATH_WARNINGS,
             note="Unrenderable formulas remain verbatim, never silently altered."),
        ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"renderwarn={len(MATH_WARNINGS)}")
    print(json.dumps(registry, ensure_ascii=False))


if __name__ == "__main__":
    main()

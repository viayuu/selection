"""CPU-only mechanical PDF rendering; scientific source files stay unchanged."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import io
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
TEMPLATE = ROOT.parent / "major_paradigm_review_20261025/render_report.py"
SOURCES = ("REPORT.md", "sources.csv", "ROUTING.json")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_layout():
    spec = importlib.util.spec_from_file_location("frozen_selector_pdf_layout", TEMPLATE)
    layout = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(layout)
    layout.ROOT = ROOT
    layout.previous.MATH_DIR = ROOT / "formula_assets"
    original_formula = layout.render_formula
    original_display = layout.display_formula
    compatibility = []

    def formula(expression, *, inline=False):
        normalized = re.sub(r"\\(?:big|Big|bigg|Bigg)[lr]?(?![A-Za-z])", "", expression)
        normalized = re.sub(r"\\middle(?![A-Za-z])", "", normalized)
        if normalized != expression:
            compatibility.append(dict(rule="Delimiter size only; preserve delimiter and expression",
                source_sha256=hashlib.sha256(expression.encode()).hexdigest()))
        return original_formula(normalized, inline=inline)

    def display(expression):
        array = re.search(r"\\begin\{array\}\{[^}]+\}(.*?)\\end\{array\}", expression, re.S)
        if array:
            rows = []
            for row in re.split(r"\\\\", array[1].replace(r"\hline", "")):
                if row.strip():
                    rows.append([r"\(" + cell.strip() + r"\)" if cell.strip() else ""
                                 for cell in row.split("&")])
            compatibility.append(dict(rule="Mathematical array rendered as aligned native table"))
            return [layout.Spacer(1, 5), layout.make_table(rows, layout.WIDTH * .78), layout.Spacer(1, 5)]
        if r"\begin{cases}" in expression:
            prefix, cases = expression.split(r"\begin{cases}", 1)
            body, suffix = cases.split(r"\end{cases}", 1)
            result = []
            for row in re.split(r"\\\\", body):
                if row.strip():
                    result.extend(display(prefix + row.replace("&", r"\quad ") + suffix))
            compatibility.append(dict(rule="Case branches repeat unchanged left-hand side and constraints"))
            return [layout.layout.KeepTogether(result)]
        expression = re.sub(r"\\(?:begin|end)\{(?:aligned|gathered)\}", "", expression)
        result = []
        # Split chains at their existing punctuation/equality, not at operators.
        lines = re.split(r"\\\\(?:\[[^]]*\])?|(?<=,)\s*\n|\n\s*(?==)", expression)
        for line in lines:
            line = line.replace("&", " ").strip()
            if not line:
                continue
            label = re.search(r"\\text\{([^{}]*[\u4e00-\u9fff][^{}]*)\}", line)
            if label:
                before, after = line[:label.start()].strip(), line[label.end():].strip()
                text = ((r"\(" + before + r"\) ") if before else "") + label[1]
                text += (r" \(" + after + r"\)") if after else ""
                result.append(layout.paragraph(text))
                compatibility.append(dict(rule="Chinese mathematical text kept in embedded CJK paragraph"))
            else:
                result.extend(original_display(line))
        return [layout.layout.KeepTogether(result)] if result else []

    def furniture(canvas, doc):
        width, height = canvas._pagesize
        canvas.saveState()
        for text, y in (
            ("FROZEN SELECTOR MIXTURES / \u79d1\u5b66\u7814\u7a76\u62a5\u544a", height - 11 * layout.mm),
            ("2026-10-30 | \u6b63\u6587\u539f\u6837\u4fdd\u7559 / CPU\u673a\u68b0\u6392\u7248", 9.5 * layout.mm),
        ):
            item = layout.paragraph(text, "SmallCN")
            item.wrap(width - 2 * layout.MARGIN - 35, 20)
            item.drawOn(canvas, layout.MARGIN, y)
        canvas.setStrokeColor(layout.layout.RULE)
        for y in (height - 14 * layout.mm, 14 * layout.mm):
            canvas.line(layout.MARGIN, y, width - layout.MARGIN, y)
        canvas.setFont("Latin", 8)
        canvas.drawRightString(width - layout.MARGIN, 9.5 * layout.mm, str(doc.page))
        canvas.restoreState()

    layout.render_formula, layout.display_formula, layout.furniture = formula, display, furniture
    return layout, compatibility


def attach_sources(payload):
    tool = shutil.which("pdfattach")
    if not tool:
        raise RuntimeError("Existing Poppler pdfattach is required for embedded source attachments")
    with tempfile.TemporaryDirectory(prefix=".pdf-build-", dir=ROOT) as temporary:
        current = Path(temporary) / "base.pdf"
        current.write_bytes(payload)
        for index, name in enumerate(SOURCES):
            target = Path(temporary) / f"attached-{index}.pdf"
            subprocess.run([tool, str(current), str(ROOT / name), str(target)],
                           check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            current = target
        return current.read_bytes()


def main():
    before = {name: digest(ROOT / name) for name in SOURCES}
    route = json.loads((ROOT / "ROUTING.json").read_text())
    if route["identity_evidence"]["scientific_synthesis"]["status"] != "completed":
        raise RuntimeError("Only render the completed scientific source")
    with (ROOT / "sources.csv").open(newline="") as stream:
        references = list(csv.DictReader(stream))
    if len(references) != 10 or len({r["source_id"] for r in references}) != 10:
        raise ValueError("Expected the ten supplied source records, without editing them")
    layout, compatibility = load_layout()
    source = (ROOT / "REPORT.md").read_text()
    layout.CURRENT_SOURCE = "REPORT.md"
    story = layout.parse(source)
    layout.append_break(story)
    registry = ["## \u9644\u5f55\uff1a\u5f15\u7528\u7f16\u53f7\u4e0e\u4e0b\u8f7d\u94fe\u63a5", "",
        "\u4ee5\u4e0b\u4ec5\u673a\u68b0\u6c47\u603b sources.csv \u539f\u59cb\u7f16\u53f7\u3001\u9898\u540d\u548c URL\uff0c\u4e0d\u6539\u5199\u79d1\u5b66\u6b63\u6587\u3002", ""]
    for row in references:
        registry.extend([f"**{row['source_id']}** {row['title']}", "", row["primary_url"], ""])
    story.extend(layout.parse("\n".join(registry)))
    if layout.previous.MATH_WARNINGS:
        raise RuntimeError("Formula rendering failed: " + json.dumps(layout.previous.MATH_WARNINGS, ensure_ascii=False))
    target = io.BytesIO()
    document = layout.document(target)
    document.title = source.splitlines()[0].lstrip("# ")
    document.author = "Completed Astra scientific source; CPU mechanical rendering only"
    document.subject = ROOT.name
    document.build(story, canvasmaker=layout.EmbeddedCanvas)
    payload = attach_sources(target.getvalue())
    checks = layout.verify_pdf(payload, "\u51bb\u7ed3\u5b8c\u6574\u9009\u62e9\u5668", require_link=True)
    if layout.UNSUPPORTED_GLYPHS:
        raise RuntimeError("Unsupported glyphs: " + str(sorted(layout.UNSUPPORTED_GLYPHS)))
    if before != {name: digest(ROOT / name) for name in SOURCES}:
        raise RuntimeError("Scientific inputs changed during rendering; do not publish")
    (ROOT / "REPORT.pdf").write_bytes(payload)
    notes = dict(complete=True, scientific_sources_changed=False, source_sha256=before,
        output="REPORT.pdf", sha256=hashlib.sha256(payload).hexdigest(), bytes=len(payload),
        tools=dict(reportlab=layout.REPORTLAB_VERSION, mathematics="Matplotlib Mathtext", pdf_attachment="Poppler pdfattach"),
        layout_template=dict(path=str(TEMPLATE), sha256=digest(TEMPLATE)),
        cjk_font=str(layout.layout.FONT), embedded_source_attachments=list(SOURCES),
        reference_records=len(references), formula_assets=len(list((ROOT / "formula_assets").glob("*.png"))),
        formula_fallbacks=layout.previous.MATH_WARNINGS, typography_compatibility=compatibility,
        layout_warnings=layout.LAYOUT_WARNINGS, renderer_sha256=digest(__file__), **checks)
    (ROOT / "render_notes.json").write_text(json.dumps(notes, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(notes, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

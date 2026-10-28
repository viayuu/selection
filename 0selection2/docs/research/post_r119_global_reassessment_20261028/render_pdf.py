"""Render Astra's completed research without editing scientific source text."""

import argparse
import hashlib
import importlib.util
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent
TEMPLATE = ROOT.parent / "major_paradigm_review_20261025/render_report.py"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_renderer():
    spec = importlib.util.spec_from_file_location("global_reassessment_layout", TEMPLATE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.ROOT = ROOT
    module.previous.MATH_DIR = ROOT / "formula_assets"
    return module


def render():
    routing = json.loads((ROOT / "ROUTING.json").read_text())
    if (routing.get("phase") != "research_complete"
            or routing.get("requested_model") != "gpt-6-astra"
            or routing.get("requested_reasoning_effort") != "max"):
        raise RuntimeError("Require actual Astra completion; do not render a live draft")
    names = ["REPORT.md", "HANDOFF.md", "sources.csv", "directions.csv"]
    names.extend(name for name in ("evidence.md", "evidence.csv") if (ROOT / name).is_file())
    before = {name: sha(ROOT / name) for name in names}
    layout = load_renderer()
    original_formula = layout.render_formula
    original_display = layout.display_formula

    def render_formula(expression, *, inline=False):
        normalized = re.sub(r"\\(?:big|Big|bigg|Bigg)[lr]?(?![A-Za-z])", "", expression)
        if normalized != expression:
            layout.FORMULA_COMPATIBILITY.append(dict(
                source=layout.CURRENT_SOURCE,
                rule="Keep delimiters while omitting unsupported delimiter-size commands",
                source_expression_sha256=hashlib.sha256(expression.encode()).hexdigest(),
            ))
        return original_formula(normalized, inline=inline)

    def display_formula(expression):
        label = re.search(r"\\text\{([^{}]*[\u4e00-\u9fff][^{}]*)\}", expression)
        if label is None:
            return original_display(expression)
        result = original_display(expression[:label.start()])
        suffix = expression[label.end():].strip()
        result.append(layout.paragraph(label[1] + " $" + suffix + "$"))
        layout.FORMULA_COMPATIBILITY.append(dict(
            source=layout.CURRENT_SOURCE,
            rule="Keep Chinese equation label in embedded CJK text beside its unchanged equation",
            source_expression_sha256=hashlib.sha256(expression.encode()).hexdigest(),
        ))
        return result

    layout.render_formula, layout.display_formula = render_formula, display_formula
    counts = {name: len(layout.read_csv(ROOT / name)[1]) for name in names if name.endswith(".csv")}
    if not all(counts.values()):
        raise ValueError("Completed scientific CSV tables must be nonempty")
    original_document = layout.document

    def furniture(canvas, doc):
        width, height = canvas._pagesize
        canvas.saveState()
        for text, y in (
            ("SOLVER SELECTION / GLOBAL RESEARCH REASSESSMENT", height - 11 * layout.mm),
            ("2026-10-28 | Astra science / Sol mechanical production", 9.5 * layout.mm),
        ):
            item = layout.paragraph(text, "SmallCN")
            item.wrap(width - 2 * layout.MARGIN - 30, 20)
            item.drawOn(canvas, layout.MARGIN, y)
        canvas.setFont("Latin", 8)
        canvas.drawRightString(width - layout.MARGIN, 9.5 * layout.mm, str(doc.page))
        canvas.restoreState()

    def document(target):
        result = original_document(target)
        result.title = "Full18 Top1 And Missed Winners: Global Scientific Reassessment"
        result.author = "gpt-6-astra scientific research; Sol mechanical PDF production"
        result.subject = ROOT.name
        return result

    layout.document, layout.furniture = document, furniture
    report_inputs = ["REPORT.md"]
    if "evidence.md" in names:
        report_inputs.append("evidence.md")
    report_inputs.append("HANDOFF.md")
    outputs = [layout.build("REPORT.pdf", report_inputs), layout.build("HANDOFF.pdf", ["HANDOFF.md"])]
    if layout.previous.MATH_WARNINGS:
        raise RuntimeError("Mechanical formula rendering: " + json.dumps(layout.previous.MATH_WARNINGS, ensure_ascii=False))
    if before != {name: sha(ROOT / name) for name in names}:
        raise RuntimeError("Scientific sources changed during PDF generation")
    notes = dict(
        complete=True,
        scientific_source_changed=False,
        inputs=before,
        research_agent_id=routing["research_agent_id"],
        requested_model="gpt-6-astra",
        requested_reasoning_effort="max",
        rendering_role="Sol mechanical only",
        template=dict(path=str(TEMPLATE), sha256=sha(TEMPLATE)),
        source_csv_rows=counts,
        pdfs=outputs,
        layout_warnings=layout.LAYOUT_WARNINGS,
        formula_typography_compatibility=layout.FORMULA_COMPATIBILITY,
        unsupported_glyphs=sorted(layout.UNSUPPORTED_GLYPHS),
        generator_sha256=sha(__file__),
    )
    (ROOT / "render_notes.json").write_text(json.dumps(notes, ensure_ascii=False, indent=2) + "\n")
    (ROOT / "DELIVERY.md").write_text(
        "# Research Delivery\n\n"
        "Astra/max authored all scientific text and source tables. Sol rendered the sources unchanged.\n\n"
        + "\n".join(f"- {item['file']}: {item['pages']} pages." for item in outputs)
        + "\n\nThe autonomous experiment goal remains PAUSED. Existing R119 pipelines were not interrupted.\n"
        "No new experiment, model inference, solver execution or dataset change was performed for this report.\n"
        "R119 selector observations are provisional; completed historical results are identified separately.\n"
        "Runtime evidence records the explicit Astra/max configuration, not independent backend certification.\n"
    )
    print(json.dumps(notes, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true", help="Import the existing layout without reading drafts")
    args = parser.parse_args()
    if args.prepare:
        load_renderer()
        print(json.dumps(dict(layout_imported=True, scientific_drafts_read=False)))
    else:
        render()

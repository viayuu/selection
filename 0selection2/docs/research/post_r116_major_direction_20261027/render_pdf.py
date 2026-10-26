"""Sol's mechanical PDF handoff; never renders an unfinished Astra draft."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
TEMPLATE = ROOT.parent / "major_paradigm_review_20261025/render_report.py"
SCIENTIFIC = ("REPORT.md", "sources.csv", "directions.csv")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_renderer():
    spec = importlib.util.spec_from_file_location("post_r116_pdf_layout", TEMPLATE)
    renderer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(renderer)
    renderer.ROOT = ROOT
    renderer.previous.MATH_DIR = ROOT / "formula_assets"
    return renderer


def render():
    routing = json.loads((ROOT / "ROUTING.json").read_text())
    if (routing.get("phase") != "research_complete" or routing.get("requested_model") != "gpt-6-astra"
            or routing.get("requested_reasoning_effort") != "max"):
        raise RuntimeError("Main must record actual Astra completion before rendering; no draft polling")
    before = {name: sha(ROOT / name) for name in SCIENTIFIC}
    renderer = load_renderer()
    csv_counts = {name: len(renderer.read_csv(ROOT / name)[1]) for name in SCIENTIFIC[1:]}
    if not all(csv_counts.values()):
        raise ValueError("Completed scientific CSVs must be nonempty")
    original_document = renderer.document

    def furniture(canvas, doc):
        width, height = canvas._pagesize
        canvas.saveState()
        for text, x, y in (
            ("SOLVER SELECTION / 完整实验后的方向研究", renderer.MARGIN, height - 11 * renderer.mm),
            ("2026-10-27 | Astra 科学正文 / Sol 机械排版", renderer.MARGIN, 9.5 * renderer.mm),
        ):
            item = renderer.paragraph(text, "SmallCN")
            item.wrap(width - 2 * renderer.MARGIN - 30, 20)
            item.drawOn(canvas, x, y)
        canvas.setFont("Latin", 8)
        canvas.drawRightString(width - renderer.MARGIN, 9.5 * renderer.mm, str(doc.page))
        canvas.restoreState()

    def document(target):
        value = original_document(target)
        value.title = "零选择与平台期：R115/R116之后的大方向研究"
        value.author = "gpt-6-astra scientific research; gpt-6.1-sol mechanical production"
        value.subject = ROOT.name
        return value

    renderer.furniture, renderer.document = furniture, document
    result = renderer.build("REPORT.pdf", ["REPORT.md"])
    if renderer.previous.MATH_WARNINGS:
        raise ValueError("Formula rendering needs a mechanical correction; scientific source remains untouched")
    if before != {name: sha(ROOT / name) for name in SCIENTIFIC}:
        raise RuntimeError("Scientific source changed during PDF generation")
    notes = dict(complete=True, scientific_source_changed=False, inputs=before,
        actual_agent_id=routing["agent_id"], requested_scientific_model="gpt-6-astra",
        requested_scientific_reasoning_effort="max", rendering_role="Sol mechanical only",
        template=dict(path=str(TEMPLATE), sha256=sha(TEMPLATE)),
        source_csv_rows=csv_counts, pdf=result, unsupported_glyphs=sorted(renderer.UNSUPPORTED_GLYPHS),
        layout_warnings=renderer.LAYOUT_WARNINGS, generator_sha256=sha(__file__))
    (ROOT / "render_notes.json").write_text(json.dumps(notes, ensure_ascii=False, indent=2) + "\n")
    (ROOT / "DELIVERY.md").write_text(
        "# Research Delivery\n\n"
        "Astra/max authored REPORT.md, sources.csv and directions.csv. Sol rendered the unchanged scientific text.\n\n"
        f"REPORT.pdf: {result['pages']} pages. Scientific sources: {csv_counts['sources.csv']} records.\n\n"
        "This is scientific direction research, not an experiment result or a final launch protocol.\n"
        "Sol will design and execute the next finite whole18 experiment using only the two authorized RTX3090s.\n"
        "The overall zero-selection/improvement goal remains ACTIVE.\n")
    print(json.dumps(notes, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true", help="Check the existing layout import without reading drafts")
    args = parser.parse_args()
    if args.prepare:
        r = load_renderer()
        print(json.dumps(dict(layout_imported=True, font=str(r.layout.FONT), scientific_drafts_read=False)))
    else:
        render()

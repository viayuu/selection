"""Render the completed scientific handoff without changing its source."""

import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
TEMPLATE = ROOT.parent / "major_paradigm_review_20261025/render_report.py"
SCIENTIFIC = ("REPORT.md", "sources.csv", "directions.csv")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def renderer():
    spec = importlib.util.spec_from_file_location("post_r117_scientific_layout", TEMPLATE)
    layout = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(layout)
    layout.ROOT = ROOT
    layout.previous.MATH_DIR = ROOT / "formula_assets"
    return layout


def render():
    routing = json.loads((ROOT / "ROUTING.json").read_text())
    if (routing.get("phase") != "research_complete"
            or routing.get("requested_model") != "gpt-6-astra"
            or routing.get("requested_reasoning_effort") != "max"):
        raise RuntimeError("Require an actual Astra completion notification before rendering")
    before = {name: sha(ROOT / name) for name in SCIENTIFIC}
    layout = renderer()
    counts = {name: len(layout.read_csv(ROOT / name)[1]) for name in SCIENTIFIC[1:]}
    if not all(counts.values()):
        raise ValueError("Scientific source tables must be nonempty")
    with (ROOT / "sources.csv").open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        primary = (sum(row.get("source_type") == "primary_paper" for row in reader)
                   if "source_type" in (reader.fieldnames or []) else None)
    original_document = layout.document

    def furniture(canvas, doc):
        width, height = canvas._pagesize
        canvas.saveState()
        for text, y in (("SOLVER SELECTION / SCIENTIFIC DECISION AFTER R117", height - 11 * layout.mm),
                        ("2026-10-27 | Astra science / Sol mechanical production", 9.5 * layout.mm)):
            item = layout.paragraph(text, "SmallCN")
            item.wrap(width - 2 * layout.MARGIN - 30, 20)
            item.drawOn(canvas, layout.MARGIN, y)
        canvas.setFont("Latin", 8)
        canvas.drawRightString(width - layout.MARGIN, 9.5 * layout.mm, str(doc.page))
        canvas.restoreState()

    def document(target):
        result = original_document(target)
        result.title = "Scientific Decision After R117"
        result.author = "gpt-6-astra scientific research; Sol mechanical production"
        result.subject = ROOT.name
        return result

    layout.document, layout.furniture = document, furniture
    pdf = layout.build("REPORT.pdf", ["REPORT.md"])
    if layout.previous.MATH_WARNINGS:
        raise ValueError("Mechanical formula rendering needs correction; do not rewrite scientific sources")
    if before != {name: sha(ROOT / name) for name in SCIENTIFIC}:
        raise RuntimeError("Scientific sources changed during rendering")
    notes = dict(complete=True, scientific_source_changed=False, inputs=before,
        actual_agent_id=routing["research_agent_id"], requested_scientific_model="gpt-6-astra",
        requested_scientific_reasoning_effort="max", rendering_role="Sol mechanical only",
        template=dict(path=str(TEMPLATE), sha256=sha(TEMPLATE)), source_csv_rows=counts,
        primary_papers=primary, pdf=pdf, unsupported_glyphs=sorted(layout.UNSUPPORTED_GLYPHS),
        layout_warnings=layout.LAYOUT_WARNINGS, generator_sha256=sha(__file__))
    (ROOT / "render_notes.json").write_text(json.dumps(notes, ensure_ascii=False, indent=2) + "\n")
    source_note = (f"{primary} primary papers and {counts['sources.csv'] - primary} other records"
                   if primary is not None else "record classification retained in sources.csv")
    (ROOT / "DELIVERY.md").write_text(
        "# Scientific Decision Delivery\n\n"
        "Astra/max authored the scientific text and source tables; Sol rendered them unchanged.\n\n"
        f"REPORT.pdf: {pdf['pages']} pages; sources.csv: {counts['sources.csv']} records "
        f"({source_note}).\n\n"
        "R117 completed and was published before this scientific request. This report is not a new\n"
        "experiment result, operational protocol or approval to launch its suggested direction.\n"
        "Sol's separate early-context feasibility memo is included as an explicitly operational sidecar.\n"
        "The overall improvement goal remains ACTIVE. No new GPU experiment was launched by this report.\n")
    print(json.dumps(notes, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    args = parser.parse_args()
    if args.prepare:
        renderer()
        print(json.dumps(dict(layout_imported=True, scientific_drafts_read=False)))
    else:
        render()

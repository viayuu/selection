from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml
from pypdf import PdfReader


DEFAULT_YAML_PATH = Path("adaptive/papers.yaml")
DEFAULT_OUT_DIR = Path("adaptive/fulltext")


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        obj = yaml.safe_load(f)
    if not isinstance(obj, dict) or "papers" not in obj or not isinstance(obj["papers"], list):
        raise ValueError(f"Invalid YAML schema: expected {{papers: [...]}}: {path}")
    return obj


def _extract_text(pdf_path: Path) -> str:
    reader = PdfReader(str(pdf_path))
    parts: list[str] = []
    for p in reader.pages:
        parts.append(p.extract_text() or "")
    return "\n\n".join(parts).strip() + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Extract plain text from PDFs in adaptive/pdfs/ for local search/indexing.\n"
            "Note: Please respect each paper's license; the generated text is intended for personal use."
        )
    )
    ap.add_argument("--yaml", type=str, default=str(DEFAULT_YAML_PATH), help="Path to papers.yaml")
    ap.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT_DIR), help="Output directory (default: adaptive/fulltext)")
    ap.add_argument("--force", action="store_true", help="Regenerate even if output exists")
    ap.add_argument("--as-md", action="store_true", help="Write .md instead of .txt (still plain text wrapped in Markdown)")
    args = ap.parse_args()

    yaml_path = Path(args.yaml)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = _load_yaml(yaml_path)

    total = 0
    ok = 0
    skipped = 0
    failed = 0

    for paper in meta["papers"]:
        if not isinstance(paper, dict):
            continue
        pid = paper.get("id")
        local_pdf = paper.get("local_pdf")
        if not pid or not local_pdf:
            continue
        pdf_path = Path(local_pdf)
        if not pdf_path.exists():
            print(f"[skip] {pid}: missing pdf {pdf_path}")
            skipped += 1
            continue

        suffix = ".md" if args.as_md else ".txt"
        out_path = out_dir / (pdf_path.stem + suffix)
        total += 1

        if out_path.exists() and out_path.stat().st_size > 0 and not args.force:
            print(f"[skip] {pid}: exists {out_path}")
            skipped += 1
            continue

        try:
            print(f"[extract] {pid}: {pdf_path} -> {out_path}")
            text = _extract_text(pdf_path)
            if args.as_md:
                body = "# Extracted text (for local use)\n\n" + text
            else:
                body = text
            out_path.write_text(body, encoding="utf-8")
            ok += 1
        except Exception as e:
            failed += 1
            print(f"[fail] {pid}: {type(e).__name__}: {e}")

    print(f"Done. total={total} ok={ok} skipped={skipped} failed={failed}")
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())


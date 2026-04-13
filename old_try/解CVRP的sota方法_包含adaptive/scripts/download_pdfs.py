from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import requests
import yaml


DEFAULT_YAML_PATH = Path("adaptive/papers.yaml")


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        obj = yaml.safe_load(f)
    if not isinstance(obj, dict) or "papers" not in obj:
        raise ValueError(f"Invalid YAML schema: expected top-level dict with key 'papers': {path}")
    if not isinstance(obj["papers"], list):
        raise ValueError(f"Invalid YAML schema: 'papers' must be a list: {path}")
    return obj


def _download(url: str, out_path: Path, *, timeout_s: int = 180, user_agent: str = "Mozilla/5.0") -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    headers = {"User-Agent": user_agent}
    with requests.get(url, headers=headers, stream=True, timeout=timeout_s) as r:
        r.raise_for_status()
        tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
        with tmp_path.open("wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 256):
                if chunk:
                    f.write(chunk)
        tmp_path.replace(out_path)


def main() -> int:
    ap = argparse.ArgumentParser(description="Download open-access PDFs listed in adaptive/papers.yaml")
    ap.add_argument("--yaml", type=str, default=str(DEFAULT_YAML_PATH), help="Path to papers.yaml")
    ap.add_argument("--force", action="store_true", help="Redownload even if local_pdf exists")
    ap.add_argument("--timeout", type=int, default=180, help="HTTP timeout (seconds)")
    args = ap.parse_args()

    yaml_path = Path(args.yaml)
    meta = _load_yaml(yaml_path)

    sess = requests.Session()
    total = 0
    ok = 0
    skipped = 0
    failed = 0

    for paper in meta["papers"]:
        if not isinstance(paper, dict):
            continue
        pid = paper.get("id")
        url = paper.get("open_pdf_url")
        local_pdf = paper.get("local_pdf")
        if not pid or not url or not local_pdf:
            continue
        total += 1

        out_path = Path(local_pdf)
        if out_path.exists() and out_path.stat().st_size > 0 and not args.force:
            print(f"[skip] {pid}: {out_path} ({out_path.stat().st_size} bytes)")
            skipped += 1
            continue

        try:
            print(f"[downloading] {pid}: {url}")
            headers = {"User-Agent": "Mozilla/5.0"}
            with sess.get(url, headers=headers, stream=True, timeout=args.timeout) as r:
                r.raise_for_status()
                out_path.parent.mkdir(parents=True, exist_ok=True)
                tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
                with tmp_path.open("wb") as f:
                    for chunk in r.iter_content(chunk_size=1024 * 256):
                        if chunk:
                            f.write(chunk)
                tmp_path.replace(out_path)
            print(f"[ok] {pid}: {out_path} ({out_path.stat().st_size} bytes)")
            ok += 1
        except Exception as e:
            failed += 1
            print(f"[fail] {pid}: {type(e).__name__}: {e}")

    print(f"Done. total={total} ok={ok} skipped={skipped} failed={failed}")
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())


#!/usr/bin/env python3
"""
ARIS Research Wiki — Helper utilities.
Provides slug generation, page creation, edge management, query_pack generation, and stats.
Called by the /research-wiki skill and integration hooks in other skills.

Usage:
    python3 research_wiki.py init <wiki_root>
    python3 research_wiki.py slug "<paper title>" --author "<last name>" --year 2025
    python3 research_wiki.py add_edge <wiki_root> --from <node_id> --to <node_id> --type <edge_type> --evidence "<text>"
    python3 research_wiki.py rebuild_query_pack <wiki_root> [--max-chars 8000]
    python3 research_wiki.py lint <wiki_root>
    python3 research_wiki.py stats <wiki_root>
    python3 research_wiki.py log <wiki_root> "<message>"
"""

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

try:
    import yaml
except ImportError:  # pragma: no cover - fallback is line-based
    yaml = None


FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---\n?", re.S)
HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.M)
PAPER_REQUIRED_SECTIONS = [
    "One-line thesis",
    "Problem / Gap",
    "Method",
    "Key Results",
    "Assumptions",
    "Limitations / Failure Modes",
    "Reusable Ingredients",
    "Open Questions",
    "Claims",
    "Connections",
    "Relevance to This Project",
]


def slugify(title: str, author_last: str = "", year: int = 0) -> str:
    """Generate a canonical slug: author_last + year + keyword."""
    # Extract first meaningful word from title
    stop_words = {"a", "an", "the", "of", "for", "in", "on", "with", "via", "and", "to", "by"}
    words = re.sub(r"[^a-z0-9\s]", "", title.lower()).split()
    keywords = [w for w in words if w not in stop_words and len(w) > 2]
    keyword = "_".join(keywords[:3]) if keywords else "untitled"

    author = re.sub(r"[^a-z]", "", author_last.lower()) if author_last else "unknown"
    yr = str(year) if year else "0000"
    return f"{author}{yr}_{keyword}"


def init_wiki(wiki_root: str):
    """Initialize wiki directory structure."""
    root = Path(wiki_root)
    dirs = ["papers", "ideas", "experiments", "claims", "graph"]
    for d in dirs:
        (root / d).mkdir(parents=True, exist_ok=True)

    # Create empty files if they don't exist
    for f in ["index.md", "log.md", "gap_map.md", "query_pack.md"]:
        path = root / f
        if not path.exists():
            if f == "index.md":
                path.write_text("# Research Wiki Index\n\n_Auto-generated. Do not edit._\n")
            elif f == "log.md":
                path.write_text("# Research Wiki Log\n\n_Append-only timeline._\n")
            elif f == "gap_map.md":
                path.write_text("# Gap Map\n\n_Field gaps with stable IDs._\n")
            elif f == "query_pack.md":
                path.write_text("# Query Pack\n\n_Auto-generated for /idea-creator. Max 8000 chars._\n")

    # Create empty edges file
    edges_path = root / "graph" / "edges.jsonl"
    if not edges_path.exists():
        edges_path.write_text("")

    append_log(wiki_root, "Wiki initialized")
    print(f"Research wiki initialized at {root}")


def add_edge(wiki_root: str, from_id: str, to_id: str, edge_type: str, evidence: str = ""):
    """Add a typed edge to the relationship graph."""
    VALID_TYPES = {
        "extends", "contradicts", "addresses_gap", "inspired_by",
        "tested_by", "supports", "invalidates", "supersedes",
    }
    if edge_type not in VALID_TYPES:
        print(f"Warning: unknown edge type '{edge_type}'. Valid: {VALID_TYPES}", file=sys.stderr)

    edges_path = Path(wiki_root) / "graph" / "edges.jsonl"

    # Dedup check
    existing_edges = []
    if edges_path.exists():
        for line in edges_path.read_text().strip().split("\n"):
            if line.strip():
                try:
                    existing_edges.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    # Check if edge already exists
    for e in existing_edges:
        if e.get("from") == from_id and e.get("to") == to_id and e.get("type") == edge_type:
            print(f"Edge already exists: {from_id} --{edge_type}--> {to_id}")
            return

    edge = {
        "from": from_id,
        "to": to_id,
        "type": edge_type,
        "evidence": evidence,
        "added": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }

    with open(edges_path, "a") as f:
        f.write(json.dumps(edge, ensure_ascii=False) + "\n")

    print(f"Edge added: {from_id} --{edge_type}--> {to_id}")


def rebuild_query_pack(wiki_root: str, max_chars: int = 8000):
    """Generate a compressed query_pack.md for /idea-creator."""
    root = Path(wiki_root)
    sections = []

    # 1. Project direction (300 chars)
    brief_path = root.parent / "RESEARCH_BRIEF.md"
    if brief_path.exists():
        brief = brief_path.read_text()[:300]
        sections.append(f"## Project Direction\n{brief}\n")

    # 2. Gap map (1200 chars)
    gap_path = root / "gap_map.md"
    if gap_path.exists():
        gaps = gap_path.read_text()[:1200]
        if gaps.strip() and gaps.strip() != "# Gap Map\n\n_Field gaps with stable IDs._":
            sections.append(f"## Open Gaps\n{gaps}\n")

    # 3. Failed ideas (1400 chars) — highest anti-repetition value
    ideas_dir = root / "ideas"
    if ideas_dir.exists():
        failed = []
        for f in sorted(ideas_dir.glob("*.md")):
            content = f.read_text()
            if "outcome: negative" in content or "outcome: mixed" in content:
                # Extract frontmatter title and failure notes
                lines = content.split("\n")
                title = ""
                failure = ""
                for line in lines:
                    if line.startswith("title:"):
                        title = line.split(":", 1)[1].strip().strip('"')
                    if "failure" in line.lower() or "lesson" in line.lower():
                        idx = lines.index(line)
                        failure = "\n".join(lines[idx:idx+3])
                if title:
                    failed.append(f"- **{title}**: {failure[:200]}")
        if failed:
            failed_text = "\n".join(failed)[:1400]
            sections.append(f"## Failed Ideas (avoid repeating)\n{failed_text}\n")

    # 4. Paper summaries (1800 chars) — top by relevance
    papers_dir = root / "papers"
    if papers_dir.exists():
        paper_summaries = []
        for f in sorted(papers_dir.glob("*.md")):
            content = f.read_text()
            # Extract one-line thesis and key fields
            node_id = ""
            title = ""
            thesis = ""
            for line in content.split("\n"):
                if line.startswith("node_id:"):
                    node_id = line.split(":", 1)[1].strip()
                if line.startswith("title:"):
                    title = line.split(":", 1)[1].strip().strip('"')
                if line.startswith("# One-line thesis"):
                    idx = content.split("\n").index(line)
                    next_lines = content.split("\n")[idx+1:idx+3]
                    thesis = " ".join(l for l in next_lines if l.strip() and not l.startswith("#"))
            if title:
                paper_summaries.append(f"- [{node_id}] {title}: {thesis[:150]}")

        if paper_summaries:
            papers_text = "\n".join(paper_summaries[:12])[:1800]
            sections.append(f"## Key Papers ({len(paper_summaries)} total)\n{papers_text}\n")

    # 5. Active relationship chains (900 chars)
    edges_path = root / "graph" / "edges.jsonl"
    if edges_path.exists():
        edges = []
        for line in edges_path.read_text().strip().split("\n"):
            if line.strip():
                try:
                    edges.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        if edges:
            chains = []
            for e in edges[-20:]:  # recent edges
                chains.append(f"  {e['from']} --{e['type']}--> {e['to']}")
            chains_text = "\n".join(chains)[:900]
            sections.append(f"## Recent Relationships ({len(edges)} total)\n{chains_text}\n")

    # Assemble
    pack = "# Research Wiki Query Pack\n\n_Auto-generated. Do not edit._\n\n"
    for s in sections:
        if len(pack) + len(s) <= max_chars:
            pack += s
        else:
            remaining = max_chars - len(pack) - 20
            if remaining > 100:
                pack += s[:remaining] + "\n...(truncated)\n"
            break

    pack_path = root / "query_pack.md"
    pack_path.write_text(pack)
    print(f"query_pack.md rebuilt: {len(pack)} chars")


def get_stats(wiki_root: str):
    """Print wiki statistics."""
    root = Path(wiki_root)

    def count_files(subdir):
        d = root / subdir
        return len(list(d.glob("*.md"))) if d.exists() else 0

    def count_by_field(subdir, field, value):
        d = root / subdir
        if not d.exists():
            return 0
        count = 0
        for f in d.glob("*.md"):
            if f"{field}: {value}" in f.read_text():
                count += 1
        return count

    papers = count_files("papers")
    ideas = count_files("ideas")
    experiments = count_files("experiments")
    claims = count_files("claims")

    edges_path = root / "graph" / "edges.jsonl"
    edge_count = 0
    if edges_path.exists():
        edge_count = sum(1 for line in edges_path.read_text().strip().split("\n") if line.strip())

    print(f"📚 Research Wiki Stats")
    print(f"Papers:      {papers}")
    print(f"Ideas:       {ideas} ({count_by_field('ideas', 'outcome', 'negative')} failed, "
          f"{count_by_field('ideas', 'outcome', 'positive')} succeeded)")
    print(f"Experiments: {experiments}")
    print(f"Claims:      {claims} ({count_by_field('claims', 'status', 'supported')} supported, "
          f"{count_by_field('claims', 'status', 'invalidated')} invalidated)")
    print(f"Edges:       {edge_count}")
    print(f"Wiki root:   {root}")


def append_log(wiki_root: str, message: str):
    """Append a timestamped entry to log.md."""
    log_path = Path(wiki_root) / "log.md"
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    entry = f"- `{ts}` {message}\n"

    if log_path.exists():
        with open(log_path, "a") as f:
            f.write(entry)
    else:
        log_path.write_text(f"# Research Wiki Log\n\n{entry}")


def _parse_inline_list(value: str):
    value = value.strip()
    if not value.startswith("[") or not value.endswith("]"):
        return value

    inner = value[1:-1].strip()
    if not inner:
        return []

    parts = [p.strip() for p in re.split(r',(?=(?:[^"]*"[^"]*")*[^"]*$)', inner)]
    cleaned = []
    for part in parts:
        if not part:
            continue
        cleaned.append(part.strip().strip('"').strip("'"))
    return cleaned


def _parse_frontmatter_text(frontmatter_text: str):
    if yaml is not None:
        try:
            data = yaml.safe_load(frontmatter_text)
            if isinstance(data, dict):
                return data
        except Exception:
            pass

    data = {}
    for raw_line in frontmatter_text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if value.lower() == "null":
            data[key] = None
        elif value.isdigit():
            data[key] = int(value)
        elif value.startswith("[") and value.endswith("]"):
            data[key] = _parse_inline_list(value)
        else:
            data[key] = value.strip('"').strip("'")
    return data


def _load_page(path: Path, entity_type: str):
    text = path.read_text()
    meta = {}
    body = text

    match = FRONTMATTER_RE.match(text)
    if match:
        meta = _parse_frontmatter_text(match.group(1)) or {}
        body = text[match.end():]

    sections = {}
    matches = list(HEADING_RE.finditer(body))
    for idx, heading_match in enumerate(matches):
        heading = heading_match.group(2).strip()
        start = heading_match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(body)
        content = body[start:end].strip()
        sections[heading] = content

    return {
        "path": path,
        "type": meta.get("type", entity_type),
        "node_id": meta.get("node_id", ""),
        "meta": meta,
        "body": body,
        "sections": sections,
    }


def _load_pages(wiki_root: str):
    root = Path(wiki_root)
    pages = []
    for entity_type in ["papers", "ideas", "experiments", "claims"]:
        entity_dir = root / entity_type
        if not entity_dir.exists():
            continue
        singular = entity_type[:-1] if entity_type.endswith("s") else entity_type
        for path in sorted(entity_dir.glob("*.md")):
            pages.append(_load_page(path, singular))
    return pages


def _load_edges(wiki_root: str):
    edges_path = Path(wiki_root) / "graph" / "edges.jsonl"
    if not edges_path.exists():
        return []

    edges = []
    for line in edges_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            edges.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return edges


def _parse_timestamp(value: str):
    if not value:
        return None
    value = value.strip()
    try:
        if value.endswith("Z"):
            return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
        return datetime.fromisoformat(value)
    except ValueError:
        try:
            return datetime.strptime(value, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        except ValueError:
            return None


def _find_implicit_edges(pages, edge_triplets):
    implicit_inspired = []
    implicit_gap = []

    for page in pages:
        if page["type"] != "idea" or not page["node_id"]:
            continue

        based_on = page["meta"].get("based_on") or []
        target_gaps = page["meta"].get("target_gaps") or []

        for paper_id in based_on:
            if (page["node_id"], paper_id, "inspired_by") not in edge_triplets:
                implicit_inspired.append((page["node_id"], paper_id))

        for gap_id in target_gaps:
            gap_raw = str(gap_id)
            gap_node = gap_raw if gap_raw.startswith("gap:") else f"gap:{gap_raw}"
            if (
                (page["node_id"], gap_raw, "addresses_gap") not in edge_triplets
                and (page["node_id"], gap_node, "addresses_gap") not in edge_triplets
            ):
                implicit_gap.append((page["node_id"], gap_raw))

    return implicit_inspired, implicit_gap


def lint_wiki(wiki_root: str):
    """Run health checks and write LINT_REPORT.md."""
    root = Path(wiki_root)
    pages = _load_pages(wiki_root)
    edges = _load_edges(wiki_root)
    edge_triplets = {
        (edge.get("from", ""), edge.get("to", ""), edge.get("type", ""))
        for edge in edges
    }

    node_lookup = {
        page["node_id"]: page for page in pages
        if page["node_id"]
    }
    adjacency = defaultdict(int)
    for edge in edges:
        if edge.get("from") in node_lookup:
            adjacency[edge["from"]] += 1
        if edge.get("to") in node_lookup:
            adjacency[edge["to"]] += 1

    orphan_pages = [
        page for page in pages
        if page["node_id"] and adjacency[page["node_id"]] == 0
    ]
    orphan_pages.sort(key=lambda page: (page["type"], page["node_id"]))

    now = datetime.now(timezone.utc)
    stale_claims = []
    for page in pages:
        if page["type"] != "claim":
            continue
        status = str(page["meta"].get("status", "")).strip().lower()
        if status != "reported":
            continue
        ts = (
            _parse_timestamp(page["meta"].get("updated_at"))
            or _parse_timestamp(page["meta"].get("created_at"))
            or datetime.fromtimestamp(page["path"].stat().st_mtime, tz=timezone.utc)
        )
        age_days = (now - ts).days
        if age_days > 14:
            stale_claims.append((page, age_days))

    supported_targets = defaultdict(int)
    invalidated_targets = defaultdict(int)
    for edge in edges:
        if edge.get("type") == "supports":
            supported_targets[edge.get("to", "")] += 1
        elif edge.get("type") == "invalidates":
            invalidated_targets[edge.get("to", "")] += 1

    contradictions = []
    for page in pages:
        if page["type"] != "claim" or not page["node_id"]:
            continue
        if supported_targets[page["node_id"]] and invalidated_targets[page["node_id"]]:
            contradictions.append(page)

    paper_pairs_with_edges = set()
    for edge in edges:
        from_id = edge.get("from", "")
        to_id = edge.get("to", "")
        if from_id.startswith("paper:") and to_id.startswith("paper:"):
            paper_pairs_with_edges.add(tuple(sorted((from_id, to_id))))

    paper_pages = [page for page in pages if page["type"] == "paper" and page["node_id"]]
    missing_connections = []
    for idx, left in enumerate(paper_pages):
        left_tags = set(left["meta"].get("tags") or [])
        for right in paper_pages[idx + 1:]:
            right_tags = set(right["meta"].get("tags") or [])
            shared_tags = sorted(left_tags & right_tags)
            if len(shared_tags) < 2:
                continue
            pair_key = tuple(sorted((left["node_id"], right["node_id"])))
            if pair_key in paper_pairs_with_edges:
                continue
            missing_connections.append({
                "left": left,
                "right": right,
                "shared_tags": shared_tags,
            })
    missing_connections.sort(
        key=lambda item: (-len(item["shared_tags"]), item["left"]["node_id"], item["right"]["node_id"])
    )

    tested_ideas = set()
    for edge in edges:
        edge_type = edge.get("type", "")
        from_id = edge.get("from", "")
        to_id = edge.get("to", "")
        if edge_type == "tested_by" and from_id.startswith("idea:"):
            tested_ideas.add(from_id)
        if edge_type in {"supports", "invalidates"} and to_id.startswith("idea:"):
            tested_ideas.add(to_id)

    dead_ideas = [
        page for page in pages
        if page["type"] == "idea"
        and str(page["meta"].get("stage", "")).strip().lower() == "proposed"
        and page["node_id"] not in tested_ideas
    ]
    dead_ideas.sort(key=lambda page: page["node_id"])

    sparse_pages = []
    for page in pages:
        if page["type"] == "paper":
            missing_sections = [
                section for section in PAPER_REQUIRED_SECTIONS
                if not page["sections"].get(section, "").strip()
            ]
            if len(missing_sections) >= 3:
                sparse_pages.append((page, missing_sections))
        else:
            empty_sections = [
                heading for heading, content in page["sections"].items()
                if not content.strip()
            ]
            if len(empty_sections) >= 3:
                sparse_pages.append((page, empty_sections))
    sparse_pages.sort(key=lambda item: (-len(item[1]), item[0]["node_id"]))

    implicit_inspired, implicit_gap = _find_implicit_edges(pages, edge_triplets)

    report_lines = [
        "# Research Wiki Lint Report",
        "",
        f"Generated: `{now.strftime('%Y-%m-%dT%H:%M:%SZ')}`",
        f"Wiki root: `{root}`",
        "",
        "## Summary",
        "",
        "| Check | Count | Notes |",
        "| --- | ---: | --- |",
        f"| Orphan pages | {len(orphan_pages)} | Graph-connected node pages with zero in/out edges |",
        f"| Stale claims | {len(stale_claims)} | `status: reported` older than 14 days |",
        f"| Contradictions | {len(contradictions)} | Claims with both `supports` and `invalidates` evidence |",
        f"| Missing connections | {len(missing_connections)} | Paper pairs with 2+ shared tags but no graph relation |",
        f"| Dead ideas | {len(dead_ideas)} | `stage: proposed` ideas with no experiment linkage |",
        f"| Sparse pages | {len(sparse_pages)} | Pages missing 3+ required/expected sections |",
        "",
    ]

    if not edges:
        report_lines.extend([
            "## Global Observation",
            "",
            "- `graph/edges.jsonl` is currently empty. That makes every node page orphaned and leaves idea/paper relationships implicit instead of materialized.",
            f"- The frontmatter already exposes at least {len(implicit_inspired)} candidate `inspired_by` edges and {len(implicit_gap)} candidate `addresses_gap` edges that could be backfilled automatically.",
            "",
        ])

    report_lines.extend([
        "## 1. Orphan Pages",
        "",
    ])
    if orphan_pages:
        grouped_orphans = defaultdict(list)
        for page in orphan_pages:
            grouped_orphans[page["type"]].append(page["node_id"])
        for entity_type in ["paper", "idea", "experiment", "claim"]:
            ids = grouped_orphans.get(entity_type, [])
            if ids:
                report_lines.append(f"- `{entity_type}` ({len(ids)}): " + ", ".join(f"`{node_id}`" for node_id in ids))
        if implicit_inspired or implicit_gap:
            report_lines.append("- Suggested fix: backfill graph edges from idea frontmatter before curating higher-order relationships.")
    else:
        report_lines.append("- No orphan pages found.")
    report_lines.append("")

    report_lines.extend([
        "## 2. Stale Claims",
        "",
    ])
    if stale_claims:
        for page, age_days in stale_claims:
            report_lines.append(f"- `{page['node_id']}` — {age_days} days since last update.")
    else:
        report_lines.append("- No stale claims found.")
    report_lines.append("")

    report_lines.extend([
        "## 3. Contradictions",
        "",
    ])
    if contradictions:
        for page in contradictions:
            report_lines.append(
                f"- `{page['node_id']}` has both `supports` ({supported_targets[page['node_id']]}) and `invalidates` ({invalidated_targets[page['node_id']]}) edges."
            )
    else:
        report_lines.append("- No contradictory claim evidence found.")
    report_lines.append("")

    report_lines.extend([
        "## 4. Missing Connections",
        "",
    ])
    if missing_connections:
        report_lines.extend([
            "| Pair | Shared tags | Suggested review |",
            "| --- | --- | --- |",
        ])
        for item in missing_connections:
            left = item["left"]["node_id"]
            right = item["right"]["node_id"]
            shared = ", ".join(f"`{tag}`" for tag in item["shared_tags"])
            report_lines.append(
                f"| `{left}` ↔ `{right}` | {shared} | Consider adding `extends`, `supersedes`, or `contradicts` after manual review. |"
            )
    else:
        report_lines.append("- No missing paper-to-paper relationship candidates found.")
    report_lines.append("")

    report_lines.extend([
        "## 5. Dead Ideas",
        "",
    ])
    if dead_ideas:
        for page in dead_ideas:
            title = page["meta"].get("title", page["path"].stem)
            report_lines.append(f"- `{page['node_id']}` — {title}")
        report_lines.extend([
            "- Suggested fix: attach `tested_by` / experiment evidence for active pilots, or archive/subsume legacy proposals that are no longer live.",
        ])
    else:
        report_lines.append("- No untested proposed ideas found.")
    report_lines.append("")

    report_lines.extend([
        "## 6. Sparse Pages",
        "",
    ])
    if sparse_pages:
        report_lines.append("_Paper pages are checked against the research-wiki paper schema; other entities are checked for explicitly empty headings._")
        report_lines.append("")
        report_lines.extend([
            "| Page | Missing section count | Missing sections |",
            "| --- | ---: | --- |",
        ])
        for page, missing_sections in sparse_pages:
            preview = ", ".join(f"`{section}`" for section in missing_sections[:6])
            if len(missing_sections) > 6:
                preview += ", ..."
            report_lines.append(
                f"| `{page['node_id']}` | {len(missing_sections)} | {preview} |"
            )
    else:
        report_lines.append("- No sparse pages found.")
    report_lines.append("")

    report_lines.extend([
        "## Suggested Fix Order",
        "",
        f"1. Backfill the {len(implicit_inspired)} implicit `inspired_by` edges and {len(implicit_gap)} implicit `addresses_gap` edges already present in idea frontmatter.",
        "2. Normalize gap node references before adding gap edges. Current frontmatter uses bare IDs such as `G1`; the skill spec prefers canonical IDs like `gap:G1`.",
        "3. Convert legacy proposal pages that are only kept for context from `stage: proposed` to an archived/subsumed state so they stop surfacing as dead ideas.",
        "4. Expand the core paper pages first (`gao2025_nss`, `zhou2025_urs`, `yu2026_coeks`) so the wiki has at least a few fully populated anchors before filling the long tail.",
    ])

    report_path = root / "LINT_REPORT.md"
    report_path.write_text("\n".join(report_lines).rstrip() + "\n")

    print(f"Lint report written to {report_path}")
    print(f"Orphan pages: {len(orphan_pages)}")
    print(f"Stale claims: {len(stale_claims)}")
    print(f"Contradictions: {len(contradictions)}")
    print(f"Missing connections: {len(missing_connections)}")
    print(f"Dead ideas: {len(dead_ideas)}")
    print(f"Sparse pages: {len(sparse_pages)}")


def main():
    parser = argparse.ArgumentParser(description="ARIS Research Wiki utilities")
    subparsers = parser.add_subparsers(dest="command")

    # init
    p_init = subparsers.add_parser("init")
    p_init.add_argument("wiki_root")

    # slug
    p_slug = subparsers.add_parser("slug")
    p_slug.add_argument("title")
    p_slug.add_argument("--author", default="")
    p_slug.add_argument("--year", type=int, default=0)

    # add_edge
    p_edge = subparsers.add_parser("add_edge")
    p_edge.add_argument("wiki_root")
    p_edge.add_argument("--from", dest="from_id", required=True)
    p_edge.add_argument("--to", dest="to_id", required=True)
    p_edge.add_argument("--type", dest="edge_type", required=True)
    p_edge.add_argument("--evidence", default="")

    # rebuild_query_pack
    p_qp = subparsers.add_parser("rebuild_query_pack")
    p_qp.add_argument("wiki_root")
    p_qp.add_argument("--max-chars", type=int, default=8000)

    # lint
    p_lint = subparsers.add_parser("lint")
    p_lint.add_argument("wiki_root")

    # stats
    p_stats = subparsers.add_parser("stats")
    p_stats.add_argument("wiki_root")

    # log
    p_log = subparsers.add_parser("log")
    p_log.add_argument("wiki_root")
    p_log.add_argument("message")

    args = parser.parse_args()

    if args.command == "init":
        init_wiki(args.wiki_root)
    elif args.command == "slug":
        print(slugify(args.title, args.author, args.year))
    elif args.command == "add_edge":
        add_edge(args.wiki_root, args.from_id, args.to_id, args.edge_type, args.evidence)
    elif args.command == "rebuild_query_pack":
        rebuild_query_pack(args.wiki_root, args.max_chars)
    elif args.command == "lint":
        lint_wiki(args.wiki_root)
    elif args.command == "stats":
        get_stats(args.wiki_root)
    elif args.command == "log":
        append_log(args.wiki_root, args.message)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

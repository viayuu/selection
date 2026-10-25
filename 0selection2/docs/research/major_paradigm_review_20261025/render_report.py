"""Mechanical PDF production only; the default prepare stage never reads drafts.

Reuse the previous renderer and /tmp/shiys_astra_pdf_dependencies_compat.
Run prepare --self-test now; use finalmain --finalized only after notification.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import importlib.util
import io
import json
import math
import re
import shutil
import subprocess
import sys
import unicodedata
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from xml.etree import ElementTree

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
DEPENDENCIES = Path("/tmp/shiys_astra_pdf_dependencies_compat")
PREVIOUS = ROOT.parent / "comprehensive_direction_review_20261025/render_report.py"
sys.path.insert(0, str(DEPENDENCIES))

spec = importlib.util.spec_from_file_location("major_paradigm_previous_renderer", PREVIOUS)
previous = importlib.util.module_from_spec(spec)
spec.loader.exec_module(previous)
layout = previous.layout
previous.MATH_DIR = ROOT / "formula_assets"

from reportlab import Version as REPORTLAB_VERSION
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen.canvas import Canvas
from reportlab.platypus import (
    BaseDocTemplate, Flowable, Frame, HRFlowable, Image, LongTable,
    NextPageTemplate, PageBreak, PageTemplate, Paragraph, Spacer, TableStyle,
)

APPENDICES = (
    "01_computational_architectures.md",
    "02_representation_learning.md",
    "03_prediction_objects.md",
    "04_data_and_learning_paradigms.md",
)
SOURCE_CSVS = tuple(f"{number:02d}_sources.csv" for number in range(1, 5))
OPTIONAL_MAIN_SOURCE_CSVS = ("synthesis_sources.csv",)
MAIN_INPUTS = ("REPORT.md", *APPENDICES, "SYNTHESIS_DIALOGUE.md")
FINAL_INPUTS = (*MAIN_INPUTS, *SOURCE_CSVS, "research_directions.csv")
MARGIN = 17 * mm
TOP = 21 * mm
BOTTOM = 20 * mm
WIDTH = A4[0] - 2 * MARGIN
WIDE_WIDTH = landscape(A4)[0] - 2 * MARGIN
LAYOUT_WARNINGS = []
FORMULA_COMPATIBILITY = []
UNSUPPORTED_GLYPHS = set()
CURRENT_SOURCE = "synthetic-layout-test"
CURRENT_WIDTH = WIDTH
pdfmetrics.registerFont(TTFont("DiagramMono", str(layout.LATIN_DIR / "DejaVuSansMono.ttf")))


@contextmanager
def available_width(width):
    global CURRENT_WIDTH
    saved = CURRENT_WIDTH
    CURRENT_WIDTH = width
    try:
        yield
    finally:
        CURRENT_WIDTH = saved


def font_for(character, mono=False):
    primary = "DiagramMono" if mono else "Latin"
    for name in (primary, "CJK", "Latin"):
        if ord(character) in pdfmetrics.getFont(name).face.charToGlyph:
            return name
    if not character.isspace():
        UNSUPPORTED_GLYPHS.add(f"U+{ord(character):04X} {character}")
    return primary


def text_markup(text):
    runs = []
    for character in text:
        font = font_for(character)
        if runs and runs[-1][0] == font:
            runs[-1][1] += character
        else:
            runs.append([font, character])
    return "".join(f'<font name="{font}">{html.escape(value)}</font>' for font, value in runs)


INLINE_TOKEN = re.compile(
    r"(?P<math>\\\((?P<expression>.*?)\\\)|(?<!\\)\$(?!\$)(?P<dollar>[^\n$]+)\$)"
    r"|(?P<link>\[(?P<label>[^\]\n]+)\]\((?P<url>(?:[^()\n]|\([^()\n]*\))*)\))"
    r"|(?P<bold>\*\*(?P<strong>[^*]+)\*\*)"
    r"|(?P<code>`(?P<literal>[^`]+)`)"
    r"|(?P<br><br\s*/?>)"
    r"|(?P<bare>https?://[^\s<>\u2e80-\u9fff]+)", re.DOTALL | re.IGNORECASE,
)


def render_formula(expression, *, inline=False):
    original = expression
    conversions = []

    def replace(pattern, replacement, rule):
        nonlocal expression
        expression, count = re.subn(pattern, replacement, expression)
        if count:
            conversions.append(dict(rule=rule, occurrences=count))

    # Keep symbols intact while adapting commands absent in the installed Mathtext.
    replace(r"\\text\{([\x20-\x7a\x7c\x7e]*)\}",
            lambda match: r"\mathrm{" + match[1].replace(" ", r"\ ") + "}",
            "ASCII text to roman literal")
    replace(r"\\setminus(?![A-Za-z])", lambda match: r"\smallsetminus",
            "set difference command alias")
    replace(r"\\mathrel\{\+\}", lambda match: "{+}",
            "plus relation typography wrapper")
    replace(r"\\overline\s+([A-Za-z0-9])",
            lambda match: r"\overline{" + match[1] + "}",
            "brace the single-token overline argument")

    def bold_italic(match):
        letter = match[1] or match[2]
        start = 0x1D468 if letter.isupper() else 0x1D482
        first = "A" if letter.isupper() else "a"
        return "{" + chr(start + ord(letter) - ord(first)) + "}"

    replace(r"\\boldsymbol\s*(?:\{([A-Za-z])\}|([A-Za-z]))", bold_italic,
            "single Latin bold italic to equivalent mathematical Unicode glyph")
    if conversions:
        record = dict(source=CURRENT_SOURCE, inline=inline, rules=conversions,
                      source_expression_sha256=hashlib.sha256(original.encode()).hexdigest())
        if record not in FORMULA_COMPATIBILITY:
            FORMULA_COMPATIBILITY.append(record)
    result = previous.formula(expression, inline=inline)
    if result is None:
        previous.MATH_WARNINGS[-1].update(source=CURRENT_SOURCE, source_expression=original)
    return result


def inline(text):
    parts = []
    cursor = 0
    for match in INLINE_TOKEN.finditer(text):
        parts.append(text_markup(text[cursor:match.start()]))
        if match["math"]:
            expression = match["expression"] if match["expression"] is not None else match["dollar"]
            saved = previous.INLINE_LIMIT
            previous.INLINE_LIMIT = max(12, CURRENT_WIDTH * 0.82)
            try:
                rendered = render_formula(expression, inline=True)
            finally:
                previous.INLINE_LIMIT = saved
            if rendered:
                path, width, height = rendered
                parts.append(f'<img src="{path}" width="{width:.2f}" height="{height:.2f}" valign="-2"/>')
            else:
                parts.append(text_markup("[公式原文 / 排版回退: " + expression + "]"))
        elif match["link"]:
            target = match["url"].strip().strip("<>")
            label = inline(match["label"])
            if target.startswith(("https://", "http://")):
                parts.append(f'<link href="{html.escape(target, quote=True)}" color="#135968">{label}</link>')
            else:
                parts.append(label)
        elif match["bold"]:
            parts.append("<b>" + inline(match["strong"]) + "</b>")
        elif match["code"]:
            parts.append('<font color="#345A67">' + text_markup(match["literal"]) + "</font>")
        elif match["br"]:
            parts.append("<br/>")
        else:
            target = match["bare"].rstrip(".,;:!?，。；")
            tail = match["bare"][len(target):]
            parts.append(f'<link href="{html.escape(target, quote=True)}" color="#135968">{text_markup(target)}</link>')
            parts.append(text_markup(tail))
        cursor = match.end()
    parts.append(text_markup(text[cursor:]))
    return "".join(parts)


def paragraph(text, style="BodyCN"):
    content = inline(text)
    selected = ParagraphStyle("Major" + style, parent=layout.STYLES[style],
                              autoLeading="max", splitLongWords=True)
    # CJK wrapping can hang one punctuation glyph beyond its nominal line width.
    selected.rightIndent += selected.fontSize
    if "<img " in content:
        selected.wordWrap = None
    return Paragraph(content, selected)


def table_cells(line):
    separators = [m.start() for m in re.finditer(
        r"`[^`]*`|\\\(.*?\\\)|(?<!\\)\$[^$]*\$|\\\||\|", line) if m[0] == "|"]
    start = 0
    cells = []
    for position in separators:
        cells.append(line[start:position].strip().replace(r"\|", "|"))
        start = position + 1
    cells.append(line[start:].strip().replace(r"\|", "|"))
    if line.startswith("|"):
        cells.pop(0)
    if line.endswith("|"):
        cells.pop()
    return cells


def make_table(rows, width):
    count = len(rows[0])
    if not count or any(len(row) != count for row in rows):
        raise ValueError(f"{CURRENT_SOURCE}: inconsistent Markdown table columns")
    minimum = 49
    if count * minimum > width:
        raise ValueError(f"{CURRENT_SOURCE}: {count}-column table needs a dedicated layout; no text was dropped")
    weights = []
    for column in range(count):
        lengths = sorted(len(row[column]) for row in rows)
        typical = lengths[min(len(lengths) - 1, int(len(lengths) * 0.7))]
        weights.append(math.sqrt(max(8, min(typical, 70))))
    remaining = width - count * minimum
    widths = [minimum + remaining * weight / sum(weights) for weight in weights]
    data = []
    for row in rows:
        cells = []
        for value, cell_width in zip(row, widths):
            with available_width(cell_width - 12):
                cells.append(paragraph(value, "TableCN"))
        data.append(cells)
    result = LongTable(data, colWidths=widths, repeatRows=1, splitByRow=1,
                       splitInRow=1, hAlign="LEFT", spaceAfter=10)
    result.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), "Latin"),
        ("BACKGROUND", (0, 0), (-1, 0), layout.PALE),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [layout.colors.white, layout.colors.HexColor("#F8FAFB")]),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LINEBELOW", (0, 0), (-1, 0), 0.5, layout.RULE),
        ("LINEBELOW", (0, 1), (-1, -1), 0.25, layout.RULE),
    ]))
    return result


def cell_count(text):
    return sum(2 if unicodedata.east_asian_width(c) in {"W", "F"} else 1 for c in text)


class TextDiagram(Flowable):
    """Draw on a character grid, preserving indentation and box-drawing alignment."""

    def __init__(self, lines, width, size=8.5):
        super().__init__()
        self.lines = [line.expandtabs(4) for line in lines] or [""]
        columns = max(cell_count(line) for line in self.lines)
        self.size = min(size, (width - 16) / max(1, columns) / 0.6021)
        if self.size < 7:
            raise ValueError(f"{CURRENT_SOURCE}: architecture/code block cannot fit at a readable 7pt")
        self.leading = self.size * 1.65
        self.width = width
        self.height = len(self.lines) * self.leading + 14

    def wrap(self, availWidth, availHeight):
        return self.width, self.height

    def split(self, availWidth, availHeight):
        count = int((availHeight - 14) // self.leading)
        if count <= 0:
            return []
        if count >= len(self.lines):
            return [self]
        return [TextDiagram(self.lines[:count], self.width, self.size),
                TextDiagram(self.lines[count:], self.width, self.size)]

    def draw(self):
        self.canv.setFillColor(layout.PALE)
        self.canv.rect(0, 0, self.width, self.height, fill=1, stroke=0)
        self.canv.setFillColor(layout.INK)
        for index, line in enumerate(self.lines):
            x = 8
            y = self.height - 7 - self.size - index * self.leading
            for character in line:
                self.canv.setFont(font_for(character, mono=True), self.size)
                self.canv.drawString(x, y, character)
                x += cell_count(character) * self.size * 0.6021


def append_break(story):
    if story and not isinstance(story[-1], PageBreak):
        story.append(PageBreak())


def wide_block(story, block):
    story.extend([NextPageTemplate("landscape"), PageBreak(), block,
                  NextPageTemplate("portrait"), PageBreak()])


def display_formula(expression):
    result = []
    original = expression
    expression = re.sub(r"\\(?:begin|end)\{(?:aligned|gathered)\}", "", expression)
    for line in re.split(r"\\\\(?:\[[^]]*\])?", expression):
        if not line.strip():
            continue
        rendered = render_formula(line.replace("&", " "))
        if rendered:
            path, width, height = rendered
            result.extend([Spacer(1, 4), Image(str(path), width=width, height=height), Spacer(1, 7)])
        else:
            result.append(paragraph("公式排版回退，保留原文，不审查数学含义。", "SmallCN"))
            result.append(paragraph("`" + original.replace("`", "&#96;") + "`", "CodeCN"))
            break
    return result


def parse(text):
    story = []
    references = {}
    fenced = False
    lines = []
    for line in text.splitlines():
        if re.match(r"\s*(```|~~~)", line):
            fenced = not fenced
        reference = re.match(r"^\[([^]]+)\]:\s*(\S+)", line) if not fenced else None
        if reference:
            references[reference[1].casefold()] = reference[2].strip("<>")
        else:
            lines.append(line)

    def resolve(value):
        return re.sub(r"\[([^]]+)\]\[([^]]*)\]", lambda m:
                      f"[{m[1]}]({references.get((m[2] or m[1]).casefold(), '')})"
                      if (m[2] or m[1]).casefold() in references else m[0], value)

    index = 0
    while index < len(lines):
        line = lines[index].strip()
        if not line:
            index += 1
            continue
        if line == "<!-- pagebreak -->":
            append_break(story)
            index += 1
            continue
        if line.startswith("<!--"):
            while "-->" not in lines[index] and index + 1 < len(lines):
                index += 1
            index += 1
            continue
        fence = re.match(r"^(`{3,}|~{3,})", line)
        if fence:
            block = []
            index += 1
            while index < len(lines) and not lines[index].strip().startswith(fence[1]):
                block.append(lines[index])
                index += 1
            if index == len(lines):
                raise ValueError(f"{CURRENT_SOURCE}: unclosed code fence")
            natural_width = max((cell_count(s.expandtabs(4)) for s in block), default=0) * 8.5 * 0.6021 + 16
            if natural_width > WIDTH:
                wide_block(story, TextDiagram(block, WIDE_WIDTH))
            else:
                story.extend([TextDiagram(block, WIDTH), Spacer(1, 8)])
            index += 1
            continue
        if line.startswith((r"\[", "$$")):
            opening = line[:2]
            closing = r"\]" if opening == r"\[" else "$$"
            expression = lines[index].strip()[2:]
            while closing not in expression and index + 1 < len(lines):
                index += 1
                expression += "\n" + lines[index]
            if closing not in expression:
                raise ValueError(f"{CURRENT_SOURCE}: unclosed display formula")
            expression, tail = expression.split(closing, 1)
            story.extend(display_formula(expression))
            if tail.strip():
                story.append(paragraph(resolve(tail)))
            index += 1
            continue
        if line.startswith("|"):
            rows = []
            while index < len(lines) and lines[index].strip().startswith("|"):
                row = table_cells(lines[index].strip())
                if not all(re.fullmatch(r":?-+:?", value.replace(" ", "")) for value in row):
                    rows.append([resolve(value) for value in row])
                index += 1
            if rows:
                if len(rows[0]) > 5:
                    wide_block(story, make_table(rows, WIDE_WIDTH))
                else:
                    story.append(make_table(rows, WIDTH))
            continue
        heading = re.match(r"^(#{1,6})\s+(.*)", line)
        if heading:
            style = "TitleCN" if len(heading[1]) == 1 else "H2CN" if len(heading[1]) == 2 else "H3CN"
            item = paragraph(resolve(heading[2]), style)
            item.outline_title = re.sub(r"[*`]+", "", heading[2])
            story.append(item)
            index += 1
            continue
        if line in {"---", "***", "___"}:
            story.append(HRFlowable(width="100%", color=layout.RULE, spaceBefore=7, spaceAfter=7))
            index += 1
            continue
        if line.startswith(">"):
            story.append(paragraph(resolve(line.lstrip("> ")), "BodyCN"))
            index += 1
            continue
        if re.match(r"^(?:[-+*]|\d+[.)])\s+", line):
            story.append(paragraph(resolve(line)))
            index += 1
            continue
        parts = [line]
        index += 1
        while index < len(lines) and lines[index].strip():
            candidate = lines[index].strip()
            if re.match(r"^(?:#{1,6}\s|\||>|```|~~~|<!--|[-+*]\s|\d+[.)]\s|\\\[|\$\$|---$)", candidate):
                break
            parts.append(candidate)
            index += 1
        story.append(paragraph(resolve(" ".join(parts))))
    while story and isinstance(story[-1], PageBreak):
        story.pop()
    return story


def furniture(canvas, doc):
    page_width, page_height = canvas._pagesize
    canvas.saveState()
    header = paragraph("SOLVER SELECTION / 大范式研究", "SmallCN")
    header.wrap(page_width - 2 * MARGIN, 20)
    header.drawOn(canvas, MARGIN, page_height - 11 * mm)
    canvas.setStrokeColor(layout.RULE)
    for y in (page_height - 14 * mm, 14 * mm):
        canvas.line(MARGIN, y, page_width - MARGIN, y)
    footer = paragraph("2026-10-25 | Astra 科学稿 / 机械排版", "SmallCN")
    footer.wrap(page_width - 2 * MARGIN - 30, 20)
    footer.drawOn(canvas, MARGIN, 9.5 * mm)
    canvas.setFillColor(layout.MUTED)
    canvas.setFont("Latin", 8)
    canvas.drawRightString(page_width - MARGIN, 9.5 * mm, str(doc.page))
    canvas.restoreState()


class ResearchDocument(BaseDocTemplate):
    def afterFlowable(self, flowable):
        if hasattr(flowable, "outline_title"):
            key = "heading-" + str(self.seq.nextf("heading"))
            self.canv.bookmarkPage(key)
            self.canv.addOutlineEntry(flowable.outline_title, key, level=0, closed=False)


class EmbeddedCanvas(Canvas):
    def __init__(self, *args, **kwargs):
        kwargs["initialFontName"] = kwargs.get("initialFontName") or "Latin"
        super().__init__(*args, **kwargs)


def document(target):
    doc = ResearchDocument(target, pagesize=A4, title="大范式研究与综合取舍",
                           author="Astra research; mechanical document production",
                           subject="Research report, not a model-version delivery",
                           allowSplitting=1, pageCompression=1)
    templates = []
    for name, pagesize in (("portrait", A4), ("landscape", landscape(A4))):
        width, height = pagesize
        frame = Frame(MARGIN, BOTTOM, width - 2 * MARGIN, height - TOP - BOTTOM,
                      leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
        templates.append(PageTemplate(name, [frame], onPage=furniture, pagesize=pagesize))
    doc.addPageTemplates(templates)
    return doc


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream, strict=True)
        fields = reader.fieldnames
        if not fields or len(fields) != len(set(fields)) or any(not name for name in fields):
            raise ValueError(f"{path.name}: missing or duplicate CSV headers")
        rows = []
        for record_number, row in enumerate(reader, 1):
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"{path.name}: malformed CSV record {record_number}")
            rows.append(row)
        return fields, rows


def source_records(source_names):
    records = []
    reserved = ("lens", "source_csv", "source_record", "record_kind", "normalized_title")
    fields = list(reserved)
    for name in source_names:
        source_fields, rows = read_csv(ROOT / name)
        if set(reserved).intersection(source_fields):
            raise ValueError(f"{name}: source headers collide with registry metadata")
        fields.extend(field for field in source_fields if field not in fields)
        for number, row in enumerate(rows, 1):
            title = " ".join(unicodedata.normalize("NFKC", row.get("title", "")).split()).casefold()
            lens = "synthesis" if name in OPTIONAL_MAIN_SOURCE_CSVS else name[:2]
            records.append(dict(lens=lens, source_csv=name, source_record=number,
                                record_kind=row.get("source_type", "") or "unclassified_as_supplied",
                                normalized_title=title, **row))
    return fields, records


def source_registry(source_names=SOURCE_CSVS):
    fields, records = source_records(source_names)
    target = ROOT / "source_registry.csv"
    with target.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    titles = Counter(record["normalized_title"] for record in records if record["normalized_title"])
    return dict(status="generated", records=len(records), source_files=list(source_names),
                normalized_unique_titles=len(titles),
                duplicate_normalized_title_groups=sum(count > 1 for count in titles.values()),
                duplicate_title_additional_records=sum(count - 1 for count in titles.values()),
                records_without_title=sum(not record["normalized_title"] for record in records),
                source_type_counts_as_supplied=dict(Counter(r.get("source_type", "") for r in records)),
                reading_scope_fields=[field for field in fields if any(
                    token in field.casefold() for token in ("read", "scope", "access"))],
                note="All reading records and original fields retained, including embedded newlines; normalized titles are mechanical grouping only, not verified paper identities. No full-text reading or paper count inferred.")


def verify_pdf(payload, searchable_probe=None, require_link=False):
    tools = {name: shutil.which(name) for name in ("pdfinfo", "pdffonts", "pdftotext")}
    if not all(tools.values()):
        raise RuntimeError("Existing Poppler tools are required; no dependencies will be installed")

    def run(name, *args):
        return subprocess.run([tools[name], *args], input=payload, stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, check=True).stdout.decode("utf-8")

    info = run("pdfinfo", "-")
    fonts = run("pdffonts", "-")
    font_rows = fonts.splitlines()[2:]
    if not font_rows or any(re.search(r"\s(?:no)\s+(?:yes|no)\s+(?:yes|no)\s+\d+\s+\d+\s*$", row) for row in font_rows):
        raise RuntimeError("PDF font embedding check failed")
    text = run("pdftotext", "-", "-")
    if searchable_probe and searchable_probe not in re.sub(r"\s+", "", text):
        raise RuntimeError("Searchable Chinese text round-trip failed")
    if "\ufffd" in text:
        raise RuntimeError("PDF text extraction contains replacement characters")
    if require_link and b"/URI" not in payload:
        raise RuntimeError("PDF has no clickable source URI annotation")
    bbox = ElementTree.fromstring(run("pdftotext", "-bbox", "-", "-"))
    overflows = []
    for page_number, page in enumerate(bbox.iter("{http://www.w3.org/1999/xhtml}page"), 1):
        width, height = float(page.attrib["width"]), float(page.attrib["height"])
        for word in page.iter("{http://www.w3.org/1999/xhtml}word"):
            a = word.attrib
            if float(a["xMin"]) < MARGIN - 2 or float(a["xMax"]) > width - MARGIN + 2 or float(a["yMin"]) < 0 or float(a["yMax"]) > height:
                overflows.append(dict(page=page_number, text=word.text))
    if overflows:
        raise RuntimeError("Text exceeds printable bounds: " + json.dumps(overflows[:8], ensure_ascii=False))
    pages = int(re.search(r"^Pages:\s*(\d+)", info, re.MULTILINE)[1])
    return dict(pages=pages, embedded_fonts=True, chinese_probe_checked=bool(searchable_probe),
                replacement_characters=0, text_bounds_checked=True, clickable_uri_annotation=b"/URI" in payload)


def self_test():
    quoted = io.StringIO(newline="")
    writer = csv.DictWriter(quoted, fieldnames=["title", "read_scope"])
    writer.writeheader()
    writer.writerow({"title": "A, B", "read_scope": "line one\nline two"})
    quoted.seek(0)
    assert list(csv.DictReader(quoted, strict=True))[0]["read_scope"] == "line one\nline two"
    assert table_cells(r"| a\|b | $|x|$ | `a|b` |") == ["a|b", "$|x|$", "`a|b`"]
    sample = "\n".join([
        "# 机械排版合成样例", "中文检索探针，不使用任何科学稿。", "",
        "[论文链接](https://example.org/paper_(test).pdf?a=1&b=2)", "",
        r"行内公式 \(p(y\mid x)\)，以及 $x^2$。", "",
        r"\[", r"\sum_{i=1}^{n} p_i = 1", r"\]", "",
        r"\[", r"\boldsymbol C\mathrel{+}=A\setminus B\quad\text{s.t. }x>0", r"\]", "",
        "```text", "输入实例     ┌─────────┐", "  └────────→ │ 文本模块 │", "             └─────────┘", "```", "",
        "| 大方向 | 文本架构 | 说明 |", "| --- | --- | --- |", "| 合成行 | 保持可读 | 仅测试布局，不评价结果 |", "",
        "| 方向 | 字段二 | 字段三 | 字段四 | 字段五 | 字段六 | 字段七 | 字段八 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
        *["| 合成方向 | 逐字保留 | 多行单元格<br/>第二行 | A | B | C | D | E |" for _ in range(18)],
        "", "## 回到纵向正文", "中文检索探针。",
    ])
    buffer = io.BytesIO()
    document(buffer).build(parse(sample), canvasmaker=EmbeddedCanvas)
    result = verify_pdf(buffer.getvalue(), "中文检索探针", require_link=True)
    if UNSUPPORTED_GLYPHS or previous.MATH_WARNINGS:
        raise RuntimeError("Synthetic layout test has font or formula fallbacks")
    return dict(status="passed", synthetic_only=True, pdf_written=False,
                multiline_csv_round_trip=True, table_pipe_tokens=True, **result)


def digest_inputs(names):
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in names}


def build(output, names):
    global CURRENT_SOURCE
    story = []
    source_texts = []
    for name in names:
        if story:
            append_break(story)
        CURRENT_SOURCE = name
        source_text = (ROOT / name).read_text(encoding="utf-8")
        source_texts.append(source_text)
        story.extend(parse(source_text))
    target = io.BytesIO()
    document(target).build(story, canvasmaker=EmbeddedCanvas)
    payload = target.getvalue()
    probe = re.search(r"[\u4e00-\u9fff]{4}", "\n".join(source_texts))
    checks = verify_pdf(payload, probe[0] if probe else None,
                        require_link=any(re.search(r"https?://", text) for text in source_texts))
    if UNSUPPORTED_GLYPHS:
        raise RuntimeError("Unsupported source glyphs: " + ", ".join(sorted(UNSUPPORTED_GLYPHS)))
    (ROOT / output).write_bytes(payload)
    return dict(file=output, bytes=len(payload), sources=list(names),
                sha256=hashlib.sha256(payload).hexdigest(), **checks)


def delivery_text(notes):
    command = "source /public/home/shiys/miniconda3/etc/profile.d/conda.sh && conda activate easynco && PYTHONDONTWRITEBYTECODE=1 python -B " + str(ROOT / "render_report.py")
    prepared = notes["stage"] == "prepare"
    rows = "\n".join(f"| {item['file']} | {item['pages']} | {item['bytes']} |" for item in notes["outputs"])
    output_section = "尚未渲染任何科学正文，也未生成来源汇总；等待后续明确的定稿通知。" if prepared else "| PDF | 页数 | 字节 |\n| --- | ---: | ---: |\n" + rows
    if notes["stage"] == "appendices":
        output_section += "\n\n仅四个专题获准渲染；未读取或渲染主稿、综合对话和方向表，未读取综合者来源 CSV。收到主线程明确通知后才执行 `finalmain --finalized`，不轮询等待。"
    registry = notes["source_registry"]
    if registry.get("status") == "generated":
        output_section += (f"\n\n`source_registry.csv` 保留 {registry['records']} 条原始来源记录，"
                           f"按规范化标题机械分组为 {registry['normalized_unique_titles']} 组，"
                           f"其中 {registry['duplicate_normalized_title_groups']} 组有重复记录；"
                           "重复来源记录和原文阅读范围均保留，不将组数称为论文数或全文阅读数。")
    if notes["status"] == "failed_not_deliverable":
        output_section = "本轮机械检查失败，输出不可交付。错误：`" + notes.get("error", "unknown") + "`\n\n" + output_section
    return f"""# 大范式研究报告：机械交付说明

## 当前状态

阶段：`{notes['stage']}`。本目录是新一轮研究报告，不是模型版本交付。

机械检查状态：`{notes['status']}`。

{output_section}

## 使用方法

准备检查与合成布局测试，不读取科学稿内容、不输出样例 PDF：

```bash
{command} prepare --self-test
```

仅在收到所有稿件定稿通知后执行，文件存在本身不代表定稿：

```bash
{command} finalmain --finalized
```

该命令生成 `REPORT.pdf`、`RESEARCH_FULL.pdf`、四个专题 PDF、`source_registry.csv` 和 `render_notes.json`。完整版固定顺序为主报告、01 计算架构、02 表示学习、03 预测对象、04 数据与学习范式、综合对话与取舍。主报告与完整版不拼接其它版本的科学结论。

四个专题独立定稿时，使用同一命令前缀加 `appendices --finalized`。未传 `--finalized` 的任何正文渲染阶段都会拒绝执行；默认阶段是 `prepare`。不轮询、不等待、不自动判断稿件已完成。

## 布局与来源

- 复用 `{PREVIOUS}` 及其既有基础布局，不改旧目录；禁止导入时写入旧目录的字节码缓存。
- 复用 `{DEPENDENCIES}`、现有 Matplotlib/Pillow、DroidSansFallback 与 DejaVu 字体，不安装全局依赖。
- 中文正文使用嵌入 TrueType 字体，保持可检索；PDF 中图片与公式资产全部嵌入，阅读不依赖 `formula_assets/` 或外部文件。
- 常规表格纵向排版，超过五列切换横向页，重复表头并允许长单元格跨页；不删列、不裁剪正文，不无限缩小字体。超宽表格会明确报错而非静默漏字。
- 文本架构图按字符网格保留空格、缩进与框线，必要时改横向页，最小字号 7pt；不将代码块改写成科学示意图。
- Markdown HTTP/HTTPS 论文链接、引用式链接和直接 URL 保留为可点击的 PDF URI 注释；不联网访问、查证或补写论文。
- 四份专题来源 CSV 均使用 `csv.DictReader(..., newline='')` 逐记录读取，支持字段内换行，原字段并集保留，重复阅读记录不删除。规范化标题仅用于机械分组，不认证论文身份或阅读范围；未提供类型时标为 `unclassified_as_supplied`，不推断论文数或全文阅读数。
- `synthesis_sources.csv` 为可选输入：仅在获准 `finalmain` 时，若启动时已存在，则纳入同一次来源汇总和输入哈希核对；缺失不报错。附录阶段不读取它，避免混入综合中的记录。各记录保留 `source_csv`、`source_record` 和实读范围等原字段。
- `research_directions.csv` 属于 Astra 输入，仅作读取与输入哈希留档，不改写，也不自动替代主报告的大方向总览表。
- 公式沿用上一版的 Mathtext 处理，另兼容 ASCII 文本、集合差命令、加号关系排版包装、单记号上划线参数花括号及单个拉丁字母粗斜体；转换规则与源公式哈希写入 `render_notes.json`。不能排版的公式保留原文并写入警告；不修改科学稿，不审查公式含义。公式图片不保证数学文本可检索。

## 检查与边界

准备阶段只用合成样例检查中文检索、嵌入字体、可点击链接、宽表、文本图和 CSV 字段内换行。`render_notes.json` 记录实际结果，不把准备测试当作正式正文排版已验收。

正式生成时检查 PDF 字体嵌入、文本提取替换字符与页面边界，记录输入 SHA-256，并在结束前核对输入是否被并行作者改动；若变化则报告该轮不可交付，不覆盖科学输入。仍需针对定稿后的总览表、公式和文本架构图做少量目视抽查。

本执行者仅负责机械排版、来源聚合和交付记录。研究稿、`REQUEST.md`、`agent_trace.json` 均为只读；不审查科学结果、不提研究方向，不读取 GPU/实验状态、不跑预测或训练、不操作 goal 或 Git、不撤销其他作者的修改。

用户指定制作模型为 `gpt-6.1-sol/max`。本轮运行未提供可独立核验的实际模型身份或 reasoning effort；不冒称实际配置已认证。科学内容完全来自 Astra 源稿，本执行者仅作机械交付。
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", nargs="?", choices=("prepare", "appendices", "finalmain"), default="prepare")
    parser.add_argument("--finalized", action="store_true", help="Explicit caller confirmation; never inferred from file existence")
    parser.add_argument("--self-test", action="store_true", help="Synthetic in-memory layout checks only")
    args = parser.parse_args()
    if args.stage != "prepare" and not args.finalized:
        parser.error("Body rendering is locked. Wait for a finalization notification, then pass --finalized.")
    notes_path = ROOT / "render_notes.json"
    if args.stage == "prepare" and notes_path.exists():
        existing = json.loads(notes_path.read_text(encoding="utf-8"))
        if existing.get("outputs"):
            parser.error("Preparation will not replace an existing delivery manifest.")
    previous.MATH_WARNINGS.clear()
    LAYOUT_WARNINGS.clear()
    FORMULA_COMPATIBILITY.clear()
    UNSUPPORTED_GLYPHS.clear()
    notes = dict(stage=args.stage, status="prepared_not_rendered" if args.stage == "prepare" else "rendering",
                 timestamp_utc=datetime.now(timezone.utc).isoformat(),
                 renderer=str(Path(__file__).resolve()), reused_renderer=str(PREVIOUS),
                 dependency_directory=str(DEPENDENCIES), reportlab_version=REPORTLAB_VERSION,
                 explicit_finalized_flag=args.finalized, outputs=[],
                 source_registry=dict(status="not_built_waiting_for_finalized_inputs"),
                 self_test=dict(status="not_requested"),
                 expected_inputs=list(FINAL_INPUTS),
                 optional_main_source_inputs=list(OPTIONAL_MAIN_SOURCE_CSVS),
                 optional_main_source_policy="Only snapshot existing optional sources at explicitly finalized finalmain start; never read them during appendices.",
                 requested_executor_model="gpt-6.1-sol/max",
                 actual_executor_model_verification="unavailable_in_this_turn",
                 visual_checks=dict(status="not_performed"),
                 scope="Mechanical production only; no scientific validation, experiment/GPU state, prediction, goal or Git operations.")
    try:
        if args.self_test:
            notes["self_test"] = self_test()
        if args.stage != "prepare":
            optional_sources = tuple(name for name in OPTIONAL_MAIN_SOURCE_CSVS
                                     if args.stage == "finalmain" and (ROOT / name).is_file())
            source_names = (*SOURCE_CSVS, *optional_sources)
            required = (*FINAL_INPUTS, *optional_sources) if args.stage == "finalmain" else (*APPENDICES, *SOURCE_CSVS)
            missing = [name for name in required if not (ROOT / name).is_file()]
            if missing:
                raise FileNotFoundError("Missing finalized inputs: " + ", ".join(missing))
            before = digest_inputs(required)
            notes["input_sha256"] = before
            for name in source_names:
                read_csv(ROOT / name)
            if args.stage == "finalmain":
                read_csv(ROOT / "research_directions.csv")
                notes["outputs"].append(build("REPORT.pdf", ("REPORT.md",)))
                notes["outputs"].append(build("RESEARCH_FULL.pdf", MAIN_INPUTS))
            for name in APPENDICES:
                notes["outputs"].append(build(Path(name).with_suffix(".pdf").name, (name,)))
            notes["source_registry"] = source_registry(source_names)
            changed = [name for name, digest in digest_inputs(required).items() if before[name] != digest]
            notes["inputs_unchanged"] = not changed
            if changed:
                raise RuntimeError("Inputs changed concurrently; outputs are not deliverable: " + ", ".join(changed))
            notes["status"] = "rendered_mechanical_checks_passed"
    except Exception as error:
        notes["status"] = "failed_not_deliverable"
        notes["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        notes["formula_fallbacks"] = previous.MATH_WARNINGS
        notes["formula_compatibility"] = FORMULA_COMPATIBILITY
        notes["renderwarn_count"] = len(previous.MATH_WARNINGS)
        notes["layout_warnings"] = LAYOUT_WARNINGS
        notes["unsupported_glyphs"] = sorted(UNSUPPORTED_GLYPHS)
        notes_path.write_text(json.dumps(notes, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (ROOT / "DELIVERY.md").write_text(delivery_text(notes), encoding="utf-8")
    print(json.dumps(dict(status=notes["status"], stage=args.stage, self_test=notes["self_test"],
                          outputs=notes["outputs"], renderwarn_count=notes["renderwarn_count"]), ensure_ascii=False))


if __name__ == "__main__":
    main()

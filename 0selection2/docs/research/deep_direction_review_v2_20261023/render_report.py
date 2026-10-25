"""Render the research Markdown as a standalone, searchable Chinese PDF.

This CPU-only renderer does not import experiment code or inspect model weights.
Dependencies: ReportLab 4.2.5, plus the existing Matplotlib/fonts. During this
research, ReportLab was installed only in /tmp/shiys_astra_pdf_dependencies_compat
and supplied through PYTHONPATH, without changing the training environment.
"""

from __future__ import annotations

import html
import re
from pathlib import Path

import matplotlib
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    HRFlowable,
    KeepTogether,
    LongTable,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    TableStyle,
)

ROOT = Path(__file__).resolve().parent
FONT = Path("/usr/share/fonts/google-droid/DroidSansFallback.ttf")
LATIN_DIR = Path(matplotlib.get_data_path()) / "fonts/ttf"
pdfmetrics.registerFont(TTFont("CJK", str(FONT)))
pdfmetrics.registerFontFamily("CJK", normal="CJK", bold="CJK", italic="CJK", boldItalic="CJK")
pdfmetrics.registerFont(TTFont("Latin", str(LATIN_DIR / "DejaVuSans.ttf")))
pdfmetrics.registerFont(TTFont("LatinBold", str(LATIN_DIR / "DejaVuSans-Bold.ttf")))
pdfmetrics.registerFontFamily("Latin", normal="Latin", bold="LatinBold", italic="Latin", boldItalic="LatinBold")

INK = colors.HexColor("#24313B")
ACCENT = colors.HexColor("#135968")
MUTED = colors.HexColor("#627780")
PALE = colors.HexColor("#EEF4F6")
RULE = colors.HexColor("#CCD9DD")
PAGE_W, PAGE_H = A4
MARGIN = 17 * mm
WIDTH = PAGE_W - 2 * MARGIN
STYLES = getSampleStyleSheet()
for name, size, leading in (
    ("BodyCN", 10, 15.5),
    ("TitleCN", 22, 30),
    ("H2CN", 14, 21),
    ("H3CN", 11.2, 17),
    ("TableCN", 8.5, 12.6),
    ("SmallCN", 8.3, 12.4),
    ("CodeCN", 9, 14),
):
    STYLES.add(ParagraphStyle(
        name,
        fontName="Latin",
        fontSize=size,
        leading=leading,
        textColor=ACCENT if name in {"TitleCN", "H2CN", "H3CN"} else INK,
        alignment=TA_LEFT,
        wordWrap="CJK",
        allowWidows=0,
        allowOrphans=0,
        spaceAfter=7 if name == "BodyCN" else 5,
        spaceBefore=10 if name == "H2CN" else 0,
        keepWithNext=name in {"TitleCN", "H2CN", "H3CN"},
    ))


def inline(text: str) -> str:
    text = html.escape(text)
    text = re.sub(
        r"\[([^\]]+)\]\(([^)]+)\)",
        lambda m: f'<link href="{m[2]}" color="#135968">{m[1]}</link>'
        if m[2].startswith(("https://", "http://")) else m[1],
        text,
    )
    text = re.sub(r"\*\*([^*]+)\*\*", r'<b><font color="#135968">\1</font></b>', text)
    text = re.sub(r"`([^`]+)`", r'<font color="#345A67">\1</font>', text)
    text = re.sub(r"([\u2e80-\u9fff\uf900-\ufaff\uff00-\uffef]+)", r'<font name="CJK">\1</font>', text)
    return text


def para(text: str, style: str = "BodyCN") -> Paragraph:
    return Paragraph(inline(text), STYLES[style])


def table(rows: list[list[str]]) -> LongTable:
    count = len(rows[0])
    if count == 2:
        ratios = [0.25, 0.75]
    elif count == 3:
        ratios = [0.24, 0.38, 0.38]
    elif count == 4:
        ratios = [0.27, 0.24, 0.24, 0.25]
    elif count == 5:
        ratios = [0.26, 0.18, 0.18, 0.19, 0.19]
    else:
        ratios = [1 / count] * count
    cells = [[para(cell, "TableCN") for cell in row] for row in rows]
    result = LongTable(cells, colWidths=[WIDTH * x for x in ratios], repeatRows=1,
                       hAlign="LEFT", spaceAfter=10)
    result.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), PALE),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFB")]),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LINEBELOW", (0, 0), (-1, 0), 0.5, RULE),
        ("LINEBELOW", (0, 1), (-1, -1), 0.25, RULE),
    ]))
    return result


def parse(text: str) -> list:
    story = []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        if line == "<!-- pagebreak -->":
            story.append(PageBreak())
            i += 1
            continue
        if line.startswith("<!--"):
            i += 1
            continue
        if line.startswith("```"):
            block = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("```"):
                block.append(para(lines[i], "CodeCN"))
                i += 1
            story.extend(block)
            story.append(Spacer(1, 7))
            i += 1
            continue
        if line.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                row = [s.strip() for s in lines[i].strip().strip("|").split("|")]
                if not all(re.fullmatch(r":?-+:?", s.replace(" ", "")) for s in row):
                    rows.append(row)
                i += 1
            if any(len(row) != len(rows[0]) for row in rows):
                raise ValueError("Inconsistent table columns")
            story.append(table(rows))
            continue
        heading = re.match(r"(#{1,3})\s+(.*)", line)
        if heading:
            style = {1: "TitleCN", 2: "H2CN", 3: "H3CN"}[len(heading[1])]
            story.append(para(heading[2], style))
            i += 1
            continue
        if line == "---":
            story.append(HRFlowable(width="100%", color=RULE, spaceBefore=7, spaceAfter=7))
            i += 1
            continue
        if line.startswith(">"):
            story.append(KeepTogether([Spacer(1, 4), para(line.lstrip("> "), "H3CN"), Spacer(1, 5)]))
            i += 1
            continue
        parts = [line]
        i += 1
        while i < len(lines) and lines[i].strip() and not lines[i].lstrip().startswith(("#", "|", ">", "```", "<!--", "- ")):
            parts.append(lines[i].strip())
            i += 1
        story.append(para(" ".join(parts)))
    return story


def furniture(canvas, doc) -> None:
    canvas.saveState()
    canvas.setFont("Latin", 8)
    canvas.setFillColor(MUTED)
    header = para("SOLVER SELECTION  /  Astra 深度研究 II", "SmallCN")
    header.wrap(WIDTH, 20)
    header.drawOn(canvas, MARGIN, PAGE_H - 11 * mm)
    canvas.setStrokeColor(RULE)
    canvas.line(MARGIN, PAGE_H - 14 * mm, PAGE_W - MARGIN, PAGE_H - 14 * mm)
    canvas.line(MARGIN, 14 * mm, PAGE_W - MARGIN, 14 * mm)
    footer = para("2026-10-24 | 固定历史结果与代码分析 | 未启动或改动训练", "SmallCN")
    footer.wrap(WIDTH - 20, 20)
    footer.drawOn(canvas, MARGIN, 9.5 * mm)
    canvas.drawRightString(PAGE_W - MARGIN, 9.5 * mm, str(doc.page))
    canvas.restoreState()


def main() -> None:
    source = ROOT / "REPORT.md"
    output = ROOT / "REPORT.pdf"
    doc = SimpleDocTemplate(
        str(output), pagesize=A4, leftMargin=MARGIN, rightMargin=MARGIN,
        topMargin=20 * mm, bottomMargin=21 * mm,
        title="零选择与71%平台：代码、误差预算和下一步研究",
        author="Astra research synthesis", subject="Full18 solver selection research, second pass",
    )
    doc.build(parse(source.read_text(encoding="utf-8")), onFirstPage=furniture, onLaterPages=furniture)
    print(output)


if __name__ == "__main__":
    main()

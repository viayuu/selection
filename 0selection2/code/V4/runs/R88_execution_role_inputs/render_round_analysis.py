"""Render only ROUND_ANALYSIS.md to a two-page CPU-only PDF."""

import re
from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.font_manager import FontProperties
from fontTools.fontBuilder import FontBuilder
from fontTools.pens.cu2quPen import Cu2QuPen
from fontTools.pens.ttGlyphPen import TTGlyphPen
from fontTools.ttLib import TTFont


ROOT = Path(__file__).resolve().parent
FONT = "/usr/share/fonts/google-noto-cjk/NotoSansCJK-Regular.ttc"
PAGE_W, PAGE_H = 595.276, 841.89
MARGIN = 33
WIDTH = PAGE_W - 2 * MARGIN
FONT_PROPERTIES = FontProperties(fname=FONT)


def make_pdf_font(text, path):
    # Matplotlib labels CFF TTC data as TrueType; convert only used outlines.
    with TTFont(FONT, fontNumber=0) as source:
        cmap = {k: v for k, v in source.getBestCmap().items() if chr(k) in text}
        order = [".notdef"] + sorted(set(cmap.values()) - {".notdef"})
        glyph_set = source.getGlyphSet()
        glyphs = {}
        for name in order:
            pen = TTGlyphPen(None)
            glyph_set[name].draw(Cu2QuPen(pen, max_err=0.6, reverse_direction=True))
            glyphs[name] = pen.glyph()
        builder = FontBuilder(source["head"].unitsPerEm, isTTF=True)
        builder.setupGlyphOrder(order)
        builder.setupCharacterMap(cmap)
        builder.setupGlyf(glyphs)
        builder.setupHorizontalMetrics({name: source["hmtx"][name] for name in order})
        builder.setupHorizontalHeader(
            ascent=source["hhea"].ascent, descent=source["hhea"].descent,
        )
        builder.setupNameTable({
            "familyName": "Noto Sans CJK R88 Subset", "styleName": "Regular",
            "uniqueFontIdentifier": "NotoSansCJK-R88Subset",
            "fullName": "Noto Sans CJK R88 Subset",
            "psName": "NotoSansCJK-R88Subset",
        })
        builder.setupOS2(
            sTypoAscender=source["OS/2"].sTypoAscender,
            sTypoDescender=source["OS/2"].sTypoDescender,
            usWinAscent=source["OS/2"].usWinAscent,
            usWinDescent=source["OS/2"].usWinDescent,
        )
        builder.setupPost()
        builder.save(path)


def plain(text):
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)
    return text.replace("**", "").replace("`", "")


class Page:
    def __init__(self, pdf, number):
        self.pdf = pdf
        self.number = number
        self.fig = plt.figure(figsize=(PAGE_W / 72, PAGE_H / 72), dpi=144)
        self.fig.patch.set_facecolor("white")
        self.fig.canvas.draw()
        self.renderer = self.fig.canvas.get_renderer()
        self.y = PAGE_H - MARGIN

    def width(self, text, size):
        prop = FONT_PROPERTIES.copy()
        prop.set_size(size)
        pixels = self.renderer.get_text_width_height_descent(text, prop, False)[0]
        return pixels * 72 / self.fig.dpi

    def wrap(self, text, size, width):
        lines, line = [], ""
        for token in re.findall(r"[A-Za-z0-9_./%=+,:;-]+|.", plain(text)):
            closing = ",.;:!?\u3001\u3002\uff0c\uff1b\uff1a\uff01\uff1f\uff09\u300b"
            if line and token not in closing and self.width(line + token, size) > width:
                lines.append(line.rstrip())
                line = token.lstrip()
            else:
                line += token
        if line:
            lines.append(line.rstrip())
        return lines

    def draw_text(self, x, y, text, size, color="#20252b", align="left"):
        self.fig.text(
            x / PAGE_W, y / PAGE_H, text, fontsize=size,
            fontproperties=FONT_PROPERTIES, color=color,
            ha=align, va="top", linespacing=1.3,
        )

    def paragraph(self, text):
        size = 8.5
        for line in self.wrap(text, size, WIDTH):
            self.draw_text(MARGIN, self.y, line, size)
            self.y -= size * 1.42
        self.y -= 5

    def heading(self, text, level):
        size = 16 if level == 1 else 10
        self.y -= 2 if level == 1 else 4
        self.draw_text(MARGIN, self.y, plain(text), size, "#176b67")
        self.y -= size * 1.45 + 4

    def table(self, lines):
        rows = [
            [cell.strip() for cell in line.strip().strip("|").split("|")]
            for line in lines
        ]
        rows.pop(1)
        columns = len(rows[0])
        size = 7.7 if columns >= 7 else 8
        if columns == 7:
            weights = [0.55, 0.65, 1.05, 1.25, 1.35, 1, 1]
            if rows[0][0] == "\u9898\u76ee":
                weights = [0.65, 1, 1, 1.2, 1.2, 1.1, 1.1]
        elif columns == 8:
            weights = [0.55, 1, 1.9, 0.7, 0.55, 0.8, 1, 0.85]
        elif columns == 5:
            weights = [0.8, 0.95, 1.3, 1.3, 1.75]
        else:
            weights = [1.6, 1, 1.3, 1.3]
        widths = [WIDTH * w / sum(weights) for w in weights]
        for index, row in enumerate(rows):
            wrapped = [self.wrap(cell, size, w - 8) for cell, w in zip(row, widths)]
            height = max(len(cell) for cell in wrapped) * size * 1.3 + 7
            shade = "#e9f2f0" if index == 0 else ("#f5f6f7" if index % 2 else "#ffffff")
            self.fig.patches.append(plt.Rectangle(
                (MARGIN / PAGE_W, (self.y - height) / PAGE_H),
                WIDTH / PAGE_W, height / PAGE_H,
                transform=self.fig.transFigure, facecolor=shade,
                edgecolor="none", zorder=-1,
            ))
            x = MARGIN
            for cell, w in zip(wrapped, widths):
                self.draw_text(x + 4, self.y - 3, "\n".join(cell), size)
                x += w
            self.y -= height
        self.y -= 7

    def finish(self):
        if self.y < 32:
            raise RuntimeError(f"Page {self.number} overflows: bottom={self.y:.1f}pt")
        self.draw_text(MARGIN, 22, "R88 | locked CSV results | 2026-10-20", 7, "#677277")
        self.draw_text(PAGE_W - MARGIN, 22, f"{self.number} / 2", 7, "#677277", "right")
        self.pdf.savefig(self.fig)
        plt.close(self.fig)
        print(f"page={self.number} content_bottom_pt={self.y:.1f}")


def render():
    global FONT_PROPERTIES
    pages = (ROOT / "ROUND_ANALYSIS.md").read_text(encoding="utf-8").split("<!-- pagebreak -->")
    if len(pages) != 2:
        raise ValueError("Expected exactly two explicit pages")
    with TemporaryDirectory(prefix="r88_report_font_") as temp:
        path = Path(temp) / "noto_report.ttf"
        make_pdf_font("".join(pages) + "R88 | locked CSV results | 2026-10-20 / 12", path)
        FONT_PROPERTIES = FontProperties(fname=path)
        render_pages(pages)


def render_pages(pages):
    with PdfPages(ROOT / "ROUND_ANALYSIS.pdf", metadata={
        "Title": "R88 Locked Result Analysis",
        "Subject": "Existing CSV results only; no inference or checkpoint selection",
        "Author": "R88 result reporting",
    }) as pdf:
        for number, content in enumerate(pages, 1):
            page = Page(pdf, number)
            lines = content.strip().splitlines()
            i = 0
            while i < len(lines):
                line = lines[i].strip()
                if not line:
                    i += 1
                elif line.startswith("#"):
                    level = len(line) - len(line.lstrip("#"))
                    page.heading(line.lstrip("# "), level)
                    i += 1
                elif line.startswith("|"):
                    table = []
                    while i < len(lines) and lines[i].strip().startswith("|"):
                        table.append(lines[i])
                        i += 1
                    page.table(table)
                else:
                    paragraph = []
                    while i < len(lines) and lines[i].strip():
                        paragraph.append(lines[i].strip())
                        i += 1
                    page.paragraph(" ".join(paragraph))
            page.finish()


if __name__ == "__main__":
    render()

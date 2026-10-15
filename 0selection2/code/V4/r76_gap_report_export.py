"""Export the existing gap report to a self-contained HTML and a complete PDF."""

import base64
import csv
import html
from html.parser import HTMLParser
import json
import mimetypes
import os
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("MPLCONFIGDIR", "/tmp/r76_gap_report_export_mpl")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image


ROOT = Path(__file__).resolve().parent / "runs/R76_R45A_single_reld_mt/missed_winner_probabilities/winner_selected_gap_report"
ORDER = ("TSP", "CVRP", "ATSP", "OVRP", "VRPB", "VRPL", "VRPTW", "OVRPTW", "OVRPB", "OVRPL",
         "VRPBL", "VRPBTW", "VRPLTW", "OVRPBL", "OVRPBTW", "OVRPLTW", "VRPBLTW", "OVRPBLTW")


def read_csv(name):
    with (ROOT / name).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def uri(path):
    mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode("ascii")


SCRIPT = r'''
const report=JSON.parse(document.getElementById('data').textContent);
const images=JSON.parse(document.getElementById('export-images').textContent);
const scope=document.getElementById('scope'),problem=document.getElementById('problem');
function render(){
 const rows=report.pairs.filter(r=>(scope.value==='all'||r.zero_group)&&(problem.value==='all'||r.problem===problem.value));
 const body=document.getElementById('pairbody');body.replaceChildren();
 for(const r of rows){const tr=document.createElement('tr');
  const values=[r.problem,r.true_winner+' → '+r.selected_solver,r.count.toLocaleString(),
   r.winner_p_mean_pct.toFixed(2)+' → '+r.selected_p_mean_pct.toFixed(2),
   [r.delta_p_mean_pp,r.delta_p_median_pp,r.delta_p_p90_pp].map(x=>x.toFixed(2)).join(' / '),
   r.cost_gap_mean_pct.toFixed(4)+' / '+r.cost_gap_median_pct.toFixed(4),r.full18_regret_contribution_pp.toFixed(6)];
  for(const value of values){const td=document.createElement('td');td.textContent=value;tr.appendChild(td)}body.appendChild(tr)}
 document.getElementById('count').textContent=rows.length+' 组 · '+rows.reduce((a,r)=>a+r.count,0).toLocaleString()+' 错例';
 document.getElementById('globalplot').src=images[scope.value==='zero'?'zero':'all'];
}
scope.addEventListener('change',render);problem.addEventListener('change',render);render();
'''


class Standalone(HTMLParser):
    def __init__(self, assets):
        super().__init__(convert_charrefs=False)
        self.assets, self.parts, self.csv_used = assets, [], set()
        self.in_script, self.keep_script = False, False

    def handle_decl(self, value):
        self.parts.append("<!" + value + ">")

    def handle_starttag(self, tag, attributes):
        attrs = dict(attributes)
        if tag == "script":
            self.in_script = True
            self.keep_script = attrs.get("type") == "application/json"
            if not self.keep_script:
                return
        if tag == "img":
            attrs["src"] = self.assets[attrs["src"]]
        if tag == "a":
            target = attrs.get("href", "")
            if target.startswith("figures/"):
                attrs["href"] = "#task-" + Path(target).stem.removesuffix("_zero_winners")
            elif target == "SUMMARY.md":
                attrs["href"] = "#embedded-summary"
            elif target.endswith(".csv"):
                attrs["href"], attrs["download"] = uri(ROOT / target), Path(target).name
                self.csv_used.add(target)
        if tag == "main":
            attrs.setdefault("id", "report-top")
            self.parts.append("<style>.export-summary{white-space:pre-wrap;overflow-wrap:anywhere;font:13px/1.7 monospace}.export-downloads{display:flex;flex-wrap:wrap;gap:12px}</style>")
        serialized = "".join(" " + key if value is None else f' {key}="{html.escape(value, quote=True)}"'
                             for key, value in attrs.items())
        self.parts.append("<" + tag + serialized + ">")

    def handle_startendtag(self, tag, attrs):
        self.handle_starttag(tag, attrs)

    def handle_endtag(self, tag):
        if tag == "script":
            if self.keep_script:
                self.parts.append("</script>")
            self.in_script = self.keep_script = False
            return
        if tag == "main":
            self.appendix()
        if tag == "html":
            dynamic = {scope: self.assets[f"figures/global_top18_{scope}_pairs.png"] for scope in ("zero", "all")}
            self.parts.append('<script type="application/json" id="export-images">' + json.dumps(dynamic) + "</script><script>" + SCRIPT + "</script>")
        self.parts.append("</" + tag + ">")

    def handle_data(self, value):
        if not self.in_script or self.keep_script:
            self.parts.append(value)

    def handle_entityref(self, name):
        self.parts.append("&" + name + ";")

    def handle_charref(self, name):
        self.parts.append("&#" + name + ";")

    def appendix(self):
        self.parts.append('<section id="embedded-summary"><h2>中文摘要原文</h2><pre class="export-summary">' + html.escape((ROOT / "SUMMARY.md").read_text(encoding="utf-8")) + "</pre></section>")
        extra = [path for path in sorted(ROOT.glob("*.csv")) if path.name not in self.csv_used]
        links = "".join(f'<a download="{path.name}" href="{uri(path)}">{path.name}</a>' for path in extra)
        self.parts.append('<section><h2>其它内嵌CSV</h2><div class="export-downloads">' + links + "</div></section>")
        for task in ORDER:
            image = self.assets[f"figures/{task}_zero_winners.png"]
            self.parts.append(f'<section id="task-{task}"><h2>{task} · 原零选择组</h2><img alt="{task}成对差距" src="{image}"><p><a href="#report-top">返回总览</a></p></section>')


def export_html():
    assets = {"figures/" + path.name: uri(path) for path in sorted((ROOT / "figures").glob("*.png"))}
    parser = Standalone(assets)
    parser.feed((ROOT / "index.html").read_text(encoding="utf-8"))
    parser.close()
    output = ROOT / "winner_selected_gap_report_standalone.html"
    output.write_text("".join(parser.parts), encoding="utf-8")
    return output


def setup_font():
    font = Path("/usr/share/fonts/google-noto-cjk/NotoSansCJK-Regular.ttc")
    font_manager.fontManager.addfont(str(font))
    plt.rcParams.update({"font.family": font_manager.FontProperties(fname=str(font)).get_name(),
                         "font.size": 10, "pdf.fonttype": 42, "axes.unicode_minus": False,
                         "text.color": "#17252b"})


def page(pdf, title, size=(11.69, 8.27)):
    fig = plt.figure(figsize=size, facecolor="white")
    fig.text(.04, .95, title, fontsize=16, va="top", weight="bold")
    fig.text(.04, .035, "R76 · 已有CSV与图片导出 · 概率未校准 · 成本以native/Oracle基准分别标注", fontsize=8)
    fig.text(.96, .035, str(pdf.get_pagecount() + 1), fontsize=9, ha="right")
    return fig


def table(fig, headers, rows, box, widths=None, zero_rows=None, fontsize=9):
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    artist = ax.table(cellText=rows, colLabels=headers, colWidths=widths, cellLoc="center", bbox=box)
    artist.auto_set_font_size(False)
    artist.set_fontsize(fontsize)
    for (r, c), cell in artist.get_celld().items():
        cell.set_edgecolor("#d6dfe3")
        cell.set_linewidth(.45)
        cell.PAD = .035
        if r == 0:
            cell.set_facecolor("#e7eff2")
            cell.set_text_props(weight="bold")
            cell.set_height(cell.get_height() * 1.5)
        elif zero_rows and zero_rows[r - 1]:
            cell.set_facecolor("#edf6f1")
        elif r % 2 == 0:
            cell.set_facecolor("#f6f8f9")
    return artist


def values(row, prefix, digits):
    return " / ".join(f"{float(row[prefix + '_' + suffix]):.{digits}f}" for suffix in ("mean", "median", "p90"))


def image_page(pdf, title, filename, large=False):
    fig = page(pdf, title, (16.54, 11.69) if large else (11.69, 8.27))
    ax = fig.add_axes([.025, .075, .95, .84])
    with Image.open(ROOT / "figures" / filename) as image:
        ax.imshow(image.convert("RGB"))
    ax.set_axis_off()
    pdf.savefig(fig, dpi=180)
    plt.close(fig)


def export_pdf():
    scopes, pairs = read_csv("scope_summary.csv"), read_csv("by_pair.csv")
    winners = read_csv("by_winner.csv")
    output = ROOT / "winner_selected_gap_report.pdf"
    checkpoints, covered = {}, 0
    setup_font()
    with PdfPages(output, metadata={"Title": "R76 真实winner与实际selected：完整成对差距报告",
                                    "Subject": "已有聚合CSV和图片的只读导出，包含全部18题与全部成对记录"}) as pdf:
        fig = page(pdf, "R76 真实winner与实际selected：完整离线报告")
        fig.text(.04, .885, "仅重排已有摘要、聚合CSV和图片；不重新统计错例，不训练或推理。全部范围和原52个zero组均保留。", fontsize=10)
        rows = [["全部错例" if row["scope"] == "all" else "原zero组", f"{int(row['count']):,}",
                 values(row, "delta_p", 3), values(row, "cost_gap", 4), f"{float(row['full18_regret_contribution_pp']):.9f}"] for row in scopes]
        table(fig, ["范围", "错例数", "Δp 均值/中位/P90\n(pp)", "误选多花成本 均值/中位/P90\n(%)", "全18 regret贡献\n(pp)"],
              rows, [.04, .68, .92, .17], [.12, .10, .27, .31, .20], fontsize=9)
        rank = [[row["scope"], f"{float(row['rank2_pct']):.2f}%", f"{float(row['rank3_pct']):.2f}%", f"{float(row['rank_gt3_pct']):.2f}%",
                 row["selected_native_cost_ties"], row["native_nonminimum"], row["negative_cost_gaps"]] for row in scopes]
        table(fig, ["范围", "winner第2", "第3", ">3", "selected/native\n成本并列", "native\n非最小", "成本负差"], rank,
              [.04, .48, .92, .15], fontsize=9)
        lehd = next(row for row in winners if row["problem"] == "TSP" and row["true_winner"] == "LEHD")
        fig.text(.04, .42, f"TSP / LEHD：{lehd['count']}个错例，平均p {float(lehd['winner_p_mean_pct']):.4f}% → {float(lehd['selected_p_mean_pct']):.4f}%，Δp +{float(lehd['delta_p_mean_pp']):.4f}pp。", fontsize=11)
        lehd_rows = [[row["selected_solver"], row["count"], f"{float(row['winner_p_mean_pct']):.4f}",
                      f"{float(row['selected_p_mean_pct']):.4f}", f"{float(row['delta_p_mean_pp']):+.4f}", f"{float(row['cost_gap_mean_pct']):.6f}"]
                     for row in pairs if row["problem"] == "TSP" and row["true_winner"] == "LEHD"]
        table(fig, ["实际selected", "次数", "native均p (%)", "selected均p (%)", "Δp均值 (pp)", "误选多花成本均值 (%)"],
              lehd_rows, [.04, .21, .92, .16], fontsize=9)
        fig.text(.04, .145, "误选多花成本(%) = 100 × (selected成本 − native winner成本) / native winner成本。", fontsize=9)
        fig.text(.04, .112, "全池actual regret(%) = 100 × (selected成本 − Oracle成本) / Oracle成本；贡献 = 每例regret / (18 × 本题全体3,895)。", fontsize=9)
        fig.text(.04, .079, "概率双方来自同一错例子集；pp是概率差单位，不是胜率。排名距离 = winner_rank − 1；原生标签不改。", fontsize=9)
        pdf.savefig(fig)
        plt.close(fig)
        checkpoints["summary"] = 1
        for title, name, unit in (("概率差固定分桶", "probability_gap_bins.csv", "pp"), ("误选多花成本固定分桶", "cost_gap_bins.csv", "%")):
            rows = [[row["scope"], row["bin"] + " " + unit, row["count"], f"{float(row['missed_share_pct']):.2f}%",
                     f"{float(row['full18_regret_contribution_pp']):.9f}"] for row in read_csv(name)]
            fig = page(pdf, title + "：全部错例及原zero范围")
            table(fig, ["范围", "固定桶", "错例数", "范围内错例占比", "全18 regret贡献 (pp)"], rows, [.04, .15, .92, .70], fontsize=10)
            fig.text(.04, .09, "错例占比以各scope错例数为分母；regret贡献以每题全部3,895例、18题等权为分母。", fontsize=9)
            pdf.savefig(fig)
            plt.close(fig)
        checkpoints["global_zero"] = pdf.get_pagecount() + 1
        image_page(pdf, "全局Top18成对差距 / 原zero范围", "global_top18_zero_pairs.png", large=True)
        checkpoints["global_all"] = pdf.get_pagecount() + 1
        image_page(pdf, "全局Top18成对差距 / 全部错例", "global_top18_all_pairs.png", large=True)
        headers = ["原zero", "真实winner", "实际selected", "错例数", "均p native → selected\n(%)",
                   "Δp 均值/中位/P90\n(pp)", "误选多花成本\n均值/中位/P90 (%)", "全18 regret\n贡献 (pp)"]
        for task in ORDER:
            image_page(pdf, task + " / 原零选择组的成对概率与成本", task + "_zero_winners.png")
            task_pairs = [row for row in pairs if row["problem"] == task]
            for start in range(0, len(task_pairs), 22):
                block = task_pairs[start:start + 22]
                checkpoints.setdefault(task + "_pairs", pdf.get_pagecount() + 1)
                fig = page(pdf, f"{task} / 全部成对错例 · 第{start + 1}–{start + len(block)}组 / 共{len(task_pairs)}组")
                rows, zero = [], []
                for row in block:
                    is_zero = row["zero_group"].lower() == "true"
                    zero.append(is_zero)
                    rows.append(["是" if is_zero else "否", row["true_winner"], row["selected_solver"], row["count"],
                                 f"{float(row['winner_p_mean_pct']):.2f} → {float(row['selected_p_mean_pct']):.2f}",
                                 values(row, "delta_p", 2) if "delta_p_mean" in row else
                                 " / ".join(f"{float(row['delta_p_' + suffix + '_pp']):.2f}" for suffix in ("mean", "median", "p90")),
                                 " / ".join(f"{float(row['cost_gap_' + suffix + '_pct']):.4f}" for suffix in ("mean", "median", "p90")),
                                 f"{float(row['full18_regret_contribution_pp']):.6f}"])
                height = min(.72, (len(block) + 1.5) * .032)
                table(fig, headers, rows, [.035, .84 - height, .93, height],
                      [.055, .115, .13, .065, .15, .18, .20, .105], zero_rows=zero, fontsize=9)
                fig.text(.04, .09, "包含原zero及其余全部错例；浅绿色标记原zero组。按已有CSV顺序保留，不截断为屏幕可见行。", fontsize=9)
                pdf.savefig(fig)
                plt.close(fig)
                covered += len(block)
        if covered != len(pairs):
            raise ValueError("Some saved pair records were not exported")
        pages = pdf.get_pagecount()
    return output, pages, len(pairs), checkpoints


def main():
    standalone = export_html()
    print(f"Standalone HTML ready: {standalone}", flush=True)
    pdf, pages, pairs, checkpoints = export_pdf()
    print(json.dumps(dict(pdf=str(pdf), pdf_pages=pages, pdf_bytes=pdf.stat().st_size,
                          standalone_html=str(standalone), html_bytes=standalone.stat().st_size,
                          pair_rows_exported=pairs, task_figures=18, representative_pages=checkpoints), indent=2))


if __name__ == "__main__":
    main()

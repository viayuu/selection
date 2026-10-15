"""Render the short interpretation without rerunning any analysis."""

from pathlib import Path

import markdown


ROOT = Path(__file__).resolve().parent / "runs" / "Plateau_diagnosis_R70"


def main():
    source = ROOT / "SUMMARY.md"
    body = markdown.markdown(source.read_text(encoding="utf-8"), extensions=["tables"])
    page = """<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Selector Analysis: Short Report</title>
<style>
body { margin: 0; background: #fafafa; color: #202528;
       font: 16px/1.7 "Noto Sans CJK SC", "Microsoft YaHei", sans-serif; }
main { max-width: 1100px; margin: auto; padding: 28px 24px 64px; }
h1 { font-size: 30px; margin: 0 0 20px; }
h2 { font-size: 22px; margin: 38px 0 12px; padding-top: 18px;
     border-top: 1px solid #d8dddf; }
h3 { font-size: 18px; margin: 26px 0 12px; }
p { margin: 12px 0; }
table { border-collapse: collapse; width: 100%; font-size: 14px;
        display: block; overflow-x: auto; background: #fff; }
th, td { padding: 9px 12px; border: 1px solid #d8dddf; white-space: nowrap; }
th { background: #edf1f2; }
strong { color: #173f42; }
a { color: #13676d; }
li { margin: 7px 0; }
@media (max-width: 600px) { main { padding: 20px 14px 40px; }
                          h1 { font-size: 26px; } h2 { font-size: 20px; } }
@media print { body { background: #fff; } main { max-width: none; padding: 0; }
               table { display: table; font-size: 10px; } th, td { padding: 4px; }
               h2, h3 { break-after: avoid; } tr { break-inside: avoid; } }
</style>
</head>
<body><main>""" + body + "</main></body></html>\n"
    target = ROOT / "summary.html"
    target.write_text(page, encoding="utf-8")
    print(target)


if __name__ == "__main__":
    main()

# Scripts

这些脚本用于复现/更新 `adaptive/` 目录的论文素材。

## 1) 下载 PDF（按 `adaptive/papers.yaml`）

```bash
python adaptive/scripts/download_pdfs.py
```

## 2) 从 PDF 提取纯文本（可选，本地检索用）

```bash
python adaptive/scripts/pdf_to_text.py
```

输出默认写到 `adaptive/fulltext/`（已在 `.gitignore` 中忽略）。

> 提示：提取文本仅建议用于个人学习/检索；请自行遵守论文许可与转载规则。


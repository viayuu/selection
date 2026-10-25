# 大范式研究报告：机械交付说明

## 当前状态

阶段：`finalmain`。本目录是新一轮研究报告，不是模型版本交付。

机械检查状态：`rendered_mechanical_checks_passed`。

| PDF | 页数 | 字节 |
| --- | ---: | ---: |
| REPORT.pdf | 13 | 378520 |
| RESEARCH_FULL.pdf | 46 | 883786 |
| 01_computational_architectures.pdf | 7 | 275297 |
| 02_representation_learning.pdf | 6 | 252927 |
| 03_prediction_objects.pdf | 6 | 215066 |
| 04_data_and_learning_paradigms.pdf | 5 | 176677 |

`source_registry.csv` 保留 74 条原始来源记录，按规范化标题机械分组为 71 组，其中 2 组有重复记录；重复来源记录和原文阅读范围均保留，不将组数称为论文数或全文阅读数。

## 本轮交付检查

- `REPORT.pdf` 为 13 页；`RESEARCH_FULL.pdf` 为 46 页。PDF 的中文字体、图片和公式均嵌入，下载后可独立阅读，来源 URI 可点击；公式图片不保证数学文本可检索。
- 仅抽查主稿第1页关键总览表及第6页修复后的公式，未见明显裁切、重叠或缺字。未重新全面目检附录，先前附录视觉记录已保存在 `render_notes.json` 的 `appendix_visual_evidence_reused`。这是机械版式检查，不是科学、数学或实读范围认证。
- 本轮唯一排版兼容修复是将 renderer 中单记号 `\overline C` 的参数规范为等价的 `\overline{C}`；科学源稿未改。最终公式回退、不支持字形及已检测到的未解决排版问题均为 0；这不表示已穷尽检查所有页面。
- 来源仅汇总 `01_sources.csv` 至 `04_sources.csv`：25、29、10、10 条，共 74 条原始记录，按规范化标题机械分为 71 组，2 个重复标题组包含 3 条额外记录。原始来源、实读范围和重复记录逐字段保留，不将 74 或 71 称为论文数或全文阅读数。`synthesis_sources.csv` 在输入快照时不存在，未补造；`research_directions.csv` 是方向表，不混入来源计数。
- 已核对冻结输入与最终 PDF 的 SHA-256，以及完整版各部分的起始页。未修改训练、GPU、goal、Git 或本报告目录以外的文件。

## 完整版页码

| 部分 | 起始页 | 页数 |
| --- | ---: | ---: |
| 主报告 | 1 | 13 |
| 01 计算架构 | 14 | 7 |
| 02 表示学习 | 21 | 6 |
| 03 预测对象 | 27 | 6 |
| 04 数据与学习范式 | 33 | 5 |
| 综合对话与取舍 | 38 | 9 |

## 本轮全部路径

下列 27 个文件内容新增或变更，均在本报告目录内。四份独立附录由 `finalmain` 按既定流程机械重生成，未重做全面视觉检查；`formula_assets/` 中新增 16 个公式 PNG 和 2 个版式抽查 PNG，最终 PDF 不依赖这些外置文件。

- [01_computational_architectures.pdf](/public/home/shiys/0selection2/docs/research/major_paradigm_review_20261025/01_computational_architectures.pdf)
- [02_representation_learning.pdf](/public/home/shiys/0selection2/docs/research/major_paradigm_review_20261025/02_representation_learning.pdf)
- [03_prediction_objects.pdf](/public/home/shiys/0selection2/docs/research/major_paradigm_review_20261025/03_prediction_objects.pdf)
- [04_data_and_learning_paradigms.pdf](/public/home/shiys/0selection2/docs/research/major_paradigm_review_20261025/04_data_and_learning_paradigms.pdf)
- [DELIVERY.md](/public/home/shiys/0selection2/docs/research/major_paradigm_review_20261025/DELIVERY.md)
- [REPORT.pdf](/public/home/shiys/0selection2/docs/research/major_paradigm_review_20261025/REPORT.pdf)
- [RESEARCH_FULL.pdf](/public/home/shiys/0selection2/docs/research/major_paradigm_review_20261025/RESEARCH_FULL.pdf)
- [formula_assets/003be281570a848ddae9.png](/public/home/shiys/0selection2/docs/research/major_paradigm_review_20261025/formula_assets/003be281570a848ddae9.png)
- [formula_assets/16ea326f34478232660b.png](/public/home/shiys/0selection2/docs/research/major_paradigm_review_20261025/formula_assets/16ea326f34478232660b.png)
- [formula_assets/18619717c7d6f7ab2ab2.png](/public/home/shiys/0selection2/docs/research/major_paradigm_review_20261025/formula_assets/18619717c7d6f7ab2ab2.png)
- [formula_assets/2bb1b6ccaf8a3a979292.png](/public/home/shiys/0selection2/docs/research/major_paradigm_review_20261025/formula_assets/2bb1b6ccaf8a3a979292.png)
- [formula_assets/31ea7acfe154b0f2fbd5.png](/public/home/shiys/0selection2/docs/research/major_paradigm_review_20261025/formula_assets/31ea7acfe154b0f2fbd5.png)
- [formula_assets/364b9077d34241239c5d.png](/public/home/shiys/0selection2/docs/research/major_paradigm_review_20261025/formula_assets/364b9077d34241239c5d.png)
- [formula_assets/6b67afa14cdc5ff204be.png](/public/home/shiys/0selection2/docs/research/major_paradigm_review_20261025/formula_assets/6b67afa14cdc5ff204be.png)
- [formula_assets/76f201bda98555eb774a.png](/public/home/shiys/0selection2/docs/research/major_paradigm_review_20261025/formula_assets/76f201bda98555eb774a.png)
- [formula_assets/87eb86bfd2a4b264b7ce.png](/public/home/shiys/0selection2/docs/research/major_paradigm_review_20261025/formula_assets/87eb86bfd2a4b264b7ce.png)
- [formula_assets/92b2f12a2047407a05c7.png](/public/home/shiys/0selection2/docs/research/major_paradigm_review_20261025/formula_assets/92b2f12a2047407a05c7.png)
- [formula_assets/a671182f025a739ace8f.png](/public/home/shiys/0selection2/docs/research/major_paradigm_review_20261025/formula_assets/a671182f025a739ace8f.png)
- [formula_assets/b0cbf2db448b47703f06.png](/public/home/shiys/0selection2/docs/research/major_paradigm_review_20261025/formula_assets/b0cbf2db448b47703f06.png)
- [formula_assets/cf8d4c1cb30f6079e0cb.png](/public/home/shiys/0selection2/docs/research/major_paradigm_review_20261025/formula_assets/cf8d4c1cb30f6079e0cb.png)
- [formula_assets/d35490fb80e5335b33a8.png](/public/home/shiys/0selection2/docs/research/major_paradigm_review_20261025/formula_assets/d35490fb80e5335b33a8.png)
- [formula_assets/d9a3b951970221002ab2.png](/public/home/shiys/0selection2/docs/research/major_paradigm_review_20261025/formula_assets/d9a3b951970221002ab2.png)
- [formula_assets/ee6fcdb818542d0acd99.png](/public/home/shiys/0selection2/docs/research/major_paradigm_review_20261025/formula_assets/ee6fcdb818542d0acd99.png)
- [formula_assets/layout_check_REPORT_p01.png](/public/home/shiys/0selection2/docs/research/major_paradigm_review_20261025/formula_assets/layout_check_REPORT_p01.png)
- [formula_assets/layout_check_REPORT_p06.png](/public/home/shiys/0selection2/docs/research/major_paradigm_review_20261025/formula_assets/layout_check_REPORT_p06.png)
- [render_notes.json](/public/home/shiys/0selection2/docs/research/major_paradigm_review_20261025/render_notes.json)
- [render_report.py](/public/home/shiys/0selection2/docs/research/major_paradigm_review_20261025/render_report.py)

另有以下 1 个文件由聚合器重写，但与本轮开始时内容完全一致，不计为内容变更：

- [source_registry.csv](/public/home/shiys/0selection2/docs/research/major_paradigm_review_20261025/source_registry.csv)

## 使用方法

准备检查与合成布局测试，不读取科学稿内容、不输出样例 PDF：

```bash
source /public/home/shiys/miniconda3/etc/profile.d/conda.sh && conda activate easynco && PYTHONDONTWRITEBYTECODE=1 python -B /public/home/shiys/0selection2/docs/research/major_paradigm_review_20261025/render_report.py prepare --self-test
```

仅在收到所有稿件定稿通知后执行，文件存在本身不代表定稿：

```bash
source /public/home/shiys/miniconda3/etc/profile.d/conda.sh && conda activate easynco && PYTHONDONTWRITEBYTECODE=1 python -B /public/home/shiys/0selection2/docs/research/major_paradigm_review_20261025/render_report.py finalmain --finalized
```

该命令生成 `REPORT.pdf`、`RESEARCH_FULL.pdf`、四个专题 PDF、`source_registry.csv` 和 `render_notes.json`。完整版固定顺序为主报告、01 计算架构、02 表示学习、03 预测对象、04 数据与学习范式、综合对话与取舍。主报告与完整版不拼接其它版本的科学结论。

四个专题独立定稿时，使用同一命令前缀加 `appendices --finalized`。未传 `--finalized` 的任何正文渲染阶段都会拒绝执行；默认阶段是 `prepare`。不轮询、不等待、不自动判断稿件已完成。

## 布局与来源

- 复用 `/public/home/shiys/0selection2/docs/research/comprehensive_direction_review_20261025/render_report.py` 及其既有基础布局，不改旧目录；禁止导入时写入旧目录的字节码缓存。
- 复用 `/tmp/shiys_astra_pdf_dependencies_compat`、现有 Matplotlib/Pillow、DroidSansFallback 与 DejaVu 字体，不安装全局依赖。
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

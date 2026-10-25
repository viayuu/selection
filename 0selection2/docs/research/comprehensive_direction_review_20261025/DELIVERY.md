# Astra 研究报告交付说明

日期：2026-10-25。基于 Astra 综合者确认定稿的 REPORT.md、SYNTHESIS_DIALOGUE.md、research_directions.csv 执行 finalmain；本文件仅记录机械交付，不增加研究建议，也不表示用户已经批准实施这些建议。

交付目录：[comprehensive_direction_review_20261025](/public/home/shiys/0selection2/docs/research/comprehensive_direction_review_20261025)。

## PDF 与阅读方式

| 文件 | 实际页数 | 字节 | 内容 |
| --- | ---: | ---: | --- |
| [REPORT.pdf](/public/home/shiys/0selection2/docs/research/comprehensive_direction_review_20261025/REPORT.pdf) | 16 | 430258 | 主报告，建议先读 |
| [RESEARCH_FULL.pdf](/public/home/shiys/0selection2/docs/research/comprehensive_direction_review_20261025/RESEARCH_FULL.pdf) | 55 | 1090951 | 主报告、三个专题、综合质询与取舍 |
| [01_decision_and_identifiability.pdf](/public/home/shiys/0selection2/docs/research/comprehensive_direction_review_20261025/01_decision_and_identifiability.pdf) | 10 | 368768 | 专题 01 |
| [02_representation_and_computation.pdf](/public/home/shiys/0selection2/docs/research/comprehensive_direction_review_20261025/02_representation_and_computation.pdf) | 7 | 263306 | 专题 02 |
| [03_learning_and_optimization.pdf](/public/home/shiys/0selection2/docs/research/comprehensive_direction_review_20261025/03_learning_and_optimization.pdf) | 14 | 409154 | 专题 03 |

完整版页码：主报告 1-16；专题 01 为 17-26；专题 02 为 27-33；专题 03 为 34-47；综合质询与取舍为 48-55。

所有 PDF 均自包含，中文正文字体和公式图片已嵌入，不需要旁置图片或联网阅读。正文可检索；公式采用 Matplotlib 图片，不保证能作为数学文本提取。

## 辅助文件

- [research_directions.csv](/public/home/shiys/0selection2/docs/research/comprehensive_direction_review_20261025/research_directions.csv)：Astra 定稿的方向表，5386 字节，本轮未修改。
- [source_registry.csv](/public/home/shiys/0selection2/docs/research/comprehensive_direction_review_20261025/source_registry.csv)：机械合并的来源表，31915 字节，保留来源表名、专题、原始字段及规范化标题。
- [render_notes.json](/public/home/shiys/0selection2/docs/research/comprehensive_direction_review_20261025/render_notes.json)：本轮生成文件、页数、来源统计和公式回退记录，1256 字节。
- [render_report.py](/public/home/shiys/0selection2/docs/research/comprehensive_direction_review_20261025/render_report.py)：交付脚本，10152 字节，本轮无需兼容修补。
- [formula_assets](/public/home/shiys/0selection2/docs/research/comprehensive_direction_review_20261025/formula_assets)：公式渲染缓存，仅供重建；阅读 PDF 不依赖此目录。
- [DELIVERY.md](/public/home/shiys/0selection2/docs/research/comprehensive_direction_review_20261025/DELIVERY.md)：本交付说明。

科学源稿：[REPORT.md](/public/home/shiys/0selection2/docs/research/comprehensive_direction_review_20261025/REPORT.md)、[SYNTHESIS_DIALOGUE.md](/public/home/shiys/0selection2/docs/research/comprehensive_direction_review_20261025/SYNTHESIS_DIALOGUE.md)、[01_decision_and_identifiability.md](/public/home/shiys/0selection2/docs/research/comprehensive_direction_review_20261025/01_decision_and_identifiability.md)、[02_representation_and_computation.md](/public/home/shiys/0selection2/docs/research/comprehensive_direction_review_20261025/02_representation_and_computation.md)、[03_learning_and_optimization.md](/public/home/shiys/0selection2/docs/research/comprehensive_direction_review_20261025/03_learning_and_optimization.md)。

原始来源表：[01_sources.csv](/public/home/shiys/0selection2/docs/research/comprehensive_direction_review_20261025/01_sources.csv)、[02_sources.csv](/public/home/shiys/0selection2/docs/research/comprehensive_direction_review_20261025/02_sources.csv)、[03_sources.csv](/public/home/shiys/0selection2/docs/research/comprehensive_direction_review_20261025/03_sources.csv)。

## 模型分工

用户指定 gpt-6-astra / max 负责论文研究、三个专题、科学综合与研究建议；gpt-6.1-sol / max 仅负责 PDF 排版、来源机械汇总与本交付说明，不参与新思路或科学结果审查。上述名称记录用户指定的分工，不构成服务端实际模型身份的独立认证。

## 生成命令

```bash
source /public/home/shiys/miniconda3/etc/profile.d/conda.sh && conda activate easynco
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/tmp/shiys_astra_pdf_dependencies_compat python /public/home/shiys/0selection2/docs/research/comprehensive_direction_review_20261025/render_report.py finalmain
```

使用现成 ReportLab 临时依赖、Matplotlib 和 DroidSansFallback；未安装新全局库，未改训练环境。重建还依赖现有布局脚本 [deep_direction_review_v2_20261023/render_report.py](/public/home/shiys/0selection2/docs/research/deep_direction_review_v2_20261023/render_report.py)；这不影响已生成 PDF 的独立阅读。

## 来源与限制

来源汇总共 51 条记录：31 条论文阅读记录，对应 30 个规范化唯一论文标题，另有 20 条本地证据。保留重复论文在不同专题中的阅读记录，不宣称逐页读完 30 篇。阅读范围、访问状态和证据限制均沿用 Astra 来源表，未独立重读或认证。

本轮实际运行 finalmain；用 pdfinfo 确认五份 PDF 页数，用 pdftotext 确认两份主 PDF 正文和完整版各部分存在，文本提取中的替换字符均为 0。目视检查主报告第 1、3 页及完整版第 48 页；三个附录已在上一交付阶段完成首页和公式或表格页抽查。本次 renderwarn=0，formula_fallbacks 为空。

页数、字节和零排版回退仅表示机械生成与抽样排版检查通过，不代表数学正确性、研究结论或实验结果获得独立复核；研究意见维持原稿的同族 LLM provisional 证据限制。

九份科学源稿和原始来源表的生成前后 SHA-256 一致。本轮未改科学文本、agent_trace、训练代码、GPU 或协议，未启动、停止或重启训练，未操作 goal、commit/push 或记忆。

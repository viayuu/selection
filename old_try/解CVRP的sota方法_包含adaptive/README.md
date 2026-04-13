# CVRP Adaptive Papers (Literature Pack)

本目录用于收集若干“带自适应/算法选择机制”的 CVRP 经典/近年 SOTA 论文，并整理它们的 **adaptive/algorithm selection** 设计点，便于你后续做 solver/pipeline 选择研究的对照与启发。

## 重要说明（版权与可复现）

- `adaptive/pdfs/` 中的 PDF 来自公开可访问的作者/机构仓库或预印本（arXiv/HAL/机构库）。不同论文可能有不同许可/转载条件；如需再分发请自行核对许可。
- 由于版权合规原因，本目录 **不包含** “论文全文逐字转写为 Markdown”的内容；每篇仅提供结构化笔记与对 **adaptive** 机制的理解摘要。

## 目录结构

- `adaptive/pdfs/`：下载的论文 PDF（按编号命名）。
- `adaptive/notes/`：每篇论文的阅读笔记（重点：adaptive/选择机制 + 与你研究的接口）。
- `adaptive/adaptive_synthesis.md`：跨论文对比总结（统一视角梳理“自适应=选择什么、依据什么、怎么更新、作用在哪一层”）。
- `adaptive/operator_selection_classics.md`：跨领域“算子选择/启发式选择/AOS/超启发式/bandit/RL”经典方法综述（偏内层迭代算子）。
- `adaptive/instance_based_algorithm_selection.md`：跨领域“根据问题性质按实例选择算法/portfolio/schedule/ISAC/AutoFolio”综述（偏外层算法选择）。
- `adaptive/papers.yaml`：论文元信息（标题/DOI/来源链接/本地路径）。
- `adaptive/scripts/`：可复现脚本（重新下载 PDF、（可选）把 PDF 提取成纯文本供本地检索等）。

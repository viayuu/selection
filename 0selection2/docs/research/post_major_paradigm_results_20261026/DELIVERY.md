# 2026-10-26 完整研究版本

主要交付：[中文 REPORT.pdf](REPORT.pdf)，18 页，全文可搜索、字体和公式图像均已内嵌，公式同时保留机器可读的原始表达式。正文完整来自 [REPORT.md](REPORT.md)；机器可读取的附表为 [sources.csv](sources.csv)（10 篇原论文）及 [directions.csv](directions.csv)（4 个方向）。科学正文和 CSV 单元格未修改。

科学研究调用请求 `gpt-6-astra/max`；本次排版、核对与发布调用为 `f7ece9228eec9e192e59a6bb`，请求 `gpt-6.1-sol/max`。精确 CLI 标志与存活进程身份已核对，但它们不是独立的 provider 生成路由证明。

本版本绑定七个唯一 VAL checkpoint 的保存证据。R109/R112/R113 没有提供稳定的 whole18 联合突破；R114 是 `complete_labels` 的 ACQUISITION/response pilot，尚无已训练的响应函数学习器，不能据此宣告下游学习成功或失败。后续方向仅为研究建议，本次未启动新模型或求解任务；whole18 持续改进目标仍 ACTIVE。

[已验证证据包](../../../runs/research_followthrough_20261026/evidence_csv_materialization/evidence_pack.md)、[280 行指标](../../../runs/research_followthrough_20261026/evidence_csv_materialization/metrics.csv)和 [checkpoint 表](../../../runs/research_followthrough_20261026/evidence_csv_materialization/checkpoints.csv)随版本发布。保存的 native int64 `ind` 已核对原 FP64 winner；SBS 在完整冻结 TRAIN 拟合，ALL 按题等权。研究输入规范化 snapshot SHA256：`4f74c4802781fff805a62c709c377baa58feabc1315d297a3a764aa95444fd89`。

旧严格字节检查保持失败记录；92 项 CSV 差异仅为 CRLF→LF，追加式 [materialization](../../../runs/research_followthrough_20261026/evidence_csv_materialization/publication_csv_materialization.json)及其 [主线程 authority](../../../runs/research_followthrough_20261026/evidence_csv_reconciliation.json)保留双哈希与单元格等价说明。[publication_manifest.json](publication_manifest.json)另绑定本次实际提交的规范化文本与二进制 PDF SHA256，不把工作区原始字节和 Git blob 混为一谈。

验证：43 项监控器测试、7 项发布防护测试及 wrapper shell 语法通过；14 行 ALL、18 行逐题锚点、28 项 NSS 对照、54 项逐题差值与已验证指标一致。PDF 808 个正文片段/表格单元格完整回读，无缺字、替换字符或页面越界；抽查第 1、4、6、10、14、18 页版面。渲染代码和元数据位于 monitor_root；原实验、证据及历史回执保持不变。所有发布路径精确列明，运行日志、action 目录、权重、原始数组、缓存及 ready 文件不在发布范围。

复现机械排版（不执行模型）：

```bash
source /public/home/shiys/miniconda3/etc/profile.d/conda.sh
conda activate easynco
python -B /public/home/shiys/0selection2/runs/research_followthrough_20261026/render_research.py
```

发布仅执行一次；本机 `publication_attempt.json` 在暂存前持久化，`publication_receipt.json` 在实际远端提交及全部产物哈希确认后落盘。这两项事务回执不另建第二个提交。重启须先对账，禁止盲目重复 commit/push。

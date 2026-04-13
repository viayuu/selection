# AGENTS.md

本文件用于给后续自动化助手提供 `/public/home/zhoucl/shiys` 整个工作区的全局记忆与协作约定；适用范围为 `shiys/` 根目录及其子目录，除非某个更深层目录下有自己的 `AGENTS.md` 覆盖更具体规则。

## 0. 最重要的总认识

- `shiys/` 不是单一项目，而是一个长期研究工作区。
- 这里混合了：
  - 当前毕业论文主线代码与论文
  - EasyNCO 平台
  - NSS / ReLD / URS 等不同历史项目
  - 文献、方法源码、实验日志、自动化工具、浏览器扩展、agent 本地状态
- 如果用户说“当前论文”“当前主线”“现在这个研究”，默认优先指向：
  - `old_try/2cmab2`
  - `old_try/sustechthesis-1.3.9`
- 如果用户说“平台”“跑评测”“EasyNCO”，默认指向：
  - `EasyNCO/`

## 1. 环境与运行约定

- 当前是在用户服务器上直接工作，可以直接跑代码。
- Python / PyTorch 相关任务默认先执行：
  - `conda activate easynco_zhoucl`
- 长时间运行的训练、评测、监控、LaTeX 编译，优先放在 `tmux` 中。
- 机器有直接 GPU 访问：
  - `1x RTX 3090 24GB`
- 根目录已经初始化为 git 总控仓库，但工作区内部还存在若干自带 `.git` 的独立子仓库；不要把它们的内部历史误当成根仓库内容去重写。

## 2. 根目录内容总览

### 2.1 隐藏目录：工具、状态、配置

- `.agents/`
  - 本地 agent skills 目录。
  - 主要是可安装或自定义 skill 的存放地。
- `.bin/`
  - 个人命令行辅助脚本集合。
  - 包含 `claude-*`、`codex-*`、`tokscale-*`、shell 启动器等。
- `.claude/`
  - Claude 工具的本地状态、会话、插件、规则、脚本、日志、下载缓存。
  - 这是工具运行目录，不是研究代码主仓。
- `.codex/`
  - Codex 本地状态、memory、skills、sessions、rules、logs。
  - `.codex/AGENTS.md` 里保留了一份偏“当前论文主线”的记忆摘要。
- `.cache/`
  - 各类缓存。
- `.config/`
  - 用户级配置，如 `go`、`matplotlib`、`opencode`、`tokscale`。
- `.local/`
  - 本地工具状态和共享数据。
- `.npm/`
  - npm 缓存和日志。
- `.vscode/`
  - 当前工作区的 VSCode 配置、任务、终端脚本。
  - 这是工作区级配置，和研究代码配合较强。

### 2.2 根目录普通文件

- `FULL_PROJECT_CONTEXT_FOR_NEXT_MODEL_2026-04-13.md`
  - 当前最重要的项目交接总文档。
  - 新模型理解当前毕业论文主线时，建议第一份先读它。
- `skills-lock.json`
  - skills 相关锁文件。
- `初始化数据-实验设计.md`
  - 中文实验设计 / 初始化相关说明。
- `平台方法统计统计.md`
  - 早期 motivation / 方法统计的重要材料。

## 3. 顶层项目目录地图

### 3.1 `old_try/`：当前论文与大多数研究演化的主聚集区

这是整个工作区里最关键的研究目录，包含当前主线、历史分支、分析输出、论文工程和旧原型。

关键子目录：

- `old_try/2cmab2/`
  - 当前毕业论文主方法实现。
  - 主题：`TSP + CVRP` 上的 `initialization-only` 实例级方法选择。
  - 建模：单步 `contextual bandit`。
  - 主方法：`Neural-LinUCB`。
  - 关键文件：
    - `run_offline.py`
    - `evaluate_nss_benchmarks.py`
    - `build_dataset.py`
    - `features.py`
    - `encoders.py`
    - `neural_linucb.py`
    - `default_settings.py`
  - 当前论文默认数据源是 `nss`，不是老的 EasyNCO 结果 JSON 主线。

- `old_try/2l2r/`
  - 与 `2cmab2` 并行的监督 learning-to-rank 对照分支。
  - 不是当前毕业论文主线，但对“bandit vs supervised ranking”对照很有价值。

- `old_try/sustechthesis-1.3.9/`
  - 当前 thesis 和答辩 slides 的 LaTeX 工程。
  - 关键文件：
    - `main.tex`
    - `slides.tex`
    - `sections/thesis/introduction.tex`
    - `sections/thesis/method.tex`
    - `sections/thesis/experiments.tex`
    - `sections/thesis/results.tex`
    - `sections/thesis/conclusion.tex`

- `old_try/analysis-output/`
  - 当前较系统的分析输出目录。
  - 特别重要的是：
    - `2026-04-06-full-project-results/analysis-report.md`

- `old_try/1two_gate/`
  - 双 gate RL 原型。
  - 主要价值是负结果和研究路径演化证据，不是当前主线。

- `old_try/1step/`
  - 每步 operator 选择原型。
  - 主要价值是更细粒度控制为何没有成为主线的证据。

- `old_try/1online/`
  - 单实例 fully-online RL 原型。
  - 主要是 smoke / prototype 价值。

- `old_try/2cmab/`
  - `2cmab2` 的早期 contextual bandit 版本。
  - 已基本被 `2cmab2` 取代。

- `old_try/9motivation/`
  - motivation 验证相关材料。

- `old_try/0old/`
  - 更老的历史实现与整理目录。

- `old_try/model原作者/`
  - 原作者模型或参考实现相关内容。

- `old_try/参考文献/`
  - 参考资料与文献整理。

- `old_try/解CVRP的sota方法_包含adaptive/`
  - CVRP SOTA 方法资料整理。

- `old_try/9Auto-claude-code-research-in-sleep/`
  - 自动化研究相关旧副本，且自带独立 `.git`。

- `old_try/Claudix/`
  - 独立 git 子项目。

- `old_try/codex-oauth-automation-extension-3.0.0/`
  - 旧版 codex oauth 扩展副本。

- `old_try/FULL_PROJECT_CONTEXT_FOR_PAPER_WRITING_2026-04-06.md`
  - 旧版全项目论文写作交接文档，仍有参考价值。

- `old_try/AutiSAEA.md` / `old_try/AutoSAEA.md`
  - 旧的笔记或方案文档。

### 3.2 `EasyNCO/`：统一平台与评测执行主目录

- 这是用户自己的 NCO 平台代码目录。
- 包含多问题、多方法统一训练 / 评测 / exact solver / GUI / benchmark 支撑。
- 关键入口：
  - `train.py`
  - `eval.py`
  - `configs.yaml`
  - `settings/`
  - `neural_solvers/`
  - `phases/`
  - `data/`
  - `exact_solvers/`
- 还包含 NSS 评测脚本：
  - `run_nss_eval_suite.py`
  - `launch_nss_eval.sh`
  - `watch_nss_eval_progress.py`
  - `pause_nss_eval.sh`
  - `stop_nss_eval.sh`
- `results/` 目录下通常有较多评测输出。

### 3.3 `9nss论文/`：NSS 论文材料与复现 / 变体目录

- 这个目录围绕 Neural Solver Selection 论文展开。
- 关键内容：
  - `neural-solver-selection/`
    - NSS 原始或接近原始的主代码目录。
  - `neural-solver-selection_schemeA/`
  - `neural-solver-selection_schemeB/`
    - 多任务或变体方案目录。
  - `solver_selection.pdf`
    - NSS 论文 PDF。
  - `论文原文.md`
    - 论文原文整理稿。
  - `NSS论文实现讲解_模型设计与训练流程.md`
    - 对该项目实现的中文讲解。
  - `PAUSE_STATUS_2026-04-02.md`
    - 某次多任务训练 / 评测暂停状态记录。
  - `multitask_train_master.log`
    - 训练日志。
  - `launch_multitask_schemes_ab.sh`
    - 方案 A/B 启动脚本。

### 3.4 `9Reld/`：ReLD 相关目录

- 与 ReLD 相关的实验 / 平台内容。
- 当前可见关键子目录：
  - `compare/`
  - `final/`
- `9Reld/final/AGENTS.md` 已存在，说明这个子树内部有自己的更具体 agent 记忆。

### 3.5 `methods源代码和论文/`：第三方方法源码与论文资料库

- 用于保存不同 NCO / routing 方法的源码和论文资料。
- 当前关键子目录：
  - `0论文内容/`
  - `DACT/`
  - `DIFUSCO/`
  - `GLOP/`
  - `H-TSP/`
  - `LEHD/`
- 根下还有：
  - `初始化迭代组合分析.md`

### 3.6 `a3_Revised_URS_FinalRefine_UnifiedEnv/`：URS / UnifiedEnv 旧项目

- 一个相对独立的旧研究项目目录。
- 关键代码文件：
  - `train.py`
  - `test.py`
  - `Trainer.py`
  - `Tester.py`
  - `UNIEnv.py`
  - `Model.py`
  - `Model_LIB.py`
  - `ProblemDef.py`
- 还带有结果目录：
  - `result_models_icam_11Tasks/`
  - `result_test_urs/`
  - `result_train_urs_11tasks/`
  - `result_train_urs_12tasks/`

### 3.7 `已有文献/`：文献笔记

- 当前可见内容：
  - `moe.md`
  - `nss.md`
  - `urs.md`

### 3.8 `plan/`：使用说明与流程文档

- 当前可见内容：
  - `skills-codex-usage-guide.md`

### 3.9 `temp/`：临时目录

- 临时文件和中间产物存放地，默认不应视为稳定主线代码。

### 3.10 `Auto-claude-code-research-in-sleep/`：独立 git 子项目

- 该目录自带 `.git`。
- 应把它视为独立项目，不要把它和 `shiys` 根仓库的历史混在一起处理。

### 3.11 `codex-oauth-automation-extension-4.0.0/`：独立 git 子项目

- 浏览器扩展项目，自带 `.git`。
- 关键文件：
  - `manifest.json`
  - `background.js`
  - `content/`
  - `sidepanel/`
  - `icons/`
  - `data/`

## 4. 当前毕业论文主线记忆

如果用户没有特别说明，默认“当前研究 / 当前论文 / 现在主要在做什么”指下面这条主线：

- 目录：
  - `old_try/2cmab2`
  - `old_try/sustechthesis-1.3.9`
- 问题：
  - 在 `TSP + CVRP` 上做 `initialization-only` 的实例级方法选择。
- 建模：
  - 单步 `contextual bandit`
- 主方法：
  - `Neural-LinUCB`
- 研究动机：
  - 不同神经求解器在不同实例上存在互补性。
  - 目标不是统一最强 solver，而是按实例选方法。

### 4.1 当前不应误判成主线的目录

- `old_try/1two_gate`
- `old_try/1step`
- `old_try/1online`
- `old_try/2l2r`

它们很重要，但更多承担：

- 失败经验
- 对照实验
- 方法演化证据
- discussion / limitation / 答辩解释材料

### 4.2 当前可信结论摘要

- 合成混合数据集上，`Neural-LinUCB` 优于 `SingleBest`
- TSP benchmark 上有一定泛化
- CVRP benchmark 上泛化不足
- `top1` 与 `mean cost / mean length` 经常失配
- reward 定义与最终指标之间存在目标失配，这是主文里的重要 limitation

## 5. 运行与操作边界

### 5.1 git 边界

- `shiys/` 根目录现在是一个总控 git 仓库。
- 以下目录本身又是独立 git 仓库：
  - `Auto-claude-code-research-in-sleep/`
  - `codex-oauth-automation-extension-4.0.0/`
  - `old_try/9Auto-claude-code-research-in-sleep/`
  - `old_try/Claudix/`
- 处理这些目录时：
  - 不要在根仓库里粗暴覆盖它们的 `.git` 语义
  - 若需要管理其提交历史，应进入子仓库单独操作

### 5.2 默认运行习惯

- 训练 / 评测脚本优先在 `tmux` 中启动。
- 所有 Python 命令默认先激活 `easynco_zhoucl`。
- 面向当前论文主线时，优先从 `old_try/2cmab2/run_offline.py` 和 `evaluate_nss_benchmarks.py` 入手。
- 面向 thesis / slides 时，优先改 `old_try/sustechthesis-1.3.9/`。
- 面向平台评测 / solver 集成时，优先改 `EasyNCO/`。

## 6. 新助手接手时的推荐阅读顺序

### 6.1 如果目标是当前毕业论文

1. `FULL_PROJECT_CONTEXT_FOR_NEXT_MODEL_2026-04-13.md`
2. `old_try/sustechthesis-1.3.9/sections/thesis/introduction.tex`
3. `old_try/sustechthesis-1.3.9/sections/thesis/method.tex`
4. `old_try/sustechthesis-1.3.9/sections/thesis/results.tex`
5. `old_try/2cmab2/build_dataset.py`
6. `old_try/2cmab2/encoders.py`
7. `old_try/2cmab2/neural_linucb.py`
8. `old_try/analysis-output/2026-04-06-full-project-results/analysis-report.md`

### 6.2 如果目标是平台 / 统一评测

1. `EasyNCO/README.md`
2. `EasyNCO/configs.yaml`
3. `EasyNCO/settings/`
4. `EasyNCO/train.py`
5. `EasyNCO/eval.py`
6. `EasyNCO/run_nss_eval_suite.py`

### 6.3 如果目标是 NSS 原论文 / 原实现

1. `9nss论文/论文原文.md`
2. `9nss论文/NSS论文实现讲解_模型设计与训练流程.md`
3. `9nss论文/neural-solver-selection/`

## 7. 协作原则

- 不要把整个 `shiys/` 误当成单一、干净、线性的代码仓库。
- 先判断用户现在说的是哪个子项目，再动手。
- 默认保留历史目录，不因“看起来旧”就删除或覆盖。
- 当前主线是 `2cmab2 + thesis/slides`，但用户也常会同时操作 `EasyNCO`、`9nss论文`、旧 RL 原型与独立工具项目。
- 如果用户要求“整理全局”“管理整个工作区”“做根级文档 / git / 工具化”，以本 `AGENTS.md` 的顶层地图为准。


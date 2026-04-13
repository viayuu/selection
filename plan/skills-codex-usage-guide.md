# ARIS `skills-codex` 使用流程示例

本文档面向已经把 `skills-codex` 安装到 Codex 的场景，整理这套 skills 的核心用法、工作流组织方式，以及一个可以直接参考的完整示例。

本文主要基于：

- `skills-codex/README_CN.md`
- `skills-codex` 中各个 `SKILL.md`

## 1. 这套 skills 是什么

`skills-codex` 是 ARIS 的 **Codex 版本技能包**。它保留了主线科研工作流的核心结构，但把原版里依赖 Claude Code + Codex MCP reviewer 的部分，改成了更适合 Codex 的执行方式。

核心目标仍然是覆盖完整科研链路：

1. 找 idea
2. 精炼方案
3. 写实验代码并部署
4. 自动 review 和补实验
5. 写论文

## 2. 最重要的主入口

这套 skills 最重要的 4 个大入口是：

- `idea-discovery`
- `experiment-bridge`
- `auto-review-loop`
- `paper-writing`

如果你想全自动一条龙，还有：

- `research-pipeline`

如果你想细粒度拆开控制，则常用小 skill 有：

- `research-lit`
- `idea-creator`
- `novelty-check`
- `research-review`
- `research-refine`
- `experiment-plan`
- `run-experiment`
- `monitor-experiment`
- `paper-plan`
- `paper-figure`
- `paper-write`
- `paper-compile`
- `auto-paper-improvement-loop`

## 3. 推荐理解方式

可以把使用方式理解成三层：

### 第一层：小 skill

适合你想精细控制每一步。

例如：

- `research-lit`：先做文献调研
- `idea-creator`：只做 brainstorm
- `novelty-check`：只查新
- `experiment-plan`：只产出实验路线图

### 第二层：中型工作流

适合你想一键跑完整个阶段。

例如：

- `idea-discovery`：完整工作流 1
- `experiment-bridge`：完整工作流 1.5
- `auto-review-loop`：完整工作流 2
- `paper-writing`：完整工作流 3

### 第三层：总入口

- `research-pipeline`

适合从“研究方向”直接跑到“投稿准备完成”的大流程。

## 4. 一套完整示例：以“数字图像的识别与分类”为例

下面给出一套完整但易理解的示例，主题统一设为：

**数字图像的识别与分类**

为了让任务更像研究问题，而不是普通课程作业，建议在实际使用时把它收缩为：

- 数字图像分类中的鲁棒性
- 手写数字识别中的数据增强
- 数字图像分类中的轻量模型
- 数字分类在噪声、遮挡、旋转扰动下的稳定性

### 4.1 工作流 1：Idea 发现与方案精炼

#### 方式 A：拆开一步一步跑

先调研：

```text
请使用 research-lit skill，调研“数字图像的识别与分类，重点关注鲁棒性、轻量模型与数据增强”，sources: web。
```

再生成 idea：

```text
请使用 idea-creator skill，围绕“数字图像的识别与分类”生成一批可发表的研究 idea。要求：实验规模尽量小，可快速验证。
```

然后你从结果里挑出 top 2-3 个 idea，再继续：

```text
请使用 novelty-check skill，检查“遮挡增强对数字图像识别鲁棒性的影响”是否已经被做过。
```

```text
请使用 research-review skill，从审稿人视角批判“遮挡增强对数字图像识别鲁棒性的影响”这个 idea。
```

```text
请使用 research-refine skill，把“遮挡增强对数字图像识别鲁棒性的影响”这个 idea 精炼成清晰的问题定义、方法描述和实验目标。
```

最后生成实验计划：

```text
请使用 experiment-plan skill，为当前精炼后的研究方案生成 claim-driven 实验路线图。
```

#### 方式 B：直接跑工作流 1

```text
请使用 idea-discovery skill，研究方向是“数字图像的识别与分类，重点关注鲁棒性、轻量模型和数据增强”，AUTO_PROCEED: false。
```

这一步通常会产出：

- `IDEA_REPORT.md`
- `refine-logs/FINAL_PROPOSAL.md`
- `refine-logs/EXPERIMENT_PLAN.md`

### 4.2 工作流 1.5：实验桥接

当你已经有 `EXPERIMENT_PLAN.md` 后，进入实验桥接。

#### 一键跑工作流 1.5

```text
请使用 experiment-bridge skill，读取 refine-logs/EXPERIMENT_PLAN.md，把实验计划变成可运行的实验代码。CODE_REVIEW: true，AUTO_DEPLOY: false，SANITY_FIRST: true。
```

它会尝试完成：

1. 解析实验计划
2. 复用现有代码结构
3. 写训练/评测脚本
4. 加配置、日志、随机种子
5. 做代码审查
6. 跑 sanity check

如果你想手动控制正式实验部署，可以再单独调用：

```text
请使用 run-experiment skill，运行当前已生成的数字图像分类实验。
```

运行后可继续：

```text
请使用 monitor-experiment skill，监控当前实验进度并总结初始结果。
```

### 4.3 工作流 2：自动科研循环

当你已经有第一轮实验结果后，可以启动自动科研循环。

```text
请使用 auto-review-loop skill，主题是“数字图像识别中的鲁棒性与分类性能研究”，HUMAN_CHECKPOINT: true。
```

它会反复执行：

1. reviewer 打分
2. 找弱点
3. 修改代码、补实验、改叙事
4. 再跑实验
5. 再 review

如果你想更严格：

```text
请使用 auto-review-loop skill，主题是“数字图像识别中的鲁棒性与分类性能研究”，difficulty: hard。
```

如果你想极限压测：

```text
请使用 auto-review-loop skill，主题是“数字图像识别中的鲁棒性与分类性能研究”，difficulty: nightmare。
```

### 4.4 工作流 3：论文写作流水线

当你已经有较稳定的实验结果和叙事后，先整理 `NARRATIVE_REPORT.md`：

```text
请根据当前项目中的实验结果、AUTO_REVIEW.md、关键发现和对比实验，整理一份 NARRATIVE_REPORT.md。
```

然后进入论文流水线。

#### 方式 A：拆开跑

```text
请使用 paper-plan skill，根据 NARRATIVE_REPORT.md 生成论文大纲和 claims-evidence 矩阵。
```

```text
请使用 paper-figure skill，根据现有结果生成论文图表。
```

```text
请使用 paper-write skill，根据 NARRATIVE_REPORT.md 和论文大纲生成 LaTeX 正文。
```

```text
请使用 paper-compile skill，编译论文并修复常见 LaTeX 错误。
```

```text
请使用 auto-paper-improvement-loop skill，对论文做内容与格式的自动改进。
```

#### 方式 B：一键跑工作流 3

```text
请使用 paper-writing skill，输入文件是 NARRATIVE_REPORT.md。
```

## 5. 一条推荐的完整使用路径

如果你不想一上来就用 `research-pipeline`，最推荐的主线顺序是：

1. `idea-discovery`
2. `experiment-bridge`
3. `auto-review-loop`
4. `paper-writing`

用自然语言写成就是：

```text
请使用 idea-discovery skill，研究方向是“数字图像的识别与分类，重点关注鲁棒性、轻量模型和数据增强”，AUTO_PROCEED: false。
```

等工作流 1 完成后：

```text
请使用 experiment-bridge skill，读取 refine-logs/EXPERIMENT_PLAN.md，把方案落地成实验代码并先做 sanity check。
```

拿到初始实验结果后：

```text
请使用 auto-review-loop skill，主题是“数字图像识别中的鲁棒性与分类性能研究”，HUMAN_CHECKPOINT: true。
```

最后：

```text
请使用 paper-writing skill，输入文件是 NARRATIVE_REPORT.md。
```

## 6. 大 skill 和小 skill 的关系

大 skill 不是完全新的东西，而是把小 skill 串起来。

### 总表

| 大 skill | 内部主要包含的小 skill |
|----------|------------------------|
| `idea-discovery` | `research-lit` + `idea-creator` + `novelty-check` + `research-review` + `research-refine-pipeline` |
| `experiment-bridge` | 读取 `EXPERIMENT_PLAN.md` + 写代码 + `run-experiment` + `monitor-experiment` + 可选 `training-check` + 可选 `ablation-planner` |
| `auto-review-loop` | reviewer 循环 + 可选查新/复核 + `run-experiment` + `monitor-experiment` + 结果读取与修复迭代 |
| `paper-writing` | `paper-plan` + `paper-figure` + `paper-write` + `paper-compile` + `auto-paper-improvement-loop` |
| `research-pipeline` | `idea-discovery` + implement + `run-experiment` + `auto-review-loop` |

下面把每个大 skill 展开说明。

### `idea-discovery`

大致对应：

- `research-lit`
- `idea-creator`
- `novelty-check`
- `research-review`
- `research-refine-pipeline`

这条映射在 skill 本体里写得最明确：

- `idea-discovery` 开头直接声明了：
  - `research-lit -> idea-creator -> novelty-check -> research-review -> research-refine-pipeline`

### `experiment-bridge`

这一个和 `idea-discovery`、`paper-writing` 不完全一样，因为它不是单纯“调用 5 个已有 skill”，而是：

1. 自己先读取并解析：
   - `refine-logs/EXPERIMENT_PLAN.md`
   - `refine-logs/EXPERIMENT_TRACKER.md`
   - `refine-logs/FINAL_PROPOSAL.md`
2. 自己负责：
   - 写实验代码
   - 做自检
   - 做 sanity 阶段实现
3. 然后显式调用：
   - `run-experiment`
   - `monitor-experiment`
4. 在结果阶段还会可选调用：
   - `training-check`
   - `ablation-planner`

所以更准确的理解是：

- `experiment-bridge`
  - 不是一个纯“编排器”
  - 而是“自己做大量桥接工作 + 串起几个下游实验 skill”

### `auto-review-loop`

这个大 skill 也不是简单地按 README 风格列几个固定子 skill，而是一个循环控制器。它内部逻辑大致包含：

1. reviewer 审查
2. 解析 review 结果
3. 主流程实现修复
4. 如有需要，重新跑实验：
   - `run-experiment`
   - `monitor-experiment`
5. 继续下一轮 review

从 README 的工作流描述和 skill 内容来看，它功能上通常会联动这些能力：

- `research-review`
- `novelty-check`
- `run-experiment`
- `monitor-experiment`
- `analyze-results`

但要注意：

- `auto-review-loop` 本体不是简单“顺序调用上述 5 个 skill”
- 它更像一个总控循环
- 上面这些更接近“它在循环中会用到的能力模块”

### `paper-writing`

大致对应：

- `paper-plan`
- `paper-figure`
- `paper-write`
- `paper-compile`
- `auto-paper-improvement-loop`

这个映射在 skill 本体里也写得非常明确：

- `paper-plan -> paper-figure -> paper-write -> paper-compile -> auto-paper-improvement-loop`

### `research-pipeline`

这是总入口，不是简单把所有小 skill 全都平铺，而是把几个大阶段再串起来：

1. `idea-discovery`
2. implement
3. `run-experiment`
4. `auto-review-loop`

其中 `idea-discovery` 自己又会展开成：

- `research-lit`
- `idea-creator`
- `novelty-check`
- `research-review`
- `research-refine-pipeline`

所以如果你把它完全展开来看，`research-pipeline` 是最外层总控。

## 7. 行内覆盖怎么用

这套 skills 常用的行内覆盖格式是：

```text
请使用 <skill-name> skill，主题是“...” ，参数1: 值1，参数2: 值2。
```

例如：

```text
请使用 idea-discovery skill，研究方向是“数字图像的识别与分类”，AUTO_PROCEED: false，ARXIV_DOWNLOAD: true。
```

```text
请使用 auto-review-loop skill，主题是“数字图像识别中的鲁棒性与分类性能研究”，compact: true，human checkpoint: true，difficulty: hard。
```

```text
请使用 research-lit skill，调研“数字图像的识别与分类”，sources: web，arxiv download: true。
```

## 8. 推荐你第一次怎么用

第一次使用时，不建议一上来就直接全自动。

推荐顺序：

1. 先用 `idea-discovery`
2. 再用 `experiment-bridge`
3. 有初始结果后再用 `auto-review-loop`
4. 最后再用 `paper-writing`

原因：

- 更容易看懂每个阶段到底产出了什么
- 更容易在中间人工纠偏
- 更适合第一次熟悉这套 skills

## 9. 最后一句总结

如果你把 `skills-codex` 当成一个科研操作系统来理解，那么最值得记住的主路径其实只有这一条：

```text
idea-discovery -> experiment-bridge -> auto-review-loop -> paper-writing
```

拆开时用小 skill，想快时用大 skill，一切都围绕这条主线展开。

如果进一步展开成“每个大 skill 内的小 skill”，最常用的完整脑图就是：

```text
research-pipeline
  ├─ idea-discovery
  │    ├─ research-lit
  │    ├─ idea-creator
  │    ├─ novelty-check
  │    ├─ research-review
  │    └─ research-refine-pipeline
  ├─ implement
  ├─ run-experiment
  └─ auto-review-loop
       ├─ research-review / reviewer loop
       ├─ novelty-check（按需）
       ├─ run-experiment
       ├─ monitor-experiment
       └─ analyze-results（按需）

paper-writing
  ├─ paper-plan
  ├─ paper-figure
  ├─ paper-write
  ├─ paper-compile
  └─ auto-paper-improvement-loop
```

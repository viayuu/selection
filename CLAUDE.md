## GPU 环境

- 这台机器有直接 GPU 访问（不需要 SSH）
- GPU：2x RTX 3090 24GB
- 实验环境：`easynco_zhoucl`
- 激活前任何 Python 命令：`source ~/.bashrc && conda activate easynco_zhoucl`
- 代码目录：`/public/home/zhoucl/shiys`

## 当前目录结构说明
你不用关注的文件/文件夹：old_try,dataset-plan-logs-v3，EasyNCO，AGENTS.md，DATASET_LABEL_PLAN_REPORT_V3.md，skills-lock.json，easynco_v3_bridge

参考文献文件夹：是我自己想出来的，我的项目可以参考的实现思路
观测指标文件夹：在监控实验时，明确要监控的指标和需要绘制的图
literature文件夹，之前的ai帮我找出的参考文献，你可以参考，可以在这个上面扩展
neural-solver-selection_unified_cost：我当前的主实现。是在nss代码的基础上改的
nss代码文件夹：neural solver selection论文对应的代码，也是我的主要参考代码
reference_repos：之前的ai帮我找出的参考论文对应的代码（即literature文件夹的论文对应的代码）
refine-logs：之前的ai在找idea的时候的产物文件
URS代码：URS论文的代码
IDEA_REPORT.md：之前的ai帮我找出的idea
data文件夹：训练数据的来源




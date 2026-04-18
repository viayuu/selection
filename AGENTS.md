- wandb: true
- wandb_project: selector
- wandb_entity: jkds

GPU 环境
- 这台机器有直接 GPU 访问（不需要 SSH）
- GPU：2x RTX 3090 24GB
- 实验环境：`easynco_zhoucl`
- 任何命令都在 conda activate easynco_zhoucl 环境中执行
- 激活前任何 Python 命令：`conda activate easynco_zhoucl`
- 代码目录：`/public/home/zhoucl/shiys/0selection`

当前目录（0selection文件夹）结构说明

你永远不用看、不要关注的文件/文件夹：tools文件夹

文件夹的简单介绍
literature文件夹：是我自己想出来的，我的项目可以参考的论文和对应的代码
观测指标文件夹：在监控实验时，明确要监控的指标和需要绘制的图
data文件夹：训练数据的来源


你更改AGENTS.md的时候，不要更改以上的内容。
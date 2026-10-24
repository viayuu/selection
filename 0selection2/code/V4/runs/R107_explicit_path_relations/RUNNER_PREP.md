# R107 runner preparation

`code.V4.r107_experiment`以私有拷贝复用已跑通的R84逻辑batch训练引擎，旧版本模块不被重绑定或修改。

入口：`train`、`test`、`timing`、`describe`、`status`；共同参数为`--root`、`--arm`、`--data-root`、`--prepared-root`、`--cache-root`、`--protocol-lock`、`--device`、`--gpu-type`、`--large-micro-cap 256`、`--eval-batch-size 128`、`--wandb-mode offline`。其它问题microcap1024，逻辑batch1024；最大规模TRAIN输入双Dropout反传的显存记录在`preparation/kernel_witness/B_larger256_1024/complete.json`。

训练保留原row loss按真实micro行数归一化、同一逻辑batch的AMP失败重放、成功更新轴LR和完整任务遍历。独立checkpoint factory恢复新增关系状态；TEST必须绑定唯一VAL checkpoint。没有当前标签前向、底层solver执行、路线辅助监督或新特征缓存。

W&B沿用已记录无凭证情况下的显式offline模式，不声称在线上传。完整版本交付后集中GitHub发布。模型及报告完成并通过必要CPU测试、显存见证前，不生成正式ready或声称已开始训练。

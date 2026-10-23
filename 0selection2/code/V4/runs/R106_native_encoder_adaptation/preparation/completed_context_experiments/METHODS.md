# CPU复现

所有输出仅写本目录，不训练、不推理、不运行solver，不读weights或prediction arrays。只复用11个已锁定版本的小型CSV/JSON；既有RESULTS说明提供机制与边界。无外部review，无新gate，无新增hash计算。

```bash
source /public/home/shiys/miniconda3/etc/profile.d/conda.sh && conda activate easynco && python code/V4/runs/R106_native_encoder_adaptation/preparation/completed_context_experiments/build_report.py
```

字体已先用 `fc-list :lang=zh family file` 确认。最终采用已安装的 `/usr/share/fonts/google-droid/DroidSansFallback.ttf`，避免CFF字体按TrueType嵌入的兼容警告；PDF内嵌中文字体，不依赖外置图片。PDF固定3页，与RESULTS.md共用表数据。缺失字段统一NA。

## 计算口径

- 每版本使用full_model_statistics/protocol.json的唯一VAL锁定checkpoint。TRAIN只有在保存CSV的checkpoint标识等于锁定标识，且保存路径是selected_best或对应selected_update时才纳入。不读实际checkpoint。
- Top1用native/FP64 ind；不采用top1_fp32_argmin_pct。ALL沿用18题等权。每题4,500个VAL、4,500个TEST，TRAIN每题81,000。
- `metrics_by_problem.csv` 保留TRAIN/VAL/TEST全部18题与ALL的Top1、mean_cost、SBS、Oracle、vs_SBS、vs_Oracle、actual regret，并增加相对R89B的描述性差值。
- SBS为冻结TRAIN拟合；Oracle为逐实例合法候选最低成本。actual regret为实例成本比oracle减1的题内均值，再等题平均；vs_Oracle为各题均值比减1，二者不混用。跨题raw mean_cost及差值仅描述，不用于替代regret判断。
- 固定36组是R84A锁定VAL中选择占比不超过0.1%且有native winner的组；原保存target_category分为zero29和near_zero7。TEST winner分别2,731和1,867，总数4,598，不重新选组。
- `recovery_summary.csv` 由现有逐组小CSV求和。recall=correct/winner；precision=correct/selected；false=selected-correct。每个model/split/subset各一行，零分母为NA。
- `target_groups_test.csv` 是36组TEST的保存计数与成本项，保留来源和原参考。strict_zero是R84A VAL定义，不意味着当前模型或TEST的选择为零。29组中27组TEST有winner，另2组TEST无winner。
- 第3页“最大仍未恢复组”与 `unrecovered_focus.csv` 只从本目录 `target_groups_test.csv` 取R102A、TEST、winner_count>0且correct_count=0的组，按winner_count降序取前5，同数按问题/方法排序。保留选择次数、正确/误选及旧类别；zero仍为R84A-VAL旧清单，不将当前预测重新叫作strict_zero。LEHD选择1次但正确0/误选1，不算解决漏选。来源同时记录在inputs_provenance.json，其余表与数字不变，不重读预测数组。
- `recovery_by_reference.csv` 显式保留每个参考，net_rescues=rescued-lost。全18净mean_cost差直接复用同参考paired_changes的ALL字段，不把目标子集成本、原始误选成本总额或regret贡献当成该字段。未提供的字段为NA。
- `target_regret_delta_pp` 是固定目标winner子集对full18 actual regret的贡献差，不是目标winner的条件平均regret。既有new_false_pick_cost_delta_contribution等字段原样保留，缺失不补猜。
- `matched_ab.csv` 直接保留已有B-vs-A ALL行；R104标记非严格matched。没有重算逐实例改变或显著性，也不跨参考拼接rescued/lost。
- R101保留hint阶段的instance_encoder，Stage II重置优化器/scaler/RNG/采样流；部署不使用teacher。这里的reset不是把已迁移encoder重新随机化。
- `nss_comparison.csv` 原样复用同版本四数据集的NSS与当前模型Top1/mean_cost；同一模型不跨版本拼接。NSS为历史R79权重，不是同预算NEW90k重训对照。

## 必要核对

只核对11个模型、18题分母与合法池、native winner口径、selected TRAIN来源，以及固定36/29组求和；不扩展成审计。PDF检查页数、可抽取文字和渲染可读性。原Rxx源、报告、协议、状态均不写入。

R99/R102使用保存cached-feature TEST，未作bitwise online效能认证。原timing失败及R102原metadata/额外strict-timing失败保留。单seed历史比较不作因果/显著性或上限结论，TEST已参与研发，不宣称盲测。本报告不提出新实验或更改协议，R105/R106结果均排除。

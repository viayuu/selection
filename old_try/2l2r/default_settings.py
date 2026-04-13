from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_ROOT = Path(__file__).resolve().parent


class RunL2RDefaults:
    """
    `2l2r/run_l2r.py` 的默认参数集中定义。

    你可以把这里理解成：
    - 当前 supervised learning-to-rank 实验的“默认配方”
    - 命令行只是在这些默认值上做覆盖
    """

    # ========================================================
    # 一、实验对象与数据路径
    # ========================================================
    RESULTS_DIR = str(REPO_ROOT / "EasyNCO" / "results" / "test")
    TSP_DATASET_PATH = ""
    CVRP_DATASET_PATH = ""
    REWARD_MODE = "linear_zero_one"

    # ========================================================
    # 二、数据切分与基础实验控制
    # ========================================================
    TRAIN_RATIO = 0.7
    VAL_RATIO = 0.15
    SEED = 0
    MAX_SAMPLES_PER_PROBLEM = 0

    # ========================================================
    # 三、模型结构
    # ========================================================
    HIDDEN_DIM = 64
    ARM_EMBED_DIM = 16
    FREEZE_ENCODER = False

    # ========================================================
    # 四、训练超参数
    # ========================================================
    EPOCHS = 20
    BATCH_SIZE = 32
    EVAL_BATCH_SIZE = 64
    LR = 1e-4
    WEIGHT_DECAY = 1e-4
    GRAD_CLIP_NORM = 1.0

    # 当前支持：
    # - "pairwise_logistic"
    # - "listwise_ce"
    # - "best_vs_all_regret"
    # - "two_stage_default_gate"
    # - "one_vs_default_two_stage"
    # - "default_gain_regression"
    # - "one_vs_default_gain"
    # - "one_vs_default_beat"
    # - "one_vs_default_margin"
    # - "one_vs_default_hard_rank"
    # - "one_vs_default_switch_rank"
    # - "one_vs_default_coupled_switch_rank"
    # - "one_vs_default_switch_advantage"
    #
    # 当前推荐的主线是 two_stage_default_gate：
    # 先默认 stay 在 single_best_per_problem，
    # 再学习什么时候值得 switch，以及 switch 到谁。
    LOSS_NAME = "two_stage_default_gate"

    # pairwise logistic 中，是否按 reward gap 给 pair 加权。
    PAIRWISE_GAP_WEIGHT = True

    # listwise 目标分布的锐度。
    # 越大，目标分布越集中在高 reward 的 arm 上。
    LISTWISE_TARGET_SCALE = 8.0

    # best-vs-all regret 训练时：
    # 是否对“固定 single_best_per_problem 做不好的实例”加更高权重。
    # 0 表示关闭；越大表示越重视 hard instances。
    HARD_INSTANCE_WEIGHT_ALPHA = 2.0

    # hard-instance 权重上限，避免极少数样本权重过大。
    # <=0 表示不截断。
    HARD_INSTANCE_WEIGHT_CAP = 5.0

    # 是否在监督排序分数上加入 problem-specific 的 arm 先验。
    # 当前支持：
    # - "none"
    # - "problem_arm_avg_neg_regret"
    SCORE_PRIOR_MODE = "none"

    # ========================================================
    # 四点五、two-stage default gate 专用参数
    # ========================================================
    # switch 标签阈值 = switch_label_scale * problem_mean_hardness
    #
    # 这里的 problem_mean_hardness 指：
    # 训练集上 single_best_per_problem 相对 oracle 的平均 normalized regret。
    #
    # 例如：
    # - TSP 平均 harder 程度较小
    # - CVRP 平均 harder 程度较大
    #
    # 用 problem-specific 的尺度，而不是全局一个绝对阈值，
    # 更适合当前 TSP / CVRP 混合训练。
    SWITCH_LABEL_SCALE = 1.0

    # 对 switch 正样本额外加权。
    #
    # 权重公式大致是：
    #   1 + alpha * max(0, improvement / threshold - 1)
    #
    # 含义：
    # - 只有默认策略吃亏明显的样本，才被放大
    # - improvement 刚刚超过阈值的样本，权重接近 1
    SWITCH_POSITIVE_WEIGHT_ALPHA = 1.0

    # 上面这个正样本权重的上限，避免个别特别难的实例主导训练。
    SWITCH_POSITIVE_WEIGHT_CAP = 5.0

    # ========================================================
    # 四点六、default gain regression 专用参数
    # ========================================================
    # 把 relative gain 除以 problem_mean_hardness 后再做回归。
    # 这样 TSP / CVRP 的目标量级都更接近 1。
    GAIN_NORMALIZE_BY_PROBLEM_HARDNESS = True

    # 正收益样本的额外权重：
    # weight = 1 + alpha * target_gain
    GAIN_POSITIVE_WEIGHT_ALPHA = 1.0

    # 对 target_gain == 0 的项降权，避免被大量“不如默认 arm”的动作淹没。
    GAIN_ZERO_WEIGHT = 0.2

    # 回归损失类型。
    # - "smooth_l1"
    # - "mse"
    GAIN_LOSS_TYPE = "smooth_l1"

    # 推理时：
    # - 如果 max_pred_gain > threshold，就切换
    # - 否则 stay 在默认 arm
    #
    # 这个阈值会在 val 上做网格搜索。
    GAIN_THRESHOLD_GRID = "0.0,0.1,0.2,0.3,0.5,0.8,1.0,1.5,2.0"

    # ========================================================
    # 四点七、one-vs-default 候选分解专用参数
    # ========================================================
    # 训练集上，如果某个 arm 在该问题上“优于默认 arm”的样本比例
    # 小于这个阈值，就不把它纳入候选集合。
    ONE_VS_DEFAULT_MIN_POSITIVE_RATE = 0.02

    # 除了比例之外，再加一个最小正样本数约束，避免极小频次 arm 混入。
    ONE_VS_DEFAULT_MIN_POSITIVE_COUNT = 20

    # 每个问题最多保留多少个候选替代 arm。
    ONE_VS_DEFAULT_MAX_CANDIDATES = 6

    # beat-default 二分类时使用的概率阈值搜索网格。
    ONE_VS_DEFAULT_BEAT_PROB_GRID = "0.3,0.4,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9"

    # 模型 checkpoint / early stopping 使用什么验证指标：
    # - "overall_mean_cost": 和以前一样，val overall mean_cost 越小越好
    # - "gain_vs_single_best_per_problem": 更关注是否真的超过 per-problem 固定策略
    MODEL_SELECTION_METRIC = "gain_vs_single_best_per_problem"

    # 训练日志间隔：每多少个优化 step 打一次摘要。
    TRAIN_LOG_EVERY = 20

    # 周期性验证 / checkpoint。
    EVAL_EVERY_EPOCHS = 1
    EVAL_EVERY_STEPS = 0
    CHECKPOINT_EVERY_EPOCHS = 1
    CHECKPOINT_EVERY_STEPS = 0

    # 早停。
    # <=0 表示关闭。
    EARLY_STOPPING_PATIENCE = 5

    # ========================================================
    # 五、运行环境与输出控制
    # ========================================================
    DEVICE = ""
    OUTPUT_DIR = ""
    DISABLE_PROGRESS = False
    QUIET = False

    # 是否把评测 trace 实时打印到终端。
    PRINT_STEP_JSON = False
    STEP_LOG_EVERY = 1

    # 评测阶段进度日志间隔。
    EVAL_LOG_EVERY = 500

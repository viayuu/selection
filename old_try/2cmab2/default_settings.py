from __future__ import annotations

from pathlib import Path


# ============================================================
# 2cmab2 默认参数集中定义
# ============================================================
# 你以后如果想改“默认实验配置”，优先改这个文件。
#
# 设计原则：
# 1. run_offline.py 和 analysis.py 里所有对外暴露的可调默认参数，都尽量集中到这里
# 2. 命令行如果显式传参，会覆盖这里的默认值
# 3. 这个文件只负责“默认值”和“参数分组说明”，不负责实际训练逻辑
#
# 这样做的好处是：
# - 以后调参不用到处翻 argparse
# - 更容易形成你自己的“常用默认配置”
# - 改动更集中，也更不容易漏
# ============================================================


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_ROOT = Path(__file__).resolve().parent


class RunOfflineDefaults:
    """
    `2cmab2/run_offline.py` 的默认参数。

    你可以把它理解成：
    - 这里决定“如果命令行什么都不传，程序默认怎么跑”
    - 命令行传参只是对这里的默认值做局部覆盖
    """

    # ========================================================
    # 一、实验对象与数据路径
    # ========================================================
    # 主算法。
    # 可选：
    # - "linucb"
    # - "neural_linucb"
    # - "neural_ucb_diag"
    METHOD = "neural_linucb"

    # EasyNCO 测试结果根目录。
    # 程序会从这个目录下扫描 TSP/CVRP 各方法的 result json。
    RESULTS_DIR = str(REPO_ROOT / "EasyNCO" / "results" / "test")

    # 数据源类型。
    # 可选：
    # - "easynco_results": 从 EasyNCO 的 results/test + .pt 数据集构造 joint dataset
    # - "nss": 直接使用 `9nss论文/neural-solver-selection/datasets` 自带的 train/val/test split
    DATA_SOURCE = "nss"

    # NSS 数据集根目录。
    # 当 DATA_SOURCE="nss" 时使用。
    NSS_DATASET_ROOT = str(REPO_ROOT / "9nss论文" / "neural-solver-selection" / "datasets")

    # 如果留空，程序会尽量从 EasyNCO 的 hydra 配置里自动恢复 TSP 测试集路径。
    # 自动恢复失败时，再手动填这里。
    TSP_DATASET_PATH = ""

    # 同上，这是 CVRP 测试集路径。
    CVRP_DATASET_PATH = ""

    # reward 口径。
    # 可选：
    # - "linear_zero_one": 最好=1，最差=0
    # - "autosaea": 参考 AutoSAEA 的排名 reward 风格
    REWARD_MODE = "linear_zero_one"

    # ========================================================
    # 二、数据切分与基础实验控制
    # ========================================================
    # train 集比例。当前是 instance-level 分层切分。
    TRAIN_RATIO = 0.7

    # val 集比例。test 比例会自动补成剩余部分。
    VAL_RATIO = 0.15

    # 训练 epoch 数。
    # 这里的 epoch 不是神经网络大 epoch 的意思，
    # 而是“把 train split 按 bandit 顺序完整跑几遍”。
    EPOCHS = 1

    # checkpoint 保存间隔。
    # <=0 表示关闭自动 checkpoint。
    # >0 表示每隔多少个 epoch 自动保存一次。
    # 例如：
    # - 1: 每个 epoch 都保存
    # - 5: 每 5 个 epoch 保存一次
    CHECKPOINT_EVERY = 1

    # step 级 checkpoint 保存间隔。
    # <=0 表示关闭“按 step 自动保存”。
    # >0 表示每累计多少个训练 step 保存一次。
    #
    # 这个参数更适合当前 contextual bandit / online 风格训练：
    # - 即使 epochs=1，也能在一个长 epoch 中间定期落盘
    # - 程序中途停止时，已经保存过的 checkpoint 不会丢
    CHECKPOINT_EVERY_STEPS = 1000

    # 随机种子。影响：
    # - 数据切分
    # - 训练顺序打乱
    # - 部分随机初始化
    SEED = 0

    # 最多每个问题保留多少个样本。
    # 0 表示不裁剪，直接使用全部可用样本。
    # 调试时常用 200 / 1000。
    MAX_SAMPLES_PER_PROBLEM = 0

    # 训练过程中，每隔多少个 step 做一次验证集评测。
    # <=0 表示关闭训练中的周期性 val。
    #
    # 适用场景：
    # - 当前 epochs=1 时，一个 epoch 往往就是完整 online/bandit 训练过程
    # - 如果只在末尾才做 val，你中途看不到模型是否在变好
    # - 打开这个参数后，会每隔若干 step 评一次 val，并把结果实时写盘
    EVAL_EVERY_STEPS = 1000

    # 评测阶段的进度日志间隔。
    # 例如 test 有 3000 个实例时：
    # - 500 会打印 500/3000, 1000/3000, ...
    # - 1000 会打印 1000/3000, 2000/3000, 3000/3000
    EVAL_LOG_EVERY = 500

    # ========================================================
    # 三、共享 bandit 的公共超参数
    # ========================================================
    # UCB 探索强度。
    # 越大越偏探索，越小越偏利用。
    ALPHA = 1.0

    # 线性正则项。
    # LinUCB / Neural-LinUCB 的线性头都会用到它。
    REG = 1.0

    # warm start：每个 arm 至少被强制尝试多少次。
    # 这样可以避免某个 arm 在一开始完全没有被拉过。
    INITIAL_PULLS = 2

    # ========================================================
    # 四、Neural-LinUCB / NeuralUCB 相关神经网络超参数
    # ========================================================
    # phi(x,a) 的输出维度，也是最后线性头的维度。
    HIDDEN_DIM = 64

    # NeuralUCB-Diag 中 arm embedding 的维度。
    # 这个 embedding 主要学习“方法之间是否相似”。
    ARM_EMBED_DIM = 16

    # 是否在 NeuralUCB-Diag 里把 arm one-hot 直接拼到网络输入。
    # 打开后，reward_net 会同时看到：
    # - 连续的 arm embedding
    # - 离散的 arm 身份 one-hot
    #
    # 这样可以明显增强不同 arm 之间的可区分性，
    # 避免网络把多个 arm 学成几乎一样的 raw score。
    USE_ARM_ONEHOT = True

    # 是否为 NeuralUCB-Diag 增加显式的 arm 偏置项。
    # 这个偏置项相当于“每个方法自己的全局基准分”，
    # 在网络还没完全学出实例级差异时，能帮助模型更快地区分不同方法。
    USE_ARM_BIAS = True

    # Adam 学习率。
    LR = 1e-3

    # 每累计多少次 bandit 反馈，才触发一次神经网络训练。
    # 对不同方法的含义：
    # - Neural-LinUCB: 表示层更新频率
    # - NeuralUCB-Diag: reward_net 的周期训练间隔
    TRAIN_EVERY = 100

    # 每次触发表示层训练后，在表示层窗口上跑多少轮优化。
    REPRESENTATION_STEPS = 10

    # 表示层训练窗口大小。
    # 只控制表示层训练用的 recent buffer。
    # <=0 表示保留全部历史。
    REPRESENTATION_BUFFER_SIZE = 5000

    # 表示层训练时的 mini-batch 大小。
    # 只影响表示层神经网络训练，不影响 bandit 主循环的一步一更新。
    # <=0 表示每轮直接用整个表示层窗口。
    REPRESENTATION_BATCH_SIZE = 512

    # 表示层训练时，是否对 history 做“均衡采样”。
    # 可选：
    # - "none": 不做均衡，直接随机打乱
    # - "arm": 按 arm 频次均衡
    # - "problem_arm": 按 (problem, arm) 频次均衡
    #
    # 当前 TSP + CVRP 共训默认推荐 "problem_arm"，
    # 这样可以同时抑制“单一 problem 的主导偏置”和“单一 arm 的主导偏置”。
    REPRESENTATION_BALANCE_MODE = "problem_arm"

    # 均衡采样 / loss 重加权的强度参数。
    # 实际使用的是 inverse-frequency 幂次：
    #   weight ~ count^(-power)
    #
    # 常见理解：
    # - 0.0: 不做频次修正
    # - 0.5: 温和均衡
    # - 1.0: 强均衡
    REPRESENTATION_BALANCE_POWER = 0.5

    # 是否在表示层训练时对 batch loss 再做 inverse-frequency 重加权。
    # 打开后，低频 arm / problem-arm 样本即使数量少，
    # 也不会在 loss 里被高频样本完全淹没。
    REPRESENTATION_LOSS_REWEIGHT = True

    # 训练阶段额外增加一个 epsilon 探索地板。
    # 只在 train 选臂时生效，不影响 val/test 的 greedy 或 ucb 评测。
    TRAIN_EPSILON = 0.10

    # Neural-LinUCB 的最后一层线性 UCB 头是否按 problem 分开维护。
    #
    # True：
    # - TSP 一套 A / b / theta
    # - CVRP 一套 A / b / theta
    # - 更符合“不同问题上的方法偏好不同”这一现实
    #
    # False：
    # - 所有问题共享一套线性头
    # - 更接近旧版本实现
    PROBLEM_SPECIFIC_HEADS = True

    # NeuralUCB-Diag 训练 reward_net 时的 mini-batch 大小。
    # <=0 表示训练时直接使用当前全部 history。
    TRAIN_BATCH_SIZE = 256

    # 线性头重建窗口大小。
    # <0 表示先跟随 representation_buffer_size。
    # =0 表示保留全部历史。
    # >0 表示使用显式指定的窗口大小。
    # 注意：如果跟随后得到的 representation_buffer_size <= 0，
    # 那么线性头最终也会保留全部历史。
    # 一般建议它比表示层窗口更大。
    LINEAR_HEAD_BUFFER_SIZE = 10000

    # NeuralUCB-Diag 的不确定性缩放参数。
    # 当前主要给 `neural_ucb_diag` 使用。
    NU = 0.1

    # NeuralUCB-Diag 每次触发训练时，最多做多少个优化 step。
    TRAIN_STEPS_PER_UPDATE = 20

    # 是否启用“按问题的经验坏臂惩罚”。
    # 这个机制主要是为了压住那些在某个问题上已经明显很差、
    # 但仍然会因为 score 非常接近而偶尔被选中的高风险 arm。
    USE_PROBLEM_ARM_SAFETY = True

    # safety 生效前，某个 (problem, arm) 至少要被拉多少次。
    SAFETY_MIN_PULLS = 20

    # 只有当当前 arm 的经验平均 reward
    # 明显落后于该问题上的最好 arm 时，才开始惩罚。
    SAFETY_REWARD_GAP = 0.15

    # 惩罚强度：
    # penalty = scale * max(0, gap - threshold)
    SAFETY_PENALTY_SCALE = 3.0

    # 单个 arm 的最大惩罚上限。
    SAFETY_PENALTY_CAP = 2.0

    # ========================================================
    # 四点五、周期性验证与 best checkpoint 选择
    # ========================================================
    # 是否根据 periodic val 选出 best checkpoint，
    # 并在最终 val/test 前先恢复该 best checkpoint。
    USE_BEST_CHECKPOINT_FOR_FINAL_EVAL = True

    # 用哪个验证指标来挑 best checkpoint。
    # 采用点路径格式，从 periodic val 的 metrics 结构里取值。
    BEST_CHECKPOINT_METRIC = "val_greedy.overall.mean_cost"

    # best checkpoint 指标方向：
    # - "min": 越小越好，例如 mean_cost / mean_regret / mean_rank
    # - "max": 越大越好，例如 mean_reward / top1_accuracy
    BEST_CHECKPOINT_MODE = "min"

    # ========================================================
    # 五、运行环境与输出控制
    # ========================================================
    # 是否冻结底层 NSS 图编码器。
    # True：
    # - 更省显存/时间
    # - 但表示能力可能受限
    FREEZE_ENCODER = False

    # 设备字符串。
    # 留空时程序会自动选择：
    # - 有 CUDA 就用 cuda
    # - 否则用 cpu
    DEVICE = ""

    # 输出目录。
    # 留空时程序会自动生成带时间戳的目录。
    OUTPUT_DIR = ""

    # 是否关闭 tqdm 进度条。
    DISABLE_PROGRESS = False

    # 是否减少阶段日志输出。
    QUIET = False

    # 是否把每一步的详细选择结果实时打印到终端。
    # 打开后适合调试，不适合大规模正式实验。
    PRINT_STEP_JSON = False

    # 如果开启 PRINT_STEP_JSON，控制“每隔多少步打印一次”。
    STEP_LOG_EVERY = 1


class AnalysisDefaults:
    """
    `2cmab2/analysis.py` 的默认参数。
    """

    # 要汇总的实验结果根目录。
    INPUT_ROOT = str(MODULE_ROOT / "outputs")

    # 分析报告输出目录。
    OUTPUT_DIR = str(MODULE_ROOT / "analysis" / "latest")

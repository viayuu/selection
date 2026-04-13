"""
neural_ts.py — Neural Thompson Sampling (对角近似版)
[Zhang et al., ICLR 2021] "Neural Thompson Sampling"

与 NeuralUCB-Diag 的唯一区别:
  - NeuralUCB-Diag: score = mu + sigma           (确定性 UCB bonus)
  - NeuralTS:        score ~ N(mu, sigma²)        (从后验采样, 随机性探索)

其余完全相同: 梯度计算, U 更新, 网络训练.

NeuralTS 的优势: 不需要手调 alpha, 探索力度由后验方差自适应决定.
"""
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

logger = logging.getLogger("neural_ts")


class ContextProj(nn.Module):
    """259 → 128 → 64 → 1 的 reward 预测网络"""
    def __init__(self, input_dim=259, hidden_dim=128, phi_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, phi_dim),
            nn.ReLU(),
            nn.Linear(phi_dim, 1),
        )
    def forward(self, x):
        return self.net(x)


class NeuralTS:
    """
    Neural Thompson Sampling (对角近似版).

    选臂: score ~ N(mu, sigma²), 选 score 最大的 (Thompson Sampling)
    更新 U: U += g_selected * g_selected
    训练: 周期性用历史数据重训网络 (SGD)
    """

    def __init__(self, n_arms: int, encoder: nn.Module,
                 arm_names: list = None,
                 nu: float = 1.0, lamdba: float = 1.0,
                 lr: float = 1e-2,
                 train_every: int = 50, train_steps: int = 100,
                 batch_size: int = 64,
                 max_history: int = 5000,
                 device: str = "cpu"):
        """
        Args:
            nu: 探索缩放系数
            lamdba: 正则化 / U 初始值
            lr: SGD 学习率 (官方用 SGD + weight_decay)
            train_every: 每隔多少步重训网络
            train_steps: 每次重训的最大 SGD 步数
        """
        self.n_arms = n_arms
        self.nu = nu
        self.lamdba = lamdba
        self.device = device
        self.train_every = train_every
        self.train_steps = train_steps
        self.batch_size = batch_size
        self.max_history = max_history
        self.arm_names = arm_names or [str(i) for i in range(n_arms)]

        # ---- 网络: encoder (冻结) + 每臂一个 reward_net ----
        self.encoder = encoder.to(device)
        self.encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad = False

        input_dim = encoder.output_dim + 3  # 256 + 1 scale + 2 onehot = 259
        self.reward_nets = nn.ModuleList([
            ContextProj(input_dim=input_dim) for _ in range(n_arms)
        ]).to(device)

        # 可训练参数: 只有 reward_nets
        self.train_params = list(self.reward_nets.parameters())
        self.total_param = sum(p.numel() for p in self.train_params)
        self.optimizer = optim.SGD(self.train_params, lr=lr, weight_decay=lamdba)

        # 对角近似: 每臂一个 U 向量
        self.U = [torch.full((self.total_param,), lamdba,
                             dtype=torch.float32, device=device)
                  for _ in range(n_arms)]

        # 历史
        self.history = []
        self.t = 0
        self.arm_pull_counts = np.zeros(n_arms, dtype=int)
        self.arm_reward_sums = np.zeros(n_arms)

    def _build_context(self, coords):
        """构建 259 维上下文, 支持 batch"""
        # coords 已是 tensor [B, N, 2/3] on device
        # 但需要 problem_type 信息, 这里在外部处理
        raise NotImplementedError("use _build_context_single instead")

    def _build_context_single(self, coords, problem_type, scale):
        """单实例构建 context [1, 259]"""
        coords_t = torch.tensor(coords, dtype=torch.float32).unsqueeze(0).to(self.device)
        mask_t = torch.zeros(1, coords.shape[0], device=self.device)
        with torch.no_grad():
            graph_emb = self.encoder(coords_t, mask_t, problem_type)  # [1, 256]
        oh = torch.tensor(
            [[1.0, 0.0]] if problem_type.lower() == "tsp" else [[0.0, 1.0]],
            device=self.device)
        sc = torch.tensor([[float(scale)]], device=self.device)
        return torch.cat([graph_emb, sc, oh], dim=-1)  # [1, 259]

    def _get_gradient(self, context, arm_idx):
        """计算 reward_nets[arm_idx](context) 对所有可训练参数的梯度 (无梯度的填零)"""
        self.reward_nets.zero_grad()
        pred = self.reward_nets[arm_idx](context).squeeze()
        pred.backward()
        g = torch.cat([p.grad.flatten() if p.grad is not None
                        else torch.zeros(p.numel(), device=self.device)
                        for p in self.train_params]).detach()
        return float(pred.item()), g

    # ===================== 在线交互 =====================

    def select(self, coords, attn_mask, problem_type, scale, action_mask) -> int:
        """Thompson Sampling: score ~ N(mu, sigma²), 选 score 最大的"""
        self.reward_nets.eval()
        context = self._build_context_single(coords, problem_type, scale)

        arm_scores = {}
        best_score, best_arm, best_g = -np.inf, 0, None

        for k in range(self.n_arms):
            if action_mask[k] < 0.5:
                continue
            mu, g = self._get_gradient(context, k)
            sigma = torch.sqrt(torch.sum(self.lamdba * self.nu * g * g / self.U[k]))
            sigma_val = float(sigma.item())

            # Thompson Sampling: 从后验 N(mu, sigma²) 采样
            sample_r = np.random.normal(mu, sigma_val)

            arm_scores[k] = {"mu": mu, "sigma": sigma_val, "sample": sample_r}
            if sample_r > best_score:
                best_score, best_arm, best_g = sample_r, k, g

        # 更新 U (对齐官方: self.U += g*g)
        if best_g is not None:
            self.U[best_arm] = self.U[best_arm] + best_g * best_g

        # 日志
        log_parts = [f"t={self.t} | {problem_type.upper()} | select arm={best_arm}({self.arm_names[best_arm]})"]
        for k in sorted(arm_scores.keys()):
            s = arm_scores[k]
            marker = " <--" if k == best_arm else ""
            log_parts.append(
                f"    {k:2d}({self.arm_names[k]:16s}): "
                f"mu={s['mu']:+.4f}  sigma={s['sigma']:.4f}  sample={s['sample']:+.4f}{marker}"
            )
        logger.info("\n".join(log_parts))
        return best_arm

    def update(self, arm: int, coords, attn_mask, problem_type, scale, reward: float):
        """记录历史 + 周期性重训网络"""
        self.history.append((coords, attn_mask, problem_type, scale, arm, reward))
        self.arm_pull_counts[arm] += 1
        self.arm_reward_sums[arm] += reward
        self.t += 1

        avg_r = self.arm_reward_sums[arm] / self.arm_pull_counts[arm]
        logger.info(
            f"t={self.t} | update arm={arm}({self.arm_names[arm]}) | "
            f"reward={reward:.4f} | pulls={self.arm_pull_counts[arm]} | "
            f"avg_reward={avg_r:.4f}"
        )

        # 周期性重训
        if self.t % self.train_every == 0 and len(self.history) > 0:
            logger.info(f"t={self.t} | === 触发网络重训 (train_every={self.train_every}) ===")
            self._train_on_history()

    def _train_on_history(self):
        """用历史数据重训网络 (对齐官方 train 方法)"""
        self.reward_nets.train()

        train_history = self.history
        if len(train_history) > self.max_history:
            train_history = train_history[-self.max_history:]

        n = len(train_history)
        indices = np.arange(n)
        steps_done = 0
        total_loss = 0.0

        # 官方做法: 随机抽样训练, 最多 train_steps 步
        np.random.shuffle(indices)
        for idx in indices:
            if steps_done >= self.train_steps:
                break
            coords, attn_mask, ptype, scale, arm, reward = train_history[idx]
            context = self._build_context_single(coords, ptype, scale)
            pred = self.reward_nets[arm](context).squeeze()
            target = torch.tensor(float(reward), dtype=torch.float32, device=self.device)
            loss = (pred - target) ** 2
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            steps_done += 1

        self.reward_nets.eval()
        avg_loss = total_loss / max(steps_done, 1)
        logger.info(f"  [train] steps={steps_done}, avg_loss={avg_loss:.6f}, n_history={n}")

    # ===================== Checkpoint =====================

    def save_checkpoint(self, path: str):
        checkpoint = {
            "reward_nets": self.reward_nets.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "U": [u.cpu() for u in self.U],
            "t": self.t,
            "arm_pull_counts": self.arm_pull_counts,
            "arm_reward_sums": self.arm_reward_sums,
            "config": {
                "n_arms": self.n_arms,
                "nu": self.nu,
                "lamdba": self.lamdba,
                "train_every": self.train_every,
                "train_steps": self.train_steps,
            },
        }
        torch.save(checkpoint, path)
        logger.info(f"  [checkpoint] 已保存到 {path} (t={self.t})")

    def load_checkpoint(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.reward_nets.load_state_dict(checkpoint["reward_nets"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.U = [u.to(self.device) for u in checkpoint["U"]]
        self.t = checkpoint["t"]
        self.arm_pull_counts = checkpoint["arm_pull_counts"]
        self.arm_reward_sums = checkpoint["arm_reward_sums"]
        logger.info(f"  [checkpoint] 已从 {path} 加载 (t={self.t})")

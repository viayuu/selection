"""
neural_linucb.py — Online Neural-LinUCB
[Xu et al., 2020] "Neural Contextual Bandits with UCB-based Exploration"

核心算法 (Algorithm 1):
  1. 每轮收到 context x_t, 用当前网络提取 φ(x_t), LinUCB 选臂
  2. 观察 reward r_t, 更新 LinUCB 统计量 (Z, b)
  3. 每隔 q 步, 用所有历史数据重新训练网络 (梯度下降)
  4. 网络更新后, 用新的 φ 重建所有 arm 的 Z 和 b

对齐原论文设计:
  - reward_head 是持久组件, 不每次重建 (对齐论文中的统一网络 f(x,a;θ))
  - 历史记录只保留最近 H 条, 防止 t 增大后重训变慢
  - 优化器包含所有网络参数, 在上一轮基础上继续优化

性能优化:
  - 训练和 rebuild 时按问题类型分组, 批量过 encoder (真 mini-batch)
  - 避免逐样本 encoder 调用, GPU 利用率大幅提升
"""
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

logger = logging.getLogger("neural_linucb")


class ContextProj(nn.Module):
    """259 → 128 → 64 的投影网络"""
    def __init__(self, input_dim=259, hidden_dim=128, output_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
            nn.GELU(),
        )
    def forward(self, x):
        return self.net(x)


class NeuralLinUCB:
    """
    Online Neural-LinUCB (Xu et al., 2020, Algorithm 1)

    在线交互流程:
      select → 用当前 φ 做 LinUCB 选臂
      update → 存历史, 更新 Z/b, 每 q 步重训网络并重建统计量
    """

    def __init__(self, n_arms: int, encoder: nn.Module,
                 arm_names: list = None,
                 phi_dim: int = 64,
                 alpha: float = 1.0, reg: float = 1.0,
                 lr_encoder: float = 1e-4, lr_head: float = 1e-3,
                 train_every: int = 50, train_steps: int = 10,
                 batch_size: int = 64,
                 max_history: int = 5000,
                 device: str = "cpu"):
        self.n_arms = n_arms
        self.alpha = alpha
        self.phi_dim = phi_dim
        self.reg = reg
        self.device = device
        self.train_every = train_every
        self.train_steps = train_steps
        self.batch_size = batch_size
        self.max_history = max_history
        self.arm_names = arm_names or [str(i) for i in range(n_arms)]

        # ---- 网络组件 (全部持久保留, 对齐论文) ----
        self.encoder = encoder.to(device)
        self.context_proj = ContextProj(
            input_dim=encoder.output_dim + 3,  # 256 + scale(1) + one_hot(2) = 259
            output_dim=phi_dim
        ).to(device)
        self.reward_heads = nn.ModuleList([
            nn.Linear(phi_dim, 1, bias=False) for _ in range(n_arms)
        ]).to(device)

        # ---- 统一优化器 ----
        self.optimizer = optim.Adam([
            {"params": self.encoder.parameters(), "lr": lr_encoder},
            {"params": self.context_proj.parameters(), "lr": lr_head},
            {"params": self.reward_heads.parameters(), "lr": lr_head},
        ])

        # ---- LinUCB 统计量 ----
        self._reset_linucb_stats()

        # ---- 历史记录 ----
        self.history = []
        self.t = 0

        # ---- 统计 ----
        self.arm_pull_counts = np.zeros(n_arms, dtype=int)
        self.arm_reward_sums = np.zeros(n_arms)

    def _reset_linucb_stats(self):
        """重置 LinUCB 统计量 Z, b"""
        self.Z = [np.eye(self.phi_dim) * self.reg for _ in range(self.n_arms)]
        self.b_vec = [np.zeros(self.phi_dim) for _ in range(self.n_arms)]
        self.Z_inv = [np.eye(self.phi_dim) / self.reg for _ in range(self.n_arms)]
        self._z_dirty = [False] * self.n_arms

    # ===================== 批量编码工具 =====================

    def _build_context(self, coords_t, mask_t, problem_type, scale_val):
        """构建 259 维上下文向量 (单一问题类型, 支持 batch)"""
        graph_emb = self.encoder(coords_t, mask_t, problem_type)  # [B, 256]
        B = graph_emb.shape[0]
        is_tsp = torch.tensor(
            [[1.0, 0.0]] if problem_type.lower() == "tsp" else [[0.0, 1.0]]
        ).expand(B, -1).to(self.device)
        scale_t = torch.tensor([[float(scale_val)]]).expand(B, -1).to(self.device)
        return torch.cat([graph_emb, scale_t, is_tsp], dim=-1)  # [B, 259]

    def _encode_batch(self, records, grad=False):
        """
        批量编码一组历史记录, 按问题类型分组过 encoder.
        返回 phi [N, 64] (在 self.device 上).

        records: [(coords, attn_mask, ptype, scale, arm, reward), ...]
        grad: True 时保留梯度 (训练用), False 时无梯度 (rebuild 用)
        """
        n = len(records)
        if n == 0:
            return torch.zeros(0, self.phi_dim, device=self.device)

        # 按问题类型分组
        tsp_idxs, cvrp_idxs = [], []
        for i, (_, _, ptype, _, _, _) in enumerate(records):
            (tsp_idxs if ptype.lower() == "tsp" else cvrp_idxs).append(i)

        # 存放结果, 保持原始顺序
        context_parts = [None] * n

        for ptype, idxs in [("tsp", tsp_idxs), ("cvrp", cvrp_idxs)]:
            if not idxs:
                continue
            # 拼成 batch tensor
            coords_batch = torch.stack([
                torch.tensor(records[i][0], dtype=torch.float32) for i in idxs
            ]).to(self.device)  # [B_type, N, 2/3]
            mask_batch = torch.zeros(
                len(idxs), coords_batch.shape[1], device=self.device
            )  # [B_type, N]
            scale_val = records[idxs[0]][3]  # 同类型 scale 相同 (100)

            # 一次 encoder 前向
            ctx = self._build_context(coords_batch, mask_batch, ptype, scale_val)
            # [B_type, 259]
            for pos, orig_idx in enumerate(idxs):
                context_parts[orig_idx] = ctx[pos]

        # 拼成完整 batch
        context_all = torch.stack(context_parts)  # [N, 259]
        phi = self.context_proj(context_all)       # [N, 64]
        return phi

    def _extract_phi_single(self, coords, problem_type, scale):
        """提取单个实例的 φ(x), 无梯度, 返回 numpy"""
        coords_t = torch.tensor(coords, dtype=torch.float32).unsqueeze(0).to(self.device)
        mask_t = torch.zeros(1, coords.shape[0], device=self.device)
        with torch.no_grad():
            ctx = self._build_context(coords_t, mask_t, problem_type, scale)
            phi = self.context_proj(ctx)
        return phi.squeeze(0).cpu().numpy()

    # ===================== 在线交互接口 =====================

    def select(self, coords, attn_mask, problem_type, scale, action_mask) -> int:
        """UCB 选臂 (单实例, 无梯度)"""
        self.encoder.eval()
        self.context_proj.eval()

        phi_np = self._extract_phi_single(coords, problem_type, scale)

        arm_scores = {}
        best_ucb, best_arm = -np.inf, 0
        for k in range(self.n_arms):
            if action_mask[k] < 0.5:
                continue
            if self._z_dirty[k]:
                self.Z_inv[k] = np.linalg.inv(self.Z[k])
                self._z_dirty[k] = False
            theta = self.Z_inv[k] @ self.b_vec[k]
            mu = float(theta @ phi_np)
            bonus = self.alpha * np.sqrt(float(phi_np @ self.Z_inv[k] @ phi_np))
            ucb = mu + bonus
            arm_scores[k] = {"mu": mu, "bonus": bonus, "ucb": ucb}
            if ucb > best_ucb:
                best_ucb, best_arm = ucb, k

        log_parts = [f"t={self.t} | {problem_type.upper()} | select arm={best_arm}({self.arm_names[best_arm]})"]
        score_strs = []
        for k in sorted(arm_scores.keys()):
            s = arm_scores[k]
            marker = " <--" if k == best_arm else ""
            score_strs.append(
                f"    {k:2d}({self.arm_names[k]:16s}): "
                f"mu={s['mu']:+.4f}  bonus={s['bonus']:.4f}  ucb={s['ucb']:+.4f}{marker}"
            )
        logger.info("\n".join(log_parts + score_strs))
        return best_arm

    def update(self, arm: int, coords, attn_mask, problem_type, scale, reward: float):
        """在线更新: LinUCB 统计量 + 记录历史 + 周期性重训"""
        self.encoder.eval()
        self.context_proj.eval()

        # 1. 用当前 φ 更新 LinUCB
        phi_np = self._extract_phi_single(coords, problem_type, scale)
        self.Z[arm] += np.outer(phi_np, phi_np)
        self.b_vec[arm] += reward * phi_np
        self._z_dirty[arm] = True

        # 2. 记录历史 + 统计
        self.history.append((coords, attn_mask, problem_type, scale, arm, reward))
        self.arm_pull_counts[arm] += 1
        self.arm_reward_sums[arm] += reward
        self.t += 1

        avg_r = self.arm_reward_sums[arm] / self.arm_pull_counts[arm]
        logger.info(
            f"t={self.t} | update arm={arm}({self.arm_names[arm]}) | "
            f"reward={reward:.4f} | pulls={self.arm_pull_counts[arm]} | "
            f"avg_reward={avg_r:.4f} | phi_norm={np.linalg.norm(phi_np):.4f}"
        )

        # 3. 每 q 步重训网络并重建统计量
        if self.t % self.train_every == 0 and len(self.history) > 0:
            logger.info(f"t={self.t} | === 触发网络重训 (train_every={self.train_every}) ===")
            self._log_arm_summary()
            self._fit_representation()
            self._rebuild_stats()
            logger.info(f"t={self.t} | === 网络重训 + 统计量重建完成 ===")

    def _log_arm_summary(self):
        lines = ["  臂统计汇总:"]
        for k in range(self.n_arms):
            if self.arm_pull_counts[k] > 0:
                avg = self.arm_reward_sums[k] / self.arm_pull_counts[k]
                lines.append(
                    f"    {k:2d}({self.arm_names[k]:16s}): "
                    f"pulls={self.arm_pull_counts[k]:5d}  avg_reward={avg:.4f}"
                )
        logger.info("\n".join(lines))

    # ===================== 训练 (真 mini-batch) =====================

    def _fit_representation(self):
        """
        用历史数据训练网络 (论文 Algorithm 1, Step 10-12)
        按问题类型分组批量过 encoder, 真 mini-batch 训练.
        """
        self.encoder.train()
        self.context_proj.train()
        self.reward_heads.train()

        # 截取训练用的历史
        train_history = self.history
        if len(train_history) > self.max_history:
            train_history = train_history[-self.max_history:]
            logger.info(f"  [fit_repr] 历史 {len(self.history)} 条, 截取最近 {self.max_history} 条训练")

        n = len(train_history)
        indices = np.arange(n)
        bs = self.batch_size

        # 预提取 arm 和 reward 数组 (避免重复索引)
        arms = np.array([rec[4] for rec in train_history])
        rewards = np.array([rec[5] for rec in train_history], dtype=np.float32)

        # 堆叠所有 reward_heads 的权重到一个矩阵 [n_arms, phi_dim]
        # 这样可以用索引一次取出 batch 中每条样本对应 arm 的权重
        def _get_head_weights():
            return torch.stack([self.reward_heads[k].weight.squeeze(0)
                                for k in range(self.n_arms)])  # [n_arms, phi_dim]

        for step in range(self.train_steps):
            np.random.shuffle(indices)
            total_loss = 0.0
            n_batches = 0

            for start in range(0, n, bs):
                batch_idx = indices[start:start + bs]
                batch_records = [train_history[int(i)] for i in batch_idx]
                batch_arms = arms[batch_idx]           # [B]
                batch_rewards = rewards[batch_idx]     # [B]

                # 批量编码: 按问题类型分组过 encoder → [B, 64]
                phi = self._encode_batch(batch_records, grad=True)  # [B, 64]

                # 批量取 reward_head 权重并计算预测
                head_w = _get_head_weights()  # [n_arms, phi_dim]
                arm_tensor = torch.as_tensor(batch_arms, dtype=torch.long, device=self.device)
                w_selected = head_w[arm_tensor]  # [B, phi_dim]
                pred = torch.sum(phi * w_selected, dim=1)  # [B]

                target = torch.as_tensor(batch_rewards, dtype=torch.float32, device=self.device)
                loss = torch.mean((pred - target) ** 2)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item() * len(batch_idx)
                n_batches += 1

            avg_loss = total_loss / n
            logger.info(
                f"  [fit_repr] step {step+1}/{self.train_steps}, "
                f"loss={avg_loss:.6f}, n_train={n}, n_batches={n_batches}"
            )

    # ===================== 统计量重建 (批量) =====================

    def _rebuild_stats(self):
        """
        网络更新后, 用新 φ 重建所有 arm 的 Z 和 b.
        批量过 encoder, 避免逐样本前向.
        """
        self.encoder.eval()
        self.context_proj.eval()
        self._reset_linucb_stats()

        n = len(self.history)
        if n == 0:
            return

        bs = self.batch_size * 2  # rebuild 无梯度, 可以用更大 batch

        for start in range(0, n, bs):
            batch = self.history[start:start + bs]
            with torch.no_grad():
                phi_batch = self._encode_batch(batch, grad=False)  # [B, 64]
            phi_np = phi_batch.cpu().numpy()  # [B, 64]

            for j, (_, _, _, _, arm, reward) in enumerate(batch):
                phi_j = phi_np[j]
                self.Z[arm] += np.outer(phi_j, phi_j)
                self.b_vec[arm] += reward * phi_j
                self._z_dirty[arm] = True

        logger.info(f"  [rebuild] 用新 φ 重建 LinUCB 统计量 (n={n})")

    # ===================== Checkpoint =====================

    def save_checkpoint(self, path: str):
        checkpoint = {
            "encoder": self.encoder.state_dict(),
            "context_proj": self.context_proj.state_dict(),
            "reward_heads": self.reward_heads.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "Z": self.Z,
            "b_vec": self.b_vec,
            "t": self.t,
            "arm_pull_counts": self.arm_pull_counts,
            "arm_reward_sums": self.arm_reward_sums,
            "config": {
                "n_arms": self.n_arms,
                "phi_dim": self.phi_dim,
                "alpha": self.alpha,
                "reg": self.reg,
                "train_every": self.train_every,
                "train_steps": self.train_steps,
                "batch_size": self.batch_size,
                "max_history": self.max_history,
            },
        }
        torch.save(checkpoint, path)
        logger.info(f"  [checkpoint] 模型已保存到 {path} (t={self.t})")

    def load_checkpoint(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.encoder.load_state_dict(checkpoint["encoder"])
        self.context_proj.load_state_dict(checkpoint["context_proj"])
        self.reward_heads.load_state_dict(checkpoint["reward_heads"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.Z = checkpoint["Z"]
        self.b_vec = checkpoint["b_vec"]
        self.Z_inv = [np.linalg.inv(z) for z in self.Z]
        self._z_dirty = [False] * self.n_arms
        self.t = checkpoint["t"]
        self.arm_pull_counts = checkpoint["arm_pull_counts"]
        self.arm_reward_sums = checkpoint["arm_reward_sums"]
        logger.info(f"  [checkpoint] 模型已从 {path} 加载 (t={self.t})")

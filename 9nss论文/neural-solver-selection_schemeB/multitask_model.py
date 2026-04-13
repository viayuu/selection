import torch
import torch.nn as nn

from model import Encoder_block_h
from multitask_common import CVRP_NUM_CLASSES, GLOBAL_NUM_CLASSES, TSP_NUM_CLASSES


class SharedHierarchicalEncoder(nn.Module):
    """
    方案 B：
    - 仍然保留 TSP / CVRP 两个输入 stem
    - 中间 trunk 共享
    - 最后只接一个统一 12 维输出头
    """

    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = model_params["embedding_dim"]

        self.tsp_embedding = nn.Linear(2, embedding_dim)
        self.cvrp_embedding = nn.Linear(3, embedding_dim)
        self.blocks = nn.ModuleList([Encoder_block_h(**model_params) for _ in range(model_params["block_num"])])
        self.nonlinear = nn.GELU()

    def masked_mean(self, embs, mask):
        emb_mask = torch.where(mask == float("-inf"), 0, 1)
        embs = embs * emb_mask[:, :, None].expand_as(embs)
        return embs.sum(dim=1) / emb_mask.sum(-1)[:, None]

    def masked_max(self, embs, mask):
        embs = embs + mask[:, :, None].expand_as(embs)
        return embs.max(dim=1)[0]

    def forward(self, data, mask, problem_ids):
        batch_s, problem_s, _ = data.shape
        embedding_dim = self.model_params["embedding_dim"]
        out = torch.zeros(batch_s, problem_s, embedding_dim, device=data.device, dtype=data.dtype)

        tsp_mask = problem_ids == 0
        cvrp_mask = problem_ids == 1
        if tsp_mask.any():
            out[tsp_mask] = self.tsp_embedding(data[tsp_mask][:, :, :2])
        if cvrp_mask.any():
            out[cvrp_mask] = self.cvrp_embedding(data[cvrp_mask])

        for block_idx, block in enumerate(self.blocks):
            _graph_emb, out, mask = block(out, mask, block_idx)

        mean_emb = self.masked_mean(out, mask)
        max_emb = self.masked_max(out, mask)
        graph_emb = self.nonlinear(torch.cat((mean_emb, max_emb), dim=1))
        return graph_emb


class MultiTaskSelectionModel(nn.Module):
    """
    方案 B：
    - 共享 encoder trunk
    - 统一单个输出头，直接输出 12 个全局 arm 的分数
    - 训练/评测时再通过 feasible mask 只读取当前问题可用的方法
    """

    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        self.encoder = SharedHierarchicalEncoder(**model_params)
        feature_dim = 2 * model_params["embedding_dim"] + 1
        self.shared_head = nn.Sequential(
            nn.Linear(feature_dim, model_params["embedding_dim"]),
            nn.GELU(),
            nn.Linear(model_params["embedding_dim"], GLOBAL_NUM_CLASSES),
        )

    def encode(self, points, scales, mask, problem_ids):
        graph_emb = self.encoder(points, mask, problem_ids)
        return torch.cat((graph_emb, scales[:, None]), dim=1)

    def forward(self, points, scales, mask, problem_ids):
        instance_feature = self.encode(points, scales, mask, problem_ids)
        logits = self.shared_head(instance_feature)

        # 方案 B 虽然是统一 12 维输出头，但当前 batch 中每个样本
        # 实际只允许访问自己所属问题的那一段输出。
        # 这里显式把无效位置打成极小值，便于后续统一分析和调试。
        tsp_mask = problem_ids == 0
        cvrp_mask = problem_ids == 1
        if tsp_mask.any():
            logits[tsp_mask, TSP_NUM_CLASSES:] = -1e9
        if cvrp_mask.any():
            logits[cvrp_mask, :TSP_NUM_CLASSES] = -1e9

        return logits

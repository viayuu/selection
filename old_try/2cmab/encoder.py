"""
encoder.py — DualEncoder: TSP/CVRP 各一个 NSS Encoder_h

NSS Encoder_h 输入层因问题类型不同 (TSP: Linear(2,128), CVRP: Linear(3,128)),
所以需要两个实例。两者输出都是 256 维 (2 * embedding_dim)。
nss_encoder.py 是从 1two_gate/paper_encoder.py 直接复制来的。
"""
import torch
import torch.nn as nn

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from nss_encoder import Encoder_h

# NSS 默认参数 (对齐 1two_gate/features.py)
DEFAULT_PARAMS = dict(
    embedding_dim=128, encoder_layer_num=2, block_num=2,
    head_num=8, qkv_dim=16, ff_hidden_dim=512,
    norm="rezero", downsample_ratio=0.8,
)


class DualEncoder(nn.Module):
    """TSP/CVRP 各一个 Encoder_h, 输出统一 256 维"""

    def __init__(self, params: dict = None):
        super().__init__()
        p = {**DEFAULT_PARAMS, **(params or {})}
        self.tsp_encoder = Encoder_h(**{**p, "problem_type": "TSP"})
        self.cvrp_encoder = Encoder_h(**{**p, "problem_type": "CVRP"})
        self.output_dim = 2 * p["embedding_dim"]  # 256

    def forward(self, coords: torch.Tensor, mask: torch.Tensor, problem_type: str) -> torch.Tensor:
        """
        coords: [B, N, 2/3], mask: [B, N] (0=有效, -inf=填充)
        返回: [B, 256]
        """
        if problem_type.lower() == "tsp":
            return self.tsp_encoder(coords, mask)
        else:
            return self.cvrp_encoder(coords, mask)

    def load_nss_checkpoint(self, tsp_path=None, cvrp_path=None):
        """加载 NSS 预训练权重 (如果有)"""
        for path, enc, name in [(tsp_path, self.tsp_encoder, "TSP"), (cvrp_path, self.cvrp_encoder, "CVRP")]:
            if path is None:
                continue
            state = torch.load(path, map_location="cpu")
            enc_state = {k.replace("encoder.", ""): v for k, v in state.items() if k.startswith("encoder.")}
            if enc_state:
                enc.load_state_dict(enc_state, strict=False)
                print(f"  {name} encoder: 加载 {len(enc_state)} 个参数")

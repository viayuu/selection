import torch
import json
import numpy as np


# 假设您使用 voyageai 的官方库，或者您自己的 embedding 接口
import voyageai


def get_voyage_embedding(text_list, model="voyage-code-3"):
    """
    这里替换为您实际调用 Voyage API 或本地模型的代码
    """
    # 模拟返回随机向量用于演示
    # 在实际使用中，请调用 client.embed(text_list, model=model).embeddings
    print(f"Embedding {len(text_list)} features using {model}...")
    vo = voyageai.Client(api_key="pa-iLfU04WvPBA2oZGULpUH-a1FqRp4VcevLPFwtZ3Ttsa")
    return vo.embed(text_list, model=model, output_dimension=256).embeddings


# ==========================================
# 1. 定义特征语义锚点 (Feature Semantic Anchors)
# ==========================================
feature_definitions = [
    "demand[i] # Customer demand load",  # 0
    "prize[i] # Prize value for visiting node",  # 1
    "penalty[i] # Penalty cost for skipping node",  # 2
    "tw_start[i] # Time window start time",  # 3
    "tw_end[i] # Time window end time",  # 4
    "service[i] # Service duration at node",  # 5
    "is_depot[i] # Depot node indicator",  # 6
    "is_pickup[i] # Pickup location (serve first)",  # 7 (PDP专用：先服务)
    "is_delivery[i] # Delivery location (serve later)",  # 8 (PDP/CVRP专用：后服务)
    "vehicle_capacity # Multiple vehicles constraint",  # 9
    "is_open_route # Open route (no return)",  # 10
    "is_backhaul[i] # Backhaul node (must serve last)",  # 11 (VRPB专用：最后服务)
    "is_linehaul[i] # Linehaul node (must serve first)",  # 12 (VRPB专用：最先服务)
]

# ==========================================
# 2. 生成 Embeddings
# ==========================================
# 请确保这里使用和 Problem Representation 完全相同的模型！
feature_embeddings = get_voyage_embedding(feature_definitions, model="voyage-code-3")

# 转换为 Tensor
feature_embedding_tensor = torch.tensor(feature_embeddings, dtype=torch.float32)

print("Feature Embedding Shape:", feature_embedding_tensor.shape)  # 应该输出 (11, 256)

# ==========================================
# 3. 保存以供模型加载
# ==========================================
torch.save(feature_embedding_tensor, "feature_name_embeddings.pt")
print("Saved to feature_name_embeddings.pt")

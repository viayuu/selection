
import torch
import torch.nn as nn
import torch.nn.functional as F

from Model_LIB import *


class Model(nn.Module):

    def __init__(self, representation_dim=256, **model_params):
        super().__init__()
        self.model_params = model_params
        self.representation_dim = representation_dim

        self.encoder = Encoder(representation_dim=representation_dim, **model_params)
        self.decoder = Decoder(representation_dim=representation_dim, **model_params)

        embedding_dim = self.model_params['embedding_dim']
        self.position_embedding  = nn.Linear(3, embedding_dim, bias=False) # coordinate embedding
        if representation_dim == 13:
            self.attribute_embedding = nn.Linear(6, embedding_dim, bias=False)  # attribute embedding
            self.node_type_embedding = nn.Linear(5, embedding_dim, bias=False)  # node embedding
            self.embedding_enhancer = None
            self.universal_feature_encoder = None
        elif representation_dim == 256:
            self.embedding_enhancer = nn.Sequential(
                nn.LayerNorm(256),  # 关键！归一化能显著放大 0.99 相似度的差异
                nn.Linear(256, 512),
                nn.ReLU(),
                nn.Linear(512, 256),  # 保持维度不变，或者映射到更适合的维度
            )
            self.universal_feature_encoder = UniversalFeatureEncoder(
                embedding_dim=embedding_dim,
                task_emb_dim=256,
                feature_emb_path="/public/home/chenrs/project/Mask/URS_mask/feature_name_embeddings.pt"
            )

        self.problem_name = None
        self.problem_representation = None

        self.encoded_nodes = None
        # shape: (batch, node, embedding)

    def set_decoder_type(self,decoder_type):
        self.model_params['eval_type'] = decoder_type

    def pre_forward(self, reset_state, problem_name, problem_representation, mask_embedding=None):

        self.problem_name = problem_name
        raw_task_embedding = problem_representation
        if self.embedding_enhancer is not None and problem_representation is not None:
            # 增强 problem_representation
            # 注意：如果 mask_embedding 和 problem_representation 是同一个 tensor
            # 经过增强后，它们就变成了经过"特征清洗"的版本
            enhanced_repr = self.embedding_enhancer(problem_representation)

            # 更新成员变量
            self.problem_representation = enhanced_repr
            self.mask_embedding = enhanced_repr
        else:
            # 兼容旧逻辑 (representation_dim=13)
            self.problem_representation = problem_representation
            self.mask_embedding = mask_embedding
        self.decoder.assign(self.problem_representation)
        dist = reset_state.dist

        log_scale = reset_state.log_scale
        self.decoder.log_scale = log_scale # for decoder

        # distance normalization
        ##################################################
        dist_normed = distance_normalization(dist, dist_norm_style="all_max")
        dist_normed_transpose = distance_normalization(dist.transpose(1, 2), dist_norm_style="all_max")

        negative_scale_dist = -1 * log_scale * dist_normed
        negative_scale_dist_transpose = -1 * log_scale * dist_normed_transpose
        # shape: (batch, problem, problem)

        #relation matrix
        ##################################################
        relation = reset_state.relation
        negative_scale_relation = None
        if relation is not None:
            # 1. 移除 'pd' 断言，允许 TSP/CVRP/Open/MD 使用 R 矩阵
            # assert 'pd' in problem_name, "relation matrix is only for pdp problem"
            
            # 2. 对 Relation 矩阵进行归一化，使其数值范围与 Distance 一致
            # 因为我们现在的 R 是距离（势能），不再是 0/1 Mask
            relation_normed = distance_normalization(relation, dist_norm_style="all_max")
            
            # 3. 引入 log_scale (log N)，让模型能感知规模带来的势能衰减
            # 公式: f(alpha, N, R) = -alpha * logN * R_normed
            negative_scale_relation = -1 * log_scale * relation_normed

        # unified node feature extraction
        ##################################################
        # position embedding
        position_features = unified_node_position_construction(reset_state.problems, problem_name)
        # shape: (batch, problem, 3)  # 3: random identifier, x, y
        position_embedded = self.position_embedding(position_features)
        # shape: (batch, problem, embedding)

        if self.representation_dim == 13:
            attribute_node_type_features = unified_node_attribute_construction(reset_state.problems, problem_name,demand_max1=self.model_params['demand_max1'])
            attribute_embedded = self.attribute_embedding(attribute_node_type_features[:, :, :6])
            # shape: (batch, problem, embedding), attribute embedded
            node_type_embedded = self.node_type_embedding(attribute_node_type_features[:, :, 6:])
            # shape: (batch, problem, embedding), node type embedded
            init_emb = position_embedded + attribute_embedded + node_type_embedded
        elif self.representation_dim == 256:
            active_values, active_feature_indices = unified_node_attribute_construction_wide(
                reset_state.problems,
                problem_name,
                demand_max1=self.model_params['demand_max1']
            )
            if active_values.shape[-1] > 0:
                # [关键] 传入 raw_task_embedding
                attribute_embedded = self.universal_feature_encoder(
                    active_values,
                    active_feature_indices, # LongTensor of indices
                    raw_task_embedding # <--- 传入原始向量！
                )
            else:
                attribute_embedded = 0 # TSP Case
            init_emb = position_embedded + attribute_embedded
        # shape: (batch, problem, embedding)
        self.encoded_nodes = self.encoder(
            init_emb,
            negative_scale_dist,
            negative_scale_dist_transpose,
            negative_scale_relation,
            self.problem_representation,
            self.mask_embedding,
        )
        # shape: (batch, problem, embedding_dim)
        self.decoder.set_kv(self.encoded_nodes)

    def forward(self, state,cur_dist):
        batch_size = state.batch_size
        pomo_size = state.pomo_size
        if state.selected_count == 0:  # First Move, depot
            if self.problem_name in get_problem_list("single_depot_list"):
                # For problem with one depot, we need to select the depot node first
                selected = torch.zeros(size=(batch_size, pomo_size), dtype=torch.long)
                prob = torch.ones(size=(batch_size, pomo_size))
            elif self.problem_name in ['tsp','atsp']:
                selected = torch.arange(pomo_size)[None, :].expand(batch_size, pomo_size)
                prob = torch.ones(size=(batch_size, pomo_size))
            elif "md" in self.problem_name:
                selected = torch.arange(state.depot_num).repeat_interleave(pomo_size // state.depot_num)
                selected = selected.unsqueeze(0).expand(batch_size, -1)
                prob = torch.ones(size=(batch_size, pomo_size))
            else:
                raise NotImplementedError(f"problem_name: {self.problem_name} is not implemented!")

            encoded_first_node = get_encoding(self.encoded_nodes, selected)
            # shape: (batch, pomo, embedding)
            self.decoder.set_q1(encoded_first_node)

        elif state.selected_count == 1 and pomo_size > 1 and self.problem_name not in ['tsp','atsp']:  # Second Move, POMO
            if self.problem_name in get_problem_list("constraint_b_list"):
                #For VRPB, node with negative demand can not be selected as second Move
                selected = state.START_NODE
                prob = torch.ones(size=(batch_size, pomo_size))
            elif "md" in self.problem_name:
                problem_size = cur_dist.shape[-1]-state.depot_num
                selected = torch.arange(start=state.depot_num, end=problem_size + state.depot_num)
                selected = selected.repeat(batch_size, (pomo_size + problem_size - 1) // problem_size)
                selected = selected[:, :pomo_size]
                prob = torch.ones(size=(batch_size, pomo_size))
            else:
                selected = torch.arange(start=1, end=pomo_size+1)[None, :].expand(batch_size, pomo_size)
                prob = torch.ones(size=(batch_size, pomo_size))
        else:
            encoded_last_node = get_encoding(self.encoded_nodes, state.current_node)
            # shape: (batch, pomo, embedding)
            if self.problem_name in ['tsp','atsp','pdp','apdp']:
                constraint = None
            elif "vrp" in self.problem_name:
                constraint = state.load.unsqueeze(-1)
                # shape: (batch, pomo, 1)
            elif self.problem_name in ['op']:
                constraint = (state.tour_maxlength / 4.0).unsqueeze(-1)
            elif self.problem_name in ['pctsp', 'spctsp']:
                constraint = (1.0 - state.collected_prize).unsqueeze(-1)  # 还差多少就能出去
            else:
                raise NotImplementedError(f"problem_name: {self.problem_name} is not implemented!")


            probs = self.decoder(encoded_last_node,
                                 cur_dist,
                                 ninf_mask=state.ninf_mask,
                                 constraint=constraint,
                                 mask_embedding=self.mask_embedding)
            # shape: (batch, pomo, problem)

            selected, prob = select_next_node(probs, decoding_strategy=self.model_params['eval_type'])

        return selected, prob


########################################
# ENCODER
########################################

class Encoder(nn.Module):
    def __init__(self, representation_dim=256, **model_params):
        super().__init__()
        self.model_params = model_params
        self.encoder_layer_num = self.model_params['encoder_layer_num']

        self.layers = nn.ModuleList([EncoderLayer(representation_dim=representation_dim, **model_params) for _ in range(self.encoder_layer_num)])

    def forward(self, init_emb,negative_scale_dist, negative_scale_dist_transpose,negative_scale_relation,problem_representation, mask_embedding=None):
        # col_emb.shape: (batch, col_cnt, embedding)
        # row_emb.shape: (batch, row_cnt, embedding)a
        # dist.shape: (batch, row_cnt, col_cnt)

        out = init_emb
        for layer in self.layers:
            out = layer(out,negative_scale_dist, negative_scale_dist_transpose,negative_scale_relation,problem_representation, mask_embedding=mask_embedding)

        return out

class EncoderLayer(nn.Module):
    def __init__(self, representation_dim=256, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']

        self.Wq_row = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.Wk_row = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.Wv_row = nn.Linear(embedding_dim, embedding_dim, bias=False)

        self.Wq_col = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.Wk_col = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.Wv_col = nn.Linear(embedding_dim, embedding_dim, bias=False)

        self.Wq_relation = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.Wk_relation = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.Wv_relation = nn.Linear(embedding_dim, embedding_dim, bias=False)

        self.adaptation_bias_module_row = Adaptation_Bias_Module(representation_dim=representation_dim, **model_params)
        self.adaptation_bias_module_col = Adaptation_Bias_Module(representation_dim=representation_dim, **model_params)
        self.adaptation_bias_module_relation = Adaptation_Bias_Module(representation_dim=representation_dim, **model_params)

        self.row_col_relation_combine = nn.Linear(3 * embedding_dim, embedding_dim, bias=False)

        self.alpha_attn_row = None  # adaptation bias for attention
        self.alpha_attn_col = None  # adaptation bias for attention
        self.alpha_attn_relation = None  # adaptation bias for attention

        self.add_n_normalization_1 = AddAndInstanceNormalization(**model_params)
        self.feed_forward = Feed_Forward_Module(**model_params)
        self.add_n_normalization_2 = AddAndInstanceNormalization(**model_params)
        
        self.film = FiLM_Generator(256, embedding_dim)

    def forward(self, input_emb, negative_scale_dist, negative_scale_dist_transpose,negative_scale_relation,problem_representation, mask_embedding=None):
        # input_emb.shape: (batch, problem, embedding)
        # cost_mat.shape: (batch, problem, problem)

        # row, standard distance
        ########################################################
        q_row = self.Wq_row(input_emb)
        k_row = self.Wk_row(input_emb)
        v_row = self.Wv_row(input_emb)
        # shape: (batch, problem, embedding_dim)
        self.alpha_attn_row = self.adaptation_bias_module_row(problem_representation)
        alpha_adaptation_bias_attn_row = self.alpha_attn_row * negative_scale_dist
        out_attn_row = adaptation_attention_free_module(q_row, k_row, v_row, alpha_adaptation_bias_attn_row)
        # shape: (batch, problem, embedding)

        # column, distance transpose
        #########################################################
        q_col = self.Wq_col(input_emb)
        k_col = self.Wk_col(input_emb)
        v_col = self.Wv_col(input_emb)
        # shape: (batch, problem, embedding_dim)
        self.alpha_attn_col = self.adaptation_bias_module_col(problem_representation)
        alpha_adaptation_bias_attn_col = self.alpha_attn_col * negative_scale_dist_transpose
        out_attn_col = adaptation_attention_free_module(q_col, k_col, v_col, alpha_adaptation_bias_attn_col)
        # shape: (batch, problem, embedding)

        if negative_scale_relation is not None:
            q_relation = self.Wq_relation(input_emb)
            k_relation = self.Wk_relation(input_emb)
            v_relation = self.Wv_relation(input_emb)
            # shape: (batch, problem, embedding_dim)
            self.alpha_attn_relation = self.adaptation_bias_module_relation(problem_representation)
            alpha_adaptation_bias_attn_relation = self.alpha_attn_relation * negative_scale_relation #
            out_attn_relation = adaptation_attention_free_module(q_relation, k_relation, v_relation, alpha_adaptation_bias_attn_relation)
        else:
            out_attn_relation = torch.zeros_like(out_attn_row)

        # combine row, column and relation
        out_attn = self.row_col_relation_combine(torch.cat([out_attn_row, out_attn_col,out_attn_relation], dim=-1))
        # shape: (batch, problem, embedding)

        out1 = self.add_n_normalization_1(input_emb, out_attn)
        if mask_embedding is not None:
            out_modulated = self.film(out1, mask_embedding)
        else:
            out_modulated = out1
        out2 = self.feed_forward(out_modulated)
        out3 = self.add_n_normalization_2(out_modulated, out2)

        return out3
        # shape: (batch, problem, embedding)

########################################
# DECODER
########################################

class Decoder(nn.Module):
    def __init__(self, representation_dim=256, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']

        hyper_input_dim = representation_dim
        hyper_hidden_embd_dim = 256
        self.embd_dim = hyper_input_dim  # 这个维度 应该是固定的
        self.hyper_output_dim = 5 * self.embd_dim

        self.hyper_fc1 = nn.Linear(hyper_input_dim, hyper_hidden_embd_dim, bias=True)  # problem_type_num -> 256
        self.hyper_fc2 = nn.Linear(hyper_hidden_embd_dim, hyper_hidden_embd_dim, bias=True)  # 256->256
        self.hyper_fc3 = nn.Linear(hyper_hidden_embd_dim, self.hyper_output_dim, bias=True)  # 256-> 45

        self.hyper_Wq_first = nn.Linear(self.embd_dim, embedding_dim * embedding_dim, bias=False)
        self.hyper_Wq_last = nn.Linear(self.embd_dim, embedding_dim * embedding_dim, bias=False)
        self.hyper_Wk = nn.Linear(self.embd_dim, embedding_dim * embedding_dim, bias=False)
        self.hyper_Wv = nn.Linear(self.embd_dim, embedding_dim * embedding_dim, bias=False)
        self.hyper_Wc = nn.Linear(self.embd_dim, 1 * embedding_dim, bias=False)  # constraints embedding

        self.Wq_first_para = None
        self.Wq_last_para = None
        self.Wk_para = None
        self.Wv_para = None
        self.Wc_para = None

        self.k = None  # saved key, for multi-head attention
        self.v = None  # saved value, for multi-head_attention
        self.single_head_key = None  # saved, for single-head attention
        self.q_first = None  # saved q1, for multi-head attention

        self.adaptation_bias_module_attn = Adaptation_Bias_Module(representation_dim=representation_dim, **model_params)
        self.adaptation_bias_module_com = Adaptation_Bias_Module(representation_dim=representation_dim, **model_params)
        self.alpha_attn = None  # adaptation bias for attention
        self.alpha_com = None  # adaptation bias for compatibility
        self.logit_clipping = self.model_params['logit_clipping']
        self.film_query = FiLM_Generator(256, embedding_dim)

    def assign(self, problem_representation):  # assign->pre_forward->forward
        embedding_dim = self.model_params['embedding_dim']

        hyper_embd = self.hyper_fc1(problem_representation)
        hyper_embd = self.hyper_fc2(hyper_embd)
        mid_embd = self.hyper_fc3(hyper_embd)
        # mid_embd.shape: (hyper_output_dim,)

        self.Wq_first_para = self.hyper_Wq_first(mid_embd[:self.embd_dim]).reshape(embedding_dim, embedding_dim)
        self.Wq_last_para = self.hyper_Wq_last(mid_embd[self.embd_dim: 2 * self.embd_dim]).reshape(embedding_dim,embedding_dim)
        self.Wk_para = self.hyper_Wk(mid_embd[2 * self.embd_dim: 3 * self.embd_dim]).reshape(embedding_dim,embedding_dim)
        self.Wv_para = self.hyper_Wv(mid_embd[3 * self.embd_dim: 4 * self.embd_dim]).reshape(embedding_dim,embedding_dim)
        # Note that F.linear execute calculation in the form of xW^T + b, so we need to transpose the weight matrix.
        self.Wc_para = self.hyper_Wc(mid_embd[4 * self.embd_dim: 5 * self.embd_dim]).reshape(embedding_dim, 1)

        self.alpha_attn = self.adaptation_bias_module_attn(problem_representation)
        self.alpha_com = self.adaptation_bias_module_com(problem_representation)

    def set_kv(self, encoded_nodes):
        # encoded_nodes.shape: (batch, problem, embedding)

        self.k = F.linear(encoded_nodes, self.Wk_para)
        self.v = F.linear(encoded_nodes, self.Wv_para)
        # shape: (batch, problem, embedding)
        self.single_head_key = encoded_nodes.transpose(1, 2)
        # shape: (batch, embedding, problem)

    def set_q1(self, encoded_q1):
        # encoded_q.shape: (batch, n, embedding)  # n can be 1 or pomo
        self.q_first = F.linear(encoded_q1, self.Wq_first_para)
        # shape: (batch, head_num, pomo, qkv_dim)

    def forward(self, encoded_last_node, cur_dist,ninf_mask, constraint=None, mask_embedding=None):
        # encoded_last_node.shape: (batch, pomo, embedding)
        # ninf_mask.shape: (batch, pomo, problem)
        # cur_dist.shape: (batch, pomo, problem)
        # constraints.shape: (batch, pomo,x)
        cur_dist = distance_normalization(cur_dist, dist_norm_style="sep_max")
        negative_scale_dist = -1 * self.log_scale * cur_dist # smaller value means better

        q_last = F.linear(encoded_last_node, self.Wq_last_para)
        # shape: (batch, pomo, embedding)

        if constraint is None:
            q = self.q_first + q_last
            # shape: (batch, pomo, embedding_dim)
        else:
            constraint_embedded = F.linear(constraint.clone(), self.Wc_para)
            # shape: (batch, pomo, embedding_dim)
            q = self.q_first + q_last + constraint_embedded
            # shape: (batch, pomo, embedding_dim)

        if self.film_query is not None and mask_embedding is not None:
            q = self.film_query(q, mask_embedding)

        #  We use AAFM to replace the multi-head attention
        #######################################################
        alpha_adaptation_bias_attn = self.alpha_attn * negative_scale_dist
        out_attn = adaptation_attention_free_module(q, self.k, self.v, alpha_adaptation_bias_attn, ninf_mask)
        # shape: (batch, pomo, embedding)

        #  Single-Head Attention, for probability calculation
        #######################################################
        score = torch.matmul(out_attn, self.single_head_key)
        # shape: (batch, pomo, problem)
        score_scaled = score / torch.sqrt(torch.tensor(self.model_params['embedding_dim'], dtype=torch.float))
        # shape: (batch, pomo, problem)

        alpha_adaptation_bias_com = self.alpha_com * negative_scale_dist
        score_scaled = score_scaled + alpha_adaptation_bias_com
        # shape: (batch, pomo, problem)
        score_clipped = self.logit_clipping * torch.tanh(score_scaled)
        # shape: (batch, pomo, problem)
        score_masked = score_clipped + ninf_mask

        probs = F.softmax(score_masked, dim=-1)
        # shape: (batch, pomo, problem)

        return probs


class FiLM_Generator(nn.Module):
    def __init__(self, task_emb_dim, feature_dim):
        super().__init__()
        # 将 Task Embedding 映射到 2倍的特征维度 (一份给gamma, 一份给beta)
        self.linear = nn.Linear(task_emb_dim, 2 * feature_dim)
        self._init_weights()  # [新增] 初始化权重

    # [新增] 零初始化方法
    def _init_weights(self):
        # 将权重初始化为0，使得初始状态下 Gamma=0, Beta=0
        # 这样 output = x * (1+0) + 0 = x
        # 也就是一开始 FiLM 不起作用，随着训练进行，模型慢慢学会调制
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x, task_emb):
        # ... (后续逻辑保持不变)
        if task_emb.dim() == 1:
            task_emb = task_emb.unsqueeze(0)

        params = self.linear(task_emb)

        if x.dim() == 3:
            params = params.unsqueeze(1)
        elif x.dim() == 2:
            pass

        gamma, beta = torch.split(params, params.size(-1) // 2, dim=-1)

        return x * (1 + gamma) + beta


class UniversalFeatureEncoder(nn.Module):
    def __init__(
        self,
        embedding_dim,
        task_emb_dim=256,
        feature_emb_path="feature_name_embeddings.pt",
    ):
        super().__init__()
        self.embedding_dim = embedding_dim

        # 加载特征名 Embedding (11, 256)
        # 确保路径正确，并且 map_location 适配设备
        loaded_emb = torch.load(feature_emb_path, map_location="cpu")
        self.register_buffer("feature_registry", loaded_emb)

        self.value_encoder = nn.Sequential(
            nn.Linear(1, embedding_dim // 2),
            nn.ReLU(),
            nn.Linear(embedding_dim // 2, embedding_dim),
        )

        self.query_proj = nn.Linear(task_emb_dim, task_emb_dim, bias=False)
        self.key_proj = nn.Linear(task_emb_dim, task_emb_dim, bias=False)

        self.base_task_embedding = nn.Parameter(torch.zeros(1, task_emb_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(10))

        self.output_projection = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )

    def forward(self, feature_values, feature_indices, raw_task_embedding):
        """
        feature_values: (Batch, Node, K) - 数值
        feature_indices: (K) - 对应的特征索引 (LongTensor)
        raw_task_embedding: (Batch, 256) 或 (1, 256) - 任务向量
        """
        batch_size = feature_values.size(0)
        device = feature_values.device
        self.feature_registry = self.feature_registry.to(device)

        # A. 提取当前用到的特征名向量
        # (K, 256)
        active_name_embs = self.feature_registry[feature_indices]

        # B. 构建 Query (Task)
        # 1. 差分
        delta_task = raw_task_embedding - self.base_task_embedding

        # 2. 投影并归一化
        query = F.normalize(self.query_proj(delta_task), dim=-1).unsqueeze(1) # (1, 1, 256) 或 (B, 1, 256)

        # [修复核心]: 显式扩展 Batch 维度以匹配 feature_values
        if query.size(0) == 1 and batch_size > 1:
            query = query.expand(batch_size, -1, -1) # (B, 1, 256)

        # C. 构建 Key (Feature Names)
        keys = F.normalize(self.key_proj(active_name_embs), dim=-1) # (K, 256)
        # 扩展到 Batch: (B, K, 256)
        keys = keys.unsqueeze(0).expand(batch_size, -1, -1) 

        # D. 计算匹配度
        # bmm: (B, 1, 256) x (B, 256, K) -> (B, 1, K)
        logits = torch.bmm(query, keys.transpose(1, 2)) * self.logit_scale.exp()
        attn_gates = torch.sigmoid(logits).unsqueeze(2) # (B, 1, 1, K)

        # E. 数值编码与融合
        val_embs = self.value_encoder(feature_values.unsqueeze(-1)) # (B, N, K, Dim)
        fused = (val_embs * attn_gates.transpose(2, 3)).sum(dim=2) # (B, N, Dim)

        active_count = attn_gates.sum(dim=-1).squeeze(1) + 1e-6  # 防止除0
        fused = fused / torch.sqrt(active_count).unsqueeze(-1)

        fused = self.output_projection(fused)

        return fused

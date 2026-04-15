import torch
import torch.nn as nn
import torch.nn.functional as F


class UnifiedSelectionModel(nn.Module):
    def __init__(self, solver_features=None, **model_params):
        super().__init__()
        self.model_params = model_params
        self.num_solvers = model_params["num_solvers"]
        self.num_problems = model_params["num_problems"]
        self.embedding_dim = model_params["embedding_dim"]
        self.use_instance_stats = bool(model_params.get("use_instance_stats", True))
        self.use_film = bool(model_params.get("use_film", True)) and self.use_instance_stats
        self.use_solver_interaction = bool(model_params.get("use_solver_interaction", True))

        if model_params["pooling"]:
            self.encoder = Encoder_h(**model_params)
            graph_feature_dim = 2 * self.embedding_dim
        else:
            self.encoder = Naive_Encoder(**model_params)
            graph_feature_dim = self.embedding_dim

        self.problem_embedding = nn.Embedding(self.num_problems, self.embedding_dim)
        self.problem_feature_proj = nn.Linear(model_params["problem_feature_dim"], self.embedding_dim)
        self.scale_proj = nn.Linear(1, self.embedding_dim)
        self.stats_proj = nn.Sequential(
            nn.Linear(model_params["instance_stats_dim"], self.embedding_dim),
            nn.GELU(),
            nn.Linear(self.embedding_dim, self.embedding_dim),
        )
        self.film_gen = nn.Sequential(
            nn.Linear(2 * self.embedding_dim, 2 * graph_feature_dim),
            nn.GELU(),
            nn.Linear(2 * graph_feature_dim, 2 * graph_feature_dim),
        )
        self.instance_proj = nn.Sequential(
            nn.Linear(graph_feature_dim + 4 * self.embedding_dim, 2 * self.embedding_dim),
            nn.GELU(),
            nn.Linear(2 * self.embedding_dim, self.embedding_dim),
        )

        self.solver_embedding = nn.Embedding(self.num_solvers, self.embedding_dim)
        self.solver_feature_proj = None
        if solver_features is not None:
            solver_features = torch.as_tensor(solver_features, dtype=torch.float32)
            self.register_buffer("solver_feature_table", solver_features)
            self.solver_feature_proj = nn.Linear(model_params["solver_feature_dim"], self.embedding_dim)
        else:
            self.register_buffer("solver_feature_table", None)

        scorer_input_dim = 3 * self.embedding_dim if self.use_solver_interaction else 2 * self.embedding_dim
        self.scorer = nn.Sequential(
            nn.Linear(scorer_input_dim, self.embedding_dim),
            nn.GELU(),
            nn.Linear(self.embedding_dim, 1),
        )

    def forward(self, points, scales, node_mask, problem_ids, problem_features, feasible_mask, instance_stats):
        graph_emb = self.encoder(points, node_mask)
        scale_emb = self.scale_proj(torch.log1p(scales).unsqueeze(-1))
        problem_emb = self.problem_embedding(problem_ids)
        problem_feature_emb = self.problem_feature_proj(problem_features)
        if self.use_instance_stats:
            stats_emb = self.stats_proj(instance_stats)
        else:
            stats_emb = torch.zeros_like(problem_emb)

        if self.use_film:
            gamma_beta = self.film_gen(torch.cat((problem_emb, stats_emb), dim=-1))
            gamma, beta = torch.chunk(gamma_beta, 2, dim=-1)
            graph_emb = graph_emb * (1.0 + 0.1 * torch.tanh(gamma)) + 0.1 * beta

        instance_repr = self.instance_proj(
            torch.cat((graph_emb, scale_emb, problem_emb, problem_feature_emb, stats_emb), dim=-1)
        )

        solver_repr = self.solver_embedding.weight
        if self.solver_feature_proj is not None and self.solver_feature_table is not None:
            solver_repr = solver_repr + self.solver_feature_proj(self.solver_feature_table)

        batch_size = instance_repr.size(0)
        pair_terms = [
            instance_repr[:, None, :].expand(batch_size, self.num_solvers, -1),
            solver_repr[None, :, :].expand(batch_size, self.num_solvers, -1),
        ]
        if self.use_solver_interaction:
            pair_terms.append(
                instance_repr[:, None, :].expand(batch_size, self.num_solvers, -1)
                * solver_repr[None, :, :].expand(batch_size, self.num_solvers, -1)
            )
        pair_repr = torch.cat(pair_terms, dim=-1)
        logits = self.scorer(pair_repr).squeeze(-1)
        logits = logits.masked_fill(~feasible_mask.bool(), -1e9)
        return logits

class Naive_Encoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        node_dim = self.model_params['node_dim']
        self.embedding = nn.Linear(node_dim, embedding_dim)
        
        encoder_layer_num = self.model_params['encoder_layer_num'] * self.model_params['block_num']
        self.layers = nn.ModuleList([EncoderLayer(**model_params) for _ in range(encoder_layer_num)])
    
    def forward(self, points, mask):
        # points shape: (batch, n', node_dim)
        # mask shape: (batch, n', n')
        embs = self.embedding(points)

        for layer in self.layers:
            embs = layer(embs, mask=mask)

        # Mask embs of padded nodes as 0
        emb_mask = torch.where(mask == float('-inf'), 0, 1)
        
        embs = embs * emb_mask[:, :, None].expand_as(embs)
        graph_emb = embs.sum(dim=1) / emb_mask.sum(-1)[:, None]

        return graph_emb

class Encoder_h(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        node_dim = self.model_params['node_dim']
        self.embedding = nn.Linear(node_dim, embedding_dim)
        self.blocks = nn.ModuleList([Encoder_block_h(**model_params) for _ in range(model_params['block_num'])])
        self.nonlinear = nn.GELU()

    def masked_mean(self, embs, mask):
        # Mask embs of padded nodes as 0
        emb_mask = torch.where(mask == float('-inf'), 0, 1)
        
        embs = embs * emb_mask[:, :, None].expand_as(embs)
        mean_emb = embs.sum(dim=1) / emb_mask.sum(-1)[:, None]

        return mean_emb
    
    def masked_max(self, embs, mask):
        # Mask embs of padded nodes as -inf
        embs = embs + mask[:, :, None].expand_as(embs)
        max_emb = embs.max(dim=1)[0]

        return max_emb

    def forward(self, data, mask):
        # data.shape: (batch, problem, 2)
        out = self.embedding(data)
        # shape: (batch, problem, embedding)

        # Hierachy embedding
        i = 0
        graph_emb_h = 0.
        for block in self.blocks:
            graph_emb, out, mask = block(out, mask, i)
            graph_emb_h += graph_emb
            i += 1

        mean_emb = self.masked_mean(out, mask)
        max_emb = self.masked_max(out, mask)
        graph_emb = self.nonlinear(torch.cat((mean_emb, max_emb), dim=1))
        # shape: (batch, 2 * embedding)
        graph_emb_h += graph_emb

        return graph_emb

class Encoder_block_h(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        encoder_layer_num = self.model_params['encoder_layer_num']
        self.layers = nn.ModuleList([EncoderLayer(**model_params) for _ in range(encoder_layer_num)])
        self.layer_score = EncoderLayer(**model_params)
        self.p = nn.Linear(model_params['embedding_dim'], 1)
        self.modulate = nn.Linear(1, model_params['embedding_dim'])
        self.act = nn.Tanh()
        self.nonlinear = nn.GELU()
    
    def padding_concate(self, list_):
        lengths = torch.tensor([t.shape[0] for t in list_], device=list_[0].device)
        max_len = lengths.max().item()
        mask = torch.zeros(len(list_), max_len, device=list_[0].device)
        for i, t in enumerate(list_):
            list_[i] = F.pad(t, (0, 0, 0, max_len - lengths[i]))[None, :, :]
            mask[i, lengths[i]: max_len] = float('-inf')
        embs = torch.cat(list_, dim=0)

        return embs, mask

    def masked_mean(self, embs, mask):
        # Mask embs of padded nodes as 0
        emb_mask = torch.where(mask == float('-inf'), 0, 1)
        
        embs = embs * emb_mask[:, :, None].expand_as(embs)
        mean_emb = embs.sum(dim=1) / emb_mask.sum(-1)[:, None]

        return mean_emb
    
    def masked_max(self, embs, mask):
        # Mask embs of padded nodes as -inf
        embs = embs + mask[:, :, None].expand_as(embs)
        max_emb = embs.max(dim=1)[0]

        return max_emb

    def forward(self, embs, mask, i):
        # embs shape: (batch, n', embedding)
        # adj_mat shape: (batch, n', n')
        # mask shape: (batch, n', n')

        for layer in self.layers:
            embs = layer(embs, mask)
        
        mean_emb = self.masked_mean(embs, mask)
        max_emb = self.masked_max(embs, mask)
        graph_emb = self.nonlinear(torch.cat((mean_emb, max_emb), dim=1))
        # shape: (batch, 2 * embedding)

        # Downsampling
        score_embs = self.layer_score(embs, mask)
        scores = self.act(self.p(score_embs))
        scores = scores.squeeze(-1)
        scores = scores + mask

        selected_embs_list = []
        unselected_embs_list = []
        lengths = (mask == 0).sum(dim=-1, keepdim=True)
        for i in range(embs.shape[0]):
            score, ind = scores[i].topk(int(lengths[i] * self.model_params['downsample_ratio']), dim=-1, largest=True)
            selected_emb = embs[i].take_along_dim(ind[:, None].expand(-1, embs.shape[-1]), dim=0)
            selected_emb = selected_emb + score[:, None]
            # selected_emb = selected_emb + self.act(self.modulate(score[:, None]))
            selected_embs_list.append(selected_emb)


        # Pad and cat
        selected_embs, selected_mask = self.padding_concate(selected_embs_list)

        return graph_emb, selected_embs, selected_mask

class EncoderLayer(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        head_num = self.model_params['head_num']
        qkv_dim = self.model_params['qkv_dim']

        self.Wq = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)

        self.addAndNormalization1 = Add_And_Normalization_Module(**model_params)
        self.feedForward = Feed_Forward_Module(**model_params)
        self.addAndNormalization2 = Add_And_Normalization_Module(**model_params)

    def forward(self, input1, mask=None, edges=None, kv=None):
        # input.shape: (batch, problem, EMBEDDING_DIM)
        head_num = self.model_params['head_num']
        if kv is None:
            kv = input1
        q = reshape_by_heads(self.Wq(input1), head_num=head_num)
        k = reshape_by_heads(self.Wk(kv), head_num=head_num)
        v = reshape_by_heads(self.Wv(kv), head_num=head_num)
        # q shape: (batch, HEAD_NUM, problem, KEY_DIM)

        out_concat = multi_head_attention(q, k, v, rank2_ninf_mask=mask)
        # shape: (batch, problem, HEAD_NUM*KEY_DIM)

        multi_head_out = self.multi_head_combine(out_concat)
        # shape: (batch, problem, EMBEDDING_DIM)

        out1 = self.addAndNormalization1(input1, multi_head_out)
        out2 = self.feedForward(out1)
        out3 = self.addAndNormalization2(out1, out2)

        return out3
        # shape: (batch, problem, EMBEDDING_DIM)

def reshape_by_heads(qkv, head_num):
    # q.shape: (batch, n, head_num*key_dim)   : n can be either 1 or PROBLEM_SIZE
    batch_s = qkv.size(0)
    n = qkv.size(1)

    q_reshaped = qkv.reshape(batch_s, n, head_num, -1)
    # shape: (batch, n, head_num, key_dim)

    q_transposed = q_reshaped.transpose(1, 2)
    # shape: (batch, head_num, n, key_dim)

    return q_transposed

def multi_head_attention(q, k, v, rank2_ninf_mask=None, rank3_ninf_mask=None):
    # q shape: (batch, head_num, n, key_dim)   : n can be either 1 or PROBLEM_SIZE
    # k,v shape: (batch, head_num, problem, key_dim)
    # rank2_ninf_mask.shape: (batch, problem)
    # rank3_ninf_mask.shape: (batch, group, problem)

    batch_s = q.size(0)
    head_num = q.size(1)
    n = q.size(2)
    key_dim = q.size(-1)

    input_s = k.size(2)

    score = torch.matmul(q, k.transpose(2, 3))
    # shape: (batch, head_num, n, problem)

    score_scaled = score / (float(key_dim) ** 0.5)

    if rank2_ninf_mask is not None:
        score_scaled = score_scaled + rank2_ninf_mask[:, None, None, :].expand(batch_s, head_num, n, input_s)
    if rank3_ninf_mask is not None:
        score_scaled = score_scaled + rank3_ninf_mask[:, None, :, :].expand(batch_s, head_num, n, input_s)

    weights = nn.Softmax(dim=3)(score_scaled)
    # shape: (batch, head_num, n, problem)
    assert not score_scaled.isinf().all(dim=-1).any(), "All the valid nodes are filtered! Check the pooling operation."

    out = torch.matmul(weights, v)
    # shape: (batch, head_num, n, key_dim)
        
    out_transposed = out.transpose(1, 2)
    # shape: (batch, n, head_num, key_dim)
    out_concat = out_transposed.reshape(batch_s, n, head_num * key_dim)
    # shape: (batch, n, head_num*key_dim)
    
    return out_concat

class Add_And_Normalization_Module(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params['embedding_dim']
        if model_params["norm"] == "batch":
            self.norm = nn.BatchNorm1d(embedding_dim, affine=True, track_running_stats=True)
        elif model_params["norm"] == "batch_no_track":
            self.norm = nn.BatchNorm1d(embedding_dim, affine=True, track_running_stats=False)
        elif model_params["norm"] == "instance":
            self.norm = nn.InstanceNorm1d(embedding_dim, affine=True, track_running_stats=False)
        elif model_params["norm"] == "rezero":
            self.norm = torch.nn.Parameter(torch.Tensor([0.]), requires_grad=True)
        else:
            self.norm = None

    def forward(self, input1, input2):
        # input.shape: (batch, problem, embedding)

        if isinstance(self.norm, nn.InstanceNorm1d):
            added = input1 + input2
            transposed = added.transpose(1, 2)
            # shape: (batch, embedding, problem)
            normalized = self.norm(transposed)
            # shape: (batch, embedding, problem)
            back_trans = normalized.transpose(1, 2)
            # shape: (batch, problem, embedding)
        elif isinstance(self.norm, nn.BatchNorm1d):
            added = input1 + input2
            batch, problem, embedding = added.size()
            normalized = self.norm(added.reshape(batch * problem, embedding))
            back_trans = normalized.reshape(batch, problem, embedding)
        elif isinstance(self.norm, nn.Parameter):
            back_trans = input1 + self.norm * input2
        else:
            back_trans = input1 + input2

        return back_trans

class Feed_Forward_Module(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params['embedding_dim']
        ff_hidden_dim = model_params['ff_hidden_dim']

        self.W1 = nn.Linear(embedding_dim, ff_hidden_dim)
        self.W2 = nn.Linear(ff_hidden_dim, embedding_dim)

    def forward(self, input1):
        # input.shape: (batch, problem, embedding)

        return self.W2(F.relu(self.W1(input1)))


import numpy as np
import torch
from tensordict import TensorDict
import torch.nn as nn


from EasyNCO.neural_solvers.methods.am.am_encoder import AttentionModelEncoder
from EasyNCO.data.data_utils import augment_pomo
from EasyNCO.neural_solvers.backbones.Transformer.attention import  Compatibility
from EasyNCO.neural_solvers.backbones.Transformer.attr_component import multi_head_attention,reshape_by_heads




class Policy_SHPP(nn.Module): 
    
    def __init__(self, 
                 env_name: str = "tsp",
                 embed_dim: int = 128,
                 qkv_dim: int = 16, 
                 num_encoder_layers: int = 12,
                 num_heads: int = 8,
                 normalization: str = "batch", 
                 feedforward_hidden: int = 512,
                 ):
        super().__init__()

        
        self.encoder = AttentionModelEncoder(
                        env_name=env_name,
                        embed_dim=embed_dim,
                        num_heads=num_heads,
                        qkv_dim=qkv_dim,
                        num_layers=num_encoder_layers,
                        normalization=normalization,
                        feedforward_hidden=feedforward_hidden,
                        )
        
    
        self.decoder = PathDecoder(
                        embed_dim = 128, 
                        num_heads = 8,
                        tanh_clipping = 10.0,
                        num_decoding_neighbors = None)


    def forward(self, batch, source_nodes, target_nodes, 
                val_type = "x8Aug_2Traj", group_size = 2, greedy = True):

        self.device = batch.device

        if "x8Aug" in val_type:
            kwargs = {"problems":batch,'aug_factor':8}
            batch = augment_pomo(**kwargs)
            source_nodes = torch.repeat_interleave(source_nodes, 8, dim=0)  #（B*8，1）
            target_nodes = torch.repeat_interleave(target_nodes, 8, dim=0)  #（B*8，1）

        B, N, _ = batch.shape
        G = group_size
        batch_idx_range = torch.arange(B)[:, None].expand(B, G) 
        group_idx_range = torch.arange(G)[None, :].expand(B, G)
        source_action = source_nodes.view(B, 1).expand(B, G) 
        target_action = target_nodes.view(B, 1).expand(B, G)

        env = SHPPEnv(batch)
        state, reward, done = env.reset(group_size=G, source=source_action, target=target_action)

        td = TensorDict({'locs':batch},batch_size=batch.shape[0])
        embeddings,_ = self.encoder(td)


        first_action = torch.arange(N, device=self.device)[None, :G].expand(B, G)
        state, reward, done = env.step(first_action)
        self.decoder.reset(batch, embeddings, env.ninf_mask, source_action, target_action, first_action)

        for _ in range(N - 2):
            action_probs = self.decoder(env.current_node)
            if greedy:
                action = action_probs.argmax(dim=2)
            else:
                action = (action_probs.reshape(B * G, -1).multinomial(1).squeeze(dim=1).reshape(B, G))

            state, reward, done = env.step(action)

        interval = torch.tensor([-1], device=self.device).long().expand(B, G)
        selected_node_list = torch.cat((state.selected_node_list, interval[:, :, None]),dim=2).flatten()
        unique_selected_node_list = selected_node_list.unique_consecutive()
        pi = unique_selected_node_list.view([B, G, -1])[..., :-1]  


        if val_type == "noAug_1Traj":
            max_reward = reward
            best_pi = pi
        elif val_type == "noAug_nTraj":
            max_reward, idx_dim_1 = reward.max(dim=1)
            idx_dim_1 = idx_dim_1.reshape(B, 1, 1)
            best_pi = pi.gather(1, idx_dim_1.repeat(1, 1, N))
        else:
            B = round(B / 8)
            reward = reward.reshape(8, B, G)
            max_reward, idx_dim_2 = reward.max(dim=2)
            max_reward, idx_dim_0 = max_reward.max(dim=0)
            pi = pi.reshape(8, B, G, N)

            idx_dim_0 = idx_dim_0.reshape(1, B, 1, 1)
            idx_dim_2 = idx_dim_2.reshape(8, B, 1, 1).gather(0, idx_dim_0)
            best_pi = pi.gather(0, idx_dim_0.repeat(1, 1, G, N))
            best_pi = best_pi.gather(2, idx_dim_2.repeat(1, 1, 1, N)) 


        best_pi = best_pi.squeeze()



        return -max_reward, best_pi


    def low_level_training(self, batch, norm_reward = False, group_size = None):

        B, N, _ = batch.shape
        G = group_size
        assert G <= self.cfg.graph_size
        batch_idx_range = torch.arange(B, device=self.device)[:, None].expand(B, G)
        group_idx_range = torch.arange(G, device=self.device)[None, :].expand(B, G)
        group_prob_list = torch.zeros(B, G, 0, device=self.device)

        source_nodes, target_nodes, _ = torch.split(tensor=torch.argsort(torch.rand(B, N, device=self.device)),
                                        split_size_or_sections=[1, 1, N - 2],dim=-1)
        # source.shape[B,1];target.shape[B,1];_.shape[B,N-2] 

        # we need manually set source node in env
        source_action = source_nodes.view(B, 1).expand(B, G)
        target_action = target_nodes.view(B, 1).expand(B, G)

        env = SHPPEnv(batch)
        s, r, d = env.reset(group_size=G, source=source_action, target=target_action)
        td = TensorDict({'locs':batch},batch_size=batch.shape[0])
        embeddings = self.encoder(td)

        first_action = torch.randperm(N, device=self.device)[None, :G].expand(B, G)
        s, r, d = env.step(first_action)

        self.decoder.reset(batch, embeddings, s.ninf_mask, source_action, target_action, first_action)

        for _ in range(N - 2):
            action_probs = self.decoder(s.current_node)
            action = (action_probs.reshape(B * G, -1).multinomial(1).squeeze(dim=1).reshape(B, G))

            chosen_action_prob = action_probs[batch_idx_range, group_idx_range, action].reshape(B, G)
            group_prob_list = torch.cat((group_prob_list, chosen_action_prob[:, :, None]), dim=2)
            s, r, d = env.step(action)

        eps = torch.finfo(r.dtype).eps
        # Note that when G == 1, we can only use the PG without baseline so far
        if norm_reward:
            advantage = (r - r.mean(dim=1, keepdim=True)) / (r.std(dim=1, keepdim=True) + 1e-5) if G != 1 else r
        else:
            advantage = r - r.mean(dim=1, keepdim=True) if G != 1 else r

        log_prob = group_prob_list.log().sum(dim=2)
        entropy = -(log_prob.exp() * log_prob).mean()
        loss = (-advantage * log_prob).mean()
        length = -r.max(dim=1)[0].mean().clone().detach().item()  
        return  {"loss": loss, "length": length}


class PathDecoder(nn.Module):
    def __init__(self,
                embed_dim: int = 128,  ## ??
                num_heads: int = 8,
                tanh_clipping: int = 10.0 ,
                num_decoding_neighbors =None
        ):
    
        super().__init__()
        self.embedding_dim = embed_dim
        self.n_heads = num_heads
        self.tanh_clipping = tanh_clipping
        self.n_decoding_neighbors = num_decoding_neighbors

        self.Wq_graph = torch.nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
        self.Wq_source = torch.nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
        self.Wq_target = torch.nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
        self.Wq_first = torch.nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
        self.Wq_last = torch.nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
        self.Wk = torch.nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
        self.Wv = torch.nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)

        self.multi_head_combine = torch.nn.Linear(self.embedding_dim, self.embedding_dim)

        self.q_graph = None  # saved q1, for multi-head attention
        self.q_source = None  # saved q2, for multi-head attention
        self.q_target = None  # saved q3, for multi-head attention
        self.q_first = None  # saved q4, for multi-head attention
        self.glimpse_k = None  # saved key, for multi-head attention
        self.glimpse_v = None  # saved value, for multi-head_attention
        self.logit_k = None  # saved, for single-head attention
        self.group_ninf_mask = None  # reference to ninf_mask owned by state

    def reset(self, coordinates, embeddings, group_ninf_mask, source_node, target_node, first_node):

        # embeddings.shape = [B, N, H]
        # graph_embedding.shape = [B, 1, H]
        B, N, H = embeddings.shape
        G = group_ninf_mask.size(1)
        self.coordinates = coordinates
        self.embeddings = embeddings
        graph_embedding = self.embeddings.mean(dim=1, keepdim=True)

        #q_graph.shape = [B, n_heads, G, key_dim]
        self.q_graph = reshape_by_heads(self.Wq_graph(graph_embedding), self.n_heads)

        # q_source.shape = [B, n_heads, G, key_dim]
        source_node_index = source_node.view(B, G, 1).expand(B, G, H)
        source_node_embedding = self.embeddings.gather(1, source_node_index)
        self.q_source = reshape_by_heads(self.Wq_source(source_node_embedding), self.n_heads)

        # q_target.shape = [B, n_heads, G, key_dim]
        target_node_index = target_node.view(B, G, 1).expand(B, G, H)
        target_node_embedding = self.embeddings.gather(1, target_node_index)
        self.q_target = reshape_by_heads(self.Wq_target(target_node_embedding), self.n_heads)

        # q_first.shape = [B, n_heads, G, key_dim]
        first_node_index = first_node.view(B, G, 1).expand(B, G, H)
        first_node_embedding = self.embeddings.gather(1, first_node_index)
        self.q_first = reshape_by_heads(self.Wq_first(first_node_embedding), self.n_heads)


        # glimpse_k.shape = glimpse_v.shape =[B, n_heads, N, key_dim]
        # logit_k.shape = [B, H, N]
        # group_ninf_mask.shape = [B, G, N]
        self.glimpse_k = reshape_by_heads(self.Wk(embeddings), self.n_heads)
        self.glimpse_v = reshape_by_heads(self.Wv(embeddings), self.n_heads)
        self.logit_k = embeddings.transpose(1, 2)
        self.group_ninf_mask = group_ninf_mask

    def forward(self, last_node):

        B, N, H = self.embeddings.shape
        G = self.group_ninf_mask.size(1)

        # q_last.shape = q_last.shape = [B, n_heads, G, key_dim]
        last_node_index = last_node.view(B, G, 1).expand(-1, -1, H)
        last_node_embedding = self.embeddings.gather(1, last_node_index)
        q_last = reshape_by_heads(self.Wq_last(last_node_embedding), self.n_heads)

        # glimpse_q.shape = [B, n_heads, G, key_dim]
        glimpse_q = self.q_graph + self.q_source + self.q_target + self.q_first + q_last

        if self.n_decoding_neighbors is not None:
            D = self.coordinates.size(-1)
            K = torch.count_nonzero(self.group_ninf_mask[0, 0] == 0.0).item()
            K = min(self.n_decoding_neighbors, K)
            last_node_coordinate = self.coordinates.gather(dim=1, index=last_node.unsqueeze(-1).expand(B, G, D))
            distances = torch.cdist(last_node_coordinate, self.coordinates)
            distances[self.group_ninf_mask == -np.inf] = np.inf
            indices = distances.topk(k=K, dim=-1, largest=False).indices
            glimpse_mask = torch.ones_like(self.group_ninf_mask) * (-np.inf)
            glimpse_mask.scatter_(dim=-1, index=indices, src=torch.zeros_like(glimpse_mask))
        else:
            glimpse_mask = self.group_ninf_mask

        attn_out = multi_head_attention(q=glimpse_q, k=self.glimpse_k, v=self.glimpse_v, mask=glimpse_mask)

        # mha_out.shape = [B, G, H]
        # score.shape = [B, G, N]
        final_q = self.multi_head_combine(attn_out)

        cal_probs = Compatibility(embed_dim = H, n_heads = self.n_heads, qkv_dim= None, key_dim = H , am_mode = False)
        probs = cal_probs(q = final_q, encoded_nodes = self.embeddings, mask = self.group_ninf_mask)

        
        assert (probs == probs).all(), "Probs should not contain any nans!"
        return probs




# =================================================================
# SHPP Env for low level training
# =================================================================
import torch


class SHPPEnv: 
    def __init__(self,x):
        self.x = x 
        self.batch_size = self.B = x.size(0)
        self.graph_size = self.N = x.size(1)
        self.node_dim = self.C = x.size(2)
        self.group_size = self.G = None
        self.device = x.device
        self.source = None
        self.target = None

    

    def reset(self, group_size, source, target):

        self.group_size = group_size
        self.target = target
        self.source = source

        ## group_state parameter
        self.selected_count = 0
        self.current_node = None 
        self.selected_node_list = torch.zeros(self.batch_size, self.group_size, 0, device=self.device).long()
        self.ninf_mask = torch.zeros(self.batch_size, self.group_size, self.graph_size, device=self.device)

        # distance between source and target
        self.fixed_edge_length = self._get_edge_length(source, target) 

        reward = None
        done = False

        return self, reward, done

    def step(self, selected_idx_mat):
        # move state
        self.state_move_to(selected_idx_mat)

        # returning values
        done = self.selected_count == (self.graph_size - 1)
        if done:
            reward = -self._get_path_distance() 
        else:
            reward = None
        return self, reward, done
    

    def state_move_to(self, selected_idx_mat):
        # selected_idx_mat.shape = [B, G]
        self.selected_count += 1
        self.__move_and_mask(selected_idx_mat)
        next_selected_idx_mat = self.__connect_source_target_city(selected_idx_mat)
        if (selected_idx_mat != next_selected_idx_mat).any():
            self.__move_and_mask(next_selected_idx_mat)

    def __move_and_mask(self, selected_idx_mat):
        self.current_node = selected_idx_mat
        self.selected_node_list = torch.cat((self.selected_node_list, selected_idx_mat[:, :, None]), dim=2)
        self.ninf_mask.scatter_(dim=-1, index=selected_idx_mat[:, :, None], value=-torch.inf)


    def __connect_source_target_city(self, selected_idx_mat):
        source_idx = torch.where(selected_idx_mat == self.source)
        target_idx = torch.where(selected_idx_mat == self.target)
        next_selected_idx_mat = selected_idx_mat.clone()
        next_selected_idx_mat[source_idx] = self.target[source_idx]
        next_selected_idx_mat[target_idx] = self.source[target_idx] 
        return next_selected_idx_mat

    def _get_edge_length(self, source, target):
        idx_shp = (self.batch_size, self.group_size, 1, self.node_dim)
        coord_shp = (self.batch_size, self.group_size, self.graph_size, self.node_dim)
        source_idx = source[..., None, None].expand(*idx_shp)
        target_idx = target[..., None, None].expand(*idx_shp)
        fixed_edge_idx = torch.cat([source_idx, target_idx], dim=2)
        seq_expanded = self.x[:, None, :, :].expand(*coord_shp)
        ordered_seq = seq_expanded.gather(dim=2, index=fixed_edge_idx)
        rolled_seq = ordered_seq.roll(dims=2, shifts=-1)
        delta = (ordered_seq - rolled_seq)[:, :, :-1, :]
        edge_length = (delta**2).sum(3).sqrt().sum(2)
        return edge_length

    def _get_path_distance(self) -> torch.Tensor:
        # selected_node_list.shape = [B, G, selected_count]
        interval = (torch.tensor([-1], device=self.x.device).long().expand(self.B, self.group_size))
        selected_node_list = torch.cat((self.selected_node_list, interval[:, :, None]),dim=2,).flatten()
        unique_selected_node_list = selected_node_list.unique_consecutive() 

        unique_selected_node_list = unique_selected_node_list.view([self.B, self.group_size, -1])[..., :-1]
        shp = (self.B, self.group_size, self.N, self.C)
        gathering_index = unique_selected_node_list.unsqueeze(3).expand(*shp)
        seq_expanded = self.x[:, None, :, :].expand(*shp)
        ordered_seq = seq_expanded.gather(dim=2, index=gathering_index)
        rolled_seq = ordered_seq.roll(dims=2, shifts=-1)
        delta = ordered_seq - rolled_seq
        tour_distances = (delta**2).sum(3).sqrt().sum(2)
        # minus the length of the fixed edge
        path_distances = tour_distances - self.fixed_edge_length
        return path_distances
    







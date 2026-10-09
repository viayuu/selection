"""R46A: source-view interaction first, instance-conditioned solver aggregation last."""

import torch
from torch import nn

from code.unified_selector.registry import GLOBAL_SOLVERS
from .V4Model import InstanceEncoder, MultiHeadAttention, _masked_mean, _masked_max
from .dual_stream import JointEncoderLayer
from .pairwise_selector import ScoreDifferenceSelector
from .solver_code_encoder import identity_digest
from .solver_code_tokens import DynamicSourcePool, SolverCodeTokens


class SourceTokenSelector(ScoreDifferenceSelector):
    def __init__(self, source, **params):
        # Do not construct the old solver encoder: there is no hidden ID/34-bit branch.
        nn.Module.__init__(self)
        self.params = params
        d = params['embedding_dim']
        self.instance_encoder = InstanceEncoder(**dict(params, node_only=True))
        self.solver_encoder = SolverCodeTokens(**source, **params)
        self.joint_layers = nn.ModuleList(JointEncoderLayer(**params) for _ in range(params['joint_layer_num']))
        self.pool = nn.Sequential(nn.Linear(2*d, d), nn.GELU(), nn.Linear(d, d))
        self.source_pool = DynamicSourcePool(d)
        self.decoder = MultiHeadAttention(d, params['head_num'], params['qkv_dim'], params['dropout'], params.get('sdpa', False))
        self.score_head = nn.Sequential(nn.Linear(3*d, params['head_hidden_dim']), nn.GELU(),
                                        nn.Dropout(params['dropout']), nn.Linear(params['head_hidden_dim'], 1))

    def encode_source_state(self, batch):
        nodes, node_mask = self.instance_encoder(batch)
        ids = batch['pool_ids'].to(nodes.device)
        if ids.ndim == 1:
            ids = ids[None].expand(nodes.shape[0], -1)
        mask = batch.get('solver_mask', torch.ones_like(ids, dtype=torch.bool)).to(nodes.device).bool()
        if ids.shape != mask.shape or not mask.any(-1).all():
            raise ValueError('Each instance needs at least one valid, aligned solver')
        ids = ids.masked_fill(~mask, 0)
        codes, token_mask = self.solver_encoder(ids, mask)
        shape = codes.shape
        codes, flat_mask = codes.flatten(1, 2), token_mask.flatten(1, 2)
        for layer in self.joint_layers:
            nodes, codes = layer(nodes, codes, node_mask, flat_mask)
        h = self.pool(torch.cat((_masked_mean(nodes, node_mask), _masked_max(nodes, node_mask)), -1))
        return nodes, codes.reshape(shape), h, node_mask, mask, token_mask

    def encode_state(self, batch):
        nodes, codes, h, node_mask, solver_mask, token_mask = self.encode_source_state(batch)
        solvers, _ = self.source_pool(h, codes, token_mask)
        return nodes, solvers, h, node_mask, solver_mask


def make_r46_model(params, bundle=None, state_dict=None):
    if params.get('solver_representation') != 'source_tokens' or params.get('pair_mode') != 'score_difference':
        raise ValueError('R46A uses source tokens and the unchanged score-difference/maximin decision')
    fields = ('code_vectors', 'role_mask', 'chunk_vectors', 'chunk_indices')
    if state_dict is None:
        from .solver_code_embeddings import validate_bundle
        validate_bundle(bundle)
        source = dict(vectors=bundle['vectors'], **{k: bundle[k] for k in fields[1:]})
    else:
        if not torch.equal(state_dict['solver_encoder.solver_identity'].cpu(), identity_digest()):
            raise ValueError('R46 solver identity/order differs from the current registry')
        source = {('vectors' if k == 'code_vectors' else k): state_dict['solver_encoder.' + k].cpu() for k in fields}
    model = SourceTokenSelector(source, **params)
    if state_dict is not None:
        model.load_state_dict(state_dict, strict=True)
    return model


def load_r46_checkpoint(path, device='cpu'):
    from .solver_code_embeddings import validate_bundle
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    config, state = checkpoint['args'], checkpoint['model']
    if config['solver_names'] != GLOBAL_SOLVERS or config['group'] != 'A':
        raise ValueError('Only the correctly mapped R46A checkpoint is supported')
    fixed = {('vectors' if k == 'code_vectors' else k): state['solver_encoder.' + k]
             for k in ('code_vectors', 'role_mask', 'chunk_vectors', 'chunk_indices')}
    validate_bundle(dict(config['embedding_provenance'], **fixed))
    return make_r46_model(config['model_params'], state_dict=state).to(device).eval(), checkpoint

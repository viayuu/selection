"""Trainable, encoder-only ReLD-MTL transfer for full-pool solver selection."""

import copy
import hashlib
import json
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from code.unified_selector.registry import PROBLEMS
from .pairwise_selector import ScoreDifferenceSelector


DIRECTORY = Path(__file__).resolve().parent
R43_ARGS = DIRECTORY / 'runs/R43_pairwise_selection/score_difference_seed2/args.json'
R50_PROVENANCE = DIRECTORY / 'runs/R50_pretrained_solver_encoder/pretraining_provenance.json'
NATIVE_CONFIG = dict(embedding_dim=128, encoder_layer_num=6, head_num=8, qkv_dim=16,
                     ff_hidden_dim=512, norm='none', norm_loc='norm_last')
NATIVE_PREFIX = 'instance_encoder.native_encoder.'
TRANSFER_PROBLEMS = tuple(p for p in PROBLEMS if p not in ('TSP', 'ATSP'))


def _file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _check_native_config(config):
    if any(config.get(key) != value for key, value in NATIVE_CONFIG.items()):
        raise ValueError('R59 requires the exact R50 dense MTL encoder configuration, including norm=none')


class _NativeResidual(nn.Module):
    """The pinned native Add_And_Normalization_Module has norm=None."""

    def forward(self, first, second):
        return first + second


class _NativeFeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.W1 = nn.Linear(config['embedding_dim'], config['ff_hidden_dim'])
        self.W2 = nn.Linear(config['ff_hidden_dim'], config['embedding_dim'])

    def forward(self, value):
        return self.W2(F.relu(self.W1(value)))


class _NativeEncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.head_num = config['head_num']
        self.qkv_dim = config['qkv_dim']
        d, width = config['embedding_dim'], self.head_num * self.qkv_dim
        self.Wq = nn.Linear(d, width, bias=False)
        self.Wk = nn.Linear(d, width, bias=False)
        self.Wv = nn.Linear(d, width, bias=False)
        self.multi_head_combine = nn.Linear(width, d)
        self.addAndNormalization1 = _NativeResidual()
        self.feedForward = _NativeFeedForward(config)
        self.addAndNormalization2 = _NativeResidual()

    def forward(self, nodes, mask):
        b, n, _ = nodes.shape
        def heads(projection):
            return projection(nodes).reshape(b, n, self.head_num, self.qkv_dim).transpose(1, 2)
        q, k, v = heads(self.Wq), heads(self.Wk), heads(self.Wv)
        if nodes.is_cuda:
            # Preserve native attention semantics without retaining dense score/softmax tapes on CUDA.
            values = F.scaled_dot_product_attention(q, k, v, attn_mask=mask[:, None, None, :],
                                                    dropout_p=0., is_causal=False)
        else:
            scores = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(self.qkv_dim)
            scores = scores.masked_fill(~mask[:, None, None, :], float('-inf'))
            values = torch.matmul(scores.softmax(-1), v)
        values = values.transpose(1, 2).reshape(b, n, self.head_num * self.qkv_dim)
        nodes = self.addAndNormalization1(nodes, self.multi_head_combine(values))
        nodes = self.addAndNormalization2(nodes, self.feedForward(nodes))
        return nodes.masked_fill(~mask[..., None], 0.)


class PaddingAwareMTLEncoder(nn.Module):
    """Native state-dict names and FP32 math, with invalid keys/outputs excluded."""

    def __init__(self, **config):
        super().__init__()
        _check_native_config(config)
        self.model_params = copy.deepcopy(config)
        self.embedding_depot = nn.Linear(2, config['embedding_dim'])
        self.embedding_node = nn.Linear(5, config['embedding_dim'])
        self.layers = nn.ModuleList(_NativeEncoderLayer(config) for _ in range(config['encoder_layer_num']))

    def forward(self, depot_xy, customer_features, node_mask=None):
        if (depot_xy.ndim != 3 or depot_xy.shape[1:] != (1, 2) or
                customer_features.ndim != 3 or customer_features.size(-1) != 5 or
                depot_xy.size(0) != customer_features.size(0)):
            raise ValueError('Native inputs must be depot[B,1,2] and customers[B,N,5]')
        b, n, _ = customer_features.shape
        if node_mask is None:
            node_mask = torch.ones(b, n + 1, dtype=torch.bool, device=depot_xy.device)
        if (node_mask.shape != (b, n + 1) or node_mask.dtype != torch.bool or
                not node_mask[:, 0].all() or not node_mask[:, 1:].any(-1).all()):
            raise ValueError('Native mask requires a valid depot0 and at least one customer per graph')
        customer_features = customer_features.masked_fill(~node_mask[:, 1:, None], 0.)
        nodes = torch.cat((self.embedding_depot(depot_xy), self.embedding_node(customer_features)), 1)
        nodes = nodes.masked_fill(~node_mask[..., None], 0.)
        for layer in self.layers:
            nodes = layer(nodes, node_mask)
        return nodes


def native_inputs(batch):
    """Unified fields: xy, signed demand, depot, limit, service, TW start/end."""
    node, mask = batch['node'], batch['node_mask']
    if node.ndim != 3 or node.size(-1) not in (4, 8) or mask.shape != node.shape[:2]:
        raise ValueError('Routing transfer expects unified 4D/8D nodes and a matching mask')
    if mask.dtype != torch.bool or not mask[:, 0].all():
        raise ValueError('Routing transfer requires a valid depot at node0')
    if node.size(-1) == 4:
        node = F.pad(node, (0, 4))
    node = node.masked_fill(~mask[..., None], 0.)
    if (not (node[:, 0, 3] == 1).all() or
            not (node[:, 1:, 3].masked_fill(~mask[:, 1:], 0.) == 0).all()):
        raise ValueError('Unified routing input must have exactly one depot at node0')
    tw = node[:, 1:, 6:8]
    has_tw = batch['cbits'][:, 4].to(device=node.device, dtype=torch.bool)
    tw = torch.where(has_tw[:, None, None], tw, torch.zeros_like(tw))
    customers = torch.cat((node[:, 1:, :3], tw), -1)
    return node[:, :1, :2], customers, mask


class TransferInstanceEncoder(nn.Module):
    def __init__(self, legacy_encoder, **params):
        super().__init__()
        self.legacy_encoder = legacy_encoder
        self.native_encoder = PaddingAwareMTLEncoder(**params['native_encoder_params'])
        d = params['embedding_dim']
        self.fusion = nn.Sequential(nn.Linear(3 * d, d), nn.GELU(), nn.LayerNorm(d))

    @property
    def geometry_residual(self):
        return self.legacy_encoder.geometry_residual

    def forward(self, batch):
        pid = int(batch['problem_id'])
        if not 0 <= pid < len(PROBLEMS):
            raise ValueError('Unknown problem_id')
        if PROBLEMS[pid] not in TRANSFER_PROBLEMS:
            return self.legacy_encoder(batch)
        if batch['kind'] != 'coord':
            raise ValueError('Native routing transfer requires coordinate inputs')
        depot, customers, mask = native_inputs(batch)
        node = self.legacy_encoder._pad_node(batch['node']).masked_fill(~mask[..., None], 0.)
        stats = self.legacy_encoder._coord_stats(node, mask)
        condition, _ = self.legacy_encoder.condition_encoder(batch, stats, kind_id=0)
        raw = self.legacy_encoder.coord_proj(node)
        if hasattr(self.legacy_encoder, 'geometry_residual'):
            raw = raw + self.geometry_residual(node, mask, batch['n'], batch.get('node_geom'))
        native = self.native_encoder(depot, customers, mask)
        nodes = self.fusion(torch.cat((native, raw, condition[:, None].expand_as(raw)), -1))
        return nodes.masked_fill(~mask[..., None], 0.), mask


class RoutingTransferSelector(ScoreDifferenceSelector):
    def __init__(self, **params):
        super().__init__(**params)
        self.instance_encoder = TransferInstanceEncoder(self.instance_encoder, **params)


def make_r59_model(params, state_dict=None):
    """Offline architecture factory: never read native source, weights or config files."""
    if (params.get('architecture') != 'r59_transfer' or params.get('transfer_arm') not in ('A', 'B') or
            params.get('pair_mode') != 'score_difference' or
            params.get('solver_representation') != 'handcrafted' or params.get('embedding_dim') != 128):
        raise ValueError('R59 requires full-pool R45A scoring with the fixed128D transfer architecture')
    _check_native_config(params['native_encoder_params'])
    model = RoutingTransferSelector(**copy.deepcopy(params))
    if state_dict is not None:
        model.load_state_dict(state_dict, strict=True)
    return model


def native_parameters(model):
    """Both arms use LR3e-5 for these parameters, LR1e-4 for the complement."""
    return model.instance_encoder.native_encoder.parameters()


def _provenance():
    recorded = json.loads(R50_PROVENANCE.read_text())
    entry = recorded['solvers']['RELD_MTL']
    config = {key: entry['runtime_params_contract'][key] for key in NATIVE_CONFIG}
    _check_native_config(config)
    source = next(path for path in recorded['source_files'] if Path(path).name == 'MTLModel.py')
    if _file_hash(source) != recorded['source_files'][source]:
        raise ValueError('Native MTL source differs from the pinned R50 source')
    if _file_hash(entry['checkpoint']['path']) != entry['checkpoint']['sha256']:
        raise ValueError('Native MTL weight differs from the pinned R50 checkpoint')
    return dict(checkpoint=copy.deepcopy(entry['checkpoint']), source=dict(path=source,
        sha256=recorded['source_files'][source]), r50_provenance=dict(path=str(R50_PROVENANCE),
        sha256=_file_hash(R50_PROVENANCE)), epoch=5000, problem='Train_ALL',
        config=config, limitations=copy.deepcopy(recorded['limitations']),
        legacy_backhaul_warning='Legacy signed-load labels are not a unified classic-backhaul benchmark')


def initialize(arm, geometry, device='cpu', seed=2):
    if arm not in ('A', 'B'):
        raise ValueError('R59 arms are A=random and B=pretrained')
    provenance = _provenance()
    params = copy.deepcopy(json.loads(R43_ARGS.read_text())['model_params'])
    params.update(architecture='r59_transfer', pair_mode='score_difference', solver_representation='handcrafted',
        sdpa=str(device).startswith('cuda'), transfer_arm=arm, native_encoder_params=provenance['config'],
        native_provenance=provenance, native_checkpoint=provenance['checkpoint'],
        native_pretrained_loaded=arm == 'B', initialization_seed=seed,
        native_attention='CUDA SDPA, boolean padding mask, default1/sqrt(qkv_dim), dropout0; manual CPU FP32 parity',
        input_fusion='Linear([native128, raw+geometry128, condition128],128) -> GELU -> LayerNorm; no gate')
    # Only CPU initialization uses RNG here; copying B weights cannot perturb any common initial parameter.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        model = make_r59_model(params)
    if arm == 'B':
        checkpoint = torch.load(provenance['checkpoint']['path'], map_location='cpu', weights_only=False)
        if (checkpoint.get('epoch'), checkpoint.get('problem')) != (5000, 'Train_ALL'):
            raise ValueError('Unexpected R50 native checkpoint identity')
        encoder = {key[len('encoder.'):]: value for key, value in checkpoint['model_state_dict'].items()
                   if key.startswith('encoder.')}
        model.instance_encoder.native_encoder.load_state_dict(encoder, strict=True)
    if hasattr(model.instance_encoder, 'geometry_residual'):
        branch = model.instance_encoder.geometry_residual
        mean, std = torch.as_tensor(geometry['mean'], dtype=torch.float32), torch.as_tensor(geometry['std'], dtype=torch.float32)
        if (mean.shape != branch.mean.shape or std.shape != branch.std.shape or
                not torch.isfinite(mean).all() or not torch.isfinite(std).all() or not (std > 0).all()):
            raise ValueError('Geometry statistics must be finite, correctly sized training-only means/positive stds')
        with torch.no_grad():
            branch.mean.copy_(mean)
            branch.std.copy_(std)
    return model.to(device), params

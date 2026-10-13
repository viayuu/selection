"""R64: the R45A dual stream with its native utility classifier."""

import torch
import torch.nn.functional as F

from .dual_stream import DualStreamSelector
from .pairwise_objective import inputs_only
from .r42_experiment import symmetric_kl
from .r45_experiment import parameters
from .train import set_seed


INFERENCE_RULE = 'FP32 eval; full legal pool; highest direct utility; exact ties by ascending global solver ID'


class NativeUtilitySelector(DualStreamSelector):
    def forward(self, batch):
        features, mask = self.encode_pairs(inputs_only(batch))
        utility = self.score_head(features).squeeze(-1).float()
        return dict(logits=utility.masked_fill(~mask, float('-inf')), solver_mask=mask)


def initialize(geometry, device='cpu', seed=2):
    set_seed(seed)
    params = dict(parameters('A', sdpa=str(device).startswith('cuda')),
                  architecture='dual_stream_native_utility', inference_rule=INFERENCE_RULE)
    required = dict(embedding_dim=128, encoder_layer_num=4, joint_layer_num=2,
                    head_num=4, ff_hidden_dim=512, head_hidden_dim=256, dropout=.1,
                    solver_representation='handcrafted', local_geometry=True)
    if any(params.get(k) != v for k, v in required.items()) or params['solver_feature_spec']['feature_dim'] != 34:
        raise ValueError('R64 requires the adopted complete R45A handcrafted34+ID backbone')
    model = NativeUtilitySelector(**params).to(device)
    with torch.no_grad():
        branch = model.instance_encoder.geometry_residual
        branch.mean.copy_(torch.tensor(geometry['mean'], device=device))
        branch.std.copy_(torch.tensor(geometry['std'], device=device))
    return model, params


def classification_objective(first, second, winner, consistency_weight):
    mask = first['solver_mask']
    if not torch.equal(mask, second['solver_mask']) or not mask.gather(1, winner[:, None]).all():
        raise ValueError('Native winners must belong to the same legal pool in both forwards')
    ce = .5 * (F.cross_entropy(first['logits'].float(), winner)
               + F.cross_entropy(second['logits'].float(), winner))
    kl = symmetric_kl(first['logits'], second['logits'], mask)
    return ce + consistency_weight * kl, dict(ce=ce, kl=kl)

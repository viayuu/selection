"""Frozen saved R65B x plus one fresh native-utility correction member."""

from pathlib import Path

import numpy as np
import torch

from code.unified_selector.registry import GLOBAL_SOLVERS, POOLS, PROBLEMS, problem_to_pool_mask
from .multitask_probe import file_hash, state_hash
from .r64_model import NativeUtilitySelector
from .r66_objective import composite_output, training_output
from .train import set_seed


WORKSPACE = Path(__file__).resolve().parents[2]
BASE_DIRECTORY = WORKSPACE/'code/V4/runs/R65_schedule_free_full18/B_schedule_free_seed2'
BASE_PATH = BASE_DIRECTORY/'best.pt'
BASE_SHA256 = 'd880cb2ed4605f410742b4c37945494323213168af95d02b22794d232b9e7ad8'
MEMBER_SEED = 660002
INFERENCE_RULE = ('FP32 eval; live frozen R65B40 saved x plus live R66 member averaged x; '
                  'coefficient1; full legal pool; highest summed direct utility; '
                  'exact ties by ascending global solver ID; one final solver')
INPUT_KEYS = frozenset(('problem_id', 'kind', 'node', 'node_mask', 'd_in', 'matrix',
                        'cbits', 'problem_desc', 'coord_dist', 'pool_ids', 'mask', 'n',
                        'node_geom', 'solver_mask'))
CACHE_KEYS = ('logits', 'indices', 'pool_ids', 'pool')
CACHE_INPUT = 'r66_base_logits'


def model_inputs(batch):
    return {key: value for key, value in batch.items() if key in INPUT_KEYS}


def load_base_checkpoint():
    if file_hash(BASE_PATH) != BASE_SHA256:
        raise ValueError('The adopted immutable R65B40 saved-x checkpoint changed')
    checkpoint = torch.load(BASE_PATH, map_location='cpu', weights_only=False)
    if (checkpoint['epoch'] != 39 or checkpoint['successful_updates'] != 9600
            or checkpoint['parameter_mode'] != 'averaged x'):
        raise ValueError('R66 requires exactly the completed R65B40 averaged-x model')
    return checkpoint


def fresh_member(params, geometry):
    set_seed(MEMBER_SEED)
    member = NativeUtilitySelector(**params)
    with torch.no_grad():
        member.instance_encoder.geometry_residual.mean.copy_(torch.tensor(geometry['mean']))
        member.instance_encoder.geometry_residual.std.copy_(torch.tensor(geometry['std']))
        member.score_head[-1].weight.zero_()
        member.score_head[-1].bias.zero_()
    return member


def shared_initialization(params, geometry):
    first = fresh_member(params, geometry)
    second = NativeUtilitySelector(**params)
    second.load_state_dict(first.state_dict(), strict=True)
    for name, parameter in first.named_parameters():
        torch.testing.assert_close(parameter, dict(second.named_parameters())[name], rtol=0, atol=0)
    for member in (first, second):
        assert not member.score_head[-1].weight.count_nonzero()
        assert not member.score_head[-1].bias.count_nonzero()
    state = {key: value.detach().cpu().clone() for key, value in first.state_dict().items()}
    return state, dict(member_seed=MEMBER_SEED, explicit_copy_and_comparison=True,
        matching_trainable_parameters=True, zero_scalar_weight_and_bias=True,
        shared_state_sha256=state_hash(state), fresh=True, inherited_member_weights=False)


class AdditiveSelector(torch.nn.Module):
    def __init__(self, base, member, arm, use_cached_base=False):
        super().__init__()
        if arm not in ('A', 'B'):
            raise ValueError('Unknown R66 arm')
        self.base, self.member, self.arm = base, member, arm
        self.use_cached_base = use_cached_base
        self.base.requires_grad_(False)
        self.train(False)

    def train(self, mode=True):
        super().train(mode)
        self.base.eval()
        self.base.requires_grad_(False)
        return self

    def _base_logits(self, batch, clean):
        if self.use_cached_base:
            if CACHE_INPUT not in batch:
                raise ValueError('TRAIN/VAL cached mode requires aligned original-row base scores')
            return batch[CACHE_INPUT].detach().float()
        with torch.no_grad():
            return self.base(clean)['logits'].detach().float()

    def forward(self, batch):
        clean = model_inputs(batch)
        member = self.member(clean)
        if self.training and self.arm == 'A':
            return member
        base = self._base_logits(batch, clean)
        return training_output(member, base, self.arm) if self.training else composite_output(member, base)

    def deployment_forward(self, batch):
        if self.training:
            raise ValueError('Deployment requires eval-mode member x')
        clean = model_inputs(batch)
        with torch.no_grad():
            base = self.base(clean)['logits']
        return composite_output(self.member(clean), base)


def build_selector(params, base_state, member_state, arm, device='cpu', cached=False):
    base, member = NativeUtilitySelector(**params), NativeUtilitySelector(**params)
    base.load_state_dict(base_state, strict=True)
    member.load_state_dict(member_state, strict=True)
    return AdditiveSelector(base, member, arm, use_cached_base=cached).to(device)


def load_deployment_bundle(path, device='cpu'):
    bundle = torch.load(path, map_location='cpu', weights_only=False)
    if (bundle['inference_rule'] != INFERENCE_RULE or bundle['coefficient'] != 1.
            or bundle['parameter_mode'] != 'averaged x'
            or bundle['base_checkpoint_sha256'] != BASE_SHA256
            or state_hash(bundle['base']) != bundle['base_state_sha256']
            or bundle['cache_allowed_at_deployment'] is not False):
        raise ValueError('Deployment requires the complete frozen coefficient1 two-model x bundle')
    model = build_selector(bundle['args']['model_params'], bundle['base'], bundle['member'],
                           bundle['args']['group'], device, cached=False)
    return model.eval()


def load_base_caches(expected=None, device='cpu'):
    caches, receipt = {}, {}
    for split, size in (('train', 10000), ('val', 1000)):
        caches[split], receipt[split] = {}, {}
        for task in PROBLEMS:
            path = BASE_DIRECTORY/(split+'_predictions')/'best'/f'{task}.npz'
            digest = file_hash(path)
            if expected is not None and digest != expected[split][task]['sha256']:
                raise ValueError('Frozen TRAIN/VAL base cache changed: '+str(path))
            # Access only these four arrays; archive supervision is never extracted.
            with np.load(path, allow_pickle=False) as archive:
                fields = {key: archive[key] for key in CACHE_KEYS}
            ids = np.asarray(problem_to_pool_mask(task)[0])
            if expected is None:
                np.testing.assert_array_equal(fields['indices'], np.arange(size))
                np.testing.assert_array_equal(fields['pool_ids'], ids)
                np.testing.assert_array_equal(fields['pool'], np.asarray(POOLS[task]))
                np.testing.assert_array_equal(fields['pool'], np.asarray(GLOBAL_SOLVERS)[ids])
                if fields['logits'].dtype != np.float32 or not np.isfinite(fields['logits']).all():
                    raise ValueError('Base utilities must be finite saved FP32 full-pool scores')
            if fields['logits'].shape != (size, len(ids)):
                raise ValueError('Base cache must cover every original row and complete legal pool')
            caches[split][task] = torch.from_numpy(fields['logits'].copy()).to(device)
            receipt[split][task] = dict(path=str(path.relative_to(WORKSPACE)), sha256=digest,
                rows=size, pool_ids=ids.tolist(), pool_names=fields['pool'].tolist(),
                original_row_names='str(original zero-based dataset/label row index)',
                extracted_fields=list(CACHE_KEYS), forward_fields=['logits'],
                supervision_extracted=False)
    return caches, receipt

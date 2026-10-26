"""Matched full18 elimination processes with exact champion marginalization.

A computes a fixed full-pool elimination preference and only renormalizes it
inside remaining sets. B recomputes its interaction with each remaining set.
Both use identical parameters, arithmetic shapes and all-path probability DP.
"""

import copy

import torch
from torch import nn

from .pairwise_selector import ScoreDifferenceSelector
from .r77_data import parameter_hash
from .r88_model import prepare_role_inputs
from .r116_subset import champion_log_probs, subset_structure


ARMS = ("A_static", "B_subset")
BRANCH_SEED = 116002
SPEC = dict(experiment="R116", schema_version=1, role_arm="B_execution_anchor",
    node_layers=4, joint_layers=2, embedding_dim=128, candidate_features="original decoded 3d",
    feature_width=64, set_width=32, maximum_candidates=10,
    static="full-set mean and full-set size reused in every subset",
    dynamic="remaining-set mean and size recomputed before nonlinear candidate interaction",
    compute_match="same all-subset head tensor shapes in both arms; not greedy vs DP",
    deployment="exact log-space sequential-elimination champion marginal, then argmax",
    no_scalar_score_bypass=True, current_query_labels_in_forward=False,
    new_labels=False, branch_seed=BRANCH_SEED)


class EliminationSelector(ScoreDifferenceSelector):
    def __init__(self, arm, **params):
        if arm not in ARMS:
            raise ValueError("Unknown R116 matched arm")
        if params.get("r116_arm", arm) != arm or params.get("r116_spec", SPEC) != SPEC:
            raise ValueError("R116 architecture/arm differs from its saved configuration")
        params = dict(params, r116_arm=arm, r116_spec=copy.deepcopy(SPEC),
                      role_arm=SPEC["role_arm"], ignore_coord_dist=True,
                      selection_head="exact_elimination_champion_marginal")
        super().__init__(**params)
        del self.score_head
        self.arm = arm
        d, width, hidden = params["embedding_dim"], SPEC["feature_width"], SPEC["set_width"]
        # Shared fresh backbone is restored separately; new head has its own seed.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(BRANCH_SEED)
            self.elimination_features = nn.Sequential(nn.LayerNorm(3 * d), nn.Linear(3 * d, width),
                nn.GELU(), nn.Dropout(params["dropout"]))
            self.candidate_projection = nn.Linear(width, hidden)
            self.set_projection = nn.Linear(width, hidden)
            self.set_gate = nn.Linear(width, hidden)
            self.size_projection = nn.Linear(1, hidden, bias=False)
            self.elimination_out = nn.Linear(hidden, 1)

    def forward(self, batch):
        prepared = prepare_role_inputs(batch, SPEC["role_arm"])
        features, legal = self.encode_pairs(prepared)
        k = features.size(1)
        if not 1 <= k <= SPEC["maximum_candidates"]:
            raise ValueError("R116 finite DP supports the frozen pools of at most ten methods")
        if not bool(legal.all()):
            raise ValueError("R116 requires a task's predeclared homogeneous legal pool, not instance-wise pruning")
        feature = self.elimination_features(features)
        if k == 1:
            log_champion = feature.sum(-1) * 0
            subset_logits = feature.new_empty((len(feature), 0, 1))
        else:
            masks = subset_structure(k, device=feature.device)["masks"]
            membership = masks.to(feature.dtype)
            counts = membership.sum(-1)
            if self.arm == "A_static":
                # Use all subset rows, not a cheaper greedy/static-only code path.
                membership = torch.ones_like(membership)
                counts = torch.full_like(counts, k)
            context = torch.einsum("sk,bkd->bsd", membership, feature) / counts[None, :, None]
            size = counts.log()[:, None] / feature.new_tensor(float(k)).log()
            candidate = self.candidate_projection(feature)[:, None]
            interaction = candidate + self.set_projection(context)[:, :, None]
            interaction = interaction + candidate * torch.sigmoid(self.set_gate(context))[:, :, None]
            interaction = interaction + self.size_projection(size)[None, :, None]
            subset_logits = self.elimination_out(torch.tanh(interaction)).squeeze(-1).float()
            log_champion = champion_log_probs(subset_logits)
        margins = log_champion[:, :, None] - log_champion[:, None, :]
        return dict(logits=log_champion, utility=log_champion, pair_margin=margins,
                    selection_scores=log_champion, solver_mask=legal, subset_logits=subset_logits)


def make_model(params, arm, state_dict=None):
    model = EliminationSelector(arm, **params)
    if state_dict is not None:
        model.load_state_dict(state_dict, strict=True)
    return model


def initialize_shared(initial, geometry, arm, device="cpu"):
    if (initial.get("seed") != 2 or initial.get("initialization") != "scratch"
            or parameter_hash(initial["trainable"]) != initial.get("trainable_sha256")):
        raise ValueError("Only the frozen, never-trained R84 seed2 initialization is allowed")
    torch.manual_seed(2)
    params = dict(initial["model_params"])
    model = make_model(params, arm)
    reference, weights = initial["trainable"], dict(model.named_parameters())
    common = {name: tensor for name, tensor in weights.items()
              if name in reference and tensor.shape == reference[name].shape}
    with torch.no_grad():
        for name, tensor in common.items():
            tensor.copy_(reference[name])
        geometry_branch = model.instance_encoder.geometry_residual
        geometry_branch.mean.copy_(torch.as_tensor(geometry["mean"], dtype=geometry_branch.mean.dtype))
        geometry_branch.std.copy_(torch.as_tensor(geometry["std"], dtype=geometry_branch.std.dtype))
    receipt = dict(seed=2, initialization="scratch_seed2_matched_elimination",
        historical_checkpoint_loaded=False, fresh_reference_sha256=initial["trainable_sha256"],
        common_parameter_sha256=parameter_hash(common), trainable_sha256=parameter_hash(weights),
        parameter_layout_sha256=parameter_hash({name: torch.tensor(t.shape) for name, t in weights.items()}),
        common_parameter_names=sorted(common), new_parameter_names=sorted(set(weights) - set(common)),
        removed_parameter_names=sorted(set(reference) - set(common)),
        geometry_fitted_on="frozen R84 81000 TRAIN only")
    return model.to(device), copy.deepcopy(model.params), receipt

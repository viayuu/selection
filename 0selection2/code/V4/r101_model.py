"""Schema of the unchanged deployed student, not a new architecture.

Construction is delegated to R88's original role-aware selector by the runner.
Teacher arrays, a donor module, hint heads and alignment adapters are absent.
"""


SPEC = dict(r101_schema_version=1, backbone='original_R88B_R77_full18',
    architecture_unchanged=True, trainable_parameter_count=1842953,
    added_parameters=0, embedding_dim=128, encoder_layers=4, joint_layers=2,
    solver_features=34, solver_ID=True, physical_TSP_anchor='original_node0',
    selection_head='original_score_mlp_maximin', dropout=.1,
    transfer='only instance_encoder from this run fixed TRAIN hint endpoint',
    stageII_hint_loss=False, teacher_at_inference=False,
    learned_alignment_adapter=False, route_or_cost_auxiliary_head=False,
    original_raw_inputs_and_geometry_retained=True)

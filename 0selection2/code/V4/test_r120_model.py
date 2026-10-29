import copy

import pytest
import torch

from .r120_model import ALL_PROBLEMS, MultidomainSelector, default_config


def model():
    torch.manual_seed(2)
    config = default_config(['BQ', 'NR::first', 'NR::second'], ALL_PROBLEMS,
                            embedding_dim=32, qkv_dim=8, ff_hidden_dim=64,
                            head_hidden_dim=32, encoder_layer_num=1,
                            joint_layer_num=1, dropout=0.)
    return MultidomainSelector(config)


def raw_inputs(problem):
    if problem == 'MIS':
        return dict(num_nodes=torch.tensor([4]), edge_index=torch.tensor([[[0, 1], [1, 2]]]),
                    node_mask=torch.ones(1, 4, dtype=torch.bool), edge_mask=torch.ones(1, 2, dtype=torch.bool))
    if problem in ('JSSP', 'FJSP'):
        value = dict(num_jobs=torch.tensor([2]), num_machines=torch.tensor([2]),
                     job_ptr=torch.tensor([[0, 2, 4]]), operation_job=torch.tensor([[0, 0, 1, 1]]),
                     operation_index=torch.tensor([[0, 1, 0, 1]]),
                     precedence_edge_index=torch.tensor([[[0, 2], [1, 3]]]),
                     job_mask=torch.ones(1, 2, dtype=torch.bool),
                     machine_mask=torch.ones(1, 2, dtype=torch.bool),
                     operation_mask=torch.ones(1, 4, dtype=torch.bool),
                     precedence_mask=torch.ones(1, 2, dtype=torch.bool),
                     job_ptr_mask=torch.ones(1, 3, dtype=torch.bool))
        if problem == 'JSSP':
            value.update(duration=torch.tensor([[2, 3, 4, 5]]), machine=torch.tensor([[0, 1, 1, 0]]))
        else:
            value.update(processing_time=torch.tensor([[[2, -1], [3, 4], [-1, 4], [5, 6]]]))
            value['compatibility'] = value['processing_time'] > 0
        return value
    if problem == 'BPP':
        return dict(sizes=torch.tensor([[6, 5, 4, 3]]), capacity=torch.tensor([10]),
                    item_mask=torch.ones(1, 4, dtype=torch.bool))
    if problem == 'MKP':
        return dict(profits=torch.tensor([[6, 5, 4]]),
                    resources=torch.tensor([[[2, 1], [1, 3], [2, 2]]]),
                    capacities=torch.tensor([[4, 4]]), item_mask=torch.ones(1, 3, dtype=torch.bool),
                    resource_mask=torch.ones(1, 2, dtype=torch.bool))
    return dict(p=torch.tensor([[2, 4, 3]]), w=torch.tensor([[3, 1, 2]]), d=torch.tensor([[0, 4, 8]]),
                job_mask=torch.ones(1, 3, dtype=torch.bool))


@pytest.mark.parametrize('problem', ALL_PROBLEMS[-6:])
def test_all_six_raw_domains_reach_the_shared_decision_and_gradients(problem):
    selector = model()
    output = selector(dict(problem=problem, inputs=raw_inputs(problem), pool_ids=torch.tensor([1, 2])))
    assert output['logits'].shape == (1, 2)
    assert torch.isfinite(output['logits']).all()
    (output['utility'][0, 0] + output['advantage'][0, 1]).backward()
    assert selector.layers[0].source.weight.grad is not None
    assert selector.joint_layers[0].solver_update.attn.Wq.weight.grad is not None
    assert selector.score_head[-1].weight.grad.abs().sum() > 0


def test_candidate_order_equivariance_and_no_label_forward():
    selector = model().eval()
    batch = dict(problem='MIS', inputs=raw_inputs('MIS'), pool_ids=torch.tensor([1, 2]))
    normal = selector(batch)['utility']
    flipped = selector(dict(batch, pool_ids=torch.tensor([2, 1])))['utility']
    torch.testing.assert_close(normal, flipped.flip(-1), atol=2e-6, rtol=2e-5)
    with pytest.raises(ValueError, match='Labels'):
        selector(dict(batch, labels={'winner': 0}))


def test_graph_topology_is_retained_and_isolated_nodes_are_valid():
    selector = model().eval()
    raw = raw_inputs('MIS')
    first = selector(dict(problem='MIS', inputs=raw, pool_ids=torch.tensor([1, 2])))['utility']
    permutation = torch.tensor([2, 0, 3, 1])
    inverse = permutation.argsort()
    permuted = copy.deepcopy(raw)
    permuted['edge_index'] = inverse[raw['edge_index']]
    permuted['execution_rank'] = torch.arange(4)[permutation][None]
    second = selector(dict(problem='MIS', inputs=permuted, pool_ids=torch.tensor([1, 2])))['utility']
    torch.testing.assert_close(first, second, atol=2e-6, rtol=2e-5)
    graph = selector.frontend('MIS', raw)
    assert graph.mask.sum() == 5  # Four real vertices, including vertex 3, plus context.
    assert graph.source.numel() == 4


def test_fjsp_incompatible_machine_cannot_become_a_zero_time_edge():
    selector = model()
    graph = selector.frontend('FJSP', raw_inputs('FJSP'))
    # Two precedence arcs in both directions and six compatible pairs in both directions.
    assert graph.source.numel() == 16
    assert torch.isfinite(graph.edge_value).all()

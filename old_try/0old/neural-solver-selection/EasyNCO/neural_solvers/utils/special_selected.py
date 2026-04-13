import torch
from tensordict import TensorDict
import random
'''
This file contains the special node selection for the CO problems.
It is used to select the special node in the solving process. 
For example, in the CVRP, the special node is the depot node and first visited node expect the depot node.
And in the TSP, the special node is the first node.
'''

def special_selected(env_name: str, config: dict):
    """
    Get environment special selected node in decoding.
    The special selected node is used to select the special node in the solving process.

    Args:
        env: Environment or its name.
        config: A dictionary of configuration options for the environment.
    """
    node_selected = {
        "tsp": TSPSpecifialSelected,
        "cvrp": CVRPSpecifialSelected,
        'kp': KPSpecialSelected,
    }

    if env_name not in node_selected.keys():
        raise ValueError(
            f"Unknown environment name '{env_name}'. Available special selected nodes: {node_selected.keys()}"
        )

    return node_selected[env_name](**config)

def TSPSpecifialSelected(td: TensorDict, **kwargs):
    """
    Special node selection for the Traveling Salesman Problem (TSP).
    Select the first node in the TSP.
    """

    pomo_size = td.batch_size[1]
    batch_size = td.batch_size[0]
    first_mode = kwargs.get("first_mode", 'random')
    first_placeholder = kwargs.get("first_placeholder", False)

    if first_placeholder:
        first_mode = 'placeholder'

    if first_mode == 'random' and (td["first_node"] == -1).all():
        selected = torch.arange(pomo_size)[None, :].expand(batch_size, pomo_size)
        probs = torch.ones(size=(batch_size, pomo_size))
    else:
        selected = None
        probs = None
    return selected, probs


def CVRPSpecifialSelected(td: TensorDict, **kwargs):
    """
    Only for pomo_size > 1
    Special node selection for the Capacitated Vehicle Routing Problem (CVRP).
    """
    pomo_size = td.batch_size[1]
    batch_size = td.batch_size[0]


    if (td['next']['selected_count'] == 0).all():
        selected = torch.zeros(size=(batch_size, pomo_size), dtype=torch.long)
        probs = torch.ones(size=(batch_size, pomo_size))
        return selected, probs
    elif (td['next']['selected_count'] == 1).all() and pomo_size > 1:
        selected = torch.arange(start=1,end=pomo_size+1)[None, :].expand(batch_size, pomo_size)
        probs = torch.ones(size=(batch_size, pomo_size))
    else:
        selected = None
        probs = None
    return selected, probs

def KPSpecialSelected( **kwargs):
    selected = None
    probs = None

    return selected, probs
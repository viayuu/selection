from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class SolutionBatch:
    tour: torch.Tensor
    length: torch.Tensor

    @property
    def reward(self) -> torch.Tensor:
        return -self.length


def tsp_tour_length(coords: torch.Tensor, tour: torch.Tensor) -> torch.Tensor:
    ordered = coords.gather(1, tour.unsqueeze(-1).expand(-1, -1, coords.size(-1)))
    rolled = ordered.roll(shifts=-1, dims=1)
    seg = ((ordered - rolled) ** 2).sum(dim=-1).sqrt()
    return seg.sum(dim=1)


def tsp_succ_length(coords: torch.Tensor, succ: torch.Tensor) -> torch.Tensor:
    succ_xy = coords.gather(1, succ.unsqueeze(-1).expand(-1, -1, coords.size(-1)))
    seg = ((succ_xy - coords) ** 2).sum(dim=-1).sqrt()
    return seg.sum(dim=1)


def perm_to_succ(tour: torch.Tensor) -> torch.Tensor:
    nxt = tour.roll(shifts=-1, dims=1)
    succ = torch.empty_like(tour)
    succ.scatter_(1, tour, nxt)
    return succ


def succ_to_perm(succ: torch.Tensor, start_node: int = 0) -> torch.Tensor:
    batch_size, problem_size = succ.shape
    tour = torch.empty((batch_size, problem_size), dtype=torch.long, device=succ.device)
    cur = torch.full((batch_size,), int(start_node), dtype=torch.long, device=succ.device)
    for i in range(problem_size):
        tour[:, i] = cur
        cur = succ.gather(1, cur[:, None]).squeeze(1)
    return tour


def validate_tour(tour: torch.Tensor) -> torch.Tensor:
    n = tour.size(1)
    target = torch.arange(n, device=tour.device, dtype=tour.dtype).view(1, n)
    return (tour.sort(dim=1).values == target).all(dim=1)


def group_indices(actions: torch.Tensor, num_actions: int) -> list[torch.Tensor]:
    groups = []
    for action_id in range(num_actions):
        idx = (actions == action_id).nonzero(as_tuple=False).flatten()
        groups.append(idx)
    return groups


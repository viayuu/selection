"""New classical-backhaul decoding state, isolated from all legacy environments."""

from types import SimpleNamespace

import numpy as np
import torch

from .r58_scenario import CONTRACT, fields


class RoutingState:
    """Shared physical transition for fixed-weight ReLD/MTPOMO/MVMoE deployments.

    Decoder load now means remaining capacity in the active classical phase,
    not the old signed-load quantity. This is an adapted deployment, not a
    claim about the solver's original published performance.
    """

    def __init__(self, problem, items, device):
        self.problem, self.device = problem, torch.device(device)
        specs = [fields(problem, item) for item in items]
        self.b, self.n = len(items), specs[0]['n']
        if any(spec['n'] != self.n for spec in specs):
            raise ValueError('Decode batch must have one true size')
        self.p = self.n
        def tensor(key):
            return torch.as_tensor(np.stack([s[key] for s in specs]), dtype=torch.float32, device=self.device)
        self.xy, self.demand = tensor('xy'), tensor('demand')
        self.capacity = tensor('capacity')
        self.capacity_tolerance = CONTRACT['tolerance']['feasibility_abs'] / self.capacity[:, None, None]
        self.demand = self.demand / self.capacity[:, None]
        self.service, self.early, self.late = tensor('service'), tensor('start'), tensor('end')
        self.speed, self.limit = tensor('speed'), tensor('limit')
        self.opened, self.tw, self.backhaul = specs[0]['opened'], specs[0]['tw'], specs[0]['backhaul']
        self.batch = torch.arange(self.b, device=self.device)[:, None].expand(-1, self.p)
        self.starts = torch.arange(1, self.n + 1, device=self.device)[None].expand(self.b, -1)
        self.current = torch.zeros(self.b, self.p, dtype=torch.long, device=self.device)
        self.visited = torch.zeros(self.b, self.p, self.n + 1, dtype=torch.bool, device=self.device)
        self.delivered = torch.zeros(self.b, self.p, device=self.device)
        self.picked = self.delivered.clone()
        self.pickup_phase = torch.zeros_like(self.delivered, dtype=torch.bool)
        self.length, self.cost = self.delivered.clone(), self.delivered.clone()
        self.clock = self.early[:, :1].expand(-1, self.p).clone()
        self.finished = torch.zeros_like(self.pickup_phase)
        self.tour = torch.empty(self.b, self.p, 0, dtype=torch.long, device=self.device)
        self.count = 0
        self.refresh_mask()

    @property
    def load(self):
        return 1. - torch.where(self.pickup_phase, self.picked, self.delivered)

    @property
    def coords(self):
        return self.xy[self.batch, self.current]

    def refresh_mask(self):
        tol = CONTRACT['tolerance']['feasibility_abs']
        distance = (self.coords[:, :, None] - self.xy[:, None]).norm(dim=-1)
        back = (self.xy - self.xy[:, :1]).norm(dim=-1)
        positive, negative = self.demand.clamp_min(0), (-self.demand).clamp_min(0)
        allowed = ~self.visited
        allowed = allowed & (positive[:, None] + self.delivered[..., None] <= 1 + self.capacity_tolerance)
        allowed = allowed & (negative[:, None] + self.picked[..., None] <= 1 + self.capacity_tolerance)
        if self.backhaul:
            allowed = allowed & ~(self.pickup_phase[..., None] & (positive[:, None] > 0))
        route_length = self.length[..., None] + distance + (0 if self.opened else back[:, None])
        allowed = allowed & (route_length <= self.limit[:, None, None] + tol)
        if self.tw:
            arrival = torch.maximum(self.clock[..., None] + distance / self.speed[:, None, None], self.early[:, None])
            allowed = allowed & (arrival <= self.late[:, None] + tol)
            if not self.opened:
                return_time = arrival + self.service[:, None] + back[:, None] / self.speed[:, None, None]
                allowed = allowed & (return_time <= self.late[:, None, :1] + tol)
        allowed[:, :, 0] = self.current != 0
        allowed[:, :, 0] |= self.finished
        if ((~allowed.any(-1)) & ~self.finished).any():
            where = ((~allowed.any(-1)) & ~self.finished).nonzero().cpu().tolist()
            raise RuntimeError(f'No feasible action, including a fresh vehicle: {where[:5]}')
        self.mask = torch.where(allowed, 0., -torch.inf)

    def step(self, selected):
        selected = selected.to(device=self.device, dtype=torch.long)
        if selected.shape != self.current.shape:
            raise ValueError('Invalid action shape')
        if self.count == 0:
            if (selected != 0).any():
                raise ValueError('Mandatory initial depot action missing')
        elif not torch.isfinite(self.mask.gather(2, selected[..., None])).all():
            raise RuntimeError('Decoder chose a forbidden action under scenario_v2')
        if self.count > 2 * self.n + 2:
            raise RuntimeError('Unbounded decoding loop')
        demand = self.demand[self.batch, selected]
        depot = selected == 0
        travel = (self.coords - self.xy[self.batch, selected]).norm(dim=-1)
        self.cost += travel * (~depot if self.opened else 1)
        self.length += travel
        if self.tw:
            self.clock = torch.maximum(self.clock + travel / self.speed[:, None], self.early[self.batch, selected])
            self.clock += self.service[self.batch, selected]
        self.delivered += demand.clamp_min(0)
        self.picked += (-demand).clamp_min(0)
        self.pickup_phase |= demand < 0
        self.delivered[depot] = 0
        self.picked[depot] = 0
        self.pickup_phase[depot] = False
        self.length[depot] = 0
        self.clock = torch.where(depot, self.early[:, :1], self.clock)
        self.visited.scatter_(2, selected[..., None], True)
        self.current = selected
        self.tour = torch.cat((self.tour, selected[..., None]), dim=-1)
        self.count += 1
        self.finished |= self.visited[:, :, 1:].all(-1) & depot
        self.refresh_mask()
        return bool(self.finished.all())

    def reld_reset(self):
        # Non-TW neural inputs stay finite; feasibility uses separate infinite bounds.
        node_late = self.late[:, 1:] if self.tw else torch.zeros_like(self.late[:, 1:])
        return SimpleNamespace(depot_xy=self.xy[:, :1], node_xy=self.xy[:, 1:],
            node_demand=self.demand[:, 1:], node_service_time=self.service[:, 1:],
            node_tw_start=self.early[:, 1:], node_tw_end=node_late,
            prob_emb=torch.tensor([[1, self.opened, self.backhaul, 'L' in self.problem, self.tw]],
                                  device=self.device, dtype=torch.float32))

    def reld_step(self):
        return SimpleNamespace(BATCH_IDX=self.batch,
            POMO_IDX=torch.arange(self.p, device=self.device)[None].expand(self.b, -1),
            START_NODE=self.starts, PROBLEM=self.problem, selected_count=self.count,
            current_node=self.current if self.count else None, ninf_mask=self.mask,
            finished=self.finished, load=self.load, current_time=self.clock,
            length=self.length, open=torch.full_like(self.load, float(self.opened)), current_coord=self.coords)

    def native_reset(self):
        from tensordict import TensorDict
        late = self.late if self.tw else torch.zeros_like(self.late)
        return TensorDict(dict(depot_node_xy=self.xy, depot_node_demand=self.demand[..., None],
            depot_node_tw_start=self.early, depot_node_tw_end=late,
            depot_node_service_time=self.service,
            prob_emb=torch.tensor([[self.opened, self.backhaul, 'L' in self.problem, self.tw]],
                dtype=torch.float32, device=self.device).expand(self.b, -1)), batch_size=[self.b])

    def native_step(self):
        from tensordict import TensorDict
        return TensorDict(dict(action=self.current, current_node=self.current,
            start_node=self.starts, load=self.load, current_time=self.clock, length=self.length,
            open=torch.full_like(self.load, float(self.opened)), current_coord=self.coords,
            finished=self.finished, next=dict(ninf_mask=self.mask,
                selected_count=torch.full_like(self.current, self.count), selected_node_list=self.tour)),
            batch_size=[self.b, self.p])


def routefinder_mask(td):
    """Classical B mask with explicit inclusive TW bounds, also for MoSES."""
    tol = CONTRACT['tolerance']['feasibility_abs']
    xy, curr = td['locs'], td['current_node'].reshape(-1)
    coordinates = xy[torch.arange(len(xy), device=xy.device), curr]
    travel = (coordinates[:, None] - xy).norm(dim=-1)
    back = (xy - xy[:, :1]).norm(dim=-1)
    opened = td['open_route'].reshape(-1, 1)
    delivery, pickup = td['demand_linehaul'], td['demand_backhaul']
    cap_tol = tol / td['capacity_original']
    allowed = ~td['visited']
    allowed &= delivery + td['used_capacity_linehaul'] <= td['vehicle_capacity'] + cap_tol
    allowed &= pickup + td['used_capacity_backhaul'] <= td['vehicle_capacity'] + cap_tol
    allowed &= ~((td['used_capacity_backhaul'] > 0) & (delivery > 0))
    length = td['current_route_length'] + travel + back * ~opened
    allowed &= length <= td['distance_limit'] + tol
    arrival = torch.maximum(td['current_time'] + travel / td['speed'], td['time_windows'][..., 0])
    allowed &= arrival <= td['time_windows'][..., 1] + tol
    allowed &= opened | (arrival + td['service_time'] + back / td['speed'] <= td['time_windows'][:, :1, 1] + tol)
    allowed[:, 0] = curr != 0
    done = td['visited'][:, 1:].all(-1)
    allowed[:, 0] |= done
    if not allowed.any(-1).all():
        raise RuntimeError('No feasible RouteFinder action under scenario_v2')
    return allowed


def routefinder_depot_clock(td):
    at_depot = td['current_node'].reshape(-1, 1) == 0
    td['current_time'] = torch.where(at_depot, td['time_windows'][:, :1, 0], td['current_time'])
    td['action_mask'] = routefinder_mask(td)
    return td

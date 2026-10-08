"""
Unified Environment for all 107+ VRP variants
This environment can handle all problem types by using problem-specific masks
"""
from dataclasses import dataclass
import math
import torch
from typing import Dict, Optional, Any
from ProblemDef import get_random_problems
from multi_hot_set import get_problem_list


@dataclass
class Reset_State:
    problem_name: str = None
    problems: torch.Tensor = None
    # All problem data
    xy: torch.Tensor = None
    # shape: (batch, problem+depot_num, 2)
    demand: torch.Tensor = None
    # shape: (batch, problem+depot_num)
    dist: torch.Tensor = None
    # shape: (batch, problem+depot_num, problem+depot_num)
    prize: torch.Tensor = None
    # shape: (batch, problem+depot_num)
    penalty: torch.Tensor = None
    # shape: (batch, problem+depot_num)
    fake_prize: torch.Tensor = None
    # shape: (batch, problem+depot_num)
    service_time: torch.Tensor = None
    # shape: (batch, problem+depot_num)
    tw_start: torch.Tensor = None
    # shape: (batch, problem+depot_num)
    tw_end: torch.Tensor = None
    # shape: (batch, problem+depot_num)
    log_scale: float = None
    # scale factor for distance normalization
    route_limit: torch.Tensor = None
    # shape: (batch,)
    log_scale: float = None
    relation: torch.Tensor = None


@dataclass
class Step_State:
    batch_size: int = None
    pomo_size: int = None
    selected_count: int = None
    current_node: torch.Tensor = None
    # shape: (batch, pomo)
    current_time: torch.Tensor = None
    # shape: (batch, pomo)
    current_route: torch.Tensor = None
    # shape: (batch, pomo)
    load: torch.Tensor = None
    # shape: (batch, pomo)
    collected_prize: torch.Tensor = None
    # shape: (batch, pomo)
    ninf_mask: torch.Tensor = None
    # shape: (batch, pomo, problem+depot_num)
    finished: torch.Tensor = None
    # shape: (batch, pomo)
    START_NODE: torch.Tensor = None
    # shape: (batch, pomo) - for VRPB, nodes with positive demand
    depot_num: int = None
    tour_maxlength: torch.Tensor = None
    # shape: (batch, pomo) - for OP, maximum tour length


class UnifiedEnv:
    """
    Unified Environment that can handle all VRP variants:
    - TSP, ATSP, mTSP
    - CVRP, SDVRP, MDCVRP, ACVRP
    - CVRPTW, OVRP, OVRPTW, VRPB, VRPL, VRPBL, VRPLTW, VRPBTW, VRPBLTW, etc.
    - OP, PCTSP, SPCTSP
    - PDP
    - and all other 107+ variants
    """
    
    def __init__(self, problem_name: Optional[str] = None):
        """
        Args:
            problem_name: Name of the problem type (e.g., 'cvrp', 'tsp', 'cvrptw', etc.)
                         Can be set later in load_problems()
        """
        self.problem_name = problem_name
        
        # Problem configuration
        self.problem_size = None
        self.pomo_size = None
        self.batch_size = None
        self.device = None
        
        # Problem-specific parameters (will be set when problem_name is known)
        if problem_name is not None:
            self.depot_num = self._get_depot_num(problem_name)
            self.open_route = self._is_open_route(problem_name)
            self.has_capacity = self._has_capacity(problem_name)
            self.has_time_window = self._has_time_window(problem_name)
            self.has_prize = self._has_prize(problem_name)
            self.has_route_limit = self._has_route_limit(problem_name)
            self.has_backhaul = self._has_backhaul(problem_name)
            self.has_pickup_delivery = self._has_pickup_delivery(problem_name)
        else:
            self.depot_num = 1
            self.open_route = False
            self.has_capacity = False
            self.has_time_window = False
            self.has_prize = False
            self.has_route_limit = False
            self.has_backhaul = False
            self.has_pickup_delivery = False
        
        # Constants
        self.round_error_epsilon = 0.00001
        self.speed = 1.0
        
        # Static problem data (set in load_problems)
        self.xy = None
        self.demand = None
        self.dist = None
        self.prize = None
        self.penalty = None
        self.fake_prize = None
        self.service_time = None
        self.tw_start = None
        self.tw_end = None
        self.route_limit = None
        self.capacity = None
        
        # Dynamic state variables
        self.selected_count = None
        self.current_node = None
        self.current_time = None
        self.current_route = None
        self.load = None
        self.collected_prize = None
        self.selected_node_list = None
        self.visited_ninf_flag = None
        self.ninf_mask = None
        self.finished = None
        self.at_the_depot = None
        
        # PDP-specific variables
        self.pdp_mode = 0  # 0: no extra required, 1: FIFO, 2: LIFO
        self.to_deliver = None  # Boolean mask: True for pickup nodes, False for delivery nodes
        
        # For saved problems
        self.FLAG__use_saved_problems = False
        self.saved_data = None
        self.saved_index = None
        
        # States to return
        self.reset_state = Reset_State()
        self.step_state = Step_State()
    
    def _get_depot_num(self, problem_name: str) -> int:
        """Determine number of depots based on problem type"""
        if problem_name in ['tsp', 'atsp', 'mtsp']:
            return 0  # No depot
        elif problem_name.startswith('md') or problem_name.startswith('amd'):  # Multi-depot problems
            return 3  # Follow routefinder - 3 depots for multi-depot problems
        elif problem_name in ['pdp']:
            return 1
        else:
            return 1
    
    def _is_open_route(self, problem_name: str) -> bool:
        """Check if problem has open routes"""
        # OP and PCTSP: can end at any node (including depot)
        # OVRP variants: must end at depot but don't need to return
        return 'ovrp' in problem_name or problem_name in ['op', 'pctsp', 'spctsp']
    
    def _has_capacity(self, problem_name: str) -> bool:
        """Check if problem has capacity constraints"""
        # Most VRP variants have capacity constraints except pure TSP and OP
        if problem_name in ['tsp', 'atsp', 'op', 'pctsp', 'spctsp', 'mtsp']:
            return False
        # All other VRP variants (cvrp, ovrp, vrpb, etc.) have capacity
        if 'vrp' in problem_name or 'pdp' in problem_name:
            return True
        return False
    
    def _has_time_window(self, problem_name: str) -> bool:
        """Check if problem has time window constraints"""
        return 'tw' in problem_name
    
    def _has_prize(self, problem_name: str) -> bool:
        """Check if problem has prize collection"""
        return any(p in problem_name for p in ['op', 'pctsp', 'spctsp'])
    
    def _has_route_limit(self, problem_name: str) -> bool:
        """Check if problem has route length limit"""
        return 'l' in problem_name or 'op' in problem_name or 'pctsp' in problem_name
    
    def _has_backhaul(self, problem_name: str) -> bool:
        """Check if problem has backhaul"""
        return 'b' in problem_name and 'vrp' in problem_name
    
    def _has_pickup_delivery(self, problem_name: str) -> bool:
        """Check if problem has pickup-delivery pairs"""
        return 'pdp' in problem_name
    
    def input_saved_data(self, saved_data: Dict, device):
        """Input pre-saved problem data"""
        self.FLAG__use_saved_problems = True
        self.saved_data = saved_data
        self.saved_index = 0
        self.device = device
    
    def load_problems(self, 
                     batch_size: int, 
                     problem_size: int,
                     pomo_size: Optional[int] = None,
                     capacity: Optional[int] = None,
                     problem_name: Optional[str] = None,
                     lib_data: Optional[Dict] = None,
                     validation_data: Optional[Dict] = None,
                     aug_factor: int = 1,
                     device: Optional[torch.device] = None,
                     start: int = 0,
                     **kwargs):
        """
        Load problems for the environment
        
        Args:
            batch_size: Number of problem instances
            problem_size: Number of nodes (excluding depot)
            pomo_size: Number of parallel solutions (default: problem_size)
            capacity: Vehicle capacity (for capacity-constrained problems)
            problem_name: Name of the problem type
            lib_data: Pre-loaded library data
            validation_data: Validation dataset
            aug_factor: Data augmentation factor (e.g., 8 for 8-fold augmentation)
            device: PyTorch device
            start: Starting index for validation data
            **kwargs: Additional problem-specific parameters
        """
        if problem_name is not None:
            self.problem_name = problem_name
            # Update problem-specific features
            self.depot_num = self._get_depot_num(problem_name)
            self.open_route = self._is_open_route(problem_name)
            self.has_capacity = self._has_capacity(problem_name)
            self.has_time_window = self._has_time_window(problem_name)
            self.has_prize = self._has_prize(problem_name)
            self.has_route_limit = self._has_route_limit(problem_name)
            self.has_backhaul = self._has_backhaul(problem_name)
            self.has_pickup_delivery = self._has_pickup_delivery(problem_name)
        
        self.batch_size = batch_size
        self.problem_size = problem_size
        self.pomo_size = pomo_size if pomo_size is not None else problem_size
        
        # CRITICAL: For PDP, limit pomo_size to problem_size//2 (matches Original URS line 182-184)
        # This is because PDP has pickup-delivery pairs, so only half the nodes can be starting points
        if "pd" in self.problem_name and self.pomo_size > problem_size // 2:
            self.pomo_size = problem_size // 2
        
        # CRITICAL: For multi-depot problems, pomo_size = problem_size * depot_num (matches Original URS line 190)
        if "md" in self.problem_name:
            self.pomo_size = problem_size * self.depot_num

        
        self.device = device if device is not None else self.device
        self.capacity = capacity
        
        # Load problem data
        if lib_data is not None:
            # Load from library (e.g., TSPLIB, CVRPLIB)
            data = self._load_lib_data(lib_data)
        elif validation_data is not None:
            # Load from validation dataset
            data = self._load_validation_data(validation_data, start, batch_size)
        else:
            # Generate random problems
            if self.FLAG__use_saved_problems:
                data = self._load_saved_data(batch_size)
            else:
                # Add ATSP-specific parameters
                if self.problem_name == 'atsp' and 'problem_gen_params' not in kwargs:
                    kwargs['problem_gen_params'] = {
                        'int_min': 0,
                        'int_max': 1000*1000,
                        'scaler': 1000*1000
                    }
                
                # Don't pass depot_end to get_random_problems - will override later like original URS
                
                data = get_random_problems(
                    batch_size, 
                    problem_size, 
                    capacity if capacity is not None else 50,
                    self.problem_name,
                    depot_num=self.depot_num,
                    **kwargs
                )
        
        # Apply augmentation
        if aug_factor > 1:
            data = self._augment_data(data, aug_factor)
            self.batch_size = self.batch_size * aug_factor
        
        # Store problem data
        self._store_problem_data(data)
        
        # Override depot tw_end for time window problems (matching original URS approach)
        # This must happen AFTER data is loaded and stored
        if lib_data is None:
            if 'tw' in self.problem_name:
                if 'a' in self.problem_name:
                    self.depot_end = 1.0
                else:
                    self.depot_end = 3.0
                # Overwrite the depot's tw_end values
                self.tw_end[:, :self.depot_num] = self.depot_end
        
        # Prepare reset state
        self._prepare_reset_state()
    
    def _load_lib_data(self, lib_data: Dict) -> Dict:
        """Load data from library format"""
        data = {}
        for key in ['xy', 'demand', 'dist', 'prize', 'penalty', 'fake_prize', 
                    'service_time', 'tw_start', 'tw_end', 'route_limit']:
            if key in lib_data:
                data[key] = lib_data[key].to(self.device)
        return data
    
    def _load_validation_data(self, validation_data: Dict, start: int, batch_size: int) -> Dict:
        """Load data from validation dataset"""
        data = {}
        
        # Load xy coordinates (or create dummy zeros if not present)
        if 'xy' in validation_data:
            data['xy'] = validation_data['xy'][start:start+batch_size].to(self.device)
        else:
            # Create dummy zero coordinates like original URS does
            data['xy'] = torch.zeros(batch_size, self.problem_size + self.depot_num, 2, device=self.device)
        
        # Load other data
        for key in validation_data.keys():
            if key != 'xy' and isinstance(validation_data[key], torch.Tensor):
                data[key] = validation_data[key][start:start+batch_size].to(self.device)
        
        # Fill in missing fields with zeros/defaults like original URS
        if 'demand' not in data:
            data['demand'] = torch.zeros(batch_size, self.problem_size + self.depot_num, device=self.device)
        if 'prize' not in data:
            data['prize'] = torch.zeros(batch_size, self.problem_size + self.depot_num, device=self.device)
        if 'penalty' not in data:
            data['penalty'] = torch.zeros(batch_size, self.problem_size + self.depot_num, device=self.device)
        if 'fake_prize' not in data:
            data['fake_prize'] = torch.zeros(batch_size, self.problem_size + self.depot_num, device=self.device)
        if 'service_time' not in data:
            data['service_time'] = torch.zeros(batch_size, self.problem_size + self.depot_num, device=self.device)
        if 'tw_start' not in data:
            data['tw_start'] = torch.zeros(batch_size, self.problem_size + self.depot_num, device=self.device)
        if 'tw_end' not in data:
            data['tw_end'] = torch.full((batch_size, self.problem_size + self.depot_num), float('inf'), device=self.device)
        if 'route_limit' not in data:
            data['route_limit'] = torch.full((batch_size,), float('inf'), device=self.device)
        
        return data
    
    def _load_saved_data(self, batch_size: int) -> Dict:
        """Load data from saved problems"""
        data = {}
        for key in self.saved_data.keys():
            if isinstance(self.saved_data[key], torch.Tensor):
                data[key] = self.saved_data[key][self.saved_index:self.saved_index+batch_size].to(self.device)
        self.saved_index += batch_size
        return data
    
    def _augment_data(self, data: Dict, aug_factor: int) -> Dict:
        """Apply data augmentation (e.g., 8-fold for rotation/reflection)"""
        from ProblemDef import augment_xy_data_by_8_fold
        
        augmented_data = {}
        for key, value in data.items():
            if value is None:
                augmented_data[key] = None
            elif key in ['xy'] and aug_factor == 8:
                augmented_data[key] = augment_xy_data_by_8_fold(value)
            elif key in ['dist'] and aug_factor == 8:
                augmented_data[key] = value.repeat(aug_factor, 1, 1)
            elif isinstance(value, torch.Tensor) and value.dim() >= 1:
                augmented_data[key] = value.repeat(aug_factor, *([1] * (value.dim() - 1)))
            else:
                augmented_data[key] = value
        
        return augmented_data
    
    def _store_problem_data(self, data: Dict):
        """Store problem data in environment"""
        # Move all data to the correct device
        self.xy = data.get('xy').to(self.device) if data.get('xy') is not None else None
        self.demand = data.get('demand').to(self.device) if data.get('demand') is not None else None
        self.dist = data.get('dist').to(self.device) if data.get('dist') is not None else None
        self.prize = data.get('prize').to(self.device) if data.get('prize') is not None else None
        self.penalty = data.get('penalty').to(self.device) if data.get('penalty') is not None else None
        self.fake_prize = data.get('fake_prize').to(self.device) if data.get('fake_prize') is not None else None
        self.service_time = data.get('service_time').to(self.device) if data.get('service_time') is not None else None
        self.tw_start = data.get('tw_start').to(self.device) if data.get('tw_start') is not None else None
        self.tw_end = data.get('tw_end').to(self.device) if data.get('tw_end') is not None else None
        self.route_limit = data.get('route_limit').to(self.device) if data.get('route_limit') is not None else None
        if self.route_limit is not None and self.route_limit.dim() == 1:
            self.route_limit = self.route_limit[:, None]
        
        # Compute distance matrix if not provided
        if self.dist is None and self.xy is not None:
            self.dist = torch.cdist(self.xy, self.xy, p=2, compute_mode='donot_use_mm_for_euclid_dist')
        
        # Set actual node count based on data
        # For some problems like ATSP, the dist matrix doesn't include depot concept
        if self.dist is not None:
            self.node_cnt = self.dist.shape[-1]
        elif self.xy is not None:
            self.node_cnt = self.xy.shape[1]
        elif self.demand is not None:
            self.node_cnt = self.demand.shape[1]
        else:
            self.node_cnt = self.problem_size + self.depot_num
        
        # Adjust pomo_size for VRPB problems (only positive demand nodes can be starting nodes)
        # CRITICAL: bp (backhaul-pickup) problems do NOT need pomo_size adjustment (they can start from any node)
        # Following original URS logic: if 'b' in problem_name and 'bp' not in problem_name
        is_backhaul_not_bp = 'b' in self.problem_name and 'bp' not in self.problem_name
        if is_backhaul_not_bp:
            if self.demand is not None and self.depot_num > 0:
                # Count positive demand nodes (excluding depot)
                positive_demand_count = (self.demand[:, self.depot_num:] > 0).sum(dim=1).min().item()
                # Update pomo_size to be at most the minimum number of positive nodes across batches
                self.pomo_size = min(int(positive_demand_count * self.depot_num), self.pomo_size)
    
    def _prepare_reset_state(self):
        """Prepare reset state with all problem information"""
        self.reset_state.problem_name = self.problem_name
        self.reset_state.xy = self.xy
        self.reset_state.demand = self.demand
        self.reset_state.dist = self.dist
        self.reset_state.prize = self.prize
        self.reset_state.penalty = self.penalty
        self.reset_state.fake_prize = self.fake_prize
        self.reset_state.service_time = self.service_time
        self.reset_state.tw_start = self.tw_start
        self.reset_state.tw_end = self.tw_end
        self.reset_state.route_limit = self.route_limit
        self.reset_state.log_scale = math.log2(self.problem_size)
        
        # Set problems tensor (for backward compatibility with Model.pre_forward)
        # Match source code URS/Env/UNIEnv.py lines 447-467
        # Following original URS logic for problem tensor construction
        if self.problem_name in ['tsp', 'atsp', 'cvrp', 'sdvrp', 'pdp', 'mdcvrp', 'acvrp', 'hcvrp', 'mtsp', 
                                 'pdcvrp', 'amdcvrp', 'pdvrp', 'apdp', 'pdcvrpl', 'apdcvrp', 'apdcvrpl', 
                                 'opdcvrp', 'aopdcvrp']:
            # All these problems use [xy, demand] format (3 columns)
            if self.problem_name == 'atsp':
                # ATSP is special - uses distance matrix
                self.reset_state.problems = self.dist
            elif self.xy is not None and self.demand is not None:
                self.reset_state.problems = torch.cat([self.xy, self.demand.unsqueeze(-1)], dim=-1)
            else:
                # Fallback for missing data
                self.reset_state.problems = self.dist if self.dist is not None else self.xy
        elif self.problem_name == 'op':
            # OP: [xy, prize] - 3 columns
            self.reset_state.problems = torch.cat([self.xy, self.prize.unsqueeze(-1)], dim=-1)
        elif self.problem_name == 'pctsp':
            # PCTSP: [xy, prize, penalty] - 4 columns
            self.reset_state.problems = torch.cat([self.xy, self.prize.unsqueeze(-1), self.penalty.unsqueeze(-1)], dim=-1)
        elif self.problem_name == 'spctsp':
            # SPCTSP: [xy, prize, fake_prize, penalty] - 5 columns
            self.reset_state.problems = torch.cat([self.xy, self.prize.unsqueeze(-1), self.fake_prize.unsqueeze(-1), self.penalty.unsqueeze(-1)], dim=-1)
        elif 'vrpmix' in str(type(self).__name__).lower() or self.problem_name in get_problem_list("all_vrpmix_list"):
            # VRP mix variants
            if 'tw' in self.problem_name:
                # VRP with time windows: [xy, demand, tw_start, tw_end, service_time] - 6 columns
                self.reset_state.problems = torch.cat([
                    self.xy, 
                    self.demand.unsqueeze(-1), 
                    self.tw_start.unsqueeze(-1), 
                    self.tw_end.unsqueeze(-1), 
                    self.service_time.unsqueeze(-1)
                ], dim=-1)
            else:
                # VRP mix without TW: [xy, demand] - 3 columns
                self.reset_state.problems = torch.cat([self.xy, self.demand.unsqueeze(-1)], dim=-1)
        else:
            # Fallback for other problems
            if self.xy is not None and self.demand is not None:
                self.reset_state.problems = torch.cat([self.xy, self.demand.unsqueeze(-1)], dim=-1)
            elif self.xy is not None:
                self.reset_state.problems = self.xy
            elif self.dist is not None:
                self.reset_state.problems = self.dist
    
    def reset(self):
        """Reset environment to initial state"""
        self.selected_count = 0
        self.current_node = None
        self.current_depot = torch.zeros(size=(self.batch_size, self.pomo_size), dtype=torch.long, device=self.device)
        self.selected_node_list = torch.zeros((self.batch_size, self.pomo_size, 0), dtype=torch.long, device=self.device)
        
        # Initialize capacity/load
        if self.has_capacity:
            self.load = torch.ones(size=(self.batch_size, self.pomo_size), device=self.device)
        else:
            self.load = None
        
        # Initialize time
        if self.has_time_window:
            self.current_time = torch.zeros(size=(self.batch_size, self.pomo_size), device=self.device)
        else:
            self.current_time = None
        
        # Initialize route counter and cumulative length
        if self.has_route_limit:
            self.current_route = torch.zeros(size=(self.batch_size, self.pomo_size), dtype=torch.long, device=self.device)
            self.cumulative_length = torch.zeros(size=(self.batch_size, self.pomo_size), device=self.device)
        else:
            self.current_route = None
            self.cumulative_length = None
        
        # Initialize collected prize
        if self.has_prize:
            self.collected_prize = torch.zeros(size=(self.batch_size, self.pomo_size), device=self.device)
        else:
            self.collected_prize = None
        
        # Initialize tour_maxlength for OP
        if self.problem_name in ['op']:
            self.max_length = 4.0
            self.tour_maxlength = torch.ones(size=(self.batch_size, self.pomo_size), device=self.device) * self.max_length
        else:
            self.tour_maxlength = None
        
        # Initialize masks
        self.visited_ninf_flag = torch.zeros(size=(self.batch_size, self.pomo_size, self.node_cnt), device=self.device)
        self.ninf_mask = torch.zeros(size=(self.batch_size, self.pomo_size, self.node_cnt), device=self.device)
        
        # Initialize dynamic_demand for SDVRP (use expand without clone to match source)
        if self.demand is not None:
            # For SDVRP, dynamic_demand will be updated via scatter_add which creates new tensor
            # For other problems, it's just a view of demand
            if self.problem_name == 'sdvrp':
                self.dynamic_demand = self.demand[:, None, :].expand(self.batch_size, self.pomo_size, -1).clone()
            else:
                self.dynamic_demand = self.demand[:, None, :].expand(self.batch_size, self.pomo_size, -1)
        else:
            self.dynamic_demand = None
        
        # Initialize state flags
        self.at_the_depot = torch.ones(size=(self.batch_size, self.pomo_size), dtype=torch.bool, device=self.device)
        self.finished = torch.zeros(size=(self.batch_size, self.pomo_size), dtype=torch.bool, device=self.device)
        
        # Initialize PDP to_deliver mask (matches Original URS lines 548-563)
        if 'pdp' in self.problem_name or 'pd' in self.problem_name:
            # to_deliver: front half (including depot) is True (pickup), back half is False (delivery)
            # Shape: [batch, pomo, problem_size+depot_num]

            self.to_deliver = torch.cat(
                [
                    torch.ones(
                        self.batch_size,
                        self.pomo_size,
                        self.problem_size // 2 + self.depot_num,
                        dtype=torch.bool,
                        device=self.device
                    ),
                    torch.zeros(
                        self.batch_size,
                        self.pomo_size,
                        self.problem_size // 2,
                        dtype=torch.bool,
                        device=self.device
                    ),
                ],
                dim=-1,
            )
            
            # CRITICAL: Initialize PDP relation matrix (matches Original URS lines 567-575)
            # This encodes pickup-delivery relationships for the encoder!
            pd_matrix = torch.ones(self.batch_size, self.problem_size + self.depot_num, self.problem_size + self.depot_num, device=self.device)
            half = self.problem_size // 2
            pairs = torch.arange(1, half + 1, device=self.device)
            pickup = pairs
            deliver = pairs + half
            row = torch.cat([pickup, deliver])
            col = torch.cat([deliver, pickup])
            pd_matrix[:, row, col] = 0  # related pairs = 0, unrelated = 1
            self.reset_state.relation = pd_matrix
        else:
            self.to_deliver = None
        
        # Initialize START_NODE for VRPB-type problems
        if self.has_backhaul or self.problem_name in ['vrpb', 'vrpbl', 'vrpbtw', 'vrpbltw', 'vrpbp', 'vrpbpl', 'vrpbpltw', 'vrpbptw', 
                                                       'ovrpb', 'ovrpbl', 'ovrpbltw', 'ovrpbp', 'ovrpbpl', 'ovrpbpltw', 'ovrpbptw',
                                                       'mdovrpb', 'mdovrpbl', 'mdovrpbltw', 'mdovrpbp', 'mdovrpbpl', 'mdovrpbpltw', 'mdovrpbptw',
                                                       'amdovrpb', 'amdovrpbl', 'amdovrpbltw', 'amdovrpbp', 'amdovrpbpl', 'amdovrpbpltw', 'amdovrpbptw']:
            # For VRPB, only nodes with positive demand can be starting nodes
            if self.demand is not None and self.depot_num > 0:
                # Create a START_NODE tensor for each batch by filtering positive demand nodes
                START_NODE_list = []
                for b in range(self.batch_size):
                    # Get indices of positive demand nodes (excluding depot)
                    positive_indices = torch.where(self.demand[b, self.depot_num:] > 0)[0] + self.depot_num
                    # Take first pomo_size nodes
                    START_NODE_list.append(positive_indices[:self.pomo_size])
                self.START_NODE = torch.stack(START_NODE_list, dim=0)
            else:
                # Fallback: use first pomo_size customer nodes
                self.START_NODE = torch.arange(start=self.depot_num, end=self.depot_num+self.pomo_size, device=self.device)[None, :].expand(self.batch_size, -1)
        else:
            self.START_NODE = None
        
        # Set log_scale for distance normalization (matches Original URS line 542)
        self.reset_state.log_scale = math.log2(self.problem_size)
        
        # Prepare step state
        self.step_state.batch_size = self.batch_size
        self.step_state.pomo_size = self.pomo_size
        self.step_state.depot_num = self.depot_num
        
        reward = None
        done = False
        return self.reset_state, reward, done
    
    def pre_step(self):
        """Prepare state before action selection"""
        self.step_state.selected_count = self.selected_count
        self.step_state.current_node = self.current_node
        self.step_state.current_time = self.current_time
        self.step_state.current_route = self.current_route
        self.step_state.load = self.load
        self.step_state.collected_prize = self.collected_prize
        self.step_state.ninf_mask = self.ninf_mask
        self.step_state.finished = self.finished
        self.step_state.START_NODE = getattr(self, 'START_NODE', None)
        self.step_state.tour_maxlength = getattr(self, 'tour_maxlength', None)
        self.step_state.cumulative_length = self.cumulative_length
        
        reward = None
        done = False
        return self.step_state, reward, done
    
    def step(self, selected: torch.Tensor, mask_fn=None):
        """
        Execute one step with selected actions
        
        Args:
            selected: Selected node indices, shape (batch, pomo)
            mask_fn: Optional custom mask function from mask registry
        
        Returns:
            step_state: Updated step state
            reward: Reward (only at end of episode)
            done: Whether episode is finished
        """
        # Update selection history
        self.selected_count += 1
        
        # Debug: track depot returns
        if hasattr(self, '_debug_step_counter'):
            self._debug_step_counter += 1
        else:
            self._debug_step_counter = 1
        
        # Removed debug code
        
        # CRITICAL: Save old current_node BEFORE updating, needed for distance calculation
        old_node = self.current_node
        
        self.current_node = selected
        # Note: Keep gradient graph intact for proper backpropagation
        self.selected_node_list = torch.cat(
            (self.selected_node_list, self.current_node.unsqueeze(-1)), dim=2
        )
        
        # Update depot flag (for multi-depot: check if selected < depot_num)
        self.at_the_depot = (selected < self.depot_num)
        
        # Track current depot (for tour_length_constraint_mask)
        # Matches Original URS line 632
        self.current_depot[self.at_the_depot] = selected[self.at_the_depot]
        
        # Always update dynamic states (capacity, prize, time, etc.)
        # Pass old_node for correct distance calculation (matches Original URS last_node)
        self._update_states(selected, old_node=old_node)
        
        # Update masks: use custom mask function if provided, otherwise use default
        if mask_fn is not None:
            # Use custom mask function
            # CRITICAL: Use no_grad to prevent any gradient graph creation in mask computation
            with torch.no_grad():
                self.ninf_mask = mask_fn(self, selected)
        else:
            # Use default mask update
            self._update_masks(selected)
        
        # Check if finished
        self._check_finished()
        
        # Update step state
        self._update_step_state()
        
        # Calculate reward if done
        # For TSP/ATSP, done when all nodes visited (selected_count == node_cnt)
        if self.problem_name in ['tsp', 'atsp']:
            done = (self.selected_count == self.node_cnt)
        else:
            done = self.finished.all()
        
        if done:
            reward = self._calculate_reward()
        else:
            reward = None
        
        return self.step_state, reward, done
    
    def _update_states(self, selected: torch.Tensor, old_node: Optional[torch.Tensor] = None):
        """Update dynamic states based on problem type
        
        Args:
            selected: Newly selected node
            old_node: Previous current_node (before update), needed for distance calculation
        """
        # Update capacity
        if self.has_capacity and self.demand is not None:
            gathering_index = selected.unsqueeze(-1)
            
            # SDVRP: Split delivery - use dynamic_demand and min(demand, load)
            if self.problem_name == 'sdvrp':
                # For SDVRP, use dynamic_demand (which tracks remaining demand)
                demand_list = self.dynamic_demand
                selected_demand = demand_list.gather(dim=2, index=gathering_index).squeeze(dim=2)
                actual_selected_demand = torch.min(selected_demand, self.load)
                self.load -= actual_selected_demand
                self.dynamic_demand = self.dynamic_demand.scatter_add(-1, gathering_index, -actual_selected_demand.unsqueeze(-1))
            else:
                demand_list = self.demand[:, None, :].expand(self.batch_size, self.pomo_size, -1)
                selected_demand = demand_list.gather(dim=2, index=gathering_index).squeeze(dim=2)
                
                # VRPBP: If departing from depot and selecting backhaul node, load starts from 0
                # This matches URS original line 658-661
                if old_node is not None and "bp" in self.problem_name:
                    last_at_depot = old_node < self.depot_num
                    begin_with_backhaul = selected_demand < 0
                    self.load[last_at_depot & begin_with_backhaul] = 0
                
                self.load -= selected_demand
            
            # Standard: refill to 1.0 at depot (matches original URS line 673)
            self.load[self.at_the_depot] = 1.0
            
            # VRPB: Special load reset logic (matches original URS line 693-703)
            # This MUST come AFTER the standard refill above
            # CRITICAL: This runs EVERY step, not just when at depot!
            # Apply to all vrpmix problems with 'b' but not 'bp' (backhaul without pickup-delivery)
            from multi_hot_set import get_problem_list
            is_backhaul_vrpmix = (self.problem_name in get_problem_list("all_vrpmix_list") and 
                                  'b' in self.problem_name and 'bp' not in self.problem_name)
            if is_backhaul_vrpmix:
                # Need to recalculate demand_list for backhaul logic
                demand_list = self.demand[:, None, :].expand(self.batch_size, self.pomo_size, -1)
                # Create visited mask (matches original URS line 698)
                from masks_unified.mask_registry import mask_visited_nodes
                visited_mask = mask_visited_nodes(self.current_node, self.selected_node_list, 
                                                 self.problem_size + self.depot_num, self.device)
                # For non-depot pomo instances, zero out visited_mask (line 699)
                visited_mask[~self.at_the_depot] = 0
                
                # Unvisited demand: visited=-inf, unvisited=demand (line 700)
                unvisited_demand = demand_list + visited_mask
                # shape: (batch, pomo, problem+1)
                
                # Check if any linehauls unserved (line 701)
                linehauls_unserved = torch.where(unvisited_demand > 0., True, False)

                # Reset load to 0 at depot if no linehauls remain (line 702-703)
                reset_index = self.at_the_depot & (~linehauls_unserved.any(dim=-1))
                # shape: (batch, pomo)
                self.load[reset_index] = 0.
        
        # Update time
        if self.has_time_window:
            self._update_time(selected, old_node=old_node)
        
        # Update collected prize
        if self.has_prize and self.prize is not None:
            prize_list = self.prize[:, None, :].expand(self.batch_size, self.pomo_size, -1)
            gathering_index = selected.unsqueeze(-1)
            selected_prize = prize_list.gather(dim=2, index=gathering_index).squeeze(dim=2)
            self.collected_prize += selected_prize
        
        # Update tour_maxlength for OP
        if self.problem_name in ['op'] and self.tour_maxlength is not None:
            if self.selected_count == 1:
                # First move: distance from depot to selected node
                selected_expanded = selected.unsqueeze(-1)
                depot_to_selected = self.dist[:, 0:1, :].expand(-1, self.pomo_size, -1).gather(dim=2, index=selected_expanded).squeeze(-1)
                self.tour_maxlength -= depot_to_selected
            else:
                # Subsequent moves: distance from previous node to current node
                prev_node = self.selected_node_list[:, :, -2]
                batch_idx = torch.arange(self.batch_size, device=selected.device)[:, None].expand(-1, self.pomo_size)
                travel_dist = self.dist[batch_idx, prev_node, selected]
                self.tour_maxlength -= travel_dist
        
        # Update cumulative length and route counter
        if self.has_route_limit:
            if self.selected_count > 1:
                prev_node = self.selected_node_list[:, :, -2]
                batch_idx = torch.arange(self.batch_size, device=selected.device)[:, None].expand(-1, self.pomo_size)
                new_length = self.dist[batch_idx, prev_node, selected]
                self.cumulative_length = self.cumulative_length + new_length
                self.cumulative_length[self.at_the_depot] = 0.0
            
            newly_at_depot = self.at_the_depot & (self.selected_count > 1)
            self.current_route += newly_at_depot.long()
        
        # Update PDP picked_up state
        if self.problem_name == 'pdp':
            n = (self.node_cnt - 1) // 2
            if not hasattr(self, 'picked_up'):
                self.picked_up = torch.zeros((self.batch_size, self.pomo_size, n + 1), 
                                             dtype=torch.bool, device=selected.device)
            # Mark pickup nodes as picked up
            is_pickup = (selected >= 1) & (selected <= n)
            pickup_idx = selected.clone()
            pickup_idx[~is_pickup] = 0
            self.picked_up.scatter_(dim=-1, index=pickup_idx.unsqueeze(-1), value=True)
        
        # Mark visited nodes (must be done here so mask_fn can use it)
        # SDVRP: visited_ninf_flag is updated in mask based on dynamic_demand == 0
        # For SDVRP, nodes can be partially satisfied, so we don't mark them visited here
        if self.problem_name != "sdvrp":
            gathering_index = selected.unsqueeze(-1)
            self.visited_ninf_flag.scatter_(dim=-1, index=gathering_index, value=float('-inf'))
        
            # Depot is unvisited unless at depot (for closed routes)
            if not self.open_route:
                self.visited_ninf_flag[:, :, 0][~self.at_the_depot] = 0
    
    def _update_time(self, selected: torch.Tensor, old_node: Optional[torch.Tensor] = None):
        """Update time for time window constraints
        
        Args:
            selected: Newly selected node
            old_node: Previous current_node (before update), needed for distance calculation
                     Matches Original URS last_node usage
        """
        if self.dist is None or self.service_time is None:
            return
        
        # CRITICAL: Use old_node (before update) for distance calculation
        # Matches Original URS: new_length = self.dist[batch_idx, last_node, selected]
        if old_node is None:
            # Fallback: shouldn't happen in normal operation
            old_node = self.current_node
        
        # Get travel time from old_node to selected node
        batch_idx = torch.arange(self.batch_size, device=self.device)[:, None]
        travel_dist = self.dist[batch_idx, old_node, selected]
        travel_time = travel_dist / self.speed
        
        # Get tw_start and service_time for selected nodes
        # Matches Original URS line 731-733:
        # self.current_time = torch.max(self.current_time + new_length / self.speed,
        #                               self.tw_start[torch.arange(self.batch_size)[:, None], selected]) + \
        #                     self.service_time[torch.arange(self.batch_size)[:, None], selected]
        selected_tw_start = self.tw_start[batch_idx, selected]
        selected_service_time = self.service_time[batch_idx, selected]
        
        # Update current_time: max(arrival_time, tw_start) + service_time
        arrival_time = self.current_time + travel_time
        self.current_time = torch.max(arrival_time, selected_tw_start) + selected_service_time
        
        # Reset time at depot (same as original URS line 711)
        # This applies to both open and closed routes
        self.current_time[self.at_the_depot] = 0.0
    
    def _update_masks(self, selected: torch.Tensor):
        """Update masks - should not be called, mask_fn should always be provided"""
        # This should not be reached in normal operation as mask_fn is always provided in step()
        # Raise an error to catch any misuse
        raise NotImplementedError(
            "Default mask update not implemented. Please provide mask_fn to step() method."
        )
    
    def _apply_time_window_mask(self):
        """Apply time window constraints to mask"""
        if self.dist is None or self.tw_start is None or self.tw_end is None:
            return
        
        # Calculate arrival times to all nodes
        if self.current_node is None:
            return
        
        current_node_expanded = self.current_node.unsqueeze(-1).expand(-1, -1, self.node_cnt)
        current_to_all_dist = self.dist.gather(dim=1, index=current_node_expanded)
        
        next_time_required = self.current_time[:, :, None] + current_to_all_dist / self.speed
        tw_start_expanded = self.tw_start[:, None, :].expand(-1, self.pomo_size, -1)
        arrival_time = torch.max(next_time_required, tw_start_expanded)
        
        # Check if arrival time exceeds time window
        tw_end_expanded = self.tw_end[:, None, :].expand(-1, self.pomo_size, -1)
        out_of_tw = arrival_time > tw_end_expanded + self.round_error_epsilon
        self.ninf_mask[out_of_tw] = float('-inf')
        
        # For closed routes, check if can return to depot
        if not self.open_route and self.service_time is not None:
            service_time_expanded = self.service_time[:, None, :].expand(-1, self.pomo_size, -1)
            # Distance from each node to depot (node 0)
            to_depot_dist = self.dist[:, :, 0]  # Shape: (batch, node_cnt)
            to_depot_dist_expanded = to_depot_dist[:, None, :].expand(-1, self.pomo_size, -1)
            
            depot_end = self.tw_end[:, 0:1]
            fail_return_depot = (arrival_time + service_time_expanded + to_depot_dist_expanded / self.speed 
                                > depot_end[:, None, :].expand(-1, self.pomo_size, self.node_cnt) + self.round_error_epsilon)
            self.ninf_mask[fail_return_depot] = float('-inf')
    
    def _check_finished(self):
        """Check if episodes are finished"""
        # For TSP/ATSP, check if all nodes visited
        if self.problem_name in ['tsp', 'atsp']:
            # TSP/ATSP finished when selected_count reaches node_cnt
            # This will be checked in step() method
            pass
        elif self.problem_name in ['op']:
            # OP: finished when at depot and selected_count > 1 (original URS line 1127)
            # The mask function already handles masking all customers when finished
            newly_finished = self.at_the_depot & (self.selected_count > 1)
            self.finished = self.finished | newly_finished
            
            # CRITICAL: Unmask depot for finished episodes to prevent all-inf mask (which causes NaN in softmax)
            self.ninf_mask[:, :, 0][self.finished] = 0
        elif self.problem_name in ['pctsp', 'spctsp']:
            # PCTSP: finished when at depot and selected_count > 1 (similar to OP)
            newly_finished = self.at_the_depot & (self.selected_count > 1)
            self.finished = self.finished | newly_finished
            
            # CRITICAL: Unmask depot for finished episodes to prevent all-inf mask
            self.ninf_mask[:, :, 0][self.finished] = 0
        else:
            # For VRP problems, finished when all nodes are masked (same as original URS line 718)
            newly_finished = (self.ninf_mask == float('-inf')).all(dim=2)
            self.finished = self.finished | newly_finished
            
            # Unmask depot for finished episodes (same as original URS line 723)
            # This allows the final selection of depot to end the episode
            self.ninf_mask[:, :, 0][self.finished] = 0
    
    def _update_step_state(self):
        """Update step state for output"""
        self.step_state.selected_count = self.selected_count
        self.step_state.current_node = self.current_node
        self.step_state.current_time = self.current_time
        self.step_state.current_route = self.current_route
        self.step_state.load = self.load
        self.step_state.collected_prize = self.collected_prize
        self.step_state.ninf_mask = self.ninf_mask
        self.step_state.finished = self.finished
    
    def _calculate_reward(self) -> torch.Tensor:
        """Calculate reward based on problem type"""
        # PCTSP/SPCTSP: minimize (travel_distance + penalty) - different from OP!
        if self.problem_name in ['pctsp', 'spctsp']:
            travel_distance = self._get_travel_distance()
            # Calculate penalty for unvisited nodes
            visited = torch.zeros_like(self.visited_ninf_flag)
            visited[self.visited_ninf_flag == float('-inf')] = 1
            if self.penalty is not None:
                penalty_expanded = self.penalty[:, None, :].expand(-1, self.pomo_size, -1)
                unvisited_penalty = ((1 - visited) * penalty_expanded).sum(dim=-1)
            else:
                unvisited_penalty = 0
            # Return negative (distance + penalty) to maximize reward
            return -(travel_distance + unvisited_penalty)
        elif self.has_prize:
            # For OP: maximize collected prize
            reward = self.collected_prize
            return reward
        else:
            # For routing problems, reward is negative travel distance
            travel_distance = self._get_travel_distance()
            return -travel_distance
    
    def _get_travel_distance(self) -> torch.Tensor:
        """Calculate total travel distance for the solution"""
        # Match original URS behavior:
        # - For most problems: use dist matrix (original URS _get_total_distance, line 935-957)
        # - For PCTSP/SPCTSP: use xy coordinates (original URS _get_travel_distance, line 970-1027)
        # - For asymmetric problems: must use dist matrix
        # - For multi-depot: use _get_md_reward style calculation
        
        use_xy_for_distance = (
            self.xy is not None and 
            self.problem_name in ['pctsp', 'spctsp']  # Only PCTSP uses xy in original URS
        )
        
        if use_xy_for_distance:
            # Calculate from coordinates (only for PCTSP/SPCTSP)
            gathering_index = self.selected_node_list[:, :, :, None].expand(-1, -1, -1, 2)
            all_xy = self.xy[:, None, :, :].expand(-1, self.pomo_size, -1, -1)
            ordered_seq = all_xy.gather(dim=2, index=gathering_index)
            rolled_seq = ordered_seq.roll(dims=2, shifts=-1)
            segment_lengths = ((ordered_seq - rolled_seq) ** 2).sum(3).sqrt()
            # For PCTSP, exclude last segment (matches original URS line 1027)
            if self.problem_name in ['pctsp', 'spctsp']:
                segment_lengths = segment_lengths[:, :, :-1]
            # For other open routes, also exclude last segment
            elif self.open_route:
                segment_lengths = segment_lengths[:, :, :-1]
            
            travel_distances = segment_lengths.sum(2)
        elif self.dist is not None:
            # Multi-depot problems need special handling (matches Original URS _get_md_reward)
            if self.depot_num > 1:
                travel_distances = self._get_md_travel_distance()
            else:
                # Single depot: simpler calculation
                node_from = self.selected_node_list
                seq_len = node_from.size(-1)
                node_to = self.selected_node_list.roll(dims=2, shifts=-1)
                
                # Create batch index with correct shape
                batch_idx = torch.arange(self.batch_size, device=self.dist.device)[:, None].expand(self.batch_size, self.pomo_size)
                batch_index = batch_idx[:, :, None].expand(self.batch_size, self.pomo_size, seq_len)
                
                selected_cost = self.dist[batch_index, node_from, node_to]
                
                # For open routes, exclude segments back to depot
                if self.open_route:
                    not_to_depot = self.selected_node_list.roll(dims=2, shifts=-1) != 0
                    travel_distances = (selected_cost * not_to_depot).sum(2)
                else:
                    travel_distances = selected_cost.sum(2)
        else:
            raise ValueError("Either xy coordinates or distance matrix must be provided")

        return travel_distances
    
    def _get_md_travel_distance(self) -> torch.Tensor:
        """Calculate travel distance for multi-depot problems
        
        Matches Original URS _get_md_reward function
        Key differences from single-depot:
        - When going to a depot: if open_route, set distance to 0, otherwise distance to actual depot
        - depot to depot transitions: set distance to 0
        """
        num_depots = self.depot_num
        
        # Flatten batch and pomo dimensions
        actions = self.selected_node_list.reshape(self.batch_size * self.pomo_size, -1)
        dist_matrix = self.dist.repeat_interleave(self.pomo_size, dim=0)
        b, seq_len = actions.size()
        
        # go_from / go_to node indices
        go_from = actions  # [b, seq_len]
        go_to = torch.roll(go_from, -1, dims=1)  # [b, seq_len]
        
        # Batch index for gathering from dist_matrix
        batch_idx = torch.arange(b, device=dist_matrix.device).unsqueeze(1).expand_as(go_from)
        
        # Get starting depot for each position (for closed route return)
        starting_points = self._get_starting_points(actions, num_depots)  # [b, seq_len]
        actual_depot = torch.roll(starting_points, 1, dims=1)  # [b, seq_len]
        
        # Get distances from dist_matrix
        distances = dist_matrix[batch_idx, go_from, go_to]  # [b, seq_len]
        
        if self.open_route:
            distances_to_depot = torch.zeros_like(distances)
        else:
            distances_to_depot = dist_matrix[batch_idx, go_from, actual_depot]  # [b, seq_len]
        
        # When going to depot, use distance_to_depot (0 for open route)
        is_depot = go_to < num_depots
        distances = torch.where(is_depot, distances_to_depot, distances)
        
        # depot to depot: set distance to 0
        is_depot_to_depot = (go_from < num_depots) & (go_to < num_depots)
        distances = torch.where(is_depot_to_depot, torch.zeros_like(distances), distances)
        
        # Sum distances
        tour_length = distances.sum(-1)  # [b]
        
        return tour_length.reshape(self.batch_size, self.pomo_size)
    
    def _get_starting_points(self, actions: torch.Tensor, num_depots: int) -> torch.Tensor:
        """Get which depot (starting point) each action in the sequence starts from
        
        Matches Original URS get_starting_points function
        Example:
        >>> actions = torch.tensor([[1, 10, 2, 0, 3, 30, 21, 2], [2, 15, 20, 1, 25, 30, 0, 1]])
        >>> get_starting_points(actions, 3) -> torch.tensor([[1, 1, 2, 0, 0, 0, 0, 2], [2, 2, 2, 1, 1, 1, 0, 1]])
        """
        # Create mask for numbers < num_depots
        mask = actions < num_depots  # shape: (batch_size, seq_len)
        batch_size, seq_len = actions.shape

        # Compute the cumulative sum of the mask to get segment IDs
        segment_ids = torch.cumsum(mask.long(), dim=1)  # shape: (batch_size, seq_len)

        # Adjust segment IDs for indexing (shift by -1)
        segment_indices = segment_ids - 1

        # Create a mask for valid segment positions
        valid_positions = segment_ids > 0

        # Compute the number of masked elements per batch
        num_values_per_batch = mask.sum(dim=1)  # shape: (batch_size,)
        max_num_values = num_values_per_batch.max().item()

        # Generate batch indices
        batch_indices = (
            torch.arange(batch_size, device=actions.device)
            .unsqueeze(1)
            .expand(batch_size, seq_len)
        )

        # Get indices where mask is True
        masked_indices = torch.where(
            mask,
            torch.cumsum(mask.long(), dim=1) - 1,
            torch.tensor(-1, device=actions.device),
        )
        valid_masked_positions = masked_indices >= 0

        # Gather valid batch and masked indices
        valid_batch_indices = batch_indices[valid_masked_positions]
        valid_masked_indices = masked_indices[valid_masked_positions]
        valid_actions = actions[valid_masked_positions]

        # Initialize padded values tensor
        values_padded = torch.zeros(
            batch_size, max_num_values, dtype=actions.dtype, device=actions.device
        )

        # Fill in the padded values tensor
        values_padded[valid_batch_indices, valid_masked_indices] = valid_actions

        # Initialize the starting_points tensor
        starting_points = torch.zeros_like(actions)

        # Fill in the starting_points tensor using advanced indexing
        starting_points[valid_positions] = values_padded[
            batch_indices[valid_positions], segment_indices[valid_positions]
        ]

        return starting_points
    
    def get_local_feature(self) -> Optional[torch.Tensor]:
        """Get local features (distances from current node to all nodes)"""
        if self.current_node is None or self.dist is None:
            return None
        
        # Use actual dist shape instead of problem_size + depot_num
        # because for some problems (like ATSP), dist doesn't include depot
        num_nodes = self.dist.shape[-1]
        current_node_expanded = self.current_node.unsqueeze(-1).expand(-1, -1, num_nodes)
        cur_dist = self.dist.gather(dim=1, index=current_node_expanded)
        
        return cur_dist

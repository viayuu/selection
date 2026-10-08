"""
Mask Registry for Unified Environment
Maps problem types to their specific mask functions
"""
import torch
from typing import Dict, Callable, Optional


def mask_visited_nodes(current_node: torch.Tensor, visited_nodes: torch.Tensor, num_nodes: int, device=None) -> torch.Tensor:
    """
    Create a fresh mask marking visited nodes.
    Returns a NEW tensor (not a clone) to avoid gradient graph accumulation.
    
    Args:
        current_node: Current node ID. Shape: [batch_size, pomo_size]
        visited_nodes: List of visited node IDs. Shape: [batch_size, pomo_size, num_visited]
        num_nodes: Total number of nodes
        device: Device for the mask tensor
    
    Returns:
        mask: Float mask with -inf for visited nodes. Shape: [batch_size, pomo_size, num_nodes]
    """
    batch_size = current_node.size(0)
    pomo_size = current_node.size(1)
    if device is None:
        device = current_node.device
    
    mask = torch.zeros((batch_size, pomo_size, num_nodes), device=device)
    
    # Only scatter if there are visited nodes
    if visited_nodes.size(2) > 0:
        mask.scatter_(-1, visited_nodes, float('-inf'))
    
    # Always scatter current node
    mask.scatter_(-1, current_node.unsqueeze(-1), float('-inf'))
    return mask


class MaskRegistry:
    """
    Registry that manages mask functions for different problem types.
    Each problem can have its own custom mask function that defines
    the constraints for that specific problem variant.
    """
    
    def __init__(self):
        self._masks: Dict[str, Callable] = {}
    
    def register(self, problem_name: str, mask_fn: Callable):
        """
        Register a mask function for a problem type
        
        Args:
            problem_name: Name of the problem (e.g., 'cvrp', 'tsp', 'cvrptw')
            mask_fn: Function that takes (env, selected) and returns mask
        """
        self._masks[problem_name] = mask_fn
        print(f"Registered mask for problem: {problem_name}")
    
    def get_mask_fn(self, problem_name: str) -> Optional[Callable]:
        """
        Get mask function for a problem type
        
        Args:
            problem_name: Name of the problem
            
        Returns:
            Mask function if registered, None otherwise
        """
        return self._masks.get(problem_name)
    
    def list_registered_problems(self):
        """List all registered problem types"""
        return list(self._masks.keys())
    
    def __contains__(self, problem_name: str) -> bool:
        """Check if problem has registered mask"""
        return problem_name in self._masks


# Global mask registry instance
mask_registry = MaskRegistry()


def register_mask(problem_name: str):
    """
    Decorator to register a mask function
    
    Usage:
        @register_mask('cvrp')
        def mask_cvrp(env, selected):
            # ... mask logic ...
            return mask
    """
    def decorator(mask_fn: Callable):
        mask_registry.register(problem_name, mask_fn)
        return mask_fn
    return decorator

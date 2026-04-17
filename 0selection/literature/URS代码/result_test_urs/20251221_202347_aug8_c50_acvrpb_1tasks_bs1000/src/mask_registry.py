from typing import Dict, Callable, List, Optional
from functools import wraps

class MaskRegistry:
    def __init__(self):
        self._constraint_masks: Dict[str, Callable] = {}
        self._basic_constraint = 'visited'

    def register(self, constraint: str, mask_fn: Callable):
        self._constraint_masks[constraint] = mask_fn
        print(f"Registered mask for constraint: {constraint}")

    def get_mask_combo(self, problem_name: str) -> Optional[Callable]:
        needed_masks = []
        if self._basic_constraint in self._constraint_masks:
            needed_masks.append(self._constraint_masks[self._basic_constraint])

        target_constraints = ['op', 'pc', 'pd', 'vrp', 'tw', 'l', 'bp']
        for constraint in target_constraints:
            if constraint in problem_name and constraint in self._constraint_masks:
                needed_masks.append(self._constraint_masks[constraint])

        if not needed_masks:
            print(f"No valid masks found for problem: {problem_name}")
            return None

        @wraps(needed_masks[0])
        def combined_mask(env):
            final_mask = needed_masks[0](env)
            for mask_fn in needed_masks[1:]:
                current_mask = mask_fn(env)
                final_mask +=current_mask
            return final_mask

        return combined_mask

    def list_registered_problems(self):
        """List all registered problem types"""
        return list(self._constraint_masks.keys())

    def __contains__(self, problem_name: str) -> bool:
        """Check if problem has registered mask"""
        return problem_name in self._constraint_masks


# Global mask registry instance
mask_registry = MaskRegistry()


def register_mask(constraint: str):
    """
    Decorator to register a mask function

    Usage:
        @register_mask('tw')
        def tw_constraint_mask(env, selected):
            # ... mask logic ...
            return mask
    """

    def decorator(mask_fn: Callable):
        mask_registry.register(constraint, mask_fn)
        return mask_fn

    return decorator











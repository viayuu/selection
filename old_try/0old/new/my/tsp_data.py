from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence


@dataclass(frozen=True)
class GaussianMixtureParams:
    """
    Synthetic TSP Gaussian-mixture generator params copied from the paper code:
    `neural-solver-selection/datasets/data_utils.py:generate_tsp_data_gaussian`.
    """

    no_cov: bool = False
    var_lower: float = 1.0
    var_upper: float = 100.0
    num_modes_lower: int = 0
    num_modes_upper: int = 15
    center_lower: float = 0.0
    center_upper: float = 100.0


def _minmax_scale(xy):
    import numpy as np

    xy = np.asarray(xy)
    mn = xy.min(axis=0, keepdims=True)
    mx = xy.max(axis=0, keepdims=True)
    denom = mx - mn
    # Match sklearn MinMaxScaler behavior for degenerate dims: keep zeros.
    denom = np.where(denom < 1e-12, 1.0, denom)
    return (xy - mn) / denom


def _normalize_dist_name(name: str) -> str:
    n = name.strip().lower().replace("-", "_")
    if n in {"gmm", "gaussian_mixture"}:
        return "gaussian"
    return n


def _generate_tsp_instance_uniform(problem_size: int, rs):
    # In the paper code, the uniform case is `torch.rand(...).numpy().tolist()`.
    # Here we use NumPy, which is equivalent in distribution.
    return rs.rand(int(problem_size), 2).astype("float32", copy=False)


def _generate_tsp_instance_gaussian(problem_size: int, params: GaussianMixtureParams, rs):
    import numpy as np

    # Important: in the paper's dataset generator, `generate_tsp_data_gaussian(...)` is called with dataset_size=1,
    # so `num_modes` is sampled per-instance. We follow that behavior here.
    if params.num_modes_lower == params.num_modes_upper:
        num_modes = int(params.num_modes_lower)
    else:
        num_modes = int(rs.randint(int(params.num_modes_lower), int(params.num_modes_upper)))

    if num_modes == 0:
        # uniform distribution
        return _generate_tsp_instance_uniform(problem_size, rs)

    mix_proportion = rs.rand(num_modes)
    nums = rs.multinomial(int(problem_size), mix_proportion / float(mix_proportion.sum()))

    xy_parts = []
    for num in nums:
        num = int(num)
        if num <= 0:
            continue
        if params.no_cov:
            var = float(rs.uniform(params.var_lower, params.var_upper))
            cov = [[var, 0.0], [0.0, var]]
        else:
            var_x = float(rs.uniform(params.var_lower, params.var_upper))
            var_y = float(rs.uniform(params.var_lower, params.var_upper))
            cov_xy = float(rs.uniform(-np.sqrt(var_x * var_y), np.sqrt(var_x * var_y)))
            cov = [[var_x, cov_xy], [cov_xy, var_y]]

        center = rs.uniform(params.center_lower, params.center_upper, size=(1, 2))
        nxy = rs.multivariate_normal(mean=center.squeeze(), cov=cov, size=(num,))
        xy_parts.append(nxy)

    xy = np.concatenate(xy_parts, axis=0)
    xy = _minmax_scale(xy)
    return xy.astype("float32", copy=False)


def generate_tsp_batch(
    *,
    batch_size: int,
    problem_size: int,
    distributions: Sequence[str],
    gaussian_params: GaussianMixtureParams,
    seed: int,
    device: str = "cpu",
):
    """
    Generate a TSP batch with the same synthetic distribution logic as the paper's code.

    - If multiple `distributions` are provided, we sample one distribution *per instance* uniformly.
    - Supported dists (case-insensitive): "uniform", "gaussian" (Gaussian-mixture; includes uniform when num_modes=0).
    """
    import numpy as np
    import torch

    if not distributions:
        raise ValueError("distributions must be non-empty")

    rs = np.random.RandomState(int(seed))
    dists = tuple(_normalize_dist_name(d) for d in distributions)

    coords_np = np.empty((int(batch_size), int(problem_size), 2), dtype="float32")
    for i in range(int(batch_size)):
        dist = dists[0] if len(dists) == 1 else str(rs.choice(dists))
        if dist == "uniform":
            coords_np[i] = _generate_tsp_instance_uniform(problem_size, rs)
        elif dist == "gaussian":
            coords_np[i] = _generate_tsp_instance_gaussian(problem_size, gaussian_params, rs)
        else:
            raise ValueError(f"Unsupported TSP distribution: {dist}. Supported: uniform, gaussian")

    coords = torch.from_numpy(coords_np)
    return coords.to(device=device)


class TSPBatchGenerator:
    """
    Stateful infinite generator: call `.sample(...)` every step.
    """

    def __init__(
        self,
        *,
        problem_size: int,
        distributions: Sequence[str],
        gaussian_params: GaussianMixtureParams,
        seed: int,
    ):
        import numpy as np

        if not distributions:
            raise ValueError("distributions must be non-empty")
        self.problem_size = int(problem_size)
        self.dists = tuple(_normalize_dist_name(d) for d in distributions)
        self.params = gaussian_params
        self.rs = np.random.RandomState(int(seed))

    def sample(self, batch_size: int, *, device: str = "cpu"):
        import numpy as np
        import torch

        coords_np = np.empty((int(batch_size), self.problem_size, 2), dtype="float32")
        for i in range(int(batch_size)):
            dist = self.dists[0] if len(self.dists) == 1 else str(self.rs.choice(self.dists))
            if dist == "uniform":
                coords_np[i] = _generate_tsp_instance_uniform(self.problem_size, self.rs)
            elif dist == "gaussian":
                coords_np[i] = _generate_tsp_instance_gaussian(self.problem_size, self.params, self.rs)
            else:
                raise ValueError(f"Unsupported TSP distribution: {dist}. Supported: uniform, gaussian")

        coords = torch.from_numpy(coords_np)
        return coords.to(device=device)


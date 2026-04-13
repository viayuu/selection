from __future__ import annotations

from typing import Any, Literal

import torch
from tensordict import TensorDict
from torchrl.envs import EnvBase

from EasyNCO.neural_solvers.pipeline.iteration import Iteration


class InitOnlyIteration(Iteration):
    """
    EasyNCO evaluation helper: make *all* methods "initialization-only".

    Why it exists:
    - EasyNCO's `ARREINFORCELightning.test_step()` always calls `iteration.run(...)`.
    - For most methods, `initialization.run(..., phase='eval')` already returns a dict like:
        {'no_aug_score': <scalar>, 'aug_score': <scalar>}
      so a trivial `NoIteration` works.
    - But some methods return extra keys (e.g. LEHD adds 'batch') or a TensorDict
      (e.g. DACT returns a TensorDict with `current_length`), so `NoIteration`
      would break `score, aug_score = test_out.values()`.

    This Iteration implementation:
    - Never performs any improvement steps.
    - Converts whatever `initialization_out` is into exactly **two** scalar scores.
    """

    @staticmethod
    def _to_python_float(value: Any) -> float:
        if isinstance(value, float):
            return value
        if isinstance(value, (int, bool)):
            return float(value)
        if isinstance(value, torch.Tensor):
            if value.numel() == 1:
                return float(value.item())
            return float(value.float().mean().item())
        # Support basic scalar-like types (e.g., numpy scalars) without importing numpy.
        try:
            return float(value)
        except Exception as e:  # noqa: BLE001
            raise TypeError(f"Cannot convert to float: type={type(value)} value={value!r}") from e

    def run(
        self,
        td: TensorDict,
        env: EnvBase,
        initialization_out: Any,
        phase: Literal["train", "eval"],
        max_steps: int = 0,
        **kwargs,
    ) -> dict:
        if phase != "eval":
            raise NotImplementedError("InitOnlyIteration is intended for eval-only use.")

        # Case 1: dict-like output that already contains scores.
        if isinstance(initialization_out, dict):
            # Common patterns across methods.
            if "no_aug_score" in initialization_out and "aug_score" in initialization_out:
                return {
                    "no_aug_score": self._to_python_float(initialization_out["no_aug_score"]),
                    "aug_score": self._to_python_float(initialization_out["aug_score"]),
                }

            # HTSP uses different key names (and has a typo).
            if "score" in initialization_out:
                aug_key = "aug_score" if "aug_score" in initialization_out else "aug_socre"
                if aug_key in initialization_out:
                    return {
                        "no_aug_score": self._to_python_float(initialization_out["score"]),
                        "aug_score": self._to_python_float(initialization_out[aug_key]),
                    }

            raise ValueError(
                "InitOnlyIteration cannot extract scores from dict output. "
                f"keys={list(initialization_out.keys())}"
            )

        # Case 2: DACT initialization returns a TensorDict with `current_length`.
        if isinstance(initialization_out, TensorDict):
            if "current_length" not in initialization_out.keys():
                raise ValueError(
                    "InitOnlyIteration cannot extract scores from TensorDict output. "
                    f"keys={list(initialization_out.keys())}"
                )
            cur = initialization_out["current_length"]
            # `current_length` is per-instance cost; return batch mean for logging consistency.
            if isinstance(cur, torch.Tensor):
                score = cur.float().mean()
            else:
                score = torch.as_tensor(cur).float().mean()
            score_f = self._to_python_float(score)
            return {"no_aug_score": score_f, "aug_score": score_f}

        raise TypeError(
            "InitOnlyIteration expected dict or TensorDict from initialization, got "
            f"{type(initialization_out)}"
        )

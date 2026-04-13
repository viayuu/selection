#!/usr/bin/env python3
"""
Offline checkpoint conversion for EasyNCO (`final/`).

Goal:
  Convert an author-style checkpoint (e.g. ReLD Multi-Task `epoch-5000.pt` with `model_state_dict`)
  into an EasyNCO-friendly checkpoint containing `{"state_dict": ...}` whose keys match the
  platform policy implementation.

How it works:
  - Instantiate the policy from a settings YAML (Hydra-style `_target_`).
  - Load the source checkpoint using EasyNCO's online-compat loader (`EasyNCO.utils.utils.load_model`)
    ONCE (CPU), so key-mapping is resolved.
  - Save the resulting `policy.state_dict()` to a new file as `{"state_dict": ...}`.

Example (run from `final/`):
  # MOE_LIGHT variant (platform model name: mtreld)
  python convert_checkpoint_offline.py \
    --settings settings/mtreld_settings.yaml \
    --model-name mtreld \
    --src pretrained/pretrained/reld_moe_light/epoch-5000.pt \
    --dst pretrained/converted/mtreld_epoch-5000.pt

  # MTL variant (platform model name: mtreld_mtl)
  python convert_checkpoint_offline.py \
    --settings settings/mtreld_mtl_settings.yaml \
    --model-name mtreld_mtl \
    --src pretrained/pretrained/reld_mtl/epoch-5000.pt \
    --dst pretrained/converted/mtreld_mtl_epoch-5000.pt
"""

from __future__ import annotations

import argparse
import os
import sys
import types
from pathlib import Path


def _ensure_local_easynco(final_dir: Path) -> None:
    """Make `import EasyNCO.*` resolve to this `final/` folder."""
    local_pkg = types.ModuleType("EasyNCO")
    local_pkg.__path__ = [str(final_dir)]
    sys.modules["EasyNCO"] = local_pkg
    sys.path.insert(0, str(final_dir.parent))


def _resolve_settings_path(final_dir: Path, settings: str) -> Path:
    p = Path(settings)
    if p.is_absolute() and p.exists():
        return p
    # Prefer explicit relative path first (e.g. "settings/mtreld_settings.yaml")
    cand1 = (final_dir / p).resolve()
    if cand1.exists():
        return cand1
    # Fallback: treat as basename under settings/
    cand2 = (final_dir / "settings" / p.name).resolve()
    if cand2.exists():
        return cand2
    raise FileNotFoundError(f"Cannot find settings file: {settings!r} (tried {cand1} and {cand2})")


def _infer_model_name(settings_path: Path) -> str:
    name = settings_path.name.lower()
    if "mtreld_mtl" in name:
        return "mtreld_mtl"
    if "mtreld" in name:
        return "mtreld"
    return "unknown"


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert author checkpoints to EasyNCO platform format (offline).")
    parser.add_argument("--settings", default="settings/mtreld_settings.yaml", help="Settings YAML path (relative to final/).")
    parser.add_argument("--model-name", default=None, help="Model name passed to EasyNCO load_model (e.g. mtreld, mtreld_mtl).")
    parser.add_argument("--src", required=True, help="Source checkpoint path (author format).")
    parser.add_argument("--dst", default=None, help="Output checkpoint path (EasyNCO format).")
    args = parser.parse_args()

    final_dir = Path(__file__).resolve().parent
    _ensure_local_easynco(final_dir)

    settings_path = _resolve_settings_path(final_dir, args.settings)
    model_name = args.model_name or _infer_model_name(settings_path)

    src_path = Path(args.src)
    if not src_path.is_absolute():
        src_path = (Path.cwd() / src_path).resolve()
    if not src_path.exists():
        raise FileNotFoundError(f"Source checkpoint not found: {src_path}")

    if args.dst is None:
        out_dir = (final_dir / "pretrained" / "converted")
        out_name = f"{src_path.stem}.{model_name}.easynco.pt"
        dst_path = (out_dir / out_name).resolve()
    else:
        dst_path = Path(args.dst)
        if not dst_path.is_absolute():
            dst_path = (Path.cwd() / dst_path).resolve()

    dst_path.parent.mkdir(parents=True, exist_ok=True)

    # Heavy imports after path setup
    import torch
    import hydra
    from omegaconf import OmegaConf

    from EasyNCO.utils.utils import load_model

    if model_name == "nlns":
        raise ValueError("This script currently targets single-policy checkpoints (not NLNS multi-file checkpoints).")

    settings_cfg = OmegaConf.load(str(settings_path))
    if "model" not in settings_cfg:
        raise ValueError(f"Invalid settings file (missing `model` section): {settings_path}")

    policy = hydra.utils.instantiate(settings_cfg["model"])
    policy = load_model(policy, str(src_path), device=1, model_name=model_name)  # CPU

    # Save only tensors + plain metadata (avoid pickling code objects).
    state_dict = {k: v.detach().cpu() for k, v in policy.state_dict().items()}
    payload = {
        "state_dict": state_dict,
        "meta": {
            "source": str(src_path),
            "settings": os.path.relpath(str(settings_path), str(final_dir)),
            "model_name": model_name,
        },
    }
    torch.save(payload, str(dst_path))

    # Optional sanity check: strict load the saved state_dict back into a fresh policy.
    fresh = hydra.utils.instantiate(settings_cfg["model"])
    ckpt = torch.load(str(dst_path), map_location="cpu")
    res = fresh.load_state_dict(ckpt["state_dict"], strict=True)
    assert not getattr(res, "missing_keys", []), f"Missing keys after strict load: {res.missing_keys}"
    assert not getattr(res, "unexpected_keys", []), f"Unexpected keys after strict load: {res.unexpected_keys}"

    print(f"[OK] Converted: {src_path}")
    print(f"     -> {dst_path}")
    print(f"     keys: {len(state_dict)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


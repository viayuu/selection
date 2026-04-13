from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path


def _maybe_set_tmpdir():
    """
    Torch import may probe temp dirs (via `tempfile.gettempdir()`).
    In some environments `/tmp` is not writable, so we point TMPDIR to a workspace folder.
    """
    repo_root = Path(__file__).resolve().parents[1]
    tmpdir = repo_root / "my" / "outputs" / "tmp"
    os.environ.setdefault("TMPDIR", str(tmpdir))
    try:
        tmpdir.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass


@dataclass(frozen=True)
class FixedLEHDRRCTestConfig:
    seed: int = 2024
    device: str = "cpu"
    batch_size: int = 128
    decoder_strategy: str = "greedy"
    rrc_steps: int = 3

    data_path: str = ""
    out_path: str = ""
    max_instances: int = 0
    no_tour: bool = False

    model_dir: str = "model"
    lehd_ckpt: str = ""


def _load_tsp_coords_pt(path: str):
    import torch

    obj = torch.load(path, map_location="cpu")

    if isinstance(obj, dict):
        # EasyNCO TSP datasets commonly store coordinates under "node_xy".
        for k in ("node_xy", "coords", "locs", "problems", "data"):
            if k in obj:
                coords = obj[k]
                break
        else:
            raise KeyError(f"Unsupported .pt dict format (no coords key found). Keys={list(obj.keys())[:20]}")
    else:
        coords = obj

    if not isinstance(coords, torch.Tensor):
        raise TypeError(f"Unsupported coords type: {type(coords)} (expect torch.Tensor or dict containing tensor)")
    if coords.ndim != 3 or coords.size(-1) != 2:
        raise ValueError(f"Expected coords shape [num,N,2], got {tuple(coords.shape)}")

    return coords.float()


def main():
    _maybe_set_tmpdir()

    parser = argparse.ArgumentParser(
        description="Run a fixed (LEHD initializer + LEHD-RRC iterator) pipeline on a TSP .pt dataset to verify EasyNCO integration."
    )
    parser.add_argument("--data_path", type=str, required=True, help="TSP dataset .pt, e.g. final/data/datasets/test_dataset_tsp_uniform/test_tsp100_nums10000_uniform.pt")
    parser.add_argument("--out_path", type=str, default="", help="Output .txt path (optional)")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--max_instances", type=int, default=0, help="If >0, only run the first K instances")

    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--decoder_strategy", type=str, choices=["greedy", "sampling"], default="greedy")
    parser.add_argument("--rrc_steps", type=int, default=3)

    parser.add_argument("--model_dir", type=str, default="model")
    parser.add_argument("--lehd_ckpt", type=str, default="model/lehd_tsp100.ckpt")

    parser.add_argument("--no_tour", action="store_true", help="Do not print tour permutation in each line (shorter txt)")

    args = parser.parse_args()
    cfg = FixedLEHDRRCTestConfig(
        seed=int(args.seed),
        device=str(args.device),
        batch_size=int(args.batch_size),
        decoder_strategy=str(args.decoder_strategy),
        rrc_steps=int(args.rrc_steps),
        data_path=str(args.data_path),
        out_path=str(args.out_path),
        max_instances=int(args.max_instances),
        no_tour=bool(args.no_tour),
        model_dir=str(args.model_dir),
        lehd_ckpt=str(args.lehd_ckpt),
    )

    print("[config]", json.dumps(asdict(cfg), indent=2, ensure_ascii=False))

    # Local imports (after TMPDIR is set)
    import torch

    from easynco_bootstrap import ensure_local_easynco

    ensure_local_easynco()

    from my.train_two_gate_tsp import (
        TrainConfig,
        _build_initializer_zoo,
        _build_iterator_zoo,
        _configure_torch_defaults,
        _set_seed,
    )

    _configure_torch_defaults(cfg.device)
    _set_seed(cfg.seed)

    coords_all = _load_tsp_coords_pt(cfg.data_path)
    if cfg.max_instances and cfg.max_instances > 0:
        coords_all = coords_all[: int(cfg.max_instances)]

    problem_size = int(coords_all.size(1))

    # Reuse the same solver-builder logic as the training script, but fix the zoo.
    train_cfg = TrainConfig(
        seed=cfg.seed,
        device=cfg.device,
        problem_size=problem_size,
        batch_size=cfg.batch_size,
        decoder_strategy=cfg.decoder_strategy,
        rrc_steps=cfg.rrc_steps,
        init_zoo=("lehd",),
        iter_zoo=("rrc_lehd",),
        model_dir=None if cfg.model_dir.strip() == "" else cfg.model_dir.strip(),
        lehd_ckpt=None if cfg.lehd_ckpt.strip() == "" else cfg.lehd_ckpt.strip(),
    )

    init_zoo = _build_initializer_zoo(train_cfg)
    iter_zoo = _build_iterator_zoo(train_cfg)
    if len(init_zoo) != 1 or init_zoo[0].name != "lehd":
        raise RuntimeError(f"Unexpected initializer zoo: {[s.name for s in init_zoo]}")
    if len(iter_zoo) != 1 or not iter_zoo[0].name.startswith("rrc_lehd_"):
        raise RuntimeError(f"Unexpected iterator zoo: {[s.name for s in iter_zoo]}")

    initializer = init_zoo[0]
    iterator = iter_zoo[0]

    out_path = cfg.out_path.strip()
    if out_path == "":
        out_dir = Path("my") / "outputs"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = str(out_dir / (Path(cfg.data_path).stem + f"_lehd_rrc_n{problem_size}.txt"))
    else:
        p = Path(out_path)
        if p.suffix.lower() == ".pt":
            p = p.with_suffix(".txt")
        elif p.suffix == "":
            p = p.with_suffix(".txt")
        p.parent.mkdir(parents=True, exist_ok=True)
        out_path = str(p)

    device = torch.device(cfg.device)
    total_len0 = 0.0
    total_len1 = 0.0
    total_count = 0
    best_len1 = float("inf")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"# data_path={cfg.data_path}\n")
        f.write(f"# device={cfg.device} batch_size={cfg.batch_size}\n")
        f.write(f"# initializer={initializer.name} iterator={iterator.name} decoder_strategy={cfg.decoder_strategy} rrc_steps={cfg.rrc_steps}\n")
        f.write(f"# instances={int(coords_all.size(0))} N={problem_size}\n")
        if cfg.no_tour:
            f.write("# idx\tlen0\tlen1\n")
        else:
            f.write("# idx\tlen0\tlen1\ttour\n")

        with torch.no_grad():
            for start in range(0, int(coords_all.size(0)), int(cfg.batch_size)):
                batch = coords_all[start : start + int(cfg.batch_size)].to(device)

                sol0 = initializer.solve(batch)
                sol1 = iterator.improve(batch, sol0)

                len0_cpu = sol0.length.detach().cpu()
                len1_cpu = sol1.length.detach().cpu()
                tour1_cpu = sol1.tour.detach().cpu()

                total_len0 += float(len0_cpu.sum().item())
                total_len1 += float(len1_cpu.sum().item())
                total_count += int(len1_cpu.numel())
                best_len1 = min(best_len1, float(len1_cpu.min().item()))

                for j in range(int(len1_cpu.size(0))):
                    idx = start + j
                    if cfg.no_tour:
                        f.write(f"{idx}\t{len0_cpu[j].item():.6f}\t{len1_cpu[j].item():.6f}\n")
                    else:
                        tour_str = " ".join(str(x) for x in tour1_cpu[j].tolist())
                        f.write(f"{idx}\t{len0_cpu[j].item():.6f}\t{len1_cpu[j].item():.6f}\t{tour_str}\n")

        mean_len0 = total_len0 / max(total_count, 1)
        mean_len1 = total_len1 / max(total_count, 1)
        f.write(f"# mean_len0={mean_len0:.6f} mean_len1={mean_len1:.6f} best_len1={best_len1:.6f}\n")

    print(f"[test] instances={total_count} N={problem_size} mean_len0={mean_len0:.6f} mean_len1={mean_len1:.6f} best_len1={best_len1:.6f}")
    print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()

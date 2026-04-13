from __future__ import annotations

import argparse
import os
from pathlib import Path


def _maybe_set_tmpdir():
    """
    Torch import may probe temp dirs (via `tempfile.gettempdir()`).
    In some sandboxed environments `/tmp` is not writable, so we point TMPDIR to a workspace folder.
    """
    repo_root = Path(__file__).resolve().parents[1]
    tmpdir = repo_root / "my" / "outputs" / "tmp"
    os.environ.setdefault("TMPDIR", str(tmpdir))
    try:
        tmpdir.mkdir(parents=True, exist_ok=True)
    except Exception:
        # If the filesystem is read-only, user must create it manually.
        pass


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


def _as_tuple(xs):
    if xs is None:
        return None
    if isinstance(xs, tuple):
        return xs
    if isinstance(xs, list):
        return tuple(xs)
    return tuple(xs)


def main():
    _maybe_set_tmpdir()

    parser = argparse.ArgumentParser()
    parser.add_argument("--gate_ckpt", type=str, required=True, help="Trained gate checkpoint (.pt), e.g. my/outputs/two_gate_tsp_seed2024_n50.pt")
    parser.add_argument("--data_path", type=str, required=True, help="TSP dataset .pt, e.g. final/data/datasets/test_dataset_tsp_uniform/test_tsp100_nums10000_uniform.pt")
    parser.add_argument("--out_path", type=str, default="", help="Output .txt path (optional)")
    parser.add_argument("--no_tour", action="store_true", help="Do not print the tour permutation in each line (shorter txt)")

    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--max_instances", type=int, default=0, help="If >0, only solve the first K instances")

    # Zoo override (if not provided, we use the order saved in the gate checkpoint cfg)
    parser.add_argument("--init_zoo", nargs="+", default=None)
    parser.add_argument("--iter_zoo", nargs="+", default=None)

    # Solver options (same semantics as train script; only the ones needed for inference)
    parser.add_argument("--model_dir", type=str, default="model")
    parser.add_argument("--decoder_strategy", type=str, choices=["greedy", "sampling"], default="greedy")
    parser.add_argument("--pomo_size", type=int, default=0, help="POMO parallel starts; 0 means problem_size")
    parser.add_argument("--rrc_steps", type=int, default=3)
    parser.add_argument("--dact_steps", type=int, default=50)
    parser.add_argument("--lih_steps", type=int, default=50)
    parser.add_argument("--two_opt_iters", type=int, default=200)

    parser.add_argument("--pomo_ckpt", type=str, default="")
    parser.add_argument("--am_ckpt", type=str, default="")
    parser.add_argument("--lehd_ckpt", type=str, default="")
    parser.add_argument("--elg_ckpt", type=str, default="")
    parser.add_argument("--difusco_ckpt", type=str, default="")
    parser.add_argument("--dact_ckpt", type=str, default="")
    parser.add_argument("--lih_ckpt", type=str, default="")

    args = parser.parse_args()

    # Local imports (after TMPDIR is set)
    import torch

    # `ensure_local_easynco()` injects the repo root into `sys.path`,
    # so `import my.*` works even when this script is executed as `python my/solve_dataset_tsp.py`.
    from easynco_bootstrap import ensure_local_easynco

    ensure_local_easynco()

    from my.features import tsp_gate2_features, tsp_instance_features
    from my.solver_zoo import run_initializers_by_action, run_iterators_by_action
    from my.train_two_gate_tsp import (
        TrainConfig,
        _build_gates,
        _build_initializer_zoo,
        _build_iterator_zoo,
        _configure_torch_defaults,
    )

    _configure_torch_defaults(args.device)

    gate_ckpt = torch.load(args.gate_ckpt, map_location=args.device)
    ckpt_cfg = gate_ckpt.get("cfg", {}) if isinstance(gate_ckpt, dict) else {}

    coords_all = _load_tsp_coords_pt(args.data_path)
    if args.max_instances and args.max_instances > 0:
        coords_all = coords_all[: int(args.max_instances)]

    problem_size = int(coords_all.size(1))
    batch_size = int(args.batch_size)

    init_zoo = args.init_zoo if args.init_zoo is not None else ckpt_cfg.get("init_zoo")
    iter_zoo = args.iter_zoo if args.iter_zoo is not None else ckpt_cfg.get("iter_zoo")
    if init_zoo is None or iter_zoo is None:
        raise ValueError("Missing init_zoo/iter_zoo: pass --init_zoo/--iter_zoo, or use a gate checkpoint that contains cfg.init_zoo/cfg.iter_zoo.")

    cfg = TrainConfig(
        device=args.device,
        problem_size=problem_size,
        batch_size=batch_size,
        init_zoo=_as_tuple(init_zoo),
        iter_zoo=_as_tuple(iter_zoo),
        decoder_strategy=args.decoder_strategy,
        pomo_size=None if args.pomo_size == 0 else int(args.pomo_size),
        rrc_steps=int(args.rrc_steps),
        dact_steps=int(args.dact_steps),
        lih_steps=int(args.lih_steps),
        two_opt_iters=int(args.two_opt_iters),
        model_dir=None if args.model_dir.strip() == "" else args.model_dir.strip(),
        pomo_ckpt=None if args.pomo_ckpt.strip() == "" else args.pomo_ckpt.strip(),
        am_ckpt=None if args.am_ckpt.strip() == "" else args.am_ckpt.strip(),
        lehd_ckpt=None if args.lehd_ckpt.strip() == "" else args.lehd_ckpt.strip(),
        elg_ckpt=None if args.elg_ckpt.strip() == "" else args.elg_ckpt.strip(),
        difusco_ckpt=None if args.difusco_ckpt.strip() == "" else args.difusco_ckpt.strip(),
        dact_ckpt=None if args.dact_ckpt.strip() == "" else args.dact_ckpt.strip(),
        lih_ckpt=None if args.lih_ckpt.strip() == "" else args.lih_ckpt.strip(),
        # Baseline doesn't affect inference; keep it minimal.
        baseline="batch_mean",
    )

    # Build solvers (frozen) + gates
    init_solvers = _build_initializer_zoo(cfg)
    iter_solvers = _build_iterator_zoo(cfg)
    policy, _opt, feat_cfg = _build_gates(cfg, num_init=len(init_solvers), num_iter=len(iter_solvers))

    if isinstance(gate_ckpt, dict):
        if gate_ckpt.get("encoder") is not None and policy.encoder is not None:
            policy.encoder.load_state_dict(gate_ckpt["encoder"])
        policy.gate1.load_state_dict(gate_ckpt["gate1"])
        policy.gate2.load_state_dict(gate_ckpt["gate2"])

    policy.eval()

    device = torch.device(args.device)
    a1_all = []
    a2_all = []
    # NOTE: `my/train_two_gate_tsp.py:_configure_torch_defaults()` may set the global default
    # tensor type to CUDA, which can make even `dtype=torch.long` tensors land on GPU.
    # We keep the counters explicitly on CPU since they are only used for logging/output.
    a1_counts = torch.zeros((len(init_solvers),), dtype=torch.long, device="cpu")
    a2_counts = torch.zeros((len(iter_solvers),), dtype=torch.long, device="cpu")
    total_len = 0.0
    best_len = float("inf")
    total_count = 0

    out_path = args.out_path.strip()
    if out_path == "":
        out_dir = Path("my") / "outputs"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = str(out_dir / (Path(args.data_path).stem + f"_solved_n{problem_size}.txt"))
    else:
        p = Path(out_path)
        if p.suffix.lower() == ".pt":
            p = p.with_suffix(".txt")
        elif p.suffix == "":
            p = p.with_suffix(".txt")
        p.parent.mkdir(parents=True, exist_ok=True)
        out_path = str(p)

    init_names = [s.name for s in init_solvers]
    iter_names = [s.name for s in iter_solvers]

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"# data_path={args.data_path}\n")
        f.write(f"# gate_ckpt={args.gate_ckpt}\n")
        f.write(f"# instances={int(coords_all.size(0))} N={problem_size}\n")
        f.write(f"# init_zoo={init_names}\n")
        f.write(f"# iter_zoo={iter_names}\n")
        if args.no_tour:
            f.write("# idx\tlen0\tlen1\tinit\titer\n")
        else:
            f.write("# idx\tlen0\tlen1\tinit\titer\ttour\n")

        with torch.no_grad():
            for start in range(0, int(coords_all.size(0)), batch_size):
                batch = coords_all[start : start + batch_size].to(device)

                feat1 = tsp_instance_features(batch, cfg=feat_cfg, encoder=policy.encoder)
                a1 = policy.gate1(feat1).argmax(dim=1)
                sol0 = run_initializers_by_action(batch, a1, init_solvers)

                feat2 = tsp_gate2_features(batch, sol0.tour, sol0.length, cfg=feat_cfg, encoder=policy.encoder)
                a2 = policy.gate2(feat2).argmax(dim=1)
                sol1 = run_iterators_by_action(batch, a2, sol0, iter_solvers)

                a1_cpu = a1.detach().cpu()
                a2_cpu = a2.detach().cpu()
                a1_all.append(a1_cpu)
                a2_all.append(a2_cpu)
                a1_counts += torch.bincount(a1_cpu, minlength=len(init_names))
                a2_counts += torch.bincount(a2_cpu, minlength=len(iter_names))

                len0_cpu = sol0.length.detach().cpu()
                len1_cpu = sol1.length.detach().cpu()
                tour1_cpu = sol1.tour.detach().cpu()

                total_len += float(len1_cpu.sum().item())
                total_count += int(len1_cpu.numel())
                best_len = min(best_len, float(len1_cpu.min().item()))

                for j in range(int(len1_cpu.size(0))):
                    idx = start + j
                    init_name = init_names[int(a1_cpu[j].item())]
                    iter_name = iter_names[int(a2_cpu[j].item())]
                    if args.no_tour:
                        f.write(f"{idx}\t{len0_cpu[j].item():.6f}\t{len1_cpu[j].item():.6f}\t{init_name}\t{iter_name}\n")
                    else:
                        tour_str = " ".join(str(x) for x in tour1_cpu[j].tolist())
                        f.write(
                            f"{idx}\t{len0_cpu[j].item():.6f}\t{len1_cpu[j].item():.6f}\t{init_name}\t{iter_name}\t{tour_str}\n"
                        )

        mean_len = total_len / max(total_count, 1)
        p1 = (a1_counts.float() / max(float(a1_counts.sum().item()), 1.0)).tolist()
        p2 = (a2_counts.float() / max(float(a2_counts.sum().item()), 1.0)).tolist()
        p1_str = ", ".join(f"{init_names[i]}:{p1[i]:.3f}" for i in range(len(init_names)))
        p2_str = ", ".join(f"{iter_names[i]}:{p2[i]:.3f}" for i in range(len(iter_names)))

        f.write(f"# mean_len={mean_len:.6f} best_len={best_len:.6f}\n")
        f.write(f"# gate1=({p1_str})\n")
        f.write(f"# gate2=({p2_str})\n")

    print(f"[solve] instances={total_count} N={problem_size} mean_len={mean_len:.6f} best_len={best_len:.6f}")
    print(f"[solve] gate1=({p1_str})")
    print(f"[solve] gate2=({p2_str})")
    print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()

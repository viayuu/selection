import argparse

import torch
import torch.nn.functional as F

from code.unified_selector.registry import PROBLEMS

from .V4Model import ProblemToSolverSelector, get_default_model_params
from .train import make_loader, to_device


def build_small_params(args):
    params = get_default_model_params()
    params.update(
        embedding_dim=args.d,
        head_num=args.heads,
        qkv_dim=args.d // args.heads,
        ff_hidden_dim=args.ff_hidden,
        head_hidden_dim=args.head_hidden,
        encoder_layer_num=args.encoder_layers,
        solver_set_layer_num=args.set_layers,
        query_num=args.query_num,
        dropout=0.0,
        support_branch=args.support_branch,
        support_generators=args.support_generators if args.support_branch else 0,
        support_hidden=args.support_hidden,
        support_score_weight=args.support_score_weight if args.support_branch else 0.0,
        support_score_mode=args.support_score_mode,
    )
    return params


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--amp-dtype", choices=["fp16", "bf16"], default="fp16")
    parser.add_argument("--problems", default="")
    parser.add_argument("--d", type=int, default=128)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--ff-hidden", type=int, default=512)
    parser.add_argument("--head-hidden", type=int, default=256)
    parser.add_argument("--encoder-layers", type=int, default=4)
    parser.add_argument("--set-layers", type=int, default=2)
    parser.add_argument("--query-num", type=int, default=4)
    parser.add_argument("--support-branch", action="store_true")
    parser.add_argument("--support-generators", type=int, default=4)
    parser.add_argument("--support-hidden", type=int, default=128)
    parser.add_argument("--support-score-weight", type=float, default=0.10)
    parser.add_argument("--support-score-mode", choices=["g0", "max"], default="g0")
    args = parser.parse_args()

    problems = [p.strip() for p in args.problems.split(",") if p.strip()] if args.problems else list(PROBLEMS)
    device = torch.device(args.device)
    amp_dtype = torch.float16 if args.amp_dtype == "fp16" else torch.bfloat16
    model = ProblemToSolverSelector(**build_small_params(args)).to(device).train()

    for problem in problems:
        loader = make_loader(problem, "train", args.batch_size, args.num_workers, coord_augment=0, shuffle=False)
        batch = next(iter(loader))
        assert "problem_desc" in batch, f"{problem}: missing problem_desc"
        assert batch["costs"].shape[1] == batch["pool_ids"].numel(), (
            problem,
            tuple(batch["costs"].shape),
            tuple(batch["pool_ids"].shape),
        )
        batch = to_device(batch, device)
        with torch.cuda.amp.autocast(enabled=args.amp, dtype=amp_dtype):
            out = model(batch)
            loss = F.cross_entropy(out["logits"], batch["costs"].argmin(dim=1))
        for key in ("logits", "pre_score", "pred_gap", "utility"):
            assert torch.isfinite(out[key]).all(), f"{problem}: {key} has non-finite values"
        if args.support_branch:
            assert out["support_logits"] is not None, f"{problem}: missing support_logits"
            assert out["support_logits"].shape[:2] == (batch["costs"].size(0), args.support_generators)
            assert out["support_logits"].shape[2] == batch["costs"].size(1)
            assert torch.isfinite(out["support_logits"]).all(), f"{problem}: support_logits has non-finite values"
        loss.backward()
        model.zero_grad(set_to_none=True)
        print(
            f"{problem:>10} OK costs={tuple(batch['costs'].shape)} "
            f"pool={tuple(batch['pool_ids'].shape)} loss={float(loss.detach().cpu()):.4f}"
        )

    print("ALL PROBLEMS OK")


if __name__ == "__main__":
    main()

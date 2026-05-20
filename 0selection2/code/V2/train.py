import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from code.unified_selector.data import UnifiedProblemDataset, collate_single_problem
from code.unified_selector.registry import PROBLEMS

from .V2Model import SolverConditionedSelector, get_default_model_params


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def configure_torch():
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass


def to_device(batch, device):
    out = {}
    for k, v in batch.items():
        out[k] = v.to(device, non_blocking=True) if torch.is_tensor(v) else v
    return out


def make_loader(problem, split, batch_size, num_workers, coord_augment=0, shuffle=True):
    dataset = UnifiedProblemDataset(problem, split, coord_augment=coord_augment)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_single_problem,
        drop_last=(split == "train"),
        pin_memory=(num_workers > 0),
        persistent_workers=(num_workers > 0),
    )


def get_split_sbs_vbs(problem, split):
    dataset = UnifiedProblemDataset(problem, split, coord_augment=0)
    costs = []
    for i in range(dataset.base_N):
        costs.append(np.asarray(dataset.labels[str(i)]["cost"][:dataset.K_p], dtype=np.float32))
    costs = np.stack(costs, axis=0)
    method_mean = costs.mean(axis=0)
    return float(method_mean.min()), float(costs.min(axis=1).mean())


def pairwise_order_loss(score, costs):
    # score high is good, cost low is good.
    diff_score = score[:, :, None] - score[:, None, :]
    diff_cost = costs[:, None, :] - costs[:, :, None]
    sign = diff_cost.sign()
    margin = diff_cost.abs() / (costs.abs().mean(dim=1, keepdim=True)[:, None, :] + 1.0e-9)
    weight = margin.clamp(0.05, 5.0)
    valid = sign != 0
    loss = F.softplus(-sign * diff_score) * weight
    return loss[valid].mean() if valid.any() else score.sum() * 0


def gap_loss(gap_pred, costs):
    best = costs.min(dim=1, keepdim=True).values
    target = (costs - best) / (best.abs() + 1.0e-9)
    return F.huber_loss(gap_pred, target.clamp(max=1.0), delta=0.05)


def risk_loss(score, costs):
    prob = F.softmax(score, dim=1)
    pred_cost = (prob * costs).sum(dim=1)
    best = costs.min(dim=1).values
    return ((pred_cost - best) / (best.abs() + 1.0e-9)).clamp(max=1.0).mean()


def selector_loss(score, gap_pred, costs, ce_weight, pair_weight, gap_weight, risk_weight):
    winner = costs.argmin(dim=1)
    ce = F.cross_entropy(score, winner)
    pair = pairwise_order_loss(score, costs)
    gap = gap_loss(gap_pred, costs)
    risk = risk_loss(score, costs)
    loss = ce_weight * ce + pair_weight * pair + gap_weight * gap + risk_weight * risk
    return loss, {"ce": ce.item(), "pair": pair.item(), "gap": gap.item(), "risk": risk.item()}


@torch.no_grad()
def evaluate(model, problems, split, batch_size, num_workers, device):
    model.eval()
    per_problem = {}
    for problem in problems:
        loader = make_loader(problem, split, batch_size, num_workers, coord_augment=0, shuffle=False)
        top1 = top2 = top3 = n_total = 0
        cost_sum = 0.0
        pick_count = None
        for batch in loader:
            batch = to_device(batch, device)
            score, _ = model(batch)
            costs = batch["costs"]
            true_rank = torch.argsort(costs, dim=1)
            pred_rank = torch.argsort(-score, dim=1)
            best = true_rank[:, 0]
            top1 += (pred_rank[:, 0] == best).sum().item()
            top2 += (pred_rank[:, : min(2, score.size(1))] == best[:, None]).any(dim=1).sum().item()
            top3 += (pred_rank[:, : min(3, score.size(1))] == best[:, None]).any(dim=1).sum().item()
            pred = pred_rank[:, 0]
            cost_sum += costs.gather(1, pred[:, None]).sum().item()
            n_total += costs.size(0)
            cur_count = torch.bincount(pred.detach().cpu(), minlength=score.size(1)).float()
            pick_count = cur_count if pick_count is None else pick_count + cur_count
        sbs, vbs = get_split_sbs_vbs(problem, split)
        mean_cost = cost_sum / max(1, n_total)
        pick_dist = (pick_count / pick_count.sum().clamp_min(1)).tolist()
        per_problem[problem] = dict(
            top1=top1 / n_total,
            top2=top2 / n_total,
            top3=top3 / n_total,
            mean_cost=mean_cost,
            sbs=sbs,
            vbs=vbs,
            vs_sbs_pct=(mean_cost - sbs) / (abs(sbs) + 1.0e-9) * 100,
            vbs_gap_closed_pct=((sbs - mean_cost) / (sbs - vbs + 1.0e-9)) * 100 if sbs != vbs else 0.0,
            pick_dist=pick_dist,
            n=n_total,
        )
    macro = dict(
        macro_top1=float(np.mean([v["top1"] for v in per_problem.values()])),
        macro_top2=float(np.mean([v["top2"] for v in per_problem.values()])),
        macro_top3=float(np.mean([v["top3"] for v in per_problem.values()])),
        macro_vs_sbs_pct=float(np.mean([v["vs_sbs_pct"] for v in per_problem.values()])),
        macro_vbs_gap_closed_pct=float(np.mean([v["vbs_gap_closed_pct"] for v in per_problem.values()])),
    )
    model.train()
    return per_problem, macro


def print_eval(tag, per_problem, macro):
    print(
        f"[{tag}] macro_top1={macro['macro_top1']:.4f} "
        f"top2={macro['macro_top2']:.4f} vs_sbs={macro['macro_vs_sbs_pct']:+.3f}% "
        f"vbs_closed={macro['macro_vbs_gap_closed_pct']:+.2f}%"
    )
    for p in per_problem:
        r = per_problem[p]
        print(
            f"{p:>10}: top1={r['top1']:.3f} mean_cost={r['mean_cost']:.4f} "
            f"(sbs={r['sbs']:.4f}) vs_sbs={r['vs_sbs_pct']:+.2f}%"
        )


def build_model(args):
    params = get_default_model_params()
    params.update(
        embedding_dim=args.d,
        head_num=args.heads,
        qkv_dim=args.d // args.heads,
        ff_hidden_dim=args.ff_hidden,
        head_hidden_dim=args.head_hidden,
        encoder_layer_num=args.encoder_layers,
        cross_layer_num=args.cross_layers,
        solver_set_layer_num=args.set_layers,
        gap_score_weight=args.gap_score_weight,
    )
    return SolverConditionedSelector(**params), params


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-dir", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-per-problem", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2.0e-4)
    parser.add_argument("--wd", type=float, default=1.0e-4)
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--coord-augment", type=int, default=0)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--problems", default="")
    parser.add_argument("--d", type=int, default=128)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--ff-hidden", type=int, default=256)
    parser.add_argument("--head-hidden", type=int, default=256)
    parser.add_argument("--encoder-layers", type=int, default=3)
    parser.add_argument("--cross-layers", type=int, default=2)
    parser.add_argument("--set-layers", type=int, default=1)
    parser.add_argument("--ce-weight", type=float, default=0.35)
    parser.add_argument("--pair-weight", type=float, default=0.30)
    parser.add_argument("--gap-weight", type=float, default=0.25)
    parser.add_argument("--risk-weight", type=float, default=0.10)
    parser.add_argument("--gap-score-weight", type=float, default=0.0)
    args = parser.parse_args()

    set_seed(args.seed)
    configure_torch()
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    problems = [p.strip() for p in args.problems.split(",") if p.strip()] if args.problems else list(PROBLEMS)

    model, model_params = build_model(args)
    model.to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    train_loaders = {
        p: make_loader(p, "train", args.batch_per_problem, args.num_workers, args.coord_augment, shuffle=True)
        for p in problems
    }

    config = vars(args)
    config["model_params"] = model_params
    (save_dir / "args.json").write_text(json.dumps(config, indent=2))
    best_score = -1.0e9
    global_step = 0
    start = time.time()
    print(f"[train] problems={problems}")
    print(f"[train] model_params={model_params}")

    for epoch in range(args.epochs):
        order = list(problems)
        random.shuffle(order)
        loss_meter = []
        part_meter = {"ce": [], "pair": [], "gap": [], "risk": []}
        for p in order:
            for batch in train_loaders[p]:
                batch = to_device(batch, args.device)
                score, gap_pred = model(batch)
                loss, parts = selector_loss(
                    score,
                    gap_pred,
                    batch["costs"],
                    ce_weight=args.ce_weight,
                    pair_weight=args.pair_weight,
                    gap_weight=args.gap_weight,
                    risk_weight=args.risk_weight,
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                global_step += 1
                loss_meter.append(loss.item())
                for k, v in parts.items():
                    part_meter[k].append(v)
                if global_step % 50 == 0:
                    msg = " ".join([f"{k}={np.mean(v):.4f}" for k, v in part_meter.items() if v])
                    print(f"ep{epoch:03d} step{global_step:06d} loss={np.mean(loss_meter):.4f} {msg} time={(time.time()-start)/60:.1f}m")

        if epoch % args.eval_every == 0:
            per_problem, macro = evaluate(model, problems, "val", args.batch_per_problem, args.num_workers, args.device)
            print_eval(f"eval epoch {epoch}", per_problem, macro)
            (save_dir / f"eval_epoch{epoch}.json").write_text(json.dumps({"per_problem": per_problem, "macro": macro}, indent=2))
            score = macro["macro_top1"] - 0.25 * max(0.0, macro["macro_vs_sbs_pct"])
            if score > best_score:
                best_score = score
                torch.save({"model": model.state_dict(), "args": config, "epoch": epoch, "macro": macro}, save_dir / "best.pt")
                print(f"[save] best.pt epoch={epoch} score={score:.4f}")

    per_problem, macro = evaluate(model, problems, "test", args.batch_per_problem, args.num_workers, args.device)
    print_eval("test final", per_problem, macro)
    (save_dir / "test_final.json").write_text(json.dumps({"per_problem": per_problem, "macro": macro}, indent=2))
    torch.save({"model": model.state_dict(), "args": config, "epoch": args.epochs - 1, "macro": macro}, save_dir / "last.pt")


if __name__ == "__main__":
    main()

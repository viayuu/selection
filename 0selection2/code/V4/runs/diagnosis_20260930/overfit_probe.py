"""Small training-only capacity probe, not a new production experiment."""
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import wandb

from code.V4.evaluate import load_model
from code.V4.train import configure_torch
from code.unified_selector.data import UnifiedProblemDataset, collate_single_problem


def main():
    root = Path(__file__).resolve().parent
    torch.set_num_threads(1)
    configure_torch()
    records = {}
    run = wandb.init(project="selector", entity="yjkds-southern-university-of-science-technology",
                     name="R33a_diagnostic_overfit64", mode="offline", dir=str(root),
                     config=dict(samples_per_problem=64, lr=2e-4, max_steps=800,
                                 loss="unweighted CE", dropout=False, validation=False))
    for problem in ("TSP", "CVRP", "VRPTW"):
        torch.manual_seed(29)
        model, _ = load_model(root.parent / "R33a_dualstream_seed2_4090/best.pt", "cuda:0")
        # Eval mode disables dropout, but autograd remains enabled.
        model.eval()
        ds = UnifiedProblemDataset(problem, "train")
        ids = np.random.default_rng(29).choice(len(ds), 64, replace=False)
        batch = collate_single_problem([ds[i] for i in ids])
        batch = {k: v.cuda() if torch.is_tensor(v) else v for k, v in batch.items()}
        target = batch["costs"].argmin(1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0)
        history = []
        for step in range(801):
            logits = model(batch)["logits"]
            loss = F.cross_entropy(logits, target)
            if step % 50 == 0:
                accuracy = logits.argmax(1).eq(target).float().mean().item()
                history.append(dict(step=step, loss=loss.item(), train_top1=accuracy))
                print(problem, history[-1], flush=True)
                run.log({f"{problem}/ce": loss.item(), f"{problem}/top1": accuracy, f"{problem}/step": step})
                if accuracy == 1.0 or step == 800:
                    break
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
        records[problem] = dict(indices=ids.tolist(), history=history)
        (root / "overfit_probe.json").write_text(json.dumps(records, indent=2) + "\n")
        del model, optimizer, ds, batch
    run.finish()


if __name__ == "__main__":
    main()

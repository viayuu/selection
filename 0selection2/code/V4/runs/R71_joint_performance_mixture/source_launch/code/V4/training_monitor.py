"""Resume state and plots for controlled training experiments."""

import random

import numpy as np
import torch


def capture_rng():
    return dict(python=random.getstate(), numpy=np.random.get_state(),
                torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"]:
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])


def plot_history(history, directory):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for key in ("loss", "ce", "pair", "risk"):
        points = [r for r in history if key in r["train"]]
        if points:
            axes[0, 0].plot([r["epoch"] + 1 for r in points], [r["train"][key] for r in points], label=key)
    axes[0, 0].set_title("Training objective components"); axes[0, 0].legend()
    for source in ("train_eval", "val"):
        rows = [r for r in history if source in r]
        for ax, metric in ((axes[0, 1], "macro_ce"), (axes[0, 2], "macro_top1"), (axes[1, 0], "macro_vs_sbs_pct")):
            ax.plot([r["epoch"] + 1 for r in rows], [r[source][metric] for r in rows], label=source)
            ax.set_title(metric); ax.legend()
    axes[1, 1].plot([r["epoch"] + 1 for r in history], [r["lr"] for r in history]); axes[1, 1].set_title("Learning rate")
    axes[1, 2].plot([r["epoch"] + 1 for r in history], [r["train"]["grad_norm"] for r in history]); axes[1, 2].set_title("Unclipped gradient norm")
    for ax in axes.flat:
        ax.set_xlabel("Epoch (1-based)"); ax.grid(alpha=.25)
    fig.tight_layout(); fig.savefig(directory / "training_curves.png", dpi=150); plt.close(fig)

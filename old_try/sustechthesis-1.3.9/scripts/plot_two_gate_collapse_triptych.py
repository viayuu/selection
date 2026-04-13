#!/usr/bin/env python3
from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


TRAIN_PATTERN = re.compile(
    r"\[train step\s+(\d+)\]\s+"
    r"loss=([\d\.\-]+)\s+"
    r"len0=([\d\.\-]+)\s+"
    r"len1=([\d\.\-]+)\s+"
    r"adv=([\d\.\-]+)±([\d\.\-]+)\s+"
    r"critic_mse=([\d\.\-]+)\s+"
    r"entropy=([\d\.\-]+)\s+"
    r"gate1=\((.*?)\)\s+"
    r"gate2=\((.*?)\)"
)

EVAL_PATTERN = re.compile(
    r"\[eval step\s+(\d+)\]\s+"
    r"mean_len=([\d\.\-]+)\s+"
    r"best=([\d\.\-]+)\s+"
    r"gate1=\((.*?)\)\s+"
    r"gate2=\((.*?)\)"
)


def parse_distribution(dist_str: str) -> dict[str, float]:
    result: dict[str, float] = {}
    for item in dist_str.split(", "):
        if ":" not in item:
            continue
        key, value = item.split(":")
        result[key] = float(value)
    return result


def load_config(lines: list[str]) -> dict:
    config_start = -1
    for i, line in enumerate(lines):
        if line.strip() == "[config] {":
            config_start = i
            break
    if config_start < 0:
        return {}

    config_lines: list[str] = []
    brace_count = 0
    for i in range(config_start, len(lines)):
        config_lines.append(lines[i])
        brace_count += lines[i].count("{") - lines[i].count("}")
        if brace_count == 0:
            break
    config_json = "".join(config_lines).replace("[config] ", "")
    return json.loads(config_json)


def pretty_label(name: str) -> str:
    name = name.replace("_10", "")
    name = name.replace("_200", "")
    return name


def plot_probability_panel(
    ax: plt.Axes,
    steps: list[int],
    dists: list[dict[str, float]],
    options: list[str],
    title: str,
    color_map: dict[str, str],
) -> None:
    for option in options:
        values = [dist.get(option, 0.0) for dist in dists]
        ax.plot(
            steps,
            values,
            linewidth=2.2,
            label=pretty_label(option),
            color=color_map.get(option),
        )
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Train step")
    ax.set_ylabel("Selection probability")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=9, loc="center right")


def main() -> None:
    repo_root = Path("/public/home/zhoucl/shiys")
    log_path = repo_root / "1two_gate/outputs/two_gate_tsp_seed2024_n100.log.txt"
    output_path = (
        repo_root
        / "sustechthesis-1.3.9/figures/research/figure-05-two-gate-collapse-triptych.png"
    )

    lines = log_path.read_text(encoding="utf-8").splitlines()
    config = load_config([line + "\n" for line in lines])

    train_steps: list[int] = []
    train_gate1: list[dict[str, float]] = []
    train_gate2: list[dict[str, float]] = []
    eval_steps: list[int] = []
    eval_mean_len: list[float] = []

    for line in lines:
        train_match = TRAIN_PATTERN.search(line)
        if train_match:
            train_steps.append(int(train_match.group(1)))
            train_gate1.append(parse_distribution(train_match.group(9)))
            train_gate2.append(parse_distribution(train_match.group(10)))
            continue

        eval_match = EVAL_PATTERN.search(line)
        if eval_match:
            eval_steps.append(int(eval_match.group(1)))
            eval_mean_len.append(float(eval_match.group(2)))

    init_options = list(config.get("init_zoo", []))
    iter_options = [
        f"{name}_{config['rrc_steps']}" if name == "rrc_lehd" else f"{name}_{config['two_opt_iters']}" if name == "two_opt" else name
        for name in config.get("iter_zoo", [])
    ]

    init_colors = {
        "lehd": "#4C78A8",
        "elg": "#F58518",
        "difusco": "#54A24B",
    }
    iter_colors = {
        "none": "#E45756",
        "rrc_lehd_10": "#72B7B2",
        "two_opt_10": "#B279A2",
    }

    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(1, 3, figsize=(15.6, 4.7))

    axes[0].plot(
        eval_steps,
        eval_mean_len,
        color="#2F5BEA",
        marker="o",
        markersize=4.5,
        linewidth=2.4,
    )
    axes[0].set_title("(a) Evaluation mean length", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Train step")
    axes[0].set_ylabel("Mean length")
    axes[0].grid(True, alpha=0.25)
    if eval_mean_len:
        y_min = min(eval_mean_len)
        y_max = max(eval_mean_len)
        pad = max(0.002, (y_max - y_min) * 0.15)
        axes[0].set_ylim(y_min - pad, y_max + pad)

    plot_probability_panel(
        axes[1],
        train_steps,
        train_gate1,
        init_options,
        "(b) Initializer selection probabilities",
        init_colors,
    )
    plot_probability_panel(
        axes[2],
        train_steps,
        train_gate2,
        iter_options,
        "(c) Iterator selection probabilities",
        iter_colors,
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")


if __name__ == "__main__":
    main()

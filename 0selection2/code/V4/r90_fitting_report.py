"""CPU-only fitting report from the ten explicitly allowed R84/R87/R88 JSONs."""

from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
import statistics
import tempfile


ROOT = Path(__file__).resolve().parents[2]
RUNS = (
    ("R84A", "R84_new90k_attention/A_baseline"),
    ("R87A", "R87_private_method_residual/A_shared_residual"),
    ("R87B", "R87_private_method_residual/B_private_residual"),
    ("R88A", "R88_execution_role_inputs/A_canonical_anchor"),
    ("R88B", "R88_execution_role_inputs/B_execution_anchor"),
)
OUTPUT = ROOT / "code/V4/runs/R90_training_mechanism_notes/fitting_curves"
FONT = Path("/usr/share/fonts/google-noto-cjk/NotoSansCJK-Regular.ttc")
EXPECTED_STEPS = list(range(1440, 57601, 1440))
REGULAR_TRAIN_STEPS = set(range(7200, 57601, 7200))
WINNER_RULE = (
    "native int64 ind from locked source cache, verified against original "
    "FP64 argmin; never FP32 argmin"
)


def close(actual: float, expected: float, context: str) -> None:
    if not math.isclose(actual, expected, rel_tol=0, abs_tol=1e-12):
        raise ValueError(f"{context}: {actual} != {expected}")


def load_runs() -> list[dict]:
    runs = []
    reference_pools = None
    reference_weights = None
    for label, relative in RUNS:
        directory = ROOT / "code/V4/runs" / relative
        history_path = directory / "history.json"
        lock_path = directory / "final_test_lock.json"
        history = json.loads(history_path.read_text())
        lock = json.loads(lock_path.read_text())
        if [row["successful_updates"] for row in history] != EXPECTED_STEPS:
            raise ValueError(f"Unexpected VAL update grid: {label}")
        selected = lock["selected_update"]
        expected_train = REGULAR_TRAIN_STEPS | {selected}
        if {r["successful_updates"] for r in history if r["train"]} != expected_train:
            raise ValueError(f"Unexpected TRAIN update grid: {label}")
        if not lock["selected_without_test"] or lock["per_problem_assembly"]:
            raise ValueError(f"Unexpected lock selection scope: {label}")
        if lock["winner_rule"] != WINNER_RULE:
            raise ValueError(f"Unexpected winner definition: {label}")
        weights = [r["optimization"]["consistency_weight"] for r in history]
        if reference_weights is None:
            reference_weights = weights
        if weights != reference_weights:
            raise ValueError("Logged consistency-weight trajectories differ")
        for row in history:
            update = row["successful_updates"]
            if row["sample_exposures"] != update * 1024:
                raise ValueError(f"Logical-batch exposure mismatch: {label}/{update}")
            for split, n in (("train", 81000), ("val", 4500)):
                evaluation = row[split]
                if evaluation is None:
                    continue
                if evaluation["successful_updates"] != update:
                    raise ValueError(f"Evaluation step mismatch: {label}/{split}")
                problems = evaluation["per_problem"]
                if len(problems) != 18 or {p["n"] for p in problems.values()} != {n}:
                    raise ValueError(f"Not full18/full split: {label}/{split}/{update}")
                if evaluation["winner_rule"] != WINNER_RULE:
                    raise ValueError(f"Winner-rule mismatch: {label}/{split}/{update}")
                for metric in ("top1", "ce"):
                    close(
                        evaluation["macro"][f"macro_{metric}"],
                        statistics.fmean(p[metric] for p in problems.values()),
                        f"{label}/{split}/{update}/macro_{metric}",
                    )
                close(
                    evaluation["macro"]["macro_top1"],
                    sum(p["native_correct"] for p in problems.values()) / (18 * n),
                    f"{label}/{split}/{update}/native Top1",
                )
                pools = {name: p["pool"] for name, p in problems.items()}
                if reference_pools is None:
                    reference_pools = pools
                if pools != reference_pools:
                    raise ValueError(f"Logged candidate pools differ: {label}/{split}")
            if row["train"]:
                if (
                    row["train"]["identity"]["model_state_sha256"]
                    != row["val"]["identity"]["model_state_sha256"]
                ):
                    raise ValueError(f"Paired evaluations have different states: {label}")
        selected_row = next(r for r in history if r["successful_updates"] == selected)
        for metric in ("macro_top1", "macro_ce", "macro_vs_sbs_pct"):
            close(lock["val"][metric], selected_row["val"]["macro"][metric], label)
        if selected_row["val"]["macro"]["macro_vs_sbs_pct"] != min(
            r["val"]["macro"]["macro_vs_sbs_pct"] for r in history
        ):
            raise ValueError(f"Lock is not the logged minimum VAL vs_SBS: {label}")
        runs.append(
            dict(
                label=label,
                relative=relative,
                history=history,
                lock=lock,
                history_path=str(history_path),
                lock_path=str(lock_path),
            )
        )
    return runs


def paired_metrics(row: dict) -> dict:
    train, val = row["train"]["macro"], row["val"]["macro"]
    return {
        "train_top1_pct": train["macro_top1"] * 100,
        "val_top1_pct": val["macro_top1"] * 100,
        "top1_gap_train_minus_val_pp": (train["macro_top1"] - val["macro_top1"]) * 100,
        "train_eval_ce": train["macro_ce"],
        "val_eval_ce": val["macro_ce"],
        "ce_gap_val_minus_train": val["macro_ce"] - train["macro_ce"],
    }


def build_rows(runs: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    curves, summaries, paired = [], [], []
    for run in runs:
        selected = run["lock"]["selected_update"]
        for row in run["history"]:
            update = row["successful_updates"]
            metrics = paired_metrics(row) if row["train"] else {}
            origin = ""
            if row["train"]:
                origin = (
                    "scheduled_7200"
                    if update in REGULAR_TRAIN_STEPS
                    else "backfilled_selected"
                )
                pair = dict(run=run["label"], successful_updates=update, **metrics)
                paired.append(pair)
                if update == selected or update == 57600:
                    summaries.append(
                        dict(
                            run=run["label"],
                            arm=run["relative"],
                            point="selected" if update == selected else "last_57600",
                            successful_updates=update,
                            train_point_origin=origin,
                            **metrics,
                            val_macro_vs_sbs_pct=row["val"]["macro"]["macro_vs_sbs_pct"],
                            selected_update=selected,
                            source_history=run["history_path"],
                            source_lock=run["lock_path"],
                        )
                    )
            for split, n in (("train", 81000), ("val", 4500)):
                evaluation = row[split]
                if evaluation is None:
                    continue
                curves.append(
                    dict(
                        run=run["label"],
                        arm=run["relative"],
                        split=split,
                        successful_updates=update,
                        sample_exposures=row["sample_exposures"],
                        logical_batch=1024,
                        problems=18,
                        n_per_problem=n,
                        n_total=18 * n,
                        eval_tag=evaluation["tag"],
                        macro_top1_pct=evaluation["macro"]["macro_top1"] * 100,
                        macro_eval_ce=evaluation["macro"]["macro_ce"],
                        top1_gap_train_minus_val_pp=metrics.get(
                            "top1_gap_train_minus_val_pp", ""
                        ),
                        ce_gap_val_minus_train=metrics.get("ce_gap_val_minus_train", ""),
                        train_point_origin=origin if split == "train" else "",
                        is_selected=update == selected,
                        is_last_57600=update == 57600,
                        selected_update=selected,
                        source_history=run["history_path"],
                        source_lock=run["lock_path"],
                    )
                )
    return curves, summaries, paired


def write_csv(name: str, rows: list[dict]) -> None:
    with (OUTPUT / name).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def span(rows: list[dict], key: str, precision: int = 4) -> str:
    values = [r[key] for r in rows]
    return f"{min(values):.{precision}f}–{max(values):.{precision}f}"


def markdown_table(rows: list[dict]) -> str:
    lines = [
        "| 模型 | 更新步 | TRAIN Top1(%) | VAL Top1(%) | TRAIN-VAL(pp) | TRAIN CE | VAL CE | VAL-TRAIN CE |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['run']} | {row['successful_updates']} | "
            f"{row['train_top1_pct']:.4f} | {row['val_top1_pct']:.4f} | "
            f"{row['top1_gap_train_minus_val_pp']:.4f} | "
            f"{row['train_eval_ce']:.6f} | {row['val_eval_ce']:.6f} | "
            f"{row['ce_gap_val_minus_train']:.6f} |"
        )
    return "\n".join(lines)


def write_markdown(runs: list[dict], summaries: list[dict], paired: list[dict]) -> None:
    selected = [r for r in summaries if r["point"] == "selected"]
    last = [r for r in summaries if r["point"] == "last_57600"]
    sources = "\n".join(
        f"- {r['label']}：`{r['history_path']}`；`{r['lock_path']}`。" for r in runs
    )
    report = f"""# NEW90k full18：TRAIN / VAL 拟合轨迹

主交付：`R90_fitting_report.pdf`（2页，嵌入 NotoSansCJK-Regular.ttc 字形）；辅助：`curve_points.csv`、`selected_and_last.csv`。

## 范围与口径

仅解析下列五组 `history.json`、`final_test_lock.json`，不打开权重、NPZ、缓存、TEST/LIB 或协议源码；不训练、不推理、不使用 GPU、不重选 checkpoint。
任务给定数据版本为 NEW90k：18题每题 TRAIN 81,000 / VAL 4,500；日志中的各题样本数、18题等权 macro、逻辑 batch 1,024（exposures/update）相符，预算均为 57,600 成功更新。
横轴使用 `successful_updates`，不是 epoch 或墙钟。VAL：1,440起每1,440步，共40点/模型；TRAIN：7,200起每7,200步，外加选中步补评。R87B选中36,000已在定期点中，其余四组选中点为补评；TRAIN共44点。未补造0步或缺测点。
自然 Top1 为与 native int64 `ind` winner 的逐例一致率（日志声明其与原始 FP64 argmin 核对，非 FP32 argmin），不使用 tie-aware Top1。未重新审计底层标签/缓存。
CE严格取 `history[i]['train'/'val']['macro']['macro_ce']`，两侧均为完整划分 eval/maximin CE；不是 `optimization['ce']`（区间、dropout训练态）。
Top1 gap = TRAIN - VAL，单位百分点(pp)；CE gap = VAL - TRAIN，单位为CE原值；只在同一成功步且模型状态哈希一致的实测点上计算。

## 已锁定选中步

各组选中规则为完整VAL `macro_vs_sbs_pct` 严格最小（共同的TRAIN拟合SBS、并列取首次），不是最高Top1或最低CE。本报告沿用锁定结果。

{markdown_table(selected)}

## 末步 57,600

{markdown_table(last)}

## 可支持的结论与限制

末步TRAIN-VAL Top1差为 **{span(last, 'top1_gap_train_minus_val_pp')} pp**，VAL-TRAIN CE差为 **{span(last, 'ce_gap_val_minus_train', 6)}**。选中步Top1差为 {span(selected, 'top1_gap_train_minus_val_pp')} pp。全部{len(paired)}个配对实测点Top1差为 {span(paired, 'top1_gap_train_minus_val_pp')} pp。
这些已完成NEW90k轨迹的full18等权宏平均未见巨大的TRAIN/VAL分离；不据此推断欠拟合根因、Bayes上限、没有可学习信号，或减弱正则一定有效，也不延伸为每题/每候选结论。聚合JSON不能定位具体漏掉的真实winner；无TEST结论或新实验计划。末步只作描述，不代替已锁定checkpoint。
候选池：五组全部已记录TRAIN/VAL点的逐题pool名称与顺序相同；这是日志级比较，未核验实例成员或底层成本。R87包含shared/private残差变体（B有私有参数），R88包含canonical/execution角色锚点变体；跨版本不是严格单因素对照，不能归因于某一参数或正则。
microcap是否恒定/跨模型相同、dropout及weight decay均未写入这两类JSON，故未核验，不假设相同。日志可核对的一致性权重轨迹五组相同：1,440步为0.1166667，2,880步为0.2333333，4,320步起为0.35；这不等于完整正则配置一致。

## 数据来源

{sources}

`curve_points.csv`仅含实测全量评估点（200个VAL、44个TRAIN）；未配对的VAL行差值留空。`train_point_origin`区分定期评估与选中步补评。`selected_and_last.csv`含10行选中/末步精确指标和锁定VAL vs_SBS。CSV保留浮点精度，PDF/本说明中Top1保留4位、CE保留6位。
"""
    (OUTPUT / "README.md").write_text(report, encoding="utf-8")


def write_pdf(runs: list[dict], curves: list[dict], summaries: list[dict], paired: list[dict]) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["OMP_NUM_THREADS"] = "1"
    with tempfile.TemporaryDirectory(prefix=".mpl-", dir=OUTPUT) as config:
        os.environ["MPLCONFIGDIR"] = config
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib import font_manager
        from matplotlib import pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages
        from matplotlib.lines import Line2D
        from matplotlib.ticker import MaxNLocator

        if not FONT.is_file():
            raise FileNotFoundError(FONT)
        font_manager.fontManager.addfont(str(FONT))
        family = font_manager.FontProperties(fname=str(FONT)).get_name()
        plt.rcParams.update(
            {
                "font.family": family,
                "pdf.fonttype": 42,
                "ps.fonttype": 42,
                "font.size": 8,
                "axes.unicode_minus": False,
                "axes.linewidth": 0.5,
                "axes.labelsize": 8,
                "xtick.labelsize": 7,
                "ytick.labelsize": 7,
                "savefig.facecolor": "white",
            }
        )
        blue, red, ink, gray = "#0072B2", "#D55E00", "#242424", "#686868"
        page_size = (210 / 25.4, 297 / 25.4)
        selected = [r for r in summaries if r["point"] == "selected"]
        last = [r for r in summaries if r["point"] == "last_57600"]

        def text(fig, x, y, value, size=8, color=ink):
            return fig.text(x, y, value, fontsize=size, va="top", color=color, linespacing=1.5)

        def check_bounds(fig):
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            for item in fig.texts:
                box = item.get_window_extent(renderer)
                if not fig.bbox.contains(box.x0, box.y0) or not fig.bbox.contains(box.x1, box.y1):
                    raise ValueError(f"Page text exceeds bounds: {item.get_text()}")
            for ax in fig.axes:
                box = ax.get_tightbbox(renderer)
                if box.x0 < 0 or box.y0 < 0 or box.x1 > fig.bbox.x1 or box.y1 > fig.bbox.y1:
                    raise ValueError("Plot/table exceeds page bounds")

        with PdfPages(OUTPUT / "R90_fitting_report.pdf") as pdf:
            pdf.infodict().update(
                Title="NEW90k full18 TRAIN/VAL 拟合轨迹",
                Author="R90 bounded CPU JSON analysis",
                Subject="Full-split eval Top1 and maximin CE; no TEST/LIB analysis",
            )
            fig, axes = plt.subplots(5, 2, figsize=page_size)
            fig.subplots_adjust(left=0.10, right=0.96, top=0.83, bottom=0.14, hspace=0.63, wspace=0.28)
            text(fig, 0.065, 0.965, "NEW90k full18：TRAIN / VAL 拟合轨迹", 15)
            text(fig, 0.065, 0.923, "18题等权宏平均；每题 TRAIN 81,000 / VAL 4,500；逻辑 batch 1,024。", 9)
            text(fig, 0.065, 0.900, "横轴=成功更新步（非epoch/墙钟）；VAL首点1,440，TRAIN首点7,200；预算57,600。", 8.4)
            handles = [
                Line2D([], [], color=blue, marker="o", markersize=3, label="TRAIN 全量eval"),
                Line2D([], [], color=red, linestyle="--", label="VAL 全量eval"),
                Line2D([], [], color=gray, linestyle=":", marker="D", markersize=3, label="锁定选中步"),
            ]
            fig.legend(handles=handles, loc="center", bbox_to_anchor=(0.52, 0.865), ncol=3, frameon=False, fontsize=8.8)
            for index, run in enumerate(runs):
                for column, (key, title) in enumerate((("macro_top1_pct", "Top1 (%)"), ("macro_eval_ce", "eval CE"))):
                    ax = axes[index, column]
                    for split, color, style in (("train", blue, "-"), ("val", red, "--")):
                        points = [r for r in curves if r["run"] == run["label"] and r["split"] == split]
                        ax.plot(
                            [p["successful_updates"] for p in points], [p[key] for p in points],
                            color=color, linestyle=style, linewidth=1,
                            marker="o" if split == "train" else None, markersize=2.5,
                        )
                        point = next(p for p in points if p["is_selected"])
                        ax.scatter(point["successful_updates"], point[key], color=color, marker="D", s=15, zorder=5)
                    update = run["lock"]["selected_update"]
                    ax.axvline(update, color=gray, linestyle=":", linewidth=0.8, zorder=0)
                    ax.set_title(f"{run['label']} | {title}", loc="left", fontsize=8.5, pad=5)
                    ax.set_title(f"选中 {update:,}", loc="right", fontsize=7.5, color=gray, pad=5)
                    ax.set_xlim(1440, 57600)
                    values = [p[key] for p in curves]
                    pad = 0.35 if column == 0 else 0.008
                    ax.set_ylim(min(values) - pad, max(values) + pad)
                    ax.set_xticks([1440, 28800, 57600], ["1,440", "28,800", "57,600"])
                    ax.yaxis.set_major_locator(MaxNLocator(3))
                    ax.grid(axis="y", color="#DEDEDE", linewidth=0.5)
                    ax.spines[["top", "right"]].set_visible(False)
                    if index == 4:
                        ax.set_xlabel("成功更新步", labelpad=3)
            text(fig, 0.065, 0.095, "TRAIN每7,200步一次，选中步另补评（已有定期点则复用）；VAL每1,440步一次。", 8)
            text(fig, 0.065, 0.073, "CE只取history[split]['macro']['macro_ce']，均为eval/maximin CE，不是optimization['ce']。", 7.8)
            text(fig, 0.065, 0.051, "折线只连接实测点，不补造0步或缺测TRAIN点。自然Top1采用native ind / 原始FP64 winner。", 7.8)
            text(fig, 0.91, 0.024, "1 / 2", 8, gray)
            check_bounds(fig)
            pdf.savefig(fig)
            plt.close(fig)

            fig = plt.figure(figsize=page_size)
            text(fig, 0.065, 0.965, "数值核对与解释边界", 15)
            text(fig, 0.065, 0.925, "Top1 gap = TRAIN - VAL（百分点pp）；CE gap = VAL - TRAIN（CE原值）。", 8.6)
            text(fig, 0.065, 0.904, "差值均由未舍入的同一步完整评估计算；正值方向均为TRAIN更优。", 8.6)

            def table(rows, bottom):
                ax = fig.add_axes([0.065, bottom, 0.88, 0.18])
                ax.axis("off")
                cells = [
                    [r["run"], f"{r['successful_updates']:,}", f"{r['train_top1_pct']:.4f}",
                     f"{r['val_top1_pct']:.4f}", f"{r['top1_gap_train_minus_val_pp']:.4f}",
                     f"{r['train_eval_ce']:.6f}", f"{r['val_eval_ce']:.6f}",
                     f"{r['ce_gap_val_minus_train']:.6f}"]
                    for r in rows
                ]
                headers = ["模型", "更新步", "TRAIN\nTop1(%)", "VAL\nTop1(%)", "Top1 gap\n(pp)", "TRAIN CE", "VAL CE", "CE gap"]
                artist = ax.table(cellText=cells, colLabels=headers, cellLoc="center", bbox=[0, 0, 1, 1], colWidths=[0.09, 0.115, 0.13, 0.13, 0.12, 0.14, 0.14, 0.135])
                artist.auto_set_font_size(False)
                artist.set_fontsize(7.9)
                for (r, _), cell in artist.get_celld().items():
                    cell.set_linewidth(0.35)
                    cell.set_edgecolor("#C7C7C7")
                    cell.set_facecolor("#EAF1F4" if r == 0 else "white")

            text(fig, 0.065, 0.873, "锁定选中步（沿用原锁定，不重新选型）", 10.5)
            table(selected, 0.674)
            text(fig, 0.065, 0.642, "末步 57,600（只作轨迹描述，不代替锁定checkpoint）", 10.5)
            table(last, 0.443)
            text(fig, 0.065, 0.423, "原选型依据：完整VAL macro_vs_sbs_pct严格最小，共同TRAIN拟合SBS，并列取首次。", 8)
            text(fig, 0.065, 0.403, "因此，选中步不等于最高Top1或最低CE；末步仍有轻微波动，不据此重选。", 8)
            text(fig, 0.065, 0.369, "核心观察", 10.5)
            text(fig, 0.065, 0.344, f"末步Top1 gap：{span(last, 'top1_gap_train_minus_val_pp')} pp；CE gap：{span(last, 'ce_gap_val_minus_train', 6)}。", 9)
            text(fig, 0.065, 0.321, f"选中步Top1 gap：{span(selected, 'top1_gap_train_minus_val_pp')} pp；全部{len(paired)}个配对点：{span(paired, 'top1_gap_train_minus_val_pp')} pp。", 8.5)
            text(fig, 0.065, 0.294, "仅支持：这些已完成NEW90k轨迹的full18宏平均未见巨大TRAIN/VAL分离。", 8.5)
            text(fig, 0.065, 0.272, "不证明欠拟合根因、Bayes上限、无可学习信号或减弱正则有效；不推广到每题/候选。", 8.2)
            text(fig, 0.065, 0.250, "日志仅有聚合量，无法定位具体漏掉的真实winner；本报告无TEST结论或新实验计划。", 8.2)
            text(fig, 0.065, 0.215, "比较范围", 10.5)
            text(fig, 0.065, 0.190, "日志逐题候选pool名称/顺序相同；R87含shared/private残差，R88含角色锚点变体，", 8)
            text(fig, 0.065, 0.170, "涉及残差/私有参数或角色位变化，跨版本不是严格单因素实验，不作因果归因。", 8)
            text(fig, 0.065, 0.148, "microcap是否恒定/相同、dropout、weight decay未记录，未核验；不能假设一致。", 8)
            text(fig, 0.065, 0.128, "已记录一致性权重轨迹相同：0.1167、0.2333、0.35（4,320步起）；非完整正则审计。", 8)
            text(fig, 0.065, 0.097, "简称：R84A=A_baseline；R87A=A_shared_residual；R87B=B_private_residual；", 7.7, gray)
            text(fig, 0.065, 0.078, "R88A=A_canonical_anchor；R88B=B_execution_anchor。完整路径与精确指标见MD/CSV。", 7.7, gray)
            text(fig, 0.065, 0.053, "来源仅五组history.json / final_test_lock.json；未读取权重、NPZ、TEST/LIB或缓存。", 7.7, gray)
            text(fig, 0.91, 0.024, "2 / 2", 8, gray)
            check_bounds(fig)
            pdf.savefig(fig)
            plt.close(fig)


def main() -> None:
    runs = load_runs()
    curves, summaries, paired = build_rows(runs)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    write_csv("curve_points.csv", curves)
    write_csv("selected_and_last.csv", summaries)
    write_markdown(runs, summaries, paired)
    write_pdf(runs, curves, summaries, paired)
    print(f"Output: {OUTPUT}")
    print(f"Curve points: {len(curves)}; paired TRAIN/VAL points: {len(paired)}")
    print(markdown_table([r for r in summaries if r["point"] == "last_57600"]))


if __name__ == "__main__":
    main()

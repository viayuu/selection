from __future__ import annotations


def format_eval_block(
    header: str,
    macro: dict,
    per_problem: dict[str, dict],
    problem_order: list[str] | None = None,
    marks: dict[str, str] | None = None,
    extra_macro_fields: list[tuple[str, str]] | None = None,
) -> str:
    parts = [header, f"macro_top1={macro['macro_top1']:.4f}"]
    if "macro_vs_sbs_pct" in macro:
        parts.append(f"vs_sbs={macro['macro_vs_sbs_pct']:+.3f}%")
    if "macro_vbs_gap_closed_pct" in macro:
        parts.append(f"vbs_closed={macro['macro_vbs_gap_closed_pct']:+.2f}%")
    if "macro_fixable_recovery_rate" in macro:
        parts.append(f"fixable_recovery={macro['macro_fixable_recovery_rate']:.4f}")
    if "macro_safe_harm_rate" in macro:
        parts.append(f"safe_harm={macro['macro_safe_harm_rate']:.4f}")
    if "macro_hidden_winner_mass" in macro:
        parts.append(f"hidden_mass={macro['macro_hidden_winner_mass']:.4f}")
    if extra_macro_fields:
        for key, value in extra_macro_fields:
            parts.append(f"{key}={value}")

    order = problem_order or list(per_problem.keys())
    marks = marks or {}
    lines = [" ".join(parts)]
    for problem in order:
        if problem not in per_problem:
            continue
        row = per_problem[problem]
        sbs = row.get("sbs_cost", row.get("sbs", row.get("sbs_mean")))
        suffix = marks.get(problem, "")
        if sbs is None:
            lines.append(
                f"{problem:>13}: top1={row['top1']:.3f} "
                f"mean_cost={row['mean_cost']:.4f} "
                f"vs_sbs={row['vs_sbs_pct']:+.2f}%{suffix}"
            )
        else:
            lines.append(
                f"{problem:>13}: top1={row['top1']:.3f} "
                f"mean_cost={row['mean_cost']:.4f} "
                f"(sbs={float(sbs):.4f}) "
                f"vs_sbs={row['vs_sbs_pct']:+.2f}%{suffix}"
            )
    return "\n".join(lines)

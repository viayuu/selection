from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


DEFAULT_INIT_NAMES = ('lehd', 'elg', 'difusco')
DEFAULT_OPERATOR_NAMES = ('two_opt', 'lehd_rrc_step', 'dact_2opt_step')


def _load_name_defaults(output_dir: Path) -> tuple[list[str], list[str]]:
    config_path = output_dir / 'config.json'
    if not config_path.exists():
        return list(DEFAULT_INIT_NAMES), list(DEFAULT_OPERATOR_NAMES)
    try:
        config = json.loads(config_path.read_text(encoding='utf-8'))
    except Exception:
        return list(DEFAULT_INIT_NAMES), list(DEFAULT_OPERATOR_NAMES)

    init_names = config.get('init_zoo')
    op_names = config.get('operator_zoo')
    if not isinstance(init_names, list) or not init_names:
        init_names = list(DEFAULT_INIT_NAMES)
    if not isinstance(op_names, list) or not op_names:
        op_names = list(DEFAULT_OPERATOR_NAMES)
    return [str(x) for x in init_names], [str(x) for x in op_names]


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def _moving_average(values: Sequence[float], window: int = 20) -> list[float]:
    out: list[float] = []
    for i in range(len(values)):
        lo = max(0, i - window + 1)
        chunk = values[lo : i + 1]
        out.append(sum(chunk) / max(len(chunk), 1))
    return out


def _avg(values: Sequence[float]) -> float:
    return sum(values) / max(len(values), 1)


def _dict_from_distribution(names: Sequence[str], values: Sequence[float]) -> dict[str, float]:
    return {str(name): float(value) for name, value in zip(names, values)}


def _split_monotonic_segments(rows: list[dict]) -> list[list[dict]]:
    if not rows:
        return []
    segments: list[list[dict]] = [[rows[0]]]
    for row in rows[1:]:
        if row['step'] <= segments[-1][-1]['step']:
            segments.append([row])
        else:
            segments[-1].append(row)
    return segments


def _plot_train_lengths(train: list[dict], out_dir: Path):
    steps = [r['step'] for r in train]
    init_len = [r['init_length'] for r in train]
    final_len = [r['final_length'] for r in train]
    improve = [r['improvement'] for r in train]

    plt.figure(figsize=(10, 5))
    plt.plot(steps, init_len, label='init_length', linewidth=1.2, alpha=0.8)
    plt.plot(steps, final_len, label='final_length', linewidth=1.2, alpha=0.8)
    plt.plot(steps, _moving_average(improve, 20), label='improvement (MA20)', linewidth=2.0)
    plt.xlabel('train step')
    plt.ylabel('length / improvement')
    plt.title('Training trajectory: init vs final length and improvement')
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / 'train_lengths_improvement.png', dpi=180)
    plt.close()


def _plot_action_distributions(train: list[dict], out_dir: Path, init_names: Sequence[str], op_names: Sequence[str]):
    steps = [r['step'] for r in train]
    init_dist = list(zip(*[r['init_distribution'] for r in train]))
    op_dist = list(zip(*[r['operator_distribution'] for r in train]))

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axes[0].stackplot(steps, init_dist, labels=init_names, alpha=0.85)
    axes[0].set_title('Initializer distribution over training steps')
    axes[0].set_ylabel('probability')
    axes[0].set_ylim(0, 1)
    axes[0].grid(True, alpha=0.2)
    axes[0].legend(loc='upper right')

    axes[1].stackplot(steps, op_dist, labels=op_names, alpha=0.85)
    axes[1].set_title('Operator distribution over training steps')
    axes[1].set_xlabel('train step')
    axes[1].set_ylabel('probability')
    axes[1].set_ylim(0, 1)
    axes[1].grid(True, alpha=0.2)
    axes[1].legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(out_dir / 'action_distributions.png', dpi=180)
    plt.close()


def _plot_timing(train: list[dict], out_dir: Path):
    steps = [r['step'] for r in train]
    reset_t = [r['reset_seconds'] for r in train]
    rollout_t = [r['rollout_seconds'] for r in train]
    opt_t = [r['optimize_seconds'] for r in train]
    total_t = [r['total_seconds'] for r in train]

    plt.figure(figsize=(10, 5))
    plt.stackplot(steps, reset_t, rollout_t, opt_t, labels=['reset', 'rollout', 'optimize'], alpha=0.85)
    plt.plot(steps, total_t, color='black', linewidth=1.6, label='total')
    plt.xlabel('train step')
    plt.ylabel('seconds')
    plt.title('Per-step training time breakdown')
    plt.grid(True, alpha=0.25)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(out_dir / 'timing_breakdown.png', dpi=180)
    plt.close()


def _plot_signals(train: list[dict], out_dir: Path):
    steps = [r['step'] for r in train]
    loss = [r['loss'] for r in train]
    entropy = [r['entropy'] for r in train]
    improve = [r['improvement'] for r in train]

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    axes[0].plot(steps, loss, label='loss', color='tab:red', alpha=0.85)
    axes[0].plot(steps, _moving_average(loss, 20), label='loss (MA20)', color='black', linewidth=2)
    axes[0].set_ylabel('loss')
    axes[0].grid(True, alpha=0.25)
    axes[0].legend()

    axes[1].plot(steps, entropy, label='entropy', color='tab:purple', alpha=0.85)
    axes[1].plot(steps, _moving_average(entropy, 20), label='entropy (MA20)', color='black', linewidth=2)
    axes[1].set_ylabel('entropy')
    axes[1].grid(True, alpha=0.25)
    axes[1].legend()

    axes[2].plot(steps, improve, label='improvement', color='tab:green', alpha=0.85)
    axes[2].plot(steps, _moving_average(improve, 20), label='improvement (MA20)', color='black', linewidth=2)
    axes[2].set_xlabel('train step')
    axes[2].set_ylabel('improvement')
    axes[2].grid(True, alpha=0.25)
    axes[2].legend()

    fig.suptitle('Optimization signals during training', y=0.995)
    plt.tight_layout()
    plt.savefig(out_dir / 'optimization_signals.png', dpi=180)
    plt.close()


def _plot_eval(evals: list[dict], out_dir: Path, init_names: Sequence[str], op_names: Sequence[str]):
    if not evals:
        return

    steps = [r['step'] for r in evals]
    init_len = [r['init_length'] for r in evals]
    final_len = [r['final_length'] for r in evals]
    improve = [r['improvement'] for r in evals]
    init_dist = list(zip(*[r['init_distribution'] for r in evals]))
    op_dist = list(zip(*[r['operator_distribution'] for r in evals]))

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0, 0].plot(steps, init_len, marker='o', label='init_length')
    axes[0, 0].plot(steps, final_len, marker='o', label='final_length')
    axes[0, 0].set_title('Eval lengths across checkpoints')
    axes[0, 0].set_xlabel('step')
    axes[0, 0].set_ylabel('length')
    axes[0, 0].grid(True, alpha=0.25)
    axes[0, 0].legend()

    axes[0, 1].plot(steps, improve, marker='o', color='tab:green')
    axes[0, 1].set_title('Eval improvement across checkpoints')
    axes[0, 1].set_xlabel('step')
    axes[0, 1].set_ylabel('improvement')
    axes[0, 1].grid(True, alpha=0.25)

    axes[1, 0].stackplot(steps, init_dist, labels=init_names, alpha=0.85)
    axes[1, 0].set_title('Eval initializer distribution')
    axes[1, 0].set_xlabel('step')
    axes[1, 0].set_ylabel('probability')
    axes[1, 0].set_ylim(0, 1)
    axes[1, 0].grid(True, alpha=0.2)
    axes[1, 0].legend(loc='upper right')

    axes[1, 1].stackplot(steps, op_dist, labels=op_names, alpha=0.85)
    axes[1, 1].set_title('Eval operator distribution')
    axes[1, 1].set_xlabel('step')
    axes[1, 1].set_ylabel('probability')
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].grid(True, alpha=0.2)
    axes[1, 1].legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(out_dir / 'eval_overview.png', dpi=180)
    plt.close()


def analyze(output_dir: Path, init_names: Sequence[str], op_names: Sequence[str]) -> None:
    train_all = _load_jsonl(output_dir / 'train_metrics.jsonl')
    evals_all = _load_jsonl(output_dir / 'eval_metrics.jsonl')
    if not train_all:
        raise FileNotFoundError(f'No train metrics found at {(output_dir / "train_metrics.jsonl").as_posix()}')

    train_segments = _split_monotonic_segments(train_all)
    train = train_segments[-1]
    prev_train_max = max((seg[-1]['step'] for seg in train_segments[:-1]), default=0)
    evals = [row for row in evals_all if prev_train_max < row['step'] <= train[-1]['step']]

    out_dir = output_dir / 'analysis'
    out_dir.mkdir(parents=True, exist_ok=True)

    _plot_train_lengths(train, out_dir)
    _plot_action_distributions(train, out_dir, init_names, op_names)
    _plot_timing(train, out_dir)
    _plot_signals(train, out_dir)
    _plot_eval(evals, out_dir, init_names, op_names)

    steps = [r['step'] for r in train]
    init_len = [r['init_length'] for r in train]
    final_len = [r['final_length'] for r in train]
    improve = [r['improvement'] for r in train]
    entropy = [r['entropy'] for r in train]
    loss = [r['loss'] for r in train]
    reset_t = [r['reset_seconds'] for r in train]
    rollout_t = [r['rollout_seconds'] for r in train]
    opt_t = [r['optimize_seconds'] for r in train]
    total_t = [r['total_seconds'] for r in train]

    best_train = max(train, key=lambda r: r['improvement'])
    best_eval = min(evals, key=lambda r: r['final_length']) if evals else None
    last = train[-1]

    summary = {
        'num_train_steps': len(train),
        'num_eval_records': len(evals),
        'num_train_segments_detected': len(train_segments),
        'previous_train_max_step': prev_train_max,
        'train_step_range': [steps[0], steps[-1]],
        'train_init_length_mean': _avg(init_len),
        'train_final_length_mean': _avg(final_len),
        'train_improvement_mean': _avg(improve),
        'train_improvement_max': max(improve),
        'train_improvement_min': min(improve),
        'train_entropy_first20_mean': _avg(entropy[:20]),
        'train_entropy_last20_mean': _avg(entropy[-20:]),
        'train_loss_first20_mean': _avg(loss[:20]),
        'train_loss_last20_mean': _avg(loss[-20:]),
        'last_step': last['step'],
        'last_init_distribution': _dict_from_distribution(init_names, last['init_distribution']),
        'last_operator_distribution': _dict_from_distribution(op_names, last['operator_distribution']),
        'best_train_step': best_train['step'],
        'best_train_improvement': best_train['improvement'],
        'mean_total_seconds': _avg(total_t),
        'mean_reset_seconds': _avg(reset_t),
        'mean_rollout_seconds': _avg(rollout_t),
        'mean_optimize_seconds': _avg(opt_t),
    }
    if evals:
        summary['last_eval'] = evals[-1]
        summary['best_eval'] = best_eval

    (out_dir / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')

    lines: list[str] = []
    lines.append('# Operator Policy Output Analysis')
    lines.append('')
    lines.append(f'- Output dir: `{output_dir.as_posix()}`')
    lines.append(f'- Train steps (latest run): `{len(train)}`')
    lines.append(f'- Eval checkpoints (latest run): `{len(evals)}`')
    lines.append(f'- Train segments detected in log: `{len(train_segments)}`')
    if len(train_segments) > 1:
        lines.append(f'- Detected appended history; analysis uses only the latest monotonic train segment after previous max step `{prev_train_max}`.')
    lines.append('')
    lines.append('## Charts')
    lines.append(f'- `{(out_dir / "train_lengths_improvement.png").as_posix()}`')
    lines.append(f'- `{(out_dir / "action_distributions.png").as_posix()}`')
    lines.append(f'- `{(out_dir / "timing_breakdown.png").as_posix()}`')
    lines.append(f'- `{(out_dir / "optimization_signals.png").as_posix()}`')
    if evals:
        lines.append(f'- `{(out_dir / "eval_overview.png").as_posix()}`')
    lines.append('')
    lines.append('## Key Findings')
    lines.append(f'- Mean train init length = `{summary["train_init_length_mean"]:.4f}`, mean final length = `{summary["train_final_length_mean"]:.4f}`, mean improvement = `{summary["train_improvement_mean"]:.4f}`.')
    lines.append(f'- Best train improvement occurs at step `{best_train["step"]}` with value `{best_train["improvement"]:.4f}`.')
    lines.append(f'- Mean timing per step: total `{summary["mean_total_seconds"]:.2f}s`, reset `{summary["mean_reset_seconds"]:.2f}s`, rollout `{summary["mean_rollout_seconds"]:.2f}s`, optimize `{summary["mean_optimize_seconds"]:.2f}s`.')
    lines.append(f'- Last train initializer distribution: `{summary["last_init_distribution"]}`.')
    lines.append(f'- Last train operator distribution: `{summary["last_operator_distribution"]}`.')
    if evals:
        last_eval = evals[-1]
        lines.append(f'- Last eval @ step `{last_eval["step"]}`: init `{last_eval["init_length"]:.4f}` -> final `{last_eval["final_length"]:.4f}`, improvement `{last_eval["improvement"]:.4f}`.')
        lines.append(f'- Best eval final length appears at step `{best_eval["step"]}` with final `{best_eval["final_length"]:.4f}` and improvement `{best_eval["improvement"]:.4f}`.')
        lines.append(f'- Last eval initializer distribution: `{_dict_from_distribution(init_names, last_eval["init_distribution"])}`.')
        lines.append(f'- Last eval operator distribution: `{_dict_from_distribution(op_names, last_eval["operator_distribution"])}`.')
    lines.append('')
    lines.append('## Interpretation')
    lines.append('- `train_lengths_improvement.png` only provides an online training signal because each train step uses a different batch; use eval curves for strict performance comparison.')
    lines.append('- `action_distributions.png` is the most informative for policy behavior: it shows whether the selector remains exploratory or collapses to a narrow subset of actions.')
    lines.append('- `timing_breakdown.png` reveals whether training is bottlenecked by solver execution (`reset`/`rollout`) or by optimizer compute.')
    lines.append('- `optimization_signals.png` should be read qualitatively: entropy decay indicates confidence increase, while REINFORCE loss is not expected to monotonically decrease.')
    if evals:
        lines.append('- `eval_overview.png` is the most reliable view of actual progress because it compares checkpoints on the eval set rather than across different train batches.')
    lines.append('')
    lines.append('## Caveat')
    lines.append('- If `advantage_mean` stays near zero, that is normal here because the baseline is the batch mean; this field is not a strong diagnostic signal in the current implementation.')
    (out_dir / 'ANALYSIS.md').write_text('\n'.join(lines), encoding='utf-8')


def main() -> None:
    parser = argparse.ArgumentParser(description='Analyze operator-policy output directory and generate charts/report.')
    parser.add_argument('output_dir', type=str, help='Path like 0new/operator_policy/outputs/new')
    parser.add_argument('--init-names', nargs='+', default=None)
    parser.add_argument('--op-names', nargs='+', default=None)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    default_init_names, default_op_names = _load_name_defaults(output_dir)
    init_names = args.init_names if args.init_names is not None else default_init_names
    op_names = args.op_names if args.op_names is not None else default_op_names
    analyze(output_dir, init_names, op_names)


if __name__ == '__main__':
    main()

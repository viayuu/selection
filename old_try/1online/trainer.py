from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
import json
from pathlib import Path
import time

import torch
from torch.distributions import Categorical


@dataclass
class StepMetrics:
    loss: float
    init_length: float
    final_length: float
    improvement: float
    reward_mean: float
    entropy: float
    baseline: float
    advantage_mean: float
    init_distribution: list[float] = field(default_factory=list)
    operator_distribution: list[float] = field(default_factory=list)
    trace_path: str | None = None
    trace_meta_path: str | None = None
    trace_step_summary_path: str | None = None
    select_seconds: float = 0.0
    reset_seconds: float = 0.0
    rollout_seconds: float = 0.0
    optimize_seconds: float = 0.0
    total_seconds: float = 0.0



def _distribution_from_actions(actions: torch.Tensor, num_actions: int) -> list[float]:
    counts = torch.bincount(actions.detach().cpu(), minlength=num_actions).float()
    denom = counts.sum().clamp_min(1)
    return (counts / denom).tolist()



def _sample_from_logits(logits: torch.Tensor, greedy: bool):
    dist = Categorical(logits=logits)
    if greedy:
        actions = logits.argmax(dim=-1)
    else:
        actions = dist.sample()
    log_prob = dist.log_prob(actions)
    entropy = dist.entropy()
    return actions, log_prob, entropy



def _trace_side_paths(trace_path: str | Path) -> tuple[Path, Path, Path]:
    path = Path(trace_path)
    meta = path.with_suffix('.meta.json')
    step_summary = path.with_name(f"{path.stem}_step_summary.json")
    return path, meta, step_summary



def _as_named_distribution(names: Sequence[str] | None, values: Sequence[float]) -> dict[str, float] | list[float]:
    if names is None:
        return [float(v) for v in values]
    return {str(name): float(values[idx]) for idx, name in enumerate(names)}



def _format_prob_vector(names: Sequence[str] | None, values: Sequence[float], precision: int = 4) -> str:
    if names is not None and len(names) == len(values):
        return '[' + ', '.join(f'{names[idx]}:{float(values[idx]):.{precision}f}' for idx in range(len(values))) + ']'
    return '[' + ', '.join(f'{float(v):.{precision}f}' for v in values) + ']'



def _safe_name(names: Sequence[str] | None, action_id: int) -> str:
    if names is None:
        return str(action_id)
    if 0 <= action_id < len(names):
        return str(names[action_id])
    return str(action_id)



def _format_sequence(seq: Sequence[str]) -> str:
    return ' -> '.join(str(x) for x in seq) if seq else '(empty)'



def _format_aug_counts(augmentation_types: Sequence[str] | None) -> str:
    if augmentation_types is None:
        return 'unknown'
    counts = Counter(str(x) for x in augmentation_types)
    return ', '.join(f'{name}:{counts[name]}' for name in sorted(counts))


class ReinforceTrainer:
    def __init__(
        self,
        policy,
        env,
        optimizer,
        rollout_steps: int,
        *,
        init_names: Sequence[str] | None = None,
        op_names: Sequence[str] | None = None,
        entropy_coef: float = 1e-3,
        grad_clip: float = 1.0,
        single_sample_baseline_momentum: float = 0.9,
    ):
        self.policy = policy
        self.env = env
        self.optimizer = optimizer
        self.rollout_steps = int(rollout_steps)
        self.init_names = tuple(str(x) for x in init_names) if init_names is not None else None
        self.op_names = tuple(str(x) for x in op_names) if op_names is not None else None
        self.entropy_coef = float(entropy_coef)
        self.grad_clip = float(grad_clip)
        self.single_sample_baseline_momentum = float(single_sample_baseline_momentum)
        self.running_baseline: torch.Tensor | None = None

    def _baseline(self, episode_return: torch.Tensor) -> torch.Tensor:
        batch_mean = episode_return.mean().detach()
        if self.running_baseline is None:
            self.running_baseline = batch_mean
        else:
            self.running_baseline = (
                self.single_sample_baseline_momentum * self.running_baseline
                + (1.0 - self.single_sample_baseline_momentum) * batch_mean
            )
        if episode_return.numel() > 1:
            return batch_mean
        return self.running_baseline.to(device=episode_return.device, dtype=episode_return.dtype)

    def _init_records(
        self,
        *,
        mode: str,
        coords: torch.Tensor,
        augmentation_types: Sequence[str] | None,
        init_actions: torch.Tensor,
        init_probs: torch.Tensor,
        batch_index: int = 1,
        instance_offset: int = 0,
        train_step: int | None = None,
    ) -> list[dict]:
        batch_size = coords.size(0)
        records: list[dict] = []
        init_actions_cpu = init_actions.detach().cpu()
        init_probs_cpu = init_probs.detach().cpu()
        for idx in range(batch_size):
            action_id = int(init_actions_cpu[idx].item())
            records.append(
                {
                    'mode': mode,
                    'train_step': (int(train_step) if train_step is not None else None),
                    'instance_id': int(instance_offset + idx),
                    'batch_index': int(batch_index),
                    'augmentation_type': (str(augmentation_types[idx]) if augmentation_types is not None else None),
                    'problem_size': int(coords.size(1)),
                    'init_action_id': action_id,
                    'init_action_name': _safe_name(self.init_names, action_id),
                    'init_probs': [float(v) for v in init_probs_cpu[idx].tolist()],
                    'operator_action_ids': [],
                    'operator_action_names': [],
                    'operator_probs': [],
                    'current_length_by_step': [],
                    'best_length_by_step': [],
                    'reward_by_step': [],
                }
            )
        return records

    def _finalize_records(
        self,
        records: list[dict],
        state,
        episode_reward: torch.Tensor,
        relative_improvement: torch.Tensor,
    ) -> None:
        init_len_cpu = state.init_length.detach().cpu()
        final_len_cpu = state.best_length.detach().cpu()
        reward_cpu = episode_reward.detach().cpu()
        relative_cpu = relative_improvement.detach().cpu()
        for idx, record in enumerate(records):
            init_len = float(init_len_cpu[idx].item())
            final_len = float(final_len_cpu[idx].item())
            record['init_length'] = init_len
            record['final_length'] = final_len
            record['improvement'] = float(init_len - final_len)
            record['normalized_improvement'] = float(relative_cpu[idx].item())
            record['episode_reward'] = float(reward_cpu[idx].item())

    def _write_trace_bundle(
        self,
        *,
        trace_path: str | Path | None,
        mode: str,
        records: Sequence[dict],
        step_op_counts: torch.Tensor,
    ) -> tuple[str | None, str | None, str | None]:
        if trace_path is None:
            return None, None, None
        trace_file, meta_file, step_summary_file = _trace_side_paths(trace_path)
        trace_file.parent.mkdir(parents=True, exist_ok=True)
        trace_file.write_text('\n'.join(json.dumps(row, ensure_ascii=False) for row in records) + ('\n' if records else ''), encoding='utf-8')
        meta_payload = {
            'mode': mode,
            'format': 'jsonl',
            'selection_mode': 'sampling' if mode == 'train' else 'argmax',
            'rollout_steps': self.rollout_steps,
            'init_zoo': list(self.init_names) if self.init_names is not None else None,
            'operator_zoo': list(self.op_names) if self.op_names is not None else None,
            'fields': {
                'instance_id': '0-based sample index in this run',
                'augmentation_type': 'identity / rotate / reflect / mix',
                'init_probs': 'softmax probabilities over initializer zoo',
                'operator_action_ids': 'selected operator id for each rollout step',
                'operator_action_names': 'selected operator name for each rollout step',
                'operator_probs': 'softmax probabilities over operator zoo at each rollout step',
                'current_length_by_step': 'current tour length after each rollout step',
                'best_length_by_step': 'best-so-far length after each rollout step',
                'reward_by_step': 'raw single-step gain in best length after each rollout step',
                'normalized_improvement': 'diagnostic only: (init_length - final_length) / init_length',
                'episode_reward': 'training reward for REINFORCE, defined as -final_length',
            },
        }
        meta_file.write_text(json.dumps(meta_payload, indent=2, ensure_ascii=False), encoding='utf-8')

        step_rows = []
        for step_idx in range(self.rollout_steps):
            counts = step_op_counts[step_idx].detach().cpu().to(torch.float64)
            denom = counts.sum().clamp_min(1.0)
            dist = (counts / denom).tolist()
            step_rows.append(
                {
                    'step': step_idx + 1,
                    'counts': [int(round(x)) for x in counts.tolist()],
                    'distribution': [float(v) for v in dist],
                    'named_distribution': _as_named_distribution(self.op_names, dist),
                }
            )
        step_summary_payload = {
            'mode': mode,
            'rollout_steps': self.rollout_steps,
            'operator_zoo': list(self.op_names) if self.op_names is not None else None,
            'steps': step_rows,
        }
        step_summary_file.write_text(json.dumps(step_summary_payload, indent=2, ensure_ascii=False), encoding='utf-8')
        return str(trace_file), str(meta_file), str(step_summary_file)

    def _log_init_records(
        self,
        *,
        log_fn: Callable[[str], None] | None,
        tag: str,
        prefix: str,
        records: Sequence[dict],
        init_lengths: torch.Tensor,
        prob_precision: int,
    ) -> None:
        if log_fn is None:
            return
        init_len_cpu = init_lengths.detach().cpu()
        for idx, record in enumerate(records):
            log_fn(
                f"[{tag}:item:init]{prefix} item={record['instance_id']:04d} batch={record['batch_index']:04d} "
                f"aug={record['augmentation_type']} init={record['init_action_name']} "
                f"init_probs={_format_prob_vector(self.init_names, record['init_probs'], prob_precision)} "
                f"init_len={float(init_len_cpu[idx].item()):.4f}"
            )

    def _log_rollout_records(
        self,
        *,
        log_fn: Callable[[str], None] | None,
        tag: str,
        prefix: str,
        records: Sequence[dict],
        rollout_idx: int,
        current_cpu: torch.Tensor,
        best_cpu: torch.Tensor,
        reward_cpu: torch.Tensor,
        stagnation_cpu: torch.Tensor,
        prob_precision: int,
    ) -> None:
        if log_fn is None:
            return
        for idx, record in enumerate(records):
            op_name = record['operator_action_names'][-1] if record['operator_action_names'] else '(none)'
            op_probs = record['operator_probs'][-1] if record['operator_probs'] else []
            log_fn(
                f"[{tag}:item:rollout]{prefix} iter={rollout_idx + 1:03d}/{self.rollout_steps:03d} "
                f"item={record['instance_id']:04d} aug={record['augmentation_type']} op={op_name} "
                f"op_probs={_format_prob_vector(self.op_names, op_probs, prob_precision)} "
                f"current={float(current_cpu[idx].item()):.4f} best={float(best_cpu[idx].item()):.4f} "
                f"step_gain={float(reward_cpu[idx].item()):.4f} stagnation={int(stagnation_cpu[idx].item())}"
            )

    def _log_final_records(
        self,
        *,
        log_fn: Callable[[str], None] | None,
        tag: str,
        prefix: str,
        records: Sequence[dict],
    ) -> None:
        if log_fn is None:
            return
        for record in records:
            log_fn(
                f"[{tag}:item:final]{prefix} item={record['instance_id']:04d} aug={record['augmentation_type']} "
                f"init={record['init_action_name']} init_len={record['init_length']:.4f} final_len={record['final_length']:.4f} "
                f"improve={record['improvement']:.4f} rel_improve={record['normalized_improvement']:.4f} "
                f"episode_reward={record['episode_reward']:.4f} op_seq={_format_sequence(record['operator_action_names'])}"
            )

    def train_step(
        self,
        coords: torch.Tensor,
        *,
        augmentation_types: Sequence[str] | None = None,
        trace_path: str | Path | None = None,
        step_idx: int | None = None,
        log_fn: Callable[[str], None] | None = None,
        rollout_log_every: int = 1,
        log_instance_details: bool = True,
        log_prob_precision: int = 4,
    ) -> StepMetrics:
        total_t0 = time.perf_counter()
        self.policy.train()
        select_t0 = time.perf_counter()
        init_logits = self.policy.init_logits(coords)
        init_probs = torch.softmax(init_logits, dim=-1)
        init_actions, init_log_prob, init_entropy = _sample_from_logits(init_logits, greedy=False)
        init_distribution = _distribution_from_actions(init_actions, init_logits.size(-1))
        select_seconds = time.perf_counter() - select_t0
        prefix = f" step={step_idx:05d}" if step_idx is not None else ''

        if log_fn is not None:
            log_fn(
                f"[train:init]{prefix} batch={coords.size(0)} n={coords.size(1)} device={coords.device} "
                f"aug_counts={_format_aug_counts(augmentation_types)} select_t={select_seconds:.2f}s "
                f"init_mean_entropy={init_entropy.mean().item():.4f} init_dist={_format_prob_vector(self.init_names, init_distribution, 3)}"
            )

        reset_t0 = time.perf_counter()
        state = self.env.reset(coords, init_actions, log_fn=log_fn)
        reset_seconds = time.perf_counter() - reset_t0
        if log_fn is not None:
            log_fn(f"[train:reset]{prefix} reset_t={reset_seconds:.2f}s init_len={state.init_length.mean().item():.4f}")

        records = self._init_records(
            mode='train',
            coords=coords,
            augmentation_types=augmentation_types,
            init_actions=init_actions,
            init_probs=init_probs,
            batch_index=1,
            instance_offset=0,
            train_step=step_idx,
        )
        if log_instance_details:
            self._log_init_records(
                log_fn=log_fn,
                tag='train',
                prefix=prefix,
                records=records,
                init_lengths=state.init_length,
                prob_precision=log_prob_precision,
            )

        total_log_prob = init_log_prob
        total_entropy = init_entropy
        op_dim = self.policy.op_head[-1].out_features
        op_counts = torch.zeros((op_dim,), dtype=torch.float32)
        step_op_counts = torch.zeros((self.rollout_steps, op_dim), dtype=torch.float64)
        rollout_t0 = time.perf_counter()
        rollout_log_every = int(rollout_log_every)

        for rollout_idx in range(self.rollout_steps):
            op_logits = self.policy.operator_logits(
                coords=state.coords,
                tour=state.current.tour,
                current_length=state.current.length,
                best_length=state.best_length,
                init_length=state.init_length,
                init_action=state.init_action,
                prev_operator=state.prev_operator,
                step_index=state.step_index,
                rollout_steps=self.rollout_steps,
                stagnation=state.stagnation,
            )
            op_probs = torch.softmax(op_logits, dim=-1)
            op_actions, op_log_prob, op_entropy = _sample_from_logits(op_logits, greedy=False)
            batch_counts = torch.bincount(op_actions.detach().cpu(), minlength=op_dim).float()
            op_counts += batch_counts
            step_op_counts[rollout_idx] += batch_counts.to(dtype=step_op_counts.dtype)

            next_state, reward, _ = self.env.step(state, op_actions)
            step_gain = reward

            op_actions_cpu = op_actions.detach().cpu()
            op_probs_cpu = op_probs.detach().cpu()
            current_cpu = next_state.current.length.detach().cpu()
            best_cpu = next_state.best_length.detach().cpu()
            reward_cpu = step_gain.detach().cpu()
            stagnation_cpu = next_state.stagnation.detach().cpu()
            for idx, record in enumerate(records):
                op_id = int(op_actions_cpu[idx].item())
                record['operator_action_ids'].append(op_id)
                record['operator_action_names'].append(_safe_name(self.op_names, op_id))
                record['operator_probs'].append([float(v) for v in op_probs_cpu[idx].tolist()])
                record['current_length_by_step'].append(float(current_cpu[idx].item()))
                record['best_length_by_step'].append(float(best_cpu[idx].item()))
                record['reward_by_step'].append(float(reward_cpu[idx].item()))

            state = next_state
            total_log_prob = total_log_prob + op_log_prob
            total_entropy = total_entropy + op_entropy
            should_log_rollout = (
                log_fn is not None
                and (
                    rollout_log_every > 0 and (rollout_idx + 1) % rollout_log_every == 0
                    or rollout_idx == 0
                    or rollout_idx + 1 == self.rollout_steps
                )
            )
            if should_log_rollout:
                dist = _distribution_from_actions(op_actions, op_dim)
                log_fn(
                    f"[train:rollout]{prefix} iter={rollout_idx + 1:03d}/{self.rollout_steps:03d} "
                    f"current={state.current.length.mean().item():.4f} best={state.best_length.mean().item():.4f} "
                    f"step_gain_mean={step_gain.mean().item():.4f} op_dist={_format_prob_vector(self.op_names, dist, 3)}"
                )
                if log_instance_details:
                    self._log_rollout_records(
                        log_fn=log_fn,
                        tag='train',
                        prefix=prefix,
                        records=records,
                        rollout_idx=rollout_idx,
                        current_cpu=current_cpu,
                        best_cpu=best_cpu,
                        reward_cpu=reward_cpu,
                        stagnation_cpu=stagnation_cpu,
                        prob_precision=log_prob_precision,
                    )

        rollout_seconds = time.perf_counter() - rollout_t0
        # 训练 reward 直接使用最终解质量，而不是相对初始化解的改进。
        # 这样初始化器和后续 operator 都会共同为“最终解更短”负责。
        episode_reward = -state.best_length
        relative_improvement = (state.init_length - state.best_length) / state.init_length.clamp_min(1e-6)
        baseline = self._baseline(episode_reward)
        advantage = episode_reward - baseline
        raw_advantage_mean = float(advantage.mean().item())
        if advantage.numel() > 1:
            adv_std = advantage.std(unbiased=False)
            if torch.isfinite(adv_std) and adv_std.item() > 1e-6:
                advantage = advantage / (adv_std + 1e-6)

        optimize_t0 = time.perf_counter()
        loss = -((total_log_prob * advantage.detach()).mean()) - self.entropy_coef * total_entropy.mean()
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.grad_clip)
        self.optimizer.step()
        optimize_seconds = time.perf_counter() - optimize_t0
        total_seconds = time.perf_counter() - total_t0

        self._finalize_records(records, state, episode_reward, relative_improvement)
        if log_instance_details:
            self._log_final_records(log_fn=log_fn, tag='train', prefix=prefix, records=records)
        trace_out, meta_out, step_summary_out = self._write_trace_bundle(
            trace_path=trace_path,
            mode='train',
            records=records,
            step_op_counts=step_op_counts,
        )
        operator_distribution = (op_counts / op_counts.sum().clamp_min(1)).tolist()

        if log_fn is not None:
            log_fn(
                f"[train:done]{prefix} total_t={total_seconds:.2f}s rollout_t={rollout_seconds:.2f}s "
                f"opt_t={optimize_seconds:.2f}s loss={loss.item():.4f} init={state.init_length.mean().item():.4f} "
                f"final={state.best_length.mean().item():.4f} improve={(state.init_length - state.best_length).mean().item():.4f} "
                f"rel_improve={relative_improvement.mean().item():.4f} reward_mean={episode_reward.mean().item():.4f} "
                f"baseline={baseline.mean().item():.4f} entropy={total_entropy.mean().item():.4f} "
                f"advantage_mean={raw_advantage_mean:.4f}"
            )
            if trace_out is not None:
                log_fn(f"[train:trace]{prefix} jsonl={trace_out}")
                log_fn(f"[train:trace]{prefix} meta={meta_out}")
                log_fn(f"[train:trace]{prefix} step_summary={step_summary_out}")

        return StepMetrics(
            loss=float(loss.item()),
            init_length=float(state.init_length.mean().item()),
            final_length=float(state.best_length.mean().item()),
            improvement=float((state.init_length - state.best_length).mean().item()),
            reward_mean=float(episode_reward.mean().item()),
            entropy=float(total_entropy.mean().item()),
            baseline=float(baseline.mean().item()),
            advantage_mean=raw_advantage_mean,
            init_distribution=init_distribution,
            operator_distribution=operator_distribution,
            trace_path=trace_out,
            trace_meta_path=meta_out,
            trace_step_summary_path=step_summary_out,
            select_seconds=float(select_seconds),
            reset_seconds=float(reset_seconds),
            rollout_seconds=float(rollout_seconds),
            optimize_seconds=float(optimize_seconds),
            total_seconds=float(total_seconds),
        )

    @torch.no_grad()
    def _evaluate_batch(
        self,
        coords: torch.Tensor,
        augmentation_types: Sequence[str] | None,
        init_counts: torch.Tensor,
        op_counts: torch.Tensor,
        step_op_counts: torch.Tensor,
        trace_handle,
        *,
        mode: str,
        batch_index: int,
        instance_offset: int,
        log_fn: Callable[[str], None] | None = None,
        log_instance_details: bool = True,
        log_prob_precision: int = 4,
    ) -> dict:
        init_logits = self.policy.init_logits(coords)
        init_probs = torch.softmax(init_logits, dim=-1)
        init_actions, _, _ = _sample_from_logits(init_logits, greedy=True)
        state = self.env.reset(coords, init_actions)
        init_counts += torch.bincount(init_actions.detach().cpu(), minlength=init_counts.numel()).float()

        records = self._init_records(
            mode=mode,
            coords=coords,
            augmentation_types=augmentation_types,
            init_actions=init_actions,
            init_probs=init_probs,
            batch_index=batch_index,
            instance_offset=instance_offset,
        )
        if log_instance_details:
            self._log_init_records(
                log_fn=log_fn,
                tag=f'eval:{mode}',
                prefix='',
                records=records,
                init_lengths=state.init_length,
                prob_precision=log_prob_precision,
            )

        for rollout_idx in range(self.rollout_steps):
            op_logits = self.policy.operator_logits(
                coords=state.coords,
                tour=state.current.tour,
                current_length=state.current.length,
                best_length=state.best_length,
                init_length=state.init_length,
                init_action=state.init_action,
                prev_operator=state.prev_operator,
                step_index=state.step_index,
                rollout_steps=self.rollout_steps,
                stagnation=state.stagnation,
            )
            op_probs = torch.softmax(op_logits, dim=-1)
            op_actions, _, _ = _sample_from_logits(op_logits, greedy=True)
            batch_counts = torch.bincount(op_actions.detach().cpu(), minlength=op_counts.numel()).float()
            op_counts += batch_counts
            step_op_counts[rollout_idx] += batch_counts.to(dtype=step_op_counts.dtype)

            next_state, reward, _ = self.env.step(state, op_actions)
            step_gain = reward

            op_actions_cpu = op_actions.detach().cpu()
            op_probs_cpu = op_probs.detach().cpu()
            current_cpu = next_state.current.length.detach().cpu()
            best_cpu = next_state.best_length.detach().cpu()
            reward_cpu = step_gain.detach().cpu()
            stagnation_cpu = next_state.stagnation.detach().cpu()
            for idx, record in enumerate(records):
                op_id = int(op_actions_cpu[idx].item())
                record['operator_action_ids'].append(op_id)
                record['operator_action_names'].append(_safe_name(self.op_names, op_id))
                record['operator_probs'].append([float(v) for v in op_probs_cpu[idx].tolist()])
                record['current_length_by_step'].append(float(current_cpu[idx].item()))
                record['best_length_by_step'].append(float(best_cpu[idx].item()))
                record['reward_by_step'].append(float(reward_cpu[idx].item()))

            state = next_state
            if log_fn is not None:
                dist = _distribution_from_actions(op_actions, op_counts.numel())
                log_fn(
                    f"[eval:{mode}:rollout] batch={batch_index:04d} iter={rollout_idx + 1:03d}/{self.rollout_steps:03d} "
                    f"current={state.current.length.mean().item():.4f} best={state.best_length.mean().item():.4f} "
                    f"step_gain_mean={step_gain.mean().item():.4f} op_dist={_format_prob_vector(self.op_names, dist, 3)}"
                )
                if log_instance_details:
                    self._log_rollout_records(
                        log_fn=log_fn,
                        tag=f'eval:{mode}',
                        prefix='',
                        records=records,
                        rollout_idx=rollout_idx,
                        current_cpu=current_cpu,
                        best_cpu=best_cpu,
                        reward_cpu=reward_cpu,
                        stagnation_cpu=stagnation_cpu,
                        prob_precision=log_prob_precision,
                    )

        episode_reward = -state.best_length
        relative_improvement = (state.init_length - state.best_length) / state.init_length.clamp_min(1e-6)
        self._finalize_records(records, state, episode_reward, relative_improvement)
        if log_instance_details:
            self._log_final_records(log_fn=log_fn, tag=f'eval:{mode}', prefix='', records=records)
        if trace_handle is not None:
            for row in records:
                trace_handle.write(json.dumps(row, ensure_ascii=False) + '\n')

        return {
            'init_sum': float(state.init_length.sum().item()),
            'final_sum': float(state.best_length.sum().item()),
            'improvement_sum': float((state.init_length - state.best_length).sum().item()),
            'reward_sum': float(episode_reward.sum().item()),
            'count': int(coords.size(0)),
        }

    @torch.no_grad()
    def evaluate_batches(
        self,
        batches: Iterable,
        *,
        mode: str,
        progress_log_fn: Callable[[str], None] | None = None,
        target_items: int | None = None,
        log_every_batches: int = 0,
        trace_path: str | Path | None = None,
        log_instance_details: bool = True,
        log_prob_precision: int = 4,
    ) -> dict:
        self.policy.eval()
        num_init = self.policy.init_head[-1].out_features
        num_op = self.policy.op_head[-1].out_features
        init_counts = torch.zeros((num_init,), dtype=torch.float32)
        op_counts = torch.zeros((num_op,), dtype=torch.float32)
        step_op_counts = torch.zeros((self.rollout_steps, num_op), dtype=torch.float64)
        total_init = 0.0
        total_final = 0.0
        total_improvement = 0.0
        total_reward = 0.0
        total_items = 0
        batch_index = 0
        start = time.perf_counter()

        trace_out, meta_out, step_summary_out = None, None, None
        trace_handle = None
        if trace_path is not None:
            trace_file, meta_file, step_summary_file = _trace_side_paths(trace_path)
            trace_file.parent.mkdir(parents=True, exist_ok=True)
            trace_handle = trace_file.open('w', encoding='utf-8', buffering=1)
            meta_payload = {
                'mode': mode,
                'format': 'jsonl',
                'selection_mode': 'argmax',
                'rollout_steps': self.rollout_steps,
                'init_zoo': list(self.init_names) if self.init_names is not None else None,
                'operator_zoo': list(self.op_names) if self.op_names is not None else None,
            }
            meta_file.write_text(json.dumps(meta_payload, indent=2, ensure_ascii=False), encoding='utf-8')
            trace_out, meta_out, step_summary_out = str(trace_file), str(meta_file), str(step_summary_file)

        try:
            for item in batches:
                if item is None:
                    continue
                if isinstance(item, tuple) and len(item) == 2:
                    coords, augmentation_types = item
                else:
                    coords, augmentation_types = item, None
                if coords is None or coords.numel() == 0:
                    continue
                batch_index += 1
                if progress_log_fn is not None:
                    progress_log_fn(
                        f"[eval:{mode}:prepare] batch={batch_index:04d} batch_size={coords.size(0)} n={coords.size(1)} "
                        f"aug_counts={_format_aug_counts(augmentation_types)}"
                    )
                stats = self._evaluate_batch(
                    coords,
                    augmentation_types,
                    init_counts,
                    op_counts,
                    step_op_counts,
                    trace_handle,
                    mode=mode,
                    batch_index=batch_index,
                    instance_offset=total_items,
                    log_fn=progress_log_fn,
                    log_instance_details=log_instance_details,
                    log_prob_precision=log_prob_precision,
                )
                total_init += stats['init_sum']
                total_final += stats['final_sum']
                total_improvement += stats['improvement_sum']
                total_reward += stats['reward_sum']
                total_items += stats['count']
                should_log = (
                    progress_log_fn is not None
                    and (
                        batch_index == 1
                        or (log_every_batches > 0 and batch_index % max(log_every_batches, 1) == 0)
                    )
                )
                if should_log:
                    elapsed = time.perf_counter() - start
                    total_target = target_items if target_items is not None else total_items
                    progress_log_fn(
                        f"[eval:{mode}] batch={batch_index:04d} items={total_items}/{total_target} "
                        f"avg_init={total_init / max(total_items, 1):.4f} avg_final={total_final / max(total_items, 1):.4f} "
                        f"avg_improve={total_improvement / max(total_items, 1):.4f} avg_reward={total_reward / max(total_items, 1):.4f} "
                        f"init_dist={_format_prob_vector(self.init_names, (init_counts / init_counts.sum().clamp_min(1)).tolist(), 3)} "
                        f"op_dist={_format_prob_vector(self.op_names, (op_counts / op_counts.sum().clamp_min(1)).tolist(), 3)} "
                        f"elapsed={elapsed:.2f}s"
                    )
        finally:
            if trace_handle is not None:
                trace_handle.close()

        if step_summary_out is not None:
            step_rows = []
            for step_idx in range(self.rollout_steps):
                counts = step_op_counts[step_idx].detach().cpu().to(torch.float64)
                denom = counts.sum().clamp_min(1.0)
                dist = (counts / denom).tolist()
                step_rows.append(
                    {
                        'step': step_idx + 1,
                        'counts': [int(round(x)) for x in counts.tolist()],
                        'distribution': [float(v) for v in dist],
                        'named_distribution': _as_named_distribution(self.op_names, dist),
                    }
                )
            Path(step_summary_out).write_text(
                json.dumps(
                    {
                        'mode': mode,
                        'rollout_steps': self.rollout_steps,
                        'operator_zoo': list(self.op_names) if self.op_names is not None else None,
                        'steps': step_rows,
                    },
                    indent=2,
                    ensure_ascii=False,
                ),
                encoding='utf-8',
            )

        return {
            'mode': mode,
            'init_length': total_init / max(total_items, 1),
            'final_length': total_final / max(total_items, 1),
            'improvement': total_improvement / max(total_items, 1),
            'reward_mean': total_reward / max(total_items, 1),
            'init_distribution': (init_counts / init_counts.sum().clamp_min(1)).tolist(),
            'operator_distribution': (op_counts / op_counts.sum().clamp_min(1)).tolist(),
            'num_items': total_items,
            'num_batches': batch_index,
            'elapsed_seconds': time.perf_counter() - start,
            'trace_path': trace_out,
            'trace_meta_path': meta_out,
            'trace_step_summary_path': step_summary_out,
        }

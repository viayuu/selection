from __future__ import annotations

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
    entropy: float
    baseline: float
    advantage_mean: float
    init_distribution: list[float] = field(default_factory=list)
    operator_distribution: list[float] = field(default_factory=list)
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


class ReinforceTrainer:
    def __init__(
        self,
        policy,
        env,
        optimizer,
        rollout_steps: int,
        entropy_coef: float = 1e-3,
        grad_clip: float = 1.0,
        single_sample_baseline_momentum: float = 0.9,
    ):
        self.policy = policy
        self.env = env
        self.optimizer = optimizer
        self.rollout_steps = int(rollout_steps)
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

    def train_step(
        self,
        coords: torch.Tensor,
        *,
        step_idx: int | None = None,
        log_fn: Callable[[str], None] | None = None,
        rollout_log_every: int = 0,
    ) -> StepMetrics:
        total_t0 = time.perf_counter()
        self.policy.train()
        select_t0 = time.perf_counter()
        init_logits = self.policy.init_logits(coords)
        init_actions, init_log_prob, init_entropy = _sample_from_logits(init_logits, greedy=False)
        init_distribution = _distribution_from_actions(init_actions, init_logits.size(-1))
        select_seconds = time.perf_counter() - select_t0
        if log_fn is not None:
            prefix = f" step={step_idx:05d}" if step_idx is not None else ""
            log_fn(
                f"[train:init]{prefix} batch={coords.size(0)} n={coords.size(1)} device={coords.device} "
                f"select_t={select_seconds:.2f}s init_mean_entropy={init_entropy.mean().item():.4f}"
            )

        reset_t0 = time.perf_counter()
        state = self.env.reset(coords, init_actions, log_fn=log_fn)
        reset_seconds = time.perf_counter() - reset_t0
        if log_fn is not None:
            prefix = f" step={step_idx:05d}" if step_idx is not None else ""
            log_fn(
                f"[train:reset]{prefix} reset_t={reset_seconds:.2f}s init_len={state.init_length.mean().item():.4f}"
            )

        total_log_prob = init_log_prob
        total_entropy = init_entropy
        op_dim = self.policy.op_head[-1].out_features
        op_counts = torch.zeros((op_dim,), dtype=torch.float32)
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
            op_actions, op_log_prob, op_entropy = _sample_from_logits(op_logits, greedy=False)
            op_counts += torch.bincount(op_actions.detach().cpu(), minlength=op_dim).float()
            state, _, _ = self.env.step(state, op_actions)
            total_log_prob = total_log_prob + op_log_prob
            total_entropy = total_entropy + op_entropy
            should_log_rollout = (
                log_fn is not None
                and (
                    rollout_idx == 0
                    or rollout_idx + 1 == self.rollout_steps
                    or (rollout_log_every > 0 and (rollout_idx + 1) % rollout_log_every == 0)
                )
            )
            if should_log_rollout:
                prefix = f" step={step_idx:05d}" if step_idx is not None else ""
                log_fn(
                    f"[train:rollout]{prefix} iter={rollout_idx + 1:02d}/{self.rollout_steps:02d} "
                    f"current={state.current.length.mean().item():.4f} best={state.best_length.mean().item():.4f} "
                    f"stagnation={state.stagnation.float().mean().item():.2f}"
                )
        rollout_seconds = time.perf_counter() - rollout_t0

        episode_return = state.init_length - state.best_length
        baseline = self._baseline(episode_return)
        advantage = episode_return - baseline
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
        operator_distribution = (op_counts / op_counts.sum().clamp_min(1)).tolist()
        if log_fn is not None:
            prefix = f" step={step_idx:05d}" if step_idx is not None else ""
            log_fn(
                f"[train:done]{prefix} total_t={total_seconds:.2f}s rollout_t={rollout_seconds:.2f}s "
                f"opt_t={optimize_seconds:.2f}s loss={loss.item():.4f} init={state.init_length.mean().item():.4f} "
                f"final={state.best_length.mean().item():.4f} improve={(state.init_length - state.best_length).mean().item():.4f} "
                f"baseline={baseline.mean().item():.4f} entropy={total_entropy.mean().item():.4f}"
            )
        return StepMetrics(
            loss=float(loss.item()),
            init_length=float(state.init_length.mean().item()),
            final_length=float(state.best_length.mean().item()),
            improvement=float((state.init_length - state.best_length).mean().item()),
            entropy=float(total_entropy.mean().item()),
            baseline=float(baseline.mean().item()),
            advantage_mean=raw_advantage_mean,
            init_distribution=init_distribution,
            operator_distribution=operator_distribution,
            select_seconds=float(select_seconds),
            reset_seconds=float(reset_seconds),
            rollout_seconds=float(rollout_seconds),
            optimize_seconds=float(optimize_seconds),
            total_seconds=float(total_seconds),
        )

    def _write_eval_trace_rows(
        self,
        trace_handle,
        *,
        batch_index: int,
        instance_offset: int,
        init_actions: torch.Tensor,
        op_history: list[torch.Tensor],
        init_length: torch.Tensor,
        final_length: torch.Tensor,
        init_names: Sequence[str] | None,
        op_names: Sequence[str] | None,
    ) -> None:
        init_cpu = init_actions.detach().cpu()
        init_len_cpu = init_length.detach().cpu()
        final_len_cpu = final_length.detach().cpu()
        op_stack = torch.stack([actions.detach().cpu() for actions in op_history], dim=1) if op_history else torch.empty((init_cpu.size(0), 0), dtype=torch.long)
        improvement_cpu = init_len_cpu - final_len_cpu
        for local_idx in range(init_cpu.size(0)):
            init_action_id = int(init_cpu[local_idx].item())
            operator_ids = op_stack[local_idx].tolist()
            row = {
                "instance_id": int(instance_offset + local_idx),
                "batch_index": int(batch_index),
                "init_action_id": init_action_id,
                "init_action_name": (str(init_names[init_action_id]) if init_names is not None else None),
                "operator_action_ids": operator_ids,
                "operator_action_names": ([str(op_names[idx]) for idx in operator_ids] if op_names is not None else None),
                "init_length": float(init_len_cpu[local_idx].item()),
                "final_length": float(final_len_cpu[local_idx].item()),
                "improvement": float(improvement_cpu[local_idx].item()),
            }
            trace_handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    @torch.no_grad()
    def _evaluate_batch(
        self,
        batch: torch.Tensor,
        init_counts: torch.Tensor,
        op_counts: torch.Tensor,
        *,
        step_op_counts: torch.Tensor | None = None,
        trace_handle=None,
        batch_index: int = 0,
        instance_offset: int = 0,
        init_names: Sequence[str] | None = None,
        op_names: Sequence[str] | None = None,
    ) -> dict:
        init_logits = self.policy.init_logits(batch)
        init_actions, _, _ = _sample_from_logits(init_logits, greedy=True)
        state = self.env.reset(batch, init_actions)
        init_counts += torch.bincount(init_actions.detach().cpu(), minlength=init_counts.numel()).float()
        op_history: list[torch.Tensor] = []
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
            op_actions, _, _ = _sample_from_logits(op_logits, greedy=True)
            op_cpu = op_actions.detach().cpu()
            batch_counts = torch.bincount(op_cpu, minlength=op_counts.numel()).float()
            op_counts += batch_counts
            if step_op_counts is not None:
                step_op_counts[rollout_idx] += batch_counts.to(dtype=step_op_counts.dtype)
            op_history.append(op_actions)
            state, _, _ = self.env.step(state, op_actions)

        if trace_handle is not None:
            self._write_eval_trace_rows(
                trace_handle,
                batch_index=batch_index,
                instance_offset=instance_offset,
                init_actions=init_actions,
                op_history=op_history,
                init_length=state.init_length,
                final_length=state.best_length,
                init_names=init_names,
                op_names=op_names,
            )

        return {
            "init_sum": float(state.init_length.sum().item()),
            "final_sum": float(state.best_length.sum().item()),
            "improvement_sum": float((state.init_length - state.best_length).sum().item()),
            "count": int(batch.size(0)),
        }

    @torch.no_grad()
    def evaluate(self, coords: torch.Tensor, batch_size: int, **kwargs) -> dict:
        def batches() -> Iterable[torch.Tensor]:
            for start in range(0, coords.size(0), batch_size):
                yield coords[start : start + batch_size]

        return self.evaluate_batches(batches(), **kwargs)

    @torch.no_grad()
    def evaluate_batches(
        self,
        batches: Iterable[torch.Tensor],
        *,
        progress_log_fn: Callable[[str], None] | None = None,
        target_items: int | None = None,
        log_every_batches: int = 0,
        trace_path: str | Path | None = None,
        init_names: Sequence[str] | None = None,
        op_names: Sequence[str] | None = None,
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
        total_items = 0
        batch_index = 0
        start = time.perf_counter()

        trace_file = Path(trace_path) if trace_path is not None else None
        trace_handle = None
        trace_meta_path = None
        trace_step_path = None
        if trace_file is not None:
            trace_file.parent.mkdir(parents=True, exist_ok=True)
            trace_handle = trace_file.open("w", encoding="utf-8", buffering=1)
            trace_meta_path = trace_file.with_suffix(".meta.json")
            trace_step_path = trace_file.with_name(f"{trace_file.stem}_step_summary.json")
            trace_meta_payload = {
                "format": "jsonl",
                "rollout_steps": self.rollout_steps,
                "init_zoo": [str(x) for x in init_names] if init_names is not None else None,
                "operator_zoo": [str(x) for x in op_names] if op_names is not None else None,
                "fields": {
                    "instance_id": "0-based global index within this evaluation run",
                    "batch_index": "1-based eval batch index",
                    "init_action_id": "initializer action id",
                    "init_action_name": "initializer name",
                    "operator_action_ids": f"operator ids for each of the {self.rollout_steps} rollout steps",
                    "operator_action_names": f"operator names for each of the {self.rollout_steps} rollout steps",
                    "init_length": "tour length after initialization",
                    "final_length": "best tour length after rollout",
                    "improvement": "init_length - final_length",
                },
            }
            trace_meta_path.write_text(json.dumps(trace_meta_payload, indent=2, ensure_ascii=False), encoding="utf-8")

        try:
            for batch in batches:
                if batch is None or batch.numel() == 0:
                    continue
                batch_index += 1
                stats = self._evaluate_batch(
                    batch,
                    init_counts,
                    op_counts,
                    step_op_counts=step_op_counts,
                    trace_handle=trace_handle,
                    batch_index=batch_index,
                    instance_offset=total_items,
                    init_names=init_names,
                    op_names=op_names,
                )
                total_init += stats["init_sum"]
                total_final += stats["final_sum"]
                total_improvement += stats["improvement_sum"]
                total_items += stats["count"]

                should_log = (
                    progress_log_fn is not None
                    and (
                        batch_index == 1
                        or (log_every_batches > 0 and batch_index % max(log_every_batches, 1) == 0)
                    )
                )
                if should_log:
                    elapsed = time.perf_counter() - start
                    avg_init = total_init / max(total_items, 1)
                    avg_final = total_final / max(total_items, 1)
                    avg_improve = total_improvement / max(total_items, 1)
                    total_target = target_items if target_items is not None else total_items
                    progress_log_fn(
                        f"[eval] batch={batch_index:04d} items={total_items}/{total_target} "
                        f"avg_init={avg_init:.4f} avg_final={avg_final:.4f} improve={avg_improve:.4f} "
                        f"elapsed={elapsed:.2f}s"
                    )
        finally:
            if trace_handle is not None:
                trace_handle.close()

        result = {
            "init_length": total_init / max(total_items, 1),
            "final_length": total_final / max(total_items, 1),
            "improvement": total_improvement / max(total_items, 1),
            "init_distribution": (init_counts / init_counts.sum().clamp_min(1)).tolist(),
            "operator_distribution": (op_counts / op_counts.sum().clamp_min(1)).tolist(),
        }
        if target_items is not None:
            result["num_items"] = total_items
            result["num_batches"] = batch_index
            result["elapsed_seconds"] = time.perf_counter() - start

        if trace_step_path is not None:
            step_rows = []
            for step_idx in range(self.rollout_steps):
                counts = step_op_counts[step_idx]
                denom = counts.sum().clamp_min(1.0)
                dist = (counts / denom).tolist()
                step_rows.append(
                    {
                        "step": step_idx + 1,
                        "counts": [int(round(x)) for x in counts.tolist()],
                        "distribution": dist,
                        "named_distribution": {str(name): float(dist[idx]) for idx, name in enumerate(op_names or range(len(dist)))},
                    }
                )
            trace_step_payload = {
                "rollout_steps": self.rollout_steps,
                "evaluated_items": total_items,
                "operator_zoo": [str(x) for x in op_names] if op_names is not None else None,
                "steps": step_rows,
            }
            trace_step_path.write_text(json.dumps(trace_step_payload, indent=2, ensure_ascii=False), encoding="utf-8")
            result["trace_path"] = str(trace_file)
            result["trace_meta_path"] = str(trace_meta_path)
            result["trace_step_summary_path"] = str(trace_step_path)

        return result

from __future__ import annotations

import argparse
from collections import Counter
import json
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from augmentations import OnlineAugmentationGenerator
from easynco_bootstrap import enable_easynco_runtime_patches, ensure_local_easynco
from env import TSPImprovementEnv
from instance_data import load_tsplib_instance
from policy import TSPSelectorPolicy
from solver_zoo import build_initializer_zoo, build_operator_zoo
from trainer import ReinforceTrainer

ensure_local_easynco()


@dataclass(frozen=True)
class TrainConfig:
    """训练入口的统一配置。

    这里把训练、周期评测、日志、solver 适配层、以及 checkpoint 覆盖项都放在一起，
    这样 `config.json` / `selector_*.pt` / `evaluate_online_instance_policy.py`
    可以共用同一套字段，不会出现“训练时一个参数名、评测时另一个参数名”的割裂情况。
    """

    # ===== 基础运行配置 =====
    seed: int = 2024
    device: str = 'cpu'
    tsplib_root: str | None = None
    instance_name: str | None = None
    instance_path: str | None = None
    output_dir: str | None = None
    problem_size: int | None = None

    # ===== 训练 / 评测节奏 =====
    train_steps: int = 500
    aug_batch_size: int = 16
    eval_aug_size: int = 128
    eval_batch_size: int = 32
    eval_every: int = 50
    save_every: int = 20
    rollout_steps: int = 100

    # ===== 日志相关 =====
    # `log_every` 决定“每隔多少个训练 step 打一轮主日志”；
    # `rollout_log_every` 决定“在一个 step 内，每隔多少个 rollout step 打一次逐步日志”；
    # `log_instance_details` 决定是否把每个实例每一步的动作、概率、长度都写进 run.log / 终端。
    log_every: int = 1
    rollout_log_every: int = 1
    log_instance_details: bool = True
    log_prob_precision: int = 4

    # ===== selector 网络结构与优化超参 =====
    lr: float = 1e-4
    entropy_coef: float = 1e-3
    grad_clip: float = 1.0
    hidden_dim: int = 128
    num_heads: int = 8
    num_layers: int = 2
    cpe_dim: int = 32
    dropout: float = 0.0

    # ===== online RL 的动作空间与增强配置 =====
    decoder_strategy: str = 'greedy'
    init_zoo: tuple[str, ...] = ('lehd', 'elg', 'difusco')
    operator_zoo: tuple[str, ...] = ('two_opt', 'lehd_rrc_step', 'dact_2opt_step')
    augmentations: tuple[str, ...] = ('rotate', 'reflect', 'mix')
    mix_prop: float = 0.5

    # ===== EasyNCO / DACT 后端求解器运行与加载配置 =====
    solver_ckpt_dir: str = 'EasyNCO/pretrained'
    easynco_solver_device: str = 'cpu'
    dact_solver_device: str = 'cpu'
    two_opt_iterations: int = 1
    glop_pomo_size: int = 2
    glop_aug_factor: int = 4
    glop_revision_iters: int = 1

    # ===== 启发式初始化器的可选配置 =====
    # v1 在线训练不一定会直接启用这些初始化器，但保留这些字段可以无缝复用 solver_zoo。
    heuristic_init_restarts: int = 4
    heuristic_metric_strategy: str = 'cartesian'
    heuristic_metric_p: int = 2
    heuristic_regret_k: int = 1

    # ===== 外部模型 checkpoint 覆盖项 =====
    # 如果显式传入，则优先于自动搜索到的默认路径。
    pomo_size: int | None = None
    pomo_ckpt: str | None = None
    am_ckpt: str | None = None
    lehd_ckpt: str | None = None
    elg_ckpt: str | None = None
    difusco_ckpt: str | None = None
    glop_ckpt: str | None = None
    dact_ckpt: str | None = None
    lih_ckpt: str | None = None
    load_path: str | None = None



def _set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)



def _make_logger(log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handle = log_path.open('a', encoding='utf-8', buffering=1)

    def _emit(msg: str):
        print(msg, flush=True)
        handle.write(msg + '\n')
        handle.flush()

    return _emit, handle



def _write_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding='utf-8')



def _append_jsonl(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as f:
        f.write(json.dumps(payload, ensure_ascii=False) + '\n')



def _save_checkpoint(policy, cfg: TrainConfig, output_dir: Path, name: str):
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save({'state_dict': policy.state_dict(), 'config': asdict(cfg)}, output_dir / name)



def _format_seconds(value: float) -> str:
    return f'{value:.2f}s'



def _format_distribution(names: tuple[str, ...], values: list[float], precision: int = 2) -> str:
    return ', '.join(f'{name}:{value:.{precision}f}' for name, value in zip(names, values))



def _format_aug_counts(augmentation_types: tuple[str, ...] | list[str] | None) -> str:
    if augmentation_types is None:
        return 'unknown'
    counts = Counter(str(x) for x in augmentation_types)
    return ', '.join(f'{name}:{counts[name]}' for name in sorted(counts))



def _chunk_batch(coords: torch.Tensor, augmentation_types: tuple[str, ...], chunk_size: int):
    chunk_size = max(int(chunk_size), 1)
    for start in range(0, coords.size(0), chunk_size):
        end = min(start + chunk_size, coords.size(0))
        yield coords[start:end], augmentation_types[start:end]



def _run_eval(
    trainer: ReinforceTrainer,
    cfg: TrainConfig,
    *,
    mode: str,
    coords: torch.Tensor,
    augmentation_types: tuple[str, ...],
    trace_path: Path,
    log_fn,
):
    return trainer.evaluate_batches(
        _chunk_batch(coords, augmentation_types, cfg.eval_batch_size),
        mode=mode,
        progress_log_fn=log_fn,
        target_items=coords.size(0),
        log_every_batches=1,
        trace_path=trace_path,
        log_instance_details=cfg.log_instance_details,
        log_prob_precision=cfg.log_prob_precision,
    )



def _log_config_details(log_fn, cfg: TrainConfig, base_instance, output_dir: Path):
    """把训练配置逐项写入 `run.log`，并给出中文说明，便于直接照着日志复现实验。"""
    rows = [
        ('seed', cfg.seed, '随机种子；控制增强采样、策略采样、参数初始化等随机过程。'),
        ('device', cfg.device, 'selector 策略网络所在设备。'),
        ('easynco_solver_device', cfg.easynco_solver_device, 'EasyNCO 初始化器/迭代器推理设备。'),
        ('dact_solver_device', cfg.dact_solver_device, 'DACT 相关 operator 推理设备。'),
        ('tsplib_root', cfg.tsplib_root, 'TSPLIB 根目录；若不直接给实例路径，则从这里搜索实例。'),
        ('instance_name', base_instance.name, '当前训练的基实例名称。'),
        ('instance_path', base_instance.path, '当前训练实例的实际文件路径。'),
        ('problem_size', base_instance.problem_size, '实例规模 n。这里始终以真实读到的实例规模为准。'),
        ('output_dir', str(output_dir), '本次 run 的日志、trace、checkpoint 输出目录。'),
        ('train_steps', cfg.train_steps, '总训练步数；每一步都会对一个 augmentation batch 做一次完整 rollout。'),
        ('aug_batch_size', cfg.aug_batch_size, '每个训练 step 的 augmentation 样本数。'),
        ('eval_aug_size', cfg.eval_aug_size, '周期评测时，augmented eval set 的样本数。'),
        ('eval_batch_size', cfg.eval_batch_size, '评测时送入 selector/solver 的分块 batch 大小。'),
        ('eval_every', cfg.eval_every, '每隔多少个训练 step 做一次周期评测。'),
        ('save_every', cfg.save_every, '每隔多少个训练 step 保存一次 `selector_last.pt`。'),
        ('rollout_steps', cfg.rollout_steps, '每个样本在初始化之后，固定执行多少步 operator 选择。'),
        ('log_every', cfg.log_every, '每隔多少个训练 step 打一轮主日志。默认 1 表示每步都打。'),
        ('rollout_log_every', cfg.rollout_log_every, '在一个 rollout 内每隔多少步打印一次步级日志。默认 1 表示每步都打。'),
        ('log_instance_details', cfg.log_instance_details, '是否把每个实例每一步的动作、概率、长度写入终端和 run.log。'),
        ('log_prob_precision', cfg.log_prob_precision, '日志里打印概率向量时保留的小数位数。'),
        ('lr', cfg.lr, 'Adam 学习率。'),
        ('entropy_coef', cfg.entropy_coef, '熵正则系数；越大越鼓励探索。'),
        ('grad_clip', cfg.grad_clip, '梯度裁剪阈值，避免训练不稳定。'),
        ('hidden_dim', cfg.hidden_dim, '状态编码器与 MLP head 的主隐藏维度。'),
        ('num_heads', cfg.num_heads, 'Transformer 编码器的注意力头数。'),
        ('num_layers', cfg.num_layers, 'Transformer 编码器层数。'),
        ('cpe_dim', cfg.cpe_dim, 'tour 顺序特征中的 cyclic positional encoding 维度。'),
        ('dropout', cfg.dropout, 'selector 网络里的 dropout。'),
        ('decoder_strategy', cfg.decoder_strategy, '保留字段；主要用于与 EasyNCO solver_zoo 的接口保持一致。'),
        ('init_zoo', ', '.join(cfg.init_zoo), '初始化方法候选集合；每个样本先从这里选一次。'),
        ('operator_zoo', ', '.join(cfg.operator_zoo), '迭代阶段的 operator 候选集合；每一步从这里选一个。'),
        ('augmentations', ', '.join(('identity',) + tuple(cfg.augmentations)), '在线增强集合；identity 会自动保留为原始实例。'),
        ('mix_prop', cfg.mix_prop, 'mix 增强内部使用的混合比例。'),
        ('solver_ckpt_dir', cfg.solver_ckpt_dir, '默认预训练 solver checkpoint 搜索目录。'),
        ('two_opt_iterations', cfg.two_opt_iterations, 'two-opt operator 单次调用时内部迭代次数。'),
        ('glop_pomo_size', cfg.glop_pomo_size, 'GLOP 相关初始化/修正时使用的 POMO 宽度。'),
        ('glop_aug_factor', cfg.glop_aug_factor, 'GLOP 使用的增强倍数。'),
        ('glop_revision_iters', cfg.glop_revision_iters, 'GLOP revision operator 的迭代次数。'),
        ('heuristic_init_restarts', cfg.heuristic_init_restarts, '启发式初始化器的重启次数。'),
        ('heuristic_metric_strategy', cfg.heuristic_metric_strategy, '启发式距离度量策略。'),
        ('heuristic_metric_p', cfg.heuristic_metric_p, '启发式距离的 p 范数。'),
        ('heuristic_regret_k', cfg.heuristic_regret_k, 'regret insertion 使用的 regret-k。'),
        ('pomo_size', cfg.pomo_size, '如需覆盖默认 POMO 宽度，可在这里指定。'),
        ('pomo_ckpt', cfg.pomo_ckpt, '显式指定 POMO checkpoint；空则自动搜索。'),
        ('am_ckpt', cfg.am_ckpt, '显式指定 Attention Model checkpoint。'),
        ('lehd_ckpt', cfg.lehd_ckpt, '显式指定 LEHD checkpoint。'),
        ('elg_ckpt', cfg.elg_ckpt, '显式指定 ELG checkpoint。'),
        ('difusco_ckpt', cfg.difusco_ckpt, '显式指定 DIFUSCO checkpoint。'),
        ('glop_ckpt', cfg.glop_ckpt, '显式指定 GLOP checkpoint。'),
        ('dact_ckpt', cfg.dact_ckpt, '显式指定 DACT checkpoint。'),
        ('lih_ckpt', cfg.lih_ckpt, '显式指定 LIH checkpoint。'),
        ('load_path', cfg.load_path, 'selector 自身的恢复训练 checkpoint 路径。'),
    ]
    coord_min = float(base_instance.coords.min().item())
    coord_max = float(base_instance.coords.max().item())
    coord_mean = float(base_instance.coords.mean().item())
    log_fn('[config] ===== 训练配置开始 =====')
    log_fn(f'[config] instance.coord_min={coord_min:.6f} | 归一化后坐标最小值。')
    log_fn(f'[config] instance.coord_max={coord_max:.6f} | 归一化后坐标最大值。')
    log_fn(f'[config] instance.coord_mean={coord_mean:.6f} | 归一化后坐标均值。')
    for name, value, desc in rows:
        log_fn(f'[config] {name}={value} | {desc}')
    log_fn('[config] ===== 训练配置结束 =====')



def run(cfg: TrainConfig):
    _set_seed(cfg.seed)
    start_time = time.perf_counter()
    base_instance = load_tsplib_instance(
        tsplib_root=cfg.tsplib_root,
        instance_name=cfg.instance_name,
        instance_path=cfg.instance_path,
        device=cfg.device,
    )

    if cfg.output_dir is None:
        output_dir = Path('0online') / 'outputs' / base_instance.name
    else:
        output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # `problem_size` 不从 CLI 直接指定，而是以真实读到的 TSPLIB 实例大小为准。
    # 这样后面的 solver_zoo 与 policy 构造都会基于同一个真实规模，不会出现错配。
    cfg = TrainConfig(**{**asdict(cfg), 'problem_size': base_instance.problem_size})

    # 落盘时把实际实例名、路径、规模一并写入，便于后续评测直接复现。
    cfg_dict = asdict(cfg)
    cfg_dict['instance_name'] = base_instance.name
    cfg_dict['instance_path'] = base_instance.path
    cfg_dict['problem_size'] = base_instance.problem_size
    (output_dir / 'config.json').write_text(json.dumps(cfg_dict, indent=2, ensure_ascii=False), encoding='utf-8')

    log, handle = _make_logger(output_dir / 'run.log')
    train_metrics_path = output_dir / 'train_metrics.jsonl'
    eval_metrics_path = output_dir / 'eval_metrics.jsonl'
    latest_metrics_path = output_dir / 'latest_metrics.json'

    try:
        log(
            f"[run] seed={cfg.seed} device={cfg.device} instance={base_instance.name} problem_size={base_instance.problem_size} "
            f"aug_batch_size={cfg.aug_batch_size} rollout_steps={cfg.rollout_steps}"
        )
        log(
            f"[run] init_zoo={','.join(cfg.init_zoo)} operator_zoo={','.join(cfg.operator_zoo)} "
            f"augmentations={','.join(('identity',) + tuple(cfg.augmentations))} mix_prop={cfg.mix_prop}"
        )
        log(
            f"[run] eval_aug_size={cfg.eval_aug_size} eval_batch_size={cfg.eval_batch_size} "
            f"easynco_solver_device={cfg.easynco_solver_device} dact_solver_device={cfg.dact_solver_device}"
        )
        log(f"[run] output_dir={output_dir}")
        _log_config_details(log, cfg, base_instance, output_dir)

        patched = enable_easynco_runtime_patches(cfg.easynco_solver_device)
        if patched:
            log(f"[run] enabled_easynco_runtime_patches={','.join(patched)}")

        if base_instance.problem_size != 100:
            log(f"[warn] current pretrained zoo is mainly tsp100-oriented, but instance size is {base_instance.problem_size}")

        init_zoo = build_initializer_zoo(cfg, log=log)
        operator_zoo = build_operator_zoo(cfg, log=log)
        log(f"[run] solver_zoo_ready init={len(init_zoo)} operators={len(operator_zoo)} elapsed={_format_seconds(time.perf_counter() - start_time)}")

        policy = TSPSelectorPolicy(
            num_initializers=len(init_zoo),
            num_operators=len(operator_zoo),
            hidden_dim=cfg.hidden_dim,
            num_heads=cfg.num_heads,
            num_layers=cfg.num_layers,
            cpe_dim=cfg.cpe_dim,
            dropout=cfg.dropout,
        ).to(cfg.device)
        if cfg.load_path is not None:
            ckpt = torch.load(cfg.load_path, map_location=cfg.device)
            state_dict = ckpt['state_dict'] if isinstance(ckpt, dict) and 'state_dict' in ckpt else ckpt
            policy.load_state_dict(state_dict)
            log(f"[run] loaded checkpoint={cfg.load_path}")

        env = TSPImprovementEnv(init_zoo=init_zoo, operator_zoo=operator_zoo, rollout_steps=cfg.rollout_steps)
        optimizer = torch.optim.Adam(policy.parameters(), lr=cfg.lr)
        trainer = ReinforceTrainer(
            policy=policy,
            env=env,
            optimizer=optimizer,
            rollout_steps=cfg.rollout_steps,
            init_names=cfg.init_zoo,
            op_names=cfg.operator_zoo,
            entropy_coef=cfg.entropy_coef,
            grad_clip=cfg.grad_clip,
        )

        train_augmentor = OnlineAugmentationGenerator(
            base_instance.coords,
            augmentations=cfg.augmentations,
            mix_prop=cfg.mix_prop,
            seed=cfg.seed,
        )
        eval_augmentor = OnlineAugmentationGenerator(
            base_instance.coords,
            augmentations=cfg.augmentations,
            mix_prop=cfg.mix_prop,
            seed=cfg.seed + 1000,
        )

        best_original_final = float('inf')
        last_eval_original = None
        last_eval_augmented = None
        log('[run] training started')
        for step in range(1, cfg.train_steps + 1):
            step_t0 = time.perf_counter()
            batch = train_augmentor.sample(cfg.aug_batch_size, device=cfg.device)
            train_trace_path = output_dir / 'train_traces' / f'step_{step:06d}.jsonl'
            should_log_this_step = (step == 1 or step % cfg.log_every == 0)
            if should_log_this_step:
                log(
                    f"[train:prepare] step={step:06d}/{cfg.train_steps:06d} batch_size={batch.coords.size(0)} n={batch.coords.size(1)} "
                    f"aug_counts={_format_aug_counts(batch.augmentation_types)} trace={train_trace_path}"
                )
            metrics = trainer.train_step(
                batch.coords,
                augmentation_types=batch.augmentation_types,
                trace_path=train_trace_path,
                step_idx=step,
                log_fn=log if should_log_this_step else None,
                rollout_log_every=cfg.rollout_log_every,
                log_instance_details=cfg.log_instance_details,
                log_prob_precision=cfg.log_prob_precision,
            )
            train_payload = {
                'step': step,
                'mode': 'train',
                'instance_name': base_instance.name,
                'loss': metrics.loss,
                'init_length': metrics.init_length,
                'final_length': metrics.final_length,
                'improvement': metrics.improvement,
                'reward_mean': metrics.reward_mean,
                'entropy': metrics.entropy,
                'baseline': metrics.baseline,
                'advantage_mean': metrics.advantage_mean,
                'init_distribution': metrics.init_distribution,
                'operator_distribution': metrics.operator_distribution,
                'trace_path': metrics.trace_path,
                'trace_meta_path': metrics.trace_meta_path,
                'step_summary_path': metrics.trace_step_summary_path,
                'select_seconds': metrics.select_seconds,
                'reset_seconds': metrics.reset_seconds,
                'rollout_seconds': metrics.rollout_seconds,
                'optimize_seconds': metrics.optimize_seconds,
                'total_seconds': metrics.total_seconds,
                'elapsed_since_step_start': time.perf_counter() - step_t0,
            }
            _append_jsonl(train_metrics_path, train_payload)
            if should_log_this_step:
                log(
                    f"[train] step={step:06d} loss={metrics.loss:.4f} init={metrics.init_length:.4f} final={metrics.final_length:.4f} "
                    f"improve={metrics.improvement:.4f} reward_mean={metrics.reward_mean:.4f} entropy={metrics.entropy:.4f} "
                    f"baseline={metrics.baseline:.4f} advantage_mean={metrics.advantage_mean:.4f}"
                )
                log(f"[train] step={step:06d} init_dist={_format_distribution(cfg.init_zoo, metrics.init_distribution, cfg.log_prob_precision)}")
                log(f"[train] step={step:06d} op_dist={_format_distribution(cfg.operator_zoo, metrics.operator_distribution, cfg.log_prob_precision)}")
                log(
                    f"[train] step={step:06d} time(select/reset/rollout/opt/total)="
                    f"{metrics.select_seconds:.2f}/{metrics.reset_seconds:.2f}/{metrics.rollout_seconds:.2f}/"
                    f"{metrics.optimize_seconds:.2f}/{metrics.total_seconds:.2f}s"
                )
                if metrics.trace_path is not None:
                    log(f"[train] step={step:06d} trace={metrics.trace_path}")
                if metrics.trace_meta_path is not None:
                    log(f"[train] step={step:06d} trace_meta={metrics.trace_meta_path}")
                if metrics.trace_step_summary_path is not None:
                    log(f"[train] step={step:06d} step_summary={metrics.trace_step_summary_path}")

            if step % max(cfg.save_every, 1) == 0 or step == cfg.train_steps:
                _save_checkpoint(policy, cfg, output_dir, 'selector_last.pt')
                latest_payload = {
                    'step': step,
                    'best_original_final': (best_original_final if best_original_final < float('inf') else None),
                    'last_train': train_payload,
                    'last_eval_original': last_eval_original,
                    'last_eval_augmented': last_eval_augmented,
                }
                _write_json(latest_metrics_path, latest_payload)
                log(f"[save] wrote checkpoint and metrics snapshot -> {latest_metrics_path}")

            if step % cfg.eval_every == 0 or step == cfg.train_steps:
                log(f"[eval] step={step:06d} starting original evaluation")
                original_coords = base_instance.coords.unsqueeze(0)
                original_types = ('identity',)
                original_trace_path = output_dir / 'eval_traces' / f'original_step_{step:06d}.jsonl'
                original_metrics = _run_eval(
                    trainer,
                    cfg,
                    mode='eval_original',
                    coords=original_coords,
                    augmentation_types=original_types,
                    trace_path=original_trace_path,
                    log_fn=log,
                )
                original_payload = {'step': step, **original_metrics}
                _append_jsonl(eval_metrics_path, original_payload)
                last_eval_original = original_payload
                log(
                    f"[eval] step={step:06d} mode=original init={original_metrics['init_length']:.4f} final={original_metrics['final_length']:.4f} "
                    f"improve={original_metrics['improvement']:.4f} reward_mean={original_metrics['reward_mean']:.4f}"
                )
                log(f"[eval] step={step:06d} mode=original init_dist={_format_distribution(cfg.init_zoo, original_metrics['init_distribution'], cfg.log_prob_precision)}")
                log(f"[eval] step={step:06d} mode=original op_dist={_format_distribution(cfg.operator_zoo, original_metrics['operator_distribution'], cfg.log_prob_precision)}")
                log(f"[eval] step={step:06d} mode=original trace={original_metrics['trace_path']}")
                log(f"[eval] step={step:06d} mode=original step_summary={original_metrics['trace_step_summary_path']}")

                log(f"[eval] step={step:06d} starting augmented evaluation")
                aug_batch = eval_augmentor.sample(cfg.eval_aug_size, device=cfg.device, deterministic_seed=cfg.seed + 77)
                log(f"[eval] step={step:06d} mode=augmented aug_counts={_format_aug_counts(aug_batch.augmentation_types)}")
                aug_trace_path = output_dir / 'eval_traces' / f'augmented_step_{step:06d}.jsonl'
                augmented_metrics = _run_eval(
                    trainer,
                    cfg,
                    mode='eval_augmented',
                    coords=aug_batch.coords,
                    augmentation_types=aug_batch.augmentation_types,
                    trace_path=aug_trace_path,
                    log_fn=log,
                )
                augmented_payload = {'step': step, **augmented_metrics}
                _append_jsonl(eval_metrics_path, augmented_payload)
                last_eval_augmented = augmented_payload
                log(
                    f"[eval] step={step:06d} mode=augmented init={augmented_metrics['init_length']:.4f} final={augmented_metrics['final_length']:.4f} "
                    f"improve={augmented_metrics['improvement']:.4f} reward_mean={augmented_metrics['reward_mean']:.4f}"
                )
                log(f"[eval] step={step:06d} mode=augmented init_dist={_format_distribution(cfg.init_zoo, augmented_metrics['init_distribution'], cfg.log_prob_precision)}")
                log(f"[eval] step={step:06d} mode=augmented op_dist={_format_distribution(cfg.operator_zoo, augmented_metrics['operator_distribution'], cfg.log_prob_precision)}")
                log(f"[eval] step={step:06d} mode=augmented trace={augmented_metrics['trace_path']}")
                log(f"[eval] step={step:06d} mode=augmented step_summary={augmented_metrics['trace_step_summary_path']}")

                if original_metrics['final_length'] < best_original_final:
                    best_original_final = original_metrics['final_length']
                    _save_checkpoint(policy, cfg, output_dir, 'selector_best.pt')
                    log(f"[ckpt] saved new best {output_dir / 'selector_best.pt'} final={best_original_final:.4f}")

                latest_payload = {
                    'step': step,
                    'best_original_final': best_original_final,
                    'last_train': train_payload,
                    'last_eval_original': last_eval_original,
                    'last_eval_augmented': last_eval_augmented,
                }
                _write_json(latest_metrics_path, latest_payload)
                log(f"[save] wrote eval snapshot -> {latest_metrics_path}")

        log(f"[run] training finished elapsed={_format_seconds(time.perf_counter() - start_time)}")
    finally:
        handle.close()



def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(
        description='Train single-instance online RL selector for TSP.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    def add(group, *flags, help_text: str, **kwargs):
        group.add_argument(*flags, help=help_text, **kwargs)

    # ===== 实例与路径 =====
    instance_group = parser.add_argument_group('实例定位')
    add(instance_group, '--tsplib_root', type=str, default=None, help_text='TSPLIB 根目录；若不直接给 `--instance_path`，则会在这个目录下搜索实例文件。')
    add(instance_group, '--instance_name', type=str, default=None, help_text='实例名，例如 `kroA100`；通常与 `--tsplib_root` 配合使用。')
    add(instance_group, '--instance_path', type=str, default=None, help_text='直接给出实例文件路径；若提供，则优先于 `tsplib_root + instance_name`。')
    add(instance_group, '--output_dir', type=str, default=None, help_text='输出目录；为空时默认写到 `0online/outputs/<实例名>`。')
    add(instance_group, '--seed', type=int, default=TrainConfig.seed, help_text='随机种子；会同时影响增强采样、策略采样与 PyTorch 随机性。')

    # ===== 设备 =====
    device_group = parser.add_argument_group('设备与求解器后端')
    add(device_group, '--device', type=str, default=TrainConfig.device, help_text='selector 策略网络所使用的设备，例如 `cpu`、`cuda:0`。')
    add(device_group, '--easynco_solver_device', type=str, default=TrainConfig.easynco_solver_device, help_text='EasyNCO 初始化器和大部分 operator 推理设备。')
    add(device_group, '--dact_solver_device', type=str, default=TrainConfig.dact_solver_device, help_text='DACT operator 推理设备；可与主 solver 设备分开设置。')

    # ===== 训练节奏 =====
    schedule_group = parser.add_argument_group('训练 / 评测节奏')
    add(schedule_group, '--train_steps', type=int, default=TrainConfig.train_steps, help_text='总训练步数。每个 step 都会对一个 augmentation batch 做一次完整 rollout 并更新参数。')
    add(schedule_group, '--rollout_steps', type=int, default=TrainConfig.rollout_steps, help_text='每个样本初始化后固定执行多少步 operator 选择。')
    add(schedule_group, '--aug_batch_size', type=int, default=TrainConfig.aug_batch_size, help_text='训练时每个 step 的 augmentation batch 大小。')
    add(schedule_group, '--batch_size', type=int, default=None, help_text='兼容旧脚本的别名；若提供，则会覆盖 `aug_batch_size`。')
    add(schedule_group, '--eval_aug_size', type=int, default=TrainConfig.eval_aug_size, help_text='周期评测时，augmented eval set 的样本总数。')
    add(schedule_group, '--eval_batch_size', type=int, default=TrainConfig.eval_batch_size, help_text='评测时分块送入模型的 batch 大小。')
    add(schedule_group, '--eval_every', type=int, default=TrainConfig.eval_every, help_text='每隔多少个训练 step 做一次周期评测。')
    add(schedule_group, '--save_every', type=int, default=TrainConfig.save_every, help_text='每隔多少个训练 step 保存一次 `selector_last.pt`。')

    # ===== 日志 =====
    log_group = parser.add_argument_group('日志与可观测性')
    add(log_group, '--log_every', type=int, default=TrainConfig.log_every, help_text='每隔多少个训练 step 打一轮主日志。默认 1 表示每步都输出。')
    add(log_group, '--rollout_log_every', type=int, default=TrainConfig.rollout_log_every, help_text='在一个 rollout 内，每隔多少步打印一次 step 级日志。默认 1 表示每一步都打印。')
    add(log_group, '--log_prob_precision', type=int, default=TrainConfig.log_prob_precision, help_text='日志中打印概率向量时的小数位数。')
    log_group.add_argument(
        '--disable_instance_logs',
        action='store_true',
        help='关闭逐实例逐步日志。默认会把每个实例每一步的 init/operator 选择与概率写入 run.log 和终端。',
    )

    # ===== selector 网络 =====
    model_group = parser.add_argument_group('selector 网络与优化')
    add(model_group, '--lr', type=float, default=TrainConfig.lr, help_text='Adam 学习率。')
    add(model_group, '--entropy_coef', type=float, default=TrainConfig.entropy_coef, help_text='熵正则系数；越大越鼓励探索。')
    add(model_group, '--grad_clip', type=float, default=TrainConfig.grad_clip, help_text='梯度裁剪阈值。')
    add(model_group, '--hidden_dim', type=int, default=TrainConfig.hidden_dim, help_text='状态编码器和 MLP head 的主隐藏维度。')
    add(model_group, '--num_heads', type=int, default=TrainConfig.num_heads, help_text='Transformer 编码器注意力头数。')
    add(model_group, '--num_layers', type=int, default=TrainConfig.num_layers, help_text='Transformer 编码器层数。')
    add(model_group, '--cpe_dim', type=int, default=TrainConfig.cpe_dim, help_text='cyclic positional encoding 的维度。')
    add(model_group, '--dropout', type=float, default=TrainConfig.dropout, help_text='selector 网络中的 dropout。')
    add(model_group, '--load_path', type=str, default=None, help_text='若提供，则从该 checkpoint 恢复 selector 参数后继续训练。')

    # ===== 动作空间与增强 =====
    action_group = parser.add_argument_group('动作空间与增强')
    add(action_group, '--decoder_strategy', type=str, default=TrainConfig.decoder_strategy, help_text='保留字段；主要用于与 EasyNCO solver_zoo 接口保持兼容。')
    add(action_group, '--augmentations', nargs='+', default=list(TrainConfig.augmentations), help_text='启用的在线增强类型列表；identity 会自动保留。')
    add(action_group, '--mix_prop', type=float, default=TrainConfig.mix_prop, help_text='mix 增强内部的混合比例。')
    add(action_group, '--init_zoo', nargs='+', default=list(TrainConfig.init_zoo), help_text='初始化器候选列表。每个样本先从这里选一次。')
    add(action_group, '--operator_zoo', nargs='+', default=list(TrainConfig.operator_zoo), help_text='迭代 operator 候选列表。初始化后每一步从这里选一个。')

    # ===== solver / checkpoint =====
    solver_group = parser.add_argument_group('solver 附加配置与 checkpoint 覆盖')
    add(solver_group, '--solver_ckpt_dir', type=str, default=TrainConfig.solver_ckpt_dir, help_text='默认预训练 solver checkpoint 搜索目录。')
    add(solver_group, '--two_opt_iterations', type=int, default=TrainConfig.two_opt_iterations, help_text='two-opt operator 每次调用时的内部迭代次数。')
    add(solver_group, '--glop_pomo_size', type=int, default=TrainConfig.glop_pomo_size, help_text='GLOP 相关流程使用的 POMO 宽度。')
    add(solver_group, '--glop_aug_factor', type=int, default=TrainConfig.glop_aug_factor, help_text='GLOP 使用的增强倍数。')
    add(solver_group, '--glop_revision_iters', type=int, default=TrainConfig.glop_revision_iters, help_text='GLOP revision operator 的迭代次数。')
    add(solver_group, '--heuristic_init_restarts', type=int, default=TrainConfig.heuristic_init_restarts, help_text='启发式初始化器的重启次数。')
    add(solver_group, '--heuristic_metric_strategy', type=str, default=TrainConfig.heuristic_metric_strategy, help_text='启发式初始化器使用的距离度量策略。')
    add(solver_group, '--heuristic_metric_p', type=int, default=TrainConfig.heuristic_metric_p, help_text='启发式距离度量的 p 范数。')
    add(solver_group, '--heuristic_regret_k', type=int, default=TrainConfig.heuristic_regret_k, help_text='regret insertion 中 regret-k 的参数。')
    add(solver_group, '--pomo_size', type=int, default=None, help_text='若要覆盖默认 POMO 宽度，可在这里指定。')
    add(solver_group, '--pomo_ckpt', type=str, default=None, help_text='显式指定 POMO checkpoint 路径。')
    add(solver_group, '--am_ckpt', type=str, default=None, help_text='显式指定 Attention Model checkpoint 路径。')
    add(solver_group, '--lehd_ckpt', type=str, default=None, help_text='显式指定 LEHD checkpoint 路径。')
    add(solver_group, '--elg_ckpt', type=str, default=None, help_text='显式指定 ELG checkpoint 路径。')
    add(solver_group, '--difusco_ckpt', type=str, default=None, help_text='显式指定 DIFUSCO checkpoint 路径。')
    add(solver_group, '--glop_ckpt', type=str, default=None, help_text='显式指定 GLOP checkpoint 路径。')
    add(solver_group, '--dact_ckpt', type=str, default=None, help_text='显式指定 DACT checkpoint 路径。')
    add(solver_group, '--lih_ckpt', type=str, default=None, help_text='显式指定 LIH checkpoint 路径。')

    args = parser.parse_args()

    # `batch_size` 只是为了和旧脚本习惯兼容；在线训练里真实使用的是增强 batch。
    effective_batch_size = args.batch_size if args.batch_size is not None else args.aug_batch_size
    return TrainConfig(
        seed=args.seed,
        device=args.device,
        tsplib_root=args.tsplib_root,
        instance_name=args.instance_name,
        instance_path=args.instance_path,
        output_dir=args.output_dir,
        problem_size=None,
        train_steps=args.train_steps,
        aug_batch_size=effective_batch_size,
        eval_aug_size=args.eval_aug_size,
        eval_batch_size=args.eval_batch_size,
        eval_every=args.eval_every,
        save_every=args.save_every,
        rollout_steps=args.rollout_steps,
        log_every=args.log_every,
        rollout_log_every=args.rollout_log_every,
        log_instance_details=(not args.disable_instance_logs),
        log_prob_precision=args.log_prob_precision,
        lr=args.lr,
        entropy_coef=args.entropy_coef,
        grad_clip=args.grad_clip,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        cpe_dim=args.cpe_dim,
        dropout=args.dropout,
        decoder_strategy=args.decoder_strategy,
        init_zoo=tuple(args.init_zoo),
        operator_zoo=tuple(args.operator_zoo),
        augmentations=tuple(args.augmentations),
        mix_prop=args.mix_prop,
        solver_ckpt_dir=args.solver_ckpt_dir,
        easynco_solver_device=args.easynco_solver_device,
        dact_solver_device=args.dact_solver_device,
        two_opt_iterations=args.two_opt_iterations,
        glop_pomo_size=args.glop_pomo_size,
        glop_aug_factor=args.glop_aug_factor,
        glop_revision_iters=args.glop_revision_iters,
        heuristic_init_restarts=args.heuristic_init_restarts,
        heuristic_metric_strategy=args.heuristic_metric_strategy,
        heuristic_metric_p=args.heuristic_metric_p,
        heuristic_regret_k=args.heuristic_regret_k,
        pomo_size=args.pomo_size,
        pomo_ckpt=args.pomo_ckpt,
        am_ckpt=args.am_ckpt,
        lehd_ckpt=args.lehd_ckpt,
        elg_ckpt=args.elg_ckpt,
        difusco_ckpt=args.difusco_ckpt,
        glop_ckpt=args.glop_ckpt,
        dact_ckpt=args.dact_ckpt,
        lih_ckpt=args.lih_ckpt,
        load_path=args.load_path,
    )


if __name__ == '__main__':
    run(parse_args())

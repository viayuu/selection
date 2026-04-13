from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path

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
from train_online_instance_policy import (
    TrainConfig,
    _chunk_batch,
    _format_aug_counts,
    _format_distribution,
    _make_logger,
    _set_seed,
)

ensure_local_easynco()



def _dict_to_cfg(data: dict) -> TrainConfig:
    # checkpoint 里存的是 JSON 友好的 dict，这里把它重新恢复为 TrainConfig。
    # 其中 list -> tuple 的归一化很重要，否则后续格式化和比较会不一致。
    payload = dict(data)
    for key in ('init_zoo', 'operator_zoo', 'augmentations'):
        if key in payload and isinstance(payload[key], list):
            payload[key] = tuple(payload[key])
    valid = {name for name in TrainConfig.__dataclass_fields__.keys()}
    payload = {k: v for k, v in payload.items() if k in valid}
    return TrainConfig(**payload)



def _write_outputs(output_dir: Path, cfg: TrainConfig, instance_name: str, original_metrics: dict, augmented_metrics: dict):
    payload = {
        'instance_name': instance_name,
        'config': asdict(cfg),
        'original': original_metrics,
        'augmented': augmented_metrics,
    }
    (output_dir / 'eval_metrics.json').write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding='utf-8')

    lines = []
    lines.append('# Online Instance Evaluation')
    lines.append('')
    lines.append(f"- Instance: `{instance_name}`")
    lines.append(f"- Rollout steps: `{cfg.rollout_steps}`")
    lines.append(f"- Init zoo: `{', '.join(cfg.init_zoo)}`")
    lines.append(f"- Operator zoo: `{', '.join(cfg.operator_zoo)}`")
    lines.append('')
    lines.append('## Original')
    lines.append('')
    lines.append(f"- Init length: `{original_metrics['init_length']:.6f}`")
    lines.append(f"- Final length: `{original_metrics['final_length']:.6f}`")
    lines.append(f"- Improvement: `{original_metrics['improvement']:.6f}`")
    lines.append(f"- Reward mean: `{original_metrics['reward_mean']:.6f}`")
    lines.append(f"- Init distribution: `{_format_distribution(cfg.init_zoo, original_metrics['init_distribution'], cfg.log_prob_precision)}`")
    lines.append(f"- Operator distribution: `{_format_distribution(cfg.operator_zoo, original_metrics['operator_distribution'], cfg.log_prob_precision)}`")
    lines.append(f"- Trace: `{original_metrics['trace_path']}`")
    lines.append(f"- Step summary: `{original_metrics['trace_step_summary_path']}`")
    lines.append('')
    lines.append('## Augmented')
    lines.append('')
    lines.append(f"- Init length: `{augmented_metrics['init_length']:.6f}`")
    lines.append(f"- Final length: `{augmented_metrics['final_length']:.6f}`")
    lines.append(f"- Improvement: `{augmented_metrics['improvement']:.6f}`")
    lines.append(f"- Reward mean: `{augmented_metrics['reward_mean']:.6f}`")
    lines.append(f"- Init distribution: `{_format_distribution(cfg.init_zoo, augmented_metrics['init_distribution'], cfg.log_prob_precision)}`")
    lines.append(f"- Operator distribution: `{_format_distribution(cfg.operator_zoo, augmented_metrics['operator_distribution'], cfg.log_prob_precision)}`")
    lines.append(f"- Trace: `{augmented_metrics['trace_path']}`")
    lines.append(f"- Step summary: `{augmented_metrics['trace_step_summary_path']}`")
    (output_dir / 'REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')



def _log_eval_config_details(log_fn, cfg: TrainConfig, args, instance, output_dir: Path, checkpoint_path: str):
    rows = [
        ('checkpoint', checkpoint_path, '待评测的 selector checkpoint。'),
        ('device', cfg.device, 'selector 策略网络所在设备。'),
        ('easynco_solver_device', cfg.easynco_solver_device, 'EasyNCO 初始化器/迭代器推理设备。'),
        ('dact_solver_device', cfg.dact_solver_device, 'DACT operator 推理设备。'),
        ('instance_name', instance.name, '本次评测实例名。'),
        ('instance_path', instance.path, '本次评测实例实际文件路径。'),
        ('problem_size', instance.problem_size, '真实实例规模。'),
        ('output_dir', str(output_dir), '评测输出目录。'),
        ('eval_aug_size', cfg.eval_aug_size, 'augmented evaluation 中的样本总数。'),
        ('eval_batch_size', cfg.eval_batch_size, '评测时的分块 batch 大小。'),
        ('eval_log_every_batches', args.eval_log_every_batches, '评测时每隔多少个 batch 打一轮聚合日志。'),
        ('rollout_steps', cfg.rollout_steps, '每个样本初始化后固定执行多少步 operator 选择。'),
        ('log_instance_details', cfg.log_instance_details, '是否把每个实例每一步动作和概率写入 eval.log / 终端。'),
        ('log_prob_precision', cfg.log_prob_precision, '日志中概率向量的小数位数。'),
        ('init_zoo', ', '.join(cfg.init_zoo), '初始化器候选集合。'),
        ('operator_zoo', ', '.join(cfg.operator_zoo), '迭代 operator 候选集合。'),
        ('augmentations', ', '.join(('identity',) + tuple(cfg.augmentations)), '评测增强集合；identity 会自动保留。'),
        ('mix_prop', cfg.mix_prop, 'mix 增强内部的混合比例。'),
        ('pomo_ckpt', cfg.pomo_ckpt, 'POMO checkpoint 覆盖项。'),
        ('am_ckpt', cfg.am_ckpt, 'AM checkpoint 覆盖项。'),
        ('lehd_ckpt', cfg.lehd_ckpt, 'LEHD checkpoint 覆盖项。'),
        ('elg_ckpt', cfg.elg_ckpt, 'ELG checkpoint 覆盖项。'),
        ('difusco_ckpt', cfg.difusco_ckpt, 'DIFUSCO checkpoint 覆盖项。'),
        ('glop_ckpt', cfg.glop_ckpt, 'GLOP checkpoint 覆盖项。'),
        ('dact_ckpt', cfg.dact_ckpt, 'DACT checkpoint 覆盖项。'),
        ('lih_ckpt', cfg.lih_ckpt, 'LIH checkpoint 覆盖项。'),
    ]
    coord_min = float(instance.coords.min().item())
    coord_max = float(instance.coords.max().item())
    coord_mean = float(instance.coords.mean().item())
    log_fn('[config] ===== 评测配置开始 =====')
    log_fn(f'[config] instance.coord_min={coord_min:.6f} | 归一化后坐标最小值。')
    log_fn(f'[config] instance.coord_max={coord_max:.6f} | 归一化后坐标最大值。')
    log_fn(f'[config] instance.coord_mean={coord_mean:.6f} | 归一化后坐标均值。')
    for name, value, desc in rows:
        log_fn(f'[config] {name}={value} | {desc}')
    log_fn('[config] ===== 评测配置结束 =====')



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Evaluate single-instance online RL selector for TSP.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    def add(group, *flags, help_text: str, **kwargs):
        group.add_argument(*flags, help=help_text, **kwargs)

    required_group = parser.add_argument_group('必选参数')
    add(required_group, '--checkpoint', type=str, required=True, help_text='待评测的 selector checkpoint 路径。')

    instance_group = parser.add_argument_group('实例定位')
    add(instance_group, '--tsplib_root', type=str, default=None, help_text='TSPLIB 根目录；若不直接给 `--instance_path`，则会在这个目录下搜索实例。')
    add(instance_group, '--instance_name', type=str, default=None, help_text='实例名，例如 `kroA100`；通常与 `--tsplib_root` 配合使用。')
    add(instance_group, '--instance_path', type=str, default=None, help_text='直接给出实例文件路径；若提供，则优先于 `tsplib_root + instance_name`。')
    add(instance_group, '--output_dir', type=str, default=None, help_text='评测输出目录；为空时默认写到 checkpoint 同目录下的 `eval_<实例名>`。')

    override_group = parser.add_argument_group('配置覆盖项')
    add(override_group, '--device', type=str, default=None, help_text='覆盖 checkpoint 中记录的 selector 设备。')
    add(override_group, '--easynco_solver_device', type=str, default=None, help_text='覆盖 EasyNCO solver 的推理设备。')
    add(override_group, '--dact_solver_device', type=str, default=None, help_text='覆盖 DACT solver 的推理设备。')
    add(override_group, '--eval_aug_size', type=int, default=None, help_text='覆盖 augmented eval set 的样本数。')
    add(override_group, '--eval_batch_size', type=int, default=None, help_text='覆盖评测分块 batch 大小。')
    add(override_group, '--rollout_steps', type=int, default=None, help_text='覆盖每个样本的 rollout 步数。')
    add(override_group, '--augmentations', nargs='+', default=None, help_text='覆盖增强集合；identity 会自动保留。')
    add(override_group, '--seed', type=int, default=None, help_text='覆盖随机种子；主要影响 augmented eval set 的采样。')
    add(override_group, '--pomo_ckpt', type=str, default=None, help_text='覆盖 POMO checkpoint 路径。')
    add(override_group, '--am_ckpt', type=str, default=None, help_text='覆盖 AM checkpoint 路径。')
    add(override_group, '--lehd_ckpt', type=str, default=None, help_text='覆盖 LEHD checkpoint 路径。')
    add(override_group, '--elg_ckpt', type=str, default=None, help_text='覆盖 ELG checkpoint 路径。')
    add(override_group, '--difusco_ckpt', type=str, default=None, help_text='覆盖 DIFUSCO checkpoint 路径。')
    add(override_group, '--glop_ckpt', type=str, default=None, help_text='覆盖 GLOP checkpoint 路径。')
    add(override_group, '--dact_ckpt', type=str, default=None, help_text='覆盖 DACT checkpoint 路径。')
    add(override_group, '--lih_ckpt', type=str, default=None, help_text='覆盖 LIH checkpoint 路径。')
    add(override_group, '--log_prob_precision', type=int, default=None, help_text='覆盖日志中概率向量的小数位数。')

    log_group = parser.add_argument_group('日志与可观测性')
    add(log_group, '--eval_log_every_batches', type=int, default=1, help_text='评测时每隔多少个 batch 打一轮聚合日志。默认 1 表示每个 batch 都打印。')
    log_group.add_argument(
        '--disable_instance_logs',
        action='store_true',
        help='关闭逐实例逐步日志。默认会把每个实例每一步的 init/operator 选择与概率写入 eval.log 和终端。',
    )
    return parser.parse_args()



def main():
    args = parse_args()

    # 先加载 checkpoint 自带配置，再用 CLI 覆盖。优先级为：
    # 显式命令行参数 > checkpoint 内配置 > TrainConfig 默认值。
    ckpt = torch.load(args.checkpoint, map_location='cpu')
    ckpt_cfg = _dict_to_cfg(ckpt['config'] if isinstance(ckpt, dict) and 'config' in ckpt else {})
    cfg = replace(
        ckpt_cfg,
        device=args.device if args.device is not None else ckpt_cfg.device,
        tsplib_root=args.tsplib_root if args.tsplib_root is not None else ckpt_cfg.tsplib_root,
        instance_name=args.instance_name if args.instance_name is not None else ckpt_cfg.instance_name,
        instance_path=args.instance_path if args.instance_path is not None else ckpt_cfg.instance_path,
        easynco_solver_device=args.easynco_solver_device if args.easynco_solver_device is not None else ckpt_cfg.easynco_solver_device,
        dact_solver_device=args.dact_solver_device if args.dact_solver_device is not None else ckpt_cfg.dact_solver_device,
        eval_aug_size=args.eval_aug_size if args.eval_aug_size is not None else ckpt_cfg.eval_aug_size,
        eval_batch_size=args.eval_batch_size if args.eval_batch_size is not None else ckpt_cfg.eval_batch_size,
        rollout_steps=args.rollout_steps if args.rollout_steps is not None else ckpt_cfg.rollout_steps,
        augmentations=tuple(args.augmentations) if args.augmentations is not None else ckpt_cfg.augmentations,
        seed=args.seed if args.seed is not None else ckpt_cfg.seed,
        log_instance_details=(False if args.disable_instance_logs else ckpt_cfg.log_instance_details),
        log_prob_precision=args.log_prob_precision if args.log_prob_precision is not None else ckpt_cfg.log_prob_precision,
        pomo_ckpt=args.pomo_ckpt if args.pomo_ckpt is not None else ckpt_cfg.pomo_ckpt,
        am_ckpt=args.am_ckpt if args.am_ckpt is not None else ckpt_cfg.am_ckpt,
        lehd_ckpt=args.lehd_ckpt if args.lehd_ckpt is not None else ckpt_cfg.lehd_ckpt,
        elg_ckpt=args.elg_ckpt if args.elg_ckpt is not None else ckpt_cfg.elg_ckpt,
        difusco_ckpt=args.difusco_ckpt if args.difusco_ckpt is not None else ckpt_cfg.difusco_ckpt,
        glop_ckpt=args.glop_ckpt if args.glop_ckpt is not None else ckpt_cfg.glop_ckpt,
        dact_ckpt=args.dact_ckpt if args.dact_ckpt is not None else ckpt_cfg.dact_ckpt,
        lih_ckpt=args.lih_ckpt if args.lih_ckpt is not None else ckpt_cfg.lih_ckpt,
    )

    _set_seed(cfg.seed)

    # 评测时同样以真实实例规模覆盖 `problem_size`，避免 checkpoint 里的旧值失真。
    instance = load_tsplib_instance(
        tsplib_root=cfg.tsplib_root,
        instance_name=cfg.instance_name,
        instance_path=cfg.instance_path,
        device=cfg.device,
    )
    cfg = replace(cfg, problem_size=instance.problem_size)
    if args.output_dir is None:
        output_dir = Path(args.checkpoint).resolve().parent / f'eval_{instance.name}'
    else:
        output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 评测日志同时写终端和 `eval.log`，便于边跑边观察。
    log, handle = _make_logger(output_dir / 'eval.log')
    try:
        log(f"[eval] checkpoint={args.checkpoint}")
        log(f"[eval] instance={instance.name} problem_size={instance.problem_size} path={instance.path}")
        log(f"[eval] init_zoo={','.join(cfg.init_zoo)} operator_zoo={','.join(cfg.operator_zoo)}")
        log(f"[eval] augmentations={','.join(('identity',) + tuple(cfg.augmentations))} eval_aug_size={cfg.eval_aug_size}")
        _log_eval_config_details(log, cfg, args, instance, output_dir, args.checkpoint)
        patched = enable_easynco_runtime_patches(cfg.easynco_solver_device)
        if patched:
            log(f"[eval] enabled_easynco_runtime_patches={','.join(patched)}")

        init_zoo = build_initializer_zoo(cfg, log=log)
        operator_zoo = build_operator_zoo(cfg, log=log)
        policy = TSPSelectorPolicy(
            num_initializers=len(init_zoo),
            num_operators=len(operator_zoo),
            hidden_dim=cfg.hidden_dim,
            num_heads=cfg.num_heads,
            num_layers=cfg.num_layers,
            cpe_dim=cfg.cpe_dim,
            dropout=cfg.dropout,
        ).to(cfg.device)
        state_dict = ckpt['state_dict'] if isinstance(ckpt, dict) and 'state_dict' in ckpt else ckpt
        policy.load_state_dict(state_dict)
        env = TSPImprovementEnv(init_zoo=init_zoo, operator_zoo=operator_zoo, rollout_steps=cfg.rollout_steps)
        trainer = ReinforceTrainer(
            policy=policy,
            env=env,
            optimizer=None,
            rollout_steps=cfg.rollout_steps,
            init_names=cfg.init_zoo,
            op_names=cfg.operator_zoo,
        )

        original_coords = instance.coords.unsqueeze(0)
        original_types = ('identity',)
        log('[eval] starting original greedy evaluation')
        original_metrics = trainer.evaluate_batches(
            _chunk_batch(original_coords, original_types, 1),
            mode='eval_original',
            progress_log_fn=log,
            target_items=1,
            log_every_batches=max(args.eval_log_every_batches, 1),
            trace_path=output_dir / 'original_instance_actions.jsonl',
            log_instance_details=cfg.log_instance_details,
            log_prob_precision=cfg.log_prob_precision,
        )
        log(f"[eval] original init_dist={_format_distribution(cfg.init_zoo, original_metrics['init_distribution'], cfg.log_prob_precision)}")
        log(f"[eval] original op_dist={_format_distribution(cfg.operator_zoo, original_metrics['operator_distribution'], cfg.log_prob_precision)}")
        log(f"[eval] original trace={original_metrics['trace_path']}")
        log(f"[eval] original step_summary={original_metrics['trace_step_summary_path']}")

        eval_augmentor = OnlineAugmentationGenerator(instance.coords, augmentations=cfg.augmentations, mix_prop=cfg.mix_prop, seed=cfg.seed + 1000)
        aug_batch = eval_augmentor.sample(cfg.eval_aug_size, device=cfg.device, deterministic_seed=cfg.seed + 77)
        log(f"[eval] starting augmented greedy evaluation aug_counts={_format_aug_counts(aug_batch.augmentation_types)}")
        augmented_metrics = trainer.evaluate_batches(
            _chunk_batch(aug_batch.coords, aug_batch.augmentation_types, cfg.eval_batch_size),
            mode='eval_augmented',
            progress_log_fn=log,
            target_items=aug_batch.coords.size(0),
            log_every_batches=max(args.eval_log_every_batches, 1),
            trace_path=output_dir / 'augmented_instance_actions.jsonl',
            log_instance_details=cfg.log_instance_details,
            log_prob_precision=cfg.log_prob_precision,
        )
        log(f"[eval] augmented init_dist={_format_distribution(cfg.init_zoo, augmented_metrics['init_distribution'], cfg.log_prob_precision)}")
        log(f"[eval] augmented op_dist={_format_distribution(cfg.operator_zoo, augmented_metrics['operator_distribution'], cfg.log_prob_precision)}")
        log(f"[eval] augmented trace={augmented_metrics['trace_path']}")
        log(f"[eval] augmented step_summary={augmented_metrics['trace_step_summary_path']}")

        _write_outputs(output_dir, cfg, instance.name, original_metrics, augmented_metrics)
        log(f"[eval] outputs={output_dir}")
        print(json.dumps({'original': original_metrics, 'augmented': augmented_metrics}, indent=2, ensure_ascii=False))
    finally:
        handle.close()


if __name__ == '__main__':
    main()

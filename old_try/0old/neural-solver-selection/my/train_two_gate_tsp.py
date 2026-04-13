from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from easynco_bootstrap import ensure_local_easynco


def _make_print_logger(log_path: str | None):
    """
    Create a lightweight logger that prints to stdout and (optionally) appends to a txt file.

    Returns:
        (log_fn, close_fn)
    """
    if log_path is None or str(log_path).strip() == "":
        return print, (lambda: None)

    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fp = open(path, "a", encoding="utf-8", buffering=1)

    def _log(*args, **kwargs):
        print(*args, **kwargs)
        file_kwargs = dict(kwargs)
        file_kwargs.pop("file", None)
        # Always flush file logs so we don't lose progress on interruption.
        file_kwargs["flush"] = True
        print(*args, file=fp, **file_kwargs)

    return _log, fp.close


@dataclass(frozen=True)
class TrainConfig:
    seed: int = 2024
    device: str = "cpu"
    problem_size: int = 50
    batch_size: int = 32
    train_steps: int = 200
    lr: float = 1e-3
    log_every: int = 20
    eval_every: int = 100
    eval_batches: int = 3

    # Data generation (follow the paper's synthetic distributions)
    # - train: infinite stream (generated on the fly)
    # - eval: fixed dataset (generated once per run, reproducible by eval_seed)
    train_dists: tuple[str, ...] = ("uniform",)
    eval_dist: str = "uniform"
    eval_seed: int = 2025
    eval_data_path: str | None = None  # optional: load a fixed eval .pt instead of generating

    # Gaussian-mixture params (copied from `neural-solver-selection/datasets/data_config.yml`)
    gmm_no_cov: bool = False
    gmm_var_lower: float = 1.0
    gmm_var_upper: float = 100.0
    gmm_num_modes_lower: int = 0
    gmm_num_modes_upper: int = 15
    gmm_center_lower: float = 0.0
    gmm_center_upper: float = 100.0

    # Zoo config (focus: make the framework run; not performance)
    init_zoo: tuple[str, ...] = ("pomo", "am")
    iter_zoo: tuple[str, ...] = ("none", "rrc_lehd")

    # Solver hyperparams
    pomo_size: int | None = None  # default: = problem_size
    decoder_strategy: Literal["greedy", "sampling"] = "greedy"
    rrc_steps: int = 10
    dact_steps: int = 10
    lih_steps: int = 10

    # Optional solver checkpoints (framework first; performance optional)
    model_dir: str | None = "model"  # auto-resolve ckpts in this dir when explicit paths are not provided
    pomo_ckpt: str | None = None
    am_ckpt: str | None = None
    lehd_ckpt: str | None = None  # used by both lehd initializer and rrc iterator
    elg_ckpt: str | None = None
    difusco_ckpt: str | None = None
    dact_ckpt: str | None = None
    lih_ckpt: str | None = None

    # Optional: include a deterministic iterator for debugging
    add_two_opt: bool = False
    two_opt_iters: int = 10

    # Runtime options
    load_path: str | None = None
    only_eval: bool = False

    # Baseline / AC options
    baseline: Literal["batch_mean", "critic", "critic_batch_mean"] = "batch_mean"
    critic_coef: float = 0.5


def _set_seed(seed: int):
    import random

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _configure_torch_defaults(device: str) -> None:
    """
    Configure global torch defaults so `final/` components work on the intended device.

    Rationale:
    - EasyNCO envs / pipelines sometimes create tensors without explicit `device=...`.
    - If we run on CUDA but keep the default tensor type on CPU, this can trigger CPU/CUDA mismatch errors.
    """
    import torch

    if device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(f"Requested device='{device}' but CUDA is not available.")
        # Ensure the current CUDA device matches the requested one.
        if ":" in device:
            _, idx = device.split(":", 1)
            if idx.strip() != "":
                torch.cuda.set_device(int(idx))
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
    else:
        torch.set_default_tensor_type("torch.FloatTensor")


def _make_tsp_batch(batch_size: int, problem_size: int, device: str):
    import torch

    return torch.rand((batch_size, problem_size, 2), device=device)


def _load_tsp_coords_pt(path: str):
    import torch

    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict):
        # EasyNCO-style datasets commonly store coordinates under "node_xy".
        for k in ("node_xy", "coords", "locs", "problems", "data"):
            if k in obj:
                coords = obj[k]
                break
        else:
            raise KeyError(f"Unsupported .pt dict format (no coords key found). Keys={list(obj.keys())[:20]}")
    else:
        coords = obj

    if not isinstance(coords, torch.Tensor):
        raise TypeError(f"Unsupported coords type: {type(coords)} (expect torch.Tensor or dict containing tensor)")
    if coords.ndim != 3 or coords.size(-1) != 2:
        raise ValueError(f"Expected coords shape [num,N,2], got {tuple(coords.shape)}")

    return coords.float()


class _TempSeed:
    """
    Temporarily set RNG seeds for reproducible evaluation, then restore states.
    """

    def __init__(self, seed: int):
        self.seed = int(seed)

    def __enter__(self):
        import random

        import numpy as np
        import torch

        self._py_state = random.getstate()
        self._np_state = np.random.get_state()
        self._torch_state = torch.random.get_rng_state()
        self._cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None

        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        return self

    def __exit__(self, exc_type, exc, tb):
        import random

        import numpy as np
        import torch

        random.setstate(self._py_state)
        np.random.set_state(self._np_state)
        torch.random.set_rng_state(self._torch_state)
        if self._cuda_state is not None:
            torch.cuda.set_rng_state_all(self._cuda_state)
        return False


def _auto_find_ckpt(model_dir: str | None, keyword: str) -> str | None:
    """
    Find a checkpoint file in `model_dir` whose filename contains `keyword` (case-insensitive).
    """
    if model_dir is None:
        return None
    root = Path(model_dir)
    if not root.is_dir():
        return None

    keyword_l = keyword.lower()
    exts = {".ckpt", ".pt", ".pth"}
    candidates = [
        p for p in root.iterdir()
        if p.is_file()
        and p.suffix.lower() in exts
        and keyword_l in p.name.lower()
    ]
    candidates.sort()
    return str(candidates[0]) if candidates else None


def _build_initializer_zoo(cfg: TrainConfig, *, log=print):
    # These imports require torch + tensordict + torchrl, and rely on EasyNCO alias.
    from EasyNCO.neural_solvers.envs.TSPEnv import TSPEnv
    from EasyNCO.neural_solvers.methods.am.policy import AttentionModelPolicy
    from EasyNCO.neural_solvers.methods.lehd.initialization import LEHDInitialization
    from EasyNCO.neural_solvers.methods.lehd.policy import LEHDPolicy
    from EasyNCO.neural_solvers.methods.pomo.initialization import POMOInitialization
    from EasyNCO.neural_solvers.methods.pomo.policy import POMOPolicy
    from EasyNCO.neural_solvers.pipeline.initialization import ARInitialization

    from my.solver_zoo import DifuscoInitializer, NeuralARInitializer
    from my.checkpoints import load_policy_checkpoint

    pomo_size = cfg.pomo_size or cfg.problem_size

    zoo = []
    for name in cfg.init_zoo:
        if name == "pomo":
            env = TSPEnv(problem_size=cfg.problem_size, pomo_size=pomo_size, device=cfg.device, aug_type=None, aug_factor=1)
            policy = POMOPolicy(env_name="tsp")
            policy.to(cfg.device)
            ckpt_path = cfg.pomo_ckpt or _auto_find_ckpt(cfg.model_dir, "pomo")
            if ckpt_path is not None:
                rep = load_policy_checkpoint(policy, ckpt_path, device=cfg.device)
                log(f"[solver_load] pomo: {rep['path']} missing={len(rep['missing_keys'])} unexpected={len(rep['unexpected_keys'])}")
            policy.requires_grad_(False)
            init = POMOInitialization(policy=policy)
            zoo.append(
                NeuralARInitializer(name="pomo", env=env, initialization=init, decoder_strategy=cfg.decoder_strategy)
            )
        elif name == "am":
            env = TSPEnv(problem_size=cfg.problem_size, pomo_size=1, device=cfg.device, aug_type=None, aug_factor=1)
            policy = AttentionModelPolicy(env_name="tsp")
            policy.to(cfg.device)
            ckpt_path = cfg.am_ckpt or _auto_find_ckpt(cfg.model_dir, "am")
            if ckpt_path is not None:
                rep = load_policy_checkpoint(policy, ckpt_path, device=cfg.device)
                log(f"[solver_load] am: {rep['path']} missing={len(rep['missing_keys'])} unexpected={len(rep['unexpected_keys'])}")
            policy.requires_grad_(False)
            init = ARInitialization(policy=policy)
            zoo.append(
                NeuralARInitializer(name="am", env=env, initialization=init, decoder_strategy=cfg.decoder_strategy)
            )
        elif name == "lehd":
            env = TSPEnv(
                problem_size=cfg.problem_size,
                pomo_size=1,
                device=cfg.device,
                aug_type=None,
                aug_factor=1,
                method_name="lehd",
            )
            policy = LEHDPolicy(phase="test", env_name="tsp")
            policy.to(cfg.device)
            ckpt_path = cfg.lehd_ckpt or _auto_find_ckpt(cfg.model_dir, "lehd")
            if ckpt_path is not None:
                rep = load_policy_checkpoint(policy, ckpt_path, device=cfg.device)
                log(f"[solver_load] lehd: {rep['path']} missing={len(rep['missing_keys'])} unexpected={len(rep['unexpected_keys'])}")
            policy.requires_grad_(False)
            init = LEHDInitialization(policy=policy)
            zoo.append(
                NeuralARInitializer(name="lehd", env=env, initialization=init, decoder_strategy=cfg.decoder_strategy)
            )
        elif name == "elg":
            from EasyNCO.neural_solvers.methods.elg.initialization import ELGInitialization
            from EasyNCO.neural_solvers.methods.elg.policy import ELGPolicy

            env = TSPEnv(
                problem_size=cfg.problem_size,
                pomo_size=pomo_size,
                device=cfg.device,
                aug_type=None,
                aug_factor=1,
                method_name="elg",
            )
            policy = ELGPolicy(
                env_name="tsp",
                embed_dim=128,
                num_heads=8,
                qkv_dim=16,
                num_encoder_layers=6,
                normalization="instance",
                feedforward_hidden=512,
                logit_clipping=50,
                use_graph_mean=False,
                am_mode=False,
                first_placeholder=False,
                first_mode="random",
                local_enable=False,
                local_size=40,
                xi=-1,
                euclidean=False,
            )
            policy.to(cfg.device)
            ckpt_path = cfg.elg_ckpt or _auto_find_ckpt(cfg.model_dir, "elg")
            if ckpt_path is not None:
                rep = load_policy_checkpoint(policy, ckpt_path, device=cfg.device)
                log(f"[solver_load] elg: {rep['path']} missing={len(rep['missing_keys'])} unexpected={len(rep['unexpected_keys'])}")
            policy.requires_grad_(False)
            init = ELGInitialization(policy=policy)
            zoo.append(
                NeuralARInitializer(name="elg", env=env, initialization=init, decoder_strategy=cfg.decoder_strategy)
            )
        elif name == "difusco":
            from EasyNCO.neural_solvers.methods.difusco.policy import DIFUSCOPolicy

            policy = DIFUSCOPolicy(
                env_name="tsp",
                n_layers=12,
                hidden_dim=256,
                aggregation="sum",
                diffusion_type="categorical",
                diffusion_schedule="linear",
                diffusion_steps=1000,
                sparse_factor=-1,
                use_activation_checkpoint=False,
                parallel_sampling=1,
                sequential_sampling=1,
                inference_diffusion_steps=20,
                inference_schedule="cosine",
                inference_trick="ddim",
                node_feature_only=False,
            )
            model = policy._get_model()
            model.to(cfg.device)
            ckpt_path = cfg.difusco_ckpt or _auto_find_ckpt(cfg.model_dir, "difusco")
            if ckpt_path is not None:
                rep = load_policy_checkpoint(policy, ckpt_path, device=cfg.device)
                log(f"[solver_load] difusco: {rep['path']} missing={len(rep['missing_keys'])} unexpected={len(rep['unexpected_keys'])}")
            model.requires_grad_(False)
            zoo.append(DifuscoInitializer(name="difusco", model=model))
        else:
            raise ValueError(f"Unknown initializer '{name}'")

    return zoo


def _build_iterator_zoo(cfg: TrainConfig, *, log=print):
    from EasyNCO.neural_solvers.envs.TSPEnv import TSPEnv
    from EasyNCO.neural_solvers.methods.lehd.policy import LEHDPolicy

    from my.solver_zoo import DACTIterator, LIHIterator, NoIterationIterator, RRCLIHStyleLEHDIterator, TwoOptIterator
    from my.checkpoints import load_policy_checkpoint

    zoo = []
    for name in cfg.iter_zoo:
        if name == "none":
            zoo.append(NoIterationIterator())
        elif name == "two_opt":
            zoo.append(TwoOptIterator(max_iterations=cfg.two_opt_iters, device=cfg.device))
        elif name == "rrc_lehd":
            # RRC uses LEHD-style destroy/repair, driven by LEHDPolicy.
            env = TSPEnv(
                problem_size=cfg.problem_size,
                pomo_size=1,
                device=cfg.device,
                aug_type=None,
                aug_factor=1,
                method_name="lehd",
            )
            policy = LEHDPolicy(phase="test", env_name="tsp")
            policy.to(cfg.device)
            ckpt_path = cfg.lehd_ckpt or _auto_find_ckpt(cfg.model_dir, "lehd")
            if ckpt_path is not None:
                rep = load_policy_checkpoint(policy, ckpt_path, device=cfg.device)
                log(f"[solver_load] rrc_lehd(policy): {rep['path']} missing={len(rep['missing_keys'])} unexpected={len(rep['unexpected_keys'])}")
            policy.requires_grad_(False)
            zoo.append(
                RRCLIHStyleLEHDIterator(
                    name=f"rrc_lehd_{cfg.rrc_steps}",
                    env=env,
                    policy=policy,
                    max_steps=cfg.rrc_steps,
                    decoder_strategy=cfg.decoder_strategy,
                )
            )
        elif name == "dact":
            from EasyNCO.neural_solvers.methods.dact.iteration import DACTIteration
            from EasyNCO.neural_solvers.methods.dact.policy import DACTPolicy

            # DACT is a policy-guided 2-opt local search. The current EasyNCO implementation is most robust on CPU.
            solver_device = "cpu"
            env = TSPEnv(problem_size=cfg.problem_size, pomo_size=1, device=solver_device, aug_type=None, aug_factor=1)
            policy = DACTPolicy(env_name="tsp")
            policy.initial_problem_size(cfg.problem_size)
            policy.to(solver_device)
            ckpt_path = cfg.dact_ckpt or _auto_find_ckpt(cfg.model_dir, "dact")
            if ckpt_path is not None:
                rep = load_policy_checkpoint(policy, ckpt_path, device=solver_device)
                log(f"[solver_load] dact: {rep['path']} missing={len(rep['missing_keys'])} unexpected={len(rep['unexpected_keys'])}")
            policy.requires_grad_(False)
            iteration = DACTIteration(policy)
            zoo.append(
                DACTIterator(
                    name=f"dact_{cfg.dact_steps}",
                    env=env,
                    policy=policy,
                    iteration=iteration,
                    max_steps=cfg.dact_steps,
                    solver_device=solver_device,
                )
            )
        elif name == "lih":
            from EasyNCO.neural_solvers.methods.lih.iteration import LIHIteration
            from EasyNCO.neural_solvers.methods.lih.policy import LIHPolicy

            # LIH hard-codes `.cuda()` in multiple places; only enable it when the run device is CUDA.
            if not cfg.device.startswith("cuda"):
                raise RuntimeError("Iterator 'lih' requires --device cuda[:idx]")
            env = TSPEnv(problem_size=cfg.problem_size, pomo_size=1, device=cfg.device, aug_type=None, aug_factor=1)
            policy = LIHPolicy(env_name="tsp")
            policy.to(cfg.device)
            ckpt_path = cfg.lih_ckpt or _auto_find_ckpt(cfg.model_dir, "lih")
            if ckpt_path is not None:
                rep = load_policy_checkpoint(policy, ckpt_path, device=cfg.device)
                log(f"[solver_load] lih: {rep['path']} missing={len(rep['missing_keys'])} unexpected={len(rep['unexpected_keys'])}")
            policy.requires_grad_(False)
            iteration = LIHIteration(policy)
            zoo.append(
                LIHIterator(
                    name=f"lih_{cfg.lih_steps}",
                    env=env,
                    policy=policy,
                    iteration=iteration,
                    max_steps=cfg.lih_steps,
                )
            )
        else:
            raise ValueError(f"Unknown iterator '{name}'")

    if cfg.add_two_opt:
        zoo.append(TwoOptIterator(max_iterations=cfg.two_opt_iters, device=cfg.device))

    return zoo


def _build_gates(cfg: TrainConfig, num_init: int, num_iter: int):
    import torch
    import torch.optim as optim

    from my.features import FeatureConfig, build_paper_tsp_encoder, tsp_gate2_features, tsp_instance_features
    from my.gates import GateMLPConfig, TwoGateAC, build_mlp, build_value_mlp

    feat_cfg = FeatureConfig()
    encoder = build_paper_tsp_encoder(feat_cfg)
    if encoder is not None:
        encoder.to(cfg.device)

    # infer dims by one dummy forward
    dummy = torch.rand((2, cfg.problem_size, 2), device=cfg.device)
    d1 = tsp_instance_features(dummy, cfg=feat_cfg, encoder=encoder).size(1)
    dummy_tour = torch.arange(cfg.problem_size, device=cfg.device)[None, :].repeat(2, 1)
    d2 = tsp_gate2_features(
        dummy,
        dummy_tour,
        torch.ones(2, device=cfg.device),
        cfg=feat_cfg,
        encoder=encoder,
    ).size(1)

    gate1 = build_mlp(GateMLPConfig(in_dim=d1, out_dim=num_init))
    gate2 = build_mlp(GateMLPConfig(in_dim=d2, out_dim=num_iter))
    value1 = None
    value2 = None
    if cfg.baseline in ("critic", "critic_batch_mean"):
        value1 = build_value_mlp(in_dim=d1)
        value2 = build_value_mlp(in_dim=d2)

    policy = TwoGateAC(encoder=encoder, gate1=gate1, gate2=gate2, value1=value1, value2=value2).to(cfg.device)

    optimizer = optim.Adam(list(policy.parameters()), lr=cfg.lr)
    return policy, optimizer, feat_cfg


def train(cfg: TrainConfig, *, log=print):
    import torch
    from torch.distributions import Categorical

    ensure_local_easynco()

    from my.features import tsp_gate2_features, tsp_instance_features
    from my.solver_zoo import run_initializers_by_action, run_iterators_by_action
    from my.tsp_data import GaussianMixtureParams, TSPBatchGenerator

    _configure_torch_defaults(cfg.device)
    _set_seed(cfg.seed)

    gmm_params = GaussianMixtureParams(
        no_cov=cfg.gmm_no_cov,
        var_lower=cfg.gmm_var_lower,
        var_upper=cfg.gmm_var_upper,
        num_modes_lower=cfg.gmm_num_modes_lower,
        num_modes_upper=cfg.gmm_num_modes_upper,
        center_lower=cfg.gmm_center_lower,
        center_upper=cfg.gmm_center_upper,
    )
    train_gen = TSPBatchGenerator(
        problem_size=cfg.problem_size,
        distributions=cfg.train_dists,
        gaussian_params=gmm_params,
        seed=cfg.seed,
    )

    init_zoo = _build_initializer_zoo(cfg, log=log)
    iter_zoo = _build_iterator_zoo(cfg, log=log)
    policy, optimizer, feat_cfg = _build_gates(cfg, num_init=len(init_zoo), num_iter=len(iter_zoo))
    policy.train()

    # Build a fixed evaluation set (either load from disk or generate once).
    eval_size = int(cfg.eval_batches) * int(cfg.batch_size)
    if cfg.eval_data_path is not None:
        eval_coords = _load_tsp_coords_pt(cfg.eval_data_path)
        if int(eval_coords.size(1)) != int(cfg.problem_size):
            raise ValueError(
                f"eval_data_path N={int(eval_coords.size(1))} != cfg.problem_size={int(cfg.problem_size)}"
            )
        # Keep eval cost bounded: use the first `eval_size` instances by default.
        eval_coords = eval_coords[:eval_size]
    else:
        eval_gen = TSPBatchGenerator(
            problem_size=cfg.problem_size,
            distributions=(cfg.eval_dist,),
            gaussian_params=gmm_params,
            seed=cfg.eval_seed,
        )
        eval_coords = eval_gen.sample(eval_size, device="cpu")

    best_eval = float("inf")

    if cfg.load_path is not None:
        ckpt = torch.load(cfg.load_path, map_location=cfg.device)
        if ckpt.get("encoder") is not None and policy.encoder is not None:
            policy.encoder.load_state_dict(ckpt["encoder"])
        policy.gate1.load_state_dict(ckpt["gate1"])
        policy.gate2.load_state_dict(ckpt["gate2"])
        if cfg.baseline in ("critic", "critic_batch_mean"):
            # Backward-compat: older checkpoints used key "value" (single critic on feat1).
            if policy.value1 is not None:
                state = ckpt.get("value1") or ckpt.get("value")
                if state is not None:
                    try:
                        policy.value1.load_state_dict(state)
                    except Exception:
                        pass
            if policy.value2 is not None:
                state = ckpt.get("value2")
                if state is not None:
                    try:
                        policy.value2.load_state_dict(state)
                    except Exception:
                        pass
        if "optimizer" in ckpt:
            try:
                optimizer.load_state_dict(ckpt["optimizer"])
            except Exception:
                # Optimizer state is optional; mismatches are OK for a prototype.
                pass
        log(f"[loaded] {cfg.load_path}")

    if cfg.only_eval:
        eval_metrics = evaluate(cfg, policy, init_zoo, iter_zoo, feat_cfg, eval_coords)
        log(
            f"[only_eval] mean_len={eval_metrics['mean_len']:.4f} "
            f"gate1=({eval_metrics['gate1_dist']}) gate2=({eval_metrics['gate2_dist']})"
        )
        return policy, optimizer

    for step in range(1, cfg.train_steps + 1):
        coords = train_gen.sample(cfg.batch_size, device=cfg.device)

        # Gate1: pick initializer
        feat1 = tsp_instance_features(coords, cfg=feat_cfg, encoder=policy.encoder)
        dist1 = Categorical(logits=policy.gate1(feat1))
        a1 = dist1.sample()
        logp1 = dist1.log_prob(a1)

        sol0 = run_initializers_by_action(coords, a1, init_zoo)

        # Gate2: pick iterator (conditioned on sol0 via length0)
        feat2 = tsp_gate2_features(coords, sol0.tour, sol0.length, cfg=feat_cfg, encoder=policy.encoder)
        dist2 = Categorical(logits=policy.gate2(feat2))
        a2 = dist2.sample()
        logp2 = dist2.log_prob(a2)

        sol1 = run_iterators_by_action(coords, a2, sol0, iter_zoo)

        length = sol1.length
        reward = -length  # your notation: reward = -length

        critic_loss = torch.zeros((), device=coords.device)
        if cfg.baseline == "batch_mean":
            # Your requirement: baseline = mean(length over batch)
            baseline_length = length.mean()
            baseline_reward = -baseline_length  # convert to reward space
            baseline1 = baseline_reward
            baseline2 = baseline_reward
        elif cfg.baseline == "critic_batch_mean":
            if policy.value1 is None or policy.value2 is None:
                raise RuntimeError("baseline=critic_batch_mean but value network is not initialized")
            # Combine:
            #   - batch mean length baseline (a scalar, converted to reward space)
            #   - a learned critic V(s) that predicts the residual reward around that mean
            baseline_length = length.mean()
            baseline_reward = -baseline_length  # scalar
            value1_pred = policy.value1(feat1).squeeze(-1)  # [B]
            value2_pred = policy.value2(feat2).squeeze(-1)  # [B]
            baseline1 = baseline_reward + value1_pred
            baseline2 = baseline_reward + value2_pred
            critic_loss1 = ((reward - baseline_reward - value1_pred) ** 2).mean()
            critic_loss2 = ((reward - baseline_reward - value2_pred) ** 2).mean()
            critic_loss = 0.5 * (critic_loss1 + critic_loss2)
        elif cfg.baseline == "critic":
            if policy.value1 is None or policy.value2 is None:
                raise RuntimeError("baseline=critic but value network is not initialized")
            baseline1 = policy.value1(feat1).squeeze(-1)
            baseline2 = policy.value2(feat2).squeeze(-1)
            critic_loss1 = ((reward - baseline1) ** 2).mean()
            critic_loss2 = ((reward - baseline2) ** 2).mean()
            critic_loss = 0.5 * (critic_loss1 + critic_loss2)
        else:
            raise ValueError(f"Unknown baseline mode: {cfg.baseline}")

        adv1 = reward - baseline1
        adv2 = reward - baseline2

        actor_loss = -(logp1 * adv1.detach() + logp2 * adv2.detach()).mean()
        loss = actor_loss + cfg.critic_coef * critic_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step % cfg.log_every == 0 or step == 1:
            with torch.no_grad():
                mean_len = float(length.mean().item())
                mean_l0 = float(sol0.length.mean().item())
                ent = float((dist1.entropy().mean() + dist2.entropy().mean()).item())
                # For logging, report the Gate2 advantage (usually higher-variance and more informative).
                adv_mean = float(adv2.mean().item())
                adv_std = float(adv2.std(unbiased=False).item())
                critic_l = float(critic_loss.item()) if cfg.baseline in ("critic", "critic_batch_mean") else 0.0

                a1_cpu = a1.detach().cpu()
                a2_cpu = a2.detach().cpu()
                p1 = torch.bincount(a1_cpu, minlength=len(init_zoo)).float() / float(a1_cpu.numel())
                p2 = torch.bincount(a2_cpu, minlength=len(iter_zoo)).float() / float(a2_cpu.numel())
                p1_str = ", ".join(f"{init_zoo[i].name}:{p1[i].item():.2f}" for i in range(len(init_zoo)))
                p2_str = ", ".join(f"{iter_zoo[i].name}:{p2[i].item():.2f}" for i in range(len(iter_zoo)))
                log(
                    f"[train step {step:>6}] loss={loss.item():.4f} "
                    f"len0={mean_l0:.4f} len1={mean_len:.4f} "
                    f"adv={adv_mean:.4f}±{adv_std:.4f} critic_mse={critic_l:.4f} entropy={ent:.4f} "
                    f"gate1=({p1_str}) gate2=({p2_str})"
                )

        if step % cfg.eval_every == 0 or step == cfg.train_steps:
            eval_metrics = evaluate(cfg, policy, init_zoo, iter_zoo, feat_cfg, eval_coords)
            eval_len = eval_metrics["mean_len"]
            best_eval = min(best_eval, eval_len)
            log(
                f"[eval step {step:>6}] mean_len={eval_len:.4f} best={best_eval:.4f} "
                f"gate1=({eval_metrics['gate1_dist']}) gate2=({eval_metrics['gate2_dist']})"
            )
            policy.train()

    return policy, optimizer


def evaluate(cfg: TrainConfig, policy, init_zoo, iter_zoo, feat_cfg, eval_coords):
    import torch

    ensure_local_easynco()

    from my.features import tsp_gate2_features, tsp_instance_features
    from my.solver_zoo import run_initializers_by_action, run_iterators_by_action

    policy.eval()

    with torch.no_grad():
        total_len = 0.0
        total_count = 0
        # Keep counters on CPU to avoid device-mismatch when CUDA is the default tensor type.
        a1_counts = torch.zeros((len(init_zoo),), dtype=torch.float32, device="cpu")
        a2_counts = torch.zeros((len(iter_zoo),), dtype=torch.float32, device="cpu")

        # For reproducible evaluation (important when iterators are stochastic, e.g., RRC),
        # we fix RNG seeds during evaluation and restore them afterwards.
        with _TempSeed(cfg.eval_seed):
            for start in range(0, int(eval_coords.size(0)), int(cfg.batch_size)):
                coords = eval_coords[start : start + int(cfg.batch_size)].to(cfg.device)

                logits1 = policy.gate1(tsp_instance_features(coords, cfg=feat_cfg, encoder=policy.encoder))
                a1 = logits1.argmax(dim=1)
                sol0 = run_initializers_by_action(coords, a1, init_zoo)

                logits2 = policy.gate2(
                    tsp_gate2_features(coords, sol0.tour, sol0.length, cfg=feat_cfg, encoder=policy.encoder)
                )
                a2 = logits2.argmax(dim=1)
                sol1 = run_iterators_by_action(coords, a2, sol0, iter_zoo)

                total_len += float(sol1.length.sum().item())
                total_count += int(sol1.length.numel())

                a1_cpu = a1.detach().cpu()
                a2_cpu = a2.detach().cpu()
                a1_counts += torch.bincount(a1_cpu, minlength=len(init_zoo)).float()
                a2_counts += torch.bincount(a2_cpu, minlength=len(iter_zoo)).float()

        mean_len = total_len / max(total_count, 1)
        p1 = a1_counts / max(a1_counts.sum().item(), 1.0)
        p2 = a2_counts / max(a2_counts.sum().item(), 1.0)
        p1_str = ", ".join(f"{init_zoo[i].name}:{p1[i].item():.2f}" for i in range(len(init_zoo)))
        p2_str = ", ".join(f"{iter_zoo[i].name}:{p2[i].item():.2f}" for i in range(len(iter_zoo)))

        return {"mean_len": float(mean_len), "gate1_dist": p1_str, "gate2_dist": p2_str}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem_size", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--train_steps", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--decoder_strategy", type=str, choices=["greedy", "sampling"], default="greedy")
    parser.add_argument(
        "--init_zoo",
        nargs="+",
        default=None,
        help="Initializer zoo list, e.g. --init_zoo pomo am lehd elg difusco",
    )
    parser.add_argument(
        "--iter_zoo",
        nargs="+",
        default=None,
        help="Iteration zoo list, e.g. --iter_zoo none rrc_lehd two_opt dact (lih requires CUDA)",
    )

    parser.add_argument("--pomo_size", type=int, default=0, help="POMO parallel starts; 0 means problem_size")
    parser.add_argument("--rrc_steps", type=int, default=10)
    parser.add_argument("--dact_steps", type=int, default=10)
    parser.add_argument("--lih_steps", type=int, default=10
    )

    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--eval_every", type=int, default=100)
    parser.add_argument("--eval_batches", type=int, default=3)

    parser.add_argument(
        "--train_dists",
        nargs="+",
        default=["gaussian", "uniform"],
        help="Training instance distribution(s). Supported: uniform, gaussian. Example: --train_dists gaussian uniform",
    )
    parser.add_argument(
        "--eval_dist",
        type=str,
        default="gaussian",
        help="Evaluation instance distribution (default: first of --train_dists). Supported: uniform, gaussian",
    )
    parser.add_argument("--eval_seed", type=int, default=TrainConfig.eval_seed, help="Seed for the fixed eval set")
    parser.add_argument(
        "--eval_data_path",
        type=str,
        default="",
        help="Optional fixed eval dataset .pt (overrides --eval_dist/--eval_seed). Must be shaped [num,N,2] or dict with 'node_xy'.",
    )
    parser.add_argument("--gmm_no_cov", action="store_true", help="Gaussian mixture: use diagonal covariance")
    parser.add_argument("--gmm_var_lower", type=float, default=TrainConfig.gmm_var_lower)
    parser.add_argument("--gmm_var_upper", type=float, default=TrainConfig.gmm_var_upper)
    parser.add_argument("--gmm_num_modes_lower", type=int, default=TrainConfig.gmm_num_modes_lower)
    parser.add_argument("--gmm_num_modes_upper", type=int, default=TrainConfig.gmm_num_modes_upper)
    parser.add_argument("--gmm_center_lower", type=float, default=TrainConfig.gmm_center_lower)
    parser.add_argument("--gmm_center_upper", type=float, default=TrainConfig.gmm_center_upper)

    parser.add_argument("--add_two_opt", action="store_true")
    parser.add_argument("--two_opt_iters", type=int, default=10)
    parser.add_argument("--load", type=str, default="", help="Load saved gates .pt (optional)")
    parser.add_argument("--only_eval", action="store_true", help="Only run greedy evaluation and exit")
    parser.add_argument(
        "--baseline",
        type=str,
        choices=["batch_mean", "critic", "critic_batch_mean"],
        default=TrainConfig.baseline,
        help="Baseline mode: batch_mean (mean length), critic (V(s)), or critic_batch_mean (V(s) + mean length).",
    )
    parser.add_argument("--critic_coef", type=float, default=0.5)
    parser.add_argument("--model_dir", type=str, default="model", help="Auto-resolve solver ckpts from this dir (optional)")
    parser.add_argument("--pomo_ckpt", type=str, default="", help="Optional checkpoint path for POMO policy")
    parser.add_argument("--am_ckpt", type=str, default="", help="Optional checkpoint path for AM policy")
    parser.add_argument("--lehd_ckpt", type=str, default="", help="Optional checkpoint path for LEHD policy (init + rrc)")
    parser.add_argument("--elg_ckpt", type=str, default="", help="Optional checkpoint path for ELG policy (initializer)")
    parser.add_argument("--difusco_ckpt", type=str, default="", help="Optional checkpoint path for DIFUSCO model (initializer)")
    parser.add_argument("--dact_ckpt", type=str, default="", help="Optional checkpoint path for DACT policy (iterator)")
    parser.add_argument("--lih_ckpt", type=str, default="", help="Optional checkpoint path for LIH policy (iterator)")
    parser.add_argument(
        "--log_path",
        type=str,
        default="",
        help="Optional training log txt path. Default: my/outputs/two_gate_tsp_1000_seed{seed}_n{N}.log.txt",
    )

    args = parser.parse_args()

    ensure_local_easynco()

    train_dists = tuple(args.train_dists) if args.train_dists is not None else TrainConfig.train_dists
    eval_dist = args.eval_dist.strip() if args.eval_dist.strip() != "" else train_dists[0]

    cfg = TrainConfig(
        problem_size=args.problem_size,
        batch_size=args.batch_size,
        train_steps=args.train_steps,
        lr=args.lr,
        seed=args.seed,
        device=args.device,
        decoder_strategy=args.decoder_strategy,
        pomo_size=None if args.pomo_size == 0 else args.pomo_size,
        rrc_steps=args.rrc_steps,
        dact_steps=args.dact_steps,
        lih_steps=args.lih_steps,
        model_dir=None if args.model_dir.strip() == "" else args.model_dir.strip(),
        pomo_ckpt=None if args.pomo_ckpt.strip() == "" else args.pomo_ckpt.strip(),
        am_ckpt=None if args.am_ckpt.strip() == "" else args.am_ckpt.strip(),
        lehd_ckpt=None if args.lehd_ckpt.strip() == "" else args.lehd_ckpt.strip(),
        elg_ckpt=None if args.elg_ckpt.strip() == "" else args.elg_ckpt.strip(),
        difusco_ckpt=None if args.difusco_ckpt.strip() == "" else args.difusco_ckpt.strip(),
        dact_ckpt=None if args.dact_ckpt.strip() == "" else args.dact_ckpt.strip(),
        lih_ckpt=None if args.lih_ckpt.strip() == "" else args.lih_ckpt.strip(),
        log_every=args.log_every,
        eval_every=args.eval_every,
        eval_batches=args.eval_batches,
        train_dists=train_dists,
        eval_dist=eval_dist,
        eval_seed=args.eval_seed,
        eval_data_path=None if args.eval_data_path.strip() == "" else args.eval_data_path.strip(),
        gmm_no_cov=args.gmm_no_cov,
        gmm_var_lower=args.gmm_var_lower,
        gmm_var_upper=args.gmm_var_upper,
        gmm_num_modes_lower=args.gmm_num_modes_lower,
        gmm_num_modes_upper=args.gmm_num_modes_upper,
        gmm_center_lower=args.gmm_center_lower,
        gmm_center_upper=args.gmm_center_upper,
        add_two_opt=args.add_two_opt,
        two_opt_iters=args.two_opt_iters,
        load_path=None if args.load.strip() == "" else args.load.strip(),
        only_eval=args.only_eval,
        init_zoo=tuple(args.init_zoo) if args.init_zoo is not None else TrainConfig.init_zoo,
        iter_zoo=tuple(args.iter_zoo) if args.iter_zoo is not None else TrainConfig.iter_zoo,
        baseline=args.baseline,
        critic_coef=args.critic_coef,
    )

    out_dir = Path("my") / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    default_log_path = out_dir / f"two_gate_tsp_seed{cfg.seed}_n{cfg.problem_size}.log.txt"
    log_path = Path(args.log_path) if args.log_path.strip() != "" else default_log_path
    log, close_log = _make_print_logger(str(log_path))
    try:
        log("[config]", json.dumps(asdict(cfg), indent=2, ensure_ascii=False))
        log(f"[log_path] {log_path}")

        policy, optimizer = train(cfg, log=log)

        if cfg.only_eval:
            return

        # Save only gates (solvers are external / fixed)
        import torch

        ckpt_path = out_dir / f"two_gate_tsp_seed{cfg.seed}_n{cfg.problem_size}.pt"
        torch.save(
            {
                "cfg": asdict(cfg),
                "encoder": None if policy.encoder is None else policy.encoder.state_dict(),
                "gate1": policy.gate1.state_dict(),
                "gate2": policy.gate2.state_dict(),
                # Backward-compat: keep "value" as the Gate1 critic.
                "value": None if policy.value1 is None else policy.value1.state_dict(),
                "value1": None if policy.value1 is None else policy.value1.state_dict(),
                "value2": None if policy.value2 is None else policy.value2.state_dict(),
                "optimizer": optimizer.state_dict(),
            },
            ckpt_path,
        )
        log(f"[saved] {ckpt_path}")
    finally:
        close_log()


if __name__ == "__main__":
    try:
        import torch  # noqa: F401
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "This prototype requires PyTorch. Please install torch + tensordict + torchrl first."
        ) from e

    main()

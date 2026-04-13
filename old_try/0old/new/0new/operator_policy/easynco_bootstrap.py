from __future__ import annotations

import importlib
import os
import sys
import types
from pathlib import Path


def ensure_local_easynco() -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    easynco_dir = repo_root / "EasyNCO"
    if not easynco_dir.is_dir():
        raise FileNotFoundError(f"Expected EasyNCO source dir at {easynco_dir}")

    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)

    mod = sys.modules.get("EasyNCO")
    if mod is None:
        pkg = types.ModuleType("EasyNCO")
        pkg.__path__ = [str(easynco_dir)]
        pkg.__file__ = os.path.join(str(easynco_dir), "__init__.py")
        sys.modules["EasyNCO"] = pkg
    else:
        pkg_path = getattr(mod, "__path__", None)
        if pkg_path is None:
            mod.__path__ = [str(easynco_dir)]
        elif str(easynco_dir) not in list(pkg_path):
            pkg_path.append(str(easynco_dir))

    data_dir = easynco_dir / "data"
    if data_dir.is_dir():
        data_mod = sys.modules.get("EasyNCO.data")
        if data_mod is None:
            data_pkg = types.ModuleType("EasyNCO.data")
            data_pkg.__path__ = [str(data_dir)]
            data_pkg.__file__ = os.path.join(str(data_dir), "__init__.py")

            def _light_tsp_generator(data_size, problem_size, batch_size, device="cpu", path=None, **kwargs):
                import torch
                from torch.utils.data import DataLoader, Dataset

                if path is not None:
                    raise RuntimeError("Lightweight TSPGenerator stub does not support loading from path.")

                class _Rand(Dataset):
                    def __len__(self):
                        return int(data_size)

                    def __getitem__(self, _idx):
                        return torch.rand((int(problem_size), 2), device=device)

                return DataLoader(_Rand(), batch_size=int(batch_size), shuffle=False, num_workers=0)

            data_pkg.TSPGenerator = _light_tsp_generator
            sys.modules["EasyNCO.data"] = data_pkg
        else:
            data_path = getattr(data_mod, "__path__", None)
            if data_path is None:
                data_mod.__path__ = [str(data_dir)]
            elif str(data_dir) not in list(data_path):
                data_path.append(str(data_dir))

    methods_dir = easynco_dir / "neural_solvers" / "methods"
    if methods_dir.is_dir():
        methods_mod = sys.modules.get("EasyNCO.neural_solvers.methods")
        if methods_mod is None:
            methods_pkg = types.ModuleType("EasyNCO.neural_solvers.methods")
            methods_pkg.__path__ = [str(methods_dir)]
            methods_pkg.__file__ = os.path.join(str(methods_dir), "__init__.py")
            sys.modules["EasyNCO.neural_solvers.methods"] = methods_pkg
        else:
            m_path = getattr(methods_mod, "__path__", None)
            if m_path is None:
                methods_mod.__path__ = [str(methods_dir)]
            elif str(methods_dir) not in list(m_path):
                m_path.append(str(methods_dir))

    envs_dir = easynco_dir / "neural_solvers" / "envs"
    if envs_dir.is_dir():
        envs_mod = sys.modules.get("EasyNCO.neural_solvers.envs")
        if envs_mod is None:
            envs_pkg = types.ModuleType("EasyNCO.neural_solvers.envs")
            envs_pkg.__path__ = [str(envs_dir)]
            envs_pkg.__file__ = os.path.join(str(envs_dir), "__init__.py")
            sys.modules["EasyNCO.neural_solvers.envs"] = envs_pkg
        else:
            e_path = getattr(envs_mod, "__path__", None)
            if e_path is None:
                envs_mod.__path__ = [str(envs_dir)]
            elif str(envs_dir) not in list(e_path):
                e_path.append(str(envs_dir))

    backbones_dir = easynco_dir / "neural_solvers" / "backbones"
    if backbones_dir.is_dir():
        backbones_mod = sys.modules.get("EasyNCO.neural_solvers.backbones")
        if backbones_mod is None:
            backbones_pkg = types.ModuleType("EasyNCO.neural_solvers.backbones")
            backbones_pkg.__path__ = [str(backbones_dir)]
            backbones_pkg.__file__ = os.path.join(str(backbones_dir), "__init__.py")
            sys.modules["EasyNCO.neural_solvers.backbones"] = backbones_pkg
            _try_fill_backbones_exports(backbones_pkg)
        else:
            b_path = getattr(backbones_mod, "__path__", None)
            if b_path is None:
                backbones_mod.__path__ = [str(backbones_dir)]
            elif str(backbones_dir) not in list(b_path):
                b_path.append(str(backbones_dir))
            _try_fill_backbones_exports(backbones_mod)

    if methods_dir.is_dir():
        methods_mod = sys.modules.get("EasyNCO.neural_solvers.methods")
        if methods_mod is not None:
            _try_fill_methods_exports(methods_mod)

    neural_solvers_dir = easynco_dir / "neural_solvers"
    if neural_solvers_dir.is_dir():
        ns_mod = sys.modules.get("neural_solvers")
        if ns_mod is None:
            ns_pkg = types.ModuleType("neural_solvers")
            ns_pkg.__path__ = [str(neural_solvers_dir)]
            ns_pkg.__file__ = os.path.join(str(neural_solvers_dir), "__init__.py")
            sys.modules["neural_solvers"] = ns_pkg
        else:
            ns_path = getattr(ns_mod, "__path__", None)
            if ns_path is None:
                ns_mod.__path__ = [str(neural_solvers_dir)]
            elif str(neural_solvers_dir) not in list(ns_path):
                ns_path.append(str(neural_solvers_dir))

    _try_install_difusco_cython_merge_fallback(easynco_dir)
    return easynco_dir


def _try_install_difusco_cython_merge_fallback(easynco_dir: Path) -> None:
    modname = "EasyNCO.neural_solvers.methods.difusco.merge.cython_merge"
    try:
        importlib.import_module(modname)
        return
    except Exception:
        pass

    merge_dir = easynco_dir / "neural_solvers" / "methods" / "difusco" / "merge"
    if not merge_dir.is_dir():
        return

    pkg_name = "EasyNCO.neural_solvers.methods.difusco.merge"
    if pkg_name not in sys.modules:
        pkg = types.ModuleType(pkg_name)
        pkg.__path__ = [str(merge_dir)]
        pkg.__file__ = str(merge_dir / "__init__.py")
        sys.modules[pkg_name] = pkg

    m = types.ModuleType(modname)

    def merge_cython(coords, adj_mat):
        import numpy as np

        points = np.asarray(coords, dtype=np.float64)
        adj = np.asarray(adj_mat, dtype=np.float64)
        n = int(points.shape[0])

        dist = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=-1)
        np.fill_diagonal(dist, np.inf)

        sorted_edges = np.argsort((-adj / dist).reshape(-1)).astype(np.int64)

        A = np.zeros((n, n), dtype=np.float64)
        route_begin = np.arange(n, dtype=np.int64)
        route_end = np.arange(n, dtype=np.int64)

        def find_begin(i: int) -> int:
            root = i
            while route_begin[root] != root:
                root = int(route_begin[root])
            while route_begin[i] != i:
                parent = int(route_begin[i])
                route_begin[i] = root
                i = parent
            return int(root)

        def find_end(i: int) -> int:
            root = i
            while route_end[root] != root:
                root = int(route_end[root])
            while route_end[i] != i:
                parent = int(route_end[i])
                route_end[i] = root
                i = parent
            return int(root)

        merge_iterations = 0
        merge_count = 0
        for edge in sorted_edges:
            merge_iterations += 1
            i = int(edge // n)
            j = int(edge % n)

            begin_i = find_begin(i)
            end_i = find_end(i)
            begin_j = find_begin(j)
            end_j = find_end(j)

            if begin_i == begin_j:
                continue
            if i != begin_i and i != end_i:
                continue
            if j != begin_j and j != end_j:
                continue

            A[j, i] = 1.0
            A[i, j] = 1.0
            merge_count += 1

            if i == begin_i and j == end_j:
                route_begin[begin_i] = begin_j
                route_end[end_j] = end_i
            elif i == end_i and j == begin_j:
                route_begin[begin_j] = begin_i
                route_end[end_i] = end_j
            elif i == begin_i and j == begin_j:
                route_begin[begin_i] = end_j
                route_begin[begin_j] = end_j
                route_begin[end_j] = end_j
                route_end[end_j] = end_i
                route_end[begin_j] = end_i
            elif i == end_i and j == end_j:
                route_end[end_i] = begin_j
                route_begin[begin_j] = begin_i
                route_begin[end_j] = begin_i
                route_end[end_j] = begin_j
                route_end[begin_j] = begin_j

            if merge_count == n - 1:
                break

        final_begin = find_begin(0)
        final_end = find_end(0)
        A[final_end, final_begin] = 1.0
        A[final_begin, final_end] = 1.0
        return A, int(merge_iterations)

    m.merge_cython = merge_cython
    sys.modules[modname] = m


def _try_fill_methods_exports(methods_pkg: types.ModuleType) -> None:
    required = ("AttentionModelEncoder", "AttentionModelDecoder", "LEHDEncoder", "LEHDDecoder")
    if all(hasattr(methods_pkg, k) for k in required):
        return

    try:
        from EasyNCO.neural_solvers.methods.am.am_decoder import AttentionModelDecoder
        from EasyNCO.neural_solvers.methods.am.am_encoder import AttentionModelEncoder
        from EasyNCO.neural_solvers.methods.lehd.lehd_decoder import LEHDDecoder
        from EasyNCO.neural_solvers.methods.lehd.lehd_encoder import LEHDEncoder

        methods_pkg.AttentionModelEncoder = AttentionModelEncoder
        methods_pkg.AttentionModelDecoder = AttentionModelDecoder
        methods_pkg.LEHDEncoder = LEHDEncoder
        methods_pkg.LEHDDecoder = LEHDDecoder
    except Exception:
        return


def _try_fill_backbones_exports(backbones_pkg: types.ModuleType) -> None:
    required = (
        "reshape_by_heads",
        "multi_head_attention",
        "TransformerNet",
        "Compatibility",
        "positional_encoding_DIFUSCO",
    )
    if all(hasattr(backbones_pkg, k) for k in required):
        return

    try:
        from EasyNCO.neural_solvers.backbones.Transformer.attr_component import (
            FeedForward,
            Normalization,
            SkipConnection,
            multi_head_attention,
            positional_encoding_DIFUSCO,
            positional_encoding_ELG,
            positional_encoding_init,
            reshape_by_heads,
        )

        backbones_pkg.reshape_by_heads = reshape_by_heads
        backbones_pkg.multi_head_attention = multi_head_attention
        backbones_pkg.FeedForward = FeedForward
        backbones_pkg.SkipConnection = SkipConnection
        backbones_pkg.Normalization = Normalization
        backbones_pkg.positional_encoding_init = positional_encoding_init
        backbones_pkg.positional_encoding_ELG = positional_encoding_ELG
        backbones_pkg.positional_encoding_DIFUSCO = positional_encoding_DIFUSCO
    except Exception:
        return

    try:
        from EasyNCO.neural_solvers.backbones.Transformer.attention import (
            Compatibility,
            MultiHeadAttentionLayer,
            TransformerNet,
        )

        backbones_pkg.TransformerNet = TransformerNet
        backbones_pkg.Compatibility = Compatibility
        backbones_pkg.MultiHeadAttentionLayer = MultiHeadAttentionLayer
    except Exception:
        return

    try:
        from EasyNCO.neural_solvers.backbones.GNN.partition_net import glop_partition_net

        backbones_pkg.glop_partition_net = glop_partition_net
    except Exception:
        pass




_GPU_RUNTIME_PATCHED = False


def enable_easynco_runtime_patches(solver_device: str | None = None) -> list[str]:
    global _GPU_RUNTIME_PATCHED
    if _GPU_RUNTIME_PATCHED:
        return []
    if solver_device is None or not str(solver_device).startswith("cuda"):
        return []

    ensure_local_easynco()

    import torch
    from tensordict import TensorDict

    patched: list[str] = []

    import EasyNCO.neural_solvers.utils.special_selected as special_selected_mod

    def _tsp_special_selected(td: TensorDict, **kwargs):
        pomo_size = td.batch_size[1]
        batch_size = td.batch_size[0]
        first_mode = kwargs.get("first_mode", "random")
        first_placeholder = kwargs.get("first_placeholder", False)
        if first_placeholder:
            first_mode = "placeholder"
        device = td["first_node"].device
        if first_mode == "random" and (td["first_node"] == -1).all():
            selected = torch.arange(pomo_size, device=device)[None, :].expand(batch_size, pomo_size)
            probs = torch.ones(size=(batch_size, pomo_size), device=device)
        else:
            selected = None
            probs = None
        return selected, probs

    def _cvrp_special_selected(td: TensorDict, **kwargs):
        pomo_size = td.batch_size[1]
        batch_size = td.batch_size[0]
        device = td["next"]["selected_count"].device
        if (td["next"]["selected_count"] == 0).all():
            selected = torch.zeros(size=(batch_size, pomo_size), dtype=torch.long, device=device)
            probs = torch.ones(size=(batch_size, pomo_size), device=device)
            return selected, probs
        elif (td["next"]["selected_count"] == 1).all() and pomo_size > 1:
            selected = torch.arange(start=1, end=pomo_size + 1, device=device)[None, :].expand(batch_size, pomo_size)
            probs = torch.ones(size=(batch_size, pomo_size), device=device)
        else:
            selected = None
            probs = None
        return selected, probs

    special_selected_mod.TSPSpecifialSelected = _tsp_special_selected
    special_selected_mod.CVRPSpecifialSelected = _cvrp_special_selected
    patched.append("special_selected")

    from EasyNCO.neural_solvers.envs.TSPEnv import TSPEnv

    if not hasattr(TSPEnv, "_operator_policy_orig_reset"):
        TSPEnv._operator_policy_orig_reset = TSPEnv._reset
    if not hasattr(TSPEnv, "_operator_policy_orig_step"):
        TSPEnv._operator_policy_orig_step = TSPEnv._step

    def _reset_gpu_compatible(self, td: TensorDict = None, batch_size=None):
        out = TSPEnv._operator_policy_orig_reset(self, td, batch_size)
        device = self.problems.device if isinstance(self.problems, torch.Tensor) else torch.device(self.device)
        tensor_attrs = [
            "current_node",
            "selected_node_list",
            "ninf_mask",
            "dummy_flag_bool",
            "dummy_flag_long",
            "selected_count",
            "first_node",
        ]
        for attr in tensor_attrs:
            value = getattr(self, attr, None)
            if torch.is_tensor(value):
                setattr(self, attr, value.to(device))
        return out.to(device)

    def _step_gpu_compatible(self, td: TensorDict):
        target_device = self.selected_node_list.device if torch.is_tensor(self.selected_node_list) else (self.problems.device if isinstance(self.problems, torch.Tensor) else torch.device(self.device))
        td_local = td.clone(recurse=False)
        if "action" in td_local.keys():
            td_local.set("action", td_local["action"].to(target_device))
        return TSPEnv._operator_policy_orig_step(self, td_local)

    TSPEnv._reset = _reset_gpu_compatible
    TSPEnv._step = _step_gpu_compatible
    patched.append("TSPEnv")

    from EasyNCO.neural_solvers.pipeline.initialization import ARInitialization

    if not hasattr(ARInitialization, "_operator_policy_orig_play_episode"):
        ARInitialization._operator_policy_orig_play_episode = ARInitialization.play_episode

    def _ar_play_episode_gpu_compatible(self, env, decoder_strategy: str = "sampling"):
        reset_td = env.reset()
        self.policy.set_decoder_strategy(decoder_strategy)
        self.policy.pre_forward(reset_td)
        device = env.problems.device if hasattr(env, "problems") and isinstance(env.problems, torch.Tensor) else None
        likelihood = torch.zeros(size=(env.batch_size[0], env.pomo_size, 0), device=device)
        done = False
        reward = None
        state_td = env.pre_step()
        while not done:
            next_td = self.policy(state_td)
            prob = next_td["prob"].to(likelihood.device)
            state_td = env.step(next_td)
            likelihood = torch.cat((likelihood, prob[:, :, None]), dim=2)
            reward = state_td["reward"]
            done = state_td["done"].all()
        policy_out = {"reward": reward, "likelihood": likelihood}
        return state_td, policy_out

    ARInitialization.play_episode = _ar_play_episode_gpu_compatible
    patched.append("ARInitialization")

    from EasyNCO.neural_solvers.methods.elg.initialization import ELGInitialization
    from EasyNCO.neural_solvers.methods.pomo.initialization import POMOInitialization

    if not hasattr(ELGInitialization, "_operator_policy_orig_play_episode"):
        ELGInitialization._operator_policy_orig_play_episode = ELGInitialization.play_episode

    def _elg_play_episode_gpu_compatible(self, env, decoder_strategy: str = "sampling"):
        reset_td = env.reset()
        self.policy.set_decoder_strategy(decoder_strategy)
        self.policy.pre_forward(reset_td)
        device = env.problems.device if hasattr(env, "problems") and isinstance(env.problems, torch.Tensor) else None
        likelihood = torch.zeros(size=(env.batch_size[0], env.pomo_size, 0), device=device)
        done = False
        reward = None
        state_td = env.pre_step()
        step = 0
        while not done:
            if step > 1:
                self.policy.enable_local_policy()
            next_td = self.policy(state_td)
            prob = next_td["prob"].to(likelihood.device)
            state_td = env.step(next_td)
            likelihood = torch.cat((likelihood, prob[:, :, None]), dim=2)
            reward = state_td["reward"]
            done = state_td["done"].all()
            step += 1
        policy_out = {"reward": reward, "likelihood": likelihood}
        return state_td, policy_out

    ELGInitialization.play_episode = _elg_play_episode_gpu_compatible
    patched.append("ELGInitialization")

    if not hasattr(POMOInitialization, "_operator_policy_orig_play_episode"):
        POMOInitialization._operator_policy_orig_play_episode = POMOInitialization.play_episode

    def _pomo_play_episode_gpu_compatible(self, env, decoder_strategy: str = "sampling"):
        reset_td = env.reset()
        self.policy.set_decoder_strategy(decoder_strategy)
        self.policy.pre_forward(reset_td)
        device = env.problems.device if hasattr(env, "problems") and isinstance(env.problems, torch.Tensor) else None
        likelihood = torch.zeros(size=(env.batch_size[0], env.pomo_size, 0), device=device)
        done = False
        reward = None
        state_td = env.pre_step()
        if env.env_name == "kp":
            while not done:
                temp_td = self.policy(state_td)
                selected = temp_td.get("action", None)
                prob = temp_td.get("prob", None)
                action_w_finished = selected.clone()
                action_w_finished[temp_td.get("done", False)] = env.problem_size
                temp_td["action"] = action_w_finished
                next_td = env.step(temp_td)
                if prob is not None:
                    chosen_prob = prob.to(likelihood.device).clone()
                    chosen_prob[next_td["done"].to(likelihood.device)] = 1
                    likelihood = torch.cat((likelihood, chosen_prob[:, :, None]), dim=2)
                reward = next_td["reward"]
                done = next_td["done"].all()
        else:
            while not done:
                next_td = self.policy(state_td)
                prob = next_td["prob"].to(likelihood.device)
                state_td = env.step(next_td)
                likelihood = torch.cat((likelihood, prob[:, :, None]), dim=2)
                reward = state_td["reward"]
                done = state_td["done"].all()
        policy_out = {"reward": reward, "likelihood": likelihood}
        return state_td, policy_out

    POMOInitialization.play_episode = _pomo_play_episode_gpu_compatible
    patched.append("POMOInitialization")

    from EasyNCO.neural_solvers.methods.lehd.policy import LEHDPolicy

    if not hasattr(LEHDPolicy, "_operator_policy_orig_forward"):
        LEHDPolicy._operator_policy_orig_forward = LEHDPolicy.forward

    def _lehd_forward_gpu_compatible(self, td: TensorDict):
        if self.phase == "train":
            _, probs = self.decoder(self.encoder(td["locs"]), td, self.phase)
            selected, prob = self.select_next_node(probs, td)
        elif self.phase == "test":
            if "partial_length" not in td.keys():
                if "next" in td.keys() and "selected_count" in td["next"].keys():
                    td_device = td["next"]["selected_count"].device
                elif "first_node" in td.keys():
                    td_device = td["first_node"].device
                else:
                    td_device = self.embed_data.device
                td["partial_length"] = torch.full(
                    (td.batch_size[0], td.batch_size[1]),
                    fill_value=self.problem_size,
                    dtype=torch.int,
                    device=td_device,
                )
            selected, probs = self.decoder(self.embed_data, td, self.phase, self.first_mode)
            if selected is not None:
                if self.problem_type == "cvrp":
                    selected = torch.cat((
                        selected[:, :, None],
                        torch.ones((td.batch_size[0], td.batch_size[1], 1), dtype=selected.dtype, device=selected.device),
                    ), dim=-1)
                prob = probs
            else:
                selected, prob = self.select_next_node(probs, td)
        else:
            raise RuntimeError(f"Invalid learning phase: {self.phase}.")
        td.set("action", selected)
        td.set("prob", prob)
        return td

    LEHDPolicy.forward = _lehd_forward_gpu_compatible
    patched.append("LEHDPolicy.forward")

    from EasyNCO.neural_solvers.methods.lehd.lehd_decoder import LEHDDecoder

    def _lehd_get_available_embed_node(self, data, td):
        selected_list = td["next"]["selected_node_list"]
        batch_size = td.batch_size[0]
        pomo_size = td.batch_size[1]
        cur_problem_size = td["partial_length"][0, 0]
        selected_count = td["next"]["selected_count"][0, 0]
        if self.problem_type == "tsp":
            cur_selected_node_list = selected_list
        elif self.problem_type == "cvrp":
            cur_selected_node_list = selected_list[:, :, :, 0] - 1
        else:
            raise RuntimeError(f"Invalid problem type: {self.problem_type}.")
        device = data.device
        new_selected_node_list = torch.arange(cur_problem_size, device=device)[None, None, :].expand(batch_size, pomo_size, -1).clone()
        new_data_len = cur_problem_size - selected_count
        new_selected_node_list.scatter_(dim=-1, index=cur_selected_node_list, value=-2)
        unselect_node_list = new_selected_node_list[torch.gt(new_selected_node_list, -1)].view(batch_size, pomo_size, new_data_len)
        new_data = importlib.import_module("EasyNCO.utils.utils")._get_encoding(data, unselect_node_list)
        if new_data.shape[2] == 0:
            new_data = data
        return new_data

    def _lehd_calculate_probs(self, out, td):
        selected_list = td["next"]["selected_node_list"]
        batch_size = td.batch_size[0]
        pomo_size = td.batch_size[1]
        cur_problem_size = td["partial_length"][0, 0]
        selected_count = td["next"]["selected_count"][0, 0]
        if self.problem_type == "tsp":
            cur_selected_node_list = selected_list
            out[:, :, [0, -1]] = out[:, :, [0, -1]] + float("-inf")
            probs = torch.softmax(out, dim=-1)
            probs_available = probs[:, :, 1:-1].clone()
            new_probs = torch.zeros(batch_size, pomo_size, cur_problem_size, device=probs_available.device, dtype=probs_available.dtype)
            index_small = torch.le(probs_available, 1e-5)
            probs_available[index_small] = probs_available[index_small] + torch.tensor(1e-7, dtype=probs_available[index_small].dtype, device=probs_available.device)
            cur_selected_node_index = cur_selected_node_list
        elif self.problem_type == "cvrp":
            cur_selected_node_list = selected_list[:, :, :, 0] - 1
            out[:, :, [0, -1], :] = out[:, :, [0, -1], :] + float("-inf")
            out = torch.cat((out[:, :, :, 0], out[:, :, :, 1]), dim=-1)
            available_node_num = cur_problem_size - selected_count
            probs = torch.softmax(out, dim=-1)
            probs_available = torch.cat((probs[:, :, 1:available_node_num + 1], probs[:, :, -1 - available_node_num:-1]), dim=-1).clone()
            new_probs = torch.zeros(batch_size, pomo_size, 2 * cur_problem_size, device=probs_available.device, dtype=probs_available.dtype)
            index_small = torch.le(probs_available, 1e-5)
            probs_available[index_small] = probs_available[index_small] + torch.tensor(1e-7, dtype=probs_available[index_small].dtype, device=probs_available.device)
            cur_selected_node_index = torch.cat((cur_selected_node_list, cur_problem_size + cur_selected_node_list), dim=-1)
        else:
            raise RuntimeError(f"Invalid problem type: {self.problem_type}.")
        new_probs.scatter_(dim=-1, index=cur_selected_node_index, value=-2)
        index = torch.gt(new_probs, -1).view(batch_size, pomo_size, -1)
        new_probs[index] = probs_available.ravel()
        return new_probs

    LEHDDecoder.get_available_embed_node = _lehd_get_available_embed_node
    LEHDDecoder.calculate_probs = _lehd_calculate_probs
    patched.append("LEHDDecoder")

    _GPU_RUNTIME_PATCHED = True
    return patched

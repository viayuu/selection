from __future__ import annotations

import os
import sys
import types
from pathlib import Path


def ensure_local_easynco() -> Path:
    """
    Make `import EasyNCO.*` resolve to this repo's `final/` directory.

    Background:
    - Code under `final/` uses absolute imports like `EasyNCO.utils.utils`.
    - In this repo, the directory is named `final/`, not `EasyNCO/`.
    - We avoid modifying `final/` by creating a runtime alias package.
    """
    repo_root = Path(__file__).resolve().parents[1]
    final_dir = repo_root / "final"
    if not final_dir.is_dir():
        raise FileNotFoundError(f"Expected EasyNCO source dir at {final_dir}")

    # Ensure repo root is importable (so `my.*` and local files can be found).
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)

    # If user already installed an `EasyNCO` package, we prefer the local one for this repo.
    mod = sys.modules.get("EasyNCO")
    if mod is None:
        pkg = types.ModuleType("EasyNCO")
        pkg.__path__ = [str(final_dir)]  # allow `EasyNCO.data`, `EasyNCO.neural_solvers`, ...
        pkg.__file__ = os.path.join(str(final_dir), "__init__.py")
        sys.modules["EasyNCO"] = pkg
    else:
        # Make sure local `final/` is in the package search path.
        pkg_path = getattr(mod, "__path__", None)
        if pkg_path is None:
            mod.__path__ = [str(final_dir)]
        elif str(final_dir) not in list(pkg_path):
            pkg_path.append(str(final_dir))

    # ---------------------------------------------------------------------
    # Lightweight stubs to avoid importing heavy `__init__.py` modules
    # ---------------------------------------------------------------------

    # 1) `EasyNCO.data` in this repo imports many optional deps (torch_geometric, sklearn, ...).
    #    For our two-gate prototype we only need `EasyNCO.data.data_utils` (used by envs) and a
    #    placeholder `TSPGenerator` so that `TSPEnv` can be imported without the full data stack.
    data_dir = final_dir / "data"
    if data_dir.is_dir():
        data_mod = sys.modules.get("EasyNCO.data")
        if data_mod is None:
            data_pkg = types.ModuleType("EasyNCO.data")
            data_pkg.__path__ = [str(data_dir)]
            data_pkg.__file__ = os.path.join(str(data_dir), "__init__.py")

            def _light_tsp_generator(data_size, problem_size, batch_size, device="cpu", path=None, **kwargs):
                import torch
                from torch.utils.data import Dataset, DataLoader

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
            if not hasattr(data_mod, "TSPGenerator"):
                # Backfill stub for existing module.
                def _light_tsp_generator(data_size, problem_size, batch_size, device="cpu", path=None, **kwargs):
                    import torch
                    from torch.utils.data import Dataset, DataLoader

                    if path is not None:
                        raise RuntimeError("Lightweight TSPGenerator stub does not support loading from path.")

                    class _Rand(Dataset):
                        def __len__(self):
                            return int(data_size)

                        def __getitem__(self, _idx):
                            return torch.rand((int(problem_size), 2), device=device)

                    return DataLoader(_Rand(), batch_size=int(batch_size), shuffle=False, num_workers=0)

                data_mod.TSPGenerator = _light_tsp_generator

    # 2) `EasyNCO.neural_solvers.methods.__init__` imports a huge zoo (and optional deps like scipy).
    #    Many policies import encoders/decoders via `from EasyNCO.neural_solvers.methods import ...`.
    #    We provide a lightweight package module so importing submodules does NOT execute that huge __init__.
    methods_dir = final_dir / "neural_solvers" / "methods"
    if methods_dir.is_dir():
        methods_mod = sys.modules.get("EasyNCO.neural_solvers.methods")
        if methods_mod is None:
            methods_pkg = types.ModuleType("EasyNCO.neural_solvers.methods")
            methods_pkg.__path__ = [str(methods_dir)]
            methods_pkg.__file__ = os.path.join(str(methods_dir), "__init__.py")
            # Register first so importing submodules does not execute the original heavy __init__.py.
            sys.modules["EasyNCO.neural_solvers.methods"] = methods_pkg

        else:
            m_path = getattr(methods_mod, "__path__", None)
            if m_path is None:
                methods_mod.__path__ = [str(methods_dir)]
            elif str(methods_dir) not in list(m_path):
                m_path.append(str(methods_dir))

    # 3) `EasyNCO.neural_solvers.envs.__init__` imports all envs (some may pull optional deps).
    #    Our prototype only needs `TSPEnv`, so we stub the package to avoid importing everything.
    envs_dir = final_dir / "neural_solvers" / "envs"
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

    # 4) `EasyNCO.neural_solvers.backbones.__init__` imports GNN components that require torch_geometric.
    #    AM/POMO/LEHD only need Transformer pieces, so we stub the package and export minimal symbols.
    backbones_dir = final_dir / "neural_solvers" / "backbones"
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

    # Now that the Transformer backbone symbols are available, export minimal method symbols
    # (AttentionModelEncoder/Decoder, LEHDEncoder/Decoder) into `EasyNCO.neural_solvers.methods`.
    # This is required because many policy modules do `from EasyNCO.neural_solvers.methods import ...`.
    if methods_dir.is_dir():
        methods_mod = sys.modules.get("EasyNCO.neural_solvers.methods")
        if methods_mod is not None:
            _try_fill_methods_exports(methods_mod)

    # Some methods under `final/` use legacy imports like `from neural_solvers.pipeline import ...`.
    # Provide a compatible top-level alias without modifying `final/`.
    neural_solvers_dir = final_dir / "neural_solvers"
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

    # 5) DIFUSCO uses a Cython extension `cython_merge` to merge a soft adjacency matrix into a tour.
    #    In many environments Cython isn't installed (or building extensions is undesired). Provide a
    #    small NumPy fallback module so `merge_tour()` can still run (slower but fine for small N).
    _try_install_difusco_cython_merge_fallback(final_dir)

    return final_dir


def _try_install_difusco_cython_merge_fallback(final_dir: Path) -> None:
    import importlib

    modname = "EasyNCO.neural_solvers.methods.difusco.merge.cython_merge"
    try:
        importlib.import_module(modname)
        return
    except Exception:
        pass

    merge_dir = final_dir / "neural_solvers" / "methods" / "difusco" / "merge"
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
    """
    Fill minimal symbols used by a subset of policies:
    - AM / POMO: AttentionModelEncoder/Decoder
    - LEHD: LEHDEncoder/Decoder
    """
    required = ("AttentionModelEncoder", "AttentionModelDecoder", "LEHDEncoder", "LEHDDecoder")
    if all(hasattr(methods_pkg, k) for k in required):
        return

    try:
        from EasyNCO.neural_solvers.methods.am.am_encoder import AttentionModelEncoder  # type: ignore
        from EasyNCO.neural_solvers.methods.am.am_decoder import AttentionModelDecoder  # type: ignore
        from EasyNCO.neural_solvers.methods.lehd.lehd_encoder import LEHDEncoder  # type: ignore
        from EasyNCO.neural_solvers.methods.lehd.lehd_decoder import LEHDDecoder  # type: ignore

        methods_pkg.AttentionModelEncoder = AttentionModelEncoder
        methods_pkg.AttentionModelDecoder = AttentionModelDecoder
        methods_pkg.LEHDEncoder = LEHDEncoder
        methods_pkg.LEHDDecoder = LEHDDecoder
    except Exception:
        # If torch/other deps aren't installed yet, leave it empty until next run.
        return


def _try_fill_backbones_exports(backbones_pkg: types.ModuleType) -> None:
    """
    Export minimal Transformer backbone symbols used by AM/POMO/LEHD without importing torch_geometric.
    """
    required = ("reshape_by_heads", "multi_head_attention", "TransformerNet", "Compatibility", "positional_encoding_DIFUSCO")
    if all(hasattr(backbones_pkg, k) for k in required):
        return

    try:
        # 1) low-level components
        from EasyNCO.neural_solvers.backbones.Transformer.attr_component import (  # type: ignore
            FeedForward,
            Normalization,
            SkipConnection,
            multi_head_attention,
            reshape_by_heads,
        )

        backbones_pkg.reshape_by_heads = reshape_by_heads
        backbones_pkg.multi_head_attention = multi_head_attention
        # NOTE: Upstream `positional_encoding_DIFUSCO` does `torch.tensor(x, ...)` even when `x` is already a Tensor,
        # which triggers a noisy warning. We provide an equivalent implementation without that warning.
        def positional_encoding_DIFUSCO_no_warning(x, embed_dim, max_timescale, min_timescale):
            import numpy as np
            import torch
            import torch.nn.functional as F

            if isinstance(x, torch.Tensor):
                device = x.device
                position = x.to(dtype=torch.float32)
            else:
                device = torch.device("cpu")
                position = torch.as_tensor(x, dtype=torch.float32, device=device)

            num_timescale = embed_dim // 2
            log_timescale_increment = np.log(float(max_timescale) / float(min_timescale)) / num_timescale
            inv_timescales = min_timescale * torch.exp(
                torch.arange(num_timescale, dtype=torch.float32, device=device) * -log_timescale_increment
            )

            scaled_time = position.unsqueeze(1) * inv_timescales.unsqueeze(0)
            signal = torch.cat([torch.cos(scaled_time), torch.sin(scaled_time)], dim=1)
            signal = F.pad(signal, (0, 0, 0, embed_dim % 2))
            return signal

        backbones_pkg.positional_encoding_DIFUSCO = positional_encoding_DIFUSCO_no_warning
        backbones_pkg.FeedForward = FeedForward
        backbones_pkg.SkipConnection = SkipConnection
        backbones_pkg.Normalization = Normalization

        # 2) attention / transformer net
        from EasyNCO.neural_solvers.backbones.Transformer.attention import (  # type: ignore
            Compatibility,
            MultiHeadAttentionLayer,
            TransformerNet,
        )

        backbones_pkg.TransformerNet = TransformerNet
        backbones_pkg.Compatibility = Compatibility
        backbones_pkg.MultiHeadAttentionLayer = MultiHeadAttentionLayer
    except Exception:
        return

    # NOTE: method-level exports are handled by `_try_fill_methods_exports` after this function runs.

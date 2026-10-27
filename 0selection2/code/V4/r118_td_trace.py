"""Observe one TRAIN query on a production TSP/LEHD or ATSP/MATNET backend.

``capture_td(backend, [native_item])`` returns the original InferenceResult,
NumPy arrays, and JSON-safe info. The caller owns TRAIN membership and RNG.
No labels/costs are accepted, no policy is replaced, and run() is called once.

Step arrays have axes [step, POMO, native_node] (or [step, POMO]). Only
augmentation row 0 is serialized; ALL production augmentations still execute.
decoder_probs is the unmodified node-indexed output, including LEHD's -2
visited sentinel. Forced actions have zero probabilities and decoder_valid=False.
pre/post_visited comes from native tour history; pre/post_feasible means a finite
native ninf_mask entry, NOT independent feasibility certification. Initial
MatNet encoder role inputs, if exposed, have axes [native_node, embedding].
"""

import numpy as np
import torch


def capture_td(backend, items, *, max_bytes=384 * 2**20):
    """Capture a complete native episode, failing closed on shape/alignment/cap."""
    kind = (backend.problem, backend.method)
    if kind not in {("TSP", "LEHD"), ("ATSP", "MATNET")}:
        raise ValueError("capture_td supports only TSP/LEHD and ATSP/MATNET")
    if len(items) != 1 or not isinstance(items[0], torch.Tensor):
        raise ValueError("capture_td requires one native tensor query")
    item = items[0]
    if item.ndim != 3 or item.shape[0] != 1 or item.shape[1] < 1:
        raise ValueError("The native query must have a singleton batch axis")
    if not isinstance(max_bytes, int) or isinstance(max_bytes, bool) or max_bytes <= 0:
        raise ValueError("max_bytes must be a positive integer")
    n, tsp = item.shape[1], kind[0] == "TSP"
    if item.shape[2] != (2 if tsp else n):
        raise ValueError("Expected native [1,N,2] coordinates or [1,N,N] matrix")
    env, model = backend.env, backend.model
    arrays, handles = {}, []
    steps, pending, used, scratch = 0, None, 0, 0
    batch = pomo = None

    def allocate(specs):
        nonlocal used, scratch
        sizes = [int(np.prod(shape)) * np.dtype(dtype).itemsize
                 for shape, dtype, _ in specs.values()]
        # Reserve a row-copy/indexing workspace too, without a final stack copy.
        copy_sizes = [int(np.prod(shape[-2:])) * np.dtype(dtype).itemsize
                      for shape, dtype, _ in specs.values()]
        peak = max([scratch, 16 * pomo * n, *copy_sizes])
        if used + sum(sizes) + peak > max_bytes:
            raise MemoryError("Native trace exceeds max_bytes before allocation")
        arrays.update({key: np.full(shape, fill, dtype=dtype)
                       for key, (shape, dtype, fill) in specs.items()})
        used += sum(sizes)
        scratch = peak

    def ensure():
        nonlocal batch, pomo
        if batch is not None:
            return
        batch, pomo, nodes = env.ninf_mask.shape
        if nodes != n or batch != env.aug_factor or pomo != env.pomo_size:
            raise ValueError("Unexpected native augmentation/POMO/node axes")
        specs = {"action": ((n, pomo), np.int64, -1),
                 "decoder_probs": ((n, pomo, n), np.float32, 0),
                 "decoder_valid": ((n, pomo), bool, False),
                 "decoder_forced": ((n, pomo), bool, True),
                 "decoder_called": ((n,), bool, False),
                 "decoder_env_count": ((n,), np.int64, 0),
                 "decoder_selected": ((n, pomo), np.int64, -1),
                 "first_node": ((pomo,), np.int64, -1),
                 "final_rewards": ((pomo,), np.float32, 0),
                 "final_costs": ((pomo,), np.float32, 0),
                 "final_tours": ((pomo, n), np.int64, -1)}
        for prefix in ("pre", "post"):
            for field, shape, dtype, fill in (
                ("current", (n, pomo), np.int64, -1),
                ("selected_count", (n, pomo), np.int64, 0),
                ("visited", (n, pomo, n), bool, False),
                ("feasible", (n, pomo, n), bool, False),
                ("finished", (n, pomo), bool, False),
            ):
                specs[prefix + "_" + field] = (shape, dtype, fill)
        allocate(specs)

    def aligned():
        ensure()
        if steps >= n or not bool((env.selected_count == steps).all().item()):
            raise ValueError("Native decoder/env.step timestep mismatch")

    def copy_row(destination, value):
        if tuple(value.shape) != (batch, *destination.shape):
            raise ValueError("Native tensor axes changed during capture")
        dtype = torch.from_numpy(destination).dtype
        np.copyto(destination, value[0].detach().to(device="cpu", dtype=dtype).numpy())

    def snapshot(prefix, td):
        native = td["next"] if tsp else td
        current = getattr(env, "current_node", None)
        if current is not None:
            copy_row(arrays[prefix + "_current"][steps], current)
        copy_row(arrays[prefix + "_selected_count"][steps], env.selected_count)
        copy_row(arrays[prefix + "_finished"][steps], td["done"])
        mask = native["ninf_mask"]
        if tuple(mask.shape) != (batch, pomo, n):
            raise ValueError("Native mask axes changed during capture")
        arrays[prefix + "_feasible"][steps] = torch.isfinite(mask[0]).cpu().numpy()
        history = env.selected_node_list[0].detach().cpu().numpy()
        arrays[prefix + "_visited"][steps, np.arange(pomo)[:, None], history] = True

    def decoder_hook(_module, _args, output):
        nonlocal pending
        aligned()
        if pending is not None:
            raise ValueError("Multiple decoder calls before one native env.step")
        selected = None
        if tsp:
            selected, probs = output
        else:
            probs = output
            while isinstance(probs, (tuple, list)) and len(probs) == 1:
                probs = probs[0]
        if not isinstance(probs, torch.Tensor):
            raise ValueError("Decoder did not expose native probabilities")
        if tsp and selected is not None:
            if tuple(probs.shape) != (batch, pomo):
                raise ValueError("Unexpected forced LEHD probability placeholder")
            copy_row(arrays["decoder_selected"][steps], selected)
        else:
            # Only singleton wrapper axes may differ; never flatten native axes.
            if not tsp:
                native_axes = tuple(d for d in (batch, pomo, n) if d != 1)
                if tuple(d for d in probs.shape if d != 1) != native_axes:
                    raise ValueError("Unexpected MatNet probability wrapper axes")
                probs = probs.squeeze().view(batch, pomo, n)
            copy_row(arrays["decoder_probs"][steps], probs)
            arrays["decoder_valid"][steps] = True
            arrays["decoder_forced"][steps] = False
        arrays["decoder_called"][steps] = True
        arrays["decoder_env_count"][steps] = batch
        pending = steps

    def role_hook(_module, args):
        ensure()
        if steps or "initial_col_roles" in arrays:
            raise ValueError("Multiple MatNet initial role draws in one query")
        row, col = args[:2]
        if row.ndim != 3 or col.shape != row.shape or row.shape[:2] != (batch, n):
            raise ValueError("Unexpected MatNet encoder role inputs")
        allocate({key: ((n, row.shape[-1]), np.float32, 0)
                  for key in ("initial_row_roles", "initial_col_roles")})
        copy_row(arrays["initial_row_roles"], row)
        copy_row(arrays["initial_col_roles"], col)

    original_step = env.step
    had_step = "step" in vars(env)
    own_step = vars(env).get("step")

    def step(td):
        nonlocal steps, pending
        aligned()
        if pending != steps and (tsp or steps != 0 or pending is not None):
            raise ValueError("Missing or misaligned decoder output for native env.step")
        copy_row(arrays["action"][steps], td["action"])
        if tsp and arrays["decoder_forced"][steps].all():
            if not np.array_equal(arrays["decoder_selected"][steps], arrays["action"][steps]):
                raise ValueError("Forced decoder action differs from native env.step")
        snapshot("pre", td)
        output = original_step(td)
        snapshot("post", output)
        if not bool((env.selected_count == steps + 1).all().item()):
            raise ValueError("Native env.step did not advance exactly once")
        if steps == 0:
            arrays["first_node"][:] = arrays["action"][0]
        if steps == n - 1:
            copy_row(arrays["final_rewards"], output["reward"])
            copy_row(arrays["final_tours"], env.selected_node_list)
            np.negative(arrays["final_rewards"], out=arrays["final_costs"])
        steps += 1
        pending = None
        return output

    try:
        env.step = step
        handles.append(model.decoder.register_forward_hook(decoder_hook))
        encoder = getattr(model, "encoder", None)
        if not tsp and hasattr(encoder, "register_forward_pre_hook"):
            handles.append(encoder.register_forward_pre_hook(role_hook))
        result = backend.run(items)
        if steps != n or pending is not None or not arrays["post_finished"][-1].all():
            raise ValueError("Native episode ended without a complete aligned trace")
    finally:
        for handle in handles:
            handle.remove()
        if had_step:
            env.step = own_step
        else:
            del env.step

    arange_start = np.array_equal(arrays["first_node"], np.arange(pomo))
    roles_recorded = "initial_col_roles" in arrays
    info = dict(
        schema="r118_td_trace_v1", problem=backend.problem, method=backend.method,
        steps=steps, nodes=n, pomo_size=pomo, executed_augmentations=batch,
        configured_augmentations=int(env.aug_factor), serialized_augmentation=0,
        serialized_augmentations=1, all_pomo_columns_retained=True,
        node_id_space="unchanged native input positions; decoder last axis is native node ID",
        result_selection="maximum POMO reward on augmentation0, unchanged backend.run",
        result_best_pomo=int(arrays["final_rewards"].argmax()),
        decoder_calls=int(arrays["decoder_called"].sum()),
        decoder_probability_semantics="raw native node-indexed float32; LEHD visited=-2; forced=0/invalid",
        forced_probability_semantics="not a node distribution; never synthesize one-hot probabilities",
        first_mode=getattr(model, "first_mode", None),
        first_node_rule=("native arange(P); P=1 is deterministic physical node0, not a random draw"
                         if arange_start and arrays["decoder_forced"][0].all()
                         else "observed native first action; no randomness inferred from configuration"),
        initial_roles=("recorded native encoder inputs for augmentation0; stochastic roles unchanged, not fixed identity"
                       if roles_recorded else "not applicable" if tsp
                       else "unrecorded: encoder role inputs not exposed; no roles invented"),
        native_masks_are_independent_certificates=False,
        memory_bytes=used, memory_reserved_peak_bytes=used + scratch,
        max_bytes=max_bytes, memory_scope="trace NumPy buffers and copy workspace, excludes native backend",
        train_membership="caller must supply frozen TRAIN only",
    )
    return result, arrays, info

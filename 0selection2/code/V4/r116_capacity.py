"""One bounded R116 shape-corner capacity check; no persistent model training."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from code.unified_selector.registry import PROBLEMS
from .pairwise_objective import inputs_only
from .r76_experiment import open_cache, read_json, write_json
from .r84_prepare import DEFAULT_DATA_ROOT, DEFAULT_PREPARED_ROOT
from .r116_model import initialize_shared
from .r116_subset import full_ranking_nll, stable_cost_order


CAPS = {p: 256 if p in ("TSP", "CVRP", "ATSP") else 512 for p in PROBLEMS}
CACHE = Path("/tmp/shiys_R84_new90k_cache")


def capacity(output):
    torch.set_num_threads(2)
    if not torch.cuda.is_available() or "RTX 3090" not in torch.cuda.get_device_name(0):
        raise RuntimeError("Capacity must execute on a launcher-authorized RTX3090")
    device = torch.device("cuda:0")
    initial = torch.load(DEFAULT_PREPARED_ROOT / "identicalinit_seed2.pt", map_location="cpu", weights_only=False)
    geometry = read_json(DEFAULT_PREPARED_ROOT / "geometry_stats.json")
    model, params, identity = initialize_shared(initial, geometry, "B_subset", device)
    other, _, paired = initialize_shared(initial, geometry, "A_static", "cpu")
    if paired["trainable_sha256"] != identity["trainable_sha256"]:
        raise ValueError("Fresh A/B complete initial parameters differ")
    del other, initial
    manifest = read_json(DEFAULT_PREPARED_ROOT / "source_cache_manifest.json")
    other_problem = max((p for p in PROBLEMS if p not in ("TSP", "CVRP", "ATSP")),
                        key=lambda p: manifest["splits"]["train"][p]["true_length_max"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda")
    rows = []
    started = time.monotonic()
    model.train()
    for problem in ("CVRP", "TSP", "ATSP", other_problem):
        cache = open_cache(CACHE, "train", problem, manifest, DEFAULT_DATA_ROOT)
        try:
            indices = np.argsort(cache.lengths, kind="stable")[-CAPS[problem]:].copy()
            batch = cache.batch(indices, device=device)
            fp64 = torch.as_tensor(np.array(cache.raw["costs"][indices], copy=True), device=device)
            order = stable_cost_order(fp64, batch["ind"], batch["pool_ids"])
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
            tick = time.monotonic()
            with torch.autocast("cuda", dtype=torch.float16):
                result = model(inputs_only(batch))
                champion = F.nll_loss(result["logits"], batch["ind"])
                ranking = full_ranking_nll(result["subset_logits"], order).mean()
                loss = .5 * (champion + ranking)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            if not torch.isfinite(loss) or not torch.isfinite(gradient):
                raise FloatingPointError("Nonfinite complete-subset loss/gradient in capacity check")
            scaler.step(optimizer)
            scaler.update()
            torch.cuda.synchronize(device)
            rows.append(dict(problem=problem, microbatch=CAPS[problem],
                n_max=int(batch["n"].max()), candidates=len(cache.pool),
                subset_count=int(result["subset_logits"].shape[1]),
                peak_allocated_mib=torch.cuda.max_memory_allocated(device) / 2**20,
                peak_reserved_mib=torch.cuda.max_memory_reserved(device) / 2**20,
                loss=float(loss.detach()), champion_nll=float(champion.detach()), ranking_nll=float(ranking.detach()),
                gradient_norm=float(gradient.detach()), step_seconds=time.monotonic() - tick))
            print(json.dumps(rows[-1]), flush=True)
            del batch, fp64, order, result, loss, champion, ranking
        finally:
            cache.close()
    result = dict(complete=True, formal_training_started=False, device=torch.cuda.get_device_name(0),
        gpu_uuid=__import__("os").environ["CUDA_VISIBLE_DEVICES"], microbatch_caps=CAPS,
        eval_batch_size=256, matched_initial_sha256=identity["trainable_sha256"],
        initialization=identity, corners=rows, seconds=time.monotonic() - started,
        no_actual_training_checkpoint_reused=True, no_test_reads=True,
        loss="0.5 native champion NLL +0.5 mean full elimination NLL; one dropout forward")
    write_json(output, result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        capacity(args.output)
    except Exception as exc:
        write_json(args.output.with_name("capacity_failure.json"), dict(complete=False,
            formal_training_started=False, error_type=type(exc).__name__, error=str(exc)))
        raise

"""R80 same-lock LIB evaluation, offline PDFs, and one-shot progress snapshots.

No training, checkpoint selection, solver execution, or background monitoring.
Heavy model/plot imports are deliberately lazy so --status works during startup.
"""

import argparse
import csv
import fcntl
import hashlib
import importlib
import json
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = Path(__file__).resolve().parent / "runs/R80_training_and_code_ablation"
ARMS = {"A_ce_only": 49680, "B_batch512": 99360,
        "C_lr3e4": 49680, "D_code_embedding": 49680}
LIBS = {"TSPLIB": "TSP", "CVRPLIB": "CVRP"}
WINNER_RULE = "native/FP64 raw_label ind (not FP32 cost argmin)"
METRIC_KEYS = ("top1_pct", "top2_pct", "top3_pct", "sbs_top1_pct", "mean_cost",
               "sbs_cost", "oracle_cost", "vs_sbs_pct", "vs_oracle_pct",
               "actual_regret_pct", "top1_fp32_argmin_pct")


def utc():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, ensure_ascii=True, allow_nan=False)
            stream.write("\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(part)
    return digest.hexdigest()


def json_hash(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def resolve_record_path(value, run, must_exist=True):
    path = Path(value)
    if path.is_absolute():
        candidates = [path]
    else:
        candidates = [run / path, REPO / path, Path.cwd() / path]
    existing = list(dict.fromkeys(p.resolve() for p in candidates if p.exists()))
    if len(existing) > 1:
        raise ValueError(f"Ambiguous recorded path: {value}")
    if existing:
        return existing[0]
    if must_exist:
        raise FileNotFoundError(f"Missing recorded path: {value}")
    return candidates[0].resolve()


def _optional(path):
    if not path.is_file():
        return {}
    try:
        return read_json(path)
    except (OSError, ValueError) as error:
        return {"phase": "unreadable_status", "error": str(error)}


def status_snapshot(root):
    """Read receipts only; never open checkpoints, data, or prediction archives."""
    root = root.resolve()
    rows = []
    lines = ["# R80 Status", "", f"Snapshot UTC: {utc()}",
             "One-shot snapshot, not a live monitor. Refresh with --status.", "",
             "| Arm | Runner Phase | Updates / Budget | Progress | ETA Hours | TEST Receipt | Reporting | Updated UTC |",
             "|---|---|---:|---:|---:|---|---|---|"]
    for arm, budget in ARMS.items():
        run = root / arm
        status = _optional(run / "status.json")
        training = _optional(run / "training_complete.json")
        test = _optional(run / "test_complete.json")
        post = _optional(run / "postprocess_status.json")
        launch = _optional(root / f"launch_{arm}.json")
        controller = _optional(root / f"launch_{arm}_controller.json")
        updates = status.get("successful_updates", status.get("update", status.get("updates")))
        if updates is None:
            updates = training.get("successful_updates", "unknown")
        phase = status.get("phase", status.get("status", "not_started_or_no_status"))
        row = dict(arm=arm, phase=phase, successful_updates=updates, budget=budget,
                   test_receipt=bool(test) and not test.get("error"),
                   postprocess=post.get("phase", "not_started"),
                   updated_utc=status.get("updated_utc", status.get("utc", "unknown")),
                   training_status=str(run / "status.json"),
                   postprocess_status=str(run / "postprocess_status.json"),
                   host=status.get("host", launch.get("host")), gpu=status.get("gpu"),
                   launcher_state=launch.get("state", controller.get("state", "no_receipt")),
                   launcher_error=launch.get("error", controller.get("error")),
                   eta_seconds=status.get("eta_seconds"))
        progress = f"{100 * updates / budget:.2f}%" if isinstance(updates, (int, float)) else "unknown"
        eta = row["eta_seconds"]
        eta = f"{eta / 3600:.2f}" if isinstance(eta, (int, float)) else "unknown"
        rows.append(row)
        safe = lambda value: str(value).replace("|", "/").replace("\n", " ")
        lines.append("| " + " | ".join(safe(v) for v in (
            arm, phase, f"{updates} / {budget}", progress, eta, "present" if row["test_receipt"] else "missing",
            row["postprocess"], row["updated_utc"])) + " |")
    lines += ["", "## Per-Arm Paths", ""]
    for row in rows:
        run = root / row["arm"]
        lines += [f"- `{row['arm']}`: `{run / 'status.json'}`; `{run / 'train.log'}`.",
                  f"- Report: `{run / 'postprocess_status.json'}`; `{run / 'reporting.log'}`; "
                  f"`{run / 'full_model_statistics/MODEL_STATISTICS.pdf'}`."]
        post = _optional(run / "postprocess_status.json")
        if post.get("error"):
            error = str(post["error"]).replace("\n", " ")
            lines += [f"- Reporting error: {error}. Training is retained; rerun only r80_reporting."]
        lines += [f"- Launcher: `{row['launcher_state']}`; `{root / ('launch_' + row['arm'] + '.json')}`."]
        if row["launcher_error"]:
            error = str(row["launcher_error"]).replace("\n", " ")
            lines += [f"- Launcher error: {error}."]
    lines += ["", "W&B: https://wandb.ai/yjkds-southern-university-of-science-technology/selector",
              "Specific run URL: runner logs / ARM/wandb/latest-run/logs/debug.log.",
              "Offline W&B is not a training failure. This command does not sync or monitor.", ""]
    root.mkdir(parents=True, exist_ok=True)
    path = root / "STATUS.md"
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text("\n".join(lines), encoding="utf-8")
    temporary.replace(path)
    print(json.dumps(dict(status_path=str(path), arms=rows), indent=2), flush=True)
    return rows


def check_identity(record, lock, label, require_epoch=True):
    if record.get("checkpoint_sha256") != lock["checkpoint_sha256"]:
        raise ValueError(f"{label}: checkpoint SHA256 differs from unique R80 VAL lock")
    if require_epoch and record.get("epoch", record.get("checkpoint_epoch")) != lock["epoch"]:
        raise ValueError(f"{label}: epoch differs from unique VAL lock")
    if record.get("selected_update", record.get("checkpoint_update")) != lock["selected_update"]:
        raise ValueError(f"{label}: successful update differs from unique VAL lock")
    if record.get("state_hash") != lock["state_hash"]:
        raise ValueError(f"{label}: protocol state hash differs from unique VAL lock")


def locked_inputs(root, arm):
    """Gates precede all TEST prediction reads; no model selection occurs here."""
    run = root / arm
    config = read_json(run / "protocol.json")
    complete = read_json(run / "training_complete.json")
    lock = read_json(run / "final_test_lock.json")
    checkpoint = run / "best.pt"
    if (config.get("experiment") != "R80" or config.get("arm") != arm
            or lock.get("experiment") != "R80" or lock.get("arm") != arm
            or lock.get("selected_without_test") is not True
            or lock.get("per_problem_assembly") is not False
            or complete.get("successful_updates") != ARMS[arm]
            or config.get("max_successful_updates") != ARMS[arm]
            or not 0 < int(lock.get("selected_update", 0)) <= ARMS[arm]
            or int(lock.get("epoch", 0)) < 1
            or lock.get("state_hash") != json_hash(config)
            or resolve_record_path(lock["checkpoint"], run) != checkpoint.resolve()
            or file_hash(checkpoint) != lock["checkpoint_sha256"]):
        raise ValueError("Require completed R80 arm and unchanged unique full-VAL checkpoint lock")
    if complete.get("selected_best") is not None:
        check_identity(complete["selected_best"], lock, "training completion")
    if lock.get("rule") != config.get("checkpoint_rule") or not lock.get("rule"):
        raise ValueError("Lock selection rule differs from precommitted training protocol")
    if lock.get("inference_rule") != config.get("inference_rule") or not lock.get("inference_rule"):
        raise ValueError("Lock inference rule differs from precommitted training protocol")
    from code.unified_selector.registry import PROBLEMS
    baselines = config.get("baselines", {})
    if set(baselines) != set(PROBLEMS) or any(
            v.get("selection_split") != "train" for v in baselines.values()):
        raise ValueError("Require FULL TRAIN-fitted SBS for all 18 problems")
    # The full TEST is owned by the runner; the reporter will never rerun it.
    test = read_json(run / "test_results.json")
    if test.get("complete") is not True:
        raise ValueError("Runner full TEST is missing/incomplete; rerun reporting only after it succeeds")
    check_identity(test, lock, "full TEST")
    final = read_json(run / "final_metrics.json")
    check_identity(final, lock, "locked VAL")
    if set(test["result"]["per_problem"]) != set(PROBLEMS):
        raise ValueError("Full TEST must include all 18 problems")
    if set(final["val"]["per_problem"]) != set(PROBLEMS):
        raise ValueError("Locked VAL must include all 18 problems")
    return run, config, lock, test["result"], final["val"]


def reporting_helpers():
    # The old CPU-only report sets CUDA_VISIBLE_DEVICES at import time. Restore it
    # before any R80 factory/torch call so launcher-owned device binding survives.
    key = "CUDA_VISIBLE_DEVICES"
    visible = os.environ.get(key)
    try:
        return importlib.import_module("code.V4.r77_best_model_report")
    finally:
        if visible is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = visible


def selected_val_path(run, lock):
    candidates = [run / "evaluation/val/selected_best/rank000.npz",
                  run / f"evaluation/val/epoch{lock['epoch']:03d}/rank000.npz",
                  run / f"evaluation/val/update{lock['selected_update']:07d}/rank000.npz"]
    return next((p for p in candidates if p.is_file()), candidates[0])


def metric_row(problem, split, data, saved, baseline, config, lock, source):
    import numpy as np
    helper = reporting_helpers()
    row = helper.metric_row(problem, split, data, saved, baseline,
                            Path(config["data_root"]).name, source)
    row.update(model=f"R80/{config['arm']}", checkpoint_sha256=lock["checkpoint_sha256"],
               winner_rule=WINNER_RULE, sbs_selection_split="FULL TRAIN (fixed before VAL/TEST/LIB)")
    slot = data.pool.index(baseline["sbs_solver"])
    oracle = data.costs.min(1)
    selected = data.costs[np.arange(data.n), data.pred]
    checks = dict(oracle_cost=float(oracle.mean()),
                  vs_sbs_pct=float((selected.mean() / data.costs[:, slot].mean() - 1) * 100),
                  vs_oracle_pct=float((selected.mean() / oracle.mean() - 1) * 100),
                  actual_regret_pct=float(((selected - oracle) / oracle * 100).mean()))
    for key, value in checks.items():
        np.testing.assert_allclose(row[key], value, rtol=0, atol=1e-9,
                                   err_msg=f"{split}/{problem}: {key}")
    if split in LIBS:
        row["data_version"] = "original native LIB / historical NSS comparison"
    return row


def macro_row(split, rows, saved):
    import numpy as np
    helper = reporting_helpers()
    row = helper.macro_row(split, rows, saved["macro"], rows[0]["data_version"],
                           rows[0]["metric_source"])
    row["checkpoint_sha256"] = rows[0]["checkpoint_sha256"]
    row["aggregation"] = "equal weight across 18 problems; cross-problem costs descriptive only"
    for key in METRIC_KEYS:
        np.testing.assert_allclose(row[key], np.mean([r[key] for r in rows]), rtol=0, atol=1e-9)
    return row


def load_saved_splits(run, config, lock, test, val):
    import numpy as np
    from code.unified_selector.registry import GLOBAL_SOLVERS, PROBLEMS
    helper = reporting_helpers()
    predictions, metrics, methods = {}, [], []
    expected_model_hash = test["identity"]["model_state_sha256"]
    if val["identity"]["model_state_sha256"] != expected_model_hash:
        raise ValueError("Selected VAL and full TEST were evaluated from different parameter/buffer states")
    for split, result, path, source in (
            ("VAL", val, selected_val_path(run, lock), run / "final_metrics.json"),
            ("TEST", test, run / "evaluation/test/selected_best/rank000.npz", run / "test_results.json")):
        identity = read_json(path.parent / "identity.json")
        if (identity != result["identity"] or identity.get("experiment") != "R80"
                or identity.get("arm") != config["arm"] or identity.get("split") != split.lower()
                or identity.get("successful_updates") != lock["selected_update"]
                or identity.get("state_hash") != lock["state_hash"]
                or identity.get("model_state_sha256") != expected_model_hash
                or (split == "TEST" and identity.get("checkpoint_sha256") != lock["checkpoint_sha256"])):
            raise ValueError(f"Saved {split} prediction identity differs from selected R80 model state")
        rows = []
        with np.load(path, allow_pickle=False) as archive:
            for problem in PROBLEMS:
                data = helper.load_predictions(archive, path, problem + "__")
                if (data.n != result["per_problem"][problem]["n"] or data.n != 3895
                        or data.pool_ids.tolist() != sorted(data.pool_ids.tolist())
                        or [GLOBAL_SOLVERS[i] for i in data.pool_ids] != data.pool):
                    raise ValueError(f"Invalid full new-data coverage or stable pool: {split}/{problem}")
                np.testing.assert_array_equal(data.indices, np.arange(data.n))
                predictions[split, problem] = data
                saved = result["per_problem"][problem]
                rows.append(metric_row(problem, split, data, saved, config["baselines"][problem],
                                       config, lock, source))
                observations = helper.method_rows(problem, split, saved, data, Path(config["data_root"]).name)
                methods.extend(dict(r, model=f"R80/{config['arm']}") for r in observations)
        metrics.extend([macro_row(split, rows, result), *rows])
    return predictions, metrics, methods


def r80_lib(root, run, config, lock, device_name, batch_size, phase, expected_model_hash):
    """R77 data/inference utilities, exclusively the R80 worker's model factory."""
    import numpy as np
    import torch
    from .nss_comparison import infer
    from .r41_frozen_evaluation import attach_geometry
    from .r76_experiment import fixed_sbs_metrics
    from .r77_lib import SCHEMA, checked_predictions, checked_raw, summarize
    from .r77_data import parameter_hash
    from .train import configure_torch, make_loader
    factory = importlib.import_module("code.V4.r80_experiment")
    if not callable(getattr(factory, "load_checkpoint", None)):
        raise ValueError("R80 worker must expose load_checkpoint(path, device); no historical model fallback")
    configure_torch()
    torch.set_num_threads(2)
    device = torch.device(device_name)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("LIB inference supports only CPU or launcher-assigned CUDA")
    directory = run / "lib_evaluation"
    directory.mkdir(exist_ok=True)
    model, state, buffer_reference = None, None, None
    results = {}
    for dataset, problem in LIBS.items():
        phase("lib_evaluation", dataset=dataset)
        raw = checked_raw(problem, dataset, config)
        prediction_path = directory / "predictions" / f"{dataset}.npz"
        result_path = directory / f"{dataset}_results.json"
        protocol_path = directory / f"{dataset}_protocol.json"
        started = time.monotonic()
        sources = {name: dict(path=str(Path(raw["directory"]) / name),
                             sha256=file_hash(Path(raw["directory"]) / name))
                   for name in ("dataset.pkl", "raw_label.pkl")}
        identity = dict(experiment="R80", arm=config["arm"], dataset=dataset, problem=problem,
                        checkpoint=str((run / "best.pt").resolve()),
                        checkpoint_sha256=lock["checkpoint_sha256"], epoch=lock["epoch"],
                        selected_update=lock["selected_update"], candidate_pool=raw["pool"],
                        state_hash=lock["state_hash"], model_state_sha256=expected_model_hash,
                        pool_ids=raw["pool_ids"], sources=sources, winner_rule=WINNER_RULE,
                        data_version="original native LIB / historical NSS comparison",
                        model_factory="code.V4.r80_experiment.load_checkpoint",
                        inference_rule="single FP32 forward, logits argmax, lowest global solver ID for exact ties",
                        locked_inference_rule=lock["inference_rule"],
                        normalization="all TRAIN buffers restored from same R80 best.pt, no LIB refit",
                        batch_size=batch_size, device=str(device), training=False,
                        solver_execution=False, embedding_api=False, checkpoint_reselection=False,
                        prediction_schema=SCHEMA)
        if protocol_path.exists():
            previous = read_json(protocol_path)
            # Reusing deterministic saved predictions does not require the same
            # available device; all data/model/rule identities must still match.
            for key, value in identity.items():
                if key not in ("device",) and previous.get(key) != value:
                    raise ValueError(f"Existing {dataset} protocol changed: {key}")
            if not prediction_path.is_file() and previous.get("device") != identity["device"]:
                raise ValueError(f"Retry unfinished {dataset} inference on its recorded --device {previous.get('device')}; "
                                 "do not silently change the LIB inference device")
        else:
            if prediction_path.exists() or result_path.exists():
                raise ValueError(f"Unattributed {dataset} artifacts; refusing historical/stale results")
            write_json(protocol_path, dict(identity, created_utc=utc()))
        reused = prediction_path.is_file()
        if reused:
            data = checked_predictions(prediction_path, raw, lock)
        else:
            if model is None:
                model, state = factory.load_checkpoint(run / "best.pt", device)
                if state.get("args") != config or state.get("successful_updates") != lock["selected_update"]:
                    raise ValueError("R80 factory returned another protocol/checkpoint update")
                if state.get("epoch") != lock["epoch"]:
                    raise ValueError("R80 factory checkpoint epoch differs from one-based VAL lock")
                actual, expected = model.state_dict(), state["model"]
                if set(actual) != set(expected):
                    raise ValueError("R80 factory did not restore the complete checkpoint state/buffer schema")
                if parameter_hash(expected) != expected_model_hash:
                    raise ValueError("R80 factory model state differs from locked VAL/TEST evaluation identities")
                for key, value in actual.items():
                    torch.testing.assert_close(value.detach().cpu(), expected[key].detach().cpu(),
                                               rtol=0, atol=0, msg=f"R80 factory state mismatch: {key}")
                model.eval().requires_grad_(False)
                buffer_reference = {key: value.detach().cpu().clone() for key, value in model.named_buffers()}
            with torch.inference_mode():
                loader = make_loader(problem, dataset, batch_size, 0, coord_augment=0,
                                     shuffle=False, cache_device=device)
                if loader.size != len(raw["winner"]):
                    raise ValueError("LIB loader lost native rows")
                np.testing.assert_array_equal(loader.batch["pool_ids"].cpu(), raw["pool_ids"])
                np.testing.assert_array_equal(loader.batch["ind"].cpu(), raw["winner"])
                np.testing.assert_array_equal(loader.batch["costs"].cpu(), raw["costs"].astype(np.float32))
                attach_geometry(loader, device)
                temporary = prediction_path.with_name(f".{dataset}.{os.getpid()}.npz")
                try:
                    infer(model, loader, batch_size, raw, temporary)
                    with np.load(temporary, allow_pickle=False) as archive:
                        data = {key: archive[key] for key in archive.files}
                    data.update(winner_fp32_argmin=raw["costs"].astype(np.float32).argmin(1),
                                checkpoint_sha256=np.array(lock["checkpoint_sha256"]),
                                checkpoint_epoch=np.array(lock["epoch"], dtype=np.int64),
                                checkpoint_update=np.array(lock["selected_update"], dtype=np.int64))
                    np.savez_compressed(temporary, **data)
                    checked_predictions(temporary, raw, lock)
                    for key, value in model.named_buffers():
                        torch.testing.assert_close(value.detach().cpu(), buffer_reference[key], rtol=0, atol=0,
                                                   msg=f"LIB inference changed checkpoint buffer: {key}")
                    temporary.replace(prediction_path)
                finally:
                    temporary.unlink(missing_ok=True)
                del loader
        result = dict(**summarize(data, raw, config["baselines"][problem]),
                      protocol=read_json(protocol_path), inference_reused=reused,
                      seconds=time.monotonic() - started, completed_utc=utc())
        # Main reporting always uses the TRAIN SBS reference, never LIB posthoc SBS.
        fixed, pred = fixed_sbs_metrics(data["scores"], raw, config["baselines"][problem])
        np.testing.assert_array_equal(pred, data["pred"])
        result["train_sbs_reference"] = fixed
        write_json(result_path, result)
        results[dataset] = result
        print(f"[R80 {config['arm']}] {dataset} native Top1={100*fixed['top1']:.6f}% "
              f"mean_cost={fixed['mean_cost']:.9f}; reused={reused}", flush=True)
    if file_hash(run / "best.pt") != lock["checkpoint_sha256"]:
        raise ValueError("Locked R80 checkpoint changed during LIB evaluation")
    return results


def nss_comparisons(root, config, lock, predictions, nss_root=None):
    import numpy as np
    helper = reporting_helpers()
    nss_root = nss_root or root.parent / "R79_NSS_new_data_batch1024"
    rows, alignment = [], {}
    protocol_path, lock_path = nss_root / "protocol.json", nss_root / "final_test_lock.json"
    available = protocol_path.is_file() and lock_path.is_file()
    nss_protocol, nss_lock = (read_json(protocol_path), read_json(lock_path)) if available else ({}, {})
    if available and any(Path(record["data_root"]).resolve() != Path(config["data_root"]).resolve()
                         for record in (nss_protocol, nss_lock)):
        raise ValueError("R79 NSS and R80 do not use the same new-data release")
    for dataset, problem, split in (("TSP TEST", "TSP", "TEST"), ("CVRP TEST", "CVRP", "TEST"),
                                    ("TSPLIB", "TSP", "TSPLIB"), ("CVRPLIB", "CVRP", "CVRPLIB")):
        current = predictions[split, problem]
        path = nss_root / problem / f"{'test' if split == 'TEST' else split}_predictions.npz"
        metadata_path = path.with_name(path.name.replace("_predictions.npz", "_metrics.json"))
        row = dict(dataset=dataset, problem=problem, n=current.n,
                   data_version=Path(config["data_root"]).name if split == "TEST" else "original native LIB",
                   nss_model="R79_NSS_new_data_batch1024", current_model=f"R80/{config['arm']}",
                   nss_top1_native_pct=None, current_top1_native_pct=current.top1_pct,
                   nss_mean_cost=None, current_mean_cost=current.mean_cost,
                   nss_original_fp32_top1_pct=None, current_fp32_top1_pct=current.fp32_top1_pct,
                   nss_checkpoint_sha256=None, current_checkpoint_sha256=lock["checkpoint_sha256"],
                   candidate_pool=json.dumps(current.pool), winner_rule=WINNER_RULE,
                   nss_prediction_path=str(path), current_prediction_path=str(current.path),
                   status="missing_saved_NSS", alignment_checked_rows=0)
        if available and path.is_file() and metadata_path.is_file():
            metadata = read_json(metadata_path)
            nss_identity = nss_lock["checkpoints"][problem]
            if metadata.get("checkpoint", {}).get("sha256") != nss_identity["sha256"]:
                raise ValueError(f"NSS saved {dataset} does not match R79 lock")
            with np.load(path, allow_pickle=False) as archive:
                nss = helper.load_predictions(archive, path, r77=False)
            checks = helper.check_comparison_alignment(current, nss)
            alignment[dataset] = checks
            if "top1_native_ind" in metadata:
                np.testing.assert_allclose(nss.top1_pct, metadata["top1_native_ind"] * 100, rtol=0, atol=1e-9)
            np.testing.assert_allclose(nss.mean_cost, metadata["mean_cost"], rtol=0, atol=1e-9)
            row.update(nss_top1_native_pct=nss.top1_pct, nss_mean_cost=nss.mean_cost,
                       nss_original_fp32_top1_pct=nss.fp32_top1_pct,
                       nss_checkpoint_sha256=nss_identity["sha256"], status="complete",
                       alignment_checked_rows=checks["checked_rows"])
        rows.append(row)
    return rows, alignment


def _page(helper, title, subtitle, model, lock, page):
    fig = helper.new_page(title, subtitle, page)
    fig.texts[-2].set_text(f"{model} | same checkpoint {lock['checkpoint_sha256'][:16]} | native/FP64 ind")
    return fig


def _distribution(helper, problem, pool, panels, version, model, lock):
    fig = helper.plot_distribution(problem, pool, panels, version, lock)
    # Reuse the proven fixed-size R77 layout, but replace every model-specific
    # caption. No R77 epoch, weights, inference claim, or selection is inherited.
    for text in fig.texts:
        value = text.get_text().replace("R77B", model)
        value = value.replace(helper.SHA256[:16], lock["checkpoint_sha256"][:16])
        value = value.replace("No training, inference, checkpoint reselection or Rxx mixing.",
                              "Same R80 lock; LIB inference only; no training/reselection/Rxx mixing.")
        text.set_text(value)
    return fig


def diagnostic_pdf(run, output, predictions, methods, model, lock):
    """Required observables from saved history/predictions, never new inference."""
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from code.unified_selector.registry import PROBLEMS
    helper = reporting_helpers()
    path = output / "training_and_method_diagnostics.pdf"
    temporary = path.with_name(f".{path.stem}.{os.getpid()}.pdf")
    history_path = run / "history.json"
    history = read_json(history_path) if history_path.is_file() else []
    try:
        with PdfPages(temporary, metadata={"Title": f"{model} saved training/method diagnostics"}) as pdf:
            fig = _page(helper, f"{model} | Saved Training History",
                        "Successful-update axis. Training losses and full VAL traces are descriptive only; the checkpoint is already locked.",
                        model, lock, 1)
            axes = fig.subplots(2, 2)
            fig.subplots_adjust(left=.08, right=.95, top=.80, bottom=.11, wspace=.25, hspace=.32)
            steps = [row["successful_updates"] for row in history]
            for name in ("loss", "ce", "cmp", "risk", "kl"):
                points = [(step, row["optimization"][name]) for step, row in zip(steps, history)
                          if name in row.get("optimization", {})]
                if points:
                    axes[0, 0].plot(*zip(*points), label=name, linewidth=1.5)
            axes[0, 0].set_title("TRAIN interval-mean loss components")
            if axes[0, 0].lines:
                axes[0, 0].legend(fontsize=9)
            traces = ((axes[0, 1], "macro_top1", 100., "VAL ALL native Top1 (%)"),
                      (axes[1, 0], "macro_vs_sbs_pct", 1., "VAL ALL vs fixed TRAIN SBS (%)"),
                      (axes[1, 1], "macro_actual_regret_pct", 1., "VAL ALL actual regret (%)"))
            for ax, key, factor, title in traces:
                points = [(step, row["val"]["macro"][key] * factor) for step, row in zip(steps, history)
                          if key in row.get("val", {}).get("macro", {})]
                if points:
                    ax.plot(*zip(*points), color="#177e89", linewidth=1.8)
                ax.set_title(title)
            for ax in axes.flat:
                ax.axvline(lock["selected_update"], color="#c45358", linestyle="--", linewidth=1)
                ax.set_xlabel("Successful optimizer updates")
                ax.grid(alpha=.18)
                if not history:
                    ax.text(.5, .5, "MISSING saved history.json", ha="center", transform=ax.transAxes)
            pdf.savefig(fig)
            plt.close(fig)
            for page, problem in enumerate(PROBLEMS, 2):
                splits = ["VAL", "TEST"] + [name for name, p in LIBS.items() if p == problem]
                fig = _page(helper, f"{model} | {problem}: Selector vs Single Methods",
                            "Native winner Top1 and FP64 mean cost; every selection is from the same locked model. Lower cost is better.",
                            model, lock, page)
                axes = fig.subplots(len(splits), 2, squeeze=False)
                fig.subplots_adjust(left=.15, right=.96, top=.81, bottom=.09, wspace=.37, hspace=.65)
                pool = predictions["TEST", problem].pool
                for row_idx, split in enumerate(splits):
                    observations = {r["candidate"]: r for r in methods if r["problem"] == problem and r["split"] == split}
                    rows = [observations[name] for name in pool]
                    for column, (field, selector, title) in enumerate((
                            ("method_top1_native_pct", "selector_top1_native_pct", "Native Top1 (%)"),
                            ("method_mean_cost", "selector_mean_cost", "Mean cost (FP64)"))):
                        ax = axes[row_idx, column]
                        values = [row[field] for row in rows]
                        y = np.arange(len(pool))
                        colors = ["#c45358" if row["is_train_sbs"] else "#9cabb0" for row in rows]
                        ax.barh(y, values, color=colors, height=.62)
                        ax.axvline(rows[0][selector], color="#177e89", linewidth=1.8, label="selector")
                        ax.set_yticks(y, pool, fontsize=8)
                        ax.set_ylim(len(pool)-.4, -.6)
                        ax.set_title(f"{split} | {title}", fontsize=11)
                        ax.tick_params(axis="x", labelsize=8)
                        ax.grid(axis="x", alpha=.15)
                        ax.legend(loc="upper right", fontsize=8)
                pdf.savefig(fig)
                plt.close(fig)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return dict(path=str(path), pages=19, history_status="complete" if history else "missing_saved_history")


def build_reports(root, run, config, lock, test, val, lib_results, predictions, metrics, methods, phase, nss_root=None):
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    helper = reporting_helpers()
    from code.unified_selector.registry import PROBLEMS
    for dataset, problem in LIBS.items():
        path = run / "lib_evaluation/predictions" / f"{dataset}.npz"
        with np.load(path, allow_pickle=False) as archive:
            data = helper.load_predictions(archive, path)
        predictions[dataset, problem] = data
        saved = lib_results[dataset]["train_sbs_reference"]
        metrics.append(metric_row(problem, dataset, data, saved, config["baselines"][problem], config, lock,
                                  run / f"lib_evaluation/{dataset}_results.json"))
        methods.extend(dict(row, model=f"R80/{config['arm']}") for row in
                       helper.method_rows(problem, dataset, saved, data, "original native LIB"))
    comparisons, alignment = nss_comparisons(root, config, lock, predictions, nss_root)
    output = run / "full_model_statistics"
    output.mkdir(exist_ok=True)
    figures = output / "figures"
    figures.mkdir(exist_ok=True)
    model, version = f"R80/{config['arm']}", Path(config["data_root"]).name
    panel_sets, distributions = {}, []
    for problem in PROBLEMS:
        test_data = predictions["TEST", problem]
        dataset = next((name for name, p in LIBS.items() if p == problem), None)
        lib = predictions.get((dataset, problem))
        panels = helper.make_panels(problem, test_data, lib, test_data.pool)
        panel_sets[problem] = panels
        for row in helper.distribution_rows(problem, panels, test_data.pool, version):
            row.update(model=model, checkpoint_sha256=lock["checkpoint_sha256"], winner_rule=WINNER_RULE)
            distributions.append(row)
    write_csv(output / "metrics_by_problem.csv", metrics)
    write_csv(output / "nss_comparison.csv", comparisons)
    write_csv(output / "selection_distribution.csv", distributions)
    write_csv(output / "single_method_comparison.csv", methods)
    phase("pdf_generation")
    helper.setup_style()
    metadata = dict(Title=f"{model} locked model statistics", Author="R80 locked reporting",
                    Subject="Native winner metrics, FULL TRAIN SBS, same VAL lock on all datasets")
    pdf = output / "MODEL_STATISTICS.pdf"
    dist_pdf = output / "selection_distributions.pdf"
    pdf_tmp = pdf.with_name(f".{pdf.stem}.{os.getpid()}.pdf")
    dist_tmp = dist_pdf.with_name(f".{dist_pdf.stem}.{os.getpid()}.pdf")
    headings = ["Split / Problem", "Top1 (%)", "SBS Top1 (%)", "mean_cost", "SBS cost", "Oracle cost",
                "vs_SBS (%)", "vs_Oracle (%)", "actual regret (%)"]
    page = 0
    try:
        with PdfPages(pdf_tmp, metadata=metadata) as report:
            page += 1
            fig = _page(helper, f"{model} | Locked Full-Model Statistics",
                        "VAL-selected unique checkpoint; native/FP64 ind primary; no TEST/LIB reselection.", model, lock, page)
            fig.text(0.045, 0.85, f"epoch {lock['epoch']} | selected update {lock['selected_update']} | seed 2 | from scratch", fontsize=12)
            fig.text(0.045, 0.814, f"SHA256: {lock['checkpoint_sha256']}", fontsize=10)
            fig.text(0.045, 0.778, f"New data: {version}", fontsize=9.5)
            all_rows = [next(r for r in metrics if r["split"] == split and r["problem"] == "ALL")
                        for split in ("VAL", "TEST")]
            helper.add_table(fig, headings, [helper.metric_cells(r, r["split"] + " ALL") for r in all_rows],
                             [0.045, 0.605, 0.91, 0.12], font=10)
            fig.text(0.045, 0.561, "NSS vs current R80 | Same native winner definition; saved R79 selections", fontsize=13)
            fmt = lambda row, key, digits=6: "MISSING" if row[key] is None else f"{row[key]:.{digits}f}"
            cells = [[r["dataset"], str(r["n"]), fmt(r, "nss_top1_native_pct"), fmt(r, "current_top1_native_pct"),
                      fmt(r, "nss_mean_cost", 9), fmt(r, "current_mean_cost", 9),
                      fmt(r, "nss_original_fp32_top1_pct"), fmt(r, "current_fp32_top1_pct")] for r in comparisons]
            helper.add_table(fig, ["Dataset", "n", "NSS native\nTop1 (%)", "R80 native\nTop1 (%)", "NSS mean_cost",
                                   "R80 mean_cost", "NSS FP32\nTop1 (%)", "R80 FP32\nTop1 (%)"], cells,
                             [0.045, 0.315, 0.91, 0.21], widths=[.13, .05, .13, .13, .16, .16, .12, .12], font=10)
            notes = ["Primary Top1 = 100 * mean(pred == native/FP64 ind). FP32 columns are compatibility-only.",
                     "R79 NSS TSP TEST: native 44.338896%; FP32 argmin 43.953786%. No label-definition mixing.",
                     "SBS identities are fixed on FULL TRAIN. Oracle is the per-instance legal-pool minimum.",
                     "ALL = equal-problem mean of each metric; cross-problem raw costs are descriptive only.",
                     "TSP/CVRP TEST: new data; TSPLIB/CVRPLIB: original native LIB, not old TEST results.",
                     "TEST partly informed prior multi-round R&D; not an untouched blind-set claim. No reselection here."]
            for i, text in enumerate(notes):
                fig.text(0.045, 0.274 - i * .033, text, fontsize=10.5)
            report.savefig(fig)
            plt.close(fig)
            for split in ("VAL", "TEST"):
                page += 1
                fig = _page(helper, f"{model} | {split}: ALL + 18 Problems",
                            "Native/FP64 ind; same locked checkpoint and FULL TRAIN SBS; Top2/3 and method comparisons in CSV.", model, lock, page)
                rows = [r for r in metrics if r["split"] == split]
                cells = [helper.metric_cells(r, r["problem"]) + ["per-problem" if r["problem"] == "ALL" else r["sbs_name"]]
                         for r in rows]
                helper.add_table(fig, headings + ["TRAIN SBS"], cells, [0.045, 0.215, 0.91, .62], font=10, bold_first=True)
                for i, text in enumerate((
                        "vs_SBS = 100*(mean_cost/SBS_cost - 1); vs_Oracle = 100*(mean_cost/Oracle_cost - 1).",
                        "actual regret = mean_i[100*(selected_i-min_legal_i)/min_legal_i]; not a ratio of means.",
                        "ALL rates/costs are equal-weight means across 18 problems. Do not compare mixed raw-cost ratios.",
                        f"Data: {version}")):
                    fig.text(.045, .171 - i*.035, text, fontsize=10.5 if i < 3 else 9.5)
                report.savefig(fig)
                plt.close(fig)
            page += 1
            fig = _page(helper, f"{model} | Native LIB And Evaluation Provenance",
                        "R80 factory and every source/ID/geometry checkpoint buffer restored; no historical R77/R76 LIB scores.", model, lock, page)
            lib_rows = [r for r in metrics if r["split"] in LIBS]
            helper.add_table(fig, headings, [helper.metric_cells(r, f"{r['split']} n={r['n']}") for r in lib_rows],
                             [.045, .72, .91, .12], font=10)
            rules = [f"Lock rule: {lock['rule']}",
                     "All problems, VAL/TEST, TSPLIB and CVRPLIB use one arm/checkpoint, not per-dataset best versions.",
                     "Inference: single FP32 forward; logits argmax; exact ties use lowest stable global solver ID.",
                     "Native FP64 costs and saved ind are preserved; no solver solving, embedding API, or normalization refit.",
                     "Main SBS uses the precommitted FULL TRAIN identity, not the retrospective best LIB solver.",
                     "NSS uses saved pred directly, including its original score-tie behavior; it is never retrained here.",
                     "NSS alignment: same new-data version, candidates aligned by name; three instance rows checked per dataset.",
                     "Prediction paths, data hashes, model factory, candidate order and formulas are in protocol.json.",
                     "Reporting failure preserves completed training/TEST. Re-running this reporter reuses checked same-lock LIB predictions.",
                     "Native model Top1 is shown only on model-selection panels, never assigned to true-winner panels."]
            import textwrap
            y = .67
            for text in rules:
                for line in textwrap.wrap(text, 130):
                    fig.text(.045, y, line, fontsize=10.5)
                    y -= .028
                y -= .013
            report.savefig(fig)
            plt.close(fig)
            with PdfPages(dist_tmp, metadata=metadata) as distribution_pdf:
                for problem in PROBLEMS:
                    fig = _distribution(helper, problem, predictions["TEST", problem].pool,
                                        panel_sets[problem], version, model, lock)
                    fig.savefig(figures / f"{problem}_selection_distribution.pdf")
                    distribution_pdf.savefig(fig)
                    report.savefig(fig)
                    plt.close(fig)
                    print(f"[R80 {config['arm']}] PDF distribution: {problem}", flush=True)
        pdf_tmp.replace(pdf)
        dist_tmp.replace(dist_pdf)
    finally:
        pdf_tmp.unlink(missing_ok=True)
        dist_tmp.unlink(missing_ok=True)
    diagnostics = diagnostic_pdf(run, output, predictions, methods, model, lock)
    missing = [r["dataset"] for r in comparisons if r["status"] != "complete"]
    protocol = dict(experiment="R80", arm=config["arm"], created_utc=utc(), complete=not missing,
                    missing_nss_datasets=missing, checkpoint_lock=lock,
                    model_factory="code.V4.r80_experiment.load_checkpoint",
                    source_buffer_policy="strict equality to all tensors in locked checkpoint model state",
                    training=False, test_inference=False, lib_inference=True, checkpoint_reselection=False,
                    winner_rule=WINNER_RULE, nss_decision="saved pred; no argmax recomputation",
                    fp32_rule="auxiliary cost.astype(float32).argmin only",
                    data_root=config["data_root"], data_version=version,
                    lib_protocols={name: result["protocol"] for name, result in lib_results.items()},
                    baselines=config["baselines"], nss_alignment=alignment,
                    candidate_order={p: predictions["TEST", p].pool for p in PROBLEMS},
                    metric_formulas=dict(vs_sbs="100*(mean_cost/SBS_cost-1)",
                                         vs_oracle="100*(mean_cost/Oracle_cost-1)",
                                         actual_regret="mean_i 100*(selected_i-min_legal_i)/min_legal_i",
                                         ALL="equal-problem mean of each metric; raw costs descriptive"),
                    model_top1="100*mean(pred==native winner); never largest selection frequency",
                    distribution_pages=18, main_pdf_pages=22, test_not_untouched_blind=True)
    protocol["diagnostics"] = diagnostics
    write_json(output / "protocol.json", protocol)
    return dict(output=str(output), complete=not missing, missing_nss_datasets=missing,
                model_statistics_pdf=str(pdf), selection_distributions_pdf=str(dist_pdf), diagnostics=diagnostics)


def report(args):
    root, arm = args.root.resolve(), args.arm
    run = root / arm
    if not run.is_dir():
        raise FileNotFoundError(f"Arm does not exist; reporting never starts training: {run}")
    with (run / "postprocess.lock").open("a") as claim:
        try:
            fcntl.flock(claim, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(f"Another reporter owns {arm}; no duplicate inference/reporting") from None
        status = dict(experiment="R80", arm=arm, started_utc=utc(), pid=os.getpid(), training_retained=True,
                      training_started_by_reporter=False)

        def phase(name, **fields):
            status.update(phase=name, updated_utc=utc(), **fields)
            write_json(run / "postprocess_status.json", status)

        try:
            phase("checking_unique_val_lock_and_completed_test")
            run, config, lock, test, val = locked_inputs(root, arm)
            status["checkpoint_sha256"] = lock["checkpoint_sha256"]
            phase("loading_saved_val_test")
            predictions, metrics, methods = load_saved_splits(run, config, lock, test, val)
            lib_results = r80_lib(root, run, config, lock, args.device, args.lib_batch_size, phase,
                                  test["identity"]["model_state_sha256"])
            result = build_reports(root, run, config, lock, test, val, lib_results,
                                   predictions, metrics, methods, phase, args.nss_root)
            if file_hash(run / "best.pt") != lock["checkpoint_sha256"]:
                raise ValueError("R80 checkpoint changed while generating reports")
            phase("complete" if result["complete"] else "complete_with_missing_NSS", **result, completed_utc=utc())
            print(json.dumps(status, indent=2), flush=True)
            return result
        except BaseException as error:
            failed_stage = status.get("phase")
            phase("postprocess_failed", failed_stage=failed_stage, error=f"{type(error).__name__}: {error}",
                  traceback=traceback.format_exc(), retry="rerun reporting only; keep completed training/TEST",
                  completed_utc=utc())
            raise


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    result.add_argument("--arm", choices=tuple(ARMS))
    result.add_argument("--status", action="store_true", help="Read status receipts once and write ROOT/STATUS.md; no daemon")
    result.add_argument("--device", default="cpu", help="CPU or an explicitly launcher-assigned cuda:N; never auto-allocates")
    result.add_argument("--lib-batch-size", type=int, choices=(1, 2), default=2)
    result.add_argument("--nss-root", type=Path, help="Override existing R79 saved-prediction directory; never retrain NSS")
    return result


def main(argv=None):
    cli = parser()
    args = cli.parse_args(argv)
    if args.status:
        if args.arm:
            cli.error("--status summarizes all arms; do not combine it with --arm")
        status_snapshot(args.root)
        return 0
    if not args.arm:
        cli.error("supply --arm for reports or --status for a one-shot progress snapshot")
    try:
        report(args)
        return 0
    except Exception as error:
        print(f"R80 postprocess failed: {error}. Completed training/TEST are retained.", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

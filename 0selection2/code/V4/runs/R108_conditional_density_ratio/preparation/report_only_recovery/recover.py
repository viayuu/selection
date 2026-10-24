"""Additive, saved-evidence-only R108 report recovery. Never calls inference."""

import csv
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import traceback
from datetime import datetime, timezone


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PROJECT = ROOT.parents[3]
ARMS = ("A_prior_ce", "B_density_ratio")
os.environ.update(CUDA_VISIBLE_DEVICES="", WANDB_MODE="disabled", MPLBACKEND="Agg",
                  PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
sys.path.insert(0, str(PROJECT))


def utc():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=True)
        stream.write("\n")


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        return reader.fieldnames, list(reader)


def write_csv(path, fields, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


class OutputRoot:
    """Route only newly generated comparison artifacts outside the frozen run."""

    def __fspath__(self):
        return str(ROOT)

    def __truediv__(self, child):
        if child == "comparison_artifacts":
            return HERE / "comparison_artifacts"
        return ROOT / child


def backup():
    if (HERE / "backup_manifest.json").exists():
        raise RuntimeError("Recovery already prepared; preserve the one-shot evidence")
    paths = [p for p in ROOT.iterdir() if p.is_file() and p.suffix in {".json", ".log"}]
    paths += [ROOT / arm / "full_model_statistics/nss_comparison.csv" for arm in ARMS]
    paths += sorted((PROJECT / "code/V4").glob("r108_*.py"))
    records = []
    for source in paths:
        relative = source.relative_to(PROJECT)
        dest = HERE / "original_backup" / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            raise RuntimeError("Backup is additive only: " + str(dest))
        shutil.copy2(source, dest)
        assert digest(source) == digest(dest)
        records.append(dict(path=str(source), backup=str(dest), sha256=digest(source)))
    protected = {str(p): digest(p) for p in paths}
    for arm in ARMS:
        run = ROOT / arm
        files = [run / name for name in ("best.pt", "protocol.json", "final_test_lock.json",
            "final_metrics.json", "initialization.json", "training_complete.json",
            "postprocess_status.json", "full_model_statistics/protocol.json")]
        files += list(run.glob("evaluation/*/selected_best/*"))
        files += list(run.glob("lib_evaluation/*"))
        for p in files:
            if p.is_file():
                protected[str(p)] = digest(p)
    result = dict(started_utc=utc(), backups=records, protected_sha256=protected,
        canonical_reports_unchanged=True, runtime_sources_changed=False,
        authorization="Main authorizes bounded report-only recovery and curated R108 publication")
    save(HERE / "backup_manifest.json", result)
    return result


def normalize(reporting, goal):
    proofs = []
    for arm in ARMS:
        run = ROOT / arm
        config, lock = read(run / "protocol.json"), read(run / "final_test_lock.json")
        nss_root = reporting.DEFAULT_NSS
        identity = reporting.previous.validate_nss_reference(config, nss_root)
        plan = read(nss_root / "protocol.json")
        source = run / "full_model_statistics/nss_comparison.csv"
        fields, old = read_csv(source)
        assert len(old) == 4
        rows, evidence = [], []
        for row in old:
            ds, problem = row["dataset"], row["problem"]
            nss = Path(row["nss_prediction_path"])
            meta = read(nss.with_name(nss.name.replace("_predictions.npz", "_metrics.json")))
            expected = plan["checkpoints"][problem]["sha256"]
            assert row["nss_checkpoint_sha256"] == meta["checkpoint"]["sha256"] == expected
            assert row["current_checkpoint_sha256"] == lock["checkpoint_sha256"]
            assert int(row["n"]) == meta["n"]
            assert math.isclose(float(row["nss_top1_native_pct"]),
                                100 * meta["top1_native_ind"], abs_tol=1e-9)
            assert math.isclose(float(row["nss_mean_cost"]), meta["mean_cost"], abs_tol=1e-9)
            if ds.endswith(" TEST"):
                version = str(Path(config["data_root"]).resolve()) + "; frozen TEST 4500"
                pool = config["candidate_pools"][problem]
                assert digest(nss) == identity["prediction_sha256"][problem]
            else:
                lib = read(run / "lib_evaluation" / (ds + "_protocol.json"))
                assert lib["checkpoint_sha256"] == lock["checkpoint_sha256"]
                version = lib["data_version"]
                pool = lib["candidate_pool"]
            assert pool == meta["pool"]
            rows.append(dict(row, data_version=version, candidate_pool="|".join(pool)))
            evidence.append(dict(dataset=ds, nss_prediction_path=str(nss),
                nss_prediction_sha256=digest(nss), nss_checkpoint_sha256=expected,
                current_checkpoint_sha256=lock["checkpoint_sha256"], candidate_pool=pool,
                data_version=version))
        output = HERE / "normalized" / arm / "full_model_statistics/nss_comparison.csv"
        write_csv(output, fields + ["data_version", "candidate_pool"], rows)
        _, new = read_csv(output)
        assert [{key: row[key] for key in fields} for row in new] == old
        assert goal.nss4_ready(output.parents[1], lock)
        proofs.append(dict(arm=arm, source=str(source), source_sha256=digest(source),
            derived=str(output), derived_sha256=digest(output), original_columns_unchanged=True,
            original_gate_on_derived_csv=True, historical_NSS_identity=identity, rows=evidence))
    save(HERE / "normalization_receipt.json", dict(complete=True, completed_utc=utc(),
        canonical_CSV_modified=False, numeric_or_identity_fields_modified=False, proofs=proofs))


def render_report():
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    finals = {a: read(ROOT / a / "final_metrics.json") for a in ARMS}
    metrics = []
    for arm, final in finals.items():
        for split in ("val", "test"):
            result = final[split]
            for problem, row in result["per_problem"].items():
                metrics.append(dict(arm=arm, split=split.upper(), problem=problem, n=row["n"],
                    Top1_pct=100 * row["top1"], mean_cost=row["mean_cost"],
                    SBS_cost=row["sbs_cost"], Oracle_cost=row["oracle_cost"],
                    vs_SBS_pct=row["vs_sbs_pct"], vs_Oracle_pct=row["vs_oracle_pct"],
                    actual_regret_pct=row["actual_regret_pct"],
                    checkpoint_sha256=final["checkpoint_sha256"]))
            m = result["macro"]
            metrics.append(dict(arm=arm, split=split.upper(), problem="ALL", n=81000,
                Top1_pct=100 * m["macro_top1"], mean_cost=m["macro_mean_cost"],
                SBS_cost=m["macro_sbs_cost"], Oracle_cost=m["macro_oracle_cost"],
                vs_SBS_pct=m["macro_vs_sbs_pct"], vs_Oracle_pct=m["macro_vs_oracle_pct"],
                actual_regret_pct=m["macro_actual_regret_pct"], checkpoint_sha256=final["checkpoint_sha256"]))
    write_csv(HERE / "metrics.csv", list(metrics[0]), metrics)
    columns = ["arm", "split", "Top1_pct", "mean_cost", "vs_SBS_pct", "vs_Oracle_pct", "actual_regret_pct"]
    table = [[row[k] if k in ("arm", "split") else f"{row[k]:.6f}" for k in columns]
             for row in metrics if row["problem"] == "ALL"]
    lines = ["# R108 Report-Only Recovery", "", "Original canonical model reports, runtime sources, protocols, weights and predictions are unchanged.",
        "Derived NSS CSVs only remove BOM and add provenance metadata; all original fields are byte-for-value equal after CSV parsing.",
        "TRAIN-fitted SBS; native FP64-ind winner; ALL is equal-problem macro. Raw macro costs are descriptive only.",
        "One VAL-selected checkpoint per arm (update43200); no training, inference, calibration or reselection.", "",
        "| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    lines += ["| " + " | ".join(row) + " |" for row in table]
    lines += ["", "All18 VAL/TEST per-problem metrics: metrics.csv.",
        "Fixed36 paired recovery: comparison_artifacts/target_recovery_summary_B_vs_A.csv.",
        "NSS4: normalized/<arm>/full_model_statistics/nss_comparison.csv.",
        "Existing per-arm selection_distributions.pdf and MODEL_STATISTICS.pdf remain valid primary per-model deliveries.",
        "Original canonical watcher remains blocked on its original malformed CSV; recovery gate does not claim it succeeded.",
        "TEST/LIB have development exposure; this report establishes delivery, not selection improvement."]
    with (HERE / "RESULTS.md").open("x", encoding="ascii") as stream:
        stream.write("\n".join(lines) + "\n")
    with PdfPages(HERE / "RESULTS.pdf") as pdf:
        fig, ax = plt.subplots(figsize=(15, 7))
        ax.axis("off")
        ax.set_title("R108 | Saved-evidence CPU report-only recovery", fontsize=17, pad=24)
        item = ax.table(cellText=table, colLabels=columns, loc="center", cellLoc="center")
        item.auto_set_font_size(False)
        item.set_fontsize(9)
        item.scale(1, 2)
        fig.text(.05, .12, "Native FP64-ind | fixed TRAIN SBS | equal-problem ALL | locked VAL checkpoints (43200)", fontsize=10)
        fig.text(.05, .07, "No training/inference/tuning. Raw macro costs descriptive only. Original reports/receipts preserved.", fontsize=10)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
        for split in ("VAL", "TEST"):
            for arm in ARMS:
                selected = [r for r in metrics if r["split"] == split and r["arm"] == arm]
                keys = ["problem", "Top1_pct", "mean_cost", "SBS_cost", "Oracle_cost", "vs_SBS_pct", "vs_Oracle_pct", "actual_regret_pct"]
                fig, ax = plt.subplots(figsize=(15, 11))
                ax.axis("off")
                ax.set_title("R108/" + arm + " | " + split + " full18", fontsize=16)
                cells = [[r[k] if k == "problem" else f"{r[k]:.5f}" for k in keys] for r in selected]
                item = ax.table(cellText=cells, colLabels=keys, loc="center", cellLoc="center")
                item.auto_set_font_size(False)
                item.set_fontsize(9)
                item.scale(1, 1.7)
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)


def main():
    manifest = backup()
    print("Original receipts, malformed CSVs and sources backed up", flush=True)
    reporting = importlib.import_module("code.V4.r108_reporting")
    goal = importlib.import_module("code.V4.r108_goal")
    before = goal.status(persist=False)
    save(HERE / "original_gate_status.json", before)
    assert not before["errors"] and all(a["missing"] == ["same_model_NSS4"] for a in before["arms"])
    normalize(reporting, goal)
    print("Frozen NSS identities verified; derived schema normalization passes original NSS gate", flush=True)
    receipt = dict(state="running", started_utc=utc(), cpu_only=True, training=False,
        inference=False, tuning=False, original_reporter_sha256=digest(Path(reporting.__file__)),
        original_reporter_modified=False, output_only_redirect=True, automatic_retry=False)
    save(HERE / "compare_started.json", receipt)
    try:
        args = reporting.parser().parse_args(["compare", "--root", str(ROOT), "--device", "cpu"])
        args.root = OutputRoot()
        result = reporting.compare(args)
        assert result["complete"] and not result["training"] and not result["inference"]
        receipt.update(state="complete", exit_status=0, completed_utc=utc())
        save(HERE / "compare_receipt.json", receipt)
    except BaseException as error:
        receipt.update(state="failed", exit_status=1, completed_utc=utc(),
            error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
        save(HERE / "compare_receipt.json", receipt)
        raise
    render_report()
    changed = [p for p, sha in manifest["protected_sha256"].items() if digest(p) != sha]
    assert not changed, changed
    save(HERE / "completion_receipt.json", dict(complete=True, completed_utc=utc(),
        canonical_gate_complete=False, canonical_gate_explanation="Original malformed report CSVs preserved",
        derived_NSS_original_gate_passed=True, saved_evidence_compare_complete=True,
        all_protected_files_unchanged=True, protected_file_count=len(manifest["protected_sha256"]),
        original_reporter_sha256=digest(Path(reporting.__file__)), reporter_modified=False,
        training=False, inference=False, tuning=False, source_logic_changed=False,
        publication_path="explicit additive recovery; canonical old watcher not rewritten"))
    print("CPU comparison/report complete; protected sources, weights, predictions and receipts unchanged", flush=True)


if __name__ == "__main__":
    main()

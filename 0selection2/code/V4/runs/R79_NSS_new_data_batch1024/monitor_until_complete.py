"""Read-only monitor; no model, data, GPU allocation, or checkpoint selection."""

import json
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parent
last = None
while True:
    records = {}
    for task in ("TSP", "CVRP"):
        run = ROOT / task
        status = json.loads((run / "status.json").read_text())
        best = json.loads((run / "best_validation.json").read_text())
        records[task] = dict(state=status["state"], update=status.get("update", status.get("updates")),
                             epoch=status.get("epoch"), loss=status.get("loss"),
                             seconds=status.get("seconds"), best_gap=best["top_1"], best_epoch=best["epoch"])
        for split in ("test", task + "LIB"):
            receipt = run / (split + "_once.json")
            if receipt.exists():
                value = json.loads(receipt.read_text())
                records[task][split] = value["state"]
                if value["state"] == "failed":
                    raise RuntimeError(f"{task} {split} failed; unique evaluation must not be blindly repeated")
    serialized = json.dumps(records, sort_keys=True)
    if serialized != last:
        print("R79_MONITOR", serialized, flush=True)
        last = serialized
    if (ROOT / "completion.json").exists():
        lock = json.loads((ROOT / "final_test_lock.json").read_text())
        for task in records:
            complete = json.loads((ROOT / task / "training_complete.json").read_text())
            if complete["updates"] != 27400 or complete["epochs"] != 50:
                raise RuntimeError("Incomplete original budget")
            for split, count in (("test", 3895), (task + "LIB", 49 if task == "TSP" else 100)):
                value = json.loads((ROOT / task / (split + "_once.json")).read_text())
                if (value["state"] != "complete" or value["rows"] != count
                        or value["checkpoint_sha256"] != lock["checkpoints"][task]["sha256"]):
                    raise RuntimeError("Incomplete or mismatched unique TEST/LIB receipt")
        print("R79_ALL_TRAIN_VALID_TEST_LIB_REPORT_COMPLETE", flush=True)
        break
    exists = subprocess.run(["tmux", "has-session", "-t", "=R79_NSS_new_data_batch1024"], capture_output=True)
    if exists.returncode:
        raise RuntimeError("R79 pipeline exited before completion; inspect train/pipeline logs")
    time.sleep(30)

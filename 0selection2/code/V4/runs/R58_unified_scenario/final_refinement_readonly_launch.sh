#!/usr/bin/env bash
set -euo pipefail
export R58_READONLY_STARTED="$(date +%s.%N)"
source /public/home/shiys/miniconda3/etc/profile.d/conda.sh
conda activate easynco
cd /public/home/shiys/0selection2
[[ "${R58_PARENT_GPU_BATON:-}" == final-refinement ]]
[[ "${SLURM_JOB_ID:-}" == 1465 && "$(hostname -s)" == gpu03 ]]
[[ -z "$(compgen -v CUDA_MPS_ || true)" ]]
export CUDA_VISIBLE_DEVICES=GPU-cb626bdb-4319-2f5e-b91f-17fcaa44a83d
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 WANDB_MODE=offline
export PYTHONPATH=/public/home/shiys/easynco_v3_bridge/exports/train_100k_20261002/python_compat:/public/home/shiys/0selection2
python -u - <<'READONLY_PY'
import os
import time
from pathlib import Path
from code.V4.r58_execution import PrivateMPS, gpu_guard
from code.V4.r58_scenario import ROOT, save_json

path = ROOT / 'final_refinement_readonly_spent.json'
if path.exists():
    raise SystemExit('One fresh readonly attempt only; preserve the existing receipt')
guard, passed, error = PrivateMPS(), False, None
try:
    gpu_guard()
    guard.check_machine()  # Read-only: never enter/start the MPS context here.
    passed = True
except Exception as exc:
    error = repr(exc)
finally:
    started, finished = float(os.environ['R58_READONLY_STARTED']), time.time()
    save_json(path, dict(status='complete', started=started, finished=finished,
        wall_seconds=finished - started, passed=passed, error=error,
        job_id=os.environ['SLURM_JOB_ID'], step_id=os.environ.get('SLURM_STEP_ID'),
        evidence=guard.guard_evidence, mps_started=False, service_changes=False))
raise SystemExit(0 if passed else 1)
READONLY_PY

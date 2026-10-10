#!/usr/bin/env bash
set -euo pipefail
export R58_FINAL_LAUNCH_STARTED="$(date +%s.%N)"
source /public/home/shiys/miniconda3/etc/profile.d/conda.sh
conda activate easynco
cd /public/home/shiys/0selection2
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 WANDB_MODE=offline
export PYTHONPATH=/public/home/shiys/easynco_v3_bridge/exports/train_100k_20261002/python_compat:/public/home/shiys/0selection2
export R58_ROOT=/public/home/shiys/0selection2/code/V4/runs/R58_unified_scenario
export WANDB_DIR="$R58_ROOT" MPLCONFIGDIR="$R58_ROOT/mplconfig"

# CPU-only after the final step exits; consume its numeric Slurm receipt once.
if [[ "${1:-run}" == settle ]]; then
    export CUDA_VISIBLE_DEVICES=''
    python - "$2" <<'SETTLE_PY'
import os
import sys
import time
from pathlib import Path
from code.V4.r58_scenario import file_hash, save_json
from code.V4.r58_timing_refinement import CAP_SECONDS, SHUTDOWN_SECONDS, positive, read_json, summarize

root = Path(os.environ['R58_ROOT'])
output = root / 'timing_refinement'
settled = root / 'final_refinement_outer_spent.json'
if settled.exists():
    raise SystemExit('Final outer accounting already settled; never charge it twice')
slurm_path = Path(sys.argv[1]).resolve()
slurm, outer = read_json(slurm_path), read_json(root / 'final_refinement_result.json')
if slurm['status'] != 'complete' or not slurm['finished'] or outer['status'] != 'complete':
    raise ValueError('Require completed numeric final Slurm and launcher receipts')
slurm_wall = positive(slurm['wall_seconds'], 'final Slurm elapsed')
state_path = output / 'run_state_controller.json'
state = read_json(state_path)
internal = sum(e['wall_seconds'] for e in state['ledger']['entries']
    if Path(e['path']).name in ('final_refinement_readonly_spent.json', 'final_refinement_setup_spent.json'))
controller_wall = positive(outer['wall_seconds'], 'final launcher wall')
wrapper_delta = max(0., controller_wall - state['wall_seconds'] - internal)
slurm_delta = max(0., slurm_wall - controller_wall)
receipt = dict(status='complete', finished=time.time(), wall_seconds=wrapper_delta + slurm_delta,
    components_seconds=dict(launcher_minus_refiner_and_already_charged_setup=wrapper_delta,
                            slurm_minus_launcher=slurm_delta),
    sources={str(p): file_hash(p) for p in (slurm_path, root / 'final_refinement_result.json', state_path)})
save_json(settled, receipt)
ledger = state['ledger']
if receipt['wall_seconds'] > 0:
    ledger['entries'].append(dict(path=str(settled), sha256=file_hash(settled), wall_seconds=receipt['wall_seconds']))
ledger['spent_seconds'] = sum(e['wall_seconds'] for e in ledger['entries'])
ledger['remaining_seconds'] = CAP_SECONDS - ledger['spent_seconds']
ledger['measurement_seconds'] = ledger['remaining_seconds'] - SHUTDOWN_SECONDS
state['cumulative_wall_seconds'] = ledger['spent_seconds'] + state['wall_seconds']
state['outer_accounting'] = dict(complete=True, receipt=str(settled), sha256=file_hash(settled))
if state['cumulative_wall_seconds'] > CAP_SECONDS or slurm['returncode'] != 0:
    state.update(status='incomplete', error='Final actual wall exceeds cap or final Slurm step failed')
save_json(output / 'run_state.json', state)
save_json(output / 'summary.json', summarize(read_json(output / 'selection.json'), state))
print('Final actual cumulative allowance seconds:', state['cumulative_wall_seconds'])
raise SystemExit(0 if state['status'] == 'complete' else 3)
SETTLE_PY
    exit "$?"
fi

[[ "${1:-run}" == run ]]
[[ "${R58_PARENT_GPU_BATON:-}" == final-refinement ]]
[[ "${SLURM_JOB_ID:-}" == 1465 && "$(hostname -s)" == gpu03 ]]
[[ -z "$(compgen -v CUDA_MPS_ || true)" ]]
git diff --quiet bc8acc4c50d0a6803b33083d663f0368a1647629 -- \
    code/V4/r58_scenario.py code/V4/r58_environments.py code/V4/r58_backends.py \
    code/V4/r58_labels.py code/V4/r58_execution.py code/V4/r58_timing_refinement.py
[[ ! -e "$R58_ROOT/final_refinement_result.json" && ! -e "$R58_ROOT/timing_refinement/run_state.json" ]]
export CUDA_VISIBLE_DEVICES=GPU-cb626bdb-4319-2f5e-b91f-17fcaa44a83d
exec > "$R58_ROOT/final_refinement_controller.log" 2>&1
python -u - <<'RUN_PY'
import os
import subprocess
import sys
import time
from pathlib import Path
from code.V4.r58_scenario import save_json
from code.V4.r58_timing_refinement import positive, read_json, spent_ledger, summarize

root = Path(os.environ['R58_ROOT'])
output = root / 'timing_refinement'
started = float(os.environ['R58_FINAL_LAUNCH_STARTED'])
code, error = 1, None
try:
    preflight = root / 'revised_preflight_state.json'
    receipts = [root / f'revised_preflight_readonly_1465_{step}_spend.json' for step in (74, 75)]
    receipts.append(root / 'revised_preflight_outer_overhead_spend.json')
    charged = []
    for path in receipts:
        receipt = read_json(path)
        if receipt['status'] != 'complete' or not receipt['finished']:
            raise ValueError(f'Incomplete numeric receipt: {path}')
        if positive(receipt['wall_seconds'], 'receipt wall', allow_zero=True) > 0:
            charged.append(path)
    spent_ledger(root / 'throughput_pilot', preflight, charged)
    subprocess.run([sys.executable, '-m', 'code.V4.r58_timing_refinement', 'prepare'],
                   env=dict(os.environ, CUDA_VISIBLE_DEVICES=''), check=True)
    subprocess.run(['bash', str(root / 'final_refinement_readonly_launch.sh')], check=True)
    readonly = root / 'final_refinement_readonly_spent.json'
    fresh = read_json(readonly)
    if fresh['passed'] is not True:
        raise ValueError('Fresh readonly isolation check did not pass')
    setup = root / 'final_refinement_setup_spent.json'
    finished = time.time()
    save_json(setup, dict(status='complete', finished=finished,
        wall_seconds=positive(finished - started - fresh['wall_seconds'], 'launcher setup'),
        policy='Launcher setup including CPU prepare, excluding separately charged fresh readonly wall'))
    charged.extend([readonly, setup])
    spent_ledger(root / 'throughput_pilot', preflight, charged)
    command = [sys.executable, '-u', '-m', 'code.V4.r58_timing_refinement', 'run',
               '--new-preflight-state', str(preflight)]
    for path in charged:
        command.extend(['--additional-spent-state', str(path)])
    code = subprocess.run(command).returncode
except Exception as exc:
    error = repr(exc)
finally:
    state_path = output / 'run_state.json'
    if state_path.exists():
        state = read_json(state_path)
        save_json(output / 'run_state_controller.json', state)
        state.update(status='outer_accounting_pending', controller_status=state['status'])
        save_json(state_path, state)
        save_json(output / 'summary.json', summarize(read_json(output / 'selection.json'), state))
    finished = time.time()
    save_json(root / 'final_refinement_result.json', dict(status='complete', started=started,
        finished=finished, wall_seconds=finished - started, returncode=code, error=error,
        job_id=os.environ['SLURM_JOB_ID'], step_id=os.environ.get('SLURM_STEP_ID'),
        outer_accounting_pending=True, full_generation_authorized=False))
raise SystemExit(code)
RUN_PY

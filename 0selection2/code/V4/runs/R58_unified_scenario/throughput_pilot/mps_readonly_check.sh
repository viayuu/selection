#!/usr/bin/env bash
set -euo pipefail
source /public/home/shiys/miniconda3/etc/profile.d/conda.sh
conda activate easynco
UUID=GPU-cb626bdb-4319-2f5e-b91f-17fcaa44a83d
test "$(hostname -s)" = gpu03
test "${SLURM_JOB_ID:-}" = 1465
test "${CUDA_VISIBLE_DEVICES:-}" = "$UUID"
printf 'READ_ONLY_CHECK %s host=%s job=%s uid=%s\n' "$(date --iso-8601=seconds)" "$(hostname -s)" "$SLURM_JOB_ID" "$(id -u)"
nvidia-smi -i "$UUID" --query-gpu=uuid,name,driver_version,compute_mode,memory.used,utilization.gpu --format=csv
nvidia-smi -i "$UUID" --query-compute-apps=pid,used_memory --format=csv
for name in nvidia-cuda-mps-control nvidia-cuda-mps-server; do
    if path=$(command -v "$name"); then
        printf 'COMMAND %s %s\n' "$name" "$path"
        readlink -f "$path"
        ls -l "$path"
        if command -v rpm >/dev/null; then
            rpm -qf "$path" || true
        elif command -v dpkg-query >/dev/null; then
            dpkg-query -S "$path" || true
        fi
    else
        printf 'COMMAND %s NOT_FOUND\n' "$name"
    fi
done
# -v is the control client's version-only option, never the daemon option -d.
if command -v nvidia-cuda-mps-control >/dev/null; then
    timeout 5s nvidia-cuda-mps-control -v || true
fi
printf 'USER_MPS_PROCESSES_BEGIN\n'
ps -u "$(id -u)" -o pid=,ppid=,args= | rg '(^|[[:space:]/])nvidia-cuda-mps-(control|server)([[:space:]]|$)' || true
printf 'USER_MPS_PROCESSES_END\n'
printf 'CUDA_MPS_PIPE_DIRECTORY=%s\nCUDA_MPS_LOG_DIRECTORY=%s\n' "${CUDA_MPS_PIPE_DIRECTORY:-<unset>}" "${CUDA_MPS_LOG_DIRECTORY:-<unset>}"
printf 'No server started; no GPU mode changed; no installation; no new pilot.\n'

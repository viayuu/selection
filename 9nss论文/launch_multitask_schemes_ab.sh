#!/usr/bin/env bash
set -euo pipefail

# 按用户要求：
# 1. 不改原始 NSS
# 2. 只训练两个复制版 schemeA / schemeB
# 3. 使用 conda 环境 easynco_zhoucl
# 4. 尽量通过 tmux 常驻，防止终端断开后任务退出

unset NPM_CONFIG_PREFIX || true

ROOT="/public/home/zhoucl/shiys/9nss论文"
SCHEME_A_DIR="${ROOT}/neural-solver-selection_schemeA"
SCHEME_B_DIR="${ROOT}/neural-solver-selection_schemeB"
MASTER_LOG="${ROOT}/multitask_train_master.log"

GPU_ID=0
MAX_WAIT_MEM_MB=2000
MAX_WAIT_UTIL=20
POLL_SECONDS=60

timestamp() {
  date '+%Y-%m-%d %H:%M:%S'
}

log() {
  echo "[$(timestamp)] $*" | tee -a "${MASTER_LOG}"
}

wait_for_gpu() {
  log "检查 GPU ${GPU_ID} 是否空闲..."
  while true; do
    local line
    line="$(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits | awk -F', ' -v gid="${GPU_ID}" '$1 == gid {print $0}')"
    local used_mem
    local util
    used_mem="$(echo "${line}" | awk -F', ' '{print $2}')"
    util="$(echo "${line}" | awk -F', ' '{print $3}')"

    if [[ -z "${used_mem}" || -z "${util}" ]]; then
      log "未能正确读取 GPU 状态，${POLL_SECONDS}s 后重试。"
      sleep "${POLL_SECONDS}"
      continue
    fi

    log "当前 GPU ${GPU_ID}: memory.used=${used_mem} MiB, utilization=${util}%"
    if [[ "${used_mem}" -le "${MAX_WAIT_MEM_MB}" && "${util}" -le "${MAX_WAIT_UTIL}" ]]; then
      log "GPU ${GPU_ID} 已达到启动阈值，开始训练。"
      break
    fi

    log "GPU 仍较忙，${POLL_SECONDS}s 后继续检查。"
    sleep "${POLL_SECONDS}"
  done
}

run_scheme() {
  local scheme_name="$1"
  local scheme_dir="$2"
  local stdout_log="${scheme_dir}/train_logs/full_train_stdout.log"

  mkdir -p "${scheme_dir}/train_logs"
  wait_for_gpu
  log "开始训练 ${scheme_name}"

  (
    cd "${scheme_dir}"
    PYTHONUNBUFFERED=1 conda run -n easynco_zhoucl \
      python run_multitask.py --config_name config_multitask.yml --gpu_id "${GPU_ID}"
  ) 2>&1 | tee "${stdout_log}"

  log "${scheme_name} 训练完成"
}

log "顺序训练开始：schemeA -> schemeB"
run_scheme "schemeA" "${SCHEME_A_DIR}"
run_scheme "schemeB" "${SCHEME_B_DIR}"
log "全部训练完成"

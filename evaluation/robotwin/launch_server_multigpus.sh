#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
ZERO_WAM_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd -P)

: "${MODEL_PATH:?Set MODEL_PATH to the released zero-wam-posttrain-robotwin model root}"

START_PORT=${START_PORT:-29556}
MASTER_PORT=${MASTER_PORT:-29661}
SAVE_ROOT=${SAVE_ROOT:-${ZERO_WAM_ROOT}/visualization}
LOG_ROOT=${LOG_ROOT:-${ZERO_WAM_ROOT}/logs}
TORCHINDUCTOR_CACHE_ROOT=${TORCHINDUCTOR_CACHE_ROOT:-/tmp/zero_wam_torchinductor}

export MODEL_PATH
export PYTHONPATH="${ZERO_WAM_ROOT}:${PYTHONPATH:-}"
mkdir -p "${LOG_ROOT}" "${SAVE_ROOT}" "${TORCHINDUCTOR_CACHE_ROOT}"
PID_FILE="${LOG_ROOT}/server_pids.txt"
: > "${PID_FILE}"
BATCH_TIME=$(date +%Y%m%d_%H%M%S)
SERVER_PIDS=()

cleanup() {
  trap - EXIT INT TERM
  if ((${#SERVER_PIDS[@]})); then
    kill "${SERVER_PIDS[@]}" 2>/dev/null || true
    wait "${SERVER_PIDS[@]}" 2>/dev/null || true
  fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

cd "${ZERO_WAM_ROOT}"
for i in {0..7}; do
  port=$((START_PORT + i))
  master_port=$((MASTER_PORT + i))
  log_file="${LOG_ROOT}/server_${i}_${BATCH_TIME}.log"
  CUDA_VISIBLE_DEVICES=${i} \
  TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_ROOT}/gpu_${i}" \
  python -m torch.distributed.run \
    --nproc_per_node 1 \
    --master_port "${master_port}" \
    -m wan_va.wan_va_server \
    --config-name robotwin \
    --save_root "${SAVE_ROOT}" \
    --port "${port}" > "${log_file}" 2>&1 &
  SERVER_PIDS+=("$!")
  echo "$!" >> "${PID_FILE}"
done

echo "Robotwin servers started; PIDs: ${PID_FILE}"
set +e
wait -n "${SERVER_PIDS[@]}"
status=$?
set -e
if ((status == 0)); then
  status=1
fi
echo "A Robotwin server exited; stopping the remaining servers." >&2
exit "${status}"

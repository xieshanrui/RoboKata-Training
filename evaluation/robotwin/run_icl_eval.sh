#!/usr/bin/env bash
set -euo pipefail

ZERO_WAM_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)
: "${ROBOTWIN_ROOT:?Set ROBOTWIN_ROOT to the Robotwin checkout}"
: "${MODEL_PATH:?Set MODEL_PATH to the released zero-wam-posttrain-robotwin model root}"

START_PORT=${START_PORT:-29556}
TARGET_TEXT_CFG=${TARGET_TEXT_CFG:--1}
ICL_CFG=${ICL_CFG:-5}
SEED=${SEED:-0}
TEST_NUM=${TEST_NUM:-100}
SAVE_ROOT=${SAVE_ROOT:-${ZERO_WAM_ROOT}/results}
SERVER_START_TIMEOUT=${SERVER_START_TIMEOUT:-900}
ICL_LATENT_ROOT=${ICL_LATENT_ROOT:-${ZERO_WAM_ROOT}/data/HumanGen/human_latents/robotwin}
SERVER_LAUNCHER_PID=""
CLIENT_LAUNCHER_PID=""

cleanup() {
  status=$?
  trap - EXIT INT TERM
  for pid in "${CLIENT_LAUNCHER_PID}" "${SERVER_LAUNCHER_PID}"; do
    if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
      kill "${pid}" 2>/dev/null || true
    fi
  done
  for pid in "${CLIENT_LAUNCHER_PID}" "${SERVER_LAUNCHER_PID}"; do
    if [[ -n "${pid}" ]]; then
      wait "${pid}" 2>/dev/null || true
    fi
  done
  exit "${status}"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

wait_for_servers() {
  local deadline=$((SECONDS + SERVER_START_TIMEOUT))
  while ((SECONDS < deadline)); do
    local ready=1
    for i in {0..7}; do
      if ! curl --fail --silent --max-time 1 \
        "http://127.0.0.1:$((START_PORT + i))/healthz" >/dev/null; then
        ready=0
        break
      fi
    done
    if ((ready)); then
      return 0
    fi
    if ! kill -0 "${SERVER_LAUNCHER_PID}" 2>/dev/null; then
      echo "Robotwin server launcher exited before all servers became ready." >&2
      return 1
    fi
    sleep 5
  done
  echo "Timed out waiting for Robotwin servers after ${SERVER_START_TIMEOUT}s." >&2
  return 1
}

export MODEL_PATH ROBOTWIN_ROOT ICL_LATENT_ROOT
export PYTHONPATH="${ZERO_WAM_ROOT}:${PYTHONPATH:-}"
cd "${ZERO_WAM_ROOT}"

START_PORT="${START_PORT}" SAVE_ROOT="${ZERO_WAM_ROOT}/visualization" \
bash evaluation/robotwin/launch_server_multigpus.sh &
SERVER_LAUNCHER_PID=$!

wait_for_servers
echo "All eight Robotwin servers are ready."

START_PORT="${START_PORT}" \
TARGET_TEXT_CFG="${TARGET_TEXT_CFG}" \
ICL_CFG="${ICL_CFG}" \
SEED="${SEED}" \
TEST_NUM="${TEST_NUM}" \
bash evaluation/robotwin/launch_client_multigpus.sh "${SAVE_ROOT}" &
CLIENT_LAUNCHER_PID=$!

wait "${CLIENT_LAUNCHER_PID}"

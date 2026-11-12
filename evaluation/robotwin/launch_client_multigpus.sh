#!/usr/bin/env bash
set -euo pipefail

export LD_LIBRARY_PATH="/usr/lib64:/usr/lib:${LD_LIBRARY_PATH:-}"
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
ZERO_WAM_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd -P)

: "${ROBOTWIN_ROOT:?Set ROBOTWIN_ROOT to the Robotwin checkout}"

SAVE_ROOT=${1:-${ZERO_WAM_ROOT}/results}
SEED=${SEED:-0}
TEST_NUM=${TEST_NUM:-100}
START_PORT=${START_PORT:-29556}
TARGET_TEXT_CFG=${TARGET_TEXT_CFG:--1}
ICL_CFG=${ICL_CFG:-5}
ICL_SEED=${ICL_SEED:-${SEED}}
ICL_HUMAN_VIDEO_MAP=${ICL_HUMAN_VIDEO_MAP:-${SCRIPT_DIR}/robotwin_icl_human_videos.py}
ICL_LATENT_ROOT=${ICL_LATENT_ROOT:-${ZERO_WAM_ROOT}/data/HumanGen/human_latents/robotwin}
LOG_ROOT=${LOG_ROOT:-${ZERO_WAM_ROOT}/logs}

if [[ "${SAVE_ROOT}" != /* ]]; then
  SAVE_ROOT="${ZERO_WAM_ROOT}/${SAVE_ROOT#./}"
fi

# Seven unique unseen tasks. Keep stack_blocks_three on GPU 0. The second
# place_empty_cup process is an additional repeat and writes under a separate
# root to avoid result races.
TASKS=(
  stack_blocks_three
  place_object_scale
  stamp_seal
  open_microwave
  move_stapler_pad
  place_bread_basket
  place_empty_cup
  place_empty_cup
)

export PYTHONPATH="${ZERO_WAM_ROOT}:${ROBOTWIN_ROOT}:${PYTHONPATH:-}"
mkdir -p "${LOG_ROOT}"
PID_FILE="${LOG_ROOT}/client_pids.txt"
: > "${PID_FILE}"
BATCH_TIME=$(date +%Y%m%d_%H%M%S)
CLIENT_PIDS=()

cleanup() {
  trap - EXIT INT TERM
  if ((${#CLIENT_PIDS[@]})); then
    kill "${CLIENT_PIDS[@]}" 2>/dev/null || true
    wait "${CLIENT_PIDS[@]}" 2>/dev/null || true
  fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

cd "${ROBOTWIN_ROOT}"
for i in "${!TASKS[@]}"; do
  task_name=${TASKS[$i]}
  port=$((START_PORT + i))
  task_save_root=${SAVE_ROOT}
  if [[ "${i}" == 7 ]]; then
    task_save_root="${SAVE_ROOT}/place_empty_cup_repeat"
  fi
  log_file="${LOG_ROOT}/client_${i}_${task_name}_${BATCH_TIME}.log"
  CUDA_VISIBLE_DEVICES=${i} \
  PYTHONWARNINGS=ignore::UserWarning \
  python -m evaluation.robotwin.eval_policy_client_openpi \
    --config "${ROBOTWIN_ROOT}/policy/ACT/deploy_policy.yml" \
    --port "${port}" \
    --save_root "${task_save_root}" \
    --video_guidance_scale "${TARGET_TEXT_CFG}" \
    --action_guidance_scale 1 \
    --icl_guidance_scale "${ICL_CFG}" \
    --icl_human_video_map "${ICL_HUMAN_VIDEO_MAP}" \
    --icl_latent_root "${ICL_LATENT_ROOT}" \
    --icl_seed "${ICL_SEED}" \
    --test_num "${TEST_NUM}" \
    --overrides \
    --task_name "${task_name}" \
    --task_config demo_clean \
    --train_config_name 0 \
    --model_name 0 \
    --ckpt_setting 0 \
    --seed "${SEED}" \
    --policy_name ACT > "${log_file}" 2>&1 &
  CLIENT_PIDS+=("$!")
  echo "$!" >> "${PID_FILE}"
done

echo "Robotwin clients started; PIDs: ${PID_FILE}"
status=0
for pid in "${CLIENT_PIDS[@]}"; do
  wait "${pid}" || status=$?
done
exit "${status}"

#!/usr/bin/env bash
set -euo pipefail

export LD_LIBRARY_PATH="/usr/lib64:/usr/lib:${LD_LIBRARY_PATH:-}"
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
ZERO_WAM_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd -P)

: "${ROBOTWIN_ROOT:?Set ROBOTWIN_ROOT to the Robotwin checkout}"

save_root=${1:-${ZERO_WAM_ROOT}/results}
task_name=${2:-stack_blocks_three}
seed=${SEED:-0}
PORT=${PORT:-29056}
TEST_NUM=${TEST_NUM:-100}
ICL_LATENT_ROOT=${ICL_LATENT_ROOT:-${ZERO_WAM_ROOT}/data/HumanGen/human_latents/robotwin}

if [[ "${save_root}" != /* ]]; then
    save_root="${ZERO_WAM_ROOT}/${save_root#./}"
fi

export PYTHONPATH="${ZERO_WAM_ROOT}:${ROBOTWIN_ROOT}:${PYTHONPATH:-}"
cd "${ROBOTWIN_ROOT}"

exec env PYTHONWARNINGS=ignore::UserWarning XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
python -m evaluation.robotwin.eval_policy_client_openpi \
    --config "${ROBOTWIN_ROOT}/policy/ACT/deploy_policy.yml" \
    --port "${PORT}" \
    --save_root "${save_root}" \
    --video_guidance_scale "${TARGET_TEXT_CFG:--1}" \
    --action_guidance_scale 1 \
    --icl_guidance_scale "${ICL_CFG:-5}" \
    --icl_human_video_map "${ICL_HUMAN_VIDEO_MAP:-${SCRIPT_DIR}/robotwin_icl_human_videos.py}" \
    --icl_latent_root "${ICL_LATENT_ROOT}" \
    --icl_seed "${ICL_SEED:-${seed}}" \
    --test_num "${TEST_NUM}" \
    --overrides \
    --task_name "${task_name}" \
    --task_config demo_clean \
    --train_config_name 0 \
    --model_name 0 \
    --ckpt_setting 0 \
    --seed "${seed}" \
    --policy_name ACT

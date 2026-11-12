#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
ZERO_WAM_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd -P)

: "${MODEL_PATH:?Set MODEL_PATH to the released zero-wam-posttrain-robotwin model root}"

PORT=${PORT:-29056}
MASTER_PORT=${MASTER_PORT:-29061}
SAVE_ROOT=${SAVE_ROOT:-${ZERO_WAM_ROOT}/visualization}

export MODEL_PATH
export PYTHONPATH="${ZERO_WAM_ROOT}:${PYTHONPATH:-}"
mkdir -p "${SAVE_ROOT}"
cd "${ZERO_WAM_ROOT}"

exec python -m torch.distributed.run \
    --nproc_per_node 1 \
    --master_port "${MASTER_PORT}" \
    -m wan_va.wan_va_server \
    --config-name robotwin \
    --port "${PORT}" \
    --save_root "${SAVE_ROOT}"

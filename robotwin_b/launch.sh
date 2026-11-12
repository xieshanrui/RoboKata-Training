#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${ZERO_WAM_UPSTREAM:?Set ZERO_WAM_UPSTREAM to the upstream Zero-WAM checkout}"
export PYTHONPATH="$root/backend:$root/roboicl:$ZERO_WAM_UPSTREAM"
export HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"

exec "${PYTHON_BIN:-python}" -m torch.distributed.run --standalone --nproc-per-node=4 -- \
    "$root/robotwin_b/train_distributed.py" "$@"

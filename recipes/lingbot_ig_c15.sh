#!/usr/bin/env bash
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
need CONFIG RUN CUDA_VISIBLE_DEVICES
run mkdir -p "$RUN"
run "$PYTHON" -m torch.distributed.run --standalone --nproc-per-node=8 "$ROOT/models/lingbot/lingbot_ig_text/train.py" --config "$CONFIG" "$@"

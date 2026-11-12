#!/usr/bin/env bash
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
need CONFIG RUN CUDA_VISIBLE_DEVICES
export PYTHONPATH="$ROOT/models/openpi/src:$ROOT/models/openpi/packages/openpi-client/src:$PYTHONPATH"
run "$PYTHON" -m torch.distributed.run --standalone --nproc-per-node=4 "$ROOT/models/openpi/examples/robotwin_text/train.py" --config "$CONFIG" --output "$RUN" --steps 20000 "$@"

#!/usr/bin/env bash
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
need CONFIG RUN CUDA_VISIBLE_DEVICES
export PYTHONPATH="$ROOT/models/openpi/src:$ROOT/models/openpi/examples/segmented_demo:$ROOT/models/openpi/packages/openpi-client/src:$PYTHONPATH"
run "$PYTHON" -m torch.distributed.run --standalone --nproc-per-node=2 "$ROOT/models/openpi/examples/segmented_demo/train.py" --config "$CONFIG" --output "$RUN" --steps 20000 "$@"

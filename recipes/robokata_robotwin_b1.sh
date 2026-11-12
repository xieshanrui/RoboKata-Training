#!/usr/bin/env bash
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
need DATA_RT MODEL DINO_BASE EMPTY_TEXT FEATURE_CACHE RUN CUDA_VISIBLE_DEVICES
export ZERO_WAM_UPSTREAM="$ROOT/models/zerowam"
export PYTHONPATH="$ROOT/models/robokata/backend:$ROOT/models/robokata/roboicl:$ZERO_WAM_UPSTREAM"
run "$PYTHON" -m torch.distributed.run --standalone --nproc-per-node=4 "$ROOT/models/robokata/robotwin_b/train_distributed.py" --data "$DATA_RT" --model "$MODEL" --encoder "$DINO_BASE" --empty-text "$EMPTY_TEXT" --feature-cache "$FEATURE_CACHE" --run "$RUN" --max-robot-frames 16 --epochs 200 --no-reserve "$@"

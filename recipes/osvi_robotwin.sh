#!/usr/bin/env bash
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
need DATA_RT PREPARED_RT_ORIGINAL RUN CUDA_VISIBLE_DEVICES
run "$PYTHON" -m osvi_robotwin.train --dataset robotwin --data "$DATA_RT" --prepared "$PREPARED_RT_ORIGINAL" --official "$ROOT/third_party/osvi_wm" --run "$RUN" --updates 72000 --batch-size 64 --workers 4 --seed 42 "$@"

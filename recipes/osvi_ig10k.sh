#!/usr/bin/env bash
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
need PREPARED_IG RUN CUDA_VISIBLE_DEVICES
run "$PYTHON" -m osvi_robotwin.train --dataset ig10k --data "$(dirname "$PREPARED_IG")" --prepared "$PREPARED_IG" --official "$ROOT/third_party/osvi_wm" --run "$RUN" --updates 72000 --batch-size 64 --workers 4 --seed 42 "$@"

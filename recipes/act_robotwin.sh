#!/usr/bin/env bash
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
need DATA_RT PREPARED_RT FEATURE_CACHE RUN CUDA_VISIBLE_DEVICES
run "$PYTHON" -m act_robotwin.train --data "$DATA_RT" --prepared "$PREPARED_RT" --ig-root "$IG_ROOT" --feature-cache "$FEATURE_CACHE" --run "$RUN" --batch-size 128 --epochs 10 --warmup-epochs 1 --lr 1e-4 --precision bf16 --seed 1 --workers 4 "$@"

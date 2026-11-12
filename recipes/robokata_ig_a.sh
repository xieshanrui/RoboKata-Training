#!/usr/bin/env bash
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
need IG_CACHE MANIFEST_IG STATS_IG HUMAN_ROOT SIM_ROOT MODEL EMPTY_TEXT DEMO_FEATURES RUN CUDA_VISIBLE_DEVICES
export PYTHONPATH="$ROOT/models/robokata/backend"
run "$PYTHON" -m torch.distributed.run --standalone --nproc-per-node=8 "$ROOT/models/robokata/roboicl/train_entry.py" --dataset ig10k --protocol c15 --work "$IG_CACHE" --manifest "$MANIFEST_IG" --stats "$STATS_IG" --human-root "$HUMAN_ROOT" --sim-root "$SIM_ROOT" --task-config-root "$IG_ROOT/examples/baselines/lerobot_dataset" --model "$MODEL" --empty-text "$EMPTY_TEXT" --demo-features "$DEMO_FEATURES" --run-dir "$RUN" --epochs 60 --warmup-steps 100 --save-epoch-freq 10 --activation-checkpointing --adapter-rank 106 --batch-per-gpu 8 "$@"

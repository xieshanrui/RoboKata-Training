#!/usr/bin/env bash
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
need MODEL EMPTY_TEXT HUMAN_ROOT SIM_ROOT IG_CACHE MANIFEST_IG STATS_IG RUN CUDA_VISIBLE_DEVICES
cd "$IG_ROOT"
export ZERO_WAM_MODEL="$MODEL" ZERO_WAM_EMPTY_TEXT="$EMPTY_TEXT" ZERO_WAM_CACHE="$IG_CACHE"
run "$PYTHON" -m torch.distributed.run --standalone --nproc-per-node=4 -m examples.baselines.zero_wam.train_zero_wam_imitator --protocol c15 --manifest "$MANIFEST_IG" --run-dir "$RUN" --sampling-mode episode --action-per-frame 16 --robot-frame-stride 4 --image-size 224 --robot-width 288 --human-height 320 --human-width 480 --action-representation relative_episode --action-channel-offset 14 --normalization-path "$STATS_IG" --action-history-mode zero_history_supervised_v1 --train-scope joint --adapter-rank 106 --enable-mcp --activation-checkpointing --batch-size 1 --effective-batch-size 4 --total-epochs 60 --learning-rate 1e-4 --min-learning-rate 1e-6 --lr-schedule cosine --warmup-steps 100 --lr-decay-start-step 100 --lr-decay-steps 44900 --save-epoch-freq 10 --save-step-freq 1000 --save-seconds 1200 "$@"

# Original Robot-ICL baseline

This directory contains the original Robot-ICL training, data preparation
and recorded-episode inference source with the unmodified The-Imitator-Game
backend at commit `3c3404a`. It uses the external base weights, verified RoboICL
cache, empty-text embedding and Python environment described in the parent
README. The separate launcher supplies the original provenance hashes so
the upstream task descriptions do not need to be bundled.

## Train

```bash
BASELINE="$(pwd -P)/baseline"
PY=python
export PYTHONPATH="$BASELINE/backend"

CUDA_VISIBLE_DEVICES="${TRAIN_GPUS:-0,1,2,3}" "$PY" -m torch.distributed.run --standalone --nproc-per-node=4 \
  "$BASELINE/roboicl/train_entry.py" --work "$WORK" --model "$MODEL" \
  --empty-text "$EMPTY_TEXT" --run-dir "$BASELINE_RUN" \
  --epochs 20 --batch-per-gpu 16 --warmup-steps 10 --learning-rate 1e-4
```

This reproduces the baseline recipe with global batch 64, 6,600 updates,
and one archived checkpoint per epoch. To continue an existing run, supply
the original run directory and `--resume-from "$BASELINE_RUN/checkpoints/latest.pt"`.
For a fresh cache, run `roboicl/prepare.py` through VAE cache verification.

## Infer

```bash
CUDA_VISIBLE_DEVICES="${INFER_GPU:-0}" "$PY" "$BASELINE/roboicl/infer_checkpoint.py" \
  --work "$WORK" --model "$MODEL" --empty-text "$EMPTY_TEXT" \
  --checkpoint "$BASELINE_CHECKPOINT" --pair-index 0 --output "$OUTPUT_JSON"
```

Inference generates the first 16 actions for a held-out recording.

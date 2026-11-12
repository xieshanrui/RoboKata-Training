# RoboKata · Robot-ICL + A and B1

This source snapshot contains the RoboICL preparation, DINOv2 caches, training
and recorded-episode inference entry points for Version A and A+B1. The
Zero-WAM backend was adapted from The-Imitator-Game commit `3c3404a` and
Zero-WAM commit `08e2c4a`.

Use a Python 3.10 CUDA environment with PyTorch 2.9 and the dependencies listed in `requirements-server.txt`. Activate that environment and set `PYTHONPATH` to this repository's `backend/` directory. Inputs and outputs go outside the source directory. Supply the verified RoboICL release, base Zero-WAM weights, the empty text embedding and DINOv2 weights separately.

```bash
ROOT="$(pwd -P)"
PY=python
export PYTHONPATH="$ROOT/backend"
# Set these to existing absolute paths, outside RoboKata.
# DATA, WORK, MODEL, EMPTY_TEXT, DINO, DEMO_CACHE, RUN
```

## Prepare

Reuse the original verified `WORK` and `DEMO_CACHE` when continuing an existing run. To prepare a new work directory, run:

```bash
"$PY" "$ROOT/roboicl/prepare.py" plan --root "$DATA" --work "$WORK"
"$PY" "$ROOT/roboicl/prepare.py" stats --work "$WORK"
"$PY" "$ROOT/roboicl/prepare.py" cache --work "$WORK" --model "$MODEL"
"$PY" "$ROOT/roboicl/prepare.py" verify --work "$WORK" --model "$MODEL"
"$PY" "$ROOT/roboicl/demo_features.py" --work "$WORK" --encoder "$DINO" \
  --output "$DEMO_CACHE" --device cuda:0 --bf16
```

For parallel extraction, pass `--shards N --shard I` to each cache worker. VAE workers write independent files and `verify` follows the last worker. For the DINOv2 cache, merge by running `demo_features.py` with `--shards N` and no `--shard` after all parts finish. Use the same encoder and precision for all parts and the merge.

## Train

```bash
CUDA_VISIBLE_DEVICES="${TRAIN_GPUS:-0,1,2,3}" "$PY" -m torch.distributed.run --standalone --nproc-per-node=4 \
  "$ROOT/roboicl/train_entry.py" --work "$WORK" --model "$MODEL" \
  --empty-text "$EMPTY_TEXT" --run-dir "$RUN" --demo-features "$DEMO_CACHE" \
  --epochs 20 --batch-per-gpu 16 --warmup-steps 0 --learning-rate 5e-5 \
  --cuda-cache-gib 120 --initialize-from "$BASELINE_CHECKPOINT"
```

The launcher accepts the same arguments as `roboicl/train.py` plus `--cuda-cache-gib`. The CUDA cache flag creates a reusable allocator pool and suppresses periodic cache purges. Set it only on GPUs allocated to the run. The default is zero. For an existing Version A run, use `--resume-from "$RUN/checkpoints/latest.pt"` in place of `--initialize-from` and keep the original seed, rank count, batch, schedule and cache. `--initialize-from` loads a compatible Robot-ICL baseline adapter and starts a new optimizer. Each completed epoch gets a separate checkpoint and `latest.pt` is a rolling restart point.

## Infer

```bash
CUDA_VISIBLE_DEVICES="${INFER_GPU:-0}" "$PY" "$ROOT/roboicl/infer_checkpoint.py" \
  --work "$WORK" --model "$MODEL" --empty-text "$EMPTY_TEXT" \
  --checkpoint "$RUN/checkpoints/epoch_001.pt" --demo-features "$DEMO_CACHE" \
  --pair-index 0 --output "$OUTPUT_JSON"
```

The inference entry point generates the first 16 actions for a held-out recorded pair. Add `--disable-demo-attention` to compare A-off with the same checkpoint, pair and seed. This is a recorded-episode inference entry point.

`roboicl/publish_epochs.py` is a separate, opt-in publisher for A and A+B1
epoch adapters. Supply `--run`, a dedicated `--state-dir`, and an existing
public `--repo` belonging to the signed-in Hugging Face user. Use a separate
repository and state directory for each version.

## Original baseline

See [baseline/README.md](baseline/README.md) for the first Robot-ICL run and its separate, unmodified backend.

## Version A + B1

B1 replaces latent MCP denoising with prediction of frozen DINOv2 features.
Use the same Version A demo cache and extract a second cache from the **training
queries**. Each target uses the external camera frame at the causal VAE
latent's nominal anchor, pooled to a 2×2 grid. Anchors beyond the recorded
RGB tail are masked. The original four horizons, in chunks, are 1, 3, 5 and 7.

```bash
"$PY" "$ROOT/roboicl/semantic_features.py" --work "$WORK" --encoder "$DINO" \
  --output "$SEMANTIC_CACHE" --device cuda:0 --bf16
```

Use `--shards N --shard I` for separate workers, followed by `--shards N`
without `--shard` to merge. Rerun the cache command with `--verify` to check
the merged cache, including the encoder identity, anchor mask and finite values.
Add `--semantic-features "$SEMANTIC_CACHE"` and
`--semantic-weight 0.5` to the training command and choose a new run directory.
`--initialize-from "$A_CHECKPOINT"` loads A's compatible adapter weights and
starts a new optimizer. For a matched A-only control, start from the same A
checkpoint with the same recipe and omit `--semantic-features`.

B1's heads supervise training only. Inference accepts an A+B1 checkpoint with
the same command and demo cache as Version A. A+B1 checkpoints require their
semantic target cache and loss weight when resuming training.

## Version A + B1 + Foresight Indexing

Start a new run from an A+B1 checkpoint. Use the same demo and semantic
caches as A+B1 and enable indexing in the existing training command.

```bash
CUDA_VISIBLE_DEVICES="${TRAIN_GPUS:-0,1,2,3}" "$PY" -m torch.distributed.run --standalone --nproc-per-node=4 \
  "$ROOT/roboicl/train_entry.py" --work "$WORK" --model "$MODEL" \
  --empty-text "$EMPTY_TEXT" --run-dir "$INDEX_RUN" --demo-features "$DEMO_CACHE" \
  --semantic-features "$SEMANTIC_CACHE" --semantic-weight 0.5 \
  --foresight-indexing --key-weight 0.5 --initialize-from "$B1_CHECKPOINT" \
  --epochs 20 --batch-per-gpu 16 --warmup-steps 0 --learning-rate 5e-5
```

The key heads reuse the B1 targets at offset zero and mask frames beyond
the recorded RGB tail. This variant disables noise augmentation on the video
conditioning stream so keys are trained on clean latents. The B1 future heads
still read the noisy video stream. Each demo attention layer starts with zero
index bias. Training logs the weighted key loss as `foresight_key_loss`.
Resume an indexed run with `--resume-from` in place of `--initialize-from`,
using the same caches and indexing settings.

For inference, use the existing `infer_checkpoint.py` command with an indexed
checkpoint and `--demo-features "$DEMO_CACHE"`. The checkpoint enables indexing
automatically. The final video pass stores a frame index for the action passes.
The inference command uses only the demo feature cache. Use a dedicated model
repository and state directory when publishing an indexed run.

## RoboICL dual-camera training from Zero-WAM pretrain

Use a separate preparation directory. Select `--dual-view` on `prepare.py plan`
and use the same split seed and validation fraction as the single-view
experiment. The dual plan records raw camera order `[0, 1]`. Both demo and
query VAE inputs are synchronized `224×448` RGB, with camera 0 on the left.
Demo DINO tokens are ordered by frame, camera, and grid cell (64 tokens).
The two query views are pooled separately for the semantic and key losses.
Indexing compares corresponding camera cells and assigns one bias per demo frame.

Use the released [Zero-WAM pretrain](https://huggingface.co/Robbyant-Research/zero-wam-pretrain/tree/7040c4195df216c900334ef62d5fdcf05c0601aa)
revision `7040c4195df216c900334ef62d5fdcf05c0601aa`, the matching empty-text
embedding, and frozen DINOv2 ViT-B/14 weights. Keep `WORK`, `RUN`, and the
assets outside the repository. Build and verify the three caches before training.

```bash
"$PY" "$ROOT/roboicl/prepare.py" plan --root "$DATA" --work "$WORK" --dual-view
"$PY" "$ROOT/roboicl/prepare.py" stats --work "$WORK"
"$PY" "$ROOT/roboicl/prepare.py" cache --work "$WORK" --model "$MODEL"
"$PY" "$ROOT/roboicl/prepare.py" verify --work "$WORK" --model "$MODEL"
"$PY" "$ROOT/roboicl/demo_features.py" --work "$WORK" --encoder "$DINO" \
  --output "$WORK/demo-dino.pt" --device cuda:0 --bf16
"$PY" "$ROOT/roboicl/semantic_features.py" --work "$WORK" --encoder "$DINO" \
  --output "$WORK/query-dino.pt" --device cuda:0 --bf16
"$PY" "$ROOT/roboicl/semantic_features.py" --work "$WORK" --encoder "$DINO" \
  --output "$WORK/query-dino.pt" --verify
```

The recorded experiment jointly trains all modules from the base weights on four
GPUs with 16 episodes per GPU, effective batch 64, and 20 epochs. The verified
split yields 330 optimizer updates per epoch for this topology. Keep the same
sampling configuration and caches when resuming from `checkpoints/latest.pt`.

```bash
CUDA_VISIBLE_DEVICES="${TRAIN_GPUS:-0,1,2,3}" "$PY" -m torch.distributed.run --standalone --nproc-per-node=4 \
  "$ROOT/roboicl/train_entry.py" --work "$WORK" --model "$MODEL" \
  --empty-text "$EMPTY_TEXT" --run-dir "$RUN" --demo-features "$WORK/demo-dino.pt" \
  --semantic-features "$WORK/query-dino.pt" --semantic-weight 0.5 \
  --foresight-indexing --key-weight 0.5 --batch-per-gpu 16 \
  --epochs 20 --save-epoch-freq 1 --warmup-steps 0 --learning-rate 5e-5
```

For recorded inference, use the matching verified work directory and demo cache.
The query semantic cache is used only during training. A local training
checkpoint can be passed directly. For a published adapter, keep the epoch's
`adapter.json` beside its `adapter.safetensors` and supply the repository-level
`adapter_config.json`. Published adapters use the original cache identities,
including the verified latent receipt and demo cache hashes.

```bash
"$PY" "$ROOT/roboicl/infer_checkpoint.py" --work "$WORK" --model "$MODEL" \
  --empty-text "$EMPTY_TEXT" --demo-features "$WORK/demo-dino.pt" \
  --checkpoint "$HF_EXPORT/epoch-001/adapter.safetensors" \
  --adapter-config "$HF_EXPORT/adapter_config.json" --pair-index 0 --output "$OUTPUT_JSON"
```

For the experiment weights, see [RoboKata-1110](https://huggingface.co/gulucaptain/RoboKata-1110).
To publish an indexed run, give `publish_epochs.py` the training run, a separate
state directory, and the destination model repository. It exports the adapter
and inference configuration and checks the remote file size and SHA-256.

```bash
"$PY" "$ROOT/roboicl/publish_epochs.py" --run "$RUN" --state-dir "$UPLOAD_STATE" \
  --repo gulucaptain/RoboKata-1110
```

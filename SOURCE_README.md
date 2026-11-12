# RoboKata · Robot-ICL + A and B1

This source snapshot contains the RoboICL preparation, DINOv2 caches, training
and recorded-episode inference entry points for Version A and A+B1. The
Zero-WAM backend was adapted from The-Imitator-Game commit `3c3404a` and
Zero-WAM commit `08e2c4a`.

The separate [ACT/DINOv2 RoboTwin baseline](act_robotwin/README.md) prepares
the existing HumanGen 43-task training pairs and six held-out evaluation tasks.
Its finite runtime check is complete; formal training has not started.

[LingBot-VA IG text](lingbot_ig_text/README.md) adapts the official full-parameter
trainer to C15 human descriptions and the existing Query cache. Its bounded
probe is separate from the RoboICL Demo-conditioned LingBot checkpoint.

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


## RoboICL Query wrist extension

The maintained extension keeps the Version A Demo branch head-only and adds the Query wrist camera. Each camera has its own causal VAE state; head and wrist latents are concatenated along spatial width (14×36 instead of 14×18), never along time. Offline extraction, initial observation and executed-action feedback use the same ordering. The original head-only recipe remains available with `--query-views head`.

The portable input is `DATA/train3000/{bundle.json,manifest.json,pairing.json,records/}`. Preparation verifies index hashes against the bundle and verifies each selected recording before encoding. A website upload receipt is not required. The family split and non-self directed pairs are unchanged. `--pair-limit` explicitly marks a small same-length pilot subset; omit it for the complete training split.

```bash
export PYTHONPATH="$PWD/backend"
python roboicl/prepare.py plan --root "$DATA" --work "$WORK" --query-views head_wrist
python roboicl/prepare.py stats --work "$WORK" --initialization-stats "$ORIGINAL_STATS"
python roboicl/prepare.py cache --work "$WORK" --model "$MODEL"
python roboicl/prepare.py verify --work "$WORK" --model "$MODEL"
python roboicl/demo_features.py --work "$WORK" --encoder "$DINO" --output "$DEMO_CACHE" --device cuda:0 --bf16
```

For initialization from a published Version A adapter, use `--initialize-adapter adapter.safetensors --adapter-metadata adapter.json --initialization-stats original_stats.json` in place of `--initialize-from`. Prepare `stats` with the same `--initialization-stats` file before cache verification; omit this option when training a fresh baseline with newly fitted statistics. Tensor names/shapes, published checksum and action/state normalization must match. The optimizer starts fresh. The new Query cache has a separate identity; old cache receipts and optimizer-resume files are not relabeled. A full-checkpoint `--resume-from` still requires matching architecture and cache.

Training uses actual `--batch-per-gpu` without gradient accumulation, including batch1. Equal-length batches retain the original complete-episode sampling; incomplete buckets are dropped and shuffled each epoch. Check actual memory before selecting the production batch. `--max-steps 10 --save-epoch-freq 1000` runs a bounded pipeline check and saves `checkpoints/latest.pt`; it does not establish a convergence result. The closed-loop worker is maintained in the parent project's `zerowam/adapters/roboicl_lora_worker.py` and supports these full training checkpoints.

`--adapter-rank` selects the backend's native LoRA rank; omitting it preserves rank106. For a rank32 capacity check, use `--adapter-rank 32` with fresh adapters. Published rank106 adapters and optimizer states retain their shape checks and cannot be resumed as rank32. Inference reads the saved rank from the checkpoint.

## Version A on IG-10K

Use the same `PYTHONPATH="$ROOT/backend"` and `roboicl/train_entry.py` as RoboICL. IG supplies protocol JSON resources, the verified manifest, VAE caches and train-only relative-joint statistics. Existing verified caches can be reused. `roboicl/ig10k_features.py --manifest ... --encoder ... --output ...` prepares Version A tokens from human RGB.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 "$PY" -m torch.distributed.run --standalone --nproc-per-node=8 \
  "$ROOT/roboicl/train_entry.py" --dataset ig10k --protocol c15 \
  --work "$CACHE" --manifest "$MANIFEST" --stats "$STATS" \
  --human-root "$HUMAN_ROOT" --sim-root "$SIM_ROOT" --task-config-root "$IG_TASK_CONFIG" \
  --model "$MODEL" --empty-text "$EMPTY_TEXT" --demo-features "$DEMO_CACHE" \
  --run-dir "$RUN" --epochs 60 --batch-per-gpu 8 --activation-checkpointing
```

IG follows the official complete-episode recipe: native LoRA rank106, complete Query and human Demo, learning rate `1e-4` with cosine decay to `1e-6`, 100 warmup updates for C15/C30/C45 (10 for Scratch), and epoch checkpoints every 10 epochs. The command above adapts actual batch size and activation checkpointing to eight GPUs; it uses global batch64 without gradient accumulation. `--epochs 60` specifies the C15 data-exposure budget; optimizer update counts depend on batch size.

`IG_TASK_CONFIG` is IG's `examples/baselines/lerobot_dataset` directory. For an existing cache, retain its original resource paths. IG uses relative dual-arm joints, the official PyTorch DistributedSampler order and a random same-task human Demo on each training read. It does not bucket by length or discard incomplete batches. Mixed-length batches are padded; attention and losses exclude temporal padding, with each episode retaining its original loss normalization. Native distributed sampler repeats are recorded when needed. C15 on eight GPUs at batch8 covers all 3,000 episodes in 47 updates per epoch; the last global batch has 56 episodes. Batch1 on four GPUs has 750 updates per epoch.

Both datasets share Version A, FSDP, activation checkpointing, losses, optimizer and checkpoint/resume code. `--max-steps 20` limits a runtime check without shortening the full learning-rate schedule. Use a new run directory after changing data/sampling; the old standalone IG run cannot be resumed as this sampler. `--initialize-from` accepts a compatible IG A checkpoint for target-task fine-tuning, starting a new optimizer. RoboICL world-hand weights are rejected. Omit `--demo-features` for the baseline architecture. Version A adds its DINO attention modules; physical batch size and recomputation are hardware adaptations. Resume requires the same episode sampler and topology; old exact-length runs remain separate.

`--query-latent-limit 32` keeps short Queries intact and rotates contiguous windows over longer episodes. Window starts visit a seeded permutation of every valid start before repeating. Full causal VAE caches, the original episode coordinate origin, 16 actions per latent and the complete human Demo are retained. At a mid-episode window boundary, the first action group predates the retained observation: it is zeroed and excluded from loss, and conditioning uses the measured state at that observation. MCP future shifts and validity masks operate only inside the retained window. Set the limit to zero for full episodes; resume requires the same limit. Changing the Query window requires a new run; changing the LoRA rank also changes adapter and optimizer shapes.

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

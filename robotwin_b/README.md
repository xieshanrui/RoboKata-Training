# Zero-WAM Version B on HumanGen RoboTwin

This entry point uses the verified [HumanGen RoboTwin release](https://huggingface.co/datasets/xieshanrui/HumanGen-RoboTwin-43Train-7Test).
It reads 2,149 training pairs across 43 tasks from `manifests/train.jsonl`.
The seven held-out tasks are reserved for evaluation.

The generated human demo supplies the native ICL video latent and A's frozen
DINOv2 features. Both text streams use the empty embedding. Robot observations
concatenate high, left wrist and right wrist latents along width. All three
views must share the source frame IDs. The action adapter converts RoboTwin's
16 raw action channels to the model's 30 channels. B1 pools only high-view
positions and predicts DINOv2 features at horizons of 1, 3, 5 and 7 chunks.

## Environment and data

Use the Python 3.10 environment specified in the repository requirements, with
CUDA and FFmpeg available. Set these paths before running the examples.

```bash
export ZERO_WAM_UPSTREAM=/path/to/upstream/Zero-WAM
export PYTHON_BIN=/path/to/environment/bin/python
export PYTHONPATH="$PWD/backend:$PWD/roboicl:$ZERO_WAM_UPSTREAM"
DATA=/path/to/verified/HumanGen-RoboTwin-43Train-7Test
MODEL=/path/to/zero-wam-pretrain
DINO=/path/to/dinov2-base
EMPTY="$ZERO_WAM_UPSTREAM/wan_va/assets/empty_text_emb.pt"
CACHE=/path/to/robotwin-dino-cache
RUN=/path/to/new/robotwin-run
```

## DINOv2 cache

The cache has 854 deduplicated demo entries and 2,149 high-view query entries.
The verification pass checks tensor shapes, finite values, and source anchors.

```bash
cache_args=(--data "$DATA" --encoder "$DINO" --cache "$CACHE")
"$PYTHON_BIN" robotwin_b/feature_cache.py "${cache_args[@]}" --init
pids=()
for gpu in {0..7}; do
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" robotwin_b/feature_cache.py \
    "${cache_args[@]}" --shards 8 --shard "$gpu" &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do
  wait "$pid" || failed=1
done
if [ "$failed" -eq 0 ]; then
  "$PYTHON_BIN" robotwin_b/feature_cache.py "${cache_args[@]}" --verify
fi
```

## Four-GPU training

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 bash robotwin_b/launch.sh \
  --data "$DATA" --model "$MODEL" --encoder "$DINO" --empty-text "$EMPTY" \
  --feature-cache "$CACHE" --run "$RUN" --max-robot-frames 16 --epochs 200
```

The recipe starts from frozen Zero-WAM pretrain with rank-106 LoRA and
465,675,523 trainable parameters. It uses 16 pairs per rank, 64 pairs per
optimizer update, and 33 updates per epoch. It reshuffles the dropped tail
each epoch and buckets by query length. For long episodes, consecutive
windows rotate across epochs and the relative action origin resets at the
window start. Padding is masked. The default allocator reservation target is
121 GiB per rank and is checked at update boundaries. Use `--no-reserve` to
disable it.

`checkpoints/latest.pt` stores weights, optimizer, scheduler, the training
cursor, and per-rank RNG. Re-run the same command with `--resume` after all
previous ranks have exited. The optional publisher verifies the remote
adapter digest every five epochs and uses a private model repository.

```bash
"$PYTHON_BIN" robotwin_b/publish.py --run "$RUN" \
  --state /path/to/publisher-state --repo ACCOUNT/MODEL --epochs 200
```

## Tests

```bash
"$PYTHON_BIN" robotwin_b/verify.py window --data "$DATA" --feature-cache "$CACHE"
"$PYTHON_BIN" robotwin_b/train.py \
  --data "$DATA" --model "$MODEL" --encoder "$DINO" --empty-text "$EMPTY" \
  --feature-cache "$CACHE" --run /path/to/new/smoke-run \
  --device cpu --frames 16 --compact
"$PYTHON_BIN" robotwin_b/verify.py checkpoint "$RUN/checkpoints/latest.pt" --epoch 1
```

The compact smoke test uses real input data and a narrow random backbone to
verify one forward pass, backward pass, optimizer update and adapter reload.
Omit `--compact` to test the full pretrained backbone. The window test uses
a nonzero start to check camera, action and B1 alignment. The checkpoint test
expects the named epoch boundary and reads the actual optimizer, scheduler,
and RNG state. Closed-loop RoboTwin simulation evaluation is outside these
tests.

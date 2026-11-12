# Native text pi0.5 on RoboTwin43

This adapter fine-tunes the official `pi05_base` using the shared OpenPI
PyTorch trainer. It uses the released robot instruction, current head/left
wrist/right wrist RGB and tokenized measured EEF state. It does not use human
Demo RGB, the segmented Demo model, or task IDs as language prompts.

Input is the already verified `segmented_transfer_v1` RoboTwin cache: 2143
episodes, 457680 frames and 43 training tasks, excluding the six held-out tasks
and the microwave task. It does not copy or modify that cache. Preparation
fits pi0.5's native 1st/99th percentile normalization on training data only,
and preserves absolute 16D EEF targets, quaternion ordering and the 50 Hz clock.
Future targets extend 50 frames; padding in time and channels is masked in
the flow loss. State and actions use the native padded32D interface.

```bash
export PYTHONPATH=src:packages/openpi-client/src
python examples/robotwin_text/train.py --prepare-data /path/to/verified/data \
  --assets /path/to/run/assets.json
torchrun --standalone --nproc-per-node=4 examples/robotwin_text/train.py \
  --config /path/to/run/config.json --output /path/to/run/train
```

The host config supplies `data`, `assets`, `tokenizer`, `base_checkpoint`,
`world_size`, `per_gpu_batch`, `global_batch`, `steps`, `seed`, `warmup`,
`lr_peak`, and `lr_end`. Batch is actual per GPU, without accumulation.
The loader reuses the verified uniform-frame sampler, resume cursor, metric
and checkpoint receipts. Source, data, prompt/statistics, tokenizer and base
identities are frozen in its contract. `--steps` can end a startup run early;
`--resume` continues its optimizer, RNG, loader cursor and original LR horizon.

Use the predeclared final checkpoint for held-out evaluation, retaining every
5000-update checkpoint for recovery. A successful startup/readback is not a
closed-loop success result. At inference use the same tokenizer, normalization
and camera order; unpad/denormalize the first16 action channels, normalize both
quaternions, bound source grippers to [0,1], and execute with native RoboTwin
EEF control. The six-task frozen evaluation panel remains separate.

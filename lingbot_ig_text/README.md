# LingBot-VA text on IG C15

This adapter probes native LingBot-VA full fine-tuning with IG human task
descriptions. It does not load the RoboICL Demo adapter or its LoRA weights.
The upstream Trainer, flow-matching losses, random chunk/history windows,
30-block activation checkpointing and FSDP remain native.

Pinned upstream: `Robbyant/lingbot-va@7c6ffa9bfc4b83582cafc860fab4c82cc7deeeeb`.
Foundation: `robbyant/lingbot-va-base@68b7bc1b35da6ddc67ea94c4ceb58d768fbb3f9c`.
`prepare_upstream.py` only makes the unused FlashAttention import lazy; flex
training and torch inference do not require that optional package.

## Data contract

- Use the existing verified IG C15 manifest: 3,000 complete robot episodes,
  15 training tasks. The five unseen robot tasks never enter this dataset.
- Query uses the causal zed2i VAE cache at the established IG clock: 16 native
  action samples per latent. Reuse requires matching VAE weights and config.
- The prompt is the first official `human_desc.json` description of the
  paired human task, frozen for training/evaluation. The native UMT5 encoder
  produces 512 tokens; native classifier-free text dropout is 0.1 in training.
  No human RGB, Demo pose/action, or target robot task description is passed.
- Reuse the existing train-split quantiles and episode-relative joint
  coordinates. The 16 dual-arm joint/gripper commands map to native channels
  14–27 and 28–29, preserving the original left/right order. Initial action
  history is a supervised normalized-zero group; incomplete tail positions
  and unused channels are masked.
- No extra query truncation, frame subsampling, length buckets, or accumulated
  gradients are introduced by this adapter.

## Finite verification entry points

`encode_text.py` caches the selected human descriptions with the upstream text
encoding method. `train_probe.py` injects `IGTextDataset` into the upstream
trainer and executes only the frozen probe indices from a JSON config.
`infer_probe.py` loads the resulting checkpoint and decodes one action chunk
from the true first Query latent and text. It does not actuate a robot.

The initial probe uses four GPUs, actual batch 1 each, full-parameter AdamW
at the upstream 1e-5 learning rate, BF16 FSDP, native activation recomputation,
and 12 updates (two longest-sequence updates plus ten random full-episode
updates). A saved probe checkpoint is **smoke-only**. Its loss or generated
actions are not a benchmark success result.

The probe scripts remain finite verification tools. The separate production
entry below adds complete restart state; upstream itself saves inference
weights only. The live IG rollout bridge still needs validation before scoring.

Machine paths and probe receipts are maintained by the root repository in
`docs/local/ASSET_LOCATIONS.md` and
`outputs/lingbot_ig_text_probe_20261110/`; do not copy host credentials here.

## Verified probe (2026-11-10)

All four ranks completed 12 finite updates and changed the action head.
Peak allocation was 26.944 GiB per GPU under four-way FSDP. The ten sampled
updates had a median of 9.948 s and a mean of 10.796 s, including cache reads.
Training and saving took about 244 s; this is not a full-epoch benchmark.
The checkpoint was read back and loaded for native 25-video/50-action-step
inference. One non-actuating first chunk returned 16×16 finite actions in
3.545 s, with 13.512 GiB peak allocation. Query cache resolution is 224×288.

At the tested global batch of four, C15 has 750 updates per epoch. Five epochs
extrapolate to 10.36–11.25 h; ten to 20.72–22.49 h, using the measured median
and mean respectively. Preparation, complete restart checkpoints and closed-loop
evaluation are additional. These are scheduling estimates, not convergence
guarantees. The upstream 50,000-update default is substantially larger; the
official eight-GPU example also has a different global sample budget.

## Mixed complete-episode batches

`batching.py` pads variable-length Query/action tensors and preserves the mean
of native per-episode losses. Run `prepare_upstream.py SOURCE --mixed-batch`
against a fresh pinned source copy to exclude padded tokens from native self-
and cross-attention. The native temporal attention predicate is retained.
No length bucketing, query truncation, or dropped tail batches are required.

The runtime probe accepts `world_size`, `batch_per_gpu`, `mixed_batch`,
`activation_checkpointing`, `verify_padding`, and `save_checkpoint` in its
frozen config. Real batches can be specified as lists of indices. Padding
verification perturbs only padded latent/action inputs and targets, then
requires identical real-region predictions and losses. It is part of the
bounded runtime probe, not a separate test suite.

At global batch 64, the 3,000-episode distributed shuffle gives 47 updates per
epoch, with 56 real samples in the final update. A larger batch changes the
number of optimizer updates at a fixed epoch count; retain this distinction
when reporting training budgets.

## Production C15 run

`train.py --config CONFIG` reuses the native Trainer/optimizer/loss step and
the verified padding adapter. The frozen budget is eight GPUs, actual batch8
per rank, 60 epochs/2820 updates, BF16 FSDP, native activation recomputation,
no gradient accumulation. AdamW retains upstream LR1e-5, beta(0.9,0.95),
weight decay0.1, and ten warmup updates followed by constant learning rate.
Every epoch follows DistributedSampler(seed42), covers all3000 episodes,
and retains the final global batch56. No held-out task selects checkpoints.

`checkpoint.py` uses PyTorch Distributed Checkpoint for sharded FP32 model and
Adam states, plus scheduler, per-rank Python/NumPy/CPU/CUDA RNG, epoch/batch
cursor and source/config/data fingerprints. Recovery saves after2 and20 steps,
then every100 updates and at export/final boundaries. A complete directory is
published by an atomic latest pointer before the prior rolling state is removed.
The controller first exits at `--stop-after 2`, then restarts with `--resume`.
All ranks compare full local model/optimizer tensor digests before continuing
at step3; the two initial updates count toward the formal budget.

Export native BF16 inference weights at epochs20,40,60. Use separate node-local
storage for these large checkpoints; run metadata/source remain in the shared
artifact directory. A complete recovery state is much larger than inference
weights. The task-owned controller has finite startup/training deadlines and
stops on failure without changing batch or restarting a parameter search.

Eight-GPU capacity check (2026-11-10): actual batch 8 per rank, global 64.
Without recomputation all eight ranks ran out of memory before the first
optimizer update (peak up to 77.544 GiB). With the native 30 checkpoint wrappers,
two longest-sequence and six normal mixed-batch updates passed, peaking at
41.747 GiB per GPU. All eight real-region prediction/loss padding perturbation
checks were exactly unchanged. Normal update median/mean were 23.526/23.743 s.
At this tested batch, 10 epochs extrapolate to 3.07 h and 60 epochs to 18.43 h
using the median, before save/evaluation overhead. Keep recomputation for this
batch. This is finite capacity evidence, not an evaluation success result.

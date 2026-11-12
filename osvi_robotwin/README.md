# OSVI-WM on HumanGen-RoboTwin: 43 train → 6 held-out tasks

Finite integration of [official OSVI-WM](https://github.com/raktimgg/osvi-wm),
revision `ebd1fa7131078d10dbb101c8f0e3c9b3b86edb81`. The upstream checkout is an
external dependency and retains its original authorship. This module does not
copy the official model or SoftDTW implementation into RoboKata.

This baseline uses HumanGen-RoboTwin, **not** legacy RoboICL (3000 recordings)
or new RoboKata (3600 recordings). No formal training or benchmark score is
provided by this preparation.

## Data and method contract

- Reuse `act_robotwin.protocol` for the pinned release, 2149 human/robot pairs,
  43 training tasks and six evaluation tasks. Microwave remains held out but
  is excluded from evaluation. Do not add it to training.
- Human Demo: ten uniform RGB frames. Query: six uniform head-camera RGB
  frames across the recorded episode during training, 224×224, ImageNet
  normalization. No Demo action, state, text or task-label input.
- At inference only the current Query frame is available; repeats of that
  frame fill the upstream encoder interface. Latent future states are still
  generated autoregressively; no future Query frame is supplied.
- Keep upstream ResNet18, six-block action model, latent world model,
  attention pooling and 1-through-5 waypoint supervision. The current code
  initializes ResNet18 with **ImageNet weights**. The world model and output
  head are random. Do not describe this implementation as entirely from
  scratch despite the paper's wording.
- Extend the output from single-arm XYZ/gripper to two native XYZ/WXYZ/gripper
  poses (16 channels). Train against50 samples of the native recorded EEF
  **action** trajectory. The earlier measured-state smoke remains historical;
  the user's native-data choice supersedes it before any formal update.
  Quaternion signs are made continuous along each episode;
  quaternion channels are scaled by 0.2. Closed-gripper targets are 0.2.
- Use robot/world coordinates from the source directly, rather than the
  original camera projection. Preserve upstream trajectory interpolation
  and CUDA SoftDTW independently for each arm; average the arm losses. The
  last channel of each arm remains the piecewise-constant grasp target.
- Preserve upstream stop-gradient latent L1 supervision, asymmetric mixup,
  AdamW LR2e-4/weight decay1e-3, OneCycle schedule and target-encoder EMA.
  The finite probe does not include a full spatial augmentation pipeline.
  The formal trainer adds clip-consistent ColorJitter(0.4,0.4,0.4), independently
  for human and robot clips. It omits spatial flips/crops for the native world
  pose adaptation; this differs from the official camera-projected recipe.
- Inference takes the five-waypoint head, normalizes predicted quaternions,
  thresholds grippers and uses sign-aware pose interpolation followed by
  native `take_action(..., action_type='ee')`. No depth grasp correction,
  privileged object pose or expert trajectory is used as a policy input.

These output, coordinate and controller adaptations must be disclosed when
reporting results. A smoke checkpoint is rejected by the ordinary deployment
constructor; it must never populate a paper's success-rate table.

## Environment and finite check

Use a task-owned environment with Torch2.9.0+cu128, torchvision0.24.0,
TorchCodec0.8.1, NumPy1.26.4, PyArrow and the compatible FFmpeg libraries.
Numba0.61.2's deprecated CUDA target segfaulted on the tested driver. Installing
`numba-cuda==0.30.4` and `cuda-bindings==12.9.10` into the isolated environment
fixed the upstream SoftDTW forward/backward check; llvmlite0.44.0,
cuda-core1.2.2 and cuda-pathfinder1.8.6 were recorded. Set CUDA_HOME and
CUDA_PATH to an existing compatible toolkit (CUDA12.6 in this verification).
Do not modify another project's Python or CUDA installation.

Reuse the prepared real-pair manifest from ACT; it validates source hashes,
train/test membership, source clocks and measured quaternion norms. Example:

```bash
python -m osvi_robotwin.probe --data "$DATA" --prepared "$PREPARED_SMOKE" \
  --official "$OSVI_SOURCE" --run "$RUN" --batch-size 32 --steps 3
```

Verified on 2026-11-10 with two real training pairs repeated to actual batch32,
one A800, BF16, no recomputation or gradient accumulation: three optimizer
updates; finite losses/gradients; both waypoint and world-model weights
changed; peak allocated21.2663GiB. Last two updates averaged0.5536s on cached
samples. This is not a full-dataset throughput estimate. Exact checkpoint
readback and finite 5×16 recorded inference passed; prediction after reloading
matched the saved model. Only this smoke checkpoint was saved.

The `bridge` entry point permits two predictions for an isolated SAPIEN
interface check. Its explicit smoke override is not a formal evaluator.
The workspace experiment contract records whether native execution passed.

## Next formal stage

`python -m osvi_robotwin.train` is the single-GPU formal entry point. The first
RoboTwin budget is72000 updates, batch64, BF16, no recomputation/accumulation,
max LR2e-4 with OneCycle decay. It is an adapted initial budget, not the
official MetaWorld budget or a convergence claim. Each sample is a complete
paired episode; ten passes through2149 fixed pairs would be only340 updates.
The world-model coefficient decays on updates/10 and EMA spans the frozen
update budget. Full manifest SHA/pose/RGB checks precede training. The trainer
caches clips on CPU and seeds clip augmentation by epoch and row for resume.

Checkpoints include model, optimizer, scheduler, RNG and next batch cursor.
Save rolling recovery after20 updates, every1000 updates and at the final
boundary; retain every5000 updates and the final72000 separately. Resume with
the same source/config/data and budget. The previous10000-update schedule is
retired and preserved; the72000-update runs start from the original initialization
so OneCycle, EMA and latent-loss schedules span the entire new budget.
Use the final preselected checkpoint, never the held-out task success rate
to select among checkpoints. Training throughput must be measured with all
real pairs before giving a completion estimate.

Before scoring: freeze the same six-task seeds, expert-feasibility protocol,
human Demo mapping and action budget as the comparison. Native hooks in
`policy.py` are `get_model`, `reset_model`, `eval`, with arguments
`osvi_checkpoint`, `osvi_official`, `osvi_demo_root`, `osvi_demo_map`.
The adapter's availability alone does not establish a trained baseline.

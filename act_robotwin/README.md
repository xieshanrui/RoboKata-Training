# ACT/DINOv2 on HumanGen-RoboTwin: 43 train → 6 held-out tasks

This is a video-conditioned ACT baseline for the transfer comparison. It reuses
the IG ACT DETRVAE, ResNet18 observation encoder and frozen DINOv2-large Demo
encoder. It is not RoboTwin's ordinary single-task ACT. Smoke checkpoints are
explicitly marked and rejected by deployment.

The 2026-11-10 formal run completed 10 epochs / 35,760 updates on 2,143
episodes after excluding six episodes with corrupt wrist videos. The original
release is unchanged. The preselected final checkpoint passed parameter,
optimizer, schedule, data-coverage and recorded-input inference checks. These
checks do not establish closed-loop task success.

Preparation verified on 2026-11-10: two real paired episodes, batch128 on one
A800, two optimizer steps, peak allocated41.81GiB/reserved42.30GiB, exact
checkpoint readback and finite 24×16 recorded-pair inference. The first step
uses the scheduler's zero-LR warmup start. Median update time was2.48s, or4.36s
including data loading; two cached episodes do not establish full-dataset
throughput. No physical rollout or formal training was run.

## Frozen data contract

- Dataset: [HumanGen-RoboTwin-43Train-7Test](https://huggingface.co/datasets/xieshanrui/HumanGen-RoboTwin-43Train-7Test),
  revision `e2a347bb78dd48fd869bd7642a51430a60676876`.
- `manifests/train.jsonl`: 2,149 original robot/human pairs, 43 train tasks,
  854 distinct human videos. No new pairing or task-label input.
- Keep the original 43/7 split. Exclude `open_microwave` only from evaluation;
  never add it to training. The six task IDs are in `eval_tasks.txt`.
- Query: head, left wrist, right wrist RGB in that order, resized to 224×224,
  plus 16-D absolute native dual-EEF state. Targets: the corresponding native
  absolute EEF poses and grippers, **not joint positions**. Preserve source
  quaternion order. The source time grid is 50 Hz.
- Demo: human RGB only, 10 uniform frames; frozen DINOv2-large raw CLS pooled
  over frames, then the official trainable 256-D CLS adapter and task LayerNorm.
  Text, Demo actions/states and task IDs are not policy inputs.
- Normalize action/state with mean/std from the train manifest only. Fixed
  24-action chunks repeat terminal actions at an episode boundary; no chunks
  cross an episode. Query images use the IG implementation's [0,1] convention.
- Report the final preselected checkpoint. Do not select by held-out success.

The IG source used for preparation is `55615649dd05578ca0073beb375aa4edf8d8277c`.
Every training run saves hashes of the actual imported ACT/encoder files.
The model recipe follows the IG README: hidden1024, FF4096, 16 heads, 12 encoder
and 12 decoder layers, KL10, horizon24, batch128, 10 epochs, one warmup epoch,
cosine LR1e-4, ResNet LR1e-5, AdamW weight decay1e-4. This is an adaptation to
RoboTwin's cameras/state/action representation, not an official published
RoboTwin ACT result. One GPU, FP32 by default; `--precision bf16` uses autocast
with FP32 model/optimizer parameters. No activation recomputation or gradient
accumulation. The formal 2026-11-10 run uses actual batch128 and BF16.

## Environment and paths

Use an isolated environment, not a stock repository installer. The preparation
uses Python3.12, Torch2.9.0+cu128, torchvision0.24.0+cu128,
torchcodec0.8.1, OpenCV-headless4.11.0.86, IPython8.37.0, h5py3.14.0,
plus pyarrow, transformers, huggingface_hub, diffusers, einops, scipy and numpy.
TorchCodec needs compatible FFmpeg shared libraries. Check imports before GPU
allocation. Source and datasets remain in separate directories.

Run from the RoboKata root with that root on `PYTHONPATH`. Set `IG_ROOT` to the
pinned `imitator_game` checkout and `DATA`, `WORK`, `RUN` to external output
directories. These commands are templates; the full-data steps are not
automatically authorized or launched by preparing this code.

```bash
python -m act_robotwin.prepare --data "$DATA" --output "$WORK/prepared.json" --download
python -m act_robotwin.cache_demo --data "$DATA" --prepared "$WORK/prepared.json" \
  --ig-root "$IG_ROOT" --hf-cache "$WORK/hf" --cache "$WORK/features"
python -m act_robotwin.train --data "$DATA" --prepared "$WORK/prepared.json" \
  --ig-root "$IG_ROOT" --feature-cache "$WORK/features" --run "$RUN" \
  --batch-size 128 --epochs 10 --warmup-epochs 1 --workers 4
```

If the server cannot reach HF, transfer the same pinned files and omit
`--download`. Do not substitute an unpinned mirror or another encoder. A local
`--dino-path` requires `revision.json` with `revision` and SHA256 `files` for
`config.json`, `preprocessor_config.json`, `model.safetensors`.

`prepare` validates counts, task split, action/state shape, timestamps,
quaternions and RGB frame counts; it writes file hashes and train-only stats.
`cache_demo` stores frozen pre-adapter features keyed by actual video SHA.
`train` saves source/config/data identity, loss components, finite gradients,
LR, data-loading time, update time and peak memory. `latest.pt` resumes with
optimizer/scheduler/RNG and the next batch cursor. The epoch permutation is
seeded explicitly; the resumed loader starts at the saved cursor. Saves occur
after update20, every5000 updates and each epoch. Keep the original epoch,
batch, precision and learning-rate budget on resume. Keep every second epoch
as a separate hard-linked candidate; test outcomes do not select it.

## Finite preparation check

For a small check, use `prepare --smoke-pairs 2` with a separate prepared file,
then the same feature cache command and `train --smoke-steps 2 --workers 0`.
Start with the actual intended batch128; only reduce it after a measured OOM.
The result records memory, throughput and exact checkpoint/inference readback.
Smoke mode cannot resume a formal run and never supplies paper success rates.

```bash
python -m act_robotwin.infer --data "$DATA" --prepared "$WORK/prepared_smoke.json" \
  --checkpoint "$RUN/latest.pt" --ig-root "$IG_ROOT" --dino-path "$DINO" \
  --output "$WORK/recorded_inference.json" --allow-smoke
```

This last command runs a real recorded RGB/Demo pair through denormalization,
quaternion normalization and gripper bounds without actuating a simulator.
It is not a closed-loop success check. The workspace experiment receipt records
which checks actually passed; code presence alone is not runtime verification.

## RoboTwin deployment interface

`act_robotwin.policy` exports native `get_model(usr_args)`, `reset_model(model)`
and `eval(TASK_ENV, model, observation)` hooks. Set `policy_name` to
`act_robotwin.policy`; no copy into or modification of a frozen RoboTwin
checkout is required. Required arguments are `task_name`, `act_checkpoint`,
`ig_root`, `dino_path`, `act_demo_root`, `act_demo_map`.

`act_demo_map` maps exactly the six held-out task IDs to RGB video paths beneath
`act_demo_root`. Freeze these paths/SHAs before evaluation, using the existing
comparison's `demo_files.json`. The hook reads `observation['endpose']`, not
`joint_action.vector`, executes 24 absolute actions with
`TASK_ENV.take_action(action, action_type='ee')`, and stops on native success
or the task action budget. It uses no text instruction.

```bash
python -m act_robotwin.prepare_eval --source-manifest "$REFERENCE/demo_files.json" \
  --demo-root "$REFERENCE" --output "$WORK/eval_demos"
```

The resulting `demo_map.json` and `provenance.json` preserve the reference
videos and their SHAs. Set `act_demo_root` to the same `REFERENCE` directory.

Before the formal 6×50 evaluation, freeze the expert-feasible initial states,
Demo mapping, native task fixes, seed list, finite expert-candidate limit and
wall budget with the existing RoboTwin evaluation harness. The stock
`script/eval_policy.py` hardcodes 100 rollouts and an unbounded expert search;
do not mistake adding this policy hook for the prepared 50-rollout protocol.
This preparation does not run that stock evaluator or claim its results are
already comparable to B1. Physical closed-loop integration remains a separate
pre-evaluation check after a formal checkpoint is available.

`python -m act_robotwin.bridge` runs the same `Policy` in the training environment
while a separate simulator exchanges RGB/state inputs and native EEF chunks.
It requires explicit checkpoint, pinned IG/DINO paths, frozen Demo map/root,
exchange directory and a finite wall budget. The bridge performs no action
interpolation or task scoring. The external evaluation runner owns initial-state
verification, native execution, videos, deadlines and outcome records.

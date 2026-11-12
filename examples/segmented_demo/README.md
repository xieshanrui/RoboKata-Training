# Segmented RGB Demo π0

This is the segmented RGB Demo adapter for `scripts/train_pytorch.py`. Its shared structure is learned visual pooling, RGB Demo sampled at 5 Hz, and two causal VLM layers followed by one-second hidden-state compression. The action head predicts 50 native action samples. Demo actions, states, task labels and report overlays are not inputs. The original RoboICL defaults remain dual-view Demo/Query, measured TCP10 state and 50 Hz actions.

The current baseline indexes every Query frame and uses PyTorch's shuffled `DistributedSampler` and `DataLoader`. There are no phase quotas or stationary-frame caps. Each frame selects a Demo from its frozen legal pairing list; no tenfold copy of Query frames is made. One epoch is a traversal of Query frame indices, except the distributed/batch `drop_last` tail. For 1,370,730 frames and global256, an epoch contains 5,354 updates / 1,370,624 frames; 106 shuffled tail frames are omitted. Demo selection is seeded by frame index and epoch, and loader epoch/cursor are saved for resumption.

## Data and training

`data/manifest.json` identifies canonical recordings and supervision; `pairing_file` and its SHA fix allowed directed pairs. `normalization.json` is fitted on training data only. Each `records/<id>/` contains `policy_recording.npz` and `supervision.npz`. The preparation command verifies source hashes and reconstructs the disposable `rgb.npy` memory maps:

```bash
python examples/segmented_demo/data.py --prepare-data /path/to/data
```

Use [config.example.json](config.example.json), replacing the data, tokenizer, base-checkpoint and cache paths. The completed historical run used eight GPUs × actual batch32, global256, no accumulation, 20,000 updates with weighted sampling. The new uniform-frame baseline is a separate recipe. The cosine horizon remains30,000. Raw frozen SigLIP patches can be cached using `cache_fixed.py`; pooling and projection remain trainable. Cache identity must match the data and model.

```bash
PYTHONPATH=src:examples/segmented_demo torchrun --standalone --nproc_per_node=8 examples/segmented_demo/train.py --config /path/to/config.json --output /path/to/new-run
```

The example intentionally omits host-specific CPU affinity. Set resource paths and verify actual GPU memory before launch. `--steps` sets the total update boundary; `--resume` resumes this new checkpoint format. The native trainer saves at multiples of5,000 and at the final step, including a nonmultiple such as27,000. No scheduler is started by these entries.

`--import-legacy /path/to/step_0020000` converts an old checkpoint into a new output directory while retaining its model, Adam state, torch RNG and absolute LR step. This requires `legacy_parameter_names` in the config: a JSON containing `checkpoint_marker_sha256` and the ordered `names` of trainable parameters exported from the checkpoint's frozen model source. The converter checks this ordering, parameter shapes, model configuration and data identities. It starts a new uniform-frame sampler at epoch0; describe this as sampling-adjusted continuation. Original checkpoint files are retained. Do not point the new trainer at a historical output directory.

## Checkpoint provenance and inference

Historical checkpoints strictly verify their frozen source and input hashes. To reproduce the original weighted-sampling20k run, use its archived source: `7284575` for the pre-extension training entry, `f7d2de1` for budget-only continuation. Do not edit checkpoint hashes to make a cleaned checkout match. The current worker accepts a separate `--source` for that frozen implementation and uses the shared binary protocol. New checkpoints also retain a `complete.json` with source/data identities for this worker. Current evaluation predicts50 and executes50; historical execute25 results remain separate.

The current model implementation is in `src/openpi/models/pi0_frame_demo_config.py`, `src/openpi/models_pytorch/pi0_frame_demo_pytorch.py` and `demo_compression.py`. The binary protocol now lives here in `protocol.py`; it no longer imports an old model worker. The old `examples/roboicl/`, `examples/video_icl/` and VAE model implementation are retired from the active tree.

The model supports separate explicit `world_tcp10`, `robotwin_eef16` and `ig_joint16` contracts, each trained as an independent model. All use a padded32D head; the sampler's zero-target bridge begins after the contract's10 or16 valid action channels. SigLIP stays frozen. Query cameras and Demo camera count are independent. Old defaults and their parameter shapes remain unchanged. Historical checkpoints must still use their original frozen source through `--source`; source-hash verification is not bypassed by parameter compatibility.

`configure.py` creates a local configuration without editing the defaults. `evaluate.py` runs a finite frozen panel with an explicitly assigned GPU, a persistent model and a fresh RoboTwin simulator per scene. It accepts separate resource paths and a simulator Python interpreter, verifies execute50 and same-family reset, and retains failed attempts. `pipe_client.py` supplies the simulator-side pipe. The portable path was checked on A800 with one actual batch32 update, checkpoint reload and one complete policy episode/report in an isolated simulator environment. This is deployment verification, not a policy-performance result or a new 20k training run.

The root RoboICL `docs/REPRODUCING.md` documents resource bundles, cache preparation, training, evaluation and report commands. Use `robotwin/scripts/export_segmented_eval.py` for unannotated Demo-above-policy videos and separate trajectory diagnostics. Historical server runners remain intact for provenance.

## RoboTwin43 and IG-10K C15 transfer

| Contract | Demo | Query | State / action | Native frequency |
| --- | --- | --- | --- | --- |
| RoboTwin | human RGB, one view | head, left wrist, right wrist |16D EEF /16D EEF |50 Hz |
| IG C15 | human RGB, one view |zed2i |18D measured state /16D joint target |30 Hz |

`prepare_transfer.py` materializes resized RGB memory maps, native labels, allowed pairs and train-only normalization. It uses TorchCodec for released AV1 videos. Preparation needs PyArrow, TorchCodec and OpenCV; the training environment reads NumPy/SigLIP caches and does not need these decoding dependencies. FFmpeg shared libraries must be visible to TorchCodec. No dependencies are installed by these commands.

RoboTwin uses the frozen43-task HumanGen manifest. For three Query cameras, supply the verified2143-pair clean manifest that excludes six corrupt wrist-video episodes. The six evaluation tasks remain held out; the excluded microwave task must not be added to training. IG uses the existing C15 cache manifest to identify3000 robot episodes and750 human recordings, preserving their original state/action clocks and same-task pairing. Only RGB from humans is materialized.

```bash
python examples/segmented_demo/prepare_transfer.py --dataset robotwin \
  --input /path/to/prepared_clean.json --root /path/to/HumanGen \
  --output /path/to/robotwin_data
python examples/segmented_demo/prepare_transfer.py --dataset ig10k \
  --input /path/to/c15/cache/manifests/train.json --output /path/to/ig_data
```

`--limit 16` creates an explicitly smoke-only subset including longest Query/Demo cases. Normalization still uses the full training split. Never report these short-run weights as trained baselines. Human Demo frames cover the whole recording at5 Hz; the configured capacity must contain them all.

Use the same native training entry for all datasets. `configure.py` reads the prepared manifest and selects cameras/action encoding; no task-specific training loop is added. The following is a two-GPU configuration example, **not a convergence claim**:

```bash
export PYTHONPATH=src:packages/openpi-client/src:examples/segmented_demo
python examples/segmented_demo/configure.py --data /path/to/prepared_data \
  --base /path/to/pi0_base/model.safetensors --tokenizer /path/to/paligemma_tokenizer.model \
  --cache /path/to/demo_cache --gpus 2 --batch-per-gpu 32 \
  --steps 30000 --lr-decay-steps 30000 --output /path/to/config.json
mkdir /path/to/demo_cache
python examples/segmented_demo/cache_fixed.py --data /path/to/prepared_data \
  --output /path/to/demo_cache --base /path/to/pi0_base/model.safetensors --frames 60 --world 1
python examples/segmented_demo/cache_fixed.py --data /path/to/prepared_data \
  --output /path/to/demo_cache --base /path/to/pi0_base/model.safetensors --frames 60 --world 1 --merge
torchrun --standalone --nproc_per_node=2 examples/segmented_demo/train.py \
  --config /path/to/config.json --output /path/to/new_run
```

Set cache `--frames` to the generated config's `demo_frames` if it exceeds60. Cache only raw frozen SigLIP patches; the visual projector and Demo pooling continue to learn. Current training uses DDP, activation checkpointing and no gradient accumulation. Checkpoints contain the model, optimizer, per-rank RNG and sampler cursor. Resume requires unchanged data, source, GPU topology and optimization contract. A frame epoch is not a WAM episode-sampling epoch: budget training by total Query samples and observed training-side convergence, with a checkpoint selection rule frozen before held-out evaluation.

`transfer_policy.TransferPolicy` strictly reloads these checkpoints, accepts full unannotated Demo RGB/timestamps in `reset`, and returns50 native16D targets from current Query RGB/state in `predict`. Its CLI uses the existing NPZ binary pipe: `reset` carries `rgb`, `timestamps`, `duration`, `seed`; `predict` carries `rgb`, `state`. RoboTwin output preserves quaternion ordering, normalizes quaternions and bounds grippers to[0,1]; IG output remains native joint targets. Simulators own control-step execution and evaluation termination. Use the separate historical `worker.py`/`evaluate.py` for old TCP10 recordings; never interpret native16D outputs as TCP10.

For CLI inference, `--source` selects the checkpoint-frozen OpenPI implementation before imports. Direct Python callers must import OpenPI from that same source; a mismatched imported package is rejected. This keeps a new launcher from silently running a different model implementation against an old checkpoint.

### Verified integration boundary (2026-11-11)

Both adapters passed three single-GPU updates at actual batch32, twenty fresh two-GPU updates at global64, restoration from update20 through22, and checkpoint-backed native-action inference. Adam states, both rank RNG/loader states, finite losses/gradients and padded-channel behavior were checked. Training used16 real Query recordings per dataset, including longest-recording cases; normalization used all training records. These are integration weights, not converged policy results.

| A80080GB PCIe, two GPUs, batch32 each | Median seconds/update (6–20) | Peak allocated GiB |
| --- | ---: | ---: |
| RoboTwin, three Query views |5.308 |42.334 |
| IG C15, one Query view |2.831 |36.752 |

The measurements use activation checkpointing, frozen raw Demo features and no accumulation. Full-split disk/cache working sets and checkpoint saves add overhead; these short-run timings do not establish convergence or formal evaluation latency.

### Explicit recomputation policy

`configure.py --checkpoint-strategy layer` keeps the joint decoder's per-layer activation checkpoints and removes the outer decoder, image-encoding and projection checkpoints. `nested` preserves the original behavior and remains the default. `none` disables activation checkpointing and requires a separate memory check at the intended actual batch. The setting is saved in the model configuration and checked on resume; it changes execution only, not trainable parameters, cameras, augmentation, Demo compression or supervision.

The attention implementation remains eager. A finite SDPA trial was rejected because its full-gradient relative difference exceeded the predeclared tolerance; no SDPA switch is shipped as part of this optimization.

On the same RoboTwin subset, batch32 single-GPU updates4–8 improved from4.649 to3.710 seconds with `layer`; all eight loss/gradient-norm records matched exactly. Same-input checks also matched the loss, velocity and every populated parameter gradient. Four GPUs passed20 updates plus resume through22 and native inference with `layer`. Measured four-GPU settings:

| Per-GPU batch / global batch | Checkpoint strategy | Median seconds/update | Peak allocated GiB | Pure-update budget estimate |
| --- | --- | ---: | ---: | --- |
|16 /64 |layer |3.723 |32.151 |30k updates:31.0h |
|32 /128 |layer |5.559 |42.253 |15k updates:23.2h |
|16 /64 |none |not established |OOM on update2 |not viable |

Both projected budgets above present1.92M Query frames, but the optimizer update counts differ. They do not establish equal convergence. The no-checkpointing16/GPU trial completed only its first update and then exhausted memory with Adam states resident; one successful update does not demonstrate steady-state fit. Batch32 without checkpointing also failed. Full-data preparation, cache misses and saves are excluded from the estimates, so the23.2h estimate is not a24h wall-clock guarantee.

`--pool-valid-only` gathers valid Demo frames before the trainable attention pool and restores the original padded token positions afterwards. It skips padding computation without trimming real frames, changing causal positions or caching trainable features. Same-input RoboTwin checks matched loss and velocity exactly; all populated parameter gradients differed by8.2e-8 in relative norm. Separate FP32 checks covered NaN padding and entirely padded inputs, including parameter gradients.

`--checkpoint-strategy selective --checkpoint-layers 0 1 2 3 4 5 6 7 8 9 10 11` checkpoints only the listed decoder layers. Indices are checked against the instantiated decoder depth; the remaining layers retain activations. This opt-in policy and valid-only pooling preserve the original defaults and parameter shapes. Memory acceptance requires multiple actual optimizer updates, including resident Adam states.

The entry records an explicit `allow_tf32` execution setting when supplied; `configure.py --allow-tf32` opts in. The RoboTwin optimization trial **rejected TF32**: full-gradient relative difference8.08% exceeded its predeclared3% tolerance. Its formal recipe keeps TF32 disabled. A small loss difference alone is insufficient numerical validation.

Full transfer preparation accepts `--workers 4` (default1, maximum8). Parallel decoding preserves manifest order, RGB pixels, native supervision and train-only normalization. Two real records were byte-checked against sequential preparation before using it on the full split. Preparation and checkpoint hashing share disk bandwidth; schedule bulk writes separately when they delay checkpoint completion.

With valid-only pooling and TF32 off, two A800s at actual32/global64 completed16 updates for each policy. `layer` measured4.421s/update and38.245GiB; checkpointing decoder layers0–11 of18 measured4.201s and67.752GiB. All16 loss, gradient-norm and learning-rate records matched exactly. The latter is about21% faster than the initial5.308s two-GPU recipe, while preserving30k updates /1.92M Query-frame presentations. Its pure-update estimate is35.0h; full-split I/O and saves remain additional costs.

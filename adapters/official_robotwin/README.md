# Official Zero-WAM / RoboTwin evaluation

This is the maintained version of the latest local `raster303_20261102` evaluation adapter. It uses the released dual-arm checkpoint, three Query cameras, HumanGen Demo latents with their cached text, native actions, native task success and the two task overrides supplied by Zero-WAM. It is separate from the RoboICL single-arm RGB-only adapter. Source revisions and the seven-task × 50-episode protocol are in [provenance.json](provenance.json).

The client retains expert feasibility filtering before policy execution, the narrow native empty-grasp rejection, candidate/wall limits, first-attempt policy failures, initial physical/RGB snapshots, action and camera journals, and unlabelled three-view videos. No PD, speed, action normalization, model sampler or success threshold is changed. Video export uses the original 15fps comparison sampling; it is not a claim of real-time simulation playback.

## Runtime

Use separate Python 3.12 inference and simulation environments. The effective server versions are recorded in `requirements-infer-linux-py312.txt` and `requirements-sim-linux-py312.txt`; no retired fine-tuning checkpoint is needed. The official benchmark needs a separate RoboTwin `2eeec322` checkout, SAPIEN **3.0.3** and cuRobo **0.7.8**. The root project's `robotwin/` is the single-arm collector and must not be repurposed as this benchmark checkout.

From this Zero-WAM repository, prepare dedicated environments (do not modify an existing shared environment):

```bash
python3.12 -m venv /envs/zero-infer
python3.12 -m venv /envs/zero-sim
/envs/zero-infer/bin/pip install -r adapters/official_robotwin/requirements-infer-linux-py312.txt
/envs/zero-infer/bin/pip install --no-deps 'https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl'
/envs/zero-sim/bin/pip install -r adapters/official_robotwin/requirements-sim-linux-py312.txt
git clone https://github.com/RoboTwin-Platform/RoboTwin.git /resources/official-RoboTwin
git -C /resources/official-RoboTwin checkout 2eeec322
git clone --branch v0.7.8 --single-branch https://github.com/NVlabs/curobo.git /resources/curobo-0.7.8
CUDA_HOME=/path/to/cuda-12.6 TORCH_CUDA_ARCH_LIST=8.0 MAX_JOBS=2 \
  /envs/zero-sim/bin/pip install --no-deps --no-build-isolation -e /resources/curobo-0.7.8
```

The CUDA compiler must match the simulation PyTorch CUDA12.6 build; `8.0` is the tested A800 architecture. Preserve the compiler/toolkit outside these source repositories. Both environments still need the host NVIDIA driver; headless raster rendering requires a working Vulkan loader. A deployment may provide its loader through `LD_LIBRARY_PATH`, without changing the task cameras or physics.

Download the exact released resources using `download_resources.py`. The HumanGen download selects the seven reference latents, including their cached text. Inference resolves the upstream video names to these latents; it does not need the raw-video archives. Model/asset downloads are large; reserve space before running. Versions are pinned by `provenance.json` and upstream licenses apply.

```bash
/envs/zero-infer/bin/python adapters/official_robotwin/download_resources.py \
  --model /resources/zero-model --human "$PWD/data/HumanGen" --asset-archives /resources/robotwin-archives
/envs/zero-sim/bin/python -m zipfile -e /resources/robotwin-archives/embodiments.zip /resources/official-RoboTwin/assets
/envs/zero-sim/bin/python -m zipfile -e /resources/robotwin-archives/objects.zip /resources/official-RoboTwin/assets
/envs/zero-sim/bin/python -m zipfile -e /resources/robotwin-archives/background_texture.zip /resources/official-RoboTwin/assets
```

`MODEL_PATH` below should point to `/resources/zero-model`; pass the downloaded `data/HumanGen/human_latents/robotwin` as `--icl_latent_root`. These instructions do not restart any existing server evaluation.

Run `prepare_robotwin.py /path/to/official/RoboTwin` to check the pinned files, then add `--apply` to install the recorded raster setup and the two upstream task overrides. Unknown source hashes are rejected. The preparation is idempotent and does not alter camera poses, physical settings or task definitions beyond those official overrides.

Use an idle GPU already allocated to the task. Bind both processes to the same CUDA UUID and set `ROBOICL_RENDER_PCI` to its twelve-character PCI address, e.g. `0000:41:00.0`. The renderer asserts the device match. Run the client with `SAPIEN_DISABLE_RAY_TRACING=1`; use the original `default` raster shader. Inference and simulation may use separate Python environments.

From the Zero-WAM repository, start a model server for the finite evaluation:

```bash
export MODEL_PATH=/path/to/zero-wam-posttrain-robotwin
export CUDA_VISIBLE_DEVICES=GPU-allocated-uuid
export TARGET_TEXT_CFG=-1 ICL_CFG=5
/path/to/inferenv/bin/python -m torch.distributed.run --nproc_per_node=1 --master_port=29981 \
  adapters/official_robotwin/server.py --config-name robotwin --port 29881 --save_root /path/to/run/server
```

The entry seeds the model once with 0 and binds the server to loopback. After its health endpoint is ready, run one task with an explicit external timeout (also covers native calls that cannot be interrupted by the Python wall check):

```bash
export ROBOTWIN_ROOT=/path/to/official/RoboTwin
export ROBOICL_RENDER_PCI=0000:41:00.0
export SAPIEN_DISABLE_RAY_TRACING=1
export MAX_EXPERT_CANDIDATES=1000 CLIENT_WALL_SECONDS=21600
timeout --signal=TERM --kill-after=20s 21600s /path/to/simenv/bin/python -u \
  adapters/official_robotwin/client.py \
  --config "$ROBOTWIN_ROOT/policy/ACT/deploy_policy.yml" \
  --port 29881 --save_root /path/to/new-run/place_empty_cup \
  --icl_latent_root /path/to/HumanGen/human_latents/robotwin \
  --icl_seed 0 --test_num 50 --video_guidance_scale -1 --action_guidance_scale 1 --icl_guidance_scale 5 \
  --overrides --task_name place_empty_cup --task_config demo_clean --train_config_name 0 \
  --model_name 0 --ckpt_setting 0 --seed 0 --policy_name ACT
```

Stop the task-owned model server when its assigned finite tasks finish. Check storage and occupancy before launching; this entry does not schedule GPUs, monitor other jobs or start an automatic 350-episode run. Machine-specific caches, environment paths, recovery controllers and reports stay under the run directory.

`FIXED_SMOKE_MANIFEST` can bind a prior smoke panel's seeds, prompts and latent hashes. For a reviewed continuation, `--start_seed`, `--episode_offset` and `RESUME_OFFICIAL_EPISODES=1` retain global episode numbering; `--test_num` is the total target, not the number of additional episodes. Preserve rejected expert seeds and all started policy attempts. Do not blindly rerun an interrupted output directory or recompute a failure until it becomes a success. Restarting the model resets its random stream and must be recorded.

The cleaned source has focused equivalence/readback checks; ongoing server runs continue using their original frozen source. A source cleanup is not a new physical benchmark result.

## A+B1 RoboTwin adapters

`b1_server.py` reuses the native three-camera stream, 30-channel action mapping,
quantile normalization, episode-relative EEF execution, cache and sampler. It
loads the published RoboKata A Transformer and LoRA tensors onto the **pretrain**
foundation. It validates every inference tensor and the shapes of the excluded
MCP/semantic training heads. It does not use the single-arm RoboICL action adapter.

The finite run must provide `ROBOTWIN_B1_RUN`, `ROBOKATA_ROOT`,
`ROBOTWIN_B1_CHECKPOINT`, and `B1_SERVER_RECEIPT`. `MODEL_PATH` selects the matching
foundation. The run directory contains `dino/`, `demo_files.json` and the raw RGB
videos referenced by that manifest (relative `file`, `sha256`, `frames`). The
manifest keys are the HumanGen sample-directory names. Demo RGB is decoded and
resized with `robotwin_b.data.read_rgb`, exactly as in the published training
branch, before uniform-eight midpoint sampling, FP32 DINOv2 and 2×2 spatial
pooling. Both native latent and A branches use RGB only; cached HumanGen text is
discarded. Existing official results using cached Demo text therefore have a
different conditioning contract and must be labelled when used as references.

The training entry is published in RoboKata branch `codex/robotwin-version-b`
at `e0451bad295a303c82c80f19025a3af9e876cfde`. Record that source revision, the
adapter's dataset/encoder hashes, and the native simulation protocol. An earlier
reconstruction used Torch image resizing; preserve its results as a separate
diagnostic condition rather than combining them with the aligned evaluation.

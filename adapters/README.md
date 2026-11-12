# RoboICL Zero-WAM inference adapters

Zero-WAM is currently used as a released-weight inference baseline. The retired RoboICL spatial/fixed-data fine-tuning and batch-benchmark launchers have been removed from the active tree. Their committed versions are in `archive/pre-share-cleanup-20261102`; the last uncommitted experiment was also backed up locally before removal. Official upstream training code remains unchanged.

The current official dual-arm benchmark entry is [official_robotwin](official_robotwin/README.md): released HumanGen latents/cached text, three Query views, native task scoring and the recorded SAPIEN raster adaptation. The files below are the separate RoboICL single-arm RGB-only adapter; their input contract must not be applied to the official benchmark.

- Demo uses external/head RGB only, sampled by actual duration at12fps and passed through the released causal video VAE; no wrist, states, actions, labels or overlays.
- Query uses external and wrist RGB with the native causal video/action context.
- `roboicl_worker.py` and `roboicl_client.py` provide the existing bounded inference interface.
- `roboicl_single_arm.py`, `roboicl_frames.py` and `roboicl_video.py` preserve action/frame conversion helpers and checkpoint compatibility.

Released-weight official-benchmark evaluation and RoboICL adapted-scene evaluation are distinct protocols. Do not treat an interface check or a converted rollout as official benchmark success. Collection, simulation and scoring belong to RoboTwin/root; run-specific frozen panels and reports remain outside Git.

## RoboICL fine-tuned adapters

`roboicl_lora_worker.py` loads the published Robot-ICL baseline or Version A. These checkpoints use head-only Demo and Query inputs, unlike the original released-weight worker above. Version A additionally requires `--version-a-source` pointing to the pinned `robokata` submodule, `--dino`, `--adapter-metadata`, and the matching `--baseline-contract`. Native execution remains first 16 actions then 32 with observed execution feedback. Training recipes live in `robokata`, and IG10K recipes in `imitator_game`; no retired training launcher is restored here.


The wrist extension reads `ig_query_views` from a full Version A training checkpoint. `head_wrist` requires both current Query RGB views and both executed-frame feedback views; head-only published adapters retain their original input. Use the matching `robokata` source and training `stats.json`. Demo input remains head-only. The worker returns camera count and a feedback-array hash for the execution readback.

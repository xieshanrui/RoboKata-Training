"""Run a one-update RoboICL training pilot with the IG Zero-WAM backend."""

import argparse
import hashlib
import json
import math
import os
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from examples.baselines.zero_wam.zero_wam.cache import sample_plan
from examples.baselines.zero_wam.zero_wam.dataset import IGDataset
from examples.baselines.zero_wam.zero_wam.runtime import atomic_torch_save

ACTION_CHANNELS = (*range(9), 28)
ACTION_INTERVAL = 16


def recording(root, episode, *, views=1):
    path = root / "train3000" / "records" / episode / "policy_recording.npz"
    with np.load(path, allow_pickle=False) as data:
        rgb = data["rgb"]
        actions = data["action"]
        states = data["state"]
        time = data["timestamp_s"]
        step = data["observation_step"]
    if rgb.dtype != np.uint8 or rgb.shape != (len(time), 2, 224, 224, 3):
        raise ValueError(f"Unexpected RGB schema: {path}")
    if actions.shape != (len(time), 10) or states.shape != actions.shape:
        raise ValueError(f"Unexpected action/state schema: {path}")
    if not np.all(np.diff(step) == 5) or not np.allclose(np.diff(time), 0.02, atol=1e-5):
        raise ValueError(f"Invalid control cadence: {path}")
    if not np.isfinite(actions).all() or not np.isfinite(states).all():
        raise ValueError(f"Non-finite robot data: {path}")
    pixels = rgb[:, 0] if views == 1 else np.concatenate((rgb[:, 0], rgb[:, 1]), axis=2)
    return pixels, torch.from_numpy(actions).float(), torch.from_numpy(states).float()


@torch.inference_mode()
def encode(wrapper, frames, ids, resize, *, chunk_frames=16):
    vae = wrapper.vae
    device = next(vae.parameters()).device
    dtype = next(vae.parameters()).dtype
    mean = torch.tensor(vae.config.latents_mean, device=device).view(1, -1, 1, 1, 1)
    std = torch.tensor(vae.config.latents_std, device=device).view(1, -1, 1, 1, 1)
    wrapper.clear_cache()
    chunks = [ids[:1], *[ids[i:i + chunk_frames] for i in range(1, len(ids), chunk_frames)]]
    latents = []
    for chunk in chunks:
        video = torch.from_numpy(frames[chunk]).permute(0, 3, 1, 2).float()
        video = F.interpolate(video, size=resize, mode="bilinear", align_corners=False)
        video = (video / 127.5 - 1).permute(1, 0, 2, 3).unsqueeze(0)
        result = wrapper.encode_chunk(video.to(device=device, dtype=dtype))
        mean_latent = result.chunk(2, dim=1)[0].float()
        latents.append(((mean_latent - mean) / std)[0].to(dtype).cpu())
    latent = torch.cat(latents, dim=1).contiguous()
    if not torch.isfinite(latent).all():
        raise ValueError("VAE produced a non-finite latent")
    return latent


def prepare(root, output):
    from examples.baselines.zero_wam.zero_wam.wan_va.modules.utils import (
        WanVAEStreamingWrapper,
        load_vae,
    )

    pair = json.loads((root / "pilot-pair.json").read_text())
    vae = load_vae(str(Path(os.environ["ZERO_WAM_MODEL"]) / "vae"),
                   torch_dtype=torch.bfloat16, torch_device="cuda").eval()
    wrapper = WanVAEStreamingWrapper(vae)
    output.mkdir(parents=True, exist_ok=True)
    contract = {
        "video": {
            "sim": {"source_frame_stride": 4, "action_tail_policy": "all_actions_v1"},
            "human": {"target_fps": 12.0},
            "vae_temporal_stride": 4,
        },
    }
    records = {}
    for role, kind, resize in [("demo", "human", (320, 480)), ("query", "sim", (224, 288))]:
        frames, actions, states = recording(root, pair[role])
        plan = sample_plan(len(frames), 50.0, kind, contract)
        latent = encode(wrapper, frames, plan["sampled_frame_indices"], resize)
        if latent.shape[1] != len(plan["latent_anchor_indices"]):
            raise ValueError(f"Unexpected latent length for {role}: {latent.shape}")
        path = output / f"{role}.pt"
        atomic_torch_save(dict(latent=latent, actions=actions, states=states,
                              latent_nominal_anchor_indices=torch.as_tensor(plan["latent_nominal_anchor_indices"]),
                              action_chunk_size=ACTION_INTERVAL), path)
        records[role] = {"cache_path": str(path), "repo_id": pair["task_id"]}
        print(role, pair[role], list(latent.shape), flush=True)

    stats = {}
    for name, values in [("action", actions), ("state", states)]:
        bounds = torch.quantile(values, torch.tensor([0.01, 0.99]), dim=0)
        bounds[0, 9], bounds[1, 9] = -1, 1
        stats[name] = dict(q01=bounds[0].tolist(), q99=bounds[1].tolist())
    stats["representation"] = "roboicl_world_hand10_absolute"
    stats["clip"] = 2.0
    (output / "stats.json").write_text(json.dumps(stats, indent=2))
    records["query"]["human_repo_id"] = pair["task_id"]
    manifest = dict(sim_episodes=[records["query"]], human_episodes=[records["demo"]],
                    records_sha256=hashlib.sha256(json.dumps(pair, sort_keys=True).encode()).hexdigest(),
                    pair=pair, action_channels=ACTION_CHANNELS)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))


class RoboICLDataset(IGDataset):
    """Reuse IG causal grouping with explicitly masked RoboICL action channels."""

    def __init__(self, manifest_path, empty_text_path, seed, action_per_frame,
                 normalization_path, sampling_mode, action_history_mode, **_):
        from examples.baselines.zero_wam.zero_wam.action_space import EpisodeActionSpace

        self.manifest_path = Path(manifest_path)
        self.manifest = manifest = json.loads(self.manifest_path.read_text())
        self.sampling_mode = sampling_mode
        self.action_history_mode = action_history_mode
        self.records = manifest["sim_episodes"]
        self.humans = manifest["human_episodes"]
        self.repo_indices = {self.records[0]["repo_id"]: 0}
        self.humans_by_repo = {self.humans[0]["repo_id"]: [0]}
        self.action_channels = list(manifest["action_channels"])
        self.action_space = EpisodeActionSpace(json.loads(Path(normalization_path).read_text()))
        self.action_per_frame = action_per_frame
        self.seed = seed
        self.payloads = OrderedDict()
        self.empty_text = torch.load(empty_text_path, map_location="cpu", weights_only=True)

    def episode_item(self, key):
        result = super().episode_item(key)
        # The existing observed-state projector accepts 18 inputs. The final
        # eight inputs are reserved, while the supervised action space stays 10D.
        result["ig_states"] = F.pad(result["ig_states"], (0, 8))
        return result


def train(output):
    from examples.baselines.zero_wam.zero_wam import cache, training
    from examples.baselines.zero_wam.zero_wam.config import ZeroWAMConfig

    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    native_config = training.native_config

    def smoke_config(config, selected):
        settings = native_config(config, selected)
        settings.num_steps = 1
        settings.ig_max_steps = 1
        settings.drop_icl = 0.0
        return settings

    cache.require_complete = lambda config: manifest
    training.native_config = smoke_config
    training.IGDataset = RoboICLDataset
    training.run(ZeroWAMConfig(
        task="L0_TwoRobotPickRemoteControl-v1",  # IG CLI validation only; loader uses the RoboICL manifest.
        model_path=os.environ["ZERO_WAM_MODEL"],
        empty_text_embedding=os.environ["ZERO_WAM_EMPTY_TEXT"],
        human_root=str(output),
        sim_root=str(output),
        cache_dir=str(output),
        manifest=str(manifest_path),
        run_dir=str(output / "run"),
        normalization_path=str(output / "stats.json"),
        effective_batch_size=1, total_epochs=1, warmup_steps=0,
        activation_checkpointing=True, num_workers=0,
    ))
    metrics = json.loads((output / "run/latest_metrics.json").read_text())
    if metrics["step"] != 1 or metrics["optimizer_skipped"] or not math.isfinite(metrics["loss"]):
        raise RuntimeError(f"Optimizer did not complete a valid update: {metrics}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "train"))
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.stage == "prepare":
        prepare(args.root, args.output)
    else:
        train(args.output)

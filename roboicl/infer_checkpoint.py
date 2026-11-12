"""Generate one Zero-WAM action chunk from a trained RoboICL checkpoint."""

import argparse
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from examples.baselines.zero_wam.zero_wam import policy
from examples.baselines.zero_wam.zero_wam.action_space import EpisodeActionSpace
from examples.baselines.zero_wam.zero_wam.config import ZeroWAMConfig

from pilot import ACTION_CHANNELS, recording
from prepare import DUAL_SIZE, camera_count, latent_path, load_plan, read_json, sha256


class RoboICLPolicy(policy.ZeroWAMAgent):
    def clear_cache(self):
        super().clear_cache()
        self.demo_features = None

    def _state(self, obs):
        return F.pad(self.roboicl_space.normalize(obs["state"].to(self.device), "state"), (0, 8))

    def _stream(self, data, timestep, action, cache_type, frame_id, icl=False, frame_shift=None):
        inputs = super()._stream(data, timestep, action, cache_type, frame_id, icl, frame_shift)
        indexed_video = not action and not icl and hasattr(self.transformer, "foresight_indexing")
        if self.demo_features is not None and (action or indexed_video):
            inputs["demo_features"] = self.demo_features
            if indexed_video:
                inputs["demo_index_frames"] = data.shape[2]
        return inputs


def load_checkpoint(path, adapter_config, model, plan):
    if path.suffix != ".safetensors":
        checkpoint = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
        settings = checkpoint["native_config"]
        if Path(settings["model_path"]).resolve() != model.resolve():
            raise ValueError("Base model path differs from the training checkpoint")
        return settings, checkpoint["adapter_state_dict"], checkpoint["global_step"]

    from safetensors.torch import load_file

    config = read_json(adapter_config)
    receipt = read_json(path.with_name("adapter.json"))
    if (path.stat().st_size, sha256(path)) != (receipt["size_bytes"], receipt["sha256"]):
        raise ValueError("Published adapter differs from its upload receipt")
    if not config.get("foresight_indexing") or config["video_encoding"] != plan["encoding"]:
        raise ValueError("Published adapter does not match the indexed RoboICL plan")
    settings = {
        "ig_roboicl_channels": config["action_channels"],
        "ig_roboicl_cache_sha256": config["latent_receipt_sha256"],
        "ig_demo_attention": config["demo_attention"],
        "ig_demo_cache_sha256": config["demo_features_sha256"],
        "ig_foresight_indexing": config["foresight_indexing"],
        "ig_lora_rank": config["lora_rank"],
        "snr_shift": config["video_snr_shift"],
        "action_snr_shift": config["action_snr_shift"],
        "icl_rope_h": config["icl_rope_h"],
    }
    return settings, load_file(path, device="cpu"), receipt["global_step"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("work", "model", "empty-text"):
        parser.add_argument(f"--{key}", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Training .pt or published adapter.safetensors")
    parser.add_argument("--adapter-config", type=Path, help="Published adapter_config.json for Safetensors")
    parser.add_argument("--pair-index", type=int, default=0, help="Index in the held-out validation pairs")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--demo-features", type=Path)
    parser.add_argument("--disable-demo-attention", action="store_true", help="Ablate A while retaining ICL")
    parser.add_argument("--denoise-steps", type=int, default=50)
    parser.add_argument("--output", type=Path, required=True, help="New JSON file for the predicted actions")
    args = parser.parse_args()
    plan = load_plan(args.work)
    views = camera_count(plan)
    validation = plan["pairs"]["validation"]
    if not 0 <= args.pair_index < len(validation):
        parser.error(f"--pair-index must be between 0 and {len(validation) - 1}")
    if args.output.exists():
        parser.error("--output already exists")
    if args.denoise_steps < 1:
        parser.error("--denoise-steps must be positive")
    pair = validation[args.pair_index]
    if (args.checkpoint.suffix == ".safetensors") != (args.adapter_config is not None):
        parser.error("A Safetensors checkpoint requires --adapter-config")
    settings, adapter, global_step = load_checkpoint(args.checkpoint, args.adapter_config, args.model, plan)
    if settings.get("ig_roboicl_channels") != list(ACTION_CHANNELS):
        raise ValueError("Checkpoint does not use the RoboICL 10D action channels")
    stats_path = args.work / "stats.json"
    demo_latent_path = latent_path(args.work, "demo", pair["demo"])
    cache_digest = settings.get("ig_roboicl_cache_sha256")
    if cache_digest is not None:
        receipt_path = args.work / "verified-cache.json"
        if sha256(receipt_path) != cache_digest:
            raise ValueError("RoboICL latent cache receipt differs from the training checkpoint")
        receipt = read_json(receipt_path)
        if (sha256(args.work / "plan.json"), sha256(stats_path)) != (
            receipt["plan_sha256"], receipt["stats_sha256"]
        ):
            raise ValueError("RoboICL pair plan or normalization changed since cache verification")
        stamp = demo_latent_path.stat()
        if [stamp.st_size, stamp.st_mtime_ns] != receipt["latents"].get(
            demo_latent_path.relative_to(args.work).as_posix()
        ):
            raise ValueError("The selected demo latent changed since cache verification")
    spec = settings.get("ig_demo_attention")
    index_spec = settings.get("ig_foresight_indexing")
    if bool(spec) != bool(args.demo_features or args.disable_demo_attention):
        parser.error("For an A checkpoint supply --demo-features, or use --disable-demo-attention")
    demo = None
    if args.demo_features is not None:
        from demo_features import load_features

        if sha256(args.demo_features) != settings["ig_demo_cache_sha256"]:
            raise ValueError("Demo feature cache differs from the training checkpoint")
        rows, features = load_features(args.demo_features, args.work, plan)
        demo = features[rows[pair["demo"]]][None]
    torch.manual_seed(args.seed)
    policy.unpack_actions = lambda tensor, _offset: tensor[..., list(ACTION_CHANNELS)]
    camera_size = dict(zip(("image_size", "robot_width"), DUAL_SIZE)) if views == 2 else {}
    actor = RoboICLPolicy(ZeroWAMConfig(
        model_path=str(args.model), empty_text_embedding=str(args.empty_text),
        cache_dir=str(args.work), action_dim=10,
        adapter_rank=settings.get("ig_lora_rank"),
        video_snr_shift=settings.get("snr_shift", 5.0),
        action_snr_shift=settings.get("action_snr_shift", 1.0),
        icl_rope_h=settings.get("icl_rope_h", 24),
        video_inference_steps=args.denoise_steps, action_inference_steps=args.denoise_steps,
        **camera_size,
    ))
    actor.action_channel_ids = ACTION_CHANNELS
    actor.roboicl_space = EpisodeActionSpace(read_json(stats_path))
    if spec:
        from demo_attention import attach_demo_attention, attach_foresight_indexing

        attach_demo_attention(actor.transformer, spec)
        if index_spec is not None:
            attach_foresight_indexing(actor.transformer, index_spec["schema"])
            actor.transformer.foresight_indexing.to(device=actor.device, dtype=actor.dtype)
        for index in spec["layers"]:
            actor.transformer.blocks[index].demo_attention.to(device=actor.device, dtype=actor.dtype)
    expected = {name for name, param in actor.transformer.named_parameters() if param.requires_grad}
    actor.load_adapter({name: adapter[name] for name in expected})
    actor.preview_human_latent = torch.load(
        demo_latent_path, map_location="cpu", weights_only=True,
    )["latent"].unsqueeze(0)
    rgb, _, state = recording(Path(plan["source_root"]), pair["query"], views=views)
    obs = {"rgb": torch.from_numpy(rgb[:1].copy()).float().div_(255).unsqueeze(0), "state": state[:1]}
    actor.prepare_for_eval(torch.empty(1), obs)
    if demo is not None and not args.disable_demo_attention:
        actor.demo_features = demo.to(device=actor.device, dtype=actor.dtype)
    # Checkpoint reconstruction may initialize modules before restoring weights.
    # Reset the noise seed here so A-on/A-off use the same denoising noise.
    torch.manual_seed(args.seed)
    normalized = actor.get_action(obs)
    action = actor.roboicl_space.denormalize(normalized, "action")
    if action.shape != (1, 16, 10) or not torch.isfinite(action).all():
        raise RuntimeError(f"Invalid predicted action chunk: {action.shape}")
    result = {
        "checkpoint": str(args.checkpoint.resolve()),
        "global_step": global_step,
        "pair": pair,
        "seed": args.seed,
        "control_hz": 50,
        "denoise_steps": args.denoise_steps,
        "demo_attention": actor.demo_features is not None,
        "foresight_indexing": index_spec is not None and actor.demo_features is not None,
        "action_representation": "roboicl_world_hand10_absolute",
        "actions": action[0].cpu().tolist(),
    }
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(f"Wrote {action.shape[1]} action targets to {args.output}")


if __name__ == "__main__":
    main()

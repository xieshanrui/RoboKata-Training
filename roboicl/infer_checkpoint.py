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
from prepare import latent_path, load_plan, read_json, sha256
from query_views import QueryViewsMixin


class RoboICLPolicy(QueryViewsMixin, policy.ZeroWAMAgent):
    def clear_cache(self):
        super().clear_cache()
        self.demo_features = None

    def _state(self, obs):
        return F.pad(self.roboicl_space.normalize(obs["state"].to(self.device), "state"), (0, 8))

    def _stream(self, data, timestep, action, *args, **kwargs):
        inputs = super()._stream(data, timestep, action, *args, **kwargs)
        if action and self.demo_features is not None:
            inputs["demo_features"] = self.demo_features
        return inputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("work", "model", "empty-text"):
        parser.add_argument(f"--{key}", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--pair-index", type=int, default=0, help="Index in the held-out validation pairs")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--demo-features", type=Path)
    parser.add_argument("--disable-demo-attention", action="store_true", help="Ablate A while retaining ICL")
    parser.add_argument("--denoise-steps", type=int, default=50)
    parser.add_argument("--output", type=Path, required=True, help="New JSON file for the predicted actions")
    args = parser.parse_args()
    plan = load_plan(args.work)
    validation = plan["pairs"]["validation"]
    if not 0 <= args.pair_index < len(validation):
        parser.error(f"--pair-index must be between 0 and {len(validation) - 1}")
    if args.output.exists():
        parser.error("--output already exists")
    if args.denoise_steps < 1:
        parser.error("--denoise-steps must be positive")
    pair = validation[args.pair_index]
    checkpoint = torch.load(args.checkpoint, map_location="cpu", mmap=True, weights_only=False)
    settings = checkpoint["native_config"]
    if Path(settings["model_path"]).resolve() != args.model.resolve():
        raise ValueError("Base model path differs from the training checkpoint")
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
    actor = RoboICLPolicy(ZeroWAMConfig(
        model_path=str(args.model), empty_text_embedding=str(args.empty_text),
        cache_dir=str(args.work), action_dim=10,
        adapter_rank=settings.get("ig_lora_rank"),
        video_snr_shift=settings.get("snr_shift", 5.0),
        action_snr_shift=settings.get("action_snr_shift", 1.0),
        icl_rope_h=settings.get("icl_rope_h", 24),
        video_inference_steps=args.denoise_steps, action_inference_steps=args.denoise_steps,
    ))
    actor.action_channel_ids = ACTION_CHANNELS
    actor.query_views = 2 if settings.get("ig_query_views", "head") == "head_wrist" else 1
    actor.roboicl_space = EpisodeActionSpace(read_json(stats_path))
    if spec:
        from demo_attention import attach_demo_attention

        attach_demo_attention(actor.transformer, spec)
        for index in spec["layers"]:
            actor.transformer.blocks[index].demo_attention.to(device=actor.device, dtype=actor.dtype)
    adapter = checkpoint["adapter_state_dict"]
    expected = {name for name, param in actor.transformer.named_parameters() if param.requires_grad}
    actor.load_adapter({name: adapter[name] for name in expected})
    actor.preview_human_latent = torch.load(
        demo_latent_path, map_location="cpu", weights_only=True,
    )["latent"].unsqueeze(0)
    rgb, _, state = recording(Path(plan["source_root"]), pair["query"], range(actor.query_views))
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
        "global_step": checkpoint["global_step"],
        "pair": pair,
        "seed": args.seed,
        "control_hz": 50,
        "denoise_steps": args.denoise_steps,
        "demo_attention": actor.demo_features is not None,
        "action_representation": "roboicl_world_hand10_absolute",
        "actions": action[0].cpu().tolist(),
    }
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(f"Wrote {action.shape[1]} action targets to {args.output}")


if __name__ == "__main__":
    main()

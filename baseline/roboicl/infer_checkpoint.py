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


class RoboICLPolicy(policy.ZeroWAMAgent):
    def _state(self, obs):
        return F.pad(self.roboicl_space.normalize(obs["state"].to(self.device), "state"), (0, 8))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("work", "model", "empty-text"):
        parser.add_argument(f"--{key}", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--pair-index", type=int, default=0, help="Index in the held-out validation pairs")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True, help="New JSON file for the predicted actions")
    args = parser.parse_args()
    plan = json.loads((args.work / "plan.json").read_text())
    validation = plan["pairs"]["validation"]
    if not 0 <= args.pair_index < len(validation):
        parser.error(f"--pair-index must be between 0 and {len(validation) - 1}")
    if args.output.exists():
        parser.error("--output already exists")
    pair = validation[args.pair_index]
    checkpoint = torch.load(args.checkpoint, map_location="cpu", mmap=True, weights_only=False)
    settings = checkpoint["native_config"]
    if Path(settings["model_path"]).resolve() != args.model.resolve():
        raise ValueError("Base model path differs from the training checkpoint")
    if settings.get("ig_roboicl_channels") != list(ACTION_CHANNELS):
        raise ValueError("Checkpoint does not use the RoboICL 10D action channels")
    torch.manual_seed(args.seed)
    policy.unpack_actions = lambda tensor, _offset: tensor[..., list(ACTION_CHANNELS)]
    actor = RoboICLPolicy(ZeroWAMConfig(
        model_path=str(args.model), empty_text_embedding=str(args.empty_text),
        cache_dir=str(args.work), action_dim=10,
    ))
    actor.action_channel_ids = ACTION_CHANNELS
    actor.roboicl_space = EpisodeActionSpace(json.loads((args.work / "stats.json").read_text()))
    adapter = checkpoint["adapter_state_dict"]
    expected = {name for name, param in actor.transformer.named_parameters() if param.requires_grad}
    actor.load_adapter({name: adapter[name] for name in expected})
    actor.preview_human_latent = torch.load(
        args.work / "latents/demo" / f'{pair["demo"]}.pt', map_location="cpu", weights_only=True,
    )["latent"].unsqueeze(0)
    rgb, _, state = recording(Path(plan["source_root"]), pair["query"])
    obs = {"rgb": torch.from_numpy(rgb[:1].copy()).float().div_(255).unsqueeze(0), "state": state[:1]}
    actor.prepare_for_eval(torch.empty(1), obs)
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
        "action_representation": "roboicl_world_hand10_absolute",
        "actions": action[0].cpu().tolist(),
    }
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(f"Wrote {action.shape[1]} action targets to {args.output}")


if __name__ == "__main__":
    main()

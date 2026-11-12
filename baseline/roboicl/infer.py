"""Run a held-out RoboICL policy I/O smoke test against a frozen checkpoint."""

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
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()
    plan = json.loads((args.work / "plan.json").read_text())
    pair = plan["pairs"]["validation"][0]
    torch.manual_seed(1)
    policy.unpack_actions = lambda tensor, _offset: tensor[..., list(ACTION_CHANNELS)]
    actor = RoboICLPolicy(ZeroWAMConfig(
        model_path=str(args.model), empty_text_embedding=str(args.empty_text),
        cache_dir=str(args.work), action_dim=10,
    ))
    actor.action_channel_ids = ACTION_CHANNELS
    actor.roboicl_space = EpisodeActionSpace(json.loads((args.work / "stats.json").read_text()))
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location="cpu", mmap=True, weights_only=False)
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
    print(json.dumps({"status": "passed", "mode": "offline_one_query",
                      "checkpoint": str(args.checkpoint or args.model), "pair": pair,
                      "shape": list(action.shape), "peak_gib": torch.cuda.max_memory_allocated() / 2**30}))


if __name__ == "__main__":
    main()

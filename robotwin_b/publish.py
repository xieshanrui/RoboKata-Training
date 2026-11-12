"""Publish the verified RoboTwin adapter every five epochs."""
import argparse
import json
import time
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from huggingface_hub import CommitOperationAdd, HfApi
from safetensors.torch import save_file

from examples.baselines.zero_wam.zero_wam.training import atomic_json
from publish_epochs import remote_weights
from prepare import sha256

PARAMETERS = 465675523
CARD = """---
base_model: robbyant-research/zero-wam-pretrain
tags:
- robotics
- robotwin
- zero-wam
---

# Zero-WAM RoboTwin Version B

A+B1 adapter initialized from Zero-WAM pretrain. The ICL condition is a
HumanGen demo. The robot query uses high, left wrist and right wrist views,
with future DINOv2 supervision on the high view. The base model is required.
Each epoch directory contains adapter weights and training metadata.
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()
    if args.epochs < 5:
        parser.error("--epochs must be at least five")
    torch.set_num_threads(2)
    args.state.mkdir(parents=True, exist_ok=True)
    api = HfApi()
    if api.whoami()["name"] != args.repo.split("/")[0]:
        raise ValueError("The repository must belong to the authenticated account")
    api.create_repo(args.repo, repo_type="model", private=True, exist_ok=True)
    if not api.model_info(args.repo).private:
        raise ValueError("Expected a private repository for this training run")
    if not api.get_paths_info(args.repo, ["README.md"], repo_type="model"):
        api.create_commit(args.repo, [CommitOperationAdd("README.md", CARD.encode())],
                          commit_message="Document RoboTwin Version B adapter")
    print(f"Watching {args.run} every five epochs through {args.epochs} in {args.repo}", flush=True)

    for epoch in range(5, args.epochs + 1, 5):
        stamp = args.state / f"epoch-{epoch:03d}.json"
        if stamp.is_file():
            continue
        source = args.run / "checkpoints" / f"epoch_{epoch:03d}.pt"
        while not source.is_file():
            time.sleep(20)
        checkpoint = torch.load(source, map_location="cpu", weights_only=False, mmap=True)
        config = json.loads((args.run / "config.json").read_text())
        trainable = {name for group in checkpoint["optimizer"]["param_groups"]
                     for name in group["params"]}
        weights = {name: checkpoint["adapter"][name] for name in sorted(trainable)}
        if (config["epochs"] != args.epochs or checkpoint["contract"] != config or checkpoint["epoch"] != epoch
                or checkpoint["next_batch"] != 0 or checkpoint["step"] != epoch * config["updates_per_epoch"]
                or sum(value.numel() for value in weights.values()) != PARAMETERS):
            raise ValueError(f"Unexpected RoboTwin checkpoint at epoch {epoch}")
        folder = f"epoch-{epoch:03d}"
        destination = f"{folder}/adapter.safetensors"
        with TemporaryDirectory(dir=args.state) as directory:
            path = Path(directory) / "adapter.safetensors"
            save_file({name: value.contiguous() for name, value in weights.items()}, path)
            digest = sha256(path)
            expected = path.stat().st_size, digest
            receipt = dict(epoch=epoch, step=checkpoint["step"], parameters=PARAMETERS,
                           bytes=expected[0], sha256=digest, contract=config)
            for attempt in range(1, 21):
                try:
                    existing = remote_weights(api, args.repo, destination)
                    if existing is not None and existing != expected:
                        raise ValueError(f"Different weights already exist at {destination}")
                    if existing is None:
                        api.create_commit(args.repo, [
                            CommitOperationAdd(destination, path),
                            CommitOperationAdd(f"{folder}/adapter.json", json.dumps(receipt, indent=2).encode()),
                        ], commit_message=f"RoboTwin Version B epoch {epoch}")
                    if remote_weights(api, args.repo, destination) == expected:
                        break
                except ValueError:
                    raise
                except Exception as error:
                    print(f"Epoch {epoch}, upload attempt {attempt}: {type(error).__name__}", flush=True)
                time.sleep(min(30 * attempt, 300))
            else:
                raise RuntimeError(f"Unable to verify upload for epoch {epoch}")
            atomic_json(stamp, receipt)
            print(f"Verified {destination}", flush=True)


if __name__ == "__main__":
    main()

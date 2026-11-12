"""Publish completed RoboICL A or A+B1 epoch adapters to a public model repo."""

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import time

from huggingface_hub import CommitOperationAdd, HfApi
from safetensors import safe_open
from safetensors.torch import save_file
import torch

from prepare import read_json, sha256, write_json


BASE = "Robbyant-Research/zero-wam-pretrain"
BASE_REVISION = "7040c4195df216c900334ef62d5fdcf05c0601aa"
CARD = """---
library_name: pytorch
base_model: Robbyant-Research/zero-wam-pretrain
tags:
- robotics
- roboicl
- zero-wam
- lora
---

# Robot-ICL + A

Version A adds three zero-initialized, gated cross-attention layers that
allow Zero-WAM's action branch to read tokens from a robot demonstration.
A frozen DINOv2 ViT-B/14 encodes eight RGB samples from the external camera.
Each sample contributes a 2×2 grid of pooled features with time and space
indices. Video ICL and the original MCP training objective are retained.

Training warm-starts from the Robot-ICL baseline at step 3,698, which uses
[Zero-WAM pretraining weights](https://huggingface.co/Robbyant-Research/zero-wam-pretrain)
(Apache-2.0, revision `7040c4195df216c900334ef62d5fdcf05c0601aa`).
The RoboICL fine-tuning and demo input use robot data. The inherited Zero-WAM
weights were pretrained with human demonstrations.

## Files

- `epoch-NNN/adapter.safetensors` contains the trainable tensors, including
  LoRA, state/action heads, MCP adapters and A's attention layers.
- `epoch-NNN/adapter.json` records the optimizer step, parameter count and
  file checksum.

To run inference, load the base Zero-WAM model and attach these tensors
with the matching Robot-ICL + A integration.

Training is configured for 20 epochs with 330 optimizer steps per epoch,
four GPUs, and a global batch of 64. Epoch snapshots are published after
the corresponding training checkpoint is saved. Closed-loop performance
has not been measured for these snapshots.
"""

B1_CARD = """---
library_name: pytorch
base_model: Robbyant-Research/zero-wam-pretrain
tags:
- robotics
- roboicl
- zero-wam
- lora
---

# Robot-ICL + A+B1

B1 replaces latent MCP denoising with a training-only semantic prediction
objective on frozen DINOv2 targets at future query latent anchors.
Inference uses the Version A architecture with the matching Zero-WAM base
model and Robot-ICL + A integration.

Each epoch folder contains the adapter tensors in Safetensors and an
upload receipt with their SHA-256. Closed-loop performance has not been
measured for these snapshots.
"""


def json_bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def adapter_contract(settings):
    return {
        "base_model": BASE,
        "base_model_revision": BASE_REVISION,
        "lora_rank": settings["ig_lora_rank"],
        "lora_alpha": 2 * settings["ig_lora_rank"],
        "demo_attention": settings["ig_demo_attention"],
        "demo_dropout": settings["ig_demo_dropout"],
        "action_representation": settings["ig_action_representation"],
        "action_channels": settings["ig_roboicl_channels"],
        "actions_per_latent": settings["ig_action_per_frame"],
        "video_snr_shift": settings["snr_shift"],
        "action_snr_shift": settings["action_snr_shift"],
        "icl_rope_h": settings["icl_rope_h"],
        "enable_mcp": settings["enable_mcp"],
        "effective_batch": settings["ig_effective_batch"],
        "semantic_ifp": settings.get("ig_semantic_ifp"),
    }


def export_epoch(checkpoint_path, weights_path, epoch, updates, config):
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False, mmap=True)
    if ckpt["global_step"] != epoch * updates or ckpt["progress"]["epoch"] != epoch:
        raise ValueError(f"Checkpoint is outside the expected epoch {epoch} boundary")
    if adapter_contract(ckpt["native_config"]) != config:
        raise ValueError(f"Epoch {epoch} uses a different model contract")
    weights = ckpt["adapter_state_dict"]
    shapes = ckpt["adaptation"]["trainable_shapes"]
    if weights.keys() != shapes.keys() or any(list(weights[k].shape) != shapes[k] for k in weights):
        raise ValueError("Adapter tensors do not match the checkpoint parameter manifest")
    save_file({k: v.contiguous() for k, v in weights.items()}, weights_path)
    with safe_open(weights_path, framework="pt", device="cpu") as saved:
        if set(saved.keys()) != set(shapes):
            raise ValueError("Exported Safetensors keys do not match")
    return {
        "epoch": epoch,
        "global_step": ckpt["global_step"],
        "trainable_parameters": sum(v.numel() for v in weights.values()),
        "size_bytes": weights_path.stat().st_size,
        "sha256": sha256(weights_path),
    }


def remote_weights(api, repo, path):
    files = api.get_paths_info(repo, [path], repo_type="model")
    if not files:
        return None
    entry = files[0]
    return (entry.size, entry.lfs.sha256) if entry.lfs else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--repo", required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    settings = read_json(args.run / "config.json")
    config = adapter_contract(settings)
    version = "A+B1" if config["semantic_ifp"] else "A"
    card = B1_CARD if config["semantic_ifp"] else CARD
    updates = read_json(args.run / "sampling_contract.json")["updates_per_epoch"]
    epochs = settings["ig_total_epochs"]
    args.state_dir.mkdir(parents=True, exist_ok=True)
    state_path = args.state_dir / "verified-uploads.json"
    state = read_json(state_path) if state_path.exists() else {}
    api = HfApi()
    if api.whoami()["name"] != args.repo.split("/")[0] or api.model_info(args.repo, token=False).private:
        raise RuntimeError("Expected a public model repository owned by the signed-in user")
    if not api.get_paths_info(args.repo, ["README.md"], repo_type="model"):
        api.create_commit(
            repo_id=args.repo,
            operations=[CommitOperationAdd("README.md", card.encode())],
            commit_message=f"Document Robot-ICL + {version} adapter release",
        )
    print(f"Watching {epochs} epochs in {args.run}; {updates} updates per epoch", flush=True)
    for epoch in range(1, epochs + 1):
        key = f"{epoch:03d}"
        if key in state:
            continue
        source = args.run / "checkpoints" / f"epoch_{key}.pt"
        while not source.is_file():
            time.sleep(15)
        folder = f"epoch-{key}"
        destination = f"{folder}/adapter.safetensors"
        with TemporaryDirectory(dir=args.state_dir) as temporary:
            weights = Path(temporary) / "adapter.safetensors"
            receipt = export_epoch(source, weights, epoch, updates, config)
            expected = (receipt["size_bytes"], receipt["sha256"])
            attempt = 0
            while True:
                try:
                    if remote_weights(api, args.repo, destination) != expected:
                        print(f"Uploading epoch {epoch} at step {receipt['global_step']}", flush=True)
                        api.create_commit(
                            repo_id=args.repo,
                            operations=[
                                CommitOperationAdd(destination, weights),
                                CommitOperationAdd(f"{folder}/adapter.json", json_bytes(receipt)),
                            ],
                            commit_message=f"Publish Robot-ICL + {version} epoch {epoch}",
                        )
                    if remote_weights(api, args.repo, destination) != expected:
                        raise RuntimeError("Hugging Face file size or checksum does not match")
                    break
                except Exception as error:
                    attempt += 1
                    print(f"Epoch {epoch} upload attempt {attempt}: {type(error).__name__}", flush=True)
                    time.sleep(min(30 * attempt, 300))
            state[key] = {**receipt, "path": destination}
            write_json(state_path, state)
            print(f"Verified epoch {epoch}: {destination}", flush=True)
    print("All epoch uploads verified", flush=True)


if __name__ == "__main__":
    main()

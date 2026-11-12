"""Precompute the video-only demo tokens used by Robot-ICL + A."""

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from prepare import load_plan, recording_path, sha256


FRAMES = 8
GRID = 2
SIZE = 168
FEATURE_DIM = 768 + FRAMES + GRID**2
TOKEN_COUNT = FRAMES * GRID**2
SCHEMA = "dinov2_base_uniform8_grid2_168_v1"


def demo_ids(plan):
    return sorted({pair["demo"] for split in plan["pairs"].values() for pair in split})


def sampled_video(path, expected_frames):
    # Sample the midpoint of each temporal segment from external camera 0.
    indices = ((np.arange(FRAMES) + 0.5) * expected_frames / FRAMES).astype(int)
    return read_external_rgb(path, expected_frames, indices)


def read_external_rgb(path, expected_frames, indices):
    with np.load(path, allow_pickle=False) as data:
        rgb = data["rgb"]
    if rgb.shape != (expected_frames, 2, 224, 224, 3) or rgb.dtype != np.uint8:
        raise ValueError(f"Unexpected RoboICL video format: {path}")
    return torch.from_numpy(rgb[indices, 0])


@torch.inference_mode()
def encode_patches(model, frames):
    pixels = frames.permute(0, 3, 1, 2).to(device=model.device, dtype=torch.float32) / 255
    pixels = F.interpolate(pixels, (SIZE, SIZE), mode="bilinear", align_corners=False, antialias=True)
    mean = pixels.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = pixels.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    patches = model(pixel_values=(pixels - mean) / std).last_hidden_state[:, 1:]
    side = SIZE // model.config.patch_size
    patches = patches.transpose(1, 2).reshape(len(frames), 768, side, side)
    return F.adaptive_avg_pool2d(patches, GRID).flatten(2).transpose(1, 2).cpu()


@torch.inference_mode()
def encode(model, frames):
    visual = encode_patches(model, frames).reshape(-1, FRAMES, GRID**2, 768)
    batch = len(visual)
    time = torch.eye(FRAMES).view(1, FRAMES, 1, FRAMES).expand(batch, -1, GRID**2, -1)
    space = torch.eye(GRID**2).view(1, 1, GRID**2, GRID**2).expand(batch, FRAMES, -1, -1)
    return torch.cat([visual, time, space], dim=-1).reshape(batch, TOKEN_COUNT, FEATURE_DIM).half()


def load_encoder(path, device):
    from transformers import Dinov2Model

    model = Dinov2Model.from_pretrained(path, local_files_only=True).to(device)
    if (model.config.hidden_size, model.config.patch_size) != (768, 14):
        raise ValueError("Expected a DINOv2 ViT-B/14 encoder")
    return model.eval().requires_grad_(False)


def load_features(path, work, plan):
    payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    ids = demo_ids(plan)
    if (payload["schema"], payload["plan_sha256"], payload["demo_ids"]) != (
        SCHEMA, sha256(work / "plan.json"), ids
    ):
        raise ValueError("Demo feature cache does not match this RoboICL plan")
    values = payload["features"]
    if values.shape != (len(ids), TOKEN_COUNT, FEATURE_DIM) or values.dtype != torch.float16:
        raise ValueError("Unexpected demo feature tensor")
    return {episode: index for index, episode in enumerate(ids)}, values


def merge_shards(output, ids, contract, count, save):
    features = torch.empty((len(ids), TOKEN_COUNT, FEATURE_DIM), dtype=torch.float16)
    for shard in range(count):
        part = torch.load(shard_path(output, shard, count), map_location="cpu", weights_only=True)
        indices = slice(shard, None, count)
        if any(part.get(key) != value for key, value in contract.items()) or part["demo_ids"] != ids[indices]:
            raise ValueError(f"Feature shard {shard} has a different encoder or plan")
        features[indices] = part["features"]
    save(dict(**contract, demo_ids=ids, features=features), output)


def shard_path(output, index, count):
    return output.with_name(f"{output.stem}.part-{index:03d}-of-{count:03d}{output.suffix}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--encoder", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--gpu-memory-mib", type=int, help="Cap the PyTorch CUDA allocator for shared GPUs")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--episodes-per-batch", type=int, default=4)
    parser.add_argument("--bf16", action="store_true", help="Use BF16 autocast for the frozen encoder")
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--shard", type=int, help="Write this shard. Omit to merge when --shards exceeds one")
    parser.add_argument("--smoke", action="store_true", help="Encode one demo and exit without writing a cache")
    args = parser.parse_args()
    if min(args.episodes_per_batch, args.threads, args.shards) < 1:
        parser.error("Batch size, threads and shard count must be positive")
    if args.shard is not None and not 0 <= args.shard < args.shards:
        parser.error("--shard must be in [0, --shards)")
    device = torch.device(args.device)
    if args.gpu_memory_mib is not None:
        if device.type != "cuda" or args.gpu_memory_mib < 1:
            parser.error("--gpu-memory-mib requires a CUDA device and a positive budget")
        budget = args.gpu_memory_mib * 1024**2
        total = torch.cuda.get_device_properties(device).total_memory
        torch.cuda.set_per_process_memory_fraction(min(budget / total, 1), device)
    output = args.output if args.shard is None else shard_path(args.output, args.shard, args.shards)
    if output.exists():
        parser.error(f"Output already exists: {output}")
    torch.set_num_threads(args.threads)
    from examples.baselines.zero_wam.zero_wam.runtime import atomic_torch_save

    plan = load_plan(args.work)
    ids = demo_ids(plan)
    contract = dict(schema=SCHEMA, plan_sha256=sha256(args.work / "plan.json"),
                    encoder_sha256=sha256(args.encoder / "model.safetensors"),
                    encoder_precision="bf16" if args.bf16 else "fp32")
    if args.shards > 1 and args.shard is None:
        merge_shards(output, ids, contract, args.shards, atomic_torch_save)
        print(f"Merged {len(ids)} demos into {output}")
        return
    model = load_encoder(args.encoder, device)
    root = Path(plan["source_root"])
    selected = ids[args.shard or 0 :: args.shards]
    if args.smoke:
        selected = selected[:1]
    features = torch.empty((len(selected), TOKEN_COUNT, FEATURE_DIM), dtype=torch.float16)
    for start in range(0, len(selected), args.episodes_per_batch):
        batch_ids = selected[start : start + args.episodes_per_batch]
        rgb = torch.cat([sampled_video(recording_path(root, ep), plan["records"][ep]["frames"]) for ep in batch_ids])
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=args.bf16):
            features[start : start + len(batch_ids)] = encode(model, rgb)
        end = start + len(batch_ids)
        if start == 0 or end % 64 == 0 or end == len(selected):
            print(f"Encoded {end}/{len(selected)} demos", flush=True)
    if not torch.isfinite(features).all():
        raise ValueError("DINOv2 produced non-finite features")
    if device.type == "cuda":
        print(f"Peak PyTorch CUDA reservation: {torch.cuda.max_memory_reserved(device) / 1024**2:.0f} MiB")
    if not args.smoke:
        output.parent.mkdir(parents=True, exist_ok=True)
        atomic_torch_save(dict(**contract, demo_ids=selected, features=features), output)


if __name__ == "__main__":
    main()

"""Build frozen DINOv2 targets at RoboICL query video latent anchors."""

import argparse
from pathlib import Path

import torch

from examples.baselines.zero_wam.zero_wam.runtime import atomic_torch_save

from demo_features import GRID, encode_patches, load_encoder, read_external_rgb, shard_path
from prepare import latent_plan, load_plan, recording_path, sha256


SCHEMA = "dinov2_base_query_latent_anchors_grid2_168_v1"
DIM = 768


def query_ids(plan):
    return sorted({pair["query"] for pair in plan["pairs"]["train"]})


def load_query_features(path, work, plan):
    payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    ids = query_ids(plan)
    if (payload["schema"], payload["plan_sha256"], payload["query_ids"]) != (
        SCHEMA, sha256(work / "plan.json"), ids
    ):
        raise ValueError("Semantic query cache does not match this training split")
    features, valid = payload["features"], payload["valid"]
    lengths = torch.tensor([plan["records"][episode]["frames"] for episode in ids])
    plans = [latent_plan(int(length), "query") for length in lengths]
    frame_count = max(len(entry["latent_anchor_indices"]) for entry in plans)
    expected = torch.zeros(len(ids), frame_count, dtype=torch.bool)
    for row, (entry, length) in enumerate(zip(plans, lengths.tolist())):
        nominal = torch.as_tensor(entry["latent_nominal_anchor_indices"])
        expected[row, :len(nominal)] = nominal < length
    if (features.shape != (*expected.shape, GRID**2, DIM) or features.dtype != torch.float16
            or valid.dtype != torch.bool or not torch.equal(valid, expected)):
        raise ValueError("Unexpected semantic feature shape, dtype or anchor mask")
    return {episode: row for row, episode in enumerate(ids)}, features, valid, payload["encoder_sha256"]


def encode_query(model, root, episode, record, batch_frames, bf16):
    frame_count = record["frames"]
    path = recording_path(root, episode)
    if sha256(path) != record["source_sha256"]:
        raise ValueError(f"Query recording differs from the verified plan: {episode}")
    sampling = latent_plan(frame_count, "query")
    anchors = sampling["latent_anchor_indices"]
    valid = torch.from_numpy(sampling["latent_nominal_anchor_indices"] < frame_count)
    video = read_external_rgb(path, frame_count, anchors[valid.numpy()])
    with torch.autocast(model.device.type, dtype=torch.bfloat16, enabled=bf16):
        patches = torch.cat([encode_patches(model, frames) for frames in video.split(batch_frames)]).half()
    if not torch.isfinite(patches).all():
        raise ValueError(f"Non-finite semantic features for {episode}")
    return patches, valid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("work", "encoder", "output"):
        parser.add_argument(f"--{flag}", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--batch-frames", type=int, default=32)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--shard", type=int)
    parser.add_argument("--verify", action="store_true", help="Validate the merged cache without loading DINOv2")
    args = parser.parse_args()
    if min(args.threads, args.batch_frames, args.shards) < 1:
        parser.error("Threads, batch size and shard count must be positive")
    if args.shard is not None and not 0 <= args.shard < args.shards:
        parser.error("--shard must be in [0, --shards)")
    output = args.output if args.shard is None else shard_path(args.output, args.shard, args.shards)
    if output.exists() and not args.verify:
        parser.error(f"Output already exists: {output}")
    torch.set_num_threads(args.threads)
    plan = load_plan(args.work)
    ids = query_ids(plan)
    if not ids:
        raise ValueError("No training query recordings")
    encoder_sha256 = sha256(args.encoder / "model.safetensors")
    if args.verify:
        if args.shard is not None:
            parser.error("--verify expects the merged cache, without --shard")
        _, features, valid, identity = load_query_features(output, args.work, plan)
        if identity != encoder_sha256:
            raise ValueError("Semantic cache uses a different DINOv2 checkpoint")
        for part, mask in zip(features.split(32), valid.split(32)):
            if not torch.isfinite(part).all() or torch.count_nonzero(part[~mask]).item():
                raise ValueError("Semantic cache contains non-finite values or nonzero padding")
        print(f"Verified {len(ids)} query recordings and {int(valid.sum())} real anchors in {output}")
        return
    sizes = {episode: plan["records"][episode]["frames"] for episode in ids}
    max_latents = max(len(latent_plan(size, "query")["latent_anchor_indices"]) for size in sizes.values())
    contract = dict(schema=SCHEMA, plan_sha256=sha256(args.work / "plan.json"),
                    encoder_sha256=encoder_sha256,
                    encoder_precision="bf16" if args.bf16 else "fp32")
    selected = ids[args.shard::args.shards] if args.shard is not None else ids
    features = torch.zeros((len(selected), max_latents, GRID**2, DIM), dtype=torch.float16)
    valid = torch.zeros((len(selected), max_latents), dtype=torch.bool)
    if args.shards > 1 and args.shard is None:
        for index in range(args.shards):
            path = shard_path(args.output, index, args.shards)
            part = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
            rows = slice(index, None, args.shards)
            if (any(part.get(k) != v for k, v in contract.items())
                    or part["query_ids"] != ids[rows]
                    or part["features"].shape != features[rows].shape
                    or part["valid"].shape != valid[rows].shape
                    or part["features"].dtype != features.dtype
                    or part["valid"].dtype != valid.dtype):
                raise ValueError(f"Incompatible semantic cache shard {path}")
            features[rows], valid[rows] = part["features"], part["valid"]
    else:
        model = load_encoder(args.encoder, torch.device(args.device))
        root = Path(plan["source_root"])
        for row, episode in enumerate(selected):
            patches, mask = encode_query(
                model, root, episode, plan["records"][episode], args.batch_frames, args.bf16
            )
            features[row, :len(patches)] = patches
            valid[row, :len(mask)] = mask
            if (row + 1) % 32 == 0:
                print(f"Encoded {row + 1}/{len(selected)} query recordings", flush=True)
    atomic_torch_save(dict(**contract, query_ids=selected, features=features, valid=valid), output)
    print(f"Saved {len(selected)} query recordings to {output}", flush=True)


if __name__ == "__main__":
    main()

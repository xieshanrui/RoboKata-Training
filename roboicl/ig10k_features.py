"""Version A RGB features for the official IG human-episode manifest."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from demo_features import FEATURE_DIM, FRAMES, SCHEMA, TOKEN_COUNT, encode
from prepare import sha256


def load_features(path, manifest_path):
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    ids = [r["record_sha256"] for r in manifest["human_episodes"]]
    if (payload["schema"], payload["manifest_sha256"], payload["human_records"]) != (
        SCHEMA, sha256(Path(manifest_path)), ids
    ):
        raise ValueError("DINO features differ from the official IG human manifest")
    features = payload["features"]
    if features.shape != (len(ids), TOKEN_COUNT, FEATURE_DIM) or features.dtype != torch.float16:
        raise ValueError("Invalid Version A feature shape or dtype")
    if not torch.isfinite(features).all():
        raise ValueError("Non-finite DINO features")
    return features


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--encoder", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a new output path; existing feature caches are immutable")
    from examples.baselines.zero_wam.zero_wam.cache import VideoReader, read_episode, validate_sources
    from examples.baselines.zero_wam.zero_wam.runtime import atomic_torch_save
    from transformers import Dinov2Model

    torch.set_num_threads(4)
    model = Dinov2Model.from_pretrained(args.encoder, local_files_only=True).to(args.device)
    model.eval().requires_grad_(False)
    assert (model.config.hidden_size, model.config.patch_size) == (768, 14)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    records = manifest["human_episodes"]
    features = torch.empty((len(records), TOKEN_COUNT, FEATURE_DIM), dtype=torch.float16)
    for index, record in enumerate(records):
        validate_sources(record)
        data = read_episode(record)
        source = record["source"]
        indices = ((np.arange(FRAMES) + 0.5) * len(data["timestamps"]) / FRAMES).astype(int)
        requested = source["video_from_timestamp"] + data["timestamps"][indices]
        reader = VideoReader(source["video_path"])
        if abs(reader.get_avg_fps() - source["video_fps"]) > 0.01:
            raise ValueError("Human video FPS differs from its manifest")
        absolute = np.rint(requested * reader.get_avg_fps()).astype(np.int64)
        decoded = reader.get_frame_timestamp(absolute.tolist())[:, 0]
        if np.max(np.abs(decoded - requested)) > 0.55 / reader.get_avg_fps():
            raise ValueError("Human DINO frames differ from episode timing")
        frames = torch.from_numpy(reader.get_batch(absolute.tolist()))
        with torch.autocast(torch.device(args.device).type, dtype=torch.bfloat16):
            features[index] = encode(model, frames)[0]
        print(f"Encoded human {index + 1}/{len(records)}", flush=True)
    assert torch.isfinite(features).all()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_torch_save(dict(schema=SCHEMA, manifest_sha256=sha256(args.manifest),
        human_records=[r["record_sha256"] for r in records],
        encoder_sha256=sha256(args.encoder / "model.safetensors"),
        encoder_precision="bf16", features=features), args.output)
    load_features(args.output, args.manifest)


if __name__ == "__main__":
    main()

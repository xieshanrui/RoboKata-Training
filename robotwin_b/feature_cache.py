"""Extract and verify independent RoboTwin A/B1 DINOv2 feature shards."""
import argparse
import hashlib
import json
from pathlib import Path

import torch

from data import CAMERAS, demo_features, query_features, robot_episode, train_rows
from demo_features import FEATURE_DIM, TOKEN_COUNT, load_encoder
from examples.baselines.zero_wam.zero_wam.training import atomic_json, atomic_torch
from prepare import sha256


def entry_path(cache, kind, key):
    digest = hashlib.sha256(key.encode()).hexdigest()
    return cache / kind / (digest + ".pt")


def contract(data, encoder, revision):
    return dict(schema="robotwin_a_b1_dino_fp32_v1", revision=revision,
                train_manifest_sha256=sha256(data / "manifests/train.jsonl"),
                encoder_sha256=sha256(encoder / "model.safetensors"),
                b1_camera=CAMERAS[0])


def read_entry(path, key, kind):
    entry = torch.load(path, map_location="cpu", weights_only=True)
    values = entry["features"]
    shape = (TOKEN_COUNT, FEATURE_DIM) if kind == "demo" else (len(entry["anchors"]), 4, 768)
    if (entry["key"] != key or values.shape != shape or values.dtype != torch.float16
            or not torch.isfinite(values).all()):
        raise ValueError(f"Invalid {kind} cache entry: {path}")
    return entry


def features_for_pair(cache, row, anchors, start=0):
    demo_key, query_key = row["human_video"], row["pair_id"]
    demo = read_entry(entry_path(cache, "demo", demo_key), demo_key, "demo")
    query = read_entry(entry_path(cache, "query", query_key), query_key, "query")
    window = slice(start, start + len(anchors))
    if query["anchors"][window] != anchors.tolist():
        raise ValueError(f"Cached anchors differ for {query_key}")
    return demo["features"], query["features"][window]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "encoder", "cache"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--shards", type=int, default=8)
    parser.add_argument("--shard", type=int)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--init", action="store_true")
    mode.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.shards < 1 or (not (args.init or args.verify)
                           and (args.shard is None or not 0 <= args.shard < args.shards)):
        parser.error("Specify a valid --shard, --init or --verify")
    torch.set_num_threads(2)
    rows, revision = train_rows(args.data)
    identity = contract(args.data, args.encoder, revision)
    cache = args.cache
    manifest = cache / "contract.json"
    if args.init:
        cache.mkdir(parents=True, exist_ok=False)
        (cache / "demo").mkdir()
        (cache / "query").mkdir()
        atomic_json(manifest, identity)
        return
    if json.loads(manifest.read_text()) != identity:
        raise ValueError("Dataset or encoder differs from cache contract")
    demos = sorted({row["human_video"] for row in rows})
    if len(demos) != 854:
        raise ValueError("Unexpected training demo count")

    if args.verify:
        for key in demos:
            read_entry(entry_path(cache, "demo", key), key, "demo")
        for row in rows:
            _, anchors = robot_episode(args.data, row)
            key = row["pair_id"]
            saved = read_entry(entry_path(cache, "query", key), key, "query")
            if saved["anchors"] != anchors.tolist():
                raise ValueError(f"Cache timeline differs for {key}")
        counts = dict(demos=len(demos), queries=len(rows))
        for kind, count in (("demo", counts["demos"]), ("query", counts["queries"])):
            if len(list((cache / kind).glob("*.pt"))) != count:
                raise ValueError(f"Unexpected extra {kind} cache entries")
        result = dict(verified=True, **counts, **identity)
        atomic_json(cache / "verified.json", result)
        print(json.dumps(result), flush=True)
        return

    device = torch.device(args.device)
    if device.type == "cuda":
        # Reserve only a small fraction of an H200 for extraction on shared cards.
        torch.cuda.set_per_process_memory_fraction(0.02, device)
    encoder = load_encoder(args.encoder, device)
    for index, key in enumerate(demos[args.shard::args.shards], 1):
        path = entry_path(cache, "demo", key)
        if not path.is_file():
            values = demo_features(encoder, args.data / key)
            atomic_torch(path, dict(key=key, features=values))
        if index % 25 == 0:
            print(f"shard {args.shard} demos {index}", flush=True)
    for index, row in enumerate(rows[args.shard::args.shards], 1):
        key = row["pair_id"]
        path = entry_path(cache, "query", key)
        if not path.is_file():
            _, anchors = robot_episode(args.data, row)
            values = query_features(encoder, args.data / row["robot_video"], anchors)
            atomic_torch(path, dict(key=key, anchors=anchors.tolist(), features=values))
        if index % 25 == 0:
            print(f"shard {args.shard} queries {index}", flush=True)
    print(f"shard {args.shard} complete", flush=True)


if __name__ == "__main__":
    main()

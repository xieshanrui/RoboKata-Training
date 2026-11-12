"""Frozen IG zed2i DINO targets at actual causal VAE frame anchors."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from demo_features import GRID, encode_patches, load_encoder, shard_path
from prepare import sha256
from semantic_features import DIM, SCHEMA
from examples.baselines.zero_wam.zero_wam.cache import VideoReader, read_episode, sample_plan, validate_sources
from examples.baselines.zero_wam.zero_wam.runtime import atomic_torch_save


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def anchor_masks(manifest):
    contract = read(manifest["contract_path"])
    plans = [sample_plan(r["source"]["episode_length"], r["source"]["fps"], "sim", contract)
             for r in manifest["sim_episodes"]]
    valid = torch.zeros(len(plans), max(len(p["latent_anchor_indices"]) for p in plans), dtype=torch.bool)
    for i, (p, record) in enumerate(zip(plans, manifest["sim_episodes"])):
        n = len(p["latent_anchor_indices"])
        assert n == record["expected_latent_frames"]
        valid[i, :n] = torch.from_numpy(p["latent_nominal_anchor_indices"] < record["source"]["episode_length"])
    return plans, valid


def load_query_features(path, manifest_path):
    manifest = read(manifest_path)
    payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    ids = [r["record_sha256"] for r in manifest["sim_episodes"]]
    if (payload["schema"], payload["manifest_sha256"], payload["query_records"]) != (SCHEMA, sha256(manifest_path), ids):
        raise ValueError("Semantic features differ from the IG training manifest")
    _, expected = anchor_masks(manifest)
    features, valid = payload["features"], payload["valid"]
    if features.shape != (*expected.shape, GRID**2, DIM) or features.dtype != torch.float16 or not torch.equal(valid, expected):
        raise ValueError("IG semantic feature shape, precision or anchor mask differs")
    for part, mask in zip(features.split(32), valid.split(32)):
        if not torch.isfinite(part).all() or torch.count_nonzero(part[~mask]):
            raise ValueError("Non-finite semantic features or nonzero padding")
    return features, valid, payload["encoder_sha256"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["manifest", "encoder", "output"]:
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--shard", type=int)
    parser.add_argument("--batch-frames", type=int, default=32)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.shards < 1 or args.batch_frames < 1 or (args.shard is not None and not 0 <= args.shard < args.shards):
        parser.error("Invalid shard or frame batch")
    torch.set_num_threads(4)
    manifest = read(args.manifest); records = manifest["sim_episodes"]
    plans, expected = anchor_masks(manifest)
    identity = sha256(args.encoder / "model.safetensors")
    if args.verify:
        features, valid, encoder = load_query_features(args.output, args.manifest)
        assert encoder == identity
        print(json.dumps(dict(verified=True,episodes=len(records),valid_anchors=int(valid.sum()),shape=list(features.shape))))
        return
    output = args.output if args.shard is None else shard_path(args.output, args.shard, args.shards)
    if output.exists():
        parser.error("Existing feature caches are immutable; use a new destination")
    indices = list(range(args.shard or 0, len(records), args.shards)) if args.shard is not None else list(range(len(records)))
    features = torch.zeros((len(indices), expected.shape[1], GRID**2, DIM), dtype=torch.float16)
    valid = expected[indices].clone()
    contract = dict(schema=SCHEMA,manifest_sha256=sha256(args.manifest),encoder_sha256=identity,encoder_precision="bf16")
    ids = [records[i]["record_sha256"] for i in indices]
    if args.shards > 1 and args.shard is None:
        for shard in range(args.shards):
            part = torch.load(shard_path(args.output, shard, args.shards),map_location="cpu",weights_only=True,mmap=True)
            rows = slice(shard,None,args.shards)
            assert all(part[k] == v for k,v in contract.items())
            assert part["query_records"] == ids[rows] and torch.equal(part["valid"],valid[rows])
            assert part["features"].shape == features[rows].shape
            features[rows] = part["features"]
    else:
        device = torch.device(args.device); model = load_encoder(args.encoder,device)
        verified = {}
        for row,i in enumerate(indices):
            record=records[i];validate_sources(record,verified);data=read_episode(record)
            source=record["source"];plan=plans[i];mask=valid[row,:len(plan["latent_anchor_indices"])]
            anchors=plan["latent_anchor_indices"][mask.numpy()]
            requested=source["video_from_timestamp"]+data["timestamps"][anchors]
            reader=VideoReader(source["video_path"]);fps=reader.get_avg_fps()
            assert abs(fps-source["video_fps"])<.01
            absolute=np.rint(requested*fps).astype(np.int64)
            decoded=np.asarray(reader.get_frame_timestamp(absolute.tolist()))[:,0]
            assert np.max(np.abs(decoded-requested)) <= .55/fps
            frames=torch.from_numpy(reader.get_batch(absolute.tolist()))
            with torch.inference_mode(),torch.autocast(device.type,dtype=torch.bfloat16):
                patches=torch.cat([encode_patches(model,part) for part in frames.split(args.batch_frames)]).half()
            assert patches.shape==(int(mask.sum()),GRID**2,DIM) and torch.isfinite(patches).all()
            features[row,torch.nonzero(valid[row],as_tuple=True)[0]]=patches.cpu()
            if row==0 or (row+1)%25==0:print(json.dumps(dict(completed=row+1,total=len(indices),time=__import__('time').time())),flush=True)
    output.parent.mkdir(parents=True,exist_ok=True)
    atomic_torch_save(dict(**contract,query_records=ids,features=features,valid=valid),output)
    if args.shard is None:load_query_features(output,args.manifest)
    print(json.dumps(dict(saved=str(output),episodes=len(indices),peak_gib=torch.cuda.max_memory_allocated()/2**30 if torch.cuda.is_available() else 0)))


if __name__ == "__main__":
    main()

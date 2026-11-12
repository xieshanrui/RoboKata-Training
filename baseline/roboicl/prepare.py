"""Split the verified RoboICL release and prepare the Zero-WAM training inputs."""

import argparse
import hashlib
import json
import random
import re
from pathlib import Path


ROLES = ("demo", "query")
SCHEMA = "roboicl_zerowam_v1"
ENCODING = "wan_causal_external_rgb_bf16_mu_v1"
CONTROL_HZ = 50
ACTION_INTERVAL = 16
ACTION_CHANNELS = (*range(9), 28)
ACTION_REPRESENTATION = "roboicl_world_hand10_absolute"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    return json.loads(path.read_text())


def write_json(path, value):
    import tempfile

    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
        temp = Path(stream.name)
        try:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.close()
            temp.replace(path)
        finally:
            temp.unlink(missing_ok=True)


def official_indices(root):
    release = read_json(root / "release_status.json")
    bundle_path = root / "train3000/bundle.json"
    bundle_hash = sha256(bundle_path)
    if (release["repo"], release["records"], release["pairs"]) != ("xieshanrui/RoboICL", 3000, 30000):
        raise ValueError("Expected the official RoboICL train3000 release")
    if not release["complete"] or release["bundle_sha256"] != bundle_hash:
        raise ValueError("RoboICL bundle hash differs from the official release")
    receipt = read_json(root / "verified-train3000.json")
    expected = {
        "status": "verified", "records": release["records"], "pairs": release["pairs"],
        "files": release["bundle_files"], "bytes": release["bytes"], "bundle_sha256": bundle_hash,
    }
    if any(receipt.get(key) != value for key, value in expected.items()):
        raise ValueError("Verified full-release receipt does not match the official bundle")
    files = read_json(bundle_path)["files"]
    for name in ("manifest.json", "pairing.json"):
        path = root / "train3000" / name
        if sha256(path) != files[name]["sha256"]:
            raise ValueError(f"Official index hash mismatch: {name}")
    manifest = read_json(root / "train3000/manifest.json")
    pairing = read_json(root / "train3000/pairing.json")
    if manifest["action_hz"] != CONTROL_HZ:
        raise ValueError("Expected 50 Hz policy recordings")
    return manifest["rows"], pairing["pairs"], files, bundle_hash


def make_split(rows, pairs, fraction, seed):
    records = {row["id"]: row for row in rows}
    if len(records) != len(rows) or any(not re.fullmatch(r"[A-Za-z0-9_-]+", key) for key in records):
        raise ValueError("Recording IDs must be unique and safe for use as filenames")
    families = sorted({row["state"] for row in rows})
    holdout = round(len(families) * fraction)
    if not 0 < holdout < len(families):
        raise ValueError("Validation fraction must hold out at least one family and leave some for training")
    validation = set(random.Random(seed).sample(families, holdout))
    splits = {"train": [], "validation": []}
    seen = set()
    self_pairs = cross_split = 0
    for pair in pairs:
        demo, query = pair["demo"], pair["query"]
        key = (demo, query)
        if key in seen:
            raise ValueError(f"Duplicate directed pairing: {key}")
        seen.add(key)
        demo_row, query_row = records[demo], records[query]
        if demo_row["task_id"] != query_row["task_id"] or demo_row["task_id"] != pair["task_id"]:
            raise ValueError(f"Task mismatch for directed pairing: {key}")
        if pair["self_pair"] != (demo == query):
            raise ValueError(f"Inconsistent self-pair marker: {key}")
        if demo == query:
            self_pairs += 1
            continue
        demo_val = demo_row["state"] in validation
        query_val = query_row["state"] in validation
        if demo_val != query_val:
            cross_split += 1
            continue
        splits["validation" if demo_val else "train"].append(
            {"demo": demo, "query": query, "task_id": pair["task_id"]}
        )
    if not all(splits.values()):
        raise ValueError("Both splits require at least one within-split non-self official pair")
    return records, splits, sorted(validation), {"self_pairs_excluded": self_pairs, "cross_split_excluded": cross_split}


def recording_path(root, episode):
    return root / "train3000" / "records" / episode / "policy_recording.npz"


def latent_path(work, role, episode):
    return work / "latents" / role / f"{episode}.pt"


def required_latents(plan):
    return sorted({(role, pair[role]) for split in plan["pairs"].values() for pair in split for role in ROLES})


def plan_data(root, work, fraction, seed):
    rows, pairs, files, bundle_hash = official_indices(root)
    records, splits, validation, excluded = make_split(rows, pairs, fraction, seed)
    plan = {
        "schema": SCHEMA, "encoding": ENCODING, "bundle_sha256": bundle_hash,
        "source_root": str(root), "split_seed": seed, "validation_fraction": fraction,
        "validation_families": validation, "pairs": splits, **excluded,
        "records": {
            episode: {
                "family": row["state"], "task_id": row["task_id"], "frames": row["frames"],
                "source_sha256": files[f"records/{episode}/policy_recording.npz"]["sha256"],
            }
            for episode, row in records.items()
        },
    }
    path = work / "plan.json"
    if path.exists() and read_json(path) != plan:
        raise ValueError("This preparation directory already contains a different split; use another directory")
    write_json(path, plan)
    print(json.dumps({"plan": str(path), "train_pairs": len(splits["train"]),
                      "validation_pairs": len(splits["validation"]), "cache_entries": len(required_latents(plan)),
                      "validation_families": len(validation), **excluded}, indent=2))


def load_plan(work):
    plan = read_json(work / "plan.json")
    if plan["schema"] != SCHEMA or plan["encoding"] != ENCODING:
        raise ValueError("Unsupported RoboICL preparation plan")
    return plan


def fit_stats(work, plan):
    import numpy as np

    root = Path(plan["source_root"])
    query_ids = sorted({pair["query"] for pair in plan["pairs"]["train"]})
    collected = {"action": [], "state": []}
    for episode in query_ids:
        path = recording_path(root, episode)
        row = plan["records"][episode]
        if sha256(path) != row["source_sha256"]:
            raise ValueError(f"Recording hash mismatch: {episode}")
        with np.load(path, allow_pickle=False) as recording:
            for name in collected:
                values = recording[name]
                if values.shape != (row["frames"], 10) or not np.isfinite(values).all():
                    raise ValueError(f"Unexpected RoboICL {name}: {episode}")
                collected[name].append(values.astype(np.float32, copy=False))
    stats = {"representation": ACTION_REPRESENTATION, "clip": 2.0,
             "plan_sha256": sha256(work / "plan.json"), "query_recordings": len(query_ids)}
    for name, arrays in collected.items():
        values = np.concatenate(arrays)
        q01, q99 = np.quantile(values, [0.01, 0.99], axis=0)
        q01[9], q99[9] = -1, 1
        stats[name] = {"q01": q01.tolist(), "q99": q99.tolist()}
    write_json(work / "stats.json", stats)
    print(f"Fitted statistics on {len(query_ids)} unique training query recordings", flush=True)


def encoder_identity(model):
    vae = model / "vae"
    paths = [vae / "config.json", *sorted(vae.glob("*.safetensors"))]
    if len(paths) < 2:
        raise FileNotFoundError(f"No VAE weight shards under {vae}")
    return {path.name: sha256(path) for path in paths}


def cache_contract(plan, encoder):
    return {"encoding": ENCODING, "bundle_sha256": plan["bundle_sha256"], "encoder": encoder,
            "camera": 0, "demo_resize_hw": [320, 480], "query_resize_hw": [224, 288],
            "demo_fps": 12, "query_rgb_stride": 4, "query_control_hz": CONTROL_HZ,
            "action_interval": ACTION_INTERVAL}


def latent_plan(frames, role):
    from examples.baselines.zero_wam.zero_wam.cache import sample_plan

    contract = {"video": {"sim": {"source_frame_stride": 4, "action_tail_policy": "all_actions_v1"},
                          "human": {"target_fps": 12.0}, "vae_temporal_stride": 4}}
    return sample_plan(frames, CONTROL_HZ, "human" if role == "demo" else "sim", contract)


def validate_payload(path, record, contract, role, episode):
    import torch

    payload = torch.load(path, weights_only=True, map_location="cpu", mmap=True)
    if (payload["recording"], payload["role"], payload["source_sha256"], payload["contract"]) != (
        episode, role, record["source_sha256"], contract
    ):
        raise ValueError(f"Cache identity mismatch: {path}")
    length = len(latent_plan(record["frames"], role)["latent_anchor_indices"])
    spatial = (20, 30) if role == "demo" else (14, 18)
    if payload["latent"].shape != (48, length, *spatial) or payload["latent"].dtype != torch.bfloat16:
        raise ValueError(f"Cache latent shape or dtype mismatch: {path}")
    if role == "query":
        if any(payload[name].shape != (record["frames"], 10) for name in ("actions", "states")):
            raise ValueError(f"Cache action/state shape mismatch: {path}")
        if payload["action_chunk_size"] != ACTION_INTERVAL or not torch.equal(
            payload["latent_nominal_anchor_indices"], torch.arange(length) * ACTION_INTERVAL
        ):
            raise ValueError(f"Cache action cadence mismatch: {path}")
    return payload


def build_cache(work, plan, model, shard, shards):
    import torch
    from filelock import FileLock
    from pilot import encode, recording
    from examples.baselines.zero_wam.zero_wam.runtime import atomic_torch_save
    from examples.baselines.zero_wam.zero_wam.wan_va.modules.utils import WanVAEStreamingWrapper, load_vae

    contract = cache_contract(plan, encoder_identity(model))
    root = Path(plan["source_root"])
    wrapper = None
    assignments = required_latents(plan)[shard::shards]
    for index, (role, episode) in enumerate(assignments, 1):
        record = plan["records"][episode]
        destination = latent_path(work, role, episode)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with FileLock(str(destination) + ".lock"):
            if destination.exists():
                validate_payload(destination, record, contract, role, episode)
                continue
            if sha256(recording_path(root, episode)) != record["source_sha256"]:
                raise ValueError(f"Recording hash mismatch: {episode}")
            if wrapper is None:
                vae = load_vae(str(model / "vae"), torch_dtype=torch.bfloat16, torch_device="cuda").eval()
                wrapper = WanVAEStreamingWrapper(vae)
            rgb, actions, states = recording(root, episode)
            if len(rgb) != record["frames"]:
                raise ValueError(f"Recording length mismatch: {episode}")
            sampling = latent_plan(len(rgb), role)
            latent = encode(wrapper, rgb, sampling["sampled_frame_indices"],
                            tuple(contract[f"{role}_resize_hw"]))
            if latent.shape[1] != len(sampling["latent_anchor_indices"]):
                raise ValueError(f"Unexpected VAE latent length: {episode}")
            payload = {"latent": latent, "role": role, "recording": episode,
                       "source_sha256": record["source_sha256"], "contract": contract}
            if role == "query":
                payload.update(actions=actions, states=states, action_chunk_size=ACTION_INTERVAL,
                               latent_nominal_anchor_indices=torch.as_tensor(sampling["latent_nominal_anchor_indices"]))
            atomic_torch_save(payload, destination)
            print(f"{index}/{len(assignments)} {role} {episode} {list(latent.shape)}", flush=True)


def verify(work, plan, model):
    contract = cache_contract(plan, encoder_identity(model))
    plan_digest = sha256(work / "plan.json")
    stats_path = work / "stats.json"
    if read_json(stats_path)["plan_sha256"] != plan_digest:
        raise ValueError("Normalization statistics were fitted for another split")
    stamps = {}
    for role, episode in required_latents(plan):
        path = latent_path(work, role, episode)
        validate_payload(path, plan["records"][episode], contract, role, episode)
        info = path.stat()
        stamps[str(path.relative_to(work))] = [info.st_size, info.st_mtime_ns]
    write_json(work / "verified-cache.json", {
        "plan_sha256": plan_digest, "stats_sha256": sha256(stats_path),
        "contract": contract, "model_path": str(model), "latents": stamps,
    })
    print(f"Verified {len(stamps)} latents and matching train-only statistics")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("plan", "stats", "cache", "verify"):
        sub = commands.add_parser(command)
        sub.add_argument("--work", required=True, type=Path)
        if command == "plan":
            sub.add_argument("--root", required=True, type=Path)
            sub.add_argument("--validation-fraction", type=float, default=0.10)
            sub.add_argument("--seed", type=int, default=20261105)
        if command in {"cache", "verify"}:
            sub.add_argument("--model", required=True, type=Path)
        if command == "cache":
            sub.add_argument("--shard", type=int, default=0)
            sub.add_argument("--shards", type=int, default=1)
    args = parser.parse_args()
    work = args.work.resolve()
    if args.command == "plan":
        plan_data(args.root.resolve(), work, args.validation_fraction, args.seed)
        return
    plan = load_plan(work)
    if args.command == "stats":
        fit_stats(work, plan)
    elif args.command == "cache":
        if not 0 <= args.shard < args.shards:
            parser.error("expected 0 <= --shard < --shards")
        build_cache(work, plan, args.model.resolve(), args.shard, args.shards)
    else:
        verify(work, plan, args.model.resolve())


if __name__ == "__main__":
    main()

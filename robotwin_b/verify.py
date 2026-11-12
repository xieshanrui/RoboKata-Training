"""Verify a RoboTwin training window or a distributed epoch checkpoint."""
import argparse
import json
from pathlib import Path

import torch

from data import robot_episode, train_rows
from feature_cache import entry_path, features_for_pair


def verify_window(root, cache, frames):
    rows, _ = train_rows(root)
    for row in rows:
        key = row["pair_id"]
        saved = torch.load(entry_path(cache, "query", key), map_location="cpu", weights_only=True)
        if len(saved["anchors"]) >= frames + 2:
            break
    else:
        raise ValueError("No trajectory long enough to test a nonzero window start")

    full, full_anchors = robot_episode(root, row)
    start = len(full_anchors) - frames
    window, anchors = robot_episode(root, row, frames, start)
    torch.testing.assert_close(window["latents"], full["latents"][:, start:], rtol=0, atol=0)
    if anchors.tolist() != full_anchors[start:].tolist():
        raise ValueError("The window's source anchors differ from the episode")
    if window["actions"].shape != (30, frames, 16, 1) or window["actions"][:, 0].count_nonzero():
        raise ValueError("The action window does not start with a zero history chunk")
    if not torch.isfinite(window["actions"]).all() or not window["actions_mask"].any():
        raise ValueError("Invalid action values or mask")
    _, targets = features_for_pair(cache, row, anchors, start)
    torch.testing.assert_close(targets, saved["features"][start:], rtol=0, atol=0)
    return dict(pair=key, window_start=start, anchors=anchors.tolist(), cameras=3)


def verify_checkpoint(path, epoch):
    saved = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    config = saved["contract"]
    expected_step = epoch * config["updates_per_epoch"]
    if (saved["epoch"], saved["next_batch"], saved["step"]) != (epoch, 0, expected_step):
        raise ValueError(f"Checkpoint is not at the end of epoch {epoch}")
    optimizer = saved["optimizer"]
    names = {name for group in optimizer["param_groups"] for name in group["params"]}
    parameters = sum(saved["adapter"][name].numel() for name in names)
    if (parameters != 465675523 or config["rank"] != 106
            or names != set(optimizer["state"])
            or saved["scheduler"]["last_epoch"] != expected_step
            or len(saved["rng"]) != config["world"]):
        raise ValueError("Incomplete adapter, optimizer, scheduler or RNG state")
    return dict(epoch=epoch, step=expected_step, parameters=parameters, optimizer_entries=len(names))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    window = modes.add_parser("window")
    window.add_argument("--data", type=Path, required=True)
    window.add_argument("--feature-cache", type=Path, required=True)
    window.add_argument("--frames", type=int, default=16)
    checkpoint = modes.add_parser("checkpoint")
    checkpoint.add_argument("path", type=Path)
    checkpoint.add_argument("--epoch", type=int, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.mode == "window":
        if args.frames < 1:
            parser.error("--frames must be positive")
        result = verify_window(args.data, args.feature_cache, args.frames)
    else:
        if args.epoch < 1:
            parser.error("--epoch must be positive")
        result = verify_checkpoint(args.path, args.epoch)
    print(json.dumps(result))


if __name__ == "__main__":
    main()

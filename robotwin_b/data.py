"""Read the sparse HumanGen RoboTwin release by its train manifest."""
import json
import subprocess
from pathlib import Path

import numpy as np
import torch

from demo_features import FRAMES, encode, encode_patches
from native import load_robotwin_norm_stat, preprocess_robotwin_actions, va_robotwin_cfg

CAMERAS = tuple(va_robotwin_cfg.obs_cam_keys)


def train_rows(root):
    receipt = json.loads((root / "download_receipt.json").read_text())
    rows = [json.loads(line) for line in (root / "manifests/train.jsonl").read_text().splitlines()]
    splits = json.loads((root / "splits/task_splits.json").read_text())
    tasks = {row["task"] for row in rows}
    if (len(rows) != 2149 or len({row["pair_id"] for row in rows}) != len(rows)
            or len(tasks) != 43 or tasks != set(splits["train"]) or tasks & set(splits["test"])
            or any(row["split"] != "train" for row in rows)):
        raise ValueError("Unexpected RoboTwin training split")
    return rows, receipt["revision"]


def read_rgb(path, indices=None):
    # Decode at encoder resolution. Preserve source frame numbering (no FPS conversion).
    filt = "scale=168:168"
    if indices is not None:
        select = "+".join(f"eq(n,{int(index)})" for index in indices)
        filt = f"select='{select}',{filt}"
    rgb = subprocess.check_output([
        "ffmpeg", "-v", "error", "-threads", "2", "-i", str(path), "-filter_threads", "1",
        "-vf", filt, "-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"
    ])
    frames = np.frombuffer(rgb, dtype=np.uint8).reshape(-1, 168, 168, 3).copy()
    if not len(frames) or (indices is not None and len(frames) != len(indices)):
        raise ValueError(f"Missing RGB frames in {path}")
    return torch.from_numpy(frames)


def read_latent(path):
    item = torch.load(path, map_location="cpu", weights_only=True)
    dims = tuple(int(item[f"latent_{name}"]) for name in ("num_frames", "height", "width"))
    value = item["latent"]
    if value.ndim == 2:
        value = value.reshape(*dims, 48).permute(3, 0, 1, 2)
    if tuple(value.shape) != (48, *dims):
        raise ValueError(f"Unexpected latent shape in {path}")
    return value, item


def robot_episode(root, row, limit=None, start=0):
    import pyarrow.parquet as pq

    video = Path(row["robot_video"])
    if video.parent.name != CAMERAS[0]:
        raise ValueError(f"Expected a high-view manifest path: {video}")
    views, timeline = [], None
    for camera in CAMERAS:
        folder = root / video.parts[0] / video.parts[1] / "latents" / video.parts[3] / camera
        path, = folder.glob(f"{video.stem}_*.pth")
        view, meta = read_latent(path)
        ids = np.asarray(meta["frame_ids"], dtype=np.int64)
        if timeline is not None and (not np.array_equal(timeline, ids) or view.shape != views[0].shape):
            raise ValueError(f"Camera alignment mismatch for {row['pair_id']} at {camera}")
        views.append(view)
        timeline = ids
    first, stride = int(timeline[0]), int(timeline[1] - timeline[0])
    frames, height, width = views[0].shape[-3:]
    if (first < 0 or stride * 4 != va_robotwin_cfg.action_per_frame or
            np.any(np.diff(timeline) != stride) or frames != (len(timeline) - 1) // 4 + 1
            or height % 2 or width % 2):
        raise ValueError(f"Unsupported RoboTwin frame grid: {row['pair_id']}")

    columns = pq.read_table(root / row["robot_data"], columns=[
        "frame_index", "episode_index", "action", "observation.state"
    ]).to_pydict()
    if (columns["frame_index"] != list(range(len(columns["frame_index"])))
            or set(columns["episode_index"]) != {int(row["episode_id"].split("_")[-1])}
            or timeline[-1] >= len(columns["frame_index"])):
        raise ValueError(f"Parquet timeline mismatch for {row['pair_id']}")
    if not 0 <= start < frames:
        raise ValueError(f"Invalid window start for {row['pair_id']}: {start}")
    size = frames - start if limit is None else min(frames - start, limit)
    first = int(timeline[start * 4])
    stats = load_robotwin_norm_stat(root / "robotwin_data/meta/action_stats.json")
    actions, mask = preprocess_robotwin_actions(
        np.asarray(columns["action"])[first:], np.asarray(columns["observation.state"])[first:],
        stats["q01"], stats["q99"], va_robotwin_cfg.inverse_used_action_channel_ids,
        stride * 4, size * stride * 4,
    )

    def pack(value):
        return torch.from_numpy(value.reshape(size, stride * 4, 30)).permute(2, 0, 1).unsqueeze(-1)

    return {
        "latents": torch.cat([view[:, start:start + size] for view in views], dim=-1),
        "actions": pack(actions).float(),
        "actions_mask": pack(mask),
    }, timeline[::4][start:start + size]


def demo_features(encoder, path):
    rgb = read_rgb(path)
    indices = ((np.arange(FRAMES) + 0.5) * len(rgb) / FRAMES).astype(int)
    return encode(encoder, rgb[indices]).squeeze(0)


def query_features(encoder, path, anchors):
    rgb = read_rgb(path, anchors)
    return torch.cat([encode_patches(encoder, part) for part in rgb.split(16)]).half()


def build_features(encoder, root, row, anchors, demos):
    demo_path = row["human_video"]
    if demo_path not in demos:
        demos[demo_path] = demo_features(encoder, root / demo_path)
    targets = query_features(encoder, root / row["robot_video"], anchors)
    if not torch.isfinite(targets).all() or not torch.isfinite(demos[demo_path]).all():
        raise ValueError(f"Nonfinite DINO features for {row['pair_id']}")
    return demos[demo_path], targets

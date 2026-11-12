# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
"""Metadata-driven action preprocessing for released LeRobot ICL datasets."""

from collections import OrderedDict
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
import torch
import yaml


MODEL_ACTION_LAYOUT = OrderedDict(
    (
        ("hand.position", 14),
        ("arm.position", 14),
        ("effector.position", 2),
    )
)
MODEL_ACTION_DIM = sum(MODEL_ACTION_LAYOUT.values())


def _as_numpy(value):
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _origin_value(batch, origin_spec):
    if isinstance(origin_spec, str):
        value = _as_numpy(batch[origin_spec]).copy()
        if value.ndim == 1:
            value = value[:, None]
        return value

    values = []
    for entry in origin_spec:
        if not isinstance(entry, dict) or len(entry) != 1:
            raise ValueError(f"Invalid origin_keys entry: {entry!r}")
        key, selection = next(iter(entry.items()))
        value = _as_numpy(batch[key]).reshape(len(batch[key]), -1)
        values.append(value[:, int(selection["start"]):int(selection["end"])])
    return np.concatenate(values, axis=-1)


def _pose_rotation(pose, pose_format):
    if pose_format == "xyzq":
        return Rotation.from_quat(pose[:, 3:7])
    if pose_format == "xyze":
        return Rotation.from_euler("xyz", pose[:, 3:6])
    if pose_format == "xyza":
        return Rotation.from_rotvec(pose[:, 3:6])
    raise ValueError(f"Unsupported released pose format: {pose_format}")


def _relative_pose(state, action, pose_format="xyzq", use_local_frame=False):
    state = _as_numpy(state)
    action = _as_numpy(action)
    relative_position = action[:, :3] - state[:, :3]
    state_rotation = _pose_rotation(state, pose_format)
    if use_local_frame:
        relative_position = state_rotation.inv().apply(relative_position)
    relative_rotation = (
        state_rotation.inv() * _pose_rotation(action, pose_format)
    ).as_quat()
    sign = np.where(relative_rotation[:, -1:] > 0, 1.0, -1.0)
    return np.concatenate(
        (relative_position, relative_rotation * sign), axis=-1
    )


def _relative_action(state, action, action_format, use_local_frame=False):
    if action_format == "joint":
        return action - state
    if action_format in {"xyzq", "xyze", "xyza"}:
        return _relative_pose(
            state,
            action,
            pose_format=action_format,
            use_local_frame=use_local_frame,
        )
    dual_formats = {
        "xyzqxyzq": ("xyzq", 7),
        "xyzexyze": ("xyze", 6),
        "xyzaxyza": ("xyza", 6),
    }
    if action_format in dual_formats:
        pose_format, width = dual_formats[action_format]
        return np.concatenate(
            (
                _relative_pose(
                    state[:, :width],
                    action[:, :width],
                    pose_format=pose_format,
                    use_local_frame=use_local_frame,
                ),
                _relative_pose(
                    state[:, width : 2 * width],
                    action[:, width : 2 * width],
                    pose_format=pose_format,
                    use_local_frame=use_local_frame,
                ),
            ),
            axis=-1,
        )
    raise ValueError(f"Unsupported released action format: {action_format}")


class LeRobotActionProcessor:
    """Reproduce the VA action path described by released dataset metadata."""

    def __init__(self, task_root):
        task_root = Path(task_root)
        transform_path, stats_path = action_metadata_paths(task_root)
        with transform_path.open(encoding="utf-8") as handle:
            transform = yaml.safe_load(handle)
        with stats_path.open(encoding="utf-8") as handle:
            self.stats = json.load(handle)
        if self.stats.get("method") != "abs":
            raise ValueError(
                f"Only method=abs is supported by {transform_path}, got "
                f"{self.stats.get('method')!r}"
            )

        self.state_specs = self._parse_specs(transform.get("states", []))
        self.action_specs = self._parse_specs(transform.get("actions", []))
        self.image_specs = self._parse_specs(transform.get("images", []))
        self.video_keys = [
            key
            for spec in self.image_specs.values()
            for key in self._origin_keys(spec["origin_keys"])
        ]
        self.columns = sorted(
            {
                key
                for spec in (*self.state_specs.values(), *self.action_specs.values())
                for key in self._origin_keys(spec["origin_keys"])
            }
        )
        self.window_size = int(self.stats["window_size"])

    @staticmethod
    def _parse_specs(entries):
        parsed = OrderedDict()
        for entry in entries:
            if not isinstance(entry, dict) or len(entry) != 1:
                raise ValueError(f"Invalid action transform entry: {entry!r}")
            target, spec = next(iter(entry.items()))
            if "origin_keys" not in spec:
                raise ValueError(f"Missing origin_keys for {target}")
            parsed[target] = dict(spec)
        return parsed

    @staticmethod
    def _origin_keys(origin_spec):
        if isinstance(origin_spec, str):
            return [origin_spec]
        return [next(iter(entry)) for entry in origin_spec]

    def _mapped_features(self, batch):
        states = OrderedDict()
        actions = OrderedDict()
        for target, spec in self.state_specs.items():
            value = _origin_value(batch, spec["origin_keys"])
            shift = int(spec.get("shift", 0))
            if shift > 0:
                shifted = value.copy()
                shifted[shift:] = value[:-shift]
                value = shifted
            states[target] = value
        for target, spec in self.action_specs.items():
            actions[target] = _origin_value(batch, spec["origin_keys"])
        return states, actions

    def process(
        self,
        batch,
        source_start_frame,
        latent_frame_ids,
        temporal_down_rate=4,
    ):
        states, actions = self._mapped_features(batch)
        latent_frame_ids = np.asarray(latent_frame_ids, dtype=np.int64)
        if latent_frame_ids.size == 0:
            raise ValueError("Robot latent has no frame_ids")
        act_shift = max(int(latent_frame_ids[0]) - int(source_start_frame), 0)
        frame_stride = (
            max(int(latent_frame_ids[1] - latent_frame_ids[0]), 1)
            if latent_frame_ids.size > 1
            else 1
        )
        latent_frame_num = (
            (int(latent_frame_ids.size) - 1) // int(temporal_down_rate) + 1
        )
        action_h = frame_stride * int(temporal_down_rate)
        required_size = latent_frame_num * action_h

        output = np.zeros((required_size, MODEL_ACTION_DIM), dtype=np.float64)
        output_mask = np.zeros((required_size, MODEL_ACTION_DIM), dtype=bool)
        offset = 0
        for short_name, padded_dim in MODEL_ACTION_LAYOUT.items():
            action_key = f"action.{short_name}"
            state_key = f"observation.state.{short_name}"
            if action_key in actions and state_key in states:
                action = actions[action_key][act_shift:]
                state = states[state_key][act_shift:]
                if action.shape[0] == 0 or state.shape[0] == 0:
                    raise ValueError(f"Empty aligned action feature: {action_key}")
                spec = self.action_specs[action_key]
                if bool(spec.get("absolute_value", True)) and not bool(
                    spec.get("use_absolute", False)
                ):
                    base_state = np.repeat(state[:1], action.shape[0], axis=0)
                    action = _relative_action(
                        base_state,
                        action,
                        spec.get("format", "joint"),
                        use_local_frame=bool(spec.get("use_local_frame", False)),
                    )

                feature_stats = self.stats["norm_stats"][action_key]
                q01 = np.asarray(feature_stats["q01"], dtype=np.float64)[None]
                q99 = np.asarray(feature_stats["q99"], dtype=np.float64)[None]
                action = (action - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0
                action = np.clip(action, -2.0, 2.0)
                feature_dim = int(action.shape[1])
                if feature_dim > padded_dim:
                    raise ValueError(
                        f"{action_key} has {feature_dim} channels, maximum is {padded_dim}"
                    )
                padded_action = np.concatenate(
                    (
                        np.zeros((action_h, feature_dim), dtype=np.float64),
                        action,
                    ),
                    axis=0,
                )
                if padded_action.shape[0] < required_size:
                    raise ValueError(
                        f"{action_key} has {padded_action.shape[0]} aligned rows, "
                        f"but {required_size} are required"
                    )
                output[:, offset:offset + feature_dim] = padded_action[:required_size]
                output_mask[:, offset:offset + feature_dim] = True
            offset += padded_dim
        return output, output_mask, action_h


def action_metadata_paths(task_root):
    """Resolve action metadata, preferring a task override when one exists."""
    task_root = Path(task_root)
    transform_candidates = (
        task_root / "meta" / "action_transform.yaml",
        task_root.parent / "meta" / "action_transform.yaml",
    )
    transform_path = next(
        (path for path in transform_candidates if path.is_file()),
        transform_candidates[0],
    )
    stats_candidates = []
    if transform_path.is_file():
        with transform_path.open(encoding="utf-8") as handle:
            transform = yaml.safe_load(handle) or {}
        stats_reference = transform.get("norm_stats")
        if stats_reference:
            stats_path = Path(stats_reference)
            if not stats_path.is_absolute():
                stats_path = transform_path.parent / stats_path
            stats_candidates.append(stats_path)

    # Keep existing converted datasets readable while preferring an explicit
    # norm_stats reference for new collection-level releases.
    stats_candidates.extend(
        (
            task_root / "meta" / "action_stats.json",
            task_root.parent / "meta" / "action_stats.json",
        )
    )
    stats_path = next(
        (path for path in stats_candidates if path.is_file()),
        stats_candidates[0],
    )
    return transform_path, stats_path


def has_action_transform(task_root):
    transform_path, stats_path = action_metadata_paths(task_root)
    return transform_path.is_file() and stats_path.is_file()


__all__ = [
    "LeRobotActionProcessor",
    "MODEL_ACTION_DIM",
    "MODEL_ACTION_LAYOUT",
    "action_metadata_paths",
    "has_action_transform",
]

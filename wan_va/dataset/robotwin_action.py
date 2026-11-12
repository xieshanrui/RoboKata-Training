# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
import numpy as np
from scipy.spatial.transform import Rotation as R


ROBOTWIN_ACTION_DIM = 16
MODEL_ACTION_DIM = 30


def _to_numpy(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def relative_eef_pose(state, action):
    """Express absolute xyz-quaternion actions relative to matching states."""
    state = _to_numpy(state)
    action = _to_numpy(action)
    if state.shape != action.shape or state.ndim != 2 or state.shape[1] != 7:
        raise ValueError(
            "state and action must both have shape [frames, 7], got "
            f"{state.shape} and {action.shape}"
        )

    relative_position = action[:, :3] - state[:, :3]
    state_rotation = R.from_quat(state[:, 3:7])
    action_rotation = R.from_quat(action[:, 3:7])
    relative_quaternion = (state_rotation.inv() * action_rotation).as_quat()
    sign = np.where(relative_quaternion[:, -1:] > 0, 1.0, -1.0)
    relative_quaternion *= sign
    return np.concatenate([relative_position, relative_quaternion], axis=1)


def relative_robotwin_action(action, state):
    """Convert Robotwin absolute EEF targets relative to the first robot state."""
    action = _to_numpy(action)
    state = _to_numpy(state)
    if action.ndim != 2 or action.shape[1] != ROBOTWIN_ACTION_DIM:
        raise ValueError(f"action must have shape [frames, 16], got {action.shape}")
    expected_shape = (action.shape[0], ROBOTWIN_ACTION_DIM)
    if state.shape != expected_shape:
        raise ValueError(
            f"state must match action shape {expected_shape}, got {state.shape}"
        )
    if action.shape[0] == 0:
        raise ValueError("Robotwin action segment is empty")

    initial_state = np.repeat(state[:1], action.shape[0], axis=0)
    left_action = relative_eef_pose(initial_state[:, :7], action[:, :7])
    right_action = relative_eef_pose(initial_state[:, 8:15], action[:, 8:15])
    return np.concatenate(
        [left_action, action[:, 7:8], right_action, action[:, 15:16]],
        axis=1,
    )


def preprocess_robotwin_actions(
    action,
    state,
    q01,
    q99,
    inverse_used_action_channel_ids,
    history_size,
    required_size,
):
    """Build the normalized 30-D action target for Robotwin training."""
    relative_action = relative_robotwin_action(action, state)
    raw_mask = np.ones_like(relative_action, dtype=bool)
    relative_action = np.pad(relative_action, ((0, 0), (0, 1)))
    raw_mask = np.pad(raw_mask, ((0, 0), (0, 1)))

    channel_ids = np.asarray(inverse_used_action_channel_ids)
    if channel_ids.shape != (MODEL_ACTION_DIM,):
        raise ValueError(
            "inverse_used_action_channel_ids must contain 30 entries, got "
            f"{channel_ids.shape}"
        )
    action_aligned = relative_action[:, channel_ids]
    action_mask = raw_mask[:, channel_ids]

    q01 = np.asarray(q01).reshape(1, MODEL_ACTION_DIM)
    q99 = np.asarray(q99).reshape(1, MODEL_ACTION_DIM)
    normalized = np.zeros_like(action_aligned, dtype=np.float64)
    normalized[:, action_mask[0]] = (
        (action_aligned[:, action_mask[0]] - q01[:, action_mask[0]])
        / (q99[:, action_mask[0]] - q01[:, action_mask[0]] + 1e-6)
        * 2.0
        - 1.0
    )
    normalized = np.clip(normalized, -2.0, 2.0)

    history_size = int(history_size)
    required_size = int(required_size)
    if history_size < 0 or required_size <= 0:
        raise ValueError("history_size must be non-negative and required_size positive")
    history_action = np.zeros((history_size, MODEL_ACTION_DIM), dtype=np.float64)
    history_mask = np.repeat(action_mask[:1], history_size, axis=0)
    normalized = np.concatenate([history_action, normalized], axis=0)
    action_mask = np.concatenate([history_mask, action_mask], axis=0)
    if normalized.shape[0] < required_size:
        raise ValueError(
            f"action segment has {normalized.shape[0]} rows after history padding, "
            f"but {required_size} are required"
        )
    return normalized[:required_size], action_mask[:required_size]

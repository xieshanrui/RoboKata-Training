"""Exact bridge between the released RoboTwin action interface and panda_hand.

RoboTwin API pose arrays use wxyz. Zero-WAM's released relative-action helpers
pass these arrays directly to scipy Rotation.from_quat; this adapter reproduces
that numeric encoding and its inverse, rather than changing a pretrained target.
The fixed API-to-hand transform is measured from the simulator at episode reset.
"""
import numpy as np
from scipy.spatial.transform import Rotation


def decode(action):
    """Decode RoboICL's world-frame xyz/rotation6/gripper action."""
    x = action[3:6] / np.linalg.norm(action[3:6])
    y = action[6:9] - x * (x @ action[6:9])
    y /= np.linalg.norm(y)
    pose = np.eye(4)
    pose[:3, :3] = np.column_stack([x, y, np.cross(x, y)])
    pose[:3, 3] = action[:3]
    return pose, float(np.clip((action[9]+1)/2, 0, 1))


def pose_matrix(api_pose):
    value = np.asarray(api_pose, dtype=float)
    pose = np.eye(4)
    pose[:3, 3] = value[:3]
    pose[:3, :3] = Rotation.from_quat(value[[4, 5, 6, 3]]).as_matrix()
    return pose


def pose_array(matrix):
    quaternion = Rotation.from_matrix(matrix[:3, :3]).as_quat()
    return np.r_[matrix[:3, 3], quaternion[[3, 0, 1, 2]]]


def hand_to_relative(hand_pose, gripper, initial_api, api_to_hand):
    api = pose_array(np.asarray(hand_pose) @ np.linalg.inv(api_to_hand))
    quaternion = (Rotation.from_quat(initial_api[3:7]).inv()
                  * Rotation.from_quat(api[3:7])).as_quat()
    if quaternion[-1] < 0:
        quaternion = -quaternion
    return np.r_[api[:3] - initial_api[:3], quaternion, gripper]


def relative_to_hand(action, initial_api, api_to_hand):
    action = np.asarray(action, dtype=float)
    if action.shape != (8,) or not np.isfinite(action).all() or np.linalg.norm(action[3:7]) < 1e-8:
        raise ValueError('Invalid released left-arm action')
    quaternion = (Rotation.from_quat(initial_api[3:7])
                  * Rotation.from_quat(action[3:7])).as_quat()
    api = np.r_[initial_api[:3] + action[:3], quaternion]
    hand = pose_matrix(api) @ api_to_hand
    gripper = float(np.clip(action[7], 0., 1.))
    return np.r_[hand[:3, 3], hand[:3, 0], hand[:3, 1], 2 * gripper - 1]

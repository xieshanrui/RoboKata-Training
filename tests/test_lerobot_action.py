import json

import numpy as np
import torch
import yaml

from wan_va.dataset.lerobot_action import (
    LeRobotActionProcessor,
    _relative_action,
    action_metadata_paths,
    has_action_transform,
)


def test_metadata_action_processor_maps_relative_and_absolute_features(tmp_path):
    meta = tmp_path / "meta"
    meta.mkdir()
    transform = {
        "states": [
            {
                "observation.state.arm.position": {
                    "origin_keys": "raw.state.arm",
                    "shift": 1,
                }
            },
            {
                "observation.state.effector.position": {
                    "origin_keys": "raw.state.gripper",
                    "shift": 1,
                }
            },
        ],
        "actions": [
            {
                "action.arm.position": {
                    "origin_keys": "raw.action.arm",
                    "absolute_value": True,
                }
            },
            {
                "action.effector.position": {
                    "origin_keys": "raw.action.gripper",
                    "absolute_value": True,
                    "use_absolute": True,
                }
            },
        ],
    }
    (meta / "action_transform.yaml").write_text(
        yaml.safe_dump(transform), encoding="utf-8"
    )
    stats = {
        "method": "abs",
        "window_size": 16,
        "norm_stats": {
            "action.arm.position": {
                "q01": [-1.0] * 14,
                "q99": [1.0] * 14,
            },
            "action.effector.position": {
                "q01": [0.0, 0.0],
                "q99": [1.0, 1.0],
            },
        },
    }
    (meta / "action_stats.json").write_text(json.dumps(stats), encoding="utf-8")

    processor = LeRobotActionProcessor(tmp_path)
    batch = {
        "raw.state.arm": torch.zeros(8, 14),
        "raw.state.gripper": torch.zeros(8, 2),
        "raw.action.arm": torch.full((8, 14), 0.5),
        "raw.action.gripper": torch.full((8, 2), 0.75),
    }
    action, mask, action_h = processor.process(
        batch, source_start_frame=10, latent_frame_ids=[10, 11, 12, 13, 14]
    )

    assert action_h == 4
    assert action.shape == (8, 30)
    assert mask.shape == (8, 30)
    np.testing.assert_array_equal(action[:4], 0.0)
    np.testing.assert_allclose(action[4:, 14:28], 0.5, atol=1e-6)
    np.testing.assert_allclose(action[4:, 28:30], 0.5, atol=2e-6)
    np.testing.assert_array_equal(mask[:, :14], False)
    np.testing.assert_array_equal(mask[:, 14:], True)


def test_action_processor_resolves_collection_level_metadata(tmp_path):
    task_root = tmp_path / "agibot_data" / "task_360"
    task_meta = task_root / "meta"
    collection_meta = task_root.parent / "meta"
    task_meta.mkdir(parents=True)
    collection_meta.mkdir()
    transform = {
        "states": [],
        "actions": [],
        "norm_stats": "action_stats.json",
    }
    stats = {"method": "abs", "window_size": 0, "norm_stats": {}}
    collection_transform = collection_meta / "action_transform.yaml"
    collection_transform.write_text(
        yaml.safe_dump(transform), encoding="utf-8"
    )
    global_stats = collection_meta / "action_stats.json"
    global_stats.write_text(json.dumps(stats), encoding="utf-8")

    transform_path, stats_path = action_metadata_paths(task_root)

    assert transform_path == collection_transform
    assert stats_path.resolve() == global_stats.resolve()
    assert has_action_transform(task_root)
    assert LeRobotActionProcessor(task_root).stats == stats


def test_task_action_metadata_overrides_collection_metadata(tmp_path):
    task_root = tmp_path / "agibot_data" / "task_360"
    task_meta = task_root / "meta"
    collection_meta = task_root.parent / "meta"
    task_meta.mkdir(parents=True)
    collection_meta.mkdir()
    stats = {"method": "abs", "window_size": 0, "norm_stats": {}}
    (task_meta / "action_transform.yaml").write_text(
        yaml.safe_dump({"states": [], "actions": []}), encoding="utf-8"
    )
    (task_meta / "action_stats.json").write_text(json.dumps(stats), encoding="utf-8")
    (collection_meta / "action_transform.yaml").write_text(
        yaml.safe_dump({"invalid": True}), encoding="utf-8"
    )

    transform_path, stats_path = action_metadata_paths(task_root)

    assert transform_path == task_meta / "action_transform.yaml"
    assert stats_path == task_meta / "action_stats.json"


def test_euler_pose_formats_match_va_relative_quaternion_convention():
    state = np.array([[1.0, 2.0, 3.0, 0.0, 0.0, np.pi / 2]])
    action = np.array([[2.0, 4.0, 6.0, 0.0, 0.0, np.pi]])

    relative = _relative_action(state, action, "xyze")
    local_relative = _relative_action(
        state, action, "xyze", use_local_frame=True
    )

    np.testing.assert_allclose(relative[0, :3], [1.0, 2.0, 3.0], atol=1e-7)
    np.testing.assert_allclose(
        relative[0, 3:], [0.0, 0.0, np.sqrt(0.5), np.sqrt(0.5)], atol=1e-7
    )
    np.testing.assert_allclose(local_relative[0, :3], [2.0, -1.0, 3.0], atol=1e-7)


def test_dual_euler_pose_format_and_image_keys(tmp_path):
    meta = tmp_path / "meta"
    meta.mkdir()
    transform = {
        "states": [],
        "actions": [],
        "images": [
            {
                "observation.images.camera_top": {
                    "origin_keys": "images.rgb.head"
                }
            },
            {
                "observation.images.camera_wrist_left": {
                    "origin_keys": "images.rgb.hand"
                }
            },
        ],
    }
    (meta / "action_transform.yaml").write_text(
        yaml.safe_dump(transform), encoding="utf-8"
    )
    (meta / "action_stats.json").write_text(
        json.dumps({"method": "abs", "window_size": 0, "norm_stats": {}}),
        encoding="utf-8",
    )
    processor = LeRobotActionProcessor(tmp_path)
    assert processor.video_keys == ["images.rgb.head", "images.rgb.hand"]

    state = np.zeros((1, 12))
    action = np.zeros((1, 12))
    action[0, [0, 6]] = [1.0, 2.0]
    relative = _relative_action(state, action, "xyzexyze")
    assert relative.shape == (1, 14)
    np.testing.assert_allclose(relative[0, [0, 7]], [1.0, 2.0])


def test_scalar_raw_feature_keeps_time_as_first_dimension(tmp_path):
    meta = tmp_path / "meta"
    meta.mkdir()
    transform = {
        "states": [
            {
                "observation.state.effector.position": {
                    "origin_keys": "raw.gripper"
                }
            }
        ],
        "actions": [
            {
                "action.effector.position": {
                    "origin_keys": "raw.gripper",
                    "absolute_value": True,
                    "use_absolute": True,
                }
            }
        ],
    }
    (meta / "action_transform.yaml").write_text(
        yaml.safe_dump(transform), encoding="utf-8"
    )
    stats = {
        "method": "abs",
        "window_size": 0,
        "norm_stats": {
            "action.effector.position": {"q01": [0.0], "q99": [1.0]}
        },
    }
    (meta / "action_stats.json").write_text(json.dumps(stats), encoding="utf-8")
    processor = LeRobotActionProcessor(tmp_path)
    action, mask, _ = processor.process(
        {"raw.gripper": torch.linspace(0.0, 1.0, 8)},
        source_start_frame=0,
        latent_frame_ids=[0, 1, 2, 3, 4],
    )
    assert action.shape == (8, 30)
    np.testing.assert_array_equal(mask[:, 28], True)
    np.testing.assert_array_equal(mask[:, 29], False)

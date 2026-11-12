import json
from pathlib import Path

import numpy as np

from wan_va.configs.va_robotwin_cfg import va_robotwin_cfg
from wan_va.configs.va_robotwin_train_cfg import va_robotwin_train_cfg
from wan_va.dataset.robotwin_action import (
    preprocess_robotwin_actions,
    relative_robotwin_action,
)


def _pose(x, y, z, quaternion=(0.0, 0.0, 0.0, 1.0)):
    return [x, y, z, *quaternion]


def test_robotwin_stats_are_loaded_from_release_asset():
    stats_path = Path(va_robotwin_cfg.action_stats_path)
    assert stats_path.name == "robotwin_icl.json"
    with stats_path.open(encoding="utf-8") as stats_file:
        stats = json.load(stats_file)

    assert stats["method"] == "abs"
    assert va_robotwin_cfg.norm_stat["q01"][:14] == (
        stats["norm_stats"]["action.hand.position"]["q01"]
    )
    assert va_robotwin_cfg.norm_stat["q99"][28:] == [1.0, 1.0]


def test_robotwin_training_uses_dataset_stats_path():
    expected = Path(va_robotwin_train_cfg.dataset_path) / "meta" / "action_stats.json"

    assert Path(va_robotwin_train_cfg.action_stats_path) == expected
    assert va_robotwin_train_cfg.load_robotwin_stats_from_dataset is True
    assert Path(va_robotwin_cfg.action_stats_path).name == "robotwin_icl.json"


def test_robotwin_action_is_relative_to_initial_state_not_first_action():
    state = np.array(
        [
            _pose(1.0, 2.0, 3.0) + [0.0] + _pose(4.0, 5.0, 6.0) + [1.0],
            _pose(1.1, 2.0, 3.0) + [0.0] + _pose(3.9, 5.0, 6.0) + [1.0],
        ],
        dtype=np.float32,
    )
    rotation = (0.0, 0.0, np.sqrt(0.5), np.sqrt(0.5))
    action = np.array(
        [
            _pose(1.05, 2.0, 3.0) + [1.0] + _pose(3.95, 5.0, 6.0) + [0.0],
            _pose(1.10, 2.1, 3.0, rotation)
            + [1.0]
            + _pose(3.90, 5.1, 6.0, tuple(-v for v in rotation))
            + [0.0],
        ],
        dtype=np.float32,
    )

    relative = relative_robotwin_action(action, state)

    np.testing.assert_allclose(relative[0, :3], [0.05, 0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(relative[1, :3], [0.10, 0.1, 0.0], atol=1e-6)
    np.testing.assert_allclose(relative[1, 11:15], rotation, atol=1e-6)


def test_robotwin_normalizes_before_model_space_history_padding():
    state = np.array(
        [
            _pose(1.0, 2.0, 3.0) + [0.0] + _pose(4.0, 5.0, 6.0) + [1.0],
            _pose(1.0, 2.0, 3.0) + [0.0] + _pose(4.0, 5.0, 6.0) + [1.0],
        ],
        dtype=np.float32,
    )
    action = np.array(
        [
            _pose(1.05, 2.0, 3.0) + [1.0] + _pose(3.95, 5.0, 6.0) + [0.0],
            _pose(1.10, 2.1, 3.0) + [1.0] + _pose(3.90, 5.1, 6.0) + [0.0],
        ],
        dtype=np.float32,
    )
    normalized, mask = preprocess_robotwin_actions(
        action,
        state,
        va_robotwin_cfg.norm_stat["q01"],
        va_robotwin_cfg.norm_stat["q99"],
        va_robotwin_cfg.inverse_used_action_channel_ids,
        history_size=2,
        required_size=4,
    )

    assert normalized.shape == (4, 30)
    assert mask.shape == (4, 30)
    np.testing.assert_array_equal(normalized[:2], 0.0)
    np.testing.assert_array_equal(mask[:, 14:28], False)
    np.testing.assert_array_equal(mask[:, [*range(14), 28, 29]], True)

    q01 = np.asarray(va_robotwin_cfg.norm_stat["q01"])
    q99 = np.asarray(va_robotwin_cfg.norm_stat["q99"])
    restored = (normalized[2:] + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01
    relative = relative_robotwin_action(action, state)
    expected = np.zeros((2, 30))
    expected[:, :7] = relative[:, :7]
    expected[:, 7:14] = relative[:, 8:15]
    expected[:, 28] = relative[:, 7]
    expected[:, 29] = relative[:, 15]
    np.testing.assert_allclose(restored[:, mask[0]], expected[:, mask[0]], atol=1e-6)

import json
from types import SimpleNamespace

import pytest
import torch

from wan_va.dataset.icl_lerobot_latent_dataset import (
    ICLLeRobotLatentDataset,
    _construct_icl_dataset,
    _icl_repo_list,
    _task_latent_root,
    load_icl_manifest,
)


def test_manifest_indexes_robot_episode_without_camera(tmp_path):
    manifest = tmp_path / "ICL_config_robotwin.json"
    sample = {
        "sample_id": 7,
        "status_info": {
            "video_rel_path": (
                "task/videos/chunk-000/observation.images.cam_high/"
                "episode_000042.mp4"
            )
        },
        "human_video_path": "run_robotwin/samples/007/generated_video.mp4",
    }
    manifest.write_text(json.dumps({"samples": [sample]}), encoding="utf-8")

    exact, episodes = load_icl_manifest(manifest)

    assert exact["task/videos/chunk-000/episode_000042.mp4"] == sample
    assert episodes["task/videos/chunk-000/episode_000042.mp4"] == sample


def test_robot_latents_are_resolved_inside_each_lerobot_task(tmp_path):
    task_root = tmp_path / "robotwin_task"

    assert _task_latent_root(task_root) == task_root / "latents"


def test_empty_collection_override_uses_task_latents(tmp_path):
    task_root = tmp_path / "agibot_data" / "task_360"
    config = type("Config", (), {"robot_latent_path": ""})()

    assert _task_latent_root(task_root, config) == task_root / "latents"


def test_robot_latents_can_be_resolved_from_separate_collection(tmp_path):
    task_root = tmp_path / "data" / "task_360"
    latent_root = tmp_path / "latents"
    config = type("Config", (), {"robot_latent_path": str(latent_root)})()

    assert _task_latent_root(task_root, config) == latent_root / "task_360" / "latents"


def test_manifest_lookup_prefers_converted_source_video_path(tmp_path):
    dataset = ICLLeRobotLatentDataset.__new__(ICLLeRobotLatentDataset)
    sample = {"sample_id": 3}
    source_video = (
        "task_360/videos/chunk-000/observation.images.head/"
        "episode_000230_669_851.mp4"
    )
    dataset._icl_exact_index = {
        "task_360/videos/chunk-000/episode_000230_669_851.mp4": sample
    }
    dataset._icl_episode_index = {}

    assert dataset._lookup_icl_sample(
        {"source_video_rel_path": source_video}
    ) == sample


def test_embedded_episode_pair_wins_for_duplicate_robot_intervals():
    dataset = ICLLeRobotLatentDataset.__new__(ICLLeRobotLatentDataset)
    dataset._icl_exact_index = {}
    dataset._icl_episode_index = {}
    embedded = {
        "sample_id": 12,
        "human_video_path": "run_1/samples/012/generated_video.mp4",
    }

    assert dataset._lookup_icl_sample({"icl": embedded}) is embedded


def test_human_wan_latent_is_not_normalized_twice(tmp_path):
    path = tmp_path / "human.pth"
    disk_latent = torch.arange(24, dtype=torch.bfloat16).reshape(6, 4)
    torch.save(
        {
            "latent": disk_latent,
            "latent_num_frames": 1,
            "latent_height": 2,
            "latent_width": 3,
            "text_emb": torch.randn(2, 8),
        },
        path,
    )

    latent, _ = ICLLeRobotLatentDataset._load_human_latent(path)

    restored_tokens = latent.permute(1, 2, 3, 0).reshape(6, 4)
    torch.testing.assert_close(restored_tokens, disk_latent)


def _write_lerobot_task(root, name):
    meta = root / name / "meta"
    meta.mkdir(parents=True)
    (meta / "info.json").write_text("{}", encoding="utf-8")


def test_icl_repo_list_excludes_held_out_task_variants(tmp_path):
    _write_lerobot_task(tmp_path, "adjust_bottle-demo_clean_collect_200-1000")
    _write_lerobot_task(tmp_path, "stack_blocks_three-demo_clean_collect_200-1000")
    _write_lerobot_task(tmp_path, "place_empty_cup")
    config = SimpleNamespace(
        dataset_path=str(tmp_path),
        excluded_task_names=["stack_blocks_three", "place_empty_cup"],
        expected_num_train_tasks=1,
    )

    assert [path.name for path in _icl_repo_list(config)] == [
        "adjust_bottle-demo_clean_collect_200-1000"
    ]


def test_icl_repo_list_fails_closed_on_unexpected_training_task_count(tmp_path):
    _write_lerobot_task(tmp_path, "adjust_bottle-demo_clean_collect_200-1000")
    config = SimpleNamespace(
        dataset_path=str(tmp_path),
        excluded_task_names=["stack_blocks_three"],
        expected_num_train_tasks=43,
    )

    with pytest.raises(ValueError, match="Expected 43 training task datasets"):
        _icl_repo_list(config)


def test_direct_held_out_dataset_construction_is_rejected(tmp_path):
    config = SimpleNamespace(excluded_task_names=["stack_blocks_three"])

    with pytest.raises(ValueError, match="held-out training dataset"):
        _construct_icl_dataset(
            tmp_path / "stack_blocks_three-demo_clean_collect_200-1000",
            config,
        )

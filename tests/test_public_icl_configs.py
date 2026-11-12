from pathlib import Path

from wan_va.configs import VA_CONFIGS


def test_public_icl_training_configs_use_dataset_local_action_metadata():
    expected = {
        "agibot_train": "agibot",
        "robocoin_train": "robocoin",
        "robomind_train": "robomind",
        "interna1_train": "interna1",
        "oxe_train": "oxe",
    }
    for config_name, dataset in expected.items():
        config = VA_CONFIGS[config_name]
        assert Path(config.dataset_path).name == f"{dataset}_data"
        assert Path(config.icl_manifest_path).name == f"ICL_config_{dataset}.json"
        assert Path(config.human_latent_path).name == dataset
        assert config.robot_latent_path == ""
        assert config.load_robotwin_stats_from_dataset is False
        assert config.excluded_task_names == []
        assert config.expected_num_train_tasks is None


def test_robotwin_training_config_holds_out_unseen_tasks():
    config = VA_CONFIGS["robotwin_train"]

    assert config.expected_num_train_tasks == 43
    assert set(config.excluded_task_names) == {
        "place_object_scale",
        "stamp_seal",
        "open_microwave",
        "move_stapler_pad",
        "place_bread_basket",
        "place_empty_cup",
        "stack_blocks_three",
    }

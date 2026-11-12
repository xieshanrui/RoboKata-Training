from copy import deepcopy
from types import SimpleNamespace

import pytest

from wan_va.dataset.dataset_mixture import (
    DistributedDatasetMixtureSampler,
    normalized_dataset_weights,
    parse_dataset_mixture,
)
from wan_va.configs.va_robotwin_train_cfg import va_robotwin_train_cfg
from wan_va.train import _build_dataset_sources


AVAILABLE = {"robotwin", "agibot", "robocoin"}


def test_dataset_mixture_weights_are_relative_probabilities():
    entries = parse_dataset_mixture(
        " agibot:0.5, robotwin:0.1 ", AVAILABLE
    )

    assert entries == [("agibot", 0.5), ("robotwin", 0.1)]
    probabilities = dict(normalized_dataset_weights(entries))
    assert probabilities["agibot"] == pytest.approx(5 / 6)
    assert probabilities["robotwin"] == pytest.approx(1 / 6)


def test_dataset_mixture_default_is_robotwin():
    assert parse_dataset_mixture(None, AVAILABLE) == [("robotwin", 1.0)]


@pytest.mark.parametrize(
    "value",
    [
        "",
        "robotwin",
        "robotwin:0",
        "robotwin:-1",
        "robotwin:nan",
        "unknown:1",
        "robotwin:1,robotwin:2",
    ],
)
def test_dataset_mixture_rejects_invalid_specs(value):
    with pytest.raises(ValueError):
        parse_dataset_mixture(value, AVAILABLE)


def test_sampler_uses_requested_dataset_probabilities():
    sampler = DistributedDatasetMixtureSampler(
        dataset_lengths=[7, 11],
        dataset_weights=[0.5, 0.1],
        seed=7,
        epoch_size=12000,
    )

    indices = list(sampler)
    agibot_count = sum(index < 7 for index in indices)
    assert agibot_count / len(indices) == pytest.approx(5 / 6, abs=0.015)
    assert all(0 <= index < 18 for index in indices)


def test_sampler_is_deterministic_and_distributed_by_global_stride():
    kwargs = dict(
        dataset_lengths=[3, 5],
        dataset_weights=[1, 2],
        seed=19,
        epoch_size=19,
    )
    rank_zero = DistributedDatasetMixtureSampler(
        **kwargs, num_replicas=2, rank=0
    )
    rank_one = DistributedDatasetMixtureSampler(
        **kwargs, num_replicas=2, rank=1
    )
    combined = [
        index
        for pair in zip(list(rank_zero), list(rank_one))
        for index in pair
    ]
    reference_kwargs = {**kwargs, "epoch_size": 20}
    reference = list(
        DistributedDatasetMixtureSampler(
            **reference_kwargs, num_replicas=1, rank=0
        )
    )

    assert combined == reference
    rank_zero.set_epoch(1)
    assert list(rank_zero) != combined[::2]


def test_mixed_dataset_routes_flat_indices(monkeypatch):
    import wan_va.dataset.icl_lerobot_latent_dataset as dataset_module

    class FakeDataset:
        def __init__(self, config):
            self.items = list(config.items)
            self._datasets = [self]
            self.index_cache_hits = 1
            self.index_cache_misses = 0
            self.hf_cache_hits = 1

        def __len__(self):
            return len(self.items)

        def __getitem__(self, index):
            return self.items[index]

    monkeypatch.setattr(
        dataset_module, "MultiICLLeRobotLatentDataset", FakeDataset
    )
    mixed = dataset_module.MixedICLLeRobotLatentDataset(
        [
            {
                "name": "agibot",
                "weight": 0.5,
                "config": SimpleNamespace(items=["a0", "a1"]),
            },
            {
                "name": "robotwin",
                "weight": 0.1,
                "config": SimpleNamespace(items=["r0"]),
            },
        ]
    )

    assert len(mixed) == 3
    assert [mixed[index] for index in range(3)] == ["a0", "a1", "r0"]
    assert mixed[-1] == "r0"


def _training_args(datasets, **overrides):
    values = {
        "datasets": datasets,
        "dataset_path": None,
        "icl_manifest_path": None,
        "human_latent_path": None,
        "robot_latent_path": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_dataset_sources_keep_dataset_specific_loading_configs():
    config = deepcopy(va_robotwin_train_cfg)
    config.droptext_target = 0.25
    sources = _build_dataset_sources(
        config,
        _training_args("agibot:2,robocoin:1"),
        rank=1,
        local_rank=1,
        world_size=8,
    )

    assert [source["name"] for source in sources] == ["agibot", "robocoin"]
    assert sources[0]["config"].obs_cam_keys == [
        "observation.images.head",
        "observation.images.hand_left",
        "observation.images.hand_right",
    ]
    assert sources[1]["config"].obs_cam_keys == [
        "observation.images.cam_high_rgb"
    ]
    assert all(source["config"].cfg_prob == 0.25 for source in sources)
    assert all(source["config"].world_size == 8 for source in sources)


def test_robotwin_exclusions_survive_mixed_pretraining_sources():
    sources = _build_dataset_sources(
        deepcopy(va_robotwin_train_cfg),
        _training_args("agibot:2,robotwin:1"),
        rank=0,
        local_rank=0,
        world_size=8,
    )
    configs = {source["name"]: source["config"] for source in sources}

    assert configs["robotwin"].expected_num_train_tasks == 43
    assert set(configs["robotwin"].excluded_task_names) == {
        "place_object_scale",
        "stamp_seal",
        "open_microwave",
        "move_stapler_pad",
        "place_bread_basket",
        "place_empty_cup",
        "stack_blocks_three",
    }
    assert configs["agibot"].expected_num_train_tasks is None
    assert configs["agibot"].excluded_task_names == []


def test_dataset_path_override_is_rejected_for_a_mixture():
    with pytest.raises(ValueError, match="only supported"):
        _build_dataset_sources(
            deepcopy(va_robotwin_train_cfg),
            _training_args("agibot:1,robotwin:1", dataset_path="/tmp/data"),
            rank=0,
            local_rank=0,
            world_size=1,
        )

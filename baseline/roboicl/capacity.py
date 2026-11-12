"""Measure concurrent per-device batch capacity for the cached RoboICL pilot pair."""

import os
import sys
from pathlib import Path

import torch

import pilot
from examples.baselines.zero_wam.zero_wam import training

batch_size = int(os.environ["ROBOICL_TEST_BATCH"])
limit_gib = int(os.environ["ROBOICL_TEST_LIMIT_GIB"])
original_native_config = training.native_config
original_prepare_episode = training.prepare_episode


class CapacitySampler(training.EpisodeBatchSampler):
    def __iter__(self):
        for group in super().__iter__():
            yield group * batch_size


def capacity_config(config, manifest):
    settings = original_native_config(config, manifest)
    settings.batch_size = batch_size
    settings.ig_effective_batch = batch_size
    # The upstream episode restriction applies to the production sampler.
    # This experiment batches identical-length items without temporal padding.
    settings.ig_sampling_mode = "capacity_probe"
    return settings


def prepare_episode(trainer, batch, original):
    for key in ("latents", "icl_latents", "actions", "actions_mask"):
        if batch[key].shape[0] != batch_size:
            raise RuntimeError(f"Wrong batch dimension for {key}")
    if not batch["actions_mask"].flatten(1).any(dim=1).all():
        raise RuntimeError("At least one batch item has no action supervision")
    result = original_prepare_episode(trainer, batch, original)
    trainer.ig_group_count = batch_size
    return result


training.EpisodeBatchSampler = CapacitySampler
training.native_config = capacity_config
training.prepare_episode = prepare_episode
device = torch.device("cuda", 0)
torch.cuda.set_per_process_memory_fraction(
    limit_gib * 2**30 / torch.cuda.get_device_properties(device).total_memory,
    device,
)
pilot.train(Path(sys.argv[1]))

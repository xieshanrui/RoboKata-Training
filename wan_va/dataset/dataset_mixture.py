# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
"""Dataset-mixture parsing and distributed weighted sampling."""

import math

import torch
from torch.utils.data import Sampler


def parse_dataset_mixture(value, available_names):
    """Parse ``name:weight`` entries while preserving their input order."""
    if value is None:
        value = "robotwin:1.0"
    value = value.strip()
    if not value:
        raise ValueError("DATASETS must contain at least one name:weight entry")

    available_names = set(available_names)
    entries = []
    seen = set()
    for raw_entry in value.split(","):
        entry = raw_entry.strip()
        if entry.count(":") != 1:
            raise ValueError(
                f"Invalid DATASETS entry {entry!r}; expected name:weight"
            )
        name, raw_weight = (part.strip() for part in entry.split(":", 1))
        name = name.lower()
        if name not in available_names:
            choices = ", ".join(sorted(available_names))
            raise ValueError(
                f"Unknown dataset {name!r} in DATASETS; available: {choices}"
            )
        if name in seen:
            raise ValueError(f"Duplicate dataset {name!r} in DATASETS")
        try:
            weight = float(raw_weight)
        except ValueError as exc:
            raise ValueError(
                f"Invalid sampling weight {raw_weight!r} for dataset {name!r}"
            ) from exc
        if not math.isfinite(weight) or weight <= 0:
            raise ValueError(
                f"Sampling weight for dataset {name!r} must be finite and > 0"
            )
        entries.append((name, weight))
        seen.add(name)

    return entries


def normalized_dataset_weights(entries):
    total = sum(weight for _, weight in entries)
    return [(name, weight / total) for name, weight in entries]


class DistributedDatasetMixtureSampler(Sampler):
    """Choose a dataset by weight, then sample uniformly inside that dataset."""

    def __init__(
        self,
        dataset_lengths,
        dataset_weights,
        num_replicas=1,
        rank=0,
        seed=42,
        epoch_size=None,
    ):
        self.dataset_lengths = [int(length) for length in dataset_lengths]
        self.dataset_weights = torch.as_tensor(
            dataset_weights, dtype=torch.double
        )
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.seed = int(seed)
        self.epoch = 0

        if not self.dataset_lengths:
            raise ValueError("At least one dataset is required")
        if len(self.dataset_lengths) != len(self.dataset_weights):
            raise ValueError("dataset_lengths and dataset_weights must match")
        if any(length <= 0 for length in self.dataset_lengths):
            raise ValueError("All mixed datasets must contain at least one sample")
        if (
            not torch.isfinite(self.dataset_weights).all()
            or bool((self.dataset_weights <= 0).any())
        ):
            raise ValueError("All dataset weights must be finite and > 0")
        if self.num_replicas <= 0:
            raise ValueError("num_replicas must be > 0")
        if not 0 <= self.rank < self.num_replicas:
            raise ValueError("rank must be in [0, num_replicas)")

        requested_epoch_size = (
            sum(self.dataset_lengths) if epoch_size is None else int(epoch_size)
        )
        if requested_epoch_size <= 0:
            raise ValueError("epoch_size must be > 0")
        self.num_samples = math.ceil(requested_epoch_size / self.num_replicas)
        self.total_size = self.num_samples * self.num_replicas
        self.dataset_weights /= self.dataset_weights.sum()

        offsets = [0]
        for length in self.dataset_lengths[:-1]:
            offsets.append(offsets[-1] + length)
        self.dataset_offsets = offsets

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        dataset_ids = torch.multinomial(
            self.dataset_weights,
            self.total_size,
            replacement=True,
            generator=generator,
        )
        indices = torch.empty(self.total_size, dtype=torch.long)
        for dataset_id, (offset, length) in enumerate(
            zip(self.dataset_offsets, self.dataset_lengths)
        ):
            positions = torch.nonzero(
                dataset_ids == dataset_id, as_tuple=False
            ).flatten()
            if positions.numel() == 0:
                continue
            local_indices = torch.randint(
                length,
                (positions.numel(),),
                generator=generator,
            )
            indices[positions] = local_indices + offset

        rank_indices = indices[self.rank : self.total_size : self.num_replicas]
        return iter(rank_indices.tolist())

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = int(epoch)


__all__ = [
    "DistributedDatasetMixtureSampler",
    "normalized_dataset_weights",
    "parse_dataset_mixture",
]

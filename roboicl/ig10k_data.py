"""IG-10K data selection and human pairing for the shared RoboKata trainer."""

import random

from examples.baselines.zero_wam.zero_wam.config import ZeroWAMConfig
from examples.baselines.zero_wam.zero_wam.dataset import IGDataset


class DistributedEpisodeBatches:
    """Batch the official distributed episode order, retaining the final batch."""

    def __init__(self, length, world, rank, batch, seed):
        from torch.utils.data import BatchSampler, DistributedSampler
        self.native = DistributedSampler(range(length), num_replicas=world,
            rank=rank, shuffle=True, seed=seed, drop_last=False)
        self.batches = BatchSampler(self.native, batch_size=batch, drop_last=False)
        self.world, self.rank, self.batch = world, rank, batch
        self.epoch, self.start_batch = 0, 0

    def __len__(self):
        return len(self.batches)

    @property
    def total_size(self):
        return self.native.total_size

    def set_epoch(self, epoch):
        self.epoch, self.start_batch = epoch, 0
        self.native.set_epoch(epoch)

    def sample_offset_after(self, batch_index):
        return min((batch_index + 1) * self.batch, len(self.native)) * self.world

    def __iter__(self):
        for step, indices in enumerate(self.batches):
            if step >= self.start_batch:
                yield [(index, 1., self.epoch, step, len(indices) * self.world, 1)
                       for index in indices]


def query_window_start(length, limit, seed, episode, epoch):
    """Visit every possible contiguous window once per shuffled cycle."""
    if not limit or length <= limit:
        return 0
    count = length - limit + 1
    cycle, position = divmod(epoch, count)
    starts = list(range(count))
    random.Random(f"query-window:{seed}:{episode}:{cycle}").shuffle(starts)
    return starts[position]


class WindowedIGDataset(IGDataset):
    """Slice causal full-episode caches without changing the action clock/origin."""

    def __init__(self, *args, query_latent_limit=0, **kwargs):
        super().__init__(*args, **kwargs)
        if query_latent_limit and query_latent_limit < 2:
            raise ValueError("A Query window needs an observation and future actions")
        self.query_latent_limit = query_latent_limit

    def __getitem__(self, key):
        sample = super().__getitem__(key)
        length = sample["ig_video_length"]
        start = query_window_start(length, self.query_latent_limit, self.seed, key[0], key[2])
        end = min(length, start + self.query_latent_limit) if self.query_latent_limit else length
        if end - start != length:
            # These latents were encoded causally from the complete episode;
            # retain their real past context, never re-encode a fake initial frame.
            for name in ("latents", "actions", "actions_mask"):
                sample[name] = sample[name][:, start:end].clone()
            states = sample["ig_states"]
            sample["ig_states"] = states[start:end].clone()
            if start:
                # The first retained video is the observation at start*interval.
                # Its original action group PRECEDES that observation. Exclude
                # that group from supervision and use the measured current state.
                sample["actions"][:, 0] = 0
                sample["actions_mask"][:, 0] = False
                sample["ig_states"][0] = states[start + 1]
            sample["ig_video_length"] = end - start
            sample["ig_mcp_source_latents"] = sample["latents"]
            sample["ig_mcp_source_length"] = end - start
            sample["ig_latent_start"] = start
            sample["ig_action_start"] = start * self.action_per_frame
            sample["ig_control_time"] = start * self.action_per_frame
        sample.update(ig_original_video_length=length, ig_window_start=start, ig_window_end=end)
        return sample


def data_config(args, world):
    """Use IG protocol resources without importing its separate training tree."""
    required = ("human_root", "sim_root", "manifest", "stats", "task_config_root")
    missing = [name for name in required if getattr(args, name) is None]
    if missing:
        raise ValueError("IG-10K requires " + ", ".join("--" + n.replace("_", "-") for n in missing))
    root = args.task_config_root.resolve()
    suffix = "test_config_unseen" if args.protocol == "scratch" else f"train_config_{args.protocol[1:]}"
    return ZeroWAMConfig(
        protocol=args.protocol, task=args.task,
        model_path=str(args.model.resolve()), empty_text_embedding=str(args.empty_text.resolve()),
        human_root=str(args.human_root.resolve()), sim_root=str(args.sim_root.resolve()),
        cache_dir=str(args.work.resolve()), manifest=str(args.manifest.resolve()),
        normalization_path=str(args.stats.resolve()), run_dir=str(args.run_dir.resolve()),
        human_dataset_file=str(root / "config/exp_configs" / f"human_{suffix}.json"),
        sim_dataset_file=str(root / "config/exp_configs" / f"sim_{suffix}.json"),
        task_mapping_file=str(root / "task_mapping.json"),
        human_task_description_file=str(root / "task_desc/human_desc.json"),
        sim_task_description_file=str(root / "task_desc/sim_desc.json"),
        batch_size=1, effective_batch_size=world, total_epochs=args.epochs,
        warmup_steps=(10 if args.protocol == "scratch" else 100) if args.warmup_steps is None else args.warmup_steps,
        learning_rate=args.learning_rate,
        save_epoch_freq=10 if args.save_epoch_freq is None else args.save_epoch_freq,
        seed=args.seed, num_workers=4 if args.num_workers is None else args.num_workers,
        activation_checkpointing=args.activation_checkpointing, adapter_rank=args.adapter_rank,
        resume_from=str(args.resume_from.resolve()) if args.resume_from else None,
        initialize_from=str(args.initialize_from.resolve()) if args.initialize_from else None,
        dry_run=args.dry_run,
    )
